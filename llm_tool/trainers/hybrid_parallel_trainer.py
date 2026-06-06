#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
hybrid_parallel_trainer.py

MAIN OBJECTIVE:
---------------
Implements a hybrid parallel training architecture optimized for Apple Silicon
(Mac Studio M2 Ultra) that maximizes resource utilization by running GPU training
in the main process while CPU workers run in persistent background processes.

ARCHITECTURE:
-------------
    Main Process (GPU):     [Task1] -> [Task2] -> [Task3] -> ...
    CPU Worker 1:           [TaskA] -> [TaskB] -> [TaskC] -> ...
    CPU Worker 2:           [TaskX] -> [TaskY] -> [TaskZ] -> ...
                            <---- ALL RUNNING IN PARALLEL ---->

FEATURES:
---------
1) GPU training in MAIN PROCESS (zero spawn overhead, like normal mode)
2) CPU workers in PERSISTENT BACKGROUND PROCESSES (model loaded once per worker)
3) Memory-aware worker management with automatic scaling
4) Thread-safe progress updates via queues
5) Graceful degradation on memory pressure
6) Extended metrics support (per-class, per-language)

Dependencies:
-------------
- torch
- multiprocessing
- threading
- sklearn
- llm_tool.trainers.multi_label_trainer
- llm_tool.trainers.data_utils

Author:
-------
Antoine Lemor
"""

import os
import gc
import json
import time
import queue
import logging
import threading
import traceback
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

# Import memory monitoring
from llm_tool.utils.system_resources import (
    get_memory_monitor,
    MemoryMonitor,
    MemoryPressure,
    detect_resources,
    get_model_optimized_config,
)

# Import intelligent task scheduler
from llm_tool.trainers.task_scheduler import (
    SmartTaskScheduler,
    SchedulerConfig,
    SchedulerAnalysis,
)

logger = logging.getLogger(__name__)


# ============================================================================
# HARDWARE ANALYZER (moved from parallel_training_orchestrator.py)
# ============================================================================

@dataclass
class DeviceAllocation:
    """Describes a device allocation for a training worker."""
    device_id: str  # "mps", "cuda:0", "cpu:0", "cpu:1", etc.
    device_type: str  # "gpu" or "cpu"
    batch_size: int
    num_workers: int  # DataLoader workers
    memory_limit_gb: float  # Max memory for this worker
    priority: int  # Lower = higher priority (GPU gets 0)
    gradient_accumulation_steps: int = 1
    use_fp16: bool = False
    threads_per_worker: int = 0  # how many torch threads this worker may use


@dataclass
class ResourcePlan:
    """Complete resource allocation plan for parallel training."""
    gpu_allocation: Optional[DeviceAllocation]
    cpu_allocations: List[DeviceAllocation]
    total_parallel_workers: int
    data_loading_workers_per_trainer: int
    recommended_models_per_batch: int
    system_info: Dict[str, Any]
    model_config: Dict[str, Any]


class HardwareAnalyzer:
    """
    Analyzes hardware capabilities and creates optimal resource allocation plans.
    Integrated with memory monitoring for safe parallel training.
    """

    def __init__(self):
        self._resources = None
        self.memory_monitor = get_memory_monitor()

    def analyze(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Perform comprehensive hardware analysis."""
        self._resources = detect_resources(force_refresh=force_refresh)

        return {
            "gpu": {
                "available": self._resources.gpu.available,
                "type": self._resources.gpu.device_type,
                "name": self._resources.gpu.device_names[0] if self._resources.gpu.device_names else "Unknown",
                "memory_gb": self._resources.gpu.total_memory_gb,
                "available_memory_gb": self._resources.gpu.available_memory_gb,
            },
            "cpu": {
                "physical_cores": self._resources.cpu.physical_cores,
                "logical_cores": self._resources.cpu.logical_cores,
                "name": self._resources.cpu.processor_name,
            },
            "memory": {
                "total_gb": self._resources.memory.total_gb,
                "available_gb": self._resources.memory.available_gb,
                "used_percent": self._resources.memory.percent_used,
            },
            "is_apple_silicon": self._resources.gpu.device_type == "mps",
        }

    def create_resource_plan(
        self,
        model_name: str,
        total_tasks: int,
        prefer_gpu: bool = True,
        max_cpu_workers: Optional[int] = None,
    ) -> ResourcePlan:
        """
        Create an optimal resource allocation plan with memory-aware worker limits.
        """
        if self._resources is None:
            self.analyze()

        model_config = get_model_optimized_config(model_name)
        recommended_batch_size = model_config.get('batch_size', 64)
        recommended_workers = model_config.get('num_workers', 4)

        gpu_allocation = None

        # GPU allocation
        if self._resources.gpu.available and prefer_gpu:
            gpu_type = self._resources.gpu.device_type
            gpu_memory = self._resources.gpu.available_memory_gb or self._resources.gpu.total_memory_gb

            # Conservative batch size for MPS
            gpu_batch_size = min(64, recommended_batch_size) if gpu_type == "mps" else recommended_batch_size

            gpu_allocation = DeviceAllocation(
                device_id=gpu_type if gpu_type != "rocm" else "cuda",
                device_type="gpu",
                batch_size=gpu_batch_size,
                num_workers=min(4, recommended_workers) if gpu_type == "mps" else recommended_workers,
                memory_limit_gb=gpu_memory,
                priority=0,
            )

        # Calculate safe CPU workers using memory monitor
        if max_cpu_workers is not None:
            num_cpu_workers = max_cpu_workers
        else:
            # Auto-calculate based on memory
            num_cpu_workers = self.memory_monitor.calculate_safe_cpu_workers(
                has_gpu_worker=gpu_allocation is not None
            )

            # On MPS (unified memory), scale CPU workers based on available memory
            # Real measurements show each worker uses 20-29GB
            if gpu_allocation and gpu_allocation.device_id == "mps":
                pressure = self.memory_monitor.get_memory_pressure()
                total_memory_gb = pressure.total_gb if hasattr(pressure, 'total_gb') else 128.0
                available_gb = pressure.available_gb if hasattr(pressure, 'available_gb') else 80.0

                if pressure.percent_used > 50.0:
                    # Memory already in use - reduce or avoid CPU workers
                    num_cpu_workers = 0
                    logger.info(f"MPS with {pressure.percent_used:.1f}% memory used - using GPU only")
                elif total_memory_gb >= 96:  # Mac Studio M2 Ultra (128GB) or similar
                    # For high-memory systems: allow up to 3 CPU workers
                    # GPU ~20GB + 3 CPU ~60GB + system ~16GB = ~96GB
                    max_workers_for_memory = int((available_gb - 20) / 22)  # 22GB per worker (safety margin)
                    num_cpu_workers = min(num_cpu_workers, max(1, min(3, max_workers_for_memory)))
                    logger.info(f"MPS high-memory ({total_memory_gb:.0f}GB) with {pressure.percent_used:.1f}% used - {num_cpu_workers} CPU workers")
                else:
                    # Standard MPS systems (32GB-64GB): cap at 2 CPU workers
                    num_cpu_workers = min(num_cpu_workers, 2)
                    logger.info(f"MPS with {pressure.percent_used:.1f}% memory used - {num_cpu_workers} CPU workers (capped at 2)")

        # Create CPU allocations
        cpu_allocations = []
        physical_cores = self._resources.cpu.physical_cores
        threads_per_worker = max(2, physical_cores // max(1, num_cpu_workers or 1))

        for i in range(num_cpu_workers):
            cpu_allocations.append(DeviceAllocation(
                device_id=f"cpu:{i}",
                device_type="cpu",
                batch_size=min(16, recommended_batch_size),  # Reduced for lower RAM usage
                num_workers=2,
                memory_limit_gb=2.5,
                priority=i + 1,
                threads_per_worker=threads_per_worker,
            ))

        return ResourcePlan(
            gpu_allocation=gpu_allocation,
            cpu_allocations=cpu_allocations,
            total_parallel_workers=(1 if gpu_allocation else 0) + len(cpu_allocations),
            data_loading_workers_per_trainer=2,
            recommended_models_per_batch=min((1 if gpu_allocation else 0) + len(cpu_allocations), total_tasks),
            system_info=self.analyze(),
            model_config=model_config,
        )

    def can_use_parallel_training(self, total_tasks: int = 1) -> Tuple[bool, str]:
        """
        Check if parallel training is beneficial for this system.

        Parameters
        ----------
        total_tasks : int
            Number of training tasks

        Returns
        -------
        Tuple[bool, str]
            (can_use, reason_message)
        """
        if self._resources is None:
            self.analyze()

        # Need at least 2 tasks to benefit from parallelism
        if total_tasks < 2:
            return False, "Only 1 task - parallel training not needed"

        # Check memory pressure
        pressure = self.memory_monitor.get_memory_pressure()
        if pressure.level == "emergency":
            return False, f"Emergency memory pressure ({pressure.percent_used:.1f}% used) - cannot start parallel training"

        # Check if we have enough resources
        physical_cores = self._resources.cpu.physical_cores
        available_memory = self._resources.memory.available_gb

        # Minimum requirements for parallel training
        if physical_cores < 4:
            return False, f"Only {physical_cores} CPU cores - need at least 4 for parallel training"

        if available_memory < 16:
            return False, f"Only {available_memory:.1f}GB RAM available - need at least 16GB for parallel training"

        # Calculate potential parallel workers
        has_gpu = self._resources.gpu.available
        max_cpu_workers = self.memory_monitor.calculate_safe_cpu_workers(has_gpu_worker=has_gpu)

        total_workers = (1 if has_gpu else 0) + max_cpu_workers

        if total_workers < 1:
            return False, "Insufficient resources for parallel training"

        # Parallel training is beneficial
        gpu_info = f"GPU ({self._resources.gpu.device_type})" if has_gpu else "CPU only"
        return True, f"Parallel training available: {total_workers} workers ({gpu_info} + {max_cpu_workers} CPU)"


# ============================================================================
# MEMORY-GATED WORKER POOL
# ============================================================================

class MemoryGatedWorkerPool:
    """
    Worker pool that checks memory before starting new workers.

    Implements the "Memory Gate" pattern:
    - Before starting each CPU worker, checks memory pressure
    - If emergency (>92%): Wait for a worker to complete
    - If critical (>85%): Pause, don't start new workers
    - If warning (>75%): Start with delay
    - If normal (<75%): Start immediately
    """

    def __init__(
        self,
        memory_monitor: Optional[MemoryMonitor] = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        self.memory_monitor = memory_monitor or get_memory_monitor()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._active_workers = 0
        self._lock = threading.Lock()

    def can_start_worker(self) -> Tuple[bool, str, float]:
        """
        Check if it's safe to start a new worker.

        Returns
        -------
        Tuple[bool, str, float]
            (can_start, reason, recommended_delay_seconds)
        """
        can_start, reason = self.memory_monitor.should_start_new_worker()
        pressure = self.memory_monitor.get_memory_pressure()

        return can_start, reason, pressure.recommended_wait_seconds

    def wait_for_memory(self, timeout: float = 60.0) -> bool:
        """
        Wait for memory to become available.

        Parameters
        ----------
        timeout : float
            Maximum time to wait in seconds

        Returns
        -------
        bool
            True if memory became available, False if timeout
        """
        start_time = time.time()
        check_interval = 2.0

        while time.time() - start_time < timeout:
            can_start, _, delay = self.can_start_worker()
            if can_start:
                if delay > 0:
                    time.sleep(min(delay, timeout - (time.time() - start_time)))
                return True
            time.sleep(check_interval)

        return False

    def register_worker_start(self):
        """Register that a worker has started."""
        with self._lock:
            self._active_workers += 1

    def register_worker_end(self):
        """Register that a worker has ended."""
        with self._lock:
            self._active_workers = max(0, self._active_workers - 1)

    @property
    def active_workers(self) -> int:
        """Get number of active workers."""
        with self._lock:
            return self._active_workers


class GracefulDegradationHandler:
    """
    Handles OOM errors gracefully by reducing resources and retrying.

    When OOM is detected:
    1. Reduce max_workers by 1
    2. Trigger memory cleanup (gc.collect(), torch.mps.empty_cache())
    3. Retry with reduced batch_size (50%)
    """

    def __init__(self, memory_monitor: Optional[MemoryMonitor] = None):
        self.memory_monitor = memory_monitor or get_memory_monitor()
        self._reduced_workers = 0
        self._reduced_batch_factor = 1.0

    def handle_oom(self) -> Tuple[int, float]:
        """
        Handle an OOM error.

        Returns
        -------
        Tuple[int, float]
            (workers_to_reduce, batch_size_factor)
        """
        # Reduce workers
        self._reduced_workers += 1

        # Reduce batch size by 50%
        self._reduced_batch_factor *= 0.5

        # Trigger cleanup
        self.memory_monitor.trigger_cleanup()

        logger.warning(
            f"OOM handled: reduced workers by {self._reduced_workers}, "
            f"batch factor now {self._reduced_batch_factor:.2f}"
        )

        return self._reduced_workers, self._reduced_batch_factor

    def get_adjusted_batch_size(self, original_batch_size: int) -> int:
        """Get batch size adjusted for OOM recovery."""
        return max(4, int(original_batch_size * self._reduced_batch_factor))

    def get_worker_reduction(self) -> int:
        """Get number of workers to reduce from original plan."""
        return self._reduced_workers

    def reset(self):
        """Reset degradation state for new training session."""
        self._reduced_workers = 0
        self._reduced_batch_factor = 1.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingTask:
    """A single training task."""
    task_id: str
    category_name: str
    model_name: str
    data_path: str
    output_path: str
    config: Dict[str, Any]
    priority: int = 0  # Lower = higher priority


@dataclass
class TaskResult:
    """Result from a completed training task."""
    task_id: str
    category_name: str
    status: str  # "success" or "error"
    f1_score: float = 0.0
    accuracy: float = 0.0
    best_epoch: int = 0
    elapsed_seconds: float = 0.0
    error_message: str = ""
    model_path: str = ""
    device_type: str = "cpu"
    remote_host: str = ""  # SSH hostname if trained remotely (empty = local)


@dataclass
class ProgressUpdate:
    """Progress update from a worker with extended metrics."""
    task_id: str
    worker_id: str
    device_type: str  # "gpu" or "cpu"
    category_name: str
    current_epoch: int
    total_epochs: int
    current_batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    phase: str = "training"
    message: str = ""
    # Additional fields for compatibility with TrainingProgress
    model_name: str = ""
    elapsed_seconds: float = 0.0
    estimated_remaining: float = 0.0
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_labels: int = 0
    language: str = ""
    # Extended metrics - Per-class (for detailed dashboard)
    class_names: Optional[List[str]] = None  # Real label names
    per_class_f1: Optional[List[float]] = None
    per_class_precision: Optional[List[float]] = None
    per_class_recall: Optional[List[float]] = None
    per_class_support: Optional[List[int]] = None
    # Extended metrics - Per-language
    per_language_f1: Optional[Dict[str, float]] = None
    per_language_accuracy: Optional[Dict[str, float]] = None
    per_language_samples: Optional[Dict[str, int]] = None
    # Combined score for best model selection (consistent with bert_base.py)
    combined_score: float = 0.0


# Alias for backward compatibility with code expecting TrainingProgress
TrainingProgress = ProgressUpdate


# ============================================================================
# CPU WORKER PROCESS (Persistent)
# ============================================================================

def _cpu_worker_process(
    worker_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    shutdown_event: mp.Event,
    model_name: str,
    worker_config: Dict[str, Any],
    cancelled_tasks: Optional[Dict] = None,  # Shared dict for cancellation signals
    cancelled_queue: Optional[mp.Queue] = None,  # Queue to return cancelled tasks
):
    """
    Persistent CPU worker process.

    Loads the model ONCE, then processes multiple tasks sequentially.
    This eliminates the model loading overhead between tasks.
    """
    import torch
    from llm_tool.trainers.multi_label_trainer import MultiLabelTrainer, TrainingConfig
    from llm_tool.trainers.data_utils import DataSample

    # Silence console logging inside worker to avoid Rich Live duplication
    try:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.CRITICAL + 1)
        logging.getLogger().setLevel(logging.CRITICAL + 1)
    except Exception:
        pass

    # Force CPU and scope torch thread usage to avoid oversubscription
    torch.set_num_threads(max(1, worker_config.get('num_threads', 4)))
    device = torch.device('cpu')

    logger.info(f"[{worker_id}] CPU worker starting, loading model: {model_name}")

    # Send initial status
    progress_queue.put(ProgressUpdate(
        task_id="",
        worker_id=worker_id,
        device_type="cpu",
        category_name="",
        current_epoch=0,
        total_epochs=0,
        phase="initializing",
        message=f"Loading model {model_name}...",
    ))

    # Create trainer with model (loaded once)
    # NOTE: workers are non-daemonic, so DataLoader worker processes are allowed
    dl_workers = max(0, int(worker_config.get('dataloader_workers', 2)))
    prefetch_factor = worker_config.get('dataloader_prefetch', 2) if dl_workers > 0 else None
    persistent_workers = dl_workers > 0

    config = TrainingConfig(
        model_name=model_name,
        batch_size=worker_config.get('batch_size', 16),  # Reduced default for CPU
        force_device='cpu',
        dataloader_num_workers=dl_workers,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_persistent_workers=persistent_workers,
        multi_label=worker_config.get('multi_label', False),
        multi_label_threshold=worker_config.get('multi_label_threshold', 0.5),
        # NORMAL-phase imbalance handling (session-global)
        imbalance_strategy=worker_config.get('imbalance_strategy'),
        focal_gamma=worker_config.get('focal_gamma', 2.0),
        imbalance_weight_source=worker_config.get('imbalance_weight_source', 'auto'),
        imbalance_class_weights=worker_config.get('imbalance_class_weights'),
        imbalance_weighted_sampler=worker_config.get('imbalance_weighted_sampler', True),
        # Propagate tokenizer-aware filter knobs so MultiLabelTrainer
        # applies the Split/Exclude strategy uniformly across CPU/GPU workers.
        exclude_long_texts=bool(worker_config.get('exclude_long_texts', False)),
        split_long_texts=bool(worker_config.get('split_long_texts', False)),
        max_length=int(worker_config.get('max_length', 512) or 512),
    )

    trainer = None
    tasks_completed = 0

    try:
        trainer = MultiLabelTrainer(config, verbose=False)

        logger.info(f"[{worker_id}] Model loaded, ready to process tasks")

        # Process tasks until shutdown
        while not shutdown_event.is_set():
            try:
                # Get next task (with timeout to check shutdown)
                task = task_queue.get(timeout=1.0)

                if task is None:  # Poison pill
                    logger.info(f"[{worker_id}] Received shutdown signal")
                    break

                # Process the task (with cancellation support)
                result, was_cancelled = _process_task_cpu(
                    task=task,
                    trainer=trainer,
                    worker_id=worker_id,
                    progress_queue=progress_queue,
                    worker_config=worker_config,
                    cancelled_tasks=cancelled_tasks,
                )

                if was_cancelled and cancelled_queue is not None:
                    # Task was cancelled mid-training, return it for GPU rescheduling
                    cancelled_queue.put(task)
                    logger.info(f"[{worker_id}] Task {task.category_name} cancelled, returned for GPU")
                else:
                    result_queue.put(result)
                    tasks_completed += 1

                # Periodic cleanup
                gc.collect()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{worker_id}] Error processing task: {e}")
                traceback.print_exc()

    except Exception as e:
        logger.error(f"[{worker_id}] Worker initialization failed: {e}")
        traceback.print_exc()
    finally:
        logger.info(f"[{worker_id}] Worker shutting down after {tasks_completed} tasks")
        if trainer:
            del trainer
        gc.collect()


class TaskCancelledException(Exception):
    """Raised when a CPU task is cancelled mid-training."""
    pass


def _process_task_cpu(
    task: TrainingTask,
    trainer: "MultiLabelTrainer",
    worker_id: str,
    progress_queue: mp.Queue,
    worker_config: Dict[str, Any],
    cancelled_tasks: Optional[Dict] = None,
) -> Tuple[TaskResult, bool]:
    """
    Process a single task on CPU.

    Returns
    -------
    Tuple[TaskResult, bool]
        (result, was_cancelled) - result may be partial if cancelled
    """
    from llm_tool.trainers.data_utils import DataSample
    from sklearn.model_selection import train_test_split

    start_time = time.time()
    was_cancelled = False
    logger.info(f"[{worker_id}] Starting task: {task.category_name}")

    def send_progress(epoch, total, phase, **kwargs):
        progress_queue.put(ProgressUpdate(
            task_id=task.task_id,
            worker_id=worker_id,
            device_type="cpu",
            category_name=task.category_name,
            current_epoch=epoch,
            total_epochs=total,
            phase=phase,
            train_loss=kwargs.get('train_loss', 0),
            val_loss=kwargs.get('val_loss', 0),
            f1_score=kwargs.get('f1', 0),
            accuracy=kwargs.get('acc', 0),
            combined_score=kwargs.get('combined_score', 0),
            current_batch=kwargs.get('current_batch', 0),
            total_batches=kwargs.get('total_batches', 0),
            message=kwargs.get('msg', ''),
            # Extended metrics for dashboard
            class_names=kwargs.get('class_names'),
            per_class_f1=kwargs.get('per_class_f1'),
            per_class_precision=kwargs.get('per_class_precision'),
            per_class_recall=kwargs.get('per_class_recall'),
            per_class_support=kwargs.get('per_class_support'),
            per_language_f1=kwargs.get('per_language_f1'),
            per_language_accuracy=kwargs.get('per_language_accuracy'),
            per_language_samples=kwargs.get('per_language_samples'),
        ))

    try:
        send_progress(0, 1, "loading_data", msg="Loading training data...")

        # Load data
        data_path = Path(task.data_path)
        samples = []
        all_labels = set()

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    label_data = record.get('label') or record.get('labels', [])

                    if isinstance(label_data, str) and label_data.startswith('['):
                        try:
                            import ast
                            label_data = ast.literal_eval(label_data)
                        except:
                            pass

                    if isinstance(label_data, list):
                        for lbl in label_data:
                            if lbl:
                                all_labels.add(str(lbl))
                    elif label_data:
                        all_labels.add(str(label_data))

                    sample_lang = record.get('language') or record.get('lang', 'EN')
                    samples.append(DataSample(
                        text=record.get('text', ''),
                        label=label_data,
                        lang=str(sample_lang).upper() if sample_lang else 'EN',
                    ))

        if not samples:
            raise ValueError(f"No samples loaded from {data_path}")

        # Filter by language if train_by_language is enabled
        language_code = task.config.get('language_code')
        if language_code:
            language_code_upper = str(language_code).upper()
            original_count = len(samples)
            samples = [s for s in samples if s.lang == language_code_upper]
            logger.info(f"[{worker_id}] Filtered to language '{language_code_upper}': {len(samples):,}/{original_count:,} samples")
            if not samples:
                raise ValueError(f"No samples for language '{language_code_upper}' in {data_path}")

        # Split data
        train_samples, val_samples = train_test_split(
            samples, test_size=0.2, random_state=42
        )

        # Extract unique languages from samples for per-language metrics tracking
        unique_languages = sorted(set(s.lang for s in samples if s.lang))
        track_languages = len(unique_languages) > 0

        num_labels = len(all_labels) if len(all_labels) > 2 else 2
        class_names = sorted(list(all_labels)) if all_labels else None
        epochs = task.config.get('epochs', 10)
        # CPU worker: apply per-session optimizer hyperparameters from the task
        # config (the worker-level TrainingConfig is built once with defaults).
        trainer.config.learning_rate = task.config.get('learning_rate', getattr(trainer.config, 'learning_rate', 2e-5))
        trainer.config.warmup_ratio = task.config.get('warmup_ratio', getattr(trainer.config, 'warmup_ratio', 0.0))

        send_progress(0, epochs, "training",
                     msg=f"Train: {len(train_samples):,} | Val: {len(val_samples):,} | Lang: {len(unique_languages)}")

        # Epoch-level progress callback (receives extended metrics from bert_base)
        def epoch_progress_callback(**metrics):
            """Called after each epoch with full metrics including per-class and per-language."""
            # Check for cancellation signal
            if cancelled_tasks is not None and task.task_id in cancelled_tasks:
                logger.info(f"[{worker_id}] Task {task.category_name} cancelled at epoch {metrics.get('epoch', 0)}")
                raise TaskCancelledException(f"Task {task.task_id} cancelled by scheduler")

            epoch = metrics.get('epoch', 0)
            send_progress(
                epoch, epochs, "validation",
                train_loss=metrics.get('train_loss', 0),
                val_loss=metrics.get('val_loss', 0),
                f1=metrics.get('f1_macro', 0),
                acc=metrics.get('accuracy', 0),
                combined_score=metrics.get('combined_score', metrics.get('combined_metric', 0)),
                # Extended metrics
                class_names=metrics.get('class_names'),
                per_class_f1=metrics.get('per_class_f1'),
                per_class_precision=metrics.get('per_class_precision'),
                per_class_recall=metrics.get('per_class_recall'),
                per_class_support=metrics.get('per_class_support'),
                per_language_f1=metrics.get('per_language_f1'),
                per_language_accuracy=metrics.get('per_language_accuracy'),
                per_language_samples=metrics.get('per_language_samples'),
            )

        # Batch progress callback
        def batch_callback(current_batch, total_batches, epoch, phase, train_loss=0, val_loss=0):
            send_progress(
                epoch, epochs, phase,
                train_loss=train_loss,
                val_loss=val_loss,
                current_batch=current_batch,
                total_batches=total_batches,
            )

        # Train
        try:
            result = trainer.train_single_model(
                label_name=task.category_name,
                train_samples=train_samples,
                val_samples=val_samples,
                num_labels=num_labels,
                class_names=class_names,
                session_id=task.config.get('session_id'),
                progress_callback=epoch_progress_callback,  # Extended metrics callback
                batch_progress_callback=batch_callback,
                suppress_display=True,
                model_output_dir=task.output_path,
                training_approach=task.config.get('training_approach'),
                # Enable per-language metrics tracking
                track_languages=track_languages,
                language_info=unique_languages if track_languages else None,
            )
        except RuntimeError as oom:
            # Handle occasional MPS OOM by reducing batch size and retrying once
            if "out of memory" in str(oom).lower():
                new_bs = max(8, int(config.batch_size / 2))
                logger.warning(f"[GPU] OOM detected, retrying {task.category_name} with batch_size={new_bs}")
                trainer.config.batch_size = new_bs
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    result = trainer.train_single_model(
                        label_name=task.category_name,
                        train_samples=train_samples,
                        val_samples=val_samples,
                        num_labels=num_labels,
                        class_names=class_names,
                        session_id=task.config.get('session_id'),
                        progress_callback=epoch_progress_callback,  # Extended metrics callback
                        batch_progress_callback=batch_callback,
                        suppress_display=True,
                        model_output_dir=task.output_path,
                        training_approach=task.config.get('training_approach'),
                        # Enable per-language metrics tracking
                        track_languages=track_languages,
                        language_info=unique_languages if track_languages else None,
                    )
                except Exception as e:
                    raise e
            else:
                raise oom

        # Extract metrics
        metrics = result.performance_metrics if hasattr(result, 'performance_metrics') else {}
        f1_score = metrics.get('f1_macro', metrics.get('macro_f1', 0))
        accuracy = metrics.get('accuracy', 0)
        # Extract best_epoch from metrics or result object
        best_epoch = metrics.get('best_epoch', 0)
        if best_epoch == 0 and hasattr(result, 'best_epoch'):
            best_epoch = result.best_epoch
        elapsed = time.time() - start_time

        logger.info(f"[{worker_id}] Completed: {task.category_name} | F1={f1_score:.4f} | Best epoch={best_epoch} | {elapsed:.1f}s")
        send_progress(epochs, epochs, "complete", f1=f1_score, acc=accuracy)

        return TaskResult(
            task_id=task.task_id,
            category_name=task.category_name,
            status="success",
            f1_score=f1_score,
            accuracy=accuracy,
            best_epoch=best_epoch,
            elapsed_seconds=elapsed,
            model_path=task.output_path,
            device_type="cpu",
        ), False  # was_cancelled = False

    except TaskCancelledException:
        # Task was cancelled mid-training - return partial result
        elapsed = time.time() - start_time
        logger.info(f"[{worker_id}] Task {task.category_name} cancelled after {elapsed:.1f}s")
        send_progress(0, 1, "cancelled", msg="Rescheduled to GPU")
        return TaskResult(
            task_id=task.task_id,
            category_name=task.category_name,
            status="cancelled",
            error_message="Cancelled for GPU rescheduling",
            elapsed_seconds=elapsed,
            device_type="cpu",
        ), True  # was_cancelled = True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[{worker_id}] Error on {task.category_name}: {error_msg}")
        send_progress(0, 1, "error", msg=error_msg)
        return TaskResult(
            task_id=task.task_id,
            category_name=task.category_name,
            status="error",
            error_message=error_msg,
            elapsed_seconds=time.time() - start_time,
            device_type="cpu",
        ), False  # was_cancelled = False


# ============================================================================
# GPU TRAINING (Main Process)
# ============================================================================

def _train_on_gpu(
    task: TrainingTask,
    progress_callback: Callable,
    gpu_config: Dict[str, Any],
    scheduler: Optional[SmartTaskScheduler] = None,
) -> TaskResult:
    """
    Train a model on GPU in the main process.

    This runs exactly like normal mode - no spawn overhead,
    full GPU utilization.

    Parameters
    ----------
    task : TrainingTask
        The training task to execute
    progress_callback : Callable
        Callback for progress updates
    gpu_config : Dict[str, Any]
        GPU configuration (batch size, workers, etc.)
    scheduler : SmartTaskScheduler, optional
        Scheduler for epoch progress tracking (enables dynamic rebalancing)
    """
    from llm_tool.trainers.multi_label_trainer import MultiLabelTrainer, TrainingConfig
    from llm_tool.trainers.data_utils import DataSample
    from sklearn.model_selection import train_test_split

    # Reduce console logging to keep Rich dashboard clean, but keep ERROR level visible
    # This ensures GPU errors are still propagated and logged
    original_levels = {}
    try:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler):
                original_levels[id(h)] = h.level
                h.setLevel(logging.ERROR)  # Keep ERROR and CRITICAL visible
    except Exception:
        pass

    start_time = time.time()
    last_epoch_end = start_time  # For tracking epoch duration
    logger.info(f"[GPU] Starting task: {task.category_name}")

    def send_progress(epoch, total, phase, **kwargs):
        progress_callback(ProgressUpdate(
            task_id=task.task_id,
            worker_id="gpu_main",
            device_type="gpu",
            category_name=task.category_name,
            current_epoch=epoch,
            total_epochs=total,
            phase=phase,
            train_loss=kwargs.get('train_loss', 0),
            val_loss=kwargs.get('val_loss', 0),
            f1_score=kwargs.get('f1', 0),
            accuracy=kwargs.get('acc', 0),
            combined_score=kwargs.get('combined_score', 0),
            current_batch=kwargs.get('current_batch', 0),
            total_batches=kwargs.get('total_batches', 0),
            message=kwargs.get('msg', ''),
            # Extended metrics for dashboard
            class_names=kwargs.get('class_names'),
            per_class_f1=kwargs.get('per_class_f1'),
            per_class_precision=kwargs.get('per_class_precision'),
            per_class_recall=kwargs.get('per_class_recall'),
            per_class_support=kwargs.get('per_class_support'),
            per_language_f1=kwargs.get('per_language_f1'),
            per_language_accuracy=kwargs.get('per_language_accuracy'),
            per_language_samples=kwargs.get('per_language_samples'),
        ))

    try:
        send_progress(0, 1, "loading_data", msg="Loading training data...")

        # Load data
        data_path = Path(task.data_path)
        samples = []
        all_labels = set()

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    label_data = record.get('label') or record.get('labels', [])

                    if isinstance(label_data, str) and label_data.startswith('['):
                        try:
                            import ast
                            label_data = ast.literal_eval(label_data)
                        except:
                            pass

                    if isinstance(label_data, list):
                        for lbl in label_data:
                            if lbl:
                                all_labels.add(str(lbl))
                    elif label_data:
                        all_labels.add(str(label_data))

                    sample_lang = record.get('language') or record.get('lang', 'EN')
                    samples.append(DataSample(
                        text=record.get('text', ''),
                        label=label_data,
                        lang=str(sample_lang).upper() if sample_lang else 'EN',
                    ))

        if not samples:
            raise ValueError(f"No samples loaded from {data_path}")

        # Filter by language if train_by_language is enabled
        language_code = task.config.get('language_code')
        if language_code:
            language_code_upper = str(language_code).upper()
            original_count = len(samples)
            samples = [s for s in samples if s.lang == language_code_upper]
            logger.info(f"[GPU] Filtered to language '{language_code_upper}': {len(samples):,}/{original_count:,} samples")
            if not samples:
                raise ValueError(f"No samples for language '{language_code_upper}' in {data_path}")

        # Split data
        train_samples, val_samples = train_test_split(
            samples, test_size=0.2, random_state=42
        )

        # Extract unique languages from samples for per-language metrics tracking
        unique_languages = sorted(set(s.lang for s in samples if s.lang))
        track_languages = len(unique_languages) > 0

        num_labels = len(all_labels) if len(all_labels) > 2 else 2
        class_names = sorted(list(all_labels)) if all_labels else None
        epochs = task.config.get('epochs', 10)

        send_progress(0, epochs, "loading_model",
                     msg=f"Train: {len(train_samples):,} | Val: {len(val_samples):,} | Lang: {len(unique_languages)}")

        # Create trainer for GPU
        dl_workers = gpu_config.get('dataloader_workers', 8)
        dl_prefetch = gpu_config.get('dataloader_prefetch', 4)

        config = TrainingConfig(
            model_name=task.model_name,
            n_epochs=epochs,
            batch_size=gpu_config.get('batch_size', 128),
            learning_rate=task.config.get('learning_rate', 2e-5),
            warmup_ratio=task.config.get('warmup_ratio', 0.0),
            force_device='mps',  # Force MPS for Apple Silicon
            dataloader_num_workers=dl_workers,
            dataloader_prefetch_factor=dl_prefetch,
            dataloader_persistent_workers=dl_workers > 0,
            multi_label=task.config.get('multi_label', False),
            multi_label_threshold=task.config.get('multi_label_threshold', 0.5),
            training_approach=task.config.get('training_approach'),
            # NORMAL-phase imbalance handling (session-global, carried per task)
            imbalance_strategy=task.config.get('imbalance_strategy'),
            focal_gamma=task.config.get('focal_gamma', 2.0),
            imbalance_weight_source=task.config.get('imbalance_weight_source', 'auto'),
            imbalance_class_weights=task.config.get('imbalance_class_weights'),
            imbalance_weighted_sampler=task.config.get('imbalance_weighted_sampler', True),
            # Propagate tokenizer-aware filter knobs to the GPU worker too.
            exclude_long_texts=bool(task.config.get('exclude_long_texts', False)),
            split_long_texts=bool(task.config.get('split_long_texts', False)),
            max_length=int(task.config.get('max_length', 512) or 512),
        )

        trainer = MultiLabelTrainer(config, verbose=False)

        # Epoch-level progress callback (receives extended metrics from bert_base)
        def epoch_progress_callback(**metrics):
            """Called after each epoch with full metrics including per-class and per-language."""
            nonlocal last_epoch_end

            epoch = metrics.get('epoch', 0)

            # Calculate epoch duration for scheduler (enables dynamic rebalancing)
            current_time = time.time()
            epoch_time = current_time - last_epoch_end
            last_epoch_end = current_time

            # Notify scheduler for dynamic rebalancing (if available)
            if scheduler is not None and epoch > 0:
                try:
                    scheduler.update_epoch_progress(task.task_id, epoch, epoch_time)
                except Exception as e:
                    logger.debug(f"Failed to update epoch progress: {e}")

            send_progress(
                epoch, epochs, "validation",
                train_loss=metrics.get('train_loss', 0),
                val_loss=metrics.get('val_loss', 0),
                f1=metrics.get('f1_macro', 0),
                acc=metrics.get('accuracy', 0),
                combined_score=metrics.get('combined_score', metrics.get('combined_metric', 0)),
                # Extended metrics
                class_names=metrics.get('class_names'),
                per_class_f1=metrics.get('per_class_f1'),
                per_class_precision=metrics.get('per_class_precision'),
                per_class_recall=metrics.get('per_class_recall'),
                per_class_support=metrics.get('per_class_support'),
                per_language_f1=metrics.get('per_language_f1'),
                per_language_accuracy=metrics.get('per_language_accuracy'),
                per_language_samples=metrics.get('per_language_samples'),
            )

        # Batch progress callback
        def batch_callback(current_batch, total_batches, epoch, phase, train_loss=0, val_loss=0):
            send_progress(
                epoch, epochs, phase,
                train_loss=train_loss,
                val_loss=val_loss,
                current_batch=current_batch,
                total_batches=total_batches,
            )

        # Train
        result = trainer.train_single_model(
            label_name=task.category_name,
            train_samples=train_samples,
            val_samples=val_samples,
            num_labels=num_labels,
            class_names=class_names,
            session_id=task.config.get('session_id'),
            progress_callback=epoch_progress_callback,  # Extended metrics callback
            batch_progress_callback=batch_callback,
            suppress_display=True,
            model_output_dir=task.output_path,
            training_approach=task.config.get('training_approach'),
            # Enable per-language metrics tracking
            track_languages=track_languages,
            language_info=unique_languages if track_languages else None,
        )

        # Cleanup
        del trainer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Extract metrics
        metrics = result.performance_metrics if hasattr(result, 'performance_metrics') else {}
        f1_score = metrics.get('f1_macro', metrics.get('macro_f1', 0))
        accuracy = metrics.get('accuracy', 0)
        # Extract best_epoch from metrics or result object
        best_epoch = metrics.get('best_epoch', 0)
        if best_epoch == 0 and hasattr(result, 'best_epoch'):
            best_epoch = result.best_epoch

        elapsed = time.time() - start_time
        logger.info(f"[GPU] Completed: {task.category_name} | F1={f1_score:.4f} | Best epoch={best_epoch} | {elapsed:.1f}s")

        send_progress(epochs, epochs, "complete", f1=f1_score, acc=accuracy)

        return TaskResult(
            task_id=task.task_id,
            category_name=task.category_name,
            status="success",
            f1_score=f1_score,
            accuracy=accuracy,
            best_epoch=best_epoch,
            elapsed_seconds=elapsed,
            model_path=task.output_path,
            device_type="gpu",
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[GPU] Error on {task.category_name}: {error_msg}")
        traceback.print_exc()
        send_progress(0, 1, "error", msg=error_msg)
        return TaskResult(
            task_id=task.task_id,
            category_name=task.category_name,
            status="error",
            error_message=error_msg,
            elapsed_seconds=time.time() - start_time,
            device_type="gpu",
        )

    finally:
        # Restore original logging levels to avoid affecting other parts of the code
        try:
            for h in logging.getLogger().handlers:
                if isinstance(h, logging.StreamHandler) and id(h) in original_levels:
                    h.setLevel(original_levels[id(h)])
        except Exception:
            pass


# ============================================================================
# HYBRID PARALLEL TRAINER (Main Orchestrator)
# ============================================================================

class HybridParallelTrainer:
    """
    Hybrid parallel training orchestrator with intelligent memory management.

    - GPU training runs in the MAIN PROCESS (like normal mode)
    - CPU workers run in PERSISTENT BACKGROUND PROCESSES
    - Memory gate prevents OOM by checking memory before starting workers
    - Graceful degradation handles OOM by reducing workers and batch size
    """

    def __init__(
        self,
        model_name: str,
        num_cpu_workers: Optional[int] = None,  # None = auto-calculate, 0 = GPU only
        gpu_batch_size: int = 128,
        cpu_batch_size: int = 16,  # Reduced from 64 - smaller batches use less RAM on CPU
        gpu_dataloader_workers: int = 8,
        cpu_dataloader_workers: int = 2,
        cpu_threads_per_worker: int = 4,
        gpu_prefetch: int = 2,
        cpu_prefetch: int = 2,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        scheduler_config: Optional[SchedulerConfig] = None,
        # NORMAL-phase imbalance handling (session-global; forwarded to every worker)
        imbalance_strategy: Optional[str] = None,
        focal_gamma: float = 2.0,
        imbalance_weight_source: str = 'auto',
        imbalance_class_weights: Optional[List[float]] = None,
        imbalance_weighted_sampler: bool = True,
    ):
        self.model_name = model_name

        # Memory management components
        self.memory_monitor = get_memory_monitor()
        self.memory_gate = MemoryGatedWorkerPool(self.memory_monitor)
        self.degradation_handler = GracefulDegradationHandler(self.memory_monitor)

        # Intelligent task scheduler
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.scheduler = SmartTaskScheduler(self.scheduler_config)

        # Auto-calculate CPU workers only if None (not if explicitly 0)
        if num_cpu_workers is None:
            has_gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
            num_cpu_workers = self.memory_monitor.calculate_safe_cpu_workers(has_gpu_worker=has_gpu)
            logger.info(f"Auto-calculated {num_cpu_workers} safe CPU workers based on memory")
        elif num_cpu_workers == 0:
            logger.info("GPU-only mode: 0 CPU workers (explicitly set)")

        self.num_cpu_workers = num_cpu_workers

        # Session-global imbalance handling, forwarded to both CPU and GPU workers.
        _imbalance_cfg = {
            'imbalance_strategy': imbalance_strategy,
            'focal_gamma': focal_gamma,
            'imbalance_weight_source': imbalance_weight_source,
            'imbalance_class_weights': imbalance_class_weights,
            'imbalance_weighted_sampler': imbalance_weighted_sampler,
        }

        self.gpu_config = {
            'batch_size': gpu_batch_size,
            'dataloader_workers': gpu_dataloader_workers,
            'dataloader_prefetch': gpu_prefetch,
            **_imbalance_cfg,
        }

        self.cpu_config = {
            'batch_size': cpu_batch_size,
            'dataloader_workers': cpu_dataloader_workers,
            'num_threads': cpu_threads_per_worker,
            'dataloader_prefetch': cpu_prefetch,
            'multi_label': multi_label,
            'multi_label_threshold': multi_label_threshold,
            **_imbalance_cfg,
        }

        # Multiprocessing resources
        self._manager = None
        self._task_queue = None
        self._result_queue = None
        self._progress_queue = None
        self._shutdown_event = None
        self._cpu_workers = []

        # CPU task cancellation mechanism
        self._cancelled_tasks = None  # Manager.dict() - tasks to cancel
        self._cancelled_queue = None  # Queue for tasks to reschedule to GPU
        self._cpu_task_logs = {}  # Track log files created by CPU tasks

        # Results tracking
        self._results: Dict[str, TaskResult] = {}
        self._progress_callback: Optional[Callable] = None
        self._scheduler_analysis: Optional[SchedulerAnalysis] = None

    def train(
        self,
        tasks: List[TrainingTask],
        progress_callback: Optional[Callable] = None,
    ) -> List[TaskResult]:
        """
        Train all tasks using hybrid parallelism with DYNAMIC task distribution
        and intelligent memory management.

        Features:
        - All tasks go into a shared queue
        - GPU (main process) and CPU workers pull from the same queue
        - Memory gate prevents OOM by checking memory before starting workers
        - Graceful degradation if OOM occurs during training
        """
        if not tasks:
            return []

        self._progress_callback = progress_callback
        total_tasks = len(tasks)

        # Reset degradation handler for new session
        self.degradation_handler.reset()

        # Check memory before starting
        pressure = self.memory_monitor.get_memory_pressure()
        logger.info(f"Memory status: {pressure.level} ({pressure.percent_used:.1f}% used, {pressure.available_gb:.1f}GB free)")

        if pressure.level == "emergency":
            logger.warning("Emergency memory pressure! Triggering cleanup before training...")
            self.memory_monitor.trigger_cleanup()
            time.sleep(2.0)

        # Adjust CPU workers based on current memory
        if pressure.level in ("critical", "emergency"):
            original_workers = self.num_cpu_workers
            self.num_cpu_workers = 0
            logger.warning(f"High memory pressure: reduced CPU workers from {original_workers} to 0")
        elif pressure.level == "warning" and self.num_cpu_workers > 2:
            original_workers = self.num_cpu_workers
            self.num_cpu_workers = 2
            logger.warning(f"Warning memory pressure: reduced CPU workers from {original_workers} to 2")

        # Initialize multiprocessing
        ctx = mp.get_context('spawn')
        self._manager = ctx.Manager()
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._progress_queue = ctx.Queue()
        self._shutdown_event = ctx.Event()

        # CPU task cancellation mechanism
        self._cancelled_tasks = self._manager.dict()  # {task_id: True} for tasks to cancel
        self._cancelled_queue = ctx.Queue()  # Tasks cancelled mid-training, to reschedule to GPU
        self._cpu_task_logs.clear()  # Reset log tracking

        # Initialize smart task scheduler for intelligent distribution
        self._scheduler_analysis = self.scheduler.initialize_tasks(tasks, self.num_cpu_workers)
        logger.info(f"Smart scheduler: {self._scheduler_analysis.recommendation}")

        # CPU workers still use queue interface for cross-process communication
        # We only put tasks that CPU is allowed to take (up to optimal_cpu_share)
        # GPU will use scheduler.get_next_task() directly for smart distribution
        cpu_tasks_queued = 0
        for task in tasks:
            # Only queue tasks for CPU up to the optimal share
            # The rest will be handled by GPU via scheduler
            if cpu_tasks_queued < self._scheduler_analysis.optimal_cpu_share:
                self._task_queue.put(task)
                cpu_tasks_queued += 1

        logger.info(f"Task distribution: {total_tasks} tasks (GPU via scheduler, "
                   f"{cpu_tasks_queued} queued for CPU workers)")
        logger.info(f"  Expected: GPU ~{self._scheduler_analysis.gpu_share}, "
                   f"CPU ~{self._scheduler_analysis.optimal_cpu_share}")

        # Start progress monitor thread
        progress_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        progress_thread.start()

        # Start result collector thread
        result_thread = threading.Thread(target=self._collect_results, daemon=True)
        result_thread.start()

        try:
            # CRITICAL: GPU grabs its FIRST task BEFORE starting CPU workers
            # This ensures GPU doesn't have to compete with 4 CPU workers for model loading
            first_gpu_task = None
            first_gpu_task = self.scheduler.get_next_task("gpu", "gpu_main")
            if first_gpu_task:
                logger.info(f"[GPU] Reserved first task via scheduler: {first_gpu_task.category_name}")
                # Notify scheduler that task has started
                epochs = first_gpu_task.config.get('epochs', 10)
                self.scheduler.start_task(first_gpu_task.task_id, "gpu", epochs)
            else:
                logger.warning("No tasks available for GPU from scheduler")

            # Now start CPU workers (they'll get remaining tasks)
            # This gives GPU a head start on loading the model
            self._start_cpu_workers_with_memory_gate()

            # GPU trains its first task (already has it, no competition)
            if first_gpu_task:
                logger.info(f"[GPU] Starting first task while CPU workers initialize...")
                result = _train_on_gpu(
                    task=first_gpu_task,
                    progress_callback=self._handle_progress,
                    gpu_config=self.gpu_config,
                    scheduler=self.scheduler,
                )
                self._results[first_gpu_task.task_id] = result

                # Record completion for smart scheduler ratio learning
                self.scheduler.record_completion(
                    first_gpu_task.task_id,
                    "gpu",
                    result.elapsed_seconds
                )

                # Cleanup after first GPU task
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # GPU continues pulling tasks via smart scheduler
            while True:
                # Use scheduler to get next task (handles dynamic distribution)
                task = self.scheduler.get_next_task("gpu", "gpu_main")

                if task is None:
                    # No more tasks available from scheduler
                    # Check if all tasks are done
                    if len(self._results) >= total_tasks:
                        logger.info("[GPU] All tasks completed")
                        break

                    # Check if scheduler has pending tasks
                    if not self.scheduler.has_pending_tasks():
                        logger.info("[GPU] No more pending tasks in scheduler")
                        break

                    # Otherwise, wait a bit and check again
                    # (CPU workers might still be processing, or tasks may be rebalanced)
                    time.sleep(0.5)
                    continue

                # Notify scheduler that task has started
                epochs = task.config.get('epochs', 10)
                self.scheduler.start_task(task.task_id, "gpu", epochs)

                # Train on GPU
                result = _train_on_gpu(
                    task=task,
                    progress_callback=self._handle_progress,
                    gpu_config=self.gpu_config,
                    scheduler=self.scheduler,
                )
                self._results[task.task_id] = result

                # Record completion for smart scheduler ratio learning
                self.scheduler.record_completion(
                    task.task_id,
                    "gpu",
                    result.elapsed_seconds
                )

                # Check if CPU tasks should be cancelled (GPU-only faster)
                should_cancel, tasks_to_cancel, reason = self.should_cancel_cpu_tasks()
                if should_cancel and tasks_to_cancel:
                    logger.info(f"[Scheduler] Cancelling CPU tasks: {reason}")
                    self.cancel_cpu_tasks(tasks_to_cancel, reason)

                # Process any cancelled tasks that returned from CPU workers
                rescheduled = self.process_cancelled_tasks()
                if rescheduled > 0:
                    logger.info(f"[Scheduler] {rescheduled} cancelled tasks rescheduled to GPU")

                # Cleanup between GPU tasks
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Wait for any remaining CPU tasks to complete
            timeout_start = time.time()
            while len(self._results) < total_tasks:
                if time.time() - timeout_start > 300:  # 5 min timeout
                    logger.warning(f"Timeout waiting for results: {len(self._results)}/{total_tasks}")
                    break
                time.sleep(0.5)

        finally:
            # Shutdown workers
            self._shutdown_workers()

        # Collect all results
        return list(self._results.values())

    def _start_cpu_workers_with_memory_gate(self):
        """Start persistent CPU worker processes with memory gate checks."""
        if self.num_cpu_workers == 0:
            logger.info("No CPU workers to start (GPU-only mode)")
            return

        ctx = mp.get_context('spawn')

        for i in range(self.num_cpu_workers):
            worker_id = f"cpu_worker_{i}"

            # Check memory before starting each worker
            can_start, reason, delay = self.memory_gate.can_start_worker()

            if not can_start:
                logger.warning(f"Cannot start {worker_id}: {reason}")
                # Wait for memory to become available (up to 60 seconds)
                if not self.memory_gate.wait_for_memory(timeout=60.0):
                    logger.warning(f"Timeout waiting for memory, skipping remaining CPU workers")
                    break

            if delay > 0:
                logger.info(f"Waiting {delay:.1f}s before starting {worker_id}")
                time.sleep(delay)

            p = ctx.Process(
                target=_cpu_worker_process,
                args=(
                    worker_id,
                    self._task_queue,
                    self._result_queue,
                    self._progress_queue,
                    self._shutdown_event,
                    self.model_name,
                    self.cpu_config,
                    self._cancelled_tasks,  # Shared dict for cancellation
                    self._cancelled_queue,  # Queue for cancelled tasks to reschedule
                ),
                daemon=False,
            )
            p.start()
            self._cpu_workers.append(p)
            self.memory_gate.register_worker_start()

            logger.info(f"Started {worker_id} (PID: {p.pid})")

            # Stagger worker starts to avoid memory spike
            if i < self.num_cpu_workers - 1:
                time.sleep(2.0)

    def _start_cpu_workers(self):
        """Legacy method - redirects to memory-gated version."""
        return self._start_cpu_workers_with_memory_gate()

    def _shutdown_workers(self):
        """Shutdown all CPU workers gracefully."""
        self._shutdown_event.set()

        # Send poison pills to ensure workers exit their loops
        for _ in self._cpu_workers:
            try:
                self._task_queue.put_nowait(None)
            except:
                pass

        # Wait for workers to finish (with timeout)
        for p in self._cpu_workers:
            p.join(timeout=10.0)
            if p.is_alive():
                logger.warning(f"Force terminating worker {p.pid}")
                p.terminate()
                p.join(timeout=2.0)
            # Update memory gate tracking
            self.memory_gate.register_worker_end()

        self._cpu_workers.clear()

        # Trigger cleanup after shutdown
        self.memory_monitor.trigger_cleanup()
        logger.info("All CPU workers shut down")

    def _monitor_progress(self):
        """Monitor progress updates from all workers."""
        while not self._shutdown_event.is_set():
            try:
                update = self._progress_queue.get(timeout=0.1)
                self._handle_progress(update)
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"Progress monitor error: {e}")

    def _collect_results(self):
        """Collect results from CPU workers."""
        while not self._shutdown_event.is_set():
            try:
                result = self._result_queue.get(timeout=0.1)
                self._results[result.task_id] = result

                # Record CPU completion for smart scheduler ratio learning
                self.scheduler.record_completion(
                    result.task_id,
                    "cpu",
                    result.elapsed_seconds
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"Result collector error: {e}")

    def _handle_progress(self, update: ProgressUpdate):
        """Handle a progress update."""
        if self._progress_callback:
            try:
                self._progress_callback(update)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    @classmethod
    def should_offer_parallel(
        cls,
        num_tasks: int,
        config: Optional[SchedulerConfig] = None,
    ) -> Tuple[bool, str]:
        """
        Check if parallel training should be offered for a given number of tasks.

        Parameters
        ----------
        num_tasks : int
            Number of tasks to train
        config : SchedulerConfig, optional
            Scheduler configuration (uses defaults if not provided)

        Returns
        -------
        Tuple[bool, str]
            (should_offer, explanation_message)
        """
        scheduler = SmartTaskScheduler(config or SchedulerConfig())
        return scheduler.should_offer_parallel(num_tasks)

    def cancel_cpu_tasks(self, task_ids: List[str], reason: str = "GPU-only faster") -> int:
        """
        Cancel specific CPU tasks mid-training.

        Tasks will be interrupted at the next epoch boundary and
        returned to the GPU queue for rescheduling.

        Parameters
        ----------
        task_ids : List[str]
            Task IDs to cancel
        reason : str
            Reason for cancellation (for logging)

        Returns
        -------
        int
            Number of tasks marked for cancellation
        """
        if self._cancelled_tasks is None:
            logger.warning("Cancellation mechanism not initialized")
            return 0

        cancelled_count = 0
        for task_id in task_ids:
            if task_id not in self._cancelled_tasks:
                self._cancelled_tasks[task_id] = True
                cancelled_count += 1
                logger.info(f"[Scheduler] Marked task {task_id} for cancellation: {reason}")

        return cancelled_count

    def should_cancel_cpu_tasks(self) -> Tuple[bool, List[str], str]:
        """
        Evaluate if CPU tasks should be cancelled based on real-time performance.

        Uses measured GPU and CPU times to determine if GPU-only would be faster
        than continuing with CPU workers.

        Returns
        -------
        Tuple[bool, List[str], str]
            (should_cancel, task_ids_to_cancel, reason)
        """
        status = self.scheduler.get_status()
        stats = self.scheduler.speed_tracker.get_statistics()

        gpu_avg = stats.get('gpu_avg')
        cpu_avg = stats.get('cpu_avg')

        # Need at least one completed task on each device to evaluate
        if gpu_avg is None or cpu_avg is None:
            return False, [], "Insufficient data (need completed tasks on both devices)"

        speed_ratio = cpu_avg / gpu_avg if gpu_avg > 0 else 5.0

        # Get tasks currently in progress on CPU
        cpu_in_progress = self.scheduler.runtime_estimator.get_in_progress_tasks("cpu")
        if not cpu_in_progress:
            return False, [], "No CPU tasks in progress"

        # Calculate remaining work
        remaining_tasks = status['available'] + status['reserved_gpu']
        tasks_on_cpu = len(cpu_in_progress)

        # Estimate time to finish if we cancel vs continue
        # Option A: Continue with CPU
        # CPU tasks already in progress will finish
        cpu_remaining_time = max(
            self.scheduler.runtime_estimator.estimate_remaining_time(tid)
            for tid in cpu_in_progress
        )
        # GPU handles remaining queue
        gpu_time_with_cpu = remaining_tasks * gpu_avg

        time_with_cpu = max(cpu_remaining_time, gpu_time_with_cpu)

        # Option B: Cancel CPU tasks and give them to GPU
        # All tasks go to GPU (remaining + cancelled CPU tasks)
        total_for_gpu = remaining_tasks + tasks_on_cpu
        time_gpu_only = total_for_gpu * gpu_avg

        # Factor in cancellation overhead (model reload, lost progress)
        # Assume ~30 seconds overhead per cancelled task
        cancellation_overhead = tasks_on_cpu * 30

        # Decision: Cancel if GPU-only is significantly faster (> 60s benefit)
        time_saved = time_with_cpu - (time_gpu_only + cancellation_overhead)

        if time_saved > 60:  # At least 60 seconds benefit
            reason = (f"GPU-only faster by {time_saved:.0f}s "
                     f"(ratio: {speed_ratio:.1f}x, {tasks_on_cpu} CPU tasks to cancel)")
            return True, cpu_in_progress, reason
        else:
            return False, [], f"No significant benefit ({time_saved:.0f}s)"

    def process_cancelled_tasks(self) -> int:
        """
        Process any tasks that were cancelled and returned from CPU workers.

        Moves cancelled tasks back to the GPU queue.

        Returns
        -------
        int
            Number of tasks rescheduled to GPU
        """
        if self._cancelled_queue is None:
            return 0

        rescheduled = 0
        while True:
            try:
                task = self._cancelled_queue.get_nowait()
                # Return task to available pool for GPU
                self.scheduler._available_tasks.appendleft(task)
                rescheduled += 1
                logger.info(f"[Scheduler] Rescheduled cancelled task {task.category_name} to GPU")

                # Clean up any partial logs for this task
                self._cleanup_partial_logs(task.task_id)
            except queue.Empty:
                break

        return rescheduled

    def _cleanup_partial_logs(self, task_id: str):
        """Remove partial log files created by a cancelled CPU task."""
        if task_id in self._cpu_task_logs:
            log_paths = self._cpu_task_logs.pop(task_id, [])
            for log_path in log_paths:
                try:
                    if Path(log_path).exists():
                        Path(log_path).unlink()
                        logger.debug(f"Cleaned up partial log: {log_path}")
                except Exception as e:
                    logger.debug(f"Could not clean up {log_path}: {e}")

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and statistics."""
        return self.scheduler.get_status()


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

def run_hybrid_parallel_training(
    tasks: List[Dict[str, Any]],
    model_name: str,
    session_id: str,
    output_base_dir: str = "models",
    num_cpu_workers: int = 2,
    gpu_batch_size: int = 128,
    cpu_batch_size: int = 64,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    High-level API for hybrid parallel training.

    Args:
        tasks: List of task dicts with 'category', 'data_path', 'config'
        model_name: Model to use (e.g., 'xlm-roberta-base')
        session_id: Session identifier for output organization
        output_base_dir: Base directory for model outputs
        num_cpu_workers: Number of CPU worker processes
        gpu_batch_size: Batch size for GPU training
        cpu_batch_size: Batch size for CPU training
        progress_callback: Callback for progress updates

    Returns:
        List of result dicts with training metrics
    """
    # Convert to TrainingTask objects
    training_tasks = []
    output_dir = Path(output_base_dir) / session_id / "parallel_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, task_dict in enumerate(tasks):
        category = task_dict.get('category', f'task_{i}')
        task = TrainingTask(
            task_id=f"task_{category}",
            category_name=category,
            model_name=model_name,
            data_path=task_dict['data_path'],
            output_path=str(output_dir / category),
            config={
                'epochs': task_dict.get('epochs', 10),
                'learning_rate': task_dict.get('learning_rate', 2e-5),
                'session_id': session_id,
                'multi_label': task_dict.get('multi_label', False),
                'multi_label_threshold': task_dict.get('multi_label_threshold', 0.5),
            },
        )
        training_tasks.append(task)

    # Create trainer and run
    trainer = HybridParallelTrainer(
        model_name=model_name,
        num_cpu_workers=num_cpu_workers,
        gpu_batch_size=gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
    )

    results = trainer.train(training_tasks, progress_callback)

    # Convert to dicts
    return [
        {
            'task_id': r.task_id,
            'category': r.category_name,
            'status': r.status,
            'f1_score': r.f1_score,
            'accuracy': r.accuracy,
            'best_epoch': r.best_epoch,
            'elapsed_seconds': r.elapsed_seconds,
            'model_path': r.model_path,
            'error': r.error_message if r.status == 'error' else None,
        }
        for r in results
    ]

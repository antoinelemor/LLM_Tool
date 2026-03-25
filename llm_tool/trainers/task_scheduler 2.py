#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
task_scheduler.py

MAIN OBJECTIVE:
---------------
Implements intelligent task scheduling for parallel GPU+CPU training.
Prevents CPU workers from taking tasks near the end of training when
GPU could complete them faster, reducing overall training time.

ARCHITECTURE:
-------------
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SMART TASK SCHEDULER                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
    │  │ DurationEstimator│  │ SpeedRatioTracker│  │ CutoffCalculator │  │
    │  │ - sample_count   │  │ - gpu_times[]    │  │ - remaining_tasks│  │
    │  │ - label_count    │  │ - cpu_times[]    │  │ - speed_ratio    │  │
    │  │ - historical_avg │  │ - dynamic_ratio  │  │ - cpu_depth      │  │
    │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
    │           └─────────────────────┼─────────────────────┘            │
    │                                 ▼                                   │
    │                    ┌─────────────────────────┐                     │
    │                    │   TaskReservationPool   │                     │
    │                    │   GPU Queue  [T1,T2..]  │                     │
    │                    │   Reserved   [T29,T30]  │◄── Reserved for GPU │
    │                    └─────────────────────────┘                     │
    └─────────────────────────────────────────────────────────────────────┘

FEATURES:
---------
1) TaskDurationEstimator - Estimates task duration based on metadata
2) SpeedRatioTracker - Dynamic tracking of GPU vs CPU speed ratio
3) DynamicCutoffCalculator - Calculates when to stop assigning to CPU
4) SmartTaskScheduler - Main orchestrator for intelligent task distribution

Dependencies:
-------------
- dataclasses
- typing
- threading
- time
- logging
- collections

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SchedulerConfig:
    """Configuration for intelligent task scheduling."""

    # Thresholds for parallel training decision
    gpu_only_threshold: int = 5              # < 5 tasks = GPU-only automatic
    min_tasks_for_parallel: int = 10         # Minimum to propose parallel
    recommended_tasks_for_parallel: int = 20  # Threshold for strong recommendation

    # Speed ratio settings
    default_gpu_cpu_ratio: float = 5.0       # Initial GPU/CPU speed ratio
    adaptive_ratio_learning: bool = True     # Learn ratio from actual times
    ratio_smoothing_factor: float = 0.3      # Smoothing for ratio updates (0-1)

    # GPU reservation settings
    enable_gpu_reservation: bool = True      # Reserve tasks for GPU at end
    reservation_safety_buffer: float = 1.5   # Safety buffer for reservation
    min_reserved_for_gpu: int = 2            # Minimum tasks to reserve for GPU

    # CPU cutoff settings
    cpu_cutoff_enabled: bool = True          # Enable CPU cutoff logic
    cpu_grace_tasks: int = 1                 # Tasks CPU can take before cutoff active


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskMetadata:
    """Metadata about a training task for duration estimation."""
    task_id: str
    category_name: str
    num_samples: int = 0
    num_labels: int = 0
    epochs: int = 10
    batch_size: int = 16
    estimated_duration_gpu: float = 0.0
    estimated_duration_cpu: float = 0.0
    data_path: str = ""


@dataclass
class DurationEstimate:
    """Duration estimate for a task."""
    estimated_seconds: float
    confidence: float  # 0-1, how confident we are in the estimate
    basis: str  # "historical", "formula", "default"


@dataclass
class CutoffDecision:
    """Decision about whether to assign a task to CPU."""
    should_assign: bool
    reason: str
    remaining_tasks: int
    speed_ratio: float
    cutoff_threshold: int


@dataclass
class SchedulerAnalysis:
    """Analysis result from scheduler initialization."""
    total_tasks: int
    gpu_share: int
    optimal_cpu_share: int
    speed_ratio: float
    use_cpu_cutoff: bool
    recommendation: str


# ============================================================================
# TASK DURATION ESTIMATOR
# ============================================================================

class TaskDurationEstimator:
    """
    Estimates task duration based on task metadata and historical data.

    Uses a simple formula-based approach combined with historical averages
    to predict how long a task will take on GPU vs CPU.
    """

    # Base time per 1000 samples (seconds) - empirical defaults
    BASE_TIME_PER_1K_GPU: float = 60.0   # ~1 min per 1K samples on GPU
    BASE_TIME_PER_1K_CPU: float = 300.0  # ~5 min per 1K samples on CPU

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._historical_gpu_times: Dict[str, List[float]] = {}
        self._historical_cpu_times: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def precompute_task_metadata(self, data_path: str) -> TaskMetadata:
        """
        Precompute metadata for a task by analyzing its data file.

        Parameters
        ----------
        data_path : str
            Path to the JSONL data file

        Returns
        -------
        TaskMetadata
            Metadata extracted from the file
        """
        path = Path(data_path)
        num_samples = 0
        all_labels = set()

        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            num_samples += 1
                            try:
                                record = json.loads(line)
                                label = record.get('label') or record.get('labels', [])
                                if isinstance(label, list):
                                    for lbl in label:
                                        if lbl:
                                            all_labels.add(str(lbl))
                                elif label:
                                    all_labels.add(str(label))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.debug(f"Could not analyze data file {data_path}: {e}")

        return TaskMetadata(
            task_id=path.stem,
            category_name=path.stem,
            num_samples=num_samples,
            num_labels=len(all_labels) if all_labels else 2,
            data_path=data_path,
        )

    def estimate_duration(
        self,
        metadata: TaskMetadata,
        device_type: str,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> DurationEstimate:
        """
        Estimate duration for a task on a specific device.

        Parameters
        ----------
        metadata : TaskMetadata
            Task metadata
        device_type : str
            "gpu" or "cpu"
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size

        Returns
        -------
        DurationEstimate
            Estimated duration with confidence
        """
        num_samples = metadata.num_samples or 1000

        # Check historical data first
        with self._lock:
            hist_key = f"{metadata.num_samples}_{metadata.num_labels}"
            hist_times = (self._historical_gpu_times if device_type == "gpu"
                         else self._historical_cpu_times)

            if hist_key in hist_times and hist_times[hist_key]:
                avg_time = sum(hist_times[hist_key]) / len(hist_times[hist_key])
                return DurationEstimate(
                    estimated_seconds=avg_time * (epochs / 10),  # Adjust for epochs
                    confidence=0.8,
                    basis="historical"
                )

        # Formula-based estimation
        base_time = (self.BASE_TIME_PER_1K_GPU if device_type == "gpu"
                    else self.BASE_TIME_PER_1K_CPU)

        # Factors:
        # - Samples scale linearly
        # - Epochs scale linearly
        # - Smaller batches take longer (more iterations)
        sample_factor = num_samples / 1000
        epoch_factor = epochs / 10
        batch_factor = 16 / max(batch_size, 1)  # Normalize to batch_size=16

        estimated = base_time * sample_factor * epoch_factor * batch_factor

        return DurationEstimate(
            estimated_seconds=estimated,
            confidence=0.5,
            basis="formula"
        )

    def record_actual_time(
        self,
        task_id: str,
        device_type: str,
        actual_seconds: float,
        num_samples: int = 0,
        num_labels: int = 0,
    ):
        """
        Record actual completion time for improving future estimates.

        Parameters
        ----------
        task_id : str
            Task identifier
        device_type : str
            "gpu" or "cpu"
        actual_seconds : float
            Actual time taken
        num_samples : int
            Number of samples in task
        num_labels : int
            Number of labels
        """
        with self._lock:
            hist_key = f"{num_samples}_{num_labels}"
            hist_times = (self._historical_gpu_times if device_type == "gpu"
                         else self._historical_cpu_times)

            if hist_key not in hist_times:
                hist_times[hist_key] = []

            # Keep last 10 measurements
            hist_times[hist_key].append(actual_seconds)
            if len(hist_times[hist_key]) > 10:
                hist_times[hist_key] = hist_times[hist_key][-10:]


# ============================================================================
# SPEED RATIO TRACKER
# ============================================================================

class SpeedRatioTracker:
    """
    Tracks the dynamic speed ratio between GPU and CPU training.

    The ratio represents: CPU_time / GPU_time
    A ratio of 5.0 means CPU takes 5x longer than GPU for the same task.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._gpu_times: Deque[float] = deque(maxlen=20)
        self._cpu_times: Deque[float] = deque(maxlen=20)
        self._current_ratio = self.config.default_gpu_cpu_ratio
        self._lock = threading.Lock()

    def record_completion(self, device_type: str, elapsed_seconds: float):
        """
        Record a task completion time.

        Parameters
        ----------
        device_type : str
            "gpu" or "cpu"
        elapsed_seconds : float
            Time taken to complete the task
        """
        with self._lock:
            if device_type == "gpu":
                self._gpu_times.append(elapsed_seconds)
            else:
                self._cpu_times.append(elapsed_seconds)

            # Update ratio if we have data from both
            if self._gpu_times and self._cpu_times and self.config.adaptive_ratio_learning:
                avg_gpu = sum(self._gpu_times) / len(self._gpu_times)
                avg_cpu = sum(self._cpu_times) / len(self._cpu_times)

                if avg_gpu > 0:
                    new_ratio = avg_cpu / avg_gpu
                    # Smooth the update
                    alpha = self.config.ratio_smoothing_factor
                    self._current_ratio = (alpha * new_ratio +
                                          (1 - alpha) * self._current_ratio)

                    logger.debug(f"Speed ratio updated: {self._current_ratio:.2f} "
                               f"(GPU avg: {avg_gpu:.1f}s, CPU avg: {avg_cpu:.1f}s)")

    def get_speed_ratio(self) -> float:
        """
        Get the current GPU/CPU speed ratio.

        Returns
        -------
        float
            Ratio where CPU_time = GPU_time * ratio
        """
        with self._lock:
            return self._current_ratio

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about recorded times."""
        with self._lock:
            return {
                'gpu_samples': len(self._gpu_times),
                'cpu_samples': len(self._cpu_times),
                'gpu_avg': sum(self._gpu_times) / len(self._gpu_times) if self._gpu_times else 0,
                'cpu_avg': sum(self._cpu_times) / len(self._cpu_times) if self._cpu_times else 0,
                'current_ratio': self._current_ratio,
            }


# ============================================================================
# DYNAMIC CUTOFF CALCULATOR
# ============================================================================

class DynamicCutoffCalculator:
    """
    Calculates when to stop assigning tasks to CPU workers.

    The key formula:
        CPU_STOP_CONDITION = remaining_tasks <= (speed_ratio × active_cpu_workers × safety_buffer)

    When this condition is true, remaining tasks are reserved for GPU.
    """

    def __init__(
        self,
        speed_tracker: SpeedRatioTracker,
        config: Optional[SchedulerConfig] = None,
    ):
        self.speed_tracker = speed_tracker
        self.config = config or SchedulerConfig()

    def should_assign_to_cpu(
        self,
        remaining_tasks: int,
        active_cpu_workers: int,
        gpu_is_busy: bool = False,
    ) -> CutoffDecision:
        """
        Determine if a task should be assigned to CPU.

        Parameters
        ----------
        remaining_tasks : int
            Number of tasks remaining (not yet started)
        active_cpu_workers : int
            Number of active CPU workers
        gpu_is_busy : bool
            Whether GPU is currently processing a task

        Returns
        -------
        CutoffDecision
            Decision with reasoning
        """
        if not self.config.cpu_cutoff_enabled:
            return CutoffDecision(
                should_assign=True,
                reason="CPU cutoff disabled",
                remaining_tasks=remaining_tasks,
                speed_ratio=self.speed_tracker.get_speed_ratio(),
                cutoff_threshold=0,
            )

        if active_cpu_workers == 0:
            return CutoffDecision(
                should_assign=False,
                reason="No CPU workers available",
                remaining_tasks=remaining_tasks,
                speed_ratio=self.speed_tracker.get_speed_ratio(),
                cutoff_threshold=0,
            )

        speed_ratio = self.speed_tracker.get_speed_ratio()
        safety_buffer = self.config.reservation_safety_buffer

        # Calculate cutoff threshold
        # If GPU is busy, we need to account for tasks CPU could complete
        # while GPU finishes its current task
        cutoff_threshold = int(
            speed_ratio * active_cpu_workers * safety_buffer
        )

        # Ensure minimum reservation for GPU
        cutoff_threshold = max(cutoff_threshold, self.config.min_reserved_for_gpu)

        # Decision
        should_assign = remaining_tasks > cutoff_threshold

        if should_assign:
            reason = f"Above cutoff ({remaining_tasks} > {cutoff_threshold})"
        else:
            reason = (f"Below cutoff ({remaining_tasks} <= {cutoff_threshold}), "
                     f"reserving for GPU (ratio: {speed_ratio:.1f}x)")

        return CutoffDecision(
            should_assign=should_assign,
            reason=reason,
            remaining_tasks=remaining_tasks,
            speed_ratio=speed_ratio,
            cutoff_threshold=cutoff_threshold,
        )

    def calculate_optimal_cpu_share(
        self,
        total_tasks: int,
        num_cpu_workers: int,
    ) -> int:
        """
        Calculate optimal number of tasks CPU should handle.

        This calculates upfront how many tasks CPU workers should take
        to ensure GPU finishes around the same time as CPU.

        Parameters
        ----------
        total_tasks : int
            Total number of tasks
        num_cpu_workers : int
            Number of CPU workers

        Returns
        -------
        int
            Optimal number of tasks for CPU workers combined
        """
        if num_cpu_workers == 0:
            return 0

        speed_ratio = self.speed_tracker.get_speed_ratio()

        # If GPU is R times faster than CPU:
        # - GPU can do R tasks in the time CPU does 1
        # - For N total tasks with C CPU workers:
        #   - Let x = tasks for CPU (divided among C workers)
        #   - GPU gets (N - x) tasks
        #   - Time for CPU: x / C (parallelized)
        #   - Time for GPU: (N - x) / R (in GPU time units)
        #   - Equalize: x / C = (N - x) / R
        #   - Solving: x = N * C / (C + R)

        optimal_cpu_share = int(
            total_tasks * num_cpu_workers / (num_cpu_workers + speed_ratio)
        )

        # Cap at reasonable bounds
        # CPU shouldn't get more than ~30% of tasks even with many workers
        max_cpu_share = int(total_tasks * 0.3)
        optimal_cpu_share = min(optimal_cpu_share, max_cpu_share)

        # Ensure at least some tasks for CPU if we have workers
        if num_cpu_workers > 0 and total_tasks > num_cpu_workers + 2:
            optimal_cpu_share = max(optimal_cpu_share, num_cpu_workers)

        return optimal_cpu_share


# ============================================================================
# SMART TASK SCHEDULER
# ============================================================================

class SmartTaskScheduler:
    """
    Intelligent task scheduler for hybrid GPU+CPU parallel training.

    Manages task distribution to minimize total training time by:
    1. Estimating task durations
    2. Tracking actual GPU vs CPU speed ratio
    3. Dynamically deciding when to stop assigning to CPU
    4. Reserving end-of-training tasks for GPU

    Usage:
        scheduler = SmartTaskScheduler()

        # Check if parallel training is worth it
        should_parallel, message = scheduler.should_offer_parallel(30)

        # Initialize with tasks
        analysis = scheduler.initialize_tasks(tasks, num_cpu_workers=2)

        # Workers request tasks
        task = scheduler.get_next_task("gpu", "gpu_main")
        task = scheduler.get_next_task("cpu", "cpu_worker_0")

        # Record completions
        scheduler.record_completion(task.task_id, "gpu", 45.0)
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()

        # Components
        self.duration_estimator = TaskDurationEstimator(self.config)
        self.speed_tracker = SpeedRatioTracker(self.config)
        self.cutoff_calculator = DynamicCutoffCalculator(
            self.speed_tracker, self.config
        )

        # Task pools
        self._available_tasks: Deque = deque()  # Tasks available for any worker
        self._reserved_for_gpu: Deque = deque()  # Tasks reserved for GPU only
        self._in_progress: Dict[str, Tuple[str, float]] = {}  # task_id -> (device, start_time)
        self._completed: Dict[str, float] = {}  # task_id -> elapsed_time

        # State
        self._total_tasks = 0
        self._num_cpu_workers = 0
        self._cpu_tasks_taken = 0
        self._optimal_cpu_share = 0
        self._lock = threading.RLock()
        self._initialized = False

    def should_offer_parallel(self, total_tasks: int) -> Tuple[bool, str]:
        """
        Determine if parallel training should be offered for this task count.

        Parameters
        ----------
        total_tasks : int
            Total number of tasks to train

        Returns
        -------
        Tuple[bool, str]
            (should_offer, explanation_message)
        """
        if total_tasks < self.config.gpu_only_threshold:
            return False, (
                f"GPU-only mode recommended: {total_tasks} tasks is below "
                f"threshold ({self.config.gpu_only_threshold}). "
                f"Parallel overhead would exceed benefit."
            )

        if total_tasks < self.config.min_tasks_for_parallel:
            return False, (
                f"Sequential training recommended: {total_tasks} tasks. "
                f"Parallel training is more efficient with "
                f"{self.config.min_tasks_for_parallel}+ tasks."
            )

        if total_tasks >= self.config.recommended_tasks_for_parallel:
            return True, (
                f"Parallel training strongly recommended: {total_tasks} tasks. "
                f"Expected significant speedup with GPU + CPU workers."
            )

        return True, (
            f"Parallel training available: {total_tasks} tasks. "
            f"Moderate speedup expected."
        )

    def initialize_tasks(
        self,
        tasks: List[Any],  # List of TrainingTask objects
        num_cpu_workers: int,
    ) -> SchedulerAnalysis:
        """
        Initialize the scheduler with tasks for a training session.

        Parameters
        ----------
        tasks : List[TrainingTask]
            List of training tasks
        num_cpu_workers : int
            Number of CPU workers that will be active

        Returns
        -------
        SchedulerAnalysis
            Analysis of task distribution
        """
        with self._lock:
            self._total_tasks = len(tasks)
            self._num_cpu_workers = num_cpu_workers
            self._cpu_tasks_taken = 0
            self._available_tasks.clear()
            self._reserved_for_gpu.clear()
            self._in_progress.clear()
            self._completed.clear()
            self._initialized = True

            # Calculate optimal distribution
            self._optimal_cpu_share = self.cutoff_calculator.calculate_optimal_cpu_share(
                self._total_tasks, num_cpu_workers
            )

            # Calculate GPU reserved tasks
            if self.config.enable_gpu_reservation and num_cpu_workers > 0:
                reserved_count = self.config.min_reserved_for_gpu
                speed_ratio = self.speed_tracker.get_speed_ratio()
                # Reserve enough tasks that GPU can handle while CPU finishes
                dynamic_reserve = int(speed_ratio * self.config.reservation_safety_buffer)
                reserved_count = max(reserved_count, min(dynamic_reserve, len(tasks) // 3))
            else:
                reserved_count = 0

            # Distribute tasks
            for i, task in enumerate(tasks):
                if i >= len(tasks) - reserved_count and reserved_count > 0:
                    self._reserved_for_gpu.append(task)
                else:
                    self._available_tasks.append(task)

            gpu_share = len(tasks) - self._optimal_cpu_share
            use_cutoff = self.config.cpu_cutoff_enabled and num_cpu_workers > 0

            # Generate recommendation
            if num_cpu_workers == 0:
                recommendation = "GPU-only mode: all tasks will run on GPU"
            elif self._total_tasks < self.config.min_tasks_for_parallel:
                recommendation = (
                    f"Note: {self._total_tasks} tasks is below recommended minimum "
                    f"({self.config.min_tasks_for_parallel}) for parallel training"
                )
            else:
                recommendation = (
                    f"Smart scheduling: GPU ~{gpu_share} tasks, "
                    f"CPU workers ~{self._optimal_cpu_share} tasks, "
                    f"{reserved_count} reserved for GPU finish"
                )

            logger.info(f"Task scheduler initialized: {recommendation}")
            logger.info(f"  Available pool: {len(self._available_tasks)}, "
                       f"GPU reserved: {len(self._reserved_for_gpu)}")

            return SchedulerAnalysis(
                total_tasks=self._total_tasks,
                gpu_share=gpu_share,
                optimal_cpu_share=self._optimal_cpu_share,
                speed_ratio=self.speed_tracker.get_speed_ratio(),
                use_cpu_cutoff=use_cutoff,
                recommendation=recommendation,
            )

    def get_next_task(
        self,
        worker_type: str,
        worker_id: str,
    ) -> Optional[Any]:
        """
        Get the next task for a worker.

        Parameters
        ----------
        worker_type : str
            "gpu" or "cpu"
        worker_id : str
            Unique worker identifier

        Returns
        -------
        Optional[TrainingTask]
            Next task to process, or None if no tasks available
        """
        with self._lock:
            if not self._initialized:
                logger.warning("Scheduler not initialized")
                return None

            # GPU can take from both pools
            if worker_type == "gpu":
                # First try available pool
                if self._available_tasks:
                    task = self._available_tasks.popleft()
                    self._in_progress[task.task_id] = (worker_type, time.time())
                    logger.debug(f"[{worker_id}] Taking task {task.category_name} from available pool")
                    return task

                # Then try reserved pool
                if self._reserved_for_gpu:
                    task = self._reserved_for_gpu.popleft()
                    self._in_progress[task.task_id] = (worker_type, time.time())
                    logger.debug(f"[{worker_id}] Taking task {task.category_name} from GPU reserved pool")
                    return task

                return None

            # CPU workers only take from available pool with cutoff check
            if worker_type == "cpu":
                remaining = len(self._available_tasks) + len(self._reserved_for_gpu)

                # Check cutoff condition
                decision = self.cutoff_calculator.should_assign_to_cpu(
                    remaining_tasks=remaining,
                    active_cpu_workers=self._num_cpu_workers,
                    gpu_is_busy=self._is_gpu_busy(),
                )

                if not decision.should_assign:
                    logger.debug(f"[{worker_id}] CPU cutoff active: {decision.reason}")
                    return None

                # Check if CPU has taken its share
                if (self.config.cpu_cutoff_enabled and
                    self._cpu_tasks_taken >= self._optimal_cpu_share):
                    logger.debug(f"[{worker_id}] CPU reached optimal share ({self._cpu_tasks_taken}/{self._optimal_cpu_share})")
                    return None

                # Take from available pool only
                if self._available_tasks:
                    task = self._available_tasks.popleft()
                    self._in_progress[task.task_id] = (worker_type, time.time())
                    self._cpu_tasks_taken += 1
                    logger.debug(f"[{worker_id}] Taking task {task.category_name} "
                               f"(CPU task {self._cpu_tasks_taken}/{self._optimal_cpu_share})")
                    return task

                return None

            logger.warning(f"Unknown worker type: {worker_type}")
            return None

    def record_completion(
        self,
        task_id: str,
        device_type: str,
        elapsed_seconds: float,
    ):
        """
        Record task completion for ratio learning.

        Parameters
        ----------
        task_id : str
            Completed task ID
        device_type : str
            "gpu" or "cpu"
        elapsed_seconds : float
            Time taken to complete
        """
        with self._lock:
            if task_id in self._in_progress:
                del self._in_progress[task_id]

            self._completed[task_id] = elapsed_seconds

            # Update speed ratio tracker
            self.speed_tracker.record_completion(device_type, elapsed_seconds)

            logger.debug(f"Task {task_id} completed on {device_type} in {elapsed_seconds:.1f}s")

    def _is_gpu_busy(self) -> bool:
        """Check if GPU is currently processing a task."""
        for task_id, (device, _) in self._in_progress.items():
            if device == "gpu":
                return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        with self._lock:
            return {
                'total_tasks': self._total_tasks,
                'available': len(self._available_tasks),
                'reserved_gpu': len(self._reserved_for_gpu),
                'in_progress': len(self._in_progress),
                'completed': len(self._completed),
                'cpu_tasks_taken': self._cpu_tasks_taken,
                'optimal_cpu_share': self._optimal_cpu_share,
                'speed_ratio': self.speed_tracker.get_speed_ratio(),
            }

    def has_pending_tasks(self) -> bool:
        """Check if there are any tasks still pending."""
        with self._lock:
            return bool(self._available_tasks or self._reserved_for_gpu or self._in_progress)

    def get_remaining_count(self) -> int:
        """Get count of remaining tasks (not completed)."""
        with self._lock:
            return (len(self._available_tasks) +
                   len(self._reserved_for_gpu) +
                   len(self._in_progress))


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

def should_use_parallel_training(
    num_tasks: int,
    config: Optional[SchedulerConfig] = None,
) -> Tuple[bool, str]:
    """
    Quick check if parallel training should be used.

    Parameters
    ----------
    num_tasks : int
        Number of tasks to train
    config : SchedulerConfig, optional
        Configuration to use

    Returns
    -------
    Tuple[bool, str]
        (should_use, reason)
    """
    scheduler = SmartTaskScheduler(config)
    return scheduler.should_offer_parallel(num_tasks)


def get_default_scheduler_config() -> SchedulerConfig:
    """Get the default scheduler configuration."""
    return SchedulerConfig()

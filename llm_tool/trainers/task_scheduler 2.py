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
    min_tasks_for_parallel: int = 6          # Minimum to propose parallel
    recommended_tasks_for_parallel: int = 15  # Threshold for strong recommendation

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
class RuntimeMetrics:
    """Real-time metrics for a task during execution."""
    task_id: str
    device_type: str  # "gpu" or "cpu"
    start_time: float
    end_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    epochs_completed: int = 0
    total_epochs: int = 0
    avg_epoch_time: float = 0.0  # For fine-grained estimation


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
                'gpu_avg': sum(self._gpu_times) / len(self._gpu_times) if self._gpu_times else None,
                'cpu_avg': sum(self._cpu_times) / len(self._cpu_times) if self._cpu_times else None,
                'current_ratio': self._current_ratio,
            }


# ============================================================================
# RUNTIME ESTIMATOR
# ============================================================================

class RuntimeEstimator:
    """
    Estimates remaining time DURING training execution.

    Tracks actual completion times and provides real-time estimates
    for remaining tasks. Used by DynamicBalancer for rebalancing decisions.
    """

    def __init__(self):
        self._task_metrics: Dict[str, RuntimeMetrics] = {}
        self._completed_gpu: List[float] = []  # Completion times for GPU tasks
        self._completed_cpu: List[float] = []  # Completion times for CPU tasks
        self._lock = threading.Lock()

    def start_task(self, task_id: str, device_type: str, total_epochs: int):
        """
        Record the start of a task.

        Parameters
        ----------
        task_id : str
            Unique task identifier
        device_type : str
            "gpu" or "cpu"
        total_epochs : int
            Expected number of epochs
        """
        with self._lock:
            self._task_metrics[task_id] = RuntimeMetrics(
                task_id=task_id,
                device_type=device_type,
                start_time=time.time(),
                total_epochs=total_epochs,
            )

    def update_epoch(self, task_id: str, epoch: int, epoch_time: float):
        """
        Update metrics after each epoch for fine-grained estimation.

        Parameters
        ----------
        task_id : str
            Task identifier
        epoch : int
            Current epoch number (1-indexed)
        epoch_time : float
            Time taken for this epoch in seconds
        """
        with self._lock:
            if task_id in self._task_metrics:
                m = self._task_metrics[task_id]
                m.epochs_completed = epoch
                # Exponential moving average for stability
                if m.avg_epoch_time > 0:
                    m.avg_epoch_time = m.avg_epoch_time * 0.7 + epoch_time * 0.3
                else:
                    m.avg_epoch_time = epoch_time

    def complete_task(self, task_id: str) -> float:
        """
        Record task completion and return total elapsed time.

        Parameters
        ----------
        task_id : str
            Task identifier

        Returns
        -------
        float
            Total elapsed time in seconds
        """
        with self._lock:
            if task_id in self._task_metrics:
                m = self._task_metrics[task_id]
                m.end_time = time.time()
                m.elapsed_seconds = m.end_time - m.start_time

                if m.device_type == "gpu":
                    self._completed_gpu.append(m.elapsed_seconds)
                else:
                    self._completed_cpu.append(m.elapsed_seconds)

                return m.elapsed_seconds
            return 0.0

    def estimate_remaining_time(self, task_id: str) -> float:
        """
        Estimate remaining time for an in-progress task.

        Parameters
        ----------
        task_id : str
            Task identifier

        Returns
        -------
        float
            Estimated remaining time in seconds
        """
        with self._lock:
            if task_id not in self._task_metrics:
                return 0.0

            m = self._task_metrics[task_id]
            if m.epochs_completed == 0 or m.avg_epoch_time == 0:
                # Not enough data, use historical average
                avg = self._get_avg_time(m.device_type)
                elapsed = time.time() - m.start_time
                return max(0, avg - elapsed)

            remaining_epochs = m.total_epochs - m.epochs_completed
            return remaining_epochs * m.avg_epoch_time

    def _get_avg_time(self, device_type: str) -> float:
        """Get historical average time for a device type."""
        times = self._completed_gpu if device_type == "gpu" else self._completed_cpu
        return sum(times) / len(times) if times else 300.0  # 5 min default

    def get_avg_completion_time(self, device_type: str) -> Optional[float]:
        """
        Get average completion time for a device type.

        Parameters
        ----------
        device_type : str
            "gpu" or "cpu"

        Returns
        -------
        Optional[float]
            Average time in seconds, or None if no data
        """
        with self._lock:
            times = self._completed_gpu if device_type == "gpu" else self._completed_cpu
            return sum(times) / len(times) if times else None

    def get_in_progress_tasks(self, device_type: Optional[str] = None) -> List[str]:
        """Get list of task IDs currently in progress."""
        with self._lock:
            result = []
            for task_id, m in self._task_metrics.items():
                if m.end_time is None:  # Not completed
                    if device_type is None or m.device_type == device_type:
                        result.append(task_id)
            return result


# ============================================================================
# DYNAMIC BALANCER
# ============================================================================

class DynamicBalancer:
    """
    Rebalances tasks dynamically during training execution.

    Goal: GPU and CPU workers finish at approximately the same time.

    The balancer monitors ETA for both device types and moves tasks
    between pools when significant imbalance is detected.
    """

    def __init__(
        self,
        scheduler: 'SmartTaskScheduler',
        runtime_estimator: RuntimeEstimator,
        rebalance_interval: float = 15.0,  # Check every 15 seconds (was 30)
        imbalance_threshold: float = 30.0,  # Rebalance if > 30s difference (was 60)
    ):
        self.scheduler = scheduler
        self.runtime_estimator = runtime_estimator
        self.rebalance_interval = rebalance_interval
        self.imbalance_threshold = imbalance_threshold
        self._last_rebalance = 0.0
        self._lock = threading.Lock()
        self._rebalance_count = 0

    def should_rebalance(self) -> Tuple[bool, str]:
        """
        Check if rebalancing is needed.

        Returns
        -------
        Tuple[bool, str]
            (should_rebalance, reason)
        """
        now = time.time()
        if now - self._last_rebalance < self.rebalance_interval:
            return False, "Too soon since last rebalance"

        eta_gpu, eta_cpu = self._calculate_etas()

        # Can't balance if we don't have estimates
        if eta_gpu is None or eta_cpu is None:
            return False, "Insufficient data for ETA calculation"

        imbalance = abs(eta_gpu - eta_cpu)

        if imbalance < self.imbalance_threshold:
            return False, f"Imbalance {imbalance:.0f}s is acceptable"

        return True, f"Imbalance {imbalance:.0f}s exceeds threshold ({self.imbalance_threshold}s)"

    def rebalance(self) -> Dict[str, Any]:
        """
        Perform task rebalancing between GPU and CPU pools.

        Returns
        -------
        Dict[str, Any]
            Actions taken during rebalancing
        """
        with self._lock:
            self._last_rebalance = time.time()
            self._rebalance_count += 1

            eta_gpu, eta_cpu = self._calculate_etas()

            actions = {
                'rebalance_number': self._rebalance_count,
                'eta_gpu_before': eta_gpu,
                'eta_cpu_before': eta_cpu,
                'tasks_moved': 0,
                'direction': None,
            }

            if eta_gpu is None or eta_cpu is None:
                actions['error'] = "Insufficient data"
                return actions

            if eta_gpu < eta_cpu - self.imbalance_threshold:
                # GPU will finish early → give it more tasks
                tasks_to_move = self._calculate_tasks_to_move(eta_cpu - eta_gpu, "to_gpu")
                self.scheduler._move_tasks_to_gpu(tasks_to_move)
                actions['tasks_moved'] = tasks_to_move
                actions['direction'] = 'to_gpu'

            elif eta_cpu < eta_gpu - self.imbalance_threshold:
                # CPU will finish early → give it more tasks
                tasks_to_move = self._calculate_tasks_to_move(eta_gpu - eta_cpu, "to_cpu")
                self.scheduler._move_tasks_to_cpu(tasks_to_move)
                actions['tasks_moved'] = tasks_to_move
                actions['direction'] = 'to_cpu'

            # Recalculate ETAs after rebalancing
            eta_gpu_after, eta_cpu_after = self._calculate_etas()
            actions['eta_gpu_after'] = eta_gpu_after
            actions['eta_cpu_after'] = eta_cpu_after

            logger.info(f"[Balancer] Rebalance #{self._rebalance_count}: "
                       f"moved {actions['tasks_moved']} tasks {actions['direction'] or 'nowhere'}")

            return actions

    def _calculate_etas(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate estimated time to completion for GPU and CPU.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            (eta_gpu, eta_cpu) in seconds, None if insufficient data
        """
        stats = self.scheduler.speed_tracker.get_statistics()
        gpu_avg = stats['gpu_avg']
        cpu_avg = stats['cpu_avg']

        # Need at least some data
        if gpu_avg is None and cpu_avg is None:
            return None, None

        # Use defaults if one is missing
        gpu_avg = gpu_avg or 300.0  # 5 min default for GPU
        cpu_avg = cpu_avg or 1500.0  # 25 min default for CPU

        status = self.scheduler.get_status()

        # GPU: available tasks + reserved tasks (will be processed by GPU)
        # When available is empty, GPU takes from reserved
        gpu_pending = status['reserved_gpu']
        if status['available'] > 0:
            # GPU will also take from available pool
            gpu_pending += status['available']

        # CPU: only tasks that CPU can still take (up to optimal share)
        cpu_pending = max(0, status['optimal_cpu_share'] - status['cpu_tasks_taken'])
        cpu_pending = min(cpu_pending, status['available'])

        # Account for in-progress tasks
        gpu_in_progress = self.runtime_estimator.get_in_progress_tasks("gpu")
        cpu_in_progress = self.runtime_estimator.get_in_progress_tasks("cpu")

        # ETA calculation
        # GPU: pending tasks × avg time + remaining time for current task
        eta_gpu = gpu_pending * gpu_avg
        for task_id in gpu_in_progress:
            eta_gpu += self.runtime_estimator.estimate_remaining_time(task_id)

        # CPU: (pending / num_workers) × avg time + max remaining of current tasks
        num_workers = max(1, self.scheduler._num_cpu_workers)
        eta_cpu = (cpu_pending * cpu_avg) / num_workers
        if cpu_in_progress:
            max_remaining = max(
                self.runtime_estimator.estimate_remaining_time(task_id)
                for task_id in cpu_in_progress
            )
            eta_cpu += max_remaining

        return eta_gpu, eta_cpu

    def _calculate_tasks_to_move(self, time_diff: float, direction: str) -> int:
        """
        Calculate how many tasks to move to balance workload.

        Parameters
        ----------
        time_diff : float
            Time difference to balance (in seconds)
        direction : str
            "to_gpu" or "to_cpu"

        Returns
        -------
        int
            Number of tasks to move
        """
        stats = self.scheduler.speed_tracker.get_statistics()

        if direction == "to_gpu":
            gpu_avg = stats['gpu_avg'] or 300.0
            return max(1, int(time_diff / gpu_avg))
        else:
            cpu_avg = stats['cpu_avg'] or 1500.0
            num_workers = max(1, self.scheduler._num_cpu_workers)
            return max(1, int(time_diff / (cpu_avg / num_workers)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get balancer statistics."""
        with self._lock:
            return {
                'rebalance_count': self._rebalance_count,
                'last_rebalance': self._last_rebalance,
                'rebalance_interval': self.rebalance_interval,
                'imbalance_threshold': self.imbalance_threshold,
            }

    def should_cpu_take_task(self, queued_tasks: int, avg_task_time_gpu: float,
                              avg_task_time_cpu: float) -> Tuple[bool, str]:
        """
        ETA-based decision: Should CPU take another task?

        Uses real-time data to determine if assigning a task to CPU
        would increase total training time.

        Parameters
        ----------
        queued_tasks : int
            Number of tasks still in queue (not yet assigned)
        avg_task_time_gpu : float
            Average task completion time on GPU (seconds)
        avg_task_time_cpu : float
            Average task completion time on CPU (seconds)

        Returns
        -------
        Tuple[bool, str]
            (should_assign, reason)
        """
        if queued_tasks <= 0:
            return False, "No tasks in queue"

        # Get current state
        gpu_in_progress = self.runtime_estimator.get_in_progress_tasks("gpu")
        cpu_in_progress = self.runtime_estimator.get_in_progress_tasks("cpu")
        num_cpu_workers = max(1, self.scheduler._num_cpu_workers)

        # Calculate remaining time for tasks in progress
        gpu_remaining = sum(
            self.runtime_estimator.estimate_remaining_time(tid)
            for tid in gpu_in_progress
        )
        cpu_remaining = max(
            (self.runtime_estimator.estimate_remaining_time(tid)
             for tid in cpu_in_progress),
            default=0
        )

        # Use measured times or defaults
        t_gpu = avg_task_time_gpu if avg_task_time_gpu > 0 else 300.0
        t_cpu = avg_task_time_cpu if avg_task_time_cpu > 0 else 1500.0
        speed_ratio = t_cpu / t_gpu if t_gpu > 0 else 5.0

        # Scenario A: CPU takes 1 task, GPU takes rest
        cpu_tasks_if_assign = 1
        gpu_tasks_if_assign = queued_tasks - 1

        # CPU time if it takes task: current remaining + new task time / workers
        # (assuming tasks are distributed among workers)
        active_cpu_workers = len(cpu_in_progress)
        free_cpu_workers = num_cpu_workers - active_cpu_workers

        if free_cpu_workers <= 0:
            return False, f"All {num_cpu_workers} CPU workers busy"

        eta_cpu_with_task = cpu_remaining + (t_cpu / num_cpu_workers)
        eta_gpu_with_task = gpu_remaining + (gpu_tasks_if_assign * t_gpu)
        total_time_if_cpu_takes = max(eta_cpu_with_task, eta_gpu_with_task)

        # Scenario B: GPU takes all queued tasks
        eta_cpu_without = cpu_remaining
        eta_gpu_without = gpu_remaining + (queued_tasks * t_gpu)
        total_time_if_gpu_only = max(eta_cpu_without, eta_gpu_without)

        # Decision: CPU should take task only if it reduces total time
        time_saved = total_time_if_gpu_only - total_time_if_cpu_takes

        if time_saved > 10:  # At least 10s benefit to justify CPU overhead
            return True, (f"ETA benefit: {time_saved:.0f}s saved "
                         f"(CPU: {eta_cpu_with_task:.0f}s, GPU: {eta_gpu_with_task:.0f}s)")
        else:
            return False, (f"No ETA benefit: GPU-only faster by {-time_saved:.0f}s "
                          f"(ratio: {speed_ratio:.1f}x, queued: {queued_tasks})")

    def recalculate_optimal_share(self) -> int:
        """
        Recalculate optimal CPU share based on actual measured times.

        Called periodically to adjust distribution based on real performance data.

        Returns
        -------
        int
            Updated optimal CPU share
        """
        stats = self.scheduler.speed_tracker.get_statistics()
        gpu_avg = stats['gpu_avg']
        cpu_avg = stats['cpu_avg']

        if gpu_avg is None or cpu_avg is None:
            # Not enough data yet, use current setting
            return self.scheduler._optimal_cpu_share

        # Recalculate with measured ratio
        measured_ratio = cpu_avg / gpu_avg if gpu_avg > 0 else 5.0
        total_tasks = self.scheduler._total_tasks
        num_workers = self.scheduler._num_cpu_workers

        # Optimal share formula: x = N * C / (C + R)
        new_optimal = int(total_tasks * num_workers / (num_workers + measured_ratio))

        # Apply same caps as initial calculation
        if measured_ratio >= 5.0:
            max_share = int(total_tasks * 0.20)
        elif measured_ratio >= 3.0:
            max_share = int(total_tasks * 0.25)
        else:
            max_share = int(total_tasks * 0.30)

        new_optimal = min(new_optimal, max_share)

        # Log if significant change
        old_optimal = self.scheduler._optimal_cpu_share
        if abs(new_optimal - old_optimal) >= 1:
            logger.info(f"[Balancer] Recalculated CPU share: {old_optimal} → {new_optimal} "
                       f"(measured ratio: {measured_ratio:.1f}x)")

        return new_optimal


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

        # Cap at reasonable bounds based on speed ratio
        # Higher ratio = less tasks for CPU
        if speed_ratio >= 5.0:
            # GPU is 5x+ faster: CPU gets max 20% of tasks
            max_cpu_share = int(total_tasks * 0.20)
        elif speed_ratio >= 3.0:
            # GPU is 3-5x faster: CPU gets max 25% of tasks
            max_cpu_share = int(total_tasks * 0.25)
        else:
            # GPU is <3x faster: CPU gets max 30% of tasks
            max_cpu_share = int(total_tasks * 0.30)

        optimal_cpu_share = min(optimal_cpu_share, max_cpu_share)

        # Only enforce minimum if speed ratio is low (< 3x)
        # When GPU is much faster, don't force tasks to slow CPU workers
        if speed_ratio < 3.0:
            # Ensure at least some tasks for CPU if we have workers
            if num_cpu_workers > 0 and total_tasks > num_cpu_workers + 2:
                optimal_cpu_share = max(optimal_cpu_share, num_cpu_workers)
        elif speed_ratio < 5.0:
            # Medium-high ratio (3-5x): give at least 1 task per worker
            min_for_utilization = min(num_cpu_workers, total_tasks // 3)
            optimal_cpu_share = max(optimal_cpu_share, min_for_utilization)
            logger.debug(f"Medium ratio ({speed_ratio:.1f}x): CPU share = {optimal_cpu_share}")
        else:
            # Very high speed ratio (>= 5x): minimal CPU usage
            # Only 1 task total for CPU (just to keep workers warm, not blocking GPU)
            min_for_utilization = 1 if total_tasks > 2 else 0
            optimal_cpu_share = max(optimal_cpu_share, min_for_utilization)
            logger.debug(f"High speed ratio ({speed_ratio:.1f}x): CPU share minimal = {optimal_cpu_share}")

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

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        rebalance_interval: float = 30.0,
        imbalance_threshold: float = 60.0,
    ):
        self.config = config or SchedulerConfig()

        # Core components
        self.duration_estimator = TaskDurationEstimator(self.config)
        self.speed_tracker = SpeedRatioTracker(self.config)
        self.cutoff_calculator = DynamicCutoffCalculator(
            self.speed_tracker, self.config
        )

        # Dynamic rebalancing components
        self.runtime_estimator = RuntimeEstimator()
        self.dynamic_balancer: Optional[DynamicBalancer] = None  # Initialized after self
        self._rebalance_interval = rebalance_interval
        self._imbalance_threshold = imbalance_threshold

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

        # Initialize dynamic balancer (needs self reference)
        self.dynamic_balancer = DynamicBalancer(
            scheduler=self,
            runtime_estimator=self.runtime_estimator,
            rebalance_interval=rebalance_interval,
            imbalance_threshold=imbalance_threshold,
        )

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

                # Check cutoff condition (basic threshold-based)
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

                # ETA-based check: Only assign if it reduces total training time
                # This uses actual measured times for fine-grained decision
                if self.dynamic_balancer and self._cpu_tasks_taken >= 1:
                    # After first task, use measured data for smarter decisions
                    stats = self.speed_tracker.get_statistics()
                    gpu_avg = stats['gpu_avg'] or 0
                    cpu_avg = stats['cpu_avg'] or 0

                    if gpu_avg > 0 and cpu_avg > 0:
                        queued = len(self._available_tasks)
                        should_take, eta_reason = self.dynamic_balancer.should_cpu_take_task(
                            queued_tasks=queued,
                            avg_task_time_gpu=gpu_avg,
                            avg_task_time_cpu=cpu_avg,
                        )

                        if not should_take:
                            logger.info(f"[{worker_id}] ETA-based cutoff: {eta_reason}")
                            return None
                        else:
                            logger.debug(f"[{worker_id}] ETA check passed: {eta_reason}")

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
        Record task completion for ratio learning and runtime estimation.

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

            # Update runtime estimator
            self.runtime_estimator.complete_task(task_id)

            logger.debug(f"Task {task_id} completed on {device_type} in {elapsed_seconds:.1f}s")

            # Recalculate optimal CPU share based on measured times
            if self.dynamic_balancer:
                new_optimal = self.dynamic_balancer.recalculate_optimal_share()
                if new_optimal != self._optimal_cpu_share:
                    old_optimal = self._optimal_cpu_share
                    self._optimal_cpu_share = new_optimal
                    logger.info(f"[Scheduler] Adjusted CPU share: {old_optimal} → {new_optimal} "
                               f"(based on measured performance)")

            # Check if rebalancing needed after completion
            if self.dynamic_balancer:
                should_rebalance, reason = self.dynamic_balancer.should_rebalance()
                if should_rebalance:
                    actions = self.dynamic_balancer.rebalance()
                    if actions.get('tasks_moved', 0) > 0:
                        logger.info(f"[Scheduler] Post-completion rebalance: {actions['tasks_moved']} tasks "
                                   f"{actions['direction']}")

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

    # ========================================================================
    # DYNAMIC REBALANCING METHODS
    # ========================================================================

    def start_task(self, task_id: str, device_type: str, epochs: int):
        """
        Record the start of a task for runtime estimation.

        Called by the trainer when a worker begins processing a task.

        Parameters
        ----------
        task_id : str
            Unique task identifier
        device_type : str
            "gpu" or "cpu"
        epochs : int
            Number of training epochs for this task
        """
        self.runtime_estimator.start_task(task_id, device_type, epochs)

    def update_epoch_progress(self, task_id: str, epoch: int, epoch_time: float):
        """
        Update progress after each epoch for fine-grained estimation.

        Also checks if rebalancing is needed and triggers it if so.
        This provides epoch-level granularity for dynamic scheduling decisions.

        Parameters
        ----------
        task_id : str
            Task identifier
        epoch : int
            Current epoch number (1-indexed)
        epoch_time : float
            Time taken for this epoch in seconds
        """
        self.runtime_estimator.update_epoch(task_id, epoch, epoch_time)

        # Check if rebalancing is needed (every epoch for fine-grained control)
        if self.dynamic_balancer:
            # Recalculate optimal share periodically (every 3 epochs to avoid overhead)
            if epoch % 3 == 0:
                with self._lock:
                    new_optimal = self.dynamic_balancer.recalculate_optimal_share()
                    if new_optimal != self._optimal_cpu_share:
                        old_optimal = self._optimal_cpu_share
                        self._optimal_cpu_share = new_optimal
                        logger.debug(f"[Scheduler] Epoch {epoch}: adjusted CPU share {old_optimal} → {new_optimal}")

            should_rebalance, reason = self.dynamic_balancer.should_rebalance()
            if should_rebalance:
                actions = self.dynamic_balancer.rebalance()
                if actions.get('tasks_moved', 0) > 0:
                    logger.info(f"[Scheduler] Rebalanced at epoch {epoch}: {actions['tasks_moved']} tasks "
                               f"{actions['direction']}")

    def _move_tasks_to_gpu(self, count: int) -> int:
        """
        Move tasks from available pool to GPU reserved pool.

        Used by DynamicBalancer when GPU is finishing early.

        Parameters
        ----------
        count : int
            Number of tasks to move

        Returns
        -------
        int
            Number of tasks actually moved
        """
        with self._lock:
            moved = 0
            while moved < count and self._available_tasks:
                task = self._available_tasks.pop()  # Take from end
                self._reserved_for_gpu.appendleft(task)  # Add to front
                moved += 1

            if moved > 0:
                logger.debug(f"[Scheduler] Moved {moved} tasks to GPU reserved pool")

            return moved

    def _move_tasks_to_cpu(self, count: int) -> int:
        """
        Move tasks from GPU reserved pool back to available pool.

        Used by DynamicBalancer when CPU workers are finishing early.

        Parameters
        ----------
        count : int
            Number of tasks to move

        Returns
        -------
        int
            Number of tasks actually moved
        """
        with self._lock:
            moved = 0
            while moved < count and self._reserved_for_gpu:
                task = self._reserved_for_gpu.pop()  # Take from end
                self._available_tasks.appendleft(task)  # Add to front
                moved += 1

            if moved > 0:
                logger.debug(f"[Scheduler] Moved {moved} tasks from GPU reserved to available")

            return moved

    def get_balancer_statistics(self) -> Dict[str, Any]:
        """Get statistics from the dynamic balancer."""
        if self.dynamic_balancer:
            return self.dynamic_balancer.get_statistics()
        return {}

    def get_runtime_statistics(self) -> Dict[str, Any]:
        """Get runtime estimation statistics."""
        return {
            'gpu_avg_completion': self.runtime_estimator.get_avg_completion_time("gpu"),
            'cpu_avg_completion': self.runtime_estimator.get_avg_completion_time("cpu"),
            'gpu_in_progress': len(self.runtime_estimator.get_in_progress_tasks("gpu")),
            'cpu_in_progress': len(self.runtime_estimator.get_in_progress_tasks("cpu")),
        }


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

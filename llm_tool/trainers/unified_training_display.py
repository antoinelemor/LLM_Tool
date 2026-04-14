#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
unified_training_display.py

MAIN OBJECTIVE:
---------------
Provides a unified Rich-based display for tracking multiple parallel training jobs.
Shows all models being trained simultaneously with real-time progress updates,
metrics, and resource utilization.

FEATURES:
---------
1) Real-time dashboard showing ALL parallel training jobs
2) Per-model epoch progress with loss and F1 metrics
3) Per-class metrics with real label names (F1, Precision, Recall, Support)
4) Per-language metrics (F1, Accuracy, sample counts)
5) Device utilization (GPU vs CPU indicators) with throughput
6) Global progress with ETA calculation
7) Completed models summary with best scores
8) Thread-safe updates from multiple workers
9) Smooth in-place refresh without terminal flickering

Dependencies:
-------------
- time
- threading
- dataclasses
- typing
- rich (Console, Live, Panel, Table, Progress, Text, Group, box)

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console, Control, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

# Default refresh interval (30 seconds for long training sessions)
DEFAULT_REFRESH_INTERVAL = 30.0

# Terminal clearing interval (every 2 minutes to prevent memory buildup)
DEFAULT_CLEAR_INTERVAL = 120.0

# Maximum updates before forcing a clear
DEFAULT_CLEAR_UPDATE_LIMIT = 4

# Environment variable overrides
REFRESH_INTERVAL = float(os.environ.get("LLM_TOOL_PARALLEL_REFRESH", DEFAULT_REFRESH_INTERVAL))
CLEAR_INTERVAL = float(os.environ.get("LLM_TOOL_PARALLEL_CLEAR", DEFAULT_CLEAR_INTERVAL))
CLEAR_UPDATE_LIMIT = int(os.environ.get("LLM_TOOL_PARALLEL_CLEAR_UPDATES", DEFAULT_CLEAR_UPDATE_LIMIT))


# ============================================================================
# TERMINAL HOUSEKEEPING
# ============================================================================

def _clear_terminal_buffer(console_obj) -> bool:
    """
    Best-effort clearing of the active terminal surface and scrollback.

    Returns
    -------
    bool
        True when at least one clearing strategy executed without raising.
    """
    cleared = False

    # Send erase commands via Rich's control API (supported terminals will comply)
    for target in ("scroll", "screen"):
        try:
            console_obj.control(Control("erase", target))
            cleared = True
        except Exception:
            pass

    try:
        console_obj.clear()
        cleared = True
    except Exception:
        pass

    # Ensure cursor returns to origin so the next Live frame repaints from the top
    try:
        console_obj.control(Control("home"))
        cleared = True
    except Exception:
        pass

    return cleared


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelTrainingState:
    """State of a single model training job."""
    task_id: str
    category_name: str
    model_name: str
    device_type: str  # "gpu" or "cpu"
    device_id: str  # "mps", "cuda:0", "cpu:0", etc.

    # Progress
    current_epoch: int = 0
    total_epochs: int = 10
    current_batch: int = 0
    total_batches: int = 0

    # Metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    best_f1: float = 0.0
    best_epoch: int = 0

    # Extended metrics - Per-class
    class_names: List[str] = field(default_factory=list)  # Real label names
    per_class_f1: List[float] = field(default_factory=list)  # F1 per class
    per_class_precision: List[float] = field(default_factory=list)
    per_class_recall: List[float] = field(default_factory=list)
    per_class_support: List[int] = field(default_factory=list)  # Samples per class

    # Extended metrics - Per-language
    languages: List[str] = field(default_factory=list)  # e.g., ['EN', 'FR', 'DE']
    per_language_f1: Dict[str, float] = field(default_factory=dict)  # {lang: f1}
    per_language_accuracy: Dict[str, float] = field(default_factory=dict)
    per_language_samples: Dict[str, int] = field(default_factory=dict)

    # Dataset info
    num_samples_train: int = 0
    num_samples_val: int = 0
    num_labels: int = 0

    # Timing
    start_time: float = 0.0
    epoch_start_time: float = 0.0
    last_update_time: float = 0.0

    # Status
    phase: str = "initializing"  # initializing, training, validation, complete, error
    status_message: str = ""
    is_complete: bool = False
    is_error: bool = False
    error_message: str = ""

    def get_epoch_progress(self) -> float:
        """Get epoch progress as percentage."""
        if self.total_epochs == 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100

    def get_batch_progress(self) -> float:
        """Get batch progress within current epoch."""
        if self.total_batches == 0:
            return 0.0
        return (self.current_batch / self.total_batches) * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def get_eta(self) -> float:
        """Estimate remaining time."""
        elapsed = self.get_elapsed_time()
        if self.current_epoch == 0 or elapsed == 0:
            return 0.0
        time_per_epoch = elapsed / self.current_epoch
        remaining_epochs = self.total_epochs - self.current_epoch
        return time_per_epoch * remaining_epochs


@dataclass
class GlobalTrainingState:
    """Global state across all training jobs."""
    total_models: int = 0
    completed_models: int = 0
    failed_models: int = 0
    active_gpu_jobs: int = 0
    active_cpu_jobs: int = 0
    start_time: float = 0.0

    # Best results tracking
    best_f1_overall: float = 0.0
    best_model_name: str = ""
    best_category: str = ""

    # Completed models summary (bounded deque to prevent memory growth)
    completed_results: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=100)
    )


# ============================================================================
# CUSTOM PROGRESS COLUMNS
# ============================================================================

class CompactPercentColumn(ProgressColumn):
    """Compact percentage display."""
    def render(self, task):
        percentage = task.percentage if task.percentage is not None else 0
        return Text(f"{percentage:>5.1f}%", style="bright_magenta")


class MetricsColumn(ProgressColumn):
    """Display training metrics inline."""
    def __init__(self, metrics_getter):
        super().__init__()
        self.metrics_getter = metrics_getter

    def render(self, task):
        metrics = self.metrics_getter(task.id)
        if metrics:
            return Text(
                f"F1={metrics.get('f1', 0):.3f} Loss={metrics.get('loss', 0):.4f}",
                style="cyan"
            )
        return Text("")


# ============================================================================
# UNIFIED TRAINING DISPLAY
# ============================================================================

class UnifiedTrainingDisplay:
    """
    Unified Rich-based display for tracking multiple parallel training jobs.

    Usage
    -----
    ```python
    display = UnifiedTrainingDisplay()
    display.start(total_models=30)

    # Register a new training job
    display.register_model(
        task_id="task_1",
        category_name="sentiment",
        model_name="xlm-roberta-base",
        device_type="gpu",
        device_id="mps",
        total_epochs=10,
    )

    # Update progress
    display.update_epoch(
        task_id="task_1",
        epoch=1,
        train_loss=0.5,
        val_loss=0.4,
        f1_score=0.85,
    )

    # Mark complete
    display.mark_complete(task_id="task_1", final_f1=0.92)

    display.stop()
    ```
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_rate: float = 1.0,
        min_refresh_interval: float = None,
        max_label_width: int = 50,
        expand_labels: bool = True,
        auto_clear_interval: float = None,
        auto_clear_updates: int = None,
    ):
        """
        Initialize the display.

        Parameters
        ----------
        console : Console, optional
            Rich console instance
        refresh_rate : float
            Display refresh rate in Hz (default: 1.0, effectively 1 second)
        min_refresh_interval : float, optional
            Minimum seconds between refreshes (default: from REFRESH_INTERVAL env or 30s)
        max_label_width : int
            Maximum width for label columns before wrapping (default: 50)
        expand_labels : bool
            Whether to use full label names without truncation (default: True)
        auto_clear_interval : float, optional
            Seconds between terminal buffer clears (default: 120s)
        auto_clear_updates : int, optional
            Number of updates before forcing a clear (default: 4)
        """
        self.console = console or Console()
        self.refresh_rate = refresh_rate
        self.max_label_width = max_label_width
        self.expand_labels = expand_labels

        # Throttling configuration
        self._min_refresh_interval = min_refresh_interval or REFRESH_INTERVAL
        self._auto_clear_interval = auto_clear_interval or CLEAR_INTERVAL
        self._auto_clear_updates = auto_clear_updates or CLEAR_UPDATE_LIMIT

        # State tracking
        self.model_states: OrderedDict[str, ModelTrainingState] = OrderedDict()
        self.global_state = GlobalTrainingState()
        self.task_metrics: Dict[TaskID, Dict[str, float]] = {}

        # Display components
        self.live: Optional[Live] = None
        self.progress: Optional[Progress] = None
        self.task_ids: Dict[str, TaskID] = {}  # task_id -> Rich TaskID

        # Thread safety
        self._lock = threading.RLock()
        self._running = False

        # Throttling state
        self._last_refresh_time = 0.0
        self._last_clear_time = 0.0
        self._updates_since_clear = 0
        self._pending_update = False

        # Statistics for debugging
        self._updates_performed = 0
        self._updates_skipped = 0
        self._clear_operations = 0

    def start(self, total_models: int):
        """Start the display."""
        if self._running:
            return

        with self._lock:
            self._running = True
            self.global_state = GlobalTrainingState(
                total_models=total_models,
                start_time=time.time(),
            )
            self.model_states.clear()
            self.task_ids.clear()

            # Reset throttling state
            self._last_refresh_time = time.time()
            self._last_clear_time = time.time()
            self._updates_since_clear = 0
            self._updates_performed = 0
            self._updates_skipped = 0
            self._clear_operations = 0

            # Create progress tracker
            self.progress = Progress(
                SpinnerColumn(style="bold cyan"),
                TextColumn("[bold]{task.description}[/bold]", justify="left"),
                BarColumn(bar_width=25, complete_style="green", finished_style="bold green"),
                CompactPercentColumn(),
                TextColumn("•", style="dim"),
                TextColumn("{task.fields[metrics]}", style="cyan"),
                console=self.console,
                expand=False,
                auto_refresh=False,
            )

            # Start live display with transient=True for memory efficiency
            # and manual refresh control via auto_refresh=False
            self.live = Live(
                self._create_panel(),
                console=self.console,
                refresh_per_second=1 / self._min_refresh_interval,  # Convert to Hz
                transient=True,  # Auto-cleans old displays for memory efficiency
                auto_refresh=False,  # Manual control via _refresh_display
            )
            self.live.start()

            # Log throttling config
            self.console.print(
                f"[dim cyan]Parallel display: refresh every {self._min_refresh_interval:.0f}s, "
                f"clear every {self._auto_clear_interval:.0f}s or {self._auto_clear_updates} updates[/dim cyan]"
            )

    def stop(self):
        """Stop the display."""
        if not self._running:
            return

        with self._lock:
            self._running = False

            if self.live:
                # Final update showing completion
                self.live.update(self._create_panel(), refresh=True)
                time.sleep(0.3)
                self.live.stop()
                self.live = None

            if self.progress:
                self.progress = None

            # Print final summary
            self._print_final_summary()

    def register_model(
        self,
        task_id: str,
        category_name: str,
        model_name: str,
        device_type: str,
        device_id: str,
        total_epochs: int,
    ):
        """Register a new model for tracking."""
        with self._lock:
            state = ModelTrainingState(
                task_id=task_id,
                category_name=category_name,
                model_name=model_name,
                device_type=device_type,
                device_id=device_id,
                total_epochs=total_epochs,
                start_time=time.time(),
                phase="initializing",
            )
            self.model_states[task_id] = state

            # Update device counters
            if device_type == "gpu":
                self.global_state.active_gpu_jobs += 1
            else:
                self.global_state.active_cpu_jobs += 1

            # Add to progress
            if self.progress:
                rich_task_id = self.progress.add_task(
                    self._format_task_description(state),
                    total=total_epochs,
                    completed=0,
                    metrics="initializing...",
                )
                self.task_ids[task_id] = rich_task_id

            self._refresh_display()

    def update_epoch(
        self,
        task_id: str,
        epoch: int,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        f1_score: float = 0.0,
        accuracy: float = 0.0,
        phase: str = "training",
        # Extended metrics
        class_names: Optional[List[str]] = None,
        per_class_f1: Optional[List[float]] = None,
        per_class_precision: Optional[List[float]] = None,
        per_class_recall: Optional[List[float]] = None,
        per_class_support: Optional[List[int]] = None,
        per_language_f1: Optional[Dict[str, float]] = None,
        per_language_accuracy: Optional[Dict[str, float]] = None,
        per_language_samples: Optional[Dict[str, int]] = None,
        num_samples_train: Optional[int] = None,
        num_samples_val: Optional[int] = None,
        current_batch: int = 0,
        total_batches: int = 0,
        f1_macro: float = 0.0,
    ):
        """Update epoch progress for a model with extended metrics."""
        with self._lock:
            if task_id not in self.model_states:
                return

            state = self.model_states[task_id]
            state.current_epoch = epoch
            state.train_loss = train_loss
            state.val_loss = val_loss
            state.f1_score = f1_score if f1_score > 0 else f1_macro
            state.accuracy = accuracy
            state.phase = phase
            state.last_update_time = time.time()
            state.current_batch = current_batch
            state.total_batches = total_batches

            # Extended metrics - Per-class
            if class_names is not None:
                state.class_names = class_names
                state.num_labels = len(class_names)
            if per_class_f1 is not None:
                state.per_class_f1 = per_class_f1
            if per_class_precision is not None:
                state.per_class_precision = per_class_precision
            if per_class_recall is not None:
                state.per_class_recall = per_class_recall
            if per_class_support is not None:
                state.per_class_support = per_class_support

            # Extended metrics - Per-language
            if per_language_f1 is not None:
                state.per_language_f1 = per_language_f1
                state.languages = list(per_language_f1.keys())
            if per_language_accuracy is not None:
                state.per_language_accuracy = per_language_accuracy
            if per_language_samples is not None:
                state.per_language_samples = per_language_samples

            # Dataset info
            if num_samples_train is not None:
                state.num_samples_train = num_samples_train
            if num_samples_val is not None:
                state.num_samples_val = num_samples_val

            # Track best F1
            if state.f1_score > state.best_f1:
                state.best_f1 = state.f1_score
                state.best_epoch = epoch

            # Update progress bar
            if self.progress and task_id in self.task_ids:
                metrics_str = f"F1={state.f1_score:.3f} Loss={val_loss:.4f}" if val_loss > 0 else f"Loss={train_loss:.4f}"
                self.progress.update(
                    self.task_ids[task_id],
                    description=self._format_task_description(state),
                    completed=epoch,
                    metrics=metrics_str,
                )

            self._refresh_display()

    def update_batch(
        self,
        task_id: str,
        batch: int,
        total_batches: int,
        loss: float = 0.0,
    ):
        """Update batch progress within an epoch."""
        with self._lock:
            if task_id not in self.model_states:
                return

            state = self.model_states[task_id]
            state.current_batch = batch
            state.total_batches = total_batches
            state.train_loss = loss
            state.last_update_time = time.time()

            # Mark that we have pending updates (will refresh on next interval)
            self._pending_update = True
            # Refresh is now controlled solely by time-based throttling in _refresh_display

    def mark_complete(
        self,
        task_id: str,
        final_f1: float = 0.0,
        final_accuracy: float = 0.0,
        model_path: str = "",
    ):
        """Mark a model as complete."""
        with self._lock:
            if task_id not in self.model_states:
                return

            state = self.model_states[task_id]
            state.is_complete = True
            state.phase = "complete"
            state.f1_score = final_f1
            state.accuracy = final_accuracy
            state.last_update_time = time.time()

            # Update counters
            self.global_state.completed_models += 1
            if state.device_type == "gpu":
                self.global_state.active_gpu_jobs = max(0, self.global_state.active_gpu_jobs - 1)
            else:
                self.global_state.active_cpu_jobs = max(0, self.global_state.active_cpu_jobs - 1)

            # Track best overall
            if final_f1 > self.global_state.best_f1_overall:
                self.global_state.best_f1_overall = final_f1
                self.global_state.best_model_name = state.model_name
                self.global_state.best_category = state.category_name

            # Store result
            self.global_state.completed_results.append({
                "category": state.category_name,
                "f1": final_f1,
                "accuracy": final_accuracy,
                "elapsed": state.get_elapsed_time(),
                "device": state.device_type,
            })

            # Update progress bar (use full category name)
            if self.progress and task_id in self.task_ids:
                self.progress.update(
                    self.task_ids[task_id],
                    description=f"[OK] {state.category_name}",
                    completed=state.total_epochs,
                    metrics=f"F1={final_f1:.3f} [OK]",
                )

            self._refresh_display()

    def mark_error(self, task_id: str, error_message: str):
        """Mark a model as failed."""
        with self._lock:
            if task_id not in self.model_states:
                return

            state = self.model_states[task_id]
            state.is_error = True
            state.phase = "error"
            state.error_message = error_message

            self.global_state.failed_models += 1
            if state.device_type == "gpu":
                self.global_state.active_gpu_jobs = max(0, self.global_state.active_gpu_jobs - 1)
            else:
                self.global_state.active_cpu_jobs = max(0, self.global_state.active_cpu_jobs - 1)

            # Update progress bar (use full category name)
            if self.progress and task_id in self.task_ids:
                self.progress.update(
                    self.task_ids[task_id],
                    description=f"[FAIL] {state.category_name}",
                    metrics="ERROR",
                )

            self._refresh_display()

    def _format_task_description(self, state: ModelTrainingState) -> str:
        """Format task description for progress bar."""
        device_icon = "GPU" if state.device_type == "gpu" else "CPU"

        # Use full category name if expand_labels is enabled, otherwise truncate
        cat = state.category_name
        if not self.expand_labels and len(cat) > self.max_label_width:
            cat = cat[:self.max_label_width - 3] + "..."

        if state.phase == "initializing":
            return f"{device_icon} {cat} [dim]loading...[/dim]"
        elif state.phase == "complete":
            return f"[OK] {cat}"
        elif state.phase == "error":
            return f"[FAIL] {cat}"
        else:
            return f"{device_icon} {cat} E{state.current_epoch}/{state.total_epochs}"

    def _create_per_language_table(self, model_state: ModelTrainingState) -> Table:
        """
        Create a structured table for per-language metrics.

        Parameters
        ----------
        model_state : ModelTrainingState
            The model state containing per-language metrics

        Returns
        -------
        Table
            Rich Table with per-language F1, accuracy, and sample counts
        """
        lang_table = Table(
            title="Metriques par Langue",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE,
            padding=(0, 1),
        )
        lang_table.add_column("Langue", style="bold", width=8)
        lang_table.add_column("F1", justify="right", width=10)
        lang_table.add_column("Accuracy", justify="right", width=10)
        lang_table.add_column("Samples", justify="right", width=10)

        for lang in sorted(model_state.per_language_f1.keys()):
            f1_val = model_state.per_language_f1.get(lang, 0)
            acc_val = model_state.per_language_accuracy.get(lang, 0)
            samples = model_state.per_language_samples.get(lang, 0)

            f1_style = "green" if f1_val >= 0.8 else ("yellow" if f1_val >= 0.5 else "red")

            lang_table.add_row(
                lang.upper(),
                f"[{f1_style}]{f1_val:.4f}[/{f1_style}]",
                f"{acc_val:.2%}",
                f"{samples:,}",
            )

        return lang_table

    def _create_per_label_table(self, model_state: ModelTrainingState) -> Table:
        """
        Create a table for per-label metrics with full label names.

        Parameters
        ----------
        model_state : ModelTrainingState
            The model state containing per-class metrics

        Returns
        -------
        Table
            Rich Table with per-label F1, precision, recall, and support
        """
        class_table = Table(
            title="Per-Label Metrics",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE,
            padding=(0, 1),
        )
        # Use dynamic width with wrapping instead of fixed truncation
        class_table.add_column(
            "Label",
            style="white",
            min_width=20,
            max_width=self.max_label_width,
            no_wrap=False,  # Allow wrapping for long labels
            overflow="fold",
        )
        class_table.add_column("F1", justify="right", width=8)
        class_table.add_column("Prec", justify="right", width=8)
        class_table.add_column("Rec", justify="right", width=8)
        class_table.add_column("Support", justify="right", width=8)

        for i, class_name in enumerate(model_state.class_names):
            if i < len(model_state.per_class_f1):
                f1_val = model_state.per_class_f1[i]
                prec_val = model_state.per_class_precision[i] if i < len(model_state.per_class_precision) else 0
                rec_val = model_state.per_class_recall[i] if i < len(model_state.per_class_recall) else 0
                sup_val = model_state.per_class_support[i] if i < len(model_state.per_class_support) else 0

                # Color code F1
                f1_style = "bold green" if f1_val >= 0.8 else ("yellow" if f1_val >= 0.5 else "red")

                # Use full label name (no truncation if expand_labels is True)
                if self.expand_labels:
                    label_display = class_name
                else:
                    label_display = class_name if len(class_name) <= self.max_label_width - 3 else class_name[:self.max_label_width - 6] + "..."

                class_table.add_row(
                    label_display,
                    f"[{f1_style}]{f1_val:.4f}[/{f1_style}]",
                    f"{prec_val:.4f}",
                    f"{rec_val:.4f}",
                    str(sup_val),
                )

        return class_table

    def _create_model_detail_section(self, model_state: ModelTrainingState, index: int = 0) -> List[Any]:
        """
        Create a detailed metrics section for a single model.

        Parameters
        ----------
        model_state : ModelTrainingState
            The model state to display
        index : int
            Index of this model in the active list (for visual styling)

        Returns
        -------
        List[Any]
            List of Rich renderables for this model's details
        """
        elements = []

        # Only show details if model has metrics
        if model_state.current_epoch == 0:
            return elements
        if not model_state.per_class_f1 and not model_state.per_language_f1:
            return elements

        # Header with full category name
        style = "bold green" if index == 0 else "bold yellow" if index == 1 else "bold cyan"
        elements.append(Text(""))
        elements.append(Text("═" * 100, style=f"dim {style.split()[-1]}"))
        elements.append(Text(f" MODEL {index + 1}: {model_state.category_name}", style=style))
        elements.append(Text(f"    Device: {model_state.device_id} | Epoch: {model_state.current_epoch}/{model_state.total_epochs} | F1: {model_state.f1_score:.4f}", style="dim"))
        elements.append(Text("═" * 100, style=f"dim {style.split()[-1]}"))

        # Per-label metrics table
        if model_state.per_class_f1 and model_state.class_names:
            elements.append(self._create_per_label_table(model_state))

        # Per-language metrics table (structured, not inline)
        if model_state.per_language_f1:
            elements.append(Text(""))
            elements.append(self._create_per_language_table(model_state))

        return elements

    def _create_multilabel_summary_table(self, active_states: List[ModelTrainingState]) -> Optional[Table]:
        """
        Create a summary matrix showing Labels × Languages for all active models.

        Parameters
        ----------
        active_states : List[ModelTrainingState]
            List of active model states

        Returns
        -------
        Optional[Table]
            Rich Table with multi-label summary, or None if not applicable
        """
        # Collect all unique languages from all models
        all_languages = set()
        for state in active_states:
            all_languages.update(state.per_language_f1.keys())

        if not all_languages:
            return None

        sorted_langs = sorted(all_languages)

        summary_table = Table(
            title="MULTI-LABEL SUMMARY (Labels x Languages)",
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAVY,
            padding=(0, 1),
            expand=True,
        )

        # Columns: Model/Label, F1 (global), then one column per language
        summary_table.add_column(
            "Model/Label",
            style="white",
            min_width=25,
            max_width=self.max_label_width,
            no_wrap=False,
            overflow="fold",
        )
        summary_table.add_column("F1", justify="right", width=8, style="bold")

        for lang in sorted_langs:
            summary_table.add_column(lang.upper(), justify="right", width=8)

        # Add rows for each active model
        for state in active_states[:5]:  # Limit to 5 models to prevent overflow
            if not state.per_language_f1:
                continue

            # Main model row
            f1_global = state.f1_score
            f1_style = "bold green" if f1_global >= 0.8 else ("bold yellow" if f1_global >= 0.5 else "bold red")

            # Category name (with device icon)
            dev_icon = "GPU" if state.device_type == "gpu" else "CPU"
            cat_name = f"{dev_icon} {state.category_name}"

            row = [cat_name, f"[{f1_style}]{f1_global:.3f}[/{f1_style}]"]

            # Add per-language F1 values
            for lang in sorted_langs:
                lang_f1 = state.per_language_f1.get(lang, 0)
                lang_style = "green" if lang_f1 >= 0.8 else ("yellow" if lang_f1 >= 0.5 else "red")
                row.append(f"[{lang_style}]{lang_f1:.3f}[/{lang_style}]" if lang_f1 > 0 else "[dim]-[/dim]")

            summary_table.add_row(*row)

            # Optionally add per-class breakdown (indented)
            if state.class_names and state.per_class_f1 and len(state.class_names) <= 5:
                for i, class_name in enumerate(state.class_names):
                    if i < len(state.per_class_f1):
                        class_f1 = state.per_class_f1[i]
                        class_style = "green" if class_f1 >= 0.8 else ("yellow" if class_f1 >= 0.5 else "red")

                        # Indented label name
                        indent = "  └─ " if i == len(state.class_names) - 1 else "  ├─ "
                        label_display = f"[dim]{indent}{class_name}[/dim]"

                        class_row = [label_display, f"[{class_style}]{class_f1:.3f}[/{class_style}]"]
                        # Empty language columns for per-class rows
                        class_row.extend(["[dim]-[/dim]"] * len(sorted_langs))
                        summary_table.add_row(*class_row)

        return summary_table

    def _create_panel(self) -> Panel:
        """Create the main display panel with detailed metrics."""
        with self._lock:
            # ========== HEADER ==========
            elapsed = time.time() - self.global_state.start_time if self.global_state.start_time else 0
            completed = self.global_state.completed_models
            total = self.global_state.total_models
            failed = self.global_state.failed_models
            active = len([s for s in self.model_states.values() if not s.is_complete and not s.is_error])

            # Calculate ETA
            if completed > 0:
                avg_time = elapsed / completed
                remaining = total - completed - failed
                eta = avg_time * remaining
                eta_str = self._format_time(eta)
            else:
                eta_str = "calculating..."

            # Global progress bar
            global_pct = (completed / total * 100) if total > 0 else 0

            header = Table(show_header=False, box=None, padding=(0, 1), expand=True)
            header.add_column(style="bold cyan", ratio=1)
            header.add_column(style="white", ratio=1)
            header.add_column(style="bold cyan", ratio=1)
            header.add_column(style="white", ratio=1)

            header.add_row(
                "Progress:",
                f"{completed}/{total} ({global_pct:.1f}%)",
                "Elapsed:",
                self._format_time(elapsed),
            )
            header.add_row(
                "GPU Jobs:",
                str(self.global_state.active_gpu_jobs),
                "ETA:",
                eta_str,
            )
            header.add_row(
                "CPU Jobs:",
                str(self.global_state.active_cpu_jobs),
                "Best F1:",
                f"{self.global_state.best_f1_overall:.4f}" if self.global_state.best_f1_overall > 0 else "-",
            )

            if failed > 0:
                header.add_row(
                    "Failed:",
                    f"[red]{failed}[/red]",
                    "",
                    "",
                )

            # ========== ACTIVE JOBS TABLE (Enhanced with dynamic widths) ==========
            active_table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.SIMPLE_HEAVY,
                padding=(0, 1),
                expand=True,
            )
            active_table.add_column("Dev", width=3, justify="center")
            # Dynamic category column with wrapping support
            active_table.add_column(
                "Category",
                min_width=20,
                max_width=self.max_label_width,
                no_wrap=False,
                overflow="fold",
            )
            active_table.add_column("Samples", width=12, justify="center")
            active_table.add_column("Labels", width=6, justify="center")
            active_table.add_column("Epoch", width=12, justify="center")
            active_table.add_column("Loss", width=8, justify="right")
            active_table.add_column("F1", width=7, justify="right")
            active_table.add_column("Acc", width=7, justify="right")
            active_table.add_column("Status", width=10)

            # Sort: GPU first, then by progress (descending)
            active_states = [
                s for s in self.model_states.values()
                if not s.is_complete and not s.is_error
            ]
            active_states.sort(key=lambda x: (0 if x.device_type == "gpu" else 1, -x.current_epoch))

            # Show all active jobs (no limit) with full category names
            for state in active_states:
                dev = "GPU" if state.device_type == "gpu" else "CPU"
                # Use full category name (wrapping handled by table column)
                cat = state.category_name

                # Samples info
                if state.num_samples_train > 0:
                    samples = f"{state.num_samples_train:,}/{state.num_samples_val:,}"
                else:
                    samples = "-"

                # Labels count
                labels = str(state.num_labels) if state.num_labels > 0 else "-"

                # Epoch with batch progress
                if state.total_batches > 0:
                    batch_pct = int(state.current_batch / state.total_batches * 100)
                    epoch = f"E{state.current_epoch}/{state.total_epochs} B{batch_pct}%"
                else:
                    epoch = f"E{state.current_epoch}/{state.total_epochs}"

                # Metrics
                loss = f"{state.val_loss:.4f}" if state.val_loss > 0 else (f"{state.train_loss:.4f}" if state.train_loss > 0 else "-")
                f1 = f"{state.f1_score:.4f}" if state.f1_score > 0 else "-"
                acc = f"{state.accuracy:.2%}" if state.accuracy > 0 else "-"

                phase_style = {
                    "initializing": "dim",
                    "training": "yellow",
                    "validation": "cyan",
                }.get(state.phase, "white")

                status = f"[{phase_style}]{state.phase[:8].title()}[/{phase_style}]"

                active_table.add_row(dev, cat, samples, labels, epoch, loss, f1, acc, status)

            # ========== DETAILED MODEL VIEW (ALL active models with metrics) ==========
            detail_elements = []

            # Show detailed metrics for ALL active models that have metrics
            models_with_metrics = [
                s for s in active_states
                if s.current_epoch > 0 and (s.per_class_f1 or s.per_language_f1)
            ]

            for idx, model_state in enumerate(models_with_metrics):
                detail_elements.extend(self._create_model_detail_section(model_state, idx))

            # ========== MULTI-LABEL SUMMARY TABLE (if multiple models with language metrics) ==========
            models_with_lang = [s for s in active_states if s.per_language_f1]
            if len(models_with_lang) >= 2:
                summary_table = self._create_multilabel_summary_table(models_with_lang)
                if summary_table:
                    detail_elements.append(Text(""))
                    detail_elements.append(Text("═" * 100, style="dim magenta"))
                    detail_elements.append(summary_table)

            # ========== COMPLETED SUMMARY ==========
            completed_text = ""
            if self.global_state.completed_results:
                # Get last 5 from deque (convert to list for slicing)
                recent = list(self.global_state.completed_results)[-5:]
                completed_parts = []
                for r in recent:
                    # Use full category name
                    cat_name = r["category"]
                    f1_style = "green" if r['f1'] >= 0.8 else ("yellow" if r['f1'] >= 0.5 else "red")
                    completed_parts.append(f"✓ {cat_name} ([{f1_style}]F1={r['f1']:.3f}[/{f1_style}])")
                completed_text = " | ".join(completed_parts)

            # ========== COMBINE ELEMENTS ==========
            elements = [
                header,
                Text(""),
                Text("═" * 100, style="dim cyan"),
                Text(" ACTIVE TRAINING JOBS", style="bold cyan"),
                Text("═" * 100, style="dim cyan"),
                active_table,
            ]

            # Add detailed view
            elements.extend(detail_elements)

            if self.progress:
                elements.extend([
                    Text(""),
                    Text("─" * 100, style="dim"),
                    self.progress,
                ])

            if completed_text:
                elements.extend([
                    Text(""),
                    Text("─" * 100, style="dim"),
                    Text(" RECENTLY COMPLETED", style="bold green"),
                    Text(completed_text),
                ])

            return Panel(
                Group(*elements),
                title="[bold blue]PARALLEL TRAINING DASHBOARD[/bold blue]",
                subtitle=f"[dim]{active} running | {total - completed - failed - active} queued | {completed} done | {failed} failed[/dim]",
                border_style="bold blue",
                box=box.HEAVY,
            )

    def _refresh_display(self, force: bool = False):
        """
        Refresh the live display with intelligent throttling.

        Parameters
        ----------
        force : bool
            If True, bypass throttling and refresh immediately
        """
        if not self.live or not self._running:
            return

        now = time.time()
        time_elapsed = now - self._last_refresh_time

        # Check if we should refresh based on time interval
        if not force and time_elapsed < self._min_refresh_interval:
            self._updates_skipped += 1
            return

        # Perform the update
        try:
            self._last_refresh_time = now
            self._updates_since_clear += 1
            self._updates_performed += 1
            self._pending_update = False

            self.live.update(self._create_panel(), refresh=True)

            # Apply terminal housekeeping if needed
            self._apply_terminal_housekeeping()

        except Exception:
            pass  # Ignore display errors

    def _apply_terminal_housekeeping(self):
        """
        Periodically clear terminal buffer to prevent memory buildup.

        Clears when either:
        - Time since last clear exceeds auto_clear_interval
        - Number of updates since last clear exceeds auto_clear_updates
        """
        now = time.time()
        time_since_clear = now - self._last_clear_time

        due_by_time = time_since_clear >= self._auto_clear_interval
        due_by_updates = self._updates_since_clear >= self._auto_clear_updates

        if due_by_time or due_by_updates:
            cleared = _clear_terminal_buffer(self.console)
            if cleared:
                self._last_clear_time = now
                self._updates_since_clear = 0
                self._clear_operations += 1

    def get_throttle_stats(self) -> str:
        """Get statistics about throttling performance."""
        total = self._updates_performed + self._updates_skipped
        if total == 0:
            return "No updates yet"
        skip_pct = (self._updates_skipped / total) * 100 if total > 0 else 0
        return (
            f"Updates: {self._updates_performed} performed, "
            f"{self._updates_skipped} skipped ({skip_pct:.1f}%), "
            f"{self._clear_operations} clears"
        )

    def _print_final_summary(self):
        """Print final training summary."""
        if not self.global_state.completed_results:
            return

        self.console.print()

        # Print throttle stats for debugging
        self.console.print(f"[dim]{self.get_throttle_stats()}[/dim]")
        self.console.print()

        # Summary table with dynamic category column
        summary = Table(
            title="TRAINING COMPLETE",
            show_header=True,
            header_style="bold green",
            box=box.ROUNDED,
            expand=True,
        )
        # Dynamic category column with wrapping
        summary.add_column(
            "Category",
            style="cyan",
            min_width=20,
            max_width=self.max_label_width,
            no_wrap=False,
            overflow="fold",
        )
        summary.add_column("F1 Score", justify="right", width=10)
        summary.add_column("Accuracy", justify="right", width=10)
        summary.add_column("Time", justify="right", width=12)
        summary.add_column("Device", width=10)

        # Sort by F1 descending (convert deque to list for sorting)
        sorted_results = sorted(
            list(self.global_state.completed_results),
            key=lambda x: x.get("f1", 0),
            reverse=True,
        )

        for r in sorted_results[:20]:  # Top 20
            # Use full category name (wrapping handled by column)
            cat = r["category"]

            f1_style = "bold green" if r["f1"] >= 0.8 else ("yellow" if r["f1"] >= 0.6 else "red")

            summary.add_row(
                cat,
                f"[{f1_style}]{r['f1']:.4f}[/{f1_style}]",
                f"{r.get('accuracy', 0):.4f}",
                self._format_time(r.get("elapsed", 0)),
                "GPU" if r.get("device") == "gpu" else "CPU",
            )

        self.console.print(summary)

        # Best model highlight
        if self.global_state.best_f1_overall > 0:
            self.console.print()
            self.console.print(Panel(
                f"[bold green]Best Model:[/bold green] {self.global_state.best_category}\n"
                f"[cyan]F1 Score:[/cyan] {self.global_state.best_f1_overall:.4f}",
                border_style="green",
                box=box.ROUNDED,
            ))

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human-readable string."""
        if seconds < 0:
            return "-"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# ============================================================================
# PROGRESS CALLBACK FACTORY
# ============================================================================

def create_progress_callback(display: UnifiedTrainingDisplay, task_id: str):
    """
    Create a progress callback function for a training job.

    Returns a callable that can be passed to model.run_training() as progress_callback.
    """
    def callback(epoch: int, metrics: Dict[str, Any]):
        display.update_epoch(
            task_id=task_id,
            epoch=epoch,
            train_loss=metrics.get("train_loss", 0),
            val_loss=metrics.get("val_loss", 0),
            f1_score=metrics.get("f1_macro", metrics.get("f1", 0)),
            accuracy=metrics.get("accuracy", 0),
            phase=metrics.get("phase", "training"),
        )

    return callback


if __name__ == "__main__":
    # Demo the display
    import random

    console = Console()
    display = UnifiedTrainingDisplay(console)

    # Simulate parallel training
    categories = [
        "sentiment_analysis",
        "topic_classification",
        "entity_recognition",
        "intent_detection",
        "emotion_detection",
        "language_detection",
    ]

    display.start(total_models=len(categories))

    # Register all models
    for i, cat in enumerate(categories):
        device = "gpu" if i == 0 else "cpu"
        display.register_model(
            task_id=f"task_{i}",
            category_name=cat,
            model_name="xlm-roberta-base",
            device_type=device,
            device_id="mps" if device == "gpu" else f"cpu:{i}",
            total_epochs=5,
        )
        time.sleep(0.2)

    # Simulate training
    for epoch in range(1, 6):
        for i, cat in enumerate(categories):
            f1 = random.uniform(0.6, 0.95)
            display.update_epoch(
                task_id=f"task_{i}",
                epoch=epoch,
                train_loss=random.uniform(0.2, 0.8),
                val_loss=random.uniform(0.2, 0.6),
                f1_score=f1,
                accuracy=random.uniform(0.7, 0.95),
                phase="training" if epoch < 5 else "validation",
            )
            time.sleep(0.1)

        time.sleep(0.5)

    # Mark complete
    for i, cat in enumerate(categories):
        display.mark_complete(
            task_id=f"task_{i}",
            final_f1=random.uniform(0.75, 0.95),
            final_accuracy=random.uniform(0.8, 0.95),
        )
        time.sleep(0.3)

    display.stop()

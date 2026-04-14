#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
rich_progress_manager.py

MAIN OBJECTIVE:
---------------
Enhanced progress manager with rich UI components for llm_tool
Provides a single unified progress bar with dynamic status updates
and JSON sample display during pipeline execution.

Author:
-------
Antoine Lemor
"""

import os
import sys
import time
import threading
import json
import queue
from contextlib import suppress
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn
)
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich import box


@dataclass
class RichUIConfig:
    """Configuration values that control Rich's live rendering cadence."""
    profile: str
    disable_ui: bool
    refresh_hz: float
    min_render_interval: float
    min_progress_interval: float


def _env_flag(name: str) -> bool:
    """Interpret common truthy values from environment variables."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, allow_zero: bool = False) -> float:
    """Read a float environment variable with validation."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < 0 or (not allow_zero and value == 0):
        return default
    return value


def _get_rich_ui_config() -> RichUIConfig:
    """
    Determine how aggressively Rich should refresh the terminal.

    Profiles:
        - full: default experience (higher refresh rate)
        - balanced: moderate refresh for most terminals
        - safe: conservative refresh for Electron / VS Code
        - off/disabled: disable Rich entirely
    """
    if _env_flag("LLM_TOOL_FORCE_RICH_UI"):
        profile = "full"
    else:
        profile = os.environ.get("LLM_TOOL_RICH_PROFILE", "").strip().lower() or "full"

    disable_requested = _env_flag("LLM_TOOL_DISABLE_RICH_UI") or profile in {"off", "disabled", "none"}

    if disable_requested and not _env_flag("LLM_TOOL_FORCE_RICH_UI"):
        return RichUIConfig("off", True, 0.0, 0.0, 0.0)

    defaults = {
        "full": (12.0, 0.08, 0.01),
        "balanced": (8.0, 0.12, 0.02),
        "safe": (4.0, 0.25, 0.05),
    }

    if profile not in defaults:
        profile = "full"

    refresh_hz, min_render_interval, min_progress_interval = defaults[profile]

    refresh_hz = _env_float("LLM_TOOL_RICH_REFRESH_HZ", refresh_hz)
    min_render_interval = _env_float("LLM_TOOL_RICH_MIN_RENDER_INTERVAL", min_render_interval, allow_zero=True)
    min_progress_interval = _env_float("LLM_TOOL_RICH_MIN_PROGRESS_INTERVAL", min_progress_interval, allow_zero=True)

    # Fall back to safe profile when stdout is not a TTY unless user forced full mode
    if not sys.stdout.isatty() and profile == "full" and not _env_flag("LLM_TOOL_FORCE_RICH_UI"):
        profile = "safe"
        refresh_hz, min_render_interval, min_progress_interval = defaults["safe"]

    return RichUIConfig(profile, False, refresh_hz, min_render_interval, min_progress_interval)


def _should_disable_rich_ui() -> bool:
    """Utility wrapper to match historical API if other modules import it."""
    return _get_rich_ui_config().disable_ui or not sys.stdout.isatty()


class CompactPercentColumn(ProgressColumn):
    """Compact percentage display"""
    def render(self, task):
        percentage = task.percentage if task.percentage is not None else 0
        return Text(f"{percentage:>5.1f}%", style="bright_magenta")


class GlobalProgressTracker:
    """
    Global progress tracker for tracking total training progress across ALL models and epochs.

    This tracker provides a rich UI display showing:
    - Total number of models to train
    - Total number of epochs across all models
    - Current model being trained
    - Global progress (epochs completed / total epochs)
    - Elapsed time
    - Estimated time remaining

    Usage:
        # Initialize with total models and epochs per model
        tracker = GlobalProgressTracker(total_models=5, epochs_per_model=10)

        # Start tracking
        tracker.start()

        # Update when starting a new model
        tracker.start_model("bert-base-uncased", "Health", "EN")

        # Update after each epoch
        tracker.update_epoch(epoch=1, train_loss=0.5, val_loss=0.4, f1_score=0.85)

        # Finish current model
        tracker.finish_model()

        # Stop tracking
        tracker.stop()
    """

    def __init__(self, total_models: int, epochs_per_model: int, mode: str = "training"):
        """
        Initialize global progress tracker.

        Args:
            total_models: Total number of models to train
            epochs_per_model: Number of epochs per model (can be updated per model)
            mode: Training mode (e.g., "training", "benchmark", "multi-label")
        """
        self.ui_config = _get_rich_ui_config()
        self.disable_console_ui = self.ui_config.disable_ui
        self.console = Console(
            force_terminal=not self.disable_console_ui,
            no_color=self.disable_console_ui
        )
        self.total_models = total_models
        self.default_epochs_per_model = epochs_per_model
        self.mode = mode

        # Global progress tracking
        self.total_epochs = total_models * epochs_per_model
        self.completed_epochs = 0
        self.current_model_idx = 0

        # Current model tracking
        self.current_model_name = ""
        self.current_category = ""
        self.current_language = ""
        self.current_epochs = epochs_per_model
        self.current_epoch = 0

        # Metrics tracking
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0
        self.current_f1 = 0.0

        # Timing
        self.start_time = None
        self.current_model_start_time = None
        self.total_elapsed = 0.0
        self.estimated_remaining = 0.0

        # Progress display
        self.progress = None
        self.global_task_id = None
        self.model_task_id = None
        self.live = None
        self.is_running = False
        self._last_live_refresh = 0.0
        self._last_progress_refresh = 0.0

    def start(self):
        """Start the global progress tracker."""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()
        self._last_progress_refresh = self.start_time
        self._last_live_refresh = 0.0

        if self.disable_console_ui:
            self.console.print(
                "[yellow]Rich dashboard disabled for this session "
                "(non-interactive terminal detected).[/yellow]\n"
                "[yellow]Set `LLM_TOOL_FORCE_RICH_UI=1` to override.[/yellow]"
            )
            return

        # Create progress bars
        self.progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(
                bar_width=50,
                complete_style="bold green",
                finished_style="bold green",
                pulse_style="bold cyan"
            ),
            CompactPercentColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            console=self.console,
            expand=False,
            auto_refresh=False
        )

        # Add global task (all models, all epochs)
        mode_emoji = {
            "training": "🏋️",
            "benchmark": "⚡",
            "multi-label": "🎯",
            "multi-class": "🔢"
        }.get(self.mode, "🏋️")

        self.global_task_id = self.progress.add_task(
            f"{mode_emoji} TOTAL PROGRESS: 0/{self.total_models} models",
            total=self.total_epochs,
            completed=0
        )

        # Add current model task
        self.model_task_id = self.progress.add_task(
            "📊 Current Model: Waiting...",
            total=100,
            completed=0
        )

        # Start live display
        self.live = Live(
            self._create_panel(),
            console=self.console,
            refresh_per_second=self.ui_config.refresh_hz or 4,
            transient=False
        )
        self.live.start()
        self._last_live_refresh = time.time()

    def start_model(self, model_name: str, category: str = "", language: str = "", epochs: int = None):
        """
        Start tracking a new model.

        Args:
            model_name: Name of the model being trained
            category: Category/label being trained
            language: Language of the data
            epochs: Number of epochs for this model (if different from default)
        """
        if not self.is_running:
            return

        # Update model index
        self.current_model_idx += 1

        # Store current model info
        self.current_model_name = model_name
        self.current_category = category
        self.current_language = language
        self.current_epoch = 0

        # Update epochs for this model
        if epochs is not None:
            # Adjust total epochs if this model has different epoch count
            old_epochs = self.current_epochs
            self.current_epochs = epochs
            self.total_epochs = self.total_epochs - old_epochs + epochs

            # Update global task total
            self.progress.update(self.global_task_id, total=self.total_epochs)
        else:
            self.current_epochs = self.default_epochs_per_model

        # Reset metrics
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0
        self.current_f1 = 0.0

        # Reset model timer
        self.current_model_start_time = time.time()

        # Update display
        self._update_display()

    def update_epoch(self, epoch: int, train_loss: float = 0.0, val_loss: float = 0.0,
                     f1_score: float = 0.0, accuracy: float = 0.0):
        """
        Update progress after completing an epoch.

        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Training loss
            val_loss: Validation loss
            f1_score: F1 score
            accuracy: Accuracy score
        """
        if not self.is_running:
            return

        self.current_epoch = epoch
        self.current_train_loss = train_loss
        self.current_val_loss = val_loss
        self.current_f1 = f1_score

        # Increment completed epochs
        self.completed_epochs += 1

        # Calculate timing
        if self.start_time:
            self.total_elapsed = time.time() - self.start_time

            # Estimate remaining time based on average time per epoch
            if self.completed_epochs > 0:
                avg_time_per_epoch = self.total_elapsed / self.completed_epochs
                remaining_epochs = self.total_epochs - self.completed_epochs
                self.estimated_remaining = avg_time_per_epoch * remaining_epochs

        allow_refresh = True
        now = time.time()
        if self.ui_config.min_progress_interval > 0:
            if (now - self._last_progress_refresh) < self.ui_config.min_progress_interval:
                allow_refresh = False
            else:
                self._last_progress_refresh = now

        if self.progress:
            # Update progress bars
            self.progress.update(
                self.global_task_id,
                completed=self.completed_epochs,
                description=f"{'⚡' if self.mode == 'benchmark' else '🏋️'} TOTAL PROGRESS: {self.current_model_idx}/{self.total_models} models"
            )

            model_progress = (epoch / self.current_epochs) * 100
            self.progress.update(
                self.model_task_id,
                completed=model_progress,
                description=self._get_model_description()
            )

            # Update live display
            if allow_refresh:
                self._update_display()
        else:
            metric_bits = []
            if train_loss:
                metric_bits.append(f"train={train_loss:.4f}")
            if val_loss:
                metric_bits.append(f"val={val_loss:.4f}")
            if f1_score:
                metric_bits.append(f"F1={f1_score:.4f}")
            if accuracy:
                metric_bits.append(f"acc={accuracy:.4f}")

            metrics_text = f" ({', '.join(metric_bits)})" if metric_bits else ""
            model_name = self.current_model_name or f"Model {self.current_model_idx}"
            self.console.print(
                f"   • Epoch {epoch}/{self.current_epochs} complete for {model_name}{metrics_text}"
            )

    def finish_model(self):
        """Mark current model as finished."""
        if not self.is_running:
            return

        if self.progress:
            # Set model progress to 100%
            self.progress.update(
                self.model_task_id,
                completed=100,
                description=f"✅ {self.current_model_name}: Complete"
            )

            self._update_display(force=True)
        else:
            elapsed = 0.0
            if self.current_model_start_time:
                elapsed = time.time() - self.current_model_start_time
            self.console.print(
                f"[green]✔[/green] {self.current_model_name or 'Model'} "
                f"complete in {self._format_time(elapsed)}"
            )

    def stop(self):
        """Stop the global progress tracker."""
        if not self.is_running:
            return

        # Update final status if rich UI is active
        if self.progress and self.global_task_id is not None:
            self.progress.update(
                self.global_task_id,
                completed=self.total_epochs,
                description=f"✨ COMPLETE: {self.total_models} models trained"
            )
            self._update_display(force=True)
        elif self.disable_console_ui:
            self.console.print(
                f"[green]✔[/green] Training complete for {self.total_models} model(s)."
            )

        # Stop live display
        if self.live:
            time.sleep(0.5)
            self.live.stop()

        self.is_running = False

    def _get_model_description(self) -> str:
        """Get description for current model task."""
        parts = [f"📊 Model {self.current_model_idx}/{self.total_models}"]

        if self.current_model_name:
            parts.append(f"{self.current_model_name}")

        if self.current_category:
            parts.append(f"[{self.current_category}]")

        if self.current_language:
            parts.append(f"({self.current_language})")

        parts.append(f"Epoch {self.current_epoch}/{self.current_epochs}")

        return " ".join(parts)

    def _create_panel(self) -> Panel:
        """Create the panel with progress bars and stats."""
        # Create stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 1))
        stats_table.add_column(style="cyan", width=20)
        stats_table.add_column(style="white")

        # Add statistics
        stats_table.add_row("📈 Total Models:", f"{self.total_models}")
        stats_table.add_row("🔢 Total Epochs:", f"{self.total_epochs}")
        stats_table.add_row("✅ Completed Epochs:", f"{self.completed_epochs}/{self.total_epochs}")

        if self.completed_epochs > 0:
            completion_pct = (self.completed_epochs / self.total_epochs) * 100
            stats_table.add_row("📊 Overall Progress:", f"{completion_pct:.1f}%")

        # Timing info
        if self.start_time:
            elapsed = time.time() - self.start_time
            stats_table.add_row("⏱️ Elapsed Time:", self._format_time(elapsed))

            if self.estimated_remaining > 0:
                stats_table.add_row("⏳ Est. Remaining:", self._format_time(self.estimated_remaining))

        # Current model metrics (if available)
        if self.current_model_name and self.current_epoch > 0:
            stats_table.add_row("", "")  # Spacer
            stats_table.add_row("🎯 Current Model:", self.current_model_name)
            if self.current_category:
                stats_table.add_row("  Category:", self.current_category)
            if self.current_language:
                stats_table.add_row("  Language:", self.current_language)
            stats_table.add_row("  Epoch:", f"{self.current_epoch}/{self.current_epochs}")
            if self.current_f1 > 0:
                stats_table.add_row("  F1 Score:", f"{self.current_f1:.4f}")
            if self.current_train_loss > 0:
                stats_table.add_row("  Train Loss:", f"{self.current_train_loss:.4f}")
            if self.current_val_loss > 0:
                stats_table.add_row("  Val Loss:", f"{self.current_val_loss:.4f}")

        # Combine progress bars and stats
        from rich.console import Group
        group = Group(
            self.progress,
            Text(),  # Spacer
            stats_table
        )

        # Different colors based on mode
        border_colors = {
            "training": "bold blue",
            "benchmark": "bold yellow",
            "multi-label": "bold magenta",
            "multi-class": "bold cyan"
        }

        titles = {
            "training": "🏋️ TRAINING PROGRESS",
            "benchmark": "⚡ BENCHMARK PROGRESS",
            "multi-label": "🎯 MULTI-LABEL TRAINING PROGRESS",
            "multi-class": "🔢 MULTI-CLASS TRAINING PROGRESS"
        }

        return Panel(
            group,
            title=titles.get(self.mode, "🏋️ TRAINING PROGRESS"),
            border_style=border_colors.get(self.mode, "bold blue"),
            box=box.HEAVY
        )

    def _update_display(self, force: bool = False):
        """Update the live display."""
        if not self.live or not self.is_running:
            return

        now = time.time()
        if not force:
            min_interval = self.ui_config.min_render_interval
            if min_interval > 0 and (now - self._last_live_refresh) < min_interval:
                return

        self.live.update(self._create_panel(), refresh=True)
        self._last_live_refresh = now

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


@dataclass
class ProgressState:
    """Track overall progress state"""
    current_phase: str = ""
    current_progress: float = 0.0
    current_message: str = ""
    completed_items: int = 0
    total_items: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    json_count: int = 0
    last_json_sample: Optional[Dict] = None
    start_time: float = field(default_factory=time.time)
    annotation_samples: List[Dict] = field(default_factory=list)


class RichProgressManager:
    """Enhanced progress manager with rich UI components"""

    # Phase definitions with emojis
    PHASES = {
        'initialization': {'icon': '⚙️', 'label': 'Initialization'},
        'annotation': {'icon': '✍️', 'label': 'Annotation'},
        'validation': {'icon': '✅', 'label': 'Validation'},
        'training': {'icon': '🏋️', 'label': 'Training'},
        'evaluation': {'icon': '📊', 'label': 'Evaluation'},
        'deployment': {'icon': '🚀', 'label': 'Deployment'}
    }

    def __init__(self, show_json_every: int = 10, compact_mode: bool = True):
        """Initialize with configuration"""
        self.ui_config = _get_rich_ui_config()
        self.disable_console_ui = self.ui_config.disable_ui
        self.console: Optional[Console] = None
        self.state = ProgressState()
        self._state_lock = threading.Lock()
        self.show_json_every = show_json_every
        self.compact_mode = compact_mode

        # Progress tracking - SINGLE TASK ONLY
        self.progress: Optional[Progress] = None
        self.overall_task_id: Optional[str] = None
        self.subtask_task_id: Optional[str] = None
        self.live: Optional[Live] = None

        self.is_running = False
        self._paused = False
        self._last_progress_refresh = 0.0
        self._last_preview_refresh = 0.0

        # Sample tracking
        self.current_sample = None
        self.recent_errors: List[str] = []
        self.recent_warnings: List[str] = []
        self.last_json_display = 0

        # Rendering infrastructure (initialised in start)
        self._event_queue: queue.Queue = queue.Queue(maxsize=4096)
        self._render_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._queue_overflow_warned = False

    def _create_panel(self) -> Panel:
        """Compose the Rich dashboard panel summarising current progress state."""
        with self._state_lock:
            phase = self.state.current_phase or "progress"
            progress_pct = max(0.0, min(self.state.current_progress, 100.0))
            message = self.state.current_message or ""
            completed = self.state.completed_items
            total = self.state.total_items
            error_count = self.state.error_count
            warning_count = len(self.state.warnings)
            elapsed = time.time() - self.state.start_time if self.state.start_time else 0.0

        phase_meta = self.PHASES.get(phase, {'icon': '•', 'label': phase.title()})
        header = Text(
            f"{phase_meta['icon']} {phase_meta['label']}: {message or 'Processing…'}",
            style="bold cyan"
        )

        stats_table = Table.grid(padding=(0, 1))
        stats_table.add_column(style="dim", justify="right")
        stats_table.add_column()
        stats_table.add_row("Progress:", f"{progress_pct:6.1f}%")
        if total:
            stats_table.add_row("Items:", f"{completed}/{total}")
        stats_table.add_row("Errors:", str(error_count))
        stats_table.add_row("Warnings:", str(warning_count))
        if elapsed:
            stats_table.add_row("Elapsed:", self._format_duration(elapsed))

        alerts_panel: Optional[Panel] = None
        if self.recent_errors or self.recent_warnings:
            alerts_table = Table.grid(padding=(0, 1))
            alerts_table.add_column(width=2, justify="center")
            alerts_table.add_column()
            for warn in self.recent_warnings[-3:]:
                alerts_table.add_row("[yellow]⚠[/yellow]", warn)
            for err in self.recent_errors[-3:]:
                alerts_table.add_row("[red]✖[/red]", err)
            alerts_panel = Panel(
                alerts_table,
                title="[bold yellow]Alerts[/bold yellow]",
                border_style="yellow",
                expand=False
            )

        from rich.console import Group
        renderables: List[Any] = [header]
        if self.progress is not None:
            renderables.append(self.progress)
        renderables.append(Text())
        renderables.append(stats_table)
        if alerts_panel is not None:
            renderables.append(Text())
            renderables.append(alerts_panel)

        return Panel(
            Group(*renderables),
            title="📊 Annotation Progress",
            border_style="bright_cyan",
            box=box.HEAVY
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format elapsed seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            remaining = int(seconds % 60)
            return f"{minutes}m {remaining}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    def start(self):
        """Start the progress display"""
        if self.is_running:
            return

        self.is_running = True
        self.state.start_time = time.time()
        self._last_progress_refresh = self.state.start_time
        self._queue_overflow_warned = False

        self._event_queue = queue.Queue(maxsize=4096)
        self._loop_ready.clear()

        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="LLMToolRichRenderer",
            daemon=True
        )
        self._render_thread.start()
        self._loop_ready.wait()

    def _render_loop(self):
        try:
            self._initialize_render_objects()
            self._loop_ready.set()
            while True:
                func, args, kwargs = self._event_queue.get()
                if func is None:
                    self._event_queue.task_done()
                    break
                try:
                    func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - defensive
                    target = self.console
                    if target is not None:
                        with suppress(Exception):
                            target.print(f"[red]Rich UI error: {exc}[/red]")
                finally:
                    self._event_queue.task_done()
        finally:
            self._teardown_render_objects()
            self.is_running = False
            self._loop_ready.set()

    def _initialize_render_objects(self):
        self.console = Console(
            force_terminal=not self.disable_console_ui,
            no_color=self.disable_console_ui
        )

        if self.disable_console_ui:
            with suppress(Exception):
                self.console.print(
                    "[yellow]Rich dashboard running in buffered mode (VS Code detected).[/yellow]"
                )
            return

        self.progress = Progress(
            SpinnerColumn(style="bright_cyan", spinner_name="dots"),
            TextColumn("[bold]{task.description}"),
            BarColumn(
                bar_width=50,
                complete_style="bright_green",
                finished_style="green",
                pulse_style="cyan"
            ),
            CompactPercentColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            console=self.console,
            expand=False,
            transient=False,
            auto_refresh=False
        )
        self.progress.start()
        self.overall_task_id = None
        self.subtask_task_id = None

        self.live = Live(
            self._create_panel(),
            console=self.console,
            refresh_per_second=self.ui_config.refresh_hz or 4,
            transient=False
        )
        self.live.start()
        self._last_live_refresh = time.time()

    def _teardown_render_objects(self):
        if self.live is not None:
            with suppress(Exception):
                self.live.stop()
        if self.progress is not None:
            with suppress(Exception):
                self.progress.stop()
        self.live = None
        self.progress = None
        self.console = None
        self.overall_task_id = None
        self.subtask_task_id = None

    def _call_on_ui_thread(self, func, *args, **kwargs):
        if not self.is_running:
            return
        if threading.current_thread() is self._render_thread:
            func(*args, **kwargs)
            return
        try:
            self._event_queue.put_nowait((func, args, kwargs))
        except queue.Full:
            if not self._queue_overflow_warned:
                self._queue_overflow_warned = True
                message = "[red]Rich UI queue full – dropping updates.[/red]"
                target = self.console
                if target is not None:
                    with suppress(Exception):
                        target.print(message)
                else:
                    print(message)

    def _refresh_progress(self, force: bool = False):
        """Refresh the rich progress bar with throttling."""
        if not self.progress:
            return

        now = time.time()
        if not force and self.ui_config.min_render_interval > 0:
            if (now - self._last_progress_refresh) < self.ui_config.min_render_interval:
                return

        try:
            self.progress.refresh()
        except Exception:
            # Avoid crashing the pipeline if the terminal rejects refreshes
            return

        self._last_progress_refresh = now

    def _log_plain_update(self, phase: str, progress: float, message: str, error: Optional[str] = None):
        """Fallback textual logging when Rich UI is disabled."""
        phase_info = self.PHASES.get(phase, {})
        icon = phase_info.get('icon', '•')
        label = phase_info.get('label', phase.title())
        msg = f"{icon} {label}: {progress:.1f}% - {message}"
        self.console.print(msg)
        if error:
            self.console.print(f"[red]⚠ {error}[/red]")

    def pause_for_training(self):
        """Pause progress display for training phase"""
        if not self.is_running:
            return

        def _do_pause():
            if not self.progress:
                return
            try:
                self.progress.update(
                    self.overall_task_id,
                    description="⏸ Pausing for training..."
                )
                self._refresh_progress(force=True)
                time.sleep(0.5)
                self.progress.stop()
                self.progress = None
                target_console = self.console or Console()
                with suppress(Exception):
                    target_console.print("\n[dim cyan]━━━ Progress paused for training phase ━━━[/dim cyan]\n")
                self._paused = True
            except Exception as exc:  # pragma: no cover - defensive
                with suppress(Exception):
                    (self.console or Console()).print(f"[yellow]Warning: Could not pause progress cleanly: {exc}[/yellow]")

        self._call_on_ui_thread(_do_pause)

    def resume_after_training(self):
        """Resume progress display after training"""
        if not (self.is_running and hasattr(self, '_paused')):
            return

        def _do_resume():
            target_console = self.console or Console()
            try:
                with suppress(Exception):
                    target_console.print("[dim cyan]━━━ Resuming progress display ━━━[/dim cyan]\n")

                if self.progress is not None:
                    return

                self.progress = Progress(
                    SpinnerColumn(style="bright_cyan", spinner_name="dots"),
                    TextColumn("[bold]{task.description}"),
                    BarColumn(
                        bar_width=50,
                        complete_style="bright_green",
                        finished_style="green",
                        pulse_style="cyan"
                    ),
                    CompactPercentColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(compact=True, elapsed_when_finished=True),
                    console=self.console,
                    expand=False,
                    transient=False,
                    auto_refresh=False
                )

                self.progress.start()
                self.overall_task_id = self.progress.add_task(
                    "🚀 Reprise",
                    total=100,
                    completed=max(0.0, min(self.state.current_progress, 100.0))
                )
                self.progress.update(
                    self.overall_task_id,
                    completed=max(0.0, min(self.state.current_progress, 100.0)),
                    description=f"{self.state.current_phase or 'Progress'}: {self.state.current_message}"
                )
                self._refresh_progress(force=True)
                del self._paused
            except Exception as exc:  # pragma: no cover - defensive
                with suppress(Exception):
                    target_console.print(f"[red]Warning: Could not resume progress: {exc}[/red]")

        self._call_on_ui_thread(_do_resume)

    def stop(self):
        """Stop the display"""
        if not self.is_running:
            return

        def _finalize():
            if self.progress and self.overall_task_id is not None:
                self.progress.update(
                    self.overall_task_id,
                    completed=100,
                    description="✨ Pipeline Complete"
                )
                self._refresh_progress(force=True)
                time.sleep(0.2)

        self._call_on_ui_thread(_finalize)

        try:
            self._event_queue.put_nowait((None, (), {}))
        except queue.Full:  # pragma: no cover - defensive
            self._event_queue.put((None, (), {}))

        if self._render_thread:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None

        self.is_running = False
        self.progress = None
        self.live = None
        self.overall_task_id = None
        self.subtask_task_id = None

    def update_progress(self, phase: str, progress: float, message: str,
                       subtask: Optional[Dict[str, Any]] = None,
                       json_sample: Optional[Dict] = None,
                       error: Optional[str] = None):
        """Update progress with unified display"""
        if not self.is_running:
            return

        preview_payload: Optional[Dict] = None
        ui_active = False
        desc = ""
        clamped_progress = max(0.0, min(progress, 100.0))

        with self._state_lock:
            # Snapshot UI availability while holding state lock to avoid races
            ui_active = (not self.disable_console_ui) and (self.progress is not None)

            # Update state
            self.state.current_phase = phase
            self.state.current_progress = clamped_progress
            self.state.current_message = message

            phase_info = self.PHASES.get(phase, {})
            icon = phase_info.get('icon', '•')
            label = phase_info.get('label', phase.title())
            desc = f"{icon} {label}: {message}"

            sample_data = json_sample
            current = 0
            total = 0

            if phase == 'annotation' and subtask:
                current = int(subtask.get('current', 0))
                total = int(subtask.get('total', 100))
                self.state.completed_items = current
                self.state.total_items = total

                if current > 0:
                    desc = f"{icon} {label}: {message} [{current}/{total}]"

                sample_data = subtask.get('json_data', json_sample)
                if sample_data:
                    self.current_sample = sample_data
                    should_emit_preview = (
                        current > 0
                        and self.show_json_every > 0
                        and current % self.show_json_every == 0
                        and (current - self.last_json_display) >= self.show_json_every
                    )
                    if should_emit_preview:
                        self.state.json_count += 1
                        self.state.last_json_sample = sample_data
                        self.last_json_display = current
                        preview_payload = sample_data

            if self.state.error_count > 0:
                desc += f" [red]({self.state.error_count} errors)[/red]"

            if error:
                self.state.errors.append(error)
                self.state.error_count += 1
                self.recent_errors.append(error)
                if len(self.recent_errors) > 5:
                    self.recent_errors.pop(0)

        if ui_active:
            self._call_on_ui_thread(
                self._ui_update_progress,
                desc,
                clamped_progress,
                preview_payload
            )
        else:
            self._call_on_ui_thread(
                self._ui_plain_progress,
                phase,
                clamped_progress,
                message,
                error,
                preview_payload
            )

    def _ui_update_progress(self, desc: str, progress: float, preview_payload: Optional[Dict]):
        if not self.progress:
            return

        if self.overall_task_id is None:
            self.overall_task_id = self.progress.add_task(
                desc,
                total=100,
                completed=0
            )

        self.progress.update(
            self.overall_task_id,
            completed=max(0.0, min(progress, 100.0)),
            description=desc
        )
        self._refresh_progress()

        if preview_payload:
            self._show_json_panel(preview_payload)

    def _ui_plain_progress(self, phase: str, progress: float, message: str,
                           error: Optional[str], preview_payload: Optional[Dict]):
        self._log_plain_update(phase, progress, message, error)

        if not preview_payload:
            return

        try:
            preview = json.dumps(preview_payload, indent=2, ensure_ascii=False)
            if len(preview) > 400:
                preview = preview[:400] + "\n  ..."
            target_console = self.console or Console()
            target_console.print(preview)
        except Exception:
            pass

    def _show_json_panel(self, json_data: Dict):
        """Display or refresh the JSON preview panel"""
        self.state.last_json_sample = json_data

        try:
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            lines = json_str.split('\n')
            panel_width = 80
            if lines:
                longest_line = max(len(line) for line in lines)
                panel_width = min(max(60, longest_line + 4), 140)

            syntax = Syntax(
                json_str,
                "json",
                theme="monokai",
                line_numbers=False,
                background_color="default"
            )

            panel = Panel(
                syntax,
                title=f"[bold cyan]📝 Preview #{self.state.json_count}[/bold cyan]",
                border_style="cyan",
                expand=False,
                width=panel_width
            )

            target_console = self.progress.console if self.progress is not None else self.console
            if target_console is None:
                target_console = Console()

            if self.disable_console_ui:
                target_console.print(panel)
                return

            refresh_rate = self.ui_config.refresh_hz or 4
            now = time.time()

            if not hasattr(self, '_preview_live'):
                from rich.live import Live
                self._preview_live = Live(panel, console=target_console, refresh_per_second=refresh_rate)
                self._preview_live.start()
                self._last_preview_refresh = now
            else:
                min_interval = self.ui_config.min_render_interval
                if min_interval <= 0 or (now - self._last_preview_refresh) >= min_interval:
                    self._preview_live.update(panel, refresh=True)
                    self._last_preview_refresh = now

        except Exception:
            pass  # Silently fail to avoid disrupting progress
        finally:
            if hasattr(self, '_preview_live') and getattr(self._preview_live, 'stopped', False):
                # Live may auto-stop if console errors; reset handler
                del self._preview_live

    def _print_alert_summary(self):
        """Render a consolidated warning/error panel"""
        has_warnings = bool(self.recent_warnings)
        has_errors = bool(self.recent_errors)

        if not has_warnings and not has_errors:
            body = Text("No warnings or errors", style="dim")
        else:
            table = Table.grid(padding=(0, 1))
            table.add_column(justify="left", width=2)
            table.add_column()

            for msg in self.recent_warnings[-3:]:
                table.add_row("[yellow]⚠[/yellow]", msg)

            for msg in self.recent_errors[-3:]:
                table.add_row("[red]✖[/red]", msg)

            body = table

        panel = Panel(
            body,
            title="[bold yellow]Alerts[/bold yellow]",
            border_style="yellow",
            expand=False,
            width=80
        )

        target_console = self.progress.console if self.progress is not None else self.console
        if target_console is None:
            target_console = Console()
        target_console.print(panel)

    def clear_subtask(self, subtask_name: str):
        """Clear subtask (no-op since we don't have separate subtask)"""
        pass

    def show_error(self, error: str, item_info: Optional[str] = None):
        """Display error message in a panel without disrupting progress bars"""
        error_msg = error
        if item_info:
            error_msg = f"{item_info}: {error}"

        with self._state_lock:
            self.state.errors.append(error_msg)
            self.state.error_count += 1
            self.recent_errors.append(error_msg)
            if len(self.recent_errors) > 5:
                self.recent_errors.pop(0)

        if self.is_running:
            self._call_on_ui_thread(self._print_alert_summary)
        else:  # pragma: no cover - fallback for shutdown edges
            with suppress(Exception):
                self._print_alert_summary()

    def show_warning(self, warning: str, item_info: Optional[str] = None):
        """Display warning message in a panel without disrupting progress bars"""
        warning_msg = warning
        if item_info:
            warning_msg = f"{item_info}: {warning}"

        with self._state_lock:
            self.state.warnings.append(warning_msg)
            self.recent_warnings.append(warning_msg)
            if len(self.recent_warnings) > 5:
                self.recent_warnings.pop(0)

        if self.is_running:
            self._call_on_ui_thread(self._print_alert_summary)
        else:  # pragma: no cover - fallback for shutdown edges
            with suppress(Exception):
                self._print_alert_summary()

    def show_final_stats(self):
        """Show final statistics with samples"""
        with self._state_lock:
            if not self.state.completed_items:
                return
            snapshot = {
                "completed_items": self.state.completed_items,
                "json_count": self.state.json_count,
                "error_count": self.state.error_count,
                "warning_count": len(self.state.warnings),
                "elapsed": time.time() - self.state.start_time,
                "samples": list(self.state.annotation_samples[-3:]),
            }

        if self.is_running:
            self._call_on_ui_thread(self._ui_render_final_stats, snapshot)
        else:  # pragma: no cover - fallback if called post-stop
            self._ui_render_final_stats(snapshot)

    def _ui_render_final_stats(self, snapshot: Dict[str, Any]):
        target_console = self.console or Console()
        target_console.print()

        stats = Table.grid(padding=1)
        stats.add_column(style="dim")
        stats.add_column(style="bright_white")
        stats.add_row("Items processed:", str(snapshot["completed_items"]))
        stats.add_row("JSON samples:", str(snapshot["json_count"]))
        if snapshot["error_count"] > 0:
            stats.add_row("Errors:", f"[red]{snapshot['error_count']}[/red]")
        if snapshot["warning_count"] > 0:
            stats.add_row("Warnings:", f"[yellow]{snapshot['warning_count']}[/yellow]")
        stats.add_row("Total time:", f"{snapshot['elapsed']:.1f}s")

        target_console.print(
            Panel(
                stats,
                title="[bold yellow]📊 Pipeline Summary[/bold yellow]",
                border_style="yellow",
                expand=False
            )
        )

        samples = snapshot.get("samples") or []
        if not samples:
            return

        target_console.print("\n[bold cyan]📝 Sample Annotations:[/bold cyan]")
        for i, sample in enumerate(samples, 1):
            try:
                json_str = json.dumps(sample, indent=2, ensure_ascii=False)
            except Exception:
                json_str = str(sample)

            lines = json_str.split('\n')
            if len(lines) > 6:
                json_str = '\n'.join(lines[:5]) + '\n  ...\n}'

            syntax = Syntax(
                json_str,
                "json",
                theme="monokai",
                line_numbers=False,
                background_color="default"
            )

            target_console.print(
                Panel(
                    syntax,
                    title=f"Sample {i}",
                    border_style="dim cyan",
                    expand=False,
                    width=60
                )
            )

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.is_running:
            if not exc_type:
                self.show_final_stats()
            self.stop()

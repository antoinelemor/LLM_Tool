#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
parallel_training_manager.py

MAIN OBJECTIVE:
---------------
Complete parallel training manager with:
1) Thread-safe logging (CSV, JSON, metrics)
2) Unified Rich display for all models with per-class and per-language metrics
3) Real-time chart generation for all parallel models
4) Full integration with existing package infrastructure
5) Memory-aware resource allocation

FEATURES:
---------
1) ThreadSafeMetricsLogger - Handles concurrent writes to CSV/JSON
2) ParallelTrainingDisplay - Rich dashboard showing all models with detailed metrics
3) ParallelChartManager - Generates/updates charts for all models
4) ParallelTrainingManager - Orchestrates everything with hybrid GPU+CPU training
5) ResourcePlan - Intelligent allocation of GPU/CPU resources based on system capacity

Dependencies:
-------------
- os
- gc
- csv
- json
- time
- threading
- dataclasses
- pathlib
- typing
- psutil (optional)
- rich
- torch
- llm_tool.trainers.hybrid_parallel_trainer
- llm_tool.trainers.training_metrics_chart
- llm_tool.utils.system_resources
- llm_tool.utils.logging_utils

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import csv
import os
import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_tool.utils.training_paths import (
    get_training_logs_base,
    get_session_dir,
    get_training_metrics_dir,
)

from rich.console import Console, Control, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    ProgressColumn,
    BarColumn,
    TextColumn,
    TaskID,
    SpinnerColumn,
)
from rich.table import Table
from rich.text import Text
from rich.markup import escape
from rich import box

logger = logging.getLogger(__name__)


# ============================================================================
# TERMINAL MEMORY MANAGEMENT
# ============================================================================

# Default refresh interval for parallel display (slower for long sessions)
DEFAULT_PARALLEL_REFRESH_INTERVAL = 1.0  # 1 second minimum between refreshes

# Terminal buffer clearing interval (every 2 minutes to prevent memory buildup)
DEFAULT_TERMINAL_CLEAR_INTERVAL = 120.0

# Maximum updates before forcing a buffer clear
DEFAULT_TERMINAL_CLEAR_UPDATES = 120  # At 1Hz, this is ~2 minutes

# Environment variable overrides
PARALLEL_REFRESH_INTERVAL = float(
    os.environ.get("LLM_TOOL_PARALLEL_REFRESH", DEFAULT_PARALLEL_REFRESH_INTERVAL)
)
TERMINAL_CLEAR_INTERVAL = float(
    os.environ.get("LLM_TOOL_TERMINAL_CLEAR", DEFAULT_TERMINAL_CLEAR_INTERVAL)
)
TERMINAL_CLEAR_UPDATES = int(
    os.environ.get("LLM_TOOL_TERMINAL_CLEAR_UPDATES", DEFAULT_TERMINAL_CLEAR_UPDATES)
)


def _clear_terminal_buffer(console_obj) -> bool:
    """
    Best-effort clearing of the active terminal surface and scrollback.

    This prevents memory buildup in VS Code terminals and other terminal emulators
    during long training sessions.

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

# Import scheduler config for intelligent task scheduling
try:
    from llm_tool.trainers.task_scheduler import SchedulerConfig
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False
    SchedulerConfig = None
    logger.debug("task_scheduler not available - using legacy FIFO scheduling")

# Try to import psutil for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.debug("psutil not available - resource monitoring disabled")


# ============================================================================
# THREAD-SAFE METRICS LOGGING
# ============================================================================

class ThreadSafeMetricsLogger:
    """
    Thread-safe logger for training metrics.

    Handles concurrent writes to:
    - training.csv: Epoch-by-epoch metrics for all models
    - best.csv: Best results per model
    - metrics.json: Full metrics history

    Uses a dedicated writer thread to ensure consistent file writes.
    """

    def __init__(self, output_dir: str, session_id: str):
        """
        Initialize the metrics logger.

        Parameters
        ----------
        output_dir : str
            Directory for output files
        session_id : str
            Session identifier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id

        # File paths
        self.training_csv = self.output_dir / "training.csv"
        self.best_csv = self.output_dir / "best.csv"
        self.metrics_json = self.output_dir / "metrics.json"

        # Thread-safe queue for write operations
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False

        # In-memory metrics storage (for JSON export)
        self._metrics_store: Dict[str, Dict] = {}
        self._metrics_lock = threading.RLock()

        # CSV headers - NEW: Aligned with rationalized logging schema
        self._training_headers = [
            'timestamp', 'session_id',
            'huggingface_model',     # NEW: HuggingFace model identifier
            'model_class',           # NEW: Python class name
            'training_mode',         # NEW: multi-class, multi-label, one-vs-all, binary
            'category', 'language', 'device',
            'epoch', 'train_loss', 'val_loss',
            'f1_macro', 'f1_micro', 'accuracy',
            'learning_rate', 'samples_per_second',
            'phase'
        ]
        self._best_headers = [
            'timestamp', 'session_id',
            'huggingface_model',     # NEW: HuggingFace model identifier
            'model_class',           # NEW: Python class name
            'training_mode',         # NEW: multi-class, multi-label, one-vs-all, binary
            'category', 'language', 'device',
            'best_epoch', 'best_f1_macro', 'best_f1_micro', 'best_accuracy',
            'training_time', 'total_epochs', 'final_train_loss', 'final_val_loss',
            # NEW: Hyperparameters for reproducibility
            'learning_rate', 'batch_size', 'warmup_ratio',
            # NEW: Multi-label metrics (if applicable)
            'hamming_loss', 'subset_accuracy', 'jaccard_score', 'multi_label_threshold',
        ]

        # Initialize CSV files
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        if not self.training_csv.exists():
            with open(self.training_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._training_headers)
                writer.writeheader()

        if not self.best_csv.exists():
            with open(self.best_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._best_headers)
                writer.writeheader()

    def start(self):
        """Start the writer thread."""
        if self._running:
            return

        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="MetricsWriter",
            daemon=True,
        )
        self._writer_thread.start()

    def stop(self):
        """Stop the writer thread and flush remaining writes."""
        if not self._running:
            return

        self._running = False
        # Signal thread to stop
        self._write_queue.put(None)

        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)

        # Final JSON export
        self._export_metrics_json()

    def _writer_loop(self):
        """Writer thread main loop."""
        while True:
            try:
                item = self._write_queue.get(timeout=0.5)

                if item is None:
                    # Stop signal
                    break

                op_type, data = item

                if op_type == "training":
                    self._write_training_row(data)
                elif op_type == "best":
                    self._write_best_row(data)
                elif op_type == "metrics":
                    self._update_metrics_store(data)

                self._write_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in metrics writer: {e}")

    def _write_training_row(self, data: Dict):
        """Write a row to training.csv."""
        try:
            with open(self.training_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._training_headers)
                # Only write fields that exist in headers
                row = {k: data.get(k, '') for k in self._training_headers}
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing to training.csv: {e}")

    def _write_best_row(self, data: Dict):
        """Write a row to best.csv."""
        try:
            with open(self.best_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._best_headers)
                row = {k: data.get(k, '') for k in self._best_headers}
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing to best.csv: {e}")

    def _update_metrics_store(self, data: Dict):
        """Update in-memory metrics store."""
        with self._metrics_lock:
            # Use huggingface_model as primary key, fallback to model_id for legacy
            model_id = data.get('huggingface_model') or data.get('model_id', 'unknown')
            if model_id not in self._metrics_store:
                self._metrics_store[model_id] = {
                    # NEW: Extended metadata
                    'huggingface_model': data.get('huggingface_model', ''),
                    'model_class': data.get('model_class', ''),
                    'training_mode': data.get('training_mode', ''),
                    'category': data.get('category', ''),
                    'language': data.get('language', ''),
                    'device': data.get('device', ''),
                    # Metrics arrays
                    'epochs': [],
                    'train_losses': [],
                    'val_losses': [],
                    'f1_macros': [],
                    'f1_micros': [],
                    'accuracies': [],
                    'best_epoch': None,
                    'best_f1': 0,
                    'best_combined': 0,
                }

            store = self._metrics_store[model_id]
            store['epochs'].append(data.get('epoch', 0))
            store['train_losses'].append(data.get('train_loss', 0))
            store['val_losses'].append(data.get('val_loss', 0))
            store['f1_macros'].append(data.get('f1_macro', 0))
            store['f1_micros'].append(data.get('f1_micro', 0))
            store['accuracies'].append(data.get('accuracy', 0))

            # Track best using combined_score (fallback to f1_macro if not available)
            combined = data.get('combined_score', data.get('f1_macro', 0))
            if combined > store.get('best_combined', 0):
                store['best_combined'] = combined
                store['best_f1'] = data.get('f1_macro', 0)
                store['best_epoch'] = data.get('epoch', 0)

    def _export_metrics_json(self):
        """Export all metrics to JSON file."""
        with self._metrics_lock:
            export_data = {
                'session_id': self.session_id,
                'export_timestamp': datetime.now().isoformat(),
                'models': self._metrics_store,
            }

            try:
                with open(self.metrics_json, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                logger.error(f"Error exporting metrics JSON: {e}")

    def generate_session_manifest(
        self,
        training_mode: str = "parallel",
        training_approach: str = "multi-class",
        multi_label: bool = False,
        hyperparameters: Optional[Dict] = None,
        device: str = "cpu",
    ) -> Path:
        """
        Generate a session manifest summarizing the entire training session.

        Parameters
        ----------
        training_mode : str
            Training mode (parallel, normal, benchmark)
        training_approach : str
            Training approach (multi-class, multi-label, one-vs-all)
        multi_label : bool
            Whether multi-label classification was used
        hyperparameters : dict, optional
            Default hyperparameters used
        device : str
            Primary device used

        Returns
        -------
        Path
            Path to the generated manifest file
        """
        import platform
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            torch_version = "unknown"

        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "unknown"

        with self._metrics_lock:
            categories_trained = []
            for model_id, store in self._metrics_store.items():
                categories_trained.append({
                    'name': store.get('category', model_id),
                    'huggingface_model': store.get('huggingface_model', model_id),
                    'model_class': store.get('model_class', ''),
                    'training_mode': store.get('training_mode', training_approach),
                    'language': store.get('language', ''),
                    'best_epoch': store.get('best_epoch', 0),
                    'best_macro_f1': store.get('best_f1', 0),
                    'num_epochs': len(store.get('epochs', [])),
                })

        manifest = {
            'schema_version': '2.0',
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'status': 'completed',
            'environment': {
                'python_version': platform.python_version(),
                'torch_version': torch_version,
                'transformers_version': transformers_version,
                'device': device,
            },
            'training_config': {
                'mode': training_mode,
                'approach': training_approach,
                'multi_label': multi_label,
            },
            'default_hyperparameters': hyperparameters or {},
            'categories_trained': categories_trained,
            'files_generated': {
                'training_csv': str(self.training_csv),
                'best_csv': str(self.best_csv),
                'metrics_json': str(self.metrics_json),
            },
        }

        manifest_path = self.output_dir / 'session_manifest.json'
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info(f"Session manifest saved to {manifest_path}")
        except Exception as e:
            logger.error(f"Error saving session manifest: {e}")

        return manifest_path

    def log_epoch(
        self,
        model_id: str,
        category: str,
        device: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        f1_macro: float,
        f1_micro: float = 0,
        accuracy: float = 0,
        learning_rate: float = 0,
        phase: str = "training",
        # NEW: Extended metadata for rationalized logging
        huggingface_model: str = "",
        model_class: str = "",
        training_mode: str = "",
        language: str = "",
        samples_per_second: float = 0,
    ):
        """Log metrics for an epoch."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            # NEW: Standardized model identification
            'huggingface_model': huggingface_model or model_id,
            'model_class': model_class,
            'training_mode': training_mode,
            'category': category,
            'language': language,
            'device': device,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy': accuracy,
            'learning_rate': learning_rate,
            'samples_per_second': samples_per_second,
            'phase': phase,
        }

        self._write_queue.put(("training", data))
        self._write_queue.put(("metrics", data))

    def log_best(
        self,
        model_id: str,
        category: str,
        device: str,
        best_epoch: int,
        best_f1_macro: float,
        best_accuracy: float,
        training_time: float,
        total_epochs: int,
        final_train_loss: float,
        final_val_loss: float,
        # NEW: Extended metadata for rationalized logging
        huggingface_model: str = "",
        model_class: str = "",
        training_mode: str = "",
        language: str = "",
        best_f1_micro: float = 0,
        learning_rate: float = 0,
        batch_size: int = 0,
        warmup_ratio: float = 0,
        # Multi-label specific metrics
        hamming_loss: float = 0,
        subset_accuracy: float = 0,
        jaccard_score: float = 0,
        multi_label_threshold: float = 0.5,
    ):
        """Log best results for a model."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            # NEW: Standardized model identification
            'huggingface_model': huggingface_model or model_id,
            'model_class': model_class,
            'training_mode': training_mode,
            'category': category,
            'language': language,
            'device': device,
            'best_epoch': best_epoch,
            'best_f1_macro': best_f1_macro,
            'best_f1_micro': best_f1_micro,
            'best_accuracy': best_accuracy,
            'training_time': training_time,
            'total_epochs': total_epochs,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            # NEW: Hyperparameters for reproducibility
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'warmup_ratio': warmup_ratio,
            # Multi-label metrics
            'hamming_loss': hamming_loss,
            'subset_accuracy': subset_accuracy,
            'jaccard_score': jaccard_score,
            'multi_label_threshold': multi_label_threshold,
        }

        self._write_queue.put(("best", data))


# ============================================================================
# PARALLEL TRAINING DISPLAY (ENHANCED)
# ============================================================================

class CompactPercentColumn(ProgressColumn):
    """Compact percentage display."""
    def render(self, task):
        percentage = task.percentage if task.percentage is not None else 0
        return Text(f"{percentage:>5.1f}%", style="bright_magenta")


@dataclass
class ModelState:
    """State of a training model with extended metrics."""
    model_id: str
    category: str
    device: str  # "gpu" or "cpu"
    device_id: str

    # Progress
    current_epoch: int = 0
    total_epochs: int = 10
    current_batch: int = 0
    total_batches: int = 0

    # Metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    f1_macro: float = 0.0
    accuracy: float = 0.0
    combined_score: float = 0.0
    best_f1: float = 0.0
    best_combined: float = 0.0
    best_epoch: int = 0

    # Timing
    start_time: float = 0.0
    last_update: float = 0.0

    # Status
    phase: str = "queued"  # queued -> initializing -> loading_data -> loading_model -> training -> validation -> complete
    is_complete: bool = False
    is_error: bool = False
    error_msg: str = ""
    has_started: bool = False  # Track if model has actually started (received first update)

    # ========== ENHANCED: Dataset & Model Details ==========
    # These provide better visibility into what's being trained
    num_labels: int = 0  # Number of classification labels
    num_train_samples: int = 0  # Number of training samples
    num_val_samples: int = 0  # Number of validation samples
    language: str = ""  # Language code (e.g., 'EN', 'FR') - empty for multilingual
    model_name: str = ""  # Model name (e.g., 'xlm-roberta-base')
    status_message: str = ""  # Detailed status message (e.g., "Loading tokenizer...")
    batch_size: int = 0  # Actual batch size being used

    # ========== EXTENDED: Per-class metrics ==========
    class_names: List[str] = field(default_factory=list)  # Real label names
    per_class_f1: List[float] = field(default_factory=list)
    per_class_precision: List[float] = field(default_factory=list)
    per_class_recall: List[float] = field(default_factory=list)
    per_class_support: List[int] = field(default_factory=list)

    # ========== EXTENDED: Per-language metrics ==========
    per_language_f1: Dict[str, float] = field(default_factory=dict)
    per_language_accuracy: Dict[str, float] = field(default_factory=dict)
    per_language_samples: Dict[str, int] = field(default_factory=dict)
    samples_per_second: float = 0.0  # Training throughput


class ParallelTrainingDisplay:
    """
    Enhanced Rich display for parallel training.

    Features:
    - Global progress bar (X/Y models complete)
    - Per-model TQDM-like progress bars
    - Device indicators (GPU/CPU) with resource usage
    - ETA and timing information
    - Completed models summary
    - Real-time CPU/Memory monitoring
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_rate: float = 4.0,
        max_label_width: int = 50,
        expand_labels: bool = True,
    ):
        """
        Initialize the parallel training display.

        Parameters
        ----------
        console : Console, optional
            Rich console instance
        refresh_rate : float
            Display refresh rate in Hz
        max_label_width : int
            Maximum width for label columns before wrapping (default: 50)
        expand_labels : bool
            Whether to use full label names without truncation (default: True)
        """
        # Force terminal mode for proper Live display behavior
        self.console = console or Console(force_terminal=True, markup=True, highlight=False)
        self.refresh_rate = refresh_rate
        self.max_label_width = max_label_width
        self.expand_labels = expand_labels

        # State
        self.models: OrderedDict[str, ModelState] = OrderedDict()
        self.total_models = 0
        self.completed_models = 0
        self.failed_models = 0
        self.start_time: Optional[float] = None

        # Best tracking
        self.best_f1_overall = 0.0
        self.best_category = ""

        # Display components
        self.live: Optional[Live] = None
        self._lock = threading.RLock()
        self._running = False

        # Per-model progress bars using Rich Progress
        self._progress: Optional[Progress] = None
        self._task_ids: Dict[str, TaskID] = {}

        # Resource monitoring
        self._cpu_percent: float = 0.0
        self._memory_percent: float = 0.0
        self._memory_used_gb: float = 0.0
        self._last_resource_update: float = 0.0
        self._resource_update_interval: float = 2.0  # Update every 2 seconds

        # Rate-limiting for refresh to prevent dashboard duplication
        self._last_refresh_time: float = 0.0
        self._min_refresh_interval: float = PARALLEL_REFRESH_INTERVAL  # Use configurable interval
        self._startup_grace_period: float = 2.0  # Suppress manual refreshes for first 2 seconds
        self._display_start_time: float = 0.0

        # Terminal buffer clearing to prevent memory buildup in long sessions
        self._last_clear_time: float = 0.0
        self._clear_interval: float = TERMINAL_CLEAR_INTERVAL
        self._updates_since_clear: int = 0
        self._max_updates_before_clear: int = TERMINAL_CLEAR_UPDATES
        self._clear_operations: int = 0  # Statistics

        # Logging suppression during Live display to prevent console output interference
        self._suppressed_handlers: List[logging.Handler] = []
        self._original_handler_levels: Dict[logging.Handler, int] = {}

        # Smart scheduler info (set externally)
        self._scheduler_info: Optional[Dict[str, Any]] = None

    def set_scheduler_info(self, scheduler_info: Dict[str, Any]):
        """Set scheduler info to display in the dashboard header."""
        with self._lock:
            self._scheduler_info = scheduler_info

    def update_scheduler_info(self, **kwargs):
        """Update specific scheduler info values dynamically."""
        with self._lock:
            if self._scheduler_info is None:
                self._scheduler_info = {}
            self._scheduler_info.update(kwargs)

    def start(self, total_models: int):
        """Start the display."""
        if self._running:
            return

        with self._lock:
            self._running = True
            self.total_models = total_models
            self.completed_models = 0
            self.failed_models = 0
            self.start_time = time.time()
            self.models.clear()
            self._task_ids.clear()

            # Initialize terminal clearing state
            self._last_clear_time = time.time()
            self._updates_since_clear = 0
            self._clear_operations = 0

            # Initialize resource monitoring
            self._update_resource_usage()

            # Create progress bar component for per-model tracking
            # Shows both epoch and batch progress for detailed monitoring
            self._progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold]{task.description}", justify="left"),
                BarColumn(bar_width=20, complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
                TextColumn("•"),
                TextColumn("[cyan]E{task.fields[epoch]}/{task.fields[total_epochs]}[/cyan]"),
                TextColumn("[dim]{task.fields[batch_info]}[/dim]"),  # Batch progress info
                TextColumn("•"),
                TextColumn("F1:[bold magenta]{task.fields[f1]}[/bold magenta]"),
                console=self.console,
                expand=False,
                auto_refresh=False,
            )

            self.live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=self.refresh_rate,
                transient=True,  # Clean up display when stopped (prevents duplication)
                vertical_overflow="crop",  # Crop to terminal height to prevent duplication
                screen=True,  # Clear screen on refresh like normal training view
            )
            self._display_start_time = time.time()  # Track when display started for startup grace period

            # Suppress console logging to prevent interference with Live display
            self._suppress_console_logging()

            self.live.start()

            # Log display configuration for debugging long sessions
            logger.info(
                f"Parallel display started: refresh={self._min_refresh_interval:.1f}s, "
                f"terminal_clear={self._clear_interval:.0f}s or every {self._max_updates_before_clear} updates"
            )

    def _suppress_console_logging(self):
        """Suppress console logging handlers to prevent interference with Live display."""
        self._suppressed_handlers = []
        self._original_handler_levels = {}

        # Get root logger and all relevant loggers
        loggers_to_check = [
            logging.getLogger(),  # Root logger
            logging.getLogger('llm_tool'),
            logging.getLogger('llm_tool.trainers'),
            logging.getLogger(__name__),
        ]

        for log in loggers_to_check:
            for handler in log.handlers[:]:
                # Suppress StreamHandlers that write to stdout/stderr
                if isinstance(handler, logging.StreamHandler):
                    if hasattr(handler, 'stream') and handler.stream in (
                        __import__('sys').stdout,
                        __import__('sys').stderr,
                    ):
                        # Save original level and set to CRITICAL+1 to suppress all
                        self._original_handler_levels[handler] = handler.level
                        handler.setLevel(logging.CRITICAL + 1)
                        self._suppressed_handlers.append((log, handler))

    def _restore_console_logging(self):
        """Restore console logging handlers after Live display stops."""
        for handler, original_level in self._original_handler_levels.items():
            handler.setLevel(original_level)
        self._suppressed_handlers = []
        self._original_handler_levels = {}

    def stop(self):
        """Stop the display."""
        if not self._running:
            return

        with self._lock:
            self._running = False

            if self.live:
                self.live.update(self._render(), refresh=True)
                time.sleep(0.3)
                self.live.stop()
                self.live = None

            # Restore console logging after Live display stops
            self._restore_console_logging()

            if self._progress:
                self._progress = None

            # Log memory management stats
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(
                f"Display stopped after {elapsed/60:.1f}m. "
                f"Terminal cleared {self._clear_operations} times."
            )

            self._print_summary()

    def _update_resource_usage(self):
        """Update CPU/memory usage statistics."""
        now = time.time()
        if now - self._last_resource_update < self._resource_update_interval:
            return  # Don't update too frequently

        self._last_resource_update = now

        if HAS_PSUTIL:
            try:
                # CPU usage (non-blocking)
                self._cpu_percent = psutil.cpu_percent(interval=None)

                # Memory usage
                mem = psutil.virtual_memory()
                self._memory_percent = mem.percent
                self._memory_used_gb = mem.used / (1024 ** 3)
            except Exception:
                pass  # Ignore errors in resource monitoring

    def register_model(
        self,
        model_id: str,
        category: str,
        device: str,
        device_id: str,
        total_epochs: int,
        # ========== ENHANCED: Additional model details ==========
        num_labels: int = 0,
        num_train_samples: int = 0,
        num_val_samples: int = 0,
        language: str = "",
        model_name: str = "",
        batch_size: int = 0,
    ):
        """Register a new model with detailed information."""
        with self._lock:
            self.models[model_id] = ModelState(
                model_id=model_id,
                category=category,
                device=device,
                device_id=device_id,
                total_epochs=total_epochs,
                start_time=time.time(),
                # Enhanced details
                num_labels=num_labels,
                num_train_samples=num_train_samples,
                num_val_samples=num_val_samples,
                language=language,
                model_name=model_name,
                batch_size=batch_size,
                status_message="Queued",
            )

            # Create progress task for this model
            if self._progress is not None:
                device_icon = "🖥️" if device == "gpu" else "💻"
                # Use full category name if expand_labels is enabled
                cat_display = category if self.expand_labels else (
                    category[:self.max_label_width - 2] + ".." if len(category) > self.max_label_width else category
                )
                desc = f"{device_icon} {cat_display}"

                # Use high total (epochs * 100) for smoother batch-level progress
                task_id = self._progress.add_task(
                    desc,
                    total=total_epochs * 100,  # Fine-grained progress
                    completed=0,
                    epoch=0,
                    total_epochs=total_epochs,  # For display "E{epoch}/{total_epochs}"
                    f1="-",
                    batch_info="",  # Batch progress info
                    visible=False,  # Hidden until started
                )
                self._task_ids[model_id] = task_id

            self._refresh()

    def update(self, progress: "TrainingProgress"):
        """
        Update display from a TrainingProgress object.

        This is the main interface called by the orchestrator with progress updates
        from worker processes. It routes the update to the appropriate method based
        on the progress phase.
        """
        # Import here to avoid circular imports
        from llm_tool.trainers.hybrid_parallel_trainer import TrainingProgress

        # Register model if not already registered
        if progress.task_id not in self.models:
            self.register_model(
                model_id=progress.task_id,
                category=progress.category_name,
                device="gpu" if progress.device_type == "gpu" else "cpu",
                device_id=progress.worker_id.split(":")[-1] if ":" in progress.worker_id else "0",
                total_epochs=progress.total_epochs,
                num_labels=progress.num_labels,
                num_train_samples=progress.num_train_samples,
                num_val_samples=progress.num_val_samples,
                language=progress.language,
                model_name=progress.model_name,
            )

        # Update device if it changed (e.g., rescheduled from CPU to GPU)
        if progress.task_id in self.models:
            current_device = self.models[progress.task_id].device
            new_device = "gpu" if progress.device_type == "gpu" else "cpu"
            if current_device != new_device:
                self.models[progress.task_id].device = new_device
                logger.debug(f"Model {progress.task_id} device changed: {current_device} -> {new_device}")

        # Handle completion, error, or cancelled states
        if progress.phase == "complete":
            self.mark_complete(
                progress.task_id,
                f1_macro=progress.f1_score,
                accuracy=progress.accuracy,
            )
        elif progress.phase == "error":
            self.mark_error(progress.task_id, progress.message)
        elif progress.phase == "cancelled":
            # Mark cancelled task - will be rescheduled to different device
            self.mark_cancelled(progress.task_id, progress.message)
        else:
            # Update epoch progress with enhanced details
            self.update_epoch(
                model_id=progress.task_id,
                epoch=progress.current_epoch,
                train_loss=progress.train_loss,
                val_loss=progress.val_loss,
                f1_macro=progress.f1_score,
                accuracy=progress.accuracy,
                combined_score=getattr(progress, 'combined_score', 0),  # CRITICAL: Pass combined_score for best model selection
                phase=progress.phase,
                status_message=progress.message,
                num_train_samples=progress.num_train_samples,
                num_val_samples=progress.num_val_samples,
                num_labels=progress.num_labels,
                current_batch=progress.current_batch,
                total_batches=progress.total_batches,
            )

    def update_epoch(
        self,
        model_id: str = None,
        task_id: str = None,  # Alias for model_id
        epoch: int = 0,
        train_loss: float = 0,
        val_loss: float = 0,
        f1_macro: float = 0,
        f1_score: float = 0,  # Alias for f1_macro
        accuracy: float = 0,
        combined_score: float = 0,  # Combined score for best model selection
        phase: str = "training",
        # ========== ENHANCED: Additional update details ==========
        status_message: str = "",
        num_train_samples: int = 0,
        num_val_samples: int = 0,
        num_samples_train: int = None,  # Alias
        num_samples_val: int = None,  # Alias
        num_labels: int = 0,
        current_batch: int = 0,
        total_batches: int = 0,
        # ========== EXTENDED: Per-class metrics ==========
        class_names: Optional[List[str]] = None,
        per_class_f1: Optional[List[float]] = None,
        per_class_precision: Optional[List[float]] = None,
        per_class_recall: Optional[List[float]] = None,
        per_class_support: Optional[List[int]] = None,
        # ========== EXTENDED: Per-language metrics ==========
        per_language_f1: Optional[Dict[str, float]] = None,
        per_language_accuracy: Optional[Dict[str, float]] = None,
        per_language_samples: Optional[Dict[str, int]] = None,
    ):
        """Update epoch progress with detailed status and extended metrics."""
        # Handle aliases
        if model_id is None and task_id is not None:
            model_id = task_id
        if f1_macro == 0 and f1_score > 0:
            f1_macro = f1_score
        if num_train_samples == 0 and num_samples_train is not None:
            num_train_samples = num_samples_train
        if num_val_samples == 0 and num_samples_val is not None:
            num_val_samples = num_samples_val

        with self._lock:
            if model_id not in self.models:
                return

            m = self.models[model_id]
            m.current_epoch = epoch
            m.train_loss = train_loss
            m.val_loss = val_loss
            m.f1_macro = f1_macro
            m.accuracy = accuracy
            # Use combined_score if provided, fallback to f1_macro
            m.combined_score = combined_score if combined_score > 0 else f1_macro
            m.phase = phase
            m.last_update = time.time()
            m.has_started = True  # Mark as started when receiving first update

            # Clear initial "Queued" status_message once training starts
            if m.status_message == "Queued" and phase in ("training", "validation"):
                m.status_message = ""

            # Update enhanced fields if provided
            if status_message:
                m.status_message = status_message
            if num_train_samples > 0:
                m.num_train_samples = num_train_samples
            if num_val_samples > 0:
                m.num_val_samples = num_val_samples
            if num_labels > 0:
                m.num_labels = num_labels
            if current_batch > 0:
                m.current_batch = current_batch
            if total_batches > 0:
                m.total_batches = total_batches

            # Update extended per-class metrics
            if class_names is not None:
                m.class_names = class_names
                if not m.num_labels:
                    m.num_labels = len(class_names)
            if per_class_f1 is not None:
                m.per_class_f1 = per_class_f1
            if per_class_precision is not None:
                m.per_class_precision = per_class_precision
            if per_class_recall is not None:
                m.per_class_recall = per_class_recall
            if per_class_support is not None:
                m.per_class_support = per_class_support

            # Update extended per-language metrics
            if per_language_f1 is not None:
                m.per_language_f1 = per_language_f1
            if per_language_accuracy is not None:
                m.per_language_accuracy = per_language_accuracy
            if per_language_samples is not None:
                m.per_language_samples = per_language_samples

            # Calculate throughput if we have timing info
            if m.start_time > 0 and epoch > 0 and m.num_train_samples > 0:
                elapsed = time.time() - m.start_time
                if elapsed > 0:
                    m.samples_per_second = (epoch * m.num_train_samples) / elapsed

            # Track best using combined_score (consistent with model saving in bert_base.py)
            effective_score = m.combined_score  # Already set above with fallback to f1_macro
            if effective_score > m.best_combined:
                m.best_combined = effective_score
                m.best_f1 = f1_macro
                m.best_epoch = epoch

            # Update Rich Progress task with fine-grained batch progress
            if self._progress is not None and model_id in self._task_ids:
                task_id = self._task_ids[model_id]
                f1_str = f"{f1_macro:.3f}" if f1_macro > 0 else "-"

                # Calculate fine-grained progress (epochs * 100 scale)
                # This provides smooth progress updates within each epoch
                if m.current_batch > 0 and m.total_batches > 0:
                    batch_pct = m.current_batch / m.total_batches
                    # Progress = completed epochs + current epoch batch progress
                    fine_progress = (epoch - 1) * 100 + int(batch_pct * 100)
                    batch_info = f"B:{m.current_batch}/{m.total_batches}"
                else:
                    fine_progress = epoch * 100
                    batch_info = ""

                self._progress.update(
                    task_id,
                    completed=fine_progress,
                    epoch=epoch,
                    total_epochs=m.total_epochs,
                    f1=f1_str,
                    batch_info=batch_info,
                    visible=True,  # Show when started
                )

            self._refresh()

    def mark_complete(self, model_id: str, final_f1: float = 0, final_accuracy: float = 0):
        """Mark model as complete (idempotent - safe to call multiple times)."""
        with self._lock:
            if model_id not in self.models:
                return

            m = self.models[model_id]

            # Only increment counter if not already marked complete
            if not m.is_complete:
                self.completed_models += 1

            m.is_complete = True
            m.phase = "complete"
            m.f1_macro = final_f1
            m.accuracy = final_accuracy

            if final_f1 > self.best_f1_overall:
                self.best_f1_overall = final_f1
                self.best_category = m.category

            # Update progress task to complete and hide
            if self._progress is not None and model_id in self._task_ids:
                task_id = self._task_ids[model_id]
                f1_str = f"{final_f1:.3f}" if final_f1 > 0 else "-"
                self._progress.update(
                    task_id,
                    completed=m.total_epochs * 100,  # Match fine-grained scale
                    epoch=m.total_epochs,
                    total_epochs=m.total_epochs,
                    f1=f1_str,
                    batch_info="✓",  # Show completion marker
                    visible=False,  # Hide completed tasks
                )

            self._refresh()

    def mark_error(self, model_id: str, error_msg: str):
        """Mark model as failed (idempotent - safe to call multiple times)."""
        with self._lock:
            if model_id not in self.models:
                return

            m = self.models[model_id]

            # Only increment counter if not already marked as error
            if not m.is_error:
                self.failed_models += 1

            m.is_error = True
            m.phase = "error"
            m.error_msg = error_msg

            # Hide error task from progress
            if self._progress is not None and model_id in self._task_ids:
                task_id = self._task_ids[model_id]
                self._progress.update(task_id, visible=False)

            self._refresh()

    def mark_cancelled(self, model_id: str, message: str = ""):
        """Mark model as cancelled (will be rescheduled to another device)."""
        with self._lock:
            if model_id not in self.models:
                return

            m = self.models[model_id]

            # Reset the model state so it doesn't count as "running"
            # It will be re-registered when it starts on the new device
            m.has_started = False
            m.phase = "cancelled"
            m.status_message = message or "Rescheduled"

            # Hide cancelled task from progress
            if self._progress is not None and model_id in self._task_ids:
                task_id = self._task_ids[model_id]
                self._progress.update(task_id, visible=False)

            self._refresh()

    def _refresh(self):
        """Refresh the display in place with rate-limiting to prevent duplication."""
        if self.live and self._running:
            current_time = time.time()

            # During startup grace period, let auto-refresh handle everything
            # This prevents rapid manual refreshes during worker registration
            if current_time - self._display_start_time < self._startup_grace_period:
                return  # Let auto-refresh handle startup phase

            # Rate-limit refreshes to prevent rapid updates causing duplication
            if current_time - self._last_refresh_time < self._min_refresh_interval:
                return  # Skip refresh if too soon

            try:
                # Update resource usage periodically
                self._update_resource_usage()

                # Update the Live display content (don't force refresh - let auto-refresh handle timing)
                # This prevents conflicts between manual updates and Live's internal refresh cycle
                self.live.update(self._render(), refresh=False)
                self._last_refresh_time = current_time
                self._updates_since_clear += 1

                # Periodically clear terminal buffer to prevent memory buildup
                self._apply_terminal_housekeeping()
            except Exception:
                pass

    def _apply_terminal_housekeeping(self):
        """
        Periodically clear terminal buffer to prevent memory buildup in long sessions.

        This is critical for VS Code terminals and other emulators that can accumulate
        memory during training sessions that last hours or days.

        Clears when either:
        - Time since last clear exceeds _clear_interval (default: 2 minutes)
        - Number of updates since last clear exceeds _max_updates_before_clear
        """
        current_time = time.time()
        time_since_clear = current_time - self._last_clear_time

        due_by_time = time_since_clear >= self._clear_interval
        due_by_updates = self._updates_since_clear >= self._max_updates_before_clear

        if due_by_time or due_by_updates:
            cleared = _clear_terminal_buffer(self.console)
            if cleared:
                self._last_clear_time = current_time
                self._updates_since_clear = 0
                self._clear_operations += 1
                logger.debug(
                    f"Terminal buffer cleared (operation #{self._clear_operations}, "
                    f"after {time_since_clear:.0f}s)"
                )

    def _render(self) -> Panel:
        """Render the dashboard."""
        with self._lock:
            # ========== HEADER ==========
            elapsed = time.time() - self.start_time if self.start_time else 0

            # Count models by status - distinguish "running" (has_started) from "queued"
            running_models = [m for m in self.models.values()
                           if m.has_started and not m.is_complete and not m.is_error]
            queued_models = [m for m in self.models.values()
                          if not m.has_started and not m.is_complete and not m.is_error]

            # Count running models by device (only actually running, not queued)
            gpu_running = sum(1 for m in running_models if m.device == "gpu")
            cpu_running = sum(1 for m in running_models if m.device == "cpu")
            total_running = len(running_models)
            total_queued = len(queued_models)

            # ETA calculation - sophisticated GPU/CPU aware calculation
            eta_seconds, eta_str, bottleneck = self._calculate_smart_eta(elapsed, running_models, gpu_running, cpu_running)

            # Calculate total estimated time (elapsed + remaining)
            if eta_seconds is not None and eta_seconds > 0:
                total_seconds = elapsed + eta_seconds
                total_str = self._format_time(total_seconds)
            else:
                total_str = "calculating..."

            # Global progress (including partial progress of running models)
            running_progress = sum(
                (m.current_epoch / m.total_epochs) if m.total_epochs > 0 else 0
                for m in running_models
            )
            total_progress = self.completed_models + running_progress
            global_pct = (total_progress / self.total_models * 100) if self.total_models > 0 else 0

            # ========== RESOURCE USAGE SECTION ==========
            resource_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
            resource_table.add_column(style="bold cyan", width=20)
            resource_table.add_column(style="white", width=25)
            resource_table.add_column(style="bold cyan", width=20)
            resource_table.add_column(style="white", width=25)

            # Row 1: Progress and Elapsed Time
            resource_table.add_row(
                "📊 Progress:",
                f"{self.completed_models}/{self.total_models} ({global_pct:.1f}%)",
                "⏱️ Elapsed:",
                self._format_time(elapsed),
            )

            # Row 2: Workers and ETA (remaining time)
            resource_table.add_row(
                "🖥️ GPU Workers:",
                f"[green]{gpu_running}[/green] active",
                "⏳ ETA:",
                eta_str,
            )

            # Row 3: CPU workers and Total estimated time
            resource_table.add_row(
                "💻 CPU Workers:",
                f"[cyan]{cpu_running}[/cyan] active",
                "🕐 Total:",
                f"[bold]{total_str}[/bold]" if total_str != "calculating..." else total_str,
            )

            # Row 4: Best F1
            resource_table.add_row(
                "🏆 Best F1:",
                f"[bold green]{self.best_f1_overall:.4f}[/bold green]" if self.best_f1_overall > 0 else "-",
                "",
                "",
            )

            # Row 4: Resource usage (if psutil available)
            if HAS_PSUTIL:
                cpu_color = "green" if self._cpu_percent < 70 else ("yellow" if self._cpu_percent < 90 else "red")
                mem_color = "green" if self._memory_percent < 70 else ("yellow" if self._memory_percent < 90 else "red")
                resource_table.add_row(
                    "🔲 CPU Usage:",
                    f"[{cpu_color}]{self._cpu_percent:.1f}%[/{cpu_color}]",
                    "💾 Memory:",
                    f"[{mem_color}]{self._memory_used_gb:.1f}GB ({self._memory_percent:.1f}%)[/{mem_color}]",
                )

            # Row 5: Queue info
            resource_table.add_row(
                "📋 Queued:",
                f"[dim]{total_queued}[/dim] models",
                "🔄 Running:",
                f"[bold]{total_running}[/bold] total",
            )

            # Row 6: Smart Scheduler info (if available)
            if self._scheduler_info:
                gpu_share = self._scheduler_info.get('gpu_share', 0)
                cpu_share = self._scheduler_info.get('optimal_cpu_share', 0)
                speed_ratio = self._scheduler_info.get('speed_ratio', 5.0)
                resource_table.add_row(
                    "🧠 Smart Scheduler:",
                    f"GPU ~{gpu_share} / CPU ~{cpu_share}",
                    "⚡ Speed Ratio:",
                    f"{speed_ratio:.1f}x",
                )

            if self.failed_models > 0:
                resource_table.add_row("❌ Failed:", f"[red]{self.failed_models}[/red]", "", "")

            # ========== ACTIVE MODELS TABLE (ENHANCED with dynamic widths) ==========
            models_table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.SIMPLE_HEAVY,
                padding=(0, 1),
                expand=True,
            )
            models_table.add_column("Dev", width=4, justify="center")
            # Dynamic category column with wrapping support
            models_table.add_column(
                "Category",
                min_width=20,
                max_width=self.max_label_width,
                no_wrap=False,
                overflow="fold",
            )
            models_table.add_column("Lang", width=4, justify="center")
            models_table.add_column("Samples", width=8, justify="right")
            models_table.add_column("Labels", width=5, justify="center")
            models_table.add_column("Epoch", width=10, justify="center")
            models_table.add_column("Loss", width=7, justify="right")
            models_table.add_column("F1", width=8, justify="right")
            models_table.add_column("Speed", width=7, justify="right")  # Throughput
            models_table.add_column("Status", width=12, justify="left")

            # Sort: Show running models first (GPU before CPU), then queued
            # Only show running models in the active table (has_started=True)
            active_models = [m for m in self.models.values()
                           if m.has_started and not m.is_complete and not m.is_error]
            active_models.sort(key=lambda x: (0 if x.device == "gpu" else 1, -x.current_epoch))

            # Show all active models with full category names
            for m in active_models:
                dev = "🖥️" if m.device == "gpu" else "💻"
                # Use full category name (wrapping handled by column)
                cat = m.category

                # Language (or 'ML' for multilingual)
                lang = m.language if m.language else "ML"

                # Samples info
                if m.num_train_samples > 0:
                    samples = f"{m.num_train_samples:,}"
                else:
                    samples = "-"

                # Labels
                labels = str(m.num_labels) if m.num_labels > 0 else "-"

                # Epoch and batch progress with percentage
                epoch_pct = (m.current_epoch / m.total_epochs * 100) if m.total_epochs > 0 else 0
                # Show batch progress within epoch if available
                if m.current_batch > 0 and m.total_batches > 0:
                    batch_pct = (m.current_batch / m.total_batches * 100)
                    # Combine epoch and batch progress for more granular display
                    # E.g., "E2/10 B45%" means epoch 2 of 10, 45% through current epoch
                    epoch_str = f"E{m.current_epoch}/{m.total_epochs} B{batch_pct:.0f}%"
                else:
                    epoch_str = f"E{m.current_epoch}/{m.total_epochs} ({epoch_pct:.0f}%)"

                # Loss (show train or val depending on phase)
                if m.phase == "validation" and m.val_loss > 0:
                    loss = f"{m.val_loss:.4f}"
                elif m.train_loss > 0:
                    loss = f"{m.train_loss:.4f}"
                else:
                    loss = "-"

                # F1 with best indicator
                if m.f1_macro > 0:
                    f1 = f"{m.f1_macro:.4f}"
                    if m.best_f1 > 0 and m.f1_macro >= m.best_f1:
                        f1 = f"[bold green]{f1}★[/bold green]"
                else:
                    f1 = "-"

                # Throughput (samples/second)
                if m.samples_per_second > 0:
                    speed = f"{m.samples_per_second:.0f}/s"
                else:
                    speed = "-"

                # Status with color and detailed message
                phase_colors = {
                    "queued": "dim",
                    "initializing": "cyan",
                    "loading_data": "cyan",
                    "loading_model": "cyan",
                    "training": "yellow",
                    "validation": "magenta",
                }
                phase_display = m.status_message if m.status_message else m.phase.title()
                status_color = phase_colors.get(m.phase, "white")
                status = f"[{status_color}]{phase_display[:10]}[/{status_color}]"

                models_table.add_row(dev, cat, lang, samples, labels, epoch_str, loss, f1, speed, status)

            # ========== DETAILED METRICS SECTION ==========
            # Show per-class and per-language metrics for ALL models that have validation results
            detailed_metrics_elements = []

            # Find models with per-class metrics (validation phase or just completed validation)
            models_with_details = [m for m in active_models
                                   if m.per_class_f1 and len(m.per_class_f1) > 0]

            # Show details for ALL models with metrics
            for idx, m in enumerate(models_with_details):
                # Use full category name
                cat_name = m.category

                # Per-class metrics table
                if m.class_names and m.per_class_f1:
                    # Style varies by position (first=green, second=yellow, others=cyan)
                    title_style = "bold green" if idx == 0 else "bold yellow" if idx == 1 else "bold cyan"
                    class_table = Table(
                        title=f"[{title_style}]{cat_name}[/{title_style}] - Per-Label Metrics (Epoch {m.current_epoch})",
                        show_header=True,
                        header_style="bold",
                        box=box.SIMPLE,
                        padding=(0, 1),
                        expand=False,
                    )
                    # Dynamic label column with wrapping
                    class_table.add_column(
                        "Label",
                        style="cyan",
                        min_width=20,
                        max_width=self.max_label_width,
                        no_wrap=False,
                        overflow="fold",
                    )
                    class_table.add_column("F1", justify="right", width=8)
                    class_table.add_column("Prec", justify="right", width=8)
                    class_table.add_column("Rec", justify="right", width=8)
                    class_table.add_column("Support", justify="right", width=8)

                    # Sort by F1 descending for better readability
                    indices = list(range(len(m.class_names)))
                    if m.per_class_f1:
                        indices.sort(key=lambda i: m.per_class_f1[i] if i < len(m.per_class_f1) else 0, reverse=True)

                    for label_idx in indices[:10]:  # Show top 10 labels
                        label_name = m.class_names[label_idx] if label_idx < len(m.class_names) else f"Label {label_idx}"
                        # Use full label name (wrapping handled by column)
                        label_display = label_name if self.expand_labels else (
                            label_name[:self.max_label_width - 2] + ".." if len(label_name) > self.max_label_width else label_name
                        )

                        f1_val = m.per_class_f1[label_idx] if label_idx < len(m.per_class_f1) else 0
                        prec_val = m.per_class_precision[label_idx] if m.per_class_precision and label_idx < len(m.per_class_precision) else 0
                        rec_val = m.per_class_recall[label_idx] if m.per_class_recall and label_idx < len(m.per_class_recall) else 0
                        supp_val = m.per_class_support[label_idx] if m.per_class_support and label_idx < len(m.per_class_support) else 0

                        # Color-code F1
                        f1_color = "green" if f1_val >= 0.7 else ("yellow" if f1_val >= 0.4 else "red")

                        class_table.add_row(
                            label_display,
                            f"[{f1_color}]{f1_val:.3f}[/{f1_color}]",
                            f"{prec_val:.3f}",
                            f"{rec_val:.3f}",
                            str(supp_val),
                        )

                    if len(m.class_names) > 10:
                        class_table.add_row(f"[dim]... +{len(m.class_names) - 10} more labels[/dim]", "", "", "", "")

                    detailed_metrics_elements.append(class_table)

                # Per-language metrics inline
                if m.per_language_f1:
                    lang_parts = []
                    # Sort languages by F1
                    sorted_langs = sorted(m.per_language_f1.items(), key=lambda x: x[1], reverse=True)
                    for lang_code, lang_f1 in sorted_langs[:6]:  # Top 6 languages
                        f1_color = "green" if lang_f1 >= 0.7 else ("yellow" if lang_f1 >= 0.4 else "red")
                        samples = m.per_language_samples.get(lang_code, 0) if m.per_language_samples else 0
                        acc = m.per_language_accuracy.get(lang_code, 0) if m.per_language_accuracy else 0
                        lang_parts.append(f"[{f1_color}]{lang_code.upper()}[/{f1_color}]:F1={lang_f1:.2f}/Acc={acc:.2f}({samples})")

                    if lang_parts:
                        # Combine Languages label with the data on one line
                        lang_line = "  [bold]Languages:[/bold] " + " | ".join(lang_parts)
                        detailed_metrics_elements.append(Text.from_markup(lang_line))

            # ========== QUEUED MODELS PREVIEW ==========
            queued_preview = ""
            if total_queued > 0 and total_queued <= 10:
                # Show full names of queued models
                queued_names = [m.category for m in queued_models[:5]]
                queued_preview = "[dim]Next: " + ", ".join(queued_names)
                if total_queued > 5:
                    queued_preview += f" ... +{total_queued - 5}[/dim]"
                else:
                    queued_preview += "[/dim]"
            elif total_queued > 10:
                queued_preview = f"[dim]Next: {total_queued} models queued[/dim]"

            # ========== RECENTLY COMPLETED ==========
            completed_models_list = [m for m in self.models.values() if m.is_complete]
            recent_completed = completed_models_list[-5:]  # Last 5

            # ========== COMBINE ==========
            elements = [
                resource_table,
                Text(""),
                Text("═" * 100, style="dim cyan"),
                Text(" ACTIVE TRAINING JOBS", style="bold cyan"),
                Text("─" * 100, style="dim"),
            ]

            # Add Rich Progress bars for running models if available
            if self._progress is not None and total_running > 0:
                elements.append(self._progress)
                elements.append(Text(""))

            # Add details table
            elements.append(models_table)

            # Add detailed metrics section if available
            if detailed_metrics_elements:
                elements.extend([
                    Text(""),
                    Text("─" * 100, style="dim"),
                    Text(" DETAILED METRICS (Validation)", style="bold magenta"),
                ])
                elements.extend(detailed_metrics_elements)

            # Add queued preview if any
            if queued_preview:
                elements.append(Text.from_markup(queued_preview))

            # Recently completed section
            if recent_completed:
                elements.extend([
                    Text(""),
                    Text("─" * 100, style="dim"),
                    Text(" RECENTLY COMPLETED", style="bold green"),
                ])
                # Create a mini-table for completed models with full category names
                completed_parts = []
                for m in recent_completed:
                    # Use full category name
                    cat_display = m.category
                    f1_color = "green" if m.f1_macro >= 0.7 else ("yellow" if m.f1_macro >= 0.5 else "red")
                    lang_str = f"[{m.language}]" if m.language else ""
                    completed_parts.append(
                        f"[{f1_color}]✓ {cat_display}{lang_str} F1={m.f1_macro:.3f}[/{f1_color}]"
                    )
                elements.append(Text.from_markup(" | ".join(completed_parts)))

            return Panel(
                Group(*elements),
                title="[bold blue]🚀 PARALLEL TRAINING DASHBOARD[/bold blue]",
                subtitle=f"[dim]{total_running} running | {total_queued} queued | {self.completed_models} done | {self.failed_models} failed[/dim]",
                border_style="bold blue",
                box=box.HEAVY,
            )

    def _print_summary(self):
        """Print final summary."""
        completed = [m for m in self.models.values() if m.is_complete]
        if not completed:
            return

        self.console.print()

        summary = Table(
            title="🏁 TRAINING COMPLETE",
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
        summary.add_column("Best Epoch", justify="center", width=10)
        summary.add_column("Device", width=10)

        # Sort by F1
        completed.sort(key=lambda x: x.f1_macro, reverse=True)

        for m in completed[:20]:
            # Use full category name (wrapping handled by column)
            cat = m.category
            f1_style = "bold green" if m.f1_macro >= 0.8 else ("yellow" if m.f1_macro >= 0.6 else "red")

            summary.add_row(
                cat,
                f"[{f1_style}]{m.f1_macro:.4f}[/{f1_style}]",
                f"{m.accuracy:.4f}",
                str(m.best_epoch),
                "🖥️ GPU" if m.device == "gpu" else "💻 CPU",
            )

        self.console.print(summary)

        if self.best_f1_overall > 0:
            self.console.print()
            self.console.print(Panel(
                f"[bold green]🏆 Best Model:[/bold green] {self.best_category}\n"
                f"[cyan]F1 Score:[/cyan] {self.best_f1_overall:.4f}",
                border_style="green",
            ))

    def _calculate_smart_eta(
        self,
        elapsed: float,
        running_models: List["ModelState"],
        gpu_running: int,
        cpu_running: int,
    ) -> Tuple[Optional[float], str, Optional[str]]:
        """
        Calculate sophisticated ETA considering GPU/CPU speed differences.

        This method:
        1. Tracks progress separately for GPU and CPU models
        2. Calculates time-per-model-unit for each device type
        3. Uses scheduler info for model distribution if available
        4. Accounts for parallel workers
        5. Returns max(GPU_ETA, CPU_ETA) since they run in parallel

        Parameters
        ----------
        elapsed : float
            Total elapsed time in seconds
        running_models : List[ModelState]
            Currently running models
        gpu_running : int
            Number of GPU workers active
        cpu_running : int
            Number of CPU workers active

        Returns
        -------
        Tuple[Optional[float], str, Optional[str]]
            (eta_seconds, formatted_eta_string, bottleneck_device)
            eta_seconds is None if still calculating
            bottleneck_device is "GPU" or "CPU" or None
        """
        if elapsed < 60:  # Need at least 1 minute of data
            return (None, "calculating...", None)

        # Separate completed models by device
        completed_gpu = sum(1 for m in self.models.values() if m.is_complete and m.device == "gpu")
        completed_cpu = sum(1 for m in self.models.values() if m.is_complete and m.device == "cpu")

        # Calculate partial progress of running models by device
        gpu_running_progress = 0.0
        cpu_running_progress = 0.0

        for m in running_models:
            if m.total_epochs > 0:
                # Progress = epoch progress + batch progress within current epoch
                progress = m.current_epoch / m.total_epochs
                if m.total_batches > 0 and m.current_batch > 0:
                    progress += (m.current_batch / m.total_batches) / m.total_epochs

                if m.device == "gpu":
                    gpu_running_progress += progress
                else:
                    cpu_running_progress += progress

        # Total progress by device
        gpu_progress = completed_gpu + gpu_running_progress
        cpu_progress = completed_cpu + cpu_running_progress

        # Get model distribution from scheduler info or estimate
        if self._scheduler_info:
            # Scheduler provides shares as ratios (e.g., 0.73 for GPU, 0.27 for CPU)
            gpu_share = self._scheduler_info.get("gpu_share", 0.7)
            cpu_share = self._scheduler_info.get("optimal_cpu_share", 0.3)
            gpu_total = int(self.total_models * gpu_share)
            cpu_total = self.total_models - gpu_total
        else:
            # Estimate: assume current distribution continues
            total_assigned = completed_gpu + completed_cpu + len(running_models)
            if total_assigned > 0:
                gpu_ratio = (completed_gpu + sum(1 for m in running_models if m.device == "gpu")) / total_assigned
                gpu_total = int(self.total_models * gpu_ratio)
                cpu_total = self.total_models - gpu_total
            else:
                # Default: assume 70% GPU, 30% CPU
                gpu_total = int(self.total_models * 0.7)
                cpu_total = self.total_models - gpu_total

        # Calculate time per model unit for each device type
        # Use None to indicate "not applicable" vs 0 meaning "done"
        gpu_eta = None
        cpu_eta = None

        # GPU ETA calculation
        if gpu_total > 0 and gpu_progress > 0.05:  # Need some GPU tasks and progress
            gpu_time_per_unit = elapsed / gpu_progress
            gpu_remaining = gpu_total - gpu_progress
            if gpu_remaining > 0 and gpu_running > 0:
                # Account for parallel GPU workers
                gpu_eta = (gpu_remaining / gpu_running) * gpu_time_per_unit
            elif gpu_remaining <= 0:
                gpu_eta = 0  # GPU work is done
        elif gpu_total > 0 and gpu_running > 0:
            # GPU has tasks but not enough progress to estimate yet
            gpu_eta = None  # Still calculating

        # CPU ETA calculation - only if CPU is actually being used
        if cpu_total > 0 and cpu_running > 0 and cpu_progress > 0.05:
            cpu_time_per_unit = elapsed / cpu_progress
            cpu_remaining = cpu_total - cpu_progress
            if cpu_remaining > 0:
                # Account for parallel CPU workers
                cpu_eta = (cpu_remaining / cpu_running) * cpu_time_per_unit
            else:
                cpu_eta = 0  # CPU work is done
        elif cpu_total > 0 and cpu_running > 0:
            # CPU has tasks but not enough progress to estimate yet
            cpu_eta = None  # Still calculating
        # If cpu_running == 0, CPU is not being used - don't count it

        # Combined ETA is the max (since GPU and CPU run in parallel)
        # But only count devices that are actually being used
        valid_etas = []
        if gpu_eta is not None:
            valid_etas.append(("GPU", gpu_eta))
        if cpu_eta is not None:
            valid_etas.append(("CPU", cpu_eta))

        if not valid_etas:
            # Neither device has valid ETA yet - use fallback
            total_progress = gpu_progress + cpu_progress
            if total_progress > 0.1:
                time_per_unit = elapsed / total_progress
                remaining = self.total_models - total_progress
                fallback_eta = max(0, time_per_unit * remaining)
                return (fallback_eta, self._format_time(fallback_eta), None)
            return (None, "calculating...", None)

        # Use the maximum of valid ETAs (whichever finishes last)
        final_eta = max(eta for _, eta in valid_etas)

        # Determine bottleneck device (only if both have valid non-zero ETAs)
        bottleneck = None
        if gpu_eta is not None and cpu_eta is not None and gpu_eta > 0 and cpu_eta > 0:
            if gpu_eta > cpu_eta * 1.2:
                bottleneck = "GPU"
            elif cpu_eta > gpu_eta * 1.2:
                bottleneck = "CPU"
        elif len(valid_etas) == 1:
            # Only one device is active - that's the working device
            bottleneck = valid_etas[0][0]

        # Format string with bottleneck/device info
        if bottleneck:
            eta_str = f"{self._format_time(final_eta)} ({bottleneck})"
        else:
            eta_str = self._format_time(final_eta)

        return (final_eta, eta_str, bottleneck)

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 0:
            return "-"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ============================================================================
# PARALLEL CHART MANAGER
# ============================================================================

class ParallelChartManager:
    """
    Manages chart generation for parallel training.

    Features:
    - Per-model real-time charts
    - Combined summary chart
    - Thread-safe updates
    """

    def __init__(self, output_dir: str, session_id: str, training_approach: str = 'one-vs-all'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.training_approach = training_approach

        # Chart instances per model
        self._charts: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # Metrics for summary chart
        self._model_metrics: Dict[str, Dict] = {}

    def register_model(
        self,
        model_id: str,
        category: str,
        model_name: str,
        num_labels: int = 2,
        languages: Optional[List[str]] = None,
        num_samples_train: int = 0,
        num_samples_val: int = 0,
    ):
        """Register a model for chart tracking."""
        with self._lock:
            try:
                from llm_tool.trainers.training_metrics_chart import TrainingMetricsChart

                # Create per-model chart subdirectory directly under output_dir
                # output_dir is already the charts directory (e.g., logs/.../parallel_training/charts/)
                chart_dir = self.output_dir / model_id
                chart_dir.mkdir(parents=True, exist_ok=True)

                self._charts[model_id] = TrainingMetricsChart(
                    output_dir=str(chart_dir),
                    session_id=self.session_id,
                    model_name=model_name,
                    num_labels=num_labels,
                    languages=languages,
                    training_approach=self.training_approach,
                    category_name=category,
                    enhanced_style=True,  # Use enhanced styling for parallel training charts
                    num_samples_train=num_samples_train,
                    num_samples_val=num_samples_val,
                )

                self._model_metrics[model_id] = {
                    'category': category,
                    'model_name': model_name,
                    'f1_macro': 0,
                    'best_f1_macro': 0,
                    'combined_score': 0,
                    'best_combined_score': 0,
                    'training_time': 0,
                    'best_epoch': 0,
                    'num_labels': num_labels,
                }

            except ImportError:
                logger.warning("TrainingMetricsChart not available")

    def update_model(
        self,
        model_id: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        f1_macro: float,
        f1_micro: float = 0,
        f1_class_1: Optional[float] = None,
        combined_score: Optional[float] = None,
        is_best: bool = False,
        per_class_support: Optional[Dict[str, int]] = None,
        per_class_f1: Optional[Dict[str, float]] = None,
        language_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        learning_rate: Optional[float] = None,
    ) -> Optional[str]:
        """Update model chart with new epoch data."""
        with self._lock:
            if model_id not in self._charts:
                return None

            chart = self._charts[model_id]

            try:
                path = chart.update(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    accuracy=accuracy,
                    f1_macro=f1_macro,
                    f1_micro=f1_micro,
                    f1_class_1=f1_class_1,
                    combined_score=combined_score,
                    is_best=is_best,
                    per_class_support=per_class_support,
                    per_class_f1=per_class_f1,
                    language_metrics=language_metrics,
                    learning_rate=learning_rate,
                )

                # Update summary metrics
                if model_id in self._model_metrics:
                    self._model_metrics[model_id]['f1_macro'] = f1_macro
                    if combined_score is not None:
                        self._model_metrics[model_id]['combined_score'] = combined_score
                    if is_best:
                        self._model_metrics[model_id]['best_f1_macro'] = f1_macro
                        self._model_metrics[model_id]['best_epoch'] = epoch
                        if combined_score is not None:
                            self._model_metrics[model_id]['best_combined_score'] = combined_score

                return path

            except Exception as e:
                logger.error(f"Error updating chart for {model_id}: {e}")
                return None

    def finalize_model(self, model_id: str, training_time: float):
        """Finalize model chart and save metrics."""
        with self._lock:
            if model_id in self._charts:
                try:
                    self._charts[model_id].save_metrics_json()
                except Exception as e:
                    logger.error(f"Error saving metrics JSON for {model_id}: {e}")

            if model_id in self._model_metrics:
                self._model_metrics[model_id]['training_time'] = training_time

    def generate_summary_chart(self, total_training_time: float, model_name: str) -> Optional[str]:
        """Generate summary chart for all models."""
        with self._lock:
            try:
                from llm_tool.trainers.training_metrics_chart import generate_training_summary_chart

                # output_dir is already the charts directory
                return generate_training_summary_chart(
                    output_dir=str(self.output_dir),
                    session_id=self.session_id,
                    model_name=model_name,
                    results_per_key=self._model_metrics,
                    training_approach=f'parallel-{self.training_approach}',
                    total_training_time=total_training_time,
                )

            except Exception as e:
                logger.error(f"Error generating summary chart: {e}")
                return None


# ============================================================================
# PARALLEL TRAINING MANAGER (MAIN ORCHESTRATOR)
# ============================================================================

@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training."""
    model_name: str = "xlm-roberta-base"
    epochs: int = 10
    learning_rate: float = 2e-5
    output_dir: str = "models"  # Directory for saving trained models
    session_id: str = ""
    max_cpu_workers: Optional[int] = None
    prefer_gpu: bool = True
    enable_charts: bool = True
    enable_logging: bool = True
    # Logging configuration - separate from model output
    logs_dir: Optional[str] = None  # Base directory for logs (default: logs/training_arena/{session_id})
    # Training approach and multi-label settings
    training_approach: str = "one-vs-all"  # one-vs-all, multi-label, multi-class
    multi_label: bool = False  # True for multi-label with sigmoid activation
    # Hybrid trainer: GPU in main process + CPU in background (better resource utilization)
    use_hybrid_trainer: bool = True  # Use optimized hybrid architecture
    multi_label_threshold: float = 0.5  # Threshold for multi-label predictions
    # Reinforced learning settings
    reinforced_learning: bool = False  # Enable reinforced learning for low F1 models
    reinforced_epochs: Optional[int] = None  # Manual RL epochs (None = auto-calculate)
    rl_f1_threshold: float = 0.7  # F1 threshold below which RL is triggered
    rl_oversample_factor: float = 2.0  # Oversample factor for RL
    rl_class_weight_factor: float = 2.0  # Class weight factor for RL
    # Per-language training settings
    train_by_language: bool = False  # If True, train separate model per (category, language)
    languages: Optional[List[str]] = None  # List of languages to train (e.g., ['EN', 'FR', 'DE'])
    models_by_language: Optional[Dict[str, str]] = None  # Optional: different model per language
    # Intelligent task scheduling settings
    scheduler_config: Optional[Any] = None  # SchedulerConfig for smart task distribution
    # SSH distributed training across multiple machines
    distributed_orchestrator: Optional[Any] = None  # DistributedTrainingOrchestrator instance


class ParallelTrainingManager:
    """
    Complete parallel training manager.

    Integrates:
    - Resource detection and allocation
    - Thread-safe metrics logging
    - Unified display
    - Chart generation
    - Full package integration
    """

    def __init__(self, config: ParallelTrainingConfig, console: Optional[Console] = None):
        self.config = config
        # Force terminal mode for proper Live display behavior
        self.console = console or Console(force_terminal=True, markup=True, highlight=False)

        # Generate session ID if not provided
        if not config.session_id:
            config.session_id = f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ========== MODELS DIRECTORY SETUP ==========
        # Models are saved in: {output_dir}/{session_id}/parallel_training/
        # This mirrors the normal training structure: models/{session_id}/normal_training/
        # Both are within the same session folder for consistency
        self.models_dir = Path(config.output_dir) / config.session_id / "parallel_training"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # ========== LOGS DIRECTORY SETUP ==========
        # Use centralized logs structure: logs/training_arena/{session_id}/
        # This keeps logs separate from model output directories
        if config.logs_dir:
            # User-provided logs directory
            self.logs_base_dir = Path(config.logs_dir)
        else:
            # Default: logs/training_arena/{session_id}/parallel_training/
            self.logs_base_dir = get_session_dir(config.session_id) / "parallel_training"

        # Create subdirectories for organized logging
        self.metrics_dir = self.logs_base_dir / "metrics"
        self.charts_dir = self.logs_base_dir / "charts"

        # Ensure directories exist
        self.logs_base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Models directory: {self.models_dir}")
        logger.info(f"📁 Logs directory: {self.logs_base_dir}")
        self.console.print(f"[cyan]📁 Models directory:[/cyan] {self.models_dir}")
        self.console.print(f"[cyan]📁 Logs directory:[/cyan] {self.logs_base_dir}")

        # Initialize components
        self.display = ParallelTrainingDisplay(self.console)

        if config.enable_logging:
            self.metrics_logger = ThreadSafeMetricsLogger(
                output_dir=str(self.metrics_dir),  # Use dedicated metrics directory
                session_id=config.session_id,
            )
        else:
            self.metrics_logger = None

        if config.enable_charts:
            self.chart_manager = ParallelChartManager(
                output_dir=str(self.charts_dir),  # Use dedicated charts directory
                session_id=config.session_id,
                training_approach=config.training_approach,
            )
        else:
            self.chart_manager = None

        # Resource analyzer (uses memory-aware HardwareAnalyzer)
        from llm_tool.trainers.hybrid_parallel_trainer import HardwareAnalyzer
        self.hw_analyzer = HardwareAnalyzer()

        # Results storage
        self.results: Dict[str, Dict] = {}
        self._lock = threading.RLock()

    def analyze_resources(self, total_tasks: int):
        """Analyze and display resource allocation."""
        plan = self.hw_analyzer.create_resource_plan(
            model_name=self.config.model_name,
            total_tasks=total_tasks,
            prefer_gpu=self.config.prefer_gpu,
            max_cpu_workers=self.config.max_cpu_workers,
        )

        # Display plan
        from rich.table import Table

        table = Table(
            title="📊 Resource Allocation Plan",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
        )
        table.add_column("Device", style="cyan")
        table.add_column("Type")
        table.add_column("Batch Size", justify="center")
        table.add_column("Workers", justify="center")
        table.add_column("Threads", justify="center")

        if plan.gpu_allocation:
            g = plan.gpu_allocation
            table.add_row(f"🖥️ {g.device_id.upper()}", "GPU", str(g.batch_size), str(g.num_workers), "-")

        for c in plan.cpu_allocations:
            table.add_row(f"💻 {c.device_id}", "CPU", str(c.batch_size), str(c.num_workers), str(c.threads_per_worker or "-"))

        self.console.print(table)
        self.console.print(f"\n[bold]Total parallel workers:[/bold] {plan.total_parallel_workers}")

        return plan

    def run(
        self,
        training_files: Dict[str, str],
        _progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Dict]:
        """
        Run parallel training on all training files.

        Parameters
        ----------
        training_files : Dict[str, str]
            Mapping of category names to training file paths
        _progress_callback : Callable, optional
            Optional callback for external progress tracking (reserved for future use)

        Returns
        -------
        Dict[str, Dict]
            Results for each category
        """
        from llm_tool.trainers.hybrid_parallel_trainer import TrainingTask

        num_categories = len(training_files)
        if num_categories == 0:
            return {}

        # Calculate total tasks based on per-language training setting
        languages = self.config.languages or []
        train_by_language = self.config.train_by_language and len(languages) > 1

        if train_by_language:
            # Per-language: one task per (category, language) pair
            total_tasks = num_categories * len(languages)
            logger.info(f"Per-language training: {num_categories} categories × {len(languages)} languages = {total_tasks} models")
        else:
            # Multilingual: one task per category
            total_tasks = num_categories
            logger.info(f"Multilingual training: {num_categories} models")

        # Show resource plan (BEFORE starting Live display)
        plan = self.analyze_resources(total_tasks)

        # Calculate device distribution from plan
        num_gpu_workers = 1 if plan.gpu_allocation else 0
        num_cpu_workers = len(plan.cpu_allocations)
        total_workers = num_gpu_workers + num_cpu_workers

        # Log resource allocation (BEFORE Live display starts)
        logger.info(f"[PARALLEL] Resource allocation: {num_gpu_workers} GPU + {num_cpu_workers} CPU = {total_workers} workers")
        self.console.print(f"[cyan]📊 Workers: {num_gpu_workers} GPU + {num_cpu_workers} CPU = {total_workers} parallel[/cyan]")

        # Show smart scheduler info BEFORE Live display starts
        try:
            from llm_tool.trainers.task_scheduler import SmartTaskScheduler
            temp_scheduler = SmartTaskScheduler(self.config.scheduler_config)
            should_offer, scheduler_msg = temp_scheduler.should_offer_parallel(total_tasks)
            if should_offer:
                # Calculate distribution preview
                speed_ratio = temp_scheduler.speed_tracker.get_speed_ratio()
                optimal_cpu = temp_scheduler.cutoff_calculator.calculate_optimal_cpu_share(total_tasks, num_cpu_workers)
                gpu_share = total_tasks - optimal_cpu
                self.console.print(f"[cyan]🧠 Smart Scheduler: GPU ~{gpu_share} tasks, CPU ~{optimal_cpu} tasks (ratio: {speed_ratio:.1f}x)[/cyan]")
        except Exception as e:
            logger.debug(f"Could not show scheduler preview: {e}")

        self.console.print()  # Add spacing before dashboard

        # Start components
        if self.metrics_logger:
            self.metrics_logger.start()

        # Start Live display AFTER all console prints
        self.display.start(total_tasks)

        start_time = time.time()

        try:
            # Calculate max possible epochs (including potential reinforced learning)
            max_epochs = self.config.epochs
            if self.config.reinforced_learning:
                rl_epochs = self.config.reinforced_epochs or self.config.epochs
                max_epochs = self.config.epochs + rl_epochs

            # Create training tasks
            tasks = []

            # Track task index for device assignment
            task_idx = 0

            for category, file_path in training_files.items():
                if train_by_language:
                    # ========== PER-LANGUAGE TRAINING ==========
                    # Create one task per (category, language) pair
                    for lang_code in languages:
                        # Determine model name for this language
                        if self.config.models_by_language and lang_code in self.config.models_by_language:
                            model_name_for_lang = self.config.models_by_language[lang_code]
                        else:
                            model_name_for_lang = self.config.model_name

                        # Create unique task ID and output path
                        # Models saved in: models/{session_id}/parallel_training/{category}/model_{lang_code}/
                        task_id = f"task_{category}_{lang_code}"
                        output_path = str(self.models_dir / category / f"model_{lang_code}")

                        # Display name includes language
                        display_name = f"{category} [{lang_code}]"

                        task = TrainingTask(
                            task_id=task_id,
                            category_name=display_name,
                            model_name=model_name_for_lang,
                            data_path=file_path,
                            output_path=output_path,
                            config={
                                "epochs": self.config.epochs,
                                "learning_rate": self.config.learning_rate,
                                # CRITICAL: Pass session_id to workers to ensure unified logging
                                # This prevents duplicate session directories (training_session_XXXXXX)
                                "session_id": self.config.session_id,
                                # Multi-label settings
                                "training_approach": self.config.training_approach,
                                "multi_label": self.config.multi_label,
                                "multi_label_threshold": self.config.multi_label_threshold,
                                # Reinforced learning settings
                                "reinforced_learning": self.config.reinforced_learning,
                                "reinforced_epochs": self.config.reinforced_epochs,
                                "rl_f1_threshold": self.config.rl_f1_threshold,
                                "rl_oversample_factor": self.config.rl_oversample_factor,
                                "rl_class_weight_factor": self.config.rl_class_weight_factor,
                                # Per-language settings - CRITICAL
                                "train_by_language": True,
                                "language_code": lang_code,  # Filter data to this language only
                                "base_category": category,  # Original category name without language
                            },
                        )
                        tasks.append(task)

                        # Determine device for this task based on allocation order
                        # First num_gpu_workers tasks go to GPU, rest to CPU
                        if task_idx < num_gpu_workers:
                            task_device = "gpu"
                            task_device_id = "mps"
                        else:
                            task_device = "cpu"
                            cpu_worker_idx = (task_idx - num_gpu_workers) % max(1, num_cpu_workers)
                            task_device_id = f"cpu:{cpu_worker_idx}"
                        task_idx += 1

                        # Register with display
                        self.display.register_model(
                            model_id=task_id,
                            category=display_name,
                            device=task_device,
                            device_id=task_device_id,
                            total_epochs=max_epochs,
                        )

                        # Register with chart manager
                        if self.chart_manager:
                            self.chart_manager.register_model(
                                model_id=task_id,
                                category=display_name,
                                model_name=model_name_for_lang,
                                languages=self.config.languages,
                            )
                else:
                    # ========== MULTILINGUAL TRAINING ==========
                    # Create one task per category (all languages combined)
                    # Models saved in: models/{session_id}/parallel_training/{category}/
                    task_id = f"task_{category}"
                    output_path = str(self.models_dir / category)

                    task = TrainingTask(
                        task_id=task_id,
                        category_name=category,
                        model_name=self.config.model_name,
                        data_path=file_path,
                        output_path=output_path,
                        config={
                            "epochs": self.config.epochs,
                            "learning_rate": self.config.learning_rate,
                            # CRITICAL: Pass session_id to workers to ensure unified logging
                            # This prevents duplicate session directories (training_session_XXXXXX)
                            "session_id": self.config.session_id,
                            # Multi-label settings
                            "training_approach": self.config.training_approach,
                            "multi_label": self.config.multi_label,
                            "multi_label_threshold": self.config.multi_label_threshold,
                            # Reinforced learning settings
                            "reinforced_learning": self.config.reinforced_learning,
                            "reinforced_epochs": self.config.reinforced_epochs,
                            "rl_f1_threshold": self.config.rl_f1_threshold,
                            "rl_oversample_factor": self.config.rl_oversample_factor,
                            "rl_class_weight_factor": self.config.rl_class_weight_factor,
                            # Multilingual: no language filtering
                            "train_by_language": False,
                            "language_code": None,
                            "base_category": category,
                        },
                    )
                    tasks.append(task)

                    # Determine device for this task based on allocation order
                    # First num_gpu_workers tasks go to GPU, rest to CPU
                    if task_idx < num_gpu_workers:
                        task_device = "gpu"
                        task_device_id = "mps"
                    else:
                        task_device = "cpu"
                        cpu_worker_idx = (task_idx - num_gpu_workers) % max(1, num_cpu_workers)
                        task_device_id = f"cpu:{cpu_worker_idx}"
                    task_idx += 1

                    # Register with display
                    self.display.register_model(
                        model_id=task_id,
                        category=category,
                        device=task_device,
                        device_id=task_device_id,
                        total_epochs=max_epochs,
                    )

                    # Register with chart manager
                    if self.chart_manager:
                        self.chart_manager.register_model(
                            model_id=task_id,
                            category=category,
                            model_name=self.config.model_name,
                            languages=self.config.languages,
                        )

            # Create progress callback for real-time updates
            def progress_callback(update):
                """Handle progress updates from workers with extended metrics."""
                task_id = update.task_id
                category = update.category_name

                # Handle different phases
                if update.phase == "error":
                    # Mark model as errored in display
                    self.display.mark_error(task_id, update.message)
                elif update.phase == "complete":
                    # Mark model as complete
                    self.display.mark_complete(
                        task_id,
                        final_f1=update.f1_score,
                        final_accuracy=update.accuracy
                    )
                else:
                    # Regular progress update with extended metrics
                    self.display.update_epoch(
                        task_id=task_id,
                        epoch=update.current_epoch,
                        train_loss=update.train_loss,
                        val_loss=update.val_loss,
                        f1_score=update.f1_score,
                        accuracy=update.accuracy,
                        combined_score=getattr(update, 'combined_score', 0),
                        phase=update.phase,
                        current_batch=getattr(update, 'current_batch', 0),
                        total_batches=getattr(update, 'total_batches', 0),
                        # Extended metrics - samples
                        num_samples_train=getattr(update, 'num_train_samples', None),
                        num_samples_val=getattr(update, 'num_val_samples', None),
                        # Extended metrics - per-class
                        class_names=getattr(update, 'class_names', None),
                        per_class_f1=getattr(update, 'per_class_f1', None),
                        per_class_precision=getattr(update, 'per_class_precision', None),
                        per_class_recall=getattr(update, 'per_class_recall', None),
                        per_class_support=getattr(update, 'per_class_support', None),
                        # Extended metrics - per-language
                        per_language_f1=getattr(update, 'per_language_f1', None),
                        per_language_accuracy=getattr(update, 'per_language_accuracy', None),
                        per_language_samples=getattr(update, 'per_language_samples', None),
                    )

                # Update chart (only for actual training epochs)
                if self.chart_manager and update.current_epoch > 0 and update.phase not in ("error", "complete"):
                    # Check if this is actually the best epoch by comparing with previous best combined_score
                    current_best_combined = self.chart_manager._model_metrics.get(task_id, {}).get('best_combined_score', 0)
                    update_combined = getattr(update, 'combined_score', None) or update.f1_score
                    is_best = update_combined > 0 and update.phase == "validation" and update_combined > current_best_combined

                    # Build dicts from lists for per-class metrics
                    per_class_support_dict = None
                    per_class_f1_dict = None
                    class_names = getattr(update, 'class_names', None)

                    if class_names:
                        # Convert List[int] → Dict[str, int]
                        support_list = getattr(update, 'per_class_support', None)
                        if support_list:
                            per_class_support_dict = {
                                name: support_list[i]
                                for i, name in enumerate(class_names)
                                if i < len(support_list)
                            }

                        # Convert List[float] → Dict[str, float]
                        f1_list = getattr(update, 'per_class_f1', None)
                        if f1_list:
                            per_class_f1_dict = {
                                name: f1_list[i]
                                for i, name in enumerate(class_names)
                                if i < len(f1_list)
                            }

                    # Build language_metrics from per_language_f1/accuracy
                    language_metrics_dict = None
                    per_lang_f1 = getattr(update, 'per_language_f1', None)
                    per_lang_acc = getattr(update, 'per_language_accuracy', None)
                    if per_lang_f1:
                        language_metrics_dict = {}
                        for lang, f1 in per_lang_f1.items():
                            language_metrics_dict[lang] = {
                                'f1_macro': f1,
                                'accuracy': per_lang_acc.get(lang, 0) if per_lang_acc else 0
                            }

                    self.chart_manager.update_model(
                        model_id=task_id,
                        epoch=update.current_epoch,
                        train_loss=update.train_loss,
                        val_loss=update.val_loss,
                        accuracy=update.accuracy,
                        f1_macro=update.f1_score,
                        f1_micro=getattr(update, 'f1_micro', 0),
                        f1_class_1=getattr(update, 'f1_class_1', None),
                        combined_score=getattr(update, 'combined_score', None),
                        is_best=is_best,
                        per_class_support=per_class_support_dict,
                        per_class_f1=per_class_f1_dict,
                        language_metrics=language_metrics_dict,
                        learning_rate=getattr(update, 'learning_rate', None),
                    )

                # Log to CSV (only for actual training epochs)
                if self.metrics_logger and update.current_epoch > 0 and update.phase not in ("error",):
                    self.metrics_logger.log_epoch(
                        model_id=task_id,
                        category=category,
                        device=update.device_type,
                        epoch=update.current_epoch,
                        train_loss=update.train_loss,
                        val_loss=update.val_loss,
                        f1_macro=update.f1_score,
                        accuracy=update.accuracy,
                        phase=update.phase,
                        # Extended metadata for rationalized logging
                        huggingface_model=self.config.model_name,
                        model_class=getattr(update, 'model_name', '') or self.config.model_name,
                        training_mode=self.config.training_approach,
                        language=getattr(update, 'language', ''),
                    )

            # ========== DISTRIBUTED SSH TRAINING BRANCH ==========
            if self.config.distributed_orchestrator:
                return self._run_distributed(
                    tasks=tasks,
                    plan=plan,
                    num_cpu_workers=num_cpu_workers,
                    num_gpu_workers=num_gpu_workers,
                    start_time=start_time,
                )

            # Always use the optimized hybrid trainer; legacy orchestrator is deprecated
            from llm_tool.trainers.hybrid_parallel_trainer import (
                HybridParallelTrainer,
                TrainingTask as HybridTask,
                ProgressUpdate,
            )

            if not self.config.use_hybrid_trainer:
                logger.warning("Legacy ProcessPool orchestrator is deprecated; falling back to HybridParallelTrainer.")

            # Convert tasks to hybrid format
            hybrid_tasks = []
            for task in tasks:
                hybrid_task = HybridTask(
                    task_id=task.task_id,
                    category_name=task.category_name,
                    model_name=task.model_name,
                    data_path=task.data_path,
                    output_path=task.output_path,
                    config=task.config,
                )
                hybrid_tasks.append(hybrid_task)

            # Container to hold trainer reference (allows callback to access it after creation)
            trainer_ref = {'trainer': None}

            # Adapt progress callback for hybrid trainer's ProgressUpdate
            def hybrid_progress_callback(update: ProgressUpdate):
                """Adapt hybrid ProgressUpdate to existing callback with extended metrics."""
                class ProgressAdapter:
                    pass
                adapter = ProgressAdapter()
                adapter.task_id = update.task_id
                adapter.category_name = update.category_name
                adapter.device_type = update.device_type
                adapter.current_epoch = update.current_epoch
                adapter.total_epochs = update.total_epochs
                adapter.train_loss = update.train_loss
                adapter.val_loss = update.val_loss
                adapter.f1_score = update.f1_score
                adapter.accuracy = update.accuracy
                adapter.phase = update.phase
                adapter.message = update.message
                adapter.current_batch = update.current_batch
                adapter.total_batches = update.total_batches
                # Extended metrics - samples info
                adapter.num_train_samples = update.num_train_samples
                adapter.num_val_samples = update.num_val_samples
                adapter.num_labels = update.num_labels
                # Extended metrics - per-class
                adapter.class_names = update.class_names
                adapter.per_class_f1 = update.per_class_f1
                adapter.per_class_precision = update.per_class_precision
                adapter.per_class_recall = update.per_class_recall
                adapter.per_class_support = update.per_class_support
                # Extended metrics - per-language
                adapter.per_language_f1 = update.per_language_f1
                adapter.per_language_accuracy = update.per_language_accuracy
                adapter.per_language_samples = update.per_language_samples
                progress_callback(adapter)

                # Update scheduler display with real-time metrics after each epoch validation
                if update.phase == "validation" and trainer_ref['trainer'] is not None:
                    try:
                        status = trainer_ref['trainer'].get_scheduler_status()
                        self.display.update_scheduler_info(
                            optimal_cpu_share=status.get('optimal_cpu_share'),
                            speed_ratio=status.get('speed_ratio'),
                            gpu_share=status.get('total_tasks', 0) - status.get('optimal_cpu_share', 0),
                        )
                    except Exception:
                        pass  # Don't break training for display updates

            # Pull batch sizes and worker settings from resource plan
            gpu_batch_size = plan.gpu_allocation.batch_size if plan.gpu_allocation else 128
            cpu_batch_size = plan.cpu_allocations[0].batch_size if plan.cpu_allocations else 16  # Reduced for lower RAM

            # GPU tuning for MPS: keep workers/prefetch modest to avoid unified memory spikes
            if plan.gpu_allocation:
                gpu_dl_workers = plan.gpu_allocation.num_workers
                gpu_prefetch = 2 if plan.gpu_allocation.device_id == "mps" else 4
                if plan.gpu_allocation.device_id == "mps":
                    gpu_dl_workers = min(gpu_dl_workers, 4)
                    gpu_batch_size = min(gpu_batch_size, 128)
            else:
                gpu_dl_workers = 8
                gpu_prefetch = 4

            cpu_dl_workers = plan.cpu_allocations[0].num_workers if plan.cpu_allocations else 2
            cpu_threads = plan.cpu_allocations[0].threads_per_worker if plan.cpu_allocations else max(2, (os.cpu_count() or 4) // max(1, num_cpu_workers))
            cpu_prefetch = 2

            # Create and run hybrid trainer with smart task scheduling
            hybrid_trainer = HybridParallelTrainer(
                model_name=self.config.model_name,
                num_cpu_workers=num_cpu_workers,
                gpu_batch_size=gpu_batch_size,
                cpu_batch_size=cpu_batch_size,
                gpu_dataloader_workers=gpu_dl_workers,
                cpu_dataloader_workers=cpu_dl_workers,
                cpu_threads_per_worker=cpu_threads,
                gpu_prefetch=gpu_prefetch,
                cpu_prefetch=cpu_prefetch,
                multi_label=self.config.multi_label,
                multi_label_threshold=self.config.multi_label_threshold,
                scheduler_config=self.config.scheduler_config,
            )

            logger.info(f"[HYBRID] Starting hybrid parallel training: GPU + {num_cpu_workers} CPU workers")
            # Note: Don't use console.print here - Live display is already active

            # Set trainer reference for dynamic scheduler updates in callback
            trainer_ref['trainer'] = hybrid_trainer

            # Pre-compute scheduler analysis for display (before train() blocks)
            try:
                from llm_tool.trainers.task_scheduler import SmartTaskScheduler
                temp_scheduler = SmartTaskScheduler(self.config.scheduler_config)
                pre_analysis = temp_scheduler.initialize_tasks(hybrid_tasks, num_cpu_workers)
                self.display.set_scheduler_info({
                    'gpu_share': pre_analysis.gpu_share,
                    'optimal_cpu_share': pre_analysis.optimal_cpu_share,
                    'speed_ratio': pre_analysis.speed_ratio,
                    'recommendation': pre_analysis.recommendation,
                })
            except Exception as e:
                logger.debug(f"Could not pre-compute scheduler analysis: {e}")

            hybrid_results = hybrid_trainer.train(hybrid_tasks, hybrid_progress_callback)

            # Convert results to expected format
            results_list = [
                {
                    'task_id': r.task_id,
                    'category': r.category_name,
                    'status': r.status,
                    'f1_score': r.f1_score,
                    'accuracy': r.accuracy,
                    'best_epoch': r.best_epoch,
                    'elapsed': r.elapsed_seconds,
                    'model_path': r.model_path,
                    'device': r.device_type if hasattr(r, "device_type") else ('gpu' if 'gpu' in r.task_id else 'cpu'),
                    'error': r.error_message if r.status == 'error' else None,
                }
                for r in hybrid_results
            ]

            # Process results
            for result in results_list:
                category = result.get("category", "")
                # Use actual task_id from result (worker includes it in return dict)
                # This is critical for per-language training where task_id format differs
                task_id = result.get("task_id", f"task_{category}")

                self.results[category] = result

                if result.get("status") == "success":
                    f1 = result.get("f1_score", 0)
                    acc = result.get("accuracy", 0)

                    self.display.mark_complete(task_id, f1, acc)

                    if self.metrics_logger:
                        self.metrics_logger.log_best(
                            model_id=task_id,
                            category=category,
                            device=result.get("device", "gpu"),
                            best_epoch=result.get("best_epoch", self.config.epochs),
                            best_f1_macro=f1,
                            best_accuracy=acc,
                            training_time=result.get("elapsed", 0),
                            total_epochs=self.config.epochs,
                            final_train_loss=0,
                            final_val_loss=0,
                            # Extended metadata for rationalized logging
                            huggingface_model=self.config.model_name,
                            model_class=self.config.model_name,
                            training_mode=self.config.training_approach,
                            language=result.get("language", ""),
                        )

                    if self.chart_manager:
                        self.chart_manager.finalize_model(task_id, result.get("elapsed", 0))
                else:
                    self.display.mark_error(task_id, result.get("error", "Unknown"))

            total_time = time.time() - start_time

            # Generate summary chart
            if self.chart_manager:
                self.chart_manager.generate_summary_chart(total_time, self.config.model_name)

            # Generate comprehensive cross-model summary chart
            try:
                from llm_tool.trainers.training_metrics_chart import generate_comprehensive_summary_chart

                _session_dir = self.config.logs_dir or self.config.output_dir
                comp_chart_path = generate_comprehensive_summary_chart(
                    output_dir=self.config.output_dir,
                    session_id=self.config.session_id,
                    model_name=self.config.model_name,
                    session_dir=_session_dir,
                    results_per_key=self.results,
                    training_approach=f'parallel-{self.config.training_approach}',
                    total_training_time=total_time,
                )
                if comp_chart_path:
                    logger.info(f"Comprehensive summary chart saved: {comp_chart_path}")
            except Exception as e:
                logger.warning(f"Failed to generate comprehensive summary chart: {e}")

            # ========== INTELLIGENT MODEL SELECTION ==========
            # Select best model using combined_metric (quality-based, no speed criteria)
            intelligent_selection = self._select_best_model_intelligent(self.results)
            if intelligent_selection:
                self.results['_intelligent_selection'] = intelligent_selection
                logger.info(f"🏆 Best model selected: {intelligent_selection.get('best_model')}")
                logger.info(f"   Selection method: {intelligent_selection.get('selection_method')}")

            # ========== FINAL SUMMARY ==========
            # Log where all outputs were saved (exclude metadata keys like '_intelligent_selection')
            training_results = {k: v for k, v in self.results.items() if not k.startswith('_')}
            success_count = sum(1 for r in training_results.values() if isinstance(r, dict) and r.get("status") == "success")
            error_count = sum(1 for r in training_results.values() if isinstance(r, dict) and r.get("status") == "error")

            # Build best model info for panel
            best_model_info = ""
            if intelligent_selection and intelligent_selection.get('best_model'):
                best_model_info = (
                    f"\n[bold]🏆 Best Model:[/bold]\n"
                    f"  [green]{intelligent_selection['best_model']}[/green]\n"
                    f"  [dim]Combined Score:[/dim] {intelligent_selection.get('best_combined_score', 0):.4f}\n"
                    f"  [dim]F1 Macro:[/dim] {intelligent_selection.get('best_f1_macro', 0):.4f}\n"
                )

            # Store summary panel for printing AFTER display.stop() to prevent duplication
            self._final_summary_panel = Panel(
                f"[bold green]✅ Training Complete[/bold green]\n\n"
                f"[cyan]Models trained:[/cyan] {success_count} success, {error_count} errors\n"
                f"[cyan]Total time:[/cyan] {self._format_time(total_time)}"
                f"{best_model_info}\n"
                f"[bold]📁 Output locations:[/bold]\n"
                f"  [dim]Models:[/dim] {self.models_dir}\n"
                f"  [dim]Logs:[/dim]   {self.logs_base_dir}\n"
                f"    ├── metrics/  (training.csv, best.csv, metrics.json)\n"
                f"    └── charts/   (per-model charts, summary chart)",
                title="[bold blue]PARALLEL TRAINING SUMMARY[/bold blue]",
                border_style="green",
            )

            return self.results

        finally:
            # Cleanup - FIRST stop display, THEN print summary
            self.display.stop()

            # Print summary panel AFTER Live display is stopped to prevent duplication
            if hasattr(self, '_final_summary_panel') and self._final_summary_panel:
                self.console.print()
                self.console.print(self._final_summary_panel)

            if self.metrics_logger:
                self.metrics_logger.stop()

    def _run_distributed(
        self,
        tasks: List[Any],
        plan: Any,
        num_cpu_workers: int,
        num_gpu_workers: int,
        start_time: float,
    ) -> Dict[str, Dict]:
        """
        Execute distributed training across local + remote machines.

        Uses the DistributedTrainingOrchestrator to:
        1. Create a distributed plan (local + remote task assignments)
        2. Transfer data to remote machines
        3. Run local + remote training in parallel
        4. Collect and aggregate all results

        Parameters
        ----------
        tasks : List
            All training tasks
        plan : ResourcePlan
            Local resource plan
        num_cpu_workers : int
            Number of local CPU workers
        num_gpu_workers : int
            Number of local GPU workers
        start_time : float
            Training start time

        Returns
        -------
        Dict[str, Dict]
            Aggregated results from all machines
        """
        import os
        from llm_tool.trainers.hybrid_parallel_trainer import (
            HybridParallelTrainer,
            TrainingTask as HybridTask,
            TaskResult,
        )

        orchestrator = self.config.distributed_orchestrator

        # Convert tasks to hybrid format
        hybrid_tasks = []
        for task in tasks:
            hybrid_task = HybridTask(
                task_id=task.task_id,
                category_name=task.category_name,
                model_name=task.model_name,
                data_path=task.data_path,
                output_path=task.output_path,
                config=task.config,
            )
            hybrid_tasks.append(hybrid_task)

        # Create distributed plan
        dist_plan = orchestrator.create_distributed_plan(hybrid_tasks, plan)
        orchestrator.display_distributed_plan(dist_plan)

        # In GPU-only mode, force 0 CPU workers for local training
        local_cpu_workers = 0 if dist_plan.gpu_only_mode else num_cpu_workers

        # Create local trainer factory
        def local_trainer_factory(num_cpu_workers=local_cpu_workers):
            gpu_batch_size = plan.gpu_allocation.batch_size if plan.gpu_allocation else 128
            cpu_batch_size = plan.cpu_allocations[0].batch_size if plan.cpu_allocations else 16
            gpu_dl_workers = plan.gpu_allocation.num_workers if plan.gpu_allocation else 8
            cpu_dl_workers = plan.cpu_allocations[0].num_workers if plan.cpu_allocations else 2
            cpu_threads = plan.cpu_allocations[0].threads_per_worker if plan.cpu_allocations else max(2, (os.cpu_count() or 4) // max(1, num_cpu_workers or 1))

            return HybridParallelTrainer(
                model_name=self.config.model_name,
                num_cpu_workers=num_cpu_workers,
                gpu_batch_size=gpu_batch_size,
                cpu_batch_size=cpu_batch_size,
                gpu_dataloader_workers=gpu_dl_workers,
                cpu_dataloader_workers=cpu_dl_workers,
                cpu_threads_per_worker=cpu_threads,
                multi_label=self.config.multi_label,
                multi_label_threshold=self.config.multi_label_threshold,
                scheduler_config=self.config.scheduler_config,
            )

        # Execute distributed training
        local_results, remote_assignments = orchestrator.execute(
            dist_plan,
            progress_callback=None,
            local_trainer_factory=local_trainer_factory,
        )

        # Process local results
        for r in local_results:
            category = r.category_name
            self.results[category] = {
                'task_id': r.task_id,
                'category': r.category_name,
                'status': r.status,
                'f1_score': r.f1_score,
                'accuracy': r.accuracy,
                'best_epoch': r.best_epoch,
                'elapsed': r.elapsed_seconds,
                'model_path': r.model_path,
                'device': r.device_type,
                'error': r.error_message if r.status == 'error' else None,
            }
            if r.status == "success":
                self.display.mark_complete(r.task_id, r.f1_score, r.accuracy)
            else:
                self.display.mark_error(r.task_id, r.error_message or "Unknown")

        # Process remote results
        for assignment in remote_assignments:
            category = assignment.category_name
            remote_host = assignment.remote_machine.hostname
            device_type = "remote_gpu" if assignment.remote_machine.has_gpu else "remote_cpu"

            if assignment.status == "completed":
                self.results[category] = {
                    'task_id': assignment.task_id,
                    'category': category,
                    'status': 'success',
                    'f1_score': assignment.result_f1,
                    'accuracy': assignment.result_accuracy,
                    'best_epoch': assignment.result_best_epoch,
                    'elapsed': assignment.result_elapsed,
                    'model_path': assignment.result_model_path,
                    'device': device_type,
                    'remote_host': remote_host,
                    'error': None,
                }
                self.display.mark_complete(assignment.task_id, assignment.result_f1, assignment.result_accuracy)

                if self.metrics_logger:
                    self.metrics_logger.log_best(
                        model_id=assignment.task_id,
                        category=category,
                        device=device_type,
                        best_epoch=assignment.result_best_epoch,
                        best_f1_macro=assignment.result_f1,
                        best_accuracy=assignment.result_accuracy,
                        training_time=assignment.result_elapsed,
                        total_epochs=self.config.epochs,
                        final_train_loss=0,
                        final_val_loss=0,
                        huggingface_model=self.config.model_name,
                        model_class=self.config.model_name,
                        training_mode=self.config.training_approach,
                    )
            else:
                self.results[category] = {
                    'task_id': assignment.task_id,
                    'category': category,
                    'status': 'error',
                    'f1_score': 0.0,
                    'accuracy': 0.0,
                    'device': device_type,
                    'remote_host': remote_host,
                    'error': assignment.error_message or "Remote training failed",
                }
                self.display.mark_error(assignment.task_id, assignment.error_message or "Remote failed")

        total_time = time.time() - start_time

        # Generate summary chart
        if self.chart_manager:
            self.chart_manager.generate_summary_chart(total_time, self.config.model_name)

        # Intelligent model selection
        intelligent_selection = self._select_best_model_intelligent(self.results)
        if intelligent_selection:
            self.results['_intelligent_selection'] = intelligent_selection

        # Build summary panel
        training_results = {k: v for k, v in self.results.items() if not k.startswith('_')}
        success_count = sum(1 for r in training_results.values() if isinstance(r, dict) and r.get("status") == "success")
        error_count = sum(1 for r in training_results.values() if isinstance(r, dict) and r.get("status") == "error")
        remote_count = sum(1 for r in training_results.values() if isinstance(r, dict) and r.get("remote_host"))

        best_model_info = ""
        if intelligent_selection and intelligent_selection.get('best_model'):
            best_model_info = (
                f"\n[bold]Best Model:[/bold]\n"
                f"  [green]{intelligent_selection['best_model']}[/green]\n"
                f"  [dim]F1 Macro:[/dim] {intelligent_selection.get('best_f1_macro', 0):.4f}\n"
            )

        from rich.panel import Panel
        self._final_summary_panel = Panel(
            f"[bold green]Training Distribue Termine[/bold green]\n\n"
            f"[cyan]Modeles entraines:[/cyan] {success_count} succes, {error_count} erreurs\n"
            f"[cyan]Machines:[/cyan] {dist_plan.total_machines} ({remote_count} taches distantes)\n"
            f"[cyan]Mode:[/cyan] {'GPU-Only' if dist_plan.gpu_only_mode else 'Mixte (GPU+CPU)'}\n"
            f"[cyan]Temps total:[/cyan] {self._format_time(total_time)}"
            f"{best_model_info}\n"
            f"[bold]Sorties:[/bold]\n"
            f"  [dim]Modeles:[/dim] {self.models_dir}\n"
            f"  [dim]Logs:[/dim]   {self.logs_base_dir}",
            title="[bold blue]DISTRIBUTED TRAINING SUMMARY[/bold blue]",
            border_style="green",
        )

        return self.results

    def _select_best_model_intelligent(self, all_results: Dict[str, Dict], languages: List[str] = None) -> Dict:
        """
        Intelligent model selection after parallel training.
        Uses combined_metric with language balance penalty.
        NOTE: NO SPEED CRITERIA - selection based on quality only.

        Args:
            all_results: Dict of model results keyed by model_id
            languages: Optional list of languages to consider for language balance

        Returns:
            Dict with best_model, best_combined_score, ranking, etc.
        """
        if not all_results:
            return {}

        try:
            from llm_tool.utils.benchmark_utils import compare_model_results

            # Use sophisticated ranking (quality-based, no speed)
            comparison_df = compare_model_results(
                all_results,
                f1_class_1_weight=0.7,
                use_sophisticated_ranking=True
            )

            if comparison_df.empty:
                logger.warning("No valid results for intelligent selection")
                return {}

            # Get best model (already sorted by combined_score, no speed)
            best_row = comparison_df.iloc[0]
            best_model_id = best_row['model']

            # Log all languages performance
            logger.info(f"🏆 Intelligent selection: {best_model_id}")
            logger.info(f"   Combined score: {best_row['combined_score']:.4f}")
            logger.info(f"   F1 Macro: {best_row['f1_macro']:.4f}")

            lang_penalty = best_row.get('language_balance_penalty', 0)
            if lang_penalty and lang_penalty > 0:
                logger.info(f"   Language penalty: {lang_penalty:.1%}")
                logger.info(f"   (Penalty applied for imbalanced performance across languages)")

            return {
                'best_model': best_model_id,
                'best_combined_score': float(best_row['combined_score']),
                'best_f1_macro': float(best_row['f1_macro']),
                'best_accuracy': float(best_row['accuracy']),
                'ranking': comparison_df.to_dict('records'),
                'selection_method': 'intelligent_combined_metric_no_speed'
            }

        except ImportError:
            logger.warning("benchmark_utils not available, using simple selection")
            # Fallback: simple selection by f1_macro
            best_model = None
            best_f1 = 0
            for model_id, result in all_results.items():
                if isinstance(result, dict) and result.get('status') == 'success':
                    f1 = result.get('f1_macro', result.get('best_f1_macro', 0))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model_id
            return {
                'best_model': best_model,
                'best_f1_macro': best_f1,
                'selection_method': 'simple_f1_macro'
            }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_parallel_training_with_logging(
    training_files: Dict[str, str],
    model_name: str = "xlm-roberta-base",
    epochs: int = 10,
    output_dir: str = "models",
    logs_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    console: Optional[Console] = None,
) -> Dict[str, Dict]:
    """
    Convenience function for parallel training with full logging.

    Parameters
    ----------
    training_files : Dict[str, str]
        Mapping of category names to training file paths
    model_name : str
        Model to use for training
    epochs : int
        Number of epochs
    output_dir : str
        Output directory for trained models
    logs_dir : str, optional
        Directory for logs (default: logs/training_arena/{session_id}/parallel_training/)
    session_id : str, optional
        Session identifier
    console : Console, optional
        Rich console

    Returns
    -------
    Dict[str, Dict]
        Training results
    """
    config = ParallelTrainingConfig(
        model_name=model_name,
        epochs=epochs,
        output_dir=output_dir,
        logs_dir=logs_dir,
        session_id=session_id or f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        enable_charts=True,
        enable_logging=True,
    )

    manager = ParallelTrainingManager(config, console)
    return manager.run(training_files)

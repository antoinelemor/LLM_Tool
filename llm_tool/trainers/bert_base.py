#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
bert_base.py

MAIN OBJECTIVE:
---------------
Implement the core BertBase trainer with encoding, training, evaluation,
reinforced learning, logging, and device management for transformer models.

Dependencies:
-------------
- datetime
- random
- time
- os
- shutil
- csv
- copy
- json
- warnings
- typing
- collections
- pathlib
- numpy
- torch
- scipy
- sklearn
- transformers
- tqdm
- rich
- colorama
- tabulate
- llm_tool.utils.training_paths
- llm_tool.trainers.bert_abc
- llm_tool.utils.logging_utils

MAIN FEATURES:
--------------
1) Detect hardware capabilities and configure optimised training pipelines
2) Encode text into TensorDatasets with attention masks and metadata
3) Train models with learning rate scheduling, logging, and checkpointing
4) Trigger reinforced learning loops when class metrics fall below thresholds
5) Generate comprehensive progress dashboards and export metrics to disk

Author:
-------
Antoine Lemor
"""

import datetime
import random
import time
import os
import shutil
import csv
import copy
import json
import warnings
import inspect
import re
from typing import List, Tuple, Any, Optional, Dict
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch.types import Device
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from colorama import init, Fore, Back, Style
from tabulate import tabulate
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.console import Group, Console
from rich.control import Control
from rich.text import Text
from rich import box

from llm_tool.utils.training_paths import resolve_metrics_base_dir, get_session_dir

# Create a shared console for Rich operations
console = Console()

# Initialize colorama for cross-platform colored output
init(autoreset=True)
try:                             # transformers >= 5
    from torch.optim import AdamW
except ImportError:              # transformers <= 4
    from transformers.optimization import AdamW
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME
)
import transformers
# Suppress transformers warnings globally
transformers.logging.set_verbosity_error()

from llm_tool.trainers.bert_abc import BertABC
from llm_tool.utils.logging_utils import get_logger

# Import language filtering utilities
try:
    from llm_tool.trainers.model_trainer import (
        MODEL_TARGET_LANGUAGES,
        get_model_target_languages,
        filter_data_by_language
    )
    HAS_LANGUAGE_FILTERING = True
except ImportError:
    HAS_LANGUAGE_FILTERING = False
    # Fallback function
    def get_model_target_languages(model_name):
        return None


# ================== Rich Live Update Throttling Configuration ==================
# Configuration for throttling Rich Live updates to prevent terminal saturation
# This is critical for preventing VS Code crashes when training on GPU

# Default throttle interval in seconds (0.25 = 4 Hz max)
DEFAULT_UPDATE_THROTTLE = 0.25
# When we auto-detect an Electron / VS Code terminal we fall back to a much slower cadence
DEFAULT_VSCODE_UPDATE_THROTTLE = 60.0  # once per minute unless the user overrides
DEFAULT_VSCODE_MIN_THROTTLE = 2.0      # never spam updates faster than this inside VS Code
DEFAULT_TERMINAL_CLEAR_INTERVAL = 600.0  # seconds; periodic screen clear to keep buffers bounded


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, allow_zero: bool = False) -> float:
    """Parse an environment variable as float with robust fallbacks."""
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


def _env_int(name: str, default: int) -> int:
    """Parse an environment variable as integer with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _running_inside_vscode_terminal() -> bool:
    """Best-effort detection of VS Code / Electron based terminals."""
    if os.environ.get("TERM_PROGRAM", "").lower() == "vscode":
        return True
    for key in ("VSCODE_PID", "VSCODE_CWD", "VSCODE_IPC_HOOK", "VSCODE_INJECTION"):
        if os.environ.get(key):
            return True
    return False


# Read throttle configuration from environment
# LLM_TOOL_UPDATE_THROTTLE: Min seconds between updates (default: 0.25)
# LLM_TOOL_DISABLE_LIVE_UPDATE: Set to 'true' to disable live updates completely
# LLM_TOOL_FORCE_UPDATE_INTERVAL: Force updates every N batches regardless of throttle
# LLM_TOOL_TERMINAL_CLEAR_INTERVAL: Seconds between hard clears (<=0 disables)
UPDATE_THROTTLE_USER_OVERRIDE = "LLM_TOOL_UPDATE_THROTTLE" in os.environ
UPDATE_THROTTLE = _env_float("LLM_TOOL_UPDATE_THROTTLE", DEFAULT_UPDATE_THROTTLE)
DISABLE_LIVE_UPDATE = _env_flag("LLM_TOOL_DISABLE_LIVE_UPDATE")
FORCE_UPDATE_INTERVAL = _env_int("LLM_TOOL_FORCE_UPDATE_INTERVAL", 0)
TERMINAL_CLEAR_INTERVAL = _env_float(
    "LLM_TOOL_TERMINAL_CLEAR_INTERVAL",
    DEFAULT_TERMINAL_CLEAR_INTERVAL,
    allow_zero=True,
)
if TERMINAL_CLEAR_INTERVAL <= 0:
    TERMINAL_CLEAR_INTERVAL = None

# Detect if we're running in VS Code terminal (higher risk of crashes)
IS_VSCODE_TERMINAL = _running_inside_vscode_terminal()
if IS_VSCODE_TERMINAL:
    vscode_default = _env_float("LLM_TOOL_VSCODE_SAFE_THROTTLE", DEFAULT_VSCODE_UPDATE_THROTTLE)
    vscode_min = _env_float("LLM_TOOL_VSCODE_MIN_THROTTLE", DEFAULT_VSCODE_MIN_THROTTLE)

    if not UPDATE_THROTTLE_USER_OVERRIDE:
        UPDATE_THROTTLE = vscode_default
        if console.is_terminal:
            console.log(
                f"[cyan]VS Code terminal detected – limiting live refresh to every {UPDATE_THROTTLE:.0f}s "
                "(override with LLM_TOOL_VSCODE_SAFE_THROTTLE or LLM_TOOL_UPDATE_THROTTLE).[/cyan]"
            )
    elif UPDATE_THROTTLE < vscode_min:
        UPDATE_THROTTLE = vscode_min
        if console.is_terminal:
            console.log(
                f"[yellow]LLM_TOOL_UPDATE_THROTTLE below VS Code safety floor; "
                f"using {UPDATE_THROTTLE:.1f}s minimum interval.[/yellow]"
            )


# Configure Live rendering strategy based on Rich version support
_live_signature = inspect.signature(Live.__init__)
LIVE_BASE_KWARGS = {"transient": True}
_manual_refresh_rate = 1 / 60  # ~1 refresh per minute if auto refresh is enabled internally
if "auto_refresh" in _live_signature.parameters:
    LIVE_BASE_KWARGS.update({"auto_refresh": False, "refresh_per_second": _manual_refresh_rate})
else:
    LIVE_BASE_KWARGS.update({"refresh_per_second": _manual_refresh_rate})


class ThrottledLiveUpdater:
    """
    Throttles Rich Live updates to prevent terminal saturation.

    This class wraps a Rich Live object and limits the frequency of updates
    to prevent crashes in terminals that can't handle high-frequency refreshes,
    particularly VS Code's integrated terminal when training on GPU.

    Attributes:
        live: The Rich Live object to wrap
        min_interval: Minimum seconds between updates (default from UPDATE_THROTTLE)
        batch_interval: Update every N batches regardless of time (0 = disabled)
        disabled: If True, no updates are performed
        last_update: Timestamp of last update
        batch_count: Counter for batch-based updates
        pending_update: Stores the last panel if an update was skipped
        force_next: Force the next update regardless of throttling
    """

    def __init__(self, live, min_interval=None, batch_interval=None, disabled=None, auto_clear_interval=None):
        """
        Initialize the throttled updater.

        Args:
            live: Rich Live object to wrap
            min_interval: Min seconds between updates (default: UPDATE_THROTTLE)
            batch_interval: Update every N batches (default: FORCE_UPDATE_INTERVAL)
            disabled: Disable all updates (default: DISABLE_LIVE_UPDATE)
        """
        self.live = live
        self.min_interval = min_interval if min_interval is not None else UPDATE_THROTTLE
        self.batch_interval = batch_interval if batch_interval is not None else FORCE_UPDATE_INTERVAL
        self.disabled = disabled if disabled is not None else DISABLE_LIVE_UPDATE

        if auto_clear_interval is not None:
            self.auto_clear_interval = auto_clear_interval if auto_clear_interval > 0 else None
        else:
            self.auto_clear_interval = TERMINAL_CLEAR_INTERVAL

        self.last_update = 0.0
        self.batch_count = 0
        self.pending_update = None
        self.force_next = False

        # Statistics for debugging
        self.updates_performed = 0
        self.updates_skipped = 0
        self.clear_operations = 0
        self.housekeeping_failures = 0

        # Timing information
        self.last_clear_time = time.time()
        self._last_frame_time = None
        self._interval_samples: deque[float] = deque(maxlen=240)

    def update(self, panel, force=False, is_batch=False):
        """
        Update the live display with throttling.

        Args:
            panel: The panel to display
            force: Force this update regardless of throttling (for important events)
            is_batch: True if this is a per-batch update (enables batch counting)

        Returns:
            bool: True if update was performed, False if skipped
        """
        # If updates are disabled, skip everything
        if self.disabled and not force:
            self.updates_skipped += 1
            return False

        # Increment batch counter if this is a batch update
        if is_batch:
            self.batch_count += 1

        # Check if we should force this update
        should_update = force or self.force_next

        now = time.time()

        # Check time-based throttling
        if not should_update:
            time_elapsed = now - self.last_update
            should_update = time_elapsed >= self.min_interval

        # Check batch-based forcing
        if not should_update and self.batch_interval > 0 and is_batch:
            should_update = self.batch_count % self.batch_interval == 0

        # Perform or skip the update
        if should_update:
            self._apply_terminal_housekeeping(now)
            self.live.update(panel)
            self.last_update = now
            self.force_next = False
            self.pending_update = None
            self.updates_performed += 1
            self._record_interval(now)
            return True
        else:
            # Store pending update to show later
            self.pending_update = panel
            self.updates_skipped += 1
            return False

    def force_next_update(self):
        """Mark the next update to be forced regardless of throttling."""
        self.force_next = True

    def flush_pending(self):
        """Force display of any pending update."""
        if self.pending_update is not None:
            now = time.time()
            self._apply_terminal_housekeeping(now)
            self.live.update(self.pending_update)
            self.pending_update = None
            self.last_update = now
            self.updates_performed += 1
            self._record_interval(now)

    def get_stats(self):
        """Get statistics about throttling performance."""
        total = self.updates_performed + self.updates_skipped
        if total == 0:
            return f"no updates (min_interval={self.min_interval:.1f}s)"

        skip_rate = (self.updates_skipped / total) * 100
        parts = [
            f"performed={self.updates_performed}",
            f"skipped={self.updates_skipped}",
            f"throttled={skip_rate:.1f}%",
            f"min_interval={self.min_interval:.1f}s",
        ]

        if self._interval_samples:
            avg_interval = sum(self._interval_samples) / len(self._interval_samples)
            parts.append(f"avg_interval={avg_interval:.1f}s")

        if self.auto_clear_interval:
            parts.append(f"auto_clear={self.auto_clear_interval:.0f}s")
            if self.clear_operations:
                parts.append(f"clears={self.clear_operations}")
            if self.housekeeping_failures:
                parts.append(f"clear_failures={self.housekeeping_failures}")

        return ", ".join(parts)

    def __enter__(self):
        """Context manager entry - delegate to wrapped Live object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush pending updates."""
        self.flush_pending()
        # Note: The actual Live object's __exit__ is called by the with statement

    def _apply_terminal_housekeeping(self, now: float) -> None:
        """Occasionally clear the terminal buffer to avoid unbounded growth."""
        if not self.auto_clear_interval:
            return
        console_obj = getattr(self.live, "console", None)
        if not console_obj or not getattr(console_obj, "is_terminal", False):
            return
        if (now - self.last_clear_time) < self.auto_clear_interval:
            return

        try:
            console_obj.control(Control("erase", "scroll"))
            console_obj.control(Control("erase", "screen"))
            console_obj.control(Control("home"))
            self.clear_operations += 1
        except Exception:
            # If the terminal rejects the escape sequence don't spam retries
            self.housekeeping_failures += 1
        finally:
            # Avoid repeatedly attempting to clear in a tight loop when an error occurs
            self.last_clear_time = now

    def _record_interval(self, now: float) -> None:
        """Track observed intervals between rendered frames for telemetry."""
        if self._last_frame_time is not None:
            self._interval_samples.append(now - self._last_frame_time)
        self._last_frame_time = now


class TrainingDisplay:
    """Rich Live Display for training progress with all metrics."""

    def __init__(self, model_name: str, label_key: str = None, label_value: str = None,
                 language: str = None, n_epochs: int = 10, is_reinforced: bool = False,
                 num_labels: int = 2, class_names: list = None, detected_languages: list = None,
                 global_total_models: int = None, global_current_model: int = None,
                 global_total_epochs: int = None, global_max_epochs: int = None,
                 global_completed_epochs: int = None,
                 global_start_time: float = None, reinforced_learning_enabled: bool = False):
        self.model_name = model_name
        self.label_key = label_key
        self.label_value = label_value
        self.language = language
        self.detected_languages = detected_languages or []  # Store detected languages (e.g., ['FR', 'EN'])
        self.n_epochs = n_epochs
        self.is_reinforced = is_reinforced
        self.num_labels = num_labels
        self.reinforced_learning_enabled = reinforced_learning_enabled  # Whether reinforced learning is enabled globally

        # Global progress tracking (for benchmark and multi-model training)
        self.global_total_models = global_total_models
        self.global_current_model = global_current_model
        self.global_total_epochs = global_total_epochs
        self.global_max_epochs = global_max_epochs or global_total_epochs  # Default to global_total_epochs if not provided
        self.global_completed_epochs = global_completed_epochs or 0
        self.global_start_time = global_start_time

        # Class names for display (no truncation - let table handle width)
        if class_names:
            # Multi-class or binary with distinct values: use provided class names as-is
            # Ensure all class names are strings
            self.class_names = [str(name) for name in class_names]
        elif label_value and num_labels == 2:
            # Binary with label value but no class names: Class 0 = NOT_category, Class 1 = category
            # This is for true presence/absence classification
            self.class_names = [f"NOT_{label_value}", str(label_value)]
        else:
            # Default: Class 0, Class 1, etc.
            self.class_names = [f"Class {i}" for i in range(num_labels)]

        # For backward compatibility
        if num_labels == 2:
            self.class_0_name = self.class_names[0]
            self.class_1_name = self.class_names[1]

        # Metrics storage
        self.current_epoch = 0
        self.current_phase = "Initializing"
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.train_progress = 0
        self.val_progress = 0
        self.train_total = 0
        self.val_total = 0

        # Performance metrics (initialized for num_labels classes)
        self.accuracy = 0.0
        self.precision = [0.0] * num_labels
        self.recall = [0.0] * num_labels
        self.f1_scores = [0.0] * num_labels
        self.f1_macro = 0.0
        self.support = [0] * num_labels

        # Per-language metrics
        self.language_metrics = {}

        # Best model tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.improvement = 0.0
        self.combined_metric = 0.0  # Weighted score for model selection
        self.reinforced_threshold = 0.0  # Threshold to trigger reinforced learning
        self.reinforced_triggered = False  # Whether reinforced learning was triggered
        self.language_variance = 0.0  # Coefficient of variation for F1 class 1 across languages

        # Timing
        self.train_time = 0.0
        self.val_time = 0.0
        self.epoch_time = 0.0
        self.total_time = 0.0
        self.start_time = time.time()

    def create_header(self) -> Table:
        """Create header with model and label info."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column(style="bold white")

        if self.is_reinforced:
            table.add_row("🔄 REINFORCED TRAINING", "")
        else:
            table.add_row("🏋️ MODEL TRAINING", "")

        table.add_row("Model:", self.model_name)

        if self.label_key and self.label_value:
            table.add_row("Label Key:", self.label_key)
            table.add_row("Label Value:", self.label_value)

        # Display languages - ALWAYS show if we have any language info
        if self.detected_languages and len(self.detected_languages) > 0:
            # Show all detected languages
            lang_display = ", ".join(self.detected_languages)
            table.add_row("Languages:", lang_display)
        elif self.language:
            # Fallback to single language code or MULTI
            if self.language == "MULTI":
                table.add_row("Languages:", "Multilingual")
            else:
                table.add_row("Language:", self.language)

        return table

    def create_global_progress_section(self) -> Table:
        """Create global progress section showing overall training progress across all models."""
        if self.global_total_models is None:
            return None

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold yellow", width=20)
        table.add_column(style="bold white")

        # Check if reinforced learning is enabled globally (for "minimum" indicators)
        reinforced_enabled = self.reinforced_learning_enabled

        # Global header
        table.add_row("🌍 GLOBAL PROGRESS", "")

        # Display languages tracked for this session (if known)
        language_display = None
        if self.detected_languages:
            # Preserve order but avoid duplicates
            unique_langs = list(dict.fromkeys(lang.upper() for lang in self.detected_languages if isinstance(lang, str)))
            if unique_langs:
                language_display = ", ".join(unique_langs)
        elif self.language:
            language_display = "Multilingual" if self.language == "MULTI" else self.language

        if language_display:
            table.add_row("Languages:", language_display)

        # Models progress
        if self.global_current_model is not None and self.global_total_models is not None:
            try:
                current_models = max(int(self.global_current_model), 0)
            except (TypeError, ValueError):
                current_models = 0
            try:
                total_models = int(self.global_total_models)
            except (TypeError, ValueError):
                total_models = 0
            total_models = max(total_models, current_models)
            model_pct = (current_models / total_models) * 100 if total_models > 0 else 0
            model_bar = self._create_bar(model_pct, width=40)
            table.add_row("Models:", f"{current_models}/{total_models} {model_bar}")

        # Total epochs progress (with maximum indicator if reinforced learning is enabled)
        if self.global_total_epochs is not None and self.global_completed_epochs is not None:
            # Calculate percentage based on actual completed vs current total
            epoch_pct = (self.global_completed_epochs / self.global_total_epochs) * 100 if self.global_total_epochs > 0 else 0
            # Cap at 100% for display purposes
            epoch_pct = min(epoch_pct, 100.0)
            epoch_bar = self._create_bar(epoch_pct, width=40)

            # Display format: Always show "(max X)" when maximum epochs are defined or reinforced learning is enabled
            epoch_label = f"{self.global_completed_epochs}/{self.global_total_epochs}"

            # Show maximum epochs in various scenarios:
            # 1. When reinforced learning is enabled and max epochs is defined and different
            # 2. When global_max_epochs exists and is different from global_total_epochs
            # 3. Always when reinforced learning is enabled (even if max not yet calculated)
            if self.global_max_epochs is not None and self.global_max_epochs != self.global_total_epochs:
                # We have a defined maximum that's different from current total
                epoch_label += f" (max {self.global_max_epochs})"
            elif reinforced_enabled and self.global_max_epochs is not None:
                # Reinforced learning is enabled and we have max epochs calculated
                epoch_label += f" (max {self.global_max_epochs})"
            elif reinforced_enabled:
                # Reinforced learning is enabled but max not yet calculated - show current total as minimum
                # This ensures we always show some indication when reinforced learning is active
                epoch_label += f" (max {self.global_total_epochs}+)"

            table.add_row("Total Epochs:", f"{epoch_label} {epoch_bar}")

        # Global timing
        if self.global_start_time is not None:
            elapsed = time.time() - self.global_start_time
            elapsed_str = self._format_time(elapsed)
            table.add_row("Elapsed Time:", elapsed_str)

            # Estimate remaining time based on completed epochs
            if self.global_completed_epochs > 0 and self.global_total_epochs > 0:
                avg_time_per_epoch = elapsed / self.global_completed_epochs
                remaining_epochs = self.global_total_epochs - self.global_completed_epochs
                estimated_remaining = avg_time_per_epoch * remaining_epochs
                remaining_str = self._format_time(estimated_remaining)

                # Add "min." indicator if reinforced learning might add more epochs
                time_suffix = ""
                if reinforced_enabled and self.global_completed_epochs < self.global_total_epochs:
                    time_suffix = " (minimum)"

                table.add_row("Est. Remaining:", f"{remaining_str}{time_suffix}")

                # Total estimated time
                total_estimated = elapsed + estimated_remaining
                total_str = self._format_time(total_estimated)
                table.add_row("Est. Total Time:", f"{total_str}{time_suffix}")

        return table

    def create_progress_section(self) -> Table:
        """Create progress bars for training and validation."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", width=12)
        table.add_column()

        # Epoch progress
        epoch_pct = (self.current_epoch / self.n_epochs) * 100 if self.n_epochs > 0 else 0
        epoch_bar = self._create_bar(epoch_pct, width=30)
        table.add_row("📊 Epoch:", f"{self.current_epoch}/{self.n_epochs} {epoch_bar}")

        # Phase
        table.add_row("🔄 Phase:", self.current_phase)

        # Training progress
        if self.train_total > 0:
            train_pct = (self.train_progress / self.train_total) * 100
            train_bar = self._create_bar(train_pct, width=30)
            table.add_row("📚 Training:", f"{self.train_progress}/{self.train_total} {train_bar}")

        # Validation progress
        if self.val_total > 0:
            val_pct = (self.val_progress / self.val_total) * 100
            val_bar = self._create_bar(val_pct, width=30)
            table.add_row("🔍 Validation:", f"{self.val_progress}/{self.val_total} {val_bar}")

        return table

    def create_metrics_table(self) -> Table:
        """Create table with all performance metrics."""
        table = Table(title="📈 Performance Metrics", box=box.ROUNDED, title_style="bold magenta")
        table.add_column("Metric", style="cyan", width=25)

        # Calculate dynamic column width based on longest class name
        # Min 15, max 35 to keep table readable
        # Ensure all class names are strings before calculating length
        max_class_name_length = max([len(str(name)) for name in self.class_names]) if self.class_names else 15
        class_col_width = min(max(max_class_name_length + 2, 15), 35)

        # Add columns for each class
        colors = ["yellow", "green", "magenta", "blue", "red", "cyan"]
        for i, class_name in enumerate(self.class_names):
            color = colors[i % len(colors)]
            # Ensure class_name is a string
            class_name_str = str(class_name)
            # Use calculated width or allow wrapping for very long names
            if len(class_name_str) > 35:
                # For very long names, allow wrapping
                table.add_column(class_name_str, justify="center", style=color, no_wrap=False, width=35)
            else:
                table.add_column(class_name_str, justify="center", style=color, width=class_col_width)

        table.add_column("Overall", justify="center", style="bold white", width=10)

        # Losses - empty cells for class columns
        loss_row = ["Train Loss"] + [""] * self.num_labels + [f"{self.train_loss:.4f}"]
        table.add_row(*loss_row)
        val_loss_row = ["Val Loss"] + [""] * self.num_labels + [f"{self.val_loss:.4f}"]
        table.add_row(*val_loss_row)
        separator_row = [""] * (self.num_labels + 2)
        table.add_row(*separator_row)

        # Metrics
        acc_row = ["Accuracy"] + [""] * self.num_labels + [f"{self.accuracy:.3f}"]
        table.add_row(*acc_row)

        precision_row = ["Precision"] + [f"{self.precision[i]:.3f}" for i in range(self.num_labels)] + [""]
        table.add_row(*precision_row)

        recall_row = ["Recall"] + [f"{self.recall[i]:.3f}" for i in range(self.num_labels)] + [""]
        table.add_row(*recall_row)

        f1_row = ["F1-Score"] + [f"{self.f1_scores[i]:.3f}" for i in range(self.num_labels)] + [f"{self.f1_macro:.3f}"]
        table.add_row(*f1_row)

        support_row = ["Support"] + [str(int(self.support[i])) for i in range(self.num_labels)] + [str(int(sum(self.support)))]
        table.add_row(*support_row)

        # ALWAYS add per-language metrics section if we have language info
        # Show if:
        # 1. detected_languages has ANY language (1+)
        # 2. OR language is set (including "MULTI")
        should_show_language_metrics = (
            (self.detected_languages and len(self.detected_languages) >= 1) or
            (self.language and self.language != "")
        )

        if should_show_language_metrics:
            # Add separator
            separator_row = [""] * (self.num_labels + 2)
            table.add_row(*separator_row)

            # Add language header
            lang_header_row = ["─── 🌍 Per Language ───"] + [""] * (self.num_labels + 1)
            table.add_row(*lang_header_row, style="bold cyan")

            if self.language_metrics:
                # Show metrics for each detected language
                for lang in sorted(self.detected_languages):
                    if lang in self.language_metrics:
                        lang_metrics = self.language_metrics[lang]

                        # CRITICAL: Include language name in each metric row for clarity
                        lang_upper = lang.upper()

                        # Add separator before each language
                        separator_row = [""] * (self.num_labels + 2)
                        table.add_row(*separator_row)

                        # Add accuracy with language name
                        lang_acc_row = [f"  [{lang_upper}] Accuracy"] + [""] * self.num_labels + [f"{lang_metrics.get('accuracy', 0):.3f}"]
                        table.add_row(*lang_acc_row)

                        # Add F1 scores per class with language name
                        lang_f1_row = [f"  [{lang_upper}] F1-Score"] + [f"{lang_metrics.get(f'f1_{i}', 0):.3f}" for i in range(self.num_labels)] + [f"{lang_metrics.get('macro_f1', 0):.3f}"]
                        table.add_row(*lang_f1_row)

                        # Add support with language name
                        lang_support_row = [f"  [{lang_upper}] Support"] + [str(int(lang_metrics.get(f'support_{i}', 0))) for i in range(self.num_labels)] + [str(lang_metrics.get('samples', 0))]
                        table.add_row(*lang_support_row)
            else:
                # No metrics yet - show waiting message
                wait_row = [f"  Detected: {', '.join(sorted(self.detected_languages))}"] + [""] * (self.num_labels + 1)
                table.add_row(*wait_row, style="dim")
                wait_row2 = ["  (metrics will appear after first validation)"] + [""] * (self.num_labels + 1)
                table.add_row(*wait_row2, style="dim italic")

        return table

    def create_language_table(self) -> Table:
        """Create table with per-language metrics."""
        if not self.language_metrics:
            return None

        table = Table(title="🌍 Per-Language Performance", box=box.ROUNDED, title_style="bold magenta")
        table.add_column("Language", style="cyan", width=15)

        # Calculate dynamic column width based on longest class name
        # Ensure all class names are strings before calculating length
        max_class_name_length = max([len(str(name)) for name in self.class_names]) if self.class_names else 15
        class_col_width = min(max(max_class_name_length + 2, 15), 35)

        # Add support columns for each class
        colors = ["yellow", "green", "magenta", "blue", "red", "cyan"]
        for i, class_name in enumerate(self.class_names):
            color = colors[i % len(colors)]
            # Ensure class_name is a string
            class_name_str = str(class_name)
            col_header = f"Sup {class_name_str}"
            if len(col_header) > 35:
                table.add_column(col_header, justify="center", style=color, no_wrap=False, width=35)
            else:
                table.add_column(col_header, justify="center", style=color, width=class_col_width)

        table.add_column("Accuracy", justify="center", width=10)

        # Add F1 columns for each class
        for i, class_name in enumerate(self.class_names):
            color = colors[i % len(colors)]
            # Ensure class_name is a string
            class_name_str = str(class_name)
            col_header = f"F1 {class_name_str}"
            if len(col_header) > 35:
                table.add_column(col_header, justify="center", style=color, no_wrap=False, width=35)
            else:
                table.add_column(col_header, justify="center", style=color, width=class_col_width)

        table.add_column("Macro F1", justify="center", style="bold", width=10)

        for lang, metrics in sorted(self.language_metrics.items()):
            row = [lang]

            # Support for each class
            for i in range(self.num_labels):
                row.append(str(metrics.get(f'support_{i}', 0)))

            # Accuracy
            row.append(f"{metrics.get('accuracy', 0):.3f}")

            # F1 for each class
            for i in range(self.num_labels):
                row.append(f"{metrics.get(f'f1_{i}', 0):.3f}")

            # Macro F1
            row.append(f"{metrics.get('macro_f1', 0):.3f}")

            table.add_row(*row)

        # Add average row (only count languages with actual support)
        if len(self.language_metrics) > 1:
            # Only average languages that have at least some samples
            valid_metrics = [m for m in self.language_metrics.values()
                           if sum(m.get(f'support_{i}', 0) for i in range(self.num_labels)) > 0]

            if valid_metrics:
                avg_acc = sum(m.get('accuracy', 0) for m in valid_metrics) / len(valid_metrics)
                avg_f1 = sum(m.get('macro_f1', 0) for m in valid_metrics) / len(valid_metrics)

                # Create separator row
                separator = ["─" * 8] + ["─" * 5] * self.num_labels + ["─" * 8] + ["─" * 5] * self.num_labels + ["─" * 8]
                table.add_row(*separator)

                # Create average row
                avg_row = ["AVERAGE"] + [""] * self.num_labels + [f"{avg_acc:.3f}"] + [""] * self.num_labels + [f"{avg_f1:.3f}"]
                table.add_row(*avg_row, style="bold")

        return table

    def create_summary_section(self) -> Table:
        """Create summary with timing and best model info."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", width=22, no_wrap=True)  # Wide enough for emoji labels
        table.add_column(style="white")

        # Timing
        table.add_row("⏱️ Train Time:", self._format_time(self.train_time))
        table.add_row("⏱️ Val Time:", self._format_time(self.val_time))
        table.add_row("⏱️ Epoch Time:", self._format_time(self.epoch_time))
        table.add_row("⏱️ Total Time:", self._format_time(self.total_time))

        # Best model
        if self.best_f1 > 0:
            table.add_row("", "")
            table.add_row("🏆 Best F1:", f"{self.best_f1:.4f} (Epoch {self.best_epoch})")
            if self.improvement != 0:
                sign = "+" if self.improvement > 0 else ""
                table.add_row("📈 Improvement:", f"{sign}{self.improvement:.4f}")

            # Show combined metric (weighted score) if available
            if self.combined_metric > 0:
                table.add_row("⚖️ Combined Score:", f"{self.combined_metric:.4f}")

            # Show reinforced learning info if relevant
            if self.reinforced_threshold > 0:
                table.add_row("🎯 RL Threshold:", f"{self.reinforced_threshold:.4f}")
                if self.reinforced_triggered:
                    table.add_row("🔥 RL Triggered:", "Yes", style="bold yellow")

            # Show language variance if available (for multilingual models)
            if self.language_variance > 0:
                # Color code based on variance level
                variance_style = "green" if self.language_variance < 0.3 else ("yellow" if self.language_variance < 0.7 else "red")
                table.add_row("📊 Lang Variance:", f"{self.language_variance:.3f}", style=variance_style)

        return table

    def create_panel(self) -> Panel:
        """Create the complete panel with all sections."""
        sections = []

        # Add global progress section at the top if available
        global_section = self.create_global_progress_section()
        if global_section:
            sections.append(global_section)
            sections.append(Text())  # Spacer
            sections.append(Text("─" * 80, style="dim"))  # Separator line
            sections.append(Text())  # Spacer

        sections.append(self.create_header())
        sections.append(Text())  # Spacer
        sections.append(self.create_progress_section())
        sections.append(Text())  # Spacer
        sections.append(self.create_metrics_table())

        # ALWAYS add language table if we have detected languages or language metrics
        # This ensures language performance is ALWAYS visible for multilingual models
        lang_table = self.create_language_table()
        if lang_table or self.detected_languages:
            sections.append(Text())  # Spacer
            if lang_table:
                sections.append(lang_table)
            elif self.detected_languages and not self.language_metrics:
                # Show placeholder table when training hasn't started yet
                placeholder = Table(title="🌍 Per-Language Performance", box=box.ROUNDED,
                                   title_style="bold magenta")
                placeholder.add_column("Info", style="cyan")
                placeholder.add_row(f"Detected languages: {', '.join(self.detected_languages)}")
                placeholder.add_row("Metrics will appear after first validation")
                sections.append(placeholder)

        sections.append(Text())  # Spacer
        sections.append(self.create_summary_section())

        group = Group(*sections)

        # Different colors for normal vs reinforced training
        if self.is_reinforced:
            title = "🔥 REINFORCED LEARNING"
            border_color = "bold yellow"  # Yellow/orange for reinforced
        else:
            title = "🏋️ MODEL TRAINING"
            border_color = "bold blue"  # Blue for normal

        return Panel(group, title=title, border_style=border_color, box=box.HEAVY)

    def _create_bar(self, percentage: float, width: int = 30) -> str:
        """Create a text-based progress bar."""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:.1f}%"

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


class BertBase(BertABC):
    def __init__(
            self,
            model_name: str = 'bert-base-cased',
            tokenizer: Any = BertTokenizer,
            model_sequence_classifier: Any = BertForSequenceClassification,
            device: Device | None = None,
    ):
        """
        Parameters
        ----------
        model_name: str, default='bert-base-cased'
            A model name from huggingface models: https://huggingface.co/models

        tokenizer: huggingface tokenizer, default=BertTokenizer.from_pretrained('bert-base-cased')
            Tokenizer to use

        model_sequence_classifier: huggingface sequence classifier, default=BertForSequenceClassification
            A huggingface sequence classifier that implements a from_pretrained() function

        device: torch.Device, default=None
            Device to use. If None, automatically set if GPU is available. CPU otherwise.
        """
        self.model_name = model_name
        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.model_sequence_classifier = model_sequence_classifier
        self.dict_labels = None
        self.language_metrics_history: List[Dict[str, Any]] = []
        self.last_training_summary: Optional[Dict[str, Any]] = None
        self.last_saved_model_path: Optional[str] = None
        self.logger = get_logger(f"AugmentedSocialScientistFork.{self.__class__.__name__}")

        # Multi-class support: will be set dynamically if needed
        self.num_labels: int = 2  # Default to binary classification
        self.class_names: Optional[List[str]] = None
        self.detected_languages: Optional[List[str]] = None

        # Set or detect device
        self.device = device
        if self.device is None:
            # If CUDA is available, use it
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info('Detected %d GPU(s). Using GPU %s (%s).',
                                 torch.cuda.device_count(),
                                 torch.cuda.current_device(),
                                 torch.cuda.get_device_name(torch.cuda.current_device()))
            # If MPS is available on Apple Silicon, use it
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                # Suppress MPS detection log to keep console clean
                # self.logger.info('Detected Apple Silicon MPS backend. Using the MPS device.')
            # Otherwise, use CPU
            else:
                self.device = torch.device("cpu")
                # Suppress CPU fallback log to keep console clean
                # self.logger.info('Falling back to CPU execution.')

    def _supports_token_type_ids(self, model) -> bool:
        """
        Check if the model supports token_type_ids parameter.
        DistilBERT and some other models don't support this parameter.
        """
        import inspect
        forward_signature = inspect.signature(model.forward)
        return 'token_type_ids' in forward_signature.parameters

    # ------------------------------------------------------------------
    # Internal helpers shared with enhanced variants
    # ------------------------------------------------------------------
    def _prepare_inputs(
            self,
            sequences: List[str],
            add_special_tokens: bool = True,
            progress_bar: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenise and build attention masks for a list of sequences."""
        if not sequences:
            return np.zeros((0, 0), dtype="long"), np.zeros((0, 0), dtype="long")

        iterator = tqdm(sequences, desc="Tokenizing") if progress_bar else sequences
        input_ids: List[List[int]] = []
        for idx, sent in enumerate(iterator):
            # CRITICAL: Validate input is a string
            if not isinstance(sent, str):
                self.logger.error(f"Invalid text at index {idx}: type={type(sent)}, value={sent}")
                raise TypeError(f"Text at index {idx} must be a string, got {type(sent)}. Value: {sent}")
            if not sent.strip():
                self.logger.error(f"Empty text at index {idx}")
                raise ValueError(f"Text at index {idx} is empty. All texts must be non-empty strings.")
            encoded_sent = self.tokenizer.encode(sent, add_special_tokens=add_special_tokens)
            input_ids.append(encoded_sent)

        max_len = min(max(len(sen) for sen in input_ids), 512)
        pad = np.full((len(input_ids), max_len), 0, dtype="long")
        for idx, seq in enumerate(input_ids):
            trunc = seq[:max_len]
            pad[idx, :len(trunc)] = trunc

        attention_masks: List[List[int]] = []
        mask_iter = tqdm(pad, desc="Creating attention masks") if progress_bar else pad
        for seq in mask_iter:
            attention_masks.append([int(token_id > 0) for token_id in seq])

        return pad, np.asarray(attention_masks, dtype="long")

    def _build_dataset(
            self,
            inputs: np.ndarray,
            masks: np.ndarray,
            labels: Optional[List[str | int]] = None
    ) -> Tuple[TensorDataset, Optional[Dict[str, int]]]:
        """Create a TensorDataset from arrays and optional labels."""
        inputs_tensors = torch.tensor(inputs, dtype=torch.long)
        masks_tensors = torch.tensor(masks, dtype=torch.long)

        if labels is None:
            dataset = TensorDataset(inputs_tensors, masks_tensors)
            return dataset, None

        # CRITICAL: Convert to Python list to avoid numpy array issues
        label_names = np.unique(labels).tolist()

        # CRITICAL FIX: Ensure NOT_* labels get index 0, others get higher indices
        # This matches the display convention: Class 0 = NOT_category, Class 1 = category
        sorted_labels = sorted(label_names, key=lambda x: (not str(x).startswith('NOT_'), str(x)))
        self.dict_labels = dict(zip(sorted_labels, range(len(sorted_labels))))

        labels_tensors = torch.tensor([self.dict_labels[x] for x in labels], dtype=torch.long)
        dataset = TensorDataset(inputs_tensors, masks_tensors, labels_tensors)
        return dataset, self.dict_labels

    def encode(
            self,
            sequences: List[str],
            labels: List[str | int] | None = None,
            batch_size: int = 32,
            progress_bar: bool = True,
            add_special_tokens: bool = True,
            shuffle: bool = False
    ) -> DataLoader:
        """
        Preprocess training, test or prediction data by:
          (1) Tokenizing the sequences and mapping tokens to their IDs.
          (2) Truncating or padding to a max length of 512 tokens, and creating corresponding attention masks.
          (3) Returning a pytorch DataLoader containing token ids, labels (if any) and attention masks.

        Parameters
        ----------
        sequences: 1D array-like
            List of input texts.

        labels: 1D array-like or None, default=None
            List of labels. None for unlabeled prediction data.

        batch_size: int, default=32
            Batch size for the PyTorch DataLoader.

        progress_bar: bool, default=True
            If True, print a progress bar for tokenization and mask creation.

        add_special_tokens: bool, default=True
            If True, add '[CLS]' and '[SEP]' tokens.

        shuffle: bool, default=False
            If True, shuffle the data (should be True for training, False for validation/test).

        Returns
        -------
        dataloader: torch.utils.data.DataLoader
            A PyTorch DataLoader with input_ids, attention_masks, and labels (if provided).
        """
        # CRITICAL FIX: Convert numpy types to Python native types
        # This prevents issues with numpy.int64 objects in downstream processing
        if labels is not None:
            from .data_utils import safe_convert_labels
            labels = safe_convert_labels(labels)

        inputs, masks = self._prepare_inputs(
            sequences,
            add_special_tokens=add_special_tokens,
            progress_bar=progress_bar,
        )

        dataset, label_mapping = self._build_dataset(inputs, masks, labels)

        # Suppress label mapping log to keep console clean
        # if label_mapping is not None and progress_bar:
        #     self.logger.info("Label ids mapping: %s", label_mapping)

        if shuffle:
            from torch.utils.data import RandomSampler
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def _collect_language_codes(
            self,
            primary_language: Optional[str],
            language_info: Optional[List[str]] = None,
            extra_languages: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """Aggregate normalized language codes from all training sources."""
        languages = set()

        def _add(lang_candidate):
            if not lang_candidate:
                return
            if isinstance(lang_candidate, str):
                normalized = lang_candidate.strip().upper()
                if normalized and normalized != "MULTI":
                    languages.add(normalized)
                return
            try:
                iterator = iter(lang_candidate)
            except TypeError:
                return
            for item in iterator:
                if isinstance(item, str):
                    normalized = item.strip().upper()
                    if normalized and normalized != "MULTI":
                        languages.add(normalized)

        _add(extra_languages)
        if hasattr(self, "confirmed_languages"):
            _add(getattr(self, "confirmed_languages"))
        if hasattr(self, "detected_languages") and self.detected_languages:
            _add(self.detected_languages)
        _add(language_info)

        normalized_primary = None
        if isinstance(primary_language, str) and primary_language.strip():
            normalized_primary = primary_language.strip().upper()
            if normalized_primary != "MULTI":
                languages.add(normalized_primary)

        sorted_languages = sorted(languages)

        if normalized_primary and normalized_primary != "MULTI":
            resolved_primary = normalized_primary
        elif normalized_primary == "MULTI":
            if len(sorted_languages) > 1:
                resolved_primary = "MULTI"
            elif sorted_languages:
                resolved_primary = sorted_languages[0]
            else:
                resolved_primary = "MULTI"
        elif len(sorted_languages) == 1:
            resolved_primary = sorted_languages[0]
        elif len(sorted_languages) > 1:
            resolved_primary = "MULTI"
        else:
            resolved_primary = None

        return resolved_primary, sorted_languages

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert value to float, falling back to default on failure."""
        if value is None:
            return default
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Convert value to int, falling back to default on failure."""
        if value is None:
            return default
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalize_language_metrics(
        self,
        language_metrics: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Return a JSON-serialisable copy of per-language metrics."""
        if not language_metrics:
            return {}

        normalised: Dict[str, Dict[str, Any]] = {}
        for lang, metrics in language_metrics.items():
            if not isinstance(metrics, dict):
                continue
            cleaned: Dict[str, Any] = {}
            for key, raw_value in metrics.items():
                value = raw_value
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple)):
                    cleaned[key] = [self._safe_float(v) for v in value]
                    continue
                if isinstance(value, dict):
                    cleaned[key] = {
                        sub_key: self._safe_float(sub_val)
                        for sub_key, sub_val in value.items()
                    }
                    continue
                if key.startswith("support") or key in {"support", "samples"}:
                    cleaned[key] = self._safe_int(value)
                else:
                    cleaned[key] = self._safe_float(value)
            normalised[lang] = cleaned
        return normalised

    def _compute_language_averages(
        self,
        language_metrics: Optional[Dict[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, float]]:
        """Compute average language scores when explicit averages are missing."""
        if not language_metrics:
            return None

        metrics_values = [
            metrics for metrics in language_metrics.values()
            if isinstance(metrics, dict) and metrics
        ]
        if not metrics_values:
            return None

        acc_total = 0.0
        f1_total = 0.0
        count = 0

        for metrics in metrics_values:
            acc_total += self._safe_float(metrics.get("accuracy"))
            f1_total += self._safe_float(
                metrics.get("f1_macro", metrics.get("macro_f1"))
            )
            count += 1

        if count == 0:
            return None

        avg_acc = acc_total / count
        avg_f1 = f1_total / count

        return {
            "accuracy": avg_acc,
            "f1_macro": avg_f1,
            "macro_f1": avg_f1,
        }

    def _build_final_metrics_block(
        self,
        *,
        combined_metric: Optional[Any],
        macro_f1: Optional[Any],
        accuracy: Optional[Any],
        epoch: Optional[int],
        train_loss: Optional[Any],
        val_loss: Optional[Any],
        precisions: Optional[Any],
        recalls: Optional[Any],
        f1_scores: Optional[Any],
        supports: Optional[Any],
        label_names: Optional[List[str]],
        language_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        language_averages: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assemble the detailed metrics payload stored in training_metadata.json."""
        overall: Dict[str, float] = {}

        if combined_metric is not None:
            overall["combined_metric"] = self._safe_float(combined_metric)
        if macro_f1 is not None:
            overall["macro_f1"] = self._safe_float(macro_f1)
        if accuracy is not None:
            overall["accuracy"] = self._safe_float(accuracy)
        if train_loss is not None:
            overall["train_loss"] = self._safe_float(train_loss)
        if val_loss is not None:
            overall["val_loss"] = self._safe_float(val_loss)

        per_class: List[Dict[str, Any]] = []

        def _as_list(values: Optional[Any]) -> List[Any]:
            if values is None:
                return []
            if isinstance(values, np.ndarray):
                return values.tolist()
            if isinstance(values, (list, tuple)):
                return list(values)
            return [values]

        precisions_list = _as_list(precisions)
        recalls_list = _as_list(recalls)
        f1_list = _as_list(f1_scores)
        supports_list = _as_list(supports)

        num_labels = max(
            len(precisions_list),
            len(recalls_list),
            len(f1_list),
            len(supports_list),
            len(label_names) if label_names else 0,
        )

        resolved_labels = (
            list(label_names) if label_names and len(label_names) >= num_labels
            else [f"class_{idx}" for idx in range(num_labels)]
        )

        for idx in range(num_labels):
            per_class.append({
                "label": resolved_labels[idx] if idx < len(resolved_labels) else f"class_{idx}",
                "precision": self._safe_float(precisions_list[idx]) if idx < len(precisions_list) else 0.0,
                "recall": self._safe_float(recalls_list[idx]) if idx < len(recalls_list) else 0.0,
                "f1": self._safe_float(f1_list[idx]) if idx < len(f1_list) else 0.0,
                "support": self._safe_int(supports_list[idx]) if idx < len(supports_list) else 0,
            })

        languages_cleaned = self._normalize_language_metrics(language_metrics)

        averages_cleaned = None
        if language_averages:
            averages_cleaned = {
                key: self._safe_float(value)
                for key, value in language_averages.items()
            }
        elif languages_cleaned:
            averages_cleaned = self._compute_language_averages(language_metrics)

        if not overall and not per_class and not languages_cleaned:
            return None

        metrics_block: Dict[str, Any] = {
            "epoch": epoch,
            "overall": overall,
        }

        if per_class:
            metrics_block["per_class"] = per_class
        if languages_cleaned:
            metrics_block["per_language"] = languages_cleaned
        if averages_cleaned:
            metrics_block["language_averages"] = averages_cleaned

        return metrics_block

    def _apply_languages_to_config(
            self,
            model_config,
            primary_language: Optional[str],
            confirmed_languages: List[str],
    ) -> None:
        """Persist language metadata into the Hugging Face config object."""
        languages_list = confirmed_languages or []
        model_config.confirmed_languages = languages_list
        model_config.languages = languages_list

        if primary_language:
            model_config.language = primary_language
            model_config.primary_language = primary_language
        else:
            model_config.language = None
            model_config.primary_language = None

        model_config.language_strategy = "multilingual" if len(languages_list) > 1 else "single"

    def calculate_reinforced_trigger_score(
            self,
            f1_class_0: float,
            f1_class_1: float,
            support_class_0: int,
            support_class_1: int,
            language_metrics: Optional[Dict[str, Dict[str, float]]] = None,
            reinforced_f1_threshold: float = 0.7
    ) -> Tuple[float, bool, str]:
        """
        Calculate an intelligent score to determine if reinforced learning should be triggered.

        This method considers:
        1. F1 score of class 1 (minority class)
        2. Class imbalance ratio (to overweight minority class performance)
        3. Per-language F1 scores (if available)

        Parameters
        ----------
        f1_class_0 : float
            F1 score for class 0
        f1_class_1 : float
            F1 score for class 1 (typically minority class)
        support_class_0 : int
            Number of samples in class 0
        support_class_1 : int
            Number of samples in class 1
        language_metrics : dict, optional
            Dictionary with language-specific metrics

        Returns
        -------
        trigger_score : float
            Computed score (0.0 to 1.0) - lower means worse performance
        should_trigger : bool
            True if reinforced learning should be triggered
        reason : str
            Human-readable explanation of the decision
        """
        # Calculate class imbalance ratio
        total_samples = support_class_0 + support_class_1
        if total_samples == 0:
            return 0.0, True, "No samples available"

        class_1_ratio = support_class_1 / total_samples
        class_0_ratio = support_class_0 / total_samples

        # Calculate imbalance weight (higher when class is more imbalanced)
        # When class 1 is 50%, imbalance_weight = 1.0
        # When class 1 is 10%, imbalance_weight = 2.0
        # When class 1 is 5%, imbalance_weight = 2.5
        if class_1_ratio > 0:
            imbalance_weight = min(0.5 / class_1_ratio, 3.0)  # Cap at 3.0x weight
        else:
            imbalance_weight = 3.0

        # Base score: weighted F1 of class 1
        # When class is very imbalanced, F1 of class 1 becomes more important
        base_weight = 0.4 + (0.3 * min(imbalance_weight / 3.0, 1.0))  # 0.4 to 0.7
        class_1_weighted_f1 = f1_class_1 * base_weight
        class_0_weighted_f1 = f1_class_0 * (1.0 - base_weight)

        overall_f1_score = class_1_weighted_f1 + class_0_weighted_f1

        # Factor in language-specific performance if available
        language_penalty = 0.0
        language_info = ""
        language_variance_penalty = 0.0

        if language_metrics and len(language_metrics) > 0:
            # Check if any language has poor F1 for class 1
            poor_languages = []
            f1_class1_by_lang = []

            for lang, metrics in language_metrics.items():
                lang_f1_1 = metrics.get('f1_1', 0.0)
                lang_support_1 = metrics.get('support_1', 0)

                # Track F1 class 1 for variance calculation
                if lang_support_1 >= 3:  # Only include languages with meaningful support
                    f1_class1_by_lang.append(lang_f1_1)

                # Consider a language "poor" if F1_1 < 0.5 and has reasonable support
                if lang_f1_1 < 0.5 and lang_support_1 >= 3:
                    poor_languages.append(f"{lang}(F1={lang_f1_1:.2f})")
                    # Apply penalty proportional to how bad the language is
                    language_penalty += (0.5 - lang_f1_1) * 0.15  # Max 0.15 penalty per language

            # Calculate variance penalty if we have multiple languages
            if len(f1_class1_by_lang) > 1:
                # CRITICAL: Convert to Python float to avoid numpy scalar issues
                mean_f1 = float(np.mean(f1_class1_by_lang))
                std_f1 = np.std(f1_class1_by_lang)

                # High variance = unbalanced performance across languages
                # Apply penalty if coefficient of variation > 0.5
                if mean_f1 > 0:
                    cv = std_f1 / mean_f1
                    if cv > 0.5:  # Significant variance
                        language_variance_penalty = (cv - 0.5) * 0.1  # Up to 0.1 penalty
                        language_info += f" High variance across languages (CV={cv:.2f})."

            if poor_languages:
                language_info = f" Poor language performance: {', '.join(poor_languages)}." + language_info

        # Final trigger score (penalized by poor language performance and variance)
        trigger_score = max(0.0, overall_f1_score - language_penalty - language_variance_penalty)

        # Decision logic
        # Trigger if:
        # 1. Score is below reinforced_f1_threshold (configurable threshold)
        # 2. OR class 1 F1 is below 0.40 (very poor minority class performance)
        # 3. OR any language has F1_1 < 0.30
        should_trigger = False
        reason = ""

        if trigger_score < reinforced_f1_threshold:
            should_trigger = True
            reason = (f"Trigger score {trigger_score:.3f} < {reinforced_f1_threshold:.2f} "
                     f"(Class 1 F1={f1_class_1:.3f}, imbalance={class_1_ratio:.1%}, "
                     f"weight={imbalance_weight:.2f}x).{language_info}")
        elif f1_class_1 < 0.40:
            should_trigger = True
            reason = f"Class 1 F1 ({f1_class_1:.3f}) critically low < 0.40.{language_info}"
        elif language_metrics:
            for lang, metrics in language_metrics.items():
                if metrics.get('f1_1', 0.0) < 0.30 and metrics.get('support_1', 0) >= 3:
                    should_trigger = True
                    reason = f"Language {lang} has critical F1_1={metrics['f1_1']:.3f} < 0.30.{language_info}"
                    break

        if not should_trigger:
            reason = (f"No trigger: score={trigger_score:.3f}, F1_1={f1_class_1:.3f}, "
                     f"class 1 ratio={class_1_ratio:.1%}.{language_info}")

        return trigger_score, should_trigger, reason

    def run_training(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            n_epochs: int = 3,
            lr: float = 5e-5,
            random_state: int = 42,
            save_model_as: str | None = None,
            pos_weight: torch.Tensor | None = None,
            metrics_output_dir: Optional[str] = None,
            best_model_criteria: str = "combined",
            f1_class_1_weight: float = 0.7,
            reinforced_learning: bool = False,
            n_epochs_reinforced: int = 2,
            reinforced_epochs: int | None = None,   # ← nouveau
            rescue_low_class1_f1: bool = False,
            track_languages: bool = False,
            language_info: Optional[List[str]] = None,
            f1_1_rescue_threshold: float = 0.0,
            model_identifier: Optional[str] = None,
            reinforced_f1_threshold: float = 0.7,  # Nouveau paramètre pour le seuil de déclenchement
            rl_f1_threshold: float = 0.7,  # Alias for reinforced_f1_threshold for consistency with multi_label_trainer
            rl_oversample_factor: float = 2.0,  # Oversampling factor for reinforced learning
            rl_class_weight_factor: float = 2.0,  # Class weight factor for reinforced learning
            label_key: Optional[str] = None,  # Multi-label: key being trained (e.g., 'themes', 'sentiment')
            label_value: Optional[str] = None,  # Multi-label: specific value (e.g., 'transportation', 'positive')
            language: Optional[str] = None,  # Language of the data being trained (e.g., 'EN', 'FR')
            class_names: Optional[List[str]] = None,  # Multi-class: list of class names (e.g., ['natural_sciences', 'no', 'social_sciences'])
            session_id: Optional[str] = None,  # Session timestamp for organizing logs by session
            is_benchmark: bool = False,  # Whether this is benchmark mode (adds benchmark folder to path)
            model_name_for_logging: Optional[str] = None,  # Model name for benchmark logging (e.g., 'bert-base-uncased')
            progress_callback: Optional[callable] = None,  # Callback function called after each epoch with (epoch, metrics)
            global_total_models: Optional[int] = None,  # Total number of models in this training session
            global_current_model: Optional[int] = None,  # Current model number (1-indexed)
            global_total_epochs: Optional[int] = None,  # Total epochs across all models (base scenario)
            global_max_epochs: Optional[int] = None,  # Maximum possible epochs if all models trigger reinforced learning
            global_completed_epochs: Optional[int] = None,  # Completed epochs across all models
            global_start_time: Optional[float] = None  # Start time of the entire training session
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Train, evaluate, and (optionally) save a BERT model. This method also logs training and validation
        metrics per epoch, handles best-model selection, and can optionally trigger reinforced learning if
        the best F1 on class 1 is below the reinforced_f1_threshold at the end of normal training.

        This method can also (optionally) apply a "rescue" logic for class 1 F1 scores that remain at 0
        after normal training: if ``rescue_low_class1_f1=True`` and the best model's F1 for class 1 is 0,
        the reinforced learning step will consider any small improvement of class 1's F1 (greater than
        ``f1_1_rescue_threshold``) as sufficient to select a reinforced epoch's model.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader, typically from self.encode().

        test_dataloader: torch.utils.data.DataLoader
            Test/validation dataloader, typically from self.encode().

        n_epochs: int, default=3
            Number of epochs for the normal training phase.

        lr: float, default=5e-5
            Learning rate for normal training.

        random_state: int, default=42
            Random seed for reproducibility.

        save_model_as: str, default=None
            If not None, will save the final best model to ./models/<save_model_as>.

        pos_weight: torch.Tensor, default=None
            If not None, weights the loss to favor certain classes more heavily
            (useful in binary classification).

        metrics_output_dir: Optional[str], default=None
            Base directory for saving CSV logs. When None, the active training session directory is used.

        best_model_criteria: str, default="combined"
            Criterion for best model. Currently supports:
              - "combined": Weighted combination of F1(class 1) and macro F1.

        f1_class_1_weight: float, default=0.7
            Weight for F1(class 1) in the combined metric. The remaining (1 - weight) goes to macro F1.

        reinforced_learning: bool, default=False
            If True, and if the best model after normal training has F1(class 1) < 0.7,
            a reinforced training phase will be triggered.

        n_epochs_reinforced: int, default=2
            Number of epochs for the reinforced learning phase (if triggered).

        rescue_low_class1_f1: bool, default=False
            If True, then during reinforced learning we check if the best normal-training
            F1 for class 1 is 0. In that case, any RL epoch where class 1's F1 becomes greater
            than ``f1_1_rescue_threshold`` is automatically considered a better model.

        f1_1_rescue_threshold: float, default=0.0
            The threshold above which a class 1 F1 (starting from 0 after normal training)
            is considered a sufficient improvement to pick the reinforced epoch's model.

        Returns
        -------
        scores: tuple (precision, recall, f1-score, support)
            Final best evaluation scores from sklearn.metrics.precision_recall_fscore_support,
            for each label. Shape: (4, n_labels).

        Notes
        -----
        This method generates CSV logs within the resolved training session directory, including:
            - "training_metrics.csv": metrics for each normal-training epoch.
            - "best.csv": entries for any new best model (normal or reinforced).
            - If reinforced training is triggered, it also logs a reinforced_training_metrics.csv.
            - The final best model is ultimately saved to "./models/<save_model_as>" if save_model_as is provided.
              (If reinforced training finds a better model, that replaces the previous best.)
        """
        # Reset reinforced learning flag at the start of each training session
        self._reinforced_already_triggered = False

        # Use rl_f1_threshold if provided (consistency with multi_label_trainer parameter naming)
        # Priority: rl_f1_threshold > reinforced_f1_threshold
        if rl_f1_threshold != 0.7:  # Non-default value provided
            reinforced_f1_threshold = rl_f1_threshold

        # Store global progress tracking parameters for display updates
        self.global_total_models = global_total_models
        self.global_current_model = global_current_model
        self.global_total_epochs = global_total_epochs
        self.global_max_epochs = global_max_epochs or global_total_epochs  # Default to global_total_epochs if not provided
        self.global_completed_epochs = global_completed_epochs
        self.global_start_time = global_start_time

        # CRITICAL DEBUG: Log all parameter types at entry to run_training
        self.logger.debug("=" * 80)
        self.logger.debug("run_training ENTRY - Parameter type check:")
        self.logger.debug(f"  n_epochs: type={type(n_epochs)}, value={n_epochs}")
        self.logger.debug(f"  lr: type={type(lr)}, value={lr}")
        self.logger.debug(f"  reinforced_epochs: type={type(reinforced_epochs)}, value={reinforced_epochs}")
        self.logger.debug(f"  n_epochs_reinforced: type={type(n_epochs_reinforced)}, value={n_epochs_reinforced}")
        self.logger.debug(f"  reinforced_learning: type={type(reinforced_learning)}, value={reinforced_learning}")
        self.logger.debug(f"  session_id: type={type(session_id)}, value={session_id}")
        self.logger.debug(f"  global_total_models: type={type(global_total_models)}, value={global_total_models}")
        self.logger.debug(f"  global_current_model: type={type(global_current_model)}, value={global_current_model}")
        self.logger.debug(f"  global_total_epochs: type={type(global_total_epochs)}, value={global_total_epochs}")
        self.logger.debug(f"  global_max_epochs: type={type(global_max_epochs)}, value={global_max_epochs}")
        self.logger.debug(f"  global_completed_epochs: type={type(global_completed_epochs)}, value={global_completed_epochs}")
        self.logger.debug(f"  global_start_time: type={type(global_start_time)}, value={global_start_time}")

        # Check ALL parameters for numpy types
        import numpy as np
        params_to_check = {
            'n_epochs': n_epochs,
            'n_epochs_reinforced': n_epochs_reinforced,
            'reinforced_epochs': reinforced_epochs,
            'global_total_models': global_total_models,
            'global_current_model': global_current_model,
            'global_total_epochs': global_total_epochs,
            'global_max_epochs': global_max_epochs,
            'global_completed_epochs': global_completed_epochs
        }
        for key, value in params_to_check.items():
            if value is not None and isinstance(value, (np.integer, np.floating, np.ndarray)):
                self.logger.warning(f"  ⚠️ NUMPY TYPE DETECTED: {key} = type={type(value)}, value={value}")
        self.logger.debug("=" * 80)

        # Define model_type and training_approach early to avoid UnboundLocalError
        model_type = self.model_name if hasattr(self, 'model_name') else self.__class__.__name__

        # Determine training approach based on context
        # Multi-class: multiple classes for one key
        # One-vs-all: binary classifiers for each value in a multi-label setting
        if label_value:
            # We're training for a specific value in one-vs-all approach
            training_approach = "one-vs-all"
        elif class_names and len(class_names) > 2:
            # We have multiple classes - multi-class approach
            training_approach = "multi-class"
        else:
            # Binary classification or default
            training_approach = "binary"

        # NEW STRUCTURE:
        # Normal mode:    logs/training_arena/{session_id}/training_metrics/{category}/
        # Benchmark mode: logs/training_arena/{session_id}/training_metrics/benchmark/{category}/{language}/{model}/
        # session_timestamp: Date and time of the training session (e.g., 20251007_103025)
        # category: The label/category being trained (e.g., specific_themes, sentiment)
        # language: Language code (e.g., EN, FR, MULTI) - only in benchmark mode
        # model: Model identifier (e.g., bert-base-uncased) - only in benchmark mode

        # Create or use session ID (timestamp)
        # CRITICAL: Always use "training_session_" prefix for consistency with benchmarking
        if session_id is None:
            session_id = datetime.datetime.now().strftime("training_session_%Y%m%d_%H%M%S")

        # Determine category name
        # Priority: label_value > label_key > "default"
        if label_value:
            category_name = str(label_value)
        elif label_key:
            category_name = str(label_key)
        else:
            category_name = "default"

        # Clean category name
        category_name = category_name.strip()
        category_name = re.sub(r"\s+", "_", category_name)
        category_name = re.sub(r"[^0-9A-Za-z_\-]", "", category_name)
        category_name = category_name or "category"

        # Build directory structure based on mode using the resolved session directory
        resolved_base_dir = resolve_metrics_base_dir(metrics_output_dir)
        use_configured_override = (
            metrics_output_dir is not None
            and Path(resolved_base_dir) == Path(metrics_output_dir)
        )
        if use_configured_override:
            session_root = Path(metrics_output_dir) / session_id
        else:
            session_root = get_session_dir(session_id)
        session_dir_path = session_root / "training_metrics"
        session_dir = str(session_dir_path)

        # Also create the same structure for model outputs in models/ directory
        models_base = "models"
        model_session_dir = os.path.join(models_base, session_id)

        if is_benchmark:
            # Benchmark mode: benchmark/category/language/model folders
            benchmark_dir = os.path.join(session_dir, "benchmark")
            category_dir = os.path.join(benchmark_dir, category_name)

            # Model directory with same structure
            model_benchmark_dir = os.path.join(model_session_dir, "benchmark")
            model_category_dir = os.path.join(model_benchmark_dir, category_name)

            # Add language folder if language is specified
            if language:
                lang_clean = language.upper().replace("/", "_").replace(" ", "_")
                category_dir = os.path.join(category_dir, lang_clean)
                model_category_dir = os.path.join(model_category_dir, lang_clean)

            # Add model folder if model name is specified
            if model_name_for_logging:
                model_clean = model_name_for_logging.replace("/", "_").replace(" ", "_")
                category_dir = os.path.join(category_dir, model_clean)
                model_category_dir = os.path.join(model_category_dir, model_clean)
        else:
            # Normal mode: normal_training/category/language/model folders (same structure as benchmark)
            normal_training_dir = os.path.join(session_dir, "normal_training")
            category_dir = os.path.join(normal_training_dir, category_name)

            # Model directory with same structure
            model_normal_dir = os.path.join(model_session_dir, "normal_training")
            model_category_dir = os.path.join(model_normal_dir, category_name)

            # Add language folder if language is specified
            if language:
                lang_clean = language.upper().replace("/", "_").replace(" ", "_")
                category_dir = os.path.join(category_dir, lang_clean)
                model_category_dir = os.path.join(model_category_dir, lang_clean)

            # Add model folder if model name is specified
            if model_name_for_logging:
                model_clean = model_name_for_logging.replace("/", "_").replace(" ", "_")
                category_dir = os.path.join(category_dir, model_clean)
                model_category_dir = os.path.join(model_category_dir, model_clean)

        os.makedirs(category_dir, exist_ok=True)
        # IMPORTANT: Do NOT create model_category_dir here to avoid creating
        # placeholder directories. It will be created only when actually saving a model.
        # os.makedirs(model_category_dir, exist_ok=True)  # Removed - created on-demand

        # CSV files are now general (contain all models for this category)
        training_metrics_csv = os.path.join(category_dir, "training.csv")
        best_models_csv = os.path.join(category_dir, "best.csv")

        # Note: best_models.csv headers will be written after num_labels is determined
        # This is deferred until after we collect test labels (line ~860)

        # Collect test labels for classification report
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()

        # CRITICAL: Determine num_labels for metrics and CSV headers
        # Priority: use self.num_labels if set (multi-class), otherwise infer from data
        if hasattr(self, 'num_labels') and self.num_labels > 2:
            # Multi-class mode: use the num_labels set on the model
            num_labels = self.num_labels
        else:
            # Binary mode or not set: infer from data (backward compatibility)
            # CRITICAL: Convert to Python int to avoid numpy scalar issues
            num_labels_from_data = int(np.unique(test_labels).size)
            # Ensure we have at least 2 labels for binary classification
            # This fixes the issue when all samples in test have the same label
            if num_labels_from_data < 2:
                num_labels = 2
            else:
                num_labels = num_labels_from_data

        # Use class_names from parameter if provided, otherwise use self.class_names
        if class_names is None and hasattr(self, 'class_names') and self.class_names:
            class_names = self.class_names

        # Note: Reinforcement learning now supports multi-class classification

        # Initialize CSV headers now that we know num_labels
        # CRITICAL: Use standardized class indices (_0, _1, _2) instead of label names
        csv_headers = [
            "model_name",
            "timestamp",
            "epoch",
            "train_loss",
            "val_loss",
            "combined_score",   # Combined score for ranking
            "accuracy",         # Overall accuracy
        ]

        # Add per-class metric headers dynamically based on num_labels
        # CRITICAL: Use standardized indices (_0, _1, _2) for ML consistency
        for i in range(num_labels):
            csv_headers.extend([
                f"precision_{i}",
                f"recall_{i}",
                f"f1_{i}",
                f"support_{i}"
            ])

        csv_headers.append("macro_f1")

        # Add language-specific headers ONLY for the current training language
        # Individual CSVs should only contain metrics for their specific language
        # Full cross-language metrics are in consolidated session CSVs only
        if track_languages and language_info is not None:
            # Determine which languages to include in headers
            if language and language != 'MULTI':
                # Single language training: only add columns for THIS language
                langs_for_headers = [language.upper()]
            else:
                # Multilingual or unspecified: add columns for all detected languages
                langs_for_headers = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))

            for lang in langs_for_headers:
                csv_headers.append(f"{lang}_accuracy")
                for i in range(num_labels):
                    csv_headers.extend([
                        f"{lang}_precision_{i}",
                        f"{lang}_recall_{i}",
                        f"{lang}_f1_{i}",
                        f"{lang}_support_{i}"
                    ])
                csv_headers.append(f"{lang}_macro_f1")

        # Build class legend for CSV metadata
        class_legend = "CLASS_LEGEND: " + ", ".join([f"{i}={class_names[i]}" if class_names and i < len(class_names) else f"{i}=class_{i}" for i in range(num_labels)])

        # Check if file exists to decide whether to write headers
        if os.path.exists(training_metrics_csv):
            # Append to existing file
            mode = 'a'
            write_headers = False
        else:
            # Create new file with headers
            mode = 'w'
            write_headers = True

        with open(training_metrics_csv, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_headers:
                # Write legend as comment row (starts with #)
                f.write(f"# {class_legend}\n")
                # Write column headers
                writer.writerow(csv_headers)

        # Initialize best_models.csv headers now that we know num_labels
        # CRITICAL: Use standardized class indices (_0, _1, _2) and add combined_score
        best_models_headers = [
            "model_type",
            "timestamp",
            "epoch",
            "train_loss",
            "val_loss",
            "combined_score",   # CRITICAL: Combined score for ranking (weighted F1)
            "accuracy",         # Overall accuracy
        ]

        # Add per-class metric headers dynamically based on num_labels
        # CRITICAL: Use standardized indices (_0, _1, _2) for ML consistency
        for i in range(num_labels):
            best_models_headers.extend([
                f"precision_{i}",
                f"recall_{i}",
                f"f1_{i}",
                f"support_{i}"
            ])

        best_models_headers.append("macro_f1")

        # Add language-specific headers ONLY for the current training language
        # Individual CSVs should only contain metrics for their specific language
        # Full cross-language metrics are in consolidated session CSVs only
        if track_languages and language_info is not None:
            # Determine which languages to include in headers
            if language and language != 'MULTI':
                # Single language training: only add columns for THIS language
                langs_for_headers = [language.upper()]
            else:
                # Multilingual or unspecified: add columns for all detected languages
                langs_for_headers = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))

            for lang in langs_for_headers:
                best_models_headers.append(f"{lang}_accuracy")
                for i in range(num_labels):
                    best_models_headers.extend([
                        f"{lang}_precision_{i}",
                        f"{lang}_recall_{i}",
                        f"{lang}_f1_{i}",
                        f"{lang}_support_{i}"
                    ])
                best_models_headers.append(f"{lang}_macro_f1")

        best_models_headers.extend([
            "saved_model_path",
            "training_phase"
        ])

        # Check if file exists to decide whether to write headers
        if os.path.exists(best_models_csv):
            # Append to existing file
            mode = 'a'
            write_headers = False
        else:
            # Create new file with headers
            mode = 'w'
            write_headers = True

        with open(best_models_csv, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_headers:
                # Write legend as comment row (starts with #)
                f.write(f"# {class_legend}\n")
                # Write column headers
                writer.writerow(best_models_headers)

        # CRITICAL: Extract label names from multiple sources to ensure they're always available
        # Priority: 1) class_names parameter (multi-class), 2) self.class_names, 3) self.dict_labels
        label_names = None

        # First, try class_names parameter (passed for multi-class training)
        if class_names is not None and len(class_names) > 0:
            label_names = [str(name) for name in class_names]
        # Second, try self.class_names attribute
        elif hasattr(self, 'class_names') and self.class_names is not None and len(self.class_names) > 0:
            label_names = [str(name) for name in self.class_names]
        # Third, extract from self.dict_labels (binary/multi-label training)
        elif self.dict_labels is not None:
            # Sort by index - handle both simple values and dict values
            try:
                label_names = [str(x[0]) for x in sorted(self.dict_labels.items(), key=lambda x: x[1])]
            except (TypeError, AttributeError):
                # If values are dicts or other non-comparable types, just use the keys
                label_names = [str(k) for k in self.dict_labels.keys()]

        # Set seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Initialize the model (suppress warnings about missing weights)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
            warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
            model = self.model_sequence_classifier.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                output_attentions=False,
                output_hidden_states=False
            )

        # Add label mappings to model config so they're saved with the model
        if label_names:
            model.config.id2label = {i: name for i, name in enumerate(label_names)}
            model.config.label2id = {name: i for i, name in enumerate(label_names)}

        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * n_epochs
        )

        train_loss_values = []
        val_loss_values = []

        best_metric_val = -1.0
        best_model_path = None
        best_scores = None  # Will store final best (precision, recall, f1, support)
        best_language_metrics = None  # Will store language metrics from best epoch
        best_language_averages = None  # Average metrics for languages at best epoch
        best_final_metrics_block = None  # Cached detailed metrics payload for metadata
        best_combined_metric_value = None  # Combined score tied to the saved model
        best_accuracy_value = None  # Accuracy recorded for the best model
        best_macro_f1_value = None  # Macro F1 recorded for the best model
        best_epoch_index = None  # Epoch number of the best model
        language_performance_history = []  # Store language metrics for each epoch
        self.language_metrics_history = []
        if reinforced_epochs is not None:
            n_epochs_reinforced = reinforced_epochs

        # =============== Normal Training Loop ===============
        training_start_time = time.time()  # Initialize the timer

        # Initialize metrics tracking
        training_metrics = []

        # Extract detected languages from language_info if available
        # Priority 1: Use model's detected_languages if set (most complete - from all train+val data)
        # Priority 2: Extract from language_info (validation set only)
        detected_languages = []
        if hasattr(self, 'detected_languages') and self.detected_languages and len(self.detected_languages) > 0:
            # Use model's detected_languages (set from all samples in multi_label_trainer)
            detected_languages = self.detected_languages
            self.logger.info(f"✓ Using model.detected_languages: {detected_languages}")
        elif track_languages and language_info is not None and len(language_info) > 0:
            # CRITICAL FIX: Filter out NaN/None/float values before sorting
            # Also normalize to uppercase for consistency
            filtered_langs = [lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang]
            if filtered_langs:
                detected_languages = sorted(list(set(filtered_langs)))
                self.logger.info(f"✓ Extracted languages from language_info: {detected_languages}")
            else:
                self.logger.warning(f"⚠️ language_info has {len(language_info)} items but no valid languages after filtering")
        else:
            self.logger.warning(f"⚠️ No languages detected: track_languages={track_languages}, language_info={f'{len(language_info)} items' if language_info else 'None'}, self.detected_languages={self.detected_languages if hasattr(self, 'detected_languages') else 'N/A'}")

        # Initialize Rich Live Display
        display = TrainingDisplay(
            model_name=self.model_name if hasattr(self, 'model_name') else save_model_as or "BERT",
            label_key=label_key,
            label_value=label_value,
            language=language,
            n_epochs=n_epochs,
            is_reinforced=False,
            num_labels=num_labels,
            class_names=class_names,
            detected_languages=detected_languages,
            global_total_models=global_total_models,
            global_current_model=global_current_model,
            global_total_epochs=global_total_epochs,
            global_max_epochs=global_max_epochs,
            global_completed_epochs=global_completed_epochs,
            global_start_time=global_start_time,
            reinforced_learning_enabled=reinforced_learning
        )

        live_stats_summary = None
        live_stats_min_interval = UPDATE_THROTTLE
        live_stats_clear_interval = TERMINAL_CLEAR_INTERVAL

        # Start Live display - this will remain fixed and update in place
        # Use transient=True to clear the display when context exits (prevents stacking)
        with Live(display.create_panel(), **LIVE_BASE_KWARGS) as live:
            # Wrap the live object with throttling to prevent terminal saturation
            throttled_live = ThrottledLiveUpdater(live)

            for i_epoch in range(n_epochs):
                epoch_start_time = time.time()

                # Update display for new epoch
                display.current_epoch = i_epoch + 1
                display.current_phase = "Training"
                display.train_total = len(train_dataloader)
                display.train_progress = 0
                # Force update at epoch start (important milestone)
                throttled_live.update(display.create_panel(), force=True)

                t0 = time.time()
                total_train_loss = 0.0
                model.train()

                # Training loop - no tqdm, we update the display directly
                for step, train_batch in enumerate(train_dataloader):
                    b_inputs = train_batch[0].to(self.device)
                    b_masks = train_batch[1].to(self.device)
                    b_labels = train_batch[2].to(self.device)

                    model.zero_grad()

                    # Call model with or without token_type_ids depending on model support
                    if self._supports_token_type_ids(model):
                        outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                    else:
                        outputs = model(b_inputs, attention_mask=b_masks)
                    logits = outputs[0]

                    # Weighted loss if pos_weight is specified
                    if pos_weight is not None:
                        # Check if pos_weight is a tensor with multiple elements (multi-class)
                        # or a single element (binary)
                        if pos_weight.numel() > 1:
                            # Multi-class: pos_weight is already the full weight tensor
                            weight_tensor = pos_weight.to(self.device)
                        else:
                            # Binary: construct weight tensor [1.0, pos_weight]
                            weight_tensor = torch.tensor([1.0, pos_weight.item()], device=self.device)
                        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                    else:
                        criterion = torch.nn.CrossEntropyLoss()

                    loss = criterion(logits, b_labels)
                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    # Update display with current progress
                    display.train_progress = step + 1
                    display.train_loss = loss.item()
                    display.epoch_time = time.time() - epoch_start_time  # Update elapsed time
                    display.total_time = time.time() - training_start_time
                    # Throttled update for each batch (this is the critical path for GPU training)
                    # Force update on last batch to ensure final state is shown
                    is_last_batch = (step == len(train_dataloader) - 1)
                    throttled_live.update(display.create_panel(), force=is_last_batch, is_batch=True)

                # After training loop - calculate average loss
                avg_train_loss = total_train_loss / len(train_dataloader)
                train_loss_values.append(avg_train_loss)

                # Update display with final training metrics
                display.train_loss = avg_train_loss
                display.train_time = time.time() - t0
                display.current_phase = "Validation"
                display.val_total = len(test_dataloader)
                display.val_progress = 0
                # Force update for phase transition (Training -> Validation)
                throttled_live.update(display.create_panel(), force=True)

                # =============== Validation after this epoch ===============
                t0 = time.time()
                model.eval()

                total_val_loss = 0.0
                logits_complete = []

                # Validation loop - update display directly
                for step, test_batch in enumerate(test_dataloader):
                    b_inputs = test_batch[0].to(self.device)
                    b_masks = test_batch[1].to(self.device)
                    b_labels = test_batch[2].to(self.device)

                    with torch.no_grad():
                        # Call model with or without token_type_ids depending on model support
                        if self._supports_token_type_ids(model):
                            outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
                        else:
                            outputs = model(b_inputs, attention_mask=b_masks, labels=b_labels)

                        loss = outputs.loss
                        logits = outputs.logits

                    total_val_loss += loss.item()
                    logits_complete.append(logits.detach().cpu().numpy())

                    # Update display with validation progress
                    display.val_progress = step + 1
                    display.val_loss = loss.item()
                    display.epoch_time = time.time() - epoch_start_time  # Update elapsed time
                    display.total_time = time.time() - training_start_time
                    # Throttled update for validation batches
                    # Force update on last batch to ensure final state is shown
                    is_last_batch = (step == len(test_dataloader) - 1)
                    throttled_live.update(display.create_panel(), force=is_last_batch, is_batch=True)

                # After validation loop
                logits_complete = np.concatenate(logits_complete, axis=0)
                avg_val_loss = total_val_loss / len(test_dataloader)
                val_loss_values.append(avg_val_loss)

                # Update display with final validation metrics
                display.val_loss = avg_val_loss
                display.val_time = time.time() - t0
                # Force update to show final validation metrics
                throttled_live.update(display.create_panel(), force=True)

                preds = np.argmax(logits_complete, axis=1).flatten()

                # Get actual unique classes present in predictions and labels
                # CRITICAL: Convert to Python list to avoid numpy array issues
                unique_classes = np.unique(np.concatenate([test_labels, preds])).tolist()

                # Force classification_report to include all labels (0 to num_labels-1)
                # This prevents missing metrics when a class has no predictions
                all_labels = list(range(num_labels))

                # Only use target_names if it matches the number of unique classes
                if label_names is not None and len(label_names) == num_labels:
                    report = classification_report(test_labels, preds, labels=all_labels,
                                                   target_names=label_names, output_dict=True,
                                                   zero_division=0)
                else:
                    # Don't use target_names if there's a mismatch
                    report = classification_report(test_labels, preds, labels=all_labels,
                                                   output_dict=True, zero_division=0)

                # Extract metrics for all classes
                macro_avg = report.get("macro avg", {"f1-score": 0})
                macro_f1 = macro_avg["f1-score"]

                # Calculate accuracy
                accuracy = np.sum(preds == test_labels) / len(test_labels)

                # CRITICAL FIX: Extract per-class metrics ensuring correct order
                # When sklearn returns report dict, classes might be strings or ints
                # We MUST extract in the correct order (0, 1, 2, ...) regardless
                precisions = []
                recalls = []
                f1_scores = []
                supports = []

                for i in range(num_labels):
                    # Try multiple key formats to handle sklearn inconsistencies
                    class_key = None
                    class_metrics = None

                    # Try string key first
                    if str(i) in report:
                        class_key = str(i)
                        class_metrics = report[str(i)]
                    # Try int key
                    elif i in report:
                        class_key = i
                        class_metrics = report[i]
                    # Try with label names if provided
                    elif label_names and i < len(label_names) and label_names[i] in report:
                        class_key = label_names[i]
                        class_metrics = report[label_names[i]]

                    if class_metrics:
                        precisions.append(class_metrics.get("precision", 0))
                        recalls.append(class_metrics.get("recall", 0))
                        f1_scores.append(class_metrics.get("f1-score", 0))
                        supports.append(int(class_metrics.get("support", 0)))
                    else:
                        # Class not in report - use zeros
                        precisions.append(0.0)
                        recalls.append(0.0)
                        f1_scores.append(0.0)
                        supports.append(0)

                # For backward compatibility with binary classification code
                if num_labels == 2:
                    precision_0, precision_1 = precisions[0], precisions[1]
                    recall_0, recall_1 = recalls[0], recalls[1]
                    f1_0, f1_1 = f1_scores[0], f1_scores[1]
                    support_0, support_1 = supports[0], supports[1]

                # Update display with performance metrics
                display.accuracy = accuracy
                display.precision = precisions
                display.recall = recalls
                display.f1_scores = f1_scores
                display.f1_macro = macro_f1
                display.support = supports
                # Update with calculated metrics
                throttled_live.update(display.create_panel(), force=True)

                # Initialize language_metrics (will be populated if tracking languages)
                language_metrics = {}

                # Calculate and display per-language metrics if requested
                if track_languages and language_info is not None:
                    # CRITICAL FIX: Filter out NaN/None/float values and normalize to uppercase
                    unique_languages = list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang]))

                    # CRITICAL: Initialize language_metrics for ALL detected_languages (not just those in validation data)
                    # This ensures all languages appear in the display even if not present in current split
                    if hasattr(self, 'detected_languages') and self.detected_languages:
                        for lang in self.detected_languages:
                            if lang not in language_metrics:
                                # Initialize with zero metrics - will be updated if language has data
                                language_metrics[lang] = {
                                    'samples': 0,
                                    'accuracy': 0.0,
                                    'macro_f1': 0.0
                                }
                                for i in range(num_labels):
                                    language_metrics[lang][f'precision_{i}'] = 0.0
                                    language_metrics[lang][f'recall_{i}'] = 0.0
                                    language_metrics[lang][f'f1_{i}'] = 0.0
                                    language_metrics[lang][f'support_{i}'] = 0

                    for lang_upper in sorted(unique_languages):
                        # Get indices for this language (compare with both upper and lower case)
                        lang_indices = [i for i, l in enumerate(language_info) if isinstance(l, str) and l.upper() == lang_upper]
                        if not lang_indices:
                            continue

                        # Get predictions and labels for this language
                        lang_preds = preds[lang_indices]
                        lang_labels = np.array(test_labels)[lang_indices]

                        # Calculate metrics - force all labels to be included
                        lang_report = classification_report(lang_labels, lang_preds, labels=all_labels,
                                                           output_dict=True, zero_division=0)

                        lang_acc = lang_report.get('accuracy', 0)
                        lang_macro_f1 = lang_report.get('macro avg', {}).get('f1-score', 0)
                        lang_support = len(lang_indices)

                        # Extract metrics for all classes
                        lang_metrics_dict = {
                            'samples': lang_support,
                            'accuracy': lang_acc,
                        }

                        for i in range(num_labels):
                            # CRITICAL FIX: Try multiple key formats for consistency
                            class_metrics = None

                            if str(i) in lang_report:
                                class_metrics = lang_report[str(i)]
                            elif i in lang_report:
                                class_metrics = lang_report[i]

                            if class_metrics:
                                lang_metrics_dict[f'precision_{i}'] = class_metrics.get('precision', 0)
                                lang_metrics_dict[f'recall_{i}'] = class_metrics.get('recall', 0)
                                lang_metrics_dict[f'f1_{i}'] = class_metrics.get('f1-score', 0)
                                lang_metrics_dict[f'support_{i}'] = int(class_metrics.get('support', 0))
                            else:
                                # Class not present for this language
                                lang_metrics_dict[f'precision_{i}'] = 0
                                lang_metrics_dict[f'recall_{i}'] = 0
                                lang_metrics_dict[f'f1_{i}'] = 0
                                lang_metrics_dict[f'support_{i}'] = 0

                        # CRITICAL: Store both f1_macro and macro_f1 for consistency
                        lang_metrics_dict['f1_macro'] = lang_macro_f1
                        lang_metrics_dict['macro_f1'] = lang_macro_f1  # Keep backward compatibility

                        language_metrics[lang_upper] = lang_metrics_dict

                    # Update display with language metrics
                    display.language_metrics = language_metrics

                    # Calculate language variance for F1 class 1
                    f1_class1_values = [m['f1_1'] for m in language_metrics.values() if m['support_1'] >= 3]
                    if len(f1_class1_values) > 1:
                        # CRITICAL: Convert to Python float to avoid numpy scalar issues
                        mean_f1 = float(np.mean(f1_class1_values))
                        std_f1 = np.std(f1_class1_values)
                        display.language_variance = (std_f1 / mean_f1) if mean_f1 > 0 else 0

                    # Update with calculated metrics
                    throttled_live.update(display.create_panel(), force=True)

                    # Store language metrics for this epoch
                    averages: Optional[Dict[str, float]] = None
                    if language_metrics:
                        avg_acc = sum(m['accuracy'] for m in language_metrics.values()) / len(language_metrics)
                        # CRITICAL: Both keys are now available in metrics, use f1_macro
                        avg_f1 = sum(m.get('f1_macro', m.get('macro_f1', 0)) for m in language_metrics.values()) / len(language_metrics)
                        # Store both keys for consistency
                        averages = {'accuracy': avg_acc, 'f1_macro': avg_f1, 'macro_f1': avg_f1}

                        epoch_record = {
                            'epoch': i_epoch + 1,
                            'metrics': language_metrics
                        }
                        if averages is not None:
                            epoch_record['averages'] = averages
                        language_performance_history.append(epoch_record)

                # NOTE: CSV writing will be done after combined_metric calculation

                # Update global completed epochs in display and model
                # CRITICAL: Do this BEFORE calling callback so it gets the updated count
                if global_total_models is not None:
                    display.global_completed_epochs += 1
                    self.global_completed_epochs = display.global_completed_epochs

                # Call progress callback if provided
                if progress_callback is not None:
                    callback_metrics = {
                        'epoch': i_epoch + 1,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'accuracy': accuracy,
                        'f1_macro': macro_f1,
                        'global_completed_epochs': display.global_completed_epochs if global_total_models is not None else None
                    }
                    # Add per-class F1 scores
                    for i in range(num_labels):
                        callback_metrics[f'f1_{i}'] = f1_scores[i]
                    # Add binary compatibility
                    if num_labels == 2:
                        callback_metrics['f1_score'] = f1_1  # Binary: use f1_1 as main f1_score
                    else:
                        callback_metrics['f1_score'] = macro_f1  # Multi-class: use macro_f1

                    try:
                        progress_callback(**callback_metrics)
                    except Exception as e:
                        # Don't fail training if callback fails
                        self.logger.warning(f"Progress callback failed: {e}")

                # Compute "combined" metric if best_model_criteria is "combined"
                if best_model_criteria == "combined":
                    # Binary classification: use weighted combination of f1_1 and macro_f1
                    if num_labels == 2:
                        combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1
                    else:
                        # Multi-class: use macro_f1 directly
                        combined_metric = macro_f1

                    # Apply language balance penalty if we have per-language metrics
                    if language_metrics:
                        if num_labels == 2:
                            # Binary: Calculate variance in F1 class 1 across languages
                            f1_class1_values = [m.get('f1_1', 0) for m in language_metrics.values()
                                               if m.get('support_1', 0) > 0]  # Only count languages with positive examples
                        else:
                            # Multi-class: Calculate variance in macro_f1 across languages
                            # CRITICAL: Use f1_macro with fallback to macro_f1
                            f1_class1_values = [m.get('f1_macro', m.get('macro_f1', 0)) for m in language_metrics.values()]

                        if len(f1_class1_values) > 1:
                            # Calculate coefficient of variation (std / mean) as a measure of imbalance
                            # CRITICAL: Convert to Python float to avoid numpy scalar issues
                            mean_f1_class1 = float(np.mean(f1_class1_values))
                            std_f1_class1 = np.std(f1_class1_values)

                            # Avoid division by zero
                            if mean_f1_class1 > 0:
                                cv = std_f1_class1 / mean_f1_class1  # Coefficient of variation
                                # Apply penalty: reduce combined_metric by up to 20% based on variance
                                # CV of 0 = no penalty, CV of 1.0+ = full 20% penalty
                                language_balance_penalty = min(cv * 0.2, 0.2)
                                combined_metric = combined_metric * (1 - language_balance_penalty)
                else:
                    # Fallback or alternative strategy
                    if num_labels == 2:
                        combined_metric = (f1_1 + macro_f1) / 2.0
                    else:
                        # Multi-class: use macro_f1 directly
                        combined_metric = macro_f1

                # Compute epoch timing
                epoch_time = time.time() - epoch_start_time
                display.epoch_time = epoch_time
                display.total_time = time.time() - training_start_time

                # NOW write to training_metrics.csv with the calculated combined_metric
                with open(training_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Get timestamp for this entry
                    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Build row with combined_metric
                    row = [
                        self.model_name if hasattr(self, 'model_name') else self.__class__.__name__,
                        current_timestamp,
                        i_epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        combined_metric,    # Use the actual combined_metric calculated above
                        accuracy,
                    ]

                    # Add per-class metrics for all classes
                    for i in range(num_labels):
                        row.extend([
                            precisions[i],
                            recalls[i],
                            f1_scores[i],
                            supports[i]
                        ])

                    row.append(macro_f1)

                    # Add language metrics ONLY for the current training language
                    if track_languages and language_info is not None:
                        # Determine which languages to include (must match headers logic)
                        if language and language != 'MULTI':
                            # Single language training: only write data for THIS language
                            langs_for_data = [language.upper()]
                        else:
                            # Multilingual or unspecified: write data for all detected languages
                            langs_for_data = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))

                        for lang in langs_for_data:
                            if language_metrics and lang in language_metrics:
                                row.append(language_metrics[lang]['accuracy'])
                                for i in range(num_labels):
                                    row.extend([
                                        language_metrics[lang][f'precision_{i}'],
                                        language_metrics[lang][f'recall_{i}'],
                                        language_metrics[lang][f'f1_{i}'],
                                        language_metrics[lang][f'support_{i}']
                                    ])
                                # CRITICAL: Use fallback for f1_macro
                                row.append(language_metrics[lang].get('f1_macro', language_metrics[lang].get('macro_f1', 0)))
                            else:
                                # Default values: 1 accuracy + num_labels*4 metrics + 1 macro_f1
                                row.extend([0] * (1 + num_labels * 4 + 1))

                    writer.writerow(row)

                # Check if this is a new best model
                if combined_metric > best_metric_val:
                    # We found a new best model
                    display.improvement = combined_metric - best_metric_val
                    display.best_f1 = macro_f1
                    display.best_epoch = i_epoch + 1
                    display.combined_metric = combined_metric
                    # Update with calculated metrics
                    throttled_live.update(display.create_panel(), force=True)

                # Remove old best model folder if it exists
                if best_model_path is not None:
                    try:
                        shutil.rmtree(best_model_path)
                    except OSError:
                        pass

                    best_metric_val = combined_metric
                    best_combined_metric_value = combined_metric
                    best_accuracy_value = accuracy
                    best_macro_f1_value = macro_f1
                    best_epoch_index = i_epoch + 1

                    # Always save best_scores for reinforced learning trigger check
                    best_scores = precision_recall_fscore_support(test_labels, preds)

                    # Save language metrics from best epoch (if available)
                    if track_languages and 'language_metrics' in locals():
                        best_language_metrics = copy.deepcopy(language_metrics)

                        language_averages_for_best = None
                        if language_performance_history:
                            for record in reversed(language_performance_history):
                                if record.get('epoch') == i_epoch + 1:
                                    language_averages_for_best = record.get('averages')
                                    break
                        if language_averages_for_best is None:
                            language_averages_for_best = self._compute_language_averages(language_metrics)
                        best_language_averages = copy.deepcopy(language_averages_for_best)
                    else:
                        language_averages_for_best = None

                    if save_model_as is not None:
                        # Save the new best model in session-organized directory
                        # Use model_category_dir from session structure
                        best_model_path = os.path.join(model_category_dir, f"{save_model_as}_epoch_{i_epoch+1}")
                        os.makedirs(best_model_path, exist_ok=True)

                        model_to_save = model.module if hasattr(model, 'module') else model

                        # CRITICAL: Ensure label mappings are in config before saving
                        # This ensures annotation studio reducer mode can access label names
                        if label_names:
                            model_to_save.config.id2label = {i: name for i, name in enumerate(label_names)}
                            model_to_save.config.label2id = {name: i for i, name in enumerate(label_names)}

                        primary_lang_code, confirmed_languages = self._collect_language_codes(
                            language,
                            language_info,
                        )
                        self._apply_languages_to_config(
                            model_to_save.config,
                            primary_lang_code,
                            confirmed_languages,
                        )

                        output_model_file = os.path.join(best_model_path, WEIGHTS_NAME)
                        output_config_file = os.path.join(best_model_path, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        self.tokenizer.save_vocabulary(best_model_path)

                        # Save training metadata for annotation studio
                        metadata_file = os.path.join(best_model_path, "training_metadata.json")

                        training_metadata = {
                            "model_type": model_type,
                            "training_approach": training_approach,
                            "num_labels": num_labels,
                            "label_names": label_names if label_names else [],
                            "label_key": label_key if label_key else None,
                            "label_value": label_value if label_value else None,
                            "language": primary_lang_code if primary_lang_code else (language.upper() if isinstance(language, str) and language else None),
                            "confirmed_languages": confirmed_languages,
                            "epoch": i_epoch + 1,
                            "combined_metric": combined_metric,
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                            "training_phase": "normal",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        final_metrics_block = self._build_final_metrics_block(
                            combined_metric=combined_metric,
                            macro_f1=macro_f1,
                            accuracy=accuracy,
                            epoch=i_epoch + 1,
                            train_loss=avg_train_loss,
                            val_loss=avg_val_loss,
                            precisions=precisions,
                            recalls=recalls,
                            f1_scores=f1_scores,
                            supports=supports,
                            label_names=label_names,
                            language_metrics=language_metrics if track_languages else None,
                            language_averages=language_averages_for_best,
                        )
                        if final_metrics_block:
                            training_metadata["final_metrics"] = final_metrics_block
                            best_final_metrics_block = copy.deepcopy(final_metrics_block)
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(training_metadata, f, indent=2, ensure_ascii=False)
                    else:
                        best_model_path = None

                    # Check if this is truly the best model for this model type
                    # Read existing best_models.csv to check if we already have a better model
                    should_update_best = True

                    if os.path.exists(best_models_csv) and os.path.getsize(best_models_csv) > 0:
                        # Read existing best models to check if this model type already has a better score
                        with open(best_models_csv, 'r', encoding='utf-8') as f_read:
                            # Read all lines first
                            all_lines = f_read.readlines()

                            # Find where data starts (skip metadata lines starting with # and empty lines)
                            data_start_idx = 0
                            for i, line in enumerate(all_lines):
                                if not line.startswith('#') and line.strip() != '':
                                    data_start_idx = i
                                    break

                            # Create a reader from the data lines
                            if data_start_idx < len(all_lines):
                                import io
                                data_content = ''.join(all_lines[data_start_idx:])
                                reader = csv.DictReader(io.StringIO(data_content))

                                for existing_row in reader:
                                    # Check if same model type and identifier
                                    if (existing_row.get('model_identifier') == (model_identifier if model_identifier else "") and
                                        existing_row.get('model_type') == model_type):
                                        # Compare scores
                                        existing_macro_f1 = float(existing_row.get('macro_f1', 0))
                                        if existing_macro_f1 >= macro_f1:
                                            should_update_best = False
                                            break

                    if should_update_best:
                        # First, remove any existing entry for this model type/identifier combination
                        if os.path.exists(best_models_csv) and os.path.getsize(best_models_csv) > 0:
                            # Read all rows, skipping metadata lines
                            rows_to_keep = []
                            metadata_lines = []
                            headers_dict = None

                            with open(best_models_csv, 'r', encoding='utf-8') as f_read:
                                # Read all lines first
                                all_lines = f_read.readlines()

                                # Separate metadata from data
                                data_start_idx = 0
                                for i, line in enumerate(all_lines):
                                    if line.startswith('#') or line.strip() == '':
                                        metadata_lines.append(line)
                                    else:
                                        # This is the header line
                                        data_start_idx = i
                                        break

                                # Create a reader from the data lines
                                if data_start_idx < len(all_lines):
                                    import io
                                    data_content = ''.join(all_lines[data_start_idx:])
                                    reader = csv.DictReader(io.StringIO(data_content))
                                    headers_dict = reader.fieldnames

                                    for row in reader:
                                        # Keep rows that are NOT the same model type/identifier
                                        if not (row.get('model_identifier') == (model_identifier if model_identifier else "") and
                                               row.get('model_type') == model_type):
                                            rows_to_keep.append(row)

                            # Rewrite the file without the old entry
                            if headers_dict:
                                with open(best_models_csv, 'w', newline='', encoding='utf-8') as f_write:
                                    # Write metadata lines first
                                    for meta_line in metadata_lines:
                                        f_write.write(meta_line)
                                    # Write CSV data
                                    writer = csv.DictWriter(f_write, fieldnames=headers_dict)
                                    writer.writeheader()
                                    writer.writerows(rows_to_keep)

                        # Now append the new best model
                        with open(best_models_csv, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)

                            # Get timestamp
                            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                            # CRITICAL: Use the actual combined_metric that determined this is the best model
                            # This is the exact same sophisticated score used for model selection
                            # NOT a recalculation - we use the value that was already calculated

                            # Create row matching the headers exactly
                            row = [
                                model_type,
                                current_timestamp,
                                i_epoch + 1,
                                avg_train_loss,
                                avg_val_loss,
                                combined_metric,                        # CRITICAL: The actual combined_metric used for selection
                                accuracy,                               # Overall accuracy
                            ]

                            # Add per-class metrics for all classes
                            for i in range(num_labels):
                                row.extend([
                                    precisions[i] if i < len(precisions) else 0,
                                    recalls[i] if i < len(recalls) else 0,
                                    f1_scores[i] if i < len(f1_scores) else 0,
                                    supports[i] if i < len(supports) else 0
                                ])

                            row.append(macro_f1)

                            # Add language metrics ONLY for the current training language
                            if track_languages and language_info is not None:
                                # Determine which languages to include (must match headers logic)
                                if language and language != 'MULTI':
                                    # Single language training: only write data for THIS language
                                    langs_for_data = [language.upper()]
                                else:
                                    # Multilingual or unspecified: write data for all detected languages
                                    langs_for_data = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))
                            else:
                                langs_for_data = []

                            for lang in langs_for_data:
                                if track_languages and language_info is not None and language_metrics and lang in language_metrics:
                                    row.append(language_metrics[lang]['accuracy'])
                                    # Add per-class metrics for this language
                                    for i in range(num_labels):
                                        row.extend([
                                            language_metrics[lang].get(f'precision_{i}', 0),
                                            language_metrics[lang].get(f'recall_{i}', 0),
                                            language_metrics[lang].get(f'f1_{i}', 0),
                                            language_metrics[lang].get(f'support_{i}', 0)
                                        ])
                                    # CRITICAL: Use fallback for f1_macro
                                    row.append(language_metrics[lang].get('f1_macro', language_metrics[lang].get('macro_f1', 0)))
                                else:
                                    # Fill with empty values to maintain CSV structure
                                    # 1 accuracy + num_labels*4 metrics + 1 macro_f1
                                    row.extend([''] * (1 + num_labels * 4 + 1))

                            row.extend([
                                best_model_path if best_model_path else "Not saved to disk",
                                "normal"  # training phase
                            ])

                            writer.writerow(row)

                else:
                    # No new best model this epoch, but still update display to show current epoch timing
                    # Update with calculated metrics
                    throttled_live.update(display.create_panel(), force=True)

                # Epoch summary removed - Rich Live display is sufficient
                # (keeping print() was pushing the Rich table down on every epoch)

            # End of normal training (after all epochs) - display final summary
            display.current_phase = "Training Complete"
            display.total_time = time.time() - training_start_time
            # Final update at the end of all epochs
            throttled_live.update(display.create_panel(), force=True)

            # Save language performance history if available
            if track_languages and language_performance_history and best_model_path:
                language_metrics_json = os.path.join(best_model_path, "language_performance.json")
                with open(language_metrics_json, 'w', encoding='utf-8') as f:
                    json.dump(language_performance_history, f, indent=2, ensure_ascii=False)

            # If we have a best model, rename it to the final user-specified name (for normal training)
            final_path = None
            if save_model_as is not None and best_model_path is not None:
                # Use session-organized directory structure
                final_path = os.path.join(model_category_dir, save_model_as)

                # Ensure parent directory exists before moving
                os.makedirs(model_category_dir, exist_ok=True)

                # Remove existing final path if any
                if os.path.exists(final_path):
                    shutil.rmtree(final_path)
                shutil.move(best_model_path, final_path)
                best_model_path = final_path

                # Log model save confirmation
                self.logger.info(f"✅ Best model saved to: {final_path}")
            elif save_model_as is not None and best_model_path is None:
                # Save current model as fallback
                final_path = os.path.join(model_category_dir, save_model_as)

                os.makedirs(final_path, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model

                # CRITICAL: Ensure label mappings are in config before saving
                # This ensures annotation studio reducer mode can access label names
                if label_names:
                    model_to_save.config.id2label = {i: name for i, name in enumerate(label_names)}
                    model_to_save.config.label2id = {name: i for i, name in enumerate(label_names)}

                primary_lang_code, confirmed_languages = self._collect_language_codes(
                    language,
                    language_info,
                )
                self._apply_languages_to_config(
                    model_to_save.config,
                    primary_lang_code,
                    confirmed_languages,
                )

                output_model_file = os.path.join(final_path, WEIGHTS_NAME)
                output_config_file = os.path.join(final_path, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                self.tokenizer.save_vocabulary(final_path)
                best_model_path = final_path

                # Save training metadata for annotation studio
                metadata_file = os.path.join(final_path, "training_metadata.json")

                fallback_combined_metric = best_combined_metric_value if best_combined_metric_value is not None else best_metric_val
                precision_scores = best_scores[0] if best_scores is not None else None
                recall_scores = best_scores[1] if best_scores is not None else None
                f1_scores_scores = best_scores[2] if best_scores is not None else None
                support_scores = best_scores[3] if best_scores is not None else None

                def _to_sequence(values: Optional[Any]) -> List[Any]:
                    if values is None:
                        return []
                    if isinstance(values, np.ndarray):
                        return values.tolist()
                    if isinstance(values, (list, tuple)):
                        return list(values)
                    return [values]

                fallback_macro_f1 = best_macro_f1_value
                f1_values_for_macro = _to_sequence(f1_scores_scores)
                if fallback_macro_f1 is None and f1_values_for_macro:
                    fallback_macro_f1 = float(np.mean([self._safe_float(v) for v in f1_values_for_macro]))

                fallback_accuracy = best_accuracy_value
                precision_values_for_acc = _to_sequence(precision_scores)
                support_values_for_acc = _to_sequence(support_scores)
                if fallback_accuracy is None and precision_values_for_acc and support_values_for_acc:
                    denom = sum(self._safe_int(s) for s in support_values_for_acc)
                    if denom:
                        fallback_accuracy = sum(
                            self._safe_float(p) * self._safe_int(s)
                            for p, s in zip(precision_values_for_acc, support_values_for_acc)
                        ) / denom

                epoch_value = best_epoch_index if best_epoch_index is not None else num_train_epochs

                final_metrics_block = None
                if best_final_metrics_block:
                    final_metrics_block = copy.deepcopy(best_final_metrics_block)
                else:
                    final_metrics_block = self._build_final_metrics_block(
                        combined_metric=fallback_combined_metric,
                        macro_f1=fallback_macro_f1,
                        accuracy=fallback_accuracy,
                        epoch=epoch_value,
                        train_loss=None,
                        val_loss=None,
                        precisions=precision_scores,
                        recalls=recall_scores,
                        f1_scores=f1_scores_scores,
                        supports=support_scores,
                        label_names=label_names,
                        language_metrics=best_language_metrics,
                        language_averages=best_language_averages,
                    )

                training_metadata = {
                    "model_type": model_type,
                    "training_approach": training_approach,
                    "num_labels": num_labels,
                    "label_names": label_names if label_names else [],
                    "label_key": label_key if label_key else None,
                    "label_value": label_value if label_value else None,
                    "language": primary_lang_code if primary_lang_code else (language.upper() if isinstance(language, str) and language else None),
                    "confirmed_languages": confirmed_languages,
                    "final_epoch": epoch_value,
                    "combined_metric": self._safe_float(fallback_combined_metric) if fallback_combined_metric is not None else None,
                    "macro_f1": self._safe_float(fallback_macro_f1) if fallback_macro_f1 is not None else None,
                    "accuracy": self._safe_float(fallback_accuracy) if fallback_accuracy is not None else None,
                    "training_phase": "normal",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                if final_metrics_block:
                    training_metadata["final_metrics"] = final_metrics_block
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(training_metadata, f, indent=2, ensure_ascii=False)

                # Log model save confirmation
                self.logger.info(f"✅ Model saved to: {final_path}")

            # ==================== Reinforced Training Check ====================
            # Use a flag to ensure reinforced learning only triggers once per training session
            if not hasattr(self, '_reinforced_already_triggered'):
                self._reinforced_already_triggered = False

            reinforced_triggered = False
            # Reinforcement learning now supports both binary and multi-class
            if best_scores is not None and reinforced_learning and n_epochs_reinforced > 0 and not self._reinforced_already_triggered:
                # Extract metrics from best_scores
                best_precision = best_scores[0]  # (precision_0, precision_1, ...)
                best_recall = best_scores[1]     # (recall_0, recall_1, ...)
                best_f1_scores = best_scores[2]  # (f1_0, f1_1, ...)
                best_support = best_scores[3]    # (support_0, support_1, ...)

                # CRITICAL DEBUG: Log best_f1_scores type to diagnose numpy.int64 has no len() error
                self.logger.debug(f"REINFORCED LEARNING CHECK:")
                self.logger.debug(f"  best_scores type: {type(best_scores)}")
                self.logger.debug(f"  best_f1_scores type: {type(best_f1_scores)}, value: {best_f1_scores}")
                self.logger.debug(f"  best_precision type: {type(best_precision)}, value: {best_precision}")
                self.logger.debug(f"  best_recall type: {type(best_recall)}, value: {best_recall}")
                self.logger.debug(f"  best_support type: {type(best_support)}, value: {best_support}")

                # CRITICAL: Convert numpy arrays to ensure they are iterable
                import numpy as np
                if isinstance(best_f1_scores, (np.integer, np.floating)):
                    # Scalar - wrap in list
                    self.logger.warning(f"⚠️ best_f1_scores is a scalar {type(best_f1_scores)}: {best_f1_scores}")
                    best_f1_scores = [float(best_f1_scores)]
                    best_precision = [float(best_precision)]
                    best_recall = [float(best_recall)]
                    best_support = [int(best_support)]
                elif hasattr(best_f1_scores, 'tolist'):
                    # Numpy array - convert to list for safety
                    self.logger.debug(f"Converting numpy arrays to lists for safety")
                    best_f1_scores = best_f1_scores.tolist()
                    best_precision = best_precision.tolist()
                    best_recall = best_recall.tolist()
                    best_support = best_support.tolist()

                # Handle binary vs multi-class
                if num_labels == 2:
                    # Binary classification
                    try:
                        f1_scores_len = len(best_f1_scores)
                    except TypeError as e:
                        self.logger.error(f"❌ ERROR: Cannot get len() of best_f1_scores: type={type(best_f1_scores)}, value={best_f1_scores}")
                        raise TypeError(f"Cannot get len() of best_f1_scores: type={type(best_f1_scores)}, value={best_f1_scores}") from e

                    if f1_scores_len >= 2:
                        best_f1_0 = best_f1_scores[0]
                        best_f1_1 = best_f1_scores[1]
                        best_support_0 = int(best_support[0])
                        best_support_1 = int(best_support[1])
                    else:
                        # Single class - use the only F1 score
                        best_f1_0 = 0.0
                        best_f1_1 = best_f1_scores[0]
                        best_support_0 = 0
                        best_support_1 = int(best_support[0])

                    # Use intelligent trigger logic for binary
                    trigger_score, should_trigger, trigger_reason = self.calculate_reinforced_trigger_score(
                        f1_class_0=best_f1_0,
                        f1_class_1=best_f1_1,
                        support_class_0=best_support_0,
                        support_class_1=best_support_1,
                        language_metrics=best_language_metrics,
                        reinforced_f1_threshold=reinforced_f1_threshold
                    )
                else:
                    # Multi-class: trigger if ANY class has low F1
                    worst_f1 = min(best_f1_scores) if len(best_f1_scores) > 0 else 0.0
                    avg_f1 = sum(best_f1_scores) / len(best_f1_scores) if len(best_f1_scores) > 0 else 0.0

                    # Trigger if worst class F1 < threshold/2 or average F1 < threshold
                    # Using reinforced_f1_threshold for consistency
                    should_trigger = worst_f1 < (reinforced_f1_threshold * 0.7) or avg_f1 < reinforced_f1_threshold
                    trigger_score = worst_f1
                    trigger_reason = f"Multi-class: worst F1={worst_f1:.2f}, avg F1={avg_f1:.2f}"

                # Update display with reinforced learning threshold info
                display.reinforced_threshold = trigger_score
                display.reinforced_triggered = should_trigger

                # Trigger reinforced learning if needed (don't log - would break Rich Live display)
                if should_trigger:
                    reinforced_triggered = True
                    self._reinforced_already_triggered = True  # Mark as triggered to prevent re-triggering

                    # Update global total epochs to account for additional reinforced learning epochs
                    if self.global_total_epochs is not None:
                        self.global_total_epochs += n_epochs_reinforced
                        display.global_total_epochs = self.global_total_epochs

                    # ========== INLINE REINFORCED TRAINING (ROBUST SOLUTION) ==========
                    # Instead of calling separate function, run reinforced training INLINE
                    # within the SAME Live context to ensure display updates properly

                    # Switch display to reinforced mode
                    display.is_reinforced = True
                    display.n_epochs = n_epochs_reinforced
                    display.current_epoch = 0
                    display.current_phase = "🔥 Starting Reinforced Training"

                    # Reset all metrics for clean transition
                    display.train_progress = 0
                    display.val_progress = 0
                    display.train_total = 0
                    display.val_total = 0

                    # IMMEDIATELY update display to show transition (prevents panel stacking)
                    throttled_live.update(display.create_panel(), force=True)
                    throttled_live.flush_pending()  # Ensure update is immediately visible
                    time.sleep(0.2)  # Brief pause for clean visual transition

                    # Prepare reinforced training setup
                    # Use per-model naming for reinforced metrics
                    reinforced_metrics_csv = training_metrics_csv.replace('_training.csv', '_reinforced.csv').replace('training.csv', 'reinforced.csv')

                    # Create headers for reinforced metrics CSV
                    # CRITICAL: Use same structure as normal training CSV with standardized indices
                    reinforced_headers = [
                        "model_name",
                        "label_key",        # Multi-label: key (e.g., 'themes', 'sentiment')
                        "label_value",      # Multi-label: value (e.g., 'transportation', 'positive')
                        "language",         # Language of the data (e.g., 'EN', 'FR', 'MULTI')
                        "timestamp",
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "accuracy",         # Overall accuracy
                    ]

                    # Add per-class metric headers dynamically based on num_labels
                    # CRITICAL: Use standardized indices (_0, _1, _2) for consistency
                    for i in range(num_labels):
                        reinforced_headers.extend([
                            f"precision_{i}",
                            f"recall_{i}",
                            f"f1_{i}",
                            f"support_{i}"
                        ])

                    reinforced_headers.append("macro_f1")

                    # Add language-specific headers ONLY for the current training language
                    # Individual CSVs should only contain metrics for their specific language
                    if track_languages and language_info is not None:
                        # Determine which languages to include in headers
                        if language and language != 'MULTI':
                            # Single language training: only add columns for THIS language
                            langs_for_headers = [language.upper()]
                        else:
                            # Multilingual or unspecified: add columns for all detected languages
                            langs_for_headers = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))

                        for lang in langs_for_headers:
                            reinforced_headers.append(f"{lang}_accuracy")
                            for i in range(num_labels):
                                reinforced_headers.extend([
                                    f"{lang}_precision_{i}",
                                    f"{lang}_recall_{i}",
                                    f"{lang}_f1_{i}",
                                    f"{lang}_support_{i}"
                                ])
                            reinforced_headers.append(f"{lang}_macro_f1")

                    with open(reinforced_metrics_csv, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # Write legend as comment row (starts with #) - same as normal training CSV
                        f.write(f"# {class_legend}\n")
                        # Write column headers
                        writer.writerow(reinforced_headers)

                    # Extract dataset from train_dataloader and apply WeightedRandomSampler
                    dataset = train_dataloader.dataset
                    labels = dataset.tensors[2].numpy()

                    class_sample_count = np.bincount(labels)
                    weight_per_class = 1.0 / class_sample_count
                    sample_weights = [weight_per_class[t] for t in labels]

                    sampler = WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True
                    )

                    # Build new train dataloader with bigger batch size
                    new_batch_size = 64
                    new_train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=new_batch_size)

                    # Get intelligent reinforced parameters
                    from .reinforced_params import get_reinforced_params, should_use_advanced_techniques

                    model_name_for_params = self.__class__.__name__
                    reinforced_params = get_reinforced_params(
                        model_name_for_params,
                        worst_f1 if num_labels > 2 else best_f1_1,
                        lr,
                        num_classes=num_labels,
                        class_f1_scores=list(best_f1_scores) if num_labels > 2 else None
                    )
                    advanced_techniques = should_use_advanced_techniques(worst_f1 if num_labels > 2 else best_f1_1)

                    new_lr = reinforced_params['learning_rate']

                    # Create weight tensor based on number of classes
                    if num_labels == 2:
                        # Binary: use class_1_weight
                        pos_weight_val = reinforced_params['class_1_weight']
                        weight_tensor = torch.tensor([1.0, pos_weight_val], dtype=torch.float)
                    else:
                        # Multi-class: use class_weights
                        if reinforced_params['class_weights'] is not None:
                            weight_tensor = torch.tensor(reinforced_params['class_weights'], dtype=torch.float)
                        else:
                            # Fallback: uniform weights
                            weight_tensor = torch.ones(num_labels, dtype=torch.float)

                    if 'n_epochs' in reinforced_params:
                        # Use manual reinforced_epochs if provided, otherwise use auto-calculated
                        if reinforced_epochs is None:
                            n_epochs_reinforced = reinforced_params['n_epochs']
                        display.n_epochs = n_epochs_reinforced

                    # Load best model as starting point (suppress logs to avoid interfering with Rich display)
                    if best_model_path is not None:
                        model_state = torch.load(os.path.join(best_model_path, WEIGHTS_NAME), map_location=self.device)
                        model.load_state_dict(model_state)

                    # Create new optimizer and scheduler for reinforced training
                    optimizer = AdamW(model.parameters(), lr=new_lr, eps=1e-8)
                    total_steps = len(new_train_dataloader) * n_epochs_reinforced
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=total_steps
                    )

                    # Track reinforced training time
                    reinforced_start_time = time.time()

                    # Reinforced training loop - INLINE within same Live context
                    # NOTE: Suppress logger during reinforced training to avoid interfering with Rich Live display
                    for epoch in range(n_epochs_reinforced):
                        epoch_start_time = time.time()

                        # Update display for new epoch
                        display.current_epoch = epoch + 1
                        display.current_phase = f"🔥 Reinforced Epoch {epoch + 1}/{n_epochs_reinforced}"
                        display.train_total = len(new_train_dataloader)
                        display.train_progress = 0
                        # Update with calculated metrics
                        throttled_live.update(display.create_panel(), force=True)  # ✅ INLINE update - same context

                        # Training phase
                        t0 = time.time()
                        model.train()
                        running_loss = 0.0

                        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

                        for step, train_batch in enumerate(new_train_dataloader):
                            b_inputs = train_batch[0].to(self.device)
                            b_masks = train_batch[1].to(self.device)
                            b_labels = train_batch[2].to(self.device)

                            model.zero_grad()
                            # Call model with or without token_type_ids depending on model support
                            if self._supports_token_type_ids(model):
                                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                            else:
                                outputs = model(b_inputs, attention_mask=b_masks)
                            logits = outputs[0]

                            loss = criterion(logits, b_labels)
                            running_loss += loss.item()

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                            optimizer.step()
                            scheduler.step()

                            # Update display with progress
                            display.train_progress = step + 1
                            display.train_loss = loss.item()
                            display.epoch_time = time.time() - epoch_start_time
                            display.total_time = time.time() - reinforced_start_time
                            # Update with calculated metrics
                            throttled_live.update(display.create_panel(), force=True)  # ✅ INLINE update

                        avg_train_loss = running_loss / len(new_train_dataloader)
                        display.train_loss = avg_train_loss
                        display.train_time = time.time() - t0
                        display.current_phase = "Validation (Reinforced)"
                        display.val_total = len(test_dataloader)
                        display.val_progress = 0
                        # Update with calculated metrics
                        throttled_live.update(display.create_panel(), force=True)  # ✅ INLINE update

                        # Validation phase
                        model.eval()
                        total_val_loss = 0.0
                        logits_complete = []
                        eval_labels = []

                        for step, test_batch in enumerate(test_dataloader):
                            b_inputs = test_batch[0].to(self.device)
                            b_masks = test_batch[1].to(self.device)
                            b_labels = test_batch[2].to(self.device)
                            eval_labels.extend(b_labels.cpu().numpy())

                            with torch.no_grad():
                                # Call model with or without token_type_ids depending on model support
                                if self._supports_token_type_ids(model):
                                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
                                else:
                                    outputs = model(b_inputs, attention_mask=b_masks, labels=b_labels)

                            val_loss = outputs.loss
                            val_logits = outputs.logits

                            total_val_loss += val_loss.item()
                            logits_complete.append(val_logits.detach().cpu().numpy())

                            # Update display with validation progress
                            display.val_progress = step + 1
                            display.val_loss = val_loss.item()
                            display.epoch_time = time.time() - epoch_start_time
                            display.total_time = time.time() - reinforced_start_time
                            # Update with calculated metrics
                            throttled_live.update(display.create_panel(), force=True)  # ✅ INLINE update

                        avg_val_loss = total_val_loss / len(test_dataloader)
                        logits_complete = np.concatenate(logits_complete, axis=0)
                        val_preds = np.argmax(logits_complete, axis=1).flatten()

                        # Calculate metrics - force all labels to be included
                        report = classification_report(eval_labels, val_preds, labels=all_labels,
                                                       output_dict=True, zero_division=0)
                        macro_avg = report.get("macro avg", {"f1-score": 0})
                        macro_f1 = macro_avg["f1-score"]
                        accuracy = np.sum(val_preds == np.array(eval_labels)) / len(eval_labels)

                        # CRITICAL FIX: Extract per-class metrics ensuring correct order (reinforced)
                        precisions = []
                        recalls = []
                        f1_scores = []
                        supports = []

                        for i in range(num_labels):
                            class_metrics = None

                            # Try multiple key formats
                            if str(i) in report:
                                class_metrics = report[str(i)]
                            elif i in report:
                                class_metrics = report[i]

                            if class_metrics:
                                precisions.append(class_metrics.get("precision", 0))
                                recalls.append(class_metrics.get("recall", 0))
                                f1_scores.append(class_metrics.get("f1-score", 0))
                                supports.append(int(class_metrics.get("support", 0)))
                            else:
                                precisions.append(0.0)
                                recalls.append(0.0)
                                f1_scores.append(0.0)
                                supports.append(0)

                        # For backward compatibility with binary classification code
                        if num_labels == 2:
                            precision_0, precision_1 = precisions[0], precisions[1]
                            recall_0, recall_1 = recalls[0], recalls[1]
                            f1_0, f1_1 = f1_scores[0], f1_scores[1]
                            support_0, support_1 = supports[0], supports[1]

                        # Update display with metrics
                        display.val_loss = avg_val_loss
                        display.val_time = time.time() - t0 - display.train_time
                        display.accuracy = accuracy
                        display.precision = precisions
                        display.recall = recalls
                        display.f1_scores = f1_scores
                        display.f1_macro = macro_f1
                        display.support = [int(s) for s in supports]

                        # Calculate language-specific metrics if tracking
                        language_metrics = {}
                        if track_languages and language_info is not None:
                            # CRITICAL FIX: Filter out NaN/None/float values and normalize to uppercase
                            unique_languages = list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang]))
                            for lang_upper in sorted(unique_languages):
                                lang_indices = [i for i, l in enumerate(language_info) if isinstance(l, str) and l.upper() == lang_upper]
                                if not lang_indices:
                                    continue

                                lang_preds = val_preds[lang_indices]
                                lang_labels = np.array(eval_labels)[lang_indices]

                                # Force all labels to be included in the report
                                lang_report = classification_report(lang_labels, lang_preds, labels=all_labels,
                                                                   output_dict=True, zero_division=0)
                                lang_acc = lang_report.get('accuracy', 0)
                                lang_macro_f1 = lang_report.get('macro avg', {}).get('f1-score', 0)

                                # Extract metrics for all classes
                                lang_metrics_dict = {'accuracy': lang_acc}

                                for i in range(num_labels):
                                    # CRITICAL FIX: Handle sklearn's inconsistent key format
                                    class_metrics = None
                                    # Try string key first
                                    if str(i) in lang_report:
                                        class_metrics = lang_report[str(i)]
                                    # Try int key
                                    elif i in lang_report:
                                        class_metrics = lang_report[i]
                                    # Try with label names if provided
                                    elif label_names and i < len(label_names) and label_names[i] in lang_report:
                                        class_metrics = lang_report[label_names[i]]

                                    if class_metrics:
                                        lang_metrics_dict[f'precision_{i}'] = class_metrics.get('precision', 0)
                                        lang_metrics_dict[f'recall_{i}'] = class_metrics.get('recall', 0)
                                        lang_metrics_dict[f'f1_{i}'] = class_metrics.get('f1-score', 0)
                                        lang_metrics_dict[f'support_{i}'] = int(class_metrics.get('support', 0))
                                    else:
                                        lang_metrics_dict[f'precision_{i}'] = 0
                                        lang_metrics_dict[f'recall_{i}'] = 0
                                        lang_metrics_dict[f'f1_{i}'] = 0
                                        lang_metrics_dict[f'support_{i}'] = 0

                                # CRITICAL: Store both f1_macro and macro_f1 for consistency
                                lang_metrics_dict['f1_macro'] = lang_macro_f1
                                lang_metrics_dict['macro_f1'] = lang_macro_f1  # Keep backward compatibility

                                language_metrics[lang_upper] = lang_metrics_dict

                            # Update display with language metrics
                            display.language_metrics = language_metrics

                        # Update with calculated metrics
                        throttled_live.update(display.create_panel(), force=True)  # ✅ INLINE update with all metrics

                        # Write epoch metrics to CSV
                        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        resolved_label_key = label_key or category_name or ""
                        resolved_label_value = label_value or label_key or category_name or ""

                        reinforced_row = [
                            self.model_name if hasattr(self, 'model_name') else self.__class__.__name__,
                            resolved_label_key,
                            resolved_label_value,
                            language if language else "MULTI",
                            current_timestamp,
                            epoch + 1,
                            avg_train_loss,
                            avg_val_loss,
                            accuracy,
                        ]

                        # Add per-class metrics
                        for i in range(num_labels):
                            reinforced_row.extend([
                                precisions[i], recalls[i], f1_scores[i], int(supports[i])
                            ])

                        reinforced_row.append(macro_f1)

                        # Add language metrics ONLY for the current training language
                        if track_languages and language_info is not None:
                            # Determine which languages to include (must match headers logic)
                            if language and language != 'MULTI':
                                # Single language training: only write data for THIS language
                                langs_for_data = [language.upper()]
                            else:
                                # Multilingual or unspecified: write data for all detected languages
                                langs_for_data = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))

                            for lang in langs_for_data:
                                if lang in language_metrics:
                                    lm = language_metrics[lang]
                                    reinforced_row.append(lm['accuracy'])
                                    for i in range(num_labels):
                                        reinforced_row.extend([
                                            lm[f'precision_{i}'], lm[f'recall_{i}'],
                                            lm[f'f1_{i}'], lm[f'support_{i}']
                                        ])
                                    reinforced_row.append(lm['macro_f1'])
                                else:
                                    # Default values: 1 accuracy + num_labels*4 metrics + 1 macro_f1
                                    reinforced_row.extend([0] * (1 + num_labels * 4 + 1))

                        with open(reinforced_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(reinforced_row)

                        # Update global completed epochs in display (reinforced)
                        # CRITICAL: Do this BEFORE calling callback so it gets the updated count
                        if global_total_models is not None:
                            display.global_completed_epochs += 1

                        # Call progress callback if provided (reinforced learning)
                        if progress_callback is not None:
                            callback_metrics = {
                                'epoch': n_epochs + epoch + 1,  # Add normal epochs to reinforced epoch number
                                'train_loss': avg_train_loss,
                                'val_loss': avg_val_loss,
                                'accuracy': accuracy,
                                'f1_macro': macro_f1,
                                'global_completed_epochs': display.global_completed_epochs if global_total_models is not None else None
                            }
                            # Add per-class F1 scores
                            for i in range(num_labels):
                                callback_metrics[f'f1_{i}'] = f1_scores[i]
                            # Add binary compatibility
                            if num_labels == 2:
                                callback_metrics['f1_score'] = f1_1  # Binary: use f1_1 as main f1_score
                            else:
                                callback_metrics['f1_score'] = macro_f1  # Multi-class: use macro_f1

                            try:
                                progress_callback(**callback_metrics)
                            except Exception as e:
                                # Don't fail training if callback fails
                                self.logger.warning(f"Progress callback failed (reinforced): {e}")

                        # Check if this is a new best model
                        if num_labels == 2:
                            combined_metric = (1 - f1_class_1_weight) * f1_0 + f1_class_1_weight * f1_1
                        else:
                            # Multi-class: use macro_f1 directly
                            combined_metric = macro_f1

                        if best_model_criteria == "combined":
                            current_metric = combined_metric
                        elif best_model_criteria == "macro_f1":
                            current_metric = macro_f1
                        elif best_model_criteria == "accuracy":
                            current_metric = accuracy
                        else:
                            current_metric = combined_metric

                        if current_metric > best_metric_val:
                            best_metric_val = current_metric
                            best_combined_metric_value = combined_metric
                            best_accuracy_value = accuracy
                            best_macro_f1_value = macro_f1
                            best_epoch_index = epoch + 1

                            # Save new best model
                            if save_model_as is not None:
                                # Use session-organized directory structure
                                temp_reinforced_path = os.path.join(model_category_dir, f"{save_model_as}_reinforced_temp")
                                os.makedirs(temp_reinforced_path, exist_ok=True)

                                model_to_save = model.module if hasattr(model, 'module') else model

                                # CRITICAL: Ensure label mappings are in config before saving
                                # This ensures annotation studio reducer mode can access label names
                                if label_names:
                                    model_to_save.config.id2label = {i: name for i, name in enumerate(label_names)}
                                    model_to_save.config.label2id = {name: i for i, name in enumerate(label_names)}

                                primary_lang_code, confirmed_languages = self._collect_language_codes(
                                    language,
                                    language_info,
                                )
                                self._apply_languages_to_config(
                                    model_to_save.config,
                                    primary_lang_code,
                                    confirmed_languages,
                                )

                                output_model_file = os.path.join(temp_reinforced_path, WEIGHTS_NAME)
                                output_config_file = os.path.join(temp_reinforced_path, CONFIG_NAME)

                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                self.tokenizer.save_vocabulary(temp_reinforced_path)

                                # Save training metadata for annotation studio
                                metadata_file = os.path.join(temp_reinforced_path, "training_metadata.json")

                                language_averages_for_best = None
                                if track_languages:
                                    if language_metrics:
                                        best_language_metrics = copy.deepcopy(language_metrics)
                                        language_averages_for_best = self._compute_language_averages(language_metrics)
                                        best_language_averages = copy.deepcopy(language_averages_for_best)
                                    else:
                                        best_language_metrics = None
                                        best_language_averages = None

                                training_metadata = {
                                    "model_type": self.model_name if hasattr(self, 'model_name') else self.__class__.__name__,
                                    "training_approach": training_approach,
                                    "num_labels": num_labels,
                                    "label_names": label_names if label_names else [],
                                    "label_key": label_key if label_key else None,
                                    "label_value": label_value if label_value else None,
                                    "language": primary_lang_code if primary_lang_code else (language.upper() if isinstance(language, str) and language else None),
                                    "confirmed_languages": confirmed_languages,
                                    "epoch": epoch + 1,
                                    "combined_metric": combined_metric,
                                    "macro_f1": macro_f1,
                                    "accuracy": accuracy,
                                    "training_phase": "reinforced",
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                final_metrics_block = self._build_final_metrics_block(
                                    combined_metric=combined_metric,
                                    macro_f1=macro_f1,
                                    accuracy=accuracy,
                                    epoch=epoch + 1,
                                    train_loss=avg_train_loss,
                                    val_loss=avg_val_loss,
                                    precisions=precisions,
                                    recalls=recalls,
                                    f1_scores=f1_scores,
                                    supports=supports,
                                    label_names=label_names,
                                    language_metrics=language_metrics if track_languages else None,
                                    language_averages=language_averages_for_best,
                                )
                                if final_metrics_block:
                                    training_metadata["final_metrics"] = final_metrics_block
                                    best_final_metrics_block = copy.deepcopy(final_metrics_block)
                                with open(metadata_file, 'w', encoding='utf-8') as f:
                                    json.dump(training_metadata, f, indent=2, ensure_ascii=False)

                                best_model_path = temp_reinforced_path

                                # Update best_scores
                                best_scores = precision_recall_fscore_support(eval_labels, val_preds, average=None, zero_division=0)

                                # Update best_models.csv with reinforced model
                                with open(best_models_csv, mode='a', newline='', encoding='utf-8') as f:
                                    writer = csv.writer(f)
                                    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                                    # CRITICAL: Use the actual combined_metric that was calculated earlier
                                    # This is the exact same sophisticated score used for model selection
                                    # NOT a recalculation - we use the value that determined this is the best model

                                    # Create row matching the headers exactly
                                    row = [
                                        self.model_name if hasattr(self, 'model_name') else self.__class__.__name__,
                                        current_timestamp,
                                        epoch + 1,  # reinforced epoch number
                                        avg_train_loss,
                                        avg_val_loss,
                                        combined_metric,                        # CRITICAL: The actual combined_metric used for selection
                                        accuracy,
                                    ]

                                    # Add per-class metrics
                                    for i in range(num_labels):
                                        row.extend([
                                            precisions[i] if i < len(precisions) else 0,
                                            recalls[i] if i < len(recalls) else 0,
                                            f1_scores[i] if i < len(f1_scores) else 0,
                                            supports[i] if i < len(supports) else 0
                                        ])

                                    row.append(macro_f1)

                                    # Add language metrics ONLY for the current training language
                                    if track_languages and language_info is not None:
                                        # Determine which languages to include (must match headers logic)
                                        if language and language != 'MULTI':
                                            # Single language training: only write data for THIS language
                                            langs_for_data = [language.upper()]
                                        else:
                                            # Multilingual or unspecified: write data for all detected languages
                                            langs_for_data = sorted(list(set([lang.upper() if isinstance(lang, str) and lang else None for lang in language_info if isinstance(lang, str) and lang])))
                                    else:
                                        langs_for_data = []

                                    for lang in langs_for_data:
                                        if track_languages and lang in language_metrics:
                                            lm = language_metrics[lang]
                                            row.append(lm['accuracy'])
                                            for i in range(num_labels):
                                                row.extend([
                                                    lm[f'precision_{i}'], lm[f'recall_{i}'],
                                                    lm[f'f1_{i}'], lm[f'support_{i}']
                                                ])
                                            # CRITICAL: Use fallback for f1_macro
                                            row.append(lm.get('f1_macro', lm.get('macro_f1', 0)))
                                        else:
                                            row.extend([''] * (1 + num_labels * 4 + 1))

                                    row.extend([
                                        temp_reinforced_path,
                                        "reinforced"  # training phase
                                    ])

                                    writer.writerow(row)

                                # Show in display instead of logger (to avoid breaking Rich Live)
                                display.current_phase = f"🔥 NEW BEST! Metric: {current_metric:.4f}"
                                # Update with calculated metrics
                                throttled_live.update(display.create_panel(), force=True)

                    # Finalize reinforced model path
                    if best_model_path and best_model_path.endswith("_reinforced_temp"):
                        final_path = best_model_path.replace("_reinforced_temp", "")
                        if os.path.exists(final_path):
                            shutil.rmtree(final_path)
                        os.rename(best_model_path, final_path)
                        best_model_path = final_path
                        # Log reinforced model save
                        self.logger.info(f"✅ Reinforced model saved to: {final_path}")

                    # Reset display back to normal mode with clean transition
                    display.is_reinforced = False
                    display.current_phase = "✅ Training Complete (Reinforced)"
                    display.train_progress = 0
                    display.val_progress = 0
                    # Final update after reinforced training
                    throttled_live.update(display.create_panel(), force=True)
                    throttled_live.flush_pending()  # Ensure update is immediately visible
                    time.sleep(0.15)  # Brief pause for visual clarity

            if track_languages and language_performance_history and best_model_path:
                history_path = os.path.join(best_model_path, "language_metrics_history.json")
                try:
                    with open(history_path, "w", encoding="utf-8") as history_file:
                        json.dump(language_performance_history, history_file, indent=2, ensure_ascii=False)
                except OSError:
                    pass

            live_stats_summary = throttled_live.get_stats()
            live_stats_min_interval = throttled_live.min_interval
            live_stats_clear_interval = throttled_live.auto_clear_interval

        if live_stats_summary:
            clear_suffix = ""
            if live_stats_clear_interval:
                clear_suffix = f", auto_clear={int(live_stats_clear_interval)}s"
            self.logger.info(
                "Rich live updates summary (min_interval=%.1fs%s): %s",
                live_stats_min_interval,
                clear_suffix,
                live_stats_summary,
            )

        # Finally, if reinforced training was triggered and found a better model, it might have placed it
        # in a temporary folder. The method already handles rename at the end. So at this point we are done.
        # CRITICAL: Calculate total training time
        total_training_time = time.time() - training_start_time

        # Return enhanced scores dictionary for benchmark compatibility
        if best_scores is not None:
            # Convert to dictionary format for new benchmark
            scores_dict = {
                'precision': best_scores[0].tolist() if hasattr(best_scores[0], 'tolist') else best_scores[0],
                'recall': best_scores[1].tolist() if hasattr(best_scores[1], 'tolist') else best_scores[1],
                'f1': best_scores[2].tolist() if hasattr(best_scores[2], 'tolist') else best_scores[2],
                'support': best_scores[3].tolist() if hasattr(best_scores[3], 'tolist') else best_scores[3],
                # CRITICAL: Convert np.mean results to Python floats to avoid numpy scalar issues
                'f1_macro': float(np.mean(best_scores[2])) if best_scores[2] is not None else 0.0,  # CRITICAL: Use f1_macro not macro_f1 for consistency
                'macro_f1': float(np.mean(best_scores[2])) if best_scores[2] is not None else 0.0,  # Keep backward compatibility
                'accuracy': np.sum([p * s for p, s in zip(best_scores[0], best_scores[3])]) / np.sum(best_scores[3]) if best_scores[3] is not None else 0,
                'f1_0': best_scores[2][0] if len(best_scores[2]) > 0 else 0,
                'f1_1': best_scores[2][1] if len(best_scores[2]) > 1 else 0,
                'precision_0': best_scores[0][0] if len(best_scores[0]) > 0 else 0,
                'precision_1': best_scores[0][1] if len(best_scores[0]) > 1 else 0,
                'recall_0': best_scores[1][0] if len(best_scores[1]) > 0 else 0,
                'recall_1': best_scores[1][1] if len(best_scores[1]) > 1 else 0,
                'val_loss': val_loss_values[-1] if val_loss_values else 0,
                'best_model_path': best_model_path,
                'reinforced_triggered': reinforced_triggered,
                'training_time': total_training_time  # CRITICAL: Add training time to summary
            }

            # Add language metrics if available
            if track_languages and language_performance_history:
                scores_dict['language_metrics'] = language_performance_history[-1]['metrics'] if language_performance_history else {}
                scores_dict['language_history'] = language_performance_history

            self.language_metrics_history = language_performance_history
            self.last_training_summary = scores_dict
            self.last_saved_model_path = best_model_path

            # CRITICAL FIX: Load and store best model for downstream use
            # This allows model_trainer.py to access the trained model via model_instance.model
            if best_model_path and os.path.exists(best_model_path):
                # Load the best model from disk (the one with highest F1)
                self.model = self.load_model(best_model_path)
                self.model.to(self.device)
                self.logger.info(f"📦 Loaded best model from: {best_model_path}")
            else:
                # Fallback: use current model in memory (last epoch)
                self.model = model
                self.logger.warning(f"⚠️ No saved model found, using last epoch model in memory")

            # Return the expected 3-tuple for backward compatibility
            return best_metric_val, best_model_path, best_scores

        # Return the expected 3-tuple even when best_scores is None
        # CRITICAL FIX: Store model even when best_scores is None
        if best_model_path and os.path.exists(best_model_path):
            self.model = self.load_model(best_model_path)
            self.model.to(self.device)
            self.logger.info(f"📦 Loaded best model from: {best_model_path}")
        else:
            self.model = model
            self.logger.warning(f"⚠️ No saved model found, using last epoch model in memory")
        return best_metric_val, best_model_path, best_scores

    def predict(
            self,
            dataloader: DataLoader,
            model: Any,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Predict with a trained model.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            Dataloader for prediction, from self.encode().

        model: huggingface model
            A trained model to use for inference.

        proba: bool, default=True
            If True, return probability distributions (softmax).
            If False, return raw logits.

        progress_bar: bool, default=True
            If True, display a progress bar during prediction.

        Returns
        -------
        pred: ndarray of shape (n_samples, n_labels)
            Probabilities or logits for each sample.
        """
        logits_complete = []
        if progress_bar:
            loader = tqdm(dataloader, desc="Predicting")
        else:
            loader = dataloader

        model.eval()

        for batch in loader:
            batch = tuple(t.to(self.device) for t in batch)
            if len(batch) == 3:
                b_input_ids, b_input_mask, _ = batch
            else:
                b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            logits_complete.append(logits)

            del outputs
            torch.cuda.empty_cache()

        pred = np.concatenate(logits_complete, axis=0)

        if progress_bar:
            print(f"label ids: {self.dict_labels}")

        return softmax(pred, axis=1) if proba else pred

    def load_model(
            self,
            model_path: str
    ):
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        model_path: str
            Path to the saved model folder.

        Returns
        -------
        model: huggingface model
            The loaded model instance.
        """
        return self.model_sequence_classifier.from_pretrained(model_path)

    def predict_with_model(
            self,
            dataloader: DataLoader,
            model_path: str,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Convenience method that loads a model from the specified path, moves it to self.device,
        and performs prediction on the given dataloader.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for prediction.

        model_path: str
            Path to the model to load.

        proba: bool, default=True
            If True, return probability distributions (softmax).
            If False, return raw logits.

        progress_bar: bool, default=True
            If True, display a progress bar during prediction.

        Returns
        -------
        ndarray
            Probability or logit predictions.
        """
        model = self.load_model(model_path)
        model.to(self.device)
        return self.predict(dataloader, model, proba, progress_bar)

    def format_time(
            self,
            elapsed: float | int
    ) -> str:
        """
        Format a time duration to hh:mm:ss.

        Parameters
        ----------
        elapsed: float or int
            Elapsed time in seconds.

        Returns
        -------
        str
            The time in hh:mm:ss format.
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

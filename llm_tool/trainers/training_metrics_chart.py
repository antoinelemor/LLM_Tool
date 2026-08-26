#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
training_metrics_chart.py

MAIN OBJECTIVE:
---------------
Generate publication-quality training metrics charts that update in real-time
during model training. Provides visual monitoring of training progress with
loss curves, F1 scores, accuracy, and per-language performance metrics.

Dependencies:
-------------
- os
- json
- logging
- pathlib
- typing
- datetime
- numpy
- matplotlib

MAIN FEATURES:
--------------
1) Generate training/validation loss curves with best model markers
2) Track F1 scores (macro, micro, per-class) over epochs
3) Visualize per-language performance for multilingual models
4) Display learning rate schedule if available
5) Auto-update charts after each epoch for live monitoring
6) Export metrics history as JSON for further analysis

Author:
-------
Antoine Lemor
"""

# Force non-interactive backend for multiprocessing compatibility on macOS
# Must be set BEFORE any pyplot import to avoid "Cannot create a GUI FigureManager
# outside the main thread using the MacOS backend" error in worker processes
import matplotlib
matplotlib.use('Agg')

import os
import os.path
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# Configure logging
logger = logging.getLogger(__name__)

# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Publication-quality style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette for consistent styling
COLORS = {
    'train_loss': '#2563eb',      # Blue
    'val_loss': '#dc2626',        # Red
    'f1_macro': '#059669',        # Green
    'f1_micro': '#7c3aed',        # Purple
    'accuracy': '#d97706',        # Orange
    'lr': '#6b7280',              # Gray
    'combined': '#be185d',        # Pink/Magenta - Combined score
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'info': '#0891b2',
    'dark': '#1f2937',
    'light': '#f3f4f6',
    'muted': '#6b7280',
}

# Per-class colors (for multi-class visualization)
CLASS_COLORS = [
    '#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed',
    '#0891b2', '#db2777', '#84cc16', '#f59e0b', '#6366f1',
    '#14b8a6', '#f43f5e', '#22c55e', '#3b82f6', '#a855f7',
]

# Language colors
LANGUAGE_COLORS = {
    'EN': '#2563eb',
    'FR': '#dc2626',
    'DE': '#059669',
    'ES': '#d97706',
    'IT': '#7c3aed',
    'PT': '#0891b2',
    'NL': '#db2777',
    'MULTI': '#6b7280',
}


class TrainingMetricsChart:
    """
    Generates and updates training metrics charts during model training.

    Features:
    - Loss curves (training and validation)
    - F1 score progression (macro, micro, per-class)
    - Accuracy tracking
    - Per-language performance (for multilingual models)
    - Learning rate schedule visualization
    - Best model markers
    - Publication-quality styling
    """

    def __init__(
        self,
        output_dir: str,
        session_id: str,
        model_name: str,
        num_labels: int,
        label_names: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        training_approach: str = 'single-label',
        category_name: Optional[str] = None,
        enhanced_style: bool = False,
        # Dataset size tracking
        num_samples_train: int = 0,
        num_samples_val: int = 0,
        per_class_support: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the training metrics chart generator.

        Args:
            output_dir: Directory to save charts
            session_id: Training session identifier
            model_name: Name of the model being trained
            num_labels: Number of classes/labels
            label_names: Optional list of label names for display
            languages: Optional list of languages for multilingual tracking
            training_approach: Type of training (single-label, multi-label, multi-class, one-vs-all)
            category_name: Category/key name for per-key training
            enhanced_style: Use enhanced styling for parallel training charts (thicker lines,
                           smoothing, better legends). Default False to preserve existing behavior.
            num_samples_train: Number of training samples (for dataset size display)
            num_samples_val: Number of validation samples (for dataset size display)
            per_class_support: Dict mapping class names to number of positive instances
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id
        self.model_name = self._clean_model_name(model_name)
        self.num_labels = num_labels
        self.label_names = label_names or [f"Class_{i}" for i in range(num_labels)]
        self.languages = languages
        self.training_approach = training_approach
        self.category_name = category_name
        self.enhanced_style = enhanced_style

        # Dataset size tracking for scientific charts
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.per_class_support: Dict[str, int] = per_class_support.copy() if per_class_support else {}

        # Metrics history
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.accuracies: List[float] = []
        self.f1_macros: List[float] = []
        self.f1_micros: List[float] = []
        self.combined_scores: List[float] = []  # Combined score evolution
        self.learning_rates: List[float] = []

        # Per-class F1 history
        self.per_class_f1: Dict[str, List[float]] = {name: [] for name in self.label_names}

        # Per-language metrics history (initialized from languages, but can grow dynamically)
        self.language_metrics: Dict[str, Dict[str, List[float]]] = {}
        if languages:
            for lang in languages:
                self._ensure_language_initialized(lang)

        # Best model tracking
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_f1_micro: Optional[float] = None
        self.best_combined_score: Optional[float] = None

        # Determine best metric name based on training approach
        # Binary: 0.7 × F1(class 1) + 0.3 × macro_f1
        # Multi-class/Multi-label: 0.6 × macro_f1 + 0.4 × micro_f1
        if num_labels == 2:
            self.best_metric_name: str = 'combined (0.7*F1_class1 + 0.3*macro)'
        else:
            self.best_metric_name: str = 'combined (0.6*macro + 0.4*micro)'

        # Chart settings - enhanced mode uses higher quality settings
        if self.enhanced_style:
            self.dpi = 200
            self.figsize = (18, 14)
            self.line_width_primary = 3.0
            self.line_width_secondary = 2.0
            self.marker_size_primary = 8
            self.marker_size_secondary = 5
            self.smoothing_window = 3  # Moving average window for noise reduction
        else:
            self.dpi = 150
            self.figsize = (16, 12)
            self.line_width_primary = 2.0
            self.line_width_secondary = 1.5
            self.marker_size_primary = 4
            self.marker_size_secondary = 3
            self.smoothing_window = 0  # No smoothing

        # Fixed chart filename (single file that gets updated)
        self._chart_filename: Optional[str] = None

        logger.info(f"TrainingMetricsChart initialized: {self.model_name}, {self.training_approach}")

    def _ensure_language_initialized(self, lang: str) -> None:
        """
        Ensure a language has an entry in language_metrics.
        Creates the entry with empty lists if it doesn't exist.
        This allows dynamic discovery of languages during training.

        Args:
            lang: Language code (e.g., 'EN', 'FR', 'ES')
        """
        if lang not in self.language_metrics:
            # Pad with zeros for missed epochs to maintain alignment with other languages
            current_epoch_count = len(self.epochs)
            self.language_metrics[lang] = {
                'f1_macro': [0.0] * current_epoch_count,
                'accuracy': [0.0] * current_epoch_count,
            }
            if current_epoch_count > 0:
                logger.debug(f"Dynamically added language '{lang}' to metrics tracking (padded {current_epoch_count} epochs)")

    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name for use in filenames."""
        return model_name.replace('/', '_').replace('\\', '_')

    def _smooth_data(self, data: List[float]) -> List[float]:
        """Apply moving average smoothing to reduce noise in charts."""
        if not self.enhanced_style or self.smoothing_window <= 1 or len(data) <= self.smoothing_window:
            return data

        # Pad the beginning to keep same length
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - self.smoothing_window + 1)
            window = data[start:i + 1]
            smoothed.append(sum(window) / len(window))
        return smoothed

    def _get_chart_filename(self) -> str:
        """
        Generate the chart basename for this instance.

        Uses a single fixed filename per chart instance so the same file
        is updated epoch-by-epoch rather than creating multiple files.

        Notes
        -----
        The basename used to repeat the category, the model and the session id
        -- all three of which the output *directory* already encodes
        (``training_metrics/<mode>/<category>/<LANG>/<model>/charts/``). That
        made a real path in this repository 282 characters relative to the
        project root, which crosses Windows' 260-character limit as soon as any
        drive prefix is added, and ``mkdir``/``savefig`` then fail with
        ``FileNotFoundError``. The timestamp stays: it is what keeps two chart
        instances writing into the same directory distinct.
        """
        if self._chart_filename is None:
            self._chart_filename = 'chart_' + datetime.now().strftime("%Y%m%d_%H%M%S")

        return self._chart_filename

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        f1_macro: float,
        f1_micro: Optional[float] = None,
        f1_class_1: Optional[float] = None,
        per_class_f1: Optional[Dict[str, float]] = None,
        language_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        learning_rate: Optional[float] = None,
        combined_score: Optional[float] = None,
        is_best: bool = False,
        # Dataset size updates (can be set once at first epoch)
        num_samples_train: Optional[int] = None,
        num_samples_val: Optional[int] = None,
        per_class_support: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Update metrics and regenerate the chart.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            accuracy: Validation accuracy
            f1_macro: Macro F1 score
            f1_micro: Optional micro F1 score
            f1_class_1: Optional F1 for class 1 (binary classification)
            per_class_f1: Optional dict of per-class F1 scores
            language_metrics: Optional dict of per-language metrics
            learning_rate: Optional current learning rate
            combined_score: Optional pre-calculated combined score (from trainer)
            is_best: Whether this is the best model so far
            num_samples_train: Number of training samples (optional update)
            num_samples_val: Number of validation samples (optional update)
            per_class_support: Dict of class names to positive instance counts (optional update)

        Returns:
            Path to the saved chart
        """
        # Update dataset sizes if provided
        if num_samples_train is not None:
            self.num_samples_train = num_samples_train
        if num_samples_val is not None:
            self.num_samples_val = num_samples_val
        if per_class_support is not None:
            self.per_class_support.update(per_class_support)
        # Update history
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracies.append(accuracy)
        self.f1_macros.append(f1_macro)

        if f1_micro is not None:
            self.f1_micros.append(f1_micro)

        if learning_rate is not None:
            self.learning_rates.append(learning_rate)

        # Calculate and track combined score
        # Use pre-calculated score if provided, otherwise compute locally
        if combined_score is not None:
            current_combined = combined_score
        else:
            # Calculate combined score based on classification type
            if self.num_labels == 2:
                # Binary: 0.7 × F1(class 1) + 0.3 × macro_f1
                if f1_class_1 is not None:
                    current_combined = 0.7 * f1_class_1 + 0.3 * f1_macro
                elif per_class_f1 and len(per_class_f1) >= 2:
                    # Try to extract F1 class 1 from per_class_f1
                    class_1_f1 = list(per_class_f1.values())[1] if len(per_class_f1) > 1 else f1_macro
                    current_combined = 0.7 * class_1_f1 + 0.3 * f1_macro
                else:
                    current_combined = f1_macro
            else:
                # Multi-class/Multi-label: 0.6 × macro + 0.4 × micro
                if f1_micro is not None:
                    current_combined = 0.6 * f1_macro + 0.4 * f1_micro
                else:
                    current_combined = f1_macro

        self.combined_scores.append(current_combined)

        # Update per-class F1
        if per_class_f1:
            for name, f1 in per_class_f1.items():
                if name in self.per_class_f1:
                    self.per_class_f1[name].append(f1)

        # Update per-language metrics (dynamically add new languages discovered during training)
        if language_metrics:
            for lang, metrics in language_metrics.items():
                # Dynamically add language if not already initialized
                self._ensure_language_initialized(lang)

                # Store metrics (only f1_macro and accuracy used for chart)
                for metric_name in ['f1_macro', 'accuracy']:
                    if metric_name in metrics and metric_name in self.language_metrics[lang]:
                        self.language_metrics[lang][metric_name].append(metrics[metric_name])

        # Track best model
        if is_best:
            self.best_epoch = epoch
            self.best_metric = f1_macro
            if f1_micro is not None:
                self.best_f1_micro = f1_micro
            self.best_combined_score = current_combined

        # Generate and save chart
        return self._generate_chart()

    def _generate_chart(self) -> str:
        """Generate the training metrics chart."""
        # Determine layout based on available data
        has_per_class = any(len(v) > 0 for v in self.per_class_f1.values())
        has_languages = any(len(v['f1_macro']) > 0 for v in self.language_metrics.values()) if self.language_metrics else False
        has_lr = len(self.learning_rates) > 0

        # Check if we have enough data for scientific analysis charts
        # Need >= 2 categories with both support data and F1 scores (reduced from 3 to support binary classification)
        categories_with_support_data = [
            c for c in self.per_class_support.keys()
            if c in self.per_class_f1 and self.per_class_f1[c]
        ]
        has_support_data = bool(self.per_class_support) and len(categories_with_support_data) >= 2

        # Count active classes for dynamic sizing
        num_active_classes = sum(1 for v in self.per_class_f1.values() if len(v) > 0)

        # Calculate number of rows needed and height ratios
        num_rows = 2  # Always have loss and main metrics
        height_ratios = [1.0, 1.0]  # First two rows have equal height

        if has_per_class and self.num_labels > 2:
            num_rows += 1
            height_ratios.append(1.0)
        if has_languages:
            num_rows += 1
            height_ratios.append(1.0)
        if has_support_data:
            num_rows += 1
            height_ratios.append(1.0)  # Scatter plots row
            num_rows += 1
            height_ratios.append(0.6)  # Reference table row below scatter plots

        # Dynamic figure sizing
        base_width = 16
        base_height = 4 * num_rows

        # Add width for legend if many classes
        if has_per_class and num_active_classes > 5:
            base_width = 18 if num_active_classes <= 15 else 20

        # Create figure
        fig = plt.figure(figsize=(base_width, base_height))
        fig.patch.set_facecolor('white')

        # Calculate right margin for legend (only if many classes)
        right_margin = 0.85 if (has_per_class and num_active_classes > 5) else 1.0

        # Create grid with dynamic margins and height ratios
        bottom_margin = 0.08
        gs = GridSpec(num_rows, 2, figure=fig, hspace=0.45, wspace=0.25,
                      left=0.08, right=right_margin, top=0.93, bottom=bottom_margin,
                      height_ratios=height_ratios)

        row_idx = 0

        # ============================================================
        # ROW 1: Loss curves and Main Metrics
        # ============================================================

        # Loss curves (left)
        ax_loss = fig.add_subplot(gs[row_idx, 0])
        self._plot_loss_curves(ax_loss)

        # Main metrics (right)
        ax_metrics = fig.add_subplot(gs[row_idx, 1])
        self._plot_main_metrics(ax_metrics)

        row_idx += 1

        # ============================================================
        # ROW 2: Accuracy/LR and Summary Table
        # ============================================================

        # Accuracy and LR (left)
        ax_acc = fig.add_subplot(gs[row_idx, 0])
        self._plot_accuracy_lr(ax_acc, has_lr)

        # Summary table (right)
        ax_summary = fig.add_subplot(gs[row_idx, 1])
        self._plot_summary_table(ax_summary)

        row_idx += 1

        # ============================================================
        # ROW 3: Per-class F1 (if applicable)
        # ============================================================
        if has_per_class and self.num_labels > 2:
            ax_per_class = fig.add_subplot(gs[row_idx, :])
            self._plot_per_class_f1(ax_per_class)
            row_idx += 1

        # ============================================================
        # ROW 4: Per-language metrics (if applicable)
        # ============================================================
        if has_languages:
            ax_lang = fig.add_subplot(gs[row_idx, :])
            self._plot_language_metrics(ax_lang)
            row_idx += 1

        # ============================================================
        # ROW 5: Scientific Analysis Charts (if sufficient data)
        # ============================================================
        if has_support_data:
            # Quality vs Training Size (left)
            ax_train_quality = fig.add_subplot(gs[row_idx, 0])
            ref_data = self._plot_quality_vs_train_size(ax_train_quality)

            # Quality vs Validation Size (right)
            ax_val_quality = fig.add_subplot(gs[row_idx, 1])
            self._plot_quality_vs_val_size(ax_val_quality)
            row_idx += 1

            # Reference table row (full-width) below scatter plots
            ax_ref = fig.add_subplot(gs[row_idx, :])
            if ref_data is not None:
                self._plot_class_reference_table(ax_ref, *ref_data)
            else:
                ax_ref.axis('off')
            row_idx += 1

        # Add title with proper spacing (y=0.995 keeps title close to content)
        title = self._build_title()
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        # Adjust layout to prevent overlap (bottom margin at 0.02)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        # Save chart (overwrite existing file)
        filename = self._get_chart_filename()
        png_path = self.output_dir / f"{filename}.png"

        plt.savefig(png_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.3)
        plt.close(fig)

        logger.debug(f"Training metrics chart saved: {png_path}")

        return str(png_path)

    def _build_title(self) -> str:
        """Build chart title (without epoch since chart is updated each epoch)."""
        parts = []

        # Training approach
        approach_display = {
            'single-label': 'Single-Label',
            'multi-label': 'Multi-Label',
            'multi-class': 'Multi-Class',
            'one-vs-all': 'One-vs-All Binary',
            'hybrid': 'Hybrid (Multi-Class + One-vs-All)',
            'custom': 'Custom',
            'binary': 'Binary',
        }.get(self.training_approach, self.training_approach)

        if self.category_name:
            parts.append(f"{approach_display} Training: {self.category_name}")
        else:
            parts.append(f"{approach_display} Training")

        parts.append(f"Model: {self.model_name}")

        # Epoch removed - chart is overwritten each epoch so epoch info is redundant

        return " | ".join(parts)

    def _plot_loss_curves(self, ax: plt.Axes):
        """Plot training and validation loss curves."""
        epochs = self.epochs

        # Apply smoothing for enhanced mode
        train_losses = self._smooth_data(self.train_losses)
        val_losses = self._smooth_data(self.val_losses)

        # Use enhanced styling parameters
        lw = self.line_width_primary
        ms = self.marker_size_primary

        ax.plot(epochs, train_losses, 'o-', color=COLORS['train_loss'],
               linewidth=lw, markersize=ms, label='Training Loss', alpha=0.9)
        ax.plot(epochs, val_losses, 's-', color=COLORS['val_loss'],
               linewidth=lw, markersize=ms, label='Validation Loss', alpha=0.9)

        # Mark best epoch
        if self.best_epoch is not None and self.best_epoch in epochs:
            idx = epochs.index(self.best_epoch)
            ax.axvline(x=self.best_epoch, color=COLORS['success'], linestyle='--',
                      alpha=0.7, linewidth=2, label=f'Best (Epoch {self.best_epoch})')
            ax.scatter([self.best_epoch], [val_losses[idx]], color=COLORS['success'],
                      s=150 if self.enhanced_style else 100, zorder=10, marker='*',
                      edgecolors='white', linewidths=1)

        ax.set_xlabel('Epoch', fontsize=12 if self.enhanced_style else 11)
        ax.set_ylabel('Loss', fontsize=12 if self.enhanced_style else 11)
        ax.set_title('Training & Validation Loss', fontsize=14 if self.enhanced_style else 12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10 if self.enhanced_style else 9, framealpha=0.95)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        if self.train_losses and self.val_losses:
            all_losses = self.train_losses + self.val_losses
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            padding = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
            ax.set_ylim(max(0, min_loss - padding), max_loss + padding)

    def _plot_main_metrics(self, ax: plt.Axes):
        """Plot main performance metrics (F1, accuracy, combined score)."""
        epochs = self.epochs

        # Apply smoothing for enhanced mode
        accuracies = self._smooth_data(self.accuracies)
        f1_macros = self._smooth_data(self.f1_macros)
        f1_micros = self._smooth_data(self.f1_micros) if self.f1_micros else []
        combined_scores = self._smooth_data(self.combined_scores) if self.combined_scores else []

        # Use enhanced styling parameters
        lw_primary = self.line_width_primary
        lw_secondary = self.line_width_secondary
        ms_primary = self.marker_size_primary
        ms_secondary = self.marker_size_secondary

        # Plot order: Accuracy first (bottom), then F1 Micro, F1 Macro, Combined (top)
        # This ensures Combined Score is most visible as it's the selection metric

        ax.plot(epochs, accuracies, '^-', color=COLORS['accuracy'],
               linewidth=lw_secondary, markersize=ms_secondary, label='Accuracy', alpha=0.7, zorder=2)

        if f1_micros:
            ax.plot(epochs, f1_micros, 's-', color=COLORS['f1_micro'],
                   linewidth=lw_secondary, markersize=ms_secondary, label='F1 Micro', alpha=0.7, zorder=3)

        # F1 Macro
        ax.plot(epochs, f1_macros, 'o-', color='#047857',  # Darker green
               linewidth=lw_primary, markersize=ms_primary, label='F1 Macro', alpha=0.9, zorder=4)

        # Combined Score - the main selection metric (most prominent)
        if combined_scores:
            ax.plot(epochs[:len(combined_scores)], combined_scores, 'D-',
                   color=COLORS['combined'], linewidth=lw_primary + 0.5, markersize=ms_primary + 1,
                   label='Combined Score', alpha=1.0, zorder=5)

        # Mark best epoch
        if self.best_epoch is not None and self.best_epoch in epochs:
            idx = epochs.index(self.best_epoch)
            ax.axvline(x=self.best_epoch, color=COLORS['success'], linestyle='--',
                      alpha=0.7, linewidth=2, zorder=1)
            # Mark best combined score point
            if combined_scores and idx < len(combined_scores):
                ax.scatter([self.best_epoch], [combined_scores[idx]], color=COLORS['combined'],
                          s=200 if self.enhanced_style else 150, zorder=10, marker='*',
                          edgecolors='white', linewidths=1.5)

        ax.set_xlabel('Epoch', fontsize=12 if self.enhanced_style else 11)
        ax.set_ylabel('Score', fontsize=12 if self.enhanced_style else 11)
        ax.set_title('Performance Metrics (Selection: Combined Score)', fontsize=14 if self.enhanced_style else 12, fontweight='bold')

        # Enhanced legend positioning - outside the plot area in enhanced mode
        if self.enhanced_style:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=1,
                     bbox_to_anchor=(0.02, 0.98))
        else:
            ax.legend(loc='lower right', fontsize=8, ncol=2)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, zorder=0)

        # Add horizontal lines for reference
        ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=0)
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=0)

    def _plot_accuracy_lr(self, ax: plt.Axes, has_lr: bool):
        """Plot accuracy and learning rate."""
        epochs = self.epochs

        # Apply smoothing for enhanced mode
        accuracies = self._smooth_data(self.accuracies)

        # Use enhanced styling parameters
        lw = self.line_width_primary
        ms = self.marker_size_primary
        fontsize = 12 if self.enhanced_style else 11

        # Plot accuracy
        color_acc = COLORS['accuracy']
        ax.plot(epochs, accuracies, 'o-', color=color_acc,
               linewidth=lw, markersize=ms, label='Accuracy')
        ax.set_xlabel('Epoch', fontsize=fontsize)
        ax.set_ylabel('Accuracy', color=color_acc, fontsize=fontsize)
        ax.tick_params(axis='y', labelcolor=color_acc)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot learning rate on secondary axis if available
        if has_lr and self.learning_rates:
            ax2 = ax.twinx()
            color_lr = COLORS['lr']
            ax2.plot(epochs[:len(self.learning_rates)], self.learning_rates, '--',
                    color=color_lr, linewidth=self.line_width_secondary, label='Learning Rate', alpha=0.7)
            ax2.set_ylabel('Learning Rate', color=color_lr, fontsize=fontsize)
            ax2.tick_params(axis='y', labelcolor=color_lr)
            ax2.set_yscale('log')

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right',
                     fontsize=10 if self.enhanced_style else 9, framealpha=0.95)
        else:
            ax.legend(loc='lower right', fontsize=10 if self.enhanced_style else 9, framealpha=0.95)

        ax.set_title('Accuracy & Learning Rate', fontsize=14 if self.enhanced_style else 12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_summary_table(self, ax: plt.Axes):
        """Plot summary statistics table with F1 Micro and Combined Score."""
        ax.axis('off')

        # Gather current stats
        current_epoch = self.epochs[-1] if self.epochs else 0
        current_train_loss = self.train_losses[-1] if self.train_losses else 0
        current_val_loss = self.val_losses[-1] if self.val_losses else 0
        current_f1 = self.f1_macros[-1] if self.f1_macros else 0
        current_acc = self.accuracies[-1] if self.accuracies else 0

        best_f1 = max(self.f1_macros) if self.f1_macros else 0
        best_acc = max(self.accuracies) if self.accuracies else 0
        min_val_loss = min(self.val_losses) if self.val_losses else 0

        # F1 Micro values
        current_f1_micro = self.f1_micros[-1] if self.f1_micros else None
        best_f1_micro = self.best_f1_micro if self.best_f1_micro else (max(self.f1_micros) if self.f1_micros else None)

        # Combined score from history
        current_combined = self.combined_scores[-1] if self.combined_scores else 0
        max_combined = max(self.combined_scores) if self.combined_scores else 0

        # Build table data with title row
        table_data = [
            ['Current Epoch', f'{current_epoch}'],
            ['Training Loss', f'{current_train_loss:.4f}'],
            ['Validation Loss', f'{current_val_loss:.4f}'],
            ['', ''],
            ['F1 Macro (current)', f'{current_f1:.4f}'],
            ['F1 Macro (best)', f'{best_f1:.4f}'],
        ]

        # Add F1 Micro only if available
        if current_f1_micro is not None:
            table_data.append(['F1 Micro (current)', f'{current_f1_micro:.4f}'])
            table_data.append(['F1 Micro (best)', f'{best_f1_micro:.4f}' if best_f1_micro else 'N/A'])

        # Add Combined Score (the metric used for best model selection)
        table_data.append(['', ''])
        table_data.append(['Combined Score (current)', f'{current_combined:.4f}'])
        # Use best_combined_score if available (from is_best marking), else use max from history
        best_combined_display = self.best_combined_score if self.best_combined_score is not None else max_combined
        table_data.append(['Combined Score (best)', f'{best_combined_display:.4f}'])

        table_data.extend([
            ['Accuracy (current)', f'{current_acc:.4f}'],
            ['Accuracy (best)', f'{best_acc:.4f}'],
            ['', ''],
            ['Best Epoch', f'{self.best_epoch}' if self.best_epoch else 'N/A'],
            ['Selection Metric', self.best_metric_name],
            ['Min Val Loss', f'{min_val_loss:.4f}'],
        ])

        # Add Dataset Information section if available
        if self.num_samples_train > 0 or self.num_samples_val > 0:
            table_data.append(['', ''])  # Separator
            table_data.append(['Training Set', f'{self.num_samples_train:,}'])
            table_data.append(['Validation Set', f'{self.num_samples_val:,}'])
            total_samples = self.num_samples_train + self.num_samples_val
            table_data.append(['Total Samples', f'{total_samples:,}'])

        # Add title as text above table
        ax.text(0.5, 0.98, 'Training Summary', transform=ax.transAxes,
               fontsize=12, fontweight='bold', ha='center', va='top')

        # Create table with adjusted position (lower to accommodate title)
        # Enlarged dimensions for better readability of long formulas
        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='left',
            colWidths=[0.60, 0.40],  # More space for formulas
            bbox=[0.02, 0.02, 0.96, 0.90]  # Wider and taller table
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)  # Slightly smaller font to fit more content
        table.scale(1.0, 1.35)  # Less vertical padding for more rows

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Style rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if table_data[i-1][0] == '':
                    table[(i, j)].set_facecolor('#f8fafc')
                elif i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8fafc')
                else:
                    table[(i, j)].set_facecolor('white')
                table[(i, j)].set_edgecolor('#e2e8f0')

    def _plot_per_class_f1(self, ax: plt.Axes):
        """Plot per-class F1 scores over epochs with smart legend names."""
        epochs = self.epochs

        # Collect active classes
        active_classes = [(name, f1_hist) for name, f1_hist in self.per_class_f1.items()
                          if len(f1_hist) > 0]
        num_active = len(active_classes)

        if num_active == 0:
            ax.text(0.5, 0.5, 'No per-class data available',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
            return

        # Calculate smart display names for legend
        display_names = self._get_smart_legend_names(
            [name for name, _ in active_classes],
            category_prefix=self.category_name
        )

        # Plot each class
        for i, ((label_name, f1_history), display_name) in enumerate(zip(active_classes, display_names)):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.plot(epochs[:len(f1_history)], f1_history, 'o-',
                   color=color, linewidth=1.5, markersize=3,
                   label=display_name, alpha=0.8)

        # Add macro F1 line
        ax.plot(epochs, self.f1_macros, 'k-', linewidth=2.5,
               label='Macro F1', alpha=0.9)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Use adaptive legend positioning
        self._add_adaptive_legend(ax, num_active)

    def _get_smart_legend_names(self, names: List[str], category_prefix: Optional[str] = None) -> List[str]:
        """
        Generate smart legend names by finding the differentiating parts.

        Args:
            names: List of full class/category names.
            category_prefix: Optional category prefix to strip before computing
                common prefix (e.g. 'auth' strips 'auth_' from each name).

        Examples:
        - ['auth_authority_business_positive', 'auth_authority_state_negative', 'auth_none']
          with category_prefix='auth'
          -> ['1. authority business positive', '2. authority state negative', '3. none']
        """
        if len(names) <= 1:
            return [n.replace('_', ' ')[:35] for n in names]

        # Strip category prefix from each name if provided
        working_names = []
        for name in names:
            n = name
            if category_prefix:
                prefix_str = category_prefix + '_'
                if n.startswith(prefix_str):
                    n = n[len(prefix_str):]
            working_names.append(n)

        # Find longest common prefix among the working names
        prefix = os.path.commonprefix(working_names)
        # Find longest common suffix
        reversed_names = [n[::-1] for n in working_names]
        suffix = os.path.commonprefix(reversed_names)[::-1]

        # Extract unique part for each name
        display_names = []
        for wname in working_names:
            unique_part = wname
            # Remove common prefix if significant
            if prefix and len(prefix) > 3:
                unique_part = unique_part[len(prefix):]
            # Remove common suffix if significant
            if suffix and len(suffix) > 3:
                unique_part = unique_part[:-len(suffix)]

            # Clean leading/trailing underscores
            unique_part = unique_part.strip('_')

            # Replace underscores with spaces for readability
            unique_part = unique_part.replace('_', ' ')

            # Truncate if still too long
            if len(unique_part) > 35:
                unique_part = unique_part[:32] + '...'

            # If empty after extraction, use the working name (with prefix stripped)
            if not unique_part:
                fallback = wname.replace('_', ' ')
                unique_part = fallback[:32] + '...' if len(fallback) > 35 else fallback

            display_names.append(unique_part)

        # Add numbering when duplicates exist or when many classes (>5) for clarity
        if len(set(display_names)) < len(display_names) or len(display_names) > 5:
            display_names = [f"{i+1}. {n}" for i, n in enumerate(display_names)]

        return display_names

    def _add_adaptive_legend(self, ax: plt.Axes, num_classes: int):
        """Add legend with adaptive positioning based on number of classes."""
        if num_classes <= 5:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        elif num_classes <= 10:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=8, ncol=1, framealpha=0.95)
        elif num_classes <= 20:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=7, ncol=1, framealpha=0.95,
                     handlelength=1.5, handletextpad=0.5)
        else:
            # For many classes, use 2 columns
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=6, ncol=2, framealpha=0.95,
                     columnspacing=0.8, handlelength=1, handletextpad=0.3)

    def _plot_class_reference_table(self, ax: plt.Axes, categories: List[str],
                                      supports: list, f1_scores: list):
        """
        Draw a full-width reference table mapping numbered scatter-plot points
        to their full original class names, training counts, and F1 scores.

        Args:
            ax: Matplotlib axes (full-width row below scatter plots).
            categories: Full original category names.
            supports: Training-set counts per category.
            f1_scores: Final F1 scores per category.
        """
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        n = len(categories)
        if n == 0:
            return

        # Title
        ax.text(0.5, 0.97, 'Class Reference \u2014 Per-Class Detail',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold', color=COLORS['dark'])

        # Layout: up to 3 columns
        ncols = 3 if n > 6 else (2 if n > 3 else 1)
        rows_per_col = int(np.ceil(n / ncols))

        col_width = 1.0 / ncols
        line_height = min(0.85 / max(rows_per_col, 1), 0.11)
        y_start = 0.88

        for idx in range(n):
            col = idx // rows_per_col
            row = idx % rows_per_col

            x_base = col * col_width + 0.02
            y_pos = y_start - row * line_height

            color = CLASS_COLORS[idx % len(CLASS_COLORS)]

            # Colored dot
            ax.plot(x_base, y_pos, 'o', color=color, markersize=8,
                    transform=ax.transAxes, clip_on=False)

            # Number + full name (underscores → spaces) + count + F1
            display_name = categories[idx].replace('_', ' ')
            sup = supports[idx] if idx < len(supports) else 0
            f1 = f1_scores[idx] if idx < len(f1_scores) else 0.0
            label = f"{idx+1}. {display_name}  (n={sup:,}  F1={f1:.3f})"

            fontsize = 7.5 if ncols >= 3 else 8
            ax.text(x_base + 0.02, y_pos, label,
                    transform=ax.transAxes, ha='left', va='center',
                    fontsize=fontsize, fontfamily='sans-serif', color=COLORS['dark'])

    def _plot_language_metrics(self, ax: plt.Axes):
        """Plot per-language F1 scores with adaptive legend."""
        epochs = self.epochs
        num_langs = 0

        for lang, metrics in self.language_metrics.items():
            if len(metrics['f1_macro']) > 0:
                num_langs += 1
                color = LANGUAGE_COLORS.get(lang.upper(), COLORS['muted'])
                ax.plot(epochs[:len(metrics['f1_macro'])], metrics['f1_macro'],
                       'o-', color=color, linewidth=2, markersize=4,
                       label=f'{lang.upper()}', alpha=0.9)

        # Add overall macro F1
        ax.plot(epochs, self.f1_macros, 'k--', linewidth=2,
               label='Overall', alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('F1 Macro', fontsize=11)
        ax.set_title('Per-Language Performance', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Adaptive legend positioning
        if num_langs <= 6:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=8, ncol=1, framealpha=0.95)

    def _get_short_category_name(self, name: str, max_len: int = 18) -> str:
        """Shorten category name for legend display."""
        if len(name) <= max_len:
            return name
        # Try to find a meaningful truncation point
        if '_' in name:
            parts = name.split('_')
            # Keep last part if it's meaningful
            if len(parts[-1]) >= 3:
                short = parts[-1]
                if len(short) <= max_len:
                    return short
        # Simple truncation with ellipsis
        return name[:max_len-2] + '..'

    def _plot_binary_class_impact(self, ax: plt.Axes, categories: list, train_supports: list, f1_scores: list, colors: list):
        """
        Scientific visualization for binary classification:
        F1 vs Training Size with Wilson Confidence Intervals and trend line.
        """
        from matplotlib.patches import Polygon

        # Wilson CI calculation
        def wilson_ci(p, n, z=1.96):
            denom = 1 + z**2/n
            centre = (p + z**2/(2*n)) / denom
            margin = (z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / denom
            return max(0, centre - margin), min(1, centre + margin)

        # Calculate CIs
        cis = [wilson_ci(f1, n) for f1, n in zip(f1_scores, train_supports)]
        ci_lower = [ci[0] for ci in cis]
        ci_upper = [ci[1] for ci in cis]
        yerr_lower = [f1 - ci[0] for f1, ci in zip(f1_scores, cis)]
        yerr_upper = [ci[1] - f1 for f1, ci in zip(f1_scores, cis)]

        # Scatter with error bars
        for i, (sup, f1, color, cat) in enumerate(zip(train_supports, f1_scores, colors, categories)):
            ax.errorbar(sup, f1, yerr=[[yerr_lower[i]], [yerr_upper[i]]],
                        fmt='o', markersize=12, color=color, ecolor=color,
                        capsize=8, capthick=2, elinewidth=2, label=cat)

        # Trend line connecting the two points
        ax.plot(train_supports, f1_scores, '--', color='gray', linewidth=1.5, alpha=0.6, zorder=1)

        # CI shaded region (polygon between the two points)
        verts = [(train_supports[0], ci_lower[0]), (train_supports[0], ci_upper[0]),
                 (train_supports[1], ci_upper[1]), (train_supports[1], ci_lower[1])]
        poly = Polygon(verts, alpha=0.1, color='blue', zorder=0)
        ax.add_patch(poly)

        # Calculate statistics
        h = 2 * np.arcsin(np.sqrt(max(0, min(1, f1_scores[0])))) - 2 * np.arcsin(np.sqrt(max(0, min(1, f1_scores[1]))))
        r = np.corrcoef(train_supports, f1_scores)[0, 1] if len(train_supports) > 1 else 0

        # Annotations on points
        for i, (sup, f1) in enumerate(zip(train_supports, f1_scores)):
            offset = (10, 10) if i == 1 else (10, -15)
            ax.annotate(f'F1={f1:.3f}\nn={sup:,}', (sup, f1),
                        textcoords='offset points', xytext=offset, fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        # Styling
        ax.set_xlabel('Training Set Size (positive instances)', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(f'F1 vs Training Size\nr={r:.2f} | Cohen\'s h={h:.3f} | 95% Wilson CI', fontsize=11, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0.5, 1.05)
        ax.set_xlim(0, max(train_supports) * 1.15)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    def _plot_binary_confidence_assessment(self, ax: plt.Axes, categories: list, val_sizes: list, f1_scores: list, colors: list):
        """
        Scientific visualization for binary confidence assessment:
        F1 vs Validation Size with confidence zones and 80% power detection band.
        """
        # Background confidence zones (subtle)
        max_val = max(val_sizes) * 1.2 if max(val_sizes) > 0 else 60
        ax.axvspan(0, 50, alpha=0.08, color='red', zorder=0)
        ax.axvspan(50, 200, alpha=0.08, color='orange', zorder=0)
        ax.axvspan(200, max(max_val, 250), alpha=0.08, color='green', zorder=0)

        # Calculate SE for F1 (approximation)
        se_f1 = [np.sqrt(f1 * (1 - f1) / max(n, 1)) for f1, n in zip(f1_scores, val_sizes)]

        # Scatter with error bars
        for i, (val, f1, se, color, cat) in enumerate(zip(val_sizes, f1_scores, se_f1, colors, categories)):
            ax.errorbar(val, f1, yerr=1.96 * se,
                        fmt='s', markersize=12, color=color, ecolor=color,
                        capsize=8, capthick=2, elinewidth=2, label=cat)

        # Power detection band (80% power curve)
        n_range = np.linspace(10, max(max_val, 100), 100)
        avg_f1 = np.mean(f1_scores)
        detectable_effect = 2.8 * np.sqrt(avg_f1 * (1 - avg_f1) / n_range)
        ax.fill_between(n_range, avg_f1 - detectable_effect, avg_f1 + detectable_effect,
                        alpha=0.1, color='purple', label='80% power band')

        # Reference line for good F1
        ax.axhline(y=0.8, color='darkgreen', linestyle=':', alpha=0.6, linewidth=1.5)

        # Annotations on points
        for i, (val, f1) in enumerate(zip(val_sizes, f1_scores)):
            offset = (10, 10) if i == 1 else (10, -15)
            ax.annotate(f'F1={f1:.3f}\nn={val:,}', (val, f1),
                        textcoords='offset points', xytext=offset, fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        # Zone labels at bottom
        ax.text(25, 0.52, 'Low', fontsize=8, ha='center', color='darkred', alpha=0.8)
        ax.text(125, 0.52, 'Moderate', fontsize=8, ha='center', color='darkorange', alpha=0.8)
        if max_val > 250:
            ax.text(max_val * 0.6, 0.52, 'High confidence', fontsize=8, ha='center', color='darkgreen', alpha=0.8)

        # Styling
        ax.set_xlabel('Validation Set Size', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        total_val = sum(val_sizes)
        ax.set_title(f'Confidence Assessment\nTotal: {total_val:,} | 95% CI', fontsize=11, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0.5, 1.05)
        ax.set_xlim(0, max(max_val, 100))
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    def _plot_quality_vs_train_size(self, ax: plt.Axes) -> Optional[Tuple[List[str], list, list]]:
        """
        Plot scientific chart: F1 vs positive instances per category.

        Uses numbered colored markers instead of a below-chart legend.
        The caller renders a separate reference table row with full class names.

        Returns:
            (categories, supports, f1_scores) tuple for the reference table,
            or None if insufficient data.
        """
        from scipy import stats

        # Verify sufficient data (>= 2 categories with both support and F1)
        categories_with_data = [
            c for c in self.per_class_support.keys()
            if c in self.per_class_f1 and self.per_class_f1[c]
        ]

        if len(categories_with_data) < 2:
            ax.text(0.5, 0.5, 'Analysis not available\n(< 2 categories with data)',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, style='italic', color='gray')
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            return None

        # For binary classification, use specialized visualization
        if len(categories_with_data) == 2:
            categories = categories_with_data
            supports = [self.per_class_support[c] for c in categories]
            f1_scores = [
                self.per_class_f1[c][-1] if self.per_class_f1[c] else 0.0
                for c in categories
            ]
            self._plot_binary_class_impact(ax, categories, supports, f1_scores, CLASS_COLORS)
            return (categories, supports, f1_scores)

        # Extract data
        categories = categories_with_data
        supports = np.array([self.per_class_support[c] for c in categories])
        f1_scores = np.array([
            self.per_class_f1[c][-1] if self.per_class_f1[c] else 0.0
            for c in categories
        ])

        # Filter out zero supports for regression (avoid log issues)
        valid_mask = supports > 0
        if np.sum(valid_mask) < 3:
            ax.text(0.5, 0.5, 'Insufficient non-zero support\nfor regression analysis',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, style='italic', color='gray')
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            return None

        supports_valid = supports[valid_mask]
        f1_valid = f1_scores[valid_mask]

        # Scatter plot with numbered colored markers
        for i, (cat, sup, f1) in enumerate(zip(categories, supports, f1_scores)):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.scatter([sup], [f1], c=[color], s=120, alpha=0.85,
                      edgecolors='white', linewidths=1.5, zorder=3)
            # Add number annotation next to each point
            ax.annotate(str(i + 1), (sup, f1),
                        textcoords='offset points', xytext=(8, 6),
                        fontsize=8, fontweight='bold', color=color, zorder=4)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(supports_valid, f1_valid)
        x_line = np.linspace(supports_valid.min(), supports_valid.max(), 100)
        y_line = slope * x_line + intercept

        # 95% confidence interval
        n = len(supports_valid)
        if n > 2:
            t_crit = stats.t.ppf(0.975, n - 2)
            residuals = f1_valid - (slope * supports_valid + intercept)
            s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
            x_mean = supports_valid.mean()
            ss_x = np.sum((supports_valid - x_mean)**2)
            ci = t_crit * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)

            ax.plot(x_line, y_line, 'r-', linewidth=2, zorder=2)
            ax.fill_between(x_line, y_line - ci, y_line + ci,
                          color='red', alpha=0.15, zorder=1)

        # Spearman correlation (robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(supports_valid, f1_valid)

        # Statistics box (top-left)
        stats_text = (f"Pearson r = {r_value:.3f} (p = {p_value:.3f})\n"
                      f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3f})\n"
                      f"R² = {r_value**2:.3f}")
        props = dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9, edgecolor='#d4a574')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', fontfamily='monospace', bbox=props)

        # Labels and style
        ax.set_xlabel('Positive Instances (Training)', fontsize=11, fontweight='medium')
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='medium')
        ax.set_title('Quality vs Training Set Size', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)

        return (categories, list(supports), list(f1_scores))

    def _plot_quality_vs_val_size(self, ax: plt.Axes):
        """
        Plot scientific chart: F1 vs validation set size.

        Uses numbered colored markers instead of a below-chart legend.
        """
        from scipy import stats

        # Calculate validation sizes per class (proportional to support)
        if not self.per_class_support or len(self.per_class_support) < 2:
            ax.text(0.5, 0.5, 'Validation data\ninsufficient',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, style='italic', color='gray')
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            return

        # Estimate val_size per class (using split ratio)
        total_samples = self.num_samples_train + self.num_samples_val
        val_ratio = self.num_samples_val / max(total_samples, 1)
        val_ratio = val_ratio if val_ratio > 0 else 0.1  # Default 10% if not set

        categories_with_data = [
            c for c in self.per_class_support.keys()
            if c in self.per_class_f1 and self.per_class_f1[c]
        ]

        if len(categories_with_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient categories\nfor analysis',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, style='italic', color='gray')
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            return

        categories = categories_with_data
        train_supports = np.array([self.per_class_support[c] for c in categories])
        # Estimate validation sizes from training supports
        val_sizes = np.maximum((train_supports * val_ratio / max(1 - val_ratio, 0.01)).astype(int), 1)
        f1_scores = np.array([
            self.per_class_f1[c][-1] if self.per_class_f1[c] else 0.0
            for c in categories
        ])

        # For binary classification, use specialized visualization
        if len(categories_with_data) == 2:
            self._plot_binary_confidence_assessment(ax, categories, list(val_sizes), list(f1_scores), CLASS_COLORS)
            return

        # Scatter plot with numbered colored markers
        for i, (cat, val_size, f1) in enumerate(zip(categories, val_sizes, f1_scores)):
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.scatter([val_size], [f1], c=[color], s=120, alpha=0.85,
                      edgecolors='white', linewidths=1.5, zorder=3)
            # Add number annotation next to each point
            ax.annotate(str(i + 1), (val_size, f1),
                        textcoords='offset points', xytext=(8, 6),
                        fontsize=8, fontweight='bold', color=color, zorder=4)

        # Error bars (Wilson interval approximation: SE(F1) ≈ sqrt(F1*(1-F1)/n))
        se_f1 = np.sqrt(np.maximum(f1_scores * (1 - f1_scores), 0.001) / np.maximum(val_sizes, 1))
        ax.errorbar(val_sizes, f1_scores, yerr=1.96*se_f1, fmt='none',
                   ecolor='gray', alpha=0.5, capsize=3, capthick=1, zorder=2)

        # Regression (if enough valid points)
        valid_mask = val_sizes > 0
        if np.sum(valid_mask) >= 3:
            val_valid = val_sizes[valid_mask]
            f1_valid = f1_scores[valid_mask]
            slope, intercept, r_value, p_value, _ = stats.linregress(val_valid, f1_valid)
            x_line = np.linspace(val_valid.min(), val_valid.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', linewidth=1.5, alpha=0.7, zorder=1)

            stats_text = f"R² = {r_value**2:.3f}\np = {p_value:.3f}"
            props = dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9, edgecolor='#7eb3d4')
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   fontfamily='monospace', bbox=props)

        # Reference line: minimum recommended sample size (n=30 rule of thumb)
        ax.axvline(x=30, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(32, 0.05, 'n=30\n(min.)', fontsize=8, color='orange', fontweight='bold')

        # Labels and style
        ax.set_xlabel('Estimated Validation Set Size', fontsize=11, fontweight='medium')
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='medium')
        ax.set_title('Confidence: F1 vs Validation Size', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)

    def save_metrics_json(self) -> str:
        """Save all metrics to a JSON file."""
        metrics_data = {
            'session_id': self.session_id,
            'model_name': self.model_name,
            'training_approach': self.training_approach,
            'category_name': self.category_name,
            'num_labels': self.num_labels,
            'label_names': self.label_names,
            'languages': self.languages,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'accuracies': self.accuracies,
            'f1_macros': self.f1_macros,
            'f1_micros': self.f1_micros,
            'combined_scores': self.combined_scores,
            'learning_rates': self.learning_rates,
            'per_class_f1': self.per_class_f1,
            'language_metrics': self.language_metrics,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'best_combined_score': self.best_combined_score,
            'selection_metric_formula': self.best_metric_name,
            'timestamp': datetime.now().isoformat(),
            # Dataset size information
            'num_samples_train': self.num_samples_train,
            'num_samples_val': self.num_samples_val,
            'per_class_support': self.per_class_support,
        }

        filename = self._get_chart_filename() + '_metrics.json'
        json_path = self.output_dir / filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

        return str(json_path)


def create_training_chart(
    output_dir: str,
    session_id: str,
    model_name: str,
    num_labels: int,
    label_names: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    training_approach: str = 'single-label',
    category_name: Optional[str] = None,
) -> TrainingMetricsChart:
    """
    Factory function to create a TrainingMetricsChart instance.

    Args:
        output_dir: Directory to save charts
        session_id: Training session identifier
        model_name: Name of the model being trained
        num_labels: Number of classes/labels
        label_names: Optional list of label names
        languages: Optional list of languages
        training_approach: Training type
        category_name: Category name for per-key training

    Returns:
        TrainingMetricsChart instance
    """
    return TrainingMetricsChart(
        output_dir=output_dir,
        session_id=session_id,
        model_name=model_name,
        num_labels=num_labels,
        label_names=label_names,
        languages=languages,
        training_approach=training_approach,
        category_name=category_name,
    )


def generate_training_summary_chart(
    output_dir: str,
    session_id: str,
    model_name: str,
    results_per_key: Dict[str, Dict[str, Any]],
    training_approach: str = 'multi-label',
    total_training_time: Optional[float] = None,
) -> Optional[str]:
    """
    Generate a summary chart comparing all trained models.

    This creates a publication-quality overview chart showing performance
    metrics across all models trained in a multi-model session.

    Args:
        output_dir: Directory to save the summary chart
        session_id: Training session identifier
        model_name: Base model name used for training
        results_per_key: Dictionary mapping key names to their training results
            Expected format: {
                'key_name': {
                    'f1_macro': float,
                    'best_f1_macro': float,
                    'training_time': float,
                    'best_epoch': int,
                    'num_labels': int,
                    ...
                }
            }
        training_approach: Training approach used
        total_training_time: Total time for all training (seconds)

    Returns:
        Path to the generated summary chart, or None if generation failed
    """
    try:
        from pathlib import Path
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import numpy as np

        # Filter out failed results
        valid_results = {
            k: v for k, v in results_per_key.items()
            if isinstance(v, dict) and 'error' not in v
        }

        if not valid_results:
            logger.warning("No valid results to generate summary chart")
            return None

        # Extract metrics - prefer combined_score as it's the selection metric
        keys = list(valid_results.keys())
        # Combined score is the primary metric for model selection
        combined_scores = [
            valid_results[k].get('combined_score') or
            valid_results[k].get('best_combined_score') or
            valid_results[k].get('f1_macro') or
            valid_results[k].get('best_f1_macro', 0)
            for k in keys
        ]
        f1_macro_scores = [valid_results[k].get('f1_macro') or valid_results[k].get('best_f1_macro', 0) for k in keys]
        training_times = [valid_results[k].get('training_time', 0) for k in keys]
        best_epochs = [valid_results[k].get('best_epoch', 0) for k in keys]
        num_labels = [valid_results[k].get('num_labels', 0) for k in keys]

        # Create figure
        fig = plt.figure(figsize=(14, 11))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Title with proper spacing
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        title_lines = [
            f"Training Summary - {training_approach.replace('-', ' ').title()}",
            f"Model: {model_name} | {len(valid_results)} models | {timestamp}"
        ]
        fig.suptitle('\n'.join(title_lines), fontsize=14, fontweight='bold', y=0.98)

        # Color palette
        colors = [
            '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
            '#f43f5e', '#ef4444', '#f97316', '#f59e0b', '#eab308',
            '#84cc16', '#22c55e', '#10b981', '#14b8a6', '#06b6d4',
        ]
        bar_colors = [colors[i % len(colors)] for i in range(len(keys))]

        # Truncate long key names for display
        display_keys = [k[:20] + '...' if len(k) > 20 else k for k in keys]

        # ========== 1. Combined Score Comparison (top left) ==========
        ax1 = fig.add_subplot(gs[0, 0])
        y_pos = np.arange(len(keys))
        bars1 = ax1.barh(y_pos, combined_scores, color=bar_colors, alpha=0.85, edgecolor='white')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(display_keys, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Combined Score (Selection Metric)', fontsize=10)
        ax1.set_title('Model Performance Comparison', fontsize=11, fontweight='bold', pad=10)
        ax1.set_xlim(0, 1.0)

        # Add value labels
        for bar, score in zip(bars1, combined_scores):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

        # Add average line
        avg_combined = np.mean(combined_scores)
        ax1.axvline(x=avg_combined, color='#dc2626', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(avg_combined + 0.02, len(keys) - 0.5, f'Avg: {avg_combined:.4f}',
                color='#dc2626', fontsize=9, fontweight='bold')

        # ========== 2. Training Time (top right) ==========
        ax2 = fig.add_subplot(gs[0, 1])

        # Convert to minutes for readability
        times_minutes = [t / 60 for t in training_times]
        bars2 = ax2.barh(y_pos, times_minutes, color=bar_colors, alpha=0.85, edgecolor='white')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(display_keys, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Training Time (minutes)', fontsize=10)
        ax2.set_title('Training Duration per Model', fontsize=11, fontweight='bold', pad=10)

        # Add value labels
        max_time = max(times_minutes) if times_minutes else 1
        for bar, time_val in zip(bars2, times_minutes):
            width = bar.get_width()
            label = f'{time_val:.1f}m' if time_val < 60 else f'{time_val/60:.1f}h'
            ax2.text(width + max_time * 0.02, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=9)

        # ========== 3. Best Epoch Distribution (bottom left) ==========
        ax3 = fig.add_subplot(gs[1, 0])
        bars3 = ax3.bar(np.arange(len(keys)), best_epochs, color=bar_colors, alpha=0.85, edgecolor='white')
        ax3.set_xticks(np.arange(len(keys)))
        ax3.set_xticklabels(display_keys, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Best Epoch', fontsize=10)
        ax3.set_title('Best Model Epoch per Category', fontsize=11, fontweight='bold', pad=10)

        # Add value labels
        for bar, epoch in zip(bars3, best_epochs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    str(int(epoch)), ha='center', va='bottom', fontsize=9, fontweight='bold')

        # ========== 4. Summary Statistics (bottom right) ==========
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Build summary text
        total_time_str = ""
        if total_training_time:
            if total_training_time > 3600:
                total_time_str = f"{total_training_time/3600:.1f} hours"
            else:
                total_time_str = f"{total_training_time/60:.1f} minutes"
        else:
            total_time_str = f"{sum(training_times)/60:.1f} minutes"

        summary_lines = [
            f"{'─' * 40}",
            f"  TRAINING SUMMARY",
            f"{'─' * 40}",
            f"",
            f"  Models Trained:     {len(valid_results)}",
            f"  Failed Models:      {len(results_per_key) - len(valid_results)}",
            f"",
            f"  Best Combined:      {max(combined_scores):.4f}",
            f"  Worst Combined:     {min(combined_scores):.4f}",
            f"  Average Combined:   {avg_combined:.4f}",
            f"  Std Dev Combined:   {np.std(combined_scores):.4f}",
            f"",
            f"  Best F1 Macro:      {max(f1_macro_scores):.4f}",
            f"",
            f"  Total Training:     {total_time_str}",
            f"  Avg per Model:      {np.mean(times_minutes):.1f} min",
            f"",
            f"  Top Performer:",
            f"    {keys[np.argmax(combined_scores)]}",
            f"    Combined = {max(combined_scores):.4f}",
            f"{'─' * 40}",
        ]

        props = dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#e2e8f0', alpha=0.9)
        ax4.text(0.5, 0.5, '\n'.join(summary_lines),
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                bbox=props, family='monospace')

        # Adjust layout with proper margins
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        # Save chart
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_model = model_name.replace('/', '_').replace('\\', '_')[:30]
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_summary_{session_id[:20]}_{safe_model}_{timestamp_file}.png"
        chart_path = output_path / filename

        fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close(fig)

        logger.info(f"Training summary chart saved to {chart_path}")

        # Also save summary as JSON
        summary_data = {
            'session_id': session_id,
            'model_name': model_name,
            'training_approach': training_approach,
            'timestamp': datetime.now().isoformat(),
            'models_trained': len(valid_results),
            'models_failed': len(results_per_key) - len(valid_results),
            'total_training_time': total_training_time or sum(training_times),
            # Combined score metrics (primary selection metric)
            'average_combined': float(avg_combined),
            'best_combined': float(max(combined_scores)),
            'worst_combined': float(min(combined_scores)),
            'std_combined': float(np.std(combined_scores)),
            # F1 macro metrics (for reference)
            'average_f1_macro': float(np.mean(f1_macro_scores)),
            'best_f1_macro': float(max(f1_macro_scores)),
            'worst_f1_macro': float(min(f1_macro_scores)),
            'std_f1_macro': float(np.std(f1_macro_scores)),
            # Top performer based on combined score
            'top_performer': keys[np.argmax(combined_scores)],
            'top_performer_combined_score': float(max(combined_scores)),
            'per_model_results': {
                k: {
                    'combined_score': combined_scores[i],
                    'f1_macro': f1_macro_scores[i],
                    'training_time': valid_results[k].get('training_time', 0),
                    'best_epoch': valid_results[k].get('best_epoch', 0),
                }
                for i, k in enumerate(keys)
            }
        }

        json_path = output_path / f"training_summary_{session_id[:20]}_{safe_model}_{timestamp_file}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        return str(chart_path)

    except Exception as e:
        logger.error(f"Error generating training summary chart: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# COMPREHENSIVE CROSS-MODEL SUMMARY
# ============================================================================

def collect_session_metrics_from_disk(session_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan a session directory for all *_metrics.json files and return a unified
    dictionary of per-model metrics suitable for generate_comprehensive_summary_chart().

    Args:
        session_dir: Path to the session directory (e.g. logs/training_arena/{session_id})

    Returns:
        Dict mapping model/category names to their metrics:
        {
            'category_name': {
                'combined_score': float,
                'f1_macro': float,
                'f1_micro': float,
                'accuracy': float,
                'best_epoch': int,
                'training_time': float,  # estimated from epoch count
                'val_losses': list,
                'train_losses': list,
                'per_class_f1': dict,
                'language_metrics': dict,
                'num_labels': int,
                'num_samples_train': int,
                'num_samples_val': int,
            }
        }
    """
    session_path = Path(session_dir)
    collected: Dict[str, Dict[str, Any]] = {}

    if not session_path.exists():
        logger.warning(f"Session directory does not exist: {session_dir}")
        return collected

    # Search for all metrics JSON files in the session tree
    metrics_files = list(session_path.rglob("*_metrics.json"))

    if not metrics_files:
        logger.warning(f"No *_metrics.json files found in {session_dir}")
        return collected

    logger.info(f"Found {len(metrics_files)} metrics JSON files in {session_dir}")

    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            # Determine the category/model name
            cat_name = data.get('category_name') or metrics_file.parent.name or metrics_file.stem
            # Avoid duplicates: append parent dir if name already exists
            if cat_name in collected:
                cat_name = f"{cat_name}_{metrics_file.parent.name}"

            # Extract best-epoch metrics
            best_epoch = data.get('best_epoch', 0)
            best_idx = best_epoch - 1 if best_epoch and best_epoch > 0 else -1

            combined_scores = data.get('combined_scores', [])
            f1_macros = data.get('f1_macros', [])
            f1_micros = data.get('f1_micros', [])
            accuracies = data.get('accuracies', [])

            def _safe_get(arr, idx, default=0.0):
                """Get value from array at index, with fallback."""
                if not arr:
                    return default
                try:
                    return float(arr[idx])
                except (IndexError, TypeError, ValueError):
                    try:
                        return float(arr[-1])
                    except (IndexError, TypeError, ValueError):
                        return default

            entry = {
                'combined_score': _safe_get(combined_scores, best_idx),
                'f1_macro': _safe_get(f1_macros, best_idx),
                'f1_micro': _safe_get(f1_micros, best_idx),
                'accuracy': _safe_get(accuracies, best_idx),
                'best_epoch': best_epoch or 0,
                'num_labels': data.get('num_labels', 0),
                'num_samples_train': data.get('num_samples_train', 0),
                'num_samples_val': data.get('num_samples_val', 0),
                # Time series for overlay charts
                'val_losses': data.get('val_losses', []),
                'train_losses': data.get('train_losses', []),
                'per_class_f1': data.get('per_class_f1', {}),
                'language_metrics': data.get('language_metrics', {}),
                'label_names': data.get('label_names', []),
                'model_name': data.get('model_name', ''),
                'per_class_support': data.get('per_class_support', {}),
            }

            collected[cat_name] = entry
            logger.debug(f"Collected metrics for '{cat_name}': combined={entry['combined_score']:.4f}")

        except Exception as e:
            logger.warning(f"Failed to read metrics from {metrics_file}: {e}")
            continue

    logger.info(f"Collected metrics for {len(collected)} models from disk")
    return collected


# ── Helpers for comprehensive summary chart ──────────────────────────────────

def _ema(values, alpha=0.35):
    """Exponential moving average for loss curve smoothing."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _load_per_class_training_counts(session_dir, category):
    """Read JSONL training files to count per-class training instances.

    Handles both multiclass (labels=string) and multilabel (labels=list) formats.
    Returns Dict[str, int] mapping label -> count.
    """
    counts: Dict[str, int] = {}
    if not session_dir:
        return counts
    session_path = Path(session_dir)
    # Try multiclass JSONL first, then multilabel
    jsonl_files = list(session_path.glob(f"training_data/multiclass_{category}_*.jsonl"))
    if not jsonl_files:
        jsonl_files = list(session_path.glob(f"training_data/multilabel_{category}_*.jsonl"))
    if not jsonl_files:
        jsonl_files = list(session_path.glob(f"training_data/*_{category}_*.jsonl"))
    if not jsonl_files:
        return counts
    try:
        with open(jsonl_files[0], 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lbl = obj.get('labels', '')
                if isinstance(lbl, list):
                    for l in lbl:
                        counts[str(l)] = counts.get(str(l), 0) + 1
                else:
                    counts[str(lbl)] = counts.get(str(lbl), 0) + 1
    except Exception as e:
        logger.debug(f"Could not read JSONL for {category}: {e}")
    return counts


def _build_per_class_data(metrics, session_dir):
    """Merge per_class_f1 from metrics with training counts.

    Returns dict: {category: {classes: {label: {train_count, f1}}, n_pos, n_neg, best_epoch}}.
    """
    data = {}
    for cat, cat_metrics in metrics.items():
        if not isinstance(cat_metrics, dict) or 'error' in cat_metrics:
            continue
        per_class_f1 = cat_metrics.get('per_class_f1', {})
        per_class_support = cat_metrics.get('per_class_support', {})
        best_epoch = cat_metrics.get('best_epoch', 0)
        best_idx = best_epoch - 1 if best_epoch and best_epoch > 0 else -1

        # Training counts: prefer per_class_support from metrics; fall back to JSONL
        train_counts = {}
        if per_class_support:
            train_counts = {str(k): int(v) for k, v in per_class_support.items()}
        elif session_dir:
            train_counts = _load_per_class_training_counts(session_dir, cat)

        # Merge F1 + training count per class
        all_labels = set(list(per_class_f1.keys()) + list(train_counts.keys()))
        classes = {}
        for label in all_labels:
            f1_arr = per_class_f1.get(label, [])
            try:
                f1_val = float(f1_arr[best_idx]) if f1_arr else 0.0
            except (IndexError, TypeError, ValueError):
                try:
                    f1_val = float(f1_arr[-1]) if f1_arr else 0.0
                except (IndexError, TypeError, ValueError):
                    f1_val = 0.0
            classes[label] = {
                'train_count': train_counts.get(label, 0),
                'f1': f1_val,
            }

        # N+ / N- from training set (using _none suffix convention)
        n_pos = sum(v['train_count'] for l, v in classes.items()
                    if not l.endswith('_none'))
        n_neg = sum(v['train_count'] for l, v in classes.items()
                    if l.endswith('_none'))

        data[cat] = {
            'classes': classes,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'best_epoch': best_epoch,
        }
    return data


def _build_language_heatmap_data(metrics, categories, languages):
    """Extract F1 macro per category x language at best epoch.

    Returns dict: {category: {lang: float|nan}}.
    """
    heatmap = {}
    for cat in categories:
        cat_metrics = metrics.get(cat, {})
        if not isinstance(cat_metrics, dict):
            continue
        lang_data = cat_metrics.get('language_metrics', {})
        best_epoch = cat_metrics.get('best_epoch', 0)
        idx = best_epoch - 1 if best_epoch and best_epoch > 0 else -1
        heatmap[cat] = {}
        for lang in languages:
            if lang in lang_data and isinstance(lang_data[lang], dict):
                f1_arr = lang_data[lang].get('f1_macro', [])
                if f1_arr:
                    try:
                        heatmap[cat][lang] = float(f1_arr[idx])
                    except (IndexError, TypeError, ValueError):
                        try:
                            heatmap[cat][lang] = float(f1_arr[-1])
                        except (IndexError, TypeError, ValueError):
                            heatmap[cat][lang] = np.nan
                else:
                    heatmap[cat][lang] = np.nan
            else:
                heatmap[cat][lang] = np.nan
    return heatmap


def _wrap_label(text, max_len=22):
    """Wrap long label onto two lines at a word boundary."""
    if len(text) <= max_len:
        return text
    mid = len(text) // 2
    space = text.rfind(' ', 0, mid + 6)
    if space > 3:
        return text[:space] + '\n' + text[space + 1:]
    return text


# ── Drawing helpers for comprehensive summary chart ──────────────────────────

def _draw_bar_chart(ax, per_model, categories):
    """Grouped horizontal bar chart (combined, f1_macro, f1_micro, accuracy)."""
    try:
        sorted_cats = sorted(categories,
                             key=lambda c: per_model.get(c, {}).get('combined_score', 0),
                             reverse=True)
        n = len(sorted_cats)
        metrics_spec = [
            ('combined_score', 'Combined', COLORS['combined']),
            ('f1_macro',       'F1 Macro', COLORS['f1_macro']),
            ('f1_micro',       'F1 Micro', COLORS['f1_micro']),
            ('accuracy',       'Accuracy', COLORS['accuracy']),
        ]
        n_metrics = len(metrics_spec)
        bar_height = 0.18
        y_base = np.arange(n)

        for m_idx, (key, label, color) in enumerate(metrics_spec):
            offset = (m_idx - n_metrics / 2 + 0.5) * bar_height
            values = [per_model.get(c, {}).get(key, 0) for c in sorted_cats]
            bars = ax.barh(y_base + offset, values, height=bar_height * 0.9,
                           color=color, alpha=0.85, label=label, edgecolor='white',
                           linewidth=0.5, zorder=3)
            if key == 'combined_score':
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                                f'{val:.3f}', va='center', ha='left',
                                fontsize=7.5, fontweight='bold', color=color)

        ax.set_yticks(y_base)
        ax.set_yticklabels(sorted_cats, fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('Score', fontsize=10)

        # Dynamic x-axis limit
        max_val = max(
            (per_model.get(c, {}).get('combined_score', 0) for c in sorted_cats),
            default=0
        )
        ax.set_xlim(0, max(max_val * 1.15, 1.0))

        ax.set_title('Performance Comparison', fontsize=12, fontweight='bold', pad=10)

        # Average line
        avg = np.mean([per_model.get(c, {}).get('combined_score', 0) for c in sorted_cats])
        if avg > 0:
            ax.axvline(avg, color=COLORS['danger'], linestyle='--', linewidth=1.0,
                       alpha=0.5, zorder=4)
            ax.text(avg + 0.005, -0.6, f'Avg: {avg:.3f}', color=COLORS['danger'],
                    fontsize=7.5, fontweight='bold', va='top')

        ax.legend(fontsize=8, loc='lower right', frameon=True, fancybox=True,
                  edgecolor='#d1d5db', framealpha=0.95, ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.grid(axis='y', visible=False)
    except Exception as e:
        logger.warning(f"Error drawing bar chart: {e}")
        ax.text(0.5, 0.5, 'Bar chart unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='#94a3b8')


def _draw_loss_curves(ax, metrics, categories):
    """EMA-smoothed validation loss with best-epoch stars."""
    try:
        has_data = False
        max_epoch = 0
        for i, cat in enumerate(categories):
            cat_metrics = metrics.get(cat, {})
            val_losses = cat_metrics.get('val_losses', [])
            if not val_losses:
                continue
            has_data = True
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            raw = [float(v) for v in val_losses]
            smoothed = _ema(raw, alpha=0.35)
            epochs = list(range(1, len(raw) + 1))
            max_epoch = max(max_epoch, len(raw))

            ax.plot(epochs, raw, color=color, linewidth=0.6, alpha=0.18)
            ax.plot(epochs, smoothed, color=color, linewidth=2.5, alpha=0.90, label=cat)

            be = cat_metrics.get('best_epoch', 0)
            if be and 0 < be <= len(smoothed):
                ax.plot(be, smoothed[be - 1], '*', color=color, markersize=11,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=6)

        if has_data:
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Validation Loss', fontsize=10)
            ax.set_title('Validation Loss (EMA-smoothed)', fontsize=12,
                         fontweight='bold', pad=10)
            ax.grid(axis='both', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            n_cats = len(categories)
            ax.legend(fontsize=7 if n_cats > 10 else 8,
                      loc='upper right', frameon=True,
                      fancybox=True, edgecolor='#d1d5db', framealpha=0.95,
                      ncol=2 if n_cats > 10 else 1)
            ax.set_xlim(0.5, max_epoch + 0.5)
        else:
            ax.text(0.5, 0.5, 'No loss curve data available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='#94a3b8')
            ax.set_title('Validation Loss Curves', fontsize=12,
                         fontweight='bold', pad=10)
    except Exception as e:
        logger.warning(f"Error drawing loss curves: {e}")
        ax.text(0.5, 0.5, 'Loss curves unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='#94a3b8')


def _draw_booktabs_table(ax, per_model, categories, agg, per_class_data,
                         model_name, training_approach, total_training_time,
                         top_performer):
    """Publication-quality booktabs table with N+/N- columns."""
    try:
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        sorted_cats = sorted(categories,
                             key=lambda c: per_model.get(c, {}).get('combined_score', 0),
                             reverse=True)

        # Find best values for highlighting
        best = {}
        for key in ['combined_score', 'f1_macro', 'f1_micro', 'accuracy']:
            vals = [per_model.get(c, {}).get(key, 0) for c in sorted_cats]
            best[key] = max(vals) if vals else 0

        col_labels = ['Category', 'Combined', 'F1 Macro', 'F1 Micro', 'Accuracy',
                      'Best Ep.', 'N+', 'N\u2013']
        n_cols = len(col_labels)

        cell_text = []
        for cat in sorted_cats:
            d = per_model.get(cat, {})
            pcd = per_class_data.get(cat, {})
            n_pos = pcd.get('n_pos', 0)
            n_neg = pcd.get('n_neg', 0)
            cell_text.append([
                cat,
                f"{d.get('combined_score', 0):.4f}",
                f"{d.get('f1_macro', 0):.4f}",
                f"{d.get('f1_micro', 0):.4f}",
                f"{d.get('accuracy', 0):.4f}",
                str(d.get('best_epoch', 0)),
                str(n_pos) if n_pos > 0 else '--',
                str(n_neg) if n_neg > 0 else '--',
            ])

        # Summary row (Avg +/- Std)
        def _fmt_agg(key):
            a = agg.get(key, {})
            mean = a.get('mean', 0)
            std = a.get('std', 0)
            return f"{mean:.3f}\u00b1{std:.3f}"

        cell_text.append([
            'Avg \u00b1 Std',
            _fmt_agg('combined_score'),
            _fmt_agg('f1_macro'),
            _fmt_agg('f1_micro'),
            '', '', '', '',
        ])

        n_data = len(sorted_cats)
        white = (1.0, 1.0, 1.0, 1.0)
        alt_gray = (0.975, 0.978, 0.985, 1.0)
        cell_colors = []
        for i in range(n_data):
            cell_colors.append([alt_gray if i % 2 == 1 else white] * n_cols)
        cell_colors.append([(0.95, 0.95, 0.97, 1.0)] * n_cols)

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=[white] * n_cols,
            cellLoc='center',
            loc='center',
            bbox=[0.02, 0.08, 0.96, 0.82],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)

        n_total = n_data + 1

        for key, cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_linewidth(0)
            cell.PAD = 0.05

        for j in range(n_cols):
            table[0, j].set_text_props(fontweight='bold', fontsize=13, color=COLORS['dark'])
            table[0, j].set_facecolor('white')

        metric_keys_col = {1: 'combined_score', 2: 'f1_macro', 3: 'f1_micro', 4: 'accuracy'}
        for i in range(n_data):
            row_idx = i + 1
            table[row_idx, 0].set_text_props(fontweight='bold', fontsize=13)
            for j in range(1, n_cols):
                table[row_idx, j].set_text_props(fontsize=12, family='monospace')
            for j, key in metric_keys_col.items():
                if per_model.get(sorted_cats[i], {}).get(key, 0) == best[key] and best[key] > 0:
                    table[row_idx, j].set_text_props(fontweight='bold', color=COLORS['success'])

        for j in range(n_cols):
            table[n_total, j].set_text_props(
                fontsize=11, fontstyle='italic', color=COLORS['muted'], family='monospace')
        table[n_total, 0].set_text_props(
            fontsize=11, fontstyle='italic', fontweight='bold', color=COLORS['muted'])

        # Booktabs rules
        bbox = [0.02, 0.08, 0.96, 0.82]
        x_left, x_right = bbox[0], bbox[0] + bbox[2]
        y_top, y_bottom = bbox[1] + bbox[3], bbox[1]
        row_h = bbox[3] / (n_total + 1)

        ax.plot([x_left, x_right], [y_top + 0.005] * 2,
                color=COLORS['dark'], linewidth=2.0,
                transform=ax.transAxes, clip_on=False)
        ax.plot([x_left, x_right], [y_top - row_h + 0.002] * 2,
                color=COLORS['dark'], linewidth=0.8,
                transform=ax.transAxes, clip_on=False)
        ax.plot([x_left, x_right], [y_bottom + row_h + 0.002] * 2,
                color=COLORS['muted'], linewidth=0.5,
                transform=ax.transAxes, clip_on=False)
        ax.plot([x_left, x_right], [y_bottom - 0.005] * 2,
                color=COLORS['dark'], linewidth=2.0,
                transform=ax.transAxes, clip_on=False)

        ax.set_title('Training Summary', fontsize=14, fontweight='bold', pad=4)

        # Info footer
        total_time = total_training_time or 0
        if total_time > 3600:
            time_str = f"{total_time / 3600:.1f}h"
        elif total_time > 0:
            time_str = f"{total_time / 60:.0f}min"
        else:
            time_str = "N/A"

        top_name = top_performer.get('name', '?') if top_performer else '?'
        top_score = top_performer.get('combined_score', 0) if top_performer else 0
        safe_model = model_name.replace('/', '_') if model_name else '?'
        info = (f"{safe_model}  |  {training_approach}  |  "
                f"{time_str}  |  Best: {top_name} ({top_score:.3f})")
        ax.text(0.5, 0.01, info, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=8, color=COLORS['muted'],
                style='italic')
    except Exception as e:
        logger.warning(f"Error drawing booktabs table: {e}")
        ax.text(0.5, 0.5, 'Summary table unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='#94a3b8')


def _draw_f1_vs_support(ax, per_class_data, categories):
    """Numbered scatter plot of F1 vs training instances. Returns points list."""
    points = []
    try:
        num = 1
        for cat in categories:
            if cat not in per_class_data:
                continue
            for label, info in sorted(per_class_data[cat]['classes'].items(),
                                      key=lambda x: -x[1]['train_count']):
                if info['train_count'] == 0:
                    continue
                short = label.replace(f'{cat}_', '').replace('_', ' ')
                points.append((num, cat, short, info['train_count'], info['f1']))
                num += 1

        if not points:
            ax.text(0.5, 0.5, 'No per-class data available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='#94a3b8')
            ax.set_title('Per-Class F1 vs Training Set Size', fontsize=12,
                         fontweight='bold', pad=10)
            return points

        counts = np.array([p[3] for p in points], dtype=float)
        f1s = np.array([p[4] for p in points])
        cats = [p[1] for p in points]

        # Plot per category
        for i, cat in enumerate(categories):
            mask = np.array(cats) == cat
            if not mask.any():
                continue
            ax.scatter(counts[mask], f1s[mask],
                       color=CLASS_COLORS[i % len(CLASS_COLORS)],
                       s=80, alpha=0.85, edgecolors='white', linewidths=0.6,
                       label=cat, zorder=5)

        # Number annotations
        for idx, (num_id, cat, short, tc, f1) in enumerate(points):
            color = CLASS_COLORS[categories.index(cat) % len(CLASS_COLORS)]
            x_off = 6 if idx % 2 == 0 else -8
            y_off = 6 if idx % 3 != 2 else -8
            ax.annotate(str(num_id), (counts[idx], f1s[idx]),
                        textcoords='offset points', xytext=(x_off, y_off),
                        fontsize=7, fontweight='bold', color=color,
                        ha='center', va='center', zorder=7)

        # Trend line
        valid = f1s > 0
        if valid.sum() > 2:
            log_c = np.log10(counts[valid])
            f1_valid = f1s[valid]
            coeffs = np.polyfit(log_c, f1_valid, 1)
            r = np.corrcoef(log_c, f1_valid)[0, 1]
            x_fit = np.linspace(np.log10(counts.min()), np.log10(counts.max()), 100)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(10**x_fit, y_fit, '--', color=COLORS['dark'], linewidth=1.5,
                    alpha=0.5, zorder=4, label=f'trend (r={r:.2f})')

        # Zero-F1 markers
        zero_mask = f1s == 0
        if zero_mask.any():
            ax.scatter(counts[zero_mask], f1s[zero_mask],
                       color='#ef4444', s=45, marker='x', linewidths=1.5,
                       zorder=6, alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlabel('Training instances (log scale)', fontsize=10)
        ax.set_ylabel('F1 Score (at best epoch)', fontsize=10)
        ax.set_title('Per-Class F1 vs Training Set Size', fontsize=12,
                     fontweight='bold', pad=10)
        ax.set_ylim(-0.05, 1.0)
        ax.grid(axis='both', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7.5, loc='upper left', frameon=True,
                  fancybox=True, edgecolor='#d1d5db', framealpha=0.95, ncol=2)
    except Exception as e:
        logger.warning(f"Error drawing F1 vs support scatter: {e}")
        ax.text(0.5, 0.5, 'Scatter plot unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='#94a3b8')

    return points


def _draw_scatter_legend(ax, points, categories):
    """3-column reference table mapping point numbers to class names."""
    try:
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        n = len(points)
        if n == 0:
            ax.text(0.5, 0.5, 'No scatter legend data',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='#94a3b8')
            return

        n_cols_layout = 3
        rows_per_col = (n + n_cols_layout - 1) // n_cols_layout
        col_width = 1.0 / n_cols_layout
        row_h = 0.82 / max(rows_per_col, 1)
        top_y = 0.94

        # Title
        ax.text(0.5, 1.0, 'Scatter Plot Legend \u2014 Per-Class Detail',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color=COLORS['dark'])

        # Column headers
        for c in range(n_cols_layout):
            x_base = c * col_width + 0.01
            ax.text(x_base, top_y, '#', transform=ax.transAxes,
                    fontsize=9.5, fontweight='bold', color=COLORS['muted'], va='top')
            ax.text(x_base + 0.025, top_y, 'Class', transform=ax.transAxes,
                    fontsize=9.5, fontweight='bold', color=COLORS['muted'], va='top')
            ax.text(x_base + 0.275, top_y, 'N', transform=ax.transAxes,
                    fontsize=9.5, fontweight='bold', color=COLORS['muted'], va='top',
                    ha='right')
            ax.text(x_base + 0.32, top_y, 'F1', transform=ax.transAxes,
                    fontsize=9.5, fontweight='bold', color=COLORS['muted'], va='top',
                    ha='right')

        # Header separator
        ax.axhline(y=top_y - 0.02, xmin=0.005, xmax=0.995,
                   color=COLORS['muted'], linewidth=0.5)

        # Data rows
        for idx, (num_id, cat, short, tc, f1) in enumerate(points):
            col_idx = idx // rows_per_col
            row_idx = idx % rows_per_col
            x_base = col_idx * col_width + 0.01
            y = top_y - 0.05 - row_idx * row_h

            cat_idx = categories.index(cat) if cat in categories else 0
            color = CLASS_COLORS[cat_idx % len(CLASS_COLORS)]
            f1_color = COLORS['danger'] if f1 == 0 else (
                COLORS['success'] if f1 >= 0.6 else COLORS['dark'])

            # Colored dot + number
            ax.plot(x_base + 0.005, y, 'o', color=color, markersize=5,
                    transform=ax.transAxes, clip_on=False)
            ax.text(x_base + 0.022, y, str(num_id), transform=ax.transAxes,
                    fontsize=9, fontweight='bold', color=color,
                    va='center', ha='left')

            # Class name (wrapped)
            display_name = _wrap_label(short)
            ax.text(x_base + 0.045, y, display_name, transform=ax.transAxes,
                    fontsize=9, color=COLORS['dark'], va='center', ha='left',
                    family='monospace', linespacing=0.85)

            # N train
            ax.text(x_base + 0.275, y, str(tc), transform=ax.transAxes,
                    fontsize=9, color=COLORS['dark'], va='center', ha='right',
                    family='monospace')

            # F1
            ax.text(x_base + 0.32, y, f'{f1:.2f}', transform=ax.transAxes,
                    fontsize=9,
                    fontweight='bold' if f1 >= 0.6 or f1 == 0 else 'normal',
                    color=f1_color, va='center', ha='right',
                    family='monospace')
    except Exception as e:
        logger.warning(f"Error drawing scatter legend: {e}")


def _draw_language_heatmap(ax, heatmap_data, categories, languages):
    """Heatmap with colorbar and row averages."""
    try:
        n_cats = len(categories)
        n_langs = len(languages)

        matrix = np.full((n_cats, n_langs), np.nan)
        for i, cat in enumerate(categories):
            for j, lang in enumerate(languages):
                matrix[i, j] = heatmap_data.get(cat, {}).get(lang, np.nan)

        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color=COLORS['light'])

        valid_vals = matrix[~np.isnan(matrix)]
        vmin = max(0, np.nanmin(valid_vals) - 0.05) if len(valid_vals) > 0 else 0
        vmax = min(1, np.nanmax(valid_vals) + 0.05) if len(valid_vals) > 0 else 1

        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_xticks(np.arange(n_langs))
        ax.set_xticklabels(languages, fontsize=10, fontweight='bold')
        ax.set_yticks(np.arange(n_cats))
        ax.set_yticklabels(categories, fontsize=10, fontweight='bold')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        for i in range(n_cats):
            for j in range(n_langs):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, '--', ha='center', va='center',
                            fontsize=10, color=COLORS['muted'])
                else:
                    norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = 'white' if norm_val > 0.7 or norm_val < 0.2 else COLORS['dark']
                    fw = 'bold' if val >= 0.45 else 'normal'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=11, fontweight=fw, color=text_color)

        for i in range(n_cats + 1):
            ax.axhline(i - 0.5, color='white', linewidth=2.5)
        for j in range(n_langs + 1):
            ax.axvline(j - 0.5, color='white', linewidth=2.5)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04, aspect=20)
        cbar.set_label('F1 Macro', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title('F1 Macro by Category x Language',
                     fontsize=12, fontweight='bold', pad=12)

        # Row averages
        row_avgs = np.nanmean(matrix, axis=1)
        for i, avg_val in enumerate(row_avgs):
            if not np.isnan(avg_val):
                ax.text(n_langs - 0.5 + 0.25, i, f'{avg_val:.2f}',
                        ha='left', va='center', fontsize=9, color=COLORS['dark'],
                        fontweight='bold')
    except Exception as e:
        logger.warning(f"Error drawing language heatmap: {e}")
        ax.text(0.5, 0.5, 'Language heatmap unavailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='#94a3b8')


def generate_comprehensive_summary_chart(
    output_dir: str,
    session_id: str,
    model_name: str,
    session_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    session_dir: Optional[str] = None,
    results_per_key: Optional[Dict[str, Dict[str, Any]]] = None,
    training_approach: str = 'multi-model',
    total_training_time: Optional[float] = None,
) -> Optional[str]:
    """
    Generate a comprehensive publication-quality summary chart comparing all
    models trained in a session.

    This is the enhanced version of generate_training_summary_chart() with:
    - Loss curve overlays for all models
    - Language performance heatmap (if multilingual)
    - Dynamic layout (3x2 if multilingual, 2x2 otherwise)
    - JSON companion file with structured metrics

    Data source fallback chain:
        1. session_metrics (pre-collected dict)
        2. session_dir (scans disk via collect_session_metrics_from_disk)
        3. results_per_key (delegates to simple generate_training_summary_chart)

    Args:
        output_dir: Directory to save the chart
        session_id: Training session identifier
        model_name: Base model name
        session_metrics: Pre-collected metrics dict (from collect_session_metrics_from_disk)
        session_dir: Session directory path (will scan for metrics on disk)
        results_per_key: Legacy results dict (fallback)
        training_approach: Training approach label
        total_training_time: Total training time in seconds

    Returns:
        Path to the generated chart PNG, or None on failure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # ========== DATA ACQUISITION (fallback chain) ==========
        metrics = session_metrics

        if not metrics and session_dir:
            logger.info(f"No pre-collected metrics, scanning disk: {session_dir}")
            metrics = collect_session_metrics_from_disk(session_dir)

        if not metrics and results_per_key:
            # Delegate to the simpler chart function
            logger.info("Using results_per_key fallback (legacy path)")
            return generate_training_summary_chart(
                output_dir=output_dir,
                session_id=session_id,
                model_name=model_name,
                results_per_key=results_per_key,
                training_approach=training_approach,
                total_training_time=total_training_time,
            )

        if not metrics:
            logger.warning("No metrics available for comprehensive summary chart")
            return None

        # Filter out entries with errors
        valid = {
            k: v for k, v in metrics.items()
            if isinstance(v, dict) and 'error' not in v
        }
        if not valid:
            logger.warning("No valid model metrics for summary chart")
            return None

        # ========== EXTRACT METRICS ==========
        categories = sorted(valid.keys())
        n_models = len(categories)

        per_model: Dict[str, Dict[str, Any]] = {}
        for cat in categories:
            d = valid[cat]
            per_model[cat] = {
                'combined_score': float(d.get('combined_score', 0) or 0),
                'f1_macro': float(d.get('f1_macro', 0) or 0),
                'f1_micro': float(d.get('f1_micro', 0) or 0),
                'accuracy': float(d.get('accuracy', 0) or 0),
                'best_epoch': int(d.get('best_epoch', 0) or 0),
                'num_samples_train': int(d.get('num_samples_train', 0) or 0),
                'num_samples_val': int(d.get('num_samples_val', 0) or 0),
            }

        combined_scores = [per_model[c]['combined_score'] for c in categories]
        f1_macros = [per_model[c]['f1_macro'] for c in categories]
        f1_micros = [per_model[c]['f1_micro'] for c in categories]
        best_epochs = [per_model[c]['best_epoch'] for c in categories]

        # Aggregate metrics
        aggregate_metrics = {
            'combined_score': {
                'mean': float(np.mean(combined_scores)),
                'std': float(np.std(combined_scores)),
                'min': float(min(combined_scores)) if combined_scores else 0.0,
                'max': float(max(combined_scores)) if combined_scores else 0.0,
            },
            'f1_macro': {
                'mean': float(np.mean(f1_macros)),
                'std': float(np.std(f1_macros)),
                'min': float(min(f1_macros)) if f1_macros else 0.0,
                'max': float(max(f1_macros)) if f1_macros else 0.0,
            },
            'f1_micro': {
                'mean': float(np.mean(f1_micros)),
                'std': float(np.std(f1_micros)),
                'min': float(min(f1_micros)) if f1_micros else 0.0,
                'max': float(max(f1_micros)) if f1_micros else 0.0,
            },
        }

        # Top performer
        best_idx = int(np.argmax(combined_scores)) if combined_scores else 0
        top_performer = {
            'name': categories[best_idx],
            'combined_score': float(combined_scores[best_idx]),
            'f1_macro': float(f1_macros[best_idx]),
        }

        # Detect data availability
        has_loss_data = any(valid[k].get('val_losses') for k in categories)

        has_language = any(valid[k].get('language_metrics') for k in categories)
        all_languages: list = []
        if has_language:
            lang_set: set = set()
            for k in categories:
                lang_data = valid[k].get('language_metrics', {})
                if isinstance(lang_data, dict):
                    lang_set.update(lang_data.keys())
            all_languages = sorted(lang_set)
        has_heatmap = has_language and len(all_languages) > 1

        # Build per-class data
        per_class_data = _build_per_class_data(valid, session_dir)
        has_scatter = bool(per_class_data) and any(
            any(info['train_count'] > 0
                for info in pcd['classes'].values())
            for pcd in per_class_data.values()
        )

        # Build language heatmap data
        heatmap_data = {}
        if has_heatmap:
            heatmap_data = _build_language_heatmap_data(valid, categories, all_languages)

        # ========== DYNAMIC LAYOUT ==========
        # Determine row count and height ratios
        if has_scatter and has_heatmap:
            n_rows = 4
            height_ratios = [0.70, 0.55, 0.80, 0.60]
            base_height = 27
        elif has_scatter:
            n_rows = 4
            height_ratios = [0.70, 0.55, 0.80, 0.60]
            base_height = 27
        elif has_heatmap:
            n_rows = 3
            height_ratios = [0.70, 0.55, 0.80]
            base_height = 20
        else:
            n_rows = 2
            height_ratios = [0.70, 0.55]
            base_height = 14

        # Add extra height for many categories
        extra_height = max(0, (n_models - 10)) * 0.4
        fig_height = base_height + extra_height

        fig = plt.figure(figsize=(16, fig_height), dpi=200)
        gs = GridSpec(n_rows, 2, figure=fig, hspace=0.20, wspace=0.28,
                      height_ratios=height_ratios)

        # Title
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.suptitle(
            f'Comprehensive Training Summary\n'
            f'Model: {model_name} | {n_models} categories | '
            f'{training_approach.replace("-", " ").title()} | {timestamp}',
            fontsize=14, fontweight='bold', y=0.99)

        # ========== ROW 0: Bar chart + Loss curves ==========
        ax_bar = fig.add_subplot(gs[0, 0])
        _draw_bar_chart(ax_bar, per_model, categories)

        ax_loss = fig.add_subplot(gs[0, 1])
        _draw_loss_curves(ax_loss, valid, categories)

        # ========== ROW 1: Booktabs table (full width) ==========
        ax_table = fig.add_subplot(gs[1, :])
        _draw_booktabs_table(ax_table, per_model, categories, aggregate_metrics,
                             per_class_data, model_name, training_approach,
                             total_training_time, top_performer)

        # ========== ROW 2+: Scatter / Heatmap / Legend (conditional) ==========
        scatter_points = []
        if has_scatter and has_heatmap:
            # Row 2: scatter + heatmap side by side
            ax_scatter = fig.add_subplot(gs[2, 0])
            scatter_points = _draw_f1_vs_support(ax_scatter, per_class_data, categories)

            ax_heat = fig.add_subplot(gs[2, 1])
            _draw_language_heatmap(ax_heat, heatmap_data, categories, all_languages)

            # Row 3: scatter legend (full width)
            ax_legend = fig.add_subplot(gs[3, :])
            _draw_scatter_legend(ax_legend, scatter_points, categories)

        elif has_scatter:
            # Row 2: scatter (left), empty/legend placeholder (right)
            ax_scatter = fig.add_subplot(gs[2, 0])
            scatter_points = _draw_f1_vs_support(ax_scatter, per_class_data, categories)

            # Row 2 right: empty (could be used for something else)
            ax_empty = fig.add_subplot(gs[2, 1])
            ax_empty.axis('off')

            # Row 3: scatter legend (full width)
            ax_legend = fig.add_subplot(gs[3, :])
            _draw_scatter_legend(ax_legend, scatter_points, categories)

        elif has_heatmap:
            # Row 2: heatmap (full width)
            ax_heat = fig.add_subplot(gs[2, :])
            _draw_language_heatmap(ax_heat, heatmap_data, categories, all_languages)

        # ========== SAVE ==========
        fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.01)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_model = model_name.replace('/', '_').replace('\\', '_')[:30]
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_summary_{session_id[:20]}_{safe_model}_{timestamp_file}.png"
        chart_path = output_path / filename

        fig.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close(fig)

        logger.info(f"Comprehensive summary chart saved to {chart_path}")

        # ========== JSON COMPANION (backward-compatible + n_pos/n_neg) ==========
        summary_data = {
            'chart_type': 'comprehensive_summary',
            'session_id': session_id,
            'model_name': model_name,
            'training_approach': training_approach,
            'timestamp': datetime.now().isoformat(),
            'models_count': n_models,
            'has_language_data': has_language,
            'languages': all_languages if has_language else [],
            'total_training_time': total_training_time,
            'aggregate_metrics': aggregate_metrics,
            'top_performer': top_performer,
            'per_model': {
                cat: {
                    'combined_score': per_model[cat]['combined_score'],
                    'f1_macro': per_model[cat]['f1_macro'],
                    'f1_micro': per_model[cat]['f1_micro'],
                    'accuracy': per_model[cat]['accuracy'],
                    'best_epoch': per_model[cat]['best_epoch'],
                    'num_samples_train': per_model[cat]['num_samples_train'],
                    'num_samples_val': per_model[cat]['num_samples_val'],
                    'n_pos': per_class_data.get(cat, {}).get('n_pos', 0),
                    'n_neg': per_class_data.get(cat, {}).get('n_neg', 0),
                }
                for cat in categories
            },
        }

        json_path = output_path / f"comprehensive_summary_{session_id[:20]}_{safe_model}_{timestamp_file}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Comprehensive summary JSON saved to {json_path}")
        return str(chart_path)

    except Exception as e:
        logger.error(f"Error generating comprehensive summary chart: {e}")
        import traceback
        traceback.print_exc()
        return None

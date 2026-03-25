#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
annotation_metrics_chart.py

MAIN OBJECTIVE:
---------------
Generate publication-quality annotation metrics charts for LLM annotation sessions.
Provides comprehensive visualizations of annotation distribution, success rates,
key/value breakdowns, and language distribution for annotation workflows.

Dependencies:
-------------
- os
- json
- logging
- pathlib
- typing
- datetime
- numpy
- pandas
- matplotlib

MAIN FEATURES:
--------------
1) Generate annotation value distribution charts (pie charts, bar charts)
2) Visualize success/error rate breakdown
3) Display per-key annotation distribution with value counts
4) Show language distribution for multilingual datasets
5) Track annotation timing and throughput metrics
6) Export metrics as JSON for further analysis

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
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
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
    'success': '#22c55e',        # Green
    'error': '#ef4444',          # Red
    'warning': '#f59e0b',        # Amber
    'info': '#3b82f6',           # Blue
    'primary': '#6366f1',        # Indigo
    'secondary': '#8b5cf6',      # Violet
    'neutral': '#6b7280',        # Gray
    'dark': '#1f2937',
    'light': '#f3f4f6',
}

# Value distribution colors (for pie charts and bar charts)
VALUE_COLORS = [
    '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
    '#f43f5e', '#ef4444', '#f97316', '#f59e0b', '#eab308',
    '#84cc16', '#22c55e', '#10b981', '#14b8a6', '#06b6d4',
    '#0ea5e9', '#3b82f6', '#2563eb', '#4f46e5', '#7c3aed',
]

# Language colors
LANGUAGE_COLORS = {
    'en': '#3b82f6',
    'fr': '#ef4444',
    'de': '#22c55e',
    'es': '#f59e0b',
    'it': '#8b5cf6',
    'pt': '#06b6d4',
    'nl': '#ec4899',
    'ru': '#6366f1',
    'zh': '#14b8a6',
    'ja': '#f43f5e',
    'ko': '#84cc16',
    'ar': '#d946ef',
    'unknown': '#6b7280',
}


class AnnotationMetricsChart:
    """
    Generates publication-quality annotation metrics charts.

    This class creates comprehensive visualizations of annotation results
    including value distributions, success rates, and language breakdowns.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        session_id: str,
        model_name: str,
        annotation_keys: Optional[List[str]] = None,
        workflow_name: str = "The Annotator",
    ):
        """
        Initialize the annotation metrics chart generator.

        Args:
            output_dir: Directory to save charts
            session_id: Unique session identifier
            model_name: Name of the model used for annotation
            annotation_keys: List of annotation key names
            workflow_name: Name of the workflow (The Annotator or Annotator Factory)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id
        self.model_name = model_name
        self.annotation_keys = annotation_keys or []
        self.workflow_name = workflow_name
        self.timestamp = datetime.now()

        # Metrics storage
        self.metrics: Dict[str, Any] = {
            'total_rows': 0,
            'annotated_rows': 0,
            'success_count': 0,
            'error_count': 0,
            'value_counts_per_key': {},
            'language_distribution': {},
            'timing_metrics': {},
            'status_breakdown': {},
        }

        logger.info(f"AnnotationMetricsChart initialized for session {session_id}")

    def update_from_dataframe(
        self,
        df: pd.DataFrame,
        annotation_column: str,
        text_column: Optional[str] = None,
        language_column: Optional[str] = None,
    ) -> None:
        """
        Update metrics from an annotated DataFrame.

        Args:
            df: DataFrame with annotation results
            annotation_column: Name of the column containing annotations
            text_column: Optional name of the text column
            language_column: Optional name of the language column
        """
        try:
            self.metrics['total_rows'] = len(df)

            # Count annotated rows (non-null annotations)
            annotated_mask = df[annotation_column].notna()
            if annotated_mask.sum() > 0:
                # Check if values are empty strings or empty dicts
                def is_valid_annotation(val):
                    if pd.isna(val):
                        return False
                    if isinstance(val, str):
                        val = val.strip()
                        if not val or val in ('{}', '[]', 'null', 'None'):
                            return False
                    return True

                annotated_mask = df[annotation_column].apply(is_valid_annotation)

            self.metrics['annotated_rows'] = int(annotated_mask.sum())

            # Parse annotations and count values per key
            value_counts_per_key: Dict[str, Counter] = {}

            for idx, row in df[annotated_mask].iterrows():
                annotation = row[annotation_column]

                try:
                    if isinstance(annotation, str):
                        try:
                            annotation_dict = json.loads(annotation)
                        except json.JSONDecodeError:
                            import ast
                            annotation_dict = ast.literal_eval(annotation)
                    elif isinstance(annotation, dict):
                        annotation_dict = annotation
                    else:
                        continue

                    for key, value in annotation_dict.items():
                        if key not in value_counts_per_key:
                            value_counts_per_key[key] = Counter()

                        if isinstance(value, list):
                            for v in value:
                                if v is not None and v != '':
                                    value_counts_per_key[key][str(v)] += 1
                        elif value is not None and value != '':
                            value_counts_per_key[key][str(value)] += 1

                except (json.JSONDecodeError, ValueError, TypeError, SyntaxError):
                    continue

            # Convert Counters to regular dicts for JSON serialization
            self.metrics['value_counts_per_key'] = {
                key: dict(counter) for key, counter in value_counts_per_key.items()
            }

            # Update annotation keys if not already set
            if not self.annotation_keys and value_counts_per_key:
                self.annotation_keys = list(value_counts_per_key.keys())

            # Language distribution
            if language_column and language_column in df.columns:
                lang_counts = df[language_column].value_counts().to_dict()
                self.metrics['language_distribution'] = {
                    str(k): int(v) for k, v in lang_counts.items() if pd.notna(k)
                }

            logger.info(f"Updated metrics from DataFrame: {self.metrics['annotated_rows']} annotated rows")

        except Exception as e:
            logger.error(f"Error updating metrics from DataFrame: {e}")

    def update_from_results(
        self,
        annotation_results: Dict[str, Any],
    ) -> None:
        """
        Update metrics from annotation results dictionary.

        Args:
            annotation_results: Dictionary with annotation results from the workflow
        """
        try:
            if 'total_annotated' in annotation_results:
                self.metrics['annotated_rows'] = annotation_results['total_annotated']

            if 'success_count' in annotation_results:
                self.metrics['success_count'] = annotation_results['success_count']

            if 'status_breakdown' in annotation_results:
                self.metrics['status_breakdown'] = annotation_results['status_breakdown']

            if 'mean_inference_time' in annotation_results:
                self.metrics['timing_metrics']['mean_inference_time'] = annotation_results['mean_inference_time']

            if 'prompt_success_count' in annotation_results:
                self.metrics['timing_metrics']['prompt_success_count'] = annotation_results['prompt_success_count']

            logger.info(f"Updated metrics from results: {self.metrics}")

        except Exception as e:
            logger.error(f"Error updating metrics from results: {e}")

    def generate_chart(self) -> Optional[Path]:
        """
        Generate the comprehensive annotation metrics chart.

        Returns:
            Path to the generated chart, or None if generation failed
        """
        try:
            # Determine layout based on available data
            has_multiple_keys = len(self.metrics.get('value_counts_per_key', {})) > 1
            has_language_data = bool(self.metrics.get('language_distribution'))

            # Create figure with appropriate size
            if has_multiple_keys:
                fig = plt.figure(figsize=(16, 12))
            else:
                fig = plt.figure(figsize=(14, 10))

            # Create grid layout
            if has_multiple_keys and has_language_data:
                gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)
            elif has_multiple_keys:
                gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1], hspace=0.35, wspace=0.3)
            else:
                gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

            # Title
            title_parts = [
                f"Annotation Metrics - {self.workflow_name}",
                f"Session: {self.session_id}",
                f"Model: {self.model_name}",
                f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
            ]
            fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold', y=0.98)

            # 1. Summary statistics (top left)
            ax_summary = fig.add_subplot(gs[0, 0])
            self._plot_summary_stats(ax_summary)

            # 2. Success/Error breakdown (top right)
            ax_status = fig.add_subplot(gs[0, 1])
            self._plot_status_breakdown(ax_status)

            # 3. Value distribution per key
            value_counts = self.metrics.get('value_counts_per_key', {})
            if value_counts:
                if len(value_counts) == 1:
                    # Single key - larger plot
                    ax_values = fig.add_subplot(gs[1, :])
                    key = list(value_counts.keys())[0]
                    self._plot_value_distribution(ax_values, key, value_counts[key])
                else:
                    # Multiple keys - grid of subplots
                    keys = list(value_counts.keys())
                    n_keys = len(keys)
                    cols = min(3, n_keys)
                    rows = (n_keys + cols - 1) // cols

                    for i, key in enumerate(keys[:6]):  # Limit to 6 keys
                        row_idx = i // cols
                        col_idx = i % cols
                        ax = fig.add_subplot(gs[1 + row_idx, col_idx])
                        self._plot_value_distribution(ax, key, value_counts[key], compact=True)

            # 4. Language distribution (if available)
            if has_language_data:
                ax_lang = fig.add_subplot(gs[-1, :] if not has_multiple_keys else gs[-1, -1])
                self._plot_language_distribution(ax_lang)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Generate filename
            safe_model_name = self.model_name.replace('/', '_').replace(':', '_')[:30]
            chart_filename = (
                f"annotation_metrics_{self.session_id}_"
                f"{safe_model_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            )
            chart_path = self.output_dir / chart_filename

            # Save chart
            fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"Annotation metrics chart saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Error generating annotation metrics chart: {e}")
            return None

    def _plot_summary_stats(self, ax: plt.Axes) -> None:
        """Plot summary statistics as a text box."""
        ax.axis('off')

        total = self.metrics.get('total_rows', 0)
        annotated = self.metrics.get('annotated_rows', 0)
        success = self.metrics.get('success_count', annotated)

        success_rate = (success / annotated * 100) if annotated > 0 else 0
        coverage = (annotated / total * 100) if total > 0 else 0

        timing = self.metrics.get('timing_metrics', {})
        mean_time = timing.get('mean_inference_time', None)

        # Build stats text
        stats_lines = [
            f"Total Rows: {total:,}",
            f"Annotated: {annotated:,}",
            f"Coverage: {coverage:.1f}%",
            f"Success Rate: {success_rate:.1f}%",
        ]

        if mean_time:
            stats_lines.append(f"Avg Time: {mean_time:.2f}s")

        # Draw text box
        props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], alpha=0.8)
        ax.text(0.5, 0.5, '\n'.join(stats_lines),
                transform=ax.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                bbox=props, family='monospace')

        ax.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=10)

    def _plot_status_breakdown(self, ax: plt.Axes) -> None:
        """Plot status breakdown as a pie chart."""
        status_breakdown = self.metrics.get('status_breakdown', {})

        if not status_breakdown:
            # Create default from success/annotated
            annotated = self.metrics.get('annotated_rows', 0)
            success = self.metrics.get('success_count', annotated)
            error = annotated - success if annotated > success else 0

            if annotated > 0:
                status_breakdown = {'success': success}
                if error > 0:
                    status_breakdown['error'] = error

        if not status_breakdown:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No status data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        labels = list(status_breakdown.keys())
        sizes = list(status_breakdown.values())

        # Assign colors based on status type
        colors = []
        for label in labels:
            label_lower = label.lower()
            if 'success' in label_lower or 'ok' in label_lower:
                colors.append(COLORS['success'])
            elif 'error' in label_lower or 'fail' in label_lower:
                colors.append(COLORS['error'])
            elif 'warn' in label_lower:
                colors.append(COLORS['warning'])
            else:
                colors.append(COLORS['info'])

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.75, labeldistance=1.1
        )

        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        ax.set_title('Status Breakdown', fontsize=11, fontweight='bold', pad=10)

    def _plot_value_distribution(
        self,
        ax: plt.Axes,
        key: str,
        value_counts: Dict[str, int],
        compact: bool = False,
    ) -> None:
        """Plot value distribution for a single key."""
        if not value_counts:
            ax.axis('off')
            ax.text(0.5, 0.5, f'No values for {key}',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Sort by count and limit to top values
        sorted_items = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        max_values = 8 if compact else 15

        if len(sorted_items) > max_values:
            top_items = sorted_items[:max_values-1]
            other_count = sum(count for _, count in sorted_items[max_values-1:])
            top_items.append(('Other', other_count))
            sorted_items = top_items

        values = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Truncate long value names
        display_values = []
        for v in values:
            if len(v) > 20:
                display_values.append(v[:17] + '...')
            else:
                display_values.append(v)

        # Create horizontal bar chart
        y_pos = np.arange(len(values))
        colors = [VALUE_COLORS[i % len(VALUE_COLORS)] for i in range(len(values))]

        bars = ax.barh(y_pos, counts, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_values, fontsize=8 if compact else 10)
        ax.invert_yaxis()

        # Add count labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=8 if compact else 9)

        ax.set_xlabel('Count', fontsize=9)
        ax.set_title(f'Distribution: {key}', fontsize=10 if compact else 11, fontweight='bold', pad=10)
        ax.set_xlim(0, max(counts) * 1.15)

    def _plot_language_distribution(self, ax: plt.Axes) -> None:
        """Plot language distribution."""
        lang_dist = self.metrics.get('language_distribution', {})

        if not lang_dist:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No language data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Sort by count
        sorted_items = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
        languages = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Get colors
        colors = [LANGUAGE_COLORS.get(lang.lower(), COLORS['neutral']) for lang in languages]

        # Create bar chart
        x_pos = np.arange(len(languages))
        bars = ax.bar(x_pos, counts, color=colors, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([lang.upper() for lang in languages], fontsize=10)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{count:,}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Language Distribution', fontsize=11, fontweight='bold', pad=10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def save_metrics_json(self) -> Optional[Path]:
        """
        Save metrics to a JSON file for further analysis.

        Returns:
            Path to the JSON file, or None if saving failed
        """
        try:
            metrics_data = {
                'session_id': self.session_id,
                'model_name': self.model_name,
                'workflow_name': self.workflow_name,
                'timestamp': self.timestamp.isoformat(),
                'annotation_keys': self.annotation_keys,
                'metrics': self.metrics,
            }

            safe_model_name = self.model_name.replace('/', '_').replace(':', '_')[:30]
            json_filename = (
                f"annotation_metrics_{self.session_id}_"
                f"{safe_model_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            )
            json_path = self.output_dir / json_filename

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Annotation metrics JSON saved to {json_path}")
            return json_path

        except Exception as e:
            logger.error(f"Error saving metrics JSON: {e}")
            return None


def generate_annotation_chart(
    output_dir: Union[str, Path],
    session_id: str,
    model_name: str,
    df: Optional[pd.DataFrame] = None,
    annotation_column: Optional[str] = None,
    annotation_results: Optional[Dict[str, Any]] = None,
    language_column: Optional[str] = None,
    workflow_name: str = "The Annotator",
) -> Optional[Path]:
    """
    Convenience function to generate an annotation metrics chart.

    Args:
        output_dir: Directory to save the chart
        session_id: Session identifier
        model_name: Name of the model used
        df: Optional DataFrame with annotations
        annotation_column: Name of the annotation column
        annotation_results: Optional results dictionary from annotation workflow
        language_column: Optional language column name
        workflow_name: Name of the workflow

    Returns:
        Path to the generated chart, or None if generation failed
    """
    try:
        chart = AnnotationMetricsChart(
            output_dir=output_dir,
            session_id=session_id,
            model_name=model_name,
            workflow_name=workflow_name,
        )

        if df is not None and annotation_column:
            chart.update_from_dataframe(
                df=df,
                annotation_column=annotation_column,
                language_column=language_column,
            )

        if annotation_results:
            chart.update_from_results(annotation_results)

        chart_path = chart.generate_chart()
        chart.save_metrics_json()

        return chart_path

    except Exception as e:
        logger.error(f"Error generating annotation chart: {e}")
        return None

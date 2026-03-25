#!/usr/bin/env python3
"""
Data Quality vs Model Performance Analysis
==========================================

Visualizes the relationship between training data characteristics
and model performance metrics. Helps understand:

- How class imbalance affects F1 scores
- Correlation between sample size and model performance
- Optimal sample thresholds for reliable predictions
- Data quality indicators and their impact

Author: Antoine Lemor
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import curve_fit
import warnings
import json

warnings.filterwarnings('ignore')

# Scientific style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def parse_class_legend(csv_path: str) -> Dict[int, str]:
    """Extract class names from CSV header comment."""
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    if not first_line.startswith('# CLASS_LEGEND:'):
        return {}

    legend_str = first_line.replace('# CLASS_LEGEND:', '').strip()
    class_map = {}

    for item in legend_str.split(', '):
        if '=' in item:
            idx, name = item.split('=', 1)
            # Clean class name
            clean_name = name.split('_')[-1] if '_' in name else name
            # Handle multi-word names
            if 'rights_liberties' in name.lower():
                clean_name = 'rights_liberties'
            elif 'law_and_crime' in name.lower():
                clean_name = 'law_crime'
            elif 'culture_nationalism' in name.lower():
                clean_name = 'culture_nation'
            elif 'domestic_commerce' in name.lower():
                clean_name = 'commerce'
            elif 'foreign_trade' in name.lower():
                clean_name = 'foreign_trade'
            elif 'governments_governance' in name.lower():
                clean_name = 'governance'
            elif 'international_affairs' in name.lower():
                clean_name = 'intl_affairs'
            elif 'macroeconomics' in name.lower():
                clean_name = 'macroeconomy'
            elif 'public_lands' in name.lower():
                clean_name = 'public_lands'
            elif 'social_welfare' in name.lower():
                clean_name = 'social_welfare'
            class_map[int(idx)] = clean_name.capitalize()

    return class_map


def load_training_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[int, str], int]:
    """Load training CSV and extract class information."""
    class_map = parse_class_legend(csv_path)
    n_classes = len(class_map) if class_map else 0

    df = pd.read_csv(csv_path, skiprows=1)

    # Auto-detect number of classes if not from legend
    if n_classes == 0:
        f1_cols = [c for c in df.columns if c.startswith('f1_') and c[3:].isdigit()]
        n_classes = len(f1_cols)
        class_map = {i: f'Class_{i}' for i in range(n_classes)}

    return df, class_map, n_classes


def count_training_samples(jsonl_path: str) -> Dict[str, int]:
    """Count samples per class from training JSONL file."""
    from collections import Counter

    with open(jsonl_path, 'r') as f:
        labels = []
        for line in f:
            data = json.loads(line)
            label = data.get('labels', data.get('label', ''))
            if isinstance(label, list):
                labels.extend(label)
            else:
                labels.append(label)

    return dict(Counter(labels))


def logarithmic_fit(x, a, b, c):
    """Logarithmic function for fitting: f(x) = a * log(x + 1) + b"""
    return a * np.log(x + 1) + b


def power_fit(x, a, b, c):
    """Power function for fitting: f(x) = a * x^b + c"""
    return a * np.power(x, b) + c


def create_sample_f1_scatter(ax, samples: np.ndarray, f1_scores: np.ndarray,
                             class_names: List[str], title: str = 'Sample Size vs F1 Score'):
    """Create scatter plot with regression showing sample size effect on F1."""

    # Color by F1 score
    colors = plt.cm.RdYlGn(f1_scores)

    # Create scatter
    scatter = ax.scatter(samples, f1_scores, c=f1_scores, cmap='RdYlGn',
                        s=120, edgecolors='black', linewidth=0.5, alpha=0.8,
                        vmin=0, vmax=1)

    # Add class labels
    for i, (x, y, name) in enumerate(zip(samples, f1_scores, class_names)):
        # Offset to avoid overlap
        offset_x = 0.02 * samples.max()
        offset_y = 0.02
        ax.annotate(name, (x, y), xytext=(x + offset_x, y + offset_y),
                   fontsize=7, alpha=0.8,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))

    # Fit logarithmic curve (sample size usually has diminishing returns)
    try:
        # Filter out zero F1 scores for fitting
        mask = f1_scores > 0.05
        if mask.sum() >= 3:
            x_fit = samples[mask]
            y_fit = f1_scores[mask]

            # Logarithmic fit
            popt, _ = curve_fit(logarithmic_fit, x_fit, y_fit,
                               p0=[0.1, 0.3, 0], maxfev=5000)

            x_line = np.linspace(samples.min(), samples.max(), 100)
            y_line = logarithmic_fit(x_line, *popt)
            y_line = np.clip(y_line, 0, 1)

            ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
                   label=f'Fit: F1 = {popt[0]:.3f}*log(n+1) + {popt[1]:.3f}')

            # Calculate R-squared
            y_pred = logarithmic_fit(x_fit, *popt)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontweight='bold')
    except Exception as e:
        print(f"Could not fit curve: {e}")

    # Add threshold zones
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='F1 = 0.5 threshold')
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='F1 = 0.7 threshold')

    # Add sample size recommendation zones
    ax.axvspan(0, 50, alpha=0.1, color='red', label='Insufficient (<50)')
    ax.axvspan(50, 200, alpha=0.1, color='yellow')
    ax.axvspan(200, samples.max(), alpha=0.1, color='green')

    ax.set_xlabel('Training Samples', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlim(-samples.max() * 0.05, samples.max() * 1.1)
    ax.set_ylim(-0.05, 1.1)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('F1 Score', fontweight='bold')

    ax.legend(loc='lower right', fontsize=8)

    return scatter


def create_imbalance_impact(ax, samples: np.ndarray, f1_scores: np.ndarray,
                           class_names: List[str]):
    """Show how class imbalance affects performance."""

    # Calculate imbalance ratio (class size / max class size)
    max_samples = samples.max()
    imbalance_ratio = samples / max_samples

    # Group by imbalance quartiles (allow duplicates for skewed distributions)
    quartile_labels = ['Q1 (Rare)', 'Q2', 'Q3', 'Q4 (Frequent)']
    imbalance_series = pd.Series(imbalance_ratio)
    try:
        quartiles = pd.qcut(imbalance_series, q=4, labels=False, duplicates='drop')
        # Map numeric bins to labels based on actual number of bins
        n_bins = int(quartiles.max()) + 1
        label_map = {i: quartile_labels[min(i, len(quartile_labels)-1)] for i in range(n_bins)}
        quartiles = quartiles.map(label_map)
    except ValueError:
        # Fallback to cut if qcut fails
        quartiles = pd.cut(imbalance_series, bins=4, labels=quartile_labels)

    # Create box plot with actual categories present
    actual_categories = [q for q in quartile_labels if (quartiles == q).any()]
    data_for_plot = []
    for q in actual_categories:
        mask = quartiles == q
        data_for_plot.append(f1_scores[mask])

    short_labels = {'Q1 (Rare)': 'Q1\n(Rare)', 'Q2': 'Q2', 'Q3': 'Q3', 'Q4 (Frequent)': 'Q4\n(Frequent)'}
    bp = ax.boxplot(data_for_plot, labels=[short_labels[q] for q in actual_categories],
                    patch_artist=True)

    # Color boxes
    all_colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
    colors = [all_colors[quartile_labels.index(q)] for q in actual_categories]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, q in enumerate(actual_categories):
        mask = quartiles == q
        x_jitter = np.random.normal(i + 1, 0.05, mask.sum())
        ax.scatter(x_jitter, f1_scores[mask], alpha=0.6, s=40, color='black', zorder=5)

    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_xlabel('Class Frequency Quartile', fontweight='bold')
    ax.set_title('Impact of Class Imbalance on Performance', fontweight='bold', pad=10)
    ax.set_ylim(-0.05, 1.1)

    # Add statistics
    means = [np.mean(d) for d in data_for_plot]
    for i, (mean, d) in enumerate(zip(means, data_for_plot)):
        ax.text(i + 1, 1.05, f'n={len(d)}\n$\\mu$={mean:.2f}',
               ha='center', fontsize=8, fontweight='bold')


def create_threshold_analysis(ax, samples: np.ndarray, f1_scores: np.ndarray):
    """Analyze minimum sample threshold for reliable performance."""

    # Create sample thresholds
    thresholds = np.arange(10, min(500, samples.max()), 20)

    mean_f1_above = []
    mean_f1_below = []
    n_above = []
    n_below = []

    for thresh in thresholds:
        mask_above = samples >= thresh
        mask_below = samples < thresh

        if mask_above.sum() > 0:
            mean_f1_above.append(f1_scores[mask_above].mean())
            n_above.append(mask_above.sum())
        else:
            mean_f1_above.append(np.nan)
            n_above.append(0)

        if mask_below.sum() > 0:
            mean_f1_below.append(f1_scores[mask_below].mean())
            n_below.append(mask_below.sum())
        else:
            mean_f1_below.append(np.nan)
            n_below.append(0)

    # Plot
    ax.plot(thresholds, mean_f1_above, 'g-o', linewidth=2, markersize=5,
           label='Classes above threshold')
    ax.plot(thresholds, mean_f1_below, 'r-s', linewidth=2, markersize=5,
           label='Classes below threshold')

    # Fill between
    ax.fill_between(thresholds, mean_f1_above, mean_f1_below,
                    alpha=0.2, color='gray')

    # Find optimal threshold (maximum gap)
    gap = np.array(mean_f1_above) - np.array(mean_f1_below)
    valid_mask = ~np.isnan(gap)
    if valid_mask.any():
        best_idx = np.nanargmax(gap)
        best_thresh = thresholds[best_idx]

        ax.axvline(x=best_thresh, color='blue', linestyle='--', linewidth=2,
                  label=f'Optimal threshold: {best_thresh}')

        ax.annotate(f'Max gap at {best_thresh} samples\n'
                   f'Above: {mean_f1_above[best_idx]:.2f}\n'
                   f'Below: {mean_f1_below[best_idx]:.2f}',
                   xy=(best_thresh, (mean_f1_above[best_idx] + mean_f1_below[best_idx])/2),
                   xytext=(best_thresh + 50, 0.5),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='blue'))

    ax.set_xlabel('Sample Size Threshold', fontweight='bold')
    ax.set_ylabel('Mean F1 Score', fontweight='bold')
    ax.set_title('Minimum Sample Threshold Analysis', fontweight='bold', pad=10)
    ax.legend(loc='center right')
    ax.set_ylim(0, 1.1)


def create_quality_indicators(ax, samples: np.ndarray, f1_scores: np.ndarray,
                             class_names: List[str]):
    """Create radar chart of data quality indicators."""

    # Calculate quality indicators
    total_samples = samples.sum()
    n_classes = len(samples)

    # 1. Balance score (1 = perfectly balanced, 0 = highly imbalanced)
    ideal_per_class = total_samples / n_classes
    balance_score = 1 - np.std(samples) / (samples.max() - samples.min() + 1)
    balance_score = max(0, min(1, balance_score))

    # 2. Coverage score (how many classes have "enough" samples)
    min_threshold = 50
    coverage_score = (samples >= min_threshold).sum() / n_classes

    # 3. Performance consistency (1 - std of F1 scores)
    consistency_score = 1 - np.std(f1_scores) if np.std(f1_scores) < 1 else 0

    # 4. Mean F1 score
    mean_f1 = f1_scores.mean()

    # 5. Minimum class viability (F1 of worst class)
    min_viability = f1_scores.min()

    # 6. Sample efficiency (F1 / log(samples) ratio normalized)
    efficiency = (f1_scores / np.log1p(samples)).mean()
    efficiency_score = min(1, efficiency * 2)  # Normalize

    # Radar chart
    categories = ['Balance', 'Coverage', 'Consistency', 'Mean F1', 'Min Viability', 'Efficiency']
    values = [balance_score, coverage_score, consistency_score, mean_f1, min_viability, efficiency_score]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]

    ax.plot(angles, values_plot, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values_plot, alpha=0.25, color='steelblue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Data Quality Indicators', fontweight='bold', pad=20)

    # Add value labels
    for angle, value, cat in zip(angles[:-1], values, categories):
        ax.annotate(f'{value:.2f}', xy=(angle, value), xytext=(angle, value + 0.1),
                   ha='center', fontsize=8, fontweight='bold')


def create_learning_efficiency(ax, df: pd.DataFrame, samples: np.ndarray,
                              n_classes: int, class_names: List[str]):
    """Show learning efficiency: epochs needed to reach good F1 vs sample size."""

    epochs = df['epoch'].values
    threshold_f1 = 0.5  # F1 threshold to consider "learned"

    epochs_to_learn = []
    for i in range(n_classes):
        col = f'f1_{i}'
        if col in df.columns:
            f1_series = df[col].values
            reached = np.where(f1_series >= threshold_f1)[0]
            if len(reached) > 0:
                epochs_to_learn.append(epochs[reached[0]])
            else:
                epochs_to_learn.append(epochs[-1] + 5)  # Never reached
        else:
            epochs_to_learn.append(np.nan)

    epochs_to_learn = np.array(epochs_to_learn)

    # Scatter plot
    colors = plt.cm.viridis(samples / samples.max())

    scatter = ax.scatter(samples, epochs_to_learn, c=samples, cmap='viridis',
                        s=100, edgecolors='black', linewidth=0.5)

    # Add class names for slow learners
    for i, (x, y, name) in enumerate(zip(samples, epochs_to_learn, class_names)):
        if y > epochs.max() or y > np.median(epochs_to_learn) * 1.5:
            ax.annotate(name, (x, y), fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Training Samples', fontweight='bold')
    ax.set_ylabel(f'Epochs to reach F1 >= {threshold_f1}', fontweight='bold')
    ax.set_title('Learning Efficiency by Class Size', fontweight='bold', pad=10)

    # Add reference line
    ax.axhline(y=epochs[-1], color='red', linestyle='--', alpha=0.5,
              label=f'Max epochs ({epochs[-1]})')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Sample Size', fontweight='bold')

    ax.legend()


def create_conceptual_diagram(ax):
    """Create a conceptual diagram showing the data-to-performance pipeline."""

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Data Quality Impact on Model Performance',
           ha='center', fontsize=14, fontweight='bold')

    # Boxes
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                     edgecolor='black', linewidth=2)

    # Data Quality box
    ax.text(1.5, 4, 'DATA QUALITY', ha='center', va='center',
           fontsize=11, fontweight='bold', bbox=box_style)

    # Factors
    factors = [
        ('Sample Size', 1.5, 3, '#90EE90'),
        ('Class Balance', 1.5, 2.3, '#FFB6C1'),
        ('Label Quality', 1.5, 1.6, '#DDA0DD'),
    ]

    for text, x, y, color in factors:
        ax.add_patch(plt.Rectangle((x-0.8, y-0.25), 1.6, 0.5,
                                   facecolor=color, edgecolor='black'))
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    # Arrow
    ax.annotate('', xy=(4, 2.5), xytext=(2.5, 2.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='gray'))

    # Model Training box
    ax.text(5.5, 4, 'MODEL TRAINING', ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                    edgecolor='black', linewidth=2))

    ax.add_patch(plt.Rectangle((4.2, 1.5), 2.6, 2,
                               facecolor='lightyellow', edgecolor='black',
                               alpha=0.5, linestyle='--'))
    ax.text(5.5, 2.5, 'Learning\nDynamics', ha='center', va='center', fontsize=9)

    # Arrow
    ax.annotate('', xy=(8, 2.5), xytext=(7, 2.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='gray'))

    # Performance box
    ax.text(8.5, 4, 'PERFORMANCE', ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                    edgecolor='black', linewidth=2))

    metrics = [
        ('F1 Score', 8.5, 3),
        ('Precision', 8.5, 2.3),
        ('Recall', 8.5, 1.6),
    ]

    for text, x, y in metrics:
        ax.add_patch(plt.Rectangle((x-0.7, y-0.25), 1.4, 0.5,
                                   facecolor='lightgreen', edgecolor='black'))
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    # Key insights
    insights_box = dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='gray', linestyle='--')
    insights = (
        "Key Findings:\n"
        "1. F1 ~ log(samples) - diminishing returns\n"
        "2. Classes <50 samples: unreliable\n"
        "3. Balance affects consistency\n"
        "4. Quality > Quantity after threshold"
    )
    ax.text(5, 0.6, insights, ha='center', va='center', fontsize=9,
           bbox=insights_box, family='monospace')


def generate_report(csv_path: str, output_dir: str, jsonl_path: Optional[str] = None):
    """Generate comprehensive data quality vs performance report."""

    # Load training metrics
    df, class_map, n_classes = load_training_data(csv_path)
    epochs = df['epoch'].values

    class_names = [class_map.get(i, f'Class_{i}') for i in range(n_classes)]

    # Get sample counts
    if jsonl_path and Path(jsonl_path).exists():
        # From actual training data
        sample_counts = count_training_samples(jsonl_path)
        samples = np.array([
            sample_counts.get(f'consensus_themes_extra_large_{name.lower()}',
            sample_counts.get(name.lower(), 0))
            for name in class_names
        ])
        # Fallback to support columns if needed
        if samples.sum() == 0:
            samples = np.array([df[f'support_{i}'].iloc[-1] for i in range(n_classes)])
    else:
        # From validation support (approximate)
        samples = np.array([df[f'support_{i}'].iloc[-1] * 5 for i in range(n_classes)])  # Estimate training from val

    # Get final F1 scores
    f1_scores = np.array([df[f'f1_{i}'].iloc[-1] for i in range(n_classes)])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Figure 1: Main Analysis
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

    fig1.suptitle('Training Data Quality vs Model Performance Analysis',
                  fontsize=16, fontweight='bold', y=0.98)

    # Sample size vs F1 scatter
    ax1 = fig1.add_subplot(gs[0, 0])
    create_sample_f1_scatter(ax1, samples, f1_scores, class_names)

    # Imbalance impact
    ax2 = fig1.add_subplot(gs[0, 1])
    create_imbalance_impact(ax2, samples, f1_scores, class_names)

    # Threshold analysis
    ax3 = fig1.add_subplot(gs[1, 0])
    create_threshold_analysis(ax3, samples, f1_scores)

    # Learning efficiency
    ax4 = fig1.add_subplot(gs[1, 1])
    create_learning_efficiency(ax4, df, samples, n_classes, class_names)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(output_path / 'data_quality_performance_main.png',
                 dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path / 'data_quality_performance_main.png'}")

    # =========================================================================
    # Figure 2: Quality Indicators + Conceptual Diagram
    # =========================================================================
    fig2 = plt.figure(figsize=(14, 6))

    # Radar chart
    ax_radar = fig2.add_subplot(121, projection='polar')
    create_quality_indicators(ax_radar, samples, f1_scores, class_names)

    # Conceptual diagram
    ax_concept = fig2.add_subplot(122)
    create_conceptual_diagram(ax_concept)

    plt.tight_layout()
    fig2.savefig(output_path / 'data_quality_performance_overview.png',
                 dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path / 'data_quality_performance_overview.png'}")

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "="*70)
    print("DATA QUALITY vs PERFORMANCE SUMMARY")
    print("="*70)

    # Correlation
    valid_mask = (samples > 0) & (f1_scores > 0)
    if valid_mask.sum() >= 3:
        corr, p_value = stats.pearsonr(np.log1p(samples[valid_mask]), f1_scores[valid_mask])
        print(f"\nCorrelation (log(samples) vs F1): {corr:.3f} (p={p_value:.4f})")

    print(f"\nClass Distribution:")
    print(f"  Total samples: {samples.sum():,}")
    print(f"  Classes: {n_classes}")
    print(f"  Mean per class: {samples.mean():.0f}")
    print(f"  Std per class: {samples.std():.0f}")
    print(f"  Min: {samples.min():.0f} ({class_names[samples.argmin()]})")
    print(f"  Max: {samples.max():.0f} ({class_names[samples.argmax()]})")
    print(f"  Imbalance ratio: {samples.max() / samples.min():.1f}x")

    print(f"\nPerformance:")
    print(f"  Mean F1: {f1_scores.mean():.3f}")
    print(f"  Std F1: {f1_scores.std():.3f}")
    print(f"  Min F1: {f1_scores.min():.3f} ({class_names[f1_scores.argmin()]})")
    print(f"  Max F1: {f1_scores.max():.3f} ({class_names[f1_scores.argmax()]})")

    # Classes below thresholds
    print(f"\nClasses with F1 < 0.5:")
    for i, (name, f1, n) in enumerate(zip(class_names, f1_scores, samples)):
        if f1 < 0.5:
            print(f"  - {name}: F1={f1:.3f}, n={n:.0f}")

    print("="*70)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze relationship between training data quality and model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_data_quality_performance.py training.csv --output ./figures
  python plot_data_quality_performance.py training.csv --training-data data.jsonl -o ./output
        """
    )

    parser.add_argument('csv_path', type=str,
                        help='Path to training.csv file with per-class metrics')
    parser.add_argument('-o', '--output', type=str, default='paper/figures/data_quality',
                        help='Output directory for figures (default: paper/figures/data_quality)')
    parser.add_argument('--training-data', type=str, default=None,
                        help='Path to training JSONL file for accurate sample counts')

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        return 1

    generate_report(args.csv_path, args.output, args.training_data)
    return 0


if __name__ == '__main__':
    exit(main())

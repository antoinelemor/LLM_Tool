#!/usr/bin/env python3
"""
Per-Class Training Metrics Visualization
=========================================

Generates sophisticated scientific visualizations for multi-class classification
training metrics, showing per-class F1 scores evolution across epochs.

Features:
- Heatmap of F1 scores per class per epoch
- Individual class trajectories with confidence bands
- Class ranking evolution
- Correlation analysis between classes
- Per-language breakdown

Author: Antoine Lemor
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Scientific style configuration
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
    """Extract class names from the CSV header comment."""
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    if not first_line.startswith('# CLASS_LEGEND:'):
        return {}

    legend_str = first_line.replace('# CLASS_LEGEND:', '').strip()
    class_map = {}

    for item in legend_str.split(', '):
        if '=' in item:
            idx, name = item.split('=', 1)
            # Clean class name: remove prefix
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
    """Load training CSV and extract per-class metrics."""
    # Parse class legend
    class_map = parse_class_legend(csv_path)
    n_classes = len(class_map)

    # Load data (skip first line which is the legend)
    df = pd.read_csv(csv_path, skiprows=1)

    return df, class_map, n_classes


def extract_global_metrics(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract global metrics (macro_f1, micro_f1, accuracy) across epochs."""
    metrics = {}

    if 'macro_f1' in df.columns:
        metrics['macro_f1'] = df['macro_f1'].values
    if 'micro_f1' in df.columns:
        metrics['micro_f1'] = df['micro_f1'].values
    if 'accuracy' in df.columns:
        metrics['accuracy'] = df['accuracy'].values

    return metrics


def extract_per_class_f1(df: pd.DataFrame, n_classes: int, prefix: str = '') -> np.ndarray:
    """Extract F1 scores for each class across epochs.

    Note: F1 scores are validated to be in [0, 1] range. Invalid values
    (> 1 or < 0) are replaced with NaN to handle corrupted data.
    """
    epochs = len(df)
    f1_matrix = np.zeros((epochs, n_classes))

    for i in range(n_classes):
        col_name = f'{prefix}f1_{i}'
        if col_name in df.columns:
            values = df[col_name].values.copy()
            # F1 scores must be in [0, 1] range - mark invalid values as NaN
            invalid_mask = (values < 0) | (values > 1)
            if np.any(invalid_mask):
                warnings.warn(f"Column {col_name} contains {invalid_mask.sum()} invalid values (outside [0,1]). "
                             f"These will be treated as missing data.")
                values[invalid_mask] = np.nan
            f1_matrix[:, i] = values

    return f1_matrix


def create_heatmap(ax, f1_matrix: np.ndarray, class_names: List[str],
                   epochs: np.ndarray, title: str = 'F1 Score Evolution by Class'):
    """Create a sophisticated heatmap of F1 scores."""
    # Custom colormap: white -> light blue -> dark blue
    colors = ['#ffffff', '#e6f2ff', '#99ccff', '#4da6ff', '#0066cc', '#003366']
    cmap = mcolors.LinearSegmentedColormap.from_list('f1_cmap', colors, N=256)

    im = ax.imshow(f1_matrix.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    # Axis configuration
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Class', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('F1 Score', fontweight='bold')

    # Add text annotations for final epoch
    final_epoch_idx = len(epochs) - 1
    for i, class_name in enumerate(class_names):
        val = f1_matrix[final_epoch_idx, i]
        color = 'white' if val > 0.5 else 'black'
        ax.text(final_epoch_idx, i, f'{val:.2f}', ha='center', va='center',
                color=color, fontsize=7, fontweight='bold')

    return im


def create_trajectory_plot(ax, f1_matrix: np.ndarray, class_names: List[str],
                          epochs: np.ndarray, top_n: Optional[int] = None):
    """Create line plot showing F1 trajectory for all or top N classes.

    Args:
        top_n: Number of top classes to show. If None, show all classes.
    """
    # Get final F1 scores and sort
    final_f1 = f1_matrix[-1, :]
    sorted_indices = np.argsort(final_f1)[::-1]

    # If top_n is None, show all classes
    n_to_show = len(class_names) if top_n is None else min(top_n, len(class_names))

    # Color palette - use tab20 for up to 20, tab20b+tab20c for more
    if n_to_show <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_to_show))
    else:
        # Combine multiple colormaps for more than 20 classes
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.tab20b(np.linspace(0, 1, min(n_to_show - 20, 20)))
        colors = np.vstack([colors1, colors2[:n_to_show - 20]])

    for rank, idx in enumerate(sorted_indices[:n_to_show]):
        ax.plot(epochs, f1_matrix[:, idx], '-o', color=colors[rank],
                linewidth=2, markersize=4, label=f'{class_names[idx]} ({final_f1[idx]:.2f})')

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    title = 'All Classes F1 Score Evolution' if top_n is None else f'Top {top_n} Classes by Final F1 Score'
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True,
              fancybox=True, shadow=True, fontsize=8)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(epochs)


def create_ranking_evolution(ax, f1_matrix: np.ndarray, class_names: List[str],
                             epochs: np.ndarray):
    """Create bump chart showing class ranking evolution."""
    n_epochs, n_classes = f1_matrix.shape

    # Calculate rankings for each epoch (1 = best)
    rankings = np.zeros_like(f1_matrix)
    for e in range(n_epochs):
        rankings[e, :] = n_classes - np.argsort(np.argsort(f1_matrix[e, :]))

    # Color by final ranking
    final_rankings = rankings[-1, :]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_classes))
    color_map = {i: colors[int(final_rankings[i]) - 1] for i in range(n_classes)}

    for i in range(n_classes):
        ax.plot(epochs, rankings[:, i], '-', color=color_map[i],
                linewidth=1.5, alpha=0.7)
        # Add class label at the end
        ax.text(epochs[-1] + 0.3, rankings[-1, i], class_names[i],
                fontsize=7, va='center', ha='left', color=color_map[i])

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Rank (1 = Best)', fontweight='bold')
    ax.set_title('Class Ranking Evolution', fontweight='bold', pad=10)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 3)
    ax.set_ylim(0.5, n_classes + 0.5)
    ax.invert_yaxis()
    ax.set_xticks(epochs)


def create_final_distribution(ax, f1_matrix: np.ndarray, class_names: List[str],
                              support_values: Optional[np.ndarray] = None):
    """Create horizontal bar chart of final F1 scores with support."""
    final_f1 = f1_matrix[-1, :]
    sorted_indices = np.argsort(final_f1)

    # Color gradient based on F1 score
    colors = plt.cm.RdYlGn(final_f1[sorted_indices])

    y_pos = np.arange(len(class_names))
    bars = ax.barh(y_pos, final_f1[sorted_indices], color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([class_names[i] for i in sorted_indices])
    ax.set_xlabel('F1 Score', fontweight='bold')
    ax.set_title('Final F1 Score by Class (Sorted)', fontweight='bold', pad=10)
    ax.set_xlim(0, 1.1)

    # Add value labels
    for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', ha='left', fontsize=8)

    # Add mean line
    mean_f1 = np.mean(final_f1)
    ax.axvline(x=mean_f1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_f1:.3f}')
    ax.legend(loc='lower right')


def create_correlation_heatmap(ax, f1_matrix: np.ndarray, class_names: List[str]):
    """Create correlation heatmap between class performances."""
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(f1_matrix.T)

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Plot
    sns.heatmap(corr_matrix, mask=mask, ax=ax, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

    ax.set_title('Class Performance Correlation', fontweight='bold', pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)


def create_learning_phases(ax, f1_matrix: np.ndarray, class_names: List[str],
                          epochs: np.ndarray):
    """Analyze and visualize learning phases for each class."""
    n_epochs, n_classes = f1_matrix.shape

    # Calculate rate of change (derivative)
    derivatives = np.diff(f1_matrix, axis=0)

    # Define phases: early (high growth), middle (steady), late (plateau/decay)
    early_phase = derivatives[:n_epochs//3, :].mean(axis=0)
    mid_phase = derivatives[n_epochs//3:2*n_epochs//3, :].mean(axis=0)
    late_phase = derivatives[2*n_epochs//3:, :].mean(axis=0)

    # Sort by early phase learning speed
    sorted_indices = np.argsort(early_phase)[::-1]

    x = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, early_phase[sorted_indices], width, label='Early (Epochs 1-6)', color='#2ecc71')
    ax.bar(x, mid_phase[sorted_indices], width, label='Middle (Epochs 7-13)', color='#3498db')
    ax.bar(x + width, late_phase[sorted_indices], width, label='Late (Epochs 14-19)', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in sorted_indices], rotation=45, ha='right')
    ax.set_ylabel('Avg. F1 Change per Epoch', fontweight='bold')
    ax.set_title('Learning Rate by Phase', fontweight='bold', pad=10)
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def create_global_metrics_plot(ax, df: pd.DataFrame, epochs: np.ndarray):
    """Create plot showing macro F1, micro F1, and accuracy evolution."""
    global_metrics = extract_global_metrics(df)

    if 'macro_f1' in global_metrics:
        ax.plot(epochs, global_metrics['macro_f1'], 'o-', color='#e74c3c',
                linewidth=2.5, markersize=6, label=f"Macro F1 (final: {global_metrics['macro_f1'][-1]:.3f})")

    if 'micro_f1' in global_metrics:
        ax.plot(epochs, global_metrics['micro_f1'], 's-', color='#3498db',
                linewidth=2.5, markersize=6, label=f"Micro F1 (final: {global_metrics['micro_f1'][-1]:.3f})")

    if 'accuracy' in global_metrics:
        ax.plot(epochs, global_metrics['accuracy'], '^-', color='#2ecc71',
                linewidth=2, markersize=5, alpha=0.8, label=f"Accuracy (final: {global_metrics['accuracy'][-1]:.3f})")

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Global Metrics Evolution (Macro F1 vs Micro F1)', fontweight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)


def generate_full_report(csv_path: str, output_dir: str, prefix: str = ''):
    """Generate comprehensive per-class visualization report."""
    # Load data
    df, class_map, n_classes = load_training_data(csv_path)
    epochs = df['epoch'].values

    # Get class names in order
    class_names = [class_map.get(i, f'Class_{i}') for i in range(n_classes)]

    # Extract F1 matrices
    f1_overall = extract_per_class_f1(df, n_classes, prefix='')
    f1_en = extract_per_class_f1(df, n_classes, prefix='EN_')
    f1_fr = extract_per_class_f1(df, n_classes, prefix='FR_')

    # Extract global metrics
    global_metrics = extract_global_metrics(df)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Figure 1: Main Heatmap + Trajectories
    # =========================================================================
    fig1 = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig1, height_ratios=[1.2, 1], width_ratios=[1.5, 1],
                  hspace=0.3, wspace=0.3)

    fig1.suptitle(f'Per-Class F1 Score Analysis\nMulti-Class Classification ({n_classes} Classes)',
                  fontsize=16, fontweight='bold', y=0.98)

    # Heatmap
    ax1 = fig1.add_subplot(gs[0, 0])
    create_heatmap(ax1, f1_overall, class_names, epochs,
                   title='F1 Score Heatmap by Class and Epoch')

    # Trajectory plot (show ALL classes)
    ax2 = fig1.add_subplot(gs[0, 1])
    create_trajectory_plot(ax2, f1_overall, class_names, epochs, top_n=None)

    # Final distribution
    ax3 = fig1.add_subplot(gs[1, 0])
    create_final_distribution(ax3, f1_overall, class_names)

    # Learning phases
    ax4 = fig1.add_subplot(gs[1, 1])
    create_learning_phases(ax4, f1_overall, class_names, epochs)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(output_path / 'per_class_analysis_main.png', dpi=200, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"Saved: {output_path / 'per_class_analysis_main.png'}")

    # =========================================================================
    # Figure 2: Ranking Evolution + Correlation
    # =========================================================================
    fig2, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig2.suptitle('Class Dynamics Analysis', fontsize=14, fontweight='bold', y=1.02)

    create_ranking_evolution(axes[0], f1_overall, class_names, epochs)
    create_correlation_heatmap(axes[1], f1_overall, class_names)

    plt.tight_layout()
    fig2.savefig(output_path / 'per_class_analysis_dynamics.png', dpi=200, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"Saved: {output_path / 'per_class_analysis_dynamics.png'}")

    # =========================================================================
    # Figure 2b: Global Metrics (Macro F1 vs Micro F1)
    # =========================================================================
    if global_metrics:
        fig2b, ax_global = plt.subplots(1, 1, figsize=(10, 6))
        fig2b.suptitle('Global Performance Metrics', fontsize=14, fontweight='bold', y=1.02)

        create_global_metrics_plot(ax_global, df, epochs)

        plt.tight_layout()
        fig2b.savefig(output_path / 'global_metrics_evolution.png', dpi=200, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
        print(f"Saved: {output_path / 'global_metrics_evolution.png'}")

    # =========================================================================
    # Figure 3: Per-Language Comparison
    # =========================================================================
    fig3 = plt.figure(figsize=(18, 8))
    gs3 = GridSpec(1, 3, figure=fig3, wspace=0.25)

    fig3.suptitle('Per-Language F1 Score Comparison by Class',
                  fontsize=14, fontweight='bold', y=1.02)

    ax_en = fig3.add_subplot(gs3[0, 0])
    create_heatmap(ax_en, f1_en, class_names, epochs, title='English (EN)')

    ax_fr = fig3.add_subplot(gs3[0, 1])
    create_heatmap(ax_fr, f1_fr, class_names, epochs, title='French (FR)')

    # Difference heatmap
    ax_diff = fig3.add_subplot(gs3[0, 2])
    diff_matrix = f1_en - f1_fr

    # Custom diverging colormap
    im = ax_diff.imshow(diff_matrix.T, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3)
    ax_diff.set_xticks(np.arange(len(epochs)))
    ax_diff.set_xticklabels(epochs)
    ax_diff.set_yticks(np.arange(len(class_names)))
    ax_diff.set_yticklabels(class_names)
    ax_diff.set_xlabel('Epoch', fontweight='bold')
    ax_diff.set_ylabel('Class', fontweight='bold')
    ax_diff.set_title('Difference (EN - FR)', fontweight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax_diff, shrink=0.8, pad=0.02)
    cbar.set_label('F1 Difference', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(output_path / 'per_class_analysis_languages.png', dpi=200, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"Saved: {output_path / 'per_class_analysis_languages.png'}")

    # =========================================================================
    # Figure 4: Individual Class Deep Dive (Grid)
    # =========================================================================
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    fig4.suptitle('Individual Class F1 Trajectories', fontsize=14, fontweight='bold', y=1.01)

    # Sort by final F1
    final_f1 = f1_overall[-1, :]
    sorted_indices = np.argsort(final_f1)[::-1]

    for plot_idx, class_idx in enumerate(sorted_indices):
        ax = axes[plot_idx]

        # Plot overall, EN, FR
        ax.fill_between(epochs, 0, f1_overall[:, class_idx], alpha=0.3, color='steelblue')
        ax.plot(epochs, f1_overall[:, class_idx], 'o-', color='steelblue',
                linewidth=2, markersize=4, label='Overall')
        ax.plot(epochs, f1_en[:, class_idx], '--', color='green',
                linewidth=1.5, alpha=0.8, label='EN')
        ax.plot(epochs, f1_fr[:, class_idx], '--', color='red',
                linewidth=1.5, alpha=0.8, label='FR')

        # Title with final F1
        ax.set_title(f'{class_names[class_idx]}\nF1={final_f1[class_idx]:.3f}',
                    fontsize=10, fontweight='bold')

        if plot_idx == 0:
            ax.legend(loc='lower right', fontsize=7)

        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)

    # Common labels
    fig4.text(0.5, 0.02, 'Epoch', ha='center', fontweight='bold', fontsize=12)
    fig4.text(0.02, 0.5, 'F1 Score', va='center', rotation='vertical', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])
    fig4.savefig(output_path / 'per_class_analysis_individual.png', dpi=200, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"Saved: {output_path / 'per_class_analysis_individual.png'}")

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "="*70)
    print("PER-CLASS STATISTICS SUMMARY")
    print("="*70)

    # Global metrics summary
    if global_metrics:
        print("\n📊 GLOBAL METRICS (Final Epoch):")
        print("-"*40)
        if 'macro_f1' in global_metrics:
            print(f"  Macro F1:  {global_metrics['macro_f1'][-1]:.4f}")
        if 'micro_f1' in global_metrics:
            print(f"  Micro F1:  {global_metrics['micro_f1'][-1]:.4f}")
        if 'accuracy' in global_metrics:
            print(f"  Accuracy:  {global_metrics['accuracy'][-1]:.4f}")
        print("-"*40)

    final_f1_sorted = [(class_names[i], final_f1[i]) for i in sorted_indices]

    print(f"\n{'Class':<25} {'Final F1':>10} {'Best F1':>10} {'Best Epoch':>12}")
    print("-"*60)

    for class_idx in sorted_indices:
        name = class_names[class_idx]
        final = f1_overall[-1, class_idx]
        best = f1_overall[:, class_idx].max()
        best_epoch = epochs[f1_overall[:, class_idx].argmax()]
        print(f"{name:<25} {final:>10.4f} {best:>10.4f} {best_epoch:>12}")

    print("-"*60)
    print(f"{'MEAN (per-class)':<25} {final_f1.mean():>10.4f}")
    print(f"{'STD':<25} {final_f1.std():>10.4f}")
    print(f"{'MIN':<25} {final_f1.min():>10.4f}")
    print(f"{'MAX':<25} {final_f1.max():>10.4f}")
    print("="*70)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-class training metrics visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_per_class_metrics.py training.csv --output ./figures
  python plot_per_class_metrics.py training.csv -o ./output --prefix experiment1_
        """
    )

    parser.add_argument('csv_path', type=str,
                        help='Path to training.csv file')
    parser.add_argument('-o', '--output', type=str, default='paper/figures/per_class_metrics',
                        help='Output directory for figures (default: paper/figures/per_class_metrics)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output filenames')

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        return 1

    generate_full_report(args.csv_path, args.output, args.prefix)
    return 0


if __name__ == '__main__':
    exit(main())

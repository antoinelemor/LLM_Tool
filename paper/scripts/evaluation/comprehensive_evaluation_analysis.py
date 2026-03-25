#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
comprehensive_evaluation_analysis

MAIN OBJECTIVE:
---------------
In-depth analysis of model performance with respect to inter-annotator agreement.
Stratifies evaluation by annotation ambiguity and produces consolidated reports
with publication-quality visualizations.

Dependencies:
-------------
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pathlib
- json

MAIN FEATURES:
--------------
1) Load predictions and ground truth annotations
2) Stratify by inter-annotator agreement levels
3) Calculate per-theme and aggregate metrics
4) Generate publication-quality figures (PNG/PDF)
5) Statistical analysis with significance tests
6) Export consolidated CSV reports

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import json
import ast
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================

ANNOTATORS = ['shdin', 'Jeremy', 'jdrouin', 'BenjaminCarignan']
ANNOTATOR_DISPLAY = {
    'shdin': 'Annotator 1',
    'Jeremy': 'Annotator 2',
    'jdrouin': 'Annotator 3',
    'BenjaminCarignan': 'Annotator 4'
}

MODELS_LARGE = [
    'gpt_4_1_large', 'gpt_5_large', 'gpt_oss_large',
    'gemma_large', 'llama_large', 'nemotron_large'
]

MODELS_SMALL = [
    'gpt_4_1_small', 'gpt_5_small', 'gpt_oss_small',
    'gemma_small', 'llama_small', 'nemotron_small'
]

MODELS_ALL = MODELS_LARGE + MODELS_SMALL + ['manual_manual']

# For backward compatibility
MODELS = MODELS_LARGE + ['manual_manual']

MODEL_DISPLAY = {
    # Large models
    'gpt_4_1_large': 'GPT-4.1 (L)',
    'gpt_5_large': 'GPT-5 (L)',
    'gpt_oss_large': 'GPT-OSS (L)',
    'gemma_large': 'Gemma (L)',
    'llama_large': 'Llama (L)',
    'nemotron_large': 'Nemotron (L)',
    # Small models
    'gpt_4_1_small': 'GPT-4.1 (S)',
    'gpt_5_small': 'GPT-5 (S)',
    'gpt_oss_small': 'GPT-OSS (S)',
    'gemma_small': 'Gemma (S)',
    'llama_small': 'Llama (S)',
    'nemotron_small': 'Nemotron (S)',
    # Manual
    'manual_manual': 'Manual'
}

MODEL_COLORS = {
    # Large models (darker shades)
    'GPT-4.1 (L)': '#1f77b4',
    'GPT-5 (L)': '#2ca02c',
    'GPT-OSS (L)': '#17becf',
    'Gemma (L)': '#ff7f0e',
    'Llama (L)': '#d62728',
    'Nemotron (L)': '#9467bd',
    # Small models (lighter shades)
    'GPT-4.1 (S)': '#aec7e8',
    'GPT-5 (S)': '#98df8a',
    'GPT-OSS (S)': '#9edae5',
    'Gemma (S)': '#ffbb78',
    'Llama (S)': '#ff9896',
    'Nemotron (S)': '#c5b0d5',
    # Manual
    'Manual': '#8c564b'
}

ALL_THEMES = [
    'theme_agriculture', 'theme_culture_nationalism', 'theme_defense',
    'theme_domestic_commerce', 'theme_education', 'theme_energy',
    'theme_environment', 'theme_foreign_trade', 'theme_governments_governance',
    'theme_health', 'theme_housing', 'theme_immigration', 'theme_indigenous_affairs',
    'theme_international_affairs', 'theme_labor', 'theme_law_and_crime',
    'theme_macroeconomics', 'theme_public_lands',
    'theme_rights_liberties_minorities_discrimination', 'theme_social_welfare',
    'theme_technology', 'theme_transportation'
]

THEME_DISPLAY = {t: t.replace('theme_', '').replace('_', ' ').title() for t in ALL_THEMES}

# Agreement level thresholds and colors
AGREEMENT_LEVELS = {
    'Almost Perfect': {'min': 0.81, 'max': 1.01, 'color': '#2ecc71', 'marker': 'o'},
    'Substantial': {'min': 0.61, 'max': 0.81, 'color': '#3498db', 'marker': 's'},
    'Moderate': {'min': 0.41, 'max': 0.61, 'color': '#f39c12', 'marker': '^'},
    'Fair': {'min': 0.21, 'max': 0.41, 'color': '#e74c3c', 'marker': 'D'},
    'Slight': {'min': 0.00, 'max': 0.21, 'color': '#95a5a6', 'marker': 'v'}
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_parse(val):
    """Parse JSON/list from string safely."""
    if pd.isna(val) or val == '' or val == '[]':
        return set()
    try:
        if isinstance(val, set):
            return val
        if isinstance(val, list):
            return set(val)
        return set(ast.literal_eval(val))
    except:
        try:
            return set(json.loads(val))
        except:
            return set()


def fleiss_kappa(ratings_matrix):
    """Compute Fleiss' Kappa for multiple annotators."""
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = int(ratings_matrix.sum(axis=1).mean())

    if n_raters <= 1:
        return np.nan

    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_e >= 1:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def get_agreement_level(kappa):
    """Get agreement level label from kappa value."""
    for level, params in AGREEMENT_LEVELS.items():
        if params['min'] <= kappa < params['max']:
            return level
    return 'Slight'


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return np.nan, np.nan

    bootstrapped = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = bootstrapped.mean(axis=1)

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)

    return lower, upper


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(base_path):
    """Load all required data files."""
    data_path = base_path / 'data' / 'sets'

    annotators_df_full = pd.read_csv(data_path / 'benchmark_annotations_by_annotator.csv')
    annotated_df = pd.read_csv(data_path / 'final_test_small_annotated.csv')

    # Filter to only include IDs in the test set (250 samples)
    test_ids = set(annotated_df['id'].tolist())
    annotators_df = annotators_df_full[annotators_df_full['id'].isin(test_ids)].copy()

    id_to_annotators = {row['id']: row for _, row in annotators_df.iterrows()}

    return annotators_df, annotated_df, id_to_annotators


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def compute_theme_statistics(annotators_df):
    """Compute comprehensive statistics for each theme."""
    results = []

    for theme in ALL_THEMES:
        theme_name = theme.replace('theme_', '')

        # Build ratings matrix for Fleiss' Kappa
        ratings = []
        annotator_counts = {ann: 0 for ann in ANNOTATORS}

        for _, row in annotators_df.iterrows():
            present = 0
            for ann in ANNOTATORS:
                if theme in safe_parse(row[f'{ann}_themes']):
                    present += 1
                    annotator_counts[ann] += 1
            absent = len(ANNOTATORS) - present
            ratings.append([absent, present])

        kappa = fleiss_kappa(np.array(ratings))

        # Support in consensus
        support_consensus = sum(
            1 for _, row in annotators_df.iterrows()
            if theme in safe_parse(row['consensus_themes'])
        )

        # Prevalence (% of samples with this theme in consensus)
        prevalence = support_consensus / len(annotators_df) * 100

        # Annotator agreement rate (% where all agree)
        full_agreement = sum(
            1 for _, row in annotators_df.iterrows()
            if len(set(1 if theme in safe_parse(row[f'{ann}_themes']) else 0 for ann in ANNOTATORS)) == 1
        )
        agreement_rate = full_agreement / len(annotators_df) * 100

        # Standard deviation of annotator counts
        count_std = np.std(list(annotator_counts.values()))

        results.append({
            'theme': theme,
            'theme_display': THEME_DISPLAY[theme],
            'fleiss_kappa': kappa,
            'agreement_level': get_agreement_level(kappa),
            'support_consensus': support_consensus,
            'prevalence_pct': prevalence,
            'agreement_rate_pct': agreement_rate,
            'annotator_std': count_std,
            **{f'support_{ann}': annotator_counts[ann] for ann in ANNOTATORS}
        })

    return pd.DataFrame(results).sort_values('fleiss_kappa', ascending=False)


def compute_model_performance(annotated_df, id_to_annotators, theme_stats_df):
    """Compute detailed model performance metrics."""
    # Create theme to kappa mapping
    theme_kappa = dict(zip(theme_stats_df['theme'], theme_stats_df['fleiss_kappa']))
    theme_level = dict(zip(theme_stats_df['theme'], theme_stats_df['agreement_level']))

    results = []

    for model in MODELS:
        model_name = MODEL_DISPLAY[model]

        # Per-theme metrics
        theme_metrics = {}

        for theme in ALL_THEMES:
            tp, fp, fn = 0, 0, 0

            for _, row in annotated_df.iterrows():
                row_id = row['id']
                if row_id not in id_to_annotators:
                    continue

                ann_row = id_to_annotators[row_id]
                all_ann = json.loads(row['all_annotations'])
                pipelines = all_ann.get('pipelines', {})

                consensus = safe_parse(ann_row['consensus_themes'])
                pred = set(pipelines.get(model, {}).get('themes', []))

                pred_has = theme in pred
                true_has = theme in consensus

                if pred_has and true_has:
                    tp += 1
                elif pred_has and not true_has:
                    fp += 1
                elif not pred_has and true_has:
                    fn += 1

            precision, recall, f1 = compute_metrics(tp, fp, fn)

            theme_metrics[theme] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': tp + fn,
                'kappa': theme_kappa.get(theme, np.nan),
                'level': theme_level.get(theme, 'Unknown')
            }

        # Aggregate by agreement level
        for level in AGREEMENT_LEVELS.keys():
            level_themes = [t for t in ALL_THEMES if theme_level.get(t) == level]

            if not level_themes:
                continue

            total_tp = sum(theme_metrics[t]['tp'] for t in level_themes)
            total_fp = sum(theme_metrics[t]['fp'] for t in level_themes)
            total_fn = sum(theme_metrics[t]['fn'] for t in level_themes)

            precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

            results.append({
                'model': model_name,
                'category': 'By Agreement Level',
                'subcategory': level,
                'n_themes': len(level_themes),
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'support': total_tp + total_fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Overall
        total_tp = sum(m['tp'] for m in theme_metrics.values())
        total_fp = sum(m['fp'] for m in theme_metrics.values())
        total_fn = sum(m['fn'] for m in theme_metrics.values())

        precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

        results.append({
            'model': model_name,
            'category': 'Overall',
            'subcategory': 'All Themes',
            'n_themes': len(ALL_THEMES),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'support': total_tp + total_fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # Per-theme results
        for theme, metrics in theme_metrics.items():
            results.append({
                'model': model_name,
                'category': 'Per Theme',
                'subcategory': THEME_DISPLAY[theme],
                'n_themes': 1,
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'support': metrics['support'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'kappa': metrics['kappa'],
                'agreement_level': metrics['level']
            })

    return pd.DataFrame(results)


def compute_human_baseline(annotators_df, theme_stats_df):
    """Compute human baseline (each annotator vs consensus)."""
    theme_level = dict(zip(theme_stats_df['theme'], theme_stats_df['agreement_level']))

    results = []

    for annotator in ANNOTATORS:
        ann_name = ANNOTATOR_DISPLAY[annotator]

        # Per agreement level
        for level in AGREEMENT_LEVELS.keys():
            level_themes = [t for t in ALL_THEMES if theme_level.get(t) == level]

            if not level_themes:
                continue

            tp, fp, fn = 0, 0, 0

            for _, row in annotators_df.iterrows():
                consensus = safe_parse(row['consensus_themes'])
                ann_themes = safe_parse(row[f'{annotator}_themes'])

                for theme in level_themes:
                    pred_has = theme in ann_themes
                    true_has = theme in consensus

                    if pred_has and true_has:
                        tp += 1
                    elif pred_has and not true_has:
                        fp += 1
                    elif not pred_has and true_has:
                        fn += 1

            precision, recall, f1 = compute_metrics(tp, fp, fn)

            results.append({
                'annotator': ann_name,
                'category': 'By Agreement Level',
                'subcategory': level,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'support': tp + fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Overall
        tp, fp, fn = 0, 0, 0

        for _, row in annotators_df.iterrows():
            consensus = safe_parse(row['consensus_themes'])
            ann_themes = safe_parse(row[f'{annotator}_themes'])

            tp += len(ann_themes & consensus)
            fp += len(ann_themes - consensus)
            fn += len(consensus - ann_themes)

        precision, recall, f1 = compute_metrics(tp, fp, fn)

        results.append({
            'annotator': ann_name,
            'category': 'Overall',
            'subcategory': 'All Themes',
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': tp + fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return pd.DataFrame(results)


def compute_correlation_analysis(model_perf_df, theme_stats_df):
    """Compute correlation between model performance and inter-annotator agreement."""
    results = []

    # Get per-theme data
    per_theme = model_perf_df[model_perf_df['category'] == 'Per Theme'].copy()

    for model in per_theme['model'].unique():
        model_data = per_theme[per_theme['model'] == model]

        # Merge with theme stats
        merged = model_data.merge(
            theme_stats_df[['theme_display', 'fleiss_kappa', 'support_consensus', 'prevalence_pct']],
            left_on='subcategory',
            right_on='theme_display'
        )

        # Filter out themes with no support
        merged = merged[merged['support'] > 0]

        if len(merged) < 3:
            continue

        # Pearson correlation: F1 vs Kappa
        corr_kappa, p_kappa = stats.pearsonr(merged['f1'], merged['fleiss_kappa'])

        # Spearman correlation: F1 vs Kappa
        spearman_kappa, sp_p_kappa = stats.spearmanr(merged['f1'], merged['fleiss_kappa'])

        # Correlation: F1 vs Support
        corr_support, p_support = stats.pearsonr(merged['f1'], merged['support'])

        results.append({
            'model': model,
            'pearson_f1_vs_kappa': corr_kappa,
            'pearson_p_value': p_kappa,
            'spearman_f1_vs_kappa': spearman_kappa,
            'spearman_p_value': sp_p_kappa,
            'pearson_f1_vs_support': corr_support,
            'support_p_value': p_support,
            'n_themes': len(merged)
        })

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_figure_1_agreement_distribution(theme_stats_df, output_path):
    """Figure 1: Distribution of themes by inter-annotator agreement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1a: Bar chart of themes by agreement level
    ax1 = axes[0]

    level_order = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Slight']
    level_counts = theme_stats_df['agreement_level'].value_counts().reindex(level_order).fillna(0)

    colors = [AGREEMENT_LEVELS[l]['color'] for l in level_order]
    bars = ax1.bar(range(len(level_order)), level_counts.values, color=colors, edgecolor='black', linewidth=0.5)

    ax1.set_xticks(range(len(level_order)))
    ax1.set_xticklabels([l.replace(' ', '\n') for l in level_order], fontsize=10)
    ax1.set_ylabel('Number of Themes', fontsize=12)
    ax1.set_xlabel('Agreement Level', fontsize=12)
    ax1.set_title('(a) Theme Distribution by Agreement Level', fontsize=13, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars, level_counts.values):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylim(0, max(level_counts.values) * 1.15)

    # 1b: Scatter plot of kappa vs support
    ax2 = axes[1]

    for level in level_order:
        level_data = theme_stats_df[theme_stats_df['agreement_level'] == level]
        ax2.scatter(
            level_data['fleiss_kappa'],
            level_data['support_consensus'],
            c=AGREEMENT_LEVELS[level]['color'],
            marker=AGREEMENT_LEVELS[level]['marker'],
            s=100,
            label=level,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.8
        )

    # Add theme labels for notable points
    for _, row in theme_stats_df.iterrows():
        if row['support_consensus'] > 100 or row['fleiss_kappa'] < 0.4:
            ax2.annotate(
                row['theme_display'].replace(' ', '\n'),
                (row['fleiss_kappa'], row['support_consensus']),
                fontsize=7,
                ha='center',
                va='bottom',
                xytext=(0, 5),
                textcoords='offset points'
            )

    ax2.set_xlabel("Fleiss' Kappa (κ)", fontsize=12)
    ax2.set_ylabel('Support (N in Consensus)', fontsize=12)
    ax2.set_title('(b) Agreement vs. Theme Frequency', fontsize=13, fontweight='bold')
    ax2.legend(title='Agreement Level', loc='upper left', fontsize=9)

    # Add vertical lines for thresholds
    for level, params in AGREEMENT_LEVELS.items():
        if params['min'] > 0:
            ax2.axvline(x=params['min'], color=params['color'], linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path / 'fig1_agreement_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig1_agreement_distribution.pdf', bbox_inches='tight')
    plt.close()


def create_figure_2_model_performance(model_perf_df, human_baseline_df, output_path):
    """Figure 2: Model performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2a: Overall F1 comparison
    ax1 = axes[0]

    # Get overall metrics
    overall_models = model_perf_df[
        (model_perf_df['category'] == 'Overall') &
        (model_perf_df['subcategory'] == 'All Themes')
    ].sort_values('f1', ascending=True)

    overall_human = human_baseline_df[
        (human_baseline_df['category'] == 'Overall') &
        (human_baseline_df['subcategory'] == 'All Themes')
    ]['f1'].mean()

    # Plot
    y_pos = range(len(overall_models))
    colors = [MODEL_COLORS.get(m, '#333333') for m in overall_models['model']]

    bars = ax1.barh(y_pos, overall_models['f1'], color=colors, edgecolor='black', linewidth=0.5)

    # Add human baseline line
    ax1.axvline(x=overall_human, color='red', linestyle='--', linewidth=2, label=f'Human Baseline ({overall_human:.3f})')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(overall_models['model'], fontsize=11)
    ax1.set_xlabel('Micro-F1 Score', fontsize=12)
    ax1.set_title('(a) Overall Model Performance', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)

    # Add value labels
    for bar, f1 in zip(bars, overall_models['f1']):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{f1:.3f}', va='center', fontsize=10)

    ax1.set_xlim(0, max(overall_models['f1'].max(), overall_human) * 1.15)

    # 2b: Performance by agreement level
    ax2 = axes[1]

    level_order = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair']

    # Prepare data
    by_level = model_perf_df[model_perf_df['category'] == 'By Agreement Level'].copy()
    by_level = by_level[by_level['subcategory'].isin(level_order)]

    human_by_level = human_baseline_df[human_baseline_df['category'] == 'By Agreement Level'].copy()
    human_by_level = human_by_level[human_by_level['subcategory'].isin(level_order)]
    human_avg = human_by_level.groupby('subcategory')['f1'].mean()

    # Plot grouped bars
    x = np.arange(len(level_order))
    width = 0.1
    n_models = len(MODELS)

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_data = by_level[by_level['model'] == model]
        values = [model_data[model_data['subcategory'] == l]['f1'].values[0]
                  if len(model_data[model_data['subcategory'] == l]) > 0 else 0
                  for l in level_order]

        offset = (i - n_models/2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=model, color=MODEL_COLORS.get(model, '#333333'),
                edgecolor='black', linewidth=0.3)

    # Add human baseline
    human_values = [human_avg.get(l, 0) for l in level_order]
    ax2.plot(x, human_values, 'r--', linewidth=2, marker='o', markersize=8, label='Human Baseline')

    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(' ', '\n') for l in level_order], fontsize=10)
    ax2.set_ylabel('Micro-F1 Score', fontsize=12)
    ax2.set_xlabel('Agreement Level', fontsize=12)
    ax2.set_title('(b) Performance by Agreement Level', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path / 'fig2_model_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig2_model_performance.pdf', bbox_inches='tight')
    plt.close()


def create_figure_3_f1_vs_kappa(model_perf_df, theme_stats_df, output_path):
    """Figure 3: Scatter plot of F1 vs Kappa for all models."""
    fig, ax = plt.subplots(figsize=(12, 8))

    per_theme = model_perf_df[model_perf_df['category'] == 'Per Theme'].copy()

    # Merge with theme stats
    merged = per_theme.merge(
        theme_stats_df[['theme_display', 'fleiss_kappa', 'support_consensus']],
        left_on='subcategory',
        right_on='theme_display'
    )

    # Filter themes with support
    merged = merged[merged['support'] > 0]

    # Plot each model
    for model in MODEL_DISPLAY.values():
        model_data = merged[merged['model'] == model]
        ax.scatter(
            model_data['fleiss_kappa'],
            model_data['f1'],
            c=MODEL_COLORS.get(model, '#333333'),
            label=model,
            alpha=0.7,
            s=model_data['support'] * 2 + 20,
            edgecolors='black',
            linewidth=0.3
        )

    # Add trend line (average across models)
    avg_by_kappa = merged.groupby('fleiss_kappa')['f1'].mean().reset_index()
    z = np.polyfit(merged['fleiss_kappa'], merged['f1'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['fleiss_kappa'].min(), merged['fleiss_kappa'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7, label=f'Trend (slope={z[0]:.3f})')

    # Add correlation annotation
    corr, pval = stats.pearsonr(merged['fleiss_kappa'], merged['f1'])
    ax.annotate(
        f'Pearson r = {corr:.3f}\np < {pval:.3f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Add agreement level bands
    for level, params in AGREEMENT_LEVELS.items():
        if params['min'] < 1:
            ax.axvspan(params['min'], min(params['max'], 1.0),
                      alpha=0.1, color=params['color'], label=f'_{level}')

    ax.set_xlabel("Inter-Annotator Agreement (Fleiss' κ)", fontsize=12)
    ax.set_ylabel('Model F1 Score', fontsize=12)
    ax.set_title('Model Performance vs. Inter-Annotator Agreement\n(point size ∝ theme support)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(-0.02, max(merged['f1'].max() * 1.1, 0.5))

    plt.tight_layout()
    plt.savefig(output_path / 'fig3_f1_vs_kappa.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig3_f1_vs_kappa.pdf', bbox_inches='tight')
    plt.close()


def create_figure_4_heatmap(model_perf_df, theme_stats_df, output_path):
    """Figure 4: Heatmap of F1 scores by model and theme."""
    fig, ax = plt.subplots(figsize=(14, 10))

    per_theme = model_perf_df[model_perf_df['category'] == 'Per Theme'].copy()

    # Create pivot table
    pivot = per_theme.pivot(index='subcategory', columns='model', values='f1')

    # Sort by kappa
    theme_order = theme_stats_df.sort_values('fleiss_kappa', ascending=False)['theme_display'].tolist()
    theme_order = [t for t in theme_order if t in pivot.index]
    pivot = pivot.reindex(theme_order)

    # Sort models by overall F1
    model_order = ['GPT-4.1', 'GPT-5', 'GPT-OSS', 'Gemma', 'Llama', 'Nemotron', 'Manual']
    model_order = [m for m in model_order if m in pivot.columns]
    pivot = pivot[model_order]

    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.15,
        vmin=0,
        vmax=0.5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'F1 Score'}
    )

    # Add kappa values on y-axis
    kappa_values = [theme_stats_df[theme_stats_df['theme_display'] == t]['fleiss_kappa'].values[0]
                   for t in theme_order]
    new_labels = [f'{t} (κ={k:.2f})' for t, k in zip(theme_order, kappa_values)]
    ax.set_yticklabels(new_labels, fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Theme (sorted by agreement)', fontsize=12)
    ax.set_title('F1 Score Heatmap: Models × Themes\n(themes sorted by inter-annotator agreement)',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path / 'fig4_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig4_heatmap.pdf', bbox_inches='tight')
    plt.close()


def create_figure_5_human_vs_model_ratio(model_perf_df, human_baseline_df, output_path):
    """Figure 5: Ratio of model performance to human baseline."""
    fig, ax = plt.subplots(figsize=(12, 6))

    level_order = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Overall']

    # Get human baseline averages
    human_by_level = human_baseline_df.groupby(['category', 'subcategory'])['f1'].mean().reset_index()

    # Compute ratios
    ratio_data = []

    for model in MODEL_DISPLAY.values():
        model_data = model_perf_df[model_perf_df['model'] == model]

        for level in level_order:
            if level == 'Overall':
                model_f1 = model_data[
                    (model_data['category'] == 'Overall') &
                    (model_data['subcategory'] == 'All Themes')
                ]['f1'].values
                human_f1 = human_by_level[
                    (human_by_level['category'] == 'Overall') &
                    (human_by_level['subcategory'] == 'All Themes')
                ]['f1'].values
            else:
                model_f1 = model_data[
                    (model_data['category'] == 'By Agreement Level') &
                    (model_data['subcategory'] == level)
                ]['f1'].values
                human_f1 = human_by_level[
                    (human_by_level['category'] == 'By Agreement Level') &
                    (human_by_level['subcategory'] == level)
                ]['f1'].values

            if len(model_f1) > 0 and len(human_f1) > 0 and human_f1[0] > 0:
                ratio = model_f1[0] / human_f1[0] * 100
                ratio_data.append({
                    'model': model,
                    'level': level,
                    'ratio': ratio
                })

    ratio_df = pd.DataFrame(ratio_data)

    # Plot
    x = np.arange(len(level_order))
    width = 0.1
    n_models = len(MODEL_DISPLAY)

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_ratios = ratio_df[ratio_df['model'] == model]
        values = [model_ratios[model_ratios['level'] == l]['ratio'].values[0]
                  if len(model_ratios[model_ratios['level'] == l]) > 0 else 0
                  for l in level_order]

        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=MODEL_COLORS.get(model, '#333333'),
               edgecolor='black', linewidth=0.3)

    # Add 100% reference line
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Human Level (100%)')

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' ', '\n') for l in level_order], fontsize=10)
    ax.set_ylabel('% of Human Performance', fontsize=12)
    ax.set_xlabel('Agreement Level', fontsize=12)
    ax.set_title('Model Performance as Percentage of Human Baseline', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, 110)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path / 'fig5_human_ratio.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'fig5_human_ratio.pdf', bbox_inches='tight')
    plt.close()


def create_figure_6_ranking_with_ci(model_perf_df, human_baseline_df, annotated_df, id_to_annotators, theme_stats_df, output_path):
    """Figure 6: Model ranking with bootstrap confidence intervals and support (N)."""
    np.random.seed(42)
    n_bootstrap = 2000

    # Get theme to level mapping
    theme_level = dict(zip(theme_stats_df['theme'], theme_stats_df['agreement_level']))

    # Count number of themes per agreement level
    n_themes_by_level = {}
    for level in AGREEMENT_LEVELS.keys():
        n_themes_by_level[level] = len([t for t in ALL_THEMES if theme_level.get(t) == level])
    n_themes_by_level['Overall'] = len(ALL_THEMES)

    # Number of samples (sentences)
    n_samples = len(annotated_df)

    # Compute per-sample F1 for bootstrap
    def compute_sample_metrics(annotated_df, id_to_annotators, model, level=None):
        """Compute per-sample metrics for a model, optionally filtered by level."""
        metrics = []

        for _, row in annotated_df.iterrows():
            row_id = row['id']
            if row_id not in id_to_annotators:
                continue

            ann_row = id_to_annotators[row_id]
            all_ann = json.loads(row['all_annotations'])
            pipelines = all_ann.get('pipelines', {})

            consensus = safe_parse(ann_row['consensus_themes'])
            pred = set(pipelines.get(model, {}).get('themes', []))

            # Filter by level if specified
            if level:
                level_themes = [t for t in ALL_THEMES if theme_level.get(t) == level]
                consensus = consensus & set(level_themes)
                pred = pred & set(level_themes)

            tp = len(pred & consensus)
            fp = len(pred - consensus)
            fn = len(consensus - pred)

            metrics.append({'tp': tp, 'fp': fp, 'fn': fn})

        return metrics

    def compute_human_sample_metrics(annotators_df, level=None):
        """Compute per-sample metrics for human annotators vs consensus."""
        all_metrics = []

        for annotator in ANNOTATORS:
            for _, row in annotators_df.iterrows():
                consensus = safe_parse(row['consensus_themes'])
                ann_themes = safe_parse(row[f'{annotator}_themes'])

                # Filter by level if specified
                if level:
                    level_themes = [t for t in ALL_THEMES if theme_level.get(t) == level]
                    consensus = consensus & set(level_themes)
                    ann_themes = ann_themes & set(level_themes)

                tp = len(ann_themes & consensus)
                fp = len(ann_themes - consensus)
                fn = len(consensus - ann_themes)

                all_metrics.append({'tp': tp, 'fp': fp, 'fn': fn})

        return all_metrics

    def bootstrap_f1(metrics, n_bootstrap=2000):
        """Bootstrap F1 score from per-sample metrics."""
        if len(metrics) == 0:
            return 0, 0, 0, 0

        metrics_arr = np.array([[m['tp'], m['fp'], m['fn']] for m in metrics])

        # Compute overall F1
        total_tp = metrics_arr[:, 0].sum()
        total_fp = metrics_arr[:, 1].sum()
        total_fn = metrics_arr[:, 2].sum()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Bootstrap
        f1_boots = []
        n_samples = len(metrics_arr)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_sample = metrics_arr[idx]

            boot_tp = boot_sample[:, 0].sum()
            boot_fp = boot_sample[:, 1].sum()
            boot_fn = boot_sample[:, 2].sum()

            boot_prec = boot_tp / (boot_tp + boot_fp) if (boot_tp + boot_fp) > 0 else 0
            boot_rec = boot_tp / (boot_tp + boot_fn) if (boot_tp + boot_fn) > 0 else 0
            boot_f1 = 2 * boot_prec * boot_rec / (boot_prec + boot_rec) if (boot_prec + boot_rec) > 0 else 0

            f1_boots.append(boot_f1)

        ci_lower = np.percentile(f1_boots, 2.5)
        ci_upper = np.percentile(f1_boots, 97.5)

        support = total_tp + total_fn

        return f1, ci_lower, ci_upper, support

    # Load annotators_df
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / 'data' / 'sets'
    annotators_df = pd.read_csv(data_path / 'benchmark_annotations_by_annotator.csv')

    # Compute metrics with CI for all models and levels
    results = []
    level_order = ['Overall', 'Almost Perfect', 'Substantial', 'Moderate', 'Fair']

    # Human baseline
    for level in level_order:
        level_filter = None if level == 'Overall' else level
        metrics = compute_human_sample_metrics(annotators_df, level_filter)
        f1, ci_lower, ci_upper, support = bootstrap_f1(metrics, n_bootstrap)

        results.append({
            'entity': 'Human',
            'category': level,
            'f1': f1,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'support': support
        })

    # Models
    for model_key, model_name in MODEL_DISPLAY.items():
        for level in level_order:
            level_filter = None if level == 'Overall' else level
            metrics = compute_sample_metrics(annotated_df, id_to_annotators, model_key, level_filter)
            f1, ci_lower, ci_upper, support = bootstrap_f1(metrics, n_bootstrap)

            results.append({
                'entity': model_name,
                'category': level,
                'f1': f1,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'support': support
            })

    results_df = pd.DataFrame(results)

    # Add percentage of human baseline
    human_f1_by_level = results_df[results_df['entity'] == 'Human'].set_index('category')['f1']
    human_ci_lower_by_level = results_df[results_df['entity'] == 'Human'].set_index('category')['ci_lower']
    human_ci_upper_by_level = results_df[results_df['entity'] == 'Human'].set_index('category')['ci_upper']

    results_df['pct_human'] = results_df.apply(
        lambda row: row['f1'] / human_f1_by_level[row['category']] * 100 if human_f1_by_level[row['category']] > 0 else 0,
        axis=1
    )
    results_df['pct_ci_lower'] = results_df.apply(
        lambda row: row['ci_lower'] / human_f1_by_level[row['category']] * 100 if human_f1_by_level[row['category']] > 0 else 0,
        axis=1
    )
    results_df['pct_ci_upper'] = results_df.apply(
        lambda row: row['ci_upper'] / human_f1_by_level[row['category']] * 100 if human_f1_by_level[row['category']] > 0 else 0,
        axis=1
    )

    # Save detailed results
    results_df.to_csv(output_path / 'model_ranking_micro_f1_with_ci.csv', index=False)

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 10))

    # Panel (a): Overall ranking with CI
    ax1 = fig.add_subplot(2, 2, 1)

    overall = results_df[results_df['category'] == 'Overall'].sort_values('f1', ascending=True)
    y_pos = range(len(overall))

    colors = ['#27ae60' if e == 'Human' else MODEL_COLORS.get(e, '#333333') for e in overall['entity']]

    ax1.barh(y_pos, overall['f1'], color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.errorbar(
        overall['f1'], y_pos,
        xerr=[overall['f1'] - overall['ci_lower'], overall['ci_upper'] - overall['f1']],
        fmt='none', color='black', capsize=3, linewidth=1.5
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(overall['entity'], fontsize=11)
    ax1.set_xlabel('Micro-F1 Score', fontsize=12)
    ax1.set_title(f'(a) Overall Model Ranking with 95% CI\n(n = {n_samples} samples, k = {n_themes_by_level.get("Overall", 22)} themes)',
                  fontsize=13, fontweight='bold')

    # Add value labels with support
    for i, (idx, row) in enumerate(overall.iterrows()):
        label = f'{row["f1"]:.3f} [{row["ci_lower"]:.3f}-{row["ci_upper"]:.3f}]'
        ax1.text(row['ci_upper'] + 0.01, i, label, va='center', fontsize=9)

    ax1.set_xlim(0, overall['ci_upper'].max() * 1.35)
    ax1.axvline(x=overall[overall['entity'] == 'Human']['f1'].values[0],
                color='#27ae60', linestyle='--', linewidth=2, alpha=0.7)

    # Panel (b): Performance by agreement level as % of human with support in labels
    ax2 = fig.add_subplot(2, 2, 2)

    level_order_plot = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair']
    models_only = results_df[(results_df['entity'] != 'Human') & (results_df['category'].isin(level_order_plot))]

    x = np.arange(len(level_order_plot))
    width = 0.11
    n_models = len(MODEL_DISPLAY)

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_data = models_only[models_only['entity'] == model]
        values = []
        errors_lower = []
        errors_upper = []

        for level in level_order_plot:
            level_data = model_data[model_data['category'] == level]
            if len(level_data) > 0:
                values.append(level_data['pct_human'].values[0])
                errors_lower.append(level_data['pct_human'].values[0] - level_data['pct_ci_lower'].values[0])
                errors_upper.append(level_data['pct_ci_upper'].values[0] - level_data['pct_human'].values[0])
            else:
                values.append(0)
                errors_lower.append(0)
                errors_upper.append(0)

        offset = (i - n_models/2 + 0.5) * width
        bars = ax2.bar(x + offset, values, width, label=model,
                      color=MODEL_COLORS.get(model, '#333333'), edgecolor='black', linewidth=0.3)
        ax2.errorbar(x + offset, values, yerr=[errors_lower, errors_upper],
                    fmt='none', color='black', capsize=2, linewidth=0.8)

    ax2.axhline(y=100, color='#27ae60', linestyle='--', linewidth=2, label='Human (100%)')

    # Create x-tick labels with number of themes
    xticklabels = [f'{level}\n(k={n_themes_by_level.get(level, 0)} themes)' for level in level_order_plot]
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels, fontsize=10)
    ax2.set_ylabel('% of Human Performance', fontsize=12)
    ax2.set_xlabel('Agreement Level', fontsize=12)
    ax2.set_title(f'(b) Performance by Agreement Level (% of Human Baseline)\n(n = {n_samples} samples)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.set_ylim(0, 130)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Panel (c): Summary table with support
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis('off')

    # Create table data
    entities = ['Human'] + list(MODEL_DISPLAY.values())
    table_data = []

    for entity in entities:
        row_data = [entity]
        for level in ['Overall'] + level_order_plot:
            data = results_df[(results_df['entity'] == entity) & (results_df['category'] == level)]
            if len(data) > 0:
                f1 = data['f1'].values[0]
                ci_l = data['ci_lower'].values[0]
                ci_u = data['ci_upper'].values[0]
                pct = data['pct_human'].values[0]
                row_data.append(f'{f1:.3f}\n[{ci_l:.3f}-{ci_u:.3f}]\n({pct:.1f}%)')
            else:
                row_data.append('-')
        table_data.append(row_data)

    # Column headers with number of themes
    col_labels = ['Model'] + [f'{level}\n(k={n_themes_by_level.get(level, 0)} themes)'
                              for level in ['Overall'] + level_order_plot]

    table = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colColours=['#e8e8e8'] * len(col_labels)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)

    # Highlight Human row
    for j in range(len(col_labels)):
        table[(1, j)].set_facecolor('#d5f5e3')

    ax3.set_title('(c) Detailed Results: Micro-F1 [95% CI] (% of Human)', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    figures_path = output_path.parent.parent / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path / 'fig6_model_ranking_micro_f1.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_path / 'fig6_model_ranking_micro_f1.pdf', bbox_inches='tight')
    plt.close()

    print(f"  - Figure 6: Model ranking with CI and support (N)")


def create_figure_7_micro_f1_by_level(model_perf_df, human_baseline_df, theme_stats_df, output_path):
    """Figure 7: Clean micro-F1 visualization by agreement level with support."""

    level_order = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair']

    # Get theme to level mapping
    theme_level = dict(zip(theme_stats_df['theme'], theme_stats_df['agreement_level']))

    # Count number of themes per agreement level
    n_themes_by_level = {}
    for level in level_order:
        n_themes_by_level[level] = len([t for t in ALL_THEMES if theme_level.get(t) == level])

    # Number of samples
    n_samples = len(model_perf_df[model_perf_df['category'] == 'Per Theme']['subcategory'].unique())
    # Actually get from original data - use theme_stats length as proxy for now
    # We'll pass n_samples as parameter later

    # Prepare data
    by_level = model_perf_df[model_perf_df['category'] == 'By Agreement Level'].copy()
    by_level = by_level[by_level['subcategory'].isin(level_order)]

    human_by_level = human_baseline_df[human_baseline_df['category'] == 'By Agreement Level'].copy()
    human_by_level = human_by_level[human_by_level['subcategory'].isin(level_order)]
    human_avg = human_by_level.groupby('subcategory').agg({
        'f1': 'mean',
        'support': 'first'
    })

    # Also get overall
    overall_models = model_perf_df[
        (model_perf_df['category'] == 'Overall') &
        (model_perf_df['subcategory'] == 'All Themes')
    ]
    overall_human = human_baseline_df[
        (human_baseline_df['category'] == 'Overall') &
        (human_baseline_df['subcategory'] == 'All Themes')
    ]['f1'].mean()

    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel (a): Overall micro-F1 ranking
    ax1 = axes[0, 0]

    overall_sorted = overall_models.sort_values('f1', ascending=True)
    y_pos = range(len(overall_sorted))
    colors = [MODEL_COLORS.get(m, '#333333') for m in overall_sorted['model']]

    bars = ax1.barh(y_pos, overall_sorted['f1'], color=colors, edgecolor='black', linewidth=0.5)

    # Human baseline line
    ax1.axvline(x=overall_human, color='#27ae60', linestyle='--', linewidth=2.5,
                label=f'Human Baseline ({overall_human:.3f})')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(overall_sorted['model'], fontsize=11)
    ax1.set_xlabel('Micro-F1 Score', fontsize=12)
    ax1.set_title(f'(a) Overall Micro-F1 Ranking\n(k = {len(ALL_THEMES)} themes)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)

    # Add value labels
    for bar, (idx, row) in zip(bars, overall_sorted.iterrows()):
        pct = row['f1'] / overall_human * 100
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{row["f1"]:.3f} ({pct:.1f}%)', va='center', fontsize=10)

    ax1.set_xlim(0, overall_human * 1.2)

    # Panel (b): Grouped bar chart by level with support
    ax2 = axes[0, 1]

    x = np.arange(len(level_order))
    width = 0.1
    n_models = len(MODEL_DISPLAY)

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_data = by_level[by_level['model'] == model]
        values = [model_data[model_data['subcategory'] == l]['f1'].values[0]
                  if len(model_data[model_data['subcategory'] == l]) > 0 else 0
                  for l in level_order]

        offset = (i - n_models/2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=model, color=MODEL_COLORS.get(model, '#333333'),
                edgecolor='black', linewidth=0.3)

    # Human baseline
    human_values = [human_avg.loc[l, 'f1'] if l in human_avg.index else 0 for l in level_order]
    ax2.plot(x, human_values, 'o--', color='#27ae60', linewidth=2.5, markersize=10, label='Human Baseline')

    # X-tick labels with number of themes
    xticklabels = [f'{level}\n(k={n_themes_by_level.get(level, 0)})' for level in level_order]
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels, fontsize=10)
    ax2.set_ylabel('Micro-F1 Score', fontsize=12)
    ax2.set_xlabel('Agreement Level', fontsize=12)
    ax2.set_title('(b) Micro-F1 by Agreement Level (Theme-based)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Panel (c): Heatmap of micro-F1 by model and level
    ax3 = axes[1, 0]

    # Prepare heatmap data
    heatmap_data = []
    for model in MODEL_DISPLAY.values():
        row = []
        for level in level_order:
            model_data = by_level[(by_level['model'] == model) & (by_level['subcategory'] == level)]
            if len(model_data) > 0:
                row.append(model_data['f1'].values[0])
            else:
                row.append(0)
        heatmap_data.append(row)

    # Add human row
    human_row = [human_avg.loc[l, 'f1'] if l in human_avg.index else 0 for l in level_order]
    heatmap_data.append(human_row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=list(MODEL_DISPLAY.values()) + ['Human'],
        columns=[f'{l}\n(k={n_themes_by_level.get(l, 0)})' for l in level_order]
    )

    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.4, vmin=0, vmax=1.0, ax=ax3, linewidths=0.5,
                cbar_kws={'label': 'Micro-F1'})

    ax3.set_title('(c) Micro-F1 Heatmap: Models x Agreement Level', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Agreement Level', fontsize=12)
    ax3.set_ylabel('Model', fontsize=12)

    # Panel (d): Performance gap (Human - Model)
    ax4 = axes[1, 1]

    gap_data = []
    for model in MODEL_DISPLAY.values():
        for level in level_order:
            model_data = by_level[(by_level['model'] == model) & (by_level['subcategory'] == level)]
            human_f1 = human_avg.loc[level, 'f1'] if level in human_avg.index else 0

            if len(model_data) > 0 and human_f1 > 0:
                model_f1 = model_data['f1'].values[0]
                gap = human_f1 - model_f1
                gap_data.append({
                    'Model': model,
                    'Level': level,
                    'Gap': gap,
                    'N_themes': n_themes_by_level.get(level, 0)
                })

    gap_df = pd.DataFrame(gap_data)

    x = np.arange(len(level_order))
    width = 0.1

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_gaps = gap_df[gap_df['Model'] == model]
        values = [model_gaps[model_gaps['Level'] == l]['Gap'].values[0]
                  if len(model_gaps[model_gaps['Level'] == l]) > 0 else 0
                  for l in level_order]

        offset = (i - n_models/2 + 0.5) * width
        ax4.bar(x + offset, values, width, label=model, color=MODEL_COLORS.get(model, '#333333'),
                edgecolor='black', linewidth=0.3)

    ax4.axhline(y=0, color='#27ae60', linestyle='--', linewidth=2)

    xticklabels = [f'{level}\n(k={n_themes_by_level.get(level, 0)})' for level in level_order]
    ax4.set_xticks(x)
    ax4.set_xticklabels(xticklabels, fontsize=10)
    ax4.set_ylabel('Performance Gap (Human - Model)', fontsize=12)
    ax4.set_xlabel('Agreement Level', fontsize=12)
    ax4.set_title('(d) Performance Gap from Human Baseline', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8, ncol=2)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    figures_path = output_path.parent.parent / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path / 'fig7_micro_f1_by_level.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_path / 'fig7_micro_f1_by_level.pdf', bbox_inches='tight')
    plt.close()

    print(f"  - Figure 7: Micro-F1 by agreement level with support")


def create_figure_8_sentence_level_agreement(annotated_df, id_to_annotators, annotators_df, output_path):
    """Figure 8: Model performance by sentence-level agreement (not theme-level)."""

    # For each sentence, compute agreement score across all themes
    sentence_agreement = []

    for _, ann_row in annotators_df.iterrows():
        row_id = ann_row['id']

        # Count agreements across all themes for this sentence
        n_full_agreement = 0
        n_themes_evaluated = 0

        for theme in ALL_THEMES:
            # Check if all annotators agree on this theme for this sentence
            theme_votes = []
            for ann in ANNOTATORS:
                ann_themes = safe_parse(ann_row[f'{ann}_themes'])
                theme_votes.append(1 if theme in ann_themes else 0)

            n_themes_evaluated += 1
            if len(set(theme_votes)) == 1:  # All agree (all 0 or all 1)
                n_full_agreement += 1

        # Compute agreement rate for this sentence
        agreement_rate = n_full_agreement / n_themes_evaluated if n_themes_evaluated > 0 else 0

        sentence_agreement.append({
            'id': row_id,
            'agreement_rate': agreement_rate,
            'n_full_agreement': n_full_agreement,
            'n_themes': n_themes_evaluated
        })

    sentence_df = pd.DataFrame(sentence_agreement)

    # Categorize sentences by agreement level
    def get_sentence_agreement_level(rate):
        if rate >= 0.95:
            return 'High Agreement (≥95%)'
        elif rate >= 0.85:
            return 'Good Agreement (85-95%)'
        elif rate >= 0.75:
            return 'Moderate Agreement (75-85%)'
        else:
            return 'Low Agreement (<75%)'

    sentence_df['agreement_level'] = sentence_df['agreement_rate'].apply(get_sentence_agreement_level)

    # Compute model performance for each agreement level
    level_order = ['High Agreement (≥95%)', 'Good Agreement (85-95%)', 'Moderate Agreement (75-85%)', 'Low Agreement (<75%)']
    level_colors = {'High Agreement (≥95%)': '#27ae60', 'Good Agreement (85-95%)': '#3498db',
                   'Moderate Agreement (75-85%)': '#f39c12', 'Low Agreement (<75%)': '#e74c3c'}

    # Count sentences per level
    n_sentences_by_level = sentence_df['agreement_level'].value_counts().to_dict()

    # Create id to agreement level mapping
    id_to_level = dict(zip(sentence_df['id'], sentence_df['agreement_level']))

    # Compute metrics for each model and level
    results = []

    for model_key, model_name in MODEL_DISPLAY.items():
        for level in level_order:
            level_ids = sentence_df[sentence_df['agreement_level'] == level]['id'].tolist()

            tp, fp, fn = 0, 0, 0

            for row_id in level_ids:
                if row_id not in id_to_annotators:
                    continue

                ann_row = id_to_annotators[row_id]

                # Find the row in annotated_df
                ann_df_row = annotated_df[annotated_df['id'] == row_id]
                if len(ann_df_row) == 0:
                    continue

                all_ann = json.loads(ann_df_row.iloc[0]['all_annotations'])
                pipelines = all_ann.get('pipelines', {})

                consensus = safe_parse(ann_row['consensus_themes'])
                pred = set(pipelines.get(model_key, {}).get('themes', []))

                tp += len(pred & consensus)
                fp += len(pred - consensus)
                fn += len(consensus - pred)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'model': model_name,
                'level': level,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'n_sentences': n_sentences_by_level.get(level, 0)
            })

    # Human baseline (annotators vs consensus)
    for level in level_order:
        level_ids = sentence_df[sentence_df['agreement_level'] == level]['id'].tolist()

        all_tp, all_fp, all_fn = 0, 0, 0

        for annotator in ANNOTATORS:
            for row_id in level_ids:
                if row_id not in id_to_annotators:
                    continue

                ann_row = id_to_annotators[row_id]
                consensus = safe_parse(ann_row['consensus_themes'])
                ann_themes = safe_parse(ann_row[f'{annotator}_themes'])

                all_tp += len(ann_themes & consensus)
                all_fp += len(ann_themes - consensus)
                all_fn += len(consensus - ann_themes)

        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'model': 'Human',
            'level': level,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'n_sentences': n_sentences_by_level.get(level, 0)
        })

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_path / 'model_performance_by_sentence_agreement.csv', index=False)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel (a): Distribution of sentences by agreement level
    ax1 = axes[0, 0]

    counts = [n_sentences_by_level.get(l, 0) for l in level_order]
    colors = [level_colors[l] for l in level_order]
    bars = ax1.bar(range(len(level_order)), counts, color=colors, edgecolor='black', linewidth=0.5)

    ax1.set_xticks(range(len(level_order)))
    ax1.set_xticklabels([l.replace(' ', '\n') for l in level_order], fontsize=9)
    ax1.set_ylabel('Number of Sentences', fontsize=12)
    ax1.set_xlabel('Sentence Agreement Level', fontsize=12)
    ax1.set_title(f'(a) Distribution of Sentences by Agreement Level\n(n = {len(sentence_df)} total sentences)',
                  fontsize=13, fontweight='bold')

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Panel (b): Model F1 by sentence agreement level
    ax2 = axes[0, 1]

    x = np.arange(len(level_order))
    width = 0.1
    n_models = len(MODEL_DISPLAY)

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_data = results_df[results_df['model'] == model]
        values = [model_data[model_data['level'] == l]['f1'].values[0]
                  if len(model_data[model_data['level'] == l]) > 0 else 0
                  for l in level_order]

        offset = (i - n_models/2 + 0.5) * width
        ax2.bar(x + offset, values, width, label=model, color=MODEL_COLORS.get(model, '#333333'),
                edgecolor='black', linewidth=0.3)

    # Human baseline
    human_data = results_df[results_df['model'] == 'Human']
    human_values = [human_data[human_data['level'] == l]['f1'].values[0]
                   if len(human_data[human_data['level'] == l]) > 0 else 0
                   for l in level_order]
    ax2.plot(x, human_values, 'o--', color='#27ae60', linewidth=2.5, markersize=10, label='Human Baseline')

    xticklabels = [f'{l.split()[0]}\n(n={n_sentences_by_level.get(l, 0)})' for l in level_order]
    ax2.set_xticks(x)
    ax2.set_xticklabels(xticklabels, fontsize=9)
    ax2.set_ylabel('Micro-F1 Score', fontsize=12)
    ax2.set_xlabel('Sentence Agreement Level', fontsize=12)
    ax2.set_title('(b) Micro-F1 by Sentence Agreement Level', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Panel (c): Heatmap
    ax3 = axes[1, 0]

    heatmap_data = []
    for model in list(MODEL_DISPLAY.values()) + ['Human']:
        row = []
        for level in level_order:
            model_data = results_df[(results_df['model'] == model) & (results_df['level'] == level)]
            if len(model_data) > 0:
                row.append(model_data['f1'].values[0])
            else:
                row.append(0)
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=list(MODEL_DISPLAY.values()) + ['Human'],
        columns=[f'{l.split()[0]}\n(n={n_sentences_by_level.get(l, 0)})' for l in level_order]
    )

    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.4, vmin=0, vmax=1.0, ax=ax3, linewidths=0.5,
                cbar_kws={'label': 'Micro-F1'})

    ax3.set_title('(c) Micro-F1 Heatmap: Models x Sentence Agreement', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Sentence Agreement Level', fontsize=12)
    ax3.set_ylabel('Model', fontsize=12)

    # Panel (d): Performance as % of human by sentence agreement
    ax4 = axes[1, 1]

    pct_data = []
    for model in MODEL_DISPLAY.values():
        for level in level_order:
            model_f1 = results_df[(results_df['model'] == model) & (results_df['level'] == level)]['f1'].values
            human_f1 = results_df[(results_df['model'] == 'Human') & (results_df['level'] == level)]['f1'].values

            if len(model_f1) > 0 and len(human_f1) > 0 and human_f1[0] > 0:
                pct = model_f1[0] / human_f1[0] * 100
                pct_data.append({
                    'model': model,
                    'level': level,
                    'pct': pct
                })

    pct_df = pd.DataFrame(pct_data)

    for i, model in enumerate(MODEL_DISPLAY.values()):
        model_pcts = pct_df[pct_df['model'] == model]
        values = [model_pcts[model_pcts['level'] == l]['pct'].values[0]
                  if len(model_pcts[model_pcts['level'] == l]) > 0 else 0
                  for l in level_order]

        offset = (i - n_models/2 + 0.5) * width
        ax4.bar(x + offset, values, width, label=model, color=MODEL_COLORS.get(model, '#333333'),
                edgecolor='black', linewidth=0.3)

    ax4.axhline(y=100, color='#27ae60', linestyle='--', linewidth=2, label='Human (100%)')

    xticklabels = [f'{l.split()[0]}\n(n={n_sentences_by_level.get(l, 0)})' for l in level_order]
    ax4.set_xticks(x)
    ax4.set_xticklabels(xticklabels, fontsize=9)
    ax4.set_ylabel('% of Human Performance', fontsize=12)
    ax4.set_xlabel('Sentence Agreement Level', fontsize=12)
    ax4.set_title('(d) Model Performance as % of Human Baseline', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8, ncol=2)
    ax4.set_ylim(0, 130)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    figures_path = output_path.parent.parent / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path / 'fig8_sentence_level_agreement.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_path / 'fig8_sentence_level_agreement.pdf', bbox_inches='tight')
    plt.close()

    # Save sentence-level agreement data
    sentence_df.to_csv(output_path / 'sentence_agreement_scores.csv', index=False)

    print(f"  - Figure 8: Performance by sentence-level agreement (n = {len(sentence_df)} sentences)")


def create_figure_9_small_vs_large(annotated_df, id_to_annotators, annotators_df, theme_stats_df, output_path):
    """Figure 9: Comparison of Small vs Large model variants."""

    # Get theme to level mapping
    theme_level = dict(zip(theme_stats_df['theme'], theme_stats_df['agreement_level']))

    # Compute metrics for all models (small and large)
    def compute_model_metrics(model_key):
        tp, fp, fn = 0, 0, 0

        for _, row in annotated_df.iterrows():
            row_id = row['id']
            if row_id not in id_to_annotators:
                continue

            ann_row = id_to_annotators[row_id]
            all_ann = json.loads(row['all_annotations'])
            pipelines = all_ann.get('pipelines', {})

            consensus = safe_parse(ann_row['consensus_themes'])
            pred = set(pipelines.get(model_key, {}).get('themes', []))

            tp += len(pred & consensus)
            fp += len(pred - consensus)
            fn += len(consensus - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

    # Compute human baseline
    human_tp, human_fp, human_fn = 0, 0, 0
    for annotator in ANNOTATORS:
        for _, ann_row in annotators_df.iterrows():
            consensus = safe_parse(ann_row['consensus_themes'])
            ann_themes = safe_parse(ann_row[f'{annotator}_themes'])
            human_tp += len(ann_themes & consensus)
            human_fp += len(ann_themes - consensus)
            human_fn += len(consensus - ann_themes)

    human_prec = human_tp / (human_tp + human_fp) if (human_tp + human_fp) > 0 else 0
    human_rec = human_tp / (human_tp + human_fn) if (human_tp + human_fn) > 0 else 0
    human_f1 = 2 * human_prec * human_rec / (human_prec + human_rec) if (human_prec + human_rec) > 0 else 0

    # Compute metrics for all models
    results = []
    base_models = ['gpt_4_1', 'gpt_5', 'gpt_oss', 'gemma', 'llama', 'nemotron']

    for base in base_models:
        # Large variant
        large_key = f'{base}_large'
        large_metrics = compute_model_metrics(large_key)
        results.append({
            'base_model': base.upper().replace('_', '-'),
            'variant': 'Large',
            'model_key': large_key,
            **large_metrics
        })

        # Small variant
        small_key = f'{base}_small'
        small_metrics = compute_model_metrics(small_key)
        results.append({
            'base_model': base.upper().replace('_', '-'),
            'variant': 'Small',
            'model_key': small_key,
            **small_metrics
        })

    # Add manual
    manual_metrics = compute_model_metrics('manual_manual')
    results.append({
        'base_model': 'MANUAL',
        'variant': 'N/A',
        'model_key': 'manual_manual',
        **manual_metrics
    })

    results_df = pd.DataFrame(results)
    results_df['pct_human'] = results_df['f1'] / human_f1 * 100

    # Save results
    results_df.to_csv(output_path / 'model_comparison_small_vs_large.csv', index=False)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel (a): Grouped bar chart - Small vs Large by model
    ax1 = axes[0, 0]

    base_models_display = ['GPT-4-1', 'GPT-5', 'GPT-OSS', 'GEMMA', 'LLAMA', 'NEMOTRON']
    x = np.arange(len(base_models_display))
    width = 0.35

    large_f1 = [results_df[(results_df['base_model'] == b) & (results_df['variant'] == 'Large')]['f1'].values[0]
                for b in base_models_display]
    small_f1 = [results_df[(results_df['base_model'] == b) & (results_df['variant'] == 'Small')]['f1'].values[0]
                for b in base_models_display]

    bars1 = ax1.bar(x - width/2, large_f1, width, label='Large', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, small_f1, width, label='Small', color='#e74c3c', edgecolor='black', linewidth=0.5)

    # Human baseline line
    ax1.axhline(y=human_f1, color='#3498db', linestyle='--', linewidth=2, label=f'Human ({human_f1:.3f})')

    # Manual baseline
    manual_f1 = results_df[results_df['base_model'] == 'MANUAL']['f1'].values[0]
    ax1.axhline(y=manual_f1, color='#8c564b', linestyle=':', linewidth=2, label=f'Manual ({manual_f1:.3f})')

    ax1.set_xticks(x)
    ax1.set_xticklabels(base_models_display, fontsize=10, rotation=45, ha='right')
    ax1.set_ylabel('Micro-F1 Score', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_title('(a) Small vs Large Model Comparison\n(Micro-F1 Score)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, max(human_f1 * 1.1, 0.2))
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # Panel (b): Difference (Large - Small)
    ax2 = axes[0, 1]

    diff = [l - s for l, s in zip(large_f1, small_f1)]
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]

    bars = ax2.bar(x, diff, width=0.6, color=colors, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(base_models_display, fontsize=10, rotation=45, ha='right')
    ax2.set_ylabel('F1 Difference (Large - Small)', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_title('(b) Performance Gain from Large Training Set\n(Positive = Large is better)', fontsize=13, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Add value labels
    for bar, d in zip(bars, diff):
        va = 'bottom' if d >= 0 else 'top'
        offset = 0.001 if d >= 0 else -0.001
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{d:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    # Panel (c): % of Human performance
    ax3 = axes[1, 0]

    large_pct = [results_df[(results_df['base_model'] == b) & (results_df['variant'] == 'Large')]['pct_human'].values[0]
                 for b in base_models_display]
    small_pct = [results_df[(results_df['base_model'] == b) & (results_df['variant'] == 'Small')]['pct_human'].values[0]
                 for b in base_models_display]

    bars1 = ax3.bar(x - width/2, large_pct, width, label='Large', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, small_pct, width, label='Small', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax3.axhline(y=100, color='#3498db', linestyle='--', linewidth=2, label='Human (100%)')

    ax3.set_xticks(x)
    ax3.set_xticklabels(base_models_display, fontsize=10, rotation=45, ha='right')
    ax3.set_ylabel('% of Human Performance', fontsize=12)
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_title('(c) Performance as % of Human Baseline', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 120)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Panel (d): Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    for base in base_models_display:
        large_data = results_df[(results_df['base_model'] == base) & (results_df['variant'] == 'Large')]
        small_data = results_df[(results_df['base_model'] == base) & (results_df['variant'] == 'Small')]

        if len(large_data) > 0 and len(small_data) > 0:
            l_f1 = large_data['f1'].values[0]
            s_f1 = small_data['f1'].values[0]
            diff = l_f1 - s_f1
            l_pct = large_data['pct_human'].values[0]
            s_pct = small_data['pct_human'].values[0]

            table_data.append([
                base,
                f'{l_f1:.3f}',
                f'{s_f1:.3f}',
                f'{diff:+.3f}',
                f'{l_pct:.1f}%',
                f'{s_pct:.1f}%'
            ])

    # Add manual row
    manual_data = results_df[results_df['base_model'] == 'MANUAL']
    if len(manual_data) > 0:
        m_f1 = manual_data['f1'].values[0]
        m_pct = manual_data['pct_human'].values[0]
        table_data.append(['MANUAL', f'{m_f1:.3f}', '-', '-', f'{m_pct:.1f}%', '-'])

    # Add human row
    table_data.append(['HUMAN', f'{human_f1:.3f}', '-', '-', '100.0%', '-'])

    col_labels = ['Model', 'Large F1', 'Small F1', 'Diff', 'Large %', 'Small %']

    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colColours=['#e8e8e8'] * len(col_labels)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight human row
    for j in range(len(col_labels)):
        table[(len(table_data), j)].set_facecolor('#d5f5e3')

    ax4.set_title('(d) Summary: Small vs Large Comparison\n(n = 249 sentences, k = 22 themes)', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    figures_path = output_path.parent.parent / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path / 'fig9_small_vs_large.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_path / 'fig9_small_vs_large.pdf', bbox_inches='tight')
    plt.close()

    print(f"  - Figure 9: Small vs Large model comparison")


# =============================================================================
# CSV GENERATION
# =============================================================================

def create_consolidated_csv(theme_stats_df, model_perf_df, human_baseline_df, correlation_df, output_path):
    """Create a beautifully formatted consolidated CSV."""

    # Sheet 1: Executive Summary
    summary_rows = []

    # Header
    summary_rows.append(['=' * 80])
    summary_rows.append(['COMPREHENSIVE EVALUATION ANALYSIS - EXECUTIVE SUMMARY'])
    summary_rows.append(['=' * 80])
    summary_rows.append([])

    # Inter-annotator agreement
    summary_rows.append(['INTER-ANNOTATOR AGREEMENT'])
    summary_rows.append(['-' * 40])
    summary_rows.append(['Metric', 'Value'])
    summary_rows.append(["Mean Fleiss' Kappa", f"{theme_stats_df['fleiss_kappa'].mean():.3f}"])
    summary_rows.append(["Std Fleiss' Kappa", f"{theme_stats_df['fleiss_kappa'].std():.3f}"])
    summary_rows.append(['Min Kappa', f"{theme_stats_df['fleiss_kappa'].min():.3f}"])
    summary_rows.append(['Max Kappa', f"{theme_stats_df['fleiss_kappa'].max():.3f}"])
    summary_rows.append([])

    # Distribution by level
    summary_rows.append(['THEMES BY AGREEMENT LEVEL'])
    summary_rows.append(['-' * 40])
    for level in ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Slight']:
        count = len(theme_stats_df[theme_stats_df['agreement_level'] == level])
        summary_rows.append([level, count])
    summary_rows.append([])

    # Human baseline
    human_overall = human_baseline_df[
        (human_baseline_df['category'] == 'Overall') &
        (human_baseline_df['subcategory'] == 'All Themes')
    ]['f1'].mean()

    summary_rows.append(['HUMAN BASELINE (Annotator vs Consensus)'])
    summary_rows.append(['-' * 40])
    summary_rows.append(['Average F1', f"{human_overall:.3f}"])
    summary_rows.append([])

    # Model performance
    model_overall = model_perf_df[
        (model_perf_df['category'] == 'Overall') &
        (model_perf_df['subcategory'] == 'All Themes')
    ].sort_values('f1', ascending=False)

    summary_rows.append(['MODEL PERFORMANCE (Overall Micro-F1)'])
    summary_rows.append(['-' * 40])
    summary_rows.append(['Model', 'F1', 'Precision', 'Recall', '% of Human'])
    for _, row in model_overall.iterrows():
        ratio = row['f1'] / human_overall * 100
        summary_rows.append([row['model'], f"{row['f1']:.3f}", f"{row['precision']:.3f}",
                           f"{row['recall']:.3f}", f"{ratio:.1f}%"])

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / 'summary_executive.csv', index=False, header=False)

    # Sheet 2: Theme Analysis
    theme_export = theme_stats_df.copy()
    theme_export = theme_export.round(4)
    theme_export.columns = [c.replace('_', ' ').title() for c in theme_export.columns]
    theme_export.to_csv(output_path / 'analysis_themes.csv', index=False)

    # Sheet 3: Model Performance Matrix
    # Pivot for readability
    by_level = model_perf_df[model_perf_df['category'] == 'By Agreement Level'].copy()
    overall = model_perf_df[
        (model_perf_df['category'] == 'Overall') &
        (model_perf_df['subcategory'] == 'All Themes')
    ].copy()
    overall['subcategory'] = 'Overall'
    combined = pd.concat([by_level, overall])

    # Create wide format
    pivot_f1 = combined.pivot(index='model', columns='subcategory', values='f1')
    pivot_prec = combined.pivot(index='model', columns='subcategory', values='precision')
    pivot_rec = combined.pivot(index='model', columns='subcategory', values='recall')
    pivot_support = combined.pivot(index='model', columns='subcategory', values='support')

    # Combine with multi-level columns
    level_order = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Overall']

    rows = []
    for model in pivot_f1.index:
        row = {'Model': model}
        for level in level_order:
            if level in pivot_f1.columns:
                row[f'{level} F1'] = f"{pivot_f1.loc[model, level]:.3f}" if pd.notna(pivot_f1.loc[model, level]) else ''
                row[f'{level} Prec'] = f"{pivot_prec.loc[model, level]:.3f}" if pd.notna(pivot_prec.loc[model, level]) else ''
                row[f'{level} Rec'] = f"{pivot_rec.loc[model, level]:.3f}" if pd.notna(pivot_rec.loc[model, level]) else ''
                row[f'{level} N'] = int(pivot_support.loc[model, level]) if pd.notna(pivot_support.loc[model, level]) else ''
        rows.append(row)

    perf_matrix = pd.DataFrame(rows)
    perf_matrix.to_csv(output_path / 'analysis_model_performance.csv', index=False)

    # Sheet 4: Human Baseline
    human_export = human_baseline_df.copy()
    human_export = human_export.round(4)
    human_export.to_csv(output_path / 'analysis_human_baseline.csv', index=False)

    # Sheet 5: Correlation Analysis
    if len(correlation_df) > 0:
        correlation_df.round(4).to_csv(output_path / 'analysis_correlations.csv', index=False)

    # Sheet 6: Per-theme model performance
    per_theme = model_perf_df[model_perf_df['category'] == 'Per Theme'].copy()
    per_theme_pivot = per_theme.pivot(index='subcategory', columns='model', values='f1')

    # Add kappa column
    kappa_map = dict(zip(theme_stats_df['theme_display'], theme_stats_df['fleiss_kappa']))
    per_theme_pivot['Kappa'] = per_theme_pivot.index.map(kappa_map)
    per_theme_pivot = per_theme_pivot.sort_values('Kappa', ascending=False)

    # Reorder columns
    cols = ['Kappa'] + [c for c in per_theme_pivot.columns if c != 'Kappa']
    per_theme_pivot = per_theme_pivot[cols]
    per_theme_pivot = per_theme_pivot.round(3)
    per_theme_pivot.to_csv(output_path / 'analysis_per_theme_f1.csv')

    # Create master consolidated file
    with open(output_path / 'CONSOLIDATED_ANALYSIS.csv', 'w') as f:
        f.write("COMPREHENSIVE MODEL EVALUATION ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        f.write("SECTION 1: INTER-ANNOTATOR AGREEMENT BY THEME\n")
        f.write("-" * 50 + "\n")
        theme_export.to_csv(f, index=False)

        f.write("\n\nSECTION 2: MODEL PERFORMANCE BY AGREEMENT LEVEL\n")
        f.write("-" * 50 + "\n")
        perf_matrix.to_csv(f, index=False)

        f.write("\n\nSECTION 3: HUMAN BASELINE\n")
        f.write("-" * 50 + "\n")
        human_by_level_pivot = human_baseline_df[human_baseline_df['category'] != 'Per Theme'].pivot(
            index='annotator', columns='subcategory', values='f1'
        ).round(3)
        human_by_level_pivot.to_csv(f)

        f.write("\n\nSECTION 4: PER-THEME MODEL F1 (sorted by agreement)\n")
        f.write("-" * 50 + "\n")
        per_theme_pivot.to_csv(f)

        if len(correlation_df) > 0:
            f.write("\n\nSECTION 5: CORRELATION ANALYSIS\n")
            f.write("-" * 50 + "\n")
            correlation_df.round(4).to_csv(f, index=False)

    print(f"  - CONSOLIDATED_ANALYSIS.csv (master file)")
    print(f"  - summary_executive.csv")
    print(f"  - analysis_themes.csv")
    print(f"  - analysis_model_performance.csv")
    print(f"  - analysis_human_baseline.csv")
    print(f"  - analysis_correlations.csv")
    print(f"  - analysis_per_theme_f1.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis function."""
    # Setup paths
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'paper' / 'results' / 'evaluation_results'
    figures_path = base_path / 'paper' / 'figures'

    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[1/8] Loading data...")
    annotators_df, annotated_df, id_to_annotators = load_data(base_path)
    print(f"  - {len(annotators_df)} samples, {len(ANNOTATORS)} annotators, {len(MODELS)} models")

    # Compute theme statistics
    print("\n[2/8] Computing theme statistics...")
    theme_stats_df = compute_theme_statistics(annotators_df)
    print(f"  - Mean κ = {theme_stats_df['fleiss_kappa'].mean():.3f}")

    # Compute model performance
    print("\n[3/8] Computing model performance...")
    model_perf_df = compute_model_performance(annotated_df, id_to_annotators, theme_stats_df)
    print(f"  - {len(model_perf_df)} performance records")

    # Compute human baseline
    print("\n[4/8] Computing human baseline...")
    human_baseline_df = compute_human_baseline(annotators_df, theme_stats_df)

    # Compute correlations
    print("\n[5/8] Computing correlation analysis...")
    correlation_df = compute_correlation_analysis(model_perf_df, theme_stats_df)

    # Generate figures
    print("\n[6/8] Generating figures...")
    create_figure_1_agreement_distribution(theme_stats_df, figures_path)
    print("  - Figure 1: Agreement distribution")

    create_figure_2_model_performance(model_perf_df, human_baseline_df, figures_path)
    print("  - Figure 2: Model performance comparison")

    create_figure_3_f1_vs_kappa(model_perf_df, theme_stats_df, figures_path)
    print("  - Figure 3: F1 vs Kappa scatter")

    create_figure_4_heatmap(model_perf_df, theme_stats_df, figures_path)
    print("  - Figure 4: Performance heatmap")

    create_figure_5_human_vs_model_ratio(model_perf_df, human_baseline_df, figures_path)
    print("  - Figure 5: Human ratio comparison")

    create_figure_6_ranking_with_ci(model_perf_df, human_baseline_df, annotated_df, id_to_annotators, theme_stats_df, output_path)

    create_figure_7_micro_f1_by_level(model_perf_df, human_baseline_df, theme_stats_df, output_path)

    create_figure_8_sentence_level_agreement(annotated_df, id_to_annotators, annotators_df, output_path)

    create_figure_9_small_vs_large(annotated_df, id_to_annotators, annotators_df, theme_stats_df, output_path)

    # Generate CSVs
    print("\n[7/8] Generating CSV reports...")
    create_consolidated_csv(theme_stats_df, model_perf_df, human_baseline_df, correlation_df, output_path)

    # Print summary
    print("\n[8/8] Analysis complete!")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nInter-Annotator Agreement:")
    print(f"  Mean Fleiss' κ: {theme_stats_df['fleiss_kappa'].mean():.3f} ± {theme_stats_df['fleiss_kappa'].std():.3f}")

    print(f"\nHuman Baseline (Overall F1):")
    human_overall = human_baseline_df[
        (human_baseline_df['category'] == 'Overall') &
        (human_baseline_df['subcategory'] == 'All Themes')
    ]['f1'].mean()
    print(f"  {human_overall:.3f}")

    print(f"\nModel Performance (Overall Micro-F1):")
    model_overall = model_perf_df[
        (model_perf_df['category'] == 'Overall') &
        (model_perf_df['subcategory'] == 'All Themes')
    ].sort_values('f1', ascending=False)

    for _, row in model_overall.iterrows():
        ratio = row['f1'] / human_overall * 100
        print(f"  {row['model']:12s}: {row['f1']:.3f} ({ratio:5.1f}% of human)")

    if len(correlation_df) > 0:
        print(f"\nCorrelation (F1 vs κ):")
        avg_corr = correlation_df['pearson_f1_vs_kappa'].mean()
        print(f"  Average Pearson r: {avg_corr:.3f}")

    print("\n" + "=" * 80)
    print(f"Figures saved to: {figures_path}")
    print(f"CSVs saved to: {output_path}")
    print("=" * 80)

    return {
        'theme_stats': theme_stats_df,
        'model_performance': model_perf_df,
        'human_baseline': human_baseline_df,
        'correlations': correlation_df
    }


if __name__ == '__main__':
    results = main()

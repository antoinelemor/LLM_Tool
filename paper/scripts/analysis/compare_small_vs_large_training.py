#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
compare_small_vs_large_training.py

MAIN OBJECTIVE:
---------------
Compare model performance between models trained on Small vs Large datasets.
Analyzes whether increasing training data volume improves downstream model
performance, stratified by inter-annotator agreement levels.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- json

MAIN FEATURES:
--------------
1) Load predictions from Small and Large trained models
2) Compare micro-F1 scores by training set size
3) Stratify analysis by annotation agreement level
4) Generate comparative visualizations
5) Export results to CSV

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# CONFIGURATION
# =============================================================================

ANNOTATORS = ['shdin', 'Jeremy', 'jdrouin', 'BenjaminCarignan']

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

# Map theme label names (from predictions) to standard theme names
THEME_LABEL_MAP = {
    'agriculture': 'theme_agriculture',
    'culture_nationalism': 'theme_culture_nationalism',
    'defense': 'theme_defense',
    'domestic_commerce': 'theme_domestic_commerce',
    'education': 'theme_education',
    'energy': 'theme_energy',
    'environment': 'theme_environment',
    'foreign_trade': 'theme_foreign_trade',
    'governments_governance': 'theme_governments_governance',
    'health': 'theme_health',
    'housing': 'theme_housing',
    'immigration': 'theme_immigration',
    'indigenous_affairs': 'theme_indigenous_affairs',
    'international_affairs': 'theme_international_affairs',
    'labor': 'theme_labor',
    'law_and_crime': 'theme_law_and_crime',
    'macroeconomics': 'theme_macroeconomics',
    'public_lands': 'theme_public_lands',
    'rights_liberties_minorities_discrimination': 'theme_rights_liberties_minorities_discrimination',
    'social_welfare': 'theme_social_welfare',
    'technology': 'theme_technology',
    'transportation': 'theme_transportation'
}

MODEL_COLORS = {
    'gpt_4_1': '#1f77b4',
    'gpt_5': '#2ca02c',
    'gpt_oss': '#17becf',
    'gemma': '#ff7f0e',
    'llama': '#d62728',
    'nemotron': '#9467bd'
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


def extract_predictions_from_labels(df):
    """
    Extract model predictions from label columns.

    Labels have format: model_category_trainset_specific IS/IS NOT
    e.g., "gpt_oss_themes_large_technology IS NOT"
    e.g., "llama_themes_large_governments_governance IS"

    Returns dict: {row_id: {model: {category: set of positive predictions}}}
    """
    label_cols = [c for c in df.columns if '_label' in c and '_label_id' not in c]

    predictions = {}

    for idx, row in df.iterrows():
        row_id = row['id']
        predictions[row_id] = defaultdict(lambda: defaultdict(set))

        for col in label_cols:
            label = row[col]
            if pd.isna(label):
                continue

            label_str = str(label).strip()

            # Parse label: model_category_trainset_specific IS/IS NOT
            # Theme names can have underscores (e.g., governments_governance)

            # Handle themes - use greedy match for theme name
            match = re.match(r'^(\w+?)_themes_(large|small)_(.+)\s+(IS|IS NOT)$', label_str)
            if match:
                model, trainset, theme, prediction = match.groups()
                if prediction == 'IS':
                    theme_full = f'theme_{theme}'
                    if theme_full in ALL_THEMES:
                        predictions[row_id][model]['themes'].add(theme_full)
                continue

            # Handle sentiment (no IS/IS NOT, just the value)
            match = re.match(r'^(\w+?)_sentiment_(large|small)_(\w+)$', label_str)
            if match:
                model, trainset, sentiment = match.groups()
                predictions[row_id][model]['sentiment'] = f'sentiment_{sentiment}'
                continue

            # Handle political parties
            match = re.match(r'^(\w+?)_political_parties_(large|small)_(\w+)\s+(IS|IS NOT)$', label_str)
            if match:
                model, trainset, party, prediction = match.groups()
                if prediction == 'IS':
                    predictions[row_id][model]['political_parties'].add(party)
                continue

            # Handle specific themes (can have underscores too)
            match = re.match(r'^(\w+?)_specific_themes_(large|small)_(.+)$', label_str)
            if match:
                model, trainset, specific = match.groups()
                predictions[row_id][model]['specific_themes'].add(specific)
                continue

    return predictions


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    base_path = Path(__file__).parent.parent.parent

    # Paths to the two annotation folders
    large_folder = base_path / 'logs' / 'annotation_studio' / 'large_benchmark_large_20260123_105620'
    small_folder = base_path / 'logs' / 'annotation_studio' / 'small_benchmark_large_20260123_081645'

    output_path = base_path / 'paper' / 'results' / 'small_vs_large_comparison'
    figures_path = base_path / 'paper' / 'figures'

    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SMALL VS LARGE TRAINING SET COMPARISON")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    large_df = pd.read_csv(large_folder / 'annotations.csv')
    small_df = pd.read_csv(small_folder / 'annotations.csv')

    print(f"  - Large training models: {len(large_df)} samples")
    print(f"  - Small training models: {len(small_df)} samples")

    # Extract predictions
    print("\n[2/5] Extracting predictions...")
    large_preds = extract_predictions_from_labels(large_df)
    small_preds = extract_predictions_from_labels(small_df)

    # Get unique models from each
    large_models = set()
    small_models = set()

    for row_preds in large_preds.values():
        large_models.update(row_preds.keys())
    for row_preds in small_preds.values():
        small_models.update(row_preds.keys())

    print(f"  - Models in LARGE training: {sorted(large_models)}")
    print(f"  - Models in SMALL training: {sorted(small_models)}")

    # All unique models
    all_models = sorted(large_models | small_models)

    # Compute metrics for themes
    print("\n[3/5] Computing theme metrics...")

    results = []

    for model in all_models:
        for training_set, df, preds in [('Large', large_df, large_preds), ('Small', small_df, small_preds)]:

            # Check if model exists in this training set
            model_exists = any(model in row_preds for row_preds in preds.values())
            if not model_exists:
                continue

            # Overall metrics
            tp, fp, fn = 0, 0, 0

            for idx, row in df.iterrows():
                row_id = row['id']
                consensus = safe_parse(row['consensus_themes'])

                if row_id in preds and model in preds[row_id]:
                    pred_themes = preds[row_id][model].get('themes', set())
                else:
                    pred_themes = set()

                tp += len(pred_themes & consensus)
                fp += len(pred_themes - consensus)
                fn += len(consensus - pred_themes)

            precision, recall, f1 = compute_metrics(tp, fp, fn)

            results.append({
                'model': model,
                'training_set': training_set,
                'category': 'themes',
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            })

    # Compute human baseline
    print("\n[4/5] Computing human baseline...")

    # Use large_df for human annotations (both should have same human data)
    human_tp, human_fp, human_fn = 0, 0, 0

    for idx, row in large_df.iterrows():
        consensus = safe_parse(row['consensus_themes'])

        for annotator in ANNOTATORS:
            ann_themes = safe_parse(row[f'{annotator}_themes'])
            human_tp += len(ann_themes & consensus)
            human_fp += len(ann_themes - consensus)
            human_fn += len(consensus - ann_themes)

    human_prec, human_rec, human_f1 = compute_metrics(human_tp, human_fp, human_fn)

    results.append({
        'model': 'Human',
        'training_set': 'N/A',
        'category': 'themes',
        'tp': human_tp,
        'fp': human_fp,
        'fn': human_fn,
        'precision': human_prec,
        'recall': human_rec,
        'f1': human_f1,
        'support': human_tp + human_fn
    })

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df['pct_human'] = results_df['f1'] / human_f1 * 100

    # Save CSV
    results_df.to_csv(output_path / 'small_vs_large_comparison.csv', index=False)

    print(f"\n  Human baseline F1: {human_f1:.3f}")

    # Print summary
    print("\n[5/5] Generating figures...")

    # Get models that exist in both training sets for comparison
    models_in_both = []
    for model in all_models:
        in_large = len(results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Large')]) > 0
        in_small = len(results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Small')]) > 0
        if in_large and in_small:
            models_in_both.append(model)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel (a): Grouped bar chart - Small vs Large by model
    ax1 = axes[0, 0]

    x = np.arange(len(all_models))
    width = 0.35

    large_f1 = []
    small_f1 = []

    for model in all_models:
        large_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Large')]
        small_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Small')]

        large_f1.append(large_data['f1'].values[0] if len(large_data) > 0 else 0)
        small_f1.append(small_data['f1'].values[0] if len(small_data) > 0 else 0)

    bars1 = ax1.bar(x - width/2, large_f1, width, label='Large Training', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, small_f1, width, label='Small Training', color='#e74c3c', edgecolor='black', linewidth=0.5)

    # Human baseline line
    ax1.axhline(y=human_f1, color='#3498db', linestyle='--', linewidth=2, label=f'Human ({human_f1:.3f})')

    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper().replace('_', '-') for m in all_models], fontsize=10, rotation=45, ha='right')
    ax1.set_ylabel('Micro-F1 Score', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_title(f'(a) Small vs Large Training Set Comparison\n(n = {len(large_df)} sentences, k = {len(ALL_THEMES)} themes)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, max(max(large_f1), max(small_f1), human_f1) * 1.15)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    # Add value labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    for bar in bars2:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # Panel (b): Difference (Large - Small) for models in both
    ax2 = axes[0, 1]

    if models_in_both:
        x2 = np.arange(len(models_in_both))
        diff = []
        for model in models_in_both:
            large_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Large')]
            small_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Small')]
            l_f1 = large_data['f1'].values[0] if len(large_data) > 0 else 0
            s_f1 = small_data['f1'].values[0] if len(small_data) > 0 else 0
            diff.append(l_f1 - s_f1)

        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]
        bars = ax2.bar(x2, diff, width=0.6, color=colors, edgecolor='black', linewidth=0.5)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([m.upper().replace('_', '-') for m in models_in_both], fontsize=10, rotation=45, ha='right')
        ax2.set_ylabel('F1 Difference (Large - Small)', fontsize=12)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_title('(b) Performance Gain from Large Training Set\n(Positive = Large is better)', fontsize=13, fontweight='bold')
        ax2.yaxis.grid(True, linestyle='--', alpha=0.5)

        for bar, d in zip(bars, diff):
            va = 'bottom' if d >= 0 else 'top'
            offset = 0.002 if d >= 0 else -0.002
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f'{d:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No models in both training sets', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(b) Performance Gain from Large Training Set', fontsize=13, fontweight='bold')

    # Panel (c): % of Human performance
    ax3 = axes[1, 0]

    large_pct = [results_df[(results_df['model'] == m) & (results_df['training_set'] == 'Large')]['pct_human'].values[0]
                 if len(results_df[(results_df['model'] == m) & (results_df['training_set'] == 'Large')]) > 0 else 0
                 for m in all_models]
    small_pct = [results_df[(results_df['model'] == m) & (results_df['training_set'] == 'Small')]['pct_human'].values[0]
                 if len(results_df[(results_df['model'] == m) & (results_df['training_set'] == 'Small')]) > 0 else 0
                 for m in all_models]

    bars1 = ax3.bar(x - width/2, large_pct, width, label='Large Training', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, small_pct, width, label='Small Training', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax3.axhline(y=100, color='#3498db', linestyle='--', linewidth=2, label='Human (100%)')

    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper().replace('_', '-') for m in all_models], fontsize=10, rotation=45, ha='right')
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
    for model in all_models:
        large_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Large')]
        small_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Small')]

        l_f1 = large_data['f1'].values[0] if len(large_data) > 0 else None
        s_f1 = small_data['f1'].values[0] if len(small_data) > 0 else None
        l_pct = large_data['pct_human'].values[0] if len(large_data) > 0 else None
        s_pct = small_data['pct_human'].values[0] if len(small_data) > 0 else None

        diff = (l_f1 - s_f1) if (l_f1 is not None and s_f1 is not None) else None

        table_data.append([
            model.upper().replace('_', '-'),
            f'{l_f1:.3f}' if l_f1 is not None else '-',
            f'{s_f1:.3f}' if s_f1 is not None else '-',
            f'{diff:+.3f}' if diff is not None else '-',
            f'{l_pct:.1f}%' if l_pct is not None else '-',
            f'{s_pct:.1f}%' if s_pct is not None else '-'
        ])

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

    ax4.set_title(f'(d) Summary: Small vs Large Training Comparison\n(n = {len(large_df)} sentences)',
                  fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(figures_path / 'fig10_small_vs_large_training.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_path / 'fig10_small_vs_large_training.pdf', bbox_inches='tight')
    plt.close()

    print(f"  - Figure 10: Small vs Large training comparison")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nHuman Baseline: F1 = {human_f1:.3f}")
    print(f"\nModel Performance (Themes Micro-F1):")
    print(f"{'Model':<15} {'Large F1':>10} {'Small F1':>10} {'Diff':>10} {'Large %':>10} {'Small %':>10}")
    print("-" * 70)

    for model in all_models:
        large_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Large')]
        small_data = results_df[(results_df['model'] == model) & (results_df['training_set'] == 'Small')]

        l_f1 = large_data['f1'].values[0] if len(large_data) > 0 else None
        s_f1 = small_data['f1'].values[0] if len(small_data) > 0 else None
        l_pct = large_data['pct_human'].values[0] if len(large_data) > 0 else None
        s_pct = small_data['pct_human'].values[0] if len(small_data) > 0 else None
        diff = (l_f1 - s_f1) if (l_f1 is not None and s_f1 is not None) else None

        print(f"{model.upper().replace('_', '-'):<15} "
              f"{l_f1:>10.3f}" if l_f1 else f"{'':>10} "
              f"{s_f1:>10.3f}" if s_f1 else f"{'-':>10} "
              f"{diff:>+10.3f}" if diff else f"{'-':>10} "
              f"{l_pct:>9.1f}%" if l_pct else f"{'-':>10} "
              f"{s_pct:>9.1f}%" if s_pct else f"{'-':>10}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Figure saved to: {figures_path / 'fig10_small_vs_large_training.png'}")
    print("=" * 80)


if __name__ == '__main__':
    main()

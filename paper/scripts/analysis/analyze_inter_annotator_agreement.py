#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
analyze_inter_annotator_agreement.py

MAIN OBJECTIVE:
---------------
Comprehensive analysis of inter-annotator agreement and model performance
stratified by annotation ambiguity levels. Computes agreement metrics between
human annotators and evaluates model performance on samples grouped by how
much annotators agreed.

Dependencies:
-------------
- pandas
- numpy
- json
- collections
- itertools
- pathlib

MAIN FEATURES:
--------------
1) Inter-annotator agreement metrics (Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha)
2) Model performance by agreement level (micro-F1, properly handling empty cases)
3) Per-theme detailed analysis with support counts
4) Human baseline comparisons
5) Statistical summaries and CSV exports

OUTPUT FILES:
-------------
- paper/results/evaluation_results/comprehensive_agreement_analysis.csv

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
import json
import ast
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

ANNOTATORS = ['shdin', 'Jeremy', 'jdrouin', 'BenjaminCarignan']

MODELS = [
    'gpt_4_1_large', 'gpt_5_large', 'gpt_oss_large',
    'gemma_large', 'llama_large', 'nemotron_large', 'manual_manual'
]

MODEL_DISPLAY_NAMES = {
    'gpt_4_1_large': 'GPT-4.1',
    'gpt_5_large': 'GPT-5',
    'gpt_oss_large': 'GPT-OSS',
    'gemma_large': 'Gemma',
    'llama_large': 'Llama',
    'nemotron_large': 'Nemotron',
    'manual_manual': 'Manual'
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

# Agreement thresholds (Landis & Koch, 1977)
AGREEMENT_THRESHOLDS = {
    'almost_perfect': 0.81,
    'substantial': 0.61,
    'moderate': 0.41,
    'fair': 0.21,
    'slight': 0.0
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


def cohens_kappa(y1, y2):
    """Compute Cohen's Kappa for two binary arrays."""
    n = len(y1)
    if n == 0:
        return np.nan

    # Observed agreement
    po = sum(1 for i in range(n) if y1[i] == y2[i]) / n

    # Expected agreement
    p1_yes = sum(y1) / n
    p2_yes = sum(y2) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    if pe == 1:
        return 1.0

    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_matrix):
    """
    Compute Fleiss' Kappa for multiple annotators.
    ratings_matrix: n_subjects x n_categories
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = int(ratings_matrix.sum(axis=1).mean())

    if n_raters <= 1:
        return np.nan

    # Proportion of assignments to each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)

    # Agreement per subject
    P_i = ((ratings_matrix ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_e >= 1:
        return 1.0

    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha_binary(data_matrix):
    """
    Compute Krippendorff's Alpha for binary data.
    data_matrix: n_annotators x n_items
    """
    n_annotators, n_items = data_matrix.shape

    # Count observed disagreement
    Do = 0
    n_pairs = 0

    for item in range(n_items):
        values = [data_matrix[a, item] for a in range(n_annotators)
                  if not np.isnan(data_matrix[a, item])]
        n = len(values)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i+1, n):
                if values[i] != values[j]:
                    Do += 1
                n_pairs += 1

    if n_pairs == 0:
        return 1.0

    Do = Do / n_pairs

    # Expected disagreement
    all_values = data_matrix[~np.isnan(data_matrix)]
    if len(all_values) == 0:
        return np.nan
    p1 = all_values.mean()
    De = 2 * p1 * (1 - p1)

    if De == 0:
        return 1.0

    return 1 - Do / De


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    support = tp + fn
    return precision, recall, f1, support


def get_agreement_level(kappa):
    """Classify kappa into agreement level."""
    if kappa >= AGREEMENT_THRESHOLDS['almost_perfect']:
        return 'Almost Perfect'
    elif kappa >= AGREEMENT_THRESHOLDS['substantial']:
        return 'Substantial'
    elif kappa >= AGREEMENT_THRESHOLDS['moderate']:
        return 'Moderate'
    elif kappa >= AGREEMENT_THRESHOLDS['fair']:
        return 'Fair'
    else:
        return 'Slight'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(base_path):
    """Load all required data files."""
    data_path = base_path / 'data' / 'sets'

    annotators_df = pd.read_csv(data_path / 'benchmark_annotations_by_annotator.csv')
    annotated_df = pd.read_csv(data_path / 'final_test_small_annotated.csv')

    # Create ID mapping
    id_to_annotators = {row['id']: row for _, row in annotators_df.iterrows()}

    return annotators_df, annotated_df, id_to_annotators


# =============================================================================
# INTER-ANNOTATOR AGREEMENT ANALYSIS
# =============================================================================

def compute_theme_agreement(annotators_df):
    """Compute agreement metrics for each theme."""
    results = []

    for theme in ALL_THEMES:
        theme_name = theme.replace('theme_', '')

        # Build binary arrays for each annotator
        annotator_arrays = {}
        for ann in ANNOTATORS:
            arr = []
            for _, row in annotators_df.iterrows():
                themes = safe_parse(row[f'{ann}_themes'])
                arr.append(1 if theme in themes else 0)
            annotator_arrays[ann] = arr

        # Cohen's Kappa (pairwise)
        kappas = []
        for ann1, ann2 in combinations(ANNOTATORS, 2):
            k = cohens_kappa(annotator_arrays[ann1], annotator_arrays[ann2])
            if not np.isnan(k):
                kappas.append(k)
        avg_cohen_kappa = np.mean(kappas) if kappas else np.nan

        # Fleiss' Kappa
        ratings = []
        for _, row in annotators_df.iterrows():
            present = sum(1 for ann in ANNOTATORS if theme in safe_parse(row[f'{ann}_themes']))
            absent = len(ANNOTATORS) - present
            ratings.append([absent, present])
        fleiss_k = fleiss_kappa(np.array(ratings))

        # Krippendorff's Alpha
        data = np.zeros((len(ANNOTATORS), len(annotators_df)))
        for a_idx, ann in enumerate(ANNOTATORS):
            for i, (_, row) in enumerate(annotators_df.iterrows()):
                themes = safe_parse(row[f'{ann}_themes'])
                data[a_idx, i] = 1 if theme in themes else 0
        kripp_alpha = krippendorff_alpha_binary(data)

        # Support (in consensus)
        support_consensus = sum(
            1 for _, row in annotators_df.iterrows()
            if theme in safe_parse(row['consensus_themes'])
        )

        # Support per annotator
        support_per_annotator = {
            ann: sum(1 for _, row in annotators_df.iterrows()
                    if theme in safe_parse(row[f'{ann}_themes']))
            for ann in ANNOTATORS
        }
        avg_support = np.mean(list(support_per_annotator.values()))

        # Observed agreement (% of items where all annotators agree)
        full_agreement = 0
        for _, row in annotators_df.iterrows():
            ann_values = [1 if theme in safe_parse(row[f'{ann}_themes']) else 0 for ann in ANNOTATORS]
            if len(set(ann_values)) == 1:  # All same
                full_agreement += 1
        pct_full_agreement = full_agreement / len(annotators_df) * 100

        results.append({
            'Theme': theme_name,
            'Cohen_Kappa_Avg': round(avg_cohen_kappa, 4),
            'Fleiss_Kappa': round(fleiss_k, 4),
            'Krippendorff_Alpha': round(kripp_alpha, 4),
            'Agreement_Level': get_agreement_level(fleiss_k),
            'Support_Consensus': support_consensus,
            'Support_Avg_Annotators': round(avg_support, 1),
            'Pct_Full_Agreement': round(pct_full_agreement, 1),
            **{f'Support_{ann}': support_per_annotator[ann] for ann in ANNOTATORS}
        })

    return pd.DataFrame(results).sort_values('Fleiss_Kappa', ascending=False)


def compute_pairwise_agreement(annotators_df):
    """Compute pairwise agreement between annotators."""
    results = []

    for ann1, ann2 in combinations(ANNOTATORS, 2):
        # Theme agreement
        theme_kappas = []
        for theme in ALL_THEMES:
            y1 = [1 if theme in safe_parse(row[f'{ann1}_themes']) else 0
                  for _, row in annotators_df.iterrows()]
            y2 = [1 if theme in safe_parse(row[f'{ann2}_themes']) else 0
                  for _, row in annotators_df.iterrows()]
            k = cohens_kappa(y1, y2)
            if not np.isnan(k):
                theme_kappas.append(k)

        # Set-based F1
        f1s = []
        for _, row in annotators_df.iterrows():
            t1 = safe_parse(row[f'{ann1}_themes'])
            t2 = safe_parse(row[f'{ann2}_themes'])
            if not t1 and not t2:
                f1s.append(1.0)
            elif not t1 or not t2:
                f1s.append(0.0)
            else:
                tp = len(t1 & t2)
                p = tp / len(t1)
                r = tp / len(t2)
                f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0)

        # Sentiment agreement
        sent_agree = 0
        sent_total = 0
        for _, row in annotators_df.iterrows():
            s1 = row[f'{ann1}_sentiment']
            s2 = row[f'{ann2}_sentiment']
            if pd.notna(s1) and pd.notna(s2) and s1 and s2:
                sent_total += 1
                if s1 == s2:
                    sent_agree += 1

        results.append({
            'Annotator_1': ann1,
            'Annotator_2': ann2,
            'Themes_Cohen_Kappa_Avg': round(np.mean(theme_kappas), 4),
            'Themes_F1_Macro': round(np.mean(f1s), 4),
            'Sentiment_Agreement': round(sent_agree / sent_total, 4) if sent_total > 0 else np.nan,
            'N_Samples': len(annotators_df)
        })

    return pd.DataFrame(results)


# =============================================================================
# MODEL PERFORMANCE ANALYSIS
# =============================================================================

def compute_model_performance_by_theme(annotated_df, id_to_annotators, theme_agreement_df):
    """Compute model performance for each theme."""
    # Create theme to agreement level mapping
    theme_to_kappa = dict(zip(
        'theme_' + theme_agreement_df['Theme'],
        theme_agreement_df['Fleiss_Kappa']
    ))
    theme_to_level = dict(zip(
        'theme_' + theme_agreement_df['Theme'],
        theme_agreement_df['Agreement_Level']
    ))

    results = []

    for theme in ALL_THEMES:
        theme_name = theme.replace('theme_', '')
        kappa = theme_to_kappa.get(theme, np.nan)
        level = theme_to_level.get(theme, 'Unknown')

        # Ground truth support
        gt_support = sum(
            1 for _, row in annotated_df.iterrows()
            if row['id'] in id_to_annotators and
            theme in safe_parse(id_to_annotators[row['id']]['consensus_themes'])
        )

        for model in MODELS:
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

            precision, recall, f1, support = compute_metrics(tp, fp, fn)

            results.append({
                'Theme': theme_name,
                'Model': MODEL_DISPLAY_NAMES.get(model, model),
                'Fleiss_Kappa': round(kappa, 4),
                'Agreement_Level': level,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Support': support,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1': round(f1, 4)
            })

    return pd.DataFrame(results)


def compute_model_performance_by_agreement_level(annotated_df, id_to_annotators, theme_agreement_df):
    """Compute aggregated model performance by agreement level."""
    # Classify themes by agreement level
    theme_levels = {}
    for _, row in theme_agreement_df.iterrows():
        theme = 'theme_' + row['Theme']
        theme_levels[theme] = row['Agreement_Level']

    levels = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Slight']

    results = []

    for model in MODELS:
        model_results = {'Model': MODEL_DISPLAY_NAMES.get(model, model)}

        # Overall metrics
        overall_tp, overall_fp, overall_fn = 0, 0, 0

        for level in levels:
            level_themes = [t for t, l in theme_levels.items() if l == level]
            if not level_themes:
                continue

            tp, fp, fn = 0, 0, 0
            n_both_empty = 0
            n_samples = 0

            for _, row in annotated_df.iterrows():
                row_id = row['id']
                if row_id not in id_to_annotators:
                    continue

                n_samples += 1
                ann_row = id_to_annotators[row_id]
                all_ann = json.loads(row['all_annotations'])
                pipelines = all_ann.get('pipelines', {})

                consensus = safe_parse(ann_row['consensus_themes']) & set(level_themes)
                pred = set(pipelines.get(model, {}).get('themes', [])) & set(level_themes)

                if not consensus and not pred:
                    n_both_empty += 1

                tp += len(pred & consensus)
                fp += len(pred - consensus)
                fn += len(consensus - pred)

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

            precision, recall, f1, support = compute_metrics(tp, fp, fn)
            level_key = level.replace(' ', '_')

            model_results[f'{level_key}_TP'] = tp
            model_results[f'{level_key}_FP'] = fp
            model_results[f'{level_key}_FN'] = fn
            model_results[f'{level_key}_Support'] = support
            model_results[f'{level_key}_Precision'] = round(precision, 4)
            model_results[f'{level_key}_Recall'] = round(recall, 4)
            model_results[f'{level_key}_F1'] = round(f1, 4)
            model_results[f'{level_key}_Pct_Both_Empty'] = round(n_both_empty / n_samples * 100, 1) if n_samples > 0 else 0
            model_results[f'{level_key}_N_Themes'] = len(level_themes)

        # Overall
        precision, recall, f1, support = compute_metrics(overall_tp, overall_fp, overall_fn)
        model_results['Overall_TP'] = overall_tp
        model_results['Overall_FP'] = overall_fp
        model_results['Overall_FN'] = overall_fn
        model_results['Overall_Support'] = support
        model_results['Overall_Precision'] = round(precision, 4)
        model_results['Overall_Recall'] = round(recall, 4)
        model_results['Overall_F1'] = round(f1, 4)

        results.append(model_results)

    return pd.DataFrame(results)


def compute_human_baseline_by_level(annotators_df, theme_agreement_df):
    """Compute human baseline (annotator vs consensus) by agreement level."""
    theme_levels = {}
    for _, row in theme_agreement_df.iterrows():
        theme = 'theme_' + row['Theme']
        theme_levels[theme] = row['Agreement_Level']

    levels = ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Slight']

    results = []

    for annotator in ANNOTATORS:
        ann_results = {'Annotator': annotator}

        overall_tp, overall_fp, overall_fn = 0, 0, 0

        for level in levels:
            level_themes = [t for t, l in theme_levels.items() if l == level]
            if not level_themes:
                continue

            tp, fp, fn = 0, 0, 0

            for _, row in annotators_df.iterrows():
                consensus = safe_parse(row['consensus_themes']) & set(level_themes)
                ann_themes = safe_parse(row[f'{annotator}_themes']) & set(level_themes)

                tp += len(ann_themes & consensus)
                fp += len(ann_themes - consensus)
                fn += len(consensus - ann_themes)

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

            precision, recall, f1, support = compute_metrics(tp, fp, fn)
            level_key = level.replace(' ', '_')

            ann_results[f'{level_key}_F1'] = round(f1, 4)
            ann_results[f'{level_key}_Support'] = support

        # Overall
        precision, recall, f1, support = compute_metrics(overall_tp, overall_fp, overall_fn)
        ann_results['Overall_F1'] = round(f1, 4)
        ann_results['Overall_Support'] = support

        results.append(ann_results)

    # Add average
    df = pd.DataFrame(results)
    avg_row = {'Annotator': 'AVERAGE'}
    for col in df.columns:
        if col != 'Annotator':
            avg_row[col] = round(df[col].mean(), 4)
    results.append(avg_row)

    return pd.DataFrame(results)


def compute_model_vs_each_annotator(annotated_df, id_to_annotators):
    """Compute model performance against each individual annotator."""
    results = []

    for model in MODELS:
        for annotator in ANNOTATORS:
            tp, fp, fn = 0, 0, 0

            for _, row in annotated_df.iterrows():
                row_id = row['id']
                if row_id not in id_to_annotators:
                    continue

                ann_row = id_to_annotators[row_id]
                all_ann = json.loads(row['all_annotations'])
                pipelines = all_ann.get('pipelines', {})

                ann_themes = safe_parse(ann_row[f'{annotator}_themes'])
                pred = set(pipelines.get(model, {}).get('themes', []))

                tp += len(pred & ann_themes)
                fp += len(pred - ann_themes)
                fn += len(ann_themes - pred)

            precision, recall, f1, support = compute_metrics(tp, fp, fn)

            results.append({
                'Model': MODEL_DISPLAY_NAMES.get(model, model),
                'Annotator': annotator,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Support': support,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1': round(f1, 4)
            })

    return pd.DataFrame(results)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary_statistics(theme_agreement_df, model_by_level_df, human_baseline_df):
    """Compute summary statistics for the analysis."""
    results = []

    # Agreement level distribution
    level_counts = theme_agreement_df['Agreement_Level'].value_counts()
    for level, count in level_counts.items():
        results.append({
            'Category': 'Theme Distribution',
            'Metric': f'N Themes - {level}',
            'Value': count
        })

    # Overall agreement stats
    results.append({
        'Category': 'Inter-Annotator Agreement',
        'Metric': 'Mean Fleiss Kappa',
        'Value': round(theme_agreement_df['Fleiss_Kappa'].mean(), 4)
    })
    results.append({
        'Category': 'Inter-Annotator Agreement',
        'Metric': 'Std Fleiss Kappa',
        'Value': round(theme_agreement_df['Fleiss_Kappa'].std(), 4)
    })
    results.append({
        'Category': 'Inter-Annotator Agreement',
        'Metric': 'Min Fleiss Kappa',
        'Value': round(theme_agreement_df['Fleiss_Kappa'].min(), 4)
    })
    results.append({
        'Category': 'Inter-Annotator Agreement',
        'Metric': 'Max Fleiss Kappa',
        'Value': round(theme_agreement_df['Fleiss_Kappa'].max(), 4)
    })

    # Human baseline
    human_avg = human_baseline_df[human_baseline_df['Annotator'] == 'AVERAGE'].iloc[0]
    results.append({
        'Category': 'Human Baseline',
        'Metric': 'Overall F1 (Annotator vs Consensus)',
        'Value': human_avg['Overall_F1']
    })

    # Best model performance
    best_model_f1 = model_by_level_df['Overall_F1'].max()
    best_model = model_by_level_df.loc[model_by_level_df['Overall_F1'].idxmax(), 'Model']
    results.append({
        'Category': 'Model Performance',
        'Metric': f'Best Model F1 ({best_model})',
        'Value': best_model_f1
    })

    # Ratio best model / human
    results.append({
        'Category': 'Model Performance',
        'Metric': 'Best Model / Human Baseline Ratio',
        'Value': round(best_model_f1 / human_avg['Overall_F1'], 4)
    })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis function."""
    # Setup paths
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'paper' / 'results' / 'evaluation_results'
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[1/7] Loading data...")
    annotators_df, annotated_df, id_to_annotators = load_data(base_path)
    print(f"  - Loaded {len(annotators_df)} annotated samples")
    print(f"  - {len(ANNOTATORS)} annotators: {', '.join(ANNOTATORS)}")
    print(f"  - {len(MODELS)} models to evaluate")

    # Compute theme agreement
    print("\n[2/7] Computing inter-annotator agreement per theme...")
    theme_agreement_df = compute_theme_agreement(annotators_df)
    print(f"  - Analyzed {len(ALL_THEMES)} themes")
    print(f"  - Mean Fleiss Kappa: {theme_agreement_df['Fleiss_Kappa'].mean():.3f}")

    # Compute pairwise agreement
    print("\n[3/7] Computing pairwise annotator agreement...")
    pairwise_df = compute_pairwise_agreement(annotators_df)
    print(f"  - {len(pairwise_df)} annotator pairs analyzed")

    # Compute model performance by theme
    print("\n[4/7] Computing model performance per theme...")
    model_theme_df = compute_model_performance_by_theme(
        annotated_df, id_to_annotators, theme_agreement_df
    )
    print(f"  - {len(model_theme_df)} model-theme combinations")

    # Compute model performance by agreement level
    print("\n[5/7] Computing model performance by agreement level...")
    model_level_df = compute_model_performance_by_agreement_level(
        annotated_df, id_to_annotators, theme_agreement_df
    )

    # Compute human baseline
    print("\n[6/7] Computing human baseline by agreement level...")
    human_baseline_df = compute_human_baseline_by_level(annotators_df, theme_agreement_df)

    # Compute model vs each annotator
    print("\n[7/7] Computing model vs each annotator...")
    model_annotator_df = compute_model_vs_each_annotator(annotated_df, id_to_annotators)

    # Compute summary
    summary_df = compute_summary_statistics(theme_agreement_df, model_level_df, human_baseline_df)

    # Save all results to a single Excel file with multiple sheets
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save individual CSVs
    theme_agreement_df.to_csv(output_path / 'agreement_per_theme.csv', index=False)
    pairwise_df.to_csv(output_path / 'agreement_pairwise.csv', index=False)
    model_theme_df.to_csv(output_path / 'model_performance_per_theme.csv', index=False)
    model_level_df.to_csv(output_path / 'model_performance_by_level.csv', index=False)
    human_baseline_df.to_csv(output_path / 'human_baseline_by_level.csv', index=False)
    model_annotator_df.to_csv(output_path / 'model_vs_annotator.csv', index=False)
    summary_df.to_csv(output_path / 'analysis_summary.csv', index=False)

    print(f"  - agreement_per_theme.csv")
    print(f"  - agreement_pairwise.csv")
    print(f"  - model_performance_per_theme.csv")
    print(f"  - model_performance_by_level.csv")
    print(f"  - human_baseline_by_level.csv")
    print(f"  - model_vs_annotator.csv")
    print(f"  - analysis_summary.csv")

    # Create comprehensive CSV
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE CSV")
    print("=" * 80)

    # Build comprehensive results
    comprehensive_rows = []

    # Section 1: Theme Agreement
    comprehensive_rows.append({'Section': 'INTER-ANNOTATOR AGREEMENT PER THEME', 'Metric': '', 'Value': ''})
    for _, row in theme_agreement_df.iterrows():
        comprehensive_rows.append({
            'Section': 'Theme Agreement',
            'Metric': row['Theme'],
            'Fleiss_Kappa': row['Fleiss_Kappa'],
            'Krippendorff_Alpha': row['Krippendorff_Alpha'],
            'Agreement_Level': row['Agreement_Level'],
            'Support_Consensus': row['Support_Consensus'],
            'Pct_Full_Agreement': row['Pct_Full_Agreement']
        })

    # Section 2: Model Performance by Level
    comprehensive_rows.append({'Section': '', 'Metric': '', 'Value': ''})
    comprehensive_rows.append({'Section': 'MODEL PERFORMANCE BY AGREEMENT LEVEL (Micro-F1)', 'Metric': '', 'Value': ''})

    levels = ['Overall', 'Almost_Perfect', 'Substantial', 'Moderate', 'Fair', 'Slight']
    for _, row in model_level_df.iterrows():
        for level in levels:
            f1_col = f'{level}_F1'
            support_col = f'{level}_Support'
            if f1_col in row and pd.notna(row.get(f1_col)):
                comprehensive_rows.append({
                    'Section': 'Model Performance',
                    'Model': row['Model'],
                    'Agreement_Level': level.replace('_', ' '),
                    'F1': row.get(f1_col, ''),
                    'Support': row.get(support_col, ''),
                    'TP': row.get(f'{level}_TP', ''),
                    'FP': row.get(f'{level}_FP', ''),
                    'FN': row.get(f'{level}_FN', '')
                })

    # Section 3: Human Baseline
    comprehensive_rows.append({'Section': '', 'Metric': '', 'Value': ''})
    comprehensive_rows.append({'Section': 'HUMAN BASELINE (Annotator vs Consensus)', 'Metric': '', 'Value': ''})
    for _, row in human_baseline_df.iterrows():
        for level in levels:
            f1_col = f'{level}_F1'
            if f1_col in row and pd.notna(row.get(f1_col)):
                comprehensive_rows.append({
                    'Section': 'Human Baseline',
                    'Annotator': row['Annotator'],
                    'Agreement_Level': level.replace('_', ' '),
                    'F1': row.get(f1_col, ''),
                    'Support': row.get(f'{level}_Support', '')
                })

    comprehensive_df = pd.DataFrame(comprehensive_rows)
    comprehensive_df.to_csv(output_path / 'comprehensive_agreement_analysis.csv', index=False)
    print(f"  - comprehensive_agreement_analysis.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nInter-Annotator Agreement (Fleiss' Kappa):")
    print(f"  Mean: {theme_agreement_df['Fleiss_Kappa'].mean():.3f}")
    print(f"  Range: [{theme_agreement_df['Fleiss_Kappa'].min():.3f}, {theme_agreement_df['Fleiss_Kappa'].max():.3f}]")

    print("\nThemes by Agreement Level:")
    for level in ['Almost Perfect', 'Substantial', 'Moderate', 'Fair', 'Slight']:
        count = len(theme_agreement_df[theme_agreement_df['Agreement_Level'] == level])
        if count > 0:
            themes = theme_agreement_df[theme_agreement_df['Agreement_Level'] == level]['Theme'].tolist()
            print(f"  {level}: {count} themes")

    print("\nHuman Baseline (Annotator vs Consensus, Overall F1):")
    for _, row in human_baseline_df.iterrows():
        if row['Annotator'] != 'AVERAGE':
            print(f"  {row['Annotator']}: {row['Overall_F1']:.3f}")
    print(f"  AVERAGE: {human_baseline_df[human_baseline_df['Annotator'] == 'AVERAGE']['Overall_F1'].values[0]:.3f}")

    print("\nModel Performance (Overall Micro-F1):")
    for _, row in model_level_df.sort_values('Overall_F1', ascending=False).iterrows():
        human_avg = human_baseline_df[human_baseline_df['Annotator'] == 'AVERAGE']['Overall_F1'].values[0]
        ratio = row['Overall_F1'] / human_avg * 100
        print(f"  {row['Model']}: {row['Overall_F1']:.3f} ({ratio:.1f}% of human)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        'theme_agreement': theme_agreement_df,
        'pairwise': pairwise_df,
        'model_theme': model_theme_df,
        'model_level': model_level_df,
        'human_baseline': human_baseline_df,
        'model_annotator': model_annotator_df,
        'summary': summary_df
    }


if __name__ == '__main__':
    results = main()

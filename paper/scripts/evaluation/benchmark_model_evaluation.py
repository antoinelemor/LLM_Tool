#!/usr/bin/env python3
"""
Comprehensive Model Benchmark Evaluation
=========================================

Evaluates model performance against human annotations with SOTA metrics,
taking into account annotation ambiguity and inter-annotator disagreement.

Features:
- Standard classification metrics (F1, Precision, Recall, Accuracy)
- Inter-annotator agreement (Krippendorff's alpha, Fleiss' kappa)
- Human ceiling analysis (model vs annotator agreement)
- Ambiguity-aware evaluation (hard vs easy cases)
- Calibration analysis (ECE, reliability diagrams)
- Per-class performance breakdown
- Statistical significance tests

Author: Antoine Lemor
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import warnings
import json
import ast
from scipy import stats
from itertools import combinations

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


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cohen_kappa: float
    mcc: float  # Matthews Correlation Coefficient
    hamming_loss: float
    jaccard_score: float


def safe_eval(s):
    """Safely evaluate string representation of list."""
    if pd.isna(s) or s == '' or s == '[]':
        return []
    if isinstance(s, list):
        return s
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else [result]
    except:
        return []


def extract_theme_name(theme: str) -> str:
    """Extract clean theme name from full label."""
    if not theme:
        return ''
    # Remove all known prefixes (order matters - longest first)
    prefixes = [
        'consensus_themes_extra_large_',
        'consensus_themes_large_',
        'consensus_themes_',
        'theme_',
        'specific_themes_',
    ]
    theme_lower = theme.lower()
    for prefix in prefixes:
        if theme_lower.startswith(prefix):
            theme_lower = theme_lower[len(prefix):]
            break
    return theme_lower


def parse_annotations(df: pd.DataFrame, column: str) -> List[Set[str]]:
    """Parse annotation column into list of sets of themes."""
    annotations = []
    for val in df[column]:
        themes = safe_eval(val)
        clean_themes = {extract_theme_name(t) for t in themes if t}
        annotations.append(clean_themes)
    return annotations


def compute_classification_metrics(y_true: List[Set[str]],
                                   y_pred: List[Set[str]],
                                   all_labels: Set[str]) -> EvaluationResult:
    """Compute comprehensive classification metrics for multi-label setting."""
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        hamming_loss, jaccard_score, cohen_kappa_score, matthews_corrcoef
    )

    # Convert to binary matrix
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # For Cohen's kappa and MCC, flatten to single-label (majority class)
    # This is an approximation for multi-label
    y_true_flat = [list(s)[0] if s else 'none' for s in y_true]
    y_pred_flat = [list(s)[0] if s else 'none' for s in y_pred]
    all_flat_labels = sorted(set(y_true_flat) | set(y_pred_flat))
    label_to_idx = {l: i for i, l in enumerate(all_flat_labels)}
    y_true_idx = [label_to_idx[l] for l in y_true_flat]
    y_pred_idx = [label_to_idx[l] for l in y_pred_flat]

    return EvaluationResult(
        accuracy=accuracy_score(y_true_bin, y_pred_bin),
        precision_micro=precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
        recall_micro=recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
        f1_micro=f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
        precision_macro=precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        recall_macro=recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        f1_macro=f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
        cohen_kappa=cohen_kappa_score(y_true_idx, y_pred_idx),
        mcc=matthews_corrcoef(y_true_idx, y_pred_idx),
        hamming_loss=hamming_loss(y_true_bin, y_pred_bin),
        jaccard_score=jaccard_score(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    )


def compute_per_class_metrics(y_true: List[Set[str]],
                              y_pred: List[Set[str]],
                              all_labels: Set[str]) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1."""
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import precision_score, recall_score, f1_score

    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    results = {}
    for i, label in enumerate(mlb.classes_):
        tp = ((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1)).sum()
        fp = ((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1)).sum()
        fn = ((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0)).sum()
        tn = ((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = int(y_true_bin[:, i].sum())

        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }

    return results


def compute_fleiss_kappa(annotations: List[List[Set[str]]], all_labels: Set[str]) -> float:
    """Compute Fleiss' kappa for multiple annotators."""
    n_samples = len(annotations[0])
    n_annotators = len(annotations)
    n_categories = len(all_labels)
    label_to_idx = {l: i for i, l in enumerate(sorted(all_labels))}

    # Build category count matrix
    category_counts = np.zeros((n_samples, n_categories))

    for annotator_annotations in annotations:
        for i, themes in enumerate(annotator_annotations):
            for theme in themes:
                if theme in label_to_idx:
                    category_counts[i, label_to_idx[theme]] += 1

    # Normalize by number of annotators
    n = category_counts.sum(axis=1)
    n = np.where(n > 0, n, 1)  # Avoid division by zero

    # Compute P_i (agreement for each item)
    P_i = (np.sum(category_counts ** 2, axis=1) - n) / (n * (n - 1) + 1e-10)
    P_bar = np.mean(P_i)

    # Compute P_j (proportion of assignments to each category)
    p_j = np.sum(category_counts, axis=0) / (n_samples * n_annotators + 1e-10)
    P_e = np.sum(p_j ** 2)

    # Fleiss' kappa
    kappa = (P_bar - P_e) / (1 - P_e + 1e-10)

    return kappa


def compute_krippendorff_alpha(annotations: List[List[Set[str]]], all_labels: Set[str]) -> float:
    """Compute Krippendorff's alpha (nominal) for multiple annotators."""
    n_samples = len(annotations[0])
    n_annotators = len(annotations)
    label_to_idx = {l: i for i, l in enumerate(sorted(all_labels))}
    n_categories = len(all_labels)

    # Convert to primary label (first label or 'none')
    def primary_label(themes):
        themes_list = list(themes)
        return themes_list[0] if themes_list else 'none'

    all_labels_with_none = sorted(all_labels | {'none'})
    label_to_idx = {l: i for i, l in enumerate(all_labels_with_none)}
    n_categories = len(all_labels_with_none)

    # Build reliability data matrix (n_annotators x n_samples)
    data = np.zeros((n_annotators, n_samples), dtype=int)
    for a, annotator_annotations in enumerate(annotations):
        for i, themes in enumerate(annotator_annotations):
            data[a, i] = label_to_idx[primary_label(themes)]

    # Compute coincidence matrix
    coincidence = np.zeros((n_categories, n_categories))

    for u in range(n_samples):
        # Get values for this unit, excluding missing
        values = data[:, u]
        n_u = len(values)
        if n_u < 2:
            continue

        # Add to coincidence matrix
        for c1 in values:
            for c2 in values:
                if c1 != c2:
                    coincidence[c1, c2] += 1 / (n_u - 1)
                else:
                    coincidence[c1, c2] += 1 / (n_u - 1)

    # Compute observed and expected disagreement
    n_total = coincidence.sum()
    if n_total == 0:
        return 0.0

    # Marginal frequencies
    n_c = coincidence.sum(axis=1)

    # Observed disagreement (nominal: 0 if same, 1 if different)
    D_o = 0
    D_e = 0

    for c in range(n_categories):
        for k in range(n_categories):
            if c != k:
                D_o += coincidence[c, k]
                D_e += n_c[c] * n_c[k]

    D_e /= (n_total - 1) if n_total > 1 else 1

    if D_e == 0:
        return 1.0

    alpha = 1 - D_o / D_e

    return alpha


def compute_pairwise_agreement(ann1: List[Set[str]], ann2: List[Set[str]]) -> Dict[str, float]:
    """Compute pairwise agreement metrics between two annotators."""
    n = len(ann1)

    # Exact match
    exact_matches = sum(1 for a, b in zip(ann1, ann2) if a == b)
    exact_agreement = exact_matches / n if n > 0 else 0

    # Jaccard similarity (set overlap)
    jaccard_scores = []
    for a, b in zip(ann1, ann2):
        if not a and not b:
            jaccard_scores.append(1.0)
        elif not a or not b:
            jaccard_scores.append(0.0)
        else:
            intersection = len(a & b)
            union = len(a | b)
            jaccard_scores.append(intersection / union if union > 0 else 0)

    mean_jaccard = np.mean(jaccard_scores)

    # Partial match (at least one label in common)
    partial_matches = sum(1 for a, b in zip(ann1, ann2) if a & b)
    partial_agreement = partial_matches / n if n > 0 else 0

    return {
        'exact_agreement': exact_agreement,
        'jaccard_similarity': mean_jaccard,
        'partial_agreement': partial_agreement
    }


def compute_annotator_disagreement_index(annotator_annotations: List[List[Set[str]]]) -> np.ndarray:
    """Compute disagreement index for each sample (0 = full agreement, 1 = full disagreement)."""
    n_samples = len(annotator_annotations[0])
    n_annotators = len(annotator_annotations)

    disagreement = np.zeros(n_samples)

    for i in range(n_samples):
        # Get all annotations for this sample
        all_themes = [ann[i] for ann in annotator_annotations]
        non_empty = [t for t in all_themes if t]

        if len(non_empty) < 2:
            disagreement[i] = 0
            continue

        # Compute pairwise Jaccard and take 1 - mean
        pairwise_jaccards = []
        for a, b in combinations(non_empty, 2):
            if a or b:
                intersection = len(a & b)
                union = len(a | b)
                pairwise_jaccards.append(intersection / union if union > 0 else 0)

        if pairwise_jaccards:
            disagreement[i] = 1 - np.mean(pairwise_jaccards)
        else:
            disagreement[i] = 0

    return disagreement


def compute_calibration_metrics(probabilities: np.ndarray,
                                correct: np.ndarray,
                                n_bins: int = 10) -> Dict[str, float]:
    """Compute calibration metrics (ECE, MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    calibration_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = probabilities[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()

            calibration_error = np.abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)

            calibration_data.append({
                'bin_mid': (bin_lower + bin_upper) / 2,
                'confidence': avg_confidence,
                'accuracy': avg_accuracy,
                'count': in_bin.sum()
            })

    return {
        'ece': ece,
        'mce': mce,
        'calibration_data': calibration_data
    }


def evaluate_model_vs_annotators(df: pd.DataFrame,
                                 model_col: str,
                                 annotator_cols: List[str],
                                 consensus_col: str) -> Dict:
    """Comprehensive evaluation of model against annotators."""

    # Parse all annotations
    model_preds = parse_annotations(df, model_col)
    consensus = parse_annotations(df, consensus_col)
    annotator_anns = [parse_annotations(df, col) for col in annotator_cols]

    # Get all unique labels
    all_labels = set()
    for ann in model_preds + consensus:
        all_labels.update(ann)
    for annotator in annotator_anns:
        for ann in annotator:
            all_labels.update(ann)
    all_labels.discard('')

    results = {
        'all_labels': sorted(all_labels),
        'n_samples': len(df),
        'n_annotators': len(annotator_cols)
    }

    # 1. Model vs Consensus
    print("Computing Model vs Consensus metrics...")
    results['model_vs_consensus'] = compute_classification_metrics(consensus, model_preds, all_labels)
    results['model_vs_consensus_per_class'] = compute_per_class_metrics(consensus, model_preds, all_labels)

    # 2. Model vs each annotator
    print("Computing Model vs Individual Annotators...")
    results['model_vs_annotators'] = {}
    for col, ann in zip(annotator_cols, annotator_anns):
        annotator_name = col.replace('_themes', '')
        metrics = compute_classification_metrics(ann, model_preds, all_labels)
        pairwise = compute_pairwise_agreement(ann, model_preds)
        results['model_vs_annotators'][annotator_name] = {
            'metrics': metrics,
            'pairwise': pairwise
        }

    # 3. Inter-annotator agreement
    print("Computing Inter-Annotator Agreement...")
    # Pairwise between all annotators
    annotator_pairs = {}
    for (col1, ann1), (col2, ann2) in combinations(zip(annotator_cols, annotator_anns), 2):
        name1 = col1.replace('_themes', '')
        name2 = col2.replace('_themes', '')
        pair_name = f"{name1}_vs_{name2}"
        annotator_pairs[pair_name] = compute_pairwise_agreement(ann1, ann2)

    results['inter_annotator_pairwise'] = annotator_pairs

    # Fleiss' kappa
    results['fleiss_kappa'] = compute_fleiss_kappa(annotator_anns, all_labels)

    # Krippendorff's alpha
    results['krippendorff_alpha'] = compute_krippendorff_alpha(annotator_anns, all_labels)

    # 4. Human ceiling analysis
    print("Computing Human Ceiling Analysis...")
    # Average model-annotator agreement vs average annotator-annotator agreement
    model_annotator_jaccards = [
        results['model_vs_annotators'][col.replace('_themes', '')]['pairwise']['jaccard_similarity']
        for col in annotator_cols
    ]
    annotator_annotator_jaccards = [v['jaccard_similarity'] for v in annotator_pairs.values()]

    results['human_ceiling'] = {
        'avg_model_annotator_jaccard': np.mean(model_annotator_jaccards),
        'std_model_annotator_jaccard': np.std(model_annotator_jaccards),
        'avg_annotator_annotator_jaccard': np.mean(annotator_annotator_jaccards),
        'std_annotator_annotator_jaccard': np.std(annotator_annotator_jaccards),
        'human_ceiling_ratio': np.mean(model_annotator_jaccards) / (np.mean(annotator_annotator_jaccards) + 1e-10)
    }

    # 5. Disagreement-stratified evaluation
    print("Computing Disagreement-Stratified Evaluation...")
    disagreement = compute_annotator_disagreement_index(annotator_anns)

    # Split into low/medium/high disagreement
    low_dis_mask = disagreement < 0.33
    med_dis_mask = (disagreement >= 0.33) & (disagreement < 0.66)
    high_dis_mask = disagreement >= 0.66

    results['disagreement_stratified'] = {
        'low_disagreement': {
            'n_samples': int(low_dis_mask.sum()),
            'metrics': compute_classification_metrics(
                [c for c, m in zip(consensus, low_dis_mask) if m],
                [p for p, m in zip(model_preds, low_dis_mask) if m],
                all_labels
            ) if low_dis_mask.sum() > 0 else None
        },
        'medium_disagreement': {
            'n_samples': int(med_dis_mask.sum()),
            'metrics': compute_classification_metrics(
                [c for c, m in zip(consensus, med_dis_mask) if m],
                [p for p, m in zip(model_preds, med_dis_mask) if m],
                all_labels
            ) if med_dis_mask.sum() > 0 else None
        },
        'high_disagreement': {
            'n_samples': int(high_dis_mask.sum()),
            'metrics': compute_classification_metrics(
                [c for c, m in zip(consensus, high_dis_mask) if m],
                [p for p, m in zip(model_preds, high_dis_mask) if m],
                all_labels
            ) if high_dis_mask.sum() > 0 else None
        }
    }

    # 6. Calibration (if probability available)
    print("Computing Calibration Metrics...")
    if 'labels_probability' in df.columns:
        probs = df['labels_probability'].apply(lambda x: safe_eval(x)[0] if safe_eval(x) else 0.5)
        probs = np.array(probs.tolist())

        # Check if top prediction matches consensus
        correct = np.array([
            bool(model_preds[i] & consensus[i]) if consensus[i] else False
            for i in range(len(model_preds))
        ])

        results['calibration'] = compute_calibration_metrics(probs, correct)

    return results


def create_evaluation_figures(results: Dict, output_dir: Path):
    """Generate visualization figures for evaluation results."""

    # =========================================================================
    # Figure 1: Main Performance Dashboard
    # =========================================================================
    fig1 = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)

    fig1.suptitle('Model Benchmark Evaluation Dashboard\n'
                  f'N={results["n_samples"]} samples, {results["n_annotators"]} annotators',
                  fontsize=14, fontweight='bold', y=0.98)

    # 1.1 Model vs Consensus Summary
    ax1 = fig1.add_subplot(gs[0, 0])
    metrics = results['model_vs_consensus']
    metric_names = ['Accuracy', 'F1 Micro', 'F1 Macro', 'Precision', 'Recall', 'Jaccard']
    metric_values = [metrics.accuracy, metrics.f1_micro, metrics.f1_macro,
                    metrics.precision_macro, metrics.recall_macro, metrics.jaccard_score]

    colors = plt.cm.RdYlGn(np.array(metric_values))
    bars = ax1.barh(metric_names, metric_values, color=colors, edgecolor='black')
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Score', fontweight='bold')
    ax1.set_title('Model vs Consensus', fontweight='bold')

    for bar, val in zip(bars, metric_values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontweight='bold')

    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # 1.2 Model vs Each Annotator
    ax2 = fig1.add_subplot(gs[0, 1])
    annotator_names = list(results['model_vs_annotators'].keys())
    annotator_f1s = [results['model_vs_annotators'][a]['metrics'].f1_macro for a in annotator_names]
    annotator_jaccards = [results['model_vs_annotators'][a]['pairwise']['jaccard_similarity'] for a in annotator_names]

    x = np.arange(len(annotator_names))
    width = 0.35
    ax2.bar(x - width/2, annotator_f1s, width, label='F1 Macro', color='steelblue')
    ax2.bar(x + width/2, annotator_jaccards, width, label='Jaccard', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(annotator_names, rotation=45, ha='right')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Model vs Each Annotator', fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 1)

    # 1.3 Human Ceiling Analysis
    ax3 = fig1.add_subplot(gs[0, 2])
    ceiling = results['human_ceiling']
    categories = ['Model-Annotator\nAgreement', 'Annotator-Annotator\nAgreement']
    values = [ceiling['avg_model_annotator_jaccard'], ceiling['avg_annotator_annotator_jaccard']]
    errors = [ceiling['std_model_annotator_jaccard'], ceiling['std_annotator_annotator_jaccard']]

    colors = ['steelblue', 'forestgreen']
    bars = ax3.bar(categories, values, yerr=errors, capsize=5, color=colors, edgecolor='black')
    ax3.set_ylabel('Jaccard Similarity', fontweight='bold')
    ax3.set_title('Human Ceiling Analysis', fontweight='bold')
    ax3.set_ylim(0, 1)

    # Add ratio annotation
    ratio = ceiling['human_ceiling_ratio']
    ax3.annotate(f'Ratio: {ratio:.2%}', xy=(0.5, max(values) + 0.1),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 1.4 Per-Class F1 Scores
    ax4 = fig1.add_subplot(gs[1, 0])
    per_class = results['model_vs_consensus_per_class']
    sorted_classes = sorted(per_class.keys(), key=lambda x: per_class[x]['f1'], reverse=True)
    class_f1s = [per_class[c]['f1'] for c in sorted_classes]
    class_supports = [per_class[c]['support'] for c in sorted_classes]

    # Truncate long names
    short_names = [c[:12] + '...' if len(c) > 12 else c for c in sorted_classes]

    colors = plt.cm.RdYlGn(np.array(class_f1s))
    bars = ax4.barh(short_names, class_f1s, color=colors, edgecolor='black')
    ax4.set_xlabel('F1 Score', fontweight='bold')
    ax4.set_title('Per-Class F1 (Model vs Consensus)', fontweight='bold')
    ax4.set_xlim(0, 1.1)

    # Add support labels
    for i, (bar, support) in enumerate(zip(bars, class_supports)):
        ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'n={support}', va='center', fontsize=7)

    # 1.5 Disagreement-Stratified Performance
    ax5 = fig1.add_subplot(gs[1, 1])
    strat = results['disagreement_stratified']
    strat_names = ['Low\nDisagreement', 'Medium\nDisagreement', 'High\nDisagreement']
    strat_f1s = []
    strat_counts = []

    for level in ['low_disagreement', 'medium_disagreement', 'high_disagreement']:
        if strat[level]['metrics']:
            strat_f1s.append(strat[level]['metrics'].f1_macro)
        else:
            strat_f1s.append(0)
        strat_counts.append(strat[level]['n_samples'])

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax5.bar(strat_names, strat_f1s, color=colors, edgecolor='black')
    ax5.set_ylabel('F1 Macro', fontweight='bold')
    ax5.set_title('Performance by Annotator Disagreement', fontweight='bold')
    ax5.set_ylim(0, 1)

    for bar, count in zip(bars, strat_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={count}', ha='center', fontweight='bold')

    # 1.6 Inter-Annotator Agreement Matrix
    ax6 = fig1.add_subplot(gs[1, 2])
    annotator_names = list(results['model_vs_annotators'].keys())
    n_ann = len(annotator_names)

    # Build agreement matrix
    agreement_matrix = np.eye(n_ann)
    for pair_name, pair_data in results['inter_annotator_pairwise'].items():
        parts = pair_name.split('_vs_')
        if len(parts) == 2:
            i = annotator_names.index(parts[0]) if parts[0] in annotator_names else -1
            j = annotator_names.index(parts[1]) if parts[1] in annotator_names else -1
            if i >= 0 and j >= 0:
                agreement_matrix[i, j] = pair_data['jaccard_similarity']
                agreement_matrix[j, i] = pair_data['jaccard_similarity']

    sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
               xticklabels=annotator_names, yticklabels=annotator_names,
               ax=ax6, vmin=0, vmax=1, cbar_kws={'label': 'Jaccard'})
    ax6.set_title('Inter-Annotator Agreement', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(output_dir / 'benchmark_evaluation_dashboard.png',
                 dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'benchmark_evaluation_dashboard.png'}")

    # =========================================================================
    # Figure 2: Calibration Analysis (if available)
    # =========================================================================
    if 'calibration' in results:
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig2.suptitle('Model Calibration Analysis', fontsize=14, fontweight='bold')

        cal_data = results['calibration']['calibration_data']
        confidences = [d['confidence'] for d in cal_data]
        accuracies = [d['accuracy'] for d in cal_data]
        counts = [d['count'] for d in cal_data]

        # Reliability diagram
        ax = axes[0]
        ax.bar(confidences, accuracies, width=0.08, alpha=0.8, color='steelblue',
               edgecolor='black', label='Observed Accuracy')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontweight='bold')
        ax.set_title('Reliability Diagram', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add ECE annotation
        ece = results['calibration']['ece']
        ax.annotate(f'ECE = {ece:.4f}', xy=(0.05, 0.95), fontsize=12,
                   fontweight='bold', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat'))

        # Confidence histogram
        ax = axes[1]
        ax.bar(confidences, counts, width=0.08, alpha=0.8, color='coral', edgecolor='black')
        ax.set_xlabel('Predicted Probability', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Prediction Confidence Distribution', fontweight='bold')

        plt.tight_layout()
        fig2.savefig(output_dir / 'benchmark_calibration.png',
                     dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_dir / 'benchmark_calibration.png'}")

    # =========================================================================
    # Figure 3: Summary Statistics Table
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Build summary table
    m = results['model_vs_consensus']
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['', '', ''],
        ['MODEL VS CONSENSUS', '', ''],
        ['Accuracy', f'{m.accuracy:.4f}', 'Exact match rate'],
        ['F1 Micro', f'{m.f1_micro:.4f}', 'Instance-weighted F1'],
        ['F1 Macro', f'{m.f1_macro:.4f}', 'Class-weighted F1'],
        ['Precision Macro', f'{m.precision_macro:.4f}', 'Avoid false positives'],
        ['Recall Macro', f'{m.recall_macro:.4f}', 'Avoid false negatives'],
        ['Cohen\'s Kappa', f'{m.cohen_kappa:.4f}', f'{"Substantial" if m.cohen_kappa > 0.6 else "Moderate" if m.cohen_kappa > 0.4 else "Fair"}'],
        ['MCC', f'{m.mcc:.4f}', 'Balanced measure'],
        ['Jaccard Score', f'{m.jaccard_score:.4f}', 'Set overlap'],
        ['Hamming Loss', f'{m.hamming_loss:.4f}', 'Lower is better'],
        ['', '', ''],
        ['INTER-ANNOTATOR AGREEMENT', '', ''],
        ['Fleiss\' Kappa', f'{results["fleiss_kappa"]:.4f}', f'{"Substantial" if results["fleiss_kappa"] > 0.6 else "Moderate"}'],
        ['Krippendorff\'s Alpha', f'{results["krippendorff_alpha"]:.4f}', f'{"Acceptable" if results["krippendorff_alpha"] > 0.667 else "Low"}'],
        ['', '', ''],
        ['HUMAN CEILING', '', ''],
        ['Model-Annotator Jaccard', f'{results["human_ceiling"]["avg_model_annotator_jaccard"]:.4f} ± {results["human_ceiling"]["std_model_annotator_jaccard"]:.4f}', ''],
        ['Annotator-Annotator Jaccard', f'{results["human_ceiling"]["avg_annotator_annotator_jaccard"]:.4f} ± {results["human_ceiling"]["std_annotator_annotator_jaccard"]:.4f}', ''],
        ['Human Ceiling Ratio', f'{results["human_ceiling"]["human_ceiling_ratio"]:.2%}', 'Model vs Human agreement ratio'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.35, 0.25, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(3):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#4a90d9')
        table[(0, i)].set_text_props(color='white')

    # Style section headers
    for row_idx in [2, 13, 17]:
        for col_idx in range(3):
            table[(row_idx, col_idx)].set_facecolor('#e6f2ff')
            table[(row_idx, col_idx)].set_text_props(fontweight='bold')

    fig3.suptitle('Benchmark Evaluation Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig3.savefig(output_dir / 'benchmark_summary_table.png',
                 dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'benchmark_summary_table.png'}")


def print_detailed_report(results: Dict):
    """Print detailed evaluation report to console."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL BENCHMARK EVALUATION REPORT")
    print("="*80)

    print(f"\nDataset: {results['n_samples']} samples, {results['n_annotators']} annotators")
    print(f"Labels: {len(results['all_labels'])} unique themes")

    # Model vs Consensus
    print("\n" + "-"*80)
    print("1. MODEL vs CONSENSUS (Gold Standard)")
    print("-"*80)
    m = results['model_vs_consensus']
    print(f"   Accuracy:        {m.accuracy:.4f}")
    print(f"   F1 Micro:        {m.f1_micro:.4f}")
    print(f"   F1 Macro:        {m.f1_macro:.4f}")
    print(f"   Precision Macro: {m.precision_macro:.4f}")
    print(f"   Recall Macro:    {m.recall_macro:.4f}")
    print(f"   Cohen's Kappa:   {m.cohen_kappa:.4f}")
    print(f"   MCC:             {m.mcc:.4f}")
    print(f"   Jaccard Score:   {m.jaccard_score:.4f}")
    print(f"   Hamming Loss:    {m.hamming_loss:.4f}")

    # Per-class breakdown
    print("\n   Per-Class Performance:")
    print(f"   {'Class':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Support':>8}")
    print("   " + "-"*60)
    per_class = results['model_vs_consensus_per_class']
    for cls in sorted(per_class.keys(), key=lambda x: per_class[x]['f1'], reverse=True):
        d = per_class[cls]
        print(f"   {cls:<25} {d['f1']:>8.3f} {d['precision']:>8.3f} {d['recall']:>8.3f} {d['support']:>8}")

    # Model vs Annotators
    print("\n" + "-"*80)
    print("2. MODEL vs INDIVIDUAL ANNOTATORS")
    print("-"*80)
    for name, data in results['model_vs_annotators'].items():
        m = data['metrics']
        p = data['pairwise']
        print(f"\n   {name}:")
        print(f"      F1 Macro: {m.f1_macro:.4f}, Jaccard: {p['jaccard_similarity']:.4f}, "
              f"Exact: {p['exact_agreement']:.4f}")

    # Inter-annotator agreement
    print("\n" + "-"*80)
    print("3. INTER-ANNOTATOR AGREEMENT")
    print("-"*80)
    print(f"   Fleiss' Kappa:       {results['fleiss_kappa']:.4f}")
    print(f"   Krippendorff's Alpha: {results['krippendorff_alpha']:.4f}")
    print("\n   Pairwise Agreement (Jaccard):")
    for pair, data in results['inter_annotator_pairwise'].items():
        print(f"      {pair}: {data['jaccard_similarity']:.4f}")

    # Human ceiling
    print("\n" + "-"*80)
    print("4. HUMAN CEILING ANALYSIS")
    print("-"*80)
    h = results['human_ceiling']
    print(f"   Avg Model-Annotator Jaccard:     {h['avg_model_annotator_jaccard']:.4f} ± {h['std_model_annotator_jaccard']:.4f}")
    print(f"   Avg Annotator-Annotator Jaccard: {h['avg_annotator_annotator_jaccard']:.4f} ± {h['std_annotator_annotator_jaccard']:.4f}")
    print(f"   Human Ceiling Ratio:             {h['human_ceiling_ratio']:.2%}")

    if h['human_ceiling_ratio'] >= 0.9:
        print("   -> Model performs at or above human level!")
    elif h['human_ceiling_ratio'] >= 0.75:
        print("   -> Model performs close to human level")
    else:
        print("   -> Model has room for improvement vs human agreement")

    # Disagreement-stratified
    print("\n" + "-"*80)
    print("5. PERFORMANCE BY ANNOTATOR DISAGREEMENT")
    print("-"*80)
    for level in ['low_disagreement', 'medium_disagreement', 'high_disagreement']:
        data = results['disagreement_stratified'][level]
        if data['metrics']:
            print(f"   {level.replace('_', ' ').title()}: F1={data['metrics'].f1_macro:.4f} (n={data['n_samples']})")
        else:
            print(f"   {level.replace('_', ' ').title()}: N/A (n={data['n_samples']})")

    # Calibration
    if 'calibration' in results:
        print("\n" + "-"*80)
        print("6. CALIBRATION METRICS")
        print("-"*80)
        print(f"   Expected Calibration Error (ECE): {results['calibration']['ece']:.4f}")
        print(f"   Maximum Calibration Error (MCE):  {results['calibration']['mce']:.4f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive model benchmark evaluation with ambiguity analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_model_evaluation.py annotations.csv
  python benchmark_model_evaluation.py annotations.csv -o ./results --model-col labels_label
        """
    )

    parser.add_argument('csv_path', type=str,
                        help='Path to CSV with model and annotator annotations')
    parser.add_argument('-o', '--output', type=str, default='paper/figures/benchmark',
                        help='Output directory for figures')
    parser.add_argument('--model-col', type=str, default='labels_label',
                        help='Column name for model predictions')
    parser.add_argument('--consensus-col', type=str, default='consensus_themes',
                        help='Column name for consensus annotations')
    parser.add_argument('--annotator-suffix', type=str, default='_themes',
                        help='Suffix to identify annotator columns')

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"Error: File not found: {args.csv_path}")
        return 1

    # Load data
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # Find annotator columns (exclude consensus and specific_themes unless it's the target suffix)
    annotator_cols = [col for col in df.columns
                      if col.endswith(args.annotator_suffix)
                      and col != args.consensus_col
                      and not col.startswith('labels_')
                      and not col.startswith('consensus_')
                      and 'specific' not in col.lower()]  # Exclude specific_themes by default

    if not annotator_cols:
        # Fallback: try to find any annotator columns
        annotator_cols = [col for col in df.columns
                          if col.endswith(args.annotator_suffix)
                          and col != args.consensus_col
                          and not col.startswith('labels_')]

    print(f"Found {len(annotator_cols)} annotators: {annotator_cols}")

    # Run evaluation
    results = evaluate_model_vs_annotators(
        df, args.model_col, annotator_cols, args.consensus_col
    )

    # Print report
    print_detailed_report(results)

    # Generate figures
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_evaluation_figures(results, output_dir)

    print(f"\nFigures saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())

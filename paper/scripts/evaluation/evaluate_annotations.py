#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
evaluate_annotations.py

MAIN OBJECTIVE:
---------------
Scientific evaluation script comparing annotation pipelines against the benchmark
(manual annotations in final_test_small.csv). Computes rigorous multi-label
classification metrics for themes, political parties, and specific themes.

Dependencies:
-------------
- pandas
- numpy
- pathlib
- json
- dataclasses
- logging

MAIN FEATURES:
--------------
1) Multi-label metrics: Precision, Recall, F1 (micro, macro, weighted)
2) Subset Accuracy (exact match), Jaccard Score
3) Per-category breakdown for themes, political_parties, specific_themes
4) Statistical analysis by LLM and dataset size
5) LaTeX table generation for paper

OUTPUT FILES:
-------------
- pipeline_comparison.csv: Summary metrics for all pipelines
- per_theme_metrics.csv: Per-theme breakdown for each pipeline
- per_party_metrics.csv: Per-party breakdown for each pipeline
- per_specific_metrics.csv: Per-specific-theme breakdown
- statistical_analysis.json: Statistical tests and analysis
- results_table.tex: LaTeX table for paper

Author:
-------
Antoine Lemor
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

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

ALL_PARTIES = [
    'political_parties_BQ', 'political_parties_CAQ', 'political_parties_CPC',
    'political_parties_GPC', 'political_parties_LPC', 'political_parties_NDP',
    'political_parties_PCQ', 'political_parties_PLQ', 'political_parties_PQ',
    'political_parties_QS'
]

ALL_SPECIFIC_THEMES = [
    'specific_themes_early_learning_childcare',
    'specific_themes_public_finance',
    'specific_themes_welfare_state'
]

ALL_SENTIMENTS = ['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MultiLabelMetrics:
    """Aggregated multi-label classification metrics."""
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    subset_accuracy: float = 0.0
    hamming_loss: float = 0.0
    jaccard_score: float = 0.0
    n_samples: int = 0


@dataclass
class CategoryResult:
    """Results for a single category."""
    category: str
    precision: float
    recall: float
    f1: float
    support: int
    tp: int
    fp: int
    fn: int


@dataclass
class PipelineResults:
    """Complete evaluation results for a pipeline."""
    pipeline_name: str
    n_samples: int = 0
    overall: MultiLabelMetrics = field(default_factory=MultiLabelMetrics)
    themes: MultiLabelMetrics = field(default_factory=MultiLabelMetrics)
    political_parties: MultiLabelMetrics = field(default_factory=MultiLabelMetrics)
    specific_themes: MultiLabelMetrics = field(default_factory=MultiLabelMetrics)
    sentiment_accuracy: float = 0.0
    sentiment_f1_macro: float = 0.0
    sentiment_f1_weighted: float = 0.0
    per_theme: List[CategoryResult] = field(default_factory=list)
    per_party: List[CategoryResult] = field(default_factory=list)
    per_specific: List[CategoryResult] = field(default_factory=list)


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_multilabel_metrics(
    y_true: List[Set[str]],
    y_pred: List[Set[str]],
    all_labels: List[str]
) -> Tuple[MultiLabelMetrics, List[CategoryResult]]:
    """Compute multi-label classification metrics."""
    n_samples = len(y_true)
    n_labels = len(all_labels)

    if n_samples == 0 or n_labels == 0:
        return MultiLabelMetrics(n_samples=n_samples), []

    # Build binary matrices
    y_true_binary = np.zeros((n_samples, n_labels), dtype=int)
    y_pred_binary = np.zeros((n_samples, n_labels), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    for i, (true_set, pred_set) in enumerate(zip(y_true, y_pred)):
        for label in true_set:
            if label in label_to_idx:
                y_true_binary[i, label_to_idx[label]] = 1
        for label in pred_set:
            if label in label_to_idx:
                y_pred_binary[i, label_to_idx[label]] = 1

    # MICRO METRICS
    tp_micro = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp_micro = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn_micro = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    # PER-LABEL METRICS
    per_label_metrics = []
    precisions, recalls, f1s, supports = [], [], [], []

    for j, label in enumerate(all_labels):
        tp = np.sum((y_true_binary[:, j] == 1) & (y_pred_binary[:, j] == 1))
        fp = np.sum((y_true_binary[:, j] == 0) & (y_pred_binary[:, j] == 1))
        fn = np.sum((y_true_binary[:, j] == 1) & (y_pred_binary[:, j] == 0))
        support = int(np.sum(y_true_binary[:, j]))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(support)

        per_label_metrics.append(CategoryResult(
            category=label, precision=p, recall=r, f1=f,
            support=support, tp=int(tp), fp=int(fp), fn=int(fn)
        ))

    # MACRO METRICS
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = np.mean(f1s)

    # WEIGHTED METRICS
    total_support = sum(supports)
    if total_support > 0:
        weights = np.array(supports) / total_support
        precision_weighted = np.sum(np.array(precisions) * weights)
        recall_weighted = np.sum(np.array(recalls) * weights)
        f1_weighted = np.sum(np.array(f1s) * weights)
    else:
        precision_weighted = recall_weighted = f1_weighted = 0.0

    # SUBSET ACCURACY
    exact_matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    subset_accuracy = exact_matches / n_samples

    # HAMMING LOSS
    hamming_loss = np.mean(y_true_binary != y_pred_binary)

    # JACCARD SCORE
    jaccard_scores = []
    for t, p in zip(y_true, y_pred):
        if len(t) == 0 and len(p) == 0:
            jaccard_scores.append(1.0)
        elif len(t) == 0 or len(p) == 0:
            jaccard_scores.append(0.0)
        else:
            jaccard_scores.append(len(t & p) / len(t | p))
    jaccard_score = np.mean(jaccard_scores)

    metrics = MultiLabelMetrics(
        precision_micro=precision_micro, recall_micro=recall_micro, f1_micro=f1_micro,
        precision_macro=precision_macro, recall_macro=recall_macro, f1_macro=f1_macro,
        precision_weighted=precision_weighted, recall_weighted=recall_weighted, f1_weighted=f1_weighted,
        subset_accuracy=subset_accuracy, hamming_loss=hamming_loss, jaccard_score=jaccard_score,
        n_samples=n_samples
    )

    return metrics, per_label_metrics


def compute_multiclass_metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, float]:
    """Compute multiclass classification metrics for sentiment."""
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred)
                   if t and p and t not in ['None', 'null', ''] and p not in ['None', 'null', '']]

    if not valid_pairs:
        return 0.0, 0.0, 0.0

    y_true_f, y_pred_f = zip(*valid_pairs)
    n_samples = len(y_true_f)
    accuracy = sum(1 for t, p in zip(y_true_f, y_pred_f) if t == p) / n_samples

    all_classes = sorted(set(y_true_f) | set(y_pred_f))
    f1s, supports = [], []

    for cls in all_classes:
        tp = sum(1 for t, p in zip(y_true_f, y_pred_f) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true_f, y_pred_f) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true_f, y_pred_f) if t == cls and p != cls)
        support = sum(1 for t in y_true_f if t == cls)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        supports.append(support)

    f1_macro = np.mean(f1s) if f1s else 0.0
    total_support = sum(supports)
    f1_weighted = np.sum(np.array(f1s) * np.array(supports) / total_support) if total_support > 0 else 0.0

    return accuracy, f1_macro, f1_weighted


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def parse_annotations(ann_str: str) -> Dict:
    """Parse JSON annotation string."""
    if pd.isna(ann_str) or not ann_str:
        return {'benchmark': {}, 'pipelines': {}}
    try:
        return json.loads(ann_str)
    except:
        return {'benchmark': {}, 'pipelines': {}}


def extract_benchmark_data(benchmark: Dict) -> Tuple[Set[str], Set[str], Set[str], str]:
    """Extract benchmark labels."""
    themes = set(benchmark.get('manual_themes', []) or [])
    parties = set(benchmark.get('manual_political_parties', []) or [])
    specific = set(benchmark.get('manual_specific_themes', []) or [])
    sentiment = benchmark.get('manual_sentiment', '') or ''
    return themes, parties, specific, sentiment


def extract_pipeline_data(pipeline: Dict) -> Tuple[Set[str], Set[str], Set[str], str]:
    """Extract pipeline predictions."""
    themes = set(pipeline.get('themes', []) or [])
    parties = set(pipeline.get('political_parties', []) or [])
    specific = set(pipeline.get('specific_themes', []) or [])
    sentiment = pipeline.get('sentiment', '') or ''
    return themes, parties, specific, sentiment


# =============================================================================
# PIPELINE EVALUATION
# =============================================================================

def evaluate_pipeline(df: pd.DataFrame, pipeline_name: str) -> Optional[PipelineResults]:
    """Evaluate a single pipeline against the benchmark."""
    themes_true, themes_pred = [], []
    parties_true, parties_pred = [], []
    specific_true, specific_pred = [], []
    sentiment_true, sentiment_pred = [], []
    n_valid = 0

    for _, row in df.iterrows():
        annotations = parse_annotations(row.get('all_annotations', '{}'))
        benchmark = annotations.get('benchmark', {})
        pipelines = annotations.get('pipelines', {})

        if pipeline_name not in pipelines:
            continue

        n_valid += 1
        pipeline = pipelines[pipeline_name]

        b_themes, b_parties, b_specific, b_sentiment = extract_benchmark_data(benchmark)
        p_themes, p_parties, p_specific, p_sentiment = extract_pipeline_data(pipeline)

        themes_true.append(b_themes)
        themes_pred.append(p_themes)
        parties_true.append(b_parties)
        parties_pred.append(p_parties)
        specific_true.append(b_specific)
        specific_pred.append(p_specific)
        sentiment_true.append(b_sentiment)
        sentiment_pred.append(p_sentiment)

    if n_valid == 0:
        return None

    results = PipelineResults(pipeline_name=pipeline_name, n_samples=n_valid)

    results.themes, results.per_theme = compute_multilabel_metrics(themes_true, themes_pred, ALL_THEMES)
    results.political_parties, results.per_party = compute_multilabel_metrics(parties_true, parties_pred, ALL_PARTIES)
    results.specific_themes, results.per_specific = compute_multilabel_metrics(specific_true, specific_pred, ALL_SPECIFIC_THEMES)
    results.sentiment_accuracy, results.sentiment_f1_macro, results.sentiment_f1_weighted = compute_multiclass_metrics(sentiment_true, sentiment_pred)

    overall_true = [t | p | s for t, p, s in zip(themes_true, parties_true, specific_true)]
    overall_pred = [t | p | s for t, p, s in zip(themes_pred, parties_pred, specific_pred)]
    results.overall, _ = compute_multilabel_metrics(overall_true, overall_pred, ALL_THEMES + ALL_PARTIES + ALL_SPECIFIC_THEMES)

    return results


def get_all_pipelines(df: pd.DataFrame) -> List[str]:
    """Get all pipeline names from the dataset."""
    pipelines = set()
    for _, row in df.iterrows():
        annotations = parse_annotations(row.get('all_annotations', '{}'))
        pipelines.update(annotations.get('pipelines', {}).keys())
    return sorted(list(pipelines))


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

def perform_statistical_analysis(results: List[PipelineResults]) -> Dict:
    """Perform statistical analysis on pipeline results."""
    analysis = {}

    ranking = sorted([(r.pipeline_name, r.overall.f1_micro) for r in results], key=lambda x: x[1], reverse=True)
    analysis['ranking_f1_micro'] = ranking
    analysis['best_pipeline'] = ranking[0][0] if ranking else None
    analysis['best_f1_micro'] = ranking[0][1] if ranking else 0.0

    themes_ranking = sorted([(r.pipeline_name, r.themes.f1_micro) for r in results], key=lambda x: x[1], reverse=True)
    analysis['ranking_themes_f1'] = themes_ranking
    analysis['best_themes_pipeline'] = themes_ranking[0][0] if themes_ranking else None

    parties_ranking = sorted([(r.pipeline_name, r.political_parties.f1_micro) for r in results], key=lambda x: x[1], reverse=True)
    analysis['ranking_parties_f1'] = parties_ranking
    analysis['best_parties_pipeline'] = parties_ranking[0][0] if parties_ranking else None

    sentiment_ranking = sorted([(r.pipeline_name, r.sentiment_accuracy) for r in results], key=lambda x: x[1], reverse=True)
    analysis['ranking_sentiment_accuracy'] = sentiment_ranking
    analysis['best_sentiment_pipeline'] = sentiment_ranking[0][0] if sentiment_ranking else None

    # Group by LLM
    llm_performance = defaultdict(list)
    for r in results:
        llm = r.pipeline_name.rsplit('_', 1)[0] if '_' in r.pipeline_name else r.pipeline_name
        llm_performance[llm].append(r.overall.f1_micro)
    analysis['llm_avg_f1'] = {llm: np.mean(scores) for llm, scores in llm_performance.items()}

    # Group by dataset size
    dataset_performance = defaultdict(list)
    for r in results:
        dataset = r.pipeline_name.rsplit('_', 1)[1] if '_' in r.pipeline_name else 'unknown'
        dataset_performance[dataset].append(r.overall.f1_micro)
    analysis['dataset_avg_f1'] = {dataset: np.mean(scores) for dataset, scores in dataset_performance.items()}

    return analysis


def generate_summary_table(results: List[PipelineResults]) -> pd.DataFrame:
    """Generate summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            'Pipeline': r.pipeline_name,
            'N_Samples': r.n_samples,
            'Overall_F1_Micro': r.overall.f1_micro,
            'Overall_F1_Macro': r.overall.f1_macro,
            'Overall_Precision': r.overall.precision_micro,
            'Overall_Recall': r.overall.recall_micro,
            'Overall_Jaccard': r.overall.jaccard_score,
            'Overall_Subset_Acc': r.overall.subset_accuracy,
            'Themes_F1_Micro': r.themes.f1_micro,
            'Themes_F1_Macro': r.themes.f1_macro,
            'Themes_Precision': r.themes.precision_micro,
            'Themes_Recall': r.themes.recall_micro,
            'Parties_F1_Micro': r.political_parties.f1_micro,
            'Parties_F1_Macro': r.political_parties.f1_macro,
            'Parties_Precision': r.political_parties.precision_micro,
            'Parties_Recall': r.political_parties.recall_micro,
            'Specific_F1_Micro': r.specific_themes.f1_micro,
            'Specific_F1_Macro': r.specific_themes.f1_macro,
            'Sentiment_Accuracy': r.sentiment_accuracy,
            'Sentiment_F1_Macro': r.sentiment_f1_macro,
        })
    df = pd.DataFrame(rows)
    return df.sort_values('Overall_F1_Micro', ascending=False)


def generate_per_category_table(results: List[PipelineResults], category_type: str) -> pd.DataFrame:
    """Generate per-category breakdown table."""
    rows = []
    for r in results:
        if category_type == 'themes':
            categories = r.per_theme
        elif category_type == 'parties':
            categories = r.per_party
        elif category_type == 'specific':
            categories = r.per_specific
        else:
            continue

        for cat in categories:
            rows.append({
                'Pipeline': r.pipeline_name,
                'Category': cat.category,
                'Precision': cat.precision,
                'Recall': cat.recall,
                'F1': cat.f1,
                'Support': cat.support,
                'TP': cat.tp,
                'FP': cat.fp,
                'FN': cat.fn,
            })
    return pd.DataFrame(rows)


def generate_latex_table(summary_df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table."""
    top_df = summary_df.head(15)
    latex = r"""\begin{table}[htbp]
\centering
\caption{Pipeline Performance Comparison on Test Set}
\label{tab:pipeline_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Pipeline} & \textbf{F1$_\mu$} & \textbf{F1$_M$} & \textbf{P$_\mu$} & \textbf{R$_\mu$} & \textbf{Themes F1} & \textbf{Parties F1} & \textbf{Sent. Acc.} & \textbf{Jaccard} \\
\midrule
"""
    for _, row in top_df.iterrows():
        name = row['Pipeline'].replace('_', r'\_')
        latex += f"{name} & {row['Overall_F1_Micro']:.3f} & {row['Overall_F1_Macro']:.3f} & "
        latex += f"{row['Overall_Precision']:.3f} & {row['Overall_Recall']:.3f} & "
        latex += f"{row['Themes_F1_Micro']:.3f} & {row['Parties_F1_Micro']:.3f} & "
        latex += f"{row['Sentiment_Accuracy']:.3f} & {row['Overall_Jaccard']:.3f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    with open(output_path, 'w') as f:
        f.write(latex)


def print_report(summary_df: pd.DataFrame, analysis: Dict):
    """Print detailed report to console."""
    print("\n" + "="*80)
    print("ANNOTATION PIPELINE EVALUATION REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "-"*80)
    print("TOP 10 PIPELINES (by Overall F1 Micro)")
    print("-"*80)

    for i, (_, row) in enumerate(summary_df.head(10).iterrows()):
        print(f"\n{i+1}. {row['Pipeline']}")
        print(f"   Overall F1 μ: {row['Overall_F1_Micro']:.4f}  |  F1 M: {row['Overall_F1_Macro']:.4f}")
        print(f"   Themes F1:    {row['Themes_F1_Micro']:.4f}  |  Parties F1: {row['Parties_F1_Micro']:.4f}")
        print(f"   Sentiment:    {row['Sentiment_Accuracy']:.4f}  |  Jaccard:    {row['Overall_Jaccard']:.4f}")

    print("\n" + "-"*80)
    print("BEST PIPELINES BY CATEGORY")
    print("-"*80)
    print(f"\nBest Overall:   {analysis.get('best_pipeline', 'N/A')} ({analysis.get('best_f1_micro', 0):.4f})")
    print(f"Best Themes:    {analysis.get('best_themes_pipeline', 'N/A')}")
    print(f"Best Parties:   {analysis.get('best_parties_pipeline', 'N/A')}")
    print(f"Best Sentiment: {analysis.get('best_sentiment_pipeline', 'N/A')}")

    print("\n" + "-"*80)
    print("AVERAGE PERFORMANCE BY LLM")
    print("-"*80)
    for llm, score in sorted(analysis.get('llm_avg_f1', {}).items(), key=lambda x: x[1], reverse=True):
        print(f"   {llm}: {score:.4f}")

    print("\n" + "-"*80)
    print("AVERAGE PERFORMANCE BY DATASET SIZE")
    print("-"*80)
    for dataset, score in sorted(analysis.get('dataset_avg_f1', {}).items(), key=lambda x: x[1], reverse=True):
        print(f"   {dataset}: {score:.4f}")

    print("\n" + "="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main evaluation function."""
    # Paths relative to paper/scripts/
    base_path = Path(__file__).parent.parent.parent  # LLM_Tool root
    data_path = base_path / 'data' / 'sets'
    output_path = base_path / 'paper' / 'results' / 'evaluation_results'
    output_path.mkdir(parents=True, exist_ok=True)

    annotations_file = data_path / 'final_test_small_annotated.csv'

    if not annotations_file.exists():
        logger.error(f"Annotations file not found: {annotations_file}")
        logger.info("Please run compile_annotations.py first")
        return

    logger.info(f"Loading annotations from {annotations_file}")
    df = pd.read_csv(annotations_file)
    logger.info(f"Loaded {len(df)} rows")

    pipelines = get_all_pipelines(df)
    logger.info(f"Found {len(pipelines)} pipelines to evaluate")

    logger.info("Evaluating pipelines...")
    results = []
    for pipeline_name in pipelines:
        logger.info(f"  Evaluating {pipeline_name}...")
        result = evaluate_pipeline(df, pipeline_name)
        if result:
            results.append(result)

    if not results:
        logger.error("No valid results!")
        return

    logger.info("Generating summary table...")
    summary_df = generate_summary_table(results)

    themes_df = generate_per_category_table(results, 'themes')
    parties_df = generate_per_category_table(results, 'parties')
    specific_df = generate_per_category_table(results, 'specific')

    logger.info("Performing statistical analysis...")
    analysis = perform_statistical_analysis(results)

    # Save results with fixed filenames (overwrite previous)
    summary_df.to_csv(output_path / 'pipeline_comparison.csv', index=False)
    themes_df.to_csv(output_path / 'per_theme_metrics.csv', index=False)
    parties_df.to_csv(output_path / 'per_party_metrics.csv', index=False)
    specific_df.to_csv(output_path / 'per_specific_metrics.csv', index=False)

    with open(output_path / 'statistical_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    generate_latex_table(summary_df, output_path / 'results_table.tex')

    logger.info(f"Results saved to {output_path}")

    print_report(summary_df, analysis)

    return summary_df, results, analysis


if __name__ == '__main__':
    summary_df, results, analysis = main()

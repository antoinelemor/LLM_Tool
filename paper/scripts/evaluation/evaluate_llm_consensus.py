#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
evaluate_llm_consensus.py

MAIN OBJECTIVE:
---------------
Evaluate LLM consensus approach for theme annotation. Tests whether combining
predictions from multiple LLMs through voting improves annotation quality
compared to individual models.

Dependencies:
-------------
- pandas
- json
- os
- collections.Counter

MAIN FEATURES:
--------------
1) Load LLM annotations from LLMs_training_set.csv
2) Create consensus labels with different thresholds (1-6 LLMs must agree)
3) Evaluate consensus against human benchmark
4) Compare with individual LLM performance
5) Export comparative results to CSV

Author:
-------
Antoine Lemor
"""

import pandas as pd
import json
import os
from collections import Counter

# Paths - go up 3 levels from scripts/evaluation/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAINING_SET = os.path.join(BASE_DIR, 'data/sets/LLMs_training_set.csv')
BENCHMARK = os.path.join(BASE_DIR, 'data/sets/final_test_large.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper/results/llm_consensus')

# All 22 themes
ALL_THEMES = [
    'macroeconomics', 'rights_liberties_minorities_discrimination',
    'health', 'agriculture', 'labor', 'education',
    'environment', 'energy', 'immigration', 'transportation',
    'law_and_crime', 'social_welfare', 'housing',
    'domestic_commerce', 'defense', 'technology',
    'foreign_trade', 'international_affairs', 'governments_governance',
    'public_lands', 'culture_nationalism', 'indigenous_affairs'
]

# Models to consider for consensus
MODELS = ['nemotron', 'llama', 'gpt_oss', 'gemma', 'gpt_4_1', 'gpt_5']


def parse_annotations(json_str):
    """Parse JSON annotations string."""
    if pd.isna(json_str):
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}


def get_model_themes(annotations, model, suffix='large'):
    """Extract themes predicted by a specific model."""
    key = f'{model}_themes_{suffix}'
    themes = annotations.get(key, None)
    if themes is None:
        return None  # Model didn't annotate
    return set(themes)


def get_ground_truth_themes(annotations):
    """Extract themes from ground truth annotations."""
    themes = annotations.get('manual_themes', [])
    if themes is None:
        return set()
    # Remove 'theme_' prefix for consistency
    return set(t.replace('theme_', '') for t in themes)


def create_consensus(annotations, threshold=3):
    """
    Create consensus labels from multiple LLM annotations.

    Args:
        annotations: Parsed JSON with LLM annotations
        threshold: Minimum number of LLMs that must agree

    Returns:
        Set of themes where >= threshold LLMs agree
    """
    theme_votes = Counter()
    models_voted = 0

    for model in MODELS:
        themes = get_model_themes(annotations, model)
        if themes is not None:  # Model provided annotations
            models_voted += 1
            for theme in themes:
                theme_votes[theme] += 1

    if models_voted == 0:
        return set(), 0

    # Themes with >= threshold votes
    consensus_themes = set(theme for theme, votes in theme_votes.items()
                          if votes >= threshold)

    return consensus_themes, models_voted


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_predictions(predictions_dict, ground_truth_dict):
    """
    Evaluate predictions against ground truth.

    Args:
        predictions_dict: {text: set of predicted themes}
        ground_truth_dict: {text: set of true themes}

    Returns:
        Dictionary with metrics
    """
    tp = fp = fn = 0

    for text, gt_themes in ground_truth_dict.items():
        pred_themes = predictions_dict.get(text, set())

        for theme in ALL_THEMES:
            in_gt = theme in gt_themes
            in_pred = theme in pred_themes

            if in_gt and in_pred:
                tp += 1
            elif in_pred and not in_gt:
                fp += 1
            elif in_gt and not in_pred:
                fn += 1

    precision, recall, f1 = calculate_metrics(tp, fp, fn)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    print("=" * 70)
    print("ÉVALUATION DU CONSENSUS LLM")
    print("=" * 70)

    # Load data
    print("\nChargement des données...")
    benchmark_df = pd.read_csv(BENCHMARK)
    training_df = pd.read_csv(TRAINING_SET)

    print(f"  Benchmark: {len(benchmark_df)} samples")
    print(f"  Training set: {len(training_df)} samples")

    # Deduplicate by text
    benchmark_dedup = benchmark_df.drop_duplicates(subset='text', keep='first')
    training_dedup = training_df.drop_duplicates(subset='text', keep='first')

    # Merge on text
    merged = benchmark_dedup.merge(
        training_dedup[['text', 'annotations_large']],
        on='text',
        how='inner'
    )
    print(f"  Samples en commun: {len(merged)}")

    # Parse annotations
    print("\nParsing des annotations...")
    merged['gt_parsed'] = merged['annotations'].apply(parse_annotations)
    merged['llm_parsed'] = merged['annotations_large'].apply(parse_annotations)

    # Build ground truth dict
    ground_truth = {}
    for _, row in merged.iterrows():
        gt_themes = get_ground_truth_themes(row['gt_parsed'])
        ground_truth[row['text']] = gt_themes

    # Evaluate individual LLMs
    print("\n" + "=" * 70)
    print("PERFORMANCE DES LLMs INDIVIDUELS")
    print("=" * 70)

    individual_results = []

    for model in MODELS:
        predictions = {}
        n_with_preds = 0

        for _, row in merged.iterrows():
            themes = get_model_themes(row['llm_parsed'], model)
            if themes is not None:
                n_with_preds += 1
                predictions[row['text']] = themes
            else:
                predictions[row['text']] = set()

        metrics = evaluate_predictions(predictions, ground_truth)
        metrics['model'] = model
        metrics['n_samples'] = n_with_preds
        individual_results.append(metrics)

        print(f"\n{model.upper()}:")
        print(f"  Samples avec prédictions: {n_with_preds}/{len(merged)}")
        print(f"  TP={metrics['tp']:4d}  FP={metrics['fp']:4d}  FN={metrics['fn']:4d}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Micro F1:  {metrics['f1']:.4f}")

    # Evaluate consensus with different thresholds
    print("\n" + "=" * 70)
    print("PERFORMANCE DU CONSENSUS LLM")
    print("=" * 70)

    consensus_results = []

    for threshold in [1, 2, 3, 4, 5, 6]:
        predictions = {}
        total_models_voted = 0

        for _, row in merged.iterrows():
            consensus_themes, n_models = create_consensus(row['llm_parsed'], threshold)
            predictions[row['text']] = consensus_themes
            total_models_voted += n_models

        avg_models = total_models_voted / len(merged)
        metrics = evaluate_predictions(predictions, ground_truth)
        metrics['threshold'] = threshold
        metrics['avg_models_voted'] = avg_models
        consensus_results.append(metrics)

        print(f"\nSeuil >= {threshold} LLMs:")
        print(f"  TP={metrics['tp']:4d}  FP={metrics['fp']:4d}  FN={metrics['fn']:4d}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Micro F1:  {metrics['f1']:.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF")
    print("=" * 70)

    print("\n{:<20} {:>10} {:>10} {:>10}".format("Méthode", "Precision", "Recall", "F1"))
    print("-" * 52)

    # Individual LLMs
    for r in individual_results:
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            r['model'], r['precision'], r['recall'], r['f1']))

    print("-" * 52)

    # Consensus
    for r in consensus_results:
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            f"Consensus >= {r['threshold']}", r['precision'], r['recall'], r['f1']))

    # Find best approaches
    print("\n" + "=" * 70)
    print("MEILLEURES APPROCHES")
    print("=" * 70)

    all_results = individual_results + consensus_results
    best_f1 = max(all_results, key=lambda x: x['f1'])
    best_precision = max(all_results, key=lambda x: x['precision'])
    best_recall = max(all_results, key=lambda x: x['recall'])

    print(f"\nMeilleur F1: ", end="")
    if 'model' in best_f1:
        print(f"{best_f1['model']} avec F1={best_f1['f1']:.4f}")
    else:
        print(f"Consensus >= {best_f1['threshold']} avec F1={best_f1['f1']:.4f}")

    print(f"Meilleure Precision: ", end="")
    if 'model' in best_precision:
        print(f"{best_precision['model']} avec P={best_precision['precision']:.4f}")
    else:
        print(f"Consensus >= {best_precision['threshold']} avec P={best_precision['precision']:.4f}")

    print(f"Meilleur Recall: ", end="")
    if 'model' in best_recall:
        print(f"{best_recall['model']} avec R={best_recall['recall']:.4f}")
    else:
        print(f"Consensus >= {best_recall['threshold']} avec R={best_recall['recall']:.4f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    individual_df = pd.DataFrame(individual_results)
    individual_df.to_csv(os.path.join(OUTPUT_DIR, 'individual_llm_results.csv'), index=False)

    consensus_df = pd.DataFrame(consensus_results)
    consensus_df.to_csv(os.path.join(OUTPUT_DIR, 'consensus_results.csv'), index=False)

    print(f"\n\nRésultats sauvegardés dans: {OUTPUT_DIR}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMANDATION")
    print("=" * 70)

    # Find best consensus
    best_consensus = max(consensus_results, key=lambda x: x['f1'])
    best_individual = max(individual_results, key=lambda x: x['f1'])

    print(f"\nMeilleur LLM individuel: {best_individual['model']} (F1={best_individual['f1']:.4f})")
    print(f"Meilleur consensus: seuil >= {best_consensus['threshold']} (F1={best_consensus['f1']:.4f})")

    improvement = (best_consensus['f1'] - best_individual['f1']) / best_individual['f1'] * 100

    if improvement > 0:
        print(f"\n✓ Le consensus améliore le F1 de {improvement:.1f}%")
        print(f"  → Utiliser le seuil >= {best_consensus['threshold']} pour créer les données d'entraînement")
    else:
        print(f"\n✗ Le consensus n'améliore pas le F1 ({improvement:.1f}%)")
        print(f"  → Utiliser les annotations de {best_individual['model']} directement")

    return individual_results, consensus_results


if __name__ == '__main__':
    individual, consensus = main()

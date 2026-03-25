#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
evaluate_models_on_benchmark.py

MAIN OBJECTIVE:
---------------
Evaluate LLM annotation quality on the large benchmark. Calculates micro F1
per model by comparing LLM predictions from LLMs_training_set.csv against
human ground truth from final_test_large.csv.

Dependencies:
-------------
- pandas
- json
- os

MAIN FEATURES:
--------------
1) Load and merge LLM annotations with benchmark
2) Calculate micro F1, precision, recall per model
3) Compare small vs large training set annotations
4) Export results to CSV

Author:
-------
Antoine Lemor
"""

import pandas as pd
import json
import os

# Paths - go up 3 levels from scripts/evaluation/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAINING_SET = os.path.join(BASE_DIR, 'data/sets/LLMs_training_set.csv')
BENCHMARK = os.path.join(BASE_DIR, 'data/sets/final_test_large.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper/results/model_evaluation')

# All 22 themes
ALL_THEMES = [
    'theme_macroeconomics', 'theme_rights_liberties_minorities_discrimination',
    'theme_health', 'theme_agriculture', 'theme_labor', 'theme_education',
    'theme_environment', 'theme_energy', 'theme_immigration', 'theme_transportation',
    'theme_law_and_crime', 'theme_social_welfare', 'theme_housing',
    'theme_domestic_commerce', 'theme_defense', 'theme_technology',
    'theme_foreign_trade', 'theme_international_affairs', 'theme_governments_governance',
    'theme_public_lands', 'theme_culture_nationalism', 'theme_indigenous_affairs'
]

# Models
MODELS = ['nemotron', 'llama', 'gpt_oss', 'gemma', 'gpt_4_1', 'gpt_5']
# Note: extra_large excluded - only 29/1427 samples have predictions (not meaningful)
TRAINING_SIZES = ['small', 'large']


def parse_annotations(json_str):
    """Parse JSON annotations string."""
    if pd.isna(json_str):
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}


def get_ground_truth_themes(annotations):
    """Extract themes from ground truth annotations."""
    themes = annotations.get('manual_themes', [])
    if themes is None:
        return set()
    return set(themes)


def get_model_predictions(annotations, model, training_size):
    """Extract themes predicted by a specific model."""
    key = f'{model}_themes_{training_size}'
    themes = annotations.get(key, [])
    if themes is None:
        return set()
    # Convert to standard format (add theme_ prefix if needed)
    standardized = set()
    for t in themes:
        if t.startswith('theme_'):
            standardized.add(t)
        else:
            standardized.add(f'theme_{t}')
    return standardized


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    print("=" * 70)
    print("EVALUATION DES MODÈLES SUR LE LARGE BENCHMARK")
    print("=" * 70)

    # Load data
    print("\nChargement des données...")
    benchmark_df = pd.read_csv(BENCHMARK)
    training_df = pd.read_csv(TRAINING_SET)

    print(f"  Benchmark: {len(benchmark_df)} samples ({benchmark_df['text'].nunique()} textes uniques)")
    print(f"  Training set: {len(training_df)} samples ({training_df['text'].nunique()} textes uniques)")

    # Deduplicate by text (more reliable than ID - same ID can have different texts)
    training_df_dedup = training_df.drop_duplicates(subset='text', keep='first')
    print(f"  Training set après déduplication par texte: {len(training_df_dedup)} samples")

    # Also deduplicate benchmark by text
    benchmark_df_dedup = benchmark_df.drop_duplicates(subset='text', keep='first')
    print(f"  Benchmark après déduplication par texte: {len(benchmark_df_dedup)} samples")

    # Merge on text (not ID - IDs can be unreliable)
    merged = benchmark_df_dedup.merge(
        training_df_dedup[['text', 'annotations_small', 'annotations_large']],
        on='text',
        how='inner'
    )
    print(f"  Samples en commun (par texte): {len(merged)}")

    # Parse annotations
    print("\nParsing des annotations...")
    merged['gt_parsed'] = merged['annotations'].apply(parse_annotations)
    merged['small_parsed'] = merged['annotations_small'].apply(parse_annotations)
    merged['large_parsed'] = merged['annotations_large'].apply(parse_annotations)

    # Calculate metrics per model and training size
    results = []

    print("\n" + "=" * 70)
    print("RÉSULTATS PAR MODÈLE ET TAILLE D'ENTRAINEMENT")
    print("=" * 70)

    for training_size in TRAINING_SIZES:
        parsed_col = {
            'small': 'small_parsed',
            'large': 'large_parsed'
        }[training_size]

        print(f"\n{'='*30} {training_size.upper()} {'='*30}")

        for model in MODELS:
            tp = fp = fn = 0
            n_samples_with_predictions = 0

            for _, row in merged.iterrows():
                gt_themes = get_ground_truth_themes(row['gt_parsed'])
                pred_themes = get_model_predictions(row[parsed_col], model, training_size)

                if pred_themes:  # Model has predictions for this sample
                    n_samples_with_predictions += 1

                # Calculate TP, FP, FN for this sample
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

            results.append({
                'model': model,
                'training_size': training_size,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'micro_f1': f1,
                'n_samples': len(merged),
                'n_with_predictions': n_samples_with_predictions
            })

            print(f"\n{model.upper()}:")
            print(f"  Samples avec prédictions: {n_samples_with_predictions}/{len(merged)}")
            print(f"  TP={tp:4d}  FP={fp:4d}  FN={fn:4d}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  Micro F1:  {f1:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'model_evaluation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n\nRésultats sauvegardés dans: {output_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("TABLEAU RÉCAPITULATIF (Micro F1)")
    print("=" * 70)

    pivot = results_df.pivot(index='model', columns='training_size', values='micro_f1')
    pivot = pivot[['small', 'large']]  # Order columns

    print("\n" + pivot.to_string())

    # Best models
    print("\n" + "=" * 70)
    print("MEILLEURS MODÈLES")
    print("=" * 70)

    for size in TRAINING_SIZES:
        subset = results_df[results_df['training_size'] == size]
        best = subset.loc[subset['micro_f1'].idxmax()]
        print(f"\n{size.upper()}: {best['model']} avec F1={best['micro_f1']:.4f}")

    # Calculate improvement from small to large
    print("\n" + "=" * 70)
    print("AMÉLIORATION SMALL → LARGE")
    print("=" * 70)

    for model in MODELS:
        small_f1 = results_df[(results_df['model'] == model) &
                              (results_df['training_size'] == 'small')]['micro_f1'].values[0]
        large_f1 = results_df[(results_df['model'] == model) &
                              (results_df['training_size'] == 'large')]['micro_f1'].values[0]

        if small_f1 > 0:
            improvement = (large_f1 - small_f1) / small_f1 * 100
            print(f"{model:12s}: {small_f1:.4f} → {large_f1:.4f} ({improvement:+.1f}%)")
        else:
            print(f"{model:12s}: {small_f1:.4f} → {large_f1:.4f} (N/A)")

    return results_df


if __name__ == '__main__':
    results = main()

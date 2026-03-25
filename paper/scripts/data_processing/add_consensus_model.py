#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
add_consensus_model.py

MAIN OBJECTIVE:
---------------
Add a consensus model to LLMs_training_set.csv by combining predictions from
multiple LLMs. The script evaluates different consensus thresholds against the
human benchmark, automatically selects the optimal configuration, and adds
consensus annotations to all annotation columns.

Dependencies:
-------------
- pandas
- json
- os
- collections.Counter

MAIN FEATURES:
--------------
1) Evaluates consensus thresholds from 1 to 6 LLMs
2) Automatically selects optimal threshold maximizing F1
3) Adds consensus_themes_{small,large,extra_large} to CSV
4) 100% reproducible based on data
5) Creates backup before modifying original file

Author:
-------
Antoine Lemor
"""

import pandas as pd
import json
import os
from collections import Counter

# Paths - go up 3 levels from scripts/data_processing/
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

# All political parties (Canadian federal + Quebec provincial)
ALL_PARTIES = [
    'LPC', 'CPC', 'NDP', 'BQ', 'GPC',  # Federal
    'CAQ', 'PLQ', 'PQ', 'QS', 'PCQ'     # Quebec provincial
]

# Specific themes
ALL_SPECIFIC_THEMES = [
    'early_learning_childcare', 'welfare_state', 'public_finance'
]

# All models (sorted for reproducibility)
ALL_MODELS = sorted(['nemotron', 'llama', 'gpt_oss', 'gemma', 'gpt_4_1', 'gpt_5'])


def parse_annotations(json_str):
    """Parse JSON annotations string."""
    if pd.isna(json_str):
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}


def get_model_values(annotations, model, category, suffix):
    """Extract values predicted by a specific model for a given category."""
    key = f'{model}_{category}_{suffix}'
    values = annotations.get(key, None)
    if values is None:
        return None
    # Handle both list and single value
    if isinstance(values, list):
        # Filter out malformed values (single characters)
        return set(v for v in values if isinstance(v, str) and len(v) > 2)
    elif isinstance(values, str):
        return values if len(values) > 2 else None
    return None


def get_model_themes(annotations, model, suffix):
    """Extract themes predicted by a specific model."""
    return get_model_values(annotations, model, 'themes', suffix)


def get_ground_truth_themes(annotations):
    """Extract themes from ground truth annotations."""
    themes = annotations.get('manual_themes', [])
    if themes is None:
        return set()
    return set(t.replace('theme_', '') for t in themes)


def get_ground_truth_parties(annotations):
    """Extract political parties from ground truth annotations."""
    parties = annotations.get('manual_political_parties', [])
    if parties is None:
        return set()
    return set(p.replace('political_parties_', '') for p in parties)


def get_ground_truth_specific_themes(annotations):
    """Extract specific themes from ground truth annotations."""
    specific = annotations.get('manual_specific_themes', [])
    if specific is None:
        return set()
    return set(s.replace('specific_themes_', '') for s in specific)


def create_consensus_multilabel(annotations, models, threshold, category, suffix):
    """
    Create consensus labels from multiple LLM annotations for multi-label categories.

    Args:
        annotations: Parsed JSON with LLM annotations
        models: List of models to consider
        threshold: Minimum number of LLMs that must agree
        category: 'themes', 'political_parties', or 'specific_themes'
        suffix: 'small', 'large', or 'extra_large'

    Returns:
        Set of values where >= threshold LLMs agree, number of models that voted
    """
    value_votes = Counter()
    models_voted = 0

    for model in models:
        values = get_model_values(annotations, model, category, suffix)
        if values is not None and isinstance(values, set):
            models_voted += 1
            for value in values:
                value_votes[value] += 1

    if models_voted == 0:
        return set(), 0

    # Values with >= threshold votes
    consensus_values = set(value for value, votes in value_votes.items()
                          if votes >= threshold)

    return consensus_values, models_voted


def create_consensus_sentiment(annotations, models, suffix):
    """
    Create consensus for sentiment (single-label) using majority vote.

    Args:
        annotations: Parsed JSON with LLM annotations
        models: List of models to consider
        suffix: 'small', 'large', or 'extra_large'

    Returns:
        Majority sentiment value or None, number of models that voted
    """
    sentiment_votes = Counter()
    models_voted = 0

    for model in models:
        sentiment = get_model_values(annotations, model, 'sentiment', suffix)
        if sentiment is not None and isinstance(sentiment, str):
            models_voted += 1
            sentiment_votes[sentiment] += 1

    if models_voted == 0:
        return None, 0

    # Get majority vote (most common sentiment)
    most_common = sentiment_votes.most_common(1)
    if most_common:
        sentiment, count = most_common[0]
        # Only return if at least 2 models agree (to avoid single-model decisions)
        if count >= 2:
            return sentiment, models_voted

    return None, models_voted


def create_consensus(annotations, models, threshold, suffix):
    """
    Create consensus labels from multiple LLM annotations (for themes).
    Kept for backward compatibility.
    """
    return create_consensus_multilabel(annotations, models, threshold, 'themes', suffix)


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_consensus(merged_df, models, threshold, suffix='large', category='themes'):
    """
    Evaluate consensus performance against ground truth for any category.

    Args:
        merged_df: DataFrame with gt_parsed and llm_parsed columns
        models: List of models to use for consensus
        threshold: Minimum number of LLMs that must agree
        suffix: 'small', 'large', or 'extra_large'
        category: 'themes', 'political_parties', or 'specific_themes'
    """
    tp = fp = fn = 0

    # Select the right ground truth function and label set
    if category == 'themes':
        get_gt = get_ground_truth_themes
        all_labels = ALL_THEMES
    elif category == 'political_parties':
        get_gt = get_ground_truth_parties
        all_labels = ALL_PARTIES
    elif category == 'specific_themes':
        get_gt = get_ground_truth_specific_themes
        all_labels = ALL_SPECIFIC_THEMES
    else:
        raise ValueError(f"Unknown category: {category}")

    for _, row in merged_df.iterrows():
        gt_values = get_gt(row['gt_parsed'])
        consensus_values, _ = create_consensus_multilabel(row['llm_parsed'], models, threshold, category, suffix)

        for label in all_labels:
            in_gt = label in gt_values
            in_pred = label in consensus_values

            if in_gt and in_pred:
                tp += 1
            elif in_pred and not in_gt:
                fp += 1
            elif in_gt and not in_pred:
                fn += 1

    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def find_optimal_consensus(merged_df, suffix='large', category='themes'):
    """
    Find optimal consensus configuration (models and threshold) for a specific category.

    Args:
        merged_df: DataFrame with gt_parsed and llm_parsed columns
        suffix: 'small', 'large', or 'extra_large'
        category: 'themes', 'political_parties', or 'specific_themes'

    Returns:
        Tuple of (best_models, best_threshold, best_f1, all_results)
    """
    print(f"\n  Recherche du consensus optimal pour '{category}' (suffix={suffix})...")

    all_results = []

    # Test different thresholds with all models
    for threshold in range(1, len(ALL_MODELS) + 1):
        metrics = evaluate_consensus(merged_df, ALL_MODELS, threshold, suffix, category)
        result = {
            'models': ALL_MODELS.copy(),
            'n_models': len(ALL_MODELS),
            'threshold': threshold,
            **metrics
        }
        all_results.append(result)
        print(f"    Seuil >= {threshold}: F1={metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")

    # Find best threshold
    best_result = max(all_results, key=lambda x: x['f1'])

    return ALL_MODELS.copy(), best_result['threshold'], best_result['f1'], all_results


def add_consensus_to_annotations(annotations_str, models, thresholds, suffix):
    """
    Add consensus annotations for ALL categories to existing annotations JSON.

    Args:
        annotations_str: Original JSON string
        models: List of models for consensus
        thresholds: Dict with threshold per category {'themes': X, 'political_parties': Y, 'specific_themes': Z}
        suffix: 'small', 'large', or 'extra_large'

    Returns:
        Updated JSON string with all consensus fields added, stats dict
    """
    if pd.isna(annotations_str):
        return annotations_str, {'themes': 0, 'parties': 0, 'specific': 0, 'sentiment': 0}

    try:
        annotations = json.loads(annotations_str)
    except:
        return annotations_str, {'themes': 0, 'parties': 0, 'specific': 0, 'sentiment': 0}

    stats = {'themes': 0, 'parties': 0, 'specific': 0, 'sentiment': 0}

    # Consensus for themes (multi-label, category-specific threshold)
    consensus_themes, n_models = create_consensus_multilabel(
        annotations, models, thresholds['themes'], 'themes', suffix)
    annotations[f'consensus_themes_{suffix}'] = sorted(list(consensus_themes)) if consensus_themes else None
    if consensus_themes:
        stats['themes'] = 1

    # Consensus for political_parties (multi-label, category-specific threshold)
    consensus_parties, n_models = create_consensus_multilabel(
        annotations, models, thresholds['political_parties'], 'political_parties', suffix)
    annotations[f'consensus_political_parties_{suffix}'] = sorted(list(consensus_parties)) if consensus_parties else None
    if consensus_parties:
        stats['parties'] = 1

    # Consensus for specific_themes (multi-label, category-specific threshold)
    consensus_specific, n_models = create_consensus_multilabel(
        annotations, models, thresholds['specific_themes'], 'specific_themes', suffix)
    annotations[f'consensus_specific_themes_{suffix}'] = sorted(list(consensus_specific)) if consensus_specific else None
    if consensus_specific:
        stats['specific'] = 1

    # Consensus for sentiment (single-label, majority vote - no threshold needed)
    consensus_sentiment, n_models = create_consensus_sentiment(annotations, models, suffix)
    annotations[f'consensus_sentiment_{suffix}'] = consensus_sentiment
    if consensus_sentiment:
        stats['sentiment'] = 1

    return json.dumps(annotations, ensure_ascii=False), stats


def main():
    print("=" * 70)
    print("AJOUT DU MODÈLE CONSENSUS À LLMs_training_set.csv")
    print("=" * 70)

    # Load data
    print("\n[1/5] Chargement des données...")
    benchmark_df = pd.read_csv(BENCHMARK)
    training_df = pd.read_csv(TRAINING_SET)

    print(f"  Benchmark: {len(benchmark_df)} samples")
    print(f"  Training set: {len(training_df)} samples")

    # Prepare evaluation data (merge benchmark with training by text)
    print("\n[2/5] Préparation des données d'évaluation...")
    benchmark_dedup = benchmark_df.drop_duplicates(subset='text', keep='first')
    training_dedup = training_df.drop_duplicates(subset='text', keep='first')

    merged = benchmark_dedup.merge(
        training_dedup[['text', 'annotations_large']],
        on='text',
        how='inner'
    )
    print(f"  Samples pour évaluation: {len(merged)}")

    # Parse annotations for evaluation
    merged['gt_parsed'] = merged['annotations'].apply(parse_annotations)
    merged['llm_parsed'] = merged['annotations_large'].apply(parse_annotations)

    # Find optimal consensus for EACH CATEGORY separately
    print("\n[3/5] Recherche du consensus optimal PAR CATÉGORIE...")

    # Optimize each category independently using 'large' annotations
    category_configs = {}

    # 1. Themes
    print("\n  --- THEMES ---")
    _, themes_threshold, themes_f1, _ = find_optimal_consensus(merged, 'large', 'themes')
    category_configs['themes'] = {'threshold': themes_threshold, 'f1': themes_f1}

    # 2. Political parties
    print("\n  --- POLITICAL PARTIES ---")
    _, parties_threshold, parties_f1, _ = find_optimal_consensus(merged, 'large', 'political_parties')
    category_configs['political_parties'] = {'threshold': parties_threshold, 'f1': parties_f1}

    # 3. Specific themes
    print("\n  --- SPECIFIC THEMES ---")
    _, specific_threshold, specific_f1, _ = find_optimal_consensus(merged, 'large', 'specific_themes')
    category_configs['specific_themes'] = {'threshold': specific_threshold, 'f1': specific_f1}

    # Build thresholds dict for each suffix (same thresholds, different suffix)
    optimal_configs = {}
    for suffix in ['small', 'large', 'extra_large']:
        optimal_configs[suffix] = {
            'models': ALL_MODELS.copy(),
            'thresholds': {
                'themes': category_configs['themes']['threshold'],
                'political_parties': category_configs['political_parties']['threshold'],
                'specific_themes': category_configs['specific_themes']['threshold']
            },
            'f1_per_category': {
                'themes': category_configs['themes']['f1'],
                'political_parties': category_configs['political_parties']['f1'],
                'specific_themes': category_configs['specific_themes']['f1']
            }
        }

    # Display optimal configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION OPTIMALE PAR CATÉGORIE")
    print("=" * 70)

    print(f"\n  Modèles utilisés: {', '.join(ALL_MODELS)}")
    print(f"\n  Seuils optimaux:")
    print(f"    - themes:            >= {category_configs['themes']['threshold']} LLMs (F1={category_configs['themes']['f1']:.4f})")
    print(f"    - political_parties: >= {category_configs['political_parties']['threshold']} LLMs (F1={category_configs['political_parties']['f1']:.4f})")
    print(f"    - specific_themes:   >= {category_configs['specific_themes']['threshold']} LLMs (F1={category_configs['specific_themes']['f1']:.4f})")

    # Apply consensus to all rows in training set
    print("\n[4/5] Application du consensus à toutes les lignes...")

    # Create a copy of the training dataframe
    training_updated = training_df.copy()

    # Process each row
    total_rows = len(training_updated)

    for suffix in ['small', 'large', 'extra_large']:
        print(f"\n  Traitement de annotations_{suffix}...")

        col_name = f'annotations_{suffix}'
        thresholds = optimal_configs[suffix]['thresholds']  # Dict per category
        models = optimal_configs[suffix]['models']

        updated_annotations = []
        total_stats = {'themes': 0, 'parties': 0, 'specific': 0, 'sentiment': 0}
        rows_processed = 0

        for _, row in training_updated.iterrows():
            annotations_str = row[col_name]

            if pd.isna(annotations_str):
                updated_annotations.append(annotations_str)
                continue

            rows_processed += 1

            # Add consensus for ALL categories with category-specific thresholds
            updated_str, stats = add_consensus_to_annotations(annotations_str, models, thresholds, suffix)
            updated_annotations.append(updated_str)

            # Aggregate stats
            for key in total_stats:
                total_stats[key] += stats[key]

        training_updated[col_name] = updated_annotations
        print(f"    Lignes traitées: {rows_processed}/{total_rows}")
        print(f"    - consensus_themes: {total_stats['themes']} lignes (seuil>={thresholds['themes']})")
        print(f"    - consensus_political_parties: {total_stats['parties']} lignes (seuil>={thresholds['political_parties']})")
        print(f"    - consensus_specific_themes: {total_stats['specific']} lignes (seuil>={thresholds['specific_themes']})")
        print(f"    - consensus_sentiment: {total_stats['sentiment']} lignes (majority vote)")

    # Save updated file
    print("\n[5/5] Sauvegarde du fichier mis à jour...")

    # Backup original file
    backup_path = TRAINING_SET.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_path):
        training_df.to_csv(backup_path, index=False)
        print(f"  Backup créé: {backup_path}")

    # Save updated file
    training_updated.to_csv(TRAINING_SET, index=False)
    print(f"  Fichier mis à jour: {TRAINING_SET}")

    # Save configuration for reproducibility
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path = os.path.join(OUTPUT_DIR, 'consensus_config.json')

    config_to_save = {
        'models': ALL_MODELS,
        'thresholds_per_category': {
            'themes': category_configs['themes']['threshold'],
            'political_parties': category_configs['political_parties']['threshold'],
            'specific_themes': category_configs['specific_themes']['threshold'],
            'sentiment': 'majority_vote (>=2)'
        },
        'f1_per_category': {
            'themes': category_configs['themes']['f1'],
            'political_parties': category_configs['political_parties']['f1'],
            'specific_themes': category_configs['specific_themes']['f1']
        },
        'applied_to': ['annotations_small', 'annotations_large', 'annotations_extra_large'],
        'consensus_keys_added': [
            'consensus_themes_{suffix}',
            'consensus_political_parties_{suffix}',
            'consensus_specific_themes_{suffix}',
            'consensus_sentiment_{suffix}'
        ]
    }

    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"  Configuration sauvegardée: {config_path}")

    # Verify
    print("\n" + "=" * 70)
    print("VÉRIFICATION")
    print("=" * 70)

    # Reload and check
    verification_df = pd.read_csv(TRAINING_SET)
    sample_row = verification_df[verification_df['annotations_large'].notna()].iloc[0]
    sample_annotations = json.loads(sample_row['annotations_large'])

    print(f"\n  Exemple de ligne mise à jour:")
    print(f"    Texte: {sample_row['text'][:60]}...")

    # Show all consensus fields
    print(f"\n    Consensus calculés:")
    print(f"      themes: {sample_annotations.get('consensus_themes_large', None)}")
    print(f"      political_parties: {sample_annotations.get('consensus_political_parties_large', None)}")
    print(f"      specific_themes: {sample_annotations.get('consensus_specific_themes_large', None)}")
    print(f"      sentiment: {sample_annotations.get('consensus_sentiment_large', None)}")

    # Count themes from individual models
    print(f"\n    Détail par modèle (themes):")
    for model in ALL_MODELS:
        themes = sample_annotations.get(f'{model}_themes_large', None)
        if themes:
            print(f"      {model}: {themes}")

    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    print(f"\nLe modèle 'consensus' a été ajouté avec des seuils OPTIMISÉS PAR CATÉGORIE:")
    print(f"  - Modèles: {', '.join(ALL_MODELS)}")
    print(f"  - Seuils optimaux:")
    print(f"      themes:            >= {category_configs['themes']['threshold']} LLMs (F1={category_configs['themes']['f1']:.4f})")
    print(f"      political_parties: >= {category_configs['political_parties']['threshold']} LLMs (F1={category_configs['political_parties']['f1']:.4f})")
    print(f"      specific_themes:   >= {category_configs['specific_themes']['threshold']} LLMs (F1={category_configs['specific_themes']['f1']:.4f})")
    print(f"      sentiment:         majority vote (>=2 LLMs)")

    return optimal_configs


if __name__ == '__main__':
    configs = main()

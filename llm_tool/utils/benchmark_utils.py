#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
benchmark_utils.py

MAIN OBJECTIVE:
---------------
Utilities for sophisticated model benchmarking including class imbalance analysis,
category selection, and multi-model comparison.

Dependencies:
-------------
- pandas
- numpy
- sklearn
- rich

MAIN FEATURES:
--------------
1) Class imbalance analysis for categorical data
2) Category selection based on imbalance profiles
3) Benchmark execution with multiple models
4) Results comparison and ranking
5) Statistical significance testing

Author:
-------
Antoine Lemor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging
from pathlib import Path


def calculate_class_imbalance(labels: List[Any]) -> Dict[str, float]:
    """
    Calculate class imbalance metrics for a set of labels.

    Args:
        labels: List of label values

    Returns:
        Dict with imbalance metrics:
        - 'imbalance_ratio': max_count / min_count
        - 'gini_index': Gini coefficient (0=perfect balance, 1=maximum imbalance)
        - 'entropy': Shannon entropy (higher=more balanced)
        - 'cv': Coefficient of variation
    """
    if not labels:
        return {
            'imbalance_ratio': 1.0,
            'gini_index': 0.0,
            'entropy': 0.0,
            'cv': 0.0,
            'num_classes': 0,
            'total_samples': 0
        }

    # Count occurrences
    counts = Counter(labels)
    values = np.array(list(counts.values()))
    total = len(labels)

    # Calculate metrics
    if len(counts) == 1:
        # Only one class
        return {
            'imbalance_ratio': 1.0,
            'gini_index': 1.0,  # Maximum imbalance (all in one class)
            'entropy': 0.0,
            'cv': 0.0,
            'num_classes': 1,
            'total_samples': total,
            'min_count': values[0],
            'max_count': values[0],
            'mean_count': values[0]
        }

    # Imbalance ratio
    imbalance_ratio = values.max() / values.min()

    # Gini index (0 = perfect equality, 1 = maximum inequality)
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    # Shannon entropy (higher = more balanced)
    proportions = values / total
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
    max_entropy = np.log2(len(counts))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Coefficient of variation
    cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

    return {
        'imbalance_ratio': float(imbalance_ratio),
        'gini_index': float(gini),
        'entropy': float(normalized_entropy),
        'cv': float(cv),
        'num_classes': len(counts),
        'total_samples': total,
        'min_count': int(values.min()),
        'max_count': int(values.max()),
        'mean_count': float(values.mean()),
        'class_distribution': dict(counts)
    }


def analyze_categories_imbalance(data: pd.DataFrame, annotation_column: str) -> Dict[str, Dict]:
    """
    Analyze class imbalance for all categories in annotation data.

    Args:
        data: DataFrame with annotations
        annotation_column: Column containing annotations (JSON or dict format)

    Returns:
        Dict mapping category names to their imbalance metrics
    """
    import json

    # Parse annotations
    categories_data = {}

    for idx, row in data.iterrows():
        annotation = row[annotation_column]

        # Parse if string
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)
            except:
                continue

        if not isinstance(annotation, dict):
            continue

        # Collect values for each category
        for key, value in annotation.items():
            if key not in categories_data:
                categories_data[key] = []

            # Handle different value formats
            if isinstance(value, list):
                categories_data[key].extend(value)
            elif value:  # Skip None/empty
                categories_data[key].append(value)

    # Calculate imbalance for each category
    results = {}
    for category, values in categories_data.items():
        if values:  # Only analyze categories with data
            results[category] = calculate_class_imbalance(values)
            results[category]['category_name'] = category

    return results


def select_benchmark_categories(
    imbalance_analysis: Dict[str, Dict],
    num_categories: int = 3
) -> Dict[str, List[str]]:
    """
    Select categories for benchmarking based on imbalance profiles.

    Selects:
    - One balanced category (low imbalance)
    - One medium imbalance category
    - One high imbalance category

    Args:
        imbalance_analysis: Results from analyze_categories_imbalance
        num_categories: Number of categories to select (default 3)

    Returns:
        Dict with 'balanced', 'medium', 'imbalanced' keys mapping to category names
    """
    if not imbalance_analysis:
        return {'balanced': [], 'medium': [], 'imbalanced': []}

    # Filter categories with sufficient data (at least 2 classes, at least 20 samples)
    valid_categories = {
        name: metrics
        for name, metrics in imbalance_analysis.items()
        if metrics['num_classes'] >= 2 and metrics['total_samples'] >= 20
    }

    if not valid_categories:
        return {'balanced': [], 'medium': [], 'imbalanced': []}

    # Sort by Gini index (primary) and imbalance ratio (secondary)
    sorted_categories = sorted(
        valid_categories.items(),
        key=lambda x: (x[1]['gini_index'], x[1]['imbalance_ratio'])
    )

    result = {'balanced': [], 'medium': [], 'imbalanced': []}

    if len(sorted_categories) == 1:
        # Only one category available
        result['medium'] = [sorted_categories[0][0]]
    elif len(sorted_categories) == 2:
        # Two categories available
        result['balanced'] = [sorted_categories[0][0]]
        result['imbalanced'] = [sorted_categories[1][0]]
    elif len(sorted_categories) >= 3:
        # Three or more categories
        # Most balanced
        result['balanced'] = [sorted_categories[0][0]]
        # Most imbalanced
        result['imbalanced'] = [sorted_categories[-1][0]]
        # Medium (closest to median)
        mid_idx = len(sorted_categories) // 2
        result['medium'] = [sorted_categories[mid_idx][0]]

    return result


def format_imbalance_summary(metrics: Dict) -> str:
    """
    Format imbalance metrics into human-readable summary.

    Args:
        metrics: Imbalance metrics dict

    Returns:
        Formatted string
    """
    if not metrics:
        return "No data"

    # Classify imbalance level
    gini = metrics.get('gini_index', 0)
    ratio = metrics.get('imbalance_ratio', 1)

    if gini < 0.2 and ratio < 2:
        level = "[green]Balanced[/green]"
        emoji = "⚖️"
    elif gini < 0.4 and ratio < 5:
        level = "[yellow]Moderate[/yellow]"
        emoji = "📊"
    else:
        level = "[red]Imbalanced[/red]"
        emoji = "⚠️"

    return (
        f"{emoji} {level} | "
        f"Ratio: {ratio:.1f}:1 | "
        f"Gini: {gini:.2f} | "
        f"{metrics.get('num_classes', 0)} classes | "
        f"{metrics.get('total_samples', 0)} samples"
    )


def create_benchmark_dataset(
    data: pd.DataFrame,
    annotation_column: str,
    selected_categories: List[str],
    text_column: str = 'text',
    output_path: Optional[Path] = None
) -> Path:
    """
    Create a focused dataset for benchmarking with selected categories.

    Args:
        data: Full dataset
        annotation_column: Column with annotations
        selected_categories: List of categories to include
        text_column: Column with text data
        output_path: Where to save the dataset

    Returns:
        Path to created dataset file
    """
    import json
    from datetime import datetime

    # Filter to only selected categories
    filtered_rows = []

    for idx, row in data.iterrows():
        annotation = row[annotation_column]

        # Parse if string
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)
            except:
                continue

        if not isinstance(annotation, dict):
            continue

        # Check if has any selected category
        has_category = any(cat in annotation for cat in selected_categories)
        if not has_category:
            continue

        # Filter annotation to only selected categories
        filtered_annotation = {
            k: v for k, v in annotation.items()
            if k in selected_categories
        }

        if not filtered_annotation:
            continue

        # Create filtered row
        filtered_row = {
            'text': row[text_column],
            'annotation': filtered_annotation
        }

        # Add language if available
        for lang_col in ['language', 'lang']:
            if lang_col in row:
                filtered_row['lang'] = row[lang_col]
                break

        # Add ID if available
        for id_col in ['id', 'video_id', 'document_id']:
            if id_col in row:
                filtered_row['id'] = row[id_col]
                break

        filtered_rows.append(filtered_row)

    # Create DataFrame
    benchmark_df = pd.DataFrame(filtered_rows)

    # Save to JSONL
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/benchmark/benchmark_dataset_{timestamp}.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_json(output_path, orient='records', lines=True, force_ascii=False)

    logging.info(f"Created benchmark dataset: {output_path}")
    logging.info(f"  • Rows: {len(benchmark_df)}")
    logging.info(f"  • Categories: {', '.join(selected_categories)}")

    return output_path


def compare_model_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare results from multiple models and create ranking.

    Args:
        results: Dict mapping model_id to training results

    Returns:
        DataFrame with model comparison and ranking
    """
    comparison_data = []

    for model_id, result in results.items():
        row = {
            'model': model_id,
            'f1_macro': result.get('f1_macro', result.get('best_f1_macro', 0)),
            'accuracy': result.get('accuracy', result.get('best_accuracy', 0)),
            'precision': result.get('precision', 0),
            'recall': result.get('recall', 0),
            'training_time': result.get('training_time', 0),
        }
        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Sort by F1 (primary), then accuracy (secondary)
    df = df.sort_values(['f1_macro', 'accuracy'], ascending=False)

    # Add rank
    df.insert(0, 'rank', range(1, len(df) + 1))

    return df

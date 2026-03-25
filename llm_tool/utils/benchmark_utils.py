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


def analyze_categories_imbalance(
    data: pd.DataFrame,
    annotation_column: str,
    filter_categories: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Analyze class imbalance for all categories in annotation data.

    Args:
        data: DataFrame with annotations
        annotation_column: Column containing annotations (JSON or dict format)
        filter_categories: Optional list of category names to analyze (ignores others)

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
            # CRITICAL: Skip categories not in filter_categories (if provided)
            if filter_categories is not None and key not in filter_categories:
                continue

            if key not in categories_data:
                categories_data[key] = []

            # Handle different value formats
            # CRITICAL: Skip 'null' string values
            if isinstance(value, list):
                clean_values = [v for v in value if v and v != 'null']
                categories_data[key].extend(clean_values)
            elif value and value != 'null':  # Skip None/empty/'null'
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

    This function strategically selects categories with different class distribution
    characteristics to provide a comprehensive benchmark:

    Selection Strategy:
    - One BALANCED category (Gini < 0.2, ratio < 2:1)
      → Tests baseline performance on easy, well-distributed data
    - One MEDIUM imbalance category (Gini 0.2-0.4, ratio 2-5:1)
      → Tests performance on moderately challenging data
    - One HIGHLY IMBALANCED category (Gini > 0.4, ratio > 5:1)
      → Tests robustness on real-world skewed distributions

    Why This Matters:
    - Models performing well only on balanced data may fail in production
    - Models handling imbalanced data well are more robust and production-ready
    - Comparing performance across profiles reveals true model capabilities

    Args:
        imbalance_analysis: Results from analyze_categories_imbalance
        num_categories: Number of categories to select (default 3)

    Returns:
        Dict with 'balanced', 'medium', 'imbalanced' keys mapping to category names
        Each key contains a list of category names matching that imbalance profile

    Example:
        >>> analysis = analyze_categories_imbalance(df, 'annotations')
        >>> selected = select_benchmark_categories(analysis)
        >>> selected
        {
            'balanced': ['sentiment'],
            'medium': ['topics'],
            'imbalanced': ['rare_events']
        }
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


def compare_model_results(
    results: Dict[str, Dict],
    f1_class_1_weight: float = 0.7,
    use_sophisticated_ranking: bool = True
) -> pd.DataFrame:
    """
    Compare results from multiple models and create sophisticated ranking.

    The ranking system mirrors the epoch selection criteria used during training,
    ensuring consistency between how epochs and models are evaluated.

    Ranking Methodology (when use_sophisticated_ranking=True):
    --------------------------------------------------------
    1. COMBINED METRIC CALCULATION (similar to best epoch selection):
       For binary classification:
       - Combined Score = (f1_class_1_weight × F1_class_1) + ((1-f1_class_1_weight) × F1_macro)
       - Default: 70% weight on minority class F1, 30% on macro F1
       - This prioritizes models that detect the minority class well

       For multi-class classification:
       - Combined Score = F1_macro (balanced across all classes)

    2. LANGUAGE BALANCE PENALTY (if multilingual data):
       - Calculates coefficient of variation (CV) across language-specific F1 scores
       - Penalizes models that perform well in one language but poorly in another
       - Penalty = min(CV × 0.2, 0.2) → up to 20% score reduction
       - Example: Model with F1=0.9 (EN) and F1=0.3 (FR) gets penalized

    3. ACCURACY AS TIEBREAKER:
       - When combined scores are equal, higher accuracy wins
       - Ensures models with similar F1 are differentiated

    4. TRAINING TIME AS FINAL TIEBREAKER:
       - When combined score AND accuracy are equal, faster model wins
       - Promotes efficiency in production deployments

    Args:
        results: Dict mapping model_id to training results dict containing:
                - f1_macro or best_f1_macro: Macro-averaged F1 score
                - accuracy or best_accuracy: Overall accuracy
                - f1_0, f1_1: Class-specific F1 scores (binary)
                - precision_0, precision_1, recall_0, recall_1: Per-class metrics
                - language_metrics: Dict of {language: metrics} (optional)
                - training_time: Time in seconds
        f1_class_1_weight: Weight for F1 class 1 in combined metric (0.0-1.0)
                          Default 0.7 means 70% class 1, 30% macro F1
        use_sophisticated_ranking: If True, uses combined metric with language penalties.
                                  If False, simple F1 macro ranking (legacy behavior)

    Returns:
        DataFrame with columns:
        - rank: 1-based rank (1=best)
        - model: model identifier
        - combined_score: The sophisticated combined metric used for ranking
        - f1_macro: Macro F1 score
        - f1_class_1: F1 for minority class (binary only)
        - accuracy: Overall accuracy
        - precision, recall: Overall metrics
        - training_time: Training duration
        - ranking_explanation: Text explaining why this rank was assigned

    Example:
        >>> results = {
        ...     'xlm-roberta-base': {
        ...         'f1_macro': 0.75, 'f1_0': 0.85, 'f1_1': 0.65,
        ...         'accuracy': 0.78, 'training_time': 120.5,
        ...         'language_metrics': {'EN': {'f1_1': 0.70}, 'FR': {'f1_1': 0.60}}
        ...     },
        ...     'bert-base': {
        ...         'f1_macro': 0.73, 'f1_0': 0.80, 'f1_1': 0.66,
        ...         'accuracy': 0.76, 'training_time': 95.2
        ...     }
        ... }
        >>> df = compare_model_results(results)
        >>> print(df[['rank', 'model', 'combined_score', 'f1_macro']])
    """
    import numpy as np

    comparison_data = []

    for model_id, result in results.items():
        # Extract base metrics - handle multiple possible key names for backward compatibility
        # Priority: f1_macro > f1 > best_f1_macro (different return formats from model_trainer.py)
        f1_macro = result.get('f1_macro', result.get('f1', result.get('best_f1_macro', 0)))
        f1_0 = result.get('f1_0', result.get('f1_class_0', result.get('best_f1_0', 0)))
        f1_1 = result.get('f1_1', result.get('f1_class_1', result.get('best_f1_1', 0)))
        accuracy = result.get('accuracy', result.get('best_accuracy', 0))
        precision = result.get('precision', result.get('macro_precision', 0))
        recall = result.get('recall', result.get('macro_recall', 0))
        training_time = result.get('training_time', 0)

        # Get language-specific metrics if available
        language_metrics = result.get('language_metrics', {})

        # Calculate combined metric (matching epoch selection logic)
        if use_sophisticated_ranking:
            # Detect if binary classification (has f1_1) or multi-class
            is_binary = f1_1 > 0 or 'f1_1' in result or 'best_f1_1' in result

            if is_binary and f1_macro > 0:
                # Binary: weighted combination of F1_class_1 and macro F1
                combined_score = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * f1_macro
            else:
                # Multi-class: use macro F1 directly
                combined_score = f1_macro

            # Apply language balance penalty if multilingual
            language_penalty = 0.0
            if language_metrics and len(language_metrics) > 1:
                # Extract F1 scores across languages
                if is_binary:
                    # Binary: use F1_class_1 per language
                    f1_values = [
                        lang_data.get('f1_1', 0)
                        for lang_data in language_metrics.values()
                        if lang_data.get('support_1', 0) > 0  # Only languages with positive examples
                    ]
                else:
                    # Multi-class: use macro F1 per language
                    # CRITICAL: Try f1_macro first (new standard), fallback to macro_f1
                    f1_values = [
                        lang_data.get('f1_macro', lang_data.get('macro_f1', 0))
                        for lang_data in language_metrics.values()
                    ]

                # Calculate coefficient of variation (CV) as imbalance measure
                if len(f1_values) > 1:
                    mean_f1 = np.mean(f1_values)
                    std_f1 = np.std(f1_values)

                    if mean_f1 > 0:
                        cv = std_f1 / mean_f1  # Coefficient of variation
                        # Penalty: up to 20% reduction based on variance
                        language_penalty = min(cv * 0.2, 0.2)

            # Apply penalty to combined score
            combined_score = combined_score * (1 - language_penalty)

            # Generate ranking explanation
            explanation_parts = []
            if is_binary:
                explanation_parts.append(
                    f"Combined: {f1_class_1_weight:.0%}×F1₁({f1_1:.3f}) + "
                    f"{(1-f1_class_1_weight):.0%}×F1ₘ({f1_macro:.3f})"
                )
            else:
                explanation_parts.append(f"F1 Macro: {f1_macro:.3f}")

            if language_penalty > 0:
                explanation_parts.append(f"Lang penalty: -{language_penalty:.1%}")

            ranking_explanation = "; ".join(explanation_parts)
        else:
            # Legacy: simple F1 macro ranking
            combined_score = f1_macro
            ranking_explanation = f"Simple F1 Macro: {f1_macro:.3f}"

        row = {
            'model': model_id,
            'combined_score': combined_score,
            'f1_macro': f1_macro,
            'f1_class_0': f1_0,
            'f1_class_1': f1_1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_time': training_time,
            'language_balance_penalty': language_penalty if use_sophisticated_ranking else 0,
            'ranking_explanation': ranking_explanation
        }
        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Sophisticated ranking: Combined score (primary), Accuracy (secondary), Time (tertiary)
    df = df.sort_values(
        ['combined_score', 'accuracy', 'training_time'],
        ascending=[False, False, True]  # Lower time is better
    )

    # Add rank
    df.insert(0, 'rank', range(1, len(df) + 1))

    return df


def consolidate_session_csvs(session_dir: Path, session_id: str) -> Dict[str, Path]:
    """
    Consolidate all CSV files from a training session into summary files.

    Creates two consolidated CSV files at the session root:
    - {session_id}_training_metrics.csv: All training metrics from ALL epochs and ALL modes
    - {session_id}_best_models.csv: ONLY the final best selected models with combined scores

    Args:
        session_dir: Path to the session directory (e.g., logs/training_arena/20251007_141900/training_metrics)
        session_id: Session ID timestamp string

    Returns:
        Dict with paths to created files: {'training': Path, 'best': Path}
    """
    import glob
    import os
    import numpy as np

    # Find all training.csv, reinforced.csv, and best.csv files recursively
    training_csvs = list(session_dir.rglob("training.csv"))
    reinforced_csvs = list(session_dir.rglob("reinforced.csv"))
    best_csvs = list(session_dir.rglob("best.csv"))

    # Detect if this is benchmark mode by checking if any file contains "benchmark" in path
    # Note: We save consolidated files at session root if ANY benchmark files exist
    is_benchmark_mode = False
    for csv_path in training_csvs + reinforced_csvs + best_csvs:
        if 'benchmark' in str(csv_path) or '/benchmark/' in str(csv_path):
            is_benchmark_mode = True
            break

    # Determine output directory
    # For benchmark mode: save to session root (parent of training_metrics)
    # For normal mode: save to session_dir (training_metrics directory)
    if is_benchmark_mode:
        output_dir = session_dir.parent
    else:
        output_dir = session_dir

    consolidated_files = {}

    # Consolidate training metrics (both normal training.csv and reinforced.csv)
    # CRITICAL: This should contain ALL epochs from ALL modes (benchmark, normal, reinforced)
    if training_csvs or reinforced_csvs:
        all_training_data = []

        # CRITICAL: Extract legends from CSV files (lines starting with #)
        collected_legends = set()  # Use set to avoid duplicates

        # Process normal training files (includes benchmark mode and normal training mode)
        for csv_path in training_csvs:
            try:
                # CRITICAL: Extract legend from first line if it exists
                with open(csv_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        collected_legends.add(first_line)

                # Read CSV, skipping comment lines
                df = pd.read_csv(csv_path, comment='#')

                # Extract metadata from path
                # Path structure: session_dir/[benchmark|normal_training]/category/[language/][model/]training.csv
                rel_path = csv_path.relative_to(session_dir)
                parts = list(rel_path.parts[:-1])  # Exclude 'training.csv'

                # Add path-based metadata
                metadata = {}
                metadata['phase'] = 'normal'  # Mark as normal training (vs reinforced)

                # Check if benchmark or normal_training mode
                if parts and parts[0] == 'benchmark':
                    metadata['mode'] = 'benchmark'
                    parts = parts[1:]  # Remove 'benchmark' from parts
                elif parts and parts[0] == 'normal_training':
                    metadata['mode'] = 'normal_training'  # Keep full mode name
                    parts = parts[1:]  # Remove 'normal_training' from parts
                else:
                    # Legacy: old structure without benchmark or normal_training prefix
                    metadata['mode'] = 'unknown'

                # Extract category, language, model from remaining parts
                if len(parts) >= 1:
                    metadata['category'] = parts[0]
                if len(parts) >= 2:
                    metadata['language'] = parts[1]
                if len(parts) >= 3:
                    metadata['model'] = parts[2]

                # Add metadata columns to dataframe
                for key, value in metadata.items():
                    if key not in df.columns:  # Only add if not already present
                        df[key] = value

                all_training_data.append(df)

            except Exception as e:
                logging.warning(f"Failed to read {csv_path}: {e}")
                continue

        # Process reinforced training files
        for csv_path in reinforced_csvs:
            try:
                # CRITICAL: Extract legend from first line if it exists
                with open(csv_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        collected_legends.add(first_line)

                # Read CSV, skipping comment lines
                df = pd.read_csv(csv_path, comment='#')

                # Extract metadata from path
                # Path structure: session_dir/[benchmark|normal_training]/category/[language/][model/]reinforced.csv
                rel_path = csv_path.relative_to(session_dir)
                parts = list(rel_path.parts[:-1])  # Exclude 'reinforced.csv'

                # Add path-based metadata
                metadata = {}
                metadata['phase'] = 'reinforced'  # Mark as reinforced training

                # Check if benchmark or normal_training mode
                if parts and parts[0] == 'benchmark':
                    metadata['mode'] = 'benchmark'
                    parts = parts[1:]  # Remove 'benchmark' from parts
                elif parts and parts[0] == 'normal_training':
                    metadata['mode'] = 'normal_training'  # Keep full mode name
                    parts = parts[1:]  # Remove 'normal_training' from parts
                else:
                    # Legacy: old structure without benchmark or normal_training prefix
                    metadata['mode'] = 'unknown'

                # Extract category, language, model from remaining parts
                if len(parts) >= 1:
                    metadata['category'] = parts[0]
                if len(parts) >= 2:
                    metadata['language'] = parts[1]
                if len(parts) >= 3:
                    metadata['model'] = parts[2]

                # Add metadata columns to dataframe
                for key, value in metadata.items():
                    if key not in df.columns:  # Only add if not already present
                        df[key] = value

                all_training_data.append(df)

            except Exception as e:
                logging.warning(f"Failed to read {csv_path}: {e}")
                continue

        if all_training_data:
            # Combine all dataframes
            consolidated_df = pd.concat(all_training_data, ignore_index=True)

            # Reorder columns to put metadata first
            metadata_cols = ['phase', 'mode', 'category', 'language', 'model']
            other_cols = [col for col in consolidated_df.columns if col not in metadata_cols]

            # Only include metadata columns that exist
            existing_metadata_cols = [col for col in metadata_cols if col in consolidated_df.columns]
            consolidated_df = consolidated_df[existing_metadata_cols + other_cols]

            # Save consolidated training metrics (to output_dir determined above)
            training_output = output_dir / f"{session_id}_training_metrics.csv"

            # CRITICAL: Write legends as comment lines first
            with open(training_output, 'w', encoding='utf-8', newline='') as f:
                # Write all collected legends
                for legend in sorted(collected_legends):
                    f.write(f"{legend}\n")

            # Append the CSV data
            consolidated_df.to_csv(training_output, mode='a', index=False)
            consolidated_files['training'] = training_output
            logging.info(f"Created consolidated training metrics: {training_output}")
            logging.info(f"  • Total rows: {len(consolidated_df)}")
            logging.info(f"  • Normal training rows: {len(consolidated_df[consolidated_df['phase'] == 'normal']) if 'phase' in consolidated_df.columns else 'N/A'}")
            logging.info(f"  • Reinforced training rows: {len(consolidated_df[consolidated_df['phase'] == 'reinforced']) if 'phase' in consolidated_df.columns else 'N/A'}")

    # Consolidate best models
    if best_csvs:
        all_best_data = []

        # CRITICAL: Extract legends from CSV files (lines starting with #)
        best_legends = set()  # Use set to avoid duplicates

        for csv_path in best_csvs:
            try:
                # CRITICAL: Extract legend from first line if it exists
                with open(csv_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        best_legends.add(first_line)

                # Read CSV, skipping comment lines
                df = pd.read_csv(csv_path, comment='#')

                # Extract metadata from path
                # Path structure: session_dir/[benchmark|normal_training]/category/[language/][model/]best.csv
                rel_path = csv_path.relative_to(session_dir)
                parts = list(rel_path.parts[:-1])  # Exclude 'best.csv'

                # Add path-based metadata
                metadata = {}

                # Check if benchmark or normal_training mode
                if parts and parts[0] == 'benchmark':
                    metadata['mode'] = 'benchmark'
                    parts = parts[1:]
                elif parts and parts[0] == 'normal_training':
                    metadata['mode'] = 'normal_training'  # Keep full mode name for consistency
                    parts = parts[1:]
                else:
                    # Legacy: old structure without benchmark or normal_training prefix
                    metadata['mode'] = 'unknown'

                # Extract category, language, model from remaining parts
                if len(parts) >= 1:
                    metadata['category'] = parts[0]
                if len(parts) >= 2:
                    metadata['language'] = parts[1]
                if len(parts) >= 3:
                    metadata['model'] = parts[2]

                # Add metadata columns
                for key, value in metadata.items():
                    df[key] = value

                all_best_data.append(df)

            except Exception as e:
                logging.warning(f"Failed to read {csv_path}: {e}")
                continue

        if all_best_data:
            # Combine all dataframes
            consolidated_df = pd.concat(all_best_data, ignore_index=True)

            # Reorder columns to put metadata first
            metadata_cols = ['mode', 'category', 'language', 'model']
            other_cols = [col for col in consolidated_df.columns if col not in metadata_cols]

            # Only include metadata columns that exist
            existing_metadata_cols = [col for col in metadata_cols if col in consolidated_df.columns]
            consolidated_df = consolidated_df[existing_metadata_cols + other_cols]

            # CRITICAL: Calculate combined_score if not present
            # This ensures the final CSV has combined scores for ranking
            if 'combined_score' not in consolidated_df.columns:
                logging.info("Calculating combined_score for best models...")

                # Detect if binary or multi-class based on columns
                f1_cols = [col for col in consolidated_df.columns if col.startswith('f1_') and col[3:].isdigit()]
                num_labels = len(f1_cols)

                if num_labels == 2 and 'f1_1' in consolidated_df.columns:
                    # Binary classification: 70% F1_class_1 + 30% F1_macro
                    f1_class_1_weight = 0.7
                    consolidated_df['combined_score'] = (
                        f1_class_1_weight * consolidated_df['f1_1'].fillna(0) +
                        (1 - f1_class_1_weight) * consolidated_df['macro_f1'].fillna(0)
                    )
                elif 'macro_f1' in consolidated_df.columns:
                    # Multi-class: use macro F1
                    consolidated_df['combined_score'] = consolidated_df['macro_f1'].fillna(0)
                else:
                    # Fallback: try to find any F1 metric
                    for col in ['f1_macro', 'val_f1_macro', 'best_f1_macro']:
                        if col in consolidated_df.columns:
                            consolidated_df['combined_score'] = consolidated_df[col].fillna(0)
                            break
                    else:
                        # Last resort: use accuracy
                        for col in ['accuracy', 'val_accuracy', 'best_accuracy']:
                            if col in consolidated_df.columns:
                                consolidated_df['combined_score'] = consolidated_df[col].fillna(0)
                                break
                        else:
                            consolidated_df['combined_score'] = 0

            # Filter to keep ONLY the best model per (category, language, model) combination
            # Identify the grouping columns that exist
            grouping_cols = [col for col in ['category', 'language', 'model'] if col in consolidated_df.columns]

            if grouping_cols:
                # CRITICAL: Use combined_score for ranking (now guaranteed to exist)
                metric_col = 'combined_score'

                # Sort by grouping columns and metric (descending)
                consolidated_df = consolidated_df.sort_values(
                    by=grouping_cols + [metric_col],
                    ascending=[True] * len(grouping_cols) + [False]
                )

                # Keep only the first (best) entry for each group
                consolidated_df = consolidated_df.drop_duplicates(subset=grouping_cols, keep='first')

                logging.info(f"Filtered to best models using metric: {metric_col}")
                logging.info(f"  • Grouping by: {', '.join(grouping_cols)}")
            else:
                logging.warning("No grouping columns found for filtering best models")

            # Save consolidated best models (to output_dir determined above)
            best_output = output_dir / f"{session_id}_best_models.csv"

            # CRITICAL: Write legends as comment lines first
            with open(best_output, 'w', encoding='utf-8', newline='') as f:
                # Write all collected legends
                for legend in sorted(best_legends):
                    f.write(f"{legend}\n")

            # Append the CSV data
            consolidated_df.to_csv(best_output, mode='a', index=False)
            consolidated_files['best'] = best_output
            logging.info(f"Created consolidated best models: {best_output}")
            logging.info(f"  • Total models: {len(consolidated_df)}")

    return consolidated_files

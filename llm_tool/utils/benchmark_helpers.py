#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
benchmark_helpers.py

MAIN OBJECTIVE:
---------------
Robust utilities for handling benchmark data preprocessing,
validation, and error recovery. Provides helper functions for
multi-label classification benchmarking, data validation, and
result aggregation.

Dependencies:
-------------
- logging
- numbers (Number)
- typing
- collections (Counter, defaultdict)
- pandas
- json
"""
import logging
from numbers import Number
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set
from collections import Counter, defaultdict
import pandas as pd
import json

logger = logging.getLogger(__name__)


def extract_numeric_metric(
    result: Any,
    keys: Iterable[str],
    nested_fields: Iterable[str] = ('metrics', 'final_metrics', 'best_metrics', 'training_summary'),
    default: Optional[float] = 0.0
) -> Optional[float]:
    """
    Safely extract a numeric metric from heterogeneous result structures.

    Args:
        result: Training result (dict, dataclass, etc.)
        keys: Ordered keys to probe (e.g., ['f1_macro', 'best_f1_macro'])
        nested_fields: Nested dictionaries to search when top-level fails
        default: Fallback value when nothing usable is found

    Returns:
        First numeric value discovered, averaged when necessary, otherwise `default`
    """
    key_candidates = list(keys)

    def _normalize(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, Number):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        if isinstance(value, (list, tuple, set)):
            numeric = []
            for item in value:
                norm = _normalize(item)
                if norm is not None:
                    numeric.append(norm)
            if numeric:
                return sum(numeric) / len(numeric)
            return None
        if isinstance(value, dict):
            for key in key_candidates:
                if key in value:
                    norm = _normalize(value[key])
                    if norm is not None:
                        return norm
            for alias in ('f1_macro', 'macro_f1', 'best_f1_macro', 'accuracy', 'precision', 'recall', 'f1'):
                if alias in value:
                    norm = _normalize(value[alias])
                    if norm is not None:
                        return norm
            return None
        return None

    def _lookup(container: Any) -> Optional[float]:
        if isinstance(container, dict):
            for key in key_candidates:
                if key in container:
                    norm = _normalize(container[key])
                    if norm is not None:
                        return norm
        else:
            for key in key_candidates:
                if hasattr(container, key):
                    norm = _normalize(getattr(container, key))
                    if norm is not None:
                        return norm
        return None

    value = _lookup(result)
    if value is not None:
        return value

    if not isinstance(result, dict) and hasattr(result, '__dict__'):
        value = _lookup(result.__dict__)
        if value is not None:
            return value

    for field in nested_fields:
        nested = result.get(field) if isinstance(result, dict) else getattr(result, field, None)
        if nested is None:
            continue
        if isinstance(nested, dict):
            value = _lookup(nested)
            if value is not None:
                return value
        elif isinstance(nested, (list, tuple)):
            for item in nested:
                value = _lookup(item)
                if value is not None:
                    return value

    return default


def validate_label_sufficiency(
    data: pd.DataFrame,
    label_column: str,
    min_samples_per_class: int = 2,
    strategy: str = 'multi-label'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and filter data to ensure sufficient samples per class.

    Args:
        data: DataFrame with training data
        label_column: Column containing labels
        min_samples_per_class: Minimum samples required per class
        strategy: 'multi-label' or 'single-label'

    Returns:
        Tuple of (filtered_dataframe, validation_report)
    """
    validation_report = {
        'original_samples': len(data),
        'removed_classes': [],
        'removed_samples': 0,
        'final_samples': 0,
        'class_distribution': {}
    }

    if strategy == 'multi-label':
        # Parse multi-label annotations
        all_labels = {}
        for idx, row in data.iterrows():
            annotation = row[label_column]

            # Handle different annotation formats
            if isinstance(annotation, list):
                # List of label strings format (e.g., ['category_value1', 'category_value2'])
                for label in annotation:
                    if label and '_' in label:
                        # Split category_value format
                        parts = label.split('_', 1)
                        if len(parts) == 2:
                            key, value = parts
                            if key not in all_labels:
                                all_labels[key] = {}
                            all_labels[key][value] = all_labels[key].get(value, 0) + 1
            elif isinstance(annotation, str):
                # Parse JSON if string
                try:
                    annotation = json.loads(annotation)
                except:
                    continue

                if isinstance(annotation, dict):
                    for key, value in annotation.items():
                        if key not in all_labels:
                            all_labels[key] = {}

                        # Handle list values (multi-class within label)
                        if isinstance(value, list):
                            for v in value:
                                if v and v != 'null':
                                    all_labels[key][v] = all_labels[key].get(v, 0) + 1
                        elif value and value != 'null':
                            all_labels[key][value] = all_labels[key].get(value, 0) + 1
            elif isinstance(annotation, dict):
                for key, value in annotation.items():
                    if key not in all_labels:
                        all_labels[key] = {}

                    # Handle list values (multi-class within label)
                    if isinstance(value, list):
                        for v in value:
                            if v and v != 'null':
                                all_labels[key][v] = all_labels[key].get(v, 0) + 1
                    elif value and value != 'null':
                        all_labels[key][value] = all_labels[key].get(value, 0) + 1

        # Identify insufficient classes
        insufficient_classes = []
        for label_key, class_counts in all_labels.items():
            for class_value, count in class_counts.items():
                if count < min_samples_per_class:
                    insufficient_classes.append(f"{label_key}_{class_value}")
                    validation_report['removed_classes'].append({
                        'label': label_key,
                        'class': class_value,
                        'count': count
                    })

        # Filter data: Remove insufficient classes from annotations
        if insufficient_classes:
            filtered_data = []
            for idx, row in data.iterrows():
                annotation = row[label_column]

                # Handle different annotation formats
                if isinstance(annotation, list):
                    # List format: filter out insufficient labels
                    filtered_labels = [label for label in annotation
                                     if label not in insufficient_classes]
                    if filtered_labels:
                        row = row.copy()
                        row[label_column] = filtered_labels
                        filtered_data.append(row)
                    else:
                        validation_report['removed_samples'] += 1

                elif isinstance(annotation, str):
                    try:
                        annotation = json.loads(annotation)
                    except:
                        filtered_data.append(row)
                        continue

                    if isinstance(annotation, dict):
                        # Remove insufficient classes from annotation
                        filtered_annotation = {}
                        for key, value in annotation.items():
                            # Check if this key-value combo should be kept
                            keep = True
                            if isinstance(value, list):
                                filtered_values = []
                                for v in value:
                                    if v and f"{key}_{v}" not in insufficient_classes:
                                        filtered_values.append(v)
                                if filtered_values:
                                    filtered_annotation[key] = filtered_values
                            elif value and f"{key}_{value}" not in insufficient_classes:
                                filtered_annotation[key] = value

                        # Only keep row if it still has annotations
                        if filtered_annotation:
                            row = row.copy()
                            row[label_column] = json.dumps(filtered_annotation)
                            filtered_data.append(row)
                        else:
                            validation_report['removed_samples'] += 1

                elif isinstance(annotation, dict):
                    # Dict format: filter insufficient classes
                    filtered_annotation = {}
                    for key, value in annotation.items():
                        if isinstance(value, list):
                            filtered_values = []
                            for v in value:
                                if v and f"{key}_{v}" not in insufficient_classes:
                                    filtered_values.append(v)
                            if filtered_values:
                                filtered_annotation[key] = filtered_values
                        elif value and f"{key}_{value}" not in insufficient_classes:
                            filtered_annotation[key] = value

                    if filtered_annotation:
                        row = row.copy()
                        row[label_column] = filtered_annotation
                        filtered_data.append(row)
                    else:
                        validation_report['removed_samples'] += 1

            data = pd.DataFrame(filtered_data)

    validation_report['final_samples'] = len(data)

    # Calculate final class distribution
    if strategy == 'multi-label':
        for idx, row in data.iterrows():
            annotation = row[label_column]

            # Handle different annotation formats
            if isinstance(annotation, list):
                # List format: count label strings
                for label in annotation:
                    if label and '_' in label:
                        parts = label.split('_', 1)
                        if len(parts) == 2:
                            key, value = parts
                            if key not in validation_report['class_distribution']:
                                validation_report['class_distribution'][key] = {}
                            validation_report['class_distribution'][key][value] = \
                                validation_report['class_distribution'][key].get(value, 0) + 1

            elif isinstance(annotation, str):
                try:
                    annotation = json.loads(annotation)
                except:
                    continue

                if isinstance(annotation, dict):
                    for key, value in annotation.items():
                        if key not in validation_report['class_distribution']:
                            validation_report['class_distribution'][key] = {}

                        if isinstance(value, list):
                            for v in value:
                                if v:
                                    validation_report['class_distribution'][key][v] = \
                                        validation_report['class_distribution'][key].get(v, 0) + 1
                        elif value:
                            validation_report['class_distribution'][key][value] = \
                                validation_report['class_distribution'][key].get(value, 0) + 1

            elif isinstance(annotation, dict):
                for key, value in annotation.items():
                    if key not in validation_report['class_distribution']:
                        validation_report['class_distribution'][key] = {}

                    if isinstance(value, list):
                        for v in value:
                            if v:
                                validation_report['class_distribution'][key][v] = \
                                    validation_report['class_distribution'][key].get(v, 0) + 1
                    elif value:
                        validation_report['class_distribution'][key][value] = \
                            validation_report['class_distribution'][key].get(value, 0) + 1

    return data, validation_report


def split_benchmark_by_category(
    data: pd.DataFrame,
    label_column: str,
    selected_categories: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split benchmark data into separate datasets per category.
    This allows independent training and partial success handling.

    Args:
        data: Full benchmark dataset
        label_column: Column with labels
        selected_categories: Categories to include (None = all)

    Returns:
        Dict mapping category name to its filtered dataset
    """
    category_datasets = {}

    for idx, row in data.iterrows():
        annotation = row[label_column]

        # Handle different annotation formats
        if isinstance(annotation, list):
            # List format: extract category from label strings
            for label in annotation:
                if label and '_' in label:
                    # Try to match with selected categories (handle multi-underscore categories)
                    category_found = None
                    value_found = None

                    # If selected_categories provided, try to match them first
                    if selected_categories:
                        for cat in selected_categories:
                            if label.startswith(cat + '_'):
                                category_found = cat
                                value_found = label[len(cat) + 1:]  # Everything after category_
                                break
                    else:
                        # Fallback to simple split if no selected_categories
                        parts = label.split('_', 1)
                        if len(parts) == 2:
                            category_found = parts[0]
                            value_found = parts[1]

                    if not category_found:
                        continue

                    if selected_categories and category_found not in selected_categories:
                        continue

                    if category_found not in category_datasets:
                        category_datasets[category_found] = []

                    # Create single-category annotation (keep as list for consistency)
                    single_cat_row = row.copy()
                    single_cat_row[label_column] = [label]
                    category_datasets[category_found].append(single_cat_row)

        elif isinstance(annotation, str):
            # Parse JSON if string
            try:
                annotation = json.loads(annotation)
            except:
                continue

            if isinstance(annotation, dict):
                # Create separate dataset for each category
                for category, value in annotation.items():
                    if selected_categories and category not in selected_categories:
                        continue

                    if value and value != 'null':
                        if category not in category_datasets:
                            category_datasets[category] = []

                        # Create single-category annotation
                        single_cat_row = row.copy()
                        single_cat_annotation = {category: value}
                        single_cat_row[label_column] = json.dumps(single_cat_annotation)
                        category_datasets[category].append(single_cat_row)

        elif isinstance(annotation, dict):
            # Create separate dataset for each category
            for category, value in annotation.items():
                if selected_categories and category not in selected_categories:
                    continue

                if value and value != 'null':
                    if category not in category_datasets:
                        category_datasets[category] = []

                    # Create single-category annotation
                    single_cat_row = row.copy()
                    single_cat_annotation = {category: value}
                    single_cat_row[label_column] = single_cat_annotation
                    category_datasets[category].append(single_cat_row)

    # Convert lists to DataFrames
    for category in category_datasets:
        category_datasets[category] = pd.DataFrame(category_datasets[category])

    return category_datasets


def filter_languages_for_category(
    data: pd.DataFrame,
    category: str,
    min_samples_per_class: int = 3,
    language_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter a category dataset to keep only languages with sufficient samples per class.

    Args:
        data: Category-specific DataFrame produced by split_benchmark_by_category
        category: Category name (e.g. 'prompt_tech_technosolutionism')
        min_samples_per_class: Minimum samples required per class per language
        language_column: Optional explicit language column name

    Returns:
        Tuple of (filtered_dataframe, filtering_report)
    """
    report: Dict[str, Any] = {
        'category': category,
        'filtered': False,
        'filtered_all': False,
        'total_samples_before': int(len(data)),
        'total_samples_after': int(len(data)),
        'languages_kept': [],
        'languages_dropped': [],
        'drop_reasons': {},
        'analysis_per_language': {},
        'min_samples_per_class': min_samples_per_class,
        'language_column': language_column
    }

    if data.empty:
        report['reason'] = 'empty_dataset'
        return data, report

    lang_col = language_column or ('lang' if 'lang' in data.columns else None)
    if lang_col is None and 'language' in data.columns:
        lang_col = 'language'

    if lang_col is None or data[lang_col].dropna().empty:
        report['reason'] = 'no_language_column'
        return data, report

    per_language_counts: Dict[str, Counter] = defaultdict(Counter)
    class_set: Set[str] = set()

    for _, row in data.iterrows():
        lang_val = row.get(lang_col)
        if not isinstance(lang_val, str) or not lang_val.strip():
            continue
        language = lang_val.strip().upper()

        raw_labels = row.get('labels')
        label_strings: List[str] = []
        if isinstance(raw_labels, list):
            label_strings = [lbl for lbl in raw_labels if isinstance(lbl, str)]
        elif isinstance(raw_labels, str):
            label_strings = [raw_labels]
        elif isinstance(raw_labels, dict):
            for key, value in raw_labels.items():
                if isinstance(value, list):
                    label_strings.extend(f"{key}_{v}" for v in value if isinstance(v, str))
                elif isinstance(value, str):
                    label_strings.append(f"{key}_{value}")

        matched_labels = [
            lbl for lbl in label_strings
            if isinstance(lbl, str) and lbl.startswith(f"{category}_")
        ]

        if not matched_labels:
            continue

        for label in matched_labels:
            class_value = label[len(category) + 1:]
            class_set.add(class_value)
            per_language_counts[language][class_value] += 1

    if not class_set or not per_language_counts:
        report['reason'] = 'no_matching_labels'
        return data, report

    languages_kept: Set[str] = set()
    languages_dropped: Set[str] = set()
    drop_reasons: Dict[str, Dict[str, int]] = {}

    for language, counts in per_language_counts.items():
        insufficient = {
            class_name: counts.get(class_name, 0)
            for class_name in class_set
            if counts.get(class_name, 0) < min_samples_per_class
        }
        if insufficient:
            languages_dropped.add(language)
            drop_reasons[language] = insufficient
        else:
            languages_kept.add(language)

    if not languages_kept and languages_dropped:
        def language_score(item: Tuple[str, Counter]) -> Tuple[int, int]:
            lang, counts = item
            min_count = min(counts.get(cls, 0) for cls in class_set)
            total = sum(counts.get(cls, 0) for cls in class_set)
            return (min_count, total)

        best_lang, _ = max(per_language_counts.items(), key=language_score)
        languages_kept.add(best_lang)
        if best_lang in languages_dropped:
            languages_dropped.remove(best_lang)
            drop_reasons.pop(best_lang, None)

    if languages_kept:
        filtered = data[
            data[lang_col].apply(
                lambda val: isinstance(val, str) and val.strip().upper() in languages_kept
            )
        ].copy()
    else:
        filtered = data.copy()

    report.update({
        'filtered': bool(languages_dropped),
        'filtered_all': filtered.empty and bool(per_language_counts),
        'total_samples_after': int(len(filtered)),
        'languages_kept': sorted(languages_kept) if languages_kept else [],
        'languages_dropped': sorted(languages_dropped),
        'drop_reasons': drop_reasons,
        'analysis_per_language': {
            lang: {cls: per_language_counts[lang].get(cls, 0) for cls in class_set}
            for lang in per_language_counts
        }
    })

    if report['filtered_all']:
        report['reason'] = 'no_language_with_sufficient_samples'

    return filtered, report


def aggregate_benchmark_results(
    results_by_category: Dict[str, Dict[str, Any]],
    model_id: str
) -> Dict[str, Any]:
    """
    Intelligently aggregate results from multiple category trainings.

    Args:
        results_by_category: Dict mapping category to its training results
        model_id: Model identifier

    Returns:
        Aggregated results with proper error handling
    """
    successful_results: List[Dict[str, Any]] = []
    failed_categories: List[str] = []
    total_training_time = 0.0

    for category, result in results_by_category.items():
        if not result or result.get('error'):
            failed_categories.append(category)
            continue

        f1_macro = extract_numeric_metric(result, ('f1_macro', 'best_f1_macro', 'macro_f1', 'f1'), default=0.0)
        accuracy = extract_numeric_metric(result, ('accuracy', 'best_accuracy'), default=0.0)
        precision = extract_numeric_metric(result, ('precision', 'macro_precision', 'best_precision'), default=0.0)
        recall = extract_numeric_metric(result, ('recall', 'macro_recall', 'best_recall'), default=0.0)

        if f1_macro > 0 or accuracy > 0:
            training_time = float(result.get('training_time', 0) or 0.0)
            f1_class_0 = extract_numeric_metric(result, ('f1_0', 'f1_class_0'), default=None)
            f1_class_1 = extract_numeric_metric(result, ('f1_1', 'f1_class_1'), default=None)

            successful_results.append({
                'category': category,
                'f1_macro': f1_macro,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'training_time': training_time,
                'f1_0': f1_class_0,
                'f1_1': f1_class_1
            })
            total_training_time += training_time
        else:
            failed_categories.append(category)

    # Calculate aggregated metrics
    if successful_results:
        avg_f1 = sum(r['f1_macro'] for r in successful_results) / len(successful_results)
        avg_acc = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)

        aggregated = {
            'f1_macro': avg_f1,
            'accuracy': avg_acc,
            'precision': avg_precision,
            'recall': avg_recall,
            'training_time': total_training_time,
            'successful_categories': len(successful_results),
            'failed_categories': len(failed_categories),
            'category_details': successful_results,
            'partial_success': len(failed_categories) > 0,
            # Backward compatibility keys expected elsewhere in the codebase
            'best_f1_macro': avg_f1,
            'best_accuracy': avg_acc
        }

        # Add per-class metrics if available (for binary classification)
        f1_0_values = [r['f1_0'] for r in successful_results if r.get('f1_0') is not None]
        f1_1_values = [r['f1_1'] for r in successful_results if r.get('f1_1') is not None]

        if f1_0_values:
            aggregated['f1_0'] = sum(f1_0_values) / len(f1_0_values)
        if f1_1_values:
            aggregated['f1_1'] = sum(f1_1_values) / len(f1_1_values)

        logger.info(f"Benchmark aggregation for {model_id}: "
                   f"{len(successful_results)}/{len(results_by_category)} categories successful")
    else:
        # All categories failed
        aggregated = {
            'f1_macro': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'training_time': 0,
            'successful_categories': 0,
            'failed_categories': len(failed_categories),
            'error': 'All categories failed training'
        }

        logger.warning(f"Benchmark for {model_id}: All {len(failed_categories)} categories failed")

    return aggregated


def preprocess_benchmark_data(
    data_path: str,
    selected_categories: List[str],
    min_samples_per_class: int = 3,
    test_ratio: float = 0.2
) -> Tuple[str, Dict[str, Any]]:
    """
    Comprehensive benchmark data preprocessing with validation.

    Args:
        data_path: Path to input data
        selected_categories: Categories to benchmark
        min_samples_per_class: Minimum samples per class (higher than 2 for stability)
        test_ratio: Ratio for train/test split

    Returns:
        Tuple of (path_to_processed_data, preprocessing_report)
    """
    import tempfile
    from pathlib import Path

    # Load data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
        label_column = 'labels'  # Adjust as needed
    else:
        # JSONL format
        data = pd.read_json(data_path, lines=True)
        label_column = 'labels'

    preprocessing_report = {
        'original_samples': len(data),
        'selected_categories': selected_categories,
        'validation_results': {},
        'category_sample_counts': {}
    }

    # Step 1: Filter to selected categories
    filtered_rows = []
    for idx, row in data.iterrows():
        annotation = row[label_column]

        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)
            except:
                continue

        if isinstance(annotation, dict):
            filtered_annotation = {k: v for k, v in annotation.items()
                                 if k in selected_categories}
            if filtered_annotation:
                row = row.copy()
                row[label_column] = json.dumps(filtered_annotation)
                filtered_rows.append(row)

    data = pd.DataFrame(filtered_rows)
    preprocessing_report['after_category_filter'] = len(data)

    # Step 2: Validate and filter insufficient classes
    data, validation_report = validate_label_sufficiency(
        data, label_column, min_samples_per_class, 'multi-label'
    )
    preprocessing_report['validation_results'] = validation_report

    # Step 3: Split by category for independent processing
    category_datasets = split_benchmark_by_category(data, label_column, selected_categories)

    for category, cat_data in category_datasets.items():
        preprocessing_report['category_sample_counts'][category] = len(cat_data)

    # Step 4: Save processed data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        data.to_json(f, orient='records', lines=True, force_ascii=False)
        processed_path = f.name

    preprocessing_report['processed_path'] = processed_path
    preprocessing_report['final_samples'] = len(data)

    return processed_path, preprocessing_report

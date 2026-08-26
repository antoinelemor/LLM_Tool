#!/usr/bin/env python3
"""
Language Filtering Utilities
=============================
Functions to handle insufficient samples per language in training data.
"""
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


def is_valid_language(lang: Any) -> bool:
    """
    Check if a language value is valid (not nan, None, empty, etc.).

    Args:
        lang: Language value to check

    Returns:
        True if the language is a valid, non-empty string
    """
    if lang is None:
        return False
    if isinstance(lang, float) and pd.isna(lang):
        return False
    if not isinstance(lang, str):
        # Try to convert to string and check
        try:
            lang = str(lang)
        except:
            return False
    # Check for common invalid string representations
    lang_lower = lang.strip().lower()
    invalid_values = {'nan', 'none', 'null', '', 'na', 'n/a', 'undefined'}
    return lang_lower not in invalid_values


def analyze_language_distribution(
    texts: List[str],
    labels: List[str],
    languages: Optional[List[str]] = None,
    min_samples_per_class: int = 2
) -> Dict[str, Dict[str, int]]:
    """
    Analyze the distribution of classes per language.

    Args:
        texts: List of text samples
        labels: List of labels (can be multi-class or binary)
        languages: Optional list of language codes
        min_samples_per_class: Minimum samples required per class

    Returns:
        Dict with analysis per language and recommendations
    """
    if not languages:
        # If no languages provided, treat as single language
        languages = ['UNKNOWN'] * len(texts)

    # Group by language and label
    lang_label_counts = defaultdict(lambda: defaultdict(int))

    for text, label, lang in zip(texts, labels, languages):
        lang_label_counts[lang][label] += 1

    analysis = {}
    for lang, label_counts in lang_label_counts.items():
        total_samples = sum(label_counts.values())
        num_classes = len(label_counts)

        # Check for insufficient samples
        insufficient_classes = [
            (label, count) for label, count in label_counts.items()
            if count < min_samples_per_class
        ]

        analysis[lang] = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'label_distribution': dict(label_counts),
            'insufficient_classes': insufficient_classes,
            'can_train': len(insufficient_classes) == 0,
            'min_samples': min(label_counts.values()) if label_counts else 0,
            'max_samples': max(label_counts.values()) if label_counts else 0
        }

    return analysis


def filter_languages_with_sufficient_samples(
    texts: List[str],
    labels: List[str],
    languages: Optional[List[str]] = None,
    min_samples_per_class: int = 2,
    min_train_samples: int = 1,
    min_val_samples: int = 1
) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    """
    Filter out classes that don't have enough samples for training.

    IMPORTANT: This function now filters by CLASS, not by LANGUAGE.
    If a class has insufficient samples in ANY language, that class is dropped
    from ALL languages (for consistency), but the languages themselves are kept.

    Args:
        texts: List of text samples
        labels: List of labels
        languages: Optional list of language codes
        min_samples_per_class: Minimum total samples per class
        min_train_samples: Minimum samples per class in training set
        min_val_samples: Minimum samples per class in validation set

    Returns:
        Tuple of (filtered_texts, filtered_labels, filtered_languages, filtering_report)
    """
    if not languages:
        # No language filtering needed
        return texts, labels, languages, {'filtered': False, 'all_languages_kept': True}

    # First, filter out samples with invalid language values (nan, None, '', etc.)
    valid_texts = []
    valid_labels = []
    valid_languages = []
    invalid_language_samples = 0

    for text, label, lang in zip(texts, labels, languages):
        if is_valid_language(lang):
            valid_texts.append(text)
            valid_labels.append(label)
            valid_languages.append(lang)
        else:
            invalid_language_samples += 1

    if invalid_language_samples > 0:
        logger.warning(
            f"Dropped {invalid_language_samples} samples with invalid/missing language values "
            f"(nan, None, empty, etc.)"
        )

    # If no valid samples remain after filtering invalid languages
    if not valid_texts:
        logger.error("No samples with valid language values remaining after filtering")
        return [], [], [], {
            'filtered': True,
            'total_samples_before': len(texts),
            'total_samples_after': 0,
            'invalid_language_samples_dropped': invalid_language_samples,
            'languages_kept': [],
            'languages_dropped': [],
            'drop_reasons': {},
            'error': 'All samples had invalid language values'
        }

    # Analyze distribution on valid samples only
    min_required = min_train_samples + min_val_samples
    analysis = analyze_language_distribution(
        valid_texts, valid_labels, valid_languages,
        min_samples_per_class=min_required
    )

    # NEW APPROACH: Find classes with insufficient samples in ANY language
    # and drop those CLASSES (not languages)
    classes_to_drop = set()
    drop_reasons = {}

    for lang, stats in analysis.items():
        for class_name, count in stats['insufficient_classes']:
            classes_to_drop.add(class_name)
            if class_name not in drop_reasons:
                drop_reasons[class_name] = {
                    'reason': 'insufficient_samples_per_language',
                    'languages_affected': [],
                    'counts_per_language': {}
                }
            drop_reasons[class_name]['languages_affected'].append(lang)
            drop_reasons[class_name]['counts_per_language'][lang] = count

    # Log which classes are being dropped
    if classes_to_drop:
        logger.warning(
            f"[!]  Filtering {len(classes_to_drop)} class(es) with insufficient samples per language (< {min_required}):"
        )
        for cls in sorted(classes_to_drop):
            counts = drop_reasons[cls]['counts_per_language']
            counts_str = ', '.join([f"{lang}: {cnt}" for lang, cnt in sorted(counts.items())])
            logger.warning(f"   • Dropping class '{cls}': {counts_str}")

    # Filter samples - keep all languages, drop only problematic classes
    filtered_texts = []
    filtered_labels = []
    filtered_languages = []
    samples_dropped_per_class = Counter()

    for text, label, lang in zip(valid_texts, valid_labels, valid_languages):
        if label not in classes_to_drop:
            filtered_texts.append(text)
            filtered_labels.append(label)
            filtered_languages.append(lang)
        else:
            samples_dropped_per_class[label] += 1

    # Determine which languages are actually kept (those with at least one sample remaining)
    languages_kept = sorted(set(filtered_languages))
    languages_in_original = sorted(set(valid_languages))
    languages_dropped = [lang for lang in languages_in_original if lang not in languages_kept]

    if samples_dropped_per_class:
        total_dropped = sum(samples_dropped_per_class.values())
        logger.warning(f"   → Dropped {total_dropped} samples total")
        logger.info(f"   → Keeping {len(filtered_texts)} samples in {len(languages_kept)} languages: {languages_kept}")

    # Create report
    any_filtering = len(classes_to_drop) > 0 or invalid_language_samples > 0
    report = {
        'filtered': any_filtering,
        'total_samples_before': len(texts),
        'total_samples_after': len(filtered_texts),
        'invalid_language_samples_dropped': invalid_language_samples,
        'languages_kept': languages_kept,
        'languages_dropped': languages_dropped,
        'classes_dropped': sorted(list(classes_to_drop)),
        'samples_dropped_per_class': dict(samples_dropped_per_class),
        'drop_reasons': drop_reasons,
        'analysis_per_language': analysis
    }

    # Log the filtering
    if report['filtered']:
        if classes_to_drop:
            logger.info(
                f"Class filtering: Dropped {len(classes_to_drop)} class(es), "
                f"keeping {len(languages_kept)} languages: {languages_kept}"
            )
        if invalid_language_samples > 0:
            logger.info(f"  Also dropped {invalid_language_samples} samples with invalid language values")
        if languages_dropped:
            logger.warning(f"  Languages dropped (no samples remaining): {languages_dropped}")

    return filtered_texts, filtered_labels, filtered_languages, report


def create_training_report(
    category_name: str,
    filtering_report: Dict[str, Any],
    training_successful: bool,
    metrics: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive training report including language filtering info.

    Args:
        category_name: Name of the category being trained
        filtering_report: Report from language filtering
        training_successful: Whether training succeeded
        metrics: Training metrics if successful
        error_message: Error message if failed

    Returns:
        Complete training report
    """
    report = {
        'category': category_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'training_successful': training_successful,
        'language_filtering': filtering_report,
    }

    if training_successful and metrics:
        report['metrics'] = metrics
        report['status'] = 'success'
    else:
        report['status'] = 'failed'
        report['error'] = error_message or 'Unknown error'

    # Add summary
    if filtering_report['filtered']:
        report['summary'] = (
            f"Trained on {len(filtering_report['languages_kept'])} languages "
            f"({filtering_report['total_samples_after']} samples). "
            f"Dropped {len(filtering_report['languages_dropped'])} languages due to insufficient samples."
        )
    else:
        report['summary'] = f"Trained on all languages ({filtering_report['total_samples_after']} samples)"

    return report


def merge_language_specific_metrics(
    metrics_per_language: Dict[str, Dict[str, Any]],
    languages_used: List[str],
    languages_skipped: List[str]
) -> Dict[str, Any]:
    """
    Merge metrics from different languages into a comprehensive report.

    Args:
        metrics_per_language: Metrics for each language that was trained
        languages_used: Languages that were successfully trained
        languages_skipped: Languages that were skipped

    Returns:
        Merged metrics with language-specific breakdowns
    """
    merged = {
        'languages_trained': languages_used,
        'languages_skipped': languages_skipped,
        'per_language_metrics': metrics_per_language,
    }

    # Calculate weighted averages
    if metrics_per_language:
        total_samples = sum(
            m.get('support', 0) for m in metrics_per_language.values()
        )

        if total_samples > 0:
            weighted_f1 = sum(
                m.get('f1_macro', 0) * m.get('support', 0)
                for m in metrics_per_language.values()
            ) / total_samples

            weighted_acc = sum(
                m.get('accuracy', 0) * m.get('support', 0)
                for m in metrics_per_language.values()
            ) / total_samples

            merged['overall_f1_weighted'] = weighted_f1
            merged['overall_accuracy_weighted'] = weighted_acc
            merged['total_samples'] = total_samples

    return merged


def save_filtered_languages_log(
    session_id: str,
    category: str,
    languages_dropped: List[str],
    drop_reasons: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Save a log of filtered languages for audit trail.

    Args:
        session_id: Training session ID
        category: Category being trained
        languages_dropped: List of dropped languages
        drop_reasons: Reasons for dropping each language
        output_dir: Directory to save log
    """
    import json
    from pathlib import Path

    log_dir = Path(output_dir) / 'language_filtering_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{session_id}_{category}_filtered_languages.json'

    log_data = {
        'session_id': session_id,
        'category': category,
        'timestamp': pd.Timestamp.now().isoformat(),
        'languages_dropped': languages_dropped,
        'drop_reasons': drop_reasons
    }

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Saved language filtering log to {log_file}")
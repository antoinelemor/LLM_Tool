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
    Filter out languages that don't have enough samples for training.

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

    # Analyze distribution
    analysis = analyze_language_distribution(
        texts, labels, languages,
        min_samples_per_class=min_train_samples + min_val_samples
    )

    # Identify languages to keep
    languages_to_keep = set()
    languages_to_drop = set()
    drop_reasons = {}

    for lang, stats in analysis.items():
        if stats['can_train']:
            languages_to_keep.add(lang)
        else:
            languages_to_drop.add(lang)
            drop_reasons[lang] = {
                'reason': 'insufficient_samples',
                'details': stats['insufficient_classes'],
                'total_samples': stats['total_samples']
            }

    # If all languages would be dropped, keep the best ones
    if not languages_to_keep and languages_to_drop:
        # Sort by minimum samples per class (descending)
        sorted_langs = sorted(
            analysis.items(),
            key=lambda x: (x[1]['min_samples'], x[1]['total_samples']),
            reverse=True
        )
        # Keep at least the best language
        best_lang = sorted_langs[0][0]
        languages_to_keep.add(best_lang)
        languages_to_drop.discard(best_lang)
        logger.warning(f"All languages had insufficient samples. Keeping best language: {best_lang}")

    # Filter the data
    filtered_texts = []
    filtered_labels = []
    filtered_languages = []

    for text, label, lang in zip(texts, labels, languages):
        if lang in languages_to_keep:
            filtered_texts.append(text)
            filtered_labels.append(label)
            filtered_languages.append(lang)

    # Create report
    report = {
        'filtered': len(languages_to_drop) > 0,
        'total_samples_before': len(texts),
        'total_samples_after': len(filtered_texts),
        'languages_kept': sorted(list(languages_to_keep)),
        'languages_dropped': sorted(list(languages_to_drop)),
        'drop_reasons': drop_reasons,
        'analysis_per_language': analysis
    }

    # Log the filtering
    if report['filtered']:
        logger.info(f"Language filtering: Kept {len(languages_to_keep)} languages, dropped {len(languages_to_drop)}")
        for lang in languages_to_drop:
            reason = drop_reasons[lang]
            logger.warning(
                f"  Dropped {lang}: {reason['total_samples']} samples, "
                f"insufficient classes: {reason['details']}"
            )

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

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Saved language filtering log to {log_file}")
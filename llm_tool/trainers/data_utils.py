#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
data_utils.py

MAIN OBJECTIVE:
---------------
Offer rich data structures and helpers for preparing training datasets,
preserving metadata, and tracking performance across labels and languages.

Dependencies:
-------------
- json
- csv
- typing
- dataclasses
- collections
- numpy
- torch
- logging

MAIN FEATURES:
--------------
1) Provide DataSample and MetadataDataset abstractions that retain metadata
2) Implement safe conversion utilities for numpy/pandas interoperability
3) Deliver loaders that build TensorDataset objects with aligned metadata
4) Track per-label and per-language metrics via PerformanceTracker helpers
5) Export utilities for serialising metrics and confusion matrices to JSON/CSV

Author:
-------
Antoine Lemor
"""

import json
import csv
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import logging


# ==================== TYPE CONVERSION UTILITIES ====================
# These utilities ensure numpy types are converted to Python native types
# to prevent issues with downstream processing (e.g., "object of type 'numpy.int64' has no len()")

def safe_convert_label(label: Any) -> Union[int, str]:
    """
    Safely convert a single label from numpy types to Python native types.

    Args:
        label: Label to convert (can be numpy.int64, int, str, etc.)

    Returns:
        Python native int or str

    Examples:
        >>> safe_convert_label(np.int64(5))
        5
        >>> safe_convert_label("positive")
        "positive"
    """
    if isinstance(label, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(label)
    elif isinstance(label, (np.floating, np.float64, np.float32)):
        return float(label)
    elif isinstance(label, np.str_):
        return str(label)
    elif isinstance(label, (int, float, str)):
        return label
    else:
        # Fallback: try to convert to string
        return str(label)


def safe_convert_labels(labels: List[Any]) -> List[Union[int, str]]:
    """
    Safely convert a list of labels from numpy types to Python native types.

    Args:
        labels: List of labels to convert

    Returns:
        List of Python native ints or strs

    Examples:
        >>> safe_convert_labels([np.int64(1), np.int64(0), np.int64(1)])
        [1, 0, 1]
    """
    return [safe_convert_label(label) for label in labels]


def safe_tolist(data: Any, column_name: Optional[str] = None) -> List[Any]:
    """
    Safely convert pandas Series or numpy arrays to Python lists with native types.

    This function handles the common pattern of:
    - df['column'].tolist() → returns numpy types
    - Converting numpy types to Python native types

    Args:
        data: pandas Series, numpy array, or any iterable
        column_name: Optional column name for better error messages

    Returns:
        List with Python native types (int, str, float)

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'labels': [1, 0, 1]})
        >>> safe_tolist(df['labels'])
        [1, 0, 1]  # Python ints, not numpy.int64
    """
    try:
        # Convert to list first
        if hasattr(data, 'tolist'):
            result = data.tolist()
        elif hasattr(data, '__iter__'):
            result = list(data)
        else:
            result = [data]

        # Convert any numpy types to Python native types
        return safe_convert_labels(result)
    except Exception as e:
        col_info = f" for column '{column_name}'" if column_name else ""
        raise ValueError(f"Failed to convert data{col_info} to list: {e}")


# ==================== END TYPE CONVERSION UTILITIES ====================


@dataclass
class DataSample:
    """Represents a single data sample with metadata."""
    text: str
    label: Union[str, int]
    id: Optional[str] = None
    lang: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetadataDataset(Dataset):
    """Dataset that preserves metadata alongside tensors."""

    def __init__(self,
                 input_ids: torch.Tensor,
                 attention_masks: torch.Tensor,
                 labels: torch.Tensor,
                 metadata: Optional[List[Dict]] = None):
        """
        Initialize dataset with tensors and metadata.

        Args:
            input_ids: Tokenized input IDs
            attention_masks: Attention masks
            labels: Label tensor
            metadata: List of metadata dicts for each sample
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.metadata = metadata or [{}] * len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.labels[idx],
            self.metadata[idx]
        )


class DataLoader:
    """Enhanced data loader with metadata support."""

    @staticmethod
    def load_jsonl(filepath: str,
                   text_field: str = 'text',
                   label_field: str = 'label',
                   id_field: Optional[str] = 'id',
                   lang_field: Optional[str] = 'lang') -> List[DataSample]:
        """
        Load data from JSONL file with metadata.

        Args:
            filepath: Path to JSONL file
            text_field: Field name for text content
            label_field: Field name for label
            id_field: Field name for sample ID
            lang_field: Field name for language code

        Returns:
            List of DataSample objects

        Example JSONL format:
        {"text": "This is a sample", "label": 1, "id": "001", "lang": "en", "custom_field": "value"}
        """
        samples = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Extract required fields
                    if text_field not in data:
                        logging.warning(f"Line {line_num}: Missing '{text_field}' field, skipping")
                        continue
                    if label_field not in data:
                        logging.warning(f"Line {line_num}: Missing '{label_field}' field, skipping")
                        continue

                    # Extract metadata
                    metadata = {k: v for k, v in data.items()
                              if k not in [text_field, label_field, id_field, lang_field]}

                    sample = DataSample(
                        text=data[text_field],
                        label=data[label_field],
                        id=data.get(id_field),
                        lang=data.get(lang_field),
                        metadata=metadata if metadata else None
                    )
                    samples.append(sample)

                except json.JSONDecodeError as e:
                    logging.warning(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                except Exception as e:
                    logging.warning(f"Line {line_num}: Error processing - {e}")
                    continue

        return samples

    @staticmethod
    def load_csv(filepath: str,
                 text_column: str = 'text',
                 label_column: str = 'label',
                 id_column: Optional[str] = 'id',
                 lang_column: Optional[str] = 'lang',
                 delimiter: str = ',') -> List[DataSample]:
        """
        Load data from CSV file with metadata.

        Args:
            filepath: Path to CSV file
            text_column: Column name for text
            label_column: Column name for label
            id_column: Column name for ID
            lang_column: Column name for language
            delimiter: CSV delimiter

        Returns:
            List of DataSample objects
        """
        samples = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row_num, row in enumerate(reader, 1):
                try:
                    if text_column not in row:
                        logging.warning(f"Row {row_num}: Missing '{text_column}' column")
                        continue
                    if label_column not in row:
                        logging.warning(f"Row {row_num}: Missing '{label_column}' column")
                        continue

                    # Extract metadata from other columns
                    metadata = {k: v for k, v in row.items()
                              if k not in [text_column, label_column, id_column, lang_column]}

                    sample = DataSample(
                        text=row[text_column],
                        label=row[label_column],
                        id=row.get(id_column),
                        lang=row.get(lang_column),
                        metadata=metadata if metadata else None
                    )
                    samples.append(sample)

                except Exception as e:
                    logging.warning(f"Row {row_num}: Error processing - {e}")
                    continue

        return samples

    @staticmethod
    def prepare_splits(samples: List[DataSample],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      stratify_by_lang: bool = False,
                      stratify_by_label: bool = True,
                      random_seed: int = 42,
                      min_train_per_class: int = 1,
                      min_val_per_class: int = 1) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Split data into train/val/test sets with GUARANTEED minimum samples per class.

        CRITICAL GUARANTEES:
        - At least min_train_per_class samples per class in training set (default: 1)
        - At least min_val_per_class samples per class in validation set (default: 1)
        - If a class has < (min_train + min_val) samples, raises clear error
        - Stratification when possible, with intelligent fallback otherwise

        Args:
            samples: List of DataSample objects
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by_lang: Whether to stratify splits by language
            stratify_by_label: Whether to stratify splits by label (default: True, CRITICAL for class balance)
            random_seed: Random seed for reproducibility
            min_train_per_class: Minimum samples per class in train set (default: 1, DO NOT SET TO 0)
            min_val_per_class: Minimum samples per class in val set (default: 1, DO NOT SET TO 0)

        Returns:
            Tuple of (train_samples, val_samples, test_samples)

        Raises:
            ValueError: If any class has fewer than (min_train_per_class + min_val_per_class) samples
        """
        from sklearn.model_selection import train_test_split

        np.random.seed(random_seed)

        # CRITICAL: Enforce minimum requirements
        if min_train_per_class < 1:
            min_train_per_class = 1
            logging.warning("min_train_per_class must be >= 1. Setting to 1.")
        if min_val_per_class < 1:
            min_val_per_class = 1
            logging.warning("min_val_per_class must be >= 1. Setting to 1.")

        min_total_required = min_train_per_class + min_val_per_class

        # Determine stratification strategy
        if not stratify_by_label and not stratify_by_lang:
            # No stratification - simple random split (NOT RECOMMENDED)
            n = len(samples)
            indices = np.random.permutation(n)

            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_samples = [samples[i] for i in indices[:train_end]]
            val_samples = [samples[i] for i in indices[train_end:val_end]]
            test_samples = [samples[i] for i in indices[val_end:]]

            return train_samples, val_samples, test_samples

        # Build stratification keys and check minimum requirements
        stratify_keys = []
        for sample in samples:
            if stratify_by_label and stratify_by_lang:
                # Stratify by both label AND language
                lang = sample.lang or 'unknown'
                key = f"{sample.label}_{lang}"
            elif stratify_by_label:
                # Stratify by label only (CRITICAL: ensures all classes in train AND val)
                key = str(sample.label)
            else:
                # Stratify by language only
                key = sample.lang or 'unknown'
            stratify_keys.append(key)

        # Count samples per class
        key_counts = Counter(stratify_keys)

        # CRITICAL: Check absolute minimums
        insufficient_classes = {k: v for k, v in key_counts.items() if v < min_total_required}
        if insufficient_classes:
            error_msg = (
                f"❌ CRITICAL ERROR: Cannot split dataset - some classes have insufficient samples:\n"
            )
            for key, count in insufficient_classes.items():
                error_msg += f"   • Class '{key}': {count} sample(s) (minimum required: {min_total_required})\n"
            error_msg += f"\n   Required: at least {min_train_per_class} train + {min_val_per_class} val = {min_total_required} total per class"
            raise ValueError(error_msg)

        # Check for classes at the critical minimum (exactly min_total_required)
        critical_classes = {k: v for k, v in key_counts.items() if v == min_total_required}
        if critical_classes:
            logging.warning(
                f"⚠️  CRITICAL WARNING: Some classes have exactly the minimum required samples "
                f"({min_total_required} = {min_train_per_class} train + {min_val_per_class} val):"
            )
            for key, count in critical_classes.items():
                logging.warning(f"   • Class '{key}': {count} samples (MINIMUM THRESHOLD)")
            logging.warning(
                f"   → Training may be unstable. Consider collecting more data for these classes."
            )

        # Check for low-sample classes (< 10 total)
        low_sample_classes = {k: v for k, v in key_counts.items()
                            if min_total_required < v < 10}
        if low_sample_classes:
            logging.warning(
                f"⚠️  WARNING: Some classes have very few samples (recommended: >= 10):"
            )
            for key, count in low_sample_classes.items():
                logging.warning(f"   • Class '{key}': {count} samples")

        # Perform guaranteed minimum split
        # Strategy: manually ensure minimum samples per class, then use stratification for the rest

        # Group samples by class
        class_samples = defaultdict(list)
        for sample, key in zip(samples, stratify_keys):
            class_samples[key].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        # For each class, allocate minimum required samples first
        for class_key, class_sample_list in class_samples.items():
            n_class = len(class_sample_list)

            # Shuffle class samples
            np.random.shuffle(class_sample_list)

            # Allocate minimums first (1 train + 1 val guaranteed)
            train_samples.extend(class_sample_list[:min_train_per_class])
            val_samples.extend(class_sample_list[min_train_per_class:min_train_per_class + min_val_per_class])

            # Distribute remaining samples according to ratios
            remaining = class_sample_list[min_train_per_class + min_val_per_class:]
            if remaining:
                n_remaining = len(remaining)

                # Calculate how many more go to each set (proportional to original ratios)
                total_ratio = train_ratio + val_ratio + test_ratio
                if total_ratio > 0:
                    train_extra = int(n_remaining * (train_ratio / total_ratio))
                    val_extra = int(n_remaining * (val_ratio / total_ratio))
                    test_extra = n_remaining - train_extra - val_extra  # Remainder goes to test

                    idx = 0
                    train_samples.extend(remaining[idx:idx + train_extra])
                    idx += train_extra
                    val_samples.extend(remaining[idx:idx + val_extra])
                    idx += val_extra
                    test_samples.extend(remaining[idx:])
                else:
                    # Default: put remaining in train
                    train_samples.extend(remaining)

        # Verify minimum guarantees (sanity check)
        if stratify_by_label:
            # Check that each class has minimum in train and val
            train_labels = Counter([str(s.label) for s in train_samples])
            val_labels = Counter([str(s.label) for s in val_samples])

            for class_key in key_counts.keys():
                # Extract just the label part if combined with language
                if stratify_by_label and stratify_by_lang:
                    # CRITICAL FIX: Use rsplit to split on LAST underscore only
                    # Labels can contain underscores (e.g., 'welfare_state', 'early_learning_childcare')
                    # Format is: {label}_{lang}, so split on last '_' to extract label
                    label_part = class_key.rsplit('_', 1)[0]
                else:
                    label_part = class_key

                if train_labels.get(label_part, 0) < min_train_per_class:
                    logging.error(f"❌ Guarantee violation: Class '{label_part}' has {train_labels.get(label_part, 0)} train samples (required: {min_train_per_class})")
                if val_labels.get(label_part, 0) < min_val_per_class:
                    logging.error(f"❌ Guarantee violation: Class '{label_part}' has {val_labels.get(label_part, 0)} val samples (required: {min_val_per_class})")

        return train_samples, val_samples, test_samples


class PerformanceTracker:
    """Track and analyze performance metrics with metadata support."""

    def __init__(self):
        self.predictions = []
        self.labels = []
        self.metadata = []
        self.language_metrics = defaultdict(lambda: {'predictions': [], 'labels': []})

    def add_batch(self,
                  predictions: np.ndarray,
                  labels: np.ndarray,
                  metadata: List[Dict]):
        """Add a batch of predictions with metadata."""
        self.predictions.extend(predictions.tolist())
        self.labels.extend(labels.tolist())
        self.metadata.extend(metadata)

        # Track by language if available
        for pred, label, meta in zip(predictions, labels, metadata):
            if 'lang' in meta and meta['lang']:
                self.language_metrics[meta['lang']]['predictions'].append(pred)
                self.language_metrics[meta['lang']]['labels'].append(label)

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics including per-language performance."""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(self.labels, self.predictions),
            'precision_recall_f1': precision_recall_fscore_support(
                self.labels, self.predictions, average=None
            ),
            'macro_f1': precision_recall_fscore_support(
                self.labels, self.predictions, average='macro'
            )[2],
            'confusion_matrix': confusion_matrix(self.labels, self.predictions).tolist()
        }

        # Per-language metrics
        language_metrics = {}
        for lang, data in self.language_metrics.items():
            if len(data['predictions']) > 0:
                prec, rec, f1, supp = precision_recall_fscore_support(
                    data['labels'], data['predictions'], average=None, zero_division=0
                )
                language_metrics[lang] = {
                    'n_samples': len(data['predictions']),
                    'accuracy': accuracy_score(data['labels'], data['predictions']),
                    'precision': prec.tolist(),
                    'recall': rec.tolist(),
                    'f1': f1.tolist(),
                    'support': supp.tolist(),
                    'macro_f1': precision_recall_fscore_support(
                        data['labels'], data['predictions'], average='macro', zero_division=0
                    )[2]
                }

        # Distribution statistics
        language_distribution = Counter(m.get('lang', 'unknown') for m in self.metadata)
        label_distribution = Counter(self.labels)

        return {
            'overall': overall_metrics,
            'per_language': language_metrics,
            'language_distribution': dict(language_distribution),
            'label_distribution': dict(label_distribution),
            'total_samples': len(self.predictions)
        }

    def save_detailed_report(self, filepath: str):
        """Save detailed performance report to JSON file."""
        metrics = self.calculate_metrics()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print formatted performance summary."""
        metrics = self.calculate_metrics()

        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        # Overall metrics
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['overall']['accuracy']:.3f}")
        print(f"  Macro F1: {metrics['overall']['macro_f1']:.3f}")

        # Per-language metrics
        if metrics['per_language']:
            print(f"\nPer-Language Performance:")
            print(f"{'Language':<10} {'Samples':<10} {'Accuracy':<10} {'Macro F1':<10}")
            print("-"*40)

            for lang, lang_metrics in sorted(metrics['per_language'].items()):
                print(f"{lang:<10} {lang_metrics['n_samples']:<10} "
                      f"{lang_metrics['accuracy']:<10.3f} {lang_metrics['macro_f1']:<10.3f}")

        # Distribution
        print(f"\nLanguage Distribution:")
        for lang, count in sorted(metrics['language_distribution'].items()):
            print(f"  {lang}: {count} ({count/metrics['total_samples']*100:.1f}%)")

        print("="*60 + "\n")


def extract_samples_data(samples: List[DataSample]) -> Tuple[List[str], List, List[Dict]]:
    """
    Extract texts, labels and metadata from DataSample list.

    Args:
        samples: List of DataSample objects

    Returns:
        Tuple of (texts, labels, metadata)
    """
    texts = [s.text for s in samples]
    labels = [s.label for s in samples]
    metadata = [s.to_dict() for s in samples]

    return texts, labels, metadata

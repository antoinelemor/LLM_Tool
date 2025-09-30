"""
PROJECT:
-------
LLMTool

TITLE:
------
data_utils.py

MAIN OBJECTIVE:
---------------
This script provides comprehensive data handling utilities with metadata support,
enabling language-specific performance tracking, stratified sampling, and preserving
sample metadata throughout the training pipeline for social science research.

Dependencies:
-------------
- torch (PyTorch tensors and datasets)
- numpy (numerical operations)
- json & csv (file I/O)
- scikit-learn (metrics calculation)

MAIN FEATURES:
--------------
1) DataSample class for structured data with metadata (ID, language, custom fields)
2) MetadataDataset for preserving metadata alongside tensors
3) Enhanced DataLoader supporting JSONL and CSV with metadata extraction
4) PerformanceTracker for language-specific and stratified metrics
5) Automatic language detection for samples
6) Data splitting with stratification by language and label
7) Batch-wise performance tracking during training
8) Comprehensive metrics export (confusion matrices, per-class metrics)

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
                      random_seed: int = 42) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Split data into train/val/test sets with optional language stratification.

        Args:
            samples: List of DataSample objects
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by_lang: Whether to stratify splits by language
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        np.random.seed(random_seed)

        if stratify_by_lang and any(s.lang for s in samples):
            # Group samples by language
            lang_groups = defaultdict(list)
            for sample in samples:
                lang_groups[sample.lang or 'unknown'].append(sample)

            train_samples = []
            val_samples = []
            test_samples = []

            # Split each language group
            for lang, lang_samples in lang_groups.items():
                n = len(lang_samples)
                indices = np.random.permutation(n)

                train_end = int(n * train_ratio)
                val_end = train_end + int(n * val_ratio)

                train_samples.extend([lang_samples[i] for i in indices[:train_end]])
                val_samples.extend([lang_samples[i] for i in indices[train_end:val_end]])
                test_samples.extend([lang_samples[i] for i in indices[val_end:]])

        else:
            # Simple random split
            n = len(samples)
            indices = np.random.permutation(n)

            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_samples = [samples[i] for i in indices[:train_end]]
            val_samples = [samples[i] for i in indices[train_end:val_end]]
            test_samples = [samples[i] for i in indices[val_end:]]

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
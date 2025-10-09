#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
training_data_utils.py

MAIN OBJECTIVE:
---------------
Centralized utilities for training data management with systematic logging,
session organization, and distribution validation across ALL training modes.

Dependencies:
-------------
- pandas
- numpy
- json
- pathlib

MAIN FEATURES:
--------------
1) Session-based organization of training data (training_data/{session_id}/)
2) Comprehensive distribution logging for all datasets
3) Validation warnings for insufficient data
4) Split summary reports (train/val/test statistics)
5) Cross-mode compatibility (multi-label, multi-class, one-vs-all, etc.)

Author:
-------
Antoine Lemor
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class TrainingDataSessionManager:
    """
    Manages session-based organization of training data with comprehensive logging.

    New centralized structure:
        logs/training_arena/{session_id}/
        ├── training_data/              # Dataset analysis and reports
        │   ├── model_catalog.csv
        │   ├── SESSION_SUMMARY.txt
        │   ├── quick_summary.csv
        │   ├── split_summary.csv
        │   └── distribution_report.json
        ├── training_metrics/           # Training metrics (from trainer)
        │   └── (populated by ModelTrainer)
        └── training_session_metadata/  # Session parameters
            └── training_metadata.json

        data/training_data/{session_id}/
        └── training_data/              # Actual JSONL dataset files
            ├── multiclass_*.jsonl
            ├── multilabel_*.jsonl
            └── onevsall_*.jsonl
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session manager with centralized log structure.

        Args:
            session_id: Session ID (timestamp). If None, creates new session.
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Data directory: Raw JSONL files
        self.data_base_dir = Path("data/training_data")
        self.datasets_dir = self.data_base_dir / self.session_id / "training_data"

        # Logs directory: All reports and analysis
        self.logs_base_dir = Path("logs/training_arena")
        self.session_dir = self.logs_base_dir / self.session_id
        self.training_data_logs_dir = self.session_dir / "training_data"
        self.training_metrics_dir = self.session_dir / "training_metrics"
        self.metadata_dir = self.session_dir / "training_session_metadata"

        # Create all directories
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_logs_dir.mkdir(parents=True, exist_ok=True)
        self.training_metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validation warnings log
        self.warnings_log = self.training_data_logs_dir / "validation_warnings.log"
        self.warnings_logger = self._setup_warnings_logger()

        # Distribution data accumulator
        self.distribution_data = defaultdict(dict)

    def _setup_warnings_logger(self) -> logging.Logger:
        """Setup dedicated logger for validation warnings."""
        warnings_logger = logging.getLogger(f"training_data_warnings_{self.session_id}")
        warnings_logger.setLevel(logging.WARNING)

        # Prevent warnings from propagating to root logger (avoid console spam)
        warnings_logger.propagate = False

        # File handler - warnings only go to file, not console
        handler = logging.FileHandler(self.warnings_log)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        warnings_logger.addHandler(handler)

        return warnings_logger

    def get_dataset_path(self, dataset_name: str, suffix: str = ".jsonl") -> Path:
        """
        Get path for a dataset file in the session datasets directory.

        Args:
            dataset_name: Name of the dataset
            suffix: File suffix (default: .jsonl)

        Returns:
            Path to dataset file
        """
        return self.datasets_dir / f"{dataset_name}{suffix}"

    def log_distribution(self,
                        dataset_name: str,
                        train_samples: List[Any],
                        val_samples: List[Any],
                        test_samples: Optional[List[Any]] = None,
                        label_key: Optional[str] = None,
                        metadata: Optional[Dict] = None):
        """
        Log dataset distribution statistics.

        Args:
            dataset_name: Name of the dataset/category
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples (optional)
            label_key: Label key name for multi-label datasets
            metadata: Additional metadata to store
        """
        # Extract labels from samples
        train_labels = self._extract_labels(train_samples)
        val_labels = self._extract_labels(val_samples)
        test_labels = self._extract_labels(test_samples) if test_samples else []

        # Calculate distributions
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        test_dist = Counter(test_labels) if test_labels else Counter()

        # Calculate class imbalance metrics
        all_labels = train_labels + val_labels + test_labels
        imbalance_metrics = self._calculate_imbalance_metrics(all_labels)

        # Check for validation issues
        warnings = self._validate_distribution(
            dataset_name, train_dist, val_dist, test_dist
        )

        # Store distribution data
        self.distribution_data[dataset_name] = {
            'label_key': label_key,
            'total_samples': len(all_labels),
            'train_samples': len(train_labels),
            'val_samples': len(val_labels),
            'test_samples': len(test_labels),
            'num_classes': len(set(all_labels)),
            'train_distribution': dict(train_dist),
            'val_distribution': dict(val_dist),
            'test_distribution': dict(test_dist) if test_dist else {},
            'imbalance_metrics': imbalance_metrics,
            'warnings': warnings,
            'metadata': metadata or {}
        }

        # Log warnings if any
        if warnings:
            for warning in warnings:
                self.warnings_logger.warning(f"[{dataset_name}] {warning}")

    def _extract_labels(self, samples: List[Any]) -> List[Any]:
        """Extract labels from samples (supports dict, DataSample, and direct labels)."""
        if not samples:
            return []

        labels = []
        for sample in samples:
            if isinstance(sample, dict):
                # Try different possible label keys
                label = sample.get('label', sample.get('labels', sample.get('target')))
            elif hasattr(sample, 'label'):
                label = sample.label
            else:
                label = sample

            # Handle list labels (multi-label case) - convert to string for JSON compatibility
            if isinstance(label, list):
                label = str(tuple(sorted(label)))  # Convert to hashable string
            elif isinstance(label, tuple):
                label = str(label)  # Convert existing tuples to strings

            labels.append(label)

        return labels

    def _calculate_imbalance_metrics(self, labels: List[Any]) -> Dict[str, float]:
        """Calculate class imbalance metrics."""
        if not labels:
            return {
                'imbalance_ratio': 1.0,
                'gini_index': 0.0,
                'min_samples': 0,
                'max_samples': 0,
                'mean_samples': 0.0
            }

        counts = Counter(labels)
        values = np.array(list(counts.values()))

        if len(counts) == 1:
            return {
                'imbalance_ratio': 1.0,
                'gini_index': 1.0,
                'min_samples': int(values[0]),
                'max_samples': int(values[0]),
                'mean_samples': float(values[0])
            }

        # Imbalance ratio
        imbalance_ratio = values.max() / values.min()

        # Gini index
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

        return {
            'imbalance_ratio': float(imbalance_ratio),
            'gini_index': float(gini),
            'min_samples': int(values.min()),
            'max_samples': int(values.max()),
            'mean_samples': float(values.mean())
        }

    def _validate_distribution(self,
                               dataset_name: str,
                               train_dist: Counter,
                               val_dist: Counter,
                               test_dist: Counter) -> List[str]:
        """
        Validate distribution and return list of warnings.

        Checks:
        - Minimum samples per class
        - Classes present in both train and val
        - Empty validation set
        - Severe class imbalance

        Note: If validation set is empty, this assumes the split hasn't happened yet,
        so most validation checks are skipped.
        """
        warnings = []

        # Check validation set not empty
        if not val_dist:
            # No validation set means split hasn't happened yet - skip detailed validation
            # Only warn if there are very few total samples
            total_samples = sum(train_dist.values())
            if total_samples < 50:
                warnings.append(
                    f"Only {total_samples} total samples - consider adding more data "
                    f"(minimum recommended: 50)"
                )
            return warnings

        # Only perform detailed validation if we have a validation set
        # Check minimum samples
        MIN_SAMPLES_PER_CLASS = 10
        for label, count in train_dist.items():
            if count < MIN_SAMPLES_PER_CLASS:
                warnings.append(
                    f"Class '{label}' has only {count} training samples "
                    f"(minimum recommended: {MIN_SAMPLES_PER_CLASS})"
                )

        # Check classes in both train and val
        train_classes = set(train_dist.keys())
        val_classes = set(val_dist.keys())

        missing_in_val = train_classes - val_classes
        if missing_in_val:
            warnings.append(
                f"Classes present in train but missing in validation: {sorted(list(missing_in_val))}"
            )

        missing_in_train = val_classes - train_classes
        if missing_in_train:
            warnings.append(
                f"Classes present in validation but missing in train: {sorted(list(missing_in_train))}"
            )

        # Check severe imbalance
        if len(train_dist) > 1:
            counts = list(train_dist.values())
            imbalance_ratio = max(counts) / min(counts)
            if imbalance_ratio > 50:
                warnings.append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    f"Consider using reinforced learning."
                )

        # Check very small validation set
        MIN_VAL_SAMPLES_PER_CLASS = 3
        for label, count in val_dist.items():
            if count < MIN_VAL_SAMPLES_PER_CLASS:
                warnings.append(
                    f"Class '{label}' has only {count} validation samples "
                    f"(minimum recommended: {MIN_VAL_SAMPLES_PER_CLASS})"
                )

        return warnings

    def _sanitize_for_json(self, obj):
        """
        Recursively sanitize data structure for JSON serialization.
        Converts sets to sorted lists, handles nested dicts/lists.
        """
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        else:
            return obj

    def save_distribution_report(self):
        """Save comprehensive distribution report as JSON."""
        report_path = self.training_data_logs_dir / "distribution_report.json"

        # Sanitize data to ensure JSON compatibility (convert sets to lists)
        sanitized_data = self._sanitize_for_json({
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'datasets': dict(self.distribution_data),
            'summary': self._generate_summary()
        })

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Distribution report saved: {report_path}")

    def save_model_catalog_csv(self):
        """
        Save comprehensive model catalog CSV listing ALL models to be trained.

        For multi-class datasets: Creates 1 entry
        For one-vs-all/multilabel datasets: Creates N entries (one per label value)

        This ensures complete traceability of every model that will be trained.
        """
        catalog_rows = []
        model_id = 1

        for dataset_name, data in self.distribution_data.items():
            metadata = data.get('metadata', {})
            split_config = metadata.get('split_config', {})
            strategy = metadata.get('strategy', '')
            training_approach = metadata.get('training_approach', '')

            # Extract split configuration
            custom_splits = split_config.get('custom_splits', {})
            defaults = split_config.get('defaults', {'train': 0.8, 'val': 0.2, 'test': 0.0})

            if dataset_name in custom_splits:
                split_ratio = custom_splits[dataset_name]
            else:
                split_ratio = defaults

            train_ratio = split_ratio.get('train', 0.8)
            val_ratio = split_ratio.get('val', 0.2)
            test_ratio = split_ratio.get('test', 0.0)

            # Get dataset info
            total = data['total_samples']
            num_classes = data['num_classes']
            train_dist = data.get('train_distribution', {})
            imbalance = data.get('imbalance_metrics', {})
            warnings = data.get('warnings', [])
            split_indicator = "CUSTOM" if dataset_name in custom_splits else "DEFAULT"

            # Detect if this is a one-vs-all or multilabel dataset based on metadata
            # These should be expanded into individual binary classifiers
            # Check the training_approach from metadata to determine if it's really one-vs-all
            is_multilabel_collection = False
            if training_approach == 'one-vs-all':
                # Only expand if explicitly one-vs-all approach
                is_multilabel_collection = True
            elif 'multilabel' in dataset_name.lower() and training_approach != 'multi-class':
                # For multilabel datasets, only expand if not explicitly multi-class
                is_multilabel_collection = True

            if is_multilabel_collection and num_classes > 2:
                # This is a collection of binary classifiers - expand it
                # Each "class" in the distribution represents a unique label combination
                # that will be used to train a binary classifier

                # Parse the label combinations to extract individual labels
                import ast
                all_labels = set()
                for class_key in train_dist.keys():
                    if class_key == '()':
                        continue
                    try:
                        # Parse as Python literal (tuple)
                        parsed = ast.literal_eval(str(class_key))
                        if isinstance(parsed, tuple):
                            all_labels.update(parsed)
                        else:
                            all_labels.add(str(parsed))
                    except:
                        # If parsing fails, treat as single label
                        if class_key:
                            all_labels.add(str(class_key))

                # Create one model entry per unique label
                for label in sorted(all_labels):
                    if label and label != '()':
                        # Calculate approximate samples for this binary classifier
                        # In one-vs-all, each binary model uses all samples
                        train_proj = int(total * train_ratio)
                        val_proj = int(total * val_ratio)
                        test_proj = int(total * test_ratio)

                        catalog_rows.append({
                            'model_id': model_id,
                            'model_name': f"{dataset_name}_{label}",
                            'base_dataset': dataset_name,
                            'type': 'binary (one-vs-all)',
                            'target_label': label,
                            'num_classes': 2,
                            'classes': f"{label}, not-{label}",
                            'split_config': split_indicator,
                            'train_ratio': f"{train_ratio*100:.1f}%",
                            'val_ratio': f"{val_ratio*100:.1f}%",
                            'test_ratio': f"{test_ratio*100:.1f}%",
                            'total_samples': total,
                            'train_samples_projected': train_proj,
                            'val_samples_projected': val_proj,
                            'test_samples_projected': test_proj,
                            'imbalance_ratio': 'varies',
                            'num_warnings': len(warnings),
                            'warnings_summary': '',
                            'file_size_mb': round(metadata.get('file_size_mb', 0) / len(all_labels), 2),
                            'strategy': strategy,
                            'training_approach': training_approach,
                        })
                        model_id += 1
            else:
                # Multi-class model - create single entry
                train_proj = int(total * train_ratio)
                val_proj = int(total * val_ratio)
                test_proj = int(total * test_ratio)

                classes_list = sorted([str(k) for k in train_dist.keys()])
                classes_str = ', '.join(classes_list[:5])
                if len(classes_list) > 5:
                    classes_str += f" ... (+{len(classes_list)-5} more)"

                imbalance_ratio = imbalance.get('imbalance_ratio', 1.0)
                warnings_str = '; '.join(warnings[:3]) if warnings else ''
                if len(warnings) > 3:
                    warnings_str += f' (+{len(warnings)-3} more)'

                catalog_rows.append({
                    'model_id': model_id,
                    'model_name': dataset_name,
                    'base_dataset': dataset_name,
                    'type': 'multi-class',
                    'target_label': 'N/A',
                    'num_classes': num_classes,
                    'classes': classes_str,
                    'split_config': split_indicator,
                    'train_ratio': f"{train_ratio*100:.1f}%",
                    'val_ratio': f"{val_ratio*100:.1f}%",
                    'test_ratio': f"{test_ratio*100:.1f}%",
                    'total_samples': total,
                    'train_samples_projected': train_proj,
                    'val_samples_projected': val_proj,
                    'test_samples_projected': test_proj,
                    'imbalance_ratio': round(imbalance_ratio, 2),
                    'num_warnings': len(warnings),
                    'warnings_summary': warnings_str,
                    'file_size_mb': round(metadata.get('file_size_mb', 0), 2),
                    'strategy': strategy,
                    'training_approach': training_approach,
                })
                model_id += 1

        if catalog_rows:
            df = pd.DataFrame(catalog_rows)
            catalog_path = self.training_data_logs_dir / "model_catalog.csv"
            df.to_csv(catalog_path, index=False)
            logger.info(f"Model catalog saved to {catalog_path} with {len(catalog_rows)} models")

    def save_split_summary_csv(self):
        """Save comprehensive split summary as CSV with detailed statistics."""
        rows = []

        for dataset_name, data in self.distribution_data.items():
            # Get metadata
            metadata = data.get('metadata', {})
            imbalance = data.get('imbalance_metrics', {})

            # Overall dataset row (summary)
            rows.append({
                'dataset_name': dataset_name,
                'split': 'ALL',
                'label': 'TOTAL',
                'samples': data['total_samples'],
                'percentage': 100.0,
                'unique_classes': data['num_classes'],
                'imbalance_ratio': round(imbalance.get('imbalance_ratio', 1.0), 2),
                'min_samples_per_class': imbalance.get('min_samples', 0),
                'max_samples_per_class': imbalance.get('max_samples', 0),
                'mean_samples_per_class': round(imbalance.get('mean_samples', 0.0), 2),
                'file_size_mb': metadata.get('file_size_mb', 0),
                'strategy': metadata.get('strategy', ''),
                'training_approach': metadata.get('training_approach', ''),
            })

            # Split-level summaries
            for split_name in ['train', 'val', 'test']:
                split_samples = data.get(f'{split_name}_samples', 0)
                dist = data.get(f'{split_name}_distribution', {})

                if split_samples > 0:
                    # Split summary row
                    rows.append({
                        'dataset_name': dataset_name,
                        'split': split_name.upper(),
                        'label': 'SUBTOTAL',
                        'samples': split_samples,
                        'percentage': round(split_samples / data['total_samples'] * 100, 2) if data['total_samples'] > 0 else 0,
                        'unique_classes': len(dist),
                        'imbalance_ratio': '',
                        'min_samples_per_class': min(dist.values()) if dist else 0,
                        'max_samples_per_class': max(dist.values()) if dist else 0,
                        'mean_samples_per_class': round(sum(dist.values()) / len(dist), 2) if dist else 0,
                        'file_size_mb': '',
                        'strategy': '',
                        'training_approach': '',
                    })

                # Individual class rows
                if dist:
                    for label, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                        rows.append({
                            'dataset_name': dataset_name,
                            'split': split_name.upper(),
                            'label': str(label),
                            'samples': count,
                            'percentage': round(count / split_samples * 100, 2) if split_samples > 0 else 0,
                            'unique_classes': '',
                            'imbalance_ratio': '',
                            'min_samples_per_class': '',
                            'max_samples_per_class': '',
                            'mean_samples_per_class': '',
                            'file_size_mb': '',
                            'strategy': '',
                            'training_approach': '',
                        })

        if rows:
            df = pd.DataFrame(rows)

            # Reorder columns for better readability
            column_order = [
                'dataset_name',
                'split',
                'label',
                'samples',
                'percentage',
                'unique_classes',
                'imbalance_ratio',
                'min_samples_per_class',
                'max_samples_per_class',
                'mean_samples_per_class',
                'file_size_mb',
                'strategy',
                'training_approach'
            ]
            df = df[column_order]

            csv_path = self.training_data_logs_dir / "split_summary.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Split summary saved: {csv_path}")

            # Also create a simplified summary CSV for quick overview
            summary_rows = []
            for dataset_name, data in self.distribution_data.items():
                metadata = data.get('metadata', {})
                imbalance = data.get('imbalance_metrics', {})

                # Extract split config for this dataset
                split_config = metadata.get('split_config', {})
                planned_split = "N/A"
                if split_config:
                    # Check if there's a custom config for this dataset
                    if 'custom_splits' in split_config:
                        custom = split_config['custom_splits']
                        # Look for value-specific or key-specific config
                        if dataset_name in custom:
                            cfg = custom[dataset_name]
                            planned_split = f"{cfg.get('train', 0)*100:.0f}/{cfg.get('val', 0)*100:.0f}/{cfg.get('test', 0)*100:.0f}"

                    # Fall back to defaults
                    if planned_split == "N/A" and 'defaults' in split_config:
                        defaults = split_config['defaults']
                        planned_split = f"{defaults.get('train', 0)*100:.0f}/{defaults.get('val', 0)*100:.0f}/{defaults.get('test', 0)*100:.0f}"

                # Determine data status
                has_val = data.get('val_samples', 0) > 0
                has_test = data.get('test_samples', 0) > 0
                data_status = "SPLIT" if (has_val or has_test) else "PRE-SPLIT"

                summary_rows.append({
                    'dataset': dataset_name,
                    'data_status': data_status,
                    'total_samples': data['total_samples'],
                    'train_samples': data.get('train_samples', 0),
                    'val_samples': data.get('val_samples', 0),
                    'test_samples': data.get('test_samples', 0),
                    'train_pct': round(data.get('train_samples', 0) / data['total_samples'] * 100, 1) if data['total_samples'] > 0 else 0,
                    'val_pct': round(data.get('val_samples', 0) / data['total_samples'] * 100, 1) if data['total_samples'] > 0 else 0,
                    'test_pct': round(data.get('test_samples', 0) / data['total_samples'] * 100, 1) if data['total_samples'] > 0 else 0,
                    'planned_split': planned_split,
                    'unique_classes': data['num_classes'],
                    'imbalance_ratio': round(imbalance.get('imbalance_ratio', 1.0), 2),
                    'file_size_mb': round(metadata.get('file_size_mb', 0), 2),
                    'warnings': len(data.get('warnings', [])),
                    'strategy': metadata.get('strategy', ''),
                    'training_approach': metadata.get('training_approach', ''),
                })

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                quick_summary_path = self.training_data_logs_dir / "quick_summary.csv"
                summary_df.to_csv(quick_summary_path, index=False)
                logger.info(f"Quick summary saved: {quick_summary_path}")

    def _generate_summary(self) -> Dict:
        """Generate overall summary statistics."""
        if not self.distribution_data:
            return {}

        total_samples = sum(d['total_samples'] for d in self.distribution_data.values())
        total_train = sum(d['train_samples'] for d in self.distribution_data.values())
        total_val = sum(d['val_samples'] for d in self.distribution_data.values())
        total_test = sum(d['test_samples'] for d in self.distribution_data.values())

        total_warnings = sum(len(d['warnings']) for d in self.distribution_data.values())
        datasets_with_warnings = sum(1 for d in self.distribution_data.values() if d['warnings'])

        return {
            'total_datasets': len(self.distribution_data),
            'total_samples': total_samples,
            'total_train_samples': total_train,
            'total_val_samples': total_val,
            'total_test_samples': total_test,
            'total_warnings': total_warnings,
            'datasets_with_warnings': datasets_with_warnings,
            'split_ratios': {
                'train': total_train / total_samples if total_samples > 0 else 0,
                'val': total_val / total_samples if total_samples > 0 else 0,
                'test': total_test / total_samples if total_samples > 0 else 0
            }
        }

    def finalize(self, training_context: Optional[Dict[str, Any]] = None):
        """
        Finalize session: save all reports and logs.

        Args:
            training_context: Optional training/benchmark context information
        """
        self.save_distribution_report()
        self.save_model_catalog_csv()
        self.save_split_summary_csv()

        # Create session summary file
        summary_path = self.session_dir / "SESSION_SUMMARY.txt"
        summary = self._generate_summary()

        # Store training context for use in report generation
        self.training_context = training_context

        # Extract configuration metadata from first dataset
        config_metadata = {}
        if self.distribution_data:
            first_dataset = next(iter(self.distribution_data.values()))
            config_metadata = first_dataset.get('metadata', {})

        with open(summary_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("═" * 80 + "\n")
            f.write(f"  TRAINING DATA SESSION SUMMARY\n")
            f.write(f"  Session ID: {self.session_id}\n")
            f.write("═" * 80 + "\n\n")

            # Configuration Section
            f.write("📋 CONFIGURATION\n")
            f.write("─" * 80 + "\n")
            if config_metadata:
                f.write(f"  Training Strategy:     {config_metadata.get('strategy', 'N/A')}\n")
                f.write(f"  Training Approach:     {config_metadata.get('training_approach', 'N/A')}\n")
                f.write(f"  Text Column:           {config_metadata.get('text_column', 'N/A')}\n")
                f.write(f"  Label Column:          {config_metadata.get('label_column', 'N/A')}\n")
                f.write(f"  Source File:           {config_metadata.get('source_file', 'N/A')}\n")

                # Categories
                categories = config_metadata.get('categories', [])
                if categories:
                    f.write(f"\n  Selected Categories ({len(categories)}):\n")
                    for i, cat in enumerate(categories, 1):
                        f.write(f"    {i}. {cat}\n")

                # Languages
                languages = config_metadata.get('confirmed_languages', [])
                if languages:
                    f.write(f"\n  Languages:             {', '.join(languages)}\n")

                # Split configuration
                split_config = config_metadata.get('split_config', {})
                if split_config:
                    f.write(f"\n  Data Split Configuration:\n")
                    # Extract actual split ratios from defaults or custom configs
                    defaults = split_config.get('defaults', {})
                    train_ratio = defaults.get('train', 0.8)
                    val_ratio = defaults.get('val', 0.2)
                    test_ratio = defaults.get('test', 0.0)

                    # Format as percentages
                    f.write(f"    Train:               {train_ratio*100:.0f}%\n")
                    f.write(f"    Validation:          {val_ratio*100:.0f}%\n")
                    f.write(f"    Test:                {test_ratio*100:.0f}%\n")
            else:
                f.write("  No configuration metadata available\n")

            f.write("\n")

            # Dataset Summary Section
            f.write("📊 DATASET SUMMARY\n")
            f.write("─" * 80 + "\n")

            # Determine if data is pre-split or post-split
            has_split_data = summary.get('total_val_samples', 0) > 0 or summary.get('total_test_samples', 0) > 0
            data_status = "SPLIT (train/val/test separated)" if has_split_data else "PRE-SPLIT (will be split during training)"

            f.write(f"  Data Status:           {data_status}\n")
            f.write(f"  Total Datasets:        {summary.get('total_datasets', 0)}\n")
            f.write(f"  Total Samples:         {summary.get('total_samples', 0):,}\n")

            if has_split_data:
                # Post-split: show actual distribution
                f.write(f"    - Training:          {summary.get('total_train_samples', 0):,}\n")
                f.write(f"    - Validation:        {summary.get('total_val_samples', 0):,}\n")
                f.write(f"    - Test:              {summary.get('total_test_samples', 0):,}\n")

                split_ratios = summary.get('split_ratios', {})
                if split_ratios:
                    f.write(f"\n  Split Ratios:\n")
                    f.write(f"    - Train:             {split_ratios.get('train', 0)*100:.1f}%\n")
                    f.write(f"    - Validation:        {split_ratios.get('val', 0)*100:.1f}%\n")
                    f.write(f"    - Test:              {split_ratios.get('test', 0)*100:.1f}%\n")
            else:
                # Pre-split: show planned configuration
                f.write(f"\n  ⏱️  Data will be split during training according to configuration.\n")

                # Extract split config from metadata
                if config_metadata and 'split_config' in config_metadata:
                    split_config = config_metadata['split_config']
                    if 'defaults' in split_config:
                        defaults = split_config['defaults']
                        f.write(f"\n  Planned Default Split:\n")
                        f.write(f"    - Train:             {defaults.get('train', 0)*100:.0f}%\n")
                        f.write(f"    - Validation:        {defaults.get('val', 0)*100:.0f}%\n")
                        f.write(f"    - Test:              {defaults.get('test', 0)*100:.0f}%\n")

                    if 'custom_splits' in split_config and split_config['custom_splits']:
                        f.write(f"\n  Custom Splits Configured:\n")
                        for key, cfg in split_config['custom_splits'].items():
                            f.write(f"    - {key}: {cfg.get('train', 0)*100:.0f}% / {cfg.get('val', 0)*100:.0f}% / {cfg.get('test', 0)*100:.0f}%\n")

            f.write("\n")

            # Models to be trained section
            f.write("🎯 MODELS TO BE TRAINED\n")
            f.write("─" * 80 + "\n")

            # Load model catalog for detailed listing
            catalog_path = self.training_data_logs_dir / "model_catalog.csv"
            if catalog_path.exists():
                import pandas as pd
                catalog_df = pd.read_csv(catalog_path)

                # Count models by type from catalog
                multiclass_count = len(catalog_df[catalog_df['type'] == 'multi-class'])
                binary_count = len(catalog_df[catalog_df['type'] == 'binary (one-vs-all)'])
                total_models = len(catalog_df)

                f.write(f"  Total Models:          {total_models}\n")
                f.write(f"    - Multi-class:       {multiclass_count} model(s)\n")
                f.write(f"    - Binary (1-vs-all): {binary_count} model(s)\n\n")

                # List multi-class models
                multiclass_df = catalog_df[catalog_df['type'] == 'multi-class']
                if len(multiclass_df) > 0:
                    f.write(f"  📊 MULTI-CLASS MODELS ({len(multiclass_df)} models)\n")
                    f.write(f"  {'-' * 76}\n\n")

                    for idx, row in multiclass_df.iterrows():
                        f.write(f"    Model #{row['model_id']}: {row['model_name']}\n")
                        f.write(f"      Type:              {row['type']}\n")
                        f.write(f"      Classes:           {row['num_classes']} ({row['classes']})\n")
                        f.write(f"      Split Ratio:       {row['train_ratio']} / {row['val_ratio']} / {row['test_ratio']}")
                        if row['split_config'] == 'CUSTOM':
                            f.write(" ← CUSTOM\n")
                        else:
                            f.write("\n")
                        f.write(f"      Samples (projected):\n")
                        f.write(f"        - Train:         {row['train_samples_projected']:,}\n")
                        f.write(f"        - Validation:    {row['val_samples_projected']:,}\n")
                        f.write(f"        - Test:          {row['test_samples_projected']:,}\n")
                        f.write(f"      Imbalance Ratio:   {row['imbalance_ratio']}:1\n")
                        if row['num_warnings'] > 0:
                            f.write(f"      ⚠️  Warnings:        {row['num_warnings']} ({row['warnings_summary'][:80]}...)\n")
                        f.write("\n")

                # List binary (one-vs-all) models
                binary_df = catalog_df[catalog_df['type'] == 'binary (one-vs-all)']
                if len(binary_df) > 0:
                    f.write(f"\n  ⚡ BINARY (ONE-VS-ALL) MODELS ({len(binary_df)} models)\n")
                    f.write(f"  {'-' * 76}\n\n")

                    # Group by base_dataset
                    for base_dataset in binary_df['base_dataset'].unique():
                        group_df = binary_df[binary_df['base_dataset'] == base_dataset]
                        f.write(f"    {base_dataset} ({len(group_df)} binary classifiers)\n\n")

                        # Show first 10 models, then summarize
                        max_to_show = 10
                        for idx, row in group_df.head(max_to_show).iterrows():
                            f.write(f"      Model #{row['model_id']}: {row['target_label']}\n")
                            f.write(f"        Classes:         {row['classes']}\n")
                            f.write(f"        Split Ratio:     {row['train_ratio']} / {row['val_ratio']} / {row['test_ratio']}")
                            if row['split_config'] == 'CUSTOM':
                                f.write(" ← CUSTOM\n")
                            else:
                                f.write("\n")
                            f.write(f"        Train/Val/Test:  {row['train_samples_projected']:,} / {row['val_samples_projected']:,} / {row['test_samples_projected']:,}\n")
                            f.write("\n")

                        if len(group_df) > max_to_show:
                            f.write(f"      ... and {len(group_df) - max_to_show} more binary classifiers\n")
                            f.write(f"      (See model_catalog.csv for complete list)\n\n")

                f.write(f"  💡 Complete list: model_catalog.csv ({total_models} models with full details)\n\n")
            else:
                # Fallback if catalog doesn't exist yet
                total_models = len(self.distribution_data)
                f.write(f"  Total Datasets:        {total_models}\n")
                f.write(f"  ⚠️  Model catalog not yet generated\n\n")

            f.write("\n")

            # Training Execution Summary (if training context provided)
            if self.training_context:
                f.write("🎓 TRAINING EXECUTION SUMMARY\n")
                f.write("─" * 80 + "\n")

                mode = self.training_context.get('mode', 'unknown')
                models_trained = self.training_context.get('models_trained', [])

                f.write(f"  Training Mode:         {mode.upper()}\n")
                f.write(f"  Models Trained:        {len(models_trained)}\n")

                if models_trained:
                    f.write(f"\n  Trained Models:\n")
                    for i, model in enumerate(models_trained[:10], 1):
                        f.write(f"    {i}. {model}\n")
                    if len(models_trained) > 10:
                        f.write(f"    ... and {len(models_trained) - 10} more\n")

                # Benchmark-specific information
                if mode == 'benchmark':
                    f.write(f"\n  Benchmark Mode:        ACTIVE\n")
                    benchmark_results = self.training_context.get('benchmark_results')
                    if benchmark_results:
                        f.write(f"  Benchmark Results:     Included in reports\n")
                        f.write(f"  Purpose:               Model comparison and selection\n")

                    f.write(f"\n  💡 Benchmark datasets are used for model comparison only.\n")
                    f.write(f"     Final training may use different data splits or configurations.\n")

                # Runtime parameters
                runtime_params = self.training_context.get('runtime_params', {})
                if runtime_params:
                    f.write(f"\n  Runtime Parameters:\n")
                    for key, value in list(runtime_params.items())[:5]:
                        f.write(f"    {key}: {value}\n")

                f.write("\n")

            # Individual Dataset Details
            if self.distribution_data:
                f.write("📁 INDIVIDUAL DATASETS\n")
                f.write("─" * 80 + "\n")
                for dataset_name, data in sorted(self.distribution_data.items()):
                    f.write(f"\n  Dataset: {dataset_name}\n")
                    f.write(f"    Total Samples:       {data['total_samples']:,}\n")
                    f.write(f"    Unique Classes:      {data['num_classes']}\n")

                    # Imbalance metrics
                    imbalance = data.get('imbalance_metrics', {})
                    if imbalance and imbalance.get('imbalance_ratio', 1.0) > 1.0:
                        f.write(f"    Imbalance Ratio:     {imbalance.get('imbalance_ratio', 0):.2f}:1\n")
                        f.write(f"    Min/Max Samples:     {imbalance.get('min_samples', 0)} / {imbalance.get('max_samples', 0)}\n")

                    # File info
                    file_path = data.get('metadata', {}).get('file_path', '')
                    if file_path:
                        file_size = data.get('metadata', {}).get('file_size_mb', 0)
                        f.write(f"    File Size:           {file_size:.2f} MB\n")
                        f.write(f"    File Path:           {file_path}\n")

                f.write("\n")

            # Validation Section
            f.write("⚠️  VALIDATION\n")
            f.write("─" * 80 + "\n")
            f.write(f"  Total Warnings:        {summary.get('total_warnings', 0)}\n")
            f.write(f"  Datasets with Issues:  {summary.get('datasets_with_warnings', 0)}\n")

            if summary.get('total_warnings', 0) > 0:
                f.write(f"\n  ⚠️  Some datasets have quality issues.\n")
                f.write(f"  Review: {self.training_data_logs_dir / 'validation_warnings.log'}\n")
            else:
                f.write(f"\n  ✓ All datasets passed validation checks\n")

            f.write("\n")

            # Files & Locations
            f.write("📂 FILES & LOCATIONS\n")
            f.write("─" * 80 + "\n")
            f.write(f"  Session Directory:     {self.session_dir}/\n")
            f.write(f"  Training Data:         {self.datasets_dir}/\n")
            f.write(f"  Logs & Reports:        {self.training_data_logs_dir}/\n")
            f.write(f"\n  Reports:\n")

            # Check if catalog exists to show model count
            if catalog_path.exists():
                f.write(f"    - Model Catalog:     model_catalog.csv (ALL {total_models} models - PRIMARY REPORT)\n")
            else:
                f.write(f"    - Model Catalog:     model_catalog.csv (PRIMARY REPORT)\n")

            f.write(f"    - Quick Summary:     quick_summary.csv (dataset overview)\n")
            f.write(f"    - Split Details:     split_summary.csv (detailed class breakdown)\n")
            f.write(f"    - Distribution:      distribution_report.json (complete raw data)\n")
            f.write(f"    - Warnings Log:      validation_warnings.log (quality issues)\n")

            f.write("\n")
            f.write("═" * 80 + "\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═" * 80 + "\n")

        logger.info(f"Session finalized: {self.session_dir}")

        # Return warnings summary for CLI display
        return summary.get('total_warnings', 0), summary.get('datasets_with_warnings', 0)

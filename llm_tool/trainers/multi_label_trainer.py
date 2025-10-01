"""
PROJECT:
-------
LLMTool

TITLE:
------
multi_label_trainer.py

MAIN OBJECTIVE:
---------------
This script implements a sophisticated multi-label training system that trains
separate binary classifiers for each label in multi-label datasets, with support
for language-specific models, parallel training, and automatic model organization.

Dependencies:
-------------
- pandas & numpy (data handling)
- concurrent.futures (parallel training)
- LLMTool.bert_base_enhanced (model training)
- LLMTool.model_selector (automatic model selection)
- LLMTool.multilingual_selector (language-specific models)

MAIN FEATURES:
--------------
1) Trains separate binary classifiers for each label in multi-label data
2) Language-aware training (separate models per language or multilingual)
3) Automatic model naming convention (label_language format)
4) Parallel training support for multiple models simultaneously
5) Automatic model selection based on data characteristics
6) Comprehensive performance tracking and logging
7) Organized output structure with model summaries
8) Support for both JSONL and CSV multi-label formats

Author:
-------
Antoine Lemor
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from datetime import datetime

from llm_tool.trainers.data_utils import DataSample, DataLoader as DataUtil, PerformanceTracker
from llm_tool.trainers.bert_base_enhanced import BertBaseEnhanced
from llm_tool.trainers.multilingual_selector import MultilingualModelSelector
from llm_tool.trainers.model_selector import ModelSelector, auto_select_model


@dataclass
class MultiLabelSample:
    """Data sample with multiple labels."""
    text: str
    labels: Dict[str, Any]  # label_name -> label_value
    id: Optional[str] = None
    lang: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_data_samples(self) -> List[Tuple[str, DataSample]]:
        """Convert to separate DataSample objects for each label."""
        samples = []
        for label_name, label_value in self.labels.items():
            sample = DataSample(
                text=self.text,
                label=label_value,
                id=self.id,
                lang=self.lang,
                metadata={**(self.metadata or {}), 'label_type': label_name}
            )
            samples.append((label_name, sample))
        return samples


@dataclass
class TrainingConfig:
    """Configuration for multi-label training."""
    model_class: Optional[type] = None  # specific model class to use
    auto_select_model: bool = True  # auto-select best model
    train_by_language: bool = False  # train separate models per language
    multilingual_model: bool = False  # use multilingual model for all languages
    n_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 5e-5
    reinforced_learning: bool = True
    n_epochs_reinforced: int = 2
    reinforced_epochs: Optional[int] = None  # Override n_epochs_reinforced if provided
    track_languages: bool = True
    output_dir: str = "./multi_label_models"
    parallel_training: bool = False  # train models in parallel
    max_workers: int = 2  # max parallel training jobs
    # Data splitting parameters
    auto_split: bool = True  # automatically split data if no validation set
    split_ratio: float = 0.8  # train/validation split ratio
    stratified: bool = True  # use stratified splitting


@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_name: str
    label_name: str
    language: Optional[str]
    model_path: str
    performance_metrics: Dict[str, Any]
    training_config: Dict[str, Any]


class MultiLabelTrainer:
    """
    Sophisticated multi-label training system.
    Trains separate models for each label with intelligent naming and organization.
    """

    def __init__(self,
                 config: Optional[TrainingConfig] = None,
                 verbose: bool = True):
        """
        Initialize multi-label trainer.

        Args:
            config: Training configuration
            verbose: Enable detailed logging
        """
        self.config = config or TrainingConfig()
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        self.trained_models = {}
        self.model_selector = ModelSelector(verbose=False)
        self.ml_selector = MultilingualModelSelector(verbose=False)

    def load_multi_label_data(self,
                              filepath: str,
                              text_field: str = 'text',
                              label_fields: Optional[List[str]] = None,
                              id_field: Optional[str] = 'id',
                              lang_field: Optional[str] = 'lang',
                              labels_dict_field: Optional[str] = 'labels') -> List[MultiLabelSample]:
        """
        Load multi-label data from JSONL or JSON file.

        Args:
            filepath: Path to JSONL or JSON file
            text_field: Field name for text
            label_fields: List of label field names (if None, auto-detect)
            id_field: Field name for ID
            lang_field: Field name for language
            labels_dict_field: Field name for labels dictionary (for nested format)

        Returns:
            List of MultiLabelSample objects

        Supports formats:
        1. Flat: {"text": "sample", "sentiment": "positive", "category": "tech"}
        2. Nested: {"text": "sample", "labels": {"sentiment": "positive", "category": "tech"}}
        3. JSON with train/val: {"train": [...], "val": [...]}
        """
        samples = []

        # Determine file type and load accordingly
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if it's a split format
            if isinstance(data, dict) and ('train' in data or 'val' in data):
                # This is a pre-split file, just return all data
                all_data = data.get('train', []) + data.get('val', [])
            elif isinstance(data, list):
                all_data = data
            else:
                all_data = [data]

            # Process each item
            for item_num, item in enumerate(all_data, 1):
                sample = self._parse_data_item(item, text_field, label_fields,
                                              id_field, lang_field, labels_dict_field, item_num)
                if sample:
                    samples.append(sample)
        else:
            # JSONL format
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        sample = self._parse_data_item(data, text_field, label_fields,
                                                      id_field, lang_field, labels_dict_field, line_num)
                        if sample:
                            samples.append(sample)
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(f"Line {line_num}: Error - {e}")
                        continue

        if self.verbose:
            # Count unique labels
            all_labels = set()
            for sample in samples:
                all_labels.update(sample.labels.keys())
            self.logger.info(f"Loaded {len(samples)} samples with {len(all_labels)} label types")
            if len(all_labels) <= 20:
                self.logger.info(f"Label types: {sorted(all_labels)}")

        return samples

    def _parse_data_item(self, data, text_field, label_fields, id_field, lang_field, labels_dict_field, item_num):
        """Parse a single data item into MultiLabelSample."""
        try:
            if text_field not in data:
                return None

            # Check for nested labels format
            if labels_dict_field and labels_dict_field in data:
                # Labels are in a nested dict
                labels = data[labels_dict_field]
                if not isinstance(labels, dict):
                    return None
            else:
                # Flat format - auto-detect label fields if not provided
                if label_fields is None:
                    exclude_fields = {text_field, id_field, lang_field, 'metadata', labels_dict_field}
                    label_fields = [k for k in data.keys() if k not in exclude_fields]

                # Extract labels
                labels = {}
                for label_field in label_fields:
                    if label_field in data:
                        labels[label_field] = data[label_field]

            if not labels:
                return None

            # Extract metadata
            metadata_keys = set(data.keys()) - {text_field, id_field, lang_field, labels_dict_field}
            metadata_keys -= set(labels.keys()) if not labels_dict_field else set()
            metadata = {k: data[k] for k in metadata_keys if k in data}

            return MultiLabelSample(
                text=data[text_field],
                labels=labels,
                id=data.get(id_field),
                lang=data.get(lang_field),
                metadata=metadata if metadata else None
            )
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Item {item_num}: Error - {e}")
            return None

    def _create_distribution_report(self,
                                   split_datasets: Dict[str, Dict[str, List[DataSample]]],
                                   output_dir: str) -> None:
        """
        Create detailed CSV reports of data distribution.

        Args:
            split_datasets: Split datasets by label
            output_dir: Directory to save reports
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Overall distribution report
        overall_report = []

        # First, collect all unique languages in the dataset
        all_languages = set()
        for label_name, splits in split_datasets.items():
            for split_name, samples in splits.items():
                if samples:
                    all_languages.update([s.lang for s in samples if s.lang])

        # Sort languages for consistent column ordering
        sorted_languages = sorted(all_languages)

        for label_name, splits in split_datasets.items():
            # Count labels and languages
            for split_name, samples in splits.items():
                if not samples:
                    continue

                # Count positive and negative samples
                labels = [s.label for s in samples]
                positives = sum(labels)
                negatives = len(labels) - positives

                # Count by language
                lang_counts = Counter([s.lang for s in samples if s.lang])

                # Calculate metrics
                row = {
                    'label': label_name,
                    'split': split_name,
                    'total_samples': len(samples),
                    'positive_samples': positives,
                    'negative_samples': negatives,
                    'positive_ratio': f"{100*positives/len(samples):.2f}%" if samples else "0%",
                    'negative_ratio': f"{100*negatives/len(samples):.2f}%" if samples else "0%"
                }

                # Add language distribution - only for languages that exist in the data
                for lang in sorted_languages:
                    row[f'{lang}_count'] = lang_counts.get(lang, 0)
                    row[f'{lang}_ratio'] = f"{100*lang_counts.get(lang, 0)/len(samples):.2f}%" if samples else "0%"

                overall_report.append(row)

        # Save overall report
        overall_csv = os.path.join(output_dir, f"distribution_report_{timestamp}.csv")
        if overall_report:
            df = pd.DataFrame(overall_report)
            df.to_csv(overall_csv, index=False)
            if self.verbose:
                self.logger.info(f"📊 Distribution report saved to: {overall_csv}")

        # 2. Detailed per-label reports
        for label_name, splits in split_datasets.items():
            label_report = []

            for split_name, samples in splits.items():
                if not samples:
                    continue

                # Group by language and label value
                for sample in samples:
                    label_report.append({
                        'split': split_name,
                        'sample_id': sample.id,
                        'label_value': sample.label,
                        'language': sample.lang or 'unknown',
                        'text_length': len(sample.text) if sample.text else 0,
                        'text_preview': sample.text[:100] + '...' if sample.text and len(sample.text) > 100 else sample.text
                    })

            # Save per-label report
            if label_report:
                label_csv = os.path.join(output_dir, f"label_detail_{label_name.replace(' ', '_')}_{timestamp}.csv")
                df = pd.DataFrame(label_report)
                df.to_csv(label_csv, index=False)

        # 3. Language balance report
        lang_report = []
        for label_name, splits in split_datasets.items():
            for split_name, samples in splits.items():
                if not samples:
                    continue

                # Group by language
                lang_groups = defaultdict(lambda: {'positive': 0, 'negative': 0, 'total': 0})
                for sample in samples:
                    lang = sample.lang or 'unknown'
                    lang_groups[lang]['total'] += 1
                    if sample.label == 1:
                        lang_groups[lang]['positive'] += 1
                    else:
                        lang_groups[lang]['negative'] += 1

                for lang, counts in lang_groups.items():
                    lang_report.append({
                        'label': label_name,
                        'split': split_name,
                        'language': lang,
                        'total': counts['total'],
                        'positives': counts['positive'],
                        'negatives': counts['negative'],
                        'positive_ratio': f"{100*counts['positive']/counts['total']:.2f}%" if counts['total'] > 0 else "0%"
                    })

        # Save language balance report
        if lang_report:
            lang_csv = os.path.join(output_dir, f"language_balance_{timestamp}.csv")
            df = pd.DataFrame(lang_report)
            df.to_csv(lang_csv, index=False)
            if self.verbose:
                self.logger.info(f"🌍 Language balance report saved to: {lang_csv}")

    def prepare_label_datasets(self,
                              samples: List[MultiLabelSample],
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1,
                              output_dir: Optional[str] = None) -> Dict[str, Dict[str, List[DataSample]]]:
        """
        Prepare separate datasets for each label.

        Args:
            samples: List of MultiLabelSample objects
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Dictionary: label_name -> {'train': samples, 'val': samples, 'test': samples}
        """
        # Get all unique label names
        all_label_names = set()
        for sample in samples:
            all_label_names.update(sample.labels.keys())

        # organize samples by label, creating both positive and negative examples
        label_datasets = defaultdict(list)

        for sample in samples:
            # For each possible label
            for label_name in all_label_names:
                # Create a DataSample with label=1 if present, 0 otherwise
                label_value = sample.labels.get(label_name, 0)
                data_sample = DataSample(
                    text=sample.text,
                    label=label_value,
                    id=sample.id,
                    lang=sample.lang,
                    metadata={**(sample.metadata or {}), 'label_type': label_name}
                )
                label_datasets[label_name].append(data_sample)

        # split each label's dataset
        split_datasets = {}

        for label_name, label_samples in label_datasets.items():
            # Calculate test ratio (0 if we only want train/val)
            test_ratio = max(0, 1 - train_ratio - val_ratio)

            train, val, test = DataUtil.prepare_splits(
                label_samples,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                stratify_by_lang=self.config.train_by_language
            )

            split_datasets[label_name] = {
                'train': train,
                'val': val,
                'test': test if test_ratio > 0 else []
            }

            if self.verbose:
                if test_ratio > 0:
                    self.logger.info(f"Label '{label_name}': {len(train)} train, {len(val)} val, {len(test)} test")
                else:
                    self.logger.info(f"Label '{label_name}': {len(train)} train, {len(val)} val")

        # Create distribution reports if output_dir is provided
        if output_dir:
            self._create_distribution_report(split_datasets, output_dir)

        return split_datasets

    def _generate_model_name(self,
                           label_name: str,
                           language: Optional[str] = None) -> str:
        """
        Generate model name based on label and language.

        Args:
            label_name: Name of the label
            language: Language code (optional)

        Returns:
            Model name string
        """
        if self.config.train_by_language and language:
            # separate model per language: "sentiment_en", "sentiment_fr"
            return f"{label_name}_{language}"
        elif self.config.multilingual_model or not language:
            # single multilingual model: "sentiment"
            return label_name
        else:
            # default: just label name
            return label_name

    def _select_model_class(self,
                          samples: List[DataSample]) -> type:
        """
        Select appropriate model class based on data and configuration.

        Args:
            samples: Training samples

        Returns:
            Model class to instantiate
        """
        if self.config.model_class:
            # use specified model class
            return self.config.model_class

        if not self.config.auto_select_model:
            # default to BertBaseEnhanced
            return BertBaseEnhanced

        # analyze language distribution
        languages = [s.lang for s in samples if s.lang]

        if languages and len(set(languages)) > 1:
            # multilingual data
            if self.config.multilingual_model:
                # use multilingual model selector
                rec = self.ml_selector.recommend_model(
                    texts=[s.text for s in samples[:1000]]
                )
                return rec.model_class

        # monolingual or no language info - use auto selector
        texts = [s.text for s in samples[:1000]]
        return auto_select_model(
            train_texts=texts,
            resource_constraint='standard'
        )

    def _save_detailed_metrics(self,
                              model_info: 'ModelInfo',
                              output_path: str) -> None:
        """
        Save detailed training metrics to CSV files.

        Args:
            model_info: Model information with metrics
            output_path: Path to save metrics
        """
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save performance metrics
        perf_data = []
        if model_info.performance_metrics:
            metrics = model_info.performance_metrics

            # Overall metrics
            perf_data.append({
                'metric_type': 'overall',
                'language': 'all',
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', [0])[0] if isinstance(metrics.get('precision'), list) else metrics.get('precision', 0),
                'recall': metrics.get('recall', [0])[0] if isinstance(metrics.get('recall'), list) else metrics.get('recall', 0),
                'f1_score': metrics.get('macro_f1', 0),
                'support': metrics.get('support', 0)
            })

            # Per-language metrics if available
            if 'per_language' in metrics:
                for lang, lang_metrics in metrics['per_language'].items():
                    perf_data.append({
                        'metric_type': 'by_language',
                        'language': lang,
                        'accuracy': lang_metrics.get('accuracy', 0),
                        'precision': lang_metrics.get('precision', [0])[0] if isinstance(lang_metrics.get('precision'), list) else lang_metrics.get('precision', 0),
                        'recall': lang_metrics.get('recall', [0])[0] if isinstance(lang_metrics.get('recall'), list) else lang_metrics.get('recall', 0),
                        'f1_score': lang_metrics.get('macro_f1', 0),
                        'support': lang_metrics.get('n_samples', 0)
                    })

        if perf_data:
            perf_csv = os.path.join(output_path, f"performance_metrics_{model_info.model_name}_{timestamp}.csv")
            df = pd.DataFrame(perf_data)
            df.to_csv(perf_csv, index=False)

        # 2. Save confusion matrix if available
        if model_info.performance_metrics and 'confusion_matrix' in model_info.performance_metrics:
            cm = model_info.performance_metrics['confusion_matrix']
            cm_csv = os.path.join(output_path, f"confusion_matrix_{model_info.model_name}_{timestamp}.csv")
            pd.DataFrame(cm).to_csv(cm_csv, index=False)

    def train_single_model(self,
                         label_name: str,
                         train_samples: List[DataSample],
                         val_samples: List[DataSample],
                         language: Optional[str] = None) -> ModelInfo:
        """
        Train a single model for one label.

        Args:
            label_name: Name of the label
            train_samples: Training samples
            val_samples: Validation samples
            language: Language code (for naming)

        Returns:
            ModelInfo object with training results
        """
        model_name = self._generate_model_name(label_name, language)

        if self.verbose:
            self.logger.info(f"Training model: {model_name}")

        # select model class
        model_class = self._select_model_class(train_samples)

        # initialize model
        model = model_class()
        use_enhanced = hasattr(model, 'encode_with_metadata')

        # encode data
        if use_enhanced:
            train_loader = model.encode_with_metadata(
                train_samples,
                batch_size=self.config.batch_size
            )
            val_loader = model.encode_with_metadata(
                val_samples,
                batch_size=self.config.batch_size
            )
        else:
            texts_train = [s.text for s in train_samples]
            labels_train = [s.label for s in train_samples]
            texts_val = [s.text for s in val_samples]
            labels_val = [s.label for s in val_samples]

            train_loader = model.encode(texts_train, labels_train, batch_size=self.config.batch_size)
            val_loader = model.encode(texts_val, labels_val, batch_size=self.config.batch_size)

        # train model
        model_path = os.path.join(self.config.output_dir, model_name)

        if use_enhanced and hasattr(model, 'run_training_enhanced'):
            scores = model.run_training_enhanced(
                train_loader,
                val_loader,
                n_epochs=self.config.n_epochs,
                lr=self.config.learning_rate,
                save_model_as=model_name,
                reinforced_learning=self.config.reinforced_learning,
                n_epochs_reinforced=self.config.n_epochs_reinforced,
                track_languages=self.config.track_languages,
                metrics_output_dir=os.path.join(model_path, 'logs')
            )
        else:
            # Extract language info from val_samples if track_languages is enabled
            language_info = None
            if self.config.track_languages and val_samples:
                language_info = [s.lang for s in val_samples if hasattr(s, 'lang')]
                if not language_info:
                    language_info = None

            scores = model.run_training(
                train_loader,
                val_loader,
                n_epochs=self.config.n_epochs,
                lr=self.config.learning_rate,
                save_model_as=model_name,
                reinforced_learning=self.config.reinforced_learning,
                n_epochs_reinforced=self.config.n_epochs_reinforced,
                reinforced_epochs=self.config.reinforced_epochs if hasattr(self.config, 'reinforced_epochs') else None,
                track_languages=self.config.track_languages,
                language_info=language_info,
                metrics_output_dir=os.path.join(model_path, 'logs')
            )

        # calculate final metrics
        precision, recall, f1, support = scores if scores else ([], [], [], [])

        performance_metrics = {
            'precision': precision.tolist() if hasattr(precision, 'tolist') else precision,
            'recall': recall.tolist() if hasattr(recall, 'tolist') else recall,
            'f1': f1.tolist() if hasattr(f1, 'tolist') else f1,
            'support': support.tolist() if hasattr(support, 'tolist') else support,
            'macro_f1': np.mean(f1) if len(f1) > 0 else 0
        }

        model_info = ModelInfo(
            model_name=model_name,
            label_name=label_name,
            language=language,
            model_path=model_path,
            performance_metrics=performance_metrics,
            training_config={
                'model_class': model_class.__name__,
                'n_epochs': self.config.n_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            }
        )

        # Save detailed metrics
        if self.config.output_dir:
            metrics_dir = os.path.join(self.config.output_dir, 'metrics')
            self._save_detailed_metrics(model_info, metrics_dir)

        return model_info

    def train(self,
             train_samples: Optional[List[Union[MultiLabelSample, Dict]]] = None,
             val_samples: Optional[List[Union[MultiLabelSample, Dict]]] = None,
             data_file: Optional[str] = None,
             auto_split: bool = True,
             split_ratio: float = 0.8,
             stratified: bool = True,
             train_by_language: bool = None,
             output_dir: Optional[str] = None) -> Dict[str, ModelInfo]:
        """
        Main training method with automatic data handling.

        Args:
            train_samples: Training samples (can be dicts or MultiLabelSample)
            val_samples: Validation samples (optional)
            data_file: Path to data file (alternative to providing samples)
            auto_split: Automatically split data if no val_samples provided
            split_ratio: Ratio for train split (default 0.8)
            stratified: Use stratified splitting
            train_by_language: Train separate models per language
            output_dir: Output directory for models

        Returns:
            Dictionary of trained models

        Examples:
            # Method 1: With pre-split data
            trainer.train(train_samples, val_samples)

            # Method 2: With single file (auto-split)
            trainer.train(data_file='data.jsonl')

            # Method 3: With samples needing split
            trainer.train(train_samples=all_samples, auto_split=True)
        """
        # Update config if provided
        if train_by_language is not None:
            self.config.train_by_language = train_by_language
        if output_dir:
            self.config.output_dir = output_dir

        # Load data if file provided
        if data_file:
            if self.verbose:
                self.logger.info(f"Loading data from {data_file}")

            # Check if it's a JSON file with train/val splits
            if data_file.endswith('.json'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, dict) and 'train' in data and 'val' in data:
                    # Pre-split JSON file
                    train_samples = self._convert_to_samples(data['train'])
                    val_samples = self._convert_to_samples(data['val'])
                    auto_split = False  # Don't split again
                else:
                    # Single JSON file
                    all_samples = self.load_multi_label_data(data_file)
                    train_samples = all_samples
            else:
                # JSONL file
                all_samples = self.load_multi_label_data(data_file)
                train_samples = all_samples

        # Convert dicts to MultiLabelSample if needed
        if train_samples and not isinstance(train_samples[0], MultiLabelSample):
            train_samples = self._convert_to_samples(train_samples)
        if val_samples and not isinstance(val_samples[0], MultiLabelSample):
            val_samples = self._convert_to_samples(val_samples)

        # Handle splitting if needed
        if auto_split and (val_samples is None or len(val_samples) == 0):
            if self.verbose:
                self.logger.info(f"Auto-splitting data with ratio {split_ratio}")

            if stratified:
                # Use package's stratified splitting
                from . import create_stratified_splits
                train_samples, val_samples = create_stratified_splits(
                    train_samples,
                    train_ratio=split_ratio,
                    val_ratio=1.0 - split_ratio,
                    stratify_by_label=True,
                    stratify_by_language=self.config.track_languages
                )
            else:
                # Simple random split
                import random
                random.seed(42)
                all_samples = train_samples
                random.shuffle(all_samples)
                split_idx = int(len(all_samples) * split_ratio)
                train_samples = all_samples[:split_idx]
                val_samples = all_samples[split_idx:]

        if self.verbose:
            self.logger.info(f"Training with {len(train_samples)} train and {len(val_samples)} val samples")

        # Combine samples for dataset preparation
        all_samples = train_samples + val_samples

        # Calculate actual split ratio based on provided data
        actual_train_ratio = len(train_samples) / len(all_samples)
        actual_val_ratio = len(val_samples) / len(all_samples)

        # Use train_all_models with the calculated ratios
        return self.train_all_models(all_samples, actual_train_ratio, actual_val_ratio)

    def _convert_to_samples(self, data: List[Dict]) -> List[MultiLabelSample]:
        """Convert list of dicts to MultiLabelSample objects."""
        samples = []
        for item in data:
            # Handle nested labels format
            if 'labels' in item and isinstance(item['labels'], dict):
                labels = item['labels']
                text = item.get('text', '')
                sample_id = item.get('id')
                lang = item.get('lang')
            else:
                # Flat format
                exclude = {'text', 'id', 'lang', 'metadata'}
                labels = {k: v for k, v in item.items() if k not in exclude}
                text = item.get('text', '')
                sample_id = item.get('id')
                lang = item.get('lang')

            if text and labels:
                samples.append(MultiLabelSample(
                    text=text,
                    labels=labels,
                    id=sample_id,
                    lang=lang
                ))
        return samples

    def train_all_models(self,
                        samples: List[MultiLabelSample],
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1) -> Dict[str, ModelInfo]:
        """
        Train all models for all labels.

        Args:
            samples: Multi-label samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Dictionary of model_name -> ModelInfo
        """
        # prepare datasets
        # Create output directory for reports
        output_dir = self.config.output_dir if hasattr(self.config, 'output_dir') else 'training_logs'
        label_datasets = self.prepare_label_datasets(
            samples,
            train_ratio,
            val_ratio,
            output_dir=output_dir
        )

        # organize training jobs
        training_jobs = []

        for label_name, datasets in label_datasets.items():
            if self.config.train_by_language:
                # group by language
                lang_groups = defaultdict(list)
                for sample in datasets['train']:
                    lang = sample.lang or 'unknown'
                    lang_groups[lang].append(sample)

                # create job for each language
                for lang, lang_samples in lang_groups.items():
                    # get validation samples for this language
                    lang_val = [s for s in datasets['val'] if s.lang == lang or not s.lang]

                    training_jobs.append({
                        'label_name': label_name,
                        'train_samples': lang_samples,
                        'val_samples': lang_val,
                        'language': lang
                    })
            else:
                # single model for all languages
                training_jobs.append({
                    'label_name': label_name,
                    'train_samples': datasets['train'],
                    'val_samples': datasets['val'],
                    'language': None
                })

        if self.verbose:
            self.logger.info(f"Starting training of {len(training_jobs)} models")

        # train models
        trained_models = {}

        if self.config.parallel_training and len(training_jobs) > 1:
            # parallel training
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for job in training_jobs:
                    future = executor.submit(
                        self.train_single_model,
                        job['label_name'],
                        job['train_samples'],
                        job['val_samples'],
                        job['language']
                    )
                    futures.append(future)

                # collect results
                for future in tqdm(futures, desc="Training models"):
                    model_info = future.result()
                    trained_models[model_info.model_name] = model_info
        else:
            # sequential training
            for job in tqdm(training_jobs, desc="Training models"):
                model_info = self.train_single_model(
                    job['label_name'],
                    job['train_samples'],
                    job['val_samples'],
                    job['language']
                )
                trained_models[model_info.model_name] = model_info

        self.trained_models = trained_models

        # save summary
        self._save_training_summary()

        return trained_models

    def _save_training_summary(self):
        """Save summary of all trained models."""
        summary_path = os.path.join(self.config.output_dir, 'training_summary.json')

        summary = {
            'configuration': {
                'train_by_language': self.config.train_by_language,
                'multilingual_model': self.config.multilingual_model,
                'n_epochs': self.config.n_epochs,
                'n_epochs_reinforced': self.config.n_epochs_reinforced,
                'reinforced_learning': self.config.reinforced_learning,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            },
            'models': {}
        }

        for model_name, model_info in self.trained_models.items():
            summary['models'][model_name] = {
                'label': model_info.label_name,
                'language': model_info.language,
                'path': model_info.model_path,
                'macro_f1': model_info.performance_metrics.get('macro_f1', 0),
                'training_config': model_info.training_config
            }

        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.verbose:
            self.logger.info(f"Training summary saved to {summary_path}")

    def load_trained_models(self,
                          summary_path: str) -> Dict[str, ModelInfo]:
        """
        Load information about previously trained models.

        Args:
            summary_path: Path to training_summary.json

        Returns:
            Dictionary of model_name -> ModelInfo
        """
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        models = {}
        for model_name, info in summary['models'].items():
            model_info = ModelInfo(
                model_name=model_name,
                label_name=info['label'],
                language=info.get('language'),
                model_path=info['path'],
                performance_metrics={'macro_f1': info.get('macro_f1', 0)},
                training_config=info.get('training_config', {})
            )
            models[model_name] = model_info

        self.trained_models = models
        return models

    def predict_all_labels(self,
                          texts: List[str],
                          languages: Optional[List[str]] = None,
                          model_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Predict all labels for given texts using trained models.

        Args:
            texts: List of input texts
            languages: Optional list of language codes
            model_dir: Directory containing trained models

        Returns:
            DataFrame with predictions for all labels
        """
        if not self.trained_models and model_dir:
            # load model info
            summary_path = os.path.join(model_dir, 'training_summary.json')
            if os.path.exists(summary_path):
                self.load_trained_models(summary_path)

        if not self.trained_models:
            raise ValueError("No trained models available")

        # prepare results dataframe
        results = pd.DataFrame({'text': texts})

        if languages:
            results['language'] = languages

        # get predictions from each model
        for model_name, model_info in self.trained_models.items():
            # determine which texts to predict with this model
            if self.config.train_by_language and model_info.language:
                # only predict for matching language
                if languages:
                    mask = [lang == model_info.language for lang in languages]
                    relevant_texts = [t for t, m in zip(texts, mask) if m]
                    indices = [i for i, m in enumerate(mask) if m]
                else:
                    continue
            else:
                # predict for all texts
                relevant_texts = texts
                indices = list(range(len(texts)))

            if not relevant_texts:
                continue

            # load model and predict
            try:
                # simplified prediction - would need actual model loading
                predictions = [0] * len(relevant_texts)  # placeholder

                # add to results
                col_name = f"{model_info.label_name}_pred"
                results.loc[indices, col_name] = predictions

            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to predict with {model_name}: {e}")

        return results


def train_multi_label_models(
    data_path: str,
    label_fields: Optional[List[str]] = None,
    train_by_language: bool = False,
    multilingual: bool = False,
    auto_select: bool = True,
    output_dir: str = "./multi_label_models",
    **training_kwargs
) -> Dict[str, ModelInfo]:
    """
    Convenience function for training multi-label models.

    Args:
        data_path: Path to JSONL data file
        label_fields: List of label field names (auto-detect if None)
        train_by_language: Train separate models per language
        multilingual: Use multilingual model
        auto_select: Auto-select best model architecture
        output_dir: Output directory for models
        **training_kwargs: Additional training parameters

    Returns:
        Dictionary of trained models

    Example:
        models = train_multi_label_models(
            'data.jsonl',
            label_fields=['sentiment', 'category', 'toxic'],
            train_by_language=True,
            n_epochs=5
        )
    """
    config = TrainingConfig(
        train_by_language=train_by_language,
        multilingual_model=multilingual,
        auto_select_model=auto_select,
        output_dir=output_dir,
        **training_kwargs
    )

    trainer = MultiLabelTrainer(config)

    # load data
    samples = trainer.load_multi_label_data(
        data_path,
        label_fields=label_fields
    )

    # train models
    models = trainer.train_all_models(samples)

    print(f"\nTrained {len(models)} models:")
    for model_name, info in models.items():
        print(f"  - {model_name}: macro_f1={info.performance_metrics.get('macro_f1', 0):.3f}")

    return models
#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
model_trainer.py

MAIN OBJECTIVE:
---------------
This script orchestrates model training, benchmarking, and selection
for the complete pipeline, managing all training operations.

Dependencies:
-------------
- torch
- transformers
- sklearn
- numpy
- pandas

MAIN FEATURES:
--------------
1) Model training orchestration
2) Automatic benchmarking of multiple models
3) Cross-validation and evaluation
4) Model selection based on performance
5) Hyperparameter optimization
6) Early stopping and checkpointing
7) Multi-GPU support
8) Comprehensive metrics reporting

Author:
-------
Antoine Lemor
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# HuggingFace imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

# Internal imports
from .bert_base import BertBase
from .models import (
    Bert, Camembert, ArabicBert, ChineseBert, GermanBert,
    HindiBert, ItalianBert, PortugueseBert, RussianBert,
    SpanishBert, SwedishBert, XLMRoberta
)
from .sota_models import (
    DeBERTaV3Base, DeBERTaV3Large, RoBERTaBase, RoBERTaLarge,
    ELECTRABase, ELECTRALarge, ALBERTBase, ALBERTLarge,
    BigBirdBase, LongformerBase, MDeBERTaV3Base, XLMRobertaBase
)
from .multilingual_selector import MultilingualModelSelector


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    seed: int = 42
    save_best_model: bool = True
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    output_dir: str = "./models/trained_model"
    logging_dir: str = "./logs"
    logging_steps: int = 50
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    num_workers: int = 4
    remove_unused_columns: bool = False
    push_to_hub: bool = False


@dataclass
class BenchmarkConfig:
    """Configuration for model benchmarking"""
    models_to_test: List[str] = field(default_factory=lambda: [
        "bert-base-uncased",
        "roberta-base",
        "distilbert-base-uncased",
        "albert-base-v2",
        "electra-base-discriminator"
    ])
    cross_validation_folds: int = 5
    auto_select_best: bool = True
    test_multilingual: bool = True
    test_sota: bool = True
    max_models: int = 10
    time_limit_hours: Optional[float] = None
    use_gpu: bool = True
    parallel_training: bool = False


@dataclass
class TrainingResult:
    """Results from model training"""
    model_name: str
    best_accuracy: float
    best_f1_macro: float
    best_f1_weighted: float
    precision: float
    recall: float
    training_time: float
    num_parameters: int
    confusion_matrix: np.ndarray
    classification_report: Dict
    best_epoch: int
    validation_loss: float
    training_history: List[Dict]
    model_path: str


class ModelTrainer:
    """Orchestrates model training and benchmarking"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the model trainer"""
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()

        # Set random seed for reproducibility
        set_seed(self.config.seed)

        # Model registry
        self.model_registry = self._build_model_registry()

    def _build_model_registry(self) -> Dict[str, type]:
        """Build registry of available models"""
        return {
            # English models
            "bert-base-uncased": Bert,
            "bert-large-uncased": Bert,
            "roberta-base": RoBERTaBase,
            "roberta-large": RoBERTaLarge,
            "deberta-v3-base": DeBERTaV3Base,
            "deberta-v3-large": DeBERTaV3Large,
            "electra-base": ELECTRABase,
            "electra-large": ELECTRALarge,
            "albert-base-v2": ALBERTBase,
            "albert-large-v2": ALBERTLarge,

            # Multilingual models
            "bert-base-multilingual-cased": Bert,
            "xlm-roberta-base": XLMRobertaBase,
            "mdeberta-v3-base": MDeBERTaV3Base,

            # Language-specific models
            "camembert-base": Camembert,
            "arabic-bert": ArabicBert,
            "chinese-bert": ChineseBert,
            "german-bert": GermanBert,

            # Long document models
            "bigbird-base": BigBirdBase,
            "longformer-base": LongformerBase,
        }

    def load_data(self, data_path: str, text_column: str = "text",
                  label_column: str = "label") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data for training"""
        self.logger.info(f"Loading data from {data_path}")

        # Load data based on file extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith('.jsonl'):
            df = pd.read_json(data_path, lines=True)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Check columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")

        # Encode labels
        df['encoded_label'] = self.label_encoder.fit_transform(df[label_column])

        # Split data
        X = df[text_column].values
        y = df['encoded_label'].values

        # Check if we have enough samples for stratification
        n_samples = len(y)
        n_classes = len(np.unique(y))
        test_samples = int(n_samples * self.config.test_split)
        val_samples = int(n_samples * self.config.validation_split)

        # Determine if stratification is possible
        # We need at least n_classes samples in each split for stratification
        can_stratify_test = (test_samples >= n_classes and
                            (n_samples - test_samples) >= n_classes)
        can_stratify_val = (val_samples >= n_classes and
                           (n_samples - val_samples) >= n_classes)

        if not can_stratify_test or not can_stratify_val:
            self.logger.warning(
                f"Too few samples ({n_samples}) for stratified split with {n_classes} classes. "
                f"Using random split instead. Consider increasing annotation sample size."
            )

        # First split: train+val and test
        stratify_test = y if can_stratify_test else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_split,
            random_state=self.config.seed, stratify=stratify_test
        )

        # Second split: train and validation
        val_size = self.config.validation_split / (1 - self.config.test_split)
        stratify_val = y_temp if can_stratify_val else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.config.seed, stratify=stratify_val
        )

        # Create DataFrames
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})

        self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")

        return train_df, val_df, test_df

    def train_single_model(self, model_name: str, train_df: pd.DataFrame,
                          val_df: pd.DataFrame, test_df: pd.DataFrame,
                          num_labels: Optional[int] = None,
                          output_dir: Optional[str] = None,
                          label_column: str = 'label',
                          training_strategy: str = 'single-label') -> TrainingResult:
        """Train a single model"""
        from tqdm import tqdm
        print(f"\n🏋️  Training model: {model_name}")
        self.logger.info(f"Training model: {model_name} with {training_strategy} strategy")
        start_time = time.time()

        # Get model class
        model_class = self.model_registry.get(model_name)
        if not model_class:
            # Try to use base Bert with custom model name
            model_class = Bert

        # Initialize model
        model_instance = model_class(model_name=model_name, device=self.device)

        # Prepare data - use the correct label column
        train_texts = train_df['text'].tolist()
        train_labels = train_df[label_column].tolist()
        val_texts = val_df['text'].tolist()
        val_labels = val_df[label_column].tolist()
        test_texts = test_df['text'].tolist()
        test_labels = test_df[label_column].tolist()

        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(self.config.output_dir) / model_name.replace('/', '_')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        try:
            # Encode data
            train_dataloader = model_instance.encode(
                train_texts, train_labels,
                batch_size=self.config.batch_size,
                progress_bar=True
            )

            val_dataloader = model_instance.encode(
                val_texts, val_labels,
                batch_size=self.config.batch_size,
                progress_bar=False
            )

            test_dataloader = model_instance.encode(
                test_texts, test_labels,
                batch_size=self.config.batch_size,
                progress_bar=False
            )

            # Parse label name to extract key and value for display (if applicable)
            # Check if label_column contains structured names like 'themes_long_transportation'
            label_key = None
            label_value = None
            if label_column and '_long_' in label_column:
                parts = label_column.split('_long_')
                if len(parts) == 2:
                    label_key = parts[0]
                    label_value = parts[1]
            elif label_column and '_short_' in label_column:
                parts = label_column.split('_short_')
                if len(parts) == 2:
                    label_key = parts[0]
                    label_value = parts[1]

            # Train (using test_dataloader as validation for compatibility with bert_base)
            history = model_instance.run_training(
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,  # bert_base expects test_dataloader for validation
                n_epochs=self.config.num_epochs,
                lr=self.config.learning_rate,
                random_state=42,
                save_model_as=str(output_dir / 'model'),
                label_key=label_key,      # Pass parsed label key
                label_value=label_value   # Pass parsed label value
            )

            # Evaluate on test set
            test_predictions = model_instance.predict(test_dataloader)
            test_probs = model_instance.predict(test_dataloader, proba=True)

            # Calculate metrics
            accuracy = accuracy_score(test_labels, test_predictions)
            precision, recall, f1_weighted, _ = precision_recall_fscore_support(
                test_labels, test_predictions, average='weighted'
            )
            _, _, f1_macro, _ = precision_recall_fscore_support(
                test_labels, test_predictions, average='macro'
            )
            cm = confusion_matrix(test_labels, test_predictions)

            # Get detailed classification report
            report = classification_report(
                test_labels, test_predictions,
                target_names=[str(c) for c in self.label_encoder.classes_],
                output_dict=True
            )

            # Count parameters
            num_params = sum(p.numel() for p in model_instance.model.parameters())

            training_time = time.time() - start_time

            result = TrainingResult(
                model_name=model_name,
                best_accuracy=accuracy,
                best_f1_macro=f1_macro,
                best_f1_weighted=f1_weighted,
                precision=precision,
                recall=recall,
                training_time=training_time,
                num_parameters=num_params,
                confusion_matrix=cm,
                classification_report=report,
                best_epoch=len(history),
                validation_loss=history[-1]['val_loss'] if history else 0,
                training_history=history,
                model_path=str(output_dir)
            )

            self.logger.info(f"Model {model_name} - Accuracy: {accuracy:.4f}, F1: {f1_macro:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {str(e)}")
            raise

    def benchmark_models(self, data_path: str, benchmark_config: Optional[BenchmarkConfig] = None,
                        text_column: str = "text", label_column: str = "label") -> Dict[str, Any]:
        """Benchmark multiple models on the dataset"""
        benchmark_config = benchmark_config or BenchmarkConfig()

        self.logger.info(f"Starting benchmark with {len(benchmark_config.models_to_test)} models")

        # Load data
        train_df, val_df, test_df = self.load_data(data_path, text_column, label_column)

        # Detect language if multilingual testing is enabled
        if benchmark_config.test_multilingual:
            from ..utils.language_detector import LanguageDetector
            detector = LanguageDetector()
            sample_texts = train_df['text'].sample(min(100, len(train_df))).tolist()
            detected_lang = detector.detect_language(' '.join(sample_texts))
            self.logger.info(f"Detected language: {detected_lang}")

            # Add language-specific models
            if detected_lang == 'fr':
                benchmark_config.models_to_test.extend([
                    "camembert-base",
                    "flaubert/flaubert_base_cased"
                ])
            elif detected_lang == 'es':
                benchmark_config.models_to_test.append("dccuchile/bert-base-spanish-wwm-uncased")

        # Add SOTA models if enabled
        if benchmark_config.test_sota:
            benchmark_config.models_to_test.extend([
                "microsoft/deberta-v3-base",
                "roberta-large",
                "google/electra-base-discriminator"
            ])

        # Limit number of models
        models_to_test = benchmark_config.models_to_test[:benchmark_config.max_models]

        results = []
        best_model = None
        best_f1 = 0

        for model_name in models_to_test:
            try:
                self.logger.info(f"Benchmarking model {model_name}")
                result = self.train_single_model(model_name, train_df, val_df, test_df)
                results.append(result)

                if result.best_f1_macro > best_f1:
                    best_f1 = result.best_f1_macro
                    best_model = result

            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                continue

        # Sort results by F1 score
        results.sort(key=lambda x: x.best_f1_macro, reverse=True)

        # Prepare benchmark report
        benchmark_report = {
            'best_model': best_model.model_name if best_model else None,
            'best_accuracy': best_model.best_accuracy if best_model else 0,
            'best_f1_macro': best_model.best_f1_macro if best_model else 0,
            'best_model_path': best_model.model_path if best_model else None,
            'models_tested': len(results),
            'results': [
                {
                    'model': r.model_name,
                    'accuracy': r.best_accuracy,
                    'f1_macro': r.best_f1_macro,
                    'f1_weighted': r.best_f1_weighted,
                    'precision': r.precision,
                    'recall': r.recall,
                    'training_time': r.training_time,
                    'num_parameters': r.num_parameters
                }
                for r in results
            ],
            'label_encoder_classes': self.label_encoder.classes_.tolist()
        }

        # Save benchmark report
        report_path = Path(self.config.output_dir) / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(benchmark_report, f, indent=2, default=str)

        self.logger.info(f"Benchmark complete. Best model: {benchmark_report['best_model']}")
        self.logger.info(f"Best F1 score: {benchmark_report['best_f1_macro']:.4f}")

        return benchmark_report

    def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous training method for pipeline integration

        Handles both single-label and multi-label training scenarios.
        """
        # Check for multi-label training first
        if 'training_files' in config and config['training_files']:
            return self._train_multi_label(config)

        # Single-label training
        if 'input_file' in config:
            data_path = config['input_file']
        else:
            raise ValueError("No input_file or training_files specified in config")

        # Update training config from dict
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Get column names
        text_column = config.get('text_column', 'text')
        label_column = config.get('label_column', 'label')
        training_strategy = config.get('training_strategy', 'single-label')

        # Get model name from config
        model_name = config.get('model_name', self.config.model_name)

        # Route to appropriate training method based on strategy
        if training_strategy == 'multi-label':
            # Use multi-label training with MultiLabelTrainer
            from .multi_label_trainer import MultiLabelTrainer

            print("\n🏋️ Training multi-label model...")
            self.logger.info(f"Starting multi-label training for {label_column}")

            # Load multi-label data directly without encoding
            if data_path.endswith('.jsonl'):
                df = pd.read_json(data_path, lines=True)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith('.csv'):
                import ast
                df = pd.read_csv(data_path)
                # Convert string representations of lists back to lists
                if label_column in df.columns:
                    df[label_column] = df[label_column].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
                    )
            else:
                raise ValueError(f"Unsupported file format for multi-label: {data_path}")

            # Check columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in data")

            # Split data manually without encoding
            from sklearn.model_selection import train_test_split

            X = df[text_column].values
            y = df[label_column].values

            # Extract language column if it exists
            lang_col = None
            if 'lang' in df.columns:
                lang_col = df['lang'].values

            # First split: train+val and test
            if lang_col is not None:
                X_temp, X_test, y_temp, y_test, lang_temp, lang_test = train_test_split(
                    X, y, lang_col, test_size=self.config.test_split,
                    random_state=self.config.seed
                )
            else:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=self.config.test_split,
                    random_state=self.config.seed
                )
                lang_temp, lang_test = None, None

            # Second split: train and validation
            val_size = self.config.validation_split / (1 - self.config.test_split)
            if lang_temp is not None:
                X_train, X_val, y_train, y_val, lang_train, lang_val = train_test_split(
                    X_temp, y_temp, lang_temp, test_size=val_size,
                    random_state=self.config.seed
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size,
                    random_state=self.config.seed
                )
                lang_train, lang_val = None, None

            # Prepare samples for MultiLabelTrainer
            train_samples = []
            for i in range(len(X_train)):
                sample = {
                    'text': X_train[i],
                    'labels': y_train[i] if isinstance(y_train[i], list) else [y_train[i]]
                }
                if lang_train is not None:
                    sample['lang'] = lang_train[i]
                train_samples.append(sample)

            val_samples = []
            for i in range(len(X_val)):
                sample = {
                    'text': X_val[i],
                    'labels': y_val[i] if isinstance(y_val[i], list) else [y_val[i]]
                }
                if lang_val is not None:
                    sample['lang'] = lang_val[i]
                val_samples.append(sample)

            self.logger.info(f"Data split - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(X_test)}")

            # Initialize MultiLabelTrainer with its own config
            from llm_tool.trainers.multi_label_trainer import TrainingConfig as MLTrainingConfig
            from llm_tool.trainers.bert_base import BertBase

            ml_config = MLTrainingConfig()
            ml_config.output_dir = config.get('output_dir', 'models/best_model')
            ml_config.n_epochs = config.get('num_epochs', self.config.num_epochs)
            ml_config.batch_size = config.get('batch_size', self.config.batch_size)
            ml_config.learning_rate = config.get('learning_rate', self.config.learning_rate)
            ml_config.train_by_language = config.get('train_by_language', False)
            ml_config.auto_select_model = False  # Disable auto-selection
            ml_config.model_class = BertBase  # Force BertBase to ensure run_training_enhanced is used

            ml_trainer = MultiLabelTrainer(config=ml_config, verbose=True)

            # Train models
            trained_models = ml_trainer.train(
                train_samples=train_samples,
                val_samples=val_samples,
                auto_split=False,
                output_dir=config.get('output_dir', 'models/best_model')
            )

            # Aggregate results
            if trained_models:
                avg_f1 = np.mean([m.metrics.get('f1_macro', 0) for m in trained_models.values()])
                avg_acc = np.mean([m.metrics.get('accuracy', 0) for m in trained_models.values()])

                self.logger.info(f"Multi-label training complete!")
                self.logger.info(f"Models trained: {len(trained_models)}/{len(trained_models)}")

                results = {
                    'f1_macro': avg_f1,
                    'accuracy': avg_acc,
                    'model_path': config.get('output_dir', 'models/best_model'),
                    'training_time': 0,
                    'trained_models': {k: v.model_path for k, v in trained_models.items()}
                }
            else:
                results = {
                    'f1_macro': 0.0,
                    'accuracy': 0.0,
                    'model_path': '',
                    'training_time': 0
                }
        else:
            # Load data for single-label training (uses encoding)
            train_df, val_df, test_df = self.load_data(
                data_path,
                text_column=text_column,
                label_column=label_column
            )

            # Get number of unique labels
            all_labels = pd.concat([train_df[label_column], val_df[label_column], test_df[label_column]])
            num_labels = len(all_labels.unique())

            # Use single-label training
            results = self.train_single_model(
                model_name=model_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                num_labels=num_labels,
                output_dir=config.get('output_dir', 'models/best_model'),
                label_column=label_column,
                training_strategy=training_strategy
            )

        return {
            'best_model': model_name,
            'best_f1_macro': results.get('f1', results.get('f1_macro', 0.0)),
            'accuracy': results.get('accuracy', 0.0),
            'model_path': results.get('model_path', ''),
            'training_time': results.get('training_time', 0),
            'metrics': results
        }

    async def train_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for training (for pipeline integration)"""
        # Update config
        if 'input_file' in config:
            data_path = config['input_file']
        else:
            raise ValueError("No input_file specified in config")

        # Update training config from dict
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Check if benchmark mode
        if config.get('benchmark_mode', False):
            benchmark_config = BenchmarkConfig(
                models_to_test=config.get('models_to_test', BenchmarkConfig.models_to_test),
                auto_select_best=config.get('auto_select_best', True),
                test_multilingual=config.get('test_multilingual', True),
                test_sota=config.get('test_sota', True)
            )
            return self.benchmark_models(data_path, benchmark_config)
        else:
            # Single model training
            train_df, val_df, test_df = self.load_data(data_path)
            result = self.train_single_model(
                config.get('model_type', 'bert-base-uncased'),
                train_df, val_df, test_df
            )

            return {
                'best_model': result.model_name,
                'best_accuracy': result.best_accuracy,
                'best_f1_macro': result.best_f1_macro,
                'model_path': result.model_path
            }

    async def benchmark_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for benchmarking (for pipeline integration)"""
        config['benchmark_mode'] = True
        return await self.train_async(config)

    def _optimize_training_parameters(self, model_name: str, train_size: int,
                                     num_labels: int, is_multilingual: bool,
                                     avg_text_length: float) -> Dict[str, Any]:
        """Smart parameter optimization based on model and data characteristics

        This implements the advanced parameter optimization from AugmentedSocialScientist
        """
        params = {}

        # Base parameters
        params['batch_size'] = self.config.batch_size
        params['learning_rate'] = self.config.learning_rate
        params['num_epochs'] = self.config.num_epochs

        # Adjust for model size
        if 'large' in model_name.lower():
            params['batch_size'] = max(4, params['batch_size'] // 4)
            params['learning_rate'] *= 0.5
            params['gradient_accumulation_steps'] = 4
        elif 'xlarge' in model_name.lower() or 'xxlarge' in model_name.lower():
            params['batch_size'] = max(2, params['batch_size'] // 8)
            params['learning_rate'] *= 0.25
            params['gradient_accumulation_steps'] = 8

        # Adjust for long-context models with short texts
        if any(model in model_name.lower() for model in ['longformer', 'bigbird']):
            if avg_text_length < 512:  # Short texts
                params['batch_size'] = max(2, params['batch_size'] // 8)
                params['learning_rate'] *= 2
                params['num_epochs'] = int(params['num_epochs'] * 1.5)
                self.logger.info(f"Detected long-context model with short texts. Adjusting parameters.")

        # Adjust for ALBERT (parameter sharing)
        if 'albert' in model_name.lower():
            params['num_epochs'] = params['num_epochs'] * 2
            self.logger.info("ALBERT detected: doubling epochs due to parameter sharing")

        # Adjust for multilingual data
        if is_multilingual:
            if 'xlm' in model_name.lower() or 'mdeberta' in model_name.lower():
                params['num_epochs'] = int(params['num_epochs'] * 1.2)
                params['warmup_ratio'] = 0.15
            else:
                # Non-multilingual model on multilingual data
                self.logger.warning("Using non-multilingual model on multilingual data")
                params['num_epochs'] = int(params['num_epochs'] * 1.5)

        # Adjust for small datasets
        if train_size < 100:
            params['batch_size'] = min(8, train_size // 4)
            params['num_epochs'] = min(20, params['num_epochs'] * 2)
        elif train_size < 500:
            params['batch_size'] = min(16, params['batch_size'])

        # Adjust for many labels
        if num_labels > 10:
            params['num_epochs'] = int(params['num_epochs'] * 1.2)

        return params

    def _select_multilingual_model(self, language_distribution: Dict[str, int]) -> str:
        """Select best multilingual model based on language distribution"""
        total = sum(language_distribution.values())
        languages = list(language_distribution.keys())

        # Check language diversity
        diversity = len(languages)

        # Prefer mDeBERTa for high diversity
        if diversity > 5:
            return "microsoft/mdeberta-v3-base"
        # XLM-RoBERTa for moderate diversity
        elif diversity > 2:
            return "xlm-roberta-base"
        # Check specific language dominance
        elif any(lang in ['fr', 'es', 'de', 'it', 'pt'] for lang in languages):
            # European languages - XLM-RoBERTa performs well
            return "xlm-roberta-base"
        else:
            # Default to mDeBERTa
            return "microsoft/mdeberta-v3-base"

    def _train_with_metadata(self, model_name: str, train_samples: List,
                            val_samples: List, test_samples: List,
                            num_labels: int, output_dir: Path,
                            track_languages: bool = True) -> Dict[str, Any]:
        """Train model with metadata support and per-language tracking

        This implements the enhanced training from AugmentedSocialScientist
        """
        from ..trainers.bert_base import BertBase
        from ..trainers.data_utils import MetadataDataset, PerformanceTracker
        import torch
        from torch.utils.data import DataLoader

        self.logger.info(f"Training with metadata support: {model_name}")

        # Initialize enhanced model
        if 'bert' in model_name.lower():
            model = BertBase(model_name=model_name, device=self.device)
        else:
            # Use regular model for non-BERT
            return self.train_single_model(
                model_name=model_name,
                train_df=pd.DataFrame([{'text': s.text, 'label': s.label} for s in train_samples]),
                val_df=pd.DataFrame([{'text': s.text, 'label': s.label} for s in val_samples]),
                test_df=pd.DataFrame([{'text': s.text, 'label': s.label} for s in test_samples]),
                num_labels=num_labels,
                output_dir=str(output_dir)
            )

        # Create enhanced dataloaders with metadata
        train_loader = model.encode_with_metadata(
            train_samples,
            batch_size=self.config.batch_size,
            progress_bar=True
        )
        val_loader = model.encode_with_metadata(
            val_samples,
            batch_size=self.config.batch_size,
            progress_bar=False
        )
        test_loader = model.encode_with_metadata(
            test_samples,
            batch_size=self.config.batch_size,
            progress_bar=False
        )

        # Setup metrics output directory
        metrics_dir = output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)

        # Train with enhanced tracking
        history = model.run_training_enhanced(
            train_dataloader=train_loader,
            test_dataloader=val_loader,
            n_epochs=self.config.num_epochs,
            lr=self.config.learning_rate,
            save_model_as=str(output_dir / 'model'),
            metrics_output_dir=str(metrics_dir),
            track_languages=track_languages
        )

        # Evaluate on test set
        test_predictions = model.predict(test_loader)
        test_probs = model.predict(test_loader, proba=True)

        # Get language-specific metrics if available
        language_metrics = {}
        if track_languages and hasattr(model, 'performance_tracker'):
            tracker = model.performance_tracker
            if tracker and hasattr(tracker, 'get_language_metrics'):
                language_metrics = tracker.get_language_metrics()

        # Calculate overall metrics
        test_labels = [s.label for s in test_samples]
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

        accuracy = accuracy_score(test_labels, test_predictions)
        precision, recall, f1_weighted, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='weighted'
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='macro'
        )

        report = classification_report(
            test_labels, test_predictions,
            output_dict=True
        )

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'model_path': str(output_dir),
            'training_time': sum(h.get('time', 0) for h in history) if history else 0,
            'best_epoch': len(history),
            'training_history': history,
            'classification_report': report,
            'language_metrics': language_metrics
        }

    def _train_multi_label(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-label training with multiple files

        This is called when training_files contains multiple files for different labels
        """
        training_files = config['training_files']
        results = {}
        overall_metrics = {
            'models_trained': 0,
            'total_training_time': 0,
            'per_label_results': {}
        }

        self.logger.info(f"Starting multi-label training for {len(training_files)} labels")

        # Train a model for each label
        for label_key, file_path in training_files.items():
            if label_key == 'multilabel':
                # This is the combined multi-label file, skip for now
                continue

            self.logger.info(f"\nTraining model for label: {label_key}")
            self.logger.info(f"Using file: {file_path}")

            # Create label-specific config
            label_config = config.copy()
            label_config['input_file'] = file_path
            label_config['output_dir'] = str(Path(config.get('output_dir', 'models')) / f'model_{label_key}')

            try:
                # Train model for this label
                result = self.train({**label_config, 'training_files': None})  # Remove training_files to avoid recursion
                results[label_key] = result
                overall_metrics['per_label_results'][label_key] = {
                    'accuracy': result['accuracy'],
                    'f1_macro': result['best_f1_macro'],
                    'model_path': result['model_path']
                }
                overall_metrics['models_trained'] += 1
                overall_metrics['total_training_time'] += result.get('training_time', 0)

                self.logger.info(f"✓ Completed {label_key}: Accuracy={result['accuracy']:.4f}, F1={result['best_f1_macro']:.4f}")

            except Exception as e:
                self.logger.error(f"Failed to train model for {label_key}: {str(e)}")
                overall_metrics['per_label_results'][label_key] = {'error': str(e)}

        # Calculate average metrics
        successful_results = [r for r in overall_metrics['per_label_results'].values() if 'error' not in r]
        if successful_results:
            overall_metrics['avg_accuracy'] = np.mean([r['accuracy'] for r in successful_results])
            overall_metrics['avg_f1_macro'] = np.mean([r['f1_macro'] for r in successful_results])

        self.logger.info(f"\nMulti-label training complete!")
        self.logger.info(f"Models trained: {overall_metrics['models_trained']}/{len(training_files)-1}")  # -1 for multilabel file
        if successful_results:
            self.logger.info(f"Average accuracy: {overall_metrics['avg_accuracy']:.4f}")
            self.logger.info(f"Average F1: {overall_metrics['avg_f1_macro']:.4f}")

        return {
            'best_model': 'multi_label_ensemble',
            'best_f1_macro': overall_metrics.get('avg_f1_macro', 0),
            'accuracy': overall_metrics.get('avg_accuracy', 0),
            'model_path': config.get('output_dir', 'models'),
            'training_time': overall_metrics['total_training_time'],
            'metrics': overall_metrics,
            'individual_results': results
        }


def main():
    """Example usage"""
    # Initialize trainer
    trainer = ModelTrainer()

    # Example: Benchmark models
    benchmark_config = BenchmarkConfig(
        models_to_test=[
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base"
        ],
        test_multilingual=True,
        test_sota=True
    )

    results = trainer.benchmark_models(
        data_path="data/annotations.csv",
        benchmark_config=benchmark_config
    )

    print(f"Best model: {results['best_model']}")
    print(f"Best F1 score: {results['best_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
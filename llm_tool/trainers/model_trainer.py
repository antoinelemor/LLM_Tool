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

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_split,
            random_state=self.config.seed, stratify=y
        )

        # Second split: train and validation
        val_size = self.config.validation_split / (1 - self.config.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.config.seed, stratify=y_temp
        )

        # Create DataFrames
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})

        self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")

        return train_df, val_df, test_df

    def train_single_model(self, model_name: str, train_df: pd.DataFrame,
                          val_df: pd.DataFrame, test_df: pd.DataFrame) -> TrainingResult:
        """Train a single model"""
        self.logger.info(f"Training model: {model_name}")
        start_time = time.time()

        # Get model class
        model_class = self.model_registry.get(model_name)
        if not model_class:
            # Try to use base Bert with custom model name
            model_class = Bert

        # Initialize model
        model_instance = model_class(model_name=model_name, device=self.device)

        # Prepare data
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].tolist()
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist()

        # Create output directory
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

            # Train
            history = model_instance.run_training(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                warmup_proportion=self.config.warmup_ratio,
                train_batch_size=self.config.batch_size,
                patience=self.config.early_stopping_patience,
                delta=self.config.early_stopping_threshold,
                save_path=str(output_dir),
                metric=self.config.metric_for_best_model
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
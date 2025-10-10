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
    SpanishBert, SwedishBert, MultiBERT, XLMRoberta
)
from .sota_models import (
    DeBERTaV3Base, DeBERTaV3Large, RoBERTaBase, RoBERTaLarge,
    ELECTRABase, ELECTRALarge, ALBERTBase, ALBERTLarge,
    BigBirdBase, LongformerBase, MDeBERTaV3Base, XLMRobertaBase,
    LongT5Base, LongT5TGlobalBase, get_model_class_for_name
)
from llm_tool.utils.training_paths import (
    get_training_logs_base,
    get_training_metrics_dir,
    resolve_metrics_base_dir,
)
from .multilingual_selector import MultilingualModelSelector
from ..utils.data_filter_logger import get_filter_logger
from .data_utils import safe_tolist, safe_convert_labels


__all__ = [
    'ModelTrainer',
    'TrainingConfig',
    'BenchmarkConfig',
    'TrainingResults',
    'MODEL_TARGET_LANGUAGES',
    'get_model_target_languages',
    'filter_data_by_language',
    'set_detected_languages_on_model',
]


# Model-to-target-language mapping
# Multilingual models (None = train on all languages)
# Monolingual models (specific language codes = filter data to only that language)
MODEL_TARGET_LANGUAGES = {
    # English-specific models
    "bert-base-uncased": ["EN"],
    "bert-large-uncased": ["EN"],
    "bert-base-cased": ["EN"],
    "bert-large-cased": ["EN"],
    "roberta-base": ["EN"],
    "roberta-large": ["EN"],
    "distilbert-base-uncased": ["EN"],
    "distilbert-base-cased": ["EN"],
    "electra-base-discriminator": ["EN"],
    "electra-large-discriminator": ["EN"],
    "albert-base-v2": ["EN"],
    "albert-large-v2": ["EN"],
    "albert-xlarge-v2": ["EN"],
    "albert-xxlarge-v2": ["EN"],
    "deberta-base": ["EN"],
    "deberta-large": ["EN"],
    "deberta-v3-base": ["EN"],
    "deberta-v3-large": ["EN"],
    "microsoft/deberta-v3-base": ["EN"],
    "microsoft/deberta-v3-large": ["EN"],

    # French-specific models
    "camembert-base": ["FR"],
    "camembert/camembert-base": ["FR"],
    "camembert/camembert-large": ["FR"],
    "flaubert/flaubert_base_cased": ["FR"],
    "flaubert/flaubert_large_cased": ["FR"],
    "cmarkea/distilcamembert-base": ["FR"],

    # Spanish-specific models
    "dccuchile/bert-base-spanish-wwm-cased": ["ES"],
    "PlanTL-GOB-ES/roberta-base-bne": ["ES"],

    # German-specific models
    "bert-base-german-cased": ["DE"],
    "deepset/gbert-base": ["DE"],
    "deepset/gbert-large": ["DE"],

    # Italian-specific models
    "dbmdz/bert-base-italian-cased": ["IT"],
    "Musixmatch/umberto-commoncrawl-cased-v1": ["IT"],

    # Portuguese-specific models
    "neuralmind/bert-base-portuguese-cased": ["PT"],
    "neuralmind/bert-large-portuguese-cased": ["PT"],

    # Arabic-specific models
    "asafaya/bert-base-arabic": ["AR"],
    "aubmindlab/bert-base-arabertv2": ["AR"],

    # Chinese-specific models
    "bert-base-chinese": ["ZH"],
    "hfl/chinese-bert-wwm": ["ZH"],
    "hfl/chinese-roberta-wwm-ext": ["ZH"],

    # Russian-specific models
    "DeepPavlov/rubert-base-cased": ["RU"],

    # Dutch-specific models
    "GroNLP/bert-base-dutch-cased": ["NL"],
    "pdelobelle/robbert-v2-dutch-base": ["NL"],

    # Polish-specific models
    "dkleczek/bert-base-polish-cased-v1": ["PL"],

    # Multilingual models (train on ALL languages)
    "bert-base-multilingual-cased": None,
    "bert-base-multilingual-uncased": None,
    "xlm-roberta-base": None,
    "xlm-roberta-large": None,
    "FacebookAI/xlm-roberta-base": None,
    "FacebookAI/xlm-roberta-large": None,
    "microsoft/mdeberta-v3-base": None,
    "microsoft/Multilingual-MiniLM-L12-H384": None,
    "distilbert-base-multilingual-cased": None,
}


def set_detected_languages_on_model(model, train_samples=None, val_samples=None,
                                     train_languages=None, val_languages=None,
                                     confirmed_languages=None, logger=None):
    """
    UNIFIED function to extract and set detected languages on a model instance.

    This function ensures CONSISTENT language detection across ALL training modes:
    - category-csv (single-label and multi-label)
    - jsonl modes
    - benchmark mode
    - quick mode

    Args:
        model: Model instance (BertBase or subclass)
        train_samples: Training samples with .lang attribute (optional)
        val_samples: Validation samples with .lang attribute (optional)
        train_languages: List of language codes from training data (optional)
        val_languages: List of language codes from validation data (optional)
        confirmed_languages: Pre-confirmed languages from user/config (optional, takes priority)
        logger: Logger instance for debug messages (optional)

    Returns:
        List of detected languages (uppercase, sorted)
    """
    detected_languages = []

    # Priority 1: Use confirmed_languages if provided (from user selection or config)
    if confirmed_languages:
        detected_languages = sorted(set(
            lang.upper() for lang in confirmed_languages
            if isinstance(lang, str) and lang
        ))
        if logger:
            logger.info(f"🌍 Using confirmed languages: {', '.join(detected_languages)}")

    # Priority 2: Extract from samples
    elif train_samples or val_samples:
        all_samples = []
        if train_samples:
            all_samples.extend(train_samples)
        if val_samples:
            all_samples.extend(val_samples)

        detected_languages = sorted(set(
            s.lang.upper() for s in all_samples
            if hasattr(s, 'lang') and s.lang and isinstance(s.lang, str)
        ))
        if logger:
            logger.info(f"🌍 Detected languages from samples: {', '.join(detected_languages)}")

    # Priority 3: Extract from language lists
    elif train_languages or val_languages:
        all_languages = []
        if train_languages:
            all_languages.extend(train_languages)
        if val_languages:
            all_languages.extend(val_languages)

        detected_languages = sorted(set(
            lang.upper() for lang in all_languages
            if isinstance(lang, str) and lang
        ))
        if logger:
            logger.info(f"🌍 Detected languages from language lists: {', '.join(detected_languages)}")

    # Set on model if we detected any languages
    if detected_languages:
        model.detected_languages = detected_languages
        if logger:
            logger.info(f"✓ Set model.detected_languages = {detected_languages}")
    elif logger:
        logger.warning("⚠️  No languages detected in data")

    return detected_languages


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
    metrics_output_dir: str = field(default_factory=lambda: str(get_training_logs_base()))
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


def filter_data_by_language(df: pd.DataFrame, target_languages: List[str],
                            language_column: str = 'language') -> pd.DataFrame:
    """
    Filter dataframe to keep only samples in target languages.

    Args:
        df: Input dataframe with language information
        target_languages: List of language codes to keep (e.g., ['EN', 'FR'])
        language_column: Name of column containing language codes

    Returns:
        Filtered dataframe with only target language samples
    """
    if language_column not in df.columns:
        logging.warning(f"Language column '{language_column}' not found. Returning all data.")
        return df

    # Normalize target languages to uppercase
    target_languages = [lang.upper() for lang in target_languages]

    # Filter dataframe
    # Use apply() to avoid pandas dtype issues with .str accessor
    original_size = len(df)
    filtered_df = df[df[language_column].apply(lambda x: str(x).upper() if pd.notna(x) else '').isin(target_languages)].copy()
    filtered_size = len(filtered_df)

    logging.info(f"Filtered data: {original_size} → {filtered_size} samples "
                f"(kept languages: {', '.join(target_languages)})")

    return filtered_df


def get_model_target_languages(model_name: str) -> Optional[List[str]]:
    """
    Get target languages for a model.

    Args:
        model_name: Model identifier (e.g., 'camembert-base', 'xlm-roberta-base')

    Returns:
        List of target language codes (e.g., ['FR']) or None for multilingual models
    """
    # Try exact match first
    if model_name in MODEL_TARGET_LANGUAGES:
        return MODEL_TARGET_LANGUAGES[model_name]

    # Try partial match (e.g., 'camembert' in 'camembert-base')
    model_name_lower = model_name.lower()
    for key, langs in MODEL_TARGET_LANGUAGES.items():
        if model_name_lower in key.lower() or key.lower() in model_name_lower:
            return langs

    # Default: assume multilingual
    logging.info(f"Model '{model_name}' not in language map. Treating as multilingual.")
    return None


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
            "bert-base-multilingual-cased": MultiBERT,
            "xlm-roberta-base": XLMRobertaBase,
            "microsoft/mdeberta-v3-base": MDeBERTaV3Base,

            # Language-specific models
            "camembert-base": Camembert,
            "arabic-bert": ArabicBert,
            "chinese-bert": ChineseBert,
            "german-bert": GermanBert,

            # Long document models
            "bigbird-base": BigBirdBase,
            "longformer-base": LongformerBase,

            # Multilingual long document models (T5-based)
            "google/long-t5-local-base": LongT5Base,
            "google/long-t5-tglobal-base": LongT5TGlobalBase,
        }

    def load_data(self, data_path: str, text_column: str = "text",
                  label_column: str = "label",
                  split_config: Optional[Dict[str, Any]] = None,
                  category_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data for training

        Args:
            data_path: Path to the data file
            text_column: Name of the text column
            label_column: Name of the label column
            split_config: Configuration for data splitting (from bundle.metadata)
            category_name: Name of the category being trained (key or key_value)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
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

        # Handle case where labels are in list format (legacy/compatibility)
        # Multiclass files should have string labels, but handle lists for backward compatibility
        if df[label_column].dtype == 'object' and len(df) > 0:
            # Check if any value is a list
            first_val = df[label_column].iloc[0]
            if isinstance(first_val, list):
                self.logger.info(f"Label column '{label_column}' contains lists. Extracting first value (multiclass expects strings).")
                df[label_column] = df[label_column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

        # Encode labels with custom ordering: NOT_* labels first (class 0), others later
        # CRITICAL FIX: LabelEncoder always re-sorts internally, so we need manual mapping
        try:
            unique_labels = df[label_column].unique()
        except TypeError as e:
            # If unique() fails due to unhashable type, convert all values to strings
            if 'unhashable type' in str(e):
                self.logger.warning(f"Label column contains unhashable types (likely lists). Converting to strings.")
                df[label_column] = df[label_column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x))
                unique_labels = df[label_column].unique()
            else:
                raise
        # CRITICAL: Convert to Python list to avoid numpy array issues
        sorted_labels = sorted(unique_labels.tolist(), key=lambda x: (not str(x).startswith('NOT_'), str(x)))

        # Create manual label mapping to ensure NOT_* = 0, others = 1+
        label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}

        # Still use label_encoder for consistency with rest of code, but override classes_
        self.label_encoder.fit(unique_labels)  # Fit with any order
        self.label_encoder.classes_ = np.array(sorted_labels)  # Override with correct order

        # Apply manual mapping
        df['encoded_label'] = df[label_column].map(label_to_id)

        # Split data
        X = df[text_column].values
        y = df['encoded_label'].values

        # CRITICAL: Preserve language column if it exists
        lang_col = None
        lang_col_name = None
        if 'lang' in df.columns:
            lang_col = df['lang'].values
            lang_col_name = 'lang'
        elif 'language' in df.columns:
            lang_col = df['language'].values
            lang_col_name = 'language'

        # Check if we have at least 2 instances per class for stratification
        from collections import Counter
        from rich.prompt import Confirm
        from rich.console import Console
        from ..utils.data_filter_logger import get_filter_logger

        label_counts = Counter(y)
        min_count = min(label_counts.values())

        if min_count < 2:
            # Find which classes have insufficient instances
            insufficient_classes = [cls for cls, count in label_counts.items() if count < 2]
            # Map back to original labels
            insufficient_labels = [sorted_labels[cls] for cls in insufficient_classes]

            # Display warning to user
            console = Console()
            console.print(f"\n[yellow]⚠️  Found {len(insufficient_labels)} label(s) with insufficient samples for training:[/yellow]")
            for label in insufficient_labels:
                # Find the class index for this label
                label_idx = sorted_labels.index(label)
                count = label_counts[label_idx]
                console.print(f"  • [red]'{label}'[/red]: {count} sample(s) - need at least 2 for train/test split")

            console.print(f"\n[bold]What would you like to do?[/bold]")
            console.print(f"  [cyan]1.[/cyan] [green]Remove[/green] these {len(insufficient_labels)} value(s) and continue with remaining data")
            console.print(f"  [cyan]2.[/cyan] [red]Cancel[/red] training to add more samples manually\n")

            remove_labels = Confirm.ask(
                f"[bold yellow]Remove insufficient labels and continue?[/bold yellow]",
                default=True
            )

            if remove_labels:
                # Filter out insufficient labels
                filter_logger = get_filter_logger()

                # Find indices of samples with insufficient labels
                mask = df[label_column].isin(insufficient_labels)
                indices_to_remove = df[mask].index.tolist()

                # Log filtered samples
                if indices_to_remove:
                    filter_logger.log_dataframe_filtering(
                        df_before=df,
                        df_after=df[~mask],
                        reason="insufficient_samples_per_class",
                        location="model_trainer.load_data",
                        text_column=text_column,
                        log_filtered_samples=min(5, len(indices_to_remove))
                    )

                    console.print(f"\n[green]✓ Removing {len(indices_to_remove)} sample(s) with insufficient labels:[/green]")
                    for label in insufficient_labels:
                        console.print(f"  • [dim]'{label}'[/dim]")

                # Remove from dataframe
                df = df[~mask].reset_index(drop=True)

                # Update arrays
                X = df[text_column].values
                y = df[label_column].values
                # CRITICAL: Convert to Python list to avoid numpy array issues
                sorted_labels = sorted(np.unique(y).tolist())

                # Update lang_col if present
                if lang_col_name:
                    lang_col = df[lang_col_name].values

                # Recompute label_counts
                label_counts = Counter(y)

                console.print(f"[green]✓ Continuing with {len(df)} samples and {len(sorted_labels)} unique labels[/green]\n")
            else:
                error_msg = (
                    f"Training cancelled by user.\n"
                    f"Please add more samples for: {', '.join(insufficient_labels)}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Extract split ratios from split_config or use defaults
        # NOTE: Validation is ALWAYS present for training evaluation
        # Test is optional for final evaluation
        validation_split = self.config.validation_split
        test_split = self.config.test_split

        if split_config is not None:
            mode = split_config.get('mode', 'uniform')
            use_test_set = split_config.get('use_test_set', False)

            if mode == 'uniform':
                # Uniform mode: same ratios for all categories
                uniform_config = split_config.get('uniform', {})
                validation_split = uniform_config.get('validation_ratio', validation_split)
                test_split = uniform_config.get('test_ratio', test_split) if use_test_set else 0.0
            elif mode == 'custom':
                # Custom mode: different ratios per category
                default_config = split_config.get('defaults', {})
                custom_by_key = split_config.get('custom_by_key', {})
                custom_by_value = split_config.get('custom_by_value', {})

                # Check if we have a custom config for this category
                if category_name and category_name in custom_by_key:
                    # Category-specific (key) configuration
                    category_config = custom_by_key[category_name]
                    validation_split = category_config.get('validation_ratio', default_config.get('validation_ratio', validation_split))
                    test_split = category_config.get('test_ratio', default_config.get('test_ratio', test_split))
                elif category_name and category_name in custom_by_value:
                    # Value-specific configuration
                    category_config = custom_by_value[category_name]
                    validation_split = category_config.get('validation_ratio', default_config.get('validation_ratio', validation_split))
                    test_split = category_config.get('test_ratio', default_config.get('test_ratio', test_split))
                else:
                    # Use default custom ratios
                    validation_split = default_config.get('validation_ratio', validation_split)
                    test_split = default_config.get('test_ratio', test_split)

                # If not using test set, set to 0.0
                if not use_test_set:
                    test_split = 0.0

            self.logger.info(f"Using split ratios for category '{category_name}': validation={validation_split:.2%}, test={test_split:.2%}")
        else:
            self.logger.info(f"Using default split ratios: validation={validation_split:.2%}, test={test_split:.2%}")

        # Check if we have enough samples for stratification
        n_samples = len(y)
        # CRITICAL: Convert to Python int to avoid numpy scalar issues
        n_classes = int(len(np.unique(y)))
        test_samples = int(n_samples * test_split) if test_split > 0 else 0
        val_samples = int(n_samples * validation_split)

        # Determine if stratification is possible
        # We need at least n_classes samples in each split for stratification
        can_stratify_test = (test_split > 0 and test_samples >= n_classes and
                            (n_samples - test_samples) >= n_classes)
        can_stratify_val = (val_samples >= n_classes and
                           (n_samples - val_samples) >= n_classes)

        if (test_split > 0 and not can_stratify_test) or not can_stratify_val:
            self.logger.warning(
                f"Too few samples ({n_samples}) for stratified split with {n_classes} classes. "
                f"Using random split instead. Consider increasing annotation sample size."
            )

        # NEW LOGIC: Validation is ALWAYS created, test is optional
        if test_split > 0.0:
            # Split 1: Separate test set from train+val
            stratify_test = y if can_stratify_test else None
            if lang_col is not None:
                X_temp, X_test, y_temp, y_test, lang_temp, lang_test = train_test_split(
                    X, y, lang_col, test_size=test_split,
                    random_state=self.config.seed, stratify=stratify_test
                )
            else:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_split,
                    random_state=self.config.seed, stratify=stratify_test
                )
                lang_temp, lang_test = None, None
        else:
            # No test set - all data goes to train+val
            X_temp, y_temp, lang_temp = X, y, lang_col
            X_test, y_test, lang_test = np.array([]), np.array([]), (np.array([]) if lang_col is not None else None)

        # Split 2: ALWAYS split train and validation (validation is mandatory)
        val_size = validation_split / (1 - test_split) if test_split > 0 else validation_split
        stratify_val = y_temp if can_stratify_val else None
        if lang_temp is not None:
            X_train, X_val, y_train, y_val, lang_train, lang_val = train_test_split(
                X_temp, y_temp, lang_temp, test_size=val_size,
                random_state=self.config.seed, stratify=stratify_val
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size,
                random_state=self.config.seed, stratify=stratify_val
            )
            lang_train, lang_val = None, None

        # Create DataFrames - preserve language column if it exists
        if lang_train is not None:
            train_df = pd.DataFrame({'text': X_train, 'label': y_train, lang_col_name: lang_train})
            val_df = pd.DataFrame({'text': X_val, 'label': y_val, lang_col_name: lang_val})
            test_df = pd.DataFrame({'text': X_test, 'label': y_test, lang_col_name: lang_test})
        else:
            train_df = pd.DataFrame({'text': X_train, 'label': y_train})
            val_df = pd.DataFrame({'text': X_val, 'label': y_val})
            test_df = pd.DataFrame({'text': X_test, 'label': y_test})

        if test_split > 0.0:
            self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        else:
            self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)} (no test set - validation used for final eval)")
        self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")

        return train_df, val_df, test_df

    def train_single_model(self, model_name: str, train_df: pd.DataFrame,
                          val_df: pd.DataFrame, test_df: pd.DataFrame,
                          num_labels: Optional[int] = None,
                          output_dir: Optional[str] = None,
                          label_column: str = 'label',
                          training_strategy: str = 'single-label',
                          category_name: Optional[str] = None,
                          class_names_override: Optional[List[str]] = None,
                          session_id: Optional[str] = None,
                          global_total_models: Optional[int] = None,
                          global_current_model: Optional[int] = None,
                          global_total_epochs: Optional[int] = None,
                          global_max_epochs: Optional[int] = None,
                          global_completed_epochs: Optional[int] = None,
                          global_start_time: Optional[float] = None,
                          reinforced_learning: bool = False,
                          reinforced_epochs: Optional[int] = None,
                          rl_f1_threshold: float = 0.7,
                          rl_oversample_factor: float = 2.0,
                          rl_class_weight_factor: float = 2.0) -> TrainingResult:
        """Train a single model"""
        from tqdm import tqdm
        print(f"\n🏋️  Training model: {model_name}")
        self.logger.info(f"Training model: {model_name} with {training_strategy} strategy")
        start_time = time.time()

        # Get model class
        model_class = self.model_registry.get(model_name)
        if not model_class:
            # Use mapping function to get correct class for model name
            model_class = get_model_class_for_name(model_name)

        # Initialize model
        model_instance = model_class(model_name=model_name, device=self.device)

        # ==================== LANGUAGE FILTERING FOR MONOLINGUAL MODELS ====================
        # Check if this model should only train on specific languages
        target_languages = get_model_target_languages(model_name)

        if target_languages is not None:
            # This is a language-specific model - filter data
            self.logger.info(f"🌍 Model '{model_name}' targets {target_languages}. Filtering data...")

            # Check if language column exists (can be 'language' or 'lang')
            lang_col = None
            if 'language' in train_df.columns:
                lang_col = 'language'
            elif 'lang' in train_df.columns:
                lang_col = 'lang'

            if lang_col:
                # Filter all splits
                train_df_original_size = len(train_df)
                train_df = filter_data_by_language(train_df, target_languages, lang_col)
                val_df = filter_data_by_language(val_df, target_languages, lang_col)
                test_df = filter_data_by_language(test_df, target_languages, lang_col)

                self.logger.info(f"✓ Filtered: {train_df_original_size} → {len(train_df)} train samples")

                # Verify we still have enough data
                if len(train_df) < 10:
                    self.logger.warning(f"⚠️  Very few training samples ({len(train_df)}) for {model_name} "
                                       f"targeting {target_languages}. Training may be unstable.")
            else:
                self.logger.warning(f"⚠️  No 'language' or 'lang' column found. Cannot filter for {model_name}. "
                                   f"Training on ALL data (not recommended for monolingual models).")
        else:
            # Multilingual model - use all data
            self.logger.info(f"🌍 Model '{model_name}' is multilingual. Using all language data.")

        # Prepare data - use the correct label column
        # CRITICAL: Filter out empty/invalid texts BEFORE processing
        # Get filter logger for tracking
        filter_logger = get_filter_logger()
        location = f"model_trainer.train_single_model({model_name})"

        # Clean training dataframe
        train_df_before = train_df.copy()
        train_df = train_df[train_df['text'].notna() & (train_df['text'].astype(str).str.strip() != '')]
        if len(train_df) < len(train_df_before):
            filter_logger.log_dataframe_filtering(
                df_before=train_df_before,
                df_after=train_df,
                reason="empty_or_nan_text",
                location=location + ".train_set",
                text_column='text',
                log_filtered_samples=5
            )

        # Clean validation dataframe
        val_df_before = val_df.copy()
        val_df = val_df[val_df['text'].notna() & (val_df['text'].astype(str).str.strip() != '')]
        if len(val_df) < len(val_df_before):
            filter_logger.log_dataframe_filtering(
                df_before=val_df_before,
                df_after=val_df,
                reason="empty_or_nan_text",
                location=location + ".validation_set",
                text_column='text',
                log_filtered_samples=5
            )

        # Clean test dataframe
        test_df_before = test_df.copy()
        test_df = test_df[test_df['text'].notna() & (test_df['text'].astype(str).str.strip() != '')]
        if len(test_df) < len(test_df_before):
            filter_logger.log_dataframe_filtering(
                df_before=test_df_before,
                df_after=test_df,
                reason="empty_or_nan_text",
                location=location + ".test_set",
                text_column='text',
                log_filtered_samples=5
            )

        # Now extract texts as strings
        train_texts = train_df['text'].astype(str).str.strip().tolist()
        # CRITICAL FIX: Convert labels to Python native types (int or str) using safe_tolist
        # This prevents issues with numpy.int64 objects in downstream processing
        train_labels = safe_tolist(train_df[label_column], column_name=label_column)
        val_texts = val_df['text'].astype(str).str.strip().tolist()
        val_labels = safe_tolist(val_df[label_column], column_name=label_column)
        test_texts = test_df['text'].astype(str).str.strip().tolist()
        test_labels = safe_tolist(test_df[label_column], column_name=label_column)

        self.logger.info(f"Final dataset sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

        # CRITICAL FIX: Extract language information for language-specific metrics
        train_languages = None
        val_languages = None
        test_languages = None
        track_languages = False

        # Check for language column (both 'language' and 'lang' for compatibility)
        if 'language' in train_df.columns:
            train_languages = safe_tolist(train_df['language'], column_name='language')
            val_languages = safe_tolist(val_df['language'], column_name='language')
            test_languages = safe_tolist(test_df['language'], column_name='language')
            track_languages = True
            self.logger.info(f"✓ Found 'language' column. Will track per-language metrics.")
        elif 'lang' in train_df.columns:
            train_languages = safe_tolist(train_df['lang'], column_name='lang')
            val_languages = safe_tolist(val_df['lang'], column_name='lang')
            test_languages = safe_tolist(test_df['lang'], column_name='lang')
            track_languages = True
            self.logger.info(f"✓ Found 'lang' column. Will track per-language metrics.")

        # UNIFIED: Use centralized function to set detected languages on model
        if track_languages:
            set_detected_languages_on_model(
                model=model_instance,
                train_languages=train_languages,
                val_languages=val_languages,
                logger=self.logger
            )

        # Determine output directory path (used for save_model_as parameter)
        # IMPORTANT: Do NOT create this directory here - it's a placeholder.
        # The actual model directory is created by bert_base.py when saving, using session_id.
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(self.config.output_dir) / model_name.replace('/', '_')
        # output_dir.mkdir(parents=True, exist_ok=True)  # Removed - created on-demand by bert_base.py

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
            # Priority 1: Check if category_name was passed (from one-vs-all training)
            # Priority 2: Parse label_column for structured names like 'themes_long_transportation'
            label_key = None
            label_value = None

            # First check if we have an explicit category_name (one-vs-all workflow)
            if category_name:
                label_value = category_name
            # Otherwise try to parse from label_column name
            elif label_column and '_long_' in label_column:
                parts = label_column.split('_long_')
                if len(parts) == 2:
                    label_key = parts[0]
                    label_value = parts[1]
            elif label_column and '_short_' in label_column:
                parts = label_column.split('_short_')
                if len(parts) == 2:
                    label_key = parts[0]
                    label_value = parts[1]

            # CRITICAL: Get class names from label encoder for display in metrics tables
            # self.label_encoder.classes_ contains the actual label names in correct order
            class_names_for_display = list(self.label_encoder.classes_) if hasattr(self, 'label_encoder') and self.label_encoder is not None else None
            if class_names_override is not None:
                class_names_for_display = [str(name) for name in class_names_override]
            elif class_names_for_display is not None:
                class_names_for_display = [str(name) for name in class_names_for_display]

            if class_names_for_display is not None and num_labels is not None:
                if len(class_names_for_display) != num_labels:
                    self.logger.warning(
                        f"⚠️ class_names length ({len(class_names_for_display)}) does not match num_labels ({num_labels}). "
                        "Adjusting to keep them in sync."
                    )
                    if len(class_names_for_display) > num_labels:
                        class_names_for_display = class_names_for_display[:num_labels]
                    else:
                        start_idx = len(class_names_for_display)
                        class_names_for_display = class_names_for_display + [
                            f"Class {i}" for i in range(start_idx, num_labels)
                        ]

            # Train (using test_dataloader as validation for compatibility with bert_base)
            # Initialize global progress tracking for single model training
            import time as time_module

            # Use provided global progress parameters if available (benchmark mode), otherwise use defaults (quick mode)
            if global_start_time is None:
                global_start_time = time_module.time()
            if global_total_models is None:
                global_total_models = 1
            if global_current_model is None:
                global_current_model = 1
            if global_total_epochs is None:
                global_total_epochs = self.config.num_epochs
            if global_completed_epochs is None:
                global_completed_epochs = 0

            # Calculate global_max_epochs if not provided
            # This accounts for potential reinforced learning epochs
            if global_max_epochs is None:
                if reinforced_learning and reinforced_epochs:
                    # Maximum possible epochs if reinforced learning is triggered
                    global_max_epochs = self.config.num_epochs + reinforced_epochs
                else:
                    # No reinforced learning or no extra epochs specified
                    global_max_epochs = global_total_epochs

            # CRITICAL: Convert to Python int to avoid numpy.int64 issues
            global_total_models = int(global_total_models) if global_total_models is not None else None
            global_current_model = int(global_current_model) if global_current_model is not None else None
            global_total_epochs = int(global_total_epochs) if global_total_epochs is not None else None
            global_max_epochs = int(global_max_epochs) if global_max_epochs is not None else None
            global_completed_epochs = int(global_completed_epochs) if global_completed_epochs is not None else None

            # CRITICAL DEBUG: Log all parameter types before calling run_training
            import numpy as np
            self.logger.debug("=" * 80)
            self.logger.debug("BEFORE run_training (train_single_model) - Parameter type check:")
            self.logger.debug(f"  n_epochs: type={type(self.config.num_epochs)}, value={self.config.num_epochs}")
            self.logger.debug(f"  lr: type={type(self.config.learning_rate)}, value={self.config.learning_rate}")
            self.logger.debug(f"  reinforced_epochs: type={type(reinforced_epochs)}, value={reinforced_epochs}")
            self.logger.debug(f"  reinforced_learning: type={type(reinforced_learning)}, value={reinforced_learning}")
            self.logger.debug(f"  session_id: type={type(session_id)}, value={session_id}")
            self.logger.debug(f"  global_total_models: type={type(global_total_models)}, value={global_total_models}")
            self.logger.debug(f"  global_current_model: type={type(global_current_model)}, value={global_current_model}")
            self.logger.debug(f"  global_total_epochs: type={type(global_total_epochs)}, value={global_total_epochs}")
            self.logger.debug(f"  global_max_epochs: type={type(global_max_epochs)}, value={global_max_epochs}")
            self.logger.debug(f"  global_completed_epochs: type={type(global_completed_epochs)}, value={global_completed_epochs}")
            self.logger.debug(f"  global_start_time: type={type(global_start_time)}, value={global_start_time}")

            # Check ALL locals for numpy types
            for key, value in locals().items():
                if isinstance(value, (np.integer, np.floating, np.ndarray)):
                    self.logger.warning(f"  ⚠️ NUMPY TYPE DETECTED: {key} = type={type(value)}, value={value}")
            self.logger.debug("=" * 80)

            metrics_base_dir = resolve_metrics_base_dir(getattr(self.config, "metrics_output_dir", None))

            best_metric, best_model_path, best_scores = model_instance.run_training(
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,  # bert_base expects test_dataloader for validation
                n_epochs=self.config.num_epochs,
                lr=self.config.learning_rate,
                random_state=42,
                save_model_as='model',  # Just the model name, bert_base.py will construct full path
                metrics_output_dir=str(metrics_base_dir),
                label_key=label_key,      # Pass parsed label key
                label_value=label_value,  # Pass parsed label value
                track_languages=track_languages,  # CRITICAL: Enable language tracking
                language_info=val_languages,  # CRITICAL: Pass language info for validation set
                class_names=class_names_for_display,  # CRITICAL: Pass class names for table display
                session_id=session_id,  # CRITICAL: Unified session ID for all runs
                global_total_models=global_total_models,
                global_current_model=global_current_model,
                global_total_epochs=global_total_epochs,
                global_max_epochs=global_max_epochs,
                global_completed_epochs=global_completed_epochs,
                global_start_time=global_start_time,
                reinforced_learning=reinforced_learning,
                reinforced_epochs=reinforced_epochs,
                reinforced_f1_threshold=rl_f1_threshold,
                # Add other RL params if needed
            )

            # Log final model path
            if best_model_path:
                self.logger.info(f"🎯 Training complete. Best model at: {best_model_path}")
            else:
                self.logger.warning(f"⚠️ Training complete but no model path returned")

            # FINAL EVALUATION: Use test set if available, otherwise use validation metrics
            if len(test_df) > 0:
                # Evaluate on separate test set (final unbiased evaluation)
                self.logger.info(f"📊 Final evaluation on test set ({len(test_df)} samples)...")
                test_predictions = model_instance.predict(test_dataloader, model_instance.model)
                test_probs = model_instance.predict(test_dataloader, model_instance.model, proba=True)

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
            else:
                # No separate test set - use validation metrics from training
                self.logger.info(f"📊 Using validation metrics as final evaluation (no separate test set)")
                # CRITICAL FIX: Use last_training_summary which is a dict, not best_scores (tuple)
                # best_scores is a numpy tuple, but last_training_summary is the dict created by bert_base.py
                training_summary = getattr(model_instance, 'last_training_summary', {})
                if training_summary:
                    accuracy = training_summary.get('accuracy', 0.0)
                    precision = training_summary.get('precision', [0.0])[0] if isinstance(training_summary.get('precision'), list) else training_summary.get('precision', 0.0)
                    recall = training_summary.get('recall', [0.0])[0] if isinstance(training_summary.get('recall'), list) else training_summary.get('recall', 0.0)
                    f1_weighted = training_summary.get('f1_weighted', 0.0)
                    f1_macro = training_summary.get('macro_f1', 0.0)
                    cm = training_summary.get('confusion_matrix', np.array([]))
                    report = training_summary.get('classification_report', {})
                else:
                    # Fallback: extract from numpy tuple if summary not available
                    if best_scores is not None and len(best_scores) >= 4:
                        # CRITICAL: Convert all np results to Python floats to avoid numpy scalar issues
                        accuracy = float(np.sum([p * s for p, s in zip(best_scores[0], best_scores[3])]) / np.sum(best_scores[3])) if best_scores[3] is not None and np.sum(best_scores[3]) > 0 else 0.0
                        precision = float(np.mean(best_scores[0])) if best_scores[0] is not None else 0.0
                        recall = float(np.mean(best_scores[1])) if best_scores[1] is not None else 0.0
                        f1_weighted = float(np.mean(best_scores[2])) if best_scores[2] is not None else 0.0
                        f1_macro = float(np.mean(best_scores[2])) if best_scores[2] is not None else 0.0
                        cm = np.array([])
                        report = {}
                    else:
                        accuracy = precision = recall = f1_weighted = f1_macro = 0.0
                        cm = np.array([])
                        report = {}

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
                best_epoch=0,  # TODO: Extract from training logs
                validation_loss=0,  # TODO: Extract from training logs
                training_history=[],  # TODO: Extract from training logs
                model_path=best_model_path if best_model_path else str(output_dir)  # CRITICAL: Use actual saved model path
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
            detection_result = detector.detect(' '.join(sample_texts))
            detected_lang = detection_result.get('language', 'en')
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
        # CRITICAL: Convert global progress parameters to Python int to avoid numpy.int64 issues
        if 'global_total_models' in config and config['global_total_models'] is not None:
            config['global_total_models'] = int(config['global_total_models'])
        if 'global_current_model' in config and config['global_current_model'] is not None:
            config['global_current_model'] = int(config['global_current_model'])
        if 'global_total_epochs' in config and config['global_total_epochs'] is not None:
            config['global_total_epochs'] = int(config['global_total_epochs'])
        if 'global_max_epochs' in config and config['global_max_epochs'] is not None:
            config['global_max_epochs'] = int(config['global_max_epochs'])
        if 'global_completed_epochs' in config and config['global_completed_epochs'] is not None:
            config['global_completed_epochs'] = int(config['global_completed_epochs'])

        # CRITICAL: Check training_strategy FIRST before checking training_files
        # This ensures single-label is correctly routed even if training_files exists
        training_strategy = config.get('training_strategy', 'single-label')

        # Route based on explicit strategy first
        if training_strategy == 'multi-label':
            # Check if we have multi-label data files
            if 'training_files' in config and config['training_files']:
                return self._train_multi_label(config)
            elif 'input_file' in config:
                # Multi-label with single file (e.g., JSONL format)
                data_path = config['input_file']
            else:
                raise ValueError("No input_file or training_files specified for multi-label training")
        elif training_strategy == 'single-label':
            # Single-label training - use input_file
            if 'input_file' not in config:
                raise ValueError("No input_file specified for single-label training")
            data_path = config['input_file']
        else:
            # Fallback to old behavior for backward compatibility
            if 'training_files' in config and config['training_files']:
                return self._train_multi_label(config)
            elif 'input_file' in config:
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

            # Filter by language if requested (for per-language training)
            filter_by_language = config.get('filter_by_language')
            if filter_by_language and lang_col is not None:
                # Normalize language comparison (uppercase)
                filter_lang = filter_by_language.upper()
                lang_mask = np.array([str(lang).upper() == filter_lang for lang in lang_col])

                # Apply filter
                original_count = len(df)
                df = df[lang_mask].reset_index(drop=True)
                X = df[text_column].values
                y = df[label_column].values
                lang_col = df['lang'].values if 'lang' in df.columns else None

                self.logger.info(f"Filtered data for language {filter_lang}: {len(df)}/{original_count} samples")

            # Extract split ratios from split_config or use defaults
            split_config = config.get('split_config')
            category_name = config.get('category_name')
            validation_split = self.config.validation_split
            test_split = self.config.test_split

            if split_config is not None:
                mode = split_config.get('mode', 'uniform')
                use_test_set = split_config.get('use_test_set', False)

                if mode == 'uniform':
                    # Uniform mode: same ratios for all categories
                    uniform_config = split_config.get('uniform', {})
                    validation_split = uniform_config.get('validation_ratio', validation_split)
                    test_split = uniform_config.get('test_ratio', test_split) if use_test_set else 0.0
                elif mode == 'custom':
                    # Custom mode: use default ratios for multi-label (category_name typically not specific here)
                    default_config = split_config.get('defaults', {})
                    validation_split = default_config.get('validation_ratio', validation_split)
                    test_split = default_config.get('test_ratio', test_split) if use_test_set else 0.0

                self.logger.info(f"Using split ratios for multi-label: validation={validation_split:.2%}, test={test_split:.2%}")

            # NEW LOGIC: Validation is ALWAYS created, test is optional
            if test_split > 0.0:
                # Split 1: Separate test set from train+val
                if lang_col is not None:
                    X_temp, X_test, y_temp, y_test, lang_temp, lang_test = train_test_split(
                        X, y, lang_col, test_size=test_split,
                        random_state=self.config.seed
                    )
                else:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_split,
                        random_state=self.config.seed
                    )
                    lang_temp, lang_test = None, None
            else:
                # No test set - all data goes to train+val
                X_temp, y_temp, lang_temp = X, y, lang_col
                X_test, y_test, lang_test = np.array([]), np.array([]), (np.array([]) if lang_col is not None else None)

            # Split 2: ALWAYS split train and validation (validation is mandatory)
            val_size = validation_split / (1 - test_split) if test_split > 0 else validation_split
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
                # Handle labels: keep dicts and lists as-is, wrap other types in list
                labels = y_train[i]
                if not isinstance(labels, (list, dict)):
                    labels = [labels]

                sample = {
                    'text': X_train[i],
                    'labels': labels
                }
                if lang_train is not None:
                    sample['lang'] = lang_train[i]
                train_samples.append(sample)

            val_samples = []
            for i in range(len(X_val)):
                # Handle labels: keep dicts and lists as-is, wrap other types in list
                labels = y_val[i]
                if not isinstance(labels, (list, dict)):
                    labels = [labels]

                sample = {
                    'text': X_val[i],
                    'labels': labels
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
            # Use the model specified by the user
            ml_config.model_name = config.get('model_name', self.config.model_name)
            # Pass manual reinforced epochs if provided
            if 'reinforced_epochs' in config and config['reinforced_epochs'] is not None:
                ml_config.reinforced_epochs = config['reinforced_epochs']

            ml_trainer = MultiLabelTrainer(config=ml_config, verbose=True)

            # Detect multi-class groups
            from llm_tool.trainers.multi_label_trainer import MultiLabelSample

            # Convert train samples to MultiLabelSample with dict labels
            ml_train_samples = []
            for s in train_samples:
                # Handle both dict and list formats for labels
                labels_raw = s.get('labels', {})
                if isinstance(labels_raw, list):
                    # Check if list contains a single dict (wrapped dict case from benchmark)
                    if len(labels_raw) == 1 and isinstance(labels_raw[0], dict):
                        labels = labels_raw[0]
                    else:
                        # Convert list to dict: ['label1', 'label2'] -> {'label1': 1, 'label2': 1}
                        labels = {label: 1 for label in labels_raw if label and isinstance(label, str)}
                else:
                    labels = labels_raw

                ml_train_samples.append(MultiLabelSample(
                    text=s['text'],
                    labels=labels,
                    id=s.get('id'),
                    lang=s.get('lang'),
                    metadata=s.get('metadata')
                ))

            # Convert val samples to MultiLabelSample with dict labels
            ml_val_samples = []
            for s in val_samples:
                # Handle both dict and list formats for labels
                labels_raw = s.get('labels', {})
                if isinstance(labels_raw, list):
                    # Check if list contains a single dict (wrapped dict case from benchmark)
                    if len(labels_raw) == 1 and isinstance(labels_raw[0], dict):
                        labels = labels_raw[0]
                    else:
                        # Convert list to dict: ['label1', 'label2'] -> {'label1': 1, 'label2': 1}
                        labels = {label: 1 for label in labels_raw if label and isinstance(label, str)}
                else:
                    labels = labels_raw

                ml_val_samples.append(MultiLabelSample(
                    text=s['text'],
                    labels=labels,
                    id=s.get('id'),
                    lang=s.get('lang'),
                    metadata=s.get('metadata')
                ))

            # Combine for multiclass group detection
            ml_samples = ml_train_samples + ml_val_samples

            # Use multiclass_groups from config if provided, otherwise detect
            # CRITICAL: Do NOT auto-detect if user explicitly chose one-vs-all training
            training_approach = config.get('training_approach')
            multiclass_groups = config.get('multiclass_groups')
            if multiclass_groups is None and training_approach != 'one-vs-all':
                multiclass_groups = ml_trainer.detect_multiclass_groups(ml_samples)
                # Only log if we detected it ourselves (not passed from config)
                if multiclass_groups:
                    self.logger.info(f"Detected {len(multiclass_groups)} multi-class group(s):")
                    for group_name, labels in multiclass_groups.items():
                        value_names = [lbl[len(group_name)+1:] if lbl.startswith(group_name+'_') else lbl for lbl in labels]
                        self.logger.info(f"  • {group_name}: {', '.join(value_names)}")
            elif training_approach == 'one-vs-all':
                # User explicitly chose one-vs-all, force binary training for each label
                self.logger.info("One-vs-all training: creating separate binary models for each label")

            # Train models - CRITICAL: Pass converted samples with dict labels, not original samples
            # CRITICAL: Start timer for training time measurement
            training_start_time = time.time()

            trained_models = ml_trainer.train(
                train_samples=ml_train_samples,
                val_samples=ml_val_samples,
                auto_split=False,
                output_dir=config.get('output_dir', 'models/best_model'),
                multiclass_groups=multiclass_groups,  # Pass detected or provided groups
                confirmed_languages=config.get('confirmed_languages'),  # Pass all detected languages
                session_id=config.get('session_id'),  # Pass unified session ID for benchmark
                is_benchmark=config.get('is_benchmark', False),  # Pass benchmark flag
                model_name_for_logging=config.get('model_name'),  # Pass model name for logging
                global_total_models=config.get('global_total_models'),
                global_current_model=config.get('global_current_model'),
                global_total_epochs=config.get('global_total_epochs'),
                global_max_epochs=config.get('global_max_epochs'),  # CRITICAL: Pass max epochs for "(max X)" display
                global_completed_epochs=config.get('global_completed_epochs'),
                global_start_time=config.get('global_start_time'),
                reinforced_learning=config.get('reinforced_learning', False),  # CRITICAL: Enable "(max X)" display
                reinforced_epochs=config.get('reinforced_epochs'),  # CRITICAL: Pass manual reinforced epochs if configured
                rl_f1_threshold=config.get('rl_f1_threshold', 0.7),
                rl_oversample_factor=config.get('rl_oversample_factor', 2.0),
                rl_class_weight_factor=config.get('rl_class_weight_factor', 2.0),
                progress_callback=config.get('progress_callback')  # CRITICAL: Pass progress callback for epoch tracking
            )

            # CRITICAL: Calculate total training time
            total_training_time = time.time() - training_start_time

            # Aggregate results
            if trained_models:
                # CRITICAL: Convert np.mean results to Python floats to avoid numpy scalar issues
                avg_f1 = float(np.mean([m.performance_metrics.get('f1_macro', 0) for m in trained_models.values()]))
                avg_acc = float(np.mean([m.performance_metrics.get('accuracy', 0) for m in trained_models.values()]))

                self.logger.info(f"Multi-label training complete!")
                self.logger.info(f"Models trained: {len(trained_models)}/{len(trained_models)}")

                # Extract per-category metrics for detailed reporting (especially for benchmark)
                category_metrics = {}
                for model_name, model_info in trained_models.items():
                    category_metrics[model_name] = {
                        'f1_macro': model_info.performance_metrics.get('f1_macro', 0),
                        'accuracy': model_info.performance_metrics.get('accuracy', 0),
                        'precision': model_info.performance_metrics.get('precision', 0),
                        'recall': model_info.performance_metrics.get('recall', 0),
                        'model_path': model_info.model_path
                    }

                # CRITICAL FIX: If single category, return its detailed metrics directly
                # This ensures benchmark display works correctly for single-label datasets
                if len(trained_models) == 1:
                    # Single category - return its metrics directly (not averaged)
                    single_model = list(trained_models.values())[0]
                    perf = single_model.performance_metrics

                    # CRITICAL: Get training_time from performance_metrics or use calculated time
                    training_time = perf.get('training_time', total_training_time)

                    results = {
                        'f1_macro': perf.get('f1_macro', 0),
                        'f1': perf.get('f1_macro', 0),  # Alias for compatibility
                        'accuracy': perf.get('accuracy', 0),
                        'precision': perf.get('precision', 0),
                        'recall': perf.get('recall', 0),
                        'model_path': single_model.model_path,
                        'training_time': training_time,
                        'trained_models': {k: v.model_path for k, v in trained_models.items()},
                        'category_metrics': category_metrics
                    }

                    # Add per-class metrics if available (binary classification)
                    if 'f1_0' in perf:
                        results['f1_0'] = perf['f1_0']
                        results['f1_class_0'] = perf['f1_0']
                    if 'f1_1' in perf:
                        results['f1_1'] = perf['f1_1']
                        results['f1_class_1'] = perf['f1_1']
                    if 'precision_0' in perf:
                        results['precision_0'] = perf['precision_0']
                    if 'precision_1' in perf:
                        results['precision_1'] = perf['precision_1']
                    if 'recall_0' in perf:
                        results['recall_0'] = perf['recall_0']
                    if 'recall_1' in perf:
                        results['recall_1'] = perf['recall_1']

                    # Add language metrics if available
                    if 'language_metrics' in perf:
                        results['language_metrics'] = perf['language_metrics']
                else:
                    # Multiple categories - return averaged metrics
                    # CRITICAL: Average training time across all models or use total time
                    # Convert np.mean result to Python float to avoid numpy scalar issues
                    avg_training_time = float(np.mean([m.performance_metrics.get('training_time', 0) for m in trained_models.values()]))
                    if avg_training_time == 0:
                        avg_training_time = total_training_time

                    results = {
                        'f1_macro': avg_f1,
                        'f1': avg_f1,  # Alias for compatibility
                        'accuracy': avg_acc,
                        'model_path': config.get('output_dir', 'models/best_model'),
                        'training_time': avg_training_time,
                        'trained_models': {k: v.model_path for k, v in trained_models.items()},
                        'category_metrics': category_metrics  # NEW: Detailed metrics per category
                    }
            else:
                results = {
                    'f1_macro': 0.0,
                    'accuracy': 0.0,
                    'model_path': '',
                    'training_time': total_training_time
                }
        else:
            # Load data for single-label training (uses encoding)
            split_config = config.get('split_config')
            category_name = config.get('category_name')

            train_df, val_df, test_df = self.load_data(
                data_path,
                text_column=text_column,
                label_column=label_column,
                split_config=split_config,
                category_name=category_name
            )

            # Get number of unique labels
            # Note: load_data returns dataframes with 'label' column (hardcoded)
            all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])
            # CRITICAL: Convert to Python int to avoid numpy.int64 issues
            num_labels = int(len(all_labels.unique()))

            # ==================== PER-LANGUAGE MODEL TRAINING ====================
            # Check if we should train separate models per language
            models_by_language = config.get('models_by_language')

            # CRITICAL: Validate models_by_language is a dict (defensive fix for numpy type issues)
            if models_by_language is not None and not isinstance(models_by_language, dict):
                self.logger.warning(f"⚠️  models_by_language has invalid type: {type(models_by_language)} (value: {models_by_language})")
                self.logger.warning("   Expected dict, got scalar or other type. Disabling per-language training.")
                models_by_language = None

            if models_by_language:
                # Train separate model for each language
                self.logger.info(f"🌍 Training {len(models_by_language)} language-specific models")

                # Check if language column exists (can be 'language' or 'lang')
                lang_col_name = None
                if 'language' in train_df.columns:
                    lang_col_name = 'language'
                    self.logger.info(f"✓ Found 'language' column for filtering")
                elif 'lang' in train_df.columns:
                    lang_col_name = 'lang'
                    self.logger.info(f"✓ Found 'lang' column for filtering")

                if not lang_col_name:
                    self.logger.error("❌ Cannot train per-language models: No 'language' or 'lang' column found in data")
                    raise ValueError("Per-language training requested but no language column found in data")

                # Train a model for each language
                language_results = {}
                total_training_time = 0

                for lang_code, lang_model in models_by_language.items():
                    self.logger.info(f"\n🏋️  Training {lang_code} model: {lang_model}")

                    # Filter data to this language only
                    lang_code_upper = lang_code.upper()

                    # Ensure language column is string type before filtering
                    # Use apply() instead of astype(str).str to avoid pandas dtype issues
                    if len(train_df) > 0:
                        train_df_lang = train_df[train_df[lang_col_name].apply(lambda x: str(x).upper() if pd.notna(x) else '') == lang_code_upper].copy()
                    else:
                        train_df_lang = train_df.copy()

                    if len(val_df) > 0:
                        val_df_lang = val_df[val_df[lang_col_name].apply(lambda x: str(x).upper() if pd.notna(x) else '') == lang_code_upper].copy()
                    else:
                        val_df_lang = val_df.copy()

                    if len(test_df) > 0:
                        test_df_lang = test_df[test_df[lang_col_name].apply(lambda x: str(x).upper() if pd.notna(x) else '') == lang_code_upper].copy()
                    else:
                        test_df_lang = test_df.copy()

                    self.logger.info(f"  • Filtered to {lang_code}: Train={len(train_df_lang)}, Val={len(val_df_lang)}, Test={len(test_df_lang)}")

                    # Skip if insufficient data
                    if len(train_df_lang) < 10:
                        self.logger.warning(f"⚠️  Skipping {lang_code}: Insufficient training samples ({len(train_df_lang)})")
                        continue

                    # CRITICAL: Re-encode labels to ensure they are contiguous (0, 1, 2, ..., n-1)
                    # After filtering by language, we may have gaps in label encoding
                    # Example: Original labels [0, 1, 2] → After filtering EN: [0, 2] → Need to remap to [0, 1]

                    # Only concat non-empty dataframes for label remapping
                    dfs_to_concat = [train_df_lang['label'], test_df_lang['label']]
                    if len(val_df_lang) > 0:
                        dfs_to_concat.insert(1, val_df_lang['label'])
                    all_labels_lang = pd.concat(dfs_to_concat)
                    # CRITICAL: Convert to Python list to avoid numpy.int64 scalar issues
                    unique_labels_lang = sorted(all_labels_lang.unique().tolist())

                    # Create mapping from old labels to new contiguous labels
                    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels_lang)}
                    self.logger.info(f"  • {lang_code} label remapping: {label_mapping}")

                    # Apply remapping
                    train_df_lang['label'] = train_df_lang['label'].map(label_mapping)
                    if len(val_df_lang) > 0:
                        val_df_lang['label'] = val_df_lang['label'].map(label_mapping)
                    test_df_lang['label'] = test_df_lang['label'].map(label_mapping)

                    class_names_lang = None
                    if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                        all_class_names = list(self.label_encoder.classes_)
                        class_names_lang = []
                        for old_label in unique_labels_lang:
                            if 0 <= old_label < len(all_class_names):
                                class_names_lang.append(str(all_class_names[old_label]))
                            else:
                                self.logger.warning(
                                    f"  ⚠️ Label index {old_label} out of range for class names (len={len(all_class_names)}). "
                                    "Using fallback placeholder."
                                )
                                class_names_lang.append(f"Class {len(class_names_lang)}")
                    num_labels_lang = len(class_names_lang) if class_names_lang is not None else len(unique_labels_lang)
                    self.logger.info(f"  • {lang_code} has {num_labels_lang} unique label(s): {list(range(num_labels_lang))}")
                    if class_names_lang:
                        self.logger.info(f"  • {lang_code} class names: {class_names_lang}")

                    # Create language-specific output directory
                    base_output_dir = config.get('output_dir', 'models/best_model')
                    lang_output_dir = str(Path(base_output_dir) / f'model_{lang_code}')

                    # Train model for this language
                    try:
                        lang_result = self.train_single_model(
                            model_name=lang_model,
                            train_df=train_df_lang,
                            val_df=val_df_lang,
                            test_df=test_df_lang,
                            num_labels=num_labels_lang,  # Use language-specific num_labels
                            output_dir=lang_output_dir,
                            label_column='label',
                            training_strategy=training_strategy,
                            category_name=config.get('category_name'),
                            class_names_override=class_names_lang,
                            session_id=config.get('session_id'),  # CRITICAL: Unified session ID
                            global_total_models=config.get('global_total_models'),
                            global_current_model=config.get('global_current_model'),
                            global_total_epochs=config.get('global_total_epochs'),
                            global_max_epochs=config.get('global_max_epochs'),
                            global_completed_epochs=config.get('global_completed_epochs'),
                            global_start_time=config.get('global_start_time'),
                            reinforced_learning=config.get('reinforced_learning', False),
                            reinforced_epochs=config.get('reinforced_epochs'),
                            rl_f1_threshold=config.get('rl_f1_threshold', 0.7),
                            rl_oversample_factor=config.get('rl_oversample_factor', 2.0),
                            rl_class_weight_factor=config.get('rl_class_weight_factor', 2.0)
                        )

                        language_results[lang_code] = {
                            'model': lang_model,
                            'accuracy': lang_result.best_accuracy,
                            'f1_macro': lang_result.best_f1_macro,
                            'f1_weighted': lang_result.best_f1_weighted,
                            'model_path': lang_result.model_path,
                            'training_time': lang_result.training_time,
                            'samples_trained': len(train_df_lang)
                        }
                        total_training_time += lang_result.training_time

                        self.logger.info(f"✓ {lang_code} model complete: Accuracy={lang_result.best_accuracy:.4f}, F1={lang_result.best_f1_macro:.4f}")

                    except Exception as e:
                        # CRITICAL: Log full traceback to identify where numpy.int64 has no len() error occurs
                        import traceback
                        self.logger.error(f"❌ Failed to train {lang_code} model: {str(e)}", exc_info=True)
                        self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        language_results[lang_code] = {'error': str(e)}
                        # CRITICAL: Re-raise to see the actual error instead of "All language-specific models failed"
                        raise

                # Calculate aggregate metrics
                successful_langs = [lang for lang, res in language_results.items() if 'error' not in res]
                if successful_langs:
                    # CRITICAL: Convert np.mean results to Python float to avoid numpy scalar issues
                    avg_accuracy = float(np.mean([language_results[lang]['accuracy'] for lang in successful_langs]))
                    avg_f1 = float(np.mean([language_results[lang]['f1_macro'] for lang in successful_langs]))

                    self.logger.info(f"\n✅ Per-language training complete!")
                    self.logger.info(f"   Models trained: {len(successful_langs)}/{len(models_by_language)}")
                    self.logger.info(f"   Average accuracy: {avg_accuracy:.4f}")
                    self.logger.info(f"   Average F1: {avg_f1:.4f}")

                    # Return aggregated results
                    results = {
                        'accuracy': avg_accuracy,
                        'f1_macro': avg_f1,
                        'model_path': config.get('output_dir', 'models/best_model'),
                        'training_time': total_training_time,
                        'language_results': language_results,
                        'models_trained': len(successful_langs)
                    }
                else:
                    raise ValueError("All language-specific models failed to train")
            else:
                # Use single-label training with single model
                results = self.train_single_model(
                    model_name=model_name,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    num_labels=num_labels,
                    output_dir=config.get('output_dir', 'models/best_model'),
                    label_column='label',  # load_data returns 'label' column
                    training_strategy=training_strategy,
                    category_name=config.get('category_name'),  # Pass category name for display
                    session_id=config.get('session_id'),  # CRITICAL: Unified session ID
                    global_total_models=config.get('global_total_models'),
                    global_current_model=config.get('global_current_model'),
                    global_total_epochs=config.get('global_total_epochs'),
                    global_max_epochs=config.get('global_max_epochs'),
                    global_completed_epochs=config.get('global_completed_epochs'),
                    global_start_time=config.get('global_start_time'),
                    reinforced_learning=config.get('reinforced_learning', False),
                    reinforced_epochs=config.get('reinforced_epochs'),
                    rl_f1_threshold=config.get('rl_f1_threshold', 0.7),
                    rl_oversample_factor=config.get('rl_oversample_factor', 2.0),
                    rl_class_weight_factor=config.get('rl_class_weight_factor', 2.0)
                )

        # Return results (handle both single-model and per-language cases)
        if config.get('models_by_language'):
            return {
                'best_model': f"per_language_ensemble ({len(results.get('language_results', {}))} models)",
                'best_f1_macro': results.get('f1_macro', 0.0),
                'accuracy': results.get('accuracy', 0.0),
                'model_path': results.get('model_path', ''),
                'training_time': results.get('training_time', 0),
                'metrics': results,
                'models_by_language': config.get('models_by_language'),
                'language_results': results.get('language_results', {})
            }
        else:
            # Handle both TrainingResult objects and dictionaries
            if isinstance(results, TrainingResult):
                # Convert TrainingResult object to dictionary format
                return {
                    'best_model': model_name,
                    'best_f1_macro': results.best_f1_macro,
                    'accuracy': results.best_accuracy,
                    'model_path': results.model_path,
                    'training_time': results.training_time,
                    'metrics': {
                        'f1_macro': results.best_f1_macro,
                        'f1_weighted': results.best_f1_weighted,
                        'accuracy': results.best_accuracy,
                        'precision': results.precision,
                        'recall': results.recall,
                        'best_epoch': results.best_epoch,
                        'validation_loss': results.validation_loss,
                        'classification_report': results.classification_report,
                        'confusion_matrix': results.confusion_matrix,
                        'training_history': results.training_history
                    }
                }
            else:
                # results is already a dictionary
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

        # Initialize enhanced model using correct model class
        model_class = get_model_class_for_name(model_name)
        if model_class == BertBase:
            # Only BertBase supports the advanced metadata features currently
            model = model_class(model_name=model_name, device=self.device)
        else:
            # Use regular training for other model types
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

        metrics_base_dir = resolve_metrics_base_dir(getattr(self.config, "metrics_output_dir", None))

        # Train with enhanced tracking
        # Extract language info from samples if available
        val_language_info = [s.lang for s in val_samples if hasattr(s, 'lang')] if track_languages else None

        # Initialize global progress tracking for single model training
        import time as time_module
        global_start_time = time_module.time()

        # CRITICAL DEBUG: Log all parameter types before calling run_training
        import numpy as np
        self.logger.debug("=" * 80)
        self.logger.debug("BEFORE run_training - Parameter type check:")
        self.logger.debug(f"  n_epochs: type={type(self.config.num_epochs)}, value={self.config.num_epochs}")
        self.logger.debug(f"  lr: type={type(self.config.learning_rate)}, value={self.config.learning_rate}")
        self.logger.debug(f"  session_id: type={type(config.get('session_id'))}, value={config.get('session_id')}")
        self.logger.debug(f"  global_total_models: type={type(1)}, value=1")
        self.logger.debug(f"  global_current_model: type={type(1)}, value=1")
        self.logger.debug(f"  global_total_epochs: type={type(self.config.num_epochs)}, value={self.config.num_epochs}")
        self.logger.debug(f"  global_completed_epochs: type={type(0)}, value=0")
        self.logger.debug(f"  global_start_time: type={type(global_start_time)}, value={global_start_time}")

        # Check for numpy types
        for key, value in locals().items():
            if isinstance(value, (np.integer, np.floating, np.ndarray)):
                self.logger.warning(f"  ⚠️ NUMPY TYPE DETECTED: {key} = type={type(value)}, value={value}")
        self.logger.debug("=" * 80)

        # CRITICAL FIX: Use run_training (not run_training_enhanced which doesn't exist) with language_info
        best_metric, best_model_path, best_scores = model.run_training(
            train_dataloader=train_loader,
            test_dataloader=val_loader,
            n_epochs=self.config.num_epochs,
            lr=self.config.learning_rate,
            save_model_as='model',  # Just the model name, bert_base.py will construct full path
            metrics_output_dir=str(metrics_base_dir),
            track_languages=track_languages,
            language_info=val_language_info,  # CRITICAL: Pass language info for per-language metrics
            reinforced_learning=self.config.reinforced_learning if hasattr(self.config, 'reinforced_learning') else False,
            session_id=config.get('session_id'),
            global_total_models=1,  # Quick mode trains single model
            global_current_model=1,
            global_total_epochs=self.config.num_epochs,
            global_max_epochs=self.config.num_epochs + (self.config.reinforced_epochs if hasattr(self.config, 'reinforced_epochs') and self.config.reinforced_epochs and self.config.reinforced_learning else 0),
            global_completed_epochs=0,
            global_start_time=global_start_time
        )

        # Log final model path
        if best_model_path:
            self.logger.info(f"🎯 Training complete. Best model at: {best_model_path}")
        else:
            self.logger.warning(f"⚠️ Training complete but no model path returned")

        # Evaluate on test set
        test_predictions = model.predict(test_loader, model.model)
        test_probs = model.predict(test_loader, model.model, proba=True)

        # Get language-specific metrics if available - CRITICAL FIX: Use last_training_summary
        language_metrics = {}
        training_summary = getattr(model, 'last_training_summary', {})

        if track_languages and training_summary:
            language_metrics = training_summary.get('language_metrics', {})
        elif track_languages and hasattr(model, 'performance_tracker'):
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
        # Get per-class F1 scores
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            test_labels, test_predictions, average=None
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='macro'
        )

        report = classification_report(
            test_labels, test_predictions,
            output_dict=True
        )

        # CRITICAL FIX: Return keys matching both old and new formats
        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1': f1_macro,  # Legacy key
            'f1_macro': f1_macro,  # New key for consistency
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'model_path': best_model_path if best_model_path else str(output_dir),
            'output_dir': str(output_dir),
            'best_metric': best_metric,
            'best_scores': best_scores,
            'training_time': 0,  # TODO: Extract from training logs
            'best_epoch': 0,  # TODO: Extract from training logs
            'classification_report': report,
            'language_metrics': language_metrics
        }

        # Add per-class F1 scores if binary classification
        if len(f1_per_class) == 2:
            result['f1_0'] = f1_per_class[0]
            result['f1_1'] = f1_per_class[1]
            result['f1_class_0'] = f1_per_class[0]
            result['f1_class_1'] = f1_per_class[1]
            result['precision_0'] = precision_per_class[0]
            result['precision_1'] = precision_per_class[1]
            result['recall_0'] = recall_per_class[0]
            result['recall_1'] = recall_per_class[1]

        return result

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
                # Train model for this label - CRITICAL: Change strategy to 'single-label'
                # Each category CSV file is a binary classification problem (label 0 or 1)
                # Pass category name for proper label display (e.g., "Health IS" vs "Health IS NOT")
                result = self.train({
                    **label_config,
                    'training_files': None,
                    'training_strategy': 'single-label',
                    'category_name': label_key  # Pass category name for display
                })
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
            # CRITICAL: Convert np.mean results to Python floats to avoid numpy scalar issues
            overall_metrics['avg_accuracy'] = float(np.mean([r['accuracy'] for r in successful_results]))
            overall_metrics['avg_f1_macro'] = float(np.mean([r['f1_macro'] for r in successful_results]))

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

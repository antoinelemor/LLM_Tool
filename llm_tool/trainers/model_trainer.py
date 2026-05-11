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
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
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
    TwitterXLMRobertaBase,
    LongT5Base, LongT5TGlobalBase, get_model_class_for_name
)
from llm_tool.utils.training_paths import (
    get_training_logs_base,
    get_training_metrics_dir,
    resolve_metrics_base_dir,
)
from .multilingual_selector import MultilingualModelSelector
from ..utils.data_filter_logger import get_filter_logger
from ..utils.system_resources import detect_resources
from .data_utils import safe_tolist, safe_convert_labels


if TYPE_CHECKING:
    from ..utils.system_resources import SystemResources


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
    "cardiffnlp/twitter-xlm-roberta-base": None,
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
            logger.info(f" Using confirmed languages: {', '.join(detected_languages)}")

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
            logger.info(f" Detected languages from samples: {', '.join(detected_languages)}")

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
            logger.info(f" Detected languages from language lists: {', '.join(detected_languages)}")

    # Set on model if we detected any languages
    if detected_languages:
        model.detected_languages = detected_languages
        if logger:
            logger.info(f"✓ Set model.detected_languages = {detected_languages}")
    elif logger:
        logger.warning("[!]  No languages detected in data")

    return detected_languages


def filter_long_texts_in_dataframe(
    df: 'pd.DataFrame',
    *,
    text_column: str,
    model_name: str,
    max_length: int,
    logger: Optional[logging.Logger] = None,
    split_name: str = '',
) -> 'pd.DataFrame':
    """Drop rows whose tokenised length exceeds ``max_length``.

    Centralised helper used by every training path that needs to honour
    ``TrainingConfig.exclude_long_texts``. Loads the model's tokenizer once,
    batch-encodes the text column, and returns a filtered DataFrame. Raises
    on tokenizer load failure rather than silently keeping long samples.

    Parameters
    ----------
    df:
        DataFrame to filter (modified copy is returned; input not mutated).
    text_column:
        Column holding the raw text.
    model_name:
        HF model id whose tokenizer measures the true token length.
    max_length:
        Maximum tokenised length to keep.
    logger:
        Optional logger; falls back to module logger.
    split_name:
        Short label (e.g. ``"train"``) used in the INFO log line.
    """
    if not model_name or max_length is None:
        return df
    log = logger if logger is not None else logging.getLogger(__name__)
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"exclude_long_texts was requested but the tokenizer for "
            f"{model_name!r} could not be loaded ({exc}). Refusing to "
            f"keep the dataset without applying the filter."
        ) from exc
    texts = df[text_column].astype(str).tolist()
    if not texts:
        return df
    encoded = tok(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    lengths = np.asarray([len(ids) for ids in encoded['input_ids']])
    keep_mask = lengths <= int(max_length)
    n_before = len(df)
    n_dropped = int((~keep_mask).sum())
    if n_dropped > 0:
        longest_dropped = int(lengths[~keep_mask].max())
        pct = 100.0 * n_dropped / max(n_before, 1)
        log.info(
            "Exclude-long-texts [%s]: dropped %d/%d samples (%.2f%%) "
            "exceeding %d tokens (longest dropped: %d).",
            split_name or 'split', n_dropped, n_before, pct,
            int(max_length), longest_dropped,
        )
    return df.loc[keep_mask].reset_index(drop=True)


def split_long_texts_in_dataframe(
    df: 'pd.DataFrame',
    *,
    text_column: str,
    model_name: str,
    max_length: int,
    logger: Optional[logging.Logger] = None,
    split_name: str = '',
    id_column: Optional[str] = None,
) -> 'pd.DataFrame':
    """Chunk rows whose tokenised length exceeds ``max_length`` into multiple
    rows of at most ``max_length`` tokens each.

    Each chunk inherits all non-text columns from the parent row, including
    labels. Texts that already fit are kept untouched. When ``id_column`` is
    provided and present in ``df``, chunk IDs are suffixed with ``_chunk_{i}``
    so downstream code can still trace a chunk back to its source row.

    Implementation detail: the helper uses the *content* tokenisation
    (``add_special_tokens=False``) to decide where to cut, then reserves two
    slots in the budget for the CLS/SEP tokens that BertBase adds back during
    its own encode step. So the chunk size we actually emit is
    ``max_length - 2``. If, after re-tokenising the decoded chunk, BertBase
    still finds it over budget, the runtime warning in
    ``BertBase._prepare_inputs`` will flag it as a final safety net.

    Raises
    ------
    RuntimeError
        If the tokenizer cannot be loaded. We never silently fall back —
        unchunked oversize samples would re-introduce the truncation bug.
    """
    if not model_name or max_length is None:
        return df
    if max_length < 8:
        raise ValueError(f"max_length must be >= 8 for chunking, got {max_length}")
    log = logger if logger is not None else logging.getLogger(__name__)
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"split_long_texts was requested but the tokenizer for "
            f"{model_name!r} could not be loaded ({exc}). Refusing to "
            f"keep the dataset without applying the chunker."
        ) from exc

    chunk_size = int(max_length) - 2  # leave 2 slots for CLS/SEP at encode time

    new_rows = []
    n_parents_chunked = 0
    n_chunks_emitted = 0
    longest_seen = 0
    id_col = id_column if id_column and id_column in df.columns else None

    for _, row in df.iterrows():
        text = row[text_column]
        if not isinstance(text, str) or not text.strip():
            new_rows.append(row)
            continue
        ids = tok.encode(text, add_special_tokens=False, truncation=False)
        if len(ids) <= chunk_size:
            new_rows.append(row)
            continue
        if len(ids) > longest_seen:
            longest_seen = len(ids)
        n_parents_chunked += 1
        # Slice in disjoint chunks (no overlap by default).
        for chunk_idx, start in enumerate(range(0, len(ids), chunk_size)):
            chunk_ids = ids[start:start + chunk_size]
            chunk_text = tok.decode(chunk_ids, skip_special_tokens=True).strip()
            if not chunk_text:
                continue
            new_row = row.copy()
            new_row[text_column] = chunk_text
            if id_col is not None:
                base_id = row[id_col]
                new_row[id_col] = f"{base_id}_chunk_{chunk_idx}"
            new_rows.append(new_row)
            n_chunks_emitted += 1

    if n_parents_chunked > 0:
        import pandas as _pd_local  # noqa: F401
        log.info(
            "Split-long-texts [%s]: chunked %d parent row(s) into %d chunks "
            "(max_length=%d, longest parent=%d tokens).",
            split_name or 'split', n_parents_chunked, n_chunks_emitted,
            int(max_length), longest_seen,
        )

    return df.__class__(new_rows).reset_index(drop=True)


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
    # Multi-label classification parameters
    multi_label: bool = False  # Enable true multi-label classification (BCEWithLogitsLoss + sigmoid)
    multi_label_threshold: float = 0.5  # Threshold for multi-label predictions
    # When True, samples whose tokenized length exceeds ``max_length`` are dropped
    # at load time (using the actual tokenizer of ``model_name``). Only honoured by
    # the multi-label path that routes through MultiLabelTrainer.load_multi_label_data.
    exclude_long_texts: bool = False
    # When True, samples whose tokenized length exceeds ``max_length`` are
    # split into multiple chunks (with the actual tokenizer of ``model_name``).
    # Each chunk inherits the parent's labels. Mutually exclusive with
    # ``exclude_long_texts``; if both flags are True, split wins.
    split_long_texts: bool = False


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
    class_names: Optional[List[str]] = None
    global_completed_epochs: Optional[int] = None


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
        self._config_provided = config is not None
        self.logger = logging.getLogger(__name__)
        self.resource_recommendations: Dict[str, Any] = {}

        resources = self._try_detect_resources()
        preferred_device = self._apply_resource_recommendations(resources)
        self.device = self._select_device(preferred_device)
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
            "cardiffnlp/twitter-xlm-roberta-base": TwitterXLMRobertaBase,
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

    def _try_detect_resources(self) -> Optional["SystemResources"]:
        """Safely run system resource detection."""
        try:
            return detect_resources()
        except Exception as exc:
            self.logger.warning(
                "System resource detection failed (%s). Falling back to runtime device checks.",
                exc
            )
            return None

    def _apply_resource_recommendations(
        self,
        resources: Optional["SystemResources"]
    ) -> Optional[str]:
        """
        Apply resource-based recommendations to training configuration.

        Returns the preferred device type when available.
        """
        if resources is None:
            return None

        try:
            recommendations = resources.get_recommendation()
        except Exception as exc:
            self.logger.warning(
                "Unable to compute resource recommendations (%s). Using defaults.",
                exc
            )
            return None

        self.resource_recommendations = recommendations

        if not self._config_provided:
            default_config = TrainingConfig()
            updates: Dict[str, Any] = {}

            new_batch = recommendations.get('batch_size')
            if new_batch is not None and self.config.batch_size == default_config.batch_size:
                self.config.batch_size = new_batch
                updates['batch_size'] = new_batch

            new_grad_acc = recommendations.get('gradient_accumulation_steps')
            if new_grad_acc is not None and self.config.gradient_accumulation_steps == default_config.gradient_accumulation_steps:
                self.config.gradient_accumulation_steps = new_grad_acc
                updates['gradient_accumulation_steps'] = new_grad_acc

            use_fp16 = recommendations.get('use_fp16')
            if use_fp16 is not None and self.config.fp16 == default_config.fp16:
                self.config.fp16 = use_fp16
                updates['fp16'] = use_fp16

            new_workers = recommendations.get('num_workers')
            if (
                new_workers is not None and
                hasattr(self.config, 'num_workers') and
                self.config.num_workers == default_config.num_workers
            ):
                self.config.num_workers = new_workers
                updates['num_workers'] = new_workers

            if updates:
                self.logger.info(
                    "Applied system resource tuning to training defaults: %s",
                    updates
                )
        else:
            self.logger.debug(
                "Custom training configuration provided; skipping automatic tuning of batch settings."
            )

        return recommendations.get('device')

    def _get_model_optimized_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get optimized training configuration for a specific model.

        This method returns GPU-optimized parameters that account for both
        system resources AND the specific model size (e.g., xlm-roberta-large).

        Parameters
        ----------
        model_name : str
            Name of the model being trained

        Returns
        -------
        dict
            Optimized configuration with fp16, batch_size, gradient_accumulation_steps
        """
        try:
            from ..utils.system_resources import get_model_optimized_config
            return get_model_optimized_config(model_name)
        except Exception as exc:
            self.logger.warning(f"Could not get model-specific optimization: {exc}")
            return self.resource_recommendations

    def _apply_model_optimizations(self, model_name: str) -> None:
        """
        Apply model-specific optimizations to training config.

        This is called before training to ensure optimal GPU utilization
        for the specific model being trained.
        """
        if self._config_provided:
            # Don't override user-provided config
            self.logger.debug("User-provided config: skipping model-specific auto-tuning")
            return

        model_config = self._get_model_optimized_config(model_name)

        # Track what was changed
        changes = []

        # Apply optimizations if they differ from current config
        if model_config.get('use_fp16') and not self.config.fp16:
            self.config.fp16 = True
            changes.append(f"fp16=True")

        new_batch = model_config.get('batch_size')
        if new_batch and new_batch != self.config.batch_size:
            self.config.batch_size = new_batch
            changes.append(f"batch_size={new_batch}")

        new_grad_acc = model_config.get('gradient_accumulation_steps')
        if new_grad_acc and new_grad_acc != self.config.gradient_accumulation_steps:
            self.config.gradient_accumulation_steps = new_grad_acc
            changes.append(f"grad_acc={new_grad_acc}")

        if changes:
            effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps
            model_category = model_config.get('model_category', 'unknown')
            self.logger.info(
                f" Model-specific GPU optimization for {model_name} ({model_category}): "
                f"{', '.join(changes)} [effective batch: {effective_batch}]"
            )

    def _select_device(self, preferred: Optional[str]) -> torch.device:
        """Select the optimal torch device using recommendations and runtime checks."""
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

        if preferred == "cuda" and torch.cuda.is_available():
            # Detect if this is AMD ROCm or NVIDIA CUDA
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            gpu_type = "AMD ROCm" if is_rocm else "NVIDIA CUDA"
            self.logger.info(f"Using {gpu_type} GPU (from system resource analysis).")
            return torch.device("cuda")

        if preferred == "mps" and mps_available:
            self.logger.info("Using Apple Silicon GPU (MPS) from system resource analysis.")
            return torch.device("mps")

        if torch.cuda.is_available():
            # Detect if this is AMD ROCm or NVIDIA CUDA
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            gpu_type = "AMD ROCm" if is_rocm else "NVIDIA CUDA"
            self.logger.info(f"Using {gpu_type} GPU (auto-detected).")
            return torch.device("cuda")

        if mps_available:
            self.logger.info("Using Apple Silicon GPU (MPS).")
            return torch.device("mps")

        self.logger.info("No GPU backend detected; defaulting to CPU execution.")
        return torch.device("cpu")

    def _to_label_predictions(self, predictions: Any) -> np.ndarray:
        """Convert model outputs (probabilities/logits) to discrete label ids."""
        preds = np.asarray(predictions)

        if preds.size == 0:
            return np.array([], dtype=int)

        if preds.ndim == 1:
            if np.issubdtype(preds.dtype, np.floating):
                return (preds >= 0.5).astype(int)
            return preds.astype(int)

        if preds.ndim == 2:
            if preds.shape[1] == 0:
                return np.array([], dtype=int)
            if preds.shape[1] == 1:
                column = preds[:, 0]
                if np.issubdtype(column.dtype, np.floating):
                    return (column >= 0.5).astype(int)
                return column.astype(int)
            return np.argmax(preds, axis=1).astype(int)

        # Fallback for higher dimensional outputs: reduce across last axis.
        return np.argmax(preds, axis=-1).astype(int)

    def load_data(self, data_path: str, text_column: str = "text",
                  label_column: str = "label",
                  split_config: Optional[Dict[str, Any]] = None,
                  category_name: Optional[str] = None,
                  auto_remove_insufficient_labels: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data for training

        Args:
            data_path: Path to the data file
            text_column: Name of the text column
            label_column: Name of the label column
            split_config: Configuration for data splitting (from bundle.metadata)
            category_name: Name of the category being trained (key or key_value)
            auto_remove_insufficient_labels: If True, auto-remove labels with <2 samples
                without interactive prompt (pre-validated upstream)

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
            console.print(f"\n[yellow]Found {len(insufficient_labels)} label(s) with insufficient samples for training:[/yellow]")
            for label in insufficient_labels:
                # Find the class index for this label
                label_idx = sorted_labels.index(label)
                count = label_counts[label_idx]
                console.print(f"  • [red]'{label}'[/red]: {count} sample(s) - need at least 2 for train/test split")

            if auto_remove_insufficient_labels:
                # Auto-remove without prompting (pre-validated upstream)
                self.logger.info(f"Auto-removing {len(insufficient_labels)} insufficient label(s) (pre-validated upstream)")
                console.print(f"\n[cyan]Auto-removing insufficient labels (pre-validated upstream)[/cyan]")
                remove_labels = True
            else:
                # Interactive prompt
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

        # CRITICAL FIX: Ensure each class has at least 1 sample in validation set
        # This fixes the issue where minority classes with few samples might all end up in train
        val_label_counts = Counter(y_val)
        train_label_counts = Counter(y_train)
        all_classes = set(y_temp)

        missing_in_val = [cls for cls in all_classes if val_label_counts.get(cls, 0) == 0]
        if missing_in_val:
            self.logger.warning(
                f"[!]  Some classes have 0 samples in validation set after split. "
                f"Moving 1 sample per missing class from train to val."
            )

            # Convert to lists for manipulation
            X_train_list = list(X_train)
            y_train_list = list(y_train)
            X_val_list = list(X_val)
            y_val_list = list(y_val)
            lang_train_list = list(lang_train) if lang_train is not None else None
            lang_val_list = list(lang_val) if lang_val is not None else None

            for cls in missing_in_val:
                # Find first occurrence of this class in train
                for i, label in enumerate(y_train_list):
                    if label == cls:
                        # Move this sample from train to val
                        X_val_list.append(X_train_list.pop(i))
                        y_val_list.append(y_train_list.pop(i))
                        if lang_train_list is not None:
                            lang_val_list.append(lang_train_list.pop(i))

                        # Get class name for logging
                        cls_name = sorted_labels[int(cls)] if int(cls) < len(sorted_labels) else str(cls)
                        self.logger.info(f"   → Moved 1 '{cls_name}' sample from train to val")
                        break

            # Convert back to numpy arrays
            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)
            X_val = np.array(X_val_list)
            y_val = np.array(y_val_list)
            if lang_train_list is not None:
                lang_train = np.array(lang_train_list)
                lang_val = np.array(lang_val_list)

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
                          progress_callback: Optional[callable] = None,
                          reinforced_learning: bool = False,
                          reinforced_epochs: Optional[int] = None,
                          rl_f1_threshold: float = 0.7,
                          rl_oversample_factor: float = 2.0,
                          rl_class_weight_factor: float = 2.0,
                          is_benchmark: bool = False,
                          model_name_for_logging: Optional[str] = None,
                          training_approach: Optional[str] = None) -> TrainingResult:
        """Train a single model"""
        from tqdm import tqdm
        print(f"\n️  Training model: {model_name}")
        self.logger.info(f"Training model: {model_name} with {training_strategy} strategy")
        start_time = time.time()

        # Apply model-specific GPU optimizations (batch size, FP16, gradient accumulation)
        # This ensures optimal GPU utilization for large models like xlm-roberta-large
        self._apply_model_optimizations(model_name)

        # Get model class
        model_class = self.model_registry.get(model_name)
        if not model_class:
            # Use mapping function to get correct class for model name
            model_class = get_model_class_for_name(model_name)

        # Initialize model with resource recommendations for optimized DataLoader.
        # max_length comes from TrainingConfig (set upstream by tokenizer-aware
        # analysis or left at the dataclass default 512).
        model_instance = model_class(
            model_name=model_name,
            device=self.device,
            resource_recommendations=self.resource_recommendations,
            max_length=int(getattr(self.config, 'max_length', 512) or 512),
        )

        # ==================== LANGUAGE FILTERING FOR MONOLINGUAL MODELS ====================
        # Check if this model should only train on specific languages
        target_languages = get_model_target_languages(model_name)

        if target_languages is not None:
            # This is a language-specific model - filter data
            self.logger.info(f" Model '{model_name}' targets {target_languages}. Filtering data...")

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
                    self.logger.warning(f"[!]  Very few training samples ({len(train_df)}) for {model_name} "
                                       f"targeting {target_languages}. Training may be unstable.")
            else:
                self.logger.warning(f"[!]  No 'language' or 'lang' column found. Cannot filter for {model_name}. "
                                   f"Training on ALL data (not recommended for monolingual models).")
        else:
            # Multilingual model - use all data
            self.logger.info(f" Model '{model_name}' is multilingual. Using all language data.")

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

        # Tokenizer-aware "Split" or "Exclude" strategy at load time. This is
        # the single-label / per-language / one-vs-all binary path. Split wins
        # over Exclude when both flags are set.
        _ml_single = int(getattr(self.config, 'max_length', 512) or 512)
        if bool(getattr(self.config, 'split_long_texts', False)) and model_name:
            train_df = split_long_texts_in_dataframe(
                train_df, text_column='text', model_name=model_name,
                max_length=_ml_single, logger=self.logger, split_name='train',
            )
            val_df = split_long_texts_in_dataframe(
                val_df, text_column='text', model_name=model_name,
                max_length=_ml_single, logger=self.logger, split_name='val',
            )
            if len(test_df) > 0:
                test_df = split_long_texts_in_dataframe(
                    test_df, text_column='text', model_name=model_name,
                    max_length=_ml_single, logger=self.logger, split_name='test',
                )
            if len(train_df) == 0 or len(val_df) == 0:
                raise RuntimeError(
                    "Split-long-texts produced an empty train or val split: "
                    "aborting training."
                )
        elif bool(getattr(self.config, 'exclude_long_texts', False)) and model_name:
            train_df = filter_long_texts_in_dataframe(
                train_df, text_column='text', model_name=model_name,
                max_length=_ml_single, logger=self.logger, split_name='train',
            )
            val_df = filter_long_texts_in_dataframe(
                val_df, text_column='text', model_name=model_name,
                max_length=_ml_single, logger=self.logger, split_name='val',
            )
            if len(test_df) > 0:
                test_df = filter_long_texts_in_dataframe(
                    test_df, text_column='text', model_name=model_name,
                    max_length=_ml_single, logger=self.logger, split_name='test',
                )
            if len(train_df) == 0 or len(val_df) == 0:
                raise RuntimeError(
                    "Exclude-long-texts left an empty train or val split: "
                    "either lower max_length, increase the data, or pick a "
                    "different strategy at the Token Length step."
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
            # CRITICAL FIX: Create unified label mapping from ALL labels (train + val + test)
            # This ensures consistent label indices across all data splits.
            # Without this, different splits could have different label-to-index mappings
            # if they have different label distributions, causing evaluation failures.
            all_labels = list(set(train_labels + val_labels + test_labels))
            # Sort to ensure consistent ordering: NOT_* labels first, then alphabetical
            sorted_labels = sorted(all_labels, key=lambda x: (not str(x).startswith('NOT_'), str(x)))
            unified_label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}

            self.logger.info(f"Created unified label mapping with {len(unified_label_mapping)} classes")

            # Encode data with unified label mapping
            train_dataloader = model_instance.encode(
                train_texts, train_labels,
                batch_size=self.config.batch_size,
                progress_bar=True,
                shuffle=True,  # Shuffle training data
                label_mapping=unified_label_mapping
            )

            val_dataloader = model_instance.encode(
                val_texts, val_labels,
                batch_size=self.config.batch_size,
                progress_bar=False,
                shuffle=False,  # Don't shuffle validation data
                label_mapping=unified_label_mapping
            )

            test_dataloader = model_instance.encode(
                test_texts, test_labels,
                batch_size=self.config.batch_size,
                progress_bar=False,
                shuffle=False,  # Don't shuffle test data
                label_mapping=unified_label_mapping
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
            elif class_names_for_display is None and category_name and num_labels == 2:
                # One-vs-all binary classification: use category_name to construct proper names
                # ONLY when label encoder didn't provide class names (true binary presence/absence)
                # Class 0 = NOT_category (negative), Class 1 = category (positive)
                class_names_for_display = [f"NOT_{category_name}", category_name]
                self.logger.info(f"️  One-vs-all class names: {class_names_for_display}")
            elif class_names_for_display is not None:
                class_names_for_display = [str(name) for name in class_names_for_display]

            if class_names_for_display is not None and num_labels is not None:
                if len(class_names_for_display) != num_labels:
                    self.logger.warning(
                        f"[!] class_names length ({len(class_names_for_display)}) does not match num_labels ({num_labels}). "
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
                    self.logger.warning(f"  [!] NUMPY TYPE DETECTED: {key} = type={type(value)}, value={value}")
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
                progress_callback=progress_callback,
                reinforced_learning=reinforced_learning,
                reinforced_epochs=reinforced_epochs,
                reinforced_f1_threshold=rl_f1_threshold,
                is_benchmark=is_benchmark,
                model_name_for_logging=model_name_for_logging,
                # GPU Optimization parameters
                fp16=self.config.fp16,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                # Multi-label classification parameters
                multi_label=self.config.multi_label,
                multi_label_threshold=self.config.multi_label_threshold,
                # CRITICAL: Pass training_approach for correct chart labeling
                training_approach=training_approach,
            )

            # Capture global_completed_epochs from model_instance BEFORE it goes out of scope
            final_global_completed_epochs = getattr(model_instance, 'global_completed_epochs', None)

            # Log final model path
            if best_model_path:
                self.logger.info(f" Training complete. Best model at: {best_model_path}")
            else:
                self.logger.warning(f"[!] Training complete but no model path returned")

            # FINAL EVALUATION: Use test set if available, otherwise use validation metrics
            if len(test_df) > 0:
                # Evaluate on separate test set (final unbiased evaluation)
                self.logger.info(f" Final evaluation on test set ({len(test_df)} samples)...")
                test_probs = model_instance.predict(test_dataloader, model_instance.model, proba=True)
                test_predictions = self._to_label_predictions(test_probs)

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
                classes = getattr(self.label_encoder, "classes_", None)
                label_ids = list(range(len(classes))) if classes is not None else []
                target_names_override = None
                if class_names_for_display and label_ids and len(class_names_for_display) == len(label_ids):
                    target_names_override = [str(name) for name in class_names_for_display]
                report = classification_report(
                    test_labels,
                    test_predictions,
                    labels=label_ids if label_ids else None,
                    target_names=target_names_override or ([str(c) for c in classes] if classes is not None else None),
                    output_dict=True,
                    zero_division=0
                )
            else:
                # No separate test set - use validation metrics from training
                self.logger.info(f" Using validation metrics as final evaluation (no separate test set)")
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

            # Extract best_epoch from training summary
            training_summary = getattr(model_instance, 'last_training_summary', {})
            best_epoch = training_summary.get('best_epoch', 0) if training_summary else 0

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
                best_epoch=best_epoch,
                validation_loss=training_summary.get('val_loss', 0) if training_summary else 0,
                training_history=[],  # TODO: Extract from training logs
                model_path=best_model_path if best_model_path else str(output_dir)  # CRITICAL: Use actual saved model path
            )

            if class_names_for_display:
                result.class_names = [str(name) for name in class_names_for_display]

            # Attach global_completed_epochs to result for propagation back to caller
            result.global_completed_epochs = final_global_completed_epochs

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

        # Save benchmark report - use session-aware path via resolve_metrics_base_dir
        metrics_base = resolve_metrics_base_dir(self.config.metrics_output_dir)
        report_path = metrics_base / "benchmark_report.json"
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

        # CRITICAL: Check training_approach to distinguish between one-vs-all and true multi-label
        # - training_approach == 'one-vs-all': Multiple binary classifiers (use _train_multi_label)
        # - training_approach == 'multi-label': Single model with BCEWithLogitsLoss (use MultiLabelTrainer)
        training_approach = config.get('training_approach', '')

        # DEBUG: Log routing decision
        self.logger.info(f"[ROUTING] training_strategy={training_strategy}, training_approach={training_approach}")
        self.logger.info(f"[ROUTING] has_training_files={bool(config.get('training_files'))}, has_input_file={bool(config.get('input_file'))}")

        # Route based on explicit strategy first
        if training_strategy == 'multi-label':
            # Check if we have multi-label data files
            if 'training_files' in config and config['training_files']:
                # CRITICAL: Only use binary classifier approach for one-vs-all
                # For true multi-label, use the single model path even if training_files exists
                if training_approach == 'one-vs-all':
                    self.logger.info("[ROUTING] → Using _train_multi_label (one-vs-all binary classifiers)")
                    return self._train_multi_label(config)
                elif 'input_file' in config:
                    # True multi-label: use single file (JSONL) with all labels
                    self.logger.info(f"[ROUTING] → Using TRUE multi-label with BCEWithLogitsLoss (single model)")
                    data_path = config['input_file']
                else:
                    # Fallback to _train_multi_label for backward compatibility
                    # but only if training_approach is NOT explicitly 'multi-label'
                    if training_approach != 'multi-label':
                        self.logger.info("[ROUTING] → Fallback to _train_multi_label (backward compatibility)")
                        return self._train_multi_label(config)
                    else:
                        raise ValueError("True multi-label training requires input_file with combined JSONL data")
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
            # Check if this is TRUE multi-label (one model with sigmoid) or one-vs-all (multiple binary models)
            training_approach = config.get('training_approach', '')

            if training_approach == 'multi-label':
                # TRUE MULTI-LABEL: One model with BCEWithLogitsLoss + sigmoid
                # Each text can have 0, 1, or many labels simultaneously
                return self._train_true_multi_label(config, data_path, text_column, label_column, model_name)

            # Otherwise, use MultiLabelTrainer for one-vs-all or multiclass
            from .multi_label_trainer import MultiLabelTrainer

            print("\n️ Training multi-label model...")
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

            # Import language filtering utilities
            from llm_tool.utils.language_filtering import (
                filter_languages_with_sufficient_samples,
                create_training_report,
                save_filtered_languages_log
            )

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

            # Apply language filtering for insufficient samples
            if lang_col is not None and config.get('enable_language_filtering', True):
                # Check if we need to filter languages with insufficient samples
                min_samples_per_class = config.get('min_samples_per_class', 2)
                min_train_samples = config.get('min_train_samples', 1)
                min_val_samples = config.get('min_val_samples', 1)

                # Convert labels to strings for consistency
                # Handle multi-label case: if label is a list, use the first element
                # This prevents issues where str(['a', 'b']) becomes "['a', 'b']"
                def label_to_str(label):
                    if isinstance(label, (list, tuple, np.ndarray)):
                        if len(label) > 0:
                            return str(label[0])
                        return ''
                    return str(label)
                y_str = [label_to_str(label) for label in y]

                # Apply language filtering
                filtered_texts, filtered_labels, filtered_languages, filtering_report = \
                    filter_languages_with_sufficient_samples(
                        texts=X.tolist(),
                        labels=y_str,
                        languages=lang_col.tolist() if lang_col is not None else None,
                        min_samples_per_class=min_samples_per_class,
                        min_train_samples=min_train_samples,
                        min_val_samples=min_val_samples
                    )

                # Log filtering results
                if filtering_report['filtered']:
                    classes_dropped = filtering_report.get('classes_dropped', [])
                    if classes_dropped:
                        self.logger.warning(f"Class filtering applied for {label_column}")
                        self.logger.warning(f"  - Classes dropped (insufficient samples): {classes_dropped}")
                        self.logger.warning(f"  - Languages kept: {filtering_report['languages_kept']}")
                    else:
                        self.logger.warning(f"Data filtering applied for {label_column}")
                        self.logger.warning(f"  - Languages kept: {filtering_report['languages_kept']}")
                    self.logger.warning(f"  - Samples: {filtering_report['total_samples_before']} → {filtering_report['total_samples_after']}")

                    # Save filtering log
                    session_id = config.get('session_id', 'default')
                    output_dir = config.get('output_dir', 'models')
                    save_filtered_languages_log(
                        session_id=session_id,
                        category=label_column,
                        languages_dropped=filtering_report['languages_dropped'],
                        drop_reasons=filtering_report['drop_reasons'],
                        output_dir=output_dir
                    )

                    # Update data with filtered results
                    X = np.array(filtered_texts)
                    y = np.array(filtered_labels)
                    lang_col = np.array(filtered_languages) if filtered_languages else None

                    # Store filtering report in config for later use
                    config['language_filtering_report'] = filtering_report

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
            # Propagate tokenizer-aware settings so MultiLabelTrainer can honor them.
            # exclude_long_texts is the user's "Exclude" strategy from the Token
            # Length step; max_length feeds both BertBase tokenization and the
            # filter threshold. Without forwarding these, MultiLabelTrainer would
            # silently use its dataclass defaults and re-introduce the truncation bug.
            ml_config.max_length = int(getattr(self.config, 'max_length', 512) or 512)
            ml_config.exclude_long_texts = bool(getattr(self.config, 'exclude_long_texts', False))
            ml_config.split_long_texts = bool(getattr(self.config, 'split_long_texts', False))
            # CRITICAL: Enable true multi-label classification with BCEWithLogitsLoss + sigmoid
            ml_config.multi_label = True
            ml_config.multi_label_threshold = config.get('multi_label_threshold', 0.5)
            # Distribution-aware training parameters
            ml_config.distribution_aware = config.get('distribution_aware', True)
            ml_config.use_asymmetric_loss = config.get('use_asymmetric_loss', True)
            # Reinforced learning parameters
            ml_config.reinforced_learning = config.get('reinforced_learning', False)
            ml_config.rl_f1_threshold = config.get('rl_f1_threshold', 0.7)
            # Pass manual reinforced epochs if provided
            if 'reinforced_epochs' in config and config['reinforced_epochs'] is not None:
                ml_config.reinforced_epochs = config['reinforced_epochs']
            # Resume at RL Phase 2
            ml_config.skip_to_rl = config.get('skip_to_rl', False)
            ml_config.rl_state_path = config.get('rl_state_path')

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
                progress_callback=config.get('progress_callback'),  # CRITICAL: Pass progress callback for epoch tracking
                training_approach=config.get('training_approach'),  # CRITICAL: Pass training approach for correct chart labeling
            )

            # CRITICAL: Calculate total training time
            total_training_time = time.time() - training_start_time

            # Aggregate results - CRITICAL: Handle partial success
            # Even if some models fail, we should return metrics for successful ones
            if trained_models:
                # CRITICAL: Convert np.mean results to Python floats to avoid numpy scalar issues
                # Only average over models that have non-zero metrics (successful training)
                successful_models = [m for m in trained_models.values()
                                   if m.performance_metrics.get('f1_macro', 0) > 0
                                   or m.performance_metrics.get('accuracy', 0) > 0]

                if successful_models:
                    avg_f1 = float(np.mean([m.performance_metrics.get('f1_macro', 0) for m in successful_models]))
                    avg_acc = float(np.mean([m.performance_metrics.get('accuracy', 0) for m in successful_models]))

                    self.logger.info(f"Multi-label training complete!")
                    self.logger.info(f"Models trained: {len(successful_models)}/{len(trained_models)} successful")
                else:
                    # All models failed but we still have model objects
                    avg_f1 = 0.0
                    avg_acc = 0.0
                    self.logger.warning("All models failed training - returning zero metrics")

                # Extract per-category metrics for detailed reporting (especially for benchmark)
                category_metrics = {}
                for model_name, model_info in trained_models.items():
                    # Include all models, even failed ones (with 0 metrics)
                    category_metrics[model_name] = {
                        'f1_macro': model_info.performance_metrics.get('f1_macro', 0),
                        'accuracy': model_info.performance_metrics.get('accuracy', 0),
                        'precision': model_info.performance_metrics.get('precision', 0),
                        'recall': model_info.performance_metrics.get('recall', 0),
                        'model_path': model_info.model_path if hasattr(model_info, 'model_path') else ''
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
                category_name=category_name,
                auto_remove_insufficient_labels=config.get('auto_remove_insufficient_labels', False)
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
                self.logger.warning(f"[!]  models_by_language has invalid type: {type(models_by_language)} (value: {models_by_language})")
                self.logger.warning("   Expected dict, got scalar or other type. Disabling per-language training.")
                models_by_language = None

            if models_by_language:
                # Train separate model for each language
                self.logger.info(f" Training {len(models_by_language)} language-specific models")

                # Check if language column exists (can be 'language' or 'lang')
                lang_col_name = None
                if 'language' in train_df.columns:
                    lang_col_name = 'language'
                    self.logger.info(f"✓ Found 'language' column for filtering")
                elif 'lang' in train_df.columns:
                    lang_col_name = 'lang'
                    self.logger.info(f"✓ Found 'lang' column for filtering")

                if not lang_col_name:
                    self.logger.error(" Cannot train per-language models: No 'language' or 'lang' column found in data")
                    raise ValueError("Per-language training requested but no language column found in data")

                # Train a model for each language
                language_results = {}
                total_training_time = 0

                for lang_code, lang_model in models_by_language.items():
                    self.logger.info(f"\n️  Training {lang_code} model: {lang_model}")

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
                        self.logger.warning(f"[!]  Skipping {lang_code}: Insufficient training samples ({len(train_df_lang)})")
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
                                    f"  [!] Label index {old_label} out of range for class names (len={len(all_class_names)}). "
                                    "Using fallback placeholder."
                                )
                                class_names_lang.append(f"Class {len(class_names_lang)}")
                    override_names = config.get('class_names_override')
                    if override_names and isinstance(override_names, (list, tuple)) and len(override_names) >= len(unique_labels_lang):
                        mapped_override = []
                        for old_label in unique_labels_lang:
                            if isinstance(old_label, (int, np.integer)) and 0 <= old_label < len(override_names):
                                mapped_override.append(str(override_names[int(old_label)]))
                            else:
                                fallback_name = (
                                    class_names_lang[unique_labels_lang.index(old_label)]
                                    if class_names_lang and unique_labels_lang.index(old_label) < len(class_names_lang)
                                    else f"Class {len(mapped_override)}"
                                )
                                mapped_override.append(fallback_name)
                        class_names_lang = mapped_override
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
                            rl_class_weight_factor=config.get('rl_class_weight_factor', 2.0),
                            is_benchmark=config.get('is_benchmark', False),
                            model_name_for_logging=lang_model
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
                        self.logger.error(f" Failed to train {lang_code} model: {str(e)}", exc_info=True)
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
                    progress_callback=config.get('progress_callback'),
                    reinforced_learning=config.get('reinforced_learning', False),
                    reinforced_epochs=config.get('reinforced_epochs'),
                    rl_f1_threshold=config.get('rl_f1_threshold', 0.7),
                    rl_oversample_factor=config.get('rl_oversample_factor', 2.0),
                    rl_class_weight_factor=config.get('rl_class_weight_factor', 2.0),
                    class_names_override=config.get('class_names_override'),
                    is_benchmark=config.get('is_benchmark', False),
                    model_name_for_logging=config.get('model_name'),
                    training_approach=config.get('training_approach')
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
                'language_results': results.get('language_results', {}),
                # Global progress tracking (CRITICAL for multi-model training)
                'global_completed_epochs': results.get('global_completed_epochs') if isinstance(results, dict) else None,
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
                    'class_names': getattr(results, 'class_names', None),
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
                    },
                    # Global progress tracking (CRITICAL for multi-model training)
                    'global_completed_epochs': getattr(results, 'global_completed_epochs', None),
                }
            else:
                # results is already a dictionary
                return {
                    'best_model': model_name,
                    'best_f1_macro': results.get('f1', results.get('f1_macro', 0.0)),
                    'accuracy': results.get('accuracy', 0.0),
                    'model_path': results.get('model_path', ''),
                    'training_time': results.get('training_time', 0),
                    'class_names': results.get('class_names'),
                    'metrics': results,
                    # Global progress tracking (CRITICAL for multi-model training)
                    'global_completed_epochs': results.get('global_completed_epochs'),
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
        # NOTE: batch_size is already optimized by get_model_optimized_config() based on
        # system resources and model size - DO NOT override it here
        params['batch_size'] = self.config.batch_size
        params['learning_rate'] = self.config.learning_rate
        params['num_epochs'] = self.config.num_epochs
        params['gradient_accumulation_steps'] = self.config.gradient_accumulation_steps

        # Adjust learning rate for model size (batch_size already optimized by system detection)
        if 'large' in model_name.lower():
            params['learning_rate'] *= 0.5
        elif 'xlarge' in model_name.lower() or 'xxlarge' in model_name.lower():
            params['learning_rate'] *= 0.25

        # Adjust for long-context models with short texts (batch_size controlled by system detection)
        if any(model in model_name.lower() for model in ['longformer', 'bigbird']):
            if avg_text_length < 512:  # Short texts
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

        # Adjust epochs for small datasets (batch_size controlled by system detection)
        # More epochs compensate for fewer gradient updates with small datasets
        if train_size < 100:
            params['num_epochs'] = min(20, params['num_epochs'] * 2)
        elif train_size < 500:
            params['num_epochs'] = int(params['num_epochs'] * 1.5)

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
            model = model_class(
                model_name=model_name,
                device=self.device,
                resource_recommendations=self.resource_recommendations,
                max_length=int(getattr(self.config, 'max_length', 512) or 512),
            )
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
                self.logger.warning(f"  [!] NUMPY TYPE DETECTED: {key} = type={type(value)}, value={value}")
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
            global_start_time=global_start_time,
            # GPU Optimization parameters
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # Multi-label classification parameters
            multi_label=self.config.multi_label,
            multi_label_threshold=self.config.multi_label_threshold,
            # CRITICAL: Pass training_approach for correct chart labeling
            training_approach=config.get('training_approach'),
        )

        # Log final model path
        if best_model_path:
            self.logger.info(f" Training complete. Best model at: {best_model_path}")
        else:
            self.logger.warning(f"[!] Training complete but no model path returned")

        # Evaluate on test set
        test_probs = model.predict(test_loader, model.model, proba=True)
        test_predictions = self._to_label_predictions(test_probs)

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

        label_encoder = getattr(model, "label_encoder", None)
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            label_ids = list(range(len(label_encoder.classes_)))
            report = classification_report(
                test_labels,
                test_predictions,
                labels=label_ids if label_ids else None,
                target_names=[str(c) for c in label_encoder.classes_] if label_ids else None,
                output_dict=True,
                zero_division=0
            )
        else:
            report = classification_report(
                test_labels,
                test_predictions,
                output_dict=True,
                zero_division=0
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

    def _train_true_multi_label(self, config: Dict[str, Any], data_path: str,
                                 text_column: str, label_column: str, model_name: str) -> Dict[str, Any]:
        """
        TRUE Multi-Label Training: Single model with BCEWithLogitsLoss + sigmoid

        This creates ONE model that can predict multiple labels per text simultaneously.
        Each label is treated as an independent binary classification (sigmoid output).

        Key differences from one-vs-all:
        - ONE model with num_labels = total number of unique labels
        - BCEWithLogitsLoss for training (not CrossEntropyLoss)
        - Sigmoid activation (not softmax) - labels are independent
        - Multi-hot label encoding (not one-hot)
        - Threshold-based prediction (default 0.5)

        Args:
            config: Training configuration dict
            data_path: Path to JSONL/CSV data file
            text_column: Name of text column
            label_column: Name of label column (can contain lists for multi-label)
            model_name: Model identifier to train

        Returns:
            Dict with training results
        """
        import ast
        import numpy as np
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        from .bert_base import BertBase
        from .sota_models import get_model_class_for_name

        self.logger.info("=" * 80)
        self.logger.info("️ TRUE MULTI-LABEL TRAINING")
        self.logger.info("=" * 80)
        self.logger.info(f"  Model: {model_name}")
        self.logger.info(f"  Data: {data_path}")
        self.logger.info(f"  Mode: Single model with BCEWithLogitsLoss + sigmoid")
        print("\n️ Starting TRUE multi-label training (single model, multiple labels per text)")

        # ========== STEP 1: Load Data ==========
        if data_path.endswith('.jsonl'):
            df = pd.read_json(data_path, lines=True)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            # Convert string representations of lists back to lists
            if label_column in df.columns:
                df[label_column] = df[label_column].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
                )
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Validate columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")

        self.logger.info(f"  Loaded {len(df)} samples")

        # ========== STEP 1a: Optional tokenizer-aware Split or Exclude ==========
        # Split wins if both flags are set: chunking preserves more data.
        _ml = int(getattr(self.config, 'max_length', 512) or 512)
        if bool(getattr(self.config, 'split_long_texts', False)) and model_name:
            df = split_long_texts_in_dataframe(
                df,
                text_column=text_column,
                model_name=model_name,
                max_length=_ml,
                logger=self.logger,
                split_name='true multi-label',
            )
            if len(df) == 0:
                raise RuntimeError(
                    "Split-long-texts produced an empty dataframe: aborting training."
                )
        elif bool(getattr(self.config, 'exclude_long_texts', False)) and model_name:
            df = filter_long_texts_in_dataframe(
                df,
                text_column=text_column,
                model_name=model_name,
                max_length=_ml,
                logger=self.logger,
                split_name='true multi-label',
            )
            if len(df) == 0:
                raise RuntimeError(
                    "Exclude-long-texts removed every row: aborting training."
                )

        # ========== STEP 1b: Detect Language Column ==========
        language_column = None
        track_languages = False
        if 'language' in df.columns:
            language_column = 'language'
            track_languages = True
            self.logger.info(f"  ✓ Found 'language' column. Will track per-language metrics.")
        elif 'lang' in df.columns:
            language_column = 'lang'
            track_languages = True
            self.logger.info(f"  ✓ Found 'lang' column. Will track per-language metrics.")

        # ========== STEP 2: Extract All Unique Labels ==========
        # Labels can be: single string, single int, or list of strings/ints
        all_labels = set()
        for labels in df[label_column]:
            if isinstance(labels, (list, tuple, np.ndarray)):
                all_labels.update(str(l) for l in labels)
            else:
                all_labels.add(str(labels))

        # Sort labels for consistent ordering
        all_labels = sorted(all_labels)
        num_labels = len(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        self.logger.info(f"  Found {num_labels} unique labels:")
        for label in all_labels[:10]:  # Show first 10
            self.logger.info(f"    - {label}")
        if num_labels > 10:
            self.logger.info(f"    ... and {num_labels - 10} more")

        print(f"   {num_labels} unique labels found")

        # ========== STEP 3: Create Multi-Hot Label Vectors ==========
        def labels_to_multi_hot(labels, label_to_idx, num_labels):
            """Convert label(s) to multi-hot vector"""
            multi_hot = np.zeros(num_labels, dtype=np.float32)
            if isinstance(labels, (list, tuple, np.ndarray)):
                for label in labels:
                    idx = label_to_idx.get(str(label))
                    if idx is not None:
                        multi_hot[idx] = 1.0
            else:
                idx = label_to_idx.get(str(labels))
                if idx is not None:
                    multi_hot[idx] = 1.0
            return multi_hot

        # Convert all labels to multi-hot
        multi_hot_labels = np.array([
            labels_to_multi_hot(labels, label_to_idx, num_labels)
            for labels in df[label_column]
        ])

        texts = df[text_column].values

        # ========== STEP 4: Train/Val Split ==========
        val_size = config.get('validation_split', 0.2)
        random_state = config.get('random_state', 42)

        # Include language info in split if available
        if track_languages and language_column:
            languages = df[language_column].values
            X_train, X_val, y_train, y_val, lang_train, lang_val = train_test_split(
                texts, multi_hot_labels, languages,
                test_size=val_size,
                random_state=random_state
            )
            val_language_info = lang_val.tolist()
            # Set detected languages on model
            detected_languages = sorted(list(set(str(l).upper() for l in languages if pd.notna(l) and str(l).strip())))
            self.logger.info(f"  ✓ Languages detected: {detected_languages}")
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                texts, multi_hot_labels,
                test_size=val_size,
                random_state=random_state
            )
            val_language_info = None
            detected_languages = None

        self.logger.info(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}")

        # ========== STEP 5: Initialize Model ==========
        model_class = get_model_class_for_name(model_name)
        model = model_class(
            model_name=model_name,
            device=self.device,
            resource_recommendations=self.resource_recommendations,
            max_length=int(getattr(self.config, 'max_length', 512) or 512),
        )

        # Set num_labels for multi-label
        model.num_labels = num_labels

        # ========== STEP 6: Tokenize and Create DataLoaders ==========
        # Tokenize texts
        train_inputs, train_masks = model._prepare_inputs(
            X_train.tolist(),
            add_special_tokens=True,
            progress_bar=True
        )
        val_inputs, val_masks = model._prepare_inputs(
            X_val.tolist(),
            add_special_tokens=True,
            progress_bar=False
        )

        # Create TensorDatasets with multi-hot labels (2D tensor)
        train_dataset = TensorDataset(
            torch.tensor(train_inputs, dtype=torch.long),
            torch.tensor(train_masks, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32)  # CRITICAL: float32 for BCEWithLogitsLoss
        )
        val_dataset = TensorDataset(
            torch.tensor(val_inputs, dtype=torch.long),
            torch.tensor(val_masks, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.float32)
        )

        batch_size = config.get('batch_size', self.config.batch_size)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # ========== STEP 7: Set Model Label Mapping ==========
        # Store label mapping on model for inference
        model.dict_labels = label_to_idx
        model.idx_to_label = idx_to_label
        model.label_encoder = type('LabelEncoder', (), {
            'classes_': all_labels,
            'inverse_transform': lambda self, indices: [idx_to_label[i] for i in indices]
        })()

        # ========== STEP 8: Train with BCEWithLogitsLoss ==========
        metrics_base_dir = resolve_metrics_base_dir(getattr(self.config, "metrics_output_dir", None))
        multi_label_threshold = config.get('multi_label_threshold', 0.5)

        import time as time_module
        global_start_time = time_module.time()

        # ASL (Asymmetric Loss) - SOTA for multi-label classification
        use_asymmetric_loss = config.get('use_asymmetric_loss', True)  # Default to SOTA
        asl_gamma_neg = config.get('asl_gamma_neg', 4.0)
        asl_gamma_pos = config.get('asl_gamma_pos', 1.0)
        asl_clip = config.get('asl_clip', 0.05)
        optimize_thresholds = config.get('optimize_thresholds', True)  # Per-class threshold optimization

        loss_type = "ASL (SOTA)" if use_asymmetric_loss else "BCE"
        self.logger.info(f"   Starting training with multi_label=True, threshold={multi_label_threshold}, loss={loss_type}")
        print(f"   Training with {loss_type} (threshold={multi_label_threshold})")

        # Set detected languages on model for per-language tracking
        if track_languages and detected_languages:
            model.detected_languages = detected_languages
            self.logger.info(f"  ✓ Set detected_languages on model: {detected_languages}")

        # Get global progress tracking from config (passed from training arena)
        config_global_total_models = config.get('global_total_models', 1)
        config_global_current_model = config.get('global_current_model', 1)
        config_global_total_epochs = config.get('global_total_epochs', self.config.num_epochs)
        config_global_max_epochs = config.get('global_max_epochs', self.config.num_epochs)
        config_global_completed_epochs = config.get('global_completed_epochs', 0)

        best_metric, best_model_path, best_scores = model.run_training(
            train_dataloader=train_dataloader,
            test_dataloader=val_dataloader,
            n_epochs=self.config.num_epochs,
            lr=self.config.learning_rate,
            save_model_as='model',
            metrics_output_dir=str(metrics_base_dir),
            track_languages=track_languages,  # Enable per-language tracking if language column exists
            language_info=val_language_info,  # Pass validation language info for per-language metrics
            session_id=config.get('session_id'),
            # CRITICAL: Pass label names for display and CSV logging
            class_names=all_labels,  # List of label names in order (index 0 = all_labels[0], etc.)
            label_key=config.get('label_column', 'labels'),  # For logging purposes
            category_name=config.get('category_name'),  # Actual category name (e.g., 'political_parties_long')
            global_total_models=config_global_total_models,
            global_current_model=config_global_current_model,
            global_total_epochs=config_global_total_epochs,
            global_max_epochs=config_global_max_epochs,
            global_completed_epochs=config_global_completed_epochs,
            global_start_time=global_start_time,
            # GPU Optimization
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # CRITICAL: Multi-label flags
            multi_label=True,
            multi_label_threshold=multi_label_threshold,
            # SOTA: Asymmetric Loss parameters
            use_asymmetric_loss=use_asymmetric_loss,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
            # CRITICAL: Pass training_approach for correct chart labeling
            training_approach=config.get('training_approach', 'multi-label'),
            # Distribution-aware training (AdaptiveASL + WeightedRandomSampler)
            distribution_aware=config.get('distribution_aware', False),
            # Reinforced learning parameters
            reinforced_learning=config.get('reinforced_learning', False),
            reinforced_f1_threshold=config.get('rl_f1_threshold', 0.7),
            rl_f1_threshold=config.get('rl_f1_threshold', 0.7),
            reinforced_epochs=config.get('reinforced_epochs'),
            # Resume at RL Phase 2
            skip_to_rl=config.get('skip_to_rl', False),
            rl_state_path=config.get('rl_state_path'),
        )

        # ========== STEP 9: Evaluate ==========
        self.logger.info(f"  ✅ Training complete. Best model: {best_model_path}")
        print(f"  ✅ Training complete!")

        # Get predictions on validation set
        # CRITICAL: Pass multi_label=True to use sigmoid instead of softmax
        val_probs = model.predict(val_dataloader, model.model, proba=True, multi_label=True)

        # Calculate multi-label metrics
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            hamming_loss, jaccard_score
        )

        # ===== Fixed Threshold Evaluation (baseline) =====
        val_predictions_fixed = (val_probs >= multi_label_threshold).astype(int)
        f1_macro_fixed = f1_score(y_val, val_predictions_fixed, average='macro', zero_division=0)

        # ===== Per-Class Threshold Optimization (SOTA) =====
        optimized_thresholds = None
        if optimize_thresholds:
            from .bert_base import optimize_thresholds_per_class, apply_optimized_thresholds

            self.logger.info("   Optimizing per-class thresholds...")
            optimized_thresholds, threshold_scores = optimize_thresholds_per_class(
                val_probs, y_val, metric='f1', search_range=(0.1, 0.9), num_points=50
            )

            # Apply optimized thresholds
            val_predictions = apply_optimized_thresholds(val_probs, optimized_thresholds)

            # Log improvement
            f1_macro_optimized = f1_score(y_val, val_predictions, average='macro', zero_division=0)
            improvement = (f1_macro_optimized - f1_macro_fixed) / max(f1_macro_fixed, 0.001) * 100

            self.logger.info(f"    Fixed threshold ({multi_label_threshold}): F1(macro)={f1_macro_fixed:.4f}")
            self.logger.info(f"    Optimized thresholds: F1(macro)={f1_macro_optimized:.4f} ({improvement:+.1f}%)")

            # Log per-class thresholds for first few classes
            for i, (label, thresh) in enumerate(zip(all_labels[:5], optimized_thresholds[:5])):
                self.logger.info(f"      {label}: threshold={thresh:.3f}, F1={threshold_scores[i]:.4f}")
            if len(all_labels) > 5:
                self.logger.info(f"      ... and {len(all_labels) - 5} more")

            print(f"   Threshold optimization: F1(macro) {f1_macro_fixed:.4f} → {f1_macro_optimized:.4f} ({improvement:+.1f}%)")

            # Store optimized thresholds on model for inference
            model.optimized_thresholds = optimized_thresholds
        else:
            # Use fixed threshold
            val_predictions = val_predictions_fixed

        # ===== Calculate Final Metrics =====
        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(y_val, val_predictions)

        # Hamming loss (fraction of wrong labels)
        h_loss = hamming_loss(y_val, val_predictions)

        # Sample-averaged metrics
        f1_samples = f1_score(y_val, val_predictions, average='samples', zero_division=0)
        precision_samples = precision_score(y_val, val_predictions, average='samples', zero_division=0)
        recall_samples = recall_score(y_val, val_predictions, average='samples', zero_division=0)

        # Micro/Macro averaged metrics
        f1_micro = f1_score(y_val, val_predictions, average='micro', zero_division=0)
        f1_macro = f1_score(y_val, val_predictions, average='macro', zero_division=0)

        # Jaccard (IoU)
        jaccard = jaccard_score(y_val, val_predictions, average='samples', zero_division=0)

        self.logger.info("  Multi-label evaluation metrics (final):")
        self.logger.info(f"    Subset Accuracy: {subset_accuracy:.4f}")
        self.logger.info(f"    Hamming Loss: {h_loss:.4f}")
        self.logger.info(f"    F1 (samples): {f1_samples:.4f}")
        self.logger.info(f"    F1 (micro): {f1_micro:.4f}")
        self.logger.info(f"    F1 (macro): {f1_macro:.4f}")
        self.logger.info(f"    Jaccard: {jaccard:.4f}")

        print(f"   Metrics: F1(macro)={f1_macro:.4f}, Subset Acc={subset_accuracy:.4f}, Hamming={h_loss:.4f}")

        # ========== STEP 9b: Calculate Combined Metric ==========
        # Combined metric for multi-label: 0.6 × macro + 0.4 × micro
        # This formula ensures:
        #   - macro (60%): Model doesn't ignore rare classes
        #   - micro (40%): Model maintains good overall performance
        combined_metric = 0.6 * f1_macro + 0.4 * f1_micro
        language_balance_penalty = 0.0

        self.logger.info(f"   Combined metric (base): {combined_metric:.4f} (0.6×{f1_macro:.4f} + 0.4×{f1_micro:.4f})")

        # Apply language balance penalty if multilingual
        if track_languages and detected_languages and len(detected_languages) > 1 and val_language_info:
            # Calculate per-language F1 scores
            lang_f1_values = []
            val_language_info_upper = [str(l).upper() for l in val_language_info]

            for lang in detected_languages:
                # Get indices for this language
                lang_mask = [l == lang for l in val_language_info_upper]
                if any(lang_mask):
                    lang_indices = [i for i, m in enumerate(lang_mask) if m]
                    if len(lang_indices) > 0:
                        lang_preds = val_predictions[lang_indices]
                        lang_true = y_val[lang_indices]
                        lang_f1 = f1_score(lang_true, lang_preds, average='macro', zero_division=0)
                        lang_f1_values.append(lang_f1)

            if len(lang_f1_values) > 1:
                mean_f1 = np.mean(lang_f1_values)
                std_f1 = np.std(lang_f1_values)
                if mean_f1 > 0:
                    cv = std_f1 / mean_f1
                    language_balance_penalty = min(cv * 0.2, 0.2)
                    combined_metric = combined_metric * (1 - language_balance_penalty)
                    self.logger.info(f"   Language balance: CV={cv:.3f}, penalty={language_balance_penalty:.1%}")
                    self.logger.info(f"   Combined metric (final): {combined_metric:.4f}")

        # ========== STEP 10: Return Results ==========
        result = {
            'model_name': model_name,
            'training_type': 'true_multi_label',
            'num_labels': num_labels,
            'labels': all_labels,
            'label_mapping': label_to_idx,
            # Multi-label specific metrics
            'subset_accuracy': subset_accuracy,
            'hamming_loss': h_loss,
            'f1_samples': f1_samples,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_macro_fixed': f1_macro_fixed,  # Baseline with fixed threshold
            'precision_samples': precision_samples,
            'recall_samples': recall_samples,
            'jaccard': jaccard,
            # Standard metrics (use macro F1 as primary)
            'accuracy': subset_accuracy,
            'f1': f1_macro,
            'precision': precision_samples,
            'recall': recall_samples,
            # Model info
            'model_path': best_model_path,
            'best_metric': best_metric,
            'best_scores': best_scores,
            'multi_label_threshold': multi_label_threshold,
            # SOTA: ASL and threshold optimization info
            'use_asymmetric_loss': use_asymmetric_loss,
            'asl_gamma_neg': asl_gamma_neg if use_asymmetric_loss else None,
            'asl_gamma_pos': asl_gamma_pos if use_asymmetric_loss else None,
            'asl_clip': asl_clip if use_asymmetric_loss else None,
            'optimized_thresholds': optimized_thresholds.tolist() if optimized_thresholds is not None else None,
            'threshold_optimization_enabled': optimize_thresholds,
            # Training info
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            # Intelligent selection metrics
            'combined_metric': combined_metric,
            'language_balance_penalty': language_balance_penalty,
            # Global progress tracking (CRITICAL for multi-model training)
            'global_completed_epochs': model.global_completed_epochs if hasattr(model, 'global_completed_epochs') else None,
        }

        self.logger.info("=" * 80)
        self.logger.info("️ TRUE MULTI-LABEL TRAINING COMPLETE")
        self.logger.info("=" * 80)

        return result

    def _train_multi_label(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multi-label training with multiple files (ONE-VS-ALL approach)

        This is called when training_files contains multiple files for different labels.
        Creates SEPARATE binary classifiers for each label (one-vs-all).

        NOTE: This is NOT true multi-label classification. For true multi-label
        (single model with BCEWithLogitsLoss), use _train_true_multi_label instead.
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

        # ===== Intelligent Ensemble Ranking =====
        # Rank individual models by combined_metric (quality-based, no speed criteria)
        ranked_results = []
        for label_key, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Calculate combined_metric for each binary classifier
                f1_macro = result.get('best_f1_macro', result.get('f1_macro', 0))
                f1_1 = result.get('f1_class_1', result.get('f1_1', 0))  # Minority class F1

                # Binary model: 70% F1_1 + 30% F1_macro (if F1_1 available)
                if f1_1 and f1_1 > 0:
                    combined = 0.7 * f1_1 + 0.3 * f1_macro
                else:
                    combined = f1_macro  # Fall back to F1_macro if F1_1 not available

                # Apply language penalty if available
                lang_metrics = result.get('language_metrics', {})
                if lang_metrics and len(lang_metrics) > 1:
                    f1_values = [m.get('f1_1', m.get('f1_macro', 0)) for m in lang_metrics.values()
                                 if isinstance(m, dict) and m.get('support_1', m.get('support', 0)) > 0]
                    if len(f1_values) > 1:
                        mean_f1 = np.mean(f1_values)
                        if mean_f1 > 0:
                            cv = np.std(f1_values) / mean_f1
                            combined *= (1 - min(cv * 0.2, 0.2))

                ranked_results.append({
                    'label': label_key,
                    'combined_metric': combined,
                    'f1_macro': f1_macro,
                    'f1_1': f1_1 if f1_1 else None,
                    'accuracy': result.get('accuracy', 0),
                    'model_path': result.get('model_path', '')
                })

        # Sort by combined_metric (quality-based, NO speed criteria)
        ranked_results.sort(key=lambda x: x['combined_metric'], reverse=True)
        overall_metrics['ranked_models'] = ranked_results

        # Ensemble combined_metric: average of all models
        if ranked_results:
            overall_metrics['ensemble_combined_metric'] = float(np.mean([r['combined_metric'] for r in ranked_results]))
            overall_metrics['min_combined_metric'] = float(min(r['combined_metric'] for r in ranked_results))
            # Penalize if any model is weak (min < 0.5)
            if overall_metrics['min_combined_metric'] < 0.5:
                overall_metrics['ensemble_combined_metric'] *= 0.9  # 10% penalty
                self.logger.info(f"  [!] Weak model detected (min={overall_metrics['min_combined_metric']:.4f}), 10% penalty applied")

            self.logger.info(f"   Ensemble ranking: {len(ranked_results)} models")
            self.logger.info(f"     Ensemble combined_metric: {overall_metrics['ensemble_combined_metric']:.4f}")
            self.logger.info(f"     Best individual: {ranked_results[0]['label']} ({ranked_results[0]['combined_metric']:.4f})")
            self.logger.info(f"     Worst individual: {ranked_results[-1]['label']} ({ranked_results[-1]['combined_metric']:.4f})")

        # Get the final global_completed_epochs from the last successful result
        final_global_completed_epochs = None
        for r in results.values():
            if isinstance(r, dict) and 'global_completed_epochs' in r and r['global_completed_epochs'] is not None:
                final_global_completed_epochs = r['global_completed_epochs']

        return {
            'best_model': 'multi_label_ensemble',
            'best_f1_macro': overall_metrics.get('avg_f1_macro', 0),
            'accuracy': overall_metrics.get('avg_accuracy', 0),
            'model_path': config.get('output_dir', 'models'),
            'training_time': overall_metrics['total_training_time'],
            'metrics': overall_metrics,
            'individual_results': results,
            # Intelligent selection metrics (quality-based, no speed criteria)
            'combined_metric': overall_metrics.get('ensemble_combined_metric', overall_metrics.get('avg_f1_macro', 0)),
            'ensemble_combined_metric': overall_metrics.get('ensemble_combined_metric'),
            'min_combined_metric': overall_metrics.get('min_combined_metric'),
            'ranked_models': overall_metrics.get('ranked_models', []),
            # Global progress tracking (CRITICAL for multi-model training)
            'global_completed_epochs': final_global_completed_epochs,
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

"""
PROJECT:
-------
AugmentedSocialScientistFork

TITLE:
------
bert_base.py

MAIN OBJECTIVE:
---------------
This script implements the base BERT model class with comprehensive training,
evaluation, and prediction capabilities including reinforced learning, per-language
metrics tracking, and automatic device detection (CUDA/MPS/CPU).

Dependencies:
-------------
- torch (PyTorch for deep learning)
- transformers (HuggingFace transformers library)
- numpy (numerical operations)
- scikit-learn (metrics calculation)
- tqdm (progress bars)
- colorama & tabulate (enhanced console output)

MAIN FEATURES:
--------------
1) Automatic device detection (CUDA GPU, Apple Silicon MPS, or CPU)
2) Text encoding and tokenization with attention masks
3) Comprehensive training with per-epoch checkpointing and metrics logging
4) Reinforced learning for imbalanced datasets (automatic trigger when F1 < 0.60)
5) Per-language performance tracking for multilingual datasets
6) CSV and JSON logging of all training metrics
7) Model saving/loading with HuggingFace compatibility
8) Batch prediction with softmax probabilities

Author:
-------
Antoine Lemor
"""

import datetime
import random
import time
import os
import shutil
import csv
import json
import warnings
from typing import List, Tuple, Any, Optional, Dict
from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax
from torch.types import Device
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from colorama import init, Fore, Back, Style
from tabulate import tabulate
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.console import Group, Console
from rich.text import Text
from rich import box

# Create a shared console for Rich operations
console = Console()

# Initialize colorama for cross-platform colored output
init(autoreset=True)
try:                             # transformers >= 5
    from torch.optim import AdamW
except ImportError:              # transformers <= 4
    from transformers.optimization import AdamW
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME
)
import transformers
# Suppress transformers warnings globally
transformers.logging.set_verbosity_error()

from llm_tool.trainers.bert_abc import BertABC
from llm_tool.utils.logging_utils import get_logger


class TrainingDisplay:
    """Rich Live Display for training progress with all metrics."""

    def __init__(self, model_name: str, label_key: str = None, label_value: str = None,
                 language: str = None, n_epochs: int = 10, is_reinforced: bool = False):
        self.model_name = model_name
        self.label_key = label_key
        self.label_value = label_value
        self.language = language
        self.n_epochs = n_epochs
        self.is_reinforced = is_reinforced

        # Class names for display
        if label_value:
            # Truncate if too long
            value_short = label_value[:15] if len(label_value) > 15 else label_value
            self.class_0_name = f"NOT {value_short}"
            self.class_1_name = f"IS {value_short}"
        else:
            self.class_0_name = "Class 0"
            self.class_1_name = "Class 1"

        # Metrics storage
        self.current_epoch = 0
        self.current_phase = "Initializing"
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.train_progress = 0
        self.val_progress = 0
        self.train_total = 0
        self.val_total = 0

        # Performance metrics
        self.accuracy = 0.0
        self.precision = [0.0, 0.0]
        self.recall = [0.0, 0.0]
        self.f1_scores = [0.0, 0.0]
        self.f1_macro = 0.0
        self.support = [0, 0]

        # Per-language metrics
        self.language_metrics = {}

        # Best model tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.improvement = 0.0
        self.combined_metric = 0.0  # Weighted score for model selection
        self.reinforced_threshold = 0.0  # Threshold to trigger reinforced learning
        self.reinforced_triggered = False  # Whether reinforced learning was triggered
        self.language_variance = 0.0  # Coefficient of variation for F1 class 1 across languages

        # Timing
        self.train_time = 0.0
        self.val_time = 0.0
        self.epoch_time = 0.0
        self.total_time = 0.0
        self.start_time = time.time()

    def create_header(self) -> Table:
        """Create header with model and label info."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column(style="bold white")

        if self.is_reinforced:
            table.add_row("🔄 REINFORCED TRAINING", "")
        else:
            table.add_row("🏋️ MODEL TRAINING", "")

        table.add_row("Model:", self.model_name)

        if self.label_key and self.label_value:
            table.add_row("Label Key:", self.label_key)
            table.add_row("Label Value:", self.label_value)

        if self.language:
            lang_display = self.language if self.language != "MULTI" else f"MULTI (multilingual)"
            table.add_row("Language:", lang_display)

        return table

    def create_progress_section(self) -> Table:
        """Create progress bars for training and validation."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", width=12)
        table.add_column()

        # Epoch progress
        epoch_pct = (self.current_epoch / self.n_epochs) * 100 if self.n_epochs > 0 else 0
        epoch_bar = self._create_bar(epoch_pct, width=30)
        table.add_row("📊 Epoch:", f"{self.current_epoch}/{self.n_epochs} {epoch_bar}")

        # Phase
        table.add_row("🔄 Phase:", self.current_phase)

        # Training progress
        if self.train_total > 0:
            train_pct = (self.train_progress / self.train_total) * 100
            train_bar = self._create_bar(train_pct, width=30)
            table.add_row("📚 Training:", f"{self.train_progress}/{self.train_total} {train_bar}")

        # Validation progress
        if self.val_total > 0:
            val_pct = (self.val_progress / self.val_total) * 100
            val_bar = self._create_bar(val_pct, width=30)
            table.add_row("🔍 Validation:", f"{self.val_progress}/{self.val_total} {val_bar}")

        return table

    def create_metrics_table(self) -> Table:
        """Create table with all performance metrics."""
        table = Table(title="📈 Performance Metrics", box=box.ROUNDED, title_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column(self.class_0_name, justify="center", style="yellow")
        table.add_column(self.class_1_name, justify="center", style="green")
        table.add_column("Overall", justify="center", style="bold white")

        # Losses
        table.add_row("Train Loss", "", "", f"{self.train_loss:.4f}")
        table.add_row("Val Loss", "", "", f"{self.val_loss:.4f}")
        table.add_row("", "", "", "")  # Separator

        # Metrics
        table.add_row("Accuracy", "", "", f"{self.accuracy:.3f}")
        table.add_row("Precision", f"{self.precision[0]:.3f}", f"{self.precision[1]:.3f}", "")
        table.add_row("Recall", f"{self.recall[0]:.3f}", f"{self.recall[1]:.3f}", "")
        table.add_row("F1-Score", f"{self.f1_scores[0]:.3f}", f"{self.f1_scores[1]:.3f}", f"{self.f1_macro:.3f}")
        table.add_row("Support", str(int(self.support[0])), str(int(self.support[1])), str(int(sum(self.support))))

        return table

    def create_language_table(self) -> Table:
        """Create table with per-language metrics."""
        if not self.language_metrics:
            return None

        table = Table(title="🌍 Per-Language Performance", box=box.ROUNDED, title_style="bold magenta")
        table.add_column("Language", style="cyan")
        table.add_column("Support 0", justify="center")
        table.add_column("Support 1", justify="center")
        table.add_column("Accuracy", justify="center")
        table.add_column("F1 Class 0", justify="center")
        table.add_column("F1 Class 1", justify="center")
        table.add_column("Macro F1", justify="center", style="bold")

        for lang, metrics in sorted(self.language_metrics.items()):
            table.add_row(
                lang,
                str(metrics.get('support_0', 0)),
                str(metrics.get('support_1', 0)),
                f"{metrics.get('accuracy', 0):.3f}",
                f"{metrics.get('f1_0', 0):.3f}",
                f"{metrics.get('f1_1', 0):.3f}",
                f"{metrics.get('macro_f1', 0):.3f}"
            )

        # Add average row (only count languages with actual support)
        if len(self.language_metrics) > 1:
            # Only average languages that have at least some samples in class 1
            # (to avoid skewing average with languages that have no positive examples)
            valid_metrics = [m for m in self.language_metrics.values()
                           if m.get('support_0', 0) + m.get('support_1', 0) > 0]

            if valid_metrics:
                avg_acc = sum(m.get('accuracy', 0) for m in valid_metrics) / len(valid_metrics)
                avg_f1 = sum(m.get('macro_f1', 0) for m in valid_metrics) / len(valid_metrics)
                table.add_row("-" * 8, "-" * 9, "-" * 9, "-" * 8, "-" * 10, "-" * 10, "-" * 8)
                table.add_row("AVERAGE", "", "", f"{avg_acc:.3f}", "", "", f"{avg_f1:.3f}", style="bold")

        return table

    def create_summary_section(self) -> Table:
        """Create summary with timing and best model info."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", width=22, no_wrap=True)  # Wide enough for emoji labels
        table.add_column(style="white")

        # Timing
        table.add_row("⏱️ Train Time:", self._format_time(self.train_time))
        table.add_row("⏱️ Val Time:", self._format_time(self.val_time))
        table.add_row("⏱️ Epoch Time:", self._format_time(self.epoch_time))
        table.add_row("⏱️ Total Time:", self._format_time(self.total_time))

        # Best model
        if self.best_f1 > 0:
            table.add_row("", "")
            table.add_row("🏆 Best F1:", f"{self.best_f1:.4f} (Epoch {self.best_epoch})")
            if self.improvement != 0:
                sign = "+" if self.improvement > 0 else ""
                table.add_row("📈 Improvement:", f"{sign}{self.improvement:.4f}")

            # Show combined metric (weighted score) if available
            if self.combined_metric > 0:
                table.add_row("⚖️ Combined Score:", f"{self.combined_metric:.4f}")

            # Show reinforced learning info if relevant
            if self.reinforced_threshold > 0:
                table.add_row("🎯 RL Threshold:", f"{self.reinforced_threshold:.4f}")
                if self.reinforced_triggered:
                    table.add_row("🔥 RL Triggered:", "Yes", style="bold yellow")

            # Show language variance if available (for multilingual models)
            if self.language_variance > 0:
                # Color code based on variance level
                variance_style = "green" if self.language_variance < 0.3 else ("yellow" if self.language_variance < 0.7 else "red")
                table.add_row("📊 Lang Variance:", f"{self.language_variance:.3f}", style=variance_style)

        return table

    def create_panel(self) -> Panel:
        """Create the complete panel with all sections."""
        sections = [self.create_header()]
        sections.append(Text())  # Spacer
        sections.append(self.create_progress_section())
        sections.append(Text())  # Spacer
        sections.append(self.create_metrics_table())

        # Add language table if available
        lang_table = self.create_language_table()
        if lang_table:
            sections.append(Text())  # Spacer
            sections.append(lang_table)

        sections.append(Text())  # Spacer
        sections.append(self.create_summary_section())

        group = Group(*sections)

        # Different colors for normal vs reinforced training
        if self.is_reinforced:
            title = "🔥 REINFORCED LEARNING"
            border_color = "bold yellow"  # Yellow/orange for reinforced
        else:
            title = "🏋️ MODEL TRAINING"
            border_color = "bold blue"  # Blue for normal

        return Panel(group, title=title, border_style=border_color, box=box.HEAVY)

    def _create_bar(self, percentage: float, width: int = 30) -> str:
        """Create a text-based progress bar."""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:.1f}%"

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class BertBase(BertABC):
    def __init__(
            self,
            model_name: str = 'bert-base-cased',
            tokenizer: Any = BertTokenizer,
            model_sequence_classifier: Any = BertForSequenceClassification,
            device: Device | None = None,
    ):
        """
        Parameters
        ----------
        model_name: str, default='bert-base-cased'
            A model name from huggingface models: https://huggingface.co/models

        tokenizer: huggingface tokenizer, default=BertTokenizer.from_pretrained('bert-base-cased')
            Tokenizer to use

        model_sequence_classifier: huggingface sequence classifier, default=BertForSequenceClassification
            A huggingface sequence classifier that implements a from_pretrained() function

        device: torch.Device, default=None
            Device to use. If None, automatically set if GPU is available. CPU otherwise.
        """
        self.model_name = model_name
        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.model_sequence_classifier = model_sequence_classifier
        self.dict_labels = None
        self.language_metrics_history: List[Dict[str, Any]] = []
        self.last_training_summary: Optional[Dict[str, Any]] = None
        self.last_saved_model_path: Optional[str] = None
        self.logger = get_logger(f"AugmentedSocialScientistFork.{self.__class__.__name__}")

        # Set or detect device
        self.device = device
        if self.device is None:
            # If CUDA is available, use it
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info('Detected %d GPU(s). Using GPU %s (%s).',
                                 torch.cuda.device_count(),
                                 torch.cuda.current_device(),
                                 torch.cuda.get_device_name(torch.cuda.current_device()))
            # If MPS is available on Apple Silicon, use it
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                # Suppress MPS detection log to keep console clean
                # self.logger.info('Detected Apple Silicon MPS backend. Using the MPS device.')
            # Otherwise, use CPU
            else:
                self.device = torch.device("cpu")
                # Suppress CPU fallback log to keep console clean
                # self.logger.info('Falling back to CPU execution.')

    # ------------------------------------------------------------------
    # Internal helpers shared with enhanced variants
    # ------------------------------------------------------------------
    def _prepare_inputs(
            self,
            sequences: List[str],
            add_special_tokens: bool = True,
            progress_bar: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenise and build attention masks for a list of sequences."""
        if not sequences:
            return np.zeros((0, 0), dtype="long"), np.zeros((0, 0), dtype="long")

        iterator = tqdm(sequences, desc="Tokenizing") if progress_bar else sequences
        input_ids: List[List[int]] = []
        for sent in iterator:
            encoded_sent = self.tokenizer.encode(sent, add_special_tokens=add_special_tokens)
            input_ids.append(encoded_sent)

        max_len = min(max(len(sen) for sen in input_ids), 512)
        pad = np.full((len(input_ids), max_len), 0, dtype="long")
        for idx, seq in enumerate(input_ids):
            trunc = seq[:max_len]
            pad[idx, :len(trunc)] = trunc

        attention_masks: List[List[int]] = []
        mask_iter = tqdm(pad, desc="Creating attention masks") if progress_bar else pad
        for seq in mask_iter:
            attention_masks.append([int(token_id > 0) for token_id in seq])

        return pad, np.asarray(attention_masks, dtype="long")

    def _build_dataset(
            self,
            inputs: np.ndarray,
            masks: np.ndarray,
            labels: Optional[List[str | int]] = None
    ) -> Tuple[TensorDataset, Optional[Dict[str, int]]]:
        """Create a TensorDataset from arrays and optional labels."""
        inputs_tensors = torch.tensor(inputs, dtype=torch.long)
        masks_tensors = torch.tensor(masks, dtype=torch.long)

        if labels is None:
            dataset = TensorDataset(inputs_tensors, masks_tensors)
            return dataset, None

        label_names = np.unique(labels)
        self.dict_labels = dict(zip(label_names, range(len(label_names))))
        labels_tensors = torch.tensor([self.dict_labels[x] for x in labels], dtype=torch.long)
        dataset = TensorDataset(inputs_tensors, masks_tensors, labels_tensors)
        return dataset, self.dict_labels

    def encode(
            self,
            sequences: List[str],
            labels: List[str | int] | None = None,
            batch_size: int = 32,
            progress_bar: bool = True,
            add_special_tokens: bool = True
    ) -> DataLoader:
        """
        Preprocess training, test or prediction data by:
          (1) Tokenizing the sequences and mapping tokens to their IDs.
          (2) Truncating or padding to a max length of 512 tokens, and creating corresponding attention masks.
          (3) Returning a pytorch DataLoader containing token ids, labels (if any) and attention masks.

        Parameters
        ----------
        sequences: 1D array-like
            List of input texts.

        labels: 1D array-like or None, default=None
            List of labels. None for unlabeled prediction data.

        batch_size: int, default=32
            Batch size for the PyTorch DataLoader.

        progress_bar: bool, default=True
            If True, print a progress bar for tokenization and mask creation.

        add_special_tokens: bool, default=True
            If True, add '[CLS]' and '[SEP]' tokens.

        Returns
        -------
        dataloader: torch.utils.data.DataLoader
            A PyTorch DataLoader with input_ids, attention_masks, and labels (if provided).
        """
        inputs, masks = self._prepare_inputs(
            sequences,
            add_special_tokens=add_special_tokens,
            progress_bar=progress_bar,
        )

        dataset, label_mapping = self._build_dataset(inputs, masks, labels)

        # Suppress label mapping log to keep console clean
        # if label_mapping is not None and progress_bar:
        #     self.logger.info("Label ids mapping: %s", label_mapping)

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def calculate_reinforced_trigger_score(
            self,
            f1_class_0: float,
            f1_class_1: float,
            support_class_0: int,
            support_class_1: int,
            language_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[float, bool, str]:
        """
        Calculate an intelligent score to determine if reinforced learning should be triggered.

        This method considers:
        1. F1 score of class 1 (minority class)
        2. Class imbalance ratio (to overweight minority class performance)
        3. Per-language F1 scores (if available)

        Parameters
        ----------
        f1_class_0 : float
            F1 score for class 0
        f1_class_1 : float
            F1 score for class 1 (typically minority class)
        support_class_0 : int
            Number of samples in class 0
        support_class_1 : int
            Number of samples in class 1
        language_metrics : dict, optional
            Dictionary with language-specific metrics

        Returns
        -------
        trigger_score : float
            Computed score (0.0 to 1.0) - lower means worse performance
        should_trigger : bool
            True if reinforced learning should be triggered
        reason : str
            Human-readable explanation of the decision
        """
        # Calculate class imbalance ratio
        total_samples = support_class_0 + support_class_1
        if total_samples == 0:
            return 0.0, True, "No samples available"

        class_1_ratio = support_class_1 / total_samples
        class_0_ratio = support_class_0 / total_samples

        # Calculate imbalance weight (higher when class is more imbalanced)
        # When class 1 is 50%, imbalance_weight = 1.0
        # When class 1 is 10%, imbalance_weight = 2.0
        # When class 1 is 5%, imbalance_weight = 2.5
        if class_1_ratio > 0:
            imbalance_weight = min(0.5 / class_1_ratio, 3.0)  # Cap at 3.0x weight
        else:
            imbalance_weight = 3.0

        # Base score: weighted F1 of class 1
        # When class is very imbalanced, F1 of class 1 becomes more important
        base_weight = 0.4 + (0.3 * min(imbalance_weight / 3.0, 1.0))  # 0.4 to 0.7
        class_1_weighted_f1 = f1_class_1 * base_weight
        class_0_weighted_f1 = f1_class_0 * (1.0 - base_weight)

        overall_f1_score = class_1_weighted_f1 + class_0_weighted_f1

        # Factor in language-specific performance if available
        language_penalty = 0.0
        language_info = ""
        language_variance_penalty = 0.0

        if language_metrics and len(language_metrics) > 0:
            # Check if any language has poor F1 for class 1
            poor_languages = []
            f1_class1_by_lang = []

            for lang, metrics in language_metrics.items():
                lang_f1_1 = metrics.get('f1_1', 0.0)
                lang_support_1 = metrics.get('support_1', 0)

                # Track F1 class 1 for variance calculation
                if lang_support_1 >= 3:  # Only include languages with meaningful support
                    f1_class1_by_lang.append(lang_f1_1)

                # Consider a language "poor" if F1_1 < 0.5 and has reasonable support
                if lang_f1_1 < 0.5 and lang_support_1 >= 3:
                    poor_languages.append(f"{lang}(F1={lang_f1_1:.2f})")
                    # Apply penalty proportional to how bad the language is
                    language_penalty += (0.5 - lang_f1_1) * 0.15  # Max 0.15 penalty per language

            # Calculate variance penalty if we have multiple languages
            if len(f1_class1_by_lang) > 1:
                mean_f1 = np.mean(f1_class1_by_lang)
                std_f1 = np.std(f1_class1_by_lang)

                # High variance = unbalanced performance across languages
                # Apply penalty if coefficient of variation > 0.5
                if mean_f1 > 0:
                    cv = std_f1 / mean_f1
                    if cv > 0.5:  # Significant variance
                        language_variance_penalty = (cv - 0.5) * 0.1  # Up to 0.1 penalty
                        language_info += f" High variance across languages (CV={cv:.2f})."

            if poor_languages:
                language_info = f" Poor language performance: {', '.join(poor_languages)}." + language_info

        # Final trigger score (penalized by poor language performance and variance)
        trigger_score = max(0.0, overall_f1_score - language_penalty - language_variance_penalty)

        # Decision logic
        # Trigger if:
        # 1. Score is below 0.70 (standard threshold)
        # 2. OR class 1 F1 is below 0.40 (very poor minority class performance)
        # 3. OR any language has F1_1 < 0.30
        should_trigger = False
        reason = ""

        if trigger_score < 0.70:
            should_trigger = True
            reason = (f"Trigger score {trigger_score:.3f} < 0.70 "
                     f"(Class 1 F1={f1_class_1:.3f}, imbalance={class_1_ratio:.1%}, "
                     f"weight={imbalance_weight:.2f}x).{language_info}")
        elif f1_class_1 < 0.40:
            should_trigger = True
            reason = f"Class 1 F1 ({f1_class_1:.3f}) critically low < 0.40.{language_info}"
        elif language_metrics:
            for lang, metrics in language_metrics.items():
                if metrics.get('f1_1', 0.0) < 0.30 and metrics.get('support_1', 0) >= 3:
                    should_trigger = True
                    reason = f"Language {lang} has critical F1_1={metrics['f1_1']:.3f} < 0.30.{language_info}"
                    break

        if not should_trigger:
            reason = (f"No trigger: score={trigger_score:.3f}, F1_1={f1_class_1:.3f}, "
                     f"class 1 ratio={class_1_ratio:.1%}.{language_info}")

        return trigger_score, should_trigger, reason

    def run_training(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            n_epochs: int = 3,
            lr: float = 5e-5,
            random_state: int = 42,
            save_model_as: str | None = None,
            pos_weight: torch.Tensor | None = None,
            metrics_output_dir: str = "./training_logs",
            best_model_criteria: str = "combined",
            f1_class_1_weight: float = 0.7,
            reinforced_learning: bool = False,
            n_epochs_reinforced: int = 2,
            reinforced_epochs: int | None = None,   # ← nouveau
            rescue_low_class1_f1: bool = False,
            track_languages: bool = False,
            language_info: Optional[List[str]] = None,
            f1_1_rescue_threshold: float = 0.0,
            model_identifier: Optional[str] = None,
            reinforced_f1_threshold: float = 0.7,  # Nouveau paramètre pour le seuil de déclenchement
            label_key: Optional[str] = None,  # Multi-label: key being trained (e.g., 'themes', 'sentiment')
            label_value: Optional[str] = None,  # Multi-label: specific value (e.g., 'transportation', 'positive')
            language: Optional[str] = None  # Language of the data being trained (e.g., 'EN', 'FR')
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Train, evaluate, and (optionally) save a BERT model. This method also logs training and validation
        metrics per epoch, handles best-model selection, and can optionally trigger reinforced learning if
        the best F1 on class 1 is below the reinforced_f1_threshold at the end of normal training.

        This method can also (optionally) apply a "rescue" logic for class 1 F1 scores that remain at 0
        after normal training: if ``rescue_low_class1_f1=True`` and the best model's F1 for class 1 is 0,
        the reinforced learning step will consider any small improvement of class 1's F1 (greater than
        ``f1_1_rescue_threshold``) as sufficient to select a reinforced epoch's model.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            Training dataloader, typically from self.encode().

        test_dataloader: torch.utils.data.DataLoader
            Test/validation dataloader, typically from self.encode().

        n_epochs: int, default=3
            Number of epochs for the normal training phase.

        lr: float, default=5e-5
            Learning rate for normal training.

        random_state: int, default=42
            Random seed for reproducibility.

        save_model_as: str, default=None
            If not None, will save the final best model to ./models/<save_model_as>.

        pos_weight: torch.Tensor, default=None
            If not None, weights the loss to favor certain classes more heavily
            (useful in binary classification).

        metrics_output_dir: str, default="./training_logs"
            Directory for saving CSV logs: training_metrics.csv and best_models.csv.

        best_model_criteria: str, default="combined"
            Criterion for best model. Currently supports:
              - "combined": Weighted combination of F1(class 1) and macro F1.

        f1_class_1_weight: float, default=0.7
            Weight for F1(class 1) in the combined metric. The remaining (1 - weight) goes to macro F1.

        reinforced_learning: bool, default=False
            If True, and if the best model after normal training has F1(class 1) < 0.7,
            a reinforced training phase will be triggered.

        n_epochs_reinforced: int, default=2
            Number of epochs for the reinforced learning phase (if triggered).

        rescue_low_class1_f1: bool, default=False
            If True, then during reinforced learning we check if the best normal-training
            F1 for class 1 is 0. In that case, any RL epoch where class 1's F1 becomes greater
            than ``f1_1_rescue_threshold`` is automatically considered a better model.

        f1_1_rescue_threshold: float, default=0.0
            The threshold above which a class 1 F1 (starting from 0 after normal training)
            is considered a sufficient improvement to pick the reinforced epoch's model.

        Returns
        -------
        scores: tuple (precision, recall, f1-score, support)
            Final best evaluation scores from sklearn.metrics.precision_recall_fscore_support,
            for each label. Shape: (4, n_labels).

        Notes
        -----
        This method generates:
            - "<metrics_output_dir>/training_metrics.csv": logs metrics for each normal-training epoch.
            - "<metrics_output_dir>/best_models.csv": logs any new best model (normal or reinforced).
            - If reinforced training is triggered, it also logs a reinforced_training_metrics.csv.
            - The final best model is ultimately saved to "./models/<save_model_as>" if save_model_as is provided.
              (If reinforced training finds a better model, that replaces the previous best.)
        """
        # Reset reinforced learning flag at the start of each training session
        self._reinforced_already_triggered = False

        # Ensure metric output directory exists
        os.makedirs(metrics_output_dir, exist_ok=True)
        training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics.csv")
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Initialize CSV for normal training metrics
        csv_headers = [
            "model_identifier",
            "model_name",
            "label_key",        # Multi-label: key (e.g., 'themes', 'sentiment')
            "label_value",      # Multi-label: value (e.g., 'transportation', 'positive')
            "language",         # Language of the data (e.g., 'EN', 'FR', 'MULTI')
            "timestamp",
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",         # Overall accuracy
            "precision_0",
            "recall_0",
            "f1_0",
            "support_0",
            "precision_1",
            "recall_1",
            "f1_1",
            "support_1",
            "macro_f1"
        ]

        # Add comprehensive language-specific headers if tracking languages
        if track_languages and language_info is not None:
            unique_langs = sorted(list(set(language_info)))
            for lang in unique_langs:
                csv_headers.extend([
                    f"{lang}_accuracy",
                    f"{lang}_precision_0",
                    f"{lang}_recall_0",
                    f"{lang}_f1_0",
                    f"{lang}_support_0",
                    f"{lang}_precision_1",
                    f"{lang}_recall_1",
                    f"{lang}_f1_1",
                    f"{lang}_support_1",
                    f"{lang}_macro_f1"
                ])

        # Check if file exists to decide whether to write headers
        if os.path.exists(training_metrics_csv):
            # Append to existing file
            mode = 'a'
            write_headers = False
        else:
            # Create new file with headers
            mode = 'w'
            write_headers = True

        with open(training_metrics_csv, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(csv_headers)

        # Initialize CSV for best models (both normal and reinforced)
        # We'll include a "training_phase" column to indicate normal or reinforced.
        # To ensure consistency across different models with different languages,
        # we'll use a standard set of language columns (EN, FR) always present
        best_models_headers = [
            "model_identifier",
            "model_type",
            "label_key",        # Multi-label: key (e.g., 'themes', 'sentiment')
            "label_value",      # Multi-label: value (e.g., 'transportation', 'positive')
            "language",         # Language of the data (e.g., 'EN', 'FR', 'MULTI')
            "timestamp",
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",         # Overall accuracy
            "precision_0",
            "recall_0",
            "f1_0",
            "support_0",
            "precision_1",
            "recall_1",
            "f1_1",
            "support_1",
            "macro_f1"
        ]

        # Always add standard language columns (EN, FR) for consistency
        # This ensures all rows have the same structure
        standard_languages = ['EN', 'FR']
        for lang in standard_languages:
            best_models_headers.extend([
                f"{lang}_accuracy",
                f"{lang}_precision_0",
                f"{lang}_recall_0",
                f"{lang}_f1_0",
                f"{lang}_support_0",
                f"{lang}_precision_1",
                f"{lang}_recall_1",
                f"{lang}_f1_1",
                f"{lang}_support_1",
                f"{lang}_macro_f1"
            ])

        best_models_headers.extend([
            "saved_model_path",
            "training_phase"
        ])

        # Check if file exists to decide whether to write headers
        if os.path.exists(best_models_csv):
            # Append to existing file
            mode = 'a'
            write_headers = False
        else:
            # Create new file with headers
            mode = 'w'
            write_headers = True

        with open(best_models_csv, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(best_models_headers)

        # Collect test labels for classification report
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()
        num_labels = np.unique(test_labels).size

        # Ensure we have at least 2 labels for binary classification
        # This fixes the issue when all samples in test have the same label
        if num_labels < 2:
            num_labels = 2

        # Potentially store label names (if dict_labels is available)
        if self.dict_labels is None:
            label_names = None
        else:
            # Sort by index - handle both simple values and dict values
            try:
                label_names = [str(x[0]) for x in sorted(self.dict_labels.items(), key=lambda x: x[1])]
            except (TypeError, AttributeError):
                # If values are dicts or other non-comparable types, just use the keys
                label_names = [str(k) for k in self.dict_labels.keys()]

        # Set seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Initialize the model (suppress warnings about missing weights)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
            warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
            model = self.model_sequence_classifier.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                output_attentions=False,
                output_hidden_states=False
            )
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * n_epochs
        )

        train_loss_values = []
        val_loss_values = []

        best_metric_val = -1.0
        best_model_path = None
        best_scores = None  # Will store final best (precision, recall, f1, support)
        best_language_metrics = None  # Will store language metrics from best epoch
        language_performance_history = []  # Store language metrics for each epoch
        self.language_metrics_history = []
        if reinforced_epochs is not None:
            n_epochs_reinforced = reinforced_epochs

        # =============== Normal Training Loop ===============
        training_start_time = time.time()  # Initialize the timer

        # Initialize metrics tracking
        training_metrics = []

        # Initialize Rich Live Display
        display = TrainingDisplay(
            model_name=self.model_name if hasattr(self, 'model_name') else save_model_as or "BERT",
            label_key=label_key,
            label_value=label_value,
            language=language,
            n_epochs=n_epochs,
            is_reinforced=False
        )

        # Start Live display - this will remain fixed and update in place
        # Use transient=True to clear the display when context exits (prevents stacking)
        with Live(display.create_panel(), refresh_per_second=4, transient=True) as live:
            for i_epoch in range(n_epochs):
                epoch_start_time = time.time()

                # Update display for new epoch
                display.current_epoch = i_epoch + 1
                display.current_phase = "Training"
                display.train_total = len(train_dataloader)
                display.train_progress = 0
                live.update(display.create_panel())

                t0 = time.time()
                total_train_loss = 0.0
                model.train()

                # Training loop - no tqdm, we update the display directly
                for step, train_batch in enumerate(train_dataloader):
                    b_inputs = train_batch[0].to(self.device)
                    b_masks = train_batch[1].to(self.device)
                    b_labels = train_batch[2].to(self.device)

                    model.zero_grad()

                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                    logits = outputs[0]

                    # Weighted loss if pos_weight is specified
                    if pos_weight is not None:
                        weight_tensor = torch.tensor([1.0, pos_weight.item()], device=self.device)
                        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                    else:
                        criterion = torch.nn.CrossEntropyLoss()

                    loss = criterion(logits, b_labels)
                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    # Update display with current progress
                    display.train_progress = step + 1
                    display.train_loss = loss.item()
                    display.epoch_time = time.time() - epoch_start_time  # Update elapsed time
                    display.total_time = time.time() - training_start_time
                    live.update(display.create_panel())

                # After training loop - calculate average loss
                avg_train_loss = total_train_loss / len(train_dataloader)
                train_loss_values.append(avg_train_loss)

                # Update display with final training metrics
                display.train_loss = avg_train_loss
                display.train_time = time.time() - t0
                display.current_phase = "Validation"
                display.val_total = len(test_dataloader)
                display.val_progress = 0
                live.update(display.create_panel())

                # =============== Validation after this epoch ===============
                t0 = time.time()
                model.eval()

                total_val_loss = 0.0
                logits_complete = []

                # Validation loop - update display directly
                for step, test_batch in enumerate(test_dataloader):
                    b_inputs = test_batch[0].to(self.device)
                    b_masks = test_batch[1].to(self.device)
                    b_labels = test_batch[2].to(self.device)

                    with torch.no_grad():
                        outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                        loss = outputs.loss
                        logits = outputs.logits

                    total_val_loss += loss.item()
                    logits_complete.append(logits.detach().cpu().numpy())

                    # Update display with validation progress
                    display.val_progress = step + 1
                    display.val_loss = loss.item()
                    display.epoch_time = time.time() - epoch_start_time  # Update elapsed time
                    display.total_time = time.time() - training_start_time
                    live.update(display.create_panel())

                # After validation loop
                logits_complete = np.concatenate(logits_complete, axis=0)
                avg_val_loss = total_val_loss / len(test_dataloader)
                val_loss_values.append(avg_val_loss)

                # Update display with final validation metrics
                display.val_loss = avg_val_loss
                display.val_time = time.time() - t0
                live.update(display.create_panel())

                preds = np.argmax(logits_complete, axis=1).flatten()

                # Get actual unique classes present in predictions and labels
                unique_classes = np.unique(np.concatenate([test_labels, preds]))

                # Only use target_names if it matches the number of unique classes
                if label_names is not None and len(label_names) == len(unique_classes):
                    report = classification_report(test_labels, preds, target_names=label_names, output_dict=True)
                else:
                    # Don't use target_names if there's a mismatch
                    report = classification_report(test_labels, preds, output_dict=True)

                # Extract metrics for classes 0 and 1 (assuming binary classification or focusing on first two classes)
                class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
                class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
                macro_avg = report.get("macro avg", {"f1-score": 0})

                precision_0 = class_0_metrics["precision"]
                recall_0 = class_0_metrics["recall"]
                f1_0 = class_0_metrics["f1-score"]
                support_0 = class_0_metrics["support"]

                precision_1 = class_1_metrics["precision"]
                recall_1 = class_1_metrics["recall"]
                f1_1 = class_1_metrics["f1-score"]
                support_1 = class_1_metrics["support"]

                macro_f1 = macro_avg["f1-score"]

                # Calculate accuracy
                accuracy = np.sum(preds == test_labels) / len(test_labels)

                # Update display with performance metrics
                display.accuracy = accuracy
                display.precision = [precision_0, precision_1]
                display.recall = [recall_0, recall_1]
                display.f1_scores = [f1_0, f1_1]
                display.f1_macro = macro_f1
                display.support = [support_0, support_1]
                live.update(display.create_panel())

                # Calculate and display per-language metrics if requested
                if track_languages and language_info is not None:
                    unique_languages = list(set(language_info))
                    language_metrics = {}

                    for lang in sorted(unique_languages):
                        # Get indices for this language
                        lang_indices = [i for i, l in enumerate(language_info) if l == lang]
                        if not lang_indices:
                            continue

                        # Get predictions and labels for this language
                        lang_preds = preds[lang_indices]
                        lang_labels = np.array(test_labels)[lang_indices]

                        # Calculate metrics
                        lang_report = classification_report(lang_labels, lang_preds, output_dict=True, zero_division=0)

                        lang_acc = lang_report.get('accuracy', 0)

                        # Extract detailed metrics for class 0
                        lang_precision_0 = lang_report.get('0', {}).get('precision', 0) if '0' in lang_report else 0
                        lang_recall_0 = lang_report.get('0', {}).get('recall', 0) if '0' in lang_report else 0
                        lang_f1_0 = lang_report.get('0', {}).get('f1-score', 0) if '0' in lang_report else 0
                        lang_support_0 = int(lang_report.get('0', {}).get('support', 0)) if '0' in lang_report else 0

                        # Extract detailed metrics for class 1
                        lang_precision_1 = lang_report.get('1', {}).get('precision', 0) if '1' in lang_report else 0
                        lang_recall_1 = lang_report.get('1', {}).get('recall', 0) if '1' in lang_report else 0
                        lang_f1_1 = lang_report.get('1', {}).get('f1-score', 0) if '1' in lang_report else 0
                        lang_support_1 = int(lang_report.get('1', {}).get('support', 0)) if '1' in lang_report else 0

                        lang_macro_f1 = lang_report.get('macro avg', {}).get('f1-score', 0)
                        lang_support = len(lang_indices)

                        language_metrics[lang] = {
                            'samples': lang_support,
                            'accuracy': lang_acc,
                            'precision_0': lang_precision_0,
                            'recall_0': lang_recall_0,
                            'f1_0': lang_f1_0,
                            'support_0': lang_support_0,
                            'precision_1': lang_precision_1,
                            'recall_1': lang_recall_1,
                            'f1_1': lang_f1_1,
                            'support_1': lang_support_1,
                            'macro_f1': lang_macro_f1
                        }

                    # Update display with language metrics
                    display.language_metrics = language_metrics

                    # Calculate language variance for F1 class 1
                    f1_class1_values = [m['f1_1'] for m in language_metrics.values() if m['support_1'] >= 3]
                    if len(f1_class1_values) > 1:
                        mean_f1 = np.mean(f1_class1_values)
                        std_f1 = np.std(f1_class1_values)
                        display.language_variance = (std_f1 / mean_f1) if mean_f1 > 0 else 0

                    live.update(display.create_panel())

                    # Store language metrics for this epoch
                    averages: Optional[Dict[str, float]] = None
                    if language_metrics:
                        avg_acc = sum(m['accuracy'] for m in language_metrics.values()) / len(language_metrics)
                        avg_f1 = sum(m['macro_f1'] for m in language_metrics.values()) / len(language_metrics)
                        averages = {'accuracy': avg_acc, 'macro_f1': avg_f1}

                        epoch_record = {
                            'epoch': i_epoch + 1,
                            'metrics': language_metrics
                        }
                        if averages is not None:
                            epoch_record['averages'] = averages
                        language_performance_history.append(epoch_record)

                # Append to training_metrics.csv (normal training phase)
                with open(training_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Get timestamp for this entry
                    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    row = [
                        model_identifier if model_identifier else "",
                        self.model_name if hasattr(self, 'model_name') else self.__class__.__name__,
                        label_key if label_key else "",         # Add label_key
                        label_value if label_value else "",     # Add label_value
                        language if language else "",           # Add language
                        current_timestamp,
                        i_epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        accuracy,                               # Add accuracy
                        precision_0,
                        recall_0,
                        f1_0,
                        support_0,
                        precision_1,
                        recall_1,
                        f1_1,
                        support_1,
                        macro_f1
                    ]

                    # Add language metrics if available
                    if track_languages and language_info is not None and 'language_metrics' in locals():
                        unique_languages = list(set(language_info))
                        for lang in sorted(unique_languages):
                            if lang in language_metrics:
                                row.extend([
                                    language_metrics[lang]['accuracy'],
                                    language_metrics[lang]['precision_0'],
                                    language_metrics[lang]['recall_0'],
                                    language_metrics[lang]['f1_0'],
                                    language_metrics[lang]['support_0'],
                                    language_metrics[lang]['precision_1'],
                                    language_metrics[lang]['recall_1'],
                                    language_metrics[lang]['f1_1'],
                                    language_metrics[lang]['support_1'],
                                    language_metrics[lang]['macro_f1']
                                ])
                            else:
                                row.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Default values if language not in this epoch

                    writer.writerow(row)

                # Compute "combined" metric if best_model_criteria is "combined"
                if best_model_criteria == "combined":
                    combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1

                    # Apply language balance penalty if we have per-language metrics
                    if language_metrics:
                        # Calculate variance in F1 class 1 across languages
                        f1_class1_values = [m.get('f1_1', 0) for m in language_metrics.values()
                                           if m.get('support_1', 0) > 0]  # Only count languages with positive examples

                        if len(f1_class1_values) > 1:
                            # Calculate coefficient of variation (std / mean) as a measure of imbalance
                            mean_f1_class1 = np.mean(f1_class1_values)
                            std_f1_class1 = np.std(f1_class1_values)

                            # Avoid division by zero
                            if mean_f1_class1 > 0:
                                cv = std_f1_class1 / mean_f1_class1  # Coefficient of variation
                                # Apply penalty: reduce combined_metric by up to 20% based on variance
                                # CV of 0 = no penalty, CV of 1.0+ = full 20% penalty
                                language_balance_penalty = min(cv * 0.2, 0.2)
                                combined_metric = combined_metric * (1 - language_balance_penalty)
                else:
                    # Fallback or alternative strategy
                    combined_metric = (f1_1 + macro_f1) / 2.0

                # Compute epoch timing
                epoch_time = time.time() - epoch_start_time
                display.epoch_time = epoch_time
                display.total_time = time.time() - training_start_time

                # Check if this is a new best model
                if combined_metric > best_metric_val:
                    # We found a new best model
                    display.improvement = combined_metric - best_metric_val
                    display.best_f1 = macro_f1
                    display.best_epoch = i_epoch + 1
                    display.combined_metric = combined_metric
                    live.update(display.create_panel())
                    # Remove old best model folder if it exists
                    if best_model_path is not None:
                        try:
                            shutil.rmtree(best_model_path)
                        except OSError:
                            pass

                    best_metric_val = combined_metric

                    # Always save best_scores for reinforced learning trigger check
                    best_scores = precision_recall_fscore_support(test_labels, preds)

                    # Save language metrics from best epoch (if available)
                    if track_languages and 'language_metrics' in locals():
                        best_language_metrics = language_metrics.copy()

                    if save_model_as is not None:
                        # Save the new best model in a temporary folder
                        best_model_path = f"./models/{save_model_as}_epoch_{i_epoch+1}"
                        os.makedirs(best_model_path, exist_ok=True)

                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(best_model_path, WEIGHTS_NAME)
                        output_config_file = os.path.join(best_model_path, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        self.tokenizer.save_vocabulary(best_model_path)
                    else:
                        best_model_path = None

                    # Check if this is truly the best model for this model type
                    # Read existing best_models.csv to check if we already have a better model
                    should_update_best = True
                    model_type = self.model_name if hasattr(self, 'model_name') else self.__class__.__name__

                    if os.path.exists(best_models_csv) and os.path.getsize(best_models_csv) > 0:
                        # Read existing best models to check if this model type already has a better score
                        with open(best_models_csv, 'r', encoding='utf-8') as f_read:
                            reader = csv.DictReader(f_read)
                            for existing_row in reader:
                                # Check if same model type and identifier
                                if (existing_row.get('model_identifier') == (model_identifier if model_identifier else "") and
                                    existing_row.get('model_type') == model_type):
                                    # Compare scores
                                    existing_macro_f1 = float(existing_row.get('macro_f1', 0))
                                    if existing_macro_f1 >= macro_f1:
                                        should_update_best = False
                                        break

                    if should_update_best:
                        # First, remove any existing entry for this model type/identifier combination
                        if os.path.exists(best_models_csv) and os.path.getsize(best_models_csv) > 0:
                            # Read all rows
                            rows_to_keep = []
                            with open(best_models_csv, 'r', encoding='utf-8') as f_read:
                                reader = csv.DictReader(f_read)
                                headers_dict = reader.fieldnames
                                for row in reader:
                                    # Keep rows that are NOT the same model type/identifier
                                    if not (row.get('model_identifier') == (model_identifier if model_identifier else "") and
                                           row.get('model_type') == model_type):
                                        rows_to_keep.append(row)

                            # Rewrite the file without the old entry
                            if headers_dict:
                                with open(best_models_csv, 'w', newline='', encoding='utf-8') as f_write:
                                    writer = csv.DictWriter(f_write, fieldnames=headers_dict)
                                    writer.writeheader()
                                    writer.writerows(rows_to_keep)

                        # Now append the new best model
                        with open(best_models_csv, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)

                            # Get timestamp
                            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                            row = [
                                model_identifier if model_identifier else "",
                                model_type,
                                label_key if label_key else "",         # Add label_key
                                label_value if label_value else "",     # Add label_value
                                language if language else "",           # Add language
                                current_timestamp,
                                i_epoch + 1,
                                avg_train_loss,
                                avg_val_loss,
                                accuracy,                               # Add accuracy
                                precision_0,
                                recall_0,
                                f1_0,
                                support_0,
                                precision_1,
                                recall_1,
                                f1_1,
                                support_1,
                                macro_f1
                            ]

                            # Add language metrics for standard languages (EN, FR)
                            # Always add these columns to maintain CSV consistency
                            standard_languages = ['EN', 'FR']
                            for lang in standard_languages:
                                if track_languages and language_info is not None and 'language_metrics' in locals() and lang in language_metrics:
                                    row.extend([
                                        language_metrics[lang]['accuracy'],
                                        language_metrics[lang]['precision_0'],
                                        language_metrics[lang]['recall_0'],
                                        language_metrics[lang]['f1_0'],
                                        language_metrics[lang]['support_0'],
                                        language_metrics[lang]['precision_1'],
                                        language_metrics[lang]['recall_1'],
                                        language_metrics[lang]['f1_1'],
                                        language_metrics[lang]['support_1'],
                                        language_metrics[lang]['macro_f1']
                                    ])
                                else:
                                    # Fill with empty values to maintain CSV structure
                                    row.extend(['', '', '', '', '', '', '', '', '', ''])

                            row.extend([
                                best_model_path if best_model_path else "Not saved to disk",
                                "normal"  # training phase
                            ])

                            writer.writerow(row)

                else:
                    # No new best model this epoch, but still update display to show current epoch timing
                    live.update(display.create_panel())

            # End of normal training (after all epochs) - display final summary
            display.current_phase = "Training Complete"
            display.total_time = time.time() - training_start_time
            live.update(display.create_panel())

            # Save language performance history if available
            if track_languages and language_performance_history:
                language_metrics_json = os.path.join(metrics_output_dir, "language_performance.json")
                with open(language_metrics_json, 'w', encoding='utf-8') as f:
                    json.dump(language_performance_history, f, indent=2, ensure_ascii=False)

            # If we have a best model, rename it to the final user-specified name (for normal training)
            final_path = None
            if save_model_as is not None and best_model_path is not None:
                final_path = f"./models/{save_model_as}"
                # Remove existing final path if any
                if os.path.exists(final_path):
                    shutil.rmtree(final_path)
                shutil.move(best_model_path, final_path)
                best_model_path = final_path
            elif save_model_as is not None and best_model_path is None:
                # Save current model as fallback
                final_path = f"./models/{save_model_as}"
                os.makedirs(final_path, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(final_path, WEIGHTS_NAME)
                output_config_file = os.path.join(final_path, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                self.tokenizer.save_vocabulary(final_path)
                best_model_path = final_path

            # ==================== Reinforced Training Check ====================
            # Use a flag to ensure reinforced learning only triggers once per training session
            if not hasattr(self, '_reinforced_already_triggered'):
                self._reinforced_already_triggered = False

            reinforced_triggered = False
            if best_scores is not None and reinforced_learning and n_epochs_reinforced > 0 and not self._reinforced_already_triggered:
                # Extract metrics from best_scores
                best_precision = best_scores[0]  # (precision_0, precision_1, ...)
                best_recall = best_scores[1]     # (recall_0, recall_1, ...)
                best_f1_scores = best_scores[2]  # (f1_0, f1_1, ...)
                best_support = best_scores[3]    # (support_0, support_1, ...)

                # Handle single-class or multi-class cases
                if len(best_f1_scores) >= 2:
                    best_f1_0 = best_f1_scores[0]
                    best_f1_1 = best_f1_scores[1]
                    best_support_0 = int(best_support[0])
                    best_support_1 = int(best_support[1])
                else:
                    # Single class - use the only F1 score
                    best_f1_0 = 0.0
                    best_f1_1 = best_f1_scores[0]
                    best_support_0 = 0
                    best_support_1 = int(best_support[0])

                # Use intelligent trigger logic
                trigger_score, should_trigger, trigger_reason = self.calculate_reinforced_trigger_score(
                    f1_class_0=best_f1_0,
                    f1_class_1=best_f1_1,
                    support_class_0=best_support_0,
                    support_class_1=best_support_1,
                    language_metrics=best_language_metrics
                )

                # Update display with reinforced learning threshold info
                display.reinforced_threshold = trigger_score
                display.reinforced_triggered = should_trigger

                # Trigger reinforced learning if needed (don't log - would break Rich Live display)
                if should_trigger:
                    reinforced_triggered = True
                    self._reinforced_already_triggered = True  # Mark as triggered to prevent re-triggering

                    # ========== INLINE REINFORCED TRAINING (ROBUST SOLUTION) ==========
                    # Instead of calling separate function, run reinforced training INLINE
                    # within the SAME Live context to ensure display updates properly

                    # Switch display to reinforced mode
                    display.is_reinforced = True
                    display.n_epochs = n_epochs_reinforced
                    display.current_epoch = 0
                    display.current_phase = "🔥 Starting Reinforced Training"

                    # Reset all metrics for clean transition
                    display.train_progress = 0
                    display.val_progress = 0
                    display.train_total = 0
                    display.val_total = 0

                    # IMMEDIATELY update display to show transition (prevents panel stacking)
                    live.update(display.create_panel(), refresh=True)
                    time.sleep(0.2)  # Brief pause for clean visual transition

                    # Prepare reinforced training setup
                    os.makedirs(metrics_output_dir, exist_ok=True)
                    reinforced_metrics_csv = os.path.join(metrics_output_dir, "reinforced_training_metrics.csv")

                    # Create headers for reinforced metrics CSV
                    reinforced_headers = [
                        "model_identifier", "model_type", "label_key", "label_value", "language",
                        "epoch", "train_loss", "val_loss", "accuracy",
                        "precision_0", "recall_0", "f1_0", "support_0",
                        "precision_1", "recall_1", "f1_1", "support_1", "macro_f1"
                    ]

                    if track_languages and language_info is not None:
                        unique_langs = sorted(list(set(language_info)))
                        for lang in unique_langs:
                            reinforced_headers.extend([
                                f"{lang}_accuracy", f"{lang}_precision_0", f"{lang}_recall_0", f"{lang}_f1_0", f"{lang}_support_0",
                                f"{lang}_precision_1", f"{lang}_recall_1", f"{lang}_f1_1", f"{lang}_support_1", f"{lang}_macro_f1"
                            ])

                    with open(reinforced_metrics_csv, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(reinforced_headers)

                    # Extract dataset from train_dataloader and apply WeightedRandomSampler
                    dataset = train_dataloader.dataset
                    labels = dataset.tensors[2].numpy()

                    class_sample_count = np.bincount(labels)
                    weight_per_class = 1.0 / class_sample_count
                    sample_weights = [weight_per_class[t] for t in labels]

                    sampler = WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True
                    )

                    # Build new train dataloader with bigger batch size
                    new_batch_size = 64
                    new_train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=new_batch_size)

                    # Get intelligent reinforced parameters
                    from .reinforced_params import get_reinforced_params, should_use_advanced_techniques

                    model_name_for_params = self.__class__.__name__
                    reinforced_params = get_reinforced_params(model_name_for_params, best_f1_1, lr)
                    advanced_techniques = should_use_advanced_techniques(best_f1_1)

                    new_lr = reinforced_params['learning_rate']
                    pos_weight_val = reinforced_params['class_1_weight']
                    weight_tensor = torch.tensor([1.0, pos_weight_val], dtype=torch.float)

                    if 'n_epochs' in reinforced_params:
                        n_epochs_reinforced = reinforced_params['n_epochs']
                        display.n_epochs = n_epochs_reinforced

                    # Load best model as starting point (suppress logs to avoid interfering with Rich display)
                    if best_model_path is not None:
                        model_state = torch.load(os.path.join(best_model_path, WEIGHTS_NAME), map_location=self.device)
                        model.load_state_dict(model_state)

                    # Create new optimizer and scheduler for reinforced training
                    optimizer = AdamW(model.parameters(), lr=new_lr, eps=1e-8)
                    total_steps = len(new_train_dataloader) * n_epochs_reinforced
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=total_steps
                    )

                    # Track reinforced training time
                    reinforced_start_time = time.time()

                    # Reinforced training loop - INLINE within same Live context
                    # NOTE: Suppress logger during reinforced training to avoid interfering with Rich Live display
                    for epoch in range(n_epochs_reinforced):
                        epoch_start_time = time.time()

                        # Update display for new epoch
                        display.current_epoch = epoch + 1
                        display.current_phase = f"🔥 Reinforced Epoch {epoch + 1}/{n_epochs_reinforced}"
                        display.train_total = len(new_train_dataloader)
                        display.train_progress = 0
                        live.update(display.create_panel())  # ✅ INLINE update - same context

                        # Training phase
                        t0 = time.time()
                        model.train()
                        running_loss = 0.0

                        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

                        for step, train_batch in enumerate(new_train_dataloader):
                            b_inputs = train_batch[0].to(self.device)
                            b_masks = train_batch[1].to(self.device)
                            b_labels = train_batch[2].to(self.device)

                            model.zero_grad()
                            outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                            logits = outputs[0]

                            loss = criterion(logits, b_labels)
                            running_loss += loss.item()

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                            optimizer.step()
                            scheduler.step()

                            # Update display with progress
                            display.train_progress = step + 1
                            display.train_loss = loss.item()
                            display.epoch_time = time.time() - epoch_start_time
                            display.total_time = time.time() - reinforced_start_time
                            live.update(display.create_panel())  # ✅ INLINE update

                        avg_train_loss = running_loss / len(new_train_dataloader)
                        display.train_loss = avg_train_loss
                        display.train_time = time.time() - t0
                        display.current_phase = "Validation (Reinforced)"
                        display.val_total = len(test_dataloader)
                        display.val_progress = 0
                        live.update(display.create_panel())  # ✅ INLINE update

                        # Validation phase
                        model.eval()
                        total_val_loss = 0.0
                        logits_complete = []
                        eval_labels = []

                        for step, test_batch in enumerate(test_dataloader):
                            b_inputs = test_batch[0].to(self.device)
                            b_masks = test_batch[1].to(self.device)
                            b_labels = test_batch[2].to(self.device)
                            eval_labels.extend(b_labels.cpu().numpy())

                            with torch.no_grad():
                                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                            val_loss = outputs.loss
                            val_logits = outputs.logits

                            total_val_loss += val_loss.item()
                            logits_complete.append(val_logits.detach().cpu().numpy())

                            # Update display with validation progress
                            display.val_progress = step + 1
                            display.val_loss = val_loss.item()
                            display.epoch_time = time.time() - epoch_start_time
                            display.total_time = time.time() - reinforced_start_time
                            live.update(display.create_panel())  # ✅ INLINE update

                        avg_val_loss = total_val_loss / len(test_dataloader)
                        logits_complete = np.concatenate(logits_complete, axis=0)
                        val_preds = np.argmax(logits_complete, axis=1).flatten()

                        # Calculate metrics
                        report = classification_report(eval_labels, val_preds, output_dict=True, zero_division=0)
                        class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
                        class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
                        macro_avg = report.get("macro avg", {"f1-score": 0})

                        precision_0 = class_0_metrics["precision"]
                        recall_0 = class_0_metrics["recall"]
                        f1_0 = class_0_metrics["f1-score"]
                        support_0 = class_0_metrics["support"]

                        precision_1 = class_1_metrics["precision"]
                        recall_1 = class_1_metrics["recall"]
                        f1_1 = class_1_metrics["f1-score"]
                        support_1 = class_1_metrics["support"]

                        macro_f1 = macro_avg["f1-score"]
                        accuracy = np.sum(val_preds == np.array(eval_labels)) / len(eval_labels)

                        # Update display with metrics
                        display.val_loss = avg_val_loss
                        display.val_time = time.time() - t0 - display.train_time
                        display.accuracy = accuracy
                        display.precision = [precision_0, precision_1]
                        display.recall = [recall_0, recall_1]
                        display.f1_scores = [f1_0, f1_1]
                        display.f1_macro = macro_f1
                        display.support = [int(support_0), int(support_1)]

                        # Calculate language-specific metrics if tracking
                        language_metrics = {}
                        if track_languages and language_info is not None:
                            unique_languages = list(set(language_info))
                            for lang in sorted(unique_languages):
                                lang_indices = [i for i, l in enumerate(language_info) if l == lang]
                                if not lang_indices:
                                    continue

                                lang_preds = val_preds[lang_indices]
                                lang_labels = np.array(eval_labels)[lang_indices]

                                lang_report = classification_report(lang_labels, lang_preds, output_dict=True, zero_division=0)
                                lang_acc = lang_report.get('accuracy', 0)

                                lang_precision_0 = lang_report.get('0', {}).get('precision', 0) if '0' in lang_report else 0
                                lang_recall_0 = lang_report.get('0', {}).get('recall', 0) if '0' in lang_report else 0
                                lang_f1_0 = lang_report.get('0', {}).get('f1-score', 0) if '0' in lang_report else 0
                                lang_support_0 = int(lang_report.get('0', {}).get('support', 0)) if '0' in lang_report else 0

                                lang_precision_1 = lang_report.get('1', {}).get('precision', 0) if '1' in lang_report else 0
                                lang_recall_1 = lang_report.get('1', {}).get('recall', 0) if '1' in lang_report else 0
                                lang_f1_1 = lang_report.get('1', {}).get('f1-score', 0) if '1' in lang_report else 0
                                lang_support_1 = int(lang_report.get('1', {}).get('support', 0)) if '1' in lang_report else 0

                                lang_macro_f1 = lang_report.get('macro avg', {}).get('f1-score', 0)
                                if lang_macro_f1 == 0 and (lang_f1_0 > 0 or lang_f1_1 > 0):
                                    lang_macro_f1 = (lang_f1_0 + lang_f1_1) / 2.0

                                language_metrics[lang] = {
                                    'accuracy': lang_acc,
                                    'precision_0': lang_precision_0,
                                    'recall_0': lang_recall_0,
                                    'f1_0': lang_f1_0,
                                    'support_0': lang_support_0,
                                    'precision_1': lang_precision_1,
                                    'recall_1': lang_recall_1,
                                    'f1_1': lang_f1_1,
                                    'support_1': lang_support_1,
                                    'macro_f1': lang_macro_f1
                                }

                            # Update display with language metrics
                            display.language_metrics = language_metrics

                        live.update(display.create_panel())  # ✅ INLINE update with all metrics

                        # Write epoch metrics to CSV
                        reinforced_row = [
                            model_identifier if model_identifier else "Unknown",
                            self.__class__.__name__,
                            label_key if label_key else "",
                            label_value if label_value else "",
                            language if language else "MULTI",
                            epoch + 1,
                            avg_train_loss,
                            avg_val_loss,
                            accuracy,
                            precision_0, recall_0, f1_0, int(support_0),
                            precision_1, recall_1, f1_1, int(support_1),
                            macro_f1
                        ]

                        if track_languages and language_info is not None:
                            for lang in sorted(unique_langs):
                                if lang in language_metrics:
                                    lm = language_metrics[lang]
                                    reinforced_row.extend([
                                        lm['accuracy'],
                                        lm['precision_0'], lm['recall_0'], lm['f1_0'], lm['support_0'],
                                        lm['precision_1'], lm['recall_1'], lm['f1_1'], lm['support_1'],
                                        lm['macro_f1']
                                    ])
                                else:
                                    reinforced_row.extend([0] * 10)

                        with open(reinforced_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(reinforced_row)

                        # Check if this is a new best model
                        combined_metric = (1 - f1_class_1_weight) * f1_0 + f1_class_1_weight * f1_1

                        if best_model_criteria == "combined":
                            current_metric = combined_metric
                        elif best_model_criteria == "macro_f1":
                            current_metric = macro_f1
                        elif best_model_criteria == "accuracy":
                            current_metric = accuracy
                        else:
                            current_metric = combined_metric

                        if current_metric > best_metric_val:
                            best_metric_val = current_metric

                            # Save new best model
                            if save_model_as is not None:
                                temp_reinforced_path = f"./models/{save_model_as}_reinforced_temp"
                                os.makedirs(temp_reinforced_path, exist_ok=True)

                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(temp_reinforced_path, WEIGHTS_NAME)
                                output_config_file = os.path.join(temp_reinforced_path, CONFIG_NAME)

                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                self.tokenizer.save_vocabulary(temp_reinforced_path)

                                best_model_path = temp_reinforced_path

                                # Update best_scores
                                best_scores = precision_recall_fscore_support(eval_labels, val_preds, average=None, zero_division=0)

                                # Show in display instead of logger (to avoid breaking Rich Live)
                                display.current_phase = f"🔥 NEW BEST! Metric: {current_metric:.4f}"
                                live.update(display.create_panel())

                    # Finalize reinforced model path
                    if best_model_path and best_model_path.endswith("_reinforced_temp"):
                        final_path = best_model_path.replace("_reinforced_temp", "")
                        if os.path.exists(final_path):
                            shutil.rmtree(final_path)
                        os.rename(best_model_path, final_path)
                        best_model_path = final_path
                        # Don't log here - would interfere with Rich display

                    # Reset display back to normal mode with clean transition
                    display.is_reinforced = False
                    display.current_phase = "✅ Training Complete (Reinforced)"
                    display.train_progress = 0
                    display.val_progress = 0
                    live.update(display.create_panel(), refresh=True)
                    time.sleep(0.15)  # Brief pause for visual clarity

            if track_languages and language_performance_history:
                history_path = os.path.join(metrics_output_dir, "language_metrics_history.json")
                try:
                    with open(history_path, "w", encoding="utf-8") as history_file:
                        json.dump(language_performance_history, history_file, indent=2, ensure_ascii=False)
                except OSError:
                    pass

        # Finally, if reinforced training was triggered and found a better model, it might have placed it
        # in a temporary folder. The method already handles rename at the end. So at this point we are done.
        # Return enhanced scores dictionary for benchmark compatibility
        if best_scores is not None:
            # Convert to dictionary format for new benchmark
            scores_dict = {
                'precision': best_scores[0].tolist() if hasattr(best_scores[0], 'tolist') else best_scores[0],
                'recall': best_scores[1].tolist() if hasattr(best_scores[1], 'tolist') else best_scores[1],
                'f1': best_scores[2].tolist() if hasattr(best_scores[2], 'tolist') else best_scores[2],
                'support': best_scores[3].tolist() if hasattr(best_scores[3], 'tolist') else best_scores[3],
                'macro_f1': np.mean(best_scores[2]) if best_scores[2] is not None else 0,
                'accuracy': np.sum([p * s for p, s in zip(best_scores[0], best_scores[3])]) / np.sum(best_scores[3]) if best_scores[3] is not None else 0,
                'f1_0': best_scores[2][0] if len(best_scores[2]) > 0 else 0,
                'f1_1': best_scores[2][1] if len(best_scores[2]) > 1 else 0,
                'precision_0': best_scores[0][0] if len(best_scores[0]) > 0 else 0,
                'precision_1': best_scores[0][1] if len(best_scores[0]) > 1 else 0,
                'recall_0': best_scores[1][0] if len(best_scores[1]) > 0 else 0,
                'recall_1': best_scores[1][1] if len(best_scores[1]) > 1 else 0,
                'val_loss': val_loss_values[-1] if val_loss_values else 0,
                'best_model_path': best_model_path,
                'reinforced_triggered': reinforced_triggered
            }

            # Add language metrics if available
            if track_languages and language_performance_history:
                scores_dict['language_metrics'] = language_performance_history[-1]['metrics'] if language_performance_history else {}
                scores_dict['language_history'] = language_performance_history

            self.language_metrics_history = language_performance_history
            self.last_training_summary = scores_dict
            self.last_saved_model_path = best_model_path

            return scores_dict

        return best_scores

    def reinforced_training(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            base_model_path: str | None,
            random_state: int = 42,
            metrics_output_dir: str = "./training_logs",
            save_model_as: str | None = None,
            best_model_criteria: str = "combined",
            f1_class_1_weight: float = 0.7,
            previous_best_metric: float = -1.0,
            n_epochs_reinforced: int = 2,
            rescue_low_class1_f1: bool = False,
            f1_1_rescue_threshold: float = 0.0,
            prev_best_f1_1: float = 0.0,
            original_lr: float = 5e-5,
            track_languages: bool = False,
            language_info: Optional[List[str]] = None,
            model_identifier: Optional[str] = None,
            label_key: Optional[str] = None,  # Multi-label: key being trained
            label_value: Optional[str] = None,  # Multi-label: specific value
            language: Optional[str] = None,  # Language of the data
            live: Any = None,  # Live context to reuse (if None, create new)
            display: Any = None  # Display object to reuse (if None, create new)
    ) -> Tuple[float, str | None, Tuple[Any, Any, Any, Any] | None]:
        """
        A "reinforced training" procedure that is triggered if the final best model from normal
        training has F1(class 1) < 0.7 (and reinforced_learning is True). This method:
          - Oversamples class 1 via WeightedRandomSampler.
          - Increases batch size to 64 (by default).
          - Reduces learning rate (e.g., 1/10 of the original normal training).
          - Uses a weighted cross-entropy loss to emphasize class 1.
          - Logs each epoch's metrics to "reinforced_training_metrics.csv".
          - Uses the same best-model selection logic as normal training and logs to best_models.csv
            with a "training_phase" = "reinforced".
          - If `rescue_low_class1_f1` is True and the best normal-training F1 for class 1 was 0,
            then any RL epoch where class 1 F1 becomes > `f1_1_rescue_threshold` is automatically
            selected as the best, overriding the standard combined metric.

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            The original training dataloader (we will rebuild it internally for oversampling).

        test_dataloader: torch.utils.data.DataLoader
            The test/validation dataloader.

        base_model_path: str or None
            Path to the best model saved after normal training. If provided, we load that model
            as the starting point for reinforced training. If None, we load a fresh model from self.model_name.

        random_state: int, default=42
            Random seed for reproducibility.

        metrics_output_dir: str, default="./training_logs"
            Directory for logs (reinforced_training_metrics.csv and best_models.csv).

        save_model_as: str, default=None
            If not None, the final best reinforced model will be saved in ./models/<save_model_as>
            (overwriting any previous normal-training best model if we find a better one).

        best_model_criteria: str, default="combined"
            How to select the best model (same logic as in run_training).

        f1_class_1_weight: float, default=0.7
            Weight for F1(class 1) in the combined metric.

        previous_best_metric: float, default=-1.0
            The best metric value from normal training. We'll only overwrite if we find a better metric here.

        n_epochs_reinforced: int, default=2
            Number of epochs for the reinforced training phase.

        rescue_low_class1_f1: bool, default=False
            If True, then if the best normal model had class 1 F1 == 0, any RL epoch achieving
            class 1 F1 > `f1_1_rescue_threshold` is automatically considered an improvement.

        f1_1_rescue_threshold: float, default=0.0
            The threshold to detect a "small improvement" of class 1 F1 from 0.

        Returns
        -------
        (best_metric_val, best_model_path, best_scores)
            Where:
              - best_metric_val is the updated best metric value after reinforced training.
              - best_model_path is the path to the best model (reinforced if improved).
              - best_scores is the final (precision, recall, f1, support) from sklearn metrics.
        """
        # Reuse display if provided, otherwise create new one
        create_new_live = False
        if display is None:
            display = TrainingDisplay(
                model_name=self.model_name if hasattr(self, 'model_name') else save_model_as or "BERT",
                label_key=label_key,
                label_value=label_value,
                language=language,
                n_epochs=n_epochs_reinforced,
                is_reinforced=True
            )
            create_new_live = True
        else:
            # Switch existing display to reinforced mode
            display.is_reinforced = True
            display.n_epochs = n_epochs_reinforced
            display.current_epoch = 0

        # Track total training time for reinforced phase
        training_start_time = time.time()

        # Prepare new CSV for reinforced training metrics
        os.makedirs(metrics_output_dir, exist_ok=True)
        reinforced_metrics_csv = os.path.join(metrics_output_dir, "reinforced_training_metrics.csv")

        # Create headers for reinforced metrics CSV (include model identifiers)
        reinforced_headers = [
            "model_identifier",
            "model_type",
            "label_key",        # Multi-label: key (e.g., 'themes', 'sentiment')
            "label_value",      # Multi-label: value (e.g., 'transportation', 'positive')
            "language",         # Language of the data (e.g., 'EN', 'FR', 'MULTI')
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",         # Overall accuracy
            "precision_0",
            "recall_0",
            "f1_0",
            "support_0",
            "precision_1",
            "recall_1",
            "f1_1",
            "support_1",
            "macro_f1"
        ]

        # Add language-specific headers if tracking languages
        if track_languages and language_info is not None:
            unique_langs = sorted(list(set(language_info)))
            for lang in unique_langs:
                reinforced_headers.extend([
                    f"{lang}_accuracy",
                    f"{lang}_precision_0",
                    f"{lang}_recall_0",
                    f"{lang}_f1_0",
                    f"{lang}_support_0",
                    f"{lang}_precision_1",
                    f"{lang}_recall_1",
                    f"{lang}_f1_1",
                    f"{lang}_support_1",
                    f"{lang}_macro_f1"
                ])

        with open(reinforced_metrics_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(reinforced_headers)

        # We'll also append to best_models.csv if we find improvements
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Extract the original dataset from train_dataloader to apply WeightedRandomSampler
        dataset = train_dataloader.dataset  # Should be TensorDataset(input_ids, masks, labels)
        labels = dataset.tensors[2].numpy()  # third item = labels

        class_sample_count = np.bincount(labels)
        weight_per_class = 1.0 / class_sample_count  # inverse frequency
        sample_weights = [weight_per_class[t] for t in labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Build a new train dataloader with a bigger batch size
        new_batch_size = 64
        new_train_dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=new_batch_size
        )

        # Get intelligent reinforced parameters based on model and performance
        from .reinforced_params import get_reinforced_params, should_use_advanced_techniques

        # Get the model name for parameter adaptation
        model_name_for_params = self.__class__.__name__
        original_lr = 5e-5  # Default, should be passed from training

        # Get adaptive parameters
        reinforced_params = get_reinforced_params(model_name_for_params, prev_best_f1_1, original_lr)
        advanced_techniques = should_use_advanced_techniques(prev_best_f1_1)

        # Apply parameters
        new_lr = reinforced_params['learning_rate']
        pos_weight_val = reinforced_params['class_1_weight']
        weight_tensor = torch.tensor([1.0, pos_weight_val], dtype=torch.float)

        # Adjust epochs if specified
        if 'n_epochs' in reinforced_params:
            n_epochs_reinforced = reinforced_params['n_epochs']
            display.n_epochs = n_epochs_reinforced

        # Set seeds again
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Load from base_model_path if given, else from self.model_name (suppress warnings)
        self.logger.info(f"Loading model from: {base_model_path if base_model_path else self.model_name}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
            warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
            if base_model_path:
                model = self.model_sequence_classifier.from_pretrained(base_model_path)
            else:
                model = self.model_sequence_classifier.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    output_attentions=False,
                    output_hidden_states=False
                )
        self.logger.info("Model loaded successfully, moving to device...")
        model.to(self.device)
        self.logger.info("Model ready for reinforced training")

        optimizer = AdamW(model.parameters(), lr=new_lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(new_train_dataloader) * n_epochs_reinforced
        )

        best_metric_val = previous_best_metric
        best_model_path_local = base_model_path  # Start from the best model from normal training
        best_scores = None

        # Collect test labels for final metrics
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()

        # Detect if the best normal-training F1 for class 1 was exactly 0
        # We'll use this to trigger the "rescue" logic below.
        best_normal_f1_class_1_was_zero = False
        # If we have a best_scores from normal training, check F1(class1)
        if best_model_path_local and (previous_best_metric != -1.0):
            # Attempt to compute the actual F1 from best_scores
            # But best_scores might come from run_training
            # We'll rely on the prior classification if needed.
            # For safety, let's rely on best_scores if it's stored properly.
            pass  # We'll handle logic if best_scores was carried over
        else:
            # If there's no prior metric or best_model_path, we consider normal training inconclusive
            pass

        # If the user explicitly wants rescue logic, let's see if we have a known F1=0 scenario
        # We'll rely on the fact that if previous_best_metric is > -1, we had a valid model
        # but let's not forcibly re-check that; we do it dynamically later.

        # Update display for reinforced training
        display.current_phase = "Reinforced Training"

        # Debug log
        self.logger.info("🔥 Starting reinforced learning phase...")

        # ROBUST SOLUTION: Reuse the SAME Live context throughout
        # The key is to just update the renderable, not create a new Live
        if create_new_live:
            # Only create NEW context if called standalone (not from run_training)
            live_context = Live(display.create_panel(), refresh_per_second=4)
            live = live_context.__enter__()
            self.logger.info("Created new Live context for reinforced learning")
        else:
            # Reuse existing Live - just update the renderable
            self.logger.info("Reusing existing Live context for reinforced learning")

            # The Live is already active in the parent, just update it
            live.update(display.create_panel())
            live_context = None  # Don't manage the context here

        try:
            # Reinforced training epochs
            self.logger.info(f"Starting {n_epochs_reinforced} reinforced epochs...")
            for epoch in range(n_epochs_reinforced):
                epoch_start_time = time.time()
                self.logger.info(f"🔥 Reinforced Epoch {epoch + 1}/{n_epochs_reinforced}")

                # Update display for new epoch
                display.current_epoch = epoch + 1
                display.current_phase = "Training (Reinforced)"
                display.train_total = len(new_train_dataloader)
                display.train_progress = 0
                live.update(display.create_panel())
                live.refresh()  # Force immediate redraw

                t0 = time.time()
                model.train()
                running_loss = 0.0

                # Weighted cross entropy (for 2 classes) with emphasis on class 1
                criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

                # Training loop - update display directly
                for step, train_batch in enumerate(new_train_dataloader):
                    b_inputs = train_batch[0].to(self.device)
                    b_masks = train_batch[1].to(self.device)
                    b_labels = train_batch[2].to(self.device)

                    model.zero_grad()
                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                    logits = outputs[0]

                    loss = criterion(logits, b_labels)
                    running_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    # Update display with current progress
                    display.train_progress = step + 1
                    display.train_loss = loss.item()
                    display.epoch_time = time.time() - epoch_start_time  # Update elapsed time
                    display.total_time = time.time() - training_start_time
                    live.update(display.create_panel())

                avg_train_loss = running_loss / len(new_train_dataloader)
                display.train_loss = avg_train_loss
                display.train_time = time.time() - t0
                display.current_phase = "Validation (Reinforced)"
                display.val_total = len(test_dataloader)
                display.val_progress = 0
                live.update(display.create_panel())

                # Validation phase
                model.eval()
                total_val_loss = 0.0
                logits_complete = []
                eval_labels = []

                # Validation loop - update display directly
                for step, test_batch in enumerate(test_dataloader):
                    b_inputs = test_batch[0].to(self.device)
                    b_masks = test_batch[1].to(self.device)
                    b_labels = test_batch[2].to(self.device)
                    eval_labels.extend(b_labels.cpu().numpy())

                    with torch.no_grad():
                        outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                    val_loss = outputs.loss
                    val_logits = outputs.logits

                    total_val_loss += val_loss.item()
                    logits_complete.append(val_logits.detach().cpu().numpy())

                    # Update display with validation progress
                    display.val_progress = step + 1
                    display.val_loss = val_loss.item()
                    display.epoch_time = time.time() - epoch_start_time  # Update elapsed time
                    display.total_time = time.time() - display.start_time
                    live.update(display.create_panel())

                avg_val_loss = total_val_loss / len(test_dataloader)
                logits_complete = np.concatenate(logits_complete, axis=0)
                val_preds = np.argmax(logits_complete, axis=1).flatten()

                # Classification report
                report = classification_report(eval_labels, val_preds, output_dict=True)
                class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
                class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
                macro_avg = report.get("macro avg", {"f1-score": 0})

                precision_0 = class_0_metrics["precision"]
                recall_0 = class_0_metrics["recall"]
                f1_0 = class_0_metrics["f1-score"]
                support_0 = class_0_metrics["support"]

                precision_1 = class_1_metrics["precision"]
                recall_1 = class_1_metrics["recall"]
                f1_1 = class_1_metrics["f1-score"]
                support_1 = class_1_metrics["support"]

                macro_f1 = macro_avg["f1-score"]

                # Update display with performance metrics
                display.val_loss = avg_val_loss
                display.val_time = time.time() - t0
                accuracy = np.sum(val_preds == eval_labels) / len(eval_labels)
                display.accuracy = accuracy
                display.precision = [precision_0, precision_1]
                display.recall = [recall_0, recall_1]
                display.f1_scores = [f1_0, f1_1]
                display.f1_macro = macro_f1
                display.support = [int(support_0), int(support_1)]
                live.update(display.create_panel())

                # Calculate language-specific metrics if tracking
                language_metrics = {}
                if track_languages and language_info is not None:
                    unique_languages = list(set(language_info))
                    for lang in sorted(unique_languages):
                        # Get indices for this language
                        lang_indices = [i for i, l in enumerate(language_info) if l == lang]
                        if not lang_indices:
                            continue

                        # Get predictions and labels for this language
                        lang_preds = val_preds[lang_indices]
                        lang_labels = np.array(eval_labels)[lang_indices]

                        # Calculate metrics
                        lang_report = classification_report(lang_labels, lang_preds, output_dict=True, zero_division=0)

                        lang_acc = lang_report.get('accuracy', 0)

                        # Extract detailed metrics for class 0
                        lang_precision_0 = lang_report.get('0', {}).get('precision', 0) if '0' in lang_report else 0
                        lang_recall_0 = lang_report.get('0', {}).get('recall', 0) if '0' in lang_report else 0
                        lang_f1_0 = lang_report.get('0', {}).get('f1-score', 0) if '0' in lang_report else 0
                        lang_support_0 = int(lang_report.get('0', {}).get('support', 0)) if '0' in lang_report else 0

                        # Extract detailed metrics for class 1
                        lang_precision_1 = lang_report.get('1', {}).get('precision', 0) if '1' in lang_report else 0
                        lang_recall_1 = lang_report.get('1', {}).get('recall', 0) if '1' in lang_report else 0
                        lang_f1_1 = lang_report.get('1', {}).get('f1-score', 0) if '1' in lang_report else 0
                        lang_support_1 = int(lang_report.get('1', {}).get('support', 0)) if '1' in lang_report else 0

                        # Get macro F1 from report, or calculate manually if not available
                        lang_macro_f1 = lang_report.get('macro avg', {}).get('f1-score', 0)
                        if lang_macro_f1 == 0 and (lang_f1_0 > 0 or lang_f1_1 > 0):
                            # Manual calculation if classification_report didn't provide it
                            lang_macro_f1 = (lang_f1_0 + lang_f1_1) / 2.0

                        language_metrics[lang] = {
                            'samples': len(lang_indices),
                            'accuracy': lang_acc,
                            'precision_0': lang_precision_0,
                            'recall_0': lang_recall_0,
                            'f1_0': lang_f1_0,
                            'support_0': lang_support_0,
                            'precision_1': lang_precision_1,
                            'recall_1': lang_recall_1,
                            'f1_1': lang_f1_1,
                            'support_1': lang_support_1,
                            'macro_f1': lang_macro_f1
                        }

                    # Update display with language metrics
                    display.language_metrics = language_metrics
                    live.update(display.create_panel())

            # Save epoch metrics to reinforced_training_metrics.csv
            # Get model type for logging
            model_type = self.model_name if hasattr(self, 'model_name') else self.__class__.__name__

            with open(reinforced_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    model_identifier if model_identifier else "",
                    model_type,
                    label_key if label_key else "",         # Add label_key
                    label_value if label_value else "",     # Add label_value
                    language if language else "",           # Add language
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    accuracy,                               # Add accuracy
                    precision_0,
                    recall_0,
                    f1_0,
                    support_0,
                    precision_1,
                    recall_1,
                    f1_1,
                    support_1,
                    macro_f1
                ]

                # Add language metrics if available
                if track_languages and language_info is not None and language_metrics:
                    unique_languages = list(set(language_info))
                    for lang in sorted(unique_languages):
                        if lang in language_metrics:
                            row.extend([
                                language_metrics[lang]['accuracy'],
                                language_metrics[lang]['precision_0'],
                                language_metrics[lang]['recall_0'],
                                language_metrics[lang]['f1_0'],
                                language_metrics[lang]['support_0'],
                                language_metrics[lang]['precision_1'],
                                language_metrics[lang]['recall_1'],
                                language_metrics[lang]['f1_1'],
                                language_metrics[lang]['support_1'],
                                language_metrics[lang]['macro_f1']
                            ])
                        else:
                            row.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                writer.writerow(row)

            # Also append to training_metrics.csv (to consolidate all epochs in one file)
            training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics.csv")
            with open(training_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Get timestamp for this entry
                current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                row = [
                    model_identifier if model_identifier else "",
                    model_type,
                    label_key if label_key else "",         # Add label_key
                    label_value if label_value else "",     # Add label_value
                    language if language else "",           # Add language
                    current_timestamp,
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    accuracy,                               # Add accuracy
                    precision_0,
                    recall_0,
                    f1_0,
                    support_0,
                    precision_1,
                    recall_1,
                    f1_1,
                    support_1,
                    macro_f1
                ]

                # Add language metrics if available
                if track_languages and language_info is not None and language_metrics:
                    unique_languages = list(set(language_info))
                    for lang in sorted(unique_languages):
                        if lang in language_metrics:
                            row.extend([
                                language_metrics[lang]['accuracy'],
                                language_metrics[lang]['precision_0'],
                                language_metrics[lang]['recall_0'],
                                language_metrics[lang]['f1_0'],
                                language_metrics[lang]['support_0'],
                                language_metrics[lang]['precision_1'],
                                language_metrics[lang]['recall_1'],
                                language_metrics[lang]['f1_1'],
                                language_metrics[lang]['support_1'],
                                language_metrics[lang]['macro_f1']
                            ])
                        else:
                            row.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                writer.writerow(row)

            # -- Rescue Logic for Class 1 F1 = 0 from normal training --
            # We'll interpret "best_model_path_local" and "previous_best_metric" to see if normal training
            # might have had class 1's F1 = 0. If so, any improvement above `f1_1_rescue_threshold` is considered better.
            rescue_override = False
            if rescue_low_class1_f1 and previous_best_metric != -1.0:
                # We must check if the best F1_1 from normal training was effectively 0.
                # The simplest check: if combined metric is extremely low, or we keep track separately.
                # Instead, let's rely on classification_report from the last best_scores if possible:
                # best_scores is (precision, recall, f1, support).
                # best_scores[2] -> f1 array, best_scores[2][1] is f1 for class1.
                # If that was 0, we do the rescue logic.
                if best_scores is not None:
                    prev_f1_1 = best_scores[2][1]
                    if prev_f1_1 == 0.0 and f1_1 > f1_1_rescue_threshold:
                        # This RL epoch is automatically an improvement
                        print(f"[Rescue Logic Triggered] Class 1 F1 moved from 0.0 to {f1_1:.4f}, "
                              f"exceeding threshold {f1_1_rescue_threshold:.4f}")
                        rescue_override = True

            # Check if this epoch yields a new best model by normal combined logic
            if best_model_criteria == "combined":
                combined_metric = f1_class_1_weight * f1_1 + (1.0 - f1_class_1_weight) * macro_f1
            else:
                combined_metric = (f1_1 + macro_f1) / 2.0

            # If the rescue logic is triggered, we override combined_metric comparison
            if rescue_override:
                # Force "infinite" improvement to ensure we treat it as a new best
                new_metric_val = combined_metric + 9999.0
            else:
                new_metric_val = combined_metric

            # Standard best-model selection logic
            if new_metric_val > best_metric_val:
                # Update display with best model info
                display.improvement = new_metric_val - best_metric_val
                display.best_f1 = macro_f1
                display.best_epoch = epoch + 1
                live.update(display.create_panel())

                # Remove old best model if needed
                if best_model_path_local is not None and os.path.isdir(best_model_path_local):
                    try:
                        shutil.rmtree(best_model_path_local)
                    except OSError:
                        pass

                best_metric_val = new_metric_val

                # Save new best model to a temporary path
                if save_model_as is not None:
                    best_model_path_local = f"./models/{save_model_as}_reinforced_epoch_{epoch+1}"
                    os.makedirs(best_model_path_local, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_model_path_local, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path_local, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path_local)
                else:
                    best_model_path_local = None

                # Check if this is truly the best model for this model type (reinforced)
                should_update_best = True
                model_type = self.model_name if hasattr(self, 'model_name') else self.__class__.__name__

                if os.path.exists(best_models_csv) and os.path.getsize(best_models_csv) > 0:
                    # Read existing best models to check if this model type already has a better score
                    with open(best_models_csv, 'r', encoding='utf-8') as f_read:
                        reader = csv.DictReader(f_read)
                        for existing_row in reader:
                            # Check if same model type and identifier
                            if (existing_row.get('model_identifier') == (model_identifier if model_identifier else "") and
                                existing_row.get('model_type') == model_type):
                                # Compare scores
                                existing_macro_f1 = float(existing_row.get('macro_f1', 0))
                                if existing_macro_f1 >= macro_f1:
                                    should_update_best = False
                                    break

                if should_update_best:
                    # Remove any existing entry for this model type/identifier
                    if os.path.exists(best_models_csv) and os.path.getsize(best_models_csv) > 0:
                        rows_to_keep = []
                        with open(best_models_csv, 'r', encoding='utf-8') as f_read:
                            reader = csv.DictReader(f_read)
                            headers_dict = reader.fieldnames
                            for row in reader:
                                if not (row.get('model_identifier') == (model_identifier if model_identifier else "") and
                                       row.get('model_type') == model_type):
                                    rows_to_keep.append(row)

                        if headers_dict:
                            with open(best_models_csv, 'w', newline='', encoding='utf-8') as f_write:
                                writer = csv.DictWriter(f_write, fieldnames=headers_dict)
                                writer.writeheader()
                                writer.writerows(rows_to_keep)

                    # Log in best_models.csv
                    with open(best_models_csv, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)

                        # Get timestamp
                        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                        row = [
                            model_identifier if model_identifier else "",
                            model_type,
                            label_key if label_key else "",         # Add label_key
                            label_value if label_value else "",     # Add label_value
                            language if language else "",           # Add language
                            current_timestamp,
                            epoch + 1,
                            avg_train_loss,
                            avg_val_loss,
                            accuracy,                               # Add accuracy
                            precision_0,
                            recall_0,
                            f1_0,
                            support_0,
                            precision_1,
                            recall_1,
                            f1_1,
                            support_1,
                            macro_f1
                        ]

                        # Add language metrics for standard languages (EN, FR)
                        # Always add these columns to maintain CSV consistency
                        standard_languages = ['EN', 'FR']
                        for lang in standard_languages:
                            if track_languages and language_info is not None and language_metrics and lang in language_metrics:
                                row.extend([
                                    language_metrics[lang]['accuracy'],
                                    language_metrics[lang]['precision_0'],
                                    language_metrics[lang]['recall_0'],
                                    language_metrics[lang]['f1_0'],
                                    language_metrics[lang]['support_0'],
                                    language_metrics[lang]['precision_1'],
                                    language_metrics[lang]['recall_1'],
                                    language_metrics[lang]['f1_1'],
                                    language_metrics[lang]['support_1'],
                                    language_metrics[lang]['macro_f1']
                                ])
                            else:
                                # Fill with empty values to maintain CSV structure
                                row.extend(['', '', '', '', '', '', '', '', '', ''])

                        row.extend([
                            best_model_path_local if best_model_path_local else "Not saved to disk",
                            "reinforced"  # training phase
                        ])

                        writer.writerow(row)

                    best_scores = precision_recall_fscore_support(eval_labels, val_preds)

                # Update epoch timing
                display.epoch_time = time.time() - epoch_start_time
                display.total_time = time.time() - display.start_time
                live.update(display.create_panel())

            # After finishing the reinforced epochs, update display
            display.current_phase = "Reinforced Training Complete"
            live.update(display.create_panel())

        finally:
            # Close live context only if we created it
            if create_new_live and live_context:
                live_context.__exit__(None, None, None)

        # After finishing the reinforced epochs, if we have found a better model, rename it to final
        if best_model_path_local and (best_model_path_local != base_model_path):
            # If user wants to save the final best model
            if save_model_as is not None and best_model_path_local is not None:
                final_path = f"./models/{save_model_as}"
                if os.path.exists(final_path):
                    shutil.rmtree(final_path)
                os.rename(best_model_path_local, final_path)
                best_model_path_local = final_path
            elif save_model_as is not None and best_model_path_local is None:
                # Keep the previous best model if it exists
                best_model_path_local = f"./models/{save_model_as}"
                if not os.path.exists(best_model_path_local):
                    # Save current model as fallback
                    os.makedirs(best_model_path_local, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_model_path_local, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path_local, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path_local)

        return best_metric_val, best_model_path_local, best_scores

    def predict(
            self,
            dataloader: DataLoader,
            model: Any,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Predict with a trained model.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            Dataloader for prediction, from self.encode().

        model: huggingface model
            A trained model to use for inference.

        proba: bool, default=True
            If True, return probability distributions (softmax).
            If False, return raw logits.

        progress_bar: bool, default=True
            If True, display a progress bar during prediction.

        Returns
        -------
        pred: ndarray of shape (n_samples, n_labels)
            Probabilities or logits for each sample.
        """
        logits_complete = []
        if progress_bar:
            loader = tqdm(dataloader, desc="Predicting")
        else:
            loader = dataloader

        model.eval()

        for batch in loader:
            batch = tuple(t.to(self.device) for t in batch)
            if len(batch) == 3:
                b_input_ids, b_input_mask, _ = batch
            else:
                b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            logits_complete.append(logits)

            del outputs
            torch.cuda.empty_cache()

        pred = np.concatenate(logits_complete, axis=0)

        if progress_bar:
            print(f"label ids: {self.dict_labels}")

        return softmax(pred, axis=1) if proba else pred

    def load_model(
            self,
            model_path: str
    ):
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        model_path: str
            Path to the saved model folder.

        Returns
        -------
        model: huggingface model
            The loaded model instance.
        """
        return self.model_sequence_classifier.from_pretrained(model_path)

    def predict_with_model(
            self,
            dataloader: DataLoader,
            model_path: str,
            proba: bool = True,
            progress_bar: bool = True
    ):
        """
        Convenience method that loads a model from the specified path, moves it to self.device,
        and performs prediction on the given dataloader.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for prediction.

        model_path: str
            Path to the model to load.

        proba: bool, default=True
            If True, return probability distributions (softmax).
            If False, return raw logits.

        progress_bar: bool, default=True
            If True, display a progress bar during prediction.

        Returns
        -------
        ndarray
            Probability or logit predictions.
        """
        model = self.load_model(model_path)
        model.to(self.device)
        return self.predict(dataloader, model, proba, progress_bar)

    def format_time(
            self,
            elapsed: float | int
    ) -> str:
        """
        Format a time duration to hh:mm:ss.

        Parameters
        ----------
        elapsed: float or int
            Elapsed time in seconds.

        Returns
        -------
        str
            The time in hh:mm:ss format.
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

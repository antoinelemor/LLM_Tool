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

from AugmentedSocialScientistFork.bert_abc import BertABC
from AugmentedSocialScientistFork.logging_utils import get_logger


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
                self.logger.info('Detected Apple Silicon MPS backend. Using the MPS device.')
            # Otherwise, use CPU
            else:
                self.device = torch.device("cpu")
                self.logger.info('Falling back to CPU execution.')

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

        if label_mapping is not None and progress_bar:
            self.logger.info("Label ids mapping: %s", label_mapping)

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

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
            reinforced_f1_threshold: float = 0.7  # Nouveau paramètre pour le seuil de déclenchement
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
        # Ensure metric output directory exists
        os.makedirs(metrics_output_dir, exist_ok=True)
        training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics.csv")
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Initialize CSV for normal training metrics
        csv_headers = [
            "model_identifier",
            "model_name",
            "timestamp",
            "epoch",
            "train_loss",
            "val_loss",
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
            "timestamp",
            "epoch",
            "train_loss",
            "val_loss",
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

        # Initialize the model
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
        language_performance_history = []  # Store language metrics for each epoch
        self.language_metrics_history = []
        if reinforced_epochs is not None:
            n_epochs_reinforced = reinforced_epochs

        # =============== Normal Training Loop ===============
        training_start_time = time.time()  # Initialize the timer

        # Initialize metrics tracking
        training_metrics = []

        self.logger.info("\n%s", f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.CYAN}{'TRAINING START':^80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

        if track_languages and language_info:
            unique_langs_logged = ", ".join(sorted(set(language_info)))
            self.logger.info("Tracking per-language validation metrics for: %s", unique_langs_logged)

        for i_epoch in range(n_epochs):
            epoch_start_time = time.time()

            # Epoch header with color
            self.logger.info("%s", f"\n{Fore.YELLOW}{'━'*80}{Style.RESET_ALL}")
            self.logger.info("%s", f"{Fore.YELLOW}  Epoch {i_epoch + 1}/{n_epochs}{Style.RESET_ALL}")
            self.logger.info("%s", f"{Fore.YELLOW}{'━'*80}{Style.RESET_ALL}\n")

            # Training phase
            self.logger.info("%s", f"{Fore.GREEN}📚 Training Phase{Style.RESET_ALL}")

            t0 = time.time()
            total_train_loss = 0.0
            model.train()

            # Create progress bar for training batches
            train_pbar = tqdm(train_dataloader,
                            desc=f"  Training",
                            unit="batch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                            colour='green')

            for step, train_batch in enumerate(train_pbar):

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

                # Update progress bar with current loss
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_pbar.close()
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_loss_values.append(avg_train_loss)

            self.logger.info("  ✓ Average training loss: %s%.4f%s", Fore.CYAN, avg_train_loss, Style.RESET_ALL)
            self.logger.info("  ⏱  Training time: %s", self.format_time(time.time() - t0))

            # =============== Validation after this epoch ===============
            # Show language information if available
            if track_languages and language_info:
                unique_langs = list(set(language_info))
                lang_str = ", ".join(sorted(unique_langs))
                self.logger.info("%s", f"\n{Fore.BLUE}🔍 Validation Phase (Languages: {lang_str}){Style.RESET_ALL}")
            else:
                self.logger.info("%s", f"\n{Fore.BLUE}🔍 Validation Phase{Style.RESET_ALL}")

            t0 = time.time()
            model.eval()

            total_val_loss = 0.0
            logits_complete = []

            # Create progress bar for validation batches
            val_pbar = tqdm(test_dataloader,
                          desc=f"  Validating",
                          unit="batch",
                          bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          colour='blue')

            for test_batch in val_pbar:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)

                with torch.no_grad():
                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                logits_complete.append(logits.detach().cpu().numpy())

                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            val_pbar.close()
            logits_complete = np.concatenate(logits_complete, axis=0)
            avg_val_loss = total_val_loss / len(test_dataloader)
            val_loss_values.append(avg_val_loss)

            self.logger.info("  ✓ Average validation loss: %s%.4f%s", Fore.CYAN, avg_val_loss, Style.RESET_ALL)
            self.logger.info("  ⏱  Validation time: %s", self.format_time(time.time() - t0))

            preds = np.argmax(logits_complete, axis=1).flatten()
            report = classification_report(test_labels, preds, target_names=label_names, output_dict=True)

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

            # Print metrics in a beautiful table format
            self.logger.info("%s", f"\n{Fore.MAGENTA}📊 Performance Metrics{Style.RESET_ALL}")

            # Create metrics table
            metrics_table = [
                ["Metric", "Class 0", "Class 1", "Overall"],
                ["─" * 15, "─" * 15, "─" * 15, "─" * 15],
                ["Precision", f"{precision_0:.3f}", f"{precision_1:.3f}", ""],
                ["Recall", f"{recall_0:.3f}", f"{recall_1:.3f}", ""],
                ["F1-Score", f"{f1_0:.3f}", f"{f1_1:.3f}", f"{macro_f1:.3f}"],
                ["Support", f"{support_0}", f"{support_1}", f"{support_0 + support_1}"]
            ]

            for row in metrics_table:
                if row[0] == "F1-Score" and f1_1 >= 0.7:
                    # Highlight good F1 score for class 1
                    self.logger.info("  %s", f"{row[0]:<15} {row[1]:<15} {Fore.GREEN}{row[2]:<15}{Style.RESET_ALL} {row[3]:<15}")
                elif row[0] == "F1-Score" and f1_1 < 0.5:
                    # Highlight poor F1 score for class 1
                    self.logger.info("  %s", f"{row[0]:<15} {row[1]:<15} {Fore.RED}{row[2]:<15}{Style.RESET_ALL} {row[3]:<15}")
                else:
                    self.logger.info("  %s", f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")

            # Calculate and display per-language metrics if requested
            if track_languages and language_info is not None:
                self.logger.info("%s", f"\n{Fore.MAGENTA}🌍 Per-Language Performance (Epoch {i_epoch + 1}){Style.RESET_ALL}")
                unique_validation_langs = sorted(set(language_info))
                self.logger.info(
                    "   Validating %d language(s): %s",
                    len(unique_validation_langs),
                    ", ".join(unique_validation_langs)
                )

                lang_headers = ["Language", "Samples", "Accuracy", "F1 Class 0", "F1 Class 1", "Macro F1", "Balance"]
                self.logger.info(
                    "  %s",
                    f"{lang_headers[0]:<12} {lang_headers[1]:<10} {lang_headers[2]:<12} {lang_headers[3]:<12} {lang_headers[4]:<12} {lang_headers[5]:<12} {lang_headers[6]:<10}",
                )
                self.logger.info("  %s", f"{'─'*12} {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

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

                    # Calculate class balance
                    balance_ratio = lang_support_1 / (lang_support_0 + lang_support_1) if (lang_support_0 + lang_support_1) > 0 else 0
                    balance_str = f"{balance_ratio:.2%}"

                    # Color code based on performance
                    acc_color = Fore.GREEN if lang_acc >= 0.8 else Fore.YELLOW if lang_acc >= 0.6 else Fore.RED
                    f1_color = Fore.GREEN if lang_f1_1 >= 0.7 else Fore.YELLOW if lang_f1_1 >= 0.5 else Fore.RED
                    balance_color = Fore.GREEN if 0.3 <= balance_ratio <= 0.7 else Fore.YELLOW if 0.2 <= balance_ratio <= 0.8 else Fore.RED

                    self.logger.info(
                        "  %s",
                        f"{lang:<12} {lang_support:<10} {acc_color}{lang_acc:<12.3f}{Style.RESET_ALL} {lang_f1_0:<12.3f} {f1_color}{lang_f1_1:<12.3f}{Style.RESET_ALL} {lang_macro_f1:<12.3f} {balance_color}{balance_str:<10}{Style.RESET_ALL}",
                    )

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

                self.logger.info("%s", "-" * 90)

                averages: Optional[Dict[str, float]] = None
                if language_metrics:
                    avg_acc = sum(m['accuracy'] for m in language_metrics.values()) / len(language_metrics)
                    avg_f1 = sum(m['macro_f1'] for m in language_metrics.values()) / len(language_metrics)
                    averages = {'accuracy': avg_acc, 'macro_f1': avg_f1}
                    self.logger.info(
                        "  %s",
                        f"{'AVERAGE':<12} {'':<10} {avg_acc:<12.3f} {'':<12} {'':<12} {avg_f1:<12.3f}"
                    )
                self.logger.info("%s", "=" * 90)

                # Store language metrics for this epoch
                if 'language_metrics' in locals():
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
                    current_timestamp,
                    i_epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
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
            else:
                # Fallback or alternative strategy
                combined_metric = (f1_1 + macro_f1) / 2.0

            # Enhanced epoch summary with language info
            self.logger.info("%s", f"\n{Fore.CYAN}─── Epoch {i_epoch + 1}/{n_epochs} Summary ───{Style.RESET_ALL}")
            epoch_time = time.time() - epoch_start_time
            self.logger.info("  ⏱  Epoch Time: %s", self.format_time(epoch_time))
            self.logger.info("  📊 Combined Metric: %.4f", combined_metric)

            # Add language-specific summary if available
            if track_languages and language_info and 'language_metrics' in locals():
                self.logger.info("  🌍 Languages trained: %d", len(language_metrics))
                best_lang = max(language_metrics.items(), key=lambda x: x[1]['macro_f1'])
                worst_lang = min(language_metrics.items(), key=lambda x: x[1]['macro_f1'])
                self.logger.info("  📈 Best performing: %s (F1: %.3f)", best_lang[0], best_lang[1]['macro_f1'])
                self.logger.info("  📉 Needs attention: %s (F1: %.3f)", worst_lang[0], worst_lang[1]['macro_f1'])

            if combined_metric > best_metric_val:
                # We found a new best model
                self.logger.info("%s", f"  {Fore.GREEN}✨ NEW BEST MODEL! (Δ +{combined_metric - best_metric_val:.4f}){Style.RESET_ALL}")
                # Remove old best model folder if it exists
                if best_model_path is not None:
                    try:
                        shutil.rmtree(best_model_path)
                    except OSError:
                        pass

                best_metric_val = combined_metric

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
                            current_timestamp,
                            i_epoch + 1,
                            avg_train_loss,
                            avg_val_loss,
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

                best_scores = precision_recall_fscore_support(test_labels, preds)

        # End of normal training
        self.logger.info("%s", f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.CYAN}{'TRAINING COMPLETE':^80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

        self.logger.info("%s", f"{Fore.YELLOW}📈 Training Summary Dashboard{Style.RESET_ALL}\n")

        total_time = time.time() - training_start_time
        self.logger.info("  ⏱  Total Training Time: %s", self.format_time(total_time))
        self.logger.info("  🔄 Total Epochs: %d", n_epochs)
        self.logger.info("  📦 Batch Size: %d", train_dataloader.batch_size)
        self.logger.info("  📚 Training Samples: %d", len(train_dataloader.dataset))
        self.logger.info("  🧪 Validation Samples: %d", len(test_dataloader.dataset))

        if best_scores is not None:
            self.logger.info("%s", f"\n{Fore.YELLOW}🏆 Best Model Performance{Style.RESET_ALL}")
            best_f1_0 = best_scores[2][0]
            best_f1_1 = best_scores[2][1]
            best_macro = (best_f1_0 + best_f1_1) / 2

            perf_table = [
                ["", "F1-Score"],
                ["─" * 15, "─" * 15],
                ["Class 0", f"{best_f1_0:.3f}"],
                ["Class 1", f"{best_f1_1:.3f}" if best_f1_1 >= 0.7 else f"{Fore.YELLOW}{best_f1_1:.3f}{Style.RESET_ALL}" if best_f1_1 >= 0.5 else f"{Fore.RED}{best_f1_1:.3f}{Style.RESET_ALL}"],
                ["Macro Avg", f"{best_macro:.3f}"]
            ]

            for row in perf_table:
                self.logger.info("  %s", f"{row[0]:<15} {row[1]}")

        # Loss Evolution
        if train_loss_values and val_loss_values:
            self.logger.info("%s", f"\n{Fore.YELLOW}📉 Loss Evolution{Style.RESET_ALL}")
            self.logger.info(
                "  Training Loss:   %.4f → %.4f (Δ %+0.4f)",
                train_loss_values[0],
                train_loss_values[-1],
                train_loss_values[-1] - train_loss_values[0],
            )
            self.logger.info(
                "  Validation Loss: %.4f → %.4f (Δ %+0.4f)",
                val_loss_values[0],
                val_loss_values[-1],
                val_loss_values[-1] - val_loss_values[0],
            )

        # Save language performance history if available
        if track_languages and language_performance_history:
            language_metrics_json = os.path.join(metrics_output_dir, "language_performance.json")
            with open(language_metrics_json, 'w', encoding='utf-8') as f:
                json.dump(language_performance_history, f, indent=2, ensure_ascii=False)
            self.logger.info("  💾 Language performance saved to: %s", language_metrics_json)

        # If we have a best model, rename it to the final user-specified name (for normal training)
        final_path = None
        if save_model_as is not None and best_model_path is not None:
            final_path = f"./models/{save_model_as}"
            # Remove existing final path if any
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.move(best_model_path, final_path)
            best_model_path = final_path
            self.logger.info("Best model from normal training is available at: %s", best_model_path)
        elif save_model_as is not None and best_model_path is None:
            self.logger.warning("No best model was found during training (combined metric never improved)")
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
            self.logger.info("Current model saved as fallback at: %s", best_model_path)

        # ==================== Reinforced Training Check ====================
        print("\n" + "="*80)
        print("REINFORCED TRAINING CHECK")
        print("="*80)

        self.logger.info("="*80)
        self.logger.info("REINFORCED TRAINING CHECK")
        self.logger.info("="*80)

        reinforced_triggered = False
        if best_scores is not None:
            best_f1_1 = best_scores[2][1]  # best_scores = (precision, recall, f1, support)

            # Debug avec PRINT pour être sûr de voir
            print(f"🔍 Reinforced check:")
            print(f"   - Model: {self.__class__.__name__}")
            print(f"   - F1_1: {best_f1_1:.3f}")
            print(f"   - Threshold: {reinforced_f1_threshold:.3f}")
            print(f"   - reinforced_learning: {reinforced_learning}")
            print(f"   - n_epochs_reinforced: {n_epochs_reinforced}")
            print(f"   - Will trigger? {best_f1_1 < reinforced_f1_threshold and reinforced_learning and n_epochs_reinforced > 0}")

            # Debug logging aussi
            self.logger.info(f"🔍 Reinforced check: F1_1={best_f1_1:.3f} vs threshold={reinforced_f1_threshold:.3f}")
            self.logger.info(f"   - reinforced_learning enabled: {reinforced_learning}")
            self.logger.info(f"   - n_epochs_reinforced: {n_epochs_reinforced}")

            if not reinforced_learning:
                print("⚠️ Reinforced learning DÉSACTIVÉ")
                self.logger.warning("⚠️ Reinforced learning DÉSACTIVÉ - ne se déclenchera pas même si F1_1 < seuil")
            elif n_epochs_reinforced == 0:
                print("⚠️ n_epochs_reinforced = 0")
                self.logger.warning("⚠️ n_epochs_reinforced = 0 - pas d'époques de reinforced configurées!")
            elif best_f1_1 < reinforced_f1_threshold and reinforced_learning and n_epochs_reinforced > 0:
                reinforced_triggered = True
                self.logger.warning(
                    "The best model's F1 score for class 1 (%.3f) is below %.2f. Triggering reinforced training...",
                    best_f1_1,
                    reinforced_f1_threshold,
                )

                # Perform reinforced training
                # This returns updated best_metric_val, best_model_path, best_scores
                (best_metric_val,
                 best_model_path,
                 best_scores) = self.reinforced_training(
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    base_model_path=best_model_path,
                    random_state=random_state,
                    metrics_output_dir=metrics_output_dir,
                    save_model_as=save_model_as,
                    best_model_criteria=best_model_criteria,
                    f1_class_1_weight=f1_class_1_weight,
                    previous_best_metric=best_metric_val,
                    n_epochs_reinforced=n_epochs_reinforced,
                    rescue_low_class1_f1=rescue_low_class1_f1,
                    f1_1_rescue_threshold=f1_1_rescue_threshold,
                    prev_best_f1_1=best_f1_1,  # Pass the F1_1 for adaptive parameters
                    original_lr=lr,  # Pass original LR
                    track_languages=track_languages,
                    language_info=language_info,
                    model_identifier=model_identifier  # Pass model_identifier for CSV logging
                )
            else:
                self.logger.info("No reinforced training triggered.")
        else:
            self.logger.warning("No valid best scores found after normal training (unexpected). Reinforced training skipped.")

        if track_languages and language_performance_history:
            history_path = os.path.join(metrics_output_dir, "language_metrics_history.json")
            try:
                with open(history_path, "w", encoding="utf-8") as history_file:
                    json.dump(language_performance_history, history_file, indent=2, ensure_ascii=False)
                self.logger.info("Saved per-language validation history to %s", history_path)
            except OSError as exc:  # pragma: no cover - filesystem-specific failures
                self.logger.warning("Could not persist language history to %s: %s", history_path, exc)

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
            model_identifier: Optional[str] = None
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
        # Reinforced training header with color formatting
        self.logger.info("\n%s", f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.MAGENTA}{'REINFORCED TRAINING START':^80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")

        # Prepare new CSV for reinforced training metrics
        os.makedirs(metrics_output_dir, exist_ok=True)
        reinforced_metrics_csv = os.path.join(metrics_output_dir, "reinforced_training_metrics.csv")

        # Create headers for reinforced metrics CSV
        reinforced_headers = [
            "epoch",
            "train_loss",
            "val_loss",
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
            self.logger.info(f"Adjusted reinforced epochs: {n_epochs_reinforced} (was {n_epochs_reinforced})")

        # Display parameters with formatting
        self.logger.info("%s", f"{Fore.CYAN}🎯 Reinforced Training Parameters{Style.RESET_ALL}")
        self.logger.info("  Model: %s", model_name_for_params)
        self.logger.info("  Learning rate: %s", f"{Fore.GREEN}{new_lr:.2e}{Style.RESET_ALL}")
        self.logger.info("  Class 1 weight: %s", f"{Fore.GREEN}{pos_weight_val:.1f}{Style.RESET_ALL}")
        self.logger.info("  Epochs: %s", f"{Fore.GREEN}{n_epochs_reinforced}{Style.RESET_ALL}")
        active_techniques = [k.replace('use_', '') for k, v in advanced_techniques.items() if v]
        if active_techniques:
            self.logger.info("  Advanced techniques: %s", f"{Fore.CYAN}{', '.join(active_techniques)}{Style.RESET_ALL}")
        self.logger.info("")

        # Set seeds again
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Load from base_model_path if given, else from self.model_name
        if base_model_path:
            model = self.model_sequence_classifier.from_pretrained(base_model_path)
            self.logger.info("📥 Loaded base model from: %s", base_model_path)
        else:
            model = self.model_sequence_classifier.from_pretrained(
                self.model_name,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
            self.logger.info("📥 Using fresh model from: %s", self.model_name)
        model.to(self.device)

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

        # Reinforced training epochs
        for epoch in range(n_epochs_reinforced):
            epoch_start_time = time.time()

            # Epoch header with color (matching normal training)
            self.logger.info("%s", f"\n{Fore.MAGENTA}{'━'*80}{Style.RESET_ALL}")
            self.logger.info("%s", f"{Fore.MAGENTA}  Reinforced Epoch {epoch + 1}/{n_epochs_reinforced}{Style.RESET_ALL}")
            self.logger.info("%s", f"{Fore.MAGENTA}{'━'*80}{Style.RESET_ALL}\n")

            # Training phase
            self.logger.info("%s", f"{Fore.GREEN}📚 Training Phase{Style.RESET_ALL}")

            t0 = time.time()
            model.train()
            running_loss = 0.0

            # Weighted cross entropy (for 2 classes) with emphasis on class 1
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

            # Create progress bar for training batches
            train_pbar = tqdm(new_train_dataloader,
                            desc=f"  Training",
                            unit="batch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                            colour='green')

            for step, train_batch in enumerate(train_pbar):
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

            avg_train_loss = running_loss / len(new_train_dataloader)
            elapsed_str = self.format_time(time.time() - t0)

            # Training summary
            self.logger.info("")
            self.logger.info("  Training complete")
            self.logger.info("    Average loss: %s", f"{Fore.YELLOW}{avg_train_loss:.4f}{Style.RESET_ALL}")
            self.logger.info("    Time elapsed: %s", elapsed_str)
            self.logger.info("")

            # Validation phase
            self.logger.info("%s", f"{Fore.BLUE}🔍 Validation Phase{Style.RESET_ALL}")

            model.eval()
            total_val_loss = 0.0
            logits_complete = []
            eval_labels = []

            # Create progress bar for validation
            val_pbar = tqdm(test_dataloader,
                          desc=f"  Validating",
                          unit="batch",
                          bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                          colour='blue')

            for test_batch in val_pbar:
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

            # Display validation metrics with formatting
            self.logger.info("")
            self.logger.info("  Validation complete")
            self.logger.info("    Validation loss: %s", f"{Fore.YELLOW}{avg_val_loss:.4f}{Style.RESET_ALL}")
            self.logger.info("")

            # Display classification metrics in a table format
            self.logger.info("%s", f"{Fore.CYAN}📊 Performance Metrics{Style.RESET_ALL}")
            self.logger.info("  %s", "Class    Precision  Recall     F1-score   Support")
            self.logger.info("  %s", "─"*60)

            # Color code F1 scores
            f1_0_color = Fore.GREEN if f1_0 >= 0.7 else Fore.YELLOW if f1_0 >= 0.5 else Fore.RED
            f1_1_color = Fore.GREEN if f1_1 >= 0.7 else Fore.YELLOW if f1_1 >= 0.5 else Fore.RED

            self.logger.info("  %s", f"0        {precision_0:.4f}     {recall_0:.4f}     {f1_0_color}{f1_0:.4f}{Style.RESET_ALL}     {int(support_0)}")
            self.logger.info("  %s", f"1        {precision_1:.4f}     {recall_1:.4f}     {f1_1_color}{f1_1:.4f}{Style.RESET_ALL}     {int(support_1)}")
            self.logger.info("  %s", "─"*60)
            macro_color = Fore.GREEN if macro_f1 >= 0.7 else Fore.YELLOW if macro_f1 >= 0.5 else Fore.RED
            self.logger.info("  %s", f"Macro F1: {macro_color}{macro_f1:.4f}{Style.RESET_ALL}")
            self.logger.info("")

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

                    lang_macro_f1 = lang_report.get('macro avg', {}).get('f1-score', 0)

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

                # Display language metrics with formatting
                self.logger.info("%s", f"{Fore.CYAN}🌍 Language-Specific Metrics{Style.RESET_ALL}")
                self.logger.info("  %s", f"{'Language':<12} {'Accuracy':<12} {'F1 Class 0':<12} {'F1 Class 1':<12} {'Macro F1':<12}")
                self.logger.info("  %s", f"{'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
                for lang in sorted(language_metrics.keys()):
                    metrics = language_metrics[lang]
                    # Color code F1 class 1
                    lang_f1_1_color = Fore.GREEN if metrics['f1_1'] >= 0.7 else Fore.YELLOW if metrics['f1_1'] >= 0.5 else Fore.RED
                    self.logger.info("  %s", f"{lang:<12} {metrics['accuracy']:<12.3f} {metrics['f1_0']:<12.3f} {lang_f1_1_color}{metrics['f1_1']:<12.3f}{Style.RESET_ALL} {metrics['macro_f1']:<12.3f}")
                self.logger.info("")

            # Save epoch metrics to reinforced_training_metrics.csv
            with open(reinforced_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
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
                self.logger.info("%s", f"{Fore.GREEN}✓ New best model! Combined metric: {combined_metric:.4f}{Style.RESET_ALL}")
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
                            current_timestamp,
                            epoch + 1,
                            avg_train_loss,
                            avg_val_loss,
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
        else:
            # Even if we didn't find a better model, calculate the final scores
            # Use the last evaluation results
            if 'val_preds' in locals() and 'eval_labels' in locals():
                best_scores = precision_recall_fscore_support(eval_labels, val_preds)
            else:
                # If no evaluation was done, keep the scores from normal training
                # (passed as prev_best_f1_1 but we need the full scores)
                # This shouldn't happen in normal circumstances
                best_scores = None

        # After finishing the reinforced epochs, if we have found a better model, rename it to final
        if best_model_path_local and (best_model_path_local != base_model_path):
            # If user wants to save the final best model
            if save_model_as is not None and best_model_path_local is not None:
                final_path = f"./models/{save_model_as}"
                if os.path.exists(final_path):
                    shutil.rmtree(final_path)
                os.rename(best_model_path_local, final_path)
                best_model_path_local = final_path
                self.logger.info("")
                self.logger.info("%s", f"{Fore.GREEN}💾 Best model saved at: {best_model_path_local}{Style.RESET_ALL}")
            elif save_model_as is not None and best_model_path_local is None:
                self.logger.warning("%s", f"{Fore.YELLOW}⚠️  No improvement found during reinforced training{Style.RESET_ALL}")
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
                    self.logger.info("%s", f"{Fore.CYAN}💾 Current model saved as fallback at: {best_model_path_local}{Style.RESET_ALL}")

        # Completion message
        self.logger.info("\n%s", f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.MAGENTA}{'REINFORCED TRAINING COMPLETE':^80}{Style.RESET_ALL}")
        self.logger.info("%s", f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}\n")
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

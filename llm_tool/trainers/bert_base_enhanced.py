"""
PROJECT:
-------
LLMTool

TITLE:
------
bert_base_enhanced.py

MAIN OBJECTIVE:
---------------
This script extends BertBase with enhanced metadata support, enabling per-language
performance tracking, stratified analysis by custom fields, and comprehensive
multilingual metrics logging for social science research.

Dependencies:
-------------
- torch (PyTorch for deep learning)
- transformers (HuggingFace transformers library)
- numpy (numerical operations)
- scikit-learn (metrics calculation)
- LLMTool.data_utils (metadata handling)

MAIN FEATURES:
--------------
1) Full metadata support (ID, language, custom fields) throughout training pipeline
2) Per-language performance tracking with detailed metrics per epoch
3) Enhanced data loading with MetadataDataset for preserving sample metadata
4) Stratified analysis by any metadata field (language, source, category, etc.)
5) Comprehensive JSON logging of language-specific performance
6) Confusion matrices per language/metadata group
7) Distribution statistics and balance tracking
8) Backward compatible with standard BertBase interface

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
from typing import List, Tuple, Any, Optional, Dict, Union
from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax
from torch.types import Device
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

try:
    from torch.optim import AdamW
except ImportError:
    from transformers.optimization import AdamW

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME
)

from llm_tool.trainers.bert_abc import BertABC
from llm_tool.trainers.data_utils import (
    MetadataDataset,
    PerformanceTracker,
    DataSample,
    extract_samples_data
)


class BertBaseEnhanced(BertABC):
    """Enhanced BERT base class with metadata and per-language tracking support."""

    def __init__(
            self,
            model_name: str = 'bert-base-cased',
            tokenizer: Any = BertTokenizer,
            model_sequence_classifier: Any = BertForSequenceClassification,
            device: Device | None = None,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.model_sequence_classifier = model_sequence_classifier
        self.dict_labels = None
        self.performance_tracker = None

        # Set or detect device
        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print('There are %d GPU(s) available.' % torch.cuda.device_count())
                print('We will use GPU {}:'.format(torch.cuda.current_device()),
                      torch.cuda.get_device_name(torch.cuda.current_device()))
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print('MPS is available. Using the Apple Silicon GPU!')
            else:
                self.device = torch.device("cpu")
                print('No GPU available, using the CPU instead.')

    def encode_with_metadata(
            self,
            samples: List[DataSample],
            batch_size: int = 32,
            progress_bar: bool = True,
            add_special_tokens: bool = True
    ) -> DataLoader:
        """
        Encode data samples with metadata preservation.

        Parameters
        ----------
        samples: List[DataSample]
            List of DataSample objects with text, labels and metadata

        batch_size: int, default=32
            Batch size for the PyTorch DataLoader

        progress_bar: bool, default=True
            If True, print a progress bar

        add_special_tokens: bool, default=True
            If True, add '[CLS]' and '[SEP]' tokens

        Returns
        -------
        dataloader: DataLoader
            Enhanced DataLoader with metadata support
        """
        texts, labels, metadata = extract_samples_data(samples)

        # Tokenize
        input_ids = []
        if progress_bar:
            text_loader = tqdm(texts, desc="Tokenizing")
        else:
            text_loader = texts

        for text in text_loader:
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens
            )
            input_ids.append(encoded)

        # Calculate max length (capped at 512)
        max_len = min(max([len(sen) for sen in input_ids]), 512)

        # Pad/truncate
        pad = np.full((len(input_ids), max_len), 0, dtype='long')
        for idx, s in enumerate(input_ids):
            trunc = s[:max_len]
            pad[idx, :len(trunc)] = trunc

        input_ids = pad

        # Create attention masks
        attention_masks = []
        if progress_bar:
            input_loader = tqdm(input_ids, desc="Creating attention masks")
        else:
            input_loader = input_ids

        for sent in input_loader:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        # Build label dictionary
        label_names = np.unique(labels)
        self.dict_labels = dict(zip(label_names, range(len(label_names))))

        if progress_bar:
            print(f"label ids: {self.dict_labels}")

        # Create tensors
        inputs_tensors = torch.tensor(input_ids)
        masks_tensors = torch.tensor(attention_masks)
        labels_tensors = torch.tensor([self.dict_labels[x] for x in labels])

        # Create dataset with metadata
        dataset = MetadataDataset(
            inputs_tensors,
            masks_tensors,
            labels_tensors,
            metadata
        )

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        return dataloader

    def encode(
            self,
            sequences: List[str],
            labels: List[str | int] | None = None,
            batch_size: int = 32,
            progress_bar: bool = True,
            add_special_tokens: bool = True
    ) -> DataLoader:
        """Original encode method for backward compatibility."""
        input_ids = []
        if progress_bar:
            sent_loader = tqdm(sequences, desc="Tokenizing")
        else:
            sent_loader = sequences

        for sent in sent_loader:
            encoded_sent = self.tokenizer.encode(
                sent,
                add_special_tokens=add_special_tokens
            )
            input_ids.append(encoded_sent)

        # Calculate max length (capped at 512)
        max_len = min(max([len(sen) for sen in input_ids]), 512)

        # Pad/truncate input tokens to max_len
        pad = np.full((len(input_ids), max_len), 0, dtype='long')
        for idx, s in enumerate(input_ids):
            trunc = s[:max_len]
            pad[idx, :len(trunc)] = trunc

        input_ids = pad

        # Create attention masks
        attention_masks = []
        if progress_bar:
            input_loader = tqdm(input_ids, desc="Creating attention masks")
        else:
            input_loader = input_ids

        for sent in input_loader:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        # If no labels, return DataLoader without labels
        if labels is None:
            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)

            data = TensorDataset(inputs_tensors, masks_tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
            return dataloader
        else:
            # Build a dictionary of labels if needed
            label_names = np.unique(labels)
            self.dict_labels = dict(zip(label_names, range(len(label_names))))

            if progress_bar:
                print(f"label ids: {self.dict_labels}")

            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)
            labels_tensors = torch.tensor([self.dict_labels[x] for x in labels])

            data = TensorDataset(inputs_tensors, masks_tensors, labels_tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
            return dataloader

    def run_training_enhanced(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            n_epochs: int = 3,
            lr: float = 5e-5,
            random_state: int = 42,
            save_model_as: str | None = None,
            pos_weight: torch.Tensor | None = None,
            metrics_output_dir: str = "./training_logs",
            track_languages: bool = True,
            **kwargs
    ):
        """
        Enhanced training with per-language performance tracking.

        Parameters
        ----------
        train_dataloader: DataLoader
            Training dataloader with optional metadata

        test_dataloader: DataLoader
            Test/validation dataloader with optional metadata

        track_languages: bool, default=True
            Whether to track per-language performance

        Other parameters same as run_training
        """
        # Ensure metric output directory exists
        os.makedirs(metrics_output_dir, exist_ok=True)

        training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics_enhanced.csv")
        language_metrics_json = os.path.join(metrics_output_dir, "language_performance.json")

        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()

        # Check if dataloaders have metadata
        has_metadata = isinstance(train_dataloader.dataset, MetadataDataset)

        # Initialize CSV for training metrics
        with open(training_metrics_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = [
                "epoch",
                "train_loss",
                "val_loss",
                "accuracy",
                "macro_f1"
            ]

            # Add per-class metrics
            if self.dict_labels:
                for label in sorted(self.dict_labels.values()):
                    headers.extend([
                        f"precision_{label}",
                        f"recall_{label}",
                        f"f1_{label}",
                        f"support_{label}"
                    ])

            writer.writerow(headers)

        # Collect test labels and metadata
        test_labels = []
        test_metadata = []

        for batch in test_dataloader:
            if has_metadata and len(batch) == 4:
                test_labels += batch[2].numpy().tolist()
                test_metadata.extend(batch[3])
            else:
                test_labels += batch[2].numpy().tolist()
                test_metadata.extend([{}] * len(batch[2]))

        num_labels = np.unique(test_labels).size

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
        best_scores = None

        language_performance_history = []

        # Training Loop
        for i_epoch in range(n_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(i_epoch + 1, n_epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0.0
            model.train()

            for step, train_batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

                b_inputs = train_batch[0].to(self.device)
                b_masks = train_batch[1].to(self.device)
                b_labels = train_batch[2].to(self.device)

                model.zero_grad()

                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                logits = outputs[0]

                # Weighted loss if specified
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

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

            # Validation
            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()

            total_val_loss = 0.0
            logits_complete = []
            epoch_metadata = []

            for test_batch in test_dataloader:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)

                if has_metadata and len(test_batch) == 4:
                    epoch_metadata.extend(test_batch[3])
                else:
                    epoch_metadata.extend([{}] * len(b_labels))

                with torch.no_grad():
                    outputs = model(b_inputs, token_type_ids=None,
                                  attention_mask=b_masks, labels=b_labels)

                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                logits_complete.append(logits.detach().cpu().numpy())

            logits_complete = np.concatenate(logits_complete, axis=0)
            avg_val_loss = total_val_loss / len(test_dataloader)
            val_loss_values.append(avg_val_loss)

            print("")
            print("  Average validation loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

            # Predictions and metrics
            preds = np.argmax(logits_complete, axis=1).flatten()

            # Track performance with metadata
            if track_languages and has_metadata:
                self.performance_tracker.add_batch(preds, np.array(test_labels), epoch_metadata)
                perf_metrics = self.performance_tracker.calculate_metrics()

                # Save language performance for this epoch
                language_performance_history.append({
                    'epoch': i_epoch + 1,
                    'metrics': perf_metrics
                })

                # Print language-specific performance
                if perf_metrics['per_language']:
                    print("\nPer-Language Performance:")
                    print(f"{'Language':<10} {'Accuracy':<10} {'Macro F1':<10}")
                    print("-"*30)
                    for lang, metrics in sorted(perf_metrics['per_language'].items()):
                        print(f"{lang:<10} {metrics['accuracy']:<10.3f} "
                              f"{metrics['macro_f1']:<10.3f}")

            # Overall metrics
            precision, recall, fscore, support = precision_recall_fscore_support(
                test_labels, preds, average=None, zero_division=0
            )
            macro_f1 = precision_recall_fscore_support(
                test_labels, preds, average='macro', zero_division=0
            )[2]

            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(test_labels, preds)

            print(f"\nOverall Accuracy: {accuracy:.3f}")
            print(f"Overall Macro F1: {macro_f1:.3f}")

            # Write to CSV
            with open(training_metrics_csv, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    i_epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    accuracy,
                    macro_f1
                ]

                # Add per-class metrics
                for i in range(len(precision)):
                    row.extend([precision[i], recall[i], fscore[i], support[i]])

                writer.writerow(row)

            # Check for best model
            combined_metric = 0.7 * (fscore[1] if len(fscore) > 1 else fscore[0]) + 0.3 * macro_f1

            if combined_metric > best_metric_val:
                print(f"New best model found at epoch {i_epoch + 1} with combined metric={combined_metric:.4f}.")

                if best_model_path is not None:
                    try:
                        shutil.rmtree(best_model_path)
                    except OSError:
                        pass

                best_metric_val = combined_metric
                best_scores = (precision, recall, fscore, support)

                if save_model_as is not None:
                    best_model_path = f"./models/{save_model_as}_epoch_{i_epoch+1}"
                    os.makedirs(best_model_path, exist_ok=True)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)

        # Save language performance history
        if track_languages and language_performance_history:
            with open(language_metrics_json, 'w', encoding='utf-8') as f:
                json.dump(language_performance_history, f, indent=2, ensure_ascii=False)
            print(f"\nLanguage performance saved to {language_metrics_json}")

        # Move best model to final location
        if save_model_as is not None and best_model_path is not None:
            final_model_path = f"./models/{save_model_as}"
            if os.path.exists(final_model_path):
                shutil.rmtree(final_model_path)
            shutil.move(best_model_path, final_model_path)
            print(f"Best model saved to {final_model_path}")

        return best_scores

    def format_time(self, elapsed):
        """Format time in seconds to hh:mm:ss format."""
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Keep original methods for backward compatibility
    def run_training(self, *args, **kwargs):
        """Original training method for backward compatibility."""
        # Redirect to parent class implementation or implement original logic
        return super().run_training(*args, **kwargs) if hasattr(super(), 'run_training') else None

    def predict(self, *args, **kwargs):
        """Original predict method."""
        return super().predict(*args, **kwargs) if hasattr(super(), 'predict') else None

    def predict_with_model(self, *args, **kwargs):
        """Original predict_with_model method."""
        return super().predict_with_model(*args, **kwargs) if hasattr(super(), 'predict_with_model') else None

    def load_model(self, *args, **kwargs):
        """Original load_model method."""
        return super().load_model(*args, **kwargs) if hasattr(super(), 'load_model') else None
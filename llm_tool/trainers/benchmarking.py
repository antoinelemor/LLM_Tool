#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
benchmarking.py

MAIN OBJECTIVE:
---------------
Benchmark multiple transformer models on labelled datasets with automatic
language handling, dataset preparation, and metrics aggregation.

Dependencies:
-------------
- csv
- json
- os
- random
- shutil
- dataclasses
- pathlib
- typing
- numpy
- torch
- llm_tool.utils.logging_utils
- llm_tool.trainers.bert_base
- llm_tool.trainers.models
- llm_tool.trainers.benchmark_dataset_builder
- llm_tool.trainers.model_selector
- llm_tool.trainers.multilingual_selector
- llm_tool.trainers.sota_models
- llm_tool.trainers.data_utils
- llm_tool.utils.training_paths

MAIN FEATURES:
--------------
1) Configure benchmark runs with per-category, per-language training loops
2) Auto-build balanced datasets and manage reinforcement rescue strategies
3) Select candidate models based on language coverage and resource profiles
4) Persist metrics, logs, and best-model snapshots for each experiment
5) Aggregate results into CSV/JSON outputs for reporting and analysis

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import ast
import csv
import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from llm_tool.utils.logging_utils import get_logger
from llm_tool.trainers.bert_base import BertBase
from llm_tool.trainers.models import (
    Bert, Camembert, GermanBert, SpanishBert, ItalianBert,
    PortugueseBert, ChineseBert, ArabicBert, RussianBert, HindiBert
)
from llm_tool.trainers.benchmark_dataset_builder import BenchmarkDatasetBuilder, BenchmarkDataset
from llm_tool.trainers.model_selector import ModelSelector, TaskComplexity, ResourceProfile
from llm_tool.trainers.multilingual_selector import MultilingualModelSelector
from llm_tool.trainers.sota_models import (
    MDeBERTaV3Base, XLMRobertaBase, XLMRobertaLarge,
    DeBERTaV3Base, DeBERTaV3Large, DeBERTaV3XSmall,
    RoBERTaBase, RoBERTaLarge, DistilRoBERTa,
    ELECTRABase, ELECTRALarge, ELECTRASmall,
    ALBERTBase, ALBERTLarge
)
from llm_tool.trainers.data_utils import safe_tolist, safe_convert_labels
from llm_tool.utils.training_paths import resolve_metrics_base_dir


REQUIRED_COLS = [
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
    "macro_f1",
    "saved_model_path",
    "training_phase",
]


@dataclass
class BenchmarkConfig:
    """Hyperparameters and behavioural flags for dataset benchmarking."""

    epochs: int = 20
    learning_rate: float = 5e-5
    batch_size: int = 32
    random_state: int = 42
    reinforced_learning: bool = True
    reinforced_epochs: int = 20
    best_model_criteria: str = "combined"
    f1_class_1_weight: float = 0.7
    rescue_low_class1_f1: bool = True  # CRITICAL: Enable automatic rescue for low F1
    f1_rescue_threshold: float = 0.50  # Trigger rescue if F1 classe 1 < 50%
    track_languages: bool = True
    # Dataset building options
    auto_build_datasets: bool = True
    balance_benchmark_classes: bool = False
    balance_method: str = "undersample"
    test_split_size: float = 0.2
    min_samples_per_class: int = 10
    save_benchmark_csv: bool = True
    # Benchmark-specific reinforced learning
    use_reinforced_in_benchmark: bool = False  # Allow reinforced learning during benchmark
    reinforced_f1_threshold: float = 0.60  # Trigger reinforced if F1_1 < this value
    # Short sequence optimization for large models
    optimize_for_short_sequences: bool = False  # Adjust params for short texts
    short_sequence_threshold: int = 100  # Consider sequences < this as short
    large_model_adjustments: bool = True  # Apply special settings for large models
    backbone_map: Dict[str, str] = field(default_factory=lambda: {
        "FR": "camembert-base",
        "EN": "bert-base-cased",
    })


@dataclass
class TrainingRunSummary:
    """Structured result for one (category, language) training run."""

    dataset: Optional[str]
    category: str
    language: str
    model_path: Optional[Path]
    log_dir: Path
    summary_csv: Path
    metrics: Dict[str, object]


class BenchmarkRunner:
    """Replicates the vitrine pipeline training flow inside the package."""

    def __init__(
        self,
        data_root: Path | str = Path("data/processed"),
        models_root: Path | str = Path("models"),
        backbone_dir: Path | str = Path("backbones"),
        config: Optional[BenchmarkConfig] = None,
        logger_name: str = "LLMTool.BenchmarkRunner",
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.data_root = Path(data_root)
        self.models_root = Path(models_root)
        self.backbone_dir = Path(backbone_dir)
        self.logs_subdir = "logs"
        self.logger = get_logger(logger_name)
        self.results: List[TrainingRunSummary] = []

        self._model_cache: Dict[str, BertBase] = {}

        # Session ID for unified logging across all training runs in this session
        self.session_id: Optional[str] = None

        # Initialize dataset builder if auto-building is enabled
        self.dataset_builder = None
        if self.config.auto_build_datasets:
            benchmark_csv_path = self.models_root / "benchmark_datasets.csv" if self.config.save_benchmark_csv else None
            self.dataset_builder = BenchmarkDatasetBuilder(
                logger_name=f"{logger_name}.DatasetBuilder",
                random_state=self.config.random_state,
                test_size=self.config.test_split_size,
                stratify=True,
                balance_classes=self.config.balance_benchmark_classes,
                balance_method=self.config.balance_method,
                min_samples_per_class=self.config.min_samples_per_class,
                save_splits=False,  # We'll handle saving differently
                csv_log_path=benchmark_csv_path
            )

        self.models_root.mkdir(parents=True, exist_ok=True)
        self.backbone_dir.mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    @staticmethod
    def _normalize_category_name(name: str) -> str:
        """Normalize raw category names coming from filesystem folders."""
        cleaned = (name or "").strip()
        if not cleaned:
            return "category"

        cleaned = cleaned.replace("\n", " ").replace("\r", " ")

        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                parsed = None
            else:
                if isinstance(parsed, (list, tuple)) and parsed:
                    cleaned = str(parsed[0])
                elif isinstance(parsed, str):
                    cleaned = parsed

        cleaned = cleaned.strip("[]'\" ")
        cleaned = re.sub(r"\s+", "_", cleaned)

        return cleaned or "category"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_backbones_cached(self, offline: bool = True) -> None:
        """Mirror the script pre-download step so future runs are offline."""
        os.environ["HF_HOME"] = str(self.backbone_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.backbone_dir)

        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError:
            self.logger.warning(
                "huggingface_hub is not installed; backbone caching skipped."
            )
            return

        for repo in self.config.backbone_map.values():
            try:
                self.logger.info("Caching backbone %s", repo)
                snapshot_download(
                    repo_id=repo,
                    cache_dir=self.backbone_dir,
                    resume_download=True,
                    local_files_only=False,
                )
            except Exception as exc:  # pragma: no cover - depends on network availability
                self.logger.warning("Could not cache %s: %s", repo, exc)

        if offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def run(
        self,
        multiple_datasets: bool = False,
        dataset_filter: Optional[Sequence[str]] = None,
    ) -> List[TrainingRunSummary]:
        """Launch benchmarking on one or many datasets."""
        self.results.clear()

        # Create unified session ID for all training runs in this benchmark session
        # CRITICAL FIX: Only create new session_id if one doesn't already exist
        # This ensures benchmark run from Training Arena reuses the existing session
        import datetime
        if self.session_id is None:
            self.session_id = datetime.datetime.now().strftime("training_session_%Y%m%d_%H%M%S")
        self.logger.info("Starting benchmark session: %s", self.session_id)

        if multiple_datasets:
            training_dirs = sorted(
                p for p in self.data_root.iterdir()
                if p.is_dir() and p.name.startswith("training_data")
            )
            if dataset_filter:
                dataset_filter_set = {name.lower() for name in dataset_filter}
                training_dirs = [
                    p for p in training_dirs
                    if p.name.lower() in dataset_filter_set
                ]

            if not training_dirs:
                self.logger.warning("No matching training_data directories found under %s", self.data_root)
                return self.results

            self.logger.info("Found %d dataset(s) to benchmark", len(training_dirs))
            for training_dir in training_dirs:
                dataset_name = training_dir.name.replace("training_data_", "")
                if dataset_name == "training_data":
                    dataset_name = "default"
                self._process_dataset(training_dir, dataset_name)
        else:
            training_root = self.data_root / "training_data"
            if not training_root.exists():
                self.logger.error("Dataset folder %s is missing", training_root)
                return self.results
            self._process_dataset(training_root, None)

        return self.results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_dataset(self, training_root: Path, dataset_name: Optional[str]) -> None:
        logs_root = training_root / self.logs_subdir
        summary_csv = training_root / "models_summary.csv"

        models_output_dir = self.models_root / dataset_name if dataset_name else self.models_root
        models_output_dir.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)

        human_name = dataset_name or "principal"
        self.logger.info("\n%s", "=" * 80)
        self.logger.info("Benchmarking dataset: %s", human_name)
        self.logger.info("Data directory: %s", training_root)
        self.logger.info("Models target: %s", models_output_dir)
        self.logger.info("%s", "=" * 80)

        # Calculate total models for global progress tracking
        total_models = 0
        for category_dir in sorted(p for p in training_root.iterdir() if p.is_dir()):
            if category_dir.name == self.logs_subdir:
                continue
            for lang_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                # Check if there's data to train
                train_files = list((lang_dir / "train").glob("*_train.jsonl"))
                test_files = list((lang_dir / "test").glob("*_test.jsonl"))
                if (train_files and test_files) or self.config.auto_build_datasets:
                    total_models += 1

        # Initialize global tracking parameters
        import time
        global_start_time = time.time()
        global_total_epochs = total_models * self.config.epochs

        # Calculate maximum possible epochs (if all models trigger reinforced learning)
        if self.config.reinforced_learning and self.config.reinforced_epochs:
            global_max_epochs = total_models * (self.config.epochs + self.config.reinforced_epochs)
        else:
            global_max_epochs = global_total_epochs

        global_completed_epochs = 0
        current_model = 0

        for category_dir in sorted(p for p in training_root.iterdir() if p.is_dir()):
            if category_dir.name == self.logs_subdir:
                continue

            for lang_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                lang = lang_dir.name.upper()

                # Pre-check if we should train this model
                train_files = list((lang_dir / "train").glob("*_train.jsonl"))
                test_files = list((lang_dir / "test").glob("*_test.jsonl"))
                should_train = (train_files and test_files) or self.config.auto_build_datasets

                if not should_train:
                    continue  # Skip this model, don't increment counter

                current_model += 1
                epochs_completed = self._train_category_language(
                    category_dir,
                    lang_dir,
                    lang,
                    dataset_name,
                    logs_root,
                    summary_csv,
                    models_output_dir,
                    global_total_models=total_models,
                    global_current_model=current_model,
                    global_total_epochs=global_total_epochs,
                    global_max_epochs=global_max_epochs,
                    global_completed_epochs=global_completed_epochs,
                    global_start_time=global_start_time,
                )
                # Update completed epochs for next model
                if epochs_completed is not None:
                    global_completed_epochs = epochs_completed

    def _train_category_language(
        self,
        category_dir: Path,
        lang_dir: Path,
        language: str,
        dataset_name: Optional[str],
        logs_root: Path,
        summary_csv: Path,
        models_output_dir: Path,
        global_total_models: Optional[int] = None,
        global_current_model: Optional[int] = None,
        global_total_epochs: Optional[int] = None,
        global_max_epochs: Optional[int] = None,
        global_completed_epochs: Optional[int] = None,
        global_start_time: Optional[float] = None,
    ) -> Optional[int]:
        raw_cat_name = category_dir.name
        cat_name = self._normalize_category_name(raw_cat_name)
        if cat_name != raw_cat_name:
            self.logger.debug("Sanitized category name '%s' -> '%s'", raw_cat_name, cat_name)

        train_files = list((lang_dir / "train").glob("*_train.jsonl"))
        test_files = list((lang_dir / "test").glob("*_test.jsonl"))

        # Try to use existing train/test splits
        if len(train_files) == 1 and len(test_files) == 1:
            train_texts, train_labels = self._read_jsonl(train_files[0])
            test_texts, test_labels = self._read_jsonl(test_files[0])
        # Otherwise, try to build automatically from available data
        elif self.config.auto_build_datasets and self.dataset_builder:
            self.logger.info("No train/test splits found for %s/%s, attempting auto-build...", cat_name, language)

            # Look for any JSONL files in the directory
            all_jsonl_files = list(lang_dir.glob("*.jsonl"))
            if not all_jsonl_files:
                all_jsonl_files = list(lang_dir.parent.glob("*.jsonl"))

            if not all_jsonl_files:
                self.logger.warning("No JSONL files found for %s/%s", cat_name, language)
                return

            # Try to build from the first available file
            dataset = self.dataset_builder.build_from_jsonl(
                data_path=all_jsonl_files[0],
                category=cat_name,
                max_samples=None  # Use all available data
            )

            if not dataset:
                self.logger.warning("Failed to auto-build dataset for %s/%s", cat_name, language)
                return

            train_texts = dataset.train_texts
            train_labels = dataset.train_labels
            test_texts = dataset.test_texts
            test_labels = dataset.test_labels
            # Get per-sample language info if available
            test_languages = dataset.test_languages if hasattr(dataset, 'test_languages') else None

            # Log the dataset statistics
            self.logger.info(dataset.get_summary())
        else:
            self.logger.warning("Skipping %s/%s – no train/test files and auto-build disabled", cat_name, language)
            return

        if not train_texts or not test_texts:
            self.logger.warning("Skipping %s/%s – empty split", cat_name, language)
            return

        # CRITICAL: Do NOT use pos_weight for initial training
        # Let the model learn naturally first, then reinforcement learning
        # will apply adaptive weights if needed (when F1 is low)
        pos_weight = None
        model = self._get_model_for_language(language)

        # ==================== LANGUAGE FILTERING FOR MONOLINGUAL MODELS ====================
        # Import language filtering utilities
        from .model_trainer import get_model_target_languages, filter_data_by_language
        import pandas as pd

        # Get model name (e.g., 'camembert-base', 'xlm-roberta-base')
        model_name = model.model_name if hasattr(model, 'model_name') else model.__class__.__name__
        target_languages = get_model_target_languages(model_name)

        if target_languages is not None and test_languages is not None:
            # Language-specific model - filter data to target language only
            self.logger.info(f"🌍 Model '{model_name}' targets {target_languages}. Filtering data...")

            # Create temporary dataframes for filtering
            train_df = pd.DataFrame({'text': train_texts, 'label': train_labels, 'language': test_languages[:len(train_texts)]})
            test_df = pd.DataFrame({'text': test_texts, 'label': test_labels, 'language': test_languages})

            # Filter
            train_df_filtered = filter_data_by_language(train_df, target_languages, 'language')
            test_df_filtered = filter_data_by_language(test_df, target_languages, 'language')

            # Update lists - CRITICAL: Use safe_tolist to convert numpy types to Python native types
            train_texts = train_df_filtered['text'].tolist()
            train_labels = safe_tolist(train_df_filtered['label'], column_name='label')
            test_texts = test_df_filtered['text'].tolist()
            test_labels = safe_tolist(test_df_filtered['label'], column_name='label')

            # Update language info - CRITICAL FIX: only use test set languages for validation
            if test_languages:
                test_languages = safe_tolist(test_df_filtered['language'], column_name='language')

            self.logger.info(f"✓ Filtered to {len(train_texts)} train + {len(test_texts)} test samples")

        train_loader = model.encode(
            train_texts,
            train_labels,
            batch_size=self.config.batch_size,
            progress_bar=True,
            shuffle=True  # Shuffle training data
        )
        test_loader = model.encode(
            test_texts,
            test_labels,
            batch_size=self.config.batch_size,
            progress_bar=True,
            shuffle=False  # Don't shuffle test data
        )

        # UNIFIED: Use centralized function to set detected languages (SAME AS BENCHMARK)
        if test_languages:
            from .model_trainer import set_detected_languages_on_model
            # Combine train and test languages (train uses first len(train) from test_languages)
            train_langs = test_languages[:len(train_texts)] if len(test_languages) >= len(train_texts) else []
            set_detected_languages_on_model(
                model=model,
                train_languages=train_langs,
                val_languages=test_languages,
                logger=self.logger
            )

        log_dir = logs_root / cat_name / language
        log_dir.mkdir(parents=True, exist_ok=True)

        dataset_suffix = f"[{dataset_name}]" if dataset_name else ""
        model_name = f"{cat_name}_{language}"
        self.logger.info(
            "\n🔁 Training %s %s on %d train / %d validation samples",
            model_name,
            dataset_suffix,
            len(train_texts),
            len(test_texts),
        )

        temp_model_name = f"temp_{model_name}" if dataset_name is None else f"temp_{dataset_name}_{model_name}"

        # Log dataset composition before training
        self._log_dataset_composition(cat_name, language, train_labels, test_labels, log_dir)

        metrics_output_dir = str(resolve_metrics_base_dir(None))
        summary = model.run_training(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            n_epochs=self.config.epochs,
            lr=self.config.learning_rate,
            random_state=self.config.random_state,
            save_model_as=temp_model_name,
            pos_weight=pos_weight,
            metrics_output_dir=metrics_output_dir,
            best_model_criteria=self.config.best_model_criteria,
            f1_class_1_weight=self.config.f1_class_1_weight,
            reinforced_learning=self.config.reinforced_learning,
            reinforced_epochs=self.config.reinforced_epochs,
            rescue_low_class1_f1=self.config.rescue_low_class1_f1,
            f1_1_rescue_threshold=self.config.f1_rescue_threshold,
            reinforced_f1_threshold=self.config.reinforced_f1_threshold,
            track_languages=self.config.track_languages,
            language_info=test_languages,
            model_identifier=f"{cat_name}_{language}_{model.__class__.__name__}" if self.config.track_languages and test_languages else None,
            label_key=None,  # Benchmark uses label_value directly
            label_value=cat_name,  # CRITICAL: Category name for organized logs (e.g., "Health")
            language=language,  # CRITICAL: Language for metadata (e.g., "EN", "FR")
            global_total_models=global_total_models,
            global_current_model=global_current_model,
            global_total_epochs=global_total_epochs,
            global_max_epochs=global_max_epochs,
            global_completed_epochs=global_completed_epochs,
            global_start_time=global_start_time,
            session_id=self.session_id  # CRITICAL: Unified session ID for all runs
        )

        best_path = model.last_saved_model_path
        if best_path:
            best_path = Path(best_path)

        if best_path and best_path.exists():
            final_path = models_output_dir / model_name
            if final_path.exists():
                shutil.rmtree(final_path)
            shutil.move(str(best_path), final_path)
            best_path = final_path
            self.logger.info("✓ Model saved at %s", final_path)
        else:
            self.logger.warning("Training completed but no saved model found for %s", model_name)

        self._record_best_model(log_dir, model_name, summary_csv, dataset_name, best_path)

        self.results.append(
            TrainingRunSummary(
                dataset=dataset_name,
                category=cat_name,
                language=language,
                model_path=best_path,
                log_dir=log_dir,
                summary_csv=summary_csv,
                metrics=summary if isinstance(summary, dict) else {},
            )
        )

        # Return updated global_completed_epochs for the next model
        if global_total_models is not None and hasattr(model, 'global_completed_epochs'):
            return model.global_completed_epochs
        return None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    def _get_model_for_language(self, language: str) -> Bert:
        lang = language.upper()
        if lang not in self._model_cache:
            if lang == "FR":
                self._model_cache[lang] = Camembert()
            else:
                self._model_cache[lang] = Bert()
        return self._model_cache[lang]

    def _compute_pos_weight(self, labels: Sequence[int]) -> Optional[torch.Tensor]:
        positives = sum(labels)
        negatives = len(labels) - positives
        if positives == 0 or negatives == 0:
            return None
        return torch.tensor(negatives / positives, dtype=torch.float)

    def _record_best_model(
        self,
        log_dir: Path,
        model_name: str,
        summary_csv: Path,
        dataset_name: Optional[str],
        best_model_path: Optional[Path],
    ) -> None:
        best_file = log_dir / "best_models.csv"
        if not best_file.exists():
            self.logger.warning("best_models.csv missing in %s", log_dir)
            return

        with best_file.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        if not rows:
            self.logger.warning("best_models.csv empty for %s", model_name)
            return

        metric_key = next(
            (k for k in ("macro_f1", "f1_1", "f1", "f1_score") if k in rows[0]),
            None,
        )
        best_row = max(rows, key=lambda r: float(r.get(metric_key, 0))) if metric_key else rows[-1]

        record: Dict[str, str] = {"modele": model_name}
        if dataset_name:
            record["dataset"] = dataset_name
        for col in REQUIRED_COLS:
            record[col] = best_row.get(col, "")

        if best_model_path is not None:
            record["saved_model_path"] = str(best_model_path)

        existing: List[Dict[str, str]] = []
        if summary_csv.exists():
            with summary_csv.open(encoding="utf-8") as fh:
                existing = list(csv.DictReader(fh))

        def _match(rec: Dict[str, str]) -> bool:
            same_model = rec.get("modele") == model_name
            same_dataset = rec.get("dataset") == dataset_name if dataset_name else True
            return same_model and same_dataset

        existing = [rec for rec in existing if not _match(rec)]
        existing.append(record)

        fieldnames = ["modele"]
        if dataset_name:
            fieldnames.append("dataset")
        fieldnames.extend(REQUIRED_COLS)

        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(existing, key=lambda r: (r.get("dataset", ""), r["modele"])))

        self.logger.info("Summary updated at %s", summary_csv)

    def _log_dataset_composition(
        self,
        category: str,
        language: str,
        train_labels: Sequence[int],
        test_labels: Sequence[int],
        log_dir: Path
    ) -> None:
        """Log detailed dataset composition to CSV."""
        from collections import Counter

        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)

        # Calculate balance metrics
        train_balance = min(train_counts.values()) / max(train_counts.values()) if train_counts else 0
        test_balance = min(test_counts.values()) / max(test_counts.values()) if test_counts else 0

        # Log to console with language info
        self.logger.info(
            "📊 Dataset composition for %s/%s:\n"
            "   Train: %d samples (Class 0: %d, Class 1: %d, Balance: %.2f%%)\n"
            "   Test: %d samples (Class 0: %d, Class 1: %d, Balance: %.2f%%)",
            category, language,
            len(train_labels), train_counts[0], train_counts[1], train_balance * 100,
            len(test_labels), test_counts[0], test_counts[1], test_balance * 100
        )

        # Save to CSV
        composition_csv = log_dir / "dataset_composition.csv"
        record = {
            'category': category,
            'language': language,
            'train_total': len(train_labels),
            'train_class_0': train_counts[0],
            'train_class_1': train_counts[1],
            'train_balance': f"{train_balance:.2%}",
            'test_total': len(test_labels),
            'test_class_0': test_counts[0],
            'test_class_1': test_counts[1],
            'test_balance': f"{test_balance:.2%}"
        }

        file_exists = composition_csv.exists()
        with open(composition_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    @staticmethod
    def _read_jsonl(path: Path, reference_category: Optional[str] = None) -> Tuple[List[str], List[int], List[str]]:
        """Read JSONL file and return texts, labels, and languages.

        Supports both binary (label) and multi-label (labels) formats.
        For multi-label, uses reference_category or most common category.
        """
        texts: List[str] = []
        labels: List[int] = []
        languages: List[str] = []

        # First pass to detect format and find most common category if needed
        is_multilabel = False
        category_counts = {}

        with path.open(encoding="utf-8") as fh:
            first_line = fh.readline()
            if first_line:
                first_record = json.loads(first_line)
                is_multilabel = "labels" in first_record

                # If multi-label and no reference category, count categories
                if is_multilabel and not reference_category:
                    fh.seek(0)  # Reset to beginning
                    for line in fh:
                        record = json.loads(line)
                        if "labels" in record:
                            for cat, val in record["labels"].items():
                                if val:
                                    category_counts[cat] = category_counts.get(cat, 0) + 1

                    # Use most common category
                    if category_counts:
                        reference_category = max(category_counts, key=category_counts.get)
                        print(f"\n🏷️  Multi-label dataset detected. Using '{reference_category}' for benchmark")
                        print(f"   (Most common category with {category_counts[reference_category]} positive samples)")

        # Second pass to extract data
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)

                # Extract text
                texts.append(record.get("text", ""))

                # Extract label
                if "label" in record:
                    # Binary format
                    labels.append(int(record["label"]))
                elif "labels" in record and reference_category:
                    # Multi-label format - convert to binary for benchmark
                    label_value = record["labels"].get(reference_category, 0)
                    labels.append(int(label_value))
                else:
                    # Default to 0 if no label found
                    labels.append(0)

                # Extract language if available
                if "lang" in record:
                    languages.append(record["lang"])
                elif "language" in record:
                    languages.append(record["language"])

        return texts, labels, languages

    def run_comprehensive_benchmark(
        self,
        data_path: Path,
        benchmark_epochs: int = 3,
        test_all_models: bool = True,
        allow_user_selection: bool = False,
        models_to_test: Optional[List[str]] = None,
        verbose: bool = True,
        save_detailed_log: bool = True,
        save_best_models_log: bool = True
    ) -> Optional[str]:
        """
        Run comprehensive benchmark testing multiple models.

        Args:
            data_path: Path to data file (JSONL)
            benchmark_epochs: Number of epochs for benchmark
            test_all_models: Test all available models
            allow_user_selection: Let user select model after benchmark
            models_to_test: Specific list of models to test
            verbose: Print detailed results

        Returns:
            Name of best/selected model or None
        """
        # Create unified session ID for all training runs in this comprehensive benchmark session
        # CRITICAL FIX: Only create new session_id if one doesn't already exist
        # This ensures benchmark run from Training Arena reuses the existing session
        import datetime
        if self.session_id is None:
            self.session_id = datetime.datetime.now().strftime("training_session_%Y%m%d_%H%M%S")
        self.logger.info("Starting comprehensive benchmark session: %s", self.session_id)
        self.logger.info("Data path: %s", data_path)

        # Load and analyze data (auto-detect format)
        print("\n🔍 Analyzing dataset format...")
        texts, labels, languages = self._read_jsonl(data_path)

        if not texts:
            self.logger.error("No data found in %s", data_path)
            return None

        # Detect languages
        from collections import Counter
        detected_languages = []
        if languages:
            lang_counts = Counter(languages)
            detected_languages = list(lang_counts.keys())
            is_multilingual = len(detected_languages) > 1
        else:
            # Try to detect from text
            is_multilingual = False
            detected_languages = ["EN"]  # Default

        if verbose:
            print(f"\n📊 Dataset Analysis:")
            print(f"   - Total samples: {len(texts)}")
            print(f"   - Languages: {', '.join(detected_languages)}")
            print(f"   - Multilingual: {'Yes' if is_multilingual else 'No'}")

            label_dist = dict(Counter(labels))
            print(f"   - Class distribution: Class 0: {label_dist.get(0, 0)}, Class 1: {label_dist.get(1, 0)}")
            print(f"   - Positive ratio: {label_dist.get(1, 0) / len(labels) * 100:.1f}%" if labels else "N/A")

        # Determine models to test
        if models_to_test is None:
            models_to_test = self._select_models_for_benchmark(
                is_multilingual,
                detected_languages,
                test_all_models
            )
        elif isinstance(models_to_test, list) and models_to_test and isinstance(models_to_test[0], str):
            # Convert list of model names to list of (model_name, languages)
            # Each model will be tested on its supported languages that match the data
            selector = ModelSelector(verbose=False)
            converted_models = []

            for model_name in models_to_test:
                if model_name in selector.MODEL_PROFILES:
                    profile = selector.MODEL_PROFILES[model_name]
                    supported_langs = profile.supported_languages

                    if '*' in supported_langs:
                        # Multilingual model - test on all detected languages
                        converted_models.append((model_name, detected_languages))
                    else:
                        # Language-specific model - only test on matching languages
                        matching_languages = []
                        for lang in detected_languages:
                            lang_norm = lang.lower()
                            if lang_norm in supported_langs or lang_norm[:2] in supported_langs:
                                matching_languages.append(lang)

                        if matching_languages:
                            converted_models.append((model_name, matching_languages))
                        else:
                            self.logger.warning(f"Model {model_name} doesn't support any of the detected languages: {detected_languages}")
                else:
                    self.logger.warning(f"Model {model_name} not found in MODEL_PROFILES")

            models_to_test = converted_models

        # Prepare benchmark dataset
        dataset_builder = BenchmarkDatasetBuilder(
            random_state=self.config.random_state,
            test_size=self.config.test_split_size,
            stratify=True,
            balance_classes=self.config.balance_benchmark_classes,
            balance_method=self.config.balance_method,
            min_samples_per_class=self.config.min_samples_per_class,
            save_splits=False,
            csv_log_path=self.models_root / "benchmark_dataset_stats.csv" if self.config.save_benchmark_csv else None
        )

        # Build dataset with language info
        dataset = dataset_builder._create_splits(
            texts=texts,
            labels=labels,
            languages=languages if languages else None,
            primary_language=detected_languages[0] if detected_languages else None,
            category="benchmark"
        )

        if not dataset:
            self.logger.error("Failed to create benchmark dataset")
            return None

        if verbose:
            print(dataset.get_summary())

        # Initialize global progress tracking
        total_models = len(models_to_test)
        global_start_time = time.time()

        # Calculate total expected epochs (base scenario without reinforced learning)
        global_total_epochs = total_models * benchmark_epochs

        # Calculate maximum possible epochs (if all models trigger reinforced learning)
        if self.config.reinforced_learning and self.config.reinforced_epochs:
            global_max_epochs = total_models * (benchmark_epochs + self.config.reinforced_epochs)
        else:
            global_max_epochs = global_total_epochs

        # Run benchmark on all models
        results = self._benchmark_models_with_languages(
            dataset=dataset,
            models_to_test=models_to_test,
            detected_languages=detected_languages,
            benchmark_epochs=benchmark_epochs,
            verbose=verbose,
            global_total_models=total_models,
            global_total_epochs=global_total_epochs,
            global_max_epochs=global_max_epochs,
            global_start_time=global_start_time
        )

        if not results:
            self.logger.error("No benchmark results")
            return None

        # Sort results by F1 score
        results.sort(key=lambda x: x.get('f1_score', 0), reverse=True)

        # Display comprehensive results
        if verbose:
            self._display_benchmark_results(
                results=results,
                detected_languages=detected_languages,
                is_multilingual=is_multilingual
            )

        # Save detailed logs if requested
        if save_detailed_log or save_best_models_log:
            self._save_benchmark_logs(
                results=results,
                detected_languages=detected_languages,
                is_multilingual=is_multilingual,
                data_path=data_path,
                benchmark_epochs=benchmark_epochs,
                save_detailed=save_detailed_log,
                save_best=save_best_models_log
            )

        # Let user select or auto-select best
        if allow_user_selection:
            return self._user_select_model(results)
        else:
            # Select best model considering all languages if multilingual
            if is_multilingual and results:
                best_model = self._select_best_multilingual_model(results, detected_languages)
                if verbose:
                    print(f"\n✅ Best model selected (considering all languages): {best_model}")
                return best_model
            else:
                return results[0]['model_name'] if results else None

    def _select_models_for_benchmark(
        self,
        is_multilingual: bool,
        detected_languages: List[str],
        test_all: bool = True
    ) -> List[Tuple[str, List[str]]]:
        """
        Select models to benchmark based on their supported languages.
        Returns list of (model_name, [languages_to_test])
        """
        models_with_languages = []

        # Get model profiles from ModelSelector
        selector = ModelSelector(verbose=False)
        model_profiles = selector.MODEL_PROFILES

        # Normalize language codes for comparison
        detected_langs_normalized = []
        for lang in detected_languages:
            # Convert common variations to standard codes
            lang_upper = lang.upper()
            if lang_upper in ['EN', 'ENGLISH']:
                detected_langs_normalized.append('en')
            elif lang_upper in ['FR', 'FRENCH', 'FRANÇAIS']:
                detected_langs_normalized.append('fr')
            elif lang_upper in ['DE', 'GERMAN', 'DEUTSCH']:
                detected_langs_normalized.append('de')
            elif lang_upper in ['ES', 'SPANISH', 'ESPAÑOL']:
                detected_langs_normalized.append('es')
            elif lang_upper in ['IT', 'ITALIAN', 'ITALIANO']:
                detected_langs_normalized.append('it')
            elif lang_upper in ['PT', 'PORTUGUESE', 'PORTUGUÊS']:
                detected_langs_normalized.append('pt')
            elif lang_upper in ['ZH', 'CHINESE', '中文']:
                detected_langs_normalized.append('zh')
            elif lang_upper in ['AR', 'ARABIC', 'العربية']:
                detected_langs_normalized.append('ar')
            elif lang_upper in ['RU', 'RUSSIAN', 'РУССКИЙ']:
                detected_langs_normalized.append('ru')
            elif lang_upper in ['HI', 'HINDI', 'हिन्दी']:
                detected_langs_normalized.append('hi')
            else:
                detected_langs_normalized.append(lang.lower())

        # Select models based on language support
        french_models_selected = []
        for model_name, profile in model_profiles.items():
            supported_langs = profile.supported_languages

            if '*' in supported_langs:
                # Multilingual model - test on all detected languages
                models_with_languages.append((model_name, detected_languages))
            else:
                # Language-specific model - only test if language matches
                matching_languages = []
                for i, norm_lang in enumerate(detected_langs_normalized):
                    if norm_lang in supported_langs:
                        matching_languages.append(detected_languages[i])
                        # Track French models specifically
                        if norm_lang == 'fr' and model_name not in french_models_selected:
                            french_models_selected.append(model_name)

                if matching_languages:
                    models_with_languages.append((model_name, matching_languages))

        # Log French models specifically if French was detected
        if 'fr' in detected_langs_normalized and french_models_selected:
            self.logger.info(f"🇫🇷 French-specific models selected: {', '.join(sorted(french_models_selected))}")

        # If not testing all models, select a representative subset
        if not test_all:
            if len(models_with_languages) > 10:
                # Prioritize: multilingual, language-specific, then general models
                priority_models = []

                # Add best multilingual models
                for model_name, langs in models_with_languages:
                    if model_name in ['MDeBERTaV3Base', 'XLMRobertaBase']:
                        priority_models.append((model_name, langs))

                # Add language-specific models
                for lang in detected_langs_normalized:
                    for model_name, langs in models_with_languages:
                        profile = model_profiles.get(model_name)
                        if profile and '*' not in profile.supported_languages and lang in profile.supported_languages:
                            if (model_name, langs) not in priority_models:
                                priority_models.append((model_name, langs))
                                break

                # Add some high-performing general models
                for model_name in ['DeBERTaV3Base', 'RoBERTaBase', 'ELECTRABase']:
                    for m_name, langs in models_with_languages:
                        if m_name == model_name and (m_name, langs) not in priority_models:
                            priority_models.append((m_name, langs))
                            if len(priority_models) >= 10:
                                break

                models_with_languages = priority_models[:10]
        else:
            # When test_all=True, show info about models being tested
            self.logger.info(f"Testing {len(models_with_languages)} models for languages: {', '.join(detected_languages)}")

            # Group models by language support for logging
            multilingual_models = []
            language_specific = {}

            for model_name, langs in models_with_languages:
                profile = model_profiles.get(model_name)
                if profile and '*' in profile.supported_languages:
                    multilingual_models.append(model_name)
                else:
                    for lang in langs:
                        if lang not in language_specific:
                            language_specific[lang] = []
                        if model_name not in language_specific[lang]:
                            language_specific[lang].append(model_name)

            if multilingual_models:
                self.logger.info(f"  Multilingual models ({len(multilingual_models)}): {', '.join(sorted(multilingual_models))}")

            for lang, models in language_specific.items():
                lang_emoji = "🇫🇷" if lang.upper() == 'FR' else "🇬🇧" if lang.upper() == 'EN' else "🌍"
                self.logger.info(f"  {lang_emoji} {lang}-specific models ({len(models)}): {', '.join(sorted(models))}")

        return models_with_languages

    def _benchmark_models_with_languages(
        self,
        dataset: BenchmarkDataset,
        models_to_test: List[Tuple[str, List[str]]],
        detected_languages: List[str],
        benchmark_epochs: int,
        verbose: bool = True,
        global_total_models: Optional[int] = None,
        global_total_epochs: Optional[int] = None,
        global_max_epochs: Optional[int] = None,
        global_start_time: Optional[float] = None
    ) -> List[Dict]:
        """
        Benchmark multiple models with language-specific testing.
        """
        all_results = []
        global_completed_epochs = 0  # Track completed epochs across all models

        for idx, (model_name, test_languages) in enumerate(models_to_test, 1):
            if verbose:
                print(f"\n[{idx}/{len(models_to_test)}] Testing {model_name} on {', '.join(test_languages)}...")

            try:
                # Get model class
                model_class = self._get_model_class(model_name)
                if not model_class:
                    self.logger.warning(f"Model class not found for {model_name}")
                    continue

                # Filter data by languages if needed
                if test_languages != detected_languages and dataset.test_languages:
                    # Filter for specific languages
                    train_idx = [i for i, lang in enumerate(dataset.train_languages) if lang in test_languages]
                    test_idx = [i for i, lang in enumerate(dataset.test_languages) if lang in test_languages]

                    if not train_idx or not test_idx:
                        if verbose:
                            print(f"   ⚠️ No data for {model_name} in {', '.join(test_languages)}")
                        continue

                    filtered_train_texts = [dataset.train_texts[i] for i in train_idx]
                    filtered_train_labels = [dataset.train_labels[i] for i in train_idx]
                    filtered_test_texts = [dataset.test_texts[i] for i in test_idx]
                    filtered_test_labels = [dataset.test_labels[i] for i in test_idx]
                    filtered_test_languages = [dataset.test_languages[i] for i in test_idx]
                else:
                    # Use all data
                    filtered_train_texts = dataset.train_texts
                    filtered_train_labels = dataset.train_labels
                    filtered_test_texts = dataset.test_texts
                    filtered_test_labels = dataset.test_labels
                    filtered_test_languages = dataset.test_languages

                # Initialize model
                model = model_class()

                # ==================== LANGUAGE FILTERING FOR MONOLINGUAL MODELS ====================
                # Apply additional filtering for language-specific models
                from .model_trainer import get_model_target_languages, filter_data_by_language
                import pandas as pd

                # Get model name from class
                actual_model_name = model.model_name if hasattr(model, 'model_name') else model_name
                target_languages = get_model_target_languages(actual_model_name)

                if target_languages is not None and filtered_test_languages:
                    # This model should only train on specific languages
                    if verbose:
                        print(f"   🌍 Filtering data to {target_languages} for {model_name}")

                    # Create dataframes for filtering
                    train_df = pd.DataFrame({
                        'text': filtered_train_texts,
                        'label': filtered_train_labels,
                        'language': filtered_test_languages[:len(filtered_train_texts)]
                    })
                    test_df = pd.DataFrame({
                        'text': filtered_test_texts,
                        'label': filtered_test_labels,
                        'language': filtered_test_languages
                    })

                    # Filter to target language
                    train_df_filtered = filter_data_by_language(train_df, target_languages, 'language')
                    test_df_filtered = filter_data_by_language(test_df, target_languages, 'language')

                    # Check if we have enough data
                    if len(train_df_filtered) < 10 or len(test_df_filtered) < 5:
                        if verbose:
                            print(f"   ⚠️ Insufficient data after language filtering: "
                                 f"{len(train_df_filtered)} train, {len(test_df_filtered)} test. Skipping.")
                        continue

                    # Update filtered data - CRITICAL: Use safe_tolist to convert numpy types to Python native types
                    filtered_train_texts = train_df_filtered['text'].tolist()
                    filtered_train_labels = safe_tolist(train_df_filtered['label'], column_name='label')
                    filtered_test_texts = test_df_filtered['text'].tolist()
                    filtered_test_labels = safe_tolist(test_df_filtered['label'], column_name='label')
                    filtered_test_languages = safe_tolist(train_df_filtered['language'], column_name='language') + safe_tolist(test_df_filtered['language'], column_name='language')

                    if verbose:
                        print(f"   ✓ Filtered to {len(filtered_train_texts)} train + {len(filtered_test_texts)} test samples")

                # Determine if model needs adjustment based on its characteristics
                # Large models or models designed for long sequences often struggle with short texts
                needs_adjustment = (
                    'Large' in model_name or
                    'large' in model_name.lower() or
                    'XLarge' in model_name or
                    'Longformer' in model_name or
                    'BigBird' in model_name or
                    # Models with specific architectures that need more training
                    'ALBERT' in model_name or  # Parameter sharing requires more epochs
                    'XLM' in model_name  # Multilingual models need more adaptation
                )

                # Calculate average sequence length (always calculate for problematic models)
                avg_seq_length = 0
                if needs_adjustment:
                    # Always calculate for problematic models
                    avg_seq_length = np.mean([len(t.split()) for t in filtered_train_texts[:100]]) * 1.3  # Approx tokens
                    if verbose:
                        print(f"   📏 Avg sequence length: ~{avg_seq_length:.0f} tokens")
                elif self.config.optimize_for_short_sequences:
                    # Also calculate if optimization is requested
                    avg_seq_length = np.mean([len(t.split()) for t in filtered_train_texts[:100]]) * 1.3  # Approx tokens
                    if verbose:
                        print(f"   📏 Avg sequence length: ~{avg_seq_length:.0f} tokens")

                # Adjust parameters for problematic models with short sequences
                adjusted_batch_size = self.config.batch_size
                adjusted_lr = self.config.learning_rate
                adjusted_epochs = benchmark_epochs

                if needs_adjustment and self.config.large_model_adjustments:
                    # SMART PARAMETER ADJUSTMENT BASED ON MODEL CHARACTERISTICS

                    # Determine adjustment strategy based on model type and sequence length
                    is_short_seq = avg_seq_length > 0 and avg_seq_length < self.config.short_sequence_threshold
                    is_long_context_model = 'Longformer' in model_name or 'BigBird' in model_name
                    is_parameter_sharing = 'ALBERT' in model_name
                    is_multilingual = 'XLM' in model_name or 'mDeBERTa' in model_name
                    is_large_model = 'Large' in model_name or 'large' in model_name.lower()

                    # Apply adjustments based on characteristics
                    if is_short_seq:
                        # Short sequences require specific adjustments
                        if is_long_context_model:
                            # Long-context models struggle with short sequences
                            adjusted_batch_size = max(4, self.config.batch_size // 4)
                            adjusted_lr = self.config.learning_rate * 2.0
                            adjusted_epochs = benchmark_epochs * 2
                            adjustment_reason = "long-context model with short sequences"
                        elif is_multilingual:
                            # Multilingual models need aggressive adjustments for short sequences
                            adjusted_batch_size = max(4, self.config.batch_size // 4)
                            adjusted_lr = self.config.learning_rate * 2.0
                            adjusted_epochs = benchmark_epochs * 2
                            adjustment_reason = "multilingual model with short sequences"
                        elif is_large_model:
                            # Large models tend to overfit on short sequences
                            adjusted_batch_size = max(8, self.config.batch_size // 2)
                            adjusted_lr = self.config.learning_rate * 0.3
                            adjusted_epochs = benchmark_epochs * 1.5
                            adjustment_reason = "large model with short sequences"
                        else:
                            # Generic adjustment for short sequences
                            adjusted_batch_size = max(8, self.config.batch_size // 2)
                            adjusted_lr = self.config.learning_rate * 1.2
                            adjusted_epochs = benchmark_epochs * 1.5
                            adjustment_reason = "model with short sequences"
                    else:
                        # Normal sequences but model still needs help
                        if is_parameter_sharing:
                            # ALBERT needs more epochs due to parameter sharing
                            adjusted_epochs = benchmark_epochs * 2
                            adjusted_lr = self.config.learning_rate * 0.5
                            adjustment_reason = "parameter sharing architecture"
                        elif is_multilingual:
                            # Multilingual models need more training
                            adjusted_epochs = benchmark_epochs * 1.5
                            adjusted_lr = self.config.learning_rate * 1.2
                            adjustment_reason = "multilingual model"
                        elif is_large_model:
                            # Large models need careful tuning
                            adjusted_batch_size = max(8, self.config.batch_size // 2)
                            adjusted_lr = self.config.learning_rate * 0.5
                            adjusted_epochs = benchmark_epochs * 1.5
                            adjustment_reason = "large model architecture"
                        else:
                            # Default adjustment for models that need help
                            adjusted_epochs = benchmark_epochs * 1.5
                            adjusted_lr = self.config.learning_rate * 0.8
                            adjustment_reason = "model needs extra support"

                    # Ensure epochs is an integer
                    adjusted_epochs = int(adjusted_epochs)

                    if verbose and (adjusted_batch_size != self.config.batch_size or
                                   adjusted_lr != self.config.learning_rate or
                                   adjusted_epochs != benchmark_epochs):
                        print(f"   ⚙️ Auto-adjusting parameters ({adjustment_reason}):")
                        if adjusted_batch_size != self.config.batch_size:
                            print(f"      Batch: {self.config.batch_size} → {adjusted_batch_size}")
                        if adjusted_lr != self.config.learning_rate:
                            print(f"      LR: {self.config.learning_rate:.2e} → {adjusted_lr:.2e}")
                        if adjusted_epochs != benchmark_epochs:
                            print(f"      Epochs: {benchmark_epochs} → {adjusted_epochs}")

                # Create data loaders with adjusted batch size
                train_loader = model.encode(
                    filtered_train_texts,
                    filtered_train_labels,
                    batch_size=adjusted_batch_size,
                    progress_bar=False,
                    shuffle=True  # Shuffle training data
                )
                test_loader = model.encode(
                    filtered_test_texts,
                    filtered_test_labels,
                    batch_size=adjusted_batch_size,
                    progress_bar=False,
                    shuffle=False  # Don't shuffle test data
                )

                # UNIFIED: Use centralized function to set detected languages (SAME AS BENCHMARK)
                if filtered_test_languages:
                    from .model_trainer import set_detected_languages_on_model
                    set_detected_languages_on_model(
                        model=model,
                        val_languages=filtered_test_languages,
                        logger=self.logger
                    )

                # Run training
                import time
                import numpy as np
                start_time = time.time()

                # Create progress callback to track completed epochs
                def progress_callback(**metrics):
                    """Callback to increment global completed epochs counter"""
                    nonlocal global_completed_epochs
                    global_completed_epochs += 1

                # Decide whether to use reinforced learning
                # Pass reinforced_learning if it's enabled in config
                # This allows bert_base.py to automatically trigger if F1_1 < threshold
                use_reinforced = self.config.reinforced_learning

                # Log reinforced status for transparency
                if verbose:
                    if use_reinforced:
                        print(f"   ⚡ Reinforced learning enabled (will auto-trigger if F1_1 < {self.config.reinforced_f1_threshold:.2f})")

                        # Additional context for models likely to need it
                        if needs_adjustment:
                            if avg_seq_length > 0 and avg_seq_length < self.config.short_sequence_threshold:
                                print(f"      Note: Model has short sequences (~{avg_seq_length:.0f} tokens) - reinforced likely needed")
                            else:
                                print(f"      Note: Model architecture often benefits from reinforced training")
                    else:
                        print(f"   ℹ️ Reinforced learning disabled (activate in config to enable auto-triggering)")

                metrics_output_dir = str(resolve_metrics_base_dir(None))
                summary = model.run_training(
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    n_epochs=adjusted_epochs,
                    lr=adjusted_lr,
                    save_model_as=f"benchmark_{model_name.lower()}",
                    track_languages=self.config.track_languages and filtered_test_languages is not None,
                    language_info=filtered_test_languages if filtered_test_languages else None,
                    reinforced_learning=use_reinforced,
                    n_epochs_reinforced=self.config.reinforced_epochs if use_reinforced else 0,
                    reinforced_epochs=self.config.reinforced_epochs if use_reinforced else None,
                    f1_class_1_weight=self.config.f1_class_1_weight,
                    rescue_low_class1_f1=self.config.rescue_low_class1_f1,
                    f1_1_rescue_threshold=self.config.f1_rescue_threshold,
                    reinforced_f1_threshold=self.config.reinforced_f1_threshold,
                    model_identifier=f"benchmark_{model_name.lower()}",
                    metrics_output_dir=metrics_output_dir,
                    label_key=None,  # Benchmark uses label_value directly
                    label_value=dataset.category,  # CRITICAL: Category name from dataset
                    language=dataset.language,  # CRITICAL: Language from dataset
                    progress_callback=progress_callback,  # Add callback for global progress tracking
                    global_total_models=global_total_models,
                    global_current_model=idx,
                    global_total_epochs=global_total_epochs,
                    global_max_epochs=global_max_epochs,
                    global_completed_epochs=global_completed_epochs,
                    global_start_time=global_start_time,
                    session_id=self.session_id  # CRITICAL: Unified session ID for all runs
                )

                training_time = time.time() - start_time

                # Log if reinforced learning was triggered
                if use_reinforced and summary.get('reinforced_triggered', False):
                    if verbose:
                        print(f"   ⚡ Reinforced learning was triggered (F1_1 < {self.config.reinforced_f1_threshold})")

                # Clean up model
                if hasattr(model, 'last_saved_model_path') and model.last_saved_model_path:
                    try:
                        shutil.rmtree(model.last_saved_model_path, ignore_errors=True)
                    except:
                        pass

                # Prepare result
                result = {
                    'model_name': model_name,
                    'tested_languages': test_languages,
                    'f1_score': summary.get('macro_f1', 0),
                    'accuracy': summary.get('accuracy', 0),
                    'f1_class_0': summary.get('f1_0', 0),
                    'f1_class_1': summary.get('f1_1', 0),
                    'precision_0': summary.get('precision_0', 0),
                    'precision_1': summary.get('precision_1', 0),
                    'recall_0': summary.get('recall_0', 0),
                    'recall_1': summary.get('recall_1', 0),
                    'training_time': training_time,
                    'inference_time': training_time / (benchmark_epochs * len(filtered_test_texts)),
                    'language_metrics': summary.get('language_metrics', {})
                }

                all_results.append(result)

            except Exception as e:
                self.logger.error(f"Error benchmarking {model_name}: {e}")
                if verbose:
                    print(f"   ❌ Error: {e}")

        return all_results

    def _get_model_class(self, model_name: str):
        """Get model class from name using ModelSelector."""
        # Get from ModelSelector profiles first
        selector = ModelSelector(verbose=False)
        if model_name in selector.MODEL_PROFILES:
            return selector.MODEL_PROFILES[model_name].model_class

        # Fallback to legacy mapping for backward compatibility
        model_map = {
            'Bert': Bert,
            'Camembert': Camembert
        }
        return model_map.get(model_name)

    def _display_benchmark_results(
        self,
        results: List[Dict],
        detected_languages: List[str],
        is_multilingual: bool
    ):
        """Display comprehensive benchmark results."""
        print("\n" + "="*120)
        print("🏆 COMPREHENSIVE BENCHMARK RESULTS")
        print("="*120)

        # Main results table
        print(f"\n{'#':<4} {'Model':<25} {'Languages':<15} {'F1 Score':<12} {'Accuracy':<12} {'F1 Cl.1':<12} {'Time (ms)':<12}")
        print("-"*120)

        for i, result in enumerate(results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            lang_str = ', '.join(result.get('tested_languages', []))

            print(f"{emoji}[{i}] {result['model_name']:<25} {lang_str:<15} "
                  f"{result['f1_score']:<12.3f} {result['accuracy']:<12.3f} "
                  f"{result['f1_class_1']:<12.3f} {result['inference_time']*1000:<12.1f}")

        print("="*120)

        # Language-specific metrics for ALL models
        if is_multilingual:
            print("\n🌍 LANGUAGE-SPECIFIC PERFORMANCE (ALL MODELS):")
            print("="*120)
            print(f"{'Model':<25} {'Language':<10} {'F1 Score':<12} {'Accuracy':<12} {'F1 Cl.0':<12} {'F1 Cl.1':<12}")
            print("-"*120)

            for result in results:
                if result.get('language_metrics'):
                    for lang, metrics in result['language_metrics'].items():
                        print(f"{result['model_name']:<25} {lang:<10} "
                              f"{metrics.get('macro_f1', 0):<12.3f} "
                              f"{metrics.get('accuracy', 0):<12.3f} "
                              f"{metrics.get('f1_0', 0):<12.3f} "
                              f"{metrics.get('f1_1', 0):<12.3f}")
                else:
                    # Show aggregate for each tested language
                    for lang in result.get('tested_languages', []):
                        print(f"{result['model_name']:<25} {lang:<10} "
                              f"{result['f1_score']:<12.3f} "
                              f"{result['accuracy']:<12.3f} "
                              f"{result['f1_class_0']:<12.3f} "
                              f"{result['f1_class_1']:<12.3f}")

            print("="*120)

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print(f"   ✅ Best overall F1: {results[0]['model_name']} ({results[0]['f1_score']:.3f})")

        # Best for minority class
        best_minority = max(results, key=lambda x: x.get('f1_class_1', 0))
        print(f"   🎯 Best for minority class: {best_minority['model_name']} (F1: {best_minority['f1_class_1']:.3f})")

        # Fastest
        fastest = min(results, key=lambda x: x.get('inference_time', float('inf')))
        print(f"   ⚡ Fastest: {fastest['model_name']} ({fastest['inference_time']*1000:.1f} ms/sample)")

    def _select_best_multilingual_model(self, results: List[Dict], languages: List[str]) -> str:
        """
        Select best model considering performance across all languages.

        For multilingual data, we calculate a weighted score based on:
        - Overall F1 macro across all languages
        - Minimum F1_1 across languages (to avoid models that fail in one language)
        - Balance between languages
        """
        best_score = -1
        best_model = None

        for result in results:
            # Calculate composite score
            overall_f1 = result.get('f1_score', 0)
            overall_f1_class1 = result.get('f1_class_1', 0)

            # Get language-specific metrics
            lang_metrics = result.get('language_metrics', {})

            if lang_metrics:
                # Calculate minimum and average F1_1 across languages
                f1_1_scores = []
                f1_macro_scores = []

                for lang in languages:
                    if lang in lang_metrics:
                        lang_data = lang_metrics[lang]
                        f1_1_scores.append(lang_data.get('f1_1', 0))
                        f1_macro_scores.append(lang_data.get('macro_f1', 0))

                if f1_1_scores:
                    min_f1_1 = min(f1_1_scores)
                    avg_f1_1 = sum(f1_1_scores) / len(f1_1_scores)
                    avg_f1_macro = sum(f1_macro_scores) / len(f1_macro_scores)

                    # Composite score:
                    # - 40% overall F1 macro
                    # - 30% minimum F1_1 across languages (penalize if one language fails)
                    # - 20% average F1_1 across languages
                    # - 10% overall F1_1
                    composite_score = (
                        0.4 * avg_f1_macro +
                        0.3 * min_f1_1 +
                        0.2 * avg_f1_1 +
                        0.1 * overall_f1_class1
                    )
                else:
                    # Fallback to overall metrics
                    composite_score = 0.7 * overall_f1 + 0.3 * overall_f1_class1
            else:
                # No language metrics, use overall
                composite_score = 0.7 * overall_f1 + 0.3 * overall_f1_class1

            if composite_score > best_score:
                best_score = composite_score
                best_model = result['model_name']

        return best_model if best_model else (results[0]['model_name'] if results else None)

    def _user_select_model(self, results: List[Dict]) -> Optional[str]:
        """Let user select model from results."""
        print("\n" + "="*120)
        print("🎮 MODEL SELECTION")
        print("="*120)

        print("\nSelect model for full training:")
        print("  [0] Cancel")

        for i, result in enumerate(results[:15], 1):  # Show up to 15 models
            print(f"  [{i}] {result['model_name']}")

        while True:
            try:
                choice = input("\nYour choice: ").strip()

                if choice == '0':
                    return None
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(results):
                        selected = results[idx]['model_name']
                        print(f"\n✅ Selected: {selected}")
                        return selected
                else:
                    # Try to match by name
                    for result in results:
                        if choice.lower() in result['model_name'].lower():
                            print(f"\n✅ Selected: {result['model_name']}")
                            return result['model_name']

                print("❌ Invalid choice")

            except KeyboardInterrupt:
                return None

    def _save_benchmark_logs(
        self,
        results: List[Dict],
        detected_languages: List[str],
        is_multilingual: bool,
        data_path: Path,
        benchmark_epochs: int,
        save_detailed: bool = True,
        save_best: bool = True
    ):
        """Save comprehensive benchmark results to CSV files."""
        import datetime

        # Create logs directory
        logs_dir = self.models_root / "benchmark_logs"
        logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Also create a consolidated file that accumulates all benchmarks
        consolidated_file = logs_dir / "benchmark_all_runs_consolidated.csv"
        consolidated_best_file = logs_dir / "benchmark_best_models_consolidated.csv"

        # Save detailed log with ALL models and ALL metrics
        if save_detailed and results:
            detailed_file = logs_dir / f"benchmark_detailed_{timestamp}.csv"

            # Prepare headers
            headers = [
                'timestamp', 'rank', 'model_name', 'huggingface_name', 'tested_languages',
                'f1_macro', 'accuracy', 'f1_class_0', 'f1_class_1',
                'precision_0', 'precision_1', 'recall_0', 'recall_1',
                'training_time', 'inference_time_ms'
            ]

            # Add language-specific headers if multilingual
            if is_multilingual:
                for lang in detected_languages:
                    headers.extend([
                        f'{lang}_f1_macro', f'{lang}_accuracy',
                        f'{lang}_f1_0', f'{lang}_f1_1',
                        f'{lang}_samples'
                    ])

            # Add metadata headers
            headers.extend(['data_path', 'benchmark_epochs', 'total_samples'])

            # Write detailed CSV
            with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                for rank, result in enumerate(results, 1):
                    # Get the actual HuggingFace model name from ModelSelector
                    selector = ModelSelector(verbose=False)
                    huggingface_name = result['model_name']
                    if result['model_name'] in selector.MODEL_PROFILES:
                        huggingface_name = selector.MODEL_PROFILES[result['model_name']].name

                    row = {
                        'timestamp': timestamp,
                        'rank': rank,
                        'model_name': result['model_name'],
                        'huggingface_name': huggingface_name,
                        'tested_languages': '|'.join(result.get('tested_languages', [])),
                        'f1_macro': result.get('f1_score', 0),
                        'accuracy': result.get('accuracy', 0),
                        'f1_class_0': result.get('f1_class_0', 0),
                        'f1_class_1': result.get('f1_class_1', 0),
                        'precision_0': result.get('precision_0', 0),
                        'precision_1': result.get('precision_1', 0),
                        'recall_0': result.get('recall_0', 0),
                        'recall_1': result.get('recall_1', 0),
                        'training_time': result.get('training_time', 0),
                        'inference_time_ms': result.get('inference_time', 0) * 1000,
                        'data_path': str(data_path),
                        'benchmark_epochs': benchmark_epochs,
                        'total_samples': len(result.get('tested_languages', [])) * 100  # Approximate
                    }

                    # Add language-specific metrics
                    if is_multilingual and result.get('language_metrics'):
                        for lang in detected_languages:
                            lang_metrics = result['language_metrics'].get(lang, {})
                            row[f'{lang}_f1_macro'] = lang_metrics.get('macro_f1', 0)
                            row[f'{lang}_accuracy'] = lang_metrics.get('accuracy', 0)
                            row[f'{lang}_f1_0'] = lang_metrics.get('f1_0', 0)
                            row[f'{lang}_f1_1'] = lang_metrics.get('f1_1', 0)
                            row[f'{lang}_samples'] = lang_metrics.get('samples', 0)

                    writer.writerow(row)

            self.logger.info(f"💾 Detailed benchmark log saved: {detailed_file}")
            print(f"\n💾 Detailed log saved: {detailed_file}")

            # Also append to consolidated file
            consolidated_exists = consolidated_file.exists()
            with open(consolidated_file, 'a' if consolidated_exists else 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not consolidated_exists:
                    writer.writeheader()

                # Re-write all rows to consolidated file
                for rank, result in enumerate(results, 1):
                    selector = ModelSelector(verbose=False)
                    huggingface_name = result['model_name']
                    if result['model_name'] in selector.MODEL_PROFILES:
                        huggingface_name = selector.MODEL_PROFILES[result['model_name']].name

                    row = {
                        'timestamp': timestamp,
                        'rank': rank,
                        'model_name': result['model_name'],
                        'huggingface_name': huggingface_name,
                        'tested_languages': '|'.join(result.get('tested_languages', [])),
                        'f1_macro': result.get('f1_score', 0),
                        'accuracy': result.get('accuracy', 0),
                        'f1_class_0': result.get('f1_class_0', 0),
                        'f1_class_1': result.get('f1_class_1', 0),
                        'precision_0': result.get('precision_0', 0),
                        'precision_1': result.get('precision_1', 0),
                        'recall_0': result.get('recall_0', 0),
                        'recall_1': result.get('recall_1', 0),
                        'training_time': result.get('training_time', 0),
                        'inference_time_ms': result.get('inference_time', 0) * 1000,
                        'data_path': str(data_path),
                        'benchmark_epochs': benchmark_epochs,
                        'total_samples': len(result.get('tested_languages', [])) * 100
                    }

                    if is_multilingual and result.get('language_metrics'):
                        for lang in detected_languages:
                            lang_metrics = result['language_metrics'].get(lang, {})
                            row[f'{lang}_f1_macro'] = lang_metrics.get('macro_f1', 0)
                            row[f'{lang}_accuracy'] = lang_metrics.get('accuracy', 0)
                            row[f'{lang}_f1_0'] = lang_metrics.get('f1_0', 0)
                            row[f'{lang}_f1_1'] = lang_metrics.get('f1_1', 0)
                            row[f'{lang}_samples'] = lang_metrics.get('samples', 0)

                    writer.writerow(row)

            print(f"✅ Consolidated log updated: {consolidated_file}")

        # Save best models summary
        if save_best and results:
            best_file = logs_dir / f"benchmark_best_models_{timestamp}.csv"

            # Select best models by different criteria
            best_models = {
                'overall_f1': results[0] if results else None,
                'minority_class': max(results, key=lambda x: x.get('f1_class_1', 0)),
                'fastest': min(results, key=lambda x: x.get('inference_time', float('inf'))),
                'balanced': min(results, key=lambda x: abs(x.get('f1_class_0', 0) - x.get('f1_class_1', 0)))
            }

            # Add language-specific best if multilingual
            if is_multilingual:
                for lang in detected_languages:
                    # Find best model for this language
                    lang_results = []
                    for r in results:
                        if r.get('language_metrics', {}).get(lang):
                            lang_f1 = r['language_metrics'][lang].get('macro_f1', 0)
                            lang_results.append((lang_f1, r))

                    if lang_results:
                        best_models[f'best_{lang}'] = max(lang_results, key=lambda x: x[0])[1]

            # Write best models CSV
            with open(best_file, 'w', newline='', encoding='utf-8') as f:
                headers = ['criterion', 'model_name', 'huggingface_name', 'f1_score', 'accuracy',
                          'f1_class_1', 'inference_time_ms', 'reason']
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                # Get ModelSelector for HuggingFace names
                selector = ModelSelector(verbose=False)

                for criterion, model in best_models.items():
                    if model:
                        # Get actual HuggingFace name
                        huggingface_name = model['model_name']
                        if model['model_name'] in selector.MODEL_PROFILES:
                            huggingface_name = selector.MODEL_PROFILES[model['model_name']].name

                        reason = {
                            'overall_f1': f"Highest macro F1 score ({model['f1_score']:.3f})",
                            'minority_class': f"Best minority class detection (F1: {model['f1_class_1']:.3f})",
                            'fastest': f"Fastest inference ({model['inference_time']*1000:.1f}ms)",
                            'balanced': f"Most balanced predictions (diff: {abs(model.get('f1_class_0', 0) - model.get('f1_class_1', 0)):.3f})"
                        }.get(criterion, f"Best for {criterion}")

                        writer.writerow({
                            'criterion': criterion,
                            'model_name': model['model_name'],
                            'huggingface_name': huggingface_name,
                            'f1_score': model.get('f1_score', 0),
                            'accuracy': model.get('accuracy', 0),
                            'f1_class_1': model.get('f1_class_1', 0),
                            'inference_time_ms': model.get('inference_time', 0) * 1000,
                            'reason': reason
                        })

            self.logger.info(f"🏆 Best models summary saved: {best_file}")
            print(f"🏆 Best models saved: {best_file}")

            # Also append to consolidated best models file
            consolidated_best_exists = consolidated_best_file.exists()
            with open(consolidated_best_file, 'a' if consolidated_best_exists else 'w', newline='', encoding='utf-8') as f:
                headers_best = ['timestamp', 'criterion', 'model_name', 'huggingface_name', 'f1_score',
                               'accuracy', 'f1_class_1', 'inference_time_ms', 'reason', 'data_path']
                writer = csv.DictWriter(f, fieldnames=headers_best)
                if not consolidated_best_exists:
                    writer.writeheader()

                for criterion, model in best_models.items():
                    if model:
                        huggingface_name = model['model_name']
                        if model['model_name'] in selector.MODEL_PROFILES:
                            huggingface_name = selector.MODEL_PROFILES[model['model_name']].name

                        reason = {
                            'overall_f1': f"Highest macro F1 score ({model['f1_score']:.3f})",
                            'minority_class': f"Best minority class detection (F1: {model['f1_class_1']:.3f})",
                            'fastest': f"Fastest inference ({model['inference_time']*1000:.1f}ms)",
                            'balanced': f"Most balanced predictions (diff: {abs(model.get('f1_class_0', 0) - model.get('f1_class_1', 0)):.3f})"
                        }.get(criterion, f"Best for {criterion}")

                        writer.writerow({
                            'timestamp': timestamp,
                            'criterion': criterion,
                            'model_name': model['model_name'],
                            'huggingface_name': huggingface_name,
                            'f1_score': model.get('f1_score', 0),
                            'accuracy': model.get('accuracy', 0),
                            'f1_class_1': model.get('f1_class_1', 0),
                            'inference_time_ms': model.get('inference_time', 0) * 1000,
                            'reason': reason,
                            'data_path': str(data_path)
                        })

            print(f"✅ Consolidated best models log updated: {consolidated_best_file}")

        # Also save a JSON version for programmatic access
        json_file = logs_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'data_path': str(data_path),
                'benchmark_epochs': benchmark_epochs,
                'languages_detected': detected_languages,
                'is_multilingual': is_multilingual,
                'models_tested': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"📋 JSON results saved: {json_file}")

#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
training_data_builder.py

MAIN OBJECTIVE:
---------------
Convert heterogeneous annotation exports into trainer-ready datasets for the
Training Arena, handling session-aware output paths and metadata collection.

Dependencies:
-------------
- dataclasses
- pathlib
- typing
- json
- logging
- datetime
- pandas
- llm_tool.utils.annotation_to_training

MAIN FEATURES:
--------------
1) Support multiple input formats (LLM JSON, category CSV, JSONL, binary tables)
2) Construct TrainingDataBundle objects with strategy and metadata details
3) Manage session-scoped output directories for reproducible pipelines
4) Provide helper methods that transform raw annotations into flat datasets
5) Produce trainer configuration snippets tailored to downstream trainers

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import ast
import logging
from datetime import datetime

import pandas as pd

from llm_tool.utils.language_normalizer import LanguageNormalizer

from llm_tool.utils.annotation_to_training import AnnotationToTrainingConverter


LOGGER = logging.getLogger(__name__)


def _timestamp() -> str:
    """Return a compact timestamp suitable for filenames."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class TrainingDataBundle:
    """Description of the assets generated for a training session."""

    primary_file: Optional[Path] = None
    #: Mapping label -> path for binary datasets (used for multi-label flows)
    training_files: Dict[str, Path] = field(default_factory=dict)
    #: Strategy expected by the trainer (`single-label` or `multi-label`)
    strategy: str = "single-label"
    text_column: str = "text"
    label_column: str = "label"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_trainer_config(
        self,
        output_dir: Path,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the configuration dictionary expected by `ModelTrainer`."""

        config: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "training_strategy": self.strategy,
            "text_column": self.text_column,
            "label_column": self.label_column,
        }

        if self.training_files:
            config["training_files"] = {k: str(v) for k, v in self.training_files.items()}
            if self.primary_file:
                config["input_file"] = str(self.primary_file)
        elif self.primary_file:
            config["input_file"] = str(self.primary_file)

        if extra:
            config.update(extra)

        return config


@dataclass
class TrainingDataRequest:
    """Parameters describing how to build a dataset."""

    input_path: Path
    format: str
    text_column: str = "text"
    label_column: str = "label"
    annotation_column: str = "annotation"
    annotation_keys: Optional[List[str]] = None
    label_strategy: str = "key_value"
    mode: str = "single-label"
    category_column: Optional[str] = None
    value_column: Optional[str] = None
    id_column: Optional[str] = None
    lang_column: Optional[str] = None
    key_strategies: Optional[Dict[str, str]] = None  # {key_name: 'multi-class' or 'one-vs-all'}
    output_dir: Optional[Path] = None
    output_format: str = "jsonl"


class TrainingDatasetBuilder:
    """Factory capable of converting several raw formats into training assets."""

    SUPPORTED_FORMATS = {
        "llm_json": "_build_llm_annotations",
        "category_csv": "_build_category_csv",
        "binary_long_csv": "_build_binary_long",
        "jsonl_single": "_build_jsonl_single",
        "jsonl_multi": "_build_jsonl_multi",
        "prepared": "_build_prepared",
    }

    def __init__(self, base_output_dir: Path, session_id: Optional[str] = None,
                 use_training_data_subdir: bool = True):
        """
        Initialize training dataset builder.

        Args:
            base_output_dir: Base directory for training data outputs
            session_id: Optional session ID for session-based organization.
                       If provided, files will be saved to {base_output_dir}/{session_id}/training_data/
                       If None, files will be saved directly to {base_output_dir}/ (flat structure)
            use_training_data_subdir: If True, add /training_data subdir. If False, use base_output_dir directly.
                                    For Annotator Factory, set to True to organize in training_data/
        """
        if session_id:
            # Session-based: base_output_dir/{session_id}/training_data/
            if use_training_data_subdir:
                self.base_output_dir = Path(base_output_dir) / session_id / "training_data"
            else:
                self.base_output_dir = Path(base_output_dir) / session_id
        else:
            # Flat structure
            if use_training_data_subdir:
                self.base_output_dir = Path(base_output_dir) / "training_data"
            else:
                self.base_output_dir = Path(base_output_dir)

        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.converter = AnnotationToTrainingConverter(verbose=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, request: TrainingDataRequest) -> TrainingDataBundle:
        handler_name = self.SUPPORTED_FORMATS.get(request.format)
        if not handler_name or not hasattr(self, handler_name):
            raise ValueError(
                f"Unsupported format '{request.format}'. Supported formats: {sorted(self.SUPPORTED_FORMATS)}"
            )

        handler = getattr(self, handler_name)
        LOGGER.info("Building training dataset", extra={"format": request.format, "input": str(request.input_path)})
        bundle: TrainingDataBundle = handler(request)
        bundle.metadata.setdefault("source_format", request.format)
        bundle.metadata.setdefault("input_path", str(request.input_path))
        return bundle

    # ------------------------------------------------------------------
    # Format handlers
    # ------------------------------------------------------------------
    def _build_llm_annotations(self, request: TrainingDataRequest) -> TrainingDataBundle:
        dataset_dir = self._ensure_output_dir(request.output_dir)
        suffix = _timestamp()

        analysis = self.converter.analyze_annotations(
            csv_path=str(request.input_path),
            text_column=request.text_column,
            annotation_column=request.annotation_column,
        )

        metadata = {"analysis": analysis}
        annotation_keys = request.annotation_keys or list(analysis.get("annotation_keys", {}).keys())

        if request.mode == "single-label":
            # CRITICAL: Hybrid/Custom mode support with key_strategies
            # key_strategies = {key_name: 'multi-class' or 'one-vs-all'}
            if request.key_strategies and len(annotation_keys) > 1:
                LOGGER.info(f"Hybrid/Custom mode with {len(annotation_keys)} keys")

                category_files: Dict[str, Path] = {}
                multiclass_keys = []
                onevsall_keys = []

                # Separate keys by strategy
                for key in annotation_keys:
                    strategy = request.key_strategies.get(key, 'multi-class')
                    if strategy == 'multi-class':
                        multiclass_keys.append(key)
                    else:
                        onevsall_keys.append(key)

                # Create files for multi-class keys (one file per key)
                for key in multiclass_keys:
                    key_output_file = dataset_dir / f"multiclass_{self._slugify(key)}_{suffix}.jsonl"
                    key_path = self.converter.create_single_key_dataset(
                        csv_path=str(request.input_path),
                        output_path=str(key_output_file),
                        text_column=request.text_column,
                        annotation_column=request.annotation_column,
                        annotation_key=key,
                        label_strategy=request.label_strategy,
                        id_column=request.id_column,
                        lang_column=request.lang_column,
                    )

                    if key_path:
                        category_files[key] = Path(key_path)
                        LOGGER.info(f"Created multi-class file for key '{key}': {key_output_file}")

                # Create files for one-vs-all keys (one file per value)
                # We'll use the multi-label dataset with only those keys
                if onevsall_keys:
                    onevsall_multilabel_path = dataset_dir / f"onevsall_keys_{suffix}.jsonl"
                    self.converter.create_multi_label_dataset(
                        csv_path=str(request.input_path),
                        output_path=str(onevsall_multilabel_path),
                        text_column=request.text_column,
                        annotation_column=request.annotation_column,
                        annotation_keys=onevsall_keys,
                        label_strategy=request.label_strategy,
                        id_column=request.id_column,
                        lang_column=request.lang_column,
                    )
                    category_files['onevsall_multilabel'] = onevsall_multilabel_path
                    LOGGER.info(f"Created one-vs-all multilabel file for {len(onevsall_keys)} keys")

                # Create a consolidated multi-label file for compatibility (optional)
                multilabel_path = dataset_dir / f"multilabel_all_keys_{suffix}.jsonl"
                self.converter.create_multi_label_dataset(
                    csv_path=str(request.input_path),
                    output_path=str(multilabel_path),
                    text_column=request.text_column,
                    annotation_column=request.annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=request.label_strategy,
                    id_column=request.id_column,
                    lang_column=request.lang_column,
                )

                metadata.update({
                    "labels_detected": annotation_keys,
                    "num_keys": len(annotation_keys),
                    "training_approach": "hybrid" if multiclass_keys and onevsall_keys else ("multi-class" if multiclass_keys else "one-vs-all"),
                    "key_strategies": request.key_strategies,
                    "multiclass_keys": multiclass_keys,
                    "onevsall_keys": onevsall_keys,
                    "files_per_key": {k: str(v) for k, v in category_files.items()},
                })

                # Return bundle with training_files (mixed: some per key, some multilabel)
                return TrainingDataBundle(
                    primary_file=multilabel_path,
                    training_files={**category_files, "multilabel": multilabel_path},
                    strategy="multi-label",  # Use multi-label infrastructure
                    text_column="text",
                    label_column="labels",
                    metadata=metadata,
                )

            # CRITICAL FIX: Multi-class training with multiple keys (legacy: all keys same strategy)
            # When user selects "all keys" + "multi-class" → train ONE model PER KEY
            # Each key gets its own file with only its values
            elif len(annotation_keys) > 1:
                LOGGER.info(f"Multi-class mode with {len(annotation_keys)} keys - creating one file per key")

                # Create ONE file per key (not per value)
                category_files: Dict[str, Path] = {}

                for key in annotation_keys:
                    # Create single-label dataset for this key only
                    key_output_file = dataset_dir / f"multiclass_{self._slugify(key)}_{suffix}.jsonl"
                    key_path = self.converter.create_single_key_dataset(
                        csv_path=str(request.input_path),
                        output_path=str(key_output_file),
                        text_column=request.text_column,
                        annotation_column=request.annotation_column,
                        annotation_key=key,  # Only this key
                        label_strategy=request.label_strategy,
                        id_column=request.id_column,
                        lang_column=request.lang_column,
                    )

                    if key_path:
                        category_files[key] = Path(key_path)
                        LOGGER.info(f"Created file for key '{key}': {key_output_file}")

                if not category_files:
                    raise RuntimeError("Failed to create any key-specific datasets")

                # Create a consolidated multi-label file for compatibility (optional)
                multilabel_path = dataset_dir / f"multilabel_all_keys_{suffix}.jsonl"
                self.converter.create_multi_label_dataset(
                    csv_path=str(request.input_path),
                    output_path=str(multilabel_path),
                    text_column=request.text_column,
                    annotation_column=request.annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=request.label_strategy,
                    id_column=request.id_column,
                    lang_column=request.lang_column,
                )

                metadata.update({
                    "labels_detected": annotation_keys,
                    "num_keys": len(annotation_keys),
                    "training_approach": "multi-class",
                    "files_per_key": {k: str(v) for k, v in category_files.items()},
                })

                # Return bundle with training_files (one per key) and primary multilabel file
                return TrainingDataBundle(
                    primary_file=multilabel_path,
                    training_files={**category_files, "multilabel": multilabel_path},
                    strategy="multi-label",  # Use multi-label infrastructure with multiclass_groups
                    text_column="text",
                    label_column="labels",
                    metadata=metadata,
                )

            # Single key: create one consolidated file with all values of that key
            output_file = dataset_dir / f"singlelabel_dataset_{suffix}.jsonl"
            path = self.converter.create_multi_label_dataset(
                csv_path=str(request.input_path),
                output_path=str(output_file),
                text_column=request.text_column,
                annotation_column=request.annotation_column,
                annotation_keys=annotation_keys or None,
                label_strategy=request.label_strategy,
                id_column=request.id_column,
                lang_column=request.lang_column,
            )

            if not path:
                raise RuntimeError("Failed to create single-label dataset from annotations")

            metadata.update({
                "labels_detected": annotation_keys,
                "output_file": str(output_file),
            })

            # CRITICAL: Use multi-label strategy because create_multi_label_dataset produces
            # labels as lists (e.g., ["sentiment_neutral"]), which requires multi-label code path
            return TrainingDataBundle(
                primary_file=Path(path),
                strategy="multi-label",  # FIXED: was "single-label" but data format is multi-label (labels in lists)
                text_column="text",
                label_column="labels",
                metadata=metadata,
            )

        # Multi-label mode (one-vs-all): create ONE consolidated file with all labels
        # This will be used with MultiLabelTrainer which trains one model per VALUE
        output_file = dataset_dir / f"multilabel_dataset_{suffix}.jsonl"
        path = self.converter.create_multi_label_dataset(
            csv_path=str(request.input_path),
            output_path=str(output_file),
            text_column=request.text_column,
            annotation_column=request.annotation_column,
            annotation_keys=annotation_keys or None,
            label_strategy=request.label_strategy,
            id_column=request.id_column,
            lang_column=request.lang_column,
        )

        if not path:
            raise RuntimeError("Failed to create multi-label dataset from annotations")

        metadata.update({
            "labels_detected": annotation_keys,
            "output_file": str(output_file),
        })

        # Return bundle with primary_file but NO training_files
        # This will trigger MultiLabelTrainer path instead of multiple files path
        return TrainingDataBundle(
            primary_file=Path(path),
            strategy="multi-label",
            text_column="text",
            label_column="labels",  # MultiLabelTrainer expects 'labels' column
            metadata=metadata,
        )

    def _build_category_csv(self, request: TrainingDataRequest) -> TrainingDataBundle:
        dataset_dir = self._ensure_output_dir(request.output_dir)
        df = pd.read_csv(request.input_path)

        if request.text_column not in df.columns:
            raise ValueError(f"Text column '{request.text_column}' not present in {request.input_path}")
        if request.label_column not in df.columns:
            raise ValueError(f"Label column '{request.label_column}' not present in {request.input_path}")

        # Check if multi-label mode (one-vs-all) was requested
        if request.mode == "multi-label":
            # Convert to one-vs-all format: one binary file per category
            return self._build_category_csv_one_vs_all(request, df, dataset_dir)

        # CRITICAL FIX: Include language column if it exists
        columns_to_copy = [request.text_column, request.label_column]
        rename_mapping = {request.text_column: "text", request.label_column: "label"}

        if request.lang_column and request.lang_column in df.columns:
            columns_to_copy.append(request.lang_column)
            rename_mapping[request.lang_column] = "language"

        out_df = df[columns_to_copy].rename(columns=rename_mapping)

        output_path = dataset_dir / f"single_label_{_timestamp()}.csv"
        out_df.to_csv(output_path, index=False)

        metadata = {
            "label_distribution": out_df["label"].value_counts().to_dict(),
            "num_samples": len(out_df),
        }

        return TrainingDataBundle(
            primary_file=output_path,
            strategy="single-label",
            text_column="text",
            label_column="label",
            metadata=metadata,
        )

    def _build_category_csv_one_vs_all(self, request: TrainingDataRequest, df: pd.DataFrame, dataset_dir: Path) -> TrainingDataBundle:
        """Convert category CSV to one-vs-all format (one binary file per category)"""
        timestamp = _timestamp()
        category_files: Dict[str, Path] = {}
        positive_mapping: Dict[str, List[str]] = {}

        # Get all unique categories
        # CRITICAL: Convert to Python list to avoid numpy type issues
        unique_categories = df[request.label_column].unique().tolist()

        # Resolve language information once for the entire dataframe
        language_series = self._resolve_language_series(df, request)

        # Prepare columns for binary files
        columns_to_include = [request.text_column]
        if request.lang_column and request.lang_column in df.columns:
            columns_to_include.append(request.lang_column)

        # Create one binary file per category
        for category in unique_categories:
            # Create binary label: 1 if this category, 0 otherwise
            binary_df = df[columns_to_include].copy()
            binary_df['label'] = (df[request.label_column] == category).astype(int)

            # Rename columns
            rename_map = {request.text_column: "text"}
            if request.lang_column and request.lang_column in df.columns:
                rename_map[request.lang_column] = "language"
            binary_df = binary_df.rename(columns=rename_map)

            if language_series.notna().any():
                binary_df["language"] = language_series

            # Save binary file
            category_slug = self._slugify(str(category))
            output_path = dataset_dir / f"category_{category_slug}_{timestamp}.csv"
            binary_df.to_csv(output_path, index=False)
            category_files[str(category)] = output_path

            # Track positive examples
            positives = binary_df[binary_df["label"] == 1]["text"].tolist()
            positive_mapping[str(category)] = positives

        # Also create multilabel JSONL format for compatibility
        multilabel_records = []
        for _, row in df.iterrows():
            record = {
                "text": str(row[request.text_column]),
                "labels": {str(cat): 1 if row[request.label_column] == cat else 0 for cat in unique_categories}
            }
            if request.id_column and request.id_column in df.columns:
                record["id"] = str(row[request.id_column])
            lang_value = language_series.loc[row.name] if row.name in language_series.index else None
            if lang_value:
                record["lang"] = lang_value
            multilabel_records.append(record)

        multilabel_path = dataset_dir / f"multilabel_one_vs_all_{timestamp}.jsonl"
        with multilabel_path.open("w", encoding="utf-8") as fh:
            for record in multilabel_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        metadata = {
            "num_categories": len(category_files),
            "categories": list(category_files.keys()),
            "positive_examples": {k: len(v) for k, v in positive_mapping.items()},
            "num_records": len(multilabel_records),
            "training_approach": "one-vs-all",
        }

        training_files = {**category_files, "multilabel": multilabel_path}

        return TrainingDataBundle(
            primary_file=multilabel_path,
            training_files=training_files,
            strategy="multi-label",  # Use multi-label strategy for distributed training
            text_column="text",
            label_column="label",  # Individual CSV files use "label" column
            metadata=metadata,
        )

    def _build_binary_long(self, request: TrainingDataRequest) -> TrainingDataBundle:
        if not request.category_column or not request.value_column:
            raise ValueError("'binary_long_csv' requires category_column and value_column")

        dataset_dir = self._ensure_output_dir(request.output_dir)
        df = pd.read_csv(request.input_path)

        for col in (request.text_column, request.category_column, request.value_column):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not present in {request.input_path}")

        if request.id_column and request.id_column not in df.columns:
            raise ValueError(f"ID column '{request.id_column}' not present in {request.input_path}")

        positive_mapping: Dict[str, List[str]] = {}
        category_files: Dict[str, Path] = {}
        timestamp = _timestamp()
        language_series = self._resolve_language_series(df, request)

        for category, sub_df in df.groupby(request.category_column):
            binary_df = sub_df[[request.text_column, request.value_column]].rename(
                columns={request.text_column: "text", request.value_column: "label"}
            )
            if not language_series.empty:
                lang_slice = language_series.reindex(sub_df.index)
            else:
                lang_slice = pd.Series(index=sub_df.index, dtype="object")
            if (
                not lang_slice.empty
                and (lang_slice.notna().any() or "language" in binary_df.columns)
            ):
                binary_df["language"] = lang_slice

            output_path = dataset_dir / f"category_{self._slugify(category)}_{timestamp}.csv"
            binary_df.to_csv(output_path, index=False)
            category_files[str(category)] = output_path

            positives = binary_df[binary_df["label"] == 1]["text"].tolist()
            positive_mapping[str(category)] = positives

        multilabel_records = self._build_multilabel_records_from_long(
            df,
            text_column=request.text_column,
            category_column=request.category_column,
            value_column=request.value_column,
            id_column=request.id_column,
            lang_column=request.lang_column,
            language_series=language_series,
        )

        multilabel_path = dataset_dir / f"multilabel_from_long_{timestamp}.jsonl"
        with multilabel_path.open("w", encoding="utf-8") as fh:
            for record in multilabel_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        metadata = {
            "num_categories": len(category_files),
            "categories": list(category_files.keys()),
            "positive_examples": {k: len(v) for k, v in positive_mapping.items()},
            "num_records": len(multilabel_records),
        }

        training_files = {**category_files, "multilabel": multilabel_path}

        return TrainingDataBundle(
            primary_file=multilabel_path,
            training_files=training_files,
            strategy="multi-label",
            text_column="text",
            label_column="label",
            metadata=metadata,
        )

    def _build_jsonl_single(self, request: TrainingDataRequest) -> TrainingDataBundle:
        dataset_dir = self._ensure_output_dir(request.output_dir)
        df = pd.read_json(request.input_path, lines=True)

        for col in (request.text_column, request.label_column):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not present in {request.input_path}")

        # CRITICAL FIX: Include language column if it exists
        columns_to_copy = [request.text_column, request.label_column]
        rename_mapping = {request.text_column: "text", request.label_column: "label"}

        if request.lang_column and request.lang_column in df.columns:
            columns_to_copy.append(request.lang_column)
            rename_mapping[request.lang_column] = "language"

        clean_df = df[columns_to_copy].rename(columns=rename_mapping)
        language_series = self._resolve_language_series(df, request)
        if not language_series.empty:
            clean_df["language"] = language_series

        output_path = dataset_dir / f"single_label_{_timestamp()}.jsonl"
        clean_df.to_json(output_path, orient="records", lines=True, force_ascii=False)

        metadata = {
            "label_distribution": clean_df["label"].value_counts().to_dict(),
            "num_samples": len(clean_df),
            "original_format": "jsonl",
        }

        return TrainingDataBundle(
            primary_file=output_path,
            strategy="single-label",
            text_column="text",
            label_column="label",
            metadata=metadata,
        )

    def _build_jsonl_multi(self, request: TrainingDataRequest) -> TrainingDataBundle:
        dataset_dir = self._ensure_output_dir(request.output_dir)
        records = self._load_jsonl(request.input_path)
        if not records:
            raise RuntimeError("No records in JSONL file")

        label_field_candidates = [request.label_column, "labels", "label_ids", "targets"]
        label_field = next((field for field in label_field_candidates if field in records[0]), None)
        if not label_field:
            raise ValueError("Could not detect label field in JSONL multi-label data")

        category_values: Dict[str, List[Dict[str, Any]]] = {}
        multilabel_records: List[Dict[str, Any]] = []
        text_language_cache: Dict[str, str] = {}

        for item in records:
            text = item.get(request.text_column)
            if not text:
                continue

            labels = item.get(label_field)
            labels = self._coerce_labels_container(labels)
            if labels is None:
                continue

            if isinstance(labels, dict):
                active_labels = [k for k, v in labels.items() if v]
            elif isinstance(labels, (list, tuple, set)):
                active_labels = [str(v).strip() for v in labels if str(v).strip()]
            else:
                active_labels = [str(labels).strip()]

            if not active_labels:
                continue

            record_metadata = {
                "text": text,
                "labels": active_labels,
            }

            for key in (request.id_column, request.lang_column):
                if key and item.get(key) is not None:
                    record_metadata[key] = item[key]

            lang_value = self._resolve_language_from_mapping(item, request)
            if lang_value:
                record_metadata["lang"] = lang_value
                text_language_cache[text] = lang_value

            multilabel_records.append(record_metadata)

            for label in active_labels:
                category_values.setdefault(label, []).append({
                    "text": text,
                    "label": 1,
                })

        if not multilabel_records:
            raise RuntimeError("JSONL file did not yield any labelled samples")

        timestamp = _timestamp()
        multilabel_path = dataset_dir / f"multilabel_jsonl_{timestamp}.jsonl"
        with multilabel_path.open("w", encoding="utf-8") as fh:
            for record in multilabel_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        training_files: Dict[str, Path] = {"multilabel": multilabel_path}

        for category, positives in category_values.items():
            binary_records = []
            positive_texts = {entry["text"] for entry in positives}
            for item in records:
                text = item.get(request.text_column)
                if not text:
                    continue
                record = {
                    "text": text,
                    "label": 1 if text in positive_texts else 0,
                }
                lang_value = text_language_cache.get(text)
                if lang_value:
                    record["language"] = lang_value
                binary_records.append(record)

            output_path = dataset_dir / f"category_{self._slugify(category)}_{timestamp}.jsonl"
            with output_path.open("w", encoding="utf-8") as fh:
                for record in binary_records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            training_files[str(category)] = output_path

        metadata = {
            "num_categories": len(category_values),
            "categories": list(category_values.keys()),
            "num_records": len(multilabel_records),
        }

        return TrainingDataBundle(
            primary_file=multilabel_path,
            training_files=training_files,
            strategy="multi-label",
            text_column="text",
            label_column="label",
            metadata=metadata,
        )

    def _build_prepared(self, request: TrainingDataRequest) -> TrainingDataBundle:
        """Wrap an already prepared dataset without modification."""

        if not request.input_path.exists():
            raise FileNotFoundError(request.input_path)

        metadata = {
            "note": "Dataset reused as-is",
            "format": request.format,
        }

        return TrainingDataBundle(
            primary_file=request.input_path,
            strategy=request.mode or "single-label",
            text_column=request.text_column,
            label_column=request.label_column,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_output_dir(self, desired: Optional[Path]) -> Path:
        target = self.base_output_dir if desired is None else Path(desired)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # ------------------------------------------------------------------
    # Language helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_language_value(value: Any) -> Optional[str]:
        """Normalize raw language values to stable uppercase codes."""
        if value is None:
            return None

        if isinstance(value, float) and pd.isna(value):
            return None

        value_str = str(value).strip()
        if not value_str:
            return None

        lowered = value_str.lower()
        if lowered in {"unknown", "unk", "n/a", "na", "none", "null"}:
            return None

        normalized = LanguageNormalizer.normalize_language(value_str)
        if normalized:
            return normalized.upper()

        cleaned = value_str.replace("_", "-")
        if "-" in cleaned:
            head = cleaned.split("-", 1)[0].strip()
            if head:
                return head.upper()

        if len(cleaned) <= 5:
            return cleaned.upper()

        return cleaned.upper()

    @staticmethod
    def _coerce_labels_container(raw_labels: Any) -> Any:
        """
        Normalize label containers that may arrive as JSON strings.

        Accepts lists, dicts, scalars, or stringified representations of lists/dicts.
        """
        if isinstance(raw_labels, str):
            stripped = raw_labels.strip()
            if not stripped:
                return []

            if stripped[0] in ("[", "{"):
                parsed = None
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(stripped)
                        break
                    except Exception:
                        parsed = None

                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
                if isinstance(parsed, dict):
                    return parsed

            return [stripped]

        return raw_labels

    def _resolve_language_from_mapping(
        self,
        mapping: Any,
        request: "TrainingDataRequest",
        extra_candidates: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Extract a normalized language code from a pandas Series or dict."""
        candidates: List[Any] = []

        def _get(key: str) -> Any:
            if hasattr(mapping, "get"):
                return mapping.get(key)
            try:
                return mapping[key]
            except (KeyError, IndexError, TypeError):
                return None

        if request.lang_column:
            candidates.append(_get(request.lang_column))

        fallback_keys = [
            "language",
            "lang",
            "detected_language",
            "detected_lang",
            "language_code",
            "lang_code",
        ]
        if extra_candidates:
            fallback_keys = list(dict.fromkeys(list(fallback_keys) + list(extra_candidates)))

        for key in fallback_keys:
            value = _get(key)
            if value is not None:
                candidates.append(value)

        for candidate in candidates:
            normalized = self._normalize_language_value(candidate)
            if normalized:
                return normalized

        return None

    def _resolve_language_series(self, df: pd.DataFrame, request: "TrainingDataRequest") -> pd.Series:
        """Return a Series of normalized language codes aligned with df."""
        if df.empty:
            return pd.Series(dtype="object", index=df.index)

        return df.apply(lambda row: self._resolve_language_from_mapping(row, request), axis=1)

    def _build_multilabel_records_from_long(
        self,
        df: pd.DataFrame,
        *,
        text_column: str,
        category_column: str,
        value_column: str,
        id_column: Optional[str],
        lang_column: Optional[str],
        language_series: Optional[pd.Series] = None,
    ) -> List[Dict[str, Any]]:
        records: Dict[Any, Dict[str, Any]] = {}

        def _record_key(row: pd.Series) -> Any:
            if id_column and pd.notna(row.get(id_column)):
                return row[id_column]
            return row[text_column]

        for _, row in df.iterrows():
            key = _record_key(row)
            entry = records.setdefault(key, {
                "text": row[text_column],
                "labels": [],
            })

            lang_value = None
            if language_series is not None and row.name in language_series.index:
                lang_value = language_series.loc[row.name]
            if not lang_value and lang_column and pd.notna(row.get(lang_column)):
                lang_value = self._normalize_language_value(row[lang_column])

            if lang_value:
                entry.setdefault("lang", lang_value)

            if row[value_column] in (1, "1", True):
                entry["labels"].append(str(row[category_column]))

        # Remove duplicates and empty label entries
        filtered: List[Dict[str, Any]] = []
        for record in records.values():
            record["labels"] = sorted(set(record["labels"]))
            if record["labels"]:
                filtered.append(record)

        return filtered

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "category"

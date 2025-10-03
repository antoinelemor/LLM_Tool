"""Utilities for building training-ready datasets from heterogeneous sources.

This module centralises the logic that was previously scattered across the
AugmentedSocialScientist toolkit so the Training Studio can ingest the same
range of formats (LLM JSON annotations, category CSVs, binary long tables,
JSONL with metadata, …) and produce normalised assets that plug directly into
`ModelTrainer` or `MultiLabelTrainer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import logging
from datetime import datetime

import pandas as pd

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

    def __init__(self, base_output_dir: Path):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
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
            # Single-label: create one consolidated file with one label per text
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

            return TrainingDataBundle(
                primary_file=Path(path),
                strategy="single-label",
                text_column="text",
                label_column="labels",
                metadata=metadata,
            )

        # Multi-label mode: create ONE consolidated file with all labels
        # This will be used with MultiLabelTrainer which trains one model per key automatically
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

        out_df = df[[request.text_column, request.label_column]].rename(
            columns={request.text_column: "text", request.label_column: "label"}
        )

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

        for category, sub_df in df.groupby(request.category_column):
            binary_df = sub_df[[request.text_column, request.value_column]].rename(
                columns={request.text_column: "text", request.value_column: "label"}
            )
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

        clean_df = df[[request.text_column, request.label_column]].rename(
            columns={request.text_column: "text", request.label_column: "label"}
        )

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

        for item in records:
            text = item.get(request.text_column)
            if not text:
                continue

            labels = item.get(label_field)
            if labels is None:
                continue

            if isinstance(labels, dict):
                active_labels = [k for k, v in labels.items() if v]
            elif isinstance(labels, list):
                active_labels = [str(v) for v in labels]
            else:
                active_labels = [str(labels)]

            if not active_labels:
                continue

            record_metadata = {
                "text": text,
                "labels": active_labels,
            }

            for key in (request.id_column, request.lang_column):
                if key and item.get(key) is not None:
                    record_metadata[key] = item[key]

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
                binary_records.append({
                    "text": text,
                    "label": 1 if text in positive_texts else 0,
                })

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

    def _build_multilabel_records_from_long(
        self,
        df: pd.DataFrame,
        *,
        text_column: str,
        category_column: str,
        value_column: str,
        id_column: Optional[str],
        lang_column: Optional[str],
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

            if lang_column and pd.notna(row.get(lang_column)):
                entry.setdefault("lang", row[lang_column])

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


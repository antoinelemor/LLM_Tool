#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
data_detector.py

MAIN OBJECTIVE:
---------------
Detect and analyse candidate datasets for annotation and training workflows,
providing structural metadata, column heuristics, and readiness diagnostics.

Dependencies:
-------------
- json
- logging
- re
- collections
- dataclasses
- pathlib
- typing
- pandas
- pyreadr

MAIN FEATURES:
--------------
1) Recursively scan directories to surface supported dataset formats
2) Inspect files with pandas to infer schema, label columns, and sizes
3) Estimate textual content, annotation structure, and JSON-rich fields
4) Score potential language, label, and annotation columns for the CLI
5) Produce detailed analysis payloads reused by CLI wizards and reports

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PANDAS = False
    pd = None  # type: ignore


@dataclass
class DatasetInfo:
    """Lightweight description of a detected dataset."""

    path: Path
    format: str
    rows: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    size_mb: Optional[float] = None
    detected_language: Optional[str] = None
    has_labels: bool = False
    label_column: Optional[str] = None
    column_types: Dict[str, str] = field(default_factory=dict)
    text_scores: Dict[str, float] = field(default_factory=dict)


class DataDetector:
    """Auto-detect and analyse available datasets."""

    @staticmethod
    def scan_directory(directory: Path = Path.cwd()) -> List[DatasetInfo]:
        """Scan a directory (recursively) for supported datasets."""

        supported_extensions = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet", ".rdata"}
        datasets: List[DatasetInfo] = []

        if not directory.exists():
            logger.debug("Directory %s does not exist", directory)
            return datasets

        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in supported_extensions:
                continue

            info = DataDetector.analyze_file(path)
            if info:
                datasets.append(info)

        datasets.sort(key=lambda d: d.path.name)
        return datasets

    @staticmethod
    def analyze_file(file_path: Path) -> Optional[DatasetInfo]:
        """Analyse a dataset at the given path (basic info)."""

        if not file_path.exists():
            return None

        info = DatasetInfo(
            path=file_path,
            format=file_path.suffix[1:],
            size_mb=file_path.stat().st_size / (1024 * 1024),
        )

        if not HAS_PANDAS:
            return info

        try:
            if info.format == "csv":
                df = pd.read_csv(file_path, nrows=100)  # type: ignore[arg-type]
            elif info.format == "json":
                df = pd.read_json(file_path, lines=False, nrows=100)  # type: ignore[arg-type]
            elif info.format == "jsonl":
                df = pd.read_json(file_path, lines=True, nrows=100)  # type: ignore[arg-type]
            elif info.format in ["xlsx", "xls"]:
                df = pd.read_excel(file_path, nrows=100)  # type: ignore[arg-type]
            elif info.format == "parquet":
                df = pd.read_parquet(file_path).head(100)  # type: ignore[arg-type]
            elif info.format.lower() in ["rdata"]:
                try:
                    import pyreadr  # type: ignore

                    result = pyreadr.read_r(str(file_path))
                    if result:
                        df = list(result.values())[0].head(100)
                    else:
                        return info
                except ImportError:  # pragma: no cover - optional dependency
                    return info
            else:
                return info
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not analyse dataset %s: %s", file_path, exc)
            return info

        info.rows = len(df)
        info.columns = list(df.columns)
        info.column_types = {col: str(df[col].dtype) for col in df.columns}
        info.text_scores = {}

        for col in df.columns:
            if (
                pd.api.types.is_string_dtype(df[col])  # type: ignore[attr-defined]
                or str(df[col].dtype) == "object"
            ):
                sample_series = df[col].dropna().astype(str)
                if not sample_series.empty:
                    avg_len = float(sample_series.str.len().mean())
                    info.text_scores[col] = avg_len

        label_candidates = ["label", "labels", "class", "category", "target", "y"]
        for col in info.columns:
            if col.lower() in label_candidates:
                info.has_labels = True
                info.label_column = col
                break

        return info

    @staticmethod
    def analyze_file_intelligently(file_path: Path) -> Dict[str, Any]:
        """Perform a richer analysis on a dataset."""

        result: Dict[str, Any] = {
            "path": file_path,
            "format": file_path.suffix[1:],
            "issues": [],
            "text_column_candidates": [],
            "label_column_candidates": [],
            "annotation_column_candidates": [],
            "annotation_stats": {},
            "language_column_candidates": [],
            "numeric_columns": [],
            "datetime_columns": [],
            "json_columns": [],
            "all_columns": [],
        }

        if not HAS_PANDAS:
            result["issues"].append("pandas is not installed")
            return result

        try:
            df = DataDetector._load_dataframe(file_path, result["format"])
        except Exception as exc:  # pragma: no cover - defensive
            result["issues"].append(f"Analysis error: {exc}")
            return result

        result["rows"] = len(df)
        result["all_columns"] = list(df.columns)

        for col in df.columns:
            dtype = str(df[col].dtype)
            column_info: Dict[str, Any] = {
                "name": col,
                "dtype": dtype,
                "sample_values": df[col].dropna().head(5).tolist(),
            }

            if pd.api.types.is_numeric_dtype(df[col]):  # type: ignore[attr-defined]
                result["numeric_columns"].append(column_info)

            if pd.api.types.is_datetime64_any_dtype(df[col]):  # type: ignore[attr-defined]
                result["datetime_columns"].append(column_info)

            text_score = DataDetector._estimate_text_score(df[col])
            if text_score is not None:
                column_info["text_score"] = text_score
            text_metrics = DataDetector._compute_text_metrics(df[col])
            if text_metrics:
                column_info.update(text_metrics)
                if text_metrics.get("text_confidence", 0.0) >= 0.15:
                    result["text_column_candidates"].append(column_info)

            is_json = DataDetector._looks_like_json_column(df[col])
            if is_json:
                column_info["is_json"] = True
                column_info["json_fraction"] = DataDetector._json_like_fraction(
                    [str(val) for val in df[col].dropna().astype(str).head(50).tolist()]
                )
                result["json_columns"].append(column_info)

            if DataDetector._looks_like_label_column(col, df[col]):
                result["label_column_candidates"].append(column_info)

            if DataDetector._looks_like_language_column(col, df[col]):
                result["language_column_candidates"].append(column_info)

        result["text_column_candidates"].sort(
            key=lambda info: (
                info.get("text_confidence", 0.0),
                info.get("avg_length", 0.0),
                -(info.get("json_like_ratio", 0.0) or 0.0),
            ),
            reverse=True,
        )

        annotation_candidates, annotation_stats = DataDetector._analyze_annotation_columns(df)
        result["annotation_column_candidates"] = annotation_candidates
        result["annotation_stats"] = annotation_stats

        return result

    @staticmethod
    def _load_dataframe(file_path: Path, file_format: str):
        """Helper to load a dataframe with sensible defaults."""

        if file_format == "csv":
            return pd.read_csv(file_path, nrows=1000)  # type: ignore[arg-type]
        if file_format == "json":
            return pd.read_json(file_path, lines=False)  # type: ignore[arg-type]
        if file_format == "jsonl":
            return pd.read_json(file_path, lines=True, nrows=1000)  # type: ignore[arg-type]
        if file_format in ["xlsx", "xls"]:
            return pd.read_excel(file_path, nrows=1000)  # type: ignore[arg-type]
        if file_format == "parquet":
            return pd.read_parquet(file_path)  # type: ignore[arg-type]
        if file_format.lower() in ["rdata"]:
            try:
                import pyreadr  # type: ignore

                result = pyreadr.read_r(str(file_path))
                if result:
                    return list(result.values())[0]
                return pd.DataFrame()
            except ImportError:  # pragma: no cover - optional dependency
                return pd.DataFrame()

        return pd.DataFrame()

    @staticmethod
    def _estimate_text_score(series) -> Optional[float]:
        """Estimate how 'textual' a column looks."""

        try:
            sample = series.dropna().astype(str)
            if sample.empty:
                return None
            return float(sample.str.len().mean())
        except Exception:  # pragma: no cover - defensive
            return None

    @staticmethod
    def _looks_like_json_column(series) -> bool:
        """Heuristic: check if column contains JSON strings."""

        sample_values = series.dropna().astype(str).head(20)
        json_count = 0
        for val in sample_values:
            val = val.strip()
            if not val:
                continue
            if (val.startswith("{") and val.endswith("}")) or (
                val.startswith("[") and val.endswith("]")
            ):
                try:
                    json.loads(val)
                    json_count += 1
                except Exception:
                    continue
        return json_count >= len(sample_values) * 0.7 and len(sample_values) > 0

    @staticmethod
    def _json_like_fraction(values: List[str]) -> float:
        """Return the fraction of values that look like valid JSON payloads."""

        considered = 0
        matches = 0
        for raw in values:
            value = str(raw).strip()
            if not value:
                continue
            if (value.startswith("{") and value.endswith("}")) or (
                value.startswith("[") and value.endswith("]")
            ):
                considered += 1
                try:
                    json.loads(value)
                    matches += 1
                except Exception:
                    continue
        if considered == 0:
            return 0.0
        return matches / considered

    @staticmethod
    def _looks_like_label_column(column_name: str, series) -> bool:
        """Detect columns that are likely to contain labels."""

        name_lower = column_name.lower()
        label_keywords = [
            "label",
            "labels",
            "category",
            "categories",
            "class",
            "classes",
            "tag",
            "tags",
            "target",
        ]
        if any(keyword in name_lower for keyword in label_keywords):
            return True

        sample_values = series.dropna().head(20).tolist()
        structured_count = 0
        for val in sample_values:
            if isinstance(val, (list, dict)):
                structured_count += 1
                continue
            if isinstance(val, str):
                stripped = val.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    structured_count += 1
                elif stripped.startswith("{") and stripped.endswith("}"):
                    structured_count += 1
        return structured_count >= len(sample_values) * 0.5 and len(sample_values) > 0

    @staticmethod
    def _looks_like_language_column(column_name: str, series) -> bool:
        """Detect language columns (e.g., 'lang', 'language')."""

        name_lower = column_name.lower()
        if name_lower in {"language", "languages", "lang", "locale"}:
            return True
        if name_lower.endswith("_lang") or name_lower.endswith("_language"):
            return True

        sample_values = series.dropna().head(20).astype(str).tolist()
        short_values = [val for val in sample_values if 1 <= len(val.strip()) <= 5]
        return len(short_values) >= len(sample_values) * 0.7 and len(sample_values) > 0

    @staticmethod
    def _compute_text_metrics(series) -> Optional[Dict[str, float]]:
        """Compute rich heuristics to estimate how textual a column is."""

        if not HAS_PANDAS:
            return None

        try:
            non_null = series.dropna()
        except Exception:  # pragma: no cover - defensive
            return None

        if non_null.empty:
            return None

        # Skip obvious numeric/datetime columns early to avoid misclassification.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):  # type: ignore[attr-defined]
            return None

        sample = non_null.astype(str).head(200)
        if sample.empty:
            return None

        lengths = sample.str.len()
        avg_length = float(lengths.mean())
        median_length = float(lengths.median())
        alpha_ratio = float(sample.str.contains(r"[A-Za-z]", regex=True, na=False).mean())
        whitespace_ratio = float(sample.str.contains(r"\s", regex=True, na=False).mean())
        sentence_punct_ratio = float(
            sample.str.contains(r"[\.!?]", regex=True, na=False).mean()
        )
        digit_only_ratio = float(
            sample.str.fullmatch(r"[+-]?\d+(?:[\.,]\d+)?", na=False).mean()
        )
        json_like_ratio = DataDetector._json_like_fraction(sample.tolist())
        multi_token_ratio = float(
            sample.str.contains(r"\w+\s+\w+", regex=True, na=False).mean()
        )

        length_score = min(avg_length / 80.0, 1.0)
        richness_score = max(alpha_ratio, sentence_punct_ratio)
        multiword_score = max(whitespace_ratio, multi_token_ratio)
        numeric_penalty = digit_only_ratio
        json_penalty = json_like_ratio

        text_confidence = (
            0.35 * length_score
            + 0.25 * richness_score
            + 0.2 * multiword_score
            + 0.2 * (1.0 - numeric_penalty)
        )
        text_confidence -= 0.25 * json_penalty
        text_confidence = max(0.0, min(1.0, text_confidence))

        return {
            "avg_length": avg_length,
            "median_length": median_length,
            "alpha_ratio": alpha_ratio,
            "whitespace_ratio": whitespace_ratio,
            "digit_ratio": digit_only_ratio,
            "json_like_ratio": json_like_ratio,
            "multi_token_ratio": multi_token_ratio,
            "text_confidence": text_confidence,
        }

    @staticmethod
    def detect_id_column_candidates(
        df,
        text_column: Optional[str] = None,
        max_candidates: int = 10,
    ) -> List[Dict[str, Any]]:
        """Identify potential ID columns in a dataframe."""

        if not HAS_PANDAS or df is None:
            return []

        total_rows = max(len(df), 1)
        single_column_candidates: List[Dict[str, Any]] = []

        for col in df.columns:
            if text_column and col == text_column:
                continue

            series = df[col]
            if series.dropna().empty:
                continue

            dtype = str(series.dtype)
            try:
                is_numeric = pd.api.types.is_numeric_dtype(series)  # type: ignore[attr-defined]
                is_string = pd.api.types.is_string_dtype(series)  # type: ignore[attr-defined]
            except Exception:
                is_numeric = False
                is_string = False

            # Skip complex objects that pandas stores as "object" but are not strings
            if dtype == "object" and not is_string:
                continue

            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / total_rows if total_rows else 0.0
            looks_like_id = DataDetector._looks_like_identifier(col)

            keep_column = (
                unique_ratio >= 0.6
                or looks_like_id
                or bool(is_numeric)
            )

            if not keep_column:
                continue

            single_column_candidates.append(
                {
                    "name": col,
                    "dtype": dtype,
                    "unique_ratio": unique_ratio,
                    "looks_like_id": looks_like_id,
                    "is_combo": False,
                    "columns": (col,),
                    "return_value": col,
                }
            )

        # Limit base candidates to keep table readable while preserving highest scores and heuristics
        single_column_candidates.sort(
            key=lambda item: (
                item["unique_ratio"],
                1 if item["looks_like_id"] else 0,
                item["name"].lower(),
            ),
            reverse=True,
        )
        single_column_candidates = single_column_candidates[:max_candidates]

        # Generate combo candidates (pairs) to capture multi-column identifiers
        combo_candidates: List[Dict[str, Any]] = []
        if len(single_column_candidates) >= 2 and total_rows:
            # Prioritize columns that need help (low unique ratio) or look like IDs
            combo_source = sorted(
                single_column_candidates,
                key=lambda item: (
                    0 if item["unique_ratio"] >= 0.95 else 1,
                    0 if item["looks_like_id"] else 1,
                    item["unique_ratio"],
                ),
            )[: max(6, max_candidates)]

            for first, second in itertools.combinations(combo_source, 2):
                combo_cols = (first["columns"][0], second["columns"][0])

                try:
                    combo_frame = df[list(combo_cols)].copy()
                except KeyError:
                    continue

                if combo_frame.dropna(how="all").empty:
                    continue

                sample_combo_series = (
                    combo_frame.fillna("__nan__")
                    .astype(str)
                    .agg("||".join, axis=1)
                )
                combo_unique_ratio = sample_combo_series.nunique(dropna=True) / total_rows

                # Only surface combos that improve uniqueness or reach near-uniqueness
                if combo_unique_ratio < max(first["unique_ratio"], second["unique_ratio"], 0.6):
                    continue

                combo_candidates.append(
                    {
                        "name": " + ".join(combo_cols),
                        "dtype": f"combo ({len(combo_cols)})",
                        "unique_ratio": combo_unique_ratio,
                        "looks_like_id": first["looks_like_id"] or second["looks_like_id"],
                        "is_combo": True,
                        "columns": combo_cols,
                        "return_value": "+".join(combo_cols),
                    }
                )

        all_candidates = single_column_candidates + combo_candidates
        all_candidates.sort(
            key=lambda item: (
                item["unique_ratio"],
                1 if item.get("is_combo") else 0,
                1 if item.get("looks_like_id") else 0,
                item["name"].lower(),
            ),
            reverse=True,
        )
        return all_candidates

    @staticmethod
    def _compute_global_unique_ratios(
        data_path: Union[str, Path],
        column_sets: List[Tuple[str, ...]],
        max_unique_values: int = 750_000,
        chunksize: int = 50_000,
    ) -> Tuple[int, Dict[Tuple[str, ...], float]]:
        """Compute unique ratios for columns/combos across the full dataset."""

        if not HAS_PANDAS:
            return 0, {}

        path = Path(data_path)
        if not path.exists():
            return 0, {}

        suffix = path.suffix.lower()
        needed_columns = sorted({col for combo in column_sets for col in combo})

        unique_maps: Dict[Tuple[str, ...], set] = {combo: set() for combo in column_sets}
        total_rows = 0

        def _update_unique_sets(chunk):
            nonlocal total_rows
            total_rows += len(chunk)

            for combo in column_sets:
                target_set = unique_maps[combo]

                if len(combo) == 1:
                    col = combo[0]
                    if col not in chunk.columns:
                        continue
                    series = chunk[col].fillna("__nan__").astype(str)
                    target_set.update(series)
                else:
                    missing = [col for col in combo if col not in chunk.columns]
                    if missing:
                        continue
                    combo_series = chunk[list(combo)].fillna("__nan__").astype(str).agg("||".join, axis=1)
                    target_set.update(combo_series)

                if len(target_set) > max_unique_values:
                    # Prevent unbounded memory usage; signal by keeping current count and continue.
                    target_set.add("__overflow__")

        try:
            if suffix in {".csv", ".tsv", ".txt"}:
                read_kwargs = {"usecols": needed_columns, "chunksize": chunksize}
                if suffix == ".tsv":
                    read_kwargs["sep"] = "\t"
                reader = pd.read_csv(path, **read_kwargs)
                for chunk in reader:
                    _update_unique_sets(chunk)
            elif suffix in {".jsonl", ".ndjson"}:
                for chunk in pd.read_json(path, lines=True, chunksize=chunksize):
                    _update_unique_sets(chunk)
            elif suffix in {".parquet"}:
                chunk = pd.read_parquet(path, columns=needed_columns)
                _update_unique_sets(chunk)
            elif suffix in {".json"}:
                chunk = pd.read_json(path)
                _update_unique_sets(chunk)
            elif suffix in {".xlsx", ".xls"}:
                chunk = pd.read_excel(path, usecols=needed_columns)
                _update_unique_sets(chunk)
            else:
                # Fallback: attempt CSV parsing
                reader = pd.read_csv(path, usecols=needed_columns, chunksize=chunksize)
                for chunk in reader:
                    _update_unique_sets(chunk)
        except Exception as exc:
            logger.debug("Unable to compute global unique ratios for %s: %s", path, exc)
            return total_rows, {}

        if not total_rows:
            return 0, {}

        ratios: Dict[Tuple[str, ...], float] = {}
        for combo, values in unique_maps.items():
            if "__overflow__" in values:
                ratios[combo] = min(1.0, max_unique_values / total_rows)
            else:
                ratios[combo] = len(values) / total_rows
        return total_rows, ratios

    @staticmethod
    def _looks_like_identifier(column_name: str) -> bool:
        """Heuristic to decide if a column name looks like an identifier."""

        name_lower = column_name.lower()
        if name_lower in {"id", "identifier"}:
            return True
        if name_lower.endswith("_id") or name_lower.endswith("id"):
            return True
        return False

    @staticmethod
    def _analyze_annotation_columns(df) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Identify likely annotation/label columns and capture useful stats."""

        if not HAS_PANDAS:
            return [], {}

        candidates: List[Dict[str, Any]] = []
        stats: Dict[str, Dict[str, Any]] = {}
        total_rows = max(len(df), 1)

        keyword_patterns = [
            "annotation",
            "annotations",
            "label",
            "labels",
            "tag",
            "tags",
            "category",
            "categories",
            "theme",
            "themes",
            "topic",
            "topics",
            "intent",
            "intents",
            "entity",
            "entities",
            "classes",
            "classification",
        ]
        delimiter_regex = re.compile(r"[;,|/]")

        for col in df.columns:
            series = df[col]
            try:
                non_null = series.dropna()
            except Exception:  # pragma: no cover - defensive
                continue

            non_null_count = len(non_null)
            if non_null_count == 0:
                continue

            fill_rate = non_null_count / total_rows
            sample = non_null.head(200)
            sample_as_str = sample.astype(str)
            sample_size = len(sample)

            name_lower = col.lower()
            keyword_bonus = 1.0 if any(k in name_lower for k in keyword_patterns) else 0.0

            json_hits = 0
            structured_hits = 0
            list_hits = 0
            dict_hits = 0
            delimiter_hits = 0
            parsed_item_lengths: List[int] = []
            sample_labels: Counter[str] = Counter()

            for raw in sample:
                parsed, parsed_from_json = DataDetector._coerce_to_structured(raw)
                if parsed is not None:
                    structured_hits += 1
                    if isinstance(parsed, dict):
                        dict_hits += 1
                    elif isinstance(parsed, list):
                        list_hits += 1
                    if parsed_from_json:
                        json_hits += 1
                    extracted = DataDetector._extract_label_tokens(parsed, max_items=20)
                    if extracted:
                        parsed_item_lengths.append(len(extracted))
                        sample_labels.update(extracted)
                elif isinstance(raw, str):
                    value = raw.strip()
                    if not value:
                        continue
                    if delimiter_regex.search(value):
                        delimiter_hits += 1
                        parts = [part.strip() for part in delimiter_regex.split(value) if part.strip()]
                        if len(parts) >= 2:
                            parsed_item_lengths.append(len(parts))
                            sample_labels.update(parts[:20])
                    elif ":" in value and "," in value:
                        # Inline key:value style annotations
                        fragments = [frag.strip() for frag in value.split(",") if frag.strip()]
                        key_like = [frag.split(":", 1)[0].strip() for frag in fragments if ":" in frag]
                        if len(key_like) >= 2:
                            parsed_item_lengths.append(len(key_like))
                            sample_labels.update(key_like[:20])

            if sample_size == 0:
                continue

            structured_fraction = structured_hits / sample_size
            json_fraction = json_hits / sample_size
            list_fraction = list_hits / sample_size
            dict_fraction = dict_hits / sample_size
            delimiter_fraction = delimiter_hits / sample_size

            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / max(non_null_count, 1)

            avg_length = float(sample_as_str.str.len().mean())

            match_type: Optional[str] = None
            is_json_like = False

            if json_fraction >= 0.5 or dict_fraction >= 0.5:
                match_type = "json_content"
                is_json_like = True
            elif list_fraction >= 0.5:
                match_type = "list_of_labels"
                is_json_like = True
            elif delimiter_fraction >= 0.5:
                match_type = "delimited_labels"
            elif keyword_bonus and structured_fraction >= 0.2:
                match_type = "name_pattern"
            elif keyword_bonus and unique_ratio <= 0.6:
                match_type = "name_pattern"
            elif unique_ratio <= 0.3 and avg_length <= 64 and series.dtype == "object":
                match_type = "categorical"

            score = (
                0.5 * structured_fraction
                + 0.2 * keyword_bonus
                + 0.15 * delimiter_fraction
                + 0.1 * min(fill_rate, 1.0)
                + 0.05 * (1.0 - min(unique_ratio, 1.0))
            )

            if match_type == "categorical":
                score = max(score, 0.55)

            score = max(0.0, min(1.0, score))

            if not match_type and score < 0.45:
                continue

            if not sample_labels:
                # Extract a lightweight summary of categorical values if none collected yet.
                value_counts = sample_as_str.value_counts().head(10)
                sample_labels.update(
                    {str(k): int(v) for k, v in value_counts.items() if k and len(k) <= 64}
                )

            avg_items = float(sum(parsed_item_lengths) / len(parsed_item_lengths)) if parsed_item_lengths else 0.0
            stats[col] = {
                "fill_rate": fill_rate,
                "non_null_count": non_null_count,
                "is_json": is_json_like,
                "json_fraction": json_fraction,
                "structured_fraction": structured_fraction,
                "delimiter_fraction": delimiter_fraction,
                "list_fraction": list_fraction,
                "unique_ratio": unique_ratio,
                "avg_length": avg_length,
                "avg_items": avg_items,
                "top_labels": [label for label, _ in sample_labels.most_common(15)],
            }

            candidates.append(
                {
                    "name": col,
                    "score": score,
                    "match_type": match_type or ("keyword" if keyword_bonus else "content"),
                    "fill_rate": fill_rate,
                    "json_fraction": json_fraction,
                    "structured_fraction": structured_fraction,
                }
            )

        candidates.sort(
            key=lambda item: (
                item.get("score", 0.0),
                item.get("fill_rate", 0.0),
                item.get("structured_fraction", 0.0),
            ),
            reverse=True,
        )
        return candidates, stats

    @staticmethod
    def _coerce_to_structured(value: Any) -> Tuple[Optional[Any], bool]:
        """Try to coerce a value to a structured (list/dict) representation.

        Returns a tuple (parsed_value, parsed_from_json) where parsed_from_json indicates
        whether the structured value was obtained by JSON parsing a string.
        """

        if isinstance(value, dict):
            return value, False
        if isinstance(value, (list, tuple, set)):
            return list(value), False
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None, False
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    return json.loads(stripped), True
                except Exception:
                    return None, False
        return None, False

    @staticmethod
    def _extract_label_tokens(value: Any, max_items: int = 50) -> List[str]:
        """Extract string-like tokens from nested annotation structures."""

        labels: List[str] = []
        stack: List[Any] = [value]

        while stack and len(labels) < max_items:
            current = stack.pop()
            if isinstance(current, str):
                candidate = current.strip()
                if 0 < len(candidate) <= 80:
                    labels.append(candidate)
            elif isinstance(current, dict):
                stack.extend(list(current.values()))
                stack.extend(list(current.keys()))
            elif isinstance(current, (list, tuple, set)):
                stack.extend(list(current))
            elif isinstance(current, (int, float)):
                labels.append(str(current))

        return labels

    @staticmethod
    def display_and_select_id_column(
        console,
        df,
        text_column: Optional[str] = None,
        step_label: str = "Identifier Column Selection",
        data_path: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        """Interactive helper to pick an ID column using Rich."""

        if not HAS_PANDAS or console is None:
            return None

        from rich import box
        from rich.prompt import Prompt, Confirm
        from rich.table import Table

        console.print(f"\n[bold]{step_label}[/bold]")
        console.print()
        console.print("[bold cyan]📋 What is an ID column?[/bold cyan]")
        console.print("  • [green]Tracks each text[/green] through the annotation process")
        console.print("  • [green]Links results back[/green] to your original data")
        console.print("  • [green]Resumes annotation[/green] if interrupted")
        console.print("  • [yellow]Not mandatory[/yellow] - we can generate one automatically")
        console.print()

        candidates = DataDetector.detect_id_column_candidates(df, text_column)

        total_rows_full = 0
        if data_path and candidates:
            try:
                total_rows_full, ratios = DataDetector._compute_global_unique_ratios(
                    data_path,
                    [tuple(candidate["columns"]) for candidate in candidates],
                )
                for candidate in candidates:
                    candidate["global_unique_ratio"] = ratios.get(tuple(candidate["columns"]))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Unable to compute full unique ratios for %s: %s", data_path, exc)

        if candidates:
            id_table = Table(title="Candidate ID Columns", box=box.ROUNDED)
            id_table.add_column("#", style="cyan", width=4)
            id_table.add_column("Candidate", style="green", min_width=26)
            id_table.add_column("Type", style="yellow", width=14)
            id_table.add_column("Unique %", style="cyan", width=20, justify="right")
            id_table.add_column("Notes", style="magenta", overflow="fold")

            sample_rows = len(df)
            for idx, candidate in enumerate(candidates, 1):
                sample_ratio = candidate["unique_ratio"]
                global_ratio = candidate.get("global_unique_ratio")

                if global_ratio is not None:
                    ratio_display = f"{global_ratio * 100:.1f}%"
                    if abs(global_ratio - sample_ratio) > 0.05:
                        ratio_display += f" (sample {sample_ratio * 100:.1f}%)"
                    else:
                        ratio_display += " (full)"
                else:
                    ratio_display = f"{sample_ratio * 100:.1f}% (sample)"

                notes: List[str] = []
                if candidate.get("is_combo"):
                    notes.append("combined columns")
                if candidate.get("looks_like_id") and not candidate.get("is_combo"):
                    notes.append("id-like name")

                check_ratio = global_ratio if global_ratio is not None else sample_ratio
                if check_ratio >= 0.999:
                    notes.append("✓ unique")
                elif check_ratio >= 0.95:
                    notes.append("near-unique")
                else:
                    notes.append("⚠ duplicates remain")

                id_table.add_row(
                    str(idx),
                    candidate["name"],
                    candidate["dtype"],
                    ratio_display,
                    ", ".join(notes),
                )

            id_table.add_row("0", "[dim]None (auto-generate)[/dim]", "", "", "")
            if len(getattr(df, "columns", [])) >= 2:
                id_table.add_row("C", "[dim]Combine columns manually[/dim]", "", "", "")

            console.print(id_table)
            if total_rows_full and total_rows_full != sample_rows:
                console.print(
                    f"[dim]Sample analysed: {sample_rows:,} rows • Full dataset: {total_rows_full:,} rows[/dim]"
                )
            console.print()
        else:
            console.print("[yellow]ℹ️  No likely ID columns detected automatically[/yellow]")
            console.print("[dim]  → An 'llm_annotation_id' column can be created automatically[/dim]")

            df_columns = list(df.columns) if hasattr(df, "columns") else []
            if df_columns:
                preview_cols = ", ".join(map(str, df_columns[:10]))
                if len(df_columns) > 10:
                    preview_cols += ", ..."
                console.print(f"[dim]Available columns: {preview_cols}[/dim]")

                if Confirm.ask(
                    "[bold yellow]Would you like to select an ID column yourself?[/bold yellow]",
                    default=False,
                ):
                    console.print(
                        "[dim]Enter the column name (or press Enter to auto-generate IDs)[/dim]"
                    )
                    while True:
                        manual_choice = Prompt.ask(
                            "[bold yellow]ID column (leave blank to auto-generate)[/bold yellow]",
                            default="",
                        ).strip()

                        if not manual_choice:
                            console.print(
                                "[dim]✓ An 'llm_annotation_id' will be generated automatically[/dim]"
                            )
                            return None

                        if manual_choice in df_columns:
                            console.print(f"[green]✓ ID column: '{manual_choice}'[/green]")
                            return manual_choice

                        console.print(
                            f"[red]✗ Column '{manual_choice}' not found. Try again or press Enter to auto-generate.[/red]"
                        )

            console.print(
                "[dim]Proceeding without an existing ID column; auto-generated IDs will be used.[/dim]"
            )
            return None

        # Prompt for selection
        default_choice = "1" if candidates else "0"
        numeric_choices = [str(i) for i in range(0, len(candidates) + 1)]
        selection_choices = numeric_choices.copy()

        allow_manual_combine = len(getattr(df, "columns", [])) >= 2
        if allow_manual_combine:
            selection_choices.extend(["C", "c"])
            if not candidates:
                default_choice = "c"

        choice = Prompt.ask(
            "[bold yellow]Select ID column[/bold yellow]",
            choices=selection_choices,
            default=default_choice,
        ).strip()

        if choice.lower() == "c":
            columns = list(map(str, df.columns))
            console.print("\n[bold]Select columns to combine into a unique identifier:[/bold]")
            for idx, col in enumerate(columns, 1):
                console.print(f"  {idx}. {col}")
            console.print("[dim]Enter column numbers separated by commas (e.g., '1,2')[/dim]")

            while True:
                selection = Prompt.ask("[bold yellow]Columns to combine[/bold yellow]", default="").strip()
                if not selection:
                    console.print("[yellow]No columns selected. Keeping auto-generated IDs.[/yellow]")
                    return None
                try:
                    indices = [int(part.strip()) - 1 for part in selection.split(",")]
                    if not indices or any(idx < 0 or idx >= len(columns) for idx in indices):
                        raise ValueError
                except ValueError:
                    console.print("[red]Invalid selection. Please enter valid column numbers (e.g., '1,3').[/red]")
                    continue

                selected_cols = [columns[idx] for idx in indices]
                combined_name = " + ".join(selected_cols)
                return_value = "+".join(selected_cols)
                console.print(f"[green]✓ ID column: '{combined_name}'[/green]")
                return return_value

        if choice == "0":
            console.print("[dim]✓ An 'llm_annotation_id' will be generated automatically[/dim]")
            return None

        selected_idx = int(choice) - 1
        selected_candidate = candidates[selected_idx]
        console.print(f"[green]✓ ID column: '{selected_candidate['name']}'[/green]")
        return selected_candidate["return_value"]

    @staticmethod
    def suggest_text_column(dataset: DatasetInfo) -> Optional[str]:
        """Suggest the most likely text column from a dataset."""

        text_candidates = [
            "text",
            "content",
            "message",
            "comment",
            "review",
            "description",
            "body",
            "document",
            "sentence",
            "paragraph",
        ]

        column_types = dataset.column_types or {}

        def is_probably_identifier(name: str) -> bool:
            lower = name.lower()
            if lower in {"id", "identifier"}:
                return True
            if lower.endswith("_id") or lower.endswith("id"):
                return True
            return False

        def candidate_score(name: str) -> float:
            return dataset.text_scores.get(name, 0.0)

        textual_columns = []
        for col in dataset.columns:
            dtype = column_types.get(col, "").lower()
            if "object" in dtype or "string" in dtype:
                textual_columns.append(col)

        if not textual_columns:
            textual_columns = list(dataset.columns)

        exact_matches = [
            col
            for col in textual_columns
            if col.lower() in text_candidates and not is_probably_identifier(col)
        ]
        if exact_matches:
            exact_matches.sort(key=candidate_score, reverse=True)
            return exact_matches[0]

        partial_matches = []
        for col in textual_columns:
            col_lower = col.lower()
            if is_probably_identifier(col):
                continue
            for candidate in text_candidates:
                if candidate in col_lower:
                    partial_matches.append(col)
                    break

        if partial_matches:
            partial_matches.sort(key=candidate_score, reverse=True)
            return partial_matches[0]

        if dataset.text_scores:
            for col, _ in sorted(dataset.text_scores.items(), key=lambda item: item[1], reverse=True):
                if col in textual_columns and not is_probably_identifier(col):
                    return col

        for col in textual_columns:
            if not is_probably_identifier(col):
                return col

        return dataset.columns[0] if dataset.columns else None


__all__ = ["DatasetInfo", "DataDetector", "HAS_PANDAS"]

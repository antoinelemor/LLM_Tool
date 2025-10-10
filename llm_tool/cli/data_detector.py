#!/usr/bin/env python3
"""
Utilities to analyse datasets for the CLI workflows.

This module centralises the dataset detection logic that used to live in
`advanced_cli.py` so it can be reused by both the Training Arena and
Annotator Factory integrations without circular imports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        """Scan a directory (recursively) for supported dataset files."""

        datasets: List[DatasetInfo] = []

        if not directory.exists():
            return datasets

        patterns = [
            "**/*.csv",
            "**/*.json",
            "**/*.jsonl",
            "**/*.xlsx",
            "**/*.xls",
            "**/*.parquet",
            "**/*.RData",
            "**/*.rdata",
        ]

        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    dataset_info = DataDetector.analyze_file(file_path)
                    if dataset_info:
                        datasets.append(dataset_info)

        return datasets

    @staticmethod
    def analyze_file(file_path: Path) -> Optional[DatasetInfo]:
        """Analyse a single file to extract dataset information."""

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
                result["text_column_candidates"].append(column_info)

            is_json = DataDetector._looks_like_json_column(df[col])
            if is_json:
                column_info["is_json"] = True
                result["json_columns"].append(column_info)

            if DataDetector._looks_like_label_column(col, df[col]):
                result["label_column_candidates"].append(column_info)

            if DataDetector._looks_like_language_column(col, df[col]):
                result["language_column_candidates"].append(column_info)

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
    def detect_id_column_candidates(df, text_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """Identify potential ID columns in a dataframe."""

        if not HAS_PANDAS:
            return []

        candidates: List[Dict[str, Any]] = []

        for col in df.columns:
            series = df[col]
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            dtype = str(series.dtype)
            is_numeric = pd.api.types.is_numeric_dtype(series)  # type: ignore[attr-defined]
            is_string = pd.api.types.is_string_dtype(series)  # type: ignore[attr-defined]
            looks_like_id = DataDetector._looks_like_identifier(col)

            if unique_ratio < 0.8:
                continue

            if dtype == "object" and not is_string:
                continue

            if col == text_column:
                continue

            candidates.append(
                {
                    "name": col,
                    "dtype": dtype,
                    "unique_ratio": unique_ratio,
                    "is_numeric": bool(is_numeric),
                    "has_id_in_name": looks_like_id,
                }
            )

        candidates.sort(key=lambda x: (x["unique_ratio"], x["has_id_in_name"]), reverse=True)
        return candidates

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
    def display_and_select_id_column(
        console,
        df,
        text_column: Optional[str] = None,
        step_label: str = "Identifier Column Selection",
    ) -> Optional[str]:
        """Interactive helper to pick an ID column using Rich."""

        if not HAS_PANDAS or console is None:
            return None

        from rich import box
        from rich.prompt import Prompt
        from rich.table import Table

        candidates = DataDetector.detect_id_column_candidates(df, text_column)

        console.print(f"\n[bold]{step_label}[/bold]")
        console.print()
        console.print("[bold cyan]📋 What is an ID column?[/bold cyan]")
        console.print("  • [green]Tracks each text[/green] through the annotation process")
        console.print("  • [green]Links results back[/green] to your original data")
        console.print("  • [green]Resumes annotation[/green] if interrupted")
        console.print("  • [yellow]Not mandatory[/yellow] - we can generate one automatically")
        console.print()

        if not candidates:
            console.print("[yellow]ℹ️  No unique ID columns detected[/yellow]")
            console.print("[dim]  → An 'llm_annotation_id' column will be created automatically[/dim]")
            return None

        id_table = Table(title="Candidate ID Columns", box=box.ROUNDED)
        id_table.add_column("#", style="cyan", width=6)
        id_table.add_column("Column", style="green", width=32)
        id_table.add_column("Type", style="yellow", width=14)
        id_table.add_column("Unique %", style="cyan", width=12, justify="right")

        for idx, candidate in enumerate(candidates, 1):
            id_table.add_row(
                str(idx),
                candidate["name"],
                candidate["dtype"],
                f"{candidate['unique_ratio'] * 100:.1f}%",
            )

        id_table.add_row("0", "[dim]None (auto-generate)[/dim]", "", "")

        console.print(id_table)
        console.print()

        max_choice = len(candidates)
        choices = [str(i) for i in range(0, max_choice + 1)]
        choice = Prompt.ask(
            "[bold yellow]Select ID column[/bold yellow]",
            choices=choices,
            default="0" if not candidates else "1",
        )

        if choice == "0":
            console.print("[dim]✓ An 'llm_annotation_id' will be generated automatically[/dim]")
            return None

        selected_idx = int(choice) - 1
        selected_column = candidates[selected_idx]["name"]
        console.print(f"[green]✓ ID column: '{selected_column}'[/green]")
        return selected_column

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

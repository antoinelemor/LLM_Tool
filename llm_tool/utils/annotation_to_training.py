#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
annotation_to_training.py

MAIN OBJECTIVE:
---------------
Convert LLM annotations (JSON format) into training data formats suitable for
model training. Supports single-label and multi-label classification with
flexible label creation strategies.

MAIN FEATURES:
--------------
1) Parse JSON annotations from CSV files
2) Create labels using key+value concatenation (e.g., "sentiment_positive")
3) Support single-label training (one model per annotation key)
4) Support multi-label training (one model for all labels from a key)
5) Filter and prepare only annotated rows
6) Export to JSONL format for training

Author:
-------
Antoine Lemor
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import pandas as pd

from .data_filter_logger import get_filter_logger


logger = logging.getLogger(__name__)


class AnnotationToTrainingConverter:
    """Convert annotated CSV files to training-ready JSONL format."""

    def __init__(self, verbose: bool = True):
        """
        Initialize the converter.

        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def analyze_annotations(
        self,
        csv_path: str,
        text_column: str = "sentence",
        annotation_column: str = "annotation"
    ) -> Dict[str, Any]:
        """
        Analyze the annotation structure to understand available keys and values.

        Args:
            csv_path: Path to annotated CSV file
            text_column: Column containing text data
            annotation_column: Column containing JSON annotations

        Returns:
            Dictionary with analysis results:
            - total_rows: Total number of rows
            - annotated_rows: Number of rows with annotations
            - annotation_keys: Dictionary of keys with their value distributions
            - sample_annotation: Example annotation for reference
        """
        df = pd.read_csv(csv_path)

        total_rows = len(df)

        # Check if annotation column exists
        if annotation_column not in df.columns:
            self.logger.error(f"Annotation column '{annotation_column}' not found in dataset. Available columns: {', '.join(df.columns)}")
            return {
                "total_rows": total_rows,
                "annotated_rows": 0,
                "annotation_keys": {},
                "sample_annotation": None,
                "all_columns": list(df.columns),
                "issues": [f"Column '{annotation_column}' not found in dataset"]
            }

        annotated_rows = df[annotation_column].notna().sum()

        if annotated_rows == 0:
            self.logger.warning("No annotated rows found in dataset")
            return {
                "total_rows": total_rows,
                "annotated_rows": 0,
                "annotation_keys": {},
                "sample_annotation": None
            }

        # Analyze annotation structure
        annotation_keys = {}
        sample_annotation = None

        for idx, row in df.iterrows():
            if pd.notna(row[annotation_column]):
                try:
                    # Ensure annotation is a string before parsing
                    annotation_value = row[annotation_column]
                    if not isinstance(annotation_value, str):
                        # Skip non-string values (int, float, etc.)
                        continue

                    # Try JSON first
                    annotation = json.loads(annotation_value)
                except json.JSONDecodeError:
                    # Try Python literal (handles single quotes)
                    try:
                        import ast
                        annotation = ast.literal_eval(annotation_value)
                    except (ValueError, SyntaxError):
                        continue

                # Process annotation if successfully parsed
                try:
                    if sample_annotation is None:
                        sample_annotation = annotation

                    # Track each key and its values
                    for key, value in annotation.items():
                        if key not in annotation_keys:
                            annotation_keys[key] = {
                                "type": type(value).__name__,
                                "values": Counter(),
                                "null_count": 0
                            }

                        if value is None:
                            annotation_keys[key]["null_count"] += 1
                        elif isinstance(value, list):
                            # For list values, count each item
                            for item in value:
                                annotation_keys[key]["values"][item] += 1
                        else:
                            # For scalar values
                            annotation_keys[key]["values"][value] += 1

                except (AttributeError, TypeError):
                    continue

        # Convert Counter to dict for better display
        for key in annotation_keys:
            annotation_keys[key]["values"] = dict(annotation_keys[key]["values"])

        return {
            "total_rows": total_rows,
            "annotated_rows": annotated_rows,
            "annotation_keys": annotation_keys,
            "sample_annotation": sample_annotation
        }

    def create_single_label_datasets(
        self,
        csv_path: str,
        output_dir: str,
        text_column: str = "sentence",
        annotation_column: str = "annotation",
        annotation_keys: Optional[List[str]] = None,
        label_strategy: str = "key_value",
        id_column: Optional[str] = None,
        lang_column: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create separate JSONL datasets for single-label classification.
        One dataset per annotation key, with labels as "key_value" pairs.

        Args:
            csv_path: Path to annotated CSV
            output_dir: Directory to save JSONL files
            text_column: Column with text data
            annotation_column: Column with JSON annotations
            annotation_keys: Specific keys to process (None = all keys)
            label_strategy: How to create labels:
                - "key_value": Concatenate key + value (e.g., "sentiment_positive")
                - "value_only": Use only the value (e.g., "positive")
            id_column: Column to use as ID (auto-detect if None)
            lang_column: Column to use as language (auto-detect if None)

        Returns:
            Dictionary mapping annotation keys to output file paths
        """
        df = pd.read_csv(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect ID and language columns if not specified
        if id_column is None:
            # Common ID column names
            for col_name in ['id', 'promesse_id', 'sentence_id', 'doc_id', 'item_id']:
                if col_name in df.columns:
                    id_column = col_name
                    break

        # Handle multiple ID columns (e.g., "promesse_id+sentence_id")
        id_columns = []
        if id_column and '+' in id_column:
            id_columns = [col.strip() for col in id_column.split('+')]
        elif id_column:
            id_columns = [id_column]

        if lang_column is None:
            # Common language column names
            for col_name in ['lang', 'language', 'langue', 'lng']:
                if col_name in df.columns:
                    lang_column = col_name
                    break

        # Filter only annotated rows
        filter_logger = get_filter_logger()
        df_before_filter = df.copy()
        df = df[df[annotation_column].notna()].copy()

        # Log filtered rows
        if len(df) < len(df_before_filter):
            filter_logger.log_dataframe_filtering(
                df_before=df_before_filter,
                df_after=df,
                reason="missing_annotation",
                location="annotation_to_training.create_single_label_datasets",
                text_column=text_column,
                log_filtered_samples=3
            )

        if len(df) == 0:
            self.logger.error("No annotated rows to process")
            return {}

        output_files = {}

        # Group data by annotation key
        for key in annotation_keys or self._get_all_keys(df, annotation_column):
            samples = []
            filtered_items = []

            for idx, row in df.iterrows():
                try:
                    # Ensure annotation is a string before parsing
                    annotation_value = row[annotation_column]
                    if not isinstance(annotation_value, str):
                        filtered_items.append({
                            'index': idx,
                            'reason': 'non_string_annotation',
                            'type': type(annotation_value).__name__
                        })
                        continue

                    annotation = json.loads(annotation_value)

                    if key not in annotation:
                        continue

                    value = annotation[key]

                    # Skip null values
                    if value is None:
                        filtered_items.append({
                            'index': idx,
                            'reason': 'null_value',
                            'key': key
                        })
                        continue

                    # Create label based on strategy
                    if isinstance(value, list):
                        # For lists, create multiple labels (key_item1, key_item2, ...)
                        if label_strategy == "key_value":
                            labels = [f"{key}_{item}" for item in value]
                        else:
                            labels = [str(item) for item in value]
                    else:
                        # For scalars, create single label
                        if label_strategy == "key_value":
                            labels = [f"{key}_{value}"]
                        else:
                            labels = [str(value)]

                    sample = {
                        "text": row[text_column],
                        "label": labels
                    }

                    # Add ID metadata (combine multiple columns if specified)
                    if id_columns:
                        id_parts = []
                        for col in id_columns:
                            if col in df.columns and pd.notna(row.get(col)):
                                id_parts.append(str(row[col]))
                        if id_parts:
                            sample["id"] = "_".join(id_parts)

                    # Add language metadata
                    if lang_column and lang_column in df.columns and pd.notna(row.get(lang_column)):
                        sample["lang"] = row[lang_column]

                    samples.append(sample)

                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Row {idx}: Failed to process - {e}")
                    filtered_items.append({
                        'index': idx,
                        'reason': 'parse_error',
                        'error': str(e)
                    })
                    continue

            # Log filtered items for this key
            if filtered_items:
                filter_logger.log_filtered_batch(
                    items=[f"Row {f['index']}: {f['reason']}" for f in filtered_items],
                    reason="annotation_processing_error",
                    location=f"annotation_to_training.create_single_label_datasets.{key}",
                    indices=[f['index'] for f in filtered_items]
                )

            if samples:
                output_file = output_dir / f"training_{key}.jsonl"
                self._write_jsonl(samples, output_file)
                output_files[key] = str(output_file)

                if self.verbose:
                    self.logger.info(f"Created {len(samples)} samples for '{key}' -> {output_file}")
            else:
                self.logger.warning(f"No valid samples found for key '{key}'")

        return output_files

    def create_multi_label_dataset(
        self,
        csv_path: str,
        output_path: str,
        text_column: str = "sentence",
        annotation_column: str = "annotation",
        annotation_keys: Optional[List[str]] = None,
        label_strategy: str = "key_value",
        id_column: Optional[str] = None,
        lang_column: Optional[str] = None
    ) -> str:
        """
        Create a single JSONL dataset for multi-label classification.
        Each sample has multiple labels from different annotation keys.

        Args:
            csv_path: Path to annotated CSV
            output_path: Path for output JSONL file
            text_column: Column with text data
            annotation_column: Column with JSON annotations
            annotation_keys: Specific keys to include (None = all keys)
            label_strategy: How to create labels (see create_single_label_datasets)
            id_column: Column to use as ID (auto-detect if None)
            lang_column: Column to use as language (auto-detect if None)

        Returns:
            Path to output JSONL file
        """
        df = pd.read_csv(csv_path)

        # Check if required columns exist
        if text_column not in df.columns:
            self.logger.error(f"Text column '{text_column}' not found in dataset. Available columns: {', '.join(df.columns)}")
            return None

        if annotation_column not in df.columns:
            self.logger.error(f"Annotation column '{annotation_column}' not found in dataset. Available columns: {', '.join(df.columns)}")
            return None

        # Auto-detect ID and language columns if not specified
        if id_column is None:
            for col_name in ['id', 'promesse_id', 'sentence_id', 'doc_id', 'item_id']:
                if col_name in df.columns:
                    id_column = col_name
                    break

        # Handle multiple ID columns (e.g., "promesse_id+sentence_id")
        id_columns = []
        if id_column and '+' in id_column:
            id_columns = [col.strip() for col in id_column.split('+')]
        elif id_column:
            id_columns = [id_column]

        if lang_column is None:
            for col_name in ['lang', 'language', 'langue', 'lng']:
                if col_name in df.columns:
                    lang_column = col_name
                    break

        # Filter only annotated rows
        filter_logger = get_filter_logger()
        df_before_filter = df.copy()
        df = df[df[annotation_column].notna()].copy()

        # Log filtered rows
        if len(df) < len(df_before_filter):
            filter_logger.log_dataframe_filtering(
                df_before=df_before_filter,
                df_after=df,
                reason="missing_annotation",
                location="annotation_to_training.create_multi_label_dataset",
                text_column=text_column,
                log_filtered_samples=3
            )

        if len(df) == 0:
            self.logger.error("No annotated rows to process")
            return None

        samples = []
        filtered_items = []
        keys_to_use = annotation_keys or self._get_all_keys(df, annotation_column)

        for idx, row in df.iterrows():
            try:
                annotation_val = row[annotation_column]

                # Skip empty or null annotations
                if pd.isna(annotation_val) or annotation_val == '' or annotation_val == 'null':
                    filtered_items.append({
                        'index': idx,
                        'reason': 'empty_or_null_annotation',
                        'value': str(annotation_val)
                    })
                    continue

                # Try to parse JSON with robust error handling
                annotation = None
                try:
                    if isinstance(annotation_val, str):
                        # Try standard JSON parsing first
                        annotation = json.loads(annotation_val)
                    elif isinstance(annotation_val, dict):
                        annotation = annotation_val
                    else:
                        # Skip non-string, non-dict values
                        filtered_items.append({
                            'index': idx,
                            'reason': 'invalid_annotation_type',
                            'type': type(annotation_val).__name__
                        })
                        continue
                except json.JSONDecodeError as je:
                    # Try to parse as Python literal (handles single quotes with escapes)
                    try:
                        import ast
                        annotation = ast.literal_eval(annotation_val)
                    except (ValueError, SyntaxError):
                        # Try to fix common JSON issues as fallback
                        try:
                            # Replace single quotes with double quotes (simple case only)
                            fixed_json = annotation_val.replace("'", '"')
                            annotation = json.loads(fixed_json)
                        except:
                            # If still fails, log and skip
                            self.logger.warning(f"Row {idx}: Malformed JSON - {str(je)[:100]}")
                            filtered_items.append({
                                'index': idx,
                                'reason': 'malformed_json',
                                'error': str(je)[:100]
                            })
                            continue

                if not annotation or not isinstance(annotation, dict):
                    filtered_items.append({
                        'index': idx,
                        'reason': 'empty_or_invalid_annotation',
                        'annotation': str(annotation)
                    })
                    continue

                # Collect all labels from specified keys as a FLAT list
                all_labels = []

                for key in keys_to_use:
                    if key not in annotation:
                        continue

                    value = annotation[key]

                    # Skip null values
                    if value is None or value == '' or value == 'null':
                        continue

                    # Create labels based on strategy and add to flat list
                    if isinstance(value, list):
                        # Filter out empty or null values from list
                        clean_values = [v for v in value if v is not None and v != '' and v != 'null']
                        if label_strategy == "key_value":
                            all_labels.extend([f"{key}_{item}" for item in clean_values])
                        else:
                            all_labels.extend(clean_values)
                    else:
                        if label_strategy == "key_value":
                            all_labels.append(f"{key}_{value}")
                        else:
                            all_labels.append(str(value))

                # Only include if we have at least one non-null label
                if all_labels:
                    sample = {
                        "text": str(row[text_column]),
                        "labels": all_labels
                    }

                    # Add ID metadata (combine multiple columns if specified)
                    if id_columns:
                        id_parts = []
                        for col in id_columns:
                            if col in df.columns and pd.notna(row.get(col)):
                                id_parts.append(str(row[col]))
                        if id_parts:
                            sample["id"] = "_".join(id_parts)

                    # Add language metadata
                    if lang_column and lang_column in df.columns and pd.notna(row.get(lang_column)):
                        sample["lang"] = str(row[lang_column])

                    samples.append(sample)

            except (KeyError, AttributeError, TypeError, ValueError) as e:
                self.logger.warning(f"Row {idx}: Failed to process - {e}")
                filtered_items.append({
                    'index': idx,
                    'reason': 'processing_error',
                    'error': str(e)
                })
                continue

        # Log all filtered items
        if filtered_items:
            filter_logger.log_filtered_batch(
                items=[f"Row {f['index']}: {f['reason']}" for f in filtered_items],
                reason="annotation_processing_error",
                location="annotation_to_training.create_multi_label_dataset",
                indices=[f['index'] for f in filtered_items]
            )

        if samples:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_jsonl(samples, output_path)

            if self.verbose:
                self.logger.info(f"Created {len(samples)} multi-label samples -> {output_path}")

            return str(output_path)
        else:
            self.logger.error("No valid samples created")
            return None

    def _get_all_keys(self, df: pd.DataFrame, annotation_column: str) -> List[str]:
        """Extract all unique keys from annotations."""
        keys = set()

        for idx, row in df.iterrows():
            if pd.notna(row[annotation_column]):
                try:
                    # Ensure annotation is a string before parsing
                    annotation_value = row[annotation_column]
                    if not isinstance(annotation_value, str):
                        continue

                    # Try JSON first
                    annotation = json.loads(annotation_value)
                    keys.update(annotation.keys())
                except json.JSONDecodeError:
                    # Try Python literal (handles single quotes)
                    try:
                        import ast
                        annotation = ast.literal_eval(annotation_value)
                        if isinstance(annotation, dict):
                            keys.update(annotation.keys())
                    except (ValueError, SyntaxError):
                        continue

        return sorted(list(keys))

    def _write_jsonl(self, data: List[Dict], output_path: Path):
        """Write data to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

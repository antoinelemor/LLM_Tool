#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
doccano_exporter.py

MAIN OBJECTIVE:
---------------
This script exports annotated data to Doccano-compatible JSONL format
for human validation and quality control of LLM annotations.

Dependencies:
-------------
- sys
- json
- pandas
- pathlib
- typing
- random
- logging

MAIN FEATURES:
--------------
1) Convert annotations to Doccano JSONL format
2) Support for classification and sequence labeling tasks
3) Random sampling for validation
4) Multiple label format support
5) Metadata preservation

Author:
-------
Antoine Lemor
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd


class DoccanoExporter:
    """Export annotations to Doccano format for human validation"""

    def __init__(self):
        """Initialize the Doccano exporter"""
        self.logger = logging.getLogger(__name__)

    def export_to_doccano(
        self,
        data: Union[pd.DataFrame, List[Dict], str],
        output_path: str,
        text_column: str = "text",
        label_column: str = "label",
        task_type: str = "text_classification",
        sample_size: Optional[int] = None,
        random_sample: bool = True,
        include_metadata: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Export data to Doccano-compatible JSONL format.

        Parameters
        ----------
        data : pd.DataFrame, List[Dict], or str
            Input data (DataFrame, list of dicts, or path to file)
        output_path : str
            Path to save the JSONL file
        text_column : str
            Name of the text column
        label_column : str
            Name of the label/annotation column
        task_type : str
            Type of task ('text_classification', 'sequence_labeling', 'seq2seq')
        sample_size : int, optional
            Number of samples to export (None for all)
        random_sample : bool
            Whether to randomly sample (if sample_size is set)
        include_metadata : bool
            Include additional metadata in export
        confidence_threshold : float, optional
            Only export items above this confidence level

        Returns
        -------
        dict
            Export statistics and information
        """
        # Load data if path is provided
        if isinstance(data, str):
            data = self._load_data(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        # Validate columns
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in data.columns:
            self.logger.warning(f"Label column '{label_column}' not found, will create empty labels")
            data[label_column] = None

        # Filter by confidence if threshold is provided
        if confidence_threshold is not None:
            confidence_col = f"{label_column}_confidence"
            if confidence_col in data.columns:
                initial_size = len(data)
                data = data[data[confidence_col] >= confidence_threshold]
                filtered = initial_size - len(data)
                self.logger.info(f"Filtered {filtered} items below confidence threshold {confidence_threshold}")

        # Sample data if requested
        if sample_size is not None and sample_size < len(data):
            if random_sample:
                data = data.sample(n=sample_size, random_state=42)
                self.logger.info(f"Randomly sampled {sample_size} items")
            else:
                data = data.head(sample_size)
                self.logger.info(f"Selected first {sample_size} items")

        # Convert based on task type
        if task_type == "text_classification":
            doccano_data = self._convert_text_classification(
                data, text_column, label_column, include_metadata
            )
        elif task_type == "sequence_labeling":
            doccano_data = self._convert_sequence_labeling(
                data, text_column, label_column, include_metadata
            )
        elif task_type == "seq2seq":
            doccano_data = self._convert_seq2seq(
                data, text_column, label_column, include_metadata
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Write to JSONL file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in doccano_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Return statistics
        stats = {
            'total_exported': len(doccano_data),
            'output_path': str(output_path),
            'task_type': task_type,
            'unique_labels': self._count_unique_labels(doccano_data),
            'file_size_kb': output_path.stat().st_size / 1024
        }

        self.logger.info(f"Exported {stats['total_exported']} items to {output_path}")
        return stats

    def _convert_text_classification(
        self,
        data: pd.DataFrame,
        text_column: str,
        label_column: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Convert data for text classification task"""
        doccano_data = []

        for idx, row in data.iterrows():
            item = {
                "text": str(row[text_column])
            }

            # Handle label
            label_value = row[label_column]
            if pd.notna(label_value):
                if isinstance(label_value, str):
                    try:
                        # Try to parse JSON annotation
                        label_dict = json.loads(label_value)
                        if isinstance(label_dict, dict):
                            # Extract label from dict
                            if 'label' in label_dict:
                                item['label'] = label_dict['label']
                            elif 'category' in label_dict:
                                item['label'] = label_dict['category']
                            elif 'class' in label_dict:
                                item['label'] = label_dict['class']
                            else:
                                # Use first value as label
                                first_key = next(iter(label_dict.keys()))
                                item['label'] = label_dict[first_key]

                            # Add confidence if available
                            if 'confidence' in label_dict:
                                item['confidence'] = label_dict['confidence']
                        else:
                            item['label'] = str(label_dict)
                    except json.JSONDecodeError:
                        # Plain text label
                        item['label'] = label_value
                elif isinstance(label_value, (list, dict)):
                    # Already parsed
                    if isinstance(label_value, dict):
                        item['label'] = label_value.get('label', str(label_value))
                    else:
                        item['label'] = label_value
                else:
                    item['label'] = str(label_value)
            else:
                item['label'] = ""

            # Add metadata if requested
            if include_metadata:
                metadata = {}
                # Add ID if available
                if 'id' in row:
                    metadata['id'] = row['id']
                # Add inference time if available
                time_col = f"{label_column}_inference_time"
                if time_col in row:
                    metadata['inference_time'] = row[time_col]
                # Add any other relevant columns
                for col in data.columns:
                    if col not in [text_column, label_column] and not col.endswith('_per_prompt'):
                        if pd.notna(row[col]):
                            metadata[col] = str(row[col])

                if metadata:
                    item['metadata'] = metadata

            doccano_data.append(item)

        return doccano_data

    def _convert_sequence_labeling(
        self,
        data: pd.DataFrame,
        text_column: str,
        label_column: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Convert data for sequence labeling (NER) task"""
        doccano_data = []

        for idx, row in data.iterrows():
            item = {
                "text": str(row[text_column]),
                "labels": []
            }

            # Parse entities from label
            label_value = row[label_column]
            if pd.notna(label_value):
                if isinstance(label_value, str):
                    try:
                        label_dict = json.loads(label_value)
                        if 'entities' in label_dict:
                            entities = label_dict['entities']
                        elif 'labels' in label_dict:
                            entities = label_dict['labels']
                        else:
                            entities = []

                        # Convert to Doccano format
                        for entity in entities:
                            if isinstance(entity, dict):
                                label_item = [
                                    entity.get('start', 0),
                                    entity.get('end', 0),
                                    entity.get('label', entity.get('type', 'ENTITY'))
                                ]
                                item['labels'].append(label_item)

                    except json.JSONDecodeError:
                        pass

            # Add metadata
            if include_metadata:
                item['metadata'] = {
                    'id': row.get('id', idx),
                    'source': 'llm_annotation'
                }

            doccano_data.append(item)

        return doccano_data

    def _convert_seq2seq(
        self,
        data: pd.DataFrame,
        text_column: str,
        label_column: str,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Convert data for sequence-to-sequence task"""
        doccano_data = []

        for idx, row in data.iterrows():
            item = {
                "text": str(row[text_column]),
                "target": ""
            }

            # Handle target/label
            label_value = row[label_column]
            if pd.notna(label_value):
                if isinstance(label_value, str):
                    try:
                        label_dict = json.loads(label_value)
                        if isinstance(label_dict, dict):
                            # Extract target text
                            if 'summary' in label_dict:
                                item['target'] = label_dict['summary']
                            elif 'translation' in label_dict:
                                item['target'] = label_dict['translation']
                            elif 'answer' in label_dict:
                                item['target'] = label_dict['answer']
                            else:
                                item['target'] = json.dumps(label_dict, ensure_ascii=False)
                        else:
                            item['target'] = str(label_dict)
                    except json.JSONDecodeError:
                        item['target'] = label_value
                else:
                    item['target'] = str(label_value)

            # Add metadata
            if include_metadata:
                item['metadata'] = {'id': row.get('id', idx)}

            doccano_data.append(item)

        return doccano_data

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        file_path = Path(file_path)

        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return pd.DataFrame(data)
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _count_unique_labels(self, doccano_data: List[Dict]) -> int:
        """Count unique labels in the data"""
        labels = set()

        for item in doccano_data:
            if 'label' in item:
                labels.add(item['label'])
            elif 'labels' in item:
                for label in item['labels']:
                    if isinstance(label, list) and len(label) >= 3:
                        labels.add(label[2])

        return len(labels)

    def create_validation_sample(
        self,
        annotated_file: str,
        output_file: str,
        sample_size: int = 100,
        stratified: bool = True,
        text_column: str = "text",
        label_column: str = "annotation"
    ) -> Dict[str, Any]:
        """
        Create a validation sample for human review.

        Parameters
        ----------
        annotated_file : str
            Path to the annotated data file
        output_file : str
            Path to save the validation sample
        sample_size : int
            Number of samples to include
        stratified : bool
            Whether to use stratified sampling (preserve label distribution)
        text_column : str
            Name of text column
        label_column : str
            Name of label/annotation column

        Returns
        -------
        dict
            Validation sample statistics
        """
        # Load data
        data = self._load_data(annotated_file)

        # Perform sampling
        if stratified and label_column in data.columns:
            # Try to stratify by label
            try:
                # Parse labels for stratification
                label_values = []
                for val in data[label_column]:
                    if pd.notna(val):
                        if isinstance(val, str):
                            try:
                                parsed = json.loads(val)
                                if isinstance(parsed, dict):
                                    label_values.append(
                                        parsed.get('label', parsed.get('category', 'unknown'))
                                    )
                                else:
                                    label_values.append(str(parsed))
                            except:
                                label_values.append(val)
                        else:
                            label_values.append(str(val))
                    else:
                        label_values.append('unlabeled')

                data['_stratify_label'] = label_values

                # Stratified sampling
                from sklearn.model_selection import train_test_split
                _, sample = train_test_split(
                    data,
                    test_size=min(sample_size / len(data), 1.0),
                    stratify=data['_stratify_label'],
                    random_state=42
                )
                data = sample.drop('_stratify_label', axis=1)
                self.logger.info(f"Created stratified sample of {len(data)} items")

            except Exception as e:
                self.logger.warning(f"Stratified sampling failed: {e}, using random sampling")
                data = data.sample(n=min(sample_size, len(data)), random_state=42)
        else:
            # Random sampling
            data = data.sample(n=min(sample_size, len(data)), random_state=42)
            self.logger.info(f"Created random sample of {len(data)} items")

        # Export to Doccano format
        stats = self.export_to_doccano(
            data,
            output_file,
            text_column=text_column,
            label_column=label_column,
            include_metadata=True
        )

        return stats
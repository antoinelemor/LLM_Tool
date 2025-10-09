#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_training_data_builder.py

MAIN OBJECTIVE:
---------------
Pytest test suite for TrainingDatasetBuilder to verify training data generation
for different formats (category CSV, binary long format, LLM annotations).

Dependencies:
-------------
- json
- pathlib
- pandas
- llm_tool.trainers.training_data_builder

MAIN FEATURES:
--------------
1) Test category CSV format training data building
2) Test binary long format with multi-label support
3) Test LLM annotations JSON format parsing
4) Verify output file generation and structure
5) Validate multi-label dataset creation
6) Check label column and value correctness

Author:
-------
Antoine Lemor
"""

import json
from pathlib import Path

import pandas as pd

from llm_tool.trainers.training_data_builder import (
    TrainingDatasetBuilder,
    TrainingDataRequest,
)


def test_builder_category_csv(tmp_path):
    data = pd.DataFrame(
        {
            "text": ["Alpha", "Beta", "Gamma"],
            "label": ["positive", "negative", "positive"],
        }
    )
    source = tmp_path / "category.csv"
    data.to_csv(source, index=False)

    builder = TrainingDatasetBuilder(tmp_path / "out")
    bundle = builder.build(
        TrainingDataRequest(
            input_path=source,
            format="category_csv",
            text_column="text",
            label_column="label",
            mode="single-label",
        )
    )

    assert bundle.primary_file is not None
    result_df = pd.read_csv(bundle.primary_file)
    assert list(result_df.columns) == ["text", "label"]
    assert set(result_df["label"]) == {"positive", "negative"}


def test_builder_binary_long(tmp_path):
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "text": ["Alpha", "Alpha", "Beta", "Beta"],
            "category": ["economy", "health", "economy", "health"],
            "value": [1, 0, 0, 1],
        }
    )
    source = tmp_path / "binary.csv"
    data.to_csv(source, index=False)

    builder = TrainingDatasetBuilder(tmp_path / "out")
    bundle = builder.build(
        TrainingDataRequest(
            input_path=source,
            format="binary_long_csv",
            text_column="text",
            category_column="category",
            value_column="value",
            id_column="id",
            mode="multi-label",
        )
    )

    assert bundle.strategy == "multi-label"
    assert "economy" in bundle.training_files
    assert "health" in bundle.training_files
    multilabel_path = bundle.training_files.get("multilabel")
    assert multilabel_path is not None and Path(multilabel_path).exists()

    # Verify binary dataset structure
    economy_df = pd.read_csv(bundle.training_files["economy"])
    assert set(economy_df.columns) == {"text", "label"}
    assert set(economy_df["label"]) == {0, 1}


def test_builder_llm_annotations_single(tmp_path):
    rows = [
        {
            "sentence": "Alpha",
            "annotation": json.dumps({"sentiment": "positive", "theme": ["economy"]}),
        },
        {
            "sentence": "Beta",
            "annotation": json.dumps({"sentiment": "negative", "theme": ["health"]}),
        },
    ]
    df = pd.DataFrame(rows)
    source = tmp_path / "annotations.csv"
    df.to_csv(source, index=False)

    builder = TrainingDatasetBuilder(tmp_path / "out")
    bundle = builder.build(
        TrainingDataRequest(
            input_path=source,
            format="llm_json",
            text_column="sentence",
            annotation_column="annotation",
            mode="single-label",
        )
    )

    assert bundle.strategy == "multi-label"
    assert bundle.training_files
    assert any(key.startswith("sentiment") for key in bundle.training_files)

    # Ensure generated datasets exist
    for path in bundle.training_files.values():
        assert Path(path).exists()

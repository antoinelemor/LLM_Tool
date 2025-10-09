#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_annotation_pipeline.py

MAIN OBJECTIVE:
---------------
Pytest test suite for annotation pipeline with stubbed LLM clients to verify
LLMAnnotator and PipelineController functionality without external dependencies.

Dependencies:
-------------
- asyncio
- json
- pathlib
- pandas
- pytest
- llm_tool.annotators.llm_annotator
- llm_tool.pipelines.pipeline_controller

MAIN FEATURES:
--------------
1) Dummy local client for stable test results
2) Sample dataset fixture generation
3) LLMAnnotator test with mocked Ollama client
4) PipelineController annotation phase test
5) Monkeypatch-based mocking for isolation
6) Output file verification

Author:
-------
Antoine Lemor
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from llm_tool.annotators.llm_annotator import LLMAnnotator
from llm_tool.pipelines.pipeline_controller import PipelineController


class _DummyLocalClient:
    def generate(self, prompt: str, **_: dict) -> str:
        # Always return a stable JSON payload regardless of the prompt
        return json.dumps({"sentiment": "positive", "confidence": 0.99})


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "text": [
                "The economy is improving thanks to new policies.",
                "Citizens are concerned about healthcare access.",
                "Education reforms are receiving mixed feedback.",
            ],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_llm_annotator_with_stub(tmp_path: Path, sample_dataset: Path, monkeypatch):
    output_path = tmp_path / "annotations.csv"

    # Mock OllamaClient to return dummy client
    class MockOllamaClient:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate(self, prompt: str, **kwargs) -> str:
            return json.dumps({"sentiment": "positive", "confidence": 0.99})

    # Patch at module level where it's imported
    monkeypatch.setattr("llm_tool.annotators.llm_annotator.OllamaClient", MockOllamaClient)
    monkeypatch.setattr("llm_tool.annotators.llm_annotator.HAS_LOCAL_MODELS", True)

    annotator = LLMAnnotator()
    config = {
        "data_source": "csv",
        "file_path": str(sample_dataset),
        "text_column": "text",
        "text_columns": ["text"],
        "identifier_column": "record_id",
        "annotation_column": "annotation",
        # Use the correct API that the annotator expects
        "model": "stub-model",
        "provider": "ollama",
        "prompts": [
            {
                "prompt": "Classify the sentiment of the text as positive or negative.",
                "expected_keys": ["sentiment", "confidence"],
            }
        ],
        "output_path": str(output_path),
        "output_format": "csv",
        "warmup": False,
        "use_parallel": False,
        "num_processes": 1,
    }

    summary = annotator.annotate(config)

    assert summary["successful"] == 3
    assert summary["errors"] == 0
    assert summary["annotated_rows"] == 3
    assert Path(summary["output_file"]).exists()

    annotated_df = pd.read_csv(summary["output_file"])
    assert "annotation" in annotated_df.columns
    assert annotated_df["annotation"].notna().all()


def test_pipeline_controller_annotation_single_phase(tmp_path: Path, sample_dataset: Path, monkeypatch):
    output_path = tmp_path / "annotations.csv"

    async def fake_annotate_async(self, config):  # pragma: no cover - simple stub
        return {
            "successful": 3,
            "errors": 0,
            "annotated_rows": 3,
            "output_file": config.get("output_path"),
        }

    monkeypatch.setattr(LLMAnnotator, "annotate_async", fake_annotate_async)

    controller = PipelineController()
    config = {
        "mode": "file",
        "data_source": "csv",
        "data_format": "csv",
        "file_path": str(sample_dataset),
        "text_column": "text",
        "text_columns": ["text"],
        "identifier_column": "record_id",
        "annotation_column": "annotation",
        # Use the correct API that the annotator expects
        "model": "stub-model",
        "provider": "ollama",
        "prompts": [
            {
                "prompt": "Placeholder prompt",
                "expected_keys": ["sentiment"],
            }
        ],
        "output_path": str(output_path),
        "output_format": "csv",
        "warmup": False,
        "max_workers": 1,
        "num_processes": 1,
        "use_parallel": False,
        "run_annotation": True,
        "run_validation": False,
        "run_training": False,
        "run_deployment": False,
    }

    summary = controller.run_annotation(config)

    assert summary["successful"] == 3
    assert summary["errors"] == 0
    assert summary["annotated_rows"] == 3
    assert summary["output_file"] == str(output_path)

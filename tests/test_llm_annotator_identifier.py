import pytest
from unittest.mock import MagicMock

pd = pytest.importorskip("pandas")

from llm_tool.annotators.llm_annotator import LLMAnnotator


def test_composite_identifier_column_created():
    df = pd.DataFrame(
        {
            "sentence_id": [1, 2, 3],
            "doc_id": ["docA", None, "docC"],
            "text": ["alpha", "beta", "gamma"],
        }
    )

    annotator = LLMAnnotator(settings=MagicMock())
    column_name = "sentence_id+doc_id"

    resolved = annotator._resolve_identifier_column(df, column_name)

    assert resolved == column_name
    assert column_name in df.columns
    assert list(df[column_name]) == ["1|docA", "2|__MISSING__", "3|docC"]


def test_composite_identifier_missing_source_falls_back():
    df = pd.DataFrame(
        {
            "sentence_id": [10, 20],
            "text": ["delta", "epsilon"],
        }
    )

    annotator = LLMAnnotator(settings=MagicMock())

    resolved = annotator._resolve_identifier_column(df, "sentence_id+doc_id")

    assert resolved == "llm_annotation_id"
    assert "llm_annotation_id" in df.columns
    assert list(df["llm_annotation_id"]) == [1, 2]

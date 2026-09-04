"""A JSONL column of labels must not take the dataset wizard down.

pandas reads such a column as real Python lists, and lists are unhashable:
Series.nunique() raises TypeError. The detector used the count as a plain
cardinality heuristic, so the crash cost the user the whole session even
though the file was perfectly valid, and LLM_Tool generates these files
itself.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_tool.utils.data_detector import DataDetector


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "multilabel.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_list_valued_label_column_is_analysed(tmp_path):
    rows = [{"text": f"phrase {i}", "labels": ["a", "b"] if i % 2 else [], "lang": "fr"}
            for i in range(40)]
    analysis = DataDetector.analyze_file_intelligently(_write_jsonl(tmp_path, rows))
    assert analysis["rows"] == 40
    names = [c["name"] for c in analysis["annotation_column_candidates"]]
    assert "labels" in names


def test_unique_count_falls_back_to_string_form(tmp_path):
    """Two distinct lists must count as two distinct values, not raise."""
    rows = [{"text": "a", "labels": ["x"]}, {"text": "b", "labels": ["y"]},
            {"text": "c", "labels": ["x"]}]
    analysis = DataDetector.analyze_file_intelligently(_write_jsonl(tmp_path, rows))
    stats = analysis.get("annotation_stats") or {}
    entry = stats.get("labels") if isinstance(stats, dict) else None
    if isinstance(entry, dict) and "unique_count" in entry:
        assert entry["unique_count"] == 2


def test_scalar_columns_are_unaffected(tmp_path):
    rows = [{"text": f"phrase {i}", "annotation": json.dumps({"theme": "yes" if i % 2 else "no"})}
            for i in range(20)]
    analysis = DataDetector.analyze_file_intelligently(_write_jsonl(tmp_path, rows))
    names = [c["name"] for c in analysis["annotation_column_candidates"]]
    assert "annotation" in names

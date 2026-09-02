"""Per-head datasets must not absorb out-of-scope rows as negatives.

An upstream builder expresses per-head class balancing by OMITTING a key
from a row's annotation: the row is out of scope for that head, not a
negative example of it. Keeping those rows silently rebalances the
classes, and the damage is invisible in the reported metrics because the
model still converges -- onto the majority class.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_tool.utils.annotation_to_training import AnnotationToTrainingConverter


def _wave(tmp_path):
    """A wave where head_a is balanced 2:4 and most rows are out of scope."""
    rows = []
    for i in range(2):
        rows.append({"text": f"positif {i}", "annotation": json.dumps({"head_a": "yes"})})
    for i in range(4):
        rows.append({"text": f"negatif {i}", "annotation": json.dumps({"head_a": "no"})})
    for i in range(94):
        rows.append({"text": f"hors perimetre {i}", "annotation": json.dumps({"head_b": "yes"})})
    csv = tmp_path / "wave.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_single_key_file_drops_out_of_scope_rows(tmp_path):
    out = tmp_path / "head_a.jsonl"
    AnnotationToTrainingConverter().create_multi_label_dataset(
        csv_path=str(_wave(tmp_path)), output_path=str(out),
        text_column="text", annotation_column="annotation",
        annotation_keys=["head_a"], label_strategy="key_value",
    )
    samples = _read(out)
    assert len(samples) == 6, "only the rows annotated for head_a belong in its dataset"
    assert not [s for s in samples if not s["labels"]], "no unlabelled row may survive"
    labels = [lbl for s in samples for lbl in s["labels"]]
    assert labels.count("head_a_yes") == 2
    assert labels.count("head_a_no") == 4


def test_multi_key_file_keeps_empty_label_rows(tmp_path):
    """A genuine multi-label file still means 'none of these apply' by [] ."""
    out = tmp_path / "both.jsonl"
    AnnotationToTrainingConverter().create_multi_label_dataset(
        csv_path=str(_wave(tmp_path)), output_path=str(out),
        text_column="text", annotation_column="annotation",
        annotation_keys=["head_a", "head_b"], label_strategy="key_value",
    )
    assert len(_read(out)) == 100, "multi-key files keep every annotated row"


def test_drop_unlabelled_can_be_forced(tmp_path):
    out = tmp_path / "forced.jsonl"
    AnnotationToTrainingConverter().create_multi_label_dataset(
        csv_path=str(_wave(tmp_path)), output_path=str(out),
        text_column="text", annotation_column="annotation",
        annotation_keys=["head_a", "head_b"], label_strategy="key_value",
        drop_unlabelled=True,
    )
    assert all(s["labels"] for s in _read(out))


def test_class_ratio_is_preserved(tmp_path):
    """The regression this guards: 1:2 intended, 1:49 delivered."""
    out = tmp_path / "ratio.jsonl"
    AnnotationToTrainingConverter().create_multi_label_dataset(
        csv_path=str(_wave(tmp_path)), output_path=str(out),
        text_column="text", annotation_column="annotation",
        annotation_keys=["head_a"], label_strategy="key_value",
    )
    samples = _read(out)
    pos = sum(1 for s in samples if "head_a_yes" in s["labels"])
    neg = len(samples) - pos
    assert neg / pos == pytest.approx(2.0), f"expected 1:2, got 1:{neg/pos:.1f}"

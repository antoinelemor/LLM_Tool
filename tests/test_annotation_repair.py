#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_annotation_repair.py

MAIN OBJECTIVE:
---------------
Cover the repair path for annotation files written before the key ordering
fix. The critical property is that repairing changes the key order and nothing
else: every value, every column and every row must survive untouched.

Dependencies:
-------------
- csv
- importlib
- json
- pathlib
- pytest

MAIN FEATURES:
--------------
1) Verify a scrambled CSV column is restored to the declared order
2) Verify no value, column or row is altered by the repair
3) Verify already-ordered files are reported as unchanged
4) Verify Doccano JSONL meta blocks and label lists are realigned
5) Verify in-place repair keeps a backup of the source

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = _load_module("annotation_repair", "llm_tool/utils/annotation_repair.py")


DECLARED = [
    "is_immigration",
    "jurisdictional_position",
    "immigration_policy",
    "secondary_immigration_policy",
    "direction",
    "dominant_frame",
    "secondary_frame",
    "tone",
    "secondary_tone",
]

# The exact scrambled order observed in the affected run.
SCRAMBLED = [
    "secondary_immigration_policy",
    "immigration_policy",
    "direction",
    "dominant_frame",
    "is_immigration",
    "secondary_frame",
    "tone",
    "jurisdictional_position",
    "secondary_tone",
]

VALUES = {
    "is_immigration": "yes",
    "jurisdictional_position": "na",
    "immigration_policy": "volume",
    "secondary_immigration_policy": "origins",
    "direction": "neutral",
    "dominant_frame": "administrative",
    "secondary_frame": "na",
    "tone": "neutral",
    "secondary_tone": "na",
}


def _scrambled_payload() -> dict:
    return {key: VALUES[key] for key in SCRAMBLED}


def _write_csv(path: Path, rows: int = 3) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sentence_uid", "text", "annotation", "score"]
        )
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "sentence_uid": f"uid-{index}",
                    "text": f"Phrase {index}, avec une virgule et des accents é à ü.",
                    "annotation": json.dumps(_scrambled_payload(), ensure_ascii=False),
                    "score": "0.999",
                }
            )


# ── CSV repair ────────────────────────────────────────────────────


def test_csv_repair_restores_declared_order(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    report = repair.repair_csv(source, DECLARED)

    rows = list(csv.DictReader(Path(report["output"]).open(encoding="utf-8")))
    for row in rows:
        assert list(json.loads(row["annotation"])) == DECLARED


def test_csv_repair_preserves_every_value(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    before = list(csv.DictReader(source.open(encoding="utf-8")))

    report = repair.repair_csv(source, DECLARED)
    after = list(csv.DictReader(Path(report["output"]).open(encoding="utf-8")))

    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert old.keys() == new.keys()
        for column in old:
            if column == "annotation":
                assert json.loads(old[column]) == json.loads(new[column])
            else:
                assert old[column] == new[column]


def test_csv_repair_reports_counts_and_orders(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source, rows=5)
    report = repair.repair_csv(source, DECLARED)

    assert report["rows"] == 5
    assert report["rows_reordered"] == 5
    assert report["order_before"] == SCRAMBLED
    assert report["order_after"] == DECLARED


def test_csv_repair_leaves_source_untouched_by_default(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    original = source.read_text(encoding="utf-8")

    repair.repair_csv(source, DECLARED)

    assert source.read_text(encoding="utf-8") == original


def test_csv_repair_is_idempotent(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    first = repair.repair_csv(source, DECLARED)

    second = repair.repair_csv(Path(first["output"]), DECLARED)

    assert second["rows_reordered"] == 0


def test_csv_repair_in_place_keeps_a_backup(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    original = source.read_text(encoding="utf-8")

    report = repair.repair_csv(source, DECLARED, in_place=True)

    assert report["output"] == str(source)
    backup = tmp_path / "annotations.csv.bak"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original
    assert list(json.loads(next(csv.DictReader(source.open(encoding="utf-8")))["annotation"])) == DECLARED


def test_csv_repair_rejects_unknown_column(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    with pytest.raises(ValueError, match="not found"):
        repair.repair_csv(source, DECLARED, column="absente")


def test_csv_repair_skips_unparseable_payloads(tmp_path):
    source = tmp_path / "annotations.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation"])
        writer.writeheader()
        writer.writerow({"annotation": "pas du json"})
        writer.writerow({"annotation": ""})
    report = repair.repair_csv(source, DECLARED)

    rows = list(csv.DictReader(Path(report["output"]).open(encoding="utf-8")))
    assert [row["annotation"] for row in rows] == ["pas du json", ""]
    assert report["rows_reordered"] == 0


def test_csv_repair_keeps_undeclared_keys(tmp_path):
    source = tmp_path / "annotations.csv"
    payload = _scrambled_payload()
    payload["extra_field"] = "kept"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation"])
        writer.writeheader()
        writer.writerow({"annotation": json.dumps(payload, ensure_ascii=False)})

    report = repair.repair_csv(source, DECLARED)
    row = next(csv.DictReader(Path(report["output"]).open(encoding="utf-8")))
    assert list(json.loads(row["annotation"])) == DECLARED + ["extra_field"]


def test_csv_repair_does_not_invent_missing_keys(tmp_path):
    source = tmp_path / "annotations.csv"
    partial = {"tone": "neutral", "is_immigration": "yes"}
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation"])
        writer.writeheader()
        writer.writerow({"annotation": json.dumps(partial)})

    report = repair.repair_csv(source, DECLARED)
    row = next(csv.DictReader(Path(report["output"]).open(encoding="utf-8")))
    assert list(json.loads(row["annotation"])) == ["is_immigration", "tone"]


# ── JSONL repair ──────────────────────────────────────────────────


def _write_jsonl(path: Path) -> None:
    entry = {
        "id": 1,
        "text": "Une phrase.",
        "label": [f"{key}_{VALUES[key]}" for key in SCRAMBLED],
        "meta": {
            "annotation_json": _scrambled_payload(),
            "llm_annotations": {
                "raw": _scrambled_payload(),
                "labels": [f"{key}_{VALUES[key]}" for key in SCRAMBLED],
            },
        },
    }
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def test_jsonl_repair_reorders_both_meta_blocks(tmp_path):
    source = tmp_path / "export.jsonl"
    _write_jsonl(source)

    report = repair.repair_jsonl(source, DECLARED)
    entry = json.loads(Path(report["output"]).read_text(encoding="utf-8").strip())

    assert list(entry["meta"]["annotation_json"]) == DECLARED
    assert list(entry["meta"]["llm_annotations"]["raw"]) == DECLARED


def test_jsonl_repair_reorders_flat_labels(tmp_path):
    source = tmp_path / "export.jsonl"
    _write_jsonl(source)

    report = repair.repair_jsonl(source, DECLARED)
    entry = json.loads(Path(report["output"]).read_text(encoding="utf-8").strip())

    assert entry["label"] == [f"{key}_{VALUES[key]}" for key in DECLARED]


def test_jsonl_label_reordering_binds_longest_key(tmp_path):
    """secondary_tone_na must bind to secondary_tone, never to tone."""
    ordered = repair._reorder_labels(
        ["secondary_tone_na", "tone_neutral"], ["tone", "secondary_tone"]
    )
    assert ordered == ["tone_neutral", "secondary_tone_na"]


def test_jsonl_repair_keeps_unmatched_labels_last(tmp_path):
    ordered = repair._reorder_labels(
        ["mystere", "tone_neutral"], ["tone", "secondary_tone"]
    )
    assert ordered == ["tone_neutral", "mystere"]


def test_jsonl_repair_preserves_text_and_id(tmp_path):
    source = tmp_path / "export.jsonl"
    _write_jsonl(source)

    report = repair.repair_jsonl(source, DECLARED)
    entry = json.loads(Path(report["output"]).read_text(encoding="utf-8").strip())

    assert entry["id"] == 1
    assert entry["text"] == "Une phrase."
    assert report["entries"] == 1
    assert report["entries_reordered"] == 1


# ── CLI ───────────────────────────────────────────────────────────


def test_cli_accepts_explicit_keys(tmp_path, capsys):
    source = tmp_path / "annotations.csv"
    _write_csv(source)

    exit_code = repair.main([str(source), "--keys", ",".join(DECLARED)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Rewrote 3 of 3 records." in captured
    output = tmp_path / "annotations_reordered.csv"
    row = next(csv.DictReader(output.open(encoding="utf-8")))
    assert list(json.loads(row["annotation"])) == DECLARED


def test_cli_derives_order_from_prompt(tmp_path, capsys):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    prompt = tmp_path / "prompt.txt"
    schema = json.dumps({key: "" for key in DECLARED}, indent=2)
    prompt.write_text(
        f"Annotate the sentence.\n\n**Expected JSON:**\n{schema}\n", encoding="utf-8"
    )

    exit_code = repair.main([str(source), "--prompt", str(prompt)])

    assert exit_code == 0
    output = tmp_path / "annotations_reordered.csv"
    row = next(csv.DictReader(output.open(encoding="utf-8")))
    assert list(json.loads(row["annotation"])) == DECLARED


def test_cli_rejects_prompt_without_keys(tmp_path):
    source = tmp_path / "annotations.csv"
    _write_csv(source)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Aucune structure JSON ici.", encoding="utf-8")

    with pytest.raises(ValueError, match="No JSON keys"):
        repair.main([str(source), "--prompt", str(prompt)])

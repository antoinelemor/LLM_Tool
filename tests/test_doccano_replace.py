#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_doccano_replace.py

MAIN OBJECTIVE:
---------------
Cover the offline half of the Doccano replacement tool: turning a repaired
annotation CSV into push payloads. Push order and key order together decide
the order Doccano creates its category types, so both are asserted here.
Network calls are exercised against a stubbed transport, never a live server.

Dependencies:
-------------
- csv
- importlib
- json
- pathlib
- pytest

MAIN FEATURES:
--------------
1) Verify payload annotations come out in canonical key order
2) Verify row order is preserved, since it drives category creation order
3) Verify remaining columns are carried as metadata and empties dropped
4) Verify missing required columns are rejected
5) Verify a backup refuses to silently produce an empty file

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


replace = _load_module("doccano_replace", "llm_tool/utils/doccano_replace.py")


DECLARED = [
    "is_immigration",
    "jurisdictional_position",
    "immigration_policy",
    "direction",
    "tone",
]

SCRAMBLED = ["tone", "direction", "is_immigration", "immigration_policy", "jurisdictional_position"]

VALUES = {
    "is_immigration": "yes",
    "jurisdictional_position": "na",
    "immigration_policy": "volume",
    "direction": "neutral",
    "tone": "neutral",
}


def _write_csv(path: Path, rows: int = 3) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sentence_uid", "text", "annotation", "party", "vide"]
        )
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "sentence_uid": f"uid-{index}",
                    "text": f"Phrase {index}",
                    "annotation": json.dumps({k: VALUES[k] for k in SCRAMBLED}),
                    "party": "CAQ",
                    "vide": "",
                }
            )


def test_rows_carry_annotation_in_canonical_order(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source)

    rows = replace.rows_from_csv(source, DECLARED, text_column="text")

    for row in rows:
        assert list(row["annotation"]) == DECLARED


def test_rows_preserve_file_order(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source, rows=4)

    rows = replace.rows_from_csv(source, DECLARED, text_column="text")

    assert [row["text"] for row in rows] == [f"Phrase {i}" for i in range(4)]


def test_rows_carry_other_columns_as_meta(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source)

    rows = replace.rows_from_csv(source, DECLARED, text_column="text")

    assert rows[0]["meta"]["sentence_uid"] == "uid-0"
    assert rows[0]["meta"]["party"] == "CAQ"


def test_rows_drop_empty_meta_values(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source)

    rows = replace.rows_from_csv(source, DECLARED, text_column="text")

    assert "vide" not in rows[0]["meta"]


def test_rows_accept_explicit_meta_columns(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source)

    rows = replace.rows_from_csv(
        source, DECLARED, text_column="text", meta_columns=["sentence_uid"]
    )

    assert set(rows[0]["meta"]) == {"sentence_uid"}


def test_rows_skip_blank_texts(tmp_path):
    source = tmp_path / "repaired.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "annotation"])
        writer.writeheader()
        writer.writerow({"text": "   ", "annotation": "{}"})
        writer.writerow({"text": "Gardée", "annotation": "{}"})

    rows = replace.rows_from_csv(source, DECLARED, text_column="text")

    assert [row["text"] for row in rows] == ["Gardée"]


def test_rows_reject_missing_text_column(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source)

    with pytest.raises(ValueError, match="not found"):
        replace.rows_from_csv(source, DECLARED, text_column="absente")


def test_rows_reject_missing_annotation_column(tmp_path):
    source = tmp_path / "repaired.csv"
    _write_csv(source)

    with pytest.raises(ValueError, match="not found"):
        replace.rows_from_csv(
            source, DECLARED, text_column="text", annotation_column="absente"
        )


def test_rows_tolerate_unparseable_annotation(tmp_path):
    source = tmp_path / "repaired.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "annotation"])
        writer.writeheader()
        writer.writerow({"text": "Phrase", "annotation": "pas du json"})

    rows = replace.rows_from_csv(source, DECLARED, text_column="text")

    assert rows[0]["annotation"] == {}


# ── Backup guard ──────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_backup_writes_every_example(tmp_path, monkeypatch):
    examples = [{"id": i, "text": f"t{i}", "meta": {}} for i in range(3)]

    def fake_get(url, headers=None, params=None, timeout=None):
        if params and params.get("offset", 0) > 0:
            return _FakeResponse({"results": []})
        return _FakeResponse({"results": examples})

    monkeypatch.setattr(replace.requests, "get", fake_get)
    target = tmp_path / "backup.json"

    report = replace.backup_project("http://x", "tok", 1, target)

    saved = json.loads(target.read_text(encoding="utf-8"))
    assert report["example_count"] == 3
    assert saved["examples"] == examples
    assert saved["project_id"] == 1


def test_backup_records_zero_examples_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        replace.requests,
        "get",
        lambda *a, **k: _FakeResponse({"results": []}),
    )
    target = tmp_path / "backup.json"

    report = replace.backup_project("http://x", "tok", 1, target)

    assert report["example_count"] == 0
    assert target.exists()


def _stub_count(monkeypatch, count: int) -> None:
    monkeypatch.setattr(
        replace.requests,
        "get",
        lambda *a, **k: _FakeResponse({"count": count}),
    )


def test_clear_project_deletes_examples_then_label_types(monkeypatch):
    calls = []
    _stub_count(monkeypatch, 7)

    def fake_delete(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse({"deleted": 7})

    monkeypatch.setattr(replace.requests, "delete", fake_delete)

    result = replace.clear_project("http://x", "tok", 42)

    assert calls == [
        "http://x/doccano/projects/42/examples",
        "http://x/doccano/projects/42/label-types",
    ]
    assert result == {"examples_deleted": 7, "label_types_deleted": 7}


def test_clear_project_can_keep_label_types(monkeypatch):
    calls = []
    _stub_count(monkeypatch, 1)
    monkeypatch.setattr(
        replace.requests,
        "delete",
        lambda url, headers=None, timeout=None: (
            calls.append(url) or _FakeResponse({"deleted": 1})
        ),
    )

    result = replace.clear_project("http://x", "tok", 42, drop_label_types=False)

    assert calls == ["http://x/doccano/projects/42/examples"]
    assert result["label_types_deleted"] == 0


def test_clear_project_skips_example_delete_when_already_empty(monkeypatch):
    """An empty project must not trigger the Telegram confirmation prompt."""
    calls = []
    _stub_count(monkeypatch, 0)
    monkeypatch.setattr(
        replace.requests,
        "delete",
        lambda url, headers=None, timeout=None: (
            calls.append(url) or _FakeResponse({"deleted": 3})
        ),
    )

    result = replace.clear_project("http://x", "tok", 42)

    assert calls == ["http://x/doccano/projects/42/label-types"]
    assert result["examples_deleted"] == 0


def test_push_rows_batches_and_preserves_order(monkeypatch):
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.extend(item["text"] for item in json["items"])
        return _FakeResponse({"pushed": len(json["items"])})

    monkeypatch.setattr(replace.requests, "post", fake_post)
    rows = [{"text": f"t{i}", "annotation": {}, "meta": {}} for i in range(5)]

    result = replace.push_rows("http://x", "tok", 1, rows, batch_size=2)

    assert seen == [f"t{i}" for i in range(5)]
    assert result["pushed"] == 5
    assert result["failures"] == []


def test_push_rows_reports_failures(monkeypatch):
    class _Bad(_FakeResponse):
        def __init__(self):
            super().__init__({})
            self.status_code = 500
            self.text = "boom"

    monkeypatch.setattr(replace.requests, "post", lambda *a, **k: _Bad())
    rows = [{"text": "t", "annotation": {}, "meta": {}}]

    result = replace.push_rows("http://x", "tok", 1, rows)

    assert result["pushed"] == 0
    assert len(result["failures"]) == 1

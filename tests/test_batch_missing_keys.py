#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_batch_missing_keys.py

MAIN OBJECTIVE:
---------------
Guard the rule that decides which OpenAI batch answers are re-run
synchronously. The cleanup pass used to treat a key returned as null, "" or
[] as missing, while LLM Tool's own prompts ask for null when a category does
not apply. On a scheme with optional keys it therefore re-ran nearly every
valid batch answer one by one, at the standard price: 5,577 of 5,578 answers
on a real corpus, where none was incomplete. Only a key the model did not
return is missing.

Dependencies:
-------------
- importlib
- pathlib
- re

MAIN FEATURES:
--------------
1) A key returned as null, "" or [] counts as answered
2) An absent key is missing, reported in declared order
3) A payload that is not a mapping lacks every expected key
4) A real answer with many nulls needs no retry
5) The batch cleanup relies on this rule, not on the values of the keys

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    """Load a module by path, bypassing the package __init__ and its heavy deps."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


key_order = _load_module("key_order", "llm_tool/utils/key_order.py")
missing_keys = key_order.missing_keys

EXPECTED = ["theme", "party", "sentiment", "actors"]


def test_null_empty_and_empty_list_are_answers():
    payload = {"theme": None, "party": "", "sentiment": [], "actors": None}
    assert missing_keys(payload, EXPECTED) == []


def test_absent_keys_are_missing_in_declared_order():
    payload = {"party": "LPC", "theme": None}
    assert missing_keys(payload, EXPECTED) == ["sentiment", "actors"]


def test_non_mapping_payload_lacks_every_key():
    assert missing_keys(None, EXPECTED) == EXPECTED
    assert missing_keys(["theme"], EXPECTED) == EXPECTED


def test_expected_keys_are_deduplicated():
    assert missing_keys({}, ["theme", "theme", None, "tone"]) == ["theme", "tone"]


def test_real_answer_with_optional_keys_needs_no_retry():
    # A valid answer of a gated scheme: the unit mentions no AI, so every
    # gated key is legitimately null.
    answer = {"ai_mention": "none", "actors": None, "ai_uses": None, "ai_uses_other": None,
              "ai_tools": None, "frame": None, "tone": None, "cities": None}
    assert missing_keys(answer, list(answer)) == []


def test_batch_cleanup_uses_key_presence_not_values():
    source = (REPO_ROOT / "llm_tool/annotators/llm_annotator.py").read_text(encoding="utf-8")
    start = source.index("cleanup_enabled = config.get('openai_batch_cleanup_missing'")
    block = source[start:start + 2500]
    assert "missing_keys(merged_payload, expected_keys)" in block
    assert not re.search(r"in \(None, '', \[\]\)", block)

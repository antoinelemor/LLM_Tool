#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
test_key_order.py

MAIN OBJECTIVE:
---------------
Guard the canonical ordering of annotation keys. Annotation keys used to be
collected in Python sets, whose iteration order derives from randomised string
hashing, so every run scrambled the JSON column order differently. These tests
pin the order to the prompt declaration for every prompt format the extractor
supports, and prove the ordering survives a change of hash seed.

Dependencies:
-------------
- importlib
- json
- os
- pathlib
- subprocess
- sys
- pytest

MAIN FEATURES:
--------------
1) Verify ordered deduplication and prefix application
2) Verify the canonical order built from single and multi prompt setups
3) Verify payload reordering, including keys outside the declared schema
4) Verify every supported prompt format yields the declared key order
5) Verify the order is stable across interpreter hash seeds

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    """Load a module by path, bypassing the package __init__ and its heavy deps."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


key_order = _load_module("key_order", "llm_tool/utils/key_order.py")
json_cleaner = _load_module("json_cleaner", "llm_tool/annotators/json_cleaner.py")

ordered_unique = key_order.ordered_unique
prefixed_keys = key_order.prefixed_keys
canonical_key_order = key_order.canonical_key_order
reorder_payload = key_order.reorder_payload
extract_expected_keys = json_cleaner.extract_expected_keys


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


# ── ordered_unique ────────────────────────────────────────────────


def test_ordered_unique_preserves_first_appearance():
    assert ordered_unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_ordered_unique_drops_empty_and_none():
    assert ordered_unique(["a", None, "", "b"]) == ["a", "b"]


def test_ordered_unique_handles_none_iterable():
    assert ordered_unique(None) == []


def test_ordered_unique_accepts_generators():
    assert ordered_unique(k for k in ["x", "y", "x"]) == ["x", "y"]


# ── prefixed_keys ─────────────────────────────────────────────────


def test_prefixed_keys_without_prefix():
    assert prefixed_keys(["a", "b"], "") == ["a", "b"]
    assert prefixed_keys(["a", "b"], None) == ["a", "b"]


def test_prefixed_keys_with_prefix_preserves_order():
    assert prefixed_keys(["a", "b"], "p") == ["p_a", "p_b"]


# ── canonical_key_order ───────────────────────────────────────────


def test_canonical_order_single_prompt_matches_declaration():
    prompts = [{"expected_keys": DECLARED, "prefix": ""}]
    assert canonical_key_order(prompts) == DECLARED


def test_canonical_order_multi_prompt_concatenates_in_prompt_order():
    prompts = [
        {"expected_keys": ["topic", "tone"], "prefix": "a"},
        {"expected_keys": ["stance"], "prefix": "b"},
    ]
    assert canonical_key_order(prompts) == ["a_topic", "a_tone", "b_stance"]


def test_canonical_order_deduplicates_across_prompts():
    prompts = [
        {"expected_keys": ["topic", "tone"]},
        {"expected_keys": ["tone", "stance"]},
    ]
    assert canonical_key_order(prompts) == ["topic", "tone", "stance"]


def test_canonical_order_empty_without_declared_keys():
    assert canonical_key_order([{"prefix": "a"}]) == []
    assert canonical_key_order(None) == []


def test_canonical_order_appends_extra_keys_last():
    prompts = [{"expected_keys": ["topic"]}]
    assert canonical_key_order(prompts, extra_keys=["z"]) == ["topic", "z"]


# ── reorder_payload ───────────────────────────────────────────────


def test_reorder_payload_restores_declared_order():
    scrambled = {k: "na" for k in reversed(DECLARED)}
    assert list(reorder_payload(scrambled, DECLARED)) == DECLARED


def test_reorder_payload_keeps_values_attached_to_their_keys():
    scrambled = {"tone": "negative", "is_immigration": "yes"}
    result = reorder_payload(scrambled, ["is_immigration", "tone"])
    assert result == {"is_immigration": "yes", "tone": "negative"}


def test_reorder_payload_fills_missing_keys_with_none():
    result = reorder_payload({"tone": "neutral"}, ["is_immigration", "tone"])
    assert result == {"is_immigration": None, "tone": "neutral"}


def test_reorder_payload_can_skip_missing_keys():
    result = reorder_payload(
        {"tone": "neutral"}, ["is_immigration", "tone"], include_missing=False
    )
    assert result == {"tone": "neutral"}


def test_reorder_payload_appends_undeclared_keys_last():
    payload = {"surprise": 1, "tone": "neutral"}
    result = reorder_payload(payload, ["tone"])
    assert list(result) == ["tone", "surprise"]


def test_reorder_payload_without_order_preserves_payload_order():
    payload = {"b": 1, "a": 2}
    assert list(reorder_payload(payload, [])) == ["b", "a"]


def test_reorder_payload_handles_none_payload():
    assert reorder_payload(None, ["a"]) == {"a": None}


# ── prompt formats ────────────────────────────────────────────────

SCHEMA_BLOCK = json.dumps({k: "" for k in DECLARED}, indent=2)

PROMPT_FORMATS = {
    "expected_json_header": f"Annotate the sentence.\n\n**Expected JSON:**\n{SCHEMA_BLOCK}\n",
    "expected_json_keys_header": f"Annotate the sentence.\n\nExpected JSON Keys\n\n{SCHEMA_BLOCK}\n",
    "json_structure_header": f"Annotate.\n\n## JSON Structure\n\n```json\n{SCHEMA_BLOCK}\n```\n",
    "output_format_header": f"Annotate.\n\nOutput Format:\n{SCHEMA_BLOCK}\n",
    "response_format_header": f"Annotate.\n\nResponse Format:\n{SCHEMA_BLOCK}\n",
    "bare_schema_block": f"Return this object:\n\n{SCHEMA_BLOCK}\n",
    "fenced_schema_no_header": f"Return:\n\n```json\n{SCHEMA_BLOCK}\n```\n",
    "inline_key_definitions": "Return a JSON object with:\n"
    + "\n".join(f'- "{k}": "<value>"' for k in DECLARED),
    "key_list": "Use the following keys: ["
    + ", ".join(f'"{k}"' for k in DECLARED)
    + "]",
}


@pytest.mark.parametrize("fmt_name", sorted(PROMPT_FORMATS))
def test_every_prompt_format_yields_declared_order(fmt_name):
    """Whatever the prompt format, the extracted keys follow the declaration."""
    keys = extract_expected_keys(PROMPT_FORMATS[fmt_name])
    assert keys == DECLARED, f"format {fmt_name} returned {keys}"


@pytest.mark.parametrize("fmt_name", sorted(PROMPT_FORMATS))
def test_prompt_format_feeds_canonical_order(fmt_name):
    """The extracted order survives the trip through the prompt configuration."""
    keys = extract_expected_keys(PROMPT_FORMATS[fmt_name])
    prompts = [{"expected_keys": keys, "prefix": ""}]
    assert canonical_key_order(prompts) == DECLARED


def test_schema_block_wins_over_annotated_examples():
    """A prompt carrying worked examples still keys off the declared schema."""
    example = json.dumps(
        {k: "na" for k in reversed(DECLARED)}, indent=2
    )
    prompt = (
        "## Expected JSON Structure\n\n"
        f"```json\n{SCHEMA_BLOCK}\n```\n\n"
        "### Example 1\n\n"
        f"```json\n{example}\n```\n"
    )
    assert extract_expected_keys(prompt) == DECLARED


# ── hash seed independence ────────────────────────────────────────

_SEED_SCRIPT = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("key_order", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
declared = json.loads(sys.argv[2])
prompts = [{"expected_keys": declared, "prefix": ""}]
scrambled = {k: "na" for k in reversed(declared)}
order = m.canonical_key_order(prompts)
print(json.dumps(list(m.reorder_payload(scrambled, order))))
"""


@pytest.mark.parametrize("seed", ["0", "1", "42", "12345"])
def test_order_is_stable_across_hash_seeds(seed, tmp_path):
    """The whole point: a different PYTHONHASHSEED must not change the order.

    Before the fix, the annotator collected these keys in a set, so this test
    would have produced a different permutation for each seed.
    """
    script = tmp_path / "probe.py"
    script.write_text(_SEED_SCRIPT, encoding="utf-8")
    env = dict(os.environ, PYTHONHASHSEED=seed)
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(REPO_ROOT / "llm_tool/utils/key_order.py"),
            json.dumps(DECLARED),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(proc.stdout) == DECLARED


def test_set_based_collection_would_have_broken_the_order():
    """Document the regression this module exists to prevent.

    A set of these nine keys iterates in hash order. Asserting that at least one
    interpreter disagrees with the declared order would be flaky on a single
    seed, so instead we assert the ordered helper never depends on hashing.
    """
    from_set = list(set(DECLARED))
    assert sorted(from_set) == sorted(DECLARED)
    assert ordered_unique(DECLARED) == DECLARED

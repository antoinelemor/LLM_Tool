#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
key_order.py

MAIN OBJECTIVE:
---------------
Preserve the canonical ordering of annotation keys end to end, from the order
declared in the prompt to the JSON written in the output files and pushed to
Doccano. Python sets iterate in an order derived from string hashing, which is
randomised per interpreter run (PYTHONHASHSEED). Any set used to collect
annotation keys therefore scrambles the column order in a way that is stable
within a run but arbitrary across runs. This module provides the ordered,
deduplicated primitives that replace those sets.

Dependencies:
-------------
- typing

MAIN FEATURES:
--------------
1) Deduplicate any iterable of keys while preserving first-appearance order
2) Build the canonical key order from a list of prompt configurations
3) Apply prompt prefixes without losing the declared ordering
4) Reorder an annotation payload onto the canonical order
5) Keep unexpected keys returned by the model, appended in arrival order
6) List the expected keys an answer lacks, a null value counting as an answer

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


__all__ = [
    "ordered_unique",
    "prefixed_keys",
    "canonical_key_order",
    "reorder_payload",
    "missing_keys",
]


def ordered_unique(items: Optional[Iterable[Any]]) -> List[str]:
    """Deduplicate an iterable of keys while preserving first-appearance order.

    This is the ordered replacement for ``set(...)`` wherever annotation keys
    are collected. Empty and ``None`` entries are dropped so callers can feed
    raw prompt configurations without pre-filtering.

    Parameters
    ----------
    items : iterable or None
        Keys to deduplicate. ``None`` is treated as an empty iterable.

    Returns
    -------
    list of str
        Unique keys, in the order they were first encountered.
    """
    result: List[str] = []
    seen = set()
    if not items:
        return result
    for item in items:
        if item is None:
            continue
        key = item if isinstance(item, str) else str(item)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def prefixed_keys(keys: Optional[Iterable[Any]], prefix: Optional[str]) -> List[str]:
    """Apply a prompt prefix to each key, preserving the declared order.

    Parameters
    ----------
    keys : iterable or None
        Keys declared by a single prompt, in prompt order.
    prefix : str or None
        Prefix configured for that prompt. Falsy values leave keys untouched.

    Returns
    -------
    list of str
        Prefixed keys, deduplicated, in prompt order.
    """
    normalized = ordered_unique(keys)
    clean_prefix = (prefix or "").strip()
    if not clean_prefix:
        return normalized
    return [f"{clean_prefix}_{key}" for key in normalized]


def canonical_key_order(
    prompts: Optional[Sequence[Mapping[str, Any]]],
    extra_keys: Optional[Iterable[Any]] = None,
) -> List[str]:
    """Build the canonical annotation key order from prompt configurations.

    The order is the concatenation of each prompt's ``expected_keys``, taken in
    prompt order and prefixed where a ``prefix`` is configured. This mirrors the
    order the annotator asks the model to produce, whatever prompt format was
    used upstream: ``extract_expected_keys`` preserves the declaration order for
    every format it supports (JSON schema block, header-anchored block, inline
    ``"key": value`` patterns, or an explicit key list).

    Parameters
    ----------
    prompts : sequence of mapping or None
        Prompt configurations, each optionally carrying ``expected_keys`` and
        ``prefix``.
    extra_keys : iterable or None
        Additional keys appended after the prompt-declared ones, used when a
        caller already knows about keys produced outside the prompts.

    Returns
    -------
    list of str
        Canonical key order. Empty when no prompt declares any key, in which
        case callers must fall back to the payload's own arrival order.
    """
    collected: List[str] = []
    for prompt_cfg in prompts or []:
        if not isinstance(prompt_cfg, Mapping):
            continue
        collected.extend(
            prefixed_keys(prompt_cfg.get("expected_keys"), prompt_cfg.get("prefix"))
        )
    if extra_keys:
        collected.extend(ordered_unique(extra_keys))
    return ordered_unique(collected)


def reorder_payload(
    payload: Optional[Mapping[str, Any]],
    key_order: Optional[Sequence[str]],
    include_missing: bool = True,
) -> Dict[str, Any]:
    """Reorder an annotation payload onto the canonical key order.

    Keys present in ``key_order`` come first, in that order. Keys the model
    returned outside the declared schema are appended afterwards in their own
    arrival order, so nothing is silently dropped.

    Parameters
    ----------
    payload : mapping or None
        Annotation values keyed by annotation key.
    key_order : sequence of str or None
        Canonical order. When empty, the payload's own order is preserved.
    include_missing : bool
        When True, canonical keys absent from the payload are emitted with a
        ``None`` value so every row carries the same key set. When False, only
        keys actually present are emitted.

    Returns
    -------
    dict
        A new dictionary in canonical order.
    """
    source: Dict[str, Any] = dict(payload or {})
    if not key_order:
        return source

    ordered: Dict[str, Any] = {}
    for key in ordered_unique(key_order):
        if key in source:
            ordered[key] = source[key]
        elif include_missing:
            ordered[key] = None
    for key, value in source.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def missing_keys(payload: Optional[Mapping[str, Any]], expected: Optional[Iterable[Any]]) -> List[str]:
    """Return the expected keys that are absent from ``payload``, in declared order.

    A key the model returned with ``null``, ``""`` or ``[]`` is present: it is
    an answer, not a gap. LLM Tool's own prompts ask the model to use ``null``
    when a category does not apply ("Ensure that all keys are present in the
    JSON, using `null` when necessary"), so treating such a value as missing
    would flag nearly every valid answer of a scheme with optional keys.
    Only a key the model did not return at all is missing.
    """
    if not isinstance(payload, Mapping):
        return ordered_unique(expected)
    return [key for key in ordered_unique(expected) if key not in payload]

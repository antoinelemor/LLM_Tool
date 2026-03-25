#!/usr/bin/env python3
"""
Utility helpers to derive compact performance summaries from model metadata.

These helpers extract the macro F1 score, per-class F1 values, and per-language
breakdowns from the rich metadata generated after training. They provide a
consistent structure that CLI surfaces can render without duplicating parsing
logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["load_language_metrics", "summarize_final_metrics"]


def _safe_float(value: Any) -> Optional[float]:
    """Return ``value`` as ``float`` when possible."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: Optional[float]) -> Optional[float]:
    """Return the first value that is not ``None`` (preserves zeros)."""
    for candidate in values:
        if candidate is not None:
            return candidate
    return None


def _coerce_label(entry: Dict[str, Any], fallback_idx: int) -> str:
    """Extract a readable label name from ``entry``."""
    for key in ("label", "class", "name", "category", "id"):
        if key in entry and entry[key]:
            return str(entry[key])
    return f"Class {fallback_idx + 1}"


def _normalise_language_code(code: Any) -> str:
    """Return a normalised language code for display."""
    if isinstance(code, str):
        cleaned = code.strip()
        if cleaned:
            return cleaned.upper()
    return "UNKNOWN"


def load_language_metrics(model_dir: Path) -> Dict[str, Any]:
    """
    Load language-specific metrics saved during training for ``model_dir``.

    Returns an empty dictionary when no metrics are available or readable.
    """
    metrics_path = model_dir / "language_performance.json"
    if not metrics_path.exists():
        return {}

    try:
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            data = json.load(metrics_file)
    except Exception:
        return {}

    if isinstance(data, list) and data:
        latest = data[-1]
        averages = latest.get("averages", {}) or {}
        macro = _first_not_none(
            _safe_float(averages.get("macro_f1")),
            _safe_float(averages.get("f1_macro")),
        )
        per_language: Dict[str, float] = {}
        metrics_by_language = latest.get("metrics", {}) or {}
        if isinstance(metrics_by_language, dict):
            for lang_code, values in metrics_by_language.items():
                if not isinstance(values, dict):
                    continue
                score = _first_not_none(
                    _safe_float(values.get("macro_f1")),
                    _safe_float(values.get("f1_macro")),
                )
                if score is not None:
                    per_language[_normalise_language_code(lang_code)] = score
        return {
            "macro_f1": macro,
            "per_language": per_language,
            "raw": latest,
        }

    return {}


def summarize_final_metrics(
    training_metadata: Optional[Dict[str, Any]],
    language_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a consistent performance summary from training metadata.

    Args:
        training_metadata: Parsed ``training_metadata.json`` contents.
        language_metrics: Optional payload from ``language_performance.json``.

    Returns:
        Dictionary with keys:
            ``macro_f1`` (Optional[float])
            ``per_class`` (List[Tuple[str, Optional[float]]])
            ``per_language`` (Dict[str, float])
    """
    metadata = training_metadata or {}
    final_section = metadata.get("final_metrics")
    if not isinstance(final_section, dict):
        final_section = {}

    overall_section = final_section.get("overall")
    if not isinstance(overall_section, dict):
        overall_section = {}

    macro_candidates: Iterable[Optional[float]] = (
        _safe_float(overall_section.get("macro_f1")),
        _safe_float(overall_section.get("f1_macro")),
        _safe_float(final_section.get("macro_f1")),
        _safe_float(final_section.get("f1_macro")),
        _safe_float(metadata.get("macro_f1")),
        _safe_float(metadata.get("combined_metric")),
    )

    macro_f1: Optional[float] = next((value for value in macro_candidates if value is not None), None)

    per_class: List[Tuple[str, Optional[float]]] = []
    per_class_block = final_section.get("per_class") or metadata.get("per_class") or []
    if isinstance(per_class_block, list):
        for idx, raw_entry in enumerate(per_class_block):
            if not isinstance(raw_entry, dict):
                continue
            label_name = _coerce_label(raw_entry, idx)
            score = _first_not_none(
                _safe_float(raw_entry.get("f1")),
                _safe_float(raw_entry.get("macro_f1")),
                _safe_float(raw_entry.get("f1_macro")),
            )
            per_class.append((label_name, score))

    per_language: Dict[str, float] = {}
    per_language_block = final_section.get("per_language") or metadata.get("per_language")
    if isinstance(per_language_block, dict):
        for lang_code, values in per_language_block.items():
            score = None
            if isinstance(values, dict):
                score = _first_not_none(
                    _safe_float(values.get("macro_f1")),
                    _safe_float(values.get("f1_macro")),
                    _safe_float(values.get("f1")),
                )
            else:
                score = _safe_float(values)
            if score is not None:
                per_language[_normalise_language_code(lang_code)] = score
    elif isinstance(per_language_block, list):
        for entry in per_language_block:
            if not isinstance(entry, dict):
                continue
            lang_code = _normalise_language_code(entry.get("language") or entry.get("code"))
            score = _first_not_none(
                _safe_float(entry.get("macro_f1")),
                _safe_float(entry.get("f1_macro")),
                _safe_float(entry.get("f1")),
            )
            if score is not None:
                per_language[lang_code] = score

    if isinstance(language_metrics, dict):
        metric_macro = _safe_float(language_metrics.get("macro_f1"))
        if macro_f1 is None and metric_macro is not None:
            macro_f1 = metric_macro
        lang_breakdown = language_metrics.get("per_language", {})
        if isinstance(lang_breakdown, dict):
            for lang_code, score in lang_breakdown.items():
                score_value = _safe_float(score)
                if score_value is None:
                    continue
                normalised = _normalise_language_code(lang_code)
                # Preserve values from training metadata when already present.
                per_language.setdefault(normalised, score_value)

    return {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "per_language": dict(sorted(per_language.items())),
    }

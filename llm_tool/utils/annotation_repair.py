#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
annotation_repair.py

MAIN OBJECTIVE:
---------------
Repair annotation files whose JSON keys were scrambled by the set-based key
collection that preceded ``llm_tool.utils.key_order``. The annotation values
themselves are correct: only the key order was lost. This module rewrites the
order in place from the prompt declaration, so existing runs are salvaged
without paying for a re-annotation.

Dependencies:
-------------
- argparse
- csv
- json
- pathlib
- shutil
- sys
- typing

MAIN FEATURES:
--------------
1) Derive the canonical key order from a prompt file or an explicit key list
2) Repair a CSV annotation column, preserving every other column untouched
3) Repair a Doccano JSONL export, realigning both labels and annotation meta
4) Report the before and after ordering plus the number of rows rewritten
5) Never overwrite the source file unless explicitly asked

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # package import
    from .key_order import ordered_unique, reorder_payload
except ImportError:  # direct execution by path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from key_order import ordered_unique, reorder_payload  # type: ignore


__all__ = [
    "canonical_order_from_prompt",
    "repair_csv",
    "repair_jsonl",
]


# ── Canonical order sources ───────────────────────────────────────


def canonical_order_from_prompt(prompt_path: Path) -> List[str]:
    """Extract the declared key order from a prompt file.

    Delegates to ``extract_expected_keys``, which handles every supported
    prompt format and preserves the declaration order in each of them.

    Parameters
    ----------
    prompt_path : Path
        Path to the prompt text file used for the annotation run.

    Returns
    -------
    list of str
        Keys in declaration order.

    Raises
    ------
    ValueError
        If no key could be extracted from the prompt.
    """
    try:  # package import
        from ..annotators.json_cleaner import extract_expected_keys
    except ImportError:  # direct execution by path
        import importlib.util

        cleaner_path = (
            Path(__file__).resolve().parent.parent
            / "annotators"
            / "json_cleaner.py"
        )
        spec = importlib.util.spec_from_file_location("json_cleaner", cleaner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extract_expected_keys = module.extract_expected_keys

    text = Path(prompt_path).read_text(encoding="utf-8")
    keys = ordered_unique(extract_expected_keys(text))
    if not keys:
        raise ValueError(
            f"No JSON keys could be extracted from {prompt_path}. "
            "Pass --keys to supply the order explicitly."
        )
    return keys


# ── Repair helpers ────────────────────────────────────────────────


def _repair_json_string(
    raw: Optional[str], key_order: Sequence[str]
) -> Tuple[Optional[str], bool]:
    """Reorder one serialised annotation payload.

    Returns
    -------
    tuple
        The rewritten JSON string and whether the order actually changed.
    """
    if raw is None:
        return raw, False
    text = str(raw).strip()
    if not text:
        return raw, False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return raw, False
    if not isinstance(payload, dict):
        return raw, False

    before = list(payload)
    repaired = reorder_payload(payload, key_order, include_missing=False)
    after = list(repaired)
    if before == after:
        return raw, False
    return json.dumps(repaired, ensure_ascii=False), True


def repair_csv(
    input_path: Path,
    key_order: Sequence[str],
    column: str = "annotation",
    output_path: Optional[Path] = None,
    in_place: bool = False,
) -> Dict[str, Any]:
    """Rewrite the key order of a CSV annotation column.

    Every other column, and every value, is copied verbatim. Only the order of
    the keys inside the JSON column changes.

    Parameters
    ----------
    input_path : Path
        Source CSV produced by an annotation run.
    key_order : sequence of str
        Canonical key order to apply.
    column : str
        Name of the column holding the serialised annotation.
    output_path : Path or None
        Destination. Defaults to ``<input>_reordered.csv`` unless in_place.
    in_place : bool
        Rewrite the source file, after copying it to ``<input>.bak``.

    Returns
    -------
    dict
        Report with the row counts and the observed orders.
    """
    input_path = Path(input_path)
    # A leading byte order mark would otherwise be glued to the first column
    # name, which breaks the lookup when the annotation column comes first.
    # It is written back only if the source carried one.
    raw_head = input_path.open("rb").read(3)
    has_bom = raw_head.startswith(b"\xef\xbb\xbf")
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            raise ValueError(
                f"Column '{column}' not found in {input_path}. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        rows = list(reader)

    order_before: Optional[List[str]] = None
    changed = 0
    for row in rows:
        raw = row.get(column)
        if order_before is None and raw:
            try:
                parsed = json.loads(str(raw))
                if isinstance(parsed, dict):
                    order_before = list(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        repaired, did_change = _repair_json_string(raw, key_order)
        if did_change:
            row[column] = repaired
            changed += 1

    target = _resolve_target(input_path, output_path, in_place, suffix="_reordered")
    with target.open(
        "w", encoding="utf-8-sig" if has_bom else "utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input": str(input_path),
        "output": str(target),
        "rows": len(rows),
        "rows_reordered": changed,
        "order_before": order_before,
        "order_after": list(key_order),
    }


def repair_jsonl(
    input_path: Path,
    key_order: Sequence[str],
    output_path: Optional[Path] = None,
    in_place: bool = False,
) -> Dict[str, Any]:
    """Rewrite the key order of a Doccano JSONL export.

    Reorders ``meta.annotation_json`` and ``meta.llm_annotations.raw`` when
    present, and sorts the flat ``label`` list onto the same canonical order so
    Doccano creates its category types in the declared sequence.

    Parameters
    ----------
    input_path : Path
        Source JSONL export.
    key_order : sequence of str
        Canonical key order to apply.
    output_path : Path or None
        Destination. Defaults to ``<input>_reordered.jsonl`` unless in_place.
    in_place : bool
        Rewrite the source file, after copying it to ``<input>.bak``.

    Returns
    -------
    dict
        Report with the entry counts and the observed orders.
    """
    input_path = Path(input_path)
    entries: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    order_before: Optional[List[str]] = None
    changed = 0
    for entry in entries:
        entry_changed = False
        meta = entry.get("meta")
        if isinstance(meta, dict):
            for path in (("annotation_json",), ("llm_annotations", "raw")):
                holder: Any = meta
                for step in path[:-1]:
                    holder = holder.get(step) if isinstance(holder, dict) else None
                    if holder is None:
                        break
                if not isinstance(holder, dict):
                    continue
                payload = holder.get(path[-1])
                if not isinstance(payload, dict):
                    continue
                if order_before is None:
                    order_before = list(payload)
                repaired = reorder_payload(payload, key_order, include_missing=False)
                if list(repaired) != list(payload):
                    holder[path[-1]] = repaired
                    entry_changed = True

        labels = entry.get("label")
        if isinstance(labels, list) and labels:
            reordered = _reorder_labels(labels, key_order)
            if reordered != labels:
                entry["label"] = reordered
                entry_changed = True

        if entry_changed:
            changed += 1

    target = _resolve_target(input_path, output_path, in_place, suffix="_reordered")
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "input": str(input_path),
        "output": str(target),
        "entries": len(entries),
        "entries_reordered": changed,
        "order_before": order_before,
        "order_after": list(key_order),
    }


def _reorder_labels(labels: Sequence[Any], key_order: Sequence[str]) -> List[Any]:
    """Sort flat ``key_value`` labels onto the canonical key order.

    A label is matched to the longest declared key it starts with, so
    ``secondary_tone_na`` binds to ``secondary_tone`` rather than ``tone``.
    Labels matching no declared key keep their relative order at the end.
    """
    canonical = ordered_unique(key_order)
    position = {key: index for index, key in enumerate(canonical)}
    ranked = sorted(canonical, key=len, reverse=True)

    def rank(label: Any) -> Tuple[int, int]:
        text = str(label)
        for key in ranked:
            if text == key or text.startswith(f"{key}_"):
                return (0, position[key])
        return (1, 0)

    indexed = list(enumerate(labels))
    indexed.sort(key=lambda pair: (rank(pair[1]), pair[0]))
    return [label for _, label in indexed]


def _resolve_target(
    input_path: Path,
    output_path: Optional[Path],
    in_place: bool,
    suffix: str,
) -> Path:
    """Pick the destination path, backing up the source when writing in place."""
    if in_place:
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(input_path, backup)
        return input_path
    if output_path is not None:
        return Path(output_path)
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


# ── CLI ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotation_repair",
        description=(
            "Restore the prompt-declared key order in annotation files produced "
            "before the key ordering fix."
        ),
    )
    parser.add_argument("input", type=Path, help="CSV or JSONL annotation file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompt", type=Path, help="Prompt file the run was annotated with"
    )
    source.add_argument(
        "--keys", help="Comma-separated key order, used instead of --prompt"
    )
    parser.add_argument(
        "--column",
        default="annotation",
        help="CSV column holding the serialised annotation (default: annotation)",
    )
    parser.add_argument("--output", type=Path, help="Destination file")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the source file, keeping a .bak copy alongside it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.keys:
        key_order = ordered_unique(k.strip() for k in args.keys.split(","))
    else:
        key_order = canonical_order_from_prompt(args.prompt)

    if args.input.suffix.lower() == ".jsonl":
        report = repair_jsonl(
            args.input, key_order, output_path=args.output, in_place=args.in_place
        )
        count_key, total_key = "entries_reordered", "entries"
    else:
        report = repair_csv(
            args.input,
            key_order,
            column=args.column,
            output_path=args.output,
            in_place=args.in_place,
        )
        count_key, total_key = "rows_reordered", "rows"

    print(f"Canonical order ({len(key_order)} keys):")
    for index, key in enumerate(key_order, 1):
        print(f"  {index}. {key}")
    if report.get("order_before"):
        print("\nOrder found in the source file:")
        for index, key in enumerate(report["order_before"], 1):
            print(f"  {index}. {key}")
    print(
        f"\nRewrote {report[count_key]} of {report[total_key]} records."
        f"\nWritten to: {report['output']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

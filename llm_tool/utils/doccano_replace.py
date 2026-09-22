#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
doccano_replace.py

MAIN OBJECTIVE:
---------------
Rebuild a Doccano project from a repaired annotation file. A project filled
before the key ordering fix carries both scrambled annotation metadata and
category types created in that scrambled order, and deleting the examples
alone leaves the stale categories behind. This tool backs the project up,
clears examples and category types, then pushes the repaired rows so Doccano
recreates everything in the prompt-declared order.

Dependencies:
-------------
- argparse
- csv
- datetime
- json
- pathlib
- sys
- typing
- requests

MAIN FEATURES:
--------------
1) Back up every example, label and metadata block before touching anything
2) Refuse to delete when the backup could not be written
3) Clear examples and category types in the right order
4) Push repaired rows in batches, preserving the canonical key order
5) Support a dry run that reports the plan without modifying the project

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

try:  # package import
    from .key_order import ordered_unique, reorder_payload
except ImportError:  # direct execution by path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from key_order import ordered_unique, reorder_payload  # type: ignore


__all__ = [
    "backup_project",
    "clear_project",
    "count_examples",
    "push_rows",
    "rows_from_csv",
]

DEFAULT_TIMEOUT = 60


# ── HTTP helpers ──────────────────────────────────────────────────


def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _base(api_url: str) -> str:
    return api_url.rstrip("/")


def count_examples(api_url: str, token: str, project_id: int) -> int:
    """Return how many examples a project currently holds.

    Parameters
    ----------
    api_url : str
        Base URL of the Infer API.
    token : str
        Bearer token.
    project_id : int
        Project to count.

    Returns
    -------
    int
        Number of examples.
    """
    resp = requests.get(
        f"{_base(api_url)}/doccano/projects/{project_id}/examples/count",
        headers=_headers(token),
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return int(resp.json().get("count", 0))


# ── Backup ────────────────────────────────────────────────────────


def backup_project(
    api_url: str,
    token: str,
    project_id: int,
    output_path: Path,
) -> Dict[str, Any]:
    """Write a full snapshot of a project to disk before any destructive step.

    The snapshot holds every example with its text, metadata and annotations,
    so the project can be reconstructed if the replacement goes wrong.

    Parameters
    ----------
    api_url : str
        Base URL of the Infer API exposing the Doccano endpoints.
    token : str
        Bearer token for that API.
    project_id : int
        Project to snapshot.
    output_path : Path
        Destination JSON file.

    Returns
    -------
    dict
        Report with the example count and the snapshot path.

    Raises
    ------
    RuntimeError
        If the snapshot is empty or could not be written.
    """
    base = _base(api_url)
    headers = _headers(token)

    examples: List[dict] = []
    offset = 0
    page_size = 1000
    while True:
        resp = requests.get(
            f"{base}/doccano/projects/{project_id}/examples",
            headers=headers,
            params={"limit": page_size, "offset": offset},
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        page = data.get("results", data if isinstance(data, list) else [])
        examples.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    snapshot = {
        "project_id": project_id,
        "api_url": base,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "example_count": len(examples),
        "examples": examples,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    written = output_path.stat().st_size
    if written == 0:
        raise RuntimeError(f"Backup file {output_path} is empty; aborting.")

    return {
        "path": str(output_path),
        "example_count": len(examples),
        "bytes": written,
    }


# ── Clearing ──────────────────────────────────────────────────────


def clear_project(
    api_url: str,
    token: str,
    project_id: int,
    drop_label_types: bool = True,
) -> Dict[str, Any]:
    """Delete every example, then every category type, from a project.

    Examples go first: removing category types while annotations still
    reference them is rejected by Doccano on some versions.

    Parameters
    ----------
    api_url : str
        Base URL of the Infer API.
    token : str
        Bearer token.
    project_id : int
        Project to clear.
    drop_label_types : bool
        Also delete the project's category types.

    Returns
    -------
    dict
        Counts of deleted examples and category types.
    """
    base = _base(api_url)
    headers = _headers(token)

    # Deleting examples asks the API administrator for a Telegram confirmation,
    # so skip the call entirely when there is nothing to delete.
    remaining = count_examples(api_url, token, project_id)
    deleted_examples = 0
    if remaining:
        resp = requests.delete(
            f"{base}/doccano/projects/{project_id}/examples",
            headers=headers,
            timeout=DEFAULT_TIMEOUT * 5,
        )
        resp.raise_for_status()
        deleted_examples = resp.json().get("deleted", 0)

    deleted_labels = 0
    if drop_label_types:
        resp = requests.delete(
            f"{base}/doccano/projects/{project_id}/label-types",
            headers=headers,
            timeout=DEFAULT_TIMEOUT * 5,
        )
        resp.raise_for_status()
        deleted_labels = resp.json().get("deleted", 0)

    return {"examples_deleted": deleted_examples, "label_types_deleted": deleted_labels}


# ── Pushing ───────────────────────────────────────────────────────


def rows_from_csv(
    csv_path: Path,
    key_order: Sequence[str],
    text_column: str,
    annotation_column: str = "annotation",
    meta_columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Read a repaired CSV into push payloads.

    Each payload carries the text, the annotation in canonical key order, and
    the remaining columns as metadata.

    Parameters
    ----------
    csv_path : Path
        Repaired annotation CSV.
    key_order : sequence of str
        Canonical key order, applied defensively even on a repaired file.
    text_column : str
        Column holding the annotated text.
    annotation_column : str
        Column holding the serialised annotation.
    meta_columns : sequence of str or None
        Columns to carry as metadata. Defaults to every other column.

    Returns
    -------
    list of dict
        Push payloads in file order.
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for required in (text_column, annotation_column):
            if required not in fieldnames:
                raise ValueError(
                    f"Column '{required}' not found in {csv_path}. "
                    f"Available columns: {', '.join(fieldnames)}"
                )
        carried = (
            list(meta_columns)
            if meta_columns is not None
            else [c for c in fieldnames if c not in (text_column, annotation_column)]
        )
        rows: List[Dict[str, Any]] = []
        for record in reader:
            text = (record.get(text_column) or "").strip()
            if not text:
                continue
            raw = record.get(annotation_column)
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            annotation = reorder_payload(payload, key_order, include_missing=False)
            meta = {
                column: record[column]
                for column in carried
                if record.get(column) not in (None, "")
            }
            rows.append({"text": text, "annotation": annotation, "meta": meta})
    return rows


def push_rows(
    api_url: str,
    token: str,
    project_id: int,
    rows: Sequence[Dict[str, Any]],
    batch_size: int = 50,
    on_progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Push payloads into a project, in file order.

    Order matters: Doccano creates a category type the first time it meets a
    label, and that creation order is what the annotation panel displays.

    Parameters
    ----------
    api_url : str
        Base URL of the Infer API.
    token : str
        Bearer token.
    project_id : int
        Destination project.
    rows : sequence of dict
        Payloads from ``rows_from_csv``.
    batch_size : int
        Rows per request.
    on_progress : callable or None
        Called with ``(pushed_so_far, total)`` after each batch.

    Returns
    -------
    dict
        Counts of pushed rows and failures.
    """
    base = _base(api_url)
    headers = _headers(token)
    pushed = 0
    failures: List[str] = []

    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        resp = requests.post(
            f"{base}/doccano/push/batch",
            headers=headers,
            json={"project_id": project_id, "items": chunk},
            timeout=DEFAULT_TIMEOUT * 5,
        )
        if resp.status_code not in (200, 201):
            failures.append(f"rows {start}-{start + len(chunk)}: {resp.status_code} {resp.text[:200]}")
        else:
            pushed += len(chunk)
        if on_progress:
            on_progress(pushed, len(rows))

    return {"pushed": pushed, "total": len(rows), "failures": failures}


# ── CLI ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doccano_replace",
        description=(
            "Back up a Doccano project, clear its examples and category types, "
            "then refill it from a repaired annotation CSV."
        ),
    )
    parser.add_argument("--api-url", required=True, help="Infer API base URL")
    parser.add_argument("--token", required=True, help="Infer API bearer token")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--csv", type=Path, required=True, help="Repaired CSV")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--annotation-column", default="annotation")
    parser.add_argument(
        "--keys", required=True, help="Comma-separated canonical key order"
    )
    parser.add_argument("--backup", type=Path, help="Backup file path")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--keep-label-types",
        action="store_true",
        help="Keep the project's existing category types",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Back up and report the plan without deleting or pushing",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    key_order = ordered_unique(k.strip() for k in args.keys.split(","))

    rows = rows_from_csv(
        args.csv,
        key_order,
        text_column=args.text_column,
        annotation_column=args.annotation_column,
    )
    print(f"Read {len(rows)} rows from {args.csv}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = args.backup or Path(
        f"doccano_project_{args.project_id}_backup_{stamp}.json"
    )
    report = backup_project(args.api_url, args.token, args.project_id, backup_path)
    print(
        f"Backed up {report['example_count']} examples "
        f"({report['bytes']} bytes) to {report['path']}"
    )

    if args.dry_run:
        print("\nDry run: nothing deleted, nothing pushed.")
        print(f"Would clear project {args.project_id} and push {len(rows)} rows.")
        return 0

    cleared = clear_project(
        args.api_url,
        args.token,
        args.project_id,
        drop_label_types=not args.keep_label_types,
    )
    print(
        f"Cleared {cleared['examples_deleted']} examples and "
        f"{cleared['label_types_deleted']} category types"
    )

    def _progress(done: int, total: int) -> None:
        print(f"  pushed {done}/{total}", end="\r", flush=True)

    result = push_rows(
        args.api_url,
        args.token,
        args.project_id,
        rows,
        batch_size=args.batch_size,
        on_progress=_progress,
    )
    print()
    print(f"Pushed {result['pushed']}/{result['total']} rows")
    for failure in result["failures"]:
        print(f"  FAILED {failure}")
    return 1 if result["failures"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

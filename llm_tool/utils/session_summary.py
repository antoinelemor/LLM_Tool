#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
session_summary.py

MAIN OBJECTIVE:
---------------
Maintain a consistent resume metadata schema for interactive sessions and
provide helpers to read, merge, and aggregate summaries across workflows.

Dependencies:
-------------
- json
- dataclasses
- datetime
- pathlib
- typing

MAIN FEATURES:
--------------
1) Normalised SessionSummary dataclass with timestamp management
2) Robust JSON serialisation/deserialisation helpers with fallbacks
3) Merge utilities that reconcile persisted and in-memory updates safely
4) Collection helpers to surface recent sessions per interactive mode
5) Step tracking conveniences for resume prompts and status reporting

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    """Return current timestamp in ISO format."""
    return datetime.now().strftime(ISO_FORMAT)


def _serialise(value: Any) -> Any:
    """Best-effort serialisation for JSON."""
    if isinstance(value, datetime):
        return value.strftime(ISO_FORMAT)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    return value


def _deserialise_timestamp(value: Optional[str]) -> Optional[str]:
    """Normalise timestamps to ISO format when possible."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "")).strftime(ISO_FORMAT)
    except ValueError:
        return value


@dataclass
class SessionSummary:
    """Normalised metadata describing a resumable session."""

    mode: str
    session_id: str
    session_name: str
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_step_key: Optional[str] = None
    last_step_name: Optional[str] = None
    last_step_no: Optional[int] = None
    last_event_at: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialise summary to a JSON-friendly dict."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_step": {
                "key": self.last_step_key,
                "name": self.last_step_name,
                "number": self.last_step_no,
                "updated_at": self.last_event_at,
            },
            "extra": _serialise(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionSummary":
        """Create summary from dict, tolerating partial data."""
        last_step = payload.get("last_step", {}) or {}
        return cls(
            mode=payload.get("mode", "unknown"),
            session_id=payload.get("session_id", "unknown_session"),
            session_name=payload.get("session_name", payload.get("session_id", "session")),
            status=payload.get("status", "pending"),
            created_at=_deserialise_timestamp(payload.get("created_at")) or _now_iso(),
            updated_at=_deserialise_timestamp(payload.get("updated_at")) or _now_iso(),
            last_step_key=last_step.get("key"),
            last_step_name=last_step.get("name"),
            last_step_no=last_step.get("number"),
            last_event_at=_deserialise_timestamp(last_step.get("updated_at")),
            extra=payload.get("extra", {}) or {},
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def bump_updated(self) -> None:
        """Update ``updated_at`` and keep ``last_event_at`` defaulted."""
        timestamp = _now_iso()
        self.updated_at = timestamp
        if not self.last_event_at:
            self.last_event_at = timestamp

    def record_step(
        self,
        *,
        key: str,
        name: Optional[str] = None,
        number: Optional[int] = None,
        status: Optional[str] = None,
        summary: Optional[str] = None,
        event_at: Optional[str] = None,
    ) -> None:
        """
        Update the last step information and optionally tweak status/summary.
        """
        self.last_step_key = key
        self.last_step_name = name or self.last_step_name
        self.last_step_no = number if number is not None else self.last_step_no
        self.last_event_at = _deserialise_timestamp(event_at) or _now_iso()
        if status:
            self.status = status
        if summary:
            self.extra["last_step_summary"] = summary
        self.bump_updated()


def read_summary(path: Path) -> Optional[SessionSummary]:
    """Read summary from path if it exists."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return SessionSummary.from_dict(data)


def write_summary(path: Path, summary: SessionSummary) -> None:
    """Persist summary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_summary(path: Path, updates: SessionSummary) -> SessionSummary:
    """
    Merge ``updates`` into an existing summary stored at ``path`` and persist it.
    """
    current = read_summary(path)
    if current is None:
        current = updates
    else:
        # Overwrite with non-empty fields from updates
        for field_name in (
            "mode",
            "session_id",
            "session_name",
            "status",
            "created_at",
            "updated_at",
            "last_step_key",
            "last_step_name",
            "last_step_no",
            "last_event_at",
            "schema_version",
        ):
            value = getattr(updates, field_name)
            if value not in (None, "", []):
                setattr(current, field_name, value)
        if updates.extra:
            merged_extra = dict(current.extra)
            merged_extra.update(updates.extra)
            current.extra = merged_extra
        current.bump_updated()

    write_summary(path, current)
    return current


@dataclass
class SummaryRecord:
    """Utility container tying a SessionSummary to its on-disk location."""

    summary: SessionSummary
    directory: Path
    resume_path: Path

    @property
    def mode(self) -> str:
        return self.summary.mode


def _iter_session_dirs(base_dir: Path) -> Iterable[Path]:
    """Yield session directories (newest first) under the provided base path."""
    if not base_dir.exists():
        return []
    session_dirs: List[Tuple[datetime, Path]] = []
    for candidate in base_dir.iterdir():
        if not candidate.is_dir():
            continue
        updated_ts = candidate.stat().st_mtime
        session_dirs.append((datetime.fromtimestamp(updated_ts), candidate))
    session_dirs.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in session_dirs]


def collect_summaries_for_mode(
    base_dir: Path,
    mode: str,
    limit: Optional[int] = None,
) -> List[SummaryRecord]:
    """
    Return recent SessionSummary records stored under ``base_dir`` for a mode.

    Args:
        base_dir: Directory containing per-session subdirectories.
        mode: Logical mode identifier to enforce in case summaries are missing it.
        limit: Optional maximum number of records to return.
    """
    records: List[SummaryRecord] = []
    for session_dir in _iter_session_dirs(base_dir):
        resume_path = session_dir / "resume.json"
        summary = read_summary(resume_path)
        if summary is None:
            continue
        if not summary.mode or summary.mode == "unknown":
            summary.mode = mode
        records.append(SummaryRecord(summary=summary, directory=session_dir, resume_path=resume_path))
        if limit and len(records) >= limit:
            break
    records.sort(key=lambda record: record.summary.updated_at, reverse=True)
    return records[:limit] if limit else records


def collect_all_summaries(
    mode_roots: Dict[str, Path],
    *,
    limit_per_mode: Optional[int] = None,
    total_limit: Optional[int] = None,
) -> List[SummaryRecord]:
    """
    Aggregate SessionSummary records across multiple modes.

    Args:
        mode_roots: Mapping of mode identifiers to their base directory.
        limit_per_mode: Optional per-mode cap.
        total_limit: Optional cap applied after aggregation.
    """
    records: List[SummaryRecord] = []
    for mode, base_dir in mode_roots.items():
        records.extend(collect_summaries_for_mode(base_dir, mode, limit_per_mode))

    def _safe_datetime(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.min

    records.sort(key=lambda record: _safe_datetime(record.summary.updated_at), reverse=True)
    if total_limit is not None:
        records = records[:total_limit]
    return records

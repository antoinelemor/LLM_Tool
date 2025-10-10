#!/usr/bin/env python3
"""
Utilities to provide persistent session logging and resume support for
the BERT Annotation Studio workflow.

This module mirrors the capabilities offered in other interactive modes:
each session is recorded under ``logs/annotation_studio/{session_id}``,
with JSON metadata that tracks step progression, user inputs, and
artifacts. When re-launching the studio, the manager makes it possible
to relaunch or resume any session with a detailed step navigator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Optional rich dependency for nicer tables
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover - graceful fallback
    Console = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    box = None  # type: ignore


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _serialize_for_json(value: Any) -> Any:
    """Best-effort conversion for JSON serialization."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.strftime(ISO_FORMAT)
    if isinstance(value, dict):
        return {k: _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(v) for v in value]
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


@dataclass(frozen=True)
class SessionStep:
    key: str
    name: str
    description: str
    step_no: int


class AnnotationStudioSessionManager:
    """Centralises session persistence for the BERT Annotation Studio."""

    STEPS: List[SessionStep] = [
        SessionStep("select_models", "Select trained models", "Pick checkpoints to run", 1),
        SessionStep("configure_pipeline", "Configure pipeline", "Ordering, reductions, cascading", 2),
        SessionStep("select_dataset", "Choose dataset", "Data source selection", 3),
        SessionStep("map_columns", "Inspect & map columns", "Column identification and validation", 4),
        SessionStep("output_columns", "Name output columns", "Prediction column naming", 5),
        SessionStep("language_detection", "Language detection", "Verify language compatibility", 6),
        SessionStep("text_correction", "Text correction", "Optional preprocessing hooks", 7),
        SessionStep("annotation_options", "Annotation options", "Parallelisation & coverage", 8),
        SessionStep("export_options", "Export options", "Define outputs and formats", 9),
        SessionStep("review_launch", "Review & launch", "Final confirmation and execution", 10),
    ]

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        console: Optional[Console] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.base_dir = Path(base_dir or Path("logs/annotation_studio"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.console = console
        self.logger = logger or logging.getLogger(__name__)

        self.session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.metadata: Dict[str, Any] = {}
        self.step_cache: Dict[str, Any] = {}

        self._step_lookup = {step.key: step for step in self.STEPS}

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def start_new_session(self, slug: str) -> str:
        """Create a brand new session directory with metadata scaffold."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{slug}_{timestamp}"
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "session_id": session_id,
            "session_name": slug,
            "created_at": datetime.now().strftime(ISO_FORMAT),
            "updated_at": datetime.now().strftime(ISO_FORMAT),
            "status": "active",
            "steps": {
                step.key: {"status": "pending", "name": step.name, "order": step.step_no}
                for step in self.STEPS
            },
            "resume_log": [],
        }

        self.session_id = session_id
        self.session_dir = session_dir
        self.metadata = metadata
        self.step_cache = {}

        _write_json(self.session_dir / "session.json", metadata)
        (self.session_dir / "steps").mkdir(exist_ok=True)
        (self.session_dir / "artifacts").mkdir(exist_ok=True)

        self.logger.info("Created annotation studio session %s", session_id)
        return session_id

    def resume_session(self, session_id: str) -> None:
        """Load existing session metadata and cached steps."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        self.session_id = session_id
        self.session_dir = session_dir
        self.metadata = _load_json(session_dir / "session.json")
        self.step_cache = {}

        steps_dir = session_dir / "steps"
        if steps_dir.exists():
            for json_path in steps_dir.glob("*.json"):
                try:
                    step_payload = _load_json(json_path)
                except Exception:
                    continue
                step_key = step_payload.get("key")
                if step_key:
                    self.step_cache[step_key] = step_payload.get("data")

        self.logger.info("Loaded annotation studio session %s", session_id)

    def list_sessions(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Return metadata summaries for existing sessions (sorted by recency)."""
        sessions: List[Dict[str, Any]] = []
        for session_dir in sorted(self.base_dir.glob("*"), reverse=True):
            if not session_dir.is_dir():
                continue
            metadata = _load_json(session_dir / "session.json")
            if not metadata:
                continue
            updated_at = metadata.get("updated_at") or metadata.get("created_at")
            sessions.append(
                {
                    "session_id": metadata.get("session_id", session_dir.name),
                    "session_name": metadata.get("session_name", session_dir.name),
                    "status": metadata.get("status", "unknown"),
                    "updated_at": updated_at or "",
                    "last_step": metadata.get("last_completed_step"),
                    "path": session_dir,
                }
            )
        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return sessions[:limit]

    def record_resume(self, step_no: int) -> None:
        """Append a resume event to session metadata."""
        if not self.metadata:
            return
        resume_log = self.metadata.setdefault("resume_log", [])
        resume_log.append(
            {
                "resumed_at": datetime.now().strftime(ISO_FORMAT),
                "step_no": int(step_no),
            }
        )
        self._touch_metadata()

    def set_status(self, status: str) -> None:
        """Update high-level session status."""
        if not self.metadata:
            return
        self.metadata["status"] = status
        self._touch_metadata()

    # ------------------------------------------------------------------
    # Step state management
    # ------------------------------------------------------------------
    def mark_step_started(self, step_key: str) -> None:
        """Mark a step as in-progress."""
        step_meta = self._ensure_step_metadata(step_key)
        step_meta["status"] = "in_progress"
        step_meta["started_at"] = datetime.now().strftime(ISO_FORMAT)
        self._touch_metadata()

    def mark_step_failed(self, step_key: str, reason: str) -> None:
        """Record a failure for the given step."""
        step_meta = self._ensure_step_metadata(step_key)
        step_meta["status"] = "failed"
        step_meta["failed_at"] = datetime.now().strftime(ISO_FORMAT)
        step_meta["failure_reason"] = reason
        self._touch_metadata()

    def save_step(self, step_key: str, payload: Dict[str, Any], summary: str = "") -> None:
        """Persist step payload and mark as completed."""
        if self.session_dir is None:
            raise RuntimeError("No active session to save the step.")

        step_info = {
            "key": step_key,
            "saved_at": datetime.now().strftime(ISO_FORMAT),
            "data": _serialize_for_json(payload),
        }

        steps_dir = self.session_dir / "steps"
        order = self._step_lookup[step_key].step_no
        file_name = f"{order:02d}_{step_key}.json"
        _write_json(steps_dir / file_name, step_info)

        self.step_cache[step_key] = payload

        step_meta = self._ensure_step_metadata(step_key)
        step_meta["status"] = "completed"
        step_meta["completed_at"] = datetime.now().strftime(ISO_FORMAT)
        if summary:
            step_meta["summary"] = summary

        self.metadata["last_completed_step"] = step_key
        self._touch_metadata()

    def get_step_data(self, step_key: str) -> Optional[Dict[str, Any]]:
        """Return previously saved payload for the given step (if any)."""
        data = self.step_cache.get(step_key)
        if isinstance(data, dict):
            return data
        return None

    def get_step_name(self, step_key: str) -> str:
        return self._step_lookup[step_key].name

    def next_pending_step(self) -> int:
        """Return next step number that is not completed."""
        steps = self.metadata.get("steps", {}) if self.metadata else {}
        for step in self.STEPS:
            status = steps.get(step.key, {}).get("status")
            if status != "completed":
                return step.step_no
        return self.STEPS[-1].step_no

    # ------------------------------------------------------------------
    # Presentation helpers (Rich optional)
    # ------------------------------------------------------------------
    def render_sessions_table(self, sessions: List[Dict[str, Any]]) -> None:
        """Pretty print session list if Rich is available."""
        if not sessions:
            if self.console:
                self.console.print("[yellow]No sessions recorded yet.[/yellow]")
            return

        if not (self.console and Table and Panel):
            # Fallback to simple stdout
            for idx, session in enumerate(sessions, 1):
                print(f"{idx}. {session['session_id']} [{session['status']}] {session.get('updated_at','')}")
            return

        table = Table(title="Annotation Studio Sessions", box=box.ROUNDED, show_lines=False)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Session", style="bright_white", overflow="fold")
        table.add_column("Status", style="green", width=12)
        table.add_column("Updated", style="magenta", width=20)
        table.add_column("Last step", style="cyan", overflow="fold")

        for idx, session in enumerate(sessions, 1):
            table.add_row(
                str(idx),
                session["session_id"],
                session.get("status", "unknown"),
                session.get("updated_at", ""),
                str(session.get("last_step") or "-"),
            )

        panel = Panel(table, border_style="cyan", title="📂 Session Navigator")
        self.console.print(panel)

    def render_step_status(self) -> None:
        """Display per-step status for the active session."""
        if not (self.console and Table):
            return
        steps_meta = self.metadata.get("steps", {}) if self.metadata else {}

        table = Table(box=box.ROUNDED)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Step", style="white", width=30)
        table.add_column("Status", style="green", width=12)
        table.add_column("Summary", style="dim")

        for step in self.STEPS:
            meta = steps_meta.get(step.key, {})
            status = meta.get("status", "pending")
            summary = meta.get("summary", "")
            table.add_row(str(step.step_no), step.name, status, summary)

        self.console.print(table)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_step_metadata(self, step_key: str) -> Dict[str, Any]:
        if not self.metadata:
            self.metadata = {"steps": {}}
        steps = self.metadata.setdefault("steps", {})
        if step_key not in steps:
            step = self._step_lookup[step_key]
            steps[step_key] = {"status": "pending", "name": step.name, "order": step.step_no}
        return steps[step_key]

    def _touch_metadata(self) -> None:
        if not (self.session_dir and self.metadata):
            return
        self.metadata["updated_at"] = datetime.now().strftime(ISO_FORMAT)
        _write_json(self.session_dir / "session.json", _serialize_for_json(self.metadata))

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    @staticmethod
    def slugify(raw_name: str) -> str:
        """Generate a filesystem-friendly slug from user input."""
        sanitized = raw_name.strip().lower().replace(" ", "_")
        sanitized = "".join(ch for ch in sanitized if ch.isalnum() or ch in {"_", "-"})
        return sanitized or "session"


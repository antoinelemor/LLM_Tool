#!/usr/bin/env python3
"""Shared helpers to centralize training output directories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


_TRAINING_LOGS_BASE: Path = Path("logs/training_arena")
_TRAINING_SESSION_DIR: Optional[Path] = None


def set_training_logs_base(base_path: Optional[Path], session_dir: Optional[Path] = None) -> None:
    """
    Update the base directory for training logs and the active session directory.

    Args:
        base_path: Root directory for all training sessions.
        session_dir: Concrete session directory (used when the session lives outside the base path).
    """
    global _TRAINING_LOGS_BASE, _TRAINING_SESSION_DIR

    _TRAINING_LOGS_BASE = Path(base_path) if base_path else Path("logs/training_arena")
    _TRAINING_SESSION_DIR = Path(session_dir) if session_dir else None


def get_training_logs_base() -> Path:
    """Return the configured base directory for training logs."""
    return _TRAINING_LOGS_BASE


def get_session_dir(session_id: str) -> Path:
    """
    Return the directory for a specific training session.

    If an explicit session directory was registered, it is returned directly.
    Otherwise the directory is derived from the base path and session identifier.
    """
    if _TRAINING_SESSION_DIR is not None:
        return _TRAINING_SESSION_DIR
    return _TRAINING_LOGS_BASE / session_id


def get_training_metrics_dir(session_id: str) -> Path:
    """Return the directory where metrics for a session should be stored."""
    return get_session_dir(session_id) / "training_metrics"


def get_training_data_dir(session_id: str) -> Path:
    """Return the directory where training data reports should be stored."""
    return get_session_dir(session_id) / "training_data"


def get_training_metadata_dir(session_id: str) -> Path:
    """Return the directory where session metadata should be stored."""
    return get_session_dir(session_id) / "training_session_metadata"


def resolve_metrics_base_dir(configured: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the metrics base directory, honoring explicit overrides while keeping
    the active training session context.

    Args:
        configured: Optional user-configured directory.

    Returns:
        Path to the base directory where metrics should be written.
    """
    base_dir = get_training_logs_base()
    if configured:
        configured_path = Path(configured)
        # Allow custom overrides unless it's the legacy default and a session-specific base is active
        if configured_path != Path("logs/training_arena") or base_dir == Path("logs/training_arena"):
            return configured_path
    return base_dir

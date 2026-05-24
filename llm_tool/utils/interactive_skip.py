"""
PROJECT:
-------
LLMTool

TITLE:
------
interactive_skip.py

MAIN OBJECTIVE:
---------------
Global, robust interactive skip listener for long-running multi-stage
operations (sequential model training, multi-prompt annotation, etc.).

A single background thread reads stdin; each consumer "scope" (e.g. one
model in a multi-model training run) gets a fresh request flag that can
be checked from the foreground thread without blocking.

The previous in-trainer implementation spawned a daemon thread per model
and used a blocking `for line in sys.stdin` loop. That meant zombie
threads accumulated across models, and a single keystroke could be
consumed by any of them — leading to the skip silently failing on the
second and later models. This module fixes both issues.

KEY FEATURES:
-------------
1) Singleton listener thread (started once per process).
2) Scope-based skip flag — call `.reset()` between models to clear stale
   state.
3) Triggers on lines matching {'s', 'skip', 'next'} (case-insensitive).
4) No-op when stdin is not a TTY (CI, redirected stdin) — never crashes.

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import sys
import threading
from typing import Optional


class _SkipListener:
    """Singleton background-thread stdin listener with scope-reset semantics."""

    _instance_lock = threading.Lock()
    _instance: Optional["_SkipListener"] = None

    SKIP_TOKENS = {"s", "skip", "next"}

    def __init__(self) -> None:
        self._flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._enabled = False
        self._start_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_SkipListener":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def enable(self) -> None:
        """Start the background listener if not already running.

        Safe to call multiple times. No-op if stdin is not interactive.
        """
        with self._start_lock:
            if self._started:
                self._enabled = True
                return
            if not self._stdin_is_interactive():
                self._enabled = False
                self._started = True  # never try again
                return
            self._enabled = True
            self._started = True
            self._thread = threading.Thread(
                target=self._listen_loop,
                name="InteractiveSkipListener",
                daemon=True,
            )
            self._thread.start()

    def reset(self) -> None:
        """Clear any pending skip request. Call at the start of each new scope."""
        self._flag.clear()

    def consume(self) -> bool:
        """Return True if a skip was requested since the last reset, and clear it."""
        if not self._enabled:
            return False
        if self._flag.is_set():
            self._flag.clear()
            return True
        return False

    def is_set(self) -> bool:
        """Non-consuming check."""
        return self._enabled and self._flag.is_set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _stdin_is_interactive() -> bool:
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            return False

    def _listen_loop(self) -> None:
        """Run forever in a daemon thread; set flag when a skip token arrives."""
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                # EOF — terminal closed
                return
            token = line.strip().lower()
            if token in self.SKIP_TOKENS:
                self._flag.set()


def get_skip_listener() -> _SkipListener:
    """Return the process-wide skip listener (does not auto-enable)."""
    return _SkipListener.instance()

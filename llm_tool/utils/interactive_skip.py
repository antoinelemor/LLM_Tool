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

A single background thread polls stdin via `select.select()` with a short
timeout, reads bytes as they arrive, and sets a process-wide Event when a
skip token is detected. The foreground loop checks the flag at every
epoch boundary.

Why this design (vs. blocking `sys.stdin.readline()`):
------------------------------------------------------
- Rich's `Live` display does not put the terminal into raw mode, but on
  some platforms / shells line-buffered `readline()` can still be eaten
  by parent process buffering, or block forever if the user never types
  Enter — making `s` + Enter unreliable.
- Spawning a fresh blocking thread per model (the previous approach)
  also accumulated zombie threads stuck in `readline()`, and a single
  keystroke could be consumed by the wrong one.

The current implementation:
  * Detects `s` / `skip` / `next` either as a full line (Enter pressed)
    or as a lone `s` keystroke (no Enter needed) when the tty is put
    into cbreak mode by us. We do NOT modify the tty mode globally
    because Rich already manages display; we just read whatever bytes
    arrive on stdin.
  * Polls via `select()` with a 200 ms timeout so the thread never
    blocks indefinitely and reacts to EOF cleanly.
  * Single process-wide listener; `reset()` between models clears stale
    state.
  * Falls back to no-op when stdin is not a TTY (CI, piped input).

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

try:
    import select  # POSIX; available on macOS / Linux
    _HAS_SELECT = True
except ImportError:
    _HAS_SELECT = False


class _SkipListener:
    """Singleton background-thread stdin listener with scope-reset semantics."""

    _instance_lock = threading.Lock()
    _instance: Optional["_SkipListener"] = None

    # Full-line tokens (when user types `s` + Enter, `skip` + Enter, ...).
    SKIP_LINE_TOKENS = {"s", "skip", "next"}
    # Single-char trigger — accepted whether or not Enter follows.
    SKIP_CHAR = "s"
    # Poll interval for select() in seconds.
    POLL_INTERVAL = 0.2

    def __init__(self) -> None:
        self._flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._enabled = False
        self._stop = threading.Event()
        self._start_lock = threading.Lock()
        self._stdin_fd: Optional[int] = None
        self._buffer = ""

    @classmethod
    def instance(cls) -> "_SkipListener":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def enable(self) -> None:
        """Start the background listener if not already running.

        Safe to call multiple times. No-op if stdin is not interactive
        or if `select` is unavailable.
        """
        with self._start_lock:
            if self._started:
                self._enabled = True
                return
            if not self._stdin_is_interactive() or not _HAS_SELECT:
                self._enabled = False
                self._started = True  # never try again
                return
            try:
                self._stdin_fd = sys.stdin.fileno()
            except (OSError, ValueError):
                self._enabled = False
                self._started = True
                return
            self._enabled = True
            self._started = True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._listen_loop,
                name="InteractiveSkipListener",
                daemon=True,
            )
            self._thread.start()

    def reset(self) -> None:
        """Clear any pending skip request. Call at the start of each new scope."""
        self._flag.clear()
        self._buffer = ""

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
        """Poll stdin with select(); set flag on skip token.

        Reads bytes (not lines) so a bare `s` keystroke is detected even
        if the terminal happens to be in raw / cbreak mode (e.g. Rich
        Live on certain shells); also recognises `s\\n`, `skip\\n`, and
        `next\\n` from line-buffered stdin.
        """
        fd = self._stdin_fd
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], self.POLL_INTERVAL)
            except (OSError, ValueError):
                return  # fd closed
            if not ready:
                continue
            try:
                chunk = os.read(fd, 1024)
            except (OSError, ValueError):
                return
            if not chunk:
                # EOF — terminal closed; stop quietly.
                return
            try:
                text = chunk.decode("utf-8", errors="ignore")
            except Exception:
                continue
            self._buffer += text

            # Process buffered input. We accept either:
            #   - a complete line whose stripped value is a skip token,
            #   - or a bare single 's' / 'S' char (no Enter required).
            triggered = False
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip().lower() in self.SKIP_LINE_TOKENS:
                    triggered = True
            # Bare-char trigger: a leftover single 's' with no newline yet.
            if not triggered and self._buffer and self._buffer.strip().lower() == self.SKIP_CHAR:
                triggered = True
                self._buffer = ""

            if triggered:
                self._flag.set()
                # Keep listening for subsequent models — do not break.


def get_skip_listener() -> _SkipListener:
    """Return the process-wide skip listener (does not auto-enable)."""
    return _SkipListener.instance()

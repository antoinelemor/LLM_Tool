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

A single background thread polls stdin -- with `select.select()` on POSIX,
with `msvcrt.kbhit()` on Windows -- reads bytes as they arrive, and sets a
process-wide Event when a skip token is detected. The foreground loop checks
the flag at every epoch boundary.

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
  * Polls with a 200 ms timeout so the thread never blocks indefinitely
    and reacts to EOF cleanly. On POSIX that is `select()` on the stdin
    file descriptor; on Windows `select()` only accepts sockets, so the
    thread uses `msvcrt.kbhit()` / `msvcrt.getwch()` instead.
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
import time
from typing import Optional

# select() exists on Windows but only accepts sockets there: handing it the
# stdin file descriptor raises OSError (WinError 10038), which would kill the
# listener thread and silently disable skipping. msvcrt is the console API.
if os.name == "nt":
    _HAS_SELECT = False
    try:
        import msvcrt
        _HAS_MSVCRT = True
    except ImportError:  # pragma: no cover - msvcrt ships with CPython on Windows
        _HAS_MSVCRT = False
else:
    _HAS_MSVCRT = False
    try:
        import select  # POSIX; available on macOS / Linux
        _HAS_SELECT = True
    except ImportError:  # pragma: no cover - select is always present on POSIX
        _HAS_SELECT = False

_HAS_STDIN_POLL = _HAS_SELECT or _HAS_MSVCRT


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
        or if no stdin polling primitive is available.
        """
        with self._start_lock:
            if self._started:
                self._enabled = True
                return
            if not self._stdin_is_interactive() or not _HAS_STDIN_POLL:
                self._enabled = False
                self._started = True  # never try again
                return
            if _HAS_SELECT:
                # msvcrt reads the console directly and needs no descriptor.
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

    def _read_available(self) -> Optional[str]:
        """Return whatever is waiting on stdin, ``""`` on timeout, ``None`` on EOF.

        Reads bytes (not lines) so a bare `s` keystroke is detected even
        if the terminal happens to be in raw / cbreak mode (e.g. Rich
        Live on certain shells); also recognises `s\\n`, `skip\\n`, and
        `next\\n` from line-buffered stdin.
        """
        if _HAS_MSVCRT:
            # The Windows console has no descriptor to wait on, so poll
            # kbhit() and sleep between checks for the same 200 ms cadence.
            # getwch() returns a str and never blocks once kbhit() is true.
            if not msvcrt.kbhit():
                time.sleep(self.POLL_INTERVAL)
                return ""
            text = ""
            while msvcrt.kbhit():
                try:
                    char = msvcrt.getwch()
                except (OSError, ValueError):
                    return None
                # Arrow keys and function keys arrive as a two-character
                # sequence led by \x00 or \xe0; drop both halves so they
                # cannot be mistaken for a token.
                if char in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                # The console reports Enter as a bare carriage return.
                text += "\n" if char == "\r" else char
            return text

        try:
            ready, _, _ = select.select([self._stdin_fd], [], [], self.POLL_INTERVAL)
        except (OSError, ValueError):
            return None  # fd closed
        if not ready:
            return ""
        try:
            chunk = os.read(self._stdin_fd, 1024)
        except (OSError, ValueError):
            return None
        if not chunk:
            return None  # EOF — terminal closed.
        return chunk.decode("utf-8", errors="ignore")

    def _listen_loop(self) -> None:
        """Poll stdin; set the flag when a skip token arrives."""
        while not self._stop.is_set():
            text = self._read_available()
            if text is None:
                return
            if not text:
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

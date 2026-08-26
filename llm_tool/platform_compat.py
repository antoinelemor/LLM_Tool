#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
platform_compat.py

MAIN OBJECTIVE:
---------------
Single place for the handful of behaviours that differ between POSIX and
Windows, so the rest of the package can stay platform-agnostic.

Three problems live here:

1) **Console encoding.** On Windows, ``sys.stdout`` is opened with the ANSI code
   page (cp1252 on a French or English install, cp932 on a Japanese one) unless
   UTF-8 mode is on. This CLI prints emoji, box-drawing characters and text in
   French, Arabic and Chinese, so the first menu would raise
   ``UnicodeEncodeError`` and take the process down. :func:`configure_console`
   re-opens the standard streams as UTF-8 and turns on ANSI escape handling.

2) **Filename legality.** POSIX only forbids ``/`` and NUL in a filename;
   Windows also forbids ``\\ : * ? " < > |``, reserves names like ``CON`` and
   ``NUL``, and rejects trailing dots or spaces. Model identifiers such as
   ``llama3.2:3b`` and ``microsoft/deberta-v3-base`` are used as directory
   names, so :func:`sanitize_path_component` normalises them everywhere.

3) **Filesystem calls with different semantics.** ``os.rename`` overwrites
   silently on POSIX but raises ``FileExistsError`` on Windows, and a writable
   working directory cannot be assumed. :func:`replace_path` and
   :func:`writable_dir` cover those.

Dependencies:
-------------
- os
- sys
- re
- ctypes (Windows only, standard library)
- pathlib
- shutil
- tempfile

MAIN FEATURES:
--------------
1) IS_WINDOWS / IS_MACOS / IS_LINUX constants
2) configure_console(): UTF-8 standard streams + ANSI escapes on Windows
3) sanitize_path_component(): filename-safe on every supported platform
4) replace_path(): rename-or-overwrite with POSIX semantics everywhere
5) writable_dir(): first writable candidate, with a temp-dir last resort
6) supports_unicode(): whether the console can render non-ASCII glyphs

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

__all__ = [
    "IS_WINDOWS",
    "IS_MACOS",
    "IS_LINUX",
    "configure_console",
    "supports_unicode",
    "sanitize_path_component",
    "replace_path",
    "writable_dir",
]


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

# Device names inherited from DOS. Windows still resolves them as devices in
# every directory, so a file called "con.json" cannot be created.
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Everything Windows rejects in a path component, plus the C0 control range.
_ILLEGAL_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_console_configured = False


def configure_console() -> None:
    """
    Make the standard streams safe for the CLI's Unicode output.

    Idempotent, and a no-op everywhere except Windows: POSIX terminals already
    default to UTF-8, and reconfiguring them would only risk breaking a locale
    the user chose deliberately.

    Notes
    -----
    ``PYTHONIOENCODING`` is also exported so that child processes -- the
    ``spawn``-ed DataLoader workers and training subprocesses, which do not
    inherit a reconfigured stream object -- start out with UTF-8 too.
    """
    global _console_configured
    if _console_configured or not IS_WINDOWS:
        _console_configured = True
        return
    _console_configured = True

    # Inherited by every child process, including spawned workers.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for name in ("stdout", "stderr", "stdin"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # Redirected to a plain object (pytest capture, a Tee wrapper, a
            # pythonw.exe run where the stream is None): nothing to do.
            continue
        try:
            # errors="replace" keeps a stray unencodable glyph from killing a
            # long training run: a lost character beats a lost session.
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue

    _enable_windows_ansi()


def _enable_windows_ansi() -> None:
    """
    Turn on ANSI escape interpretation for the legacy Windows console.

    Windows Terminal enables it already; ``conhost.exe`` (the window you get
    from cmd.exe on an older machine) does not, and without it every colour
    code the CLI emits outside Rich is printed literally as ``←[32m``.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except (ImportError, ValueError):  # pragma: no cover - non-Windows
        return

    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            if handle in (0, -1, None):
                continue
            mode = wintypes.DWORD()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue  # Redirected to a file or pipe: not a console.
            kernel32.SetConsoleMode(
                handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            )
    except Exception:  # pragma: no cover - defensive, never fatal
        return


def supports_unicode(stream=None) -> bool:
    """
    Report whether ``stream`` can encode the non-ASCII glyphs the CLI uses.

    Parameters
    ----------
    stream : file-like, optional
        Defaults to ``sys.stdout``.

    Returns
    -------
    bool
        True when a representative emoji and box-drawing character both encode.
    """
    stream = stream if stream is not None else sys.stdout
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "✓─\U0001F680".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def sanitize_path_component(
    value: Optional[str],
    fallback: str = "unnamed",
    max_length: int = 96,
) -> str:
    """
    Turn an arbitrary label into one path component that is legal everywhere.

    Parameters
    ----------
    value : str, optional
        The raw label, e.g. an Ollama model tag (``llama3.2:3b``), a HuggingFace
        repository id (``microsoft/deberta-v3-base``) or a dataset name.
    fallback : str, default "unnamed"
        Returned when `value` is empty or sanitises down to nothing.
    max_length : int, default 96
        Components are truncated to this many characters. Windows caps a full
        path at 260 characters unless long paths are enabled, and checkpoint
        directories nest several levels deep.

    Returns
    -------
    str
        A component with no separator, no reserved device name, no trailing dot
        or space, and no character Windows rejects.

    Examples
    --------
    >>> sanitize_path_component("llama3.2:3b")
    'llama3.2_3b'
    >>> sanitize_path_component("microsoft/deberta-v3-base")
    'microsoft_deberta-v3-base'
    >>> sanitize_path_component("aux")
    'aux_'
    """
    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    text = _ILLEGAL_PATH_CHARS.sub("_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")

    # Windows strips trailing dots and spaces from a name, so "model." and
    # "model" would collide and one of the two directories would go missing.
    text = text.rstrip(". ")

    if len(text) > max_length:
        text = text[:max_length].rstrip("_. ")

    if not text:
        return fallback

    # Reserved names are matched without their extension, so "nul.log" counts.
    stem = text.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        text = f"{text}_"

    return text


def replace_path(source, destination) -> Path:
    """
    Move `source` onto `destination`, overwriting it on every platform.

    ``os.rename`` overwrites an existing destination on POSIX but raises
    ``FileExistsError`` on Windows, and ``os.replace`` -- which has POSIX
    semantics everywhere -- refuses to overwrite a *directory* and cannot cross
    filesystems. This helper covers all three cases.

    Parameters
    ----------
    source : str or pathlib.Path
        Existing file or directory to move.
    destination : str or pathlib.Path
        Target path; replaced if it already exists.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    src = Path(source)
    dst = Path(destination)

    if dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst, ignore_errors=True)
        else:
            try:
                dst.unlink()
            except FileNotFoundError:
                pass

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.replace(src, dst)
    except OSError:
        # Different volumes: os.replace cannot span them, shutil.move can.
        shutil.move(str(src), str(dst))

    return dst


def writable_dir(candidates: Iterable, prefix: str = "llm_tool") -> Path:
    """
    Return the first candidate directory that can actually be written to.

    Parameters
    ----------
    candidates : iterable of str or pathlib.Path
        Directories to try, in order of preference.
    prefix : str, default "llm_tool"
        Prefix for the temporary directory used when no candidate works.

    Returns
    -------
    pathlib.Path
        An existing, writable directory.

    Notes
    -----
    A package installed into ``C:\\Program Files`` or a system ``site-packages``
    sits in a read-only tree, and a working directory on a network share may be
    read-only too. Falling back to the temp directory keeps the CLI able to
    start instead of dying on an unwritable log path.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".{prefix}_write_test"
            probe.touch()
            probe.unlink()
            return path
        except OSError:
            continue

    fallback = Path(tempfile.gettempdir()) / prefix
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

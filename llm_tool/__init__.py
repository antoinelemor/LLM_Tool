#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
__init__.py

MAIN OBJECTIVE:
---------------
This script initializes the LLMTool package, providing a unified interface for
LLM annotation, model training, validation, and large-scale data processing.

Dependencies:
-------------
- sys
- importlib
- typing

MAIN FEATURES:
--------------
1) Imports and exposes all annotation modules (local LLMs and APIs)
2) Exposes model training components from AugmentedSocialScientist
3) Provides unified pipeline orchestration
4) Includes validation and export utilities
5) Multi-language support with automatic detection

Author:
-------
Antoine Lemor
"""

# Standard imports
import os
import sys

# Environment helpers -----------------------------------------------------
def _env_flag(name: str) -> bool:
    """Interpret environment variable as boolean flag."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _running_inside_vscode() -> bool:
    """
    Heuristic detection for VS Code / Electron terminals.

    VS Code on macOS doesn't always expose TERM_PROGRAM=vscode, so we inspect
    several integration hooks that are stable across versions.
    """
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if term_program == "vscode":
        return True

    if os.environ.get("VSCODE_PID"):
        return True

    if os.environ.get("VSCODE_CWD") and os.environ.get("ELECTRON_RUN_AS_NODE"):
        return True

    if os.environ.get("VSCODE_IPC_HOOK") or os.environ.get("VSCODE_INJECTION"):
        return True

    return False


def _configure_rich_environment():
    """
    Apply safe Rich UI defaults for fragile terminals while keeping the UI enabled.

    Users can override the auto-detected profile via:
      * LLM_TOOL_RICH_PROFILE (full | balanced | safe | off)
      * LLM_TOOL_FORCE_RICH_UI=1 to force the full experience
      * LLM_TOOL_DISABLE_RICH_UI=1 to turn off Rich entirely
    """
    if _env_flag("LLM_TOOL_DISABLE_RICH_UI") or _env_flag("LLM_TOOL_FORCE_RICH_UI"):
        return

    if os.environ.get("LLM_TOOL_RICH_PROFILE"):
        # Respect explicit user configuration
        return

    if _running_inside_vscode():
        # Electron terminals on macOS are prone to crashing with aggressive
        # refresh rates. Favour a conservative profile by default.
        os.environ.setdefault("LLM_TOOL_RICH_PROFILE", "safe")
        os.environ.setdefault("LLM_TOOL_RICH_REFRESH_HZ", "4")
        os.environ.setdefault("LLM_TOOL_RICH_MIN_RENDER_INTERVAL", "0.25")
        os.environ.setdefault("LLM_TOOL_RICH_MIN_PROGRESS_INTERVAL", "0.05")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    else:
        # Provide a balanced default for other non-TTY environments
        if not sys.stdout.isatty():
            os.environ.setdefault("LLM_TOOL_RICH_PROFILE", "balanced")
            os.environ.setdefault("LLM_TOOL_RICH_REFRESH_HZ", "6")
            os.environ.setdefault("LLM_TOOL_RICH_MIN_RENDER_INTERVAL", "0.15")
            os.environ.setdefault("LLM_TOOL_RICH_MIN_PROGRESS_INTERVAL", "0.03")


_configure_rich_environment()

# Simple setup - disable tokenizers parallelism to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__version__ = "1.0.0"
__author__ = "Antoine Lemor"

# Import core components when they're ready
__all__ = [
    '__version__',
    '__author__',
]

# Lazy imports will be added as modules are implemented
def get_version():
    """Return the current version of LLMTool"""
    return __version__

def get_author():
    """Return the author information"""
    return __author__

"""
PROJECT:
-------
LLMTool

TITLE:
------
logging_utils.py

MAIN OBJECTIVE:
---------------
This script provides centralized logging utilities for consistent logging across
the entire package, with formatted output to stdout and configurable log levels
for debugging and monitoring.

Dependencies:
-------------
- logging (Python standard library)
- sys (standard output stream)

MAIN FEATURES:
--------------
1) Centralized logger creation with consistent formatting
2) Timestamp-based log formatting with module names
3) Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
4) Stream handler output to stdout
5) Hierarchical logger management for package modules
6) Dynamic log level adjustment for debugging
7) Prevents duplicate handlers on repeated calls
8) Non-propagating loggers to avoid interference

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout.

    The helper creates a stream handler the first time a logger is requested.
    Subsequent calls with the same name reuse the existing configuration.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_log_level(level: int) -> None:
    """Raise or lower the default level for all package loggers."""
    logging.getLogger("LLMTool").setLevel(level)

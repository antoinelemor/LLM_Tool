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
This script initializes the database module providing unified access to
PostgreSQL and file-based data handlers.

Dependencies:
-------------
- sys

MAIN FEATURES:
--------------
1) Export PostgreSQL handler
2) Export file handlers
3) Export utility functions

Author:
-------
Antoine Lemor
"""

# Import handlers with availability checking
try:
    from .postgresql_handler import PostgreSQLHandler
    HAS_POSTGRESQL = True
except ImportError:
    PostgreSQLHandler = None
    HAS_POSTGRESQL = False

from .file_handlers import (
    FileHandler,
    IncrementalFileWriter,
    create_file_handler,
    strip_log_columns,
    write_log_csv
)

__all__ = [
    'PostgreSQLHandler',
    'FileHandler',
    'IncrementalFileWriter',
    'create_file_handler',
    'strip_log_columns',
    'write_log_csv',
    'HAS_POSTGRESQL'
]

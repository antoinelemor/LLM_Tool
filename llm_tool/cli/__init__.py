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
This script initializes the CLI module for LLMTool package.

Dependencies:
-------------
- sys

MAIN FEATURES:
--------------
1) Export main CLI interface
2) Provide entry point function

Author:
-------
Antoine Lemor
"""

from .main_cli import LLMToolCLI, main

__all__ = ['LLMToolCLI', 'main']
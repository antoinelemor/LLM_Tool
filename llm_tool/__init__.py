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
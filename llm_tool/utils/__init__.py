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
This script initializes the utils module.

Dependencies:
-------------
- sys

MAIN FEATURES:
--------------
1) Export utility classes

Author:
-------
Antoine Lemor
"""

from .language_detector import LanguageDetector, DetectionMethod

__all__ = ['LanguageDetector', 'DetectionMethod']
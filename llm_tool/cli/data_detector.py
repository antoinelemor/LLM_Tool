#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
data_detector.py

MAIN OBJECTIVE:
---------------
Provide a backward-compatible shim so legacy CLI imports continue to expose
the dataset detection utilities now hosted under `llm_tool.utils.data_detector`.

Dependencies:
-------------
- llm_tool.utils.data_detector

MAIN FEATURES:
--------------
1) Re-export DataDetector symbols without modifying downstream callers
2) Shield existing CLI code from module reorganisation within the package
3) Preserve wildcard import behaviour expected by historical scripts
4) Simplify transition by keeping documentation and imports in one place
5) Facilitate future deprecation by centralising the compatibility layer

Author:
-------
Antoine Lemor
"""

from llm_tool.utils.data_detector import *  # noqa: F401,F403

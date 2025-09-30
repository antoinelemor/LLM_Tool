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
This script initializes the annotators module, exposing all annotation
functionality for LLM-based text annotation.

Dependencies:
-------------
- sys

MAIN FEATURES:
--------------
1) Export main annotation classes
2) Provide unified annotation interface

Author:
-------
Antoine Lemor
"""

from .llm_annotator import LLMAnnotator
from .api_clients import OpenAIClient, AnthropicClient, GoogleClient
from .local_models import OllamaClient, LlamaCPPClient
from .prompt_manager import PromptManager
from .json_cleaner import JSONCleaner, repair_json_string, clean_json_output

__all__ = [
    'LLMAnnotator',
    'OpenAIClient',
    'AnthropicClient',
    'GoogleClient',
    'OllamaClient',
    'LlamaCPPClient',
    'PromptManager',
    'JSONCleaner',
    'repair_json_string',
    'clean_json_output'
]
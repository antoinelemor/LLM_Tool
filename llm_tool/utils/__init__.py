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
from .system_resources import (
    detect_resources,
    SystemResourceDetector,
    SystemResources,
    GPUInfo,
    CPUInfo,
    MemoryInfo,
    StorageInfo,
    SystemInfo,
    get_device_recommendation,
    get_optimal_workers,
    get_optimal_batch_size,
    check_minimum_requirements
)
from .resource_display import (
    display_resources,
    create_resource_table,
    create_recommendations_table,
    create_compact_resource_panel,
    display_resource_header,
    get_resource_summary_text,
    create_visual_resource_panel,
    create_mode_resource_banner,
    create_detailed_mode_panel
)

__all__ = [
    'LanguageDetector',
    'DetectionMethod',
    'detect_resources',
    'SystemResourceDetector',
    'SystemResources',
    'GPUInfo',
    'CPUInfo',
    'MemoryInfo',
    'StorageInfo',
    'SystemInfo',
    'get_device_recommendation',
    'get_optimal_workers',
    'get_optimal_batch_size',
    'check_minimum_requirements',
    'display_resources',
    'create_resource_table',
    'create_recommendations_table',
    'create_compact_resource_panel',
    'display_resource_header',
    'get_resource_summary_text',
    'create_visual_resource_panel',
    'create_mode_resource_banner',
    'create_detailed_mode_panel'
]
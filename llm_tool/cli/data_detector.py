"""
Backward-compatible shim for dataset detection utilities.

The actual implementation now lives in ``llm_tool.utils.data_detector`` so that
it can be shared outside of CLI workflows.
"""

from llm_tool.utils.data_detector import *  # noqa: F401,F403

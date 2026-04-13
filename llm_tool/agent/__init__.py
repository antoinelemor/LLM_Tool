"""
LLMTool Agent - Conversational LLM interface for pipeline orchestration.

Provides a natural language interface where users describe tasks and an LLM agent
orchestrates annotation, training, inference, and validation operations via tool_use.
"""

from .config import AgentConfig
from .agent_cli import AgentCLI

__all__ = ["AgentCLI", "AgentConfig"]

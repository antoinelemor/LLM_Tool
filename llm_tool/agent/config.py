"""
Agent configuration - resolves provider, model, and credentials from env/settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the LLM agent orchestrator."""

    provider: str = "ollama"
    model: str = "llama3.2"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    keep_alive: Optional[str] = None
    temperature: float = 0.3
    max_conversation_turns: int = 50
    max_tool_calls_per_turn: int = 10

    @classmethod
    def from_env_and_settings(cls, settings) -> "AgentConfig":
        """Resolve agent config from environment variables and LLMTool settings."""
        provider = os.environ.get("LLM_TOOL_AGENT_PROVIDER", "ollama")
        model = os.environ.get("LLM_TOOL_AGENT_MODEL", "")
        api_key = None
        base_url = os.environ.get("LLM_TOOL_AGENT_BASE_URL")
        keep_alive = os.environ.get("LLM_TOOL_AGENT_KEEP_ALIVE")

        if provider == "anthropic":
            api_key = (
                os.environ.get("ANTHROPIC_API_KEY")
                or settings.get_api_key("anthropic")
            )
            if not model:
                model = "claude-sonnet-4-20250514"
        elif provider == "openai":
            api_key = (
                os.environ.get("OPENAI_API_KEY")
                or settings.get_api_key("openai")
            )
            if not model:
                model = "gpt-4o"
        elif provider == "ollama":
            if not model:
                model = "llama3.2"
            if not base_url:
                base_url = "http://localhost:11434/v1"
            if not keep_alive:
                keep_alive = "30m"
        else:
            if not model:
                model = "llama3.2"

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            keep_alive=keep_alive,
        )

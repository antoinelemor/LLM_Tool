"""
Agent configuration - resolves provider, model, and credentials from env/settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from .providers import DEFAULT_OLLAMA_BASE_URL
from ..config.providers import get_provider, resolve_api_key


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
    def from_env_and_settings(cls, settings, api_key: Optional[str] = None) -> "AgentConfig":
        """
        Resolve agent config from environment variables and LLMTool settings.

        Parameters
        ----------
        settings : Settings
            LLMTool settings, consulted for stored provider credentials.
        api_key : str, optional
            Explicit credential (e.g. from a command-line flag). Takes precedence
            over the environment and the stored key.
        """
        provider = os.environ.get("LLM_TOOL_AGENT_PROVIDER", "ollama")
        model = os.environ.get("LLM_TOOL_AGENT_MODEL", "")
        explicit_key = api_key
        base_url = os.environ.get("LLM_TOOL_AGENT_BASE_URL")
        keep_alive = os.environ.get("LLM_TOOL_AGENT_KEEP_ALIVE")

        # Cloud providers all resolve the same way: the provider's environment
        # variables (aliases included), then the encrypted store, then the
        # registry's default model. Ollama is handled separately below because a
        # local daemon needs no credential and does need a base URL.
        spec = get_provider(provider)
        if spec and spec.kind == "cloud":
            api_key = explicit_key or resolve_api_key(provider) or settings.get_api_key(provider)
            if not model:
                model = spec.default_agent_model or ""
            if not base_url and spec.openai_compat_base_url:
                base_url = spec.openai_compat_base_url
        if provider == "ollama":
            # A local daemon needs no credential, but the same provider id also
            # drives ollama.com, where inference is authenticated.
            api_key = explicit_key or (
                os.environ.get("OLLAMA_API_KEY")
                or settings.get_api_key("ollama")
            )
            if not model:
                model = "llama3.2"
            if not base_url:
                base_url = DEFAULT_OLLAMA_BASE_URL
            if not keep_alive:
                keep_alive = "30m"
        else:
            api_key = explicit_key
            if not model:
                model = "llama3.2"

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            keep_alive=keep_alive,
        )

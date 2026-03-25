"""
Multi-provider tool_use abstraction for the LLM agent.

Supports:
- OpenAI (GPT models) via openai SDK
- Ollama (local models) via OpenAI-compatible API
- Anthropic (Claude models) via anthropic SDK
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentMessage:
    """A message in the agent conversation."""
    role: str  # "system", "user", "assistant", "tool_result"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    # Store raw provider-specific data for faithful round-tripping
    _raw: Optional[Any] = field(default=None, repr=False)


class BaseAgentProvider(ABC):
    """Base class for agent LLM providers with tool_use support."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def chat(
        self,
        messages: List[AgentMessage],
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> AgentMessage:
        """Send messages and return response (potentially with tool calls)."""
        ...


class OpenAIAgentProvider(BaseAgentProvider):
    """OpenAI-compatible provider (also used as base for Ollama)."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, keep_alive: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.base_url = base_url
        self.keep_alive = keep_alive
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def _to_openai_messages(self, messages: List[AgentMessage]) -> List[Dict]:
        """Convert AgentMessage list to OpenAI message format."""
        result = []
        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content or ""})

            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content or ""})

            elif msg.role == "assistant":
                entry: Dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    entry["content"] = msg.content
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    if not msg.content:
                        entry["content"] = None
                result.append(entry)

            elif msg.role == "tool_result":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                })

        return result

    def _to_openai_tools(self, tools: List[Dict[str, Any]]) -> List[Dict]:
        """Convert tool definitions to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            }
            for tool in tools
        ]

    def chat(
        self,
        messages: List[AgentMessage],
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> AgentMessage:
        openai_messages = self._to_openai_messages(messages)
        openai_tools = self._to_openai_tools(tools)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
        if self.keep_alive:
            kwargs["extra_body"] = {"keep_alive": self.keep_alive}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return AgentMessage(
                role="assistant",
                content=f"Error communicating with the LLM: {e}",
            )

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return AgentMessage(
            role="assistant",
            content=message.content,
            tool_calls=tool_calls,
        )


class OllamaAgentProvider(OpenAIAgentProvider):
    """Ollama provider using OpenAI-compatible API."""

    def __init__(self, model: str, base_url: Optional[str] = None, keep_alive: Optional[str] = None, **kwargs):
        super().__init__(
            model=model,
            api_key="ollama",
            base_url=base_url or "http://localhost:11434/v1",
            keep_alive=keep_alive,
            **kwargs,
        )


class AnthropicAgentProvider(BaseAgentProvider):
    """Anthropic provider using native tool_use."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Claude agent. "
                    "Install with: pip install anthropic"
                )
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def _to_anthropic_messages(self, messages: List[AgentMessage]):
        """Convert AgentMessage list to Anthropic format.

        Returns (system_prompt, messages_list).
        """
        system_parts: List[str] = []
        result = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)

            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content or ""})

            elif msg.role == "assistant":
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                if content_blocks:
                    result.append({"role": "assistant", "content": content_blocks})

            elif msg.role == "tool_result":
                # Anthropic expects tool results as user messages with tool_result blocks
                # Group consecutive tool_results into a single user message
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
                # Check if last message is already a user message with tool_results
                if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                    result[-1]["content"].append(tool_result_block)
                else:
                    result.append({
                        "role": "user",
                        "content": [tool_result_block],
                    })

        system_prompt = "\n\n".join(system_parts)
        return system_prompt, result

    def _to_anthropic_tools(self, tools: List[Dict[str, Any]]) -> List[Dict]:
        """Convert tool definitions to Anthropic format."""
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", tool.get("parameters", {})),
            }
            for tool in tools
        ]

    def chat(
        self,
        messages: List[AgentMessage],
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> AgentMessage:
        system_prompt, anthropic_messages = self._to_anthropic_messages(messages)
        anthropic_tools = self._to_anthropic_tools(tools)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = self.client.messages.create(**kwargs)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return AgentMessage(
                role="assistant",
                content=f"Error communicating with the LLM: {e}",
            )

        # Parse response content blocks
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return AgentMessage(
            role="assistant",
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )


def create_agent_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    keep_alive: Optional[str] = None,
) -> BaseAgentProvider:
    """Create the appropriate agent provider."""
    provider = provider.lower().strip()

    if provider == "anthropic":
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or use --api-key."
            )
        return AnthropicAgentProvider(model=model, api_key=api_key)

    elif provider == "openai":
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key."
            )
        return OpenAIAgentProvider(model=model, api_key=api_key, base_url=base_url)

    elif provider == "ollama":
        return OllamaAgentProvider(
            model=model,
            base_url=base_url or "http://localhost:11434/v1",
            keep_alive=keep_alive,
        )

    else:
        raise ValueError(
            f"Unsupported agent provider: {provider}. "
            f"Supported: anthropic, openai, ollama"
        )

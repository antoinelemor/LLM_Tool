"""
Terminal-based conversational agent for LLMTool.

Provides a Rich-based chat interface where the user describes tasks in natural language
and the LLM agent orchestrates pipeline operations via tool_use.
"""

import json
import logging
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.text import Text
from rich.table import Table

from .config import AgentConfig
from .providers import AgentMessage, BaseAgentProvider, ToolCall, create_agent_provider
from .system_prompt import build_system_prompt
from .tool_executor import ToolExecutor
from .tools import get_tool_definitions

# Classic CLI display helpers (reused for agent welcome screen)
from ..cli.advanced_cli import LLMDetector, TrainerModelDetector
from ..utils.data_detector import DatasetInfo, DataDetector
from ..utils.resource_display import create_visual_resource_panel
from ..utils.system_resources import SystemResourceDetector

logger = logging.getLogger(__name__)


class AgentCLI:
    """Terminal-based conversational agent for LLMTool."""

    # Context management constants
    MAX_TOOL_RESULT_CHARS = 3000  # Truncate tool results in conversation history
    MAX_CONVERSATION_MESSAGES = 40  # Compact conversation when exceeding this
    COMPACT_KEEP_RECENT = 10  # Keep this many recent messages when compacting

    def __init__(self, config: AgentConfig, settings, model_explicit: bool = False):
        self.config = config
        self.settings = settings
        self.console = Console()
        self._model_explicit = model_explicit  # True if user passed --agent-model
        self.provider: Optional[BaseAgentProvider] = None
        self.executor = ToolExecutor(
            settings=settings,
            progress_callback=self._progress_callback,
        )
        self.conversation: List[AgentMessage] = []
        self.tools = get_tool_definitions()
        self._tool_name_set = {tool["name"] for tool in self.tools}
        self._turn_count = 0
        self._consecutive_text_turns = 0  # Track text-only responses for anti-loop
        self._state_message_tag = "[SESSION STATE]"
        self._state_message_index: Optional[int] = None
        self._start_new_session()
        self._reset_session_state()

    def _start_new_session(self):
        """Initialize a new agent session directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"agent_{timestamp}"
        base_dir = Path(self.settings.paths.data_dir)
        logs_dir = Path(self.settings.paths.logs_dir)
        self.session_base_dir = base_dir / "agent_sessions" / self.session_id
        self.session_audio_dir = self.session_base_dir / "audio"
        self.session_transcripts_dir = self.session_base_dir / "transcripts"
        self.session_annotations_dir = self.session_base_dir / "annotations"
        self.session_logs_dir = logs_dir / "agent_sessions" / self.session_id
        for dir_path in [
            self.session_base_dir,
            self.session_audio_dir,
            self.session_transcripts_dir,
            self.session_annotations_dir,
            self.session_logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.executor.set_session_id(self.session_id)

    def _create_provider(self):
        """Create the LLM provider with current config."""
        self.provider = create_agent_provider(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            keep_alive=self.config.keep_alive,
        )

    def _startup_model_selection(self):
        """Interactive provider/model selection at startup, styled like the classic CLI."""
        # If user explicitly set the model via CLI, skip selection
        if self._model_explicit:
            self._create_provider()
            return

        # ── Provider selection in a styled panel ──
        provider_table = Table.grid(padding=0)
        provider_table.add_column(width=3)
        provider_table.add_column()

        provider_table.add_row("[bold cyan]1[/bold cyan]", "🖥️  Ollama (local models)")
        provider_table.add_row("[bold cyan]2[/bold cyan]", "☁️  OpenAI (GPT)")
        provider_table.add_row("[bold cyan]3[/bold cyan]", "☁️  Anthropic (Claude)")

        self.console.print(Panel(
            provider_table,
            title="[bold]⚙️  Agent Configuration[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))

        provider_choice = Prompt.ask(
            "\n[bold yellow]Select provider[/bold yellow]",
            choices=["1", "2", "3"],
            default="1",
        )

        if provider_choice == "2":
            self.config.provider = "openai"
            self._select_cloud_model("openai")
        elif provider_choice == "3":
            self.config.provider = "anthropic"
            self._select_cloud_model("anthropic")
        else:
            self.config.provider = "ollama"
            self.config.base_url = self.config.base_url or "http://localhost:11434/v1"
            self._select_ollama_model()

        self._create_provider()
        self.console.print()

    def _select_ollama_model(self):
        """Let user pick from installed Ollama models, styled like the classic CLI."""
        self.console.print()

        # Use already-scanned Ollama models if available, else fetch
        res = getattr(self, '_scanned_resources', {})
        models = res.get("ollama_models", None)
        if models is None:
            models = self._get_ollama_models()

        if not models:
            self.console.print("[yellow]No Ollama models found. Using default: llama3.2[/yellow]")
            self.config.model = "llama3.2"
            return

        model_table = Table.grid(padding=(0, 1))
        model_table.add_column(width=3, justify="right")
        model_table.add_column(min_width=20)
        model_table.add_column(justify="right")

        for i, model in enumerate(models, 1):
            name = model["name"]
            size = model.get("size", "")
            marker = " [dim](default)[/dim]" if name == self.config.model else ""
            model_table.add_row(
                f"[bold cyan]{i}[/bold cyan]",
                f"{name}{marker}",
                f"[dim]{size}[/dim]" if size else "",
            )

        self.console.print(Panel(
            model_table,
            title="[bold]🤖 Select Ollama Model[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))

        default_idx = "1"
        for i, m in enumerate(models, 1):
            if m["name"] == self.config.model:
                default_idx = str(i)
                break

        choice = Prompt.ask(
            "\n[bold yellow]Model[/bold yellow]",
            default=default_idx,
        )

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                self.config.model = models[idx]["name"]
            else:
                self.console.print(f"[yellow]Invalid choice, using {self.config.model}[/yellow]")
        except ValueError:
            self.config.model = choice.strip()

    def _select_cloud_model(self, provider: str):
        """Let user configure a cloud provider model, styled like the classic CLI."""
        import os

        self.console.print()

        if provider == "openai":
            models = ["gpt-4o", "gpt-4o-mini", "o3-mini", "gpt-4.1"]
            default = "gpt-4o"
            env_key = "OPENAI_API_KEY"
        else:
            models = ["claude-sonnet-4-20250514", "claude-haiku-4-20250514", "claude-opus-4-20250514"]
            default = "claude-sonnet-4-20250514"
            env_key = "ANTHROPIC_API_KEY"

        model_table = Table.grid(padding=(0, 1))
        model_table.add_column(width=3, justify="right")
        model_table.add_column(min_width=30)

        for i, model in enumerate(models, 1):
            marker = " [dim](default)[/dim]" if model == default else ""
            model_table.add_row(
                f"[bold cyan]{i}[/bold cyan]",
                f"{model}{marker}",
            )

        self.console.print(Panel(
            model_table,
            title=f"[bold]☁️  Select {provider.title()} Model[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))

        choice = Prompt.ask("\n[bold yellow]Model[/bold yellow]", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                self.config.model = models[idx]
            else:
                self.config.model = default
        except ValueError:
            self.config.model = choice.strip()

        # Resolve API key
        api_key = os.environ.get(env_key) or self.settings.get_api_key(provider)
        if not api_key:
            api_key = Prompt.ask(f"\n[bold]{env_key}[/bold]", password=True)
        self.config.api_key = api_key

    def _get_ollama_models(self) -> List[dict]:
        """Get list of installed Ollama models with sizes."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []

            models = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if not parts:
                    continue
                name = parts[0]
                # Extract size like "9.1 GB" or "17 GB"
                size_match = re.search(r'([\d.]+\s*[GM]B)', line)
                size = size_match.group(1) if size_match else ""
                models.append({"name": name, "size": size})
            return models
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def run(self):
        """Main conversation loop."""
        # Show welcome banner first (classic CLI style), then model selection
        self._display_welcome_banner()
        self._startup_model_selection()
        self._display_agent_panel()
        self._init_conversation()
        self._send_first_agent_message()

        while True:
            try:
                user_input = self._get_user_input()
                if user_input is None:
                    break

                # Handle slash commands
                if user_input.startswith("/"):
                    should_exit = self._handle_slash_command(user_input)
                    if should_exit:
                        break
                    continue

                # Add user message
                self._update_state_from_user_input(user_input)
                self.conversation.append(AgentMessage(role="user", content=user_input))
                self._turn_count += 1

                # Check turn limit
                if self._turn_count > self.config.max_conversation_turns:
                    self.console.print(
                        "\n[yellow]Conversation turn limit reached. "
                        "Use /clear to start a new conversation.[/yellow]"
                    )
                    continue

                # Run agent turn
                self._run_agent_turn()

            except KeyboardInterrupt:
                self.console.print("\n[dim]Type /quit to exit[/dim]")
            except Exception as e:
                logger.exception("Agent error")
                self.console.print(f"\n[red]Error: {e}[/red]")

    def _init_conversation(self):
        """Initialize conversation with system prompt."""
        system_prompt = build_system_prompt(self.settings)
        self.conversation = [
            AgentMessage(role="system", content=system_prompt),
            AgentMessage(role="system", content=self._tool_list_message()),
        ]
        self._reset_session_state()
        self._refresh_state_message()

    def _tool_list_message(self) -> str:
        """Build a concise tool list reminder for the agent."""
        tools_sorted = ", ".join(sorted(self._tool_name_set))
        return (
            "[AVAILABLE TOOLS]\n"
            f"Use ONLY these tools: {tools_sorted}\n"
            "[END AVAILABLE TOOLS]"
        )

    def _reset_session_state(self):
        """Reset persistent session state for the agent."""
        self.session_state: Dict[str, Any] = {
            "session_id": self.session_id,
            "session_base_dir": str(self.session_base_dir),
            "session_audio_dir": str(self.session_audio_dir),
            "session_transcripts_dir": str(self.session_transcripts_dir),
            "session_annotations_dir": str(self.session_annotations_dir),
            "session_logs_dir": str(self.session_logs_dir),
            "goal": "annotation",
            "source_url": None,
            "language": None,
            "whisper_model": None,
            "diarize": None,
            "clean_audio": None,
            "audio_files": [],
            "transcripts": [],
            "annotations": [],
            "last_tool": None,
            "last_error": None,
        }
        self._state_message_index = None

    def _find_state_message_index(self) -> Optional[int]:
        for idx, msg in enumerate(self.conversation):
            if msg.role == "system" and msg.content and msg.content.startswith(self._state_message_tag):
                return idx
        return None

    def _refresh_state_message(self):
        """Update or insert the session state message in the conversation."""
        content = self._format_state_message()
        idx = self._find_state_message_index()
        if idx is None:
            system_count = sum(1 for msg in self.conversation if msg.role == "system")
            insert_at = system_count
            self.conversation.insert(insert_at, AgentMessage(role="system", content=content))
            self._state_message_index = insert_at
        else:
            self.conversation[idx].content = content
            self._state_message_index = idx

    def _format_state_message(self) -> str:
        """Format a compact session state summary for the agent."""
        state = self.session_state

        def _compact_list(values: List[str], limit: int = 3) -> str:
            if not values:
                return ""
            trimmed = values[-limit:]
            suffix = f" (+{len(values) - limit} more)" if len(values) > limit else ""
            return ", ".join(trimmed) + suffix

        lines = [self._state_message_tag]
        lines.append(f"session_id: {state.get('session_id')}")
        lines.append(f"session_base_dir: {state.get('session_base_dir')}")
        lines.append(f"session_audio_dir: {state.get('session_audio_dir')}")
        lines.append(f"session_transcripts_dir: {state.get('session_transcripts_dir')}")
        lines.append(f"session_annotations_dir: {state.get('session_annotations_dir')}")
        if state.get("goal"):
            lines.append(f"goal: {state['goal']}")
        if state.get("source_url"):
            lines.append(f"source_url: {state['source_url']}")
        if state.get("language"):
            lines.append(f"language: {state['language']}")
        if state.get("whisper_model"):
            lines.append(f"whisper_model: {state['whisper_model']}")
        if state.get("diarize") is not None:
            lines.append(f"diarize: {state['diarize']}")
        if state.get("clean_audio") is not None:
            lines.append(f"clean_audio: {state['clean_audio']}")
        audio_list = _compact_list(state.get("audio_files", []))
        if audio_list:
            lines.append(f"audio_files: {audio_list}")
        transcript_list = _compact_list(state.get("transcripts", []))
        if transcript_list:
            lines.append(f"transcripts: {transcript_list}")
        annotation_list = _compact_list(state.get("annotations", []))
        if annotation_list:
            lines.append(f"annotations: {annotation_list}")
        if state.get("last_tool"):
            lines.append(f"last_tool: {state['last_tool']}")
        if state.get("last_error"):
            lines.append(f"last_error: {state['last_error']}")

        lines.append("[END SESSION STATE]")
        return "\n".join(lines)

    def _truncate_tool_result(self, result_json: str) -> str:
        """Truncate tool result to fit in context window."""
        if len(result_json) <= self.MAX_TOOL_RESULT_CHARS:
            return result_json
        # Keep the first portion and add a truncation notice
        truncated = result_json[:self.MAX_TOOL_RESULT_CHARS]
        return truncated + '\n... [truncated, full result was shown to user]'

    def _extract_json_spans(self, text: str) -> List[Tuple[int, int]]:
        """Extract JSON object/array spans from text for fallback tool parsing."""
        spans: List[Tuple[int, int]] = []
        stack: List[str] = []
        start_idx: Optional[int] = None
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue
                stack.pop()
                if not stack and start_idx is not None:
                    spans.append((start_idx, i + 1))
                    start_idx = None

        return spans

    def _strip_json_from_text(self, text: str) -> str:
        """Remove JSON spans from text to avoid echoing tool call payloads."""
        spans = self._extract_json_spans(text)
        if not spans:
            return text
        parts = []
        last_end = 0
        for start, end in spans:
            parts.append(text[last_end:start])
            last_end = end
        parts.append(text[last_end:])
        cleaned = "".join(parts)
        return cleaned.strip()

    def _infer_data_format(self, file_path: str) -> Optional[str]:
        """Infer data format from file extension."""
        suffix = Path(file_path).suffix.lower()
        mapping = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".jsonl": "jsonl",
            ".parquet": "parquet",
        }
        return mapping.get(suffix)

    def _latest_file(self, directory: Path, extensions: Optional[List[str]] = None) -> Optional[str]:
        """Return the most recently modified file in a directory."""
        if not directory.exists():
            return None
        candidates = []
        for item in directory.rglob("*"):
            if not item.is_file():
                continue
            if extensions and item.suffix.lower() not in extensions:
                continue
            candidates.append(item)
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def _resolve_dir(self, value: Optional[str], default_dir: Path) -> str:
        """Resolve directory to absolute path with defaults."""
        if not value:
            return str(default_dir)
        path = Path(value)
        if not path.is_absolute():
            path = (Path(self.settings.paths.base_dir) / path).resolve()
        if path.exists():
            return str(path)
        return str(default_dir)

    def _resolve_file(self, value: Optional[str], fallback_dirs: Optional[List[Path]] = None) -> Optional[str]:
        """Resolve file path, searching fallback directories if needed."""
        if not value:
            return None
        path = Path(value)
        if path.exists():
            return str(path)
        if not path.is_absolute():
            candidate = (Path(self.settings.paths.base_dir) / path).resolve()
            if candidate.exists():
                return str(candidate)
        if fallback_dirs:
            name = path.name
            for folder in fallback_dirs:
                candidate = folder / name
                if candidate.exists():
                    return str(candidate)
        return None

    def _normalize_tool_arguments(self, tool_call: ToolCall) -> ToolCall:
        """Normalize tool arguments for agent mode (paths, defaults, aliases)."""
        args = tool_call.arguments
        if not isinstance(args, dict):
            return tool_call

        name = tool_call.name
        alias_map = {
            "list": "manage_prompts",
            "list_prompts": "manage_prompts",
            "prompt_list": "manage_prompts",
            "read_prompt": "manage_prompts",
            "show_prompt": "manage_prompts",
            "show_prompts": "manage_prompts",
        }
        if name in alias_map:
            tool_call.name = alias_map[name]
            name = tool_call.name
        normalized = dict(args)

        if name in ("download_youtube_audio", "download_tiktok_audio"):
            url = normalized.get("url") or normalized.get("source_url") or normalized.get("link") or self.session_state.get("source_url")
            if url:
                normalized["url"] = url
            normalized["output_dir"] = str(self.session_audio_dir)
            if "audio_format" not in normalized and "output_format" in normalized:
                normalized["audio_format"] = normalized["output_format"]

        elif name == "process_local_audio":
            normalized["output_dir"] = self._resolve_dir(normalized.get("output_dir"), self.session_audio_dir)
            if "file_path" in normalized:
                normalized["file_path"] = self._resolve_file(
                    normalized.get("file_path"),
                    fallback_dirs=[self.session_audio_dir, Path(self.settings.paths.data_dir)],
                )

        elif name == "transcribe_audio":
            audio_file = normalized.get("audio_file") or normalized.get("audio_path") or normalized.get("file_path")
            if not audio_file and self.session_state.get("audio_files"):
                audio_file = self.session_state["audio_files"][-1]
            if audio_file:
                normalized["audio_file"] = self._resolve_file(
                    audio_file,
                    fallback_dirs=[self.session_audio_dir],
                )
            if not normalized.get("audio_file"):
                normalized["audio_file"] = self._latest_file(
                    self.session_audio_dir,
                    extensions=[".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"],
                )
            if "output_dir" not in normalized:
                normalized["output_dir"] = str(self.session_transcripts_dir)
            else:
                normalized["output_dir"] = self._resolve_dir(normalized.get("output_dir"), self.session_transcripts_dir)
            if "output_format" not in normalized:
                normalized["output_format"] = "csv"
            if "language" not in normalized and self.session_state.get("language"):
                normalized["language"] = self.session_state.get("language")
            if "diarize" not in normalized and self.session_state.get("diarize") is not None:
                normalized["diarize"] = self.session_state.get("diarize")
            if "model" not in normalized and self.session_state.get("whisper_model"):
                normalized["model"] = self.session_state.get("whisper_model")

        elif name == "run_transcription_pipeline":
            audio_files = normalized.get("audio_files")
            if not audio_files and self.session_state.get("audio_files"):
                audio_files = self.session_state["audio_files"]
            if audio_files:
                resolved = []
                for item in audio_files:
                    resolved.append(self._resolve_file(item, fallback_dirs=[self.session_audio_dir]) or item)
                normalized["audio_files"] = resolved
            normalized["output_dir"] = self._resolve_dir(normalized.get("output_dir"), self.session_transcripts_dir)
            if "output_format" not in normalized:
                normalized["output_format"] = "csv"
            if "language" not in normalized and self.session_state.get("language"):
                normalized["language"] = self.session_state.get("language")
            if "diarize" not in normalized and self.session_state.get("diarize") is not None:
                normalized["diarize"] = self.session_state.get("diarize")
            if "model" not in normalized and self.session_state.get("whisper_model"):
                normalized["model"] = self.session_state.get("whisper_model")

        elif name == "run_annotation":
            file_path = normalized.get("file_path")
            if not file_path and self.session_state.get("transcripts"):
                file_path = self.session_state["transcripts"][-1]
            if not file_path:
                file_path = self._latest_file(self.session_transcripts_dir, extensions=[".csv", ".json", ".jsonl"])
            if file_path:
                normalized["file_path"] = self._resolve_file(file_path, fallback_dirs=[self.session_transcripts_dir])
            if not normalized.get("file_path"):
                normalized["file_path"] = file_path
            if "data_format" not in normalized and normalized.get("file_path"):
                inferred = self._infer_data_format(normalized["file_path"])
                if inferred:
                    normalized["data_format"] = inferred
            if "text_column" not in normalized:
                normalized["text_column"] = "text"
            if "output_path" not in normalized:
                stem = Path(normalized.get("file_path", "data")).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                normalized["output_path"] = str(self.session_annotations_dir / f"{stem}_annotations_{timestamp}.csv")

        elif name == "run_training":
            input_file = normalized.get("input_file")
            if not input_file and self.session_state.get("annotations"):
                input_file = self.session_state["annotations"][-1]
            if input_file:
                normalized["input_file"] = self._resolve_file(input_file, fallback_dirs=[self.session_annotations_dir])

        elif name == "run_bert_inference":
            input_file = normalized.get("input_file")
            if input_file:
                normalized["input_file"] = self._resolve_file(input_file, fallback_dirs=[self.session_base_dir])

        elif name == "manage_prompts":
            if "action" not in normalized or not normalized.get("action"):
                normalized["action"] = "list"
            if "prompt_path" not in normalized:
                prompt_candidate = normalized.get("prompt") or normalized.get("path")
                if prompt_candidate:
                    normalized["prompt_path"] = prompt_candidate

        tool_call.arguments = normalized
        return tool_call

    def _normalize_tool_calls(self, payload: Any) -> List[ToolCall]:
        """Normalize JSON payloads into ToolCall objects."""
        calls: List[ToolCall] = []
        if isinstance(payload, list):
            for item in payload:
                calls.extend(self._normalize_tool_calls(item))
            return calls

        if not isinstance(payload, dict):
            return calls

        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            for item in payload["tool_calls"]:
                calls.extend(self._normalize_tool_calls(item))
            return calls

        function_payload = payload.get("function") if isinstance(payload.get("function"), dict) else None
        name = (
            payload.get("action")
            or payload.get("tool")
            or payload.get("tool_name")
            or payload.get("name")
            or payload.get("function")
        )

        if isinstance(name, dict):
            name = name.get("name")

        args = None
        for key in ("args", "arguments", "input", "params", "parameters"):
            if key in payload:
                args = payload.get(key)
                break
        if args is None and function_payload:
            for key in ("args", "arguments", "input", "params", "parameters"):
                if key in function_payload:
                    args = function_payload.get(key)
                    break
        if args is None:
            args = {k: v for k, v in payload.items() if k not in {"action", "tool", "tool_name", "name", "function"}}

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}

        if name and isinstance(args, dict):
            calls.append(ToolCall(id=str(uuid.uuid4()), name=str(name), arguments=args))

        return calls

    def _extract_tool_calls_from_text(self, text: str) -> List[ToolCall]:
        """Extract tool calls from plain text when models don't support tool_use."""
        calls: List[ToolCall] = []
        spans = self._extract_json_spans(text)
        for start, end in spans:
            snippet = text[start:end]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            calls.extend(self._normalize_tool_calls(payload))
        return calls

    def _detect_auto_action(self) -> Optional[List[ToolCall]]:
        """Detect clear user intent and auto-generate tool calls when the model fails.

        This is a last-resort fallback: if the model responded with text instead of
        calling tools, we check the user's last message for unambiguous patterns
        (URLs, keywords) and auto-trigger the corresponding tool.
        """
        # Find the last user message
        last_user_msg = None
        for msg in reversed(self.conversation):
            if msg.role == "user" and msg.content and not msg.content.startswith("["):
                last_user_msg = msg.content
                break
        if not last_user_msg:
            return None

        text = last_user_msg
        text_lower = text.lower()

        # ── YouTube URL → download_youtube_audio ──
        yt_match = re.search(r'https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+', text)
        if yt_match:
            url = yt_match.group(0).rstrip('.,;:!?)')
            self.console.print(
                "\n[bold yellow]⚡ Auto-action:[/bold yellow] "
                f"[dim]Detected YouTube URL → calling download_youtube_audio[/dim]"
            )
            return [ToolCall(
                id=str(uuid.uuid4()),
                name="download_youtube_audio",
                arguments={"url": url},
            )]

        # ── TikTok URL → download_tiktok_audio ──
        tt_match = re.search(r'https?://(?:www\.)?(?:tiktok\.com|vm\.tiktok\.com)/\S+', text)
        if tt_match:
            url = tt_match.group(0).rstrip('.,;:!?)')
            self.console.print(
                "\n[bold yellow]⚡ Auto-action:[/bold yellow] "
                f"[dim]Detected TikTok URL → calling download_tiktok_audio[/dim]"
            )
            return [ToolCall(
                id=str(uuid.uuid4()),
                name="download_tiktok_audio",
                arguments={"url": url},
            )]

        # ── "transcribe" + known audio file in session state → transcribe_audio ──
        if any(kw in text_lower for kw in ("transcri", "transcris", "transcrip")):
            audio_files = self.session_state.get("audio_files", [])
            if audio_files:
                last_audio = audio_files[-1]
                model = self.session_state.get("whisper_model", "large-v3")
                diarize = self.session_state.get("diarize", False)
                language = self.session_state.get("language")
                args: Dict[str, Any] = {
                    "audio_file": last_audio,
                    "model": model,
                    "diarize": diarize,
                }
                if language:
                    args["language"] = language
                self.console.print(
                    "\n[bold yellow]⚡ Auto-action:[/bold yellow] "
                    f"[dim]Transcription requested → calling transcribe_audio[/dim]"
                )
                return [ToolCall(
                    id=str(uuid.uuid4()),
                    name="transcribe_audio",
                    arguments=args,
                )]

        # ── File path detected → analyze_dataset or preview_data ──
        file_match = re.search(r'[\w/\\.-]+\.(?:csv|json|jsonl|xlsx|xls|parquet|tsv)', text)
        if file_match:
            file_path = file_match.group(0)
            # Resolve relative to data dir if not absolute
            if not Path(file_path).is_absolute():
                candidate = self.settings.paths.data_dir / file_path
                if candidate.exists():
                    file_path = str(candidate)
            self.console.print(
                "\n[bold yellow]⚡ Auto-action:[/bold yellow] "
                f"[dim]Detected data file → calling analyze_dataset[/dim]"
            )
            return [ToolCall(
                id=str(uuid.uuid4()),
                name="analyze_dataset",
                arguments={"file_path": file_path},
            )]

        return None

    def _update_state_from_user_input(self, user_input: str):
        """Lightweight heuristic updates to session state from user input."""
        text = user_input.lower()
        self.session_state["last_error"] = None

        if "annot" in text:
            self.session_state["goal"] = "annotation"
        if "transcrib" in text or "transcript" in text:
            self.session_state["goal"] = "transcription"

        url_match = re.search(r"https?://(www\.)?(youtube\.com|youtu\.be|tiktok\.com)/\S+", user_input)
        if url_match:
            self.session_state["source_url"] = url_match.group(0)

        if "diar" in text or "speaker" in text:
            if any(neg in text for neg in ("no diar", "sans diar", "no speaker", "without diar")):
                self.session_state["diarize"] = False
            else:
                self.session_state["diarize"] = True

        if "clean" in text or "nettoie" in text or "denoise" in text:
            self.session_state["clean_audio"] = True

        model_match = re.search(r"\b(large-v3|large-v2|large|medium|small|base|tiny)\b", text)
        if model_match:
            self.session_state["whisper_model"] = model_match.group(1)
        elif "v3" in text and "large" in text:
            self.session_state["whisper_model"] = "large-v3"

        if "english" in text or "anglais" in text or " en " in f" {text} ":
            self.session_state["language"] = "en"
        elif "french" in text or "français" in text or " francais" in text or " fr " in f" {text} ":
            self.session_state["language"] = "fr"

        self._refresh_state_message()

    def _append_state_list(self, key: str, value: Optional[str], limit: int = 25):
        if not value:
            return
        values = self.session_state.get(key, [])
        if value not in values:
            values.append(value)
        if len(values) > limit:
            values = values[-limit:]
        self.session_state[key] = values

    def _update_state_from_tool_result(self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any]):
        """Update session state based on tool execution results."""
        if result.get("error"):
            self.session_state["last_error"] = result.get("error")
        self.session_state["last_tool"] = tool_name

        if tool_name in ("download_youtube_audio", "download_tiktok_audio"):
            if arguments.get("url"):
                self.session_state["source_url"] = arguments.get("url")
            if result.get("audio_path"):
                self._append_state_list("audio_files", result["audio_path"])
            elif result.get("results"):
                for item in result.get("results", []):
                    if item.get("path"):
                        self._append_state_list("audio_files", item["path"])

        elif tool_name == "process_local_audio":
            if result.get("audio_path"):
                self._append_state_list("audio_files", result["audio_path"])
            elif result.get("results"):
                for item in result.get("results", []):
                    if item.get("audio_path"):
                        self._append_state_list("audio_files", item["audio_path"])

        elif tool_name in ("transcribe_audio", "run_transcription_pipeline"):
            model = arguments.get("model")
            language = arguments.get("language")
            diarize = arguments.get("diarize")
            if model:
                self.session_state["whisper_model"] = model
            if language:
                self.session_state["language"] = language
            if diarize is not None:
                self.session_state["diarize"] = diarize

            if result.get("output_file"):
                self._append_state_list("transcripts", result["output_file"])
            elif result.get("results"):
                for item in result.get("results", []):
                    if item.get("output_file"):
                        self._append_state_list("transcripts", item["output_file"])

        elif tool_name in ("run_annotation", "run_full_pipeline"):
            annotation_results = result.get("annotation_results", {})
            output_file = annotation_results.get("output_file") or result.get("output_file")
            if output_file:
                self._append_state_list("annotations", output_file)

        self._refresh_state_message()

    def _compact_conversation(self):
        """Compact conversation history to prevent context overflow.

        Keeps the system prompt, a summary of old messages, and recent messages.
        """
        if len(self.conversation) <= self.MAX_CONVERSATION_MESSAGES:
            return

        system_messages = [m for m in self.conversation if m.role == "system"]
        non_system_messages = [m for m in self.conversation if m.role != "system"]
        if len(system_messages) + len(non_system_messages) <= self.MAX_CONVERSATION_MESSAGES:
            return

        # Messages to summarize (old) vs keep (recent)
        old_messages = non_system_messages[:-self.COMPACT_KEEP_RECENT]
        recent_messages = non_system_messages[-self.COMPACT_KEEP_RECENT:]

        if not old_messages:
            return

        # Build a compact summary of old exchanges
        summary_parts = []
        for msg in old_messages:
            if msg.role == "user":
                summary_parts.append(f"User said: {(msg.content or '')[:200]}")
            elif msg.role == "assistant" and msg.tool_calls:
                tool_names = [tc.name for tc in msg.tool_calls]
                summary_parts.append(f"Agent called: {', '.join(tool_names)}")
            elif msg.role == "assistant" and msg.content:
                summary_parts.append(f"Agent replied: {(msg.content or '')[:150]}")
            # Skip tool_result messages in summary (they're verbose)

        summary_text = (
            "[CONVERSATION HISTORY SUMMARY]\n"
            + "\n".join(summary_parts)
            + "\n[END SUMMARY — recent messages follow]"
        )

        # Rebuild conversation
        new_conversation = []
        new_conversation.extend(system_messages)
        new_conversation.append(AgentMessage(role="user", content=summary_text))
        new_conversation.append(AgentMessage(
            role="assistant",
            content="Understood. I have the conversation context. Continuing.",
        ))
        new_conversation.extend(recent_messages)

        self.conversation = new_conversation
        logger.info(
            f"Compacted conversation: {len(old_messages)} old messages → summary, "
            f"keeping {len(recent_messages)} recent"
        )

    def _run_agent_turn(self):
        """Execute one agent turn, which may involve multiple tool calls."""
        tool_call_count = 0

        # Compact conversation if too long
        self._compact_conversation()

        # Anti-loop: if the model has produced 2+ consecutive text-only responses
        # (no tool calls), inject a nudge to encourage action
        if self._consecutive_text_turns >= 2:
            nudge = AgentMessage(
                role="user",
                content=(
                    "[SYSTEM REMINDER] You have already asked questions in previous turns. "
                    "The user has provided information. Please call the appropriate tool NOW "
                    "with the information available. Use sensible defaults for anything not specified. "
                    "Do NOT ask more questions — ACT."
                ),
            )
            self.conversation.append(nudge)
            self._consecutive_text_turns = 0  # Reset after nudge

        while True:
            # Show thinking indicator
            with self.console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
                response = self.provider.chat(
                    messages=self.conversation,
                    tools=self.tools,
                    temperature=self.config.temperature,
                )

            # Add assistant response to conversation
            self.conversation.append(response)

            # Fallback: extract tool calls from JSON in plain text responses
            if not response.tool_calls and response.content:
                fallback_calls = self._extract_tool_calls_from_text(response.content)
                if fallback_calls:
                    response.tool_calls = fallback_calls
                    cleaned = self._strip_json_from_text(response.content)
                    response.content = cleaned if cleaned else None
                    self.conversation[-1] = response

            # Auto-action fallback: if the model still didn't call tools,
            # detect clear intent from the user's last message and auto-execute
            if not response.tool_calls:
                auto_calls = self._detect_auto_action()
                if auto_calls:
                    response.tool_calls = auto_calls
                    self.conversation[-1] = response

            # If the LLM returned text with no tool calls, display and break
            if not response.tool_calls:
                if response.content:
                    self._display_assistant_message(response.content)
                self._consecutive_text_turns += 1
                break

            # Tool calls made — reset the consecutive text counter
            self._consecutive_text_turns = 0

            # Display any text before tool calls
            if response.content:
                self._display_assistant_message(response.content)

            # Process tool calls
            tool_call_count += len(response.tool_calls)
            if tool_call_count > self.config.max_tool_calls_per_turn:
                self.console.print(
                    "\n[yellow]Tool call limit reached for this turn.[/yellow]"
                )
                break

            for tool_call in response.tool_calls:
                tool_call = self._normalize_tool_arguments(tool_call)
                self._display_tool_call(tool_call)

                if tool_call.name not in self._tool_name_set:
                    result = {
                        "error": (
                            f"Unknown tool: {tool_call.name}. "
                            "Use only the tools provided in the AVAILABLE TOOLS list."
                        )
                    }
                else:
                    # Execute the tool
                    result = self.executor.execute(tool_call.name, tool_call.arguments)

                self._display_tool_result(tool_call.name, result)
                self._display_contextual_hint(tool_call.name, result)
                self._update_state_from_tool_result(tool_call.name, tool_call.arguments, result)

                # Add tool result to conversation (truncated for context management)
                result_json = json.dumps(result, default=str, ensure_ascii=False)
                self.conversation.append(AgentMessage(
                    role="tool_result",
                    content=self._truncate_tool_result(result_json),
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                ))

            # Continue loop so LLM can process tool results

    # ── Display methods ──────────────────────────────────────────────────

    def _display_welcome_banner(self):
        """Show a rich welcome banner with environment scanning, like the classic CLI."""
        width = self.console.width

        self.console.print()
        self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
        self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
        self.console.print()

        # ── Giant LLM TOOL ASCII art ──
        self.console.print(Align.center("[bright_magenta]██╗     [bright_yellow]██╗     [bright_green]███╗   ███╗    [bright_cyan]████████╗ [bright_red]██████╗  [bright_blue]██████╗ [bright_white]██╗     "))
        self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]████╗ ████║    [bright_cyan]╚══██╔══╝[bright_red]██╔═══██╗[bright_blue]██╔═══██╗[bright_white]██║     "))
        self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██╔████╔██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
        self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██║╚██╔╝██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
        self.console.print(Align.center("[bright_magenta]███████╗[bright_yellow]███████╗[bright_green]██║ ╚═╝ ██║       [bright_cyan]██║   [bright_red]╚██████╔╝[bright_blue]╚██████╔╝[bright_white]███████╗"))
        self.console.print(Align.center("[bright_magenta]╚══════╝[bright_yellow]╚══════╝[bright_green]╚═╝     ╚═╝       [bright_cyan]╚═╝    [bright_red]╚═════╝  [bright_blue]╚═════╝ [bright_white]╚══════╝"))

        self.console.print()
        self.console.print(Align.center("[bold bright_yellow on blue]  🚀 LLM-powered Intelligent Annotation & Training Pipeline 🚀  [/bold bright_yellow on blue]"))
        self.console.print()

        # ── Pipeline flow with emojis ──
        pipeline_text = Text()
        pipeline_text.append("📊 Data ", style="bold bright_yellow on black")
        pipeline_text.append("→ ", style="bold white")
        pipeline_text.append("🤖 LLM Annotation ", style="bold bright_green on black")
        pipeline_text.append("→ ", style="bold white")
        pipeline_text.append("🧹 Clean ", style="bold bright_cyan on black")
        pipeline_text.append("→ ", style="bold white")
        pipeline_text.append("🎯 Label ", style="bold bright_magenta on black")
        pipeline_text.append("→ ", style="bold white")
        pipeline_text.append("🧠 Train ", style="bold bright_red on black")
        pipeline_text.append("→ ", style="bold white")
        pipeline_text.append("📈 Deploy ", style="bold bright_blue on black")

        self.console.print(Align.center(pipeline_text))
        self.console.print()
        self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
        self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
        self.console.print()

        # ── Welcome info panel ──
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_row("📚 Version:", "[bright_green]1.0[/bright_green]")
        info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")
        info_table.add_row("🚀 Features:", "[cyan]50+ BERT Models, 2 LLM Providers (OpenAI/Ollama), Parallel GPU/CPU, Reinforcement Learning[/cyan]")
        info_table.add_row("🎯 Capabilities:", "[magenta]Multi-Label Classification, 100+ Languages, SQL/File I/O, Doccano/Label Studio Export[/magenta]")

        self.console.print(Panel(
            info_table,
            title="[bold bright_cyan]✨ Welcome to LLM Tool ✨[/bold bright_cyan]",
            border_style="bright_blue",
            padding=(1, 2)
        ))
        self.console.print()

        # ── Environment scan ──
        self._scanned_resources = {}
        with self.console.status("[bold green]🔍 Scanning environment...", spinner="dots"):
            self._scan_environment()

        self._display_scanned_resources()

    def _display_agent_panel(self):
        """Display the agent interactive panel (shown after model selection)."""
        res = self._scanned_resources

        # Model info line
        model_line = Text()
        model_line.append("🤖 Active Model: ", style="bold white")
        model_line.append(self.config.provider.title(), style="bold cyan")
        model_line.append("/", style="dim")
        model_line.append(self.config.model, style="bold bright_green")

        # Capabilities menu (matching classic CLI menu style)
        cap_table = Table.grid(padding=0)
        cap_table.add_column(width=3)
        cap_table.add_column()

        capabilities = [
            ("🎤", "Data Preparation - Audio Extraction (YouTube/TikTok/Local) → Whisper Transcription (75+ Languages) → NLP Pipeline"),
            ("🎨", "LLM Annotation - Zero-Shot Annotation (OpenAI/Ollama) → Label Studio/Doccano Export"),
            ("🏭", "Annotator Factory - LLM Annotations → Training Data → Fine-Tuned BERT Models"),
            ("🎮", "Training Arena - Train 50+ Models (BERT/RoBERTa/DeBERTa) with Multi-Label & Benchmarking"),
            ("🤖", "BERT Annotation Studio - High-Throughput Inference (Parallel GPU/CPU, 100+ Languages)"),
            ("🔍", "Validation Lab - Quality Scoring, Stratified Sampling, Inter-Annotator Agreement"),
        ]

        for icon, desc in capabilities:
            cap_table.add_row(
                f"[bold cyan]{icon}[/bold cyan]",
                desc,
            )

        # Commands line
        commands = Text()
        commands.append("\n")
        commands.append("💬 Just describe what you want to do in natural language", style="italic bright_yellow")
        commands.append("\n")
        commands.append("/help", style="bold cyan")
        commands.append("  ", style="")
        commands.append("/status", style="bold cyan")
        commands.append("  ", style="")
        commands.append("/clear", style="bold cyan")
        commands.append("  ", style="")
        commands.append("/quit", style="bold cyan")

        # Smart subtitle (like classic CLI)
        subtitle_parts = []
        all_llms = res.get("all_llms", {})
        local_count = len(all_llms.get("local", []))
        if local_count:
            subtitle_parts.append(f"{local_count} local LLMs available")
        datasets = res.get("datasets", [])
        if datasets:
            subtitle_parts.append(f"{len(datasets)} datasets found")
        prompt_files = res.get("prompt_files", [])
        if prompt_files:
            subtitle_parts.append(f"{len(prompt_files)} prompts")
        subtitle = " | ".join(subtitle_parts) if subtitle_parts else None

        panel = Panel(
            Group(model_line, Text(), cap_table, commands),
            title="[bold]🤖 LLM Tool Agent[/bold]",
            subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _scan_environment(self):
        """Scan for available data files, prompts, Ollama models, trained models, LLMs, datasets, and system resources."""
        resources = {}

        # Scan Ollama models (lightweight, for backward compat)
        ollama_models = self._get_ollama_models()
        if ollama_models:
            resources["ollama_models"] = ollama_models

        # Scan ALL LLMs (Ollama + OpenAI + Anthropic) with full metadata
        try:
            resources["all_llms"] = LLMDetector.detect_all_llms()
        except Exception:
            resources["all_llms"] = {}

        # Scan available training models (50+ BERT variants)
        try:
            resources["trainer_models"] = TrainerModelDetector.get_available_models()
        except Exception:
            resources["trainer_models"] = {}

        # Scan data files
        data_dir = Path(self.settings.paths.data_dir)
        data_files = []
        if data_dir.exists():
            for ext in ("*.csv", "*.json", "*.jsonl", "*.xlsx", "*.parquet", "*.txt"):
                data_files.extend(data_dir.glob(ext))
        resources["data_files"] = sorted(data_files, key=lambda p: p.name)[:20]

        # Scan prompts
        prompts_dir = Path(self.settings.paths.prompts_dir)
        prompt_files = []
        if prompts_dir.exists():
            for ext in ("*.txt", "*.json", "*.md"):
                prompt_files.extend(p for p in prompts_dir.glob(ext) if p.name != "__pycache__")
        resources["prompt_files"] = sorted(prompt_files, key=lambda p: p.name)

        # Scan trained models
        models_dir = Path(self.settings.paths.models_dir)
        trained_models = []
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    trained_models.append(item)
        resources["trained_models"] = sorted(trained_models, key=lambda p: p.name)

        # Scan annotation outputs
        annotation_files = []
        for search_dir in [Path.cwd() / "annotations", Path(self.settings.paths.data_dir)]:
            if search_dir.exists():
                annotation_files.extend(search_dir.glob("*annotations*.csv"))
                annotation_files.extend(search_dir.glob("*annotated*.csv"))
        resources["annotation_files"] = sorted(set(annotation_files), key=lambda p: p.name)[:10]

        # Detect datasets with full metadata (format, size, columns)
        try:
            dataset_roots = []
            if data_dir.exists():
                dataset_roots.append((data_dir, "data"))
            logs_dir = Path(self.settings.paths.logs_dir)
            if logs_dir.exists():
                dataset_roots.append((logs_dir, "logs"))
                for sub in ["training_arena", "annotator_factory"]:
                    sub_dir = logs_dir / sub
                    if sub_dir.exists():
                        dataset_roots.append((sub_dir, f"logs/{sub}"))
            if dataset_roots:
                resources["datasets"] = DataDetector.scan_directories(dataset_roots)
            else:
                resources["datasets"] = []
        except Exception:
            resources["datasets"] = []

        # Detect system resources (GPU, CPU, RAM, Storage)
        try:
            resources["system_resources"] = SystemResourceDetector().detect_all()
        except Exception:
            resources["system_resources"] = None

        self._scanned_resources = resources

    def _display_scanned_resources(self):
        """Display scanned resources in the same detailed format as the classic CLI."""
        res = self._scanned_resources

        # ═══════════════════════════════════════════════════════════════════
        # 1) Available LLMs for Annotation (side-by-side with Training models)
        # ═══════════════════════════════════════════════════════════════════
        all_llms = res.get("all_llms", {})
        llms_table = Table(
            title="🤖 Available LLMs for Annotation",
            border_style="cyan",
            show_lines=True,
            expand=False,
            width=75,
        )
        llms_table.add_column("Provider", style="cyan", width=10)
        llms_table.add_column("Model", style="white", width=22)
        llms_table.add_column("Size", style="yellow", width=8)
        llms_table.add_column("Context", style="green", width=11)
        llms_table.add_column("Status", style="green", width=12)

        # Show all local (Ollama) models
        local_llms = all_llms.get("local", [])
        for model in local_llms:
            llms_table.add_row(
                "Ollama",
                model.name,
                model.size or "N/A",
                f"{model.context_length:,}" if model.context_length else "N/A",
                "✓ Ready",
            )

        # Show API models (top 2 per provider)
        for provider in ["openai", "anthropic"]:
            api_models = all_llms.get(provider, [])
            for model in api_models[:2]:
                llms_table.add_row(
                    provider.title(),
                    model.name,
                    "API",
                    f"{model.context_length:,}" if model.context_length else "N/A",
                    "🔑 API Key" if model.requires_api_key else "✓ Ready",
                )

        # ── Training models table ──
        trainer_models = res.get("trainer_models", {})
        trainer_table = Table(
            title="🏋️ Available Models for Training",
            border_style="magenta",
            show_lines=False,
            expand=False,
            width=85,
        )
        trainer_table.add_column("Category", style="magenta bold", width=20)
        trainer_table.add_column("Models", style="white", width=56)

        desired_order = [
            "Multilingual Models",
            "Long Document Models (Multilingual)",
            "Long Document Models (Language-Specific)",
            "Efficient Models",
            "English Models",
            "French Models",
            "Other Language Models",
        ]

        shown_categories = set()
        for category in desired_order:
            if category in trainer_models:
                models = trainer_models[category]
                model_names = [m["name"] for m in models[:4]]
                if len(models) > 4:
                    model_names.append(f"(+{len(models) - 4} more)")
                trainer_table.add_row(category, ", ".join(model_names))
                shown_categories.add(category)

        # Add any remaining categories
        for category, models in trainer_models.items():
            if category not in shown_categories:
                model_names = [m["name"] for m in models[:4]]
                if len(models) > 4:
                    model_names.append(f"(+{len(models) - 4} more)")
                trainer_table.add_row(category, ", ".join(model_names))

        # Display LLMs and Training models side by side
        self.console.print(Columns([llms_table, trainer_table], equal=False, expand=True))
        self.console.print()

        # ═══════════════════════════════════════════════════════════════════
        # 2) Detected Datasets
        # ═══════════════════════════════════════════════════════════════════
        datasets = res.get("datasets", [])
        datasets_table = Table(
            title="📊 Detected Datasets",
            border_style="yellow",
            show_lines=False,
            expand=True,
        )
        datasets_table.add_column("File", style="cyan", no_wrap=True, width=30)
        datasets_table.add_column("Format", style="white bold", width=12, justify="center")
        datasets_table.add_column("Size", style="green", width=10, justify="right")
        datasets_table.add_column("Origin", style="yellow", width=24, no_wrap=True)
        datasets_table.add_column("Columns", style="dim", width=35)

        if datasets:
            format_styles = {
                "CSV": "cyan bold",
                "JSON": "green bold",
                "JSONL": "blue bold",
                "EXCEL": "magenta bold",
                "PARQUET": "red bold",
                "TSV": "yellow bold",
            }
            for dataset in datasets:
                columns_preview = ", ".join(dataset.columns[:3]) if dataset.columns else "N/A"
                if len(dataset.columns) > 3:
                    columns_preview += f" (+{len(dataset.columns) - 3} more)"

                fmt_style = format_styles.get(dataset.format.upper(), "white")
                origin_label = dataset.source or ""
                if not origin_label:
                    try:
                        origin_label = str(dataset.path.parent.relative_to(Path.cwd()))
                    except ValueError:
                        origin_label = dataset.path.parent.name or dataset.path.parent.as_posix()

                datasets_table.add_row(
                    dataset.path.name,
                    f"[{fmt_style}]{dataset.format.upper()}[/{fmt_style}]",
                    f"{dataset.size_mb:.1f} MB" if dataset.size_mb else "Unknown",
                    origin_label,
                    columns_preview,
                )
        else:
            datasets_table.add_row("No datasets found", "-", "-", "-", "Place CSV/JSON files in data/")

        self.console.print(datasets_table)
        self.console.print()

        # ═══════════════════════════════════════════════════════════════════
        # 3) Supported Formats & Databases
        # ═══════════════════════════════════════════════════════════════════
        all_formats_text = Text(justify="center")
        all_formats_text.append("📦 Supported Formats: ", style="bold cyan")
        all_formats_text.append("CSV", style="cyan bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("JSON/JSONL", style="green bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("Excel", style="magenta bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("Parquet", style="red bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("RData", style="yellow bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("TSV", style="blue bold")
        all_formats_text.append("\n💾 Databases: ", style="bold cyan")
        all_formats_text.append("PostgreSQL", style="blue bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("MySQL", style="yellow bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("SQLite", style="green bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("MongoDB", style="magenta bold")

        self.console.print(Panel(all_formats_text, border_style="cyan", padding=(0, 2)))
        self.console.print()

        # ═══════════════════════════════════════════════════════════════════
        # 4) System Resources & Recommendations
        # ═══════════════════════════════════════════════════════════════════
        system_resources = res.get("system_resources")
        if system_resources:
            resource_panel = create_visual_resource_panel(
                system_resources,
                show_recommendations=True,
            )
            self.console.print(resource_panel)
            self.console.print()

        # ═══════════════════════════════════════════════════════════════════
        # 5) Compact environment summary (data files, prompts, trained models)
        # ═══════════════════════════════════════════════════════════════════
        data_files = res.get("data_files", [])
        prompt_files = res.get("prompt_files", [])
        trained_models = res.get("trained_models", [])

        env_table = Table(
            title="Environment",
            border_style="green",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold green",
            expand=False,
            padding=(0, 1),
        )
        env_table.add_column("Resource", style="dim")
        env_table.add_column("Found", style="green")
        env_table.add_column("Details", style="white")

        if data_files:
            names = ", ".join(f.name for f in data_files[:5])
            if len(data_files) > 5:
                names += f" (+{len(data_files) - 5} more)"
            env_table.add_row("Data files", str(len(data_files)), names)
        else:
            env_table.add_row("Data files", "0", "[dim]Place CSV/JSON/Excel files in data/[/dim]")

        if prompt_files:
            names = ", ".join(f.stem for f in prompt_files[:4])
            if len(prompt_files) > 4:
                names += f" (+{len(prompt_files) - 4} more)"
            env_table.add_row("Prompts", str(len(prompt_files)), names)
        else:
            env_table.add_row("Prompts", "0", "[dim]No prompt files found in prompts/[/dim]")

        if trained_models:
            names = ", ".join(m.name for m in trained_models[:3])
            if len(trained_models) > 3:
                names += f" (+{len(trained_models) - 3} more)"
            env_table.add_row("Trained models", str(len(trained_models)), names)
        else:
            env_table.add_row("Trained models", "0", "[dim]Train a model to see it here[/dim]")

        self.console.print(env_table)
        self.console.print()

    def _build_resource_context(self) -> str:
        """Build a resource context string to inject into the first turn."""
        res = self._scanned_resources
        lines = ["[ENVIRONMENT RESOURCES]"]

        # All LLMs (Ollama + OpenAI + Anthropic)
        all_llms = res.get("all_llms", {})
        for provider_key, label in [("local", "Ollama"), ("openai", "OpenAI"), ("anthropic", "Anthropic")]:
            provider_models = all_llms.get(provider_key, [])
            if provider_models:
                model_list = ", ".join(m.name for m in provider_models)
                lines.append(f"{label} models: {model_list}")

        # Fallback to simple Ollama list if all_llms is empty
        if not all_llms:
            ollama_models = res.get("ollama_models", [])
            if ollama_models:
                model_list = ", ".join(m["name"] for m in ollama_models)
                lines.append(f"Ollama models installed: {model_list}")

        # Training model categories
        trainer_models = res.get("trainer_models", {})
        if trainer_models:
            total = sum(len(v) for v in trainer_models.values())
            categories = ", ".join(trainer_models.keys())
            lines.append(f"Training models: {total} models across categories: {categories}")

        # Data files
        data_files = res.get("data_files", [])
        if data_files:
            file_list = ", ".join(f.name for f in data_files[:10])
            lines.append(f"Data files available: {file_list}")
        else:
            lines.append("Data files: none found (user can provide files or URLs)")

        # Prompts
        prompt_files = res.get("prompt_files", [])
        if prompt_files:
            prompt_list = ", ".join(f.name for f in prompt_files)
            lines.append(f"Annotation prompts available: {prompt_list}")
        else:
            lines.append("Annotation prompts: none found")

        # Trained models
        trained_models = res.get("trained_models", [])
        if trained_models:
            model_list = ", ".join(m.name for m in trained_models[:5])
            lines.append(f"Trained models available: {model_list}")

        # Datasets summary
        datasets = res.get("datasets", [])
        if datasets:
            lines.append(f"Detected datasets: {len(datasets)} files")

        # System resources (SystemResources dataclass)
        system_resources = res.get("system_resources")
        if system_resources:
            try:
                gpu = system_resources.gpu
                if gpu and gpu.available:
                    gpu_name = ", ".join(gpu.device_names) if gpu.device_names else gpu.device_type
                    lines.append(f"GPU: {gpu_name} ({gpu.total_memory_gb:.0f} GB)")
                mem = system_resources.memory
                if mem:
                    lines.append(f"RAM: {mem.total_gb:.0f} GB total, {mem.available_gb:.0f} GB free")
            except Exception:
                pass

        lines.append("[END ENVIRONMENT RESOURCES]")
        return "\n".join(lines)

    def _send_first_agent_message(self):
        """Trigger the agent to deliver its own engaging welcome message."""
        # Detect likely language from system locale
        lang_hint = "en"
        try:
            import locale
            # getlocale() preferred over deprecated getdefaultlocale()
            loc = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "")
            if "fr" in loc.lower():
                lang_hint = "fr"
        except Exception:
            pass

        # Build context with discovered resources
        resource_context = self._build_resource_context()

        first_turn_msg = f"[FIRST_TURN] lang:{lang_hint}\n{resource_context}"
        self.conversation.append(AgentMessage(role="user", content=first_turn_msg))

        # Run the agent turn to get its welcome
        self._run_agent_turn()

    def _get_user_input(self) -> Optional[str]:
        """Get input from user."""
        try:
            self.console.print()
            self.console.rule(style="dim")
            user_input = Prompt.ask("[bold green] You [/bold green]")
            if not user_input or user_input.strip().lower() in ("/quit", "/exit", "/q"):
                self.console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
                return None
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
            return None

    def _display_assistant_message(self, content: str):
        """Show agent response with enhanced visual framing."""
        self.console.print()

        # Agent label with badge styling
        label = Text()
        label.append(" Agent ", style="bold white on blue")
        self.console.print(label)
        self.console.print()

        # Render content as Markdown inside a subtle left-border panel
        try:
            md = Markdown(content)
            panel = Panel(
                md,
                border_style="blue",
                box=box.SIMPLE,
                padding=(0, 1),
                expand=True,
            )
            self.console.print(panel)
        except Exception:
            self.console.print(content)

    def _display_tool_call(self, tool_call: ToolCall):
        """Show the tool being called with structured formatting."""
        self.console.print()

        # Tool name with badge
        tool_label = Text()
        tool_label.append(" TOOL ", style="bold black on yellow")
        tool_label.append("  ", style="")
        tool_label.append(tool_call.name, style="bold yellow")
        self.console.print(tool_label)

        # Arguments as a compact key=value list
        if tool_call.arguments:
            args_items = list(tool_call.arguments.items())[:6]
            for key, value in args_items:
                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                self.console.print(f"    [dim]{key}:[/dim] {val_str}")

    def _display_tool_result(self, tool_name: str, result: dict):
        """Show tool result with visual distinction for success/error."""
        if result.get("error"):
            error_text = Text()
            error_text.append(" ERROR ", style="bold white on red")
            error_text.append(f"  {result['error']}", style="red")
            self.console.print(error_text)
        else:
            summary = self._summarize_result(tool_name, result)
            result_text = Text()
            result_text.append(" DONE ", style="bold white on green")
            result_text.append(f"  {summary}", style="green")
            self.console.print(result_text)
        self.console.print()

    def _summarize_result(self, tool_name: str, result: dict) -> str:
        """Generate a brief summary line for a tool result."""
        if tool_name == "list_available_resources":
            counts = []
            for key, val in result.items():
                if isinstance(val, list):
                    counts.append(f"{len(val)} {key.replace('_', ' ')}")
            return ", ".join(counts) if counts else ""

        elif tool_name == "preview_data":
            shape = result.get("shape", {})
            cols = result.get("columns", [])
            return f"{shape.get('rows', '?')} rows, {len(cols)} columns"

        elif tool_name in ("run_annotation", "run_full_pipeline"):
            ar = result.get("annotation_results", {})
            total = ar.get("total_annotated", result.get("total_annotated", "?"))
            return f"{total} items annotated"

        elif tool_name == "download_youtube_audio":
            if result.get("audio_path"):
                return f"saved {Path(result['audio_path']).name}"
            if result.get("downloaded") is not None:
                return f"{result.get('downloaded', 0)} downloaded"
            return ""

        elif tool_name == "download_tiktok_audio":
            if result.get("audio_path"):
                return f"saved {Path(result['audio_path']).name}"
            if result.get("downloaded") is not None:
                return f"{result.get('downloaded', 0)} downloaded"
            return ""

        elif tool_name == "transcribe_audio":
            output_file = result.get("output_file")
            if output_file:
                return f"output {Path(output_file).name}"
            return ""

        elif tool_name == "run_transcription_pipeline":
            transcribed = result.get("transcribed", "?")
            total = result.get("total_files", "?")
            output_dir = result.get("output_dir")
            if output_dir:
                return f"{transcribed}/{total} → {output_dir}"
            return f"{transcribed}/{total}"

        elif tool_name == "manage_prompts":
            if result.get("count") is not None:
                return f"{result.get('count')} prompts"
            if result.get("prompts"):
                return f"{len(result.get('prompts'))} prompts"
            return ""

        elif tool_name == "run_training":
            return f"best={result.get('best_model', '?')}, f1={result.get('f1', '?')}"

        elif tool_name == "run_bert_inference":
            return f"{result.get('total_predicted', '?')} predictions"

        elif tool_name == "run_validation":
            return f"{result.get('total_samples', '?')} samples validated"

        elif tool_name == "analyze_dataset":
            rows = result.get("num_rows", result.get("shape", {}).get("rows", "?"))
            cols = result.get("num_columns", result.get("shape", {}).get("columns", "?"))
            return f"{rows} rows, {cols} columns"

        elif tool_name == "estimate_annotation_cost":
            cost = result.get("estimated_cost", result.get("cost", "?"))
            tokens = result.get("estimated_tokens", result.get("tokens", "?"))
            return f"~{tokens} tokens, est. ${cost}"

        elif tool_name == "split_dataset":
            train = result.get("train_size", "?")
            val = result.get("val_size", "?")
            test = result.get("test_size", "?")
            return f"train={train}, val={val}, test={test}"

        elif tool_name == "load_document":
            chars = result.get("num_characters", result.get("length", "?"))
            return f"{chars} characters extracted"

        return ""

    def _display_contextual_hint(self, tool_name: str, result: dict):
        """Show a contextual hint about what the user can do next."""
        if result.get("error"):
            return

        hints = {
            "download_youtube_audio": "Try: 'transcribe it' or 'transcribe in French with diarization'",
            "download_tiktok_audio": "Try: 'transcribe it' or 'transcribe with speaker diarization'",
            "process_local_audio": "Try: 'transcribe all files' or 'transcribe with the large-v3 model'",
            "transcribe_audio": "Try: 'annotate with the sentiment prompt' or 'show me a preview'",
            "run_transcription_pipeline": "Try: 'annotate the transcripts' or 'preview the data'",
            "run_annotation": "Try: 'train a model' or 'export to Doccano for review'",
            "run_full_pipeline": "Try: 'run inference on new data' or 'validate the results'",
            "run_training": "Try: 'run inference on new data' or 'benchmark with more models'",
            "analyze_dataset": "Try: 'annotate it' or 'detect the language'",
            "preview_data": "Try: 'annotate this file' or 'analyze the dataset'",
            "detect_language": "Try: 'annotate with the appropriate prompt' or 'train a model'",
            "estimate_annotation_cost": "Try: 'annotate it' or 'try with a different provider'",
            "detect_system_resources": "Try: 'train a model' or 'transcribe with the large-v3 model'",
            "list_available_resources": "Try: 'preview [filename]' or 'annotate [filename]'",
            "manage_prompts": "Try: 'annotate with [prompt]' or 'estimate annotation cost'",
            "run_bert_inference": "Try: 'validate the predictions' or 'export to Doccano'",
            "run_validation": "Try: 'export consensus labels' or 'train on the validated data'",
            "load_document": "Try: 'annotate it' or 'tokenize the text'",
            "tokenize_text": "Try: 'annotate the tokenized data' or 'preview it'",
            "convert_annotations_to_training": "Try: 'train a model' or 'split the dataset'",
            "export_to_doccano": "Try: 'import the JSONL into Doccano for human review'",
            "split_dataset": "Try: 'train a model on the training split'",
        }

        hint = hints.get(tool_name)
        if hint:
            self.console.print(f"  [dim italic]{hint}[/dim italic]")

    # ── Slash commands ───────────────────────────────────────────────────

    def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if should exit."""
        cmd = command.strip().lower().split()[0]
        arg = command.strip()[len(cmd):].strip() if len(command.strip()) > len(cmd) else ""

        if cmd in ("/quit", "/exit", "/q"):
            self.console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
            return True

        elif cmd == "/clear":
            self._turn_count = 0
            self._consecutive_text_turns = 0
            self._start_new_session()
            self._init_conversation()
            self.console.clear()
            self._display_welcome_banner()
            self._send_first_agent_message()

        elif cmd == "/status":
            self._show_status()

        elif cmd == "/classic":
            self.console.print("\n[cyan]Switching to classic CLI...[/cyan]\n")
            try:
                from ..cli.advanced_cli import AdvancedCLI
                cli = AdvancedCLI()
                cli.run()
            except ImportError:
                from ..cli.main_cli import LLMToolCLI
                cli = LLMToolCLI()
                cli.run()
            return True

        elif cmd == "/help":
            self._show_help()

        elif cmd == "/provider":
            self._switch_provider(arg)

        else:
            self.console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            self.console.print("[dim]Available: /quit /clear /status /classic /help /provider[/dim]")

        return False

    def _show_status(self):
        """Show current agent status with enhanced visual design."""
        self.console.print()

        # Configuration table
        info_table = Table(
            title="Agent Status",
            border_style="cyan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        info_table.add_column("Setting", style="dim", no_wrap=True)
        info_table.add_column("Value", style="cyan")

        info_table.add_row("Provider", f"[bold]{self.config.provider}[/bold]")
        info_table.add_row("Model", f"[bold]{self.config.model}[/bold]")
        info_table.add_row("Temperature", str(self.config.temperature))
        info_table.add_row("Session", self.session_id)
        info_table.add_row("Turns", f"{self._turn_count}/{self.config.max_conversation_turns}")
        info_table.add_row("Messages in context", str(len(self.conversation)))
        info_table.add_row("Session directory", str(self.session_base_dir))

        self.console.print(info_table)

        # Session progress (show what data has been processed)
        state = self.session_state
        has_progress = any([
            state.get("source_url"), state.get("audio_files"),
            state.get("transcripts"), state.get("annotations"),
            state.get("language"), state.get("whisper_model"),
        ])
        if has_progress:
            self.console.print()
            progress_table = Table(
                title="Session Progress",
                border_style="green",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold green",
            )
            progress_table.add_column("Stage", style="dim")
            progress_table.add_column("Status", style="green")

            if state.get("source_url"):
                progress_table.add_row("Source URL", state["source_url"])
            if state.get("audio_files"):
                progress_table.add_row("Audio files", f"{len(state['audio_files'])} file(s)")
            if state.get("transcripts"):
                progress_table.add_row("Transcripts", f"{len(state['transcripts'])} file(s)")
            if state.get("annotations"):
                progress_table.add_row("Annotations", f"{len(state['annotations'])} file(s)")
            if state.get("language"):
                progress_table.add_row("Language", state["language"])
            if state.get("whisper_model"):
                progress_table.add_row("Whisper model", state["whisper_model"])
            if state.get("last_tool"):
                progress_table.add_row("Last tool used", state["last_tool"])

            self.console.print(progress_table)

    def _show_help(self):
        """Show help information with enhanced visual design."""
        self.console.print()

        # Commands table
        cmd_table = Table(
            title="Commands",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        cmd_table.add_column("Command", style="bold yellow", no_wrap=True)
        cmd_table.add_column("Description", style="white")

        cmd_table.add_row("/quit, /exit", "Exit the agent")
        cmd_table.add_row("/clear", "Clear conversation and start a fresh session")
        cmd_table.add_row("/status", "Show agent configuration and session info")
        cmd_table.add_row("/classic", "Switch to the classic interactive CLI")
        cmd_table.add_row("/provider <name> <model>", "Switch LLM provider (e.g., /provider ollama llama3.2)")
        cmd_table.add_row("/help", "Show this help")

        self.console.print(cmd_table)
        self.console.print()

        # Examples table
        ex_table = Table(
            title="Example Commands (just type naturally)",
            border_style="green",
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 1),
        )
        ex_table.add_column("Example", style="italic")

        ex_table.add_row('"List my available data files and prompts"')
        ex_table.add_row('"Annotate my_data.csv using GPT-4o with the sentiment prompt"')
        ex_table.add_row('"Transcribe this YouTube video: https://youtube.com/..."')
        ex_table.add_row('"Train a CamemBERT model on my annotated data"')
        ex_table.add_row('"Run inference on new_data.csv with my trained model"')
        ex_table.add_row('"How much would it cost to annotate 10,000 tweets with GPT-4o?"')

        self.console.print(ex_table)

    def _switch_provider(self, arg: str):
        """Switch the agent LLM provider on the fly."""
        parts = arg.strip().split()
        if len(parts) < 1:
            self.console.print("[yellow]Usage: /provider <name> [model][/yellow]")
            self.console.print("[dim]Examples: /provider ollama llama3.2, /provider openai gpt-4o[/dim]")
            return

        new_provider = parts[0]
        new_model = parts[1] if len(parts) > 1 else None

        # Determine model defaults
        if not new_model:
            defaults = {"ollama": "llama3.2", "openai": "gpt-4o", "anthropic": "claude-sonnet-4-20250514"}
            new_model = defaults.get(new_provider, "llama3.2")

        # Resolve API key
        api_key = self.config.api_key
        if new_provider == "anthropic":
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY") or self.settings.get_api_key("anthropic")
        elif new_provider == "openai":
            import os
            api_key = os.environ.get("OPENAI_API_KEY") or self.settings.get_api_key("openai")

        try:
            self.provider = create_agent_provider(
                provider=new_provider,
                model=new_model,
                api_key=api_key,
                base_url=self.config.base_url if new_provider == "ollama" else None,
                keep_alive=self.config.keep_alive if new_provider == "ollama" else None,
            )
            self.config.provider = new_provider
            self.config.model = new_model
            self.config.api_key = api_key
            self.console.print(
                f"\n[green]Switched to[/green] [cyan bold]{new_provider}/{new_model}[/cyan bold]"
            )
        except Exception as e:
            self.console.print(f"\n[red]Failed to switch provider: {e}[/red]")

    # ── Progress callback ────────────────────────────────────────────────

    def _progress_callback(self, phase: str, progress: float, message: str, subtask=None):
        """Callback for long-running pipeline operations with enhanced styling."""
        bar_len = 30
        filled = int(progress / 100 * bar_len)
        bar_filled = "[green]" + "=" * filled + "[/green]"
        bar_empty = "[dim]" + "-" * (bar_len - filled) + "[/dim]"
        pct_style = "bold green" if progress >= 100 else "bold cyan"
        self.console.print(
            f"\r  [bold cyan]{phase}[/bold cyan] [{bar_filled}{bar_empty}] [{pct_style}]{progress:.0f}%[/{pct_style}] [dim]{message}[/dim]",
            end="",
            highlight=False,
        )
        if progress >= 100:
            self.console.print()

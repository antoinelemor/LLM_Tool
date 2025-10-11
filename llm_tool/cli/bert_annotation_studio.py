#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
bert_annotation_studio.py

MAIN OBJECTIVE:
---------------
Advanced annotation mode using trained BERT models with sophisticated features including
pipeline configuration, multi-model inference, language detection, and parallel processing.

Dependencies:
-------------
- os
- json
- re
- collections
- pathlib
- typing
- datetime
- time
- pandas
- numpy
- rich
- sqlalchemy
- transformers
- llm_tool.trainers.parallel_inference
- llm_tool.cli.advanced_cli
- llm_tool.utils.system_resources
- llm_tool.utils.language_detector

MAIN FEATURES:
--------------
1) Trained BERT model selection and management
2) Multi-model pipeline configuration with ordering and reduction
3) Dataset loading and column mapping
4) Language detection and validation
5) Text correction and preprocessing
6) Parallel inference with batch processing
7) Multiple data source support (CSV, Excel, JSON, Parquet, PostgreSQL)
8) Export options (CSV, JSON, Label Studio, Doccano)
9) Step-by-step wizard interface
10) Performance monitoring and progress tracking

Author:
-------
Antoine Lemor
"""

from __future__ import annotations
import os
import json
import re
from collections import Counter, defaultdict
import textwrap
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Iterable, Union, Mapping
from datetime import datetime
import time
import shutil
from contextlib import nullcontext
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich import box
from rich.align import Align
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import create_engine, inspect, text

# Package imports
from llm_tool.trainers.parallel_inference import parallel_predict
from llm_tool.cli.advanced_cli import LanguageNormalizer
from llm_tool.utils.data_detector import DataDetector
from llm_tool.utils.system_resources import detect_resources
from llm_tool.utils.language_detector import LanguageDetector
from llm_tool.utils.annotation_session_manager import AnnotationStudioSessionManager
from transformers import AutoTokenizer

# Optional progress bar support
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


MODEL_LANGUAGE_MAP = {
    'bert': 'EN',
    'camembert': 'FR',
    'arabic-bert': 'AR',
    'chinese-bert': 'ZH',
    'german-bert': 'DE',
    'hindi-bert': 'HI',
    'italian-bert': 'IT',
    'portuguese-bert': 'PT',
    'russian-bert': 'RU',
    'spanish-bert': 'ES',
    'swedish-bert': 'SV',
    'xlm-roberta': 'MULTI',
}


class BERTAnnotationStudio:
    """Advanced annotation studio for trained BERT models"""

    def __init__(
        self,
        console: Console,
        settings,
        logger,
        *,
        session_base_dir: Optional[Union[str, Path]] = None,
        allowed_model_paths: Optional[Iterable[Union[str, Path]]] = None,
        default_session_slug: Optional[str] = None,
        factory_session_id: Optional[str] = None,
        factory_context: Optional[Dict[str, Any]] = None,
    ):
        self.console = console
        self.settings = settings
        self.logger = logger
        self.models_dir = Path(getattr(self.settings.paths, "models_dir", "models"))
        self.data_dir = Path(getattr(self.settings.paths, "data_dir", "data"))
        self._language_assignments: Optional[pd.Series] = None
        self._language_annotation_mask: Optional[pd.Series] = None
        self._allowed_annotation_languages: List[str] = []
        self._total_steps: int = 9
        self._is_factory_context: bool = False
        self._annotated_row_mask: Optional[pd.Series] = None
        self._factory_annotated_count: int = 0
        self._factory_notice_shown: Set[str] = set()
        allowed_set: Optional[Set[Path]] = None
        if allowed_model_paths:
            allowed_set = set()
            for candidate in allowed_model_paths:
                try:
                    allowed_set.add(Path(candidate).expanduser().resolve())
                except Exception:
                    allowed_set.add(Path(candidate).expanduser())
        self._allowed_model_paths: Optional[Set[Path]] = allowed_set

        base_dir = Path(session_base_dir).expanduser() if session_base_dir else Path("logs") / "annotation_studio"
        self.session_manager = AnnotationStudioSessionManager(
            base_dir=base_dir,
            console=self.console,
            logger=self.logger,
        )

        self._session_base_dir = base_dir
        sanitized_default = (
            AnnotationStudioSessionManager.slugify(default_session_slug)
            if default_session_slug
            else "bert_annotation"
        )
        self._default_session_slug = sanitized_default
        self._factory_session_id = factory_session_id
        self._factory_context: Dict[str, Any] = dict(factory_context or {})
        self._factory_launch_config: Optional[Dict[str, Any]] = None
        self._factory_launch_active: bool = False

        self.session_id: Optional[str] = None
        self._resume_mode: bool = False
        self._resume_from_step: int = 1
        self._step_cache: Dict[str, Any] = {}

    @staticmethod
    def _sanitize_for_metadata(payload: Any, depth: int = 0) -> Any:
        """
        Convert arbitrary objects into JSON-serialisable structures.

        Non-serialisable values are replaced by their string representation so
        session metadata can always be persisted.
        """
        if depth > 6:  # avoid excessively deep recursion
            return str(payload)
        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload
        if isinstance(payload, Path):
            return str(payload)
        if isinstance(payload, Mapping):
            return {
                str(key): BERTAnnotationStudio._sanitize_for_metadata(value, depth + 1)
                for key, value in payload.items()
            }
        if isinstance(payload, (list, tuple, set)):
            return [
                BERTAnnotationStudio._sanitize_for_metadata(item, depth + 1)
                for item in payload
            ]
        return str(payload)

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def _prompt_session_name(self) -> str:
        if self.console:
            self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
            self.console.print("[bold cyan]           📝 Session Name Configuration                       [/bold cyan]")
            self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
            self.console.print("[bold]Give this session a descriptive identifier.[/bold]")
            self.console.print("  • [green]Traceability:[/green] link annotations, exports, and metrics")
            self.console.print("  • [green]Collaboration:[/green] teammates understand the use-case")
            self.console.print("  • [green]Audit trail:[/green] timestamp guarantees uniqueness\n")
            self.console.print("[dim]Format: {session_name}_{yyyymmdd_hhmmss}[/dim]")
            self.console.print("[dim]Example: legal_reviews_20251130_093050[/dim]\n")

        raw_name = Prompt.ask(
            "[bold yellow]Enter session name[/bold yellow]",
            default=getattr(self, "_default_session_slug", "bert_annotation"),
        ).strip()
        slug = AnnotationStudioSessionManager.slugify(raw_name)
        if self.console and slug != raw_name:
            self.console.print(f"[dim]Sanitized session name:[/dim] {slug}")
        return slug

    def _show_session_history(self) -> None:
        sessions = self.session_manager.list_sessions(limit=40)
        if not sessions:
            if self.console:
                self.console.print("[yellow]No previous sessions to display.[/yellow]")
            return
        self.session_manager.render_sessions_table(sessions)

    def _initialize_session(
        self,
        session_action: Optional[str] = None,
        resume_session_id: Optional[str] = None,
    ) -> bool:
        """Display the navigation menu and prepare the active session."""
        action = session_action
        while True:
            if action is None:
                if self.console:
                    if Table and Panel:
                        options_table = Table(show_header=False, box=box.ROUNDED if box else None, padding=(0, 2))
                        options_table.add_column("Option", style="cyan", width=8)
                        options_table.add_column("Description", style="white")
                        options_table.add_row("[bold cyan]1[/bold cyan]", "🆕 Start new session (recommended)")
                        options_table.add_row("[bold cyan]2[/bold cyan]", "🔄 Resume existing session")
                        options_table.add_row("[bold cyan]3[/bold cyan]", "📚 View session history")
                        options_table.add_row("[bold cyan]0[/bold cyan]", "⬅️  Back")
                        panel = Panel(options_table, title="[bold]Annotation Studio Navigator[/bold]", border_style="cyan")
                        self.console.print(panel)
                    else:
                        self.console.print("\n1) Start new session")
                        self.console.print("2) Resume existing session")
                        self.console.print("3) View session history")
                        self.console.print("0) Back")
                action = Prompt.ask(
                    "\n[bold yellow]Select an option[/bold yellow]",
                    choices=["0", "1", "2", "3"],
                    default="1",
                )

            if action == "0":
                return False

            if action == "3":
                self._show_session_history()
                action = None
                continue

            if action == "1":
                session_slug = self._prompt_session_name()
                self.session_id = self.session_manager.start_new_session(session_slug)
                self._resume_mode = False
                self._resume_from_step = 1
                self._step_cache = {}
                if self.console:
                    self.console.print(f"\n[bold green]✓ Session ID:[/bold green] [cyan]{self.session_id}[/cyan]")
                    self.console.print(f"[dim]Logs: {self.session_manager.session_dir}[/dim]")
                return True

            if action == "2":
                if self._resume_existing_session(resume_session_id):
                    return True
                action = None
                continue

            # Unknown action -> re-display menu
            action = None

    def _resume_existing_session(self, preferred_id: Optional[str] = None) -> bool:
        sessions = self.session_manager.list_sessions(limit=40)
        if not sessions:
            if self.console:
                self.console.print("[yellow]No saved sessions available yet.[/yellow]")
            return False

        chosen_session: Optional[Dict[str, Any]] = None
        if preferred_id:
            chosen_session = next(
                (entry for entry in sessions if entry["summary"].session_id == preferred_id),
                None,
            )
            if chosen_session is None and self.console:
                self.console.print(f"[yellow]Session '{preferred_id}' not found. Showing session selector.[/yellow]")

        while chosen_session is None:
            self.session_manager.render_sessions_table(sessions)
            choices = [str(i) for i in range(1, len(sessions) + 1)]
            selection = Prompt.ask(
                "\n[bold yellow]Choose a session (0 to cancel)[/bold yellow]",
                choices=choices + ["0"],
                default="1",
            )
            if selection == "0":
                return False
            chosen_session = sessions[int(selection) - 1]

        summary = chosen_session["summary"]
        session_id = summary.session_id
        self.session_manager.resume_session(session_id)
        self.session_id = session_id
        self._step_cache = dict(self.session_manager.step_cache)
        self._resume_mode = True

        if self.console:
            self.console.print(f"\n[bold green]✓ Loaded session:[/bold green] [cyan]{session_id}[/cyan]")
            self.console.print(f"[dim]Directory: {self.session_manager.session_dir}[/dim]\n")
            self.session_manager.render_step_status()
            last_step_display = summary.last_step_name or summary.last_step_key
            if last_step_display:
                prefix = f"{summary.last_step_no}. " if summary.last_step_no else ""
                self.console.print(f"[dim]Last completed step: {prefix}{last_step_display}[/dim]")
            self.console.print(f"[dim]Status: {summary.status} • Updated: {summary.updated_at}[/dim]")

        default_step = self.session_manager.next_pending_step()
        step_choices = [str(i) for i in range(1, self._total_steps + 1)]
        resume_choice = Prompt.ask(
            "\n[bold yellow]Resume from which step?[/bold yellow]",
            choices=step_choices,
            default=str(default_step),
        )
        self._resume_from_step = int(resume_choice)
        self.session_manager.record_resume(self._resume_from_step)
        return True

    def _determine_step_reuse(
        self,
        step_key: str,
        step_no: int,
        saved_payload: Optional[Dict[str, Any]],
    ) -> bool:
        if not self._resume_mode or saved_payload is None:
            return False

        if step_no < self._resume_from_step:
            if self.console:
                self.console.print(
                    f"[dim]Using saved configuration for Step {step_no}: {self.session_manager.get_step_name(step_key)}[/dim]"
                )
            return True

        if step_no == self._resume_from_step:
            return Confirm.ask(
                f"Reuse saved configuration for Step {step_no} ({self.session_manager.get_step_name(step_key)})?",
                default=True,
            )

        return False

    def _serialize_selected_models(self, models: List[Dict[str, Any]]) -> List[str]:
        return [entry["relative_name"] for entry in models]

    def _rehydrate_selected_models(self, saved_payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        saved_models = saved_payload.get("selected_models") if saved_payload else None
        if not saved_models:
            return None
        available = {entry["relative_name"]: entry for entry in self._collect_trained_models()}
        missing = [name for name in saved_models if name not in available]
        if missing:
            if self.console:
                self.console.print(
                    f"[yellow]Some saved models are missing on disk: {', '.join(missing)}[/yellow]"
                )
            return None
        return [available[name] for name in saved_models]

    def _serialize_pipeline_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for entry in plan:
            info = entry.get("info", {})
            serialized.append(
                {
                    "id": entry.get("id"),
                    "model": info.get("relative_name"),
                    "scope": entry.get("scope"),
                    "prefix": entry.get("prefix"),
                }
            )
        return serialized

    def _rehydrate_pipeline_plan(
        self,
        selected_models: List[Dict[str, Any]],
        saved_payload: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        saved_plan = saved_payload.get("plan")
        if not isinstance(saved_plan, list) or not saved_plan:
            return None

        models_by_name = {entry["relative_name"]: entry for entry in selected_models}
        plan: List[Dict[str, Any]] = []
        missing_models: Set[str] = set()
        for item in saved_plan:
            model_name = item.get("model")
            if model_name not in models_by_name:
                missing_models.add(model_name or "unknown")
                continue
            plan.append(
                {
                    "id": item.get("id") or models_by_name[model_name]["relative_name"],
                    "info": models_by_name[model_name],
                    "scope": item.get("scope") or {"type": "full"},
                    "prefix": item.get("prefix"),
                }
            )

        if missing_models:
            if self.console:
                self.console.print(
                    f"[yellow]Pipeline references missing models: {', '.join(sorted(missing_models))}[/yellow]"
                )
            return None

        # Ensure deterministic ordering
        plan.sort(key=lambda entry: saved_plan.index(next(item for item in saved_plan if item.get("model") == entry["info"]["relative_name"])))
        return plan

    def _load_dataset_from_state(self, data_source: Dict[str, Any]) -> Optional[pd.DataFrame]:
        try:
            if data_source.get("type") == "file":
                file_path = Path(data_source["path"]).expanduser()
                file_format = data_source.get("format", "").lower()
                if file_format == "csv":
                    return pd.read_csv(file_path)
                if file_format == "tsv":
                    return pd.read_csv(file_path, sep="\t")
                if file_format == "excel":
                    return pd.read_excel(file_path)
                if file_format == "json":
                    return pd.read_json(file_path)
                if file_format == "jsonl":
                    return pd.read_json(file_path, lines=True)
                if file_format == "parquet":
                    return pd.read_parquet(file_path)
                if file_format in {"rdata", "rds"}:
                    import pyreadr  # type: ignore

                    result = pyreadr.read_r(file_path)
                    if result.keys():
                        first_key = next(iter(result.keys()))
                        return result[first_key]
                raise ValueError(f"Unsupported file format for resume: {file_format}")

            if data_source.get("type") == "sql":
                connection_string = data_source.get("connection_string")
                table = data_source.get("table")
                query = data_source.get("query")
                limit = data_source.get("limit")

                if not connection_string:
                    raise ValueError("Missing connection string in saved session.")

                engine = create_engine(connection_string)
                try:
                    if query:
                        return pd.read_sql_query(text(query), engine)
                    if not table:
                        raise ValueError("Saved session missing table name.")
                    sql = f"SELECT * FROM {table}"
                    if limit:
                        sql = f"{sql} LIMIT {int(limit)}"
                    return pd.read_sql_query(text(sql), engine)
                finally:
                    engine.dispose()

            raise ValueError("Unsupported data source type.")

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to reload dataset for session %s: %s", self.session_id, exc)
            if self.console:
                self.console.print(f"[yellow]Unable to reload dataset automatically: {exc}[/yellow]")
            return None

    def run(
        self,
        session_action: Optional[str] = None,
        resume_session_id: Optional[str] = None,
    ):
        """Main entry point with session-aware navigation."""
        factory_mode = self._factory_launch_active and self._factory_launch_config is not None

        if factory_mode:
            slug = self._default_session_slug or "bert_annotation"
            if self.console:
                self.console.print(
                    f"\n[cyan]Annotator Factory: launching annotation studio session '{slug}'.[/cyan]\n"
                )
            self.session_id = self.session_manager.start_new_session(slug)
            self._resume_mode = False
            self._resume_from_step = 1
            if self._factory_context:
                sanitized_context = self._sanitize_for_metadata(self._factory_context)
                self.session_manager.metadata.setdefault("factory_context", sanitized_context)
                # Persist immediately so the session folder mirrors the factory structure
                self.session_manager._touch_metadata()
        else:
            self._display_welcome()
            if not self._initialize_session(session_action, resume_session_id):
                return

        selected_models: Optional[List[Dict[str, Any]]] = None
        pipeline_plan: Optional[List[Dict[str, Any]]] = None
        data_source: Optional[Dict[str, Any]] = None
        df: Optional[pd.DataFrame] = None
        column_mapping: Optional[Dict[str, Any]] = None
        language_info: Optional[Dict[str, Any]] = None
        annotation_config: Optional[Dict[str, Any]] = None
        export_config: Optional[Dict[str, Any]] = None
        execution_summary: Optional[Dict[str, Any]] = None

        try:
            # ------------------------- STEP 1 -------------------------
            self._render_step_header(1, "Select Trained Models", "🎯 Pick the fine-tuned checkpoints you want to apply.")
            step_key = "select_models"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 1, saved_step)
            if reuse and saved_step:
                selected_models = self._rehydrate_selected_models(saved_step)
            if not selected_models:
                self.session_manager.mark_step_started(step_key)
                selected_models = self._select_trained_models()
                if not selected_models:
                    self.session_manager.mark_step_failed(step_key, "no_models_selected")
                    self.session_manager.set_status("cancelled")
                    return
                self.session_manager.save_step(
                    step_key,
                    {"selected_models": self._serialize_selected_models(selected_models)},
                    summary=f"{len(selected_models)} model(s)",
                )

            # ------------------------- STEP 2 -------------------------
            self._render_step_header(2, "Configure Pipeline", "⚙️ Order models, set priorities, optionally enable reduction.")
            step_key = "configure_pipeline"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 2, saved_step)
            if reuse and saved_step:
                pipeline_plan = self._rehydrate_pipeline_plan(selected_models, saved_step)
            if not pipeline_plan:
                self.session_manager.mark_step_started(step_key)
                pipeline_plan = self._configure_pipeline(selected_models)
                if not pipeline_plan:
                    self.session_manager.mark_step_failed(step_key, "pipeline_cancelled")
                    self.session_manager.set_status("cancelled")
                    return
                self.session_manager.save_step(
                    step_key,
                    {"plan": self._serialize_pipeline_plan(pipeline_plan)},
                    summary=f"{len(pipeline_plan)} stage(s)",
                )

            # ------------------------- STEP 3 -------------------------
            self._render_step_header(3, "Choose Dataset", "📁 Load the texts you want to annotate.")
            step_key = "select_dataset"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 3, saved_step)
            if reuse and saved_step:
                data_source = saved_step.get("data_source")
                if data_source and data_source.get("type") == "file":
                    if not Path(data_source.get("path", "")).expanduser().exists():
                        if self.console:
                            self.console.print("[yellow]Saved dataset path not found. Please re-select.[/yellow]")
                        data_source = None
            if not data_source:
                factory_source: Optional[Dict[str, Any]] = None
                if factory_mode:
                    factory_source = self._build_factory_data_source()
                    if factory_source:
                        self.session_manager.mark_step_started(step_key)
                        data_source = factory_source
                        self.session_manager.save_step(
                            step_key,
                            {"data_source": data_source},
                            summary=str(Path(data_source["path"]).name),
                        )
                        if self.console:
                            self.console.print(
                                f"[cyan]Annotator Factory: using dataset {data_source['path']}[/cyan]"
                            )
                if not data_source:
                    self.session_manager.mark_step_started(step_key)
                    data_source = self._select_data_source()
                    if data_source is None:
                        self.session_manager.mark_step_failed(step_key, "dataset_cancelled")
                        self.session_manager.set_status("cancelled")
                        return
                    self.session_manager.save_step(
                        step_key,
                        {"data_source": data_source},
                        summary=data_source.get("path") or data_source.get("display_name", "dataset"),
                    )

            # ------------------------- STEP 4 -------------------------
            self._render_step_header(4, "Inspect & Map Columns", "🔍 Tell the studio where the text and identifiers live.")
            step_key = "map_columns"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 4, saved_step)
            if reuse and saved_step:
                column_mapping = saved_step.get("column_mapping")
                df = self._load_dataset_from_state(data_source) if data_source else None
                if df is not None:
                    self._detect_factory_context(data_source, df)
                if df is None or column_mapping is None:
                    if self.console:
                        self.console.print("[yellow]Failed to reuse saved column mapping. Restarting step interactively.[/yellow]")
                    reuse = False
            if not reuse:
                self.session_manager.mark_step_started(step_key)
                df, column_mapping = self._load_and_analyze_data(data_source, pipeline_plan)
                if df is None or column_mapping is None:
                    self.session_manager.mark_step_failed(step_key, "mapping_cancelled")
                    self.session_manager.set_status("cancelled")
                    return
                if factory_mode and df is not None and column_mapping is not None:
                    column_mapping = self._apply_factory_column_defaults(df, column_mapping)
                row_count = int(df.shape[0]) if hasattr(df, "shape") else 0
                self.session_manager.save_step(
                    step_key,
                    {
                        "column_mapping": column_mapping,
                        "row_count": row_count,
                        "columns": list(df.columns),
                    },
                    summary=f"{row_count:,} rows detected",
                )
            elif df is None or column_mapping is None:
                self.session_manager.mark_step_failed(step_key, "mapping_unavailable")
                self.session_manager.set_status("failed")
                return

            # ------------------------- STEP 5 -------------------------
            self._render_step_header(5, "Name Output Columns", "📝 Define how prediction columns will be named.")
            step_key = "output_columns"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 5, saved_step)
            if reuse and saved_step:
                rehydrated_plan = self._rehydrate_pipeline_plan(selected_models, saved_step)
                if rehydrated_plan:
                    pipeline_plan = rehydrated_plan
                else:
                    reuse = False
            if not reuse:
                self.session_manager.mark_step_started(step_key)
                pipeline_plan = self._configure_output_columns(pipeline_plan, df, column_mapping)
                self.session_manager.save_step(
                    step_key,
                    {"plan": self._serialize_pipeline_plan(pipeline_plan)},
                    summary="Output columns confirmed",
                )

            # ------------------------- STEP 6 -------------------------
            self._render_step_header(6, "Language Detection", "🌍 Verify language compatibility for your dataset.")
            step_key = "language_detection"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 6, saved_step)
            if reuse and saved_step:
                language_info = saved_step
            if not reuse:
                self.session_manager.mark_step_started(step_key)
                models_for_language = [entry["info"] for entry in pipeline_plan]
                language_info = self._detect_and_validate_language(df, column_mapping, models_for_language)
                if language_info is None:
                    self.session_manager.mark_step_failed(step_key, "language_validation_cancelled")
                    self.session_manager.set_status("cancelled")
                    return
                self.session_manager.save_step(
                    step_key,
                    language_info,
                    summary=f"Primary language: {language_info.get('primary_language')}",
                )

            # ------------------------- STEP 7 -------------------------
            self._render_step_header(7, "Annotation Options", "⚡ Parallelism, batching strategy, and dataset coverage.")
            step_key = "annotation_options"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 7, saved_step)
            if reuse and saved_step:
                annotation_config = saved_step
            else:
                self.session_manager.mark_step_started(step_key)
                annotation_config = self._configure_annotation_options(pipeline_plan, df, column_mapping)
                self.session_manager.save_step(
                    step_key,
                    annotation_config,
                    summary=f"Parallel: {'yes' if annotation_config.get('parallel') else 'no'}",
                )

            # ------------------------- STEP 8 -------------------------
            self._render_step_header(8, "Export Options", "💾 Choose what gets written to disk.")
            step_key = "export_options"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 8, saved_step)
            if reuse and saved_step:
                export_config = saved_step
            else:
                self.session_manager.mark_step_started(step_key)
                export_config = self._configure_export_options()
                self.session_manager.save_step(step_key, export_config, summary="Export destinations configured")

            # ------------------------- STEP 9 -------------------------
            self._render_step_header(9, "Review & Launch", "🚀 Final checks before the annotation run.")
            step_key = "review_launch"
            saved_step = self.session_manager.get_step_data(step_key)
            reuse = self._determine_step_reuse(step_key, 9, saved_step)
            if reuse and saved_step and saved_step.get("executed"):
                execution_summary = saved_step
                if self.console:
                    self.console.print("[dim]Previous execution already completed for this session. Skipping launch.[/dim]")
            else:
                self.session_manager.mark_step_started(step_key)
                df_for_execution = df
                language_mask = getattr(self, "_language_annotation_mask", None)
                allowed_languages = getattr(self, "_allowed_annotation_languages", [])
                if df is not None and language_mask is not None and len(language_mask) == len(df):
                    filtered_df = df[language_mask].copy()
                    retained_rows = len(filtered_df)
                    skipped_rows = len(df) - retained_rows
                    if retained_rows == 0:
                        self.console.print("[red]✗ No rows eligible for annotation after applying language filters.[/red]")
                        self.session_manager.mark_step_failed(step_key, "language_filter_empty")
                        self.session_manager.set_status("cancelled")
                        return
                    if skipped_rows > 0:
                        self.console.print(
                            f"[dim]Language filter applied before execution: {retained_rows:,} rows retained "
                            f"({skipped_rows:,} skipped).[/dim]"
                        )
                    df_for_execution = filtered_df
                    scope_cfg = annotation_config.get('scope', {})
                    if isinstance(scope_cfg, dict):
                        if scope_cfg.get('type') in {'head', 'random'}:
                            scope_cfg['size'] = min(int(scope_cfg.get('size', retained_rows)), retained_rows)
                        annotation_config['scope'] = scope_cfg
                if allowed_languages:
                    annotation_config['languages_to_annotate'] = allowed_languages
                executed = self._confirm_and_execute(
                    pipeline_plan,
                    data_source,
                    df_for_execution,
                    column_mapping,
                    language_info,
                    annotation_config,
                    export_config,
                )
                if executed:
                    execution_summary = {
                        "executed": True,
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "row_count": int(df_for_execution.shape[0]) if hasattr(df_for_execution, "shape") else None,
                    }
                    self.session_manager.save_step(
                        step_key,
                        execution_summary,
                        summary="Annotation completed",
                    )
                    if self.console:
                        self.console.print("\n[bold green]✓ Annotation completed successfully![/bold green]")
                else:
                    self.session_manager.mark_step_failed(step_key, "execution_cancelled")
                    self.session_manager.set_status("cancelled")
                    return

            if execution_summary and execution_summary.get("executed"):
                self.session_manager.set_status("completed")

        except KeyboardInterrupt:
            if self.console:
                self.console.print("\n[yellow]Annotation cancelled[/yellow]")
            self.session_manager.set_status("cancelled")
        except Exception as exc:
            if self.console:
                self.console.print(f"\n[bold red]✗ Error:[/bold red] {exc}", markup=False, highlight=False)
            self.logger.exception("BERT Annotation Studio error")
            self.session_manager.set_status("failed")
        finally:
            input("\nPress Enter to continue...")

    def _display_welcome(self):
        """Display welcome - Banner now handled by advanced_cli.py"""
        # Banner and mode info are now displayed by advanced_cli.py before calling run()
        # This keeps the interface consistent across all modes
        pass

    def _render_step_header(self, step_no: int, title: str, description: Optional[str] = None) -> None:
        """Render a step header in Training Arena style with separator lines."""
        separator = "[bold cyan]" + "━" * 86 + "[/bold cyan]"

        self.console.print(f"\n{separator}")
        self.console.print(f"[bold cyan]  STEP {step_no}:[/bold cyan] [bold white]{title}[/bold white]")
        self.console.print(separator)

        if description:
            self.console.print(f"[dim]{description}[/dim]")

    def _print_table(self, table: Table) -> None:
        """Center tables consistently across the interface."""
        self.console.print(Align.center(table))

    def _configure_pipeline(self, models: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Ask for execution order and optional reduction rules."""
        if not models:
            return None

        ordered_models = list(models)
        show_table = len(ordered_models) > 1 and not self._factory_launch_active
        if show_table:
            summary = Table(title="Selected Models", box=box.ROUNDED, show_lines=False)
            summary.add_column("#", style="cyan", justify="center", width=4)
            summary.add_column("Model", style="green", overflow="fold")
            summary.add_column("Task", style="bright_white", overflow="ellipsis")
            summary.add_column("Lang", style="magenta", justify="center", width=8)
            summary.add_column("Labels", style="cyan", justify="right", width=6)
            summary.add_column("Categories", style="white", overflow="fold")
            summary.add_column("Macro F1", style="bright_white", justify="right", width=10)

            for idx, model in enumerate(ordered_models, 1):
                macro = model['metrics'].get('macro_f1')
                macro_text = f"{macro:.3f}" if isinstance(macro, (int, float)) else "—"
                per_language_metrics = model.get("metrics_per_language") or model['metrics'].get('per_language', {})
                if per_language_metrics:
                    per_parts = []
                    for lang_key in sorted(per_language_metrics):
                        value = per_language_metrics[lang_key]
                        if isinstance(value, (int, float)):
                            per_parts.append(f"{lang_key}:{value:.3f}")
                    if per_parts:
                        macro_text = f"{macro_text}\n[dim]{' | '.join(per_parts)}[/dim]"
                id2label_pairs = model.get("id2label_pairs") or []
                label_total = model.get("label_count") or len(id2label_pairs)
                model_display = self._condense_relative_name(model['relative_name'])
                base_display = self._shorten_base_model(model['base_model'])
                if base_display and base_display.lower() not in model_display.lower():
                    model_display = f"{model_display}\n[dim]{base_display}[/dim]"
                task_display = str(model.get("label_value") or "—")
                confirmed_langs = model.get("confirmed_languages") or [model.get("language")]
                lang_display = ", ".join(confirmed_langs)
                summary.add_row(
                    str(idx),
                    model_display,
                    task_display,
                    lang_display,
                    str(label_total),
                    self._format_id2label_pairs(id2label_pairs),
                    macro_text,
                )

            self._print_table(summary)

        if len(ordered_models) > 1:
            if self.console:
                self.console.print("\n[bold cyan]Ordering Strategy:[/bold cyan]")
                self.console.print("[dim][1] Keep current priority (same order as selection)[/dim]")
                self.console.print("[dim][2] Sort alphabetically (A → Z by model name)[/dim]\n")

            order_choice = Prompt.ask(
                "[cyan]Ordering mode[/cyan]",
                choices=["1", "2"],
                default="1",
            )

            if order_choice == "2":
                ordered_models = sorted(ordered_models, key=lambda m: m['relative_name'].lower())

        plan: List[Dict[str, Any]] = [
            {
                "id": model["relative_name"],
                "info": model,
                "scope": {"type": "full"},
                "prefix": None,
            }
            for model in ordered_models
        ]

        if len(plan) <= 1:
            return plan

        self.console.print("\n[bold magenta]Reduction Mode (optional)[/bold magenta]")
        self.console.print("[dim]A reducer scans the full dataset and keeps only the rows that match specific labels.[/dim]")
        self.console.print("[dim]Child models then run on that focused slice, saving time on expensive checkpoints.[/dim]")
        schema_lines = [
            "[bold cyan]Full dataset[/bold cyan]",
            "        │",
            "        ▼",
            "[bold magenta]Reducer[/bold magenta] ── (positive labels) ──▶ [bold green]Child models[/bold green]",
            "        │                               ├─▶ Child 1",
            "        └─ (other labels) ──▶ [dim]Skipped[/dim] └─▶ Child 2",
        ]
        schema_panel = Panel("\n".join(schema_lines), border_style="magenta", title="Pipeline diagram")
        self.console.print(Align.center(schema_panel))
        self.console.print("[dim]Tip: repeat an index to reuse the same child model multiple times with different label filters.[/dim]\n")

        if not Confirm.ask("Enable reduction mode?", default=False):
            return plan

        while True:
            reducers = [entry for entry in plan]
            reducer_table = Table(box=box.ROUNDED, title="Available Reducers")
            reducer_table.add_column("#", style="cyan", justify="center", width=4)
            reducer_table.add_column("Model", style="green", overflow="fold")
            reducer_table.add_column("Task", style="bright_white", overflow="ellipsis")
            reducer_table.add_column("Lang", style="magenta", justify="center", width=8)
            reducer_table.add_column("Labels", style="cyan", justify="right", width=6)
            for idx, entry in enumerate(reducers, 1):
                model_info = entry["info"]
                id2label_pairs = model_info.get("id2label_pairs") or []
                label_total = model_info.get("label_count") or len(id2label_pairs)
                reducer_table.add_row(
                    str(idx),
                    self._condense_relative_name(model_info["relative_name"]),
                    str(model_info.get("label_value") or "—"),
                    model_info["language"],
                    str(label_total),
                )
            self._print_table(reducer_table)

            choices = [str(i) for i in range(1, len(reducers) + 1)]
            choices.append("0")
            reducer_choice = Prompt.ask(
                "[cyan]Select reducer (0 to finish)[/cyan]",
                choices=choices,
                default="0",
            )
            if reducer_choice == "0":
                break

            reducer_entry = reducers[int(reducer_choice) - 1]
            labels = self._extract_label_names(reducer_entry["info"])
            if not labels:
                self.console.print("[yellow]Reducer has no label names; skipping.[/yellow]")
                continue

            label_table = Table(box=box.ROUNDED, title="Reducer Labels")
            label_table.add_column("#", style="cyan", justify="center", width=4)
            label_table.add_column("Label", style="green")
            for idx, label in enumerate(labels, 1):
                label_table.add_row(str(idx), label)
            self._print_table(label_table)

            default_label_idx = next((i for i, name in enumerate(labels, 1) if "pos" in name.lower()), 1)
            raw = Prompt.ask(
                "[cyan]Positive label indices (comma-separated)[/cyan]",
                default=str(default_label_idx),
            )
            try:
                label_indices = self._parse_index_list(raw, len(labels))
            except ValueError:
                self.console.print("[red]Invalid label selection. Try again.[/red]")
                continue

            positive_labels = [labels[i - 1] for i in label_indices]

            available_children = [
                entry for entry in plan
                if entry["id"] != reducer_entry["id"]
            ]

            if not available_children:
                self.console.print("[yellow]No models are currently available for cascading.[/yellow]")
                break

            children_table = Table(box=box.ROUNDED, title="Attach Child Models to Positive Slice")
            children_table.add_column("#", style="cyan", justify="center", width=4)
            children_table.add_column("Model", style="green", overflow="fold")
            children_table.add_column("Task", style="bright_white", overflow="ellipsis")
            children_table.add_column("Lang", style="magenta", justify="center", width=8)
            children_table.add_column("Current scope", style="yellow", overflow="fold")
            for idx, entry in enumerate(available_children, 1):
                info = entry["info"]
                children_table.add_row(
                    str(idx),
                    self._condense_relative_name(info["relative_name"]),
                    str(info.get("label_value") or "—"),
                    info["language"],
                    self._describe_child_scope(entry, plan),
                )
            self._print_table(children_table)
            self.console.print("[dim]Tip: repeat an index to duplicate a child model on this filter.[/dim]")

            child_choice_raw = Prompt.ask(
                "[cyan]Model indices to run on positives (comma-separated)[/cyan]",
                default="1",
            )
            try:
                child_indices = self._parse_index_list(child_choice_raw, len(available_children), allow_duplicates=True)
            except ValueError:
                self.console.print("[red]Invalid model selection. Try again.[/red]")
                continue

            for idx in child_indices:
                child_entry = available_children[idx - 1]
                new_scope = {
                    "type": "positive",
                    "parent_id": reducer_entry["id"],
                    "labels": list(positive_labels),
                }
                if child_entry["scope"].get("type") == "full":
                    child_entry["scope"] = new_scope
                else:
                    self._clone_plan_entry(child_entry, plan, new_scope)

            if Confirm.ask("Configure another reducer?", default=False):
                continue
            break

        self._reorder_plan_with_children(plan)
        return plan

    def _configure_output_columns(
        self,
        plan: List[Dict[str, Any]],
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Let the user confirm or override the base name used for generated columns."""
        if not plan:
            return plan

        text_column = column_mapping.get("text")
        id_column = column_mapping.get("id")
        sample_id = None
        sample_text = ""
        if not df.empty and text_column in df.columns:
            first_row = df.iloc[0]
            sample_text = str(first_row.get(text_column, "")) if isinstance(first_row, pd.Series) else ""
            if id_column and id_column in df.columns:
                sample_id = str(first_row.get(id_column, ""))
            else:
                sample_id = str(df.index[0])
        sample_preview = sample_text.replace("\n", " ").strip()
        if len(sample_preview) > 140:
            sample_preview = sample_preview[:137] + "…"

        used_prefixes: set[str] = set()

        for entry in plan:
            model_info = entry["info"]
            default_base = self._suggest_output_base(model_info)
            language_suffix = model_info.get("language", "").lower()
            if language_suffix and not default_base.endswith(f"_{language_suffix}"):
                default_base = f"{default_base}_{language_suffix}"

            base_candidate = default_base
            counter = 2
            while base_candidate in used_prefixes:
                base_candidate = f"{default_base}_{counter}"
                counter += 1
            if base_candidate != default_base:
                default_base = base_candidate

            model_name_display = self._condense_relative_name(model_info["relative_name"])
            example_columns = [
                f"{default_base}_label",
                f"{default_base}_label_id",
                f"{default_base}_probability (optional)",
            ]
            body_lines = [
                f"Model: {model_name_display}",
                f"Default base: [bold]{default_base}[/bold]",
            ]
            if sample_id is not None:
                body_lines.append(f"Example row id: {sample_id}")
            if sample_preview:
                body_lines.append(f"Text preview: {sample_preview}")
            body_lines.append("")
            body_lines.append("Columns will look like:")
            body_lines.extend(f"  • {col}" for col in example_columns)

            self.console.print("\n[bold cyan]Output Column Naming:[/bold cyan]")
            for line in body_lines:
                self.console.print(f"[dim]{line}[/dim]")
            self.console.print()

            while True:
                raw_name = Prompt.ask(
                    "[cyan]Column base name[/cyan]",
                    default=default_base,
                ).strip()
                if not raw_name:
                    raw_name = default_base
                sanitized = self._sanitize_model_prefix(raw_name)
                if not sanitized:
                    self.console.print("[red]Please provide at least one alphanumeric character.[/red]")
                    continue
                final_name = sanitized
                suffix_idx = 2
                while final_name in used_prefixes:
                    final_name = f"{sanitized}_{suffix_idx}"
                    suffix_idx += 1
                if final_name != sanitized:
                    self.console.print(
                        f"[yellow]Name already used. Using '{final_name}' to keep column names unique.[/yellow]"
                    )
                entry["prefix"] = final_name
                entry["columns"] = {
                    "label": f"{final_name}_label",
                    "label_id": f"{final_name}_label_id",
                    "probability": f"{final_name}_probability",
                    "ci_lower": f"{final_name}_ci_lower",
                    "ci_upper": f"{final_name}_ci_upper",
                    "language": f"{final_name}_language",
                    "annotated": f"{final_name}_annotated",
                }
                used_prefixes.add(final_name)
                break

        id_to_entry = {entry["id"]: entry for entry in plan}
        for entry in plan:
            scope = entry.get("scope", {})
            if scope.get("type") == "positive":
                parent = id_to_entry.get(scope["parent_id"])
                if parent and parent.get("prefix"):
                    scope["parent_prefix"] = parent["prefix"]
        return plan

    def _suggest_output_base(self, model_info: Dict[str, Any]) -> str:
        """Guess a human friendly base name for output columns."""
        config = model_info.get("config", {})
        candidates = [
            config.get("task_name"),
            config.get("project_name"),
            config.get("run_name"),
            config.get("dataset_name"),
            config.get("workflow_name"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return self._sanitize_model_prefix(candidate)

        relative = model_info.get("relative_name", "")
        parts = [part for part in relative.split("/") if part]
        if len(parts) >= 2:
            return self._sanitize_model_prefix(parts[-2])
        if parts:
            return self._sanitize_model_prefix(parts[-1])

        base_model = model_info.get("base_model", "model")
        return self._sanitize_model_prefix(base_model)

    @staticmethod
    def _parse_index_list(raw: str, max_index: int, allow_duplicates: bool = False) -> List[int]:
        """Parse a comma-separated list of indices."""
        parts = [chunk.strip() for chunk in raw.replace(";", ",").split(",") if chunk.strip()]
        if not parts:
            raise ValueError("empty selection")
        indices: List[int] = []
        for part in parts:
            if not part.isdigit():
                raise ValueError(f"{part} is not a number")
            value = int(part)
            if value < 1 or value > max_index:
                raise ValueError(f"{value} is out of range")
            indices.append(value)

        if allow_duplicates:
            return indices

        deduped: List[int] = []
        for value in indices:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _extract_id2label_pairs(config: Dict[str, Any]) -> List[Tuple[int, str]]:
        """Extract ordered (id, label) pairs from a Hugging Face config."""
        mapping = config.get("id2label")
        if isinstance(mapping, dict) and mapping:
            numeric_pairs: List[Tuple[int, str]] = []
            fallback_values: List[str] = []
            for key, value in mapping.items():
                label_name = str(value)
                idx: Optional[int] = None
                if isinstance(key, int):
                    idx = key
                elif isinstance(key, str):
                    if key.isdigit():
                        idx = int(key)
                    else:
                        match = re.search(r"(\\d+)$", key)
                        if match:
                            idx = int(match.group(1))
                if idx is not None:
                    numeric_pairs.append((idx, label_name))
                else:
                    fallback_values.append(label_name)
            if numeric_pairs:
                numeric_pairs.sort(key=lambda item: item[0])
                return numeric_pairs
            if fallback_values:
                return [(idx, name) for idx, name in enumerate(fallback_values)]

        if isinstance(mapping, list):
            return [(idx, str(name)) for idx, name in enumerate(mapping)]

        label2id = config.get("label2id")
        if isinstance(label2id, dict) and label2id:
            numeric_pairs = []
            fallback_values = []
            for label_name, idx in label2id.items():
                try:
                    numeric_pairs.append((int(idx), str(label_name)))
                except (TypeError, ValueError):
                    fallback_values.append(str(label_name))
            if numeric_pairs:
                numeric_pairs.sort(key=lambda item: item[0])
                return numeric_pairs
            if fallback_values:
                return [(idx, name) for idx, name in enumerate(fallback_values)]

        return []

    @staticmethod
    def _format_id2label_pairs(pairs: List[Tuple[int, str]]) -> str:
        """Build a multi-line textual representation of an id2label mapping."""
        if not pairs:
            return "—"
        return "\n".join(f"{idx}: {label}" for idx, label in pairs)

    def _describe_child_scope(self, entry: Dict[str, Any], plan: List[Dict[str, Any]]) -> str:
        """Provide a human-readable summary of a pipeline entry's current scope."""
        scope = entry.get("scope", {})
        scope_type = scope.get("type")
        if scope_type == "positive":
            parent_id = scope.get("parent_id")
            parent = next((item for item in plan if item["id"] == parent_id), None)
            parent_name = (
                self._condense_relative_name(parent["info"]["relative_name"])
                if parent and parent.get("info")
                else str(parent_id)
            )
            labels = scope.get("labels") or []
            if labels:
                preview = ", ".join(str(label) for label in labels[:3])
                if len(labels) > 3:
                    preview += ", …"
            else:
                preview = "all"
            return f"Filtered by {parent_name} ({preview})"
        return "Full dataset"

    @staticmethod
    def _extract_label_names(model_info: Dict[str, Any]) -> List[str]:
        """Return label names for a model from stored info or config."""
        stored_pairs = model_info.get("id2label_pairs")
        if isinstance(stored_pairs, list) and stored_pairs:
            return [str(label) for _, label in stored_pairs]

        metadata_names = model_info.get("metadata_label_names")
        if isinstance(metadata_names, list) and metadata_names:
            return [str(name) for name in metadata_names]

        config = model_info.get("config", {})
        label_map = config.get("id2label")
        if isinstance(label_map, dict):
            try:
                ordered = sorted(label_map.items(), key=lambda item: int(item[0]))
                return [str(name) for _, name in ordered]
            except ValueError:
                return [str(name) for name in label_map.values()]
        if isinstance(label_map, list):
            return [str(name) for name in label_map]
        label_count = model_info.get("label_count")
        if isinstance(label_count, int) and label_count > 0:
            return [f"Label {i}" for i in range(label_count)]
        return []

    @staticmethod
    def _reorder_plan_with_children(plan: List[Dict[str, Any]]) -> None:
        """Ensure children scoped to reducers run immediately after their parent."""
        parent_to_children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in plan:
            scope = entry.get("scope", {})
            if scope.get("type") == "positive":
                parent_to_children[scope["parent_id"]].append(entry)

        new_plan: List[Dict[str, Any]] = []
        added: set[str] = set()
        for entry in plan:
            if entry["id"] in added:
                continue
            new_plan.append(entry)
            added.add(entry["id"])
            for child in parent_to_children.get(entry["id"], []):
                if child["id"] not in added:
                    new_plan.append(child)
                    added.add(child["id"])

        plan.clear()
        plan.extend(new_plan)

    def _clone_plan_entry(
        self,
        entry: Dict[str, Any],
        plan: List[Dict[str, Any]],
        new_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a duplicate plan entry so a model can run in multiple cascades."""
        existing_ids = {item["id"] for item in plan}
        clone_id = self._generate_unique_plan_id(entry["id"], existing_ids)
        clone = {
            "id": clone_id,
            "info": entry["info"],
            "scope": new_scope,
            "prefix": entry.get("prefix"),
        }
        plan.append(clone)
        return clone

    @staticmethod
    def _generate_unique_plan_id(base_id: str, existing_ids: set[str]) -> str:
        """Create a unique identifier for duplicated plan entries."""
        if base_id not in existing_ids:
            return base_id
        counter = 2
        while True:
            candidate = f"{base_id}__{counter}"
            if candidate not in existing_ids:
                return candidate
            counter += 1

    def _select_trained_models(self) -> Optional[List[Dict[str, Any]]]:
        """Select one or more trained models"""
        if not self.models_dir.exists():
            self.console.print(f"[red]✗ Models directory not found: {self.models_dir}[/red]")
            self.console.print("[yellow]Tip: Train a model first using Training Studio (Mode 5)[/yellow]")
            return None

        model_entries = self._collect_trained_models()
        if not model_entries:
            self.console.print("[yellow]No trained models found[/yellow]")
            return None

        self.console.print("\n[cyan]Pick one or several trained checkpoints. You can run them sequentially or cascade them later.[/cyan]")
        self.console.print("[cyan]Tip: combine a high-recall model with specialised models to refine positives.[/cyan]")
        self.console.print("\n[bold]🎯 Selection Mode:[/bold]")
        self.console.print("[dim]Select the model(s) that will be used to annotate your texts.[/dim]")
        self.console.print("[dim]You can chain multiple models: each will add its dedicated columns.[/dim]")

        model_table = Table(title="Available Trained Models", box=box.ROUNDED, show_lines=False)
        model_table.add_column("#", style="cyan", width=4, justify="center")
        model_table.add_column("Model", style="green", overflow="fold")
        model_table.add_column("Task", style="bright_white", overflow="ellipsis")
        model_table.add_column("Lang", style="magenta", width=8, justify="center")
        model_table.add_column("Labels", style="cyan", width=6, justify="right")
        model_table.add_column("Categories", style="white", overflow="fold")
        model_table.add_column("Macro F1", style="bright_white", width=10, justify="right")
        model_table.add_column("Updated", style="dim", width=19)

        for idx, model_info in enumerate(model_entries, 1):
            macro = model_info['metrics'].get('macro_f1')
            macro_text = f"{macro:.3f}" if isinstance(macro, (int, float)) else "—"
            id2label_pairs = model_info.get("id2label_pairs") or []
            label_total = model_info.get("label_count") or len(id2label_pairs)
            model_display = self._condense_relative_name(model_info['relative_name'])
            base_display = self._shorten_base_model(model_info['base_model'])
            if base_display and base_display.lower() not in model_display.lower():
                model_display = f"{model_display}\n[dim]{base_display}[/dim]"
            task_display = model_info.get("label_value") or "—"
            task_display = str(task_display)
            id2label_text = self._format_id2label_pairs(id2label_pairs)
            model_table.add_row(
                str(idx),
                model_display,
                task_display,
                model_info['language'],
                str(label_total),
                id2label_text,
                macro_text,
                model_info['updated_at'].strftime("%Y-%m-%d %H:%M"),
            )

        self._print_table(model_table)

        self.console.print("\n[bold magenta]Selection Options:[/bold magenta]")
        self.console.print("[dim][1] Single model (pick one)[/dim]")
        self.console.print("[dim][2] Multiple models (enter list: e.g., 1,3,5)[/dim]")
        self.console.print("[dim][3] All available models[/dim]\n")

        selection_mode = Prompt.ask(
            "[cyan]Choose a mode[/cyan]",
            choices=["1", "2", "3"],
            default="1"
        )

        def parse_indices(raw: str) -> List[int]:
            parts = [chunk.strip() for chunk in raw.replace(";", ",").split(",") if chunk.strip()]
            indices: List[int] = []
            for part in parts:
                if not part.isdigit():
                    raise ValueError
                value = int(part)
                if value < 1 or value > len(model_entries):
                    raise ValueError
                if value not in indices:
                    indices.append(value)
            if not indices:
                raise ValueError
            return indices

        if selection_mode == "1":
            choice = Prompt.ask(
                "\n[cyan]Select model[/cyan]",
                choices=[str(i) for i in range(1, len(model_entries) + 1)],
                default="1"
            )
            selection = [int(choice)]
        elif selection_mode == "2":
            while True:
                raw = Prompt.ask(
                    "\n[cyan]Model indices (e.g.: 1,3,4)[/cyan]",
                    default="1"
                )
                try:
                    selection = parse_indices(raw)
                    break
                except ValueError:
                    self.console.print("[red]Please enter valid indices separated by commas.[/red]")
        else:
            selection = list(range(1, len(model_entries) + 1))

        if len(selection) > 1:
            if self.console:
                self.console.print(
                    "\n[cyan]Execution order decides which model runs first during inference.[/cyan]"
                )
                self.console.print(
                    "[cyan]• Keep selection order: run models exactly in the sequence you just picked.[/cyan]"
                )
                self.console.print(
                    "[cyan]• Alphabetical: run models from A → Z by their display name.[/cyan]"
                )
                self.console.print(
                    "[cyan]• Custom order: type the indices again to define a new priority right now.[/cyan]"
                )
            order_choice = Prompt.ask(
                "\n[cyan]Choose execution order[/cyan] ([1] keep selection order • [2] alphabetical • [3] custom order)",
                choices=["1", "2", "3"],
                default="1",
            )
            if order_choice == "2":
                selection.sort(key=lambda idx: model_entries[idx - 1]['relative_name'].lower())
            elif order_choice == "3":
                while True:
                    raw_order = Prompt.ask(
                        "\n[cyan]Enter the exact execution order (e.g., 2,1,3)[/cyan]",
                        default=",".join(str(idx) for idx in selection),
                    )
                    try:
                        reordered = parse_indices(raw_order)
                        if len(reordered) != len(selection):
                            raise ValueError
                        selection = reordered
                        break
                    except ValueError:
                        self.console.print(
                            "[red]Please enter each selected index once to set the execution order.[/red]"
                        )

        chosen_models = [model_entries[idx - 1] for idx in selection]

        summary = Table(title="Selected Models", box=box.ROUNDED, show_lines=False)
        summary.add_column("#", style="cyan", width=4, justify="center")
        summary.add_column("Model", style="green", overflow="fold")
        summary.add_column("Task", style="bright_white", overflow="ellipsis")
        summary.add_column("Lang", style="magenta", width=8, justify="center")
        summary.add_column("Labels", style="cyan", width=6, justify="right")
        summary.add_column("Categories", style="white", overflow="fold")
        summary.add_column("Macro F1", style="bright_white", width=10, justify="right")

        for idx, model in enumerate(chosen_models, 1):
            macro = model['metrics'].get('macro_f1')
            macro_text = f"{macro:.3f}" if isinstance(macro, (int, float)) else "—"
            id2label_pairs = model.get("id2label_pairs") or []
            label_total = model.get("label_count") or len(id2label_pairs)
            model_display = self._condense_relative_name(model['relative_name'])
            base_display = self._shorten_base_model(model['base_model'])
            if base_display and base_display.lower() not in model_display.lower():
                model_display = f"{model_display}\n[dim]{base_display}[/dim]"
            task_display = str(model.get("label_value") or "—")
            summary.add_row(
                str(idx),
                model_display,
                task_display,
                model['language'],
                str(label_total),
                self._format_id2label_pairs(id2label_pairs),
                macro_text,
            )
        self._print_table(summary)
        if len(chosen_models) > 1:
            self.console.print("[dim]FYI: inference follows this execution order. You can still adjust it in the pipeline configuration step next.[/dim]")
        self.console.print(f"\n[green]✓ {len(chosen_models)} model(s) ready for annotation[/green]")
        return chosen_models

    def _collect_trained_models(self) -> List[Dict[str, Any]]:
        """Discover trained model folders with metadata."""
        entries: List[Dict[str, Any]] = []

        try:
            config_paths = sorted(self.models_dir.rglob("config.json"))
        except Exception as exc:
            self.logger.error("Failed to scan models directory %s: %s", self.models_dir, exc)
            return entries

        for config_path in config_paths:
            model_dir = config_path.parent
            if not model_dir.is_dir():
                continue

            if self._allowed_model_paths:
                try:
                    resolved_dir = model_dir.resolve()
                except Exception:
                    resolved_dir = model_dir
                if resolved_dir not in self._allowed_model_paths:
                    continue

            has_weights = any(
                model_dir.glob(pattern)
                for pattern in ("pytorch_model.bin", "*.safetensors", "tf_model.h5")
            )
            if not has_weights:
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as cfg_file:
                    config = json.load(cfg_file)
            except Exception as exc:
                self.logger.debug("Skipping model at %s (invalid config): %s", model_dir, exc)
                continue

            metrics = self._load_language_metrics(model_dir)
            training_metadata = self._load_training_metadata(model_dir)
            metadata_label_names_raw = training_metadata.get("label_names", [])
            metadata_label_names = (
                [str(name) for name in metadata_label_names_raw]
                if isinstance(metadata_label_names_raw, list)
                else []
            )
            relative_name = str(model_dir.relative_to(self.models_dir)).replace(os.sep, "/")
            base_model = model_dir.name
            language = self._infer_language(model_dir, base_model, config)
            confirmed_langs_raw = config.get("confirmed_languages") or []
            confirmed_languages = sorted(
                {
                    (LanguageNormalizer.normalize_language(lang) or str(lang).strip() or "UNKNOWN").upper()
                    for lang in confirmed_langs_raw
                    if lang is not None
                }
            )
            if not confirmed_languages:
                fallback_lang = LanguageNormalizer.normalize_language(language) or language or "UNKNOWN"
                confirmed_languages = [str(fallback_lang).upper()]

            id2label_pairs = self._extract_id2label_pairs(config)
            if not id2label_pairs and metadata_label_names:
                id2label_pairs = [(idx, label_name) for idx, label_name in enumerate(metadata_label_names)]

            label_count = len(id2label_pairs)
            label_map = config.get("id2label")
            if not label_count and isinstance(label_map, list):
                id2label_pairs = [(idx, str(name)) for idx, name in enumerate(label_map)]
                label_count = len(id2label_pairs)
            if not label_count and isinstance(label_map, dict):
                label_count = len(label_map)

            if not label_count:
                label2id = config.get("label2id")
                if isinstance(label2id, dict):
                    numeric_pairs: List[Tuple[int, str]] = []
                    fallback_labels: List[str] = []
                    for label_name, label_idx in label2id.items():
                        try:
                            numeric_pairs.append((int(label_idx), str(label_name)))
                        except (TypeError, ValueError):
                            fallback_labels.append(str(label_name))
                    if numeric_pairs:
                        numeric_pairs.sort(key=lambda item: item[0])
                        id2label_pairs = numeric_pairs
                        label_count = len(id2label_pairs)
                    elif fallback_labels:
                        id2label_pairs = [(idx, name) for idx, name in enumerate(fallback_labels)]
                        label_count = len(id2label_pairs)

            if not label_count:
                num_labels = config.get("num_labels")
                if isinstance(num_labels, int) and num_labels > 0:
                    label_count = num_labels
            if label_count and not id2label_pairs:
                id2label_pairs = [(idx, f"Label {idx}") for idx in range(label_count)]

            try:
                newest_mtime = max(
                    (p.stat().st_mtime for p in model_dir.rglob("*") if p.is_file()),
                    default=config_path.stat().st_mtime,
                )
                updated_at = datetime.fromtimestamp(newest_mtime)
            except Exception:
                updated_at = datetime.now()

            label_value = training_metadata.get("label_value") or training_metadata.get("label_key")
            if not label_value:
                label_value = config.get("label_value") or config.get("finetuning_task")
            if not label_value:
                # fall back to folder name (skip trailing "model" folder if present)
                parent = model_dir.parent
                label_value = parent.name if model_dir.name.lower() == "model" else model_dir.name

            entries.append(
                {
                    "path": model_dir,
                    "config": config,
                    "relative_name": relative_name,
                    "base_model": base_model,
                    "language": language,
                    "confirmed_languages": confirmed_languages,
                    "is_multilingual": language == "MULTI" or len(confirmed_languages) > 1,
                    "label_count": label_count if label_count else 0,
                    "metrics": metrics,
                    "metrics_per_language": metrics.get("per_language", {}) if isinstance(metrics, dict) else {},
                    "updated_at": updated_at,
                    "column_prefix": self._sanitize_model_prefix(relative_name),
                    "label_value": label_value,
                    "id2label_pairs": id2label_pairs,
                    "metadata_label_names": metadata_label_names,
                }
            )

        entries.sort(key=lambda entry: entry["updated_at"], reverse=True)
        return entries

    def _load_language_metrics(self, model_dir: Path) -> Dict[str, Any]:
        """Load language metrics if available."""
        metrics_path = model_dir / "language_performance.json"
        if not metrics_path.exists():
            return {}

        try:
            with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                data = json.load(metrics_file)
        except Exception as exc:
            self.logger.debug("Could not read metrics for %s: %s", model_dir, exc)
            return {}

        if isinstance(data, list) and data:
            latest = data[-1]
            averages = latest.get("averages", {})
            macro = averages.get("macro_f1") or averages.get("f1_macro")
            per_language: Dict[str, float] = {}
            metrics_by_language = latest.get("metrics", {}) or {}
            for lang_code, values in metrics_by_language.items():
                if not isinstance(values, dict):
                    continue
                normalized = LanguageNormalizer.normalize_language(lang_code) or str(lang_code)
                lang_key = normalized.upper()
                score = values.get("macro_f1") or values.get("f1_macro")
                if isinstance(score, (int, float)):
                    per_language[lang_key] = float(score)
            return {
                "macro_f1": macro,
                "per_language": per_language,
                "raw": latest,
            }

        return {}

    def _load_training_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """Load training metadata (task info, label names, etc.) if present."""
        metadata_path = model_dir / "training_metadata.json"
        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                data = json.load(metadata_file)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            self.logger.debug("Could not read training metadata for %s: %s", model_dir, exc)
        return {}

    def _infer_language(self, model_dir: Path, base_model: str, config: Dict[str, Any]) -> str:
        """Infer language code from path hints and model metadata."""
        # Check path components for explicit language markers
        for part in reversed(model_dir.parts):
            upper_part = part.upper()
            if upper_part in MODEL_LANGUAGE_MAP.values():
                return upper_part

            normalized = LanguageNormalizer.normalize_language(part)
            if normalized:
                return normalized.upper()

        # Check base model name and HuggingFace model type
        candidates = [base_model.lower(), str(config.get("model_type", "")).lower()]
        for candidate in candidates:
            for key, lang_code in MODEL_LANGUAGE_MAP.items():
                if key in candidate:
                    # For xlm-roberta, try to get languages from training metadata
                    if key == 'xlm-roberta' and lang_code == 'MULTI':
                        # Check if there's a training metadata file that contains language info
                        metadata_path = model_dir / "training_metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)
                                    languages = metadata.get("confirmed_languages", [])
                                    if languages:
                                        return "/".join([lang.upper() for lang in languages])
                            except:
                                pass
                        # Fallback to MULTI if no specific languages found
                        return "MULTI"  # Default when languages are unknown
                    return lang_code

        return "MULTI" if "xlm" in base_model.lower() else "EN"

    def _condense_relative_name(self, relative_name: str) -> str:
        """Collapse long model paths with an ellipsis while keeping key anchors."""
        parts = relative_name.split("/")
        if len(parts) <= 5:
            return relative_name

        head = parts[0]
        section = parts[1]
        task = parts[2]
        tail = "/".join(parts[-2:])
        return f"{head}/{section}/{task}/…/{tail}"

    def _shorten_base_model(self, base_model: str) -> str:
        """Light clean-up for base model folder names (underscores → dashes)."""
        pretty = base_model.replace("_", "-")
        pretty = re.sub(r"-{2,}", "-", pretty)
        return pretty

    def _sanitize_model_prefix(self, name: str) -> str:
        """Create a safe prefix for generated columns based on the model name."""
        sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
        return sanitized.lower() if sanitized else "model"

    def _load_tokenizer(self, model_info: Dict[str, Any]) -> Optional[AutoTokenizer]:
        """Try to load tokenizer from fine-tuned folder or fallback to base model."""
        search_candidates = []
        model_path = model_info.get('path')
        if model_path:
            search_candidates.append(str(model_path))

        config = model_info.get('config', {})
        for key in ['_name_or_path', 'model_name', 'base_model_name_or_path']:
            value = config.get(key)
            if isinstance(value, str):
                search_candidates.append(value)

        architectures = config.get('architectures')
        if isinstance(architectures, list) and architectures:
            search_candidates.append(architectures[0])

        tried = []
        for candidate in search_candidates:
            if candidate in tried:
                continue
            tried.append(candidate)
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                return tokenizer
            except Exception as exc:
                self.logger.debug("Tokenizer load failed for %s: %s", candidate, exc)

        self.console.print("[yellow]⚠ Unable to load tokenizer for token analysis.[/yellow]")
        return None

    def _analyze_languages(
        self,
        df: pd.DataFrame,
        text_column: str,
        sample_size: int = 200,  # kept for signature compatibility; full dataset is analysed regardless
    ) -> Dict[str, Any]:
        """Analyse language usage across the entire dataset and cache per-row assignments."""
        from llm_tool.utils.language_detector import LanguageDetector

        results: Dict[str, Any] = {
            'languages_detected': {},
            'text_length_stats': {
                'avg_length': 0,
                'max_length': 0,
                'min_length': 0,
                'median_length': 0,
            },
            'long_document_percentage': 0,
            'user_prefers_long_models': False,
        }

        if text_column not in df.columns:
            return results

        text_series = df[text_column]
        if text_series.empty:
            return results

        texts = text_series.fillna("").astype(str)
        if texts.empty:
            return results

        detector = LanguageDetector()
        if detector.method is None:
            detected_series = pd.Series(["UNKNOWN"] * len(df), index=df.index)
        else:
            self.console.print("\n[bold cyan]🔍 Language detection in progress...[/bold cyan]")
            self.console.print(f"[cyan]Analyzing {len(texts):,} texts to detect their language.[/cyan]\n")
            show_progress = HAS_TQDM and len(texts) > 0
            detections = detector.detect_batch(
                texts.tolist(),
                parallel=len(texts) > 50,
                show_progress=show_progress,
                desc="Detecting languages"
            )
            normalized_codes: List[str] = []
            for res in detections:
                if isinstance(res, dict):
                    lang_code = res.get('language') or 'UNKNOWN'
                else:
                    lang_code = res or 'UNKNOWN'
                normalized = LanguageNormalizer.normalize_language(lang_code)
                if normalized:
                    normalized_codes.append(normalized.upper())
                else:
                    lang_str = str(lang_code).strip().upper()
                    normalized_codes.append(lang_str if lang_str else 'UNKNOWN')
            # Ensure we preserve alignment with the dataframe index
            detected_series = pd.Series(normalized_codes, index=texts.index).reindex(df.index, fill_value='UNKNOWN')

        # Cache full assignments for downstream reuse
        self._language_assignments = detected_series

        language_counts = Counter(code.lower() for code in detected_series)
        results['languages_detected'] = dict(language_counts)

        text_lengths = texts.str.len()
        if not text_lengths.empty:
            import statistics  # pylint: disable=import-outside-toplevel

            avg_length = float(text_lengths.mean())
            max_length = int(text_lengths.max())
            min_length = int(text_lengths.min())
            median_length = statistics.median(text_lengths.tolist())
            results['text_length_stats'] = {
                'avg_length': avg_length,
                'max_length': max_length,
                'min_length': min_length,
                'median_length': median_length,
            }

            long_docs = int((text_lengths > 2048).sum())
            if len(text_lengths) > 0:
                results['long_document_percentage'] = (long_docs / len(text_lengths)) * 100
                results['user_prefers_long_models'] = results['long_document_percentage'] > 20

        return results

    def _present_language_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Display language detection results consistently with Training Arena."""
        languages_detected = analysis_results.get('languages_detected', {})
        text_stats = analysis_results.get('text_length_stats', {})

        if languages_detected:
            self.console.print("\n[bold]🌍 Languages Detected:[/bold]")
            total = sum(languages_detected.values())
            for lang, count in sorted(languages_detected.items(), key=lambda item: -item[1]):
                share = (count / total * 100) if total else 0
                self.console.print(f"  • {lang.upper()}: {count} samples ({share:.1f}%)")
        else:
            self.console.print("\n[yellow]⚠ Unable to detect languages automatically for this dataset.[/yellow]")

        if text_stats:
            self.console.print("\n[bold]📊 Text Statistics:[/bold]")
            self.console.print(f"  • Average length: {text_stats.get('avg_length', 0):.0f} characters")
            self.console.print(f"  • Median length: {text_stats.get('median_length', 0):.0f} characters")
            self.console.print(f"  • Max length: {text_stats.get('max_length', 0):.0f} characters")

        long_pct = analysis_results.get('long_document_percentage', 0)
        if long_pct:
            self.console.print(f"  • Long documents (>512 tokens): {long_pct:.1f}%")
            if analysis_results.get('user_prefers_long_models'):
                self.console.print("[yellow]💡 Consider long-context models (Longformer, BigBird, etc.).[/yellow]")

    def _detect_factory_context(self, data_source: Dict[str, Any], df: pd.DataFrame) -> bool:
        """Return True when running inside an Annotator Factory workflow."""
        if not isinstance(data_source, dict):
            return False
        if data_source.get('context') == 'annotator_factory' or data_source.get('factory_context'):
            return True
        path_candidates: List[Any] = [
            data_source.get('path') if data_source.get('type') == 'file' else None,
            data_source.get('directory'),
        ]
        for hint in path_candidates:
            if isinstance(hint, (str, Path)) and "annotator_factory" in str(hint):
                return True
        return False

    def _identify_annotated_rows(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Build a boolean mask selecting rows that already carry annotations."""
        mask: Optional[pd.Series] = None

        if 'annotation_status_per_prompt' in df.columns:
            statuses = df['annotation_status_per_prompt'].fillna('').astype(str).str.lower()
            status_mask = statuses.str.contains('success') | statuses.str.contains('complete')
            mask = status_mask

        if 'annotation' in df.columns:
            annotations = df['annotation'].fillna('').astype(str).str.strip()
            annotation_mask = annotations != ''
            mask = annotation_mask if mask is None else (mask | annotation_mask)

        if mask is not None and mask.any():
            return mask.astype(bool)
        return None

    def _maybe_limit_to_annotated(
        self,
        df: pd.DataFrame,
        context_key: str,
        display_label: str,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Restrict analysis to annotated rows when running inside Annotator Factory.

        Returns the DataFrame to use and whether the filter was applied.
        """
        if not self._is_factory_context or self._annotated_row_mask is None:
            return df, False

        annotated_df = df[self._annotated_row_mask]
        if annotated_df.empty or len(annotated_df) == len(df):
            return df, False

        if context_key not in self._factory_notice_shown:
            self.console.print(
                f"[dim]Annotator Factory: {display_label} limited to "
                f"{len(annotated_df):,} annotated rows (out of {len(df):,}).[/dim]"
            )
            self._factory_notice_shown.add(context_key)

        return annotated_df, True

    def _update_factory_context(self, data_source: Dict[str, Any], df: pd.DataFrame) -> None:
        """Refresh Annotator Factory context tracking for the currently loaded dataset."""
        self._factory_notice_shown = set()
        self._annotated_row_mask = None
        self._factory_annotated_count = 0
        self._is_factory_context = self._detect_factory_context(data_source, df)

        if not self._is_factory_context:
            return

        mask = self._identify_annotated_rows(df)
        if mask is None:
            self._is_factory_context = False
            return

        self._annotated_row_mask = mask
        self._factory_annotated_count = int(mask.sum())

    def _display_text_length_stats(self, df: pd.DataFrame, text_column: str, model_info: Dict[str, Any]) -> None:
        """Show descriptive statistics for text length (characters, words, tokens)."""
        analysis_df, _ = self._maybe_limit_to_annotated(
            df,
            context_key="text_length",
            display_label="text length analysis",
        )
        series = analysis_df[text_column].fillna("").astype(str)
        if series.empty:
            self.console.print("[yellow]⚠ Unable to calculate lengths (empty column).[/yellow]")
            return

        total_rows = len(df[text_column])
        subset_rows = len(series)
        max_full_analysis = 10000
        sample_size = 5000
        self.console.print("[cyan]Analysis in progress: measuring text lengths...[/cyan]")
        analysis_series = series
        sampled = False

        if subset_rows > max_full_analysis:
            sampled = Confirm.ask(
                f"[cyan]{subset_rows:,} rows detected. Analyze a random sample of {sample_size}?[/cyan]",
                default=True
            )
            if sampled:
                analysis_series = series.sample(sample_size, random_state=42)
            else:
                analysis_series = series

        texts_list = analysis_series.tolist()
        batch_size_chars = 1024
        use_length_pbar = HAS_TQDM and len(texts_list) > batch_size_chars
        iterator = (
            tqdm(
                range(0, len(texts_list), batch_size_chars),
                desc="Measuring text lengths",
                unit="rows",
                leave=False,
            )
            if use_length_pbar
            else range(0, len(texts_list), batch_size_chars)
        )
        char_lengths: List[int] = []
        word_counts: List[int] = []
        try:
            for start in iterator:
                chunk = texts_list[start:start + batch_size_chars]
                char_lengths.extend(len(text) for text in chunk)
                word_counts.extend(len(text.split()) if text else 0 for text in chunk)
        finally:
            if use_length_pbar:
                iterator.close()

        if not char_lengths:
            self.console.print("[yellow]⚠ Unable to calculate lengths (empty column).[/yellow]")
            return

        lengths = np.asarray(char_lengths, dtype=np.int64)
        words_array = np.asarray(word_counts, dtype=np.int64)

        percentiles = [50, 75, 90, 95, 99]
        percentile_values = np.percentile(lengths, percentiles)

        stats_table = Table(title="Text Length Analysis", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green", justify="right")

        stats_table.add_row("Rows analyzed", f"{len(analysis_series):,} / {total_rows:,}")
        stats_table.add_row("Average length (char.)", f"{lengths.mean():.1f}")
        stats_table.add_row("Median length (char.)", f"{percentile_values[0]:.0f}")
        stats_table.add_row("Average length (words)", f"{words_array.mean():.1f}")
        stats_table.add_row("Max (char.)", f"{lengths.max():,}")
        stats_table.add_row("90th percentile (char.)", f"{percentile_values[2]:.0f}")
        stats_table.add_row("95th percentile (char.)", f"{percentile_values[3]:.0f}")
        stats_table.add_row("99th percentile (char.)", f"{percentile_values[4]:.0f}")

        over_512 = (lengths > 512).mean() * 100
        over_1024 = (lengths > 1024).mean() * 100
        stats_table.add_row(">512 char.", f"{over_512:.1f}%")
        stats_table.add_row(">1024 char.", f"{over_1024:.1f}%")

        # Token-level analysis
        tokenizer = self._load_tokenizer(model_info)
        token_percentiles_values = None
        if tokenizer is not None:
            token_lengths: List[int] = []
            batch_size = 256
            use_token_pbar = HAS_TQDM and len(texts_list) > batch_size
            token_iterator = (
                tqdm(
                    range(0, len(texts_list), batch_size),
                    desc="Analyzing token lengths",
                    unit="batch",
                    leave=False,
                )
                if use_token_pbar
                else range(0, len(texts_list), batch_size)
            )
            try:
                for i in token_iterator:
                    batch = texts_list[i:i + batch_size]
                    try:
                        encoded = tokenizer(
                            batch,
                            add_special_tokens=True,
                            truncation=False,
                            return_attention_mask=False,
                            return_token_type_ids=False
                        )
                        token_lengths.extend(len(ids) for ids in encoded['input_ids'])
                    except Exception as exc:
                        self.logger.debug("Token analysis failed on batch %s: %s", i, exc)
                        token_lengths = []
                        break
            finally:
                if use_token_pbar:
                    token_iterator.close()

            if token_lengths:
                token_array = np.array(token_lengths)
                token_percentiles_values = np.percentile(token_array, percentiles)
                stats_table.add_row("Average length (tokens)", f"{token_array.mean():.1f}")
                stats_table.add_row("Median length (tokens)", f"{token_percentiles_values[0]:.0f}")
                stats_table.add_row("Max (tokens)", f"{token_array.max():,}")
                stats_table.add_row("90th percentile (tokens)", f"{token_percentiles_values[2]:.0f}")
                stats_table.add_row("95th percentile (tokens)", f"{token_percentiles_values[3]:.0f}")
                stats_table.add_row("99th percentile (tokens)", f"{token_percentiles_values[4]:.0f}")

                over_512_tok = (token_array > 512).mean() * 100
                over_1024_tok = (token_array > 1024).mean() * 100
                stats_table.add_row(">512 tokens", f"{over_512_tok:.1f}%")
                stats_table.add_row(">1024 tokens", f"{over_1024_tok:.1f}%")

        self._print_table(stats_table)

        if sampled:
            self.console.print(
                f"[dim]Analysis performed on a random sample of {len(analysis_series):,} rows.[/dim]"
            )

        if over_512 > 10 or (token_percentiles_values is not None and token_percentiles_values[3] > 512):
            self.console.print(
                "[yellow]⚠ A significant proportion of texts exceeds 512 tokens: consider a long-context model (Longformer, BigBird, LED...).[/yellow]"
            )

    def _is_unique_series(self, series: pd.Series) -> bool:
        """Check uniqueness and absence of missing values."""
        if series.isna().any():
            return False
        return series.nunique(dropna=False) == len(series)

    def _compute_worker_counts(self, config: Dict[str, Any], resources) -> Tuple[int, int]:
        """Estimate CPU and GPU worker counts based on configuration."""
        import os

        total_cpus = max(os.cpu_count() or 1, 1)
        default_cpu_workers = max(total_cpus - 1, 1)
        gpu_available = resources.gpu.available

        parallel = config.get('parallel', True)
        device_mode = config.get('device_mode', 'both')

        cpu_workers = 0
        gpu_workers = 0

        if not parallel:
            if device_mode == 'gpu' and gpu_available:
                gpu_workers = 1
            elif device_mode == 'both' and gpu_available:
                gpu_workers = 1
            else:
                cpu_workers = 1
        else:
            if device_mode == 'cpu':
                cpu_workers = default_cpu_workers
            elif device_mode == 'gpu':
                gpu_workers = 1
            elif device_mode == 'both':
                if gpu_available:
                    gpu_workers = 1
                    cpu_workers = default_cpu_workers
                else:
                    cpu_workers = default_cpu_workers

        return cpu_workers, gpu_workers

    def _ensure_unique_identifier(self, df: pd.DataFrame, column_mapping: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enforce a unique identifier column for each row."""
        text_column = column_mapping['text']
        current_id = column_mapping.get('id')

        self.console.print("\n[bold cyan]Why an ID column matters:[/bold cyan]")
        self.console.print("[dim]Every row needs a stable identifier so you can reconcile predictions with the original data.[/dim]")
        self.console.print("[dim]Pick an existing unique column, combine several columns, or generate a brand new one.[/dim]\n")

        def select_candidate_id() -> Optional[str]:
            candidates = []
            total = len(df)
            for col in df.columns:
                if col == text_column:
                    continue
                unique_ratio = df[col].nunique(dropna=False) / max(total, 1)
                if unique_ratio > 0.98 and not df[col].isna().any():
                    candidates.append((col, unique_ratio))
            if not candidates:
                return None

            candidates.sort(key=lambda x: -x[1])
            table = Table(title="Highly Unique Columns", box=box.ROUNDED)
            table.add_column("#", style="cyan", width=4)
            table.add_column("Column", style="green")
            table.add_column("Estimated uniqueness", style="magenta", justify="right")
            for idx, (col, ratio) in enumerate(candidates[:20], 1):
                table.add_row(str(idx), col, f"{ratio*100:.1f}%")
            self._print_table(table)
            choice = Prompt.ask(
                "Select a column (or 0 to cancel)",
                choices=[str(i) for i in range(0, min(len(candidates), 20) + 1)],
                default="0"
            )
            if choice == "0":
                return None
            return candidates[int(choice) - 1][0]

        while True:
            if current_id and current_id in df.columns and self._is_unique_series(df[current_id]):
                column_mapping['id'] = current_id
                return df, column_mapping

            if current_id:
                duplicates = int(df[current_id].duplicated(keep=False).sum())
                missing = int(df[current_id].isna().sum())
                self.console.print(
                    f"[yellow]⚠ Column '{current_id}' is not usable yet "
                    f"(duplicates: {duplicates:,}, missing: {missing:,}).[/yellow]"
                )
                current_id = None
            else:
                self.console.print("[yellow]⚠ A unique identifier is required for each row.[/yellow]")

            options = [
                ("1", "Choose another column"),
                ("2", "Combine multiple columns"),
                ("3", "Generate sequential identifier"),
                ("4", "Cancel annotation")
            ]
            menu = Table(box=box.ROUNDED)
            menu.add_column("#", style="cyan", width=4)
            menu.add_column("Option", style="green")
            for key, label in options:
                menu.add_row(key, label)
            self.console.print(menu)

            choice = Prompt.ask("Selection", choices=[opt[0] for opt in options], default="3")

            if choice == "1":
                candidate = select_candidate_id()
                if candidate:
                    current_id = candidate
                    self.console.print(f"[green]✓ Column '{candidate}' selected.[/green]")
            elif choice == "2":
                cols_input = Prompt.ask("Columns to combine (comma-separated)")
                cols = [c.strip() for c in cols_input.split(",") if c.strip() in df.columns]
                if len(cols) < 2:
                    self.console.print("[red]❌ Select at least two valid columns.[/red]")
                    continue
                new_col = "combined_id"
                base_name = new_col
                counter = 1
                while new_col in df.columns:
                    counter += 1
                    new_col = f"{base_name}_{counter}"
                df[new_col] = df[cols].astype(str).agg("::".join, axis=1)
                if self._is_unique_series(df[new_col]):
                    current_id = new_col
                    self.console.print(f"[green]✓ Column '{new_col}' created from {', '.join(cols)}[/green]")
                else:
                    self.console.print("[red]❌ Combination is not unique. Try another combination.[/red]")
                    df.drop(columns=[new_col], inplace=True)
            elif choice == "3":
                base_name = "annotation_id"
                new_col = base_name
                counter = 1
                while new_col in df.columns:
                    counter += 1
                    new_col = f"{base_name}_{counter}"
                df[new_col] = [f"{new_col}_{i+1}" for i in range(len(df))]
                current_id = new_col
                self.console.print(f"[green]✓ Sequential identifier '{new_col}' generated ({len(df)} rows).[/green]")
            else:
                raise KeyboardInterrupt("Annotation cancelled by user (missing identifier).")

        # Should never reach here
        return df, column_mapping

    def _get_or_compute_row_languages(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
        language_info: Optional[Dict[str, Any]]
    ) -> pd.Series:
        """Return a series with ISO language codes for each row."""
        if (
            self._language_assignments is not None
            and len(self._language_assignments) == len(df)
        ):
            return self._language_assignments

        info = language_info or {}
        text_column = column_mapping['text']
        lang_column = info.get('language_column')

        def normalize_value(val: Any) -> str:
            if pd.isna(val):
                return 'UNKNOWN'
            norm = LanguageNormalizer.normalize_language(val)
            if norm:
                return norm.upper()
            val_str = str(val).strip()
            return val_str.upper() if val_str else 'UNKNOWN'

        if lang_column and lang_column in df.columns:
            series = df[lang_column].map(normalize_value)
        else:
            detector = LanguageDetector()
            if detector.method is None:
                self.console.print(
                    "[yellow]⚠ No language detection module available. All rows will be marked as UNKNOWN.[/yellow]"
                )
                series = pd.Series(['UNKNOWN'] * len(df), index=df.index)
            else:
                self.console.print("\n[bold cyan]🔍 Language detection in progress...[/bold cyan]")
                self.console.print(f"[cyan]Analyzing {len(df):,} texts to detect their language.[/cyan]\n")
                texts_list = df[text_column].fillna("").astype(str).tolist()
                show_progress = HAS_TQDM and len(texts_list) > 0
                results = detector.detect_batch(
                    texts_list,
                    parallel=len(texts_list) > 20,
                    show_progress=show_progress,
                    desc="Detecting languages",
                )
                codes = []
                for res in results:
                    lang = res.get('language') or 'UNKNOWN'
                    codes.append(lang.upper())
                series = pd.Series(codes, index=df.index)
                info['language_column'] = info.get('language_column') or '__detected_language__'
                info['detection_source'] = info.get('detection_source') or 'detector'

        self._language_assignments = series
        return series

    def _display_annotation_examples(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
        language_series: pd.Series,
        language_mask: Optional[pd.Series],
        languages_to_annotate: List[str],
    ) -> None:
        """Display a didactic preview of rows that will receive annotations."""
        if not languages_to_annotate:
            return

        text_column = column_mapping.get("text")
        if not text_column or text_column not in df.columns:
            return

        id_column = column_mapping.get("id")

        if language_mask is not None and len(language_mask) == len(df):
            eligible_index = df.index[language_mask]
        else:
            eligible_index = df.index

        if eligible_index.empty:
            return

        sample_table = Table(title="Upcoming Annotation Examples", box=box.ROUNDED)
        sample_table.add_column("Language", style="cyan", width=10)
        sample_table.add_column("Row ID", style="green", width=14)
        sample_table.add_column("Excerpt", style="white", overflow="fold")

        added = 0
        for lang in languages_to_annotate:
            lang_indices = language_series[language_series == lang].index.intersection(eligible_index)
            if lang_indices.empty:
                continue
            for row_idx in list(lang_indices[:3]):
                row = df.loc[row_idx]
                row_id = row[id_column] if id_column and id_column in df.columns else row_idx
                raw_text = str(row.get(text_column, ""))
                excerpt = textwrap.shorten(raw_text.replace("\n", " "), width=120, placeholder="…")
                sample_table.add_row(lang, str(row_id), excerpt)
                added += 1

        if added:
            self.console.print("\n[bold cyan]📝 Preview: Rows that WILL be annotated[/bold cyan]")
            self.console.print("[dim]These examples show rows whose language matches the model(s) you selected.[/dim]")
            self._print_table(sample_table)

    def _apply_dataset_scope(self, df: pd.DataFrame, scope: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Return a dataframe subset according to the configured coverage scope."""
        if not scope or scope.get("type") == "full":
            return df.copy()

        scope_type = scope.get("type")
        if scope_type == "head":
            size = max(1, min(int(scope.get("size", len(df))), len(df)))
            return df.head(size).copy()

        if scope_type == "random":
            size = max(1, min(int(scope.get("size", len(df))), len(df)))
            seed = int(scope.get("seed", 42))
            sampled = df.sample(n=size, random_state=seed)
            return sampled.sort_index().copy()

        return df.copy()

    def _resolve_entry_mask(self, entry: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        """Compute the boolean mask of rows eligible for a pipeline entry."""
        scope = entry.get("scope", {}) or {}
        scope_type = scope.get("type")
        if scope_type == "positive":
            parent_prefix = scope.get("parent_prefix")
            if not parent_prefix:
                return pd.Series(False, index=df.index)
            label_col = f"{parent_prefix}_label"
            if label_col not in df.columns:
                return pd.Series(False, index=df.index)

            labels = scope.get("labels") or []
            label_series = df[label_col]
            if labels:
                normalized_labels = {str(label) for label in labels}
                mask = label_series.astype(str).isin(normalized_labels)
            else:
                mask = label_series.notna()

            annotated_col = f"{parent_prefix}_annotated"
            if annotated_col in df.columns:
                mask = mask & df[annotated_col].fillna(False).astype(bool)

            return mask.reindex(df.index).fillna(False)

        return pd.Series(True, index=df.index)

    def _persist_run_metadata(self, metadata: Dict[str, Any], output_dir: Path) -> Path:
        """Write run metadata alongside exports and return the file path."""
        metadata_path = output_dir / "session_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return metadata_path

    def _confirm_parallel_config(self, config: Dict[str, Any], resources) -> bool:
        """Explain parallel parameters and ask for confirmation."""
        device_mode = config.get('device_mode', 'both')
        batch_cpu = config.get('batch_size_cpu', 32)
        batch_gpu = config.get('batch_size_gpu', 64)
        chunk_size = config.get('chunk_size', 1024)

        cpu_workers, gpu_workers = self._compute_worker_counts(config, resources)
        total_cpus = max(os.cpu_count() or 1, 1)
        gpu_available = resources.gpu.available

        summary = Table(title="Parallelisation Summary", box=box.ROUNDED)
        summary.add_column("Parameter", style="cyan")
        summary.add_column("Value", style="green", overflow="fold")
        summary.add_row("Mode", device_mode.upper())
        summary.add_row("CPU workers", str(cpu_workers))
        summary.add_row("GPU workers", str(gpu_workers))
        summary.add_row("CPU batch", str(batch_cpu))
        summary.add_row("GPU batch", str(batch_gpu))
        summary.add_row("Chunk size", str(chunk_size))
        summary.add_row("Total CPU cores", str(total_cpus))
        summary.add_row("GPU available", "Yes" if gpu_available else "No")
        self._print_table(summary)

        self.console.print("\n[dim]• Batch size: texts processed before the model updates.[/dim]")
        self.console.print("[dim]• Chunk size: payload sent to each worker (keeps transfers efficient).[/dim]")
        self.console.print("[dim]• CPU workers handle batches in parallel while GPU workers focus on larger batches.[/dim]")
        self.console.print("[dim]• Increase GPU batch size only if you have enough GPU memory.[/dim]\n")

        return Confirm.ask("[cyan]Confirm this configuration?[/cyan]", default=True)

    def _select_data_source(self) -> Optional[Dict[str, Any]]:
        """Select data source"""
        source_choices = [
            "📁 Local file (CSV, TSV, Excel, JSON, JSONL, Parquet, RData/RDS)",
            "🗄️  SQL database (PostgreSQL/MySQL/SQLite/SQL Server/Custom)",
            "← Back"
        ]

        self.console.print("\n[cyan]Load the dataset you want to annotate. You can browse local files or connect to a database.[/cyan]\n")

        source_table = Table(box=box.ROUNDED)
        source_table.add_column("#", style="cyan", width=4)
        source_table.add_column("Data Source", style="green", width=70)

        for idx, choice in enumerate(source_choices, 1):
            source_table.add_row(str(idx), choice)

        self._print_table(source_table)

        choice = Prompt.ask("\n[cyan]Select data source[/cyan]", choices=["1", "2", "3"], default="1")

        if choice == "3":
            return None
        if choice == "1":
            return self._select_file_source()
        return self._select_sql_source()

    def _select_file_source(self) -> Optional[Dict[str, Any]]:
        """Select file source with auto-detection"""
        from llm_tool.utils.data_detector import DataDetector, DatasetInfo

        # Auto-detect datasets in data directory
        data_dir = self.data_dir
        detector = DataDetector()
        detected_datasets = detector.scan_directory(data_dir)

        if detected_datasets:
            self.console.print(f"\n[bold cyan]📊 Found {len(detected_datasets)} dataset(s) in {data_dir}:[/bold cyan]\n")

            # Create table with dataset preview
            datasets_table = Table(title="Available Datasets", border_style="cyan", show_header=True, box=box.ROUNDED)
            datasets_table.add_column("#", style="bold yellow", width=4)
            datasets_table.add_column("Filename", style="white", width=35)
            datasets_table.add_column("Format", style="green", width=10)
            datasets_table.add_column("Size", style="magenta", width=12)
            datasets_table.add_column("Rows", style="cyan", width=10)
            datasets_table.add_column("Columns", style="blue", width=10)
            datasets_table.add_column("Preview", style="dim", width=40)

            for i, ds in enumerate(detected_datasets[:20], 1):
                # Format size
                if ds.size_mb < 0.1:
                    size_str = f"{ds.size_mb * 1024:.1f} KB"
                else:
                    size_str = f"{ds.size_mb:.1f} MB"

                # Format rows and columns
                rows_str = f"{ds.rows:,}" if ds.rows else "?"
                cols_str = str(len(ds.columns)) if ds.columns else "?"

                # Create preview of columns with text scores
                preview_parts = []
                if ds.columns:
                    # Show up to 3 columns with highest text scores
                    sorted_cols = sorted(
                        [(col, ds.text_scores.get(col, 0)) for col in ds.columns],
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    preview_parts = [f"{col} ({score:.0f})" for col, score in sorted_cols if score > 0]

                preview_str = ", ".join(preview_parts) if preview_parts else "No text columns"

                datasets_table.add_row(
                    str(i),
                    ds.path.name,
                    ds.format.upper(),
                    size_str,
                    rows_str,
                    cols_str,
                    preview_str[:40] + "..." if len(preview_str) > 40 else preview_str
                )

            self._print_table(datasets_table)
            self.console.print()

            # Ask user: use detected or manual path
            use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)

            if use_detected:
                choice = Prompt.ask(
                    "[cyan]Select dataset[/cyan]",
                    choices=[str(i) for i in range(1, min(len(detected_datasets) + 1, 21))],
                    default="1"
                )
                selected_dataset = detected_datasets[int(choice) - 1]
                file_path = selected_dataset.path

                self.console.print(f"\n[green]✓ Selected: {file_path.name}[/green]")
            else:
                # Manual path entry
                file_path = Prompt.ask("\n[cyan]File path[/cyan]")
                file_path = Path(file_path).expanduser()

                if not file_path.exists():
                    self.console.print(f"[red]✗ File not found[/red]")
                    return None
        else:
            # No datasets detected - ask for manual path
            self.console.print("[yellow]⚠ No datasets auto-detected in data directory[/yellow]")
            file_path = Prompt.ask("\n[cyan]File path[/cyan]")
            file_path = Path(file_path).expanduser()

            if not file_path.exists():
                self.console.print(f"[red]✗ File not found[/red]")
                return None

        # Determine format (support ALL formats from the package)
        suffix = file_path.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.tsv': 'tsv',
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.parquet': 'parquet',
            '.rdata': 'rdata',
            '.rds': 'rds'
        }
        file_format = format_map.get(suffix, 'unknown')

        if file_format == 'unknown':
            self.console.print(f"[red]✗ Unsupported format: {suffix}[/red]")
            return None

        self.console.print(f"[green]✓ Format: {file_format.upper()}[/green]")
        return {'type': 'file', 'path': str(file_path), 'format': file_format}

    def _build_factory_data_source(self) -> Optional[Dict[str, Any]]:
        """Return a pre-configured data source when launched from Annotator Factory."""
        if not self._factory_launch_active:
            return None
        config = self._factory_launch_config or {}
        dataset_path = config.get("dataset_path")
        if not dataset_path:
            return None
        candidate = Path(dataset_path).expanduser()
        if not candidate.exists():
            if self.console:
                self.console.print(f"[yellow]Annotator Factory: dataset not found at {candidate}[/yellow]")
            return None
        format_map = {
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.parquet': 'parquet',
            '.rdata': 'rdata',
            '.rds': 'rds',
        }
        detected_format = format_map.get(candidate.suffix.lower())
        if not detected_format:
            if self.console:
                self.console.print(f"[yellow]Annotator Factory: unsupported dataset format {candidate.suffix}[/yellow]")
            return None
        return {
            'type': 'file',
            'path': str(candidate),
            'format': detected_format,
        }

    def _apply_factory_column_defaults(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject factory-provided defaults (like the text column) into the column mapping."""
        config = self._factory_launch_config or {}
        desired_text = config.get("text_column")
        if desired_text and desired_text in df.columns:
            column_mapping = dict(column_mapping)
            column_mapping['text'] = desired_text
        return column_mapping

    def _select_sql_source(self) -> Optional[Dict[str, Any]]:
        """Interactive SQL source selector with connection helper."""
        self.console.print("\n[cyan]Available database types[/cyan]\n")

        db_table = Table(box=box.ROUNDED)
        db_table.add_column("#", style="cyan", width=4)
        db_table.add_column("Database", style="green", width=30)
        db_table.add_column("Driver Hint", style="dim", width=28)
        db_table.add_row("1", "PostgreSQL", "Requires psycopg2 or pg8000")
        db_table.add_row("2", "MySQL / MariaDB", "Requires pymysql or mysqlclient")
        db_table.add_row("3", "SQLite", "Built-in (file path or :memory:)")
        db_table.add_row("4", "Microsoft SQL Server", "Requires pyodbc")
        db_table.add_row("5", "Custom SQLAlchemy URL", "Paste full URL")
        db_table.add_row("6", "← Back", "")
        self._print_table(db_table)

        choice = Prompt.ask("\n[cyan]Database type[/cyan]", choices=[str(i) for i in range(1, 7)], default="1")
        if choice == "6":
            return None

        connection_string = None
        display_name = ""

        if choice == "1":  # PostgreSQL
            host = Prompt.ask("Host", default="localhost")
            port = IntPrompt.ask("Port", default=5432)
            database = Prompt.ask("Database name")
            username = Prompt.ask("Username", default="postgres")
            password = Prompt.ask("Password", password=True)
            connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
            display_name = f"PostgreSQL • {database}@{host}:{port}"

        elif choice == "2":  # MySQL
            host = Prompt.ask("Host", default="localhost")
            port = IntPrompt.ask("Port", default=3306)
            database = Prompt.ask("Database name")
            username = Prompt.ask("Username", default="root")
            password = Prompt.ask("Password", password=True)
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            display_name = f"MySQL • {database}@{host}:{port}"

        elif choice == "3":  # SQLite
            file_path = Prompt.ask("SQLite file path (or :memory:)", default=str(self.data_dir / "database.sqlite"))
            if file_path != ":memory:":
                file_path = str(Path(file_path).expanduser())
            connection_string = f"sqlite:///{file_path}"
            display_name = f"SQLite • {file_path}"

        elif choice == "4":  # SQL Server
            host = Prompt.ask("Host", default="localhost")
            port = IntPrompt.ask("Port", default=1433)
            database = Prompt.ask("Database name")
            username = Prompt.ask("Username")
            password = Prompt.ask("Password", password=True)
            driver = Prompt.ask("ODBC driver", default="ODBC Driver 17 for SQL Server")
            connection_string = (
                f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}"
                f"?driver={driver.replace(' ', '+')}"
            )
            display_name = f"SQL Server • {database}@{host}:{port}"

        elif choice == "5":  # Custom URL
            connection_string = Prompt.ask(
                "Enter SQLAlchemy connection URL",
                default="postgresql+psycopg2://user:password@host:5432/database"
            )
            display_name = "Custom SQL"

        if not connection_string:
            self.console.print("[red]✗ Invalid connection information[/red]")
            return None

        self.console.print("\n[cyan]Testing database connection...[/cyan]")
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.console.print("[green]✓ Connection successful[/green]")
        except Exception as exc:
            self.console.print(f"[red]✗ Connection failed: {exc}[/red]")
            self.console.print("[yellow]Tip: ensure the appropriate database driver is installed.[/yellow]")
            return None

        try:
            inspector = inspect(engine)
            schemas = inspector.get_schema_names()
        except Exception as exc:
            self.console.print(f"[red]✗ Failed to inspect database: {exc}[/red]")
            engine.dispose()
            return None

        schema = inspector.default_schema_name
        if schemas and len(schemas) > 1:
            schema_table = Table(box=box.ROUNDED)
            schema_table.add_column("#", style="cyan", width=4)
            schema_table.add_column("Schema", style="green", width=30)
            for idx, sch in enumerate(schemas, 1):
                schema_table.add_row(str(idx), sch)
            self.console.print("\n[cyan]Available schemas[/cyan]")
            self._print_table(schema_table)
            schema_choice = Prompt.ask(
                "\n[cyan]Schema[/cyan]",
                choices=[str(i) for i in range(1, len(schemas) + 1)],
                default=str(schemas.index(schema) + 1 if schema in schemas else 1)
            )
            schema = schemas[int(schema_choice) - 1]

        try:
            tables = inspector.get_table_names(schema=schema)
        except Exception as exc:
            self.console.print(f"[red]✗ Failed to list tables: {exc}[/red]")
            engine.dispose()
            return None

        tables = sorted(tables)

        table_table = Table(box=box.ROUNDED)
        table_table.add_column("#", style="cyan", width=4)
        table_table.add_column("Table", style="green")
        if tables:
            for idx, table_name in enumerate(tables[:50], 1):
                table_table.add_row(str(idx), table_name)
        self.console.print("\n[cyan]Tables (top 50)[/cyan]")
        if tables:
            self._print_table(table_table)
        else:
            self.console.print("[yellow]No tables found in this schema.[/yellow]")

        use_custom_query = Confirm.ask("\n[cyan]Use a custom SQL query instead of selecting a table?[/cyan]", default=False)
        query = None
        selected_table = None
        limit = None

        if use_custom_query:
            example = tables[0] if tables else "your_table"
            query = Prompt.ask(
                "Enter SQL query",
                default=f"SELECT * FROM {example} LIMIT 1000"
            )
        else:
            if not tables:
                self.console.print("[red]✗ Cannot select a table because none were detected[/red]")
                engine.dispose()
                return None

            choice_table = Prompt.ask(
                "\n[cyan]Select table[/cyan]",
                choices=[str(i) for i in range(1, min(len(tables), 50) + 1)],
                default="1"
            )
            selected_table = tables[int(choice_table) - 1]
            if Confirm.ask("Apply LIMIT when loading data?", default=True):
                limit = IntPrompt.ask("Row limit", default=25000)

        engine.dispose()

        return {
            'type': 'sql',
            'connection_string': connection_string,
            'schema': schema,
            'table': selected_table,
            'query': query,
            'limit': limit,
            'display_name': display_name
        }

    def _load_and_analyze_data(
        self,
        data_source: Dict[str, Any],
        models_or_plan: List[Dict[str, Any]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Load the dataset and help the user map mandatory columns."""
        self._language_assignments = None
        self._language_annotation_mask = None
        self._allowed_annotation_languages = []
        model_infos: List[Dict[str, Any]] = [
            entry["info"] if isinstance(entry, dict) and "info" in entry else entry
            for entry in (models_or_plan or [])
        ]

        try:
            if data_source['type'] == 'file':
                file_format = data_source['format']
                file_path = data_source['path']

                if file_format == 'csv':
                    df = pd.read_csv(file_path)
                elif file_format == 'tsv':
                    df = pd.read_csv(file_path, sep='\t')
                elif file_format == 'excel':
                    df = pd.read_excel(file_path)
                elif file_format == 'json':
                    df = pd.read_json(file_path)
                elif file_format == 'jsonl':
                    df = pd.read_json(file_path, lines=True)
                elif file_format == 'parquet':
                    df = pd.read_parquet(file_path)
                elif file_format in {'rdata', 'rds'}:
                    try:
                        import pyreadr  # type: ignore
                    except ImportError:
                        self.console.print("[red]✗ pyreadr not installed. Install with: pip install pyreadr[/red]")
                        return None, None

                    result = pyreadr.read_r(file_path)
                    if result:
                        df = list(result.values())[0]
                    else:
                        self.console.print("[red]✗ Empty RData/RDS file[/red]")
                        return None, None
                else:
                    self.console.print(f"[red]✗ Unsupported format: {file_format}[/red]")
                    return None, None
            else:
                connection_string = data_source['connection_string']
                schema = data_source.get('schema')
                selected_table = data_source.get('table')
                query = data_source.get('query')
                limit = data_source.get('limit')

                engine = create_engine(connection_string)
                try:
                    if query:
                        sql_query = text(query)
                        df = pd.read_sql_query(sql_query, engine)
                    else:
                        if not selected_table:
                            self.console.print("[red]✗ No table selected for SQL source[/red]")
                            return None, None

                        qualified = f"{schema}.{selected_table}" if schema and schema != "" else selected_table
                        dialect = engine.dialect.name.lower()

                        if limit:
                            if dialect in {"mssql", "sybase"}:
                                sql_query = text(f"SELECT TOP ({int(limit)}) * FROM {qualified}")
                            else:
                                sql_query = text(f"SELECT * FROM {qualified} LIMIT {int(limit)}")
                            df = pd.read_sql_query(sql_query, engine)
                        else:
                            try:
                                df = pd.read_sql_table(selected_table, con=engine, schema=schema)
                            except Exception:
                                # Fallback to generic SELECT *
                                df = pd.read_sql_query(text(f"SELECT * FROM {qualified}"), engine)
                finally:
                    engine.dispose()

            self.console.print(f"[green]✓ Loaded {len(df):,} rows, {len(df.columns)} columns[/green]\n")
            self._update_factory_context(data_source, df)

            analysis: Dict[str, Any] = {}
            if data_source['type'] == 'file':
                try:
                    analysis = DataDetector.analyze_file_intelligently(Path(file_path))
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.debug("Failed to analyse dataset structure for suggestions: %s", exc)

            column_names = list(df.columns)
            total_rows = len(df)

            overview_table = Table(
                title=f"Dataset Overview ({len(column_names)} columns, {total_rows:,} rows)",
                box=box.ROUNDED,
                show_lines=False,
            )
            overview_table.add_column("#", style="cyan", width=4)
            overview_table.add_column("Column Name", style="green", width=30)
            overview_table.add_column("Type", style="yellow", width=12)
            overview_table.add_column("Missing %", style="magenta", justify="right", width=10)
            overview_table.add_column("Unique", style="cyan", justify="right", width=10)
            overview_table.add_column("Sample Values", style="white", width=40, overflow="fold")

            for idx, col_name in enumerate(column_names, 1):
                series = df[col_name]
                dtype = str(series.dtype)
                missing_pct = (series.isna().sum() / total_rows * 100) if total_rows else 0.0
                unique_count = series.nunique(dropna=True)
                samples = series.dropna().astype(str).head(3).tolist()
                sample_preview = ", ".join(
                    f"{sample[:30]}…" if len(sample) > 30 else sample for sample in samples
                ) or "[empty]"
                overview_table.add_row(
                    str(idx),
                    col_name,
                    dtype,
                    f"{missing_pct:.1f}%",
                    f"{unique_count:,}",
                    sample_preview,
                )

            self._print_table(overview_table)
            self.console.print("\n[bold]💡 Helpful Suggestions[/bold] [dim](auto-detected candidates)[/dim]")
            self.console.print("[dim]You can pick any column from the overview above; these are just shortcuts.[/dim]\n")

            text_candidates = analysis.get('text_column_candidates', []) if analysis else []
            if not text_candidates:
                for col_name in column_names:
                    series = df[col_name]
                    if series.dtype == 'object':
                        non_null = series.dropna()
                        if not non_null.empty:
                            avg_length = non_null.astype(str).str.len().mean()
                            if avg_length >= 20:
                                text_candidates.append({'name': col_name, 'avg_length': avg_length})
                text_candidates.sort(key=lambda item: -item['avg_length'])

            id_candidates: List[Dict[str, Any]] = []
            for col_name in column_names:
                series = df[col_name]
                if series.isna().any():
                    continue
                unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
                if unique_ratio >= 0.98:
                    id_candidates.append({
                        'name': col_name,
                        'unique_ratio': unique_ratio,
                        'dtype': str(series.dtype),
                    })
            id_candidates.sort(key=lambda item: -item['unique_ratio'])

            suggestions_table = Table(box=box.SIMPLE)
            suggestions_table.add_column("Purpose", style="yellow")
            suggestions_table.add_column("Top Suggestion", style="green")
            suggestions_table.add_column("Why?", style="white", overflow="fold")

            text_column_default = None
            if text_candidates:
                top_text = text_candidates[0]
                text_column_default = top_text['name']
                suggestions_table.add_row(
                    "📝 Text column",
                    top_text['name'],
                    f"Avg length ≈ {top_text.get('avg_length', 0):.0f} characters",
                )
            else:
                suggestions_table.add_row("📝 Text column", "—", "No strong text-like column detected")

            if id_candidates:
                top_id = id_candidates[0]
                suggestions_table.add_row(
                    "🔑 ID column",
                    top_id['name'],
                    f"{top_id['unique_ratio']*100:.1f}% unique values ({top_id['dtype']})",
                )
            else:
                suggestions_table.add_row(
                    "🔑 ID column",
                    "—",
                    "No fully unique column found (you can create one next)",
                )

            self._print_table(suggestions_table)

            self.console.print("[bold cyan]Text Column[/bold cyan]")
            self.console.print("[dim]Pick the column that stores the raw text to annotate.[/dim]")
            self.console.print("[dim]Prefer long-form sentences/messages rather than IDs or metadata.[/dim]\n")

            if text_column_default and text_column_default in column_names:
                default_text_choice = str(column_names.index(text_column_default) + 1)
            else:
                default_text_choice = "1"

            text_col_idx = Prompt.ask(
                "[cyan]Select TEXT column[/cyan]",
                choices=[str(i) for i in range(1, len(column_names) + 1)],
                default=default_text_choice
            )
            text_column = column_names[int(text_col_idx) - 1]

            id_column = None
            if id_candidates:
                id_table = Table(title="Candidate ID Columns", box=box.ROUNDED)
                id_table.add_column("#", style="cyan", width=4)
                id_table.add_column("Column", style="green", width=30)
                id_table.add_column("Type", style="yellow", width=12)
                id_table.add_column("Unique %", style="cyan", width=10, justify="right")
                for idx, candidate in enumerate(id_candidates, 1):
                    id_table.add_row(
                        str(idx),
                        candidate['name'],
                        candidate['dtype'],
                        f"{candidate['unique_ratio'] * 100:.1f}%"
                    )
                id_table.add_row("0", "[dim]None[/dim]", "", "")

                self.console.print("\n[bold magenta]Identifier Column[/bold magenta]")
                self.console.print("[dim]A stable ID keeps predictions aligned with the original dataset.[/dim]")
                self.console.print("[dim]Choose a column with UNIQUE values or request an auto-generated one.[/dim]\n")
                self._print_table(id_table)

                id_choice = Prompt.ask(
                    "[cyan]Select ID column[/cyan]",
                    choices=[str(i) for i in range(0, len(id_candidates) + 1)],
                    default="0"
                )
                if id_choice != "0":
                    id_column = id_candidates[int(id_choice) - 1]['name']
            else:
                self.console.print("\n[bold magenta]Identifier Column[/bold magenta]")
                self.console.print("[dim]No highly unique column detected automatically.[/dim]")
                self.console.print("[dim]You will be able to craft a unique identifier (combine columns or auto-generate) in the next step.[/dim]\n")

            column_mapping = {'text': text_column, 'id': id_column, 'language': None}
            self.console.print(f"\n[green]✓ Text column: {text_column}[/green]")
            if id_column:
                self.console.print(f"[green]✓ ID column: {id_column}[/green]")

            if model_infos:
                self._display_text_length_stats(df, text_column, model_infos[0])
            df, column_mapping = self._ensure_unique_identifier(df, column_mapping)

            return df, column_mapping

        except Exception as e:
            self.console.print(f"[red]✗ Error:[/red] {str(e)}", markup=False, highlight=False)
            return None, None

    def _detect_and_validate_language(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
        models_or_plan: List[Dict[str, Any]],
    ) -> Optional[Dict]:
        """Detect dominant languages and confirm model compatibility."""
        working_df, _ = self._maybe_limit_to_annotated(
            df,
            context_key="language_detection",
            display_label="language detection",
        )
        text_column = column_mapping['text']
        model_infos: List[Dict[str, Any]] = [
            entry["info"] if isinstance(entry, dict) and "info" in entry else entry
            for entry in (models_or_plan or [])
        ]
        model_language_map: Dict[str, List[str]] = {}
        language_to_models: Dict[str, List[Dict[str, Any]]] = {}
        for model in model_infos:
            model_key = model.get("relative_name") or str(
                model.get("label_value")
                or model.get("base_model")
                or model.get("path")
                or "model"
            )
            raw_langs = model.get("confirmed_languages") or [model.get("language")]
            normalized_langs = sorted(
                {
                    (LanguageNormalizer.normalize_language(lang) or str(lang).strip() or "UNKNOWN").upper()
                    for lang in raw_langs
                    if lang is not None
                }
            )
            if not normalized_langs:
                fallback_lang = (
                    LanguageNormalizer.normalize_language(model.get("language"))
                    or model.get("language")
                    or "UNKNOWN"
                )
                normalized_langs = [str(fallback_lang).upper()]
            model_language_map[model_key] = normalized_langs
            for lang in normalized_langs:
                language_to_models.setdefault(lang, []).append(model)

        self.console.print("\n[cyan]We will analyze all texts in your dataset to detect their language.[/cyan]")
        self.console.print("[cyan]If you already track language codes, you can reuse that column to avoid auto-detection.[/cyan]\n")

        candidate_language_columns: List[Dict[str, Any]] = []
        for col in working_df.columns:
            if col in {text_column, column_mapping.get('id')}:
                continue
            if working_df[col].dtype != 'object':
                continue
            counts = LanguageNormalizer.detect_languages_in_column(working_df, col)
            if counts:
                candidate_language_columns.append({'name': col, 'counts': counts})

        language_column: Optional[str] = None
        detection_source = "detector"
        language_counts: Dict[str, int] = {}

        if candidate_language_columns:
            lang_table = Table(title="Detected Language Columns", box=box.ROUNDED)
            lang_table.add_column("#", style="cyan", width=4)
            lang_table.add_column("Column", style="green", width=30)
            lang_table.add_column("Languages", style="magenta", overflow="fold")
            for idx, candidate in enumerate(candidate_language_columns, 1):
                lang_summary = ", ".join(
                    f"{lang.upper()} ({count})" for lang, count in candidate['counts'].items()
                )
                lang_table.add_row(str(idx), candidate['name'], lang_summary)
            self._print_table(lang_table)

            if Confirm.ask("Use one of these columns for language detection?", default=True):
                lang_choice = Prompt.ask(
                    "\n[cyan]Select language column[/cyan]",
                    choices=[str(i) for i in range(1, len(candidate_language_columns) + 1)],
                    default="1",
                )
                selected = candidate_language_columns[int(lang_choice) - 1]
                language_column = selected['name']
                language_counts = selected['counts']
                detection_source = "column"

        analysis_results = self._analyze_languages(working_df, text_column)
        if language_counts:
            normalized_counts: Dict[str, int] = {}
            for lang, count in language_counts.items():
                norm = LanguageNormalizer.normalize_language(lang) or lang
                normalized_counts[norm.lower()] = normalized_counts.get(norm.lower(), 0) + count
            analysis_results['languages_detected'] = normalized_counts
            language_counts = normalized_counts
        else:
            language_counts = analysis_results.get('languages_detected', {})

        if not language_counts:
            language_counts = {'en': 1}
            analysis_results['languages_detected'] = language_counts
            self.console.print("[yellow]⚠ Unable to confidently detect language, defaulting to English.[/yellow]")

        self._present_language_analysis(analysis_results)

        language_counts_upper: Dict[str, int] = {}
        for lang, count in language_counts.items():
            normalized = LanguageNormalizer.normalize_language(lang) or lang
            key = normalized.upper()
            language_counts_upper[key] = language_counts_upper.get(key, 0) + count
        language_counts = language_counts_upper

        sorted_langs = sorted(language_counts.items(), key=lambda item: item[1], reverse=True)
        primary_lang = sorted_langs[0][0]
        unique_languages = set(language_counts.keys())

        if detection_source == "column" and language_column:
            column_mapping['language'] = language_column
            normalized_series = working_df[language_column].map(
                lambda val: (LanguageNormalizer.normalize_language(val) or str(val).strip() or "UNKNOWN").upper()
            )
            if len(normalized_series) == len(df):
                self._language_assignments = normalized_series
        else:
            column_mapping['language'] = '__detected_language__'

        languages_supported = set(itertools.chain.from_iterable(model_language_map.values()))
        has_multilingual = any(len(langs) > 1 for langs in model_language_map.values())
        model_lang_str = ", ".join(sorted(languages_supported)) if languages_supported else "—"

        languages_detected_sorted = sorted(unique_languages)
        languages_without_model = sorted(lang for lang in unique_languages if lang not in languages_supported)
        languages_to_annotate = sorted(lang for lang in unique_languages if lang in languages_supported)

        language_series = self._language_assignments
        if language_series is None or len(language_series) != len(df):
            language_series = self._get_or_compute_row_languages(
                df,
                column_mapping,
                {
                    'language_column': language_column,
                    'detection_source': detection_source,
                },
            )
        language_series = language_series.astype(str).str.upper()
        self._language_assignments = language_series

        language_mask = (
            language_series.isin(languages_to_annotate)
            if languages_to_annotate
            else pd.Series([True] * len(language_series), index=language_series.index)
        )
        eligible_count = int(language_mask.sum())
        skipped_count = len(language_series) - eligible_count

        if languages_detected_sorted:
            self.console.print("\n[bold]📊 Language Distribution & Model Coverage[/bold]")
            self.console.print("[dim]Each model will only annotate rows in its supported language(s).[/dim]\n")

            for lang in languages_detected_sorted:
                lang_count = language_counts.get(lang.lower(), 0)
                models_for_lang = language_to_models.get(lang, [])
                if models_for_lang:
                    formatted_models: List[str] = []
                    for mdl in models_for_lang:
                        display_name = self._condense_relative_name(
                            mdl.get("relative_name")
                            or mdl.get("label_value")
                            or str(mdl.get("base_model") or "model")
                        )
                        per_metrics = mdl.get("metrics_per_language") or mdl.get("metrics", {}).get("per_language", {})
                        score = per_metrics.get(lang) if isinstance(per_metrics, dict) else None
                        if isinstance(score, (int, float)):
                            formatted_models.append(f"{display_name} (F1: {score:.3f})")
                        else:
                            formatted_models.append(display_name)
                    self.console.print(f"  [green]✓ {lang}[/green]: {lang_count:,} rows → will be annotated")
                    self.console.print(f"    [dim]Model: {', '.join(formatted_models)}[/dim]")
                else:
                    self.console.print(f"  [yellow]⊘ {lang}[/yellow]: {lang_count:,} rows → will be skipped (no compatible model)")

        self.console.print()
        if skipped_count > 0:
            self.console.print(
                f"[bold cyan]Summary:[/bold cyan] {eligible_count:,} rows will be annotated, "
                f"{skipped_count:,} rows will be skipped (no compatible model)."
            )
            self.console.print(
                f"[dim]This is expected when your dataset contains multiple languages but your models only support some of them.[/dim]"
            )
        else:
            self.console.print(f"[green]✓ All {eligible_count:,} rows will be annotated (full language coverage).[/green]")

        if eligible_count == 0:
            self.console.print(
                "[red]✗ No rows remain after applying language compatibility filters. "
                "Select another model or adjust your dataset.[/red]"
            )
            return None

        self._language_annotation_mask = language_mask
        self._allowed_annotation_languages = languages_to_annotate

        try:
            self._display_annotation_examples(
                df,
                column_mapping,
                language_series,
                language_mask,
                languages_to_annotate,
            )
        except Exception as exc:
            self.logger.debug("Unable to display annotation examples: %s", exc)

        # Show language compatibility summary
        if len(unique_languages) > 1:
            if has_multilingual:
                self.console.print(f"\n[green]✓ Dataset contains {len(unique_languages)} languages. Your model(s) support multiple languages.[/green]")
            else:
                covered_pct = (eligible_count / len(df)) * 100 if len(df) > 0 else 0
                self.console.print(f"\n[cyan]ℹ Dataset contains {len(unique_languages)} languages (Primary: {primary_lang}).[/cyan]")
                self.console.print(f"[cyan]  Model language(s): {model_lang_str}[/cyan]")
                self.console.print(f"[cyan]  Coverage: {covered_pct:.1f}% of rows will be annotated ({eligible_count:,}/{len(df):,}).[/cyan]")

                if primary_lang not in languages_supported:
                    self.console.print(
                        f"\n[yellow]⚠ Note: Your primary language ({primary_lang}) doesn't match the model language(s) ({model_lang_str}).[/yellow]"
                    )
                    self.console.print(
                        f"[yellow]  Only rows in {model_lang_str} will receive annotations. This is normal for multilingual datasets.[/yellow]"
                    )
                    if language_column and len(unique_languages) > 1:
                        self.console.print(
                            f"[dim]  Tip: You can filter your dataset by the '{language_column}' column before annotation if you prefer.[/dim]"
                        )
                    if not Confirm.ask("\nProceed with annotation?", default=True):
                        return None
        else:
            # Single language case
            is_compatible = primary_lang in languages_supported or has_multilingual
            if is_compatible:
                self.console.print(f"\n[green]✓ Language compatibility confirmed ({primary_lang}). All rows will be annotated.[/green]")
            else:
                self.console.print(
                    f"\n[red]✗ Language mismatch: Dataset is in {primary_lang}, but model only supports {model_lang_str}.[/red]"
                )
                if not Confirm.ask("Proceed anyway?", default=False):
                    return None

        if column_mapping.get('language') == '__detected_language__':
            try:
                df['__detected_language__'] = language_series.reindex(df.index)
            except Exception:
                df['__detected_language__'] = language_series.values

        language_info = {
            'primary_language': primary_lang,
            'languages': languages_detected_sorted,
            'counts': {lang: int(language_counts.get(lang, 0)) for lang in languages_detected_sorted},
            'language_column': language_column,
            'detection_source': detection_source,
            'model_languages': model_language_map,
            'language_to_models': {
                lang: [
                    mdl.get("relative_name")
                    or str(mdl.get("label_value") or mdl.get("base_model") or "model")
                    for mdl in language_to_models.get(lang, [])
                ]
                for lang in languages_detected_sorted
            },
            'languages_supported': sorted(languages_supported),
            'languages_to_annotate': languages_to_annotate,
            'languages_without_model': languages_without_model,
            'eligible_row_count': eligible_count,
            'skipped_row_count': skipped_count,
            'total_row_count': int(len(df)),
        }
        return language_info

    def _configure_annotation_options(
        self,
        plan: List[Dict[str, Any]],
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        resources = detect_resources()
        gpu_available = resources.gpu.available
        recommendations = resources.get_recommendation()
        recommended_batch = max(8, recommendations.get('batch_size', 16))
        language_mask = getattr(self, "_language_annotation_mask", None)
        usable_df = df
        if language_mask is not None and len(language_mask) == len(df):
            eligible_rows = int(language_mask.sum())
            if eligible_rows <= 0:
                self.console.print("[red]✗ No eligible rows remain for annotation after language filtering.[/red]")
                raise ValueError("Language filter removed all rows")
            if eligible_rows < len(df):
                skipped = len(df) - eligible_rows
                self.console.print(
                    f"[dim]Language filter active: {eligible_rows:,}/{len(df):,} rows eligible "
                    f"({skipped:,} skipped).[/dim]"
                )
            usable_df = df[language_mask]
        total_rows = len(usable_df)

        self.console.print("\n[cyan]Tune how inference workers run and decide how much of the dataset to annotate.[/cyan]\n")

        resource_table = Table(title="Detected Resources", box=box.ROUNDED)
        resource_table.add_column("Component", style="cyan", width=16)
        resource_table.add_column("Details", style="green", overflow="fold")
        resource_table.add_row("GPU", "Available" if gpu_available else "CPU only")
        if gpu_available:
            resource_table.add_row("GPU Type", ", ".join(resources.gpu.device_names) or resources.gpu.device_type.upper())
            resource_table.add_row("GPU Memory", f"{resources.gpu.total_memory_gb:.1f} GB")
        resource_table.add_row("CPU", f"{resources.cpu.physical_cores} cores / {resources.cpu.logical_cores} threads")
        resource_table.add_row("RAM Available", f"{resources.memory.available_gb:.1f} GB / {resources.memory.total_gb:.1f} GB")
        self._print_table(resource_table)

        annotation_config: Dict[str, Any] = {}
        while True:
            strategy_table = Table(title="Parallelisation Strategies", box=box.ROUNDED)
            strategy_table.add_column("#", style="cyan", width=4)
            strategy_table.add_column("Strategy", style="green", width=26)
            strategy_table.add_column("Description", style="magenta", overflow="fold")
            strategy_table.add_row("1", "Auto (recommended)", "Balance CPU and GPU workers automatically using detected hardware.")
            if gpu_available:
                strategy_table.add_row("2", "GPU only", "Force workloads onto the GPU for maximum throughput.")
            strategy_table.add_row("3", "CPU only", "Run on CPU workers only, ideal for CPU-only servers.")
            strategy_table.add_row("4", "Manual", "Specify device mode, worker counts, and batch sizes yourself.")
            self._print_table(strategy_table)

            valid_choices = ["1", "3", "4"] if not gpu_available else ["1", "2", "3", "4"]
            strategy_choice = Prompt.ask("[cyan]Select parallelisation strategy[/cyan]", choices=valid_choices, default="1")

            config: Dict[str, Any] = {}
            if strategy_choice == "1":
                config['parallel'] = True
                if gpu_available:
                    config['device_mode'] = 'both'
                    config['batch_size_gpu'] = max(32, recommended_batch)
                    config['batch_size_cpu'] = max(8, recommended_batch // 2)
                else:
                    config['device_mode'] = 'cpu'
                    config['batch_size_cpu'] = max(8, recommended_batch)
                    config['batch_size_gpu'] = config['batch_size_cpu']
                base = config['batch_size_gpu'] if gpu_available else config['batch_size_cpu']
                config['chunk_size'] = max(256, base * 8)
            elif strategy_choice == "2":
                config['parallel'] = True
                config['device_mode'] = 'gpu'
                config['batch_size_gpu'] = max(32, recommended_batch)
                config['batch_size_cpu'] = max(8, recommended_batch // 2)
                config['chunk_size'] = max(256, config['batch_size_gpu'] * 8)
            elif strategy_choice == "3":
                config['parallel'] = True
                config['device_mode'] = 'cpu'
                config['batch_size_cpu'] = max(8, recommended_batch)
                config['batch_size_gpu'] = config['batch_size_cpu']
                config['chunk_size'] = max(256, config['batch_size_cpu'] * 6)
            else:
                device_mode_choices = ["cpu"]
                if gpu_available:
                    device_mode_choices.extend(["gpu", "both"])
                device_mode = Prompt.ask("[cyan]Device mode[/cyan]", choices=device_mode_choices, default="both" if gpu_available else "cpu")
                parallel = Confirm.ask("Enable multiprocessing?", default=True)
                batch_size_cpu = IntPrompt.ask("CPU batch size", default=32)
                batch_size_gpu = batch_size_cpu
                if device_mode in {"gpu", "both"}:
                    batch_size_gpu = IntPrompt.ask("GPU batch size", default=64)
                default_chunk = batch_size_gpu * 8 if device_mode in {"gpu", "both"} else batch_size_cpu * 8
                chunk_size = IntPrompt.ask("Chunk size (texts per job)", default=max(128, default_chunk))
                config.update({
                    'device_mode': device_mode,
                    'parallel': parallel,
                    'batch_size_cpu': batch_size_cpu,
                    'batch_size_gpu': batch_size_gpu,
                    'chunk_size': max(64, chunk_size),
                })

            config.setdefault('batch_size_cpu', recommended_batch)
            config.setdefault('batch_size_gpu', max(32, recommended_batch))

            if self._confirm_parallel_config(config, resources):
                annotation_config = config
                break

            self.console.print("[yellow]Reconfiguration requested by user.[/yellow]\n")

        self.console.print("\n[bold magenta]Dataset Coverage[/bold magenta]")
        self.console.print(f"[dim]Rows detected: {total_rows:,}[/dim]")
        self.console.print("[dim]Choose how much of the dataset you want to annotate in this run.[/dim]\n")

        coverage_table = Table(box=box.ROUNDED)
        coverage_table.add_column("#", style="cyan", width=4)
        coverage_table.add_column("Mode", style="green", width=18)
        coverage_table.add_column("When to use it", style="magenta", overflow="fold")
        coverage_table.add_row(
            "1",
            "Full dataset",
            "Annotate every available row. Ideal once the pipeline is tuned.",
        )
        coverage_table.add_row(
            "2",
            "First rows",
            "Annotate the top chunk only. Great for smoke tests or validating column mapping.",
        )
        coverage_table.add_row(
            "3",
            "Random sample",
            "Annotate a shuffled subset to estimate quality before scaling to the entire dataset.",
        )
        self._print_table(coverage_table)

        coverage_choice = Prompt.ask(
            "[cyan]Coverage mode (1=full, 2=head, 3=random)[/cyan]",
            choices=["1", "2", "3"],
            default="1",
        )

        if coverage_choice == "2":
            default_head = min(1000, max(1, total_rows))
            head_size = IntPrompt.ask("How many top rows to annotate?", default=default_head, show_default=True)
            head_size = max(1, min(head_size, total_rows))
            scope = {"type": "head", "size": head_size}
        elif coverage_choice == "3":
            default_sample = min(5000, max(1, total_rows))
            sample_size = IntPrompt.ask("Random sample size", default=default_sample, show_default=True)
            sample_size = max(1, min(sample_size, total_rows))
            seed = IntPrompt.ask("Random seed", default=42)
            scope = {"type": "random", "size": sample_size, "seed": seed}
        else:
            scope = {"type": "full"}

        annotation_config['scope'] = scope
        annotation_config.setdefault('disable_tqdm', False)
        annotation_config['show_progress'] = True
        annotation_config['eligible_rows'] = total_rows
        return annotation_config

    def _configure_export_options(self) -> Dict[str, Any]:
        """Gather export preferences for annotated outputs."""
        default_root = Path("logs") / "annotation_studio"
        default_root.mkdir(parents=True, exist_ok=True)

        session_folder = self.session_id or "annotation_session"
        default_output_dir = default_root / session_folder
        default_output_dir.mkdir(parents=True, exist_ok=True)
        default_filename = default_output_dir / "annotations.csv"

        self.console.print("\n[bold magenta]Export Configuration[/bold magenta]")
        self.console.print("[dim]Decide where the annotated dataset should be written and which extras to include.[/dim]\n")

        format_table = Table(box=box.ROUNDED)
        format_table.add_column("#", style="cyan", width=4)
        format_table.add_column("Format", style="green", width=12)
        format_table.add_column("When it shines", style="magenta", overflow="fold")
        format_table.add_row("1", "CSV", "Universal compatibility with spreadsheets and BI tools.")
        format_table.add_row("2", "JSONL", "Great for incremental ingestion or downstream ML pipelines.")
        format_table.add_row("3", "Parquet", "Columnar format for large datasets and analytics engines.")
        self._print_table(format_table)

        format_choice = Prompt.ask(
            "[cyan]Select export format[/cyan]",
            choices=["1", "2", "3"],
            default="1",
        )
        format_map = {"1": "csv", "2": "jsonl", "3": "parquet"}
        output_format = format_map[format_choice]

        output_path = Prompt.ask(
            "[cyan]Output file path[/cyan]",
            default=str(default_filename.with_suffix(f".{output_format}")),
        ).strip()
        output_path = str(Path(output_path).expanduser())

        include_probabilities = Confirm.ask(
            "[cyan]Include prediction probabilities?[/cyan]",
            default=True,
        )
        include_metadata = True
        self.console.print("[dim]Session metadata will be saved automatically for reproducibility and resumes.[/dim]")
        archive_session = Confirm.ask(
            "[cyan]Create a zipped archive of the export folder for sharing?[/cyan]",
            default=False,
        )

        return {
            "output_format": output_format,
            "output_path": output_path,
            "include_probabilities": include_probabilities,
            "include_metadata": include_metadata,
            "archive_session": archive_session,
        }

    def _confirm_and_execute(
        self,
        pipeline_plan: List[Dict[str, Any]],
        data_source: Dict[str, Any],
        df: pd.DataFrame,
        column_mapping: Dict[str, Any],
        language_info: Dict[str, Any],
        annotation_config: Dict[str, Any],
        export_config: Dict[str, Any],
    ) -> bool:
        """Final confirmation prompt followed by the actual annotation run."""
        try:
            if not pipeline_plan:
                if self.console:
                    self.console.print("[red]✗ No models configured for annotation.[/red]")
                return False

            text_column = column_mapping.get("text")
            if not text_column or text_column not in df.columns:
                if self.console:
                    self.console.print("[red]✗ Text column is missing or invalid.[/red]")
                return False

            if self.console:
                summary_table = Table(title="Annotation Run Summary", box=box.ROUNDED)
                summary_table.add_column("Section", style="cyan", width=18)
                summary_table.add_column("Details", style="white", overflow="fold")

                data_desc = data_source.get("path") or data_source.get("display_name") or data_source.get("type", "dataset")
                summary_table.add_row("Dataset", str(data_desc))
                summary_table.add_row("Rows (post-language)", f"{len(df):,}")

                detected_langs = ", ".join(language_info.get("languages", [])) if language_info else "—"
                summary_table.add_row("Detected languages", detected_langs)

                model_names = ", ".join(
                    self._condense_relative_name(entry["info"].get("relative_name", "model"))
                    for entry in pipeline_plan
                )
                summary_table.add_row("Models", model_names or "—")
                summary_table.add_row("Output", str(export_config.get("output_path")))

                self._print_table(summary_table)
                self.console.print("[dim]Session metadata will be saved automatically for reproducibility.[/dim]")

                if not Confirm.ask("\n[bold yellow]Review complete. Launch annotation now?[/bold yellow]", default=True):
                    self.console.print("[yellow]Annotation cancelled by user.[/yellow]")
                    return False

            scope_cfg = annotation_config.get("scope") or {"type": "full"}
            working_df = self._apply_dataset_scope(df, scope_cfg)
            if working_df.empty:
                if self.console:
                    self.console.print("[red]✗ No rows selected after applying dataset coverage.[/red]")
                return False

            if len(working_df) != len(df) and self.console:
                self.console.print(
                    f"[dim]Dataset coverage applied: {len(working_df):,}/{len(df):,} rows will be processed.[/dim]"
                )

            self._reorder_plan_with_children(pipeline_plan)

            progress = Progress(
                TextColumn("[dim]{task.description}[/dim]"),
                BarColumn(bar_width=None),
                SpinnerColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            ) if self.console else None

            run_stats: List[Dict[str, Any]] = []

            progress_manager = progress if progress else nullcontext()
            with progress_manager as progress_bar:
                active_progress = progress_bar if isinstance(progress_bar, Progress) else None

                for entry in pipeline_plan:
                    info = entry.get("info", {})
                    columns = entry.get("columns") or {}

                    for key, col_name in columns.items():
                        if col_name not in working_df.columns:
                            if key in {"label", "language"}:
                                working_df[col_name] = pd.Series([pd.NA] * len(working_df), dtype="object", index=working_df.index)
                            elif key == "annotated":
                                working_df[col_name] = False
                            else:
                                working_df[col_name] = np.nan

                    mask = self._resolve_entry_mask(entry, working_df)
                    eligible_index = mask[mask].index

                    run_record = {
                        "id": entry.get("id"),
                        "model": info.get("relative_name"),
                        "rows_scheduled": int(mask.sum()),
                        "rows_annotated": int(len(eligible_index)),
                    }

                    if len(eligible_index) == 0:
                        run_stats.append(run_record)
                        continue

                    task_id = None
                    if active_progress is not None:
                        display_name = self._condense_relative_name(info.get("relative_name", "model"))
                        task_id = active_progress.add_task(f"{display_name}", total=len(eligible_index))

                        def progress_handler(processed: int, device_tag: str, task=task_id) -> None:
                            active_progress.advance(task, processed)
                    else:
                        def progress_handler(processed: int, device_tag: str) -> None:
                            return

                    texts = working_df.loc[eligible_index, text_column].fillna("").astype(str).tolist()

                    model_path = info.get("path")
                    model_path_str = str(model_path) if model_path is not None else ""

                    probabilities = parallel_predict(
                        texts,
                        model_path_str,
                        lang=info.get("language", "EN"),
                        parallel=annotation_config.get("parallel", True),
                        device_mode=annotation_config.get("device_mode", "both"),
                        batch_size_cpu=annotation_config.get("batch_size_cpu", 32),
                        batch_size_gpu=annotation_config.get("batch_size_gpu", 64),
                        chunk_size=annotation_config.get("chunk_size", 1024),
                        show_progress=False,
                        progress_handler=progress_handler,
                    )

                    if active_progress is not None and task_id is not None:
                        active_progress.update(task_id, completed=len(eligible_index))

                    probs = np.asarray(probabilities)
                    if probs.ndim == 1:
                        probs = probs[:, None]

                    if probs.shape[1] == 0:
                        max_indices = np.zeros(len(probs), dtype=int)
                        max_scores = np.zeros(len(probs))
                    else:
                        max_indices = probs.argmax(axis=1)
                        max_scores = probs[np.arange(len(probs)), max_indices]

                    label_pairs = info.get("id2label_pairs") or []
                    label_lookup = {int(idx): str(label) for idx, label in label_pairs}
                    if not label_lookup:
                        label_lookup = {idx: f"Label {idx}" for idx in range(probs.shape[1])}

                    predicted_labels = [label_lookup.get(int(idx), f"Label {idx}") for idx in max_indices]

                    if columns.get("label"):
                        working_df.loc[eligible_index, columns["label"]] = predicted_labels
                    if columns.get("label_id"):
                        working_df.loc[eligible_index, columns["label_id"]] = max_indices.astype(int)
                    if columns.get("probability"):
                        working_df.loc[eligible_index, columns["probability"]] = max_scores
                    if columns.get("ci_lower"):
                        working_df.loc[eligible_index, columns["ci_lower"]] = np.nan
                    if columns.get("ci_upper"):
                        working_df.loc[eligible_index, columns["ci_upper"]] = np.nan
                    if columns.get("language"):
                        working_df.loc[eligible_index, columns["language"]] = info.get("language", "")
                    if columns.get("annotated"):
                        working_df.loc[eligible_index, columns["annotated"]] = True

                    run_stats.append(run_record)

            export_df = working_df.copy()
            if not export_config.get("include_probabilities", True):
                drop_suffixes = ("_probability", "_ci_lower", "_ci_upper")
                drop_columns = [
                    col
                    for col in export_df.columns
                    for suffix in drop_suffixes
                    if col.endswith(suffix)
                ]
                if drop_columns:
                    export_df = export_df.drop(columns=drop_columns)

            output_path = Path(export_config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_format = export_config.get("output_format", "csv").lower()

            if output_format == "csv":
                export_df.to_csv(output_path, index=False)
            elif output_format == "jsonl":
                export_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
            elif output_format == "parquet":
                export_df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {output_format}")

            annotated_columns = [
                entry.get("columns", {}).get("annotated")
                for entry in pipeline_plan
                if entry.get("columns", {}).get("annotated") in working_df.columns
            ]
            annotated_total = 0
            if annotated_columns:
                combined_mask = working_df[annotated_columns[0]].fillna(False).astype(bool)
                for col_name in annotated_columns[1:]:
                    combined_mask = combined_mask | working_df[col_name].fillna(False).astype(bool)
                annotated_total = int(combined_mask.sum())

            models_meta = []
            for entry, stats in zip(pipeline_plan, run_stats):
                info = entry.get("info", {})
                path_value = info.get("path")
                models_meta.append(
                    {
                        "id": entry.get("id"),
                        "relative_name": info.get("relative_name"),
                        "path": str(path_value) if path_value is not None else None,
                        "language": info.get("language"),
                        "scope": entry.get("scope"),
                        "columns": entry.get("columns"),
                        "rows_scheduled": stats.get("rows_scheduled"),
                        "rows_annotated": stats.get("rows_annotated"),
                    }
                )

            metadata_payload = {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "data_source": data_source,
                "column_mapping": column_mapping,
                "language": language_info,
                "annotation_config": annotation_config,
                "export": {
                    "path": str(output_path),
                    "format": output_format,
                    "include_probabilities": export_config.get("include_probabilities", True),
                    "archive_session": export_config.get("archive_session", False),
                },
                "scoped_row_count": int(len(working_df)),
                "annotated_row_estimate": annotated_total,
                "models": models_meta,
            }

            metadata_path = self._persist_run_metadata(metadata_payload, output_path.parent)

            archive_path = None
            if export_config.get("archive_session"):
                archive_path = shutil.make_archive(str(output_path.parent), "zip", root_dir=output_path.parent)

            if self.console:
                self.console.print("\n[bold green]✓ Annotation completed successfully.[/bold green]")
                self.console.print(f"[cyan]Output file:[/cyan] {output_path}")
                if annotated_total:
                    self.console.print(f"[dim]{annotated_total:,} rows annotated across the configured pipeline.[/dim]")
                self.console.print(f"[dim]Metadata stored at {metadata_path}[/dim]")
                if archive_path:
                    self.console.print(f"[dim]Archive created at {archive_path}[/dim]")

            return True

        except Exception as exc:
            self.logger.exception("Annotation execution failed")
            if self.console:
                self.console.print(f"[bold red]✗ Error during annotation:[/bold red] {exc}", markup=True)
            return False

    def run_factory_pipeline(
        self,
        *,
        ordered_model_paths: Optional[Iterable[Union[str, Path]]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        text_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Launch the interactive studio while constraining selectable models for Annotator Factory."""
        resolved_paths: List[Path] = []
        allowed: Optional[Set[Path]] = None
        if ordered_model_paths:
            allowed = set()
            for candidate in ordered_model_paths:
                try:
                    resolved = Path(candidate).expanduser().resolve()
                except Exception:
                    resolved = Path(candidate).expanduser()
                allowed.add(resolved)
                resolved_paths.append(resolved)
        self._allowed_model_paths = allowed

        self._factory_launch_config = {
            "dataset_path": Path(dataset_path).expanduser() if dataset_path else None,
            "text_column": text_column,
            "model_paths": resolved_paths,
        }
        self._factory_launch_active = True

        if self.console:
            self.console.print(
                "[cyan]Launching BERT Annotation Studio to complete the Deploy & Annotate stage.[/cyan]"
            )

        try:
            self.run()
        finally:
            self._allowed_model_paths = None
            self._factory_launch_active = False
            self._factory_launch_config = None

        return {
            "status": "completed",
            "detail": "Interactive annotation studio launched",
        }

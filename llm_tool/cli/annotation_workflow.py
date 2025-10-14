#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
annotation_workflow.py

MAIN OBJECTIVE:
---------------
Unify the interactive annotation workflow used by the Annotator and Annotator
Factory CLIs, covering directory scaffolding, step tracking, and resume logic.

Dependencies:
-------------
- copy
- datetime
- enum
- pathlib
- typing
- rich
- llm_tool.utils.language_detector
- llm_tool.utils.data_detector
- llm_tool.utils.session_summary

MAIN FEATURES:
--------------
1) Create mode-specific session directories and metadata scaffolds
2) Normalise column selections and dataset context across CLI prompts
3) Track resume steps through AnnotationResumeTracker with step status caching
4) Persist session progress to resume.json files via shared summary helpers
5) Render dataset and workflow previews with Rich-powered tables and prompts

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
try:
    from rich import box
except ImportError:  # pragma: no cover - optional styling
    box = None

from ..utils.language_detector import LanguageDetector
from llm_tool.utils.data_detector import DataDetector
from llm_tool.utils.session_summary import SessionSummary, merge_summary, read_summary
from llm_tool.utils.model_metrics import load_language_metrics, summarize_final_metrics


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _read_training_metadata(model_dir: Path) -> Dict[str, Any]:
    """Load ``training_metadata.json`` if available."""
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _read_model_config(model_dir: Path) -> Dict[str, Any]:
    """Load ``config.json`` from a model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _format_per_class_scores(
    scores: List[Tuple[str, Optional[float]]],
    label_names: Optional[List[str]] = None,
    limit: int = 4,
) -> str:
    """Format per-class F1 scores with optional label remapping."""
    if not scores:
        return "—"
    resolved_scores: List[Tuple[str, Optional[float]]] = []
    for label, score in scores:
        resolved_label = str(label)
        idx: Optional[int] = None
        if isinstance(label, int):
            idx = label
        elif isinstance(label, str):
            token = label.strip().upper()
            if token.startswith("LABEL_") and token[6:].isdigit():
                idx = int(token[6:])
            elif token.isdigit():
                idx = int(token)
        if idx is not None and label_names and 0 <= idx < len(label_names):
            resolved_label = str(label_names[idx])
        resolved_scores.append((resolved_label, score))

    lines: List[str] = []
    for resolved_label, score in resolved_scores[:limit]:
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        lines.append(f"{resolved_label}: {score_text}")
    remaining = len(resolved_scores) - limit
    if remaining > 0:
        lines.append(f"... +{remaining} more")
    return "\n".join(lines)


def _format_language_scores(
    scores: Dict[str, Optional[float]],
    limit: int = 4,
) -> str:
    """Format per-language F1 scores."""
    if not scores:
        return "—"
    items = sorted(scores.items(), key=lambda item: item[0])
    lines: List[str] = []
    for lang, score in items[:limit]:
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        lines.append(f"{str(lang).upper()}: {score_text}")
    remaining = len(items) - limit
    if remaining > 0:
        lines.append(f"... +{remaining} more")
    return "\n".join(lines)


def _collect_confirmed_languages(
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    performance: Dict[str, Any],
) -> List[str]:
    """Aggregate confirmed languages across metadata, config, and metrics."""
    candidates: List[str] = []
    for source in (
        metadata.get("confirmed_languages"),
        config.get("confirmed_languages"),
    ):
        if isinstance(source, list):
            candidates.extend(str(entry) for entry in source if entry is not None)
    for key in ("language", "language_code"):
        if metadata.get(key):
            candidates.append(str(metadata[key]))
        if config.get(key):
            candidates.append(str(config[key]))
    per_language = performance.get("per_language", {})
    if isinstance(per_language, dict):
        candidates.extend(per_language.keys())

    normalized: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        code = str(candidate).strip().upper()
        if not code:
            continue
        if code not in seen:
            seen.add(code)
            normalized.append(code)
    return normalized


def _prompt_openai_batch_mode(cli: Any, provider: str, context: str) -> bool:
    """Ask the user whether to run the current annotation in OpenAI Batch mode."""
    if provider != 'openai':
        return False

    cli.console.print("\n[bold cyan]OpenAI Batch Mode[/bold cyan]")
    cli.console.print(
        "[dim]Batch mode uploads every prompt/text pair as a single JSONL job that OpenAI executes asynchronously. "
        "It is ideal when you have hundreds or thousands of rows because OpenAI manages queuing, retries, and durable storage.[/dim]"
    )
    cli.console.print(
        "[dim]Why enable it? You avoid local rate-limit backoffs, the run can be left unattended, and the raw request/response files are archived in your OpenAI dashboard "
        "and under logs/openai_batches/ for auditing.[/dim]"
    )
    cli.console.print(
        "[dim]What to expect: turnaround time is longer (minutes to hours depending on queue size), you only receive results once the batch finishes, "
        "and progress updates rely on periodic polling. Plan accordingly if you need rapid iteration.[/dim]"
    )

    question = f"[bold yellow]Do you want to use the OpenAI Batch API for {context}?[/bold yellow]"
    use_batch = Confirm.ask(question, default=False)

    if use_batch:
        cli.console.print(
            f"[green]✓ Batch mode enabled. The workflow will prepare the batch job and wait for OpenAI to finish processing {context}.[/green]"
        )
    else:
        cli.console.print(f"[dim]Continuing with synchronous API calls for {context}.[/dim]")

    return use_batch


def _normalize_column_choice(
    user_input: Optional[str],
    all_columns: List[str],
    candidate_columns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Normalize a user-supplied column selection across the annotation workflows.

    Supports direct column names (case-sensitive or insensitive) and numeric
    indices that map to displayed column lists.
    """
    if user_input is None:
        return None

    choice = str(user_input).strip()
    if not choice:
        return None

    if choice in all_columns:
        return choice

    lower_map = {col.lower(): col for col in all_columns}
    lowered = choice.lower()
    if lowered in lower_map:
        return lower_map[lowered]

    return None


def _coerce_to_int(value: Any) -> Optional[int]:
    """Convert common numeric token formats to int, ignoring 'all' tokens."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.lower() == "all":
            return None
        token = token.replace(",", "").strip()
        if token.startswith("+"):
            token = token[1:]
        if token.isdigit():
            try:
                return int(token)
            except ValueError:
                return None
    return None

    if choice.isdigit():
        idx = int(choice)
        one_based_idx = idx - 1

        if candidate_columns and 0 <= one_based_idx < len(candidate_columns):
            return candidate_columns[one_based_idx]

        if 0 <= one_based_idx < len(all_columns):
            return all_columns[one_based_idx]

        if 0 <= idx < len(all_columns):
            return all_columns[idx]

    return None


class AnnotationMode(Enum):
    """Workflow contexts supported by the annotation wizard."""

    ANNOTATOR = "annotator"
    FACTORY = "annotator_factory"


def create_session_directories(mode: AnnotationMode, session_id: str) -> Dict[str, Path]:
    """Create organised directory structure for an annotation session."""

    if mode == AnnotationMode.FACTORY:
        base_dir = Path("logs") / "annotator_factory" / session_id
        dirs: Dict[str, Path] = {
            "base": base_dir,
            "session_root": base_dir,
            "annotated_data": base_dir / "annotated_data",
            "metadata": base_dir / "metadata",
            "validation_exports": base_dir / "validation_exports",
            "doccano": base_dir / "validation_exports" / "doccano",
            "labelstudio": base_dir / "validation_exports" / "labelstudio",
            "training_metrics": base_dir / "training_metrics",
            "training_data": base_dir / "training_data",
            "model_annotation": base_dir / "model_annotation",
            "openai_batches": base_dir / "openai_batches",
        }
    else:
        base_dir = Path("logs") / "annotator" / session_id
        dirs = {
            "base": base_dir,
            "session_root": base_dir,
            "annotated_data": base_dir / "annotated_data",
            "metadata": base_dir / "metadata",
            "validation_exports": base_dir / "validation_exports",
            "doccano": base_dir / "validation_exports" / "doccano",
            "labelstudio": base_dir / "validation_exports" / "labelstudio",
            "openai_batches": base_dir / "openai_batches",
        }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


ANNOTATOR_RESUME_STEPS = {
    1: {"key": "data_selection", "name": "Select dataset"},
    2: {"key": "column_mapping", "name": "Configure columns"},
    3: {"key": "model_selection", "name": "Choose model"},
    4: {"key": "prompt_configuration", "name": "Configure prompts"},
    5: {"key": "advanced_settings", "name": "Review advanced options"},
    6: {"key": "run_annotation", "name": "Execute annotation"},
    7: {"key": "post_actions", "name": "Exports & follow-up"},
}


FACTORY_RESUME_STEPS = {
    1: {"key": "data_selection", "name": "Select dataset"},
    2: {"key": "column_mapping", "name": "Configure columns"},
    3: {"key": "model_selection", "name": "Choose models"},
    4: {"key": "prompt_configuration", "name": "Configure prompts"},
    5: {"key": "advanced_settings", "name": "Review advanced options"},
    6: {"key": "run_annotation", "name": "Execute annotation"},
    7: {"key": "training_prep", "name": "Prepare training data"},
    8: {"key": "training_launch", "name": "Launch training arena"},
    9: {"key": "model_annotation", "name": "Deploy trained models"},
    10: {"key": "post_actions", "name": "Exports & reports"},
}

def _launch_model_annotation_stage(
    cli,
    *,
    session_id: Optional[str],
    session_dirs: Optional[Dict[str, Path]],
    training_results: Optional[Dict[str, Any]],
    prompt_configs: List[Dict[str, Any]],
    text_column: str,
    annotation_output: Optional[str],
    dataset_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Launch BERT Annotation Studio with freshly trained models."""
    from pathlib import Path
    from rich.align import Align
    from llm_tool.cli.banners import BANNERS, STEP_NUMBERS, STEP_LABEL
    from llm_tool.cli.bert_annotation_studio import BERTAnnotationStudio
    from llm_tool.utils.annotation_session_manager import AnnotationStudioSessionManager

    def _persist_factory_annotation_metadata(summary: Dict[str, Any]) -> Optional[Path]:
        """Persist model-annotation metadata payload into the factory session structure."""
        if not summary or not session_dirs:
            return None
        metadata_root = session_dirs.get("metadata")
        if not metadata_root:
            return None
        target_dir = Path(metadata_root) / "model_annotation"
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = target_dir / f"model_annotation_{timestamp}.json"
        try:
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
        except Exception:
            return None
        return metadata_path

    training_payload = (training_results or {}).get("training_result") or {}

    session_identifier: Optional[str] = None
    if isinstance(training_results, dict):
        session_identifier = training_results.get("session_id")
    if not session_identifier:
        session_identifier = session_id

    session_model_root: Optional[Path] = None
    if session_identifier:
        session_model_root = Path("models") / session_identifier

    model_candidates: List[Tuple[Optional[str], Union[str, Path]]] = []

    def _collect_model_candidates(data: Any, hint_name: Optional[str] = None) -> None:
        if isinstance(data, dict):
            # Direct mapping (name -> path)
            if data and all(isinstance(k, str) for k in data.keys()) and all(
                isinstance(v, (str, Path)) for v in data.values()
            ):
                for key, value in data.items():
                    model_candidates.append((key, value))
            # Dict with explicit path fields
            path_value = data.get("model_path") or data.get("path")
            if path_value:
                candidate_name = (
                    data.get("model_name")
                    or data.get("name")
                    or data.get("label_value")
                    or data.get("category_name")
                    or data.get("label_key")
                    or hint_name
                )
                model_candidates.append((candidate_name, path_value))
            for key, value in data.items():
                next_hint = hint_name
                if isinstance(key, str):
                    next_hint = key
                _collect_model_candidates(value, next_hint)
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                _collect_model_candidates(item, hint_name)

    _collect_model_candidates(training_payload)

    top_level_trained = training_results.get("trained_model_paths") if isinstance(training_results, dict) else {}
    if isinstance(top_level_trained, dict):
        for key, value in top_level_trained.items():
            model_candidates.append((key, value))

    aggregated_models: Dict[str, Path] = {}
    existing_models: List[Tuple[str, Path]] = []
    seen_paths: Set[Path] = set()

    def _resolve_model_dir(path_value: Union[str, Path]) -> Optional[Path]:
        try:
            candidate = Path(path_value).expanduser()
        except Exception:
            return None
        if candidate.is_file():
            candidate = candidate.parent
        candidate = candidate.resolve()
        if candidate.is_file():
            candidate = candidate.parent
        if (candidate / "config.json").exists():
            return candidate
        for sub_dir in ("model", "best_model", "checkpoint-best"):
            option = candidate / sub_dir
            if option.is_dir() and (option / "config.json").exists():
                return option.resolve()
        try:
            config_path = next(candidate.glob("**/config.json"))
        except StopIteration:
            return None
        except Exception:
            return None
        return config_path.parent.resolve()

    def _register_model(name_hint: Optional[str], path_value: Union[str, Path]) -> None:
        resolved_dir = _resolve_model_dir(path_value)
        if resolved_dir is None or not resolved_dir.exists():
            return
        if resolved_dir in seen_paths:
            return
        display_name = name_hint
        if session_model_root and session_model_root.exists():
            try:
                relative = resolved_dir.relative_to(session_model_root)
                display_name = relative.as_posix()
            except ValueError:
                pass
        if not display_name:
            display_name = resolved_dir.name
        seen_paths.add(resolved_dir)
        aggregated_models[str(display_name)] = resolved_dir
        existing_models.append((str(display_name), resolved_dir))

    for candidate_name, candidate_path in model_candidates:
        if not candidate_path:
            continue
        _register_model(candidate_name, candidate_path)

    if not existing_models and session_identifier:
        loader = getattr(cli, "_load_saved_factory_training_results", None)
        if callable(loader):
            try:
                recon = loader(
                    session_id=session_identifier,
                    session_dirs=session_dirs,
                    training_workflow={},
                )
            except Exception:  # pragma: no cover - defensive
                recon = None
            if recon:
                recon_models = recon.get("training_result", {}).get("trained_models") or {}
                for model_name, model_path in recon_models.items():
                    _register_model(model_name, model_path)

    if not existing_models and session_identifier:
        fallback_root = Path("models") / session_identifier / "normal_training"
        if fallback_root.exists():
            try:
                for config_path in fallback_root.glob("**/config.json"):
                    rel_name = config_path.parent.relative_to(fallback_root).as_posix()
                    _register_model(rel_name, config_path.parent)
            except Exception:
                pass

    factory_trained_map: Dict[str, Path] = {}
    if isinstance(training_results, dict):
        raw_trained = training_results.get("trained_model_paths")
        if not raw_trained:
            raw_trained = (training_results.get("training_result") or {}).get("trained_model_paths")
        if not raw_trained:
            raw_trained = (training_results.get("training_result") or {}).get("trained_models")
        if isinstance(raw_trained, dict):
            for name, value in raw_trained.items():
                resolved = _resolve_model_dir(value)
                if resolved is None:
                    continue
                factory_trained_map[str(name)] = resolved

    def _safe_resolve_path(candidate: Path) -> Path:
        try:
            return candidate.resolve()
        except Exception:
            return candidate

    if factory_trained_map:
        target_paths: List[Path] = []
        for path in factory_trained_map.values():
            target_paths.append(_safe_resolve_path(path))

        filtered_models: List[Tuple[str, Path]] = []
        for display_name, model_path in existing_models:
            resolved_model_path = _safe_resolve_path(model_path)
            if resolved_model_path in target_paths:
                filtered_models.append((display_name, model_path))

        ordered_models: List[Tuple[str, Path]] = []
        for model_name, target_path in factory_trained_map.items():
            resolved_target = _safe_resolve_path(target_path)

            matched_entry = next(
                (
                    (display_name, model_path)
                    for display_name, model_path in filtered_models
                    if _safe_resolve_path(model_path) == resolved_target
                ),
                None,
            )
            if matched_entry:
                ordered_models.append(matched_entry)
            else:
                ordered_models.append((model_name, target_path))

        existing_models = ordered_models
        aggregated_models = {
            name: path
            for name, path in aggregated_models.items()
            if any(
                _safe_resolve_path(path)
                == _safe_resolve_path(target)
                for target in factory_trained_map.values()
            )
        }

        trimmed_map = {
            name: str(path)
            for name, path in factory_trained_map.items()
        }
        if isinstance(training_results, dict):
            training_results["trained_model_paths"] = trimmed_map
            training_results.setdefault("training_result", {}).update(
                {
                    "trained_models": trimmed_map,
                    "trained_model_paths": trimmed_map,
                }
            )

    if training_results is not None:
        training_entry = training_results.setdefault("training_result", {})
        combined_map: Dict[str, str] = {}
        if isinstance(training_entry.get("trained_models"), dict):
            combined_map.update(
                {
                    str(k): str(Path(v).expanduser())
                    for k, v in training_entry["trained_models"].items()
                    if v
                }
            )
        combined_map.update({name: str(path) for name, path in aggregated_models.items()})
        if combined_map:
            training_entry["trained_models"] = combined_map
            training_entry["models_trained"] = list(combined_map.keys())
            training_results["trained_model_paths"] = combined_map

    if not existing_models:
        if cli.console:
            cli.console.print("\n[yellow]No trained models detected; skipping deployment annotation stage.[/yellow]")
        return {"status": "skipped", "detail": "No trained models available"}

    if cli.console:
        cli.console.print()
        for line in STEP_LABEL.split('\n'):
            cli.console.print(Align.center(f"[bold {BANNERS['deploy_and_annotate']['color']}]{line}[/bold {BANNERS['deploy_and_annotate']['color']}]"))
        for line in STEP_NUMBERS['3/3'].split('\n'):
            cli.console.print(Align.center(f"[bold {BANNERS['deploy_and_annotate']['color']}]{line}[/bold {BANNERS['deploy_and_annotate']['color']}]"))
        cli.console.print()
        for line in BANNERS['deploy_and_annotate']['ascii'].split('\n'):
            cli.console.print(Align.center(f"[bold {BANNERS['deploy_and_annotate']['color']}]{line}[/bold {BANNERS['deploy_and_annotate']['color']}]"))
        cli.console.print(Align.center(f"[{BANNERS['deploy_and_annotate']['color']}]{BANNERS['deploy_and_annotate']['tagline']}[/{BANNERS['deploy_and_annotate']['color']}]"))
        cli.console.print()

        table_box = box.ROUNDED if box else None
        model_table = Table(title="Models Trained In This Session", box=table_box, show_lines=False)
        model_table.add_column("#", style="cyan", justify="center", width=4)
        model_table.add_column("Model Identifier", style="green")
        model_table.add_column("Langs", style="magenta", overflow="fold", width=12)
        model_table.add_column("Macro F1", style="bright_white", justify="right", width=10)
        model_table.add_column("Class F1", style="white", overflow="fold")
        model_table.add_column("Lang F1", style="white", overflow="fold")
        model_table.add_column("Location", style="dim")

        for idx, (name, path) in enumerate(existing_models, 1):
            relative_display = path.as_posix()
            if session_model_root and session_model_root.exists():
                try:
                    relative_display = path.relative_to(session_model_root).as_posix()
                except Exception:
                    relative_display = path.name
            else:
                relative_display = path.name

            metadata = _read_training_metadata(path)
            config = _read_model_config(path)
            performance = summarize_final_metrics(metadata, load_language_metrics(path))

            label_names: List[str] = []
            metadata_labels = metadata.get("label_names")
            if isinstance(metadata_labels, list):
                label_names = [str(label) for label in metadata_labels]
            else:
                config_labels = config.get("id2label")
                if isinstance(config_labels, list):
                    label_names = [str(label) for label in config_labels]
                elif isinstance(config_labels, dict):
                    try:
                        label_names = [
                            str(name)
                            for _, name in sorted(config_labels.items(), key=lambda item: int(item[0]))
                        ]
                    except Exception:
                        label_names = [str(name) for name in config_labels.values()]

            class_scores = performance.get("per_class", [])
            languages_display = ", ".join(_collect_confirmed_languages(metadata, config, performance)) or "-"
            macro_value = performance.get("macro_f1")
            macro_text = f"{macro_value:.3f}" if isinstance(macro_value, (int, float)) else "—"
            class_text = _format_per_class_scores(class_scores, label_names=label_names)
            language_text = _format_language_scores(performance.get("per_language", {}))

            model_table.add_row(
                str(idx),
                name,
                languages_display,
                macro_text,
                class_text,
                language_text,
                relative_display,
            )

        cli.console.print(model_table)
        cli.console.print()

        proceed = Confirm.ask(
            "[cyan]Would you like to launch the BERT Annotation Studio now to run these models on a dataset of your choice?[/cyan]",
            default=True,
        )
        if not proceed:
            cli.console.print("[green]Skipping Deploy & Annotate stage.[/green]\n")
            return {
                "status": "skipped",
                "detail": "User chose not to launch annotation studio",
            }

    session_base_dir: Optional[Path] = None
    if session_dirs:
        base_candidate = session_dirs.get("model_annotation")
        if base_candidate:
            session_base_dir = Path(base_candidate)
    if session_base_dir is None and session_dirs and session_dirs.get("session_root"):
        session_base_dir = Path(session_dirs["session_root"]) / "model_annotation"
    if session_base_dir is None:
        fallback_root = Path("logs") / "annotator_factory"
        fallback_id = session_id or "factory_session"
        session_base_dir = fallback_root / fallback_id / "model_annotation"
    session_base_dir.mkdir(parents=True, exist_ok=True)

    default_slug = AnnotationStudioSessionManager.slugify(f"{session_id}_bert") if session_id else "bert_annotation"
    ordered_model_paths = [path for _, path in existing_models]

    existing_data_source: Optional[Dict[str, Any]] = None
    dataset_display: Optional[str] = None
    if session_dirs:
        metadata_root = session_dirs.get("metadata")
        if metadata_root:
            annotation_meta_dir = Path(metadata_root) / "model_annotation"
            if annotation_meta_dir.exists():
                metadata_files = sorted(annotation_meta_dir.glob("model_annotation_*.json"))
                if metadata_files:
                    try:
                        with metadata_files[-1].open("r", encoding="utf-8") as handle:
                            latest_summary = json.load(handle)
                            existing_data_source = latest_summary.get("data_source")
                    except Exception:
                        existing_data_source = None

    if dataset_path:
        dataset_display = str(dataset_path)
    elif existing_data_source:
        dataset_display = (
            existing_data_source.get("display_name")
            or existing_data_source.get("path")
            or existing_data_source.get("table")
            or existing_data_source.get("connection_string")
        )

    force_dataset_selection_flag = False
    forced_step_set: Set[str] = set()
    launch_dataset_path = dataset_path
    launch_text_column = text_column

    reuse_dataset = True
    if dataset_display:
        use_different_dataset = (
            Confirm.ask(
                "[bold yellow]Use a different dataset than[/bold yellow] "
                f"[cyan]{dataset_display}[/cyan] [bold yellow]for running the trained models?[/bold yellow]",
                default=False,
            )
            if cli.console
            else False
        )
        reuse_dataset = not use_different_dataset
    elif existing_data_source:
        use_different_dataset = (
            Confirm.ask(
                "[bold yellow]Use a different dataset than the one saved in this session for the trained models?[/bold yellow]",
                default=False,
            )
            if cli.console
            else False
        )
        reuse_dataset = not use_different_dataset
    else:
        reuse_dataset = False

    if not reuse_dataset:
        launch_dataset_path = None
        launch_text_column = None
        force_dataset_selection_flag = True
        forced_step_set.update({
            "select_dataset",
            "map_columns",
            "output_columns",
            "language_detection",
            "annotation_options",
            "export_options",
            "review_launch",
        })
    else:
        if cli.console:
            adjust_mapping = Confirm.ask(
                "[cyan]Change the text/ID column mapping before running the models?[/cyan]",
                default=False,
            )
        else:
            adjust_mapping = False
        if adjust_mapping:
            launch_text_column = None
            forced_step_set.update({
                "map_columns",
                "output_columns",
                "language_detection",
                "annotation_options",
                "export_options",
                "review_launch",
            })
        else:
            if cli.console:
                rename_outputs = Confirm.ask(
                    "[cyan]Rename the annotation output columns before exporting?[/cyan]",
                    default=False,
                )
            else:
                rename_outputs = False
            if rename_outputs:
                forced_step_set.update({"output_columns", "review_launch"})

        if cli.console:
            adjust_exports = Confirm.ask(
                "[cyan]Update export destinations or formats (e.g., SQL clone, CSV)?[/cyan]",
                default=False,
            )
        else:
            adjust_exports = False
        if adjust_exports:
            forced_step_set.update({"export_options", "review_launch"})

    factory_context = {
        "session_id": session_id,
        "training_results": training_results,
        "annotation_output": str(annotation_output) if annotation_output else None,
        "text_column": launch_text_column,
        "prompt_configs": prompt_configs,
    }
    if forced_step_set:
        factory_context["forced_steps"] = sorted(forced_step_set)
    if force_dataset_selection_flag:
        factory_context["force_dataset_selection"] = True

    factory_context["dataset_path"] = str(launch_dataset_path) if launch_dataset_path else None
    factory_context["model_paths"] = [str(path) for path in ordered_model_paths]

    studio = BERTAnnotationStudio(
        console=cli.console,
        settings=cli.settings,
        logger=cli.logger,
        session_base_dir=session_base_dir,
        allowed_model_paths=[path for _, path in existing_models],
        default_session_slug=default_slug,
        factory_session_id=session_id,
        factory_context=factory_context,
    )

    try:
        result = studio.run_factory_pipeline(
            ordered_model_paths=ordered_model_paths,
            dataset_path=launch_dataset_path,
            text_column=launch_text_column,
            force_dataset_selection=force_dataset_selection_flag,
            forced_steps=sorted(forced_step_set),
        )
        if isinstance(result, dict):
            session_summary = result.get("session_summary") or {}
            metadata_path = _persist_factory_annotation_metadata(session_summary)

            extra_payload: Dict[str, Any] = result.setdefault("extra", {})
            extra_payload.setdefault(
                "models_used",
                [str(path) for _, path in existing_models],
            )
            extra_payload.setdefault("annotation_session_dir", str(session_base_dir))

            if session_summary:
                data_source = session_summary.get("data_source") or {}
                if data_source:
                    extra_payload["data_source"] = data_source

                column_mapping = session_summary.get("column_mapping") or {}
                if column_mapping:
                    extra_payload["column_mapping"] = column_mapping
                    extra_payload["text_column"] = column_mapping.get("text")
                    extra_payload["id_column"] = column_mapping.get("id")

                output_plan = session_summary.get("output_plan") or []
                if output_plan:
                    extra_payload["output_plan"] = output_plan
                    annotation_columns: List[str] = []
                    for entry in output_plan:
                        columns = entry.get("columns", {})
                        if isinstance(columns, dict):
                            label_column = columns.get("label")
                            if isinstance(label_column, str):
                                annotation_columns.append(label_column)
                    if annotation_columns:
                        extra_payload["annotation_columns"] = annotation_columns

                studio_session_id = session_summary.get("session_id")
                if studio_session_id:
                    extra_payload["studio_session_id"] = studio_session_id

                studio_session_dir = session_summary.get("session_dir")
                if studio_session_dir:
                    extra_payload["studio_session_dir"] = studio_session_dir

            if metadata_path:
                extra_payload["metadata_file"] = str(metadata_path)

            return result

        return {
            "status": "completed",
            "detail": f"Model annotation executed with {len(existing_models)} model(s)",
            "extra": {
                "models_used": [str(path) for _, path in existing_models],
                "annotation_session_dir": str(session_base_dir),
            },
        }
    except KeyboardInterrupt:
        if cli.console:
            cli.console.print("\n[yellow]Model annotation cancelled by user.[/yellow]")
        return {"status": "cancelled", "detail": "User cancelled model annotation"}
    except Exception as exc:
        cli.logger.exception("Model annotation stage failed")
        if cli.console:
            cli.console.print(f"[red]Model annotation failed:[/red] {exc}")
        return {"status": "failed", "detail": str(exc)}


class AnnotationResumeTracker:
    """Helper to keep resume metadata consistent for annotation workflows."""

    def __init__(
        self,
        mode: AnnotationMode,
        session_id: str,
        session_dirs: Dict[str, Path],
        step_catalog: Dict[int, Dict[str, str]],
        session_name: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.session_id = session_id
        self.session_name = session_name or session_id
        self.summary_path = session_dirs["base"] / "resume.json"
        self.step_catalog = step_catalog
        self._step_status: Dict[int, str] = {no: "pending" for no in step_catalog}

        self._summary: Optional[SessionSummary] = read_summary(self.summary_path)
        if self._summary:
            stored_status = self._summary.extra.get("step_status", {})
            if isinstance(stored_status, dict):
                for key, value in stored_status.items():
                    try:
                        no = int(key)
                    except (ValueError, TypeError):
                        continue
                    if no in self._step_status:
                        self._step_status[no] = str(value)
        else:
            now = datetime.now().strftime(ISO_FORMAT)
            base_summary = SessionSummary(
                mode=self.mode.value,
                session_id=self.session_id,
                session_name=self.session_name,
                status="pending",
                created_at=now,
                updated_at=now,
                extra=self._build_extra_payload(completed_steps=0),
            )
            self._summary = merge_summary(self.summary_path, base_summary)

    def _build_extra_payload(self, *, completed_steps: int) -> Dict[str, Any]:
        return {
            "total_steps": len(self.step_catalog),
            "completed_steps": completed_steps,
            "step_status": {str(k): v for k, v in self._step_status.items()},
            "mode_label": self.mode.name,
        }

    def _completed_steps(self) -> int:
        return sum(1 for status in self._step_status.values() if status == "completed")

    def mark_step(
        self,
        step_no: int,
        *,
        status: str = "completed",
        detail: Optional[str] = None,
        overall_status: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step_no in self._step_status:
            self._step_status[step_no] = status
        for prior in sorted(self.step_catalog):
            if prior < step_no and self._step_status.get(prior) == "pending":
                self._step_status[prior] = "completed"
        step_info = self.step_catalog.get(step_no, {"key": f"step_{step_no}", "name": f"Step {step_no}"})
        timestamp = datetime.now().strftime(ISO_FORMAT)
        extras = self._build_extra_payload(completed_steps=self._completed_steps())
        if detail:
            extras["last_step_summary"] = detail
        if extra:
            extras.update(extra)
        summary = SessionSummary(
            mode=self.mode.value,
            session_id=self.session_id,
            session_name=self.session_name,
            status=overall_status or (self._summary.status if self._summary else "active"),
            created_at=self._summary.created_at if self._summary else timestamp,
            updated_at=timestamp,
            last_step_key=step_info["key"],
            last_step_name=step_info["name"],
            last_step_no=step_no,
            last_event_at=timestamp,
            extra=extras,
        )
        self._summary = merge_summary(self.summary_path, summary)

    def update_status(
        self,
        status: str,
        *,
        note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = datetime.now().strftime(ISO_FORMAT)
        extras = self._build_extra_payload(completed_steps=self._completed_steps())
        if note:
            extras["status_note"] = note
        if extra:
            extras.update(extra)
        summary = SessionSummary(
            mode=self.mode.value,
            session_id=self.session_id,
            session_name=self.session_name,
            status=status,
            created_at=self._summary.created_at if self._summary else timestamp,
            updated_at=timestamp,
            last_step_key=self._summary.last_step_key if self._summary else None,
            last_step_name=self._summary.last_step_name if self._summary else None,
            last_step_no=self._summary.last_step_no if self._summary else None,
            last_event_at=timestamp,
            extra=extras,
        )
        self._summary = merge_summary(self.summary_path, summary)


def run_annotator_workflow(cli, session_id: str = None, session_dirs: Optional[Dict[str, Path]] = None):
    """Smart guided annotation wizard with all options

    Parameters
    ----------
    session_id : str, optional
        Session identifier for organizing outputs. If None, a timestamp-based ID is generated.
    """
    import pandas as pd
    from datetime import datetime

    # Generate session_id if not provided (for backward compatibility)
    if session_id is None:
        session_id = f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create session directories
    if session_dirs is None:
        session_dirs = cli._create_annotator_session_directories(session_id)

    tracker = AnnotationResumeTracker(
        mode=AnnotationMode.ANNOTATOR,
        session_id=session_id,
        session_dirs=session_dirs,
        step_catalog=ANNOTATOR_RESUME_STEPS,
        session_name=session_id,
    )
    tracker.update_status("active")

    cli.console.print("\n[bold cyan]🎯 Smart Annotate - Guided Wizard[/bold cyan]\n")

    # Step 1: Data Selection
    cli.console.print("[bold]Step 1/7: Data Selection[/bold]")

    if not cli.detected_datasets:
        cli.console.print("[yellow]No datasets auto-detected.[/yellow]")
        data_path = Path(cli._prompt_file_path("Dataset path"))
    else:
        cli.console.print(f"\n[bold cyan]📊 Found {len(cli.detected_datasets)} dataset(s):[/bold cyan]\n")

        # Create table for datasets
        datasets_table = Table(border_style="cyan", show_header=True)
        datasets_table.add_column("#", style="bold yellow", width=4)
        datasets_table.add_column("Filename", style="white")
        datasets_table.add_column("Format", style="green", width=10)
        datasets_table.add_column("Size", style="magenta", width=10)
        datasets_table.add_column("Rows", style="cyan", width=10)
        datasets_table.add_column("Columns", style="blue", width=10)

        for i, ds in enumerate(cli.detected_datasets[:20], 1):
            # Format size
            if ds.size_mb < 0.1:
                size_str = f"{ds.size_mb * 1024:.1f} KB"
            else:
                size_str = f"{ds.size_mb:.1f} MB"

            # Format rows and columns
            rows_str = f"{ds.rows:,}" if ds.rows else "?"
            cols_str = str(len(ds.columns)) if ds.columns else "?"

            datasets_table.add_row(
                str(i),
                ds.path.name,
                ds.format.upper(),
                size_str,
                rows_str,
                cols_str
            )

        cli.console.print(datasets_table)
        cli.console.print()

        use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
        if use_detected:
            choice = cli._int_prompt_with_validation("Select dataset", 1, 1, len(cli.detected_datasets))
            data_path = cli.detected_datasets[choice - 1].path
        else:
            data_path = Path(cli._prompt_file_path("Dataset path"))

    # Detect format
    data_format = data_path.suffix[1:].lower()
    if data_format == 'xlsx':
        data_format = 'excel'

    cli.console.print(f"[green]✓ Selected: {data_path.name} ({data_format})[/green]")
    tracker.mark_step(
        1,
        detail=f"{data_path.name} ({data_format})",
        extra={"dataset_path": str(data_path)},
    )

    # Step 2: Text column selection with intelligent detection
    cli.console.print("\n[bold]Step 2/7: Text Column Selection[/bold]")

    # Detect text columns
    column_info = cli._detect_text_columns(data_path)

    candidate_names = [candidate['name'] for candidate in column_info.get('text_candidates', [])]
    all_columns = column_info.get('all_columns', [])

    if candidate_names:
        cli.console.print("\n[dim]Detected text columns (sorted by confidence):[/dim]")

        # Create table for candidates
        col_table = Table(border_style="blue")
        col_table.add_column("#", style="cyan", width=3)
        col_table.add_column("Column", style="white")
        col_table.add_column("Confidence", style="yellow")
        col_table.add_column("Avg Length", style="green")
        col_table.add_column("Sample", style="dim")

        for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
            # Color code confidence
            conf_color = {
                "high": "[green]High[/green]",
                "medium": "[yellow]Medium[/yellow]",
                "low": "[orange1]Low[/orange1]",
                "very_low": "[red]Very Low[/red]"
            }
            conf_display = conf_color.get(candidate['confidence'], candidate['confidence'])

            col_table.add_row(
                str(i),
                candidate['name'],
                conf_display,
                f"{candidate['avg_length']:.0f} chars",
                candidate['sample'][:50] + "..." if len(candidate['sample']) > 50 else candidate['sample']
            )

        cli.console.print(col_table)

        # Show all columns option
        if all_columns:
            cli.console.print(f"\n[dim]All columns ({len(all_columns)}): {', '.join(all_columns)}[/dim]")

        default_col = candidate_names[0]
        prompt_message = "\n[bold yellow]Enter column name[/bold yellow] (or choose from above)"
    else:
        # No candidates detected, show all columns
        if all_columns:
            cli.console.print(f"\n[yellow]Could not auto-detect text columns.[/yellow]")
            cli.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")
        default_col = "text"
        prompt_message = "Text column name"

    while True:
        raw_choice = Prompt.ask(prompt_message, default=default_col)
        normalized = _normalize_column_choice(raw_choice, all_columns, candidate_names or all_columns)

        if normalized:
            text_column = normalized
            break

        if not all_columns:
            text_column = raw_choice.strip()
            break

        cli.console.print(f"[red]✗ Column selection '{raw_choice}' could not be resolved.[/red]")
        if candidate_names:
            cli.console.print("[dim]Enter the column name or the number shown in the table.[/dim]")
        if all_columns:
            cli.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

    # Step 2b: ID Column Selection (MODERNIZED)
    # Load dataframe to detect ID candidates
    if data_format == 'csv':
        df_for_id = pd.read_csv(data_path, nrows=1000)
    elif data_format == 'json':
        df_for_id = pd.read_json(data_path, lines=False, nrows=1000)
    elif data_format == 'jsonl':
        df_for_id = pd.read_json(data_path, lines=True, nrows=1000)
    elif data_format == 'excel':
        df_for_id = pd.read_excel(data_path, nrows=1000)
    else:
        df_for_id = pd.read_csv(data_path, nrows=1000)  # Fallback

    # Use new unified ID selection function
    identifier_column = DataDetector.display_and_select_id_column(
        cli.console,
        df_for_id,
        text_column=text_column,
        step_label="Step 2b/7: Identifier Column Selection"
    )
    identifier_source = "user" if identifier_column else "auto"
    auto_identifier_column = identifier_column or "llm_annotation_id"
    pipeline_identifier_column = identifier_column if identifier_column else None
    identifier_column = pipeline_identifier_column
    tracker.mark_step(
        2,
        detail=f"text={text_column}, id={auto_identifier_column if identifier_source == 'auto' else identifier_column}",
        extra={
            "text_column": text_column,
            "identifier_column": auto_identifier_column,
            "identifier_source": identifier_source,
        },
    )

    # Step 3: Model Selection
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 3:[/bold cyan] [bold white]Model Selection[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Choose from local (Ollama) or cloud (OpenAI/Anthropic) models for annotation.[/dim]\n")

    selected_llm = cli._select_llm_interactive()
    provider = selected_llm.provider
    model_name = selected_llm.name
    tracker.mark_step(
        3,
        detail=f"{provider}:{model_name}",
        extra={"provider": provider, "model": model_name},
    )

    # Get API key if needed
    api_key = None
    if selected_llm.requires_api_key:
        api_key = cli._get_or_prompt_api_key(provider, model_name)

    openai_batch_mode = _prompt_openai_batch_mode(cli, provider, "this annotation run")

    # Step 4: Prompt Configuration
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 4:[/bold cyan] [bold white]Prompt Configuration[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Select from existing prompts or create new annotation instructions.[/dim]")

    # Auto-detect prompts
    detected_prompts = cli._detect_prompts_in_folder()

    if detected_prompts:
        cli.console.print(f"\n[green]✓ Found {len(detected_prompts)} prompts in prompts/ folder:[/green]")
        for i, p in enumerate(detected_prompts, 1):
            # Display ALL keys, not truncated
            keys_str = ', '.join(p['keys'])
            cli.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
            cli.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

        # Explain the options clearly
        cli.console.print("\n[bold]Prompt Selection Options:[/bold]")
        cli.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts (multi-prompt mode)")
        cli.console.print("           → Each text will be annotated with all prompts")
        cli.console.print("           → Useful when you want complete annotations from all perspectives")
        cli.console.print("\n  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number (e.g., 1,3,5)")
        cli.console.print("           → Only selected prompts will be used")
        cli.console.print("           → Useful when testing or when you need only certain annotations")
        cli.console.print("\n  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
        cli.console.print("           → Interactive guided prompt creation")
        cli.console.print("           → Optional AI assistance for definitions")
        cli.console.print("           → [bold green]Recommended for new research projects![/bold green]")
        cli.console.print("\n  [cyan]custom[/cyan]  - Provide path to a prompt file NOT in prompts/ folder")
        cli.console.print("           → Use a prompt from another location")
        cli.console.print("           → Useful for testing new prompts or one-off annotations")

        prompt_choice = Prompt.ask(
            "\n[bold yellow]Prompt selection[/bold yellow]",
            choices=["all", "select", "wizard", "custom"],
            default="all"
        )

        selected_prompts = []
        if prompt_choice == "all":
            selected_prompts = detected_prompts
            cli.console.print(f"[green]✓ Using all {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "select":
            indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
            if indices.strip():  # Only process if not empty
                for idx_str in indices.split(','):
                    idx_str = idx_str.strip()
                    if idx_str:  # Skip empty strings
                        try:
                            idx = int(idx_str) - 1
                            if 0 <= idx < len(detected_prompts):
                                selected_prompts.append(detected_prompts[idx])
                        except ValueError:
                            cli.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
            if not selected_prompts:
                cli.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                selected_prompts = detected_prompts
            else:
                cli.console.print(f"[green]✓ Selected {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            # Custom path
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]
    else:
        cli.console.print("[yellow]No prompts found in prompts/ folder[/yellow]")

        # Offer wizard or custom path
        cli.console.print("\n[bold]Prompt Options:[/bold]")
        cli.console.print("  [cyan]wizard[/cyan] - 🧙‍♂️ Create prompt using Social Science Wizard (Recommended)")
        cli.console.print("  [cyan]custom[/cyan] - Provide path to existing prompt file")

        choice = Prompt.ask(
            "\n[bold yellow]Select option[/bold yellow]",
            choices=["wizard", "custom"],
            default="wizard"
        )

        if choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]


    # Language detection moved to training phase
    # Language columns will be detected and handled automatically after annotation
    lang_column = None
    available_columns = column_info.get('all_columns', []) if column_info else []
    if available_columns:
        # Silently detect potential language columns for metadata
        potential_lang_cols = [col for col in available_columns
                              if col.lower() in ['lang', 'language', 'langue', 'lng', 'iso_lang']]

        # If language column exists, note it for later use but don't ask user
        if potential_lang_cols:
            lang_column = potential_lang_cols[0]  # Use first one if found
    # Multi-prompt prefix configuration (if needed)
    prompt_configs = []
    if len(selected_prompts) > 1:
        cli.console.print("\n[bold]Multi-Prompt Mode:[/bold] Configure key prefixes")
        cli.console.print("[dim]Prefixes help identify which prompt generated which keys[/dim]\n")

        for i, prompt in enumerate(selected_prompts, 1):
            cli.console.print(f"\n[cyan]Prompt {i}: {prompt['name']}[/cyan]")
            cli.console.print(f"  Keys: {', '.join(prompt['keys'])}")

            add_prefix = Confirm.ask(f"Add prefix to keys for this prompt?", default=True)
            prefix = ""
            if add_prefix:
                default_prefix = prompt['name'].lower().replace(' ', '_')
                prefix = Prompt.ask("Prefix", default=default_prefix)
                cli.console.print(f"  [green]Keys will become: {', '.join([f'{prefix}_{k}' for k in prompt['keys'][:3]])}[/green]")

            prompt_configs.append({
                'prompt': prompt,
                'prefix': prefix
            })
    else:
        # Single prompt - no prefix needed
        prompt_configs = [{'prompt': selected_prompts[0], 'prefix': ''}]

    tracker.mark_step(
        4,
        detail=f"{len(prompt_configs)} prompt(s)",
        extra={"prompt_count": len(prompt_configs)},
    )

    # Step 1.5: Advanced Options
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.5:[/bold cyan] [bold white]Advanced Options[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Configure processing settings for optimal performance.[/dim]")

    # ============================================================
    # DATASET SCOPE
    # ============================================================
    cli.console.print("\n[bold cyan]📊 Dataset Scope[/bold cyan]")
    cli.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

    # Get total rows if possible
    total_rows = None
    if column_info.get('df') is not None:
        # We have a sample, extrapolate
        total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

    if total_rows:
        cli.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

    # Option 1: Annotate all or limited
    cli.console.print("[yellow]Option 1:[/yellow] Annotate ALL rows vs LIMIT to specific number")
    cli.console.print("  • [cyan]all[/cyan]   - Annotate the entire dataset")
    cli.console.print("           [dim]Use this for production annotations[/dim]")
    cli.console.print("  • [cyan]limit[/cyan] - Specify exact number of rows to annotate")
    cli.console.print("           [dim]Use this for testing or partial annotation[/dim]")

    scope_choice = Prompt.ask(
        "\nAnnotate entire dataset or limit rows?",
        choices=["all", "limit"],
        default="all"
    )

    annotation_limit = None
    use_sample = False
    sample_strategy = "head"
    recommended_sample = None

    if scope_choice == "limit":
        # Option 2: FIRST ask about representative sample calculation (before asking for number)
        if total_rows and total_rows > 1000:
            cli.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
            cli.console.print("  Calculate statistically representative sample size (95% confidence interval)")
            cli.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

            calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

            if calculate_sample:
                # Formula: n = (Z² × p × (1-p)) / E²
                # For 95% CI: Z=1.96, p=0.5 (max variance), E=0.05 (5% margin)
                import math
                z = 1.96
                p = 0.5
                e = 0.05
                n_infinite = (z**2 * p * (1-p)) / (e**2)
                n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                recommended_sample = int(math.ceil(n_adjusted))

                cli.console.print(f"\n[green]📈 Recommended sample size: {recommended_sample} rows[/green]")
                cli.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")
                cli.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

        # THEN ask for specific number (with recommendation as default if calculated)
        default_limit = recommended_sample if recommended_sample else 100
        annotation_limit = cli._int_prompt_with_validation(
            f"How many rows to annotate?",
            default=default_limit,
            min_value=1,
            max_value=total_rows if total_rows else 1000000
        )

        # Check if user chose the recommended sample
        if recommended_sample and annotation_limit == recommended_sample:
            use_sample = True

        # Option 3: Random sampling
        cli.console.print("\n[yellow]Option 3:[/yellow] Sampling Strategy")
        cli.console.print("  Choose how to select the rows to annotate")
        cli.console.print("  • [cyan]head[/cyan]   - Take first N rows (faster, sequential)")
        cli.console.print("           [dim]Good for testing, preserves order[/dim]")
        cli.console.print("  • [cyan]random[/cyan] - Random sample of N rows (representative)")
        cli.console.print("           [dim]Better for statistical validity, unbiased[/dim]")

        sample_strategy = Prompt.ask(
            "\nSampling strategy",
            choices=["head", "random"],
            default="random" if use_sample else "head"
        )

    # ============================================================
    # PARALLEL PROCESSING
    # ============================================================
    if openai_batch_mode:
        cli.console.print("\n[bold cyan]⚙️  Processing[/bold cyan]")
        cli.console.print(
            "[dim]OpenAI Batch mode manages concurrency, retries, and persistence. "
            "Local parallelism and incremental save settings are skipped.[/dim]\n"
        )
        num_processes = 1
        save_incrementally = False
        batch_size = 1
    else:
        cli.console.print("\n[bold cyan]⚙️  Parallel Processing[/bold cyan]")
        cli.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

        cli.console.print("[yellow]Parallel Workers:[/yellow]")
        cli.console.print("  Number of simultaneous annotation processes")
        cli.console.print("\n  [red]⚠️  IMPORTANT:[/red]")
        cli.console.print("  [dim]Most local machines can only handle 1 worker for LLM inference[/dim]")
        cli.console.print("  [dim]Parallel processing is mainly useful for API models[/dim]")
        cli.console.print("\n  • [cyan]1 worker[/cyan]  - Sequential processing")
        cli.console.print("           [dim]Recommended for: Local models (Ollama), first time users, debugging[/dim]")
        cli.console.print("  • [cyan]2-4 workers[/cyan] - Moderate parallelism")
        cli.console.print("           [dim]Recommended for: API models (OpenAI, Claude) - avoid rate limits[/dim]")
        cli.console.print("  • [cyan]4-8 workers[/cyan] - High parallelism")
        cli.console.print("           [dim]Recommended for: API models only - requires high rate limits[/dim]")

        num_processes = cli._int_prompt_with_validation("Parallel workers", 1, 1, 16)

        cli.console.print("\n[bold cyan]💾 Incremental Save[/bold cyan]")
        cli.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

        cli.console.print("[yellow]Enable incremental save?[/yellow]")
        cli.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
        cli.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
        cli.console.print("  • [red]No[/red]  - Save only at the end")
        cli.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

        save_incrementally = Confirm.ask("\n💿 Enable incremental save?", default=True)

        if save_incrementally:
            cli.console.print("\n[yellow]Batch Size:[/yellow]")
            cli.console.print("  Number of rows processed between each save")
            cli.console.print("  • [cyan]Smaller (1-10)[/cyan]   - Very frequent saves, maximum safety")
            cli.console.print("           [dim]Use for: Unstable systems, expensive APIs, testing[/dim]")
            cli.console.print("  • [cyan]Medium (10-50)[/cyan]   - Balanced safety and performance")
            cli.console.print("           [dim]Use for: Most production cases[/dim]")
            cli.console.print("  • [cyan]Larger (50-200)[/cyan]  - Less frequent saves, better performance")
            cli.console.print("           [dim]Use for: Stable systems, large datasets, local models[/dim]")

            batch_size = cli._int_prompt_with_validation("Batch size", 1, 1, 1000)
        else:
            batch_size = None  # Not used when incremental save is disabled

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    cli.console.print("\n[bold cyan]🎛️  Model Parameters[/bold cyan]")
    cli.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

    # Check if model supports parameter tuning
    model_name_lower = model_name.lower()
    is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
    supports_params = not is_o_series

    if not supports_params:
        cli.console.print(f"[yellow]⚠️  Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
        cli.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")
        configure_params = False
    else:
        cli.console.print("[yellow]Configure model parameters?[/yellow]")
        cli.console.print("  Adjust how the model generates responses")
        cli.console.print("  [dim]• Default values work well for most cases[/dim]")
        cli.console.print("  [dim]• Advanced users can fine-tune for specific needs[/dim]")
        configure_params = Confirm.ask("\nConfigure model parameters?", default=False)

    # Default values
    temperature = 0.7
    max_tokens = 1000
    top_p = 1.0
    top_k = 40

    if configure_params:
        cli.console.print("\n[bold]Parameter Explanations:[/bold]\n")

        # Temperature
        cli.console.print("[cyan]🌡️  Temperature (0.0 - 2.0):[/cyan]")
        cli.console.print("  Controls randomness in responses")
        cli.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
        cli.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
        cli.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
        cli.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
        cli.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
        cli.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
        temperature = FloatPrompt.ask("Temperature", default=0.7)

        # Max tokens
        cli.console.print("\n[cyan]📏 Max Tokens:[/cyan]")
        cli.console.print("  Maximum length of the response")
        cli.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
        cli.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
        cli.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
        cli.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

        # Top_p (nucleus sampling)
        cli.console.print("\n[cyan]🎯 Top P (0.0 - 1.0):[/cyan]")
        cli.console.print("  Nucleus sampling - alternative to temperature")
        cli.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
        cli.console.print("           [dim]More deterministic, safer outputs[/dim]")
        cli.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
        cli.console.print("           [dim]More creative, diverse outputs[/dim]")
        cli.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
        top_p = FloatPrompt.ask("Top P", default=1.0)

        # Top_k (only for some models)
        if provider in ['ollama', 'google']:
            cli.console.print("\n[cyan]🔢 Top K:[/cyan]")
            cli.console.print("  Limits vocabulary to K most likely next tokens")
            cli.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
            cli.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
            cli.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
            top_k = cli._int_prompt_with_validation("Top K", 40, 1, 100)

    tracker.mark_step(
        5,
        detail="Advanced options configured",
    )

    # Step 1.6: Execute
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.6:[/bold cyan] [bold white]Review & Execute[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Review your configuration and start the annotation process.[/dim]")

    # Determine annotation mode once for display and downstream logic
    if provider == 'openai' and openai_batch_mode:
        annotation_mode = 'openai_batch'
        annotation_mode_display = "OpenAI Batch"
    elif provider in {'openai', 'anthropic', 'google'}:
        annotation_mode = 'api'
        annotation_mode_display = "API"
    else:
        annotation_mode = 'local'
        annotation_mode_display = "Local"

    # Display comprehensive summary
    summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True)
    summary_table.add_column("Category", style="bold cyan", width=20)
    summary_table.add_column("Setting", style="yellow", width=25)
    summary_table.add_column("Value", style="white")

    # Data section
    summary_table.add_row("📁 Data", "Dataset", str(data_path.name))
    summary_table.add_row("", "Format", data_format.upper())
    summary_table.add_row("", "Text Column", text_column)
    summary_table.add_row(
        "",
        "Identifier Column",
        f"{auto_identifier_column} (auto-generated)" if identifier_source == "auto" else auto_identifier_column
    )
    if total_rows:
        summary_table.add_row("", "Total Rows", f"{total_rows:,}")
    if annotation_limit:
        summary_table.add_row("", "Rows to Annotate", f"{annotation_limit:,} ({sample_strategy})")
    else:
        summary_table.add_row("", "Rows to Annotate", "ALL")

    # Model section
    summary_table.add_row("🤖 Model", "Provider/Model", f"{provider}/{model_name}")
    summary_table.add_row("", "Temperature", f"{temperature}")
    summary_table.add_row("", "Max Tokens", f"{max_tokens}")
    if configure_params:
        summary_table.add_row("", "Top P", f"{top_p}")
        if provider in ['ollama', 'google']:
            summary_table.add_row("", "Top K", f"{top_k}")

    # Prompts section
    summary_table.add_row("📝 Prompts", "Count", f"{len(prompt_configs)}")
    for i, pc in enumerate(prompt_configs, 1):
        prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
        summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

    # Processing section
    if openai_batch_mode:
        summary_table.add_row(
            "⚙️  Processing",
            "Annotation Mode",
            "OpenAI Batch (OpenAI-managed async job)"
        )
        summary_table.add_row("", "Parallel Workers", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Batch Size", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Incremental Save", "N/A (handled after batch completion)")
    else:
        summary_table.add_row("⚙️  Processing", "Annotation Mode", annotation_mode_display)
        summary_table.add_row("", "Parallel Workers", str(num_processes))
        summary_table.add_row("", "Batch Size", str(batch_size))
        summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

    cli.console.print("\n")
    cli.console.print(summary_table)

    if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
        return

    # ============================================================
    # REPRODUCIBILITY METADATA
    # ============================================================
    cli.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
    cli.console.print("[green]✓ Session parameters are automatically saved for:[/green]\n")

    cli.console.print("  [green]1. Resume Capability[/green]")
    cli.console.print("     • Continue this annotation if it stops or crashes")
    cli.console.print("     • Annotate additional rows later with same settings")
    cli.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

    cli.console.print("  [green]2. Scientific Reproducibility[/green]")
    cli.console.print("     • Document exact parameters for research papers")
    cli.console.print("     • Reproduce identical annotations in the future")
    cli.console.print("     • Track model version, prompts, and all settings\n")

    # Metadata is ALWAYS saved automatically for reproducibility
    save_metadata = True

    # ============================================================
    # VALIDATION TOOL EXPORT OPTION
    # ============================================================
    cli.console.print("\n[bold cyan]📤 Validation Tool Export[/bold cyan]")
    cli.console.print("[dim]Export annotations to JSONL format for human validation[/dim]\n")

    cli.console.print("[yellow]Available validation tools:[/yellow]")
    cli.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
    cli.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
    cli.console.print("  • Both are open-source and free\n")

    cli.console.print("[green]Why validate with external tools?[/green]")
    cli.console.print("  • Review and correct LLM annotations")
    cli.console.print("  • Calculate inter-annotator agreement")
    cli.console.print("  • Export validated data for metrics calculation\n")

    # Initialize export flags
    export_to_doccano = False
    export_to_labelstudio = False
    export_sample_size = None

    # Step 1: Ask if user wants to export
    export_confirm = Confirm.ask(
        "[bold yellow]Export to validation tool?[/bold yellow]",
        default=False
    )

    if export_confirm:
        # Step 2: Ask which tool to export to
        tool_choice = Prompt.ask(
            "[bold yellow]Which validation tool?[/bold yellow]",
            choices=["doccano", "labelstudio"],
            default="doccano"
        )

        # Set the appropriate export flag
        if tool_choice == "doccano":
            export_to_doccano = True
        else:  # labelstudio
            export_to_labelstudio = True

        # Step 2b: If Label Studio, ask export method
        labelstudio_direct_export = False
        labelstudio_api_url = None
        labelstudio_api_key = None

        if export_to_labelstudio:
            cli.console.print("\n[yellow]Label Studio export method:[/yellow]")
            cli.console.print("  • [cyan]jsonl[/cyan] - Export to JSONL file (manual import)")
            if cli.HAS_REQUESTS:
                cli.console.print("  • [cyan]direct[/cyan] - Direct export to Label Studio via API\n")
                export_choices = ["jsonl", "direct"]
            else:
                cli.console.print("  • [dim]direct[/dim] - Direct export via API [dim](requires 'requests' library)[/dim]\n")
                export_choices = ["jsonl"]

            export_method = Prompt.ask(
                "[bold yellow]Export method[/bold yellow]",
                choices=export_choices,
                default="jsonl"
            )

            if export_method == "direct":
                labelstudio_direct_export = True

                cli.console.print("\n[cyan]Label Studio API Configuration:[/cyan]")
                labelstudio_api_url = Prompt.ask(
                    "Label Studio URL",
                    default="http://localhost:8080"
                )

                labelstudio_api_key = Prompt.ask(
                    "API Key (from Label Studio Account & Settings)"
                )

        # Step 3: Ask about LLM predictions inclusion
        cli.console.print("\n[yellow]Include LLM predictions in export?[/yellow]")
        cli.console.print("  • [cyan]with[/cyan] - Include LLM annotations as predictions (for review/correction)")
        cli.console.print("  • [cyan]without[/cyan] - Export only data without predictions (for manual annotation)")
        cli.console.print("  • [cyan]both[/cyan] - Create two files: one with and one without predictions\n")

        prediction_mode = Prompt.ask(
            "[bold yellow]Prediction mode[/bold yellow]",
            choices=["with", "without", "both"],
            default="with"
        )

        # Step 4: Ask how many sentences to export
        cli.console.print("\n[yellow]How many annotated sentences to export?[/yellow]")
        cli.console.print("  • [cyan]all[/cyan] - Export all annotated sentences")
        cli.console.print("  • [cyan]representative[/cyan] - Representative sample (stratified by labels)")
        cli.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

        sample_choice = Prompt.ask(
            "[bold yellow]Export sample[/bold yellow]",
            choices=["all", "representative", "number"],
            default="all"
        )

        if sample_choice == "all":
            export_sample_size = "all"
        elif sample_choice == "representative":
            export_sample_size = "representative"
        else:  # number
            export_sample_size = cli._int_prompt_with_validation(
                "Number of sentences to export",
                100,
                1,
                999999
            )

    # ============================================================
    # EXECUTE ANNOTATION
    # ============================================================

    # CRITICAL: Use new organized structure with dataset-specific subfolder
    # Structure: logs/annotator/{session_id}/annotated_data/{dataset_name}/
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata.setdefault('annotation_session', {})['timestamp'] = timestamp

    # Determine annotation mode before creating directories
    if provider == 'openai' and openai_batch_mode:
        annotation_mode = 'openai_batch'
    else:
        annotation_mode = 'api' if provider in {'openai', 'anthropic', 'google'} else 'local'

    if annotation_mode == 'openai_batch':
        annotation_mode_display = "OpenAI Batch"
    elif annotation_mode == 'api':
        annotation_mode_display = "API"
    else:
        annotation_mode_display = "Local"

    provider_folder = (provider or "model_provider").replace("/", "_")
    model_folder = safe_model_name

    provider_subdir = session_dirs['annotated_data'] / provider_folder
    provider_subdir.mkdir(parents=True, exist_ok=True)

    model_subdir = provider_subdir / model_folder
    model_subdir.mkdir(parents=True, exist_ok=True)

    dataset_name = data_path.stem
    dataset_subdir = model_subdir / dataset_name
    dataset_subdir.mkdir(parents=True, exist_ok=True)

    if annotation_mode == 'openai_batch':
        batch_logs_root = model_subdir / "openai_batch_jobs"
        batch_logs_root.mkdir(parents=True, exist_ok=True)
        batch_dir = batch_logs_root / timestamp
        batch_dir.mkdir(parents=True, exist_ok=True)

        archive_root = Path(session_dirs.get('openai_batches', session_dirs['annotated_data']))
        archive_dir = archive_root / provider_folder / model_folder / dataset_name / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        pointer_file = archive_dir / "LOCATION.txt"
        if not pointer_file.exists():
            try:
                pointer_file.write_text(
                    f"Dataset-local batch artifacts stored at: {batch_dir}\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
    else:
        batch_dir = dataset_subdir
        archive_dir = None

    output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
    default_output_path = dataset_subdir / output_filename

    cli.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")
    cli.console.print()

    # Prepare prompts payload for pipeline
    prompts_payload = []
    for pc in prompt_configs:
        prompts_payload.append({
            'prompt': pc['prompt']['content'],
            'expected_keys': pc['prompt']['keys'],
            'prefix': pc['prefix']
        })

    # Build pipeline config
    pipeline_config = {
        'mode': 'file',
        'data_source': data_format,
        'data_format': data_format,
        'file_path': str(data_path),
        'text_column': text_column,
        'text_columns': [text_column],
        'annotation_column': 'annotation',
        'identifier_column': identifier_column,  # From Step 2b: User-selected ID strategy
        'run_annotation': True,
        'annotation_mode': annotation_mode,
        'annotation_provider': provider,
        'annotation_model': model_name,
        'api_key': api_key if api_key else None,
        'openai_batch_mode': openai_batch_mode,
        'openai_batch_mode': openai_batch_mode,
        'prompts': prompts_payload,
        'annotation_sample_size': annotation_limit,
        'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        'annotation_sample_seed': 42,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
        'max_workers': num_processes,
        'num_processes': num_processes,
        'use_parallel': num_processes > 1,
        'warmup': False,
        'disable_tqdm': True,  # Use Rich progress instead
        'output_format': data_format,
        'output_path': str(default_output_path),
        'save_incrementally': save_incrementally,
        'batch_size': batch_size,
        'run_validation': False,
        'run_training': False,
        'lang_column': lang_column,  # From Step 4b: Language column for training metadata
        'create_annotated_subset': True,
        'session_dirs': {key: str(value) for key, value in session_dirs.items()},
        'provider_subdir': str(provider_subdir),
        'model_subdir': str(model_subdir),
        'dataset_subdir': str(dataset_subdir),
        'openai_batch_dir': str(batch_dir),
        'openai_batch_archive_dir': str(archive_dir) if archive_dir else None,
        'provider_folder': provider_folder,
        'model_folder': model_folder,
        'dataset_name': dataset_name,
    }

    if annotation_mode == 'openai_batch':
        pipeline_config.setdefault('openai_batch_poll_interval', 5)
        pipeline_config.setdefault('openai_batch_completion_window', '24h')

    # Add model-specific options
    if provider == 'ollama':
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
            'top_k': top_k
        }
        pipeline_config['options'] = options

    # ============================================================
    # SAVE REPRODUCIBILITY METADATA
    # ============================================================
    if save_metadata:
        import json

        # Build comprehensive metadata
        available_rows = total_rows if total_rows is not None else None
        if annotation_limit:
            requested_rows = annotation_limit
        elif available_rows is not None:
            requested_rows = available_rows
        else:
            requested_rows = 'all'
        initial_remaining = (
            requested_rows
            if isinstance(requested_rows, int)
            else (available_rows if available_rows is not None else 'all')
        )
        metadata = {
            'annotation_session': {
                'timestamp': timestamp,
                'tool_version': 'LLMTool v1.0',
                'workflow': 'The Annotator - Smart Annotate'
            },
            'data_source': {
                'file_path': str(data_path),
                'file_name': data_path.name,
                'data_format': data_format,
                'text_column': text_column,
                'dataset_name': dataset_name,
                'total_rows': annotation_limit if annotation_limit else 'all',
                'requested_rows': requested_rows,
                'available_rows': available_rows,
                'sampling_strategy': sample_strategy if annotation_limit else 'none (all rows)',
                'sample_seed': 42 if sample_strategy == 'random' else None,
                'identifier_column': auto_identifier_column,
                'identifier_source': identifier_source,
                'provider_folder': provider_folder,
                'model_folder': model_folder,
            },
            'annotation_progress': {
                'requested': requested_rows,
                'completed': 0,
                'remaining': initial_remaining,
            },
            'model_configuration': {
                'provider': provider,
                'model_name': model_name,
                'annotation_mode': annotation_mode,
                'openai_batch_mode': openai_batch_mode,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'top_k': top_k if provider in ['ollama', 'google'] else None
            },
            'prompts': [
                {
                    'name': pc['prompt']['name'],
                    'file_path': str(pc['prompt']['path']) if 'path' in pc['prompt'] else None,
                    'expected_keys': pc['prompt']['keys'],
                    'prefix': pc['prefix'],
                    'prompt_content': pc['prompt']['content']
                }
                for pc in prompt_configs
            ],
            'processing_configuration': {
                'parallel_workers': None if annotation_mode == 'openai_batch' else num_processes,
                'batch_size': None if annotation_mode == 'openai_batch' else batch_size,
                'incremental_save': False if annotation_mode == 'openai_batch' else save_incrementally,
                'openai_batch_mode': openai_batch_mode,
                'openai_batch_dir': str(batch_dir) if annotation_mode == 'openai_batch' else None,
                'openai_batch_archive_dir': str(archive_dir) if archive_dir else None,
                'openai_batch_archive_dir': str(archive_dir) if archive_dir else None,
                'identifier_column': auto_identifier_column,
                'auto_identifier_column': auto_identifier_column,
                'identifier_source': identifier_source,
                'provider_folder': provider_folder,
                'model_folder': model_folder,
                'dataset_name': dataset_name,
            },
            'output': {
                'output_path': str(default_output_path),
                'output_format': data_format
            },
            'export_preferences': {
                'export_to_doccano': export_to_doccano,
                'export_to_labelstudio': export_to_labelstudio,
                'export_sample_size': export_sample_size,
                'prediction_mode': prediction_mode if (export_to_doccano or export_to_labelstudio) else 'with',
                'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
                'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
                'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None
            },
            'training_workflow': {
                'enabled': False,  # Will be updated after training workflow
                'training_params_file': None,  # Will be added after training
                'note': 'Training parameters will be saved separately after annotation completes'
            }
        }

        # Save metadata JSON (PRE-ANNOTATION SAVE POINT 1)
        # Use dataset-specific subdirectory for metadata too
        metadata_subdir = session_dirs['metadata'] / provider_folder / model_folder / dataset_name
        metadata_subdir.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
        metadata_path = metadata_subdir / metadata_filename

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[bold green]✅ Metadata saved for reproducibility[/bold green]")
        cli.console.print(f"[bold cyan]📋 Metadata File:[/bold cyan]")
        cli.console.print(f"   {metadata_path}\n")
        tracker.mark_step(
            5,
            detail=f"Metadata saved: {metadata_path.name}",
            extra={
                "metadata_path": str(metadata_path),
                "dataset_path": str(data_path),
                "text_column": text_column,
                "identifier_column": identifier_column,
            },
        )

    # Execute pipeline with Rich progress
    try:
        cli.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

        # Create pipeline controller with session_id for organized logging
        from ..pipelines.pipeline_controller import PipelineController
        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=session_id  # Pass session_id for organized logging
        )

        # Use RichProgressManager for elegant display
        from ..utils.rich_progress_manager import RichProgressManager
        from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

        with RichProgressManager(
            show_json_every=1,  # Show JSON sample for every annotation
            compact_mode=False   # Full preview panels
        ) as progress_manager:
            # Wrap pipeline for enhanced JSON tracking
            enhanced_pipeline = EnhancedPipelineWrapper(
                pipeline_with_progress,
                progress_manager
            )

            # Run pipeline
            state = enhanced_pipeline.run_pipeline(pipeline_config)

            # Check for errors
            if state.errors:
                error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                cli.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                cli.console.print("[dim]Press Enter to return to menu...[/dim]")
                input()
                return

        # Get results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        # Display success message
        cli.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        # Display statistics if available
        total_annotated = annotation_results.get('total_annotated', 0)
        if total_annotated:
            cli.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

            mean_time = annotation_results.get('mean_inference_time')
            if isinstance(mean_time, (int, float)):
                cli.console.print(f"   Avg inference time: {mean_time:.2f}s")

        subset_path = annotation_results.get('annotated_subset_path')
        if subset_path:
            cli.console.print(f"   Annotated subset: {subset_path}")

        batch_output = annotation_results.get('openai_batch_output_path')
        if batch_output:
            cli.console.print(f"   Batch output JSONL: {batch_output}")

        batch_metadata_path = annotation_results.get('openai_batch_metadata_path')
        if batch_metadata_path:
            cli.console.print(f"   Batch metadata: {batch_metadata_path}")

        preview_samples = annotation_results.get('preview_samples') or []
        if preview_samples:
            cli.console.print("\n[bold cyan]📝 Sample Annotations:[/bold cyan]")
            preview_keys = list(preview_samples[0].keys())
            preview_table = Table(show_header=True, header_style="bold magenta")
            for key in preview_keys:
                preview_table.add_column(key.replace('_', ' ').title(), overflow="fold")
            for sample in preview_samples:
                preview_table.add_row(*[str(sample.get(key, '')) for key in preview_keys])
            cli.console.print(preview_table)

        tracker.mark_step(
            6,
            detail=f"Annotated rows: {total_annotated or 'n/a'}",
            extra={
                "output_file": output_file,
                "total_annotated": total_annotated,
                "success_count": annotation_results.get('success_count'),
            },
        )

        # ============================================================
        # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
        # ============================================================
        if not lang_column:
            cli.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
            cli.console.print("[yellow]No language column was provided. Detecting languages for training...[/yellow]\n")

            try:
                import pandas as pd
                from llm_tool.utils.language_detector import LanguageDetector

                # Load annotated file
                df_annotated = pd.read_csv(output_file)

                # CRITICAL: Only detect languages for ANNOTATED rows
                # The output file may contain ALL original rows, but we only want to detect
                # languages for rows that were actually annotated
                original_row_count = len(df_annotated)

                # Try to identify annotated rows by checking for annotation columns
                # Common annotation column names: 'label', 'category', 'annotation', 'labels'
                annotation_cols = [col for col in df_annotated.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]

                if annotation_cols:
                    # Filter to only rows that have annotations (non-null AND non-empty in annotation column)
                    annotation_col = annotation_cols[0]
                    df_annotated = df_annotated[(df_annotated[annotation_col].notna()) & (df_annotated[annotation_col] != '')].copy()
                    cli.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows (out of {original_row_count:,} total rows in file)[/dim]")
                else:
                    cli.console.print(f"[yellow]⚠️  Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                if len(df_annotated) == 0:
                    cli.console.print("[yellow]⚠️  No annotated rows found. Skipping language detection.[/yellow]")
                elif text_column in df_annotated.columns:
                    # Get ALL texts (including NaN) to maintain index alignment
                    all_texts = df_annotated[text_column].tolist()

                    # Count non-empty texts for display
                    non_empty_texts = sum(1 for text in all_texts if pd.notna(text) and len(str(text).strip()) > 10)

                    if non_empty_texts > 0:
                        detector = LanguageDetector()
                        detected_languages = []

                        # Progress indicator
                        from tqdm import tqdm
                        cli.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                        for text in tqdm(all_texts, desc="Detecting languages", disable=not cli.HAS_RICH):
                            # Handle NaN and empty texts
                            if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                                detected_languages.append('unknown')
                            else:
                                try:
                                    detected = detector.detect(str(text))
                                    if detected and detected.get('language'):
                                        detected_languages.append(detected['language'])
                                    else:
                                        detected_languages.append('unknown')
                                except Exception as e:
                                    cli.logger.debug(f"Language detection failed for text: {e}")
                                    detected_languages.append('unknown')

                        # Add language column to the filtered dataframe
                        df_annotated['lang'] = detected_languages

                        # Reload the FULL original file and update only the annotated rows
                        df_full = pd.read_csv(output_file)

                        # Initialize lang column if it doesn't exist
                        if 'lang' not in df_full.columns:
                            df_full['lang'] = 'unknown'

                        # Update language for annotated rows only
                        # Match by index of df_annotated within df_full
                        df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values

                        # Save updated full file with language column
                        df_full.to_csv(output_file, index=False)

                        # Show distribution
                        lang_counts = {}
                        for lang in detected_languages:
                            if lang != 'unknown':
                                lang_counts[lang] = lang_counts.get(lang, 0) + 1

                        if lang_counts:
                            total = sum(lang_counts.values())
                            cli.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                            lang_table.add_column("Language", style="cyan", width=12)
                            lang_table.add_column("Count", style="yellow", justify="right", width=12)
                            lang_table.add_column("Percentage", style="green", justify="right", width=12)

                            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total * 100) if total > 0 else 0
                                lang_table.add_row(
                                    lang.upper(),
                                    f"{count:,}",
                                    f"{percentage:.1f}%"
                                )

                            cli.console.print(lang_table)
                            cli.console.print(f"\n[green]✓ Language column 'lang' added to {output_file}[/green]")
                        else:
                            cli.console.print("[yellow]⚠️  No languages detected successfully[/yellow]")

            except Exception as e:
                cli.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
                cli.logger.exception("Language detection failed")


        # Export to Doccano JSONL if requested
        if export_to_doccano:
            cli._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs,
                data_path=data_path,
                timestamp=timestamp,
                sample_size=export_sample_size,
                session_dirs=session_dirs,
                provider_folder=provider_folder,
                model_folder=model_folder
            )

        # Export to Label Studio if requested
        if export_to_labelstudio:
            if labelstudio_direct_export:
                # Direct export to Label Studio via API
                cli._export_to_labelstudio_direct(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    api_url=labelstudio_api_url,
                    api_key=labelstudio_api_key
                )
            else:
                # Export to JSONL file
                cli._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    session_dirs=session_dirs,
                    provider_folder=provider_folder,
                    model_folder=model_folder
                )

        tracker.mark_step(
            7,
            detail="Post-processing complete",
            extra={
                "export_doccano": export_to_doccano,
                "export_labelstudio": export_to_labelstudio,
                "prediction_mode": prediction_mode,
            },
        )
        tracker.update_status("completed")

        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    except Exception as exc:
        tracker.update_status("failed", note=str(exc))
        cli.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Annotation execution failed")
        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

def run_factory_workflow(cli, session_id: str = None, session_dirs: Optional[Dict[str, Path]] = None):
    """Execute complete annotation → training workflow

    Parameters
    ----------
    session_id : str, optional
        Session identifier for organizing outputs. If None, a timestamp-based ID is generated.
    """
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    # Generate session_id if not provided (for backward compatibility)
    if session_id is None:
        session_id = f"factory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create session directories
    if session_dirs is None:
        session_dirs = cli._create_annotator_factory_session_directories(session_id)

    tracker = AnnotationResumeTracker(
        mode=AnnotationMode.FACTORY,
        session_id=session_id,
        session_dirs=session_dirs,
        step_catalog=FACTORY_RESUME_STEPS,
        session_name=session_id,
    )
    tracker.update_status("active")

    # Display Annotator Factory STEP 1 banner
    from llm_tool.cli.banners import BANNERS, STEP_NUMBERS, STEP_LABEL
    from rich.align import Align

    cli.console.print()

    # Display "STEP" label in ASCII art
    for line in STEP_LABEL.split('\n'):
        cli.console.print(Align.center(f"[bold {BANNERS['llm_annotator']['color']}]{line}[/bold {BANNERS['llm_annotator']['color']}]"))

    # Display "1/3" in ASCII art
    for line in STEP_NUMBERS['1/3'].split('\n'):
        cli.console.print(Align.center(f"[bold {BANNERS['llm_annotator']['color']}]{line}[/bold {BANNERS['llm_annotator']['color']}]"))

    cli.console.print()

    # Display main banner (centered)
    for line in BANNERS['llm_annotator']['ascii'].split('\n'):
        cli.console.print(Align.center(f"[bold {BANNERS['llm_annotator']['color']}]{line}[/bold {BANNERS['llm_annotator']['color']}]"))

    # Display tagline (centered)
    cli.console.print(Align.center(f"[{BANNERS['llm_annotator']['color']}]{BANNERS['llm_annotator']['tagline']}[/{BANNERS['llm_annotator']['color']}]"))
    cli.console.print()

    # Step 1.1: Data Source Selection
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.1:[/bold cyan] [bold white]Data Source Selection[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Choose between file-based datasets or SQL database sources.[/dim]\n")

    # Ask user to choose between files and SQL database
    cli.console.print("[yellow]Available data sources:[/yellow]")
    cli.console.print("  1. 📁 Files (CSV/Excel/JSON/etc.) - Auto-detected or manual")
    cli.console.print("  2. 🗄️  SQL Database (PostgreSQL/MySQL/SQLite/SQL Server)\n")

    data_source_choice = Prompt.ask(
        "Data source",
        choices=["1", "2"],
        default="1"
    )

    use_sql_database = (data_source_choice == "2")

    if use_sql_database:
        # SQL DATABASE WORKFLOW
        cli.console.print("\n[bold cyan]🗄️  SQL Database (Training Sample)[/bold cyan]\n")
        cli.console.print("[yellow]Note: For training, you'll select a representative sample from your database[/yellow]\n")

        # Database type selection
        db_choices = ["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server"]
        db_table = Table(title="Database Types", border_style="cyan")
        db_table.add_column("#", style="cyan", width=6)
        db_table.add_column("Database Type", style="white")
        for i, choice in enumerate(db_choices, 1):
            db_table.add_row(str(i), choice)
        cli.console.print(db_table)

        db_choice = cli._int_prompt_with_validation("Select database type", 1, 1, len(db_choices))
        db_type_name = db_choices[db_choice - 1]

        # Connection details
        if db_type_name == "SQLite":
            db_file = Prompt.ask("SQLite database file path")
            connection_string = f"sqlite:///{db_file}"
        else:
            host = Prompt.ask("Database host", default="localhost")
            default_ports = {"PostgreSQL": "5432", "MySQL": "3306", "Microsoft SQL Server": "1433"}
            port = Prompt.ask("Port", default=default_ports.get(db_type_name, "5432"))
            username = Prompt.ask("Username", default="postgres" if db_type_name == "PostgreSQL" else "root")
            password = Prompt.ask("Password", password=True)
            database = Prompt.ask("Database name")

            if db_type_name == "PostgreSQL":
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            elif db_type_name == "MySQL":
                connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            else:
                connection_string = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

        # Test connection
        cli.console.print("\nTesting connection...")
        try:
            from sqlalchemy import create_engine, inspect, text
            import pandas as pd
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            cli.console.print("[green]✓ Connected successfully![/green]\n")
        except Exception as e:
            cli.console.print(f"[red]✗ Connection failed: {str(e)}[/red]")
            input("\nPress Enter to continue...")
            return

        # Table selection
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            cli.console.print("[red]No tables found[/red]")
            input("\nPress Enter to continue...")
            return

        # Get row counts
        table_info = []
        for table in tables:
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    table_info.append((table, result.scalar()))
            except:
                table_info.append((table, None))

        tables_table = Table(title="Available Tables", border_style="cyan")
        tables_table.add_column("#", style="cyan", width=3)
        tables_table.add_column("Table Name", style="white")
        tables_table.add_column("Rows", style="green", justify="right")
        for i, (table, rows) in enumerate(table_info, 1):
            tables_table.add_row(str(i), table, f"{rows:,}" if rows else "?")
        cli.console.print(tables_table)

        table_choice = cli._int_prompt_with_validation("Select table", 1, 1, len(table_info))
        selected_table, total_rows = table_info[table_choice - 1]
        cli.console.print(f"\n[green]✓ Selected: {selected_table} ({total_rows:,} rows)[/green]\n")

        # Load ALL data to temporary CSV (will use SAME workflow as files)
        from datetime import datetime
        import pandas as pd

        df = pd.read_sql(f"SELECT * FROM {selected_table}", engine)

        # Save to CSV in data/annotations
        annotations_dir = cli.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_path = annotations_dir / f"quickstart_sql_{selected_table}_{timestamp}.csv"
        df.to_csv(data_path, index=False)
        data_format = 'csv'

        cli.console.print(f"[green]✓ Loaded {len(df):,} rows from {selected_table}[/green]")
        cli.console.print(f"[dim]Saved to: {data_path}[/dim]")

    else:
        # FILE-BASED WORKFLOW (original code)
        if not cli.detected_datasets:
            cli.console.print("[yellow]No datasets auto-detected.[/yellow]")
            data_path = Path(cli._prompt_file_path("Dataset path"))
        else:
            cli.console.print(f"\n[bold cyan]📊 Found {len(cli.detected_datasets)} dataset(s):[/bold cyan]\n")

            # Create table for datasets
            datasets_table = Table(border_style="cyan", show_header=True)
            datasets_table.add_column("#", style="bold yellow", width=4)
            datasets_table.add_column("Filename", style="white")
            datasets_table.add_column("Format", style="green", width=10)
            datasets_table.add_column("Size", style="magenta", width=10)
            datasets_table.add_column("Rows", style="cyan", width=10)
            datasets_table.add_column("Columns", style="blue", width=10)

            for i, ds in enumerate(cli.detected_datasets[:20], 1):
                # Format size
                if ds.size_mb < 0.1:
                    size_str = f"{ds.size_mb * 1024:.1f} KB"
                else:
                    size_str = f"{ds.size_mb:.1f} MB"

                # Format rows and columns
                rows_str = f"{ds.rows:,}" if ds.rows else "?"
                cols_str = str(len(ds.columns)) if ds.columns else "?"

                datasets_table.add_row(
                    str(i),
                    ds.path.name,
                    ds.format.upper(),
                    size_str,
                    rows_str,
                    cols_str
                )

            cli.console.print(datasets_table)
            cli.console.print()

            use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
            if use_detected:
                choice = cli._int_prompt_with_validation("Select dataset", 1, 1, len(cli.detected_datasets))
                data_path = cli.detected_datasets[choice - 1].path
            else:
                data_path = Path(cli._prompt_file_path("Dataset path"))

        # Detect format
        data_format = data_path.suffix[1:].lower()
        if data_format == 'xlsx':
            data_format = 'excel'

        cli.console.print(f"[green]✓ Selected: {data_path.name} ({data_format})[/green]")

    # Step 1.2: Text Column Selection (MODERNIZED - Same format as quick start)
    cli.console.print("\n[bold]Step 1.2/4: Text Column Selection[/bold]\n")

    # Detect text columns using the advanced detection system
    column_info = cli._detect_text_columns(data_path)
    import pandas as pd
    df_sample = pd.read_csv(data_path, nrows=100) if data_path.suffix == '.csv' else pd.read_excel(data_path, nrows=100)

    candidate_names = [candidate['name'] for candidate in column_info.get('text_candidates', [])]
    all_columns = column_info.get('all_columns', [])

    if candidate_names:
        cli.console.print("[dim]Detected text columns (sorted by confidence):[/dim]")

        # Create table for text candidates ONLY
        col_table = Table(border_style="blue")
        col_table.add_column("#", style="cyan", width=3)
        col_table.add_column("Column", style="white")
        col_table.add_column("Confidence", style="yellow")
        col_table.add_column("Avg Length", style="green")
        col_table.add_column("Sample", style="dim")

        for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
            # Color code confidence
            conf_color = {
                "high": "[green]High[/green]",
                "medium": "[yellow]Medium[/yellow]",
                "low": "[orange1]Low[/orange1]",
                "very_low": "[red]Very Low[/red]"
            }
            conf_display = conf_color.get(candidate['confidence'], candidate['confidence'])

            col_table.add_row(
                str(i),
                candidate['name'],
                conf_display,
                f"{candidate['avg_length']:.0f} chars",
                candidate['sample'][:50] + "..." if len(candidate['sample']) > 50 else candidate['sample']
            )

        cli.console.print(col_table)

        # Show all columns list
        if all_columns:
            cli.console.print(f"\n[dim]All columns ({len(all_columns)}): {', '.join(all_columns)}[/dim]")

        default_col = candidate_names[0]
        prompt_message = "\n[bold yellow]Enter column name[/bold yellow] (or choose from above)"
    else:
        # No candidates detected, show all columns
        if all_columns:
            cli.console.print(f"\n[yellow]Could not auto-detect text columns.[/yellow]")
            cli.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")
        default_col = "text"
        prompt_message = "Text column name"

    while True:
        raw_choice = Prompt.ask(prompt_message, default=default_col)
        normalized = _normalize_column_choice(raw_choice, all_columns, candidate_names or all_columns)

        if normalized:
            text_column = normalized
            break

        if not all_columns:
            text_column = raw_choice.strip()
            break

        cli.console.print(f"[red]✗ Column selection '{raw_choice}' could not be resolved.[/red]")
        if candidate_names:
            cli.console.print("[dim]Enter the column name or the number shown in the table.[/dim]")
        if all_columns:
            cli.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

    # Step 1.2b: ID Column Selection (MODERNIZED with new system)
    identifier_column = DataDetector.display_and_select_id_column(
        cli.console,
        df_sample,
        text_column=text_column,
        step_label="Step 1.2b/4: Identifier Column Selection"
    )
    identifier_source = "user" if identifier_column else "auto"
    auto_identifier_column = identifier_column or "llm_annotation_id"
    pipeline_identifier_column = identifier_column if identifier_column else None
    identifier_column = pipeline_identifier_column

    # Store column info for later use
    column_info['df'] = df_sample

    # Step 1.3: Model Selection
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.3:[/bold cyan] [bold white]Model Selection[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Choose from local (Ollama) or cloud (OpenAI/Anthropic) models for annotation.[/dim]\n")

    selected_llm = cli._select_llm_interactive()
    provider = selected_llm.provider
    model_name = selected_llm.name

    # Get API key if needed
    api_key = None
    if selected_llm.requires_api_key:
        api_key = cli._get_or_prompt_api_key(provider, model_name)

    openai_batch_mode = _prompt_openai_batch_mode(cli, provider, "the factory annotation stage")

    # Step 1.4: Prompt Configuration
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.4:[/bold cyan] [bold white]Prompt Configuration[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Select from existing prompts or create new annotation instructions.[/dim]")

    # Auto-detect prompts
    detected_prompts = cli._detect_prompts_in_folder()

    if detected_prompts:
        cli.console.print(f"\n[green]✓ Found {len(detected_prompts)} prompts in prompts/ folder:[/green]")
        for i, p in enumerate(detected_prompts, 1):
            # Display ALL keys, not truncated
            keys_str = ', '.join(p['keys'])
            cli.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
            cli.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

        # Explain the options clearly
        cli.console.print("\n[bold]Prompt Selection Options:[/bold]")
        cli.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts (multi-prompt mode)")
        cli.console.print("           → Each text will be annotated with all prompts")
        cli.console.print("           → Useful when you want complete annotations from all perspectives")
        cli.console.print("\n  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number (e.g., 1,3,5)")
        cli.console.print("           → Only selected prompts will be used")
        cli.console.print("           → Useful when testing or when you need only certain annotations")
        cli.console.print("\n  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
        cli.console.print("           → Interactive guided prompt creation")
        cli.console.print("           → Optional AI assistance for definitions")
        cli.console.print("           → [bold green]Recommended for new research projects![/bold green]")
        cli.console.print("\n  [cyan]custom[/cyan]  - Provide path to a prompt file NOT in prompts/ folder")
        cli.console.print("           → Use a prompt from another location")
        cli.console.print("           → Useful for testing new prompts or one-off annotations")

        prompt_choice = Prompt.ask(
            "\n[bold yellow]Prompt selection[/bold yellow]",
            choices=["all", "select", "wizard", "custom"],
            default="all"
        )

        selected_prompts = []
        if prompt_choice == "all":
            selected_prompts = detected_prompts
            cli.console.print(f"[green]✓ Using all {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "select":
            indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
            if indices.strip():  # Only process if not empty
                for idx_str in indices.split(','):
                    idx_str = idx_str.strip()
                    if idx_str:  # Skip empty strings
                        try:
                            idx = int(idx_str) - 1
                            if 0 <= idx < len(detected_prompts):
                                selected_prompts.append(detected_prompts[idx])
                        except ValueError:
                            cli.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
            if not selected_prompts:
                cli.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                selected_prompts = detected_prompts
            else:
                cli.console.print(f"[green]✓ Selected {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            # Custom path
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]
    else:
        cli.console.print("[yellow]No prompts found in prompts/ folder[/yellow]")

        # Offer wizard or custom path
        cli.console.print("\n[bold]Prompt Options:[/bold]")
        cli.console.print("  [cyan]wizard[/cyan] - 🧙‍♂️ Create prompt using Social Science Wizard (Recommended)")
        cli.console.print("  [cyan]custom[/cyan] - Provide path to existing prompt file")

        choice = Prompt.ask(
            "\n[bold yellow]Select option[/bold yellow]",
            choices=["wizard", "custom"],
            default="wizard"
        )

        if choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]


    # Language detection moved to training phase
    # Language columns will be detected and handled automatically after annotation
    lang_column = None
    available_columns = column_info.get('all_columns', []) if column_info else []
    if available_columns:
        # Silently detect potential language columns for metadata
        potential_lang_cols = [col for col in available_columns
                              if col.lower() in ['lang', 'language', 'langue', 'lng', 'iso_lang']]

        # If language column exists, note it for later use but don't ask user
        if potential_lang_cols:
            lang_column = potential_lang_cols[0]  # Use first one if found
    # Multi-prompt prefix configuration (if needed)
    prompt_configs = []
    if len(selected_prompts) > 1:
        cli.console.print("\n[bold]Multi-Prompt Mode:[/bold] Configure key prefixes")
        cli.console.print("[dim]Prefixes help identify which prompt generated which keys[/dim]\n")

        for i, prompt in enumerate(selected_prompts, 1):
            cli.console.print(f"\n[cyan]Prompt {i}: {prompt['name']}[/cyan]")
            cli.console.print(f"  Keys: {', '.join(prompt['keys'])}")

            add_prefix = Confirm.ask(f"Add prefix to keys for this prompt?", default=True)
            prefix = ""
            if add_prefix:
                default_prefix = prompt['name'].lower().replace(' ', '_')
                prefix = Prompt.ask("Prefix", default=default_prefix)
                cli.console.print(f"  [green]Keys will become: {', '.join([f'{prefix}_{k}' for k in prompt['keys'][:3]])}[/green]")

            prompt_configs.append({
                'prompt': prompt,
                'prefix': prefix
            })
    else:
        # Single prompt - no prefix needed
        prompt_configs = [{'prompt': selected_prompts[0], 'prefix': ''}]

    # Step 1.5: Advanced Options
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.5:[/bold cyan] [bold white]Advanced Options[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Configure processing settings for optimal performance.[/dim]")

    # ============================================================
    # DATASET SCOPE
    # ============================================================
    cli.console.print("\n[bold cyan]📊 Dataset Scope[/bold cyan]")
    cli.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

    # Get total rows if possible
    total_rows = None
    if column_info.get('df') is not None:
        # We have a sample, extrapolate
        total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

    if total_rows:
        cli.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

    # Option 1: Annotate all or limited
    cli.console.print("[yellow]Option 1:[/yellow] Annotate ALL rows vs LIMIT to specific number")
    cli.console.print("  • [cyan]all[/cyan]   - Annotate the entire dataset")
    cli.console.print("           [dim]Use this for production annotations[/dim]")
    cli.console.print("  • [cyan]limit[/cyan] - Specify exact number of rows to annotate")
    cli.console.print("           [dim]Use this for testing or partial annotation[/dim]")

    scope_choice = Prompt.ask(
        "\nAnnotate entire dataset or limit rows?",
        choices=["all", "limit"],
        default="all"
    )

    annotation_limit = None
    use_sample = False
    sample_strategy = "head"
    recommended_sample = None

    if scope_choice == "limit":
        # Option 2: FIRST ask about representative sample calculation (before asking for number)
        if total_rows and total_rows > 1000:
            cli.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
            cli.console.print("  Calculate statistically representative sample size (95% confidence interval)")
            cli.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

            calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

            if calculate_sample:
                # Formula: n = (Z² × p × (1-p)) / E²
                # For 95% CI: Z=1.96, p=0.5 (max variance), E=0.05 (5% margin)
                import math
                z = 1.96
                p = 0.5
                e = 0.05
                n_infinite = (z**2 * p * (1-p)) / (e**2)
                n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                recommended_sample = int(math.ceil(n_adjusted))

                cli.console.print(f"\n[green]📈 Recommended sample size: {recommended_sample} rows[/green]")
                cli.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")
                cli.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

        # THEN ask for specific number (with recommendation as default if calculated)
        default_limit = recommended_sample if recommended_sample else 100
        annotation_limit = cli._int_prompt_with_validation(
            f"How many rows to annotate?",
            default=default_limit,
            min_value=1,
            max_value=total_rows if total_rows else 1000000
        )

        # Check if user chose the recommended sample
        if recommended_sample and annotation_limit == recommended_sample:
            use_sample = True

        # Option 3: Random sampling
        cli.console.print("\n[yellow]Option 3:[/yellow] Sampling Strategy")
        cli.console.print("  Choose how to select the rows to annotate")
        cli.console.print("  • [cyan]head[/cyan]   - Take first N rows (faster, sequential)")
        cli.console.print("           [dim]Good for testing, preserves order[/dim]")
        cli.console.print("  • [cyan]random[/cyan] - Random sample of N rows (representative)")
        cli.console.print("           [dim]Better for statistical validity, unbiased[/dim]")

        sample_strategy = Prompt.ask(
            "\nSampling strategy",
            choices=["head", "random"],
            default="random" if use_sample else "head"
        )

    # ============================================================
    # PARALLEL PROCESSING
    # ============================================================
    if openai_batch_mode:
        cli.console.print("\n[bold cyan]⚙️  Processing[/bold cyan]")
        cli.console.print(
            "[dim]OpenAI Batch mode manages concurrency, retries, and persistence. "
            "Local parallelism and incremental save settings are skipped.[/dim]\n"
        )
        num_processes = 1
        save_incrementally = False
        batch_size = 1
    else:
        cli.console.print("\n[bold cyan]⚙️  Parallel Processing[/bold cyan]")
        cli.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

        cli.console.print("[yellow]Parallel Workers:[/yellow]")
        cli.console.print("  Number of simultaneous annotation processes")
        cli.console.print("\n  [red]⚠️  IMPORTANT:[/red]")
        cli.console.print("  [dim]Most local machines can only handle 1 worker for LLM inference[/dim]")
        cli.console.print("  [dim]Parallel processing is mainly useful for API models[/dim]")
        cli.console.print("\n  • [cyan]1 worker[/cyan]  - Sequential processing")
        cli.console.print("           [dim]Recommended for: Local models (Ollama), first time users, debugging[/dim]")
        cli.console.print("  • [cyan]2-4 workers[/cyan] - Moderate parallelism")
        cli.console.print("           [dim]Recommended for: API models (OpenAI, Claude) - avoid rate limits[/dim]")
        cli.console.print("  • [cyan]4-8 workers[/cyan] - High parallelism")
        cli.console.print("           [dim]Recommended for: API models only - requires high rate limits[/dim]")

        num_processes = cli._int_prompt_with_validation("Parallel workers", 1, 1, 16)

        # ============================================================
        # INCREMENTAL SAVE
        # ============================================================
        cli.console.print("\n[bold cyan]💾 Incremental Save[/bold cyan]")
        cli.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

        cli.console.print("[yellow]Enable incremental save?[/yellow]")
        cli.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
        cli.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
        cli.console.print("  • [red]No[/red]  - Save only at the end")
        cli.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

        save_incrementally = Confirm.ask("\n💿 Enable incremental save?", default=True)

        # Only ask for batch size if incremental save is enabled
        if save_incrementally:
            cli.console.print("\n[yellow]Batch Size:[/yellow]")
            cli.console.print("  Number of rows processed between each save")
            cli.console.print("  • [cyan]Smaller (1-10)[/cyan]   - Very frequent saves, maximum safety")
            cli.console.print("           [dim]Use for: Unstable systems, expensive APIs, testing[/dim]")
            cli.console.print("  • [cyan]Medium (10-50)[/cyan]   - Balanced safety and performance")
            cli.console.print("           [dim]Use for: Most production cases[/dim]")
            cli.console.print("  • [cyan]Larger (50-200)[/cyan]  - Less frequent saves, better performance")
            cli.console.print("           [dim]Use for: Stable systems, large datasets, local models[/dim]")

            batch_size = cli._int_prompt_with_validation("Batch size", 1, 1, 1000)
        else:
            batch_size = None  # Not used when incremental save is disabled

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    cli.console.print("\n[bold cyan]🎛️  Model Parameters[/bold cyan]")
    cli.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

    # Check if model supports parameter tuning
    model_name_lower = model_name.lower()
    is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
    supports_params = not is_o_series

    if not supports_params:
        cli.console.print(f"[yellow]⚠️  Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
        cli.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")
        configure_params = False
    else:
        cli.console.print("[yellow]Configure model parameters?[/yellow]")
        cli.console.print("  Adjust how the model generates responses")
        cli.console.print("  [dim]• Default values work well for most cases[/dim]")
        cli.console.print("  [dim]• Advanced users can fine-tune for specific needs[/dim]")
        configure_params = Confirm.ask("\nConfigure model parameters?", default=False)

    # Default values
    temperature = 0.7
    max_tokens = 1000
    top_p = 1.0
    top_k = 40

    if configure_params:
        cli.console.print("\n[bold]Parameter Explanations:[/bold]\n")

        # Temperature
        cli.console.print("[cyan]🌡️  Temperature (0.0 - 2.0):[/cyan]")
        cli.console.print("  Controls randomness in responses")
        cli.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
        cli.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
        cli.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
        cli.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
        cli.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
        cli.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
        temperature = FloatPrompt.ask("Temperature", default=0.7)

        # Max tokens
        cli.console.print("\n[cyan]📏 Max Tokens:[/cyan]")
        cli.console.print("  Maximum length of the response")
        cli.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
        cli.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
        cli.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
        cli.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

        # Top_p (nucleus sampling)
        cli.console.print("\n[cyan]🎯 Top P (0.0 - 1.0):[/cyan]")
        cli.console.print("  Nucleus sampling - alternative to temperature")
        cli.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
        cli.console.print("           [dim]More deterministic, safer outputs[/dim]")
        cli.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
        cli.console.print("           [dim]More creative, diverse outputs[/dim]")
        cli.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
        top_p = FloatPrompt.ask("Top P", default=1.0)

        # Top_k (only for some models)
        if provider in ['ollama', 'google']:
            cli.console.print("\n[cyan]🔢 Top K:[/cyan]")
            cli.console.print("  Limits vocabulary to K most likely next tokens")
            cli.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
            cli.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
            cli.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
            top_k = cli._int_prompt_with_validation("Top K", 40, 1, 100)

    # Step 1.6: Execute
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.6:[/bold cyan] [bold white]Review & Execute[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Review your configuration and start the annotation process.[/dim]")

    # Determine annotation mode once for display and downstream logic
    if provider == 'openai' and openai_batch_mode:
        annotation_mode = 'openai_batch'
        annotation_mode_display = "OpenAI Batch"
    elif provider in {'openai', 'anthropic', 'google'}:
        annotation_mode = 'api'
        annotation_mode_display = "API"
    else:
        annotation_mode = 'local'
        annotation_mode_display = "Local"

    # Display comprehensive summary
    summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True)
    summary_table.add_column("Category", style="bold cyan", width=20)
    summary_table.add_column("Setting", style="yellow", width=25)
    summary_table.add_column("Value", style="white")

    # Data section
    summary_table.add_row("📁 Data", "Dataset", str(data_path.name))
    summary_table.add_row("", "Format", data_format.upper())
    summary_table.add_row("", "Text Column", text_column)
    if total_rows:
        summary_table.add_row("", "Total Rows", f"{total_rows:,}")
    if annotation_limit:
        summary_table.add_row("", "Rows to Annotate", f"{annotation_limit:,} ({sample_strategy})")
    else:
        summary_table.add_row("", "Rows to Annotate", "ALL")

    # Model section
    summary_table.add_row("🤖 Model", "Provider/Model", f"{provider}/{model_name}")
    summary_table.add_row("", "Temperature", f"{temperature}")
    summary_table.add_row("", "Max Tokens", f"{max_tokens}")
    if configure_params:
        summary_table.add_row("", "Top P", f"{top_p}")
        if provider in ['ollama', 'google']:
            summary_table.add_row("", "Top K", f"{top_k}")

    # Prompts section
    summary_table.add_row("📝 Prompts", "Count", f"{len(prompt_configs)}")
    for i, pc in enumerate(prompt_configs, 1):
        prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
        summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

    # Processing section
    if openai_batch_mode:
        summary_table.add_row(
            "⚙️  Processing",
            "Annotation Mode",
            "OpenAI Batch (OpenAI-managed async job)"
        )
        summary_table.add_row("", "Parallel Workers", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Batch Size", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Incremental Save", "N/A (handled after batch completion)")
    else:
        summary_table.add_row("⚙️  Processing", "Annotation Mode", annotation_mode_display)
        summary_table.add_row("", "Parallel Workers", str(num_processes))
        summary_table.add_row("", "Batch Size", str(batch_size))
        summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

    cli.console.print("\n")
    cli.console.print(summary_table)

    if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
        return

    # ============================================================
    # REPRODUCIBILITY METADATA
    # ============================================================
    cli.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
    cli.console.print("[green]✓ Session parameters are automatically saved for:[/green]\n")

    cli.console.print("  [green]1. Resume Capability[/green]")
    cli.console.print("     • Continue this annotation if it stops or crashes")
    cli.console.print("     • Annotate additional rows later with same settings")
    cli.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

    cli.console.print("  [green]2. Scientific Reproducibility[/green]")
    cli.console.print("     • Document exact parameters for research papers")
    cli.console.print("     • Reproduce identical annotations in the future")
    cli.console.print("     • Track model version, prompts, and all settings\n")

    # Metadata is ALWAYS saved automatically for reproducibility
    save_metadata = True

    # ============================================================
    # VALIDATION TOOL EXPORT OPTION
    # ============================================================
    cli.console.print("\n[bold cyan]📤 Validation Tool Export[/bold cyan]")
    cli.console.print("[dim]Export annotations to JSONL format for human validation[/dim]\n")

    cli.console.print("[yellow]Available validation tools:[/yellow]")
    cli.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
    cli.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
    cli.console.print("  • Both are open-source and free\n")

    cli.console.print("[green]Why validate with external tools?[/green]")
    cli.console.print("  • Review and correct LLM annotations")
    cli.console.print("  • Calculate inter-annotator agreement")
    cli.console.print("  • Export validated data for metrics calculation\n")

    # Initialize export flags
    export_to_doccano = False
    export_to_labelstudio = False
    export_sample_size = None

    # Step 1: Ask if user wants to export
    export_confirm = Confirm.ask(
        "[bold yellow]Export to validation tool?[/bold yellow]",
        default=False
    )

    if export_confirm:
        # Step 2: Ask which tool to export to
        tool_choice = Prompt.ask(
            "[bold yellow]Which validation tool?[/bold yellow]",
            choices=["doccano", "labelstudio"],
            default="doccano"
        )

        # Set the appropriate export flag
        if tool_choice == "doccano":
            export_to_doccano = True
        else:  # labelstudio
            export_to_labelstudio = True

        # Step 2b: If Label Studio, ask export method
        labelstudio_direct_export = False
        labelstudio_api_url = None
        labelstudio_api_key = None

        if export_to_labelstudio:
            cli.console.print("\n[yellow]Label Studio export method:[/yellow]")
            cli.console.print("  • [cyan]jsonl[/cyan] - Export to JSONL file (manual import)")
            if cli.HAS_REQUESTS:
                cli.console.print("  • [cyan]direct[/cyan] - Direct export to Label Studio via API\n")
                export_choices = ["jsonl", "direct"]
            else:
                cli.console.print("  • [dim]direct[/dim] - Direct export via API [dim](requires 'requests' library)[/dim]\n")
                export_choices = ["jsonl"]

            export_method = Prompt.ask(
                "[bold yellow]Export method[/bold yellow]",
                choices=export_choices,
                default="jsonl"
            )

            if export_method == "direct":
                labelstudio_direct_export = True

                cli.console.print("\n[cyan]Label Studio API Configuration:[/cyan]")
                labelstudio_api_url = Prompt.ask(
                    "Label Studio URL",
                    default="http://localhost:8080"
                )

                labelstudio_api_key = Prompt.ask(
                    "API Key (from Label Studio Account & Settings)"
                )

        # Step 3: Ask about LLM predictions inclusion
        cli.console.print("\n[yellow]Include LLM predictions in export?[/yellow]")
        cli.console.print("  • [cyan]with[/cyan] - Include LLM annotations as predictions (for review/correction)")
        cli.console.print("  • [cyan]without[/cyan] - Export only data without predictions (for manual annotation)")
        cli.console.print("  • [cyan]both[/cyan] - Create two files: one with and one without predictions\n")

        prediction_mode = Prompt.ask(
            "[bold yellow]Prediction mode[/bold yellow]",
            choices=["with", "without", "both"],
            default="with"
        )

        # Step 4: Ask how many sentences to export
        cli.console.print("\n[yellow]How many annotated sentences to export?[/yellow]")
        cli.console.print("  • [cyan]all[/cyan] - Export all annotated sentences")
        cli.console.print("  • [cyan]representative[/cyan] - Representative sample (stratified by labels)")
        cli.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

        sample_choice = Prompt.ask(
            "[bold yellow]Export sample[/bold yellow]",
            choices=["all", "representative", "number"],
            default="all"
        )

        if sample_choice == "all":
            export_sample_size = "all"
        elif sample_choice == "representative":
            export_sample_size = "representative"
        else:  # number
            export_sample_size = cli._int_prompt_with_validation(
                "Number of sentences to export",
                100,
                1,
                999999
            )

    # ============================================================
    # EXECUTE ANNOTATION
    # ============================================================

    # CRITICAL: Use new organized structure with dataset-specific subfolder
    # Structure: logs/annotator/{session_id}/annotated_data/{dataset_name}/
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine annotation mode before creating directories
    if provider == 'openai' and openai_batch_mode:
        annotation_mode = 'openai_batch'
    else:
        annotation_mode = 'api' if provider in {'openai', 'anthropic', 'google'} else 'local'

    if annotation_mode == 'openai_batch':
        annotation_mode_display = "OpenAI Batch"
    elif annotation_mode == 'api':
        annotation_mode_display = "API"
    else:
        annotation_mode_display = "Local"

    provider_folder = (provider or "model_provider").replace("/", "_")
    model_folder = safe_model_name

    provider_subdir = session_dirs['annotated_data'] / provider_folder
    provider_subdir.mkdir(parents=True, exist_ok=True)

    model_subdir = provider_subdir / model_folder
    model_subdir.mkdir(parents=True, exist_ok=True)

    dataset_name = data_path.stem
    dataset_subdir = model_subdir / dataset_name
    dataset_subdir.mkdir(parents=True, exist_ok=True)

    if annotation_mode == 'openai_batch':
        batch_logs_root = model_subdir / "openai_batch_jobs"
        batch_logs_root.mkdir(parents=True, exist_ok=True)
        batch_dir = batch_logs_root / timestamp
        batch_dir.mkdir(parents=True, exist_ok=True)

        legacy_root = Path(session_dirs.get('openai_batches', session_dirs['annotated_data']))
        legacy_target = legacy_root / provider_folder / model_folder / dataset_name
        legacy_target.mkdir(parents=True, exist_ok=True)
        pointer_file = legacy_target / f"{timestamp}_LOCATION.txt"
        if not pointer_file.exists():
            try:
                pointer_file.write_text(
                    f"OpenAI batch artifacts moved to: {batch_dir}\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
    else:
        batch_dir = dataset_subdir

    output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
    default_output_path = dataset_subdir / output_filename

    cli.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")
    cli.console.print()

    # Prepare prompts payload for pipeline
    prompts_payload = []
    for pc in prompt_configs:
        prompts_payload.append({
            'prompt': pc['prompt']['content'],
            'expected_keys': pc['prompt']['keys'],
            'prefix': pc['prefix']
        })

    # Build pipeline config
    pipeline_config = {
        'mode': 'file',
        'data_source': data_format,
        'data_format': data_format,
        'file_path': str(data_path),
        'text_column': text_column,
        'text_columns': [text_column],
        'annotation_column': 'annotation',
        'identifier_column': identifier_column,  # From Step 2b: User-selected ID strategy
        'run_annotation': True,
        'annotation_mode': annotation_mode,
        'annotation_provider': provider,
        'annotation_model': model_name,
        'api_key': api_key if api_key else None,
        'openai_batch_mode': openai_batch_mode,
        'prompts': prompts_payload,
        'annotation_sample_size': annotation_limit,
        'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        'annotation_sample_seed': 42,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
        'max_workers': num_processes,
        'num_processes': num_processes,
        'use_parallel': num_processes > 1,
        'warmup': False,
        'disable_tqdm': True,  # Use Rich progress instead
        'output_format': data_format,
        'output_path': str(default_output_path),
        'save_incrementally': save_incrementally,
        'batch_size': batch_size,
        'run_validation': False,
        'run_training': False,
        'lang_column': lang_column,  # From Step 4b: Language column for training metadata
        'create_annotated_subset': True,
    }

    pipeline_config.update({
        'session_dirs': {key: str(value) for key, value in session_dirs.items()},
        'provider_folder': provider_folder,
        'model_folder': model_folder,
        'dataset_name': dataset_name,
        'provider_subdir': str(provider_subdir),
        'model_subdir': str(model_subdir),
        'dataset_subdir': str(dataset_subdir),
        'openai_batch_dir': str(batch_dir),
    })

    if annotation_mode == 'openai_batch':
        pipeline_config.setdefault('openai_batch_poll_interval', 5)
        pipeline_config.setdefault('openai_batch_completion_window', '24h')

    # Add model-specific options
    if provider == 'ollama':
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
            'top_k': top_k
        }
        pipeline_config['options'] = options

    # ============================================================
    # SAVE REPRODUCIBILITY METADATA
    # ============================================================
    if save_metadata:
        import json

        # Build comprehensive metadata
        available_rows = total_rows if total_rows is not None else None
        if annotation_limit:
            requested_rows = annotation_limit
        elif available_rows is not None:
            requested_rows = available_rows
        else:
            requested_rows = 'all'
        initial_remaining = (
            requested_rows
            if isinstance(requested_rows, int)
            else (available_rows if available_rows is not None else 'all')
        )
        metadata = {
            'annotation_session': {
                'timestamp': timestamp,
                'tool_version': 'LLMTool v1.0',
                'workflow': 'The Annotator - Smart Annotate'
            },
            'data_source': {
                'file_path': str(data_path),
                'file_name': data_path.name,
                'data_format': data_format,
                'text_column': text_column,
                'dataset_name': dataset_name,
                'total_rows': annotation_limit if annotation_limit else 'all',
                'requested_rows': requested_rows,
                'available_rows': available_rows,
                'sampling_strategy': sample_strategy if annotation_limit else 'none (all rows)',
                'sample_seed': 42 if sample_strategy == 'random' else None
            },
            'annotation_progress': {
                'requested': requested_rows,
                'completed': 0,
                'remaining': initial_remaining,
            },
            'model_configuration': {
                'provider': provider,
                'model_name': model_name,
                'annotation_mode': annotation_mode,
                'openai_batch_mode': openai_batch_mode,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'top_k': top_k if provider in ['ollama', 'google'] else None
            },
            'prompts': [
                {
                    'name': pc['prompt']['name'],
                    'file_path': str(pc['prompt']['path']) if 'path' in pc['prompt'] else None,
                    'expected_keys': pc['prompt']['keys'],
                    'prefix': pc['prefix'],
                    'prompt_content': pc['prompt']['content']
                }
                for pc in prompt_configs
            ],
            'processing_configuration': {
                'parallel_workers': None if annotation_mode == 'openai_batch' else num_processes,
                'batch_size': None if annotation_mode == 'openai_batch' else batch_size,
                'incremental_save': False if annotation_mode == 'openai_batch' else save_incrementally,
                'openai_batch_mode': openai_batch_mode,
                'openai_batch_dir': str(batch_dir) if annotation_mode == 'openai_batch' else None,
                'identifier_column': auto_identifier_column,
                'provider_folder': provider_folder,
                'model_folder': model_folder,
                'dataset_name': dataset_name
            },
            'output': {
                'output_path': str(default_output_path),
                'output_format': data_format
            },
            'export_preferences': {
                'export_to_doccano': export_to_doccano,
                'export_to_labelstudio': export_to_labelstudio,
                'export_sample_size': export_sample_size,
                'prediction_mode': prediction_mode if (export_to_doccano or export_to_labelstudio) else 'with',
                'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
                'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
                'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None
            },
            'training_workflow': {
                'enabled': False,  # Will be updated after training workflow
                'training_params_file': None,  # Will be added after training
                'note': 'Training parameters will be saved separately after annotation completes'
            }
        }

        # Save metadata JSON (PRE-ANNOTATION SAVE POINT 1)
        # Use dataset-specific subdirectory for metadata too
        metadata_subdir = session_dirs['metadata'] / provider_folder / model_folder / dataset_name
        metadata_subdir.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
        metadata_path = metadata_subdir / metadata_filename

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[bold green]✅ Metadata saved for reproducibility[/bold green]")
        cli.console.print(f"[bold cyan]📋 Metadata File:[/bold cyan]")
        cli.console.print(f"   {metadata_path}\n")

    # Execute pipeline with Rich progress
    try:
        cli.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

        # Create pipeline controller with session_id for organized logging
        from ..pipelines.pipeline_controller import PipelineController
        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=session_id  # Pass session_id for organized logging
        )

        # Use RichProgressManager for elegant display
        from ..utils.rich_progress_manager import RichProgressManager
        from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

        with RichProgressManager(
            show_json_every=1,  # Show JSON sample for every annotation
            compact_mode=False   # Full preview panels
        ) as progress_manager:
            # Wrap pipeline for enhanced JSON tracking
            enhanced_pipeline = EnhancedPipelineWrapper(
                pipeline_with_progress,
                progress_manager
            )

            # Run pipeline
            state = enhanced_pipeline.run_pipeline(pipeline_config)

            # Check for errors
            if state.errors:
                error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                cli.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                cli.console.print("[dim]Press Enter to return to menu...[/dim]")
                input()
                return

        # Get results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        # Display success message
        cli.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        # Display statistics if available
        total_annotated = annotation_results.get('total_annotated', 0)
        if total_annotated:
            cli.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

            mean_time = annotation_results.get('mean_inference_time')
            if isinstance(mean_time, (int, float)):
                cli.console.print(f"   Avg inference time: {mean_time:.2f}s")

        subset_path = annotation_results.get('annotated_subset_path')
        if subset_path:
            cli.console.print(f"   Annotated subset: {subset_path}")

        batch_output = annotation_results.get('openai_batch_output_path')
        if batch_output:
            cli.console.print(f"   Batch output JSONL: {batch_output}")

        batch_metadata_path = annotation_results.get('openai_batch_metadata_path')
        if batch_metadata_path:
            cli.console.print(f"   Batch metadata: {batch_metadata_path}")

        preview_samples = annotation_results.get('preview_samples') or []
        if preview_samples:
            cli.console.print("\n[bold cyan]📝 Sample Annotations:[/bold cyan]")
            preview_keys = list(preview_samples[0].keys())
            preview_table = Table(show_header=True, header_style="bold magenta")
            for key in preview_keys:
                preview_table.add_column(key.replace('_', ' ').title(), overflow="fold")
            for sample in preview_samples:
                preview_table.add_row(*[str(sample.get(key, '')) for key in preview_keys])
            cli.console.print(preview_table)

        tracker.mark_step(
            6,
            detail=f"Annotated rows: {total_annotated or 'n/a'}",
            extra={
                "output_file": output_file,
                "total_annotated": total_annotated,
                "success_count": annotation_results.get('success_count'),
            },
        )

        # ============================================================
        # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
        # ============================================================
        if not lang_column:
            cli.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
            cli.console.print("[yellow]No language column was provided. Detecting languages for training...[/yellow]\n")

            try:
                import pandas as pd
                from llm_tool.utils.language_detector import LanguageDetector

                # Load annotated file
                df_annotated = pd.read_csv(output_file)

                # CRITICAL: Only detect languages for ANNOTATED rows
                # The output file may contain ALL original rows, but we only want to detect
                # languages for rows that were actually annotated
                original_row_count = len(df_annotated)

                # Try to identify annotated rows by checking for annotation columns
                # Common annotation column names: 'label', 'category', 'annotation', 'labels'
                annotation_cols = [col for col in df_annotated.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]

                if annotation_cols:
                    # Filter to only rows that have annotations (non-null AND non-empty in annotation column)
                    annotation_col = annotation_cols[0]
                    df_annotated = df_annotated[(df_annotated[annotation_col].notna()) & (df_annotated[annotation_col] != '')].copy()
                    cli.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows (out of {original_row_count:,} total rows in file)[/dim]")
                else:
                    cli.console.print(f"[yellow]⚠️  Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                if len(df_annotated) == 0:
                    cli.console.print("[yellow]⚠️  No annotated rows found. Skipping language detection.[/yellow]")
                elif text_column in df_annotated.columns:
                    # Get ALL texts (including NaN) to maintain index alignment
                    all_texts = df_annotated[text_column].tolist()

                    # Count non-empty texts for display
                    non_empty_texts = sum(1 for text in all_texts if pd.notna(text) and len(str(text).strip()) > 10)

                    if non_empty_texts > 0:
                        detector = LanguageDetector()
                        detected_languages = []

                        # Progress indicator
                        from tqdm import tqdm
                        cli.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                        for text in tqdm(all_texts, desc="Detecting languages", disable=not cli.HAS_RICH):
                            # Handle NaN and empty texts
                            if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                                detected_languages.append('unknown')
                            else:
                                try:
                                    detected = detector.detect(str(text))
                                    if detected and detected.get('language'):
                                        detected_languages.append(detected['language'])
                                    else:
                                        detected_languages.append('unknown')
                                except Exception as e:
                                    cli.logger.debug(f"Language detection failed for text: {e}")
                                    detected_languages.append('unknown')

                        # Add language column to the filtered dataframe
                        df_annotated['lang'] = detected_languages

                        # Reload the FULL original file and update only the annotated rows
                        df_full = pd.read_csv(output_file)

                        # Initialize lang column if it doesn't exist
                        if 'lang' not in df_full.columns:
                            df_full['lang'] = 'unknown'

                        # Update language for annotated rows only
                        # Match by index of df_annotated within df_full
                        df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values

                        # Save updated full file with language column
                        df_full.to_csv(output_file, index=False)

                        # Show distribution
                        lang_counts = {}
                        for lang in detected_languages:
                            if lang != 'unknown':
                                lang_counts[lang] = lang_counts.get(lang, 0) + 1

                        if lang_counts:
                            total = sum(lang_counts.values())
                            cli.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                            lang_table.add_column("Language", style="cyan", width=12)
                            lang_table.add_column("Count", style="yellow", justify="right", width=12)
                            lang_table.add_column("Percentage", style="green", justify="right", width=12)

                            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total * 100) if total > 0 else 0
                                lang_table.add_row(
                                    lang.upper(),
                                    f"{count:,}",
                                    f"{percentage:.1f}%"
                                )

                            cli.console.print(lang_table)
                            cli.console.print(f"\n[green]✓ Language column 'lang' added to {output_file}[/green]")
                        else:
                            cli.console.print("[yellow]⚠️  No languages detected successfully[/yellow]")

            except Exception as e:
                cli.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
                cli.logger.exception("Language detection failed")

        # ============================================================
        # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
        # ============================================================
        training_results = cli._post_annotation_training_workflow(
            output_file=output_file,
            text_column=text_column,
            prompt_configs=prompt_configs,
            session_id=session_id,
            session_dirs=session_dirs  # Pass session directories for organized logging
        )
        tracker.mark_step(
            8,
            detail="Training workflow executed",
            extra={"training_session_id": session_id},
        )

        model_annotation_summary = _launch_model_annotation_stage(
            cli,
            session_id=session_id,
            session_dirs=session_dirs,
            training_results=training_results,
            prompt_configs=prompt_configs,
            text_column=text_column,
            annotation_output=output_file,
            dataset_path=data_path,
        )

        if isinstance(model_annotation_summary, dict):
            status = model_annotation_summary.get("status", "completed")
            detail = model_annotation_summary.get("detail")
            extra = model_annotation_summary.get("extra")
            overall_status = status if status in {"failed", "cancelled"} else None
            tracker.mark_step(
                9,
                status=status,
                detail=detail,
                overall_status=overall_status,
                extra=extra,
            )
        else:
            tracker.mark_step(
                9,
                status="skipped",
                detail="Model annotation stage skipped",
            )

        # Export to Doccano JSONL if requested
        if export_to_doccano:
            cli._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs,
                data_path=data_path,
                timestamp=timestamp,
                sample_size=export_sample_size,
                session_dirs=session_dirs,
                provider_folder=provider_folder,
                model_folder=model_folder
            )

        # Export to Label Studio if requested
        if export_to_labelstudio:
            if labelstudio_direct_export:
                # Direct export to Label Studio via API
                cli._export_to_labelstudio_direct(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    api_url=labelstudio_api_url,
                    api_key=labelstudio_api_key
                )
            else:
                # Export to JSONL file
                cli._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    session_dirs=session_dirs,
                    provider_folder=provider_folder,
                    model_folder=model_folder
                )

        tracker.mark_step(
            10,
            detail="Factory workflow complete",
            extra={
                "export_doccano": export_to_doccano,
                "export_labelstudio": export_to_labelstudio,
                "prediction_mode": prediction_mode,
            },
        )
        tracker.update_status("completed")

        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    except Exception as exc:
        tracker.update_status("failed", note=str(exc))
        cli.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Annotation execution failed")
        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

def execute_from_metadata(cli, metadata: dict, action_mode: str, metadata_file: Path, session_dirs: Optional[Dict[str, Path]] = None):
    """Execute annotation based on loaded metadata"""
    import json
    from datetime import datetime

    # Extract all parameters from metadata
    data_source = metadata.get('data_source', {})
    model_config = metadata.get('model_configuration', {})
    prompts = metadata.get('prompts', [])
    proc_config = metadata.get('processing_configuration', {})
    output_config = metadata.get('output', {})
    export_prefs = metadata.get('export_preferences', {})

    identifier_column = (
        proc_config.get('identifier_column')
        or data_source.get('identifier_column')
    )
    if not identifier_column:
        identifier_column = 'llm_annotation_id'
    elif identifier_column == 'annotation_id':
        identifier_column = 'llm_annotation_id'

    pipeline_identifier_column = (
        None if identifier_column == 'llm_annotation_id' else identifier_column
    )
    default_identifier = identifier_column
    data_source.setdefault('identifier_column', identifier_column)
    proc_config.setdefault('identifier_column', identifier_column)

    # Get export preferences
    export_to_doccano = export_prefs.get('export_to_doccano', False)
    export_to_labelstudio = export_prefs.get('export_to_labelstudio', False)
    export_sample_size = export_prefs.get('export_sample_size', 'all')

    if export_to_doccano or export_to_labelstudio:
        export_tools = []
        if export_to_doccano:
            export_tools.append("Doccano")
        if export_to_labelstudio:
            export_tools.append("Label Studio")
        cli.console.print(f"\n[cyan]ℹ️  Export enabled for: {', '.join(export_tools)} (from saved preferences)[/cyan]")
        if export_sample_size != 'all':
            cli.console.print(f"[cyan]   Sample size: {export_sample_size}[/cyan]")

    # Prepare paths
    data_path = Path(data_source.get('file_path', ''))
    data_format = data_source.get('data_format', 'csv')
    dataset_stem = data_path.stem
    if not dataset_stem:
        dataset_stem = Path(data_source.get('file_name', '') or "dataset").stem or "dataset"

    metadata_root = None
    if session_dirs:
        metadata_root = session_dirs['metadata'] / dataset_stem
        metadata_root.mkdir(parents=True, exist_ok=True)

    # Check if resuming
    if action_mode == 'resume':
        # Try to find the output file
        original_output = Path(output_config.get('output_path', ''))

        if not original_output.exists():
            cli.console.print(f"\n[yellow]⚠️  Output file not found: {original_output}[/yellow]")
            cli.console.print("[yellow]Switching to relaunch mode (fresh annotation)[/yellow]")
            action_mode = 'relaunch'
        else:
            cli.console.print(f"\n[green]✓ Found output file: {original_output.name}[/green]")

            # Count already annotated rows
            import pandas as pd
            try:
                if data_format == 'csv':
                    df_output = pd.read_csv(original_output)
                elif data_format in ['excel', 'xlsx']:
                    df_output = pd.read_excel(original_output)
                elif data_format == 'parquet':
                    df_output = pd.read_parquet(original_output)

                resolved_identifier = None
                candidate_identifiers = [
                    identifier_column,
                    data_source.get('identifier_column'),
                    proc_config.get('identifier_column'),
                    'annotation_id',
                    'llm_annotation_id',
                    'llm_annotation_uuid',
                ]
                candidate_identifiers = [
                    cand for cand in candidate_identifiers if isinstance(cand, str)
                ]
                resolved_identifier = next(
                    (
                        cand for cand in candidate_identifiers
                        if cand in df_output.columns
                    ),
                    None
                )
                if resolved_identifier is None:
                    resolved_identifier = next(
                        (
                            col for col in df_output.columns
                            if isinstance(col, str) and col.endswith('annotation_id')
                        ),
                        None
                    )
                if not resolved_identifier:
                    cli.console.print(
                        "[yellow]⚠️  Could not determine identifier column from resume file.[/yellow]"
                    )

                # Count rows with valid annotations (non-empty, non-null strings)
                if 'annotation' in df_output.columns:
                    # Count only rows where annotation exists and is not empty/whitespace
                    annotated_mask = (
                        df_output['annotation'].notna() &
                        (df_output['annotation'].astype(str).str.strip() != '') &
                        (df_output['annotation'].astype(str) != 'nan')
                    )
                    annotated_count = annotated_mask.sum()
                else:
                    annotated_count = 0

                cli.console.print(f"[cyan]  Rows already annotated: {annotated_count:,}[/cyan]\n")

                # Get total available rows from source file
                if action_mode == 'resume':
                    chosen_identifier = resolved_identifier or default_identifier
                    identifier_column = chosen_identifier
                    data_source['identifier_column'] = identifier_column
                    proc_config['identifier_column'] = identifier_column
                    cli.console.print(f"[dim]Identifier column resolved: {identifier_column}[/dim]")

                    if data_path.exists():
                        if data_format == 'csv':
                            total_available = len(pd.read_csv(data_path))
                        elif data_format in ['excel', 'xlsx']:
                            total_available = len(pd.read_excel(data_path))
                        elif data_format == 'parquet':
                            total_available = len(pd.read_parquet(data_path))
                    else:
                        total_available = len(df_output)
                else:
                    total_available = len(df_output)

                    # Track progress metadata
                    data_source['available_rows'] = total_available
                    progress = metadata.setdefault('annotation_progress', {})
                    requested_candidates = [
                        progress.get('requested'),
                        data_source.get('requested_rows'),
                        data_source.get('total_rows'),
                    ]
                    requested_total: Optional[int] = None
                    requested_was_all = False
                    for candidate in requested_candidates:
                        if candidate is None:
                            continue
                        if isinstance(candidate, str) and candidate.strip().lower() == 'all':
                            requested_was_all = True
                            continue
                        coerced = _coerce_to_int(candidate)
                        if coerced is not None:
                            requested_total = coerced
                            break
                    if requested_total is None:
                        if requested_was_all and total_available:
                            requested_total = total_available
                        elif total_available:
                            requested_total = total_available
                    if requested_total is not None and requested_total < annotated_count:
                        requested_total = annotated_count

                    remaining_from_target = (
                        max(requested_total - annotated_count, 0)
                        if requested_total is not None else None
                    )
                    remaining_from_source = max(total_available - annotated_count, 0)

                    progress['requested'] = (
                        requested_total if requested_total is not None else 'all'
                    )
                    progress['completed'] = annotated_count
                    progress['last_completed'] = annotated_count
                    progress['available'] = total_available
                    progress['remaining'] = (
                        remaining_from_target
                        if remaining_from_target is not None
                        else remaining_from_source
                    )
                    data_source['requested_rows'] = progress['requested']
                    pipeline_config['annotation_requested_total'] = (
                        requested_total if requested_total is not None else 'all'
                    )
                    pipeline_config['annotation_total_available'] = total_available
                    pipeline_config['annotation_completed_rows'] = annotated_count

                    display_total = requested_total if requested_total is not None else total_available
                    display_total_text = (
                        f"{display_total:,}" if display_total is not None else "?"
                    )
                    target_completed = (
                        remaining_from_target is not None and remaining_from_target <= 0
                    )
                    if target_completed:
                        remaining_display = remaining_from_source
                    else:
                        remaining_display = (
                            remaining_from_target
                            if remaining_from_target is not None
                            else remaining_from_source
                        )

                    cli.console.print(
                        f"[cyan]  Progress: {annotated_count:,} / {display_total_text} annotated[/cyan]"
                    )
                    cli.console.print(
                        f"[cyan]  Remaining in request: {remaining_display:,}[/cyan]"
                    )
                    cli.console.print(
                        f"[cyan]  Total available in source: {total_available:,}[/cyan]\n"
                    )

                    if remaining_from_source <= 0:
                        cli.console.print("\n[yellow]All available rows are already annotated![/yellow]")
                        continue_anyway = Confirm.ask("Continue with relaunch mode?", default=False)
                        if not continue_anyway:
                            return False
                        action_mode = 'relaunch'
                    else:
                        next_annotation = annotated_count + 1
                        if target_completed and remaining_from_source > 0:
                            prompt_text = (
                                f"You already reached the requested {display_total_text} rows. "
                                f"Annotate the remaining {remaining_from_source:,} row(s) from the source?"
                            )
                        else:
                            prompt_text = (
                                f"Resume from annotation {next_annotation:,} / {display_total_text}? "
                                f"(remaining {remaining_display:,})"
                            )
                        continue_resume = Confirm.ask(
                            f"\n[bold yellow]{prompt_text}[/bold yellow]",
                            default=True,
                        )
                        if not continue_resume:
                            cli.console.print("[yellow]Switching to relaunch mode (fresh annotation)[/yellow]")
                            action_mode = 'relaunch'
                        else:
                            resume_count = (
                                remaining_from_source if target_completed else remaining_display
                            )
                            if target_completed:
                                allow_adjust = remaining_from_source > 0
                            else:
                                allow_adjust = remaining_from_source > resume_count
                            if allow_adjust:
                                adjust_target = Confirm.ask(
                                    "[cyan]Adjust how many rows to annotate in this resume run?[/cyan]",
                                    default=False,
                                )
                                if adjust_target:
                                    safe_default = resume_count if resume_count > 0 else 1
                                    resume_count = cli._int_prompt_with_validation(
                                        f"How many rows to annotate now? (max: {remaining_from_source:,})",
                                        min(100, safe_default),
                                        1,
                                        remaining_from_source,
                                    )

                            # Update metadata for resume
                            metadata['data_source']['total_rows'] = resume_count
                            metadata['resume_mode'] = True
                            metadata['resume_from_file'] = str(original_output)
                            metadata['already_annotated'] = int(annotated_count)
                            progress['resume_next'] = resume_count
                            if isinstance(progress.get('remaining'), int):
                                progress['remaining_after_resume'] = max(
                                    progress['remaining'] - resume_count, 0
                                )

                            # Use the previously annotated file as input so we continue seamlessly.
                            data_path = original_output
                            data_source['file_path'] = str(original_output)
                            data_source['file_name'] = original_output.name
                            resume_ext = original_output.suffix.lower().lstrip('.')
                            if resume_ext:
                                if resume_ext in {'xls', 'xlsx'}:
                                    data_format = 'excel'
                                else:
                                    data_format = resume_ext
                                data_source['data_format'] = data_format

            except Exception as e:
                cli.console.print(f"\n[red]Error reading output file: {e}[/red]")
                cli.console.print("[yellow]Switching to relaunch mode[/yellow]")
                action_mode = 'relaunch'

    # Prepare output path
    annotations_dir = cli.settings.paths.data_dir / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_config.get('model_name', 'unknown').replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if action_mode == 'resume':
        output_filename = original_output.name  # Keep same filename
        default_output_path = original_output
    else:
        output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
        default_output_path = annotations_dir / output_filename
        identifier_column = default_identifier
        data_source['identifier_column'] = identifier_column
        proc_config['identifier_column'] = identifier_column

    cli.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")

    # Prepare prompts payload
    prompts_payload = []
    for p in prompts:
        prompts_payload.append({
            'prompt': p.get('prompt_content', p.get('prompt', '')),
            'expected_keys': p.get('expected_keys', []),
            'prefix': p.get('prefix', '')
        })

    # Get parameters
    provider = model_config.get('provider', 'ollama')
    model_name = model_config.get('model_name', 'llama2')
    annotation_mode = model_config.get('annotation_mode', 'local')
    openai_batch_mode = model_config.get('openai_batch_mode', annotation_mode == 'openai_batch')
    temperature = model_config.get('temperature', 0.7)
    max_tokens = model_config.get('max_tokens', 1000)
    top_p = model_config.get('top_p', 1.0)
    top_k = model_config.get('top_k', 40)

    num_processes = proc_config.get('parallel_workers', 1)
    batch_size = proc_config.get('batch_size', 1)

    total_rows = data_source.get('total_rows')
    annotation_limit = None if total_rows == 'all' else total_rows
    sample_strategy = data_source.get('sampling_strategy', 'head')

    # IMPORTANT: In resume mode, always use 'head' strategy to continue sequentially
    # This ensures we pick up exactly where we left off, not random new rows
    if action_mode == 'resume':
        sample_strategy = 'head'
        cli.console.print(f"\n[cyan]ℹ️  Resume mode: Using sequential (head) strategy to continue where you left off[/cyan]")

    # Get API key if needed
    api_key = None
    if provider in ['openai', 'anthropic', 'google']:
        api_key = cli._get_api_key(provider)
        if not api_key:
            cli.console.print(f"[red]API key required for {provider}[/red]")
            return False

    # Build pipeline config
    pipeline_config = {
        'mode': 'file',
        'data_source': data_format,
        'data_format': data_format,
        'file_path': str(data_path),
        'text_column': data_source.get('text_column', 'text'),
        'text_columns': [data_source.get('text_column', 'text')],
        'annotation_column': 'annotation',
        'identifier_column': pipeline_identifier_column,
        'run_annotation': True,
        'annotation_mode': annotation_mode,
        'annotation_provider': provider,
        'annotation_model': model_name,
        'api_key': api_key,
        'openai_batch_mode': openai_batch_mode,
        'prompts': prompts_payload,
        'annotation_sample_size': annotation_limit,
        'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        'annotation_sample_seed': 42,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
        'max_workers': num_processes,
        'num_processes': num_processes,
        'use_parallel': num_processes > 1,
        'warmup': False,
        'disable_tqdm': True,
        'output_format': data_format,
        'output_path': str(default_output_path),
        'save_incrementally': True,
        'batch_size': batch_size,
        'run_validation': False,
        'run_training': False,
    }

    # Add resume information if resuming
    if action_mode == 'resume' and metadata.get('resume_mode'):
        pipeline_config['resume_mode'] = True
        pipeline_config['resume_from_file'] = metadata.get('resume_from_file')
        pipeline_config['skip_annotated'] = True
        pipeline_config['resume'] = True

        # Load already annotated IDs to skip them
        try:
            import pandas as pd
            resume_file = Path(metadata.get('resume_from_file'))
            if resume_file.exists():
                if data_format == 'csv':
                    df_resume = pd.read_csv(resume_file)
                elif data_format in ['excel', 'xlsx']:
                    df_resume = pd.read_excel(resume_file)
                elif data_format == 'parquet':
                    df_resume = pd.read_parquet(resume_file)

                # Get IDs of rows that have valid annotations
                id_column = (
                    identifier_column if identifier_column in df_resume.columns else None
                )
                if not id_column:
                    for candidate in ('annotation_id', 'llm_annotation_id', 'llm_annotation_uuid'):
                        if candidate in df_resume.columns:
                            id_column = candidate
                            break

                if 'annotation' in df_resume.columns and id_column:
                    annotated_mask = (
                        df_resume['annotation'].notna() &
                        (df_resume['annotation'].astype(str).str.strip() != '') &
                        (df_resume['annotation'].astype(str) != 'nan')
                    )
                    already_annotated_ids = df_resume.loc[annotated_mask, id_column].tolist()
                    pipeline_config['skip_annotation_ids'] = already_annotated_ids

                    cli.console.print(f"[cyan]  Will skip {len(already_annotated_ids)} already annotated row(s)[/cyan]")
        except Exception as e:
            cli.logger.warning(f"Could not load annotated IDs from resume file: {e}")
            cli.console.print(f"[yellow]⚠️  Warning: Could not load annotated IDs - may re-annotate some rows[/yellow]")

    # Add model-specific options
    if provider == 'ollama':
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
            'top_k': top_k
        }
        pipeline_config['options'] = options

    # Save new metadata for this execution
    if action_mode == 'relaunch' and metadata_root is not None:
        new_metadata = copy.deepcopy(metadata)
        new_metadata['annotation_session']['timestamp'] = timestamp
        new_metadata['annotation_session']['relaunch_from'] = str(metadata_file.name)
        new_metadata['annotation_session']['action_mode'] = 'relaunch'
        new_metadata['session_id'] = session_id
        if 'output' not in new_metadata:
            new_metadata['output'] = {}
        new_metadata['output']['output_path'] = str(default_output_path)

        new_metadata_filename = f"{dataset_stem}_{safe_model_name}_metadata_{timestamp}.json"
        new_metadata_path = metadata_root / new_metadata_filename

        with open(new_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[green]✅ New session metadata saved[/green]")
        cli.console.print(f"[cyan]📋 Metadata File:[/cyan]")
        cli.console.print(f"   {new_metadata_path}\n")

    if action_mode == 'resume' and metadata_root is not None:
        resume_metadata = copy.deepcopy(metadata)
        resume_metadata['annotation_session']['timestamp'] = timestamp
        resume_metadata['annotation_session']['resume_from'] = str(metadata_file.name)
        resume_metadata['annotation_session']['action_mode'] = 'resume'
        resume_metadata['session_id'] = session_id
        if 'output' not in resume_metadata:
            resume_metadata['output'] = {}
        resume_metadata['output']['output_path'] = str(default_output_path)
        resume_metadata['resume_mode'] = True

        resume_metadata_filename = f"{dataset_stem}_{safe_model_name}_resume_{timestamp}.json"
        resume_metadata_path = metadata_root / resume_metadata_filename

        with open(resume_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(resume_metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[green]✅ Resume metadata saved[/green]")
        cli.console.print(f"[cyan]📋 Metadata File:[/cyan]")
        cli.console.print(f"   {resume_metadata_path}\n")

    # Execute pipeline
    try:
        cli.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")
        cli.console.print(f"[dim]Identifier column in use: {identifier_column}[/dim]")

        from ..pipelines.pipeline_controller import PipelineController

        # Extract session_id from metadata or create new one
        resume_session_id = metadata.get("session_id") if metadata else None

        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=resume_session_id  # Pass session_id for organized logging
        )

        from ..utils.rich_progress_manager import RichProgressManager
        from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

        with RichProgressManager(
            show_json_every=1,
            compact_mode=False
        ) as progress_manager:
            enhanced_pipeline = EnhancedPipelineWrapper(
                pipeline_with_progress,
                progress_manager
            )

            state = enhanced_pipeline.run_pipeline(pipeline_config)

            if state.errors:
                error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                cli.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                return False

        # Display results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        cli.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        total_annotated = annotation_results.get('total_annotated', 0)
        if total_annotated:
            cli.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

        # ============================================================
        # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
        # ============================================================
        # Build prompt_configs for training workflow
        prompt_configs_for_training = []
        for p in prompts:
            prompt_configs_for_training.append({
                'prompt': {
                    'keys': p.get('expected_keys', []),
                    'content': p.get('prompt_content', p.get('prompt', '')),
                    'name': p.get('name', 'prompt')
                },
                'prefix': p.get('prefix', '')
            })

        training_results = cli._post_annotation_training_workflow(
            output_file=output_file,
            text_column=data_source.get('text_column', 'text'),
            prompt_configs=prompt_configs_for_training,
            session_id=session_id,
            session_dirs=None  # No session_dirs in resume context
        )

        _launch_model_annotation_stage(
            cli,
            session_id=session_id,
            session_dirs=None,
            training_results=training_results,
            prompt_configs=prompt_configs_for_training,
            text_column=data_source.get('text_column', 'text'),
            annotation_output=output_file,
            dataset_path=data_path,
        )

        # Export to Doccano JSONL if enabled in preferences
        if export_to_doccano:
            # Build prompt_configs for export
            prompt_configs_for_export = []
            for p in prompts:
                prompt_configs_for_export.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt_content', p.get('prompt', '')),
                        'name': p.get('name', 'prompt')
                    },
                    'prefix': p.get('prefix', '')
                })

            provider_folder_resume = (provider or "model_provider").replace("/", "_")

            cli._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=data_source.get('text_column', 'text'),
                prompt_configs=prompt_configs_for_export,
                data_path=data_path,
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                sample_size=export_sample_size,
                session_dirs=session_dirs,
                provider_folder=provider_folder_resume
            )

        # Export to Label Studio JSONL if enabled in preferences
        if export_to_labelstudio:
            # Build prompt_configs for export
            prompt_configs_for_export = []
            for p in prompts:
                prompt_configs_for_export.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt_content', p.get('prompt', '')),
                        'name': p.get('name', 'prompt')
                    },
                    'prefix': p.get('prefix', '')
                })

            cli._export_to_labelstudio_jsonl(
                output_file=output_file,
                text_column=data_source.get('text_column', 'text'),
                prompt_configs=prompt_configs_for_export,
                data_path=data_path,
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                sample_size=export_sample_size,
                prediction_mode=prediction_mode,
                session_dirs=session_dirs,
                provider_folder=provider_folder_resume
            )

    except Exception as exc:
        cli.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Resume/Relaunch annotation failed")
        return False

    return True

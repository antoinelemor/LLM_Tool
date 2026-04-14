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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.panel import Panel
try:
    from rich import box
except ImportError:  # pragma: no cover - optional styling
    box = None

from ..utils.language_detector import LanguageDetector
from llm_tool.utils.data_detector import DataDetector
from llm_tool.utils.session_summary import SessionSummary, merge_summary, read_summary
from llm_tool.utils.cost_estimator import (
    DEFAULT_PROMPT_HEADER,
    CostEstimate,
    ModelPricing,
    estimate_annotation_cost,
    format_cost_estimate_lines,
    resolve_model_pricing,
)
from llm_tool.utils.model_metrics import load_language_metrics, summarize_final_metrics
from llm_tool.annotators.annotation_metrics_chart import generate_annotation_chart


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
TOOL_VERSION = "LLMTool v1.0"


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


def _compose_metadata_payload(
    *,
    session_id: str,
    timestamp: str,
    workflow: str,
    status: str,
    data_source: Dict[str, Any],
    model_configuration: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    processing_configuration: Dict[str, Any],
    output_config: Dict[str, Any],
    export_preferences: Dict[str, Any],
    training_workflow: Optional[Dict[str, Any]] = None,
    progress: Optional[Dict[str, Any]] = None,
    tool_version: str = TOOL_VERSION,
) -> Dict[str, Any]:
    """Build a metadata payload shared by annotator workflows."""
    payload: Dict[str, Any] = {
        "annotation_session": {
            "session_id": session_id,
            "timestamp": timestamp,
            "workflow": workflow,
            "tool_version": tool_version,
            "status": status,
        },
        "data_source": data_source,
        "model_configuration": model_configuration,
        "prompts": prompts,
        "processing_configuration": processing_configuration,
        "output": output_config,
        "export_preferences": export_preferences,
        "training_workflow": training_workflow
        or {
            "enabled": False,
            "training_params_file": None,
            "note": "Training parameters will be saved separately after annotation completes",
        },
    }
    if progress:
        payload["annotation_progress"] = progress
    return payload


def _persist_metadata_snapshot(metadata_path: Path, metadata: Dict[str, Any]) -> None:
    """Write metadata snapshot to disk (overwrite-safe)."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def _persist_metadata_bundle(
    metadata_paths: Union[Path, Iterable[Path]],
    metadata: Dict[str, Any],
) -> None:
    """Write metadata snapshot to one or multiple paths."""
    if isinstance(metadata_paths, Path):
        targets: List[Path] = [metadata_paths]
    else:
        targets = [Path(path) for path in metadata_paths if path]

    seen: Set[Path] = set()
    for target in targets:
        resolved = target.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _persist_metadata_snapshot(resolved, metadata)


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


def _load_dataframe_for_estimation(data_path: Path, data_format: str) -> pd.DataFrame:
    """Load dataset into a DataFrame for cost estimation."""
    import pandas as pd

    suffix = data_format.lower()
    if suffix == 'csv' or data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    if suffix == 'json' or data_path.suffix == '.json':
        return pd.read_json(data_path)
    if suffix == 'jsonl' or data_path.suffix == '.jsonl':
        return pd.read_json(data_path, lines=True)
    if suffix in {'xlsx', 'xls'} or data_path.suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(data_path)
    if suffix == 'parquet' or data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported data format '{data_format}' for cost estimation")


def _apply_annotation_sampling(
    df: pd.DataFrame,
    *,
    annotation_limit: Optional[int],
    sample_strategy: str,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """Apply the same sampling strategy as the annotator."""
    if annotation_limit and len(df) > annotation_limit:
        if sample_strategy == 'random':
            return df.sample(annotation_limit, random_state=sample_seed)
        return df.head(annotation_limit)
    return df


def _compute_cost_estimate_for_selection(
    *,
    provider: str,
    model_name: str,
    data_path: Path,
    data_format: str,
    text_columns: List[str],
    prompt_configs: List[Dict[str, Any]],
    annotation_limit: Optional[int],
    sample_strategy: Optional[str],
    max_tokens: int,
    prompt_header: str = DEFAULT_PROMPT_HEADER,
) -> Tuple[Optional[CostEstimate], Optional[Exception], Optional[ModelPricing]]:
    """Return cost estimate, error (if any), and pricing reference for the selection."""
    pricing = resolve_model_pricing(provider, model_name)
    if pricing is None:
        return None, None, None

    try:
        df_full = _load_dataframe_for_estimation(data_path, data_format)
        df_subset = _apply_annotation_sampling(
            df_full,
            annotation_limit=annotation_limit,
            sample_strategy=sample_strategy or 'head',
            sample_seed=42,
        )

        prompt_payload: List[Dict[str, Any]] = []
        for prompt_cfg in prompt_configs:
            prompt_section = prompt_cfg.get('prompt') or {}
            prompt_text = (
                prompt_section.get('content')
                or prompt_section.get('prompt')
                or prompt_cfg.get('content')
                or ""
            )
            raw_keys = (
                prompt_section.get('keys')
                or prompt_section.get('expected_keys')
                or prompt_cfg.get('expected_keys')
                or []
            )
            if isinstance(raw_keys, Iterable) and not isinstance(raw_keys, (str, bytes)):
                expected_keys = list(raw_keys)
            else:
                expected_keys = []

            prompt_payload.append({
                "prompt": prompt_text,
                "expected_keys": expected_keys,
            })

        cost_estimate = estimate_annotation_cost(
            df_subset=df_subset,
            text_columns=text_columns,
            prompts=prompt_payload,
            pricing=pricing,
            max_output_tokens=max_tokens,
            prompt_header=prompt_header,
        )
        return cost_estimate, None, pricing
    except Exception as exc:  # pragma: no cover - estimation remains best effort
        return None, exc, pricing


def _format_stat_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:,.0f}"


def _render_component_token_tables(console, component_stats: Optional[List[Dict[str, Any]]]) -> None:
    if not console or not component_stats:
        return

    metrics = [
        ("Minimum", "char_min", "token_min"),
        ("25th Percentile", "char_p25", "token_p25"),
        ("Median", "char_median", "token_median"),
        ("Mean", "char_mean", "token_mean"),
        ("75th Percentile", "char_p75", "token_p75"),
        ("95th Percentile", "char_p95", "token_p95"),
        ("Maximum", "char_max", "token_max"),
        ("Std Deviation", "char_std", "token_std"),
    ]

    console.print("\n[bold cyan]Token Breakdown[/bold cyan]\n")

    for component in component_stats:
        stats = component.get("stats") or {}
        table_title = f"{component.get('label', 'Component')} Token Statistics"
        stats_table = Table(
            title=table_title,
            border_style="cyan",
            show_header=True,
            header_style="bold",
            expand=True,
        )
        stats_table.add_column("Metric", style="cyan", no_wrap=True)
        stats_table.add_column("Characters", style="yellow", justify="right", no_wrap=True)
        stats_table.add_column("Tokens", style="green", justify="right", no_wrap=True)

        for metric_name, char_key, token_key in metrics:
            stats_table.add_row(
                metric_name,
                _format_stat_value(stats.get(char_key)),
                _format_stat_value(stats.get(token_key)),
            )

        console.print(stats_table)

        info_bits: List[str] = []
        samples = component.get("samples")
        if samples is not None:
            info_bits.append(f"Samples: {int(samples):,}")
        tokens_per_request = component.get("tokens_per_request")
        if tokens_per_request is not None:
            info_bits.append(f"Tokens/request: {float(tokens_per_request):,.1f}")
        tokens_total = component.get("tokens_total")
        if tokens_total is not None:
            info_bits.append(f"Total tokens: {int(tokens_total):,}")

        metadata = component.get("metadata") or {}
        if component.get("type") == "text" and metadata.get("text_columns"):
            info_bits.append(f"Columns: {', '.join(metadata['text_columns'])}")
        if component.get("type") == "prompt":
            if metadata.get("expected_keys"):
                info_bits.append(f"Expected keys: {len(metadata['expected_keys'])}")
            preview = metadata.get("preview")
            if preview:
                info_bits.append(f"Preview: {preview}")

        if info_bits:
            console.print("[dim]" + " | ".join(info_bits) + "[/dim]")
        console.print("")


def _render_component_contributions(console, contributions: Optional[List[Dict[str, Any]]]) -> None:
    if not console or not contributions:
        return

    contributions_table = Table(
        title="Token Contribution Summary",
        border_style="magenta",
        show_header=True,
        header_style="bold",
    )
    contributions_table.add_column("Component", style="cyan", no_wrap=True)
    contributions_table.add_column("Tokens/request", style="green", justify="right", no_wrap=True)
    contributions_table.add_column("Total tokens", style="yellow", justify="right", no_wrap=True)
    contributions_table.add_column("Cost ($)", style="magenta", justify="right", no_wrap=True)

    for component in contributions:
        contributions_table.add_row(
            component.get("label", "Component"),
            f"{component.get('tokens_per_request', 0.0):,.1f}",
            f"{component.get('tokens_total', 0):,}",
            f"{component.get('cost', 0.0):,.4f}",
        )
        for sub in component.get("sub_components", []):
            contributions_table.add_row(
                f"  ↳ {sub.get('label', 'Sub-component')}",
                f"{sub.get('tokens_per_request', 0.0):,.1f}",
                f"{sub.get('tokens_total', 0):,}",
                f"{sub.get('cost', 0.0):,.4f}",
                style="dim",
            )

    console.print(contributions_table)
    console.print("")


def _display_cost_estimate_panel(
    console,
    *,
    cost_estimate: Optional[CostEstimate],
    cost_pricing: Optional[ModelPricing],
    cost_estimate_error: Optional[Exception],
    provider: str,
    model_name: str,
) -> None:
    """
    Render the API cost estimate (or a fallback message) in the configuration summary.

    Parameters mirror the computation helper so CLI variants can share the same UX.
    """
    if cost_estimate:
        rich_lines = format_cost_estimate_lines(cost_estimate, rich_markup=True)
        plain_lines = format_cost_estimate_lines(cost_estimate, rich_markup=False)
        if console:
            _render_component_token_tables(console, cost_estimate.assumptions.get("component_token_stats"))
            _render_component_contributions(console, cost_estimate.assumptions.get("component_token_contributions"))
            console.print(
                Panel(
                    "\n".join(rich_lines),
                    title="Estimated API Cost",
                    border_style="bright_magenta",
                )
            )
        else:
            print("\n".join(plain_lines))
        return

    if cost_pricing:
        message = "Unable to compute API cost estimate"
        if cost_estimate_error:
            message += f": {cost_estimate_error}"
        if console:
            console.print(f"[dim yellow]{message}.[/dim yellow]")
        else:
            print(f"{message}.")
        return

    if provider in {'openai', 'anthropic', 'google'}:
        message = (
            f"No pricing snapshot registered for {provider}:{model_name}; "
            "API cost estimate unavailable."
        )
        if console:
            console.print(f"[dim yellow]{message}[/dim yellow]")
        else:
            print(message)

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


def _persist_annotation_outputs(
    cli,
    *,
    source_output_path: Union[str, Path],
    session_dirs: Dict[str, Path],
    provider_folder: str,
    model_folder: str,
    dataset_name: str,
    annotation_results: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """
    Ensure annotated CSVs are stored under the session's log structure and create
    an annotations-only companion file for quick inspection.
    """
    from pathlib import Path
    import csv
    import shutil

    paths: Dict[str, Optional[str]] = {"full_path": None, "annotations_only_path": None}

    annotated_data_root = session_dirs.get("annotated_data")
    if not annotated_data_root:
        return paths

    annotated_data_root = Path(annotated_data_root)
    dataset_root = annotated_data_root / provider_folder / model_folder / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_output_path)
    full_output_path = dataset_root / source_path.name

    # Move into organised structure if needed (avoids lingering copies in data/)
    try:
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            if source_path.resolve() != full_output_path.resolve():
                if full_output_path.exists():
                    full_output_path.unlink()
                shutil.move(str(source_path), str(full_output_path))
        else:
            cli.logger.warning(f"Annotated output not found at {source_path}")
    except Exception as exc:
        cli.logger.warning(f"Failed to organise annotated output: {exc}")
        full_output_path = source_path

    paths["full_path"] = str(full_output_path)
    annotation_results["full_output_path"] = str(full_output_path)
    annotation_results["output_file"] = str(full_output_path)

    if full_output_path.suffix.lower() == ".csv":
        def _resolve_annotation_columns(csv_path: Path) -> List[str]:
            detected: Set[str] = set()
            hints = annotation_results.get("annotation_columns")
            if isinstance(hints, (list, tuple, set)):
                for col in hints:
                    col_name = str(col or "").strip()
                    if col_name:
                        detected.add(col_name)

            explicit = str(annotation_results.get("annotation_column", "")).strip()
            if explicit:
                detected.add(explicit)
            else:
                detected.add("annotation")

            SAMPLE_LIMIT = 500
            fieldnames: List[str] = []
            samples: List[Dict[str, Any]] = []

            try:
                with csv_path.open("r", newline="", encoding="utf-8") as src_file:
                    reader = csv.DictReader(src_file)
                    fieldnames = reader.fieldnames or []
                    if not fieldnames:
                        return [col for col in detected if col]

                    if not detected:
                        for column in fieldnames:
                            lower = column.lower()
                            if lower in {"annotation", "annotations", "labels", "label", "predictions", "prediction"}:
                                detected.add(column)
                            elif lower.endswith("_annotation") or lower.endswith("_annotations"):
                                detected.add(column)
                            elif lower.endswith("_labels") or lower.endswith("_prediction"):
                                detected.add(column)

                    for idx, row in enumerate(reader):
                        if idx < SAMPLE_LIMIT:
                            samples.append(row)
                        else:
                            break
            except Exception:
                return [col for col in detected if col]

            if not detected and samples:
                for column in fieldnames:
                    for row in samples:
                        value = str(row.get(column, "") or "").strip()
                        if value and value not in {"{}", "[]"} and (value.startswith("{") or value.startswith("[")):
                            detected.add(column)
                            break

            return [col for col in detected if col]

        annotation_columns = _resolve_annotation_columns(full_output_path)
        subset_hint = annotation_results.get("annotated_subset_path") or annotation_results.get("annotations_only_path")
        subset_target = Path(subset_hint) if subset_hint else full_output_path.with_name(
            f"{full_output_path.stem}_annotated_only{full_output_path.suffix}"
        )

        # Normalise destination so every run stores subsets under the organised dataset directory.
        if subset_target.parent.resolve() != dataset_root.resolve():
            subset_target = dataset_root / subset_target.name

        try:
            # Move an already generated subset (e.g. produced by pipeline layer) into the organised directory.
            if subset_hint:
                hinted_path = Path(subset_hint)
                if hinted_path.exists() and hinted_path.resolve() != subset_target.resolve():
                    subset_target.parent.mkdir(parents=True, exist_ok=True)
                    if subset_target.exists():
                        subset_target.unlink()
                    shutil.move(str(hinted_path), str(subset_target))

            if subset_target.exists():
                paths["annotations_only_path"] = str(subset_target)
                annotation_results["annotations_only_path"] = str(subset_target)
                annotation_results["annotated_subset_path"] = str(subset_target)
            elif annotation_columns:
                subset_target.parent.mkdir(parents=True, exist_ok=True)
                annotated_rows = _generate_annotations_only_subset(
                    source_csv=full_output_path,
                    target_csv=subset_target,
                    annotation_columns=annotation_columns,
                )
                paths["annotations_only_path"] = str(subset_target)
                annotation_results["annotations_only_path"] = str(subset_target)
                annotation_results["annotated_subset_path"] = str(subset_target)

                if annotated_rows == 0:
                    cli.logger.debug(
                        "Annotations-only subset for %s contains only headers (no annotated rows detected).",
                        full_output_path.name,
                    )
            else:
                cli.logger.warning(
                    "Unable to determine annotation columns for %s; annotations-only CSV not created.",
                    full_output_path,
                )
        except Exception as exc:
            cli.logger.warning(f"Failed to create annotations-only subset: {exc}")

    # Clean up legacy artefacts in the annotated_data root (e.g., initial dataset copies).
    legacy_candidate = annotated_data_root / source_path.name
    try:
        if legacy_candidate.exists() and legacy_candidate.resolve() != full_output_path.resolve():
            legacy_candidate.unlink()
    except Exception as exc:
        cli.logger.debug(f"Could not remove legacy annotations file {legacy_candidate}: {exc}")

    try:
        for raw_candidate in annotated_data_root.glob(f"{dataset_name}.*"):
            if not raw_candidate.is_file():
                continue
            resolved = raw_candidate.resolve()
            if resolved in {full_output_path.resolve(), subset_target.resolve()}:
                continue
            raw_candidate.unlink()
    except Exception as exc:
        cli.logger.debug(f"Could not remove dataset staging artefact for %s: %s", dataset_name, exc)

    # Remove any lingering files at the annotated_data root (should only contain organised folders).
    try:
        for orphan in annotated_data_root.iterdir():
            if orphan.is_file():
                orphan.unlink()
    except Exception as exc:
        cli.logger.debug("Could not tidy annotated_data root for %s: %s", dataset_name, exc)

    return paths


def _generate_annotations_only_subset(
    *,
    source_csv: Path,
    target_csv: Path,
    annotation_columns: List[str],
) -> int:
    """Fallback generator that extracts annotated rows without loading the full CSV into memory."""
    import csv

    if not annotation_columns:
        return 0

    try:
        with source_csv.open("r", newline="", encoding="utf-8") as src_file:
            reader = csv.DictReader(src_file)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                return 0

            annotated_count = 0
            with target_csv.open("w", newline="", encoding="utf-8") as dest_file:
                writer = csv.DictWriter(dest_file, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    for column in annotation_columns:
                        value = str(row.get(column, "")).strip()
                        if value and value.lower() not in {"nan", "none", "null", "{}", "[]"}:
                            writer.writerow(row)
                            annotated_count += 1
                            break

            return annotated_count
    except Exception:
        target_csv.unlink(missing_ok=True)
        raise


def ensure_validation_session_dirs(session_id: str) -> Dict[str, Path]:
    """Create Validation Lab storage structure for a session."""
    base_dir = Path("validation")
    session_root = base_dir / session_id
    doccano_dir = session_root / "doccano"
    doccano_dir.mkdir(parents=True, exist_ok=True)
    return {
        "session_root": session_root,
        "doccano": doccano_dir,
    }


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
        model_table = Table(title="Models Trained In This Session", box=table_box, show_lines=False, expand=True)
        model_table.add_column("#", style="cyan", justify="center", width=4, no_wrap=True)
        model_table.add_column("Model Identifier", style="green", overflow="fold")
        model_table.add_column("Langs", style="magenta", no_wrap=True)
        model_table.add_column("Macro F1", style="bright_white", justify="right", no_wrap=True)
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


# =============================================================================
# Context Window Configuration Helper
# =============================================================================

def _ask_context_window_config(cli, selected_prompts: List[Dict]) -> Dict[str, Any]:
    """
    Ask user about context window configuration after prompt selection.

    The context window allows including surrounding sentences (from neighboring rows)
    when annotating, while still producing one annotation per row.

    Parameters
    ----------
    cli : AdvancedCLI
        CLI instance for user interaction
    selected_prompts : List[Dict]
        List of selected prompt configurations

    Returns
    -------
    dict
        Context window configuration with keys:
        - enabled: bool
        - sentences_before: int
        - sentences_after: int
        - labels_translated: bool
        - target_language: str or None
        - labels: dict or None
    """
    cli.console.print("\n[bold cyan]Context Window Configuration[/bold cyan]")
    cli.console.print("[dim]Include surrounding sentences for better context understanding (e.g., for interview annotation)[/dim]\n")

    enable_context = Confirm.ask(
        "Do you want to expand the context by including surrounding sentences?",
        default=False
    )

    if not enable_context:
        cli.console.print("[dim]Context window disabled - each row will be annotated independently[/dim]")
        return {
            'enabled': False,
            'sentences_before': 0,
            'sentences_after': 0,
            'labels_translated': False,
            'target_language': None,
            'labels': None
        }

    cli.console.print("\n[dim]Specify how many sentences to include from neighboring rows.[/dim]")
    cli.console.print("[dim]The annotation will still be at single-row level.[/dim]\n")

    sentences_before = IntPrompt.ask(
        "How many sentences BEFORE the sentence to annotate?",
        default=2
    )
    sentences_before = max(0, min(sentences_before, 10))  # Clamp 0-10

    sentences_after = IntPrompt.ask(
        "How many sentences AFTER the sentence to annotate?",
        default=2
    )
    sentences_after = max(0, min(sentences_after, 10))  # Clamp 0-10

    if sentences_before == 0 and sentences_after == 0:
        cli.console.print("[yellow]No context sentences selected. Context window disabled.[/yellow]")
        return {
            'enabled': False,
            'sentences_before': 0,
            'sentences_after': 0,
            'labels_translated': False,
            'target_language': None,
            'labels': None
        }

    cli.console.print(f"[green]Context window configured: {sentences_before} sentences before, {sentences_after} sentences after[/green]")

    return {
        'enabled': True,
        'sentences_before': sentences_before,
        'sentences_after': sentences_after,
        'labels_translated': False,  # Will be set during annotation
        'target_language': None,      # Will be detected during annotation
        'labels': None                 # Will be translated during annotation
    }


# =============================================================================
# End of Context Window Configuration Helper
# =============================================================================


# =============================================================================
# Infer API auth helper
# =============================================================================

def _infer_api_connect(cli, saved_url: str, saved_username: str, saved_password: str):
    """Authenticate against the Infer API using /doccano/auth/login.

    Returns (api_url, token) on success, or None on failure.
    Prompts for credentials when missing, and re-prompts on auth failure.
    """
    import requests as _req

    # ── Saved credentials? Ask whether to reuse ─────
    if saved_url and saved_username and saved_password:
        cli.console.print(
            f"[dim]Saved credentials found: [bold]{saved_username}[/bold] @ {saved_url}[/dim]"
        )
        if Confirm.ask("[bold yellow]Use saved credentials?[/bold yellow]", default=True):
            infer_api_url = saved_url
            username = saved_username
            password = saved_password
        else:
            infer_api_url = Prompt.ask(
                "[bold yellow]Infer API URL[/bold yellow]",
                default=saved_url,
            )
            username = Prompt.ask("[bold yellow]Username[/bold yellow]")
            password = Prompt.ask("[bold yellow]Password[/bold yellow]", password=True)
    else:
        # ── Gather credentials ──────────────────────────
        infer_api_url = saved_url or Prompt.ask(
            "[bold yellow]Infer API URL[/bold yellow]",
            default="https://llm-tool-infer-api.antoinelemor.com",
        )
        username = saved_username or Prompt.ask(
            "[bold yellow]Username[/bold yellow]",
        )
        password = saved_password or Prompt.ask(
            "[bold yellow]Password[/bold yellow]", password=True,
        )

    while True:
        base = infer_api_url.rstrip("/")

        # ── Login ───────────────────────────────────────
        try:
            resp = _req.post(
                f"{base}/doccano/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )
        except Exception as exc:
            cli.console.print(f"[red]Could not connect to Infer API: {exc}[/red]")
            if not Confirm.ask("[yellow]Retry with different credentials?[/yellow]", default=True):
                return None
            infer_api_url = Prompt.ask(
                "[bold yellow]Infer API URL[/bold yellow]",
                default=infer_api_url,
            )
            username = Prompt.ask("[bold yellow]Username[/bold yellow]")
            password = Prompt.ask("[bold yellow]Password[/bold yellow]", password=True)
            continue

        if resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            cli.console.print(f"[red]Authentication failed ({resp.status_code}): {detail}[/red]")
            if not Confirm.ask("[yellow]Retry with different credentials?[/yellow]", default=True):
                return None
            infer_api_url = Prompt.ask(
                "[bold yellow]Infer API URL[/bold yellow]",
                default=infer_api_url,
            )
            username = Prompt.ask("[bold yellow]Username[/bold yellow]")
            password = Prompt.ask("[bold yellow]Password[/bold yellow]", password=True)
            continue

        # ── Success ─────────────────────────────────────
        data = resp.json()
        token = data["token"]
        display_name = data.get("user", {}).get("display_name", username)
        cli.console.print(f"[green]Connected as [bold]{display_name}[/bold][/green]")

        # Save credentials for future sessions
        cli.settings.set_infer_api_credentials(infer_api_url, username, password)

        return infer_api_url, token


# =============================================================================
# Live Doccano Sync — mode selector helper
# =============================================================================

def _doccano_sync_setup(
    cli,
    infer_api_url: str,
    token: str,
    default_project_name: str,
    labels: List[str],
    *,
    data_path: Optional[Path] = None,
    text_column: Optional[str] = None,
    identifier_column: Optional[str] = None,
) -> Optional["DoccanoSyncClient"]:
    """Interactive Doccano sync mode selector.

    Lets the user choose between create / overwrite / append, then returns
    a started DoccanoSyncClient or None on cancel/error.
    """
    from ..annotators.doccano_sync import (
        DoccanoSyncClient,
        DoccanoRewriteMatcher,
        setup_doccano_project,
        setup_doccano_project_safe,
        list_doccano_projects,
        list_doccano_examples,
        count_doccano_examples,
        clear_doccano_project,
    )

    # ── List existing projects ────────────────────
    projects: List[dict] = []
    try:
        projects = list_doccano_projects(infer_api_url, token)
    except Exception as exc:
        cli.console.print(f"[yellow]Could not list Doccano projects: {exc}[/yellow]")

    if projects:
        cli.console.print("\n[bold]Existing Doccano projects:[/bold]")
        for p in projects:
            try:
                n_examples = count_doccano_examples(infer_api_url, token, p["id"])
            except Exception:
                n_examples = "?"
            cli.console.print(
                f"  [cyan]#{p['id']}[/cyan]  {p.get('name', '?')}"
                f"  [dim]({p.get('project_type', '')})[/dim]"
                f"  — {n_examples} examples"
            )
        cli.console.print()

    # ── Choose sync mode ──────────────────────────
    if projects:
        cli.console.print("[bold]Sync mode:[/bold]")
        cli.console.print("  [green]create[/green]    — Create a new project")
        cli.console.print("  [yellow]overwrite[/yellow] — Overwrite (delete all, then sync)")
        cli.console.print("  [blue]append[/blue]    — Append to existing project")
        cli.console.print("  [magenta]rewrite[/magenta]   — Rewrite LLM annotations only (keep human annotations)")
        cli.console.print("  [red]delete[/red]    — Delete all, then sync fresh")
        cli.console.print()
        mode = Prompt.ask(
            "[bold yellow]Choose sync mode[/bold yellow]",
            choices=["create", "overwrite", "append", "rewrite", "delete"],
            default="create",
        )
    else:
        mode = "create"

    # ── CREATE ────────────────────────────────────
    if mode == "create":
        project_name = Prompt.ask(
            "[bold yellow]Project name[/bold yellow]",
            default=default_project_name,
        )
        project_id = setup_doccano_project_safe(
            infer_api_url, token, project_name, labels,
        )
        if project_id is None:
            cli.console.print(
                "[red]Failed to create Doccano project (server may be down). "
                "Continuing without Doccano sync.[/red]"
            )
            return None
        cli.console.print(f"[green]Created project #{project_id} — '{project_name}'[/green]")

    # ── OVERWRITE ─────────────────────────────────
    elif mode == "overwrite":
        valid_ids = [str(p["id"]) for p in projects]
        chosen = Prompt.ask(
            f"[bold yellow]Enter project ID [{'/'.join(valid_ids)}][/bold yellow]",
            choices=valid_ids,
        )
        project_id = int(chosen)
        proj_name = next((p.get("name", "?") for p in projects if p["id"] == project_id), "?")

        try:
            count = count_doccano_examples(infer_api_url, token, project_id)
        except Exception:
            count = 0

        if count > 0:
            if not Confirm.ask(
                f"[bold red]Delete all {count} examples from '{proj_name}' (#{project_id})?[/bold red]",
                default=False,
            ):
                cli.console.print("[yellow]Overwrite cancelled. Continuing without Doccano sync.[/yellow]")
                return None
            try:
                deleted = clear_doccano_project(infer_api_url, token, project_id)
                cli.console.print(f"[green]Cleared {deleted} examples from project #{project_id}[/green]")
            except Exception as exc:
                cli.console.print(f"[red]Failed to clear project: {exc}[/red]")
                return None
        else:
            cli.console.print(f"[dim]Project #{project_id} is already empty.[/dim]")

        cli.console.print(f"[green]Overwrite mode: using project #{project_id}[/green]")

    # ── APPEND ────────────────────────────────────
    elif mode == "append":
        valid_ids = [str(p["id"]) for p in projects]
        chosen = Prompt.ask(
            f"[bold yellow]Enter project ID [{'/'.join(valid_ids)}][/bold yellow]",
            choices=valid_ids,
        )
        project_id = int(chosen)
        cli.console.print(f"[green]Append mode: using project #{project_id}[/green]")

    # ── REWRITE (update LLM annotations, keep human) ──
    elif mode == "rewrite":
        valid_ids = [str(p["id"]) for p in projects]
        chosen = Prompt.ask(
            f"[bold yellow]Enter project ID [{'/'.join(valid_ids)}][/bold yellow]",
            choices=valid_ids,
        )
        project_id = int(chosen)
        proj_name = next((p.get("name", "?") for p in projects if p["id"] == project_id), "?")

        sync_total = 0
        try:
            sync_total = count_doccano_examples(infer_api_url, token, project_id)
        except Exception:
            pass

        from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, MofNCompleteColumn

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[dim]Loading '{proj_name}' (#{project_id})[/dim]"),
                BarColumn(),
                MofNCompleteColumn(),
                console=cli.console,
            ) as progress:
                task = progress.add_task("fetch", total=sync_total or None)

                def _on_progress(fetched, total):
                    progress.update(task, completed=fetched)

                existing = list_doccano_examples(
                    infer_api_url, token, project_id,
                    total=sync_total or None, on_progress=_on_progress,
                )
        except Exception as exc:
            cli.console.print(f"[red]Failed to fetch examples: {exc}[/red]")
            return None
        cli.console.print(f"[green]Loaded {len(existing):,} existing examples from '{proj_name}'[/green]")

        # ── Interactive CSV ↔ Doccano matching ──
        match_meta_keys = ["identifier"]  # default

        if data_path is not None and text_column and Confirm.ask(
            "[bold yellow]Same data you want to re-annotate? (smart CSV ↔ Doccano matching)[/bold yellow]",
            default=True,
        ):
            import pandas as pd

            # Load sample
            dp = Path(data_path) if not isinstance(data_path, Path) else data_path
            suffix = dp.suffix.lower()
            try:
                if suffix == ".csv":
                    df_sample = pd.read_csv(dp, nrows=1000)
                elif suffix == ".json":
                    df_sample = pd.read_json(dp, lines=False)
                    if len(df_sample) > 1000:
                        df_sample = df_sample.head(1000)
                elif suffix == ".jsonl":
                    df_sample = pd.read_json(dp, lines=True, nrows=1000)
                elif suffix in (".xlsx", ".xls"):
                    df_sample = pd.read_excel(dp, nrows=1000)
                elif suffix == ".parquet":
                    df_sample = pd.read_parquet(dp)
                    if len(df_sample) > 1000:
                        df_sample = df_sample.head(1000)
                else:
                    df_sample = pd.read_csv(dp, nrows=1000)
            except Exception as exc:
                cli.console.print(f"[yellow]Could not load sample from {dp.name}: {exc}[/yellow]")
                df_sample = None

            if df_sample is not None and not df_sample.empty:
                all_cols = list(df_sample.columns)
                cli.console.print(f"\n[bold]CSV columns[/bold] (text = [cyan]{text_column}[/cyan]):")
                for i, col in enumerate(all_cols, 1):
                    tag = ""
                    if col == text_column:
                        tag = " [dim](text — always used)[/dim]"
                    elif col == identifier_column:
                        tag = " [green](suggested)[/green]"
                    cli.console.print(f"  [cyan]{i}[/cyan]  {col}{tag}")

                raw_choice = Prompt.ask(
                    "\n[bold yellow]Columns for matching (numbers or names, comma-separated)[/bold yellow]",
                    default=identifier_column or "identifier",
                )
                # Parse selection
                selected_csv_cols = []
                for part in raw_choice.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    # Try as number
                    try:
                        idx = int(part) - 1
                        if 0 <= idx < len(all_cols):
                            selected_csv_cols.append(all_cols[idx])
                            continue
                    except ValueError:
                        pass
                    # Try as column name
                    if part in all_cols:
                        selected_csv_cols.append(part)
                    else:
                        cli.console.print(f"[yellow]Ignoring unknown column: {part}[/yellow]")

                if not selected_csv_cols:
                    selected_csv_cols = [identifier_column] if identifier_column else []

                # Build preview rows with CSV→meta mapping
                # push() stores csv_row[identifier_column] as meta["identifier"]
                preview_rows = []
                for _, csv_row in df_sample.iterrows():
                    preview_row = {text_column: str(csv_row.get(text_column, ""))}
                    if identifier_column and identifier_column in csv_row.index:
                        preview_row["identifier"] = str(csv_row[identifier_column])
                    preview_rows.append(preview_row)

                # match_meta_keys = the Doccano meta keys (always "identifier")
                match_meta_keys = ["identifier"]

                # Warn about extra columns
                extra_cols = [c for c in selected_csv_cols if c != identifier_column]
                if extra_cols:
                    cli.console.print(
                        f"[yellow]Note: only 'identifier' is stored in Doccano meta by push(). "
                        f"Extra columns {extra_cols} will not be used for matching.[/yellow]"
                    )

                # Run preview matching
                temp_matcher = DoccanoRewriteMatcher(existing, match_meta_keys=match_meta_keys)
                stats = temp_matcher.preview_match(preview_rows, text_key=text_column)

                # Display Rich table
                stats_table = Table(title="Matching Preview", show_header=True, header_style="bold")
                stats_table.add_column("Category", style="bold")
                stats_table.add_column("Count", justify="right")
                stats_table.add_column("Action")
                stats_table.add_row("Matched (text + meta)", str(stats["matched"]), "[green]will update[/green]")
                stats_table.add_row("Text-only (meta differs)", str(stats["text_only"]), "[yellow]will create new[/yellow]")
                stats_table.add_row("Unmatched", str(stats["unmatched"]), "[yellow]will create new[/yellow]")
                stats_table.add_row("Ambiguous (N>1)", str(stats["ambiguous"]), "[red]will skip[/red]")
                cli.console.print(stats_table)

                sample_total = len(preview_rows)
                cli.console.print(f"[dim]Sample: {sample_total} rows from {dp.name}[/dim]")

                if stats["matched"] == 0:
                    cli.console.print("[bold red]Warning: 0 matches found — all rows will be created as new.[/bold red]")

                if not Confirm.ask("[bold yellow]Proceed with rewrite?[/bold yellow]", default=True):
                    cli.console.print("[yellow]Rewrite cancelled. Continuing without Doccano sync.[/yellow]")
                    return None
            else:
                cli.console.print("[yellow]Could not load CSV sample — falling back to manual matching.[/yellow]")
                match_keys_str = Prompt.ask(
                    "[bold yellow]Meta fields to match on (comma-separated)[/bold yellow]",
                    default="identifier",
                )
                match_meta_keys = [k.strip() for k in match_keys_str.split(",") if k.strip()]
        else:
            # No data_path or user declined smart matching
            match_keys_str = Prompt.ask(
                "[bold yellow]Meta fields to match on (comma-separated)[/bold yellow]",
                default="identifier",
            )
            match_meta_keys = [k.strip() for k in match_keys_str.split(",") if k.strip()]

        client = DoccanoSyncClient(
            api_url=infer_api_url,
            token=token,
            project_id=project_id,
            mode="rewrite",
            match_meta_keys=match_meta_keys,
            existing_examples=existing,
        )
        client.start()
        cli.console.print(f"[green]Rewrite sync started for project #{project_id} (matching on {match_meta_keys})[/green]")
        return client

    # ── DELETE (clear then sync) ─────────────────
    else:
        valid_ids = [str(p["id"]) for p in projects]
        chosen = Prompt.ask(
            f"[bold yellow]Enter project ID to delete from [{'/'.join(valid_ids)}][/bold yellow]",
            choices=valid_ids,
        )
        project_id = int(chosen)
        proj_name = next((p.get("name", "?") for p in projects if p["id"] == project_id), "?")

        try:
            count = count_doccano_examples(infer_api_url, token, project_id)
        except Exception:
            count = 0

        if count == 0:
            cli.console.print(f"[dim]Project '{proj_name}' (#{project_id}) is already empty.[/dim]")
        elif Confirm.ask(
            f"[bold red]Delete all {count} examples from '{proj_name}' (#{project_id})?[/bold red]",
            default=False,
        ):
            try:
                deleted = clear_doccano_project(infer_api_url, token, project_id)
                cli.console.print(f"[green]Deleted {deleted} examples from project #{project_id}[/green]")
            except Exception as exc:
                cli.console.print(f"[red]Failed to delete: {exc}[/red]")
                return None
        else:
            cli.console.print("[yellow]Delete cancelled.[/yellow]")
            return None

    # ── Start sync client ─────────────────────────
    client = DoccanoSyncClient(
        api_url=infer_api_url,
        token=token,
        project_id=project_id,
    )
    client.start()
    cli.console.print(f"[green]Live sync started for project #{project_id}[/green]")
    return client


# =============================================================================
# Doccano rewrite filter — match first, then annotate only matched rows
# =============================================================================

def _doccano_rewrite_filter(
    cli,
    infer_api_url: str,
    token: str,
    data_path: Path,
    data_format: str,
    text_column: str,
    identifier_column: Optional[str],
    labels: List[str],
) -> Optional[Tuple["DoccanoSyncClient", List[str]]]:
    """Connect to Doccano, match CSV rows by text to existing examples.

    Returns ``(DoccanoSyncClient in rewrite mode, list_of_matched_identifier_values)``
    or ``None`` if the user cancels or there are 0 matches.
    """
    from ..annotators.doccano_sync import (
        DoccanoSyncClient,
        DoccanoRewriteMatcher,
        list_doccano_projects,
        list_doccano_examples,
        count_doccano_examples,
    )
    import logging
    import pandas as pd

    # ── 1. List existing projects ────────────────────
    projects: List[dict] = []
    try:
        projects = list_doccano_projects(infer_api_url, token)
    except Exception as exc:
        cli.console.print(f"[yellow]Could not list Doccano projects: {exc}[/yellow]")

    if not projects:
        cli.console.print("[yellow]No Doccano projects found. Cannot rewrite.[/yellow]")
        return None

    proj_table = Table(
        title="Doccano Projects", show_header=True, header_style="bold",
        box=box.SIMPLE_HEAVY if box else None,
    )
    proj_table.add_column("ID", style="cyan", justify="right")
    proj_table.add_column("Name", style="bold")
    proj_table.add_column("Type", style="dim")
    proj_table.add_column("Examples", justify="right")
    for p in projects:
        try:
            n = count_doccano_examples(infer_api_url, token, p["id"])
        except Exception:
            n = "?"
        proj_table.add_row(
            str(p["id"]), p.get("name", "?"),
            p.get("project_type", ""), str(n),
        )
    cli.console.print(proj_table)

    # ── 2. Pick a project ────────────────────────────
    valid_ids = [str(p["id"]) for p in projects]
    chosen = Prompt.ask(
        "[bold yellow]Project ID to rewrite[/bold yellow]",
        choices=valid_ids,
    )
    project_id = int(chosen)
    proj_name = next((p.get("name", "?") for p in projects if p["id"] == project_id), "?")

    # ── 3. Fetch all examples from chosen project ────
    # Get total count for progress bar (already fetched during project listing)
    example_total = 0
    try:
        example_total = count_doccano_examples(infer_api_url, token, project_id)
    except Exception:
        pass

    if example_total == 0:
        cli.console.print("[yellow]Project has 0 examples — nothing to rewrite.[/yellow]")
        return None

    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, MofNCompleteColumn

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[dim]Loading '{proj_name}' (#{project_id})[/dim]"),
            BarColumn(),
            MofNCompleteColumn(),
            console=cli.console,
        ) as progress:
            task = progress.add_task("fetch", total=example_total)

            def _on_progress(fetched, total):
                progress.update(task, completed=fetched)

            existing = list_doccano_examples(
                infer_api_url, token, project_id,
                total=example_total, on_progress=_on_progress,
            )
    except Exception as exc:
        cli.console.print(f"[red]Failed to fetch examples: {exc}[/red]")
        return None

    if not existing:
        cli.console.print("[yellow]Project has 0 examples — nothing to rewrite.[/yellow]")
        return None

    cli.console.print(f"[green]Loaded {len(existing):,} Doccano examples[/green]")

    # ── 4. Load the full CSV / DataFrame ─────────────
    dp = Path(data_path) if not isinstance(data_path, Path) else data_path
    suffix = dp.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(dp)
        elif suffix == ".json":
            df = pd.read_json(dp, lines=False)
        elif suffix == ".jsonl":
            df = pd.read_json(dp, lines=True)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(dp)
        elif suffix == ".parquet":
            df = pd.read_parquet(dp)
        else:
            df = pd.read_csv(dp)
    except Exception as exc:
        cli.console.print(f"[red]Could not load data from {dp.name}: {exc}[/red]")
        return None

    # ── 5. Auto-detect CSV ↔ Doccano meta mappings ───
    # Collect all meta keys from Doccano examples (exclude internal keys)
    _internal_keys = {"llm_annotations"}
    doccano_meta_keys: Set[str] = set()
    for ex in existing:
        for k in (ex.get("meta") or {}):
            if k not in _internal_keys:
                doccano_meta_keys.add(k)

    csv_cols = list(df.columns)
    auto_mappings: List[Tuple[str, str, int]] = []  # (csv_col, meta_key, sample_matches)

    if doccano_meta_keys:
        # Collect ALL Doccano meta values per key
        for meta_key in sorted(doccano_meta_keys):
            doc_vals = set()
            for ex in existing:
                v = (ex.get("meta") or {}).get(meta_key)
                if v is not None:
                    doc_vals.add(str(v))
            if not doc_vals:
                continue

            # Check if meta values look like composites (contain '|')
            # e.g. identifier = "123|456" from a composite id+text_id_for_llm
            sample_val = next(iter(doc_vals))
            is_composite_meta = '|' in sample_val

            if is_composite_meta:
                # Split composite values and try matching parts against CSV columns
                parts_by_pos: Dict[int, set] = {}
                for v in doc_vals:
                    for i, part in enumerate(v.split('|')):
                        parts_by_pos.setdefault(i, set()).add(part)
                for pos, part_vals in sorted(parts_by_pos.items()):
                    for col in csv_cols:
                        if col == text_column:
                            continue
                        csv_vals = set(df[col].dropna().astype(str).unique())
                        overlap = csv_vals & part_vals
                        if overlap:
                            auto_mappings.append((col, f"{meta_key}[{pos}]", len(overlap)))
            else:
                # Direct match — try each CSV column
                for col in csv_cols:
                    if col == text_column:
                        continue
                    csv_vals = set(df[col].dropna().astype(str).unique())
                    overlap = csv_vals & doc_vals
                    if overlap:
                        auto_mappings.append((col, meta_key, len(overlap)))

    # Show detected mappings
    if auto_mappings or doccano_meta_keys:
        cli.console.print("\n[bold]CSV ↔ Doccano field detection:[/bold]")
        cli.console.print(f"  [cyan]text[/cyan] column: [bold]{text_column}[/bold] [dim](primary match key)[/dim]")

        if doccano_meta_keys:
            cli.console.print(f"  [cyan]Doccano meta keys[/cyan]: {', '.join(sorted(doccano_meta_keys))}")

        if auto_mappings:
            map_table = Table(
                show_header=True, header_style="bold",
                box=box.SIMPLE if box else None, padding=(0, 1),
            )
            map_table.add_column("CSV column", style="cyan")
            map_table.add_column("Doccano meta key", style="green")
            map_table.add_column("Sample overlap", justify="right")
            for csv_col, meta_key, overlap in sorted(auto_mappings, key=lambda x: -x[2]):
                map_table.add_row(csv_col, meta_key, f"{overlap} values")
            cli.console.print(map_table)
        else:
            cli.console.print("  [dim]No automatic column ↔ meta matches found[/dim]")
        cli.console.print()

    # Determine meta match keys from auto-detection
    # Use detected mappings for tiebreaking when multiple texts match
    match_meta_keys: List[str] = []
    csv_to_meta_map: Dict[str, str] = {}
    if auto_mappings:
        # Pick best mapping per meta key (highest overlap)
        best_per_meta: Dict[str, Tuple[str, int]] = {}
        for csv_col, meta_key, overlap in auto_mappings:
            if meta_key not in best_per_meta or overlap > best_per_meta[meta_key][1]:
                best_per_meta[meta_key] = (csv_col, overlap)
        for meta_key, (csv_col, _) in best_per_meta.items():
            match_meta_keys.append(meta_key)
            csv_to_meta_map[csv_col] = meta_key

        cli.console.print("[bold]Proposed matching strategy:[/bold]")
        cli.console.print(f"  1. [green]Text[/green] (column '{text_column}') — primary match")
        for csv_col, meta_key in csv_to_meta_map.items():
            cli.console.print(f"  2. [cyan]{csv_col}[/cyan] → meta.[green]{meta_key}[/green] — tiebreaker")

        if not Confirm.ask(
            "\n[bold yellow]Use this matching strategy?[/bold yellow]",
            default=True,
        ):
            match_meta_keys = []
            csv_to_meta_map = {}
            cli.console.print("[dim]Using text-only matching.[/dim]")
    else:
        cli.console.print("[dim]Matching by text content only.[/dim]")

    if not match_meta_keys:
        match_meta_keys = ["identifier"]

    id_col = identifier_column or "identifier"

    # Handle composite identifiers (e.g. "id+text_id_for_llm")
    # Build composite values the same way as llm_annotator._resolve_identifier_column()
    _id_is_composite = id_col and '+' in id_col
    _id_components: List[str] = []
    if _id_is_composite:
        _id_components = [c.strip() for c in id_col.split('+') if c.strip()]
        missing_cols = [c for c in _id_components if c not in df.columns]
        if missing_cols or len(_id_components) <= 1:
            _id_is_composite = False
            _id_components = []

    def _build_row_id(row) -> str:
        """Build the identifier value for a row, handling composites."""
        if _id_is_composite:
            parts = []
            for c in _id_components:
                v = row.get(c)
                parts.append(str(v) if pd.notna(v) else '__MISSING__')
            return '|'.join(parts)
        return str(row.get(id_col, ""))

    # ── 6. Build matcher and run ─────────────────────
    matcher = DoccanoRewriteMatcher(existing, match_meta_keys=match_meta_keys)

    # Suppress noisy matcher logs during bulk matching
    _matcher_logger = logging.getLogger("llm_tool.annotators.doccano_sync")
    _prev_level = _matcher_logger.level
    _matcher_logger.setLevel(logging.ERROR)

    matched_ids: List[str] = []
    stats = {"matched": 0, "unmatched": 0, "ambiguous": 0}

    for _, row in df.iterrows():
        text = str(row.get(text_column, ""))
        meta = {}
        # Build meta dict from detected mappings
        if csv_to_meta_map:
            for csv_col, meta_key in csv_to_meta_map.items():
                if csv_col in row.index:
                    meta[meta_key] = str(row[csv_col])
        else:
            if _id_is_composite:
                meta["identifier"] = _build_row_id(row)
            elif id_col in row.index:
                meta["identifier"] = str(row[id_col])

        action, _, _ = matcher.match(text, meta)
        if action == "update":
            stats["matched"] += 1
            matched_ids.append(_build_row_id(row))
        elif action == "skip":
            stats["ambiguous"] += 1
        else:
            stats["unmatched"] += 1

    _matcher_logger.setLevel(_prev_level)

    # ── 7. Show results ──────────────────────────────
    total = len(df)
    match_pct = (stats["matched"] / total * 100) if total else 0
    pct_str = "< 1%" if 0 < match_pct < 1 else f"{match_pct:.0f}%"

    results_table = Table(
        title="Rewrite Matching", show_header=True, header_style="bold",
        box=box.SIMPLE_HEAVY if box else None,
    )
    results_table.add_column("", style="bold")
    results_table.add_column("Count", justify="right")
    results_table.add_column("")
    results_table.add_row(
        "Matched", f"[green]{stats['matched']}[/green]",
        f"[green]will re-annotate ({pct_str})[/green]",
    )
    if stats["ambiguous"]:
        results_table.add_row(
            "Ambiguous", f"[red]{stats['ambiguous']}[/red]",
            "[red]skipped (multiple text matches)[/red]",
        )
    results_table.add_row(
        "Not in Doccano", f"[dim]{stats['unmatched']}[/dim]",
        "[dim]skipped[/dim]",
    )
    cli.console.print(results_table)
    cli.console.print(
        f"[dim]CSV: {total:,} rows — Doccano: {len(existing)} examples — "
        f"text column: {text_column}[/dim]"
    )

    # ── 8. Handle 0 matches ──────────────────────────
    if stats["matched"] == 0:
        cli.console.print("\n[bold red]0 matches — cannot rewrite.[/bold red]")
        return None

    # ── 9. User confirms ─────────────────────────────
    if not Confirm.ask(
        f"\n[bold yellow]Re-annotate {stats['matched']} matched rows?[/bold yellow]",
        default=True,
    ):
        cli.console.print("[yellow]Rewrite cancelled.[/yellow]")
        return None

    # ── 10. Build client ─────────────────────────────
    client = DoccanoSyncClient(
        api_url=infer_api_url,
        token=token,
        project_id=project_id,
        mode="rewrite",
        match_meta_keys=match_meta_keys,
        existing_examples=existing,
    )
    client.start()
    cli.console.print(
        f"[green]Rewrite sync ready — project '{proj_name}' "
        f"(#{project_id}, {stats['matched']} rows)[/green]"
    )
    return (client, matched_ids)


# =============================================================================
# Push training test set to Doccano (Factory Step 2.5)
# =============================================================================

def _push_test_set_to_doccano(
    cli,
    training_results: Dict[str, Any],
    infer_api_url: str,
    token: str,
    session_id: str,
    labels: List[str],
) -> Optional[Dict[str, Any]]:
    """Push training test set to Doccano for human review.

    Searches the training session directory for test JSONL files, then
    creates (or appends to) a Doccano project with those examples.

    Returns ``{project_id, example_count, label_mapping}`` or ``None``.
    """
    import glob as _glob

    from ..annotators.doccano_sync import (
        DoccanoSyncClient,
        setup_doccano_project,
        list_doccano_projects,
        count_doccano_examples,
    )

    # ── 1. Locate test JSONL files ───────────────────
    training_logs_dir = training_results.get("training_logs_dir")
    if not training_logs_dir:
        cli.console.print("[yellow]No training logs directory found — cannot locate test files.[/yellow]")
        return None

    logs_path = Path(training_logs_dir)
    test_files: List[Path] = []
    for pattern in ("**/test/*_test.jsonl", "**/test.jsonl"):
        test_files.extend(logs_path.glob(pattern))
    # Deduplicate (resolve to absolute)
    test_files = list({p.resolve(): p for p in test_files}.values())

    if not test_files:
        cli.console.print("[yellow]No test JSONL files found under training session — skipping Doccano push.[/yellow]")
        return None

    # ── 2. Load all test examples ────────────────────
    all_examples: List[dict] = []
    file_sources: Dict[str, int] = {}
    for tf in test_files:
        count = 0
        with open(tf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["_source_file"] = tf.name
                    all_examples.append(rec)
                    count += 1
                except json.JSONDecodeError:
                    continue
        file_sources[tf.name] = count

    if not all_examples:
        cli.console.print("[yellow]Test files are empty — nothing to push.[/yellow]")
        return None

    # ── 3. Build label mapping ───────────────────────
    bundle = training_results.get("bundle")
    metadata = training_results.get("metadata") or {}
    id2label: Dict[int, str] = {}

    # Try bundle metadata first
    if bundle and hasattr(bundle, "metadata"):
        id2label = bundle.metadata.get("id2label", {})
    if not id2label:
        id2label = metadata.get("id2label", {})
    # Fallback: derive from labels list
    if not id2label and labels:
        id2label = {i: lbl for i, lbl in enumerate(labels)}
    # Convert string keys to int
    id2label = {int(k): v for k, v in id2label.items()}

    # ── 4. Show summary ─────────────────────────────
    from rich.table import Table

    cli.console.print("\n[bold cyan]Step 2.5 — Test Set to Doccano[/bold cyan]")
    summary = Table(title="Test Set Files", show_header=True, header_style="bold")
    summary.add_column("File")
    summary.add_column("Examples", justify="right")
    for fname, cnt in file_sources.items():
        summary.add_row(fname, f"{cnt:,}")
    summary.add_row("[bold]Total[/bold]", f"[bold]{len(all_examples):,}[/bold]")
    cli.console.print(summary)

    if not Confirm.ask(
        "[bold yellow]Push test set to Doccano for human review?[/bold yellow]",
        default=True,
    ):
        return None

    # ── 5. Project selection ─────────────────────────
    projects: List[dict] = []
    try:
        projects = list_doccano_projects(infer_api_url, token)
    except Exception as exc:
        cli.console.print(f"[yellow]Could not list projects: {exc}[/yellow]")

    if projects:
        cli.console.print("\n[bold]Existing Doccano projects:[/bold]")
        for p in projects:
            try:
                n = count_doccano_examples(infer_api_url, token, p["id"])
            except Exception:
                n = "?"
            cli.console.print(f"  [cyan]#{p['id']}[/cyan]  {p.get('name', '?')}  — {n} examples")

    mode = Prompt.ask(
        "[bold yellow]Create new project or append to existing?[/bold yellow]",
        choices=["create", "append"],
        default="create",
    )

    project_id: Optional[int] = None
    if mode == "append" and projects:
        pid_str = Prompt.ask("[bold yellow]Enter project ID[/bold yellow]")
        try:
            project_id = int(pid_str)
        except ValueError:
            cli.console.print("[red]Invalid project ID — creating new project instead.[/red]")
            mode = "create"

    if mode == "create" or project_id is None:
        proj_name = f"test_set_{session_id}"
        try:
            project_id = setup_doccano_project(
                infer_api_url, token, proj_name,
                labels=labels,
            )
            cli.console.print(f"[green]Created Doccano project '{proj_name}' (#{project_id})[/green]")
        except Exception as exc:
            cli.console.print(f"[red]Failed to create project: {exc}[/red]")
            return None

    # ── 6. Push examples ─────────────────────────────
    client = DoccanoSyncClient(
        api_url=infer_api_url,
        token=token,
        project_id=project_id,
        mode="push",
    )
    client.start()

    pushed = 0
    from rich.progress import Progress

    with Progress(console=cli.console) as progress:
        task = progress.add_task("Pushing test set...", total=len(all_examples))
        for rec in all_examples:
            text = rec.get("text", "")
            label_id = rec.get("label")
            label_name = id2label.get(label_id, f"class_{label_id}") if label_id is not None else "unknown"
            lang = rec.get("lang", "")
            source_file = rec.pop("_source_file", "")

            meta = {
                "test_set_info": {
                    "split": "test",
                    "factory_session_id": session_id,
                    "source": source_file,
                },
                "label_name": label_name,
            }
            if label_id is not None:
                meta["label_id"] = label_id
            if lang:
                meta["lang"] = lang

            annotation = {"label": label_name}
            ok = client.push(text, annotation, meta=meta)
            if ok:
                pushed += 1
            progress.advance(task)

    stats = client.stop()
    cli.console.print(
        f"[green]Pushed {pushed:,} / {len(all_examples):,} test examples "
        f"to Doccano project #{project_id}[/green]"
    )

    return {
        "project_id": project_id,
        "example_count": pushed,
        "label_mapping": id2label,
    }


# =============================================================================
# Annotate Doccano test set with trained models (Factory Step 3.5)
# =============================================================================

def _annotate_doccano_with_trained_models(
    cli,
    training_results: Dict[str, Any],
    test_doccano_info: Dict[str, Any],
    infer_api_url: str,
    token: str,
) -> Optional[Dict[str, Any]]:
    """Run trained models on the Doccano test set, PATCH ``meta.model_annotations``.

    For each trained model found in *training_results* the function loads
    the model locally, runs inference on every example in the Doccano
    project referenced by *test_doccano_info*, and writes predictions
    into ``meta.model_annotations.<model_name>``.

    Returns annotation stats or ``None``.
    """
    import time as _time

    from ..annotators.doccano_sync import (
        list_doccano_examples,
        update_doccano_llm_annotation,
    )

    # ── 1. Collect trained model paths ───────────────
    trained_map: Dict[str, Path] = {}
    raw = training_results.get("trained_model_paths") or {}
    if isinstance(raw, dict):
        for name, path_val in raw.items():
            p = Path(path_val)
            if p.is_dir() and (p / "config.json").exists():
                trained_map[name] = p

    if not trained_map:
        cli.console.print("[yellow]No trained models found — skipping Doccano model annotation.[/yellow]")
        return None

    project_id = test_doccano_info["project_id"]

    # ── 2. Show models + confirm ─────────────────────
    from rich.table import Table

    cli.console.print("\n[bold cyan]Step 3.5 — Model Predictions to Doccano[/bold cyan]")
    model_table = Table(title="Trained Models", show_header=True, header_style="bold")
    model_table.add_column("Model")
    model_table.add_column("Path")
    for name, path in trained_map.items():
        model_table.add_row(name, str(path))
    cli.console.print(model_table)

    if not Confirm.ask(
        f"[bold yellow]Run these models on Doccano project #{project_id} and store predictions?[/bold yellow]",
        default=True,
    ):
        return None

    # ── 3. Fetch examples from Doccano ───────────────
    bert_total = 0
    try:
        bert_total = count_doccano_examples(infer_api_url, token, project_id)
    except Exception:
        pass

    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, MofNCompleteColumn

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[dim]Loading project #{project_id}[/dim]"),
            BarColumn(),
            MofNCompleteColumn(),
            console=cli.console,
        ) as progress:
            task = progress.add_task("fetch", total=bert_total or None)

            def _on_progress(fetched, total):
                progress.update(task, completed=fetched)

            examples = list_doccano_examples(
                infer_api_url, token, project_id,
                total=bert_total or None, on_progress=_on_progress,
            )
    except Exception as exc:
        cli.console.print(f"[red]Failed to fetch examples: {exc}[/red]")
        return None

    if not examples:
        cli.console.print("[yellow]No examples found in Doccano project — nothing to annotate.[/yellow]")
        return None

    cli.console.print(f"[dim]Fetched {len(examples):,} examples from project #{project_id}[/dim]")

    # ── 4. Run inference per model ───────────────────
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        cli.console.print("[red]torch / transformers not installed — cannot run inference.[/red]")
        return None

    from rich.progress import Progress

    stats: Dict[str, int] = {}

    for model_name, model_path in trained_map.items():
        cli.console.print(f"\n[bold]Running inference: {model_name}[/bold]")

        # Load id2label from model config
        model_id2label: Dict[int, str] = {}
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    cfg = json.loads(fh.read())
                raw_map = cfg.get("id2label", {})
                model_id2label = {int(k): v for k, v in raw_map.items()}
            except Exception:
                pass
        if not model_id2label:
            # Fallback from test_doccano_info
            model_id2label = {int(k): v for k, v in (test_doccano_info.get("label_mapping") or {}).items()}

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        except Exception as exc:
            cli.console.print(f"[red]Failed to load {model_name}: {exc}[/red]")
            continue

        updated = 0
        batch_size = 32
        texts = [ex.get("text", "") for ex in examples]

        with Progress(console=cli.console) as prog:
            task = prog.add_task(f"Inference: {model_name}", total=len(examples))
            for i in range(0, len(examples), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_examples = examples[i : i + batch_size]

                try:
                    inputs = tokenizer(
                        batch_texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt",
                    ).to(device)

                    t0 = _time.time()
                    with torch.no_grad():
                        outputs = model(**inputs)
                    inference_time = (_time.time() - t0) / len(batch_texts)

                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().tolist()
                except Exception as exc:
                    cli.console.print(f"[yellow]Batch inference failed: {exc}[/yellow]")
                    prog.advance(task, advance=len(batch_texts))
                    continue

                for j, (ex, prob_list) in enumerate(zip(batch_examples, probs)):
                    pred_idx = int(max(range(len(prob_list)), key=lambda k: prob_list[k]))
                    pred_label = model_id2label.get(pred_idx, f"class_{pred_idx}")
                    confidence = prob_list[pred_idx]

                    prob_dict = {}
                    for idx, p in enumerate(prob_list):
                        lbl = model_id2label.get(idx, f"class_{idx}")
                        prob_dict[lbl] = round(p, 4)

                    extra_meta = {
                        "model_annotations": {
                            model_name: {
                                "predicted_label": pred_label,
                                "confidence": round(confidence, 4),
                                "probabilities": prob_dict,
                                "inference_time": round(inference_time, 4),
                            }
                        }
                    }

                    # Deep-merge existing model_annotations if present
                    existing_meta = ex.get("meta", {})
                    existing_model_annots = existing_meta.get("model_annotations", {})
                    if existing_model_annots:
                        merged = dict(existing_model_annots)
                        merged[model_name] = extra_meta["model_annotations"][model_name]
                        extra_meta["model_annotations"] = merged

                    ok = update_doccano_llm_annotation(
                        infer_api_url, token, project_id,
                        ex["id"], existing_meta,
                        new_annotation={}, label_names=[],
                        extra_meta=extra_meta,
                        merge_only=True,
                    )
                    if ok:
                        updated += 1

                prog.advance(task, advance=len(batch_texts))

        stats[model_name] = updated

        # Cleanup
        del model, tokenizer
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 5. Summary ───────────────────────────────────
    results_table = Table(title="Model Annotation Results", show_header=True, header_style="bold")
    results_table.add_column("Model")
    results_table.add_column("Updated", justify="right")
    for mname, count in stats.items():
        results_table.add_row(mname, f"{count:,}")
    cli.console.print(results_table)

    return {"stats": stats, "project_id": project_id}


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

    factory_metadata_state: Optional[Dict[str, Any]] = None
    factory_metadata_targets: Optional[List[Path]] = None

    metadata_root = session_dirs.get("metadata", session_dirs["base"] / "metadata")
    metadata_targets: List[Path] = []
    fallback_metadata_path = metadata_root / "session_state.json"
    metadata_targets.append(fallback_metadata_path)
    session_started_at = datetime.now().strftime('%Y%m%d_%H%M%S')
    placeholder_metadata = _compose_metadata_payload(
        session_id=session_id,
        timestamp=session_started_at,
        workflow="The Annotator - Smart Annotate",
        status="draft",
        data_source={},
        model_configuration={},
        prompts=[],
        processing_configuration={},
        output_config={},
        export_preferences={},
        progress={"requested": None, "completed": 0, "remaining": None},
    )
    _persist_metadata_bundle(metadata_targets, placeholder_metadata)

    cli.console.print("\n[bold cyan]Smart Annotate - Guided Wizard[/bold cyan]\n")

    # Step 1: Data Selection
    cli.console.print("[bold]Step 1/7: Data Selection[/bold]")

    if not cli.detected_datasets:
        cli.console.print("[yellow]No datasets auto-detected.[/yellow]")
        data_path = Path(cli._prompt_file_path("Dataset path"))
    else:
        cli.console.print(f"\n[bold cyan]Found {len(cli.detected_datasets)} dataset(s):[/bold cyan]\n")

        # Create table for datasets
        datasets_table = Table(border_style="cyan", show_header=True, expand=True)
        datasets_table.add_column("#", style="bold yellow", width=4, justify="right")
        datasets_table.add_column("Filename", style="white", no_wrap=True)
        datasets_table.add_column("Format", style="green", no_wrap=True, justify="center")
        datasets_table.add_column("Size", style="magenta", no_wrap=True, justify="right")
        datasets_table.add_column("Rows", style="cyan", no_wrap=True, justify="right")
        datasets_table.add_column("Columns", style="blue", no_wrap=True, justify="right")

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

    dataset_name = data_path.stem
    staging_dir = Path(session_dirs['annotated_data'])
    staging_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    metadata_data_source: Dict[str, Any] = {
        'file_path': str(data_path),
        'file_name': data_path.name,
        'data_format': data_format,
        'dataset_name': dataset_name,
        'text_column': None,
        'identifier_column': None,
        'identifier_source': None,
        'total_rows': 'all',
        'requested_rows': 'all',
        'available_rows': None,
        'sampling_strategy': 'none',
        'sample_seed': None,
    }
    metadata_model_config: Dict[str, Any] = {}
    metadata_prompts_payload: List[Dict[str, Any]] = []
    metadata_processing_config: Dict[str, Any] = {
        'parallel_workers': None,
        'batch_size': None,
        'incremental_save': None,
        'openai_batch_mode': False,
        'openai_batch_dir': None,
        'openai_batch_archive_dir': None,
        'identifier_column': None,
        'auto_identifier_column': None,
        'identifier_source': None,
        'provider_folder': None,
        'model_folder': None,
        'dataset_name': dataset_name,
        'context_window_config': None,  # Context window configuration for surrounding sentences
    }
    metadata_output_config: Dict[str, Any] = {}
    metadata_export_preferences: Dict[str, Any] = {
        'export_to_doccano': False,
        'export_to_labelstudio': False,
        'export_sample_size': 'all',
        'prediction_mode': 'with',
        'labelstudio_direct_export': False,
        'labelstudio_api_url': None,
        'labelstudio_api_key': None,
        'validation_lab': {
            'enabled': False,
            'mode': AnnotationMode.ANNOTATOR.value,
            'session_dir': None,
            'exports': {},
        },
    }
    metadata_progress: Dict[str, Any] = {
        'requested': 'all',
        'completed': 0,
        'remaining': 'all',
    }
    metadata_path: Optional[Path] = None
    early_dataset_snapshot = _compose_metadata_payload(
        session_id=session_id,
        timestamp=run_timestamp,
        workflow="The Annotator - Smart Annotate",
        status="draft",
        data_source=metadata_data_source,
        model_configuration=metadata_model_config,
        prompts=metadata_prompts_payload,
        processing_configuration=metadata_processing_config,
        output_config=metadata_output_config,
        export_preferences=metadata_export_preferences,
        progress=metadata_progress,
    )
    _persist_metadata_bundle(metadata_targets, early_dataset_snapshot)
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
        col_table = Table(border_style="blue", expand=True)
        col_table.add_column("#", style="cyan", width=3)
        col_table.add_column("Column", style="white", no_wrap=True)
        col_table.add_column("Confidence", style="yellow", no_wrap=True)
        col_table.add_column("Avg Length", style="green", no_wrap=True)
        col_table.add_column("Sample", style="dim", ratio=1, overflow="fold")

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
        # JSON files don't support nrows - load full file then slice
        df_for_id = pd.read_json(data_path, lines=False)
        if len(df_for_id) > 1000:
            df_for_id = df_for_id.head(1000)
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
        step_label="Step 2b/7: Identifier Column Selection",
        data_path=data_path
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
    metadata_data_source['text_column'] = text_column
    metadata_data_source['identifier_column'] = auto_identifier_column
    metadata_data_source['identifier_source'] = identifier_source
    metadata_processing_config['identifier_column'] = auto_identifier_column
    metadata_processing_config['auto_identifier_column'] = auto_identifier_column
    metadata_processing_config['identifier_source'] = identifier_source

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
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    provider_folder = provider.replace("/", "_")
    model_folder = safe_model_name
    metadata_data_source['provider_folder'] = provider_folder
    metadata_data_source['model_folder'] = model_folder
    metadata_processing_config['provider_folder'] = provider_folder
    metadata_processing_config['model_folder'] = model_folder

    metadata_subdir = session_dirs['metadata'] / provider_folder / model_folder / dataset_name
    metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{run_timestamp}.json"
    metadata_path = metadata_subdir / metadata_filename
    if metadata_path not in metadata_targets:
        metadata_targets.append(metadata_path)

    default_output_path = staging_dir / f"{data_path.stem}_{safe_model_name}_annotations_{run_timestamp}.{data_format}"
    metadata_output_config = {
        'output_path': str(default_output_path),
        'output_format': data_format,
    }

    metadata_model_config.update({
        'provider': provider,
        'model_name': model_name,
    })

    # Persist early snapshot to allow relaunch before annotation starts.
    early_metadata = _compose_metadata_payload(
        session_id=session_id,
        timestamp=run_timestamp,
        workflow="The Annotator - Smart Annotate",
        status="draft",
        data_source=metadata_data_source,
        model_configuration=metadata_model_config,
        prompts=metadata_prompts_payload,
        processing_configuration=metadata_processing_config,
        output_config=metadata_output_config,
        export_preferences=metadata_export_preferences,
        progress=metadata_progress,
    )
    _persist_metadata_bundle(metadata_targets, early_metadata)

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
        cli.console.print("\n  [cyan]wizard[/cyan]  - Create NEW prompt using Social Science Wizard")
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
                            cli.console.print(f"[yellow][!] Skipping invalid number: '{idx_str}'[/yellow]")
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
        cli.console.print("  [cyan]wizard[/cyan] - Create prompt using Social Science Wizard (Recommended)")
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

    # Context Window Configuration - ask after prompt selection
    context_window_config = _ask_context_window_config(cli, selected_prompts)
    metadata_processing_config['context_window_config'] = context_window_config

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
    metadata_prompts_payload = [
        {
            'name': pc['prompt']['name'],
            'file_path': str(pc['prompt'].get('path')) if pc['prompt'].get('path') else None,
            'expected_keys': pc['prompt']['keys'],
            'prefix': pc['prefix'],
            'prompt_content': pc['prompt']['content'],
        }
        for pc in prompt_configs
    ]

    snapshot_after_prompts = _compose_metadata_payload(
        session_id=session_id,
        timestamp=run_timestamp,
        workflow="The Annotator - Smart Annotate",
        status="draft",
        data_source=metadata_data_source,
        model_configuration=metadata_model_config,
        prompts=metadata_prompts_payload,
        processing_configuration=metadata_processing_config,
        output_config=metadata_output_config,
        export_preferences=metadata_export_preferences,
        progress=metadata_progress,
    )
    _persist_metadata_bundle(metadata_targets, snapshot_after_prompts)

    # Step 1.5: Advanced Options
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.5:[/bold cyan] [bold white]Advanced Options[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Configure processing settings, Doccano integration, and scope.[/dim]")

    # ============================================================
    # DOCCANO INTEGRATION (Step 1.5 — asked first)
    # ============================================================
    doccano_sync_client = None
    doccano_rewrite_ids = None
    doccano_enabled = False

    cli.console.print("\n[bold cyan]Doccano Integration[/bold cyan]")
    cli.console.print(
        "[dim]Doccano is an open-source annotation platform.  When enabled, every\n"
        "annotation produced by the LLM is pushed [bold]in real-time[/bold] to a Doccano\n"
        "project via your Infer API server.  This lets you:[/dim]\n"
        "\n"
        "  [cyan]1.[/cyan] [white]Review & correct[/white]  — human annotators can validate LLM outputs directly in Doccano\n"
        "  [cyan]2.[/cyan] [white]Track progress[/white]    — watch annotations appear live as the pipeline runs\n"
        "  [cyan]3.[/cyan] [white]Compare sources[/white]   — rewrite mode updates existing annotations for A/B comparisons\n"
    )

    if Confirm.ask("[bold yellow]Enable Doccano integration?[/bold yellow]", default=False):
        saved_url = cli.settings.get_infer_api_url()
        saved_username = cli.settings.get_infer_api_username()
        saved_password = cli.settings.get_infer_api_password()

        creds = _infer_api_connect(cli, saved_url, saved_username, saved_password)
        if creds is not None:
            infer_api_url, token = creds

            # Gather labels from prompt configs
            _labels = []
            for pc in prompt_configs:
                keys = pc.get('prompt', {}).get('keys', [])
                _labels.extend(keys)
            _labels = list(dict.fromkeys(_labels))

            cli.console.print("\n[bold]Doccano mode:[/bold]")
            cli.console.print("  [magenta]rewrite[/magenta] — Rewrite LLM annotations only (match first, annotate matched rows)")
            cli.console.print("  [blue]sync[/blue]    — Live sync (create / overwrite / append / rewrite / delete)")
            doccano_mode = Prompt.ask(
                "[bold yellow]Choose mode[/bold yellow]",
                choices=["rewrite", "sync"],
                default="rewrite",
            )

            if doccano_mode == "rewrite":
                result = _doccano_rewrite_filter(
                    cli, infer_api_url, token,
                    data_path, data_format, text_column, identifier_column, _labels,
                )
                if result is not None:
                    doccano_sync_client, doccano_rewrite_ids = result
                    doccano_enabled = True
                    cli.console.print(
                        f"\n[green]Annotation scope restricted to "
                        f"{len(doccano_rewrite_ids)} matched rows[/green]"
                    )
                else:
                    cli.console.print("[yellow]Continuing without Doccano.[/yellow]")
            else:
                _ds_name = data_path.stem
                _safe_model = model_name.replace(':', '_').replace('/', '_')
                _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                default_project_name = f"{_ds_name}_{_safe_model}_{_ts}"

                doccano_sync_client = _doccano_sync_setup(
                    cli, infer_api_url, token, default_project_name, _labels,
                    data_path=data_path,
                    text_column=text_column,
                    identifier_column=identifier_column,
                )
                if doccano_sync_client:
                    doccano_enabled = True
        else:
            cli.console.print("[yellow]Continuing without Doccano sync.[/yellow]")

    # ============================================================
    # DATASET SCOPE (always asked, independent of Doccano)
    # ============================================================
    annotation_limit = None
    use_sample = False
    sample_strategy = "head"
    recommended_sample = None
    total_rows = None

    # Get total rows if possible
    if column_info.get('df') is not None:
        total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

    if doccano_rewrite_ids is not None:
        # Rewrite mode locks scope to matched rows
        cli.console.print(
            f"\n[bold cyan]Dataset Scope[/bold cyan]  "
            f"[green]locked to {len(doccano_rewrite_ids)} matched Doccano rows[/green]"
        )
    else:
        cli.console.print("\n[bold cyan]Dataset Scope[/bold cyan]")
        cli.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

        if total_rows:
            cli.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

        cli.console.print("  [cyan]all[/cyan]   — Annotate the entire dataset")
        cli.console.print("  [cyan]limit[/cyan] — Specify exact number of rows to annotate")

        scope_choice = Prompt.ask(
            "\n[bold yellow]Annotate entire dataset or limit rows?[/bold yellow]",
            choices=["all", "limit"],
            default="all"
        )

        if scope_choice == "limit":
            if total_rows and total_rows > 1000:
                cli.console.print("\n[yellow]Representative Sample Calculation[/yellow]")
                cli.console.print("  Calculate statistically representative sample size (95% confidence interval)")
                cli.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

                calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

                if calculate_sample:
                    import math
                    z = 1.96
                    p = 0.5
                    e = 0.05
                    n_infinite = (z**2 * p * (1-p)) / (e**2)
                    n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                    recommended_sample = int(math.ceil(n_adjusted))

                    cli.console.print(f"\n[green]Recommended sample size: {recommended_sample} rows[/green]")
                    cli.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")
                    cli.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

            default_limit = recommended_sample if recommended_sample else 100
            annotation_limit = cli._int_prompt_with_validation(
                f"How many rows to annotate?",
                default=default_limit,
                min_value=1,
                max_value=total_rows if total_rows else 1000000
            )

            if recommended_sample and annotation_limit == recommended_sample:
                use_sample = True

            cli.console.print("\n[yellow]Sampling Strategy[/yellow]")
            cli.console.print("  [cyan]head[/cyan]   — first N rows (sequential, fast)")
            cli.console.print("  [cyan]random[/cyan] — random N rows (unbiased, representative)")

            sample_strategy = Prompt.ask(
                "Strategy",
                choices=["head", "random"],
                default="random" if use_sample else "head"
            )

    # Effective annotation count (for Doccano scope display)
    _effective_rows = annotation_limit if annotation_limit else total_rows

    # ============================================================
    # DOCCANO PUSH SCOPE (only when live sync is active)
    # ============================================================
    if doccano_enabled and doccano_sync_client and doccano_rewrite_ids is None:
        cli.console.print("\n[bold cyan]Doccano Push Scope[/bold cyan]")
        if _effective_rows:
            cli.console.print(
                f"[dim]You will annotate [bold]{_effective_rows:,}[/bold] rows.  "
                f"Choose how many to push to Doccano.[/dim]\n"
            )
        else:
            cli.console.print("[dim]Choose how many annotated rows to push to Doccano.[/dim]\n")
        cli.console.print("  [cyan]all[/cyan]      — Push every annotated row to Doccano")
        cli.console.print("  [cyan]sample[/cyan]   — Push a representative subset (95 % CI, 5 % margin)")
        cli.console.print("  [cyan]limit[/cyan]    — Push a specific number of rows")

        doccano_push_scope = Prompt.ask(
            "\n[bold yellow]Push scope[/bold yellow]",
            choices=["all", "sample", "limit"],
            default="all",
        )

        if doccano_push_scope == "sample" and _effective_rows and _effective_rows > 30:
            import math
            z, p, e = 1.96, 0.5, 0.05
            n_inf = (z ** 2 * p * (1 - p)) / (e ** 2)
            n_adj = n_inf / (1 + ((n_inf - 1) / _effective_rows))
            doccano_push_limit = int(math.ceil(n_adj))
            doccano_sync_client.set_push_sample(_effective_rows, doccano_push_limit)
            cli.console.print(
                f"\n[green]Doccano will receive {doccano_push_limit:,} / {_effective_rows:,} rows "
                f"(random representative sample — 95 % CI, 5 % margin)[/green]"
            )
        elif doccano_push_scope == "limit":
            max_push = _effective_rows if _effective_rows else 1000000
            doccano_push_limit = cli._int_prompt_with_validation(
                "How many rows to push to Doccano?",
                default=min(100, max_push),
                min_value=1,
                max_value=max_push,
            )
            if _effective_rows:
                doccano_sync_client.set_push_sample(_effective_rows, doccano_push_limit)
            else:
                doccano_sync_client.push_limit = doccano_push_limit
            cli.console.print(
                f"\n[green]Doccano will receive up to {doccano_push_limit:,} rows "
                f"(randomly selected)[/green]"
            )
        else:
            cli.console.print("\n[green]All annotated rows will be pushed to Doccano[/green]")

    # ============================================================
    # PARALLEL PROCESSING
    # ============================================================
    if openai_batch_mode:
        cli.console.print("\n[bold cyan]Processing[/bold cyan]")
        cli.console.print(
            "[dim]OpenAI Batch mode manages concurrency, retries, and persistence. "
            "Local parallelism and incremental save settings are skipped.[/dim]\n"
        )
        num_processes = 1
        save_incrementally = False
        batch_size = 1
    else:
        cli.console.print("\n[bold cyan]Parallel Processing[/bold cyan]")
        cli.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

        cli.console.print("[yellow]Parallel Workers:[/yellow]")
        cli.console.print("  Number of simultaneous annotation processes")
        cli.console.print("\n  [red]IMPORTANT:[/red]")
        cli.console.print("  [dim]Most local machines can only handle 1 worker for LLM inference[/dim]")
        cli.console.print("  [dim]Parallel processing is mainly useful for API models[/dim]")
        cli.console.print("\n  • [cyan]1 worker[/cyan]  - Sequential processing")
        cli.console.print("           [dim]Recommended for: Local models (Ollama), first time users, debugging[/dim]")
        cli.console.print("  • [cyan]2-4 workers[/cyan] - Moderate parallelism")
        cli.console.print("           [dim]Recommended for: API models (OpenAI, Claude) - avoid rate limits[/dim]")
        cli.console.print("  • [cyan]4-8 workers[/cyan] - High parallelism")
        cli.console.print("           [dim]Recommended for: API models only - requires high rate limits[/dim]")

        num_processes = cli._int_prompt_with_validation("Parallel workers", 1, 1, 16)

        cli.console.print("\n[bold cyan]Incremental Save[/bold cyan]")
        cli.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

        cli.console.print("[yellow]Enable incremental save?[/yellow]")
        cli.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
        cli.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
        cli.console.print("  • [red]No[/red]  - Save only at the end")
        cli.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

        save_incrementally = Confirm.ask("\nEnable incremental save?", default=True)

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

    if factory_metadata_state is not None and factory_metadata_targets is not None:
        processing_section = factory_metadata_state.setdefault('processing_configuration', {})
        processing_section.update({
            'parallel_workers': None if openai_batch_mode else num_processes,
            'batch_size': None if openai_batch_mode else batch_size,
            'incremental_save': None if openai_batch_mode else save_incrementally,
            'openai_batch_mode': openai_batch_mode,
        })
        factory_metadata_state.setdefault('annotation_progress', {}).update({
            'requested': factory_metadata_state.get('annotation_progress', {}).get('requested', 'all'),
            'completed': 0,
            'remaining': 'all',
        })
        _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    cli.console.print("\n[bold cyan]Model Parameters[/bold cyan]")
    cli.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

    # Check if model supports parameter tuning
    model_name_lower = model_name.lower()
    is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
    is_gpt5_series = provider == 'openai' and model_name_lower.startswith('gpt-5')
    is_locked_openai = False

    supports_params = not (is_o_series or is_locked_openai or is_gpt5_series)

    if not supports_params:
        if is_o_series:
            cli.console.print(f"[yellow][!] Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
        elif is_gpt5_series:
            cli.console.print(f"[yellow][!] Model '{model_name}' uses locked sampling parameters[/yellow]")
            cli.console.print("[dim]   Temperature and top_p are fixed to 1.0; only max tokens can be adjusted.[/dim]")
        else:
            cli.console.print(f"[yellow][!] Model '{model_name}' does not allow temperature/top_p overrides[/yellow]")
            cli.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")

        configure_params = False
    else:
        cli.console.print("[yellow]Configure model parameters?[/yellow]")
        cli.console.print("  Adjust how the model generates responses")
        cli.console.print("  [dim]• Default values work well for most cases[/dim]")
        cli.console.print("  [dim]• Advanced users can fine-tune for specific needs[/dim]")
        configure_params = Confirm.ask("\nConfigure model parameters?", default=False)

    # Default values
    temperature = 1.0 if not supports_params else 0.7
    max_tokens = 1000
    top_p = 1.0
    top_k = 40

    if configure_params:
        cli.console.print("\n[bold]Parameter Explanations:[/bold]\n")

        # Temperature
        cli.console.print("[cyan]Temperature (0.0 - 2.0):[/cyan]")
        cli.console.print("  Controls randomness in responses")
        cli.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
        cli.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
        cli.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
        cli.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
        cli.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
        cli.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
        temperature = FloatPrompt.ask("Temperature", default=0.7)

        # Max tokens
        cli.console.print("\n[cyan]Max Tokens:[/cyan]")
        cli.console.print("  Maximum length of the response")
        cli.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
        cli.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
        cli.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
        cli.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

        # Top_p (nucleus sampling)
        cli.console.print("\n[cyan]Top P (0.0 - 1.0):[/cyan]")
        cli.console.print("  Nucleus sampling - alternative to temperature")
        cli.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
        cli.console.print("           [dim]More deterministic, safer outputs[/dim]")
        cli.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
        cli.console.print("           [dim]More creative, diverse outputs[/dim]")
        cli.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
        top_p = FloatPrompt.ask("Top P", default=1.0)

        # Top_k (only for some models)
        if provider in ['ollama', 'google']:
            cli.console.print("\n[cyan]Top K:[/cyan]")
            cli.console.print("  Limits vocabulary to K most likely next tokens")
            cli.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
            cli.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
            cli.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
            top_k = cli._int_prompt_with_validation("Top K", 40, 1, 100)

    if is_gpt5_series:
        cli.console.print("\n[cyan]Max Tokens (GPT-5 Series):[/cyan]")
        cli.console.print("  Maximum response length; temperature/top_p remain fixed at 1.0.")
        cli.console.print("  [dim]Note: higher values increase API usage.[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", max_tokens, 50, 8000)
        temperature = 1.0
        top_p = 1.0
        top_k = None

    if factory_metadata_state is not None and factory_metadata_targets is not None:
        model_config_section = factory_metadata_state.setdefault('model_configuration', {})
        model_config_section.update({
            'annotation_mode': 'openai_batch' if (provider == 'openai' and openai_batch_mode) else (
                'api' if provider in {'openai', 'anthropic', 'google'} else 'local'
            ),
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k if provider in ['ollama', 'google'] else None,
        })
        _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

    tracker.mark_step(
        5,
        detail="Advanced options configured",
    )
    metadata_processing_config['parallel_workers'] = None if openai_batch_mode else num_processes
    metadata_processing_config['batch_size'] = None if openai_batch_mode else batch_size
    metadata_processing_config['incremental_save'] = None if openai_batch_mode else save_incrementally
    metadata_processing_config['openai_batch_mode'] = openai_batch_mode

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

    metadata_model_config.update({
        'annotation_mode': annotation_mode,
        'openai_batch_mode': openai_batch_mode,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
    })

    cost_estimate: Optional[CostEstimate] = None
    cost_estimate_error: Optional[Exception] = None
    cost_pricing: Optional[ModelPricing] = None
    cost_estimate, cost_estimate_error, cost_pricing = _compute_cost_estimate_for_selection(
        provider=provider,
        model_name=model_name,
        data_path=data_path,
        data_format=data_format,
        text_columns=[text_column],
        prompt_configs=prompt_configs,
        annotation_limit=annotation_limit,
        sample_strategy=sample_strategy,
        max_tokens=max_tokens,
        prompt_header=DEFAULT_PROMPT_HEADER,
    )

    if cost_estimate:
        metadata_cfg = locals().get('metadata_processing_config')
        if isinstance(metadata_cfg, dict):
            metadata_cfg['estimated_cost_usd'] = round(cost_estimate.total_cost, 6)
            metadata_cfg['estimated_input_tokens'] = cost_estimate.input_tokens
            metadata_cfg['estimated_output_tokens'] = cost_estimate.output_tokens_estimated
            metadata_cfg['pricing_last_updated'] = cost_estimate.pricing_last_updated

    # Display comprehensive summary
    summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True, expand=True)
    summary_table.add_column("Category", style="bold cyan", no_wrap=True)
    summary_table.add_column("Setting", style="yellow", no_wrap=True)
    summary_table.add_column("Value", style="white", ratio=1, overflow="fold")

    # Data section
    summary_table.add_row("Data", "Dataset", str(data_path.name))
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
    summary_table.add_row("Model", "Provider/Model", f"{provider}/{model_name}")
    summary_table.add_row("", "Temperature", f"{temperature}")
    summary_table.add_row("", "Max Tokens", f"{max_tokens}")
    if configure_params:
        summary_table.add_row("", "Top P", f"{top_p}")
        if provider in ['ollama', 'google']:
            summary_table.add_row("", "Top K", f"{top_k}")

    # Prompts section
    summary_table.add_row("Prompts", "Count", f"{len(prompt_configs)}")
    for i, pc in enumerate(prompt_configs, 1):
        prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
        summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

    # Processing section
    if openai_batch_mode:
        summary_table.add_row(
            "Processing",
            "Annotation Mode",
            "OpenAI Batch (OpenAI-managed async job)"
        )
        summary_table.add_row("", "Parallel Workers", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Batch Size", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Incremental Save", "N/A (handled after batch completion)")
    else:
        summary_table.add_row("Processing", "Annotation Mode", annotation_mode_display)
        summary_table.add_row("", "Parallel Workers", str(num_processes))
        summary_table.add_row("", "Batch Size", str(batch_size))
        summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

    cli.console.print("\n")
    cli.console.print(summary_table)

    _display_cost_estimate_panel(
        cli.console,
        cost_estimate=cost_estimate,
        cost_pricing=cost_pricing,
        cost_estimate_error=cost_estimate_error,
        provider=provider,
        model_name=model_name,
    )

    if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
        return

    # ============================================================
    # REPRODUCIBILITY METADATA
    # ============================================================
    cli.console.print("\n[bold cyan]Reproducibility & Metadata[/bold cyan]")
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
    metadata_state: Optional[Dict[str, Any]] = None
    # Use the most recent metadata target if one exists (second entry is the primary file)
    metadata_path: Optional[Path] = metadata_targets[-1] if metadata_targets else None

    # ============================================================
    # VALIDATION TOOL EXPORT OPTION
    # ============================================================
    validation_lab_session_dirs: Optional[Dict[str, Path]] = None
    validation_lab_exports: Dict[str, str] = {}

    validation_hint = f"validation/{session_id}/doccano"

    # Initialize export flags
    export_to_doccano = False
    export_to_labelstudio = False
    export_sample_size = None
    labelstudio_direct_export = False
    labelstudio_api_url = None
    labelstudio_api_key = None
    prediction_mode = "with"
    validation_lab_enabled = False
    validation_dataset_label = (
        metadata_data_source.get('file_name')
        or metadata_data_source.get('dataset_name')
        or metadata_data_source.get('display_name')
    )

    if doccano_enabled and doccano_sync_client:
        # Live sync already active — skip the export question
        cli.console.print(
            "\n[bold cyan]Validation Lab Preparation[/bold cyan]  "
            "[green]Doccano live sync is active — annotations are pushed in real-time.[/green]"
        )
        export_confirm = False
    else:
        cli.console.print("\n[bold cyan]Validation Lab Preparation[/bold cyan]")
        cli.console.print(
            f"[dim]Export annotations, review them manually, then copy the validated JSONL into [white]{validation_hint}[/white] so Validation Lab (Mode 5) can compute agreement metrics.[/dim]\n"
        )

        cli.console.print("[yellow]Available validation tools:[/yellow]")
        cli.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
        cli.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
        cli.console.print("  • Both are open-source and free\n")

        cli.console.print("[green]Workflow with Validation Lab:[/green]")
        cli.console.print("  • Export annotations for human review (Doccano works best with Validation Lab)")
        cli.console.print("  • Make corrections manually, then export the validated JSONL")
        cli.console.print(
            f"  • Drop the validated file into [white]{validation_hint}[/white] and reopen Validation Lab (Mode 5)\n"
        )

        export_confirm = Confirm.ask(
            "[bold yellow]Prepare a Doccano export for Validation Lab?[/bold yellow]\n"
            f"[dim]After manual review in Doccano, copy the validated JSONL into [white]{validation_hint}[/white] and reopen Validation Lab (Mode 5) to compute metrics.[/dim]",
            default=False,
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
            validation_lab_enabled = True
            validation_lab_session_dirs = ensure_validation_session_dirs(session_id)
            metadata_export_preferences['validation_lab'] = {
                'enabled': True,
                'mode': AnnotationMode.ANNOTATOR.value,
                'session_dir': str(validation_lab_session_dirs['session_root']),
                'exports': {},
                'dataset': validation_dataset_label,
            }
        else:  # labelstudio
            export_to_labelstudio = True
            metadata_export_preferences['validation_lab'] = {
                'enabled': False,
                'mode': AnnotationMode.ANNOTATOR.value,
                'session_dir': None,
                'exports': {},
                'dataset': validation_dataset_label,
            }

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

    metadata_export_preferences.update({
        'export_to_doccano': export_to_doccano,
        'export_to_labelstudio': export_to_labelstudio,
        'export_sample_size': export_sample_size if export_sample_size is not None else 'all',
        'prediction_mode': prediction_mode if export_confirm else metadata_export_preferences.get('prediction_mode', 'with'),
        'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
        'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
        'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None,
    })

    snapshot_after_exports = _compose_metadata_payload(
        session_id=session_id,
        timestamp=run_timestamp,
        workflow="The Annotator - Smart Annotate",
        status="draft",
        data_source=metadata_data_source,
        model_configuration=metadata_model_config,
        prompts=metadata_prompts_payload,
        processing_configuration=metadata_processing_config,
        output_config=metadata_output_config,
        export_preferences=metadata_export_preferences,
        progress=metadata_progress,
    )
    _persist_metadata_bundle(metadata_targets, snapshot_after_exports)
    metadata_state = copy.deepcopy(snapshot_after_exports)

    # Doccano integration configured at Step 1.5 (above)

    # ============================================================
    # EXECUTE ANNOTATION
    # ============================================================

    # CRITICAL: Use new organized structure with dataset-specific subfolder
    # Structure: logs/annotator/{session_id}/annotated_data/{dataset_name}/
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata_state.setdefault('annotation_session', {})['timestamp'] = timestamp

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

    cli.console.print(f"\n[bold cyan]Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")
    cli.console.print()
    if factory_metadata_state is not None and factory_metadata_targets is not None:
        factory_metadata_state.setdefault('output', {}).update({
            'output_path': str(default_output_path),
            'output_format': data_format,
        })
        if cost_estimate:
            factory_metadata_state.setdefault('cost', {}).update({
                'estimated_cost_usd': cost_estimate.total_cost,
                'estimated_input_tokens': cost_estimate.input_tokens,
                'estimated_output_tokens': cost_estimate.output_tokens_estimated,
                'pricing_last_updated': cost_estimate.pricing_last_updated,
            })
        _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

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
        'model_display_name': model_name,
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
        'create_annotated_subset': True,  # ensure compact annotated-only export is materialised
        'session_dirs': {key: str(value) for key, value in session_dirs.items()},
        'provider_subdir': str(provider_subdir),
        'model_subdir': str(model_subdir),
        'dataset_subdir': str(dataset_subdir),
        'openai_batch_dir': str(batch_dir),
        'openai_batch_archive_dir': str(archive_dir) if archive_dir else None,
        'provider_folder': provider_folder,
        'model_folder': model_folder,
        'dataset_name': dataset_name,
        'session_id': session_id,
        'context_window_config': context_window_config,  # Context window for surrounding sentences
    }

    # Wire Doccano live sync client + rewrite IDs if enabled
    if doccano_sync_client:
        pipeline_config['doccano_sync_client'] = doccano_sync_client
    if doccano_rewrite_ids is not None:
        pipeline_config['doccano_rewrite_ids'] = doccano_rewrite_ids

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

    if cost_estimate:
        pipeline_config['estimated_cost_usd'] = cost_estimate.total_cost
        pipeline_config['estimated_input_tokens'] = cost_estimate.input_tokens
        pipeline_config['estimated_output_tokens'] = cost_estimate.output_tokens_estimated

    # ============================================================
    # SAVE REPRODUCIBILITY METADATA
    # ============================================================
    if save_metadata:
        available_rows = total_rows if total_rows is not None else None
        if annotation_limit:
            requested_rows = annotation_limit
        elif available_rows is not None:
            requested_rows = available_rows
        else:
            requested_rows = 'all'
        if isinstance(requested_rows, int):
            initial_remaining = requested_rows
        elif isinstance(available_rows, int):
            initial_remaining = available_rows
        else:
            initial_remaining = 'all'

        metadata_data_source.update({
            'total_rows': annotation_limit if annotation_limit else 'all',
            'requested_rows': requested_rows,
            'available_rows': available_rows,
            'sampling_strategy': sample_strategy if annotation_limit else 'none (all rows)',
            'sample_seed': 42 if sample_strategy == 'random' else None,
        })
        metadata_processing_config.update({
            'annotation_sample_size': annotation_limit,
            'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        })
        if cost_estimate:
            metadata_processing_config['estimated_cost_usd'] = round(cost_estimate.total_cost, 6)
            metadata_processing_config['estimated_input_tokens'] = cost_estimate.input_tokens
            metadata_processing_config['estimated_output_tokens'] = cost_estimate.output_tokens_estimated
            metadata_processing_config['pricing_last_updated'] = cost_estimate.pricing_last_updated
        metadata_progress.update({
            'requested': requested_rows,
            'completed': 0,
            'remaining': initial_remaining,
        })
        if annotation_mode == 'openai_batch':
            metadata_processing_config['openai_batch_dir'] = str(batch_dir)
            metadata_processing_config['openai_batch_archive_dir'] = str(archive_dir) if archive_dir else None
        else:
            metadata_processing_config['openai_batch_dir'] = None
            metadata_processing_config['openai_batch_archive_dir'] = None

        metadata_output_config['output_path'] = str(default_output_path)
        metadata_output_config['output_format'] = data_format

        ready_metadata = _compose_metadata_payload(
            session_id=session_id,
            timestamp=run_timestamp,
            workflow="The Annotator - Smart Annotate",
            status="ready",
            data_source=metadata_data_source,
            model_configuration=metadata_model_config,
            prompts=metadata_prompts_payload,
            processing_configuration=metadata_processing_config,
            output_config=metadata_output_config,
            export_preferences=metadata_export_preferences,
            progress=metadata_progress,
        )
        _persist_metadata_bundle(metadata_targets, ready_metadata)
        metadata_state = copy.deepcopy(ready_metadata)

        cli.console.print(f"\n[bold green][OK] Metadata saved for reproducibility[/bold green]")
        cli.console.print(f"[bold cyan]Metadata File:[/bold cyan]")
        cli.console.print(f"   {metadata_path if metadata_path else 'Not available'}\n")
        detail_message = f"Metadata saved: {metadata_path.name}" if metadata_path else "Metadata saved"
        tracker_extra = {
            "dataset_path": str(data_path),
            "text_column": text_column,
            "identifier_column": identifier_column,
        }
        if metadata_path:
            tracker_extra["metadata_path"] = str(metadata_path)
        tracker.mark_step(
            5,
            detail=detail_message,
            extra=tracker_extra,
        )

    # Execute pipeline with Rich progress
    try:
        cli.console.print("\n[bold green]Starting annotation...[/bold green]\n")

        running_metadata = _compose_metadata_payload(
            session_id=session_id,
            timestamp=run_timestamp,
            workflow="The Annotator - Smart Annotate",
            status="running",
            data_source=metadata_data_source,
            model_configuration=metadata_model_config,
            prompts=metadata_prompts_payload,
            processing_configuration=metadata_processing_config,
            output_config=metadata_output_config,
            export_preferences=metadata_export_preferences,
            progress=metadata_progress,
        )
        _persist_metadata_bundle(metadata_targets, running_metadata)

        # Create pipeline controller with session_id for organized logging
        from ..pipelines.pipeline_controller import PipelineController
        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=session_id,  # Pass session_id for organized logging
            session_mode=AnnotationMode.ANNOTATOR.value,
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
                failure_metadata = _compose_metadata_payload(
                    session_id=session_id,
                    timestamp=run_timestamp,
                    workflow="The Annotator - Smart Annotate",
                    status="failed",
                    data_source=metadata_data_source,
                    model_configuration=metadata_model_config,
                    prompts=metadata_prompts_payload,
                    processing_configuration=metadata_processing_config,
                    output_config=metadata_output_config,
                    export_preferences=metadata_export_preferences,
                    progress=metadata_progress,
                )
                _persist_metadata_bundle(metadata_targets, failure_metadata)
                tracker.mark_step(
                    6,
                    status="failed",
                    detail=f"Annotation error: {error_msg}",
                    overall_status="failed",
                    extra={"error": error_msg},
                )
                cli.console.print(f"\n[bold red][FAIL] Error:[/bold red] {error_msg}")
                cli.console.print("[dim]Press Enter to return to menu...[/dim]")
                input()
                return

        # Get results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))
        output_paths = _persist_annotation_outputs(
            cli,
            source_output_path=output_file,
            session_dirs=session_dirs,
            provider_folder=provider_folder,
            model_folder=model_folder,
            dataset_name=dataset_name,
            annotation_results=annotation_results,
        )
        if output_paths.get("full_path"):
            output_file = output_paths["full_path"]
            annotation_results['output_file'] = output_file
        subset_path = output_paths.get("annotations_only_path") or annotation_results.get('annotated_subset_path')
        if metadata_state:
            metadata_state = copy.deepcopy(metadata_state)
            metadata_state.setdefault('annotation_session', {})['status'] = 'completed'
            metadata_state.setdefault('output', {})['output_path'] = str(output_file)
            total_annotated = annotation_results.get('total_annotated')
            progress_section = metadata_state.setdefault('annotation_progress', {})
            if isinstance(total_annotated, int):
                progress_section['completed'] = total_annotated
                requested_value = progress_section.get('requested')
                if isinstance(requested_value, int):
                    progress_section['remaining'] = max(requested_value - total_annotated, 0)
                else:
                    progress_section['remaining'] = 0
            _persist_metadata_bundle(metadata_targets, metadata_state)

        # Display success message
        cli.console.print("\n[bold green][OK] Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        # Display statistics if available
        total_annotated = annotation_results.get('total_annotated', 0)
        status_breakdown = annotation_results.get('status_breakdown')
        if isinstance(status_breakdown, dict):
            total_prompt_events = sum(
                int(value)
                for value in status_breakdown.values()
                if isinstance(value, (int, float))
            )
        else:
            status_breakdown = {}
            total_prompt_events = 0

        prompt_success_count = annotation_results.get('prompt_success_count')
        if isinstance(prompt_success_count, (int, float)):
            prompt_success_count = int(prompt_success_count)
        else:
            prompt_success_count = None

        if total_annotated:
            cli.console.print(f"\n[bold cyan]Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if isinstance(success_count, (int, float)) and success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

            if (
                prompt_success_count is not None
                and total_prompt_events
                and (
                    total_prompt_events != total_annotated
                    or prompt_success_count != success_count
                )
            ):
                prompt_success_rate = (prompt_success_count / total_prompt_events) * 100
                cli.console.print(
                    f"   Prompt success rate: {prompt_success_rate:.1f}% "
                    f"({prompt_success_count:,}/{total_prompt_events:,})"
                )

            mean_time = annotation_results.get('mean_inference_time')
            if isinstance(mean_time, (int, float)):
                cli.console.print(f"   Avg inference time: {mean_time:.2f}s")

        if isinstance(total_annotated, int):
            metadata_progress['completed'] = total_annotated
            requested_value = metadata_progress.get('requested')
            if isinstance(requested_value, int):
                metadata_progress['remaining'] = max(requested_value - total_annotated, 0)
            elif isinstance(metadata_data_source.get('available_rows'), int):
                metadata_progress['remaining'] = max(metadata_data_source['available_rows'] - total_annotated, 0)
            else:
                metadata_progress['remaining'] = 0
        else:
            metadata_progress['completed'] = metadata_progress.get('completed', 0)
        metadata_output_config['output_path'] = str(output_file)

        completed_metadata = _compose_metadata_payload(
            session_id=session_id,
            timestamp=run_timestamp,
            workflow="The Annotator - Smart Annotate",
            status="completed",
            data_source=metadata_data_source,
            model_configuration=metadata_model_config,
            prompts=metadata_prompts_payload,
            processing_configuration=metadata_processing_config,
            output_config=metadata_output_config,
            export_preferences=metadata_export_preferences,
            progress=metadata_progress,
        )
        _persist_metadata_bundle(metadata_targets, completed_metadata)
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
            cli.console.print("\n[bold cyan]Sample Annotations:[/bold cyan]")
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
                "prompt_success_count": prompt_success_count,
                "prompt_status_breakdown": status_breakdown,
            },
        )

        # ============================================================
        # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
        # ============================================================
        if not lang_column:
            cli.console.print("\n[bold cyan]Language Detection for Training[/bold cyan]")
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
                    cli.console.print(f"[yellow][!] Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                if len(df_annotated) == 0:
                    cli.console.print("[yellow][!] No annotated rows found. Skipping language detection.[/yellow]")
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
                            cli.console.print(f"\n[bold]Languages Detected ({total:,} texts):[/bold]")

                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold", expand=True)
                            lang_table.add_column("Language", style="cyan", no_wrap=True)
                            lang_table.add_column("Count", style="yellow", justify="right", no_wrap=True)
                            lang_table.add_column("Percentage", style="green", justify="right", no_wrap=True)

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
                            cli.console.print("[yellow][!] No languages detected successfully[/yellow]")

            except Exception as e:
                cli.console.print(f"[yellow][!] Language detection failed: {e}[/yellow]")
                cli.logger.exception("Language detection failed")


        # Export to Doccano JSONL if requested
        existing_doccano_path = annotation_results.get('doccano_export_path')
        existing_validation_path = annotation_results.get('validation_doccano_export_path')

        doccano_export_result = None
        if export_to_doccano:
            validation_destination = None
            if validation_lab_session_dirs:
                validation_destination = validation_lab_session_dirs['doccano']

            can_reuse_existing = (
                existing_doccano_path
                and Path(existing_doccano_path).exists()
                and prediction_mode == "with"
                and (export_sample_size in (None, "all") or export_sample_size == "all")
            )

            if can_reuse_existing:
                cli.console.print(
                    "\n[cyan]Reusing Doccano export generated during annotation.[/cyan]"
                )
                doccano_export_result = {
                    'jsonl_path': str(existing_doccano_path),
                    'validation_copy_path': (
                        str(existing_validation_path)
                        if existing_validation_path and Path(existing_validation_path).exists()
                        else None
                    ),
                    'reused': True,
                }
            else:
                doccano_export_result = cli._export_to_doccano_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    session_dirs=session_dirs,
                    provider_folder=provider_folder,
                    model_folder=model_folder,
                    prediction_mode=prediction_mode,
                    validation_destination=validation_destination,
                    session_id=session_id,
                    annotation_mode=AnnotationMode.ANNOTATOR.value,
                )
            if doccano_export_result:
                doccano_path = doccano_export_result.get('jsonl_path')
                validation_copy_path = doccano_export_result.get('validation_copy_path')
                if doccano_path:
                    exports_entry = metadata_export_preferences.setdefault('validation_lab', {})
                    exports_dict = exports_entry.setdefault('exports', {})
                    target_path = validation_copy_path or doccano_path
                    exports_dict['doccano'] = str(target_path)
                    validation_lab_exports['doccano'] = str(target_path)

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
                "validation_lab_enabled": validation_lab_enabled,
                "validation_lab_session_dir": str(validation_lab_session_dirs['session_root']) if validation_lab_session_dirs else None,
                "validation_lab_exports": validation_lab_exports,
                "validation_lab_dataset": validation_dataset_label,
            },
        )

        # ============================================================
        # GENERATE ANNOTATION METRICS CHART
        # ============================================================
        try:
            cli.console.print("\n[bold cyan]Generating Annotation Metrics Chart...[/bold cyan]")

            # Determine chart output directory
            chart_output_dir = session_dirs.get('root', Path(output_file).parent) if session_dirs else Path(output_file).parent

            # Get model name from metadata
            model_name_for_chart = model_folder or model_config.get('llm_model') if 'model_config' in dir() else "llm_model"
            if not model_name_for_chart and 'metadata_model_config' in dir():
                model_name_for_chart = metadata_model_config.get('llm_model', 'unknown_model')

            # Load the annotated DataFrame for chart generation
            import pandas as pd
            df_for_chart = pd.read_csv(output_file)

            # Determine annotation column
            annotation_col_for_chart = None
            possible_annotation_cols = ['label', 'labels', 'category', 'annotation', 'predicted_label']
            for col in possible_annotation_cols:
                if col in df_for_chart.columns:
                    annotation_col_for_chart = col
                    break

            # Generate the chart
            chart_path = generate_annotation_chart(
                output_dir=chart_output_dir,
                session_id=session_id,
                model_name=str(model_name_for_chart) if model_name_for_chart else "llm_model",
                df=df_for_chart,
                annotation_column=annotation_col_for_chart,
                annotation_results=annotation_results,
                language_column=lang_column,
                workflow_name="The Annotator",
            )

            if chart_path:
                cli.console.print(f"[green]✓ Annotation metrics chart saved to:[/green] {chart_path}")
            else:
                cli.console.print("[yellow][!] Could not generate annotation metrics chart[/yellow]")

        except Exception as chart_exc:
            cli.logger.warning(f"Failed to generate annotation metrics chart: {chart_exc}")
            cli.console.print(f"[yellow][!] Chart generation failed: {chart_exc}[/yellow]")

        tracker.update_status(
            "completed",
            extra={
                "validation_lab_enabled": validation_lab_enabled,
                "validation_lab_session_dir": str(validation_lab_session_dirs['session_root']) if validation_lab_session_dirs else None,
                "validation_lab_dataset": validation_dataset_label,
            },
        )

        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    except Exception as exc:
        tracker.update_status("failed", note=str(exc))
        cli.console.print(f"\n[bold red][FAIL] Annotation failed:[/bold red] {exc}")
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

    is_factory_workflow = True

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

    metadata_root = session_dirs.get("metadata", session_dirs["base"] / "metadata")
    factory_metadata_targets: List[Path] = []
    factory_fallback_metadata = metadata_root / "session_state.json"
    factory_metadata_targets.append(factory_fallback_metadata)
    factory_session_started_at = datetime.now().strftime('%Y%m%d_%H%M%S')
    factory_placeholder_metadata = _compose_metadata_payload(
        session_id=session_id,
        timestamp=factory_session_started_at,
        workflow="Annotator Factory - Full Pipeline",
        status="draft",
        data_source={},
        model_configuration={},
        prompts=[],
        processing_configuration={},
        output_config={},
        export_preferences={},
        progress={"requested": None, "completed": 0, "remaining": None},
    )
    _persist_metadata_bundle(factory_metadata_targets, factory_placeholder_metadata)
    factory_metadata_state = factory_placeholder_metadata

    # Doccano credentials — set at Step 1.5, reused by Steps 2.5 and 3.5
    factory_doccano_creds: Optional[Tuple[str, str]] = None

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
    cli.console.print("  1. Files (CSV/Excel/JSON/etc.) - Auto-detected or manual")
    cli.console.print("  2. SQL Database (PostgreSQL/MySQL/SQLite/SQL Server)\n")

    data_source_choice = Prompt.ask(
        "Data source",
        choices=["1", "2"],
        default="1"
    )

    use_sql_database = (data_source_choice == "2")

    if use_sql_database:
        # SQL DATABASE WORKFLOW
        cli.console.print("\n[bold cyan]SQL Database (Training Sample)[/bold cyan]\n")
        cli.console.print("[yellow]Note: For training, you'll select a representative sample from your database[/yellow]\n")

        # Database type selection
        db_choices = ["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server"]
        db_table = Table(title="Database Types", border_style="cyan", expand=True)
        db_table.add_column("#", style="cyan", no_wrap=True)
        db_table.add_column("Database Type", style="white", ratio=1, overflow="fold")
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

        tables_table = Table(title="Available Tables", border_style="cyan", expand=True)
        tables_table.add_column("#", style="cyan", width=3)
        tables_table.add_column("Table Name", style="white", no_wrap=True)
        tables_table.add_column("Rows", style="green", justify="right", no_wrap=True)
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

        # Save CSV inside session logs workspace
        staging_dir = Path(session_dirs['annotated_data'])
        staging_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_path = staging_dir / f"sql_{selected_table}_{timestamp}.csv"
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
            cli.console.print(f"\n[bold cyan]Found {len(cli.detected_datasets)} dataset(s):[/bold cyan]\n")

            # Create table for datasets
            datasets_table = Table(border_style="cyan", show_header=True, expand=True)
            datasets_table.add_column("#", style="bold yellow", width=4, justify="right")
            datasets_table.add_column("Filename", style="white", no_wrap=True)
            datasets_table.add_column("Format", style="green", no_wrap=True, justify="center")
            datasets_table.add_column("Size", style="magenta", no_wrap=True, justify="right")
            datasets_table.add_column("Rows", style="cyan", no_wrap=True, justify="right")
            datasets_table.add_column("Columns", style="blue", no_wrap=True, justify="right")

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
        col_table = Table(border_style="blue", expand=True)
        col_table.add_column("#", style="cyan", width=3)
        col_table.add_column("Column", style="white", no_wrap=True)
        col_table.add_column("Confidence", style="yellow", no_wrap=True)
        col_table.add_column("Avg Length", style="green", no_wrap=True)
        col_table.add_column("Sample", style="dim", ratio=1, overflow="fold")

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
        step_label="Step 1.2b/4: Identifier Column Selection",
        data_path=data_path
    )
    identifier_source = "user" if identifier_column else "auto"
    auto_identifier_column = identifier_column or "llm_annotation_id"
    pipeline_identifier_column = identifier_column if identifier_column else None
    identifier_column = pipeline_identifier_column

    # Store column info for later use
    column_info['df'] = df_sample

    factory_metadata_state = _compose_metadata_payload(
        session_id=session_id,
        timestamp=factory_session_started_at,
        workflow="Annotator Factory - Full Pipeline",
        status="draft",
        data_source={
            'file_path': str(data_path),
            'file_name': data_path.name,
            'data_format': data_format,
            'dataset_name': data_path.stem,
            'text_column': text_column,
            'identifier_column': auto_identifier_column,
            'identifier_source': identifier_source,
            'total_rows': 'all',
            'requested_rows': 'all',
            'available_rows': None,
            'sampling_strategy': 'none',
            'sample_seed': None,
        },
        model_configuration={},
        prompts=[],
        processing_configuration={
            'provider_folder': None,
            'model_folder': None,
            'dataset_name': data_path.stem,
            'identifier_column': auto_identifier_column,
        },
        output_config={},
        export_preferences={},
        progress={'requested': 'all', 'completed': 0, 'remaining': 'all'},
    )
    _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

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
    model_config_section = factory_metadata_state.setdefault('model_configuration', {})
    model_config_section.update({
        'provider': provider,
        'model_name': model_name,
        'openai_batch_mode': openai_batch_mode,
    })
    processing_section = factory_metadata_state.setdefault('processing_configuration', {})
    processing_section.update({
        'provider_folder': provider.replace("/", "_"),
        'model_folder': model_name.replace(":", "_").replace("/", "_"),
        'dataset_name': data_path.stem,
    })
    _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

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
        cli.console.print("\n  [cyan]wizard[/cyan]  - Create NEW prompt using Social Science Wizard")
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
                            cli.console.print(f"[yellow][!] Skipping invalid number: '{idx_str}'[/yellow]")
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
        cli.console.print("  [cyan]wizard[/cyan] - Create prompt using Social Science Wizard (Recommended)")
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

    # Context Window Configuration - ask after prompt selection
    context_window_config = _ask_context_window_config(cli, selected_prompts)
    factory_metadata_state['processing_configuration']['context_window_config'] = context_window_config

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

    prompts_payload_for_metadata: List[Dict[str, Any]] = []
    for entry in prompt_configs:
        prompt_entry = entry['prompt']
        prompts_payload_for_metadata.append({
            'name': prompt_entry.get('name'),
            'file_path': str(prompt_entry.get('path')) if prompt_entry.get('path') else None,
            'expected_keys': prompt_entry.get('keys', []),
            'prefix': entry.get('prefix'),
            'prompt_content': prompt_entry.get('content'),
        })
    factory_metadata_state['prompts'] = prompts_payload_for_metadata
    _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

    # Step 1.5: Advanced Options
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.5:[/bold cyan] [bold white]Advanced Options[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Configure processing settings, Doccano integration, and scope.[/dim]")

    # ============================================================
    # DOCCANO INTEGRATION (Step 1.5 — live sync only in Factory)
    # ============================================================
    doccano_sync_client = None
    doccano_enabled = False

    cli.console.print("\n[bold cyan]Doccano Integration[/bold cyan]")
    cli.console.print(
        "[dim]Doccano is an open-source annotation platform.  When enabled, every\n"
        "annotation produced by the LLM is pushed [bold]in real-time[/bold] to a Doccano\n"
        "project via your Infer API server.  This lets you:[/dim]\n"
        "\n"
        "  [cyan]1.[/cyan] [white]Review & correct[/white]  — human annotators can validate LLM outputs directly in Doccano\n"
        "  [cyan]2.[/cyan] [white]Track progress[/white]    — watch annotations appear live as the pipeline runs\n"
        "  [cyan]3.[/cyan] [white]Full pipeline[/white]     — later stages can push test sets and model predictions\n"
        "                       to the same server for side-by-side comparison\n"
    )

    if Confirm.ask("[bold yellow]Enable Doccano integration?[/bold yellow]", default=False):
        saved_url = cli.settings.get_infer_api_url()
        saved_username = cli.settings.get_infer_api_username()
        saved_password = cli.settings.get_infer_api_password()

        creds = _infer_api_connect(cli, saved_url, saved_username, saved_password)
        if creds is not None:
            infer_api_url, token = creds
            factory_doccano_creds = creds
            doccano_enabled = True

            # Gather labels from prompt configs
            _labels = []
            for pc in prompt_configs:
                keys = pc.get('prompt', {}).get('keys', [])
                _labels.extend(keys)
            _labels = list(dict.fromkeys(_labels))

            _ds_name = data_path.stem
            _safe_model = model_name.replace(':', '_').replace('/', '_')
            _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_project_name = f"{_ds_name}_{_safe_model}_{_ts}"

            doccano_sync_client = _doccano_sync_setup(
                cli, infer_api_url, token, default_project_name, _labels,
                data_path=data_path,
                text_column=text_column,
                identifier_column=identifier_column,
            )
        else:
            cli.console.print("[yellow]Continuing without Doccano sync.[/yellow]")

    # ============================================================
    # DATASET SCOPE (always asked, independent of Doccano)
    # ============================================================
    annotation_limit = None
    use_sample = False
    sample_strategy = "head"
    recommended_sample = None
    total_rows = None

    # Get total rows if possible
    if column_info.get('df') is not None:
        total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

    cli.console.print("\n[bold cyan]Dataset Scope[/bold cyan]")
    cli.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

    if total_rows:
        cli.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

    cli.console.print("  [cyan]all[/cyan]   — Annotate the entire dataset")
    cli.console.print("  [cyan]limit[/cyan] — Specify exact number of rows to annotate")

    scope_choice = Prompt.ask(
        "\n[bold yellow]Annotate entire dataset or limit rows?[/bold yellow]",
        choices=["all", "limit"],
        default="all"
    )

    if scope_choice == "limit":
        if total_rows and total_rows > 1000:
            cli.console.print("\n[yellow]Representative Sample Calculation[/yellow]")
            cli.console.print("  Calculate statistically representative sample size (95% confidence interval)")
            cli.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

            calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

            if calculate_sample:
                import math
                z = 1.96
                p = 0.5
                e = 0.05
                n_infinite = (z**2 * p * (1-p)) / (e**2)
                n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                recommended_sample = int(math.ceil(n_adjusted))

                cli.console.print(f"\n[green]Recommended sample size: {recommended_sample} rows[/green]")
                cli.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")
                cli.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

        default_limit = recommended_sample if recommended_sample else 100
        annotation_limit = cli._int_prompt_with_validation(
            f"How many rows to annotate?",
            default=default_limit,
            min_value=1,
            max_value=total_rows if total_rows else 1000000
        )

        if recommended_sample and annotation_limit == recommended_sample:
            use_sample = True

        cli.console.print("\n[yellow]Sampling Strategy[/yellow]")
        cli.console.print("  [cyan]head[/cyan]   — first N rows (sequential, fast)")
        cli.console.print("  [cyan]random[/cyan] — random N rows (unbiased, representative)")

        sample_strategy = Prompt.ask(
            "Strategy",
            choices=["head", "random"],
            default="random" if use_sample else "head"
        )

    # Effective annotation count (for Doccano scope display)
    _effective_rows = annotation_limit if annotation_limit else total_rows

    # ============================================================
    # DOCCANO PUSH SCOPE (only when live sync is active)
    # ============================================================
    if doccano_enabled and doccano_sync_client:
        cli.console.print("\n[bold cyan]Doccano Push Scope[/bold cyan]")
        if _effective_rows:
            cli.console.print(
                f"[dim]You will annotate [bold]{_effective_rows:,}[/bold] rows.  "
                f"Choose how many to push to Doccano.[/dim]\n"
            )
        else:
            cli.console.print("[dim]Choose how many annotated rows to push to Doccano.[/dim]\n")
        cli.console.print("  [cyan]all[/cyan]      — Push every annotated row to Doccano")
        cli.console.print("  [cyan]sample[/cyan]   — Push a representative subset (95 % CI, 5 % margin)")
        cli.console.print("  [cyan]limit[/cyan]    — Push a specific number of rows")

        doccano_push_scope = Prompt.ask(
            "\n[bold yellow]Push scope[/bold yellow]",
            choices=["all", "sample", "limit"],
            default="all",
        )

        if doccano_push_scope == "sample" and _effective_rows and _effective_rows > 30:
            import math
            z, p, e = 1.96, 0.5, 0.05
            n_inf = (z ** 2 * p * (1 - p)) / (e ** 2)
            n_adj = n_inf / (1 + ((n_inf - 1) / _effective_rows))
            doccano_push_limit = int(math.ceil(n_adj))
            doccano_sync_client.set_push_sample(_effective_rows, doccano_push_limit)
            cli.console.print(
                f"\n[green]Doccano will receive {doccano_push_limit:,} / {_effective_rows:,} rows "
                f"(random representative sample — 95 % CI, 5 % margin)[/green]"
            )
        elif doccano_push_scope == "limit":
            max_push = _effective_rows if _effective_rows else 1000000
            doccano_push_limit = cli._int_prompt_with_validation(
                "How many rows to push to Doccano?",
                default=min(100, max_push),
                min_value=1,
                max_value=max_push,
            )
            if _effective_rows:
                doccano_sync_client.set_push_sample(_effective_rows, doccano_push_limit)
            else:
                doccano_sync_client.push_limit = doccano_push_limit
            cli.console.print(
                f"\n[green]Doccano will receive up to {doccano_push_limit:,} rows "
                f"(randomly selected)[/green]"
            )
        else:
            cli.console.print("\n[green]All annotated rows will be pushed to Doccano[/green]")

    # ============================================================
    # PARALLEL PROCESSING
    # ============================================================
    if openai_batch_mode:
        cli.console.print("\n[bold cyan]Processing[/bold cyan]")
        cli.console.print(
            "[dim]OpenAI Batch mode manages concurrency, retries, and persistence. "
            "Local parallelism and incremental save settings are skipped.[/dim]\n"
        )
        num_processes = 1
        save_incrementally = False
        batch_size = 1
    else:
        cli.console.print("\n[bold cyan]Parallel Processing[/bold cyan]")
        cli.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

        cli.console.print("[yellow]Parallel Workers:[/yellow]")
        cli.console.print("  Number of simultaneous annotation processes")
        cli.console.print("\n  [red]IMPORTANT:[/red]")
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
        cli.console.print("\n[bold cyan]Incremental Save[/bold cyan]")
        cli.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

        cli.console.print("[yellow]Enable incremental save?[/yellow]")
        cli.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
        cli.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
        cli.console.print("  • [red]No[/red]  - Save only at the end")
        cli.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

        save_incrementally = Confirm.ask("\nEnable incremental save?", default=True)

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
    cli.console.print("\n[bold cyan]Model Parameters[/bold cyan]")
    cli.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

    # Check if model supports parameter tuning
    model_name_lower = model_name.lower()
    is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
    is_gpt5_series = provider == 'openai' and model_name_lower.startswith('gpt-5')
    supports_params = not (is_o_series or is_gpt5_series)

    if not supports_params:
        if is_o_series:
            cli.console.print(f"[yellow][!] Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
            cli.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")
        elif is_gpt5_series:
            cli.console.print(f"[yellow][!] Model '{model_name}' uses locked sampling parameters[/yellow]")
            cli.console.print("[dim]   Temperature and top_p are fixed to 1.0; only max tokens can be adjusted.[/dim]")
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
        cli.console.print("[cyan]Temperature (0.0 - 2.0):[/cyan]")
        cli.console.print("  Controls randomness in responses")
        cli.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
        cli.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
        cli.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
        cli.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
        cli.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
        cli.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
        temperature = FloatPrompt.ask("Temperature", default=0.7)

        # Max tokens
        cli.console.print("\n[cyan]Max Tokens:[/cyan]")
        cli.console.print("  Maximum length of the response")
        cli.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
        cli.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
        cli.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
        cli.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

        # Top_p (nucleus sampling)
        cli.console.print("\n[cyan]Top P (0.0 - 1.0):[/cyan]")
        cli.console.print("  Nucleus sampling - alternative to temperature")
        cli.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
        cli.console.print("           [dim]More deterministic, safer outputs[/dim]")
        cli.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
        cli.console.print("           [dim]More creative, diverse outputs[/dim]")
        cli.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
        top_p = FloatPrompt.ask("Top P", default=1.0)

        # Top_k (only for some models)
        if provider in ['ollama', 'google']:
            cli.console.print("\n[cyan]Top K:[/cyan]")
            cli.console.print("  Limits vocabulary to K most likely next tokens")
            cli.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
            cli.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
            cli.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
            top_k = cli._int_prompt_with_validation("Top K", 40, 1, 100)

    if is_gpt5_series:
        cli.console.print("\n[cyan]Max Tokens (GPT-5 Series):[/cyan]")
        cli.console.print("  Maximum response length; temperature/top_p remain fixed at 1.0.")
        cli.console.print("  [dim]Note: higher values increase API usage.[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", max_tokens, 50, 8000)
        temperature = 1.0
        top_p = 1.0
        top_k = None

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

    cost_estimate: Optional[CostEstimate] = None
    cost_estimate_error: Optional[Exception] = None
    cost_pricing: Optional[ModelPricing] = None
    cost_estimate, cost_estimate_error, cost_pricing = _compute_cost_estimate_for_selection(
        provider=provider,
        model_name=model_name,
        data_path=data_path,
        data_format=data_format,
        text_columns=[text_column],
        prompt_configs=prompt_configs,
        annotation_limit=annotation_limit,
        sample_strategy=sample_strategy,
        max_tokens=max_tokens,
        prompt_header=DEFAULT_PROMPT_HEADER,
    )

    # Display comprehensive summary
    summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True, expand=True)
    summary_table.add_column("Category", style="bold cyan", no_wrap=True)
    summary_table.add_column("Setting", style="yellow", no_wrap=True)
    summary_table.add_column("Value", style="white", ratio=1, overflow="fold")

    # Data section
    summary_table.add_row("Data", "Dataset", str(data_path.name))
    summary_table.add_row("", "Format", data_format.upper())
    summary_table.add_row("", "Text Column", text_column)
    if total_rows:
        summary_table.add_row("", "Total Rows", f"{total_rows:,}")
    if annotation_limit:
        summary_table.add_row("", "Rows to Annotate", f"{annotation_limit:,} ({sample_strategy})")
    else:
        summary_table.add_row("", "Rows to Annotate", "ALL")

    # Model section
    summary_table.add_row("Model", "Provider/Model", f"{provider}/{model_name}")
    summary_table.add_row("", "Temperature", f"{temperature}")
    summary_table.add_row("", "Max Tokens", f"{max_tokens}")
    if configure_params:
        summary_table.add_row("", "Top P", f"{top_p}")
        if provider in ['ollama', 'google']:
            summary_table.add_row("", "Top K", f"{top_k}")

    # Prompts section
    summary_table.add_row("Prompts", "Count", f"{len(prompt_configs)}")
    for i, pc in enumerate(prompt_configs, 1):
        prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
        summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

    # Processing section
    if openai_batch_mode:
        summary_table.add_row(
            "Processing",
            "Annotation Mode",
            "OpenAI Batch (OpenAI-managed async job)"
        )
        summary_table.add_row("", "Parallel Workers", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Batch Size", "N/A (managed by OpenAI Batch)")
        summary_table.add_row("", "Incremental Save", "N/A (handled after batch completion)")
    else:
        summary_table.add_row("Processing", "Annotation Mode", annotation_mode_display)
        summary_table.add_row("", "Parallel Workers", str(num_processes))
        summary_table.add_row("", "Batch Size", str(batch_size))
        summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

    cli.console.print("\n")
    cli.console.print(summary_table)

    _display_cost_estimate_panel(
        cli.console,
        cost_estimate=cost_estimate,
        cost_pricing=cost_pricing,
        cost_estimate_error=cost_estimate_error,
        provider=provider,
        model_name=model_name,
    )

    if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
        return

    # ============================================================
    # REPRODUCIBILITY METADATA
    # ============================================================
    cli.console.print("\n[bold cyan]Reproducibility & Metadata[/bold cyan]")
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
    validation_lab_session_dirs: Optional[Dict[str, Path]] = None
    validation_lab_exports: Dict[str, str] = {}
    validation_lab_enabled = False
    factory_dataset_label = (
        (factory_metadata_state.get('data_source') or {}).get('file_name')
        or (factory_metadata_state.get('data_source') or {}).get('dataset_name')
        or (factory_metadata_state.get('data_source') or {}).get('display_name')
    )
    validation_lab_pref = {
        'enabled': False,
        'mode': AnnotationMode.FACTORY.value,
        'session_dir': None,
        'exports': {},
        'dataset': factory_dataset_label,
    }

    validation_hint = f"validation/{session_id}/doccano"

    # Initialize export flags
    export_to_doccano = False
    export_to_labelstudio = False
    export_sample_size = None
    labelstudio_direct_export = False
    labelstudio_api_url = None
    labelstudio_api_key = None
    prediction_mode = "with"

    if doccano_enabled and doccano_sync_client:
        # Live sync already active — skip the export question
        cli.console.print(
            "\n[bold cyan]Validation Lab Preparation[/bold cyan]  "
            "[green]Doccano live sync is active — annotations are pushed in real-time.[/green]"
        )
        export_confirm = False
    else:
        cli.console.print("\n[bold cyan]Validation Lab Preparation[/bold cyan]")
        cli.console.print(
            f"[dim]Export annotations, review them manually, then copy the validated JSONL into [white]{validation_hint}[/white] so Validation Lab (Mode 5) can compute agreement metrics.[/dim]\n"
        )

        cli.console.print("[yellow]Available validation tools:[/yellow]")
        cli.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
        cli.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
        cli.console.print("  • Both are open-source and free\n")

        cli.console.print("[green]Workflow with Validation Lab:[/green]")
        cli.console.print("  • Export annotations for human review (Doccano works best with Validation Lab)")
        cli.console.print("  • Make corrections manually, then export the validated JSONL")
        cli.console.print(
            f"  • Drop the validated file into [white]{validation_hint}[/white] and reopen Validation Lab (Mode 5)\n"
        )

        export_confirm = Confirm.ask(
            "[bold yellow]Prepare a Doccano export for Validation Lab?[/bold yellow]\n"
            f"[dim]After manual review in Doccano, copy the validated JSONL into [white]{validation_hint}[/white] and reopen Validation Lab (Mode 5) to compute metrics.[/dim]",
            default=False,
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
            validation_lab_enabled = True
            validation_lab_session_dirs = ensure_validation_session_dirs(session_id)
            validation_lab_pref.update({
                'enabled': True,
                'session_dir': str(validation_lab_session_dirs['session_root']),
                'exports': {},
            })
        else:  # labelstudio
            export_to_labelstudio = True
            validation_lab_pref.update({
                'enabled': False,
                'session_dir': None,
                'exports': {},
            })

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
    factory_metadata_state['export_preferences'] = {
        'export_to_doccano': export_to_doccano,
        'export_to_labelstudio': export_to_labelstudio,
        'export_sample_size': export_sample_size,
        'prediction_mode': prediction_mode if export_confirm else 'with',
        'labelstudio_direct_export': labelstudio_direct_export if export_confirm and export_to_labelstudio else False,
        'labelstudio_api_url': labelstudio_api_url if export_confirm and export_to_labelstudio else None,
        'labelstudio_api_key': labelstudio_api_key if export_confirm and export_to_labelstudio else None,
        'validation_lab': validation_lab_pref,
    }
    _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

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

    cli.console.print(f"\n[bold cyan]Output Location:[/bold cyan]")
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
        'model_display_name': model_name,
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
        'create_annotated_subset': True,  # ensure compact annotated-only export is materialised
    }

    if cost_estimate:
        pipeline_config['estimated_cost_usd'] = cost_estimate.total_cost
        pipeline_config['estimated_input_tokens'] = cost_estimate.input_tokens
        pipeline_config['estimated_output_tokens'] = cost_estimate.output_tokens_estimated
        pipeline_config['pricing_last_updated'] = cost_estimate.pricing_last_updated

    pipeline_config.update({
        'session_dirs': {key: str(value) for key, value in session_dirs.items()},
        'provider_folder': provider_folder,
        'model_folder': model_folder,
        'dataset_name': dataset_name,
        'provider_subdir': str(provider_subdir),
        'model_subdir': str(model_subdir),
        'dataset_subdir': str(dataset_subdir),
        'openai_batch_dir': str(batch_dir),
        'session_id': session_id,
        'context_window_config': context_window_config,  # Context window for surrounding sentences
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

    # Wire Doccano live sync client if enabled
    if doccano_sync_client:
        pipeline_config['doccano_sync_client'] = doccano_sync_client

    # ============================================================
    # SAVE REPRODUCIBILITY METADATA
    # ============================================================
    if save_metadata:
        available_rows = total_rows if total_rows is not None else None
        if annotation_limit:
            requested_rows = annotation_limit
        elif available_rows is not None:
            requested_rows = available_rows
        else:
            requested_rows = 'all'
        if isinstance(requested_rows, int):
            initial_remaining = requested_rows
        elif isinstance(available_rows, int):
            initial_remaining = available_rows
        else:
            initial_remaining = 'all'

        metadata_subdir = session_dirs['metadata'] / provider_folder / model_folder / dataset_name
        metadata_subdir.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
        metadata_path = metadata_subdir / metadata_filename
        if metadata_path not in factory_metadata_targets:
            factory_metadata_targets.append(metadata_path)

        session_section = factory_metadata_state.setdefault('annotation_session', {})
        session_section.update({
            'timestamp': timestamp,
            'tool_version': 'LLMTool v1.0',
            'workflow': 'Annotator Factory - Full Pipeline',
            'status': 'ready',
        })
        data_section = factory_metadata_state.setdefault('data_source', {})
        data_section.update({
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
            'provider_folder': provider_folder,
            'model_folder': model_folder,
        })
        progress_section = factory_metadata_state.setdefault('annotation_progress', {})
        progress_section.update({
            'requested': requested_rows,
            'completed': 0,
            'remaining': initial_remaining,
        })
        model_config_section = factory_metadata_state.setdefault('model_configuration', {})
        model_config_section.update({
            'provider': provider,
            'model_name': model_name,
            'annotation_mode': annotation_mode,
            'openai_batch_mode': openai_batch_mode,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k if provider in ['ollama', 'google'] else None,
        })
        processing_section = factory_metadata_state.setdefault('processing_configuration', {})
        processing_section.update({
            'parallel_workers': None if annotation_mode == 'openai_batch' else num_processes,
            'batch_size': None if annotation_mode == 'openai_batch' else batch_size,
            'incremental_save': False if annotation_mode == 'openai_batch' else save_incrementally,
            'openai_batch_mode': openai_batch_mode,
            'openai_batch_dir': str(batch_dir) if annotation_mode == 'openai_batch' else None,
            'identifier_column': auto_identifier_column,
            'provider_folder': provider_folder,
            'model_folder': model_folder,
            'dataset_name': dataset_name,
        })
        factory_metadata_state.setdefault('output', {}).update({
            'output_path': str(default_output_path),
            'output_format': data_format,
        })
        training_section = factory_metadata_state.setdefault('training_workflow', {})
        training_section.update({
            'enabled': False,
            'training_params_file': None,
            'note': 'Training parameters will be saved separately after annotation completes',
        })

        _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)
        metadata_state = factory_metadata_state

        cli.console.print(f"\n[bold green][OK] Metadata saved for reproducibility[/bold green]")
        cli.console.print(f"[bold cyan]Metadata File:[/bold cyan]")
        cli.console.print(f"   {metadata_path}\n")

    # Execute pipeline with Rich progress
    try:
        cli.console.print("\n[bold green]Starting annotation...[/bold green]\n")
        if metadata_state:
            metadata_state = copy.deepcopy(metadata_state)
            metadata_state.setdefault('annotation_session', {})['status'] = 'running'
            _persist_metadata_bundle(factory_metadata_targets, metadata_state)

        # Create pipeline controller with session_id for organized logging
        from ..pipelines.pipeline_controller import PipelineController
        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=session_id,  # Pass session_id for organized logging
            session_mode=AnnotationMode.FACTORY.value,
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
                if metadata_state:
                    failure_metadata = copy.deepcopy(metadata_state)
                    failure_metadata.setdefault('annotation_session', {})['status'] = 'failed'
                    _persist_metadata_bundle(factory_metadata_targets, failure_metadata)
                cli.console.print(f"\n[bold red][FAIL] Error:[/bold red] {error_msg}")
                cli.console.print("[dim]Press Enter to return to menu...[/dim]")
                input()
                return

        # Get results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))
        if metadata_state:
            metadata_state = copy.deepcopy(metadata_state)
            session_section = metadata_state.setdefault('annotation_session', {})
            session_section['status'] = 'completed'
            session_section.setdefault('completed_at', datetime.now().strftime(ISO_FORMAT))
            metadata_state.setdefault('output', {})['output_path'] = str(output_file)
            total_annotated = annotation_results.get('total_annotated')
            progress_section = metadata_state.setdefault('annotation_progress', {})
            if isinstance(total_annotated, int):
                progress_section['completed'] = total_annotated
                requested_value = progress_section.get('requested')
                if isinstance(requested_value, int):
                    progress_section['remaining'] = max(requested_value - total_annotated, 0)
                elif isinstance(metadata_state.get('data_source', {}).get('available_rows'), int):
                    total_available = metadata_state['data_source']['available_rows']
                    progress_section['remaining'] = max(total_available - total_annotated, 0)
                else:
                    progress_section['remaining'] = 0
            _persist_metadata_bundle(factory_metadata_targets, metadata_state)

        # Display success message
        cli.console.print("\n[bold green][OK] Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        # Display statistics if available
        total_annotated = annotation_results.get('total_annotated', 0)
        status_breakdown = annotation_results.get('status_breakdown')
        if isinstance(status_breakdown, dict):
            total_prompt_events = sum(
                int(value)
                for value in status_breakdown.values()
                if isinstance(value, (int, float))
            )
        else:
            status_breakdown = {}
            total_prompt_events = 0

        prompt_success_count = annotation_results.get('prompt_success_count')
        if isinstance(prompt_success_count, (int, float)):
            prompt_success_count = int(prompt_success_count)
        else:
            prompt_success_count = None

        if total_annotated:
            cli.console.print(f"\n[bold cyan]Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if isinstance(success_count, (int, float)) and success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

            if (
                prompt_success_count is not None
                and total_prompt_events
                and (
                    total_prompt_events != total_annotated
                    or prompt_success_count != success_count
                )
            ):
                prompt_success_rate = (prompt_success_count / total_prompt_events) * 100
                cli.console.print(
                    f"   Prompt success rate: {prompt_success_rate:.1f}% "
                    f"({prompt_success_count:,}/{total_prompt_events:,})"
                )

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
            cli.console.print("\n[bold cyan]Sample Annotations:[/bold cyan]")
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
                "prompt_success_count": prompt_success_count,
                "prompt_status_breakdown": status_breakdown,
            },
        )

        # ============================================================
        # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
        # ============================================================
        if not lang_column:
            cli.console.print("\n[bold cyan]Language Detection for Training[/bold cyan]")
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
                    cli.console.print(f"[yellow][!] Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                if len(df_annotated) == 0:
                    cli.console.print("[yellow][!] No annotated rows found. Skipping language detection.[/yellow]")
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
                            cli.console.print(f"\n[bold]Languages Detected ({total:,} texts):[/bold]")

                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold", expand=True)
                            lang_table.add_column("Language", style="cyan", no_wrap=True)
                            lang_table.add_column("Count", style="yellow", justify="right", no_wrap=True)
                            lang_table.add_column("Percentage", style="green", justify="right", no_wrap=True)

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
                            cli.console.print("[yellow][!] No languages detected successfully[/yellow]")

            except Exception as e:
                cli.console.print(f"[yellow][!] Language detection failed: {e}[/yellow]")
                cli.logger.exception("Language detection failed")

        # ============================================================
        # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
        # ============================================================
        training_results: Optional[Dict[str, Any]] = None
        if is_factory_workflow:
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

            # ── Step 2.5: Test Set → Doccano ────────────
            test_doccano_info = None
            if training_results and training_results.get("status") == "completed":
                # Gather labels for project creation
                _step25_labels = []
                for pc in prompt_configs:
                    _step25_labels.extend(pc.get("prompt", {}).get("keys", []))
                _step25_labels = list(dict.fromkeys(_step25_labels))

                if factory_doccano_creds is not None:
                    _url, _token = factory_doccano_creds
                    test_doccano_info = _push_test_set_to_doccano(
                        cli, training_results, _url, _token,
                        session_id, _step25_labels,
                    )
                else:
                    if Confirm.ask(
                        "[bold yellow]Push test set to Doccano for human review?[/bold yellow]",
                        default=False,
                    ):
                        saved_url = cli.settings.get_infer_api_url()
                        saved_username = cli.settings.get_infer_api_username()
                        saved_password = cli.settings.get_infer_api_password()
                        creds = _infer_api_connect(cli, saved_url, saved_username, saved_password)
                        if creds:
                            factory_doccano_creds = creds
                            _url, _token = creds
                            test_doccano_info = _push_test_set_to_doccano(
                                cli, training_results, _url, _token,
                                session_id, _step25_labels,
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

            # ── Step 3.5: Model Predictions → Doccano ───
            if test_doccano_info and training_results and factory_doccano_creds:
                _url, _token = factory_doccano_creds
                _annotate_doccano_with_trained_models(
                    cli, training_results, test_doccano_info,
                    _url, _token,
                )
        else:
            cli.console.print("\n[dim]Training workflow is reserved for Annotator Factory (mode 2). Skipping post-annotation training.[/dim]\n")
            tracker.mark_step(
                8,
                status="skipped",
                detail="Training workflow not run in Annotator mode",
            )
            tracker.mark_step(
                9,
                status="skipped",
                detail="Model annotation deployment skipped (no trained models in Annotator mode)",
            )

        # Export to Doccano JSONL if requested
        existing_doccano_path = annotation_results.get('doccano_export_path')
        existing_validation_path = annotation_results.get('validation_doccano_export_path')

        doccano_export_result = None
        if export_to_doccano:
            validation_destination = None
            if validation_lab_session_dirs:
                validation_destination = validation_lab_session_dirs['doccano']

            can_reuse_existing = (
                existing_doccano_path
                and Path(existing_doccano_path).exists()
                and prediction_mode == "with"
                and (export_sample_size in (None, "all") or export_sample_size == "all")
            )

            if can_reuse_existing:
                cli.console.print(
                    "\n[cyan]Reusing Doccano export generated during annotation.[/cyan]"
                )
                doccano_export_result = {
                    'jsonl_path': str(existing_doccano_path),
                    'validation_copy_path': (
                        str(existing_validation_path)
                        if existing_validation_path and Path(existing_validation_path).exists()
                        else None
                    ),
                    'reused': True,
                }
            else:
                doccano_export_result = cli._export_to_doccano_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    session_dirs=session_dirs,
                    provider_folder=provider_folder,
                    model_folder=model_folder,
                    prediction_mode=prediction_mode,
                    validation_destination=validation_destination,
                    session_id=session_id,
                    annotation_mode=AnnotationMode.FACTORY.value,
                )
            if doccano_export_result:
                doccano_path = doccano_export_result.get('jsonl_path')
                validation_copy_path = doccano_export_result.get('validation_copy_path')
                if doccano_path:
                    target_path = validation_copy_path or doccano_path
                    validation_lab_pref.setdefault('exports', {})['doccano'] = str(target_path)
                    validation_lab_exports['doccano'] = str(target_path)

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

        # Persist updated export preferences with Validation Lab metadata
        factory_metadata_state.setdefault('export_preferences', {})['validation_lab'] = validation_lab_pref
        _persist_metadata_bundle(factory_metadata_targets, factory_metadata_state)

        tracker.mark_step(
            10,
            detail="Factory workflow complete",
            extra={
                "export_doccano": export_to_doccano,
                "export_labelstudio": export_to_labelstudio,
                "prediction_mode": prediction_mode,
                "validation_lab_enabled": validation_lab_enabled,
                "validation_lab_session_dir": str(validation_lab_session_dirs['session_root']) if validation_lab_session_dirs else None,
                "validation_lab_exports": validation_lab_exports,
                "validation_lab_dataset": factory_dataset_label,
            },
        )

        # ============================================================
        # GENERATE ANNOTATION METRICS CHART (Annotator Factory)
        # ============================================================
        try:
            cli.console.print("\n[bold cyan]Generating Annotation Metrics Chart...[/bold cyan]")

            # Determine chart output directory
            chart_output_dir = session_dirs.get('root', Path(output_file).parent) if session_dirs else Path(output_file).parent

            # Get model name from metadata
            model_name_for_chart = model_folder or factory_model_config.get('llm_model') if 'factory_model_config' in dir() else "llm_model"
            if not model_name_for_chart and 'factory_metadata_model_config' in dir():
                model_name_for_chart = factory_metadata_model_config.get('llm_model', 'unknown_model')

            # Load the annotated DataFrame for chart generation
            import pandas as pd
            df_for_chart = pd.read_csv(output_file)

            # Determine annotation column
            annotation_col_for_chart = None
            possible_annotation_cols = ['label', 'labels', 'category', 'annotation', 'predicted_label']
            for col in possible_annotation_cols:
                if col in df_for_chart.columns:
                    annotation_col_for_chart = col
                    break

            # Generate the chart
            chart_path = generate_annotation_chart(
                output_dir=chart_output_dir,
                session_id=session_id,
                model_name=str(model_name_for_chart) if model_name_for_chart else "llm_model",
                df=df_for_chart,
                annotation_column=annotation_col_for_chart,
                annotation_results=annotation_results,
                language_column=lang_column if 'lang_column' in dir() else None,
                workflow_name="Annotator Factory",
            )

            if chart_path:
                cli.console.print(f"[green]✓ Annotation metrics chart saved to:[/green] {chart_path}")
            else:
                cli.console.print("[yellow][!] Could not generate annotation metrics chart[/yellow]")

        except Exception as chart_exc:
            cli.logger.warning(f"Failed to generate annotation metrics chart: {chart_exc}")
            cli.console.print(f"[yellow][!] Chart generation failed: {chart_exc}[/yellow]")

        tracker.update_status(
            "completed",
            extra={
                "validation_lab_enabled": validation_lab_enabled,
                "validation_lab_session_dir": str(validation_lab_session_dirs['session_root']) if validation_lab_session_dirs else None,
                "validation_lab_dataset": factory_dataset_label,
            },
        )

        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    except Exception as exc:
        tracker.update_status("failed", note=str(exc))
        cli.console.print(f"\n[bold red][FAIL] Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Annotation execution failed")
        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

def execute_from_metadata(cli, metadata: dict, action_mode: str, metadata_file: Path, session_dirs: Optional[Dict[str, Path]] = None):
    """Execute annotation based on loaded metadata"""
    import json
    from datetime import datetime
    metadata_state: Optional[Dict[str, Any]] = copy.deepcopy(metadata) if metadata else None
    annotation_session = {}
    if metadata_state:
        annotation_session = metadata_state.get('annotation_session') or {}
    if not annotation_session:
        annotation_session = (metadata or {}).get('annotation_session', {})
    workflow_name = str(annotation_session.get('workflow', '')).lower()
    is_factory_workflow = "factory" in workflow_name

    # Extract all parameters from metadata
    data_source = metadata.get('data_source', {})
    model_config = metadata.get('model_configuration', {})
    prompts = metadata.get('prompts', [])
    proc_config = metadata.get('processing_configuration', {})
    output_config = metadata.get('output', {})
    export_prefs = metadata.get('export_preferences', {})
    session_status = metadata.get('annotation_session', {}).get('status')
    if action_mode == 'resume' and session_status in {'draft', 'ready', 'pending'}:
        action_mode = 'relaunch'

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

    # Resolve session identifier early so downstream logic can rely on it
    session_info = metadata.get('annotation_session') or {}
    session_id = (
        metadata.get('session_id')
        or session_info.get('session_id')
        or session_info.get('session_name')
        or session_info.get('id')
    )
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Persist back into the working metadata snapshot so any derived files include the ID
    metadata['session_id'] = session_id
    session_info.setdefault('session_id', session_id)
    metadata['annotation_session'] = session_info

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
        cli.console.print(f"\n[cyan]Export enabled for: {', '.join(export_tools)} (from saved preferences)[/cyan]")
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
            cli.console.print(f"\n[yellow][!] Output file not found: {original_output}[/yellow]")
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
                        "[yellow][!] Could not determine identifier column from resume file.[/yellow]"
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

                # Resolve identifier column from the resume file
                chosen_identifier = resolved_identifier or default_identifier
                identifier_column = chosen_identifier
                data_source['identifier_column'] = identifier_column
                proc_config['identifier_column'] = identifier_column
                cli.console.print(f"[dim]Identifier column resolved: {identifier_column}[/dim]")

                # Get total available rows from source file
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

    # Prepare output path inside session logs workspace
    staging_dir = Path(session_dirs['annotated_data'])
    staging_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_config.get('model_name', 'unknown').replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if action_mode == 'resume':
        output_filename = original_output.name  # Keep same filename
        default_output_path = original_output
    else:
        output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
        default_output_path = staging_dir / output_filename
        identifier_column = default_identifier
        data_source['identifier_column'] = identifier_column
        proc_config['identifier_column'] = identifier_column

    cli.console.print(f"\n[bold cyan]Output Location:[/bold cyan]")
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
    if not isinstance(num_processes, int) or num_processes < 1:
        # Some older metadata stored None/strings; fall back to safe defaults
        try:
            num_processes = int(num_processes)
        except (TypeError, ValueError):
            num_processes = 1
        if num_processes < 1:
            num_processes = 1

    batch_size = proc_config.get('batch_size', 1)
    if not isinstance(batch_size, int) or batch_size < 1:
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            batch_size = 1
        if batch_size < 1:
            batch_size = 1

    total_rows = data_source.get('total_rows')
    annotation_limit = None if total_rows == 'all' else total_rows
    sample_strategy = data_source.get('sampling_strategy', 'head')

    # IMPORTANT: In resume mode, always use 'head' strategy to continue sequentially
    # This ensures we pick up exactly where we left off, not random new rows
    if action_mode == 'resume':
        sample_strategy = 'head'
        cli.console.print(f"\n[cyan]Resume mode: Using sequential (head) strategy to continue where you left off[/cyan]")

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
        'model_display_name': model_name,
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
        'session_id': session_id,
        'context_window_config': proc_config.get('context_window_config'),  # Restore from metadata if present
    }

    # Add resume information if resuming
    # Always set resume flags when action_mode is 'resume', regardless of
    # whether the metadata file already contained 'resume_mode'.  The flag
    # is set earlier in this function (line ~5678) when the user confirms
    # the resume, but older metadata files from non-resume sessions may not
    # have it, causing skip logic to be silently bypassed.
    if action_mode == 'resume':
        pipeline_config['resume_mode'] = True
        pipeline_config['resume_from_file'] = metadata.get('resume_from_file', str(data_path))
        pipeline_config['skip_annotated'] = True
        pipeline_config['resume'] = True

        # Load already annotated IDs to skip them
        resume_file_path = metadata.get('resume_from_file', str(data_path))
        try:
            import pandas as pd
            resume_file = Path(resume_file_path)
            if resume_file.exists():
                if data_format == 'csv':
                    df_resume = pd.read_csv(resume_file)
                elif data_format in ['excel', 'xlsx']:
                    df_resume = pd.read_excel(resume_file)
                elif data_format == 'parquet':
                    df_resume = pd.read_parquet(resume_file)
                else:
                    df_resume = None

                if df_resume is not None:
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
                        # Use (id, text) pairs instead of just IDs to handle non-unique identifiers
                        text_col = data_source.get('text_column', 'text')
                        if text_col in df_resume.columns:
                            already_annotated_pairs = list(zip(
                                df_resume.loc[annotated_mask, id_column].tolist(),
                                df_resume.loc[annotated_mask, text_col].tolist()
                            ))
                            pipeline_config['skip_annotation_pairs'] = already_annotated_pairs
                            pipeline_config['skip_annotation_text_column'] = text_col
                        else:
                            # Fallback to ID-only if text column not available
                            pipeline_config['skip_annotation_ids'] = df_resume.loc[annotated_mask, id_column].tolist()
                        annotated_count_from_mask = annotated_mask.sum()
                        pipeline_config['already_annotated_count'] = int(annotated_count_from_mask)
                        pipeline_config['total_rows_in_file'] = len(df_resume)

                        cli.console.print(f"\n[bold cyan]Resume Summary:[/bold cyan]")
                        cli.console.print(f"   Total rows in file: {len(df_resume):,}")
                        cli.console.print(f"   Already annotated:  {annotated_count_from_mask:,}")
                        cli.console.print(f"   Remaining to do:    {len(df_resume) - annotated_count_from_mask:,}")
                        cli.console.print(f"   Will skip {annotated_count_from_mask:,} already annotated row(s)\n")
                    elif 'annotation' not in df_resume.columns:
                        cli.console.print(f"[yellow][!] No 'annotation' column found in resume file — treating all rows as unannotated[/yellow]")
                    elif not id_column:
                        cli.console.print(f"[yellow][!] Could not find identifier column in resume file — skip-by-ID disabled[/yellow]")
                        cli.console.print(f"[cyan]   Resume will still use annotation column (NaN check) to skip completed rows[/cyan]")
            else:
                cli.console.print(f"[yellow][!] Resume file not found: {resume_file} — proceeding without skip list[/yellow]")
        except Exception as e:
            cli.logger.warning(f"Could not load annotated IDs from resume file: {e}")
            cli.console.print(f"[yellow][!] Warning: Could not load annotated IDs - will rely on annotation column NaN check[/yellow]")

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
        base_metadata_payload = metadata_state if metadata_state else metadata
        new_metadata = copy.deepcopy(base_metadata_payload)
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

        cli.console.print(f"\n[green][OK] New session metadata saved[/green]")
        cli.console.print(f"[cyan]Metadata File:[/cyan]")
        cli.console.print(f"   {new_metadata_path}\n")

    if action_mode == 'resume' and metadata_root is not None:
        base_metadata_payload = metadata_state if metadata_state else metadata
        resume_metadata = copy.deepcopy(base_metadata_payload)
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

        cli.console.print(f"\n[green][OK] Resume metadata saved[/green]")
        cli.console.print(f"[cyan]Metadata File:[/cyan]")
        cli.console.print(f"   {resume_metadata_path}\n")

    # ── Doccano Integration ──
    doccano_sync_client = None
    doccano_rewrite_ids = None
    _text_col = data_source.get('text_column', 'text')

    cli.console.print("\n[bold cyan]Doccano Integration[/bold cyan]")
    cli.console.print(
        "[dim]Doccano is an open-source annotation platform.  When enabled, every\n"
        "annotation produced by the LLM is pushed [bold]in real-time[/bold] to a Doccano\n"
        "project via your Infer API server.  This lets you:[/dim]\n"
        "\n"
        "  [cyan]1.[/cyan] [white]Review & correct[/white]  — human annotators can validate LLM outputs directly in Doccano\n"
        "  [cyan]2.[/cyan] [white]Track progress[/white]    — watch annotations appear live as the pipeline runs\n"
        "  [cyan]3.[/cyan] [white]Compare sources[/white]   — rewrite mode updates existing annotations for A/B comparisons\n"
    )

    if Confirm.ask("[bold yellow]Enable Doccano integration?[/bold yellow]", default=False):
        saved_url = cli.settings.get_infer_api_url()
        saved_username = cli.settings.get_infer_api_username()
        saved_password = cli.settings.get_infer_api_password()

        creds = _infer_api_connect(cli, saved_url, saved_username, saved_password)
        if creds is not None:
            infer_api_url, token = creds

            _labels = []
            for pp in prompts_payload:
                _labels.extend(pp.get('expected_keys', []))
            _labels = list(dict.fromkeys(_labels))

            cli.console.print("\n[bold]Doccano mode:[/bold]")
            cli.console.print("  [magenta]rewrite[/magenta] — Rewrite LLM annotations only (match first, annotate matched rows)")
            cli.console.print("  [blue]sync[/blue]    — Live sync (create / overwrite / append / rewrite / delete)")
            doccano_mode = Prompt.ask(
                "[bold yellow]Choose mode[/bold yellow]",
                choices=["rewrite", "sync"],
                default="rewrite",
            )

            if doccano_mode == "rewrite":
                _data_fmt = data_source.get('data_format', 'csv')
                result = _doccano_rewrite_filter(
                    cli, infer_api_url, token,
                    data_path, _data_fmt, _text_col, identifier_column, _labels,
                )
                if result is not None:
                    doccano_sync_client, doccano_rewrite_ids = result
                    cli.console.print(
                        f"\n[green]Annotation scope restricted to "
                        f"{len(doccano_rewrite_ids)} matched rows[/green]"
                    )
                else:
                    cli.console.print("[yellow]Continuing without Doccano.[/yellow]")
            else:
                _ds_name = data_path.stem
                _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                default_project_name = f"{_ds_name}_{safe_model_name}_{_ts}"

                doccano_sync_client = _doccano_sync_setup(
                    cli, infer_api_url, token, default_project_name, _labels,
                    data_path=data_path,
                    text_column=_text_col,
                    identifier_column=identifier_column,
                )
        else:
            cli.console.print("[yellow]Continuing without Doccano sync.[/yellow]")

    if doccano_sync_client:
        pipeline_config['doccano_sync_client'] = doccano_sync_client
    if doccano_rewrite_ids is not None:
        pipeline_config['doccano_rewrite_ids'] = doccano_rewrite_ids

    # Execute pipeline
    try:
        cli.console.print("\n[bold green]Starting annotation...[/bold green]\n")
        cli.console.print(f"[dim]Identifier column in use: {identifier_column}[/dim]")

        from ..pipelines.pipeline_controller import PipelineController

        # Extract session_id from metadata or create new one
        resume_session_id = metadata.get("session_id") if metadata else None

        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=resume_session_id,  # Pass session_id for organized logging
            session_mode=(
                AnnotationMode.FACTORY.value
                if is_factory_workflow
                else AnnotationMode.ANNOTATOR.value
            ),
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
                cli.console.print(f"\n[bold red][FAIL] Error:[/bold red] {error_msg}")
                return False

        # Display results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))
        provider_folder_name = str(model_config.get('provider', provider or 'model_provider')).replace("/", "_")
        output_paths = _persist_annotation_outputs(
            cli,
            source_output_path=output_file,
            session_dirs=session_dirs or {},
            provider_folder=provider_folder_name,
            model_folder=safe_model_name,
            dataset_name=data_path.stem,
            annotation_results=annotation_results,
        )
        if output_paths.get("full_path"):
            output_file = output_paths["full_path"]
            annotation_results['output_file'] = output_file
        annotations_only_path = output_paths.get("annotations_only_path")

        cli.console.print("\n[bold green][OK] Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")
        if annotations_only_path:
            cli.console.print(f"   [dim]Annotations-only CSV:[/dim] {annotations_only_path}")

        # Update annotation_progress in metadata after completion
        try:
            import pandas as _pd_progress
        except ImportError:
            _pd_progress = None
        if _pd_progress is not None:
            try:
                _out_p = Path(output_file)
                if _out_p.exists() and _out_p.suffix.lower() == '.csv':
                    _df_done = _pd_progress.read_csv(_out_p)
                elif _out_p.exists() and _out_p.suffix.lower() in ('.xlsx', '.xls'):
                    _df_done = _pd_progress.read_excel(_out_p)
                elif _out_p.exists() and _out_p.suffix.lower() == '.parquet':
                    _df_done = _pd_progress.read_parquet(_out_p)
                else:
                    _df_done = None

                if _df_done is not None and 'annotation' in _df_done.columns:
                    _done_mask = (
                        _df_done['annotation'].notna()
                        & (_df_done['annotation'].astype(str).str.strip() != '')
                        & (_df_done['annotation'].astype(str) != 'nan')
                    )
                    _completed = int(_done_mask.sum())
                    _total = len(_df_done)
                    progress_meta = metadata.setdefault('annotation_progress', {})
                    progress_meta['completed'] = _completed
                    progress_meta['remaining'] = max(_total - _completed, 0)
                    progress_meta['available'] = _total

                    cli.console.print(f"\n[bold cyan]Overall Progress:[/bold cyan]")
                    cli.console.print(f"   Completed: {_completed:,} / {_total:,}")
                    cli.console.print(f"   Remaining: {max(_total - _completed, 0):,}")

                    # Persist updated metadata back to the metadata file
                    if metadata_root is not None:
                        _meta_update_path = metadata_root / f"{dataset_stem}_{safe_model_name}_metadata_{timestamp}.json"
                        if not _meta_update_path.exists():
                            # Try to update the original metadata file
                            _meta_update_path = metadata_file
                        try:
                            _meta_to_save = copy.deepcopy(metadata)
                            _meta_to_save.pop('api_key', None)
                            with open(_meta_update_path, 'w', encoding='utf-8') as _mf:
                                json.dump(_meta_to_save, _mf, indent=2, ensure_ascii=False)
                        except Exception:
                            pass
            except Exception as _prog_err:
                cli.logger.debug(f"Could not update annotation_progress: {_prog_err}")

        total_annotated = annotation_results.get('total_annotated', 0)
        status_breakdown = annotation_results.get('status_breakdown')
        if isinstance(status_breakdown, dict):
            total_prompt_events = sum(
                int(value)
                for value in status_breakdown.values()
                if isinstance(value, (int, float))
            )
        else:
            status_breakdown = {}
            total_prompt_events = 0

        prompt_success_count = annotation_results.get('prompt_success_count')
        if isinstance(prompt_success_count, (int, float)):
            prompt_success_count = int(prompt_success_count)
        else:
            prompt_success_count = None

        if total_annotated:
            cli.console.print(f"\n[bold cyan]Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if isinstance(success_count, (int, float)) and success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

            if (
                prompt_success_count is not None
                and total_prompt_events
                and (
                    total_prompt_events != total_annotated
                    or prompt_success_count != success_count
                )
            ):
                prompt_success_rate = (prompt_success_count / total_prompt_events) * 100
                cli.console.print(
                    f"   Prompt success rate: {prompt_success_rate:.1f}% "
                    f"({prompt_success_count:,}/{total_prompt_events:,})"
                )

        # ============================================================
        # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
        # ============================================================
        if is_factory_workflow:
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
                session_dirs=session_dirs,
            )

            _launch_model_annotation_stage(
                cli,
                session_id=session_id,
                session_dirs=session_dirs,
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
        cli.console.print(f"\n[bold red][FAIL] Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Resume/Relaunch annotation failed")
        return False

    return True

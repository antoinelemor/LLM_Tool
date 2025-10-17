from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import csv
import json
import warnings
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from llm_tool.utils.session_summary import SessionSummary, SummaryRecord, collect_summaries_for_mode, write_summary

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is optional
    RICH_AVAILABLE = False
    Prompt = None  # type: ignore[assignment]

try:  # pragma: no cover - heavy dependency optional in tests
    from sklearn.metrics import (
        cohen_kappa_score,
        hamming_loss,
        jaccard_score,
        precision_recall_fscore_support,
    )
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


@dataclass
class ValidationCandidate:
    source: str
    mode: str
    session_id: str
    session_name: str
    dataset: str
    export_label: str
    export_path: str
    export_exists: bool
    validation_session_dir: Optional[str]
    updated_at: str


@dataclass
class AnnotatorAnnotations:
    name: str
    path: Path
    records: List[Dict[str, Any]]
    weight: float = 1.0


@dataclass
class ConsensusBundle:
    method: str
    tie_policy: str
    consensus_path: Path
    consensus_records: List[Dict[str, Any]]
    annotators: List[AnnotatorAnnotations]
    weights: Dict[str, float]
    agreement_rows: List[Dict[str, Any]]
    agreement_summary: Dict[str, Any]
    excluded_keys: List[str]
    manual_resolutions: Dict[str, Dict[str, str]]
    join_key: str


class ValidationLabController:
    """
    Orchestrates Validation Lab sessions, surfacing annotation exports eligible
    for quality review and managing resumable Validation Lab runs.
    """

    def __init__(self, cli: Any) -> None:
        self.cli = cli
        self.console = getattr(cli, "console", None)
        self.logger = getattr(cli, "logger", None)
        self._validation_logs_root = Path("logs") / "validation_lab"
        self._validation_data_root = Path("validation")

    # --------------------------------------------------------------------- #
    # Path helpers
    # --------------------------------------------------------------------- #

    def _is_within_directory(self, path: Path, directory: Path) -> bool:
        try:
            path.resolve().relative_to(directory.resolve())
            return True
        except (ValueError, RuntimeError):
            return False

    def _friendly_export_label(self, original_label: str, path_obj: Path, session_dir: Path) -> str:
        doccano_dir = session_dir / "doccano"
        if self._is_within_directory(path_obj, doccano_dir):
            return f"Model predictions · {path_obj.name}"
        if original_label and original_label.lower().startswith("doccano"):
            return path_obj.name
        return original_label or path_obj.name

    def _clean_label_token(self, label: Any) -> str:
        token = str(label).strip()
        # Strip trailing punctuation that often appears in Doccano exports (e.g. "null,")
        token = token.rstrip(",;: ")
        return token

    def _normalise_label_value(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return self._clean_label_token(value) or None

    def _split_label_dimension_value(self, label: str, known_dimensions: Set[str]) -> Tuple[Optional[str], Optional[str]]:
        token = self._clean_label_token(label)
        if not token:
            return None, None

        # Prefer known dimensions to avoid ambiguous splits.
        for dimension in sorted(known_dimensions, key=len, reverse=True):
            prefix = f"{dimension}_"
            if token.startswith(prefix):
                value = token[len(prefix) :] or "present"
                return dimension, self._normalise_label_value(value)
            if token == dimension:
                return dimension, "present"

        if ":" in token:
            dimension, value = token.split(":", 1)
            return self._clean_label_token(dimension), self._normalise_label_value(value)

        if "_" in token:
            head, tail = token.split("_", 1)
            return self._clean_label_token(head), self._normalise_label_value(tail or "present")

        return token, "present"

    def _has_disagreement(self, votes: Dict[str, Optional[str]]) -> bool:
        normalised = {self._normalise_label_value(value) for value in votes.values()}
        normalised_without_none = {value for value in normalised if value is not None}
        if not normalised_without_none:
            # Everyone abstained; no disagreement to surface.
            return False
        if len(normalised_without_none) > 1:
            return True
        # At least one annotator lacks a value while others provided one.
        return None in normalised and len(normalised) > 1

    def _is_probable_model_export(self, path: Path) -> bool:
        if path.suffix.lower() != ".jsonl":
            return False
        try:
            relative = path.resolve().relative_to(self._validation_data_root.resolve())
            parts = relative.parts
        except ValueError:
            parts = path.resolve().parts
        if "doccano" not in parts:
            return False
        doccano_idx = parts.index("doccano")
        remaining = parts[doccano_idx + 1 :]
        # Heuristic: model predictions live under doccano/<provider>/<model>/.../<file>.jsonl
        return len(remaining) >= 3

    # --------------------------------------------------------------------- #
    # Public entrypoint
    # --------------------------------------------------------------------- #

    def run(self, mode: str = "auto", preferred_session_id: Optional[str] = None) -> None:
        if mode not in {"auto", "new", "resume"}:
            mode = "auto"

        session_context: Optional[Tuple[str, Dict[str, Path], Path, SessionSummary]] = None

        if mode == "new":
            default_session_id = self._generate_default_session_id()
            session_id = self._prompt_new_session_name(default_session_id)
            session_context = self._start_new_session(session_id)
        elif mode == "resume":
            records = collect_summaries_for_mode(self._validation_logs_root, "validation_lab", limit=50)
            target = None
            if preferred_session_id:
                for record in records:
                    if record.summary.session_id == preferred_session_id:
                        target = record
                        break
            if target is None:
                warning = (
                    "\n[yellow]No matching Validation Lab session found to resume.[/yellow]\n"
                    "Select the session you want to resume from the interactive menu."
                )
                if RICH_AVAILABLE and self.console:
                    self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
                else:
                    print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
                session_context = self._select_session()
            else:
                session_context = self._resume_existing_session(target)
        else:
            session_context = self._select_session()

        if not session_context:
            return

        session_id, session_dirs, resume_path, summary = session_context
        validation_workspace = self._ensure_validation_workspace(session_id)
        self._record_session_resume(resume_path, summary, session_dirs, validation_workspace)
        self._print_workspace_instructions(session_id, validation_workspace)
        candidates = self._display_candidates(session_id)
        if not candidates:
            return
        self._metrics_workflow(
            session_id=session_id,
            session_dirs=session_dirs,
            validation_workspace=validation_workspace,
            summary=summary,
            resume_path=resume_path,
            candidates=candidates,
        )

    # --------------------------------------------------------------------- #
    # Session management
    # --------------------------------------------------------------------- #

    def _create_session_directories(self, session_id: str) -> Dict[str, Path]:
        base_dir = self._validation_logs_root / session_id
        dirs = {
            "base": base_dir,
            "metadata": base_dir / "metadata",
            "reports": base_dir / "reports",
        }
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        return dirs

    def _generate_default_session_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"validation_lab_{timestamp}"

    def _start_new_session(self, session_id: str) -> Tuple[str, Dict[str, Path], Path, SessionSummary]:
        session_dirs = self._create_session_directories(session_id)
        resume_path = session_dirs["base"] / "resume.json"
        summary = SessionSummary(
            mode="validation_lab",
            session_id=session_id,
            session_name=session_id,
            status="active",
        )
        write_summary(resume_path, summary)
        return session_id, session_dirs, resume_path, summary

    def _resume_existing_session(self, record: SummaryRecord) -> Tuple[str, Dict[str, Path], Path, SessionSummary]:
        session_dirs = {
            "base": record.directory,
            "metadata": record.directory / "metadata",
            "reports": record.directory / "reports",
        }
        for directory in session_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        return (
            record.summary.session_id,
            session_dirs,
            record.resume_path,
            record.summary,
        )

    def _select_session(self) -> Optional[Tuple[str, Dict[str, Path], Path, SessionSummary]]:
        records = collect_summaries_for_mode(self._validation_logs_root, "validation_lab", limit=25)
        default_session_id = self._generate_default_session_id()

        if not records:
            return self._start_new_session(default_session_id)

        selection = self._prompt_session_choice(records, default_session_id)
        if selection == "new":
            session_id = self._prompt_new_session_name(default_session_id)
            return self._start_new_session(session_id)

        record_index = selection if isinstance(selection, int) else None
        if record_index is None or record_index < 0 or record_index >= len(records):
            return None
        return self._resume_existing_session(records[record_index])

    def _prompt_session_choice(self, records: List[SummaryRecord], default_session_id: str) -> Optional[str]:
        options = {str(index + 1): index for index in range(len(records))}
        if RICH_AVAILABLE and self.console:
            table = Table(title="📂 Validation Lab Sessions", border_style="cyan")
            table.add_column("#", justify="right", style="cyan", width=4)
            table.add_column("Session", style="white")
            table.add_column("Status", style="green", width=10)
            table.add_column("Updated", style="yellow", width=20)
            for idx, record in enumerate(records, 1):
                summary = record.summary
                updated_display = summary.updated_at.replace("T", " ")
                table.add_row(str(idx), summary.session_id, summary.status, updated_display)
            table.caption = "[dim]Enter number to resume or N to start a new session[/dim]"
            self.console.print()
            self.console.print(table)
            self.console.print()
        else:
            print("\n=== Validation Lab Sessions ===")
            for idx, record in enumerate(records, 1):
                summary = record.summary
                print(f"{idx}. {summary.session_id} [{summary.status}] · {summary.updated_at}")
            print("N. Start a new session")

        choices = list(options.keys()) + ["n", "N"]
        selection = self._ask("Select session", default="1", choices=choices)
        if selection.lower() == "n":
            return "new"
        return options.get(selection)

    def _prompt_new_session_name(self, default_session_id: str) -> str:
        session_id = self._ask("Session name", default=default_session_id)
        session_id = (session_id or default_session_id).strip() or default_session_id
        return session_id.replace(" ", "_")

    def _record_session_resume(
        self,
        resume_path: Path,
        summary: SessionSummary,
        session_dirs: Dict[str, Path],
        validation_workspace: Dict[str, Path],
    ) -> None:
        summary.mode = "validation_lab"
        summary.status = "active"
        summary.extra = summary.extra or {}
        summary.extra.setdefault("workspace", str(session_dirs["base"]))
        summary.extra["validation_data_root"] = str(validation_workspace["root"])
        summary.extra["validation_doccano_dir"] = str(validation_workspace["doccano"])
        summary.bump_updated()
        write_summary(resume_path, summary)

    # --------------------------------------------------------------------- #
    # Candidate harvesting
    # --------------------------------------------------------------------- #

    def _display_candidates(self, session_id: str) -> List[ValidationCandidate]:
        candidates = self._collect_validation_candidates()
        filtered: List[ValidationCandidate] = []
        for candidate in candidates:
            if candidate.source == "Training Arena":
                filtered.append(candidate)
                continue
            path = Path(candidate.export_path)
            if path.suffix.lower() == ".jsonl" and self._is_probable_model_export(path):
                filtered.append(candidate)

        if filtered:
            candidates = filtered

        if not candidates:
            message = (
                "\n[yellow]No validation exports detected yet.[/yellow]\n"
                "Annotate data in Annotator or Factory, confirm the Validation Lab export, "
                "or generate validation splits in Training Arena to populate this table."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(message), border_style="yellow"))
            else:
                print(message.replace("[yellow]", "").replace("[/yellow]", ""))
            return []

        if RICH_AVAILABLE and self.console:
            table = Table(title=f"📊 Validation Candidates (session: {session_id})", border_style="green")
            table.add_column("#", justify="right", style="cyan", width=4)
            table.add_column("Source", style="cyan")
            table.add_column("Session", style="white")
            table.add_column("Dataset", style="magenta")
            table.add_column("Export", style="white")
            table.add_column("Location", style="dim")
            table.add_column("Updated", style="yellow")
            for idx, item in enumerate(candidates, 1):
                location = item.export_path
                if not item.export_exists:
                    location = f"[red]{location}[/red]"
                table.add_row(
                    str(idx),
                    item.source,
                    item.session_name or item.session_id,
                    item.dataset or "-",
                    item.export_label,
                    location,
                    item.updated_at.replace("T", " "),
                )
            self.console.print()
            self.console.print(table)
            self.console.print()
        else:
            print(f"\n=== Validation Candidates (session: {session_id}) ===")
            for idx, item in enumerate(candidates, 1):
                status = "✅" if item.export_exists else "⚠️ missing"
                print(
                    f"{idx}. {item.source}: {item.session_name or item.session_id} · "
                    f"{item.dataset or '-'} · {item.export_label} -> {item.export_path} ({status})"
                )
            print()

        return candidates

    def _ensure_validation_workspace(self, session_id: str) -> Dict[str, Path]:
        root = (self._validation_data_root / session_id).resolve()
        doccano_dir = root / "doccano"
        doccano_dir.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        return {"root": root, "doccano": doccano_dir}

    def _print_workspace_instructions(self, session_id: str, workspace: Dict[str, Path]) -> None:
        doccano_hint = workspace["doccano"]
        session_exports_dir = (self._validation_data_root / session_id / "doccano").resolve()
        message = (
            "\n[cyan]Validation workspace ready.[/cyan]\n"
            f"• Place the reviewer JSONL files you want to compare inside [white]{doccano_hint}[/white].\n"
            f"• Each annotation session also keeps its exports under [white]{session_exports_dir}[/white].\n"
            "  Copy any annotator JSONL files from there (or another session folder) into the workspace doccano directory above.\n"
            "• Validation Lab will read the model predictions from the export you select, then prompt you to pick the reviewer files.\n"
            "• Training Arena validation splits can also be copied here when you want to benchmark fine-tuned models.\n"
        )
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(Text.from_markup(message), border_style="cyan"))
        else:
            print(
                message.replace("[cyan]", "").replace("[/cyan]", "").replace("[white]", "").replace("[/white]", "")
            )

    def _metrics_workflow(
        self,
        session_id: str,
        session_dirs: Dict[str, Path],
        validation_workspace: Dict[str, Path],
        summary: SessionSummary,
        resume_path: Path,
        candidates: List[ValidationCandidate],
    ) -> None:
        if not SKLEARN_AVAILABLE:
            message = (
                "\n[yellow]scikit-learn is required to compute validation metrics.[/yellow]\n"
                "Install it with `pip install scikit-learn` then rerun Validation Lab."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(message), border_style="yellow"))
            else:
                print(message.replace("[yellow]", "").replace("[/yellow]", ""))
            return

        selection_msg = (
            "\n[cyan]Step 1: Choose the model export to evaluate.[/cyan]\n"
            "Pick the JSONL file that contains the model predictions for this session.\n"
            "You will be able to select reviewer/annotator files in the next step."
        )
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(Text.from_markup(selection_msg), border_style="cyan"))
        else:
            print(selection_msg.replace("[cyan]", "").replace("[/cyan]", ""))

        candidate: Optional[ValidationCandidate] = None
        model_path: Optional[Path] = None
        while True:
            candidate = self._choose_candidate_for_metrics(candidates)
            if not candidate:
                return

            model_path = Path(candidate.export_path)
            if model_path.suffix.lower() == ".jsonl":
                break

            warning = (
                "\n[yellow]Metrics computation currently supports Doccano JSONL exports only.[/yellow]\n"
                f"[dim]Selected export:[/dim] {model_path}\n"
                "Please choose a JSONL export created by the annotation workflow."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))

        assert model_path is not None  # for type checkers

        if not model_path.exists():
            warning = (
                "\n[red]Model export not found on disk.[/red]\n"
                f"[dim]Expected at:[/dim] {model_path}"
            )
            if self.logger:
                self.logger.warning("Validation Lab export missing at %s", model_path)
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="red"))
            else:
                print(warning.replace("[red]", "").replace("[/red]", ""))
            return

        reviewer_msg = (
            "\n[cyan]Step 2: Select the reviewer JSONL files.[/cyan]\n"
            f"Model predictions: [white]{model_path}[/white]\n"
            "Pick one or more reviewer/annotator JSONL exports to compare against the model predictions.\n"
            "Select the reviewers one at a time; after each selection you can add another or choose 'done'.\n"
            "Use 'done' when you have added every reviewer you want to include."
        )
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(Text.from_markup(reviewer_msg), border_style="cyan"))
        else:
            print(
                reviewer_msg.replace("[cyan]", "")
                .replace("[/cyan]", "")
                .replace("[white]", "")
                .replace("[/white]", "")
            )

        consensus_bundle = self._prepare_annotation_bundle(candidate, validation_workspace, model_path)
        if not consensus_bundle:
            return

        self._compute_metrics_for_candidate(
            candidate=candidate,
            model_path=model_path,
            manual_path=consensus_bundle.consensus_path,
            session_dirs=session_dirs,
            summary=summary,
            resume_path=resume_path,
            consensus=consensus_bundle,
        )

    def _choose_candidate_for_metrics(self, candidates: List[ValidationCandidate]) -> Optional[ValidationCandidate]:
        if not candidates:
            return None
        if len(candidates) == 1:
            choice = self._ask("Compute metrics for this export now?", default="y", choices=["y", "n", "Y", "N"])
            if choice.lower() != "y":
                return None
            return candidates[0]

        options = {str(idx): candidate for idx, candidate in enumerate(candidates, 1)}
        choices = list(options.keys()) + ["skip"]
        while True:
            selection = self._ask("Select export number for Validation Lab metrics", default="skip", choices=choices)
            if selection == "skip":
                return None
            chosen = options.get(selection)
            if not chosen:
                continue
            if self._is_probable_model_export(Path(chosen.export_path)):
                return chosen
            warning = (
                "\n[yellow]This export looks like a reviewer file, not model predictions.[/yellow]\n"
                "Pick a JSONL produced by the model (usually located under a doccano/<provider>/<model>/ directory)."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))

    def _prepare_annotation_bundle(
        self,
        candidate: ValidationCandidate,
        workspace: Dict[str, Path],
        model_path: Path,
    ) -> Optional[ConsensusBundle]:
        manual_candidates = self._collect_manual_jsonl_candidates(candidate, model_path, workspace)
        annotators = self._prompt_annotation_sets(candidate, manual_candidates)
        if not annotators:
            return None

        if len(annotators) == 1:
            annotator = annotators[0]
            weights = {annotator.name: annotator.weight}
            agreement_rows: List[Dict[str, Any]] = []
            bundle = ConsensusBundle(
                method="single_annotator",
                tie_policy="not_applicable",
                consensus_path=annotator.path,
                consensus_records=annotator.records,
                annotators=annotators,
                weights=weights,
                agreement_rows=agreement_rows,
                agreement_summary={},
                excluded_keys=[],
                manual_resolutions={},
                join_key="auto",
            )
            return bundle

        if len(annotators) > 1:
            multi_msg = (
                "\n[cyan]Multiple reviewer files detected.[/cyan]\n"
                "Validation Lab will build a consensus reference by asking how you want to combine these annotators.\n"
                "You can choose majority vote, weighted voting, or Dawid–Skene, and optionally resolve ties manually."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(multi_msg), border_style="cyan"))
            else:
                print(multi_msg.replace("[cyan]", "").replace("[/cyan]", ""))

        join_key, overlap = self._determine_join_key_for_annotations(annotators)
        if not join_key:
            warning = (
                "\n[yellow]Could not find a common identifier across the selected annotation files.[/yellow]\n"
                "Ensure the Doccano exports come from the same dataset and share metadata such as `source_id`, `transcript_speaker_id+sentence_id`, or `text`."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
            return None

        preview_samples = self._collect_annotation_preview_examples(annotators, join_key)
        method, weights = self._choose_consensus_method(annotators, preview_samples, overlap)
        if method is None:
            return None

        if method == "weighted_consensus":
            weights = self._prompt_annotator_weights(annotators, weights)
        else:
            weights = {annotator.name: 1.0 for annotator in annotators}

        aggregation = self._aggregate_annotations(
            annotators=annotators,
            join_key=join_key,
            method=method,
            weights=weights,
        )
        if aggregation is None:
            return None

        tie_policy, manual_resolutions, excluded_keys = self._resolve_consensus_ties(
            consensus_map=aggregation["consensus_map"],
            tie_cases=aggregation["tie_cases"],
            join_key=join_key,
            annotators=annotators,
            records_by_key=aggregation["records_by_key"],
        )
        manual_resolutions = {key: dict(value) for key, value in manual_resolutions.items()}
        excluded_keys = list(dict.fromkeys(excluded_keys))
        consensus_records = self._finalise_consensus_records(
            consensus_map=aggregation["consensus_map"],
            records_by_key=aggregation["records_by_key"],
            excluded_keys=excluded_keys,
            manual_resolutions=manual_resolutions,
            method=method,
            tie_policy=tie_policy,
            weights=weights,
        )

        if not consensus_records:
            warning = (
                "\n[yellow]No data remained after resolving tie cases.[/yellow]\n"
                "Unable to build a consensus reference set."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
            return None

        consensus_path = self._write_consensus_file(
            candidate=candidate,
            workspace=workspace,
            method=method,
            records=consensus_records,
        )

        agreement_rows, agreement_summary = self._compute_agreement_metrics(
            annotators=annotators,
            records_by_key=aggregation["records_by_key"],
            excluded_keys=excluded_keys,
        )

        bundle = ConsensusBundle(
            method=method,
            tie_policy=tie_policy,
            consensus_path=consensus_path,
            consensus_records=consensus_records,
            annotators=annotators,
            weights=weights,
            agreement_rows=agreement_rows,
            agreement_summary=agreement_summary,
            excluded_keys=excluded_keys,
            manual_resolutions=manual_resolutions,
            join_key=join_key,
        )
        return bundle

    def _prompt_annotation_sets(
        self,
        candidate: ValidationCandidate,
        manual_candidates: List[Path],
    ) -> List[AnnotatorAnnotations]:
        annotators: List[AnnotatorAnnotations] = []
        remaining = list(manual_candidates)

        if not remaining:
            workspace_doccano = workspace["doccano"]
            session_doccano = None
            if candidate.validation_session_dir:
                try:
                    session_doccano = Path(candidate.validation_session_dir).expanduser().resolve()
                except Exception:
                    session_doccano = None
            if session_doccano is None:
                session_doccano = (self._validation_data_root / candidate.session_id / "doccano").resolve()
            message = (
                "\n[yellow]No reviewer JSONL files detected for this session yet.[/yellow]\n"
                f"Copy the annotator JSONL exports for this model into [white]{workspace_doccano}[/white] or [white]{session_doccano}[/white] before continuing."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(message), border_style="yellow"))
            else:
                print(
                    message.replace("[yellow]", "")
                    .replace("[/yellow]", "")
                    .replace("[white]", "")
                    .replace("[/white]", "")
                )
            return []

        while remaining:
            if RICH_AVAILABLE and self.console:
                table = Table(
                    title=f"Available reviewer exports · {candidate.session_name or candidate.session_id}",
                    border_style="cyan",
                )
                table.add_column("#", justify="right", style="cyan", width=4)
                table.add_column("File", style="white")
                table.add_column("Label preview", style="magenta")
                for idx, path in enumerate(remaining, 1):
                    preview = self._summarise_label_preview(path)
                    table.add_row(str(idx), str(path), preview)
                self.console.print(table)
            else:
                print("\nAvailable reviewer exports:")
                for idx, path in enumerate(remaining, 1):
                    preview = self._summarise_label_preview(path)
                    print(f"  {idx}. {path} · {preview}")

            choices = [str(idx) for idx in range(1, len(remaining) + 1)]
            if annotators:
                choices.append("done")
            selection = self._ask(
                "Select an annotator JSONL (enter one number at a time, or 'done' to finish)",
                default="done" if annotators else "1",
                choices=choices,
            )
            if selection == "done":
                break
            try:
                index = int(selection) - 1
            except ValueError:
                continue
            if index < 0 or index >= len(remaining):
                continue

            path = remaining.pop(index)
            default_name = f"Annotator {len(annotators) + 1}"
            annotator_name = self._ask(
                f"Name this annotator for {path.name}",
                default=default_name,
                choices=None,
            ).strip() or default_name
            if any(existing.name == annotator_name for existing in annotators):
                suffix = 2
                base_name = annotator_name
                while any(existing.name == annotator_name for existing in annotators):
                    annotator_name = f"{base_name}_{suffix}"
                    suffix += 1

            try:
                records = self._load_annotation_file(path)
            except Exception as exc:  # pragma: no cover - defensive
                if self.logger:
                    self.logger.error("Unable to load %s: %s", path, exc)
                continue

            annotators.append(AnnotatorAnnotations(name=annotator_name, path=path, records=records))

        if not annotators:
            return []
        return annotators

    def _summarise_label_preview(self, path: Path, limit: int = 3) -> str:
        labels: Counter = Counter()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for _, raw_line in zip(range(limit), handle):
                    line = raw_line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    extracted = self._extract_labels(data)
                    labels.update(extracted)
        except Exception:
            return "preview unavailable"
        if not labels:
            return "0 label"
        preview = ", ".join(label for label, _ in labels.most_common(limit))
        return f"{preview}"

    def _determine_join_key_for_annotations(
        self,
        annotators: List[AnnotatorAnnotations],
    ) -> Tuple[str, int]:
        if not annotators:
            return "", 0

        priority_keys = [
            "source_id",
            "row_id",
            "meta_id",
            "transcript_speaker_id+sentence_id",
            "transcript_sentence_id",
            "example_id",
            "id",
            "text",
        ]

        key_presence: Dict[str, List[Set[str]]] = defaultdict(list)
        for annotator in annotators:
            for key in priority_keys:
                values: Set[str] = set()
                for record in annotator.records:
                    value = self._extract_join_value(record, key)
                    if value:
                        values.add(value)
                if values:
                    key_presence[key].append(values)

        best_key = ""
        best_overlap = 0
        for key in priority_keys:
            value_sets = key_presence.get(key, [])
            if len(value_sets) < len(annotators):
                continue
            overlap = set.intersection(*value_sets)
            overlap_size = len(overlap)
            if overlap_size > best_overlap:
                best_key = key
                best_overlap = overlap_size

        if not best_key and key_presence.get("text"):
            best_key = "text"
            overlap_sets = key_presence["text"]
            if len(overlap_sets) == len(annotators):
                best_overlap = len(set.intersection(*overlap_sets))

        return best_key, best_overlap

    def _collect_annotation_preview_examples(
        self,
        annotators: List[AnnotatorAnnotations],
        join_key: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        records_by_key, shared_keys = self._build_annotation_index(annotators, join_key)
        examples: List[Dict[str, Any]] = []
        for key in list(shared_keys)[:limit]:
            info = records_by_key[key]
            text = info["base_record"].get("text", "")
            snippet = text if len(text) <= 180 else text[:177] + "…"
            annotator_values = []
            for annotator in annotators:
                values = info["annotator_values"].get(annotator.name, {})
                annotator_values.append({"annotator": annotator.name, "values": values})
            examples.append({"key": key, "text": snippet, "annotators": annotator_values})
        return examples

    def _choose_consensus_method(
        self,
        annotators: List[AnnotatorAnnotations],
        preview_samples: List[Dict[str, Any]],
        overlap: int,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        if not annotators:
            return None, {}

        examples_markup = self._build_consensus_examples(preview_samples, annotators)

        explanation = (
            "\n[cyan]Consensus strategies to build the gold standard[/cyan]\n"
            "1. [bold]Simple majority[/bold] – each annotator counts as 1 vote; the most frequent value wins per dimension.\n"
            "2. [bold]Weighted consensus[/bold] – assign a relative weight to each annotator before tallying votes.\n"
            "3. [bold]Probabilistic (Dawid–Skene)[/bold] – estimate annotator reliability and infer the latent true label probabilistically.\n"
            f"\n[dim]Shared segments detected:[/dim] {overlap:,}"
        )
        if RICH_AVAILABLE and self.console:
            panel = Panel.fit(
                Text.from_markup(explanation + ("\n\n" + examples_markup if examples_markup else "")),
                border_style="cyan",
            )
            self.console.print(panel)
        else:
            print(
                explanation.replace("[cyan]", "")
                .replace("[/cyan]", "")
                .replace("[bold]", "")
                .replace("[/bold]", "")
                .replace("[dim]", "")
                .replace("[/dim]", "")
            )
            if examples_markup:
                print(examples_markup.replace("[bold]", "").replace("[/bold]", ""))

        selection = self._ask(
            "Choose the consensus method (1=Majority, 2=Weighted, 3=Dawid-Skene)",
            default="1",
            choices=["1", "2", "3"],
        )
        default_weights = {annotator.name: 1.0 for annotator in annotators}
        if selection == "1":
            return "majority_vote", default_weights
        if selection == "2":
            return "weighted_consensus", default_weights
        if selection == "3":
            return "dawid_skene", default_weights
        return None, {}

    def _prompt_annotator_weights(
        self,
        annotators: List[AnnotatorAnnotations],
        initial: Dict[str, float],
    ) -> Dict[str, float]:
        weights = dict(initial)
        for annotator in annotators:
            while True:
                raw = self._ask(
                    f"Relative weight for {annotator.name} (>=0)",
                    default=str(weights.get(annotator.name, 1.0)),
                    choices=None,
                ).strip()
                try:
                    value = float(raw)
                    if value < 0:
                        raise ValueError
                    weights[annotator.name] = value
                    annotator.weight = value
                    break
                except ValueError:
                    warning = "[yellow]Invalid value. Please enter a non-negative number.[/yellow]"
                    if RICH_AVAILABLE and self.console:
                        self.console.print(Text.from_markup(warning))
                    else:
                        print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
        if all(weight == 0 for weight in weights.values()):
            for annotator in annotators:
                weights[annotator.name] = 1.0
        return weights

    def _build_consensus_examples(
        self,
        preview_samples: List[Dict[str, Any]],
        annotators: List[AnnotatorAnnotations],
    ) -> str:
        if not preview_samples:
            return ""

        annotator_names = [annotator.name for annotator in annotators]

        def format_votes(sample: Dict[str, Any], dimension: str) -> str:
            buckets: Dict[str, List[str]] = defaultdict(list)
            for annotator in sample["annotators"]:
                value = annotator["values"].get(dimension)
                if value is None:
                    value = "∅"
                buckets[value].append(annotator["annotator"])
            parts = []
            for value, names in sorted(buckets.items(), key=lambda item: (item[0] or "", item[1])):
                name_list = ", ".join(sorted(names))
                parts.append(f"{value} ({name_list})")
            return "; ".join(parts)

        diff_sample = None
        diff_dimension: Optional[str] = None
        for sample in preview_samples:
            dimensions: Set[str] = set()
            for annotator in sample["annotators"]:
                dimensions.update(annotator["values"].keys())
            for dimension in dimensions:
                values = {ann["values"].get(dimension) for ann in sample["annotators"]}
                if len(values) > 1:
                    diff_sample = sample
                    diff_dimension = dimension
                    break
            if diff_sample:
                break

        primary_sample = diff_sample or preview_samples[0]
        primary_dimension = diff_dimension
        if primary_dimension is None:
            # default to first dimension present
            for annotator in primary_sample["annotators"]:
                if annotator["values"]:
                    primary_dimension = next(iter(annotator["values"].keys()))
                    break

        if not primary_dimension:
            return ""

        votes_description = format_votes(primary_sample, primary_dimension)
        sample_header = f"[bold]{primary_sample['key']}[/bold] · {primary_sample['text']}"

        majority_example = (
            f"• Majority example: {sample_header}\n"
            f"  {primary_dimension}: {votes_description}. "
            "Simple majority picks the value with the most reviewers."
        )

        lead_annotator = annotator_names[0] if annotator_names else "the lead reviewer"
        weighted_example = (
            f"• Weighted consensus: assign larger weights to trusted reviewers (e.g. give {lead_annotator} 2.0 and the others 1.0). "
            "When reviewers differ, the higher weight tilts the vote toward the reviewer you trust most."
        )

        dawid_example = (
            f"• Dawid–Skene: automatically estimates each annotator's reliability for dimensions like '{primary_dimension}'. "
            "Useful when reviewers disagree frequently or have different expertise; the algorithm learns who is usually right and infers the latent true label."
        )

        if diff_sample is None:
            majority_example = (
                f"• Majority example: {sample_header}\n"
                f"  All reviewers currently agree ({votes_description}). "
                "Majority vote would keep this value when they stay aligned."
            )
            weighted_example += " Even when everyone agrees, you can set weights now for future datasets where disagreements appear."
            dawid_example += " It will behave like majority vote when reviewers agree, and re-weight them automatically once differences show up."

        return "\n".join([majority_example, weighted_example, dawid_example])

    def _aggregate_annotations(
        self,
        annotators: List[AnnotatorAnnotations],
        join_key: str,
        method: str,
        weights: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        records_by_key, shared_keys = self._build_annotation_index(annotators, join_key)
        if not shared_keys:
            warning = (
                "\n[yellow]The selected annotators do not share any common segment identifiers.[/yellow]\n"
                "Ensure the reviewer JSONL files come from the same export or share a common join key."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
            return None

        if method == "dawid_skene":
            return self._aggregate_dawid_skene(annotators, records_by_key, shared_keys)
        return self._aggregate_majority_or_weighted(annotators, records_by_key, shared_keys, method, weights)

    def _aggregate_majority_or_weighted(
        self,
        annotators: List[AnnotatorAnnotations],
        records_by_key: Dict[str, Dict[str, Any]],
        shared_keys: Set[str],
        method: str,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        consensus_map: Dict[str, Dict[str, str]] = {}
        tie_cases: List[Dict[str, Any]] = []

        for key in shared_keys:
            info = records_by_key[key]
            consensus: Dict[str, str] = {}
            dims = set()
            for annotator in annotators:
                dims.update(info["annotator_values"].get(annotator.name, {}).keys())

            for dimension in sorted(dims):
                vote_counter: Dict[str, float] = defaultdict(float)
                for annotator in annotators:
                    value = info["annotator_values"].get(annotator.name, {}).get(dimension)
                    if value is None:
                        continue
                    weight = 1.0 if method == "majority_vote" else weights.get(annotator.name, 1.0)
                    vote_counter[value] += weight

                if not vote_counter:
                    continue

                items = list(vote_counter.items())
                items.sort(key=lambda item: (-item[1], item[0]))
                top_score = items[0][1]
                tied_values = [value for value, score in items if math.isclose(score, top_score, rel_tol=1e-9, abs_tol=1e-12)]

                if len(tied_values) == 1:
                    consensus[dimension] = tied_values[0]
                else:
                    tie_cases.append(
                        {
                            "key": key,
                            "dimension": dimension,
                            "options": dict(vote_counter),
                            "annotator_values": info["annotator_values"],
                            "text": info["base_record"].get("text", ""),
                        }
                    )

            if consensus:
                consensus_map[key] = consensus

        return {
            "consensus_map": consensus_map,
            "tie_cases": tie_cases,
            "records_by_key": records_by_key,
            "shared_keys": shared_keys,
        }

    def _aggregate_dawid_skene(
        self,
        annotators: List[AnnotatorAnnotations],
        records_by_key: Dict[str, Dict[str, Any]],
        shared_keys: Set[str],
    ) -> Dict[str, Any]:
        consensus_map: Dict[str, Dict[str, str]] = {}
        tie_cases: List[Dict[str, Any]] = []

        # Collect dimensions
        dimensions: Set[str] = set()
        for info in records_by_key.values():
            for annotator_values in info["annotator_values"].values():
                dimensions.update(annotator_values.keys())

        for dimension in sorted(dimensions):
            item_values: Dict[str, Dict[str, str]] = {}
            for key in shared_keys:
                info = records_by_key[key]
                per_item: Dict[str, str] = {}
                for annotator in annotators:
                    value = info["annotator_values"].get(annotator.name, {}).get(dimension)
                    if value is not None:
                        per_item[annotator.name] = value
                if per_item:
                    item_values[key] = per_item

            if not item_values:
                continue

            consensus, posterior = self._run_dawid_skene(item_values, annotators)
            for key, value in consensus.items():
                if value is not None:
                    consensus_map.setdefault(key, {})[dimension] = value
                else:
                    probabilities = posterior.get(key, {})
                    tie_cases.append(
                        {
                            "key": key,
                            "dimension": dimension,
                            "options": probabilities,
                            "annotator_values": records_by_key[key]["annotator_values"],
                            "text": records_by_key[key]["base_record"].get("text", ""),
                        }
                    )

        return {
            "consensus_map": consensus_map,
            "tie_cases": tie_cases,
            "records_by_key": records_by_key,
            "shared_keys": shared_keys,
        }

    def _resolve_consensus_ties(
        self,
        consensus_map: Dict[str, Dict[str, str]],
        tie_cases: List[Dict[str, Any]],
        join_key: str,
        annotators: List[AnnotatorAnnotations],
        records_by_key: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Dict[str, str]], List[str]]:
        if not tie_cases:
            return "no_ties", {}, []

        warning = (
            f"\n[yellow]{len(tie_cases)} segment(s) are tied across annotators.[/yellow]\n"
            "Would you like to resolve these ties manually (by choosing the reference label)?\n"
            "If not, the tied segments will be excluded from the consensus set."
        )
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
        else:
            print(warning.replace("[yellow]", "").replace("[/yellow]", ""))

        choice = self._ask("Resolve ties manually?", default="y", choices=["y", "n", "Y", "N"])
        manual_resolutions: Dict[str, Dict[str, str]] = defaultdict(dict)
        excluded_keys: List[str] = []

        if choice.lower() != "y":
            for case in tie_cases:
                key = case["key"]
                if key in consensus_map:
                    consensus_map.pop(key, None)
                excluded_keys.append(key)
            return "excluded", manual_resolutions, excluded_keys

        for case in tie_cases:
            key = case["key"]
            if key in excluded_keys:
                continue
            info = records_by_key.get(key)
            if not info:
                continue

            header = (
                f"\n[cyan]Segment {join_key}: {key}[/cyan]\n"
                f"{info['base_record'].get('text', '')}\n"
                f"[dim]Dimension: {case['dimension']}[/dim]"
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(header), border_style="cyan"))
            else:
                print(header.replace("[cyan]", "").replace("[/cyan]", "").replace("[dim]", "").replace("[/dim]", ""))

            for annotator in annotators:
                value = case["annotator_values"].get(annotator.name, {}).get(case["dimension"], "∅")
                line = f"  • {annotator.name}: {value}"
                if RICH_AVAILABLE and self.console:
                    self.console.print(Text(line))
                else:
                    print(line)

            options = case["options"]
            if isinstance(options, dict):
                option_items = sorted(options.items(), key=lambda item: (-item[1], str(item[0])))
            else:
                option_items = sorted(options.items(), key=lambda item: (-item[1], str(item[0])))

            option_map: Dict[str, str] = {}
            for idx, (value, score) in enumerate(option_items, 1):
                option_label = f"{value} (score {round(score, 3)})"
                option_map[str(idx)] = value
                if RICH_AVAILABLE and self.console:
                    self.console.print(Text(f"  [{idx}] {option_label}"))
                else:
                    print(f"  [{idx}] {option_label}")
            option_map["exclude"] = "__exclude__"
            selection = self._ask("Choose the consensus value or 'exclude'", default="1", choices=list(option_map.keys()))
            selected = option_map.get(selection)
            if selected == "__exclude__":
                excluded_keys.append(key)
                consensus_map.pop(key, None)
            elif selected:
                consensus_map.setdefault(key, {})[case["dimension"]] = selected
                manual_resolutions[key][case["dimension"]] = selected

        return "manual_resolution", manual_resolutions, excluded_keys

    def _finalise_consensus_records(
        self,
        consensus_map: Dict[str, Dict[str, str]],
        records_by_key: Dict[str, Dict[str, Any]],
        excluded_keys: List[str],
        manual_resolutions: Dict[str, Dict[str, str]],
        method: str,
        tie_policy: str,
        weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        excluded_set = set(excluded_keys)
        for key, consensus in consensus_map.items():
            if key in excluded_set:
                continue
            info = records_by_key.get(key)
            if not info:
                continue
            base_record = info["base_record"]
            meta = dict(base_record.get("meta", {}))
            consensus_meta = {
                "method": method,
                "tie_policy": tie_policy,
                "values": consensus,
                "manual_resolutions": manual_resolutions.get(key, {}),
                "annotator_weights": weights,
            }
            meta["consensus"] = consensus_meta

            labels: List[str] = []
            dimension_label_map: Dict[str, Dict[str, str]] = info.get("dimension_label_map", {})
            for dimension, value in consensus.items():
                label_lookup = dimension_label_map.get(dimension, {})
                label = label_lookup.get(value, f"{dimension}_{value}")
                labels.append(label)

            record = {
                "text": base_record.get("text", ""),
                "labels": labels,
                "meta": meta,
            }
            records.append(record)
        return records

    def _write_consensus_file(
        self,
        candidate: ValidationCandidate,
        workspace: Dict[str, Path],
        method: str,
        records: List[Dict[str, Any]],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self._slugify(candidate.dataset or candidate.session_name or candidate.session_id)
        gold_dir = (workspace["root"] / "gold").resolve()
        gold_dir.mkdir(parents=True, exist_ok=True)
        filename = f"consensus_{method}_{slug}_{timestamp}.jsonl"
        path = gold_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        return path

    def _compute_agreement_metrics(
        self,
        annotators: List[AnnotatorAnnotations],
        records_by_key: Dict[str, Dict[str, Any]],
        excluded_keys: List[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        agreement_rows: List[Dict[str, Any]] = []
        agreement_summary: Dict[str, Any] = {"pairwise_kappa": {}, "krippendorff_alpha": {}}

        valid_keys = [key for key in records_by_key.keys() if key not in set(excluded_keys)]
        if not valid_keys or len(annotators) < 2:
            return agreement_rows, agreement_summary

        # Collect dimensions present across valid keys
        dimensions: Set[str] = set()
        for key in valid_keys:
            info = records_by_key[key]
            for values in info["annotator_values"].values():
                dimensions.update(values.keys())

        # Pairwise Cohen's kappa
        for ann_a, ann_b in combinations(annotators, 2):
            overall_a: List[str] = []
            overall_b: List[str] = []
            for dimension in dimensions:
                ratings_a: List[str] = []
                ratings_b: List[str] = []
                for key in valid_keys:
                    info = records_by_key[key]
                    value_a = info["annotator_values"].get(ann_a.name, {}).get(dimension)
                    value_b = info["annotator_values"].get(ann_b.name, {}).get(dimension)
                    if value_a is None or value_b is None:
                        continue
                    ratings_a.append(value_a)
                    ratings_b.append(value_b)
                    overall_a.append(value_a)
                    overall_b.append(value_b)
                if len(ratings_a) >= 2:
                    kappa = cohen_kappa_score(ratings_a, ratings_b)
                    agreement_rows.append(
                        {
                            "scope": "agreement",
                            "segment": f"{ann_a.name}|{ann_b.name}::{dimension}",
                            "precision": "",
                            "recall": "",
                            "f1": "",
                            "jaccard": "",
                            "exact_match": "",
                            "hamming_loss": "",
                            "support_true": "",
                            "support_pred": "",
                            "samples": len(ratings_a),
                            "match_key": "",
                            "model_file": "",
                            "manual_file": "",
                            "cohen_kappa": round(float(kappa), 6),
                            "krippendorff_alpha": "",
                            "consensus_method": "",
                            "tie_policy": "",
                            "annotator_weights": "",
                            "aggregation_notes": "",
                        }
                    )
            if len(overall_a) >= 2:
                kappa = cohen_kappa_score(overall_a, overall_b)
                agreement_summary["pairwise_kappa"][f"{ann_a.name}|{ann_b.name}"] = round(float(kappa), 6)

        # Krippendorff's alpha per dimension
        for dimension in dimensions:
            data: Dict[str, List[str]] = {}
            for key in valid_keys:
                info = records_by_key[key]
                ratings: List[str] = []
                for annotator in annotators:
                    value = info["annotator_values"].get(annotator.name, {}).get(dimension)
                    ratings.append(value)
                if any(rating is not None for rating in ratings):
                    data[key] = ratings
            alpha = self._compute_krippendorff_alpha_nominal(data)
            agreement_summary["krippendorff_alpha"][dimension] = round(float(alpha), 6) if not math.isnan(alpha) else ""
            agreement_rows.append(
                {
                    "scope": "agreement",
                    "segment": f"alpha::{dimension}",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "jaccard": "",
                    "exact_match": "",
                    "hamming_loss": "",
                    "support_true": "",
                    "support_pred": "",
                    "samples": len(data),
                    "match_key": "",
                    "model_file": "",
                    "manual_file": "",
                    "cohen_kappa": "",
                    "krippendorff_alpha": round(float(alpha), 6) if not math.isnan(alpha) else "",
                    "consensus_method": "",
                    "tie_policy": "",
                    "annotator_weights": "",
                    "aggregation_notes": "",
                }
            )

        return agreement_rows, agreement_summary

    def _compute_krippendorff_alpha_nominal(self, data: Dict[str, List[Optional[str]]]) -> float:
        if not data:
            return float("nan")

        category_counts: Counter = Counter()
        observed_disagreement = 0.0
        pair_count = 0.0

        for ratings in data.values():
            present = [rating for rating in ratings if rating is not None]
            n = len(present)
            if n <= 1:
                continue
            pair_count += n * (n - 1)
            for rating in present:
                category_counts[rating] += 1
            for i in range(n):
                for j in range(i + 1, n):
                    if present[i] != present[j]:
                        observed_disagreement += 2.0

        if pair_count == 0:
            return float("nan")
        Do = observed_disagreement / pair_count

        total = sum(category_counts.values())
        if total <= 1:
            return float("nan")
        expected = 0.0
        for count in category_counts.values():
            expected += count * (total - count)
        De = expected / (total * (total - 1))
        if De == 0:
            return 1.0
        return 1.0 - (Do / De)

    def _format_weight_summary(self, weights: Dict[str, float]) -> str:
        if not weights:
            return ""
        parts = [f"{name}:{round(weight, 3)}" for name, weight in sorted(weights.items())]
        return "; ".join(parts)

    def _build_annotation_index(
        self,
        annotators: List[AnnotatorAnnotations],
        join_key: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
        records_by_key: Dict[str, Dict[str, Any]] = {}
        key_sets: List[Set[str]] = []
        for annotator in annotators:
            mapping: Dict[str, Dict[str, Any]] = {}
            keys: Set[str] = set()
            for record in annotator.records:
                key_value = self._extract_join_value(record, join_key)
                if not key_value:
                    continue
                if key_value not in mapping:
                    mapping[key_value] = record
                    keys.add(key_value)
            annotator._key_mapping = mapping  # type: ignore[attr-defined]
            key_sets.append(keys)

        shared_keys = set.intersection(*key_sets) if key_sets else set()
        for key in shared_keys:
            base_record: Optional[Dict[str, Any]] = None
            dimension_label_map: Dict[str, Dict[str, str]] = defaultdict(dict)
            annotator_values: Dict[str, Dict[str, str]] = {}
            for annotator in annotators:
                mapping = getattr(annotator, "_key_mapping", {})
                record = mapping.get(key)
                if not record:
                    continue
                if base_record is None:
                    base_record = record
                dims = self._extract_dimension_values(record)
                annotator_values[annotator.name] = dims
                for dimension, value in dims.items():
                    label_candidate = f"{dimension}_{value}"
                    label = label_candidate
                    for existing_label in record.get("labels", []):
                        if existing_label == label_candidate or existing_label.endswith(f"_{value}"):
                            label = existing_label
                            break
                    dimension_label_map.setdefault(dimension, {})[value] = label
            if base_record:
                records_by_key[key] = {
                    "base_record": base_record,
                    "dimension_label_map": dimension_label_map,
                    "annotator_values": annotator_values,
                }
        return records_by_key, shared_keys

    def _run_dawid_skene(
        self,
        item_values: Dict[str, Dict[str, str]],
        annotators: List[AnnotatorAnnotations],
        max_iter: int = 30,
        tol: float = 1e-4,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Dict[str, float]]]:
        items = list(item_values.keys())
        annotator_names = [annotator.name for annotator in annotators]
        worker_index = {name: idx for idx, name in enumerate(annotator_names)}

        categories = sorted({value for values in item_values.values() for value in values.values()})
        if not categories:
            return {}, {}
        if len(categories) == 1:
            category = categories[0]
            return {item: category for item in items}, {item: {category: 1.0} for item in items}

        category_index = {value: idx for idx, value in enumerate(categories)}
        n_items = len(items)
        n_workers = len(annotator_names)
        n_categories = len(categories)

        # Responses per item
        responses: List[List[Tuple[int, int]]] = [[] for _ in range(n_items)]
        for item_idx, item in enumerate(items):
            for annotator_name, value in item_values[item].items():
                worker = worker_index[annotator_name]
                cat = category_index[value]
                responses[item_idx].append((worker, cat))

        gamma = np.full((n_items, n_categories), 1.0 / n_categories)
        pi = np.full(n_categories, 1.0 / n_categories)
        confusion = np.full((n_workers, n_categories, n_categories), 1.0 / n_categories)

        for _ in range(max_iter):
            gamma_prev = gamma.copy()
            # E-step
            for item_idx in range(n_items):
                posterior = pi.copy()
                for worker, observed_cat in responses[item_idx]:
                    posterior *= confusion[worker, :, observed_cat]
                if posterior.sum() == 0.0:
                    posterior = np.full(n_categories, 1.0 / n_categories)
                else:
                    posterior /= posterior.sum()
                gamma[item_idx] = posterior

            # M-step
            pi = gamma.sum(axis=0)
            pi /= pi.sum()

            for worker in range(n_workers):
                conf = np.zeros((n_categories, n_categories))
                for item_idx in range(n_items):
                    observed = [cat for w, cat in responses[item_idx] if w == worker]
                    if not observed:
                        continue
                    for cat in observed:
                        conf[:, cat] += gamma[item_idx]
                row_sums = conf.sum(axis=1, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    conf = np.divide(
                        conf,
                        row_sums,
                        out=np.full_like(conf, 1.0 / n_categories),
                        where=row_sums > 0,
                    )
                confusion[worker] = conf

            if np.max(np.abs(gamma - gamma_prev)) < tol:
                break

        consensus: Dict[str, Optional[str]] = {}
        posterior_map: Dict[str, Dict[str, float]] = {}
        for item_idx, item in enumerate(items):
            observed_values = item_values[item]
            unique_observed = {value for value in observed_values.values() if value is not None}
            if len(unique_observed) == 1:
                chosen = next(iter(unique_observed))
                consensus[item] = chosen
                posterior_map[item] = {chosen: 1.0}
                continue
            probs = gamma[item_idx]
            top = probs.max()
            winners = [categories[idx] for idx, value in enumerate(probs) if math.isclose(value, top, rel_tol=1e-9, abs_tol=1e-12)]
            consensus[item] = winners[0] if len(winners) == 1 else None
            posterior_map[item] = {categories[idx]: float(probs[idx]) for idx in range(n_categories)}

        return consensus, posterior_map

    def _extract_dimension_values(self, record: Dict[str, Any]) -> Dict[str, str]:
        meta = record.get("meta") or {}
        annotation_json = meta.get("annotation_json")
        dims: Dict[str, str] = {}
        if isinstance(annotation_json, dict):
            for key, value in annotation_json.items():
                normalised_value = self._normalise_label_value(value)
                if normalised_value is None:
                    continue
                dims[str(key)] = normalised_value

        labels = record.get("labels", [])
        if labels:
            known_dimensions = set(dims.keys())
            for label in labels:
                dimension, value = self._split_label_dimension_value(label, known_dimensions)
                if dimension is None:
                    continue
                known_dimensions.add(dimension)
                dims[dimension] = value or "present"

        return {dimension: value for dimension, value in dims.items() if value is not None}

    def _extract_join_value(self, record: Dict[str, Any], join_key: str) -> Optional[str]:
        if join_key == "text":
            return self._normalise_text(record.get("text", ""))
        meta = record.get("meta") or {}
        value = meta.get(join_key)
        if value is None:
            if join_key == "id":
                fallback = meta.get("example_id") or meta.get("source_id")
                if fallback is not None:
                    return str(fallback)
            return None
        return str(value)


    def _collect_manual_jsonl_candidates(
        self,
        candidate: ValidationCandidate,
        model_path: Path,
        workspace: Dict[str, Path],
    ) -> List[Path]:
        search_dirs = {
            model_path.parent.resolve(),
            workspace["root"].resolve(),
            workspace["doccano"].resolve(),
        }
        validation_dir_hint = candidate.validation_session_dir
        if validation_dir_hint:
            try:
                hinted = Path(validation_dir_hint).expanduser().resolve()
                search_dirs.add(hinted)
                search_dirs.add((hinted / "doccano").resolve())
            except Exception:
                pass
        legacy_validation_dir = (self._validation_data_root / candidate.session_id).resolve()
        search_dirs.add(legacy_validation_dir)
        search_dirs.add((legacy_validation_dir / "doccano").resolve())
        manual_paths: List[Path] = []
        seen: Set[Path] = set()
        model_resolved = model_path.resolve()
        for directory in search_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.jsonl")):
                resolved = path.resolve()
                if resolved == model_resolved:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                manual_paths.append(resolved)
        return manual_paths

    def _export_disagreements(
        self,
        consensus: ConsensusBundle,
        session_dirs: Dict[str, Path],
        slug: str,
        timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        if len(consensus.annotators) <= 1:
            return None

        records_by_key, shared_keys = self._build_annotation_index(consensus.annotators, consensus.join_key)
        if not shared_keys:
            return None

        annotator_names = [annotator.name for annotator in consensus.annotators]
        disagreements_jsonl: List[Dict[str, Any]] = []
        disagreements_csv: List[Dict[str, Any]] = []

        consensus_lookup: Dict[str, Dict[str, Any]] = {}
        for record in consensus.consensus_records:
            key = self._extract_join_value(record, consensus.join_key)
            if key:
                consensus_lookup[key] = record

        for key in sorted(shared_keys):
            info = records_by_key[key]
            annotator_values = info["annotator_values"]

            dimension_votes: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
            for annotator_name, values in annotator_values.items():
                for dimension, raw_value in values.items():
                    dimension_votes[dimension][annotator_name] = self._normalise_label_value(raw_value)

            if not dimension_votes:
                continue

            for dimension in list(dimension_votes.keys()):
                for annotator_name in annotator_names:
                    dimension_votes[dimension].setdefault(annotator_name, None)

            disagreement_dimensions = {
                dimension: votes for dimension, votes in dimension_votes.items() if self._has_disagreement(votes)
            }

            if not disagreement_dimensions:
                continue

            base_record = info["base_record"]
            text = base_record.get("text", "")
            base_meta = copy.deepcopy(base_record.get("meta", {}))

            consensus_record = consensus_lookup.get(key)
            consensus_dims = self._extract_dimension_values(consensus_record) if consensus_record else {}
            consensus_labels = []
            consensus_meta = {}
            if consensus_record:
                consensus_labels = consensus_record.get("label") or consensus_record.get("labels") or []
                consensus_meta = consensus_record.get("meta") or {}

            jsonl_meta = copy.deepcopy(base_meta)
            jsonl_meta["disagreement"] = True
            jsonl_meta["join_key"] = key
            jsonl_meta["disagreement_dimensions"] = {}
            jsonl_meta["annotator_votes"] = {}
            if consensus_labels:
                jsonl_meta["consensus_labels"] = consensus_labels
            if consensus_meta:
                jsonl_meta["consensus_meta"] = consensus_meta
            if key in consensus.excluded_keys:
                jsonl_meta["excluded_from_consensus"] = True
            manual_resolution = consensus.manual_resolutions.get(key) if consensus.manual_resolutions else None
            if manual_resolution:
                jsonl_meta["manual_resolution"] = manual_resolution

            for dimension, votes in disagreement_dimensions.items():
                cleaned_votes = {
                    annotator: self._normalise_label_value(value)
                    for annotator, value in votes.items()
                }
                jsonl_meta["disagreement_dimensions"][dimension] = cleaned_votes
                for annotator_name in annotator_names:
                    jsonl_meta["annotator_votes"].setdefault(annotator_name, {})[dimension] = cleaned_votes.get(
                        annotator_name
                    )

                consensus_value = self._normalise_label_value(consensus_dims.get(dimension))
                for annotator_name in annotator_names:
                    disagreements_csv.append(
                        {
                            "join_key": key,
                            "dimension": dimension,
                            "annotator": annotator_name,
                            "value": cleaned_votes.get(annotator_name) or "",
                            "consensus": consensus_value or "",
                            "text": text,
                        }
                    )

            disagreements_jsonl.append(
                {
                    "id": key,
                    "text": text,
                    "label": [],
                    "meta": jsonl_meta,
                }
            )

        if not disagreements_jsonl:
            return None

        reports_dir = session_dirs["reports"].resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / f"disagreements_{slug}_{timestamp}.csv"
        jsonl_path = reports_dir / f"disagreements_{slug}_{timestamp}.jsonl"

        csv_fieldnames = ["join_key", "dimension", "annotator", "value", "consensus", "text"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row in disagreements_csv:
                writer.writerow(row)

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in disagreements_jsonl:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        return {
            "segments": len(disagreements_jsonl),
            "csv_path": str(csv_path),
            "jsonl_path": str(jsonl_path),
        }

    def _load_annotation_file(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    if self.logger:
                        self.logger.warning("Skipping invalid JSON at line %s in %s", idx, path)
                    continue
                text = str(data.get("text", "") or "")
                meta = data.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                records.append(
                    {
                        "text": text,
                        "labels": self._extract_labels(data),
                        "meta": meta,
                        "line": idx,
                    }
                )
        return records

    def _extract_labels(self, payload: Dict[str, Any]) -> List[str]:
        labels_field = payload.get("labels")
        if labels_field is None:
            labels_field = payload.get("label")
        labels: List[str] = []
        if isinstance(labels_field, list):
            for item in labels_field:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 3:
                        value = self._clean_label_token(item[2])
                        if value:
                            labels.append(value)
                else:
                    value = self._clean_label_token(item)
                    if value:
                        labels.append(value)
        elif isinstance(labels_field, str):
            value = self._clean_label_token(labels_field)
            if value:
                labels.append(value)
        elif labels_field is not None:
            value = self._clean_label_token(labels_field)
            if value:
                labels.append(value)
        return labels

    def _normalise_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _collect_meta_map(self, records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in records:
            meta = record.get("meta", {})
            value = meta.get(key)
            if value is None:
                continue
            if isinstance(value, (list, dict)):
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            mapping.setdefault(value_str, record)
        return mapping

    def _collect_text_map(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in records:
            text = self._normalise_text(record.get("text", ""))
            if not text:
                continue
            mapping.setdefault(text, record)
        return mapping

    def _meta_overlap(self, model_records: List[Dict[str, Any]], human_records: List[Dict[str, Any]], key: str) -> int:
        model_map = self._collect_meta_map(model_records, key)
        human_map = self._collect_meta_map(human_records, key)
        return len(set(model_map.keys()) & set(human_map.keys()))

    def _text_overlap(self, model_records: List[Dict[str, Any]], human_records: List[Dict[str, Any]]) -> int:
        model_map = self._collect_text_map(model_records)
        human_map = self._collect_text_map(human_records)
        return len(set(model_map.keys()) & set(human_map.keys()))

    def _select_join_key(
        self,
        model_records: List[Dict[str, Any]],
        human_records: List[Dict[str, Any]],
    ) -> Tuple[str, int]:
        priority_keys = [
            "source_id",
            "row_id",
            "record_id",
            "global_id",
            "doc_id",
            "document_id",
            "row_index",
            "row_number",
            "id",
            "uuid",
            "hash",
        ]
        for key in priority_keys:
            overlap = self._meta_overlap(model_records, human_records, key)
            if overlap:
                return key, overlap

        model_keys = Counter(k for record in model_records for k in record.get("meta", {}) if record["meta"].get(k))
        human_keys = Counter(k for record in human_records for k in record.get("meta", {}) if record["meta"].get(k))
        common_keys = set(model_keys) & set(human_keys)

        best_key: Optional[str] = None
        best_overlap = 0
        for key in sorted(common_keys):
            overlap = self._meta_overlap(model_records, human_records, key)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        if best_key and best_overlap:
            return best_key, best_overlap

        text_overlap = self._text_overlap(model_records, human_records)
        if text_overlap:
            return "__text__", text_overlap

        return "__index__", min(len(model_records), len(human_records))

    def _align_records(
        self,
        model_records: List[Dict[str, Any]],
        human_records: List[Dict[str, Any]],
        join_key: str,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if join_key == "__index__":
            limit = min(len(model_records), len(human_records))
            return [(human_records[idx], model_records[idx]) for idx in range(limit)]

        if join_key == "__text__":
            model_map = self._collect_text_map(model_records)
            human_map = self._collect_text_map(human_records)
        else:
            model_map = self._collect_meta_map(model_records, join_key)
            human_map = self._collect_meta_map(human_records, join_key)

        matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for key, human_record in human_map.items():
            model_record = model_map.get(key)
            if model_record:
                matched.append((human_record, model_record))
        return matched

    def _compute_metrics_for_candidate(
        self,
        candidate: ValidationCandidate,
        model_path: Path,
        manual_path: Path,
        session_dirs: Dict[str, Path],
        summary: SessionSummary,
        resume_path: Path,
        consensus: Optional[ConsensusBundle] = None,
    ) -> None:
        try:
            model_records = self._load_annotation_file(model_path)
            manual_records = self._load_annotation_file(manual_path)
        except Exception as exc:  # pragma: no cover - defensive
            if self.logger:
                self.logger.error("Failed to load annotations for Validation Lab: %s", exc)
            error = f"\n[red]Unable to read annotation files:[/red] {exc}"
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(error), border_style="red"))
            else:
                print(error.replace("[red]", "").replace("[/red]", ""))
            return

        join_key, overlap = self._select_join_key(model_records, manual_records)
        matched_pairs = self._align_records(model_records, manual_records, join_key)
        if not matched_pairs:
            warning = (
                "\n[yellow]Could not align records between model predictions and manual annotations.[/yellow]\n"
                "Ensure the validated file comes from the same Doccano export or shares identifiers with the model file."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
            return

        label_set = sorted(
            {
                label
                for record in model_records + manual_records
                for label in record.get("labels", [])
                if label
            }
        )
        if not label_set:
            warning = (
                "\n[yellow]No labels detected in the paired files.[/yellow]\n"
                "Add annotations before attempting to compute metrics."
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
            return

        label_to_index = {label: idx for idx, label in enumerate(label_set)}
        samples = len(matched_pairs)
        y_true = np.zeros((samples, len(label_set)), dtype=int)
        y_pred = np.zeros((samples, len(label_set)), dtype=int)

        for row_idx, (human_record, model_record) in enumerate(matched_pairs):
            true_labels = {label.strip() for label in human_record.get("labels", []) if label.strip()}
            pred_labels = {label.strip() for label in model_record.get("labels", []) if label.strip()}

            for label in true_labels:
                idx = label_to_index.get(label)
                if idx is not None:
                    y_true[row_idx, idx] = 1

            for label in pred_labels:
                idx = label_to_index.get(label)
                if idx is not None:
                    y_pred[row_idx, idx] = 1

        total_true = int(y_true.sum())
        total_pred = int(y_pred.sum())

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        jaccard_micro = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
        jaccard_macro = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
        jaccard_weighted = jaccard_score(y_true, y_pred, average="weighted", zero_division=0)

        per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        per_label_pred_support = y_pred.sum(axis=0).astype(int)

        exact_match = float(np.mean(np.all(y_true == y_pred, axis=1))) if samples else 0.0
        ham_loss = hamming_loss(y_true, y_pred) if samples else 0.0

        timestamp_dt = datetime.now()
        timestamp = timestamp_dt.strftime("%Y%m%d_%H%M%S")
        timestamp_iso = timestamp_dt.isoformat(timespec="seconds")
        slug = self._slugify(candidate.dataset or candidate.session_name or candidate.session_id)
        report_name = f"validation_metrics_{slug}_{timestamp}.csv"
        report_path = (session_dirs["reports"] / report_name).resolve()

        rows: List[Dict[str, Any]] = []
        rows.append(
            {
                "scope": "overall",
                "segment": "micro",
                "precision": round(precision_micro, 6),
                "recall": round(recall_micro, 6),
                "f1": round(f1_micro, 6),
                "jaccard": round(jaccard_micro, 6),
                "exact_match": "",
                "hamming_loss": "",
                "support_true": total_true,
                "support_pred": total_pred,
                "samples": samples,
                "match_key": join_key,
                "model_file": str(model_path),
                "manual_file": str(manual_path),
                "cohen_kappa": "",
                "krippendorff_alpha": "",
                "consensus_method": "",
                "tie_policy": "",
                "annotator_weights": "",
                "aggregation_notes": "",
            }
        )
        rows.append(
            {
                "scope": "overall",
                "segment": "macro",
                "precision": round(precision_macro, 6),
                "recall": round(recall_macro, 6),
                "f1": round(f1_macro, 6),
                "jaccard": round(jaccard_macro, 6),
                "exact_match": "",
                "hamming_loss": "",
                "support_true": total_true,
                "support_pred": total_pred,
                "samples": samples,
                "match_key": join_key,
                "model_file": str(model_path),
                "manual_file": str(manual_path),
                "cohen_kappa": "",
                "krippendorff_alpha": "",
                "consensus_method": "",
                "tie_policy": "",
                "annotator_weights": "",
                "aggregation_notes": "",
            }
        )
        rows.append(
            {
                "scope": "overall",
                "segment": "weighted",
                "precision": round(precision_weighted, 6),
                "recall": round(recall_weighted, 6),
                "f1": round(f1_weighted, 6),
                "jaccard": round(jaccard_weighted, 6),
                "exact_match": "",
                "hamming_loss": "",
                "support_true": total_true,
                "support_pred": total_pred,
                "samples": samples,
                "match_key": join_key,
                "model_file": str(model_path),
                "manual_file": str(manual_path),
                "cohen_kappa": "",
                "krippendorff_alpha": "",
                "consensus_method": "",
                "tie_policy": "",
                "annotator_weights": "",
                "aggregation_notes": "",
            }
        )
        rows.append(
            {
                "scope": "overall",
                "segment": "exact_match",
                "precision": "",
                "recall": "",
                "f1": "",
                "jaccard": "",
                "exact_match": round(exact_match, 6),
                "hamming_loss": round(ham_loss, 6),
                "support_true": total_true,
                "support_pred": total_pred,
                "samples": samples,
                "match_key": join_key,
                "model_file": str(model_path),
                "manual_file": str(manual_path),
                "cohen_kappa": "",
                "krippendorff_alpha": "",
                "consensus_method": "",
                "tie_policy": "",
                "annotator_weights": "",
                "aggregation_notes": "",
            }
        )

        for idx, label in enumerate(label_set):
            rows.append(
                {
                    "scope": "label",
                    "segment": label,
                    "precision": round(per_label_precision[idx], 6),
                    "recall": round(per_label_recall[idx], 6),
                    "f1": round(per_label_f1[idx], 6),
                    "jaccard": "",
                    "exact_match": "",
                    "hamming_loss": "",
                    "support_true": int(per_label_support[idx]),
                    "support_pred": int(per_label_pred_support[idx]),
                    "samples": samples,
                    "match_key": join_key,
                    "model_file": str(model_path),
                    "manual_file": str(manual_path),
                    "cohen_kappa": "",
                    "krippendorff_alpha": "",
                    "consensus_method": "",
                    "tie_policy": "",
                    "annotator_weights": "",
                    "aggregation_notes": "",
                }
            )

        if consensus:
            weight_summary = self._format_weight_summary(consensus.weights)
            excluded = len(consensus.excluded_keys)
            manual_total = sum(len(v) for v in consensus.manual_resolutions.values())
            aggregation_notes = f"excluded={excluded};manual={manual_total}" if (excluded or manual_total) else ""
            for row in rows:
                if row.get("scope") != "agreement":
                    row["consensus_method"] = consensus.method
                    row["tie_policy"] = consensus.tie_policy
                    row["annotator_weights"] = weight_summary
                    row["aggregation_notes"] = aggregation_notes

        if consensus and consensus.agreement_rows:
            weight_summary = self._format_weight_summary(consensus.weights)
            for agreement_row in consensus.agreement_rows:
                agreement_row.setdefault("consensus_method", consensus.method)
                agreement_row.setdefault("tie_policy", consensus.tie_policy)
                agreement_row.setdefault("annotator_weights", weight_summary)
                agreement_row.setdefault("aggregation_notes", "")
            rows.extend(consensus.agreement_rows)

        self._write_metrics_csv(rows, report_path)

        stats = {
            "samples": samples,
            "labels": label_set,
            "micro_f1": f1_micro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "micro_precision": precision_micro,
            "micro_recall": recall_micro,
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,
            "weighted_precision": precision_weighted,
            "weighted_recall": recall_weighted,
            "jaccard_micro": jaccard_micro,
            "jaccard_macro": jaccard_macro,
            "jaccard_weighted": jaccard_weighted,
            "exact_match": exact_match,
            "hamming_loss": ham_loss,
            "join_key": join_key,
            "timestamp_iso": timestamp_iso,
        }
        if consensus:
            stats["consensus"] = {
                "method": consensus.method,
                "tie_policy": consensus.tie_policy,
                "weights": {name: float(weight) for name, weight in consensus.weights.items()},
                "excluded_keys": consensus.excluded_keys,
                "manual_resolutions": consensus.manual_resolutions,
                "agreement_summary": consensus.agreement_summary,
                "consensus_path": str(consensus.consensus_path),
                "join_key": consensus.join_key,
            }
            disagreement_exports = self._export_disagreements(consensus, session_dirs, slug, timestamp)
            if disagreement_exports:
                stats["disagreements"] = disagreement_exports

        self._print_metrics_summary(candidate, manual_path, model_path, report_path, stats)
        self._update_summary_with_metrics(summary, resume_path, candidate, model_path, manual_path, report_path, stats)

    def _write_metrics_csv(self, rows: List[Dict[str, Any]], report_path: Path) -> None:
        fieldnames = [
            "scope",
            "segment",
            "precision",
            "recall",
            "f1",
            "jaccard",
            "exact_match",
            "hamming_loss",
            "support_true",
            "support_pred",
            "samples",
            "match_key",
            "model_file",
            "manual_file",
            "cohen_kappa",
            "krippendorff_alpha",
            "consensus_method",
            "tie_policy",
            "annotator_weights",
            "aggregation_notes",
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _print_metrics_summary(
        self,
        candidate: ValidationCandidate,
        manual_path: Path,
        model_path: Path,
        report_path: Path,
        stats: Dict[str, Any],
    ) -> None:
        message = (
            f"[bold green]Validation metrics saved:[/bold green] {report_path}\n"
            f"[cyan]Model export:[/cyan] {model_path}\n"
            f"[cyan]Manual annotations:[/cyan] {manual_path}\n"
            f"[cyan]Samples compared:[/cyan] {stats['samples']:,} • Labels: {len(stats['labels'])} • Join key: {stats['join_key']}\n"
            f"[cyan]Micro F1:[/cyan] {stats['micro_f1']:.3f}  |  [cyan]Macro F1:[/cyan] {stats['macro_f1']:.3f}  |  [cyan]Weighted F1:[/cyan] {stats['weighted_f1']:.3f}\n"
            f"[cyan]Exact match:[/cyan] {stats['exact_match']:.3f}  |  [cyan]Hamming loss:[/cyan] {stats['hamming_loss']:.3f}"
        )
        consensus_stats = stats.get("consensus")
        if consensus_stats:
            weight_summary = self._format_weight_summary(consensus_stats.get("weights", {}))
            exclusions = len(consensus_stats.get("excluded_keys", []))
            manual = sum(len(v) for v in consensus_stats.get("manual_resolutions", {}).values())
            extra = (
                f"\n[cyan]Consensus:[/cyan] {consensus_stats.get('method')} · tie={consensus_stats.get('tie_policy')}"
            )
            if weight_summary:
                extra += f" · weights: {weight_summary}"
            if exclusions or manual:
                extra += f" · excluded: {exclusions} · manual_resolutions: {manual}"
            message += extra
        disagreement_stats = stats.get("disagreements")
        if disagreement_stats:
            message += (
                f"\n[cyan]Disagreements exported:[/cyan] {disagreement_stats['segments']} segment(s)"
                f" → CSV: {disagreement_stats['csv_path']} · JSONL: {disagreement_stats['jsonl_path']}"
            )
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(Text.from_markup(message), border_style="green"))
        else:
            print(
                message.replace("[bold green]", "")
                .replace("[/bold green]", "")
                .replace("[cyan]", "")
                .replace("[/cyan]", "")
            )

    def _update_summary_with_metrics(
        self,
        summary: SessionSummary,
        resume_path: Path,
        candidate: ValidationCandidate,
        model_path: Path,
        manual_path: Path,
        report_path: Path,
        stats: Dict[str, Any],
    ) -> None:
        summary.extra = summary.extra or {}
        metrics_runs = summary.extra.setdefault("metrics_runs", [])
        metrics_runs.append(
            {
                "timestamp": stats["timestamp_iso"],
                "report_path": str(report_path),
                "model_file": str(model_path),
                "manual_file": str(manual_path),
                "source": candidate.source,
                "dataset": candidate.dataset,
                "matched_samples": stats["samples"],
                "label_count": len(stats["labels"]),
                "join_key": stats["join_key"],
                "micro_f1": round(stats["micro_f1"], 6),
                "macro_f1": round(stats["macro_f1"], 6),
                "weighted_f1": round(stats["weighted_f1"], 6),
                "exact_match": round(stats["exact_match"], 6),
                "hamming_loss": round(stats["hamming_loss"], 6),
            }
        )
        if stats.get("consensus"):
            metrics_runs[-1]["consensus"] = stats["consensus"]
        if stats.get("disagreements"):
            metrics_runs[-1]["disagreements"] = stats["disagreements"]
        summary.bump_updated()
        write_summary(resume_path, summary)

    def _slugify(self, value: str) -> str:
        slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip().lower())
        slug = slug.strip("_") or "dataset"
        return slug[:80]

    def _collect_validation_candidates(self) -> List[ValidationCandidate]:
        candidates: List[ValidationCandidate] = []
        candidates.extend(self._collect_annotation_candidates("annotator"))
        candidates.extend(self._collect_annotation_candidates("annotator_factory"))
        candidates.extend(self._collect_training_candidates())
        candidates.sort(key=lambda item: item.updated_at, reverse=True)
        return candidates

    def _collect_annotation_candidates(self, mode: str) -> List[ValidationCandidate]:
        records = self.cli._fetch_resume_records(mode, limit=50)  # type: ignore[attr-defined]
        mode_label = "Annotator" if mode == "annotator" else "Annotator Factory"
        candidates: List[ValidationCandidate] = []
        for record in records:
            candidates.extend(self._extract_annotation_validation_info(mode_label, record))
        return candidates

    def _extract_annotation_validation_info(
        self,
        mode_label: str,
        record: SummaryRecord,
    ) -> List[ValidationCandidate]:
        summary = record.summary
        extra = summary.extra or {}
        enabled = extra.get("validation_lab_enabled") or extra.get("validation_lab_opt_in")
        exports_raw = extra.get("validation_lab_exports")
        exports = dict(exports_raw) if isinstance(exports_raw, dict) else {}
        validation_session_dir = extra.get("validation_lab_session_dir")
        dataset_name = extra.get("dataset")

        metadata_info = self._load_validation_metadata_from_files(summary.session_id, record.directory)
        if metadata_info:
            enabled = metadata_info.get("enabled", enabled)
            metadata_exports = metadata_info.get("exports") or {}
            if isinstance(metadata_exports, dict):
                exports.update(metadata_exports)
            validation_session_dir = metadata_info.get("session_dir") or validation_session_dir
            dataset_name = metadata_info.get("dataset_name") or dataset_name

        resolved_session_dir = self._resolve_validation_session_dir(validation_session_dir, record.directory)
        if resolved_session_dir and resolved_session_dir.exists():
            discovered = self._discover_validation_exports_from_dir(resolved_session_dir)
            enabled = enabled or bool(discovered)
            exports = self._merge_discovered_exports(exports, discovered)
            validation_session_dir = str(resolved_session_dir)
        elif not exports:
            default_dir = (self._validation_data_root / summary.session_id).resolve()
            if default_dir.exists():
                discovered = self._discover_validation_exports_from_dir(default_dir)
                enabled = enabled or bool(discovered)
                exports = self._merge_discovered_exports(exports, discovered)
                validation_session_dir = str(default_dir)

        if not enabled or not exports:
            return []

        candidates: List[ValidationCandidate] = []
        for export_label, export_path in exports.items():
            if not export_path:
                continue
            path_obj = Path(export_path)
            friendly_label = self._friendly_export_label(export_label, path_obj, record.directory)
            candidates.append(
                ValidationCandidate(
                    source=mode_label,
                    mode=summary.mode,
                    session_id=summary.session_id,
                    session_name=summary.session_name,
                    dataset=dataset_name or "-",
                    export_label=friendly_label,
                    export_path=str(path_obj),
                    export_exists=path_obj.exists(),
                    validation_session_dir=validation_session_dir,
                    updated_at=summary.updated_at,
                )
            )
        return candidates

    def _load_validation_metadata_from_files(
        self,
        session_id: str,
        session_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        discover = getattr(self.cli, "_discover_annotation_metadata", None)
        if not discover:
            return None
        metadata_paths = discover(session_id, session_dir)
        for metadata_path in metadata_paths:
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            export_prefs = (data.get("export_preferences") or {})
            validation_meta = export_prefs.get("validation_lab") or {}
            if not validation_meta.get("enabled"):
                continue
            dataset_info = data.get("data_source") or {}
            dataset_name = dataset_info.get("file_name") or dataset_info.get("dataset_name")
            exports = validation_meta.get("exports") or {}
            session_dir_pref = validation_meta.get("session_dir")
            return {
                "enabled": True,
                "exports": exports,
                "session_dir": session_dir_pref,
                "dataset_name": dataset_name,
            }
        return None

    def _resolve_validation_session_dir(
        self,
        session_dir: Optional[str],
        record_directory: Path,
    ) -> Optional[Path]:
        if not session_dir:
            return None

        raw = Path(session_dir)
        record_root = record_directory.resolve()
        validation_root = self._validation_data_root.resolve()
        candidates: List[Path] = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    (Path.cwd() / raw),
                    record_root / raw,
                ]
            )
            relative_for_validation = raw
            if raw.parts and raw.parts[0] == validation_root.name:
                relative_for_validation = Path(*raw.parts[1:]) if len(raw.parts) > 1 else Path()
            if relative_for_validation:
                candidates.append(validation_root / relative_for_validation)
            candidates.append(validation_root / raw.name)

        seen: Set[str] = set()
        fallback: Optional[Path] = None
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if fallback is None:
                fallback = resolved
            if resolved.exists():
                return resolved
        return fallback

    def _discover_validation_exports_from_dir(self, session_dir: Path) -> List[Tuple[str, Path]]:
        exports: List[Tuple[str, Path]] = []
        if not session_dir.exists():
            return exports

        allowed_suffixes = {".jsonl", ".csv", ".parquet"}
        for path in session_dir.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in allowed_suffixes:
                continue
            if path.name.startswith("."):
                continue
            relative = path.relative_to(session_dir)
            if relative.parts:
                label = relative.name if len(relative.parts) == 1 else str(Path(*relative.parts[-2:]))
            else:
                label = path.name
            exports.append((label, path))
        return exports

    def _merge_discovered_exports(
        self,
        existing: Dict[str, str],
        discovered: List[Tuple[str, Path]],
    ) -> Dict[str, str]:
        if not discovered:
            return existing

        merged = dict(existing)
        existing_paths = {Path(p).resolve() for p in merged.values()}

        for label, path in discovered:
            resolved = path.resolve()
            if resolved in existing_paths:
                continue
            unique_label = label
            dedupe_index = 2
            while unique_label in merged:
                unique_label = f"{label} ({dedupe_index})"
                dedupe_index += 1
            merged[unique_label] = str(resolved)
            existing_paths.add(resolved)
        return merged

    def _collect_training_candidates(self) -> List[ValidationCandidate]:
        records = self.cli._fetch_resume_records("training_arena", limit=50)  # type: ignore[attr-defined]
        candidates: List[ValidationCandidate] = []
        seen_paths: Set[Path] = set()
        for record in records:
            summary = record.summary
            exports = self._detect_training_validation_exports(record.directory)
            if not exports:
                metadata_exports = self._extract_training_validation_from_metadata(record.directory)
                exports.extend(metadata_exports)
            summary_exports = self._extract_training_validation_from_summary(summary.extra or {}, record.directory)
            exports.extend(summary_exports)
            validation_dir_candidates: List[Path] = []
            extra = summary.extra or {}
            session_dir_hint = extra.get("validation_lab_session_dir") or extra.get("validation_session_dir")
            resolved_hint = self._resolve_validation_session_dir(session_dir_hint, record.directory)
            if resolved_hint and resolved_hint.exists():
                validation_dir_candidates.append(resolved_hint)
            default_validation_dir = (self._validation_data_root / summary.session_id).resolve()
            if default_validation_dir.exists():
                validation_dir_candidates.append(default_validation_dir)
            for candidate_dir in validation_dir_candidates:
                discovered = self._discover_validation_exports_from_dir(candidate_dir)
                for _, path in discovered:
                    exports.append(path)
            validation_session_dir_str = str(validation_dir_candidates[0]) if validation_dir_candidates else None
            if not exports:
                continue
            dataset_name = (
                (summary.extra or {}).get("data_source", {}).get("file_name")
                or (summary.extra or {}).get("dataset")
                or "-"
            )
            for export_path in exports:
                path_obj = Path(export_path)
                canonical = path_obj.resolve()
                if canonical in seen_paths:
                    continue
                seen_paths.add(canonical)
                candidates.append(
                    ValidationCandidate(
                        source="Training Arena",
                        mode=summary.mode,
                        session_id=summary.session_id,
                        session_name=summary.session_name,
                        dataset=dataset_name,
                        export_label=path_obj.name,
                        export_path=str(path_obj),
                        export_exists=path_obj.exists(),
                        validation_session_dir=validation_session_dir_str,
                        updated_at=summary.updated_at,
                    )
                )
        return candidates

    def _detect_training_validation_exports(self, session_dir: Path) -> List[Path]:
        exports: List[Path] = []
        validation_patterns = [
            "training_data/**/*validation*.csv",
            "training_data/**/*validation*.jsonl",
            "training_data/**/*validation*.parquet",
            "training_metrics/**/*validation*.csv",
        ]
        for pattern in validation_patterns:
            for path in session_dir.glob(pattern):
                if path.is_file():
                    exports.append(path)
        warnings_path = session_dir / "training_data" / "validation_warnings.log"
        if warnings_path.exists():
            exports.append(warnings_path)
        return exports

    def _extract_training_validation_from_metadata(self, session_dir: Path) -> List[Path]:
        metadata_dir = session_dir / "metadata"
        if not metadata_dir.exists():
            return []
        candidates: List[Path] = []
        for metadata_file in metadata_dir.glob("**/*.json"):
            try:
                data = json.loads(metadata_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            validation_info = data.get("validation") or {}
            export_path = validation_info.get("validation_set_path") or validation_info.get("export_path")
            if export_path:
                candidates.append(Path(export_path))
        return candidates

    def _extract_training_validation_from_summary(
        self,
        extra: Dict[str, Any],
        session_dir: Path,
    ) -> List[Path]:
        """Inspect training session summary metadata for validation paths."""
        candidates: Set[Path] = set()

        def _collect(value: Any, hint_from_key: bool = False) -> None:
            if isinstance(value, str):
                lower = value.lower()
                if "validation" in lower and lower.endswith((".csv", ".jsonl", ".parquet", ".log")):
                    path = Path(value)
                    if not path.is_absolute():
                        path = (session_dir / path).resolve()
                    candidates.add(path)
                elif hint_from_key and "validation" in lower:
                    path = Path(value)
                    if not path.is_absolute():
                        path = (session_dir / path).resolve()
                    candidates.add(path)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    _collect(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    key_has_validation = isinstance(key, str) and "validation" in key.lower()
                    _collect(item, hint_from_key=key_has_validation)

        _collect(extra)
        return list(candidates)

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #

    def _ask(self, prompt_text: str, default: Optional[str] = None, choices: Optional[List[str]] = None) -> str:
        if RICH_AVAILABLE and self.console and Prompt:
            kwargs: Dict[str, Any] = {"default": default} if default is not None else {}
            if choices:
                kwargs["choices"] = choices
            return Prompt.ask(f"[bold yellow]{prompt_text}[/bold yellow]", **kwargs)
        # Fallback to standard input
        suffix = f" [{'/'.join(choices)}]" if choices else ""
        default_display = f" (default: {default})" if default is not None else ""
        while True:
            response = input(f"{prompt_text}{suffix}{default_display}: ").strip()
            if not response and default is not None:
                return default
            if choices and response not in choices:
                print(f"Invalid choice. Options: {', '.join(choices)}")
                continue
            if response:
                return response


def run_validation_lab(cli: Any, mode: str = "auto", preferred_session_id: Optional[str] = None) -> None:
    controller = ValidationLabController(cli)
    controller.run(mode=mode, preferred_session_id=preferred_session_id)

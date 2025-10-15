from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import csv
import json
import warnings
from collections import Counter

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
                    "\n[yellow]Aucune session de Validation Lab correspondante trouvée.[/yellow]\n"
                    "Sélection via le menu interactif."
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
        message = (
            "\n[cyan]Validation workspace ready.[/cyan]\n"
            f"• Drop your manually-reviewed Doccano JSONL exports into [white]{doccano_hint}[/white].\n"
            "• Validation Lab will compare these files with the model predictions stored here.\n"
            "• Training Arena validation sets can also be copied into this folder for later workflows.\n"
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

        candidate = self._choose_candidate_for_metrics(candidates)
        if not candidate:
            return

        model_path = Path(candidate.export_path)
        if model_path.suffix.lower() != ".jsonl":
            warning = (
                "\n[yellow]Metrics computation currently supports Doccano JSONL exports only.[/yellow]\n"
                f"[dim]Selected export:[/dim] {model_path}"
            )
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(warning), border_style="yellow"))
            else:
                print(warning.replace("[yellow]", "").replace("[/yellow]", ""))
            return

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

        manual_path = self._prompt_manual_file(candidate, validation_workspace, model_path)
        if not manual_path:
            return

        self._compute_metrics_for_candidate(
            candidate=candidate,
            model_path=model_path,
            manual_path=manual_path,
            session_dirs=session_dirs,
            summary=summary,
            resume_path=resume_path,
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
        selection = self._ask("Select export number for Validation Lab metrics", default="skip", choices=choices)
        if selection == "skip":
            return None
        return options.get(selection)

    def _prompt_manual_file(
        self,
        candidate: ValidationCandidate,
        workspace: Dict[str, Path],
        model_path: Path,
    ) -> Optional[Path]:
        manual_candidates = self._collect_manual_jsonl_candidates(model_path, workspace)
        if manual_candidates:
            if len(manual_candidates) == 1:
                return manual_candidates[0]

            if RICH_AVAILABLE and self.console:
                table = Table(title="Validated JSONL candidates", border_style="cyan")
                table.add_column("#", justify="right", style="cyan", width=4)
                table.add_column("File", style="white")
                for idx, path in enumerate(manual_candidates, 1):
                    table.add_row(str(idx), str(path))
                self.console.print(table)
            else:
                print("\nValidated JSONL candidates:")
                for idx, path in enumerate(manual_candidates, 1):
                    print(f"  {idx}. {path}")

            choices = [str(idx) for idx in range(1, len(manual_candidates) + 1)] + ["back"]
            selection = self._ask("Select validated JSONL to compare", default="1", choices=choices)
            if selection == "back":
                return None
            return manual_candidates[int(selection) - 1]

        hint = workspace["doccano"]
        message = (
            f"\n[yellow]No validated JSONL found in[/yellow] [white]{hint}[/white].\n"
            "Export the reviewed annotations from Doccano and copy the JSONL into this folder before computing metrics.\n"
        )
        if RICH_AVAILABLE and self.console:
            self.console.print(Panel.fit(Text.from_markup(message), border_style="yellow"))
        else:
            print(message.replace("[yellow]", "").replace("[/yellow]", "").replace("[white]", "").replace("[/white]", ""))

        manual_input = self._ask("Path to validated JSONL (leave blank to skip)", default="")
        manual_input = manual_input.strip()
        if not manual_input:
            return None
        manual_path = Path(manual_input).expanduser()
        if not manual_path.exists():
            error = f"\n[red]File not found:[/red] {manual_path}"
            if RICH_AVAILABLE and self.console:
                self.console.print(Panel.fit(Text.from_markup(error), border_style="red"))
            else:
                print(error.replace("[red]", "").replace("[/red]", ""))
            return None
        return manual_path.resolve()

    def _collect_manual_jsonl_candidates(
        self,
        model_path: Path,
        workspace: Dict[str, Path],
    ) -> List[Path]:
        search_dirs = {
            model_path.parent.resolve(),
            workspace["root"].resolve(),
            workspace["doccano"].resolve(),
        }
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
                        value = str(item[2]).strip()
                        if value:
                            labels.append(value)
                else:
                    value = str(item).strip()
                    if value:
                        labels.append(value)
        elif isinstance(labels_field, str):
            value = labels_field.strip()
            if value:
                labels.append(value)
        elif labels_field is not None:
            value = str(labels_field).strip()
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
                }
            )

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
        exports = extra.get("validation_lab_exports") or {}
        validation_session_dir = extra.get("validation_lab_session_dir")
        dataset_name = extra.get("dataset")

        metadata_info = self._load_validation_metadata_from_files(summary.session_id, record.directory)
        if metadata_info:
            enabled = metadata_info.get("enabled", enabled)
            exports = metadata_info.get("exports") or exports
            validation_session_dir = metadata_info.get("session_dir") or validation_session_dir
            dataset_name = metadata_info.get("dataset_name") or dataset_name

        if not enabled or not exports:
            return []

        candidates: List[ValidationCandidate] = []
        for export_label, export_path in exports.items():
            if not export_path:
                continue
            path_obj = Path(export_path)
            candidates.append(
                ValidationCandidate(
                    source=mode_label,
                    mode=summary.mode,
                    session_id=summary.session_id,
                    session_name=summary.session_name,
                    dataset=dataset_name or "-",
                    export_label=export_label,
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
                        validation_session_dir=None,
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

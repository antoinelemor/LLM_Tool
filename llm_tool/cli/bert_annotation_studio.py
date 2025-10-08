"""
PROJECT:
-------
LLMTool

TITLE:
------
bert_annotation_studio.py

MAIN OBJECTIVE:
---------------
Advanced annotation mode using trained BERT models with sophisticated features

Author:
-------
Antoine Lemor
"""

from __future__ import annotations
import os
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import time
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
from llm_tool.utils.system_resources import detect_resources
from llm_tool.utils.language_detector import LanguageDetector
from transformers import AutoTokenizer


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

    def __init__(self, console: Console, settings, logger):
        self.console = console
        self.settings = settings
        self.logger = logger
        self.models_dir = Path(getattr(self.settings.paths, "models_dir", "models"))
        self.data_dir = Path(getattr(self.settings.paths, "data_dir", "data"))
        self._language_assignments: Optional[pd.Series] = None

    def run(self):
        """Main entry point"""
        self._display_welcome()

        try:
            selected_models = self._select_trained_models()
            if not selected_models:
                return
            data_source = self._select_data_source()
            if data_source is None:
                return

            df, column_mapping = self._load_and_analyze_data(data_source, selected_models)
            if df is None or column_mapping is None:
                return

            language_info = self._detect_and_validate_language(df, column_mapping, selected_models)
            if language_info is None:
                return

            correction_config = self._configure_correction()
            annotation_config = self._configure_annotation_options(selected_models, df, column_mapping)
            export_config = self._configure_export_options()

            if self._confirm_and_execute(
                selected_models, data_source, df, column_mapping,
                language_info, correction_config, annotation_config, export_config
            ):
                self.console.print("\n[bold green]✓ Annotation completed successfully![/bold green]")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Annotation cancelled[/yellow]")
        except Exception as e:
            self.console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]")
            self.logger.exception("BERT Annotation Studio error")
        finally:
            input("\nPress Enter to continue...")

    def _display_welcome(self):
        """Display welcome - Banner now handled by advanced_cli.py"""
        # Banner and mode info are now displayed by advanced_cli.py before calling run()
        # This keeps the interface consistent across all modes
        pass

    def _select_trained_models(self) -> Optional[List[Dict[str, Any]]]:
        """Select one or more trained models"""
        self.console.print("\n[bold cyan]Step 1/8: Select Trained Model[/bold cyan]\n")

        if not self.models_dir.exists():
            self.console.print(f"[red]✗ Models directory not found: {self.models_dir}[/red]")
            self.console.print("[yellow]Tip: Train a model first using Training Studio (Mode 5)[/yellow]")
            return None

        model_entries = self._collect_trained_models()
        if not model_entries:
            self.console.print("[yellow]No trained models found[/yellow]")
            return None

        help_panel = Panel.fit(
            Align.center(
                "Select the model(s) that will be used to annotate your texts.\n"
                "You can chain multiple models: each will add its dedicated columns.",
                vertical="middle",
            ),
            title="🎯 Selection Mode",
            border_style="cyan",
        )
        self.console.print(help_panel)

        model_table = Table(title="Available Trained Models", box=box.ROUNDED, show_lines=False)
        model_table.add_column("#", style="cyan", width=4, justify="center")
        model_table.add_column("Model", style="green", width=46, overflow="ellipsis")
        model_table.add_column("Base", style="yellow", width=18, overflow="ellipsis")
        model_table.add_column("Lang", style="magenta", width=8, justify="center")
        model_table.add_column("Labels", style="cyan", width=8, justify="right")
        model_table.add_column("Macro F1", style="bright_white", width=10, justify="right")
        model_table.add_column("Updated", style="dim", width=19)

        for idx, model_info in enumerate(model_entries, 1):
            macro = model_info['metrics'].get('macro_f1')
            macro_text = f"{macro:.3f}" if isinstance(macro, (int, float)) else "—"
            model_table.add_row(
                str(idx),
                self._condense_relative_name(model_info['relative_name']),
                self._shorten_base_model(model_info['base_model']),
                model_info['language'],
                str(model_info['label_count']),
                macro_text,
                model_info['updated_at'].strftime("%Y-%m-%d %H:%M"),
            )

        self.console.print(model_table)

        mode_panel = Panel.fit(
            Align.center(
                "[1] Single model\n"
                "[2] Custom list (order = priority)\n"
                "[3] All available models",
                vertical="middle",
            ),
            title="Selection Options",
            border_style="magenta",
        )
        self.console.print(mode_panel)

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
            order_choice = Prompt.ask(
                "\n[cyan]Annotation order[/cyan] ([1] input priority • [2] alphabetical order)",
                choices=["1", "2"],
                default="1",
            )
            if order_choice == "2":
                selection.sort(key=lambda idx: model_entries[idx - 1]['relative_name'].lower())

        chosen_models = [model_entries[idx - 1] for idx in selection]

        summary = Table(title="Selected Models", box=box.ROUNDED, show_lines=False)
        summary.add_column("#", style="cyan", width=4, justify="center")
        summary.add_column("Model", style="green", overflow="ellipsis")
        summary.add_column("Language", style="magenta", width=8, justify="center")
        for idx, model in enumerate(chosen_models, 1):
            summary.add_row(str(idx), self._condense_relative_name(model['relative_name']), model['language'])
        self.console.print(summary)
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
            relative_name = str(model_dir.relative_to(self.models_dir)).replace(os.sep, "/")
            base_model = model_dir.name
            language = self._infer_language(model_dir, base_model, config)

            id2label = config.get("id2label") or {}
            label_count = len(id2label) if isinstance(id2label, dict) else 0
            if not label_count:
                label2id = config.get("label2id")
                if isinstance(label2id, dict):
                    label_count = len(label2id)

            try:
                newest_mtime = max(
                    (p.stat().st_mtime for p in model_dir.rglob("*") if p.is_file()),
                    default=config_path.stat().st_mtime,
                )
                updated_at = datetime.fromtimestamp(newest_mtime)
            except Exception:
                updated_at = datetime.now()

            entries.append(
                {
                    "path": model_dir,
                    "config": config,
                    "relative_name": relative_name,
                    "base_model": base_model,
                    "language": language,
                    "is_multilingual": language == "MULTI",
                    "label_count": label_count if label_count else 0,
                    "metrics": metrics,
                    "updated_at": updated_at,
                    "column_prefix": self._sanitize_model_prefix(relative_name),
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
            return {"macro_f1": macro, "raw": latest}

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

    def _display_text_length_stats(self, df: pd.DataFrame, text_column: str, model_info: Dict[str, Any]) -> None:
        """Show descriptive statistics for text length (characters, words, tokens)."""
        series = df[text_column].fillna("").astype(str)
        if series.empty:
            self.console.print("[yellow]⚠ Unable to calculate lengths (empty column).[/yellow]")
            return

        total_rows = len(series)
        max_full_analysis = 10000
        sample_size = 5000
        analysis_series = series
        sampled = False

        if total_rows > max_full_analysis:
            sampled = Confirm.ask(
                f"[cyan]{total_rows:,} rows detected. Analyze a random sample of {sample_size}?[/cyan]",
                default=True
            )
            if sampled:
                analysis_series = series.sample(sample_size, random_state=42)
            else:
                analysis_series = series

        lengths = analysis_series.str.len()
        words = analysis_series.str.split().map(len)

        percentiles = [50, 75, 90, 95, 99]
        percentile_values = np.percentile(lengths, percentiles)

        stats_table = Table(title="Text Length Analysis", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green", justify="right")

        stats_table.add_row("Rows analyzed", f"{len(analysis_series):,} / {total_rows:,}")
        stats_table.add_row("Average length (char.)", f"{lengths.mean():.1f}")
        stats_table.add_row("Median length (char.)", f"{percentile_values[0]:.0f}")
        stats_table.add_row("Average length (words)", f"{words.mean():.1f}")
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
            texts_list = analysis_series.tolist()
            batch_size = 256
            for i in range(0, len(texts_list), batch_size):
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

        self.console.print(stats_table)

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
            self.console.print(table)
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
                self.console.print(f"[yellow]⚠ Column '{current_id}' contains duplicates or missing values.[/yellow]")
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
                self.console.print("[cyan]Automatic language detection for each text...[/cyan]")
                texts_list = df[text_column].fillna("").astype(str).tolist()
                results = detector.detect_batch(texts_list, parallel=len(texts_list) > 20)
                codes = []
                for res in results:
                    lang = res.get('language') or 'UNKNOWN'
                    codes.append(lang.upper())
                series = pd.Series(codes, index=df.index)
                info['language_column'] = info.get('language_column') or '__detected_language__'
                info['detection_source'] = info.get('detection_source') or 'detector'

        self._language_assignments = series
        return series

    def _confirm_parallel_config(self, config: Dict[str, Any], resources) -> bool:
        """Explain parallel parameters and ask for confirmation."""
        device_mode = config.get('device_mode', 'both')
        batch_cpu = config.get('batch_size_cpu', 32)
        batch_gpu = config.get('batch_size_gpu', 64)
        chunk_size = config.get('chunk_size', 1024)

        cpu_workers, gpu_workers = self._compute_worker_counts(config, resources)
        total_cpus = max(os.cpu_count() or 1, 1)
        gpu_available = resources.gpu.available

        explanation = Table(title="Parallelization Configuration", box=box.ROUNDED)
        explanation.add_column("Parameter", style="cyan")
        explanation.add_column("Value", style="green", overflow="fold")
        explanation.add_row("Mode", device_mode.upper())
        explanation.add_row("CPU Processes", str(cpu_workers))
        explanation.add_row("GPU Processes", str(gpu_workers))
        explanation.add_row("Batch CPU", str(batch_cpu))
        explanation.add_row("Batch GPU", str(batch_gpu))
        explanation.add_row("Chunk size", str(chunk_size))
        explanation.add_row("Total CPU", str(total_cpus))
        explanation.add_row("GPU available", "Yes" if gpu_available else "No")

        self.console.print(explanation)
        self.console.print(
            "[dim]• [green]Batch[/green] = number of texts processed per batch before model update.\n"
            "• [green]Chunk size[/green] = volume sent to each process (limits transfers).\n"
            "• CPU workers process batches in parallel while GPU worker processes large batches.\n"
            "• For large datasets, increasing GPU batch size speeds up annotation, within GPU memory limits.[/dim]"
        )

        return Confirm.ask("\n[cyan]Confirm this configuration?[/cyan]", default=True)

    def _select_data_source(self) -> Optional[Dict[str, Any]]:
        """Select data source"""
        self.console.print("\n[bold cyan]Step 2/8: Select Data Source[/bold cyan]\n")

        source_choices = [
            "📁 Local file (CSV, TSV, Excel, JSON, JSONL, Parquet, RData/RDS)",
            "🗄️  SQL database (PostgreSQL/MySQL/SQLite/SQL Server/Custom)",
            "← Back"
        ]

        source_table = Table(box=box.ROUNDED)
        source_table.add_column("#", style="cyan", width=4)
        source_table.add_column("Data Source", style="green", width=70)

        for idx, choice in enumerate(source_choices, 1):
            source_table.add_row(str(idx), choice)

        self.console.print(source_table)

        choice = Prompt.ask("\n[cyan]Select data source[/cyan]", choices=["1", "2", "3"], default="1")

        if choice == "3":
            return None
        if choice == "1":
            return self._select_file_source()
        return self._select_sql_source()

    def _select_file_source(self) -> Optional[Dict[str, Any]]:
        """Select file source with auto-detection"""
        from llm_tool.cli.advanced_cli import DataDetector, DatasetInfo

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

            self.console.print(datasets_table)
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
        self.console.print(db_table)

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
            self.console.print(schema_table)
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
            self.console.print(table_table)
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

    def _load_and_analyze_data(self, data_source: Dict[str, Any], models: List[Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Load and analyze data"""
        self.console.print("\n[bold cyan]Step 3/8: Load and Analyze Data[/bold cyan]\n")
        self._language_assignments = None

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

            # Intelligent column detection
            text_candidates: List[Dict[str, Any]] = []
            id_candidates: List[Dict[str, Any]] = []

            for col_name in df.columns:
                series = df[col_name]
                dtype = series.dtype

                if dtype == 'object':
                    non_null = series.dropna()
                    if len(non_null) > 0:
                        avg_length = non_null.astype(str).str.len().mean()
                        if avg_length > 20:
                            text_candidates.append({'name': col_name, 'avg_length': avg_length})

                # Detect potential ID columns (high uniqueness)
                unique_ratio = series.nunique(dropna=True) / max(len(series.dropna()), 1)
                if unique_ratio > 0.98 and dtype in ('object', 'int64', 'int32', 'int16', 'uint64', 'uint32'):
                    id_candidates.append({'name': col_name, 'dtype': dtype, 'unique_ratio': unique_ratio})

            text_candidates.sort(key=lambda x: -x['avg_length'])

            col_table = Table(title="Available Columns", box=box.ROUNDED, show_lines=False)
            col_table.add_column("#", style="cyan", width=4)
            col_table.add_column("Column Name", style="green", width=34)
            col_table.add_column("Type", style="yellow", width=14)
            col_table.add_column("Missing %", style="magenta", width=10, justify="right")
            col_table.add_column("Unique", style="cyan", width=10, justify="right")

            total_rows = len(df)
            for idx, col_name in enumerate(df.columns, 1):
                dtype = df[col_name].dtype
                missing_pct = (df[col_name].isna().sum() / total_rows) * 100 if total_rows else 0
                unique_count = df[col_name].nunique(dropna=True)
                col_table.add_row(
                    str(idx),
                    col_name,
                    str(dtype),
                    f"{missing_pct:.1f}%",
                    f"{unique_count:,}"
                )

            self.console.print(col_table)

            detected_text_idx = df.columns.tolist().index(text_candidates[0]['name']) + 1 if text_candidates else 1

            text_col_idx = Prompt.ask(
                "\n[cyan]Select TEXT column[/cyan]",
                choices=[str(i) for i in range(1, len(df.columns) + 1)],
                default=str(detected_text_idx)
            )
            text_column = df.columns[int(text_col_idx) - 1]

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
                        str(candidate['dtype']),
                        f"{candidate['unique_ratio'] * 100:.1f}%"
                    )
                id_table.add_row("0", "[dim]None[/dim]", "", "")
                self.console.print("\n[cyan]Optional: ID column for traceability[/cyan]")
                self.console.print(id_table)

                id_choice = Prompt.ask(
                    "\n[cyan]Select ID column[/cyan]",
                    choices=[str(i) for i in range(0, len(id_candidates) + 1)],
                    default="0"
                )
                if id_choice != "0":
                    id_column = id_candidates[int(id_choice) - 1]['name']

            column_mapping = {'text': text_column, 'id': id_column, 'language': None}
            self.console.print(f"\n[green]✓ Text column: {text_column}[/green]")
            if id_column:
                self.console.print(f"[green]✓ ID column: {id_column}[/green]")

            # Use first model for text length stats
            self._display_text_length_stats(df, text_column, models[0])
            df, column_mapping = self._ensure_unique_identifier(df, column_mapping)

            return df, column_mapping

        except Exception as e:
            self.console.print(f"[red]✗ Error: {str(e)}[/red]")
            return None, None

    def _detect_and_validate_language(self, df: pd.DataFrame, column_mapping: Dict, models: List[Dict[str, Any]]) -> Optional[Dict]:
        """Detect and validate language"""
        self.console.print("\n[bold cyan]Step 4/8: Language Detection[/bold cyan]\n")

        text_column = column_mapping['text']
        sample_texts = df[text_column].dropna().head(100).astype(str).tolist()

        candidate_language_columns: List[Dict[str, Any]] = []
        for col in df.columns:
            if col == text_column or col == column_mapping.get('id'):
                continue
            if df[col].dtype != 'object':
                continue

            counts = LanguageNormalizer.detect_languages_in_column(df, col)
            if counts:
                candidate_language_columns.append({'name': col, 'counts': counts})

        language_column = None
        language_counts: Dict[str, int] = {}
        detection_source = "auto"

        if candidate_language_columns:
            lang_table = Table(title="Detected Language Columns", box=box.ROUNDED)
            lang_table.add_column("#", style="cyan", width=4)
            lang_table.add_column("Column", style="green", width=30)
            lang_table.add_column("Languages", style="magenta", overflow="fold")

            for idx, candidate in enumerate(candidate_language_columns, 1):
                languages_preview = ", ".join(
                    f"{lang.upper()} ({count})" for lang, count in sorted(candidate['counts'].items(), key=lambda x: -x[1])
                )
                lang_table.add_row(str(idx), candidate['name'], languages_preview[:80] + ("…" if len(languages_preview) > 80 else ""))

            self.console.print("[cyan]Potential language columns detected[/cyan]")
            self.console.print(lang_table)

            if Confirm.ask("Use one of these columns for language detection?", default=True):
                lang_choice = Prompt.ask(
                    "\n[cyan]Select language column[/cyan]",
                    choices=[str(i) for i in range(1, len(candidate_language_columns) + 1)],
                    default="1"
                )
                selected = candidate_language_columns[int(lang_choice) - 1]
                language_column = selected['name']
                language_counts = selected['counts']
                detection_source = "column"

        if language_column is None:
            detected_languages = LanguageNormalizer.detect_dataset_languages(sample_texts)
            if detected_languages:
                for lang_set in detected_languages:
                    for lang in lang_set:
                        language_counts[lang] = language_counts.get(lang, 0) + 1
            else:
                language_counts['en'] = len(sample_texts)
                self.console.print("[yellow]⚠ Unable to confidently detect language, defaulting to English[/yellow]")

        if not language_counts:
            self.console.print("[red]✗ Unable to infer language[/red]")
            return None

        sorted_langs = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        primary_lang = sorted_langs[0][0].upper()
        unique_languages = {lang.upper() for lang in language_counts.keys()}

        counts_table = Table(title="Language Breakdown", box=box.ROUNDED)
        counts_table.add_column("Language", style="green", width=12)
        counts_table.add_column("Samples", style="cyan", justify="right", width=10)
        counts_table.add_column("Share", style="magenta", justify="right", width=10)
        total_detected = sum(language_counts.values())
        for lang, count in sorted_langs:
            share = (count / total_detected * 100) if total_detected else 0
            counts_table.add_row(lang.upper(), f"{count:,}", f"{share:.1f}%")
        self.console.print(counts_table)

        if language_column is None:
            column_mapping['language'] = '__detected_language__' if detection_source != "column" else None
        else:
            column_mapping['language'] = language_column

        # Check compatibility for all models
        model_languages = {m['language'] for m in models}
        has_multilingual = any(m.get('is_multilingual', False) for m in models)

        # Display model languages
        model_lang_str = ", ".join(sorted(model_languages))
        self.console.print(f"[cyan]Model language(s): {model_lang_str} • Primary data language: {primary_lang}[/cyan]")

        # Check if at least one model is compatible
        is_compatible = has_multilingual or primary_lang in model_languages

        if is_compatible:
            self.console.print("[green]✓ At least one model is compatible with detected language[/green]")
        else:
            self.console.print("[yellow]⚠ Language mismatch between models and data[/yellow]")
            if language_column and len(unique_languages) > 1:
                self.console.print("[yellow]• Consider filtering the dataset by language column before annotating.[/yellow]")
            if not Confirm.ask("Proceed with annotation despite the mismatch?", default=False):
                return None

        if len(unique_languages) > 1 and not has_multilingual:
            self.console.print("[yellow]⚠ Multiple languages detected but the model is monolingual.[/yellow]")

        return {
            'primary_language': primary_lang,
            'languages': sorted(unique_languages),
            'counts': {lang.upper(): count for lang, count in language_counts.items()},
            'language_column': language_column,
            'detection_source': detection_source
        }

    def _configure_correction(self) -> Dict[str, Any]:
        """Configure correction"""
        self.console.print("\n[bold cyan]Step 5/8: Text Correction[/bold cyan]\n")

        enable_correction = Confirm.ask("[cyan]Enable preprocessing?[/cyan]", default=True)

        if not enable_correction:
            return {'enabled': False}

        return {
            'enabled': True,
            'lowercase': Confirm.ask("  Lowercase?", default=False),
            'remove_urls': Confirm.ask("  Remove URLs?", default=True),
            'remove_emails': Confirm.ask("  Remove emails?", default=True),
            'remove_extra_spaces': Confirm.ask("  Remove extra spaces?", default=True),
        }

    def _configure_annotation_options(self, models: List[Dict[str, Any]], df: pd.DataFrame, column_mapping: Dict) -> Dict[str, Any]:
        """Configure annotation options"""
        self.console.print("\n[bold cyan]Step 6/8: Annotation Options[/bold cyan]\n")

        resources = detect_resources()
        gpu_available = resources.gpu.available
        recommendations = resources.get_recommendation()
        recommended_batch = max(8, recommendations.get('batch_size', 16))

        resource_table = Table(title="Detected Resources", box=box.ROUNDED)
        resource_table.add_column("Component", style="cyan", width=16)
        resource_table.add_column("Details", style="green", overflow="fold")
        resource_table.add_row(
            "GPU",
            "Available" if gpu_available else "CPU only"
        )
        if gpu_available:
            resource_table.add_row("GPU Type", ", ".join(resources.gpu.device_names) or resources.gpu.device_type.upper())
            resource_table.add_row("GPU Memory", f"{resources.gpu.total_memory_gb:.1f} GB")
        resource_table.add_row(
            "CPU",
            f"{resources.cpu.physical_cores} cores / {resources.cpu.logical_cores} threads"
        )
        resource_table.add_row(
            "RAM Available",
            f"{resources.memory.available_gb:.1f} GB / {resources.memory.total_gb:.1f} GB"
        )
        self.console.print(resource_table)

        while True:
            strategy_table = Table(title="Parallelisation Strategies", box=box.ROUNDED)
            strategy_table.add_column("#", style="cyan", width=4)
            strategy_table.add_column("Strategy", style="green", width=24)
            strategy_table.add_column("Description", style="magenta", overflow="fold")

            strategy_table.add_row("1", "Auto (recommended)", "Dynamic CPU/GPU mix based on detected resources")
            if gpu_available:
                strategy_table.add_row("2", "GPU only", "Single GPU worker with large batches")
            strategy_table.add_row("3", "CPU only", "Only CPU workers (multi-process if possible)")
            strategy_table.add_row("4", "Manual", "Custom device mode, workers and batch sizes")
            self.console.print(strategy_table)

            valid_choices = ["1", "3", "4"] if not gpu_available else ["1", "2", "3", "4"]
            strategy_choice = Prompt.ask(
                "\n[cyan]Select parallelisation strategy[/cyan]",
                choices=valid_choices,
                default="1"
            )

            config: Dict[str, Any] = {}
            if strategy_choice == "1":  # Auto
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

            elif strategy_choice == "2":  # GPU only
                config['parallel'] = True
                config['device_mode'] = 'gpu'
                config['batch_size_gpu'] = max(32, recommended_batch)
                config['batch_size_cpu'] = max(8, recommended_batch // 2)
                config['chunk_size'] = max(256, config['batch_size_gpu'] * 8)

            elif strategy_choice == "3":  # CPU only
                config['parallel'] = True
                config['device_mode'] = 'cpu'
                config['batch_size_cpu'] = max(8, recommended_batch)
                config['batch_size_gpu'] = config['batch_size_cpu']
                config['chunk_size'] = max(256, config['batch_size_cpu'] * 6)

            else:  # Manual
                device_mode_choices = ["cpu"]
                if gpu_available:
                    device_mode_choices.extend(["gpu", "both"])
                device_mode = Prompt.ask(
                    "Device mode",
                    choices=device_mode_choices,
                    default="both" if gpu_available else "cpu"
                )
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

            self.console.print(
                f"[green]✓ Parallel config:[/green] "
                f"mode={config.get('device_mode')} • parallel={'yes' if config.get('parallel') else 'no'} • "
                f"CPU batch={config.get('batch_size_cpu')} • GPU batch={config.get('batch_size_gpu')} • "
                f"chunk={config.get('chunk_size')}"
            )

            if self._confirm_parallel_config(config, resources):
                return config

            self.console.print("[yellow]Reconfiguration requested by user.[/yellow]\n")

    def _configure_export_options(self) -> Dict[str, Any]:
        """Configure export"""
        self.console.print("\n[bold cyan]Step 7/8: Export Options[/bold cyan]\n")

        default_output = f"bert_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        include_probs = Confirm.ask("Include prediction probability column?", default=True)
        include_ci = include_probs and Confirm.ask("Include confidence interval columns?", default=True)

        return {
            'output_file': Prompt.ask("[cyan]Output file[/cyan]", default=default_output),
            'export_to_tools': False,
            'include_probabilities': include_probs,
            'include_confidence_interval': include_ci
        }

    def _confirm_and_execute(self, models: List[Dict[str, Any]], data_source: Dict, df: pd.DataFrame,
                            column_mapping: Dict, language_info: Dict, correction_config: Dict,
                            annotation_config: Dict, export_config: Dict) -> bool:
        """Execute annotation"""
        self.console.print("\n[bold cyan]Step 8/8: Execute[/bold cyan]\n")

        # Use first model for initial compatibility check (will process all models later)
        model_info = models[0]

        row_languages = self._get_or_compute_row_languages(df, column_mapping, language_info or {})
        if model_info.get('is_multilingual'):
            eligible_mask = pd.Series(True, index=df.index)
        else:
            eligible_mask = row_languages == model_info['language']

        eligible_count = int(eligible_mask.sum())
        skipped_count = len(df) - eligible_count

        # Build models display string
        models_str = ", ".join([m['relative_name'] for m in models])
        if len(models_str) > 80:
            models_str = models_str[:77] + "..."

        summary = Table(title="Annotation Summary", box=box.ROUNDED)
        summary.add_column("Parameter", style="cyan", width=18)
        summary.add_column("Value", style="green", overflow="fold")
        summary.add_row("Model(s)", models_str)
        summary.add_row("Model language(s)", ", ".join(sorted({m['language'] for m in models})))
        summary.add_row("Rows", f"{len(df):,}")
        if language_info:
            languages_str = ", ".join(language_info.get('languages', []))
            summary.add_row("Data languages", languages_str or language_info.get('primary_language', 'N/A'))
        summary.add_row("Text column", column_mapping['text'])
        if column_mapping.get('id'):
            summary.add_row("ID column", column_mapping['id'])
        language_column_value = column_mapping.get('language')
        if language_column_value:
            if language_column_value == '__detected_language__':
                summary.add_row("Language column", "Auto (text detection)")
            else:
                summary.add_row("Language column", language_column_value)
        summary.add_row("Output", export_config['output_file'])
        summary.add_row("Rows annotated", f"{eligible_count:,}")
        if skipped_count:
            summary.add_row("Ignored (language mismatch)", f"{skipped_count:,}")
        summary.add_row(
            "Parallel strategy",
            f"{annotation_config.get('device_mode')} • chunk={annotation_config.get('chunk_size')} • "
            f"CPU batch={annotation_config.get('batch_size_cpu')} • GPU batch={annotation_config.get('batch_size_gpu')}"
        )

        self.console.print(summary)

        if skipped_count:
            self.console.print(
                f"[yellow]{skipped_count:,} row(s) will be skipped as the detected language does not match {model_info['language']}.[/yellow]"
            )

        if eligible_count == 0:
            self.console.print(
                f"[red]✗ No rows in {model_info['language']} were found for this model.[/red]"
            )
            return False

        if not Confirm.ask("\n[cyan]Start?[/cyan]", default=True):
            return False

        try:
            texts = df.loc[eligible_mask, column_mapping['text']].fillna("").astype(str).tolist()
            if not texts:
                self.console.print("[yellow]No texts found to annotate.[/yellow]")
                return False

            if correction_config['enabled']:
                texts = self._apply_corrections(texts, correction_config)

            self.console.print("\n[cyan]Running inference...[/cyan]")
            start_time = time.time()

            resources_snapshot = detect_resources()

            progress_columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ]

            cpu_workers, gpu_workers = self._compute_worker_counts(annotation_config, resources_snapshot)

            def run_inference_with_progress() -> np.ndarray:
                total = len(texts)
                chunk_size = annotation_config.get('chunk_size', 1024)
                device_counts = defaultdict(int)
                result_array: Optional[np.ndarray] = None

                with Progress(*progress_columns, console=self.console, transient=True) as progress:
                    overall_task = progress.add_task("Annotation totale", total=total)
                    device_tasks: Dict[str, int] = {}

                    def get_label(tag: str) -> str:
                        if tag in {"cuda", "gpu"}:
                            suffix = "(CUDA)" if tag == "cuda" else ""
                            return f"Worker GPU {suffix}".strip()
                        if tag == "mps":
                            return "Worker GPU (Apple MPS)"
                        cpu_info = f"{cpu_workers} worker(s)" if cpu_workers > 1 else "1 worker"
                        return f"Workers CPU ({cpu_info})"

                    def ensure_task(tag: str) -> int:
                        if tag not in device_tasks:
                            device_tasks[tag] = progress.add_task(get_label(tag), total=total)
                        return device_tasks[tag]

                    def progress_handler(processed: int, device_tag: str) -> None:
                        progress.update(overall_task, advance=processed)
                        device_task_id = ensure_task(device_tag)
                        progress.update(device_task_id, advance=processed)
                        device_counts[device_tag] += processed

                    result_array = parallel_predict(
                        texts=texts,
                        model_path=str(model_info['path']),
                        lang=model_info['language'],
                        parallel=annotation_config.get('parallel', True),
                        device_mode=annotation_config.get('device_mode', 'both'),
                        batch_size_cpu=annotation_config.get('batch_size_cpu', 32),
                        batch_size_gpu=annotation_config.get('batch_size_gpu', 64),
                        show_progress=False,
                        chunk_size=chunk_size,
                        progress_handler=progress_handler
                    )

                    for tag, task_id in device_tasks.items():
                        processed = device_counts.get(tag, 0)
                        progress.update(task_id, total=max(processed, 1), completed=processed)

                if result_array is None:
                    raise RuntimeError("Inference did not produce any output.")
                return result_array

            predictions = run_inference_with_progress()
            duration = time.time() - start_time

            predicted_label_ids = np.argmax(predictions, axis=1)
            top_probabilities = predictions[np.arange(len(predictions)), predicted_label_ids]

            label_map = model_info['config'].get('id2label', {})
            predicted_labels = [
                label_map.get(str(label_id), str(label_id))
                for label_id in predicted_label_ids
            ]

            prefix = model_info.get('column_prefix') or self._sanitize_model_prefix(model_info['relative_name'])
            label_id_col = f"{prefix}_label_id"
            label_col = f"{prefix}_label"

            df_result = df.copy()
            df_result[label_id_col] = np.nan
            df_result[label_col] = None

            prob_col = ci_low_col = ci_high_col = None
            include_prob = export_config.get('include_probabilities', True)
            include_ci = include_prob and export_config.get('include_confidence_interval', True)

            if include_prob:
                prob_col = f"{prefix}_probability"
                df_result[prob_col] = np.nan
            if include_ci:
                ci_low_col = f"{prefix}_ci_lower"
                ci_high_col = f"{prefix}_ci_upper"
                df_result[ci_low_col] = np.nan
                df_result[ci_high_col] = np.nan

            df_result[f"{prefix}_language"] = row_languages.values
            df_result[f"{prefix}_annotated"] = eligible_mask.values

            df_result.loc[eligible_mask, label_id_col] = predicted_label_ids
            df_result.loc[eligible_mask, label_col] = predicted_labels

            if include_prob:
                df_result.loc[eligible_mask, prob_col] = top_probabilities

            if include_ci:
                ci_values = [self._compute_confidence_interval(row) for row in predictions]
                df_result.loc[eligible_mask, ci_low_col] = [ci[0] for ci in ci_values]
                df_result.loc[eligible_mask, ci_high_col] = [ci[1] for ci in ci_values]

            mismatched_mask = ~eligible_mask
            if mismatched_mask.any():
                df_result.loc[mismatched_mask, label_id_col] = -1
                df_result.loc[mismatched_mask, label_col] = "LANGUAGE_MISMATCH"
                if include_prob:
                    df_result.loc[mismatched_mask, prob_col] = 0.0
                if include_ci:
                    df_result.loc[mismatched_mask, ci_low_col] = 0.0
                    df_result.loc[mismatched_mask, ci_high_col] = 0.0

            output_path = Path(export_config['output_file'])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            suffix = output_path.suffix.lower()
            if suffix == '.csv' or suffix == '':
                if suffix == '':
                    output_path = output_path.with_suffix('.csv')
                df_result.to_csv(output_path, index=False)
            elif suffix in {'.parquet', '.pq'}:
                df_result.to_parquet(output_path, index=False)
            elif suffix in {'.json', '.jsonl'}:
                df_result.to_json(output_path, orient='records', lines=(suffix == '.jsonl'), force_ascii=False)
            elif suffix in {'.xlsx', '.xls'}:
                df_result.to_excel(output_path, index=False)
            else:
                self.console.print(f"[yellow]⚠ Unsupported export format {suffix}, falling back to CSV[/yellow]")
                csv_path = output_path.with_suffix('.csv')
                df_result.to_csv(csv_path, index=False)
                output_path = csv_path
            self.console.print(
                f"\n[green]✓ Annotated {eligible_count:,} row(s) (out of {len(df_result):,}) in {duration:.1f}s[/green]"
            )
            self.console.print(f"[green]✓ Saved to {output_path}[/green]")
            return True

        except Exception as e:
            self.console.print(f"\n[red]✗ Failed: {str(e)}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False

    def _apply_corrections(self, texts: List[str], config: Dict) -> List[str]:
        """Apply corrections"""
        corrected = []
        for text in texts:
            if not isinstance(text, str):
                corrected.append("")
                continue

            if config.get('lowercase'):
                text = text.lower()
            if config.get('remove_urls'):
                text = re.sub(r'http[s]?://\S+', '', text)
            if config.get('remove_emails'):
                text = re.sub(r'\S+@\S+', '', text)
            if config.get('remove_extra_spaces'):
                text = re.sub(r'\s+', ' ', text).strip()

            corrected.append(text)

        return corrected

    @staticmethod
    def _compute_confidence_interval(probabilities: np.ndarray) -> Tuple[float, float]:
        """
        Heuristic confidence interval using margin between top-2 class probabilities.
        Provides an intuitive sense of certainty without requiring calibration metadata.
        """
        if probabilities.size == 0:
            return 0.0, 0.0

        sorted_probs = np.sort(probabilities)
        top_prob = float(sorted_probs[-1])
        second_prob = float(sorted_probs[-2]) if probabilities.size > 1 else 0.0
        margin = max(top_prob - second_prob, max(0.05 * top_prob, 0.01))
        lower = max(0.0, top_prob - margin)
        upper = min(1.0, top_prob + margin)
        return lower, upper

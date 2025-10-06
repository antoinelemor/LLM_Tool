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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich import box
from sqlalchemy import create_engine, inspect, text

# Package imports
from llm_tool.trainers.parallel_inference import parallel_predict
from llm_tool.cli.advanced_cli import LanguageNormalizer


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
        self.models_dir = Path("models")

    def run(self):
        """Main entry point"""
        self._display_welcome()

        try:
            model_info = self._select_trained_model()
            if model_info is None:
                return

            data_source = self._select_data_source()
            if data_source is None:
                return

            df, column_mapping = self._load_and_analyze_data(data_source)
            if df is None or column_mapping is None:
                return

            language_info = self._detect_and_validate_language(df, column_mapping, model_info)
            if language_info is None:
                return

            correction_config = self._configure_correction()
            annotation_config = self._configure_annotation_options()
            export_config = self._configure_export_options()

            if self._confirm_and_execute(
                model_info, data_source, df, column_mapping,
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
        """Display welcome"""
        ascii_art = """[bold cyan]
██████╗ ███████╗██████╗ ████████╗    █████╗ ███╗   ██╗███╗   ██╗ ██████╗ ████████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔════╝██╔══██╗╚══██╔══╝   ██╔══██╗████╗  ██║████╗  ██║██╔═══██╗╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
██████╔╝█████╗  ██████╔╝   ██║      ███████║██╔██╗ ██║██╔██╗ ██║██║   ██║   ██║   ███████║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══██╗██╔══╝  ██╔══██╗   ██║      ██╔══██║██║╚██╗██║██║╚██╗██║██║   ██║   ██║   ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
██████╔╝███████╗██║  ██║   ██║      ██║  ██║██║ ╚████║██║ ╚████║╚██████╔╝   ██║   ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
[/bold cyan]
[bold yellow]🚀 Advanced Annotation with Trained BERT/Transformer Models 🚀[/bold yellow]
[dim]📊 Data → 🤖 Trained Model → 🎯 Predictions → 📈 Export[/dim]"""

        self.console.print(ascii_art)

        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="cyan", justify="right")
        info_table.add_column(style="white")
        info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")
        info_table.add_row("🔄 Workflow:", "[cyan]Select Model → Load Data → Detect Language → Correct Text → Annotate → Export[/cyan]")
        info_table.add_row("🎯 Capabilities:", "[magenta]SQL/File Input, Language Detection, Text Correction, Parallel Inference[/magenta]")
        info_table.add_row("📥 Input:", "[green]SQL, CSV, Excel, JSON, JSONL, Parquet[/green]")
        info_table.add_row("📤 Output:", "[yellow]Predictions + exports[/yellow]")

        panel = Panel(info_table, title="[bold]🤖 BERT Annotation Studio[/bold]", border_style="cyan", box=box.DOUBLE)
        self.console.print(panel)

    def _select_trained_model(self) -> Optional[Dict[str, Any]]:
        """Select trained model"""
        self.console.print("\n[bold cyan]Step 1/8: Select Trained Model[/bold cyan]\n")

        if not self.models_dir.exists():
            self.console.print(f"[red]✗ Models directory not found: {self.models_dir}[/red]")
            self.console.print("[yellow]Tip: Train a model first using Training Studio (Mode 5)[/yellow]")
            return None

        model_dirs = []
        for item in self.models_dir.iterdir():
            if item.is_dir():
                config_file = item / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        model_dirs.append({'path': item, 'name': item.name, 'config': config})
                    except:
                        pass

        if not model_dirs:
            self.console.print("[yellow]No trained models found[/yellow]")
            return None

        model_table = Table(title="Available Trained Models", box=box.ROUNDED)
        model_table.add_column("#", style="cyan", width=4)
        model_table.add_column("Model Name", style="green", width=40)
        model_table.add_column("Type", style="yellow", width=20)
        model_table.add_column("Language", style="magenta", width=12)

        for idx, model_info in enumerate(model_dirs, 1):
            config = model_info['config']
            model_type = config.get('model_type', 'bert')
            lang = 'EN'
            for model_key, lang_code in MODEL_LANGUAGE_MAP.items():
                if model_key in model_type.lower():
                    lang = lang_code
                    break
            model_table.add_row(str(idx), model_info['name'], model_type, lang)

        self.console.print(model_table)

        choice = Prompt.ask("\n[cyan]Select model[/cyan]", choices=[str(i) for i in range(1, len(model_dirs) + 1)])
        selected_model = model_dirs[int(choice) - 1]

        config = selected_model['config']
        model_type = config.get('model_type', 'bert')
        model_lang = 'EN'
        for model_key, lang_code in MODEL_LANGUAGE_MAP.items():
            if model_key in model_type.lower():
                model_lang = lang_code
                break

        selected_model['language'] = model_lang
        selected_model['is_multilingual'] = (model_lang == 'MULTI')

        self.console.print(f"\n[green]✓ Selected: {selected_model['name']} ({model_lang})[/green]")
        return selected_model

    def _select_data_source(self) -> Optional[Dict[str, Any]]:
        """Select data source"""
        self.console.print("\n[bold cyan]Step 2/8: Select Data Source[/bold cyan]\n")

        source_choices = ["📁 File (CSV, Excel, JSON, JSONL, Parquet)", "🗄️  SQL Database", "← Back"]

        source_table = Table(box=box.ROUNDED)
        source_table.add_column("#", style="cyan", width=4)
        source_table.add_column("Data Source", style="green", width=60)

        for idx, choice in enumerate(source_choices, 1):
            source_table.add_row(str(idx), choice)

        self.console.print(source_table)

        choice = Prompt.ask("\n[cyan]Select data source[/cyan]", choices=["1", "2", "3"], default="1")

        if choice == "3":
            return None
        elif choice == "1":
            return self._select_file_source()
        else:
            return self._select_sql_source()

    def _select_file_source(self) -> Optional[Dict[str, Any]]:
        """Select file source with auto-detection"""
        from llm_tool.cli.advanced_cli import DataDetector, DatasetInfo

        # Auto-detect datasets in data directory
        data_dir = self.settings.paths.data_dir
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
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.parquet': 'parquet',
            '.rdata': 'rdata',
            '.RData': 'rdata'
        }
        file_format = format_map.get(suffix, 'unknown')

        if file_format == 'unknown':
            self.console.print(f"[red]✗ Unsupported format: {suffix}[/red]")
            return None

        self.console.print(f"[green]✓ Format: {file_format.upper()}[/green]")
        return {'type': 'file', 'path': str(file_path), 'format': file_format}

    def _select_sql_source(self) -> Optional[Dict[str, Any]]:
        """Select SQL source"""
        # Simplified SQL selection (full implementation same as Database Annotator)
        self.console.print("[yellow]SQL source selection - implement full version from Database Annotator[/yellow]")
        return None

    def _load_and_analyze_data(self, data_source: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Load and analyze data"""
        self.console.print("\n[bold cyan]Step 3/8: Load and Analyze Data[/bold cyan]\n")

        try:
            if data_source['type'] == 'file':
                file_format = data_source['format']
                file_path = data_source['path']

                if file_format == 'csv':
                    df = pd.read_csv(file_path)
                elif file_format == 'excel':
                    df = pd.read_excel(file_path)
                elif file_format == 'json':
                    df = pd.read_json(file_path)
                elif file_format == 'jsonl':
                    df = pd.read_json(file_path, lines=True)
                elif file_format == 'parquet':
                    df = pd.read_parquet(file_path)
                elif file_format == 'rdata':
                    # RData support (requires pyreadr)
                    try:
                        import pyreadr
                        result = pyreadr.read_r(file_path)
                        if result:
                            # Take first dataframe from RData file
                            df = list(result.values())[0]
                        else:
                            self.console.print("[red]✗ Empty RData file[/red]")
                            return None, None
                    except ImportError:
                        self.console.print("[red]✗ pyreadr not installed. Install with: pip install pyreadr[/red]")
                        return None, None
                else:
                    self.console.print(f"[red]✗ Unsupported format: {file_format}[/red]")
                    return None, None
            else:
                self.console.print("[red]SQL not yet implemented[/red]")
                return None, None

            self.console.print(f"[green]✓ Loaded {len(df):,} rows, {len(df.columns)} columns[/green]\n")

            # Intelligent column detection
            text_candidates = []
            for col_name in df.columns:
                if df[col_name].dtype == 'object':
                    non_null = df[col_name].dropna()
                    if len(non_null) > 0:
                        avg_length = non_null.astype(str).str.len().mean()
                        if avg_length > 20:
                            text_candidates.append({'name': col_name, 'avg_length': avg_length})

            text_candidates.sort(key=lambda x: -x['avg_length'])

            col_table = Table(title="Columns", box=box.ROUNDED)
            col_table.add_column("#", style="cyan", width=4)
            col_table.add_column("Column Name", style="green", width=30)
            col_table.add_column("Type", style="yellow", width=15)

            for idx, col_name in enumerate(df.columns, 1):
                col_table.add_row(str(idx), col_name, str(df[col_name].dtype))

            self.console.print(col_table)

            detected_text_idx = df.columns.tolist().index(text_candidates[0]['name']) + 1 if text_candidates else 1

            text_col_idx = Prompt.ask("\n[cyan]Select TEXT column[/cyan]",
                                      choices=[str(i) for i in range(1, len(df.columns) + 1)],
                                      default=str(detected_text_idx))
            text_column = df.columns[int(text_col_idx) - 1]

            column_mapping = {'text': text_column, 'id': None}
            self.console.print(f"\n[green]✓ Text column: {text_column}[/green]")

            return df, column_mapping

        except Exception as e:
            self.console.print(f"[red]✗ Error: {str(e)}[/red]")
            return None, None

    def _detect_and_validate_language(self, df: pd.DataFrame, column_mapping: Dict, model_info: Dict) -> Optional[Dict]:
        """Detect and validate language"""
        self.console.print("\n[bold cyan]Step 4/8: Language Detection[/bold cyan]\n")

        text_column = column_mapping['text']
        sample_texts = df[text_column].dropna().head(100).tolist()

        detected_languages = LanguageNormalizer.detect_dataset_languages(sample_texts)

        if detected_languages:
            lang_counts = {}
            for lang_set in detected_languages:
                for lang in lang_set:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

            sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
            primary_lang = sorted_langs[0][0]

            self.console.print(f"[green]✓ Detected: {primary_lang}[/green]")
        else:
            primary_lang = "EN"
            self.console.print("[yellow]⚠ Assuming English[/yellow]")

        self.console.print(f"[cyan]Model: {model_info['language']}, Data: {primary_lang}[/cyan]")

        if model_info['is_multilingual'] or model_info['language'] == primary_lang:
            self.console.print("[green]✓ Compatible[/green]")
        else:
            self.console.print(f"[yellow]⚠ Mismatch[/yellow]")
            if not Confirm.ask("Continue?", default=False):
                return None

        return {'primary_language': primary_lang}

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

    def _configure_annotation_options(self) -> Dict[str, Any]:
        """Configure annotation options"""
        self.console.print("\n[bold cyan]Step 6/8: Annotation Options[/bold cyan]\n")

        return {
            'parallel': Confirm.ask("[cyan]Enable parallel processing?[/cyan]", default=True),
            'device_mode': 'both',
            'batch_size_cpu': 32,
            'batch_size_gpu': 64
        }

    def _configure_export_options(self) -> Dict[str, Any]:
        """Configure export"""
        self.console.print("\n[bold cyan]Step 7/8: Export Options[/bold cyan]\n")

        default_output = f"bert_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return {
            'output_file': Prompt.ask("[cyan]Output file[/cyan]", default=default_output),
            'export_to_tools': False
        }

    def _confirm_and_execute(self, model_info: Dict, data_source: Dict, df: pd.DataFrame,
                            column_mapping: Dict, language_info: Dict, correction_config: Dict,
                            annotation_config: Dict, export_config: Dict) -> bool:
        """Execute annotation"""
        self.console.print("\n[bold cyan]Step 8/8: Execute[/bold cyan]\n")

        summary = Table(title="Summary", box=box.ROUNDED)
        summary.add_column("Parameter", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Model", model_info['name'])
        summary.add_row("Rows", f"{len(df):,}")
        summary.add_row("Text Column", column_mapping['text'])
        summary.add_row("Output", export_config['output_file'])

        self.console.print(summary)

        if not Confirm.ask("\n[cyan]Start?[/cyan]", default=True):
            return False

        try:
            texts = df[column_mapping['text']].fillna("").tolist()

            if correction_config['enabled']:
                texts = self._apply_corrections(texts, correction_config)

            self.console.print("\n[cyan]Running inference...[/cyan]")

            predictions = parallel_predict(
                texts=texts,
                model_path=str(model_info['path']),
                lang=model_info['language'],
                parallel=annotation_config['parallel'],
                device_mode=annotation_config['device_mode'],
                batch_size_cpu=annotation_config['batch_size_cpu'],
                batch_size_gpu=annotation_config['batch_size_gpu'],
                show_progress=True
            )

            predicted_labels = np.argmax(predictions, axis=1)
            prediction_probs = np.max(predictions, axis=1)

            df_result = df.copy()
            df_result['predicted_label'] = predicted_labels
            df_result['confidence'] = prediction_probs

            output_path = Path(export_config['output_file'])
            df_result.to_csv(output_path, index=False)

            self.console.print(f"\n[green]✓ Saved to {output_path}[/green]")
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

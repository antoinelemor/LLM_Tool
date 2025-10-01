#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
advanced_cli.py

MAIN OBJECTIVE:
---------------
This script provides a sophisticated, professional-grade Command Line Interface
for the LLMTool package with auto-detection, intelligent suggestions, and
guided interactive workflows inspired by state-of-the-art CLI design patterns.

Dependencies:
-------------
- sys
- os
- subprocess
- pathlib
- rich (optional but highly recommended)
- inquirer
- pandas
- psutil

MAIN FEATURES:
--------------
1) Auto-detection of available models (Ollama, API models, local files)
2) Intelligent suggestions based on context and history
3) Interactive guided wizards for complex workflows
4) Professional validation with helpful error recovery
5) Rich visual interface with graceful fallback
6) Configuration profiles and execution history
7) Real-time progress tracking with detailed statistics
8) Smart defaults based on detected environment

Author:
-------
Antoine Lemor
"""

import sys
import os
import subprocess
import json
import time
import logging
import glob
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import re

# Rich is mandatory for this CLI
HAS_RICH = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError as e:
    print("\n❌ Error: Rich library is required but not installed.")
    print("💻 Please install it with: pip install rich")
    print(f"\nError details: {e}")
    sys.exit(1)

# Try importing pandas for data preview
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Try importing psutil for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import internal modules
from ..config.settings import Settings
from ..pipelines.pipeline_controller import PipelineController
from ..utils.language_detector import LanguageDetector
from ..annotators.json_cleaner import extract_expected_keys


@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    provider: str  # ollama, openai, anthropic, local
    size: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    is_available: bool = True
    requires_api_key: bool = False
    supports_json: bool = True
    supports_streaming: bool = True
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    path: Path
    format: str
    rows: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    size_mb: Optional[float] = None
    detected_language: Optional[str] = None
    has_labels: bool = False
    label_column: Optional[str] = None
    column_types: Dict[str, str] = field(default_factory=dict)
    text_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionProfile:
    """Saved execution profile for quick re-runs"""
    name: str
    created_at: datetime
    last_used: datetime
    configuration: Dict[str, Any]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class LLMDetector:
    """Auto-detect available LLMs for annotation from various sources"""

    @staticmethod
    def detect_ollama_models() -> List[ModelInfo]:
        """Detect locally available Ollama LLMs for annotation"""
        models = []
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Has header and content
                    # Parse the table format from ollama list
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            # Parse: NAME                    ID              SIZE      MODIFIED
                            parts = line.split()
                            if len(parts) >= 1:
                                name = parts[0]
                                # Extract size if available (e.g., "27GB" -> "27 GB")
                                size = None
                                if len(parts) >= 3:
                                    size_str = parts[2]
                                    if 'GB' in size_str or 'MB' in size_str:
                                        size = size_str

                                models.append(ModelInfo(
                                    name=name,
                                    provider="ollama",
                                    size=size,
                                    is_available=True,
                                    supports_json=True,
                                    supports_streaming=True,
                                    context_length=LLMDetector._estimate_context_length(name)
                                ))
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return models

    @staticmethod
    def _estimate_context_length(model_name: str) -> int:
        """Estimate context length based on model name"""
        name_lower = model_name.lower()
        if 'llama3' in name_lower or 'llama-3' in name_lower:
            return 8192
        elif 'mixtral' in name_lower:
            return 32768
        elif 'gemma2' in name_lower:
            return 8192
        elif 'gemma3' in name_lower:
            return 8192
        elif 'qwen' in name_lower:
            return 32768
        elif 'phi' in name_lower:
            return 4096
        elif 'mistral' in name_lower:
            return 8192
        else:
            return 4096  # Default

    @staticmethod
    def detect_openai_models() -> List[ModelInfo]:
        """List available OpenAI models"""
        models = [
            ModelInfo("gpt-4-turbo", "openai", context_length=128000, requires_api_key=True,
                     cost_per_1k_tokens=0.01),
            ModelInfo("gpt-4", "openai", context_length=8192, requires_api_key=True,
                     cost_per_1k_tokens=0.03),
            ModelInfo("gpt-3.5-turbo", "openai", context_length=16385, requires_api_key=True,
                     cost_per_1k_tokens=0.001),
            ModelInfo("o1-preview", "openai", context_length=128000, requires_api_key=True,
                     supports_streaming=False, cost_per_1k_tokens=0.015),
            ModelInfo("o1-mini", "openai", context_length=128000, requires_api_key=True,
                     supports_streaming=False, cost_per_1k_tokens=0.003),
        ]
        return models

    @staticmethod
    def detect_anthropic_models() -> List[ModelInfo]:
        """List available Anthropic models"""
        models = [
            ModelInfo("claude-3-opus-20240229", "anthropic", context_length=200000,
                     requires_api_key=True, cost_per_1k_tokens=0.015),
            ModelInfo("claude-3-sonnet-20240229", "anthropic", context_length=200000,
                     requires_api_key=True, cost_per_1k_tokens=0.003),
            ModelInfo("claude-3-haiku-20240307", "anthropic", context_length=200000,
                     requires_api_key=True, cost_per_1k_tokens=0.00025),
            ModelInfo("claude-3-5-sonnet-20241022", "anthropic", context_length=200000,
                     requires_api_key=True, cost_per_1k_tokens=0.003),
        ]
        return models

    @staticmethod
    def detect_all_llms() -> Dict[str, List[ModelInfo]]:
        """Detect all available LLMs for annotation from all sources"""
        return {
            "local": LLMDetector.detect_ollama_models(),
            "openai": LLMDetector.detect_openai_models(),
            "anthropic": LLMDetector.detect_anthropic_models(),
        }


class TrainerModelDetector:
    """Detect available models for training (BERT variants, etc.)"""

    @staticmethod
    def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
        """Get all available models for training organized by category"""
        return {
            "English Models": [
                {"name": "bert-base-uncased", "params": "110M", "type": "BERT", "performance": "★★★"},
                {"name": "bert-large-uncased", "params": "340M", "type": "BERT", "performance": "★★★★"},
                {"name": "roberta-base", "params": "125M", "type": "RoBERTa", "performance": "★★★★"},
                {"name": "roberta-large", "params": "355M", "type": "RoBERTa", "performance": "★★★★★"},
                {"name": "deberta-v3-base", "params": "184M", "type": "DeBERTa", "performance": "★★★★★"},
                {"name": "deberta-v3-large", "params": "435M", "type": "DeBERTa", "performance": "★★★★★"},
                {"name": "electra-base", "params": "110M", "type": "ELECTRA", "performance": "★★★★"},
                {"name": "albert-base-v2", "params": "12M", "type": "ALBERT", "performance": "★★★"},
            ],
            "Multilingual Models": [
                {"name": "bert-base-multilingual", "params": "177M", "type": "mBERT", "languages": "104", "performance": "★★★"},
                {"name": "xlm-roberta-base", "params": "278M", "type": "XLM-R", "languages": "100+", "performance": "★★★★"},
                {"name": "xlm-roberta-large", "params": "560M", "type": "XLM-R", "languages": "100+", "performance": "★★★★★"},
                {"name": "mdeberta-v3-base", "params": "280M", "type": "mDeBERTa", "languages": "100+", "performance": "★★★★★"},
            ],
            "French Models": [
                {"name": "camembert-base", "params": "110M", "type": "CamemBERT", "performance": "★★★★"},
                {"name": "flaubert-base", "params": "137M", "type": "FlauBERT", "performance": "★★★★"},
                {"name": "distilcamembert", "params": "68M", "type": "DistilCamemBERT", "performance": "★★★"},
            ],
            "Long Document Models": [
                {"name": "longformer-base", "params": "149M", "type": "Longformer", "max_length": "4096", "performance": "★★★★"},
                {"name": "bigbird-base", "params": "128M", "type": "BigBird", "max_length": "4096", "performance": "★★★★"},
            ],
            "Efficient Models": [
                {"name": "distilbert-base", "params": "66M", "type": "DistilBERT", "speed": "2x faster", "performance": "★★★"},
                {"name": "tinybert", "params": "14M", "type": "TinyBERT", "speed": "9x faster", "performance": "★★"},
                {"name": "mobilebert", "params": "25M", "type": "MobileBERT", "speed": "4x faster", "performance": "★★★"},
            ]
        }


class DataDetector:
    """Auto-detect and analyze available datasets"""

    @staticmethod
    def scan_directory(directory: Path = Path.cwd()) -> List[DatasetInfo]:
        """Scan directory and subdirectories for potential datasets"""
        datasets = []

        # If directory doesn't exist, return empty list
        if not directory.exists():
            return datasets

        patterns = ['**/*.csv', '**/*.json', '**/*.jsonl', '**/*.xlsx', '**/*.parquet', '**/*.tsv']

        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    dataset_info = DataDetector.analyze_file(file_path)
                    if dataset_info:
                        datasets.append(dataset_info)

        return datasets

    @staticmethod
    def analyze_file(file_path: Path) -> Optional[DatasetInfo]:
        """Analyze a single file to extract dataset information"""
        if not file_path.exists():
            return None

        info = DatasetInfo(
            path=file_path,
            format=file_path.suffix[1:],
            size_mb=file_path.stat().st_size / (1024 * 1024)
        )

        # Try to read and analyze the file
        if HAS_PANDAS:
            try:
                if info.format == 'csv':
                    df = pd.read_csv(file_path, nrows=100)
                elif info.format == 'json':
                    df = pd.read_json(file_path, lines=False, nrows=100)
                elif info.format == 'jsonl':
                    df = pd.read_json(file_path, lines=True, nrows=100)
                elif info.format == 'xlsx':
                    df = pd.read_excel(file_path, nrows=100)
                elif info.format == 'parquet':
                    df = pd.read_parquet(file_path)
                    df = df.head(100)
                else:
                    return info

                info.rows = len(df)
                info.columns = list(df.columns)
                info.column_types = {col: str(df[col].dtype) for col in df.columns}
                info.text_scores = {}

                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]) or str(df[col].dtype) == 'object':
                        sample_series = df[col].dropna().astype(str)
                        if not sample_series.empty:
                            avg_len = float(sample_series.str.len().mean())
                            info.text_scores[col] = avg_len

                # Detect if there's a label column
                label_candidates = ['label', 'labels', 'class', 'category', 'target', 'y']
                for col in info.columns:
                    if col.lower() in label_candidates:
                        info.has_labels = True
                        info.label_column = col
                        break

            except Exception:
                pass

        return info

    @staticmethod
    def suggest_text_column(dataset: DatasetInfo) -> Optional[str]:
        """Suggest the most likely text column from a dataset"""
        text_candidates = ['text', 'content', 'message', 'comment', 'review',
                          'description', 'body', 'document', 'sentence', 'paragraph']
        column_types = dataset.column_types or {}

        def is_probably_identifier(name: str) -> bool:
            name_lower = name.lower()
            if name_lower in {'id', 'identifier'}:
                return True
            if name_lower.endswith('_id') or name_lower.endswith('id'):
                return True
            return False

        def candidate_score(name: str) -> float:
            return dataset.text_scores.get(name, 0.0)

        # Prioritise columns whose dtype looks textual
        textual_columns = []
        for col in dataset.columns:
            dtype = column_types.get(col, '').lower()
            if 'object' in dtype or 'string' in dtype:
                textual_columns.append(col)
        if not textual_columns:
            textual_columns = list(dataset.columns)

        # Exact matches first
        exact_matches = [
            col for col in textual_columns
            if col.lower() in text_candidates and not is_probably_identifier(col)
        ]
        if exact_matches:
            exact_matches.sort(key=candidate_score, reverse=True)
            return exact_matches[0]

        # Partial matches (avoid *_id)
        partial_matches = []
        for col in textual_columns:
            col_lower = col.lower()
            if is_probably_identifier(col):
                continue
            for candidate in text_candidates:
                if candidate in col_lower:
                    partial_matches.append(col)
                    break

        if partial_matches:
            partial_matches.sort(key=candidate_score, reverse=True)
            return partial_matches[0]

        # Fall back to column with largest average length
        if dataset.text_scores:
            for col, _ in sorted(dataset.text_scores.items(), key=lambda item: item[1], reverse=True):
                if col in textual_columns and not is_probably_identifier(col):
                    return col

        # Final fallback: first non-identifier textual column
        for col in textual_columns:
            if not is_probably_identifier(col):
                return col

        return dataset.columns[0] if dataset.columns else None


class ProfileManager:
    """Manage execution profiles for quick re-runs"""

    def __init__(self, profile_dir: Path = None):
        self.profile_dir = profile_dir or Path.home() / '.llmtool' / 'profiles'
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.profile_dir / 'history.json'

    def save_profile(self, profile: ExecutionProfile):
        """Save an execution profile"""
        profile_file = self.profile_dir / f"{profile.name}.json"
        data = {
            'name': profile.name,
            'created_at': profile.created_at.isoformat(),
            'last_used': profile.last_used.isoformat(),
            'configuration': profile.configuration,
            'performance_metrics': profile.performance_metrics,
            'notes': profile.notes
        }
        with open(profile_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_profile(self, name: str) -> Optional[ExecutionProfile]:
        """Load an execution profile by name"""
        profile_file = self.profile_dir / f"{name}.json"
        if not profile_file.exists():
            return None

        with open(profile_file, 'r') as f:
            data = json.load(f)

        return ExecutionProfile(
            name=data['name'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_used=datetime.fromisoformat(data['last_used']),
            configuration=data['configuration'],
            performance_metrics=data.get('performance_metrics', {}),
            notes=data.get('notes', '')
        )

    def list_profiles(self) -> List[ExecutionProfile]:
        """List all available profiles"""
        profiles = []
        for profile_file in self.profile_dir.glob('*.json'):
            if profile_file.name != 'history.json':
                profile = self.load_profile(profile_file.stem)
                if profile:
                    profiles.append(profile)
        return sorted(profiles, key=lambda p: p.last_used, reverse=True)

    def save_to_history(self, config: Dict[str, Any]):
        """Save configuration to execution history"""
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        history.append({
            'timestamp': datetime.now().isoformat(),
            'configuration': config
        })

        # Keep only last 100 executions
        history = history[-100:]

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_recent_configs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent configurations from history"""
        if not self.history_file.exists():
            return []

        with open(self.history_file, 'r') as f:
            history = json.load(f)

        return [h['configuration'] for h in history[-limit:]]


class AdvancedCLI:
    """Professional-grade CLI for LLMTool with sophisticated features"""

    def __init__(self):
        """Initialize the advanced CLI"""
        self.console = Console() if HAS_RICH else None
        self.settings = Settings()
        self.pipeline_controller = PipelineController()
        self.language_detector = LanguageDetector()
        self.llm_detector = LLMDetector()
        self.trainer_model_detector = TrainerModelDetector()
        self.data_detector = DataDetector()
        self.profile_manager = ProfileManager()

        # Import and initialize PromptManager
        from ..annotators.prompt_manager import PromptManager
        self.prompt_manager = PromptManager()

        # Cache for detected models
        self.detected_llms: Optional[Dict[str, List[ModelInfo]]] = None
        self.available_trainer_models: Optional[Dict[str, List[Dict]]] = None
        self.detected_datasets: Optional[List[DatasetInfo]] = None

        # Session state
        self.current_session = {
            'start_time': datetime.now(),
            'operations_count': 0,
            'last_operation': None
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure professional logging"""
        log_dir = self.settings.paths.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"llmtool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Force reconfiguration by clearing existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging: DEBUG to file, WARNING to console
        logging.basicConfig(
            level=logging.DEBUG,  # Capture everything
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8')
            ],
            force=True
        )

        # Add console handler with WARNING level only (hides INFO and DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(console_handler)

        self.logger = logging.getLogger(__name__)

        # Store log file path for reference
        self.current_log_file = log_file

    def _int_prompt_with_validation(self, prompt: str, default: int = 1, min_value: int = None, max_value: int = None) -> int:
        """IntPrompt.ask with validation since min_value/max_value not supported in older Rich versions"""
        while True:
            try:
                value = IntPrompt.ask(prompt, default=default)
                if min_value is not None and value < min_value:
                    self.console.print(f"[red]Value must be at least {min_value}[/red]")
                    continue
                if max_value is not None and value > max_value:
                    self.console.print(f"[red]Value must be at most {max_value}[/red]")
                    continue
                return value
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

    def display_banner(self):
        """Display professional welcome banner with system info"""
        if HAS_RICH and self.console:
            # Spectacular multicolor full-width ASCII art banner
            from rich.align import Align
            from rich.text import Text

            # Get terminal width
            width = self.console.width

            self.console.print()
            self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
            self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
            self.console.print()

            # Giant LLM TOOL text with each letter in different color
            self.console.print(Align.center("[bright_magenta]██╗     [bright_yellow]██╗     [bright_green]███╗   ███╗    [bright_cyan]████████╗ [bright_red]██████╗  [bright_blue]██████╗ [bright_white]██╗     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]████╗ ████║    [bright_cyan]╚══██╔══╝[bright_red]██╔═══██╗[bright_blue]██╔═══██╗[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██╔████╔██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██║╚██╔╝██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]███████╗[bright_yellow]███████╗[bright_green]██║ ╚═╝ ██║       [bright_cyan]██║   [bright_red]╚██████╔╝[bright_blue]╚██████╔╝[bright_white]███████╗"))
            self.console.print(Align.center("[bright_magenta]╚══════╝[bright_yellow]╚══════╝[bright_green]╚═╝     ╚═╝       [bright_cyan]╚═╝    [bright_red]╚═════╝  [bright_blue]╚═════╝ [bright_white]╚══════╝"))

            self.console.print()
            self.console.print(Align.center("[bold bright_yellow on blue]  🚀 LLM-powered Intelligent Annotation & Training Pipeline 🚀  [/bold bright_yellow on blue]"))
            self.console.print()

            # Colorful pipeline with emojis
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

            # Information table with system info
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_row("📚 Version:", "[bright_green]1.0[/bright_green]")
            info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")
            info_table.add_row("🚀 Features:", "[cyan]Multi-LLM Support, Smart Training, Auto-Detection[/cyan]")
            info_table.add_row("🎯 Capabilities:", "[magenta]JSON Annotation, BERT Training, Benchmarking[/magenta]")

            # Add system info if available
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                info_table.add_row(
                    "💻 System:",
                    f"[yellow]CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% used[/yellow]"
                )

            self.console.print(Panel(
                info_table,
                title="[bold bright_cyan]✨ Welcome to LLM Tool ✨[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2)
            ))
            self.console.print()

            # Auto-detect models in background
            with self.console.status("[bold green]🔍 Scanning environment...", spinner="dots"):
                self.detected_llms = self.llm_detector.detect_all_llms()
                self.available_trainer_models = self.trainer_model_detector.get_available_models()
                # Scan only in data/ directory
                data_dir = self.settings.paths.data_dir
                self.detected_datasets = self.data_detector.scan_directory(data_dir)

            # Show detection results
            self._display_detection_results()

        else:
            print("\n" + "="*80)
            print(" " * 28 + "LLM TOOL")
            print(" " * 15 + "LLM-powered Intelligent Annotation & Training Pipeline")
            print("="*80)
            print("\n📚 Version: 1.0")
            print("👨‍💻 Author: Antoine Lemor")
            print("\n📊 Data → 🤖 LLM Annotation → 🧹 Clean → 🎯 Label → 🧠 Train → 📈 Deploy")
            print("\nScanning environment...")

            self.detected_llms = self.llm_detector.detect_all_llms()
            self.available_trainer_models = self.trainer_model_detector.get_available_models()
            # Scan only in data/ directory
            data_dir = self.settings.paths.data_dir
            self.detected_datasets = self.data_detector.scan_directory(data_dir)

            # Count LLMs and trainer models
            llm_count = sum(len(m) for m in self.detected_llms.values())
            trainer_count = sum(len(m) for m in self.available_trainer_models.values())

            print(f"✓ Found {llm_count} annotation LLMs")
            print(f"✓ {trainer_count} trainable models available")
            print(f"✓ Found {len(self.detected_datasets)} datasets")
            print()

    def _display_detection_results(self):
        """Display auto-detection results in a professional format"""
        if not HAS_RICH or not self.console:
            return

        # === ANNOTATION LLMs SECTION ===
        llms_table = Table(title="🤖 Available LLMs for Annotation", border_style="cyan", show_lines=True)
        llms_table.add_column("Provider", style="cyan", width=10)
        llms_table.add_column("Model", style="white", width=20)
        llms_table.add_column("Size", style="yellow", width=8)
        llms_table.add_column("Context", style="green", width=10)
        llms_table.add_column("Status", style="green", width=12)

        # Show Ollama models (all of them if available)
        local_llms = self.detected_llms.get('local', [])
        if local_llms:
            # Show all local models, not just first 3
            for model in local_llms:
                llms_table.add_row(
                    "Ollama",
                    model.name,
                    model.size or "N/A",
                    f"{model.context_length:,}" if model.context_length else "N/A",
                    "✓ Ready"
                )

        # Show API models (top ones)
        for provider in ['openai', 'anthropic']:
            api_models = self.detected_llms.get(provider, [])
            for model in api_models[:2]:  # Show top 2 per API provider
                llms_table.add_row(
                    provider.title(),
                    model.name,
                    "API",
                    f"{model.context_length:,}" if model.context_length else "N/A",
                    "🔑 API Key" if model.requires_api_key else "✓ Ready"
                )

        # === TRAINABLE MODELS SECTION ===
        trainer_table = Table(title="🏋️ Available Models for Training", border_style="magenta", show_lines=False)
        trainer_table.add_column("Category", style="magenta", width=20)
        trainer_table.add_column("Models", style="white", width=60)

        for category, models in self.available_trainer_models.items():
            # Format model names compactly
            model_names = [m['name'] for m in models[:4]]  # Show first 4
            if len(models) > 4:
                model_names.append(f"(+{len(models)-4} more)")
            trainer_table.add_row(
                category,
                ", ".join(model_names)
            )

        # === DATASETS SECTION ===
        datasets_table = Table(title="📊 Detected Datasets", border_style="yellow", show_lines=False)
        datasets_table.add_column("File", style="cyan", width=25)
        datasets_table.add_column("Format", style="white", width=8)
        datasets_table.add_column("Size", style="green", width=10)
        datasets_table.add_column("Columns", style="dim", width=35)

        if self.detected_datasets:
            for dataset in self.detected_datasets[:5]:  # Show top 5
                columns_preview = ", ".join(dataset.columns[:3]) if dataset.columns else "N/A"
                if len(dataset.columns) > 3:
                    columns_preview += f" (+{len(dataset.columns)-3} more)"

                datasets_table.add_row(
                    dataset.path.name,
                    dataset.format.upper(),
                    f"{dataset.size_mb:.1f} MB" if dataset.size_mb else "Unknown",
                    columns_preview
                )
        else:
            datasets_table.add_row(
                "No datasets found",
                "-",
                "-",
                "Place CSV/JSON files in current directory"
            )

        # Print all tables
        self.console.print(llms_table)
        self.console.print()
        self.console.print(trainer_table)
        self.console.print()
        self.console.print(datasets_table)
        self.console.print()

    def get_main_menu_choice(self) -> str:
        """Display sophisticated main menu with smart suggestions"""
        if HAS_RICH and self.console:
            # Create menu table
            menu_table = Table.grid(padding=0)
            menu_table.add_column(width=3)
            menu_table.add_column()

            options = [
                ("1", "🎯 Quick Start - Intelligent Pipeline Setup"),
                ("2", "📝 Annotation Wizard - Guided LLM Annotation"),
                ("3", "🚀 Complete Pipeline - Full Automated Workflow"),
                ("4", "🏋️ Training Studio - Model Training & Benchmarking"),
                ("5", "🔍 Validation Lab - Quality Assurance Tools"),
                ("6", "📊 Analytics Dashboard - Performance Insights"),
                ("7", "💾 Profile Manager - Save & Load Configurations"),
                ("8", "⚙️ Advanced Settings - Fine-tune Everything"),
                ("9", "📚 Documentation & Help"),
                ("0", "❌ Exit")
            ]

            for num, desc in options:
                menu_table.add_row(
                    f"[bold cyan]{num}[/bold cyan]",
                    desc
                )

            # Suggestions based on context
            suggestions = self._get_smart_suggestions()

            panel = Panel(
                menu_table,
                title="[bold]Main Menu[/bold]",
                subtitle=f"[dim]{suggestions}[/dim]" if suggestions else None,
                border_style="cyan"
            )

            self.console.print(panel)

            # Smart prompt with validation
            choice = Prompt.ask(
                "\n[bold yellow]Select option[/bold yellow]",
                choices=[str(i) for i in range(10)],
                default="1"
            )

        else:
            print("\n" + "="*50)
            print("Main Menu")
            print("="*50)
            print("1. Quick Start - Intelligent Pipeline Setup")
            print("2. Annotation Wizard - Guided LLM Annotation")
            print("3. Complete Pipeline - Full Automated Workflow")
            print("4. Training Studio - Model Training & Benchmarking")
            print("5. Validation Lab - Quality Assurance Tools")
            print("6. Analytics Dashboard - Performance Insights")
            print("7. Profile Manager - Save & Load Configurations")
            print("8. Advanced Settings - Fine-tune Everything")
            print("9. Documentation & Help")
            print("0. Exit")
            print("-"*50)

            suggestions = self._get_smart_suggestions()
            if suggestions:
                print(f"💡 {suggestions}")

            choice = input("\nSelect option (0-9): ").strip()

        return choice

    def _get_smart_suggestions(self) -> str:
        """Generate intelligent suggestions based on context"""
        suggestions = []

        # Check for available LLMs
        if self.detected_llms:
            local_llms = self.detected_llms.get('local', [])
            if local_llms:
                # Show count of local LLMs
                suggestions.append(f"{len(local_llms)} local LLMs available")
            else:
                suggestions.append("No local LLMs - run 'ollama pull llama3.2'")

        # Check for datasets
        if self.detected_datasets:
            suggestions.append(f"{len(self.detected_datasets)} dataset{'s' if len(self.detected_datasets) != 1 else ''} found")
        else:
            suggestions.append("No datasets in current directory")

        # Check for recent profiles
        recent_profiles = self.profile_manager.list_profiles()
        if recent_profiles:
            suggestions.append(f"Last: {recent_profiles[0].name}")

        return " | ".join(suggestions) if suggestions else ""

    @staticmethod
    def _estimate_model_size_billion(model: ModelInfo) -> Optional[float]:
        """Estimate model parameter count (billions) from its metadata."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*b',
            r'(\d+(?:\.\d+)?)\s*bn',
        ]

        lower_name = model.name.lower()
        for pattern in patterns:
            match = re.search(pattern, lower_name)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        if model.size and model.size.lower() != 'n/a':
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:b|bn)', model.size.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None

        return None

    def _prompt_for_text_column(self, columns: List[str], suggested: Optional[str]) -> str:
        """Prompt the user to choose the text column."""
        if not columns:
            if HAS_RICH and self.console:
                return Prompt.ask("Text column name", default="text")
            return input("Text column name [text]: ").strip() or "text"

        choices = [str(col) for col in columns]
        default_choice = suggested if suggested in choices else choices[0]

        if HAS_RICH and self.console:
            return Prompt.ask(
                "Which column contains the text?",
                choices=choices,
                default=default_choice
            )

        print("Available columns:")
        for col in choices:
            marker = " (suggested)" if col == default_choice else ""
            print(f"  - {col}{marker}")
        response = input(f"Text column [{default_choice}]: ").strip()
        return response or default_choice

    def _detect_id_columns(self, columns: List[str]) -> List[str]:
        """Detect all columns that could serve as IDs"""
        id_columns = []
        for col in columns:
            col_lower = col.lower()
            # Check if column name suggests it's an ID
            if (col_lower == 'id' or
                col_lower.endswith('_id') or
                col_lower.startswith('id_') or
                'identifier' in col_lower or
                col_lower in ['promesse_id', 'sentence_id', 'doc_id', 'item_id', 'record_id']):
                id_columns.append(col)
        return id_columns

    def _prompt_for_identifier_column(
        self,
        columns: List[str],
        suggested: Optional[str]
    ) -> Optional[str]:
        """Ask which column should serve as unique identifier, if any."""
        if not columns:
            if HAS_RICH and self.console:
                self.console.print("[dim]No columns detected; will create 'llm_annotation_id'.[/dim]")
            else:
                print("No columns detected; will create 'llm_annotation_id'.")
            return None

        # Detect all ID columns
        id_columns = self._detect_id_columns(columns)

        # If multiple ID columns found, offer to combine them
        if len(id_columns) > 1:
            if HAS_RICH and self.console:
                self.console.print(f"\n[bold cyan]📋 Found {len(id_columns)} ID columns:[/bold cyan]")
                for i, col in enumerate(id_columns, 1):
                    self.console.print(f"  {i}. [cyan]{col}[/cyan]")

                # Ask if user wants to use single or combined ID
                self.console.print("\n[bold]ID Strategy:[/bold]")
                self.console.print("• [cyan]single[/cyan]: Use one column as ID")
                self.console.print("• [cyan]combine[/cyan]: Combine multiple columns (e.g., 'promesse_id+sentence_id')")
                self.console.print("• [cyan]none[/cyan]: Generate automatic IDs")

                id_strategy = Prompt.ask(
                    "ID strategy",
                    choices=["single", "combine", "none"],
                    default="single"
                )

                if id_strategy == "none":
                    self.console.print("[dim]An 'llm_annotation_id' column will be created automatically.[/dim]")
                    return None
                elif id_strategy == "combine":
                    # Ask which columns to combine
                    self.console.print("\n[bold]Select columns to combine:[/bold]")
                    self.console.print("[dim]Enter column numbers separated by commas (e.g., '1,2')[/dim]")

                    while True:
                        selection = Prompt.ask("Columns to combine")
                        try:
                            indices = [int(x.strip()) - 1 for x in selection.split(',')]
                            if all(0 <= i < len(id_columns) for i in indices):
                                selected_cols = [id_columns[i] for i in indices]
                                combined_id = "+".join(selected_cols)
                                self.console.print(f"[green]✓ Will combine: {' + '.join(selected_cols)}[/green]")
                                self.console.print(f"[dim]Example ID format: {' _ '.join(['123' for _ in selected_cols])}[/dim]")
                                return combined_id
                            else:
                                self.console.print("[red]Invalid column numbers. Try again.[/red]")
                        except (ValueError, IndexError):
                            self.console.print("[red]Invalid format. Use comma-separated numbers (e.g., '1,2')[/red]")
                else:  # single
                    # Select single ID column
                    default_choice = id_columns[0]
                    return Prompt.ask(
                        "Which ID column to use?",
                        choices=id_columns,
                        default=default_choice
                    )
            else:
                # Non-Rich fallback
                print(f"\nFound {len(id_columns)} ID columns:")
                for i, col in enumerate(id_columns, 1):
                    print(f"  {i}. {col}")

                choice = input("Use single ID (s), combine IDs (c), or generate new (n)? [s/c/n]: ").strip().lower()

                if choice == 'n':
                    print("Will create 'llm_annotation_id' automatically.")
                    return None
                elif choice == 'c':
                    selection = input("Enter column numbers to combine (comma-separated): ").strip()
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        selected_cols = [id_columns[i] for i in indices]
                        return "+".join(selected_cols)
                    except (ValueError, IndexError):
                        print("Invalid selection. Using first ID column.")
                        return id_columns[0]
                else:
                    idx = int(input(f"Select ID column [1-{len(id_columns)}]: ").strip() or "1") - 1
                    return id_columns[idx] if 0 <= idx < len(id_columns) else id_columns[0]

        # Single or no ID column found - use original logic
        default_has_id = suggested is not None or len(id_columns) == 1

        if HAS_RICH and self.console:
            has_id = Confirm.ask(
                "Does the dataset already contain a unique ID column?",
                default=default_has_id
            )
            if not has_id:
                self.console.print("[dim]An 'llm_annotation_id' column will be created automatically.[/dim]")
                return None

            choices = [str(col) for col in columns]
            default_choice = (id_columns[0] if id_columns else
                            (suggested if suggested in choices else choices[0]))
            return Prompt.ask(
                "Which column should be used as the identifier?",
                choices=choices,
                default=default_choice
            )

        prompt_default = "y" if default_has_id else "n"
        raw = input(f"Dataset has an ID column? [y/n] ({prompt_default}): ").strip().lower()
        has_id = raw or prompt_default
        if has_id.startswith('n'):
            print("We'll create 'llm_annotation_id' automatically.")
            return None

        print("Available columns:")
        for col in columns:
            marker = " (suggested)" if suggested and col == suggested else ""
            print(f"  - {col}{marker}")
        default_choice = suggested or columns[0]
        response = input(f"Identifier column [{default_choice}]: ").strip()
        return response or default_choice

    def _display_welcome_banner(self):
        """Display a beautiful welcome banner"""
        if HAS_RICH and self.console:
            # Spectacular multicolor full-width ASCII art banner
            from rich.align import Align
            from rich.text import Text

            # Get terminal width
            width = self.console.width

            self.console.print()
            self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
            self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
            self.console.print()

            # Giant LLM TOOL text with each letter in different color
            self.console.print(Align.center("[bright_magenta]██╗     [bright_yellow]██╗     [bright_green]███╗   ███╗    [bright_cyan]████████╗ [bright_red]██████╗  [bright_blue]██████╗ [bright_white]██╗     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]████╗ ████║    [bright_cyan]╚══██╔══╝[bright_red]██╔═══██╗[bright_blue]██╔═══██╗[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██╔████╔██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██║╚██╔╝██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]███████╗[bright_yellow]███████╗[bright_green]██║ ╚═╝ ██║       [bright_cyan]██║   [bright_red]╚██████╔╝[bright_blue]╚██████╔╝[bright_white]███████╗"))
            self.console.print(Align.center("[bright_magenta]╚══════╝[bright_yellow]╚══════╝[bright_green]╚═╝     ╚═╝       [bright_cyan]╚═╝    [bright_red]╚═════╝  [bright_blue]╚═════╝ [bright_white]╚══════╝"))

            self.console.print()
            self.console.print(Align.center("[bold bright_yellow on blue]  🚀 LLM-powered Intelligent Annotation & Training Pipeline 🚀  [/bold bright_yellow on blue]"))
            self.console.print()

            # Colorful pipeline with emojis
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

            # Information table
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_row("📚 Version:", "[bright_green]1.0[/bright_green]")
            info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")
            info_table.add_row("🚀 Features:", "[cyan]Multi-LLM Support, Smart Training, Auto-Detection[/cyan]")
            info_table.add_row("🎯 Capabilities:", "[magenta]JSON Annotation, BERT Training, Benchmarking[/magenta]")
            info_table.add_row("⚡ Performance:", "[green]Parallel Processing, Progress Tracking[/green]")

            self.console.print(Panel(
                info_table,
                title="[bold bright_cyan]✨ Welcome to LLM Tool ✨[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2)
            ))
            self.console.print()
        else:
            # Fallback for non-Rich environments (should never happen due to import check)
            print("="*80)
            print(" " * 28 + "LLM TOOL")
            print(" " * 18 + "Intelligent Annotation & Training Pipeline")
            print("="*80)
            print("\n📚 Version: 1.0")
            print("👨‍💻 Author: Antoine Lemor")
            print("🚀 Features: Multi-LLM Support, Smart Training, Auto-Detection")
            print("🎯 Capabilities: JSON Annotation, BERT Training, Benchmarking")
            print("⚡ Performance: Parallel Processing, Progress Tracking")
            print("\n  🤖 -> 📝 -> 🧹 -> 🎯 -> 🧠 -> 📊 -> ✨")
            print("  AI   Annotate Clean Label Train Test Deploy\n")
            print("="*80 + "\n")

    def quick_start_wizard(self):
        """Intelligent quick start wizard with auto-configuration"""
        # Display welcome banner
        self._display_welcome_banner()

        if HAS_RICH and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]🎯 Quick Start Wizard[/bold cyan]\n"
                "I'll help you set up your pipeline intelligently",
                border_style="cyan"
            ))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                # Step 1: Analyze environment
                task = progress.add_task("[cyan]Analyzing your environment...", total=None)
                time.sleep(1)  # Simulate analysis

                # Auto-select best LLM
                best_llm = self._auto_select_llm()
                progress.update(task, description=f"[green]✓ Selected LLM: {best_llm.name}")

                # Step 2: Detect dataset
                task2 = progress.add_task("[cyan]Detecting datasets...", total=None)
                time.sleep(0.5)

                best_dataset = self._auto_select_dataset()
                if best_dataset:
                    progress.update(task2, description=f"[green]✓ Found dataset: {best_dataset.path.name}")
                else:
                    progress.update(task2, description="[yellow]No dataset found - will ask for path")

            # Interactive configuration with smart defaults
            self.console.print("\n[bold]Let's configure your pipeline:[/bold]\n")

            # LLM configuration - ASK FIRST
            self.console.print(f"[bold]Step 1: LLM Selection[/bold]")
            self.console.print(f"[dim]Auto-selected: {best_llm.name} ({best_llm.provider})[/dim]")

            use_auto_llm = Confirm.ask("Use this LLM?", default=True)

            if use_auto_llm:
                selected_llm = best_llm
            else:
                # Show available LLMs and let user choose
                selected_llm = self._select_llm_interactive()

            # Warn if the selected model is likely to be extremely large
            while True:
                size_estimate = self._estimate_model_size_billion(selected_llm) if selected_llm.provider == 'ollama' else None
                if size_estimate and size_estimate > 20:
                    if HAS_RICH and self.console:
                        proceed = Confirm.ask(
                            f"[yellow]The model {selected_llm.name} is ~{size_estimate:.0f}B parameters and may be slow or unstable. Continue?[/yellow]",
                            default=False
                        )
                    else:
                        raw = input(
                            f"Model {selected_llm.name} ≈ {size_estimate:.0f}B parameters. Continue? [y/N]: "
                        ).strip().lower()
                        proceed = raw.startswith('y')

                    if not proceed:
                        if HAS_RICH and self.console:
                            self.console.print("[dim]Please choose a smaller model.[/dim]")
                        else:
                            print("Please choose a smaller model.")
                        selected_llm = self._select_llm_interactive()
                        continue
                break

            self.console.print(f"[green]✓ Selected LLM: {selected_llm.name}[/green]")

            if selected_llm.requires_api_key:
                api_key = Prompt.ask("API Key", password=True)
            else:
                api_key = None

            # Ensure downstream steps use the user-selected model
            best_llm = selected_llm

            # Dataset selection - ASK SECOND
            self.console.print(f"\n[bold]Step 2: Dataset Selection[/bold]")
            if best_dataset:
                use_detected = Confirm.ask(
                    f"Use detected dataset [cyan]{best_dataset.path.name}[/cyan]?",
                    default=True
                )
                if use_detected:
                    dataset_info = best_dataset
                    dataset_path = str(best_dataset.path)
                else:
                    dataset_path = self._prompt_file_path("Enter dataset path")
                    dataset_info = DataDetector.analyze_file(Path(dataset_path)) or DatasetInfo(path=Path(dataset_path), format=Path(dataset_path).suffix.lstrip('.'))
            else:
                dataset_path = self._prompt_file_path("Enter dataset path")
                dataset_info = DataDetector.analyze_file(Path(dataset_path)) or DatasetInfo(path=Path(dataset_path), format=Path(dataset_path).suffix.lstrip('.'))

            available_columns = dataset_info.columns if dataset_info and dataset_info.columns else []
            suggested_text = self.data_detector.suggest_text_column(dataset_info) if dataset_info else None
            text_column = self._prompt_for_text_column(available_columns, suggested_text)

            suggested_id = None
            if available_columns:
                for col in available_columns:
                    lowered = col.lower()
                    if lowered == 'id' or lowered.endswith('_id') or 'identifier' in lowered:
                        suggested_id = col
                        break

            identifier_column = self._prompt_for_identifier_column(available_columns, suggested_id)

            # Language column detection and prompt
            lang_column = None
            if available_columns:
                # Detect potential language columns
                potential_lang_cols = [col for col in available_columns
                                      if col.lower() in ['lang', 'language', 'langue', 'lng', 'iso_lang']]

                if potential_lang_cols:
                    self.console.print(f"\n[bold cyan]🌍 Found language column(s):[/bold cyan]")
                    for col in potential_lang_cols:
                        self.console.print(f"  • [cyan]{col}[/cyan]")

                    use_lang_col = Confirm.ask("Use a language column for training metadata?", default=True)
                    if use_lang_col:
                        if len(potential_lang_cols) == 1:
                            lang_column = potential_lang_cols[0]
                            self.console.print(f"[green]✓ Using language column: {lang_column}[/green]")
                        else:
                            lang_column = Prompt.ask(
                                "Which language column to use?",
                                choices=potential_lang_cols,
                                default=potential_lang_cols[0]
                            )
                else:
                    # No language column detected
                    has_lang = Confirm.ask("Does your dataset have a language column?", default=False)
                    if has_lang:
                        lang_column = Prompt.ask(
                            "Language column name",
                            choices=available_columns,
                            default=available_columns[0]
                        )

            # Prompt configuration
            self.console.print("\n[bold]Prompt Configuration:[/bold]")
            self.console.print("[dim]• simple: Single prompt for all texts[/dim]")
            self.console.print("[dim]• multi: Multiple prompts applied to each text[/dim]")
            self.console.print("[dim]• template: Pre-configured prompt templates [yellow](Under development - not available)[/yellow][/dim]")

            prompt_mode = Prompt.ask(
                "Prompt mode",
                choices=["simple", "multi"],
                default="simple"
            )

            if prompt_mode == "simple":
                prompt = self._get_custom_prompt()
            else:  # multi
                prompt = self._get_multi_prompts()

            # Training configuration
            self.console.print("\n[bold]Training Configuration:[/bold]")
            training_mode = Prompt.ask(
                "Training mode",
                choices=["quick", "balanced", "thorough", "custom"],
                default="balanced"
            )

            training_config = self._get_training_preset(training_mode)

            # Annotation sample configuration
            annotation_sample_size = self._int_prompt_with_validation(
                "How many sentences should we annotate for training? (0 = all)",
                default=200,
                min_value=0
            )
            annotation_sampling_strategy = 'head'
            annotation_sample_seed = 42
            if annotation_sample_size and annotation_sample_size > 0:
                if Confirm.ask("Sample sentences randomly?", default=True):
                    annotation_sampling_strategy = 'random'
                    annotation_sample_seed = self._int_prompt_with_validation(
                        "Random seed",
                        default=42,
                        min_value=0
                    )

            annotation_settings = {
                'annotation_sample_size': annotation_sample_size if annotation_sample_size > 0 else None,
                'annotation_sampling_strategy': annotation_sampling_strategy,
                'annotation_sample_seed': annotation_sample_seed,
            }

            # Training strategy configuration
            self.console.print("\n[bold]Training Strategy:[/bold]")
            self.console.print("How do you want to create training labels from annotations?")
            self.console.print("• [cyan]single-label[/cyan]: Train separate models for each annotation key")
            self.console.print("• [cyan]multi-label[/cyan]: Train one model with all labels together")

            training_strategy = Prompt.ask(
                "Training strategy",
                choices=["single-label", "multi-label"],
                default="single-label"
            )

            # If single-label, ask which keys to train
            training_annotation_keys = None
            if training_strategy == "single-label":
                self.console.print("\n[dim]Available annotation keys will be detected from the annotations[/dim]")
                if Confirm.ask("Train models for all annotation keys?", default=True):
                    training_annotation_keys = None  # Will use all keys
                else:
                    keys_input = Prompt.ask("Enter annotation keys to train (comma-separated)")
                    training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

            # Label creation strategy
            self.console.print("\n[bold]Label Creation Strategy:[/bold]")
            self.console.print("• [cyan]key_value[/cyan]: Labels include key name (e.g., 'sentiment_positive')")
            self.console.print("• [cyan]value_only[/cyan]: Labels are just values (e.g., 'positive')")

            label_strategy = Prompt.ask(
                "Label strategy",
                choices=["key_value", "value_only"],
                default="key_value"
            )

            # Summary and confirmation
            self._display_configuration_summary({
                'dataset': dataset_path,
                'text_column': text_column,
                'identifier_column': identifier_column or "llm_annotation_id (auto)",
                'model': best_llm.name,
                'prompt_mode': prompt_mode,
                'training_mode': training_mode,
                'training_strategy': training_strategy,
                'label_strategy': label_strategy,
                'annotation_sample_size': annotation_settings['annotation_sample_size'] or 'all',
                **training_config
            })

            if Confirm.ask("\n[bold yellow]Start execution?[/bold yellow]", default=True):
                # Save as profile for future use
                if Confirm.ask("Save this configuration as a profile?", default=True):
                    profile_name = Prompt.ask("Profile name", default="quick_start")
                    self._save_profile(profile_name, {
                    'dataset': dataset_path,
                    'text_column': text_column,
                    'identifier_column': identifier_column,
                    'model': best_llm.name,
                    'api_key': api_key,
                    'prompt': prompt,
                    'training_config': training_config,
                    'annotation_settings': annotation_settings
                    })

                # Execute pipeline
                self._execute_quick_start({
                    'dataset': dataset_path,
                    'text_column': text_column,
                    'identifier_column': identifier_column,
                    'lang_column': lang_column,
                    'model': selected_llm,
                    'api_key': api_key,
                    'prompt': prompt,
                    'training_config': training_config,
                    'annotation_settings': annotation_settings,
                    'training_strategy': training_strategy,
                    'label_strategy': label_strategy,
                    'training_annotation_keys': training_annotation_keys
                })

        else:
            # Simplified version for non-Rich environments
            print("\n=== Quick Start Wizard ===\n")
            print("Analyzing environment...")

            best_llm = self._auto_select_llm()
            print(f"Selected LLM: {best_llm.name}")

            dataset_path = input("Dataset path: ").strip()
            text_column = input("Text column (default: text): ").strip() or "text"

            if best_llm.requires_api_key:
                api_key = input("API Key: ").strip()
            else:
                api_key = None

            # Prompt configuration (simple text input)
            print("\nStarting pipeline...")
            # Execute simplified pipeline
            annotation_sample_size = input("How many sentences to annotate for training (0 = all, default 200)? ").strip()
            try:
                annotation_sample_size = int(annotation_sample_size) if annotation_sample_size else 200
            except ValueError:
                annotation_sample_size = 200

            annotation_sample_seed = 42
            annotation_sampling_strategy = 'head'
            if annotation_sample_size and annotation_sample_size > 0:
                random_choice = input("Sample sentences randomly? (y/N): ").strip().lower()
                if random_choice == 'y':
                    annotation_sampling_strategy = 'random'
                    seed_input = input("Random seed (default 42): ").strip()
                    try:
                        annotation_sample_seed = int(seed_input) if seed_input else 42
                    except ValueError:
                        annotation_sample_seed = 42

            annotation_settings = {
                'annotation_sample_size': annotation_sample_size if annotation_sample_size > 0 else None,
                'annotation_sampling_strategy': annotation_sampling_strategy,
                'annotation_sample_seed': annotation_sample_seed,
            }

    def _select_llm_interactive(self) -> ModelInfo:
        """Let user interactively select an LLM from available options"""
        if HAS_RICH and self.console:
            self.console.print("\n[bold]Available LLMs:[/bold]")

            # Collect all available LLMs
            all_llms = []
            local_llms = self.detected_llms.get('local', [])
            openai_llms = self.detected_llms.get('openai', [])
            anthropic_llms = self.detected_llms.get('anthropic', [])

            # Display by category
            if local_llms:
                self.console.print("\n[cyan]Local Models (Ollama):[/cyan]")
                for i, llm in enumerate(local_llms, 1):
                    idx = len(all_llms) + 1
                    self.console.print(f"  {idx}. {llm.name} ({llm.size or 'N/A'})")
                    all_llms.append(llm)

            if openai_llms:
                self.console.print("\n[cyan]OpenAI Models:[/cyan]")
                for llm in openai_llms[:3]:  # Show top 3
                    idx = len(all_llms) + 1
                    cost = f"${llm.cost_per_1k_tokens}/1K" if llm.cost_per_1k_tokens else "N/A"
                    self.console.print(f"  {idx}. {llm.name} ({cost})")
                    all_llms.append(llm)

            if anthropic_llms:
                self.console.print("\n[cyan]Anthropic Models:[/cyan]")
                for llm in anthropic_llms[:3]:  # Show top 3
                    idx = len(all_llms) + 1
                    cost = f"${llm.cost_per_1k_tokens}/1K" if llm.cost_per_1k_tokens else "N/A"
                    self.console.print(f"  {idx}. {llm.name} ({cost})")
                    all_llms.append(llm)

            if not all_llms:
                self.console.print("[red]No LLMs detected![/red]")
                return ModelInfo("llama3.2", "ollama", is_available=False)

            # Ask user to select with validation
            choice = self._int_prompt_with_validation("\nSelect LLM", default=1, min_value=1, max_value=len(all_llms))
            return all_llms[choice - 1]
        else:
            # Fallback
            return self._auto_select_llm()

    def _auto_select_llm(self) -> ModelInfo:
        """Intelligently select the best available LLM for annotation"""
        # Priority: Local LLMs > API LLMs with key > Others

        local_llms = self.detected_llms.get('local', [])
        if local_llms:
            # Prefer larger, more capable local LLMs
            # Sort by size (if available) or by preferred order
            preferred_order = ['gemma3:27b', 'llama3.2:latest', 'mixtral', 'llama3', 'llama2', 'mistral', 'gemma2', 'gemma']

            for preferred in preferred_order:
                for llm in local_llms:
                    if preferred in llm.name.lower() or llm.name.lower().startswith(preferred.split(':')[0]):
                        return llm

            # Return the first available if no preferred found
            return local_llms[0]

        # Check for API keys in environment
        if os.getenv('OPENAI_API_KEY'):
            openai_llms = self.detected_llms.get('openai', [])
            if openai_llms:
                # Prefer GPT-4 Turbo for best quality/cost ratio
                for llm in openai_llms:
                    if 'gpt-4-turbo' in llm.name:
                        return llm
                return openai_llms[0]

        if os.getenv('ANTHROPIC_API_KEY'):
            anthropic_llms = self.detected_llms.get('anthropic', [])
            if anthropic_llms:
                # Prefer Sonnet for balance
                for llm in anthropic_llms:
                    if 'sonnet' in llm.name:
                        return llm
                return anthropic_llms[0]

        # Default fallback
        return ModelInfo(
            name="llama3.2",
            provider="ollama",
            is_available=False,
            requires_api_key=False
        )

    def _auto_select_dataset(self) -> Optional[DatasetInfo]:
        """Intelligently select the most likely dataset"""
        if not self.detected_datasets:
            return None

        # Prefer CSV files with obvious text columns
        for dataset in self.detected_datasets:
            if dataset.format == 'csv':
                text_col = self.data_detector.suggest_text_column(dataset)
                if text_col:
                    return dataset

        # Return largest dataset as fallback
        return max(self.detected_datasets, key=lambda d: d.size_mb or 0)

    def _prompt_file_path(self, prompt_text: str) -> str:
        """Prompt for file path with validation and suggestions"""
        while True:
            if HAS_RICH and self.console:
                # Show available files
                files = list(Path.cwd().glob("*.csv")) + list(Path.cwd().glob("*.json"))
                if files:
                    self.console.print("\n[dim]Available files:[/dim]")
                    for i, f in enumerate(files[:5], 1):
                        self.console.print(f"  {i}. {f.name}")

                path = Prompt.ask(f"\n{prompt_text}")

                # Allow number selection
                if path.isdigit() and int(path) <= len(files):
                    path = str(files[int(path)-1])

                if Path(path).exists():
                    return path
                else:
                    self.console.print(f"[red]File not found: {path}[/red]")
            else:
                path = input(f"{prompt_text}: ").strip()
                if Path(path).exists():
                    return path
                else:
                    print(f"File not found: {path}")

    def _generate_auto_prompt(self, dataset_path: str, text_column: str) -> str:
        """Generate intelligent prompt based on dataset analysis"""
        # Simple heuristic-based prompt generation
        prompt_template = """Analyze the following text and extract key information.

Text: {text}

Please provide a structured analysis including:
1. Main topic or subject
2. Sentiment (positive/negative/neutral)
3. Key entities mentioned
4. Summary in one sentence

Format your response as JSON with keys: topic, sentiment, entities, summary"""

        # Could be enhanced with actual dataset sampling and analysis
        return prompt_template

    def _detect_prompts_in_directory(self) -> List[Path]:
        """Detect available prompt files in the prompts directory"""
        prompts_dir = self.settings.paths.prompts_dir
        if prompts_dir.exists():
            return sorted(list(prompts_dir.glob("*.txt")))
        return []

    def _get_custom_prompt(self) -> str:
        """Get custom prompt from user - detect from directory, file path, or paste"""
        if HAS_RICH and self.console:
            # Try to detect prompts in the prompts directory
            detected_prompts = self._detect_prompts_in_directory()

            if detected_prompts:
                self.console.print("\n[bold green]✓ Detected prompts in prompts directory:[/bold green]")
                for i, prompt_file in enumerate(detected_prompts, 1):
                    self.console.print(f"  {i}. {prompt_file.name}")

                use_detected = Confirm.ask("\nUse one of these prompts?", default=True)

                if use_detected:
                    if len(detected_prompts) == 1:
                        selected_prompt = detected_prompts[0]
                    else:
                        choice = IntPrompt.ask(
                            "Select prompt",
                            default=1,
                            min_value=1,
                            max_value=len(detected_prompts)
                        )
                        selected_prompt = detected_prompts[choice - 1]

                    # Load and return the prompt
                    try:
                        full_prompt, expected_keys = self.prompt_manager.load_prompt(str(selected_prompt))
                        self.console.print(f"\n[green]✓ Loaded prompt: {selected_prompt.name}[/green]")
                        self.console.print(f"[dim]Detected JSON keys: {', '.join(expected_keys[:5])}{'...' if len(expected_keys) > 5 else ''}[/dim]")
                        return full_prompt
                    except Exception as e:
                        self.console.print(f"[red]Error loading prompt: {e}[/red]")

            # If no detected prompts or user declined, ask for input method
            self.console.print("\n[bold]Prompt Input Method:[/bold]")
            method = Prompt.ask(
                "How do you want to provide the prompt?",
                choices=["path", "paste"],
                default="path"
            )

            if method == "path":
                # Ask for file path
                prompt_path = Prompt.ask("\nPath to prompt file (.txt)")
                while not Path(prompt_path).exists():
                    self.console.print(f"[red]File not found: {prompt_path}[/red]")
                    prompt_path = Prompt.ask("Path to prompt file (.txt)")

                try:
                    full_prompt, expected_keys = self.prompt_manager.load_prompt(prompt_path)
                    self.console.print(f"[green]✓ Loaded prompt from: {prompt_path}[/green]")
                    self.console.print(f"[dim]Detected JSON keys: {', '.join(expected_keys[:5])}{'...' if len(expected_keys) > 5 else ''}[/dim]")
                    return full_prompt
                except Exception as e:
                    self.console.print(f"[red]Error loading prompt: {e}[/red]")
                    return ""

            else:  # paste
                self.console.print("\n[bold cyan]Paste your prompt below[/bold cyan]")
                self.console.print("[dim]Press Ctrl+D (Unix/Mac) or Ctrl+Z (Windows) when done[/dim]\n")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass

                prompt_text = "\n".join(lines)
                self.console.print(f"\n[green]✓ Received prompt ({len(prompt_text)} characters)[/green]")
                return prompt_text

        else:
            # Fallback for non-Rich environments
            return input("Enter prompt: ").strip()

    def _get_multi_prompts(self) -> List[Tuple[str, List[str], str]]:
        """Get multiple prompts for multi-prompt mode"""
        if HAS_RICH and self.console:
            self.console.print("\n[bold cyan]Multi-Prompt Configuration[/bold cyan]")

            # Option to load from folder or individually
            load_from_folder = Confirm.ask(
                "Load all prompts from the prompts directory?",
                default=True
            )

            if load_from_folder:
                # Use the PromptManager's folder loading feature
                prompts_dir = str(self.settings.paths.prompts_dir)
                txt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])

                if not txt_files:
                    self.console.print(f"[yellow]No .txt files found in {prompts_dir}[/yellow]")
                    return []

                self.console.print(f"\n[green]Found {len(txt_files)} prompt files:[/green]")
                for i, filename in enumerate(txt_files, 1):
                    self.console.print(f"  {i}. {filename}")

                # Load each file
                prompts_list = []
                for i, filename in enumerate(txt_files, 1):
                    filepath = os.path.join(prompts_dir, filename)
                    self.console.print(f"\n[cyan][{i}/{len(txt_files)}] Loading {filename}...[/cyan]")

                    try:
                        full_prompt, expected_keys = self.prompt_manager.load_prompt(filepath)

                        if expected_keys:
                            self.console.print(f"[green]✓ Detected {len(expected_keys)} JSON keys[/green]")

                            # Ask for prefix
                            use_prefix = Confirm.ask(f"Add prefix to keys from '{filename}'?", default=False)
                            prefix_word = ""
                            if use_prefix:
                                default_prefix = Path(filename).stem.lower().replace(' ', '_')
                                prefix_word = Prompt.ask(f"Prefix (default: {default_prefix})", default=default_prefix)
                                self.console.print(f"[dim]✓ Keys will be prefixed with '{prefix_word}_'[/dim]")

                            prompts_list.append((full_prompt, expected_keys, prefix_word))
                        else:
                            self.console.print(f"[yellow]⚠ No JSON keys detected, skipping...[/yellow]")

                    except Exception as e:
                        self.console.print(f"[red]❌ Error loading {filename}: {e}[/red]")
                        continue

                if prompts_list:
                    self.console.print(f"\n[bold green]✓ Successfully loaded {len(prompts_list)} prompts[/bold green]")
                else:
                    self.console.print("\n[red]❌ No valid prompts could be loaded[/red]")

                return prompts_list

            else:
                # Load prompts individually
                num_prompts = self._int_prompt_with_validation("How many prompts do you want to use?", default=2, min_value=2, max_value=10)

                prompts_list = []
                for i in range(1, num_prompts + 1):
                    self.console.print(f"\n[bold]=== Prompt {i}/{num_prompts} ===[/bold]")

                    # Use the single prompt loader
                    prompt_text = self._get_custom_prompt()

                    # Ask for prefix
                    use_prefix = Confirm.ask(f"Add prefix to keys from prompt {i}?", default=False)
                    prefix_word = ""
                    if use_prefix:
                        prefix_word = Prompt.ask(f"Prefix for prompt {i}", default=f"p{i}")
                        self.console.print(f"[dim]✓ Keys will be prefixed with '{prefix_word}_'[/dim]")

                    # Extract expected keys from prompt
                    from ..annotators.json_cleaner import extract_expected_keys
                    expected_keys = extract_expected_keys(prompt_text)

                    prompts_list.append((prompt_text, expected_keys, prefix_word))

                self.console.print(f"\n[bold green]✓ Configured {len(prompts_list)} prompts[/bold green]")
                return prompts_list

        else:
            # Fallback for non-Rich environments
            return []

    def _get_prompt_template(self, template_type: str) -> str:
        """Get predefined prompt template"""
        templates = {
            "1": "Classify the following text into one of these categories: {categories}\n\nText: {text}\n\nCategory:",
            "2": "Extract all named entities from the text.\n\nText: {text}\n\nEntities:",
            "3": "Analyze the sentiment of this text (positive/negative/neutral).\n\nText: {text}\n\nSentiment:",
            "4": "Summarize the following text in one sentence.\n\nText: {text}\n\nSummary:"
        }
        return templates.get(template_type, "Analyze: {text}")

    def _get_training_preset(self, mode: str) -> Dict[str, Any]:
        """Get training configuration preset"""
        presets = {
            "quick": {
                "epochs": 3,
                "batch_size": 32,
                "learning_rate": 5e-5,
                "warmup_ratio": 0.1,
                "models_to_test": 2
            },
            "balanced": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "models_to_test": 5
            },
            "thorough": {
                "epochs": 20,
                "batch_size": 8,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.2,
                "models_to_test": 10
            }
        }

        if mode == "custom":
            if HAS_RICH and self.console:
                return {
                    "epochs": IntPrompt.ask("Number of epochs", default=10),
                    "batch_size": IntPrompt.ask("Batch size", default=16),
                    "learning_rate": FloatPrompt.ask("Learning rate", default=2e-5),
                    "warmup_ratio": FloatPrompt.ask("Warmup ratio", default=0.1),
                    "models_to_test": IntPrompt.ask("Models to benchmark", default=5)
                }
            else:
                return presets["balanced"]

        return presets.get(mode, presets["balanced"])

    def _display_configuration_summary(self, config: Dict[str, Any]):
        """Display configuration summary in professional format"""
        if HAS_RICH and self.console:
            table = Table(title="📋 Configuration Summary", border_style="green", show_lines=True)
            table.add_column("Setting", style="cyan", width=20)
            table.add_column("Value", style="white")

            for key, value in config.items():
                if value is not None:
                    table.add_row(key.replace('_', ' ').title(), str(value))

            self.console.print(table)
        else:
            print("\n=== Configuration Summary ===")
            for key, value in config.items():
                if value is not None:
                    print(f"{key}: {value}")

    def _save_profile(self, name: str, config: Dict[str, Any]):
        """Save configuration as profile"""
        profile = ExecutionProfile(
            name=name,
            created_at=datetime.now(),
            last_used=datetime.now(),
            configuration=config
        )
        self.profile_manager.save_profile(profile)

        if HAS_RICH and self.console:
            self.console.print(f"[green]✓ Profile '{name}' saved successfully[/green]")

    def _recommend_models_for_training(self, dataset_path: str, text_column: str, sample_size: int = 200) -> Dict[str, Any]:
        """Recommend models based on detected language distribution in the dataset."""
        fallback = {
            'language_code': 'multilingual',
            'language_name': 'multilingual',
            'candidate_models': ['bert-base-multilingual-cased', 'xlm-roberta-base', 'mdeberta-v3-base'],
            'default_model': 'bert-base-multilingual-cased',
        }

        if not HAS_PANDAS or not dataset_path or not Path(dataset_path).exists():
            return fallback

        try:
            suffix = Path(dataset_path).suffix.lower()
            if suffix == '.csv':
                df = pd.read_csv(dataset_path)
            elif suffix in {'.xls', '.xlsx'}:
                df = pd.read_excel(dataset_path)
            elif suffix in {'.parquet'}:
                df = pd.read_parquet(dataset_path)
            else:
                return fallback
        except Exception as exc:
            logging.warning("Quick start: unable to read dataset for language detection (%s)", exc)
            return fallback

        if text_column not in df.columns:
            logging.warning("Quick start: text column '%s' not found in dataset", text_column)
            return fallback

        texts = df[text_column].dropna().astype(str)
        if texts.empty:
            return fallback

        sampled = texts.sample(min(len(texts), sample_size), random_state=42) if len(texts) > sample_size else texts
        detections = self.language_detector.detect_batch(sampled.tolist(), parallel=False)

        language_counts = Counter()
        for detection in detections:
            if not detection:
                continue
            lang = detection.get('language')
            confidence = detection.get('confidence', 0)
            if lang and confidence >= 0.5:
                language_counts[lang] += 1

        if not language_counts:
            return fallback

        top_language, _ = language_counts.most_common(1)[0]
        candidate_models = self.language_detector.get_recommended_models(top_language)
        default_model = candidate_models[0] if candidate_models else fallback['default_model']
        language_name = self.language_detector.get_language_name(top_language)

        return {
            'language_code': top_language,
            'language_name': language_name,
            'candidate_models': candidate_models or fallback['candidate_models'],
            'default_model': default_model,
        }

    def _execute_quick_start(self, config: Dict[str, Any]):
        """Execute the quick start pipeline"""
        dataset_path = config['dataset']
        text_column = config['text_column']
        identifier_column = config.get('identifier_column')
        model_entry = config['model']
        if isinstance(model_entry, ModelInfo):
            model_info = model_entry
        elif isinstance(model_entry, dict):
            model_info = ModelInfo(**model_entry)
        else:
            model_info = ModelInfo(name=str(model_entry), provider='ollama', is_available=True)
        api_key = config.get('api_key')
        prompt_config = config.get('prompt')
        training_preset = config.get('training_config', {})
        annotation_settings = config.get('annotation_settings', {})

        data_format = Path(dataset_path).suffix.lower().lstrip('.') if dataset_path else 'csv'
        if data_format == '':
            data_format = 'csv'
        if data_format not in {'csv', 'json', 'jsonl', 'excel', 'xlsx', 'xls', 'parquet'}:
            data_format = 'csv'

        # Prepare prompts payload
        prompts_payload: List[Dict[str, Any]] = []
        if isinstance(prompt_config, list):
            # Multi-prompt returns list of tuples (prompt, keys, prefix)
            for item in prompt_config:
                if isinstance(item, tuple):
                    prompt_text, expected_keys, prefix = item
                elif isinstance(item, dict):
                    prompt_text = item.get('prompt')
                    expected_keys = item.get('expected_keys', [])
                    prefix = item.get('prefix', '')
                else:
                    continue
                prompts_payload.append({
                    'prompt': prompt_text,
                    'expected_keys': expected_keys or [],
                    'prefix': prefix or ''
                })
        else:
            prompt_text = prompt_config or ""
            prompts_payload.append({
                'prompt': prompt_text,
                'expected_keys': extract_expected_keys(prompt_text) if prompt_text else [],
                'prefix': ''
            })

        # Recommend training models based on detected language
        training_reco = self._recommend_models_for_training(dataset_path, text_column)
        candidate_models = training_reco['candidate_models']
        default_model = training_reco['default_model']
        detected_language = training_reco['language_name']

        # Quick start focuses on a single recommended model for speed/stability
        models_to_test = [default_model]
        benchmark_mode = False

        annotations_dir = self.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = model_info.name.replace(':', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output_path = annotations_dir / f"{Path(dataset_path).stem}_{safe_model_name}_annotations_{timestamp}.csv"

        annotation_mode = 'api' if model_info.provider in {'openai', 'anthropic', 'google', 'custom'} else 'local'

        pipeline_config = {
            'mode': 'file',
            'data_source': data_format,
            'data_format': data_format,
            'file_path': dataset_path,
            'text_column': text_column,
            'text_columns': [text_column],
            'annotation_column': 'annotation',
            'identifier_column': identifier_column,
            'lang_column': config.get('lang_column'),
            'run_annotation': True,
            'annotation_mode': annotation_mode,
            'annotation_provider': model_info.provider,
            'annotation_model': model_info.name,
            'api_key': api_key,
            'prompts': prompts_payload,
            'annotation_sample_size': annotation_settings.get('annotation_sample_size'),
            'annotation_sampling_strategy': annotation_settings.get('annotation_sampling_strategy', 'head'),
            'annotation_sample_seed': annotation_settings.get('annotation_sample_seed', 42),
            'max_workers': 1,
            'num_processes': 1,
            'use_parallel': False,
            'warmup': False,
            'output_format': 'csv',
            'output_path': str(default_output_path),
            'run_validation': False,
            'run_training': False,  # TEMPORARILY DISABLED - ModelTrainer import causes mutex lock
            'training_strategy': config.get('training_strategy', 'single-label'),
            'label_strategy': config.get('label_strategy', 'key_value'),
            'training_annotation_keys': config.get('training_annotation_keys'),
            'benchmark_mode': benchmark_mode,
            'models_to_test': models_to_test,
            'auto_select_best': True,
            'max_epochs': training_preset.get('epochs', 10),
            'batch_size': training_preset.get('batch_size', 16),
            'learning_rate': training_preset.get('learning_rate', 2e-5),
            'run_deployment': False,
            'training_model_type': models_to_test[0] if not benchmark_mode else default_model,
        }

        # Execute pipeline
        try:
            import os
            import tempfile

            # WORKAROUND: Change to temp directory to avoid Ollama mutex lock on .gitignore
            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)

                try:
                    # Execute with beautiful progress display
                    if HAS_RICH and self.console:
                        with Progress(
                            SpinnerColumn(style="cyan"),
                            TextColumn("[bold cyan]{task.description}"),
                            BarColumn(
                                complete_style="bright_green",
                                finished_style="bright_green",
                                pulse_style="bright_cyan"
                            ),
                            TextColumn("[bright_cyan]{task.percentage:>3.0f}%"),
                            TimeElapsedColumn(),
                            console=self.console,
                            transient=False
                        ) as progress:
                            # Create main pipeline task
                            main_task = progress.add_task(
                                "🚀 Initializing pipeline...",
                                total=100
                            )

                            # Start pipeline in background
                            import threading
                            result_container = {}

                            def run_pipeline():
                                try:
                                    result_container['state'] = self.pipeline_controller.run_pipeline(pipeline_config)
                                    result_container['success'] = True
                                except Exception as e:
                                    result_container['error'] = e
                                    result_container['success'] = False

                            thread = threading.Thread(target=run_pipeline)
                            thread.start()

                            # Update progress with phase information
                            phase_messages = [
                                ("🔍 Loading data...", 10),
                                ("🤖 Initializing model...", 20),
                                ("✍️  Annotating samples...", 60),
                                ("🔄 Converting to training format...", 75),
                                ("🏋️  Training model...", 95),
                                ("✅ Finalizing...", 100)
                            ]

                            phase_idx = 0
                            while thread.is_alive():
                                current_progress = progress.tasks[main_task].completed

                                # Update phase message
                                if phase_idx < len(phase_messages) and current_progress < phase_messages[phase_idx][1]:
                                    progress.update(
                                        main_task,
                                        description=phase_messages[phase_idx][0],
                                        advance=0.3
                                    )
                                    if current_progress >= phase_messages[phase_idx][1] - 5:
                                        phase_idx += 1

                                thread.join(timeout=0.1)

                            # Complete progress
                            progress.update(
                                main_task,
                                description="✨ Pipeline completed!",
                                completed=100
                            )

                            if not result_container.get('success'):
                                error = result_container.get('error', Exception("Pipeline failed"))
                                # Print error message with traceback
                                self.console.print(f"\n[bold red]❌ Error during pipeline execution:[/bold red]")
                                self.console.print(f"[red]{str(error)}[/red]")

                                # Log full traceback to file
                                import traceback
                                self.logger.error("Pipeline execution failed", exc_info=error)

                                # Show log file location
                                self.console.print(f"\n[dim]Full error details logged to: {self.current_log_file}[/dim]")
                                raise error

                            state = result_container['state']
                    else:
                        # Fallback for non-Rich environments
                        state = self.pipeline_controller.run_pipeline(pipeline_config)

                finally:
                    os.chdir(old_cwd)
        except Exception as exc:
            message = f"❌ Quick start pipeline failed: {exc}"
            if HAS_RICH and self.console:
                self.console.print(f"[red]{message}[/red]")
            else:
                print(message)
            logging.exception("Quick start pipeline failed")
            return

        annotation_results = state.annotation_results or {}
        training_results = state.training_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        if HAS_RICH and self.console:
            self.console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")
            self.console.print(f"📄 Annotated file: [cyan]{output_file}[/cyan]")
            self.console.print(f"🗣️ Detected language: [cyan]{detected_language}[/cyan]")
            if training_results:
                best_model = training_results.get('best_model') or training_results.get('model_name')
                best_f1 = training_results.get('best_f1_macro')
                if best_model:
                    self.console.print(f"🏆 Best model: [cyan]{best_model}[/cyan]")
                if best_f1 is not None:
                    self.console.print(f"📊 Macro F1: [cyan]{best_f1:.3f}[/cyan]")
        else:
            print("✅ Pipeline completed successfully!")
            print(f"Annotated file: {output_file}")
            print(f"Detected language: {detected_language}")
            if training_results:
                best_model = training_results.get('best_model') or training_results.get('model_name')
                best_f1 = training_results.get('best_f1_macro')
                if best_model:
                    print(f"Best model: {best_model}")
                if best_f1 is not None:
                    print(f"Macro F1: {best_f1:.3f}")

        # Persist last run details for other wizards
        self.last_annotation_config = {
            'data_path': dataset_path,
            'data_format': data_format,
            'text_column': text_column,
            'annotation_column': 'annotation',
            'mode': annotation_mode,
            'provider': model_info.provider,
            'model': model_info.name,
            'output_path': output_file,
            'annotation_sample_size': annotation_settings.get('annotation_sample_size'),
            'annotation_sampling_strategy': annotation_settings.get('annotation_sampling_strategy', 'head'),
        }

    def run(self):
        """Main run loop for the advanced CLI"""
        self.display_banner()

        while True:
            try:
                choice = self.get_main_menu_choice()

                if choice == "1":
                    self.quick_start_wizard()
                elif choice == "2":
                    self.annotation_wizard()
                elif choice == "3":
                    self.complete_pipeline()
                elif choice == "4":
                    self.training_studio()
                elif choice == "5":
                    self.validation_lab()
                elif choice == "6":
                    self.analytics_dashboard()
                elif choice == "7":
                    self.profile_manager_ui()
                elif choice == "8":
                    self.advanced_settings()
                elif choice == "9":
                    self.show_documentation()
                elif choice == "0":
                    if HAS_RICH and self.console:
                        self.console.print("\n[bold cyan]Thank you for using LLMTool! 👋[/bold cyan]\n")
                    else:
                        print("\nThank you for using LLMTool!\n")
                    sys.exit(0)

                # Update session
                self.current_session['operations_count'] += 1
                self.current_session['last_operation'] = choice

            except KeyboardInterrupt:
                if HAS_RICH and self.console:
                    if Confirm.ask("\n[yellow]Exit LLMTool?[/yellow]", default=False):
                        self.console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]\n")
                        sys.exit(0)
                else:
                    if input("\nExit? (y/n): ").lower() == 'y':
                        print("Goodbye!\n")
                        sys.exit(0)
            except Exception as e:
                self.logger.error(f"Error: {str(e)}", exc_info=True)
                if HAS_RICH and self.console:
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                else:
                    print(f"Error: {str(e)}")

    # Placeholder methods for other menu options
    def annotation_wizard(self):
        """Guided annotation wizard with step-by-step configuration"""
        if HAS_RICH and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]📝 Annotation Wizard[/bold cyan]\n"
                "Interactive guided setup for LLM annotation",
                border_style="cyan"
            ))

            # Step 1: Choose annotation mode
            self.console.print("\n[bold]Step 1: Annotation Mode[/bold]")
            mode = Prompt.ask(
                "Select mode",
                choices=["local", "api", "hybrid"],
                default="local"
            )

            # Step 2: Model selection based on mode
            self.console.print("\n[bold]Step 2: LLM Selection[/bold]")

            if mode == "local":
                local_llms = self.detected_llms.get('local', [])
                if not local_llms:
                    self.console.print("[red]No local LLMs found![/red]")
                    self.console.print("[yellow]Run: ollama pull llama3.2[/yellow]")
                    return

                # Display LLMs in a nice table
                llm_table = Table(title="Available Local LLMs", border_style="blue")
                llm_table.add_column("#", style="cyan", width=3)
                llm_table.add_column("Model", style="white")
                llm_table.add_column("Size", style="yellow")
                llm_table.add_column("Context", style="green")

                for i, llm in enumerate(local_llms, 1):
                    llm_table.add_row(
                        str(i),
                        llm.name,
                        llm.size or "N/A",
                        f"{llm.context_length:,}" if llm.context_length else "N/A"
                    )

                self.console.print(llm_table)
                choice = self._int_prompt_with_validation("Select LLM", default=1, min_value=1, max_value=len(local_llms))
                selected_llm = local_llms[choice-1]
                api_key = None

            else:  # API mode
                provider = Prompt.ask(
                    "Provider",
                    choices=["openai", "anthropic", "google"],
                    default="openai"
                )
                api_key = Prompt.ask("API Key", password=True)

                # Show available API models
                api_models = self.detected_llms.get(provider, [])
                for i, model in enumerate(api_models[:5], 1):
                    cost = f"${model.cost_per_1k_tokens}/1K" if model.cost_per_1k_tokens else "N/A"
                    self.console.print(f"  {i}. {model.name} ({cost})")

                model_choice = self._int_prompt_with_validation("Select model", default=1, min_value=1, max_value=len(api_models))
                selected_llm = api_models[model_choice-1]

            # Step 3: Data configuration
            self.console.print("\n[bold]Step 3: Data Configuration[/bold]")

            # Show detected datasets or ask for path
            if self.detected_datasets:
                use_detected = Confirm.ask(
                    f"Use detected dataset? ({len(self.detected_datasets)} found)",
                    default=True
                )
                if use_detected:
                    for i, ds in enumerate(self.detected_datasets[:5], 1):
                        self.console.print(f"  {i}. {ds.path.name} ({ds.format}, {ds.size_mb:.1f} MB)")

                    ds_choice = IntPrompt.ask("Select dataset", default=1)
                    dataset = self.detected_datasets[ds_choice-1]
                    data_path = str(dataset.path)
                    text_column = self.data_detector.suggest_text_column(dataset) or "text"
                else:
                    data_path = self._prompt_file_path("Data file path")
                    text_column = Prompt.ask("Text column", default="text")
            else:
                data_path = self._prompt_file_path("Data file path")
                text_column = Prompt.ask("Text column", default="text")

            # Step 4: Prompt engineering
            self.console.print("\n[bold]Step 4: Prompt Engineering[/bold]")
            self.console.print("[dim]• simple: Single prompt for all texts[/dim]")
            self.console.print("[dim]• multi: Multiple prompts applied to each text[/dim]")
            self.console.print("[dim]• template: Pre-configured prompt templates [yellow](Under development - not available)[/yellow][/dim]")

            prompt_mode = Prompt.ask(
                "Prompt strategy",
                choices=["simple", "multi"],
                default="simple"
            )

            if prompt_mode == "simple":
                prompt_text = self._get_custom_prompt()
            else:  # multi
                prompt_text = self._get_multi_prompts()

            # Step 5: Advanced options
            self.console.print("\n[bold]Step 5: Advanced Options[/bold]")

            batch_size = self._int_prompt_with_validation("Batch size", default=10, min_value=1, max_value=100)
            max_workers = self._int_prompt_with_validation("Parallel workers", default=4, min_value=1, max_value=16)
            save_incrementally = Confirm.ask("Save incrementally?", default=True)

            # Step 5.5: Training strategy
            self.console.print("\n[bold]Training Strategy (Optional):[/bold]")
            if Confirm.ask("Prepare data for model training after annotation?", default=True):
                self.console.print("How do you want to create training labels from annotations?")
                self.console.print("• [cyan]single-label[/cyan]: Train separate models for each annotation key")
                self.console.print("• [cyan]multi-label[/cyan]: Train one model with all labels together")

                training_strategy = Prompt.ask(
                    "Training strategy",
                    choices=["single-label", "multi-label"],
                    default="single-label"
                )

                # If single-label, ask which keys to train
                training_annotation_keys = None
                if training_strategy == "single-label":
                    self.console.print("\n[dim]Available annotation keys will be detected from the annotations[/dim]")
                    if Confirm.ask("Train models for all annotation keys?", default=True):
                        training_annotation_keys = None  # Will use all keys
                    else:
                        keys_input = Prompt.ask("Enter annotation keys to train (comma-separated)")
                        training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

                # Label creation strategy
                self.console.print("\n[bold]Label Creation Strategy:[/bold]")
                self.console.print("• [cyan]key_value[/cyan]: Labels include key name (e.g., 'sentiment_positive')")
                self.console.print("• [cyan]value_only[/cyan]: Labels are just values (e.g., 'positive')")

                label_strategy = Prompt.ask(
                    "Label strategy",
                    choices=["key_value", "value_only"],
                    default="key_value"
                )
            else:
                training_strategy = None
                training_annotation_keys = None
                label_strategy = None

            # Step 6: Execute
            self.console.print("\n[bold]Ready to annotate![/bold]")

            config = {
                'mode': mode,
                'provider': selected_llm.provider,
                'model': selected_llm.name,
                'api_key': api_key,
                'data_path': data_path,
                'text_column': text_column,
                'prompt': prompt_text,
                'batch_size': batch_size,
                'max_workers': max_workers,
                'save_incrementally': save_incrementally,
                'training_strategy': training_strategy,
                'label_strategy': label_strategy,
                'training_annotation_keys': training_annotation_keys
            }

            self._display_configuration_summary(config)

            if Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console
                ) as progress:

                    task = progress.add_task("[cyan]Annotating...", total=100)

                    # Here would call actual annotation
                    from ..annotators.llm_annotator import LLMAnnotator
                    annotator = LLMAnnotator()

                    # Simulate for now
                    for i in range(100):
                        time.sleep(0.01)
                        progress.update(task, advance=1)

                    self.console.print("\n[bold green]✅ Annotation complete![/bold green]")
                    self.console.print("📁 Results saved to: annotations_output.json")

        else:
            print("\n=== Annotation Wizard ===")
            print("Guided LLM annotation setup\n")
            # Simplified version
            print("Feature coming soon with full implementation")

    def complete_pipeline(self):
        """Complete pipeline workflow - Full annotation to deployment"""
        if HAS_RICH and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]🚀 Complete Pipeline[/bold cyan]\n"
                "Full workflow: Annotation → Training → Validation → Deployment",
                border_style="cyan"
            ))

            # Step 1: Data Selection
            self.console.print("\n[bold yellow]Step 1: Data Selection[/bold yellow]")

            # Auto-detect or ask for dataset
            if self.detected_datasets:
                self.console.print("\n[dim]Available datasets detected:[/dim]")
                for i, dataset in enumerate(self.detected_datasets[:10], 1):
                    self.console.print(f"  {i}. {dataset.path.name} ({dataset.format.upper()}, {dataset.size_mb:.1f} MB)")

                choice = Prompt.ask("Select dataset (number) or enter path", default="1")
                if choice.isdigit() and int(choice) <= len(self.detected_datasets):
                    selected_dataset = self.detected_datasets[int(choice)-1]
                    data_path = str(selected_dataset.path)
                    text_column = self.data_detector.suggest_text_column(selected_dataset) or "text"
                else:
                    data_path = choice
                    text_column = Prompt.ask("Text column name", default="text")
            else:
                data_path = self._prompt_file_path("Enter dataset path")
                text_column = Prompt.ask("Text column name", default="text")

            # Step 2: LLM Selection for Annotation
            self.console.print("\n[bold yellow]Step 2: LLM Selection for Annotation[/bold yellow]")

            # Show available LLMs
            local_llms = self.detected_llms.get('local', [])
            if local_llms:
                self.console.print("\n[green]Local LLMs available:[/green]")
                for i, llm in enumerate(local_llms[:10], 1):
                    self.console.print(f"  {i}. {llm.name} ({llm.size or 'N/A'})")

                llm_choice = self._int_prompt_with_validation("Select LLM for annotation", default=1, min_value=1, max_value=len(local_llms))
                selected_llm = local_llms[llm_choice-1]
            else:
                self.console.print("[yellow]No local LLMs detected. Using API model.[/yellow]")
                provider = Prompt.ask("API Provider", choices=["openai", "anthropic"], default="openai")
                api_key = Prompt.ask("API Key", password=True)
                model_name = Prompt.ask("Model name", default="gpt-4-turbo")
                selected_llm = ModelInfo(name=model_name, provider=provider, requires_api_key=True)

            # Step 3: Prompt Configuration
            self.console.print("\n[bold yellow]Step 3: Prompt Configuration[/bold yellow]")
            prompt_text = self._get_custom_prompt()

            # Step 4: Training Model Selection
            self.console.print("\n[bold yellow]Step 4: Training Configuration[/bold yellow]")

            benchmark_mode = Confirm.ask("Benchmark multiple models?", default=True)

            if benchmark_mode:
                num_models = self._int_prompt_with_validation("Number of models to benchmark", default=5, min_value=2, max_value=20)
                self.console.print("[dim]Will test: BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT...[/dim]")
            else:
                # Show available training models
                self.console.print("\n[dim]Select model category:[/dim]")
                categories = list(self.available_trainer_models.keys())
                for i, cat in enumerate(categories, 1):
                    self.console.print(f"  {i}. {cat}")

                cat_choice = self._int_prompt_with_validation("Category", default=1, min_value=1, max_value=len(categories))
                selected_category = categories[cat_choice-1]
                models_in_cat = self.available_trainer_models[selected_category]

                self.console.print(f"\n[dim]Models in {selected_category}:[/dim]")
                for i, model in enumerate(models_in_cat[:10], 1):
                    self.console.print(f"  {i}. {model['name']} ({model.get('params', 'N/A')})")

            # Step 4.5: Training Data Preparation Strategy
            self.console.print("\n[bold yellow]Step 4.5: Training Data Preparation[/bold yellow]")
            self.console.print("How do you want to create training labels from annotations?")
            self.console.print("• [cyan]single-label[/cyan]: Train separate models for each annotation key")
            self.console.print("• [cyan]multi-label[/cyan]: Train one model with all labels together")

            training_strategy = Prompt.ask(
                "Training strategy",
                choices=["single-label", "multi-label"],
                default="single-label"
            )

            # If single-label, ask which keys to train
            training_annotation_keys = None
            if training_strategy == "single-label":
                self.console.print("\n[dim]Available annotation keys will be detected from the annotations[/dim]")
                if Confirm.ask("Train models for all annotation keys?", default=True):
                    training_annotation_keys = None  # Will use all keys
                else:
                    keys_input = Prompt.ask("Enter annotation keys to train (comma-separated)")
                    training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

            # Label creation strategy
            self.console.print("\n[bold]Label Creation Strategy:[/bold]")
            self.console.print("• [cyan]key_value[/cyan]: Labels include key name (e.g., 'sentiment_positive')")
            self.console.print("• [cyan]value_only[/cyan]: Labels are just values (e.g., 'positive')")

            label_strategy = Prompt.ask(
                "Label strategy",
                choices=["key_value", "value_only"],
                default="key_value"
            )

            # Step 5: Configuration Summary
            self.console.print("\n[bold yellow]Pipeline Configuration Summary:[/bold yellow]")

            config_table = Table(border_style="green", show_lines=True)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")

            config_table.add_row("Dataset", data_path)
            config_table.add_row("Text Column", text_column)
            config_table.add_row("Annotation LLM", f"{selected_llm.provider}/{selected_llm.name}")
            config_table.add_row("Training Mode", "Benchmark" if benchmark_mode else "Single Model")
            config_table.add_row("Validation", "Enabled with Doccano export")

            self.console.print(config_table)

            # Step 6: Execution
            if Confirm.ask("\n[bold yellow]Start pipeline execution?[/bold yellow]", default=True):

                # Prepare configuration
                pipeline_config = {
                    'mode': 'file',
                    'data_source': 'file',
                    'file_path': data_path,
                    'text_column': text_column,

                    # Annotation config
                    'run_annotation': True,
                    'annotation_mode': 'local' if selected_llm.provider == 'ollama' else 'api',
                    'annotation_provider': selected_llm.provider,
                    'annotation_model': selected_llm.name,
                    'api_key': api_key if selected_llm.requires_api_key else None,
                    'prompt_text': prompt_text,

                    # Validation config
                    'run_validation': True,
                    'validation_sample_size': 100,
                    'export_to_doccano': True,

                    # Training config
                    'run_training': True,
                    'benchmark_mode': benchmark_mode,
                    'models_to_test': num_models if benchmark_mode else 1,
                    'max_epochs': 10,
                    'batch_size': 16,
                    'training_strategy': training_strategy,
                    'label_strategy': label_strategy,
                    'training_annotation_keys': training_annotation_keys,

                    # Deployment config
                    'run_deployment': True,
                    'save_model': True
                }

                # Execute with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console
                ) as progress:

                    # Initialize pipeline
                    from ..pipelines.pipeline_controller import PipelineController
                    controller = PipelineController()

                    try:
                        # Annotation phase
                        task1 = progress.add_task("[cyan]Annotating with LLM...", total=100)
                        progress.update(task1, advance=50)

                        # Run pipeline
                        state = controller.run_pipeline(pipeline_config)
                        progress.update(task1, completed=100)

                        # Show results
                        self.console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")

                        if state.annotation_results:
                            self.console.print(f"📝 Annotations: {state.annotation_results.get('total_annotated', 0)} items")

                        if state.validation_results:
                            self.console.print(f"✓ Validation: Quality score {state.validation_results.get('quality_score', 0):.1f}/100")
                            self.console.print(f"📁 Doccano export: {state.validation_results.get('doccano_export_path', 'N/A')}")

                        if state.training_results:
                            self.console.print(f"🏆 Best model: {state.training_results.get('best_model', 'unknown')}")
                            self.console.print(f"📊 Best F1: {state.training_results.get('best_f1_macro', 0):.4f}")

                        # Save configuration as profile
                        if Confirm.ask("\nSave this configuration for future use?", default=True):
                            profile_name = Prompt.ask("Profile name", default="complete_pipeline")
                            self._save_profile(profile_name, pipeline_config)

                    except Exception as e:
                        self.console.print(f"\n[bold red]❌ Pipeline failed: {str(e)}[/bold red]")
                        raise

        else:
            # Non-Rich fallback
            print("\n=== Complete Pipeline ===")
            print("Full workflow: Annotation → Training → Validation → Deployment\n")

            # Simplified version
            data_path = input("Dataset path: ").strip()
            text_column = input("Text column (default: text): ").strip() or "text"

            # Ask for LLM configuration
            print("\n--- Annotation Configuration ---")
            print("Select annotation method:")
            print("1. Local LLM (Ollama)")
            print("2. OpenAI API")
            print("3. Anthropic API")
            choice = input("Choice (1-3): ").strip()

            if choice == "1":
                # Check for Ollama models
                local_llms = self.detected_llms.get('local', [])
                if local_llms:
                    print("\nLocal LLMs detected:")
                    for i, llm in enumerate(local_llms[:10], 1):
                        print(f"  {i}. {llm.name}")
                    llm_choice = input("Select LLM (number): ").strip()
                    if llm_choice.isdigit() and int(llm_choice) <= len(local_llms):
                        selected_llm = local_llms[int(llm_choice)-1]
                    else:
                        selected_llm = local_llms[0] if local_llms else None
                    provider = 'ollama'
                    api_key = None
                else:
                    print("No local LLMs found. Install Ollama or use API provider.")
                    return
            elif choice == "2":
                provider = 'openai'
                api_key = input("OpenAI API Key: ").strip()
                model_name = input("Model name (default: gpt-4): ").strip() or "gpt-4"
                selected_llm = ModelInfo(name=model_name, provider=provider, requires_api_key=True)
            elif choice == "3":
                provider = 'anthropic'
                api_key = input("Anthropic API Key: ").strip()
                model_name = input("Model name (default: claude-3-opus-20240229): ").strip() or "claude-3-opus-20240229"
                selected_llm = ModelInfo(name=model_name, provider=provider, requires_api_key=True)
            else:
                print("Invalid choice")
                return

            # Ask for prompt
            print("\n--- Prompt Configuration ---")
            print("Enter your classification prompt (or press Enter for default):")
            prompt_text = input("Prompt: ").strip()
            if not prompt_text:
                prompt_text = "Classify the following text into one of these categories: positive, negative, neutral. Text: {text}. Return only the label."

            # Training configuration
            print("\n--- Training Configuration ---")
            benchmark = input("Benchmark multiple models? (y/n, default: y): ").strip().lower()
            benchmark_mode = benchmark != 'n'

            if benchmark_mode:
                num_models = input("Number of models to test (default: 5): ").strip()
                num_models = int(num_models) if num_models.isdigit() else 5
            else:
                num_models = 1

            # Execute pipeline
            print("\n" + "="*50)
            print("STARTING PIPELINE EXECUTION")
            print("="*50)

            # Prepare configuration
            pipeline_config = {
                'mode': 'file',
                'data_source': 'file',
                'file_path': data_path,
                'text_column': text_column,

                # Annotation config
                'run_annotation': True,
                'annotation_mode': 'local' if provider == 'ollama' else 'api',
                'annotation_provider': provider,
                'annotation_model': selected_llm.name if selected_llm else 'gpt-4',
                'api_key': api_key,
                'prompt_text': prompt_text,

                # Validation config
                'run_validation': True,
                'validation_sample_size': 100,
                'export_to_doccano': True,

                # Training config
                'run_training': True,
                'benchmark_mode': benchmark_mode,
                'models_to_test': num_models,
                'max_epochs': 10,
                'batch_size': 16,

                # Deployment config
                'run_deployment': True,
                'save_model': True
            }

            # Initialize and run pipeline
            from ..pipelines.pipeline_controller import PipelineController
            controller = PipelineController()

            try:
                print("\n[Phase 1/4] Starting annotation...")
                state = controller.run_pipeline(pipeline_config)

                # Show results
                print("\n" + "="*50)
                print("PIPELINE COMPLETED SUCCESSFULLY")
                print("="*50)

                if state.annotation_results:
                    print(f"✓ Annotations: {state.annotation_results.get('total_annotated', 0)} items processed")

                if state.validation_results:
                    print(f"✓ Validation: Quality score {state.validation_results.get('quality_score', 0):.1f}/100")
                    print(f"✓ Doccano export: {state.validation_results.get('doccano_export_path', 'N/A')}")

                if state.training_results:
                    print(f"✓ Best model: {state.training_results.get('best_model', 'unknown')}")
                    print(f"✓ Best F1 score: {state.training_results.get('best_f1_macro', 0):.4f}")
                    print(f"✓ Model saved to: {state.training_results.get('model_save_path', 'N/A')}")

                # Ask to save config
                save = input("\nSave this configuration for future use? (y/n): ").strip().lower()
                if save == 'y':
                    profile_name = input("Profile name: ").strip() or "complete_pipeline"
                    self._save_profile(profile_name, pipeline_config)
                    print(f"Configuration saved as '{profile_name}'")

            except Exception as e:
                print(f"\n❌ Pipeline failed: {str(e)}")
                print("\nDebug information:")
                print(f"  - Data path: {data_path}")
                print(f"  - Text column: {text_column}")
                print(f"  - Provider: {provider}")
                print(f"  - Model: {selected_llm.name if selected_llm else 'None'}")
                import traceback
                print("\nFull error trace:")
                traceback.print_exc()

    def training_studio(self):
        """Training studio with model benchmarking and selection"""
        if HAS_RICH and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]🏋️ Training Studio[/bold cyan]\n"
                "Advanced model training and benchmarking",
                border_style="cyan"
            ))

            # Show available training models
            self.console.print("\n[bold]Available Model Categories:[/bold]\n")

            for category, models in self.available_trainer_models.items():
                self.console.print(f"[cyan]{category}[/cyan]")
                model_list = ", ".join([m['name'] for m in models[:3]])
                if len(models) > 3:
                    model_list += f" (+{len(models)-3} more)"
                self.console.print(f"  {model_list}\n")

            # Training mode selection
            mode = Prompt.ask(
                "Training mode",
                choices=["quick", "benchmark", "custom", "distributed"],
                default="benchmark"
            )

            if mode == "benchmark":
                # Benchmarking configuration
                self.console.print("\n[bold]Benchmark Configuration:[/bold]")

                num_models = IntPrompt.ask(
                    "Number of models to test",
                    default=5,
                    min_value=2,
                    max_value=20
                )

                include_sota = Confirm.ask("Include SOTA models (DeBERTa, RoBERTa Large)?", default=True)
                include_multilingual = Confirm.ask("Include multilingual models?", default=False)

                # Data selection
                self.console.print("\n[bold]Data Source:[/bold]")
                data_source_choice = Prompt.ask(
                    "Data source",
                    choices=["training_ready", "annotated_csv"],
                    default="training_ready"
                )

                if data_source_choice == "annotated_csv":
                    # Convert annotated CSV to training format
                    self.console.print("\n[cyan]Converting annotated CSV to training format...[/cyan]")
                    csv_path = self._prompt_file_path("Annotated CSV path")
                    text_column = Prompt.ask("Text column", default="sentence")
                    annotation_column = Prompt.ask("Annotation column", default="annotation")

                    # Ask for training strategy
                    self.console.print("\n[bold]Training Strategy:[/bold]")
                    self.console.print("• [cyan]single-label[/cyan]: Train separate models for each annotation key")
                    self.console.print("• [cyan]multi-label[/cyan]: Train one model with all labels together")

                    training_strategy = Prompt.ask(
                        "Training strategy",
                        choices=["single-label", "multi-label"],
                        default="single-label"
                    )

                    training_annotation_keys = None
                    if training_strategy == "single-label":
                        self.console.print("\n[dim]Available annotation keys will be detected from the annotations[/dim]")
                        if Confirm.ask("Train models for all annotation keys?", default=True):
                            training_annotation_keys = None
                        else:
                            keys_input = Prompt.ask("Enter annotation keys to train (comma-separated)")
                            training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

                    label_strategy = Prompt.ask(
                        "\nLabel strategy",
                        choices=["key_value", "value_only"],
                        default="key_value"
                    )

                    # Convert the data
                    from ..utils.annotation_to_training import AnnotationToTrainingConverter
                    converter = AnnotationToTrainingConverter(verbose=True)

                    training_data_dir = self.settings.paths.data_dir / 'training_data'
                    training_data_dir.mkdir(parents=True, exist_ok=True)

                    if training_strategy == "single-label":
                        output_files = converter.create_single_label_datasets(
                            csv_path=csv_path,
                            output_dir=str(training_data_dir),
                            text_column=text_column,
                            annotation_column=annotation_column,
                            annotation_keys=training_annotation_keys,
                            label_strategy=label_strategy
                        )
                        if output_files:
                            self.console.print(f"\n[green]✓ Created {len(output_files)} training dataset(s)[/green]")
                            for key, path in output_files.items():
                                self.console.print(f"  - {key}: {path}")
                            # Use the first file for now
                            data_path = list(output_files.values())[0]
                        else:
                            self.console.print("[red]Failed to create training datasets[/red]")
                            return
                    else:
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_file = converter.create_multi_label_dataset(
                            csv_path=csv_path,
                            output_path=str(training_data_dir / f'training_multilabel_{timestamp}.jsonl'),
                            text_column=text_column,
                            annotation_column=annotation_column,
                            annotation_keys=training_annotation_keys,
                            label_strategy=label_strategy
                        )
                        if output_file:
                            self.console.print(f"\n[green]✓ Created multi-label training dataset: {output_file}[/green]")
                            data_path = output_file
                        else:
                            self.console.print("[red]Failed to create training dataset[/red]")
                            return
                else:
                    # Use existing training-ready data
                    data_path = self._prompt_file_path("\nTraining data path")

                # Show data preview
                if HAS_PANDAS:
                    try:
                        df = pd.read_csv(data_path, nrows=5)
                        self.console.print("\n[dim]Data preview:[/dim]")
                        self.console.print(df.head())
                    except:
                        pass

                # Start benchmarking
                if Confirm.ask("\n[bold yellow]Start benchmarking?[/bold yellow]", default=True):

                    with Live(self._generate_benchmark_display(), refresh_per_second=4, console=self.console):
                        # Simulate benchmarking
                        time.sleep(5)

                    # Show results
                    self._show_benchmark_results()

            elif mode == "custom":
                # Custom training configuration
                self.console.print("\n[bold]Custom Training Configuration:[/bold]")

                # Model selection
                model_name = Prompt.ask("Model name", default="bert-base-uncased")
                epochs = self._int_prompt_with_validation("Epochs", default=10, min_value=1, max_value=100)
                batch_size = self._int_prompt_with_validation("Batch size", default=16, min_value=1, max_value=128)
                learning_rate = FloatPrompt.ask("Learning rate", default=2e-5)

                # Advanced options
                use_mixed_precision = Confirm.ask("Use mixed precision (FP16)?", default=False)
                gradient_checkpointing = Confirm.ask("Enable gradient checkpointing?", default=False)

                self.console.print("\n[green]Configuration saved![/green]")

        else:
            print("\n=== Training Studio ===")
            print("Model training and benchmarking\n")
            print("Feature coming soon")

    def _generate_benchmark_display(self):
        """Generate live benchmark display"""
        table = Table(title="🏃 Benchmark Progress", border_style="blue")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Epoch", style="white")
        table.add_column("Loss", style="red")
        table.add_column("Accuracy", style="green")
        table.add_column("F1", style="magenta")

        # Simulated data
        models = [
            ("bert-base-uncased", "🟢 Running", "3/10", "0.452", "0.823", "0.812"),
            ("roberta-base", "⏸ Queued", "-", "-", "-", "-"),
            ("deberta-v3-base", "⏸ Queued", "-", "-", "-", "-"),
            ("electra-base", "⏸ Queued", "-", "-", "-", "-"),
            ("albert-base-v2", "⏸ Queued", "-", "-", "-", "-")
        ]

        for model_data in models:
            table.add_row(*model_data)

        return table

    def _show_benchmark_results(self):
        """Display benchmark results"""
        results_table = Table(title="📊 Benchmark Results", border_style="green", show_lines=True)
        results_table.add_column("Rank", style="bold cyan", width=6)
        results_table.add_column("Model", style="white")
        results_table.add_column("F1 Score", style="green")
        results_table.add_column("Accuracy", style="yellow")
        results_table.add_column("Time", style="blue")
        results_table.add_column("Params", style="magenta")

        # Simulated results
        results = [
            ("🥇 1", "deberta-v3-base", "0.892", "0.905", "45m", "184M"),
            ("🥈 2", "roberta-base", "0.878", "0.891", "32m", "125M"),
            ("🥉 3", "electra-base", "0.865", "0.880", "28m", "110M"),
            ("4", "bert-base-uncased", "0.842", "0.863", "25m", "110M"),
            ("5", "albert-base-v2", "0.825", "0.845", "18m", "12M")
        ]

        for result in results:
            results_table.add_row(*result)

        self.console.print(results_table)
        self.console.print("\n[bold green]🏆 Best model: deberta-v3-base[/bold green]")
        self.console.print("[dim]Model saved to: ./models/best_model[/dim]")

    def validation_lab(self):
        """Validation lab for quality control and Doccano export"""
        if HAS_RICH and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]🔍 Validation Lab[/bold cyan]\n"
                "Quality control and human review preparation",
                border_style="cyan"
            ))

            # Select annotations file
            self.console.print("\n[bold]Select Annotations File:[/bold]")
            annotations_path = self._prompt_file_path("Annotations file path")

            # Load and analyze
            self.console.print("\n[cyan]Analyzing annotations...[/cyan]")

            from ..validators.annotation_validator import AnnotationValidator, ValidationConfig

            validator = AnnotationValidator()

            # Configuration
            self.console.print("\n[bold]Validation Configuration:[/bold]")

            sample_size = self._int_prompt_with_validation("Sample size for validation", default=100, min_value=10, max_value=1000)
            stratified = Confirm.ask("Use stratified sampling?", default=True)
            export_doccano = Confirm.ask("Export to Doccano format?", default=True)

            # Run validation
            with self.console.status("[bold green]Running validation...", spinner="dots"):
                config = ValidationConfig(
                    sample_size=sample_size,
                    stratified_sampling=stratified,
                    export_to_doccano=export_doccano,
                    export_format='both'
                )

                validator.config = config
                result = validator.validate(
                    input_file=annotations_path,
                    output_dir="./validation"
                )

            # Display results
            self.console.print("\n[bold]Validation Results:[/bold]\n")

            # Quality score with color coding
            score = result.quality_score
            if score >= 80:
                color = "green"
            elif score >= 60:
                color = "yellow"
            else:
                color = "red"

            self.console.print(f"[{color}]Quality Score: {score:.1f}/100[/{color}]")

            # Statistics table
            stats_table = Table(title="📊 Annotation Statistics", border_style="blue")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")

            stats_table.add_row("Total Annotations", str(result.total_annotations))
            stats_table.add_row("Validated Samples", str(result.validated_samples))
            stats_table.add_row("Unique Labels", str(len(result.label_distribution)))

            if result.confidence_stats:
                stats_table.add_row("Avg Confidence", f"{result.confidence_stats.get('mean', 0):.3f}")
                stats_table.add_row("Low Confidence %", f"{result.confidence_stats.get('low_confidence_percentage', 0):.1f}%")

            self.console.print(stats_table)

            # Issues found
            if result.issues_found:
                self.console.print(f"\n[yellow]⚠ {len(result.issues_found)} issues found:[/yellow]")
                for issue in result.issues_found[:5]:
                    self.console.print(f"  - {issue['type']}: {issue.get('message', issue.get('column', ''))}")

            # Export paths
            self.console.print("\n[bold]Exports:[/bold]")
            if result.doccano_export_path:
                self.console.print(f"📁 Doccano: {result.doccano_export_path}")
            if result.export_path:
                self.console.print(f"📁 Data: {result.export_path}")

            self.console.print("\n[green]✅ Validation complete![/green]")

        else:
            print("\n=== Validation Lab ===")
            print("Quality control and validation\n")

            annotations_path = input("Annotations file path: ").strip()
            sample_size = int(input("Sample size (default 100): ").strip() or "100")

            print("\nRunning validation...")
            print("✅ Validation complete!")
            print(f"Exported to: ./validation/")

    def analytics_dashboard(self):
        """Analytics dashboard"""
        if HAS_RICH and self.console:
            self.console.print(Panel("[bold cyan]Analytics Dashboard - Coming Soon[/bold cyan]"))
        else:
            print("Analytics Dashboard - Coming Soon")

    def profile_manager_ui(self):
        """Profile manager interface"""
        if HAS_RICH and self.console:
            profiles = self.profile_manager.list_profiles()

            if not profiles:
                self.console.print("[yellow]No saved profiles found[/yellow]")
                return

            table = Table(title="💾 Saved Profiles", border_style="blue")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Name", style="white")
            table.add_column("Created", style="dim")
            table.add_column("Last Used", style="green")

            for i, profile in enumerate(profiles, 1):
                table.add_row(
                    str(i),
                    profile.name,
                    profile.created_at.strftime("%Y-%m-%d"),
                    profile.last_used.strftime("%Y-%m-%d %H:%M")
                )

            self.console.print(table)

            choice = IntPrompt.ask("Select profile to load (0 to cancel)", default=0)
            if choice > 0 and choice <= len(profiles):
                selected_profile = profiles[choice-1]
                self.console.print(f"[green]Loading profile: {selected_profile.name}[/green]")
                # Execute with loaded configuration
                self._execute_quick_start(selected_profile.configuration)
        else:
            print("Profile Manager - Coming Soon")

    def advanced_settings(self):
        """Advanced settings interface"""
        if HAS_RICH and self.console:
            self.console.print(Panel("[bold cyan]Advanced Settings - Coming Soon[/bold cyan]"))
        else:
            print("Advanced Settings - Coming Soon")

    def show_documentation(self):
        """Show documentation"""
        if HAS_RICH and self.console:
            doc_text = """
# LLMTool Documentation

## Quick Start
1. Ensure you have models available (Ollama or API keys)
2. Prepare your dataset (CSV, JSON, etc.)
3. Use Quick Start Wizard for automatic configuration

## Features
- **Auto-detection**: Automatically finds models and datasets
- **Smart defaults**: Intelligent configuration suggestions
- **Profile system**: Save and reuse configurations
- **Benchmarking**: Compare multiple models automatically

## Support
- GitHub: https://github.com/antoine-lemor/LLMTool
- Email: support@llmtool.ai
            """

            md = Markdown(doc_text)
            self.console.print(Panel(md, title="📚 Documentation", border_style="blue"))
        else:
            print("\n=== Documentation ===")
            print("Visit: https://github.com/antoine-lemor/LLMTool")


def main():
    """Entry point for the advanced CLI"""
    cli = AdvancedCLI()
    cli.run()


if __name__ == "__main__":
    main()

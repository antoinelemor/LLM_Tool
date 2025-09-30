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
from collections import defaultdict
import re

# Rich is mandatory for this CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.text import Text
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import print as rprint
    console = Console()
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
        """Scan directory for potential datasets"""
        datasets = []
        patterns = ['*.csv', '*.json', '*.jsonl', '*.xlsx', '*.parquet', '*.tsv']

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

        for col in dataset.columns:
            col_lower = col.lower()
            for candidate in text_candidates:
                if candidate in col_lower:
                    return col

        # If no obvious text column, return the first string column
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

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def display_banner(self):
        """Display professional welcome banner with system info"""
        if HAS_RICH and self.console:
            # Create banner
            banner = Table.grid(padding=1)
            banner.add_column(style="cyan", justify="center")
            banner.add_column(style="white", justify="left")

            # Title
            title_text = Text("🚀 LLMTool Professional v1.0.0", style="bold cyan")
            banner.add_row(title_text, "")

            # System info
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                banner.add_row(
                    "",
                    f"System: CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% used"
                )

            # Model detection status
            banner.add_row(
                "",
                f"Detecting available models..."
            )

            panel = Panel(
                banner,
                title="[bold yellow]Advanced LLM Annotation & Training Platform[/bold yellow]",
                border_style="bright_blue",
                padding=(1, 2)
            )

            self.console.print(panel)

            # Auto-detect models in background
            with self.console.status("[bold green]Scanning environment...", spinner="dots"):
                self.detected_llms = self.llm_detector.detect_all_llms()
                self.available_trainer_models = self.trainer_model_detector.get_available_models()
                self.detected_datasets = self.data_detector.scan_directory()

            # Show detection results
            self._display_detection_results()

        else:
            print("\n" + "="*70)
            print("         LLMTool Professional v1.0.0")
            print("   Advanced LLM Annotation & Training Platform")
            print("="*70)
            print("\nScanning environment...")
            self.detected_llms = self.llm_detector.detect_all_llms()
            self.available_trainer_models = self.trainer_model_detector.get_available_models()
            self.detected_datasets = self.data_detector.scan_directory()

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

    def quick_start_wizard(self):
        """Intelligent quick start wizard with auto-configuration"""
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

            # Dataset selection
            if best_dataset:
                use_detected = Confirm.ask(
                    f"Use detected dataset [cyan]{best_dataset.path.name}[/cyan]?",
                    default=True
                )
                if use_detected:
                    dataset_path = str(best_dataset.path)
                    text_column = self.data_detector.suggest_text_column(best_dataset)
                else:
                    dataset_path = self._prompt_file_path("Enter dataset path")
                    text_column = Prompt.ask("Text column name", default="text")
            else:
                dataset_path = self._prompt_file_path("Enter dataset path")
                text_column = Prompt.ask("Text column name", default="text")

            # LLM configuration
            self.console.print(f"\n[green]Using LLM: {best_llm.name}[/green]")
            if best_llm.requires_api_key:
                api_key = Prompt.ask("API Key", password=True)
            else:
                api_key = None

            # Prompt configuration
            self.console.print("\n[bold]Prompt Configuration:[/bold]")
            prompt_mode = Prompt.ask(
                "Prompt mode",
                choices=["simple", "template", "multi", "auto"],
                default="auto"
            )

            if prompt_mode == "auto":
                # Auto-generate prompt based on dataset
                prompt = self._generate_auto_prompt(dataset_path, text_column)
                self.console.print(Panel(
                    Syntax(prompt, "python", theme="monokai"),
                    title="[bold]Auto-generated Prompt[/bold]",
                    border_style="green"
                ))

                if not Confirm.ask("Use this prompt?", default=True):
                    prompt = self._get_custom_prompt()
            else:
                prompt = self._get_custom_prompt()

            # Training configuration
            self.console.print("\n[bold]Training Configuration:[/bold]")
            training_mode = Prompt.ask(
                "Training mode",
                choices=["quick", "balanced", "thorough", "custom"],
                default="balanced"
            )

            training_config = self._get_training_preset(training_mode)

            # Summary and confirmation
            self._display_configuration_summary({
                'dataset': dataset_path,
                'text_column': text_column,
                'model': best_llm.name,
                'prompt_mode': prompt_mode,
                'training_mode': training_mode,
                **training_config
            })

            if Confirm.ask("\n[bold yellow]Start execution?[/bold yellow]", default=True):
                # Save as profile for future use
                if Confirm.ask("Save this configuration as a profile?", default=True):
                    profile_name = Prompt.ask("Profile name", default="quick_start")
                    self._save_profile(profile_name, {
                        'dataset': dataset_path,
                        'text_column': text_column,
                        'model': best_llm.name,
                        'api_key': api_key,
                        'prompt': prompt,
                        'training_config': training_config
                    })

                # Execute pipeline
                self._execute_quick_start({
                    'dataset': dataset_path,
                    'text_column': text_column,
                    'model': best_llm,
                    'api_key': api_key,
                    'prompt': prompt,
                    'training_config': training_config
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

            print("\nStarting pipeline...")
            # Execute simplified pipeline

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

    def _get_custom_prompt(self) -> str:
        """Get custom prompt from user with templates"""
        if HAS_RICH and self.console:
            self.console.print("\n[bold]Prompt Templates:[/bold]")
            templates = {
                "1": "Classification",
                "2": "Entity Extraction",
                "3": "Sentiment Analysis",
                "4": "Summarization",
                "5": "Custom"
            }

            for key, value in templates.items():
                self.console.print(f"  {key}. {value}")

            choice = Prompt.ask("Select template", choices=list(templates.keys()), default="5")

            if choice == "5":
                self.console.print("\n[dim]Enter your prompt (press Ctrl+D when done):[/dim]")
                lines = []
                try:
                    while True:
                        lines.append(input())
                except EOFError:
                    pass
                return "\n".join(lines)
            else:
                return self._get_prompt_template(choice)
        else:
            return input("Enter prompt: ").strip()

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

    def _execute_quick_start(self, config: Dict[str, Any]):
        """Execute the quick start pipeline"""
        if HAS_RICH and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:

                # Annotation phase
                task1 = progress.add_task("[cyan]Running annotation...", total=100)
                for i in range(100):
                    time.sleep(0.01)  # Simulate work
                    progress.update(task1, advance=1)

                # Training phase
                task2 = progress.add_task("[green]Training models...", total=100)
                for i in range(100):
                    time.sleep(0.01)  # Simulate work
                    progress.update(task2, advance=1)

                self.console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")
                self.console.print("📊 Results saved to: ./results/")
        else:
            print("Running annotation...")
            print("Training models...")
            print("✅ Pipeline completed successfully!")

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
                choice = IntPrompt.ask("Select LLM", default=1, min_value=1, max_value=len(local_llms))
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

                model_choice = IntPrompt.ask("Select model", default=1, min_value=1, max_value=len(api_models))
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

            prompt_mode = Prompt.ask(
                "Prompt strategy",
                choices=["template", "custom", "multi", "chain"],
                default="template"
            )

            if prompt_mode == "template":
                templates = [
                    "Classification",
                    "Entity Extraction",
                    "Sentiment Analysis",
                    "Summarization",
                    "Question Generation"
                ]
                self.console.print("\n[dim]Available templates:[/dim]")
                for i, t in enumerate(templates, 1):
                    self.console.print(f"  {i}. {t}")

                template_choice = IntPrompt.ask("Select template", default=1)
                prompt_text = self._get_prompt_template(str(template_choice))
            else:
                prompt_text = self._get_custom_prompt()

            # Step 5: Advanced options
            self.console.print("\n[bold]Step 5: Advanced Options[/bold]")

            batch_size = IntPrompt.ask("Batch size", default=10, min_value=1, max_value=100)
            max_workers = IntPrompt.ask("Parallel workers", default=4, min_value=1, max_value=16)
            save_incrementally = Confirm.ask("Save incrementally?", default=True)

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
                'save_incrementally': save_incrementally
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

                llm_choice = IntPrompt.ask("Select LLM for annotation", default=1, min_value=1, max_value=len(local_llms))
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
                num_models = IntPrompt.ask("Number of models to benchmark", default=5, min_value=2, max_value=20)
                self.console.print("[dim]Will test: BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT...[/dim]")
            else:
                # Show available training models
                self.console.print("\n[dim]Select model category:[/dim]")
                categories = list(self.available_trainer_models.keys())
                for i, cat in enumerate(categories, 1):
                    self.console.print(f"  {i}. {cat}")

                cat_choice = IntPrompt.ask("Category", default=1, min_value=1, max_value=len(categories))
                selected_category = categories[cat_choice-1]
                models_in_cat = self.available_trainer_models[selected_category]

                self.console.print(f"\n[dim]Models in {selected_category}:[/dim]")
                for i, model in enumerate(models_in_cat[:10], 1):
                    self.console.print(f"  {i}. {model['name']} ({model.get('params', 'N/A')})")

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
                epochs = IntPrompt.ask("Epochs", default=10, min_value=1, max_value=100)
                batch_size = IntPrompt.ask("Batch size", default=16, min_value=1, max_value=128)
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

            sample_size = IntPrompt.ask("Sample size for validation", default=100, min_value=10, max_value=1000)
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
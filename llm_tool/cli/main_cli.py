#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
main_cli.py

MAIN OBJECTIVE:
---------------
This script provides the main Command Line Interface for the LLMTool package,
offering an aesthetic and unified interface for all pipeline operations.

Dependencies:
-------------
- sys
- os
- argparse
- rich
- inquirer
- typing

MAIN FEATURES:
--------------
1) Interactive menu system for pipeline operations
2) Support for LLM annotation with local models or APIs
3) Model training orchestration with benchmarking
4) Validation and export utilities
5) Multilingual support with automatic language detection

Author:
-------
Antoine Lemor
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter
import json
import logging

import pandas as pd

# Rich for better CLI interface
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich import print as rprint
    from rich.layout import Layout
    from rich.live import Live
    from rich.tree import Tree
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' library for better CLI interface: pip install rich")

# Import internal modules
from ..config.settings import Settings
from ..pipelines.pipeline_controller import PipelineController
from ..utils.language_detector import LanguageDetector
from ..annotators.prompt_manager import PromptManager


class LLMToolCLI:
    """Main CLI interface for LLMTool package"""

    def __init__(self):
        """Initialize the CLI with default settings"""
        self.console = Console() if HAS_RICH else None
        self.settings = Settings()
        self.pipeline_controller = PipelineController()
        self.language_detector = LanguageDetector()
        self.prompt_manager = PromptManager()
        self.last_annotation_config: Dict[str, Any] = {}

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    def display_banner(self):
        """Display the welcome banner"""
        if HAS_RICH and self.console:
            banner_text = Text()
            banner_text.append("╔══════════════════════════════════════════════════════════════╗\n", style="bold cyan")
            banner_text.append("║                                                              ║\n", style="bold cyan")
            banner_text.append("║                    ", style="bold cyan")
            banner_text.append("🤖 LLMTool v1.0.0 🤖", style="bold yellow")
            banner_text.append("                     ║\n", style="bold cyan")
            banner_text.append("║                                                              ║\n", style="bold cyan")
            banner_text.append("║          ", style="bold cyan")
            banner_text.append("State-of-the-Art LLM Annotation & Training", style="white")
            banner_text.append("          ║\n", style="bold cyan")
            banner_text.append("║                                                              ║\n", style="bold cyan")
            banner_text.append("╚══════════════════════════════════════════════════════════════╝", style="bold cyan")

            self.console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))
            self.console.print()
        else:
            print("\n" + "="*60)
            print("                 LLMTool v1.0.0")
            print("     State-of-the-Art LLM Annotation & Training")
            print("="*60 + "\n")

    def get_main_menu_choice(self) -> str:
        """Display main menu and get user choice"""
        if HAS_RICH and self.console:
            self.console.print("[bold cyan]Main Menu[/bold cyan]")
            self.console.print()

            options = [
                "1. 📝 Annotate data with LLM (local or API)",
                "2. 🚀 Run complete pipeline (annotation → training → deployment)",
                "3. 🏋️ Train models from existing annotations",
                "4. 🔍 Validate annotations (export to Doccano format)",
                "5. 📊 Benchmark models on dataset",
                "6. 🌍 Configure language settings",
                "7. ⚙️ Configure global settings",
                "8. ℹ️ About LLMTool",
                "9. ❌ Exit"
            ]

            for option in options:
                self.console.print(f"  {option}")

            self.console.print()
            choice = Prompt.ask(
                "[bold yellow]Select an option[/bold yellow]",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                default="1"
            )
        else:
            print("Main Menu")
            print("---------")
            print("1. Annotate data with LLM (local or API)")
            print("2. Run complete pipeline (annotation → training → deployment)")
            print("3. Train models from existing annotations")
            print("4. Validate annotations (export to Doccano format)")
            print("5. Benchmark models on dataset")
            print("6. Configure language settings")
            print("7. Configure global settings")
            print("8. About LLMTool")
            print("9. Exit")
            print()
            choice = input("Select an option (1-9): ").strip()

        return choice

    def handle_annotation_workflow(self):
        """Handle the LLM annotation workflow"""
        if HAS_RICH and self.console:
            self.console.print(Panel("[bold cyan]LLM Annotation Module[/bold cyan]", border_style="cyan"))
            self.console.print()

            # Ask for annotation mode
            self.console.print("[bold]Select annotation mode:[/bold]")
            mode = Prompt.ask(
                "Mode",
                choices=["local", "api"],
                default="local"
            )

            if mode == "api":
                # API configuration
                api_provider = Prompt.ask(
                    "API Provider",
                    choices=["openai", "anthropic", "google", "custom"],
                    default="openai"
                )
                api_key = Prompt.ask("API Key", password=True)
                model_name = Prompt.ask("Model name", default="gpt-4")
            else:
                # Local model configuration
                model_provider = Prompt.ask(
                    "Local model provider",
                    choices=["ollama", "llamacpp", "transformers"],
                    default="ollama"
                )
                model_name = Prompt.ask("Model name", default="llama3.2")

            # Data source configuration
            self.console.print("\n[bold]Data source configuration:[/bold]")
            data_format = Prompt.ask(
                "Data format",
                choices=["csv", "json", "jsonl", "excel", "parquet", "postgresql"],
                default="csv"
            )

            if data_format == "postgresql":
                # Database configuration
                db_config = {
                    "host": Prompt.ask("Database host", default="localhost"),
                    "port": IntPrompt.ask("Database port", default=5432),
                    "database": Prompt.ask("Database name"),
                    "user": Prompt.ask("Username"),
                    "password": Prompt.ask("Password", password=True),
                    "table": Prompt.ask("Table name")
                }
                text_column = Prompt.ask("Text column name")
            else:
                # File configuration
                file_path = Prompt.ask("File path")
                text_column = Prompt.ask("Text column name", default="text")

            # Prompt configuration
            self.console.print("\n[bold]Prompt configuration:[/bold]")
            use_multiple_prompts = Confirm.ask("Use multiple prompts for each text?", default=False)

            if use_multiple_prompts:
                load_from_folder = Confirm.ask("Load all prompts from a folder?", default=False)
                if load_from_folder:
                    prompts_folder = Prompt.ask("Prompts folder path")
                else:
                    num_prompts = IntPrompt.ask("Number of prompts", default=1, min=1, max=10)
                    prompts = []
                    for i in range(num_prompts):
                        prompt_path = Prompt.ask(f"Path to prompt {i+1}")
                        prompts.append(prompt_path)
            else:
                prompt_path = Prompt.ask("Path to prompt file")

            # Export configuration
            self.console.print("\n[bold]Export configuration:[/bold]")
            export_validation_sample = Confirm.ask(
                "Export validation sample to JSONL (Doccano format)?",
                default=False
            )

            if export_validation_sample:
                sample_size = IntPrompt.ask("Sample size", default=100, min=10, max=1000)
                export_path = Prompt.ask(
                    "Export path",
                    default=str(Path.cwd() / "validation" / "sample.jsonl")
                )

            # Continue to training?
            continue_to_training = Confirm.ask(
                "\n[bold yellow]Continue to model training after annotation?[/bold yellow]",
                default=False
            )

            # Start annotation
            with self.console.status("[bold green]Starting annotation process...", spinner="dots"):
                # Prepare configuration
                config = {
                    'mode': 'file',
                    'data_source': 'file',
                    'annotation_mode': mode,
                    'annotation_provider': api_provider if mode == 'api' else model_provider,
                    'annotation_model': model_name,
                    'api_key': api_key if mode == 'api' else None,
                    'data_format': data_format,
                    'text_column': text_column
                }

                # Add data source specific config
                if data_format == 'postgresql':
                    config.update(db_config)
                else:
                    config['file_path'] = file_path

                # Add prompt configuration
                if use_multiple_prompts:
                    if load_from_folder:
                        config['prompts_folder'] = prompts_folder
                    else:
                        config['prompts'] = prompts
                else:
                    config['prompt_path'] = prompt_path

                # Add export configuration
                if export_validation_sample:
                    config['export_sample'] = True
                    config['sample_size'] = sample_size
                    config['export_path'] = export_path

                # Run annotation
                results = self.pipeline_controller.run_annotation(config)

                self.console.print(f"\n[green]✅ Annotation completed![/green]")
                if results:
                    self.console.print(f"📝 Processed {results.get('total_annotated', 0)} items")
                    if results.get('output_file'):
                        self.console.print(f"💾 Results saved to: {results['output_file']}")

            # Continue to training if requested
            if continue_to_training:
                if Confirm.ask("\n[bold yellow]Ready to start training?[/bold yellow]", default=True):
                    training_config = self.get_training_config()
                    training_config['input_file'] = results.get('output_file')

                    with self.console.status("[bold green]Training models...", spinner="dots"):
                        training_results = self.pipeline_controller.run_training(training_config)

                    self.console.print(f"\n[green]✅ Training completed![/green]")
                    if training_results:
                        self.console.print(f"🏆 Best model: {training_results.get('best_model', 'unknown')}")
                        self.console.print(f"📊 Accuracy: {training_results.get('best_accuracy', 0):.2%}")

        else:
            # Fallback to simple input
            print("\n=== LLM Annotation Module ===\n")
            # Simplified version without rich

            # Get annotation configuration
            mode = input("Mode (local/api): ").strip() or 'local'

            if mode == 'api':
                api_provider = input("API Provider (openai/anthropic/google): ").strip() or 'openai'
                api_key = input("API Key: ").strip()
                model_name = input("Model name: ").strip() or 'gpt-4'
            else:
                model_provider = input("Local provider (ollama/llamacpp): ").strip() or 'ollama'
                model_name = input("Model name: ").strip() or 'llama3.2'
                api_key = None
                api_provider = None

            # Data source configuration
            print("\nData source configuration:")
            data_format = input("Data format (csv/json/excel): ").strip() or 'csv'
            file_path = input("File path: ").strip()
            text_column = input("Text column name: ").strip() or 'text'

            # Prompt configuration
            print("\nPrompt configuration:")
            use_multiple = input("Use multiple prompts? (y/n): ").strip().lower() == 'y'
            if use_multiple:
                num_prompts = int(input("Number of prompts: ").strip() or '1')
                prompts = []
                for i in range(num_prompts):
                    prompt_path = input(f"Path to prompt {i+1}: ").strip()
                    prompts.append(prompt_path)
            else:
                prompt_path = input("Path to prompt file: ").strip()
                prompts = None

            # Run annotation
            print("\nStarting annotation process...")

            config = {
                'mode': 'file',
                'data_source': 'file',
                'annotation_mode': mode,
                'annotation_provider': api_provider if mode == 'api' else model_provider,
                'annotation_model': model_name,
                'api_key': api_key,
                'data_format': data_format,
                'file_path': file_path,
                'text_column': text_column
            }

            if use_multiple:
                config['prompts'] = prompts
            else:
                config['prompt_path'] = prompt_path

            try:
                results = self.pipeline_controller.run_annotation(config)
                print(f"\n✅ Annotation completed!")
                if results:
                    print(f"Processed {results.get('total_annotated', 0)} items")
                    if results.get('output_file'):
                        print(f"Results saved to: {results['output_file']}")
            except Exception as e:
                print(f"\n❌ Annotation failed: {str(e)}")

    def handle_pipeline_workflow(self):
        """Handle the complete pipeline workflow"""
        if HAS_RICH and self.console:
            self.console.print(Panel("[bold cyan]Complete Pipeline Workflow[/bold cyan]", border_style="cyan"))
            self.console.print()

            # Pipeline configuration
            self.console.print("[bold]Pipeline Configuration[/bold]")

            # Step 1: Annotation settings
            self.console.print("\n[yellow]Step 1: Annotation[/yellow]")
            annotation_config = self.get_annotation_config()

            # Step 2: Validation settings
            self.console.print("\n[yellow]Step 2: Validation[/yellow]")
            validation_config = self.get_validation_config()

            # Step 3: Training settings
            self.console.print("\n[yellow]Step 3: Model Training[/yellow]")
            training_config = self.get_training_config()

            # Step 4: Deployment settings
            self.console.print("\n[yellow]Step 4: Deployment[/yellow]")
            deployment_config = self.get_deployment_config()

            # Confirmation
            self.console.print("\n[bold]Pipeline Summary:[/bold]")
            self.display_pipeline_summary(
                annotation_config,
                validation_config,
                training_config,
                deployment_config
            )

            if Confirm.ask("\n[bold yellow]Start pipeline execution?[/bold yellow]", default=True):
                self.execute_pipeline(
                    annotation_config,
                    validation_config,
                    training_config,
                    deployment_config
                )
        else:
            print("\n=== Complete Pipeline Workflow ===\n")
            # Simplified version without rich

            # Get annotation configuration
            print("Step 1: Annotation Configuration")
            annotation_config = {
                'mode': input("Mode (local/api): ").strip() or 'local',
                'provider': input("Provider (ollama/openai): ").strip() or 'ollama',
                'model': input("Model name: ").strip() or 'llama3.2',
                'data_format': input("Data format (csv/json): ").strip() or 'csv',
                'data_path': input("Data file path: ").strip(),
                'text_column': input("Text column name: ").strip() or 'text'
            }
            if annotation_config['mode'] == 'api':
                annotation_config['api_key'] = input("API Key: ").strip()

            # Get validation configuration
            print("\nStep 2: Validation Configuration")
            validation_config = {
                'enable_validation': input("Enable validation? (y/n): ").strip().lower() == 'y',
                'sample_size': 100,
                'export_format': 'jsonl',
                'export_to_doccano': True
            }

            # Get training configuration
            print("\nStep 3: Training Configuration")
            training_config = {
                'benchmark_mode': input("Use benchmark mode? (y/n): ").strip().lower() == 'y',
                'models_to_test': 5,
                'auto_select_best': True,
                'max_epochs': int(input("Max epochs (default 10): ").strip() or '10'),
                'batch_size': int(input("Batch size (default 16): ").strip() or '16'),
                'learning_rate': float(input("Learning rate (default 2e-5): ").strip() or '2e-5')
            }

            # Get deployment configuration
            print("\nStep 4: Deployment Configuration")
            deployment_config = {
                'save_model': input("Save trained model? (y/n): ").strip().lower() == 'y',
                'run_inference': False
            }
            if deployment_config['save_model']:
                deployment_config['model_path'] = input("Model save path: ").strip() or 'models/trained_model'

            # Confirm and execute
            print("\n" + "="*50)
            print("Pipeline Configuration Summary:")
            print(f"  Annotation: {annotation_config['mode']} - {annotation_config['model']}")
            print(f"  Validation: {'Enabled' if validation_config['enable_validation'] else 'Disabled'}")
            print(f"  Training: {'Benchmark' if training_config['benchmark_mode'] else 'Single'} mode")
            print(f"  Deployment: {'Yes' if deployment_config['save_model'] else 'No'}")
            print("="*50 + "\n")

            if input("Start pipeline execution? (y/n): ").strip().lower() == 'y':
                self.execute_pipeline(
                    annotation_config,
                    validation_config,
                    training_config,
                    deployment_config
                )

    def get_annotation_config(self) -> Dict[str, Any]:
        """Get annotation configuration from user"""
        config: Dict[str, Any] = {}

        # Model selection
        use_api = Confirm.ask("Use API for annotation?", default=False)
        if use_api:
            config['mode'] = 'api'
            config['provider'] = Prompt.ask(
                "API Provider",
                choices=["openai", "anthropic", "google", "custom"],
                default="openai"
            )
            config['api_key'] = Prompt.ask("API Key", password=True)
            config['model'] = Prompt.ask("Model", default="gpt-4")
        else:
            config['mode'] = 'local'
            config['provider'] = Prompt.ask(
                "Local provider",
                choices=["ollama", "llamacpp", "transformers"],
                default="ollama"
            )
            config['model'] = Prompt.ask("Model", default="llama3.2")

        # Data configuration
        data_format = Prompt.ask(
            "Data format",
            choices=["csv", "excel", "parquet", "postgresql"],
            default="csv"
        )
        config['data_format'] = data_format

        if data_format == 'postgresql':
            config['db_config'] = {
                "host": Prompt.ask("Database host", default="localhost"),
                "port": IntPrompt.ask("Database port", default=5432),
                "database": Prompt.ask("Database name"),
                "user": Prompt.ask("Database user"),
                "password": Prompt.ask("Database password", password=True),
                "table": Prompt.ask("Table name")
            }
            config['data_path'] = None
        else:
            default_data_dir = self.settings.paths.data_dir
            default_path = default_data_dir / f"dataset.{data_format if data_format != 'csv' else 'csv'}"
            data_path = ""
            while not data_path:
                if default_path.exists():
                    candidate = Prompt.ask("Data path", default=str(default_path))
                else:
                    candidate = Prompt.ask("Data path")
                candidate = (candidate or "").strip()
                if not candidate:
                    if HAS_RICH and self.console:
                        self.console.print("[yellow]A data path is required.[/yellow]")
                    else:
                        print("Data path is required.")
                    continue
                if not Path(candidate).exists():
                    warning_msg = f"Data file not found at {candidate}"
                    if HAS_RICH and self.console:
                        self.console.print(f"[red]{warning_msg}[/red]")
                    else:
                        print(warning_msg)
                    continue
                data_path = candidate

            config['data_path'] = data_path

        config['text_column'] = Prompt.ask("Text column", default="text")
        config['annotation_column'] = Prompt.ask("Annotation column", default="annotation")

        # Prompt configuration
        default_prompt_path = self.settings.paths.prompts_dir / "prompt_EN_long.txt"
        default_prompt_str = str(default_prompt_path) if default_prompt_path.exists() else ""

        prompt_path = ""
        while not prompt_path:
            if default_prompt_str:
                candidate = Prompt.ask("Prompt file path", default=default_prompt_str)
            else:
                candidate = Prompt.ask("Prompt file path")
            candidate = (candidate or "").strip()
            if not candidate:
                if HAS_RICH and self.console:
                    self.console.print("[yellow]A prompt is required to run the annotation pipeline.[/yellow]")
                else:
                    print("Prompt path is required.")
                continue

            if not Path(candidate).exists():
                message = f"Prompt file not found at {candidate}"
                if HAS_RICH and self.console:
                    self.console.print(f"[red]{message}[/red]")
                else:
                    print(message)
                continue

            prompt_path = candidate

        try:
            prompt_text, expected_keys = self.prompt_manager.load_prompt(prompt_path)
        except Exception as exc:
            warning_msg = f"Failed to parse prompt structure ({exc}). Proceeding with raw prompt content."
            if HAS_RICH and self.console:
                self.console.print(f"[yellow]{warning_msg}[/yellow]")
            else:
                print(warning_msg)
            prompt_text = Path(prompt_path).read_text(encoding='utf-8')
            expected_keys = []

        prefix = ""
        if Confirm.ask("Prefix JSON keys with a label?", default=False):
            default_prefix = Path(prompt_path).stem.replace(" ", "_")
            prefix = Prompt.ask("Prefix", default=default_prefix)

        config['prompts'] = [
            {
                'prompt': prompt_text,
                'expected_keys': expected_keys,
                'prefix': prefix
            }
        ]
        config['prompt_path'] = prompt_path

        # Annotation sampling configuration
        sample_size = IntPrompt.ask(
            "Number of sentences to annotate for training (0 = all)",
            default=500,
            min=0
        )
        config['annotation_sample_size'] = sample_size if sample_size > 0 else None

        if config['annotation_sample_size']:
            random_sampling = Confirm.ask("Sample sentences randomly?", default=True)
            config['annotation_sampling_strategy'] = 'random' if random_sampling else 'head'
            config['annotation_sample_seed'] = IntPrompt.ask(
                "Random seed",
                default=42,
                min=0
            ) if random_sampling else 42
        else:
            config['annotation_sampling_strategy'] = 'head'
            config['annotation_sample_seed'] = 42

        config['output_format'] = 'csv'

        # Persist for later stages (training recommendations)
        self.last_annotation_config = config.copy()

        return config

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration from user"""
        config = {}

        config['enable_validation'] = Confirm.ask("Enable validation step?", default=True)
        if config['enable_validation']:
            config['sample_size'] = IntPrompt.ask("Validation sample size", default=100, min=10)
            config['export_format'] = Prompt.ask(
                "Export format",
                choices=["jsonl", "csv", "both"],
                default="jsonl"
            )
            config['export_to_doccano'] = Confirm.ask("Export to Doccano format?", default=True)

        return config

    def recommend_models_for_training(self, sample_size: int = 200) -> Dict[str, Any]:
        """Infer default models based on detected language distribution."""
        fallback = {
            'language_code': 'multilingual',
            'language_name': 'multilingual',
            'candidate_models': ['bert-base-multilingual-cased', 'xlm-roberta-base', 'mdeberta-v3-base'],
            'default_model': 'bert-base-multilingual-cased',
            'sampled': 0,
        }

        annotation_config = getattr(self, 'last_annotation_config', None)
        if not annotation_config:
            return fallback

        data_format = annotation_config.get('data_format')
        data_path = annotation_config.get('data_path')
        text_column = annotation_config.get('text_column', 'text')

        if data_format == 'postgresql' or not data_path:
            return fallback

        try:
            if data_format == 'csv':
                df = pd.read_csv(data_path)
            elif data_format == 'excel':
                df = pd.read_excel(data_path)
            elif data_format == 'parquet':
                df = pd.read_parquet(data_path)
            else:
                return fallback
        except Exception as exc:
            logging.warning("Failed to read data for language recommendation: %s", exc)
            return fallback

        if text_column not in df.columns:
            logging.warning("Column '%s' not found for language recommendation", text_column)
            return fallback

        texts = df[text_column].dropna().astype(str)
        if texts.empty:
            return fallback

        if len(texts) > sample_size:
            sample = texts.sample(sample_size, random_state=42)
        else:
            sample = texts

        detections = self.language_detector.detect_batch(sample.tolist(), parallel=False)
        language_counts = Counter()

        for detection in detections:
            if not detection:
                continue
            language = detection.get('language')
            confidence = detection.get('confidence', 0)
            if language and confidence >= 0.5:
                language_counts[language] += 1

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
            'sampled': len(sample),
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration from user"""
        config: Dict[str, Any] = {}

        recommendation = self.recommend_models_for_training()
        language_label = recommendation['language_name']
        suggested_models = recommendation['candidate_models']
        default_model = recommendation['default_model']

        if HAS_RICH and self.console:
            self.console.print(
                f"[bold cyan]Detected language:[/bold cyan] {language_label}"
                f" (sampled {recommendation['sampled']} texts)"
            )
            self.console.print(
                f"[bold cyan]Suggested models:[/bold cyan] {', '.join(suggested_models[:5])}"
            )
        else:
            print(f"Detected language: {language_label} (sample {recommendation['sampled']})")
            print(f"Suggested models: {', '.join(suggested_models[:5])}")

        config['benchmark_mode'] = Confirm.ask("Benchmark multiple models?", default=True)

        if config['benchmark_mode']:
            config['auto_select_best'] = Confirm.ask("Auto-select best model?", default=True)
            config['pause_for_validation'] = Confirm.ask(
                "Pause for manual validation before final selection?",
                default=False
            )
            default_models_str = ", ".join(suggested_models[:5]) if suggested_models else default_model
            models_input = Prompt.ask(
                "Models to benchmark (comma separated)",
                default=default_models_str
            )
            models = [model.strip() for model in models_input.split(',') if model.strip()]
            config['models_to_test'] = models or [default_model]
        else:
            config['model_type'] = Prompt.ask(
                "Model to fine-tune",
                default=default_model
            ).strip()

        config['max_epochs'] = IntPrompt.ask("Maximum epochs", default=10, min=1, max=100)
        config['batch_size'] = IntPrompt.ask("Batch size", default=16, min=1, max=128)
        config['learning_rate'] = float(Prompt.ask("Learning rate", default="2e-5"))

        # Persist recommendation for downstream summary
        config['detected_language'] = language_label

        return config

    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration from user"""
        config = {}

        config['save_model'] = Confirm.ask("Save trained model?", default=True)
        if config['save_model']:
            config['model_path'] = Prompt.ask(
                "Model save path",
                default=str(Path.cwd() / "models" / "trained_model")
            )

        config['run_inference'] = Confirm.ask("Run inference on new data?", default=False)
        if config['run_inference']:
            config['inference_data_path'] = Prompt.ask("Inference data path")
            config['save_predictions'] = Confirm.ask("Save predictions?", default=True)

        return config

    def display_pipeline_summary(self, annotation_config, validation_config,
                                training_config, deployment_config):
        """Display a summary of the pipeline configuration"""
        if HAS_RICH and self.console:
            table = Table(title="Pipeline Configuration Summary", border_style="cyan")
            table.add_column("Stage", style="cyan", no_wrap=True)
            table.add_column("Configuration", style="white")

            # Annotation summary
            ann_summary = f"Mode: {annotation_config['mode']}\n"
            ann_summary += f"Model: {annotation_config['model']}\n"
            ann_summary += f"Data: {annotation_config['data_format']}\n"
            sample_hint = annotation_config.get('annotation_sample_size')
            ann_summary += f"Sample: {sample_hint or 'all'}"
            table.add_row("Annotation", ann_summary)

            # Validation summary
            val_summary = f"Enabled: {validation_config['enable_validation']}\n"
            if validation_config['enable_validation']:
                val_summary += f"Sample size: {validation_config['sample_size']}\n"
                val_summary += f"Export: {validation_config['export_format']}"
            table.add_row("Validation", val_summary)

            # Training summary
            train_summary = f"Mode: {'Benchmark' if training_config['benchmark_mode'] else 'Single'}\n"
            detected_language = training_config.get('detected_language')
            if detected_language:
                train_summary += f"Language: {detected_language}\n"
            if training_config['benchmark_mode']:
                models_preview = ', '.join(training_config.get('models_to_test', [])[:3])
                train_summary += f"Models: {models_preview or 'n/a'}\n"
            else:
                train_summary += f"Model: {training_config.get('model_type', 'n/a')}\n"
            train_summary += f"Epochs: {training_config['max_epochs']}\n"
            train_summary += f"Batch size: {training_config['batch_size']}"
            table.add_row("Training", train_summary)

            # Deployment summary
            deploy_summary = f"Save model: {deployment_config['save_model']}\n"
            deploy_summary += f"Run inference: {deployment_config['run_inference']}"
            table.add_row("Deployment", deploy_summary)

            self.console.print(table)

    def execute_pipeline(self, annotation_config, validation_config,
                        training_config, deployment_config):
        """Execute the complete pipeline"""
        # Prepare unified configuration for pipeline controller
        pipeline_config = {
            # Data source configuration
            'mode': 'file',
            'data_source': annotation_config['data_format'],
            'data_format': annotation_config['data_format'],
            'file_path': annotation_config.get('data_path'),
            'db_config': annotation_config.get('db_config'),
            'text_column': annotation_config['text_column'],
            'annotation_column': annotation_config.get('annotation_column', 'annotation'),

            # Annotation configuration
            'run_annotation': True,
            'annotation_mode': annotation_config['mode'],
            'annotation_provider': annotation_config['provider'],
            'annotation_model': annotation_config['model'],
            'api_key': annotation_config.get('api_key'),
            'prompts': annotation_config.get('prompts'),
            'prompt_path': annotation_config.get('prompt_path'),
            'annotation_sample_size': annotation_config.get('annotation_sample_size'),
            'annotation_sampling_strategy': annotation_config.get('annotation_sampling_strategy'),
            'annotation_sample_seed': annotation_config.get('annotation_sample_seed'),
            'output_format': annotation_config.get('output_format', 'csv'),

            # Validation configuration
            'run_validation': validation_config['enable_validation'],
            'validation_sample_size': validation_config.get('sample_size', 100),
            'validation_export_format': validation_config.get('export_format', 'jsonl'),
            'export_to_doccano': validation_config.get('export_to_doccano', True),

            # Training configuration
            'run_training': True,
            'benchmark_mode': training_config['benchmark_mode'],
            'training_model_type': training_config.get('model_type', 'bert-base-multilingual-cased'),
            'models_to_test': training_config.get('models_to_test', 5),
            'auto_select_best': training_config.get('auto_select_best', True),
            'pause_for_validation': training_config.get('pause_for_validation', False),
            'max_epochs': training_config['max_epochs'],
            'batch_size': training_config['batch_size'],
            'learning_rate': training_config['learning_rate'],

            # Deployment configuration
            'run_deployment': deployment_config['save_model'],
            'deployment_path': deployment_config.get('model_path'),
            'run_inference': deployment_config.get('run_inference', False),
            'inference_data_path': deployment_config.get('inference_data_path')
        }

        if HAS_RICH and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                # Start pipeline execution
                task = progress.add_task("[cyan]Initializing pipeline...", total=None)

                try:
                    # Run the actual pipeline
                    pipeline_state = self.pipeline_controller.run_pipeline(pipeline_config)

                    # Update progress based on completed phases
                    progress.update(task, description="[green]Pipeline completed successfully!")

                    # Display results summary
                    self.console.print("\n[bold green]✅ Pipeline execution complete![/bold green]")

                    if pipeline_state.annotation_results:
                        self.console.print(f"📝 Annotations: {pipeline_state.annotation_results.get('total_annotated', 0)} items processed")

                    if pipeline_state.validation_results:
                        self.console.print(f"✓ Validation: {pipeline_state.validation_results.get('samples_validated', 0)} samples")

                    if pipeline_state.training_results:
                        self.console.print(f"🏆 Best Model: {pipeline_state.training_results.get('best_model', 'unknown')}")
                        self.console.print(f"📊 Accuracy: {pipeline_state.training_results.get('best_accuracy', 0):.2%}")

                    if pipeline_state.deployment_results:
                        self.console.print(f"🚀 Model deployed to: {pipeline_state.deployment_results.get('deployed_model_path')}")

                    # Show any warnings or errors
                    if pipeline_state.warnings:
                        self.console.print("\n[yellow]⚠ Warnings:[/yellow]")
                        for warning in pipeline_state.warnings:
                            self.console.print(f"  - {warning}")

                    if pipeline_state.errors:
                        self.console.print("\n[red]❌ Errors:[/red]")
                        for error in pipeline_state.errors:
                            self.console.print(f"  - Phase: {error['phase']}, Error: {error['error']}")

                except Exception as e:
                    progress.update(task, description=f"[red]Pipeline failed: {str(e)}")
                    self.console.print(f"\n[bold red]❌ Pipeline execution failed![/bold red]")
                    self.console.print(f"[red]Error: {str(e)}[/red]")
                    raise
        else:
            # Non-Rich fallback
            print("\nInitializing pipeline...")
            try:
                pipeline_state = self.pipeline_controller.run_pipeline(pipeline_config)

                print("\n✅ Pipeline execution complete!")

                if pipeline_state.annotation_results:
                    print(f"Annotations: {pipeline_state.annotation_results.get('total_annotated', 0)} items processed")

                if pipeline_state.validation_results:
                    print(f"Validation: {pipeline_state.validation_results.get('samples_validated', 0)} samples")

                if pipeline_state.training_results:
                    print(f"Best Model: {pipeline_state.training_results.get('best_model', 'unknown')}")
                    print(f"Accuracy: {pipeline_state.training_results.get('best_accuracy', 0):.2%}")

                if pipeline_state.deployment_results:
                    print(f"Model deployed to: {pipeline_state.deployment_results.get('deployed_model_path')}")

            except Exception as e:
                print(f"\n❌ Pipeline execution failed!")
                print(f"Error: {str(e)}")
                raise

    def handle_language_settings(self):
        """Handle language configuration"""
        if HAS_RICH and self.console:
            self.console.print(Panel("[bold cyan]Language Settings[/bold cyan]", border_style="cyan"))
            self.console.print()

            # Language detection options
            self.console.print("[bold]Language Detection Options:[/bold]")

            detection_mode = Prompt.ask(
                "Detection mode",
                choices=["auto", "manual", "column"],
                default="auto"
            )

            if detection_mode == "auto":
                self.console.print("[green]Automatic language detection enabled[/green]")
                confidence_threshold = float(
                    Prompt.ask("Confidence threshold (0-1)", default="0.8")
                )
                fallback_language = Prompt.ask(
                    "Fallback language",
                    choices=["en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ar"],
                    default="en"
                )
            elif detection_mode == "manual":
                language = Prompt.ask(
                    "Select language",
                    choices=["en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ar"],
                    default="en"
                )
                self.console.print(f"[green]Language set to: {language}[/green]")
            else:  # column
                column_name = Prompt.ask("Language column name", default="language")
                self.console.print(f"[green]Will use column '{column_name}' for language[/green]")

            # Save settings
            if Confirm.ask("\nSave language settings?", default=True):
                self.settings.update_language_settings({
                    'mode': detection_mode,
                    'confidence_threshold': confidence_threshold if detection_mode == 'auto' else None,
                    'fallback_language': fallback_language if detection_mode == 'auto' else None,
                    'language': language if detection_mode == 'manual' else None,
                    'column_name': column_name if detection_mode == 'column' else None
                })
                self.console.print("[green]Settings saved![/green]")

    def show_about(self):
        """Display information about LLMTool"""
        if HAS_RICH and self.console:
            about_text = Text()
            about_text.append("LLMTool v1.0.0\n\n", style="bold yellow")
            about_text.append("A State-of-the-Art Python Package for:\n", style="bold")
            about_text.append("• LLM-based text annotation (local & API)\n")
            about_text.append("• Automated model training with benchmarking\n")
            about_text.append("• Validation and quality control\n")
            about_text.append("• Large-scale data processing\n")
            about_text.append("• Multilingual support\n\n")
            about_text.append("Author: ", style="bold")
            about_text.append("Antoine Lemor\n")
            about_text.append("License: ", style="bold")
            about_text.append("MIT\n")
            about_text.append("GitHub: ", style="bold")
            about_text.append("https://github.com/antoine-lemor/LLMTool\n")

            self.console.print(Panel(about_text, title="About LLMTool", border_style="cyan"))
        else:
            print("\n=== About LLMTool ===")
            print("LLMTool v1.0.0")
            print("\nA State-of-the-Art Python Package for:")
            print("• LLM-based text annotation (local & API)")
            print("• Automated model training with benchmarking")
            print("• Validation and quality control")
            print("• Large-scale data processing")
            print("• Multilingual support")
            print("\nAuthor: Antoine Lemor")
            print("License: MIT")
            print("GitHub: https://github.com/antoine-lemor/LLMTool")

    def run(self):
        """Main run loop for the CLI"""
        self.display_banner()

        while True:
            try:
                choice = self.get_main_menu_choice()

                if choice == "1":
                    self.handle_annotation_workflow()
                elif choice == "2":
                    self.handle_pipeline_workflow()
                elif choice == "3":
                    # Training from existing annotations
                    self.console.print("[yellow]Training module coming soon...[/yellow]")
                elif choice == "4":
                    # Validation
                    self.console.print("[yellow]Validation module coming soon...[/yellow]")
                elif choice == "5":
                    # Benchmarking
                    self.console.print("[yellow]Benchmark module coming soon...[/yellow]")
                elif choice == "6":
                    self.handle_language_settings()
                elif choice == "7":
                    # Global settings
                    self.console.print("[yellow]Settings module coming soon...[/yellow]")
                elif choice == "8":
                    self.show_about()
                elif choice == "9":
                    if HAS_RICH and self.console:
                        self.console.print("\n[bold cyan]Thank you for using LLMTool! Goodbye! 👋[/bold cyan]\n")
                    else:
                        print("\nThank you for using LLMTool! Goodbye!\n")
                    sys.exit(0)

                if HAS_RICH and self.console:
                    self.console.print()
                    if not Confirm.ask("Return to main menu?", default=True):
                        self.console.print("\n[bold cyan]Thank you for using LLMTool! Goodbye! 👋[/bold cyan]\n")
                        sys.exit(0)
                    self.console.print()

            except KeyboardInterrupt:
                if HAS_RICH and self.console:
                    self.console.print("\n[yellow]Operation cancelled by user[/yellow]")
                    if Confirm.ask("Exit LLMTool?", default=False):
                        self.console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]\n")
                        sys.exit(0)
                else:
                    print("\nOperation cancelled by user")
                    response = input("Exit LLMTool? (y/n): ").lower()
                    if response == 'y':
                        print("\nGoodbye!\n")
                        sys.exit(0)
            except Exception as e:
                if HAS_RICH and self.console:
                    self.console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
                else:
                    print(f"\nError: {str(e)}")


def main():
    """Entry point for the CLI"""
    cli = LLMToolCLI()
    cli.run()


if __name__ == "__main__":
    main()

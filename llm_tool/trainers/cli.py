#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
cli.py

MAIN OBJECTIVE:
---------------
This script provides a comprehensive Command Line Interface for training orchestration,
offering interactive menus for model selection, benchmarking, and multi-label training
with full parameter configuration and language-aware model selection.

Dependencies:
-------------
- argparse (command-line argument parsing)
- pathlib (file path handling)
- LLMTool.multi_label_trainer (training functionality)
- LLMTool.benchmarking (model comparison)
- LLMTool.model_selector (model selection)

MAIN FEATURES:
--------------
1) Interactive CLI with guided menus for training configuration
2) Multi-label training mode with automatic model selection
3) Comprehensive benchmark mode for model comparison
4) Language-aware model selection based on data characteristics
5) Full parameter configuration (epochs, batch size, learning rate, etc.)
6) Automatic directory creation and organization
7) Detailed CSV and JSON logging of results
8) Support for both interactive and programmatic usage

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from collections import Counter
import argparse
import logging

from .multi_label_trainer import MultiLabelTrainer, TrainingConfig
from .benchmarking import BenchmarkRunner, BenchmarkConfig
from .model_selector import ModelSelector
from .bert_base import BertBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingCLI:
    """Command Line Interface for model training orchestration."""

    def __init__(
        self,
        data_dir: Path,
        models_dir: Path,
        logs_dir: Path,
        verbose: bool = True
    ):
        """
        Initialize the training CLI.

        Args:
            data_dir: Directory containing training data
            models_dir: Directory to save trained models
            logs_dir: Directory to save logs and metrics
            verbose: Enable verbose output
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.verbose = verbose

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        """
        Run the interactive CLI.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "="*70)
        print(" " * 15 + "🚀 AUGMENTED SOCIAL SCIENTIST TRAINING CLI")
        print(" " * 20 + "Advanced Model Training System")
        print("="*70)

        # Main menu
        mode = self._select_mode()

        if mode == "quit":
            print("\n👋 Goodbye!")
            return 0

        elif mode == "multi-label":
            return self._run_multilabel_training()

        elif mode == "benchmark":
            return self._run_benchmark_mode()

        elif mode == "legacy":
            print("\n⚠️  Legacy mode not implemented in CLI.")
            print("   Use multi-label mode for better results.")
            return 1

        return 0

    def _select_mode(self) -> str:
        """
        Select the training mode.

        Returns:
            Selected mode string
        """
        print("\n🎯 SELECT TRAINING MODE")
        print("-" * 40)
        print("[1] 📊 Multi-label Training (Recommended)")
        print("[2] 🏆 Benchmark Mode (Compare models)")
        print("[3] 📦 Legacy Mode (Separate folders)")
        print("[4] ❌ Quit")

        while True:
            choice = input("\nYour choice [1-4]: ").strip()
            if choice == "1":
                return "multi-label"
            elif choice == "2":
                return "benchmark"
            elif choice == "3":
                return "legacy"
            elif choice == "4":
                return "quit"
            else:
                print("❌ Invalid choice. Please select 1-4.")

    def _run_multilabel_training(self) -> int:
        """
        Run multi-label training mode.

        Returns:
            Exit code
        """
        print("\n" + "="*60)
        print("📊 MULTI-LABEL TRAINING MODE")
        print("="*60)

        # Select data file
        data_file = self._select_data_file()
        if not data_file:
            return 1

        # Get training configuration
        config = self._get_training_config()

        # Ask about benchmark
        use_benchmark = self._confirm("\n🏆 Run benchmark to select best model?")

        if use_benchmark:
            # Run benchmark first
            benchmark_epochs = self._get_integer(
                "Number of epochs for benchmark",
                default=10,
                min_val=1,
                max_val=50
            )

            # Ask about reinforced learning
            use_reinforced_bench = self._confirm(
                "\n⚡ Enable reinforced learning for poor performers?",
                default=True
            )

            # Ask about short sequence optimization
            print("\n📏 Your data may contain short sequences (individual sentences).")
            print("   Some models (Longformer, BigBird, ALBERT Large) are optimized for long documents.")
            optimize_short_bench = self._confirm(
                "Optimize parameters for short sequences?",
                default=True
            )

            # Detect languages in data
            print("\n🔍 Analyzing dataset...")
            languages = self._detect_data_languages(data_file)

            # Display and select models for benchmark
            selected_models = self._select_benchmark_models(languages)
            if not selected_models:
                print("\n❌ No models selected for benchmark")
                return 1

            # Pass benchmark options to config
            config.use_reinforced_in_benchmark = use_reinforced_bench if 'use_reinforced_bench' in locals() else False
            config.optimize_for_short_sequences = optimize_short_bench if 'optimize_short_bench' in locals() else False

            best_model_name = self._run_benchmark(
                data_file=data_file,
                config=config,
                benchmark_epochs=benchmark_epochs,
                test_all=False,
                models_to_test=selected_models
            )

            if not best_model_name:
                print("\n❌ Benchmark failed to select a model")
                return 1

            print(f"\n✅ Benchmark selected: {best_model_name}")

            # Update config with selected model
            selector = ModelSelector(verbose=False)
            if best_model_name in selector.MODEL_PROFILES:
                profile = selector.MODEL_PROFILES[best_model_name]
                config.model_class = profile.model_class
                config.auto_select_model = False
        else:
            # Manual model selection or auto-select
            model_choice = self._select_model()
            if model_choice == "auto":
                config.auto_select_model = True
            else:
                selector = ModelSelector(verbose=False)
                if model_choice in selector.MODEL_PROFILES:
                    profile = selector.MODEL_PROFILES[model_choice]
                    config.model_class = profile.model_class
                    config.auto_select_model = False

        # Train models
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)

        try:
            trainer = MultiLabelTrainer(config)

            # Load data to show statistics
            print("\n📊 Loading and analyzing data...")
            samples = trainer.load_multi_label_data(str(data_file))

            # Display statistics
            self._display_data_statistics(samples)

            # Train models
            print("\n⏳ Training in progress...")
            models = trainer.train(
                data_file=str(data_file),
                auto_split=config.auto_split,
                split_ratio=config.split_ratio,
                stratified=config.stratified,
                output_dir=str(config.output_dir)
            )

            # Display results
            self._display_training_results(models)

            # Save comprehensive logs
            self._save_training_logs(models, data_file, config)

            print("\n✅ Training completed successfully!")
            return 0

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
            return 1

    def _run_benchmark_mode(self) -> int:
        """
        Run benchmark mode.

        Returns:
            Exit code
        """
        print("\n" + "="*60)
        print("🏆 BENCHMARK MODE")
        print("="*60)

        # Select data file
        data_file = self._select_data_file()
        if not data_file:
            return 1

        # Analyze data to detect languages
        print("\n🔍 Analyzing dataset...")
        languages = self._detect_data_languages(data_file)

        # Display available models and let user select
        selected_models = self._select_benchmark_models(languages)
        if not selected_models:
            print("\n❌ No models selected for benchmark")
            return 1

        # Get benchmark configuration
        epochs = self._get_integer("Number of epochs", default=10, min_val=1, max_val=50)
        batch_size = self._get_integer("Batch size", default=32, min_val=4, max_val=128)

        balance = self._confirm("\n📊 Balance classes (undersample)?")
        test_all = False  # We handle selection above

        # Ask about reinforced learning
        use_reinforced = self._confirm(
            "\n⚡ Enable reinforced learning for poor performers?",
            default=True
        )

        # Ask about short sequence optimization
        print("\n📏 Your data may contain short sequences (individual sentences).")
        print("   Some models (Longformer, BigBird, ALBERT Large) are optimized for long documents.")
        optimize_short = self._confirm(
            "Optimize parameters for short sequences?",
            default=True
        )

        # Create benchmark configuration
        config = BenchmarkConfig(
            epochs=epochs,
            batch_size=batch_size,
            balance_benchmark_classes=balance,
            test_split_size=0.2,
            save_benchmark_csv=True,
            track_languages=True,
            use_reinforced_in_benchmark=use_reinforced,
            reinforced_learning=use_reinforced,
            reinforced_epochs=5 if use_reinforced else 0,
            reinforced_f1_threshold=0.60,
            rescue_low_class1_f1=use_reinforced,
            f1_rescue_threshold=0.01,
            optimize_for_short_sequences=optimize_short,
            short_sequence_threshold=100,
            large_model_adjustments=True
        )

        # Create training config with benchmark options
        training_config = TrainingConfig(output_dir=str(self.models_dir))
        training_config.use_reinforced_in_benchmark = use_reinforced
        training_config.optimize_for_short_sequences = optimize_short

        # Run benchmark with selected models
        selected = self._run_benchmark(
            data_file=data_file,
            config=training_config,
            benchmark_epochs=epochs,
            test_all=test_all,
            allow_selection=True,
            models_to_test=selected_models
        )

        if selected:
            print(f"\n✅ Selected model: {selected}")
            return 0
        else:
            print("\n⚠️  No model selected")
            return 1

    def _run_benchmark(
        self,
        data_file: Path,
        config: TrainingConfig,
        benchmark_epochs: int = 3,
        test_all: bool = True,
        allow_selection: bool = True,
        models_to_test: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Run the benchmark system.

        Args:
            data_file: Path to data file
            config: Training configuration
            benchmark_epochs: Number of epochs for benchmarking
            test_all: Test all appropriate models
            allow_selection: Allow user selection
            models_to_test: Specific models to test

        Returns:
            Selected model name or None
        """
        print("\n🔍 Starting benchmark...")
        if models_to_test:
            print(f"   • Testing {len(models_to_test)} selected models")
        else:
            print("   • Auto-selecting models based on detected languages")

        # Create benchmark config
        benchmark_config = BenchmarkConfig(
            epochs=benchmark_epochs,
            batch_size=config.batch_size if hasattr(config, 'batch_size') else 32,
            learning_rate=config.learning_rate if hasattr(config, 'learning_rate') else 5e-5,
            save_benchmark_csv=True,
            track_languages=True
        )

        # Create runner
        runner = BenchmarkRunner(
            data_root=self.data_dir,
            models_root=self.models_dir,
            config=benchmark_config
        )

        # Run benchmark
        selected = runner.run_comprehensive_benchmark(
            data_path=data_file,
            benchmark_epochs=benchmark_epochs,
            test_all_models=test_all,
            models_to_test=models_to_test,  # Use selected models if provided
            allow_user_selection=allow_selection,
            verbose=self.verbose,
            save_detailed_log=True,
            save_best_models_log=True
        )

        # Show log locations
        logs_dir = runner.models_root / "benchmark_logs"
        if logs_dir.exists():
            print(f"\n📂 Benchmark logs saved to: {logs_dir}")
            print("   • benchmark_detailed_*.csv - All model metrics")
            print("   • benchmark_best_models_*.csv - Best models summary")
            print("   • benchmark_results_*.json - Complete results")

        return selected

    def _select_data_file(self) -> Optional[Path]:
        """
        Select a data file for training.

        Returns:
            Path to selected file or None
        """
        print("\n📁 SELECT DATA FILE")
        print("-" * 40)

        # Find available files
        files = []

        # Search for JSONL and JSON files
        search_patterns = ["*.jsonl", "*.json"]
        for pattern in search_patterns:
            files.extend(list(self.data_dir.rglob(pattern)))

        if not files:
            print(f"❌ No JSON/JSONL files found in {self.data_dir}")
            return None

        # Display files
        print(f"\nFound {len(files)} file(s):")
        for i, file in enumerate(files[:20], 1):
            rel_path = file.relative_to(self.data_dir)
            print(f"  [{i}] {rel_path}")

        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")

        # Get selection
        while True:
            choice = input(f"\nSelect file [1-{len(files)}]: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    selected = files[idx]
                    print(f"✓ Selected: {selected.name}")
                    return selected
            print("❌ Invalid selection")

    def _get_training_config(self) -> TrainingConfig:
        """
        Get training configuration from user.

        Returns:
            TrainingConfig object
        """
        print("\n⚙️  TRAINING CONFIGURATION")
        print("-" * 40)

        # Basic parameters
        epochs = self._get_integer("Number of epochs", default=20, min_val=1, max_val=100)
        batch_size = self._get_integer("Batch size", default=32, min_val=4, max_val=128)
        learning_rate = self._get_float("Learning rate", default=5e-5, min_val=1e-6, max_val=1e-3)

        # Split configuration
        auto_split = self._confirm("\nAuto-split data (if no validation set)?")
        split_ratio = 0.8
        stratified = True

        if auto_split:
            split_ratio = self._get_float("Train/validation split ratio", default=0.8, min_val=0.5, max_val=0.95)
            stratified = self._confirm("Use stratified split?")

        # Advanced options
        print("\n🔧 ADVANCED OPTIONS")
        reinforced = self._confirm("Enable reinforced learning?")
        reinforced_epochs = 5
        if reinforced:
            reinforced_epochs = self._get_integer("Reinforced learning epochs", default=5, min_val=1, max_val=20)

        parallel = self._confirm("\nEnable parallel training?")
        max_workers = 2
        if parallel:
            max_workers = self._get_integer("Number of workers", default=2, min_val=1, max_val=8)

        # Language strategy
        print("\n🌍 LANGUAGE STRATEGY")
        print("[1] Auto-detect from data")
        print("[2] One multilingual model per label")
        print("[3] Separate models by language")

        strategy = input("Select strategy [1-3]: ").strip()
        train_by_language = strategy == "3"
        multilingual = strategy == "2"

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = self.models_dir / f"training_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return TrainingConfig(
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            auto_select_model=True,
            train_by_language=train_by_language,
            multilingual_model=multilingual,
            reinforced_learning=reinforced,
            n_epochs_reinforced=reinforced_epochs,
            track_languages=True,
            output_dir=str(output_dir),
            parallel_training=parallel,
            max_workers=max_workers,
            auto_split=auto_split,
            split_ratio=split_ratio,
            stratified=stratified
        )

    def _select_model(self) -> str:
        """
        Select a model for training.

        Returns:
            Model name or 'auto'
        """
        print("\n🤖 MODEL SELECTION")
        print("-" * 40)

        selector = ModelSelector(verbose=False)
        models = ["auto (recommended)"] + list(selector.MODEL_PROFILES.keys())

        # Group models by language
        print("\n[0] 🔮 Auto-select (recommended)")

        print("\n📌 Multilingual Models:")
        idx = 1
        multilingual = []
        for name, profile in selector.MODEL_PROFILES.items():
            if '*' in profile.supported_languages:
                print(f"  [{idx}] {name}")
                multilingual.append(name)
                idx += 1

        print("\n🇬🇧 English Models:")
        english = []
        for name, profile in selector.MODEL_PROFILES.items():
            if profile.supported_languages == ['en']:
                print(f"  [{idx}] {name}")
                english.append(name)
                idx += 1

        print("\n🇫🇷 French Models:")
        french = []
        for name, profile in selector.MODEL_PROFILES.items():
            if profile.supported_languages == ['fr']:
                print(f"  [{idx}] {name}")
                french.append(name)
                idx += 1

        all_models = ["auto"] + multilingual + english + french

        while True:
            choice = input(f"\nSelect model [0-{len(all_models)-1}]: ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(all_models):
                    return all_models[idx]
            print("❌ Invalid selection")

    def _display_data_statistics(self, samples: List) -> None:
        """Display statistics about the loaded data."""
        print(f"\n📈 DATASET STATISTICS")
        print("="*60)
        print(f"  Total samples: {len(samples)}")

        # Language distribution
        if samples and hasattr(samples[0], 'lang'):
            lang_counts = Counter(s.lang for s in samples if s.lang)
            if lang_counts:
                print(f"\n  📍 Language distribution:")
                total_width = 40
                for lang, count in lang_counts.most_common():
                    ratio = count / len(samples)
                    bar_width = int(ratio * total_width)
                    bar = '█' * bar_width + '░' * (total_width - bar_width)
                    print(f"    {lang:3} : {bar} {count:5} ({100*ratio:5.1f}%)")

        # Category distribution
        if samples and hasattr(samples[0], 'labels'):
            category_counts = Counter()
            for sample in samples:
                if sample.labels:
                    category_counts.update(sample.labels.keys())

            if category_counts:
                print(f"\n  📑 Category distribution:")
                print(f"  {'Category':<30} {'Count':<10} {'Percentage':<10}")
                print("  " + "-"*50)
                for cat, count in sorted(category_counts.items()):
                    cat_name = cat.replace('category_', '')
                    ratio = 100 * count / len(samples)
                    print(f"  {cat_name:<30} {count:<10} {ratio:<10.1f}%")

    def _display_training_results(self, models: Dict) -> None:
        """Display training results."""
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)

        if not models:
            print("⚠️  No models were trained")
            return

        # Results table
        print(f"\n📊 RESULTS ({len(models)} models trained)")
        print("-"*60)
        print(f"{'Category':<30} {'F1 Score':<12} {'Accuracy':<12} {'Support':<10}")
        print("-"*60)

        for model_name, info in sorted(models.items()):
            metrics = info.performance_metrics
            f1 = metrics.get('macro_f1', 0)
            accuracy = metrics.get('accuracy', 0)
            support = metrics.get('support', 0)

            cat_name = model_name.replace('category_', '').split('_')[0]
            print(f"{cat_name:<30} {f1:<12.3f} {accuracy:<12.3f} {support:<10}")

        # Average performance
        if models:
            avg_f1 = sum(info.performance_metrics.get('macro_f1', 0) for info in models.values()) / len(models)
            avg_acc = sum(info.performance_metrics.get('accuracy', 0) for info in models.values()) / len(models)
            print("-"*60)
            print(f"{'AVERAGE':<30} {avg_f1:<12.3f} {avg_acc:<12.3f}")

    def _save_training_logs(self, models: Dict, data_file: Path, config: TrainingConfig) -> None:
        """Save comprehensive training logs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed CSV log
        csv_file = self.logs_dir / f"training_results_{timestamp}.csv"

        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            headers = [
                'timestamp', 'category', 'model_name', 'f1_macro',
                'accuracy', 'f1_class_0', 'f1_class_1',
                'precision_0', 'precision_1', 'recall_0', 'recall_1'
            ]

            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for model_name, info in models.items():
                metrics = info.performance_metrics
                cat_name = model_name.replace('category_', '').split('_')[0]

                row = {
                    'timestamp': timestamp,
                    'category': cat_name,
                    'model_name': model_name,
                    'f1_macro': metrics.get('macro_f1', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_class_0': metrics.get('f1_0', 0),
                    'f1_class_1': metrics.get('f1_1', 0),
                    'precision_0': metrics.get('precision_0', 0),
                    'precision_1': metrics.get('precision_1', 0),
                    'recall_0': metrics.get('recall_0', 0),
                    'recall_1': metrics.get('recall_1', 0)
                }
                writer.writerow(row)

        print(f"\n💾 Results saved to: {csv_file}")

        # Save JSON summary
        json_file = self.logs_dir / f"training_summary_{timestamp}.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data_file),
            'configuration': {
                'epochs': config.n_epochs if hasattr(config, 'n_epochs') else None,
                'batch_size': config.batch_size if hasattr(config, 'batch_size') else None,
                'learning_rate': config.learning_rate if hasattr(config, 'learning_rate') else None
            },
            'models_trained': len(models),
            'results': {
                name: {
                    'f1': info.performance_metrics.get('macro_f1', 0),
                    'accuracy': info.performance_metrics.get('accuracy', 0)
                } for name, info in models.items()
            }
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📋 Summary saved to: {json_file}")

    def _confirm(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no confirmation."""
        if default:
            suffix = " [Y/n]: "
            default_responses = ['', 'y', 'yes']
        else:
            suffix = " [y/N]: "
            default_responses = ['', 'n', 'no']

        response = input(prompt + suffix).strip().lower()

        if default:
            return response in ['', 'y', 'yes']
        else:
            return response not in ['', 'n', 'no']

    def _get_integer(self, prompt: str, default: int, min_val: int = 1, max_val: int = 100) -> int:
        """Get integer input with validation."""
        while True:
            value = input(f"\n{prompt} [{default}]: ").strip()
            if not value:
                return default
            try:
                val = int(value)
                if min_val <= val <= max_val:
                    return val
                print(f"❌ Value must be between {min_val} and {max_val}")
            except ValueError:
                print("❌ Please enter a valid number")

    def _get_float(self, prompt: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Get float input with validation."""
        while True:
            value = input(f"\n{prompt} [{default}]: ").strip()
            if not value:
                return default
            try:
                val = float(value)
                if min_val <= val <= max_val:
                    return val
                print(f"❌ Value must be between {min_val} and {max_val}")
            except ValueError:
                print("❌ Please enter a valid number")

    def _detect_data_languages(self, data_file: Path) -> List[str]:
        """Detect languages in data file."""
        from collections import Counter
        languages = []

        # Read sample of data
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100 lines
                    break
                try:
                    data = json.loads(line.strip())
                    if 'lang' in data:
                        languages.append(data['lang'].upper())
                except:
                    continue

        if not languages:
            print("⚠️  No language information found, assuming English")
            return ['EN']

        lang_counts = Counter(languages)
        detected_langs = list(lang_counts.keys())

        print(f"\n📊 Detected languages: {', '.join(detected_langs)}")
        for lang, count in lang_counts.items():
            print(f"   - {lang}: {count} samples")

        return detected_langs

    def _select_benchmark_models(self, languages: List[str]) -> Optional[List[str]]:
        """Display available models and allow user selection."""
        from llm_tool.trainers.model_selector import ModelSelector

        print("\n" + "="*60)
        print("🤖 MODEL SELECTION FOR BENCHMARK")
        print("="*60)

        selector = ModelSelector(verbose=False)

        # Get model descriptions
        model_descriptions = self._get_model_descriptions()

        # Categorize models by language support
        multilingual_models = []
        english_models = []
        french_models = []
        other_lang_models = {}

        for model_name, profile in selector.MODEL_PROFILES.items():
            supported = profile.supported_languages

            if '*' in supported:
                multilingual_models.append(model_name)
            elif 'en' in supported:
                english_models.append(model_name)
            elif 'fr' in supported:
                french_models.append(model_name)
            else:
                for lang in supported:
                    if lang not in other_lang_models:
                        other_lang_models[lang] = []
                    other_lang_models[lang].append(model_name)

        # Display models by category
        all_models = []
        model_index = 1

        print("\n🌍 MULTILINGUAL MODELS (work with all languages):")
        print("-" * 50)
        for model in multilingual_models:
            desc = model_descriptions.get(model, "")
            print(f"[{model_index:2}] {model:<25} {desc}")
            all_models.append(model)
            model_index += 1

        if 'EN' in languages or 'ENGLISH' in [l.upper() for l in languages]:
            print("\n🇬🇧 ENGLISH MODELS:")
            print("-" * 50)
            for model in english_models:
                desc = model_descriptions.get(model, "")
                print(f"[{model_index:2}] {model:<25} {desc}")
                all_models.append(model)
                model_index += 1

        if 'FR' in languages or 'FRENCH' in [l.upper() for l in languages]:
            print("\n🇫🇷 FRENCH MODELS:")
            print("-" * 50)
            for model in french_models:
                desc = model_descriptions.get(model, "")
                print(f"[{model_index:2}] {model:<25} {desc}")
                all_models.append(model)
                model_index += 1

        # Selection menu
        print("\n" + "="*60)
        print("SELECT MODELS TO TEST:")
        print("[A] Test ALL appropriate models (recommended)")
        print("[M] Test only multilingual models")
        print("[L] Test only language-specific models")
        print("[S] Select specific models")
        print("[Q] Cancel benchmark")

        choice = input("\nYour choice: ").strip().upper()

        if choice == 'Q':
            return None
        elif choice == 'A':
            print(f"\n✅ Testing all {len(all_models)} appropriate models")
            return all_models
        elif choice == 'M':
            print(f"\n✅ Testing {len(multilingual_models)} multilingual models")
            return multilingual_models
        elif choice == 'L':
            lang_specific = english_models + french_models
            for models in other_lang_models.values():
                lang_specific.extend(models)
            print(f"\n✅ Testing {len(lang_specific)} language-specific models")
            return lang_specific
        elif choice == 'S':
            print("\nEnter model numbers separated by commas (e.g., 1,3,5-8):")
            selection_str = input("> ").strip()

            selected = []
            try:
                parts = selection_str.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        for i in range(start, end + 1):
                            if 1 <= i <= len(all_models):
                                selected.append(all_models[i - 1])
                    else:
                        idx = int(part)
                        if 1 <= idx <= len(all_models):
                            selected.append(all_models[idx - 1])

                if selected:
                    print(f"\n✅ Selected {len(selected)} models:")
                    for model in selected:
                        print(f"   - {model}")
                    return selected
                else:
                    print("❌ No valid models selected")
                    return None
            except:
                print("❌ Invalid selection format")
                return None
        else:
            print("❌ Invalid choice")
            return None

    def _get_model_descriptions(self) -> Dict[str, str]:
        """Get short descriptions for each model."""
        return {
            # Multilingual
            'MDeBERTaV3Base': '🌐 278M params, SOTA multilingual',
            'XLMRobertaBase': '🌐 270M params, 100+ languages',
            'XLMRobertaLarge': '🌐 560M params, best multilingual',
            'DistilBertMultilingual': '🌐 134M params, fast & efficient',

            # English
            'DeBERTaV3XSmall': '🚀 22M params, ultra-fast',
            'DeBERTaV3Small': '⚡ 44M params, fast & accurate',
            'DeBERTaV3Base': '💪 86M params, best accuracy',
            'RobertaBase': '📚 125M params, classic SOTA',
            'RobertaLarge': '🏆 355M params, highest accuracy',
            'ElectraSmall': '⚡ 14M params, efficient',
            'ElectraBase': '🔌 110M params, good balance',
            'ElectraLarge': '⚡ 335M params, discriminative',
            'AlbertBaseV2': '💡 12M params, parameter sharing',
            'AlbertLargeV2': '💡 18M params, factorized',
            'BertBase': '🔤 110M params, original BERT',
            'BertLarge': '🔤 340M params, large BERT',
            'DistilBert': '🏃 66M params, distilled',
            'TinyBert': '🐜 4M params, ultra-light',
            'MobileBert': '📱 25M params, mobile-optimized',

            # French
            'CamembertBase': '🥖 110M params, French RoBERTa',
            'CamembertaV2Base': '🥐 110M params, French SOTA 2024',
            'FlaubertBase': '🇫🇷 137M params, French BERT',
            'FrALBERTBase': '🗼 12M params, French ALBERT',
            'DistilCamembert': '🏃 68M params, fast French',
            'FrELECTRABase': '⚡ 110M params, French discriminative',

            # German
            'GBertBase': '🇩🇪 110M params, German BERT',
            'GElectraBase': '⚡ 110M params, German ELECTRA',

            # Spanish
            'BetoBERT': '🇪🇸 110M params, Spanish BERT',
            'SpanBERTa': '🇪🇸 125M params, Spanish RoBERTa',

            # Italian
            'UmBERTo': '🇮🇹 110M params, Italian RoBERTa',

            # Portuguese
            'BERTimbau': '🇧🇷 110M params, Brazilian Portuguese'
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='LLMTool Training CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing training data files'
    )

    parser.add_argument(
        '--models-dir',
        type=Path,
        required=True,
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--logs-dir',
        type=Path,
        required=True,
        help='Directory to save logs and metrics'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    parser.add_argument(
        '--mode',
        choices=['interactive', 'multi-label', 'benchmark'],
        default='interactive',
        help='Training mode (default: interactive)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='JSON configuration file for non-interactive mode'
    )

    args = parser.parse_args()

    # Validate directories
    if not args.data_dir.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Create CLI instance
    cli = TrainingCLI(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        verbose=not args.quiet
    )

    # Run based on mode
    if args.mode == 'interactive':
        exit_code = cli.run()
    elif args.config and args.config.exists():
        # Load config and run non-interactively
        with open(args.config, 'r') as f:
            config = json.load(f)
        print("⚠️  Non-interactive mode not fully implemented")
        exit_code = 1
    else:
        exit_code = cli.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
training_arena_integrated.py

MAIN OBJECTIVE:
---------------
Provides complete Training Arena integration for training 50+ models (BERT/RoBERTa/DeBERTa)
with multi-label classification and benchmarking capabilities.

Dependencies:
-------------
- pandas
- rich (Console, Table, Prompt, Panel)
- tqdm
- llm_tool.trainers.training_data_builder
- llm_tool.utils.training_data_utils
- llm_tool.cli.advanced_cli

MAIN FEATURES:
--------------
1) Training studio interface for dataset preparation
2) Support for 50+ transformer models
3) Multi-label classification training
4) Model benchmarking and comparison
5) Interactive training configuration
6) Training session management

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set, Iterable
from datetime import datetime
from uuid import uuid4
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich import box
from rich.panel import Panel
import logging
from tqdm import tqdm
import json
import ast
import tempfile
import sys
import inspect

# Import Training Arena dependencies
from llm_tool.trainers.training_data_builder import TrainingDatasetBuilder, TrainingDataBundle, TrainingDataRequest
from llm_tool.utils.training_data_utils import TrainingDataSessionManager
from llm_tool.utils.training_paths import (
    get_training_logs_base,
    get_training_metrics_dir,
    get_training_data_dir,
)
from llm_tool.utils.data_detector import DataDetector
from llm_tool.utils.session_summary import collect_summaries_for_mode, read_summary
from llm_tool.utils.data_filter_logger import get_filter_logger
from llm_tool.utils.system_resources import get_model_optimized_config
# NEW: Import training logger for session manifest generation
from llm_tool.trainers.training_logger import (
    TrainingHyperparameters,
    SessionManifest,
    CategoryTrainingResult,
    generate_session_manifest,
    finalize_session_manifest,
)

# Constants
HAS_RICH = True


def _get_optimal_batch_size(model_name: Optional[str] = None) -> int:
    """
    Get optimal batch size based on system resources and model size.

    Uses dynamic detection to maximize GPU utilization on all platforms,
    especially Apple Silicon Macs with unified memory.

    Parameters
    ----------
    model_name : str, optional
        Name of the model (e.g., 'xlm-roberta-large')

    Returns
    -------
    int
        Optimal batch size for the system and model combination
    """
    try:
        config = get_model_optimized_config(model_name)
        return config.get('batch_size', 16)
    except Exception:
        # Fallback to reasonable default if detection fails
        return 16


def _normalize_column_choice(
    user_input: Optional[str],
    all_columns: List[str],
    candidate_columns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Normalize a user-supplied column selection.

    Accepts direct column names (case-sensitive or insensitive) as well as numeric
    selections that refer to displayed indices. Returns the resolved column name
    or ``None`` when the input cannot be mapped.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared step numbering across modes
# ---------------------------------------------------------------------------

STEP_LABEL_OVERRIDES: Dict[str, Dict[str, str]] = {
    "arena": {
        "text_length": "STEP 5",
        "language_detection": "STEP 6",
        "label_selection": "STEP 7",
        "identifier_selection": "STEP 8",
        "annotation_preview": "STEP 9",
        "value_filter": "STEP 10",
        "training_strategy": "STEP 11",
        "data_split": "STEP 12",
        "additional_columns": "STEP 13",
        "token_strategy": "STEP 14",
        "multilingual_strategy": "STEP 15",
        "model_selection": "STEP 16",
        "reinforced_learning": "STEP 17",
        "epochs": "STEP 18",
    },
    # Annotator Factory runs training as Phase 2; keep numbering scoped to that phase.
    "factory": {
        "text_length": "STEP 2.1",
        "language_detection": "STEP 2.2",
        "annotation_preview": "STEP 2.3",
        "value_filter": "STEP 2.4",
        "training_strategy": "STEP 2.5",
        "data_split": "STEP 2.6",
        "additional_columns": "STEP 2.7",
    },
    "arena_quick": {
        "token_strategy": "STEP 1",
        "multilingual_strategy": "STEP 2",
        "model_selection": "STEP 3",
        "reinforced_learning": "STEP 4",
        "epochs": "STEP 5",
    },
    "factory_quick": {
        "token_strategy": "STEP 2.8",
        "multilingual_strategy": "STEP 2.9",
        "model_selection": "STEP 2.10",
        "reinforced_learning": "STEP 2.11",
        "epochs": "STEP 2.12",
    },
}


def resolve_step_label(step_key: str, default_label: str, context: str = "arena") -> str:
    """
    Return the appropriate step label for the provided context.

    Parameters
    ----------
    step_key : str
        Identifier for the logical step (e.g., 'text_length', 'language_detection').
    default_label : str
        Baseline label used by the Training Arena workflow.
    context : str, optional
        Logical context or mode requesting the label. Defaults to 'arena'.

    Returns
    -------
    str
        Context-aware step label string.
    """
    return STEP_LABEL_OVERRIDES.get(context, {}).get(step_key, default_label)


# ============================================================================
# ALL TRAINING ARENA CODE BELOW (pasted by user)
# ============================================================================

def training_studio(self):
    """Training studio bringing dataset builders and trainers together."""
    # Display ASCII logo only
    self._display_ascii_logo()

    # Display mode-specific banner
    self._display_mode_banner('arena')

    # Display personalized mode info
    self._display_section_header(
        "Training Arena - Train 50+ Models (BERT/RoBERTa/DeBERTa) with Multi-Label & Benchmarking",
        "Professional model training with intelligent optimization, reinforcement learning, and comprehensive benchmarking",
        mode_info={
            'workflow': 'Load Data → Language Detection → Model Selection → Multi-Label Training → Reinforcement Learning → Benchmark',
            'capabilities': ['50+ Models (BERT/RoBERTa/DeBERTa/Longformer)', 'Multi-Label Classification', 'Parallel GPU/CPU', 'Class Imbalance Handling', 'Hard Negative Mining'],
            'input': 'Annotated CSV/JSON/JSONL/SQL with labels (single or multi-label)',
            'output': 'Trained models + Confusion matrices + F1 scores + Training summaries + Best model selection',
            'best_for': 'Production-ready model training with automatic optimization and comprehensive evaluation',
            'duration': '~5-30 min per model (benchmark mode: 30min-3hrs depending on data size)'
        }
    )

    if not (HAS_RICH and self.console):
        print("\nTraining Arena requires the Rich interface. Launch `llm-tool --simple` for basic commands.")
        return

    self._ensure_training_models_loaded()

    # NEW: Add resume/new menu BEFORE starting wizard
    self.console.print("\n[bold cyan]Training Session Options[/bold cyan]\n")

    session_options_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
    session_options_table.add_column("Option", style="cyan bold", no_wrap=True)
    session_options_table.add_column("Description", style="white", ratio=1, overflow="fold")

    session_options_table.add_row(
        "1",
        "Resume/Relaunch Training\n   Load saved parameters from previous training sessions"
    )
    session_options_table.add_row(
        "2",
        "New Training Session\n   Start fresh with dataset selection and configuration"
    )
    session_options_table.add_row(
        "3",
        "← Back to Main Menu"
    )

    self.console.print(session_options_table)
    self.console.print()

    session_choice = Prompt.ask(
        "[bold yellow]Select an option[/bold yellow]",
        choices=["1", "2", "3"],
        default="2"
    )

    if session_choice == "1":
        # Resume/Relaunch existing session
        self._resume_training_studio()
        return
    elif session_choice == "3":
        # Back to main menu
        return

    # Continue with NEW training session
    # CRITICAL: Ask user for session name first
    from datetime import datetime
    from llm_tool.utils.training_data_utils import TrainingDataSessionManager

    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    self.console.print("[bold cyan]           Session Name Configuration                       [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    self.console.print("[bold]Why session names matter:[/bold]")
    self.console.print("  • [green]Organization:[/green] Easily identify experiments (e.g., 'baseline', 'improved_features')")
    self.console.print("  • [green]Traceability:[/green] Track your training runs across data, logs, and models")
    self.console.print("  • [green]Collaboration:[/green] Team members understand what each session represents")
    self.console.print("  • [green]Audit trail:[/green] Timestamp ensures uniqueness\n")

    self.console.print("[dim]Format: {session_name}_{yyyymmdd_hhmmss}[/dim]")
    self.console.print("[dim]Example: sentiment_analysis_20251008_143022[/dim]\n")

    # Ask for user-defined session name
    user_session_name = Prompt.ask(
        "[bold yellow]Enter a descriptive name for this training session[/bold yellow]",
        default="training_session"
    ).strip()

    # Sanitize the user input (remove special chars, replace spaces with underscores)
    user_session_name = user_session_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    user_session_name = ''.join(c for c in user_session_name if c.isalnum() or c in ['_', '-'])

    # Create full session ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{user_session_name}_{timestamp}"

    self.console.print(f"\n[bold green]✓ Session ID:[/bold green] [cyan]{session_id}[/cyan]")
    self.console.print(f"[dim]This ID will be used consistently across all data, logs, and models[/dim]\n")

    # Initialize session manager for comprehensive data distribution logging
    session_manager = TrainingDataSessionManager(session_id=session_id)

    # Initialize builder with session-based organization
    builder = TrainingDatasetBuilder(
        session_manager.logs_base_dir,
        session_id=session_id
    )

    # Store for later use throughout the training session
    self.current_session_id = session_id
    self.current_session_manager = session_manager

    self._training_studio_show_model_catalog()

    # First, configure the dataset
    try:
        bundle = self._training_studio_dataset_wizard(builder)
    except Exception as exc:  # pylint: disable=broad-except
        self.console.print(f"[red]Dataset preparation failed:[/red] {exc}")
        self.logger.exception("Training Arena dataset preparation failed", exc_info=exc)
        return

    if bundle is None:
        self.console.print("[yellow]Training cancelled.[/yellow]")
        return

    # Show dataset summary
    self._training_studio_render_bundle_summary(bundle)

    # Note: Comprehensive logging will be done AFTER training/benchmark
    # to include complete information about what was used for what

    # Configure learning parameters and start training
    self.console.print("\n[bold cyan]Configuring learning parameters...[/bold cyan]\n")

    # Proceed directly to parameter configuration and training
    self._training_studio_confirm_and_execute(bundle, "quick")

# ------------------------------------------------------------------
# Training Arena helpers
# ------------------------------------------------------------------
def _set_onevsall_storage_root(self, root: Optional[Path]) -> None:
    """Remember the persistence directory for one-vs-all datasets."""
    self._persistent_onevsall_root = root
    self._onevsall_storage_announced = False


def _announce_onevsall_storage(self) -> None:
    """Announce the persistence directory once per session."""
    root = getattr(self, "_persistent_onevsall_root", None)
    announced = getattr(self, "_onevsall_storage_announced", False)
    if root and not announced:
        self.console.print(f"[dim]One-vs-all binary datasets will be persisted under: {root}[/dim]")
        self._onevsall_storage_announced = True


def _prepare_onevsall_workspace(self, prefix: str = "onevsall") -> Path:
    """Create a directory for storing one-vs-all intermediate files."""
    unique_suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    root = getattr(self, "_persistent_onevsall_root", None)
    if root:
        target_dir = root / f"{prefix}_{unique_suffix}"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_{unique_suffix}_"))


def _run_parallel_training(
    self,
    category_files: Dict[str, Path],
    model_name: str,
    epochs: int,
    languages: set,
    needs_language_training: bool,
    enable_reinforced_learning: bool,
    manual_rl_epochs: Optional[int],
    quick_params: Optional[Dict[str, Any]],
    session_id: Optional[str],
    bundle: Any,
    output_dir: Path,
    runtime_params: Dict[str, Any],
    models_by_language: Optional[Dict[str, str]] = None,
    training_approach: str = 'one-vs-all',
    extra_config: Optional[Dict[str, Any]] = None,
    distributed_orchestrator: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute parallel training using GPU + CPU workers with full logging and charts.

    This method uses the ParallelTrainingManager for complete integration:
    - Thread-safe logging to training.csv and best.csv
    - Real-time charts for each model
    - Unified Rich dashboard showing all models
    - Summary chart at completion

    Parameters
    ----------
    category_files : Dict[str, Path]
        Dictionary mapping category names to training data file paths
    model_name : str
        Name of the model to use for training
    epochs : int
        Number of training epochs per model
    languages : set
        Set of languages detected in the data
    needs_language_training : bool
        Whether to train separate models per language
    enable_reinforced_learning : bool
        Whether to enable reinforced learning
    manual_rl_epochs : int, optional
        Manual reinforced learning epochs
    quick_params : dict, optional
        Quick mode parameters
    session_id : str, optional
        Session ID for tracking
    bundle : TrainingDataBundle
        The training data bundle
    output_dir : Path
        Output directory for models
    runtime_params : dict
        Runtime parameters for results
    models_by_language : dict, optional
        Per-language model mappings

    Returns
    -------
    Dict[str, Any]
        Training results with full metrics and chart paths
    """
    import time
    from llm_tool.trainers.parallel_training_manager import (
        ParallelTrainingManager,
        ParallelTrainingConfig,
    )

    self.console.print("\n[bold green]Starting Parallel Training with Full Integration...[/bold green]\n")

    # Use "models" as the base directory for model outputs
    # ParallelTrainingManager will add session_id subfolder: models/{session_id}/{category}/
    # This keeps parallel training models in the same structure as normal training
    session_output_dir = Path("models")
    session_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if multi-label mode should be enabled
    is_multi_label = training_approach == 'multi-label'
    multi_label_threshold = bundle.metadata.get('multi_label_threshold', 0.5) if hasattr(bundle, 'metadata') else 0.5

    # Clarify training mode
    if training_approach == 'one-vs-all':
        mode_desc = "one-vs-all (separate binary classifiers per label)"
    elif training_approach == 'multi-label':
        mode_desc = "multi-label (single model with sigmoid, predicts all labels)"
    else:
        mode_desc = training_approach
    self.console.print(f"[dim]Training approach: {mode_desc}[/dim]")

    # Determine reinforced learning parameters
    rl_f1_threshold = 0.7
    rl_oversample_factor = 2.0
    rl_class_weight_factor = 2.0
    if quick_params:
        rl_f1_threshold = quick_params.get('rl_f1_threshold', 0.7)
        rl_oversample_factor = quick_params.get('rl_oversample_factor', 2.0)
        rl_class_weight_factor = quick_params.get('rl_class_weight_factor', 2.0)
        # Force reinforced: set threshold to 1.0 so RL always triggers
        if quick_params.get('force_reinforced', False):
            rl_f1_threshold = 1.0

    # Log reinforced learning status
    if enable_reinforced_learning:
        if quick_params and quick_params.get('force_reinforced', False):
            rl_info = f"RL enabled (FORCED - will always run, epochs={manual_rl_epochs or 'auto'})"
        else:
            rl_info = f"RL enabled (threshold={rl_f1_threshold}, epochs={manual_rl_epochs or 'auto'})"
        self.console.print(f"[dim]{rl_info}[/dim]")

    # Determine per-language training configuration
    languages_list = list(languages) if languages else []
    train_by_language = needs_language_training and len(languages_list) > 1

    if train_by_language:
        num_categories = len(category_files)
        total_models = num_categories * len(languages_list)
        self.console.print(f"[cyan]Per-language training enabled:[/cyan] {num_categories} categories × {len(languages_list)} languages = {total_models} models")
        self.console.print(f"[dim]   Languages: {', '.join(sorted(languages_list))}[/dim]")
    else:
        self.console.print(f"[dim]Multilingual training: {len(category_files)} models (all languages combined)[/dim]")

    # Configure parallel training with all features enabled
    config = ParallelTrainingConfig(
        model_name=model_name,
        epochs=int(epochs),
        learning_rate=quick_params.get('learning_rate', 2e-5) if quick_params else 2e-5,
        warmup_ratio=quick_params.get('warmup_ratio', 0.0) if quick_params else 0.0,
        output_dir=str(session_output_dir),
        session_id=session_id or f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_cpu_workers=None,  # Auto-detect based on system resources
        prefer_gpu=True,
        enable_charts=True,  # Generate per-model and summary charts
        enable_logging=True,  # Thread-safe CSV/JSON logging
        # Multi-label settings - CRITICAL for proper training mode
        training_approach=training_approach,
        multi_label=is_multi_label,
        multi_label_threshold=multi_label_threshold,
        # Reinforced learning settings
        reinforced_learning=enable_reinforced_learning,
        reinforced_epochs=manual_rl_epochs,
        rl_f1_threshold=rl_f1_threshold,
        rl_oversample_factor=rl_oversample_factor,
        rl_class_weight_factor=rl_class_weight_factor,
        # NORMAL-phase imbalance handling (session-global)
        imbalance_strategy=(quick_params.get('imbalance_strategy') if quick_params else None),
        focal_gamma=(quick_params.get('focal_gamma', 2.0) if quick_params else 2.0),
        imbalance_weight_source=(quick_params.get('imbalance_weight_source', 'auto') if quick_params else 'auto'),
        imbalance_class_weights=(quick_params.get('imbalance_class_weights') if quick_params else None),
        imbalance_weighted_sampler=(quick_params.get('imbalance_weighted_sampler', True) if quick_params else True),
        # Per-language training settings
        train_by_language=train_by_language,
        # Always pass languages for chart tracking, even in multilingual mode
        languages=languages_list if languages_list else None,
        models_by_language=models_by_language,
    )

    # Attach distributed orchestrator if available (SSH distributed training)
    if distributed_orchestrator:
        distributed_orchestrator.config = config
        config.distributed_orchestrator = distributed_orchestrator
        # If any remote machine has a GPU, force GPU-only mode (no CPU workers)
        if distributed_orchestrator.ssh_manager.any_remote_has_gpu():
            config.max_cpu_workers = 0
            self.console.print("[dim]GPU-only mode: CPU workers desactives (GPUs distants detectes)[/dim]")

    # Convert Path to str for training files
    training_files = {cat: str(path) for cat, path in category_files.items()}

    # Create and run the parallel training manager
    manager = ParallelTrainingManager(config, console=self.console)

    try:
        global_start_time = time.time()

        # Run parallel training
        results = manager.run(training_files)

        total_time = time.time() - global_start_time

        # Process results (exclude metadata keys like '_intelligent_selection')
        training_results = {k: v for k, v in results.items() if not k.startswith('_')}
        successful_results = [r for r in training_results.values() if r.get("status") == "success"]
        failed_results = [r for r in training_results.values() if r.get("status") != "success"]

        # Calculate actual total models (excluding metadata entries)
        total_models_expected = len(training_results)

        if successful_results:
            avg_f1 = sum(r.get("f1_score", 0) for r in successful_results) / len(successful_results)
            avg_accuracy = sum(r.get("accuracy", 0) for r in successful_results) / len(successful_results)
            best_result = max(successful_results, key=lambda r: r.get("f1_score", 0))
            best_category = None
            for cat, res in training_results.items():
                if res.get("f1_score", 0) == best_result.get("f1_score", 0):
                    best_category = cat
                    break

            # Log summary
            self.console.print(f"\n[bold green]{'═' * 60}[/bold green]")
            self.console.print(f"[bold green]   PARALLEL TRAINING COMPLETE[/bold green]")
            self.console.print(f"[bold green]{'═' * 60}[/bold green]")
            self.console.print(f"[green]   [OK] Models trained: {len(successful_results)}/{total_models_expected}[/green]")
            if failed_results:
                self.console.print(f"[red]   [FAIL] Models failed: {len(failed_results)}[/red]")
            if train_by_language:
                self.console.print(f"[cyan]   Per-language: {len(category_files)} categories × {len(languages_list)} languages[/cyan]")
            self.console.print(f"[cyan]   Average F1: {avg_f1:.4f}[/cyan]")
            self.console.print(f"[cyan]   Best: Best model: {best_category} (F1={best_result.get('f1_score', 0):.4f})[/cyan]")
            self.console.print(f"[cyan]   Total time: {total_time:.1f}s ({total_time/60:.1f} min)[/cyan]")
            self.console.print(f"[dim]   Logs: {session_output_dir}/training.csv[/dim]")
            self.console.print(f"[dim]   Charts: {session_output_dir}/charts/[/dim]")

            # Generate comprehensive cross-model summary chart
            try:
                from llm_tool.trainers.training_metrics_chart import generate_comprehensive_summary_chart

                self.console.print("\n[bold cyan]Generating Comprehensive Summary Chart...[/bold cyan]")

                _session_dir = str(session_output_dir)
                _chart_output = str(session_output_dir / "charts")
                _sid = session_id or f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                comp_chart_path = generate_comprehensive_summary_chart(
                    output_dir=_chart_output,
                    session_id=_sid,
                    model_name=model_name,
                    session_dir=_session_dir,
                    results_per_key=training_results,
                    training_approach='parallel',
                    total_training_time=total_time,
                )

                if comp_chart_path:
                    self.console.print(f"[green]   Comprehensive summary chart saved:[/green] {comp_chart_path}")
                else:
                    self.console.print("[yellow]   [!] Could not generate comprehensive summary chart[/yellow]")

            except Exception as e:
                self.logger.warning(f"Failed to generate comprehensive summary chart: {e}")
                self.console.print(f"[yellow]   [!] Comprehensive chart failed: {e}[/yellow]")

            return {
                'runtime_params': runtime_params,
                'best_model': model_name,
                'accuracy': avg_accuracy,
                'best_f1_macro': avg_f1,
                'best_category': best_category,
                'model_path': str(output_dir),
                'training_time': total_time,
                'models_trained': len(successful_results),
                'total_models': total_models_expected,
                'per_category_results': results,
                'parallel_training': True,
                'train_by_language': train_by_language,
                'languages': languages_list if train_by_language else None,
                'logs_dir': str(session_output_dir),
                'charts_dir': str(session_output_dir / "charts"),
            }
        else:
            self.console.print("[red][FAIL] All parallel trainings failed[/red]")
            # Show detailed error information for debugging
            if failed_results:
                self.console.print("\n[bold red]Error details:[/bold red]")
                for cat, res in training_results.items():
                    error_msg = res.get("error", "Unknown error")
                    traceback_info = res.get("traceback", "")
                    self.console.print(f"\n  [yellow]• {cat}:[/yellow]")
                    self.console.print(f"    [red]{error_msg}[/red]")
                    if traceback_info:
                        # Show last 5 lines of traceback
                        tb_lines = traceback_info.strip().split('\n')[-5:]
                        for line in tb_lines:
                            self.console.print(f"    [dim]{line}[/dim]")
            return {
                'runtime_params': runtime_params,
                'models_trained': 0,
                'best_model': None,
                'best_f1': None,
                'error': 'All parallel trainings failed',
                'per_category_results': results,
            }

    except Exception as e:
        self.logger.exception(f"Parallel training failed: {e}")
        import traceback
        self.console.print(f"\n[red][FAIL] Parallel training failed: {e}[/red]")
        self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return {
            'runtime_params': runtime_params,
            'models_trained': 0,
            'best_model': None,
            'best_f1': None,
            'error': str(e),
        }


def _display_training_diagnostic(self, bundle, quick_params=None, preloaded_config=None):
    """Display data distribution diagnostic and training strategy before final confirmation."""
    import math
    from rich.table import Table
    from rich import box
    from collections import Counter

    training_approach = bundle.metadata.get('training_approach', 'one-vs-all')
    is_multi_label = bundle.metadata.get('multi_label', False)
    value_counts = bundle.metadata.get('value_counts_by_key', {})
    rl_threshold = 0.70
    phase1_enabled = False
    phase2_enabled = False
    if quick_params:
        rl_threshold = quick_params.get('rl_f1_threshold', 0.70)
        phase1_enabled = quick_params.get('distribution_aware', False)
        phase2_enabled = quick_params.get('reinforced_learning', False) or quick_params.get('force_reinforced', False)
    elif preloaded_config:
        # Resume mode: extract from preloaded config
        rl_threshold = preloaded_config.get('rl_f1_threshold', 0.70)
        phase1_enabled = preloaded_config.get('distribution_aware', False)
        phase2_enabled = preloaded_config.get('use_reinforcement', False) or preloaded_config.get('skip_to_rl', False)

    # Filter value_counts to only labels that exist in the JSONL (fixes stale metadata from old sessions)
    if value_counts and bundle.primary_file:
        try:
            import json as json_mod
            from pathlib import Path
            pf = Path(bundle.primary_file) if not isinstance(bundle.primary_file, Path) else bundle.primary_file
            if pf.exists() and pf.suffix == '.jsonl':
                # Read actual label prefixes from first 100 lines
                actual_prefixes = set()
                with open(pf, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 100:
                            break
                        try:
                            record = json_mod.loads(line)
                            for label in record.get('labels', []):
                                # Extract prefix: "themes_long_agriculture" -> "themes_long"
                                parts = label.rsplit('_', 1)
                                if len(parts) > 1:
                                    # Try known prefixes from value_counts keys
                                    for vc_key in value_counts:
                                        if label.startswith(vc_key + '_'):
                                            actual_prefixes.add(vc_key)
                                            break
                        except (json_mod.JSONDecodeError, KeyError):
                            continue
                if actual_prefixes:
                    filtered = {k: v for k, v in value_counts.items() if k in actual_prefixes}
                    if filtered:
                        value_counts = filtered
        except Exception:
            pass

    # Fallback: if value_counts_by_key is not available, compute from the training JSONL
    if not value_counts and bundle.primary_file:
        try:
            import json
            from pathlib import Path
            primary = Path(bundle.primary_file) if not isinstance(bundle.primary_file, Path) else bundle.primary_file
            if primary.exists() and primary.suffix == '.jsonl':
                label_counter = Counter()
                total_samples = 0
                with open(primary, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            total_samples += 1
                            labels = record.get('labels', [])
                            if isinstance(labels, list):
                                for label in labels:
                                    label_counter[label] += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                if label_counter and total_samples > 0:
                    # Use annotation keys from metadata if available to group correctly
                    annotation_keys = set()
                    all_keys_vals = bundle.metadata.get('all_keys_values', {})
                    if all_keys_vals:
                        annotation_keys = set(all_keys_vals.keys())

                    key_groups = {}
                    for label, count in label_counter.items():
                        # Try to match against known annotation keys
                        matched_key = None
                        if annotation_keys:
                            for ak in annotation_keys:
                                if label.startswith(ak + '_'):
                                    matched_key = ak
                                    break
                        if matched_key:
                            key = matched_key
                            value = label[len(matched_key) + 1:]
                        else:
                            # Fallback: use the full label as both key and value
                            key = 'labels'
                            value = label
                        if key not in key_groups:
                            key_groups[key] = {}
                        key_groups[key][value] = count
                    value_counts = key_groups
                    bundle.metadata['value_counts_by_key'] = value_counts
                    bundle.metadata['_total_samples'] = total_samples
        except Exception:
            pass  # Silently fall back

    if not value_counts:
        return  # No distribution data available

    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    self.console.print("[bold cyan]           Data Distribution Diagnostic                       [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    approach_display = {
        'one-vs-all': 'One-vs-All Binary',
        'multi-label': 'True Multi-Label',
        'multi-class': 'Multi-Class',
    }.get(training_approach, training_approach)

    n_categories = len(value_counts)
    self.console.print(f"  Training Approach: [bold]{approach_display}[/bold] ({n_categories} categories)\n")

    # Get total samples for multi-label JSONL fallback
    total_samples_override = bundle.metadata.get('_total_samples', 0)

    if training_approach in ('one-vs-all', 'multi-label'):
        # For one-vs-all and multi-label, each category is a binary classification
        diag_table = Table(
            show_header=True, header_style="bold", border_style="dim",
            box=box.SIMPLE_HEAVY, expand=True,
        )
        diag_table.add_column("Category", style="cyan", no_wrap=True)
        diag_table.add_column("Positive", justify="right")
        diag_table.add_column("Total", justify="right")
        diag_table.add_column("Ratio", justify="right")
        diag_table.add_column("Gamma", justify="right")
        diag_table.add_column("Strategy", style="dim")

        for key, counts in sorted(value_counts.items(), key=lambda x: -sum(x[1].values())):
            # Detect if this is a multi-label format where each value is a label name
            # (e.g., {"themes": {"nationalism": 5328, "authority": 3213}})
            # vs binary format ({"nationalism": {"yes": 5328, "no": 94672}})
            sample_keys = [str(k).lower() for k in counts.keys()]
            has_binary_keys = any(k in ('yes', 'no', '0', '1', 'true', 'false', 'oui', 'non') or
                                  k.endswith('_yes') or k.endswith('_no')
                                  for k in sample_keys)
            is_multilabel_counts = (not has_binary_keys and len(counts) > 2)

            if is_multilabel_counts:
                # Multi-label: each value name is a label, count is how many samples have it
                # Total samples: try multiple sources for robustness
                total_for_display = total_samples_override
                if total_for_display == 0:
                    # Try various metadata keys
                    for meta_key in ('_total_samples', 'total_samples', 'num_samples', 'total_rows'):
                        val = bundle.metadata.get(meta_key, 0)
                        if val and val > 0:
                            total_for_display = val
                            break
                if total_for_display == 0:
                    # Try bundle attributes
                    for attr in ('annotated_rows', 'total_rows', 'num_rows'):
                        val = getattr(bundle, attr, 0)
                        if val and val > 0:
                            total_for_display = val
                            break
                if total_for_display == 0 and bundle.primary_file:
                    # Count lines in primary file as last resort
                    try:
                        from pathlib import Path
                        pf = Path(bundle.primary_file) if not isinstance(bundle.primary_file, Path) else bundle.primary_file
                        if pf.exists():
                            total_for_display = sum(1 for _ in open(pf)) - (1 if pf.suffix == '.csv' else 0)
                    except Exception:
                        pass
                # Display each label as a row
                for label_name, label_count in sorted(counts.items(), key=lambda x: -x[1]):
                    total = total_for_display if total_for_display > 0 else sum(counts.values())
                    pos_count = label_count
                    pos_ratio = pos_count / total if total > 0 else 0
                    neg_count = total - pos_count

                    if pos_count == 0:
                        continue
                    ratio = neg_count / pos_count
                    ratio_str = f"1:{ratio:.0f}"
                    gamma = min(max(4.0 + math.log10(1.0 / max(pos_ratio, 1e-6)), 4.0), 8.0)
                    gamma_str = f"{gamma:.1f}"

                    if pos_count < 50:
                        strategy = "fragile (< 50 samples)"
                        style = "red"
                    elif pos_count < 100:
                        strategy = "may underperform"
                        style = "yellow"
                    elif ratio > 50 and phase1_enabled:
                        strategy = "pos_weight + WeightedSampler"
                        style = "yellow"
                    elif ratio > 50:
                        strategy = "imbalanced (Phase 1 off)"
                        style = "yellow"
                    else:
                        strategy = "standard"
                        style = "green"

                    diag_table.add_row(
                        f"  {label_name}",
                        f"[{style}]{pos_count:,}[/{style}]",
                        f"{total:,}",
                        f"[{style}]{ratio_str}[/{style}]",
                        gamma_str,
                        strategy,
                    )
                continue  # Skip the standard per-key processing below

            total = sum(counts.values())
            if total == 0:
                continue

            # Detect the "positive" count from various label formats
            pos_count = 0
            neg_count_explicit = 0
            for k, v in counts.items():
                k_str = str(k).lower().strip()
                if k_str in ('1', 'yes', 'true', 'positive', 'oui') or \
                   k_str.endswith('_yes') or k_str.endswith('_true') or \
                   k_str.endswith('_1') or k_str.startswith('yes_') or \
                   'yes' in k_str.split('_'):
                    pos_count += v
                elif k_str in ('0', 'no', 'false', 'negative', 'non') or \
                     k_str.endswith('_no') or k_str.endswith('_false') or \
                     k_str.endswith('_0') or k_str.startswith('no_') or \
                     'no' in k_str.split('_'):
                    neg_count_explicit += v

            if pos_count == 0:
                if neg_count_explicit > 0:
                    pos_count = total - neg_count_explicit
                elif len(counts) == 2:
                    pos_count = min(counts.values())
                elif len(counts) > 2:
                    pos_count = 0

            pos_ratio = pos_count / total if total > 0 else 0
            neg_count = total - pos_count

            if pos_count == 0:
                ratio_str = "n/a"
                gamma_str = "n/a"
                strategy = "no positives"
                style = "red"
            else:
                ratio = neg_count / pos_count
                ratio_str = f"1:{ratio:.0f}"
                gamma = min(max(4.0 + math.log10(1.0 / max(pos_ratio, 1e-6)), 4.0), 8.0)
                gamma_str = f"{gamma:.1f}"

                if pos_count < 50:
                    strategy = "fragile (< 50 samples)"
                    style = "red"
                elif pos_count < 100:
                    strategy = "weights + focal"
                    style = "yellow"
                elif ratio > 50:
                    strategy = "ASL + WeightedSampler"
                    style = "yellow"
                else:
                    strategy = "standard"
                    style = "green"

            diag_table.add_row(
                key,
                f"[{style}]{pos_count:,}[/{style}]",
                f"{total:,}",
                f"[{style}]{ratio_str}[/{style}]",
                gamma_str,
                strategy,
            )

        self.console.print(diag_table)

    elif training_approach == 'multi-class':
        # Multi-class: show distribution per class
        diag_table = Table(
            show_header=True, header_style="bold", border_style="dim",
            box=box.SIMPLE_HEAVY, expand=True,
        )
        diag_table.add_column("Class", style="cyan", no_wrap=True)
        diag_table.add_column("Count", justify="right")
        diag_table.add_column("Percentage", justify="right")
        diag_table.add_column("Status")

        for key, counts in value_counts.items():
            total = sum(counts.values())
            for class_name, count in sorted(counts.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                if pct < 5:
                    status = "[yellow]minority[/yellow]"
                elif pct > 60:
                    status = "[yellow]majority[/yellow]"
                else:
                    status = "[green]balanced[/green]"
                diag_table.add_row(str(class_name), f"{count:,}", f"{pct:.1f}%", status)

        self.console.print(diag_table)

    # Strategy summary
    self.console.print("\n  [bold]Training Strategy:[/bold]")
    if phase1_enabled:
        self.console.print("  [green]✓[/green] Phase 1: Distribution-aware (pos_weight + WeightedSampler)")
    else:
        self.console.print("  [dim]  Phase 1: Off (standard BCE/ASL loss, no weighted sampling)[/dim]")
    if phase2_enabled:
        self.console.print(f"  [green]✓[/green] Phase 2: Reinforced learning (triggers if F1 < {rl_threshold:.2f})")
        self.console.print("    - DoRA adapters (SOTA): only ~0.7% of params trained, encoder frozen")
        self.console.print("    - Well-performing labels frozen in classifier head")
        self.console.print("    - Underperforming labels get boosted weights + more epochs")
    else:
        self.console.print("  [dim]  Phase 2: Off[/dim]")
    self.console.print()


def _training_studio_confirm_and_execute(
    self,
    bundle: TrainingDataBundle,
    mode: str,
    preloaded_config: Optional[Dict[str, Any]] = None,
    is_resume: bool = False,
    session_id: Optional[str] = None,
    step_context: str = "arena_quick"
) -> None:
    """
    Display training parameters and ask for confirmation before execution.
    This ensures the user reviews all settings before starting training.

    Parameters
    ----------
    bundle : TrainingDataBundle
        The training data bundle
    mode : str
        Training mode (quick)
    preloaded_config : dict, optional
        Pre-loaded configuration from saved session (for resume/relaunch)
    is_resume : bool
        Whether this is a resume (True) or fresh start (False)
    session_id : str, optional
        Session ID for traceability (e.g., from annotator factory)
    """
    from datetime import datetime
    from rich.prompt import Confirm

    # STEP 1: Collect mode-specific parameters BEFORE showing config summary
    quick_params = None
    if mode == "quick" and not is_resume:
        quick_params = self._collect_quick_mode_parameters(
            bundle,
            preloaded_config,
            step_context=step_context,
            session_id=session_id,
        )
        if quick_params is None:
            # User cancelled
            self.console.print("[yellow]Training cancelled by user.[/yellow]")
            return

    # STEP 2: Show configuration summary with modification loop
    while True:
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]           Training Configuration Summary                     [/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        # Create configuration table
        config_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
        config_table.add_column("Parameter", style="cyan bold", no_wrap=True)
        config_table.add_column("Value", style="white", overflow="fold", ratio=1)

        # Dataset information
        config_table.add_row("Dataset", str(bundle.primary_file.name) if bundle.primary_file else "—")
        config_table.add_row("Format", bundle.strategy)
        config_table.add_row("Text Column", bundle.text_column)
        config_table.add_row("Label Column", bundle.label_column)

        if bundle.metadata.get('confirmed_languages'):
            langs = ', '.join([l.upper() for l in bundle.metadata['confirmed_languages']])
            config_table.add_row("Languages", langs)

        # Training mode
        config_table.add_row("Training Mode", "Quick Start - Fast training with defaults")

        # Mode-specific parameters
        if mode == "quick" and quick_params:
            # Check if per-language models were selected
            if quick_params.get('models_by_language'):
                # Show each language's model
                models_display = []
                for lang, model in sorted(quick_params['models_by_language'].items()):
                    models_display.append(f"{lang}: {model}")
                config_table.add_row("Selected Models", "\n".join(models_display))
            else:
                # Single model for all languages
                config_table.add_row("Selected Model", quick_params['model_name'])

            # Reinforced learning display
            if quick_params['reinforced_learning']:
                force_rl = quick_params.get('force_reinforced', False)
                rl_thresh = quick_params.get('rl_f1_threshold', 0.70)
                mode_str = 'FORCED (always runs)' if force_rl else f'Conditional (F1 < {rl_thresh:.2f})'
                phase2_details = f"Phase 2: {mode_str}"
            else:
                phase2_details = "Phase 2: Off"

            phase1_on = quick_params.get('distribution_aware', False)
            phase1_details = "Phase 1: pos_weight + WeightedRandomSampler" if phase1_on else "Phase 1: Off (standard BCE)"

            rl_summary = f"{phase1_details}\n  {phase2_details}"
            config_table.add_row("Training Optimization", rl_summary)

            # Epochs display with reinforced learning info
            if quick_params['reinforced_learning']:
                manual_rl_epochs = quick_params.get('manual_rl_epochs')
                if manual_rl_epochs:
                    max_epochs = quick_params['epochs'] + manual_rl_epochs
                    config_table.add_row("Epochs", f"{quick_params['epochs']} (up to {max_epochs} with reinforced learning)")
                else:
                    config_table.add_row("Epochs", f"{quick_params['epochs']} (up to {quick_params['epochs']}+auto with reinforced learning)")
            else:
                config_table.add_row("Epochs", str(quick_params['epochs']))
            config_table.add_row("Batch Size", "16 (default)")
        elif mode == "quick":
            config_table.add_row("Epochs", "Will be asked (default: 10)")
            config_table.add_row("Batch Size", "16 (default)")

        # Statistics
        if bundle.metadata.get('text_length_stats'):
            stats = bundle.metadata['text_length_stats']
            avg_len = stats.get('avg_chars', stats.get('avg_length', 0))
            config_table.add_row("Avg Text Length", f"{avg_len:.0f} characters")

        self.console.print(config_table)
        self.console.print()

        # Ask for confirmation
        confirm = Confirm.ask(
            "\n[bold yellow]Confirm these parameters?[/bold yellow]",
            default=True
        )

        if confirm:
            break
        else:
            # User wants to modify - ask what to modify for quick mode
            if mode == "quick":
                self.console.print("\n[yellow]What would you like to modify?[/yellow]")

                # Ask if user wants to modify base parameters
                modify_base = Confirm.ask(
                    "[bold yellow]Modify base parameters (model, epochs)?[/bold yellow]",
                    default=False
                )

                modify_rl = False
                if quick_params.get('reinforced_learning'):
                    modify_rl = Confirm.ask(
                        "[bold yellow]Modify reinforced learning parameters?[/bold yellow]",
                        default=False
                    )

                if not modify_base and not modify_rl:
                    # User doesn't want to modify anything, ask again
                    self.console.print("[yellow]No modifications requested. Please confirm parameters again or modify them.[/yellow]\n")
                    continue

                # Only re-collect if user wants to modify something
                if modify_base or modify_rl:
                    self.console.print("\n[cyan]Modifying parameters...[/cyan]\n")
                    quick_params = self._collect_quick_mode_parameters(
                        bundle,
                        quick_params,
                        step_context=step_context,
                        session_id=session_id,
                    )
                    if quick_params is None:
                        self.console.print("[yellow]Training cancelled by user.[/yellow]")
                        return
            else:
                self.console.print("[yellow]Modification not available for this mode. Training cancelled.[/yellow]")
                return

    # STEP 3: Metadata is ALWAYS saved (mandatory for session persistence)
    # This ensures ALL training sessions are recallable for resume/relaunch
    save_metadata = True
    metadata_path = None

    if not is_resume:
        self.console.print("\n[bold cyan]Reproducibility & Metadata[/bold cyan]")
        self.console.print("  [green]✓ Session metadata will be automatically saved for:[/green]")
        self.console.print("     • Resume capability if training is interrupted")
        self.console.print("     • Complete parameter tracking for reproducibility")
        self.console.print("     • Access via 'Resume/Relaunch Training' option\n")

    # STEP 3.5: Distribution diagnostic before final confirmation
    self._display_training_diagnostic(bundle, quick_params, preloaded_config=preloaded_config)

    # STEP 4: Start training
    confirm_start = Confirm.ask(
        "\n[bold yellow]Start training now?[/bold yellow]",
        default=True
    )

    if not confirm_start:
        self.console.print("[yellow]Training cancelled by user.[/yellow]")
        return

    # Prepare COMPLETE model configuration for metadata (ALL MODES)
    # This ensures FULL reproducibility for quick, benchmark, and custom modes
    model_config = {
        # Core training mode
        'training_mode': mode,

        # Common hyperparameters
        'selected_model': preloaded_config.get('selected_model') if preloaded_config else (quick_params['model_name'] if quick_params else None),
        'epochs': preloaded_config.get('epochs') if preloaded_config else (quick_params['epochs'] if quick_params else None),
        'batch_size': preloaded_config.get('batch_size') if preloaded_config else _get_optimal_batch_size(
            preloaded_config.get('selected_model') if preloaded_config else (quick_params['model_name'] if quick_params else None)
        ),
        'learning_rate': preloaded_config.get('learning_rate') if preloaded_config else 2e-5,
        'early_stopping': True,
        'recommended_model': bundle.recommended_model if hasattr(bundle, 'recommended_model') else None,

        # Advanced training options (will be filled by each mode)
        'use_reinforcement': preloaded_config.get('use_reinforcement') if preloaded_config else (quick_params['reinforced_learning'] if quick_params else True),
        'reinforced_epochs': preloaded_config.get('reinforced_epochs') if preloaded_config else 10,
        'validation_split': preloaded_config.get('validation_split') if preloaded_config else 0.2,
        'test_split': preloaded_config.get('test_split') if preloaded_config else 0.1,
        'stratified_split': preloaded_config.get('stratified_split') if preloaded_config else True,

        # Benchmark-specific parameters (filled if mode=='benchmark')
        'selected_models': None,  # Will be filled by benchmark mode
        'selected_labels': None,  # Will be filled by benchmark mode
        'benchmark_category': None,  # Will be filled if multi-class → binary

        # Quick-specific parameters (filled if mode=='quick')
        'quick_model_name': quick_params['model_name'] if quick_params else None,
        'quick_epochs': quick_params['epochs'] if quick_params else None,

        # Custom-specific parameters (filled if mode=='custom')
        'custom_config': None,  # Will be filled by custom mode

        # Runtime parameters (to be filled during execution)
        'actual_models_trained': [],  # Will be updated post-training
        'training_start_time': None,
        'training_end_time': None,

        # Resume at RL Phase 2
        'skip_to_rl': preloaded_config.get('skip_to_rl', False) if preloaded_config else False,
        'rl_state_path': preloaded_config.get('rl_state_path') if preloaded_config else None,
        # Early stopping & interactive skip
        'early_stopping_patience': preloaded_config.get('early_stopping_patience') if preloaded_config else (quick_params.get('early_stopping_patience') if quick_params else None),
        'interactive_skip': preloaded_config.get('interactive_skip', True) if preloaded_config else (quick_params.get('interactive_skip', True) if quick_params else True),
    }

    # Get session ID BEFORE saving metadata
    # Priority: 1) Passed as parameter (from annotator factory)
    #           2) Reuse the session ID created at the beginning (self.current_session_id)
    #           3) Generate a fallback session_id
    if session_id:
        # Use session_id passed as parameter (e.g., from annotator factory for traceability)
        pass
    elif hasattr(self, 'current_session_id') and self.current_session_id:
        # Reuse the session ID created at the beginning
        session_id = self.current_session_id
    else:
        # Fallback: generate a session_id if not set (should not happen in normal flow)
        self.logger.warning("current_session_id not set, generating fallback session_id")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"training_session_{timestamp}"
        self.current_session_id = session_id

    # Save PRE-TRAINING metadata
    metadata_path = None  # Initialize before conditional block
    if save_metadata:
        try:
            metadata_path = self._save_training_metadata(
                bundle=bundle,
                mode=mode,
                model_config=model_config,
                quick_params=quick_params,  # Pass quick_params for comprehensive capture
                execution_status={
                    'status': 'pending',
                    'started_at': datetime.now().isoformat(),
                    'completed_at': None,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None
                },
                session_id=session_id,
                training_context={
                    'user_choices': {
                        'save_metadata': save_metadata,
                        'modification_requested': not confirm if mode == "quick" else False
                    }
                }
            )
            self.console.print(f"\n[green][OK] Metadata saved for reproducibility[/green]")
            self.console.print(f"[cyan]Metadata File:[/cyan]")
            self.console.print(f"   {metadata_path}\n")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            self.console.print(f"[yellow][!] Failed to save metadata: {e}[/yellow]\n")

    # Execute the selected training mode
    self.console.print("\n[green]✓ Starting training...[/green]\n")
    self.console.print(f"[dim]Session ID: {session_id}[/dim]\n")

    training_result = None
    runtime_params = {}  # Will store actual parameters used during training
    trained_models_map: Dict[str, str] = {}
    try:
        # Only quick mode is supported
        training_result = self._training_studio_run_quick(bundle, model_config, quick_params, session_id)
        runtime_params = training_result.get('runtime_params', {}) if training_result else {}
        def _merge_trained_models(source: Optional[Dict[str, Any]]) -> None:
            if not isinstance(source, dict):
                return
            for key, value in source.items():
                if not value:
                    continue
                try:
                    resolved = Path(value).expanduser().resolve()
                except Exception:
                    resolved = Path(value).expanduser()
                trained_models_map[str(key)] = str(resolved)

        if training_result and isinstance(training_result.get('trained_models'), dict):
            _merge_trained_models(training_result.get('trained_models'))

        session_identifier = session_id or getattr(self, 'current_session_id', None)
        loader = getattr(self, "_load_saved_factory_training_results", None)
        if callable(loader) and session_identifier:
            try:
                recon = loader(
                    session_id=session_identifier,
                    session_dirs=None,
                    training_workflow={}
                )
            except Exception:  # pragma: no cover - defensive
                recon = None
            if recon:
                _merge_trained_models(
                    recon.get("training_result", {}).get("trained_models")
                )

        if trained_models_map:
            if training_result is None:
                training_result = {}
            existing_map = training_result.get('trained_models')
            if isinstance(existing_map, dict):
                _merge_trained_models(existing_map)
            training_result['trained_models'] = trained_models_map
            training_result['models_trained'] = list(trained_models_map.keys())
            training_result['trained_model_paths'] = trained_models_map

        stage_summary = training_result.get('stage_summary', {}) if training_result else {}
        status = 'completed'
        if stage_summary:
            for stage_info in stage_summary.values():
                if not isinstance(stage_info, dict):
                    continue
                if stage_info.get('failed_models') or stage_info.get('errors_recorded'):
                    status = 'completed_with_errors'
                    break

        models_trained_list = training_result.get('models_trained', []) if training_result else []
        if not isinstance(models_trained_list, list):
            if isinstance(models_trained_list, (set, tuple)):
                models_trained_list = list(models_trained_list)
            elif isinstance(models_trained_list, dict):
                models_trained_list = list(models_trained_list.keys())
            elif models_trained_list:
                models_trained_list = [str(models_trained_list)]
            else:
                models_trained_list = []
        metrics_block = training_result.get('metrics') if training_result else {}
        if metrics_block is None:
            metrics_block = {}
        models_trained_count = training_result.get('models_trained_count') if training_result else None
        if not isinstance(models_trained_count, int):
            models_trained_count = len(models_trained_list)

        execution_status = {
            'status': status,
            'completed_at': datetime.now().isoformat(),
            'models_trained': models_trained_list,
            'models_trained_count': models_trained_count,
            'total_expected_models': training_result.get('total_expected_models') if training_result else None,
            'best_model': training_result.get('best_model') if training_result else None,
            'best_f1': training_result.get('best_f1') if training_result else None,
            'best_f1_macro': training_result.get('best_f1_macro') if training_result else None,
            'average_accuracy': metrics_block.get('average_accuracy'),
            'average_f1_macro': metrics_block.get('average_f1_macro'),
            'trained_model_paths': trained_models_map,
            'stage_summary': stage_summary,
            'stage_models': training_result.get('stage_models') if training_result else None,
            'onevsall_value_map': training_result.get('onevsall_value_map') if training_result else None,
            'onevsall_label_info': training_result.get('onevsall_label_info') if training_result else None,
            'training_approach': training_result.get('training_approach') if training_result else None,
            'training_time_seconds': training_result.get('training_time') if training_result else None,
        }

        # Update POST-TRAINING metadata with COMPLETE information
        if save_metadata and metadata_path:
            try:
                # Merge runtime params into model_config for complete save
                final_model_config = {**model_config, **runtime_params}

                # Update both execution_status AND model_config with runtime params
                self._update_training_metadata(
                    metadata_path,
                    execution_status=execution_status,
                    training_context={'trained_model_paths': trained_models_map},
                    model_config=final_model_config
                )
                self.console.print(f"\n[green][OK] Training metadata updated with complete parameters[/green]\n")
            except Exception as e:
                self.logger.error(f"Failed to update metadata: {e}")
        # Generate comprehensive training data logs AFTER training completion
        if hasattr(self, 'current_session_manager') and self.current_session_manager:
            try:
                training_context = {
                    'mode': mode,
                    'training_result': training_result,
                    'runtime_params': runtime_params,
                    'models_trained': training_result.get('models_trained', []) if training_result else [],
                    'models_trained_count': models_trained_count,
                    'trained_model_paths': trained_models_map,
                    'execution_status': execution_status,
                    'stage_summary': stage_summary,
                    'onevsall_label_info': training_result.get('onevsall_label_info', {}) if training_result else {},
                }
                self._log_training_data_distributions(bundle, training_context=training_context)
            except Exception as e:
                self.logger.warning(f"Could not generate comprehensive training logs: {e}")

            # Generate comprehensive summary files (CSV and JSONL) at the end of training
            try:
                from llm_tool.utils.training_summary_generator import generate_training_summaries

                self.console.print("\n[bold cyan]Generating Comprehensive Training Summaries...[/bold cyan]")
                csv_path, jsonl_path = generate_training_summaries(session_id)

                self.console.print("[green]Training summaries generated successfully:[/green]")
                self.console.print(f"  • CSV Summary: [cyan]{csv_path.name}[/cyan]")
                self.console.print(f"  • JSONL Summary: [cyan]{jsonl_path.name}[/cyan]")
                self.console.print(f"\n[dim]Full paths:[/dim]")
                self.console.print(f"  • {csv_path}")
                self.console.print(f"  • {jsonl_path}")

            except Exception as e:
                self.logger.error(f"Failed to generate training summaries: {e}")
                self.console.print(f"[yellow][!] Could not generate comprehensive summaries: {e}[/yellow]")

            # Generate comprehensive cross-model summary chart
            try:
                from llm_tool.trainers.training_metrics_chart import generate_comprehensive_summary_chart

                self.console.print("\n[bold cyan]Generating Comprehensive Summary Chart...[/bold cyan]")

                # Determine session directory for disk scan
                _session_dir = None
                if hasattr(self, 'current_session_manager') and self.current_session_manager:
                    _sm_dir = getattr(self.current_session_manager, 'session_dir', None)
                    if _sm_dir:
                        _session_dir = str(_sm_dir)

                # Determine output directory (same as session dir, or models/)
                _chart_output = _session_dir or str(Path("models") / (session_id or "unknown"))

                # Get model name from training result or config
                _model_name = (
                    (training_result.get('best_model') if training_result else None)
                    or model_config.get('selected_model')
                    or 'unknown'
                )

                # Get training time
                _total_time = (
                    training_result.get('training_time') if training_result else None
                )

                # Also collect results_per_key if available (fallback)
                _results_per_key = (
                    training_result.get('per_key_results') if training_result else None
                )

                comp_chart_path = generate_comprehensive_summary_chart(
                    output_dir=_chart_output,
                    session_id=session_id or 'unknown',
                    model_name=_model_name,
                    session_dir=_session_dir,
                    results_per_key=_results_per_key,
                    training_approach=training_result.get('training_approach', 'normal') if training_result else 'normal',
                    total_training_time=_total_time,
                )

                if comp_chart_path:
                    self.console.print(f"[green]Comprehensive summary chart saved:[/green] {comp_chart_path}")
                else:
                    self.console.print("[yellow][!] Could not generate comprehensive summary chart (insufficient data)[/yellow]")

            except Exception as e:
                self.logger.warning(f"Failed to generate comprehensive summary chart: {e}")
                self.console.print(f"[yellow][!] Comprehensive summary chart generation failed: {e}[/yellow]")

            # NEW: Generate session manifest for complete experiment reproducibility
            try:
                self.console.print("\n[bold cyan]Generating Session Manifest...[/bold cyan]")

                # Collect hyperparameters from quick_params or model_config
                hyperparams = TrainingHyperparameters(
                    learning_rate=quick_params.get('learning_rate', model_config.get('learning_rate', 2e-5)),
                    batch_size=quick_params.get('batch_size', model_config.get('batch_size', 16)),
                    epochs=quick_params.get('epochs', model_config.get('epochs', 5)),
                    warmup_ratio=quick_params.get('warmup_ratio', 0.1),
                )

                # Collect category results from training_result
                categories_results = []
                if training_result and training_result.get('stage_summary'):
                    for category, stage_info in training_result['stage_summary'].items():
                        if isinstance(stage_info, dict):
                            categories_results.append(CategoryTrainingResult(
                                name=category,
                                huggingface_model=model_config.get('model', 'unknown'),
                                training_mode=training_result.get('training_approach', 'multi-class'),
                                num_labels=stage_info.get('num_labels', 2),
                                label_names=stage_info.get('label_names', []),
                                best_epoch=stage_info.get('best_epoch', 0),
                                best_macro_f1=stage_info.get('best_f1_macro', 0),
                                best_accuracy=stage_info.get('best_accuracy', 0),
                                languages=list(stage_info.get('languages', [])),
                                training_time_seconds=stage_info.get('training_time', 0),
                            ))

                # Determine device
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

                # Generate manifest
                manifest = generate_session_manifest(
                    session_id=session_id,
                    output_dir=self.current_session_manager.session_dir,
                    training_mode="normal",
                    training_approach=training_result.get('training_approach', 'multi-class') if training_result else 'multi-class',
                    multi_label=model_config.get('multi_label', False),
                    hyperparameters=hyperparams,
                    categories_results=categories_results,
                    files_generated={
                        'training_csv': str(csv_path) if 'csv_path' in dir() else '',
                        'jsonl_summary': str(jsonl_path) if 'jsonl_path' in dir() else '',
                    },
                    device=device,
                )

                # Finalize and save manifest
                manifest_path = finalize_session_manifest(
                    manifest,
                    self.current_session_manager.session_dir,
                    status=status,
                )

                self.console.print("[green]Session manifest generated successfully:[/green]")
                self.console.print(f"  • {manifest_path}")

            except Exception as e:
                self.logger.error(f"Failed to generate session manifest: {e}")
                self.console.print(f"[yellow][!] Could not generate session manifest: {e}[/yellow]")

    except Exception as e:
        # Update metadata with failure status
        if save_metadata and metadata_path:
            try:
                execution_status = {
                    'status': 'failed',
                    'completed_at': datetime.now().isoformat(),
                    'error_message': str(e)
                }
                self._update_training_metadata(metadata_path, execution_status=execution_status)
            except:
                pass
        raise  # Re-raise the exception

    # Return the complete training result payload to callers (Annotator Factory integration relies on this)
    return training_result

def _ensure_training_models_loaded(self) -> None:
    if self.available_trainer_models:
        return

    if HAS_RICH and self.console:
        with self.console.status("[cyan]Detecting available training backbones...[/cyan]"):
            self.available_trainer_models = self.trainer_model_detector.get_available_models()
    else:
        self.available_trainer_models = self.trainer_model_detector.get_available_models()

def _training_studio_show_model_catalog(self) -> None:
    if not self.available_trainer_models:
        return

    table = Table(title="Available Model Categories (70+ models)", border_style="blue", expand=True)
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Models (sample)", style="white", ratio=1, overflow="fold")

    # Define display order for categories
    category_order = [
        "Multilingual Models",
        "Long Document Models",
        "Long Document Models - French",
        "Long Document Models - Spanish",
        "Long Document Models - German",
        "Long Document Models - Italian",
        "Long Document Models - Portuguese",
        "Long Document Models - Dutch",
        "Long Document Models - Polish",
        "Long Document Models - Chinese",
        "Long Document Models - Japanese",
        "Long Document Models - Arabic",
        "Long Document Models - Russian",
        "Efficient Models",
        "English Models",
        "French Models",
        "Other Language Models"
    ]

    # Display categories in order
    for category in category_order:
        if category in self.available_trainer_models:
            models = self.available_trainer_models[category]
            sample = ", ".join(model["name"] for model in models[:2])
            if len(models) > 2:
                sample += f" (+{len(models) - 2} more)"
            table.add_row(category, sample)

    # Add any remaining categories not in the order
    for category, models in self.available_trainer_models.items():
        if category not in category_order:
            sample = ", ".join(model["name"] for model in models[:2])
            if len(models) > 2:
                sample += f" (+{len(models) - 2} more)"
            table.add_row(category, sample)

    self.console.print(table)


    def _resolve_existing_column(self,
                                 df: pd.DataFrame,
                                 requested_column: Optional[str],
                                 column_label: str,
                                 fallback_candidates: Optional[List[str]] = None) -> Optional[str]:
        """
        Remap a persisted column reference (name or index) to an existing column in the current
        dataframe. Resume workflows often store positional indices (e.g., \"2\") which no longer
        match when schema changes, so we reconcile here to keep downstream steps functional.
        """
        if df is None or requested_column is None:
            return requested_column

        available_columns = list(df.columns)
        if requested_column in available_columns:
            return requested_column

        resolved_column = requested_column

        # 1) Handle numeric index persisted as a string (e.g., "2")
        if isinstance(requested_column, str) and requested_column.isdigit():
            idx = int(requested_column)
            if 0 <= idx < len(available_columns):
                resolved_column = available_columns[idx]

        # 2) Case-insensitive name match
        if resolved_column not in available_columns and isinstance(requested_column, str):
            lower_map = {col.lower(): col for col in available_columns}
            key = requested_column.lower()
            if key in lower_map:
                resolved_column = lower_map[key]

        # 3) Explicit fallback candidates (ordered by priority)
        if resolved_column not in available_columns and fallback_candidates:
            for candidate in fallback_candidates:
                if candidate in available_columns:
                    resolved_column = candidate
                    break

        # If no match was found, leave the original value so downstream logic can signal the issue.
        if resolved_column not in available_columns:
            return requested_column

        if self.console and resolved_column != requested_column:
            self.console.print(
                f"[yellow]Stored {column_label} '{requested_column}' not found. "
                f"Using '{resolved_column}' instead.[/yellow]"
            )

        return resolved_column

def _confirm_language_selection(self,
                                df,
                                text_column: str,
                                lang_counts: Dict[str, int],
                                detected_languages_per_text: List[Optional[str]],
                                data_path: Path,
                                lang_column: Optional[str] = None,
                                console: Optional[Console] = None) -> Tuple[Set[str], Optional[str], Dict[str, int]]:
    """Unified confirmation workflow for language selection, used across Training Arena and Annotator Factory."""
    console = console or self.console

    language_distribution: Dict[str, int] = dict(lang_counts)
    confirmed_languages: Set[str] = set(k for k, v in lang_counts.items() if v > 0)

    # Display detected languages if available
    total = sum(language_distribution.values())
    if total > 0:
        console.print(f"\n[bold]Languages Detected ({total:,} texts analyzed):[/bold]")

        lang_table = Table(border_style="cyan", show_header=True, header_style="bold", expand=True)
        lang_table.add_column("Language", style="cyan", no_wrap=True)
        lang_table.add_column("Count", style="yellow", justify="right", no_wrap=True)
        lang_table.add_column("Percentage", style="green", justify="right", no_wrap=True)

        for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            lang_table.add_row(lang.upper(), f"{count:,}", f"{percentage:.1f}%")

        console.print(lang_table)
    else:
        console.print("[yellow]Could not detect languages automatically[/yellow]")

    # Handle low-percentage languages
    LOW_PERCENTAGE_THRESHOLD = 1.0
    minority_languages = {}
    majority_languages = {}
    if total > 0:
        for lang, count in language_distribution.items():
            percentage = (count / total * 100) if total > 0 else 0
            if percentage >= LOW_PERCENTAGE_THRESHOLD:
                majority_languages[lang] = count
            else:
                minority_languages[lang] = count

    # Provide options to adjust minority languages
    if minority_languages:
        console.print(f"\n[yellow][!] Warning: {len(minority_languages)} language(s) detected with very low percentage (< {LOW_PERCENTAGE_THRESHOLD}%):[/yellow]")
        for lang, count in sorted(minority_languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            console.print(f"  • {lang.upper()}: {count} texts ({percentage:.2f}%)")

        console.print("\n[dim]These are likely detection errors. You have options:[/dim]")
        console.print("  [cyan]1. exclude[/cyan] - Exclude ALL low-percentage languages from training")
        console.print("  [cyan]2. keep[/cyan] - Keep ALL detected languages (not recommended)")
        console.print("  [cyan]3. select[/cyan] - Manually select which languages to keep")
        console.print("  [cyan]4. correct[/cyan] - Force ALL minority languages to a single language (quick fix)")

        minority_action = Prompt.ask(
            "\n[bold yellow]How to handle low-percentage languages?[/bold yellow]",
            choices=["exclude", "keep", "select", "correct"],
            default="correct"
        )

        if minority_action == "correct":
            console.print("\n[bold cyan]Quick Language Correction[/bold cyan]\n")

            all_supported_langs = [
                'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja',
                'ar', 'pl', 'tr', 'ko', 'hi', 'sv', 'no', 'da', 'fi', 'cs',
                'el', 'he', 'ro', 'uk', 'bg', 'hr', 'vi', 'th', 'id', 'fa'
            ]
            majority_lang = max(majority_languages.items(), key=lambda x: x[1])[0] if majority_languages else 'en'

            console.print(f"[bold]Available languages:[/bold]")
            console.print(f"  • Majority language detected: [green]{majority_lang.upper()}[/green] ({majority_languages.get(majority_lang, 0)} texts)")
            console.print(f"  • All supported: {', '.join([l.upper() for l in all_supported_langs])}")

            correction_target = Prompt.ask(
                f"\n[bold yellow]Force ALL minority languages to which language?[/bold yellow]",
                default=majority_lang
            ).lower().strip()

            if correction_target not in all_supported_langs:
                console.print(f"[yellow]Warning: '{correction_target}' not in standard list, but will be used anyway[/yellow]")

            total_corrected = sum(minority_languages.values())
            reclassification_map = language_distribution.get('_reclassification_map', {})
            for minority_lang in minority_languages.keys():
                if minority_lang in language_distribution:
                    del language_distribution[minority_lang]
                reclassification_map[minority_lang] = correction_target

            if correction_target in language_distribution:
                language_distribution[correction_target] += total_corrected
            else:
                language_distribution[correction_target] = total_corrected

            language_distribution['_reclassification_map'] = reclassification_map

            if detected_languages_per_text:
                for i in range(len(detected_languages_per_text)):
                    if detected_languages_per_text[i] in minority_languages:
                        detected_languages_per_text[i] = correction_target

            console.print(f"\n[green]✓ Corrected {total_corrected} texts from {len(minority_languages)} languages to {correction_target.upper()}[/green]")

            update_table = Table(title="Updated Language Distribution", border_style="green", expand=True)
            update_table.add_column("Language", style="cyan", justify="center")
            update_table.add_column("Count", justify="right")
            update_table.add_column("Percentage", justify="right")

            new_total = sum(v for k, v in language_distribution.items() if not k.startswith('_'))
            for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
                if isinstance(count, (int, float)) and count > 0 and not lang.startswith('_'):
                    percentage = (count / new_total) * 100 if new_total > 0 else 0
                    update_table.add_row(lang.upper(), f"{count:,}", f"{percentage:.1f}%")

            console.print(update_table)

        elif minority_action == "exclude":
            for lang in minority_languages.keys():
                language_distribution[lang] = 0

            if detected_languages_per_text:
                for i in range(len(detected_languages_per_text)):
                    if detected_languages_per_text[i] in minority_languages:
                        detected_languages_per_text[i] = None

            confirmed_languages = set(lang for lang, count in language_distribution.items()
                                      if isinstance(count, (int, float)) and count >= LOW_PERCENTAGE_THRESHOLD)
            excluded_count = sum(minority_languages.values())
            console.print(f"\n[yellow]✗ Excluded {excluded_count} texts from {len(minority_languages)} low-percentage language(s)[/yellow]")
            console.print(f"[green]Final languages: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

        elif minority_action == "keep":
            console.print("[yellow][!] Keeping all detected languages (including low-percentage ones)[/yellow]")

        elif minority_action == "select":
            console.print("\n[bold cyan]Language Selection:[/bold cyan]")
            console.print(f"[dim]Select which languages to keep for training (from all {len(language_distribution)} detected)[/dim]\n")

            console.print("[bold]All Detected Languages:[/bold]")
            for i, (lang, count) in enumerate(sorted(language_distribution.items(), key=lambda x: x[1], reverse=True), 1):
                if lang.startswith('_'):
                    continue
                percentage = (count / total * 100) if total > 0 else 0
                status = "[green]majority[/green]" if lang in majority_languages else "[yellow][!] minority[/yellow]"
                console.print(f"  {i:2d}. {lang.upper():5s} - {count:6,} texts ({percentage:5.2f}%) {status}")

            console.print("\n[bold yellow]Select languages to KEEP:[/bold yellow]")
            console.print("[dim]Enter language codes separated by commas (e.g., 'fr,en,de')[/dim]")
            console.print("[dim]Press Enter without typing to keep ALL languages[/dim]")

            selected_langs = Prompt.ask("\n[bold]Languages to keep[/bold]", default="")

            if selected_langs.strip():
                selected_set = set([l.strip().lower() for l in selected_langs.split(',') if l.strip()])
                invalid_langs = selected_set - set(language_distribution.keys())
                if invalid_langs:
                    console.print(f"[yellow][!] Warning: These languages were not detected: {', '.join(invalid_langs)}[/yellow]")
                    selected_set = selected_set - invalid_langs

                for lang in list(language_distribution.keys()):
                    if not lang.startswith('_') and lang not in selected_set:
                        language_distribution[lang] = 0

                if detected_languages_per_text:
                    for i in range(len(detected_languages_per_text)):
                        if detected_languages_per_text[i] and detected_languages_per_text[i] not in selected_set:
                            detected_languages_per_text[i] = None

                confirmed_languages = selected_set
                kept_count = sum([lang_counts.get(lang, 0) for lang in selected_set])
                excluded_count = total - kept_count
                console.print(f"\n[green]✓ Kept {len(selected_set)} language(s): {', '.join([l.upper() for l in sorted(selected_set)])}[/green]")
                console.print(f"[dim]  → {kept_count:,} texts kept, {excluded_count:,} texts excluded[/dim]")
            else:
                console.print("[green]Keeping all detected languages[/green]")

    # Final confirmation
    filtered_distribution = {lang: count for lang, count in language_distribution.items()
                             if not lang.startswith('_') and isinstance(count, (int, float))}
    confirmed_languages = set(lang for lang, count in filtered_distribution.items() if count > 0)

    if confirmed_languages:
        lang_list = ', '.join([l.upper() for l in sorted(confirmed_languages)])
        lang_confirmed = Confirm.ask(
            f"\n[bold]Final languages: {lang_list}. Is this correct?[/bold]",
            default=True
        )

        if not lang_confirmed:
            console.print("\n[yellow]Override with manual selection[/yellow]")

            # Check if there are potential language columns in the dataframe
            potential_lang_cols = [col for col in df.columns if col.lower() in ('lang', 'language', 'langue', 'idioma', 'sprache')]
            use_col = False  # Initialize to avoid undefined variable

            if potential_lang_cols:
                console.print(f"[cyan]Found potential language columns: {', '.join(potential_lang_cols)}[/cyan]")
                use_col = Confirm.ask(
                    f"Use an existing column for languages?",
                    default=True
                )
                if use_col:
                    if len(potential_lang_cols) == 1:
                        lang_column = potential_lang_cols[0]
                    else:
                        lang_column = Prompt.ask(
                            "Which column?",
                            choices=potential_lang_cols,
                            default=potential_lang_cols[0]
                        )

                    # Extract languages from the column
                    lang_series = df[lang_column].dropna().astype(str).str.strip().str.upper()
                    unique_langs = set(lang_series.unique())
                    confirmed_languages = set([l.lower() for l in unique_langs if l])

                    # Update detected_languages_per_text from the column
                    detected_languages_per_text = []
                    for val in df.loc[text_mask, lang_column]:
                        if pd.notna(val) and str(val).strip():
                            detected_languages_per_text.append(str(val).strip().lower())
                        else:
                            detected_languages_per_text.append(None)

                    # Update language distribution
                    language_distribution = {}
                    for lang in confirmed_languages:
                        count = lang_series.str.lower().eq(lang).sum()
                        language_distribution[lang] = count

                    console.print(f"[green]Using column '{lang_column}': {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                else:
                    # Fall back to manual entry
                    manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
                    console.print(f"[green]Manual override: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
            else:
                manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
                console.print(f"[green]Manual override: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

            # Clean up language distribution if manually entered
            if not potential_lang_cols or not use_col:
                for lang in list(language_distribution.keys()):
                    if lang.startswith('_'):
                        continue
                    if lang not in confirmed_languages:
                        language_distribution[lang] = 0

                if detected_languages_per_text:
                    for i in range(len(detected_languages_per_text)):
                        if detected_languages_per_text[i] and detected_languages_per_text[i] not in confirmed_languages:
                            detected_languages_per_text[i] = None
        else:
            console.print("[green]Languages confirmed from analysis[/green]")
    else:
        console.print("[yellow]No languages confirmed. Please specify manually if required.[/yellow]")
        manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
        if manual_langs.strip():
            confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
            for lang in list(language_distribution.keys()):
                if lang not in confirmed_languages and not lang.startswith('_'):
                    language_distribution[lang] = 0

    # Update DataFrame and persist language column if possible
    text_mask = df[text_column].notna()
    target_column = lang_column

    if detected_languages_per_text and len(detected_languages_per_text) == text_mask.sum():
        final_langs = []
        for lang in detected_languages_per_text:
            final_langs.append(lang if lang and str(lang).strip() else None)

        if target_column is None:
            target_column = 'language'
            df[target_column] = None

        df.loc[text_mask, target_column] = final_langs

        if data_path:
            try:
                df.to_csv(data_path, index=False)
                console.print(f"[dim]Language data saved to column '{target_column}'[/dim]")
            except Exception as exc:
                self.logger.warning(f"Could not save language updates: {exc}")

        # Recalculate distribution from final languages
        recalculated = {}
        for lang in final_langs:
            if lang:
                recalculated[lang] = recalculated.get(lang, 0) + 1
        for lang, count in recalculated.items():
            language_distribution[lang] = count
        for lang in list(language_distribution.keys()):
            if not lang.startswith('_') and lang not in recalculated:
                language_distribution[lang] = 0

    return confirmed_languages, target_column, language_distribution

def _training_studio_intelligent_dataset_selector(
    self,
    format_type: str
) -> Optional[Dict[str, Any]]:
    """
    Universal sophisticated interface for dataset and column selection.
    Adapted specifically for Training Arena with:
    - Automatic dataset detection
    - Intelligent column analysis with confidence scores
    - Category/label detection and display
    - Sophisticated ID strategy (single/combine/none)
    - Model recommendations based on languages and data

    Args:
        format_type: One of 'llm-json', 'category-csv', 'binary-long', 'jsonl-single', 'jsonl-multi'

    Returns:
        Dictionary with selected dataset path and all column information, or None if cancelled
    """

    # Step 1: Dataset Detection and Selection
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold cyan]  STEP 1:[/bold cyan] [bold white]Dataset Selection[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[dim]Select your annotated dataset file to prepare for training.[/dim]\n")

    # Show detected datasets if available
    if self.detected_datasets:
        datasets_table = Table(title="Detected Datasets", border_style="cyan", expand=True)
        datasets_table.add_column("#", style="cyan", width=3)
        datasets_table.add_column("Name", style="white", no_wrap=True)
        datasets_table.add_column("Format", style="yellow", no_wrap=True)
        datasets_table.add_column("Size", style="green", no_wrap=True)
        datasets_table.add_column("Origin", style="magenta", ratio=1, overflow="fold")

        for i, ds in enumerate(self.detected_datasets, 1):  # Show ALL datasets
            # Calculate file size
            try:
                if hasattr(ds, 'path') and ds.path.exists():
                    size_bytes = ds.path.stat().st_size
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = "—"
            except Exception as e:
                self.logger.debug(f"Could not get size for {ds.path}: {e}")
                size_str = "—"

            # Determine origin label, preferring recorded source
            origin_label = getattr(ds, "source", None)
            if not origin_label and hasattr(ds, "path"):
                try:
                    origin_label = str(ds.path.parent.relative_to(Path.cwd()))
                except Exception:
                    origin_label = ds.path.parent.name if ds.path.parent.name else ds.path.parent.as_posix()
            origin_label = origin_label or "data"

            datasets_table.add_row(
                str(i),
                ds.path.name if hasattr(ds, 'path') else "—",
                ds.format if hasattr(ds, 'format') else "—",
                size_str,
                origin_label
            )

        self.console.print(datasets_table)
        self.console.print()

        use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
        if use_detected:
            choice = self._int_prompt_with_validation("Select dataset", 1, 1, len(self.detected_datasets))
            data_path = self.detected_datasets[choice - 1].path
        else:
            data_path = Path(self._prompt_file_path("Dataset path"))
    else:
        self.console.print("[dim]No datasets auto-detected in data/ folder[/dim]")
        data_path = Path(self._prompt_file_path("Dataset path"))

    self.console.print(f"[green]Selected: {data_path.name} ({data_path.suffix[1:]})[/green]\n")

    # Step 2: Intelligent File Analysis
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold cyan]  STEP 2:[/bold cyan] [bold white]Analyzing Dataset Structure[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[dim]Analyzing columns, detecting types, and extracting samples...[/dim]")

    analysis = DataDetector.analyze_file_intelligently(data_path)

    if analysis['issues']:
        self.console.print("\n[yellow][!]  Analysis warnings:[/yellow]")
        for issue in analysis['issues']:
            self.console.print(f"  • {issue}")

    # Step 3: Intelligent Language Detection (MOVED HERE - before column selection)
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold cyan]  STEP 3:[/bold cyan] [bold white]Language Detection[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[dim]Detecting languages to recommend the best training model.[/dim]\n")

    languages_found_in_column = set(analysis.get('languages_detected', {}).keys())
    confirmed_languages = set()
    lang_column = None
    text_length_stats = {}  # Initialize - will be populated after text column selection
    languages_from_content = {}
    apply_auto_detection = True  # Always perform automatic detection at this stage

    # Check if we have a language column with detected languages
    has_lang_column = bool(analysis.get('language_column_candidates'))

    if has_lang_column and languages_found_in_column:
        # Option 1: Language column exists - offer to use it or detect automatically
        self.console.print("[bold]Languages Found in Column:[/bold]")
        for lang, count in analysis['languages_detected'].items():
            self.console.print(f"  • {lang.upper()}: {count:,} rows")

        lang_column_candidate = analysis['language_column_candidates'][0]
        self.console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")

        use_lang_column = Confirm.ask(
            f"\n[bold]Use language column '{lang_column_candidate}'?[/bold]",
            default=True
        )

        if use_lang_column:
            confirmed_languages = languages_found_in_column
            lang_column = lang_column_candidate
            self.console.print(f"[green]Using language column: {lang_column}[/green]")
        else:
            # User said no to language column - offer automatic detection
            self.console.print("\n[yellow]Language column not used. Applying automatic detection...[/yellow]")
    else:
        # Option 2: No language column - go straight to automatic detection
        self.console.print("[yellow]No language column detected[/yellow]")

    # We need to detect text column first for content-based language detection
    # Quick text column detection for language analysis
    temp_column_info = self._detect_text_columns(data_path)
    temp_text_column = None
    if temp_column_info.get('text_candidates'):
        temp_text_column = temp_column_info['text_candidates'][0]['name']
    else:
        temp_text_column = "text"  # fallback

    # Automatic language detection from text content
    # Automatic language detection from text content (or confirmation of existing language column)
    language_distribution: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    detected_languages_per_text: List[Optional[str]] = []
    detection_failed = False

    try:
        import pandas as pd
        import json

        df = pd.read_csv(data_path) if data_path.suffix == '.csv' else pd.read_json(data_path, lines=data_path.suffix == '.jsonl')
        text_mask = df[temp_text_column].notna()

        if lang_column:
            lang_column = self._resolve_existing_column(
                df,
                lang_column,
                "language column",
                fallback_candidates=["language", "lang"]
            )
        if lang_column and lang_column in df.columns:
            self.console.print("[dim]Using existing language column '{}' for analysis.[/dim]".format(lang_column))
            lang_series = df.loc[text_mask, lang_column].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip() else None
            )
            for value in lang_series.tolist():
                detected_languages_per_text.append(value)
                if value:
                    lang_counts[value] = lang_counts.get(value, 0) + 1

        if apply_auto_detection and not lang_counts:
            self.console.print("[dim]Analyzing ALL texts to detect languages (this may take a moment)...[/dim]")
            from llm_tool.utils.language_detector import LanguageDetector
            from tqdm import tqdm

            detector = LanguageDetector()
            self.console.print("[dim]Analyzing {} texts...[/dim]".format(int(text_mask.sum())))

            for text in tqdm(df.loc[text_mask, temp_text_column], desc="Detecting languages", disable=not HAS_RICH):
                if text and len(str(text).strip()) > 10:
                    try:
                        detected = detector.detect(str(text))
                        lang_code = None
                        if isinstance(detected, dict):
                            lang_code = detected.get('language') if detected.get('confidence', 0) >= 0.7 else None
                        elif isinstance(detected, str):
                            lang_code = detected
                        if lang_code:
                            lang_code = str(lang_code).lower()
                            lang_counts[lang_code] = lang_counts.get(lang_code, 0) + 1
                            detected_languages_per_text.append(lang_code)
                        else:
                            detected_languages_per_text.append(None)
                    except Exception as detect_exc:
                        self.logger.debug("Language detection failed for text: {}".format(detect_exc))
                        detected_languages_per_text.append(None)
                else:
                    detected_languages_per_text.append(None)

        if lang_column and lang_column in df.columns:
            # Ensure we always have a fallback distribution based on the provided language column
            from llm_tool.utils.language_normalizer import LanguageNormalizer

            normalized_langs_from_column: List[str] = []
            column_lang_counts: Dict[str, int] = {}

            for raw_value in df[lang_column].fillna("").astype(str):
                normalized = LanguageNormalizer.normalize_language(raw_value)
                if not normalized:
                    normalized = raw_value.strip().lower() or "unknown"

                normalized_langs_from_column.append(normalized)
                if normalized != "unknown":
                    column_lang_counts[normalized] = column_lang_counts.get(normalized, 0) + 1

            if not lang_counts and column_lang_counts:
                lang_counts = column_lang_counts

            if not detected_languages_per_text:
                detected_languages_per_text = normalized_langs_from_column

        confirmed_languages: Set[str] = set()
        if lang_counts or detected_languages_per_text:
            confirmed_languages, lang_column, language_distribution = self._confirm_language_selection(
                df=df,
                text_column=temp_text_column,
                lang_counts=lang_counts,
                detected_languages_per_text=detected_languages_per_text,
                data_path=data_path,
                lang_column=lang_column
            )
    except Exception as e:
        detection_failed = True
        self.logger.debug("Language detection from content failed: {}".format(e))
        self.console.print("[yellow]Automatic detection failed. Please specify manually[/yellow]")
        manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
        confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()]) if manual_langs.strip() else set()
        self.console.print("[yellow]Standard models will be used (texts will be truncated to 512 tokens)[/yellow]")

    # Step 4: Text Column Selection with Sophisticated Table
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold cyan]  STEP 4:[/bold cyan] [bold white]Text Column Selection[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold]What You Need to Select:[/bold]")
    self.console.print("   [cyan]• Text Column[/cyan] - Contains the text data to train on (input for predictions)\n")

    column_info = self._detect_text_columns(data_path)
    all_columns = column_info.get('all_columns', analysis.get('all_columns', []))

    candidate_names = [candidate['name'] for candidate in column_info.get('text_candidates', [])]

    if column_info.get('text_candidates'):
        self.console.print("[dim]Detected text columns (sorted by confidence):[/dim]")

        col_table = Table(border_style="blue", expand=True)
        col_table.add_column("#", style="cyan", width=5, no_wrap=True)
        col_table.add_column("Column", style="white", no_wrap=True)
        col_table.add_column("Confidence", style="yellow", no_wrap=True)
        col_table.add_column("Avg Length", style="green", no_wrap=True)
        col_table.add_column("Sample", style="dim", overflow="fold", ratio=1)

        for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
            conf_color = {
                "high": "[green]High[/green]",
                "medium": "[yellow]Medium[/yellow]",
                "low": "[orange1]Low[/orange1]",
                "very_low": "[red]Very Low[/red]"
            }
            conf_display = conf_color.get(candidate.get('confidence', 'low'), candidate.get('confidence', 'Unknown'))

            sample = candidate.get('sample', '')
            sample_display = (sample[:50] + "...") if len(sample) > 50 else sample

            col_table.add_row(
                str(i),
                candidate['name'],
                conf_display,
                f"{candidate.get('avg_length', 0):.0f} chars",
                sample_display
            )

        self.console.print(col_table)
        if all_columns:
            self.console.print(f"\n[dim]All columns ({len(all_columns)}): {', '.join(all_columns)}[/dim]")

        default_text_col = candidate_names[0]
    else:
        self.console.print("[yellow]No text columns auto-detected[/yellow]")
        if all_columns:
            self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")
        default_text_col = "text"

    # Ask for text column with validation
    while True:
        raw_choice = Prompt.ask(
            "\n[bold yellow]Enter column name[/bold yellow] (or choose from above)",
            default=default_text_col
        )
        normalized_choice = _normalize_column_choice(raw_choice, all_columns, candidate_names)

        if normalized_choice:
            text_column = normalized_choice
            break

        if not all_columns:
            text_column = raw_choice.strip()
            break

        self.console.print(f"[red]✗ Column selection '{raw_choice}' could not be resolved.[/red]")
        self.console.print("[dim]Enter the column name or the number shown in the table.[/dim]")
        self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

    # Step 4b: CRITICAL - Text Length Analysis (MUST be done AFTER text column selection)
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    text_length_stats = self.analyze_text_lengths(
        data_path=data_path,
        text_column=text_column,  # Use the ACTUAL selected column
        display_results=True,
        step_label=f"{resolve_step_label('text_length', 'STEP 5')}: Text Length Analysis"
    )
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

    # Store stats for later use in model selection (no user choice yet)
    # User will choose strategy in model selection step

    # Step 5: Label/Category Column Selection with Category Analysis
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    label_step = resolve_step_label("label_selection", "STEP 5")
    self.console.print(f"[bold cyan]  {label_step}:[/bold cyan] [bold white]Label/Category Column Selection[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold]What You Need to Select:[/bold]")
    self.console.print("   [cyan]• Label Column[/cyan] - Contains the labels/categories (what the model will learn to predict)\n")

    label_column_default = "labels" if "multi" in format_type else "label"

    annotation_candidates = analysis.get('annotation_column_candidates', [])
    if annotation_candidates:
        best_label = annotation_candidates[0]['name']
        label_column_default = best_label

        self.console.print(f"[green]Label column detected: '{best_label}'[/green]")

        stats = analysis.get('annotation_stats', {}).get(best_label, {})
        fill_rate = stats.get('fill_rate', 0)
        if fill_rate > 0:
            self.console.print(f"[dim]  ({fill_rate*100:.1f}% of rows have labels)[/dim]")

        # NOUVEAU: Analyze and display categories/labels
        try:
            import pandas as pd
            import json
            df = pd.read_csv(data_path) if data_path.suffix == '.csv' else pd.read_json(data_path, lines=data_path.suffix == '.jsonl')

            if best_label in df.columns:
                # Get unique categories and their counts
                if "multi" in format_type:
                    # Multi-label: try to parse lists/JSON
                    all_labels = []
                    for val in df[best_label].dropna():
                        if isinstance(val, list):
                            all_labels.extend(val)
                        elif isinstance(val, str):
                            try:
                                parsed = json.loads(val)
                                if isinstance(parsed, list):
                                    all_labels.extend(parsed)
                            except:
                                pass
                    label_counts = pd.Series(all_labels).value_counts()
                else:
                    # Single-label: direct value counts
                    label_counts = df[best_label].value_counts()

                # Display categories table
                if len(label_counts) > 0:
                    self.console.print(f"\n[bold]Detected {len(label_counts)} Categories:[/bold]")

                    cat_table = Table(border_style="green", show_header=True, header_style="bold cyan", expand=True)
                    cat_table.add_column("#", style="cyan", width=5)
                    cat_table.add_column("Category", style="white", ratio=1, overflow="fold")
                    cat_table.add_column("Count", style="yellow", no_wrap=True, justify="right")
                    cat_table.add_column("Percentage", style="green", no_wrap=True, justify="right")

                    total = label_counts.sum()
                    for i, (cat, count) in enumerate(label_counts.head(20).items(), 1):
                        percentage = (count / total * 100) if total > 0 else 0
                        cat_table.add_row(
                            str(i),
                            str(cat)[:28],
                            f"{count:,}",
                            f"{percentage:.1f}%"
                        )

                    if len(label_counts) > 20:
                        cat_table.add_row("...", f"... and {len(label_counts) - 20} more", "...", "...")

                    self.console.print(cat_table)
                    self.console.print(f"[dim]Total samples: {total:,}[/dim]")
        except Exception as e:
            self.logger.debug(f"Could not analyze categories: {e}")

    if all_columns:
        self.console.print(f"\n[dim]Available columns: {', '.join(all_columns)}[/dim]")

    # Ask for label column with validation
    while True:
        label_column = Prompt.ask("\n[bold yellow]Category/label column[/bold yellow]", default=label_column_default)
        if label_column in all_columns:
            break
        self.console.print(f"[red]✗ Column '{label_column}' not found in dataset![/red]")
        self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

    # Step 6: ID Column Selection with Modernized Interface
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    identifier_step = resolve_step_label("identifier_selection", "STEP 7")
    self.console.print(f"[bold cyan]  {identifier_step}:[/bold cyan] [bold white]Identifier Column Selection (Optional)[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

    # Load dataframe to detect ID candidates
    try:
        if data_path.suffix.lower() == '.csv':
            df_for_id_check = pd.read_csv(data_path, nrows=1000)
        elif data_path.suffix.lower() == '.json':
            df_for_id_check = pd.read_json(data_path, lines=False, nrows=1000)
        elif data_path.suffix.lower() == '.jsonl':
            df_for_id_check = pd.read_json(data_path, lines=True, nrows=1000)
        elif data_path.suffix.lower() in ['.xlsx', '.xls']:
            df_for_id_check = pd.read_excel(data_path, nrows=1000)
        elif data_path.suffix.lower() == '.parquet':
            df_for_id_check = pd.read_parquet(data_path).head(1000)
        else:
            df_for_id_check = pd.read_csv(data_path, nrows=1000)  # Fallback

        # Use modernized ID selection function
        id_column = DataDetector.display_and_select_id_column(
            self.console,
            df_for_id_check,
            text_column=text_column,
            step_label="",  # Empty since we already printed the header
            data_path=data_path
        )
    except Exception as e:
        self.logger.warning(f"Could not load dataframe for ID detection: {e}")
        self.console.print(f"[yellow][!] Could not analyze ID columns: {e}[/yellow]")
        self.console.print("[dim]An automatic ID will be generated[/dim]")
        id_column = None

    # Model selection will be done later when training mode is chosen
    # Store languages and text characteristics for later use
    model_to_use = None
    model_strategy = "multilingual"  # default
    language_model_mapping = {}  # For per-language models

    # Skip model selection - will be done in training mode
    if False and confirmed_languages and len(confirmed_languages) > 1:
        # Multiple languages detected - offer strategy choice
        self.console.print(f"[bold]Dataset contains {len(confirmed_languages)} languages:[/bold]")

        if language_distribution:
            # Filter out metadata keys (like _reclassification_map)
            lang_counts = {k: v for k, v in language_distribution.items() if not k.startswith('_') and isinstance(v, (int, float))}

            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                total = sum(lang_counts.values())
                pct = (count / total * 100) if total > 0 else 0
                self.console.print(f"  • {lang.upper()}: {count:,} texts ({pct:.1f}%)")
        else:
            for lang in sorted(confirmed_languages):
                self.console.print(f"  • {lang.upper()}")

        self.console.print("\n[bold]Model Strategy Options:[/bold]")
        self.console.print("  [cyan]1. multilingual[/cyan] - Train ONE multilingual model for all languages")
        self.console.print("     ✓ Simpler, faster, handles cross-lingual patterns")
        self.console.print("     ✗ May have slightly lower performance per language")
        self.console.print()
        self.console.print("  [cyan]2. specialized[/cyan] - Train SEPARATE specialized models per language")
        self.console.print("     ✓ Best performance for each language")
        self.console.print("     ✗ More training time, requires language column or detection")
        self.console.print()
        self.console.print("  [cyan]3. hybrid[/cyan] - Multilingual model + fine-tuned per-language models")
        self.console.print("     ✓ Best of both worlds")
        self.console.print("     ✗ Most training time and complexity")

        model_strategy = Prompt.ask(
            "\n[bold yellow]Select model strategy[/bold yellow]",
            choices=["multilingual", "specialized", "hybrid"],
            default="multilingual"
        )

        self.console.print(f"\n[green]✓ Selected strategy: {model_strategy}[/green]")

        if model_strategy == "multilingual":
            # Get ONE multilingual model for all languages
            # Consider long-document models if user prefers them
            if text_length_stats.get('user_prefers_long_models', False):
                model_to_use = self._get_long_document_model_recommendation(confirmed_languages)
            else:
                model_to_use = self._get_model_recommendation_from_languages(confirmed_languages)

        elif model_strategy == "specialized":
            # Get specialized model for EACH language
            self.console.print("\n[bold]Selecting specialized models for each language:[/bold]")

            for lang in sorted(confirmed_languages):
                # Consider long-document models if user prefers them
                if text_length_stats.get('user_prefers_long_models', False):
                    lang_recommendations = self._get_long_document_models_for_language(lang)
                else:
                    lang_recommendations = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)

                if lang_recommendations:
                    self.console.print(f"\n[cyan]For {lang.upper()}:[/cyan]")
                    for i, rec in enumerate(lang_recommendations[:3], 1):
                        self.console.print(f"  {i}. {rec['model']} - {rec['reason']}")

                    choice = Prompt.ask(
                        f"Model for {lang.upper()} (1-{min(3, len(lang_recommendations))}, or enter model name)",
                        default="1"
                    )

                    if choice.isdigit() and 0 < int(choice) <= len(lang_recommendations):
                        language_model_mapping[lang] = lang_recommendations[int(choice) - 1]['model']
                    else:
                        language_model_mapping[lang] = choice

                    self.console.print(f"  [green]✓ {lang.upper()}: {language_model_mapping[lang]}[/green]")
                else:
                    # Fallback to multilingual
                    self.console.print(f"[yellow]No specific model for {lang.upper()}, using multilingual[/yellow]")
                    if not model_to_use:
                        model_to_use = self._get_model_recommendation_from_languages(confirmed_languages)
                    language_model_mapping[lang] = model_to_use

        elif model_strategy == "hybrid":
            # First get multilingual base model
            self.console.print("\n[bold]1. Select base multilingual model:[/bold]")
            model_to_use = self._get_model_recommendation_from_languages(confirmed_languages)

            # Then get specialized models for fine-tuning
            self.console.print("\n[bold]2. Select specialized models for fine-tuning:[/bold]")
            for lang in sorted(confirmed_languages):
                lang_recommendations = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)

                if lang_recommendations:
                    self.console.print(f"\n[cyan]Fine-tuning model for {lang.upper()}:[/cyan]")
                    for i, rec in enumerate(lang_recommendations[:3], 1):
                        self.console.print(f"  {i}. {rec['model']}")

                    choice = Prompt.ask(
                        f"Model for {lang.upper()} (1-{min(3, len(lang_recommendations))}, or 'skip')",
                        default="1"
                    )

                    if choice.lower() != 'skip':
                        if choice.isdigit() and 0 < int(choice) <= len(lang_recommendations):
                            language_model_mapping[lang] = lang_recommendations[int(choice) - 1]['model']
                        else:
                            language_model_mapping[lang] = choice

                        self.console.print(f"  [green]✓ {lang.upper()}: {language_model_mapping[lang]}[/green]")

    elif confirmed_languages and len(confirmed_languages) == 1:
        # Single language - get specialized model
        lang = list(confirmed_languages)[0]
        self.console.print(f"[bold]Single language detected: {lang.upper()}[/bold]")

        # Consider long-document models if user prefers them
        if text_length_stats.get('user_prefers_long_models', False):
            lang_recommendations = self._get_long_document_models_for_language(lang)
        else:
            lang_recommendations = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)

        if lang_recommendations:
            self.console.print(f"\n[bold]Recommended Models for {lang.upper()}:[/bold]")
            for i, rec in enumerate(lang_recommendations[:5], 1):
                self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

            choice = Prompt.ask("Select model (1-5, or enter model name)", default="1")

            if choice.isdigit() and 0 < int(choice) <= len(lang_recommendations):
                model_to_use = lang_recommendations[int(choice) - 1]['model']
            else:
                model_to_use = choice

            self.console.print(f"[green]Selected: {model_to_use}[/green]")
    else:
        # No languages detected - use default
        model_to_use = self._get_model_recommendation_from_languages(set())

    # Return all collected information
    return {
        'data_path': data_path,
        'text_column': text_column,
        'label_column': label_column,
        'id_column': id_column,
        'lang_column': lang_column,
        'confirmed_languages': confirmed_languages,
        'language_distribution': language_distribution,  # Exact counts per language
        'text_length_stats': text_length_stats,  # Text length statistics and long-document preference
        'model_strategy': model_strategy,  # multilingual, specialized, or hybrid
        'recommended_model': model_to_use,  # Main/base model
        'language_model_mapping': language_model_mapping,  # Per-language models (if specialized)
        'analysis': analysis
    }

def _training_studio_dataset_wizard(self, builder: TrainingDatasetBuilder) -> Optional[TrainingDataBundle]:
    """
    Intelligent dataset wizard with comprehensive file analysis and guided setup.
    Now supports all formats with smart detection and recommendations.
    """

    # Step 1: Explain format options with Rich table
    self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[bold cyan]  STEP 1:[/bold cyan] [bold white]Dataset Format Selection[/bold white]")
    self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    self.console.print("[dim]Choose the format that matches your annotated data structure.[/dim]\n")

    formats_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
    formats_table.add_column("Format", style="cyan bold", no_wrap=True)
    formats_table.add_column("Description", style="white")
    formats_table.add_column("Example", style="dim", overflow="fold")

    formats_table.add_row(
        "llm-json",
        "CSV with a JSON annotation column\n"
        "✓ Output from LLM annotation pipelines\n"
        "✓ Supports three JSON structures:",
        ""
    )

    self.console.print(formats_table)

    # Show supported JSON structures in detail
    json_table = Table(show_header=True, header_style="bold", border_style="dim", box=box.SIMPLE, expand=True, padding=(0, 2))
    json_table.add_column("Structure", style="cyan", no_wrap=True, width=22)
    json_table.add_column("JSON Example", style="white", ratio=2)
    json_table.add_column("Use Case", style="dim", ratio=1)

    json_table.add_row(
        "Flat scalars",
        '{"sentiment": "positive", "topic": "economy"}',
        "Simple classification\nOne value per key"
    )
    json_table.add_row(
        "Flat lists",
        '{"themes": ["nationalism", "authority"]}',
        "Multi-label classification\nMultiple values per key"
    )
    json_table.add_row(
        "Nested (detected)",
        '{"nationalism": {"detected": "yes",\n "subcategories": ["nation_threat"]}}',
        "LLM annotations with\ndetection + subcategories\n(auto-flattened)"
    )
    self.console.print(json_table)
    self.console.print()
    self.console.print("[dim]  All three structures are auto-detected and handled transparently.[/dim]")
    self.console.print("[dim]  Nested annotations are flattened: detected -> yes/no label, subcategories -> individual labels.[/dim]")
    self.console.print()

    # Development notice for other formats
    other_formats = Table(show_header=False, border_style="dim", box=box.SIMPLE, expand=True, padding=(0, 1))
    other_formats.add_column("", style="dim")
    other_formats.add_column("", style="dim")
    other_formats.add_row("[dim]category-csv[/dim]", "[dim]Simple text,label CSV (under development)[/dim]")
    other_formats.add_row("[dim]binary-long[/dim]", "[dim]Long-format with binary 0/1 values (under development)[/dim]")
    other_formats.add_row("[dim]jsonl-single[/dim]", "[dim]JSONL for single-label (under development)[/dim]")
    other_formats.add_row("[dim]jsonl-multi[/dim]", "[dim]JSONL for multi-label (under development)[/dim]")
    self.console.print(other_formats)
    self.console.print()

    format_choice = Prompt.ask(
        "[bold yellow]Select dataset format[/bold yellow]",
        choices=["llm-json", "cancel", "back"],
        default="llm-json",
    )

    if format_choice == "cancel" or format_choice == "back":
        return None

    if format_choice == "llm-json":
        # Step 2: Dataset Selection
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 2:[/bold cyan] [bold white]Dataset Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Select your annotated dataset file to prepare for training.[/dim]\n")

        # Show detected datasets if available
        if self.detected_datasets:
            datasets_table = Table(title="Detected Datasets", border_style="cyan", expand=True)
            datasets_table.add_column("#", style="cyan", width=3)
            datasets_table.add_column("Name", style="white", no_wrap=True)
            datasets_table.add_column("Format", style="yellow", no_wrap=True)
            datasets_table.add_column("Size", style="green", no_wrap=True)
            datasets_table.add_column("Origin", style="magenta", ratio=1, overflow="fold")

            for i, ds in enumerate(self.detected_datasets, 1):  # Show ALL datasets, not just [:10]
                # Calculate file size
                try:
                    if hasattr(ds, 'path') and ds.path.exists():
                        size_bytes = ds.path.stat().st_size
                        if size_bytes < 1024:
                            size_str = f"{size_bytes} B"
                        elif size_bytes < 1024 * 1024:
                            size_str = f"{size_bytes / 1024:.1f} KB"
                        else:
                            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                    else:
                        size_str = "—"
                except Exception as e:
                    self.logger.debug(f"Could not get size for {ds.path}: {e}")
                    size_str = "—"

                # Determine origin label, using stored source when available
                origin_label = getattr(ds, "source", None)
                if not origin_label and hasattr(ds, "path"):
                    try:
                        origin_label = str(ds.path.parent.relative_to(Path.cwd()))
                    except Exception:
                        origin_label = ds.path.parent.name if ds.path.parent.name else ds.path.parent.as_posix()
                origin_label = origin_label or "data"

                datasets_table.add_row(
                    str(i),
                    ds.path.name if hasattr(ds, 'path') else "—",
                    ds.format if hasattr(ds, 'format') else "—",
                    size_str,
                    origin_label
                )

            self.console.print(datasets_table)
            self.console.print()
            self.console.print("[dim]Tip: You can either:[/dim]")
            self.console.print("[dim]   • Enter the [cyan]#[/cyan] number from the table above (e.g., '1', '13')[/dim]")
            self.console.print("[dim]   • Enter an [cyan]absolute path[/cyan] to any file (e.g., '/Users/name/data/file.csv')[/dim]\n")

            dataset_choice = Prompt.ask("Dataset selection", default="1")

            # Parse choice
            if not dataset_choice or dataset_choice.strip() == "":
                # Empty input - default to first dataset
                self.console.print("[yellow][!] No selection made, defaulting to first dataset[/yellow]")
                csv_path = self.detected_datasets[0].path
            elif dataset_choice.isdigit():
                idx = int(dataset_choice) - 1
                if 0 <= idx < len(self.detected_datasets):
                    csv_path = self.detected_datasets[idx].path
                else:
                    self.console.print("[red]Invalid dataset number[/red]")
                    return None
            else:
                csv_path = Path(dataset_choice)
                # Validate that it's a file, not a directory
                if csv_path.is_dir():
                    self.console.print(f"[red]Error: '{csv_path}' is a directory, not a file[/red]")
                    return None
                if not csv_path.exists():
                    self.console.print(f"[red]Error: File '{csv_path}' does not exist[/red]")
                    return None
        else:
            file_path_str = self._prompt_file_path("Annotated file path (CSV/JSON/Excel/Parquet)")
            csv_path = Path(file_path_str)

        self.console.print(f"[green]Selected: {csv_path.name} ({csv_path.suffix[1:]})[/green]\n")

        # Step 3: File Structure Analysis
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 3:[/bold cyan] [bold white]Analyzing Dataset Structure[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Analyzing columns, detecting types, and extracting samples...[/dim]")
        analysis = DataDetector.analyze_file_intelligently(csv_path)

        # Show analysis results
        if analysis['issues']:
            self.console.print("\n[yellow][!]  Analysis warnings:[/yellow]")
            for issue in analysis['issues']:
                self.console.print(f"  • {issue}")

        # Step 4: Column Selection
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 4:[/bold cyan] [bold white]Column Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold]What You Need to Select:[/bold]")
        self.console.print("   [cyan]• Text Column[/cyan]     - Contains the text data to train on (input for predictions)")
        self.console.print("   [cyan]• Annotation Column[/cyan] - Contains the JSON annotations (labels/categories for training)\n")

        # Auto-suggest text column with all available columns
        text_column_default = "sentence"
        all_columns = analysis.get('all_columns', [])

        # Read CSV to analyze ALL columns
        import pandas as pd
        import json

        # Final validation before reading
        if not csv_path or csv_path.is_dir():
            self.console.print(f"[red]Error: Invalid file path '{csv_path}'[/red]")
            return None
        if not csv_path.exists():
            self.console.print(f"[red]Error: File '{csv_path}' does not exist[/red]")
            return None

        df = pd.read_csv(csv_path)

        text_candidates = analysis.get('text_column_candidates', [])
        annotation_candidates = analysis.get('annotation_column_candidates', [])

        # Create comprehensive column overview table
        if all_columns:
            self.console.print(f"[bold]Dataset Overview ({len(all_columns)} columns, {len(df):,} rows):[/bold]\n")

            # Create detailed columns table
            all_columns_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            all_columns_table.add_column("#", style="dim", width=3, no_wrap=True)
            all_columns_table.add_column("Column Name", style="cyan bold", no_wrap=True)
            all_columns_table.add_column("Type", style="yellow", no_wrap=True)
            all_columns_table.add_column("Sample Values", style="white", overflow="fold", ratio=1)

            for idx, col in enumerate(all_columns, 1):
                # Detect column type
                col_type = "text"
                if col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        col_type = "numeric"
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        col_type = "datetime"
                    else:
                        # Check if it's likely JSON
                        sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
                        if isinstance(sample_val, str) and (sample_val.startswith('{') or sample_val.startswith('[')):
                            col_type = "json/annotation"
                        else:
                            col_type = "text"

                    # Get sample values
                    samples = df[col].dropna().head(3).tolist()
                    if samples:
                        sample_str = ", ".join([str(s)[:30] + "..." if len(str(s)) > 30 else str(s) for s in samples])
                    else:
                        sample_str = "[empty]"
                else:
                    sample_str = "—"

                all_columns_table.add_row(
                    str(idx),
                    col,
                    col_type,
                    sample_str
                )

            self.console.print(all_columns_table)

            # Now show AI suggestions
            self.console.print("\n[bold]Tip: Helpful Suggestions[/bold] [dim](not required - you choose)[/dim]")
            self.console.print("[dim]These are suggestions based on column names and content analysis.[/dim]")
            self.console.print("[dim]You are free to select ANY column from the table above.[/dim]\n")

            suggestions_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.SIMPLE, expand=True)
            suggestions_table.add_column("Purpose", style="yellow bold", no_wrap=True)
            suggestions_table.add_column("Top Suggestion", style="green bold", no_wrap=True)
            suggestions_table.add_column("Why This Column?", style="white", overflow="fold", ratio=1)

            # Text column row
            text_candidates = analysis.get('text_column_candidates', [])
            if text_candidates:
                best_text = text_candidates[0]['name']
                text_column_default = best_text
                text_stats = text_candidates[0]
                avg_len = text_stats.get('avg_length', 0)
                suggestions_table.add_row(
                    "Text Data",
                    best_text,
                    f"Contains text (avg {avg_len:.0f} chars)"
                )
            else:
                suggestions_table.add_row("Text Data", "—", "No automatic suggestion")

            # Annotation column row
            annotation_column_default = "annotation"
            has_annotation_alternatives = False
            annotation_candidates = analysis.get('annotation_column_candidates', [])
            if annotation_candidates:
                best_annotation_info = annotation_candidates[0]
                best_annotation = best_annotation_info['name']
                annotation_column_default = best_annotation
                stats = analysis['annotation_stats'].get(best_annotation, {})
                fill_rate = stats.get('fill_rate', 0)
                is_json = stats.get('is_json', False)
                match_type = best_annotation_info.get('match_type', 'name_pattern')

                if fill_rate > 0:
                    # Build reason text
                    reason_parts = []
                    if is_json:
                        if match_type == 'json_content':
                            reason_parts.append("Auto-detected JSON annotations")
                        else:
                            reason_parts.append("Contains JSON annotations")
                    else:
                        reason_parts.append("Contains labels/categories")
                    reason_parts.append(f"{fill_rate*100:.1f}% filled")

                    suggestions_table.add_row(
                        "Annotations",
                        best_annotation,
                        ", ".join(reason_parts)
                    )

                    # Mark if there are alternatives
                    if len(annotation_candidates) > 1:
                        has_annotation_alternatives = True
                else:
                    suggestions_table.add_row(
                        "Annotations",
                        best_annotation,
                        "[red][!] Column is EMPTY - cannot use[/red]"
                    )
            else:
                suggestions_table.add_row("Annotations", "—", "No automatic suggestion")

            self.console.print(suggestions_table)

            # Show alternatives AFTER the table
            if has_annotation_alternatives and len(annotation_candidates) > 1:
                alternatives = [c['name'] for c in annotation_candidates[1:3]]
                self.console.print(f"[dim]   Other annotation options: {', '.join(alternatives)}[/dim]")

            self.console.print()
        else:
            # Fallback if no columns detected
            if text_candidates:
                best_text = text_candidates[0]['name']
                text_column_default = best_text
                self.console.print(f"\n[green]✓ Suggested text column: '{best_text}'[/green]")

            annotation_column_default = "annotation"
            if annotation_candidates:
                best_annotation = annotation_candidates[0]['name']
                annotation_column_default = best_annotation
                stats = analysis['annotation_stats'].get(best_annotation, {})
                fill_rate = stats.get('fill_rate', 0)
                if fill_rate > 0:
                    self.console.print(f"[green]Suggested annotation column: '{best_annotation}' ({fill_rate*100:.1f}% filled)[/green]")
                else:
                    self.console.print(f"[red][!] Suggested annotation column '{best_annotation}' is EMPTY - cannot be used for training![/red]")

        self.console.print("[bold yellow]Your Turn - Select Columns:[/bold yellow]")
        self.console.print("[dim]   → Press [bold]Enter[/bold] to use the suggested column[/dim]")
        self.console.print("[dim]   → Or type ANY column name from the table above[/dim]")
        self.console.print("[dim]   → The suggestions are helpful, but not mandatory![/dim]\n")

        # Ask for text column with validation
        while True:
            text_column = Prompt.ask("[bold cyan]Text column[/bold cyan] (training input)", default=text_column_default)
            if text_column in all_columns:
                break
            self.console.print(f"[red]✗ Column '{text_column}' not found in dataset![/red]")
            self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

        # Ask for annotation column with validation
        while True:
            annotation_column = Prompt.ask("[bold cyan]Annotation column[/bold cyan] (training labels)", default=annotation_column_default)
            if annotation_column in all_columns:
                break
            self.console.print(f"[red]✗ Column '{annotation_column}' not found in dataset![/red]")
            self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

        # Show confirmation of selection
        self.console.print(f"\n[green]✓ Selected columns:[/green]")
        self.console.print(f"  [cyan]Text:[/cyan] '{text_column}' → Model will learn from this text")
        self.console.print(f"  [cyan]Annotations:[/cyan] '{annotation_column}' → Model will learn these labels")

        # Step 3b: CRITICAL - Text Length Analysis (MUST be done AFTER text column selection)
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        text_length_stats = self.analyze_text_lengths(
            data_path=csv_path,
            text_column=text_column,  # Use the ACTUAL selected column, not temp
            display_results=True,
            step_label=f"{resolve_step_label('text_length', 'STEP 5')}: Text Length Analysis"
        )
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        # Store stats for later use in model selection (no user choice yet)
        # User will choose strategy in model selection step

        # Step 5: Language Detection and Text Analysis (using sophisticated universal system)
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        language_step = resolve_step_label("language_detection", "STEP 3")
        self.console.print(f"[bold cyan]  {language_step}:[/bold cyan] [bold white]Language Detection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Analyzing languages to recommend the best model.[/dim]\n")

        # Read CSV for analysis
        import pandas as pd
        import json
        df = pd.read_csv(csv_path)

        # Use the SAME sophisticated language detection as category-csv
        languages_found_in_column = set(analysis.get('languages_detected', {}).keys())
        confirmed_languages = set()
        lang_column = None
        language_distribution = {}  # Store exact language counts
        apply_auto_detection = True

        # Check if we have a language column with detected languages
        has_lang_column = bool(analysis.get('language_column_candidates'))

        if has_lang_column and languages_found_in_column:
            # Option 1: Language column exists - offer to use it or detect automatically
            self.console.print("[bold]Languages Found in Column:[/bold]")
            for lang, count in analysis['languages_detected'].items():
                self.console.print(f"  • {lang.upper()}: {count:,} rows")

            lang_column_candidate = analysis['language_column_candidates'][0]
            self.console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")

            use_lang_column = Confirm.ask(
                f"\n[bold]Use language column '{lang_column_candidate}'?[/bold]",
                default=True
            )

            if use_lang_column:
                confirmed_languages = languages_found_in_column
                lang_column = lang_column_candidate
                self.console.print(f"[green]Using language column: {lang_column}[/green]")
        else:
            # Option 2: No language column
            if not has_lang_column:
                self.console.print("[yellow]No language column detected[/yellow]")

        # Automatic language detection from text content
        if apply_auto_detection:
            self.console.print("\n[dim]Analyzing ALL texts to detect languages (this may take a moment)...[/dim]")

            try:
                from llm_tool.utils.language_detector import LanguageDetector

                if text_column in df.columns:
                    # Analyze ALL texts (not just sample) for precise distribution
                    all_texts = df[text_column].dropna().tolist()

                    if all_texts:
                        detector = LanguageDetector()
                        lang_counts = {}
                        detected_languages_per_text = []  # Store language for each text

                        # Progress indicator
                        from tqdm import tqdm
                        self.console.print(f"[dim]Analyzing {len(all_texts)} texts...[/dim]")

                        for text in tqdm(all_texts, desc="Detecting languages", disable=not HAS_RICH):
                            if text and len(str(text).strip()) > 10:
                                try:
                                    detected = detector.detect(str(text))
                                    if detected:
                                        # Handle both dict and string returns
                                        if isinstance(detected, dict):
                                            lang = detected.get('language')
                                            confidence = detected.get('confidence', 0)
                                            # Use confidence threshold (optional)
                                            if lang and confidence >= 0.7:  # 70% confidence threshold
                                                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                                                detected_languages_per_text.append(lang)
                                            else:
                                                detected_languages_per_text.append(None)  # Low confidence
                                        elif isinstance(detected, str):
                                            lang_counts[detected] = lang_counts.get(detected, 0) + 1
                                            detected_languages_per_text.append(detected)
                                    else:
                                        detected_languages_per_text.append(None)
                                except Exception as e:
                                    self.logger.debug(f"Language detection failed for text: {e}")
                                    detected_languages_per_text.append(None)
                            else:
                                detected_languages_per_text.append(None)  # Empty or too short text

                        if lang_counts:
                            # Store exact distribution
                            language_distribution = lang_counts
                            total = sum(lang_counts.values())

                            self.console.print(f"\n[bold]Languages Detected from Content ({total:,} texts analyzed):[/bold]")

                            # Create detailed table
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

                            self.console.print(lang_table)

                            # Detect low-percentage languages (likely detection errors)
                            LOW_PERCENTAGE_THRESHOLD = 1.0  # Languages with < 1% are considered low
                            majority_languages = {}  # Languages above threshold
                            minority_languages = {}  # Languages below threshold (likely errors)

                            for lang, count in lang_counts.items():
                                percentage = (count / total * 100) if total > 0 else 0
                                if percentage >= LOW_PERCENTAGE_THRESHOLD:
                                    majority_languages[lang] = count
                                else:
                                    minority_languages[lang] = count

                            confirmed_languages = set(lang_counts.keys())

                            # Handle low-percentage languages if detected
                            if minority_languages:
                                self.console.print(f"\n[yellow][!] Warning: {len(minority_languages)} language(s) detected with very low percentage (< {LOW_PERCENTAGE_THRESHOLD}%):[/yellow]")
                                for lang, count in sorted(minority_languages.items(), key=lambda x: x[1], reverse=True):
                                    percentage = (count / total * 100)
                                    self.console.print(f"  • {lang.upper()}: {count} texts ({percentage:.2f}%)")

                                self.console.print("\n[dim]These are likely detection errors. You have options:[/dim]")
                                self.console.print("  [cyan]1. exclude[/cyan] - Exclude ALL low-percentage languages from training")
                                self.console.print("  [cyan]2. keep[/cyan] - Keep ALL detected languages (not recommended)")
                                self.console.print("  [cyan]3. select[/cyan] - Manually select which languages to keep")
                                self.console.print("  [cyan]4. correct[/cyan] - Force ALL minority languages to a single language (quick fix)")

                                minority_action = Prompt.ask(
                                    "\n[bold yellow]How to handle low-percentage languages?[/bold yellow]",
                                    choices=["exclude", "keep", "select", "correct"],
                                    default="correct"
                                )

                                if minority_action == "correct":
                                    # Quick correction: force all minority languages to one language
                                    self.console.print("\n[bold cyan]Quick Language Correction[/bold cyan]\n")

                                    # Show available languages
                                    all_supported_langs = [
                                        'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja',
                                        'ar', 'pl', 'tr', 'ko', 'hi', 'sv', 'no', 'da', 'fi', 'cs',
                                        'el', 'he', 'ro', 'uk', 'bg', 'hr', 'vi', 'th', 'id', 'fa'
                                    ]

                                    # Suggest the majority language
                                    majority_lang = max(majority_languages.items(), key=lambda x: x[1])[0] if majority_languages else 'en'

                                    self.console.print(f"[bold]Available languages:[/bold]")
                                    self.console.print(f"  • Majority language detected: [green]{majority_lang.upper()}[/green] ({majority_languages.get(majority_lang, 0)} texts)")
                                    self.console.print(f"  • All supported: {', '.join([l.upper() for l in all_supported_langs])}")

                                    correction_target = Prompt.ask(
                                        f"\n[bold yellow]Force ALL minority languages to which language?[/bold yellow]",
                                        default=majority_lang
                                    ).lower().strip()

                                    if correction_target not in all_supported_langs:
                                        self.console.print(f"[yellow]Warning: '{correction_target}' not in standard list, but will be used anyway[/yellow]")

                                    # Update language_distribution and confirmed_languages
                                    total_corrected = sum(minority_languages.values())

                                    # Move all minority counts to the target language
                                    for minority_lang in minority_languages.keys():
                                        if minority_lang in language_distribution:
                                            del language_distribution[minority_lang]

                                    # Add corrected texts to target language
                                    if correction_target in language_distribution:
                                        language_distribution[correction_target] += total_corrected
                                    else:
                                        language_distribution[correction_target] = total_corrected

                                    # Update confirmed languages
                                    confirmed_languages = set([correction_target] + list(majority_languages.keys()))

                                    # CRITICAL FIX: Update detected_languages_per_text with corrections
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] in minority_languages:
                                                detected_languages_per_text[i] = correction_target

                                    self.console.print(f"\n[green]✓ Corrected {total_corrected} texts from {len(minority_languages)} languages to {correction_target.upper()}[/green]")

                                    # Display updated distribution
                                    update_table = Table(title="Updated Language Distribution", border_style="green", expand=True)
                                    update_table.add_column("Language", style="cyan", justify="center")
                                    update_table.add_column("Count", justify="right")
                                    update_table.add_column("Percentage", justify="right")

                                    new_total = sum(language_distribution.values())
                                    for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
                                        if count > 0:  # Only show non-zero counts
                                            percentage = (count / new_total) * 100 if new_total > 0 else 0
                                            update_table.add_row(lang.upper(), f"{count:,}", f"{percentage:.1f}%")

                                    self.console.print(update_table)

                                elif minority_action == "exclude":
                                    # Exclude low-percentage languages
                                    for lang in minority_languages.keys():
                                        language_distribution[lang] = 0  # Mark as excluded

                                    # CRITICAL FIX: Mark excluded language texts as None
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] in minority_languages:
                                                detected_languages_per_text[i] = None

                                    confirmed_languages = set(majority_languages.keys())
                                    excluded_count = sum(minority_languages.values())
                                    self.console.print(f"\n[yellow]✗ Excluded {excluded_count} texts from {len(minority_languages)} low-percentage language(s)[/yellow]")
                                    self.console.print(f"[green]Final languages: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                                elif minority_action == "keep":
                                    self.console.print("[yellow][!] Keeping all detected languages (including low-percentage ones)[/yellow]")

                                elif minority_action == "select":
                                    # Manual selection of languages to keep
                                    self.console.print("\n[bold cyan]Language Selection:[/bold cyan]")
                                    self.console.print(f"[dim]Select which languages to keep for training (from all {len(lang_counts)} detected)[/dim]\n")

                                    # Show all languages sorted by count
                                    self.console.print("[bold]All Detected Languages:[/bold]")
                                    for i, (lang, count) in enumerate(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True), 1):
                                        percentage = (count / total * 100)
                                        status = "[green]majority[/green]" if lang in majority_languages else "[yellow][!] minority[/yellow]"
                                        self.console.print(f"  {i:2d}. {lang.upper():5s} - {count:6,} texts ({percentage:5.2f}%) {status}")

                                    self.console.print("\n[bold yellow]Select languages to KEEP:[/bold yellow]")
                                    self.console.print("[dim]Enter language codes separated by commas (e.g., 'fr,en,de')[/dim]")
                                    self.console.print("[dim]Press Enter without typing to keep ALL languages[/dim]")

                                    selected_langs = Prompt.ask("\n[bold]Languages to keep[/bold]", default="")

                                    if selected_langs.strip():
                                        # User selected specific languages
                                        selected_set = set([l.strip().lower() for l in selected_langs.split(',') if l.strip()])

                                        # Validate that selected languages exist
                                        invalid_langs = selected_set - set(lang_counts.keys())
                                        if invalid_langs:
                                            self.console.print(f"[yellow][!] Warning: These languages were not detected: {', '.join(invalid_langs)}[/yellow]")
                                            selected_set = selected_set - invalid_langs

                                        # Exclude non-selected languages
                                        for lang in lang_counts.keys():
                                            if lang not in selected_set:
                                                language_distribution[lang] = 0  # Mark as excluded

                                        # CRITICAL FIX: Mark non-selected language texts as None
                                        if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                            for i in range(len(detected_languages_per_text)):
                                                if detected_languages_per_text[i] and detected_languages_per_text[i] not in selected_set:
                                                    detected_languages_per_text[i] = None

                                        confirmed_languages = selected_set
                                        kept_count = sum([lang_counts[lang] for lang in selected_set])
                                        excluded_count = total - kept_count

                                        self.console.print(f"\n[green]✓ Kept {len(selected_set)} language(s): {', '.join([l.upper() for l in sorted(selected_set)])}[/green]")
                                        self.console.print(f"[dim]  → {kept_count:,} texts kept, {excluded_count:,} texts excluded[/dim]")
                                    else:
                                        # User pressed Enter - keep all
                                        self.console.print("[green]Keeping all detected languages[/green]")

                            # Final confirmation (allow override even after selection)
                            lang_list = ', '.join([l.upper() for l in sorted(confirmed_languages)])
                            lang_confirmed = Confirm.ask(
                                f"\n[bold]Final languages: {lang_list}. Is this correct?[/bold]",
                                default=True
                            )

                            if not lang_confirmed:
                                self.console.print("\n[yellow]Override with manual selection[/yellow]")

                                # Check for existing language columns
                                potential_lang_cols = [col for col in df.columns if col.lower() in ('lang', 'language', 'langue', 'idioma', 'sprache')]
                                use_col = False

                                if potential_lang_cols:
                                    self.console.print(f"[cyan]Found potential language columns: {', '.join(potential_lang_cols)}[/cyan]")
                                    use_col = Confirm.ask("Use an existing column for languages?", default=True)

                                    if use_col:
                                        if len(potential_lang_cols) == 1:
                                            lang_column = potential_lang_cols[0]
                                        else:
                                            lang_column = Prompt.ask("Which column?", choices=potential_lang_cols, default=potential_lang_cols[0])

                                        # Extract languages from the column
                                        lang_series = df[lang_column].dropna().astype(str).str.strip().str.upper()
                                        unique_langs = set(lang_series.unique())
                                        confirmed_languages = set([l.lower() for l in unique_langs if l])

                                        # Update detected_languages_per_text from the column
                                        text_mask = df[text_column].notna()
                                        detected_languages_per_text = []
                                        for val in df.loc[text_mask, lang_column]:
                                            if pd.notna(val) and str(val).strip():
                                                detected_languages_per_text.append(str(val).strip().lower())
                                            else:
                                                detected_languages_per_text.append(None)

                                        # Update language distribution
                                        language_distribution = {}
                                        for lang in confirmed_languages:
                                            count = lang_series.str.lower().eq(lang).sum()
                                            language_distribution[lang] = count

                                        self.console.print(f"[green]Using column '{lang_column}': {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                                if not use_col:
                                    manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

                                    # Update distribution to exclude non-selected languages
                                    for lang in lang_counts.keys():
                                        if lang not in confirmed_languages:
                                            language_distribution[lang] = 0

                                    # CRITICAL FIX: Mark non-confirmed language texts as None
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] and detected_languages_per_text[i] not in confirmed_languages:
                                                detected_languages_per_text[i] = None

                                    self.console.print(f"[green]Manual override: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                            else:
                                self.console.print("[green]Languages confirmed from content analysis[/green]")

                            # CRITICAL FIX: Add detected language column to DataFrame and save
                            if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                # Create a temporary DataFrame for non-null texts
                                temp_df = df[df[text_column].notna()].copy()

                                # Ensure same length
                                if len(detected_languages_per_text) == len(temp_df):
                                    if lang_column is None:
                                        temp_df['language'] = detected_languages_per_text

                                        # Map detected languages to the full DataFrame
                                        df['language'] = None
                                        df.loc[df[text_column].notna(), 'language'] = detected_languages_per_text

                                        # Set lang_column to use this new column
                                        lang_column = 'language'

                                        # Save updated DataFrame back to CSV
                                        df.to_csv(csv_path, index=False)
                                        self.console.print(f"[dim]Added 'language' column to dataset ({len([l for l in detected_languages_per_text if l])} texts with detected language)[/dim]")
                                    else:
                                        self.console.print("[dim]Auto-detected languages available; existing language column preserved.[/dim]")
                        else:
                            # Fallback: ask user
                            self.console.print("[yellow]Could not detect languages automatically[/yellow]")
                            manual_langs = Prompt.ask("Expected language codes (e.g., en,fr,de)", default="")
                            if manual_langs.strip():
                                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
                    else:
                        self.console.print("[yellow]Not enough text samples for language detection[/yellow]")
                        manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                        if manual_langs.strip():
                            confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

            except Exception as e:
                self.logger.debug(f"Language detection from content failed: {e}")
                self.console.print("[yellow]Automatic detection failed. Please specify manually[/yellow]")
                manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                if manual_langs.strip():
                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

        # Model selection will be done later when training mode is selected
        # Store languages for later use

        # Step 6: Annotation Data Preview
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        annotation_step = resolve_step_label("annotation_preview", "STEP 8")
        self.console.print(f"[bold cyan]  {annotation_step}:[/bold cyan] [bold white]Annotation Data Preview[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Analyzing all annotation data to show you what labels/categories will be trained...[/dim]\n")

        # df already loaded above for language detection

        all_keys_values = {}  # {key: set_of_unique_values}
        value_counts_by_key = {}  # {key: {value: occurrence_count}}
        total_samples = 0
        malformed_count = 0

        for idx, row in df.iterrows():
            annotation_val = row.get(annotation_column)
            if pd.isna(annotation_val) or annotation_val == '':
                continue

            total_samples += 1
            try:
                if isinstance(annotation_val, str):
                    # Try standard JSON first
                    try:
                        annotation_dict = json.loads(annotation_val)
                    except json.JSONDecodeError:
                        # Try Python literal (handles single quotes with escapes)
                        import ast
                        annotation_dict = ast.literal_eval(annotation_val)
                elif isinstance(annotation_val, dict):
                    annotation_dict = annotation_val
                else:
                    continue

                # Extract keys and values
                for key, value in annotation_dict.items():
                    if key not in all_keys_values:
                        all_keys_values[key] = set()
                    if key not in value_counts_by_key:
                        value_counts_by_key[key] = {}
                    counts_for_key = value_counts_by_key[key]

                    if isinstance(value, dict):
                        # Nested annotation format (e.g., {"detected": "yes", "subcategories": [...]})
                        if 'detected' in value:
                            detected = str(value['detected']).lower().strip()
                            all_keys_values[key].add(detected)
                            counts_for_key[detected] = counts_for_key.get(detected, 0) + 1
                        else:
                            for sub_key, sub_val in value.items():
                                sub_str = str(sub_val)
                                all_keys_values[key].add(sub_str)
                                counts_for_key[sub_str] = counts_for_key.get(sub_str, 0) + 1
                    elif isinstance(value, list):
                        for v in value:
                            if v is not None and v != '':
                                v_str = str(v)
                                all_keys_values[key].add(v_str)
                                counts_for_key[v_str] = counts_for_key.get(v_str, 0) + 1
                    elif value is not None and value != '':
                        v_str = str(value)
                        all_keys_values[key].add(v_str)
                        counts_for_key[v_str] = counts_for_key.get(v_str, 0) + 1

            except (json.JSONDecodeError, AttributeError, TypeError, ValueError, SyntaxError) as e:
                malformed_count += 1
                continue

        # Display comprehensive preview with Rich table
        if all_keys_values:
            self.console.print(f"\n[bold cyan]Complete Annotation Data Preview[/bold cyan]")
            self.console.print(f"[dim]Analyzed {total_samples} samples ({malformed_count} malformed)[/dim]\n")

            preview_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            preview_table.add_column("Key", style="yellow bold", no_wrap=True)
            preview_table.add_column("Unique Values", style="white", no_wrap=True, justify="center")
            preview_table.add_column("Sample Values", style="green", overflow="fold", ratio=1)

            for key in sorted(all_keys_values.keys()):
                values_set = all_keys_values[key]
                num_values = len(values_set)

                # Show first 10 values as sample
                sample_values = sorted(values_set)[:10]
                sample_str = ', '.join([f"'{v}'" for v in sample_values])
                if num_values > 10:
                    sample_str += f" ... (+{num_values - 10} more)"

                preview_table.add_row(
                    key,
                    str(num_values),
                    sample_str
                )

            self.console.print(preview_table)
            self.console.print()

            # Show selection options
            self.console.print("[bold]Training Options:[/bold]")
            self.console.print("  [dim]• You can choose to train on [cyan]ALL[/cyan] keys/values[/dim]")
            self.console.print("  [dim]• Or select [cyan]specific keys[/cyan] to train (asked later)[/dim]")
            self.console.print("  [dim]• Or select [cyan]specific values[/cyan] for each key (asked later)[/dim]\n")
        else:
            self.console.print("[yellow][!] No valid annotation data found[/yellow]\n")

        # Step 6.5: Value Filtering (Optional) - CRITICAL FOR DATA QUALITY
        if all_keys_values:
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            value_filter_step = resolve_step_label("value_filter", "STEP 9")
            self.console.print(f"[bold cyan]  {value_filter_step}:[/bold cyan] [bold white]Value Filtering (Optional)[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[dim]You can exclude specific values from your training data.[/dim]")
            self.console.print("[dim]   For example: Remove 'null' values, or exclude rare categories.[/dim]\n")

            filter_values = Confirm.ask(
                "[bold yellow]Do you want to exclude any specific values from training?[/bold yellow]",
                default=False
            )

            excluded_values = {}  # {key: [list_of_excluded_values]}
            rows_to_remove = []  # List of indices to remove from df

            if filter_values:
                self.console.print("\n[bold]Value Filtering Configuration[/bold]\n")

                # Ask for each key
                for key in sorted(all_keys_values.keys()):
                    values_set = all_keys_values[key]
                    num_values = len(values_set)

                    if num_values == 0:
                        continue

                    # Display key and its values
                    self.console.print(f"\n[cyan]Key:[/cyan] [bold]{key}[/bold] ({num_values} values)")

                    # Create table for values with counts
                    values_table = Table(show_header=True, header_style="bold magenta", border_style="dim", box=box.SIMPLE, expand=True)
                    values_table.add_column("Value", style="yellow", ratio=1, overflow="fold")
                    values_table.add_column("Count", style="white", no_wrap=True, justify="right")
                    values_table.add_column("Percentage", style="green", no_wrap=True, justify="right")

                    # Count occurrences of each value in the dataset
                    value_counts = {}
                    for idx, row in df.iterrows():
                        annotation_val = row.get(annotation_column)
                        if pd.isna(annotation_val) or annotation_val == '':
                            continue

                        try:
                            if isinstance(annotation_val, str):
                                try:
                                    annotation_dict = json.loads(annotation_val)
                                except json.JSONDecodeError:
                                    import ast
                                    annotation_dict = ast.literal_eval(annotation_val)
                            elif isinstance(annotation_val, dict):
                                annotation_dict = annotation_val
                            else:
                                continue

                            if key in annotation_dict:
                                val = annotation_dict[key]
                                if isinstance(val, list):
                                    for v in val:
                                        if v is not None and v != '':
                                            v_str = str(v)
                                            value_counts[v_str] = value_counts.get(v_str, 0) + 1
                                elif val is not None and val != '':
                                    v_str = str(val)
                                    value_counts[v_str] = value_counts.get(v_str, 0) + 1
                        except:
                            continue

                    # Display values with counts
                    sorted_values = sorted(values_set, key=lambda v: value_counts.get(v, 0), reverse=True)
                    for val in sorted_values:
                        count = value_counts.get(val, 0)
                        percentage = (count / total_samples * 100) if total_samples > 0 else 0
                        values_table.add_row(
                            val,
                            str(count),
                            f"{percentage:.1f}%"
                        )

                    self.console.print(values_table)

                    # Capture counts for later metadata usage
                    value_counts_by_key[key] = {
                        val: int(value_counts.get(val, 0))
                        for val in values_set
                    }

                    # Ask if user wants to exclude any values for this key
                    exclude_for_key = Confirm.ask(
                        f"[bold yellow]Exclude any values from '{key}'?[/bold yellow]",
                        default=False
                    )

                    if exclude_for_key:
                        self.console.print(f"[dim]Enter values to exclude (comma-separated), or type 'cancel' to skip[/dim]")
                        exclude_input = Prompt.ask(
                            f"[yellow]Values to exclude from '{key}'[/yellow]",
                            default=""
                        )

                        if exclude_input.lower() != 'cancel' and exclude_input.strip():
                            excluded_list = [v.strip() for v in exclude_input.split(',') if v.strip()]
                            # Validate that excluded values exist
                            valid_excluded = [v for v in excluded_list if v in values_set]
                            invalid_excluded = [v for v in excluded_list if v not in values_set]

                            if invalid_excluded:
                                self.console.print(f"[yellow][!] Warning: These values don't exist: {', '.join(invalid_excluded)}[/yellow]")

                            if valid_excluded:
                                excluded_values[key] = valid_excluded
                                self.console.print(f"[green]Will exclude: {', '.join(valid_excluded)}[/green]")

                # Now filter the DataFrame based on excluded values
                if excluded_values:
                    self.console.print(f"\n[bold cyan]Filtering labels from dataset...[/bold cyan]")
                    self.console.print(f"[dim]Note: Removing excluded labels from samples, not the samples themselves.[/dim]\n")

                    original_count = len(df)
                    labels_removed_count = 0
                    samples_modified = 0

                    # Filter labels from each row (NOT remove rows)
                    for idx, row in df.iterrows():
                        annotation_val = row.get(annotation_column)
                        if pd.isna(annotation_val) or annotation_val == '':
                            continue

                        try:
                            # Parse annotation
                            if isinstance(annotation_val, str):
                                try:
                                    annotation_dict = json.loads(annotation_val)
                                except json.JSONDecodeError:
                                    import ast
                                    annotation_dict = ast.literal_eval(annotation_val)
                            elif isinstance(annotation_val, dict):
                                annotation_dict = annotation_dict.copy()
                            else:
                                continue

                            # Remove excluded values from annotation (NOT the row)
                            modified = False
                            for key, excluded_vals in excluded_values.items():
                                if key in annotation_dict:
                                    val = annotation_dict[key]

                                    if isinstance(val, list):
                                        # Remove excluded values from list
                                        original_list = val.copy()
                                        val = [v for v in val if str(v) not in excluded_vals]
                                        if len(val) != len(original_list):
                                            modified = True
                                            labels_removed_count += len(original_list) - len(val)
                                        annotation_dict[key] = val if val else None

                                    elif val is not None and str(val) in excluded_vals:
                                        # Replace excluded value with None
                                        annotation_dict[key] = None
                                        modified = True
                                        labels_removed_count += 1

                            # Update the annotation in the DataFrame
                            if modified:
                                samples_modified += 1
                                # Convert back to JSON string if it was originally a string
                                if isinstance(row[annotation_column], str):
                                    df.at[idx, annotation_column] = json.dumps(annotation_dict)
                                else:
                                    df.at[idx, annotation_column] = annotation_dict

                        except Exception as e:
                            self.logger.warning(f"Error filtering row {idx}: {e}")
                            continue

                    # IMPORTANT: Do NOT remove samples even if they have no valid labels remaining
                    # Reason: Label filtering happens BEFORE key selection for training.
                    # A sample with all null/None labels might still be useful when training
                    # on specific keys later (e.g., user might select keys where null is valid).
                    # The training code will naturally skip samples without valid labels for selected keys.
                    removed_count = 0
                    filtered_count = len(df)

                    self.console.print(f"[green]Label filtering complete:[/green]")
                    self.console.print(f"  • [cyan]Samples kept:[/cyan] {original_count} → {filtered_count}")
                    self.console.print(f"  • [cyan]Samples modified:[/cyan] {samples_modified}")
                    self.console.print(f"  • [cyan]Labels removed:[/cyan] {labels_removed_count}")
                    if removed_count > 0:
                        self.console.print(f"  • [yellow]Samples removed (empty):[/yellow] {removed_count}")
                    self.console.print()

                    # Recalculate all_keys_values with filtered data
                    all_keys_values = {}
                    total_samples = 0
                    malformed_count = 0

                    for idx, row in df.iterrows():
                        annotation_val = row.get(annotation_column)
                        if pd.isna(annotation_val) or annotation_val == '':
                            continue

                        total_samples += 1
                        try:
                            if isinstance(annotation_val, str):
                                try:
                                    annotation_dict = json.loads(annotation_val)
                                except json.JSONDecodeError:
                                    import ast
                                    annotation_dict = ast.literal_eval(annotation_val)
                            elif isinstance(annotation_val, dict):
                                annotation_dict = annotation_val
                            else:
                                continue

                            # Extract keys and values (excluding the filtered ones)
                            for key, value in annotation_dict.items():
                                if key not in all_keys_values:
                                    all_keys_values[key] = set()

                                if isinstance(value, list):
                                    for v in value:
                                        if v is not None and v != '':
                                            all_keys_values[key].add(str(v))
                                elif value is not None and value != '':
                                    all_keys_values[key].add(str(value))

                        except (json.JSONDecodeError, AttributeError, TypeError, ValueError, SyntaxError) as e:
                            malformed_count += 1
                            continue

                    # Display updated summary
                    self.console.print("[bold]Updated Data Summary:[/bold]")
                    summary_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
                    summary_table.add_column("Key", style="yellow bold", no_wrap=True)
                    summary_table.add_column("Values (After Filtering)", style="white", ratio=1, overflow="fold")

                    for key in sorted(all_keys_values.keys()):
                        values_set = all_keys_values[key]
                        num_values = len(values_set)
                        sample_str = ', '.join([f"'{v}'" for v in sorted(values_set)[:5]])
                        if num_values > 5:
                            sample_str += f" ... (+{num_values - 5} more)"

                        # Show what was excluded
                        if key in excluded_values:
                            excluded_str = f"[dim red](excluded: {', '.join(excluded_values[key])})[/dim red]"
                            summary_table.add_row(
                                f"{key}\n{excluded_str}",
                                f"[green]{num_values} values[/green]: {sample_str}"
                            )
                        else:
                            summary_table.add_row(
                                key,
                                f"{num_values} values: {sample_str}"
                            )

                    self.console.print(summary_table)
                    self.console.print()
            else:
                self.console.print("[dim]No values excluded - using all data[/dim]\n")

        # Step 7: Training Strategy Selection (SIMPLIFIED)
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        strategy_step = resolve_step_label("training_strategy", "STEP 11")
        self.console.print(f"[bold cyan]  {strategy_step}:[/bold cyan] [bold white]Training Strategy Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        # Extract annotation keys and values from data
        annotation_keys_found = analysis.get('annotation_keys_found', set())
        sample_annotation = analysis.get('sample_data', {}).get(annotation_column, [])
        real_example_data = None

        if sample_annotation and len(sample_annotation) > 0:
            first_sample = sample_annotation[0]
            try:
                if isinstance(first_sample, str):
                    real_example_data = json.loads(first_sample)
                elif isinstance(first_sample, dict):
                    real_example_data = first_sample
            except:
                pass

        # Show sample annotation for context
        if real_example_data:
            self.console.print("[bold]Example annotation from your data:[/bold]")
            example_str = json.dumps(real_example_data, ensure_ascii=False, indent=2)
            self.console.print(f"[dim]{example_str}[/dim]\n")

        # Initialize
        detected_keys = []
        annotation_keys = None
        mode = "single-label"  # Will be derived from choice
        training_approach = "multi-class"  # Default

        # Step 6a: Show all annotation keys and their values
        if all_keys_values:
            detected_keys = sorted(all_keys_values.keys())
            self.console.print(f"[bold]Annotation Keys Detected in Your Data:[/bold]\n")

            # Show all keys and their values
            for key in detected_keys:
                num_values = len(all_keys_values[key])
                values_preview = ', '.join([f"'{v}'" for v in sorted(all_keys_values[key])[:5]])
                if num_values > 5:
                    values_preview += f" ... (+{num_values-5} more)"
                self.console.print(f"  • [cyan]{key}[/cyan] ({num_values} values): {values_preview}")

            self.console.print("\n[dim]Options:[/dim]")
            self.console.print(f"  • [cyan]Leave blank[/cyan] → Use ALL {len(detected_keys)} keys with ALL their values")
            self.console.print(f"  • [cyan]Enter specific keys[/cyan] → Use only selected keys with ALL their values")
            if detected_keys:
                self.console.print(f"    Example: '{detected_keys[0]}' → Use only {detected_keys[0]} key\n")
        elif analysis.get('annotation_keys_found'):
            detected_keys = sorted(analysis['annotation_keys_found'])
            self.console.print(f"\n[green]✓ Detected keys: {', '.join(detected_keys)}[/green]")
            self.console.print("[dim]Leave blank to use all keys, or specify which ones to include[/dim]\n")

        # Step 6b: Ask which keys to include
        keys_input = Prompt.ask("[bold yellow]Annotation keys to include[/bold yellow] (comma separated, or BLANK for ALL)", default="")
        annotation_keys = [key.strip() for key in keys_input.split(",") if key.strip()] or None

        # Step 6c: Ask multi-class vs one-vs-all (ALWAYS, not just for single key)
        # Determine which keys will be trained
        keys_to_train = annotation_keys if annotation_keys else detected_keys

        # Validate and auto-correct invalid keys with intelligent suggestions
        invalid_keys = [key for key in keys_to_train if key not in all_keys_values]
        if invalid_keys:
            from difflib import get_close_matches

            self.console.print(f"\n[bold yellow][!]  Some keys need correction:[/bold yellow]")

            # Auto-correct using fuzzy matching
            corrected_keys = []
            for key in keys_to_train:
                if key in all_keys_values:
                    corrected_keys.append(key)
                else:
                    # Find best match using fuzzy matching
                    matches = get_close_matches(key, all_keys_values.keys(), n=1, cutoff=0.6)
                    if matches:
                        suggestion = matches[0]
                        self.console.print(f"  • [red]'{key}'[/red] → [green]'{suggestion}'[/green] [dim](auto-corrected)[/dim]")
                        corrected_keys.append(suggestion)
                    else:
                        self.console.print(f"  • [red]'{key}'[/red] [dim](no match found, will be skipped)[/dim]")

            # Show available keys for reference
            if len(corrected_keys) < len(keys_to_train):
                self.console.print(f"\n[bold cyan]Tip: Available keys:[/bold cyan]")
                for key in sorted(all_keys_values.keys()):
                    self.console.print(f"  • [green]{key}[/green]")

            # Ask user to confirm corrections
            if corrected_keys:
                self.console.print(f"\n[green]✓ Corrected selection:[/green] {', '.join(corrected_keys)}")
                confirm = Confirm.ask("[bold yellow]Use these corrected keys?[/bold yellow]", default=True)
                if confirm:
                    keys_to_train = corrected_keys
                    annotation_keys = corrected_keys
                else:
                    self.console.print("[yellow]Training cancelled. Please try again with correct key names.[/yellow]")
                    return None
            else:
                self.console.print("[red][FAIL] No valid keys found after correction. Training cancelled.[/red]")
                return None

        # Calculate total number of models for each approach
        total_values_count = 0
        for key in keys_to_train:
            if key in all_keys_values:
                total_values_count += len(all_keys_values[key])

        num_keys = len(keys_to_train)

        # ALWAYS ask the training approach question, even for binary classification
        # User may want one-vs-all even with 2 values
        if True:  # Always ask
            self.console.print(f"\n[bold cyan]Training Approach[/bold cyan]\n")

            if annotation_keys and len(annotation_keys) == 1:
                # Single key selected
                selected_key = annotation_keys[0]
                num_unique_values = len(all_keys_values[selected_key])
                values_list = sorted(all_keys_values[selected_key])
                values_str = ', '.join([f"'{v}'" for v in values_list[:5]])
                if num_unique_values > 5:
                    values_str += f" ... (+{num_unique_values-5} more)"

                self.console.print(f"[bold]Selected:[/bold] '{selected_key}' ({num_unique_values} values)")
                self.console.print(f"[dim]Values: {values_str}[/dim]\n")
            else:
                # Multiple keys or ALL
                self.console.print(f"[bold]Selected:[/bold] {'ALL' if not annotation_keys else len(annotation_keys)} keys ({num_keys} total)")
                self.console.print(f"[dim]Total unique values across all keys: {total_values_count}[/dim]\n")

            # Create comparison table
            approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            approach_table.add_column("Approach", style="cyan bold", no_wrap=True)
            approach_table.add_column("What It Does", style="white", overflow="fold", ratio=1)

            if annotation_keys and len(annotation_keys) == 1:
                # Single key - simple explanation
                selected_key = annotation_keys[0]
                num_unique_values = len(all_keys_values[selected_key])
                values_list = sorted(all_keys_values[selected_key])

                approach_table.add_row(
                    "multi-class",
                    f"Trains ONE model for '{selected_key}'\n\n"
                    f"• Chooses between all {num_unique_values} values\n"
                    f"• Example: '{values_list[0]}' vs '{values_list[1]}' vs ...\n"
                    f"• Predicts exactly ONE value per text\n"
                    f"• [bold green]Total: 1 model[/bold green]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Mutually exclusive categories"
                )
                approach_table.add_row(
                    "one-vs-all",
                    f"Trains {num_unique_values} binary models for '{selected_key}'\n\n"
                    f"• Model 1: '{values_list[0]}' vs NOT '{values_list[0]}'\n"
                    f"• Model 2: '{values_list[1]}' vs NOT '{values_list[1]}'\n"
                    f"• ... (one model per value)\n"
                    f"• [bold yellow]Total: {num_unique_values} models[/bold yellow]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Imbalanced data"
                )
                approach_table.add_row(
                    "multi-label",
                    f"Trains ONE model with MULTIPLE labels per text\n\n"
                    f"• Uses sigmoid activation (independent probabilities)\n"
                    f"• Each text can have 0, 1, or many labels simultaneously\n"
                    f"• Threshold-based selection (default: 0.5)\n"
                    f"• [bold magenta]Total: 1 model[/bold magenta]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Texts with overlapping categories"
                )
            else:
                # Multiple keys or ALL - offer hybrid and custom modes
                # Analyze keys to determine hybrid strategy
                keys_small = []  # ≤5 values
                keys_large = []  # >5 values
                for key in keys_to_train:
                    num_values = len(all_keys_values[key])
                    if num_values <= 5:
                        keys_small.append((key, num_values))
                    else:
                        keys_large.append((key, num_values))

                hybrid_multiclass_count = len(keys_small)
                hybrid_onevsall_count = sum(num_vals for _, num_vals in keys_large)
                total_hybrid_models = hybrid_multiclass_count + hybrid_onevsall_count

                approach_table.add_row(
                    "multi-class",
                    f"Trains ONE model PER KEY (not per value)\n\n"
                    f"• {num_keys} models total (one per annotation key)\n"
                    f"• Each model learns ALL values of ITS key\n"
                    f"• Example: One model for 'political_party' learns BQ, CAQ, CPC, etc.\n"
                    f"• Example: Another model for 'sentiment' learns positive, negative, neutral\n"
                    f"• [bold green]Total: {num_keys} models (one per key)[/bold green]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Standard classification with mutually exclusive categories per key"
                )
                approach_table.add_row(
                    "one-vs-all",
                    f"Trains ONE model PER VALUE (not per key)\n\n"
                    f"• {total_values_count} binary models total (one per unique value)\n"
                    f"• Each model: 'value X' vs NOT 'value X'\n"
                    f"• Example: Separate model for 'political_party_BQ' (binary: BQ or not)\n"
                    f"• Example: Separate model for 'sentiment_positive' (binary: positive or not)\n"
                    f"• [bold yellow]Total: {total_values_count} models (one per value)[/bold yellow]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Imbalanced data"
                )
                approach_table.add_row(
                    "multi-label",
                    f"Trains ONE model with MULTIPLE labels per text\n\n"
                    f"• {num_keys} models total (one per annotation key)\n"
                    f"• Uses sigmoid activation (independent probabilities)\n"
                    f"• Each text can have 0, 1, or many labels simultaneously\n"
                    f"• Threshold-based selection (default: 0.5)\n"
                    f"• [bold magenta]Total: {num_keys} models[/bold magenta]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Texts with overlapping categories"
                )
                approach_table.add_row(
                    "hybrid",
                    f"SMART: Choose multi-label OR adaptive multi-class/one-vs-all\n\n"
                    f"• You'll choose between:\n"
                    f"  -  [bold magenta]Multi-label[/bold magenta]: 1 model per key with sigmoid (overlapping labels)\n"
                    f"  - [bold green]Multi-class + Binary[/bold green]: Adaptive per key (≤5 vals → multi-class, >5 → one-vs-all)\n"
                    f"• For multi-class+binary on your data:\n"
                    f"  - {hybrid_multiclass_count} keys use multi-class ({', '.join([k for k, _ in keys_small[:3]])}{'...' if len(keys_small) > 3 else ''})\n"
                    f"  - {len(keys_large)} keys use one-vs-all ({', '.join([k for k, _ in keys_large[:3]])}{'...' if len(keys_large) > 3 else ''})\n"
                    f"• [bold magenta]Total: varies based on choice[/bold magenta]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Mixed dataset - choose the best approach per use case (RECOMMENDED)"
                )
                approach_table.add_row(
                    "custom",
                    f"CUSTOM: You choose the strategy for EACH key individually\n\n"
                    f"• You'll be asked for each of the {num_keys} keys\n"
                    f"• Choose multi-class, one-vs-all, OR multi-label per key\n"
                    f"• Example: multi-class for 'sentiment', multi-label for 'themes'\n"
                    f"• [bold blue]Total: Variable (depends on your choices)[/bold blue]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Advanced users who want fine-grained control"
                )

            self.console.print(approach_table)
            self.console.print()

            # Determine available choices and default based on context
            if annotation_keys and len(annotation_keys) == 1:
                # Single key: no hybrid or custom modes
                available_choices = ["multi-class", "one-vs-all", "multi-label", "back"]
                default_approach = "multi-class"
            else:
                # Multiple keys: all modes available
                available_choices = ["multi-class", "one-vs-all", "multi-label", "hybrid", "custom", "back"]
                default_approach = "hybrid"

            training_approach = Prompt.ask(
                "[bold yellow]Training approach[/bold yellow]",
                choices=available_choices,
                default=default_approach
            )

            if training_approach == "back":
                return None

            # Store per-key strategy decisions
            key_strategies = {}  # {key_name: 'multi-class' or 'one-vs-all'}

            if training_approach == "hybrid":
                # Ask user to choose between multi-label or multi-class+binary
                self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
                self.console.print("[bold cyan]  Hybrid Mode: Choose Your Strategy[/bold cyan]")
                self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n")

                # Display detailed statistics per key/value to help user make informed decisions
                self.console.print("[bold cyan]Detailed Statistics per Key/Value:[/bold cyan]")
                self.console.print("[dim]This shows the number of instances for each value in each key.[/dim]\n")

                stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1), expand=True)
                stats_table.add_column("Key", style="yellow bold", no_wrap=True)
                stats_table.add_column("Value", style="white", ratio=1, overflow="fold")
                stats_table.add_column("Instances", style="cyan", no_wrap=True, justify="right")
                stats_table.add_column("% of Key", style="green", no_wrap=True, justify="right")

                for key in keys_to_train:
                    key_counts = value_counts_by_key.get(key, {})
                    total_for_key = sum(key_counts.values()) if key_counts else 0
                    sorted_values = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)

                    # Show first value with key name
                    if sorted_values:
                        first_val, first_count = sorted_values[0]
                        pct = (first_count / total_for_key * 100) if total_for_key > 0 else 0
                        stats_table.add_row(
                            f"[bold]{key}[/bold] ({len(sorted_values)} vals)",
                            first_val[:22] + "..." if len(first_val) > 25 else first_val,
                            str(first_count),
                            f"{pct:.1f}%"
                        )
                        # Show remaining values (up to 5 more)
                        for val, count in sorted_values[1:6]:
                            pct = (count / total_for_key * 100) if total_for_key > 0 else 0
                            stats_table.add_row(
                                "",
                                val[:22] + "..." if len(val) > 25 else val,
                                str(count),
                                f"{pct:.1f}%"
                            )
                        if len(sorted_values) > 6:
                            remaining = len(sorted_values) - 6
                            remaining_count = sum(c for _, c in sorted_values[6:])
                            stats_table.add_row("", f"[dim]... +{remaining} more values[/dim]", f"[dim]{remaining_count}[/dim]", "")
                        # Add separator row between keys
                        stats_table.add_row("", "", "", "")

                self.console.print(stats_table)
                self.console.print()

                # Calculate totals for display with default threshold of 5
                default_threshold = 5
                hybrid_mc_count = sum(1 for key in keys_to_train if len(all_keys_values[key]) <= default_threshold)
                hybrid_ova_count = sum(len(all_keys_values[key]) for key in keys_to_train if len(all_keys_values[key]) > default_threshold)
                hybrid_total = hybrid_mc_count + hybrid_ova_count

                hybrid_sub_table = Table(show_header=True, header_style="bold", box=box.ROUNDED, padding=(0, 2), expand=True)
                hybrid_sub_table.add_column("Option", style="bold cyan", no_wrap=True)
                hybrid_sub_table.add_column("Description", style="white", ratio=1, overflow="fold")

                hybrid_sub_table.add_row(
                    "multi-label",
                    f"[bold magenta]One model per key with sigmoid activation[/bold magenta]\n\n"
                    f"• Each text can have MULTIPLE labels simultaneously\n"
                    f"• Independent probabilities per label (BCEWithLogitsLoss)\n"
                    f"• Threshold-based selection (default: 0.5)\n"
                    f"• [bold magenta]Total: {num_keys} models (one per key)[/bold magenta]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Texts with overlapping/co-occurring categories"
                )
                hybrid_sub_table.add_row(
                    "multi-class+binary",
                    f"[bold green]Adaptive strategy per key[/bold green]\n\n"
                    f"• Keys with ≤N values → Multi-class (mutually exclusive)\n"
                    f"• Keys with >N values → One-vs-all (binary per value)\n"
                    f"• You'll choose the threshold N (default: {default_threshold})\n"
                    f"• [bold green]Total: varies based on threshold[/bold green]\n\n"
                    "[bold cyan]Best for:[/bold cyan] Standard classification (one label per text per key)"
                )

                self.console.print(hybrid_sub_table)
                self.console.print()

                hybrid_sub_choice = Prompt.ask(
                    "[bold yellow]Choose hybrid strategy[/bold yellow]",
                    choices=["multi-label", "multi-class+binary", "back"],
                    default="multi-class+binary"
                )

                if hybrid_sub_choice == "back":
                    # Return to previous step
                    return None

                if hybrid_sub_choice == "multi-label":
                    # Override training_approach to multi-label
                    training_approach = "multi-label"
                    self.console.print("\n[magenta]✓ Using multi-label strategy (one model per key with sigmoid)[/magenta]\n")
                    # All keys use multi-label
                    for key in keys_to_train:
                        key_strategies[key] = 'multi-label'
                else:
                    # multi-class+binary: Ask for threshold
                    self.console.print("\n[bold cyan]Multi-class + Binary Strategy Configuration:[/bold cyan]")
                    self.console.print("[dim]Choose the threshold N that determines which keys use multi-class vs one-vs-all.[/dim]")
                    self.console.print("[dim]Keys with ≤N values → multi-class (1 model), Keys with >N values → one-vs-all (N models)[/dim]\n")

                    # Show preview of different threshold values
                    self.console.print("[bold]Preview of threshold effects:[/bold]")
                    for preview_threshold in [3, 5, 7, 10]:
                        mc_keys = [k for k in keys_to_train if len(all_keys_values[k]) <= preview_threshold]
                        ova_keys = [k for k in keys_to_train if len(all_keys_values[k]) > preview_threshold]
                        mc_models = len(mc_keys)
                        ova_models = sum(len(all_keys_values[k]) for k in ova_keys)
                        total = mc_models + ova_models
                        self.console.print(f"  • Threshold={preview_threshold}: {mc_models} multi-class + {ova_models} one-vs-all = [bold]{total} models[/bold]")
                    self.console.print()

                    # Ask user to choose threshold
                    while True:
                        threshold_input = Prompt.ask(
                            "[bold yellow]Threshold (N) for multi-class vs one-vs-all[/bold yellow]",
                            default=str(default_threshold)
                        )
                        try:
                            hybrid_threshold = int(threshold_input)
                            if hybrid_threshold < 1:
                                self.console.print("[red]Threshold must be at least 1[/red]")
                                continue
                            break
                        except ValueError:
                            self.console.print("[red]Please enter a valid integer[/red]")

                    self.console.print(f"\n[green]✓ Using threshold: {hybrid_threshold}[/green]")
                    self.console.print(f"[dim]Keys with ≤{hybrid_threshold} values → multi-class, Keys with >{hybrid_threshold} values → one-vs-all[/dim]\n")

                    # Calculate total models for hybrid approach with user-selected threshold
                    total_hybrid_models = 0
                    for key in keys_to_train:
                        num_values = len(all_keys_values[key])
                        if num_values <= hybrid_threshold:
                            key_strategies[key] = 'multi-class'
                            total_hybrid_models += 1
                            self.console.print(f"  • [green]{key}[/green] ({num_values} values) → [bold]multi-class[/bold] (1 model)")
                        else:
                            key_strategies[key] = 'one-vs-all'
                            total_hybrid_models += num_values
                            self.console.print(f"  • [yellow]{key}[/yellow] ({num_values} values) → [bold]one-vs-all[/bold] ({num_values} models)")

                    self.console.print(f"\n[dim]Total models: {total_hybrid_models}[/dim]\n")

            elif training_approach == "custom":
                # User chooses per key
                self.console.print("\n[bold cyan] Custom Strategy Selection:[/bold cyan]")
                self.console.print("[dim]Choose the training strategy for each key individually.[/dim]\n")

                total_custom_models = 0
                has_multilabel_keys = False  # Track if any key uses multi-label
                for key in keys_to_train:
                    num_values = len(all_keys_values[key])
                    values_preview = ', '.join([f"'{v}'" for v in sorted(all_keys_values[key])[:3]])
                    if num_values > 3:
                        values_preview += f" ... (+{num_values-3} more)"

                    self.console.print(f"[bold]{key}[/bold] ({num_values} values)")
                    self.console.print(f"[dim]  Values: {values_preview}[/dim]")
                    self.console.print(f"  • [green]multi-class[/green]: 1 model learns all {num_values} values (mutually exclusive)")
                    self.console.print(f"  • [yellow]one-vs-all[/yellow]: {num_values} binary models (one per value)")
                    self.console.print(f"  • [magenta]multi-label[/magenta]: 1 model with sigmoid (overlapping labels allowed)")

                    key_choice = Prompt.ask(
                        f"  Strategy for '{key}'",
                        choices=["multi-class", "one-vs-all", "multi-label", "m", "o", "l"],
                        default="multi-class" if num_values <= 5 else "one-vs-all"
                    )

                    # Normalize shortcuts
                    if key_choice == "m":
                        key_choice = "multi-class"
                    elif key_choice == "o":
                        key_choice = "one-vs-all"
                    elif key_choice == "l":
                        key_choice = "multi-label"

                    key_strategies[key] = key_choice

                    if key_choice == "multi-class":
                        total_custom_models += 1
                        self.console.print(f"  ✓ Will train [green]1 model[/green] for {key}\n")
                    elif key_choice == "one-vs-all":
                        total_custom_models += num_values
                        self.console.print(f"  ✓ Will train [yellow]{num_values} models[/yellow] for {key}\n")
                    else:  # multi-label
                        total_custom_models += 1
                        has_multilabel_keys = True
                        self.console.print(f"  ✓ Will train [magenta]1 multi-label model[/magenta] for {key}\n")

                self.console.print(f"[bold cyan]Total models to train: {total_custom_models}[/bold cyan]\n")

                # If any key uses multi-label, set the training_approach accordingly
                if has_multilabel_keys and all(key_strategies[k] == 'multi-label' for k in keys_to_train):
                    # All keys use multi-label
                    training_approach = "multi-label"
                elif has_multilabel_keys:
                    # Mixed: some multi-label, some not - keep as custom
                    pass

            elif training_approach == "multi-class":
                # All keys use multi-class
                for key in keys_to_train:
                    key_strategies[key] = 'multi-class'

            elif training_approach == "one-vs-all":
                # All keys use one-vs-all
                for key in keys_to_train:
                    key_strategies[key] = 'one-vs-all'

        # Step 6c: Data Split Configuration
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        data_split_step = resolve_step_label("data_split", "STEP 12")
        self.console.print(f"[bold cyan]  {data_split_step}:[/bold cyan] [bold white]Data Split Configuration[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        split_config = self._configure_data_splits(
            keys_to_train=keys_to_train,
            all_keys_values=all_keys_values,
            training_approach=training_approach,
            key_strategies=key_strategies,
            total_samples=len(df)
        )

        if split_config is None:
            return None

        # Display split configuration summary
        self._display_split_summary(
            split_config=split_config,
            keys_to_train=keys_to_train,
            all_keys_values=all_keys_values,
            key_strategies=key_strategies
        )

        # Note: split_config will be stored in bundle.metadata after bundle is created

        # Step 6d: Label naming strategy
        self.console.print("\n[bold] Label Naming Strategy:[/bold]")
        self.console.print("[dim]This determines how label names appear in your training files and model predictions.[/dim]\n")

        # Generate examples based on SELECTED keys (not random example data)
        # Build concrete transformation examples
        transformation_examples = []
        for key in keys_to_train[:2]:  # Show 2 examples for clarity
            if key in all_keys_values:
                values = sorted(all_keys_values[key])[:2]  # First 2 values
                if values:
                    for val in values:
                        transformation_examples.append({
                            'key': key,
                            'value': val,
                            'key_value': f"{key}_{val}",
                            'value_only': val
                        })

        # Create comparison table
        strategy_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
        strategy_table.add_column("Strategy", style="cyan bold", no_wrap=True)
        strategy_table.add_column("Format", style="white", no_wrap=True)
        strategy_table.add_column("When to Use", style="white", ratio=1, overflow="fold")

        # Build key_value example string
        if transformation_examples:
            kv_format_examples = [f"'{ex['key_value']}'" for ex in transformation_examples[:3]]
            kv_format = f"key_value\nExample: {', '.join(kv_format_examples)}"
        else:
            kv_format = "key_value\nExample: 'sentiment_positive'"

        # Build value_only example string
        if transformation_examples:
            vo_format_examples = [f"'{ex['value_only']}'" for ex in transformation_examples[:3]]
            vo_format = f"value_only\nExample: {', '.join(vo_format_examples)}"
        else:
            vo_format = "value_only\nExample: 'positive'"

        strategy_table.add_row(
            "key_value",
            "Includes key prefix\n[dim](key_value)[/dim]",
            "✓ Training [bold]multiple keys[/bold]\n"
            "✓ Values might overlap between keys\n"
            "✓ [green]Recommended for most cases[/green]"
        )

        strategy_table.add_row(
            "value_only",
            "Only the value\n[dim](no prefix)[/dim]",
            "✓ Training [bold]single key only[/bold]\n"
            "✓ Values are unique across dataset\n"
            "[!] [yellow]Can cause conflicts with multiple keys[/yellow]"
        )

        self.console.print(strategy_table)
        self.console.print()

        # Show concrete transformation if we have examples
        if transformation_examples:
            self.console.print("[bold]How Your Data Will Be Transformed:[/bold]\n")

            transform_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.SIMPLE, expand=True)
            transform_table.add_column("Original (key → value)", style="cyan", no_wrap=True)
            transform_table.add_column("key_value format", style="green", no_wrap=True)
            transform_table.add_column("value_only format", style="yellow", no_wrap=True)

            for ex in transformation_examples[:4]:  # Show max 4 examples
                transform_table.add_row(
                    f"{ex['key']} → {ex['value']}",
                    ex['key_value'],
                    ex['value_only']
                )

            self.console.print(transform_table)
            self.console.print()

        # Show warning if multiple keys and value_only
        if len(keys_to_train) > 1:
            self.console.print("[bold yellow]Recommendation:[/bold yellow]")
            self.console.print(f"[dim]You selected {len(keys_to_train)} keys. Use [bold cyan]key_value[/bold cyan] to avoid label conflicts.")
            self.console.print(f"[dim]Example: If both 'affiliation' and 'gender' have value 'no', they would conflict with [yellow]value_only[/yellow].[/dim]\n")
        else:
            self.console.print("[dim]Tip: With a single key, both strategies work fine. [cyan]key_value[/cyan] is still recommended for consistency.[/dim]\n")

        label_strategy = Prompt.ask("Label naming strategy", choices=["key_value", "value_only", "back"], default="key_value")
        if label_strategy == "back":
            return None

        # Derive mode based on approach
        if training_approach == "one-vs-all":
            mode = "multi-label"  # one-vs-all uses multi-label infrastructure
        elif training_approach == "multi-label":
            mode = "multi-label"  # true multi-label uses multi-label infrastructure
        else:
            mode = "single-label"  # multi-class uses single-label infrastructure

        # Step 8: Additional Columns (ID, Language)
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        additional_step = resolve_step_label("additional_columns", "STEP 13")
        self.console.print(f"[bold cyan]  {additional_step}:[/bold cyan] [bold white]Additional Columns (Optional)[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Optional: Select ID and language columns if available in your dataset.[/dim]\n")

        # Determine the dataset path for optional re-load / ID heuristics
        data_path_obj: Optional[Path] = None
        local_data_path = locals().get('data_path')
        if isinstance(local_data_path, (str, Path)):
            data_path_obj = Path(local_data_path)
        elif isinstance(analysis.get('data_path'), (str, Path)):
            data_path_obj = Path(analysis['data_path'])
        elif 'selection' in locals() and isinstance(selection, dict) and selection.get('data_path'):
            data_path_obj = Path(selection['data_path'])
        elif 'bundle' in locals():
            bundle_obj = locals()['bundle']
            primary_file = getattr(bundle_obj, "primary_file", None)
            if primary_file:
                data_path_obj = Path(primary_file)

        # Use modernized ID selection - load dataframe if needed
        try:
            if not isinstance(df, pd.DataFrame):
                # Need to load dataframe for ID detection
                if not data_path_obj:
                    raise ValueError("Dataset path unavailable for ID analysis")

                suffix = data_path_obj.suffix.lower()
                if suffix == '.csv':
                    df = pd.read_csv(data_path_obj, nrows=1000)
                elif suffix == '.json':
                    df = pd.read_json(data_path_obj, lines=False, nrows=1000)
                elif suffix == '.jsonl':
                    df = pd.read_json(data_path_obj, lines=True, nrows=1000)
                elif suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(data_path_obj, nrows=1000)
                elif suffix == '.parquet':
                    df = pd.read_parquet(data_path_obj).head(1000)
                else:
                    df = pd.read_csv(data_path_obj, nrows=1000)

            # Use new unified ID selection
            id_column = DataDetector.display_and_select_id_column(
                self.console,
                df,
                text_column=text_column,
                step_label="Identifier Column (Optional)",
                data_path=data_path_obj
            )
        except Exception as e:
            self.logger.debug("Skipping automatic ID detection: %s", e)
            self.console.print("[dim yellow]Unable to auto-suggest an ID column; continuing without one.[/dim]")
            id_column = None

        # Language column handling - check if already processed in Step 5
        # Skip if we already did language detection (either with column or auto-detection)
        language_already_processed = 'lang_column' in locals() and confirmed_languages

        if language_already_processed:
            # Language was already handled in Step 5
            if lang_column:
                self.console.print(f"\n[green]✓ Language column from Step 5: '{lang_column}'[/green]")
            else:
                self.console.print(f"\n[green]✓ Languages detected in Step 5: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                self.console.print(f"[dim]  (Using automatic language detection - no specific column)[/dim]")
        elif analysis['language_column_candidates']:
            # Language column detected but Step 5 was skipped - ask user
            lang_column_candidate = analysis['language_column_candidates'][0]
            self.console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
            while True:
                override_lang = Prompt.ask("\n[bold yellow]Language column (optional)[/bold yellow]", default=lang_column_candidate)
                if not override_lang or override_lang in all_columns:
                    lang_column = override_lang if override_lang else lang_column_candidate
                    break
                self.console.print(f"[red]✗ Column '{override_lang}' not found in dataset![/red]")
                self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

        # Handle training approach with key_strategies support
        if 'training_approach' in locals() and training_approach == "one-vs-all":
            # Convert to multi-label format for one-vs-all training
            request = TrainingDataRequest(
                input_path=csv_path,
                format="llm_json",
                text_column=text_column,
                annotation_column=annotation_column,
                annotation_keys=annotation_keys,
                label_strategy=label_strategy,
                mode="multi-label",  # Use multi-label to trigger one-vs-all training
                id_column=id_column or None,
                lang_column=lang_column or None,
                key_strategies={k: 'one-vs-all' for k in (annotation_keys or [])} if 'key_strategies' not in locals() else None
            )
            bundle = builder.build(request)

            # Mark this as one-vs-all for distributed training
            if bundle:
                bundle.metadata['training_approach'] = 'one-vs-all'
                bundle.metadata['original_strategy'] = 'single-label'
                # Filter metadata to only include selected annotation keys
                selected_keys = set(annotation_keys) if annotation_keys else set(all_keys_values.keys())
                bundle.metadata['all_keys_values'] = {
                    key: sorted(list(values))
                    for key, values in all_keys_values.items()
                    if key in selected_keys
                }
                bundle.metadata['value_counts_by_key'] = {
                    key: {val: int(count) for val, count in counts.items()}
                    for key, counts in value_counts_by_key.items()
                    if key in selected_keys
                }
        elif 'training_approach' in locals() and training_approach == "multi-label":
            # True multi-label: one model with sigmoid activation for multiple labels per text
            request = TrainingDataRequest(
                input_path=csv_path,
                format="llm_json",
                text_column=text_column,
                annotation_column=annotation_column,
                annotation_keys=annotation_keys,
                label_strategy=label_strategy,
                mode="multi-label",  # Use multi-label data format
                id_column=id_column or None,
                lang_column=lang_column or None,
                key_strategies=key_strategies if 'key_strategies' in locals() else None
            )
            bundle = builder.build(request)

            # Mark this as true multi-label (not one-vs-all)
            if bundle:
                bundle.metadata['training_approach'] = 'multi-label'
                bundle.metadata['multi_label'] = True
                bundle.metadata['multi_label_threshold'] = 0.5
                # Filter metadata to only include selected annotation keys
                selected_keys = set(annotation_keys) if annotation_keys else set(all_keys_values.keys())
                bundle.metadata['all_keys_values'] = {
                    key: sorted(list(values))
                    for key, values in all_keys_values.items()
                    if key in selected_keys
                }
                bundle.metadata['value_counts_by_key'] = {
                    key: {val: int(count) for val, count in counts.items()}
                    for key, counts in value_counts_by_key.items()
                    if key in selected_keys
                }
        else:
            # Standard mode (can be multi-class, hybrid, or custom)
            # Pass key_strategies if available (from hybrid/custom mode)
            request = TrainingDataRequest(
                input_path=csv_path,
                format="llm_json",
                text_column=text_column,
                annotation_column=annotation_column,
                annotation_keys=annotation_keys,
                label_strategy=label_strategy,
                mode=mode,
                id_column=id_column or None,
                lang_column=lang_column or None,
                key_strategies=key_strategies if 'key_strategies' in locals() else None
            )
            bundle = builder.build(request)

        # Store language metadata in bundle for later use (model selection will happen in training mode)
        if bundle:
            if confirmed_languages:
                bundle.metadata['confirmed_languages'] = confirmed_languages
            if language_distribution:
                bundle.metadata['language_distribution'] = language_distribution
            # Save training approach if user made a choice (multi-label/one-vs-all)
            if 'training_approach' in locals() and training_approach:
                bundle.metadata['training_approach'] = training_approach
                # CRITICAL: Derive multi_label flag from training_approach for UI and training
                if training_approach == 'multi-label':
                    bundle.metadata['multi_label'] = True
                    if 'multi_label_threshold' not in bundle.metadata:
                        bundle.metadata['multi_label_threshold'] = 0.5
            # Store annotation keys (categories) for benchmark mode
            # Use keys_to_train (which contains all keys when user selects ALL)
            if 'keys_to_train' in locals() and keys_to_train:
                bundle.metadata['categories'] = keys_to_train
            elif 'annotation_keys' in locals() and annotation_keys:
                bundle.metadata['categories'] = annotation_keys
            # Filter metadata to only include selected annotation keys
            selected_keys = None
            if 'annotation_keys' in locals() and annotation_keys:
                selected_keys = set(annotation_keys)
            elif 'keys_to_train' in locals() and keys_to_train:
                selected_keys = set(keys_to_train)
            bundle.metadata['all_keys_values'] = {
                key: sorted(list(values))
                for key, values in all_keys_values.items()
                if selected_keys is None or key in selected_keys
            }
            bundle.metadata['value_counts_by_key'] = {
                key: {val: int(count) for val, count in counts.items()}
                for key, counts in value_counts_by_key.items()
                if selected_keys is None or key in selected_keys
            }
            # Store source file and annotation column for benchmark mode
            bundle.metadata['source_file'] = str(csv_path)
            bundle.metadata['annotation_column'] = annotation_column
            # Store split configuration if it exists
            if 'split_config' in locals() and split_config:
                bundle.metadata['split_config'] = split_config
            # Text length stats for intelligent model selection later
            # ONLY calculate if not already done (avoid duplicate analysis)
            if 'text_length_stats' in locals() and text_length_stats:
                # Already calculated with user interaction - reuse it
                bundle.metadata['text_length_stats'] = text_length_stats
            elif text_column in df.columns:
                # Not calculated yet - do it now without UI
                text_length_stats = self.analyze_text_lengths(
                    df=df,
                    text_column=text_column,
                    display_results=False  # Silent calculation
                )
                bundle.metadata['text_length_stats'] = text_length_stats

        return bundle

    if format_choice == "category-csv":
        # DEVELOPMENT MODE: This format is not yet available
        self.console.print("\n[bold red][FAIL] Error: category-csv format is currently under development[/bold red]")
        self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
        self.console.print("[dim]Please use 'llm-json' format instead.[/dim]\n")
        return None

        # Ask user for training strategy (mono-label vs multi-label)
        self.console.print("\n[bold cyan]Training Strategy Selection[/bold cyan]\n")
        self.console.print("[dim]Choose how to handle the labels in your dataset:[/dim]\n")

        strategy_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
        strategy_table.add_column("Strategy", style="cyan bold", no_wrap=True)
        strategy_table.add_column("Description", style="white", overflow="fold", ratio=1)

        strategy_table.add_row(
            "single-label",
            "Each sample has ONE label/category\n"
            "✓ Best for: classification tasks (sentiment, topic, etc.)\n"
            "✓ Example: each text is either 'positive' OR 'negative'"
        )
        strategy_table.add_row(
            "multi-label",
            "Each sample can have MULTIPLE labels\n"
            "✓ Best for: tagging, multiple categories per text\n"
            "✓ Example: a text can be 'politics' AND 'economy' AND 'urgent'"
        )

        self.console.print(strategy_table)
        self.console.print()

        mode = Prompt.ask(
            "[bold yellow]Training strategy[/bold yellow]",
            choices=["single-label", "multi-label", "back"],
            default="single-label"
        )

        if mode == "back":
            return None

        # If single-label, ALWAYS ask about training approach (even for binary/2 classes)
        training_approach = "multi-class"  # Default
        if mode == "single-label":
            # Count unique labels
            import pandas as pd
            import json
            df = pd.read_csv(selection['data_path'])
            label_column = selection['label_column']
            num_unique_labels = df[label_column].nunique()

            # Always ask, even for binary classification (user may want one-vs-all)
            self.console.print(f"\n[bold cyan]Training Approach for {num_unique_labels} Categories[/bold cyan]\n")
            if num_unique_labels == 2:
                self.console.print("[dim]Even with 2 categories, you can choose between multi-class or one-vs-all:[/dim]\n")
            else:
                self.console.print("[dim]Choose how to train with multiple categories:[/dim]\n")

            approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            approach_table.add_column("Approach", style="cyan bold", no_wrap=True)
            approach_table.add_column("Description", style="white", overflow="fold", ratio=1)

            approach_table.add_row(
                "multi-class",
                f"ONE model predicting among {num_unique_labels} categories\n"
                "✓ Faster training (1 model only)\n"
                "✓ Model learns relationships between categories\n"
                "✓ Best for: mutually exclusive categories"
            )
            approach_table.add_row(
                "one-vs-all",
                f"{num_unique_labels} binary models (one per category)\n"
                "✓ Each model: 'Category X' vs 'NOT Category X'\n"
                "✓ Better for: imbalanced data or category-specific tuning\n"
                "✓ Longer training but more flexible"
            )
            approach_table.add_row(
                "multi-label",
                f"ONE model with MULTIPLE labels per text\n"
                "✓ Uses sigmoid activation (independent probabilities)\n"
                "✓ Each text can have 0, 1, or many labels simultaneously\n"
                "✓ Best for: texts with overlapping categories"
            )

            self.console.print(approach_table)
            self.console.print()

            training_approach = Prompt.ask(
                "[bold yellow]Training approach[/bold yellow]",
                choices=["multi-class", "one-vs-all", "multi-label", "back"],
                default="multi-class"
            )

            if training_approach == "back":
                return None

        # If one-vs-all, convert to multi-label format (one binary file per category)
        if training_approach == "one-vs-all":
            # Convert single-label multi-class to multi-label one-vs-all format
            # This will create one binary file per category
            request = TrainingDataRequest(
                input_path=selection['data_path'],
                format="category_csv",
                text_column=selection['text_column'],
                label_column=selection['label_column'],
                id_column=selection.get('id_column'),
                lang_column=selection.get('lang_column'),
                mode="multi-label",  # Use multi-label to trigger one-vs-all training
            )
            bundle = builder.build(request)

            # Mark this as one-vs-all for distributed training
            if bundle:
                bundle.metadata['training_approach'] = 'one-vs-all'
                bundle.metadata['original_strategy'] = 'single-label-multiclass'
        elif training_approach == "multi-label":
            # True multi-label: one model with sigmoid activation for multiple labels per text
            request = TrainingDataRequest(
                input_path=selection['data_path'],
                format="category_csv",
                text_column=selection['text_column'],
                label_column=selection['label_column'],
                id_column=selection.get('id_column'),
                lang_column=selection.get('lang_column'),
                mode="multi-label",  # Use multi-label data format
            )
            bundle = builder.build(request)

            # Mark this as true multi-label (not one-vs-all)
            if bundle:
                bundle.metadata['training_approach'] = 'multi-label'
                bundle.metadata['multi_label'] = True
                bundle.metadata['multi_label_threshold'] = 0.5
        else:
            # Standard multi-class: one model for all categories
            request = TrainingDataRequest(
                input_path=selection['data_path'],
                format="category_csv",
                text_column=selection['text_column'],
                label_column=selection['label_column'],
                id_column=selection.get('id_column'),
                lang_column=selection.get('lang_column'),
                mode=mode,
            )
            bundle = builder.build(request)

        # Store recommended model and metadata in bundle for later use
        if bundle:
            if selection.get('recommended_model'):
                bundle.recommended_model = selection['recommended_model']
            if selection.get('confirmed_languages'):
                bundle.metadata['confirmed_languages'] = selection['confirmed_languages']
            if selection.get('language_distribution'):
                bundle.metadata['language_distribution'] = selection['language_distribution']
            if selection.get('text_length_stats'):
                bundle.metadata['text_length_stats'] = selection['text_length_stats']
            # Store split configuration if it exists
            if 'split_config' in locals() and split_config:
                bundle.metadata['split_config'] = split_config

        return bundle

    if format_choice == "binary-long":
        # DEVELOPMENT MODE: This format is not yet available
        self.console.print("\n[bold red][FAIL] Error: binary-long format is currently under development[/bold red]")
        self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
        self.console.print("[dim]Please use 'llm-json' format instead.[/dim]\n")
        return None

        # Use sophisticated universal selector
        selection = self._training_studio_intelligent_dataset_selector(format_type="binary-long")
        if not selection:
            return None

        # Binary-long specific: need category and value columns
        category_column = Prompt.ask("\n[bold yellow]Category column[/bold yellow]", default="category")
        value_column = Prompt.ask("[bold yellow]Value column (0/1)[/bold yellow]", default="value")

        request = TrainingDataRequest(
            input_path=selection['data_path'],
            format="binary_long_csv",
            text_column=selection['text_column'],
            category_column=category_column,
            value_column=value_column,
            id_column=selection.get('id_column'),
            lang_column=selection.get('lang_column'),
            mode="multi-label",
        )
        bundle = builder.build(request)

        # Store recommended model and metadata in bundle for later use
        if bundle:
            if selection.get('recommended_model'):
                bundle.recommended_model = selection['recommended_model']
            if selection.get('confirmed_languages'):
                bundle.metadata['confirmed_languages'] = selection['confirmed_languages']
            if selection.get('language_distribution'):
                bundle.metadata['language_distribution'] = selection['language_distribution']
            if selection.get('text_length_stats'):
                bundle.metadata['text_length_stats'] = selection['text_length_stats']
            # Store split configuration if it exists
            if 'split_config' in locals() and split_config:
                bundle.metadata['split_config'] = split_config

        return bundle

    if format_choice == "jsonl-single":
        # DEVELOPMENT MODE: This format is not yet available
        self.console.print("\n[bold red][FAIL] Error: jsonl-single format is currently under development[/bold red]")
        self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
        self.console.print("[dim]Please use 'llm-json' format instead.[/dim]\n")
        return None

        # Use sophisticated universal selector
        selection = self._training_studio_intelligent_dataset_selector(format_type="jsonl-single")
        if not selection:
            return None

        request = TrainingDataRequest(
            input_path=selection['data_path'],
            format="jsonl_single",
            text_column=selection['text_column'],
            label_column=selection['label_column'],
            mode="single-label",
        )
        bundle = builder.build(request)

        # Store recommended model and metadata in bundle for later use
        if bundle:
            if selection.get('recommended_model'):
                bundle.recommended_model = selection['recommended_model']
            if selection.get('confirmed_languages'):
                bundle.metadata['confirmed_languages'] = selection['confirmed_languages']
            if selection.get('language_distribution'):
                bundle.metadata['language_distribution'] = selection['language_distribution']
            if selection.get('text_length_stats'):
                bundle.metadata['text_length_stats'] = selection['text_length_stats']
            # Store split configuration if it exists
            if 'split_config' in locals() and split_config:
                bundle.metadata['split_config'] = split_config

        return bundle

    # jsonl-multi (should not be reached - format is not in choices list)
    if format_choice == "jsonl-multi":
        # DEVELOPMENT MODE: This format is not yet available
        self.console.print("\n[bold red][FAIL] Error: jsonl-multi format is currently under development[/bold red]")
        self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
        self.console.print("[dim]Please use 'llm-json' format instead.[/dim]\n")
        return None

    # Fallback: unrecognized format
    self.console.print(f"\n[bold red][FAIL] Error: Unknown format '{format_choice}'[/bold red]")
    self.console.print("[dim]Supported formats: llm-json[/dim]\n")
    return None

def _display_model_details(self, model_id: str, MODEL_METADATA: dict):
    """Display complete model information including full description."""
    from rich.panel import Panel
    from rich.text import Text

    meta = MODEL_METADATA.get(model_id, {})
    if not meta:
        self.console.print(f"[red]Model '{model_id}' not found in metadata[/red]")
        return

    # Create detailed info panel
    info = Text()
    info.append(f"Model: ", style="bold cyan")
    info.append(f"{model_id}\n\n", style="bold white")

    info.append(f"Languages: ", style="bold yellow")
    langs_list = meta.get('languages', ['?'])
    if len(langs_list) > 10:  # Multilingual models with many languages
        # Show key languages and total count
        key_langs = ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'ZH', 'JA', 'AR', 'RU']
        shown_langs = [l for l in key_langs if l in langs_list][:6]  # Show max 6 key languages
        langs = ', '.join(shown_langs) + f' + {len(langs_list)-6} more languages (Total: {len(langs_list)})'
    else:
        langs = ', '.join(langs_list)
    info.append(f"{langs}\n", style="white")

    info.append(f"Max Tokens: ", style="bold blue")
    info.append(f"{meta.get('max_length', '?')}\n", style="white")

    info.append(f"Size: ", style="bold magenta")
    info.append(f"{meta.get('size', '?')}\n\n", style="white")

    info.append(f"Description:\n", style="bold green")
    # Full description, not truncated
    full_desc = meta.get('description', 'No description available')
    info.append(full_desc, style="dim white")

    panel = Panel(info, title="Model Details", border_style="cyan", expand=False)
    self.console.print(panel)

def _run_benchmark_mode(
    self,
    bundle: TrainingDataBundle,
    languages: set,
    train_by_language: bool,
    text_length_avg: float,
    prefers_long_models: bool
) -> Optional[Dict[str, Any]]:
    """
    Execute complete benchmark mode workflow.

    Steps:
    1. Multi-model selection (≥2 per language or ≥2 multilingual)
    2. Class imbalance analysis
    3. Category selection
    4. Benchmark execution (quick training 3-5 epochs)
    5. Results display and ranking
    6. Final model selection

    Args:
        bundle: Training data bundle
        languages: Set of detected languages
        train_by_language: Whether training per-language
        text_length_avg: Average text length
        prefers_long_models: Whether long models preferred

    Returns:
        Dict with selected models or None to stop
    """
    from llm_tool.utils.model_display import get_recommended_models, MODEL_METADATA
    from llm_tool.utils.benchmark_utils import (
        analyze_categories_imbalance,
        select_benchmark_categories,
        format_imbalance_summary,
        create_benchmark_dataset,
        compare_model_results,
        select_representative_benchmark_items
    )
    from llm_tool.utils.benchmark_helpers import (
        split_benchmark_by_category,
        validate_label_sufficiency
    )
    from llm_tool.trainers.model_trainer import ModelTrainer, TrainingConfig
    from rich.prompt import IntPrompt
    from rich.table import Table
    from rich import box
    import tempfile
    from pathlib import Path
    import json
    import pandas as pd
    from datetime import datetime

    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    self.console.print("[bold cyan]           BENCHMARK MODE - Model Comparison                [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Load original source file from bundle metadata (not the transformed JSONL)
    source_file = bundle.metadata.get('source_file')
    if not source_file:
        self.console.print("[red][FAIL] Cannot run benchmark: source file not found in bundle metadata[/red]")
        self.console.print("[dim]Bundle metadata keys:[/dim] " + ", ".join(bundle.metadata.keys()))
        return None

    source_path = Path(source_file)
    if not source_path.exists():
        self.console.print(f"[red][FAIL] Source file not found: {source_path}[/red]")
        return None

    # Get annotation column from metadata
    annotation_column = bundle.metadata.get('annotation_column')
    if not annotation_column:
        self.console.print("[red][FAIL] Annotation column not found in bundle metadata[/red]")
        return None

    # Load data based on file format
    try:
        file_ext = source_path.suffix.lower()
        if file_ext == '.csv':
            original_dataframe = pd.read_csv(source_path)
        elif file_ext in ['.xlsx', '.xls']:
            original_dataframe = pd.read_excel(source_path)
        elif file_ext == '.parquet':
            original_dataframe = pd.read_parquet(source_path)
        elif file_ext in ['.json', '.jsonl']:
            original_dataframe = pd.read_json(source_path, lines=(file_ext == '.jsonl'))
        else:
            self.console.print(f"[red][FAIL] Unsupported file format: {file_ext}[/red]")
            return None
    except Exception as e:
        self.console.print(f"[red][FAIL] Error loading data: {e}[/red]")
        return None

    self.logger.debug(f"Loaded source file: {source_path}")
    self.logger.debug(f"Using annotation column: {annotation_column}")

    # ======================== STEP 1: Multi-Model Selection ========================
    self.console.print("[bold]STEP 1: Select Models to Benchmark[/bold]\n")

    selected_models_benchmark = []
    models_by_language_benchmark = {}

    if train_by_language:
        # Select multiple models per language
        self.console.print(f"[yellow]You'll select at least 2 models for each language: {', '.join(sorted(languages))}[/yellow]\n")

        for lang in sorted(languages):
            self.console.print(f"\n[bold yellow]{'─'*60}[/bold yellow]")
            self.console.print(f"[bold yellow]Selecting models for {lang} texts[/bold yellow]")
            self.console.print(f"[bold yellow]{'─'*60}[/bold yellow]\n")

            lang_models = []

            # Get recommendations
            lang_recommended = get_recommended_models(
                languages={lang},
                avg_text_length=text_length_avg,
                requires_long_model=prefers_long_models,
                top_n=10
            )

            while True:
                # Show models
                if lang_recommended:
                    self.console.print(f"[bold cyan]Top 10 Recommended Models for {lang}:[/bold cyan]\n")

                    models_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
                    models_table.add_column("#", style="yellow", width=3, no_wrap=True)
                    models_table.add_column("Model ID", style="cyan", no_wrap=True)
                    models_table.add_column("Languages", style="green", no_wrap=True)
                    models_table.add_column("Max Tokens", style="blue", no_wrap=True)
                    models_table.add_column("Size", style="magenta", no_wrap=True)
                    models_table.add_column("Description", style="white", overflow="fold", ratio=1)

                    for idx, model_id in enumerate(lang_recommended[:10], 1):
                        meta = MODEL_METADATA.get(model_id, {})
                        from llm_tool.utils.model_display import format_language_display
                        langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                        max_len = str(meta.get('max_length', '?'))
                        size = meta.get('size', '?')
                        desc = meta.get('description', '')[:44] + '..' if len(meta.get('description', '')) > 44 else meta.get('description', '')
                        models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

                    self.console.print(models_table)
                    # Default to next model in recommendations based on how many already selected
                    default_idx = min(len(lang_models), len(lang_recommended) - 1)
                    default_model = lang_recommended[default_idx]
                else:
                    default_model = 'bert-base-uncased'

                if lang_models:
                    self.console.print(f"\n[green]✓ Already selected {len(lang_models)} model(s) for {lang}:[/green]")
                    for m in lang_models:
                        self.console.print(f"  • {m}")

                # Show selection hint
                self.console.print(f"\n[dim]Tip: Tip: Type 'info X' (e.g., 'info 1') to see full details of a model[/dim]")

                model_input = Prompt.ask(
                    f"\n[bold yellow]{'Add' if lang_models else 'Select'} model #{len(lang_models)+1} for {lang}[/bold yellow]",
                    default=default_model
                )

                # Check if user wants info on a model
                if model_input.lower().startswith('info '):
                    info_target = model_input[5:].strip()
                    if info_target.isdigit():
                        info_idx = int(info_target) - 1
                        if lang_recommended and 0 <= info_idx < len(lang_recommended):
                            self._display_model_details(lang_recommended[info_idx], MODEL_METADATA)
                        else:
                            self.console.print(f"[red]Invalid model number: {info_target}[/red]")
                    else:
                        self._display_model_details(info_target, MODEL_METADATA)
                    continue  # Ask again for selection

                # Parse selection
                if model_input.isdigit():
                    idx = int(model_input) - 1
                    if lang_recommended and 0 <= idx < len(lang_recommended):
                        selected_model = lang_recommended[idx]
                    else:
                        selected_model = default_model
                else:
                    selected_model = model_input

                # Validate model exists (check in MODEL_METADATA or HuggingFace format)
                if selected_model not in MODEL_METADATA and '/' not in selected_model:
                    self.console.print(f"[yellow][!] Model '{selected_model}' not found in metadata[/yellow]")
                    # Ask if they want to use it anyway
                    use_anyway = Confirm.ask(
                        f"[yellow]Use '{selected_model}' anyway? (may fail if invalid)[/yellow]",
                        default=False
                    )
                    if not use_anyway:
                        continue  # Ask for selection again

                lang_models.append(selected_model)
                self.console.print(f"[green]Added: {selected_model}[/green]")

                # Display full model details after selection
                self._display_model_details(selected_model, MODEL_METADATA)

                # Ask to add more (require at least 2)
                if len(lang_models) >= 2:
                    add_more = Confirm.ask(
                        f"\n[cyan]Add another model for {lang}? (Current: {len(lang_models)})[/cyan]",
                        default=False
                    )
                    if not add_more:
                        break
                else:
                    self.console.print(f"[yellow][!] At least 2 models required. Please select one more.[/yellow]")

            models_by_language_benchmark[lang] = lang_models
            self.console.print(f"\n[green]✓ {len(lang_models)} models selected for {lang}[/green]")

    else:
        # Select multiple multilingual or single-language models
        self.console.print("[yellow]Select at least 2 models to benchmark[/yellow]\n")

        # Determine recommendation language
        if len(languages) > 1:
            languages_for_recommendation = {'MULTI'}
        else:
            languages_for_recommendation = languages

        recommended_models_list = get_recommended_models(
            languages=languages_for_recommendation,
            avg_text_length=text_length_avg,
            requires_long_model=prefers_long_models,
            top_n=10
        )

        while True:
            # Show models
            if recommended_models_list:
                self.console.print("[bold cyan]Top 10 Recommended Models:[/bold cyan]\n")

                models_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
                models_table.add_column("#", style="yellow", width=3, no_wrap=True)
                models_table.add_column("Model ID", style="cyan", no_wrap=True)
                models_table.add_column("Languages", style="green", no_wrap=True)
                models_table.add_column("Max Tokens", style="blue", no_wrap=True)
                models_table.add_column("Size", style="magenta", no_wrap=True)
                models_table.add_column("Description", style="white", overflow="fold", ratio=1)

                for idx, model_id in enumerate(recommended_models_list[:10], 1):
                    meta = MODEL_METADATA.get(model_id, {})
                    from llm_tool.utils.model_display import format_language_display
                    langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                    max_len = str(meta.get('max_length', '?'))
                    size = meta.get('size', '?')
                    desc = meta.get('description', '')[:44] + '..' if len(meta.get('description', '')) > 44 else meta.get('description', '')
                    models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

                self.console.print(models_table)

            if selected_models_benchmark:
                self.console.print(f"\n[green]✓ Already selected {len(selected_models_benchmark)} model(s):[/green]")
                for m in selected_models_benchmark:
                    self.console.print(f"  • {m}")

            # Default to next model in recommendations based on how many already selected
            if recommended_models_list:
                default_idx = min(len(selected_models_benchmark), len(recommended_models_list) - 1)
                default_model = recommended_models_list[default_idx]
            else:
                default_model = 'bert-base-uncased'

            # Show selection hint
            self.console.print(f"\n[dim]Tip: Tip: Type 'info X' (e.g., 'info 1') to see full details of a model[/dim]")

            model_input = Prompt.ask(
                f"\n[bold yellow]{'Add' if selected_models_benchmark else 'Select'} model #{len(selected_models_benchmark)+1}[/bold yellow]",
                default=default_model
            )

            # Check if user wants info on a model
            if model_input.lower().startswith('info '):
                info_target = model_input[5:].strip()
                if info_target.isdigit():
                    info_idx = int(info_target) - 1
                    if recommended_models_list and 0 <= info_idx < len(recommended_models_list):
                        self._display_model_details(recommended_models_list[info_idx], MODEL_METADATA)
                    else:
                        self.console.print(f"[red]Invalid model number: {info_target}[/red]")
                else:
                    self._display_model_details(info_target, MODEL_METADATA)
                continue  # Ask again for selection

            # Parse selection
            if model_input.isdigit():
                idx = int(model_input) - 1
                if recommended_models_list and 0 <= idx < len(recommended_models_list):
                    selected_model = recommended_models_list[idx]
                else:
                    selected_model = default_model
            else:
                selected_model = model_input

            # Validate model exists (check in MODEL_METADATA or HuggingFace format)
            if selected_model not in MODEL_METADATA and '/' not in selected_model:
                self.console.print(f"[yellow][!] Model '{selected_model}' not found in metadata[/yellow]")
                # Ask if they want to use it anyway
                use_anyway = Confirm.ask(
                    f"[yellow]Use '{selected_model}' anyway? (may fail if invalid)[/yellow]",
                    default=False
                )
                if not use_anyway:
                    continue  # Ask for selection again

            # Check for duplicates
            if selected_model in selected_models_benchmark:
                self.console.print(f"[yellow][!] Model '{selected_model}' is already selected. Please choose a different model.[/yellow]")
                continue

            selected_models_benchmark.append(selected_model)
            self.console.print(f"[green]Added: {selected_model}[/green]")

            # Display full model details after selection
            self._display_model_details(selected_model, MODEL_METADATA)

            # Ask to add more (require at least 2)
            if len(selected_models_benchmark) >= 2:
                add_more = Confirm.ask(
                    f"\n[cyan]Add another model? (Current: {len(selected_models_benchmark)})[/cyan]",
                    default=False
                )
                if not add_more:
                    break
            else:
                self.console.print(f"[yellow][!] At least 2 models required. Please select one more.[/yellow]")

    # Deduplicate models and track changes
    if train_by_language:
        for lang in models_by_language_benchmark:
            original_count = len(models_by_language_benchmark[lang])
            # Remove duplicates while preserving order
            models_by_language_benchmark[lang] = list(dict.fromkeys(models_by_language_benchmark[lang]))
            deduped_count = len(models_by_language_benchmark[lang])
            if deduped_count < original_count:
                self.console.print(f"\n[dim]  • {lang}: Removed {original_count - deduped_count} duplicate(s), {deduped_count} unique model(s) remaining[/dim]")
    else:
        original_count = len(selected_models_benchmark)
        selected_models_benchmark = list(dict.fromkeys(selected_models_benchmark))
        deduped_count = len(selected_models_benchmark)
        if deduped_count < original_count:
            self.console.print(f"\n[dim]  • Removed {original_count - deduped_count} duplicate(s), {deduped_count} unique model(s) remaining[/dim]")

    # Summary
    self.console.print("\n[bold green]✓ Model Selection Complete[/bold green]")
    if train_by_language:
        total_models = sum(len(models) for models in models_by_language_benchmark.values())
        for lang, models in sorted(models_by_language_benchmark.items()):
            self.console.print(f"  • {lang}: [cyan]{len(models)} model(s)[/cyan]")
            for m in models:
                self.console.print(f"    - {m}")
        if total_models < 2:
            self.console.print(f"\n[red][FAIL] Only {total_models} unique model(s) - benchmark requires at least 2 different models[/red]")
            return None
    else:
        self.console.print(f"  • [cyan]{len(selected_models_benchmark)} unique model(s)[/cyan]")
        for m in selected_models_benchmark:
            self.console.print(f"    - {m}")
        if len(selected_models_benchmark) < 2:
            self.console.print(f"\n[red][FAIL] Only {len(selected_models_benchmark)} unique model(s) - benchmark requires at least 2 different models[/red]")
            return None

    # ======================== STEP 2: Training Epochs ========================
    # Reinforced learning is DISABLED by default for benchmark (fair comparison)
    enable_benchmark_rl = False
    rl_f1_threshold = 0.70
    rl_oversample_factor = 2.0
    rl_class_weight_factor = 2.0
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    self.console.print("[bold cyan]           STEP 3: Training Epochs (Benchmark)              [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    self.console.print("[bold]What are Epochs?[/bold]")
    self.console.print("  • [cyan]One epoch[/cyan] = One complete pass through your entire training dataset")
    self.console.print("  • [cyan]More epochs[/cyan] = Model sees and learns from data more times")
    self.console.print("  • [cyan]Typical range[/cyan]: 3-15 epochs for BERT-like models\n")

    self.console.print("[bold]Guidelines:[/bold]")
    self.console.print("  • [green]Small dataset (<1000 samples)[/green]: 10-15 epochs recommended")
    self.console.print("  • [green]Medium dataset (1000-10000)[/green]: 5-10 epochs recommended")
    self.console.print("  • [green]Large dataset (>10000)[/green]: 3-5 epochs recommended\n")

    self.console.print("[bold green]Automatic Best Model Checkpointing:[/bold green]")
    self.console.print("  • [cyan]Don't worry about setting too many epochs![/cyan]")
    self.console.print("  • The [bold]BEST model[/bold] is automatically saved during training")
    self.console.print("  • System monitors [yellow]validation F1 score[/yellow] after each epoch")
    self.console.print("  • Only the checkpoint with [bold green]highest F1[/bold green] is kept")
    self.console.print("  • Early stopping prevents overfitting automatically\n")

    self.console.print("[dim]Tip: Example: You set 15 epochs, but best F1 was at epoch 8 → Model from epoch 8 is used[/dim]\n")

    benchmark_epochs = IntPrompt.ask("[bold yellow]Number of epochs[/bold yellow]", default=10)

    # ======================== OPTIONAL: Reinforced Learning ========================
    self.console.print("\n[bold]Reinforced Learning (optional)[/bold]")
    self.console.print("[dim]Reinforced learning adds extra training epochs when F1 is below a threshold.[/dim]\n")

    self.console.print("[yellow][!] NOT RECOMMENDED for benchmark:[/yellow]")
    self.console.print("  • Adds variable epochs per model (unfair comparison)")
    self.console.print("  • Significantly increases benchmark time")
    self.console.print("  • May mask differences between base models\n")

    self.console.print("[green]RECOMMENDED: Keep RL disabled for benchmark[/green]")
    self.console.print("[dim]  RL will be available during final training after model selection.[/dim]\n")

    enable_rl_choice = Confirm.ask(
        "[bold]Enable reinforced learning for benchmark?[/bold]",
        default=False
    )

    if enable_rl_choice:
        enable_benchmark_rl = True
        self.console.print("[yellow][!] Reinforced learning enabled - benchmark may take significantly longer[/yellow]\n")
    else:
        self.console.print("[green]Reinforced learning disabled - fair comparison mode[/green]\n")

    # Store RL params
    # CRITICAL: Initialize reinforced_epochs with a default value to ensure global_max_epochs calculation works
    # Default to same as base epochs (user can override manually)
    benchmark_rl_params = {
        'f1_threshold': rl_f1_threshold,
        'oversample_factor': rl_oversample_factor,
        'class_weight_factor': rl_class_weight_factor,
        'reinforced_epochs': benchmark_epochs  # Default: same as base epochs (will be overridden if manually configured)
    }

    # Calculate and display total epochs (always show, even if RL disabled)
    from ..trainers.reinforced_params import get_reinforced_params

    if enable_benchmark_rl:
        self.console.print("\n[bold yellow][!]  Reinforced Learning Epoch Calculation[/bold yellow]\n")
        self.console.print("[dim]When F1 < {:.2f}, reinforced learning adds extra epochs.[/dim]".format(rl_f1_threshold))
        self.console.print("[dim]The table below shows the MAXIMUM possible epochs (worst case: F1 = 0.0)[/dim]\n")
    else:
        self.console.print("\n[bold cyan]Total Training Epochs[/bold cyan]\n")
        self.console.print("[dim]Reinforced learning is disabled. All models will train for the same number of epochs.[/dim]\n")

    # Create table showing epoch calculation (always show)
    epoch_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
    epoch_table.add_column("Model", style="yellow", overflow="fold", ratio=1)
    epoch_table.add_column("Base Epochs", style="cyan", justify="center", no_wrap=True)
    if enable_benchmark_rl:
        epoch_table.add_column("Max Reinforced", style="red", justify="center", no_wrap=True)
        epoch_table.add_column("Max Total", style="green bold", justify="center", no_wrap=True)
    else:
        epoch_table.add_column("Total Epochs", style="green bold", justify="center", no_wrap=True)

    max_total_epochs = benchmark_epochs

    # Get all models to calculate epochs for
    models_to_calculate = []
    if train_by_language:
        for lang, models in models_by_language_benchmark.items():
            models_to_calculate.extend(models)
    else:
        models_to_calculate = selected_models_benchmark

    for model_id in models_to_calculate:
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id

        if enable_benchmark_rl:
            # Calculate potential reinforced epochs (worst case: F1 = 0.0)
            reinforced_params = get_reinforced_params(
                model_name=model_name,
                best_f1_1=0.0,  # Worst case scenario
                original_lr=5e-5,
                num_classes=2
            )
            max_reinforced_epochs = reinforced_params.get('n_epochs', 0)
            total_possible = benchmark_epochs + max_reinforced_epochs

            if total_possible > max_total_epochs:
                max_total_epochs = total_possible

            epoch_table.add_row(
                model_id,
                str(benchmark_epochs),
                str(max_reinforced_epochs),
                str(total_possible)
            )
        else:
            # No reinforced learning - just show base epochs
            epoch_table.add_row(
                model_id,
                str(benchmark_epochs),
                str(benchmark_epochs)
            )

    self.console.print(epoch_table)
    self.console.print()

    # Ask for confirmation
    if enable_benchmark_rl:
        epochs_confirmed = Confirm.ask(
            f"[bold yellow]Continue with these epoch settings? (Max {max_total_epochs} epochs per model)[/bold yellow]",
            default=True
        )
    else:
        epochs_confirmed = Confirm.ask(
            f"[bold yellow]Continue with {benchmark_epochs} epoch(s) per model?[/bold yellow]",
            default=True
        )

    # Store manual reinforced epochs if configured
    manual_reinforced_epochs = None

    if not epochs_confirmed:
        # Ask what the user wants to configure
        self.console.print("\n[yellow]What would you like to configure?[/yellow]")

        # ──────────────────────────────────────────────────────────────
        # Step 1: Base Epochs Configuration (optional)
        # ──────────────────────────────────────────────────────────────
        modify_base = Confirm.ask(
            "[bold yellow]Modify base epochs?[/bold yellow]",
            default=True
        )

        if modify_base:
            benchmark_epochs = IntPrompt.ask(
                "[bold yellow]Base epochs for benchmark[/bold yellow]",
                default=benchmark_epochs
            )
            self.console.print(f"[green]Base epochs set to: {benchmark_epochs}[/green]\n")
        else:
            self.console.print(f"[green]Keeping base epochs at: {benchmark_epochs}[/green]\n")

        # ──────────────────────────────────────────────────────────────
        # Step 2: Reinforced Learning Epochs Configuration (independent)
        # ──────────────────────────────────────────────────────────────
        # NOTE: This section executes REGARDLESS of whether base epochs
        # were modified above. Both configurations are independent.
        if enable_benchmark_rl:
            configure_rl_epochs = Confirm.ask(
                "[bold yellow]Configure reinforced learning epochs manually?[/bold yellow]\n"
                "[dim](Default: auto-calculated based on model performance)[/dim]",
                default=False
            )

            if configure_rl_epochs:
                self.console.print("\n[bold cyan]ℹ Reinforced Learning Epochs:[/bold cyan]")
                self.console.print("[dim]These epochs will be used for ALL models when F1 < {:.2f}[/dim]".format(rl_f1_threshold))
                self.console.print("[dim]Auto-calculation typically uses 8-20 epochs based on model type[/dim]\n")

                manual_reinforced_epochs = IntPrompt.ask(
                    "[bold yellow]Reinforced epochs[/bold yellow]",
                    default=10
                )

                self.console.print(f"[green]Manual reinforced epochs set to: {manual_reinforced_epochs}[/green]\n")
            else:
                self.console.print("[green]Reinforced learning epochs will be auto-calculated[/green]\n")

    # Update RL params with manual reinforced epochs if configured
    if manual_reinforced_epochs is not None:
        benchmark_rl_params['reinforced_epochs'] = manual_reinforced_epochs

    # ======================== STEP 4: Category Selection ========================
    self.console.print("\n[bold]STEP 4: Select Categories for Benchmark[/bold]\n")
    self.console.print("[dim]Analyzing training data structure...[/dim]\n")

    import json

    # First, check bundle metadata for categories (for multi-class approach)
    metadata_categories = []
    if hasattr(bundle, 'metadata') and bundle.metadata:
        metadata_categories = bundle.metadata.get('categories', [])
        if metadata_categories:
            self.console.print(f"[cyan]✓ Found {len(metadata_categories)} categories from training configuration[/cyan]")
            for cat in metadata_categories[:10]:  # Show first 10
                self.console.print(f"  • {cat}")
            if len(metadata_categories) > 10:
                self.console.print(f"  ... and {len(metadata_categories) - 10} more")
            self.console.print()

    # If no metadata categories, analyze the actual data
    if not metadata_categories:
        self.console.print("[dim]No categories in metadata, analyzing annotations...[/dim]\n")

        unique_categories = set()
        for idx, row in original_dataframe.iterrows():
            annotation = row[annotation_column]

            # Parse if string
            if isinstance(annotation, str):
                try:
                    annotation = json.loads(annotation)
                except:
                    continue

            if isinstance(annotation, dict):
                unique_categories.update(annotation.keys())

        metadata_categories = list(unique_categories)

        if metadata_categories:
            self.console.print(f"[cyan]✓ Found {len(metadata_categories)} unique categor{'y' if len(metadata_categories) == 1 else 'ies'} in annotations[/cyan]\n")

    num_categories_in_data = len(metadata_categories)

    if num_categories_in_data == 0:
        self.console.print("[red][FAIL] No categories found in training data[/red]")
        self.console.print("[yellow]This may indicate an issue with the data conversion.[/yellow]")
        self.console.print("[dim]Benchmark requires category information for analysis.[/dim]\n")
        return None

    selected_benchmark_categories = []

    if num_categories_in_data == 1:
        # Only one category: Use the full dataset, no category selection needed
        self.console.print(f"[green]Single category detected: {metadata_categories[0]}[/green]")
        self.console.print(f"[dim]Benchmarking on full dataset (no filtering needed)[/dim]\n")

        # No category filtering needed
        selected_benchmark_categories = None  # Signal to use full dataset

    else:
        # Multiple categories: Analyze and select representative ones
        self.console.print(f"[yellow]Multiple categories detected ({num_categories_in_data} total)[/yellow]")
        self.console.print("[dim]Performing class imbalance analysis to suggest representative categories...[/dim]\n")

        # Analyze categories
        # CRITICAL: Only analyze categories that were selected for training
        imbalance_analysis = analyze_categories_imbalance(
            data=original_dataframe,
            annotation_column=annotation_column,
            filter_categories=metadata_categories  # Only analyze training-selected categories
        )

        if not imbalance_analysis:
            self.console.print("[red][FAIL] No categories found in annotations[/red]")
            return None

        # Select suggested categories
        suggested_categories = select_benchmark_categories(imbalance_analysis, num_categories=3)

        # Display analysis with explanation
        self.console.print("[bold cyan]Class Imbalance Analysis[/bold cyan]\n")

        self.console.print("[bold]Why This Analysis?[/bold]")
        self.console.print("[dim]To choose the best model, we need to test how each model performs on:[/dim]")
        self.console.print("[dim]  • [cyan]Balanced categories[/cyan] - Equal class distribution (easier, baseline performance)[/dim]")
        self.console.print("[dim]  • [yellow]Imbalanced categories[/yellow] - Skewed class distribution (harder, real-world scenario)[/dim]")
        self.console.print("[dim]This reveals which model handles both easy and challenging data best.[/dim]\n")

        self.console.print("[bold]Category Selection Strategy:[/bold]")
        self.console.print("[dim]The system automatically selects a mix of:[/dim]")
        self.console.print("[dim]  • Categories with different imbalance ratios (2:1, 5:1, 10:1+)[/dim]")
        self.console.print("[dim]  • Different sample sizes (small vs large datasets)[/dim]")
        self.console.print("[dim]  • Different numbers of classes (binary vs multi-class)[/dim]")
        self.console.print("[dim]This comprehensive test ensures you pick the model that performs well across all scenarios.[/dim]\n")

        categories_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
        categories_table.add_column("Category", style="yellow", no_wrap=True)
        categories_table.add_column("Profile", style="cyan", no_wrap=True)
        categories_table.add_column("Metrics", style="white", overflow="fold", ratio=1)

        for profile, profile_cats in suggested_categories.items():
            for cat in profile_cats:
                if cat in imbalance_analysis:
                    metrics = imbalance_analysis[cat]
                    categories_table.add_row(
                        cat,
                        profile.capitalize(),
                        format_imbalance_summary(metrics)
                    )

        self.console.print(categories_table)

        # Explain metrics
        self.console.print("\n[bold]Understanding the Metrics:[/bold]")
        self.console.print("[dim]  • [cyan]Ratio[/cyan] - Largest class / Smallest class (e.g., 5.3:1 means majority class is 5.3× larger)[/dim]")
        self.console.print("[dim]  • [cyan]Gini[/cyan] - Inequality coefficient (0=perfect balance, 1=extreme imbalance)[/dim]")
        self.console.print("[dim]  • [green]Balanced[/green]: Ratio < 2:1, Gini < 0.2 | [yellow]Moderate[/yellow]: Ratio 2-5:1, Gini 0.2-0.4 | [red]Imbalanced[/red]: Ratio > 5:1, Gini > 0.4[/dim]\n")

        # Collect all suggested
        all_suggested = []
        for cats in suggested_categories.values():
            all_suggested.extend(cats)

        # User choice
        self.console.print("[bold]Select categories for benchmark:[/bold]")
        self.console.print("  • Press [cyan]ENTER[/cyan] to use all suggested categories")
        self.console.print("  • Or enter [cyan]category names[/cyan] (comma-separated)")
        self.console.print("  • Or enter [cyan]'all'[/cyan] to see all available categories\n")

        choice = Prompt.ask("Categories", default="suggested")

        if choice in ["suggested", ""]:
            selected_benchmark_categories = all_suggested
        elif choice == "all":
            # Show all categories
            self.console.print("\n[bold cyan]All Available Categories:[/bold cyan]\n")

            all_cats_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
            all_cats_table.add_column("#", style="yellow", width=3)
            all_cats_table.add_column("Category", style="cyan", ratio=1, overflow="fold")
            all_cats_table.add_column("Classes", style="green", no_wrap=True)
            all_cats_table.add_column("Samples", style="blue", no_wrap=True)
            all_cats_table.add_column("Imbalance", style="white", no_wrap=True)

            sorted_cats = sorted(imbalance_analysis.items(), key=lambda x: x[1]['total_samples'], reverse=True)

            for idx, (cat, metrics) in enumerate(sorted_cats, 1):
                ratio = metrics.get('imbalance_ratio', 1.0)
                imb_level = "Balanced" if ratio < 2 else "Moderate" if ratio < 5 else "Imbalanced"

                all_cats_table.add_row(
                    str(idx),
                    cat,
                    str(metrics.get('num_classes', 0)),
                    str(metrics.get('total_samples', 0)),
                    f"{imb_level} ({ratio:.1f}:1)"
                )

            self.console.print(all_cats_table)

            self.console.print("\n[yellow]Enter category names or numbers (comma-separated):[/yellow]")
            selection = Prompt.ask("Selection")

            # Parse selection
            selected_benchmark_categories = []
            for item in selection.split(','):
                item = item.strip()
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(sorted_cats):
                        selected_benchmark_categories.append(sorted_cats[idx][0])
                else:
                    if item in imbalance_analysis:
                        selected_benchmark_categories.append(item)
        else:
            selected_benchmark_categories = [c.strip() for c in choice.split(',')]

        if not selected_benchmark_categories:
            self.console.print("[red][FAIL] No categories selected[/red]")
            return None

        self.console.print(f"\n[green]✓ Selected {len(selected_benchmark_categories)} categories:[/green]")
        for cat in selected_benchmark_categories:
            if cat in imbalance_analysis:
                metrics = imbalance_analysis[cat]
                self.console.print(f"  • {cat} ({metrics['total_samples']} samples, {metrics['num_classes']} classes)")

    # ======================== STEP 4.5: Representative Selection for Hybrid Mode ========================
    # For hybrid mode with one-vs-all categories, we need to select specific key+value pairs
    # to avoid training ALL values (which would be too slow for benchmark)
    benchmark_onevsall_selection = None  # Will store {category: [selected_labels]} if hybrid mode

    training_approach = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None
    multiclass_keys = bundle.metadata.get('multiclass_keys', []) if hasattr(bundle, 'metadata') else []
    multilabel_keys = bundle.metadata.get('multilabel_keys', []) if hasattr(bundle, 'metadata') else []
    onevsall_keys = bundle.metadata.get('onevsall_keys', []) if hasattr(bundle, 'metadata') else []

    # Check if hybrid mode with one-vs-all categories that need representative selection
    if training_approach in ['hybrid', 'custom'] and onevsall_keys:
        # Filter to only selected categories
        selected_onevsall = [k for k in onevsall_keys if k in (selected_benchmark_categories or [])]

        if selected_onevsall:
            self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
            self.console.print("[bold cyan]     STEP 4.5: Representative Selection for One-vs-All     [/bold cyan]")
            self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

            self.console.print("[bold]Why This Step?[/bold]")
            self.console.print("[dim]Your data uses [cyan]hybrid mode[/cyan] with one-vs-all categories.[/dim]")
            self.console.print("[dim]Each one-vs-all category has multiple values, each requiring its own binary model.[/dim]")
            self.console.print("[dim]Training ALL values would take too long for benchmark purposes.[/dim]\n")

            self.console.print("[bold]Representative Selection Strategy:[/bold]")
            self.console.print("[dim]To make benchmark efficient while still testing the one-vs-all strategy:[/dim]")
            self.console.print("[dim]  • Select 1 [green]balanced[/green] value (lowest imbalance ratio) - tests baseline performance[/dim]")
            self.console.print("[dim]  • Select 1 [yellow]imbalanced[/yellow] value (highest imbalance ratio) - tests hard cases[/dim]")
            self.console.print("[dim]This gives a representative view of model performance on one-vs-all data.[/dim]\n")

            # Call the automatic selection function
            representative_selection = select_representative_benchmark_items(
                bundle_metadata=bundle.metadata,
                data=original_dataframe,
                annotation_column=annotation_column,
                max_multiclass=len(multiclass_keys) + len(multilabel_keys),  # Keep all multi-class and multi-label
                max_onevsall_per_category=2  # 1 balanced + 1 imbalanced per category
            )

            # Display the automatic selection
            self.console.print("[bold cyan]Automatic Representative Selection:[/bold cyan]\n")

            selection_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            selection_table.add_column("#", style="yellow", width=3, no_wrap=True)
            selection_table.add_column("Item", style="cyan", overflow="fold")
            selection_table.add_column("Strategy", style="green", no_wrap=True)
            selection_table.add_column("Type", style="blue", no_wrap=True)
            selection_table.add_column("Reason", style="white", overflow="fold", ratio=1)

            item_num = 0
            for item_name, rationale in representative_selection['selection_rationale'].items():
                item_num += 1
                item_type = rationale.get('type', '-')
                if item_type == '-':
                    item_type = f"{rationale.get('num_classes', '?')} classes"

                selection_table.add_row(
                    str(item_num),
                    item_name,
                    rationale['strategy'],
                    item_type,
                    rationale['reason']
                )

            self.console.print(selection_table)

            # Show summary
            breakdown = representative_selection['strategy_breakdown']
            original_onevsall = breakdown['one-vs-all'].get('original_total', 0)
            selected_onevsall_count = breakdown['one-vs-all']['models']

            self.console.print(f"\n[bold]Benchmark Reduction:[/bold]")
            self.console.print(f"  • Multi-class models: [cyan]{breakdown['multi-class']['models']}[/cyan]")
            self.console.print(f"  • One-vs-all models: [cyan]{selected_onevsall_count}[/cyan] (instead of [red]{original_onevsall}[/red])")
            self.console.print(f"  • [green]Total models per base: {representative_selection['total_models']}[/green]")

            # Calculate time savings
            if len(all_models_to_test) if 'all_models_to_test' in dir() else len(selected_models_benchmark) > 0:
                num_bases = len(selected_models_benchmark) if selected_models_benchmark else 4
                original_total = (breakdown['multi-class']['models'] + original_onevsall) * num_bases
                new_total = representative_selection['total_models'] * num_bases
                if original_total > 0:
                    savings_pct = (1 - new_total / original_total) * 100
                    self.console.print(f"  • [yellow]Time savings: ~{savings_pct:.0f}% ({original_total} → {new_total} models)[/yellow]\n")

            # Ask for confirmation
            self.console.print("[bold]Choose an option:[/bold]")
            self.console.print("  [cyan]1[/cyan] - Use this selection (recommended)")
            self.console.print("  [cyan]2[/cyan] - Train ALL values (original, very slow)")
            self.console.print("  [cyan]3[/cyan] - Customize selection\n")

            selection_choice = Prompt.ask("Selection", choices=["1", "2", "3"], default="1")

            if selection_choice == "1":
                # Use automatic selection
                benchmark_onevsall_selection = {}
                for item in representative_selection['onevsall_items']:
                    cat = item['category']
                    if cat not in benchmark_onevsall_selection:
                        benchmark_onevsall_selection[cat] = []
                    # Store the full key_value name (e.g., "manual_themes_theme_agriculture")
                    benchmark_onevsall_selection[cat].append(item['key_value'])

                self.console.print("\n[green]✓ Using representative selection[/green]")
                for cat, items in benchmark_onevsall_selection.items():
                    self.console.print(f"  • {cat}: {len(items)} items selected")
                    for item in items:
                        self.console.print(f"    - {item}")

            elif selection_choice == "2":
                # Use all - benchmark_onevsall_selection stays None
                self.console.print("\n[yellow][!] Training ALL values - this may take a very long time[/yellow]")
                benchmark_onevsall_selection = None

            else:
                # Custom selection
                self.console.print("\n[bold]Custom Selection:[/bold]")
                self.console.print("[dim]For each one-vs-all category, enter the label names to include (comma-separated)[/dim]")
                self.console.print("[dim]Press ENTER to use suggested items for that category[/dim]\n")

                benchmark_onevsall_selection = {}
                all_keys_values = bundle.metadata.get('all_keys_values', {})

                for cat in selected_onevsall:
                    values = all_keys_values.get(cat, [])
                    if not values:
                        continue

                    # Show available values
                    self.console.print(f"\n[cyan]{cat}[/cyan] ({len(values)} values):")
                    for i, val in enumerate(values[:20], 1):  # Show first 20
                        full_name = f"{cat}_{val}"
                        # Mark suggested items
                        suggested_items = [item['key_value'] for item in representative_selection['onevsall_items'] if item['category'] == cat]
                        marker = "[green]✓[/green]" if full_name in suggested_items else " "
                        self.console.print(f"  {marker} {i}. {val}")
                    if len(values) > 20:
                        self.console.print(f"  ... and {len(values) - 20} more")

                    # Get user input
                    suggested_str = ", ".join([item['value'] for item in representative_selection['onevsall_items'] if item['category'] == cat])
                    user_selection = Prompt.ask(f"Select values for {cat}", default=suggested_str)

                    # Parse selection
                    selected_values = [v.strip() for v in user_selection.split(',') if v.strip()]
                    benchmark_onevsall_selection[cat] = [f"{cat}_{v}" for v in selected_values]

                self.console.print("\n[green]✓ Custom selection configured[/green]")

    # ======================== STEP 5: Execute Benchmark ========================
    self.console.print("\n[bold]STEP 5: Running Benchmark[/bold]\n")

    # Collect all models to test (with language mapping if train_by_language)
    all_models_to_test = []
    model_to_language_map = {}  # Track which language each model should use

    if train_by_language:
        for lang, models in models_by_language_benchmark.items():
            for model in models:
                all_models_to_test.append(model)
                model_to_language_map[model] = lang  # Remember this model is for this language
    else:
        all_models_to_test = selected_models_benchmark

    self.console.print(f"  • Models to test: [cyan]{len(all_models_to_test)}[/cyan]")
    if selected_benchmark_categories is not None:
        self.console.print(f"  • Categories: [cyan]{len(selected_benchmark_categories)}[/cyan]")
    else:
        self.console.print(f"  • Dataset: [cyan]Full training dataset[/cyan]")

    # Display epochs with reinforced learning info if enabled
    if enable_benchmark_rl:
        reinforced_epochs = benchmark_rl_params.get('reinforced_epochs', None)
        if reinforced_epochs is not None:
            # Manual reinforced epochs configured
            max_epochs = benchmark_epochs + reinforced_epochs
            self.console.print(f"  • Epochs per model: [cyan]{benchmark_epochs}[/cyan] (up to [yellow]{max_epochs}[/yellow] with reinforced learning)")
        else:
            # Auto-calculated reinforced epochs (typically 8-20)
            self.console.print(f"  • Epochs per model: [cyan]{benchmark_epochs}[/cyan] (up to [yellow]{benchmark_epochs}+auto[/yellow] with reinforced learning)")
        self.console.print(f"  • Reinforced learning: [cyan]Enabled[/cyan] (F1 < {benchmark_rl_params.get('f1_threshold', 0.70):.2f})")

        # Estimate time considering potential reinforced learning
        # Conservative estimate: assume some models will trigger RL
        estimated_avg_epochs = benchmark_epochs + (reinforced_epochs // 2 if reinforced_epochs else 5)
        estimated_minutes = len(all_models_to_test) * estimated_avg_epochs // 2
    else:
        self.console.print(f"  • Epochs per model: [cyan]{benchmark_epochs}[/cyan]")
        estimated_minutes = len(all_models_to_test) * benchmark_epochs // 2

    self.console.print(f"  • Estimated time: [yellow]~{estimated_minutes} minutes[/yellow]\n")

    proceed = Confirm.ask("[bold yellow]Proceed with benchmark?[/bold yellow]", default=True)
    if not proceed:
        return None

    # Prepare benchmark dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        # For single-label, use the bundle's primary file directly
        # For multi-label with category filtering, create filtered dataset
        if selected_benchmark_categories is None:
            # Single-label: Use existing training file
            benchmark_file = bundle.primary_file
            self.console.print(f"[green]Using full training dataset: {bundle.primary_file.name}[/green]\n")
        else:
            # Multi-label: Create filtered dataset
            self.console.print("\n[dim]Creating filtered benchmark dataset...[/dim]")
            benchmark_file = Path(tmpdir) / "benchmark_data.jsonl"

            # Create filtered dataset
            import json
            benchmark_rows = []

            for idx, row in original_dataframe.iterrows():
                annotation = row[annotation_column]

                # Parse if string
                if isinstance(annotation, str):
                    try:
                        annotation = json.loads(annotation)
                    except:
                        continue

                if not isinstance(annotation, dict):
                    continue

                # Filter to selected categories
                filtered_annotation = {
                    k: v for k, v in annotation.items()
                    if k in selected_benchmark_categories
                }

                if not filtered_annotation:
                    continue

                # Transform to multi-label format: list of "key_value" strings
                # E.g., {'sentiment': 'positive', 'theme': 'politics'} → ['sentiment_positive', 'theme_politics']
                # CRITICAL: Exclude 'null' string values
                label_list = []
                for key, value in filtered_annotation.items():
                    if isinstance(value, str) and value and value != 'null':
                        # Combine key and value into single label string
                        label_list.append(f"{key}_{value}")
                    elif isinstance(value, list):
                        # Handle list values (shouldn't happen in this flow, but be defensive)
                        for v in value:
                            if isinstance(v, str) and v and v != 'null':
                                label_list.append(f"{key}_{v}")

                if not label_list:
                    continue

                # Create row
                benchmark_row = {
                    'text': row[bundle.text_column],
                    'labels': label_list  # Always a list of label strings
                }

                # Add language if available
                for lang_col in ['language', 'lang']:
                    if lang_col in row.index:
                        benchmark_row['lang'] = row[lang_col]
                        break

                benchmark_rows.append(benchmark_row)

            # Save as JSONL
            with open(benchmark_file, 'w', encoding='utf-8') as f:
                for row in benchmark_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

            self.console.print(f"[green]Benchmark dataset created: {len(benchmark_rows)} samples[/green]\n")

        # Run benchmark for each model
        benchmark_results = {}

        # CRITICAL: Reuse the session ID from the training session (self.current_session_id)
        # This ensures benchmark and full training use THE SAME session folder.
        # The session_id format is: {user_name}_{YYYYMMDD_HHMMSS} (created at line 6558)
        if hasattr(self, 'current_session_id') and self.current_session_id:
            benchmark_session_id = self.current_session_id
            self.logger.info(f"✓ Benchmark reusing existing session: {benchmark_session_id}")
        else:
            # Fallback: create session_id if not yet initialized (should not happen in Training Arena)
            import datetime
            benchmark_session_id = datetime.datetime.now().strftime("training_session_%Y%m%d_%H%M%S")
            self.current_session_id = benchmark_session_id
            self.logger.warning(f"[!]  Created new session_id for benchmark (expected to reuse existing): {benchmark_session_id}")

        # CRITICAL: Display session information to user
        session_manager = getattr(self, 'current_session_manager', None)
        if session_manager and getattr(session_manager, 'session_dir', None):
            benchmark_metrics_dir = session_manager.session_dir / "training_metrics" / "benchmark"
        else:
            benchmark_metrics_dir = get_training_metrics_dir(benchmark_session_id) / "benchmark"

        self.logger.info("="*80)
        self.logger.info("SESSION MANAGEMENT - BENCHMARK")
        self.logger.info(f"  benchmark_session_id: {benchmark_session_id}")
        self.logger.info(f"  Models will be saved to: models/{benchmark_session_id}/benchmark/")
        self.logger.info(f"  Logs will be saved to: {benchmark_metrics_dir}")
        self.logger.info("="*80)
        self.console.print(f"\n[cyan]Session ID:[/cyan] [bold]{benchmark_session_id}[/bold]")
        self.console.print(f"[dim]All benchmark models will be saved to: models/{benchmark_session_id}/benchmark/[/dim]\n")

        # ============================================================
        # CRITICAL: Save initial benchmark metadata for session tracking
        # This enables session persistence and resume capability even if
        # training is interrupted or user chooses to exit benchmark early
        # ============================================================
        try:
            self.logger.info(" Saving initial benchmark metadata for session tracking...")

            # Build comprehensive benchmark configuration for metadata
            benchmark_model_config = {
                'training_mode': 'benchmark',
                'benchmark_enabled': True,
                'selected_models': selected_models_benchmark if not train_by_language else list(all_models_to_test),
                'models_by_language': models_by_language_benchmark if train_by_language else {},
                'train_by_language': train_by_language,
                'benchmark_categories': selected_benchmark_categories,
                'benchmark_epochs': benchmark_epochs,
                'reinforced_learning_enabled': enable_benchmark_rl,
                'rl_f1_threshold': benchmark_rl_params.get('f1_threshold', 0.70),
                'rl_oversample_factor': benchmark_rl_params.get('oversample_factor', 2.0),
                'rl_class_weight_factor': benchmark_rl_params.get('class_weight_factor', 2.0),
                'reinforced_epochs': benchmark_rl_params.get('reinforced_epochs'),
                'epochs': benchmark_epochs,
                'batch_size': _get_optimal_batch_size(bundle.recommended_model if hasattr(bundle, 'recommended_model') else None),
                'learning_rate': 2e-5
            }

            # Save metadata immediately (before any training starts)
            initial_metadata_path = self._save_training_metadata(
                bundle=bundle,
                mode='benchmark',
                model_config=benchmark_model_config,
                execution_status={
                    'status': 'benchmark_starting',
                    'started_at': datetime.now().isoformat(),
                    'completed_at': None,
                    'models_trained': [],
                    'models_to_test': list(all_models_to_test),
                    'best_model': None,
                    'best_f1': None,
                    'benchmark_phase': 'initialization'
                },
                session_id=benchmark_session_id,
                training_context={
                    'benchmark_mode': True,
                    'user_choices': {
                        'enable_benchmark': True,
                        'num_models_selected': len(all_models_to_test),
                        'selected_categories': selected_benchmark_categories
                    }
                }
            )
            self.console.print(f"[dim]Session metadata saved: {initial_metadata_path.name}[/dim]\n")
            self.logger.info(f"✓ Initial benchmark metadata saved: {initial_metadata_path}")

        except Exception as e:
            self.logger.error(f"Failed to save initial benchmark metadata: {e}")
            self.console.print(f"[yellow][!] Warning: Could not save session metadata: {e}[/yellow]\n")
            # Continue anyway - metadata saving should not block training

        # Initialize global progress tracking for benchmark
        import time
        global_start_time = time.time()
        global_total_models = len(all_models_to_test)

        # Calculate total epochs accounting for all categories
        # Each model must be trained on each category, so total = models × categories × epochs
        num_categories = 1 if selected_benchmark_categories is None else len(selected_benchmark_categories)
        global_total_epochs = global_total_models * num_categories * benchmark_epochs

        # Calculate maximum possible epochs (if all models trigger reinforced learning)
        if enable_benchmark_rl and benchmark_rl_params.get('reinforced_epochs') is not None:
            global_max_epochs = global_total_models * num_categories * (benchmark_epochs + benchmark_rl_params['reinforced_epochs'])
        else:
            global_max_epochs = global_total_epochs

        global_completed_epochs = 0

        # ============================================================
        # CRITICAL: Use robust preprocessing and validation from benchmark_helpers
        # ============================================================
        from llm_tool.utils.benchmark_helpers import (
            validate_label_sufficiency,
            split_benchmark_by_category,
            aggregate_benchmark_results,
            extract_numeric_metric,
            filter_languages_for_category
        )

        # Load and validate the benchmark data
        import pandas as pd
        benchmark_data = pd.read_json(benchmark_file, lines=True)

        # Validate label sufficiency with more robust handling
        self.console.print("[yellow]Validating benchmark data...[/yellow]")
        validated_data, validation_report = validate_label_sufficiency(
            benchmark_data,
            'labels',
            min_samples_per_class=3,  # More robust with 3 samples minimum
            strategy='multi-label'
        )

        # Display validation report
        if validation_report['removed_classes']:
            self.console.print(f"[yellow][!] Removed {len(validation_report['removed_classes'])} insufficient classes:[/yellow]")
            for removed in validation_report['removed_classes'][:5]:  # Show first 5
                self.console.print(f"    • {removed['label']}_{removed['class']}: {removed['count']} samples")
            if len(validation_report['removed_classes']) > 5:
                self.console.print(f"    ... and {len(validation_report['removed_classes']) - 5} more")

        if validation_report['removed_samples'] > 0:
            self.console.print(f"[yellow][!] Removed {validation_report['removed_samples']} samples with only insufficient classes[/yellow]")

        self.console.print(f"[green]Validated dataset: {validation_report['final_samples']} samples[/green]")

        # Save validated data
        validated_benchmark_file = Path(tmpdir) / "benchmark_validated.jsonl"
        validated_data.to_json(validated_benchmark_file, orient='records', lines=True, force_ascii=False)
        benchmark_file = validated_benchmark_file

        # Split data by category for independent training
        category_datasets = split_benchmark_by_category(
            validated_data,
            'labels',
            selected_benchmark_categories
        )

        category_language_reports: Dict[str, Dict[str, Any]] = {}
        skipped_categories: Dict[str, Dict[str, Any]] = {}
        filtered_category_datasets: Dict[str, pd.DataFrame] = {}

        for category, cat_df in category_datasets.items():
            filtered_df, lang_report = filter_languages_for_category(
                cat_df,
                category,
                min_samples_per_class=3
            )
            category_language_reports[category] = lang_report

            if lang_report.get('filtered'):
                kept_display = ", ".join(lang_report.get('languages_kept', [])) or "none"
                dropped_display = ", ".join(
                    f"{lang} (min counts: {lang_report['drop_reasons'].get(lang, {})})"
                    for lang in lang_report.get('languages_dropped', [])
                ) or "none"
                self.console.print(f"[yellow][!] Language filtering applied for {category}[/yellow]")
                self.console.print(f"    • Kept: {kept_display}")
                self.console.print(f"    • Dropped: {dropped_display}")

            if filtered_df.empty:
                skipped_categories[category] = lang_report
                self.console.print(f"[red]⏭  Skipping {category}: insufficient language coverage after filtering[/red]")
                continue

            filtered_category_datasets[category] = filtered_df

        if not filtered_category_datasets:
            self.console.print("[red][FAIL] No categories have sufficient multilingual coverage for benchmarking.[/red]")
            return None

        category_datasets = filtered_category_datasets

        self.console.print(f"\n[cyan]Category Distribution:[/cyan]")
        for category, cat_data in category_datasets.items():
            self.console.print(f"  • {category}: {len(cat_data)} samples")

        if not category_datasets:
            self.console.print(f"[red][FAIL] No valid data after preprocessing[/red]")
            return None

        # ============================================================
        # HYBRID/CUSTOM MODE HANDLING FOR BENCHMARK
        # Determine which categories use multi-class vs one-vs-all vs multi-label
        # ============================================================
        training_approach = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None
        multiclass_keys = bundle.metadata.get('multiclass_keys', []) if hasattr(bundle, 'metadata') else []
        multilabel_keys = bundle.metadata.get('multilabel_keys', []) if hasattr(bundle, 'metadata') else []
        onevsall_keys = bundle.metadata.get('onevsall_keys', []) if hasattr(bundle, 'metadata') else []

        # Build mapping: category -> strategy
        category_strategy_map: Dict[str, str] = {}
        if training_approach in ['hybrid', 'custom'] and (multiclass_keys or multilabel_keys or onevsall_keys):
            self.console.print(f"\n[cyan]Hybrid/Custom benchmark mode detected:[/cyan]")
            for cat in category_datasets.keys():
                if cat in multiclass_keys:
                    category_strategy_map[cat] = 'multi-class'
                elif cat in multilabel_keys:
                    category_strategy_map[cat] = 'multi-label'
                elif cat in onevsall_keys:
                    category_strategy_map[cat] = 'one-vs-all'
                else:
                    # Default to multi-label if not explicitly assigned
                    category_strategy_map[cat] = 'multi-label'

            mc_cats = [c for c, s in category_strategy_map.items() if s == 'multi-class']
            ml_cats = [c for c, s in category_strategy_map.items() if s == 'multi-label']
            ova_cats = [c for c, s in category_strategy_map.items() if s == 'one-vs-all']
            self.console.print(f"  • Multi-class categories: {mc_cats}")
            self.console.print(f"  • Multi-label categories: {ml_cats}")
            self.console.print(f"  • One-vs-all categories: {ova_cats}")
        else:
            # All categories use the bundle's default strategy
            for cat in category_datasets.keys():
                category_strategy_map[cat] = bundle.strategy

        # ============================================================
        # RECALCULATE TOTAL MODELS FOR HYBRID MODE
        # Multi-class: 1 model per category
        # One-vs-all: N binary models per category (1 per unique label)
        # ============================================================
        def _count_unique_labels_for_category(df: pd.DataFrame, category: str) -> int:
            """Count unique labels for a one-vs-all category."""
            import json
            unique_labels = set()
            for _, row in df.iterrows():
                labels = row.get('labels', [])
                if isinstance(labels, str):
                    try:
                        labels = json.loads(labels)
                    except:
                        labels = []
                for label in labels:
                    if isinstance(label, str) and label.startswith(f"{category}_"):
                        unique_labels.add(label)
            return len(unique_labels)

        # Count total sub-models per base model
        total_submodels_per_base = 0
        for cat, strategy in category_strategy_map.items():
            if strategy == 'one-vs-all' and cat in category_datasets:
                # Count binary models needed - use representative selection if available
                if benchmark_onevsall_selection and cat in benchmark_onevsall_selection:
                    # Only count selected items for this category
                    num_binary = len(benchmark_onevsall_selection[cat])
                else:
                    # No selection filter - count all unique labels
                    num_binary = _count_unique_labels_for_category(category_datasets[cat], cat)
                total_submodels_per_base += num_binary
            else:
                # Multi-class or multi-label: 1 model
                total_submodels_per_base += 1

        # Recalculate global totals
        num_base_models = len(all_models_to_test)
        global_total_models = num_base_models * total_submodels_per_base
        global_total_epochs = global_total_models * benchmark_epochs

        if enable_benchmark_rl and benchmark_rl_params.get('reinforced_epochs') is not None:
            global_max_epochs = global_total_models * (benchmark_epochs + benchmark_rl_params['reinforced_epochs'])
        else:
            global_max_epochs = global_total_epochs

        self.console.print(f"\n[cyan]Training Summary:[/cyan]")
        self.console.print(f"  • Base models to test: {num_base_models}")
        self.console.print(f"  • Sub-models per base: {total_submodels_per_base}")
        self.console.print(f"  • Total models to train: {global_total_models}")
        self.console.print(f"  • Total epochs: {global_total_epochs} (max: {global_max_epochs})")

        # Track current sub-model index across all base models
        current_submodel_index = 0

        # Run benchmark for each model
        for idx, model_id in enumerate(all_models_to_test, 1):
            self.console.print(f"\n[bold yellow]{'═' * 70}[/bold yellow]")
            self.console.print(f"[bold yellow]Testing Model {idx}/{len(all_models_to_test)}: {model_id}[/bold yellow]")
            self.console.print(f"[bold yellow]{'═' * 70}[/bold yellow]\n")

            try:
                # Create temp output dir
                model_output_dir = Path(tmpdir) / f"model_{idx}"
                model_output_dir.mkdir(exist_ok=True)

                # Train configuration
                config = TrainingConfig()
                metrics_base_dir = get_training_logs_base()
                config.metrics_output_dir = str(metrics_base_dir)
                config.num_epochs = benchmark_epochs
                # Dynamic batch size based on model and system resources
                config.batch_size = _get_optimal_batch_size(model_id)
                config.early_stopping_patience = max(2, benchmark_epochs // 5)
                config.output_dir = str(model_output_dir)
                # Propagate the Split/Exclude strategy chosen at the Token
                # Length step so benchmark runs honour it like the main training.
                if hasattr(bundle, 'metadata') and bundle.metadata:
                    config.exclude_long_texts = bool(
                        bundle.metadata.get('exclude_long_texts', False)
                    )
                    config.split_long_texts = bool(
                        bundle.metadata.get('split_long_texts', False)
                    )

                trainer = ModelTrainer(config=config)

                # Create progress callback to track completed epochs
                def progress_callback(**metrics):
                    """Callback to sync global completed epochs counter from authoritative source"""
                    nonlocal global_completed_epochs
                    # Update from callback metrics if available (bert_base.py is the authoritative source)
                    if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                        global_completed_epochs = metrics['global_completed_epochs']

                # Prepare training params
                train_params = {
                    'input_file': str(benchmark_file),
                    'model_name': model_id,
                    'num_epochs': benchmark_epochs,
                    'text_column': 'text',
                    'label_column': 'labels',
                    'training_strategy': bundle.strategy,
                    'output_dir': str(model_output_dir),
                    'is_benchmark': True,  # Flag to enable benchmark mode log structure
                    'session_id': benchmark_session_id,  # Unified session ID for all models in benchmark
                    'progress_callback': progress_callback,  # Add callback for epoch tracking
                    # Global progress tracking parameters
                    'global_total_models': global_total_models,
                    'global_current_model': idx,
                    'global_total_epochs': global_total_epochs,
                    'global_max_epochs': global_max_epochs,
                    'global_completed_epochs': global_completed_epochs,
                    'global_start_time': global_start_time,
                    # Pass training_approach from bundle metadata (one-vs-all vs multi-class)
                    'training_approach': bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None
                }

                # Add language filtering for per-language models
                if model_id in model_to_language_map:
                    # This is a language-specific model - only train on its language
                    model_lang = model_to_language_map[model_id]
                    train_params['confirmed_languages'] = [model_lang]
                    train_params['filter_by_language'] = model_lang  # Filter data to only this language
                elif hasattr(bundle, 'metadata') and bundle.metadata.get('confirmed_languages'):
                    # Multilingual model - use all languages
                    train_params['confirmed_languages'] = bundle.metadata['confirmed_languages']

                # Add reinforced learning params if enabled
                if enable_benchmark_rl:
                    train_params['reinforced_learning'] = True
                    train_params['rl_f1_threshold'] = benchmark_rl_params.get('f1_threshold', 0.70)
                    train_params['rl_oversample_factor'] = benchmark_rl_params.get('oversample_factor', 2.0)
                    train_params['rl_class_weight_factor'] = benchmark_rl_params.get('class_weight_factor', 2.0)
                    # Pass manual reinforced epochs if configured
                    if benchmark_rl_params.get('reinforced_epochs') is not None:
                        train_params['reinforced_epochs'] = benchmark_rl_params['reinforced_epochs']

                # Train with category-wise approach for robustness
                category_results = {}

                # Helper function to convert multi-label data to single-label for multi-class training
                def _convert_to_single_label(df: pd.DataFrame, category: str) -> pd.DataFrame:
                    """Convert multi-label format to single-label for a specific category."""
                    import json
                    rows = []
                    for _, row in df.iterrows():
                        labels = row.get('labels', [])
                        if isinstance(labels, str):
                            try:
                                labels = json.loads(labels)
                            except:
                                labels = []

                        # Find the label for this category (format: category_value)
                        category_label = None
                        for label in labels:
                            if isinstance(label, str) and label.startswith(f"{category}_"):
                                # Extract the value part after the category prefix
                                category_label = label
                                break

                        if category_label:
                            new_row = row.copy()
                            new_row['label'] = category_label  # Single label
                            rows.append(new_row)

                    return pd.DataFrame(rows)

                # Helper function to create binary datasets for one-vs-all
                def _create_binary_datasets_for_category(
                    df: pd.DataFrame,
                    category: str,
                    output_dir: Path
                ) -> Dict[str, Path]:
                    """Create binary datasets for each unique label in a one-vs-all category."""
                    import json
                    import csv

                    # Collect all unique labels for this category
                    unique_labels = set()
                    for _, row in df.iterrows():
                        labels = row.get('labels', [])
                        if isinstance(labels, str):
                            try:
                                labels = json.loads(labels)
                            except:
                                labels = []
                        for label in labels:
                            if isinstance(label, str) and label.startswith(f"{category}_"):
                                unique_labels.add(label)

                    binary_files = {}
                    for label_name in sorted(unique_labels):
                        csv_path = output_dir / f"binary_{label_name}.csv"

                        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                            fieldnames = ['text', 'label', 'language']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()

                            for _, row in df.iterrows():
                                text = row.get('text', '')
                                if not text or not isinstance(text, str) or not text.strip():
                                    continue

                                labels = row.get('labels', [])
                                if isinstance(labels, str):
                                    try:
                                        labels = json.loads(labels)
                                    except:
                                        labels = []

                                is_positive = label_name in labels
                                lang = row.get('lang', row.get('language', ''))

                                writer.writerow({
                                    'text': text.strip(),
                                    'label': 1 if is_positive else 0,
                                    'language': lang if isinstance(lang, str) else str(lang) if lang else ''
                                })

                        binary_files[label_name] = csv_path

                    return binary_files

                # Train on each category independently
                for cat_idx, (category, cat_data) in enumerate(category_datasets.items(), 1):
                    cat_strategy = category_strategy_map.get(category, 'multi-label')
                    self.console.print(f"\n[cyan]Training on category {cat_idx}/{len(category_datasets)}: {category} ({cat_strategy})[/cyan]")

                    lang_report = category_language_reports.get(category, {})
                    kept_languages = lang_report.get('languages_kept') or []

                    # ============================================================
                    # STRATEGY-SPECIFIC HANDLING
                    # ============================================================
                    if cat_strategy == 'one-vs-all':
                        # ONE-VS-ALL: Create binary datasets and train each label separately
                        self.console.print(f"  [dim]One-vs-all mode: Creating binary classifiers[/dim]")

                        binary_dir = Path(tmpdir) / f"binary_{category}"
                        binary_dir.mkdir(exist_ok=True)

                        binary_files = _create_binary_datasets_for_category(cat_data, category, binary_dir)

                        if not binary_files:
                            self.console.print(f"  [red]✗ No binary datasets created for {category}[/red]")
                            category_results[category] = {'error': 'no_binary_datasets', 'language_filtering_report': lang_report}
                            continue

                        # Filter binary files to only selected items if representative selection is active
                        if benchmark_onevsall_selection and category in benchmark_onevsall_selection:
                            selected_labels = set(benchmark_onevsall_selection[category])
                            filtered_binary_files = {k: v for k, v in binary_files.items() if k in selected_labels}
                            skipped_count = len(binary_files) - len(filtered_binary_files)
                            if skipped_count > 0:
                                self.console.print(f"  [dim]  Using representative selection: {len(filtered_binary_files)} of {len(binary_files)} values[/dim]")
                            binary_files = filtered_binary_files

                        self.console.print(f"  [dim]  Training {len(binary_files)} binary datasets[/dim]")

                        # Train each binary classifier
                        binary_results = {}
                        for label_name, binary_file in binary_files.items():
                            self.console.print(f"    [dim]Training binary: {label_name}[/dim]")

                            # Create meaningful display labels (like full training does)
                            positive_label_display = f"{label_name} IS"
                            negative_label_display = f"{label_name} IS NOT"

                            binary_train_params = train_params.copy()
                            # CRITICAL: Update global progress tracking with current values
                            current_submodel_index += 1
                            binary_train_params['global_current_model'] = current_submodel_index
                            binary_train_params['global_completed_epochs'] = global_completed_epochs
                            binary_train_params['input_file'] = str(binary_file)
                            binary_train_params['output_dir'] = str(model_output_dir / category / label_name)
                            binary_train_params['training_strategy'] = 'single-label'  # Binary classification
                            binary_train_params['label_column'] = 'label'
                            binary_train_params['category_name'] = label_name
                            # Use meaningful class names instead of 0/1
                            binary_train_params['class_names_override'] = [negative_label_display, positive_label_display]
                            binary_train_params['onevsall_display_labels'] = {
                                'positive': positive_label_display,
                                'negative': negative_label_display
                            }
                            if kept_languages:
                                binary_train_params['confirmed_languages'] = kept_languages
                            binary_train_params['language_filtering_report'] = lang_report

                            try:
                                binary_result = trainer.train(binary_train_params)
                                # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                                returned_epochs = binary_result.get('global_completed_epochs') if isinstance(binary_result, dict) else None
                                if returned_epochs is not None:
                                    global_completed_epochs = max(global_completed_epochs, returned_epochs)
                                self.logger.debug(f"[EPOCH_TRACK] After benchmark binary {label_name}: global_completed_epochs={global_completed_epochs}")
                                binary_results[label_name] = binary_result

                                bin_f1 = extract_numeric_metric(binary_result, ('f1_macro', 'best_f1_macro', 'f1'), default=0.0)
                                bin_acc = extract_numeric_metric(binary_result, ('accuracy', 'best_accuracy'), default=0.0)

                                if bin_f1 > 0 or bin_acc > 0:
                                    self.console.print(f"      ✓ {label_name}: F1={bin_f1:.3f}, Acc={bin_acc:.3f}")
                                else:
                                    self.console.print(f"      ✗ {label_name}: Training failed")
                            except Exception as e:
                                self.console.print(f"      ✗ {label_name}: Error - {str(e)}")
                                binary_results[label_name] = {'error': str(e)}

                        # Aggregate binary results for this category
                        successful_binary = [r for r in binary_results.values() if isinstance(r, dict) and 'error' not in r]
                        if successful_binary:
                            avg_f1 = sum(extract_numeric_metric(r, ('f1_macro', 'best_f1_macro', 'f1'), 0.0) for r in successful_binary) / len(successful_binary)
                            avg_acc = sum(extract_numeric_metric(r, ('accuracy', 'best_accuracy'), 0.0) for r in successful_binary) / len(successful_binary)
                            category_results[category] = {
                                'f1_macro': avg_f1,
                                'best_f1_macro': avg_f1,
                                'accuracy': avg_acc,
                                'best_accuracy': avg_acc,
                                'strategy': 'one-vs-all',
                                'binary_results': binary_results,
                                'models_trained': len(successful_binary),
                                'language_filtering_report': lang_report
                            }
                            self.console.print(f"  ✓ {category} (one-vs-all): Avg F1={avg_f1:.3f}, Avg Acc={avg_acc:.3f} ({len(successful_binary)} models)")
                        else:
                            category_results[category] = {'error': 'all_binary_failed', 'binary_results': binary_results, 'language_filtering_report': lang_report}
                            self.console.print(f"  ✗ {category}: All binary classifiers failed")

                    elif cat_strategy == 'multi-class':
                        # MULTI-CLASS: Convert to single-label and train one model
                        self.console.print(f"  [dim]Multi-class mode: Training single classifier[/dim]")

                        single_label_df = _convert_to_single_label(cat_data, category)

                        if single_label_df.empty:
                            self.console.print(f"  [red]✗ No valid data after single-label conversion for {category}[/red]")
                            category_results[category] = {'error': 'no_single_label_data', 'language_filtering_report': lang_report}
                            continue

                        cat_file = Path(tmpdir) / f"benchmark_{category}_singlelab.jsonl"
                        single_label_df.to_json(cat_file, orient='records', lines=True, force_ascii=False)

                        cat_train_params = train_params.copy()
                        # CRITICAL: Update global progress tracking with current values
                        current_submodel_index += 1
                        cat_train_params['global_current_model'] = current_submodel_index
                        cat_train_params['global_completed_epochs'] = global_completed_epochs
                        cat_train_params['input_file'] = str(cat_file)
                        cat_train_params['output_dir'] = str(model_output_dir / category)
                        cat_train_params['training_strategy'] = 'single-label'  # Multi-class uses single-label strategy
                        cat_train_params['label_column'] = 'label'
                        cat_train_params['category_name'] = category
                        if kept_languages:
                            cat_train_params['confirmed_languages'] = kept_languages
                        cat_train_params['language_filtering_report'] = lang_report

                        try:
                            cat_result = trainer.train(cat_train_params)
                            # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                            returned_epochs = cat_result.get('global_completed_epochs') if isinstance(cat_result, dict) else None
                            if returned_epochs is not None:
                                global_completed_epochs = max(global_completed_epochs, returned_epochs)
                            self.logger.debug(f"[EPOCH_TRACK] After benchmark multi-class {category}: global_completed_epochs={global_completed_epochs}")
                            cat_result['strategy'] = 'multi-class'
                            category_results[category] = cat_result

                            filtering_report = cat_result.get('language_filtering_report') or lang_report
                            cat_result['language_filtering_report'] = filtering_report
                            if filtering_report and filtering_report.get('filtered', False):
                                self.console.print(f"  [yellow][!] Language filtering applied for {category}:[/yellow]")
                                self.console.print(f"    - Kept: {', '.join(filtering_report.get('languages_kept', []))}")
                                self.console.print(f"    - Dropped: {', '.join(filtering_report.get('languages_dropped', []))}")

                            cat_f1 = extract_numeric_metric(cat_result, ('f1_macro', 'best_f1_macro', 'macro_f1', 'f1'), default=0.0)
                            cat_acc = extract_numeric_metric(cat_result, ('accuracy', 'best_accuracy'), default=0.0)

                            if cat_f1 > 0 or cat_acc > 0:
                                self.console.print(f"  ✓ {category} (multi-class): F1={cat_f1:.3f}, Acc={cat_acc:.3f}")
                            else:
                                self.console.print(f"  ✗ {category}: Training failed")

                        except Exception as e:
                            self.console.print(f"  ✗ {category}: Error - {str(e)}")
                            category_results[category] = {'error': str(e), 'language_filtering_report': lang_report}

                    else:
                        # MULTI-LABEL: Original behavior (keep as-is)
                        cat_file = Path(tmpdir) / f"benchmark_{category}.jsonl"
                        cat_data.to_json(cat_file, orient='records', lines=True, force_ascii=False)

                        cat_train_params = train_params.copy()
                        # CRITICAL: Update global progress tracking with current values
                        current_submodel_index += 1
                        cat_train_params['global_current_model'] = current_submodel_index
                        cat_train_params['global_completed_epochs'] = global_completed_epochs
                        cat_train_params['input_file'] = str(cat_file)
                        cat_train_params['output_dir'] = str(model_output_dir / category)
                        if kept_languages:
                            cat_train_params['confirmed_languages'] = kept_languages
                        cat_train_params['language_filtering_report'] = lang_report

                        try:
                            cat_result = trainer.train(cat_train_params)
                            # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                            returned_epochs = cat_result.get('global_completed_epochs') if isinstance(cat_result, dict) else None
                            if returned_epochs is not None:
                                global_completed_epochs = max(global_completed_epochs, returned_epochs)
                            self.logger.debug(f"[EPOCH_TRACK] After benchmark multi-label {category}: global_completed_epochs={global_completed_epochs}")
                            cat_result['strategy'] = 'multi-label'
                            category_results[category] = cat_result

                            filtering_report = cat_result.get('language_filtering_report') or lang_report
                            cat_result['language_filtering_report'] = filtering_report
                            if filtering_report and filtering_report.get('filtered', False):
                                self.console.print(f"  [yellow][!] Language filtering applied for {category}:[/yellow]")
                                self.console.print(f"    - Kept: {', '.join(filtering_report.get('languages_kept', []))}")
                                self.console.print(f"    - Dropped: {', '.join(filtering_report.get('languages_dropped', []))}")

                            cat_f1 = extract_numeric_metric(cat_result, ('f1_macro', 'best_f1_macro', 'macro_f1', 'f1'), default=0.0)
                            cat_acc = extract_numeric_metric(cat_result, ('accuracy', 'best_accuracy'), default=0.0)

                            if cat_f1 > 0 or cat_acc > 0:
                                self.console.print(f"  ✓ {category}: F1={cat_f1:.3f}, Acc={cat_acc:.3f}")
                            else:
                                self.console.print(f"  ✗ {category}: Training failed")

                        except Exception as e:
                            self.console.print(f"  ✗ {category}: Error - {str(e)}")
                            category_results[category] = {'error': str(e), 'language_filtering_report': lang_report}
                            continue

                for skipped_category, lang_report in skipped_categories.items():
                    category_results[skipped_category] = {
                        'error': 'insufficient_language_data',
                        'language_filtering_report': lang_report
                    }

                # Aggregate results across categories
                result = aggregate_benchmark_results(category_results, model_id)

                # Collect language filtering reports
                language_filtering_summary = {}
                for cat_name, cat_result in category_results.items():
                    if isinstance(cat_result, dict) and 'language_filtering_report' in cat_result:
                        filtering_report = cat_result['language_filtering_report']
                        if filtering_report.get('filtered', False):
                            language_filtering_summary[cat_name] = filtering_report

                # Add language filtering summary to result if any filtering occurred
                if language_filtering_summary:
                    result['language_filtering_summary'] = language_filtering_summary

                # Store aggregated result
                benchmark_results[model_id] = result

                # Extract aggregated metrics
                f1_score = extract_numeric_metric(
                    result,
                    ('f1_macro', 'best_f1_macro', 'macro_f1'),
                    default=0.0
                )
                accuracy = extract_numeric_metric(
                    result,
                    ('accuracy', 'best_accuracy'),
                    default=0.0
                )

                # Display overall results
                self.console.print(f"\n[green]✓ Training Complete[/green]")

                if result.get('successful_categories', 0) > 0:
                    self.console.print(f"  • Overall F1-Score: [bold green]{f1_score:.3f}[/bold green]")
                    self.console.print(f"  • Overall Accuracy: [bold green]{accuracy:.3f}[/bold green]")

                    if result.get('partial_success'):
                        self.console.print(f"  • [yellow][!]  Partial success: {result['successful_categories']}/{result['successful_categories'] + result['failed_categories']} categories trained[/yellow]")

                    if 'training_time' in result:
                        self.console.print(f"  • Time: [cyan]{result['training_time']:.1f}s[/cyan]")

                    # Display per-category details
                    if result.get('category_details'):
                        self.console.print(f"\n  [dim]Per-Category Scores:[/dim]")
                        for cat_detail in result['category_details']:
                            self.console.print(f"    • {cat_detail['category']}: F1={cat_detail['f1_macro']:.3f}, Acc={cat_detail['accuracy']:.3f}")
                else:
                    self.console.print(f"  • [red]All categories failed for this model[/red]")

            except Exception as e:
                self.console.print(f"\n[red][FAIL] Error during training: {str(e)}[/red]")
                # Add placeholder result with normalized keys
                benchmark_results[model_id] = {
                    'f1_macro': 0.0,          # Normalized key for compare_model_results
                    'accuracy': 0.0,          # Normalized key
                    'f1_0': 0.0,             # For binary classification
                    'f1_1': 0.0,             # For binary classification
                    'precision': 0.0,         # Overall precision
                    'recall': 0.0,           # Overall recall
                    'training_time': 0,
                    'error': str(e),
                    # Keep backward compatibility
                    'best_f1_macro': 0.0,
                    'best_accuracy': 0.0
                }

    # ======================== STEP 6: Display Results ========================
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    self.console.print("[bold cyan]         STEP 6: BENCHMARK RESULTS                           [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Display ranking methodology explanation
    self.console.print("[bold yellow]How Models Are Ranked:[/bold yellow]")
    self.console.print("\n[bold]Sophisticated Combined Metric System[/bold] (mirrors epoch selection):\n")
    self.console.print("  [cyan]1. Combined Score[/cyan] (Primary Criterion)")
    self.console.print("     • Binary Classification: [green]70% × F1_minority + 30% × F1_macro[/green]")
    self.console.print("       → Prioritizes minority class detection (e.g., detecting defects, fraud)")
    self.console.print("     • Multi-Class: [green]F1_macro[/green] (balanced across all classes)\n")

    self.console.print("  [cyan]2. Language Balance Penalty[/cyan] (for multilingual data)")
    self.console.print("     • Measures performance consistency across languages")
    self.console.print("     • Penalty = [yellow]min(CV × 0.2, 0.2)[/yellow] where CV = coefficient of variation")
    self.console.print("     • Example: Model with F1=90% (EN) + F1=30% (FR) → [red]penalized[/red]")
    self.console.print("     • Example: Model with F1=70% (EN) + F1=65% (FR) → [green]minimal penalty[/green]\n")

    self.console.print("  [cyan]3. Tiebreakers[/cyan]")
    self.console.print("     • [green]Accuracy[/green] (when combined scores equal)")
    self.console.print("     • [green]Training Time[/green] (faster is better when score + accuracy equal)\n")

    self.console.print("[dim]Tip: This ensures models are ranked the same way best epochs are selected during training[/dim]\n")

    # Check if we have multi-category results
    has_category_details = any('category_metrics' in result and result['category_metrics']
                               for result in benchmark_results.values())

    if has_category_details and selected_benchmark_categories:
        # Display detailed per-category results
        self.console.print("[bold]Overall Rankings:[/bold]\n")

        # Create comparison DataFrame with sophisticated ranking
        comparison_df = compare_model_results(benchmark_results, use_sophisticated_ranking=True)

        # Overall results table
        results_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, title="[bold]Ranked Results[/bold]", expand=True)
        results_table.add_column("Rank", style="yellow", width=6, no_wrap=True)
        results_table.add_column("Model", style="cyan", overflow="fold", ratio=1)
        results_table.add_column("Combined\nScore", style="bold green", no_wrap=True, justify="right")
        results_table.add_column("Avg F1", style="green", no_wrap=True, justify="right")
        results_table.add_column("Avg Acc", style="green", no_wrap=True, justify="right")
        results_table.add_column("Time (s)", style="blue", no_wrap=True, justify="right")

        for _, row in comparison_df.iterrows():
            # Add emoji for top 3
            if row['rank'] == 1:
                rank_str = "#1"
            elif row['rank'] == 2:
                rank_str = "#2"
            elif row['rank'] == 3:
                rank_str = "#3"
            else:
                rank_str = f"   {row['rank']}"

            # Highlight combined score if different from f1_macro
            combined_score = row.get('combined_score', row['f1_macro'])
            if abs(combined_score - row['f1_macro']) > 0.001:
                combined_str = f"[bold]{combined_score:.3f}[/bold]"
            else:
                combined_str = f"{combined_score:.3f}"

            results_table.add_row(
                rank_str,
                row['model'],
                combined_str,
                f"{row['f1_macro']:.3f}",
                f"{row['accuracy']:.3f}",
                f"{row['training_time']:.1f}"
            )

        self.console.print(results_table)

        # Per-category breakdown
        self.console.print(f"\n[bold]Performance by Category:[/bold]\n")

        for category in selected_benchmark_categories:
            self.console.print(f"[bold cyan]Category: {category}[/bold cyan]")

            cat_table = Table(show_header=True, header_style="bold yellow", border_style="blue", box=box.SIMPLE, expand=True)
            cat_table.add_column("Model", style="cyan", ratio=1, overflow="fold")
            cat_table.add_column("F1-Score", style="green", no_wrap=True)
            cat_table.add_column("Accuracy", style="green", no_wrap=True)
            cat_table.add_column("Precision", style="blue", no_wrap=True)
            cat_table.add_column("Recall", style="blue", no_wrap=True)

            # Collect scores for this category across all models
            category_scores = []
            for model_id, result in benchmark_results.items():
                if 'category_metrics' in result:
                    # Find the model that corresponds to this category
                    for model_name, metrics in result['category_metrics'].items():
                        # Check if this model is for the current category
                        # Model names typically include category: "sentiment_simple_EN" or similar
                        if category.lower() in model_name.lower():
                            category_scores.append({
                                'model': model_id,
                                'f1': metrics.get('f1_macro', 0),
                                'accuracy': metrics.get('accuracy', 0),
                                'precision': metrics.get('precision', 0),
                                'recall': metrics.get('recall', 0)
                            })
                            break

            # Sort by F1 score
            category_scores.sort(key=lambda x: x['f1'], reverse=True)

            # Display
            for score_data in category_scores:
                cat_table.add_row(
                    score_data['model'],
                    f"{score_data['f1']:.3f}",
                    f"{score_data['accuracy']:.3f}",
                    f"{score_data['precision']:.3f}",
                    f"{score_data['recall']:.3f}"
                )

            self.console.print(cat_table)
            self.console.print()  # Empty line between categories

    else:
        # Simple display for single-category or no details available
        # Create comparison DataFrame with sophisticated ranking
        comparison_df = compare_model_results(benchmark_results, use_sophisticated_ranking=True)

        # Display results with combined score
        results_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, title="[bold]Ranked Results[/bold]", expand=True)
        results_table.add_column("Rank", style="yellow", width=6, no_wrap=True)
        results_table.add_column("Model", style="cyan", overflow="fold")
        results_table.add_column("Combined\nScore", style="bold green", no_wrap=True, justify="right")
        results_table.add_column("F1-Macro", style="green", no_wrap=True, justify="right")
        results_table.add_column("Accuracy", style="green", no_wrap=True, justify="right")
        results_table.add_column("Time (s)", style="blue", no_wrap=True, justify="right")

        for _, row in comparison_df.iterrows():
            # Add emoji for top 3
            if row['rank'] == 1:
                rank_str = "#1"
            elif row['rank'] == 2:
                rank_str = "#2"
            elif row['rank'] == 3:
                rank_str = "#3"
            else:
                rank_str = f"   {row['rank']}"

            # Highlight combined score if different from f1_macro
            combined_score = row.get('combined_score', row['f1_macro'])
            if abs(combined_score - row['f1_macro']) > 0.001:
                # Different → show in bold
                combined_str = f"[bold]{combined_score:.3f}[/bold]"
            else:
                combined_str = f"{combined_score:.3f}"

            results_table.add_row(
                rank_str,
                row['model'],
                combined_str,
                f"{row['f1_macro']:.3f}",
                f"{row['accuracy']:.3f}",
                f"{row['training_time']:.1f}"
            )

        self.console.print(results_table)

        # Display ranking explanations for top 3 models
        self.console.print("\n[bold cyan]Top 3 Models - Ranking Details:[/bold cyan]\n")
        for _, row in comparison_df.head(3).iterrows():
            emoji = "#1" if row['rank'] == 1 else "#2" if row['rank'] == 2 else "#3"
            self.console.print(f"{emoji} [bold]{row['model']}[/bold]")
            if 'ranking_explanation' in row and row['ranking_explanation']:
                self.console.print(f"   → {row['ranking_explanation']}")

            # Show class-specific F1 if binary classification
            if 'f1_class_1' in row and row['f1_class_1'] > 0:
                self.console.print(f"   → F1_class_0: {row['f1_class_0']:.3f} | F1_class_1: {row['f1_class_1']:.3f}")

            # Show language penalty if applicable
            if 'language_balance_penalty' in row and row['language_balance_penalty'] > 0:
                self.console.print(f"   → Language imbalance penalty: [yellow]-{row['language_balance_penalty']:.1%}[/yellow]")

            self.console.print()

    # ======================== Consolidate Session CSVs ========================
    # Create consolidated CSV files at session root
    try:
        from llm_tool.utils.benchmark_utils import consolidate_session_csvs

        # Session directory is in logs/training_arena/{session_id}/training_metrics
        if session_manager and getattr(session_manager, 'session_dir', None):
            session_dir = session_manager.session_dir / "training_metrics"
        else:
            session_dir = get_training_metrics_dir(benchmark_session_id)

        if session_dir.exists():
            self.console.print("\n[bold cyan]Consolidating session metrics...[/bold cyan]")
            consolidated_files = consolidate_session_csvs(session_dir, benchmark_session_id)

            if consolidated_files:
                self.console.print("[green]Created consolidated CSV files:[/green]")
                if 'training' in consolidated_files:
                    self.console.print(f"  • Training metrics: {consolidated_files['training'].name}")
                if 'best' in consolidated_files:
                    self.console.print(f"  • Best models: {consolidated_files['best'].name}")
    except Exception as e:
        self.console.print(f"[yellow][!] Warning: Could not consolidate CSVs: {e}[/yellow]")

    # ======================== STEP 7: Final Choice ========================
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    self.console.print("[bold cyan]         STEP 7: Final Model Selection                       [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    self.console.print("[bold]Based on benchmark results, you can:[/bold]")
    self.console.print("  [cyan]1.[/cyan] [bold]Use top-ranked model(s)[/bold] (recommended)")
    self.console.print("  [cyan]2.[/cyan] Manually select model(s)")
    self.console.print("  [cyan]3.[/cyan] Stop here (benchmark only, no full training)\n")

    choice = Prompt.ask(
        "[bold yellow]What would you like to do?[/bold yellow]",
        choices=["1", "2", "3", "top", "manual", "stop"],
        default="1"
    )

    if choice in ["3", "stop"]:
        self.console.print("\n[green]✓ Benchmark complete. Exiting without full training.[/green]")

        # ============================================================
        # CRITICAL: Update metadata with final benchmark results
        # This ensures the benchmark-only session is fully tracked
        # ============================================================
        try:
            self.logger.info(" Updating benchmark metadata with final results...")

            # Extract best model from results
            best_model = comparison_df.iloc[0]['model'] if not comparison_df.empty else None
            best_f1 = comparison_df.iloc[0]['f1_macro'] if not comparison_df.empty else None

            # Build final benchmark model config
            final_benchmark_config = {
                'training_mode': 'benchmark',
                'benchmark_enabled': True,
                'selected_models': selected_models_benchmark if not train_by_language else list(all_models_to_test),
                'models_by_language': models_by_language_benchmark if train_by_language else {},
                'train_by_language': train_by_language,
                'benchmark_categories': selected_benchmark_categories,
                'benchmark_epochs': benchmark_epochs,
                'reinforced_learning_enabled': enable_benchmark_rl,
                'epochs': benchmark_epochs,
                'batch_size': _get_optimal_batch_size(best_model),
                'learning_rate': 2e-5,
                'actual_models_trained': list(benchmark_results.keys()),
                'best_model_from_benchmark': best_model,
                'benchmark_rankings': comparison_df.to_dict('records') if not comparison_df.empty else []
            }

            # Save complete metadata
            final_metadata_path = self._save_training_metadata(
                bundle=bundle,
                mode='benchmark',
                model_config=final_benchmark_config,
                execution_status={
                    'status': 'benchmark_completed_no_training',
                    'started_at': datetime.now().isoformat(),
                    'completed_at': datetime.now().isoformat(),
                    'models_trained': list(benchmark_results.keys()),
                    'best_model': best_model,
                    'best_f1': best_f1,
                    'benchmark_phase': 'completed',
                    'user_choice': 'stop_after_benchmark'
                },
                session_id=benchmark_session_id,
                training_context={
                    'benchmark_mode': True,
                    'benchmark_results': {
                        model_id: {
                            'best_f1_macro': result.get('best_f1_macro', 0),
                            'accuracy': result.get('accuracy', 0),
                            'training_time': result.get('training_time', 0)
                        }
                        for model_id, result in benchmark_results.items()
                    },
                    'user_choices': {
                        'enable_benchmark': True,
                        'stopped_after_benchmark': True,
                        'num_models_tested': len(benchmark_results)
                    }
                }
            )
            self.console.print(f"[dim]Final metadata saved: {final_metadata_path.name}[/dim]")
            self.logger.info(f"✓ Final benchmark metadata saved: {final_metadata_path}")

        except Exception as e:
            self.logger.error(f"Failed to save final benchmark metadata: {e}")
            self.console.print(f"[yellow][!] Warning: Could not save final metadata: {e}[/yellow]")
            # Continue to summaries even if metadata fails

        # Generate comprehensive summary files for benchmark-only session
        try:
            from llm_tool.utils.training_summary_generator import generate_training_summaries

            self.console.print("\n[bold cyan]Generating Comprehensive Benchmark Summaries...[/bold cyan]")
            csv_path, jsonl_path = generate_training_summaries(benchmark_session_id)

            self.console.print("[green]Benchmark summaries generated successfully:[/green]")
            self.console.print(f"  • CSV Summary: [cyan]{csv_path.name}[/cyan]")
            self.console.print(f"  • JSONL Summary: [cyan]{jsonl_path.name}[/cyan]")
            self.console.print(f"\n[dim]Full paths:[/dim]")
            self.console.print(f"  • {csv_path}")
            self.console.print(f"  • {jsonl_path}")

        except Exception as e:
            self.logger.error(f"Failed to generate benchmark summaries: {e}")
            self.console.print(f"[yellow][!] Could not generate comprehensive summaries: {e}[/yellow]")

        return None

    # Select final models
    final_model_name = None
    final_models_by_language = None

    if choice in ["1", "top"]:
        self.console.print("\n[bold green]✓ Using top-ranked model(s)[/bold green]")

        if train_by_language:
            # Select best model per language
            final_models_by_language = {}
            for lang in languages:
                lang_models = models_by_language_benchmark[lang]
                # Find best model for this language
                lang_results = {m: benchmark_results[m] for m in lang_models}
                best_model = max(lang_results, key=lambda m: lang_results[m].get('best_f1_macro', 0))
                final_models_by_language[lang] = best_model
                self.console.print(f"  • {lang}: [cyan]{best_model}[/cyan] (F1: {benchmark_results[best_model]['best_f1_macro']:.3f})")
        else:
            # Take best model overall
            final_model_name = comparison_df.iloc[0]['model']
            self.console.print(f"  • Selected: [cyan]{final_model_name}[/cyan] (F1: {comparison_df.iloc[0]['f1_macro']:.3f})")

    elif choice in ["2", "manual"]:
        self.console.print("\n[bold]Manual Selection:[/bold]")

        if train_by_language:
            final_models_by_language = {}
            for lang in sorted(languages):
                lang_models = models_by_language_benchmark[lang]

                self.console.print(f"\n[yellow]Models for {lang}:[/yellow]")
                for idx, model in enumerate(lang_models, 1):
                    result = benchmark_results[model]
                    self.console.print(f"  {idx}. {model} (F1: {result.get('best_f1_macro', 0):.3f})")

                choice_idx = IntPrompt.ask(f"Select model for {lang}", default=1)
                idx_adj = choice_idx - 1
                if 0 <= idx_adj < len(lang_models):
                    final_models_by_language[lang] = lang_models[idx_adj]
                else:
                    final_models_by_language[lang] = lang_models[0]

                self.console.print(f"  [green]✓ {lang}: {final_models_by_language[lang]}[/green]")
        else:
            self.console.print("\n[yellow]Available models:[/yellow]")
            for idx, model in enumerate(selected_models_benchmark, 1):
                result = benchmark_results[model]
                self.console.print(f"  {idx}. {model} (F1: {result.get('best_f1_macro', 0):.3f})")

            choice_idx = IntPrompt.ask("Select model", default=1)
            idx_adj = choice_idx - 1
            if 0 <= idx_adj < len(selected_models_benchmark):
                final_model_name = selected_models_benchmark[idx_adj]
            else:
                final_model_name = selected_models_benchmark[0]

            self.console.print(f"  [green]✓ Selected: {final_model_name}[/green]")

    # Return results
    result = {}
    if final_model_name:
        result['model_name'] = final_model_name
    if final_models_by_language:
        result['models_by_language'] = final_models_by_language
        result['train_by_language'] = True

    return result

def _training_studio_render_bundle_summary(self, bundle: TrainingDataBundle) -> None:
    table = Table(title="Dataset Summary", border_style="green", expand=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    # Use training_approach from metadata if available, otherwise fallback to bundle.strategy
    training_approach = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None
    strategy_display = training_approach if training_approach else bundle.strategy
    table.add_row("Strategy", strategy_display)
    table.add_row("Primary file", str(bundle.primary_file) if bundle.primary_file else "—")
    table.add_row("Text column", bundle.text_column)
    table.add_row("Label column", bundle.label_column)
    table.add_row("Training files", str(len(bundle.training_files)))

    if bundle.metadata.get("label_distribution"):
        distribution = ", ".join(f"{k}: {v}" for k, v in bundle.metadata["label_distribution"].items())
        table.add_row("Label distribution", distribution)
    if bundle.metadata.get("categories"):
        table.add_row("Categories", ", ".join(bundle.metadata["categories"]))
    if bundle.metadata.get("analysis"):
        analysis = bundle.metadata["analysis"]
        table.add_row("Annotated rows", str(analysis.get("annotated_rows", "n/a")))

    self.console.print(table)

def _log_training_data_distributions(self, bundle: TrainingDataBundle, training_context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log comprehensive distribution information for ALL training datasets created.

    This function is called AFTER training/benchmark completion and logs:
    - ALL datasets created (multiclass, onevsall, multilabel)
    - What was used for benchmark vs normal training
    - Train/val/test splits
    - Label distributions
    - Language distributions
    - Imbalance warnings
    - Complete training context (mode, models, results)

    Args:
        bundle: TrainingDataBundle containing all created dataset files
        training_context: Optional dict with training/benchmark information:
            - mode: Training mode (quick, benchmark, custom, distributed)
            - training_result: Results from training
            - runtime_params: Runtime parameters used
            - models_trained: List of models that were trained
            - benchmark_results: Results if benchmark mode was used
    """
    import json

    # Defensive check: Ensure session attributes are initialized
    if not hasattr(self, 'current_session_manager') or not self.current_session_manager:
        self.logger.warning("_log_training_data_distributions called without session_manager initialized. Skipping logging.")
        return
    if not hasattr(self, 'current_session_id') or not self.current_session_id:
        self.logger.warning("_log_training_data_distributions called without session_id initialized. Skipping logging.")
        return

    if training_context:
        self.console.print(f"\n[bold cyan]Generating comprehensive training session report...[/bold cyan]")
        self.console.print(f"[dim]Mode: {training_context.get('mode', 'unknown')} | Models: {len(training_context.get('models_trained', []))}[/dim]")
    else:
        self.console.print("\n[dim]Logging comprehensive training data distributions...[/dim]")

    # Collect all dataset files (primary + training_files)
    all_files = []
    if bundle.primary_file:
        # Use descriptive name based on strategy
        primary_name = f"multilabel_combined" if bundle.strategy == 'multi-label' else "combined_dataset"
        all_files.append((primary_name, bundle.primary_file))
    for key, path in bundle.training_files.items():
        all_files.append((key, path))

    if not all_files:
        self.logger.warning("No training data files found in bundle to log")
        return

    # Log distribution for each dataset file
    datasets_logged = 0
    for dataset_name, dataset_path in all_files:
        try:
            # Load JSONL file
            records = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            if not records:
                self.logger.warning(f"Dataset {dataset_name} is empty: {dataset_path}")
                continue

            # Build comprehensive metadata including training context
            dataset_metadata = {
                'file_path': str(dataset_path),
                'file_size_mb': round(dataset_path.stat().st_size / (1024 * 1024), 2),
                'num_records': len(records),
                'strategy': bundle.strategy,
                'training_approach': bundle.metadata.get('training_approach', ''),
                'text_column': bundle.text_column,
                'label_column': bundle.label_column,
                'source_file': bundle.metadata.get('source_file', ''),
                'categories': bundle.metadata.get('categories', []),
                'confirmed_languages': bundle.metadata.get('confirmed_languages', []),
                'split_config': bundle.metadata.get('split_config', {}),
            }

            # Add training context if provided (mode, benchmark info, etc.)
            if training_context:
                dataset_metadata.update({
                    'training_mode': training_context.get('mode'),
                    'models_trained': training_context.get('models_trained', []),
                    'was_used_in_benchmark': training_context.get('mode') == 'benchmark',
                    'benchmark_results': training_context.get('benchmark_results') if training_context.get('mode') == 'benchmark' else None,
                })

            # Log distribution with complete metadata
            self.current_session_manager.log_distribution(
                dataset_name=dataset_name,
                train_samples=records,  # All samples (split happens during training)
                val_samples=[],  # Splitting happens during training
                test_samples=[],
                label_key=dataset_name,
                metadata=dataset_metadata
            )
            datasets_logged += 1

        except Exception as e:
            self.logger.warning(f"Could not log distribution for {dataset_name}: {e}")
            continue

    # Finalize session and generate comprehensive reports
    try:
        warnings_count, datasets_with_warnings = self.current_session_manager.finalize(training_context=training_context)

        # Display summary to user
        if training_context:
            self.console.print(f"\n[green]✓ Complete training session report generated:[/green]")
            self.console.print(f"  • [cyan]Session ID:[/cyan] {self.current_session_id}")
            self.console.print(f"  • [cyan]Training Mode:[/cyan] {training_context.get('mode', 'unknown')}")
            self.console.print(f"  • [cyan]Datasets logged:[/cyan] {datasets_logged}")
            self.console.print(f"  • [cyan]Models trained:[/cyan] {len(training_context.get('models_trained', []))}")
            if training_context.get('mode') == 'benchmark':
                self.console.print(f"  • [cyan]Benchmark:[/cyan] Results included in reports")
        else:
            self.console.print(f"\n[green]✓ Training data distribution reports generated:[/green]")
            self.console.print(f"  • [cyan]Session ID:[/cyan] {self.current_session_id}")
            self.console.print(f"  • [cyan]Datasets logged:[/cyan] {datasets_logged}")

        self.console.print(f"\n  [cyan]Reports:[/cyan]")
        self.console.print(f"     - Model Catalog:      {self.current_session_manager.training_data_logs_dir / 'model_catalog.csv'} ← ALL models with full details")
        self.console.print(f"     - Session Summary:    {self.current_session_manager.session_dir / 'SESSION_SUMMARY.txt'} ← Complete overview")
        self.console.print(f"     - Quick overview:     {self.current_session_manager.training_data_logs_dir / 'quick_summary.csv'}")
        self.console.print(f"     - Detailed breakdown: {self.current_session_manager.training_data_logs_dir / 'split_summary.csv'}")
        self.console.print(f"     - Complete data:      {self.current_session_manager.training_data_logs_dir / 'distribution_report.json'}")

        if training_context:
            self.console.print(f"\n  [dim]Tip: Reports include complete training context: mode, models trained, and benchmark results.[/dim]")
        else:
            self.console.print(f"\n  [dim]Tip: Note: Data is currently PRE-SPLIT. The train/val/test split\n"
                             f"     will be applied during model training according to your configuration.[/dim]")

        if warnings_count > 0:
            self.console.print(f"\n[yellow][!]  {warnings_count} validation warning(s) detected across {datasets_with_warnings} dataset(s)[/yellow]")
            self.console.print(f"[dim]  View details in: {self.current_session_manager.warnings_log}[/dim]")
        else:
            self.console.print(f"\n[green]✓ All data validation checks passed[/green]")

    except Exception as e:
        self.logger.warning(f"Could not finalize training data session: {e}")
        self.console.print(f"[yellow][!] Could not generate final reports: {e}[/yellow]")

def _configure_data_splits(self, keys_to_train: List[str], all_keys_values: Dict[str, set],
                           training_approach: str, key_strategies: Dict[str, str],
                           total_samples: int) -> Optional[Dict[str, Any]]:
    """
    Configure train/test/validation split ratios.

    Args:
        total_samples: Total number of samples in the dataset

    Returns:
        split_config dict or None if user cancels
    """
    from rich.prompt import Prompt, Confirm, FloatPrompt
    from rich.table import Table
    from rich import box

    self.console.print("\n[bold]Data Split Configuration[/bold]\n")
    self.console.print("[dim]Configure how your data will be split for training, validation, and testing.[/dim]\n")

    # Tableau explicatif
    split_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
    split_table.add_column("Set", style="cyan bold", no_wrap=True)
    split_table.add_column("Purpose", style="white", ratio=1, overflow="fold")

    split_table.add_row(
        "Training",
        "Used to train the model (learn patterns from data)"
    )
    split_table.add_row(
        "Validation",
        "Used DURING training to:\n"
        "  • Monitor performance at each epoch\n"
        "  • Select best model checkpoint\n"
        "  • Enable early stopping\n"
        "  • Activate reinforced learning if needed"
    )
    split_table.add_row(
        "Test (Optional)",
        "Reserved for FINAL evaluation AFTER training:\n"
        "  • Provides unbiased performance metrics\n"
        "  • Never used during training\n"
        "  • Only evaluated once at the very end"
    )

    self.console.print(split_table)
    self.console.print()

    # Dataset size information
    self.console.print(f"[bold]Dataset Size:[/bold] {total_samples:,} samples\n")

    # Question 1: Use separate test set for final evaluation?
    # Provide recommendation based on dataset size
    if total_samples < 1000:
        self.console.print("[yellow][!] With fewer than 1,000 samples, it's recommended to skip the separate test set.[/yellow]")
        self.console.print("[dim]   Reason: You need as much data as possible for training and validation.[/dim]\n")
        use_test_set_default = False
    elif total_samples < 5000:
        self.console.print("[dim]Tip: With your dataset size, a separate test set is optional but not critical.[/dim]\n")
        use_test_set_default = False
    else:
        self.console.print("[dim]Your dataset is large enough to benefit from a separate test set.[/dim]\n")
        use_test_set_default = False

    use_test_set = Confirm.ask(
        "[bold yellow]Keep a separate test set for final evaluation?[/bold yellow]",
        default=use_test_set_default
    )

    self.console.print()

    # Question: Uniform or custom splits?
    self.console.print("\n[bold]Split Mode:[/bold]\n")
    self.console.print("  • [cyan]uniform[/cyan]: Same ratios for all keys/values")
    self.console.print("  • [cyan]custom[/cyan]:  Different ratios per key or value\n")

    split_mode = Prompt.ask(
        "[bold yellow]Split mode[/bold yellow]",
        choices=["uniform", "custom", "u", "c", "back"],
        default="uniform"
    )

    if split_mode == "back":
        return None

    # Normalize shortcuts
    if split_mode == "u":
        split_mode = "uniform"
    elif split_mode == "c":
        split_mode = "custom"

    split_config = {
        'use_test_set': use_test_set,
        'mode': split_mode
    }

    # UNIFORM MODE
    if split_mode == "uniform":
        split_config['uniform'] = self._configure_uniform_splits(use_test_set)
        if split_config['uniform'] is None:
            return None

    # CUSTOM MODE
    else:
        custom_config = self._configure_custom_splits(
            keys_to_train=keys_to_train,
            all_keys_values=all_keys_values,
            training_approach=training_approach,
            key_strategies=key_strategies,
            use_test_set=use_test_set
        )

        if custom_config is None:
            return None

        split_config.update(custom_config)

    # Display summary
    self._display_split_summary(split_config, keys_to_train, all_keys_values, key_strategies)

    return split_config

def _configure_uniform_splits(self, use_test_set: bool) -> Optional[Dict[str, float]]:
    """Configure uniform split ratios.

    Args:
        use_test_set: If True, configure train/val/test. If False, configure train/val only.
    """
    from rich.prompt import FloatPrompt

    if use_test_set:
        self.console.print("\n[bold]Configure Split Ratios (Train / Validation / Test)[/bold]\n")
        self.console.print("[dim]Ratios must sum to 1.0[/dim]\n")

        train_ratio = FloatPrompt.ask("  Training ratio", default=0.7)
        # Calculate remaining ratio for val + test
        remaining_ratio = round(1.0 - train_ratio, 10)
        # Default: split remaining evenly between val and test (but favor validation slightly)
        default_val = round(min(0.2, remaining_ratio * 0.67), 10)
        default_test = round(remaining_ratio - default_val, 10)

        validation_ratio = FloatPrompt.ask("  Validation ratio", default=default_val)
        # Update test default based on what's left
        remaining_for_test = round(1.0 - train_ratio - validation_ratio, 10)
        test_ratio = FloatPrompt.ask("  Test ratio", default=max(0.0, remaining_for_test))

    else:
        self.console.print("\n[bold]Configure Split Ratios (Train / Validation)[/bold]\n")
        self.console.print("[dim]Ratios must sum to 1.0. Validation will be used for training evaluation.[/dim]\n")

        train_ratio = FloatPrompt.ask("  Training ratio", default=0.8)
        # Calculate default validation as remaining ratio
        default_validation = round(1.0 - train_ratio, 10)
        validation_ratio = FloatPrompt.ask("  Validation ratio", default=default_validation)
        test_ratio = 0.0

    # Validate and normalize
    try:
        train_ratio, validation_ratio, test_ratio = self._validate_split_ratios(
            train_ratio, validation_ratio, test_ratio
        )
    except ValueError as e:
        self.console.print(f"[red]Error: {e}[/red]")
        return None

    return {
        'train_ratio': train_ratio,
        'validation_ratio': validation_ratio,
        'test_ratio': test_ratio
    }

def _configure_custom_splits(self, keys_to_train: List[str], all_keys_values: Dict[str, set],
                              training_approach: str, key_strategies: Dict[str, str],
                              use_test_set: bool) -> Optional[Dict[str, Any]]:
    """Configure custom split ratios per key or value.

    Args:
        use_test_set: If True, configure train/val/test. If False, configure train/val only.
    """
    from rich.prompt import Confirm, FloatPrompt

    custom_config = {}

    # Configure defaults first
    self.console.print("\n[bold]Default Ratios[/bold]")
    self.console.print("[dim]Applied to keys/values not configured below[/dim]\n")

    if use_test_set:
        default_train = FloatPrompt.ask("  Default train ratio", default=0.7)
        # Calculate remaining for val + test
        remaining = 1.0 - default_train
        default_val_calc = min(0.2, remaining * 0.67)
        default_test_calc = remaining - default_val_calc

        default_validation = FloatPrompt.ask("  Default validation ratio", default=default_val_calc)
        remaining_for_test = 1.0 - default_train - default_validation
        default_test = FloatPrompt.ask("  Default test ratio", default=max(0.0, remaining_for_test))
    else:
        default_train = FloatPrompt.ask("  Default train ratio", default=0.8)
        default_val_calc = 1.0 - default_train
        default_validation = FloatPrompt.ask("  Default validation ratio", default=default_val_calc)
        default_test = 0.0

    # Validate defaults
    try:
        default_train, default_validation, default_test = self._validate_split_ratios(
            default_train, default_validation, default_test
        )
    except ValueError as e:
        self.console.print(f"[red]Error in defaults: {e}[/red]")
        return None

    custom_config['defaults'] = {
        'train_ratio': default_train,
        'validation_ratio': default_validation,
        'test_ratio': default_test
    }

    # Determine if we configure by key or by value
    if training_approach == "multi-class":
        # Configure by key
        custom_config['custom_by_key'] = self._configure_custom_by_key(
            keys_to_train, all_keys_values, use_test_set,
            default_train, default_validation, default_test
        )

    elif training_approach == "one-vs-all":
        # Configure by value
        custom_config['custom_by_value'] = self._configure_custom_by_value(
            keys_to_train, all_keys_values, use_test_set,
            default_train, default_validation, default_test
        )

    elif training_approach in ["hybrid", "custom"]:
        # Mix: some keys, some values
        custom_by_key = {}
        custom_by_value = {}

        for key in keys_to_train:
            strategy = key_strategies.get(key, 'multi-class')

            if strategy == 'multi-class':
                # Configure this key
                self.console.print(f"\n[bold cyan]{key}[/bold cyan] ([green]multi-class[/green])")
                customize = Confirm.ask(f"  Customize split for '{key}'?", default=False)

                if customize:
                    config = self._ask_split_ratios(use_test_set, default_train, default_validation, default_test)
                    if config:
                        custom_by_key[key] = config
                        self.console.print(f"  [green]✓ {key}: {config['train_ratio']:.1%} / {config['validation_ratio']:.1%} / {config['test_ratio']:.1%}[/green]")
                    else:
                        self.console.print(f"  [dim]Using defaults[/dim]")
                else:
                    self.console.print(f"  [dim]Using defaults[/dim]")

            else:  # one-vs-all
                # Configure values for this key
                self.console.print(f"\n[bold yellow]{key}[/bold yellow] ([yellow]one-vs-all[/yellow])")
                customize = Confirm.ask(f"  Customize splits for values in '{key}'?", default=False)

                if customize:
                    values = sorted(all_keys_values[key])
                    for value in values:
                        full_name = f"{key}_{value}"

                        customize_value = Confirm.ask(f"    Customize '{value}'?", default=False)

                        if customize_value:
                            config = self._ask_split_ratios(use_test_set, default_train, default_validation, default_test)
                            if config:
                                custom_by_value[full_name] = config
                                self.console.print(f"    [green]✓ {value}: {config['train_ratio']:.1%} / {config['validation_ratio']:.1%} / {config['test_ratio']:.1%}[/green]")
                            else:
                                self.console.print(f"    [dim]Using defaults[/dim]")
                        else:
                            self.console.print(f"    [dim]Using defaults[/dim]")

        if custom_by_key:
            custom_config['custom_by_key'] = custom_by_key
        if custom_by_value:
            custom_config['custom_by_value'] = custom_by_value

    return custom_config

def _configure_custom_by_key(self, keys_to_train: List[str], all_keys_values: Dict[str, set],
                              use_test_set: bool, default_train: float,
                              default_validation: float, default_test: float) -> Dict[str, Dict[str, float]]:
    """Configure custom splits per key.

    Args:
        use_test_set: If True, configure train/val/test. If False, configure train/val only.
    """
    from rich.prompt import Confirm

    custom_by_key = {}

    self.console.print("\n[bold cyan] Custom Configuration (per key)[/bold cyan]\n")

    for key in keys_to_train:
        num_values = len(all_keys_values[key])
        self.console.print(f"[bold]{key}[/bold] ({num_values} values)")

        customize = Confirm.ask(f"  Customize split for '{key}'?", default=False)

        if customize:
            config = self._ask_split_ratios(use_test_set, default_train, default_validation, default_test)
            if config:
                custom_by_key[key] = config
                self.console.print(f"  [green]✓ {key}: {config['train_ratio']:.1%} / {config['validation_ratio']:.1%} / {config['test_ratio']:.1%}[/green]")
            else:
                self.console.print(f"  [dim]Using defaults[/dim]")
        else:
            self.console.print(f"  [dim]Using defaults[/dim]")

        self.console.print()

    return custom_by_key

def _configure_custom_by_value(self, keys_to_train: List[str], all_keys_values: Dict[str, set],
                                use_test_set: bool, default_train: float,
                                default_validation: float, default_test: float) -> Dict[str, Dict[str, float]]:
    """Configure custom splits per value.

    Args:
        use_test_set: If True, configure train/val/test. If False, configure train/val only.
    """
    from rich.prompt import Confirm

    custom_by_value = {}

    self.console.print("\n[bold yellow] Custom Configuration (per value)[/bold yellow]\n")

    for key in keys_to_train:
        values = sorted(all_keys_values[key])
        self.console.print(f"[bold cyan]{key}[/bold cyan] ({len(values)} values)")

        customize_key = Confirm.ask(f"  Customize splits for values in '{key}'?", default=False)

        if customize_key:
            for value in values:
                full_name = f"{key}_{value}"

                customize_value = Confirm.ask(f"    Customize '{value}'?", default=False)

                if customize_value:
                    config = self._ask_split_ratios(use_test_set, default_train, default_validation, default_test)
                    if config:
                        custom_by_value[full_name] = config
                        self.console.print(f"    [green]✓ {value}: {config['train_ratio']:.1%} / {config['validation_ratio']:.1%} / {config['test_ratio']:.1%}[/green]")
                    else:
                        self.console.print(f"    [dim]Using defaults[/dim]")
                else:
                    self.console.print(f"    [dim]Using defaults[/dim]")

        self.console.print()

    return custom_by_value

def _ask_split_ratios(self, use_test_set: bool, default_train: float,
                      default_validation: float, default_test: float) -> Optional[Dict[str, float]]:
    """Ask for split ratios and validate them.

    Args:
        use_test_set: If True, ask for train/val/test. If False, ask for train/val only.
    """
    from rich.prompt import FloatPrompt

    try:
        train = FloatPrompt.ask("      Train ratio", default=default_train)

        # Calculate dynamic default for validation based on entered train ratio
        remaining = round(1.0 - train, 10)
        if use_test_set:
            # Split remaining between val and test
            dynamic_val_default = round(min(default_validation, remaining * 0.67), 10)
        else:
            # All remaining goes to validation
            dynamic_val_default = remaining

        validation = FloatPrompt.ask("      Validation ratio", default=dynamic_val_default)

        if use_test_set:
            # Calculate remaining for test
            remaining_for_test = round(1.0 - train - validation, 10)
            test = FloatPrompt.ask("      Test ratio", default=max(0.0, remaining_for_test))
        else:
            test = 0.0

        # Validate
        train, validation, test = self._validate_split_ratios(train, validation, test)

        return {
            'train_ratio': train,
            'validation_ratio': validation,
            'test_ratio': test
        }

    except ValueError as e:
        self.console.print(f"      [red]Error: {e}[/red]")
        return None

def _validate_labels_before_file_creation(
    self,
    csv_path: str,
    text_column: str,
    annotation_column: str,
    keys_to_train: List[str],
    key_strategies: Dict[str, str],
    min_samples: int = 2
) -> Tuple[Optional[List[str]], bool]:
    """
    Validate ALL labels BEFORE creating training files.
    Detects insufficient labels for ALL modes (multiclass, one-vs-all, hybrid/custom).

    Args:
        csv_path: Path to annotated CSV file
        text_column: Column with text data
        annotation_column: Column with JSON annotations
        keys_to_train: List of annotation keys to include
        key_strategies: Dict mapping key_name -> 'multi-class' or 'one-vs-all'
        min_samples: Minimum samples required per label (default: 2)

    Returns:
        Tuple of (labels_to_exclude, user_approved_removal)
        - labels_to_exclude: List of labels to exclude, or None if user cancelled
        - user_approved_removal: True if user approved removal, False if cancelled
    """
    import pandas as pd
    import json
    from collections import Counter
    from rich.table import Table
    from rich import box
    from rich.prompt import Confirm

    # Load CSV and count labels
    df = pd.read_csv(csv_path)

    # Filter to annotated rows only (non-null and non-empty)
    df_annotated = df[(df[annotation_column].notna()) & (df[annotation_column] != '')].copy()

    if len(df_annotated) == 0:
        self.console.print("[red]No annotated rows found in dataset[/red]")
        return None, False

    # Count labels by key and strategy
    label_counts = {}  # {label: count}

    for idx, row in df_annotated.iterrows():
        annotation_val = row.get(annotation_column)
        if pd.isna(annotation_val) or annotation_val == '':
            continue

        try:
            if isinstance(annotation_val, str):
                annotation = json.loads(annotation_val)
            elif isinstance(annotation_val, dict):
                annotation = annotation_val
            else:
                continue
        except (json.JSONDecodeError, ValueError):
            # Try Python literal eval
            try:
                import ast
                annotation = ast.literal_eval(annotation_val)
            except:
                continue

        if not isinstance(annotation, dict):
            continue

        # Process each key according to its strategy
        for key in keys_to_train:
            if key not in annotation:
                continue

            value = annotation[key]

            # Skip None and empty values
            if value is None or value == '':
                continue

            strategy = key_strategies.get(key, 'multi-class')

            # For both multi-class and one-vs-all, we need to count individual labels
            # because one-vs-all creates binary classifiers (class '1' = presence of label)
            if isinstance(value, list):
                for v in value:
                    if v is not None and v != '':
                        label_key = f"{key}_{v}"
                        label_counts[label_key] = label_counts.get(label_key, 0) + 1
            else:
                label_key = f"{key}_{value}"
                label_counts[label_key] = label_counts.get(label_key, 0) + 1

    # Find insufficient labels
    insufficient_labels = {
        label: count for label, count in label_counts.items()
        if count < min_samples
    }

    if not insufficient_labels:
        # All labels are sufficient
        return [], False

    # Display warning with comprehensive table
    self.console.print(f"\n[bold red][!]  INSUFFICIENT SAMPLES DETECTED (BEFORE FILE CREATION)[/bold red]\n")
    self.console.print(f"[yellow]The following labels have fewer than {min_samples} samples (minimum for train+validation split):[/yellow]\n")

    # Create detailed table showing strategy per label
    table = Table(border_style="red", show_header=True, header_style="bold red", box=box.ROUNDED, expand=True)
    table.add_column("Label", style="yellow bold", ratio=1, overflow="fold")
    table.add_column("Samples", style="red", justify="right", no_wrap=True)
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Status", style="red", no_wrap=True)

    for label, count in sorted(insufficient_labels.items(), key=lambda x: x[1]):
        # Extract key from label (format: key_value)
        key_name = label.split('_')[0] if '_' in label else label
        strategy = key_strategies.get(key_name, 'multi-class')

        table.add_row(
            label,
            str(count),
            strategy,
            "BLOCKED"
        )

    self.console.print(table)
    self.console.print()

    # Explain what will happen
    self.console.print("[bold]Options:[/bold]")
    self.console.print("  • [green]Remove[/green]: Automatically remove insufficient labels from the dataset")
    self.console.print("  • [red]Cancel[/red]: Stop training and fix dataset manually\n")

    should_remove = Confirm.ask(
        "Remove insufficient labels automatically?",
        default=False
    )

    if not should_remove:
        self.console.print("[yellow][FAIL] Training cancelled. Please annotate more samples or select different keys.[/yellow]")
        return None, False

    # User approved removal
    labels_to_exclude = list(insufficient_labels.keys())
    return labels_to_exclude, True

def _filter_csv_remove_insufficient_labels(
    self,
    csv_path: str,
    annotation_column: str,
    labels_to_exclude: List[str]
) -> str:
    """
    Filter CSV to remove insufficient labels from annotations.

    Args:
        csv_path: Path to original CSV
        annotation_column: Column with JSON annotations
        labels_to_exclude: List of labels to remove (format: key_value)

    Returns:
        Path to filtered CSV file
    """
    import pandas as pd
    import json
    from pathlib import Path

    df = pd.read_csv(csv_path)
    csv_path_obj = Path(csv_path)

    # Create filtered CSV path
    filtered_path = csv_path_obj.parent / f"{csv_path_obj.stem}_filtered{csv_path_obj.suffix}"

    labels_removed_count = 0
    samples_modified_count = 0

    for idx, row in df.iterrows():
        annotation_val = row.get(annotation_column)
        if pd.isna(annotation_val) or annotation_val == '':
            continue

        try:
            if isinstance(annotation_val, str):
                annotation = json.loads(annotation_val)
            elif isinstance(annotation_val, dict):
                annotation = annotation_val
            else:
                continue
        except (json.JSONDecodeError, ValueError):
            try:
                import ast
                annotation = ast.literal_eval(annotation_val)
            except:
                continue

        if not isinstance(annotation, dict):
            continue

        # Filter labels
        modified = False
        for key, value in list(annotation.items()):
            if value is None or value == '':
                continue

            if isinstance(value, list):
                # Remove values from list
                original_length = len(value)
                filtered_values = [
                    v for v in value
                    if v is not None and v != '' and f"{key}_{v}" not in labels_to_exclude
                ]
                if len(filtered_values) < original_length:
                    annotation[key] = filtered_values
                    labels_removed_count += (original_length - len(filtered_values))
                    modified = True
            else:
                # Check if this label should be excluded
                label_key = f"{key}_{value}"
                if label_key in labels_to_exclude:
                    # Set to None to indicate removal
                    annotation[key] = None
                    labels_removed_count += 1
                    modified = True

        if modified:
            samples_modified_count += 1
            # Update the annotation in the dataframe
            df.at[idx, annotation_column] = json.dumps(annotation)

    # Save filtered CSV
    df.to_csv(filtered_path, index=False)

    self.console.print(f"\n[green]✓ Filtered CSV created:[/green] {filtered_path}")
    self.console.print(f"  [cyan]• Samples modified:[/cyan] {samples_modified_count}")
    self.console.print(f"  [cyan]• Label instances removed:[/cyan] {labels_removed_count}")
    self.console.print(f"  [cyan]• Insufficient label types:[/cyan] {len(labels_to_exclude)}\n")

    return str(filtered_path)

def _validate_and_filter_insufficient_labels(
    self,
    input_file: str,
    strategy: str,
    min_samples: int = 2,
    auto_remove: bool = False,
    train_by_language: bool = False
) -> Tuple[str, bool]:
    """
    Validate that all labels have at least min_samples.
    If not, prompt user to remove insufficient labels.

    CRITICAL: This validation must be LANGUAGE-AWARE when train_by_language=True
    to match the actual splitting logic in DataUtil.prepare_splits().

    Args:
        input_file: Path to JSONL training file
        strategy: 'multi-label' or 'single-label' (multi-class)
        min_samples: Minimum samples required per label (default: 2 for train+val split)
        auto_remove: If True, automatically remove insufficient labels without prompting
        train_by_language: If True, validate per-language label counts (CRITICAL for multilingual)

    Returns:
        Tuple of (filtered_file_path, was_modified)
    """
    import json
    from collections import Counter
    from pathlib import Path
    from rich.table import Table
    from rich import box
    from rich.prompt import Confirm

    input_path = Path(input_file)
    if not input_path.exists():
        return str(input_file), False

    # Only JSON/JSONL files are supported for automated insufficiency checks
    if input_path.suffix.lower() not in {'.jsonl', '.json'}:
        return str(input_file), False

    # Read dataset and count labels
    # CRITICAL: When train_by_language=True, count per language-label combination
    label_counter = Counter()
    records = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                records.append(record)

                # Extract labels based on strategy
                labels_data = record.get('labels', record.get('label'))
                lang = record.get('lang', 'unknown') if train_by_language else None

                if strategy == 'multi-label':
                    # Labels is a list of strings
                    if isinstance(labels_data, list):
                        for label in labels_data:
                            if train_by_language:
                                # CRITICAL: Count per language (matches DataUtil.prepare_splits logic)
                                key = f"{label}_{lang}"
                            else:
                                key = str(label)
                            label_counter[key] += 1
                    elif isinstance(labels_data, str):
                        if train_by_language:
                            key = f"{labels_data}_{lang}"
                        else:
                            key = labels_data
                        label_counter[key] += 1
                else:
                    # Single-label: labels is a string
                    if labels_data:
                        if train_by_language:
                            key = f"{labels_data}_{lang}"
                        else:
                            key = str(labels_data)
                        label_counter[key] += 1

    except Exception as e:
        self.logger.warning(f"Could not validate labels: {e}")
        return str(input_file), False

    # Find insufficient labels
    insufficient_labels = {
        label: count for label, count in label_counter.items()
        if count < min_samples
    }

    if not insufficient_labels:
        # All labels have sufficient samples
        return str(input_file), False

    # Display warning
    self.console.print(f"\n[bold red][!]  INSUFFICIENT SAMPLES DETECTED[/bold red]\n")
    if train_by_language:
        self.console.print(f"[yellow]The following language-specific labels have fewer than {min_samples} samples (minimum for train+validation split):[/yellow]")
        self.console.print(f"[dim]Note: Validation is language-aware because train_by_language=True[/dim]\n")
    else:
        self.console.print(f"[yellow]The following labels have fewer than {min_samples} samples (minimum for train+validation split):[/yellow]\n")

    table = Table(border_style="red", show_header=True, header_style="bold red", box=box.ROUNDED, expand=True)
    table.add_column("Label", style="yellow bold", ratio=1, overflow="fold")
    table.add_column("Samples", style="red", justify="right", no_wrap=True)
    table.add_column("Status", style="red", no_wrap=True)

    for label, count in sorted(insufficient_labels.items(), key=lambda x: x[1]):
        table.add_row(
            label,
            str(count),
            "BLOCKED"
        )

    self.console.print(table)
    self.console.print()

    # Ask user what to do
    if not auto_remove:
        self.console.print("[bold]Options:[/bold]")
        if strategy == 'multi-label':
            self.console.print("  • [green]Remove[/green]: Automatically remove insufficient labels from samples (samples will be kept)")
            self.console.print("  • [red]Cancel[/red]: Stop training and fix dataset manually\n")
        else:
            self.console.print("  • [green]Remove[/green]: Automatically remove samples with insufficient labels")
            self.console.print("  • [red]Cancel[/red]: Stop training and fix dataset manually\n")

        should_remove = Confirm.ask(
            "Remove insufficient labels automatically?",
            default=False
        )

        if not should_remove:
            self.console.print("[yellow][FAIL] Training cancelled. Please fix dataset manually.[/yellow]")
            raise ValueError(f"Dataset contains {len(insufficient_labels)} label(s) with insufficient samples (< {min_samples})")

    # Filter dataset
    self.console.print(f"\n[yellow]Filtering dataset to remove insufficient labels...[/yellow]")

    filtered_records = []
    removed_count = 0
    labels_removed_count = 0  # Track number of label instances removed
    samples_with_removed_labels = 0  # Track samples that had labels removed but were kept

    for record in records:
        labels_data = record.get('labels', record.get('label'))
        lang = record.get('lang', 'unknown') if train_by_language else None

        if strategy == 'multi-label':
            # Filter list of labels - KEEP SAMPLE even if all labels are removed
            if isinstance(labels_data, list):
                original_labels = labels_data
                if train_by_language:
                    # Check language-specific keys
                    filtered_labels = [
                        label for label in labels_data
                        if f"{label}_{lang}" not in insufficient_labels
                    ]
                else:
                    filtered_labels = [
                        label for label in labels_data
                        if str(label) not in insufficient_labels
                    ]

                # Count removed labels
                removed_labels_in_sample = len(original_labels) - len(filtered_labels)
                if removed_labels_in_sample > 0:
                    labels_removed_count += removed_labels_in_sample
                    samples_with_removed_labels += 1

                # CRITICAL FIX: Keep record even if all labels were removed
                # The sample itself is still valid, just has no sufficient labels
                record_copy = record.copy()
                record_copy['labels'] = filtered_labels  # May be empty list
                filtered_records.append(record_copy)
            else:
                # Single label in multi-label format - convert to list and check
                if labels_data:
                    if train_by_language:
                        check_key = f"{labels_data}_{lang}"
                    else:
                        check_key = str(labels_data)

                    if check_key not in insufficient_labels:
                        # Keep as-is (string format)
                        filtered_records.append(record)
                    else:
                        # Label is insufficient - keep sample but remove label
                        labels_removed_count += 1
                        samples_with_removed_labels += 1
                        record_copy = record.copy()
                        record_copy['labels'] = []  # Empty labels list
                        filtered_records.append(record_copy)
                else:
                    # No labels at all - keep sample
                    filtered_records.append(record)
        else:
            # Single-label: MUST remove sample if label is insufficient
            # (cannot have a single-label sample with no label)
            if labels_data:
                if train_by_language:
                    check_key = f"{labels_data}_{lang}"
                else:
                    check_key = str(labels_data)

                if check_key not in insufficient_labels:
                    filtered_records.append(record)
                else:
                    # For single-label, we must remove the sample
                    removed_count += 1
                    labels_removed_count += 1
            else:
                # No label - remove sample
                removed_count += 1

    if not filtered_records:
        msg = (
            f"Dataset '{input_path.name}' has no samples after removing insufficient labels. "
            "Please annotate more data or adjust your label selection."
        )
        self.console.print(f"[yellow][!] {msg}[/yellow]\n")
        self.logger.warning(msg, extra={"dataset": input_path.name, "path": str(input_path)})
        raise ValueError(msg)

    # Save filtered dataset
    filtered_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"

    with open(filtered_path, 'w', encoding='utf-8') as f:
        for record in filtered_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    self.console.print(f"[green]Filtered dataset saved:[/green] {filtered_path.name}")
    self.console.print(f"  • [cyan]Original samples:[/cyan] {len(records)}")
    self.console.print(f"  • [cyan]Filtered samples:[/cyan] {len(filtered_records)}")

    if strategy == 'multi-label':
        # For multi-label, show label removal stats (samples are kept)
        self.console.print(f"  • [green]Samples kept:[/green] {len(filtered_records)} (all samples preserved)")
        if removed_count > 0:
            self.console.print(f"  • [yellow]Samples removed:[/yellow] {removed_count} (only if needed)")
        self.console.print(f"  • [yellow]Samples with labels removed:[/yellow] {samples_with_removed_labels}")
        self.console.print(f"  • [red]Label instances removed:[/red] {labels_removed_count}")
        self.console.print(f"  • [red]Insufficient label types:[/red] {len(insufficient_labels)}")
    else:
        # For single-label, samples must be removed if label is insufficient
        self.console.print(f"  • [yellow]Removed samples:[/yellow] {removed_count}")
        self.console.print(f"  • [red]Removed label types:[/red] {len(insufficient_labels)}")

    self.console.print()

    return str(filtered_path), True


def _validate_all_training_files_before_training(
    self,
    bundle: TrainingDataBundle,
    min_samples: int = 2,
    train_by_language: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Centralized validation of all training datasets before launching training.

    Detects insufficient labels across every generated file (primary, per-key, per-value),
    with optional language-aware counting. Any filtered datasets are written to companion
    files and the bundle is updated to point to the sanitized versions.
    """
    from collections import Counter
    import json
    from pathlib import Path
    from rich import box
    from rich.prompt import Confirm
    from rich.table import Table

    if bundle is None:
        return False, "No training bundle was produced."

    files_to_validate: List[Tuple[str, Path]] = []

    if getattr(bundle, "primary_file", None):
        files_to_validate.append(("primary", Path(bundle.primary_file)))

    training_files = getattr(bundle, "training_files", {}) or {}
    for key, file_path in training_files.items():
        if file_path:
            files_to_validate.append((key, Path(file_path)))

    if not files_to_validate:
        return True, None

    def _infer_strategy(path: Path, default_strategy: str) -> str:
        """Determine whether the dataset stores labels as lists (multi-label) or scalars."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    labels = record.get("labels")
                    if isinstance(labels, list):
                        return "multi-label"
                    if isinstance(labels, str) and labels:
                        return "single-label"
                    single = record.get("label")
                    if isinstance(single, str) and single:
                        return "single-label"
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug(f"Strategy inference failed for {path}: {exc}")
        return "multi-label" if default_strategy == "multi-label" else "single-label"

    default_strategy = bundle.metadata.get("training_approach", bundle.strategy or "multi-label")
    all_insufficient: Dict[str, Dict[str, int]] = {}

    for file_key, file_path in files_to_validate:
        if not file_path.exists():
            self.logger.warning(f"Training dataset missing: {file_path}")
            continue

        strategy = _infer_strategy(file_path, default_strategy)
        label_counter: Counter[str] = Counter()

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    labels_field = record.get("labels", record.get("label"))
                    language = record.get("lang", "unknown") if train_by_language else None

                    if isinstance(labels_field, list):
                        for label in labels_field:
                            if label is None or label == "":
                                continue
                            key = f"{label}_{language}" if language else str(label)
                            label_counter[key] += 1
                    elif isinstance(labels_field, str) and labels_field:
                        key = f"{labels_field}_{language}" if train_by_language else labels_field
                        label_counter[key] += 1
        except Exception as exc:
            self.logger.warning(f"Could not analyze {file_key} ({file_path}): {exc}")
            continue

        insufficient = {
            label: count for label, count in label_counter.items()
            if count < min_samples
        }
        if insufficient:
            all_insufficient[file_key] = insufficient

    if not all_insufficient:
        return True, None

    self.console.print(f"\n[bold red][!]  INSUFFICIENT SAMPLES DETECTED[/bold red]\n")
    if train_by_language:
        self.console.print(
            f"[yellow]Each language-specific label needs at least {min_samples} samples "
            "to support train/validation splits.[/yellow]\n"
        )
    else:
        self.console.print(
            f"[yellow]Each label needs at least {min_samples} samples to support train/validation splits.[/yellow]\n"
        )

    table = Table(border_style="red", show_header=True, header_style="bold red", box=box.ROUNDED, expand=True)
    table.add_column("Dataset", style="cyan bold", no_wrap=True)
    table.add_column("Label", style="yellow bold", overflow="fold")
    table.add_column("Samples", style="red", justify="right", no_wrap=True)
    table.add_column("Status", style="red", no_wrap=True)

    for file_key, labels in sorted(all_insufficient.items()):
        for label, count in sorted(labels.items(), key=lambda item: item[1]):
            table.add_row(file_key, label, str(count), "BLOCKED")

    self.console.print(table)
    self.console.print()
    self.console.print("[bold]Options:[/bold]")
    self.console.print("  • [green]Remove[/green]: Automatically drop insufficient labels from impacted datasets")
    self.console.print("  • [red]Cancel[/red]: Stop training and adjust the dataset manually\n")

    if not Confirm.ask("Remove insufficient labels automatically?", default=False):
        self.console.print("[yellow][FAIL] Training cancelled. Please adjust your annotations.[/yellow]")
        return False, "Insufficient samples for some labels"

    self.console.print(f"\n[yellow]Filtering training datasets to remove insufficient labels...[/yellow]\n")
    updated_files = 0
    empty_datasets: List[str] = []
    for file_key, file_path in files_to_validate:
        insufficient = all_insufficient.get(file_key)
        if not insufficient:
            continue

        labels_to_exclude = set(insufficient.keys())
        filtered_records: List[Dict[str, Any]] = []
        removed_instances = 0

        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                labels_field = record.get("labels", record.get("label"))
                language = record.get("lang", "unknown") if train_by_language else None

                if isinstance(labels_field, list):
                    original_len = len(labels_field)
                    if train_by_language:
                        cleaned = [
                            label for label in labels_field
                            if f"{label}_{language}" not in labels_to_exclude
                        ]
                        cleaned = [
                            label for label in cleaned
                            if str(label) not in labels_to_exclude
                        ]
                    else:
                        cleaned = [
                            label for label in labels_field
                            if str(label) not in labels_to_exclude
                        ]
                    removed_instances += original_len - len(cleaned)
                    record["labels"] = cleaned
                    filtered_records.append(record)
                elif isinstance(labels_field, str) and labels_field:
                    key = f"{labels_field}_{language}" if train_by_language else labels_field
                    if key not in labels_to_exclude:
                        filtered_records.append(record)
                    else:
                        removed_instances += 1
                else:
                    filtered_records.append(record)

        if not filtered_records:
            empty_datasets.append(file_key)
            warning_msg = (
                f"Dataset '{file_key}' has no samples after removing insufficient labels. "
                "Skipping this dataset."
            )
            self.console.print(f"  [yellow][!] {warning_msg}[/yellow]")
            self.logger.warning(warning_msg, extra={"dataset": file_key, "path": str(file_path)})
            continue

        filtered_path = file_path.with_name(f"{file_path.stem}_filtered{file_path.suffix}")
        with filtered_path.open("w", encoding="utf-8") as handle:
            for record in filtered_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if file_key == "primary":
            bundle.primary_file = filtered_path
        elif file_key in training_files:
            bundle.training_files[file_key] = filtered_path

        updated_files += 1
        self.console.print(
            f"  [green]✓[/green] {file_key}: kept {len(filtered_records)} records "
            f"(removed {removed_instances} label instance(s)) → {filtered_path.name}"
        )

    if empty_datasets:
        for skipped_key in empty_datasets:
            if skipped_key == "primary":
                bundle.primary_file = None
            else:
                bundle.training_files.pop(skipped_key, None)

        if bundle.metadata:
            if "multiclass_keys" in bundle.metadata:
                bundle.metadata["multiclass_keys"] = [
                    key for key in bundle.metadata.get("multiclass_keys", [])
                    if key not in empty_datasets
                ]
            if "multilabel_keys" in bundle.metadata:
                bundle.metadata["multilabel_keys"] = [
                    key for key in bundle.metadata.get("multilabel_keys", [])
                    if key not in empty_datasets
                ]
            if "onevsall_keys" in bundle.metadata:
                bundle.metadata["onevsall_keys"] = [
                    key for key in bundle.metadata.get("onevsall_keys", [])
                    if key not in empty_datasets
                ]
            if "files_per_key" in bundle.metadata:
                bundle.metadata["files_per_key"] = {
                    key: value
                    for key, value in bundle.metadata.get("files_per_key", {}).items()
                    if key not in empty_datasets
                }

        skipped_list = ", ".join(empty_datasets)
        self.console.print(
            f"[yellow][!] Skipping {len(empty_datasets)} dataset(s) with no remaining samples: {skipped_list}[/yellow]\n"
        )

    self.console.print(f"\n[green]✓ Filtered {updated_files} training file(s)[/green]\n")
    return True, None


def _validate_split_ratios(self, train: float, validation: float, test: float) -> Tuple[float, float, float]:
    """Validate and normalize split ratios."""
    # Check total
    total = train + validation + test

    if abs(total - 1.0) > 0.001:
        # Auto-adjust
        factor = 1.0 / total
        train *= factor
        validation *= factor
        test *= factor
        self.console.print(f"  [yellow][!] Ratios adjusted to sum to 1.0[/yellow]")

    # Minimum values
    if train < 0.5:
        raise ValueError("Training ratio must be at least 50%")

    if validation > 0 and validation < 0.05:
        raise ValueError("Validation ratio must be at least 5% if used")

    if test > 0 and test < 0.05:
        raise ValueError("Test ratio must be at least 5% if used")

    return train, validation, test

def _display_split_summary(self, split_config: Dict[str, Any], keys_to_train: List[str],
                           all_keys_values: Dict[str, set], key_strategies: Dict[str, str]) -> None:
    """Display summary of split configuration."""
    from rich.table import Table
    from rich import box

    self.console.print("\n[bold green]✓ Split Configuration Complete[/bold green]\n")

    mode = split_config['mode']
    use_test_set = split_config['use_test_set']

    if mode == 'uniform':
        ratios = split_config['uniform']
        self.console.print("[bold]Uniform Split (all keys/values):[/bold]")
        self.console.print(f"  • Train:      {ratios['train_ratio']:.1%}")
        self.console.print(f"  • Validation: {ratios['validation_ratio']:.1%}")
        if use_test_set:
            self.console.print(f"  • Test:       {ratios['test_ratio']:.1%}")

    else:
        self.console.print("[bold]Custom Split:[/bold]")

        custom_by_key = split_config.get('custom_by_key', {})
        custom_by_value = split_config.get('custom_by_value', {})
        defaults = split_config.get('defaults', {})

        if custom_by_key:
            self.console.print(f"\n  [green]Configured keys: {len(custom_by_key)}[/green]")
            for key, ratios in list(custom_by_key.items())[:5]:
                self.console.print(f"    • {key}: {ratios['train_ratio']:.1%} / {ratios['validation_ratio']:.1%} / {ratios['test_ratio']:.1%}")
            if len(custom_by_key) > 5:
                self.console.print(f"    ... and {len(custom_by_key) - 5} more")

        if custom_by_value:
            self.console.print(f"\n  [yellow]Configured values: {len(custom_by_value)}[/yellow]")
            for value, ratios in list(custom_by_value.items())[:5]:
                self.console.print(f"    • {value}: {ratios['train_ratio']:.1%} / {ratios['validation_ratio']:.1%} / {ratios['test_ratio']:.1%}")
            if len(custom_by_value) > 5:
                self.console.print(f"    ... and {len(custom_by_value) - 5} more")

        if defaults:
            self.console.print(f"\n  [dim]Defaults (for others): {defaults['train_ratio']:.1%} / {defaults['validation_ratio']:.1%} / {defaults['test_ratio']:.1%}[/dim]")

    self.console.print(
        "[dim]Reminder:[/dim] This validation split feeds Validation Lab (Mode 5). After human review, place the curated validation set inside your session directory under [white]validation/<session>/[/white] to compare model vs manual annotations."
    )
    self.console.print()

def _collect_quick_mode_parameters(
    self,
    bundle: TrainingDataBundle,
    preloaded_params: Optional[Dict[str, Any]] = None,
    step_context: str = "arena_quick",
    session_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Collect parameters for quick mode training (token strategy, model choice, epochs).

    Returns dict with keys: model_name, reinforced_learning, epochs
    Returns None if user cancels
    """
    training_approach = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None
    onevsall_keys = []
    multilabel_keys = []
    if hasattr(bundle, 'metadata') and bundle.metadata:
        onevsall_keys = bundle.metadata.get('onevsall_keys', []) or []
        multilabel_keys = bundle.metadata.get('multilabel_keys', []) or []

    persistent_onevsall_root: Optional[Path] = None
    effective_session_id = session_id or getattr(self, 'current_session_id', None)
    if effective_session_id:
        try:
            persistent_onevsall_root = get_training_data_dir(effective_session_id) / "onevsall_datasets"
            persistent_onevsall_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.logger.warning(
                "Could not prepare persistent directory for one-vs-all datasets (session=%s): %s",
                effective_session_id,
                exc,
            )
            persistent_onevsall_root = None

    self._set_onevsall_storage_root(persistent_onevsall_root)

    stage_models: Dict[str, Dict[str, Any]] = {}

    from rich.prompt import Prompt, IntPrompt, Confirm
    from llm_tool.utils.model_display import get_recommended_models, display_all_models
    from rich.table import Table
    from rich import box

    # Token length strategy selection
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    token_step_label = resolve_step_label("token_strategy", "STEP 1", context=step_context)
    self.console.print(f"[bold cyan]           {token_step_label}: Token Length Strategy                    [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Get languages from metadata
    languages = set()
    if hasattr(bundle, 'metadata') and bundle.metadata:
        languages = bundle.metadata.get('confirmed_languages', bundle.metadata.get('languages', set()))
    if not languages and hasattr(bundle, 'languages') and bundle.languages:
        languages = set([lang.upper() for lang in bundle.languages])
    if languages:
        languages = set([str(lang).upper() for lang in languages])

    # Get text length stats
    text_length_stats = bundle.metadata.get('text_length_stats', {}) if hasattr(bundle, 'metadata') else {}
    if text_length_stats.get('token_mean'):
        text_length_avg = text_length_stats['token_mean']
    elif text_length_stats.get('char_mean'):
        text_length_avg = text_length_stats['char_mean']
    else:
        text_length_avg = getattr(bundle, 'text_length_avg', 158)

    requires_long_model = text_length_stats.get('requires_long_model', False)

    # Get distribution data to calculate percentage exceeding 512 tokens
    distribution = text_length_stats.get('distribution', {})

    # Calculate percentage exceeding 512 tokens
    # Handle different possible structures of distribution
    total_docs = 0
    docs_exceeding_512 = 0
    pct_exceeding_512 = 0

    if distribution and isinstance(distribution, dict):
        # Try to extract counts - distribution might be nested
        try:
            # Check if values are integers (direct counts)
            if all(isinstance(v, (int, float)) for v in distribution.values()):
                total_docs = sum(distribution.values())
                docs_exceeding_512 = distribution.get('long', 0) + distribution.get('very_long', 0)
                pct_exceeding_512 = (docs_exceeding_512 / total_docs * 100) if total_docs > 0 else 0
            else:
                # Distribution might have nested structure - try to extract counts
                for key, value in distribution.items():
                    if isinstance(value, dict) and 'count' in value:
                        total_docs += value['count']
                        if key in ['long', 'very_long']:
                            docs_exceeding_512 += value['count']
                pct_exceeding_512 = (docs_exceeding_512 / total_docs * 100) if total_docs > 0 else 0
        except (TypeError, KeyError, AttributeError):
            # Fallback to percentage-based calculation
            pass

    # If we couldn't calculate from distribution, try direct percentage fields
    if pct_exceeding_512 == 0 and total_docs == 0:
        if 'pct_long' in text_length_stats and 'pct_very_long' in text_length_stats:
            pct_exceeding_512 = text_length_stats.get('pct_long', 0) + text_length_stats.get('pct_very_long', 0)
            # Estimate docs count if we have the total
            if 'total_docs' in text_length_stats:
                total_docs = text_length_stats['total_docs']
                docs_exceeding_512 = int(total_docs * pct_exceeding_512 / 100)

    # Show token length summary
    self.console.print("[bold]Your Dataset Token Analysis:[/bold]\n")

    stats_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.SIMPLE, expand=True)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="white", ratio=1, overflow="fold")

    # Use actual values from text_length_stats
    token_mean = text_length_stats.get('token_mean', text_length_stats.get('avg_tokens', 0))
    token_median = text_length_stats.get('token_median', text_length_stats.get('median_tokens', 0))
    token_p95 = text_length_stats.get('token_p95', text_length_stats.get('p95_tokens', 0))
    token_max = text_length_stats.get('token_max', text_length_stats.get('max_tokens', 0))

    stats_table.add_row("Mean tokens per document", f"{token_mean:.0f}")
    stats_table.add_row("Median tokens", f"{token_median:.0f}")
    stats_table.add_row("95th percentile", f"{token_p95:.0f}")
    stats_table.add_row("Maximum tokens", f"{token_max:.0f}")
    stats_table.add_row("[bold]% exceeding 512 tokens[/bold]", f"[bold yellow]{pct_exceeding_512:.1f}%[/bold yellow]")

    # Show distribution if available
    if distribution and total_docs > 0:
        self.console.print(stats_table)
        self.console.print()

        self.console.print("[bold]Token Length Distribution:[/bold]\n")
        dist_table = Table(show_header=True, header_style="bold magenta", border_style="blue", box=box.SIMPLE, expand=True)
        dist_table.add_column("Category", style="cyan", no_wrap=True)
        dist_table.add_column("Token Range", style="white", no_wrap=True)
        dist_table.add_column("Count", style="green", no_wrap=True, justify="right")
        dist_table.add_column("Percentage", style="yellow", no_wrap=True, justify="right")

        # Extract counts - handle both dict and int values
        def get_count(category_data):
            if isinstance(category_data, dict):
                return category_data.get('count', 0)
            elif isinstance(category_data, (int, float)):
                return int(category_data)
            return 0

        short_count = get_count(distribution.get('short', 0))
        medium_count = get_count(distribution.get('medium', 0))
        long_count = get_count(distribution.get('long', 0))
        very_long_count = get_count(distribution.get('very_long', 0))

        dist_table.add_row("Short", "< 128 tokens", f"{short_count:,}", f"{short_count/total_docs*100:.1f}%")
        dist_table.add_row("Medium", "128-511 tokens", f"{medium_count:,}", f"{medium_count/total_docs*100:.1f}%")
        dist_table.add_row("[yellow]Long[/yellow]", "[yellow]512-1023 tokens[/yellow]", f"[yellow]{long_count:,}[/yellow]", f"[yellow]{long_count/total_docs*100:.1f}%[/yellow]")
        dist_table.add_row("[red]Very Long[/red]", "[red]≥ 1024 tokens[/red]", f"[red]{very_long_count:,}[/red]", f"[red]{very_long_count/total_docs*100:.1f}%[/red]")

        self.console.print(dist_table)
    else:
        self.console.print(stats_table)

    self.console.print()

    # Check if there are ANY documents exceeding 512 tokens
    if pct_exceeding_512 == 0.0:
        # No documents exceed 512 tokens - no strategy needed!
        self.console.print("[bold green]Perfect! All documents fit within 512 tokens[/bold green]")
        self.console.print("[dim]No special handling needed - you can use any standard BERT model.[/dim]\n")

        self.console.print("[bold cyan]Why this matters:[/bold cyan]")
        self.console.print(f"  • [green]Maximum document length:[/green] {token_max:.0f} tokens (well below 512 limit)")
        self.console.print(f"  • [green]Mean document length:[/green] {token_mean:.0f} tokens")
        self.console.print(f"  • [green]95th percentile:[/green] {token_p95:.0f} tokens")
        self.console.print("  • [green]All data will be used[/green] without chunking or truncation")
        self.console.print("  • [green]Fastest training[/green] with standard models (BERT, RoBERTa, CamemBERT, etc.)\n")

        # Set default flags - no special handling needed
        prefers_long_models = False
        exclude_long_texts = False
        split_long_texts = False
    else:
        # Determine recommended strategy based on percentage (intelligent)
        if pct_exceeding_512 < 10:
            recommended_strategy = "truncate"
            rec_reason = f"Only {pct_exceeding_512:.1f}% exceed 512 tokens - splitting long documents will preserve all information"
        elif pct_exceeding_512 < 25:
            recommended_strategy = "truncate"
            rec_reason = f"{pct_exceeding_512:.1f}% exceed 512 tokens - splitting is recommended, or consider long models for better context"
        elif pct_exceeding_512 < 40:
            recommended_strategy = "long_models"
            rec_reason = f"{pct_exceeding_512:.1f}% exceed 512 tokens - long models recommended to preserve document context"
        else:
            recommended_strategy = "long_models"
            rec_reason = f"{pct_exceeding_512:.1f}% exceed 512 tokens - long models strongly recommended"

        # Present 3 strategies
        self.console.print("[bold yellow][!] Standard BERT models have a 512 token limit[/bold yellow]")
        self.console.print("[dim]You need to choose how to handle longer documents:[/dim]\n")

        strategy_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
        strategy_table.add_column("Strategy", style="cyan bold", no_wrap=True)
        strategy_table.add_column("Description", style="white", overflow="fold", ratio=1)

        truncate_mark = " [green]RECOMMENDED[/green]" if recommended_strategy == "truncate" else ""
        exclude_mark = " [green]RECOMMENDED[/green]" if recommended_strategy == "exclude" else ""
        long_mark = " [green]RECOMMENDED[/green]" if recommended_strategy == "long_models" else ""

        # Calculate how many extra samples we'd get from splitting
        estimated_extra_samples = 0
        if docs_exceeding_512 > 0:
            # Estimate based on average tokens for long docs
            estimated_extra_samples = int(docs_exceeding_512 * 1.5)  # Conservative estimate
            extra_info = f"Creates ~{estimated_extra_samples:,} additional training samples from long documents"
        else:
            extra_info = "No documents exceed 512 tokens"

        strategy_table.add_row(
            "1. Split/Chunk" + truncate_mark,
            "Split long documents into 512-token chunks (no overlap)\n"
            f"• [green]Each chunk inherits the parent labels[/green] → more training data\n"
            f"• Example: 1024-token doc → 2 samples (tokens 0-510, 510-1020)\n"
            f"• {extra_info}\n"
            f"• [yellow]Label noise risk[/yellow]: if a doc has >=3 labels, every chunk\n"
            f"  inherits all of them even if only one is present in that chunk.\n"
            f"  Prefer Exclude for short, multi-thematic documents.\n"
            f"• Works with all standard models (BERT, RoBERTa, CamemBERT, etc.)\n"
            f"• [bold]No data loss[/bold] - all text reaches the model"
        )
        strategy_table.add_row(
            "2. Exclude" + exclude_mark,
            f"Remove documents exceeding 512 tokens entirely\n"
            f"• Would exclude {docs_exceeding_512:,} documents ({pct_exceeding_512:.1f}% of dataset)\n"
            f"• [red]Reduces training data significantly[/red]\n"
            f"• Model won't learn from long documents\n"
            f"• Only use if long documents are outliers/noise"
        )
        strategy_table.add_row(
            "3. Long Models" + long_mark,
            "Use long-document models (up to 4096 tokens)\n"
            "• Preserves full document context in single sample\n"
            "• Better for tasks requiring full document understanding\n"
            "• Slower training (~15-30 min) and inference\n"
            "• Models: Longformer, BigBird, Long-T5, XLM-RoBERTa-Longformer"
        )

        self.console.print(strategy_table)
        self.console.print()

        self.console.print(f"[bold yellow]Smart Recommendation:[/bold yellow] [cyan]{rec_reason}[/cyan]\n")

        # Ask user to choose
        strategy_choice = Prompt.ask(
            "[bold yellow]Choose strategy[/bold yellow]",
            choices=["1", "2", "3", "split", "chunk", "exclude", "long", "long_models"],
            default="1" if recommended_strategy == "truncate" else ("2" if recommended_strategy == "exclude" else "3")
        )

        # Map choice to boolean flags
        # Initialize all flags
        prefers_long_models = False
        exclude_long_texts = False
        split_long_texts = False

        if strategy_choice in ["1", "split", "chunk", "truncate"]:
            split_long_texts = True
            if docs_exceeding_512 > 0:
                self.console.print(f"[green]Strategy: Split long documents into chunks (creates ~{estimated_extra_samples:,} extra samples)[/green]\n")
            else:
                self.console.print("[green]Strategy: Split long documents (if any) into chunks[/green]\n")
        elif strategy_choice in ["2", "exclude"]:
            exclude_long_texts = True
            self.console.print(f"[yellow]✓ Strategy: Exclude {docs_exceeding_512:,} documents >512 tokens ({pct_exceeding_512:.1f}% of dataset)[/yellow]\n")
        else:  # "3", "long", or "long_models"
            prefers_long_models = True
            self.console.print("[green]Strategy: Use long-document models (up to 4096 tokens)[/green]\n")

    # Store choice in text_length_stats for later use
    text_length_stats['user_prefers_long_models'] = prefers_long_models
    text_length_stats['exclude_long_texts'] = exclude_long_texts
    text_length_stats['split_long_texts'] = split_long_texts

    # ALSO mirror the flags at the top level of bundle.metadata. Downstream
    # consumers (e.g. the training-config propagation in _training_studio_run_quick)
    # read bundle.metadata.get('exclude_long_texts'), and without this mirror
    # the choice made here lived only inside the nested text_length_stats dict
    # and was silently ignored, leaving BertBase to truncate long sequences.
    if hasattr(bundle, 'metadata') and bundle.metadata is not None:
        bundle.metadata['exclude_long_texts'] = exclude_long_texts
        bundle.metadata['split_long_texts'] = split_long_texts
        bundle.metadata['user_prefers_long_models'] = prefers_long_models
        bundle.metadata['text_length_stats'] = text_length_stats

    # Multilingual strategy (if multiple languages detected)
    train_by_language = False
    if len(languages) > 1:
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        multilingual_step_label = resolve_step_label("multilingual_strategy", "STEP 2", context=step_context)
        self.console.print(f"[bold cyan]           {multilingual_step_label}: Multilingual Strategy                   [/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        self.console.print(f"[bold]Your dataset contains multiple languages:[/bold] {', '.join(sorted(languages))}\n")

        strategy_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
        strategy_table.add_column("Approach", style="cyan bold", no_wrap=True)
        strategy_table.add_column("Description", style="white", overflow="fold", ratio=1)

        strategy_table.add_row(
            "1. Multilingual Model",
            "Train ONE model that handles all languages\n"
            f"• Works across {', '.join(sorted(languages))} without distinction\n"
            "• Faster: Single training run\n"
            "• Good for: Cross-lingual tasks, similar performance needed across languages\n"
            "• Models: XLM-RoBERTa, mBERT, mT5, etc.\n"
            "• [green]Recommended if[/green]: Languages are balanced in dataset"
        )
        strategy_table.add_row(
            "2. One Model per Language",
            "Train SEPARATE specialized models for each language\n"
            f"• {len(languages)} models total: one for each language\n"
            f"• Each model specialized for its language (e.g., CamemBERT for FR, BERT for EN)\n"
            "• Better performance: Language-specific models often outperform multilingual\n"
            "• Longer training: Multiple training runs\n"
            f"• You'll select a model for each language: {', '.join(sorted(languages))}\n"
            "• [green]Recommended if[/green]: Best possible performance is priority"
        )

        self.console.print(strategy_table)
        self.console.print()

        multilingual_choice = Prompt.ask(
            "[bold yellow]Choose approach[/bold yellow]",
            choices=["1", "2", "multilingual", "per-language", "per_language"],
            default="2"  # Recommend per-language for better performance
        )

        if multilingual_choice in ["2", "per-language", "per_language"]:
            train_by_language = True
            self.console.print(f"\n[green]✓ Will train {len(languages)} specialized models (one per language)[/green]\n")
        else:
            train_by_language = False
            self.console.print("\n[green]✓ Will train 1 multilingual model[/green]\n")

    # Model selection
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    model_step_label = resolve_step_label("model_selection", "STEP 3", context=step_context)
    self.console.print(f"[bold cyan]           {model_step_label}: Model Selection                         [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    # Import model utilities
    from llm_tool.utils.model_display import get_recommended_models, MODEL_METADATA

    def _prompt_model_choice_for_stage(
        lang: str,
        stage_label: str,
        recommended_models: Optional[List[str]],
        default_model: str,
    ) -> str:
        """Interactive prompt to pick a model for a specific language/stage."""
        recommended_models = recommended_models or []
        if recommended_models:
            self.console.print(f"[bold cyan]Top 10 Recommended Models for {lang} ({stage_label} stage):[/bold cyan]\n")

            models_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            models_table.add_column("#", style="yellow", width=3, no_wrap=True)
            models_table.add_column("Model ID", style="cyan", no_wrap=True)
            models_table.add_column("Languages", style="green", no_wrap=True)
            models_table.add_column("Max Tokens", style="blue", no_wrap=True)
            models_table.add_column("Size", style="magenta", no_wrap=True)
            models_table.add_column("Description", style="white", overflow="fold", ratio=1)

            for idx, model_id in enumerate(recommended_models[:10], 1):
                meta = MODEL_METADATA.get(model_id, {})
                from llm_tool.utils.model_display import format_language_display
                langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                max_len = str(meta.get('max_length', '?'))
                size = meta.get('size', '?')
                desc = meta.get('description', '')[:43]
                models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

            self.console.print(models_table)

        prompt_label = f"\n[bold yellow]Model for {lang} ({stage_label})[/bold yellow]"
        while True:
            self.console.print(f"\n[dim]Selection options for {lang} ({stage_label}):[/dim]")
            self.console.print("[dim]  • Enter [cyan]1-10[/cyan] to select from the recommendations above[/dim]")
            self.console.print("[dim]  • Enter [cyan]'info X'[/cyan] (e.g., 'info 1') to see model details[/dim]")
            self.console.print(f"[dim]  • Enter [cyan]'all'[/cyan] to view all {len(MODEL_METADATA)} models[/dim]")
            self.console.print("[dim]  • Or enter any HuggingFace model ID directly[/dim]")

            model_input = Prompt.ask(prompt_label, default=default_model).strip()
            lower_input = model_input.lower()

            if lower_input.startswith('info '):
                info_target = model_input[5:].strip()
                if info_target.isdigit():
                    info_idx = int(info_target) - 1
                    if recommended_models and 0 <= info_idx < len(recommended_models):
                        self._display_model_details(recommended_models[info_idx], MODEL_METADATA)
                    else:
                        self.console.print(f"[red]Invalid model number: {info_target}[/red]")
                else:
                    self._display_model_details(info_target, MODEL_METADATA)
                continue

            if lower_input == 'all':
                self.console.print(f"\n[bold cyan]ALL {len(MODEL_METADATA)} Available Models:[/bold cyan]\n")
                all_models_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
                all_models_table.add_column("#", style="yellow", width=4, no_wrap=True)
                all_models_table.add_column("Model ID", style="cyan", no_wrap=True)
                all_models_table.add_column("Languages", style="green", no_wrap=True)
                all_models_table.add_column("Max Tokens", style="blue", no_wrap=True)
                all_models_table.add_column("Size", style="magenta", no_wrap=True)
                all_models_table.add_column("Description", style="white", overflow="fold", ratio=1)

                all_model_ids = list(MODEL_METADATA.keys())
                sorted_model_ids: List[str] = []
                for mid in recommended_models:
                    if mid in all_model_ids and mid not in sorted_model_ids:
                        sorted_model_ids.append(mid)
                for mid in all_model_ids:
                    if mid not in sorted_model_ids:
                        sorted_model_ids.append(mid)

                from llm_tool.utils.model_display import format_language_display
                for idx, model_id in enumerate(sorted_model_ids, 1):
                    meta = MODEL_METADATA.get(model_id, {})
                    langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                    max_len = str(meta.get('max_length', '?'))
                    size = meta.get('size', '?')
                    desc = meta.get('description', '')[:48]
                    style = "bold green" if model_id in recommended_models[:10] else "white"
                    all_models_table.add_row(
                        f"[{style}]{idx}[/{style}]",
                        f"[{style}]{model_id}[/{style}]",
                        langs,
                        max_len,
                        size,
                        desc
                    )

                self.console.print(all_models_table)
                self.console.print(f"\n[dim]Tip: Green rows are part of your top recommendations for {lang}[/dim]")
                continue

            if model_input.isdigit() and recommended_models:
                idx = int(model_input) - 1
                if 0 <= idx < len(recommended_models):
                    selected = recommended_models[idx]
                    self.console.print(f"[green]Selected for {lang}: {selected}[/green]")
                    return selected
                self.console.print(f"[yellow][!] Invalid selection. Using default: {default_model}[/yellow]")
                return default_model

            if model_input:
                return model_input
            return default_model

    def _prompt_global_model_choice_for_stage(
        stage_label: str,
        recommended_models: Optional[List[str]],
        default_model: str,
    ) -> str:
        """Prompt for a single global model selection (non per-language)."""
        recommended_models = recommended_models or []
        if recommended_models:
            self.console.print(f"[bold cyan]Top 10 Recommended Models ({stage_label} stage):[/bold cyan]\n")

            models_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            models_table.add_column("#", style="yellow", width=3, no_wrap=True)
            models_table.add_column("Model ID", style="cyan", no_wrap=True)
            models_table.add_column("Languages", style="green", no_wrap=True)
            models_table.add_column("Max Tokens", style="blue", no_wrap=True)
            models_table.add_column("Size", style="magenta", no_wrap=True)
            models_table.add_column("Description", style="white", overflow="fold", ratio=1)

            from llm_tool.utils.model_display import format_language_display
            for idx, model_id in enumerate(recommended_models[:10], 1):
                meta = MODEL_METADATA.get(model_id, {})
                langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                max_len = str(meta.get('max_length', '?'))
                size = meta.get('size', '?')
                desc = meta.get('description', '')[:43]
                models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

            self.console.print(models_table)

        prompt_label = f"\n[bold yellow]Model selection for {stage_label} stage[/bold yellow]"
        while True:
            self.console.print("\n[dim]Selection options:[/dim]")
            self.console.print("[dim]  • Enter [cyan]1-10[/cyan] to pick from recommendations[/dim]")
            self.console.print("[dim]  • Enter [cyan]'info X'[/cyan] to see model details[/dim]")
            self.console.print(f"[dim]  • Enter [cyan]'all'[/cyan] to browse the full catalog ({len(MODEL_METADATA)} models)[/dim]")
            self.console.print("[dim]  • Or provide any HuggingFace model ID[/dim]")

            model_input = Prompt.ask(prompt_label, default=default_model).strip()
            lower_input = model_input.lower()

            if lower_input.startswith('info '):
                info_target = model_input[5:].strip()
                if info_target.isdigit():
                    info_idx = int(info_target) - 1
                    if recommended_models and 0 <= info_idx < len(recommended_models):
                        self._display_model_details(recommended_models[info_idx], MODEL_METADATA)
                    else:
                        self.console.print(f"[red]Invalid model number: {info_target}[/red]")
                else:
                    self._display_model_details(info_target, MODEL_METADATA)
                continue

            if lower_input == 'all':
                self.console.print(f"\n[bold cyan]ALL {len(MODEL_METADATA)} Available Models:[/bold cyan]\n")
                all_models_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
                all_models_table.add_column("#", style="yellow", width=4, no_wrap=True)
                all_models_table.add_column("Model ID", style="cyan", no_wrap=True)
                all_models_table.add_column("Languages", style="green", no_wrap=True)
                all_models_table.add_column("Max Tokens", style="blue", no_wrap=True)
                all_models_table.add_column("Size", style="magenta", no_wrap=True)
                all_models_table.add_column("Description", style="white", overflow="fold", ratio=1)

                all_model_ids = list(MODEL_METADATA.keys())
                sorted_model_ids: List[str] = []
                for mid in recommended_models:
                    if mid in all_model_ids and mid not in sorted_model_ids:
                        sorted_model_ids.append(mid)
                for mid in all_model_ids:
                    if mid not in sorted_model_ids:
                        sorted_model_ids.append(mid)

                from llm_tool.utils.model_display import format_language_display
                for idx, model_id in enumerate(sorted_model_ids, 1):
                    meta = MODEL_METADATA.get(model_id, {})
                    langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                    max_len = str(meta.get('max_length', '?'))
                    size = meta.get('size', '?')
                    desc = meta.get('description', '')[:48]
                    style = "bold green" if model_id in recommended_models[:10] else "white"
                    all_models_table.add_row(
                        f"[{style}]{idx}[/{style}]",
                        f"[{style}]{model_id}[/{style}]",
                        langs,
                        max_len,
                        size,
                        desc
                    )

                self.console.print(all_models_table)
                continue

            if model_input.isdigit() and recommended_models:
                idx = int(model_input) - 1
                if 0 <= idx < len(recommended_models):
                    selected = recommended_models[idx]
                    self.console.print(f"[green]Selected model: {selected}[/green]")
                    return selected
                self.console.print(f"[yellow][!] Invalid selection. Using default: {default_model}[/yellow]")
                return default_model

            if model_input:
                return model_input
            return default_model

    # ASK ABOUT BENCHMARK MODE
    self.console.print("[bold]Benchmark Mode[/bold]")
    self.console.print("  • [cyan]Compare multiple models[/cyan] before full training")
    self.console.print("  • [cyan]Test on selected categories[/cyan] with class imbalance analysis")
    self.console.print("  • [cyan]See which models perform best[/cyan] on your specific data")
    self.console.print("  • [cyan]Make informed model selection[/cyan] based on real performance\n")

    self.console.print("[yellow]Requirements:[/yellow]")
    self.console.print("  • Must select at least [bold]2 models[/bold] per language (or 2+ multilingual models)")
    self.console.print("  • Benchmark runs quick training (3-5 epochs) on subset of data")
    self.console.print("  • Takes ~5-15 min depending on models selected\n")

    enable_benchmark = Confirm.ask(
        "[bold yellow]Enable benchmark mode to compare models?[/bold yellow]",
        default=False
    )

    # ============ BENCHMARK MODE INTEGRATION ============
    if enable_benchmark:
        # Run benchmark mode workflow
        benchmark_result = self._run_benchmark_mode(
            bundle=bundle,
            languages=languages,
            train_by_language=train_by_language,
            text_length_avg=text_length_avg,
            prefers_long_models=prefers_long_models
        )

        if benchmark_result is None:
            # User chose to stop
            return None

        # Extract selected models from benchmark result
        model_name = benchmark_result.get('model_name')
        models_by_language = benchmark_result.get('models_by_language', {})

        # Ensure model_name is set for compatibility
        if not model_name and models_by_language:
            # Per-language mode: use first model as primary for compatibility
            model_name = list(models_by_language.values())[0]

        # Show summary of benchmark-selected models
        if train_by_language and models_by_language:
            self.console.print(f"\n[bold green]✓ Models Selected from Benchmark:[/bold green]")
            for lang, model in sorted(models_by_language.items()):
                self.console.print(f"  • {lang}: [cyan]{model}[/cyan]")
        elif model_name:
            self.console.print(f"\n[bold green]✓ Model Selected from Benchmark:[/bold green]")
            self.console.print(f"  • [cyan]{model_name}[/cyan]")

        # Continue to rest of flow (epochs, reinforced learning, etc.)
        # with the models selected from benchmark
    else:
        # Normal flow: manual model selection

        # Get model strategy
        if train_by_language:
            model_strategy = "per-language"
        elif len(languages) > 1:
            model_strategy = "multilingual"
        elif 'FR' in languages:
            model_strategy = "fr"
        elif 'EN' in languages:
            model_strategy = "en"
        else:
            model_strategy = "multilingual"

        # Initialize models_by_language dict
        models_by_language = {}

    # ============ MODEL SELECTION (normal flow when benchmark disabled) ============
    # Handle per-language model selection
    if train_by_language and not enable_benchmark:
        # Select one model for each language
        for lang in sorted(languages):
            self.console.print(f"\n[bold yellow]{'─'*60}[/bold yellow]")
            self.console.print(f"[bold yellow]Selecting model for {lang} texts[/bold yellow]")
            self.console.print(f"[bold yellow]{'─'*60}[/bold yellow]\n")

            # Get recommendations for this specific language
            lang_recommended = get_recommended_models(
                languages={lang},  # Use set, not list
                avg_text_length=text_length_avg,
                requires_long_model=prefers_long_models,
                top_n=10
            )

            if lang_recommended:
                # Show top 10 models for this language
                self.console.print(f"[bold cyan]Top 10 Recommended Models for {lang}:[/bold cyan]\n")

                models_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
                models_table.add_column("#", style="yellow", width=3, no_wrap=True)
                models_table.add_column("Model ID", style="cyan", no_wrap=True)
                models_table.add_column("Languages", style="green", no_wrap=True)
                models_table.add_column("Max Tokens", style="blue", no_wrap=True)
                models_table.add_column("Size", style="magenta", no_wrap=True)
                models_table.add_column("Description", style="white", overflow="fold", ratio=1)

                for idx, model_id in enumerate(lang_recommended[:10], 1):
                    meta = MODEL_METADATA.get(model_id, {})
                    from llm_tool.utils.model_display import format_language_display
                    langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                    max_len = str(meta.get('max_length', '?'))
                    size = meta.get('size', '?')
                    desc = meta.get('description', '')[:43]

                    models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

                self.console.print(models_table)
                default_model = lang_recommended[0]
            else:
                # Fallback defaults by language
                if lang == 'FR':
                    default_model = 'camembert-base'
                elif lang == 'EN':
                    default_model = 'bert-base-uncased'
                else:
                    default_model = 'xlm-roberta-base'

            # Offer to display all models
            self.console.print(f"\n[dim]Tip: Selection Options:[/dim]")
            self.console.print(f"[dim]  • Enter [cyan]1-10[/cyan] to select from Top 10 recommendations[/dim]")
            self.console.print(f"[dim]  • Enter [cyan]'info X'[/cyan] (e.g., 'info 1') to see full details of a model[/dim]")
            self.console.print(f"[dim]  • Enter [cyan]'all'[/cyan] to see ALL {len(MODEL_METADATA)} available models[/dim]")
            self.console.print(f"[dim]  • Enter any [cyan]HuggingFace model ID[/cyan] directly[/dim]")

            model_input = Prompt.ask(f"\n[bold yellow]Model for {lang}[/bold yellow]", default=default_model)

            # Check if user wants info on a model
            if model_input.lower().startswith('info '):
                info_target = model_input[5:].strip()
                if info_target.isdigit():
                    info_idx = int(info_target) - 1
                    if lang_recommended and 0 <= info_idx < len(lang_recommended):
                        self._display_model_details(lang_recommended[info_idx], MODEL_METADATA)
                    else:
                        self.console.print(f"[red]Invalid model number: {info_target}[/red]")
                else:
                    self._display_model_details(info_target, MODEL_METADATA)
                # After showing info, ask again for selection
                model_input = Prompt.ask(f"\n[bold yellow]Model for {lang}[/bold yellow]", default=default_model)

            # Check if user wants to see all models
            if model_input.lower() == 'all':
                # Show ALL models with complete characteristics
                self.console.print(f"\n[bold cyan]ALL {len(MODEL_METADATA)} Available Models:[/bold cyan]\n")

                all_models_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
                all_models_table.add_column("#", style="yellow", width=4, no_wrap=True)
                all_models_table.add_column("Model ID", style="cyan", no_wrap=True)
                all_models_table.add_column("Languages", style="green", no_wrap=True)
                all_models_table.add_column("Max Tokens", style="blue", no_wrap=True)
                all_models_table.add_column("Size", style="magenta", no_wrap=True)
                all_models_table.add_column("Description", style="white", overflow="fold", ratio=1)

                # Sort models: recommended first, then by relevance
                all_model_ids = list(MODEL_METADATA.keys())
                sorted_model_ids = []
                for model_id in lang_recommended:
                    if model_id in all_model_ids:
                        sorted_model_ids.append(model_id)
                for model_id in all_model_ids:
                    if model_id not in sorted_model_ids:
                        sorted_model_ids.append(model_id)

                for idx, model_id in enumerate(sorted_model_ids, 1):
                    meta = MODEL_METADATA.get(model_id, {})
                    from llm_tool.utils.model_display import format_language_display
                    langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                    max_len = str(meta.get('max_length', '?'))
                    size = meta.get('size', '?')
                    desc = meta.get('description', '')[:48]

                    # Highlight recommended models
                    if lang_recommended and model_id in lang_recommended[:10]:
                        all_models_table.add_row(
                            f"[bold green]{idx}[/bold green]",
                            f"[bold green]{model_id}[/bold green]",
                            langs,
                            max_len,
                            size,
                            desc
                        )
                    else:
                        all_models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

                self.console.print(all_models_table)

                self.console.print(f"\n[dim]Tip: [bold green]Green models[/bold green] are in your Top 10 recommendations for {lang}[/dim]")
                self.console.print(f"\n[bold yellow]Select a model for {lang}:[/bold yellow]")
                self.console.print(f"[dim]  • Enter the # number from the table[/dim]")
                self.console.print(f"[dim]  • Or enter the model ID directly[/dim]")

                model_input_after_all = Prompt.ask(f"\nModel for {lang}", default=default_model)

                if model_input_after_all.isdigit():
                    idx = int(model_input_after_all) - 1
                    if 0 <= idx < len(sorted_model_ids):
                        lang_model = sorted_model_ids[idx]
                        self.console.print(f"[green]Selected for {lang}: {lang_model}[/green]")
                    else:
                        self.console.print(f"[yellow][!] Invalid selection. Using default: {default_model}[/yellow]")
                        lang_model = default_model
                else:
                    lang_model = model_input_after_all
            elif model_input.isdigit():
                idx = int(model_input) - 1
                if lang_recommended and 0 <= idx < len(lang_recommended):
                    lang_model = lang_recommended[idx]
                    self.console.print(f"[green]Selected for {lang}: {lang_model}[/green]")
                else:
                    self.console.print(f"[yellow][!] Invalid selection. Using default: {default_model}[/yellow]")
                    lang_model = default_model
            else:
                lang_model = model_input

            # Display full model details after selection
            self._display_model_details(lang_model, MODEL_METADATA)

            models_by_language[lang] = lang_model

        # Show summary of selected models
        self.console.print(f"\n[bold green]✓ Model Selection Complete:[/bold green]")
        for lang, model in sorted(models_by_language.items()):
            self.console.print(f"  • {lang}: [cyan]{model}[/cyan]")

        # For compatibility with rest of code, use first model as primary
        model_name = list(models_by_language.values())[0]

    elif not enable_benchmark:
        # Single model selection (multilingual or single language)
        # Display context
        strategy_desc = "Long-document models" if prefers_long_models else "Standard models (512 tokens max)"

        # Determine which languages to use for recommendations
        # If multilingual strategy was chosen, only show multilingual models
        if model_strategy == "multilingual" and len(languages) > 1:
            # User chose multilingual model - only show multilingual models
            languages_for_recommendation = {'MULTI'}
        else:
            # Per-language or single language - show language-specific models
            languages_for_recommendation = languages

        # Get intelligent recommendations using utility function
        recommended_models_list = get_recommended_models(
            languages=languages_for_recommendation,
            avg_text_length=text_length_avg,
            requires_long_model=prefers_long_models,
            top_n=10
        )

        if recommended_models_list:
            # Show top 10 with detailed characteristics
            self.console.print("[bold cyan]Top 10 Recommended Models:[/bold cyan]\n")

            models_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
            models_table.add_column("#", style="yellow", width=3, no_wrap=True)
            models_table.add_column("Model ID", style="cyan", no_wrap=True)
            models_table.add_column("Languages", style="green", no_wrap=True)
            models_table.add_column("Max Tokens", style="blue", no_wrap=True)
            models_table.add_column("Size", style="magenta", no_wrap=True)
            models_table.add_column("Description", style="white", overflow="fold", ratio=1)

            for idx, model_id in enumerate(recommended_models_list[:10], 1):
                meta = MODEL_METADATA.get(model_id, {})
                from llm_tool.utils.model_display import format_language_display
                langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                max_len = str(meta.get('max_length', '?'))
                size = meta.get('size', '?')
                desc = meta.get('description', '')[:43]

                models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

            self.console.print(models_table)
            default_model = recommended_models_list[0]
        else:
            if 'FR' in languages:
                default_model = 'camembert-base'
            elif 'EN' in languages:
                default_model = 'bert-base-uncased'
            else:
                default_model = 'xlm-roberta-base'

        # Use preloaded model if available
        if preloaded_params and preloaded_params.get('model_name'):
            default_model = preloaded_params['model_name']

        # Offer to display all models
        self.console.print(f"\n[dim]Tip: Selection Options:[/dim]")
        self.console.print(f"[dim]  • Enter [cyan]1-10[/cyan] to select from Top 10 recommendations[/dim]")
        self.console.print(f"[dim]  • Enter [cyan]'info X'[/cyan] (e.g., 'info 1') to see full details of a model[/dim]")
        self.console.print(f"[dim]  • Enter [cyan]'all'[/cyan] to see ALL {len(MODEL_METADATA)} available models with complete characteristics[/dim]")
        self.console.print(f"[dim]  • Enter any [cyan]HuggingFace model ID[/cyan] directly (e.g., 'bert-base-multilingual-cased')[/dim]")

        model_input = Prompt.ask("\n[bold yellow]Model to train[/bold yellow]", default=default_model)

        # Check if user wants info on a model
        if model_input.lower().startswith('info '):
            info_target = model_input[5:].strip()
            if info_target.isdigit():
                info_idx = int(info_target) - 1
                if recommended_models_list and 0 <= info_idx < len(recommended_models_list):
                    self._display_model_details(recommended_models_list[info_idx], MODEL_METADATA)
                else:
                    self.console.print(f"[red]Invalid model number: {info_target}[/red]")
            else:
                self._display_model_details(info_target, MODEL_METADATA)
            # After showing info, ask again for selection
            model_input = Prompt.ask("\n[bold yellow]Model to train[/bold yellow]", default=default_model)

        # Check if user wants to see all models
        if model_input.lower() == 'all':
            # Show ALL models with complete characteristics
            self.console.print(f"\n[bold cyan]ALL {len(MODEL_METADATA)} Available Models:[/bold cyan]\n")

            all_models_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED, expand=True)
            all_models_table.add_column("#", style="yellow", width=4, no_wrap=True)
            all_models_table.add_column("Model ID", style="cyan", no_wrap=True)
            all_models_table.add_column("Languages", style="green", no_wrap=True)
            all_models_table.add_column("Max Tokens", style="blue", no_wrap=True)
            all_models_table.add_column("Size", style="magenta", no_wrap=True)
            all_models_table.add_column("Description", style="white", overflow="fold", ratio=1)

            # Sort models: recommended first, then by relevance
            all_model_ids = list(MODEL_METADATA.keys())
            # Put recommended models at the top
            sorted_model_ids = []
            for model_id in recommended_models_list:
                if model_id in all_model_ids:
                    sorted_model_ids.append(model_id)
            # Add remaining models
            for model_id in all_model_ids:
                if model_id not in sorted_model_ids:
                    sorted_model_ids.append(model_id)

            for idx, model_id in enumerate(sorted_model_ids, 1):
                meta = MODEL_METADATA.get(model_id, {})
                from llm_tool.utils.model_display import format_language_display
                langs = format_language_display(meta.get('languages', ['?']), max_width=15)
                max_len = str(meta.get('max_length', '?'))
                size = meta.get('size', '?')
                desc = meta.get('description', '')[:48]

                # Highlight recommended models
                if model_id in recommended_models_list[:10]:
                    all_models_table.add_row(
                        f"[bold green]{idx}[/bold green]",
                        f"[bold green]{model_id}[/bold green]",
                        langs,
                        max_len,
                        size,
                        desc
                    )
                else:
                    all_models_table.add_row(str(idx), model_id, langs, max_len, size, desc)

            self.console.print(all_models_table)

            self.console.print(f"\n[dim]Tip: [bold green]Green models[/bold green] are in your Top 10 recommendations[/dim]")
            self.console.print(f"\n[bold yellow]Select a model:[/bold yellow]")
            self.console.print(f"[dim]  • Enter the # number from the table[/dim]")
            self.console.print(f"[dim]  • Or enter the model ID directly[/dim]")

            model_input_after_all = Prompt.ask("\nModel to train", default=default_model)

            if model_input_after_all.isdigit():
                idx = int(model_input_after_all) - 1
                if 0 <= idx < len(sorted_model_ids):
                    model_name = sorted_model_ids[idx]
                    self.console.print(f"[green]Selected: {model_name}[/green]")
                else:
                    self.console.print(f"[yellow][!] Invalid selection. Using default: {default_model}[/yellow]")
                    model_name = default_model
            else:
                model_name = model_input_after_all
        elif model_input.isdigit():
            idx = int(model_input) - 1
            if 0 <= idx < len(recommended_models_list):
                model_name = recommended_models_list[idx]
                self.console.print(f"[green]Selected: {model_name}[/green]")
            else:
                self.console.print(f"[yellow][!] Invalid selection. Using default: {default_model}[/yellow]")
                model_name = default_model
        else:
            model_name = model_input

        # Display full model details after selection
        self._display_model_details(model_name, MODEL_METADATA)

    # Capture stage-specific model selection for multi-class stage
    stage_models['multi_class'] = {
        'model_name': model_name,
        'models_by_language': dict(models_by_language) if models_by_language else None,
        'train_by_language': bool(models_by_language)
    }

    # Only one-vs-all keys require separate binary model selection
    # Multi-label keys use the same model as multi-class (just different activation)
    binary_stage_required = training_approach in ('hybrid', 'custom') and bool(onevsall_keys)
    onevsall_model_name = None
    onevsall_models_by_language: Optional[Dict[str, str]] = None

    if binary_stage_required:
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]           Hybrid Stage: One-vs-all Binary Model Selection[/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        reuse_for_binary = Confirm.ask(
            "[bold yellow]Reuse the same model selection for one-vs-all binary classifiers?[/bold yellow]",
            default=True
        )

        if reuse_for_binary:
            onevsall_model_name = model_name
            if models_by_language:
                onevsall_models_by_language = dict(models_by_language)
        else:
            if train_by_language and models_by_language:
                onevsall_models_by_language = {}
                for lang in sorted(languages):
                    self.console.print(f"\n[bold yellow]{'─'*60}[/bold yellow]")
                    self.console.print(f"[bold yellow]Selecting binary model for {lang} (one-vs-all stage)[/bold yellow]")
                    self.console.print(f"[bold yellow]{'─'*60}[/bold yellow]\n")

                    lang_recommended = get_recommended_models(
                        languages={lang},
                        avg_text_length=text_length_avg,
                        requires_long_model=prefers_long_models,
                        top_n=10
                    )

                    default_model = models_by_language.get(lang) if models_by_language else None
                    if not default_model:
                        if lang == 'FR':
                            default_model = 'camembert-base'
                        elif lang == 'EN':
                            default_model = 'bert-base-uncased'
                        else:
                            default_model = 'xlm-roberta-base'

                    selected_model = self._prompt_model_choice_for_stage(
                        lang=lang,
                        stage_label="one-vs-all",
                        recommended_models=lang_recommended,
                        default_model=default_model
                    )
                    onevsall_models_by_language[lang] = selected_model

                onevsall_model_name = list(onevsall_models_by_language.values())[0]
            else:
                # Single model scenario
                languages_for_recommendation = languages if languages else {'MULTI'}
                binary_recommended = get_recommended_models(
                    languages=languages_for_recommendation,
                    avg_text_length=text_length_avg,
                    requires_long_model=prefers_long_models,
                    top_n=10
                )

                default_binary_model = model_name
                onevsall_model_name = self._prompt_global_model_choice_for_stage(
                    stage_label="one-vs-all",
                    recommended_models=binary_recommended,
                    default_model=default_binary_model
                )
                onevsall_models_by_language = None

        stage_models['one_vs_all'] = {
            'model_name': onevsall_model_name or model_name,
            'models_by_language': dict(onevsall_models_by_language) if onevsall_models_by_language else None,
            'train_by_language': bool(onevsall_models_by_language)
        }

    # ============ TOKENIZER-AWARE MAX_LENGTH OPTIMIZATION ============
    # Goal: replace the default 512 with the smallest power-of-two bucket that
    # still fits every text, using the *real* tokenizer of the selected model.
    # Robustness rules (do NOT silently degrade):
    #   - skip entirely if user picked a long-document model (max_length is the point)
    #   - skip if per-language training (multiple tokenizers, out of scope here)
    #   - skip if split_long_texts: chunking changes the effective length distribution
    #     (not implemented in this codebase anyway — a guard upstream raises)
    #   - skip if texts cannot be loaded
    #   - skip if tokenizer cannot be loaded -> never offer a wrong shortcut
    # When exclude_long_texts is on, we still run the analysis but on the
    # *kept* subset (simulating the load-time filter) so the recommendation
    # reflects what the trainer will actually see.
    optimized_max_length: Optional[int] = None
    token_analysis_result: Optional[Dict[str, Any]] = None
    skip_token_analysis = (
        prefers_long_models
        or train_by_language
        or split_long_texts
        or not hasattr(bundle, 'primary_file')
        or bundle.primary_file is None
    )
    if not skip_token_analysis:
        try:
            from llm_tool.utils.token_analysis import analyze_tokens_with_model_tokenizer
            import pandas as _pd

            primary_path = Path(bundle.primary_file) if not isinstance(bundle.primary_file, Path) else bundle.primary_file
            text_col = getattr(bundle, 'text_column', None) or 'text'
            texts_for_analysis: List[str] = []
            if primary_path.exists():
                suffix = primary_path.suffix.lower()
                if suffix == '.csv':
                    df_texts = _pd.read_csv(primary_path, usecols=[text_col])
                elif suffix in ('.jsonl', '.json'):
                    df_texts = _pd.read_json(primary_path, lines=suffix == '.jsonl')
                else:
                    df_texts = None
                if df_texts is not None and text_col in df_texts.columns:
                    texts_for_analysis = [
                        t for t in df_texts[text_col].dropna().astype(str).tolist()
                        if t.strip()
                    ]

            if texts_for_analysis:
                self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
                self.console.print(f"[bold cyan]           Sequence Length Optimization                       [/bold cyan]")
                self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
                self.console.print(
                    f"[dim]Tokenizing {len(texts_for_analysis):,} texts with the actual tokenizer of "
                    f"[cyan]{model_name}[/cyan] to find the smallest safe max_length...[/dim]\n"
                )

                with self.console.status("[cyan]Running tokenizer-aware analysis...[/cyan]", spinner="dots"):
                    token_analysis_result = analyze_tokens_with_model_tokenizer(
                        texts_for_analysis,
                        model_name,
                        logger=self.logger,
                        # When the user opted into Exclude, simulate it so the
                        # recommendation reflects the kept subset (<=512 tokens).
                        simulate_exclude_above=512 if exclude_long_texts else None,
                    )

            if token_analysis_result:
                recommended = token_analysis_result['recommended_max_length']
                token_max = token_analysis_result['token_max']
                token_p95 = token_analysis_result['token_p95']
                token_p99 = token_analysis_result['token_p99']
                n_analyzed = token_analysis_result['texts_analyzed']
                buckets = token_analysis_result['distribution_buckets']
                n_above_512 = token_analysis_result['n_above_512']
                n_excluded_sim = token_analysis_result.get('n_excluded_by_simulation', 0)
                if exclude_long_texts and n_excluded_sim > 0:
                    pct_excluded = 100.0 * n_excluded_sim / (n_analyzed + n_excluded_sim)
                    self.console.print(
                        f"[dim]Simulating Exclude strategy: "
                        f"{n_excluded_sim:,} text(s) above 512 tokens ignored "
                        f"({pct_excluded:.2f}%). Stats below are on the kept subset.[/dim]\n"
                    )

                # Case 1: some texts already exceed 512 — no safe reduction available.
                # Surface the finding so the user knows the upstream "all docs fit in 512"
                # estimate was based on a different tokenizer, but do NOT offer to change
                # max_length: any reduction would silently truncate real data.
                if token_max > 512:
                    self.console.print(
                        f"[yellow][!] With the actual tokenizer of {model_name}, "
                        f"{n_above_512:,}/{n_analyzed:,} texts exceed 512 tokens "
                        f"(max={token_max}, p99={token_p99:.0f}).[/yellow]"
                    )
                    self.console.print(
                        "[dim]No safe reduction possible. Keeping max_length=512 "
                        "(those texts will be truncated by the model). To preserve "
                        "them, re-run and pick split-into-chunks or a long-document "
                        "model at the Token Length Strategy step.[/dim]\n"
                    )
                # Case 2: optimization is meaningful (recommended bucket strictly < 512).
                elif recommended < 512:
                    speedup = (512.0 / recommended) ** 2  # attention is quadratic in seq length
                    stats_table = Table(show_header=True, header_style="bold magenta",
                                        border_style="cyan", box=box.SIMPLE, expand=True)
                    stats_table.add_column("Metric", style="cyan", no_wrap=True)
                    stats_table.add_column("Value", style="white")
                    stats_table.add_row("Texts analyzed", f"{n_analyzed:,}")
                    stats_table.add_row("Token max (p100)", f"{token_max}")
                    stats_table.add_row("Token p99", f"{token_p99:.0f}")
                    stats_table.add_row("Token p95", f"{token_p95:.0f}")
                    stats_table.add_row(
                        "Distribution",
                        ", ".join(f"{k}: {v:,}" for k, v in buckets.items()),
                    )
                    stats_table.add_row(
                        "[bold]Recommended max_length[/bold]",
                        f"[bold green]{recommended}[/bold green] "
                        f"(vs. default 512 — zero truncation guaranteed)",
                    )
                    stats_table.add_row(
                        "Estimated speedup",
                        f"[green]~{speedup:.1f}× faster[/green] "
                        f"[dim](attention is O(n²))[/dim]",
                    )
                    self.console.print(stats_table)
                    self.console.print()

                    use_optimized = Confirm.ask(
                        f"[bold yellow]Use max_length={recommended} (recommended) instead of 512?[/bold yellow]",
                        default=True,
                    )
                    if use_optimized:
                        optimized_max_length = recommended
                        self.console.print(
                            f"[green]✓ max_length set to {recommended}. Training will be ~{speedup:.1f}× faster.[/green]\n"
                        )
                    else:
                        self.console.print("[dim]Keeping default max_length=512.[/dim]\n")
                else:
                    # recommended == 512 (or capped to 512). Make it explicit that
                    # this is only about the max_length knob and that any user-
                    # selected Exclude/Split strategy from STEP 1 is independent.
                    if exclude_long_texts and n_excluded_sim > 0:
                        self.console.print(
                            f"[dim]On the kept subset, max observed = {token_max} tokens. "
                            f"max_length=512 is the smallest safe bucket — no further "
                            f"reduction possible. The {n_excluded_sim:,} long text(s) flagged "
                            f"above will still be excluded by the Exclude strategy when "
                            f"training starts.[/dim]\n"
                        )
                    else:
                        self.console.print(
                            f"[dim]Token analysis: max observed = {token_max} tokens. "
                            f"max_length=512 is the smallest safe bucket — no further "
                            f"reduction possible.[/dim]\n"
                        )
        except Exception as exc:
            # Hard rule: never let this analysis break the training flow.
            # On any unexpected error, log and keep the default 512.
            self.logger.warning(
                "Tokenizer-aware max_length analysis failed (%s). Keeping default max_length=512.",
                exc,
            )

    # Multi-label threshold configuration (only if multi-label was selected in Training Approach)
    # The multi_label flag is already set in bundle.metadata during the Training Approach step
    if hasattr(bundle, 'metadata') and bundle.metadata and bundle.metadata.get('multi_label'):
        # Show confirmation that multi-label is enabled
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]            Multi-Label Classification: ENABLED               [/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        self.console.print("[dim]Multi-label training will use:[/dim]")
        self.console.print("  • Loss function: [cyan]BCEWithLogitsLoss[/cyan]")
        self.console.print("  • Activation: [cyan]Sigmoid[/cyan] (independent probabilities)")
        self.console.print("  • Output: [cyan]JSON array of labels[/cyan]\n")

        # Ask for threshold configuration
        current_threshold = bundle.metadata.get('multi_label_threshold', 0.5)
        configure_ml = Confirm.ask(
            f"[bold cyan]Configure multi-label threshold?[/bold cyan] [dim](current: {current_threshold})[/dim]",
            default=False
        )
        if configure_ml:
            threshold_input = Prompt.ask("Threshold (0.0-1.0)", default=str(current_threshold))
            try:
                multi_label_threshold = float(threshold_input)
                if multi_label_threshold < 0 or multi_label_threshold > 1:
                    self.console.print("[yellow][!] Threshold must be between 0 and 1. Using default 0.5[/yellow]")
                    multi_label_threshold = 0.5
            except ValueError:
                self.console.print("[yellow][!] Invalid input. Using default 0.5[/yellow]")
                multi_label_threshold = 0.5
            bundle.metadata['multi_label_threshold'] = multi_label_threshold

    # Reinforced learning
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    rl_step_label = resolve_step_label("reinforced_learning", "STEP 5", context=step_context)
    self.console.print(f"[bold cyan]           {rl_step_label}: Reinforced Learning                      [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    self.console.print("[bold]Training optimization options:[/bold]\n")

    self.console.print("[bold cyan]  Phase 1 - Distribution-aware initial training[/bold cyan]")
    self.console.print("  Modifies the loss function from the very first epoch:")
    self.console.print("    - pos_weight: penalizes false negatives for rare labels")
    self.console.print("    - WeightedRandomSampler: rare labels drawn more frequently")
    self.console.print("    [yellow]Warning: can cause over-prediction if dataset has many 'empty' samples[/yellow]")
    self.console.print("    [yellow](e.g., 80%+ samples with no labels at all)[/yellow]\n")

    self.console.print("[bold cyan]  Phase 2 - Targeted reinforcement (additional epochs)[/bold cyan]")
    self.console.print("  After initial training, per-label F1 is evaluated:")
    self.console.print("    - Labels performing well are [bold]frozen[/bold] (protected from forgetting)")
    self.console.print("    - Underperforming labels get boosted weights + extra epochs")
    self.console.print("    - Works with or without Phase 1\n")

    self.console.print("[yellow]Note:[/yellow] [dim]Phase 1 and Phase 2 are independent. You can use either, both, or neither.[/dim]\n")

    # Use preloaded value if available
    default_reinforced = preloaded_params.get('reinforced_learning', False) if preloaded_params else False

    # Phase 1: Distribution-aware initial training
    enable_phase1 = Confirm.ask(
        "[bold yellow]Enable Phase 1 (distribution-aware loss + weighted sampling)?[/bold yellow]\n"
        "[dim]Say 'n' if your dataset has many empty samples (no labels). Say 'y' for moderate imbalance.[/dim]",
        default=default_reinforced
    )

    # ---- NORMAL-phase imbalance handling (SOTA loss selection) -----------------
    # Configurable, data-adaptive class-imbalance handling for the *normal*
    # training phase. Works for every training type (multi-class, multi-label,
    # binary, one-vs-all). Independent of Phase 2 (reinforced learning).
    imbalance_strategy = None
    focal_gamma = 2.0
    imbalance_weight_source = 'auto'
    imbalance_class_weights = None
    imbalance_weighted_sampler = True
    if enable_phase1:
        _preloaded_strategy = (preloaded_params or {}).get('imbalance_strategy') if preloaded_params else None
        self.console.print()
        self.console.print("[bold]Imbalance-handling strategy[/bold] [dim](normal training loss)[/dim]")
        self.console.print("   • [green]auto[/green]: pick from the data distribution [dim](recommended)[/dim]")
        self.console.print("   • [cyan]weighted[/cyan]: class-weighted CrossEntropy / pos_weight BCE")
        self.console.print("   • [cyan]focal[/cyan]: Focal Loss (down-weights easy examples; best for severe imbalance)")
        self.console.print("   • [cyan]asymmetric[/cyan]: Asymmetric Loss (multi-label; falls back to focal otherwise)")
        self.console.print("   • [yellow]none[/yellow]: plain loss (no imbalance handling)\n")
        imbalance_strategy = Prompt.ask(
            "Strategy",
            choices=["auto", "weighted", "focal", "asymmetric", "none"],
            default=str(_preloaded_strategy) if _preloaded_strategy else "auto",
        )

        if imbalance_strategy in ("focal", "asymmetric"):
            gamma_input = Prompt.ask(
                "[bold]Focal gamma[/bold] [dim](focusing strength; 2.0 recommended, higher = more focus on hard examples)[/dim]",
                default="2.0",
            )
            try:
                focal_gamma = float(gamma_input)
                if focal_gamma < 0:
                    focal_gamma = 2.0
            except ValueError:
                self.console.print("[yellow][!] Invalid gamma. Using 2.0[/yellow]")
                focal_gamma = 2.0

        if imbalance_strategy in ("weighted", "focal", "asymmetric"):
            use_auto_weights = Confirm.ask(
                "[bold]Compute class weights automatically from the data?[/bold]\n"
                "[dim](Choose 'n' to enter weights manually)[/dim]",
                default=True,
            )
            imbalance_weight_source = 'auto' if use_auto_weights else 'manual'
            if not use_auto_weights:
                raw = Prompt.ask(
                    "[bold]Manual per-class weights[/bold] [dim](comma-separated, e.g. '1.0,5.0')[/dim]",
                    default="",
                )
                try:
                    parsed = [float(x) for x in raw.split(",") if x.strip() != ""]
                    imbalance_class_weights = parsed if parsed else None
                except ValueError:
                    self.console.print("[yellow][!] Could not parse weights; falling back to auto[/yellow]")
                    imbalance_weight_source = 'auto'

            imbalance_weighted_sampler = Confirm.ask(
                "[bold]Use a WeightedRandomSampler (oversample rare classes)?[/bold]",
                default=True,
            )

        if imbalance_strategy == "none":
            self.console.print("[green]  Imbalance handling: plain loss (none)[/green]\n")
        else:
            self.console.print(
                f"[green]  Imbalance handling: strategy={imbalance_strategy}, "
                f"weights={imbalance_weight_source}, sampler={imbalance_weighted_sampler}[/green]\n"
            )

    # ---- Advanced optimizer hyperparameters (learning rate, warmup) ------------
    # Optional, opt-in. Defaults reproduce legacy behaviour exactly (lr 2e-5,
    # warmup 0.0 = no warmup). Applies to EVERY supported model. Lowering the LR
    # and adding warmup stabilises fine-tuning of hard/small categories (cures the
    # all-negative <-> all-positive oscillation seen on difficult themes).
    learning_rate = (preloaded_params or {}).get('learning_rate', 2e-5) if preloaded_params else 2e-5
    warmup_ratio = (preloaded_params or {}).get('warmup_ratio', 0.0) if preloaded_params else 0.0
    configure_optim = Confirm.ask(
        "\n[bold cyan]Configure advanced optimizer hyperparameters (learning rate, warmup)?[/bold cyan]\n"
        "[dim](Choose 'n' to use defaults: lr=2e-5, warmup=0.0 — unchanged legacy behaviour)[/dim]",
        default=False,
    )
    if configure_optim:
        self.console.print("\n[bold]Learning rate[/bold]")
        self.console.print("   [dim]Lower it (e.g. 1e-5) to stabilise hard/small categories that oscillate.[/dim]")
        self.console.print("   • Default: [green]2e-5[/green]  • Conservative: [yellow]1e-5[/yellow]  • Very conservative: [yellow]5e-6[/yellow]\n")
        lr_input = Prompt.ask("Learning rate", default="2e-5")
        try:
            learning_rate = float(lr_input)
            if not (0 < learning_rate < 1):
                self.console.print("[yellow][!] Out of range. Using 2e-5[/yellow]"); learning_rate = 2e-5
        except ValueError:
            self.console.print("[yellow][!] Invalid. Using 2e-5[/yellow]"); learning_rate = 2e-5

        self.console.print("\n[bold]Warmup ratio[/bold]")
        self.console.print("   [dim]Fraction of training spent ramping the LR from 0. 0.0=off; 0.1–0.2 helps stability.[/dim]\n")
        wu_input = Prompt.ask("Warmup ratio", default="0.0")
        try:
            warmup_ratio = float(wu_input)
            if not (0 <= warmup_ratio < 1):
                self.console.print("[yellow][!] Out of range. Using 0.0[/yellow]"); warmup_ratio = 0.0
        except ValueError:
            self.console.print("[yellow][!] Invalid. Using 0.0[/yellow]"); warmup_ratio = 0.0
        self.console.print(f"[green]  Optimizer: lr={learning_rate:g}, warmup_ratio={warmup_ratio:g}[/green]\n")

    # Phase 2: Reinforced learning
    self.console.print()
    phase2_choice = Prompt.ask(
        "[bold yellow]Phase 2 (reinforcement after initial training)[/bold yellow]",
        choices=["always", "auto", "off"],
        default="auto"
    )

    # Default reinforced learning parameters
    rl_f1_threshold = 0.70
    rl_oversample_factor = 2.0
    rl_class_weight_factor = 2.0
    manual_rl_epochs = None
    force_reinforced = False

    # Set flags based on choices
    enable_reinforced_learning = phase2_choice != "off"
    if phase2_choice == "always":
        force_reinforced = True
        self.console.print("[green]  Phase 2 will always run after initial training.[/green]\n")
    elif phase2_choice == "auto":
        self.console.print(f"[green]  Phase 2 triggers if any label F1 < 0.70.[/green]\n")
    else:
        self.console.print("[green]  Phase 2 disabled. Standard training only.[/green]\n")

    # Only ask for manual config if Phase 2 is active
    configure_rl = False
    if phase2_choice != "off":
        configure_rl = Confirm.ask(
            "\n[bold cyan]Configure reinforced learning parameters manually?[/bold cyan]\n"
            "[dim](Choose 'n' to use recommended defaults)[/dim]",
            default=False
        )

        if configure_rl:
            self.console.print("\n[bold green] Manual Configuration[/bold green]\n")

            # F1 Threshold
            self.console.print("[bold]F1 Activation Threshold[/bold]")
            self.console.print("   [dim]When any label's F1-score drops below this value, reinforced learning activates for that label[/dim]")
            self.console.print("   [dim]Other parameters (gamma, weights, epochs) are computed automatically from your data distribution[/dim]")
            self.console.print("   • Recommended: [green]0.70[/green] (moderate)")
            self.console.print("   • Conservative: [yellow]0.50[/yellow] (only very poor labels)")
            self.console.print("   • Aggressive: [yellow]0.85[/yellow] (triggers early)\n")

            f1_input = Prompt.ask(
                "F1 threshold",
                default="0.70"
            )
            try:
                rl_f1_threshold = float(f1_input)
                if rl_f1_threshold < 0 or rl_f1_threshold > 1:
                    self.console.print("[yellow][!] F1 must be between 0 and 1. Using default 0.70[/yellow]")
                    rl_f1_threshold = 0.70
            except ValueError:
                self.console.print("[yellow][!] Invalid input. Using default 0.70[/yellow]")
                rl_f1_threshold = 0.70

            # Reinforced Epochs
            self.console.print("\n[bold]Reinforced Learning Epochs[/bold]")
            self.console.print("   [dim]Number of additional epochs when labels underperform[/dim]")
            self.console.print("   • Default: [green]Auto-calculated[/green] (8-15 epochs based on severity)")
            self.console.print("   • Manual: [yellow]Choose fixed number[/yellow]\n")

            use_auto_epochs = Confirm.ask(
                "Use auto-calculated epochs?",
                default=True
            )

            manual_rl_epochs = None
            if not use_auto_epochs:
                manual_rl_epochs = IntPrompt.ask(
                    "[bold yellow]Reinforced epochs[/bold yellow]",
                    default=10
                )

            # Summary
            self.console.print("\n[bold green]Reinforced Learning Configuration:[/bold green]")
            self.console.print(f"  • F1 Threshold: [cyan]{rl_f1_threshold:.2f}[/cyan]")
            self.console.print(f"  • Loss weights: [cyan]Auto (distribution-aware)[/cyan]")
            self.console.print(f"  • Sampling: [cyan]WeightedRandomSampler (from initial training)[/cyan]")
            if manual_rl_epochs:
                self.console.print(f"  • Reinforced Epochs: [cyan]{manual_rl_epochs}[/cyan] (manual)")
            else:
                self.console.print(f"  • Reinforced Epochs: [cyan]Auto (8-15 based on severity)[/cyan]")
            self.console.print()
        else:
            self.console.print("\n[green]Reinforced learning enabled (distribution-aware, F1 threshold=0.70)[/green]\n")

            # Ask if user wants to configure RL epochs manually
            configure_rl_epochs = Confirm.ask(
                "[bold yellow]Configure reinforced learning epochs manually?[/bold yellow]\n"
                "[dim](Default: auto-calculated based on label severity)[/dim]",
                default=False
            )

            if configure_rl_epochs:
                manual_rl_epochs = IntPrompt.ask(
                    "[bold yellow]Reinforced epochs[/bold yellow]",
                    default=10
                )

                self.console.print(f"[green]Manual reinforced epochs set to: {manual_rl_epochs}[/green]\n")
            else:
                self.console.print("[green]Reinforced learning epochs will be auto-calculated[/green]\n")
                manual_rl_epochs = None

    # Epoch configuration
    self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    epochs_step_label = resolve_step_label("epochs", "STEP 5", context=step_context)
    self.console.print(f"[bold cyan]           {epochs_step_label}: Training Epochs                           [/bold cyan]")
    self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    self.console.print("[bold]What are Epochs?[/bold]")
    self.console.print("  • [cyan]One epoch[/cyan] = One complete pass through your entire training dataset")
    self.console.print("  • [cyan]More epochs[/cyan] = Model sees and learns from data more times")
    self.console.print("  • [cyan]Typical range[/cyan]: 3-15 epochs for BERT-like models\n")

    self.console.print("[bold]Guidelines:[/bold]")
    self.console.print("  • [green]Small dataset (<1000 samples)[/green]: 10-15 epochs recommended")
    self.console.print("  • [green]Medium dataset (1000-10000)[/green]: 5-10 epochs recommended")
    self.console.print("  • [green]Large dataset (>10000)[/green]: 3-5 epochs recommended\n")

    self.console.print("[bold green]Automatic Best Model Checkpointing:[/bold green]")
    self.console.print("  • [cyan]Don't worry about setting too many epochs![/cyan]")
    self.console.print("  • The [bold]BEST model[/bold] is automatically saved during training")
    self.console.print("  • System monitors [yellow]validation F1 score[/yellow] after each epoch")
    self.console.print("  • Only the checkpoint with [bold green]highest F1[/bold green] is kept")
    self.console.print("  • Early stopping prevents overfitting automatically\n")

    self.console.print("[dim]Tip: Example: You set 15 epochs, but best F1 was at epoch 8 → Model from epoch 8 is used[/dim]\n")

    # Use preloaded value if available
    default_epochs = preloaded_params.get('epochs', 10) if preloaded_params else 10

    epochs = IntPrompt.ask("[bold yellow]Number of epochs[/bold yellow]", default=default_epochs)

    # ═══════════════════════════════════════════════════════
    #            STEP 6: Early Stopping & Interactive Skip
    # ═══════════════════════════════════════════════════════
    self.console.print(f"\n{'═'*60}")
    self.console.print(f"{'           STEP 6: Training Control':^60}")
    self.console.print(f"{'═'*60}\n")

    # Early stopping
    self.console.print("[bold]Early Stopping[/bold]")
    self.console.print("  Automatically stop training a model if F1 stops improving.")
    self.console.print("  Best model checkpoint is always preserved.\n")
    es_enabled = Confirm.ask("Enable early stopping?", default=True)
    early_stopping_patience = None
    if es_enabled:
        early_stopping_patience = IntPrompt.ask("  Patience (epochs without improvement)", default=3)
        self.console.print(f"  [green]✓[/green] Early stopping: stop after {early_stopping_patience} epochs without improvement\n")

    # Interactive skip
    self.console.print("[bold]Interactive Skip[/bold]")
    self.console.print("  Type [bold cyan]s[/bold cyan] + Enter during training to skip to next model.")
    self.console.print("  Works for both normal training and reinforced learning.\n")
    interactive_skip = Confirm.ask("Enable interactive skip (manual model skip)?", default=True)
    if interactive_skip:
        self.console.print("  [green]✓[/green] Interactive skip enabled: type [bold cyan]s[/bold cyan] + Enter to skip a model\n")
    else:
        self.console.print("  [dim]Interactive skip disabled[/dim]\n")

    # distribution_aware = Phase 1 (pos_weight + WeightedRandomSampler)
    # reinforced_learning = Phase 2 (extra epochs for underperforming labels)
    # These are now INDEPENDENT choices
    distribution_aware_enabled = enable_phase1

    # Prepare return dict
    result = {
        'model_name': model_name,
        'reinforced_learning': enable_reinforced_learning,
        'epochs': epochs,
        'early_stopping_patience': early_stopping_patience,
        'interactive_skip': interactive_skip,
        # Reinforced learning parameters
        'rl_f1_threshold': rl_f1_threshold,
        'rl_oversample_factor': rl_oversample_factor,
        'rl_class_weight_factor': rl_class_weight_factor,
        'manual_rl_epochs': manual_rl_epochs if manual_rl_epochs else None,
        'force_reinforced': force_reinforced,
        'distribution_aware': distribution_aware_enabled,
        # NORMAL-phase SOTA imbalance handling
        'imbalance_strategy': imbalance_strategy,
        'focal_gamma': focal_gamma,
        'imbalance_weight_source': imbalance_weight_source,
        'imbalance_class_weights': imbalance_class_weights,
        'imbalance_weighted_sampler': imbalance_weighted_sampler,
        # Advanced optimizer hyperparameters
        'learning_rate': learning_rate,
        'warmup_ratio': warmup_ratio,
        # Tokenizer-aware max_length: only set when user opted in. None means
        # downstream code should keep its default (currently 512).
        'max_length': optimized_max_length,
    }

    # Include models_by_language if training per-language
    if train_by_language and models_by_language:
        result['models_by_language'] = models_by_language
        result['train_by_language'] = True

    if stage_models:
        result['stage_models'] = stage_models

    return result

def _compute_stage_summary(
    trained_model_records: List[Dict[str, Any]],
    results_per_key: Dict[str, Dict[str, Any]],
    total_multiclass_models: int,
    total_onevsall_models: int,
    multiclass_needs_language_training: bool,
    binary_train_by_language: bool,
    stage_models: Optional[Dict[str, Dict[str, Any]]] = None,
    languages: Optional[Iterable[str]] = None,
    total_training_time: float = 0.0,
    avg_accuracy: float = 0.0,
    avg_f1_macro: float = 0.0,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Aggregate per-stage metrics for hybrid/custom training runs.

    Returns
    -------
    stage_summary : dict
        Summary keyed by stage identifier (multi_class / one_vs_all / overall)
    total_expected_models : int
        Total number of models expected across stages.
    """
    stage_models = stage_models or {}
    languages_list = sorted({str(lang).upper() for lang in languages}) if languages else None

    def _canonical_stage(name: str) -> str:
        if name == 'multi-class':
            return 'multi_class'
        if name == 'one-vs-all':
            return 'one_vs_all'
        return (name or 'unknown').replace('-', '_')

    stage_counts: Dict[str, int] = {}
    stage_f1_sum: Dict[str, float] = {}
    stage_acc_sum: Dict[str, float] = {}
    stage_best_f1: Dict[str, float] = {}
    stage_best_acc: Dict[str, float] = {}

    for record in trained_model_records:
        stage_key = _canonical_stage(record.get('stage', 'unknown'))
        stage_counts[stage_key] = stage_counts.get(stage_key, 0) + 1
        f1_val = float(record.get('f1', 0.0) or 0.0)
        acc_val = float(record.get('accuracy', 0.0) or 0.0)
        stage_f1_sum[stage_key] = stage_f1_sum.get(stage_key, 0.0) + f1_val
        stage_acc_sum[stage_key] = stage_acc_sum.get(stage_key, 0.0) + acc_val
        stage_best_f1[stage_key] = max(stage_best_f1.get(stage_key, 0.0), f1_val)
        stage_best_acc[stage_key] = max(stage_best_acc.get(stage_key, 0.0), acc_val)

    error_counts = {'multi_class': 0, 'one_vs_all': 0}
    for bucket in results_per_key.values():
        multi_result = bucket.get('multi_class')
        if isinstance(multi_result, dict) and multi_result.get('error'):
            error_counts['multi_class'] += 1
        one_vs_all_bucket = bucket.get('one_vs_all', {})
        if isinstance(one_vs_all_bucket, dict):
            for label_result in one_vs_all_bucket.values():
                if isinstance(label_result, dict) and label_result.get('error'):
                    error_counts['one_vs_all'] += 1

    expected_map = {
        'multi_class': int(total_multiclass_models),
        'one_vs_all': int(total_onevsall_models),
    }

    stage_summary: Dict[str, Dict[str, Any]] = {}
    for stage_key, expected in expected_map.items():
        trained = stage_counts.get(stage_key, 0)
        average_f1 = stage_f1_sum.get(stage_key, 0.0) / trained if trained else 0.0
        average_acc = stage_acc_sum.get(stage_key, 0.0) / trained if trained else 0.0
        stage_summary[stage_key] = {
            'expected_models': expected,
            'trained_models': trained,
            'failed_models': max(expected - trained, 0),
            'errors_recorded': error_counts.get(stage_key, 0),
            'average_f1_macro': average_f1,
            'average_accuracy': average_acc,
            'best_f1_macro': stage_best_f1.get(stage_key, 0.0),
            'best_accuracy': stage_best_acc.get(stage_key, 0.0),
            'train_by_language': bool(multiclass_needs_language_training) if stage_key == 'multi_class' else bool(binary_train_by_language),
            'model_selection': stage_models.get(stage_key),
        }

    total_expected = int(total_multiclass_models + total_onevsall_models)
    stage_summary['overall'] = {
        'expected_models': total_expected,
        'trained_models': sum(stage_counts.values()),
        'failed_models': max(total_expected - sum(stage_counts.values()), 0),
        'average_f1_macro': avg_f1_macro,
        'average_accuracy': avg_accuracy,
        'total_training_time_sec': total_training_time,
        'languages': languages_list,
    }

    return stage_summary, total_expected


def _training_studio_run_quick(self, bundle: TrainingDataBundle, model_config: Dict[str, Any], quick_params: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick training mode - simple and fast with sensible defaults.

    Args:
        bundle: Training data bundle
        model_config: Model configuration dict (will be updated with runtime params)
        quick_params: Pre-collected parameters (model_name, reinforced_learning, epochs)
        session_id: Session timestamp for organizing logs by session

    Returns:
        dict with keys: 'runtime_params', 'models_trained', 'best_model', 'best_f1'
    """
    self.console.print("\n[bold]Quick training[/bold] - using configured parameters.")

    session_manager = getattr(self, 'current_session_manager', None)
    session_metrics_dir = None
    if session_manager and getattr(session_manager, 'session_dir', None):
        session_metrics_dir = session_manager.session_dir / "training_metrics" / "normal_training"

    # CRITICAL: Log session management for debugging
    self.logger.info("="*80)
    self.logger.info("SESSION MANAGEMENT - FULL TRAINING")
    self.logger.info(f"  session_id (passed to function): {session_id}")
    self.logger.info(f"  self.current_session_id: {getattr(self, 'current_session_id', 'NOT SET')}")
    if session_id and hasattr(self, 'current_session_id') and session_id == self.current_session_id:
        self.logger.info("  ✓ Full training REUSING same session_id as benchmark")
        self.logger.info(f"  Models will be saved to: models/{session_id}/normal_training/")
        if session_metrics_dir is not None:
            self.logger.info(f"  Logs will be saved to: {session_metrics_dir}")
        else:
            fallback_metrics_dir = get_training_metrics_dir(session_id) / "normal_training"
            self.logger.info(f"  Logs will be saved to: {fallback_metrics_dir}")
    elif session_id:
        self.logger.warning("  [!]  session_id provided but differs from self.current_session_id")
        self.logger.info(f"  Models will be saved to: models/{session_id}/normal_training/")
    else:
        self.logger.warning("  [!]  No session_id provided - will create new one (BAD!)")
    self.logger.info("="*80)
    if session_id:
        self.console.print(f"\n[cyan]Session ID:[/cyan] [bold]{session_id}[/bold]")
        self.console.print(f"[dim]All trained models will be saved to: models/{session_id}/normal_training/[/dim]\n")
        if session_metrics_dir is not None:
            self.console.print(f"[dim]Training metrics will be saved to: {session_metrics_dir}[/dim]\n")
        else:
            fallback_metrics_dir = get_training_metrics_dir(session_id) / "normal_training"
            self.console.print(f"[dim]Training metrics will be saved to: {fallback_metrics_dir}[/dim]\n")

    if getattr(self, "_persistent_onevsall_root", None) is None and session_id:
        persistent_root = None
        try:
            persistent_root = get_training_data_dir(session_id) / "onevsall_datasets"
            persistent_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                "Could not prepare persistent directory for one-vs-all datasets during quick run (session=%s): %s",
                session_id,
                exc,
            )
        self._set_onevsall_storage_root(persistent_root)

    stage_models: Dict[str, Dict[str, Any]] = {}

    # Use parameters from quick_params (already collected before config summary)
    if quick_params:
        # CRITICAL: Debug log to capture exact type of models_by_language from quick_params
        self.logger.debug(f"quick_params keys: {quick_params.keys()}")
        if 'models_by_language' in quick_params:
            self.logger.debug(f"models_by_language type in quick_params: {type(quick_params['models_by_language'])}")
            self.logger.debug(f"models_by_language value in quick_params: {quick_params['models_by_language']}")

        base_model_name = quick_params['model_name']
        epochs = quick_params['epochs']
        enable_reinforced_learning = quick_params['reinforced_learning']
        stage_models_config = quick_params.get('stage_models') or {}
        if isinstance(stage_models_config, dict):
            stage_models = dict(stage_models_config)
        multiclass_stage_config = stage_models_config.get('multi_class') or {}
        binary_stage_config = stage_models_config.get('one_vs_all') or {}

        model_name = multiclass_stage_config.get('model_name') or base_model_name
        models_by_language = multiclass_stage_config.get('models_by_language')
        if models_by_language is None:
            models_by_language = quick_params.get('models_by_language')
        elif isinstance(models_by_language, dict):
            models_by_language = dict(models_by_language)

        train_by_language_flag = multiclass_stage_config.get('train_by_language')
        if train_by_language_flag is None:
            train_by_language_flag = bool(models_by_language) if models_by_language else quick_params.get('train_by_language', False)
        else:
            train_by_language_flag = bool(train_by_language_flag)

        binary_model_name = binary_stage_config.get('model_name') or model_name
        binary_models_by_language = binary_stage_config.get('models_by_language')
        if isinstance(binary_models_by_language, dict):
            binary_models_by_language = dict(binary_models_by_language)
        else:
            binary_models_by_language = None
        if binary_models_by_language is None and train_by_language_flag and models_by_language:
            binary_models_by_language = dict(models_by_language)

        binary_train_by_language = binary_stage_config.get('train_by_language')
        if binary_train_by_language is None:
            binary_train_by_language = bool(binary_models_by_language) or bool(quick_params.get('train_by_language', False))
        else:
            binary_train_by_language = bool(binary_train_by_language)

        manual_rl_epochs = quick_params.get('manual_rl_epochs', None)
        rl_f1_threshold = quick_params.get('rl_f1_threshold', 0.70)
    else:
        # Fallback for legacy resume mode
        # CRITICAL: Use 'or' to handle None values - dict.get() returns None if key exists with None value
        model_name = model_config.get('quick_model_name') or model_config.get('selected_model') or 'xlm-roberta-large'
        epochs = model_config.get('quick_epochs') or model_config.get('epochs') or 20
        enable_reinforced_learning = model_config.get('use_reinforcement', True)
        models_by_language = None
        train_by_language_flag = False
        stage_models_config = {}
        multiclass_stage_config = {}
        binary_stage_config = {}
        stage_models = {}
        binary_model_name = model_name
        binary_models_by_language = None
        binary_train_by_language = train_by_language_flag
        manual_rl_epochs = None
        rl_f1_threshold = 0.70

    mc_models_map: Optional[Dict[str, Any]] = dict(models_by_language) if isinstance(models_by_language, dict) else None
    binary_models_map: Optional[Dict[str, Any]] = dict(binary_models_by_language) if isinstance(binary_models_by_language, dict) else None

    if 'multi_class' not in stage_models and model_name:
        stage_models['multi_class'] = {
            'model_name': model_name,
            'models_by_language': mc_models_map,
            'train_by_language': bool(train_by_language_flag),
        }

    if 'one_vs_all' not in stage_models and binary_model_name:
        stage_models['one_vs_all'] = {
            'model_name': binary_model_name,
            'models_by_language': binary_models_map,
            'train_by_language': bool(binary_train_by_language),
        }
        binary_model_name = model_name
        binary_models_by_language = None
        binary_train_by_language = train_by_language_flag
        manual_rl_epochs = None
        rl_f1_threshold = 0.70

    # ============================================================
    # CRITICAL: Validate and filter insufficient labels BEFORE training
    # MUST happen AFTER extracting train_by_language_flag
    # ============================================================
    if bundle.primary_file:
        try:
            filtered_file, was_filtered = self._validate_and_filter_insufficient_labels(
                input_file=str(bundle.primary_file),
                strategy=bundle.strategy,
                min_samples=2,
                auto_remove=False,  # Ask user for confirmation
                train_by_language=train_by_language_flag  # CRITICAL: Language-aware validation
            )
            if was_filtered:
                # Update bundle to use filtered file
                bundle.primary_file = Path(filtered_file)
                self.console.print(f"[green]Using filtered training dataset[/green]\n")
        except ValueError as e:
            # User cancelled or validation failed
            self.console.print(f"[red]{e}[/red]")
            return {
                'runtime_params': {},
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'error': str(e)
            }
        except Exception as e:
            self.logger.warning(f"Label validation failed: {e}")
            # Continue with original file if validation fails
            pass

    # Display training configuration summary
    self.console.print()

    # CRITICAL: Validate models_by_language type before using len()
    if models_by_language and not isinstance(models_by_language, dict):
        self.console.print(f"[red][!]  ERROR: models_by_language has invalid type: {type(models_by_language)}[/red]")
        self.logger.error(f"models_by_language type error: {type(models_by_language)}, value: {models_by_language}")
        models_by_language = None  # Reset to None to prevent crash

    if models_by_language:
        self.console.print(f"  • Models: [cyan]{len(models_by_language)}[/cyan] (language-specific)")
    else:
        self.console.print(f"  • Model: [cyan]{model_name}[/cyan]")

    # Display epochs with reinforced learning info if enabled
    if enable_reinforced_learning:
        if manual_rl_epochs is not None:
            # Manual reinforced epochs configured
            max_epochs = epochs + manual_rl_epochs
            self.console.print(f"  • Epochs: [cyan]{epochs}[/cyan] (up to [yellow]{max_epochs}[/yellow] with reinforced learning)")
        else:
            # Auto-calculated reinforced epochs (typically 8-20)
            self.console.print(f"  • Epochs: [cyan]{epochs}[/cyan] (up to [yellow]{epochs}+auto[/yellow] with reinforced learning)")
        self.console.print(f"  • Reinforced learning: [cyan]Enabled[/cyan] (F1 < {rl_f1_threshold:.2f})")
    else:
        self.console.print(f"  • Epochs: [cyan]{epochs}[/cyan]")
    self.console.print()

    # Get languages from metadata (needed for training)
    languages = set()
    if hasattr(bundle, 'metadata') and bundle.metadata:
        languages = bundle.metadata.get('confirmed_languages', bundle.metadata.get('languages', set()))
    if not languages and hasattr(bundle, 'languages') and bundle.languages:
        languages = set([lang.upper() for lang in bundle.languages])
    if languages:
        languages = set([str(lang).upper() for lang in languages])

    # Capture runtime parameters for full reproducibility
    if models_by_language:
        # Per-language models selected
        runtime_params = {
            'quick_models_by_language': models_by_language,
            'quick_epochs': epochs,
            'reinforced_learning': enable_reinforced_learning,
            'actual_models_trained': list(models_by_language.values())
        }
    else:
        # Single model for all languages
        runtime_params = {
            'quick_model_name': model_name,
            'quick_epochs': epochs,
            'reinforced_learning': enable_reinforced_learning,
            'actual_models_trained': [model_name]
        }

    # CRITICAL: DO NOT create a new timestamped directory for Training Arena.
    # Models are saved using session_id which is passed in the config to bert_base.py.
    # The output_dir is only used as a fallback placeholder for save_model_as.
    # Real path: models/{session_id}/normal_training/{category}/{language}/{model}/
    # This ensures benchmark and full training use THE SAME session folder.
    output_dir = Path("models") / "placeholder_not_used"

    # Initialize multiclass_groups (will be set if detected)
    multiclass_groups = None

    # CRITICAL: Extract training_approach BEFORE the multi-label block so it's accessible later
    training_approach_from_metadata = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None

    # For multi-label OR single-label with multi-class approach, check if we need to detect groups
    # CRITICAL FIX: Also detect multiclass_groups for single-label strategy when user chose multi-class
    should_detect_multiclass = (
        bundle.strategy == "multi-label" or
        (bundle.strategy == "single-label" and training_approach_from_metadata == 'multi-class')
    )

    if should_detect_multiclass:
        # Use primary_file for one-vs-all, dataset_path otherwise
        data_path = str(bundle.primary_file) if hasattr(bundle, 'primary_file') and bundle.primary_file else None

        # CRITICAL: Only attempt multiclass detection for JSON/JSONL files
        # For CSV files, skip detection - train_single_model handles multi-class automatically via num_labels
        can_detect_multiclass = data_path and (data_path.endswith('.json') or data_path.endswith('.jsonl'))

        if can_detect_multiclass:
            # Load data to check structure
            from llm_tool.trainers.multi_label_trainer import MultiLabelTrainer, TrainingConfig as MultiLabelTrainingConfig
            ml_config = MultiLabelTrainingConfig()
            ml_metrics_base = get_training_logs_base()
            ml_config.metrics_output_dir = str(ml_metrics_base)
            ml_trainer = MultiLabelTrainer(config=ml_config, verbose=False)

            samples = ml_trainer.load_multi_label_data(
                data_path,
                text_field=bundle.text_column,
                label_fields=None,  # Will auto-detect
                id_field=bundle.id_column if hasattr(bundle, 'id_column') else None,
                lang_field=bundle.lang_column if hasattr(bundle, 'lang_column') else None,
                labels_dict_field=bundle.label_column if hasattr(bundle, 'label_column') else 'labels'
            )

            # Detect multi-class groups
            multiclass_groups = ml_trainer.detect_multiclass_groups(samples)

            # Check if user already answered this question during dataset building
            use_multiclass_training = False

            if multiclass_groups:
                if training_approach_from_metadata == 'multi-class':
                    # User already chose multi-class during dataset building
                    use_multiclass_training = True
                    self.console.print("\n[green]✓ Using multi-class training (from dataset configuration)[/green]\n")
                elif training_approach_from_metadata == 'one-vs-all':
                    # User already chose one-vs-all during dataset building
                    use_multiclass_training = False
                    multiclass_groups = None
                    self.console.print("\n[yellow]✓ Using one-vs-all training (from dataset configuration)[/yellow]\n")
                elif training_approach_from_metadata == 'multi-label':
                    # User chose true multi-label: single model with BCEWithLogitsLoss
                    use_multiclass_training = False
                    multiclass_groups = None
                    self.console.print("\n[magenta]✓ Using multi-label training (single model with sigmoid activation)[/magenta]\n")
                elif training_approach_from_metadata in ['hybrid', 'custom']:
                    # User already chose hybrid/custom - will be handled later in dedicated section
                    use_multiclass_training = False
                    multiclass_groups = None
                    self.console.print(f"\n[cyan]✓ Using {training_approach_from_metadata} training (from dataset configuration)[/cyan]\n")
                else:
                    # No previous choice - ask user
                    self.console.print("\n[yellow]ℹ Detected multi-class classification:[/yellow]")
                    for group_name, labels in multiclass_groups.items():
                        value_names = [lbl[len(group_name)+1:] if lbl.startswith(group_name+'_') else lbl for lbl in labels]
                        self.console.print(f"  • {group_name}: {', '.join(value_names)}")

                    # Ask user if they want true multi-class (1 model) or one-vs-all (N models)
                    self.console.print("\n[bold]Training approach:[/bold]")
                    self.console.print("  • [green]Multi-class[/green]: Train 1 model per group to predict among all classes")
                    self.console.print("  • [yellow]One-vs-all[/yellow]: Train N separate binary models (1 per class)")

                    use_multiclass_training = Confirm.ask(
                        "\n[bold]Use multi-class training? (recommended)[/bold]",
                        default=True
                    )

                    if use_multiclass_training:
                        self.console.print("[green]Will use multi-class training[/green]\n")
                    else:
                        self.console.print("[yellow]✓ Will train separate binary classifiers[/yellow]\n")
                        multiclass_groups = None  # Don't pass to trainer
        else:
            # CSV file or no primary_file - skip multiclass detection
            # For single-label CSV multi-class, train_single_model handles it via num_labels > 2
            if bundle.strategy == "single-label" and training_approach_from_metadata == 'multi-class':
                self.console.print("\n[green]✓ Using multi-class training for single-label CSV[/green]")
                self.console.print("[dim]  (Train will detect num_labels > 2 automatically)[/dim]\n")

    # Create TrainingConfig with user's chosen model
    from llm_tool.trainers.model_trainer import ModelTrainer, TrainingConfig
    training_config = TrainingConfig()
    metrics_base_dir = get_training_logs_base()
    training_config.metrics_output_dir = str(metrics_base_dir)
    training_config.model_name = model_name
    training_config.num_epochs = epochs
    # Optimizer hyperparameters (CLI) — feed the model_trainer-direct path too.
    _opt_src = quick_params if quick_params else (model_config or {})
    training_config.learning_rate = _opt_src.get('learning_rate', getattr(training_config, 'learning_rate', 2e-5))
    training_config.warmup_ratio = _opt_src.get('warmup_ratio', getattr(training_config, 'warmup_ratio', 0.0))
    # Early stopping: from quick_params (new session) or model_config (resume)
    _es_from_params = quick_params.get('early_stopping_patience') if quick_params else None
    _es_from_config = model_config.get('early_stopping_patience') if model_config else None
    training_config.early_stopping_patience = _es_from_params or _es_from_config
    # Interactive skip: from quick_params (new session) or model_config (resume)
    _is_from_params = quick_params.get('interactive_skip', True) if quick_params else True
    _is_from_config = model_config.get('interactive_skip', True) if model_config else True
    training_config.interactive_skip = _is_from_params if quick_params else _is_from_config
    # NORMAL-phase imbalance handling (from quick_params or resume model_config)
    _imb_src = quick_params if quick_params else (model_config or {})
    training_config.imbalance_strategy = _imb_src.get('imbalance_strategy')
    training_config.focal_gamma = _imb_src.get('focal_gamma', 2.0)
    training_config.imbalance_weight_source = _imb_src.get('imbalance_weight_source', 'auto')
    training_config.imbalance_class_weights = _imb_src.get('imbalance_class_weights')
    training_config.imbalance_weighted_sampler = _imb_src.get('imbalance_weighted_sampler', True)
    training_config.batch_size = _get_optimal_batch_size(model_name)  # Dynamic batch size based on system resources
    # Tokenizer-aware max_length: only override the default when the user accepted
    # the optimization in _collect_quick_mode_parameters. None => keep the dataclass
    # default (512), preserving legacy behavior.
    if quick_params and quick_params.get('max_length'):
        training_config.max_length = int(quick_params['max_length'])
    # Honour the "Exclude long texts" strategy the user chose at the Token
    # Length step. Stored in bundle.metadata; the filter is applied at load
    # time inside MultiLabelTrainer.load_multi_label_data.
    if hasattr(bundle, 'metadata') and bundle.metadata:
        training_config.exclude_long_texts = bool(
            bundle.metadata.get('exclude_long_texts', False)
        )
        training_config.split_long_texts = bool(
            bundle.metadata.get('split_long_texts', False)
        )

    # Multi-label classification settings from bundle metadata
    if hasattr(bundle, 'metadata') and bundle.metadata:
        # CRITICAL FIX: Derive multi_label from training_approach if not explicitly set
        training_approach = bundle.metadata.get('training_approach', '')
        training_config.multi_label = (
            bundle.metadata.get('multi_label', False) or
            training_approach == 'multi-label'
        )
        training_config.multi_label_threshold = bundle.metadata.get('multi_label_threshold', 0.5)

    # Determine if we need to train by language
    needs_language_training = False

    if models_by_language:
        # User selected different models for each language
        needs_language_training = True
        self.console.print(f"\n[yellow]Multi-language training enabled:[/yellow]")
        self.console.print(f"[dim]Training with specialized models for each language:[/dim]")
        for lang in sorted(models_by_language.keys()):
            self.console.print(f"  • {lang.upper()}: {models_by_language[lang]}")
    else:
        # Single model - check if it's monolingual and we have multiple languages
        is_multilingual = self._is_model_multilingual(model_name)
        needs_language_training = not is_multilingual and len(languages) > 1

        if needs_language_training:
            self.console.print(f"\n[yellow]Multi-language training enabled:[/yellow]")
            self.console.print(f"[dim]The model '{model_name}' is language-specific, so separate models will be trained for each language:[/dim]")
            for lang in sorted(languages):
                self.console.print(f"  • {lang.upper()}")

    multiclass_models_by_language = dict(models_by_language) if models_by_language else None
    multiclass_needs_language_training = bool(needs_language_training)

    trainer = ModelTrainer(config=training_config)

    # Build trainer config with multiclass_groups if detected
    extra_config = {
        "model_name": model_name,
        "num_epochs": epochs,
        "batch_size": _get_optimal_batch_size(model_name),  # Dynamic batch size based on system resources
        "reinforced_learning": enable_reinforced_learning,  # CRITICAL: Pass reinforced learning setting
        "train_by_language": needs_language_training,
        "confirmed_languages": list(languages) if languages else None,  # Pass all detected languages
        "training_approach": training_approach_from_metadata,  # CRITICAL: Pass training approach to prevent multiclass auto-detection for one-vs-all
        # CRITICAL: Carry the tokenizer-aware filter knobs into every
        # trainer.train({...}) call below. Setting them on training_config
        # alone is not enough — several training paths build an inline dict
        # (e.g. the true multi-label single-model branch at ~12273) and
        # ModelTrainer.train() re-applies setattr only for keys present in
        # that dict, so we must include them here.
        "exclude_long_texts": bool(training_config.exclude_long_texts),
        "split_long_texts": bool(training_config.split_long_texts),
        "max_length": int(training_config.max_length),
    }

    # Add reinforced learning parameters if enabled
    if quick_params:
        if enable_reinforced_learning or quick_params.get('force_reinforced', False):
            extra_config["rl_f1_threshold"] = quick_params.get('rl_f1_threshold', 0.70)
            extra_config["rl_oversample_factor"] = quick_params.get('rl_oversample_factor', 2.0)
            extra_config["rl_class_weight_factor"] = quick_params.get('rl_class_weight_factor', 2.0)
            if quick_params.get('manual_rl_epochs') is not None:
                extra_config["reinforced_epochs"] = quick_params['manual_rl_epochs']
        # Distribution-aware training (Phase 1) — independent of Phase 2
        if quick_params.get('distribution_aware', False):
            extra_config["distribution_aware"] = True
        # NORMAL-phase imbalance handling — independent of Phases 1/2
        if quick_params.get('imbalance_strategy') is not None:
            extra_config["imbalance_strategy"] = quick_params.get('imbalance_strategy')
            extra_config["focal_gamma"] = quick_params.get('focal_gamma', 2.0)
            extra_config["imbalance_weight_source"] = quick_params.get('imbalance_weight_source', 'auto')
            extra_config["imbalance_class_weights"] = quick_params.get('imbalance_class_weights')
            extra_config["imbalance_weighted_sampler"] = quick_params.get('imbalance_weighted_sampler', True)

    # Resume at RL Phase 2 (from model_config set by resume_rl action)
    if model_config and model_config.get('skip_to_rl'):
        extra_config['skip_to_rl'] = True
        extra_config['rl_state_path'] = model_config.get('rl_state_path')
        extra_config['reinforced_learning'] = True

    # Add models_by_language if user selected per-language models
    if models_by_language:
        extra_config["models_by_language"] = models_by_language

    # Add multiclass_groups if user opted for multi-class training
    # CRITICAL: Do NOT add multiclass_groups if user chose one-vs-all (which uses multi-label infrastructure but creates binary models)
    # CRITICAL FIX: Also add multiclass_groups for single-label when user explicitly chose multi-class approach
    if multiclass_groups and training_approach_from_metadata != 'one-vs-all':
        extra_config["multiclass_groups"] = multiclass_groups
        self.logger.info(f"[MULTICLASS] Passing {len(multiclass_groups)} multiclass group(s) to trainer: {list(multiclass_groups.keys())}")

    # CRITICAL FIX: Handle one-vs-all training properly
    # For one-vs-all, we need to train separate binary models for each label

    # DEBUG logging
    self.logger.debug(f"[ONE-VS-ALL DEBUG] training_approach_from_metadata = {training_approach_from_metadata}")
    self.logger.debug(f"[ONE-VS-ALL DEBUG] hasattr(bundle, 'training_files') = {hasattr(bundle, 'training_files')}")
    if hasattr(bundle, 'training_files'):
        self.logger.debug(f"[ONE-VS-ALL DEBUG] bundle.training_files.keys() = {list(bundle.training_files.keys()) if bundle.training_files else None}")

    if training_approach_from_metadata == 'one-vs-all':
        # One-vs-all training: create separate binary models for each label

        # One-vs-all always creates fresh binary CSV files from the JSONL.
        # Pre-generated training_files are JSONL multi-label format (labels: [...])
        # which is NOT compatible with the binary format (label: 0/1) expected by
        # the one-vs-all trainer. So we always regenerate.
        category_files = {}
        if not category_files:
            self.console.print("\n[yellow]Creating binary datasets for one-vs-all training...[/yellow]")

            # Load the JSONL file to extract labels
            import json
            data_path = str(bundle.primary_file) if hasattr(bundle, 'primary_file') else str(bundle.dataset_path)

            # Read the JSONL and collect unique labels
            all_labels_set = set()
            records = []
            _is_binary_list_format = False  # Detect key_yes/key_no format
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    records.append(record)
                    if 'labels' in record:
                        # Handle both list and dict formats
                        if isinstance(record['labels'], dict):
                            all_labels_set.update(record['labels'].keys())
                        elif isinstance(record['labels'], list):
                            for lbl in record['labels']:
                                # Detect binary format: key_yes / key_no
                                if lbl.endswith('_yes') or lbl.endswith('_no'):
                                    _is_binary_list_format = True
                                    # Extract base label name (strip _yes/_no)
                                    base = lbl.rsplit('_', 1)[0]
                                    all_labels_set.add(base)
                                else:
                                    all_labels_set.add(lbl)

            self.logger.debug(f"[ONE-VS-ALL] Found {len(records)} records")
            self.logger.debug(f"[ONE-VS-ALL] Found {len(all_labels_set)} unique labels: {sorted(all_labels_set)}")

            if not all_labels_set:
                # Debug: print first record to see the structure
                if records:
                    self.logger.error(f"[ONE-VS-ALL] No labels found! First record structure: {records[0]}")
                    self.console.print(f"\n[red]✗ Could not find labels in JSONL file[/red]")
                    self.console.print(f"[dim]First record structure: {json.dumps(records[0], indent=2)}[/dim]")
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': 'No labels found in JSONL'
                }

            # Create CSV files for each label (persist inside session when available)
            import csv

            binary_output_dir = self._prepare_onevsall_workspace()
            self._announce_onevsall_storage()

            # Get filter logger for tracking (with session context if available)
            filter_logger = get_filter_logger(session_id=getattr(self, 'current_session_id', None))
            location = "advanced_cli.one_vs_all_binary_dataset_creation"

            for label_name in sorted(all_labels_set):
                # Create binary CSV: text + label (0 or 1)
                csv_path = binary_output_dir / f"binary_{label_name}.csv"

                # Track filtered items for this label
                filtered_empty_texts = []
                filtered_invalid_texts = []
                written_count = 0

                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['text', 'label', 'language'])
                    writer.writeheader()

                    for idx, record in enumerate(records):
                        # Binary label: 1 if this label is present/True, 0 otherwise
                        labels_data = record.get('labels', {})

                        # Handle both dict and list formats
                        if isinstance(labels_data, dict):
                            # Dict format: {"agriculture": "yes", "defense": "no"}
                            # Key absent = undersampled → skip this row for this label
                            if label_name not in labels_data:
                                continue  # Skip: this row was undersampled for this label
                            label_raw = labels_data[label_name]
                            if isinstance(label_raw, bool):
                                label_value = 1 if label_raw else 0
                            elif isinstance(label_raw, (int, float)):
                                label_value = 1 if label_raw > 0 else 0
                            else:
                                label_value = 1 if str(label_raw).lower() in ['1', 'true', 'yes'] else 0
                        elif isinstance(labels_data, list):
                            # List format. Two sub-cases:
                            #  - Explicit binary: ["agriculture_yes", "defense_no", ...]
                            #    where every annotated label carries a _yes/_no suffix.
                            #    Absence of BOTH means the row was undersampled for this
                            #    label (not annotated for it) -> skip it.
                            #  - Bare multi-label: ["agriculture", "defense", ...] listing
                            #    ONLY the positive labels. Here absence means NEGATIVE (0),
                            #    not "skip": every row is implicitly annotated for every
                            #    label. Treating absence as skip collapses the one-vs-all
                            #    dataset to a single positive class (only label=1 rows),
                            #    which breaks binary training. This mirrors the hybrid path.
                            has_yes = f"{label_name}_yes" in labels_data
                            has_no = f"{label_name}_no" in labels_data
                            has_bare = label_name in labels_data  # bare positive label
                            if has_yes:
                                label_value = 1
                            elif has_no:
                                label_value = 0
                            elif has_bare:
                                label_value = 1  # bare name present = positive
                            elif _is_binary_list_format:
                                # Explicit _yes/_no format and neither suffix present:
                                # this label wasn't annotated for this row -> undersampled.
                                continue  # Skip: label not mentioned -> undersampled
                            else:
                                # Bare multi-label format: absence = negative class.
                                label_value = 0
                        else:
                            continue  # Skip: unknown format

                        # CRITICAL: Validate text is a valid non-empty string
                        text_raw = record.get('text', '')
                        if not isinstance(text_raw, str):
                            # Log invalid type
                            filtered_invalid_texts.append({
                                'index': idx,
                                'type': type(text_raw).__name__,
                                'value': str(text_raw)[:100] if text_raw else 'None'
                            })
                            text_raw = str(text_raw) if text_raw else ''

                        # Skip empty texts
                        if not text_raw.strip():
                            filtered_empty_texts.append({
                                'index': idx,
                                'id': record.get('id', 'unknown'),
                                'text_length': len(text_raw)
                            })
                            continue

                        # Ensure language is a string
                        lang_raw = record.get('lang', record.get('language', ''))
                        if not isinstance(lang_raw, str):
                            lang_raw = str(lang_raw) if lang_raw else ''

                        row = {
                            'text': text_raw.strip(),
                            'label': label_value,
                            'language': lang_raw
                        }
                        writer.writerow(row)
                        written_count += 1

                # Log filtered items
                if filtered_empty_texts:
                    filter_logger.log_filtered_batch(
                        items=[f"Record {f['index']} (id: {f['id']})" for f in filtered_empty_texts],
                        reason="empty_text",
                        location=f"{location}.{label_name}",
                        indices=[f['index'] for f in filtered_empty_texts]
                    )

                if filtered_invalid_texts:
                    filter_logger.log_filtered_batch(
                        items=[f"Record {f['index']}: {f['type']}" for f in filtered_invalid_texts],
                        reason="invalid_text_type",
                        location=f"{location}.{label_name}",
                        indices=[f['index'] for f in filtered_invalid_texts]
                    )

                category_files[label_name] = csv_path
                self.console.print(f"[dim]  Created binary dataset for: {label_name} ({written_count} samples)[/dim]")

                # Warn if too many filtered
                total_filtered = len(filtered_empty_texts) + len(filtered_invalid_texts)
                if total_filtered > 0:
                    self.console.print(f"[yellow]    [!]  Filtered {total_filtered} invalid/empty texts[/yellow]")

            self.console.print(f"[green]Created {len(category_files)} binary datasets[/green]\n")

        if category_files:
            num_models_to_train = len(category_files)
            self.console.print(f"\n[yellow][!]  One-vs-all requires training {num_models_to_train} separate binary models.[/yellow]")

            # ========== PARALLEL TRAINING OPTION ==========
            # Check if parallel training is available and beneficial using smart scheduler
            use_parallel_training = False
            distributed_orchestrator = None  # SSH distributed training orchestrator
            try:
                from llm_tool.trainers.hybrid_parallel_trainer import HardwareAnalyzer, HybridParallelTrainer
                from llm_tool.trainers.task_scheduler import SchedulerConfig

                # Use smart scheduler to determine if parallel training is beneficial
                scheduler_config = SchedulerConfig()
                should_offer, scheduler_message = HybridParallelTrainer.should_offer_parallel(
                    num_models_to_train, scheduler_config
                )

                hw_analyzer = HardwareAnalyzer()
                can_parallel, parallel_reason = hw_analyzer.can_use_parallel_training(num_models_to_train)

                if can_parallel and should_offer:
                    # Get resource plan for display
                    resource_plan = hw_analyzer.create_resource_plan(model_name, num_models_to_train)

                    # Determine recommendation level based on thresholds
                    strongly_recommended = num_models_to_train >= scheduler_config.recommended_tasks_for_parallel
                    min_threshold = scheduler_config.min_tasks_for_parallel
                    recommended_threshold = scheduler_config.recommended_tasks_for_parallel

                    self.console.print(f"\n[bold cyan]Parallel Training Available[/bold cyan]")

                    # Clear recommendation panel
                    from rich.panel import Panel
                    if strongly_recommended:
                        rec_text = (
                            f"[bold green][OK] STRONGLY RECOMMENDED[/bold green]\n\n"
                            f"You have [bold]{num_models_to_train} models[/bold] to train, which is ≥{recommended_threshold} models.\n"
                            f"Parallel training will provide [green]significant speedup[/green] by utilizing GPU + CPU workers simultaneously."
                        )
                        rec_style = "green"
                        default_choice = True
                    else:
                        # Between min_threshold and recommended_threshold
                        rec_text = (
                            f"[bold yellow][!] OPTIONAL - MARGINAL BENEFIT[/bold yellow]\n\n"
                            f"You have [bold]{num_models_to_train} models[/bold] to train ({min_threshold}-{recommended_threshold-1} range).\n"
                            f"Parallel training offers [yellow]moderate speedup[/yellow], but overhead may reduce benefits.\n\n"
                            f"[dim]• Sequential (GPU-only) is often faster for <{recommended_threshold} models\n"
                            f"• Parallel becomes efficient with ≥{recommended_threshold} models[/dim]"
                        )
                        rec_style = "yellow"
                        default_choice = False

                    self.console.print(Panel(rec_text, title="Recommendation", border_style=rec_style))

                    # Display resource allocation plan
                    from rich.table import Table
                    from rich import box

                    plan_table = Table(
                        title="Resource Allocation Plan",
                        show_header=True,
                        header_style="bold magenta",
                        box=box.ROUNDED,
                        expand=True,
                    )
                    plan_table.add_column("Device", style="cyan")
                    plan_table.add_column("Type", style="white")
                    plan_table.add_column("Batch Size", justify="center")
                    plan_table.add_column("Workers", justify="center")

                    if resource_plan.gpu_allocation:
                        g = resource_plan.gpu_allocation
                        plan_table.add_row(
                            f"GPU {g.device_id.upper()}",
                            "GPU (Priority)",
                            str(g.batch_size),
                            str(g.num_workers),
                        )

                    for cpu in resource_plan.cpu_allocations:
                        plan_table.add_row(
                            f"CPU {cpu.device_id}",
                            "CPU",
                            str(cpu.batch_size),
                            str(cpu.num_workers),
                        )

                    self.console.print(plan_table)
                    self.console.print(f"\n[dim]Total: {resource_plan.total_parallel_workers} workers available[/dim]")

                    # Ask user with clear default based on recommendation
                    from rich.prompt import Confirm
                    if strongly_recommended:
                        prompt_text = "\n[bold]Use parallel training? [green](recommended: yes)[/green][/bold]"
                    else:
                        prompt_text = "\n[bold]Use parallel training? [yellow](recommended: no)[/yellow][/bold]"

                    use_parallel_training = Confirm.ask(prompt_text, default=default_choice)

                    if use_parallel_training:
                        self.console.print("[green]Parallel training enabled with smart task scheduling[/green]\n")

                        # ========== SSH DISTRIBUTED TRAINING OPTION ==========
                        distributed_orchestrator = None
                        try:
                            from llm_tool.cli.ssh_setup_cli import SSHSetupCLI
                            ssh_cli = SSHSetupCLI(self.console)
                            ssh_manager = ssh_cli.prompt_for_remote_machines()

                            if ssh_manager and ssh_manager.machines:
                                from llm_tool.trainers.distributed_training_orchestrator import DistributedTrainingOrchestrator
                                distributed_orchestrator = DistributedTrainingOrchestrator(
                                    ssh_manager=ssh_manager,
                                    config=None,  # Will be set in _run_parallel_training
                                    console=self.console,
                                )
                                # Display GPU-only warning if applicable
                                if ssh_manager.any_remote_has_gpu():
                                    ssh_cli.display_gpu_only_warning()
                        except ImportError as e:
                            self.logger.debug(f"SSH distributed training not available: {e}")
                        except Exception as e:
                            self.logger.warning(f"SSH setup failed: {e}")
                            self.console.print("[yellow]   Poursuite avec l'entrainement parallele local uniquement.[/yellow]\n")

                    else:
                        self.console.print("[yellow]✓ Using sequential training[/yellow]\n")
                else:
                    # Show the scheduler's reason for not recommending parallel
                    self.console.print(f"[dim]   {scheduler_message}[/dim]")
                    if not can_parallel:
                        self.console.print(f"[dim]   {parallel_reason}[/dim]")
                    self.console.print("[yellow]   Training models sequentially...[/yellow]\n")

            except ImportError as e:
                self.logger.debug(f"Parallel training not available: {e}")
                self.console.print("[yellow]   Training models sequentially...[/yellow]\n")
            except Exception as e:
                self.logger.warning(f"Could not check parallel training availability: {e}")
                self.console.print("[yellow]   Training models sequentially...[/yellow]\n")

            # ========== EXECUTE TRAINING (PARALLEL OR SEQUENTIAL) ==========
            if use_parallel_training:
                # Use parallel training orchestrator
                return self._run_parallel_training(
                    category_files=category_files,
                    model_name=model_name,
                    epochs=epochs,
                    languages=languages,
                    needs_language_training=needs_language_training,
                    enable_reinforced_learning=enable_reinforced_learning,
                    manual_rl_epochs=manual_rl_epochs,
                    quick_params=quick_params,
                    session_id=session_id,
                    bundle=bundle,
                    output_dir=output_dir,
                    runtime_params=runtime_params,
                    models_by_language=models_by_language,
                    training_approach=training_approach_from_metadata or 'one-vs-all',
                    extra_config=extra_config if 'extra_config' in dir() else {},
                    distributed_orchestrator=distributed_orchestrator,
                )

            # Initialize global progress tracking for one-vs-all training
            import time
            global_start_time = time.time()

            # CRITICAL: Calculate total models based on training approach
            # One-vs-all creates one binary model per category
            # If needs_language_training=True, we train one model PER (category, language)
            num_categories = int(len(category_files))
            num_languages = int(len(languages)) if languages else 1

            if needs_language_training and num_languages > 1:
                # Per-language training: one model per (category, language) combination
                global_total_models = int(num_categories * num_languages)
                self.logger.info(f"[EPOCH CALC] One-vs-all + per-language: {num_categories} categories × {num_languages} languages = {global_total_models} total models")
            else:
                # Multilingual model: one model per category (handles all languages)
                global_total_models = int(num_categories)
                self.logger.info(f"[EPOCH CALC] One-vs-all + multilingual: {num_categories} categories = {global_total_models} total models")

            epochs = int(epochs) if epochs is not None else 10
            manual_rl_epochs = int(manual_rl_epochs) if manual_rl_epochs is not None else None
            global_total_epochs = int(global_total_models * epochs)

            # Calculate maximum possible epochs (if all models trigger reinforced learning)
            if enable_reinforced_learning and manual_rl_epochs is not None:
                global_max_epochs = int(global_total_models * (epochs + manual_rl_epochs))
            else:
                global_max_epochs = int(global_total_epochs)

            # DEBUGGING: Log the epoch calculation
            self.logger.info("="*80)
            self.logger.info("GLOBAL EPOCHS CALCULATION DEBUG")
            self.logger.info(f"  Training mode: one-vs-all")
            self.logger.info(f"  Language training: {'per-language' if needs_language_training else 'multilingual'}")
            self.logger.info(f"  Number of categories: {num_categories}")
            self.logger.info(f"  Number of languages: {num_languages}")
            self.logger.info(f"  Languages: {sorted(languages) if languages else 'N/A'}")
            self.logger.info(f"  Base epochs per model: {epochs}")
            self.logger.info(f"  RL epochs per model: {manual_rl_epochs if manual_rl_epochs else 'None'}")
            self.logger.info(f"  CALCULATED global_total_models: {global_total_models}")
            self.logger.info(f"  CALCULATED global_total_epochs: {global_total_epochs}")
            self.logger.info(f"  CALCULATED global_max_epochs: {global_max_epochs}")
            self.logger.info("="*80)

            global_completed_epochs = int(0)

            # Train each binary model sequentially
            results_per_category = {}

            # RESUME SCAN: Count already trained one-vs-all models before starting
            _ova_already_trained = 0
            _model_safe = model_name.replace("/", "_")
            for _cat_name in category_files:
                _cat_model_dir = Path("models") / session_id / "normal_training" / _cat_name / _model_safe / "model"
                if _cat_model_dir.exists() and ((_cat_model_dir / "pytorch_model.bin").exists() or (_cat_model_dir / "model.safetensors").exists()):
                    _ova_already_trained += 1
            if _ova_already_trained > 0:
                _ova_remaining = len(category_files) - _ova_already_trained
                self.console.print(f"\n[bold cyan]RESUMING TRAINING SESSION[/bold cyan]")
                self.console.print(f"[green]   \u2713 {_ova_already_trained} models already trained (will be skipped)[/green]")
                self.console.print(f"[yellow]   \u23f3 {_ova_remaining} models remaining to train[/yellow]\n")

            for idx, (category_name, category_file) in enumerate(category_files.items(), 1):
                # RESUME: Skip already trained models
                _cat_model_dir = Path("models") / session_id / "normal_training" / category_name / _model_safe / "model"
                if _cat_model_dir.exists() and ((_cat_model_dir / "pytorch_model.bin").exists() or (_cat_model_dir / "model.safetensors").exists()):
                    # Try to get best F1 info for the skip message
                    _skip_info = ""
                    try:
                        import glob as _g
                        _metrics_base = str(session_metrics_dir) if session_metrics_dir else str(get_training_metrics_dir(session_id) / "normal_training")
                        _csvs = _g.glob(f"{_metrics_base}/{category_name}/**/training.csv", recursive=True)
                        if _csvs:
                            import csv as _csv
                            with open(_csvs[0]) as _f:
                                _lines = [l for l in _f if not l.startswith('#')]
                                _rows = list(_csv.DictReader(_lines))
                            if _rows:
                                _best = max(_rows, key=lambda r: float(r.get('macro_f1', r.get('f1_1', 0))))
                                _f1 = float(_best.get('macro_f1', _best.get('f1_1', 0)))
                                _ep = _best.get('epoch', '?')
                                _total_ep = len(_rows)
                                _skip_info = f" (F1={_f1:.3f} at epoch {_ep}/{_total_ep})"
                    except Exception:
                        pass
                    self.console.print(f"\n[green]\u23ed Skipping '{category_name}' \u2014 already trained{_skip_info}[/green]")
                    self.console.print(f"[dim]   Model path: {_cat_model_dir}[/dim]")
                    global_completed_epochs += int(epochs)
                    continue

                self.console.print(f"\n[cyan]Training binary model for: {category_name}[/cyan]")

                # Create config for this specific category
                # CRITICAL: Convert all numeric values to Python int to avoid numpy.int64 issues
                category_config = {
                    'input_file': str(category_file),
                    'text_column': 'text',
                    'label_column': 'label',
                    'model_name': model_name,
                    'num_epochs': int(epochs),
                    'batch_size': _get_optimal_batch_size(model_name),  # Dynamic batch size based on system resources
                    'reinforced_learning': enable_reinforced_learning,  # CRITICAL: Pass reinforced learning setting
                    'output_dir': str(Path(output_dir) / f'model_{category_name}'),
                    'training_strategy': 'single-label',  # Binary classification
                    'training_approach': training_approach_from_metadata or 'one-vs-all',  # CRITICAL: Pass for chart labeling
                    'category_name': category_name,  # For display in metrics
                    'early_stopping_patience': training_config.early_stopping_patience if hasattr(training_config, 'early_stopping_patience') else None,
                    'interactive_skip': training_config.interactive_skip if hasattr(training_config, 'interactive_skip') else True,
                    'confirmed_languages': list(languages) if languages else None,
                    'train_by_language': needs_language_training,
                    'session_id': session_id,
                    'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                    'auto_remove_insufficient_labels': True,  # Avoid interactive prompts mid-training
                    # Global progress tracking - ALL converted to Python int
                    'global_total_models': int(global_total_models),
                    'global_current_model': int(idx),
                    'global_total_epochs': int(global_total_epochs),
                    'global_max_epochs': int(global_max_epochs),
                    'global_completed_epochs': int(global_completed_epochs),
                    'global_start_time': global_start_time
                }

                # Add reinforced learning parameters if enabled
                if enable_reinforced_learning and manual_rl_epochs is not None:
                    category_config["reinforced_epochs"] = int(manual_rl_epochs)

                # Add models_by_language if user selected per-language models
                # CRITICAL: Validate type before passing to avoid numpy type errors
                if models_by_language:
                    if not isinstance(models_by_language, dict):
                        self.console.print(f"[red][!]  ERROR: models_by_language has invalid type: {type(models_by_language)}[/red]")
                        self.logger.error(f"one-vs-all: models_by_language type error: {type(models_by_language)}, value: {models_by_language}")
                    else:
                        category_config["models_by_language"] = models_by_language

                # CRITICAL DEBUG: Log category_config to detect numpy types
                self.logger.debug("=" * 80)
                self.logger.debug(f"category_config for {category_name}:")
                for key, value in category_config.items():
                    self.logger.debug(f"  {key}: type={type(value)}, value={value}")
                self.logger.debug("=" * 80)

                try:
                    category_result = trainer.train(category_config)
                    # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                    returned_epochs = category_result.get('global_completed_epochs') if isinstance(category_result, dict) else None
                    if returned_epochs is not None:
                        global_completed_epochs = max(global_completed_epochs, returned_epochs)
                    self.logger.debug(f"[EPOCH_TRACK] After one-vs-all category {category_name}: global_completed_epochs={global_completed_epochs}")
                    results_per_category[category_name] = category_result
                    self.console.print(f"[green]Completed {category_name}: Accuracy={category_result.get('accuracy', 0):.4f}, F1={category_result.get('best_f1_macro', 0):.4f}[/green]")
                except Exception as exc:
                    self.console.print(f"[red]✗ Failed to train {category_name}: {exc}[/red]")
                    self.logger.exception(f"Training failed for {category_name}", exc_info=exc)
                    # CRITICAL: Log full traceback
                    import traceback
                    self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    results_per_category[category_name] = {'error': str(exc)}
                    # CRITICAL: Re-raise to see actual error
                    raise

            # Aggregate results
            successful_results = [r for r in results_per_category.values() if 'error' not in r]
            if successful_results:
                avg_accuracy = sum(r.get('accuracy', 0) for r in successful_results) / len(successful_results)
                avg_f1 = sum(r.get('best_f1_macro', 0) for r in successful_results) / len(successful_results)

                result = {
                    'best_model': model_name,
                    'accuracy': avg_accuracy,
                    'best_f1_macro': avg_f1,
                    'model_path': str(output_dir),
                    'training_time': sum(r.get('training_time', 0) for r in successful_results),
                    'models_trained': len(successful_results),
                    'total_models': len(category_files),
                    'per_category_results': results_per_category
                }
            else:
                self.console.print("[red]All category trainings failed[/red]")
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': 'All category trainings failed'
                }
        else:
            self.console.print("[red]No category files found for one-vs-all training[/red]")
            return {
                'runtime_params': runtime_params,
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'error': 'No category files'
            }
    elif training_approach_from_metadata in ['hybrid', 'custom'] and hasattr(bundle, 'training_files') and bundle.training_files:
        # Hybrid/Custom training: mix of multi-class, multi-label, and one-vs-all per key
        multiclass_keys = bundle.metadata.get('multiclass_keys', [])
        multilabel_keys = bundle.metadata.get('multilabel_keys', [])
        onevsall_keys = bundle.metadata.get('onevsall_keys', [])

        self.console.print(f"\n[cyan]Hybrid/Custom training:[/cyan]")
        self.console.print(f"  • {len(multiclass_keys)} keys with multi-class strategy")
        self.console.print(f"  • {len(multilabel_keys)} keys with multi-label strategy")
        self.console.print(f"  • {len(onevsall_keys)} keys with one-vs-all strategy\n")

        # Initialize global progress tracking for hybrid/custom training
        import time
        global_start_time = time.time()

        # CRITICAL: Calculate total models for hybrid/custom training
        # Multi-class keys: 1 model per key (handles all classes in that key)
        # Multi-label keys: 1 model per key (handles all labels in that key with sigmoid)
        # One-vs-all keys: N models per key (1 per class/category in that key)
        num_multiclass_models = len(multiclass_keys)
        num_multilabel_models = len(multilabel_keys)
        num_onevsall_models = 0
        value_counts_by_key_meta = bundle.metadata.get('value_counts_by_key', {})
        all_keys_values_meta = bundle.metadata.get('all_keys_values', {})

        if onevsall_keys:
            for key in onevsall_keys:
                counts_for_key = value_counts_by_key_meta.get(key)
                if counts_for_key:
                    label_count = sum(1 for _value, count in counts_for_key.items() if count is None or count > 0)
                    if label_count == 0:
                        label_count = len(all_keys_values_meta.get(key, []))
                else:
                    label_count = len(all_keys_values_meta.get(key, []))

                if label_count == 0:
                    self.logger.warning(f"[GLOBAL] No value statistics found for one-vs-all key '{key}'. Defaulting to 1 model.")
                    label_count = 1

                num_onevsall_models += label_count

        num_languages = int(len(languages)) if languages else 1
        if num_languages <= 0:
            num_languages = 1

        multiclass_multiplier = num_languages if multiclass_needs_language_training and num_languages > 1 else 1
        multilabel_multiplier = num_languages if multiclass_needs_language_training and num_languages > 1 else 1
        onevsall_multiplier = num_languages if binary_train_by_language and num_languages > 1 else 1

        total_multiclass_models = num_multiclass_models * multiclass_multiplier
        total_multilabel_models = num_multilabel_models * multilabel_multiplier
        total_onevsall_models = num_onevsall_models * onevsall_multiplier
        global_total_models = total_multiclass_models + total_multilabel_models + total_onevsall_models

        self.logger.info(
            f"[EPOCH CALC] Hybrid/custom model counts "
            f"(multiclass ×{multiclass_multiplier}, multi-label ×{multilabel_multiplier}, one-vs-all ×{onevsall_multiplier}) -> "
            f"{total_multiclass_models} + {total_multilabel_models} + {total_onevsall_models} = {global_total_models}"
        )

        global_total_epochs = global_total_models * epochs

        # Calculate maximum possible epochs (if all models trigger reinforced learning)
        if enable_reinforced_learning and manual_rl_epochs is not None:
            global_max_epochs = global_total_models * (epochs + manual_rl_epochs)
        else:
            global_max_epochs = global_total_epochs

        # DEBUGGING: Log the epoch calculation
        self.logger.info("="*80)
        self.logger.info("GLOBAL EPOCHS CALCULATION DEBUG")
        self.logger.info(f"  Training mode: hybrid/custom")
        self.logger.info(f"  Multiclass per-language: {multiclass_needs_language_training}")
        self.logger.info(f"  One-vs-all per-language: {binary_train_by_language}")
        self.logger.info(f"  Multiclass keys: {num_multiclass_models} (×{multiclass_multiplier} => {total_multiclass_models})")
        self.logger.info(f"  One-vs-all keys: {num_onevsall_models} (×{onevsall_multiplier} => {total_onevsall_models})")
        self.logger.info(f"  Number of languages: {num_languages}")
        self.logger.info(f"  Languages: {sorted(languages) if languages else 'N/A'}")
        self.logger.info(f"  Base epochs per model: {epochs}")
        self.logger.info(f"  RL epochs per model: {manual_rl_epochs if manual_rl_epochs else 'None'}")
        self.logger.info(f"  CALCULATED global_total_models: {global_total_models}")
        self.logger.info(f"  CALCULATED global_total_epochs: {global_total_epochs}")
        self.logger.info(f"  CALCULATED global_max_epochs: {global_max_epochs}")
        self.logger.info("="*80)

        global_completed_epochs = 0

        results_per_key: Dict[str, Dict[str, Any]] = {}
        trained_models_map: Dict[str, str] = {}
        trained_model_records: List[Dict[str, Any]] = []
        current_model_index = 0
        total_training_time = 0.0
        skipped_models_count = 0

        def _is_model_already_trained(category_name: str, model_base: str, stage: str = 'normal_training') -> Tuple[bool, Optional[str]]:
            """Check if a model for this category is already trained and complete.

            Returns:
                (is_complete, model_path) - True if complete model exists with pytorch_model.bin or model.safetensors
            """
            try:
                base_model_dir = Path("models") / session_id / stage / category_name
                if not base_model_dir.exists():
                    return False, None

                # Check all subdirectories for the model
                for model_dir in base_model_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    # Check for 'model/' subdirectory (complete model) vs 'model_epoch_X/' (checkpoint)
                    final_model_dir = model_dir / "model"
                    if final_model_dir.exists():
                        # Check for model weights file
                        pytorch_bin = final_model_dir / "pytorch_model.bin"
                        safetensors = final_model_dir / "model.safetensors"
                        if pytorch_bin.exists() or safetensors.exists():
                            return True, str(final_model_dir)

                    # Also check directly in model_dir (some models save directly there)
                    pytorch_bin = model_dir / "pytorch_model.bin"
                    safetensors = model_dir / "model.safetensors"
                    if pytorch_bin.exists() or safetensors.exists():
                        return True, str(model_dir)

                return False, None
            except Exception as e:
                self.logger.debug(f"Error checking if model exists for {category_name}: {e}")
                return False, None

        def _cleanup_incomplete_model(category_name: str, stage: str = 'normal_training') -> bool:
            """Remove incomplete model checkpoints for a category."""
            try:
                import shutil
                base_model_dir = Path("models") / session_id / stage / category_name
                if not base_model_dir.exists():
                    return False

                # Check for epoch checkpoints without final model
                for model_dir in base_model_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    final_model_dir = model_dir / "model"
                    # If no 'model/' directory, this might be a checkpoint
                    if not final_model_dir.exists():
                        # Check for epoch checkpoints (model_epoch_X directories)
                        for subdir in model_dir.iterdir():
                            if subdir.is_dir() and subdir.name.startswith("model_epoch_"):
                                self.console.print(f"[yellow]Cleaning up incomplete checkpoint: {subdir}[/yellow]")
                                shutil.rmtree(subdir)
                                return True
                return False
            except Exception as e:
                self.logger.warning(f"Failed to cleanup incomplete model for {category_name}: {e}")
                return False

        def _register_model_metrics(base_identifier: str, stage: str, owning_key: str, result_dict: Dict[str, Any]) -> int:
            """Collect trained model metadata from a trainer result."""
            models_added = 0
            if not isinstance(result_dict, dict):
                return models_added

            language_results = result_dict.get('language_results')
            if isinstance(language_results, dict) and language_results:
                for lang_code, lang_info in language_results.items():
                    if not isinstance(lang_info, dict) or lang_info.get('error'):
                        continue
                    identifier = f"{base_identifier}@{str(lang_code).upper()}"
                    model_path = lang_info.get('model_path') or ''
                    trained_models_map[identifier] = model_path
                    trained_model_records.append({
                        'id': identifier,
                        'stage': stage,
                        'key': owning_key,
                        'language': str(lang_code).upper(),
                        'model_name': lang_info.get('model') or result_dict.get('best_model'),
                        'f1': float(lang_info.get('f1_macro', 0.0) or 0.0),
                        'accuracy': float(lang_info.get('accuracy', 0.0) or 0.0),
                        'model_path': model_path,
                        'class_names': result_dict.get('class_names')
                    })
                    models_added += 1
            elif isinstance(result_dict.get('trained_models'), dict):
                for suffix, model_path in result_dict['trained_models'].items():
                    if not model_path:
                        continue
                    identifier = f"{base_identifier}:{suffix}"
                    trained_models_map[identifier] = str(model_path)
                    metrics = result_dict.get('metrics', {})
                    trained_model_records.append({
                        'id': identifier,
                        'stage': stage,
                        'key': owning_key,
                        'language': None,
                        'model_name': result_dict.get('best_model'),
                        'f1': float(metrics.get('f1_macro', 0.0) or 0.0),
                        'accuracy': float(metrics.get('accuracy', 0.0) or 0.0),
                        'model_path': str(model_path),
                        'class_names': result_dict.get('class_names')
                    })
                    models_added += 1
            else:
                if result_dict.get('error'):
                    return models_added
                identifier = base_identifier
                model_path = result_dict.get('model_path') or ''
                metrics = result_dict.get('metrics', {})
                trained_models_map[identifier] = model_path
                trained_model_records.append({
                    'id': identifier,
                    'stage': stage,
                    'key': owning_key,
                    'language': None,
                    'model_name': result_dict.get('best_model'),
                    'f1': float(result_dict.get('best_f1_macro', metrics.get('f1_macro', 0.0)) or 0.0),
                    'accuracy': float(result_dict.get('accuracy', metrics.get('accuracy', 0.0)) or 0.0),
                    'model_path': model_path,
                    'class_names': result_dict.get('class_names')
                })
                models_added += 1
            return models_added

        def _create_onevsall_binary_datasets(
            source_path: Path,
            target_keys: Optional[List[str]] = None,
            filter_logger_factory=get_filter_logger,
        ) -> Tuple[Dict[str, Path], Dict[str, Set[str]], Dict[str, Dict[str, Any]]]:
            """Generate binary datasets (CSV) for each one-vs-all label."""
            import json
            import csv

            allowed_keys: Optional[Set[str]] = set(target_keys) if target_keys else None
            records: List[Dict[str, Any]] = []
            all_labels: Set[str] = set()

            with open(source_path, 'r', encoding='utf-8') as src_file:
                for line in src_file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(record)
                    labels_field = record.get('labels', [])
                    if isinstance(labels_field, dict):
                        all_labels.update(str(label) for label in labels_field.keys())
                    elif isinstance(labels_field, list):
                        all_labels.update(str(label) for label in labels_field)

            if not records or not all_labels:
                return {}, {}, {}

            def _label_allowed(label: str) -> bool:
                if not allowed_keys:
                    return True
                for key in allowed_keys:
                    if label.startswith(f"{key}_"):
                        return True
                return False

            labels_to_build = [label for label in sorted(all_labels) if _label_allowed(label)]
            if not labels_to_build:
                return {}, {}, {}

            binary_output_dir = self._prepare_onevsall_workspace("onevsall_hybrid")
            self._announce_onevsall_storage()
            filter_logger = filter_logger_factory(session_id=getattr(self, 'current_session_id', None))
            location = "advanced_cli.hybrid_onevsall_binary_dataset_creation"

            category_files: Dict[str, Path] = {}
            key_value_map: Dict[str, Set[str]] = {}
            label_display_map: Dict[str, Dict[str, Any]] = {}

            for label_name in labels_to_build:
                if '_' in label_name:
                    key_prefix, raw_value = label_name.split('_', 1)
                else:
                    key_prefix, raw_value = label_name, label_name
                key_value_map.setdefault(key_prefix, set()).add(raw_value)

                csv_path = binary_output_dir / f"binary_{label_name}.csv"
                filtered_empty_texts: List[Dict[str, Any]] = []
                filtered_invalid_texts: List[Dict[str, Any]] = []
                written_count = 0
                positive_count = 0

                positive_label_display = f"{label_name} IS"
                negative_label_display = f"{label_name} IS NOT"

                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['text', 'label', 'language']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for idx, record in enumerate(records):
                        labels_field = record.get('labels', [])
                        if isinstance(labels_field, dict):
                            label_raw = labels_field.get(label_name, 0)
                            if isinstance(label_raw, (int, float)):
                                is_positive = label_raw > 0
                            else:
                                is_positive = str(label_raw).lower() in {'1', 'true', 'yes'}
                        elif isinstance(labels_field, list):
                            is_positive = label_name in labels_field
                        else:
                            is_positive = False

                        text_raw = record.get('text', '')
                        if not isinstance(text_raw, str):
                            filtered_invalid_texts.append({
                                'index': idx,
                                'type': type(text_raw).__name__,
                                'value': str(text_raw)[:100] if text_raw else 'None'
                            })
                            text_raw = str(text_raw) if text_raw else ''

                        if not text_raw.strip():
                            filtered_empty_texts.append({
                                'index': idx,
                                'id': record.get('id', 'unknown'),
                                'text_length': len(text_raw)
                            })
                            continue

                        lang_raw = record.get('lang', record.get('language', ''))
                        if not isinstance(lang_raw, str):
                            lang_raw = str(lang_raw) if lang_raw else ''

                        label_value = 1 if is_positive else 0
                        if is_positive:
                            positive_count += 1

                        writer.writerow({
                            'text': text_raw.strip(),
                            'label': label_value,
                            'language': lang_raw
                        })
                        written_count += 1

                if filtered_empty_texts:
                    filter_logger.log_filtered_batch(
                        items=[f"Record {f['index']} (id: {f['id']})" for f in filtered_empty_texts],
                        reason="empty_text",
                        location=f"{location}.{label_name}",
                        indices=[f['index'] for f in filtered_empty_texts]
                    )

                if filtered_invalid_texts:
                    filter_logger.log_filtered_batch(
                        items=[f"Record {f['index']}: {f['type']}" for f in filtered_invalid_texts],
                        reason="invalid_text_type",
                        location=f"{location}.{label_name}",
                        indices=[f['index'] for f in filtered_invalid_texts]
                    )

                category_files[label_name] = csv_path
                negative_count = written_count - positive_count
                label_display_map[label_name] = {
                    'positive_label': positive_label_display,
                    'negative_label': negative_label_display,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'total_samples': written_count,
                }
                self.console.print(
                    f"[dim]  Created binary dataset for: {label_name} "
                    f"({written_count} samples | positives: {positive_count} | negatives: {negative_count})[/dim]"
                )

            return category_files, key_value_map, label_display_map

        # ============================================================
        # PARALLEL TRAINING OPTION FOR HYBRID MODE
        # ============================================================
        # Pre-generate all training files to offer parallel training
        use_parallel_training = False
        all_hybrid_category_files: Dict[str, Path] = {}
        hybrid_binary_files: Dict[str, Path] = {}
        hybrid_label_display_map: Dict[str, Dict[str, Any]] = {}

        # Collect multi-class files
        multiclass_files = {k: Path(v) for k, v in bundle.training_files.items() if k in multiclass_keys}
        all_hybrid_category_files.update(multiclass_files)

        # Collect multi-label files (trained like multi-class but with sigmoid activation)
        multilabel_files = {k: Path(v) for k, v in bundle.training_files.items() if k in multilabel_keys}
        all_hybrid_category_files.update(multilabel_files)

        # Pre-generate binary datasets for one-vs-all if needed
        if onevsall_keys and 'onevsall_multilabel' in bundle.training_files:
            onevsall_file = Path(bundle.training_files['onevsall_multilabel'])
            if onevsall_file.exists():
                self.console.print(f"\n[dim]Preparing binary datasets for {len(onevsall_keys)} one-vs-all keys...[/dim]")
                try:
                    # Validate one-vs-all file
                    validated_file, was_filtered = self._validate_and_filter_insufficient_labels(
                        input_file=str(onevsall_file),
                        strategy='multi-label',
                        min_samples=2,
                        auto_remove=True,
                        train_by_language=needs_language_training
                    )
                    if was_filtered:
                        onevsall_file = Path(validated_file)

                    # Generate binary datasets
                    hybrid_binary_files, _, hybrid_label_display_map = _create_onevsall_binary_datasets(
                        onevsall_file, target_keys=onevsall_keys
                    )
                    all_hybrid_category_files.update(hybrid_binary_files)
                    self.console.print(f"[green]Generated {len(hybrid_binary_files)} binary datasets[/green]")
                except Exception as e:
                    self.logger.warning(f"Could not pre-generate binary datasets: {e}")
                    # Will fall back to sequential generation later

        # Check if parallel training is beneficial using smart scheduler thresholds
        num_models_to_train = len(all_hybrid_category_files)

        # Import scheduler components for intelligent threshold checking
        try:
            from llm_tool.trainers.hybrid_parallel_trainer import HardwareAnalyzer, HybridParallelTrainer
            from llm_tool.trainers.task_scheduler import SchedulerConfig, SmartTaskScheduler
            from rich.prompt import Confirm

            scheduler_config = SchedulerConfig()

            # Use scheduler's intelligent threshold instead of hardcoded value
            # Scheduler checks: < gpu_only_threshold = GPU-only, < min_tasks_for_parallel = sequential recommended
            should_offer, scheduler_message = HybridParallelTrainer.should_offer_parallel(
                num_models_to_train, scheduler_config
            )

            hw_analyzer = HardwareAnalyzer()
            can_parallel, parallel_reason = hw_analyzer.can_use_parallel_training(num_models_to_train)

            if can_parallel and should_offer:
                strongly_recommended = num_models_to_train >= scheduler_config.recommended_tasks_for_parallel
                min_threshold = scheduler_config.min_tasks_for_parallel
                recommended_threshold = scheduler_config.recommended_tasks_for_parallel

                self.console.print(f"\n[bold cyan]Parallel Training Available for Hybrid Mode[/bold cyan]")

                # Show breakdown
                self.console.print(f"\n[bold]Models breakdown:[/bold]")
                self.console.print(f"  • Multi-class models: [green]{len(multiclass_files)}[/green]")
                self.console.print(f"  • One-vs-all binary models: [yellow]{len(hybrid_binary_files)}[/yellow]")
                self.console.print(f"  • [bold]Total: {num_models_to_train} models[/bold]")

                # Clear recommendation panel
                from rich.panel import Panel
                if strongly_recommended:
                    rec_text = (
                        f"[bold green][OK] STRONGLY RECOMMENDED[/bold green]\n\n"
                        f"You have [bold]{num_models_to_train} models[/bold] to train, which is ≥{recommended_threshold} models.\n"
                        f"Parallel training will provide [green]significant speedup[/green] by utilizing GPU + CPU workers simultaneously."
                    )
                    rec_style = "green"
                    default_choice = True
                else:
                    rec_text = (
                        f"[bold yellow][!] OPTIONAL - MARGINAL BENEFIT[/bold yellow]\n\n"
                        f"You have [bold]{num_models_to_train} models[/bold] to train ({min_threshold}-{recommended_threshold-1} range).\n"
                        f"Parallel training offers [yellow]moderate speedup[/yellow], but overhead may reduce benefits.\n\n"
                        f"[dim]• Sequential (GPU-only) is often faster for <{recommended_threshold} models\n"
                        f"• Parallel becomes efficient with ≥{recommended_threshold} models[/dim]"
                    )
                    rec_style = "yellow"
                    default_choice = False

                self.console.print(Panel(rec_text, title="Recommendation", border_style=rec_style))

                # Show resource plan
                plan = hw_analyzer.create_resource_plan(model_name, num_models_to_train)

                from rich.table import Table
                from rich import box

                plan_table = Table(
                    title="Resource Allocation Plan",
                    show_header=True,
                    header_style="bold cyan",
                    box=box.ROUNDED,
                    expand=True,
                )
                plan_table.add_column("Device", style="cyan")
                plan_table.add_column("Type")
                plan_table.add_column("Batch Size", justify="center")
                plan_table.add_column("Workers", justify="center")

                if plan.gpu_allocation:
                    g = plan.gpu_allocation
                    plan_table.add_row(f"GPU {g.device_id.upper()}", "GPU (Priority)", str(g.batch_size), str(g.num_workers))

                for c in plan.cpu_allocations:
                    plan_table.add_row(f"CPU {c.device_id}", "CPU", str(c.batch_size), str(c.num_workers))

                self.console.print(plan_table)
                self.console.print(f"\n[dim]Total: {plan.total_parallel_workers} workers available[/dim]")

                # Ask user with clear default based on recommendation
                if strongly_recommended:
                    prompt_text = "\n[bold]Use parallel training? [green](recommended: yes)[/green][/bold]"
                else:
                    prompt_text = "\n[bold]Use parallel training? [yellow](recommended: no)[/yellow][/bold]"

                use_parallel_training = Confirm.ask(prompt_text, default=default_choice)

                if use_parallel_training:
                    self.console.print("\n[green]✓ Parallel training enabled for hybrid mode[/green]\n")
            elif not should_offer:
                self.console.print(f"[dim]{scheduler_message}[/dim]")
                self.console.print("[yellow]Training models sequentially...[/yellow]\n")

        except ImportError as e:
            self.logger.debug(f"Parallel training not available: {e}")
        except Exception as e:
            self.logger.warning(f"Could not check parallel training: {e}")

        # ============================================================
        # EXECUTE PARALLEL TRAINING IF CHOSEN
        # ============================================================
        if use_parallel_training and all_hybrid_category_files:
            self.console.print(f"\n[bold green]Starting Parallel Hybrid Training...[/bold green]")
            self.console.print(f"[dim]Training {len(all_hybrid_category_files)} models in parallel[/dim]\n")

            return self._run_parallel_training(
                category_files=all_hybrid_category_files,
                model_name=model_name,
                epochs=epochs,
                languages=languages,
                needs_language_training=needs_language_training,
                enable_reinforced_learning=enable_reinforced_learning,
                manual_rl_epochs=manual_rl_epochs,
                quick_params=quick_params if 'quick_params' in dir() else None,
                session_id=session_id,
                bundle=bundle,
                output_dir=output_dir,
                runtime_params=runtime_params,
                models_by_language=models_by_language,
                training_approach='hybrid',  # Use hybrid for chart labels
                extra_config=extra_config if 'extra_config' in dir() else {},
            )

        # ============================================================
        # SEQUENTIAL TRAINING (original code path)
        # ============================================================
        self.console.print("[yellow]Training models sequentially...[/yellow]\n")

        # RESUME SCAN: Count already trained models before starting
        already_trained_count = 0
        for key_name in multiclass_keys:
            is_complete, _ = _is_model_already_trained(key_name, model_name, 'normal_training')
            if is_complete:
                already_trained_count += 1
        for key_name in multilabel_keys:
            is_complete, _ = _is_model_already_trained(key_name, model_name, 'normal_training')
            if is_complete:
                already_trained_count += 1

        if already_trained_count > 0:
            models_remaining = global_total_models - already_trained_count
            self.console.print(f"\n[bold cyan]RESUMING TRAINING SESSION[/bold cyan]")
            self.console.print(f"[green]   ✓ {already_trained_count} models already trained (will be skipped)[/green]")
            self.console.print(f"[yellow]   ⏳ {models_remaining} models remaining to train[/yellow]\n")

        # Train multi-class keys (one model per key)
        key_files = {k: v for k, v in bundle.training_files.items() if k in multiclass_keys}

        # Create progress callback to track completed epochs across all models
        def progress_callback(**metrics):
            """Callback to sync global completed epochs counter from authoritative source."""
            nonlocal global_completed_epochs
            # Update from callback metrics if available (bert_base.py is the authoritative source)
            if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                global_completed_epochs = metrics['global_completed_epochs']

        for key_name in key_files.keys():
            key_file_path = key_files[key_name]

            # RESUME LOGIC: Check if model is already trained and skip if complete
            is_complete, existing_path = _is_model_already_trained(key_name, model_name, 'normal_training')
            if is_complete:
                # Read best metrics from training CSV if available
                _skip_info = ""
                try:
                    import glob as _g
                    _csvs = _g.glob(f"{session_dir}/training_metrics/normal_training/{key_name}/**/training.csv", recursive=True)
                    if _csvs:
                        import csv as _csv
                        with open(_csvs[0]) as _f:
                            _lines = [l for l in _f if not l.startswith('#')]
                            _rows = list(_csv.DictReader(_lines))
                        if _rows:
                            _best = max(_rows, key=lambda r: float(r.get('macro_f1', r.get('f1_1', 0))))
                            _f1 = float(_best.get('macro_f1', _best.get('f1_1', 0)))
                            _ep = _best.get('epoch', '?')
                            _total_ep = len(_rows)
                            _skip_info = f" (F1={_f1:.3f} at epoch {_ep}/{_total_ep})"
                except Exception:
                    pass
                self.console.print(f"\n[green]⏭ Skipping '{key_name}' — already trained{_skip_info}[/green]")
                self.console.print(f"[dim]   Model path: {existing_path}[/dim]")
                # Register the existing model in results
                key_bucket = results_per_key.setdefault(key_name, {})
                key_bucket['multi_class'] = {
                    'skipped': True,
                    'model_path': existing_path,
                    'reason': 'already_trained'
                }
                trained_models_map[key_name] = existing_path
                skipped_models_count += 1
                current_model_index += 1
                continue

            # Clean up incomplete checkpoints before retraining
            _cleanup_incomplete_model(key_name, 'normal_training')

            self.console.print(f"\n[bold]Training multi-class model for '{key_name}'[/bold] ({key_file_path.name})")

            # CRITICAL: Validate each multiclass file before training
            try:
                validated_file, was_filtered = self._validate_and_filter_insufficient_labels(
                    input_file=str(key_file_path),
                    strategy='single-label',  # Multiclass uses single-label strategy
                    min_samples=2,
                    auto_remove=True,  # Auto-remove since user already confirmed for main file
                    train_by_language=needs_language_training
                )
                if was_filtered:
                    key_file_path = Path(validated_file)
                    self.console.print(f"[green]Using filtered dataset for {key_name}[/green]")
            except ValueError as e:
                self.console.print(f"[red]✗ Failed to train {key_name}: {e}[/red]")
                self.logger.error(f"Validation failed for {key_name}: {e}")
                results_per_key[key_name] = {'error': str(e)}
                continue

            key_config = {
                'input_file': str(key_file_path),
                'model_name': model_name,
                'num_epochs': epochs,
                'output_dir': str(output_dir),  # bert_base.py will construct correct path
                'text_column': bundle.text_column,
                'label_column': bundle.label_column,
                'training_strategy': 'single-label',
                'training_approach': 'multi-class',  # CRITICAL: For correct chart labeling
                'category_name': key_name,
                'reinforced_learning': enable_reinforced_learning,
                'session_id': session_id,
                'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                # Global progress tracking
                'global_total_models': global_total_models,
                'global_current_model': current_model_index + 1,
                'global_total_epochs': global_total_epochs,
                'global_max_epochs': global_max_epochs,
                'global_completed_epochs': global_completed_epochs,
                'global_start_time': global_start_time,
                'train_by_language': needs_language_training,
                'confirmed_languages': list(languages) if languages else None,
                'progress_callback': progress_callback,  # CRITICAL: For epoch counting
                'auto_remove_insufficient_labels': True,  # Already validated above
            }

            # Merge reinforced learning hyperparameters and other shared extras
            for extra_key, extra_value in extra_config.items():
                if extra_key not in key_config:
                    key_config[extra_key] = extra_value

            if models_by_language:
                key_config["models_by_language"] = models_by_language

            try:
                key_result = trainer.train(key_config)
                models_learned = _register_model_metrics(base_identifier=key_name, stage='multi-class', owning_key=key_name, result_dict=key_result)
                increment = models_learned if models_learned > 0 else 1
                current_model_index += increment
                # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                returned_epochs = key_result.get('global_completed_epochs') if isinstance(key_result, dict) else None
                if returned_epochs is not None:
                    global_completed_epochs = max(global_completed_epochs, returned_epochs)
                self.logger.debug(f"[EPOCH_TRACK] After multi-class model {key_name}: global_completed_epochs={global_completed_epochs}")
                total_training_time += float(key_result.get('training_time', 0.0) or 0.0)
                key_bucket = results_per_key.setdefault(key_name, {})
                key_bucket['multi_class'] = key_result
                self.console.print(f"[green]Completed {key_name}: Accuracy={key_result.get('accuracy', 0):.4f}, F1={key_result.get('best_f1_macro', 0):.4f}[/green]")
            except Exception as exc:
                self.console.print(f"[red]✗ Failed to train {key_name}: {exc}[/red]")
                self.logger.exception(f"Training failed for {key_name}", exc_info=exc)
                key_bucket = results_per_key.setdefault(key_name, {})
                key_bucket['multi_class'] = {'error': str(exc)}
                current_model_index += 1

        # Train multi-label keys (one model per key with sigmoid activation)
        multilabel_key_files = {k: v for k, v in bundle.training_files.items() if k in multilabel_keys}

        for key_name in multilabel_key_files.keys():
            key_file_path = multilabel_key_files[key_name]

            # RESUME LOGIC: Check if model is already trained and skip if complete
            is_complete, trained_path = _is_model_already_trained(key_name, model_name, 'normal_training')
            if is_complete:
                _skip_info = ""
                try:
                    import glob as _g
                    _csvs = _g.glob(f"{session_dir}/training_metrics/normal_training/{key_name}/**/training.csv", recursive=True)
                    if _csvs:
                        import csv as _csv
                        with open(_csvs[0]) as _f:
                            _lines = [l for l in _f if not l.startswith('#')]
                            _rows = list(_csv.DictReader(_lines))
                        if _rows:
                            _best = max(_rows, key=lambda r: float(r.get('macro_f1', 0)))
                            _f1 = float(_best.get('macro_f1', 0))
                            _ep = _best.get('epoch', '?')
                            _total_ep = len(_rows)
                            _skip_info = f" (F1={_f1:.3f} at epoch {_ep}/{_total_ep})"
                except Exception:
                    pass
                self.console.print(f"\n[green]⏭ Skipping '{key_name}' — already trained{_skip_info}[/green]")
                current_model_index += 1
                continue

            self.console.print(f"\n[bold cyan]Training multi-label model for key: {key_name}[/bold cyan]")
            self.console.print(f"[dim]Model {current_model_index}/{global_total_models}[/dim]")
            self.console.print(f"[dim]Dataset: {key_file_path}[/dim]\n")

            # Use multi-label trainer for this key
            key_config = {
                'input_file': str(key_file_path),
                'output_dir': str(output_dir / key_name),
                'training_strategy': 'multi-label',  # Multi-label with sigmoid
                'training_approach': 'multi-label',  # CRITICAL: Required for true multi-label routing in model_trainer
                'text_column': 'text',
                'label_column': 'labels',
                'epochs': epochs,
                'model_name': model_name,
                'train_by_language': needs_language_training,
                'progress_callback': progress_callback,
            }

            # Add global progress tracking
            key_config['global_model_index'] = current_model_index
            key_config['global_total_models'] = global_total_models
            key_config['global_completed_epochs'] = global_completed_epochs
            key_config['global_total_epochs'] = global_total_epochs
            key_config['global_start_time'] = global_start_time

            if models_by_language:
                key_config["models_by_language"] = models_by_language

            try:
                key_result = trainer.train(key_config)
                models_learned = _register_model_metrics(base_identifier=key_name, stage='multi-label', owning_key=key_name, result_dict=key_result)
                increment = models_learned if models_learned > 0 else 1
                current_model_index += increment
                # Update global completed epochs
                returned_epochs = key_result.get('global_completed_epochs') if isinstance(key_result, dict) else None
                if returned_epochs is not None:
                    global_completed_epochs = max(global_completed_epochs, returned_epochs)
                self.logger.debug(f"[EPOCH_TRACK] After multi-label model {key_name}: global_completed_epochs={global_completed_epochs}")
                total_training_time += float(key_result.get('training_time', 0.0) or 0.0)
                key_bucket = results_per_key.setdefault(key_name, {})
                key_bucket['multi_label'] = key_result
                self.console.print(f"[green]Completed {key_name}: F1={key_result.get('best_f1_macro', 0):.4f}[/green]")
            except Exception as exc:
                self.console.print(f"[red]✗ Failed to train {key_name}: {exc}[/red]")
                self.logger.exception(f"Training failed for {key_name}", exc_info=exc)
                key_bucket = results_per_key.setdefault(key_name, {})
                key_bucket['multi_label'] = {'error': str(exc)}
                current_model_index += 1

        onevsall_results: Dict[str, Dict[str, Any]] = {}
        onevsall_key_value_map: Dict[str, Set[str]] = {}
        # Reuse pre-generated label display map if available from parallel check
        onevsall_label_display_map: Dict[str, Dict[str, Any]] = hybrid_label_display_map if hybrid_label_display_map else {}

        # Switch active model selections to the one-vs-all stage
        models_by_language = binary_models_by_language
        needs_language_training = bool(binary_train_by_language)
        train_by_language_flag = bool(binary_train_by_language)
        model_name = binary_model_name

        # Train one-vs-all keys (binary models per label)
        if onevsall_keys and 'onevsall_multilabel' in bundle.training_files:
            onevsall_file = Path(bundle.training_files['onevsall_multilabel'])
            self.console.print(f"\n[bold yellow]Training one-vs-all models for {len(onevsall_keys)} keys[/bold yellow]")

            # Reuse pre-generated binary files if available from parallel check
            binary_category_files: Dict[str, Path] = {}
            if hybrid_binary_files:
                self.console.print(f"[dim]Reusing {len(hybrid_binary_files)} pre-generated binary datasets[/dim]")
                binary_category_files = hybrid_binary_files
            else:
                # Generate binary datasets (fallback if not pre-generated)
                # Validate one-vs-all file before splitting into binaries
                try:
                    validated_file, was_filtered = self._validate_and_filter_insufficient_labels(
                        input_file=str(onevsall_file),
                        strategy='multi-label',
                        min_samples=2,
                        auto_remove=True,
                        train_by_language=needs_language_training
                    )
                    if was_filtered:
                        onevsall_file = Path(validated_file)
                        self.console.print(f"[green]Using filtered dataset for one-vs-all[/green]")
                except ValueError as e:
                    self.console.print(f"[red]✗ Failed to validate one-vs-all file: {e}[/red]")
                    self.logger.error(f"Validation failed for one-vs-all: {e}")
                    onevsall_file = None

                if onevsall_file and onevsall_file.exists():
                    binary_category_files, key_value_map, label_display_map = _create_onevsall_binary_datasets(onevsall_file, target_keys=onevsall_keys)
                    onevsall_key_value_map = key_value_map
                    onevsall_label_display_map = label_display_map

            if not binary_category_files:
                self.console.print("[yellow][!] No binary datasets available for one-vs-all keys[/yellow]")
            else:
                self.console.print(f"[dim]  Training {len(binary_category_files)} binary models for one-vs-all[/dim]")

                for label_name in binary_category_files.keys():
                    dataset_path = Path(binary_category_files[label_name])
                    if '_' in label_name:
                        key_prefix, raw_value = label_name.split('_', 1)
                    else:
                        key_prefix, raw_value = label_name, label_name

                    per_key_bucket = results_per_key.setdefault(key_prefix, {})
                    onevsall_bucket = per_key_bucket.setdefault('one_vs_all', {})
                    display_info = onevsall_label_display_map.get(label_name, {})

                    # RESUME LOGIC: Check if binary model is already trained and skip if complete
                    is_complete, existing_path = _is_model_already_trained(label_name, model_name, 'normal_training')
                    if is_complete:
                        # Read best metrics from training CSV if available
                        _skip_info = ""
                        try:
                            import glob as _g
                            _csvs = _g.glob(f"{session_dir}/training_metrics/normal_training/{label_name}/**/training.csv", recursive=True)
                            if _csvs:
                                import csv as _csv
                                with open(_csvs[0]) as _f:
                                    _lines = [l for l in _f if not l.startswith('#')]
                                    _rows = list(_csv.DictReader(_lines))
                                if _rows:
                                    _best = max(_rows, key=lambda r: float(r.get('macro_f1', r.get('f1_1', 0))))
                                    _f1 = float(_best.get('f1_1', _best.get('macro_f1', 0)))
                                    _ep = _best.get('epoch', '?')
                                    _total_ep = len(_rows)
                                    _skip_info = f" (F1={_f1:.3f} at epoch {_ep}/{_total_ep})"
                        except Exception:
                            pass
                        self.console.print(f"\n[green]⏭ Skipping '{label_name}' — already trained{_skip_info}[/green]")
                        self.console.print(f"[dim]   Model path: {existing_path}[/dim]")
                        onevsall_bucket[raw_value] = {
                            'skipped': True,
                            'model_path': existing_path,
                            'reason': 'already_trained'
                        }
                        trained_models_map[label_name] = existing_path
                        skipped_models_count += 1
                        current_model_index += 1
                        continue

                    # Clean up incomplete checkpoints before retraining
                    _cleanup_incomplete_model(label_name, 'normal_training')

                    self.console.print(f"\n[cyan]Training binary model for: {label_name}[/cyan]")

                    try:
                        validated_file, was_filtered = self._validate_and_filter_insufficient_labels(
                            input_file=str(dataset_path),
                            strategy='single-label',
                            min_samples=2,
                            auto_remove=True,
                            train_by_language=needs_language_training
                        )
                        if was_filtered:
                            dataset_path = Path(validated_file)
                            self.console.print(f"[green]Using filtered dataset for {label_name}[/green]")
                    except ValueError as e:
                        self.console.print(f"[red]✗ Failed to validate binary dataset for {label_name}: {e}[/red]")
                        self.logger.error(f"Validation failed for binary dataset {label_name}: {e}")
                        onevsall_bucket[raw_value] = {'error': str(e)}
                        current_model_index += 1
                        continue

                    label_config = {
                        'input_file': str(dataset_path),
                        'model_name': model_name,
                        'num_epochs': epochs,
                        'batch_size': _get_optimal_batch_size(model_name),  # Dynamic batch size based on system resources
                        'output_dir': str(output_dir / "onevsall"),
                        'text_column': bundle.text_column or 'text',
                        'label_column': 'label',
                        'training_strategy': 'single-label',
                        'training_approach': 'one-vs-all',  # Explicit: Binary classification for each category
                        'category_name': label_name,
                        'reinforced_learning': enable_reinforced_learning,
                        'session_id': session_id,
                        'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                        'auto_remove_insufficient_labels': True,  # Already validated above
                        'global_total_models': global_total_models,
                        'global_current_model': current_model_index + 1,
                        'global_total_epochs': global_total_epochs,
                        'global_max_epochs': global_max_epochs,
                        'global_completed_epochs': global_completed_epochs,
                        'global_start_time': global_start_time,
                        'train_by_language': needs_language_training,
                        'confirmed_languages': list(languages) if languages else None,
                    }

                    if display_info:
                        negative_display = display_info.get('negative_label')
                        positive_display = display_info.get('positive_label')
                        if negative_display and positive_display:
                            label_config['class_names_override'] = [negative_display, positive_display]
                            label_config['onevsall_display_labels'] = {
                                'positive': positive_display,
                                'negative': negative_display
                            }
                        label_config['onevsall_sample_counts'] = {
                            'positive': display_info.get('positive_count'),
                            'negative': display_info.get('negative_count'),
                            'total': display_info.get('total_samples')
                        }

                    if models_by_language:
                        label_config["models_by_language"] = models_by_language

                    # Merge reinforced learning parameters and shared extras
                    for extra_key, extra_value in extra_config.items():
                        if extra_key not in label_config:
                            label_config[extra_key] = extra_value

                    try:
                        label_result = trainer.train(label_config)
                        models_learned = _register_model_metrics(
                            base_identifier=label_name,
                            stage='one-vs-all',
                            owning_key=key_prefix,
                            result_dict=label_result
                        )
                        increment = models_learned if models_learned > 0 else 1
                        current_model_index += increment
                        # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                        returned_epochs = label_result.get('global_completed_epochs') if isinstance(label_result, dict) else None
                        if returned_epochs is not None:
                            global_completed_epochs = max(global_completed_epochs, returned_epochs)
                        self.logger.debug(f"[EPOCH_TRACK] After one-vs-all model {label_name}: global_completed_epochs={global_completed_epochs}")
                        total_training_time += float(label_result.get('training_time', 0.0) or 0.0)
                        onevsall_bucket[raw_value] = label_result
                        onevsall_results[label_name] = label_result
                        if display_info:
                            label_result['class_names'] = [
                                display_info.get('negative_label'),
                                display_info.get('positive_label')
                            ]
                            label_result['sample_counts'] = {
                                'positive': display_info.get('positive_count'),
                                'negative': display_info.get('negative_count'),
                                'total': display_info.get('total_samples')
                            }
                        self.console.print(f"[green]Completed {label_name}: Accuracy={label_result.get('accuracy', 0):.4f}, F1={label_result.get('best_f1_macro', 0):.4f}[/green]")
                    except Exception as exc:
                        self.console.print(f"[red]✗ Failed to train {label_name}: {exc}[/red]")
                        self.logger.exception(f"Training failed for {label_name}", exc_info=exc)
                        onevsall_bucket[raw_value] = {'error': str(exc)}
                        current_model_index += 1
        else:
            self.console.print("[yellow][!] Skipping one-vs-all training: dataset missing[/yellow]")

        # Aggregate results
        successful_records = [rec for rec in trained_model_records if rec['f1'] > 0 or rec['accuracy'] > 0]
        avg_accuracy = sum(rec['accuracy'] for rec in successful_records) / len(successful_records) if successful_records else 0.0
        avg_f1 = sum(rec['f1'] for rec in successful_records) / len(successful_records) if successful_records else 0.0

        if not trained_model_records:
            self.console.print("[red]All trainings failed[/red]")
            return {
                'runtime_params': runtime_params,
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'best_f1_macro': None,
                'error': 'All trainings failed'
            }

        best_record = max(trained_model_records, key=lambda rec: rec['f1'])
        best_model_identifier = best_record['id']
        best_model_path = trained_models_map.get(best_model_identifier, '')

        per_model_metrics = {
            rec['id']: {
                'stage': rec['stage'],
                'key': rec['key'],
                'language': rec.get('language'),
                'model_name': rec.get('model_name'),
                'f1_macro': rec['f1'],
                'accuracy': rec['accuracy'],
                'model_path': rec['model_path'],
                'class_names': rec.get('class_names')
            }
            for rec in trained_model_records
        }

        result = {
            'best_model': best_model_identifier,
            'accuracy': avg_accuracy,
            'best_f1_macro': best_record['f1'],
            'best_f1': best_record['f1'],
            'model_path': best_model_path,
            'training_time': total_training_time,
            'models_trained': list(trained_models_map.keys()),
            'models_trained_count': len(trained_models_map),
            'training_approach': training_approach_from_metadata,
            'per_key_results': results_per_key,
            'onevsall_results': onevsall_results,
            'onevsall_value_map': {k: sorted(v) for k, v in onevsall_key_value_map.items()},
            'onevsall_label_info': onevsall_label_display_map,
            'trained_models': trained_models_map,
            'trained_model_paths': trained_models_map,
            'metrics': {
                'average_accuracy': avg_accuracy,
                'average_f1_macro': avg_f1,
                'per_model': per_model_metrics
            }
        }

        stage_summary, total_expected_models = _compute_stage_summary(
            trained_model_records=trained_model_records,
            results_per_key=results_per_key,
            total_multiclass_models=total_multiclass_models,
            total_onevsall_models=total_onevsall_models,
            multiclass_needs_language_training=multiclass_needs_language_training,
            binary_train_by_language=binary_train_by_language,
            stage_models=stage_models,
            languages=languages,
            total_training_time=total_training_time,
            avg_accuracy=avg_accuracy,
            avg_f1_macro=avg_f1,
        )

        result['stage_summary'] = stage_summary
        result['total_expected_models'] = total_expected_models
    elif training_approach_from_metadata == 'multi-label':
        # ============================================================
        # TRUE MULTI-LABEL TRAINING
        # ============================================================
        # Check if we have per-key multi-label training (multiple keys selected)
        multi_label_per_key = bundle.metadata.get('multi_label_per_key', False) if hasattr(bundle, 'metadata') else False
        key_files = {}
        if hasattr(bundle, 'training_files') and bundle.training_files:
            key_files = {k: v for k, v in bundle.training_files.items() if k != 'multilabel'}

        if multi_label_per_key and key_files:
            # ============================================================
            # PER-KEY MULTI-LABEL: Train ONE multi-label model PER KEY
            # ============================================================
            # Each key (e.g., themes, sentiment, parties) gets its own multi-label model
            self.console.print(f"\n[magenta]Multi-Label Training: {len(key_files)} models (one per key)[/magenta]\n")

            # Initialize global progress tracking
            import time
            global_start_time = time.time()

            num_keys = len(key_files)
            num_languages = int(len(languages)) if languages else 1

            if needs_language_training and num_languages > 1:
                global_total_models = num_keys * num_languages
                self.logger.info(f"[EPOCH CALC] Multi-label + per-language: {num_keys} keys × {num_languages} languages = {global_total_models} total models")
            else:
                global_total_models = num_keys
                self.logger.info(f"[EPOCH CALC] Multi-label + multilingual: {num_keys} keys = {global_total_models} total models")

            global_total_epochs = global_total_models * epochs

            if enable_reinforced_learning and manual_rl_epochs is not None:
                global_max_epochs = global_total_models * (epochs + manual_rl_epochs)
            else:
                global_max_epochs = global_total_epochs

            self.logger.info("=" * 80)
            self.logger.info("️ PER-KEY MULTI-LABEL TRAINING")
            self.logger.info(f"  Model: {model_name}")
            self.logger.info(f"  Keys: {list(key_files.keys())}")
            self.logger.info(f"  Total models: {global_total_models}")
            self.logger.info(f"  Epochs per model: {epochs}")
            self.logger.info("=" * 80)

            # ============================================================
            # PRE-VALIDATION PHASE: Validate ALL keys BEFORE training starts
            # ============================================================
            self.console.print(f"\n[bold cyan]Pre-validation Phase: Checking all {len(key_files)} keys...[/bold cyan]\n")

            validated_key_files = {}  # {key_name: validated_file_path}
            keys_to_skip = []

            for key_name, key_file_path in key_files.items():
                self.console.print(f"  Validating [cyan]{key_name}[/cyan]...")
                try:
                    filtered_file, was_filtered = self._validate_and_filter_insufficient_labels(
                        input_file=str(key_file_path),
                        strategy='multi-label',
                        min_samples=2,
                        auto_remove=False,
                        train_by_language=needs_language_training
                    )
                    validated_key_files[key_name] = filtered_file
                    if was_filtered:
                        self.console.print(f"    [green]✓ Filtered dataset ready for {key_name}[/green]")
                    else:
                        self.console.print(f"    [green]✓ All labels valid for {key_name}[/green]")
                except ValueError as e:
                    self.console.print(f"    [yellow][!]  Skipping {key_name}: {e}[/yellow]")
                    keys_to_skip.append(key_name)
                except Exception as e:
                    self.logger.warning(f"Label validation failed for {key_name}: {e}")
                    validated_key_files[key_name] = str(key_file_path)  # Use original file

            # Update model counts after validation (some keys may be skipped)
            if keys_to_skip:
                num_valid_keys = len(validated_key_files)
                self.console.print(f"\n[yellow][!]  {len(keys_to_skip)} key(s) will be skipped due to insufficient samples[/yellow]")
                if num_valid_keys == 0:
                    self.console.print("[red]✗ No valid keys remaining for training[/red]")
                    return {
                        'runtime_params': runtime_params,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None,
                        'error': 'All keys have insufficient samples'
                    }
                # Update global totals
                if needs_language_training and num_languages > 1:
                    global_total_models = num_valid_keys * num_languages
                else:
                    global_total_models = num_valid_keys
                global_total_epochs = global_total_models * epochs
                if enable_reinforced_learning and manual_rl_epochs is not None:
                    global_max_epochs = global_total_models * (epochs + manual_rl_epochs)
                else:
                    global_max_epochs = global_total_epochs
                self.console.print(f"[cyan]Updated: {num_valid_keys} keys × {epochs} epochs = {global_total_models} models[/cyan]")

            self.console.print(f"\n[green][OK] Pre-validation complete. Ready to train {len(validated_key_files)} models.[/green]\n")

            # ============================================================
            # CHECK FOR PARALLEL TRAINING OPTION (Smart Scheduler)
            # ============================================================
            num_models_to_train = len(validated_key_files)
            use_parallel_training = False

            try:
                from llm_tool.trainers.hybrid_parallel_trainer import HardwareAnalyzer, HybridParallelTrainer
                from llm_tool.trainers.task_scheduler import SchedulerConfig
                from rich.prompt import Confirm

                # Use smart scheduler to determine if parallel training is beneficial
                scheduler_config = SchedulerConfig()
                should_offer, scheduler_message = HybridParallelTrainer.should_offer_parallel(
                    num_models_to_train, scheduler_config
                )

                hw_analyzer = HardwareAnalyzer()
                can_parallel, parallel_reason = hw_analyzer.can_use_parallel_training(num_models_to_train)

                if can_parallel and should_offer:
                    # Determine if strongly recommended
                    strongly_recommended = num_models_to_train >= scheduler_config.recommended_tasks_for_parallel
                    min_threshold = scheduler_config.min_tasks_for_parallel
                    recommended_threshold = scheduler_config.recommended_tasks_for_parallel

                    self.console.print(f"\n[bold cyan]Parallel Training Available[/bold cyan]")

                    # Clear recommendation panel
                    from rich.panel import Panel
                    if strongly_recommended:
                        rec_text = (
                            f"[bold green][OK] STRONGLY RECOMMENDED[/bold green]\n\n"
                            f"You have [bold]{num_models_to_train} models[/bold] to train, which is ≥{recommended_threshold} models.\n"
                            f"Parallel training will provide [green]significant speedup[/green] by utilizing GPU + CPU workers simultaneously."
                        )
                        rec_style = "green"
                        default_choice = True
                    else:
                        rec_text = (
                            f"[bold yellow][!] OPTIONAL - MARGINAL BENEFIT[/bold yellow]\n\n"
                            f"You have [bold]{num_models_to_train} models[/bold] to train ({min_threshold}-{recommended_threshold-1} range).\n"
                            f"Parallel training offers [yellow]moderate speedup[/yellow], but overhead may reduce benefits.\n\n"
                            f"[dim]• Sequential (GPU-only) is often faster for <{recommended_threshold} models\n"
                            f"• Parallel becomes efficient with ≥{recommended_threshold} models[/dim]"
                        )
                        rec_style = "yellow"
                        default_choice = False

                    self.console.print(Panel(rec_text, title="Recommendation", border_style=rec_style))

                    # Show resource plan
                    plan = hw_analyzer.create_resource_plan(model_name, num_models_to_train)

                    from rich.table import Table
                    from rich import box

                    plan_table = Table(
                        title="Resource Allocation Plan",
                        show_header=True,
                        header_style="bold cyan",
                        box=box.ROUNDED,
                        expand=True,
                    )
                    plan_table.add_column("Device", style="cyan")
                    plan_table.add_column("Type")
                    plan_table.add_column("Batch Size", justify="center")
                    plan_table.add_column("Workers", justify="center")

                    if plan.gpu_allocation:
                        g = plan.gpu_allocation
                        plan_table.add_row(f"GPU {g.device_id.upper()}", "GPU (Priority)", str(g.batch_size), str(g.num_workers))

                    for c in plan.cpu_allocations:
                        plan_table.add_row(f"CPU {c.device_id}", "CPU", str(c.batch_size), str(c.num_workers))

                    self.console.print(plan_table)
                    self.console.print(f"\n[dim]Total: {plan.total_parallel_workers} workers available[/dim]")

                    # Ask user with clear default based on recommendation
                    if strongly_recommended:
                        prompt_text = "\n[bold]Use parallel training? [green](recommended: yes)[/green][/bold]"
                    else:
                        prompt_text = "\n[bold]Use parallel training? [yellow](recommended: no)[/yellow][/bold]"

                    use_parallel_training = Confirm.ask(prompt_text, default=default_choice)

                    if use_parallel_training:
                        self.console.print("\n[green]✓ Parallel training enabled with smart task scheduling[/green]\n")
                elif not should_offer:
                    self.console.print(f"[dim]{scheduler_message}[/dim]")
            except ImportError:
                self.logger.debug("Parallel training not available (missing dependencies)")
            except Exception as e:
                self.logger.warning(f"Could not check parallel training: {e}")

            # ============================================================
            # RUN PARALLEL OR SEQUENTIAL TRAINING
            # ============================================================
            if use_parallel_training:
                # Convert validated_key_files to the format expected by _run_parallel_training
                category_files = {k: Path(v) for k, v in validated_key_files.items()}

                return self._run_parallel_training(
                    category_files=category_files,
                    model_name=model_name,
                    epochs=epochs,
                    output_dir=output_dir,
                    session_id=session_id,
                    bundle=bundle,
                    extra_config=extra_config,
                    enable_reinforced_learning=enable_reinforced_learning,
                    manual_rl_epochs=manual_rl_epochs,
                    languages=languages,
                    models_by_language=models_by_language,
                    needs_language_training=needs_language_training,
                    runtime_params=runtime_params,
                    training_approach='multi-label',
                    quick_params=quick_params if 'quick_params' in dir() else None,
                )

            # Sequential training (original code path)
            global_completed_epochs = 0
            results_per_key = {}

            # Create progress callback to track completed epochs across all models
            def progress_callback(**metrics):
                """Callback to sync global completed epochs counter from authoritative source."""
                nonlocal global_completed_epochs
                # Update from callback metrics if available (bert_base.py is the authoritative source)
                if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                    global_completed_epochs = metrics['global_completed_epochs']

            for idx, (key_name, input_file_to_use) in enumerate(validated_key_files.items(), 1):
                self.console.print(f"\n[bold magenta]Training multi-label model for key '{key_name}'[/bold magenta] ({Path(input_file_to_use).name})")

                # Create config for this key's multi-label model
                key_config = {
                    'input_file': input_file_to_use,
                    'model_name': model_name,
                    'num_epochs': epochs,
                    'output_dir': str(output_dir),
                    'text_column': bundle.text_column,
                    'label_column': bundle.label_column,
                    'training_strategy': 'multi-label',
                    'training_approach': 'multi-label',
                    'multi_label': True,
                    'multi_label_threshold': bundle.metadata.get('multi_label_threshold', 0.5),
                    'category_name': key_name,  # Used for output directory naming
                    'reinforced_learning': enable_reinforced_learning,
                    'session_id': session_id,
                    'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                    # Global progress tracking
                    'global_total_models': global_total_models,
                    'global_current_model': idx,
                    'global_total_epochs': global_total_epochs,
                    'global_max_epochs': global_max_epochs,
                    'global_completed_epochs': global_completed_epochs,
                    'global_start_time': global_start_time,
                    'progress_callback': progress_callback,  # CRITICAL: For epoch counting
                    'train_by_language': needs_language_training,
                    'confirmed_languages': list(languages) if languages else None,
                    'auto_remove_insufficient_labels': True,  # Pre-validated upstream
                }

                # Merge extra config
                for extra_key, extra_value in extra_config.items():
                    if extra_key not in key_config:
                        key_config[extra_key] = extra_value

                if models_by_language:
                    key_config["models_by_language"] = models_by_language

                if enable_reinforced_learning and manual_rl_epochs is not None:
                    key_config["reinforced_epochs"] = int(manual_rl_epochs)

                try:
                    key_result = trainer.train(key_config)
                    # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                    returned_epochs = key_result.get('global_completed_epochs') if isinstance(key_result, dict) else None
                    if returned_epochs is not None:
                        global_completed_epochs = max(global_completed_epochs, returned_epochs)
                    self.logger.debug(f"[EPOCH_TRACK] After multi-label model {key_name}: global_completed_epochs={global_completed_epochs}")
                    results_per_key[key_name] = key_result
                    f1 = key_result.get('f1_macro') or key_result.get('best_f1_macro', 0)
                    self.console.print(f"[green]Completed {key_name}: F1={f1:.4f}[/green]")
                except Exception as exc:
                    self.console.print(f"[red]✗ Failed to train {key_name}: {exc}[/red]")
                    self.logger.exception(f"Training failed for {key_name}", exc_info=exc)
                    results_per_key[key_name] = {'error': str(exc)}

            # Aggregate results
            successful_results = [r for r in results_per_key.values() if 'error' not in r]
            if successful_results:
                avg_f1 = sum(r.get('f1_macro') or r.get('best_f1_macro', 0) for r in successful_results) / len(successful_results)

                result = {
                    'best_model': model_name,
                    'f1_macro': avg_f1,
                    'best_f1_macro': avg_f1,
                    'model_path': str(output_dir),
                    'training_time': sum(r.get('training_time', 0) for r in successful_results),
                    'models_trained': len(successful_results),
                    'total_keys': len(key_files),
                    'per_key_results': results_per_key,
                    'training_approach': 'multi-label-per-key'
                }

                self.console.print(f"\n[green][OK] Per-key multi-label training complete![/green]")
                self.console.print(f"[cyan]  Models trained: {len(successful_results)}/{len(key_files)}[/cyan]")
                self.console.print(f"[cyan]  Average F1 (macro): {avg_f1:.4f}[/cyan]")

                # ============================================================
                # GENERATE TRAINING SUMMARY CHART
                # ============================================================
                try:
                    from llm_tool.trainers.training_metrics_chart import generate_training_summary_chart

                    self.console.print(f"\n[bold cyan]Generating Training Summary Chart...[/bold cyan]")

                    # Calculate total training time
                    total_time = time.time() - global_start_time if global_start_time else sum(
                        r.get('training_time', 0) for r in successful_results
                    )

                    summary_chart_path = generate_training_summary_chart(
                        output_dir=str(output_dir),
                        session_id=session_id,
                        model_name=model_name,
                        results_per_key=results_per_key,
                        training_approach='multi-label-per-key',
                        total_training_time=total_time,
                    )

                    if summary_chart_path:
                        self.console.print(f"[green]Training summary chart saved:[/green] {summary_chart_path}")
                    else:
                        self.console.print("[yellow][!] Could not generate summary chart[/yellow]")

                except Exception as chart_exc:
                    self.logger.warning(f"Failed to generate training summary chart: {chart_exc}")
                    self.console.print(f"[yellow][!] Summary chart generation failed: {chart_exc}[/yellow]")

                # Generate comprehensive cross-model summary chart (enhanced version)
                try:
                    from llm_tool.trainers.training_metrics_chart import generate_comprehensive_summary_chart

                    self.console.print("\n[bold cyan]Generating Comprehensive Summary Chart...[/bold cyan]")

                    # Determine session directory for disk scan
                    _session_dir = None
                    if hasattr(self, 'current_session_manager') and self.current_session_manager:
                        _sm_dir = getattr(self.current_session_manager, 'session_dir', None)
                        if _sm_dir:
                            _session_dir = str(_sm_dir)

                    comp_chart_path = generate_comprehensive_summary_chart(
                        output_dir=str(output_dir),
                        session_id=session_id,
                        model_name=model_name,
                        session_dir=_session_dir,
                        results_per_key=results_per_key,
                        training_approach='multi-label-per-key',
                        total_training_time=total_time,
                    )

                    if comp_chart_path:
                        self.console.print(f"[green]Comprehensive summary chart saved:[/green] {comp_chart_path}")
                    else:
                        self.console.print("[yellow][!] Could not generate comprehensive summary chart[/yellow]")

                except Exception as comp_exc:
                    self.logger.warning(f"Failed to generate comprehensive summary chart: {comp_exc}")
                    self.console.print(f"[yellow][!] Comprehensive chart failed: {comp_exc}[/yellow]")

            else:
                self.console.print("[red]All key trainings failed[/red]")
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': 'All key trainings failed'
                }

        else:
            # ============================================================
            # SINGLE MODEL MULTI-LABEL: One model with all labels (legacy)
            # ============================================================
            self.console.print(f"\n[magenta]True Multi-Label Training: Single model with sigmoid activation[/magenta]\n")

            input_file_to_use = str(bundle.primary_file)

            # Validate and filter insufficient labels
            if bundle.primary_file:
                try:
                    filtered_file, was_filtered = self._validate_and_filter_insufficient_labels(
                        input_file=str(bundle.primary_file),
                        strategy='multi-label',
                        min_samples=2,
                        auto_remove=False,
                        train_by_language=needs_language_training
                    )
                    if was_filtered:
                        input_file_to_use = filtered_file
                        self.console.print(f"[green]Using filtered training dataset[/green]\n")
                except ValueError as e:
                    self.console.print(f"[red]{e}[/red]")
                    return {
                        'runtime_params': runtime_params,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None,
                        'error': str(e)
                    }
                except Exception as e:
                    self.logger.warning(f"Label validation failed: {e}")

            # Initialize global progress tracking
            import time
            global_start_time = time.time()
            global_total_models = 1
            global_total_epochs = epochs

            if enable_reinforced_learning and manual_rl_epochs is not None:
                global_max_epochs = epochs + manual_rl_epochs
            else:
                global_max_epochs = global_total_epochs

            self.logger.info("=" * 80)
            self.logger.info("️ TRUE MULTI-LABEL TRAINING (SINGLE MODEL)")
            self.logger.info(f"  Model: {model_name}")
            self.logger.info(f"  Input: {input_file_to_use}")
            self.logger.info(f"  Mode: Single model with BCEWithLogitsLoss + sigmoid")
            self.logger.info(f"  Epochs: {epochs}")
            self.logger.info("=" * 80)

            global_completed_epochs = 0

            # Create progress callback to track completed epochs
            def progress_callback(**metrics):
                """Callback to sync global completed epochs counter from authoritative source."""
                nonlocal global_completed_epochs
                # Update from callback metrics if available (bert_base.py is the authoritative source)
                if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                    global_completed_epochs = metrics['global_completed_epochs']

            result = trainer.train({
                'input_file': input_file_to_use,
                'model_name': model_name,
                'num_epochs': epochs,
                'output_dir': str(output_dir),
                'text_column': bundle.text_column,
                'label_column': bundle.label_column,
                'training_strategy': 'multi-label',
                'training_approach': 'multi-label',
                'multi_label': True,
                'multi_label_threshold': bundle.metadata.get('multi_label_threshold', 0.5),
                'reinforced_learning': enable_reinforced_learning,
                'session_id': session_id,
                'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                'global_total_models': global_total_models,
                'global_current_model': 1,
                'global_total_epochs': global_total_epochs,
                'global_max_epochs': global_max_epochs,
                'global_completed_epochs': global_completed_epochs,
                'global_start_time': global_start_time,
                'progress_callback': progress_callback,  # CRITICAL: For epoch counting
                **extra_config
            })

            # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
            returned_epochs = result.get('global_completed_epochs') if isinstance(result, dict) else None
            if returned_epochs is not None:
                global_completed_epochs = max(global_completed_epochs, returned_epochs)
            self.logger.debug(f"[EPOCH_TRACK] After true multi-label single model: global_completed_epochs={global_completed_epochs}")

            self.console.print(f"\n[green][OK] True multi-label training complete![/green]")
            if result.get('f1_macro'):
                self.console.print(f"[cyan]  F1 (macro): {result.get('f1_macro', 0):.4f}[/cyan]")
            if result.get('subset_accuracy'):
                self.console.print(f"[cyan]  Subset Accuracy: {result.get('subset_accuracy', 0):.4f}[/cyan]")
            if result.get('hamming_loss') is not None:
                self.console.print(f"[cyan]  Hamming Loss: {result.get('hamming_loss', 0):.4f}[/cyan]")

    elif training_approach_from_metadata == 'multi-class' and hasattr(bundle, 'training_files') and bundle.training_files:
        # Multi-class training with multiple keys: train ONE model PER KEY
        # Extract the key files (exclude 'multilabel' key)
        key_files = {k: v for k, v in bundle.training_files.items() if k != 'multilabel'}

        if key_files:
            self.console.print(f"\n[cyan]Multi-class training: {len(key_files)} models (one per key)[/cyan]\n")

            # Initialize global progress tracking for multi-class training
            import time
            global_start_time = time.time()

            # CRITICAL: Calculate total models for multi-class training
            # Multi-class: 1 model per key (each model handles all classes in that key)
            num_keys = len(key_files)
            num_languages = int(len(languages)) if languages else 1

            if needs_language_training and num_languages > 1:
                # Per-language training: one model per (key, language) combination
                global_total_models = num_keys * num_languages
                self.logger.info(f"[EPOCH CALC] Multi-class + per-language: {num_keys} keys × {num_languages} languages = {global_total_models} total models")
            else:
                # Multilingual model: one model per key (handles all languages)
                global_total_models = num_keys
                self.logger.info(f"[EPOCH CALC] Multi-class + multilingual: {num_keys} keys = {global_total_models} total models")

            global_total_epochs = global_total_models * epochs

            # Calculate maximum possible epochs (if all models trigger reinforced learning)
            if enable_reinforced_learning and manual_rl_epochs is not None:
                global_max_epochs = global_total_models * (epochs + manual_rl_epochs)
            else:
                global_max_epochs = global_total_epochs

            # DEBUGGING: Log the epoch calculation
            self.logger.info("="*80)
            self.logger.info("GLOBAL EPOCHS CALCULATION DEBUG")
            self.logger.info(f"  Training mode: multi-class")
            self.logger.info(f"  Language training: {'per-language' if needs_language_training else 'multilingual'}")
            self.logger.info(f"  Number of keys: {num_keys}")
            self.logger.info(f"  Number of languages: {num_languages}")
            self.logger.info(f"  Languages: {sorted(languages) if languages else 'N/A'}")
            self.logger.info(f"  Base epochs per model: {epochs}")
            self.logger.info(f"  RL epochs per model: {manual_rl_epochs if manual_rl_epochs else 'None'}")
            self.logger.info(f"  CALCULATED global_total_models: {global_total_models}")
            self.logger.info(f"  CALCULATED global_total_epochs: {global_total_epochs}")
            self.logger.info(f"  CALCULATED global_max_epochs: {global_max_epochs}")
            self.logger.info("="*80)

            global_completed_epochs = 0

            # ============================================================
            # PRE-VALIDATION PHASE: Validate ALL keys BEFORE training starts
            # ============================================================
            self.console.print(f"\n[bold cyan]Pre-validation Phase: Checking all {len(key_files)} keys...[/bold cyan]\n")

            validated_key_files = {}  # {key_name: validated_file_path}
            keys_to_skip = []

            for key_name, key_file_path in key_files.items():
                self.console.print(f"  Validating [cyan]{key_name}[/cyan]...")
                try:
                    filtered_file, was_filtered = self._validate_and_filter_insufficient_labels(
                        input_file=str(key_file_path),
                        strategy='single-label',
                        min_samples=2,
                        auto_remove=False,
                        train_by_language=needs_language_training
                    )
                    validated_key_files[key_name] = Path(filtered_file)
                    if was_filtered:
                        self.console.print(f"    [green]✓ Filtered dataset ready for {key_name}[/green]")
                    else:
                        self.console.print(f"    [green]✓ All labels valid for {key_name}[/green]")
                except ValueError as e:
                    self.console.print(f"    [yellow][!]  Skipping {key_name}: {e}[/yellow]")
                    keys_to_skip.append(key_name)
                except Exception as e:
                    self.logger.warning(f"Label validation failed for {key_name}: {e}")
                    validated_key_files[key_name] = key_file_path  # Use original file

            # Update key_files to use validated paths
            key_files = validated_key_files

            # Update model counts after validation (some keys may be skipped)
            if keys_to_skip:
                num_valid_keys = len(validated_key_files)
                self.console.print(f"\n[yellow][!]  {len(keys_to_skip)} key(s) will be skipped due to insufficient samples[/yellow]")
                if num_valid_keys == 0:
                    self.console.print("[red]✗ No valid keys remaining for training[/red]")
                    return {
                        'runtime_params': runtime_params,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None,
                        'error': 'All keys have insufficient samples'
                    }
                # Recalculate global totals
                num_keys = num_valid_keys
                if needs_language_training and num_languages > 1:
                    global_total_models = num_keys * num_languages
                else:
                    global_total_models = num_keys
                global_total_epochs = global_total_models * epochs
                if enable_reinforced_learning and manual_rl_epochs is not None:
                    global_max_epochs = global_total_models * (epochs + manual_rl_epochs)
                else:
                    global_max_epochs = global_total_epochs
                self.console.print(f"[cyan]Updated: {num_valid_keys} keys × {epochs} epochs = {global_total_models} models[/cyan]")

            self.console.print(f"\n[green][OK] Pre-validation complete. Ready to train {len(key_files)} models.[/green]\n")

            # ============================================================
            # CHECK FOR PARALLEL TRAINING OPTION (MULTI-CLASS - Smart Scheduler)
            # ============================================================
            num_models_to_train = len(key_files)
            use_parallel_training = False

            try:
                from llm_tool.trainers.hybrid_parallel_trainer import HardwareAnalyzer, HybridParallelTrainer
                from llm_tool.trainers.task_scheduler import SchedulerConfig
                from rich.prompt import Confirm

                # Use smart scheduler to determine if parallel training is beneficial
                scheduler_config = SchedulerConfig()
                should_offer, scheduler_message = HybridParallelTrainer.should_offer_parallel(
                    num_models_to_train, scheduler_config
                )

                hw_analyzer = HardwareAnalyzer()
                can_parallel, parallel_reason = hw_analyzer.can_use_parallel_training(num_models_to_train)

                if can_parallel and should_offer:
                    # Determine if strongly recommended
                    strongly_recommended = num_models_to_train >= scheduler_config.recommended_tasks_for_parallel
                    min_threshold = scheduler_config.min_tasks_for_parallel
                    recommended_threshold = scheduler_config.recommended_tasks_for_parallel

                    self.console.print(f"\n[bold cyan]Parallel Training Available[/bold cyan]")

                    # Clear recommendation panel
                    from rich.panel import Panel
                    if strongly_recommended:
                        rec_text = (
                            f"[bold green][OK] STRONGLY RECOMMENDED[/bold green]\n\n"
                            f"You have [bold]{num_models_to_train} models[/bold] to train, which is ≥{recommended_threshold} models.\n"
                            f"Parallel training will provide [green]significant speedup[/green] by utilizing GPU + CPU workers simultaneously."
                        )
                        rec_style = "green"
                        default_choice = True
                    else:
                        rec_text = (
                            f"[bold yellow][!] OPTIONAL - MARGINAL BENEFIT[/bold yellow]\n\n"
                            f"You have [bold]{num_models_to_train} models[/bold] to train ({min_threshold}-{recommended_threshold-1} range).\n"
                            f"Parallel training offers [yellow]moderate speedup[/yellow], but overhead may reduce benefits.\n\n"
                            f"[dim]• Sequential (GPU-only) is often faster for <{recommended_threshold} models\n"
                            f"• Parallel becomes efficient with ≥{recommended_threshold} models[/dim]"
                        )
                        rec_style = "yellow"
                        default_choice = False

                    self.console.print(Panel(rec_text, title="Recommendation", border_style=rec_style))

                    # Show resource plan
                    plan = hw_analyzer.create_resource_plan(model_name, num_models_to_train)

                    from rich.table import Table
                    from rich import box

                    plan_table = Table(
                        title="Resource Allocation Plan",
                        show_header=True,
                        header_style="bold cyan",
                        box=box.ROUNDED,
                        expand=True,
                    )
                    plan_table.add_column("Device", style="cyan")
                    plan_table.add_column("Type")
                    plan_table.add_column("Batch Size", justify="center")
                    plan_table.add_column("Workers", justify="center")

                    if plan.gpu_allocation:
                        g = plan.gpu_allocation
                        plan_table.add_row(f"GPU {g.device_id.upper()}", "GPU (Priority)", str(g.batch_size), str(g.num_workers))

                    for c in plan.cpu_allocations:
                        plan_table.add_row(f"CPU {c.device_id}", "CPU", str(c.batch_size), str(c.num_workers))

                    self.console.print(plan_table)
                    self.console.print(f"\n[dim]Total: {plan.total_parallel_workers} workers available[/dim]")

                    # Ask user with clear default based on recommendation
                    if strongly_recommended:
                        prompt_text = "\n[bold]Use parallel training? [green](recommended: yes)[/green][/bold]"
                    else:
                        prompt_text = "\n[bold]Use parallel training? [yellow](recommended: no)[/yellow][/bold]"

                    use_parallel_training = Confirm.ask(prompt_text, default=default_choice)

                    if use_parallel_training:
                        self.console.print("\n[green]✓ Parallel training enabled with smart task scheduling[/green]\n")
                elif not should_offer:
                    self.console.print(f"[dim]{scheduler_message}[/dim]")
            except ImportError:
                self.logger.debug("Parallel training not available (missing dependencies)")
            except Exception as e:
                self.logger.warning(f"Could not check parallel training: {e}")

            # ============================================================
            # RUN PARALLEL OR SEQUENTIAL TRAINING (MULTI-CLASS)
            # ============================================================
            if use_parallel_training:
                # Convert key_files to the format expected by _run_parallel_training
                category_files = {k: Path(v) if not isinstance(v, Path) else v for k, v in key_files.items()}

                return self._run_parallel_training(
                    category_files=category_files,
                    model_name=model_name,
                    epochs=epochs,
                    output_dir=output_dir,
                    session_id=session_id,
                    bundle=bundle,
                    extra_config=extra_config,
                    enable_reinforced_learning=enable_reinforced_learning,
                    manual_rl_epochs=manual_rl_epochs,
                    languages=languages,
                    models_by_language=models_by_language,
                    needs_language_training=needs_language_training,
                    runtime_params=runtime_params,
                    training_approach='multi-class',
                    quick_params=quick_params if 'quick_params' in dir() else None,
                )

            # Sequential training (original code path)
            results_per_key = {}

            # Create progress callback to track completed epochs across all models
            def progress_callback(**metrics):
                """Callback to sync global completed epochs counter from authoritative source."""
                nonlocal global_completed_epochs
                # Update from callback metrics if available (bert_base.py is the authoritative source)
                if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                    global_completed_epochs = metrics['global_completed_epochs']

            for idx, (key_name, key_file_path) in enumerate(key_files.items(), 1):
                self.console.print(f"\n[bold]Training model for key '{key_name}'[/bold] ({key_file_path.name})")

                # Create config for this key
                key_config = {
                    'input_file': str(key_file_path),
                    'model_name': model_name,
                    'num_epochs': epochs,
                    'output_dir': str(output_dir),  # bert_base.py will construct correct path
                    'text_column': bundle.text_column,
                    'label_column': bundle.label_column,
                    'training_strategy': 'single-label',  # Each key file is single-label
                    'training_approach': 'multi-class',   # CRITICAL: For correct chart labeling
                    'category_name': key_name,
                    'reinforced_learning': enable_reinforced_learning,
                    'session_id': session_id,
                    'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                    # Global progress tracking
                    'global_total_models': global_total_models,
                    'global_current_model': idx,
                    'global_total_epochs': global_total_epochs,
                    'global_max_epochs': global_max_epochs,
                    'global_completed_epochs': global_completed_epochs,
                    'global_start_time': global_start_time,
                    'train_by_language': needs_language_training,
                    'confirmed_languages': list(languages) if languages else None,
                    'progress_callback': progress_callback,  # CRITICAL: For epoch counting
                    'auto_remove_insufficient_labels': True,  # Pre-validated upstream
                }

                # Merge reinforced learning hyperparameters and other extras
                for extra_key, extra_value in extra_config.items():
                    # Avoid overwriting core routing fields already set in key_config
                    if extra_key not in key_config:
                        key_config[extra_key] = extra_value

                # Add models_by_language if user selected per-language models
                if models_by_language:
                    key_config["models_by_language"] = models_by_language

                if enable_reinforced_learning and manual_rl_epochs is not None:
                    key_config["reinforced_epochs"] = int(manual_rl_epochs)

                try:
                    key_result = trainer.train(key_config)
                    # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
                    returned_epochs = key_result.get('global_completed_epochs') if isinstance(key_result, dict) else None
                    if returned_epochs is not None:
                        global_completed_epochs = max(global_completed_epochs, returned_epochs)
                    self.logger.debug(f"[EPOCH_TRACK] After multi-class key {key_name}: global_completed_epochs={global_completed_epochs}")
                    results_per_key[key_name] = key_result
                    self.console.print(f"[green]Completed {key_name}: Accuracy={key_result.get('accuracy', 0):.4f}, F1={key_result.get('best_f1_macro', 0):.4f}[/green]")
                except Exception as exc:
                    self.console.print(f"[red]✗ Failed to train {key_name}: {exc}[/red]")
                    self.logger.exception(f"Training failed for {key_name}", exc_info=exc)
                    results_per_key[key_name] = {'error': str(exc)}

            # Aggregate results
            successful_results = [r for r in results_per_key.values() if 'error' not in r]
            if successful_results:
                avg_accuracy = sum(r.get('accuracy', 0) for r in successful_results) / len(successful_results)
                avg_f1 = sum(r.get('best_f1_macro', 0) for r in successful_results) / len(successful_results)

                result = {
                    'best_model': model_name,
                    'accuracy': avg_accuracy,
                    'best_f1_macro': avg_f1,
                    'model_path': str(output_dir),
                    'training_time': sum(r.get('training_time', 0) for r in successful_results),
                    'models_trained': len(successful_results),
                    'total_keys': len(key_files),
                    'per_key_results': results_per_key,
                    'training_approach': 'multi-class'
                }
            else:
                self.console.print("[red]All key trainings failed[/red]")
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': 'All key trainings failed'
                }
        else:
            self.console.print("[yellow][!] No key files found, falling back to standard multi-label training[/yellow]")

            # ============================================================
            # CRITICAL: Validate and filter insufficient labels BEFORE training
            # ============================================================
            input_file_to_use = str(bundle.primary_file)
            if bundle.primary_file:
                try:
                    filtered_file, was_filtered = self._validate_and_filter_insufficient_labels(
                        input_file=str(bundle.primary_file),
                        strategy=bundle.strategy,
                        min_samples=2,
                        auto_remove=False,  # Ask user for confirmation
                        train_by_language=needs_language_training  # CRITICAL: Language-aware validation
                    )
                    if was_filtered:
                        input_file_to_use = filtered_file
                        self.console.print(f"[green]Using filtered training dataset[/green]\n")
                except ValueError as e:
                    # User cancelled or validation failed
                    self.console.print(f"[red]{e}[/red]")
                    return {
                        'runtime_params': runtime_params,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None,
                        'error': str(e)
                    }
                except Exception as e:
                    self.logger.warning(f"Label validation failed: {e}")
                    # Continue with original file if validation fails
                    pass

            # Fall through to standard training
            # Initialize global progress tracking
            import time
            global_start_time = time.time()
            global_total_models = 1
            global_total_epochs = epochs

            # Calculate maximum possible epochs (if model triggers reinforced learning)
            if enable_reinforced_learning and manual_rl_epochs is not None:
                global_max_epochs = epochs + manual_rl_epochs
            else:
                global_max_epochs = global_total_epochs

            # DEBUGGING: Log the epoch calculation
            num_languages = int(len(languages)) if languages else 1
            self.logger.info("="*80)
            self.logger.info("GLOBAL EPOCHS CALCULATION DEBUG")
            self.logger.info(f"  Training mode: multi-label (single model)")
            self.logger.info(f"  Number of models: {global_total_models}")
            self.logger.info(f"  Number of languages: {num_languages}")
            self.logger.info(f"  Languages: {sorted(languages) if languages else 'N/A'}")
            self.logger.info(f"  Base epochs: {epochs}")
            self.logger.info(f"  RL epochs: {manual_rl_epochs if manual_rl_epochs else 'None'}")
            self.logger.info(f"  CALCULATED global_total_models: {global_total_models}")
            self.logger.info(f"  CALCULATED global_total_epochs: {global_total_epochs}")
            self.logger.info(f"  CALCULATED global_max_epochs: {global_max_epochs}")
            self.logger.info("="*80)

            global_completed_epochs = 0

            # Create progress callback to track completed epochs
            def progress_callback(**metrics):
                """Callback to sync global completed epochs counter from authoritative source."""
                nonlocal global_completed_epochs
                # Update from callback metrics if available (bert_base.py is the authoritative source)
                if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                    global_completed_epochs = metrics['global_completed_epochs']

            result = trainer.train({
                'input_file': input_file_to_use,
                'model_name': model_name,
                'num_epochs': epochs,
                'output_dir': str(output_dir),
                'text_column': bundle.text_column,
                'label_column': bundle.label_column,
                'multiclass_groups': multiclass_groups,
                'reinforced_learning': enable_reinforced_learning,
                'session_id': session_id,
                'split_config': bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None,
                # Global progress tracking
                'global_total_models': global_total_models,
                'global_current_model': 1,
                'global_total_epochs': global_total_epochs,
                'global_max_epochs': global_max_epochs,
                'global_completed_epochs': global_completed_epochs,
                'global_start_time': global_start_time,
                'progress_callback': progress_callback,  # CRITICAL: For epoch counting
                **extra_config
            })

            # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
            returned_epochs = result.get('global_completed_epochs') if isinstance(result, dict) else None
            if returned_epochs is not None:
                global_completed_epochs = max(global_completed_epochs, returned_epochs)
            self.logger.debug(f"[EPOCH_TRACK] After single model training: global_completed_epochs={global_completed_epochs}")

    else:
        # Standard training (multi-class or multi-label)

        # ============================================================
        # CRITICAL: Validate and filter insufficient labels BEFORE training
        # ============================================================
        if bundle.primary_file:
            try:
                filtered_file, was_filtered = self._validate_and_filter_insufficient_labels(
                    input_file=str(bundle.primary_file),
                    strategy=bundle.strategy,
                    min_samples=2,
                    auto_remove=False,  # Ask user for confirmation
                    train_by_language=needs_language_training  # CRITICAL: Language-aware validation
                )
                if was_filtered:
                    # Update bundle to use filtered file
                    bundle.primary_file = Path(filtered_file)
                    self.console.print(f"[green]Using filtered training dataset[/green]\n")
            except ValueError as e:
                # User cancelled or validation failed
                self.console.print(f"[red]{e}[/red]")
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': str(e)
                }
            except Exception as e:
                self.logger.warning(f"Label validation failed: {e}")
                # Continue with original file if validation fails
                pass

        # Initialize global progress tracking
        import time
        global_start_time = time.time()
        global_total_models = 1
        global_total_epochs = epochs

        # Calculate maximum possible epochs (if model triggers reinforced learning)
        if enable_reinforced_learning and manual_rl_epochs is not None:
            global_max_epochs = epochs + manual_rl_epochs
        else:
            global_max_epochs = global_total_epochs

        # DEBUGGING: Log the epoch calculation
        num_languages = int(len(languages)) if languages else 1
        self.logger.info("="*80)
        self.logger.info("GLOBAL EPOCHS CALCULATION DEBUG")
        self.logger.info(f"  Training mode: standard (multi-label or multi-class single model)")
        self.logger.info(f"  Number of models: {global_total_models}")
        self.logger.info(f"  Number of languages: {num_languages}")
        self.logger.info(f"  Languages: {sorted(languages) if languages else 'N/A'}")
        self.logger.info(f"  Base epochs: {epochs}")
        self.logger.info(f"  RL epochs: {manual_rl_epochs if manual_rl_epochs else 'None'}")
        self.logger.info(f"  CALCULATED global_total_models: {global_total_models}")
        self.logger.info(f"  CALCULATED global_total_epochs: {global_total_epochs}")
        self.logger.info(f"  CALCULATED global_max_epochs: {global_max_epochs}")
        self.logger.info("="*80)

        global_completed_epochs = 0

        # Create progress callback to track completed epochs
        def progress_callback(**metrics):
            """Callback to sync global completed epochs counter from authoritative source."""
            nonlocal global_completed_epochs
            # Update from callback metrics if available (bert_base.py is the authoritative source)
            if 'global_completed_epochs' in metrics and metrics['global_completed_epochs'] is not None:
                global_completed_epochs = metrics['global_completed_epochs']

        config = bundle.to_trainer_config(output_dir, extra_config)
        config['session_id'] = session_id
        config['split_config'] = bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None
        # Add global progress tracking
        config['global_total_models'] = global_total_models
        config['global_current_model'] = 1
        config['global_total_epochs'] = global_total_epochs
        config['global_max_epochs'] = global_max_epochs
        config['global_completed_epochs'] = global_completed_epochs
        config['global_start_time'] = global_start_time
        config['progress_callback'] = progress_callback  # CRITICAL: For epoch counting

        try:
            result = trainer.train(config)
            # Update global completed epochs (belt-and-suspenders: use MAX of callback and return value)
            returned_epochs = result.get('global_completed_epochs') if isinstance(result, dict) else None
            if returned_epochs is not None:
                global_completed_epochs = max(global_completed_epochs, returned_epochs)
            self.logger.debug(f"[EPOCH_TRACK] After quick training: global_completed_epochs={global_completed_epochs}")
        except Exception as exc:  # pylint: disable=broad-except
            self.console.print(f"[red]Training failed:[/red] {exc}")
            self.logger.exception("Quick training failed", exc_info=exc)
            return {
                'runtime_params': runtime_params,
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'error': str(exc)
            }

    self._training_studio_show_training_result(result, bundle, title="Quick training results")

    # Return complete training info for metadata save
    trained_models_output = result.get('trained_models')
    raw_models_trained = result.get('models_trained')

    if isinstance(trained_models_output, dict):
        trained_models_output = dict(trained_models_output)
    else:
        trained_models_output = {}

    if isinstance(raw_models_trained, list):
        models_trained_list = raw_models_trained
    elif isinstance(raw_models_trained, (set, tuple)):
        models_trained_list = list(raw_models_trained)
    elif isinstance(raw_models_trained, dict):
        models_trained_list = list(raw_models_trained.keys())
    elif trained_models_output:
        models_trained_list = list(trained_models_output.keys())
    elif raw_models_trained:
        models_trained_list = [str(raw_models_trained)]
    elif model_name:
        models_trained_list = [model_name]
    else:
        models_trained_list = []

    return {
        'runtime_params': runtime_params,
        'models_trained': models_trained_list,
        'models_trained_count': len(models_trained_list),
        'trained_models': trained_models_output,
        'trained_model_paths': trained_models_output,
        'best_model': result.get('best_model'),
        'best_f1': result.get('best_f1') or result.get('best_f1_macro') or result.get('f1_macro'),
        'best_f1_macro': result.get('best_f1_macro'),
        'accuracy': result.get('accuracy'),
        'model_path': result.get('model_path'),
        'training_time': result.get('training_time'),
        'metrics': result.get('metrics'),
        'per_key_results': result.get('per_key_results'),
        'onevsall_results': result.get('onevsall_results'),
        'onevsall_value_map': result.get('onevsall_value_map'),
        'onevsall_label_info': result.get('onevsall_label_info'),
        'training_approach': result.get('training_approach'),
        'stage_summary': result.get('stage_summary'),
        'stage_models': result.get('stage_models'),
        'total_expected_models': result.get('total_expected_models'),
        'error': result.get('error')
    }

def _training_studio_resolve_multilabel_dataset(self, bundle: TrainingDataBundle) -> Optional[Tuple[Path, Optional[List[str]]]]:
    """Return the consolidated multi-label dataset path if available."""
    multilabel_path = bundle.training_files.get("multilabel")
    if multilabel_path:
        path_obj = Path(multilabel_path)
        if path_obj.exists():
            label_fields = bundle.metadata.get("labels_detected")
            return path_obj, label_fields if isinstance(label_fields, list) else None

    if bundle.strategy == "multi-label" and bundle.primary_file:
        path_obj = Path(bundle.primary_file)
        if path_obj.exists() and path_obj.suffix.lower() in {".json", ".jsonl"}:
            label_fields = bundle.metadata.get("labels_detected")
            return path_obj, label_fields if isinstance(label_fields, list) else None

    return None

def _training_studio_show_distributed_results(
    self,
    trainer: MultiLabelTrainer,
    models: Dict[str, MultiLabelModelInfo],
    output_dir: Path,
) -> None:
    """Render a summary table of distributed training results."""
    if not models:
        message = "No models were produced during distributed training."
        if HAS_RICH and self.console:
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            print(message)
        return

    if HAS_RICH and self.console:
        table = Table(title="Distributed training results", border_style="green", expand=True)
        table.add_column("Model", style="cyan", ratio=1, overflow="fold")
        table.add_column("Label", style="white", no_wrap=True)
        table.add_column("Language", style="white", no_wrap=True)
        table.add_column("Macro F1", justify="right", no_wrap=True)

        for model_name, info in sorted(models.items()):
            metrics = info.performance_metrics or {}
            macro_f1 = metrics.get("macro_f1", 0.0)
            table.add_row(
                model_name,
                info.label_name,
                info.language or "—",
                f"{macro_f1:.3f}"
            )

        self.console.print(table)
        self.console.print(f"[dim]Models saved to[/dim] {output_dir}")
    else:
        print("\nDistributed training results:")
        for model_name, info in sorted(models.items()):
            metrics = info.performance_metrics or {}
            macro_f1 = metrics.get('macro_f1', 0.0)
            print(f"  - {model_name}: label={info.label_name}, lang={info.language or '-'}, macro_f1={macro_f1:.3f}")
        print(f"Models saved to {output_dir}")


def _training_studio_show_training_result(self, result: Dict[str, Any], bundle: TrainingDataBundle, title: str) -> None:
    table = Table(title=title, border_style="green", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white", ratio=1, overflow="fold")

    table.add_row("Model", str(result.get("best_model", "n/a")))
    table.add_row("Accuracy", f"{result.get('accuracy', 0.0):.4f}")
    table.add_row("F1 macro", f"{result.get('best_f1_macro', 0.0):.4f}")
    table.add_row("Model path", result.get("model_path", "—"))

    self.console.print(table)

    # Display language filtering report if present
    if 'language_filtering_report' in result:
        filtering_report = result['language_filtering_report']
        if filtering_report.get('filtered', False):
            filter_table = Table(title="Language Filtering Applied", border_style="yellow", expand=True)
            filter_table.add_column("Info", style="cyan", no_wrap=True)
            filter_table.add_column("Value", style="white", ratio=1, overflow="fold")

            filter_table.add_row("Languages kept", ', '.join(filtering_report.get('languages_kept', [])))
            filter_table.add_row("Languages dropped", ', '.join(filtering_report.get('languages_dropped', [])))
            filter_table.add_row("Samples before filtering", str(filtering_report.get('total_samples_before', 0)))
            filter_table.add_row("Samples after filtering", str(filtering_report.get('total_samples_after', 0)))

            # Show drop reasons for each language
            drop_reasons = filtering_report.get('drop_reasons', {})
            if drop_reasons:
                for lang, reason in drop_reasons.items():
                    details = reason.get('details', [])
                    detail_str = ', '.join([f"{cls}({cnt})" for cls, cnt in details])
                    filter_table.add_row(f"  {lang} dropped", f"Insufficient: {detail_str}")

            self.console.print(filter_table)

    if bundle.strategy == "multi-label":
        metrics = result.get("metrics", {})
        per_label = metrics.get("per_label_results")
        if per_label:
            detail_table = Table(title="Per-label performance", border_style="blue", expand=True)
            detail_table.add_column("Label", ratio=1, overflow="fold")
            detail_table.add_column("Accuracy", no_wrap=True)
            detail_table.add_column("F1 macro", no_wrap=True)

            for label, stats in per_label.items():
                if isinstance(stats, dict) and "error" not in stats:
                    detail_table.add_row(
                        label,
                        f"{stats.get('accuracy', 0.0):.4f}",
                        f"{stats.get('f1_macro', 0.0):.4f}",
                    )
                elif isinstance(stats, dict):
                    detail_table.add_row(label, stats.get("error", "error"), "—")

            self.console.print(detail_table)

def _training_studio_show_benchmark_results(self, report: Dict[str, Any]) -> None:
    results = report.get("results", [])
    if not results:
        self.console.print("[yellow]No benchmark results available.[/yellow]")
        return

    # Check if any language filtering was applied
    language_filtering_summary = report.get("language_filtering_summary", {})
    if language_filtering_summary:
        self.console.print("\n[bold yellow][!] Language Filtering Applied During Benchmark[/bold yellow]")
        for category, filter_info in language_filtering_summary.items():
            if filter_info.get('filtered', False):
                self.console.print(f"  • {category}: Dropped languages {', '.join(filter_info.get('languages_dropped', []))}")
                self.console.print(f"    Kept: {', '.join(filter_info.get('languages_kept', []))}")

    table = Table(title="Benchmark results", border_style="green", expand=True)
    table.add_column("#", style="cyan", width=5)
    table.add_column("Model", style="white", ratio=1, overflow="fold")
    table.add_column("Accuracy", justify="right", no_wrap=True)
    table.add_column("F1 macro", justify="right", no_wrap=True)

    for idx, entry in enumerate(results, start=1):
        table.add_row(
            str(idx),
            entry.get("model", "?"),
            f"{entry.get('accuracy', 0.0):.4f}",
            f"{entry.get('f1_macro', 0.0):.4f}",
        )

    self.console.print(table)

    best_model = report.get("best_model")
    if best_model:
        best_f1 = report.get("best_f1_macro", 0.0)
        self.console.print(f"[green]Best model:[/green] {best_model} (F1 {best_f1:.4f})")

def _training_studio_resolve_benchmark_dataset(self, bundle: TrainingDataBundle) -> Tuple[Path, str, str]:
    # Support both single-label and multi-label datasets for benchmarking
    if bundle.primary_file:
        return bundle.primary_file, bundle.text_column, bundle.label_column

    # For multi-label distributed training, we have individual label files
    candidates = [(label, path) for label, path in bundle.training_files.items() if label != "multilabel"]

    if not candidates:
        raise ValueError("No dataset available for benchmarking.")

    if len(candidates) == 1:
        label, path = candidates[0]
        self.console.print(f"Using dataset for label [cyan]{label}[/cyan].")
        return path, "text", "label"

    self.console.print("\nSelect the label you want to benchmark:")
    for idx, (label, _) in enumerate(candidates, start=1):
        self.console.print(f"  {idx}. {label}")

    choice = self._int_prompt_with_validation("Label", default=1, min_value=1, max_value=len(candidates))
    label, path = candidates[choice - 1]
    self.console.print(f"Benchmarking label [cyan]{label}[/cyan].")
    return path, "text", "label"

def _training_studio_make_output_dir(self, prefix: str) -> Path:
    """
    Create output directory for models.

    CRITICAL: This function should NOT be used in Training Arena mode.
    Instead, models are saved directly to models/{session_id}/...
    This function is kept for backward compatibility with legacy modes.

    Args:
        prefix: Prefix for directory name (e.g., 'training_studio_quick')

    Returns:
        Path to created directory
    """
    from datetime import datetime
    directory = self.settings.paths.models_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def _flatten_trainer_models(self) -> List[str]:
    if not self.available_trainer_models:
        return []

    names: List[str] = []
    for models in self.available_trainer_models.values():
        names.extend(model["name"] for model in models)

    seen = set()
    unique: List[str] = []
    for name in names:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique

def _get_majority_language(self, languages: set, language_distribution: dict = None) -> str:
    """
    Determine the majority/dominant language from a set of languages.

    Args:
        languages: Set of language codes
        language_distribution: Optional dict mapping language codes to counts

    Returns:
        Dominant language code (lowercase) or None
    """
    if not languages:
        return None

    # If we have distribution data, use it to find the majority
    if language_distribution:
        total = sum(language_distribution.values())
        if total > 0:
            # Find language with highest percentage
            majority_lang = max(language_distribution.items(), key=lambda x: x[1])
            percentage = (majority_lang[1] / total) * 100

            # If a language represents >50%, it's the majority
            if percentage > 50:
                return majority_lang[0].lower()

    # Fallback: if only one language, return it
    if len(languages) == 1:
        return list(languages)[0].lower()

    # If multiple languages without clear majority, check for common cases
    lang_list = [l.lower() for l in languages]
    if 'fr' in lang_list:
        return 'fr'  # Often FR is dominant after correction
    elif 'en' in lang_list:
        return 'en'

    return None

def _is_model_multilingual(self, model_name: str) -> bool:
    """
    Determine if a model is multilingual or language-specific.

    Args:
        model_name: HuggingFace model ID

    Returns:
        True if multilingual, False if language-specific
    """
    # Model-to-language mapping (same as in _get_intelligent_benchmark_models)
    MULTILINGUAL_KEYWORDS = ['xlm', 'multilingual', 'mdeberta', 'long-t5']
    MONOLINGUAL_PATTERNS = {
        'camembert': 'fr',
        'flaubert': 'fr',
        'bert-base-german-cased': 'de',
        'distilbert-base-german-cased': 'de',
        'roberta': 'en',  # RoBERTa is English-only unless specified
        'bert-base-uncased': 'en',
        'bert-base-cased': 'en',
        'distilbert-base-uncased': 'en',
    }

    model_lower = model_name.lower()

    # Check for multilingual keywords
    if any(keyword in model_lower for keyword in MULTILINGUAL_KEYWORDS):
        return True

    # Check for monolingual patterns
    if any(pattern in model_lower for pattern in MONOLINGUAL_PATTERNS.keys()):
        return False

    # Default: assume multilingual for safety (won't create extra models)
    return True

def _get_intelligent_benchmark_models(self, languages: set, text_length_avg: float, model_strategy: str, recommended_model: str = None, language_distribution: dict = None, user_prefers_long_models: bool = False) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """
    HIGHLY INTELLIGENT model selection using ALL available models in the package.

    Selection criteria (scored):
    1. Language match (primary, 100 points)
    2. Text length compatibility (50 points) - BOOSTED if user_prefers_long_models=True
    3. Model size/efficiency (30 points)
    4. Model popularity/reliability (20 points)
    5. Multilingual capability (bonus for mixed languages)

    Returns:
        Tuple of (models_list, model_to_language_map)
        model_to_language_map: {model_name: language_code or None for multilingual}
    """
    lang_list = list(languages) if languages else []

    # COMPREHENSIVE language-to-model mapping using ALL models from sota_models
    MODEL_LANGUAGE_MAP = {
        # ============ MULTILINGUAL MODELS ============
        'xlm-roberta-base': None,
        'xlm-roberta-large': None,
        'bert-base-multilingual-cased': None,
        'bert-base-multilingual-uncased': None,
        'distilbert-base-multilingual-cased': None,
        'microsoft/mdeberta-v3-base': None,

        # ============ MULTILINGUAL LONG-DOCUMENT MODELS ============
        'markussagen/xlm-roberta-longformer-base-4096': None,  # Multilingual Longformer, 4096 tokens, 100+ languages
        'google/long-t5-local-base': None,  # Multilingual T5 with local attention, 4096+ tokens
        'google/long-t5-tglobal-base': None,  # Multilingual T5 with transient global attention, 4096+ tokens

        # ============ ENGLISH MODELS ============
        'bert-base-uncased': 'en',
        'bert-base-cased': 'en',
        'bert-large-uncased': 'en',
        'bert-large-cased': 'en',
        'roberta-base': 'en',
        'roberta-large': 'en',
        'distilbert-base-uncased': 'en',
        'distilbert-base-cased': 'en',
        'distilroberta-base': 'en',
        'albert-base-v2': 'en',
        'albert-large-v2': 'en',
        'albert-xlarge-v2': 'en',
        'google/electra-base-discriminator': 'en',
        'google/electra-large-discriminator': 'en',
        'microsoft/deberta-base': 'en',
        'microsoft/deberta-large': 'en',
        'microsoft/deberta-v3-base': 'en',
        'microsoft/deberta-v3-large': 'en',
        'microsoft/deberta-v3-small': 'en',
        'allenai/longformer-base-4096': 'en',
        'google/bigbird-roberta-base': 'en',
        'google/bigbird-roberta-large': 'en',
        'squeezebert/squeezebert-uncased': 'en',
        'sentence-transformers/all-MiniLM-L6-v2': 'en',

        # ============ FRENCH MODELS ============
        'camembert-base': 'fr',
        'camembert/camembert-base': 'fr',
        'camembert/camembert-large': 'fr',
        'flaubert/flaubert_base_cased': 'fr',
        'flaubert/flaubert_base_uncased': 'fr',
        'flaubert/flaubert_large_cased': 'fr',
        'cmarkea/distilcamembert-base': 'fr',
        'almanach/camembert-base': 'fr',
        'almanach/camembertav2-base': 'fr',
        'dbmdz/bert-base-french-europeana-cased': 'fr',
        'dangvantuan/sentence-camembert-base': 'fr',
        'qwant/fralbert-base': 'fr',

        # ============ GERMAN MODELS ============
        'bert-base-german-cased': 'de',
        'bert-base-german-dbmdz-cased': 'de',
        'bert-base-german-dbmdz-uncased': 'de',
        'deepset/gbert-base': 'de',
        'deepset/gbert-large': 'de',
        'distilbert-base-german-cased': 'de',
        'uklfr/gottbert-base': 'de',
        'dbmdz/bert-base-german-europeana-cased': 'de',

        # ============ SPANISH MODELS ============
        'dccuchile/bert-base-spanish-wwm-cased': 'es',
        'dccuchile/bert-base-spanish-wwm-uncased': 'es',
        'PlanTL-GOB-ES/roberta-base-bne': 'es',
        'mrm8488/electricidad-base-discriminator': 'es',
        'bertin-project/bertin-roberta-base-spanish': 'es',

        # ============ ITALIAN MODELS ============
        'dbmdz/bert-base-italian-cased': 'it',
        'dbmdz/bert-base-italian-uncased': 'it',
        'dbmdz/bert-base-italian-xxl-cased': 'it',
        'dbmdz/bert-base-italian-xxl-uncased': 'it',
        'Musixmatch/umberto-commoncrawl-cased-v1': 'it',

        # ============ PORTUGUESE MODELS ============
        'neuralmind/bert-base-portuguese-cased': 'pt',
        'neuralmind/bert-large-portuguese-cased': 'pt',
        'adalbertojunior/distilbert-portuguese-cased': 'pt',
        'pierreguillou/bert-base-cased-pt-lenerbr': 'pt',

        # ============ DUTCH MODELS ============
        'GroNLP/bert-base-dutch-cased': 'nl',
        'wietsedv/bert-base-dutch-cased': 'nl',
        'pdelobelle/robbert-v2-dutch-base': 'nl',
        'DTAI-KULeuven/robbert-2023-dutch-large': 'nl',

        # ============ POLISH MODELS ============
        'dkleczek/bert-base-polish-uncased-v1': 'pl',
        'dkleczek/bert-base-polish-cased-v1': 'pl',
        'allegro/herbert-base-cased': 'pl',
        'allegro/herbert-large-cased': 'pl',

        # ============ ARABIC MODELS ============
        'aubmindlab/bert-base-arabertv2': 'ar',
        'aubmindlab/bert-large-arabertv2': 'ar',
        'asafaya/bert-base-arabic': 'ar',
        'CAMeL-Lab/bert-base-arabic-camelbert-msa': 'ar',
        'UBC-NLP/MARBERT': 'ar',

        # ============ CHINESE MODELS ============
        'bert-base-chinese': 'zh',
        'hfl/chinese-bert-wwm': 'zh',
        'hfl/chinese-bert-wwm-ext': 'zh',
        'hfl/chinese-roberta-wwm-ext': 'zh',
        'hfl/chinese-roberta-wwm-ext-large': 'zh',
        'hfl/chinese-electra-base-discriminator': 'zh',

        # ============ RUSSIAN MODELS ============
        'DeepPavlov/rubert-base-cased': 'ru',
        'DeepPavlov/rubert-base-cased-conversational': 'ru',
        'ai-forever/ruBert-base': 'ru',
        'ai-forever/ruBert-large': 'ru',
        'cointegrated/rubert-tiny': 'ru',

        # ============ JAPANESE MODELS ============
        'cl-tohoku/bert-base-japanese': 'ja',
        'cl-tohoku/bert-base-japanese-whole-word-masking': 'ja',
        'cl-tohoku/bert-large-japanese': 'ja',
        'nlp-waseda/roberta-base-japanese': 'ja',
        'nlp-waseda/roberta-large-japanese': 'ja',

        # ============ KOREAN MODELS ============
        'klue/bert-base': 'ko',
        'kykim/bert-kor-base': 'ko',
        'beomi/kcbert-base': 'ko',
        'beomi/kcbert-large': 'ko',

        # ============ TURKISH MODELS ============
        'dbmdz/bert-base-turkish-cased': 'tr',
        'dbmdz/bert-base-turkish-uncased': 'tr',
        'dbmdz/electra-base-turkish-cased-discriminator': 'tr',

        # ============ SWEDISH MODELS ============
        'KB/bert-base-swedish-cased': 'sv',
        'af-ai-center/bert-base-swedish-uncased': 'sv',

        # ============ DANISH MODELS ============
        'Maltehb/danish-bert-botxo': 'da',
        'sarnikowski/convbert-small-da-cased': 'da',

        # ============ NORWEGIAN MODELS ============
        'ltg/norbert': 'no',
        'NbAiLab/nb-bert-base': 'no',

        # ============ FINNISH MODELS ============
        'TurkuNLP/bert-base-finnish-cased-v1': 'fi',
        'TurkuNLP/bert-base-finnish-uncased-v1': 'fi',

        # ============ HINDI MODELS ============
        'ai4bharat/indic-bert': 'hi',

        # ============ VIETNAMESE MODELS ============
        'vinai/phobert-base': 'vi',
        'vinai/phobert-large': 'vi',

        # ============ THAI MODELS ============
        'airesearch/wangchanberta-base-att-spm-uncased': 'th',

        # ============ INDONESIAN MODELS ============
        'indobenchmark/indobert-base-p1': 'id',
        'indobenchmark/indobert-large-p1': 'id',

        # ============ CZECH MODELS ============
        'Seznam/retromae-small-cs': 'cs',
        'ufal/robeczech-base': 'cs',

        # ============ GREEK MODELS ============
        'nlpaueb/bert-base-greek-uncased-v1': 'el',

        # ============ HEBREW MODELS ============
        'onlplab/alephbert-base': 'he',

        # ============ ROMANIAN MODELS ============
        'dumitrescustefan/bert-base-romanian-cased-v1': 'ro',

        # ============ BULGARIAN MODELS ============
        'iarfmoose/roberta-base-bulgarian': 'bg',

        # ============ CROATIAN MODELS ============
        'classla/bcms-bertic': 'hr',

        # ============ SERBIAN MODELS ============
        'classla/bcms-bertic': 'sr',

        # ============ UKRAINIAN MODELS ============
        'youscan/ukr-roberta-base': 'uk',
    }

    # ============================================================
    # STEP 1: Analyze language distribution
    # ============================================================
    total_samples = sum(language_distribution.values()) if language_distribution else 0
    lang_percentages = {}
    if total_samples > 0:
        lang_percentages = {lang.lower(): (count / total_samples * 100)
                            for lang, count in language_distribution.items()}

    # Find dominant language (>70%)
    dominant_lang = None
    for lang, pct in lang_percentages.items():
        if pct > 70:
            dominant_lang = lang
            break

    # Check if single language
    if len(lang_list) == 1:
        dominant_lang = lang_list[0].lower()

    # Check if balanced multilingual
    is_balanced_multilingual = len(lang_list) > 1 and not dominant_lang

    # ============================================================
    # STEP 2: Score ALL available models
    # ============================================================
    model_scores = {}

    for model_name, model_lang in MODEL_LANGUAGE_MAP.items():
        score = 0.0

        # CRITERION 1: Language Match (100 points max)
        if model_lang is None:  # Multilingual model
            if is_balanced_multilingual:
                score += 90  # Excellent for balanced multilingual
            elif len(lang_list) > 1:
                score += 70  # Good for any multilingual
            else:
                score += 40  # Okay for single language
        else:  # Language-specific model
            if dominant_lang and model_lang == dominant_lang:
                score += 100  # Perfect match!
            elif model_lang in [l.lower() for l in lang_list]:
                pct = lang_percentages.get(model_lang, 0)
                score += pct  # Score based on language percentage
            else:
                score += 0  # No language match

        # CRITERION 2: Text Length Compatibility (50 points max, BOOSTED to 150 if user wants long models)
        is_long_model = ('longformer' in model_name.lower() or
                       'bigbird' in model_name.lower() or
                       'long-t5' in model_name.lower() or
                       '4096' in model_name or
                       '16384' in model_name)

        if user_prefers_long_models:
            # User explicitly wants long-document models - BOOST them heavily
            if is_long_model:
                score += 150  # MASSIVE boost for long models when user wants them
            else:
                # Standard models get penalty when long models are preferred
                if text_length_avg > 500:
                    score += 10  # Not ideal - will truncate
                else:
                    score += 20  # Acceptable but not preferred
        else:
            # Normal scoring when user hasn't expressed preference
            if is_long_model:
                if text_length_avg > 400:
                    score += 50  # Perfect for long texts
                elif text_length_avg > 200:
                    score += 30  # Okay for medium texts
                else:
                    score += 10  # Overkill for short texts
            else:
                if text_length_avg <= 300:
                    score += 40  # Good for short/medium texts
                elif text_length_avg <= 500:
                    score += 30  # Okay for longer texts
                else:
                    score += 15  # Not ideal but works

        # CRITERION 3: Model Size/Efficiency (30 points max)
        if 'distil' in model_name.lower() or 'tiny' in model_name.lower() or 'small' in model_name.lower():
            score += 30  # Efficient models
        elif 'base' in model_name.lower():
            score += 25  # Standard models
        elif 'large' in model_name.lower() or 'xlarge' in model_name.lower():
            score += 15  # Large models (slower)
        else:
            score += 20  # Unknown size

        # CRITERION 4: Model Popularity/Reliability (20 points max)
        # Based on known high-quality models
        high_quality_models = {
            'xlm-roberta-base': 20,
            'xlm-roberta-large': 18,
            'bert-base-multilingual-cased': 18,
            'camembert-base': 19,
            'roberta-base': 20,
            'bert-base-uncased': 19,
            'flaubert-base': 17,
            'bert-base-german-cased': 18,
            'microsoft/deberta-v3-base': 20,
            'microsoft/mdeberta-v3-base': 19,
            'markussagen/xlm-roberta-longformer-base-4096': 20,  # Excellent multilingual long-document (100+ languages)
            'google/long-t5-local-base': 18,  # High-quality multilingual T5 long-document
            'google/long-t5-tglobal-base': 18,  # High-quality multilingual T5 long-document
            'allenai/longformer-base-4096': 17,  # Popular long-document (EN only)
            'google/bigbird-roberta-base': 17,  # Popular long-document (EN only)
        }
        score += high_quality_models.get(model_name, 15)  # Default 15 for others

        # BONUS: Recommended model gets extra points
        if recommended_model and model_name == recommended_model:
            score += 50

        model_scores[model_name] = score

    # ============================================================
    # STEP 3: Select top models by score
    # ============================================================
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top 10 models, then filter to diverse set
    top_candidates = [model for model, score in sorted_models[:20]]

    final_models = []
    selected_langs = set()

    # Strategy: Pick diverse models
    # 1. Add multilingual models first (max 2)
    multilingual_added = 0
    for model in top_candidates:
        if MODEL_LANGUAGE_MAP[model] is None and multilingual_added < 2:
            final_models.append(model)
            multilingual_added += 1

    # 2. Add language-specific models for each detected language (1-2 per language)
    for lang in lang_list[:3]:  # Limit to 3 languages
        lang_lower = lang.lower()
        lang_models_added = 0
        for model in top_candidates:
            if MODEL_LANGUAGE_MAP[model] == lang_lower and lang_models_added < 2:
                if model not in final_models:
                    final_models.append(model)
                    lang_models_added += 1
                    selected_langs.add(lang_lower)

    # 3. Fill remaining slots with highest scored models
    for model in top_candidates:
        if model not in final_models:
            final_models.append(model)
        if len(final_models) >= 7:
            break

    # Ensure we have at least some models
    if not final_models:
        final_models = ['xlm-roberta-base', 'bert-base-multilingual-cased', 'bert-base-uncased']

    # ============================================================
    # STEP 4: Display selection rationale
    # ============================================================
    if len(lang_list) > 1:
        lang_info = f"multilingual ({', '.join(lang_list[:3])})"
        if len(lang_list) > 3:
            lang_info += f" +{len(lang_list) - 3} more"
    elif len(lang_list) == 1:
        lang_info = f"{lang_list[0].upper()}"
    else:
        lang_info = "unknown language"

    text_len_info = "short" if text_length_avg < 150 else "medium" if text_length_avg < 350 else "long"

    self.console.print(f"[dim]AI Selection: {lang_info} dataset, {text_len_info} texts (avg {text_length_avg:.0f} chars)[/dim]")
    self.console.print(f"[dim]   Scored {len(MODEL_LANGUAGE_MAP)} models → Selected top {len(final_models)} by intelligent criteria[/dim]")

    # Build model-to-language mapping for selected models
    model_lang_map = {model: MODEL_LANGUAGE_MAP.get(model, None) for model in final_models}

    return final_models, model_lang_map

def _get_preselected_benchmark_models(self, languages: set, text_length_avg: float) -> List[str]:
    """
    Let user choose from pre-selected model categories.
    NOW INCLUDES ALL LANGUAGES SUPPORTED IN THE PACKAGE!
    """
    self.console.print("\n[bold]Pre-Selected Model Categories[/bold]\n")
    self.console.print("[dim]Choose from curated model lists organized by language and characteristics[/dim]\n")

    categories = {}
    lang_list = [l.lower() for l in languages] if languages else ['en']

    # ============ MULTILINGUAL MODELS ============
    if len(lang_list) > 1:
        categories['Multilingual'] = [
            'xlm-roberta-base',
            'xlm-roberta-large',
            'bert-base-multilingual-cased',
            'microsoft/mdeberta-v3-base'
        ]

    # ============ MAJOR LANGUAGES (Always show) ============
    categories['English'] = [
        'bert-base-uncased',
        'roberta-base',
        'distilbert-base-uncased',
        'microsoft/deberta-v3-base'
    ]

    if 'fr' in lang_list or True:  # Always show major languages
        categories['French'] = [
            'camembert-base',
            'flaubert/flaubert_base_cased',
            'cmarkea/distilcamembert-base',
            'almanach/camembert-base',
            'almanach/camembertav2-base'
        ]

    if 'de' in lang_list or True:
        categories['German'] = [
            'bert-base-german-cased',
            'deepset/gbert-base',
            'distilbert-base-german-cased'
        ]

    if 'es' in lang_list or True:
        categories['Spanish'] = [
            'dccuchile/bert-base-spanish-wwm-cased',
            'PlanTL-GOB-ES/roberta-base-bne',
            'bertin-project/bertin-roberta-base-spanish'
        ]

    # ============ EUROPEAN LANGUAGES ============
    if 'it' in lang_list:
        categories['Italian'] = [
            'dbmdz/bert-base-italian-cased',
            'dbmdz/bert-base-italian-xxl-cased'
        ]

    if 'pt' in lang_list:
        categories['Portuguese'] = [
            'neuralmind/bert-base-portuguese-cased',
            'neuralmind/bert-large-portuguese-cased'
        ]

    if 'nl' in lang_list:
        categories['Dutch'] = [
            'GroNLP/bert-base-dutch-cased',
            'pdelobelle/robbert-v2-dutch-base'
        ]

    if 'pl' in lang_list:
        categories['Polish'] = [
            'dkleczek/bert-base-polish-uncased-v1',
            'allegro/herbert-base-cased'
        ]

    if 'sv' in lang_list:
        categories['Swedish'] = ['KB/bert-base-swedish-cased']

    if 'da' in lang_list:
        categories['Danish'] = ['Maltehb/danish-bert-botxo']

    if 'no' in lang_list:
        categories['Norwegian'] = ['ltg/norbert', 'NbAiLab/nb-bert-base']

    if 'fi' in lang_list:
        categories['Finnish'] = ['TurkuNLP/bert-base-finnish-cased-v1']

    if 'el' in lang_list:
        categories['Greek'] = ['nlpaueb/bert-base-greek-uncased-v1']

    if 'tr' in lang_list:
        categories['Turkish'] = ['dbmdz/bert-base-turkish-cased']

    if 'ro' in lang_list:
        categories['Romanian'] = ['dumitrescustefan/bert-base-romanian-cased-v1']

    if 'bg' in lang_list:
        categories['Bulgarian'] = ['iarfmoose/roberta-base-bulgarian']

    if 'hr' in lang_list or 'sr' in lang_list:
        categories['Croatian/Serbian'] = ['classla/bcms-bertic']

    if 'uk' in lang_list:
        categories['Ukrainian'] = ['youscan/ukr-roberta-base']

    if 'cs' in lang_list:
        categories['Czech'] = ['ufal/robeczech-base']

    # ============ ASIAN LANGUAGES ============
    if 'zh' in lang_list:
        categories['Chinese'] = [
            'bert-base-chinese',
            'hfl/chinese-roberta-wwm-ext',
            'hfl/chinese-roberta-wwm-ext-large'
        ]

    if 'ja' in lang_list:
        categories['Japanese'] = [
            'cl-tohoku/bert-base-japanese',
            'nlp-waseda/roberta-base-japanese'
        ]

    if 'ko' in lang_list:
        categories['Korean'] = [
            'klue/bert-base',
            'beomi/kcbert-base'
        ]

    if 'ar' in lang_list:
        categories['Arabic'] = [
            'aubmindlab/bert-base-arabertv2',
            'CAMeL-Lab/bert-base-arabic-camelbert-msa',
            'UBC-NLP/MARBERT'
        ]

    if 'ru' in lang_list:
        categories['Russian'] = [
            'DeepPavlov/rubert-base-cased',
            'ai-forever/ruBert-base'
        ]

    if 'hi' in lang_list:
        categories['Hindi'] = ['ai4bharat/indic-bert']

    if 'vi' in lang_list:
        categories['Vietnamese'] = ['vinai/phobert-base']

    if 'th' in lang_list:
        categories['Thai'] = ['airesearch/wangchanberta-base-att-spm-uncased']

    if 'id' in lang_list:
        categories['Indonesian'] = ['indobenchmark/indobert-base-p1']

    if 'he' in lang_list:
        categories['Hebrew'] = ['onlplab/alephbert-base']

    # ============ SPECIAL CATEGORIES ============
    if text_length_avg > 400:
        categories['Long Documents (>400 chars, 4096 tokens)'] = [
            'markussagen/xlm-roberta-longformer-base-4096',  # Multilingual FIRST
            'google/long-t5-local-base',  # Multilingual
            'allenai/longformer-base-4096',  # English only
            'google/bigbird-roberta-base'  # English only
        ]

    categories['Efficient/Fast'] = [
        'distilbert-base-uncased',
        'distilroberta-base',
        'albert-base-v2',
        'squeezebert/squeezebert-uncased'
    ]

    categories['State-of-the-Art'] = [
        'microsoft/deberta-v3-base',
        'microsoft/mdeberta-v3-base',
        'google/electra-base-discriminator',
        'xlm-roberta-large'
    ]

    # Display categories in organized fashion
    self.console.print("[bold cyan]Available Categories:[/bold cyan]\n")
    for i, (cat_name, models) in enumerate(categories.items(), 1):
        model_list = ', '.join(models[:3])  # Show first 3
        if len(models) > 3:
            model_list += f" (+{len(models)-3} more)"
        self.console.print(f"  [green]{i}.[/green] [cyan]{cat_name}:[/cyan] {model_list}")

    self.console.print(f"\n[dim]Total: {len(categories)} categories available[/dim]")
    self.console.print("\n[yellow]Enter category names separated by commas[/yellow]")
    self.console.print("[dim]   Example: 'English,Multilingual' or 'French,Efficient'[/dim]\n")

    # Smart default based on detected languages
    default_cats = []
    if len(lang_list) > 1:
        default_cats.append("Multilingual")
    if 'en' in lang_list:
        default_cats.append("English")
    if 'fr' in lang_list:
        default_cats.append("French")
    if 'de' in lang_list:
        default_cats.append("German")
    if 'es' in lang_list:
        default_cats.append("Spanish")

    default_str = ','.join(default_cats) if default_cats else "Multilingual,English"

    selected_cats = Prompt.ask("Select categories", default=default_str)

    # Parse selected categories
    selected_models = []
    for cat in selected_cats.split(','):
        cat = cat.strip()
        # Case-insensitive matching
        for cat_name, models in categories.items():
            if cat.lower() in cat_name.lower():
                selected_models.extend(models)
                break

    # Deduplicate
    selected_models = list(dict.fromkeys(selected_models))

    return selected_models if selected_models else ['xlm-roberta-base', 'bert-base-multilingual-cased']

def _get_custom_benchmark_models(self) -> List[str]:
    """Let user manually select models"""
    self.console.print("\n[bold] Custom Model Selection[/bold]\n")

    all_models = self._flatten_trainer_models()
    self.console.print(f"[dim]Available models ({len(all_models)}):[/dim]")
    for i, model in enumerate(all_models, 1):
        if i % 3 == 0:
            self.console.print(f"  {model}")
        else:
            self.console.print(f"  {model}", end="  ")
    if len(all_models) % 3 != 0:
        self.console.print()

    self.console.print("\n[dim]Enter model names separated by commas, or HuggingFace model IDs[/dim]")
    models_input = Prompt.ask("Model names", default="bert-base-uncased,xlm-roberta-base")

    selected_models = [m.strip() for m in models_input.split(',')]
    return selected_models

def _save_training_metadata(
    self,
    bundle: TrainingDataBundle,
    mode: str,
    model_config: Dict[str, Any],
    execution_status: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    quick_params: Optional[Dict[str, Any]] = None,
    runtime_params: Optional[Dict[str, Any]] = None,
    training_context: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save COMPREHENSIVE training session metadata for reproducibility and resume capability.

    Now uses the enhanced MetadataManager for complete parameter capture.

    Parameters
    ----------
    bundle : TrainingDataBundle
        The training data bundle with all dataset information
    mode : str
        Training mode: quick, benchmark, custom, etc.
    model_config : dict
        Model configuration including selected_model, epochs, batch_size, etc.
    execution_status : dict, optional
        Execution status information (status, started_at, completed_at, etc.)
    session_id : str, optional
        Session ID to use (defaults to timestamp)
    quick_params : dict, optional
        Quick mode parameters if applicable
    runtime_params : dict, optional
        Runtime parameters from actual training
    training_context : dict, optional
        Additional training context information

    Returns
    -------
    Path
        Path to the saved metadata JSON file
    """
    from datetime import datetime
    from llm_tool.utils.metadata_manager import MetadataManager

    # Use provided session_id or create new one
    timestamp = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize metadata manager
    metadata_manager = MetadataManager(session_id=timestamp)

    # Save comprehensive metadata using the new manager
    metadata_path = metadata_manager.save_comprehensive_metadata(
        bundle=bundle,
        mode=mode,
        model_config=model_config,
        quick_params=quick_params,
        execution_status=execution_status,
        runtime_params=runtime_params,
        training_context=training_context
    )

    # Store metadata manager for later updates
    self._current_metadata_manager = metadata_manager

    return metadata_path

def _update_training_metadata(
    self,
    metadata_path: Path,
    **updates
) -> None:
    """
    Update existing training metadata file with new information (post-training).

    Now uses the enhanced MetadataManager for safe updates.

    Parameters
    ----------
    metadata_path : Path
        Path to the existing metadata JSON file
    **updates : dict
        Sections to update (e.g., execution_status={'status': 'completed'})
    """
    from llm_tool.utils.metadata_manager import MetadataManager

    try:
        # Use metadata manager for updates
        if hasattr(self, '_current_metadata_manager') and self._current_metadata_manager:
            # Use existing manager if available
            self._current_metadata_manager.update_metadata(**updates)
        else:
            # Create new manager from path
            session_id = metadata_path.parent.parent.name
            metadata_manager = MetadataManager(session_id=session_id)
            metadata_manager.update_metadata(**updates)

    except Exception as e:
        self.logger.error(f"Failed to update metadata: {e}")

        # Fallback to direct JSON update
        import json
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                for section, data in updates.items():
                    if section in metadata:
                        if isinstance(metadata[section], dict) and isinstance(data, dict):
                            metadata[section].update(data)
                        else:
                            metadata[section] = data
                    else:
                        metadata[section] = data

                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            except Exception as fallback_error:
                self.logger.error(f"Fallback update also failed: {fallback_error}")

def _reconstruct_bundle_from_metadata(self, metadata: Dict[str, Any]) -> Optional[TrainingDataBundle]:
    """
    Reconstruct a TrainingDataBundle from saved metadata for resume/relaunch.

    Now handles the comprehensive metadata format from MetadataManager.

    Parameters
    ----------
    metadata : dict
        Loaded metadata dictionary from JSON file

    Returns
    -------
    TrainingDataBundle or None
        Reconstructed bundle, or None if reconstruction fails
    """
    try:
        # Handle both old and new metadata formats
        dataset_config = metadata.get('dataset_config', {})
        language_config = metadata.get('language_config', {})
        text_analysis = metadata.get('text_analysis', {})
        split_config = metadata.get('split_config', {})
        label_config = metadata.get('label_config', {})
        preprocessing_config = metadata.get('preprocessing', {})

        # Load primary file
        primary_file_str = dataset_config.get('primary_file')
        if not primary_file_str:
            self.console.print("[red]Error: No primary file found in metadata[/red]")
            return None

        primary_file = Path(primary_file_str)
        if not primary_file.exists():
            self.console.print(f"[red]Error: Dataset file not found: {primary_file}[/red]")
            # Check for training files as fallback
            if dataset_config.get('training_files'):
                self.console.print("[yellow]Primary file missing, but training files may be available[/yellow]")
            else:
                return None

        # Create bundle with comprehensive info
        # Note: TrainingDataBundle doesn't accept format_type or samples parameters
        bundle = TrainingDataBundle(
            primary_file=primary_file if primary_file.exists() else None,
            strategy=dataset_config.get('strategy', 'single-label'),
            text_column=dataset_config.get('text_column', 'text'),
            label_column=dataset_config.get('label_column', 'label'),
            metadata={}
        )

        # Store format information in metadata instead
        bundle.metadata['format_type'] = dataset_config.get('format_type', dataset_config.get('format', 'unknown'))
        bundle.metadata['format'] = dataset_config.get('format', 'unknown')

        # Restore training_files if present
        if dataset_config.get('training_files'):
            bundle.training_files = {
                label: Path(path)
                for label, path in dataset_config['training_files'].items()
            }

        # Restore ALL metadata fields comprehensively
        # Language configuration
        bundle.metadata['confirmed_languages'] = set(language_config.get('confirmed_languages', []))
        bundle.metadata['language_distribution'] = language_config.get('language_distribution', {})
        bundle.metadata['model_strategy'] = language_config.get('model_strategy', 'multilingual')
        bundle.metadata['language_model_mapping'] = language_config.get('language_model_mapping', {})
        bundle.metadata['per_language_training'] = language_config.get('per_language_training', False)
        bundle.metadata['models_by_language'] = language_config.get('models_by_language', {})

        # Text analysis
        bundle.metadata['text_length_stats'] = text_analysis.get('text_length_stats', {})
        bundle.metadata['requires_long_document_model'] = text_analysis.get('requires_long_document_model', False)
        bundle.metadata['user_prefers_long_models'] = text_analysis.get('user_prefers_long_models', False)
        bundle.metadata['exclude_long_texts'] = text_analysis.get('exclude_long_texts', False)
        bundle.metadata['split_long_texts'] = text_analysis.get('split_long_texts', False)

        # Label configuration
        bundle.metadata['categories'] = dataset_config.get('categories', list(dataset_config.get('category_distribution', {}).keys()))
        bundle.metadata['category_distribution'] = dataset_config.get('category_distribution', {})
        bundle.metadata['num_categories'] = dataset_config.get('num_categories', len(bundle.metadata['categories']))
        bundle.metadata['label_type'] = label_config.get('label_type', 'single')
        bundle.metadata['label_mapping'] = label_config.get('label_mapping', {})
        bundle.metadata['imbalanced_labels'] = label_config.get('imbalanced_labels', [])

        # Dataset configuration
        bundle.metadata['source_file'] = dataset_config.get('source_file')
        bundle.metadata['annotation_column'] = dataset_config.get('annotation_column')
        bundle.metadata['training_approach'] = dataset_config.get('training_approach')
        bundle.metadata['original_strategy'] = dataset_config.get('original_strategy')

        # CRITICAL FIX: Restore hybrid/custom training configuration
        # These fields are REQUIRED for session relaunch to work with hybrid training
        bundle.metadata['multiclass_keys'] = dataset_config.get('multiclass_keys', [])
        bundle.metadata['multilabel_keys'] = dataset_config.get('multilabel_keys', [])
        bundle.metadata['onevsall_keys'] = dataset_config.get('onevsall_keys', [])
        bundle.metadata['key_strategies'] = dataset_config.get('key_strategies', {})
        bundle.metadata['files_per_key'] = dataset_config.get('files_per_key', {})
        raw_all_keys_values = dataset_config.get('all_keys_values', {})
        if raw_all_keys_values:
            bundle.metadata['all_keys_values'] = {
                key: {str(v) for v in values} if isinstance(values, (list, tuple, set)) else set()
                for key, values in raw_all_keys_values.items()
            }
        else:
            bundle.metadata['all_keys_values'] = {}
        raw_value_counts = dataset_config.get('value_counts_by_key', {})
        bundle.metadata['value_counts_by_key'] = {
            key: {val: int(count) for val, count in (counts or {}).items()}
            for key, counts in raw_value_counts.items()
        }

        # Split configuration
        if split_config:
            bundle.metadata['split_config'] = split_config

        # Preprocessing
        if preprocessing_config:
            bundle.metadata['preprocessing'] = preprocessing_config

        # Restore training files paths if they exist
        training_files_dict = dataset_config.get('training_files', {})
        if training_files_dict:
            bundle.training_files = {k: Path(v) for k, v in training_files_dict.items()}

        # Restore model configuration
        model_config = metadata.get('model_config', {})
        if model_config:
            bundle.metadata['recommended_model'] = model_config.get('recommended_model')
            bundle.metadata['selected_model'] = model_config.get('selected_model')
            bundle.metadata['models_by_language'] = model_config.get('models_by_language', {})

        # Restore advanced settings
        advanced_settings = metadata.get('advanced_settings', {})
        if advanced_settings:
            bundle.metadata['benchmark_mode'] = advanced_settings.get('benchmark_mode', False)
            bundle.metadata['one_vs_all'] = advanced_settings.get('one_vs_all', False)
            # CRITICAL FIX: Derive multi_label from training_approach if set, otherwise use advanced_settings
            training_approach = dataset_config.get('training_approach', '')
            bundle.metadata['multi_label'] = (
                training_approach == 'multi-label' or
                advanced_settings.get('multi_label', False)
            )

        # CRITICAL FIX: Infer multi_label_per_key for parallel training on relaunch
        # If training_approach is 'multi-label' AND we have multiple training_files (excluding 'multilabel'),
        # then this was a per-key multi-label training that should offer parallel training
        training_approach = dataset_config.get('training_approach', '')
        training_files_dict = dataset_config.get('training_files', {})
        # Count actual category files (exclude the merged 'multilabel' file)
        category_files_count = len([k for k in training_files_dict.keys() if k != 'multilabel'])

        if training_approach == 'multi-label' and category_files_count > 1:
            bundle.metadata['multi_label_per_key'] = True
            self.logger.info(f"[RELAUNCH] Detected per-key multi-label training with {category_files_count} categories - parallel training will be offered")
        else:
            bundle.metadata['multi_label_per_key'] = advanced_settings.get('multi_label_per_key', False) if advanced_settings else False

        # Set recommended model if available
        if 'recommended_model' in model_config:
            bundle.recommended_model = model_config['recommended_model']

        return bundle

    except Exception as e:
        self.logger.error(f"Failed to reconstruct bundle from metadata: {e}")
        self.console.print(f"[red]Error reconstructing dataset: {e}[/red]")
        return None

def _resolve_training_metadata(self, session_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Load or reconstruct training metadata for a session."""
    metadata_path = session_dir / "training_session_metadata" / "training_metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata.setdefault("_recovered", False)
            return metadata_path, metadata
        except Exception as err:
            if hasattr(self, "logger"):
                self.logger.warning("Could not load training metadata %s: %s", metadata_path, err)
            return None

    try:
        session_id = session_dir.name
        parts = session_id.rsplit("_", 2)
        session_name = session_id
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(parts) >= 3 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
            session_name = "_".join(parts[:-2]) or session_id
            timestamp_str = f"{parts[-2]}_{parts[-1]}"

        workflow_label = "Training Arena"
        if "factory" in session_id:
            workflow_label = "Annotator Factory Training"

        minimal_metadata: Dict[str, Any] = {
            "metadata_version": "2.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "training_session": {
                "session_id": session_id,
                "timestamp": timestamp_str,
                "workflow": workflow_label,
                "mode": "arena",
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "user": platform.node(),
            },
            "dataset_config": {
                "primary_file": None,
                "format_type": "unknown",
                "strategy": "single-label",
            },
            "language_config": {},
            "text_analysis": {},
            "split_config": {},
            "label_config": {},
            "model_config": {
                "training_mode": "quick",
                "selected_model": None,
            },
            "training_params": {},
            "reinforced_learning_config": {},
            "execution_status": {
                "status": "unknown",
                "current_model": None,
                "current_epoch": None,
                "best_model": None,
                "models_trained": [],
            },
            "output_paths": {
                "session_dir": str(session_dir),
                "models_dir": str(Path("models") / session_id),
                "logs_dir": str(session_dir),
            },
            "preprocessing": {},
            "advanced_settings": {},
            "checkpoints": {},
            "training_context": {},
            "_recovered": True,
        }

        training_data_dir = session_dir / "training_data"
        if training_data_dir.exists():
            for train_file in training_data_dir.glob("train_*.csv"):
                minimal_metadata["dataset_config"]["primary_file"] = str(train_file)
                break

        return metadata_path, minimal_metadata
    except Exception as err:
        if hasattr(self, "logger"):
            self.logger.debug("Failed to recover metadata for %s: %s", session_dir, err)
        return None


def _resume_training_studio(self, focus_session_id: Optional[str] = None):
    """Resume or relaunch training using saved parameters."""

    self.console.print("\n[bold cyan]Resume/Relaunch Training[/bold cyan]\n")
    self.console.print("[dim]Load saved parameters from previous training sessions[/dim]\n")

    base_dir = get_training_logs_base()
    if not base_dir.exists():
        self.console.print("[yellow][!] Training arena logs directory not found.[/yellow]")
        self.console.print(f"[dim]Expected location: {base_dir}[/dim]")
        self.console.print("[dim]Complete a training first to create session history.[/dim]")
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        input()
        return

    records = collect_summaries_for_mode(base_dir, "training_arena", limit=25)
    if not records:
        self.console.print("[yellow]No training sessions found.[/yellow]")
        self.console.print("[dim]Run a training session to populate the history.[/dim]")
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        input()
        return

    sessions_table = Table(
        title="Previous Training Sessions (25 most recent)",
        border_style="cyan",
        box=box.ROUNDED,
        expand=True,
    )
    sessions_table.add_column("#", style="cyan bold", width=4, no_wrap=True)
    sessions_table.add_column("Session", style="white", no_wrap=True)
    sessions_table.add_column("Date", style="yellow", no_wrap=True)
    sessions_table.add_column("Time", style="yellow", no_wrap=True)
    sessions_table.add_column("Mode", style="magenta", no_wrap=True)
    sessions_table.add_column("Dataset", style="green", overflow="fold")
    sessions_table.add_column("Model", style="blue", overflow="fold")
    sessions_table.add_column("Last Step", style="cyan", overflow="fold")
    sessions_table.add_column("Status", style="white", no_wrap=True)

    valid_sessions: List[Tuple[Path, Dict[str, Any], Any, Path]] = []
    for idx, record in enumerate(records, 1):
        resolved = self._resolve_training_metadata(record.directory)
        if not resolved:
            continue
        metadata_path, metadata = resolved
        summary = record.summary

        dataset_config = metadata.get("dataset_config", {})
        model_config = metadata.get("model_config", {})
        exec_status = metadata.get("execution_status", {})
        session_info = metadata.get("training_session", {})

        dataset_name = dataset_config.get("primary_file") or summary.extra.get("dataset") or "-"
        model_name = (
            model_config.get("selected_model")
            or exec_status.get("current_model")
            or summary.extra.get("current_model")
            or "-"
        )
        workflow_label = session_info.get("workflow", summary.extra.get("workflow", "Training Arena"))

        try:
            dt_obj = datetime.fromisoformat(summary.updated_at)
            date_str = dt_obj.strftime("%Y-%m-%d")
            time_str = dt_obj.strftime("%H:%M")
        except ValueError:
            parts = summary.updated_at.split("T")
            date_str = parts[0]
            time_str = parts[1] if len(parts) > 1 else ""

        last_step = summary.last_step_name or summary.last_step_key or "-"
        if summary.last_step_no:
            last_step = f"{summary.last_step_no}. {last_step}"

        sessions_table.add_row(
            str(idx),
            summary.session_name or summary.session_id,
            date_str,
            time_str,
            workflow_label,
            dataset_name,
            model_name,
            last_step,
            summary.status,
        )

        valid_sessions.append((metadata_path, metadata, summary, record.directory))

    if not valid_sessions:
        self.console.print("[yellow]No valid training sessions were found.[/yellow]")
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        input()
        return

    self.console.print(sessions_table)

    session_choice: Optional[int] = None
    if focus_session_id:
        for idx, (_, _, summary, _) in enumerate(valid_sessions, 1):
            if summary.session_id == focus_session_id:
                session_choice = idx
                self.console.print(f"\n[dim]Auto-selecting session {summary.session_id}[/dim]")
                break

    if session_choice is None:
        session_choice = self._int_prompt_with_validation(
            "\n[bold yellow]Select session to resume/relaunch[/bold yellow]",
            1,
            1,
            len(valid_sessions),
        )

    selected_file, metadata, summary, session_dir = valid_sessions[session_choice - 1]

    self.console.print(f"\n[green]✓ Selected: {summary.session_id}[/green]")
    last_step = summary.last_step_name or summary.last_step_key or "-"
    self.console.print(f"[dim]Status: {summary.status} • Last step: {last_step}[/dim]")

    self._display_metadata_parameters(metadata)

    is_recovered_session = metadata.get("_recovered", False)

    if is_recovered_session:
        self.console.print("\n[yellow][!]  Recovered session: parameters may be incomplete.[/yellow]")

    # Check if RL-ready state exists (normal training completed, RL can be resumed)
    rl_state_available = False
    rl_state_path = None
    try:
        import glob as glob_mod
        session_id_check = metadata.get("training_session", {}).get("session_id", "")
        rl_candidates = glob_mod.glob(f"logs/training_arena/{session_id_check}/training_metrics/**/rl_ready_state.json", recursive=True)
        if rl_candidates:
            rl_state_available = True
            rl_state_path = rl_candidates[0]
    except Exception:
        pass

    self.console.print("\n[bold cyan]Action Mode[/bold cyan]")
    self.console.print("  • [cyan]resume[/cyan]     - Continue incomplete training (if interrupted)")
    if rl_state_available:
        self.console.print("  • [cyan]resume_rl[/cyan]  - Skip to RL Phase 2 (normal training completed, RL crashed)")
    self.console.print("  • [cyan]relaunch[/cyan]  - Start fresh with same parameters\n")

    choices = ["resume", "relaunch"]
    if rl_state_available:
        choices.insert(1, "resume_rl")

    action_mode = Prompt.ask(
        "[bold yellow]Select action[/bold yellow]",
        choices=choices,
        default="resume_rl" if rl_state_available else "relaunch",
    )

    self.console.print(f"\n[cyan]Reconstructing dataset configuration...[/cyan]")
    bundle = self._reconstruct_bundle_from_metadata(metadata)

    if bundle is None:
        self.console.print("[red]Failed to reconstruct training configuration.[/red]")
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        input()
        return

    session_info = metadata.get("training_session", {})
    session_id = session_info.get("session_id")
    if not session_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"relaunch_{timestamp}"

    session_manager = TrainingDataSessionManager(session_id=session_id)
    self.current_session_id = session_id
    self.current_session_manager = session_manager

    self.console.print(f"[dim]Session ID: {session_id}[/dim]\n")

    if action_mode == "resume":
        self.console.print("\n[green]✓ Resuming training session...[/green]\n")
    elif action_mode == "resume_rl":
        self.console.print("\n[green]✓ Resuming at RL Phase 2 (skipping normal training)...[/green]\n")
    else:
        self.console.print("\n[green]✓ Relaunching training with saved parameters...[/green]\n")

    # ── Ask for early stopping & interactive skip on resume ──
    if action_mode in ("resume", "relaunch"):
        self.console.print("[bold]Training Control Options[/bold]\n")
        self.console.print("[bold]Early Stopping[/bold]")
        self.console.print("  Automatically stop a model if F1 stops improving.\n")
        _es_enabled = Confirm.ask("Enable early stopping?", default=True)
        _es_patience = None
        if _es_enabled:
            _es_patience = IntPrompt.ask("  Patience (epochs without improvement)", default=3)
            self.console.print(f"  [green]✓[/green] Early stopping: patience={_es_patience}\n")
        self.console.print("\n[bold]Interactive Skip[/bold]")
        self.console.print("  Type [bold cyan]s[/bold cyan] + Enter during training to skip to next model.\n")
        _interactive_skip = Confirm.ask("Enable interactive skip (manual model skip)?", default=True)
        if _interactive_skip:
            self.console.print("  [green]✓[/green] Interactive skip enabled\n")
        else:
            self.console.print("  [dim]Interactive skip disabled[/dim]\n")

    mode = metadata.get("model_config", {}).get("training_mode", "quick")

    preloaded = metadata.get("model_config", {})

    # Inject early stopping & interactive skip into preloaded config
    if action_mode in ("resume", "relaunch") and '_es_patience' in dir():
        preloaded['early_stopping_patience'] = _es_patience
    if action_mode in ("resume", "relaunch") and '_interactive_skip' in dir():
        preloaded['interactive_skip'] = _interactive_skip

    if action_mode == "resume_rl" and rl_state_path:
        preloaded['skip_to_rl'] = True
        preloaded['rl_state_path'] = rl_state_path
        preloaded['use_reinforcement'] = True  # Force RL on

    self._training_studio_confirm_and_execute(
        bundle,
        mode,
        preloaded_config=preloaded,
        is_resume=action_mode in ("resume", "resume_rl"),
        step_context="arena_quick",
    )
def _training_studio_default_model(self) -> str:
    models = self._flatten_trainer_models()
    return "bert-base-uncased" if "bert-base-uncased" in models else (models[0] if models else "bert-base-uncased")

def _show_analysis_and_get_columns(self, analysis: Dict[str, Any], format_type: str = "general") -> Dict[str, Any]:
    """
    Show file analysis results and intelligently detect columns with user confirmation.
    Returns dictionary with detected column names and confirmed languages.
    """
    result = {
        'text': 'text',
        'label': 'label',
        'id': None,
        'lang': None,
        'confirmed_languages': set()
    }

    # Show analysis issues
    if analysis['issues']:
        self.console.print("\n[yellow][!]  Analysis Results:[/yellow]")
        for issue in analysis['issues']:
            self.console.print(f"  {issue}")

    all_columns = analysis.get('all_columns', [])

    # Auto-suggest text column
    text_column_default = "text"
    if analysis['text_column_candidates']:
        best_text = analysis['text_column_candidates'][0]['name']
        text_column_default = best_text
        self.console.print(f"\n[green]✓ Text column detected: '{best_text}'[/green]")

    if all_columns:
        self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

    result['text'] = Prompt.ask("Text column", default=text_column_default)

    # Auto-suggest label column
    label_column_default = "labels" if "multi" in format_type else "label"
    annotation_candidates = analysis.get('annotation_column_candidates', [])
    if annotation_candidates:
        best_label = annotation_candidates[0]['name']
        label_column_default = best_label
        self.console.print(f"\n[green]✓ Label column detected: '{best_label}'[/green]")
        stats = analysis['annotation_stats'].get(best_label, {})
        fill_rate = stats.get('fill_rate', 0)
        if fill_rate > 0:
            self.console.print(f"[dim]  ({fill_rate*100:.1f}% of rows have labels)[/dim]")

    if all_columns:
        self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

    result['label'] = Prompt.ask("Label/Category column", default=label_column_default)

    # Language detection
    languages_found = set(analysis['languages_detected'].keys())

    if languages_found:
        self.console.print(f"\n[bold]Languages Detected:[/bold]")
        for lang, count in analysis['languages_detected'].items():
            self.console.print(f"  • {lang.upper()}: {count} rows")

        lang_list = ', '.join([l.upper() for l in sorted(languages_found)])
        lang_confirmed = Confirm.ask(
            f"\n[bold]Detected languages: {lang_list}. Is this correct?[/bold]",
            default=True
        )

        if lang_confirmed:
            result['confirmed_languages'] = languages_found
            self.console.print("[green]Languages confirmed[/green]")
        else:
            self.console.print("\n[yellow]Please specify languages manually[/yellow]")
            manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
            result['confirmed_languages'] = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

        # Auto-suggest language column if detected
        if analysis['language_column_candidates']:
            lang_column_default = analysis['language_column_candidates'][0]
            self.console.print(f"\n[green]✓ Language column detected: '{lang_column_default}'[/green]")
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
            result['lang'] = Prompt.ask("Language column (optional)", default=lang_column_default)
    else:
        # No language column detected - ask if user wants to apply language detection
        self.console.print("\n[yellow]ℹ No language column detected in data[/yellow]")
        apply_lang_detection = Confirm.ask(
            "Would you like to apply automatic language detection on the text column?",
            default=True
        )

        if apply_lang_detection:
            self.console.print("[cyan]Detecting languages from text content...[/cyan]")
            self.console.print("[dim]  Language detection will be applied during training[/dim]")
            manual_langs = Prompt.ask(
                "Expected language codes (optional, comma-separated, e.g., en,fr,de)",
                default=""
            )
            if manual_langs.strip():
                result['confirmed_languages'] = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

    # Auto-suggest ID column
    if analysis['id_column_candidates']:
        id_column_default = analysis['id_column_candidates'][0]
        self.console.print(f"\n[green]✓ ID column detected: '{id_column_default}'[/green]")
        if all_columns:
            self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
        result['id'] = Prompt.ask("Identifier column (optional)", default=id_column_default)

    return result

def _get_long_document_model_recommendation(self, confirmed_languages: set) -> Optional[str]:
    """
    Get long-document model recommendations based on languages.
    Prioritizes models that can handle >512 tokens.
    """
    # Available long-document models
    LONG_DOCUMENT_MODELS = [
        {
            'model': 'allenai/longformer-base-4096',
            'max_tokens': 4096,
            'languages': ['en'],
            'reason': 'English long-document model (4096 tokens)'
        },
        {
            'model': 'google/bigbird-roberta-base',
            'max_tokens': 4096,
            'languages': ['en'],
            'reason': 'English sparse-attention long-document model (4096 tokens)'
        },
        {
            'model': 'markussagen/xlm-roberta-longformer-base-4096',
            'max_tokens': 4096,
            'languages': ['multilingual'],
            'reason': 'Multilingual long-document model (4096 tokens)'
        },
        {
            'model': 'xlm-roberta-base',
            'max_tokens': 512,
            'languages': ['multilingual'],
            'reason': 'Multilingual baseline (512 tokens, fallback)'
        },
    ]

    # Filter models based on languages
    suitable_models = []
    for model in LONG_DOCUMENT_MODELS:
        if 'multilingual' in model['languages']:
            suitable_models.append(model)
        elif confirmed_languages:
            if any(lang in model['languages'] for lang in confirmed_languages):
                suitable_models.append(model)

    if not suitable_models:
        suitable_models = LONG_DOCUMENT_MODELS  # Fallback to all

    self.console.print(f"\n[bold]Long-Document Model Recommendations:[/bold]")
    for i, model_info in enumerate(suitable_models[:5], 1):
        self.console.print(f"  {i}. [cyan]{model_info['model']}[/cyan] - {model_info['reason']}")

    choice = Prompt.ask(
        f"Select model (1-{min(5, len(suitable_models))}, or enter model name)",
        default="1"
    )

    if choice.isdigit() and 0 < int(choice) <= len(suitable_models):
        model_to_use = suitable_models[int(choice) - 1]['model']
        self.console.print(f"[green]Selected: {model_to_use}[/green]")
        return model_to_use
    else:
        return choice

def _get_long_document_models_for_language(self, lang: str) -> list:
    """
    Get long-document model recommendations for a specific language.
    Returns list in LanguageNormalizer.recommend_models format.
    Uses the model catalog (TrainerModelDetector) when available.
    """
    # Try to get models from catalog first
    if self.available_trainer_models:
        # Map language codes to catalog categories
        LANG_TO_CATEGORY = {
            'en': 'Long Document Models',
            'fr': 'Long Document Models - French',
            'es': 'Long Document Models - Spanish',
            'de': 'Long Document Models - German',
            'it': 'Long Document Models - Italian',
            'pt': 'Long Document Models - Portuguese',
            'nl': 'Long Document Models - Dutch',
            'pl': 'Long Document Models - Polish',
            'ru': 'Long Document Models - Russian',
            'zh': 'Long Document Models - Chinese',
            'ja': 'Long Document Models - Japanese',
            'ar': 'Long Document Models - Arabic',
        }

        category = LANG_TO_CATEGORY.get(lang, 'Long Document Models')

        # Get models from catalog
        if category in self.available_trainer_models:
            catalog_models = self.available_trainer_models[category]
            recommendations = []

            for model in catalog_models:
                # Build reason from model metadata
                reason_parts = [
                    model.get('type', 'Unknown type'),
                    f"({model.get('max_length', '512')} tokens)"
                ]
                if model.get('performance'):
                    reason_parts.append(model['performance'])

                recommendations.append({
                    'model': model['name'],
                    'reason': ' - '.join(reason_parts)
                })

            # Add multilingual fallback if not already included
            if lang != 'en' and 'Long Document Models' in self.available_trainer_models:
                for model in self.available_trainer_models['Long Document Models'][:2]:
                    if 'xlm' in model['name'].lower() or 'multilingual' in model.get('type', '').lower():
                        recommendations.append({
                            'model': model['name'],
                            'reason': f"{model.get('type')} - Multilingual fallback ({model.get('max_length', '4096')} tokens)"
                        })

            if recommendations:
                return recommendations

    # Fallback: hardcoded comprehensive list if catalog unavailable
    LANG_LONG_MODELS = {
        'en': [
            {'model': 'allenai/longformer-base-4096', 'reason': 'English Longformer (4096 tokens, optimized for English)'},
            {'model': 'google/bigbird-roberta-base', 'reason': 'English BigBird sparse-attention (4096 tokens)'},
            {'model': 'google/long-t5-local-base', 'reason': 'Multilingual T5 for long documents (4096+ tokens)'},
            {'model': 'roberta-base', 'reason': 'English RoBERTa baseline (512 tokens, fallback)'},
        ],
        'fr': [
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting French (4096 tokens)'},
            {'model': 'google/long-t5-local-base', 'reason': 'Multilingual T5 for long documents (4096+ tokens)'},
            {'model': 'cmarkea/distilcamembert-base-nli', 'reason': 'French DistilCamemBERT optimized (512 tokens)'},
            {'model': 'camembert-base', 'reason': 'French CamemBERT baseline (512 tokens)'},
        ],
        'es': [
            {'model': 'PlanTL-GOB-ES/roberta-base-bne', 'reason': 'Spanish RoBERTa optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Spanish (4096 tokens)'},
            {'model': 'dccuchile/bert-base-spanish-wwm-cased', 'reason': 'Spanish BERT baseline (512 tokens)'},
            {'model': 'bertin-project/bertin-roberta-base-spanish', 'reason': 'Spanish BERTIN RoBERTa (512 tokens)'},
        ],
        'de': [
            {'model': 'deepset/gbert-base', 'reason': 'German GBERT optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting German (4096 tokens)'},
            {'model': 'bert-base-german-cased', 'reason': 'German BERT baseline (512 tokens)'},
            {'model': 'dbmdz/bert-base-german-uncased', 'reason': 'German BERT uncased (512 tokens)'},
        ],
        'it': [
            {'model': 'dbmdz/bert-base-italian-cased', 'reason': 'Italian BERT optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Italian (4096 tokens)'},
            {'model': 'dbmdz/bert-base-italian-xxl-cased', 'reason': 'Italian BERT XXL (512 tokens, high performance)'},
        ],
        'pt': [
            {'model': 'neuralmind/bert-base-portuguese-cased', 'reason': 'Portuguese BERT optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Portuguese (4096 tokens)'},
            {'model': 'adalbertojunior/distilbert-portuguese-cased', 'reason': 'Portuguese DistilBERT (512 tokens, efficient)'},
        ],
        'nl': [
            {'model': 'GroNLP/bert-base-dutch-cased', 'reason': 'Dutch BERT optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Dutch (4096 tokens)'},
            {'model': 'wietsedv/bert-base-dutch-cased', 'reason': 'Dutch BERT baseline (512 tokens)'},
        ],
        'pl': [
            {'model': 'allegro/herbert-base-cased', 'reason': 'Polish HerBERT optimized (514 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Polish (4096 tokens)'},
            {'model': 'dkleczek/bert-base-polish-cased-v1', 'reason': 'Polish BERT baseline (512 tokens)'},
        ],
        'ru': [
            {'model': 'DeepPavlov/rubert-base-cased', 'reason': 'Russian RuBERT optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Russian (4096 tokens)'},
            {'model': 'sberbank-ai/ruBert-base', 'reason': 'Russian BERT baseline (512 tokens)'},
        ],
        'zh': [
            {'model': 'hfl/chinese-roberta-wwm-ext', 'reason': 'Chinese RoBERTa WWM optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Chinese (4096 tokens)'},
            {'model': 'bert-base-chinese', 'reason': 'Chinese BERT baseline (512 tokens)'},
        ],
        'ja': [
            {'model': 'cl-tohoku/bert-base-japanese-whole-word-masking', 'reason': 'Japanese BERT WWM optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Japanese (4096 tokens)'},
            {'model': 'cl-tohoku/bert-base-japanese', 'reason': 'Japanese BERT baseline (512 tokens)'},
        ],
        'ar': [
            {'model': 'aubmindlab/bert-base-arabert', 'reason': 'Arabic AraBERT optimized (512 tokens)'},
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Arabic (4096 tokens)'},
            {'model': 'asafaya/bert-base-arabic', 'reason': 'Arabic BERT baseline (512 tokens)'},
        ],
        'multilingual': [
            {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer (100+ languages, 4096 tokens)'},
            {'model': 'xlm-roberta-large', 'reason': 'Multilingual XLM-RoBERTa large (100+ languages, 512 tokens)'},
            {'model': 'xlm-roberta-base', 'reason': 'Multilingual XLM-RoBERTa base (100+ languages, 512 tokens)'},
            {'model': 'bert-base-multilingual-cased', 'reason': 'Multilingual BERT baseline (104 languages, 512 tokens)'},
        ],
    }

    # Return language-specific models or multilingual as fallback
    return LANG_LONG_MODELS.get(lang, LANG_LONG_MODELS.get('multilingual', []))

def _get_model_recommendation_from_languages(self, confirmed_languages: set) -> Optional[str]:
    """
    Get model recommendations based on detected/confirmed languages.
    Returns selected model name or None.
    """
    if not confirmed_languages:
        return None

    recommendations = LanguageNormalizer.recommend_models(confirmed_languages, self.available_trainer_models)

    if not recommendations:
        return None

    self.console.print(f"\n[bold]Recommended Models for Your Languages:[/bold]")
    for i, rec in enumerate(recommendations[:5], 1):
        self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

    # Interactive model selection
    self.console.print(f"\n[bold]Select a model:[/bold]")
    self.console.print("  [cyan]1-{num}[/cyan] - Select from recommendations above".format(num=min(5, len(recommendations))))
    self.console.print("  [cyan]manual[/cyan] - Enter model name manually")
    self.console.print("  [cyan]skip[/cyan] - Use default (bert-base-uncased)")

    model_choice = Prompt.ask("Your choice", default="1")

    if model_choice == "manual":
        return Prompt.ask("\nEnter model name", default="xlm-roberta-base")
    elif model_choice == "skip":
        return "bert-base-uncased"
    elif model_choice.isdigit():
        idx = int(model_choice) - 1
        if 0 <= idx < len(recommendations):
            model_to_use = recommendations[idx]['model']
            self.console.print(f"[green]Selected: {model_to_use}[/green]")
            return model_to_use
        else:
            self.console.print("[yellow]Invalid selection, using first recommendation[/yellow]")
            return recommendations[0]['model']
    else:
        return recommendations[0]['model']

def _detect_languages_and_analyze_text(
    self,
    df: 'pd.DataFrame',
    text_column: str,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Universal function to detect languages and analyze text characteristics.
    Works for ANY dataset format and ANY training mode.

    Args:
        df: DataFrame containing the text data
        text_column: Name of the column containing text
        sample_size: Number of samples to analyze for language detection

    Returns:
        Dictionary with:
        - languages_detected: {lang: count} dictionary
        - text_length_stats: {avg_length, max_length, min_length, median_length}
        - long_document_percentage: Percentage of documents > 512 tokens
        - user_prefers_long_models: Boolean recommendation
    """
    from llm_tool.utils.language_detector import LanguageDetector

    # Initialize results
    results = {
        'languages_detected': {},
        'text_length_stats': {
            'avg_length': 0,
            'max_length': 0,
            'min_length': 0,
            'median_length': 0
        },
        'long_document_percentage': 0,
        'user_prefers_long_models': False
    }

    # Check if text column exists
    if text_column not in df.columns:
        self.logger.warning(f"Text column '{text_column}' not found in dataset")
        return results

    # Get text samples (filter out NaN values)
    text_samples = df[text_column].dropna()

    if len(text_samples) == 0:
        self.logger.warning("No text data found in dataset")
        return results

    # Sample for language detection (use up to sample_size rows)
    sample_texts = text_samples.head(sample_size).tolist()

    # Detect languages using LanguageDetector
    detector = LanguageDetector()
    language_counts = Counter()

    for text in sample_texts:
        if isinstance(text, str) and text.strip():
            detected_lang = detector.detect(text)
            if detected_lang:
                # LanguageDetector returns dict like {'language': 'fr', 'confidence': 0.95}
                if isinstance(detected_lang, dict):
                    lang = detected_lang.get('language')
                    if lang:
                        language_counts[lang] += 1
                elif isinstance(detected_lang, str):
                    language_counts[detected_lang] += 1

    results['languages_detected'] = dict(language_counts)

    # Calculate text length statistics
    text_lengths = [len(str(text)) for text in text_samples if pd.notna(text)]

    if text_lengths:
        import statistics
        results['text_length_stats'] = {
            'avg_length': sum(text_lengths) / len(text_lengths),
            'max_length': max(text_lengths),
            'min_length': min(text_lengths),
            'median_length': statistics.median(text_lengths)
        }

        # Estimate long documents (assuming ~4 chars per token)
        long_docs = sum(1 for length in text_lengths if length > 2048)  # 512 tokens * 4 chars
        results['long_document_percentage'] = (long_docs / len(text_lengths)) * 100

        # Recommend long-document models if >20% of docs are long
        results['user_prefers_long_models'] = results['long_document_percentage'] > 20

    return results

def _display_language_analysis_and_get_model(
    self,
    analysis_results: Dict[str, Any],
    interactive: bool = True
) -> Tuple[Set[str], Optional[str]]:
    """
    Display language analysis results and get model recommendation.
    Universal function that works for ANY dataset format and ANY training mode.

    Args:
        analysis_results: Results from _detect_languages_and_analyze_text
        interactive: If True, ask user to confirm languages and select model

    Returns:
        Tuple of (confirmed_languages, selected_model)
    """
    # LanguageNormalizer is defined at the module level, no need to import

    languages_found = set(analysis_results['languages_detected'].keys())
    text_stats = analysis_results['text_length_stats']
    confirmed_languages = set()
    model_to_use = None

    # Display language detection results
    if languages_found:
        self.console.print(f"\n[bold]Languages Detected:[/bold]")
        for lang, count in analysis_results['languages_detected'].items():
            self.console.print(f"  • {lang.upper()}: {count} samples")

        # Display text statistics
        self.console.print(f"\n[bold]Text Statistics:[/bold]")
        self.console.print(f"  • Average length: {text_stats['avg_length']:.0f} characters")
        self.console.print(f"  • Max length: {text_stats['max_length']:.0f} characters")
        self.console.print(f"  • Median length: {text_stats['median_length']:.0f} characters")

        if analysis_results['long_document_percentage'] > 0:
            self.console.print(f"  • Long documents (>512 tokens): {analysis_results['long_document_percentage']:.1f}%")

            if analysis_results['user_prefers_long_models']:
                self.console.print("\n[yellow]Tip: Recommendation: Consider using long-document models (e.g., Longformer, BigBird)[/yellow]")

        if interactive:
            # Confirm languages with user
            lang_list = ', '.join([l.upper() for l in sorted(languages_found)])
            lang_confirmed = Confirm.ask(
                f"\n[bold]Detected languages: {lang_list}. Is this correct?[/bold]",
                default=True
            )

            if lang_confirmed:
                confirmed_languages = languages_found
                self.console.print("[green]Languages confirmed[/green]")
            else:
                # Ask user to specify languages manually
                self.console.print("\n[yellow]Please specify languages manually[/yellow]")
                manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
        else:
            confirmed_languages = languages_found
    else:
        # No languages detected - ask user
        self.console.print("\n[yellow][!] No languages could be auto-detected[/yellow]")

        if interactive:
            manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)", default="en")
            confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
        else:
            confirmed_languages = {'en'}  # Default to English

    # Get model recommendations based on confirmed languages
    if confirmed_languages and interactive:
        # Consider long-document models if needed
        if analysis_results.get('user_prefers_long_models'):
            self.console.print("\n[bold]Recommended Long-Document Models:[/bold]")

            # Get multilingual long-doc models - MULTILINGUAL FIRST
            long_doc_recs = [
                {"model": "markussagen/xlm-roberta-longformer-base-4096", "reason": "Multilingual long-document support (100+ languages, 4096 tokens)"},
                {"model": "google/long-t5-local-base", "reason": "Multilingual T5 for long documents (4096+ tokens)"},
                {"model": "allenai/longformer-base-4096", "reason": "English-only, efficient for documents up to 4096 tokens"},
                {"model": "google/bigbird-roberta-base", "reason": "English-only, sparse attention for very long documents"}
            ]

            for i, rec in enumerate(long_doc_recs, 1):
                self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

            use_long = Confirm.ask("\n[bold]Use long-document model?[/bold]", default=True)

            if use_long:
                choice = IntPrompt.ask("Select model (1-4)", default=1)
                if 1 <= choice <= len(long_doc_recs):
                    model_to_use = long_doc_recs[choice - 1]['model']
                    self.console.print(f"[green]Selected: {model_to_use}[/green]")
                    return confirmed_languages, model_to_use

        # Get standard model recommendations
        recommendations = LanguageNormalizer.recommend_models(confirmed_languages, self.available_trainer_models)

        if recommendations:
            self.console.print(f"\n[bold]Recommended Models for Your Languages:[/bold]")
            for i, rec in enumerate(recommendations[:5], 1):
                self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

            # Store recommendations in bundle for later use, don't ask now
            self.console.print(f"\n[dim]ℹ Model selection will be done when choosing the training mode[/dim]")

            # Use first recommendation as default, but don't force it
            model_to_use = recommendations[0]['model'] if recommendations else "bert-base-uncased"

    return confirmed_languages, model_to_use


# ============================================================================
# INTEGRATION FUNCTION FOR ANNOTATOR FACTORY


# ============================================================================
# INTEGRATION FUNCTION FOR ANNOTATOR FACTORY
# ============================================================================

def integrate_training_arena_in_annotator_factory(
    cli_instance,
    output_file: Path,
    text_column: str,
    session_id: str,
    session_dirs: Optional[Dict[str, Path]] = None
) -> Dict[str, Any]:
    """
    Integration for Annotator Factory - starts at STEP 3b: Text Length Analysis.
    
    Skips:
    - STEP 1: Format selection (already known: llm-json)
    - STEP 2: File selection (uses output_file)
    - STEP 3: Dataset analysis (done automatically)
    - STEP 4: Column selection (uses text_column and "annotation")
    
    Starts at STEP 3b and continues with ALL remaining Training Arena steps.
    
    Args:
        cli_instance: AdvancedCLI instance with Training Arena methods
        output_file: Path to annotated CSV file
        text_column: Text column from annotation phase  
        session_id: Session ID from Annotator Factory
        session_dirs: Session directory structure from Annotator Factory
    
    Returns:
        Dict with training results and metadata
    """
    step_context = "factory"
    console = cli_instance.console
    from pathlib import Path

    # Pre-configured values from Annotator Factory
    csv_path = Path(output_file) if isinstance(output_file, str) else output_file
    selected_text_column = text_column
    selected_annotation_column = "annotation"
    
    # Load dataset
    import pandas as pd
    import json
    df = pd.read_csv(csv_path)
    annotated_mask_series = None  # local default; will be populated after analysis
    
    # Analyze dataset structure  
    detector = DataDetector()
    analysis = detector.analyze_file_intelligently(csv_path)
    all_columns = analysis.get('all_columns', [])
    
    text_fallbacks: List[str] = []
    for candidate in analysis.get('text_candidates', []):
        name = candidate.get('name')
        if name:
            text_fallbacks.append(name)
    text_fallbacks.append('text')

    selected_text_column = cli_instance._resolve_existing_column(
        df,
        selected_text_column,
        "text column",
        fallback_candidates=text_fallbacks
    )

    if selected_text_column not in df.columns:
        raise ValueError(f"Resolved text column '{selected_text_column}' not present in dataset columns {list(df.columns)}")

    # Display dataset confirmation (clean transition from Step 2/3 banner)
    console.print("[green]Annotations loaded successfully![/green]")
    console.print(f"  [cyan]File:[/cyan] {csv_path}")
    console.print(f"  [cyan]Text column:[/cyan] '{selected_text_column}'")
    console.print(f"  [cyan]Annotation column:[/cyan] '{selected_annotation_column}'")
    console.print(f"  [cyan]Rows:[/cyan] {len(df):,}\n")
    
    annotated_mask = pd.Series(False, index=df.index, dtype=bool)
    if selected_annotation_column in df.columns:
        annotation_series = df[selected_annotation_column]
        if annotation_series.dtype == object:
            annotation_str = annotation_series.fillna('').astype(str).str.strip()
            valid_strings = ~(annotation_str.isin({'', 'nan', '{}', '[]'}))
            annotation_mask = valid_strings
        else:
            annotation_mask = ~annotation_series.isna()
        annotated_mask |= annotation_mask

    for status_col in ["annotation_status_per_prompt", "annotation_status", "status"]:
        if status_col in df.columns:
            statuses = df[status_col].fillna('').astype(str).str.lower()
            status_mask = (
                statuses.str.contains('success')
                | statuses.str.contains('complete')
                | statuses.str.contains('done')
            )
            annotated_mask |= status_mask

    text_non_empty = df[selected_text_column].fillna('').astype(str).str.strip().ne('')
    annotated_mask &= text_non_empty

    annotated_count = int(annotated_mask.sum())
    total_text_rows = int(text_non_empty.sum())
    use_annotated_subset = annotated_count > 0 and annotated_count < total_text_rows

    if use_annotated_subset:
        console.print(
            f"[dim]Analytics focus on {annotated_count:,} annotated rows "
            f"(out of {total_text_rows:,} texts).[/dim]"
        )
    elif annotated_count == 0 and total_text_rows > 0:
        console.print("[yellow][!] No annotated rows detected; analytics will use all texts.[/yellow]")

    annotated_subset_df = df.loc[annotated_mask] if use_annotated_subset else None
    annotated_mask_series = annotated_mask if annotated_count > 0 else None

    # Import TrainingDataSessionManager for comprehensive logging
    from llm_tool.utils.training_data_utils import TrainingDataSessionManager
    from datetime import datetime

    # Determine the correct logs directory based on session context
    # For Annotator Factory, use the factory session directory directly (no nested training_session)
    if session_dirs and "session_root" in session_dirs:
        # Use the factory session root directly - NO NESTED training_session folder
        # Structure: logs/annotator_factory/factory_session_*/[training_data/, training_metrics/, ...]

        # Initialize session manager to use factory session directory directly
        session_manager = TrainingDataSessionManager(
            session_id=session_id,  # Use the factory session ID (without "training_" prefix)
            logs_base_dir=session_dirs["session_root"],  # Use factory session root directly
            use_custom_structure=True  # Use custom structure for Annotator Factory
        )

        # Store session ID for tracking (use the factory session ID as-is)
        actual_session_id = session_id  # Use the factory session ID
        training_session_id = session_id  # For compatibility with code that expects training_session_id
    else:
        # Fallback to default Training Arena structure (shouldn't happen in Annotator Factory)
        training_session_id = f"training_{session_id}"
        session_manager = TrainingDataSessionManager(
            session_id=training_session_id
        )
        actual_session_id = training_session_id

    # Store session attributes on cli_instance for use throughout training
    # CRITICAL: These are needed for _log_training_data_distributions to work
    cli_instance.current_session_id = actual_session_id
    cli_instance.current_session_manager = session_manager

    trained_models_map: Dict[str, str] = {}
    session_model_root: Optional[Path] = None
    if actual_session_id:
        try:
            session_model_root = (Path("models") / actual_session_id).resolve()
        except Exception:
            session_model_root = None

    seen_model_paths: Set[Path] = set()

    def _resolve_model_path(raw_value: Any) -> Optional[Path]:
        if raw_value is None:
            return None
        try:
            candidate = Path(str(raw_value)).expanduser()
        except Exception:
            return None
        if candidate.is_file():
            candidate = candidate.parent
        try:
            candidate = candidate.resolve()
        except Exception:
            candidate = candidate.absolute()
        if session_model_root and session_model_root.exists():
            try:
                candidate.relative_to(session_model_root)
            except ValueError:
                return None
        if not candidate.exists():
            return None
        if (candidate / "config.json").exists():
            return candidate
        for sub_dir in ("model", "best_model", "checkpoint-best"):
            option = candidate / sub_dir
            if (option / "config.json").exists():
                return option.resolve()
        try:
            config_path = next(candidate.glob("**/config.json"))
            return config_path.parent.resolve()
        except StopIteration:
            return None
        except Exception:
            return None

    def _merge_trained_models(source: Any, name_hint: Optional[str] = None) -> None:
        if source is None:
            return
        if isinstance(source, dict):
            for key, value in source.items():
                _merge_trained_models(value, str(key))
            return
        if isinstance(source, (list, tuple, set)):
            for item in source:
                _merge_trained_models(item, name_hint)
            return
        resolved = _resolve_model_path(source)
        if resolved is None:
            return
        if resolved in seen_model_paths:
            return
        seen_model_paths.add(resolved)
        model_name = name_hint or resolved.name
        if model_name in trained_models_map:
            existing_path = Path(trained_models_map[model_name])
            try:
                if existing_path.resolve() == resolved:
                    return
            except Exception:
                pass
            suffix = 2
            candidate_name = f"{model_name}_{suffix}"
            while candidate_name in trained_models_map:
                suffix += 1
                candidate_name = f"{model_name}_{suffix}"
            model_name = candidate_name
        trained_models_map[model_name] = str(resolved)

    # Initialize builder with session-based organization
    if session_dirs and "session_root" in session_dirs:
        # Use factory session root directly, with training_data subdirectory
        builder = TrainingDatasetBuilder(
            base_output_dir=session_dirs["session_root"],
            session_id=None,  # No additional session level
            use_training_data_subdir=True  # Files go in training_data/
        )
    else:
        builder = TrainingDatasetBuilder(
            base_output_dir=session_manager.logs_base_dir,
            session_id=training_session_id
        )
    
    # ========================================================================
    # STEP 3b TO END: Complete Training Arena Workflow (1247 lines)
    # ========================================================================
    
    # Step 3b: CRITICAL - Text Length Analysis (MUST be done AFTER text column selection)
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    text_length_stats = cli_instance.analyze_text_lengths(
        data_path=csv_path,
        text_column=selected_text_column,  # Use the ACTUAL selected column, not temp
        display_results=True,
        step_label=f"{resolve_step_label('text_length', 'STEP 5', context=step_context)}: Text Length Analysis",
        analysis_df=annotated_subset_df,
        total_rows_reference=total_text_rows if total_text_rows else None,
        subset_label="annotated rows" if use_annotated_subset else None,
    )
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

    # Store stats for later use in model selection (no user choice yet)
    # User will choose strategy in model selection step

    # Step 5: Language Detection and Text Analysis (using sophisticated universal system)
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    language_step = resolve_step_label("language_detection", "STEP 6", context=step_context)
    console.print(f"[bold cyan]  {language_step}:[/bold cyan] [bold white]Language Detection[/bold white]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[dim]Analyzing languages to recommend the best model.[/dim]\n")

    # Read CSV for analysis
    import pandas as pd
    import json
    df = pd.read_csv(csv_path)

    # Use the SAME sophisticated language detection as category-csv
    languages_found_in_column = set(analysis.get('languages_detected', {}).keys())
    confirmed_languages = set()
    lang_column = None
    language_distribution = {}  # Store exact language counts
    apply_auto_detection = True

    # Check if we have a language column with detected languages
    has_lang_column = bool(analysis.get('language_column_candidates'))

    if has_lang_column and languages_found_in_column:
        # Option 1: Language column exists - offer to use it or detect automatically
        console.print("[bold]Languages Found in Column:[/bold]")
        for lang, count in analysis['languages_detected'].items():
            console.print(f"  • {lang.upper()}: {count:,} rows")

        lang_column_candidate = analysis['language_column_candidates'][0]
        console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")

        use_lang_column = Confirm.ask(
            f"\n[bold]Use language column '{lang_column_candidate}'?[/bold]",
            default=True
        )

        if use_lang_column:
            confirmed_languages = languages_found_in_column
            lang_column = lang_column_candidate
            console.print(f"[green]Using language column: {lang_column}[/green]")
    else:
        # Option 2: No language column
        if not has_lang_column:
            console.print("[yellow]No language column detected[/yellow]")

    # Automatic language detection from text content
    if apply_auto_detection:
        console.print("\n[dim]Analyzing ALL texts to detect languages (this may take a moment)...[/dim]")

        try:
            from llm_tool.utils.language_detector import LanguageDetector

            if selected_text_column in df.columns:
                temp_df = df[df[selected_text_column].notna()].copy()

                if not temp_df.empty:
                    detector = LanguageDetector()
                    lang_counts = {}
                    detected_languages_per_text = []  # Store language for each text

                    if annotated_mask_series is not None:
                        detection_flags = annotated_mask_series.reindex(temp_df.index, fill_value=False)
                        annotated_detection_count = int(detection_flags.sum())
                        limit_detection = annotated_detection_count < len(temp_df)
                        if annotated_detection_count == 0:
                            detection_flags = pd.Series(True, index=temp_df.index)
                            annotated_detection_count = len(temp_df)
                            limit_detection = False
                    else:
                        detection_flags = pd.Series(True, index=temp_df.index)
                        annotated_detection_count = len(temp_df)
                        limit_detection = False

                    texts_list = temp_df[selected_text_column].astype(str).tolist()
                    flags_list = detection_flags.astype(bool).tolist()

                    if limit_detection:
                        console.print(
                            f"[dim]Analyzing {annotated_detection_count:,} annotated texts "
                            f"(out of {len(temp_df):,}).[/dim]"
                        )
                    else:
                        console.print(f"[dim]Analyzing {len(temp_df):,} texts...[/dim]")

                    from tqdm import tqdm
                    index_iterable = range(len(texts_list))
                    if HAS_RICH:
                        index_iterable = tqdm(index_iterable, desc="Detecting languages", disable=not HAS_RICH)

                    for idx in index_iterable:
                        text = texts_list[idx]
                        analyze_text = flags_list[idx]

                        if not analyze_text:
                            detected_languages_per_text.append(None)
                            continue

                        stripped = text.strip()
                        if not stripped or len(stripped) <= 10:
                            detected_languages_per_text.append(None)
                            continue

                        try:
                            detected = detector.detect(stripped)
                            if detected:
                                if isinstance(detected, dict):
                                    lang = detected.get('language')
                                    confidence = detected.get('confidence', 0)
                                    if lang and confidence >= 0.7:
                                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                                        detected_languages_per_text.append(lang)
                                    else:
                                        detected_languages_per_text.append(None)
                                elif isinstance(detected, str):
                                    lang_counts[detected] = lang_counts.get(detected, 0) + 1
                                    detected_languages_per_text.append(detected)
                                else:
                                    detected_languages_per_text.append(None)
                            else:
                                detected_languages_per_text.append(None)
                        except Exception as e:
                            logger.debug(f"Language detection failed for text: {e}")
                            detected_languages_per_text.append(None)

                    if lang_counts:
                        # Store exact distribution
                        language_distribution = lang_counts
                        total = sum(lang_counts.values())

                        console.print(f"\n[bold]Languages Detected from Content ({total:,} texts analyzed):[/bold]")

                        # Create detailed table
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

                        console.print(lang_table)

                        # Detect low-percentage languages (likely detection errors)
                        LOW_PERCENTAGE_THRESHOLD = 1.0  # Languages with < 1% are considered low
                        majority_languages = {}  # Languages above threshold
                        minority_languages = {}  # Languages below threshold (likely errors)

                        for lang, count in lang_counts.items():
                            percentage = (count / total * 100) if total > 0 else 0
                            if percentage >= LOW_PERCENTAGE_THRESHOLD:
                                majority_languages[lang] = count
                            else:
                                minority_languages[lang] = count

                        confirmed_languages = set(lang_counts.keys())

                        # Handle low-percentage languages if detected
                        if minority_languages:
                            console.print(f"\n[yellow][!] Warning: {len(minority_languages)} language(s) detected with very low percentage (< {LOW_PERCENTAGE_THRESHOLD}%):[/yellow]")
                            for lang, count in sorted(minority_languages.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total * 100)
                                console.print(f"  • {lang.upper()}: {count} texts ({percentage:.2f}%)")

                            console.print("\n[dim]These are likely detection errors. You have options:[/dim]")
                            console.print("  [cyan]1. exclude[/cyan] - Exclude ALL low-percentage languages from training")
                            console.print("  [cyan]2. keep[/cyan] - Keep ALL detected languages (not recommended)")
                            console.print("  [cyan]3. select[/cyan] - Manually select which languages to keep")
                            console.print("  [cyan]4. correct[/cyan] - Force ALL minority languages to a single language (quick fix)")

                            minority_action = Prompt.ask(
                                "\n[bold yellow]How to handle low-percentage languages?[/bold yellow]",
                                choices=["exclude", "keep", "select", "correct"],
                                default="correct"
                            )

                            if minority_action == "correct":
                                # Quick correction: force all minority languages to one language
                                console.print("\n[bold cyan]Quick Language Correction[/bold cyan]\n")

                                # Show available languages
                                all_supported_langs = [
                                    'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja',
                                    'ar', 'pl', 'tr', 'ko', 'hi', 'sv', 'no', 'da', 'fi', 'cs',
                                    'el', 'he', 'ro', 'uk', 'bg', 'hr', 'vi', 'th', 'id', 'fa'
                                ]

                                # Suggest the majority language
                                majority_lang = max(majority_languages.items(), key=lambda x: x[1])[0] if majority_languages else 'en'

                                console.print(f"[bold]Available languages:[/bold]")
                                console.print(f"  • Majority language detected: [green]{majority_lang.upper()}[/green] ({majority_languages.get(majority_lang, 0)} texts)")
                                console.print(f"  • All supported: {', '.join([l.upper() for l in all_supported_langs])}")

                                correction_target = Prompt.ask(
                                    f"\n[bold yellow]Force ALL minority languages to which language?[/bold yellow]",
                                    default=majority_lang
                                ).lower().strip()

                                if correction_target not in all_supported_langs:
                                    console.print(f"[yellow]Warning: '{correction_target}' not in standard list, but will be used anyway[/yellow]")

                                # Update language_distribution and confirmed_languages
                                total_corrected = sum(minority_languages.values())

                                # Move all minority counts to the target language
                                for minority_lang in minority_languages.keys():
                                    if minority_lang in language_distribution:
                                        del language_distribution[minority_lang]

                                # Add corrected texts to target language
                                if correction_target in language_distribution:
                                    language_distribution[correction_target] += total_corrected
                                else:
                                    language_distribution[correction_target] = total_corrected

                                # Update confirmed languages
                                confirmed_languages = set([correction_target] + list(majority_languages.keys()))

                                # CRITICAL FIX: Update detected_languages_per_text with corrections
                                if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                    for i in range(len(detected_languages_per_text)):
                                        if detected_languages_per_text[i] in minority_languages:
                                            detected_languages_per_text[i] = correction_target

                                console.print(f"\n[green]✓ Corrected {total_corrected} texts from {len(minority_languages)} languages to {correction_target.upper()}[/green]")

                                # Display updated distribution
                                update_table = Table(title="Updated Language Distribution", border_style="green", expand=True)
                                update_table.add_column("Language", style="cyan", justify="center")
                                update_table.add_column("Count", justify="right")
                                update_table.add_column("Percentage", justify="right")

                                new_total = sum(language_distribution.values())
                                for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
                                    if count > 0:  # Only show non-zero counts
                                        percentage = (count / new_total) * 100 if new_total > 0 else 0
                                        update_table.add_row(lang.upper(), f"{count:,}", f"{percentage:.1f}%")

                                console.print(update_table)

                            elif minority_action == "exclude":
                                # Exclude low-percentage languages
                                for lang in minority_languages.keys():
                                    language_distribution[lang] = 0  # Mark as excluded

                                # CRITICAL FIX: Mark excluded language texts as None
                                if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                    for i in range(len(detected_languages_per_text)):
                                        if detected_languages_per_text[i] in minority_languages:
                                            detected_languages_per_text[i] = None

                                confirmed_languages = set(majority_languages.keys())
                                excluded_count = sum(minority_languages.values())
                                console.print(f"\n[yellow]✗ Excluded {excluded_count} texts from {len(minority_languages)} low-percentage language(s)[/yellow]")
                                console.print(f"[green]Final languages: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                            elif minority_action == "keep":
                                console.print("[yellow][!] Keeping all detected languages (including low-percentage ones)[/yellow]")

                            elif minority_action == "select":
                                # Manual selection of languages to keep
                                console.print("\n[bold cyan]Language Selection:[/bold cyan]")
                                console.print(f"[dim]Select which languages to keep for training (from all {len(lang_counts)} detected)[/dim]\n")

                                # Show all languages sorted by count
                                console.print("[bold]All Detected Languages:[/bold]")
                                for i, (lang, count) in enumerate(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True), 1):
                                    percentage = (count / total * 100)
                                    status = "[green]majority[/green]" if lang in majority_languages else "[yellow][!] minority[/yellow]"
                                    console.print(f"  {i:2d}. {lang.upper():5s} - {count:6,} texts ({percentage:5.2f}%) {status}")

                                console.print("\n[bold yellow]Select languages to KEEP:[/bold yellow]")
                                console.print("[dim]Enter language codes separated by commas (e.g., 'fr,en,de')[/dim]")
                                console.print("[dim]Press Enter without typing to keep ALL languages[/dim]")

                                selected_langs = Prompt.ask("\n[bold]Languages to keep[/bold]", default="")

                                if selected_langs.strip():
                                    # User selected specific languages
                                    selected_set = set([l.strip().lower() for l in selected_langs.split(',') if l.strip()])

                                    # Validate that selected languages exist
                                    invalid_langs = selected_set - set(lang_counts.keys())
                                    if invalid_langs:
                                        console.print(f"[yellow][!] Warning: These languages were not detected: {', '.join(invalid_langs)}[/yellow]")
                                        selected_set = selected_set - invalid_langs

                                    # Exclude non-selected languages
                                    for lang in lang_counts.keys():
                                        if lang not in selected_set:
                                            language_distribution[lang] = 0  # Mark as excluded

                                    # CRITICAL FIX: Mark non-selected language texts as None
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] and detected_languages_per_text[i] not in selected_set:
                                                detected_languages_per_text[i] = None

                                    confirmed_languages = selected_set
                                    kept_count = sum([lang_counts[lang] for lang in selected_set])
                                    excluded_count = total - kept_count

                                    console.print(f"\n[green]✓ Kept {len(selected_set)} language(s): {', '.join([l.upper() for l in sorted(selected_set)])}[/green]")
                                    console.print(f"[dim]  → {kept_count:,} texts kept, {excluded_count:,} texts excluded[/dim]")
                                else:
                                    # User pressed Enter - keep all
                                    console.print("[green]Keeping all detected languages[/green]")

                        # Final confirmation (allow override even after selection)
                        lang_list = ', '.join([l.upper() for l in sorted(confirmed_languages)])
                        lang_confirmed = Confirm.ask(
                            f"\n[bold]Final languages: {lang_list}. Is this correct?[/bold]",
                            default=True
                        )

                        if not lang_confirmed:
                            console.print("\n[yellow]Override with manual selection[/yellow]")

                            # Check for existing language columns
                            potential_lang_cols = [col for col in df.columns if col.lower() in ('lang', 'language', 'langue', 'idioma', 'sprache')]
                            use_col = False

                            if potential_lang_cols:
                                console.print(f"[cyan]Found potential language columns: {', '.join(potential_lang_cols)}[/cyan]")
                                use_col = Confirm.ask("Use an existing column for languages?", default=True)

                                if use_col:
                                    if len(potential_lang_cols) == 1:
                                        lang_column = potential_lang_cols[0]
                                    else:
                                        lang_column = Prompt.ask("Which column?", choices=potential_lang_cols, default=potential_lang_cols[0])

                                    # Extract languages from the column
                                    lang_series = df[lang_column].dropna().astype(str).str.strip().str.upper()
                                    unique_langs = set(lang_series.unique())
                                    confirmed_languages = set([l.lower() for l in unique_langs if l])

                                    # Update detected_languages_per_text from the column
                                    text_mask = df[selected_text_column].notna()
                                    detected_languages_per_text = []
                                    for val in df.loc[text_mask, lang_column]:
                                        if pd.notna(val) and str(val).strip():
                                            detected_languages_per_text.append(str(val).strip().lower())
                                        else:
                                            detected_languages_per_text.append(None)

                                    # Update language distribution
                                    language_distribution = {}
                                    for lang in confirmed_languages:
                                        count = lang_series.str.lower().eq(lang).sum()
                                        language_distribution[lang] = count

                                    console.print(f"[green]Using column '{lang_column}': {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                            if not use_col:
                                manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

                                # Update distribution to exclude non-selected languages
                                for lang in lang_counts.keys():
                                    if lang not in confirmed_languages:
                                        language_distribution[lang] = 0

                                # CRITICAL FIX: Mark non-confirmed language texts as None
                                if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                    for i in range(len(detected_languages_per_text)):
                                        if detected_languages_per_text[i] and detected_languages_per_text[i] not in confirmed_languages:
                                            detected_languages_per_text[i] = None

                                console.print(f"[green]Manual override: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                        else:
                            console.print("[green]Languages confirmed from content analysis[/green]")

                        # CRITICAL FIX: Add detected language column to DataFrame and save
                        if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                            # Create a temporary DataFrame for non-null texts
                            temp_df = df[df[selected_text_column].notna()].copy()

                            # Ensure same length
                            if len(detected_languages_per_text) == len(temp_df):
                                if lang_column is None:
                                    temp_df['language'] = detected_languages_per_text

                                    # Map detected languages to the full DataFrame
                                    df['language'] = None
                                    df.loc[df[selected_text_column].notna(), 'language'] = detected_languages_per_text

                                    # Set lang_column to use this new column
                                    lang_column = 'language'

                                    # Save updated DataFrame back to CSV
                                    df.to_csv(csv_path, index=False)
                                    console.print(f"[dim]Added 'language' column to dataset ({len([l for l in detected_languages_per_text if l])} texts with detected language)[/dim]")
                                else:
                                    console.print("[dim]Auto-detected languages available; existing language column preserved.[/dim]")
                    else:
                        # Fallback: ask user
                        console.print("[yellow]Could not detect languages automatically[/yellow]")
                        manual_langs = Prompt.ask("Expected language codes (e.g., en,fr,de)", default="")
                        if manual_langs.strip():
                            confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
                else:
                    console.print("[yellow]Not enough text samples for language detection[/yellow]")
                    manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                    if manual_langs.strip():
                        confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

        except Exception as e:
            logger.debug(f"Language detection from content failed: {e}")
            console.print("[yellow]Automatic detection failed. Please specify manually[/yellow]")
            manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
            if manual_langs.strip():
                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

    # Model selection will be done later when training mode is selected
    # Store languages for later use

    # Step 6: Annotation Data Preview
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    annotation_step = resolve_step_label("annotation_preview", "STEP 8", context=step_context)
    console.print(f"[bold cyan]  {annotation_step}:[/bold cyan] [bold white]Annotation Data Preview[/bold white]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[dim]Analyzing all annotation data to show you what labels/categories will be trained...[/dim]\n")

    # df already loaded above for language detection

    def _normalize_preview_value(raw_value: Any) -> Optional[str]:
        """Normalize annotation values for display (remove stray commas/quotes)."""
        if raw_value is None:
            return None
        value_str = str(raw_value).strip()
        if not value_str:
            return None
        # Remove wrapping quotes
        if (value_str.startswith("'") and value_str.endswith("'")) or (value_str.startswith("\"") and value_str.endswith("\"")):
            value_str = value_str[1:-1].strip()
        # Drop trailing commas introduced by CSV artifacts
        while value_str.endswith(","):
            value_str = value_str[:-1].rstrip()
        return value_str if value_str else None

    all_keys_values = {}  # {key: set_of_unique_values}
    value_counts_by_key = {}  # {key: {value: occurrence_count}}
    total_samples = 0
    malformed_count = 0

    for idx, row in df.iterrows():
        annotation_val = row.get(selected_annotation_column)
        if pd.isna(annotation_val) or annotation_val == '':
            continue

        total_samples += 1
        try:
            if isinstance(annotation_val, str):
                # Try standard JSON first
                try:
                    annotation_dict = json.loads(annotation_val)
                except json.JSONDecodeError:
                    # Try Python literal (handles single quotes with escapes)
                    import ast
                    annotation_dict = ast.literal_eval(annotation_val)
            elif isinstance(annotation_val, dict):
                annotation_dict = annotation_val
            else:
                continue

            # Extract keys and values
            for key, value in annotation_dict.items():
                if key not in all_keys_values:
                    all_keys_values[key] = set()
                if key not in value_counts_by_key:
                    value_counts_by_key[key] = {}
                counts_for_key = value_counts_by_key[key]

                if isinstance(value, list):
                    for v in value:
                        normalized = _normalize_preview_value(v)
                        if normalized:
                            all_keys_values[key].add(normalized)
                            counts_for_key[normalized] = counts_for_key.get(normalized, 0) + 1
                else:
                    normalized = _normalize_preview_value(value)
                    if normalized:
                        all_keys_values[key].add(normalized)
                        counts_for_key[normalized] = counts_for_key.get(normalized, 0) + 1

        except (json.JSONDecodeError, AttributeError, TypeError, ValueError, SyntaxError) as e:
            malformed_count += 1
            continue

    # Display comprehensive preview with Rich table
    if all_keys_values:
        console.print(f"\n[bold cyan]Complete Annotation Data Preview[/bold cyan]")
        console.print(f"[dim]Analyzed {total_samples} samples ({malformed_count} malformed)[/dim]\n")

        preview_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
        preview_table.add_column("Key", style="yellow bold", no_wrap=True)
        preview_table.add_column("Unique Values", style="white", no_wrap=True, justify="center")
        preview_table.add_column("Sample Values", style="green", overflow="fold", ratio=1)

        # Determine language summary for caption
        language_display = None
        if 'language' in df.columns:
            language_values = {
                str(lang).strip().upper()
                for lang in df['language']
                if pd.notna(lang) and str(lang).strip()
            }
            if language_values:
                language_display = ", ".join(sorted(language_values))
        if not language_display:
            # Fallback to confirmed languages if available in scope
            if 'confirmed_languages' in locals() and confirmed_languages:
                language_display = ", ".join(sorted(lang.upper() for lang in confirmed_languages))

        if language_display:
            preview_table.caption = f"Languages: {language_display}"

        for key in sorted(all_keys_values.keys()):
            values_set = all_keys_values[key]
            num_values = len(values_set)

            # Show first 10 values as sample
            sample_values = sorted(values_set)[:10]
            sample_str = ', '.join([f"'{v}'" for v in sample_values])
            if num_values > 10:
                sample_str += f" ... (+{num_values - 10} more)"

            preview_table.add_row(
                key,
                str(num_values),
                sample_str
            )

        console.print(preview_table)
        console.print()

        # Show selection options
        console.print("[bold]Training Options:[/bold]")
        console.print("  [dim]• You can choose to train on [cyan]ALL[/cyan] keys/values[/dim]")
        console.print("  [dim]• Or select [cyan]specific keys[/cyan] to train (asked later)[/dim]")
        console.print("  [dim]• Or select [cyan]specific values[/cyan] for each key (asked later)[/dim]\n")
    else:
        console.print("[yellow][!] No valid annotation data found[/yellow]\n")

    # Step 6.5: Value Filtering (Optional) - CRITICAL FOR DATA QUALITY
    if all_keys_values:
        console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        value_filter_step = resolve_step_label("value_filter", "STEP 10", context=step_context)
        console.print(f"[bold cyan]  {value_filter_step}:[/bold cyan] [bold white]Value Filtering (Optional)[/bold white]")
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print("[dim]You can exclude specific values from your training data.[/dim]")
        console.print("[dim]   For example: Remove 'null' values, or exclude rare categories.[/dim]\n")

        filter_values = Confirm.ask(
            "[bold yellow]Do you want to exclude any specific values from training?[/bold yellow]",
            default=False
        )

        excluded_values = {}  # {key: [list_of_excluded_values]}
        rows_to_remove = []  # List of indices to remove from df

        if filter_values:
            console.print("\n[bold]Value Filtering Configuration[/bold]\n")

            # Ask for each key
            for key in sorted(all_keys_values.keys()):
                values_set = all_keys_values[key]
                num_values = len(values_set)

                if num_values == 0:
                    continue

                # Display key and its values
                console.print(f"\n[cyan]Key:[/cyan] [bold]{key}[/bold] ({num_values} values)")

                # Create table for values with counts
                values_table = Table(show_header=True, header_style="bold magenta", border_style="dim", box=box.SIMPLE, expand=True)
                values_table.add_column("Value", style="yellow", ratio=1, overflow="fold")
                values_table.add_column("Count", style="white", no_wrap=True, justify="right")
                values_table.add_column("Percentage", style="green", no_wrap=True, justify="right")

                # Count occurrences of each value in the dataset
                value_counts = {}
                for idx, row in df.iterrows():
                    annotation_val = row.get(selected_annotation_column)
                    if pd.isna(annotation_val) or annotation_val == '':
                        continue

                    try:
                        if isinstance(annotation_val, str):
                            try:
                                annotation_dict = json.loads(annotation_val)
                            except json.JSONDecodeError:
                                import ast
                                annotation_dict = ast.literal_eval(annotation_val)
                        elif isinstance(annotation_val, dict):
                            annotation_dict = annotation_val
                        else:
                            continue

                        if key in annotation_dict:
                            val = annotation_dict[key]
                            if isinstance(val, list):
                                for v in val:
                                    if v is not None and v != '':
                                        v_str = str(v)
                                        value_counts[v_str] = value_counts.get(v_str, 0) + 1
                            elif val is not None and val != '':
                                v_str = str(val)
                                value_counts[v_str] = value_counts.get(v_str, 0) + 1
                    except:
                        continue

                # Display values with counts
                sorted_values = sorted(values_set, key=lambda v: value_counts.get(v, 0), reverse=True)
                for val in sorted_values:
                    count = value_counts.get(val, 0)
                    percentage = (count / total_samples * 100) if total_samples > 0 else 0
                    values_table.add_row(
                        val,
                        str(count),
                        f"{percentage:.1f}%"
                    )

                console.print(values_table)

                # Capture counts for metadata/analytics
                value_counts_by_key[key] = {
                    val: int(value_counts.get(val, 0))
                    for val in values_set
                }

                # Ask if user wants to exclude any values for this key
                exclude_for_key = Confirm.ask(
                    f"[bold yellow]Exclude any values from '{key}'?[/bold yellow]",
                    default=False
                )

                if exclude_for_key:
                    console.print(f"[dim]Enter values to exclude (comma-separated), or type 'cancel' to skip[/dim]")
                    exclude_input = Prompt.ask(
                        f"[yellow]Values to exclude from '{key}'[/yellow]",
                        default=""
                    )

                    if exclude_input.lower() != 'cancel' and exclude_input.strip():
                        excluded_list = [v.strip() for v in exclude_input.split(',') if v.strip()]
                        # Validate that excluded values exist
                        valid_excluded = [v for v in excluded_list if v in values_set]
                        invalid_excluded = [v for v in excluded_list if v not in values_set]

                        if invalid_excluded:
                            console.print(f"[yellow][!] Warning: These values don't exist: {', '.join(invalid_excluded)}[/yellow]")

                        if valid_excluded:
                            excluded_values[key] = valid_excluded
                            console.print(f"[green]Will exclude: {', '.join(valid_excluded)}[/green]")

            # Now filter the DataFrame based on excluded values
            if excluded_values:
                console.print(f"\n[bold cyan]Filtering labels from dataset...[/bold cyan]")
                console.print(f"[dim]Note: Removing excluded labels from samples, not the samples themselves.[/dim]\n")

                original_count = len(df)
                labels_removed_count = 0
                samples_modified = 0

                # Filter labels from each row (NOT remove rows)
                for idx, row in df.iterrows():
                    annotation_val = row.get(selected_annotation_column)
                    if pd.isna(annotation_val) or annotation_val == '':
                        continue

                    try:
                        # Parse annotation
                        if isinstance(annotation_val, str):
                            try:
                                annotation_dict = json.loads(annotation_val)
                            except json.JSONDecodeError:
                                import ast
                                annotation_dict = ast.literal_eval(annotation_val)
                        elif isinstance(annotation_val, dict):
                            annotation_dict = annotation_dict.copy()
                        else:
                            continue

                        # Remove excluded values from annotation (NOT the row)
                        modified = False
                        for key, excluded_vals in excluded_values.items():
                            if key in annotation_dict:
                                val = annotation_dict[key]

                                if isinstance(val, list):
                                    # Remove excluded values from list
                                    original_list = val.copy()
                                    val = [v for v in val if str(v) not in excluded_vals]
                                    if len(val) != len(original_list):
                                        modified = True
                                        labels_removed_count += len(original_list) - len(val)
                                    annotation_dict[key] = val if val else None

                                elif val is not None and str(val) in excluded_vals:
                                    # Replace excluded value with None
                                    annotation_dict[key] = None
                                    modified = True
                                    labels_removed_count += 1

                        # Update the annotation in the DataFrame
                        if modified:
                            samples_modified += 1
                            # Convert back to JSON string if it was originally a string
                            if isinstance(row[selected_annotation_column], str):
                                df.at[idx, selected_annotation_column] = json.dumps(annotation_dict)
                            else:
                                df.at[idx, selected_annotation_column] = annotation_dict

                    except Exception as e:
                        logger.warning(f"Error filtering row {idx}: {e}")
                        continue

                # IMPORTANT: Do NOT remove samples even if they have no valid labels remaining
                # Reason: Label filtering happens BEFORE key selection for training.
                # A sample with all null/None labels might still be useful when training
                # on specific keys later (e.g., user might select keys where null is valid).
                # The training code will naturally skip samples without valid labels for selected keys.
                removed_count = 0
                filtered_count = len(df)

                console.print(f"[green]Label filtering complete:[/green]")
                console.print(f"  • [cyan]Samples kept:[/cyan] {original_count} → {filtered_count}")
                console.print(f"  • [cyan]Samples modified:[/cyan] {samples_modified}")
                console.print(f"  • [cyan]Labels removed:[/cyan] {labels_removed_count}")
                if removed_count > 0:
                    console.print(f"  • [yellow]Samples removed (empty):[/yellow] {removed_count}")
                console.print()

                # Recalculate all_keys_values with filtered data
                all_keys_values = {}
                value_counts_by_key = {}
                total_samples = 0
                malformed_count = 0

                for idx, row in df.iterrows():
                    annotation_val = row.get(selected_annotation_column)
                    if pd.isna(annotation_val) or annotation_val == '':
                        continue

                    total_samples += 1
                    try:
                        if isinstance(annotation_val, str):
                            try:
                                annotation_dict = json.loads(annotation_val)
                            except json.JSONDecodeError:
                                import ast
                                annotation_dict = ast.literal_eval(annotation_val)
                        elif isinstance(annotation_val, dict):
                            annotation_dict = annotation_val
                        else:
                            continue

                        # Extract keys and values (excluding the filtered ones)
                        for key, value in annotation_dict.items():
                            if key not in all_keys_values:
                                all_keys_values[key] = set()
                            if key not in value_counts_by_key:
                                value_counts_by_key[key] = {}
                            counts_for_key = value_counts_by_key[key]

                            if isinstance(value, dict):
                                if 'detected' in value:
                                    detected = str(value['detected']).lower().strip()
                                    all_keys_values[key].add(detected)
                                    counts_for_key[detected] = counts_for_key.get(detected, 0) + 1
                                else:
                                    for sub_key, sub_val in value.items():
                                        sub_str = str(sub_val)
                                        all_keys_values[key].add(sub_str)
                                        counts_for_key[sub_str] = counts_for_key.get(sub_str, 0) + 1
                            elif isinstance(value, list):
                                for v in value:
                                    if v is not None and v != '':
                                        v_str = str(v)
                                        all_keys_values[key].add(v_str)
                                        counts_for_key[v_str] = counts_for_key.get(v_str, 0) + 1
                            elif value is not None and value != '':
                                v_str = str(value)
                                all_keys_values[key].add(v_str)
                                counts_for_key[v_str] = counts_for_key.get(v_str, 0) + 1

                    except (json.JSONDecodeError, AttributeError, TypeError, ValueError, SyntaxError) as e:
                        malformed_count += 1
                        continue

                # Display updated summary
                console.print("[bold]Updated Data Summary:[/bold]")
                summary_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
                summary_table.add_column("Key", style="yellow bold", no_wrap=True)
                summary_table.add_column("Values (After Filtering)", style="white", ratio=1, overflow="fold")

                for key in sorted(all_keys_values.keys()):
                    values_set = all_keys_values[key]
                    num_values = len(values_set)
                    sample_str = ', '.join([f"'{v}'" for v in sorted(values_set)[:5]])
                    if num_values > 5:
                        sample_str += f" ... (+{num_values - 5} more)"

                    # Show what was excluded
                    if key in excluded_values:
                        excluded_str = f"[dim red](excluded: {', '.join(excluded_values[key])})[/dim red]"
                        summary_table.add_row(
                            f"{key}\n{excluded_str}",
                            f"[green]{num_values} values[/green]: {sample_str}"
                        )
                    else:
                        summary_table.add_row(
                            key,
                            f"{num_values} values: {sample_str}"
                        )

                console.print(summary_table)
                console.print()
        else:
            console.print("[dim]No values excluded - using all data[/dim]\n")

    # Step 7: Training Strategy Selection (SIMPLIFIED)
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    strategy_step = resolve_step_label("training_strategy", "STEP 11", context=step_context)
    console.print(f"[bold cyan]  {strategy_step}:[/bold cyan] [bold white]Training Strategy Selection[/bold white]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

    # Extract annotation keys and values from data
    annotation_keys_found = analysis.get('annotation_keys_found', set())
    sample_annotation = analysis.get('sample_data', {}).get(selected_annotation_column, [])
    real_example_data = None

    if sample_annotation and len(sample_annotation) > 0:
        first_sample = sample_annotation[0]
        try:
            if isinstance(first_sample, str):
                real_example_data = json.loads(first_sample)
            elif isinstance(first_sample, dict):
                real_example_data = first_sample
        except:
            pass

    # Show sample annotation for context
    if real_example_data:
        console.print("[bold]Example annotation from your data:[/bold]")
        example_str = json.dumps(real_example_data, ensure_ascii=False, indent=2)
        console.print(f"[dim]{example_str}[/dim]\n")

    # Initialize
    detected_keys = []
    annotation_keys = None
    mode = "single-label"  # Will be derived from choice
    training_approach = "multi-class"  # Default

    # Step 6a: Show all annotation keys and their values
    if all_keys_values:
        detected_keys = sorted(all_keys_values.keys())
        console.print(f"[bold]Annotation Keys Detected in Your Data:[/bold]\n")

        # Show all keys and their values
        for key in detected_keys:
            num_values = len(all_keys_values[key])
            values_preview = ', '.join([f"'{v}'" for v in sorted(all_keys_values[key])[:5]])
            if num_values > 5:
                values_preview += f" ... (+{num_values-5} more)"
            console.print(f"  • [cyan]{key}[/cyan] ({num_values} values): {values_preview}")

        console.print("\n[dim]Options:[/dim]")
        console.print(f"  • [cyan]Leave blank[/cyan] → Use ALL {len(detected_keys)} keys with ALL their values")
        console.print(f"  • [cyan]Enter specific keys[/cyan] → Use only selected keys with ALL their values")
        if detected_keys:
            console.print(f"    Example: '{detected_keys[0]}' → Use only {detected_keys[0]} key\n")
    elif analysis.get('annotation_keys_found'):
        detected_keys = sorted(analysis['annotation_keys_found'])
        console.print(f"\n[green]✓ Detected keys: {', '.join(detected_keys)}[/green]")
        console.print("[dim]Leave blank to use all keys, or specify which ones to include[/dim]\n")

    # Step 6b: Ask which keys to include
    keys_input = Prompt.ask("[bold yellow]Annotation keys to include[/bold yellow] (comma separated, or BLANK for ALL)", default="")
    annotation_keys = [key.strip() for key in keys_input.split(",") if key.strip()] or None

    # Step 6c: Ask multi-class vs one-vs-all (ALWAYS, not just for single key)
    # Determine which keys will be trained
    keys_to_train = annotation_keys if annotation_keys else detected_keys

    # Validate and auto-correct invalid keys with intelligent suggestions
    invalid_keys = [key for key in keys_to_train if key not in all_keys_values]
    if invalid_keys:
        from difflib import get_close_matches

        console.print(f"\n[bold yellow][!]  Some keys need correction:[/bold yellow]")

        # Auto-correct using fuzzy matching
        corrected_keys = []
        for key in keys_to_train:
            if key in all_keys_values:
                corrected_keys.append(key)
            else:
                # Find best match using fuzzy matching
                matches = get_close_matches(key, all_keys_values.keys(), n=1, cutoff=0.6)
                if matches:
                    suggestion = matches[0]
                    console.print(f"  • [red]'{key}'[/red] → [green]'{suggestion}'[/green] [dim](auto-corrected)[/dim]")
                    corrected_keys.append(suggestion)
                else:
                    console.print(f"  • [red]'{key}'[/red] [dim](no match found, will be skipped)[/dim]")

        # Show available keys for reference
        if len(corrected_keys) < len(keys_to_train):
            console.print(f"\n[bold cyan]Tip: Available keys:[/bold cyan]")
            for key in sorted(all_keys_values.keys()):
                console.print(f"  • [green]{key}[/green]")

        # Ask user to confirm corrections
        if corrected_keys:
            console.print(f"\n[green]✓ Corrected selection:[/green] {', '.join(corrected_keys)}")
            confirm = Confirm.ask("[bold yellow]Use these corrected keys?[/bold yellow]", default=True)
            if confirm:
                keys_to_train = corrected_keys
                annotation_keys = corrected_keys
            else:
                console.print("[yellow]Training cancelled. Please try again with correct key names.[/yellow]")
                return None
        else:
            console.print("[red][FAIL] No valid keys found after correction. Training cancelled.[/red]")
            return None

    # Calculate total number of models for each approach
    total_values_count = 0
    for key in keys_to_train:
        if key in all_keys_values:
            total_values_count += len(all_keys_values[key])

    num_keys = len(keys_to_train)

    # ALWAYS ask the training approach question, even for binary classification
    # User may want one-vs-all even with 2 values
    if True:  # Always ask
        console.print(f"\n[bold cyan]Training Approach[/bold cyan]\n")

        if annotation_keys and len(annotation_keys) == 1:
            # Single key selected
            selected_key = annotation_keys[0]
            num_unique_values = len(all_keys_values[selected_key])
            values_list = sorted(all_keys_values[selected_key])
            values_str = ', '.join([f"'{v}'" for v in values_list[:5]])
            if num_unique_values > 5:
                values_str += f" ... (+{num_unique_values-5} more)"

            console.print(f"[bold]Selected:[/bold] '{selected_key}' ({num_unique_values} values)")
            console.print(f"[dim]Values: {values_str}[/dim]\n")
        else:
            # Multiple keys or ALL
            console.print(f"[bold]Selected:[/bold] {'ALL' if not annotation_keys else len(annotation_keys)} keys ({num_keys} total)")
            console.print(f"[dim]Total unique values across all keys: {total_values_count}[/dim]\n")

        # Create comparison table
        approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
        approach_table.add_column("Approach", style="cyan bold", no_wrap=True)
        approach_table.add_column("What It Does", style="white", overflow="fold", ratio=1)

        if annotation_keys and len(annotation_keys) == 1:
            # Single key - simple explanation
            selected_key = annotation_keys[0]
            num_unique_values = len(all_keys_values[selected_key])
            values_list = sorted(all_keys_values[selected_key])

            approach_table.add_row(
                "multi-class",
                f"Trains ONE model for '{selected_key}'\n\n"
                f"• Chooses between all {num_unique_values} values\n"
                f"• Example: '{values_list[0]}' vs '{values_list[1]}' vs ...\n"
                f"• Predicts exactly ONE value per text\n"
                f"• [bold green]Total: 1 model[/bold green]\n\n"
                "[bold cyan]Best for:[/bold cyan] Mutually exclusive categories"
            )
            approach_table.add_row(
                "one-vs-all",
                f"Trains {num_unique_values} binary models for '{selected_key}'\n\n"
                f"• Model 1: '{values_list[0]}' vs NOT '{values_list[0]}'\n"
                f"• Model 2: '{values_list[1]}' vs NOT '{values_list[1]}'\n"
                f"• ... (one model per value)\n"
                f"• [bold yellow]Total: {num_unique_values} models[/bold yellow]\n\n"
                "[bold cyan]Best for:[/bold cyan] Imbalanced data"
            )
            approach_table.add_row(
                "multi-label",
                f"Trains ONE model with MULTIPLE labels per text\n\n"
                f"• Uses sigmoid activation (independent probabilities)\n"
                f"• Each text can have 0, 1, or many labels simultaneously\n"
                f"• Threshold-based selection (default: 0.5)\n"
                f"• [bold magenta]Total: 1 model[/bold magenta]\n\n"
                "[bold cyan]Best for:[/bold cyan] Texts with overlapping categories"
            )
        else:
            # Multiple keys or ALL - offer hybrid and custom modes
            # Analyze keys to determine hybrid strategy
            keys_small = []  # ≤5 values
            keys_large = []  # >5 values
            for key in keys_to_train:
                num_values = len(all_keys_values[key])
                if num_values <= 5:
                    keys_small.append((key, num_values))
                else:
                    keys_large.append((key, num_values))

            hybrid_multiclass_count = len(keys_small)
            hybrid_onevsall_count = sum(num_vals for _, num_vals in keys_large)
            total_hybrid_models = hybrid_multiclass_count + hybrid_onevsall_count

            approach_table.add_row(
                "multi-class",
                f"Trains ONE model PER KEY (not per value)\n\n"
                f"• {num_keys} models total (one per annotation key)\n"
                f"• Each model learns ALL values of ITS key\n"
                f"• Example: One model for 'political_party' learns BQ, CAQ, CPC, etc.\n"
                f"• Example: Another model for 'sentiment' learns positive, negative, neutral\n"
                f"• [bold green]Total: {num_keys} models (one per key)[/bold green]\n\n"
                "[bold cyan]Best for:[/bold cyan] Standard classification with mutually exclusive categories per key"
            )
            approach_table.add_row(
                "one-vs-all",
                f"Trains ONE model PER VALUE (not per key)\n\n"
                f"• {total_values_count} binary models total (one per unique value)\n"
                f"• Each model: 'value X' vs NOT 'value X'\n"
                f"• Example: Separate model for 'political_party_BQ' (binary: BQ or not)\n"
                f"• Example: Separate model for 'sentiment_positive' (binary: positive or not)\n"
                f"• [bold yellow]Total: {total_values_count} models (one per value)[/bold yellow]\n\n"
                "[bold cyan]Best for:[/bold cyan] Imbalanced data"
            )
            approach_table.add_row(
                "multi-label",
                f"Trains ONE model with MULTIPLE labels per text\n\n"
                f"• {num_keys} models total (one per annotation key)\n"
                f"• Uses sigmoid activation (independent probabilities)\n"
                f"• Each text can have 0, 1, or many labels simultaneously\n"
                f"• Threshold-based selection (default: 0.5)\n"
                f"• [bold magenta]Total: {num_keys} models[/bold magenta]\n\n"
                "[bold cyan]Best for:[/bold cyan] Texts with overlapping categories"
            )
            approach_table.add_row(
                "hybrid",
                f"SMART: Choose multi-label OR adaptive multi-class/one-vs-all\n\n"
                f"• You'll choose between:\n"
                f"  -  [bold magenta]Multi-label[/bold magenta]: 1 model per key with sigmoid (overlapping labels)\n"
                f"  - [bold green]Multi-class + Binary[/bold green]: Adaptive per key (≤5 vals → multi-class, >5 → one-vs-all)\n"
                f"• For multi-class+binary on your data:\n"
                f"  - {hybrid_multiclass_count} keys use multi-class ({', '.join([k for k, _ in keys_small[:3]])}{'...' if len(keys_small) > 3 else ''})\n"
                f"  - {len(keys_large)} keys use one-vs-all ({', '.join([k for k, _ in keys_large[:3]])}{'...' if len(keys_large) > 3 else ''})\n"
                f"• [bold magenta]Total: varies based on choice[/bold magenta]\n\n"
                "[bold cyan]Best for:[/bold cyan] Mixed dataset - choose the best approach per use case (RECOMMENDED)"
            )
            approach_table.add_row(
                "custom",
                f"CUSTOM: You choose the strategy for EACH key individually\n\n"
                f"• You'll be asked for each of the {num_keys} keys\n"
                f"• Choose multi-class, one-vs-all, OR multi-label per key\n"
                f"• Example: multi-class for 'sentiment', multi-label for 'themes'\n"
                f"• [bold blue]Total: Variable (depends on your choices)[/bold blue]\n\n"
                "[bold cyan]Best for:[/bold cyan] Advanced users who want fine-grained control"
            )

        console.print(approach_table)
        console.print()

        # Determine available choices and default based on context
        if annotation_keys and len(annotation_keys) == 1:
            # Single key: no hybrid or custom modes
            available_choices = ["multi-class", "one-vs-all", "multi-label", "back"]
            default_approach = "multi-class"
        else:
            # Multiple keys: all modes available
            available_choices = ["multi-class", "one-vs-all", "multi-label", "hybrid", "custom", "back"]
            default_approach = "hybrid"

        training_approach = Prompt.ask(
            "[bold yellow]Training approach[/bold yellow]",
            choices=available_choices,
            default=default_approach
        )

        if training_approach == "back":
            return None

        # Store per-key strategy decisions
        key_strategies = {}  # {key_name: 'multi-class' or 'one-vs-all'}

        if training_approach == "hybrid":
            # Ask user to choose between multi-label or multi-class+binary
            console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            console.print("[bold cyan]  Hybrid Mode: Choose Your Strategy[/bold cyan]")
            console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n")

            # Display detailed statistics per key/value to help user make informed decisions
            console.print("[bold cyan]Detailed Statistics per Key/Value:[/bold cyan]")
            console.print("[dim]This shows the number of instances for each value in each key.[/dim]\n")

            stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, padding=(0, 1), expand=True)
            stats_table.add_column("Key", style="yellow bold", no_wrap=True)
            stats_table.add_column("Value", style="white", ratio=1, overflow="fold")
            stats_table.add_column("Instances", style="cyan", no_wrap=True, justify="right")
            stats_table.add_column("% of Key", style="green", no_wrap=True, justify="right")

            for key in keys_to_train:
                key_counts = value_counts_by_key.get(key, {})
                total_for_key = sum(key_counts.values()) if key_counts else 0
                sorted_values = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)

                # Show first value with key name
                if sorted_values:
                    first_val, first_count = sorted_values[0]
                    pct = (first_count / total_for_key * 100) if total_for_key > 0 else 0
                    stats_table.add_row(
                        f"[bold]{key}[/bold] ({len(sorted_values)} vals)",
                        first_val[:22] + "..." if len(first_val) > 25 else first_val,
                        str(first_count),
                        f"{pct:.1f}%"
                    )
                    # Show remaining values (up to 5 more)
                    for val, count in sorted_values[1:6]:
                        pct = (count / total_for_key * 100) if total_for_key > 0 else 0
                        stats_table.add_row(
                            "",
                            val[:22] + "..." if len(val) > 25 else val,
                            str(count),
                            f"{pct:.1f}%"
                        )
                    if len(sorted_values) > 6:
                        remaining = len(sorted_values) - 6
                        remaining_count = sum(c for _, c in sorted_values[6:])
                        stats_table.add_row("", f"[dim]... +{remaining} more values[/dim]", f"[dim]{remaining_count}[/dim]", "")
                    # Add separator row between keys
                    stats_table.add_row("", "", "", "")

            console.print(stats_table)
            console.print()

            # Calculate totals for display with default threshold of 5
            default_threshold = 5
            hybrid_mc_count = sum(1 for key in keys_to_train if len(all_keys_values[key]) <= default_threshold)
            hybrid_ova_count = sum(len(all_keys_values[key]) for key in keys_to_train if len(all_keys_values[key]) > default_threshold)
            hybrid_total = hybrid_mc_count + hybrid_ova_count

            hybrid_sub_table = Table(show_header=True, header_style="bold", box=box.ROUNDED, padding=(0, 2), expand=True)
            hybrid_sub_table.add_column("Option", style="bold cyan", no_wrap=True)
            hybrid_sub_table.add_column("Description", style="white", ratio=1, overflow="fold")

            hybrid_sub_table.add_row(
                "multi-label",
                f"[bold magenta]One model per key with sigmoid activation[/bold magenta]\n\n"
                f"• Each text can have MULTIPLE labels simultaneously\n"
                f"• Independent probabilities per label (BCEWithLogitsLoss)\n"
                f"• Threshold-based selection (default: 0.5)\n"
                f"• [bold magenta]Total: {num_keys} models (one per key)[/bold magenta]\n\n"
                "[bold cyan]Best for:[/bold cyan] Texts with overlapping/co-occurring categories"
            )
            hybrid_sub_table.add_row(
                "multi-class+binary",
                f"[bold green]Adaptive strategy per key[/bold green]\n\n"
                f"• Keys with ≤N values → Multi-class (mutually exclusive)\n"
                f"• Keys with >N values → One-vs-all (binary per value)\n"
                f"• You'll choose the threshold N (default: {default_threshold})\n"
                f"• [bold green]Total: varies based on threshold[/bold green]\n\n"
                "[bold cyan]Best for:[/bold cyan] Standard classification (one label per text per key)"
            )

            console.print(hybrid_sub_table)
            console.print()

            hybrid_sub_choice = Prompt.ask(
                "[bold yellow]Choose hybrid strategy[/bold yellow]",
                choices=["multi-label", "multi-class+binary", "back"],
                default="multi-class+binary"
            )

            if hybrid_sub_choice == "back":
                return None

            if hybrid_sub_choice == "multi-label":
                # Override training_approach to multi-label
                training_approach = "multi-label"
                console.print("\n[magenta]✓ Using multi-label strategy (one model per key with sigmoid)[/magenta]\n")
                # All keys use multi-label
                for key in keys_to_train:
                    key_strategies[key] = 'multi-label'
            else:
                # multi-class+binary: Ask for threshold
                console.print("\n[bold cyan]Multi-class + Binary Strategy Configuration:[/bold cyan]")
                console.print("[dim]Choose the threshold N that determines which keys use multi-class vs one-vs-all.[/dim]")
                console.print("[dim]Keys with ≤N values → multi-class (1 model), Keys with >N values → one-vs-all (N models)[/dim]\n")

                # Show preview of different threshold values
                console.print("[bold]Preview of threshold effects:[/bold]")
                for preview_threshold in [3, 5, 7, 10]:
                    mc_keys = [k for k in keys_to_train if len(all_keys_values[k]) <= preview_threshold]
                    ova_keys = [k for k in keys_to_train if len(all_keys_values[k]) > preview_threshold]
                    mc_models = len(mc_keys)
                    ova_models = sum(len(all_keys_values[k]) for k in ova_keys)
                    total = mc_models + ova_models
                    console.print(f"  • Threshold={preview_threshold}: {mc_models} multi-class + {ova_models} one-vs-all = [bold]{total} models[/bold]")
                console.print()

                # Ask user to choose threshold
                while True:
                    threshold_input = Prompt.ask(
                        "[bold yellow]Threshold (N) for multi-class vs one-vs-all[/bold yellow]",
                        default=str(default_threshold)
                    )
                    try:
                        hybrid_threshold = int(threshold_input)
                        if hybrid_threshold < 1:
                            console.print("[red]Threshold must be at least 1[/red]")
                            continue
                        break
                    except ValueError:
                        console.print("[red]Please enter a valid integer[/red]")

                console.print(f"\n[green]✓ Using threshold: {hybrid_threshold}[/green]")
                console.print(f"[dim]Keys with ≤{hybrid_threshold} values → multi-class, Keys with >{hybrid_threshold} values → one-vs-all[/dim]\n")

                # Calculate total models for hybrid approach with user-selected threshold
                total_hybrid_models = 0
                for key in keys_to_train:
                    num_values = len(all_keys_values[key])
                    if num_values <= hybrid_threshold:
                        key_strategies[key] = 'multi-class'
                        total_hybrid_models += 1
                        console.print(f"  • [green]{key}[/green] ({num_values} values) → [bold]multi-class[/bold] (1 model)")
                    else:
                        key_strategies[key] = 'one-vs-all'
                        total_hybrid_models += num_values
                        console.print(f"  • [yellow]{key}[/yellow] ({num_values} values) → [bold]one-vs-all[/bold] ({num_values} models)")

                console.print(f"\n[dim]Total models: {total_hybrid_models}[/dim]\n")

        elif training_approach == "custom":
            # User chooses per key
            console.print("\n[bold cyan] Custom Strategy Selection:[/bold cyan]")
            console.print("[dim]Choose the training strategy for each key individually.[/dim]\n")

            total_custom_models = 0
            has_multilabel_keys = False  # Track if any key uses multi-label
            for key in keys_to_train:
                num_values = len(all_keys_values[key])
                values_preview = ', '.join([f"'{v}'" for v in sorted(all_keys_values[key])[:3]])
                if num_values > 3:
                    values_preview += f" ... (+{num_values-3} more)"

                console.print(f"[bold]{key}[/bold] ({num_values} values)")
                console.print(f"[dim]  Values: {values_preview}[/dim]")
                console.print(f"  • [green]multi-class[/green]: 1 model learns all {num_values} values (mutually exclusive)")
                console.print(f"  • [yellow]one-vs-all[/yellow]: {num_values} binary models (one per value)")
                console.print(f"  • [magenta]multi-label[/magenta]: 1 model with sigmoid (overlapping labels allowed)")

                key_choice = Prompt.ask(
                    f"  Strategy for '{key}'",
                    choices=["multi-class", "one-vs-all", "multi-label", "m", "o", "l"],
                    default="multi-class" if num_values <= 5 else "one-vs-all"
                )

                # Normalize shortcuts
                if key_choice == "m":
                    key_choice = "multi-class"
                elif key_choice == "o":
                    key_choice = "one-vs-all"
                elif key_choice == "l":
                    key_choice = "multi-label"

                key_strategies[key] = key_choice

                if key_choice == "multi-class":
                    total_custom_models += 1
                    console.print(f"  ✓ Will train [green]1 model[/green] for {key}\n")
                elif key_choice == "one-vs-all":
                    total_custom_models += num_values
                    console.print(f"  ✓ Will train [yellow]{num_values} models[/yellow] for {key}\n")
                else:  # multi-label
                    total_custom_models += 1
                    has_multilabel_keys = True
                    console.print(f"  ✓ Will train [magenta]1 multi-label model[/magenta] for {key}\n")

            console.print(f"[bold cyan]Total models to train: {total_custom_models}[/bold cyan]\n")

            # If any key uses multi-label, set the training_approach accordingly
            if has_multilabel_keys and all(key_strategies[k] == 'multi-label' for k in keys_to_train):
                # All keys use multi-label
                training_approach = "multi-label"
            elif has_multilabel_keys:
                # Mixed: some multi-label, some not - keep as custom
                pass

        elif training_approach == "multi-class":
            # All keys use multi-class
            for key in keys_to_train:
                key_strategies[key] = 'multi-class'

        elif training_approach == "one-vs-all":
            # All keys use one-vs-all
            for key in keys_to_train:
                key_strategies[key] = 'one-vs-all'

    # Step 6c: Data Split Configuration
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    data_split_step = resolve_step_label("data_split", "STEP 12", context=step_context)
    console.print(f"[bold cyan]  {data_split_step}:[/bold cyan] [bold white]Data Split Configuration[/bold white]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

    split_config = cli_instance._configure_data_splits(
        keys_to_train=keys_to_train,
        all_keys_values=all_keys_values,
        training_approach=training_approach,
        key_strategies=key_strategies,
        total_samples=len(df)
    )

    if split_config is None:
        return None

    # Display split configuration summary
    cli_instance._display_split_summary(
        split_config=split_config,
        keys_to_train=keys_to_train,
        all_keys_values=all_keys_values,
        key_strategies=key_strategies
    )

    # Note: split_config will be stored in bundle.metadata after bundle is created

    # Step 6d: Label naming strategy
    console.print("\n[bold] Label Naming Strategy:[/bold]")
    console.print("[dim]This determines how label names appear in your training files and model predictions.[/dim]\n")

    # Generate examples based on SELECTED keys (not random example data)
    # Build concrete transformation examples
    transformation_examples = []
    for key in keys_to_train[:2]:  # Show 2 examples for clarity
        if key in all_keys_values:
            values = sorted(all_keys_values[key])[:2]  # First 2 values
            if values:
                for val in values:
                    transformation_examples.append({
                        'key': key,
                        'value': val,
                        'key_value': f"{key}_{val}",
                        'value_only': val
                    })

    # Create comparison table
    strategy_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED, expand=True)
    strategy_table.add_column("Strategy", style="cyan bold", no_wrap=True)
    strategy_table.add_column("Format", style="white", no_wrap=True)
    strategy_table.add_column("When to Use", style="white", ratio=1, overflow="fold")

    # Build key_value example string
    if transformation_examples:
        kv_format_examples = [f"'{ex['key_value']}'" for ex in transformation_examples[:3]]
        kv_format = f"key_value\nExample: {', '.join(kv_format_examples)}"
    else:
        kv_format = "key_value\nExample: 'sentiment_positive'"

    # Build value_only example string
    if transformation_examples:
        vo_format_examples = [f"'{ex['value_only']}'" for ex in transformation_examples[:3]]
        vo_format = f"value_only\nExample: {', '.join(vo_format_examples)}"
    else:
        vo_format = "value_only\nExample: 'positive'"

    strategy_table.add_row(
        "key_value",
        "Includes key prefix\n[dim](key_value)[/dim]",
        "✓ Training [bold]multiple keys[/bold]\n"
        "✓ Values might overlap between keys\n"
        "✓ [green]Recommended for most cases[/green]"
    )

    strategy_table.add_row(
        "value_only",
        "Only the value\n[dim](no prefix)[/dim]",
        "✓ Training [bold]single key only[/bold]\n"
        "✓ Values are unique across dataset\n"
        "[!] [yellow]Can cause conflicts with multiple keys[/yellow]"
    )

    console.print(strategy_table)
    console.print()

    # Show concrete transformation if we have examples
    if transformation_examples:
        console.print("[bold]How Your Data Will Be Transformed:[/bold]\n")

        transform_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.SIMPLE, expand=True)
        transform_table.add_column("Original (key → value)", style="cyan", no_wrap=True)
        transform_table.add_column("key_value format", style="green", no_wrap=True)
        transform_table.add_column("value_only format", style="yellow", no_wrap=True)

        for ex in transformation_examples[:4]:  # Show max 4 examples
            transform_table.add_row(
                f"{ex['key']} → {ex['value']}",
                ex['key_value'],
                ex['value_only']
            )

        console.print(transform_table)
        console.print()

    # Show warning if multiple keys and value_only
    if len(keys_to_train) > 1:
        console.print("[bold yellow]Recommendation:[/bold yellow]")
        console.print(f"[dim]You selected {len(keys_to_train)} keys. Use [bold cyan]key_value[/bold cyan] to avoid label conflicts.")
        console.print(f"[dim]Example: If both 'affiliation' and 'gender' have value 'no', they would conflict with [yellow]value_only[/yellow].[/dim]\n")
    else:
        console.print("[dim]Tip: With a single key, both strategies work fine. [cyan]key_value[/cyan] is still recommended for consistency.[/dim]\n")

    label_strategy = Prompt.ask("Label naming strategy", choices=["key_value", "value_only", "back"], default="key_value")
    if label_strategy == "back":
        return None

    # Derive mode based on approach
    if training_approach == "one-vs-all":
        mode = "multi-label"  # one-vs-all uses multi-label infrastructure
    elif training_approach == "multi-label":
        mode = "multi-label"  # true multi-label uses multi-label infrastructure
    else:
        mode = "single-label"  # multi-class uses single-label infrastructure

    # Step 8: Additional Columns (ID, Language)
    console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    additional_step = resolve_step_label("additional_columns", "STEP 12", context=step_context)
    console.print(f"[bold cyan]  {additional_step}:[/bold cyan] [bold white]Additional Columns (Optional)[/bold white]")
    console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print("[dim]Optional: Select ID and language columns if available in your dataset.[/dim]\n")

    data_path = Path(csv_path) if 'csv_path' in locals() else Path(selection['data_path'])

    # Use modernized ID selection - load dataframe if needed
    try:
        if not isinstance(df, pd.DataFrame):
            # Need to load dataframe for ID detection
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path, nrows=1000)
            elif data_path.suffix.lower() == '.json':
                df = pd.read_json(data_path, lines=False, nrows=1000)
            elif data_path.suffix.lower() == '.jsonl':
                df = pd.read_json(data_path, lines=True, nrows=1000)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path, nrows=1000)
            elif data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(data_path).head(1000)
            else:
                df = pd.read_csv(data_path, nrows=1000)

        # Use new unified ID selection
        id_column = DataDetector.display_and_select_id_column(
            console,
            df,
            text_column=selected_text_column,
            step_label="Identifier Column (Optional)",
            data_path=data_path
        )
    except Exception as e:
        logger.warning(f"Could not detect ID columns: {e}")
        console.print(f"[yellow][!] Could not analyze ID columns[/yellow]")
        console.print("[dim]An automatic ID will be generated[/dim]")
        id_column = None

    # Language column handling - check if already processed in Step 5
    # Skip if we already did language detection (either with column or auto-detection)
    language_already_processed = 'lang_column' in locals() and confirmed_languages

    if language_already_processed:
        # Language was already handled in Step 5
        if lang_column:
            console.print(f"\n[green]✓ Language column from Step 5: '{lang_column}'[/green]")
        else:
            console.print(f"\n[green]✓ Languages detected in Step 5: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
            console.print(f"[dim]  (Using automatic language detection - no specific column)[/dim]")
    elif analysis['language_column_candidates']:
        # Language column detected but Step 5 was skipped - ask user
        lang_column_candidate = analysis['language_column_candidates'][0]
        console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")
        if all_columns:
            console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
        while True:
            override_lang = Prompt.ask("\n[bold yellow]Language column (optional)[/bold yellow]", default=lang_column_candidate)
            if not override_lang or override_lang in all_columns:
                lang_column = override_lang if override_lang else lang_column_candidate
                break
            console.print(f"[red]✗ Column '{override_lang}' not found in dataset![/red]")
            console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

    # Handle training approach with key_strategies support
    if 'training_approach' in locals() and training_approach == "one-vs-all":
        # Convert to multi-label format for one-vs-all training
        request = TrainingDataRequest(
            input_path=csv_path,
            format="llm_json",
            text_column=selected_text_column,
            annotation_column=selected_annotation_column,
            annotation_keys=annotation_keys,
            label_strategy=label_strategy,
            mode="multi-label",  # Use multi-label to trigger one-vs-all training
            id_column=id_column or None,
            lang_column=lang_column or None,
            key_strategies={k: 'one-vs-all' for k in (annotation_keys or [])} if 'key_strategies' not in locals() else None
        )
        bundle = builder.build(request)

        # Mark this as one-vs-all for distributed training
        if bundle:
            bundle.metadata['training_approach'] = 'one-vs-all'
            bundle.metadata['original_strategy'] = 'single-label'
    elif 'training_approach' in locals() and training_approach == "multi-label":
        # True multi-label: one model with sigmoid activation for multiple labels per text
        request = TrainingDataRequest(
            input_path=csv_path,
            format="llm_json",
            text_column=selected_text_column,
            annotation_column=selected_annotation_column,
            annotation_keys=annotation_keys,
            label_strategy=label_strategy,
            mode="multi-label",  # Use multi-label data format
            id_column=id_column or None,
            lang_column=lang_column or None,
            key_strategies=key_strategies if 'key_strategies' in locals() else None
        )
        bundle = builder.build(request)

        # Mark this as true multi-label (not one-vs-all)
        if bundle:
            bundle.metadata['training_approach'] = 'multi-label'
            bundle.metadata['multi_label'] = True
            bundle.metadata['multi_label_threshold'] = 0.5
    else:
        # Standard mode (can be multi-class, hybrid, or custom)
        # Pass key_strategies if available (from hybrid/custom mode)
        request = TrainingDataRequest(
            input_path=csv_path,
            format="llm_json",
            text_column=selected_text_column,
            annotation_column=selected_annotation_column,
            annotation_keys=annotation_keys,
            label_strategy=label_strategy,
            mode=mode,
            id_column=id_column or None,
            lang_column=lang_column or None,
            key_strategies=key_strategies if 'key_strategies' in locals() else None
        )
        bundle = builder.build(request)

    # Store language metadata in bundle for later use (model selection will happen in training mode)
    if bundle:
        if confirmed_languages:
            bundle.metadata['confirmed_languages'] = confirmed_languages
        if language_distribution:
            bundle.metadata['language_distribution'] = language_distribution
        # Save training approach if user made a choice (multi-label/one-vs-all)
        if 'training_approach' in locals() and training_approach:
            bundle.metadata['training_approach'] = training_approach
            # CRITICAL: Derive multi_label flag from training_approach for UI and training
            if training_approach == 'multi-label':
                bundle.metadata['multi_label'] = True
                if 'multi_label_threshold' not in bundle.metadata:
                    bundle.metadata['multi_label_threshold'] = 0.5
        # Store annotation keys (categories) for benchmark mode
        # Use keys_to_train (which contains all keys when user selects ALL)
        if 'keys_to_train' in locals() and keys_to_train:
            bundle.metadata['categories'] = keys_to_train
        elif 'annotation_keys' in locals() and annotation_keys:
            bundle.metadata['categories'] = annotation_keys
        # Store source file and annotation column for benchmark mode
        bundle.metadata['source_file'] = str(csv_path)
        bundle.metadata['annotation_column'] = selected_annotation_column
        bundle.metadata['all_keys_values'] = {
            key: sorted(list(values))
            for key, values in all_keys_values.items()
        }
        bundle.metadata['value_counts_by_key'] = {
            key: {val: int(count) for val, count in counts.items()}
            for key, counts in value_counts_by_key.items()
        }
        # Store split configuration if it exists
        if 'split_config' in locals() and split_config:
            bundle.metadata['split_config'] = split_config
        # Text length stats for intelligent model selection later
        # ONLY calculate if not already done (avoid duplicate analysis)
        if 'text_length_stats' in locals() and text_length_stats:
            # Already calculated with user interaction - reuse it
            bundle.metadata['text_length_stats'] = text_length_stats
        elif selected_text_column in df.columns:
            # Not calculated yet - do it now without UI
            text_length_stats = cli_instance.analyze_text_lengths(
                df=df,
                text_column=selected_text_column,
                display_results=False  # Silent calculation
            )
            bundle.metadata['text_length_stats'] = text_length_stats


    # ========================================================================
    # Save metadata to Annotator Factory session
    # ========================================================================

    if bundle and session_dirs:
        metadata_file = session_dirs.get("session_root", Path("data")) / "training_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(metadata_file, 'w') as f:
            json.dump(bundle.metadata, f, indent=2, default=str)
        console.print(f"\n[green]✓ Training metadata saved to Annotator Factory session[/green]")

    # ========================================================================
    # Bundle Summary and Training Execution (exactly like Training Arena)
    # ========================================================================

    training_result = None
    if bundle:
        # Display bundle summary (like Training Arena)
        cli_instance._training_studio_render_bundle_summary(bundle)

        # ========================================================================
        # CRITICAL: Centralized validation of ALL training files
        # This detects ALL insufficient labels for ALL modes in ONE pass
        # Replaces multiple validation prompts throughout training
        # ========================================================================
        will_train_by_language = False
        if 'confirmed_languages' in locals() and confirmed_languages and len(confirmed_languages) > 1:
            will_train_by_language = True

        can_continue, error_msg = cli_instance._validate_all_training_files_before_training(
            bundle=bundle,
            min_samples=2,
            train_by_language=will_train_by_language
        )

        if not can_continue:
            console.print(f"\n[red][FAIL] Training stopped: {error_msg}[/red]\n")
            return {
                "status": "cancelled",
                "session_id": training_session_id,
                "bundle": bundle,
                "metadata": bundle.metadata if bundle else {},
                "training_result": None,
                "error": error_msg
            }

        # Execute training (like Training Arena)
        training_result = cli_instance._training_studio_confirm_and_execute(
            bundle=bundle,
            mode='quick',
            session_id=training_session_id,
            step_context="factory_quick"
        )

    if training_result:
        _merge_trained_models(training_result.get('trained_model_paths'))
        _merge_trained_models(training_result.get('trained_models'))

    loader = getattr(cli_instance, "_load_saved_factory_training_results", None)
    if callable(loader):
        try:
            reconstructed = loader(
                session_id=actual_session_id,
                session_dirs=session_dirs,
                training_workflow={}
            )
        except Exception:
            reconstructed = None
        if reconstructed:
            _merge_trained_models(reconstructed.get("training_result", {}).get("trained_models"))

    if trained_models_map:
        if training_result is None:
            training_result = {}
        training_result['trained_models'] = dict(trained_models_map)
        training_result['trained_model_paths'] = dict(trained_models_map)
        training_result['models_trained'] = list(trained_models_map.keys())
        if bundle:
            bundle.metadata.setdefault('trained_models', {})
            bundle.metadata['trained_models'] = dict(trained_models_map)
            bundle.metadata['trained_model_paths'] = dict(trained_models_map)

    # Display where the training reports are saved (for Annotator Factory)
    if session_dirs and "session_root" in session_dirs and session_manager:
        console.print("\n[bold cyan]Training Data Organization:[/bold cyan]")
        console.print(f"  [green]{session_dirs['session_root']}/[/green]")
        console.print(f"  ├── SESSION_SUMMARY.txt         [dim]# Complete training overview[/dim]")
        console.print(f"  ├── training_data/              [dim]# Datasets & analysis reports[/dim]")
        console.print(f"  │   ├── *.jsonl                 [dim]# Training datasets[/dim]")
        console.print(f"  │   ├── model_catalog.csv       [dim]# All models to train[/dim]")
        console.print(f"  │   ├── database_reports/       [dim]# Individual .txt reports[/dim]")
        console.print(f"  │   └── ...                     [dim]# Distribution & summaries[/dim]")
        console.print(f"  ├── training_metrics/           [dim]# Model performance metrics[/dim]")
        console.print(f"  ├── training_session_metadata/  [dim]# Configuration files[/dim]")
        console.print(f"  ├── annotated_data/             [dim]# Original annotations[/dim]")
        console.print(f"  └── metadata/                   [dim]# Annotation metadata[/dim]")
        console.print()

    # Return complete results
    return {
        "status": "completed" if bundle else "failed",
        "session_id": actual_session_id,  # Use the actual session ID
        "bundle": bundle,
        "metadata": bundle.metadata if bundle else {},
        "training_result": training_result,
        "training_logs_dir": session_manager.session_dir if session_manager else None,
        "trained_model_paths": trained_models_map,
    }


def _is_training_arena_method(obj: Any) -> bool:
    """Return True if obj is a function expecting a `self` parameter."""
    return inspect.isfunction(obj) and obj.__code__.co_varnames[:1] == ('self',)


TRAINING_ARENA_METHODS = [
    name for name, obj in globals().items()
    if _is_training_arena_method(obj)
]

del _is_training_arena_method

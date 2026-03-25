#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
parallel_training_cli.py

MAIN OBJECTIVE:
---------------
CLI interface for parallel training that maximizes GPU + CPU utilization.
Provides a command-line interface to launch parallel training sessions
with unified progress tracking.

FEATURES:
---------
1) Command-line argument parsing for parallel training configuration
2) Automatic task discovery from data directories
3) Integration with ParallelTrainingManager for execution
4) Support for custom batch sizes, worker counts, and model selection

Dependencies:
-------------
- argparse
- json
- logging
- pathlib
- llm_tool.trainers.parallel_training_manager
- llm_tool.utils.logging_utils

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


def analyze_hardware():
    """Analyze and display hardware capabilities."""
    console = Console()

    try:
        from llm_tool.trainers.hybrid_parallel_trainer import HardwareAnalyzer
    except ImportError as e:
        console.print(f"[red]Error importing parallel training module: {e}[/red]")
        return

    console.print("\n[bold cyan]🔍 Analyzing Hardware Resources...[/bold cyan]\n")

    analyzer = HardwareAnalyzer()
    info = analyzer.analyze()

    # Hardware info table
    hw_table = Table(
        title="📊 System Hardware",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    hw_table.add_column("Component", style="cyan")
    hw_table.add_column("Details", style="white")
    hw_table.add_column("Status", style="green")

    # GPU
    gpu = info.get("gpu", {})
    if gpu.get("available"):
        gpu_status = "✅ Available"
        gpu_mem = f"{gpu.get('memory_gb', 0):.1f} GB total, {gpu.get('available_memory_gb', 0):.1f} GB free"
    else:
        gpu_status = "❌ Not Available"
        gpu_mem = "-"

    hw_table.add_row(
        "🖥️ GPU",
        f"{gpu.get('name', 'Unknown')} ({gpu.get('type', 'unknown')})",
        gpu_status,
    )
    hw_table.add_row("   Memory", gpu_mem, "")

    # CPU
    cpu = info.get("cpu", {})
    hw_table.add_row(
        "💻 CPU",
        cpu.get("name", "Unknown"),
        f"✅ {cpu.get('physical_cores', 0)} cores",
    )
    hw_table.add_row(
        "   Threads",
        f"{cpu.get('logical_cores', 0)} logical cores",
        "",
    )

    # Memory
    mem = info.get("memory", {})
    mem_used_pct = mem.get("used_percent", 0)
    mem_status = "✅ OK" if mem_used_pct < 80 else ("⚠️ High" if mem_used_pct < 90 else "❌ Critical")
    hw_table.add_row(
        "🧠 RAM",
        f"{mem.get('total_gb', 0):.1f} GB total",
        mem_status,
    )
    hw_table.add_row(
        "   Available",
        f"{mem.get('available_gb', 0):.1f} GB ({100 - mem_used_pct:.0f}%)",
        "",
    )

    console.print(hw_table)

    # Create resource plan for 30 models (typical benchmark)
    console.print("\n[bold cyan]📋 Resource Allocation Plan (for 30 models):[/bold cyan]\n")

    plan = analyzer.create_resource_plan("xlm-roberta-base", total_tasks=30)

    plan_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        expand=True,
    )
    plan_table.add_column("Device", style="cyan", no_wrap=True)
    plan_table.add_column("Batch Size", justify="center")
    plan_table.add_column("DataLoader Workers", justify="center")
    plan_table.add_column("Priority", justify="center")

    if plan.gpu_allocation:
        gpu = plan.gpu_allocation
        plan_table.add_row(
            f"🖥️ GPU ({gpu.device_id})",
            str(gpu.batch_size),
            str(gpu.num_workers),
            "⭐ HIGH",
        )

    for cpu in plan.cpu_allocations:
        plan_table.add_row(
            f"💻 CPU ({cpu.device_id})",
            str(cpu.batch_size),
            str(cpu.num_workers),
            "Normal",
        )

    console.print(plan_table)

    # Summary
    console.print(f"\n[bold green]📈 Parallel Capacity:[/bold green] {plan.total_parallel_workers} models simultaneously")
    console.print(f"[bold green]🎯 Recommended batch:[/bold green] {plan.recommended_models_per_batch} models per round\n")


def run_parallel_training(
    metadata_path: str,
    model_name: str = "xlm-roberta-base",
    epochs: int = 10,
    output_base: str = "models",
    max_cpu_workers: Optional[int] = None,
    dry_run: bool = False,
):
    """Run parallel training from metadata file."""
    console = Console()

    # Load metadata
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        console.print(f"[red]Error: Metadata file not found: {metadata_path}[/red]")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    training_files = metadata.get("dataset_config", {}).get("training_files", {})

    if not training_files:
        console.print("[red]Error: No training files found in metadata[/red]")
        return

    # Display training info
    console.print(Panel(
        f"[bold]Model:[/bold] {model_name}\n"
        f"[bold]Epochs:[/bold] {epochs}\n"
        f"[bold]Categories:[/bold] {len(training_files)}\n"
        f"[bold]Output:[/bold] {output_base}",
        title="🚀 Parallel Training Configuration",
        border_style="cyan",
    ))

    if dry_run:
        console.print("\n[yellow]Dry run mode - not executing training[/yellow]")
        console.print("\n[bold]Training files:[/bold]")
        for category, path in training_files.items():
            exists = "✅" if Path(path).exists() else "❌"
            console.print(f"  {exists} {category}: {path}")
        return

    # Run parallel training
    try:
        from llm_tool.trainers.multi_label_trainer import train_models_parallel

        results = train_models_parallel(
            training_files=training_files,
            model_name=model_name,
            epochs=epochs,
            output_base=output_base,
            max_cpu_workers=max_cpu_workers,
            console=console,
        )

        # Print summary
        console.print("\n[bold green]✨ Training Complete![/bold green]\n")

        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        error_count = sum(1 for r in results.values() if r.get("status") == "error")

        console.print(f"[green]✅ Successful:[/green] {success_count}")
        if error_count > 0:
            console.print(f"[red]❌ Failed:[/red] {error_count}")

        # Best models
        successful = [(k, v) for k, v in results.items() if v.get("status") == "success"]
        if successful:
            best = max(successful, key=lambda x: x[1].get("f1_score", 0))
            console.print(f"\n[bold]🏆 Best Model:[/bold] {best[0]} (F1={best[1].get('f1_score', 0):.4f})")

    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel Training CLI - Maximize GPU + CPU utilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze hardware capabilities
  python -m llm_tool.cli.parallel_training_cli --analyze

  # Run parallel training
  python -m llm_tool.cli.parallel_training_cli \\
      --metadata path/to/training_metadata.json \\
      --model xlm-roberta-base \\
      --epochs 10

  # Dry run to check configuration
  python -m llm_tool.cli.parallel_training_cli \\
      --metadata path/to/training_metadata.json \\
      --dry-run
        """,
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze hardware capabilities and display resource plan",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to training_metadata.json file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xlm-roberta-base",
        help="Model name (default: xlm-roberta-base)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output base directory (default: models)",
    )
    parser.add_argument(
        "--max-cpu-workers",
        type=int,
        default=None,
        help="Maximum CPU workers (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running training",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Execute command
    if args.analyze:
        analyze_hardware()
    elif args.metadata:
        run_parallel_training(
            metadata_path=args.metadata,
            model_name=args.model,
            epochs=args.epochs,
            output_base=args.output,
            max_cpu_workers=args.max_cpu_workers,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

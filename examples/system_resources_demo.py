#!/usr/bin/env python3
"""
Demo script showing how to use system resource detection in LLM Tool.

This script demonstrates:
1. Basic resource detection
2. Getting recommendations
3. Using recommendations in configuration
4. Checking minimum requirements
5. Displaying resources
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_tool.utils.system_resources import (
    detect_resources,
    get_device_recommendation,
    get_optimal_batch_size,
    get_optimal_workers,
    check_minimum_requirements
)
from llm_tool.utils.resource_display import (
    display_resources,
    get_resource_summary_text
)
from llm_tool.config.settings import get_settings

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def demo_basic_detection():
    """Demo 1: Basic resource detection"""
    console.print("\n[bold cyan]Demo 1: Basic Resource Detection[/bold cyan]")
    console.print("=" * 60)

    # Detect all resources
    resources = detect_resources()

    # Print summary
    summary = get_resource_summary_text(resources)
    console.print(f"\n[green]System Summary:[/green] {summary}")

    # Print detailed information
    console.print(f"\n[yellow]GPU:[/yellow]")
    console.print(f"  Type: {resources.gpu.device_type}")
    console.print(f"  Available: {resources.gpu.available}")
    if resources.gpu.available:
        console.print(f"  Name: {resources.gpu.device_names[0]}")
        console.print(f"  Memory: {resources.gpu.total_memory_gb:.1f} GB")

    console.print(f"\n[yellow]CPU:[/yellow]")
    console.print(f"  Cores: {resources.cpu.physical_cores} physical, {resources.cpu.logical_cores} logical")
    console.print(f"  Architecture: {resources.cpu.architecture}")

    console.print(f"\n[yellow]RAM:[/yellow]")
    console.print(f"  Total: {resources.memory.total_gb:.1f} GB")
    console.print(f"  Available: {resources.memory.available_gb:.1f} GB")
    console.print(f"  Used: {resources.memory.percent_used:.1f}%")


def demo_recommendations():
    """Demo 2: Getting and using recommendations"""
    console.print("\n\n[bold cyan]Demo 2: System Recommendations[/bold cyan]")
    console.print("=" * 60)

    # Get recommendations
    resources = detect_resources()
    recommendations = resources.get_recommendation()

    console.print("\n[green]Recommended settings for optimal performance:[/green]")
    console.print(f"  Device: [bold]{recommendations['device'].upper()}[/bold]")
    console.print(f"  Batch Size: [bold]{recommendations['batch_size']}[/bold]")
    console.print(f"  Workers: [bold]{recommendations['num_workers']}[/bold]")
    console.print(f"  FP16: [bold]{'Yes' if recommendations['use_fp16'] else 'No'}[/bold]")
    console.print(f"  Gradient Accumulation: [bold]{recommendations['gradient_accumulation_steps']}[/bold]")

    if recommendations['notes']:
        console.print("\n[yellow]Notes:[/yellow]")
        for note in recommendations['notes']:
            console.print(f"  • {note}")


def demo_helper_functions():
    """Demo 3: Using helper functions"""
    console.print("\n\n[bold cyan]Demo 3: Helper Functions[/bold cyan]")
    console.print("=" * 60)

    # Use helper functions
    device = get_device_recommendation()
    batch_size = get_optimal_batch_size()
    workers = get_optimal_workers()

    console.print(f"\n[green]Quick access functions:[/green]")
    console.print(f"  get_device_recommendation() → [bold]{device}[/bold]")
    console.print(f"  get_optimal_batch_size() → [bold]{batch_size}[/bold]")
    console.print(f"  get_optimal_workers() → [bold]{workers}[/bold]")


def demo_minimum_requirements():
    """Demo 4: Checking minimum requirements"""
    console.print("\n\n[bold cyan]Demo 4: Checking Minimum Requirements[/bold cyan]")
    console.print("=" * 60)

    # Check different requirement scenarios
    scenarios = [
        {"min_ram_gb": 4, "min_storage_gb": 10, "require_gpu": False},
        {"min_ram_gb": 16, "min_storage_gb": 50, "require_gpu": False},
        {"min_ram_gb": 32, "min_storage_gb": 100, "require_gpu": True},
    ]

    for i, scenario in enumerate(scenarios, 1):
        meets, issues = check_minimum_requirements(**scenario)

        console.print(f"\n[yellow]Scenario {i}:[/yellow]")
        console.print(f"  Requirements: {scenario['min_ram_gb']}GB RAM, "
                     f"{scenario['min_storage_gb']}GB storage, "
                     f"GPU {'required' if scenario['require_gpu'] else 'optional'}")

        if meets:
            console.print("  [green]✓ System meets requirements[/green]")
        else:
            console.print("  [red]✗ System does not meet requirements:[/red]")
            for issue in issues:
                console.print(f"    - {issue}")


def demo_settings_integration():
    """Demo 5: Integration with Settings"""
    console.print("\n\n[bold cyan]Demo 5: Settings Integration[/bold cyan]")
    console.print("=" * 60)

    # Get settings
    settings = get_settings()

    console.print("\n[yellow]Before applying recommendations:[/yellow]")
    console.print(f"  Batch Size: {settings.training.batch_size}")
    console.print(f"  Workers: {settings.data.max_workers}")
    console.print(f"  Device: {settings.local_model.device}")
    console.print(f"  FP16: {settings.training.fp16}")

    # Apply recommendations
    settings.apply_system_recommendations()

    console.print("\n[green]After applying recommendations:[/green]")
    console.print(f"  Batch Size: {settings.training.batch_size}")
    console.print(f"  Workers: {settings.data.max_workers}")
    console.print(f"  Device: {settings.local_model.device}")
    console.print(f"  FP16: {settings.training.fp16}")


def demo_visual_display():
    """Demo 6: Visual display of resources"""
    console.print("\n\n[bold cyan]Demo 6: Visual Display[/bold cyan]")
    console.print("=" * 60)

    # Display full resources with recommendations
    display_resources(show_recommendations=True, compact=False)


def main():
    """Run all demos"""
    console.print(Panel(
        "[bold magenta]LLM Tool - System Resource Detection Demo[/bold magenta]\n"
        "[dim]This demo shows how to detect and use system resources[/dim]",
        border_style="magenta"
    ))

    demo_basic_detection()
    demo_recommendations()
    demo_helper_functions()
    demo_minimum_requirements()
    demo_settings_integration()
    demo_visual_display()

    console.print("\n\n[bold green]✓ Demo completed successfully![/bold green]\n")


if __name__ == "__main__":
    main()

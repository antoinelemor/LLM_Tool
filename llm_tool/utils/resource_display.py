#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
resource_display.py

MAIN OBJECTIVE:
---------------
Rich-based display module for system resources.
Provides beautiful, informative tables for system resource information.

Dependencies:
-------------
- rich: For beautiful terminal output
- system_resources: For resource detection

MAIN FEATURES:
--------------
1) Create Rich tables for system resources
2) Display GPU, CPU, RAM, and storage information
3) Show recommendations
4) Color-coded status indicators
5) Compact and detailed display modes

Author:
-------
Antoine Lemor
"""

from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .system_resources import (
    SystemResources,
    detect_resources,
    get_detector
)


def _format_gb(value: float) -> str:
    """Format GB values with appropriate precision"""
    if value >= 100:
        return f"{value:.0f} GB"
    elif value >= 10:
        return f"{value:.1f} GB"
    else:
        return f"{value:.2f} GB"


def _format_percent(value: float) -> str:
    """Format percentage values"""
    return f"{value:.1f}%"


def _get_status_color(percent: float) -> str:
    """Get color based on usage percentage"""
    if percent >= 90:
        return "red"
    elif percent >= 70:
        return "yellow"
    else:
        return "green"


def _create_progress_bar(percent: float, width: int = 20) -> str:
    """Create a text-based progress bar"""
    filled = int((percent / 100) * width)
    empty = width - filled

    if percent >= 90:
        color = "red"
    elif percent >= 70:
        color = "yellow"
    else:
        color = "green"

    bar = "█" * filled + "░" * empty
    return f"[{color}]{bar}[/{color}]"


def create_resource_table(
    resources: Optional[SystemResources] = None,
    show_recommendations: bool = False,
    compact: bool = False
) -> Table:
    """
    Create a Rich table displaying system resources.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display. If None, will detect automatically.
    show_recommendations : bool
        Whether to show recommendations
    compact : bool
        Whether to use compact display mode

    Returns
    -------
    Table
        Rich table with system resource information
    """
    if resources is None:
        resources = detect_resources()

    # Create main table
    if compact:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE,
            padding=(0, 1)
        )
    else:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            title="[bold]⚙️  System Resources[/bold]",
            title_style="bold magenta"
        )

    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Details", style="white")

    if not compact:
        table.add_column("Status", justify="center")

    # GPU Information
    gpu = resources.gpu
    if gpu.available:
        gpu_type = gpu.device_type.upper()
        gpu_details = f"{gpu.device_names[0] if gpu.device_names else 'Unknown'}"

        if gpu.total_memory_gb > 0:
            gpu_details += f"\n{_format_gb(gpu.total_memory_gb)}"
            if gpu.available_memory_gb > 0:
                gpu_details += f" ({_format_gb(gpu.available_memory_gb)} available)"

        if not compact:
            if gpu.device_type == "cuda":
                status = "🟢 CUDA"
                if gpu.cuda_version:
                    status += f" {gpu.cuda_version}"
            elif gpu.device_type == "mps":
                status = "🟢 MPS"
            else:
                status = "🟡 CPU"

            table.add_row("GPU", gpu_details, status)
        else:
            table.add_row("GPU", f"{gpu_type}: {gpu_details}")
    else:
        if not compact:
            table.add_row("GPU", "No GPU detected", "🔴 None")
        else:
            table.add_row("GPU", "CPU only")

    # CPU Information
    cpu = resources.cpu
    cpu_details = f"{cpu.processor_name}"
    if cpu.physical_cores > 0:
        cpu_details += f"\n{cpu.physical_cores} cores ({cpu.logical_cores} threads)"

    if cpu.max_frequency_mhz > 0 and not compact:
        cpu_details += f"\n{cpu.max_frequency_mhz:.0f} MHz (max)"

    if not compact:
        if cpu.cpu_percent > 0:
            cpu_color = _get_status_color(cpu.cpu_percent)
            status = f"[{cpu_color}]{_format_percent(cpu.cpu_percent)}[/{cpu_color}]"
        else:
            status = "🟢 Ready"
        table.add_row("CPU", cpu_details, status)
    else:
        table.add_row("CPU", cpu_details)

    # Memory Information
    mem = resources.memory
    if mem.total_gb > 0:
        mem_details = f"{_format_gb(mem.total_gb)} total"
        mem_details += f"\n{_format_gb(mem.available_gb)} available"

        if not compact:
            mem_details += f"\n{_format_gb(mem.used_gb)} used"
            mem_color = _get_status_color(mem.percent_used)
            status = f"[{mem_color}]{_format_percent(mem.percent_used)}[/{mem_color}]"
            table.add_row("RAM", mem_details, status)
        else:
            table.add_row("RAM", mem_details)
    else:
        if not compact:
            table.add_row("RAM", "Unknown", "⚠️  N/A")
        else:
            table.add_row("RAM", "Unknown")

    # Storage Information
    storage = resources.storage
    if storage.total_gb > 0:
        storage_details = f"{_format_gb(storage.total_gb)} total"
        storage_details += f"\n{_format_gb(storage.available_gb)} available"

        if not compact:
            storage_color = _get_status_color(storage.percent_used)
            status = f"[{storage_color}]{_format_percent(storage.percent_used)}[/{storage_color}]"
            table.add_row("Storage", storage_details, status)
        else:
            table.add_row("Storage", storage_details)
    else:
        if not compact:
            table.add_row("Storage", "Unknown", "⚠️  N/A")
        else:
            table.add_row("Storage", "Unknown")

    # System Information (only in detailed mode)
    if not compact:
        system = resources.system
        sys_details = f"{system.os_name} {system.os_release}"
        sys_details += f"\nPython {system.python_version}"
        table.add_row("System", sys_details, "ℹ️  Info")

    return table


def create_recommendations_table(
    resources: Optional[SystemResources] = None,
    compact: bool = False
) -> Table:
    """
    Create a Rich table displaying recommendations.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to base recommendations on
    compact : bool
        Whether to use compact display mode

    Returns
    -------
    Table
        Rich table with recommendations
    """
    if resources is None:
        resources = detect_resources()

    recommendations = resources.get_recommendation()

    if compact:
        table = Table(
            show_header=True,
            header_style="bold green",
            box=box.SIMPLE,
            padding=(0, 1)
        )
    else:
        table = Table(
            show_header=True,
            header_style="bold green",
            box=box.ROUNDED,
            title="[bold]💡 Recommended Settings[/bold]",
            title_style="bold green"
        )

    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    if not compact:
        table.add_column("Reason", style="dim")

    # Device
    device_value = recommendations['device'].upper()
    if not compact:
        device_reason = f"Based on {resources.gpu.device_type.upper()} availability"
        table.add_row("Device", device_value, device_reason)
    else:
        table.add_row("Device", device_value)

    # Batch Size
    batch_size = str(recommendations['batch_size'])
    if not compact:
        if resources.gpu.available:
            batch_reason = f"Optimal for {_format_gb(resources.gpu.total_memory_gb)} GPU memory"
        else:
            batch_reason = "Conservative for CPU training"
        table.add_row("Batch Size", batch_size, batch_reason)
    else:
        table.add_row("Batch Size", batch_size)

    # Workers
    workers = str(recommendations['num_workers'])
    if not compact:
        worker_reason = f"Based on {resources.cpu.physical_cores} CPU cores"
        table.add_row("Workers", workers, worker_reason)
    else:
        table.add_row("Workers", workers)

    # FP16
    fp16 = "Yes" if recommendations['use_fp16'] else "No"
    if not compact:
        fp16_reason = "Mixed precision supported" if recommendations['use_fp16'] else "Not recommended"
        table.add_row("FP16", fp16, fp16_reason)
    else:
        table.add_row("FP16", fp16)

    # Gradient Accumulation
    if recommendations['gradient_accumulation_steps'] > 1:
        grad_accum = str(recommendations['gradient_accumulation_steps'])
        if not compact:
            table.add_row(
                "Grad. Accum.",
                grad_accum,
                "To simulate larger batch size"
            )
        else:
            table.add_row("Grad. Accum.", grad_accum)

    return table


def display_resources(
    resources: Optional[SystemResources] = None,
    show_recommendations: bool = True,
    compact: bool = False,
    console: Optional[Console] = None
):
    """
    Display system resources using Rich.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display
    show_recommendations : bool
        Whether to show recommendations
    compact : bool
        Whether to use compact display mode
    console : Console, optional
        Rich console to use for output
    """
    if console is None:
        console = Console()

    if resources is None:
        resources = detect_resources()

    # Display resource table
    resource_table = create_resource_table(resources, compact=compact)
    console.print(resource_table)

    # Display recommendations if requested
    if show_recommendations:
        console.print()  # Empty line
        recommendations_table = create_recommendations_table(resources, compact=compact)
        console.print(recommendations_table)

        # Display notes
        if not compact:
            recommendations = resources.get_recommendation()
            if recommendations['notes']:
                console.print()
                notes_text = "\n".join(f"• {note}" for note in recommendations['notes'])
                panel = Panel(
                    notes_text,
                    title="[bold]📝 Notes[/bold]",
                    border_style="dim",
                    padding=(1, 2)
                )
                console.print(panel)


def create_compact_resource_panel(
    resources: Optional[SystemResources] = None,
    title: str = "System Resources"
) -> Panel:
    """
    Create a compact panel with system resources.

    This is designed to be embedded in other displays (like mode home pages).

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display
    title : str
        Panel title

    Returns
    -------
    Panel
        Rich panel with compact resource information
    """
    if resources is None:
        resources = detect_resources()

    # Create compact table
    table = create_resource_table(resources, compact=True)

    # Create panel
    panel = Panel(
        table,
        title=f"[bold]{title}[/bold]",
        border_style="blue",
        padding=(0, 1)
    )

    return panel


def get_resource_summary_text(resources: Optional[SystemResources] = None) -> str:
    """
    Get a one-line summary of system resources.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to summarize

    Returns
    -------
    str
        One-line summary
    """
    if resources is None:
        resources = detect_resources()

    parts = []

    # GPU
    if resources.gpu.available:
        gpu_text = resources.gpu.device_type.upper()
        if resources.gpu.total_memory_gb > 0:
            gpu_text += f" ({resources.gpu.total_memory_gb:.0f}GB)"
        parts.append(gpu_text)
    else:
        parts.append("CPU")

    # CPU cores
    if resources.cpu.physical_cores > 0:
        parts.append(f"{resources.cpu.physical_cores}C")

    # RAM
    if resources.memory.total_gb > 0:
        parts.append(f"{resources.memory.total_gb:.0f}GB RAM")

    return " | ".join(parts)


def display_resource_header(
    resources: Optional[SystemResources] = None,
    console: Optional[Console] = None
):
    """
    Display a compact resource header (one line).

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display
    console : Console, optional
        Rich console to use for output
    """
    if console is None:
        console = Console()

    summary = get_resource_summary_text(resources)
    text = Text(f"⚙️  {summary}", style="dim cyan")
    console.print(text)


def create_visual_resource_panel(
    resources: Optional[SystemResources] = None,
    show_recommendations: bool = True
) -> Panel:
    """
    Create a highly visual and attractive resource panel for the home page.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display
    show_recommendations : bool
        Whether to show recommendations

    Returns
    -------
    Panel
        Rich panel with visual resource information
    """
    if resources is None:
        resources = detect_resources()

    from rich.table import Table
    from rich.columns import Columns

    # Create main grid with 2 columns
    main_table = Table.grid(padding=(0, 2))
    main_table.add_column()
    main_table.add_column()

    # === LEFT COLUMN: Hardware ===
    hardware_table = Table(
        show_header=False,
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        expand=False
    )
    hardware_table.add_column("Component", style="bold cyan", width=12)
    hardware_table.add_column("Details", style="white", width=45)
    hardware_table.add_column("Usage", width=25)

    # GPU Information
    gpu = resources.gpu
    if gpu.available:
        gpu_icon = "🎮" if gpu.device_type == "cuda" else "🍎" if gpu.device_type == "mps" else "💻"
        gpu_name = gpu.device_names[0] if gpu.device_names else "Unknown"

        gpu_details = Text()
        gpu_details.append(f"{gpu_icon} ", style="bold")
        gpu_details.append(gpu_name, style="bold bright_green")

        if gpu.total_memory_gb > 0:
            gpu_details.append(f"\n   {_format_gb(gpu.total_memory_gb)}", style="cyan")
            if gpu.available_memory_gb > 0:
                mem_percent = ((gpu.total_memory_gb - gpu.available_memory_gb) / gpu.total_memory_gb) * 100
                gpu_usage = Text()
                gpu_usage.append(_create_progress_bar(mem_percent, 20))
                gpu_usage.append(f"\n{_format_gb(gpu.available_memory_gb)} free", style="dim green")
            else:
                gpu_usage = Text("Ready", style="green")
        else:
            gpu_usage = Text("Ready", style="green")

        hardware_table.add_row("GPU", gpu_details, gpu_usage)
    else:
        gpu_details = Text()
        gpu_details.append("💻 CPU Only", style="yellow")
        hardware_table.add_row("GPU", gpu_details, "—")

    # CPU Information
    cpu = resources.cpu
    cpu_details = Text()
    cpu_details.append(f"⚡ ", style="bold yellow")
    if cpu.processor_name:
        # Shorten processor name if too long
        proc_name = cpu.processor_name[:35] + "..." if len(cpu.processor_name) > 35 else cpu.processor_name
        cpu_details.append(proc_name, style="bold bright_yellow")
    else:
        cpu_details.append(f"{cpu.architecture}", style="bold bright_yellow")

    if cpu.physical_cores > 0:
        cpu_details.append(f"\n   {cpu.physical_cores} cores / {cpu.logical_cores} threads", style="cyan")
        if cpu.max_frequency_mhz > 100:
            cpu_details.append(f" @ {cpu.max_frequency_mhz:.0f} MHz", style="dim")

    cpu_usage = Text()
    if cpu.cpu_percent > 0:
        cpu_usage.append(_create_progress_bar(cpu.cpu_percent, 20))
        cpu_usage.append(f"\n{cpu.cpu_percent:.1f}% used", style="dim")
    else:
        cpu_usage.append("Ready", style="green")

    hardware_table.add_row("CPU", cpu_details, cpu_usage)

    # Memory Information
    mem = resources.memory
    mem_details = Text()
    mem_details.append("🧠 ", style="bold magenta")
    mem_details.append("Memory (RAM)", style="bold bright_magenta")
    mem_details.append(f"\n   {_format_gb(mem.total_gb)} total", style="cyan")

    mem_usage = Text()
    mem_usage.append(_create_progress_bar(mem.percent_used, 20))
    mem_usage.append(f"\n{_format_gb(mem.available_gb)} free", style="dim green")

    hardware_table.add_row("RAM", mem_details, mem_usage)

    # Storage Information
    storage = resources.storage
    storage_details = Text()
    storage_details.append("💾 ", style="bold blue")
    storage_details.append("Storage (Disk)", style="bold bright_blue")
    storage_details.append(f"\n   {_format_gb(storage.total_gb)} total", style="cyan")

    storage_usage = Text()
    storage_usage.append(_create_progress_bar(storage.percent_used, 20))
    storage_usage.append(f"\n{_format_gb(storage.available_gb)} free", style="dim green")

    hardware_table.add_row("Storage", storage_details, storage_usage)

    # === RIGHT COLUMN: Recommendations ===
    if show_recommendations:
        recommendations = resources.get_recommendation()

        rec_table = Table(
            show_header=False,
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
            expand=False
        )
        rec_table.add_column("Setting", style="bold green", width=18)
        rec_table.add_column("Value", style="white", width=30)

        # Device
        device_text = Text()
        device_icon = "🎮" if recommendations['device'] == "cuda" else "🍎" if recommendations['device'] == "mps" else "💻"
        device_text.append(f"{device_icon} ", style="bold")
        device_text.append(recommendations['device'].upper(), style="bold bright_green")
        rec_table.add_row("🎯 Device", device_text)

        # Batch Size
        batch_text = Text()
        batch_text.append(str(recommendations['batch_size']), style="bold cyan")
        batch_text.append(" samples", style="dim")
        rec_table.add_row("📦 Batch Size", batch_text)

        # Workers
        workers_text = Text()
        workers_text.append(str(recommendations['num_workers']), style="bold yellow")
        workers_text.append(" threads", style="dim")
        rec_table.add_row("👷 Workers", workers_text)

        # FP16
        fp16_text = Text()
        if recommendations['use_fp16']:
            fp16_text.append("✓ Enabled", style="bold green")
        else:
            fp16_text.append("✗ Disabled", style="dim")
        rec_table.add_row("⚡ FP16", fp16_text)

        # Gradient Accumulation (only if > 1)
        if recommendations['gradient_accumulation_steps'] > 1:
            grad_text = Text()
            grad_text.append(str(recommendations['gradient_accumulation_steps']), style="bold magenta")
            grad_text.append(" steps", style="dim")
            rec_table.add_row("🔄 Grad. Accum.", grad_text)

        # Add notes
        if recommendations['notes']:
            rec_table.add_row("", "")  # Spacer
            notes_text = Text()
            for note in recommendations['notes']:
                notes_text.append(f"💡 {note}\n", style="dim cyan")
            rec_table.add_row("Notes", notes_text)

        # Add to main grid
        main_table.add_row(hardware_table, rec_table)
    else:
        main_table.add_row(hardware_table, "")

    # Create panel
    panel = Panel(
        main_table,
        title="[bold bright_cyan]⚙️  System Resources & Recommendations[/bold bright_cyan]",
        border_style="bright_blue",
        padding=(1, 2)
    )

    return panel


def create_mode_resource_banner(
    resources: Optional[SystemResources] = None
) -> Table:
    """
    Create a compact horizontal banner for mode pages.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display

    Returns
    -------
    Table
        Rich table with compact resource information
    """
    if resources is None:
        resources = detect_resources()

    # Create horizontal layout
    banner = Table(
        show_header=False,
        box=None,
        padding=(0, 3),
        expand=True
    )

    banner.add_column(justify="center")
    banner.add_column(justify="center")
    banner.add_column(justify="center")
    banner.add_column(justify="center")

    # GPU
    gpu_text = Text()
    if resources.gpu.available:
        if resources.gpu.device_type == "cuda":
            gpu_text.append("🎮 ", style="bold")
            gpu_text.append("CUDA", style="bold bright_green")
        elif resources.gpu.device_type == "mps":
            gpu_text.append("🍎 ", style="bold")
            gpu_text.append("MPS", style="bold bright_green")

        if resources.gpu.total_memory_gb > 0:
            gpu_text.append(f"\n{_format_gb(resources.gpu.total_memory_gb)}", style="cyan")
    else:
        gpu_text.append("💻 ", style="bold")
        gpu_text.append("CPU Only", style="yellow")

    # CPU
    cpu_text = Text()
    cpu_text.append("⚡ ", style="bold yellow")
    cpu_text.append(f"{resources.cpu.physical_cores} Cores", style="bold bright_yellow")
    if resources.cpu.cpu_percent > 0:
        cpu_text.append(f"\n{resources.cpu.cpu_percent:.1f}% used", style="dim")

    # RAM
    mem_text = Text()
    mem_text.append("🧠 ", style="bold magenta")
    mem_text.append(f"{_format_gb(resources.memory.total_gb)}", style="bold bright_magenta")
    mem_text.append(f"\n{_format_gb(resources.memory.available_gb)} free", style="dim green")

    # Recommendation
    rec = resources.get_recommendation()
    rec_text = Text()
    rec_text.append("💡 ", style="bold cyan")
    rec_text.append(f"Batch: {rec['batch_size']}", style="bold cyan")
    rec_text.append(f"\nWorkers: {rec['num_workers']}", style="dim")

    banner.add_row(gpu_text, cpu_text, mem_text, rec_text)

    return banner


def create_detailed_mode_panel(
    resources: Optional[SystemResources] = None,
    mode_name: str = "Mode"
) -> Panel:
    """
    Create a detailed but compact panel for mode pages.

    Parameters
    ----------
    resources : SystemResources, optional
        System resources to display
    mode_name : str
        Name of the mode

    Returns
    -------
    Panel
        Rich panel with mode-specific resource information
    """
    if resources is None:
        resources = detect_resources()

    # Create two-column layout
    main_grid = Table.grid(padding=(0, 2))
    main_grid.add_column()
    main_grid.add_column()

    # Left: Hardware
    hw_table = Table(show_header=False, box=None, padding=(0, 1))
    hw_table.add_column("", style="bold", width=8)
    hw_table.add_column("", width=30)

    # GPU
    if resources.gpu.available:
        gpu_icon = "🎮" if resources.gpu.device_type == "cuda" else "🍎"
        gpu_name = resources.gpu.device_names[0][:25] if resources.gpu.device_names else "GPU"
        gpu_line = Text()
        gpu_line.append(f"{gpu_icon} ", style="bold")
        gpu_line.append(resources.gpu.device_type.upper(), style="bold bright_green")
        gpu_line.append(f"  {gpu_name}", style="cyan")
        hw_table.add_row("GPU", gpu_line)

        if resources.gpu.total_memory_gb > 0:
            mem_bar = _create_progress_bar(
                ((resources.gpu.total_memory_gb - resources.gpu.available_memory_gb) / resources.gpu.total_memory_gb) * 100
                if resources.gpu.available_memory_gb > 0 else 0,
                15
            )
            hw_table.add_row("", Text(f"{mem_bar} {_format_gb(resources.gpu.available_memory_gb)} free"))
    else:
        hw_table.add_row("GPU", Text("💻 CPU Only", style="yellow"))

    # CPU
    cpu_line = Text()
    cpu_line.append("⚡ ", style="bold yellow")
    cpu_line.append(f"{resources.cpu.physical_cores}C/{resources.cpu.logical_cores}T", style="bold bright_yellow")
    if resources.cpu.cpu_percent > 0:
        cpu_line.append(f"  {resources.cpu.cpu_percent:.1f}% used", style="dim")
    hw_table.add_row("CPU", cpu_line)

    # RAM
    mem_bar = _create_progress_bar(resources.memory.percent_used, 15)
    mem_line = Text()
    mem_line.append(f"{mem_bar} {_format_gb(resources.memory.available_gb)}/{_format_gb(resources.memory.total_gb)}")
    hw_table.add_row("RAM", mem_line)

    # Right: Recommendations
    rec = resources.get_recommendation()
    rec_table = Table(show_header=False, box=None, padding=(0, 1))
    rec_table.add_column("", style="bold green", width=10)
    rec_table.add_column("", width=25)

    device_icon = "🎮" if rec['device'] == "cuda" else "🍎" if rec['device'] == "mps" else "💻"
    rec_table.add_row("🎯 Device", Text(f"{device_icon} {rec['device'].upper()}", style="bold green"))
    rec_table.add_row("📦 Batch", Text(str(rec['batch_size']), style="bold cyan"))
    rec_table.add_row("👷 Workers", Text(str(rec['num_workers']), style="bold yellow"))

    main_grid.add_row(hw_table, rec_table)

    panel = Panel(
        main_grid,
        title=f"[bold bright_blue]⚙️  {mode_name} - System Resources[/bold bright_blue]",
        border_style="blue",
        padding=(0, 1)
    )

    return panel


if __name__ == "__main__":
    # Test the display
    print("\n" + "=" * 60)
    print("Resource Display Test - Detailed Mode")
    print("=" * 60 + "\n")

    display_resources(show_recommendations=True, compact=False)

    print("\n\n" + "=" * 60)
    print("Resource Display Test - Compact Mode")
    print("=" * 60 + "\n")

    display_resources(show_recommendations=False, compact=True)

    print("\n\n" + "=" * 60)
    print("Resource Summary Test")
    print("=" * 60 + "\n")

    from rich.console import Console
    console = Console()
    display_resource_header(console=console)

    print("\n\n" + "=" * 60)
    print("Compact Panel Test")
    print("=" * 60 + "\n")

    panel = create_compact_resource_panel()
    console.print(panel)

#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
rich_progress_manager.py

MAIN OBJECTIVE:
---------------
Enhanced progress manager with rich UI components for llm_tool
Provides a single unified progress bar with dynamic status updates
and JSON sample display during pipeline execution.

Author:
-------
Antoine Lemor
"""

import time
import threading
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn
)
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class CompactPercentColumn(ProgressColumn):
    """Compact percentage display"""
    def render(self, task):
        percentage = task.percentage if task.percentage is not None else 0
        return Text(f"{percentage:>5.1f}%", style="bright_magenta")


@dataclass
class ProgressState:
    """Track overall progress state"""
    current_phase: str = ""
    current_progress: float = 0.0
    current_message: str = ""
    completed_items: int = 0
    total_items: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    json_count: int = 0
    last_json_sample: Optional[Dict] = None
    start_time: float = field(default_factory=time.time)
    annotation_samples: List[Dict] = field(default_factory=list)


class RichProgressManager:
    """Enhanced progress manager with rich UI components"""

    # Phase definitions with emojis
    PHASES = {
        'initialization': {'icon': '⚙️', 'label': 'Initialization'},
        'annotation': {'icon': '✍️', 'label': 'Annotation'},
        'validation': {'icon': '✅', 'label': 'Validation'},
        'training': {'icon': '🏋️', 'label': 'Training'},
        'evaluation': {'icon': '📊', 'label': 'Evaluation'},
        'deployment': {'icon': '🚀', 'label': 'Deployment'}
    }

    def __init__(self, show_json_every: int = 10, compact_mode: bool = True):
        """Initialize with configuration"""
        self.console = Console()
        self.state = ProgressState()
        self.lock = threading.Lock()
        self.show_json_every = show_json_every
        self.compact_mode = compact_mode

        # Progress tracking - SINGLE TASK ONLY
        self.progress = None
        self.overall_task_id = None
        self.subtask_task_id = None
        self.is_running = False

        # Sample tracking
        self.current_sample = None
        self.recent_errors = []
        self.recent_warnings = []
        self.last_json_display = 0

    def start(self):
        """Start the progress display"""
        if not self.is_running:
            self.is_running = True
            self.state.start_time = time.time()

            # Create single progress bar
            self.progress = Progress(
                SpinnerColumn(style="bright_cyan", spinner_name="dots"),
                TextColumn("[bold]{task.description}"),
                BarColumn(
                    bar_width=50,
                    complete_style="bright_green",
                    finished_style="green",
                    pulse_style="cyan"
                ),
                CompactPercentColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(compact=True, elapsed_when_finished=True),
                console=self.console,
                expand=False,
                transient=False
            )

            # Start progress
            self.progress.start()

            # Add SINGLE task
            self.overall_task_id = self.progress.add_task(
                "Overall: Initialisation",
                total=100,
                completed=0
            )

            self.subtask_task_id = self.progress.add_task(
                "Subtask: En attente",
                total=100,
                completed=0
            )

            # Clear line after to ensure clean display
            print()

    def pause_for_training(self):
        """Pause progress display for training phase"""
        if self.is_running and self.progress:
            try:
                # Update status
                self.progress.update(
                    self.overall_task_id,
                    description="⏸ Pausing for training..."
                )
                if self.subtask_task_id is not None:
                    self.progress.update(
                        self.subtask_task_id,
                        description="Subtask: Pause",
                        completed=0
                    )

                time.sleep(0.5)

                # Stop and cleanup
                self.progress.stop()
                self.progress = None

                print("\n━━━ Progress paused for training phase ━━━\n")

                self._paused = True

            except Exception as e:
                print(f"Warning: Could not pause progress cleanly: {e}")

    def resume_after_training(self):
        """Resume progress display after training"""
        if self.is_running and hasattr(self, '_paused'):
            try:
                self.console.print("[dim cyan]━━━ Resuming progress display ━━━[/dim cyan]\n")

                # Recreate progress
                self.progress = Progress(
                    SpinnerColumn(style="bright_cyan", spinner_name="dots"),
                    TextColumn("[bold]{task.description}"),
                    BarColumn(
                        bar_width=50,
                        complete_style="bright_green",
                        finished_style="green",
                        pulse_style="cyan"
                    ),
                    CompactPercentColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(compact=True, elapsed_when_finished=True),
                    console=self.console,
                    expand=False,
                    transient=False
                )

                self.progress.start()

                # Single task
                self.overall_task_id = self.progress.add_task(
                    "🚀 Reprise",
                    total=100,
                    completed=max(0.0, min(self.state.current_progress, 100.0))
                )
                self.subtask_task_id = self.progress.add_task(
                    "Subtask: Reprise",
                    total=100,
                    completed=0
                )

                # Refresh current display immediately
                self.progress.update(
                    self.overall_task_id,
                    completed=max(0.0, min(self.state.current_progress, 100.0)),
                    description=f"{self.state.current_phase or 'Progress'}: {self.state.current_message}"
                )
                self.progress.update(
                    self.subtask_task_id,
                    completed=0,
                    description="Subtask: En attente"
                )

                del self._paused

            except Exception as e:
                self.console.print(f"[red]Warning: Could not resume progress: {e}[/red]")

    def stop(self):
        """Stop the display"""
        if self.is_running:
            if self.progress and self.overall_task_id is not None:
                self.progress.update(
                    self.overall_task_id,
                    completed=100,
                    description="✨ Pipeline Complete"
                )

            time.sleep(0.5)

            if self.progress:
                self.progress.stop()

            self.is_running = False
            self.progress = None
            self.overall_task_id = None
            self.subtask_task_id = None

    def update_progress(self, phase: str, progress: float, message: str,
                       subtask: Optional[Dict[str, Any]] = None,
                       json_sample: Optional[Dict] = None,
                       error: Optional[str] = None):
        """Update progress with unified display"""
        with self.lock:
            if not self.is_running or not self.progress:
                return

            # Update state
            self.state.current_phase = phase
            self.state.current_progress = progress
            self.state.current_message = message

            # Get phase info
            phase_info = self.PHASES.get(phase, {})
            icon = phase_info.get('icon', '•')
            label = phase_info.get('label', phase.title())

            # Build description - combine main and subtask info
            desc = f"{icon} {label}: {message}"

            # Add subtask info directly to main description
            if phase == 'annotation' and subtask:
                current = subtask.get('current', 0)
                total = subtask.get('total', 100)

                self.state.completed_items = current
                self.state.total_items = total

                # Append subtask progress to main description
                if current > 0:
                    desc = f"{icon} {label}: {message} [{current}/{total}]"

                # Handle JSON sample
                sample_data = subtask.get('json_data', json_sample)
                if sample_data:
                    self.current_sample = sample_data
                    if current > 0 and current % self.show_json_every == 0:
                        self.state.json_count += 1
                        self.state.last_json_sample = sample_data

                        # Update preview panel in place
                        if current - self.last_json_display >= self.show_json_every:
                            self._show_json_panel(sample_data)
                            self.last_json_display = current

            # Add error count if any
            if self.state.error_count > 0:
                desc += f" [red]({self.state.error_count} errors)[/red]"

            # Update overall progress bar
            if self.overall_task_id is not None:
                overall_complete = max(0.0, min(progress, 100.0))
                self.progress.update(
                    self.overall_task_id,
                    completed=overall_complete,
                    description=desc
                )

            # Update subtask bar
            if self.subtask_task_id is not None:
                if phase == 'annotation' and subtask and subtask.get('total'):
                    current = subtask.get('current', 0)
                    total = subtask.get('total', 1)
                    sub_pct = max(0.0, min(100.0, (current / max(total, 1)) * 100))
                    sub_desc = f"Subtask: {subtask.get('message', 'Progress')} [{current}/{total}]"
                    self.progress.update(
                        self.subtask_task_id,
                        completed=sub_pct,
                        description=sub_desc
                    )
                else:
                    self.progress.update(
                        self.subtask_task_id,
                        completed=0,
                        description="Subtask: En attente"
                    )

            # Handle errors
            if error:
                self.state.errors.append(error)
                self.state.error_count += 1
                self.recent_errors.append(error)
                if len(self.recent_errors) > 5:
                    self.recent_errors.pop(0)

    def _show_json_panel(self, json_data: Dict):
        """Display or refresh the JSON preview panel"""
        self.state.last_json_sample = json_data

        try:
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            lines = json_str.split('\n')
            panel_width = 80
            if lines:
                longest_line = max(len(line) for line in lines)
                panel_width = min(max(60, longest_line + 4), 140)

            syntax = Syntax(
                json_str,
                "json",
                theme="monokai",
                line_numbers=False,
                background_color="default"
            )

            panel = Panel(
                syntax,
                title=f"[bold cyan]📝 Preview #{self.state.json_count}[/bold cyan]",
                border_style="cyan",
                expand=False,
                width=panel_width
            )

            target_console = self.progress.console if self.progress is not None else self.console

            if not hasattr(self, '_preview_live'):
                from rich.live import Live
                self._preview_live = Live(panel, console=target_console, refresh_per_second=4)
                self._preview_live.start()
            else:
                self._preview_live.update(panel, refresh=True)

        except Exception:
            pass  # Silently fail to avoid disrupting progress
        finally:
            if hasattr(self, '_preview_live') and getattr(self._preview_live, 'stopped', False):
                # Live may auto-stop if console errors; reset handler
                del self._preview_live

    def _print_alert_summary(self):
        """Render a consolidated warning/error panel"""
        has_warnings = bool(self.recent_warnings)
        has_errors = bool(self.recent_errors)

        if not has_warnings and not has_errors:
            body = Text("No warnings or errors", style="dim")
        else:
            table = Table.grid(padding=(0, 1))
            table.add_column(justify="left", width=2)
            table.add_column()

            for msg in self.recent_warnings[-3:]:
                table.add_row("[yellow]⚠[/yellow]", msg)

            for msg in self.recent_errors[-3:]:
                table.add_row("[red]✖[/red]", msg)

            body = table

        panel = Panel(
            body,
            title="[bold yellow]Alerts[/bold yellow]",
            border_style="yellow",
            expand=False,
            width=80
        )

        target_console = self.progress.console if self.progress is not None else self.console
        target_console.print(panel)

    def clear_subtask(self, subtask_name: str):
        """Clear subtask (no-op since we don't have separate subtask)"""
        pass

    def show_error(self, error: str, item_info: Optional[str] = None):
        """Display error message in a panel without disrupting progress bars"""
        error_msg = error
        if item_info:
            error_msg = f"{item_info}: {error}"

        self.state.errors.append(error_msg)
        self.state.error_count += 1
        self.recent_errors.append(error_msg)
        if len(self.recent_errors) > 5:
            self.recent_errors.pop(0)

        # Display error in panel
        try:
            self._print_alert_summary()
        except:
            pass  # Silently fail to avoid disrupting progress

    def show_warning(self, warning: str, item_info: Optional[str] = None):
        """Display warning message in a panel without disrupting progress bars"""
        warning_msg = warning
        if item_info:
            warning_msg = f"{item_info}: {warning}"

        self.state.warnings.append(warning_msg)
        self.recent_warnings.append(warning_msg)
        if len(self.recent_warnings) > 5:
            self.recent_warnings.pop(0)

        # Display warning in panel
        try:
            self._print_alert_summary()
        except:
            pass  # Silently fail to avoid disrupting progress

    def show_final_stats(self):
        """Show final statistics with samples"""
        if not self.state.completed_items:
            return

        elapsed = time.time() - self.state.start_time

        # Create stats table
        stats = Table.grid(padding=1)
        stats.add_column(style="dim")
        stats.add_column(style="bright_white")

        stats.add_row("Items processed:", str(self.state.completed_items))
        stats.add_row("JSON samples:", str(self.state.json_count))
        if self.state.error_count > 0:
            stats.add_row("Errors:", f"[red]{self.state.error_count}[/red]")
        if self.state.warnings:
            stats.add_row("Warnings:", f"[yellow]{len(self.state.warnings)}[/yellow]")
        stats.add_row("Total time:", f"{elapsed:.1f}s")

        self.console.print()
        self.console.print(
            Panel(
                stats,
                title="[bold yellow]📊 Pipeline Summary[/bold yellow]",
                border_style="yellow",
                expand=False
            )
        )

        # Show annotation samples
        if self.state.annotation_samples:
            self.console.print("\n[bold cyan]📝 Sample Annotations:[/bold cyan]")
            for i, sample in enumerate(self.state.annotation_samples[-3:], 1):
                json_str = json.dumps(sample, indent=2, ensure_ascii=False)
                lines = json_str.split('\n')
                if len(lines) > 6:
                    json_str = '\n'.join(lines[:5]) + '\n  ...\n}'

                syntax = Syntax(
                    json_str,
                    "json",
                    theme="monokai",
                    line_numbers=False,
                    background_color="default"
                )

                self.console.print(
                    Panel(
                        syntax,
                        title=f"Sample {i}",
                        border_style="dim cyan",
                        expand=False,
                        width=60
                    )
                )

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.is_running:
            if not exc_type:
                self.show_final_stats()
            self.stop()

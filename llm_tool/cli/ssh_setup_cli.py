#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
ssh_setup_cli.py

MAIN OBJECTIVE:
---------------
Interactive CLI for setting up SSH remote machines for distributed training.
Guides the user through connecting remote machines, testing connectivity,
detecting resources, and configuring distributed training.

Dependencies:
-------------
- rich (Console, Panel, Table, Prompt, Confirm, Status)
- llm_tool.trainers.ssh_remote_worker

Author:
-------
Antoine Lemor
"""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from llm_tool.trainers.ssh_remote_worker import SSHRemoteManager, SSHRemoteMachine

logger = logging.getLogger(__name__)


class SSHSetupCLI:
    """
    Interactive CLI for setting up SSH remote machines for distributed training.

    Guides the user through:
    1. Asking if they want to use remote machines
    2. Entering SSH hostnames
    3. Testing connectivity
    4. Detecting remote resources (GPU, CPU, RAM)
    5. Displaying detected capabilities
    6. Adding multiple machines
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console(force_terminal=True, markup=True, highlight=False)
        self.ssh_manager = SSHRemoteManager(console=self.console)

    def prompt_for_remote_machines(self) -> Optional[SSHRemoteManager]:
        """
        Interactive prompt to set up SSH remote machines.

        Returns
        -------
        Optional[SSHRemoteManager]
            Configured SSH manager with connected machines, or None if user declined
        """
        self.console.print()

        # Ask if user wants to use remote machines
        use_remote = Confirm.ask(
            "[bold]Voulez-vous connecter des machines distantes via SSH ?[/bold]",
            default=False,
        )

        if not use_remote:
            self.console.print("[dim]Entrainement local uniquement.[/dim]\n")
            return None

        self.console.print()

        # Loop to add machines
        while True:
            hostname = Prompt.ask(
                "[bold]Nom d'hote SSH[/bold] [dim](ex: macbook-server)[/dim]"
            ).strip()

            if not hostname:
                self.console.print("[yellow]Nom d'hote vide, veuillez reessayer.[/yellow]")
                continue

            # Test connection
            machine = self._setup_machine(hostname)

            if machine:
                self.ssh_manager.add_machine(machine)
                self.display_remote_resources(machine)

            self.console.print()

            # Ask to add another
            add_more = Confirm.ask(
                "[bold]Ajouter une autre machine ?[/bold]",
                default=False,
            )

            if not add_more:
                break

            self.console.print()

        # Summary
        if self.ssh_manager.machines:
            self._display_summary()
            return self.ssh_manager
        else:
            self.console.print("[yellow]Aucune machine distante configuree. Entrainement local uniquement.[/yellow]\n")
            return None

    def _setup_machine(self, hostname: str) -> Optional[SSHRemoteMachine]:
        """
        Test connection, detect resources, and find project path for a host.

        Parameters
        ----------
        hostname : str
            SSH hostname

        Returns
        -------
        Optional[SSHRemoteMachine]
            Configured machine, or None if setup failed
        """
        # Test SSH connection
        with self.console.status(f"[cyan]Test de la connexion SSH vers {hostname}...[/cyan]"):
            success, message = self.ssh_manager.test_connection(hostname)

        if not success:
            self.console.print(f"[red]Connexion echouee vers {hostname}: {message}[/red]")
            self.console.print("[dim]Verifiez que le nom d'hote est correct et que SSH est configure.[/dim]")
            return None

        self.console.print(f"[green]Connexion SSH vers {hostname} reussie.[/green]")

        # Detect resources
        with self.console.status(f"[cyan]Detection des ressources sur {hostname}...[/cyan]"):
            machine = self.ssh_manager.detect_remote_resources(hostname)

        if not machine:
            self.console.print(f"[red]Detection des ressources echouee sur {hostname}.[/red]")
            self.console.print("[dim]Verifiez que LLM_Tool est installe sur la machine distante.[/dim]")
            return None

        # Find project path
        with self.console.status(f"[cyan]Recherche du projet LLM_Tool sur {hostname}...[/cyan]"):
            project_path = self.ssh_manager.find_project_path(hostname)

        if project_path:
            machine.project_path = project_path
            self.console.print(f"[green]Projet LLM_Tool trouve: {project_path}[/green]")
        else:
            self.console.print(
                f"[yellow]Projet LLM_Tool non trouve sur {hostname}. "
                f"Le worker sera lance sans chemin de projet.[/yellow]"
            )

        return machine

    def display_remote_resources(self, machine: SSHRemoteMachine):
        """
        Display detected resources for a remote machine.

        Parameters
        ----------
        machine : SSHRemoteMachine
            The machine to display
        """
        # Build resource info
        lines = []
        lines.append(f"[bold]Host:[/bold] {machine.hostname}")

        if machine.has_gpu:
            gpu_info = machine.gpu_name or machine.gpu_type.upper()
            if machine.gpu_memory_gb > 0:
                gpu_info += f" ({machine.gpu_memory_gb:.0f}GB VRAM)"
            lines.append(f"[bold green]GPU:[/bold green] {gpu_info}")
        else:
            lines.append("[bold yellow]GPU:[/bold yellow] Aucun")

        lines.append(f"[bold]CPU:[/bold] {machine.cpu_cores} cores")
        lines.append(f"[bold]RAM:[/bold] {machine.memory_gb:.0f}GB total, {machine.available_memory_gb:.0f}GB disponible")

        border_style = "green" if machine.has_gpu else "cyan"
        self.console.print(
            Panel(
                "\n".join(lines),
                title=f"Ressources de {machine.hostname}",
                border_style=border_style,
            )
        )

    def display_gpu_only_warning(self):
        """
        Display a warning that GPU-only mode will be used.
        Called when at least one remote machine has a GPU.
        """
        self.console.print(
            Panel(
                "[bold yellow]GPU detecte sur machine(s) distante(s) ![/bold yellow]\n\n"
                "L'entrainement sera distribue [bold]uniquement entre les GPUs[/bold] "
                "de toutes les machines.\n"
                "[dim]Aucun CPU worker ne sera utilise sur aucune machine.\n"
                "Cela garantit une utilisation optimale des ressources GPU.[/dim]",
                title="Mode GPU-Only",
                border_style="yellow",
            )
        )
        self.console.print()

    def _display_summary(self):
        """Display a summary of all configured remote machines."""
        self.console.print()

        table = Table(
            title="Machines distantes configurees",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
        )
        table.add_column("Machine", style="cyan")
        table.add_column("GPU", style="white")
        table.add_column("CPU", justify="center")
        table.add_column("RAM", justify="center")
        table.add_column("Statut", style="green")

        for machine in self.ssh_manager.machines:
            gpu_str = machine.gpu_name or machine.gpu_type.upper() if machine.has_gpu else "-"
            table.add_row(
                machine.hostname,
                gpu_str,
                f"{machine.cpu_cores} cores",
                f"{machine.memory_gb:.0f}GB",
                "Connecte" if machine.connected else "Erreur",
            )

        self.console.print(table)

        # GPU-only mode notice
        if self.ssh_manager.any_remote_has_gpu():
            self.console.print(
                "\n[yellow]Mode GPU-Only: l'entrainement sera distribue uniquement entre les GPUs.[/yellow]"
            )

        self.console.print()

#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
distributed_training_orchestrator.py

MAIN OBJECTIVE:
---------------
Central orchestrator for distributed SSH training. Coordinates training
across the local machine and one or more remote machines connected via SSH.

ARCHITECTURE:
-------------
    Orchestrator
        ├── Local Training (existing HybridParallelTrainer)
        │   └── GPU + CPU workers (or GPU-only if remote has GPU)
        └── Remote Training (SSH)
            ├── Machine 1: SCP data → SSH train → SCP results
            ├── Machine 2: SCP data → SSH train → SCP results
            └── ...

CRITICAL RULE:
--------------
If ANY remote machine has a GPU, all training is distributed ONLY across GPUs
(local GPU + remote GPUs). No CPU workers are used on any machine.

Dependencies:
-------------
- llm_tool.trainers.ssh_remote_worker
- llm_tool.trainers.hybrid_parallel_trainer
- threading

Author:
-------
Antoine Lemor
"""

import json
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple

from llm_tool.trainers.ssh_remote_worker import (
    SSHRemoteManager,
    SSHRemoteMachine,
    RemoteTaskAssignment,
    REMOTE_TRAINING_DIR,
)

logger = logging.getLogger(__name__)

# Polling interval for remote progress monitoring (seconds)
REMOTE_POLL_INTERVAL = 5.0

# Maximum wait time for a remote training to complete (seconds)
REMOTE_TRAINING_TIMEOUT = 3600 * 12  # 12 hours

# Retry count for SSH polling failures before marking as failed
REMOTE_POLL_MAX_FAILURES = 6  # 6 failures × 5s = 30s without contact


@dataclass
class DistributedResourcePlan:
    """Unified resource plan across all machines."""
    local_tasks: List[Any]                # TrainingTask objects for local execution
    remote_assignments: List[RemoteTaskAssignment]  # Tasks assigned to remote machines
    gpu_only_mode: bool                   # True if any remote has GPU
    total_machines: int                   # Total number of machines (local + remote)
    total_gpu_workers: int                # Number of GPU workers across all machines
    total_cpu_workers: int                # Number of CPU workers (0 in gpu_only_mode)
    local_gpu_available: bool             # Whether local machine has GPU
    local_cpu_workers: int                # Number of local CPU workers


class DistributedTrainingOrchestrator:
    """
    Orchestrates training across local + remote machines.

    Handles:
    - Task distribution based on available resources
    - Data transfer to/from remote machines
    - Parallel execution (local + remote simultaneously)
    - Progress monitoring for remote tasks
    - Result collection and aggregation
    """

    def __init__(
        self,
        ssh_manager: SSHRemoteManager,
        config: Optional[Any] = None,  # ParallelTrainingConfig
        console: Optional[Any] = None,
    ):
        self.ssh_manager = ssh_manager
        self.config = config
        self.console = console

        # Remote execution state
        self._remote_processes: Dict[str, Any] = {}  # task_id -> subprocess.Popen
        self._remote_results: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._stop_monitoring = threading.Event()

    # ========================================================================
    # RESOURCE PLANNING
    # ========================================================================

    def create_distributed_plan(
        self,
        tasks: List[Any],  # List[TrainingTask]
        local_plan: Any,    # ResourcePlan
    ) -> DistributedResourcePlan:
        """
        Create a distributed resource plan that assigns tasks across all machines.

        GPU-only rule: If any remote machine has a GPU, all training happens
        exclusively on GPUs (local + remote). No CPU workers on any machine.

        Parameters
        ----------
        tasks : List[TrainingTask]
            All training tasks to distribute
        local_plan : ResourcePlan
            Local resource plan from HardwareAnalyzer

        Returns
        -------
        DistributedResourcePlan
            Complete plan with task assignments
        """
        gpu_only_mode = self.ssh_manager.any_remote_has_gpu()
        remote_machines = self.ssh_manager.machines
        local_has_gpu = local_plan.gpu_allocation is not None

        total_tasks = len(tasks)

        if gpu_only_mode:
            # GPU-ONLY MODE: distribute tasks among GPUs only
            return self._plan_gpu_only(tasks, local_plan, remote_machines, local_has_gpu)
        else:
            # MIXED MODE: local GPU+CPU, remote CPUs
            return self._plan_mixed(tasks, local_plan, remote_machines, local_has_gpu)

    def _plan_gpu_only(
        self,
        tasks: List[Any],
        local_plan: Any,
        remote_machines: List[SSHRemoteMachine],
        local_has_gpu: bool,
    ) -> DistributedResourcePlan:
        """
        Plan task distribution for GPU-only mode.
        Distributes tasks proportionally based on GPU memory.
        """
        # Count total GPU workers
        gpu_machines = []
        if local_has_gpu:
            local_gpu_mem = local_plan.gpu_allocation.memory_limit_gb
            gpu_machines.append(("localhost", local_gpu_mem, None))

        for machine in remote_machines:
            if machine.has_gpu:
                gpu_machines.append((machine.hostname, machine.gpu_memory_gb, machine))

        total_gpus = len(gpu_machines)
        total_tasks = len(tasks)

        if total_gpus == 0:
            # No GPUs at all - fallback to all local
            return DistributedResourcePlan(
                local_tasks=list(tasks),
                remote_assignments=[],
                gpu_only_mode=True,
                total_machines=1,
                total_gpu_workers=0,
                total_cpu_workers=0,
                local_gpu_available=False,
                local_cpu_workers=0,
            )

        # Distribute tasks proportionally by GPU memory
        total_gpu_mem = sum(mem for _, mem, _ in gpu_machines)
        task_distribution = []
        remaining_tasks = total_tasks

        for i, (hostname, gpu_mem, machine) in enumerate(gpu_machines):
            if i == len(gpu_machines) - 1:
                # Last GPU gets remaining tasks (handles rounding)
                share = remaining_tasks
            else:
                share = max(1, round(total_tasks * (gpu_mem / total_gpu_mem)))
                share = min(share, remaining_tasks)
            task_distribution.append((hostname, share, machine))
            remaining_tasks -= share

        # Assign tasks
        local_tasks = []
        remote_assignments = []
        task_idx = 0

        for hostname, share, machine in task_distribution:
            for _ in range(share):
                if task_idx >= len(tasks):
                    break
                task = tasks[task_idx]

                if machine is None:
                    # Local task
                    local_tasks.append(task)
                else:
                    # Remote task
                    category = task.category_name
                    remote_dir = f"{REMOTE_TRAINING_DIR}/{category}"
                    assignment = RemoteTaskAssignment(
                        task_id=task.task_id,
                        category_name=category,
                        model_name=task.model_name,
                        data_path=task.data_path,
                        remote_machine=machine,
                        remote_data_path=f"{remote_dir}/data.csv",
                        remote_output_path=f"{remote_dir}/output",
                        remote_config_path=f"{remote_dir}/task_config.json",
                        remote_progress_path=f"{remote_dir}/progress.json",
                        config=task.config,
                    )
                    remote_assignments.append(assignment)
                task_idx += 1

        logger.info(
            f"GPU-only plan: {len(local_tasks)} local tasks, "
            f"{len(remote_assignments)} remote tasks across {total_gpus} GPUs"
        )

        return DistributedResourcePlan(
            local_tasks=local_tasks,
            remote_assignments=remote_assignments,
            gpu_only_mode=True,
            total_machines=1 + len([m for m in remote_machines if m.has_gpu]),
            total_gpu_workers=total_gpus,
            total_cpu_workers=0,
            local_gpu_available=local_has_gpu,
            local_cpu_workers=0,
        )

    def _plan_mixed(
        self,
        tasks: List[Any],
        local_plan: Any,
        remote_machines: List[SSHRemoteMachine],
        local_has_gpu: bool,
    ) -> DistributedResourcePlan:
        """
        Plan task distribution for mixed mode (no remote GPUs).
        Local machine uses GPU + CPU workers, remotes use CPUs.
        """
        total_tasks = len(tasks)

        # Calculate remote CPU capacity
        # Rough estimate: 1 task per 4 CPU cores on remote
        remote_capacity = 0
        for machine in remote_machines:
            machine_tasks = max(1, machine.cpu_cores // 4)
            remote_capacity += machine_tasks

        # Local handles the bulk (GPU is much faster)
        local_share = max(1, total_tasks - remote_capacity)
        remote_share = total_tasks - local_share

        # Assign local tasks
        local_tasks = list(tasks[:local_share])

        # Distribute remote tasks across machines
        remote_assignments = []
        task_idx = local_share

        for machine in remote_machines:
            machine_capacity = max(1, machine.cpu_cores // 4)
            for _ in range(machine_capacity):
                if task_idx >= len(tasks):
                    break
                task = tasks[task_idx]
                category = task.category_name
                remote_dir = f"{REMOTE_TRAINING_DIR}/{category}"
                assignment = RemoteTaskAssignment(
                    task_id=task.task_id,
                    category_name=category,
                    model_name=task.model_name,
                    data_path=task.data_path,
                    remote_machine=machine,
                    remote_data_path=f"{remote_dir}/data.csv",
                    remote_output_path=f"{remote_dir}/output",
                    remote_config_path=f"{remote_dir}/task_config.json",
                    remote_progress_path=f"{remote_dir}/progress.json",
                    config=task.config,
                )
                remote_assignments.append(assignment)
                task_idx += 1

        local_cpu_workers = len(local_plan.cpu_allocations) if local_plan else 0

        logger.info(
            f"Mixed plan: {len(local_tasks)} local tasks (GPU+{local_cpu_workers} CPU), "
            f"{len(remote_assignments)} remote tasks across {len(remote_machines)} machines"
        )

        return DistributedResourcePlan(
            local_tasks=local_tasks,
            remote_assignments=remote_assignments,
            gpu_only_mode=False,
            total_machines=1 + len(remote_machines),
            total_gpu_workers=1 if local_has_gpu else 0,
            total_cpu_workers=local_cpu_workers + len(remote_assignments),
            local_gpu_available=local_has_gpu,
            local_cpu_workers=local_cpu_workers,
        )

    # ========================================================================
    # DISPLAY
    # ========================================================================

    def display_distributed_plan(self, plan: DistributedResourcePlan):
        """
        Display the distributed resource plan using Rich tables.

        Parameters
        ----------
        plan : DistributedResourcePlan
            The plan to display
        """
        if not self.console:
            return

        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        # Mode banner
        if plan.gpu_only_mode:
            self.console.print(
                Panel(
                    "[bold yellow]GPU-ONLY MODE[/bold yellow]\n\n"
                    "GPU detecte sur machine(s) distante(s).\n"
                    "L'entrainement est distribue uniquement entre les GPUs.\n"
                    "[dim]Aucun CPU worker ne sera utilise.[/dim]",
                    title="Distributed Training",
                    border_style="yellow",
                )
            )
        else:
            self.console.print(
                Panel(
                    "[bold cyan]MIXED MODE[/bold cyan]\n\n"
                    "Pas de GPU distant detecte.\n"
                    "GPU + CPU locaux + CPU distants seront utilises.",
                    title="Distributed Training",
                    border_style="cyan",
                )
            )

        # Resource table
        table = Table(
            title="Plan de distribution des taches",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
        )
        table.add_column("Machine", style="cyan")
        table.add_column("Device", style="white")
        table.add_column("Type", style="white")
        table.add_column("Taches", justify="center", style="green")

        # Local machine
        if plan.local_gpu_available:
            table.add_row(
                "localhost",
                "GPU",
                "GPU (Priorite)",
                str(len(plan.local_tasks)),
            )
        else:
            table.add_row(
                "localhost",
                "CPU",
                f"CPU ({plan.local_cpu_workers} workers)",
                str(len(plan.local_tasks)),
            )

        # Remote machines
        remote_by_machine: Dict[str, List[RemoteTaskAssignment]] = {}
        for assignment in plan.remote_assignments:
            hostname = assignment.remote_machine.hostname
            if hostname not in remote_by_machine:
                remote_by_machine[hostname] = []
            remote_by_machine[hostname].append(assignment)

        for hostname, assignments in remote_by_machine.items():
            machine = assignments[0].remote_machine
            if machine.has_gpu:
                device_type = f"{machine.gpu_type.upper()}"
                device_label = f"GPU ({machine.gpu_name})" if machine.gpu_name else "GPU (Distant)"
            else:
                device_type = "CPU"
                device_label = f"CPU ({machine.cpu_cores} cores)"

            table.add_row(
                hostname,
                device_type,
                device_label,
                str(len(assignments)),
            )

        self.console.print(table)

        total_tasks = len(plan.local_tasks) + len(plan.remote_assignments)
        self.console.print(
            f"\n[bold]Total: {total_tasks} taches sur {plan.total_machines} machine(s)[/bold]"
        )
        self.console.print()

    # ========================================================================
    # EXECUTION
    # ========================================================================

    def execute(
        self,
        plan: DistributedResourcePlan,
        progress_callback: Optional[Callable] = None,
        local_trainer_factory: Optional[Callable] = None,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Execute the distributed training plan.

        Runs local and remote training in parallel:
        - Local tasks use the existing HybridParallelTrainer
        - Remote tasks are executed via SSH

        Parameters
        ----------
        plan : DistributedResourcePlan
            The execution plan
        progress_callback : Callable, optional
            Callback for progress updates
        local_trainer_factory : Callable, optional
            Factory function to create the local trainer.
            If None, a default HybridParallelTrainer is created.

        Returns
        -------
        Tuple[List[TaskResult], List[RemoteTaskAssignment]]
            (local_results, completed_remote_assignments)
        """
        self._stop_monitoring.clear()

        # Step 1: Transfer data to remote machines
        if plan.remote_assignments:
            if self.console:
                self.console.print("[cyan]Transfert des donnees vers les machines distantes...[/cyan]")
            self._transfer_data_to_remotes(plan)

        # Step 2: Start remote training
        if plan.remote_assignments:
            if self.console:
                self.console.print("[cyan]Lancement de l'entrainement distant...[/cyan]")
            self._start_remote_training(plan)

        # Step 3: Start remote progress monitoring thread
        monitor_thread = None
        if plan.remote_assignments:
            monitor_thread = threading.Thread(
                target=self._monitor_remote_tasks,
                args=(plan.remote_assignments, progress_callback),
                daemon=True,
            )
            monitor_thread.start()

        # Step 4: Run local training
        local_results = []
        if plan.local_tasks:
            if self.console:
                self.console.print(f"[cyan]Entrainement local de {len(plan.local_tasks)} taches...[/cyan]\n")
            local_results = self._run_local_training(
                plan.local_tasks,
                plan,
                progress_callback,
                local_trainer_factory,
            )

        # Step 5: Wait for remote training to finish
        if plan.remote_assignments:
            if self.console:
                self.console.print("\n[cyan]Attente de la fin des entrainements distants...[/cyan]")
            self._wait_for_remote_completion(plan.remote_assignments)

        # Step 6: Stop monitoring
        self._stop_monitoring.set()
        if monitor_thread:
            monitor_thread.join(timeout=10)

        # Step 7: Collect remote results
        if plan.remote_assignments:
            if self.console:
                self.console.print("[cyan]Collecte des resultats distants...[/cyan]")
            self._collect_all_remote_results(plan)

        # Step 8: Cleanup remote machines
        self._cleanup_all_remotes(plan)

        return local_results, plan.remote_assignments

    def _transfer_data_to_remotes(self, plan: DistributedResourcePlan):
        """Transfer training data and config to all remote machines."""
        for assignment in plan.remote_assignments:
            assignment.status = "transferring"

            # Transfer training data
            success, result = self.ssh_manager.transfer_training_data(
                assignment.remote_machine,
                assignment.data_path,
                assignment.category_name,
            )
            if not success:
                logger.error(
                    f"Failed to transfer data for {assignment.category_name} "
                    f"to {assignment.remote_machine.hostname}: {result}"
                )
                assignment.status = "failed"
                assignment.error_message = f"Data transfer failed: {result}"
                continue

            assignment.remote_data_path = result

            # Build task config for remote worker
            task_config = {
                "task_id": assignment.task_id,
                "category_name": assignment.category_name,
                "model_name": assignment.model_name,
                "data_path": assignment.remote_data_path,
                "output_path": assignment.remote_output_path,
                "progress_path": assignment.remote_progress_path,
                "config": assignment.config,
            }

            # Transfer task config
            success, result = self.ssh_manager.transfer_task_config(
                assignment.remote_machine,
                task_config,
                assignment.category_name,
            )
            if not success:
                logger.error(
                    f"Failed to transfer config for {assignment.category_name} "
                    f"to {assignment.remote_machine.hostname}: {result}"
                )
                assignment.status = "failed"
                assignment.error_message = f"Config transfer failed: {result}"
                continue

            assignment.remote_config_path = result
            logger.info(
                f"Data transferred for {assignment.category_name} "
                f"to {assignment.remote_machine.hostname}"
            )

    def _start_remote_training(self, plan: DistributedResourcePlan):
        """Start training processes on all remote machines."""
        for assignment in plan.remote_assignments:
            if assignment.status == "failed":
                continue

            process = self.ssh_manager.start_remote_training(assignment)
            if process:
                assignment.status = "running"
                self._remote_processes[assignment.task_id] = process
                logger.info(
                    f"Started remote training for {assignment.category_name} "
                    f"on {assignment.remote_machine.hostname}"
                )
            else:
                assignment.status = "failed"
                assignment.error_message = "Failed to start remote training process"

    def _monitor_remote_tasks(
        self,
        assignments: List[RemoteTaskAssignment],
        progress_callback: Optional[Callable],
    ):
        """
        Monitor remote task progress in a background thread.
        Polls progress.json on remote machines every REMOTE_POLL_INTERVAL seconds.
        """
        failure_counts: Dict[str, int] = {a.task_id: 0 for a in assignments}

        while not self._stop_monitoring.is_set():
            for assignment in assignments:
                if assignment.status not in ("running",):
                    continue

                # Check if SSH process has terminated
                process = self._remote_processes.get(assignment.task_id)
                if process and process.poll() is not None:
                    # Process finished
                    if process.returncode == 0:
                        assignment.status = "completed"
                    else:
                        stderr = process.stderr.read() if process.stderr else ""
                        assignment.status = "failed"
                        assignment.error_message = f"Remote process exited with code {process.returncode}: {stderr[:500]}"
                    continue

                # Poll progress
                progress = self.ssh_manager.poll_remote_progress(assignment)
                if progress:
                    failure_counts[assignment.task_id] = 0

                    # Forward progress to callback
                    if progress_callback and progress.get("status") == "training":
                        try:
                            from llm_tool.trainers.hybrid_parallel_trainer import ProgressUpdate
                            update = ProgressUpdate(
                                task_id=assignment.task_id,
                                worker_id=f"remote_{assignment.remote_machine.hostname}",
                                device_type="remote_gpu" if assignment.remote_machine.has_gpu else "remote_cpu",
                                category_name=assignment.category_name,
                                current_epoch=progress.get("current_epoch", 0),
                                total_epochs=progress.get("total_epochs", 0),
                                train_loss=progress.get("train_loss", 0.0),
                                val_loss=progress.get("val_loss", 0.0),
                                f1_score=progress.get("f1_score", 0.0),
                                accuracy=progress.get("accuracy", 0.0),
                            )
                            progress_callback(update)
                        except Exception as e:
                            logger.debug(f"Progress callback error: {e}")

                    # Check for remote completion
                    if progress.get("status") in ("completed", "failed"):
                        assignment.status = progress["status"]
                        if progress.get("status") == "failed":
                            assignment.error_message = progress.get("error", "Unknown remote error")
                else:
                    failure_counts[assignment.task_id] += 1
                    if failure_counts[assignment.task_id] >= REMOTE_POLL_MAX_FAILURES:
                        logger.warning(
                            f"Lost contact with {assignment.remote_machine.hostname} "
                            f"for {assignment.category_name} after "
                            f"{REMOTE_POLL_MAX_FAILURES} polling failures"
                        )
                        assignment.status = "failed"
                        assignment.error_message = "Lost SSH contact during training"

            # Check if all tasks are done
            all_done = all(
                a.status in ("completed", "failed")
                for a in assignments
            )
            if all_done:
                break

            self._stop_monitoring.wait(REMOTE_POLL_INTERVAL)

    def _wait_for_remote_completion(
        self,
        assignments: List[RemoteTaskAssignment],
        timeout: float = REMOTE_TRAINING_TIMEOUT,
    ):
        """Wait for all remote training processes to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_done = all(
                a.status in ("completed", "failed")
                for a in assignments
            )
            if all_done:
                return

            # Also check subprocess completion
            for assignment in assignments:
                if assignment.status != "running":
                    continue
                process = self._remote_processes.get(assignment.task_id)
                if process and process.poll() is not None:
                    if process.returncode == 0:
                        assignment.status = "completed"
                    else:
                        stderr = ""
                        try:
                            stderr = process.stderr.read() if process.stderr else ""
                        except Exception:
                            pass
                        assignment.status = "failed"
                        assignment.error_message = (
                            f"Remote process exited with code {process.returncode}"
                            f"{': ' + stderr[:200] if stderr else ''}"
                        )

            time.sleep(2.0)

        # Timeout reached - mark remaining as failed
        for assignment in assignments:
            if assignment.status == "running":
                assignment.status = "failed"
                assignment.error_message = f"Remote training timed out after {timeout}s"
                # Kill the SSH process
                process = self._remote_processes.get(assignment.task_id)
                if process:
                    try:
                        process.kill()
                    except Exception:
                        pass

    def _run_local_training(
        self,
        tasks: List[Any],
        plan: DistributedResourcePlan,
        progress_callback: Optional[Callable],
        local_trainer_factory: Optional[Callable],
    ) -> List[Any]:
        """
        Run training on local machine using existing HybridParallelTrainer.

        Parameters
        ----------
        tasks : List[TrainingTask]
            Tasks to train locally
        plan : DistributedResourcePlan
            The distributed plan
        progress_callback : Callable, optional
            Progress callback
        local_trainer_factory : Callable, optional
            Factory to create local trainer

        Returns
        -------
        List[TaskResult]
            Local training results
        """
        if not tasks:
            return []

        from llm_tool.trainers.hybrid_parallel_trainer import HybridParallelTrainer

        # In GPU-only mode, use 0 CPU workers
        num_cpu_workers = 0 if plan.gpu_only_mode else plan.local_cpu_workers

        if local_trainer_factory:
            trainer = local_trainer_factory(num_cpu_workers=num_cpu_workers)
        else:
            trainer = HybridParallelTrainer(
                model_name=tasks[0].model_name if tasks else "xlm-roberta-base",
                num_cpu_workers=num_cpu_workers,
            )

        results = trainer.train(tasks, progress_callback)
        return results

    def _collect_all_remote_results(self, plan: DistributedResourcePlan):
        """Collect results from all completed remote training tasks."""
        if not self.config:
            return

        # Determine local output directory
        output_dir = getattr(self.config, 'output_dir', 'models')
        session_id = getattr(self.config, 'session_id', '')
        local_base = Path(output_dir) / session_id / "parallel_training"
        local_base.mkdir(parents=True, exist_ok=True)

        for assignment in plan.remote_assignments:
            if assignment.status != "completed":
                continue

            local_output = str(local_base / assignment.category_name)
            success, results = self.ssh_manager.collect_results(
                assignment, local_output
            )

            if success:
                assignment.result_f1 = results.get("f1_score", 0.0)
                assignment.result_accuracy = results.get("accuracy", 0.0)
                assignment.result_best_epoch = results.get("best_epoch", 0)
                assignment.result_elapsed = results.get("elapsed_seconds", 0.0)
                assignment.result_model_path = results.get("local_model_path", "")
                logger.info(
                    f"Collected results for {assignment.category_name} from "
                    f"{assignment.remote_machine.hostname}: "
                    f"F1={assignment.result_f1:.4f}"
                )
            else:
                assignment.status = "failed"
                assignment.error_message = results.get("error", "Failed to collect results")
                logger.error(
                    f"Failed to collect results for {assignment.category_name} "
                    f"from {assignment.remote_machine.hostname}"
                )

    def _cleanup_all_remotes(self, plan: DistributedResourcePlan):
        """Clean up temporary files on all remote machines."""
        cleaned_machines = set()
        for assignment in plan.remote_assignments:
            hostname = assignment.remote_machine.hostname
            if hostname not in cleaned_machines:
                self.ssh_manager.cleanup_all_remote(assignment.remote_machine)
                cleaned_machines.add(hostname)

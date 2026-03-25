#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
ssh_remote_worker.py

MAIN OBJECTIVE:
---------------
Manages SSH connections to remote machines for distributed training.
Handles connectivity testing, remote resource detection, data transfer (SCP),
remote training execution, progress monitoring, and result collection.

ARCHITECTURE:
-------------
    Local Machine                    Remote Machine(s)
        |                                |
        |-- test_connection() ---------> |  (ssh echo ok)
        |-- detect_remote_resources() -> |  (ssh python3 -c "detect_resources()")
        |-- transfer_training_data() --> |  (scp data.csv)
        |-- start_remote_training() ---> |  (ssh python -m ssh_training_worker)
        |-- poll_remote_progress() ----> |  (ssh cat progress.json)
        |-- collect_results() <--------- |  (scp results + model)
        |-- cleanup_remote() ----------> |  (ssh rm -rf)

Dependencies:
-------------
- subprocess (system ssh/scp commands)
- json
- logging

Author:
-------
Antoine Lemor
"""

import os
import json
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Remote working directory for training data and results
REMOTE_TRAINING_DIR = "/tmp/llm_tool_training"

# SSH connection defaults
SSH_CONNECT_TIMEOUT = 10
SSH_COMMAND_TIMEOUT = 30
SCP_TIMEOUT = 300  # 5 minutes for large files


@dataclass
class SSHRemoteMachine:
    """Represents a remote machine connected via SSH."""
    hostname: str               # SSH host (e.g., "macbook-server")
    has_gpu: bool = False
    gpu_type: str = "none"      # "cuda", "mps", "rocm", "none"
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    project_path: str = ""      # Path to LLM_Tool installation on remote
    python_path: str = "python3"  # Python executable on remote
    connected: bool = False


@dataclass
class RemoteTaskAssignment:
    """A training task assigned to a remote machine."""
    task_id: str
    category_name: str
    model_name: str
    data_path: str              # Local path to training data
    remote_machine: SSHRemoteMachine
    remote_data_path: str = ""  # Where data was SCP'd to on remote
    remote_output_path: str = ""  # Where model will be saved on remote
    remote_config_path: str = ""  # task_config.json path on remote
    remote_progress_path: str = ""  # progress.json path on remote
    status: str = "pending"     # "pending", "transferring", "running", "completed", "failed"
    config: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    # Populated after collection
    result_f1: float = 0.0
    result_accuracy: float = 0.0
    result_best_epoch: int = 0
    result_elapsed: float = 0.0
    result_model_path: str = ""


class SSHRemoteManager:
    """
    Manages SSH connections, resource detection, and remote training execution.

    Uses system ssh/scp commands via subprocess for maximum compatibility
    with existing SSH configurations (~/.ssh/config, keys, etc.).
    """

    def __init__(self, console=None):
        self.console = console
        self.machines: List[SSHRemoteMachine] = []

    # ========================================================================
    # CONNECTION
    # ========================================================================

    def test_connection(self, hostname: str, timeout: int = SSH_CONNECT_TIMEOUT) -> Tuple[bool, str]:
        """
        Test SSH connectivity to a remote host.

        Parameters
        ----------
        hostname : str
            SSH hostname (can be an alias from ~/.ssh/config)
        timeout : int
            Connection timeout in seconds

        Returns
        -------
        Tuple[bool, str]
            (success, message)
        """
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={timeout}",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=accept-new",
                    hostname,
                    "echo", "ok",
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )

            if result.returncode == 0 and "ok" in result.stdout.strip():
                logger.info(f"SSH connection to {hostname} successful")
                return True, "Connection successful"
            else:
                error_msg = result.stderr.strip() or "Unknown error"
                logger.warning(f"SSH connection to {hostname} failed: {error_msg}")
                return False, f"Connection failed: {error_msg}"

        except subprocess.TimeoutExpired:
            logger.warning(f"SSH connection to {hostname} timed out after {timeout}s")
            return False, f"Connection timed out after {timeout}s"
        except FileNotFoundError:
            logger.error("SSH command not found on this system")
            return False, "SSH command not found"
        except Exception as e:
            logger.error(f"SSH connection error: {e}")
            return False, f"Error: {e}"

    def _ssh_exec(
        self,
        hostname: str,
        command: str,
        timeout: int = SSH_COMMAND_TIMEOUT,
    ) -> Tuple[bool, str, str]:
        """
        Execute a command on a remote host via SSH.

        Returns
        -------
        Tuple[bool, str, str]
            (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                    "-o", "BatchMode=yes",
                    hostname,
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0
            return success, result.stdout.strip(), result.stderr.strip()

        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def _ssh_exec_script(
        self,
        hostname: str,
        script: str,
        timeout: int = SSH_COMMAND_TIMEOUT,
    ) -> Tuple[bool, str, str]:
        """
        Execute a Python script on a remote host via SSH stdin.

        Pipes the script to python3 via stdin to avoid shell quoting issues.

        Returns
        -------
        Tuple[bool, str, str]
            (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                    "-o", "BatchMode=yes",
                    hostname,
                    "python3",
                ],
                input=script,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0
            return success, result.stdout.strip(), result.stderr.strip()

        except subprocess.TimeoutExpired:
            return False, "", f"Script timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def _scp_to_remote(
        self,
        hostname: str,
        local_path: str,
        remote_path: str,
        recursive: bool = False,
        timeout: int = SCP_TIMEOUT,
    ) -> Tuple[bool, str]:
        """
        Copy a file or directory to a remote host via SCP.

        Returns
        -------
        Tuple[bool, str]
            (success, error_message)
        """
        cmd = ["scp", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", "-o", "BatchMode=yes"]
        if recursive:
            cmd.append("-r")
        cmd.extend([local_path, f"{hostname}:{remote_path}"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip() or "SCP failed"
        except subprocess.TimeoutExpired:
            return False, f"SCP timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def _scp_from_remote(
        self,
        hostname: str,
        remote_path: str,
        local_path: str,
        recursive: bool = False,
        timeout: int = SCP_TIMEOUT,
    ) -> Tuple[bool, str]:
        """
        Copy a file or directory from a remote host via SCP.

        Returns
        -------
        Tuple[bool, str]
            (success, error_message)
        """
        cmd = ["scp", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", "-o", "BatchMode=yes"]
        if recursive:
            cmd.append("-r")
        cmd.extend([f"{hostname}:{remote_path}", local_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip() or "SCP failed"
        except subprocess.TimeoutExpired:
            return False, f"SCP timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    # ========================================================================
    # RESOURCE DETECTION
    # ========================================================================

    def detect_remote_resources(self, hostname: str) -> Optional[SSHRemoteMachine]:
        """
        Detect hardware resources on a remote machine via SSH.

        Executes a Python script on the remote that uses llm_tool's
        detect_resources() to gather GPU, CPU, and memory information.

        Parameters
        ----------
        hostname : str
            SSH hostname

        Returns
        -------
        Optional[SSHRemoteMachine]
            Machine info if detection succeeded, None otherwise
        """
        # Python script to detect resources and output JSON
        # Uses _ssh_exec_script to pipe via stdin (avoids shell quoting issues)
        # Tries llm_tool first, then falls back to basic torch/psutil detection
        detect_script = (
            "import json, sys, os\n"
            "info = {}\n"
            "try:\n"
            "    from llm_tool.utils.system_resources import detect_resources\n"
            "    res = detect_resources()\n"
            "    gpu_name = res.gpu.device_names[0] if res.gpu.device_names else ''\n"
            "    info = {\n"
            "        'gpu_available': res.gpu.available,\n"
            "        'gpu_type': res.gpu.device_type,\n"
            "        'gpu_name': gpu_name,\n"
            "        'gpu_memory_gb': res.gpu.total_memory_gb,\n"
            "        'cpu_cores': res.cpu.physical_cores,\n"
            "        'memory_total_gb': res.memory.total_gb,\n"
            "        'memory_available_gb': res.memory.available_gb,\n"
            "    }\n"
            "except Exception:\n"
            "    # Fallback: detect resources without llm_tool\n"
            "    info['cpu_cores'] = os.cpu_count() or 1\n"
            "    info['gpu_available'] = False\n"
            "    info['gpu_type'] = 'none'\n"
            "    info['gpu_name'] = ''\n"
            "    info['gpu_memory_gb'] = 0.0\n"
            "    info['memory_total_gb'] = 0.0\n"
            "    info['memory_available_gb'] = 0.0\n"
            "    try:\n"
            "        import torch\n"
            "        if torch.cuda.is_available():\n"
            "            info['gpu_available'] = True\n"
            "            info['gpu_type'] = 'cuda'\n"
            "            info['gpu_name'] = torch.cuda.get_device_name(0)\n"
            "            props = torch.cuda.get_device_properties(0)\n"
            "            info['gpu_memory_gb'] = props.total_mem / (1024**3)\n"
            "        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
            "            info['gpu_available'] = True\n"
            "            info['gpu_type'] = 'mps'\n"
            "            info['gpu_name'] = 'Apple Silicon (MPS)'\n"
            "    except Exception:\n"
            "        pass\n"
            "    try:\n"
            "        import psutil\n"
            "        mem = psutil.virtual_memory()\n"
            "        info['memory_total_gb'] = mem.total / (1024**3)\n"
            "        info['memory_available_gb'] = mem.available / (1024**3)\n"
            "    except Exception:\n"
            "        pass\n"
            "print('RESOURCES_JSON:' + json.dumps(info))\n"
        )

        success, stdout, stderr = self._ssh_exec_script(
            hostname,
            detect_script,
            timeout=30,
        )

        if not success:
            logger.warning(f"Resource detection on {hostname} failed: {stderr}")
            return None

        # Parse JSON from output
        for line in stdout.split("\n"):
            if line.startswith("RESOURCES_JSON:"):
                try:
                    info = json.loads(line[len("RESOURCES_JSON:"):])
                    machine = SSHRemoteMachine(
                        hostname=hostname,
                        has_gpu=info.get("gpu_available", False),
                        gpu_type=info.get("gpu_type", "none"),
                        gpu_name=info.get("gpu_name", ""),
                        gpu_memory_gb=info.get("gpu_memory_gb", 0.0),
                        cpu_cores=info.get("cpu_cores", 0),
                        memory_gb=info.get("memory_total_gb", 0.0),
                        available_memory_gb=info.get("memory_available_gb", 0.0),
                        connected=True,
                    )
                    logger.info(
                        f"Detected resources on {hostname}: "
                        f"GPU={machine.has_gpu} ({machine.gpu_type}), "
                        f"CPU={machine.cpu_cores} cores, "
                        f"RAM={machine.memory_gb:.1f}GB"
                    )
                    return machine
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse resource JSON from {hostname}: {e}")
                    return None

        logger.warning(f"No resource info found in output from {hostname}")
        return None

    def find_project_path(self, hostname: str) -> Optional[str]:
        """
        Find the LLM_Tool project path on a remote machine.

        Returns
        -------
        Optional[str]
            Path to the project directory, or None if not found
        """
        # Uses _ssh_exec_script to pipe via stdin (avoids shell quoting issues)
        find_script = (
            "import sys, os\n"
            "try:\n"
            "    import llm_tool\n"
            "    pkg_path = os.path.dirname(llm_tool.__file__)\n"
            "    project_path = os.path.dirname(pkg_path)\n"
            "    print('PROJECT_PATH:' + project_path)\n"
            "except ImportError:\n"
            "    print('PROJECT_NOT_FOUND', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )

        success, stdout, stderr = self._ssh_exec_script(
            hostname,
            find_script,
            timeout=15,
        )

        if not success:
            logger.warning(f"Could not find LLM_Tool on {hostname}: {stderr}")
            return None

        for line in stdout.split("\n"):
            if line.startswith("PROJECT_PATH:"):
                path = line[len("PROJECT_PATH:"):]
                logger.info(f"Found LLM_Tool on {hostname} at: {path}")
                return path

        return None

    def add_machine(self, machine: SSHRemoteMachine):
        """Add a validated remote machine to the managed list."""
        self.machines.append(machine)
        logger.info(f"Added remote machine: {machine.hostname}")

    # ========================================================================
    # DATA TRANSFER
    # ========================================================================

    def transfer_training_data(
        self,
        machine: SSHRemoteMachine,
        local_data_path: str,
        category: str,
    ) -> Tuple[bool, str]:
        """
        Transfer training data to a remote machine.

        Creates the remote directory structure and copies the data file.

        Parameters
        ----------
        machine : SSHRemoteMachine
            Target remote machine
        local_data_path : str
            Local path to the training data CSV
        category : str
            Category name (used for remote directory naming)

        Returns
        -------
        Tuple[bool, str]
            (success, remote_data_path or error_message)
        """
        remote_dir = f"{REMOTE_TRAINING_DIR}/{category}"
        remote_data_path = f"{remote_dir}/data.csv"

        # Create remote directory
        success, _, stderr = self._ssh_exec(
            machine.hostname,
            f"mkdir -p {remote_dir}",
        )
        if not success:
            return False, f"Failed to create remote directory: {stderr}"

        # Copy data file
        success, error = self._scp_to_remote(
            machine.hostname,
            local_data_path,
            remote_data_path,
        )
        if not success:
            return False, f"Failed to copy training data: {error}"

        logger.info(f"Transferred training data to {machine.hostname}:{remote_data_path}")
        return True, remote_data_path

    def transfer_task_config(
        self,
        machine: SSHRemoteMachine,
        task_config: Dict[str, Any],
        category: str,
    ) -> Tuple[bool, str]:
        """
        Transfer a task configuration JSON to a remote machine.

        Parameters
        ----------
        machine : SSHRemoteMachine
            Target remote machine
        task_config : dict
            Task configuration to serialize as JSON
        category : str
            Category name

        Returns
        -------
        Tuple[bool, str]
            (success, remote_config_path or error_message)
        """
        remote_dir = f"{REMOTE_TRAINING_DIR}/{category}"
        remote_config_path = f"{remote_dir}/task_config.json"

        # Write config to a local temp file, then SCP
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task_config, f, indent=2)
            local_tmp = f.name

        try:
            success, error = self._scp_to_remote(
                machine.hostname,
                local_tmp,
                remote_config_path,
            )
            if not success:
                return False, f"Failed to copy task config: {error}"

            logger.info(f"Transferred task config to {machine.hostname}:{remote_config_path}")
            return True, remote_config_path
        finally:
            os.unlink(local_tmp)

    # ========================================================================
    # REMOTE EXECUTION
    # ========================================================================

    def start_remote_training(
        self,
        assignment: RemoteTaskAssignment,
    ) -> Optional[subprocess.Popen]:
        """
        Start training on a remote machine via SSH.

        Launches the ssh_training_worker.py script on the remote machine
        as a non-blocking subprocess.

        Parameters
        ----------
        assignment : RemoteTaskAssignment
            The task assignment with remote paths

        Returns
        -------
        Optional[subprocess.Popen]
            The SSH subprocess, or None if launch failed
        """
        machine = assignment.remote_machine
        config_path = assignment.remote_config_path

        # Build the remote command
        # Use the project's python to ensure correct environment
        if machine.project_path:
            remote_cmd = (
                f"cd {machine.project_path} && "
                f"{machine.python_path} -m llm_tool.trainers.ssh_training_worker "
                f"--config {config_path}"
            )
        else:
            remote_cmd = (
                f"{machine.python_path} -m llm_tool.trainers.ssh_training_worker "
                f"--config {config_path}"
            )

        try:
            process = subprocess.Popen(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                    "-o", "BatchMode=yes",
                    machine.hostname,
                    remote_cmd,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.info(
                f"Started remote training on {machine.hostname} "
                f"for {assignment.category_name} (PID: {process.pid})"
            )
            return process

        except Exception as e:
            logger.error(f"Failed to start remote training on {machine.hostname}: {e}")
            return None

    def poll_remote_progress(
        self,
        assignment: RemoteTaskAssignment,
    ) -> Optional[Dict[str, Any]]:
        """
        Poll training progress from a remote machine.

        Reads the progress.json file on the remote machine via SSH.

        Parameters
        ----------
        assignment : RemoteTaskAssignment
            The task assignment

        Returns
        -------
        Optional[Dict[str, Any]]
            Progress data dict, or None if not available
        """
        progress_path = assignment.remote_progress_path
        if not progress_path:
            return None

        success, stdout, _ = self._ssh_exec(
            assignment.remote_machine.hostname,
            f"cat {progress_path} 2>/dev/null",
            timeout=10,
        )

        if not success or not stdout:
            return None

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None

    # ========================================================================
    # RESULT COLLECTION
    # ========================================================================

    def collect_results(
        self,
        assignment: RemoteTaskAssignment,
        local_output_dir: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Collect training results from a remote machine.

        Downloads results.json and the trained model directory.

        Parameters
        ----------
        assignment : RemoteTaskAssignment
            The task assignment
        local_output_dir : str
            Local directory to store the collected results

        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (success, results_dict or error_dict)
        """
        machine = assignment.remote_machine
        remote_output = assignment.remote_output_path

        # Create local output directory
        local_dir = Path(local_output_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download results.json
        remote_results_path = f"{remote_output}/results.json"
        local_results_path = str(local_dir / "results.json")

        success, error = self._scp_from_remote(
            machine.hostname,
            remote_results_path,
            local_results_path,
        )

        if not success:
            logger.warning(f"Failed to collect results.json from {machine.hostname}: {error}")
            return False, {"status": "error", "error": f"Failed to collect results: {error}"}

        # Parse results
        try:
            with open(local_results_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            return False, {"status": "error", "error": f"Failed to parse results: {e}"}

        # 2. Download trained model directory
        remote_model_dir = f"{remote_output}/model"
        local_model_dir = str(local_dir / "model")
        Path(local_model_dir).mkdir(parents=True, exist_ok=True)

        success, error = self._scp_from_remote(
            machine.hostname,
            remote_model_dir,
            local_model_dir,
            recursive=True,
        )

        if not success:
            logger.warning(f"Failed to collect model from {machine.hostname}: {error}")
            # Results are still valid even if model download fails
            results["model_download_error"] = error

        # 3. Download training metrics if available
        for metric_file in ["training.csv", "best.csv"]:
            remote_metric = f"{remote_output}/{metric_file}"
            local_metric = str(local_dir / metric_file)
            self._scp_from_remote(machine.hostname, remote_metric, local_metric)

        logger.info(f"Collected results from {machine.hostname} for {assignment.category_name}")
        results["local_model_path"] = local_model_dir
        return True, results

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup_remote(self, machine: SSHRemoteMachine, category: str):
        """
        Clean up temporary training files on a remote machine.

        Parameters
        ----------
        machine : SSHRemoteMachine
            The remote machine
        category : str
            Category name to clean up
        """
        remote_dir = f"{REMOTE_TRAINING_DIR}/{category}"
        success, _, stderr = self._ssh_exec(
            machine.hostname,
            f"rm -rf {remote_dir}",
            timeout=10,
        )
        if success:
            logger.info(f"Cleaned up {remote_dir} on {machine.hostname}")
        else:
            logger.warning(f"Cleanup failed on {machine.hostname}: {stderr}")

    def cleanup_all_remote(self, machine: SSHRemoteMachine):
        """Clean up the entire training directory on a remote machine."""
        success, _, stderr = self._ssh_exec(
            machine.hostname,
            f"rm -rf {REMOTE_TRAINING_DIR}",
            timeout=10,
        )
        if success:
            logger.info(f"Cleaned up all training data on {machine.hostname}")
        else:
            logger.warning(f"Full cleanup failed on {machine.hostname}: {stderr}")

    # ========================================================================
    # GPU DETECTION HELPERS
    # ========================================================================

    def any_remote_has_gpu(self) -> bool:
        """Check if any connected remote machine has a GPU."""
        return any(m.has_gpu for m in self.machines)

    def get_gpu_machines(self) -> List[SSHRemoteMachine]:
        """Get all remote machines that have a GPU."""
        return [m for m in self.machines if m.has_gpu]

    def get_cpu_only_machines(self) -> List[SSHRemoteMachine]:
        """Get all remote machines that only have CPUs."""
        return [m for m in self.machines if not m.has_gpu]

    def get_total_remote_gpu_count(self) -> int:
        """Get the total number of GPUs across all remote machines."""
        return sum(1 for m in self.machines if m.has_gpu)

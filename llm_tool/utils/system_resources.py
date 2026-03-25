#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
system_resources.py

MAIN OBJECTIVE:
---------------
Sophisticated system resource detection module for Mac and Windows.
Provides comprehensive hardware information including GPU, CPU, RAM, and storage.

Dependencies:
-------------
- psutil: System and process utilities
- platform: Access to underlying platform's data
- torch: PyTorch for GPU detection
- subprocess: For system commands
- json: For data serialization
- dataclasses: For structured data

MAIN FEATURES:
--------------
1) Detect GPU availability (CUDA, MPS, CPU)
2) Get detailed CPU information (cores, frequency, usage)
3) Monitor RAM usage (total, available, used, percentage)
4) Check storage information
5) Detect OS and architecture
6) Provide recommendations for optimal pipeline configuration
7) Cache results to avoid repeated expensive operations

Author:
-------
Antoine Lemor
"""

import os
import platform
import subprocess
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from functools import lru_cache
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about GPU availability and capabilities"""
    available: bool = False
    device_type: str = "cpu"  # cpu, cuda, mps
    device_count: int = 0
    device_names: List[str] = None
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None

    def __post_init__(self):
        if self.device_names is None:
            self.device_names = []


@dataclass
class CPUInfo:
    """Information about CPU"""
    physical_cores: int = 0
    logical_cores: int = 0
    max_frequency_mhz: float = 0.0
    current_frequency_mhz: float = 0.0
    cpu_percent: float = 0.0
    architecture: str = ""
    processor_name: str = ""


@dataclass
class MemoryInfo:
    """Information about system memory (RAM)"""
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    percent_used: float = 0.0


@dataclass
class StorageInfo:
    """Information about storage"""
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    percent_used: float = 0.0


@dataclass
class SystemInfo:
    """Complete system information"""
    os_name: str = ""
    os_version: str = ""
    os_release: str = ""
    machine: str = ""
    python_version: str = ""
    detection_timestamp: str = ""


@dataclass
class SystemResources:
    """Complete system resources information"""
    gpu: GPUInfo
    cpu: CPUInfo
    memory: MemoryInfo
    storage: StorageInfo
    system: SystemInfo

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'gpu': asdict(self.gpu),
            'cpu': asdict(self.cpu),
            'memory': asdict(self.memory),
            'storage': asdict(self.storage),
            'system': asdict(self.system)
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def get_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendations for optimal configuration based on detected resources.

        Returns
        -------
        dict
            Recommendations for batch size, workers, device, etc.
        """
        recommendations = {
            'device': self.gpu.device_type,
            'batch_size': 8,
            'num_workers': 2,
            'use_fp16': False,
            'gradient_accumulation_steps': 1,
            'notes': []
        }

        # GPU-based recommendations
        if self.gpu.available:
            if self.gpu.device_type == "cuda":
                if self.gpu.total_memory_gb >= 16:
                    recommendations['batch_size'] = 32
                    recommendations['use_fp16'] = True
                    recommendations['notes'].append("Large GPU detected: Using larger batch size and FP16")
                elif self.gpu.total_memory_gb >= 8:
                    recommendations['batch_size'] = 16
                    recommendations['use_fp16'] = True
                    recommendations['notes'].append("Medium GPU detected: Using moderate batch size and FP16")
                else:
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 2
                    recommendations['notes'].append("Small GPU detected: Using smaller batch with gradient accumulation")

            elif self.gpu.device_type == "mps":
                # Apple Silicon recommendations
                recommendations['batch_size'] = 16
                recommendations['notes'].append("Apple Silicon detected: MPS acceleration enabled")
        else:
            # CPU-only recommendations
            recommendations['batch_size'] = 8
            recommendations['gradient_accumulation_steps'] = 2
            recommendations['notes'].append("No GPU detected: Using CPU with conservative settings")

        # CPU-based recommendations for workers
        if self.cpu.physical_cores >= 8:
            recommendations['num_workers'] = min(8, self.cpu.physical_cores // 2)
        else:
            recommendations['num_workers'] = max(2, self.cpu.physical_cores // 2)

        # Memory-based adjustments
        if self.memory.available_gb < 8:
            recommendations['batch_size'] = max(4, recommendations['batch_size'] // 2)
            recommendations['num_workers'] = max(2, recommendations['num_workers'] // 2)
            recommendations['notes'].append("Low RAM detected: Reduced batch size and workers")

        return recommendations


class SystemResourceDetector:
    """
    Sophisticated system resource detector for Mac and Windows.

    This class provides comprehensive detection of system resources including
    GPU, CPU, memory, and storage. It caches results to avoid repeated
    expensive detection operations.
    """

    def __init__(self, cache_duration: int = 300):
        """
        Initialize the resource detector.

        Parameters
        ----------
        cache_duration : int
            Cache duration in seconds (default: 300 = 5 minutes)
        """
        self.cache_duration = cache_duration
        self._cache: Optional[SystemResources] = None
        self._cache_time: Optional[float] = None

    def detect_all(self, force_refresh: bool = False) -> SystemResources:
        """
        Detect all system resources.

        Parameters
        ----------
        force_refresh : bool
            Force refresh even if cache is valid

        Returns
        -------
        SystemResources
            Complete system resource information
        """
        # Check cache
        if not force_refresh and self._cache is not None and self._cache_time is not None:
            import time
            if time.time() - self._cache_time < self.cache_duration:
                return self._cache

        # Detect all resources
        gpu_info = self._detect_gpu()
        cpu_info = self._detect_cpu()
        memory_info = self._detect_memory()
        storage_info = self._detect_storage()
        system_info = self._detect_system()

        resources = SystemResources(
            gpu=gpu_info,
            cpu=cpu_info,
            memory=memory_info,
            storage=storage_info,
            system=system_info
        )

        # Update cache
        import time
        self._cache = resources
        self._cache_time = time.time()

        return resources

    def _detect_gpu(self) -> GPUInfo:
        """
        Detect GPU availability and information.

        Returns
        -------
        GPUInfo
            GPU information
        """
        gpu_info = GPUInfo()

        if not HAS_TORCH:
            logger.warning("PyTorch not available. Cannot detect GPU.")
            return gpu_info

        try:
            # Check CUDA (NVIDIA GPUs)
            if torch.cuda.is_available():
                gpu_info.available = True
                gpu_info.device_type = "cuda"
                gpu_info.device_count = torch.cuda.device_count()
                gpu_info.device_names = [
                    torch.cuda.get_device_name(i)
                    for i in range(gpu_info.device_count)
                ]

                # Get memory info for first device
                if gpu_info.device_count > 0:
                    props = torch.cuda.get_device_properties(0)
                    gpu_info.total_memory_gb = props.total_memory / (1024**3)

                    # Get available memory
                    try:
                        mem_free, mem_total = torch.cuda.mem_get_info(0)
                        gpu_info.available_memory_gb = mem_free / (1024**3)
                    except:
                        gpu_info.available_memory_gb = gpu_info.total_memory_gb

                    # Get CUDA version
                    try:
                        gpu_info.cuda_version = torch.version.cuda
                    except:
                        pass

                    # Get compute capability
                    try:
                        gpu_info.compute_capability = f"{props.major}.{props.minor}"
                    except:
                        pass

            # Check MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info.available = True
                gpu_info.device_type = "mps"
                gpu_info.device_count = 1
                gpu_info.device_names = ["Apple Silicon (MPS)"]

                # Try to get Apple Silicon model
                if platform.system() == "Darwin":
                    try:
                        result = subprocess.run(
                            ['sysctl', '-n', 'machdep.cpu.brand_string'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            gpu_info.device_names = [result.stdout.strip()]
                    except:
                        pass

                # Approximate memory (unified memory on Apple Silicon)
                if HAS_PSUTIL:
                    mem = psutil.virtual_memory()
                    # Assume ~75% of RAM can be used for GPU
                    gpu_info.total_memory_gb = (mem.total / (1024**3)) * 0.75
                    gpu_info.available_memory_gb = (mem.available / (1024**3)) * 0.75

            # CPU fallback
            else:
                gpu_info.device_type = "cpu"
                gpu_info.device_names = ["CPU only"]

        except Exception as e:
            logger.error(f"Error detecting GPU: {e}")
            gpu_info.device_type = "cpu"

        return gpu_info

    def _detect_cpu(self) -> CPUInfo:
        """
        Detect CPU information.

        Returns
        -------
        CPUInfo
            CPU information
        """
        cpu_info = CPUInfo()

        try:
            # Basic CPU info
            if HAS_PSUTIL:
                cpu_info.physical_cores = psutil.cpu_count(logical=False) or 0
                cpu_info.logical_cores = psutil.cpu_count(logical=True) or 0

                # CPU frequency
                try:
                    freq = psutil.cpu_freq()
                    if freq:
                        cpu_info.max_frequency_mhz = freq.max
                        cpu_info.current_frequency_mhz = freq.current
                except:
                    pass

                # CPU usage
                try:
                    cpu_info.cpu_percent = psutil.cpu_percent(interval=0.1)
                except:
                    pass
            else:
                # Fallback without psutil
                import multiprocessing
                cpu_info.logical_cores = multiprocessing.cpu_count()
                cpu_info.physical_cores = cpu_info.logical_cores // 2

            # Architecture and processor name
            cpu_info.architecture = platform.machine()
            cpu_info.processor_name = platform.processor()

            # Try to get more detailed CPU info on different platforms
            if platform.system() == "Darwin":
                # macOS
                try:
                    result = subprocess.run(
                        ['sysctl', '-n', 'machdep.cpu.brand_string'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        cpu_info.processor_name = result.stdout.strip()
                except:
                    pass

            elif platform.system() == "Windows":
                # Windows
                try:
                    result = subprocess.run(
                        ['wmic', 'cpu', 'get', 'name'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            cpu_info.processor_name = lines[1].strip()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error detecting CPU: {e}")

        return cpu_info

    def _detect_memory(self) -> MemoryInfo:
        """
        Detect memory (RAM) information.

        Returns
        -------
        MemoryInfo
            Memory information
        """
        mem_info = MemoryInfo()

        if not HAS_PSUTIL:
            logger.warning("psutil not available. Cannot detect memory.")
            return mem_info

        try:
            mem = psutil.virtual_memory()
            mem_info.total_gb = mem.total / (1024**3)
            mem_info.available_gb = mem.available / (1024**3)
            mem_info.used_gb = mem.used / (1024**3)
            mem_info.percent_used = mem.percent

        except Exception as e:
            logger.error(f"Error detecting memory: {e}")

        return mem_info

    def _detect_storage(self) -> StorageInfo:
        """
        Detect storage information for current working directory.

        Returns
        -------
        StorageInfo
            Storage information
        """
        storage_info = StorageInfo()

        if not HAS_PSUTIL:
            logger.warning("psutil not available. Cannot detect storage.")
            return storage_info

        try:
            disk = psutil.disk_usage(os.getcwd())
            storage_info.total_gb = disk.total / (1024**3)
            storage_info.used_gb = disk.used / (1024**3)
            storage_info.available_gb = disk.free / (1024**3)
            storage_info.percent_used = disk.percent

        except Exception as e:
            logger.error(f"Error detecting storage: {e}")

        return storage_info

    def _detect_system(self) -> SystemInfo:
        """
        Detect system information.

        Returns
        -------
        SystemInfo
            System information
        """
        system_info = SystemInfo()

        try:
            system_info.os_name = platform.system()
            system_info.os_version = platform.version()
            system_info.os_release = platform.release()
            system_info.machine = platform.machine()
            system_info.python_version = platform.python_version()
            system_info.detection_timestamp = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Error detecting system info: {e}")

        return system_info

    def save_to_file(self, output_path: Path, force_refresh: bool = False):
        """
        Detect resources and save to JSON file.

        Parameters
        ----------
        output_path : Path
            Output file path
        force_refresh : bool
            Force refresh detection
        """
        resources = self.detect_all(force_refresh=force_refresh)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(resources.to_json())

        logger.info(f"System resources saved to {output_path}")


# Global detector instance
_detector: Optional[SystemResourceDetector] = None


def get_detector() -> SystemResourceDetector:
    """
    Get global system resource detector instance.

    Returns
    -------
    SystemResourceDetector
        Global detector instance
    """
    global _detector
    if _detector is None:
        _detector = SystemResourceDetector()
    return _detector


def detect_resources(force_refresh: bool = False) -> SystemResources:
    """
    Convenient function to detect all system resources.

    Parameters
    ----------
    force_refresh : bool
        Force refresh even if cache is valid

    Returns
    -------
    SystemResources
        Complete system resource information
    """
    return get_detector().detect_all(force_refresh=force_refresh)


def get_device_recommendation() -> str:
    """
    Get recommended device for PyTorch operations.

    Returns
    -------
    str
        Device string: "cuda", "mps", or "cpu"
    """
    resources = detect_resources()
    return resources.gpu.device_type


def get_optimal_workers() -> int:
    """
    Get optimal number of workers for data loading.

    Returns
    -------
    int
        Recommended number of workers
    """
    resources = detect_resources()
    return resources.get_recommendation()['num_workers']


def get_optimal_batch_size() -> int:
    """
    Get optimal batch size based on available resources.

    Returns
    -------
    int
        Recommended batch size
    """
    resources = detect_resources()
    return resources.get_recommendation()['batch_size']


def check_minimum_requirements(
    min_ram_gb: float = 4.0,
    min_storage_gb: float = 10.0,
    require_gpu: bool = False
) -> Tuple[bool, List[str]]:
    """
    Check if system meets minimum requirements.

    Parameters
    ----------
    min_ram_gb : float
        Minimum RAM in GB
    min_storage_gb : float
        Minimum available storage in GB
    require_gpu : bool
        Whether GPU is required

    Returns
    -------
    tuple
        (meets_requirements, list_of_issues)
    """
    resources = detect_resources()
    issues = []

    # Check RAM
    if resources.memory.available_gb < min_ram_gb:
        issues.append(
            f"Insufficient RAM: {resources.memory.available_gb:.1f}GB available, "
            f"{min_ram_gb:.1f}GB required"
        )

    # Check storage
    if resources.storage.available_gb < min_storage_gb:
        issues.append(
            f"Insufficient storage: {resources.storage.available_gb:.1f}GB available, "
            f"{min_storage_gb:.1f}GB required"
        )

    # Check GPU
    if require_gpu and not resources.gpu.available:
        issues.append("GPU required but not available")

    return (len(issues) == 0, issues)


if __name__ == "__main__":
    # Test the detector
    print("=" * 60)
    print("System Resource Detection Test")
    print("=" * 60)

    detector = SystemResourceDetector()
    resources = detector.detect_all()

    print("\n" + resources.to_json())

    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    recommendations = resources.get_recommendation()
    print(json.dumps(recommendations, indent=2))

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
1) Detect GPU availability (CUDA/NVIDIA, ROCm/AMD, MPS/Apple, CPU)
2) Get detailed CPU information (cores, frequency, usage)
3) Monitor RAM usage (total, available, used, percentage)
4) Check storage information
5) Detect OS and architecture
6) Provide recommendations for optimal pipeline configuration
7) Cache results to avoid repeated expensive operations
8) Fallback AMD GPU detection via system commands when PyTorch ROCm not installed

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
    device_type: str = "cpu"  # cpu, cuda, rocm, mps
    device_count: int = 0
    device_names: List[str] = None
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None  # AMD ROCm version
    compute_capability: Optional[str] = None
    is_amd: bool = False  # True if AMD GPU (ROCm)

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
                    recommendations['notes'].append("Large NVIDIA GPU detected: Using larger batch size and FP16")
                elif self.gpu.total_memory_gb >= 8:
                    recommendations['batch_size'] = 16
                    recommendations['use_fp16'] = True
                    recommendations['notes'].append("Medium NVIDIA GPU detected: Using moderate batch size and FP16")
                else:
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 2
                    recommendations['notes'].append("Small NVIDIA GPU detected: Using smaller batch with gradient accumulation")

            elif self.gpu.device_type == "rocm":
                # AMD GPU with ROCm recommendations
                # ROCm supports FP16 but may have different optimal batch sizes
                if self.gpu.total_memory_gb >= 16:
                    recommendations['batch_size'] = 32
                    recommendations['use_fp16'] = True
                    recommendations['notes'].append("Large AMD GPU detected (ROCm): Using larger batch size and FP16")
                elif self.gpu.total_memory_gb >= 8:
                    recommendations['batch_size'] = 16
                    recommendations['use_fp16'] = True
                    recommendations['notes'].append("Medium AMD GPU detected (ROCm): Using moderate batch size and FP16")
                else:
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 2
                    recommendations['notes'].append("Small AMD GPU detected (ROCm): Using smaller batch with gradient accumulation")

            elif self.gpu.device_type == "mps":
                # Apple Silicon recommendations - scale based on unified memory
                # MPS uses unified memory, so we base recommendations on available RAM
                # gpu.total_memory_gb is estimated as ~75% of system RAM for MPS
                if self.gpu.total_memory_gb >= 90:  # Mac Studio M2 Ultra (128GB RAM)
                    recommendations['batch_size'] = 64
                    recommendations['num_workers'] = 8  # M2 Ultra has 24 CPU cores
                    recommendations['notes'].append("Mac Studio M2 Ultra detected: Maximum batch size and workers")
                elif self.gpu.total_memory_gb >= 48:  # Mac Studio M2 Max (64GB) or similar
                    recommendations['batch_size'] = 48
                    recommendations['num_workers'] = 6
                    recommendations['notes'].append("Large Apple Silicon detected (Mac Studio/Pro): Using large batch size")
                elif self.gpu.total_memory_gb >= 24:  # MacBook Pro (32GB)
                    recommendations['batch_size'] = 32
                    recommendations['notes'].append("Medium-large Apple Silicon detected: Using moderate-large batch size")
                elif self.gpu.total_memory_gb >= 12:  # MacBook Pro (16GB) or Mac Mini
                    recommendations['batch_size'] = 16
                    recommendations['notes'].append("Medium Apple Silicon detected: MPS acceleration enabled")
                else:  # MacBook Air or smaller configs
                    recommendations['batch_size'] = 8
                    recommendations['notes'].append("Small Apple Silicon detected: Using conservative batch size")
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

        # MPS-specific worker adjustments
        # Apple Silicon with unified memory benefits less from many workers
        # and can have overhead from Python multiprocessing
        # Exception: Large Mac Studio configs can handle more workers
        if self.gpu.device_type == "mps" and self.gpu.total_memory_gb < 48:
            # Cap workers at 4 for smaller MPS configs to avoid overhead
            recommendations['num_workers'] = min(4, recommendations['num_workers'])

        # Memory-based adjustments
        if self.memory.available_gb < 8:
            recommendations['batch_size'] = max(4, recommendations['batch_size'] // 2)
            recommendations['num_workers'] = max(2, recommendations['num_workers'] // 2)
            recommendations['notes'].append("Low RAM detected: Reduced batch size and workers")

        return recommendations

    def get_recommendation_for_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get optimized recommendations for a specific model, accounting for model size.

        This method adjusts batch size, gradient accumulation, and FP16 based on:
        1. Available GPU memory
        2. Model size (parameters and memory footprint)
        3. Device type (CUDA, MPS, CPU)

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'xlm-roberta-large', 'bert-base-multilingual')

        Returns
        -------
        dict
            Optimized recommendations for the specific model
        """
        # Start with base recommendations
        recommendations = self.get_recommendation()

        # Model size categories (approximate memory footprint in GB for model weights)
        # These are estimates for full precision (FP32)
        MODEL_SIZES = {
            # Large models (~500M+ params, ~2GB+ weights)
            'xlm-roberta-large': {'params_m': 560, 'fp32_gb': 2.2, 'category': 'large'},
            'roberta-large': {'params_m': 355, 'fp32_gb': 1.4, 'category': 'large'},
            'bert-large': {'params_m': 340, 'fp32_gb': 1.3, 'category': 'large'},
            'electra-large': {'params_m': 335, 'fp32_gb': 1.3, 'category': 'large'},
            'deberta-v3-large': {'params_m': 434, 'fp32_gb': 1.7, 'category': 'large'},

            # Base models (~100-200M params, ~0.5-0.8GB weights)
            'xlm-roberta-base': {'params_m': 270, 'fp32_gb': 1.1, 'category': 'base'},
            'bert-base': {'params_m': 110, 'fp32_gb': 0.44, 'category': 'base'},
            'roberta-base': {'params_m': 125, 'fp32_gb': 0.5, 'category': 'base'},
            'camembert-base': {'params_m': 110, 'fp32_gb': 0.44, 'category': 'base'},
            'distilbert': {'params_m': 66, 'fp32_gb': 0.26, 'category': 'small'},
        }

        # Determine model category
        model_name_lower = model_name.lower()
        model_info = None
        model_category = 'base'  # Default

        for key, info in MODEL_SIZES.items():
            if key in model_name_lower:
                model_info = info
                model_category = info['category']
                break

        # XLarge/XXLarge models
        if 'xlarge' in model_name_lower or 'xxlarge' in model_name_lower:
            model_category = 'xlarge'
        elif 'large' in model_name_lower and model_info is None:
            model_category = 'large'

        # Available GPU memory for training (model + activations + gradients)
        # Rule of thumb: training needs ~4x model size for activations/gradients in FP32
        # With FP16: ~2x model size
        available_gpu_memory = self.gpu.available_memory_gb if self.gpu.available else 0

        # Calculate optimal settings based on model category and available memory
        if model_category == 'xlarge':
            # XLarge models: ~800M+ params, ~3.2GB+ weights
            # Need ~13GB+ for FP32 training, ~6.5GB+ for FP16
            if available_gpu_memory >= 24:
                recommendations['batch_size'] = 8
                recommendations['gradient_accumulation_steps'] = 4
                recommendations['use_fp16'] = True
            elif available_gpu_memory >= 16:
                recommendations['batch_size'] = 4
                recommendations['gradient_accumulation_steps'] = 8
                recommendations['use_fp16'] = True
            else:
                recommendations['batch_size'] = 2
                recommendations['gradient_accumulation_steps'] = 16
                recommendations['use_fp16'] = True
            recommendations['notes'].append(f"XLarge model detected ({model_name}): Using aggressive memory optimization")

        elif model_category == 'large':
            # Large models: ~350-560M params, ~1.4-2.2GB weights
            # Need ~9GB+ for FP32 training, ~4.5GB+ for FP16
            if available_gpu_memory >= 24:
                recommendations['batch_size'] = 16
                recommendations['gradient_accumulation_steps'] = 2
                recommendations['use_fp16'] = True
            elif available_gpu_memory >= 16:
                recommendations['batch_size'] = 8
                recommendations['gradient_accumulation_steps'] = 4
                recommendations['use_fp16'] = True
            elif available_gpu_memory >= 8:
                recommendations['batch_size'] = 4
                recommendations['gradient_accumulation_steps'] = 8
                recommendations['use_fp16'] = True
            else:
                recommendations['batch_size'] = 2
                recommendations['gradient_accumulation_steps'] = 16
                recommendations['use_fp16'] = True
            recommendations['notes'].append(f"Large model detected ({model_name}): FP16 enabled, gradient accumulation adjusted")

        elif model_category == 'base':
            # Base models: ~100-270M params
            # More conservative than default - FP16 is beneficial even for base models
            if available_gpu_memory >= 16:
                recommendations['batch_size'] = 32
                recommendations['gradient_accumulation_steps'] = 1
                recommendations['use_fp16'] = True
            elif available_gpu_memory >= 8:
                recommendations['batch_size'] = 16
                recommendations['gradient_accumulation_steps'] = 2
                recommendations['use_fp16'] = True
            else:
                recommendations['batch_size'] = 8
                recommendations['gradient_accumulation_steps'] = 4
                recommendations['use_fp16'] = True
            recommendations['notes'].append(f"Base model ({model_name}): Optimized settings applied")

        # MPS-specific adjustments (Apple Silicon)
        if self.gpu.device_type == 'mps':
            # MPS uses unified memory - MAXIMIZE utilization
            # FP16/autocast has limited benefit on MPS (many ops fallback to fp32)
            # Main optimization: PUSH batch sizes to the limit with unified memory
            unified_mem = self.memory.total_gb
            gpu_mem = self.gpu.total_memory_gb  # ~75% of unified memory available for GPU

            # On MPS, disable FP16 as it provides minimal benefit
            recommendations['use_fp16'] = False

            # AGGRESSIVE batch sizes for Apple Silicon - PUSH UNIFIED MEMORY TO THE MAX
            if model_category == 'xlarge':
                # XLarge models (~800M+ params): ~3.2GB model + activations
                # Estimated: ~0.31GB per sample (1.5x large models)
                # Target: 30-35% of GPU memory with gradient checkpointing
                if unified_mem >= 128:
                    # 96GB GPU avail, 30% = 28.8GB → batch 93 → use 96
                    recommendations['batch_size'] = 96
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 12
                elif unified_mem >= 64:
                    # 48GB GPU avail, 30% = 14.4GB → batch 46 → use 48
                    recommendations['batch_size'] = 48
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 10
                elif unified_mem >= 32:
                    # 24GB GPU avail, 30% = 7.2GB → batch 23 → use 24
                    recommendations['batch_size'] = 24
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 8
                elif unified_mem >= 16:
                    # 12GB GPU avail, 35% = 4.2GB → batch 14 → use 16
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 6
                elif unified_mem >= 8:
                    # 6GB GPU avail, 40% = 2.4GB → batch 8
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 4
                else:
                    recommendations['batch_size'] = 4
                    recommendations['gradient_accumulation_steps'] = 2
                recommendations['notes'].append(f"MPS XLARGE: {unified_mem:.0f}GB → batch {recommendations['batch_size']} (calibrated ~0.31GB/sample)")

            elif model_category == 'large':
                # Large models (~350-560M params).
                # CONVERGENCE CAP: micro-batch capped at 16 for fine-tuning stability.
                # Measured 2026-05-25 on M4 Max 128GB + mDeBERTa-v3-base: micro-batch >32
                # leaves the classifier head essentially un-trained (cosine between the two
                # rows of classifier.weight stays at random init ≈ 0; the saved checkpoints
                # predict the same class for ~100 % of the test set, F1 ~ 0.25). Lit. for
                # DeBERTa-v3 fine-tuning (HuggingFace/microsoft repo) uses batch 8-16 with
                # lr 1e-5..2e-5. We compensate with gradient_accumulation_steps to keep an
                # effective batch size that still gives stable gradients but with many more
                # optimizer steps per epoch, which is what the head needs to converge.
                if unified_mem >= 128:
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 4   # eff. batch = 64
                    recommendations['num_workers'] = 12
                elif unified_mem >= 64:
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 2   # eff. batch = 32
                    recommendations['num_workers'] = 10
                elif unified_mem >= 32:
                    recommendations['batch_size'] = 12
                    recommendations['gradient_accumulation_steps'] = 2   # eff. batch = 24
                    recommendations['num_workers'] = 8
                elif unified_mem >= 16:
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 2   # eff. batch = 16
                    recommendations['num_workers'] = 6
                elif unified_mem >= 8:
                    recommendations['batch_size'] = 6
                    recommendations['gradient_accumulation_steps'] = 2
                    recommendations['num_workers'] = 4
                else:
                    recommendations['batch_size'] = 4
                    recommendations['gradient_accumulation_steps'] = 4
                recommendations['notes'].append(
                    f"MPS LARGE: {unified_mem:.0f}GB → micro-batch {recommendations['batch_size']} "
                    f"× ga={recommendations['gradient_accumulation_steps']} "
                    f"(convergence cap for fine-tuning stability)"
                )

            else:
                # Base/small models (~100-280M params, incl. mDeBERTa-v3-base).
                # CONVERGENCE CAP: micro-batch capped at 16 for fine-tuning stability.
                # See LARGE branch above for the diagnosis. Measured 2026-05-26 on
                # M4 Max 128GB at equal optimizer-step count (92), micro-batch 16
                # reaches F1=0.73 / AUC=0.97 in 2 epochs while micro-batch 32 reaches
                # only F1=0.55 (same AUC ≈ 0.97 → encoder learns in both, but the
                # head only calibrates with the smaller batch). Micro-batch 256
                # (system default before the fix) gives F1≈0.25 and a head that
                # predicts the same class for 100 % of inputs.
                # gradient_accumulation_steps preserves the effective batch when the
                # caller wants a larger one (note that going from ga=1 to ga=4 on the
                # same micro-batch costs convergence too because it cuts the number
                # of optimizer steps by 4×; keep ga modest unless lr is also scaled).
                if unified_mem >= 128:
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 2   # eff. batch = 32
                    recommendations['num_workers'] = 12
                elif unified_mem >= 64:
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 2   # eff. batch = 32
                    recommendations['num_workers'] = 10
                elif unified_mem >= 32:
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 1   # eff. batch = 16
                    recommendations['num_workers'] = 8
                elif unified_mem >= 16:
                    recommendations['batch_size'] = 16
                    recommendations['gradient_accumulation_steps'] = 1   # eff. batch = 16
                    recommendations['num_workers'] = 6
                elif unified_mem >= 8:
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 4
                else:
                    recommendations['batch_size'] = 8
                    recommendations['gradient_accumulation_steps'] = 1
                    recommendations['num_workers'] = 2
                recommendations['notes'].append(
                    f"MPS BASE: {unified_mem:.0f}GB → micro-batch {recommendations['batch_size']} "
                    f"× ga={recommendations['gradient_accumulation_steps']} "
                    f"(convergence cap for fine-tuning stability)"
                )

        # Calculate effective batch size for logging
        effective_batch_size = recommendations['batch_size'] * recommendations['gradient_accumulation_steps']
        recommendations['effective_batch_size'] = effective_batch_size
        recommendations['model_category'] = model_category

        return recommendations


# ============================================================================
# MEMORY MONITORING FOR PARALLEL TRAINING
# ============================================================================

@dataclass
class MemoryPressure:
    """Current memory pressure state."""
    level: str  # "normal", "warning", "critical", "emergency"
    percent_used: float
    available_gb: float
    total_gb: float
    can_start_worker: bool
    recommended_wait_seconds: float = 0.0


class MemoryMonitor:
    """
    Real-time memory monitoring for intelligent parallel training.

    Monitors system memory and provides guidance on when to start new workers
    to prevent OOM crashes. Designed for Apple Silicon unified memory architecture.

    Thresholds:
    - normal (<75%): Safe to start new workers
    - warning (75-85%): Start with delay, reduced batch size
    - critical (85-92%): Do not start new workers, wait for completion
    - emergency (>92%): Trigger cleanup, reduce active workers
    """

    # Memory thresholds (percentage of total RAM used)
    THRESHOLD_WARNING = 75.0
    THRESHOLD_CRITICAL = 85.0
    THRESHOLD_EMERGENCY = 92.0

    # Memory constants (in GB) - based on REAL measurements (2026-01-25)
    # xlm-roberta-base: model(1.1GB) + gradients(1.1GB) + optimizer(2.2GB) +
    #                   activations(3-5GB) + tokenizer(0.5GB) + dataset(1-2GB) + overhead(1GB)
    # REAL OBSERVED: 15-25GB per CPU worker (varies by model and dataset)
    # Adjusted for Mac Studio M2 Ultra (128GB) - can be more aggressive
    MEMORY_PER_CPU_WORKER_GB = 20.0  # Realistic estimate for transformer models
    MIN_FREE_MEMORY_GB = 12.0        # Keep 12GB free for system stability
    GPU_RESERVE_GB = 16.0            # GPU shares memory with CPU on MPS unified memory

    def __init__(
        self,
        warning_threshold: float = THRESHOLD_WARNING,
        critical_threshold: float = THRESHOLD_CRITICAL,
        emergency_threshold: float = THRESHOLD_EMERGENCY,
        min_free_gb: float = MIN_FREE_MEMORY_GB,
        gpu_reserve_gb: float = GPU_RESERVE_GB,
        memory_per_worker_gb: float = MEMORY_PER_CPU_WORKER_GB,
    ):
        """
        Initialize the memory monitor.

        Parameters
        ----------
        warning_threshold : float
            Percentage threshold for warning level (default: 75%)
        critical_threshold : float
            Percentage threshold for critical level (default: 85%)
        emergency_threshold : float
            Percentage threshold for emergency level (default: 92%)
        min_free_gb : float
            Minimum free memory to always maintain (default: 8GB)
        gpu_reserve_gb : float
            Memory to reserve for GPU operations (default: 4GB)
        memory_per_worker_gb : float
            Estimated memory per CPU worker (default: 2.5GB)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.min_free_gb = min_free_gb
        self.gpu_reserve_gb = gpu_reserve_gb
        self.memory_per_worker_gb = memory_per_worker_gb

        # Cache for rate limiting
        self._last_check_time = 0.0
        self._cached_pressure: Optional[MemoryPressure] = None
        self._cache_duration = 0.5  # seconds

    def get_memory_pressure(self, force_refresh: bool = False) -> MemoryPressure:
        """
        Get current memory pressure level.

        Parameters
        ----------
        force_refresh : bool
            Force refresh even if cache is valid

        Returns
        -------
        MemoryPressure
            Current memory pressure state
        """
        import time
        now = time.time()

        # Return cached result if still valid
        if not force_refresh and self._cached_pressure is not None:
            if now - self._last_check_time < self._cache_duration:
                return self._cached_pressure

        # Get current memory stats
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            percent_used = mem.percent
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
        else:
            # Fallback for systems without psutil
            percent_used = 50.0  # Conservative default
            available_gb = 16.0
            total_gb = 32.0

        # Determine pressure level
        if percent_used >= self.emergency_threshold:
            level = "emergency"
            can_start = False
            wait_seconds = 30.0  # Wait longer in emergency
        elif percent_used >= self.critical_threshold:
            level = "critical"
            can_start = False
            wait_seconds = 10.0
        elif percent_used >= self.warning_threshold:
            level = "warning"
            can_start = True  # Can start but with caution
            wait_seconds = 3.0  # Brief delay
        else:
            level = "normal"
            can_start = True
            wait_seconds = 0.0

        # Additional check: ensure minimum free memory
        if available_gb < self.min_free_gb:
            can_start = False
            if level == "normal":
                level = "warning"
            wait_seconds = max(wait_seconds, 5.0)

        pressure = MemoryPressure(
            level=level,
            percent_used=percent_used,
            available_gb=available_gb,
            total_gb=total_gb,
            can_start_worker=can_start,
            recommended_wait_seconds=wait_seconds,
        )

        # Update cache
        self._cached_pressure = pressure
        self._last_check_time = now

        return pressure

    def should_start_new_worker(self) -> Tuple[bool, str]:
        """
        Check if it's safe to start a new worker.

        Returns
        -------
        Tuple[bool, str]
            (can_start, reason_message)
        """
        pressure = self.get_memory_pressure()

        if pressure.level == "emergency":
            return False, f"Emergency memory pressure ({pressure.percent_used:.1f}% used, {pressure.available_gb:.1f}GB free)"
        elif pressure.level == "critical":
            return False, f"Critical memory pressure ({pressure.percent_used:.1f}% used) - waiting for workers to complete"
        elif pressure.level == "warning":
            return True, f"Warning: Memory at {pressure.percent_used:.1f}%, proceeding with caution"
        else:
            return True, f"Memory OK ({pressure.percent_used:.1f}% used, {pressure.available_gb:.1f}GB free)"

    def calculate_safe_cpu_workers(self, has_gpu_worker: bool = True) -> int:
        """
        Calculate the safe number of CPU workers based on available memory.

        Parameters
        ----------
        has_gpu_worker : bool
            Whether a GPU worker will be running (reserves additional memory)

        Returns
        -------
        int
            Recommended number of CPU workers (0-4)
        """
        pressure = self.get_memory_pressure()

        available = pressure.available_gb

        # Reserve memory for GPU worker if applicable
        if has_gpu_worker:
            available -= self.gpu_reserve_gb

        # Reserve minimum free memory
        available -= self.min_free_gb

        # Calculate max workers
        if available <= 0:
            return 0

        max_workers = int(available / self.memory_per_worker_gb)

        # Cap at 4 workers (empirically, more workers don't help much)
        max_workers = min(4, max_workers)

        # Reduce workers if memory pressure is high
        if pressure.level == "warning":
            max_workers = min(2, max_workers)
        elif pressure.level == "critical":
            max_workers = 0
        elif pressure.level == "emergency":
            max_workers = 0

        return max(0, max_workers)

    def estimate_training_memory(self, model_name: str) -> float:
        """
        Estimate memory needed for training a specific model.

        Based on REAL measurements (2026-01-25):
        - xlm-roberta-base: 20-29GB per worker (observed)
        - Includes: model + gradients + optimizer + activations + dataset + overhead

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'xlm-roberta-base', 'bert-large')

        Returns
        -------
        float
            Estimated memory in GB
        """
        model_name_lower = model_name.lower()

        # Memory estimates based on REAL measurements (not theoretical)
        # These include: model + gradients + optimizer states + activations + dataset + PyTorch overhead
        if 'xlarge' in model_name_lower or 'xxlarge' in model_name_lower:
            return 50.0  # XLarge models (800M+ params)
        elif 'large' in model_name_lower:
            return 35.0  # Large models (e.g., xlm-roberta-large, 560M params)
        elif 'base' in model_name_lower:
            return 25.0  # Base models (e.g., xlm-roberta-base) - REAL: 20-29GB observed
        elif 'small' in model_name_lower or 'distil' in model_name_lower:
            return 15.0  # Small/distilled models
        else:
            return 25.0  # Default to base model estimate (conservative)

    def trigger_cleanup(self) -> None:
        """
        Trigger memory cleanup operations.

        Call this when memory pressure is high to free up resources.
        """
        import gc
        gc.collect()

        # Try to clear GPU/MPS cache if available
        if HAS_TORCH:
            try:
                import torch
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("Memory cleanup triggered")


# Global memory monitor instance
_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """
    Get global memory monitor instance.

    Returns
    -------
    MemoryMonitor
        Global monitor instance
    """
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


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

        Supports:
        - NVIDIA GPUs via CUDA
        - AMD GPUs via ROCm (uses torch.cuda API with HIP backend)
        - Apple Silicon via MPS

        Returns
        -------
        GPUInfo
            GPU information
        """
        gpu_info = GPUInfo()

        if not HAS_TORCH:
            # Fallback detection for Apple Silicon without PyTorch compiled with MPS
            if platform.system() == "Darwin" and platform.machine().lower() == "arm64":
                gpu_info.available = True
                gpu_info.device_type = "mps"
                gpu_info.device_count = 1
                # Try to get model name
                try:
                    result = subprocess.run(
                        ['sysctl', '-n', 'machdep.cpu.brand_string'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        gpu_info.device_names = [result.stdout.strip()]
                except Exception:
                    gpu_info.device_names = ["Apple Silicon (MPS)"]

                # Estimate unified memory footprint using system RAM (fallback without psutil)
                try:
                    mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())
                    total_gb = mem_bytes / (1024 ** 3)
                    gpu_info.total_memory_gb = total_gb * 0.75
                    # Estimate free memory via vm_stat
                    page_size = int(subprocess.check_output(['sysctl', '-n', 'hw.pagesize']).strip())
                    vm_out = subprocess.check_output(['vm_stat']).decode()
                    free_pages = 0
                    inactive_pages = 0
                    for line in vm_out.splitlines():
                        if 'Pages free' in line:
                            free_pages = int(line.split(':')[-1].strip().strip('.'))  # pages
                        if 'Pages inactive' in line:
                            inactive_pages = int(line.split(':')[-1].strip().strip('.'))
                    free_bytes = (free_pages + inactive_pages) * page_size
                    gpu_info.available_memory_gb = (free_bytes / (1024 ** 3)) * 0.75
                except Exception:
                    pass
            else:
                logger.warning("PyTorch not available. Cannot detect GPU.")
            return gpu_info

        try:
            # Check CUDA/ROCm (NVIDIA and AMD GPUs)
            # PyTorch with ROCm uses the same torch.cuda API but with HIP backend
            if torch.cuda.is_available():
                gpu_info.available = True
                gpu_info.device_count = torch.cuda.device_count()
                gpu_info.device_names = [
                    torch.cuda.get_device_name(i)
                    for i in range(gpu_info.device_count)
                ]

                # Detect if this is AMD ROCm or NVIDIA CUDA
                # ROCm sets torch.version.hip, CUDA sets torch.version.cuda
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

                if is_rocm:
                    # AMD GPU with ROCm
                    gpu_info.device_type = "rocm"
                    gpu_info.is_amd = True
                    try:
                        gpu_info.rocm_version = torch.version.hip
                    except:
                        pass
                    logger.info(f"AMD GPU detected via ROCm: {gpu_info.device_names}")
                else:
                    # NVIDIA GPU with CUDA
                    gpu_info.device_type = "cuda"
                    gpu_info.is_amd = False
                    try:
                        gpu_info.cuda_version = torch.version.cuda
                    except:
                        pass

                # Get memory info for first device (works for both CUDA and ROCm)
                if gpu_info.device_count > 0:
                    props = torch.cuda.get_device_properties(0)
                    gpu_info.total_memory_gb = props.total_memory / (1024**3)

                    # Get available memory
                    try:
                        mem_free, mem_total = torch.cuda.mem_get_info(0)
                        gpu_info.available_memory_gb = mem_free / (1024**3)
                    except:
                        gpu_info.available_memory_gb = gpu_info.total_memory_gb

                    # Get compute capability (NVIDIA) or GCN architecture (AMD)
                    try:
                        if is_rocm:
                            # For AMD, use architecture info from device name or props
                            gpu_info.compute_capability = f"gfx{props.gcnArchName}" if hasattr(props, 'gcnArchName') else None
                        else:
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

            # CPU fallback - but first check for undetected AMD GPU
            else:
                # Try to detect AMD GPU via system commands even if PyTorch doesn't see it
                amd_gpu = self._detect_amd_gpu_fallback()
                if amd_gpu:
                    gpu_info.available = True
                    gpu_info.device_type = "rocm"
                    gpu_info.is_amd = True
                    gpu_info.device_count = amd_gpu.get('count', 1)
                    gpu_info.device_names = amd_gpu.get('names', ["AMD GPU (ROCm not configured)"])
                    gpu_info.total_memory_gb = amd_gpu.get('memory_gb', 0)
                    logger.warning(
                        f"AMD GPU detected but PyTorch ROCm not configured: {gpu_info.device_names}. "
                        "Install PyTorch with ROCm support: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
                    )
                else:
                    gpu_info.device_type = "cpu"
                    gpu_info.device_names = ["CPU only"]

        except Exception as e:
            logger.error(f"Error detecting GPU: {e}")
            gpu_info.device_type = "cpu"

        return gpu_info

    def _detect_amd_gpu_fallback(self) -> Optional[Dict[str, Any]]:
        """
        Fallback detection for AMD GPUs when PyTorch ROCm is not available.

        Uses system commands to detect AMD GPUs:
        - Linux: lspci, rocm-smi
        - Windows: wmic

        Returns
        -------
        dict or None
            AMD GPU info dict with 'names', 'count', 'memory_gb' or None if not found
        """
        system = platform.system()

        try:
            if system == "Linux":
                # Try rocm-smi first (most reliable for ROCm)
                try:
                    result = subprocess.run(
                        ['rocm-smi', '--showproductname'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and 'GPU' in result.stdout:
                        # Parse rocm-smi output
                        lines = result.stdout.strip().split('\n')
                        gpu_names = []
                        for line in lines:
                            if 'Card series' in line or 'GPU' in line:
                                # Extract GPU name
                                parts = line.split(':')
                                if len(parts) > 1:
                                    gpu_names.append(parts[1].strip())

                        if gpu_names:
                            # Try to get memory
                            mem_gb = 0
                            try:
                                mem_result = subprocess.run(
                                    ['rocm-smi', '--showmeminfo', 'vram'],
                                    capture_output=True,
                                    text=True,
                                    timeout=5
                                )
                                if mem_result.returncode == 0:
                                    # Parse memory info
                                    for line in mem_result.stdout.split('\n'):
                                        if 'Total' in line:
                                            # Extract memory value
                                            import re
                                            match = re.search(r'(\d+)', line)
                                            if match:
                                                mem_mb = int(match.group(1))
                                                mem_gb = mem_mb / 1024
                            except:
                                pass

                            return {
                                'names': gpu_names,
                                'count': len(gpu_names),
                                'memory_gb': mem_gb
                            }
                except FileNotFoundError:
                    pass

                # Fallback to lspci
                try:
                    result = subprocess.run(
                        ['lspci'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        amd_gpus = []
                        for line in result.stdout.split('\n'):
                            if 'VGA' in line or 'Display' in line or '3D' in line:
                                if 'AMD' in line or 'ATI' in line or 'Radeon' in line:
                                    # Extract GPU name
                                    amd_gpus.append(line.split(':')[-1].strip())

                        if amd_gpus:
                            return {
                                'names': amd_gpus,
                                'count': len(amd_gpus),
                                'memory_gb': 0  # Cannot determine from lspci
                            }
                except FileNotFoundError:
                    pass

            elif system == "Windows":
                # Use wmic to detect AMD GPUs
                try:
                    result = subprocess.run(
                        ['wmic', 'path', 'win32_VideoController', 'get', 'name,adapterram'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        amd_gpus = []
                        total_mem = 0
                        for line in result.stdout.split('\n'):
                            if 'AMD' in line or 'Radeon' in line:
                                parts = line.strip().split()
                                if parts:
                                    # Try to extract memory (last number)
                                    try:
                                        mem_bytes = int(parts[-1])
                                        total_mem += mem_bytes / (1024**3)
                                        gpu_name = ' '.join(parts[:-1])
                                    except:
                                        gpu_name = line.strip()
                                    amd_gpus.append(gpu_name)

                        if amd_gpus:
                            return {
                                'names': amd_gpus,
                                'count': len(amd_gpus),
                                'memory_gb': total_mem
                            }
                except FileNotFoundError:
                    pass

        except Exception as e:
            logger.debug(f"AMD GPU fallback detection failed: {e}")

        return None

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

        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                mem_info.total_gb = mem.total / (1024**3)
                mem_info.available_gb = mem.available / (1024**3)
                mem_info.used_gb = mem.used / (1024**3)
                mem_info.percent_used = mem.percent
                return mem_info
            except Exception as e:
                logger.error(f"Error detecting memory: {e}")
        else:
            logger.warning("psutil not available. Falling back to system commands for memory detection.")

        # Fallback detection without psutil
        try:
            if platform.system() == "Darwin":
                # macOS unified memory
                mem_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())
                page_size = int(subprocess.check_output(['sysctl', '-n', 'hw.pagesize']).strip())
                vm_out = subprocess.check_output(['vm_stat']).decode()
                free_pages = inactive_pages = 0
                for line in vm_out.splitlines():
                    if 'Pages free' in line:
                        free_pages = int(line.split(':')[-1].strip().strip('.'))
                    if 'Pages inactive' in line:
                        inactive_pages = int(line.split(':')[-1].strip().strip('.'))
                available_bytes = (free_pages + inactive_pages) * page_size
                mem_info.total_gb = mem_bytes / (1024**3)
                mem_info.available_gb = available_bytes / (1024**3)
                mem_info.used_gb = mem_info.total_gb - mem_info.available_gb
                mem_info.percent_used = (mem_info.used_gb / mem_info.total_gb * 100) if mem_info.total_gb else 0.0
            else:
                # Linux fallback using /proc/meminfo
                meminfo = {}
                with open('/proc/meminfo') as f:
                    for line in f:
                        key, val = line.split(':', 1)
                        meminfo[key] = float(val.strip().split()[0])  # kB
                total_kb = meminfo.get('MemTotal', 0)
                free_kb = meminfo.get('MemFree', 0) + meminfo.get('Buffers', 0) + meminfo.get('Cached', 0)
                mem_info.total_gb = total_kb / (1024**2)
                mem_info.available_gb = free_kb / (1024**2)
                mem_info.used_gb = mem_info.total_gb - mem_info.available_gb
                mem_info.percent_used = (mem_info.used_gb / mem_info.total_gb * 100) if mem_info.total_gb else 0.0
        except Exception as e:
            logger.error(f"Fallback memory detection failed: {e}")

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

        if HAS_PSUTIL:
            try:
                disk = psutil.disk_usage(os.getcwd())
                storage_info.total_gb = disk.total / (1024**3)
                storage_info.used_gb = disk.used / (1024**3)
                storage_info.available_gb = disk.free / (1024**3)
                storage_info.percent_used = disk.percent
                return storage_info
            except Exception as e:
                logger.error(f"Error detecting storage: {e}")
        else:
            logger.warning("psutil not available. Falling back to shutil for storage detection.")

        # Fallback using shutil.disk_usage
        try:
            import shutil
            disk = shutil.disk_usage(os.getcwd())
            storage_info.total_gb = disk.total / (1024**3)
            storage_info.used_gb = disk.used / (1024**3)
            storage_info.available_gb = disk.free / (1024**3)
            storage_info.percent_used = (storage_info.used_gb / storage_info.total_gb * 100) if storage_info.total_gb else 0.0
        except Exception as e:
            logger.error(f"Fallback storage detection failed: {e}")

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
        Note: For AMD ROCm GPUs, returns "cuda" as PyTorch ROCm uses the same API
    """
    resources = detect_resources()
    device_type = resources.gpu.device_type

    # ROCm uses the same "cuda" device string in PyTorch
    if device_type == "rocm":
        return "cuda"

    return device_type


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


def get_model_optimized_config(model_name: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get optimized training configuration for a specific model.

    This is the main entry point for getting GPU-optimized training parameters
    that account for both system resources and model size.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'xlm-roberta-large', 'bert-base-multilingual')
    force_refresh : bool
        Force refresh resource detection

    Returns
    -------
    dict
        Optimized configuration with keys:
        - batch_size: Recommended batch size
        - gradient_accumulation_steps: Steps for gradient accumulation
        - use_fp16: Whether to use mixed precision
        - num_workers: DataLoader workers
        - effective_batch_size: batch_size * gradient_accumulation_steps
        - model_category: Detected model category (small/base/large/xlarge)
        - notes: List of optimization notes

    Example
    -------
    >>> config = get_model_optimized_config('xlm-roberta-large')
    >>> print(f"Batch: {config['batch_size']}, FP16: {config['use_fp16']}")
    """
    resources = detect_resources(force_refresh=force_refresh)
    return resources.get_recommendation_for_model(model_name)


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

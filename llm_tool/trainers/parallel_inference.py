#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
parallel_inference.py

MAIN OBJECTIVE:
---------------
Run high-throughput inference across multiple CPUs or GPUs by distributing
transformer workloads and managing per-worker model caches.

Dependencies:
-------------
- math
- os
- queue
- sys
- typing
- numpy
- torch
- tqdm
- llm_tool.trainers.models

MAIN FEATURES:
--------------
1) Detect available CUDA or MPS devices and configure worker processes
2) Cache tokenizers and models per worker to avoid redundant loading
3) Chunk prediction requests intelligently based on device batch capacity
4) Orchestrate multiprocessing queues with graceful fallbacks to CPU
5) Expose helpers that map language codes to the right model backends

Author:
-------
Antoine Lemor
"""

from __future__ import annotations
import gc
import math
import os
import queue
import sys
from typing import Iterable, List, Sequence, Tuple, Union, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import time

import numpy as np
import torch
from torch.multiprocessing import get_context            
from tqdm.auto import tqdm

# ---- Package-internal imports ---------------------------------------------
from llm_tool.trainers.models import (
    Bert, Camembert, ArabicBert, ChineseBert, GermanBert, HindiBert,
    ItalianBert, PortugueseBert, RussianBert, SpanishBert, SwedishBert,
    XLMRoberta, MultiBERT,
)
from llm_tool.trainers.sota_models import AutoModelWrapper

# ---------------------------------------------------------------------------

@dataclass
class InferenceProgress:
    """Statistics for parallel inference progress tracking with per-worker granularity."""
    total_texts: int = 0
    processed_texts: int = 0
    device_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    worker_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Per-worker stats
    start_time: float = 0.0
    first_result_time: float = 0.0
    n_workers: int = 0
    chunk_size: int = 0
    n_gpu_workers: int = 0
    n_cpu_workers: int = 0
    workers_initialized: bool = False
    initialization_time: float = 0.0

    def update_worker(self, worker_id: str, device_tag: str, processed: int, chunk_time: float):
        """Update statistics for a specific worker and its device."""
        # Update per-worker stats
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = {
                "processed": 0,
                "chunks": 0,
                "total_time": 0.0,
                "first_chunk_time": None,
                "device_tag": device_tag,
            }
        w_stats = self.worker_stats[worker_id]
        w_stats["processed"] += processed
        w_stats["chunks"] += 1
        w_stats["total_time"] += chunk_time
        if w_stats["first_chunk_time"] is None:
            w_stats["first_chunk_time"] = chunk_time

        # Also update aggregated device stats for backward compatibility
        if device_tag not in self.device_stats:
            self.device_stats[device_tag] = {
                "processed": 0,
                "chunks": 0,
                "total_time": 0.0,
                "first_chunk_time": None,
            }
        d_stats = self.device_stats[device_tag]
        d_stats["processed"] += processed
        d_stats["chunks"] += 1
        d_stats["total_time"] += chunk_time
        if d_stats["first_chunk_time"] is None:
            d_stats["first_chunk_time"] = chunk_time

        self.processed_texts += processed

    def update_device(self, device_tag: str, processed: int, chunk_time: float):
        """Update statistics for a specific device (deprecated, use update_worker)."""
        if device_tag not in self.device_stats:
            self.device_stats[device_tag] = {
                "processed": 0,
                "chunks": 0,
                "total_time": 0.0,
                "first_chunk_time": None,
            }
        stats = self.device_stats[device_tag]
        stats["processed"] += processed
        stats["chunks"] += 1
        stats["total_time"] += chunk_time
        if stats["first_chunk_time"] is None:
            stats["first_chunk_time"] = chunk_time
        self.processed_texts += processed

    def get_estimated_remaining_time(self) -> float:
        """Calculate estimated remaining time based on processing speed."""
        if self.processed_texts == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        # Account for startup overhead by using time since first result
        if self.first_result_time > 0:
            effective_elapsed = time.time() - self.first_result_time
            if effective_elapsed > 0:
                texts_per_sec = self.processed_texts / effective_elapsed
                remaining = self.total_texts - self.processed_texts
                return remaining / texts_per_sec if texts_per_sec > 0 else 0.0
        return 0.0

    def get_device_summary(self) -> Dict[str, str]:
        """Get a summary of processing by device."""
        summary = {}
        for device, stats in self.device_stats.items():
            processed = stats["processed"]
            avg_time = stats["total_time"] / stats["chunks"] if stats["chunks"] > 0 else 0
            texts_per_sec = processed / stats["total_time"] if stats["total_time"] > 0 else 0
            summary[device] = f"{processed:,} texts ({texts_per_sec:.1f}/s)"
        return summary

    def get_worker_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics for each individual worker."""
        summary = {}
        for worker_id, stats in self.worker_stats.items():
            processed = stats["processed"]
            chunks = stats["chunks"]
            total_time = stats["total_time"]
            rate = processed / total_time if total_time > 0 else 0
            summary[worker_id] = {
                "processed": processed,
                "chunks": chunks,
                "rate": rate,
                "device_tag": stats.get("device_tag", "cpu"),
            }
        return summary


# --- Public helper ----------------------------------------------------------
def _best_available_gpu() -> torch.device | None:
    """Return CUDA or MPS device if available, else ``None``."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


# --- Worker initialisation --------------------------------------------------
_WORKER_DEVICE: torch.device | None = None            # set once per process
_WORKER_BS_CPU = 32
_WORKER_BS_GPU = 64
_WORKER_TAG = "cpu"
_WORKER_ID = "worker-0"  # Unique ID for each worker (e.g., "gpu-0", "cpu-1", "cpu-2")
_PROGRESS_QUEUE = None  # Queue for sending per-batch progress updates to main process


def _worker_init(
    dev_q: "queue.Queue[Tuple[torch.device | None, str]]",
    bs_cpu: int,
    bs_gpu: int,
    progress_q=None,
) -> None:
    """
    Each child process pops **one** device and worker ID from *dev_q* and stores it in
    the global scope so that subsequent calls are cheap.

    Notes
    -----
    • We load the tokenizer + model **lazily** (first call per worker)
      to avoid the fork-after-initialise trap on macOS.
    • For CPU workers on Apple Silicon, we explicitly set torch.device("cpu")
      to prevent MPS memory allocation errors.
    • Each worker gets a unique ID (e.g., "gpu-0", "cpu-1") for granular progress tracking.
    • Progress queue is used to send per-batch progress updates to main process.
    """
    global _WORKER_DEVICE, _WORKER_BS_CPU, _WORKER_BS_GPU, _WORKER_TAG, _WORKER_ID, _PROGRESS_QUEUE

    try:
        _WORKER_DEVICE, _WORKER_ID = dev_q.get_nowait()
    except queue.Empty:
        _WORKER_DEVICE = torch.device("cpu")
        _WORKER_ID = "cpu-fallback"

    _WORKER_BS_CPU = bs_cpu
    _WORKER_BS_GPU = bs_gpu
    _PROGRESS_QUEUE = progress_q  # Store progress queue for per-batch updates

    # Get device type for tagging
    _WORKER_TAG = getattr(_WORKER_DEVICE, "type", "cpu")

    # IMPORTANT: For CPU workers on Apple Silicon, disable MPS fallback
    # to prevent MPS placeholder allocation errors
    if _WORKER_TAG == "cpu":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    # Reduce Hugging Face noise inside forks
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


# --- Lazy per-process cache --------------------------------------------------
# Backend cache: stores the tokenizer/model wrapper (Bert, Camembert, etc.)
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
# PyTorch model cache: stores the actual loaded PyTorch model to avoid reloading from disk
_PYTORCH_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}  # (model_path, device_str) -> loaded model


def _detect_model_type_from_path(model_path: str) -> str | None:
    """
    Detect the model type from a saved model's training_metadata.json or config.json.

    Returns
    -------
    str or None
        The model type (e.g., 'microsoft/deberta-v3-base', 'bert-base-cased') or None if not found.
    """
    import json
    import os

    # First try training_metadata.json (has the original model name)
    metadata_path = os.path.join(model_path, "training_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_type = metadata.get("model_type")
            if model_type:
                return model_type
        except Exception:
            pass

    # Fallback to config.json model_type
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("model_type")  # e.g., "deberta-v2", "bert"
        except Exception:
            pass

    return None


def _load_backend_for_model(model_path: str, lang: str, device: torch.device | None):
    """
    Instantiate the correct *BertBase* child **once per worker** based on the
    actual model type saved in the model directory.

    This function auto-detects the model architecture from the saved checkpoint
    and instantiates the correct backend class with proper tokenizer/model.

    Parameters
    ----------
    model_path : str
        Path to the saved fine-tuned model.
    lang :
        ISO-639-1 code : 'EN', 'FR', etc. (used as fallback).
    device :
        ``torch.device`` or ``None`` (CPU).

    Returns
    -------
    model_backend : LLMTool.bert_base.BertBase
    """
    # Import here to avoid circular imports
    from llm_tool.trainers.sota_models import get_model_class_for_name

    # Cache key includes model path to handle different model types
    key = (model_path, lang, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # Detect model type from saved model
    detected_model_type = _detect_model_type_from_path(model_path)

    if detected_model_type:
        # Use the detected model type to get the correct backend class
        backend_cls = get_model_class_for_name(detected_model_type)

        # Create backend with the correct model name
        if '/' in detected_model_type:
            # Full HuggingFace model name (e.g., 'microsoft/deberta-v3-base')
            backend = backend_cls(model_name=detected_model_type, device=device)
        else:
            # Short model type (e.g., 'deberta-v2') - use class defaults
            backend = backend_cls(device=device)
    else:
        # Fallback to language-based selection when model metadata is unavailable
        # Maps language codes to their best default model class
        # Uses AutoModelWrapper for languages without dedicated model classes
        _LANG_MAP = {
            # Languages with dedicated model classes
            "EN": Bert,
            "FR": Camembert,
            "AR": ArabicBert,
            "ZH": ChineseBert,
            "DE": GermanBert,
            "HI": HindiBert,
            "IT": ItalianBert,
            "PT": PortugueseBert,
            "RU": RussianBert,
            "ES": SpanishBert,
            "SV": SwedishBert,
            # Multilingual models
            "MULTI": XLMRoberta,
            # Additional languages - use XLMRoberta (multilingual) or AutoModelWrapper
            "JA": XLMRoberta,  # Japanese - XLM-R supports it
            "KO": XLMRoberta,  # Korean - XLM-R supports it
            "NL": XLMRoberta,  # Dutch - XLM-R supports it
            "PL": XLMRoberta,  # Polish - XLM-R supports it
            "TR": XLMRoberta,  # Turkish - XLM-R supports it
            "VI": XLMRoberta,  # Vietnamese - XLM-R supports it
            "TH": XLMRoberta,  # Thai - XLM-R supports it
            "ID": XLMRoberta,  # Indonesian - XLM-R supports it
            "DA": XLMRoberta,  # Danish - XLM-R supports it
            "NO": XLMRoberta,  # Norwegian - XLM-R supports it
            "FI": XLMRoberta,  # Finnish - XLM-R supports it
            "EL": XLMRoberta,  # Greek - XLM-R supports it
            "HE": XLMRoberta,  # Hebrew - XLM-R supports it
            "CS": XLMRoberta,  # Czech - XLM-R supports it
            "RO": XLMRoberta,  # Romanian - XLM-R supports it
            "BG": XLMRoberta,  # Bulgarian - XLM-R supports it
            "UK": XLMRoberta,  # Ukrainian - XLM-R supports it
            "HR": XLMRoberta,  # Croatian - XLM-R supports it
            "SR": XLMRoberta,  # Serbian - XLM-R supports it
            "SK": XLMRoberta,  # Slovak - XLM-R supports it
            "HU": XLMRoberta,  # Hungarian - XLM-R supports it
            "CA": XLMRoberta,  # Catalan - XLM-R supports it
            "EU": XLMRoberta,  # Basque - XLM-R supports it
            "GL": XLMRoberta,  # Galician - XLM-R supports it
        }
        # Use AutoModelWrapper as ultimate fallback for any unsupported language
        cls = _LANG_MAP.get(lang.upper(), AutoModelWrapper)
        backend = cls(device=device)

    # CRITICAL: Reload tokenizer from fine-tuned model path to ensure alignment
    # The backend was initialized with base model tokenizer, but the fine-tuned
    # model may have tokenizer files saved with save_pretrained()
    try:
        from transformers import AutoTokenizer
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            backend.tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        pass  # Keep original tokenizer if loading fails

    _MODEL_CACHE[key] = backend
    return backend


def _load_backend_for_language(lang: str, device: torch.device | None):
    """
    Instantiate the correct *BertBase* child **once per worker**.

    DEPRECATED: Use _load_backend_for_model instead when model_path is available.
    This function is kept for backward compatibility but may select the wrong
    model architecture if the trained model differs from the language default.

    Parameters
    ----------
    lang :
        ISO-639-1 code : 'EN', 'FR', etc.
    device :
        ``torch.device`` or ``None`` (CPU).

    Returns
    -------
    model_backend : LLMTool.bert_base.BertBase
    """
    key = ("", lang, str(device))  # Empty model_path for backward compatibility
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # Map language → concrete class (extend if you add more)
    _MAP = {
        "EN": Bert,
        "FR": Camembert,
        "AR": ArabicBert,
        "ZH": ChineseBert,
        "DE": GermanBert,
        "HI": HindiBert,
        "IT": ItalianBert,
        "PT": PortugueseBert,
        "RU": RussianBert,
        "ES": SpanishBert,
        "SV": SwedishBert,
        "MULTI": XLMRoberta,
    }
    cls = _MAP.get(lang.upper(), Bert)            # default to English BERT
    backend = cls(device=device)
    _MODEL_CACHE[key] = backend
    return backend


# --- Core predictor ---------------------------------------------------------
def _get_or_load_pytorch_model(backend, model_path: str, device) -> Any:
    """
    Load the PyTorch model once per worker and cache it.

    This avoids reloading from disk for every chunk, which is extremely slow.
    The model stays in memory for the lifetime of the worker process.

    Parameters
    ----------
    backend : BertBase subclass
        The backend (Bert, Camembert, etc.) with load_model method.
    model_path : str
        Path to the saved model.
    device : torch.device or None
        Device to move the model to.

    Returns
    -------
    model : torch.nn.Module
        The loaded and device-placed PyTorch model.
    """
    device_str = str(device) if device is not None else "cpu"
    cache_key = (model_path, device_str)

    if cache_key not in _PYTORCH_MODEL_CACHE:
        # Load model from disk (expensive - only do once per worker)
        model = backend.load_model(model_path)
        # Move to device (also expensive - only do once)
        target_device = device if device is not None else torch.device("cpu")
        model.to(target_device)
        model.eval()  # Set to evaluation mode
        _PYTORCH_MODEL_CACHE[cache_key] = model

    return _PYTORCH_MODEL_CACHE[cache_key]


def _predict_chunk(
    args: Tuple[Sequence[str], str, str],
) -> Tuple[np.ndarray, str, str]:
    """
    Worker-side prediction on a list of texts.

    This function is optimized for stability and performance:
    - Model is loaded ONCE per worker and cached in _PYTORCH_MODEL_CACHE
    - Backend (tokenizer) is loaded ONCE per worker and cached in _MODEL_CACHE
    - Only the data encoding happens per-chunk (lightweight operation)
    - Proper memory cleanup after inference to prevent fragmentation

    Parameters
    ----------
    args :
        (texts, model_path, lang)

    Returns
    -------
    tuple
        (probability matrix for the chunk, worker_id, device_tag)
        worker_id: unique worker identifier (e.g., "gpu-0", "cpu-1")
        device_tag: device type ('cpu' | 'cuda' | 'mps')
    """
    import gc
    import sys

    texts, model_path, lang = args
    device_tag = _WORKER_TAG  # Always set by _worker_init
    worker_id = _WORKER_ID    # Unique worker ID
    n_texts = len(texts)

    try:
        t0 = time.time()

        # Get or create backend (tokenizer + model wrapper) - CACHED per worker
        backend = _load_backend_for_model(model_path, lang, _WORKER_DEVICE)
        t1 = time.time()

        # Get or load PyTorch model - CACHED per worker (avoids disk reload)
        model = _get_or_load_pytorch_model(backend, model_path, _WORKER_DEVICE)
        t2 = time.time()

        # Log timing for first chunk (model loading) - only log if slow (>5s)
        load_time = t2 - t0
        if load_time > 5:  # First chunk takes longer due to model loading
            print(f"[{worker_id}] Model loaded in {load_time:.1f}s", file=sys.stderr, flush=True)

        # Encode texts to DataLoader (lightweight - only tokenization)
        # Use GPU batch size for non-CPU devices (mps, cuda)
        bs = _WORKER_BS_GPU if _WORKER_DEVICE.type != "cpu" else _WORKER_BS_CPU
        dl = backend.encode(list(texts), labels=None, batch_size=bs, progress_bar=False)
        t3 = time.time()

        # Run inference with per-batch progress updates
        from scipy.special import softmax as scipy_softmax, expit as scipy_sigmoid
        import json as _json

        # Detect multi-label from model config
        # Check both config.json (problem_type) and training_metadata.json (multi_label flag)
        is_multi_label = False
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = _json.load(f)
                    if config.get("problem_type") == "multi_label_classification":
                        is_multi_label = True

            # Also check training_metadata.json for explicit multi_label flag
            metadata_path = os.path.join(model_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = _json.load(f)
                    if metadata.get("multi_label", False):
                        is_multi_label = True
        except Exception:
            pass  # Fall back to single-label if detection fails

        all_logits = []
        target_device = _WORKER_DEVICE if _WORKER_DEVICE is not None else torch.device("cpu")

        model.eval()
        with torch.no_grad():
            for batch in dl:
                # DataLoader returns tuples: (input_ids, attention_mask) or (input_ids, attention_mask, labels)
                batch = tuple(t.to(target_device) for t in batch)
                if len(batch) == 3:
                    b_input_ids, b_input_mask, _ = batch
                else:
                    b_input_ids, b_input_mask = batch

                # Forward pass
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0].detach().cpu().numpy()
                all_logits.append(logits)

                # Send per-batch progress update to main process
                batch_size_actual = b_input_ids.shape[0]
                if _PROGRESS_QUEUE is not None:
                    try:
                        _PROGRESS_QUEUE.put_nowait((worker_id, device_tag, batch_size_actual))
                    except Exception:
                        pass  # Queue full or error - continue anyway

                del outputs
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()

        # Concatenate and apply activation function (sigmoid for multi-label, softmax for single-label)
        if all_logits:
            logits_concat = np.concatenate(all_logits, axis=0)
            if is_multi_label:
                # Multi-label: sigmoid gives independent probabilities per class
                probs = scipy_sigmoid(logits_concat)
            else:
                # Single-label: softmax gives mutually exclusive probabilities
                probs = scipy_softmax(logits_concat, axis=1)
        else:
            probs = np.array([])
        t4 = time.time()

        # Log chunk completion with timing (only show occasionally to reduce noise)
        # print(f"[{worker_id}] Chunk done: {n_texts} texts, encode={t3-t2:.1f}s, infer={t4-t3:.1f}s", file=sys.stderr)

        # Memory cleanup after inference (prevents gradual memory buildup)
        del dl
        gc.collect()

        # Clear GPU cache if applicable (prevents VRAM fragmentation)
        if _WORKER_DEVICE.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Note: MPS doesn't have empty_cache, gc.collect() above is sufficient

        return probs, worker_id, device_tag

    except Exception as e:
        # Log error but don't crash the worker - return empty result
        import sys
        print(f"[{worker_id}] Error processing chunk: {e}", file=sys.stderr)
        # Return empty array with expected shape (will be handled by caller)
        n_texts = len(texts)
        # Try to get num_labels from model config if possible
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path) as f:
                    n_labels = json.load(f).get("num_labels", 2)
            else:
                n_labels = 2
        except Exception:
            n_labels = 2
        return np.zeros((n_texts, n_labels), dtype=np.float32), worker_id, device_tag


# --- Public API -------------------------------------------------------------
def _get_model_size_gb(model_path: str) -> float:
    """
    Detect the size of a model from its path.

    Returns the estimated memory footprint in GB for loading one instance of the model.
    This includes model weights, tokenizer, and operational overhead.

    Parameters
    ----------
    model_path : str
        Path to the saved model directory

    Returns
    -------
    float
        Estimated memory per model instance in GB
    """
    import os
    from pathlib import Path

    # Default estimate if we can't detect
    default_estimate_gb = 1.5

    if not model_path or not os.path.exists(model_path):
        return default_estimate_gb

    model_dir = Path(model_path)

    # Look for model weight files
    weight_files = []
    for pattern in ["*.bin", "*.safetensors", "*.pt", "*.pth"]:
        weight_files.extend(model_dir.glob(pattern))
        weight_files.extend(model_dir.glob(f"**/{pattern}"))

    if not weight_files:
        return default_estimate_gb

    # Sum up the size of all weight files
    total_size_bytes = sum(f.stat().st_size for f in weight_files if f.is_file())
    model_size_gb = total_size_bytes / (1024 ** 3)

    # Add overhead for:
    # - Tokenizer: ~50-100MB
    # - Python process overhead: ~300MB
    # - Batch tensors during inference: ~200-500MB
    # - Safety margin: 20%
    overhead_gb = 0.5
    safety_factor = 1.2

    estimated_memory_gb = (model_size_gb + overhead_gb) * safety_factor

    return max(estimated_memory_gb, 0.5)  # Minimum 0.5GB


def _get_safe_worker_count(
    device_mode: str,
    gpu_device,
    batch_size_cpu: int,
    batch_size_gpu: int,
    model_path: str | None = None,
) -> int:
    """
    Calculate a safe number of workers based on available system resources and model size.

    This prevents crashes due to memory exhaustion when spawning too many workers,
    especially on systems with high CPU count but limited memory per model instance.

    Parameters
    ----------
    device_mode : str
        'cpu', 'gpu', or 'both'
    gpu_device :
        The GPU device or None
    batch_size_cpu : int
        Batch size for CPU workers
    batch_size_gpu : int
        Batch size for GPU workers
    model_path : str, optional
        Path to the model directory for size detection
    """
    import psutil

    n_cpu = max(os.cpu_count() or 1, 1)
    n_cpu_workers = max(n_cpu - 1, 1)

    # Get actual model size if path is provided, otherwise use conservative estimate
    if model_path:
        estimated_memory_per_worker_gb = _get_model_size_gb(model_path)
    else:
        # Fallback to conservative estimate for transformer models
        estimated_memory_per_worker_gb = 1.5

    try:
        mem_info = psutil.virtual_memory()
        available_memory_gb = mem_info.available / (1024**3)
        total_memory_gb = mem_info.total / (1024**3)

        # Reserve memory for system: 15% of total or 8GB, whichever is larger
        # Increased from 4GB to account for system stability
        system_reserve_gb = max(total_memory_gb * 0.15, 8.0)
        usable_memory_gb = max(available_memory_gb - system_reserve_gb, available_memory_gb * 0.7)

        # Calculate max workers based on memory
        max_workers_by_memory = max(1, int(usable_memory_gb / estimated_memory_per_worker_gb))

        # For Apple Silicon (MPS), be more conservative with CPU workers
        # because the GPU and CPU share memory (unified memory architecture)
        if gpu_device is not None and getattr(gpu_device, "type", "") == "mps":
            # MPS uses unified memory - reserve memory for GPU operations
            # Scale GPU reserve with model size
            gpu_reserve_gb = max(8.0, estimated_memory_per_worker_gb * 2)
            cpu_usable_memory_gb = max(usable_memory_gb - gpu_reserve_gb, usable_memory_gb * 0.5)
            max_workers_by_memory = max(1, int(cpu_usable_memory_gb / estimated_memory_per_worker_gb))

            # Hard cap based on total memory and model size to prevent OOM
            # For a 2GB model on 128GB system: allow ~40 workers max
            # For a 2GB model on 64GB system: allow ~20 workers max
            memory_based_cap = max(1, int((total_memory_gb * 0.7) / estimated_memory_per_worker_gb))
            max_workers_by_memory = min(max_workers_by_memory, memory_based_cap, n_cpu_workers)

    except Exception:
        # If psutil fails, use very conservative defaults based on model size
        if estimated_memory_per_worker_gb > 2.0:
            max_workers_by_memory = min(4, n_cpu_workers)
        elif estimated_memory_per_worker_gb > 1.0:
            max_workers_by_memory = min(8, n_cpu_workers)
        else:
            max_workers_by_memory = min(12, n_cpu_workers)

    # Apply the memory-based limit
    safe_cpu_workers = min(n_cpu_workers, max_workers_by_memory)

    return safe_cpu_workers


def parallel_predict(
    texts: Iterable[str],
    model_path: str,
    lang: str = "EN",
    *,
    parallel: bool = True,
    device_mode: str = "both",                    # 'cpu', 'gpu', 'both'
    batch_size_cpu: int = 32,
    batch_size_gpu: int = 64,
    show_progress: bool = True,
    chunk_size: int = 1024,
    progress_handler: Callable[[int, str], None] | None = None,
    enhanced_progress_handler: Callable[[int, str, str, float, InferenceProgress], None] | None = None,
    cpu_workers_override: int | None = None,
    gpu_workers_override: int | None = None,
    initialization_callback: Callable[[InferenceProgress], None] | None = None,
) -> np.ndarray:
    """
    Predict **probability distributions** for an arbitrary list of *texts*
    using the specified *model_path* (folder produced by
    :py:meth:`bert_base.BertBase.run_training`).

    The computation strategy is selected through *device_mode*:

    ============ =============================================================
    device_mode  Behaviour
    ------------ -------------------------------------------------------------
    "both"       one GPU worker  +  (N_CPU-1)    CPU workers
    "cpu"        (N_CPU-1) CPU workers, GPU idle
    "gpu"        one GPU worker, *no* extra CPU processes
    ============ =============================================================

    Parameters
    ----------
    texts :
        Anything convertible to a list of strings.
    model_path :
        Path to the *saved* fine-tuned model directory.
    lang :
        Language flag used to pick the right tokenizer/model subclass.
    parallel :
        Disable to force single-process evaluation (useful for debugging).
    device_mode :
        'cpu' | 'gpu' | 'both'  (case-insensitive).
    batch_size_cpu / batch_size_gpu :
        Per-device micro-batch sizes passed to :py:meth:`encode`.
    show_progress :
        Display a progress bar aggregated over all futures.
    chunk_size :
        Number of sentences fed to **each** worker task.  Keeping it large
        amortises serialisation cost; 1 k–2 k is usually safe.

    progress_handler :
        Optional callback receiving ``(processed_count, device_tag)`` for each completed
        chunk. Useful for external progress trackers (e.g. Rich). When provided you may
        wish to disable the internal tqdm bar via ``show_progress=False``.

    enhanced_progress_handler :
        Optional callback receiving ``(processed_count, worker_id, device_tag, chunk_time, progress_stats)``
        for detailed progress tracking including timing information and per-worker statistics.
        - worker_id: unique worker identifier (e.g., "gpu-0", "cpu-1", "cpu-13")
        - device_tag: device type ('cpu' | 'cuda' | 'mps')
        - progress_stats: ``InferenceProgress`` object with aggregated and per-worker metrics.

    initialization_callback :
        Optional callback receiving ``(progress_stats)`` when workers have finished initializing
        and the first result is received. Useful for updating UI after the model loading phase.

    Returns
    -------
    np.ndarray, shape=(n_texts, n_labels)
        Softmax probability matrix in the *original* order of *texts*.
    """
    device_mode = device_mode.lower()
    all_texts = list(texts)
    n = len(all_texts)
    if n == 0:
        raise ValueError("`texts` is empty.")

    # Initialize progress tracking
    progress_stats = InferenceProgress(
        total_texts=n,
        start_time=time.time(),
        chunk_size=chunk_size,
    )

    # ------------ Device planning -----------------------------------------
    n_cpu = max(os.cpu_count() or 1, 1)
    gpu_device = _best_available_gpu()

    # Calculate safe worker count based on available memory and model size
    safe_cpu_workers = _get_safe_worker_count(
        device_mode, gpu_device, batch_size_cpu, batch_size_gpu, model_path=model_path
    )

    # Apply user overrides if specified (but warn if it exceeds safe limit)
    if cpu_workers_override is not None:
        safe_cpu_workers = min(max(1, cpu_workers_override), n_cpu - 1)

    gpu_workers = 1  # Default GPU workers
    if gpu_workers_override is not None:
        gpu_workers = max(1, gpu_workers_override)

    # IMPORTANT: Use explicit torch.device("cpu") instead of None for CPU workers
    # to prevent MPS placeholder allocation errors on Apple Silicon
    cpu_device = torch.device("cpu")

    if device_mode == "cpu":
        n_workers = safe_cpu_workers if parallel else 1
        devices: List[torch.device] = [cpu_device] * n_workers
    elif device_mode == "gpu":
        if gpu_device is None:
            raise RuntimeError("No CUDA/MPS device detected for `device_mode='gpu'`.")
        n_workers = gpu_workers if parallel else 1
        devices = [gpu_device] * n_workers
    else:  # 'both'
        if gpu_device is None:
            # Fallback to pure CPU if no GPU present
            n_workers = safe_cpu_workers if parallel else 1
            devices = [cpu_device] * n_workers
        else:
            n_workers = safe_cpu_workers + gpu_workers if parallel else 1
            devices = [gpu_device] * gpu_workers + [cpu_device] * safe_cpu_workers

    # Update progress stats with worker info
    progress_stats.n_workers = n_workers
    progress_stats.n_gpu_workers = gpu_workers if (device_mode != "cpu" and gpu_device is not None) else 0
    progress_stats.n_cpu_workers = safe_cpu_workers if device_mode != "gpu" else 0

    if not parallel:
        # Run synchronously on the *first* requested device
        # Use model-aware backend loading to auto-detect the correct architecture
        device = devices[0]
        backend = _load_backend_for_model(model_path, lang, device)
        bs = batch_size_gpu if device.type != "cpu" else batch_size_cpu
        dl = backend.encode(all_texts, labels=None, batch_size=bs, progress_bar=show_progress)
        result = backend.predict_with_model(dl, model_path, proba=True, progress_bar=show_progress)
        if progress_handler is not None:
            progress_handler(n, device.type)
        return result

    # ------------ Multiprocessing setup ------------------------------------
    #   macOS + Metal (MPS) cannot safely fork a multi-threaded process, so we
    #   force the safer 'spawn' start method on Darwin. Same for Windows.
    use_spawn = sys.platform in {"win32", "darwin"}
    if not use_spawn and gpu_device is not None and getattr(gpu_device, "type", "") == "mps":
        use_spawn = True  # defensive: MPS is only available on macOS, but be explicit
    start_method = "spawn" if use_spawn else "fork"
    ctx = get_context(start_method)

    # Determine chunk size for work distribution
    min_chunk_size = 64
    target_chunks = n_workers * 4
    effective_chunk_size = max(min_chunk_size, min(chunk_size, n // max(target_chunks, 1)))
    progress_stats.chunk_size = effective_chunk_size

    # Prepare all chunks upfront
    chunk_list = []
    for i in range(0, n, effective_chunk_size):
        chunk = all_texts[i:i + effective_chunk_size]
        chunk_list.append((chunk, model_path, lang))

    n_chunks = len(chunk_list)

    # ========== FAIR ROUND-ROBIN DISTRIBUTION ==========
    # Problem: Pool.imap_unordered favors fast workers (GPU steals all work from CPUs)
    # Solution: Pre-assign chunks to workers in round-robin, then use Pool with
    # chunksize=chunks_per_worker to batch all of a worker's chunks together.
    # This ensures each worker gets its fair share before any worker can take more.

    # Calculate chunks per worker (round-robin assignment)
    base_chunks = n_chunks // n_workers
    extra_chunks = n_chunks % n_workers
    chunks_per_worker = [base_chunks + (1 if i < extra_chunks else 0) for i in range(n_workers)]

    if show_progress:
        print(
            f"[Parallel] {n_chunks} chunks (size={effective_chunk_size}) → {n_workers} workers",
            file=sys.stderr
        )
        print(
            f"[Parallel] Round-robin: {min(chunks_per_worker)}-{max(chunks_per_worker)} chunks/worker",
            file=sys.stderr
        )

    # Create device queue with worker IDs
    manager = ctx.Manager()
    dev_q = manager.Queue()
    progress_q = manager.Queue()  # Queue for per-batch progress updates from workers

    gpu_idx = 0
    cpu_idx = 0
    for d in devices:
        device_type = getattr(d, "type", "cpu")
        if device_type in ("cuda", "mps"):
            worker_id = f"gpu-{gpu_idx}"
            gpu_idx += 1
        else:
            worker_id = f"cpu-{cpu_idx}"
            cpu_idx += 1
        dev_q.put((d, worker_id))

    pool = ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(dev_q, batch_size_cpu, batch_size_gpu, progress_q),
        maxtasksperchild=None,
    )

    try:
        # ========== PHASE 1: WARM-UP - Ensure ALL workers load models ==========
        # Problem: CPUs take 40+ seconds to load model, GPU finishes quickly and steals all tasks
        # Solution: Use pool.map() with exactly n_workers tasks, chunksize=1
        # This ensures each worker gets exactly one task and must load its model

        # Always print warm-up status (critical for debugging worker issues)
        print(f"[Parallel] Warm-up: Loading models in {n_workers} workers...", file=sys.stderr, flush=True)

        warmup_start_time = time.time()

        # Create exactly n_workers warmup tasks - one per worker
        # With chunksize=1, pool distributes tasks round-robin at startup
        # Each worker pulls one task and loads its model while processing it
        warmup_task_count = min(n_workers, n)
        warmup_tasks = [([all_texts[i]], model_path, lang) for i in range(warmup_task_count)]

        # Use pool.map() which BLOCKS until ALL tasks complete
        # This means we wait for the slowest worker (CPU ~40s) before continuing
        print(f"[Parallel] Submitting {warmup_task_count} warmup tasks (1 per worker)...", file=sys.stderr, flush=True)
        warmup_results = pool.map(_predict_chunk, warmup_tasks, chunksize=1)

        # Collect results and track which workers responded
        worker_ids_seen = set()
        outputs: List[np.ndarray] = []

        # pool.map() blocks until ALL tasks complete, so warmup is already done
        warmup_elapsed = time.time() - warmup_start_time
        warmup_texts_used = warmup_task_count

        # Update progress stats BEFORE processing results (so callbacks get correct time)
        progress_stats.workers_initialized = True
        progress_stats.initialization_time = warmup_elapsed
        progress_stats.first_result_time = time.time()

        for probs, worker_id, device_tag in warmup_results:
            worker_ids_seen.add(worker_id)
            arr = np.asarray(probs)
            if arr.ndim == 1:
                arr = np.expand_dims(arr, 0)
            outputs.append(arr)

            # Update progress stats for warmup result (1 text per worker)
            warmup_chunk_time = warmup_elapsed / n_workers
            progress_stats.update_worker(worker_id, device_tag, 1, warmup_chunk_time)

            # Call progress handlers to update UI and remove "(loading)" label
            if progress_handler is not None:
                progress_handler(1, device_tag)
            if enhanced_progress_handler is not None:
                enhanced_progress_handler(1, worker_id, device_tag, warmup_chunk_time, progress_stats)

        # Report which workers loaded their models
        print(
            f"[Parallel] Warm-up done in {warmup_elapsed:.1f}s: {len(worker_ids_seen)}/{n_workers} workers ready",
            file=sys.stderr, flush=True
        )
        print(f"[Parallel] Workers: {sorted(worker_ids_seen)}", file=sys.stderr, flush=True)

        if len(worker_ids_seen) < n_workers:
            missing = n_workers - len(worker_ids_seen)
            print(f"[Parallel] Warning: {missing} worker(s) did not respond!", file=sys.stderr, flush=True)

        # Call initialization callback now that workers are ready
        if initialization_callback is not None:
            initialization_callback(progress_stats)

        # ========== PHASE 2: Main work distribution ==========
        # Skip texts already processed in warm-up
        remaining_chunks = []
        for i in range(warmup_texts_used, n, effective_chunk_size):
            chunk = all_texts[i:i + effective_chunk_size]
            remaining_chunks.append((chunk, model_path, lang))

        n_remaining = len(remaining_chunks)

        if show_progress:
            print(f"[Parallel] Phase 2: Distributing {n_remaining} chunks to {n_workers} workers...", file=sys.stderr)

        # Submit all chunks using apply_async for non-blocking progress updates
        async_results = []
        for chunk_task in remaining_chunks:
            ar = pool.apply_async(_predict_chunk, (chunk_task,))
            async_results.append(ar)

        failed_chunks: List[int] = []
        completed_chunks = 0
        total_chunks = len(async_results)

        # CRITICAL: Store results by their original index to preserve order
        # Results may complete out of order (GPU faster than CPU), but must be
        # reassembled in submission order for correct text-to-prediction alignment
        indexed_outputs: Dict[int, np.ndarray] = {}

        # Process results and batch-level progress updates
        while completed_chunks < total_chunks:
            # Drain the progress queue for batch-level updates
            batch_updates = 0
            while batch_updates < 200:  # Process up to 200 batch updates per iteration
                try:
                    worker_id, device_tag, batch_processed = progress_q.get_nowait()
                    batch_time = 0.05  # Approximate time per batch
                    progress_stats.update_worker(worker_id, device_tag, batch_processed, batch_time)

                    if progress_handler is not None:
                        progress_handler(batch_processed, device_tag)
                    if enhanced_progress_handler is not None:
                        enhanced_progress_handler(batch_processed, worker_id, device_tag, batch_time, progress_stats)

                    batch_updates += 1
                except Exception:
                    break  # Queue empty

            # Check for completed chunks (don't update progress again, just collect results)
            for idx, ar in enumerate(async_results):
                if ar is None:
                    continue
                if ar.ready():
                    try:
                        chunk_outputs, worker_id, device_tag = ar.get(timeout=0)
                        chunk_array = np.asarray(chunk_outputs)
                        if chunk_array.ndim == 1:
                            chunk_array = np.expand_dims(chunk_array, 0)
                        # Store by index to preserve original order
                        indexed_outputs[idx] = chunk_array
                    except Exception as e:
                        print(f"[!] Chunk {idx} failed: {e}", file=sys.stderr)
                        failed_chunks.append(idx)

                    async_results[idx] = None
                    completed_chunks += 1

            # Periodic garbage collection
            if completed_chunks > 0 and completed_chunks % 20 == 0:
                gc.collect()

            # Small sleep to avoid busy-waiting
            if completed_chunks < total_chunks:
                time.sleep(0.02)

        # Reassemble outputs in correct order
        for idx in sorted(indexed_outputs.keys()):
            outputs.append(indexed_outputs[idx])

        # Final drain of progress queue
        while True:
            try:
                worker_id, device_tag, batch_processed = progress_q.get_nowait()
                progress_stats.update_worker(worker_id, device_tag, batch_processed, 0.05)
                if progress_handler is not None:
                    progress_handler(batch_processed, device_tag)
                if enhanced_progress_handler is not None:
                    enhanced_progress_handler(batch_processed, worker_id, device_tag, 0.05, progress_stats)
            except Exception:
                break

        pool.close()
        pool.join()

        if failed_chunks:
            print(f"[!] {len(failed_chunks)} chunk(s) failed during inference", file=sys.stderr)

        if outputs:
            result = np.vstack(outputs)
        else:
            raise RuntimeError(f"All {n_chunks} inference chunks failed.")

    except Exception:
        pool.terminate()
        pool.join()
        raise

    return result

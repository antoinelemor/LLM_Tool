"""
PROJECT:
-------
LLMTool

TITLE:
------
parallel_inference.py

MAIN OBJECTIVE:
---------------
This script provides high-throughput parallel inference capabilities using multiple
GPUs/CPUs for efficient batch prediction, automatically distributing work across
available devices and managing memory efficiently.

Dependencies:
-------------
- torch (PyTorch with multiprocessing)
- numpy (array operations)
- tqdm (progress tracking)
- LLMTool.models (model implementations)

MAIN FEATURES:
--------------
1) Automatic device detection and allocation (CUDA, MPS, CPU)
2) Multi-GPU parallel processing with load balancing
3) Efficient batch processing with adaptive batch sizes
4) Memory-efficient model loading per worker
5) Progress tracking with tqdm
6) Support for all package models (BERT variants, SOTA models)
7) Automatic fallback to CPU when GPU unavailable
8) Fork-safe implementation for macOS compatibility

Author:
-------
Antoine Lemor
"""

from __future__ import annotations
import math
import os
import queue
import sys
from typing import Iterable, List, Sequence, Tuple, Union, Dict, Any

import numpy as np
import torch
from torch.multiprocessing import get_context            
from tqdm.auto import tqdm

# ---- Package-internal imports ---------------------------------------------
from llm_tool.trainers.models import (
    Bert, Camembert, ArabicBert, ChineseBert, GermanBert, HindiBert,
    ItalianBert, PortugueseBert, RussianBert, SpanishBert, SwedishBert,
    XLMRoberta,
)

# ---------------------------------------------------------------------------

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


def _worker_init(
    dev_q: "queue.Queue[torch.device | None]",
    bs_cpu: int,
    bs_gpu: int,
) -> None:
    """
    Each child process pops **one** device from *dev_q* and stores it in
    the global scope so that subsequent calls are cheap.

    Notes
    -----
    • We load the tokenizer + model **lazily** (first call per worker)
      to avoid the fork-after-initialise trap on macOS.
    """
    global _WORKER_DEVICE, _WORKER_BS_CPU, _WORKER_BS_GPU

    try:
        _WORKER_DEVICE = dev_q.get_nowait()
    except queue.Empty:
        _WORKER_DEVICE = None                     # Fallback to CPU

    _WORKER_BS_CPU = bs_cpu
    _WORKER_BS_GPU = bs_gpu

    # Reduce Hugging Face noise inside forks
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


# --- Lazy per-process cache --------------------------------------------------
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _load_backend_for_language(lang: str, device: torch.device | None):
    """
    Instantiate the correct *BertBase* child **once per worker**.

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
    key = (lang, str(device))
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
def _predict_chunk(
    args: Tuple[Sequence[str], str, str],
) -> List[np.ndarray]:
    """
    Worker-side prediction on a list of texts.

    Parameters
    ----------
    args :
        (texts, model_path, lang)

    Returns
    -------
    List[np.ndarray]  – probability vectors for each text
    """
    texts, model_path, lang = args
    backend = _load_backend_for_language(lang, _WORKER_DEVICE)

    bs = _WORKER_BS_GPU if (_WORKER_DEVICE and _WORKER_DEVICE.type != "cpu") else _WORKER_BS_CPU
    dl = backend.encode(list(texts), labels=None, batch_size=bs, progress_bar=False)
    probs = backend.predict_with_model(dl, model_path, proba=True, progress_bar=False)
    return probs


# --- Public API -------------------------------------------------------------
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

    # ------------ Device planning -----------------------------------------
    n_cpu = max(os.cpu_count() or 1, 1)
    n_cpu_workers = max(n_cpu - 1, 1)             # keep 1 core free
    gpu_device = _best_available_gpu()

    if device_mode == "cpu":
        n_workers = n_cpu_workers if parallel else 1
        devices: List[torch.device | None] = [None] * n_workers
    elif device_mode == "gpu":
        if gpu_device is None:
            raise RuntimeError("No CUDA/MPS device detected for `device_mode='gpu'`.")
        n_workers = 1 if parallel else 1          # forced single worker
        devices = [gpu_device]
    else:  # 'both'
        if gpu_device is None:
            # Fallback to pure CPU if no GPU present
            n_workers = n_cpu_workers if parallel else 1
            devices = [None] * n_workers
        else:
            n_workers = n_cpu_workers + 1 if parallel else 1
            devices = [gpu_device] + [None] * n_cpu_workers

    if not parallel:
        # Run synchronously on the *first* requested device
        backend = _load_backend_for_language(lang, devices[0])
        bs = batch_size_gpu if devices[0] and devices[0].type != "cpu" else batch_size_cpu
        dl = backend.encode(all_texts, labels=None, batch_size=bs, progress_bar=show_progress)
        return backend.predict_with_model(dl, model_path, proba=True, progress_bar=show_progress)

    # ------------ Multiprocessing pool ------------------------------------
    #   We deliberately use the *spawn* context on Windows to avoid
    #   pickling issues; fork elsewhere for speed.
    ctx = get_context("spawn" if sys.platform == "win32" else "fork")
    manager = ctx.Manager()
    dev_q = manager.Queue()
    for d in devices:
        dev_q.put(d)

    pool = ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(dev_q, batch_size_cpu, batch_size_gpu),
    )

    try:
        # Submit chunks
        jobs = []
        for i in range(0, n, chunk_size):
            chunk = all_texts[i : i + chunk_size]
            jobs.append(pool.apply_async(_predict_chunk, args=((chunk, model_path, lang),)))

        pool.close()

        # Collect
        if show_progress:
            jobs_iter = tqdm(jobs, desc="Parallel inference", unit="job")
        else:
            jobs_iter = jobs

        outputs: List[np.ndarray] = []
        for job in jobs_iter:
            outputs.extend(job.get())

        result = np.vstack(outputs)               # (n_texts, n_labels)

    finally:
        pool.terminate()
        pool.join()

    return result

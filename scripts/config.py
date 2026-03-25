#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
config.py

MAIN OBJECTIVE:
---------------
Shared configuration for all paper analysis scripts.
Centralizes paths, model definitions, task definitions, color palettes,
and matplotlib style settings for reproducible figure generation.

Dependencies:
-------------
- os
- pathlib
- matplotlib

MAIN FEATURES:
--------------
1) Project and output path definitions
2) Model metadata (session IDs, display names, parameters, inference times)
3) Annotation task definitions (column mappings, types)
4) Training hyperparameter registry
5) Color palette and matplotlib publication-quality style configuration

Author:
-------
Antoine Lemor
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = Path(__file__).resolve().parents[1]

# ── Input paths ───────────────────────────────────────────────────────────────
BENCHMARK_CSV = PROJECT_ROOT / "logs" / "annotation_studio" / "llm_tool_final_20260323_175824" / "annotations.csv"
TRAINING_ARENA_DIR = PROJECT_ROOT / "logs" / "training_arena"
ANNOTATOR_DIR = PROJECT_ROOT / "logs" / "annotator"

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR = PAPER_ROOT / "data"
FIGURES_DIR = PAPER_ROOT / "figures"
RESULTS_DIR = PAPER_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

for d in [DATA_DIR, FIGURES_DIR, RESULTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────
# Mapping from internal session IDs to clean display names
MODELS = {
    "gemma": {
        "session_id": "GEMMA_EXTRA_EXTRA_LARGE_20260224_103240",
        "annotator_session": "GEMMA_EXTRA_EXTRA_LARGE_20260127_093741",
        "display_name": "Gemma 3 (27B)",
        "short_name": "Gemma3",
        "params": "27B",
        "disk_size": "17 GB",
        "quantization": "Full",
        "context_window": "128k",
        "provider": "Ollama",
        "col_prefix": "gemma_extra_extra_large_20260224_103240",
        "mean_inference_time": 19.73,  # seconds per sample
        "total_annotations": 38_451,
    },
    "nemotron": {
        "session_id": "NEMOTRON_EXTRA_EXTRA_LARGE_20260216_075531",
        "annotator_session": "NEMOTRON_EXTRA_EXTRA_LARGE_20260129_162221",
        "display_name": "Nemotron (49B)",
        "short_name": "Nemotron",
        "params": "49B",
        "disk_size": "42 GB",
        "quantization": "Full",
        "context_window": "4k",
        "provider": "Ollama",
        "col_prefix": "nemotron_extra_extra_large_20260216_075531",
        "mean_inference_time": 78.97,  # from paper's original table
        "total_annotations": 58_672,
    },
    "mixtral": {
        "session_id": "MIXTRAL_EXTRA_EXTRA_LARGE_20260214_113559",
        "annotator_session": "MIXTRAL_EXTRA_EXTRA_LARGE_20260131_225510",
        "display_name": "Mixtral (8×22B)",
        "short_name": "Mixtral",
        "params": "8×22B",
        "disk_size": "80 GB",
        "quantization": "Full",
        "context_window": "64k",
        "provider": "Ollama",
        "col_prefix": "mixtral_extra_extra_large_20260214_113559",
        "mean_inference_time": 5.47,
        "total_annotations": 38_451,
    },
    "gpt_oss": {
        "session_id": "GPT_OSS_EXTRA_EXTRA_LARGE_20260210_095322",
        "annotator_session": "GPT-OSS_EXTRA_EXTRA_LARGE_20260127_175500",
        "display_name": "GPT-OSS (120B)",
        "short_name": "GPT-OSS",
        "params": "120B",
        "disk_size": "65 GB",
        "quantization": "Q8",
        "context_window": "128k",
        "provider": "Ollama",
        "col_prefix": "gpt_oss_extra_extra_large_20260210_095322",
        "mean_inference_time": 7.70,
        "total_annotations": 38_451,
    },
    "gpt5": {
        "session_id": "GPT-5_EXTRA_EXTRA_LARGE_20260206_153147",
        "annotator_session": "GPT_5_EXTRA_LARGE_20260124_153207",
        "display_name": "GPT-5",
        "short_name": "GPT-5",
        "params": "N/A",
        "disk_size": "N/A",
        "quantization": "N/A",
        "context_window": "128k",
        "provider": "OpenAI",
        "col_prefix": "gpt_5_extra_extra_large_20260206_153147",
        "mean_inference_time": None,  # API-based, timing not comparable
        "total_annotations": 38_451,
    },
}

# Ordered list for consistent plotting
MODEL_ORDER = ["gemma", "nemotron", "mixtral", "gpt_oss", "gpt5"]

# ── Annotation tasks ──────────────────────────────────────────────────────────
TASKS = {
    "themes": {
        "display_name": "Policy Themes",
        "col_suffix": "themes_long",
        "consensus_col": "consensus_themes",
        "type": "multilabel",
        "consensus_prefix": "theme_",
        "pred_prefix": "themes_long_",
    },
    "sentiment": {
        "display_name": "Sentiment",
        "col_suffix": "sentiment_long",
        "consensus_col": "consensus_sentiment",
        "type": "multiclass",
        "consensus_prefix": "sentiment_",
        "pred_prefix": "sentiment_long_",
    },
    "specific_themes": {
        "display_name": "Specific Themes",
        "col_suffix": "specific_themes_long",
        "consensus_col": "consensus_specific_themes",
        "type": "multilabel",
        "consensus_prefix": "specific_themes_",
        "pred_prefix": "specific_themes_long_",
    },
    "political_parties": {
        "display_name": "Political Parties",
        "col_suffix": "political_parties_long",
        "consensus_col": "consensus_political_parties",
        "type": "multilabel",
        "consensus_prefix": "political_parties_",
        "pred_prefix": "political_parties_long_",
    },
}

TASK_ORDER = ["themes", "sentiment", "political_parties", "specific_themes"]

# ── Training info ─────────────────────────────────────────────────────────────
TRAINING_MODEL = "xlm-roberta-base"
TRAINING_SAMPLES = 38_451  # per model (except Nemotron: 58,672)

HYPERPARAMETERS = {
    "learning_rate": 2e-5,
    "batch_size": 256,
    "epochs": 25,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "optimizer": "AdamW",
    "scheduler": "Linear",
    "max_seq_length": 512,
    "early_stopping_patience": 3,
    "seed": 42,
}

# ── Color palette (SOTA academic style) ──────────────────────────────────────
# Based on Tableau 10 + muted tones for accessibility
COLORS = {
    "gemma":    "#4E79A7",  # steel blue
    "nemotron": "#F28E2B",  # orange
    "mixtral":  "#E15759",  # coral red
    "gpt_oss":  "#76B7B2",  # teal
    "gpt5":     "#59A14F",  # green
    "human":    "#B07AA1",  # muted purple
}

TASK_COLORS = {
    "themes":            "#4E79A7",
    "sentiment":         "#F28E2B",
    "political_parties": "#E15759",
    "specific_themes":   "#59A14F",
}

# ── Matplotlib style settings ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.figsize": (7, 4.5),
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })


def save_figure(fig, name, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
    plt.close(fig)
    print(f"  Saved: {name} ({', '.join(formats)})")

#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
training_logger.py

MAIN OBJECTIVE:
---------------
Centralized logging module for training metrics with standardized schemas,
hyperparameter tracking, and session manifest generation.

Dependencies:
-------------
- dataclasses
- json
- csv
- pathlib

MAIN FEATURES:
--------------
1) TrainingHyperparameters dataclass for hyperparameter tracking
2) ModelMetadata dataclass for model identification
3) Standardized CSV headers for consistent logging
4) Session manifest generation for experiment reproducibility
5) Multi-label specific metrics support

Author:
-------
Antoine Lemor
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import csv
import re
import platform
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES FOR TRAINING METADATA
# ============================================================================

@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters used for training, enabling experiment reproducibility.

    All parameters that affect training behavior should be captured here.
    """
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    seed: int = 42
    multi_label_threshold: float = 0.5
    # Reinforced learning parameters
    reinforced_learning: bool = False
    reinforced_epochs: int = 2
    reinforced_f1_threshold: float = 0.7
    rl_oversample_factor: float = 2.0
    rl_class_weight_factor: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingHyperparameters':
        """Create instance from dictionary."""
        # Filter to only include valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class ModelMetadata:
    """
    Complete metadata for a trained model, enabling identification and reproducibility.
    """
    huggingface_model_name: str  # e.g., "xlm-roberta-base", "bert-base-multilingual-cased"
    model_class: str  # e.g., "XLMRobertaLongformer", "BertBase"
    training_mode: str  # "multi-class" | "multi-label" | "one-vs-all" | "binary"
    num_labels: int
    label_names: List[str]
    category: str  # e.g., "political_parties_long", "themes"
    language: Optional[str] = None  # e.g., "EN", "FR", "MULTI"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create instance from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class MultiLabelMetrics:
    """
    Specialized metrics for multi-label classification tasks.
    """
    hamming_loss: float = 0.0
    subset_accuracy: float = 0.0  # Exact match accuracy
    jaccard_score: float = 0.0  # Intersection over union
    micro_f1: float = 0.0
    macro_f1: float = 0.0
    threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================================================
# CSV HEADER DEFINITIONS
# ============================================================================

def get_base_csv_headers() -> List[str]:
    """
    Get the base CSV headers for training metrics.

    Returns headers that are always present regardless of training mode.
    """
    return [
        "huggingface_model",      # HuggingFace model identifier
        "model_class",            # Python class name
        "training_mode",          # multi-class, multi-label, one-vs-all, binary
        "category",               # Category/label group being trained
        "language",               # Language code (EN, FR, MULTI)
        "timestamp",              # When this row was written
        "epoch",                  # Training epoch
        "train_loss",             # Training loss
        "val_loss",               # Validation loss
        "combined_score",         # Combined metric for model selection
        "accuracy",               # Overall accuracy
    ]


def sanitize_class_name(name: str) -> str:
    """Sanitize a class name for use in CSV column headers.

    Replaces whitespace with underscores, removes characters that could break
    CSV parsing, collapses repeated underscores, and truncates to 50 chars.
    """
    sanitized = str(name).strip()
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = re.sub(r'[,;"\'\#\[\]\(\)\{\}]', '', sanitized)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    if len(sanitized) > 50:
        sanitized = sanitized[:50].rstrip('_')
    return sanitized or 'unknown'


def get_per_class_headers(num_labels: int, class_names: Optional[List[str]] = None) -> List[str]:
    """
    Get per-class metric headers for the given number of labels.

    Args:
        num_labels: Number of classification labels
        class_names: Optional list of class names to use instead of numeric indices

    Returns:
        List of header names for per-class metrics (precision, recall, f1, support)
    """
    headers = []
    for i in range(num_labels):
        suffix = sanitize_class_name(class_names[i]) if class_names and i < len(class_names) else str(i)
        headers.extend([
            f"precision_{suffix}",
            f"recall_{suffix}",
            f"f1_{suffix}",
            f"support_{suffix}"
        ])
    return headers


def get_aggregate_headers() -> List[str]:
    """
    Get aggregate metric headers.
    """
    return [
        "macro_f1",
        "micro_f1",
    ]


def get_multi_label_headers() -> List[str]:
    """
    Get multi-label specific metric headers.
    """
    return [
        "hamming_loss",
        "subset_accuracy",
        "jaccard_score",
        "multi_label_threshold",
    ]


def get_hyperparameter_headers() -> List[str]:
    """
    Get hyperparameter headers for CSV logging.
    """
    return [
        "learning_rate",
        "batch_size",
        "warmup_ratio",
    ]


def get_training_csv_headers() -> List[str]:
    """
    Get full training.csv headers for epoch-level logging.

    Note: Per-class metrics are added dynamically based on num_labels.
    """
    return [
        "huggingface_model",
        "model_class",
        "training_mode",
        "category",
        "language",
        "timestamp",
        "session_id",
        "epoch",
        "train_loss",
        "val_loss",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "learning_rate",
        "samples_per_second",
        "phase",  # "normal" or "reinforced"
    ]


def get_best_csv_headers(num_labels: int, languages: Optional[List[str]] = None,
                         include_multi_label: bool = False,
                         class_names: Optional[List[str]] = None) -> List[str]:
    """
    Get complete best.csv headers for best model logging.

    Args:
        num_labels: Number of classification labels
        languages: Optional list of languages for language-specific metrics
        include_multi_label: Whether to include multi-label specific metrics
        class_names: Optional list of class names to use instead of numeric indices

    Returns:
        Complete list of headers for best.csv
    """
    headers = get_base_csv_headers()
    headers.extend(get_per_class_headers(num_labels, class_names=class_names))
    headers.extend(get_aggregate_headers())

    # Multi-label specific metrics
    if include_multi_label:
        headers.extend(get_multi_label_headers())

    # Hyperparameters
    headers.extend(get_hyperparameter_headers())

    # Language-specific metrics
    if languages:
        for lang in sorted(languages):
            lang_upper = lang.upper()
            headers.append(f"{lang_upper}_accuracy")
            for i in range(num_labels):
                suffix = sanitize_class_name(class_names[i]) if class_names and i < len(class_names) else str(i)
                headers.extend([
                    f"{lang_upper}_precision_{suffix}",
                    f"{lang_upper}_recall_{suffix}",
                    f"{lang_upper}_f1_{suffix}",
                    f"{lang_upper}_support_{suffix}"
                ])
            headers.append(f"{lang_upper}_macro_f1")

    # Final columns
    headers.extend([
        "saved_model_path",
        "training_phase",
        "training_time_seconds",
    ])

    return headers


# ============================================================================
# CSV WRITING UTILITIES
# ============================================================================

def write_csv_with_legend(
    filepath: Path,
    headers: List[str],
    class_legend: str,
    mode: str = 'w'
) -> None:
    """
    Initialize or append to a CSV file with a class legend comment.

    Args:
        filepath: Path to the CSV file
        headers: List of column headers
        class_legend: Legend string (e.g., "CLASS_LEGEND: 0=class_a, 1=class_b")
        mode: File mode ('w' for write, 'a' for append)
    """
    with open(filepath, mode=mode, newline='', encoding='utf-8') as f:
        if mode == 'w':
            f.write(f"# {class_legend}\n")
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(headers)


def build_class_legend(label_names: List[str]) -> str:
    """
    Build a CLASS_LEGEND string from label names.

    Args:
        label_names: List of label names in order

    Returns:
        Legend string like "CLASS_LEGEND: 0=label_a, 1=label_b, 2=label_c"
    """
    legend_parts = [f"{i}={name}" for i, name in enumerate(label_names)]
    return "CLASS_LEGEND: " + ", ".join(legend_parts)


# ============================================================================
# SESSION MANIFEST GENERATION
# ============================================================================

@dataclass
class CategoryTrainingResult:
    """Results for a single category training."""
    name: str
    huggingface_model: str
    training_mode: str
    num_labels: int
    label_names: List[str]
    best_epoch: int
    best_macro_f1: float
    best_accuracy: float = 0.0
    languages: List[str] = field(default_factory=list)
    training_time_seconds: float = 0.0
    hyperparameters: Optional[Dict[str, Any]] = None


@dataclass
class SessionManifest:
    """
    Complete manifest for a training session.

    This provides a single source of truth for all training metadata,
    enabling experiment reproducibility and analysis.
    """
    schema_version: str = "2.0"
    session_id: str = ""
    created_at: str = ""
    completed_at: Optional[str] = None
    status: str = "in_progress"  # "in_progress", "completed", "failed"

    # Environment info
    python_version: str = ""
    torch_version: str = ""
    transformers_version: str = ""
    device: str = ""

    # Training configuration
    training_mode: str = ""  # "normal", "parallel", "benchmark"
    training_approach: str = ""  # "multi-class", "multi-label", "one-vs-all"
    multi_label: bool = False

    # Default hyperparameters
    default_hyperparameters: Optional[Dict[str, Any]] = None

    # Results
    categories_trained: List[Dict[str, Any]] = field(default_factory=list)

    # Generated files
    files_generated: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "environment": {
                "python_version": self.python_version,
                "torch_version": self.torch_version,
                "transformers_version": self.transformers_version,
                "device": self.device,
            },
            "training_config": {
                "mode": self.training_mode,
                "approach": self.training_approach,
                "multi_label": self.multi_label,
            },
            "default_hyperparameters": self.default_hyperparameters,
            "categories_trained": self.categories_trained,
            "files_generated": self.files_generated,
        }

    def save(self, filepath: Path) -> None:
        """Save manifest to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Session manifest saved to {filepath}")


def generate_session_manifest(
    session_id: str,
    output_dir: Path,
    training_mode: str = "normal",
    training_approach: str = "multi-class",
    multi_label: bool = False,
    hyperparameters: Optional[TrainingHyperparameters] = None,
    categories_results: Optional[List[CategoryTrainingResult]] = None,
    files_generated: Optional[Dict[str, str]] = None,
    device: str = "cpu",
) -> SessionManifest:
    """
    Generate a complete session manifest.

    Args:
        session_id: Unique session identifier
        output_dir: Directory to save the manifest
        training_mode: "normal", "parallel", or "benchmark"
        training_approach: "multi-class", "multi-label", or "one-vs-all"
        multi_label: Whether multi-label classification was used
        hyperparameters: Default hyperparameters used
        categories_results: List of training results per category
        files_generated: Dictionary of generated file paths
        device: Device used for training

    Returns:
        SessionManifest instance
    """
    # Get version info
    python_version = platform.python_version()

    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "unknown"

    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "unknown"

    manifest = SessionManifest(
        session_id=session_id,
        created_at=datetime.now().isoformat(),
        python_version=python_version,
        torch_version=torch_version,
        transformers_version=transformers_version,
        device=device,
        training_mode=training_mode,
        training_approach=training_approach,
        multi_label=multi_label,
        default_hyperparameters=hyperparameters.to_dict() if hyperparameters else None,
        categories_trained=[
            {
                "name": r.name,
                "huggingface_model": r.huggingface_model,
                "training_mode": r.training_mode,
                "num_labels": r.num_labels,
                "label_names": r.label_names,
                "best_epoch": r.best_epoch,
                "best_macro_f1": r.best_macro_f1,
                "best_accuracy": r.best_accuracy,
                "languages": r.languages,
                "training_time_seconds": r.training_time_seconds,
                "hyperparameters": r.hyperparameters,
            }
            for r in (categories_results or [])
        ],
        files_generated=files_generated or {},
    )

    return manifest


def finalize_session_manifest(
    manifest: SessionManifest,
    output_dir: Path,
    status: str = "completed"
) -> Path:
    """
    Finalize and save the session manifest.

    Args:
        manifest: The SessionManifest to finalize
        output_dir: Directory to save the manifest
        status: Final status ("completed" or "failed")

    Returns:
        Path to the saved manifest file
    """
    manifest.completed_at = datetime.now().isoformat()
    manifest.status = status

    manifest_path = output_dir / "session_manifest.json"
    manifest.save(manifest_path)

    return manifest_path


# ============================================================================
# TRAINING METADATA HELPERS
# ============================================================================

def create_training_metadata(
    model_metadata: ModelMetadata,
    hyperparameters: TrainingHyperparameters,
    epoch: int,
    combined_metric: float,
    macro_f1: float,
    accuracy: float,
    training_phase: str = "normal",
    multi_label_metrics: Optional[MultiLabelMetrics] = None,
    final_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a complete training_metadata.json content.

    This replaces the fragmented metadata with a unified structure.

    Args:
        model_metadata: Model identification metadata
        hyperparameters: Training hyperparameters
        epoch: Best epoch number
        combined_metric: Combined score used for model selection
        macro_f1: Macro-averaged F1 score
        accuracy: Overall accuracy
        training_phase: "normal" or "reinforced"
        multi_label_metrics: Optional multi-label specific metrics
        final_metrics: Optional additional metrics dict

    Returns:
        Complete metadata dictionary ready for JSON serialization
    """
    metadata = {
        # Model identification
        "huggingface_model_name": model_metadata.huggingface_model_name,
        "model_class": model_metadata.model_class,
        "training_mode": model_metadata.training_mode,
        "num_labels": model_metadata.num_labels,
        "label_names": model_metadata.label_names,
        "category": model_metadata.category,
        "language": model_metadata.language,

        # Training info
        "epoch": epoch,
        "combined_metric": combined_metric,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "training_phase": training_phase,
        "timestamp": datetime.now().isoformat(),

        # Hyperparameters (for reproducibility)
        "hyperparameters": hyperparameters.to_dict(),
    }

    # Add multi-label metrics if applicable
    if multi_label_metrics:
        metadata["multi_label_metrics"] = multi_label_metrics.to_dict()

    # Add final metrics if provided
    if final_metrics:
        metadata["final_metrics"] = final_metrics

    return metadata


def determine_training_mode(
    multi_label: bool = False,
    num_labels: int = 2,
    is_one_vs_all: bool = False
) -> str:
    """
    Determine the training mode string based on configuration.

    Args:
        multi_label: Whether multi-label classification is enabled
        num_labels: Number of labels/classes
        is_one_vs_all: Whether using one-vs-all binary classification

    Returns:
        Training mode string: "multi-label", "multi-class", "one-vs-all", or "binary"
    """
    if multi_label:
        return "multi-label"
    elif is_one_vs_all:
        return "one-vs-all"
    elif num_labels > 2:
        return "multi-class"
    else:
        return "binary"

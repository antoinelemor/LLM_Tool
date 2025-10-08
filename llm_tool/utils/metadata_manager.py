"""
Comprehensive Metadata Management System for Training Arena
Ensures complete session persistence and perfect resume capability.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import hashlib
import platform
import sys

logger = logging.getLogger(__name__)

class MetadataManager:
    """
    Manages comprehensive training session metadata for perfect resume capability.

    This class ensures that EVERY parameter affecting training is captured,
    allowing sessions to be perfectly resumed from any interruption point.
    """

    # Version for metadata format (for backward compatibility)
    METADATA_VERSION = "2.0"

    def __init__(self, session_id: str, base_dir: Path = Path("logs/training_arena")):
        """
        Initialize metadata manager for a training session.

        Args:
            session_id: Unique session identifier
            base_dir: Base directory for training arena logs
        """
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.metadata_dir = self.base_dir / session_id / "training_session_metadata"
        self.metadata_path = self.metadata_dir / "training_metadata.json"
        self.backup_path = self.metadata_dir / "training_metadata_backup.json"
        self.checkpoint_dir = self.metadata_dir / "checkpoints"

        # Ensure directories exist
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_comprehensive_metadata(
        self,
        bundle: Any,  # TrainingDataBundle
        mode: str,
        model_config: Dict[str, Any],
        quick_params: Optional[Dict[str, Any]] = None,
        execution_status: Optional[Dict[str, Any]] = None,
        runtime_params: Optional[Dict[str, Any]] = None,
        training_context: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save COMPREHENSIVE metadata capturing ALL parameters for perfect resume.

        This method captures:
        - All dataset configuration and preprocessing settings
        - Complete model selection and configuration
        - All training hyperparameters
        - Reinforced learning settings
        - Split configurations
        - Language-specific settings
        - Text processing parameters
        - Environment information
        - Progress and checkpoint information
        """

        # Build comprehensive metadata structure
        metadata = {
            # === METADATA VERSIONING ===
            "metadata_version": self.METADATA_VERSION,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),

            # === SESSION INFORMATION ===
            "training_session": {
                "session_id": self.session_id,
                "timestamp": self.session_id,
                "tool_version": self._get_tool_version(),
                "workflow": f"Training Arena - {mode.capitalize()}",
                "mode": mode,
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "user": self._get_safe_username()
            },

            # === DATASET CONFIGURATION ===
            "dataset_config": self._extract_dataset_config(bundle),

            # === LANGUAGE CONFIGURATION ===
            "language_config": self._extract_language_config(bundle),

            # === TEXT ANALYSIS ===
            "text_analysis": self._extract_text_analysis(bundle),

            # === SPLIT CONFIGURATION ===
            "split_config": self._extract_split_config(bundle),

            # === LABEL CONFIGURATION ===
            "label_config": self._extract_label_config(bundle),

            # === MODEL CONFIGURATION ===
            "model_config": self._build_complete_model_config(
                model_config, quick_params, runtime_params
            ),

            # === TRAINING PARAMETERS ===
            "training_params": self._extract_training_params(
                model_config, quick_params, runtime_params
            ),

            # === REINFORCED LEARNING CONFIGURATION ===
            "reinforced_learning_config": self._extract_rl_config(
                model_config, quick_params, runtime_params
            ),

            # === EXECUTION STATUS ===
            "execution_status": execution_status or self._default_execution_status(),

            # === OUTPUT PATHS ===
            "output_paths": self._extract_output_paths(bundle),

            # === PREPROCESSING SETTINGS ===
            "preprocessing": self._extract_preprocessing(bundle),

            # === ADVANCED SETTINGS ===
            "advanced_settings": self._extract_advanced_settings(
                model_config, quick_params, bundle
            ),

            # === CHECKPOINT INFORMATION ===
            "checkpoints": self._extract_checkpoint_info(),

            # === TRAINING CONTEXT ===
            "training_context": training_context or {}
        }

        # Save metadata with backup
        self._save_with_backup(metadata)

        return self.metadata_path

    def load_metadata(self, validate: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load metadata with validation and error recovery.

        Args:
            validate: Whether to validate loaded metadata

        Returns:
            Loaded metadata or None if failed
        """
        try:
            # Try primary file first
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if validate and not self._validate_metadata(metadata):
                    logger.warning("Primary metadata validation failed, trying backup")
                    return self._load_backup()

                return metadata

            # Try backup if primary doesn't exist
            return self._load_backup()

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return self._load_backup()

    def update_metadata(self, **updates) -> None:
        """
        Update existing metadata with new information.

        Supports nested updates and maintains metadata integrity.
        """
        try:
            # Load existing metadata
            metadata = self.load_metadata(validate=False) or {}

            # Update sections
            for section, data in updates.items():
                if section in metadata and isinstance(metadata[section], dict) and isinstance(data, dict):
                    # Merge dictionaries
                    metadata[section].update(data)
                else:
                    # Replace section
                    metadata[section] = data

            # Update timestamp
            metadata["last_updated"] = datetime.now().isoformat()

            # Save with backup
            self._save_with_backup(metadata)

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")

    def save_checkpoint(self, checkpoint_name: str, checkpoint_data: Dict[str, Any]) -> None:
        """
        Save a training checkpoint for resume capability.

        Args:
            checkpoint_name: Name of checkpoint (e.g., "epoch_5", "model_bert")
            checkpoint_data: Data to save in checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

        checkpoint = {
            "checkpoint_name": checkpoint_name,
            "created_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "data": checkpoint_data
        }

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    # === PRIVATE HELPER METHODS ===

    def _extract_dataset_config(self, bundle) -> Dict[str, Any]:
        """Extract comprehensive dataset configuration."""
        # Try to get format_type from metadata first, then from bundle directly
        format_type = 'unknown'
        if hasattr(bundle, 'metadata') and bundle.metadata:
            format_type = bundle.metadata.get('format_type', bundle.metadata.get('format', 'unknown'))
        if format_type == 'unknown' and hasattr(bundle, 'format_type'):
            format_type = getattr(bundle, 'format_type', 'unknown')

        config = {
            "primary_file": str(bundle.primary_file) if hasattr(bundle, 'primary_file') and bundle.primary_file else None,
            "format_type": format_type,
            "format": format_type,  # Keep both for backward compatibility
            "strategy": getattr(bundle, 'strategy', 'single-label'),
            "text_column": getattr(bundle, 'text_column', 'text'),
            "label_column": getattr(bundle, 'label_column', 'label'),
            "total_samples": len(bundle.samples) if hasattr(bundle, 'samples') and bundle.samples else 0,
            "has_validation": hasattr(bundle, 'val_samples') and bundle.val_samples,
            "has_test": hasattr(bundle, 'test_samples') and bundle.test_samples,
            "encoding": getattr(bundle, 'encoding', 'utf-8')
        }

        # Add training files if available
        if hasattr(bundle, 'training_files') and bundle.training_files:
            config["training_files"] = {k: str(v) for k, v in bundle.training_files.items()}

        # Add metadata fields
        if hasattr(bundle, 'metadata') and bundle.metadata:
            config["num_categories"] = len(bundle.metadata.get('categories', []))
            config["categories"] = bundle.metadata.get('categories', [])
            config["category_distribution"] = bundle.metadata.get('category_distribution', {})
            config["source_file"] = bundle.metadata.get('source_file')
            config["annotation_column"] = bundle.metadata.get('annotation_column')
            config["training_approach"] = bundle.metadata.get('training_approach')
            config["original_strategy"] = bundle.metadata.get('original_strategy')

            # CRITICAL FIX: Save hybrid/custom training configuration
            # These fields are REQUIRED for session relaunch to work with hybrid training
            config["multiclass_keys"] = bundle.metadata.get('multiclass_keys', [])
            config["onevsall_keys"] = bundle.metadata.get('onevsall_keys', [])
            config["key_strategies"] = bundle.metadata.get('key_strategies', {})
            config["files_per_key"] = bundle.metadata.get('files_per_key', {})

        return config

    def _extract_language_config(self, bundle) -> Dict[str, Any]:
        """Extract comprehensive language configuration."""
        config = {
            "confirmed_languages": [],
            "language_distribution": {},
            "model_strategy": "multilingual",
            "language_model_mapping": {},
            "per_language_training": False,
            "language_detection_method": None
        }

        if hasattr(bundle, 'metadata') and bundle.metadata:
            meta = bundle.metadata
            config["confirmed_languages"] = list(meta.get('confirmed_languages', []))
            config["language_distribution"] = meta.get('language_distribution', {})
            config["model_strategy"] = meta.get('model_strategy', 'multilingual')
            config["language_model_mapping"] = meta.get('language_model_mapping', {})

            # Check for per-language training
            if meta.get('models_by_language'):
                config["per_language_training"] = True
                config["models_by_language"] = meta.get('models_by_language', {})

        return config

    def _extract_text_analysis(self, bundle) -> Dict[str, Any]:
        """Extract comprehensive text analysis statistics."""
        config = {
            "text_length_stats": {},
            "requires_long_document_model": False,
            "avg_token_length": 0,
            "max_token_length": 0,
            "token_strategy": None
        }

        if hasattr(bundle, 'metadata') and bundle.metadata:
            stats = bundle.metadata.get('text_length_stats', {})
            if stats:
                config["text_length_stats"] = stats
                config["requires_long_document_model"] = bundle.metadata.get('requires_long_document_model', False)
                config["avg_token_length"] = stats.get('token_mean', stats.get('avg_tokens', 0))
                config["max_token_length"] = stats.get('token_max', stats.get('max_tokens', 0))

                # Token strategy from user choices
                config["user_prefers_long_models"] = stats.get('user_prefers_long_models', False)
                config["exclude_long_texts"] = stats.get('exclude_long_texts', False)
                config["split_long_texts"] = stats.get('split_long_texts', False)

        return config

    def _extract_split_config(self, bundle) -> Dict[str, Any]:
        """Extract train/validation/test split configuration."""
        config = {
            "mode": "uniform",
            "use_test_set": False,
            "train_ratio": 0.7,
            "validation_ratio": 0.2,
            "test_ratio": 0.1,
            "stratified": True,
            "random_seed": 42
        }

        if hasattr(bundle, 'metadata') and bundle.metadata:
            split_cfg = bundle.metadata.get('split_config', {})
            if split_cfg:
                config.update(split_cfg)

        return config

    def _extract_label_config(self, bundle) -> Dict[str, Any]:
        """Extract label configuration and mappings."""
        config = {
            "label_type": "single",
            "num_labels": 0,
            "label_names": [],
            "label_mapping": {},
            "label_distribution": {},
            "imbalanced_labels": [],
            "minority_threshold": 0.1
        }

        if hasattr(bundle, 'metadata') and bundle.metadata:
            meta = bundle.metadata
            config["label_type"] = "multi" if meta.get('strategy') == 'multi-label' else "single"
            config["label_names"] = meta.get('categories', [])
            config["num_labels"] = len(config["label_names"])
            config["label_distribution"] = meta.get('category_distribution', {})

            # Identify imbalanced labels
            if config["label_distribution"]:
                total = sum(config["label_distribution"].values())
                if total > 0:
                    for label, count in config["label_distribution"].items():
                        if count / total < config["minority_threshold"]:
                            config["imbalanced_labels"].append(label)

        return config

    def _build_complete_model_config(
        self,
        model_config: Dict[str, Any],
        quick_params: Optional[Dict[str, Any]],
        runtime_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build complete model configuration from all sources."""

        # Start with base model config
        config = dict(model_config)

        # Merge quick params if available
        if quick_params:
            config.update({
                "quick_model_name": quick_params.get('model_name'),
                "quick_epochs": quick_params.get('epochs'),
                "models_by_language": quick_params.get('models_by_language', {}),
                "train_by_language": bool(quick_params.get('models_by_language'))
            })

        # Merge runtime params if available
        if runtime_params:
            config.update(runtime_params)

        # Ensure all model selection fields are present
        config.setdefault("selected_model", None)
        config.setdefault("selected_models", [])
        config.setdefault("benchmark_models", [])
        config.setdefault("models_by_language", {})

        return config

    def _extract_training_params(
        self,
        model_config: Dict[str, Any],
        quick_params: Optional[Dict[str, Any]],
        runtime_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract all training hyperparameters."""

        # Default training parameters
        params = {
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            "fp16_opt_level": "O1",
            "optimizer": "adamw",
            "scheduler": "linear",
            "early_stopping": True,
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.001,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "save_strategy": "epoch",
            "save_steps": 500,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "evaluation_strategy": "epoch",
            "eval_steps": 500,
            "logging_steps": 50,
            "logging_first_step": True,
            "num_workers": 4,
            "seed": 42,
            "max_sequence_length": 512
        }

        # Update from model config
        for key in params:
            if key in model_config:
                params[key] = model_config[key]

        # Update from quick params
        if quick_params:
            params["epochs"] = quick_params.get('epochs', params["epochs"])
            params["batch_size"] = quick_params.get('batch_size', params["batch_size"])
            params["learning_rate"] = quick_params.get('learning_rate', params["learning_rate"])

        # Update from runtime params
        if runtime_params:
            for key in params:
                if key in runtime_params:
                    params[key] = runtime_params[key]

        return params

    def _extract_rl_config(
        self,
        model_config: Dict[str, Any],
        quick_params: Optional[Dict[str, Any]],
        runtime_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract reinforced learning configuration."""

        config = {
            "enabled": False,
            "f1_threshold": 0.70,
            "oversample_factor": 2.0,
            "class_weight_factor": 2.0,
            "reinforced_epochs": None,
            "manual_epochs": None,
            "auto_calculate_epochs": True,
            "min_reinforced_epochs": 5,
            "max_reinforced_epochs": 20,
            "reinforced_batch_size": None,
            "reinforced_learning_rate": None,
            "use_class_weights": True,
            "use_oversampling": True
        }

        # Check if RL is enabled
        if model_config.get('use_reinforcement') or (quick_params and quick_params.get('reinforced_learning')):
            config["enabled"] = True

        # Update from model config
        config["reinforced_epochs"] = model_config.get('reinforced_epochs', config["reinforced_epochs"])

        # Update from quick params
        if quick_params:
            config["enabled"] = quick_params.get('reinforced_learning', config["enabled"])
            config["f1_threshold"] = quick_params.get('rl_f1_threshold', config["f1_threshold"])
            config["oversample_factor"] = quick_params.get('rl_oversample_factor', config["oversample_factor"])
            config["class_weight_factor"] = quick_params.get('rl_class_weight_factor', config["class_weight_factor"])

            # Manual epochs override
            if quick_params.get('manual_rl_epochs'):
                config["manual_epochs"] = quick_params['manual_rl_epochs']
                config["reinforced_epochs"] = quick_params['manual_rl_epochs']
                config["auto_calculate_epochs"] = False

        # Update from runtime params
        if runtime_params:
            for key in ['rl_f1_threshold', 'rl_oversample_factor', 'rl_class_weight_factor', 'reinforced_epochs']:
                if key in runtime_params:
                    clean_key = key.replace('rl_', '')
                    if clean_key in config:
                        config[clean_key] = runtime_params[key]

        return config

    def _default_execution_status(self) -> Dict[str, Any]:
        """Create default execution status."""
        return {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "last_checkpoint": None,
            "models_trained": [],
            "models_in_progress": [],
            "models_remaining": [],
            "current_model": None,
            "current_epoch": 0,
            "total_epochs_completed": 0,
            "best_model": None,
            "best_f1": None,
            "best_accuracy": None,
            "training_time_seconds": 0,
            "errors": [],
            "warnings": []
        }

    def _extract_output_paths(self, bundle) -> Dict[str, Any]:
        """Extract all output paths."""
        from pathlib import Path

        paths = {
            "session_dir": str(self.base_dir / self.session_id),
            "models_dir": str(Path("models") / self.session_id),
            "logs_dir": str(self.base_dir / self.session_id),
            "metadata_dir": str(self.metadata_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "training_data_dir": str(self.base_dir / self.session_id / "training_data"),
            "results_csv": None,
            "benchmark_results": None
        }

        # Add training file paths if available
        if hasattr(bundle, 'training_files') and bundle.training_files:
            paths["training_files"] = {k: str(v) for k, v in bundle.training_files.items()}

        return paths

    def _extract_preprocessing(self, bundle) -> Dict[str, Any]:
        """Extract preprocessing configuration."""
        config = {
            "lowercase": False,
            "remove_punctuation": False,
            "remove_stopwords": False,
            "lemmatize": False,
            "max_sequence_length": 512,
            "truncation_strategy": "longest_first",
            "padding_strategy": "max_length",
            "tokenizer_config": {}
        }

        # Update from bundle metadata if available
        if hasattr(bundle, 'metadata') and bundle.metadata:
            preprocessing = bundle.metadata.get('preprocessing', {})
            config.update(preprocessing)

        return config

    def _extract_advanced_settings(
        self,
        model_config: Dict[str, Any],
        quick_params: Optional[Dict[str, Any]],
        bundle: Any
    ) -> Dict[str, Any]:
        """Extract advanced settings and configurations."""

        settings = {
            "distributed_training": False,
            "num_gpus": 0,
            "use_cpu": True,
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "benchmark_mode": False,
            "benchmark_categories": [],
            "one_vs_all": False,
            "multi_label": False,
            "class_imbalance_strategy": None,
            "data_augmentation": False,
            "augmentation_factor": 1.0,
            "custom_metrics": [],
            "tensorboard_logging": False,
            "wandb_logging": False,
            "debug_mode": False,
            "verbose": True
        }

        # Check for benchmark mode
        if model_config.get('selected_models') or model_config.get('benchmark_category'):
            settings["benchmark_mode"] = True
            settings["benchmark_categories"] = model_config.get('selected_labels', [])

        # Check for one-vs-all
        if hasattr(bundle, 'metadata') and bundle.metadata:
            if bundle.metadata.get('training_approach') == 'one-vs-all':
                settings["one_vs_all"] = True
            if bundle.metadata.get('strategy') == 'multi-label':
                settings["multi_label"] = True

        return settings

    def _extract_checkpoint_info(self) -> Dict[str, Any]:
        """Extract checkpoint information."""
        checkpoints = {}

        if self.checkpoint_dir.exists():
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                checkpoint_name = checkpoint_file.stem
                checkpoints[checkpoint_name] = {
                    "path": str(checkpoint_file),
                    "size": checkpoint_file.stat().st_size,
                    "modified": datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()
                }

        return checkpoints

    def _save_with_backup(self, metadata: Dict[str, Any]) -> None:
        """Save metadata with automatic backup."""

        # Create backup of existing metadata if it exists
        if self.metadata_path.exists():
            import shutil
            shutil.copy2(self.metadata_path, self.backup_path)

        # Save new metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Metadata saved to {self.metadata_path}")

    def _load_backup(self) -> Optional[Dict[str, Any]]:
        """Load backup metadata if available."""
        if self.backup_path.exists():
            try:
                with open(self.backup_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")
        return None

    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata structure and required fields."""

        # Check required top-level sections
        required_sections = [
            "training_session", "dataset_config", "model_config",
            "execution_status", "output_paths"
        ]

        for section in required_sections:
            if section not in metadata:
                logger.warning(f"Missing required section: {section}")
                return False

        # Check session has session_id
        if not metadata.get("training_session", {}).get("session_id"):
            logger.warning("Missing session_id in training_session")
            return False

        # Check dataset has primary file or training files
        dataset = metadata.get("dataset_config", {})
        if not dataset.get("primary_file") and not dataset.get("training_files"):
            logger.warning("Missing dataset files")
            return False

        return True

    def _get_tool_version(self) -> str:
        """Get tool version from package or default."""
        try:
            import pkg_resources
            return pkg_resources.get_distribution("llm-tool").version
        except:
            return "LLMTool v1.0"

    def _get_safe_username(self) -> str:
        """Get username safely without raising exceptions."""
        try:
            import os
            return os.getenv('USER', os.getenv('USERNAME', 'unknown'))
        except:
            return 'unknown'

    def generate_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Generate a hash of metadata for integrity checking."""
        # Convert to JSON string for consistent hashing
        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha256(metadata_str.encode()).hexdigest()
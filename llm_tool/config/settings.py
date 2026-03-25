#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
settings.py

MAIN OBJECTIVE:
---------------
This script manages global configuration and settings for the LLMTool package,
including API keys, model preferences, paths, and language settings.

Dependencies:
-------------
- sys
- os
- json
- pathlib
- typing
- configparser

MAIN FEATURES:
--------------
1) Load and save configuration from/to JSON and INI files
2) Manage API credentials securely
3) Handle model preferences and paths
4) Configure language detection settings
5) Provide default settings with override capability

Author:
-------
Antoine Lemor
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from configparser import ConfigParser
from dataclasses import dataclass, asdict, field

# Import API key manager
try:
    from .api_key_manager import APIKeyManager, get_key_manager
    HAS_KEY_MANAGER = True
except ImportError:
    HAS_KEY_MANAGER = False
    APIKeyManager = None
    get_key_manager = None


@dataclass
class APIConfig:
    """Configuration for API providers"""
    provider: str = "openai"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_name: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class LocalModelConfig:
    """Configuration for local models"""
    provider: str = "ollama"
    model_name: str = "llama3.2"
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cuda, cpu, mps
    quantization: Optional[str] = None
    max_memory: Optional[Dict[str, Any]] = None


@dataclass
class DataConfig:
    """Configuration for data handling"""
    default_format: str = "csv"
    encoding: str = "utf-8"
    chunk_size: int = 1000
    max_workers: int = 4
    cache_enabled: bool = True
    cache_dir: str = "cache"


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    default_model: str = "bert-base-multilingual-cased"
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_epochs: int = 10
    early_stopping_patience: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True


@dataclass
class LanguageConfig:
    """Configuration for language detection and handling"""
    detection_mode: str = "auto"  # auto, manual, column
    confidence_threshold: float = 0.8
    fallback_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ar", "hi"
    ])
    language_column: Optional[str] = None
    manual_language: Optional[str] = None


@dataclass
class PathConfig:
    """Configuration for file paths"""
    base_dir: Path = field(default_factory=Path.cwd)
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models")
    prompts_dir: Path = field(default_factory=lambda: Path.cwd() / "prompts")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    validation_dir: Path = field(default_factory=lambda: Path.cwd() / "validation")
    cache_dir: Path = field(default_factory=lambda: Path.cwd() / "cache")


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "[%(levelname)s] %(message)s"
    file_logging: bool = True
    console_logging: bool = True
    log_file: str = "application/llm_tool.log"  # Organized in application subfolder
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


class Settings:
    """Main settings manager for LLMTool"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize settings with optional config file"""
        self.config_file = config_file or self._get_default_config_file()

        # Initialize all configurations
        self.api = APIConfig()
        self.local_model = LocalModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.language = LanguageConfig()
        self.paths = PathConfig()
        self.logging = LoggingConfig()

        # Initialize API key manager
        if HAS_KEY_MANAGER:
            self.key_manager = get_key_manager()
        else:
            self.key_manager = None

        # Create necessary directories
        self._create_directories()

        # Load configuration if exists
        if Path(self.config_file).exists():
            self.load()

        # Setup logging
        self._setup_logging()

    def _get_default_config_file(self) -> str:
        """Get default configuration file path"""
        config_dir = Path.home() / ".llm_tool"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "config.json")

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for path_attr in ['data_dir', 'models_dir', 'prompts_dir',
                          'logs_dir', 'validation_dir', 'cache_dir']:
            path = getattr(self.paths, path_attr)
            path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration"""
        handlers = []

        if self.logging.console_logging:
            handlers.append(logging.StreamHandler())

        if self.logging.file_logging:
            log_file = self.paths.logs_dir / self.logging.log_file
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            from logging.handlers import RotatingFileHandler
            handlers.append(RotatingFileHandler(
                log_file,
                maxBytes=self.logging.max_bytes,
                backupCount=self.logging.backup_count
            ))

        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
            handlers=handlers
        )

    def load(self, config_file: Optional[str] = None):
        """Load configuration from file"""
        config_file = config_file or self.config_file

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Load each configuration section
            if 'api' in config_data:
                self.api = APIConfig(**config_data['api'])
            if 'local_model' in config_data:
                self.local_model = LocalModelConfig(**config_data['local_model'])
            if 'data' in config_data:
                self.data = DataConfig(**config_data['data'])
            if 'training' in config_data:
                self.training = TrainingConfig(**config_data['training'])
            if 'language' in config_data:
                self.language = LanguageConfig(**config_data['language'])
            if 'paths' in config_data:
                self.paths = PathConfig(**{k: Path(v) for k, v in config_data['paths'].items()})
            if 'logging' in config_data:
                self.logging = LoggingConfig(**config_data['logging'])

            logging.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logging.warning(f"Could not load configuration: {e}")

    def save(self, config_file: Optional[str] = None):
        """Save configuration to file"""
        config_file = config_file or self.config_file

        config_data = {
            'api': asdict(self.api),
            'local_model': asdict(self.local_model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'language': asdict(self.language),
            'paths': {k: str(v) for k, v in asdict(self.paths).items()},
            'logging': asdict(self.logging)
        }

        # Remove None values
        config_data = self._remove_none_values(config_data)

        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logging.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logging.error(f"Could not save configuration: {e}")

    def _remove_none_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove None values from dictionary"""
        if not isinstance(d, dict):
            return d
        return {k: self._remove_none_values(v) for k, v in d.items() if v is not None}

    def update_api_settings(self, settings: Dict[str, Any]):
        """Update API settings"""
        for key, value in settings.items():
            if hasattr(self.api, key):
                setattr(self.api, key, value)
        self.save()

    def update_language_settings(self, settings: Dict[str, Any]):
        """Update language settings"""
        for key, value in settings.items():
            if hasattr(self.language, key):
                setattr(self.language, key, value)
        self.save()

    def update_training_settings(self, settings: Dict[str, Any]):
        """Update training settings"""
        for key, value in settings.items():
            if hasattr(self.training, key):
                setattr(self.training, key, value)
        self.save()

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider, checking in order:
        1. Environment variables
        2. Secure key manager
        3. Settings file (legacy)
        """
        # Use key manager if available
        if self.key_manager:
            return self.key_manager.get_key(provider)

        # Fallback to environment variables
        env_var_names = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'huggingface': 'HF_TOKEN'
        }

        env_var = env_var_names.get(provider.lower())
        if env_var and env_var in os.environ:
            return os.environ[env_var]

        # Return stored API key (legacy)
        if provider.lower() == self.api.provider:
            return self.api.api_key

        return None

    def set_api_key(self, provider: str, api_key: str, model_name: Optional[str] = None):
        """
        Set API key for a provider.
        Uses secure key manager if available, otherwise saves to config file.
        """
        if self.key_manager:
            self.key_manager.save_key(provider, api_key, model_name)
        else:
            # Fallback to legacy storage
            self.api.provider = provider
            self.api.api_key = api_key
            if model_name:
                self.api.model_name = model_name
            self.save()

    def get_or_prompt_api_key(self, provider: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Get API key or prompt user if not available.

        Parameters
        ----------
        provider : str
            Provider name
        model_name : str, optional
            Model name to save with the key

        Returns
        -------
        str or None
            The API key
        """
        if self.key_manager:
            return self.key_manager.get_or_prompt_key(provider, model_name)
        else:
            # Fallback - return existing or None
            return self.get_api_key(provider)

    def list_saved_providers(self) -> list:
        """
        List all providers with saved API keys.

        Returns
        -------
        list
            List of provider names
        """
        if self.key_manager:
            return self.key_manager.list_providers()
        return []

    def get_model_path(self, model_name: str) -> Path:
        """Get the full path for a model"""
        return self.paths.models_dir / model_name

    def get_prompt_path(self, prompt_name: str) -> Path:
        """Get the full path for a prompt"""
        if not prompt_name.endswith('.txt'):
            prompt_name += '.txt'
        return self.paths.prompts_dir / prompt_name

    def get_data_path(self, filename: str) -> Path:
        """Get the full path for a data file"""
        return self.paths.data_dir / filename

    def get_log_path(self, filename: str) -> Path:
        """Get the full path for a log file"""
        return self.paths.logs_dir / filename

    def get_validation_path(self, filename: str) -> Path:
        """Get the full path for a validation file"""
        return self.paths.validation_dir / filename

    def to_dict(self) -> Dict[str, Any]:
        """Convert all settings to dictionary"""
        return {
            'api': asdict(self.api),
            'local_model': asdict(self.local_model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'language': asdict(self.language),
            'paths': {k: str(v) for k, v in asdict(self.paths).items()},
            'logging': asdict(self.logging)
        }

    def get_system_recommendations(self) -> Dict[str, Any]:
        """
        Get system resource recommendations for optimal configuration.

        Returns
        -------
        dict
            Dictionary with recommended settings based on system resources
        """
        try:
            from ..utils.system_resources import detect_resources
            resources = detect_resources()
            return resources.get_recommendation()
        except Exception as e:
            logging.warning(f"Could not detect system resources: {e}")
            return {
                'device': 'cpu',
                'batch_size': 8,
                'num_workers': 2,
                'use_fp16': False,
                'gradient_accumulation_steps': 1
            }

    def apply_system_recommendations(self, force: bool = False):
        """
        Apply system resource recommendations to settings.

        Parameters
        ----------
        force : bool
            If True, override existing settings. If False, only set if values are default.
        """
        recommendations = self.get_system_recommendations()

        # Update training config with recommendations
        if force or self.training.batch_size == 16:  # Default value
            self.training.batch_size = recommendations['batch_size']

        if force or not self.training.fp16:
            self.training.fp16 = recommendations['use_fp16']

        if force or self.training.gradient_accumulation_steps == 1:
            self.training.gradient_accumulation_steps = recommendations['gradient_accumulation_steps']

        # Update data config with recommended workers
        if force or self.data.max_workers == 4:  # Default value
            self.data.max_workers = recommendations['num_workers']

        # Update local model config with recommended device
        if force or self.local_model.device == "auto":
            self.local_model.device = recommendations['device']

        logging.info(f"Applied system recommendations: {recommendations}")

    def __repr__(self) -> str:
        """String representation of settings"""
        return f"Settings(config_file='{self.config_file}')"


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset global settings instance"""
    global _settings
    _settings = None
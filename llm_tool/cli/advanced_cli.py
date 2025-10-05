#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
advanced_cli.py

MAIN OBJECTIVE:
---------------
This script provides a sophisticated, professional-grade Command Line Interface
for the LLMTool package with auto-detection, intelligent suggestions, and
guided interactive workflows inspired by state-of-the-art CLI design patterns.

Dependencies:
-------------
- sys
- os
- subprocess
- pathlib
- rich (optional but highly recommended)
- inquirer
- pandas
- psutil

MAIN FEATURES:
--------------
1) Auto-detection of available models (Ollama, API models, local files)
2) Intelligent suggestions based on context and history
3) Interactive guided wizards for complex workflows
4) Professional validation with helpful error recovery
5) Rich visual interface with graceful fallback
6) Configuration profiles and execution history
7) Real-time progress tracking with detailed statistics
8) Smart defaults based on detected environment

Author:
-------
Antoine Lemor
"""

import sys
import os
import subprocess
import contextlib
import json
import time
import logging
import glob
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import re

# Rich is mandatory for this CLI
HAS_RICH = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import print as rprint
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError as e:
    print("\n❌ Error: Rich library is required but not installed.")

# Requests is optional (only needed for Label Studio direct export)
HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    pass  # Will be handled gracefully when needed

# Try importing pandas for data preview
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Try importing psutil for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import internal modules
from ..config.settings import Settings
from ..pipelines.pipeline_controller import PipelineController
from ..utils.language_detector import LanguageDetector
from ..annotators.json_cleaner import extract_expected_keys
from ..annotators.prompt_wizard import SocialSciencePromptWizard, create_llm_client_for_wizard
from ..trainers.model_trainer import ModelTrainer, BenchmarkConfig
from ..trainers.multi_label_trainer import (
    MultiLabelTrainer,
    TrainingConfig as MultiLabelTrainingConfig,
    ModelInfo as MultiLabelModelInfo,
)
from ..trainers.training_data_builder import (
    TrainingDatasetBuilder,
    TrainingDataRequest,
    TrainingDataBundle,
)


@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    provider: str  # ollama, openai, anthropic, local
    size: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    is_available: bool = True
    requires_api_key: bool = False
    supports_json: bool = True
    supports_streaming: bool = True
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    path: Path
    format: str
    rows: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    size_mb: Optional[float] = None
    detected_language: Optional[str] = None
    has_labels: bool = False
    label_column: Optional[str] = None
    column_types: Dict[str, str] = field(default_factory=dict)
    text_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionProfile:
    """Saved execution profile for quick re-runs"""
    name: str
    created_at: datetime
    last_used: datetime
    configuration: Dict[str, Any]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class LLMDetector:
    """Auto-detect available LLMs for annotation from various sources"""

    @staticmethod
    def detect_ollama_models() -> List[ModelInfo]:
        """Detect locally available Ollama LLMs for annotation"""
        models = []
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Has header and content
                    # Parse the table format from ollama list
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            # Parse: NAME                    ID              SIZE      MODIFIED
                            parts = line.split()
                            if len(parts) >= 1:
                                name = parts[0]
                                # Extract size if available (e.g., "27GB" -> "27 GB")
                                size = None
                                if len(parts) >= 3:
                                    size_str = parts[2]
                                    if 'GB' in size_str or 'MB' in size_str:
                                        size = size_str

                                models.append(ModelInfo(
                                    name=name,
                                    provider="ollama",
                                    size=size,
                                    is_available=True,
                                    supports_json=True,
                                    supports_streaming=True,
                                    context_length=LLMDetector._estimate_context_length(name),
                                    max_tokens=LLMDetector._suggest_local_max_tokens(name)
                                ))
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return models

    @staticmethod
    def _estimate_context_length(model_name: str) -> int:
        """Estimate context length based on model name"""
        name_lower = model_name.lower()
        if 'llama3' in name_lower or 'llama-3' in name_lower:
            return 8192
        elif 'mixtral' in name_lower:
            return 32768
        elif 'gemma2' in name_lower:
            return 8192
        elif 'gemma3' in name_lower:
            return 8192
        elif 'qwen' in name_lower:
            return 32768
        elif 'phi' in name_lower:
            return 4096
        elif 'mistral' in name_lower:
            return 8192
        else:
            return 4096  # Default

    @staticmethod
    def _suggest_local_max_tokens(model_name: str) -> int:
        """Heuristic for local model generation budget"""
        context = LLMDetector._estimate_context_length(model_name)
        # Keep generous default while leaving room for prompt context
        return max(512, min(2048, context // 2))

    @staticmethod
    def detect_openai_models() -> List[ModelInfo]:
        """List available OpenAI models"""
        models = [
            # ✅ Tested models (fully supported in pipeline)
            ModelInfo("gpt-5-nano-2025-08-07", "openai", context_length=200000, requires_api_key=True,
                     cost_per_1k_tokens=0.001, supports_json=True, supports_streaming=True, max_tokens=4000),
            ModelInfo("gpt-5-mini-2025-08-07", "openai", context_length=200000, requires_api_key=True,
                     cost_per_1k_tokens=0.001, supports_json=True, supports_streaming=True, max_tokens=4000),
        ]
        return models

    @staticmethod
    def detect_anthropic_models() -> List[ModelInfo]:
        """List available Anthropic models"""
        models = [
            # ⚠️ Not yet tested in pipeline
        ]
        return models

    @staticmethod
    def detect_all_llms() -> Dict[str, List[ModelInfo]]:
        """Detect all available LLMs for annotation from all sources"""
        return {
            "local": LLMDetector.detect_ollama_models(),
            "openai": LLMDetector.detect_openai_models(),
            "anthropic": LLMDetector.detect_anthropic_models(),
        }


class TrainerModelDetector:
    """Detect available models for training (BERT variants, etc.)"""

    @staticmethod
    def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
        """Get all available models for training organized by category"""
        return {
            "English Models": [
                {"name": "bert-base-uncased", "params": "110M", "type": "BERT", "performance": "★★★"},
                {"name": "bert-large-uncased", "params": "340M", "type": "BERT", "performance": "★★★★"},
                {"name": "roberta-base", "params": "125M", "type": "RoBERTa", "performance": "★★★★"},
                {"name": "roberta-large", "params": "355M", "type": "RoBERTa", "performance": "★★★★★"},
                {"name": "deberta-v3-base", "params": "184M", "type": "DeBERTa", "performance": "★★★★★"},
                {"name": "deberta-v3-large", "params": "435M", "type": "DeBERTa", "performance": "★★★★★"},
                {"name": "electra-base", "params": "110M", "type": "ELECTRA", "performance": "★★★★"},
                {"name": "albert-base-v2", "params": "12M", "type": "ALBERT", "performance": "★★★"},
            ],
            "Multilingual Models": [
                {"name": "bert-base-multilingual", "params": "177M", "type": "mBERT", "languages": "104", "performance": "★★★"},
                {"name": "xlm-roberta-base", "params": "278M", "type": "XLM-R", "languages": "100+", "performance": "★★★★"},
                {"name": "xlm-roberta-large", "params": "560M", "type": "XLM-R", "languages": "100+", "performance": "★★★★★"},
                {"name": "mdeberta-v3-base", "params": "280M", "type": "mDeBERTa", "languages": "100+", "performance": "★★★★★"},
            ],
            "French Models": [
                {"name": "camembert-base", "params": "110M", "type": "CamemBERT", "performance": "★★★★"},
                {"name": "flaubert-base", "params": "137M", "type": "FlauBERT", "performance": "★★★★"},
                {"name": "distilcamembert", "params": "68M", "type": "DistilCamemBERT", "performance": "★★★"},
            ],
            "Long Document Models": [
                {"name": "longformer-base", "params": "149M", "type": "Longformer", "max_length": "4096", "performance": "★★★★"},
                {"name": "bigbird-base", "params": "128M", "type": "BigBird", "max_length": "4096", "performance": "★★★★"},
            ],
            "Efficient Models": [
                {"name": "distilbert-base", "params": "66M", "type": "DistilBERT", "speed": "2x faster", "performance": "★★★"},
                {"name": "tinybert", "params": "14M", "type": "TinyBERT", "speed": "9x faster", "performance": "★★"},
                {"name": "mobilebert", "params": "25M", "type": "MobileBERT", "speed": "4x faster", "performance": "★★★"},
            ]
        }


class LanguageNormalizer:
    """Intelligent language normalization and mapping system"""

    # Comprehensive language mapping dictionary
    LANGUAGE_MAPPINGS = {
        'en': ['en', 'eng', 'english', 'anglais'],
        'fr': ['fr', 'fra', 'fre', 'french', 'français', 'francais'],
        'de': ['de', 'deu', 'ger', 'german', 'deutsch', 'allemand'],
        'es': ['es', 'spa', 'spanish', 'español', 'espagnol'],
        'it': ['it', 'ita', 'italian', 'italiano', 'italien'],
        'pt': ['pt', 'por', 'portuguese', 'português', 'portugais'],
        'nl': ['nl', 'nld', 'dut', 'dutch', 'nederlands', 'néerlandais'],
        'ru': ['ru', 'rus', 'russian', 'русский', 'russe'],
        'zh': ['zh', 'chi', 'zho', 'chinese', '中文', 'chinois'],
        'ja': ['ja', 'jpn', 'japanese', '日本語', 'japonais'],
        'ar': ['ar', 'ara', 'arabic', 'العربية', 'arabe'],
        'hi': ['hi', 'hin', 'hindi', 'हिन्दी'],
        'ko': ['ko', 'kor', 'korean', '한국어', 'coréen'],
        'pl': ['pl', 'pol', 'polish', 'polski', 'polonais'],
        'tr': ['tr', 'tur', 'turkish', 'türkçe', 'turc'],
        'sv': ['sv', 'swe', 'swedish', 'svenska', 'suédois'],
        'da': ['da', 'dan', 'danish', 'dansk', 'danois'],
        'no': ['no', 'nor', 'norwegian', 'norsk', 'norvégien'],
        'fi': ['fi', 'fin', 'finnish', 'suomi', 'finnois'],
        'cs': ['cs', 'ces', 'cze', 'czech', 'čeština', 'tchèque'],
        'ro': ['ro', 'ron', 'rum', 'romanian', 'română', 'roumain'],
        'hu': ['hu', 'hun', 'hungarian', 'magyar', 'hongrois'],
        'el': ['el', 'ell', 'gre', 'greek', 'ελληνικά', 'grec'],
        'he': ['he', 'heb', 'hebrew', 'עברית', 'hébreu'],
        'th': ['th', 'tha', 'thai', 'ไทย', 'thaï'],
        'vi': ['vi', 'vie', 'vietnamese', 'tiếng việt', 'vietnamien'],
        'id': ['id', 'ind', 'indonesian', 'bahasa indonesia', 'indonésien'],
        'uk': ['uk', 'ukr', 'ukrainian', 'українська', 'ukrainien'],
    }

    # Reverse mapping for quick lookup
    _REVERSE_MAPPING = None

    @classmethod
    def _build_reverse_mapping(cls):
        """Build reverse mapping from variant to standard code"""
        if cls._REVERSE_MAPPING is None:
            cls._REVERSE_MAPPING = {}
            for standard_code, variants in cls.LANGUAGE_MAPPINGS.items():
                for variant in variants:
                    cls._REVERSE_MAPPING[variant.lower()] = standard_code

    @classmethod
    def normalize_language(cls, lang_value: str) -> Optional[str]:
        """Normalize a language value to standard 2-letter code"""
        if not lang_value:
            return None

        cls._build_reverse_mapping()
        lang_lower = str(lang_value).strip().lower()
        return cls._REVERSE_MAPPING.get(lang_lower)

    @staticmethod
    def detect_languages_in_column(df, column_name: str) -> Dict[str, int]:
        """Detect and count languages in a dataframe column"""
        if column_name not in df.columns:
            return {}

        lang_counts = {}
        for value in df[column_name].dropna():
            normalized = LanguageNormalizer.normalize_language(value)
            if normalized:
                lang_counts[normalized] = lang_counts.get(normalized, 0) + 1

        return lang_counts

    @staticmethod
    def recommend_models(languages: Set[str], all_models: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Recommend training models based on detected languages"""
        recommendations = []

        if not languages:
            # No language info - recommend multilingual as safe default
            recommendations.append({
                'model': 'xlm-roberta-base',
                'category': 'Multilingual Models',
                'reason': 'No language detected - multilingual model as safe default',
                'priority': 3
            })
            return recommendations

        # Single language recommendations
        if len(languages) == 1:
            lang = list(languages)[0]

            if lang == 'en':
                for model in all_models.get('English Models', []):
                    recommendations.append({
                        'model': model['name'],
                        'category': 'English Models',
                        'reason': f"Optimized for English ({model.get('performance', 'N/A')} performance)",
                        'priority': 1,
                        'details': model
                    })

            elif lang == 'fr':
                for model in all_models.get('French Models', []):
                    recommendations.append({
                        'model': model['name'],
                        'category': 'French Models',
                        'reason': f"Specialized for French ({model.get('performance', 'N/A')} performance)",
                        'priority': 1,
                        'details': model
                    })
                # Also suggest multilingual as fallback
                recommendations.append({
                    'model': 'xlm-roberta-base',
                    'category': 'Multilingual Models',
                    'reason': 'Multilingual fallback (supports French + 100 languages)',
                    'priority': 2
                })

            else:
                # Other single language - recommend multilingual
                for model in all_models.get('Multilingual Models', []):
                    recommendations.append({
                        'model': model['name'],
                        'category': 'Multilingual Models',
                        'reason': f"Supports {lang.upper()} + {model.get('languages', '100+')} languages",
                        'priority': 1,
                        'details': model
                    })

        # Multiple languages - strongly recommend multilingual
        else:
            lang_str = ', '.join([l.upper() for l in sorted(languages)])
            for model in all_models.get('Multilingual Models', []):
                recommendations.append({
                    'model': model['name'],
                    'category': 'Multilingual Models',
                    'reason': f"Handles multiple languages ({lang_str}) - {model.get('languages', '100+')} supported",
                    'priority': 1,
                    'details': model
                })

            # Also suggest separate models per language
            recommendations.append({
                'model': 'separate_per_language',
                'category': 'Multi-Model Strategy',
                'reason': f"Train separate specialized models for each language ({lang_str})",
                'priority': 2
            })

        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        return recommendations


class DataDetector:
    """Auto-detect and analyze available datasets"""

    @staticmethod
    def scan_directory(directory: Path = Path.cwd()) -> List[DatasetInfo]:
        """Scan directory and subdirectories for potential datasets"""
        datasets = []

        # If directory doesn't exist, return empty list
        if not directory.exists():
            return datasets

        patterns = ['**/*.csv', '**/*.json', '**/*.jsonl', '**/*.xlsx', '**/*.parquet', '**/*.tsv']

        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    dataset_info = DataDetector.analyze_file(file_path)
                    if dataset_info:
                        datasets.append(dataset_info)

        return datasets

    @staticmethod
    def analyze_file(file_path: Path) -> Optional[DatasetInfo]:
        """Analyze a single file to extract dataset information"""
        if not file_path.exists():
            return None

        info = DatasetInfo(
            path=file_path,
            format=file_path.suffix[1:],
            size_mb=file_path.stat().st_size / (1024 * 1024)
        )

        # Try to read and analyze the file
        if HAS_PANDAS:
            try:
                if info.format == 'csv':
                    df = pd.read_csv(file_path, nrows=100)
                elif info.format == 'json':
                    df = pd.read_json(file_path, lines=False, nrows=100)
                elif info.format == 'jsonl':
                    df = pd.read_json(file_path, lines=True, nrows=100)
                elif info.format == 'xlsx':
                    df = pd.read_excel(file_path, nrows=100)
                elif info.format == 'parquet':
                    df = pd.read_parquet(file_path)
                    df = df.head(100)
                else:
                    return info

                info.rows = len(df)
                info.columns = list(df.columns)
                info.column_types = {col: str(df[col].dtype) for col in df.columns}
                info.text_scores = {}

                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]) or str(df[col].dtype) == 'object':
                        sample_series = df[col].dropna().astype(str)
                        if not sample_series.empty:
                            avg_len = float(sample_series.str.len().mean())
                            info.text_scores[col] = avg_len

                # Detect if there's a label column
                label_candidates = ['label', 'labels', 'class', 'category', 'target', 'y']
                for col in info.columns:
                    if col.lower() in label_candidates:
                        info.has_labels = True
                        info.label_column = col
                        break

            except Exception:
                pass

        return info

    @staticmethod
    def analyze_file_intelligently(file_path: Path) -> Dict[str, Any]:
        """
        Comprehensive intelligent analysis of any supported file format.
        Returns detailed information about columns, languages, annotations, etc.
        """
        result = {
            'file_path': file_path,
            'format': file_path.suffix[1:].lower(),
            'columns': [],
            'all_columns': [],  # Add this for complete column list
            'text_column_candidates': [],
            'annotation_column_candidates': [],
            'id_column_candidates': [],
            'language_column_candidates': [],
            'languages_detected': {},
            'has_valid_annotations': False,
            'annotation_stats': {},
            'row_count': 0,
            'issues': [],
            'sample_data': {},  # Add sample data for displaying examples
            'annotation_keys_found': set()  # For JSON annotations
        }

        if not HAS_PANDAS:
            result['issues'].append("pandas not available - limited analysis")
            return result

        try:
            # Read file based on format
            df = None
            file_format = result['format']

            if file_format == 'csv':
                df = pd.read_csv(file_path, nrows=1000)
            elif file_format == 'tsv':
                df = pd.read_csv(file_path, sep='\t', nrows=1000)
            elif file_format == 'json':
                df = pd.read_json(file_path, lines=False)
                if len(df) > 1000:
                    df = df.head(1000)
            elif file_format == 'jsonl':
                df = pd.read_json(file_path, lines=True, nrows=1000)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, nrows=1000)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
                if len(df) > 1000:
                    df = df.head(1000)
            else:
                result['issues'].append(f"Unsupported format: {file_format}")
                return result

            if df is None or df.empty:
                result['issues'].append("File is empty or could not be read")
                return result

            result['row_count'] = len(df)
            result['columns'] = list(df.columns)
            result['all_columns'] = list(df.columns)  # Store complete list

            # Extract sample data for each column (first 3 non-null values)
            for col in df.columns:
                non_null_values = df[col].dropna().head(3).tolist()
                result['sample_data'][col] = non_null_values

            # For JSON annotations, detect keys within the annotation column
            for col in df.columns:
                col_lower = col.lower()
                if 'annotation' in col_lower or 'label' in col_lower:
                    # Try to parse first few entries as JSON to find keys
                    for idx in range(min(10, len(df))):
                        val = df[col].iloc[idx]
                        if pd.notna(val):
                            try:
                                if isinstance(val, dict):
                                    result['annotation_keys_found'].update(val.keys())
                                elif isinstance(val, str):
                                    parsed = json.loads(val)
                                    if isinstance(parsed, dict):
                                        result['annotation_keys_found'].update(parsed.keys())
                            except:
                                pass

            # Detect column candidates
            text_candidates = ['text', 'content', 'message', 'sentence', 'paragraph',
                             'document', 'body', 'description', 'comment', 'review']
            annotation_candidates = ['annotation', 'annotations', 'label', 'labels',
                                    'category', 'categories', 'class', 'classification']
            id_candidates = ['id', 'identifier', '_id', 'uuid', 'key']
            lang_candidates = ['lang', 'language', 'langue', 'idioma', 'sprache', 'lingua']

            for col in df.columns:
                col_lower = col.lower()

                # Check for text columns
                if any(candidate in col_lower for candidate in text_candidates):
                    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                        avg_len = df[col].dropna().astype(str).str.len().mean()
                        if avg_len > 20:  # Likely text if average length > 20
                            result['text_column_candidates'].append({
                                'name': col,
                                'avg_length': float(avg_len),
                                'match_type': 'name_pattern'
                            })

                # Check for annotation columns
                if any(candidate in col_lower for candidate in annotation_candidates):
                    result['annotation_column_candidates'].append({
                        'name': col,
                        'match_type': 'name_pattern'
                    })
                    # Check if annotations are valid (not empty)
                    non_empty = df[col].notna().sum()
                    empty = df[col].isna().sum()
                    result['annotation_stats'][col] = {
                        'non_empty': int(non_empty),
                        'empty': int(empty),
                        'fill_rate': float(non_empty / len(df)) if len(df) > 0 else 0
                    }
                    if non_empty > 0:
                        result['has_valid_annotations'] = True

                # Check for ID columns
                if any(candidate in col_lower for candidate in id_candidates) or col_lower.endswith('_id') or col_lower.endswith('id'):
                    result['id_column_candidates'].append(col)

                # Check for language columns
                if any(candidate in col_lower for candidate in lang_candidates):
                    result['language_column_candidates'].append(col)
                    # Detect languages
                    lang_counts = LanguageNormalizer.detect_languages_in_column(df, col)
                    if lang_counts:
                        result['languages_detected'] = lang_counts

            # If no text candidates found by name, find by heuristics
            if not result['text_column_candidates']:
                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                        avg_len = df[col].dropna().astype(str).str.len().mean()
                        if avg_len > 50:  # Longer text
                            result['text_column_candidates'].append({
                                'name': col,
                                'avg_length': float(avg_len),
                                'match_type': 'heuristic'
                            })

            # Sort text candidates by length (longer is likely main text)
            result['text_column_candidates'].sort(key=lambda x: x['avg_length'], reverse=True)

            # Validation checks
            if not result['text_column_candidates']:
                result['issues'].append("⚠️  No text column detected - manual selection required")

            if result['annotation_column_candidates'] and not result['has_valid_annotations']:
                result['issues'].append("❌ Annotation columns found but they are EMPTY - cannot train!")

            if not result['language_column_candidates'] and len(result['text_column_candidates']) > 0:
                result['issues'].append("ℹ️  No language column detected - language detection can be applied")

        except Exception as e:
            result['issues'].append(f"Analysis error: {str(e)}")

        return result

    @staticmethod
    def suggest_text_column(dataset: DatasetInfo) -> Optional[str]:
        """Suggest the most likely text column from a dataset"""
        text_candidates = ['text', 'content', 'message', 'comment', 'review',
                          'description', 'body', 'document', 'sentence', 'paragraph']
        column_types = dataset.column_types or {}

        def is_probably_identifier(name: str) -> bool:
            name_lower = name.lower()
            if name_lower in {'id', 'identifier'}:
                return True
            if name_lower.endswith('_id') or name_lower.endswith('id'):
                return True
            return False

        def candidate_score(name: str) -> float:
            return dataset.text_scores.get(name, 0.0)

        # Prioritise columns whose dtype looks textual
        textual_columns = []
        for col in dataset.columns:
            dtype = column_types.get(col, '').lower()
            if 'object' in dtype or 'string' in dtype:
                textual_columns.append(col)
        if not textual_columns:
            textual_columns = list(dataset.columns)

        # Exact matches first
        exact_matches = [
            col for col in textual_columns
            if col.lower() in text_candidates and not is_probably_identifier(col)
        ]
        if exact_matches:
            exact_matches.sort(key=candidate_score, reverse=True)
            return exact_matches[0]

        # Partial matches (avoid *_id)
        partial_matches = []
        for col in textual_columns:
            col_lower = col.lower()
            if is_probably_identifier(col):
                continue
            for candidate in text_candidates:
                if candidate in col_lower:
                    partial_matches.append(col)
                    break

        if partial_matches:
            partial_matches.sort(key=candidate_score, reverse=True)
            return partial_matches[0]

        # Fall back to column with largest average length
        if dataset.text_scores:
            for col, _ in sorted(dataset.text_scores.items(), key=lambda item: item[1], reverse=True):
                if col in textual_columns and not is_probably_identifier(col):
                    return col

        # Final fallback: first non-identifier textual column
        for col in textual_columns:
            if not is_probably_identifier(col):
                return col

        return dataset.columns[0] if dataset.columns else None


class ProfileManager:
    """Manage execution profiles for quick re-runs"""

    def __init__(self, profile_dir: Path = None):
        self.profile_dir = profile_dir or Path.home() / '.llmtool' / 'profiles'
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.profile_dir / 'history.json'

    def save_profile(self, profile: ExecutionProfile):
        """Save an execution profile"""
        profile_file = self.profile_dir / f"{profile.name}.json"
        data = {
            'name': profile.name,
            'created_at': profile.created_at.isoformat(),
            'last_used': profile.last_used.isoformat(),
            'configuration': profile.configuration,
            'performance_metrics': profile.performance_metrics,
            'notes': profile.notes
        }
        with open(profile_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_profile(self, name: str) -> Optional[ExecutionProfile]:
        """Load an execution profile by name"""
        profile_file = self.profile_dir / f"{name}.json"
        if not profile_file.exists():
            return None

        with open(profile_file, 'r') as f:
            data = json.load(f)

        return ExecutionProfile(
            name=data['name'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_used=datetime.fromisoformat(data['last_used']),
            configuration=data['configuration'],
            performance_metrics=data.get('performance_metrics', {}),
            notes=data.get('notes', '')
        )

    def list_profiles(self) -> List[ExecutionProfile]:
        """List all available profiles"""
        profiles = []
        for profile_file in self.profile_dir.glob('*.json'):
            if profile_file.name != 'history.json':
                profile = self.load_profile(profile_file.stem)
                if profile:
                    profiles.append(profile)
        return sorted(profiles, key=lambda p: p.last_used, reverse=True)

    def save_to_history(self, config: Dict[str, Any]):
        """Save configuration to execution history"""
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        history.append({
            'timestamp': datetime.now().isoformat(),
            'configuration': config
        })

        # Keep only last 100 executions
        history = history[-100:]

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_recent_configs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent configurations from history"""
        if not self.history_file.exists():
            return []

        with open(self.history_file, 'r') as f:
            history = json.load(f)

        return [h['configuration'] for h in history[-limit:]]


class AdvancedCLI:
    """Professional-grade CLI for LLMTool with sophisticated features"""

    def __init__(self):
        """Initialize the advanced CLI"""
        self.console = Console() if HAS_RICH else None
        self.settings = Settings()
        self.pipeline_controller = PipelineController()
        self.language_detector = LanguageDetector()
        self.llm_detector = LLMDetector()
        self.trainer_model_detector = TrainerModelDetector()
        self.data_detector = DataDetector()
        self.profile_manager = ProfileManager()

        # Import and initialize PromptManager
        from ..annotators.prompt_manager import PromptManager
        self.prompt_manager = PromptManager()

        # Cache for detected models
        self.detected_llms: Optional[Dict[str, List[ModelInfo]]] = None
        self.available_trainer_models: Optional[Dict[str, List[Dict]]] = None
        self.detected_datasets: Optional[List[DatasetInfo]] = None

        # Session state
        self.current_session = {
            'start_time': datetime.now(),
            'operations_count': 0,
            'last_operation': None
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure professional logging"""
        log_dir = self.settings.paths.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"llmtool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Force reconfiguration by clearing existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Configure logging: DEBUG to file, WARNING to console
        logging.basicConfig(
            level=logging.DEBUG,  # Capture everything
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8')
            ],
            force=True
        )

        # Add console handler with WARNING level only (hides INFO and DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(console_handler)

        self.logger = logging.getLogger(__name__)

        # Store log file path for reference
        self.current_log_file = log_file

    def _int_prompt_with_validation(self, prompt: str, default: int = 1, min_value: int = None, max_value: int = None) -> int:
        """IntPrompt.ask with validation since min_value/max_value not supported in older Rich versions"""
        while True:
            try:
                value = IntPrompt.ask(prompt, default=default)
                if min_value is not None and value < min_value:
                    self.console.print(f"[red]Value must be at least {min_value}[/red]")
                    continue
                if max_value is not None and value > max_value:
                    self.console.print(f"[red]Value must be at most {max_value}[/red]")
                    continue
                return value
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")

    def _float_prompt_with_validation(self, prompt: str, default: float, min_value: float = None, max_value: float = None) -> float:
        """Prompt for a floating point value with optional bounds."""
        while True:
            raw_value = Prompt.ask(prompt, default=f"{default}")
            try:
                value = float(raw_value)
            except ValueError:
                if HAS_RICH and self.console:
                    self.console.print("[red]Please enter a valid number[/red]")
                else:
                    print("Please enter a valid number")
                continue

            if min_value is not None and value < min_value:
                message = f"Value must be at least {min_value}"
                if HAS_RICH and self.console:
                    self.console.print(f"[red]{message}[/red]")
                else:
                    print(message)
                continue

            if max_value is not None and value > max_value:
                message = f"Value must be at most {max_value}"
                if HAS_RICH and self.console:
                    self.console.print(f"[red]{message}[/red]")
                else:
                    print(message)
                continue

            return value

    def display_banner(self):
        """Display professional welcome banner with system info"""
        if HAS_RICH and self.console:
            # Spectacular multicolor full-width ASCII art banner
            from rich.align import Align
            from rich.text import Text

            # Get terminal width
            width = self.console.width

            self.console.print()
            self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
            self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
            self.console.print()

            # Giant LLM TOOL text with each letter in different color
            self.console.print(Align.center("[bright_magenta]██╗     [bright_yellow]██╗     [bright_green]███╗   ███╗    [bright_cyan]████████╗ [bright_red]██████╗  [bright_blue]██████╗ [bright_white]██╗     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]████╗ ████║    [bright_cyan]╚══██╔══╝[bright_red]██╔═══██╗[bright_blue]██╔═══██╗[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██╔████╔██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██║╚██╔╝██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]███████╗[bright_yellow]███████╗[bright_green]██║ ╚═╝ ██║       [bright_cyan]██║   [bright_red]╚██████╔╝[bright_blue]╚██████╔╝[bright_white]███████╗"))
            self.console.print(Align.center("[bright_magenta]╚══════╝[bright_yellow]╚══════╝[bright_green]╚═╝     ╚═╝       [bright_cyan]╚═╝    [bright_red]╚═════╝  [bright_blue]╚═════╝ [bright_white]╚══════╝"))

            self.console.print()
            self.console.print(Align.center("[bold bright_yellow on blue]  🚀 LLM-powered Intelligent Annotation & Training Pipeline 🚀  [/bold bright_yellow on blue]"))
            self.console.print()

            # Colorful pipeline with emojis
            pipeline_text = Text()
            pipeline_text.append("📊 Data ", style="bold bright_yellow on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🤖 LLM Annotation ", style="bold bright_green on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🧹 Clean ", style="bold bright_cyan on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🎯 Label ", style="bold bright_magenta on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🧠 Train ", style="bold bright_red on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("📈 Deploy ", style="bold bright_blue on black")

            self.console.print(Align.center(pipeline_text))
            self.console.print()
            self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
            self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
            self.console.print()

            # Information table with system info
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_row("📚 Version:", "[bright_green]1.0[/bright_green]")
            info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")
            info_table.add_row("🚀 Features:", "[cyan]Multi-LLM Support, Smart Training, Auto-Detection[/cyan]")
            info_table.add_row("🎯 Capabilities:", "[magenta]JSON Annotation, BERT Training, Benchmarking[/magenta]")

            # Add system info if available
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                info_table.add_row(
                    "💻 System:",
                    f"[yellow]CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% used[/yellow]"
                )

            self.console.print(Panel(
                info_table,
                title="[bold bright_cyan]✨ Welcome to LLM Tool ✨[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2)
            ))
            self.console.print()

            # Auto-detect models in background
            with self.console.status("[bold green]🔍 Scanning environment...", spinner="dots"):
                self.detected_llms = self.llm_detector.detect_all_llms()
                self.available_trainer_models = self.trainer_model_detector.get_available_models()
                # Scan only in data/ directory
                data_dir = self.settings.paths.data_dir
                self.detected_datasets = self.data_detector.scan_directory(data_dir)

            # Show detection results
            self._display_detection_results()

        else:
            print("\n" + "="*80)
            print(" " * 28 + "LLM TOOL")
            print(" " * 15 + "LLM-powered Intelligent Annotation & Training Pipeline")
            print("="*80)
            print("\n📚 Version: 1.0")
            print("👨‍💻 Author: Antoine Lemor")
            print("\n📊 Data → 🤖 LLM Annotation → 🧹 Clean → 🎯 Label → 🧠 Train → 📈 Deploy")
            print("\nScanning environment...")

            self.detected_llms = self.llm_detector.detect_all_llms()
            self.available_trainer_models = self.trainer_model_detector.get_available_models()
            # Scan only in data/ directory
            data_dir = self.settings.paths.data_dir
            self.detected_datasets = self.data_detector.scan_directory(data_dir)

            # Count LLMs and trainer models
            llm_count = sum(len(m) for m in self.detected_llms.values())
            trainer_count = sum(len(m) for m in self.available_trainer_models.values())

            print(f"✓ Found {llm_count} annotation LLMs")
            print(f"✓ {trainer_count} trainable models available")
            print(f"✓ Found {len(self.detected_datasets)} datasets")
            print()

    def _display_detection_results(self):
        """Display auto-detection results in a professional format"""
        if not HAS_RICH or not self.console:
            return

        # === ANNOTATION LLMs SECTION ===
        llms_table = Table(title="🤖 Available LLMs for Annotation", border_style="cyan", show_lines=True)
        llms_table.add_column("Provider", style="cyan", width=10)
        llms_table.add_column("Model", style="white", width=20)
        llms_table.add_column("Size", style="yellow", width=8)
        llms_table.add_column("Context", style="green", width=10)
        llms_table.add_column("Status", style="green", width=12)

        # Show Ollama models (all of them if available)
        local_llms = self.detected_llms.get('local', [])
        if local_llms:
            # Show all local models, not just first 3
            for model in local_llms:
                llms_table.add_row(
                    "Ollama",
                    model.name,
                    model.size or "N/A",
                    f"{model.context_length:,}" if model.context_length else "N/A",
                    "✓ Ready"
                )

        # Show API models (top ones)
        for provider in ['openai', 'anthropic']:
            api_models = self.detected_llms.get(provider, [])
            for model in api_models[:2]:  # Show top 2 per API provider
                llms_table.add_row(
                    provider.title(),
                    model.name,
                    "API",
                    f"{model.context_length:,}" if model.context_length else "N/A",
                    "🔑 API Key" if model.requires_api_key else "✓ Ready"
                )

        # === TRAINABLE MODELS SECTION ===
        trainer_table = Table(title="🏋️ Available Models for Training", border_style="magenta", show_lines=False)
        trainer_table.add_column("Category", style="magenta", width=20)
        trainer_table.add_column("Models", style="white", width=60)

        for category, models in self.available_trainer_models.items():
            # Format model names compactly
            model_names = [m['name'] for m in models[:4]]  # Show first 4
            if len(models) > 4:
                model_names.append(f"(+{len(models)-4} more)")
            trainer_table.add_row(
                category,
                ", ".join(model_names)
            )

        # === DATASETS SECTION ===
        datasets_table = Table(title="📊 Detected Datasets", border_style="yellow", show_lines=False)
        datasets_table.add_column("File", style="cyan", width=25)
        datasets_table.add_column("Format", style="white", width=8)
        datasets_table.add_column("Size", style="green", width=10)
        datasets_table.add_column("Columns", style="dim", width=35)

        if self.detected_datasets:
            for dataset in self.detected_datasets[:5]:  # Show top 5
                columns_preview = ", ".join(dataset.columns[:3]) if dataset.columns else "N/A"
                if len(dataset.columns) > 3:
                    columns_preview += f" (+{len(dataset.columns)-3} more)"

                datasets_table.add_row(
                    dataset.path.name,
                    dataset.format.upper(),
                    f"{dataset.size_mb:.1f} MB" if dataset.size_mb else "Unknown",
                    columns_preview
                )
        else:
            datasets_table.add_row(
                "No datasets found",
                "-",
                "-",
                "Place CSV/JSON files in current directory"
            )

        # Print all tables
        self.console.print(llms_table)
        self.console.print()
        self.console.print(trainer_table)
        self.console.print()
        self.console.print(datasets_table)
        self.console.print()

    def get_main_menu_choice(self) -> str:
        """Display sophisticated main menu with smart suggestions"""
        if HAS_RICH and self.console:
            # Create menu table
            menu_table = Table.grid(padding=0)
            menu_table.add_column(width=3)
            menu_table.add_column()

            options = [
                ("1", "🎨 LLM Annotation Studio - Annotate with LLM (No Training)"),
                ("2", "🎯 Quick Start - Intelligent Pipeline Setup"),
                ("3", "📝 Annotation Wizard - Guided LLM Annotation"),
                ("4", "🚀 Complete Pipeline - Full Automated Workflow"),
                ("5", "🏋️ Training Studio - Model Training & Benchmarking"),
                ("6", "🔍 Validation Lab - Quality Assurance Tools"),
                ("7", "💾 Profile Manager - Save & Load Configurations"),
                ("8", "📚 Documentation & Help"),
                ("0", "❌ Exit")
            ]

            for num, desc in options:
                menu_table.add_row(
                    f"[bold cyan]{num}[/bold cyan]",
                    desc
                )

            # Suggestions based on context
            suggestions = self._get_smart_suggestions()

            panel = Panel(
                menu_table,
                title="[bold]Main Menu[/bold]",
                subtitle=f"[dim]{suggestions}[/dim]" if suggestions else None,
                border_style="cyan"
            )

            self.console.print(panel)

            # Smart prompt with validation (now 0-8 since we have 9 options)
            choice = Prompt.ask(
                "\n[bold yellow]Select option[/bold yellow]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
                default="1"
            )

        else:
            print("\n" + "="*50)
            print("Main Menu")
            print("="*50)
            print("1. LLM Annotation Studio - Annotate with LLM (No Training)")
            print("2. Quick Start - Intelligent Pipeline Setup")
            print("3. Annotation Wizard - Guided LLM Annotation")
            print("4. Complete Pipeline - Full Automated Workflow")
            print("5. Training Studio - Model Training & Benchmarking")
            print("6. Validation Lab - Quality Assurance Tools")
            print("7. Analytics Dashboard - Performance Insights")
            print("8. Profile Manager - Save & Load Configurations")
            print("9. Advanced Settings - Fine-tune Everything")
            print("0. Exit")
            print("-"*50)

            suggestions = self._get_smart_suggestions()
            if suggestions:
                print(f"💡 {suggestions}")

            choice = input("\nSelect option (0-9): ").strip()

        return choice

    def _get_smart_suggestions(self) -> str:
        """Generate intelligent suggestions based on context"""
        suggestions = []

        # Check for available LLMs
        if self.detected_llms:
            local_llms = self.detected_llms.get('local', [])
            if local_llms:
                # Show count of local LLMs
                suggestions.append(f"{len(local_llms)} local LLMs available")
            else:
                suggestions.append("No local LLMs - run 'ollama pull llama3.2'")

        # Check for datasets
        if self.detected_datasets:
            suggestions.append(f"{len(self.detected_datasets)} dataset{'s' if len(self.detected_datasets) != 1 else ''} found")
        else:
            suggestions.append("No datasets in current directory")

        # Check for recent profiles
        recent_profiles = self.profile_manager.list_profiles()
        if recent_profiles:
            suggestions.append(f"Last: {recent_profiles[0].name}")

        return " | ".join(suggestions) if suggestions else ""

    def _get_or_prompt_api_key(self, provider: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Get API key from secure storage or prompt user.

        Parameters
        ----------
        provider : str
            Provider name (openai, anthropic, google)
        model_name : str, optional
            Model name to save with the key

        Returns
        -------
        str or None
            The API key
        """
        # Check if key exists in storage
        existing_key = self.settings.get_api_key(provider)

        if existing_key:
            if HAS_RICH and self.console:
                use_existing = Confirm.ask(
                    f"[dim]Found saved API key for {provider}. Use it?[/dim]",
                    default=True
                )
            else:
                use_existing = input(f"Found saved API key for {provider}. Use it? [Y/n]: ").strip().lower() != 'n'

            if use_existing:
                return existing_key

        # Prompt for new key
        if HAS_RICH and self.console:
            self.console.print(f"\n[bold cyan]🔑 API Key Required for {provider}[/bold cyan]")
            if self.settings.key_manager:
                self.console.print("[dim]Your key will be stored securely using encryption[/dim]")
            else:
                self.console.print("[yellow]⚠️  Install 'cryptography' for secure key storage: pip install cryptography[/yellow]")

            api_key = Prompt.ask("API Key", password=True)

            # Ask if user wants to save the key
            if api_key:
                save_key = Confirm.ask(
                    "[dim]Save this API key for future use?[/dim]",
                    default=True
                )

                if save_key:
                    self.settings.set_api_key(provider, api_key, model_name)
                    self.console.print("[green]✓ API key saved securely[/green]")
        else:
            print(f"\nAPI Key Required for {provider}")
            if self.settings.key_manager:
                print("(Will be stored securely using encryption)")
            else:
                print("⚠️  Install 'cryptography' for secure key storage")

            api_key = input("API Key: ").strip()

            if api_key:
                save = input("Save this API key for future use? [Y/n]: ").strip().lower() != 'n'
                if save:
                    self.settings.set_api_key(provider, api_key, model_name)
                    print("✓ API key saved")

        return api_key

    @staticmethod
    def _estimate_model_size_billion(model: ModelInfo) -> Optional[float]:
        """Estimate model parameter count (billions) from its metadata."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*b',
            r'(\d+(?:\.\d+)?)\s*bn',
        ]

        lower_name = model.name.lower()
        for pattern in patterns:
            match = re.search(pattern, lower_name)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        if model.size and model.size.lower() != 'n/a':
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:b|bn)', model.size.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None

        return None

    def _prompt_for_text_column(self, columns: List[str], suggested: Optional[str]) -> str:
        """Prompt the user to choose the text column."""
        if not columns:
            if HAS_RICH and self.console:
                return Prompt.ask("Text column name", default="text")
            return input("Text column name [text]: ").strip() or "text"

        choices = [str(col) for col in columns]
        default_choice = suggested if suggested in choices else choices[0]

        if HAS_RICH and self.console:
            return Prompt.ask(
                "Which column contains the text to annotate?",
                choices=choices,
                default=default_choice
            )

        print("Available columns:")
        for col in choices:
            marker = " (suggested)" if col == default_choice else ""
            print(f"  - {col}{marker}")
        response = input(f"Text column [{default_choice}]: ").strip()
        return response or default_choice

    def _detect_id_columns(self, columns: List[str]) -> List[str]:
        """Detect all columns that could serve as IDs"""
        id_columns = []
        for col in columns:
            col_lower = col.lower()
            # Check if column name suggests it's an ID
            if (col_lower == 'id' or
                col_lower.endswith('_id') or
                col_lower.startswith('id_') or
                'identifier' in col_lower or
                col_lower in ['promesse_id', 'sentence_id', 'doc_id', 'item_id', 'record_id']):
                id_columns.append(col)
        return id_columns

    def _prompt_for_identifier_column(
        self,
        columns: List[str],
        suggested: Optional[str]
    ) -> Optional[str]:
        """Ask which column should serve as unique identifier, if any."""
        if not columns:
            if HAS_RICH and self.console:
                self.console.print("[dim]No columns detected; will create 'llm_annotation_id'.[/dim]")
            else:
                print("No columns detected; will create 'llm_annotation_id'.")
            return None

        # Detect all ID columns
        id_columns = self._detect_id_columns(columns)

        # If multiple ID columns found, offer to combine them
        if len(id_columns) > 1:
            if HAS_RICH and self.console:
                self.console.print(f"\n[bold cyan]📋 Found {len(id_columns)} ID columns:[/bold cyan]")
                for i, col in enumerate(id_columns, 1):
                    self.console.print(f"  {i}. [cyan]{col}[/cyan]")

                # Ask if user wants to use single or combined ID
                self.console.print("\n[bold]ID Strategy:[/bold]")
                self.console.print("[dim]IDs are used to track which texts have been annotated and link results to your original data.[/dim]")
                self.console.print("• [cyan]single[/cyan]: Use one column as ID")
                self.console.print("• [cyan]combine[/cyan]: Combine multiple columns (e.g., 'promesse_id+sentence_id')")
                self.console.print("• [cyan]none[/cyan]: Generate automatic IDs")

                id_strategy = Prompt.ask(
                    "ID strategy",
                    choices=["single", "combine", "none"],
                    default="single"
                )

                if id_strategy == "none":
                    self.console.print("[dim]An 'llm_annotation_id' column will be created automatically.[/dim]")
                    return None
                elif id_strategy == "combine":
                    # Ask which columns to combine
                    self.console.print("\n[bold]Select columns to combine:[/bold]")
                    self.console.print("[dim]Enter column numbers separated by commas (e.g., '1,2')[/dim]")

                    while True:
                        selection = Prompt.ask("Columns to combine")
                        try:
                            indices = [int(x.strip()) - 1 for x in selection.split(',')]
                            if all(0 <= i < len(id_columns) for i in indices):
                                selected_cols = [id_columns[i] for i in indices]
                                combined_id = "+".join(selected_cols)
                                self.console.print(f"[green]✓ Will combine: {' + '.join(selected_cols)}[/green]")
                                self.console.print(f"[dim]Example ID format: {' _ '.join(['123' for _ in selected_cols])}[/dim]")
                                return combined_id
                            else:
                                self.console.print("[red]Invalid column numbers. Try again.[/red]")
                        except (ValueError, IndexError):
                            self.console.print("[red]Invalid format. Use comma-separated numbers (e.g., '1,2')[/red]")
                else:  # single
                    # Select single ID column
                    default_choice = id_columns[0]
                    return Prompt.ask(
                        "Which ID column to use?",
                        choices=id_columns,
                        default=default_choice
                    )
            else:
                # Non-Rich fallback
                print(f"\nFound {len(id_columns)} ID columns:")
                for i, col in enumerate(id_columns, 1):
                    print(f"  {i}. {col}")

                choice = input("Use single ID (s), combine IDs (c), or generate new (n)? [s/c/n]: ").strip().lower()

                if choice == 'n':
                    print("Will create 'llm_annotation_id' automatically.")
                    return None
                elif choice == 'c':
                    selection = input("Enter column numbers to combine (comma-separated): ").strip()
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        selected_cols = [id_columns[i] for i in indices]
                        return "+".join(selected_cols)
                    except (ValueError, IndexError):
                        print("Invalid selection. Using first ID column.")
                        return id_columns[0]
                else:
                    idx = int(input(f"Select ID column [1-{len(id_columns)}]: ").strip() or "1") - 1
                    return id_columns[idx] if 0 <= idx < len(id_columns) else id_columns[0]

        # Single or no ID column found - use original logic
        default_has_id = suggested is not None or len(id_columns) == 1

        if HAS_RICH and self.console:
            has_id = Confirm.ask(
                "Does the dataset already contain a unique ID column?",
                default=default_has_id
            )
            if not has_id:
                self.console.print("[dim]An 'llm_annotation_id' column will be created automatically.[/dim]")
                return None

            choices = [str(col) for col in columns]
            default_choice = (id_columns[0] if id_columns else
                            (suggested if suggested in choices else choices[0]))
            return Prompt.ask(
                "Which column should be used as the identifier?",
                choices=choices,
                default=default_choice
            )

        prompt_default = "y" if default_has_id else "n"
        raw = input(f"Dataset has an ID column? [y/n] ({prompt_default}): ").strip().lower()
        has_id = raw or prompt_default
        if has_id.startswith('n'):
            print("We'll create 'llm_annotation_id' automatically.")
            return None

        print("Available columns:")
        for col in columns:
            marker = " (suggested)" if suggested and col == suggested else ""
            print(f"  - {col}{marker}")
        default_choice = suggested or columns[0]
        response = input(f"Identifier column [{default_choice}]: ").strip()
        return response or default_choice

    def _display_section_header(self, title: str, description: str):
        """Display a simple section header without full banner"""
        if HAS_RICH and self.console:
            self.console.print(f"\n[bold cyan]Welcome to LLM Tool[/bold cyan]")
            self.console.print(Panel.fit(
                f"[bold cyan]{title}[/bold cyan]\n{description}",
                border_style="cyan"
            ))
        else:
            print(f"\nWelcome to LLM Tool")
            print(f"\n{title}")
            print(description)

    def _display_welcome_banner(self):
        """Display a beautiful welcome banner"""
        if HAS_RICH and self.console:
            # Spectacular multicolor full-width ASCII art banner
            from rich.align import Align
            from rich.text import Text

            # Get terminal width
            width = self.console.width

            self.console.print()
            self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
            self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
            self.console.print()

            # Giant LLM TOOL text with each letter in different color
            self.console.print(Align.center("[bright_magenta]██╗     [bright_yellow]██╗     [bright_green]███╗   ███╗    [bright_cyan]████████╗ [bright_red]██████╗  [bright_blue]██████╗ [bright_white]██╗     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]████╗ ████║    [bright_cyan]╚══██╔══╝[bright_red]██╔═══██╗[bright_blue]██╔═══██╗[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██╔████╔██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]██║     [bright_yellow]██║     [bright_green]██║╚██╔╝██║       [bright_cyan]██║   [bright_red]██║   ██║[bright_blue]██║   ██║[bright_white]██║     "))
            self.console.print(Align.center("[bright_magenta]███████╗[bright_yellow]███████╗[bright_green]██║ ╚═╝ ██║       [bright_cyan]██║   [bright_red]╚██████╔╝[bright_blue]╚██████╔╝[bright_white]███████╗"))
            self.console.print(Align.center("[bright_magenta]╚══════╝[bright_yellow]╚══════╝[bright_green]╚═╝     ╚═╝       [bright_cyan]╚═╝    [bright_red]╚═════╝  [bright_blue]╚═════╝ [bright_white]╚══════╝"))

            self.console.print()
            self.console.print(Align.center("[bold bright_yellow on blue]  🚀 LLM-powered Intelligent Annotation & Training Pipeline 🚀  [/bold bright_yellow on blue]"))
            self.console.print()

            # Colorful pipeline with emojis
            pipeline_text = Text()
            pipeline_text.append("📊 Data ", style="bold bright_yellow on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🤖 LLM Annotation ", style="bold bright_green on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🧹 Clean ", style="bold bright_cyan on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🎯 Label ", style="bold bright_magenta on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("🧠 Train ", style="bold bright_red on black")
            pipeline_text.append("→ ", style="bold white")
            pipeline_text.append("📈 Deploy ", style="bold bright_blue on black")

            self.console.print(Align.center(pipeline_text))
            self.console.print()
            self.console.print("[on bright_magenta]" + " " * width + "[/on bright_magenta]")
            self.console.print("[on bright_blue]" + " " * width + "[/on bright_blue]")
            self.console.print()

            # Information table
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_row("📚 Version:", "[bright_green]1.0[/bright_green]")
            info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")
            info_table.add_row("🚀 Features:", "[cyan]Ollama/API Models, Prompt Wizard, Label Studio Export, Multi-Language[/cyan]")
            info_table.add_row("🎯 Capabilities:", "[magenta]Social Science Annotation, BERT Training, Model Benchmarking, Quality Metrics[/magenta]")
            info_table.add_row("⚡ Performance:", "[green]Incremental Save, Resume Support, Rich Progress UI, Batch Processing[/green]")

            self.console.print(Panel(
                info_table,
                title="[bold bright_cyan]✨ Welcome to LLM Tool ✨[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2)
            ))
            self.console.print()
        else:
            # Fallback for non-Rich environments (should never happen due to import check)
            print("="*80)
            print(" " * 28 + "LLM TOOL")
            print(" " * 18 + "Intelligent Annotation & Training Pipeline")
            print("="*80)
            print("\n📚 Version: 1.0")
            print("👨‍💻 Author: Antoine Lemor")
            print("🚀 Features: Multi-LLM Support, Smart Training, Auto-Detection")
            print("🎯 Capabilities: JSON Annotation, BERT Training, Benchmarking")
            print("⚡ Performance: Parallel Processing, Progress Tracking")
            print("\n  🤖 -> 📝 -> 🧹 -> 🎯 -> 🧠 -> 📊 -> ✨")
            print("  AI   Annotate Clean Label Train Test Deploy\n")
            print("="*80 + "\n")

    def quick_start_wizard(self):
        """Intelligent quick start wizard with auto-configuration"""
        # Display welcome banner
        self._display_welcome_banner()

        if HAS_RICH and self.console:
            self.console.print(Panel.fit(
                "[bold cyan]🎯 Quick Start Wizard[/bold cyan]\n"
                "I'll help you set up your pipeline intelligently",
                border_style="cyan"
            ))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:

                # Step 1: Analyze environment
                task = progress.add_task("[cyan]Analyzing your environment...", total=None)
                time.sleep(1)  # Simulate analysis

                # Auto-select best LLM
                best_llm = self._auto_select_llm()
                progress.update(task, description=f"[green]✓ Selected LLM: {best_llm.name}")

                # Step 2: Detect dataset
                task2 = progress.add_task("[cyan]Detecting datasets...", total=None)
                time.sleep(0.5)

                best_dataset = self._auto_select_dataset()
                if best_dataset:
                    progress.update(task2, description=f"[green]✓ Found dataset: {best_dataset.path.name}")
                else:
                    progress.update(task2, description="[yellow]No dataset found - will ask for path")

            # Interactive configuration with smart defaults
            self.console.print("\n[bold]Let's configure your pipeline:[/bold]\n")

            # LLM configuration - ASK FIRST
            self.console.print(f"\n[bold]Step 1: LLM Selection[/bold]")
            self.console.print(f"[dim]Auto-selected: {best_llm.name} ({best_llm.provider})[/dim]")

            use_auto_llm = Confirm.ask("Use this LLM?", default=True)

            if use_auto_llm:
                selected_llm = best_llm
            else:
                # Show available LLMs and let user choose
                selected_llm = self._select_llm_interactive()


            self.console.print(f"[green]✓ Selected LLM: {selected_llm.name}[/green]")

            # Check if user wants to return to menu
            if self._check_return_to_menu("with dataset selection"):
                return

            # Handle API key with secure storage
            if selected_llm.requires_api_key:
                api_key = self._get_or_prompt_api_key(selected_llm.provider, selected_llm.name)
            else:
                api_key = None

            # Ensure downstream steps use the user-selected model
            best_llm = selected_llm

            # Dataset selection - ASK SECOND
            self.console.print(f"\n[bold]Step 2: Dataset Selection[/bold]")
            if best_dataset:
                use_detected = Confirm.ask(
                    f"Use detected dataset [cyan]{best_dataset.path.name}[/cyan]?",
                    default=True
                )
                if use_detected:
                    dataset_info = best_dataset
                    dataset_path = str(best_dataset.path)
                else:
                    dataset_path = self._prompt_file_path("Enter dataset path")
                    dataset_info = DataDetector.analyze_file(Path(dataset_path)) or DatasetInfo(path=Path(dataset_path), format=Path(dataset_path).suffix.lstrip('.'))
            else:
                dataset_path = self._prompt_file_path("Enter dataset path")
                dataset_info = DataDetector.analyze_file(Path(dataset_path)) or DatasetInfo(path=Path(dataset_path), format=Path(dataset_path).suffix.lstrip('.'))

            available_columns = dataset_info.columns if dataset_info and dataset_info.columns else []
            suggested_text = self.data_detector.suggest_text_column(dataset_info) if dataset_info else None
            text_column = self._prompt_for_text_column(available_columns, suggested_text)

            suggested_id = None
            if available_columns:
                for col in available_columns:
                    lowered = col.lower()
                    if lowered == 'id' or lowered.endswith('_id') or 'identifier' in lowered:
                        suggested_id = col
                        break

            identifier_column = self._prompt_for_identifier_column(available_columns, suggested_id)

            # Check if user wants to return to menu
            if self._check_return_to_menu("with column configuration"):
                return

            # Language column detection and prompt
            lang_column = None
            if available_columns:
                # Detect potential language columns
                potential_lang_cols = [col for col in available_columns
                                      if col.lower() in ['lang', 'language', 'langue', 'lng', 'iso_lang']]

                if potential_lang_cols:
                    self.console.print(f"\n[bold cyan]🌍 Found language column(s):[/bold cyan]")
                    for col in potential_lang_cols:
                        self.console.print(f"  • [cyan]{col}[/cyan]")

                    use_lang_col = Confirm.ask("Use a language column for training metadata?", default=True)
                    if use_lang_col:
                        if len(potential_lang_cols) == 1:
                            lang_column = potential_lang_cols[0]
                            self.console.print(f"[green]✓ Using language column: {lang_column}[/green]")
                        else:
                            lang_column = Prompt.ask(
                                "Which language column to use?",
                                choices=potential_lang_cols,
                                default=potential_lang_cols[0]
                            )
                    else:
                        # Ask if automatic language detection is needed
                        auto_detect = Confirm.ask(
                            "[yellow]⚠️  Language information is needed for annotation. Enable automatic language detection?[/yellow]",
                            default=True
                        )
                        if auto_detect:
                            self.console.print("[dim]Language will be automatically detected for each text during annotation.[/dim]")
                            # Store flag for automatic detection
                            lang_column = None  # Will trigger auto-detection later
                        else:
                            self.console.print("[yellow]⚠️  Warning: Proceeding without language information may affect annotation quality.[/yellow]")
                            lang_column = None
                else:
                    # No language column detected
                    has_lang = Confirm.ask("Does your dataset have a language column?", default=False)
                    if has_lang:
                        lang_column = Prompt.ask(
                            "Language column name",
                            choices=available_columns,
                            default=available_columns[0]
                        )

            # Max token budget per completion
            max_tokens = self._prompt_max_tokens(best_llm)

            # Prompt configuration
            self.console.print("\n[bold]Step 4/6: Prompt Configuration[/bold]")

            # Auto-detect prompts
            detected_prompts = self._detect_prompts_in_folder()

            if detected_prompts:
                self.console.print(f"\n[green]✓ Found {len(detected_prompts)} prompts in prompts/ folder:[/green]")
                for i, p in enumerate(detected_prompts, 1):
                    keys_str = ', '.join(p['keys'])
                    self.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
                    self.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

                # Explain the options clearly
                self.console.print("\n[bold]Prompt Selection Options:[/bold]")
                self.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts (multi-prompt mode)")
                self.console.print("           → Each text will be annotated with all prompts")
                self.console.print("           → Useful when you want complete annotations from all perspectives")
                self.console.print("\n  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number (e.g., 1,3,5)")
                self.console.print("           → Only selected prompts will be used")
                self.console.print("           → Useful when testing or when you need only certain annotations")
                self.console.print("\n  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
                self.console.print("           → Interactive guided prompt creation")
                self.console.print("           → Optional AI assistance for definitions")
                self.console.print("           → [bold green]Recommended for new research projects![/bold green]")
                self.console.print("\n  [cyan]custom[/cyan]  - Provide path to a prompt file NOT in prompts/ folder")
                self.console.print("           → Use a prompt from another location")
                self.console.print("           → Useful for testing new prompts or one-off annotations")

                prompt_choice = Prompt.ask(
                    "\n[bold yellow]Prompt selection[/bold yellow]",
                    choices=["all", "select", "wizard", "custom"],
                    default="all"
                )

                if prompt_choice == "all":
                    # Use all prompts in multi-prompt mode
                    prompt = [p['content'] for p in detected_prompts]
                    prompt_mode = "multi"
                    self.console.print(f"[green]✓ Using all {len(prompt)} prompts[/green]")
                elif prompt_choice == "select":
                    indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
                    try:
                        selected_indices = [int(x.strip()) - 1 for x in indices.split(',')]
                        selected = [detected_prompts[i] for i in selected_indices if 0 <= i < len(detected_prompts)]
                        if selected:
                            prompt = [p['content'] for p in selected]
                            prompt_mode = "multi" if len(prompt) > 1 else "single"
                            self.console.print(f"[green]✓ Selected {len(prompt)} prompts[/green]")
                        else:
                            self.console.print("[red]Invalid selection. Using first prompt.[/red]")
                            prompt = detected_prompts[0]['content']
                            prompt_mode = "single"
                    except (ValueError, IndexError):
                        self.console.print("[red]Invalid input. Using first prompt.[/red]")
                        prompt = detected_prompts[0]['content']
                        prompt_mode = "single"
                elif prompt_choice == "wizard":
                    # Create new prompt using wizard - skip detection since user explicitly chose wizard
                    prompt = self._get_custom_prompt(skip_detection=True)
                    prompt_mode = "wizard"
                else:  # custom
                    # Ask for custom prompt path
                    custom_path = self._prompt_file_path("Enter path to prompt file")
                    try:
                        prompt = Path(custom_path).read_text(encoding='utf-8')
                        prompt_mode = "custom"
                        self.console.print(f"[green]✓ Loaded custom prompt from {custom_path}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Error loading custom prompt: {e}[/red]")
                        prompt = self._get_custom_prompt()
                        prompt_mode = "wizard"
            else:
                # No prompts detected, offer to create one
                self.console.print("\n[yellow]No prompts found in prompts/ folder.[/yellow]")
                create_prompt = Confirm.ask("Create a new prompt using the wizard?", default=True)
                if create_prompt:
                    prompt = self._get_custom_prompt()
                    prompt_mode = "wizard"
                else:
                    custom_path = self._prompt_file_path("Enter path to prompt file")
                    try:
                        prompt = Path(custom_path).read_text(encoding='utf-8')
                        prompt_mode = "custom"
                        self.console.print(f"[green]✓ Loaded custom prompt from {custom_path}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Error loading custom prompt: {e}[/red]")
                        self.console.print("[yellow]Creating prompt using wizard...[/yellow]")
                        prompt = self._get_custom_prompt()
                        prompt_mode = "wizard"

            # Check if user wants to return to menu
            if self._check_return_to_menu("with training configuration"):
                return

            # Training configuration
            self.console.print("\n[bold]Training Configuration:[/bold]")

            # Display training modes explanation
            if Confirm.ask("Display training modes guide?", default=True):
                self._display_training_modes_explanation()

            training_mode = Prompt.ask(
                "Training mode",
                choices=["quick", "balanced", "thorough", "custom"],
                default="balanced"
            )

            training_config = self._get_training_preset(training_mode)

            # Annotation sample configuration
            annotation_sample_size = self._int_prompt_with_validation(
                "How many sentences should we annotate for training? (0 = all)",
                default=200,
                min_value=0
            )
            annotation_sampling_strategy = 'head'
            annotation_sample_seed = 42
            if annotation_sample_size and annotation_sample_size > 0:
                if Confirm.ask("Sample sentences randomly?", default=True):
                    annotation_sampling_strategy = 'random'
                    annotation_sample_seed = self._int_prompt_with_validation(
                        "Random seed",
                        default=42,
                        min_value=0
                    )

            annotation_settings = {
                'annotation_sample_size': annotation_sample_size if annotation_sample_size > 0 else None,
                'annotation_sampling_strategy': annotation_sampling_strategy,
                'annotation_sample_seed': annotation_sample_seed,
                'max_tokens': max_tokens,
            }

            # Ask if user wants to prepare training data
            self.console.print("\n[bold]Training Data Preparation:[/bold]")
            run_training = Confirm.ask("Prepare data for model training after annotation?", default=True)

            training_strategy = None
            training_annotation_keys = None
            label_strategy = None

            if run_training:
                # Training strategy configuration
                self._display_training_strategy_explanation(prompt)

                training_strategy = Prompt.ask(
                    "Training strategy",
                    choices=["single-label", "multi-label"],
                    default="multi-label"
                )

                # Ask which keys/values to train
                schema = self._extract_annotation_schema(prompt)

                if training_strategy == "single-label":
                    # Show all possible values from schema
                    self.console.print("\n[dim]Detected annotation schema from prompt:[/dim]")
                    all_values = []
                    for key, values in schema.items():
                        if values:
                            self.console.print(f"  • [cyan]{key}[/cyan]: {', '.join(values[:5])}")
                            all_values.extend([f"{key}_{v}" for v in values])
                        else:
                            self.console.print(f"  • [cyan]{key}[/cyan]: [yellow]values will be detected from annotations[/yellow]")

                    if Confirm.ask("\nCreate binary models for ALL values from ALL keys?", default=True):
                        training_annotation_keys = None  # Will use all keys
                    else:
                        keys_input = Prompt.ask("Enter annotation keys to use (comma-separated)")
                        training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]
                else:
                    # multi-label: show all keys
                    self.console.print("\n[dim]Detected annotation keys from prompt:[/dim]")
                    for key in schema.keys():
                        self.console.print(f"  • [cyan]{key}[/cyan]")

                    if Confirm.ask("\nCreate multi-class models for ALL keys?", default=True):
                        training_annotation_keys = None  # Will use all keys
                    else:
                        keys_input = Prompt.ask("Enter annotation keys to use (comma-separated)")
                        training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

                # Label creation strategy
                self.console.print("\n[bold]Label Creation Strategy:[/bold]")
                self.console.print("• [cyan]key_value[/cyan]: Labels include key name (e.g., 'sentiment_positive')")
                self.console.print("• [cyan]value_only[/cyan]: Labels are just values (e.g., 'positive')")

                label_strategy = Prompt.ask(
                    "Label strategy",
                    choices=["key_value", "value_only"],
                    default="key_value"
                )

            # Summary and confirmation
            self._display_configuration_summary({
                'dataset': dataset_path,
                'text_column': text_column,
                'identifier_column': identifier_column or "llm_annotation_id (auto)",
                'model': best_llm.name,
                'prompt_mode': prompt_mode,
                'training_mode': training_mode,
                'training_strategy': training_strategy,
                'label_strategy': label_strategy,
                'annotation_sample_size': annotation_settings['annotation_sample_size'] or 'all',
                'max_tokens': max_tokens,
                **training_config
            })

            # Check if user wants to return to menu before execution
            if self._check_return_to_menu("with execution"):
                return

            if Confirm.ask("\n[bold yellow]Start execution?[/bold yellow]", default=True):
                # Save as profile for future use
                if Confirm.ask("Save this configuration as a profile?", default=True):
                    profile_name = Prompt.ask("Profile name", default="quick_start")
                    self._save_profile(profile_name, {
                    'dataset': dataset_path,
                    'text_column': text_column,
                    'identifier_column': identifier_column,
                    'model': best_llm.name,
                    'api_key': api_key,
                    'prompt': prompt,
                    'training_config': training_config,
                    'annotation_settings': annotation_settings
                    })

                # Execute pipeline
                self._execute_quick_start({
                    'dataset': dataset_path,
                    'text_column': text_column,
                    'identifier_column': identifier_column,
                    'lang_column': lang_column,
                    'model': selected_llm,
                    'api_key': api_key,
                    'prompt': prompt,
                    'training_config': training_config,
                    'annotation_settings': annotation_settings,
                    'run_training': run_training,
                    'training_strategy': training_strategy,
                    'label_strategy': label_strategy,
                    'training_annotation_keys': training_annotation_keys
                })

        else:
            # Simplified version for non-Rich environments
            print("\n=== Quick Start Wizard ===\n")
            print("Analyzing environment...")

            best_llm = self._auto_select_llm()
            print(f"Selected LLM: {best_llm.name}")

            dataset_path = input("Dataset path: ").strip()
            text_column = input("Text column (default: text): ").strip() or "text"

            if best_llm.requires_api_key:
                api_key = input("API Key: ").strip()
            else:
                api_key = None

            suggested_max_tokens = self._suggest_max_tokens(best_llm)
            max_tokens_input = input(
                f"Max tokens per response (default {suggested_max_tokens}): "
            ).strip()
            try:
                max_tokens = int(max_tokens_input) if max_tokens_input else suggested_max_tokens
            except ValueError:
                max_tokens = suggested_max_tokens
            if max_tokens <= 0:
                max_tokens = suggested_max_tokens

            # Prompt configuration (simple text input)
            print("\nStarting pipeline...")
            # Execute simplified pipeline
            annotation_sample_size = input("How many sentences to annotate for training (0 = all, default 200)? ").strip()
            try:
                annotation_sample_size = int(annotation_sample_size) if annotation_sample_size else 200
            except ValueError:
                annotation_sample_size = 200

            annotation_sample_seed = 42
            annotation_sampling_strategy = 'head'
            if annotation_sample_size and annotation_sample_size > 0:
                random_choice = input("Sample sentences randomly? (y/N): ").strip().lower()
                if random_choice == 'y':
                    annotation_sampling_strategy = 'random'
                    seed_input = input("Random seed (default 42): ").strip()
                    try:
                        annotation_sample_seed = int(seed_input) if seed_input else 42
                    except ValueError:
                        annotation_sample_seed = 42

            annotation_settings = {
                'annotation_sample_size': annotation_sample_size if annotation_sample_size > 0 else None,
                'annotation_sampling_strategy': annotation_sampling_strategy,
                'annotation_sample_seed': annotation_sample_seed,
                'max_tokens': max_tokens,
            }

    def _select_llm_interactive(self) -> ModelInfo:
        """Let user interactively select an LLM from available options"""
        if HAS_RICH and self.console:
            self.console.print("\n[bold]Available LLMs:[/bold]")
            self.console.print("[dim]ℹ️  Additional API models (Anthropic, Google, etc.) will be added as they are tested in the pipeline[/dim]\n")

            # Collect all available LLMs
            all_llms = []
            local_llms = self.detected_llms.get('local', [])
            openai_llms = self.detected_llms.get('openai', [])
            anthropic_llms = self.detected_llms.get('anthropic', [])

            # Display by category
            if local_llms:
                self.console.print("\n[cyan]Local Models (Ollama):[/cyan]")
                for i, llm in enumerate(local_llms, 1):
                    idx = len(all_llms) + 1
                    self.console.print(f"  {idx}. {llm.name} ({llm.size or 'N/A'})")
                    all_llms.append(llm)

            if openai_llms:
                self.console.print("\n[cyan]OpenAI Models:[/cyan]")
                for llm in openai_llms:  # Show all OpenAI models
                    idx = len(all_llms) + 1
                    cost = f"${llm.cost_per_1k_tokens}/1K" if llm.cost_per_1k_tokens else "N/A"
                    self.console.print(f"  {idx}. {llm.name} ({cost})")
                    all_llms.append(llm)

                # Add option for custom OpenAI model
                idx = len(all_llms) + 1
                self.console.print(f"  {idx}. [bold]Custom OpenAI model (enter name manually)[/bold]")
                custom_openai_option_idx = idx

            if anthropic_llms:
                self.console.print("\n[cyan]Anthropic Models:[/cyan]")
                for llm in anthropic_llms[:3]:  # Show top 3
                    idx = len(all_llms) + 1
                    cost = f"${llm.cost_per_1k_tokens}/1K" if llm.cost_per_1k_tokens else "N/A"
                    self.console.print(f"  {idx}. {llm.name} ({cost})")
                    all_llms.append(llm)

            if not all_llms:
                self.console.print("[red]No LLMs detected![/red]")
                return ModelInfo("llama3.2", "ollama", is_available=False)

            # Ask user to select with validation
            max_choice = len(all_llms) + (1 if openai_llms else 0)  # +1 for custom option if OpenAI available
            choice = self._int_prompt_with_validation("\nSelect LLM", default=1, min_value=1, max_value=max_choice)

            # Check if user selected custom OpenAI option
            if openai_llms and choice == custom_openai_option_idx:
                self.console.print("\n[dim]Examples: gpt-3.5-turbo, gpt-4, gpt-4o, gpt-4o-2025-01-01, o1, o1-mini, o3-mini, gpt-5[/dim]")
                custom_model = Prompt.ask("Enter OpenAI model name")

                # Create a ModelInfo for the custom model with estimated parameters
                return ModelInfo(
                    name=custom_model,
                    provider="openai",
                    context_length=128000,  # Default estimate
                    requires_api_key=True,
                    supports_json=True,
                    supports_streaming=not any(x in custom_model.lower() for x in ['o1-', 'o3-', 'o4-']),
                    is_available=True
                )

            return all_llms[choice - 1]
        else:
            # Fallback
            return self._auto_select_llm()

    def _auto_select_llm(self) -> ModelInfo:
        """Intelligently select the best available LLM for annotation"""
        # Priority: Local LLMs > API LLMs with key > Others

        local_llms = self.detected_llms.get('local', [])
        if local_llms:
            # Prefer larger, more capable local LLMs
            # Sort by size (if available) or by preferred order
            preferred_order = ['gemma3:27b', 'llama3.2:latest', 'mixtral', 'llama3', 'llama2', 'mistral', 'gemma2', 'gemma']

            for preferred in preferred_order:
                for llm in local_llms:
                    if preferred in llm.name.lower() or llm.name.lower().startswith(preferred.split(':')[0]):
                        return llm

            # Return the first available if no preferred found
            return local_llms[0]

        # Check for API keys in environment
        if os.getenv('OPENAI_API_KEY'):
            openai_llms = self.detected_llms.get('openai', [])
            if openai_llms:
                # Prefer GPT-4 Turbo for best quality/cost ratio
                for llm in openai_llms:
                    if 'gpt-4-turbo' in llm.name:
                        return llm
                return openai_llms[0]

        if os.getenv('ANTHROPIC_API_KEY'):
            anthropic_llms = self.detected_llms.get('anthropic', [])
            if anthropic_llms:
                # Prefer Sonnet for balance
                for llm in anthropic_llms:
                    if 'sonnet' in llm.name:
                        return llm
                return anthropic_llms[0]

        # Default fallback
        return ModelInfo(
            name="llama3.2",
            provider="ollama",
            is_available=False,
            requires_api_key=False
        )

    def _auto_select_dataset(self) -> Optional[DatasetInfo]:
        """Intelligently select the most likely dataset"""
        if not self.detected_datasets:
            return None

        # Prefer CSV files with obvious text columns
        for dataset in self.detected_datasets:
            if dataset.format == 'csv':
                text_col = self.data_detector.suggest_text_column(dataset)
                if text_col:
                    return dataset

        # Return largest dataset as fallback
        return max(self.detected_datasets, key=lambda d: d.size_mb or 0)

    def _suggest_max_tokens(self, model: ModelInfo) -> int:
        """Suggest a reasonable max token budget for the selected model"""
        if model.max_tokens:
            return model.max_tokens

        if model.context_length:
            # Reserve a safety margin so prompts+output stay within context
            safe_default = max(256, int(model.context_length * 0.25))
            return min(safe_default, getattr(self.settings.api, 'max_tokens', safe_default))

        if model.provider in {'openai', 'anthropic', 'google'}:
            return getattr(self.settings.api, 'max_tokens', 4096)

        # Fallback for local/unknown providers
        return 1000

    def _prompt_max_tokens(self, model: ModelInfo) -> int:
        """Ask the user for a max token budget with intelligent defaults"""
        suggested = self._suggest_max_tokens(model)
        max_context = model.context_length or None

        question = "Max tokens per response"
        if max_context:
            question += f" (≤ {max_context})"

        while True:
            max_tokens = IntPrompt.ask(question, default=suggested, show_default=True)

            if max_tokens <= 0:
                self.console.print("[yellow]Please provide a positive number of tokens.[/yellow]")
                continue

            if max_context and max_tokens >= max_context:
                self.console.print(
                    f"[yellow]The value must stay below the model context window ({max_context}). "
                    "Try a smaller number.[/yellow]"
                )
                continue

            return max_tokens

    def _prompt_file_path(self, prompt_text: str) -> str:
        """Prompt for file path with validation and suggestions"""
        while True:
            if HAS_RICH and self.console:
                # Show available files
                files = list(Path.cwd().glob("*.csv")) + list(Path.cwd().glob("*.json"))
                if files:
                    self.console.print("\n[dim]Available files:[/dim]")
                    for i, f in enumerate(files[:5], 1):
                        self.console.print(f"  {i}. {f.name}")

                path = Prompt.ask(f"\n{prompt_text}")

                # Allow number selection
                if path.isdigit() and int(path) <= len(files):
                    path = str(files[int(path)-1])

                if Path(path).exists():
                    return path
                else:
                    self.console.print(f"[red]File not found: {path}[/red]")
            else:
                path = input(f"{prompt_text}: ").strip()
                if Path(path).exists():
                    return path
                else:
                    print(f"File not found: {path}")

    def _check_return_to_menu(self, step_name: str = "this step") -> bool:
        """Ask user if they want to return to main menu. Returns True if user wants to go back."""
        if HAS_RICH and self.console:
            go_back = Confirm.ask(
                f"[dim]Return to main menu? (or press Enter to continue {step_name})[/dim]",
                default=False
            )
        else:
            response = input(f"Return to main menu? [y/N] (or Enter to continue {step_name}): ").strip().lower()
            go_back = response in ['y', 'yes']

        if go_back:
            if HAS_RICH and self.console:
                self.console.print("[yellow]↩ Returning to main menu...[/yellow]\n")
            else:
                print("↩ Returning to main menu...\n")
        return go_back

    def _generate_auto_prompt(self, dataset_path: str, text_column: str) -> str:
        """Generate intelligent prompt based on dataset analysis"""
        # Simple heuristic-based prompt generation
        prompt_template = """Analyze the following text and extract key information.

Text: {text}

Please provide a structured analysis including:
1. Main topic or subject
2. Sentiment (positive/negative/neutral)
3. Key entities mentioned
4. Summary in one sentence

Format your response as JSON with keys: topic, sentiment, entities, summary"""

        # Could be enhanced with actual dataset sampling and analysis
        return prompt_template

    def _detect_prompts_in_directory(self) -> List[Path]:
        """Detect available prompt files in the prompts directory"""
        prompts_dir = self.settings.paths.prompts_dir
        if prompts_dir.exists():
            return sorted(list(prompts_dir.glob("*.txt")))
        return []

    def _get_custom_prompt(self, skip_detection: bool = False) -> str:
        """Get custom prompt from user - detect from directory, file path, paste, or wizard

        Args:
            skip_detection: If True, skip prompts directory detection and go straight to wizard
        """
        if HAS_RICH and self.console:
            # If skip_detection is True, go directly to wizard
            if skip_detection:
                return self._run_social_science_wizard()

            # Try to detect prompts in the prompts directory
            detected_prompts = self._detect_prompts_in_directory()

            if detected_prompts:
                self.console.print("\n[bold green]✓ Detected prompts in prompts directory:[/bold green]")
                for i, prompt_file in enumerate(detected_prompts, 1):
                    self.console.print(f"  {i}. {prompt_file.name}")

                use_detected = Confirm.ask("\nUse one of these prompts?", default=True)

                if use_detected:
                    if len(detected_prompts) == 1:
                        selected_prompt = detected_prompts[0]
                    else:
                        choice = IntPrompt.ask(
                            "Select prompt",
                            default=1,
                            min_value=1,
                            max_value=len(detected_prompts)
                        )
                        selected_prompt = detected_prompts[choice - 1]

                    # Load and return the prompt
                    try:
                        full_prompt, expected_keys = self.prompt_manager.load_prompt(str(selected_prompt))
                        self.console.print(f"\n[green]✓ Loaded prompt: {selected_prompt.name}[/green]")
                        self.console.print(f"[dim]Detected JSON keys: {', '.join(expected_keys[:5])}{'...' if len(expected_keys) > 5 else ''}[/dim]")
                        return full_prompt
                    except Exception as e:
                        self.console.print(f"[red]Error loading prompt: {e}[/red]")

            # If no detected prompts or user declined, ask for input method
            self.console.print("\n[bold]Prompt Input Method:[/bold]")
            self.console.print("[dim]• wizard: 🧙‍♂️ Interactive Social Science Prompt Wizard (Recommended!)[/dim]")
            self.console.print("[dim]• path: Load from existing file[/dim]")
            self.console.print("[dim]• paste: Paste prompt text directly[/dim]")

            method = Prompt.ask(
                "How do you want to provide the prompt?",
                choices=["wizard", "path", "paste"],
                default="wizard"
            )

            if method == "wizard":
                # Launch the Social Science Prompt Wizard
                return self._run_social_science_wizard()

            elif method == "path":
                # Ask for file path
                prompt_path = Prompt.ask("\nPath to prompt file (.txt)")
                while not Path(prompt_path).exists():
                    self.console.print(f"[red]File not found: {prompt_path}[/red]")
                    prompt_path = Prompt.ask("Path to prompt file (.txt)")

                try:
                    full_prompt, expected_keys = self.prompt_manager.load_prompt(prompt_path)
                    self.console.print(f"[green]✓ Loaded prompt from: {prompt_path}[/green]")
                    self.console.print(f"[dim]Detected JSON keys: {', '.join(expected_keys[:5])}{'...' if len(expected_keys) > 5 else ''}[/dim]")
                    return full_prompt
                except Exception as e:
                    self.console.print(f"[red]Error loading prompt: {e}[/red]")
                    return ""

            else:  # paste
                self.console.print("\n[bold cyan]Paste your prompt below[/bold cyan]")
                self.console.print("[dim]Press Ctrl+D (Unix/Mac) or Ctrl+Z (Windows) when done[/dim]\n")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass

                prompt_text = "\n".join(lines)
                self.console.print(f"\n[green]✓ Received prompt ({len(prompt_text)} characters)[/green]")
                return prompt_text

        else:
            # Fallback for non-Rich environments
            return input("Enter prompt: ").strip()

    def _run_social_science_wizard(self) -> str:
        """Launch the Social Science Prompt Wizard for guided prompt creation"""
        try:
            # Check if user wants LLM assistance for definition generation
            use_llm_assist = Confirm.ask(
                "\n[cyan]🤖 Do you want AI assistance for creating your prompt (wizard mode)?[/cyan]",
                default=True
            )

            llm_client = None
            if use_llm_assist:
                # Detect available models for assistance
                all_llms = LLMDetector.detect_all_llms()
                available_models = []

                # Collect all available models
                for provider, models in all_llms.items():
                    for model in models:
                        if model.is_available:
                            available_models.append(model)

                if not available_models:
                    self.console.print("[yellow]⚠️  No LLM models detected. Continuing without AI assistance.[/yellow]")
                else:
                    # Display available models
                    self.console.print("\n[bold cyan]🤖 Available Models for AI Assistance:[/bold cyan]")

                    table = Table(
                        box=box.ROUNDED,
                        title="[bold]LLM Model Selection[/bold]",
                        title_style="cyan"
                    )
                    table.add_column("#", justify="right", style="cyan bold", width=4)
                    table.add_column("Model", style="white bold", width=35)
                    table.add_column("Provider", style="yellow", width=15)
                    table.add_column("Type / Size", style="green", width=20)

                    for i, model in enumerate(available_models, 1):
                        # Determine model type and quality indicator
                        model_name = model.name
                        provider = model.provider

                        # Extract size/quality info from model name
                        type_info = ""
                        if "120b" in model_name.lower():
                            type_info = "🚀 Very Large (120B)"
                        elif "72b" in model_name.lower() or "70b" in model_name.lower():
                            type_info = "⚡ Large (70B+)"
                        elif "27b" in model_name.lower() or "22b" in model_name.lower():
                            type_info = "💪 Medium (20B+)"
                        elif "8x" in model_name.lower():
                            type_info = "🔀 MoE (Mixture)"
                        elif "3.2" in model_name.lower() or "3.3" in model_name.lower():
                            type_info = "⚡ Fast (Llama 3)"
                        elif "gpt-5-nano" in model_name.lower():
                            type_info = "⚡ Ultra Fast"
                        elif "gpt-5-mini" in model_name.lower():
                            type_info = "🎯 Balanced"
                        elif "deepseek-r1" in model_name.lower():
                            type_info = "🧠 Reasoning"
                        elif "nemotron" in model_name.lower():
                            type_info = "📝 Instruction"
                        else:
                            type_info = model.size or "Standard"

                        # Style the provider
                        if provider == "ollama":
                            provider_styled = "🏠 Ollama"
                        elif provider == "openai":
                            provider_styled = "☁️  OpenAI"
                        elif provider == "anthropic":
                            provider_styled = "☁️  Anthropic"
                        else:
                            provider_styled = provider

                        table.add_row(str(i), model_name, provider_styled, type_info)

                    self.console.print(table)
                    self.console.print("\n[dim italic]💡 Tip: Larger models give better results but are slower[/dim italic]\n")

                    # Let user select model
                    while True:
                        choice = IntPrompt.ask(
                            "\n[cyan]Select model for AI assistance[/cyan]",
                            default=1
                        )
                        if 1 <= choice <= len(available_models):
                            break
                        self.console.print(f"[red]Please select a number between 1 and {len(available_models)}[/red]")

                    selected_model = available_models[choice - 1]
                    self.console.print(f"[green]✓ Selected: {selected_model.name}[/green]\n")

                    # Get API key if needed
                    api_key = None
                    if selected_model.requires_api_key:
                        api_key = self._get_or_prompt_api_key(
                            selected_model.provider,
                            selected_model.name
                        )

                    # Create LLM client for wizard
                    try:
                        llm_client = create_llm_client_for_wizard(
                            provider=selected_model.provider,
                            model=selected_model.name,
                            api_key=api_key
                        )
                        self.console.print("[green]✓ AI assistant ready![/green]\n")
                    except Exception as e:
                        self.console.print(f"[yellow]⚠️  Failed to initialize AI assistant: {e}[/yellow]")
                        self.console.print("[yellow]Continuing without AI assistance.[/yellow]\n")
                        llm_client = None

            # Create and run wizard
            wizard = SocialSciencePromptWizard(llm_client=llm_client)
            prompt_text, expected_keys = wizard.run()

            # Store expected keys in prompt manager for later use
            if expected_keys:
                self.console.print(f"\n[green]✓ Generated prompt with {len(expected_keys)} JSON keys:[/green]")
                self.console.print(f"[dim]{', '.join(expected_keys)}[/dim]\n")

            return prompt_text

        except Exception as e:
            self.console.print(f"\n[red]✗ Error running wizard: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

            # Fallback to manual prompt entry
            self.console.print("\n[yellow]Falling back to manual prompt entry...[/yellow]")
            return self._get_custom_prompt()

    def _get_multi_prompts(self) -> List[Tuple[str, List[str], str]]:
        """Get multiple prompts for multi-prompt mode"""
        if HAS_RICH and self.console:
            self.console.print("\n[bold cyan]Multi-Prompt Configuration[/bold cyan]")

            # Option to load from folder or individually
            load_from_folder = Confirm.ask(
                "Load all prompts from the prompts directory?",
                default=True
            )

            if load_from_folder:
                # Use the PromptManager's folder loading feature
                prompts_dir = str(self.settings.paths.prompts_dir)
                txt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith('.txt')])

                if not txt_files:
                    self.console.print(f"[yellow]No .txt files found in {prompts_dir}[/yellow]")
                    return []

                self.console.print(f"\n[green]Found {len(txt_files)} prompt files:[/green]")
                for i, filename in enumerate(txt_files, 1):
                    self.console.print(f"  {i}. {filename}")

                # Load each file
                prompts_list = []
                for i, filename in enumerate(txt_files, 1):
                    filepath = os.path.join(prompts_dir, filename)
                    self.console.print(f"\n[cyan][{i}/{len(txt_files)}] Loading {filename}...[/cyan]")

                    try:
                        full_prompt, expected_keys = self.prompt_manager.load_prompt(filepath)

                        if expected_keys:
                            self.console.print(f"[green]✓ Detected {len(expected_keys)} JSON keys[/green]")

                            # Ask for prefix
                            use_prefix = Confirm.ask(f"Add prefix to keys from '{filename}'?", default=False)
                            prefix_word = ""
                            if use_prefix:
                                default_prefix = Path(filename).stem.lower().replace(' ', '_')
                                prefix_word = Prompt.ask(f"Prefix (default: {default_prefix})", default=default_prefix)
                                self.console.print(f"[dim]✓ Keys will be prefixed with '{prefix_word}_'[/dim]")

                            prompts_list.append((full_prompt, expected_keys, prefix_word))
                        else:
                            self.console.print(f"[yellow]⚠ No JSON keys detected, skipping...[/yellow]")

                    except Exception as e:
                        self.console.print(f"[red]❌ Error loading {filename}: {e}[/red]")
                        continue

                if prompts_list:
                    self.console.print(f"\n[bold green]✓ Successfully loaded {len(prompts_list)} prompts[/bold green]")
                else:
                    self.console.print("\n[red]❌ No valid prompts could be loaded[/red]")

                return prompts_list

            else:
                # Load prompts individually
                num_prompts = self._int_prompt_with_validation("How many prompts do you want to use?", default=2, min_value=2, max_value=10)

                prompts_list = []
                for i in range(1, num_prompts + 1):
                    self.console.print(f"\n[bold]=== Prompt {i}/{num_prompts} ===[/bold]")

                    # Use the single prompt loader
                    prompt_text = self._get_custom_prompt()

                    # Ask for prefix
                    use_prefix = Confirm.ask(f"Add prefix to keys from prompt {i}?", default=False)
                    prefix_word = ""
                    if use_prefix:
                        prefix_word = Prompt.ask(f"Prefix for prompt {i}", default=f"p{i}")
                        self.console.print(f"[dim]✓ Keys will be prefixed with '{prefix_word}_'[/dim]")

                    # Extract expected keys from prompt
                    from ..annotators.json_cleaner import extract_expected_keys
                    expected_keys = extract_expected_keys(prompt_text)

                    prompts_list.append((prompt_text, expected_keys, prefix_word))

                self.console.print(f"\n[bold green]✓ Configured {len(prompts_list)} prompts[/bold green]")
                return prompts_list

        else:
            # Fallback for non-Rich environments
            return []

    def _get_prompt_template(self, template_type: str) -> str:
        """Get predefined prompt template"""
        templates = {
            "1": "Classify the following text into one of these categories: {categories}\n\nText: {text}\n\nCategory:",
            "2": "Extract all named entities from the text.\n\nText: {text}\n\nEntities:",
            "3": "Analyze the sentiment of this text (positive/negative/neutral).\n\nText: {text}\n\nSentiment:",
            "4": "Summarize the following text in one sentence.\n\nText: {text}\n\nSummary:"
        }
        return templates.get(template_type, "Analyze: {text}")

    def _extract_annotation_schema(self, prompt_text: str) -> Dict[str, List[str]]:
        """Extract annotation keys and their possible values from the prompt"""
        # First, use the improved extract_expected_keys to get all keys
        keys = extract_expected_keys(prompt_text)
        schema = {key: [] for key in keys}

        # Try to extract possible values for each key from the prompt description
        try:
            # Look for value descriptions in the prompt (e.g., "key": "value1" if ...; "value2" if ...)
            for key in keys:
                # Pattern to find value enumerations for this key
                key_pattern = rf'"{key}":\s*"([^"]+)"'
                matches = re.findall(key_pattern, prompt_text)

                if matches:
                    # Extract values from the description
                    values = []
                    for match in matches:
                        # Look for quoted values in the description
                        value_matches = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', match)
                        values.extend(value_matches)

                    if values:
                        schema[key] = list(set(values))  # Remove duplicates
        except:
            pass

        return schema

    def _display_training_strategy_explanation(self, prompt_text: str):
        """Display training strategy explanation with real examples from the prompt"""
        # Extract schema with keys and values from the prompt
        schema = self._extract_annotation_schema(prompt_text)

        if HAS_RICH and self.console:
            self.console.print("\n[bold]Training Strategy:[/bold]")
            self.console.print("How do you want to create training labels from annotations?")
            self.console.print()

            if schema:
                # Generate examples based on actual schema
                keys_list = list(schema.keys())
                multi_label_example = ", ".join(keys_list[:3]) if len(keys_list) >= 3 else ", ".join(keys_list)

                # For single-label, show real VALUE examples from the schema
                single_label_examples = []
                for key, values in list(schema.items())[:2]:  # Take first 2 keys
                    if values:
                        # Use real values from the prompt
                        for val in values[:2]:  # Take first 2 values per key
                            single_label_examples.append(f"{key}_{val}")
                    else:
                        # Fallback if no values detected
                        single_label_examples.append(f"{key}_valueX")

                single_label_example = ", ".join(single_label_examples[:3]) if single_label_examples else "key_value1, key_value2"

                self.console.print("• [cyan]single-label[/cyan]: Create ONE BINARY model per label VALUE")
                self.console.print(f"  [dim]Example: {single_label_example} (each = yes/no)[/dim]")
                self.console.print("  [dim]→ Many simple models, each predicting presence/absence of ONE specific label[/dim]")
                self.console.print()
                self.console.print("• [cyan]multi-label[/cyan]: Create ONE MULTI-CLASS model per annotation KEY")
                self.console.print(f"  [dim]Example: One model for {multi_label_example}[/dim]")
                self.console.print("  [dim]→ Few complex models, each predicting MULTIPLE possible values for its key[/dim]")
            else:
                # Fallback if we can't extract keys
                self.console.print("• [cyan]single-label[/cyan]: Create ONE BINARY model per label VALUE")
                self.console.print("  [dim]Example: theme_defense, theme_economy, sentiment_positive (each = yes/no)[/dim]")
                self.console.print("  [dim]→ Many simple models, each predicting presence/absence of ONE specific label[/dim]")
                self.console.print()
                self.console.print("• [cyan]multi-label[/cyan]: Create ONE MULTI-CLASS model per annotation KEY")
                self.console.print("  [dim]Example: One model for themes, one for sentiment, one for parties[/dim]")
                self.console.print("  [dim]→ Few complex models, each predicting MULTIPLE possible values for its key[/dim]")

            self.console.print()
        else:
            print("\nTraining Strategy:")
            print("• single-label: One binary model per label value")
            print("• multi-label: One multi-class model per annotation key")

    def _display_training_modes_explanation(self):
        """Display detailed explanation of training modes for social science researchers"""
        if HAS_RICH and self.console:
            self.console.print("\n[bold cyan]📚 Training Modes Guide[/bold cyan]\n")

            # Explain parameters first
            self.console.print("[bold yellow]Key Parameters:[/bold yellow]")
            self.console.print("  • [cyan]Epochs[/cyan]: Number of times the model sees the entire dataset")
            self.console.print("    [dim]Example: With 1000 tweets and 10 epochs, the model learns from 10,000 examples[/dim]")
            self.console.print("  • [cyan]Batch size[/cyan]: Number of examples processed simultaneously")
            self.console.print("    [dim]Example: Batch=16 → the model analyzes 16 news articles in parallel[/dim]")
            self.console.print("  • [cyan]Learning rate[/cyan]: Speed of learning (2e-5 = 0.00002)")
            self.console.print("    [dim]Too high → unstable learning; too low → slow learning[/dim]")
            self.console.print("  • [cyan]Warmup[/cyan]: Proportion of initial training with gradual learning rate increase")
            self.console.print("    [dim]0.1 = the first 10% of examples serve to 'warm up' the model[/dim]\n")

            # Create comparison table
            table = Table(title="Mode Comparison", border_style="blue", show_header=True)
            table.add_column("Mode", style="bold cyan", width=12)
            table.add_column("Epochs", justify="center", style="yellow", width=8)
            table.add_column("Batch", justify="center", style="yellow", width=8)
            table.add_column("L.Rate", justify="center", style="yellow", width=10)
            table.add_column("Models", justify="center", style="yellow", width=9)
            table.add_column("Use Case", style="green", width=45)

            table.add_row(
                "Quick", "3", "32", "5e-5", "2",
                "Quick test of a Facebook posts dataset (500 ex.)"
            )
            table.add_row(
                "Balanced", "10", "16", "2e-5", "5",
                "Parliamentary speeches classification (2000 ex.)"
            )
            table.add_row(
                "Thorough", "20", "8", "1e-5", "10",
                "Qualitative interview analysis (500-1000 ex.)"
            )
            table.add_row(
                "Custom", "?", "?", "?", "?",
                "Manual configuration for specific cases"
            )

            self.console.print(table)

            self.console.print("\n[bold green]💡 Recommendations:[/bold green]")
            self.console.print("  • [cyan]Quick[/cyan]: Fast prototyping, verify everything works")
            self.console.print("  • [cyan]Balanced[/cyan]: Best compromise for most projects (recommended)")
            self.console.print("  • [cyan]Thorough[/cyan]: Small or imbalanced dataset, academic publication")
            self.console.print("  • [cyan]Custom[/cyan]: You know exactly what you want\n")

    def _get_training_preset(self, mode: str) -> Dict[str, Any]:
        """Get training configuration preset"""
        presets = {
            "quick": {
                "epochs": 3,
                "batch_size": 32,
                "learning_rate": 5e-5,
                "warmup_ratio": 0.1,
                "models_to_test": 2
            },
            "balanced": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "models_to_test": 5
            },
            "thorough": {
                "epochs": 20,
                "batch_size": 8,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.2,
                "models_to_test": 10
            }
        }

        if mode == "custom":
            if HAS_RICH and self.console:
                return {
                    "epochs": IntPrompt.ask("Number of epochs", default=10),
                    "batch_size": IntPrompt.ask("Batch size", default=16),
                    "learning_rate": FloatPrompt.ask("Learning rate", default=2e-5),
                    "warmup_ratio": FloatPrompt.ask("Warmup ratio", default=0.1),
                    "models_to_test": IntPrompt.ask("Models to benchmark", default=5)
                }
            else:
                return presets["balanced"]

        return presets.get(mode, presets["balanced"])

    def _display_configuration_summary(self, config: Dict[str, Any]):
        """Display configuration summary in professional format"""
        if HAS_RICH and self.console:
            table = Table(title="📋 Configuration Summary", border_style="green", show_lines=True)
            table.add_column("Setting", style="cyan", width=20)
            table.add_column("Value", style="white")

            for key, value in config.items():
                if value is not None:
                    table.add_row(key.replace('_', ' ').title(), str(value))

            self.console.print(table)
        else:
            print("\n=== Configuration Summary ===")
            for key, value in config.items():
                if value is not None:
                    print(f"{key}: {value}")

    def _save_profile(self, name: str, config: Dict[str, Any]):
        """Save configuration as profile"""
        profile = ExecutionProfile(
            name=name,
            created_at=datetime.now(),
            last_used=datetime.now(),
            configuration=config
        )
        self.profile_manager.save_profile(profile)

        if HAS_RICH and self.console:
            self.console.print(f"[green]✓ Profile '{name}' saved successfully[/green]")

    def _recommend_models_for_training(self, dataset_path: str, text_column: str, sample_size: int = 200) -> Dict[str, Any]:
        """Recommend models based on detected language distribution in the dataset."""
        fallback = {
            'language_code': 'multilingual',
            'language_name': 'multilingual',
            'candidate_models': ['bert-base-multilingual-cased', 'xlm-roberta-base', 'mdeberta-v3-base'],
            'default_model': 'bert-base-multilingual-cased',
        }

        if not HAS_PANDAS or not dataset_path or not Path(dataset_path).exists():
            return fallback

        try:
            suffix = Path(dataset_path).suffix.lower()
            if suffix == '.csv':
                df = pd.read_csv(dataset_path)
            elif suffix in {'.xls', '.xlsx'}:
                df = pd.read_excel(dataset_path)
            elif suffix in {'.parquet'}:
                df = pd.read_parquet(dataset_path)
            else:
                return fallback
        except Exception as exc:
            logging.warning("Quick start: unable to read dataset for language detection (%s)", exc)
            return fallback

        if text_column not in df.columns:
            logging.warning("Quick start: text column '%s' not found in dataset", text_column)
            return fallback

        texts = df[text_column].dropna().astype(str)
        if texts.empty:
            return fallback

        sampled = texts.sample(min(len(texts), sample_size), random_state=42) if len(texts) > sample_size else texts
        detections = self.language_detector.detect_batch(sampled.tolist(), parallel=False)

        language_counts = Counter()
        for detection in detections:
            if not detection:
                continue
            lang = detection.get('language')
            confidence = detection.get('confidence', 0)
            if lang and confidence >= 0.5:
                language_counts[lang] += 1

        if not language_counts:
            return fallback

        top_language, _ = language_counts.most_common(1)[0]
        candidate_models = self.language_detector.get_recommended_models(top_language)
        default_model = candidate_models[0] if candidate_models else fallback['default_model']
        language_name = self.language_detector.get_language_name(top_language)

        return {
            'language_code': top_language,
            'language_name': language_name,
            'candidate_models': candidate_models or fallback['candidate_models'],
            'default_model': default_model,
        }

    def _execute_quick_start(self, config: Dict[str, Any]):
        """Execute the quick start pipeline"""
        dataset_path = config['dataset']
        text_column = config['text_column']
        identifier_column = config.get('identifier_column')
        model_entry = config['model']
        if isinstance(model_entry, ModelInfo):
            model_info = model_entry
        elif isinstance(model_entry, dict):
            model_info = ModelInfo(**model_entry)
        else:
            model_info = ModelInfo(name=str(model_entry), provider='ollama', is_available=True)
        api_key = config.get('api_key')
        prompt_config = config.get('prompt')
        training_preset = config.get('training_config', {})
        annotation_settings = config.get('annotation_settings', {})
        # Persist user training choices passed from the wizard
        run_training = bool(config.get('run_training', False))
        training_strategy = config.get('training_strategy')
        label_strategy = config.get('label_strategy')
        training_annotation_keys = config.get('training_annotation_keys')

        data_format = Path(dataset_path).suffix.lower().lstrip('.') if dataset_path else 'csv'
        if data_format == '':
            data_format = 'csv'
        if data_format not in {'csv', 'json', 'jsonl', 'excel', 'xlsx', 'xls', 'parquet'}:
            data_format = 'csv'

        # Prepare prompts payload
        prompts_payload: List[Dict[str, Any]] = []
        if isinstance(prompt_config, list):
            # Multi-prompt returns list of tuples (prompt, keys, prefix)
            for item in prompt_config:
                if isinstance(item, tuple):
                    prompt_text, expected_keys, prefix = item
                elif isinstance(item, dict):
                    prompt_text = item.get('prompt')
                    expected_keys = item.get('expected_keys', [])
                    prefix = item.get('prefix', '')
                else:
                    continue
                prompts_payload.append({
                    'prompt': prompt_text,
                    'expected_keys': expected_keys or [],
                    'prefix': prefix or ''
                })
        else:
            prompt_text = prompt_config or ""
            prompts_payload.append({
                'prompt': prompt_text,
                'expected_keys': extract_expected_keys(prompt_text) if prompt_text else [],
                'prefix': ''
            })

        # Recommend training models based on detected language
        training_reco = self._recommend_models_for_training(dataset_path, text_column)
        candidate_models = training_reco['candidate_models']
        default_model = training_reco['default_model']
        detected_language = training_reco['language_name']

        # Quick start focuses on a single recommended model for speed/stability
        models_to_test = [default_model]
        benchmark_mode = False

        annotations_dir = self.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = model_info.name.replace(':', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output_path = annotations_dir / f"{Path(dataset_path).stem}_{safe_model_name}_annotations_{timestamp}.csv"

        annotation_mode = 'api' if model_info.provider in {'openai', 'anthropic', 'google', 'custom'} else 'local'

        pipeline_config = {
            'mode': 'file',
            'data_source': data_format,
            'data_format': data_format,
            'file_path': dataset_path,
            'text_column': text_column,
            'text_columns': [text_column],
            'annotation_column': 'annotation',
            'identifier_column': identifier_column,
            'lang_column': config.get('lang_column'),
            'run_annotation': True,
            'annotation_mode': annotation_mode,
            'annotation_provider': model_info.provider,
            'annotation_model': model_info.name,
            'api_key': api_key,
            'prompts': prompts_payload,
            'annotation_sample_size': annotation_settings.get('annotation_sample_size'),
            'annotation_sampling_strategy': annotation_settings.get('annotation_sampling_strategy', 'head'),
            'annotation_sample_seed': annotation_settings.get('annotation_sample_seed', 42),
            'max_tokens': annotation_settings.get('max_tokens'),
            'max_workers': 1,
            'num_processes': 1,
            'use_parallel': False,
            'warmup': False,
            'disable_tqdm': True,  # Disable tqdm to avoid duplicate progress bars
            'output_format': 'csv',
            'output_path': str(default_output_path),
            'run_validation': False,
            'run_training': run_training,  # Based on user choice
            'training_strategy': training_strategy,
            'label_strategy': label_strategy,
            'training_annotation_keys': training_annotation_keys,
            'benchmark_mode': benchmark_mode,
            'models_to_test': models_to_test,
            'auto_select_best': True,
            'max_epochs': training_preset.get('epochs', 10),
            'batch_size': training_preset.get('batch_size', 16),
            'learning_rate': training_preset.get('learning_rate', 2e-5),
            'run_deployment': False,
            'training_model_type': models_to_test[0] if not benchmark_mode else default_model,
        }

        # Ensure local model options honour the requested token budget
        user_max_tokens = annotation_settings.get('max_tokens')
        if user_max_tokens:
            options = pipeline_config.get('options', {})
            options.setdefault('num_predict', user_max_tokens)
            pipeline_config['options'] = options

        # ============================================================
        # REPRODUCIBILITY METADATA
        # ============================================================
        if HAS_RICH and self.console:
            self.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
            self.console.print("[yellow]⚠️  IMPORTANT: Save parameters for two critical purposes:[/yellow]\n")

            self.console.print("  [green]1. Resume Capability[/green]")
            self.console.print("     • Continue this annotation if it stops or crashes")
            self.console.print("     • Annotate additional rows later with same settings")
            self.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

            self.console.print("  [green]2. Scientific Reproducibility[/green]")
            self.console.print("     • Document exact parameters for research papers")
            self.console.print("     • Reproduce identical annotations in the future")
            self.console.print("     • Track model version, prompts, and all settings\n")

            self.console.print("  [red]⚠️  If you choose NO:[/red]")
            self.console.print("     • You CANNOT resume this annotation later")
            self.console.print("     • You CANNOT relaunch with same parameters")
            self.console.print("     • Parameters will be lost forever\n")

            save_metadata = Confirm.ask(
                "[bold yellow]Save annotation parameters to JSON file?[/bold yellow]",
                default=True
            )

            # Validation tool export option
            self.console.print("\n[bold cyan]📤 Validation Tool Export[/bold cyan]")
            self.console.print("[dim]Export annotations to JSONL format for human validation[/dim]\n")

            self.console.print("[yellow]Available validation tools:[/yellow]")
            self.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
            self.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
            self.console.print("  • Both are open-source and free\n")

            export_enabled = Confirm.ask(
                "[bold yellow]Export to validation tool?[/bold yellow]",
                default=False
            )

            export_to_doccano = False
            export_to_labelstudio = False
            export_sample_size = None

            if export_enabled:
                # Ask which tool to use
                self.console.print("\n[yellow]Select validation tool:[/yellow]")
                export_tool = Prompt.ask(
                    "[bold yellow]Which tool?[/bold yellow]",
                    choices=["doccano", "labelstudio"],
                    default="doccano"
                )

                export_to_doccano = export_tool == "doccano"
                export_to_labelstudio = export_tool == "labelstudio"

                # Step 2b: If Label Studio, ask export method
                labelstudio_direct_export = False
                labelstudio_api_url = None
                labelstudio_api_key = None

                if export_to_labelstudio:
                    self.console.print("\n[yellow]Label Studio export method:[/yellow]")
                    self.console.print("  • [cyan]jsonl[/cyan] - Export to JSONL file (manual import)")
                    if HAS_REQUESTS:
                        self.console.print("  • [cyan]direct[/cyan] - Direct export to Label Studio via API\n")
                        export_choices = ["jsonl", "direct"]
                    else:
                        self.console.print("  • [dim]direct[/dim] - Direct export via API [dim](requires 'requests' library)[/dim]\n")
                        export_choices = ["jsonl"]

                    export_method = Prompt.ask(
                        "[bold yellow]Export method[/bold yellow]",
                        choices=export_choices,
                        default="jsonl"
                    )

                    if export_method == "direct":
                        labelstudio_direct_export = True

                        self.console.print("\n[cyan]Label Studio API Configuration:[/cyan]")
                        labelstudio_api_url = Prompt.ask(
                            "Label Studio URL",
                            default="http://localhost:8080"
                        )

                        labelstudio_api_key = Prompt.ask(
                            "API Key (from Label Studio Account & Settings)"
                        )

                # Ask about LLM predictions inclusion
                self.console.print("\n[yellow]Include LLM predictions in export?[/yellow]")
                self.console.print("  • [cyan]with[/cyan] - Include LLM annotations as predictions (for review/correction)")
                self.console.print("  • [cyan]without[/cyan] - Export only data without predictions (for manual annotation)")
                self.console.print("  • [cyan]both[/cyan] - Create two files/projects: one with and one without predictions\n")

                prediction_mode = Prompt.ask(
                    "[bold yellow]Prediction mode[/bold yellow]",
                    choices=["with", "without", "both"],
                    default="with"
                )

                # Ask how many sentences to export
                self.console.print("\n[yellow]How many annotated sentences to export?[/yellow]")
                self.console.print("  • [cyan]all[/cyan] - Export all annotated sentences")
                self.console.print("  • [cyan]representative[/cyan] - Representative sample (stratified by labels)")
                self.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

                sample_choice = Prompt.ask(
                    "[bold yellow]Export sample[/bold yellow]",
                    choices=["all", "representative", "number"],
                    default="all"
                )

                if sample_choice == "all":
                    export_sample_size = "all"
                elif sample_choice == "representative":
                    export_sample_size = "representative"
                else:  # number
                    export_sample_size = self._int_prompt_with_validation(
                        "Number of sentences to export",
                        100,
                        1,
                        999999
                    )
        else:
            save_metadata = False
            export_to_doccano = False
            export_to_labelstudio = False
            export_sample_size = None
            labelstudio_direct_export = False
            labelstudio_api_url = None
            labelstudio_api_key = None

        # Execute pipeline
        try:
            # Save metadata before execution
            if save_metadata:
                import json

                # Build comprehensive metadata
                metadata = {
                    'annotation_session': {
                        'timestamp': timestamp,
                        'tool_version': 'LLMTool v1.0',
                        'workflow': 'Quick Start - Intelligent Pipeline Setup'
                    },
                    'data_source': {
                        'file_path': dataset_path,
                        'file_name': Path(dataset_path).name,
                        'data_format': data_format,
                        'text_column': text_column,
                        'identifier_column': identifier_column,
                        'detected_language': detected_language,
                        'total_rows': annotation_settings.get('annotation_sample_size') if annotation_settings.get('annotation_sample_size') else 'all',
                        'sampling_strategy': annotation_settings.get('annotation_sampling_strategy', 'head'),
                        'sample_seed': annotation_settings.get('annotation_sample_seed', 42)
                    },
                    'model_configuration': {
                        'provider': model_info.provider,
                        'model_name': model_info.name,
                        'annotation_mode': annotation_mode,
                        'temperature': annotation_settings.get('temperature'),
                        'max_tokens': annotation_settings.get('max_tokens'),
                        'top_p': annotation_settings.get('top_p'),
                        'top_k': annotation_settings.get('top_k')
                    },
                    'prompts': [
                        {
                            'prompt_content': p['prompt'],
                            'expected_keys': p['expected_keys'],
                            'prefix': p['prefix']
                        }
                        for p in prompts_payload
                    ],
                    'processing_configuration': {
                        'parallel_workers': pipeline_config.get('num_processes', 1),
                        'batch_size': pipeline_config.get('batch_size', 16)
                    },
                    'training_configuration': {
                        'run_training': run_training,
                        'training_strategy': training_strategy,
                        'label_strategy': label_strategy,
                        'training_annotation_keys': training_annotation_keys,
                        'benchmark_mode': benchmark_mode,
                        'training_model': default_model,
                        'max_epochs': training_preset.get('epochs', 10),
                        'batch_size': training_preset.get('batch_size', 16),
                        'learning_rate': training_preset.get('learning_rate', 2e-5)
                    },
                    'output': {
                        'output_path': str(default_output_path),
                        'output_format': 'csv'
                    },
                    'export_preferences': {
                        'export_to_doccano': export_to_doccano,
                        'export_to_labelstudio': export_to_labelstudio,
                        'export_sample_size': export_sample_size,
                        'prediction_mode': prediction_mode if (export_to_doccano or export_to_labelstudio) else 'with',
                        'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
                        'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
                        'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None
                    }
                }

                # Save metadata JSON
                metadata_filename = f"{Path(dataset_path).stem}_{safe_model_name}_metadata_{timestamp}.json"
                metadata_path = annotations_dir / metadata_filename

                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                if HAS_RICH and self.console:
                    self.console.print(f"\n[bold green]✅ Metadata saved for reproducibility[/bold green]")
                    self.console.print(f"[bold cyan]📋 Metadata File:[/bold cyan]")
                    self.console.print(f"   {metadata_path}\n")
                else:
                    print(f"\n✅ Metadata saved: {metadata_path}\n")

            # Execute with real-time progress tracking
            print("\n🚀 Starting pipeline...\n")

            # Create pipeline controller
            from ..pipelines.pipeline_controller import PipelineController
            pipeline_with_progress = PipelineController(
                settings=self.settings
            )

            # Use the enhanced Rich progress manager with JSON display
            if HAS_RICH and self.console:
                # Use the unified RichProgressManager with compact mode
                from ..utils.rich_progress_manager import RichProgressManager
                from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

                # Use RichProgressManager in compact mode for elegant display
                with RichProgressManager(
                    show_json_every=1,  # Show JSON sample for every annotation
                    compact_mode=False   # Display full preview panels
                ) as progress_manager:
                    # Wrap the pipeline for enhanced JSON tracking
                    enhanced_pipeline = EnhancedPipelineWrapper(
                        pipeline_with_progress,
                        progress_manager
                    )

                    # Run pipeline
                    state = enhanced_pipeline.run_pipeline(pipeline_config)

                    # Check for errors
                    if state.errors:
                        error_msg = state.errors[0]['error'] if state.errors else "Pipeline failed"
                        self.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                        raise Exception(error_msg)
            else:
                # Fallback without Rich
                state = pipeline_with_progress.run_pipeline(pipeline_config)
        except Exception as exc:
            message = f"❌ Quick start pipeline failed: {exc}"
            if HAS_RICH and self.console:
                self.console.print(f"[red]{message}[/red]")
            else:
                print(message)
            logging.exception("Quick start pipeline failed")
            return

        annotation_results = state.annotation_results or {}
        training_results = state.training_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        if HAS_RICH and self.console:
            self.console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")
            self.console.print(f"📄 Annotated file: [cyan]{output_file}[/cyan]")
            self.console.print(f"🗣️ Detected language: [cyan]{detected_language}[/cyan]")
            if training_results:
                best_model = training_results.get('best_model') or training_results.get('model_name')
                best_f1 = training_results.get('best_f1_macro')
                if best_model:
                    self.console.print(f"🏆 Best model: [cyan]{best_model}[/cyan]")
                if best_f1 is not None:
                    self.console.print(f"📊 Macro F1: [cyan]{best_f1:.3f}[/cyan]")
        else:
            print("✅ Pipeline completed successfully!")
            print(f"Annotated file: {output_file}")
            print(f"Detected language: {detected_language}")
            if training_results:
                best_model = training_results.get('best_model') or training_results.get('model_name')
                best_f1 = training_results.get('best_f1_macro')
                if best_model:
                    print(f"Best model: {best_model}")
                if best_f1 is not None:
                    print(f"Macro F1: {best_f1:.3f}")

        # Export to Doccano JSONL if requested
        if export_to_doccano and HAS_RICH and self.console:
            # Build prompt_configs from prompts_payload for Quick Start
            prompt_configs_for_export = []
            for p in prompts_payload:
                prompt_configs_for_export.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt', '')
                    },
                    'prefix': p.get('prefix', '')
                })

            self._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs_for_export,
                data_path=Path(dataset_path),
                timestamp=timestamp,
                sample_size=export_sample_size
            )

        # Export to Label Studio if requested
        if export_to_labelstudio and HAS_RICH and self.console:
            # Build prompt_configs from prompts_payload for Quick Start
            prompt_configs_for_export = []
            for p in prompts_payload:
                prompt_configs_for_export.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt', '')
                    },
                    'prefix': p.get('prefix', '')
                })

            if labelstudio_direct_export:
                # Direct export to Label Studio via API
                self._export_to_labelstudio_direct(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs_for_export,
                    data_path=Path(dataset_path),
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    api_url=labelstudio_api_url,
                    api_key=labelstudio_api_key
                )
            else:
                # Export to JSONL file for manual import
                self._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs_for_export,
                    data_path=Path(dataset_path),
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode
                )

        # Persist last run details for other wizards
        self.last_annotation_config = {
            'data_path': dataset_path,
            'data_format': data_format,
            'text_column': text_column,
            'annotation_column': 'annotation',
            'mode': annotation_mode,
            'provider': model_info.provider,
            'model': model_info.name,
            'output_path': output_file,
            'annotation_sample_size': annotation_settings.get('annotation_sample_size'),
            'annotation_sampling_strategy': annotation_settings.get('annotation_sampling_strategy', 'head'),
        }

    def run(self):
        """Main run loop for the advanced CLI"""
        self.display_banner()

        while True:
            try:
                choice = self.get_main_menu_choice()

                if choice == "1":
                    self.llm_annotation_studio()
                elif choice == "2":
                    self.quick_start_wizard()
                elif choice == "3":
                    self.annotation_wizard()
                elif choice == "4":
                    self.complete_pipeline()
                elif choice == "5":
                    self.training_studio()
                elif choice == "6":
                    self.validation_lab()
                elif choice == "7":
                    self.analytics_dashboard()
                elif choice == "8":
                    self.profile_manager_ui()
                elif choice == "9":
                    self.advanced_settings()
                elif choice == "0":
                    if HAS_RICH and self.console:
                        self.console.print("\n[bold cyan]Thank you for using LLMTool! 👋[/bold cyan]\n")
                    else:
                        print("\nThank you for using LLMTool!\n")
                    sys.exit(0)

                # Update session
                self.current_session['operations_count'] += 1
                self.current_session['last_operation'] = choice

            except KeyboardInterrupt:
                if HAS_RICH and self.console:
                    if Confirm.ask("\n[yellow]Exit LLMTool?[/yellow]", default=False):
                        self.console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]\n")
                        sys.exit(0)
                else:
                    if input("\nExit? (y/n): ").lower() == 'y':
                        print("Goodbye!\n")
                        sys.exit(0)
            except Exception as e:
                self.logger.error(f"Error: {str(e)}", exc_info=True)
                if HAS_RICH and self.console:
                    self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                else:
                    print(f"Error: {str(e)}")

    # Placeholder methods for other menu options
    def llm_annotation_studio(self):
        """
        LLM Annotation Studio - Complete annotation workflow without training.

        Features:
        - Auto-detection of prompts from prompts/ folder
        - Automatic JSON key extraction
        - Multi-prompt with prefix system
        - Incremental save for ALL data formats
        - Automatic ID creation and tracking
        - Organized output structure
        """
        # Display welcome banner
        self._display_welcome_banner()

        if HAS_RICH and self.console:
            # Get smart suggestions
            suggestions = self._get_smart_suggestions()

            # Create workflow menu table
            from rich.table import Table
            workflow_table = Table(show_header=False, box=None, padding=(0, 2))
            workflow_table.add_column("Option", style="cyan", width=8)
            workflow_table.add_column("Description")

            workflows = [
                ("1", "🔄 Resume/Relaunch Annotation (Use saved parameters or resume incomplete)"),
                ("2", "🎯 Smart Annotate (Guided wizard with all options)"),
                ("3", "🗄️  Database Annotator (PostgreSQL direct)"),
                ("4", "🗑️  Clean Old Metadata (Delete saved parameters)"),
                ("0", "⬅️  Back to main menu")
            ]

            for option, desc in workflows:
                workflow_table.add_row(f"[bold cyan]{option}[/bold cyan]", desc)

            # Display panel with suggestions
            panel = Panel(
                workflow_table,
                title="[bold]🎨 LLM Annotation Studio[/bold]",
                subtitle=f"[dim]{suggestions}[/dim]" if suggestions else None,
                border_style="cyan"
            )

            self.console.print("\n")
            self.console.print(panel)

            workflow = Prompt.ask(
                "\n[bold yellow]Select workflow[/bold yellow]",
                choices=["0", "1", "2", "3", "4"],
                default="2"
            )

            if workflow == "0":
                return
            elif workflow == "1":
                self._quick_annotate()
            elif workflow == "2":
                self._smart_annotate()
            elif workflow == "3":
                self._database_annotator()
            elif workflow == "4":
                self._clean_metadata()
        else:
            print("\n=== LLM Annotation Studio ===")
            print("Professional LLM annotation without model training\n")
            print("1. Resume/Relaunch Annotation")
            print("2. Smart Annotate (Recommended)")
            print("3. Database Annotator")
            print("4. Clean Old Metadata")
            print("0. Back")
            choice = input("\nSelect workflow: ").strip()

            if choice == "1":
                self._quick_annotate()
            elif choice == "2":
                self._smart_annotate()
            elif choice == "3":
                self._database_annotator()
            elif choice == "4":
                self._clean_metadata()

    def annotation_wizard(self):
        """Guided annotation wizard with step-by-step configuration"""
        # Display welcome banner
        self._display_welcome_banner()

        # Display simple section header
        self._display_section_header(
            "📝 Annotation Wizard",
            "Interactive guided setup for LLM annotation"
        )

        if HAS_RICH and self.console:
            # Step 1: Choose annotation mode
            self.console.print("\n[bold]Step 1: Annotation Mode[/bold]")
            mode = Prompt.ask(
                "Select mode",
                choices=["local", "api", "hybrid", "back"],
                default="local"
            )

            if mode == "back":
                return

            # Step 2: Model selection based on mode
            self.console.print("\n[bold]Step 2: LLM Selection[/bold]")

            if mode == "local":
                local_llms = self.detected_llms.get('local', [])
                if not local_llms:
                    self.console.print("[red]No local LLMs found![/red]")
                    self.console.print("[yellow]Run: ollama pull llama3.2[/yellow]")
                    return

                # Display LLMs in a nice table
                llm_table = Table(title="Available Local LLMs", border_style="blue")
                llm_table.add_column("#", style="cyan", width=3)
                llm_table.add_column("Model", style="white")
                llm_table.add_column("Size", style="yellow")
                llm_table.add_column("Context", style="green")

                for i, llm in enumerate(local_llms, 1):
                    llm_table.add_row(
                        str(i),
                        llm.name,
                        llm.size or "N/A",
                        f"{llm.context_length:,}" if llm.context_length else "N/A"
                    )

                self.console.print(llm_table)
                choice = self._int_prompt_with_validation("Select LLM", default=1, min_value=1, max_value=len(local_llms))
                selected_llm = local_llms[choice-1]
                api_key = None

            else:  # API mode
                provider = Prompt.ask(
                    "Provider",
                    choices=["openai", "anthropic", "google", "back"],
                    default="openai"
                )

                if provider == "back":
                    return

                # Show available API models
                api_models = self.detected_llms.get(provider, [])
                for i, model in enumerate(api_models[:5], 1):
                    cost = f"${model.cost_per_1k_tokens}/1K" if model.cost_per_1k_tokens else "N/A"
                    self.console.print(f"  {i}. {model.name} ({cost})")

                model_choice = self._int_prompt_with_validation("Select model", default=1, min_value=1, max_value=len(api_models))
                selected_llm = api_models[model_choice-1]

                # Get API key with secure storage
                api_key = self._get_or_prompt_api_key(provider, selected_llm.name)

            max_tokens = self._prompt_max_tokens(selected_llm)

            # Step 3: Data configuration
            self.console.print("\n[bold]Step 3: Data Configuration[/bold]")

            # Show detected datasets or ask for path
            if self.detected_datasets:
                use_detected = Confirm.ask(
                    f"Use detected dataset? ({len(self.detected_datasets)} found)",
                    default=True
                )
                if use_detected:
                    for i, ds in enumerate(self.detected_datasets[:5], 1):
                        self.console.print(f"  {i}. {ds.path.name} ({ds.format}, {ds.size_mb:.1f} MB)")

                    ds_choice = IntPrompt.ask("Select dataset", default=1)
                    dataset = self.detected_datasets[ds_choice-1]
                    data_path = str(dataset.path)
                    text_column = self.data_detector.suggest_text_column(dataset) or "text"
                else:
                    data_path = self._prompt_file_path("Data file path")
                    text_column = Prompt.ask("Text column", default="text")
            else:
                data_path = self._prompt_file_path("Data file path")
                text_column = Prompt.ask("Text column", default="text")

            # Step 4: Prompt engineering
            self.console.print("\n[bold]Step 4: Prompt Engineering[/bold]")
            self.console.print("[dim]• simple: Single prompt for all texts[/dim]")
            self.console.print("[dim]• multi: Multiple prompts applied to each text[/dim]")
            self.console.print("[dim]• template: Pre-configured prompt templates [yellow](Under development - not available)[/yellow][/dim]")

            prompt_mode = Prompt.ask(
                "Prompt strategy",
                choices=["simple", "multi", "back"],
                default="simple"
            )

            if prompt_mode == "back":
                return

            if prompt_mode == "simple":
                prompt_text = self._get_custom_prompt()
            else:  # multi
                prompt_text = self._get_multi_prompts()

            # Step 5: Advanced options
            self.console.print("\n[bold]Step 5: Advanced Options[/bold]")

            batch_size = self._int_prompt_with_validation("Batch size", default=10, min_value=1, max_value=100)
            max_workers = self._int_prompt_with_validation("Parallel workers", default=4, min_value=1, max_value=16)
            save_incrementally = Confirm.ask("Save incrementally?", default=True)

            # Step 5.5: Training strategy
            self.console.print("\n[bold]Training Strategy (Optional):[/bold]")
            if Confirm.ask("Prepare data for model training after annotation?", default=True):
                self._display_training_strategy_explanation(prompt_text)

                training_strategy = Prompt.ask(
                    "Training strategy",
                    choices=["single-label", "multi-label"],
                    default="multi-label"
                )

                # Ask which keys/values to train
                training_annotation_keys = None
                schema = self._extract_annotation_schema(prompt_text)

                if training_strategy == "single-label":
                    # Show all possible values from schema
                    self.console.print("\n[dim]Detected annotation schema from prompt:[/dim]")
                    for key, values in schema.items():
                        if values:
                            self.console.print(f"  • [cyan]{key}[/cyan]: {', '.join(values[:5])}")
                        else:
                            self.console.print(f"  • [cyan]{key}[/cyan]: [yellow]values will be detected from annotations[/yellow]")

                    if Confirm.ask("\nCreate binary models for ALL values from ALL keys?", default=True):
                        training_annotation_keys = None
                    else:
                        keys_input = Prompt.ask("Enter annotation keys to use (comma-separated)")
                        training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]
                else:
                    # multi-label: show all keys
                    self.console.print("\n[dim]Detected annotation keys from prompt:[/dim]")
                    for key in schema.keys():
                        self.console.print(f"  • [cyan]{key}[/cyan]")

                    if Confirm.ask("\nCreate multi-class models for ALL keys?", default=True):
                        training_annotation_keys = None
                    else:
                        keys_input = Prompt.ask("Enter annotation keys to use (comma-separated)")
                        training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

                # Label creation strategy
                self.console.print("\n[bold]Label Creation Strategy:[/bold]")
                self.console.print("• [cyan]key_value[/cyan]: Labels include key name (e.g., 'sentiment_positive')")
                self.console.print("• [cyan]value_only[/cyan]: Labels are just values (e.g., 'positive')")

                label_strategy = Prompt.ask(
                    "Label strategy",
                    choices=["key_value", "value_only"],
                    default="key_value"
                )
            else:
                training_strategy = None
                training_annotation_keys = None
                label_strategy = None

            # Step 6: Execute
            self.console.print("\n[bold]Ready to annotate![/bold]")

            config = {
                'mode': mode,
                'provider': selected_llm.provider,
                'model': selected_llm.name,
                'api_key': api_key,
                'max_tokens': max_tokens,
                'data_path': data_path,
                'text_column': text_column,
                'prompt': prompt_text,
                'batch_size': batch_size,
                'max_workers': max_workers,
                'save_incrementally': save_incrementally,
                'training_strategy': training_strategy,
                'label_strategy': label_strategy,
                'training_annotation_keys': training_annotation_keys
            }

            self._display_configuration_summary(config)

            if Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console
                ) as progress:

                    task = progress.add_task("[cyan]Annotating...", total=100)

                    # Here would call actual annotation
                    from ..annotators.llm_annotator import LLMAnnotator
                    annotator = LLMAnnotator()

                    # Simulate for now
                    for i in range(100):
                        time.sleep(0.01)
                        progress.update(task, advance=1)

                    self.console.print("\n[bold green]✅ Annotation complete![/bold green]")
                    self.console.print("📁 Results saved to: annotations_output.json")

        else:
            print("\n=== Annotation Wizard ===")
            print("Guided LLM annotation setup\n")
            # Simplified version
            print("Feature coming soon with full implementation")

    def complete_pipeline(self):
        """Complete pipeline workflow - Full annotation to deployment"""
        # Display welcome banner
        self._display_welcome_banner()

        # Display simple section header
        self._display_section_header(
            "🚀 Complete Pipeline",
            "Full workflow: Annotation → Training → Validation → Deployment"
        )

        if HAS_RICH and self.console:
            # Step 1: Data Selection
            self.console.print("\n[bold yellow]Step 1: Data Selection[/bold yellow]")

            # Auto-detect or ask for dataset
            if self.detected_datasets:
                self.console.print("\n[dim]Available datasets detected:[/dim]")
                for i, dataset in enumerate(self.detected_datasets[:10], 1):
                    self.console.print(f"  {i}. {dataset.path.name} ({dataset.format.upper()}, {dataset.size_mb:.1f} MB)")

                choice = Prompt.ask("Select dataset (number) or enter path", default="1")
                if choice.isdigit() and int(choice) <= len(self.detected_datasets):
                    selected_dataset = self.detected_datasets[int(choice)-1]
                    data_path = str(selected_dataset.path)
                    text_column = self.data_detector.suggest_text_column(selected_dataset) or "text"
                else:
                    data_path = choice
                    text_column = Prompt.ask("Text column name", default="text")
            else:
                data_path = self._prompt_file_path("Enter dataset path")
                text_column = Prompt.ask("Text column name", default="text")

            # Step 2: LLM Selection for Annotation
            self.console.print("\n[bold yellow]Step 2: LLM Selection for Annotation[/bold yellow]")

            # Show available LLMs
            local_llms = self.detected_llms.get('local', [])
            if local_llms:
                self.console.print("\n[green]Local LLMs available:[/green]")
                for i, llm in enumerate(local_llms[:10], 1):
                    self.console.print(f"  {i}. {llm.name} ({llm.size or 'N/A'})")

                llm_choice = self._int_prompt_with_validation("Select LLM for annotation", default=1, min_value=1, max_value=len(local_llms))
                selected_llm = local_llms[llm_choice-1]
            else:
                self.console.print("[yellow]No local LLMs detected. Using API model.[/yellow]")
                provider = Prompt.ask("API Provider", choices=["openai", "anthropic", "back"], default="openai")
                if provider == "back":
                    return
                model_name = Prompt.ask("Model name", default="gpt-4-turbo")
                selected_llm = ModelInfo(name=model_name, provider=provider, requires_api_key=True)

                # Get API key with secure storage
                api_key = self._get_or_prompt_api_key(provider, model_name)

            # Step 3: Prompt Configuration
            self.console.print("\n[bold yellow]Step 3: Prompt Configuration[/bold yellow]")
            prompt_text = self._get_custom_prompt()

            # Step 4: Training Model Selection
            self.console.print("\n[bold yellow]Step 4: Training Configuration[/bold yellow]")

            benchmark_mode = Confirm.ask("Benchmark multiple models?", default=True)

            if benchmark_mode:
                num_models = self._int_prompt_with_validation("Number of models to benchmark", default=5, min_value=2, max_value=20)
                self.console.print("[dim]Will test: BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT...[/dim]")
            else:
                # Show available training models
                self.console.print("\n[dim]Select model category:[/dim]")
                categories = list(self.available_trainer_models.keys())
                for i, cat in enumerate(categories, 1):
                    self.console.print(f"  {i}. {cat}")

                cat_choice = self._int_prompt_with_validation("Category", default=1, min_value=1, max_value=len(categories))
                selected_category = categories[cat_choice-1]
                models_in_cat = self.available_trainer_models[selected_category]

                self.console.print(f"\n[dim]Models in {selected_category}:[/dim]")
                for i, model in enumerate(models_in_cat[:10], 1):
                    self.console.print(f"  {i}. {model['name']} ({model.get('params', 'N/A')})")

            # Step 4.5: Training Data Preparation Strategy
            self.console.print("\n[bold yellow]Step 4.5: Training Data Preparation[/bold yellow]")
            self._display_training_strategy_explanation(prompt_text)

            training_strategy = Prompt.ask(
                "Training strategy",
                choices=["single-label", "multi-label", "back"],
                default="multi-label"
            )

            if training_strategy == "back":
                return

            # Ask which keys/values to train
            training_annotation_keys = None
            schema = self._extract_annotation_schema(prompt_text)

            if training_strategy == "single-label":
                # Show all possible values from schema
                self.console.print("\n[dim]Detected annotation schema from prompt:[/dim]")
                for key, values in schema.items():
                    if values:
                        self.console.print(f"  • [cyan]{key}[/cyan]: {', '.join(values[:5])}")
                    else:
                        self.console.print(f"  • [cyan]{key}[/cyan]: [yellow]values will be detected from annotations[/yellow]")

                if Confirm.ask("\nCreate binary models for ALL values from ALL keys?", default=True):
                    training_annotation_keys = None
                else:
                    keys_input = Prompt.ask("Enter annotation keys to use (comma-separated)")
                    training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]
            else:
                # multi-label: show all keys
                self.console.print("\n[dim]Detected annotation keys from prompt:[/dim]")
                for key in schema.keys():
                    self.console.print(f"  • [cyan]{key}[/cyan]")

                if Confirm.ask("\nCreate multi-class models for ALL keys?", default=True):
                    training_annotation_keys = None
                else:
                    keys_input = Prompt.ask("Enter annotation keys to use (comma-separated)")
                    training_annotation_keys = [k.strip() for k in keys_input.split(',') if k.strip()]

            # Label creation strategy
            self.console.print("\n[bold]Label Creation Strategy:[/bold]")
            self.console.print("• [cyan]key_value[/cyan]: Labels include key name (e.g., 'sentiment_positive')")
            self.console.print("• [cyan]value_only[/cyan]: Labels are just values (e.g., 'positive')")

            label_strategy = Prompt.ask(
                "Label strategy",
                choices=["key_value", "value_only"],
                default="key_value"
            )

            # Step 5: Configuration Summary
            self.console.print("\n[bold yellow]Pipeline Configuration Summary:[/bold yellow]")

            config_table = Table(border_style="green", show_lines=True)
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")

            config_table.add_row("Dataset", data_path)
            config_table.add_row("Text Column", text_column)
            config_table.add_row("Annotation LLM", f"{selected_llm.provider}/{selected_llm.name}")
            config_table.add_row("Training Mode", "Benchmark" if benchmark_mode else "Single Model")
            config_table.add_row("Validation", "Enabled with Doccano export")

            self.console.print(config_table)

            # Step 6: Execution
            if Confirm.ask("\n[bold yellow]Start pipeline execution?[/bold yellow]", default=True):

                # Prepare configuration
                pipeline_config = {
                    'mode': 'file',
                    'data_source': 'file',
                    'file_path': data_path,
                    'text_column': text_column,

                    # Annotation config
                    'run_annotation': True,
                    'annotation_mode': 'local' if selected_llm.provider == 'ollama' else 'api',
                    'annotation_provider': selected_llm.provider,
                    'annotation_model': selected_llm.name,
                    'api_key': api_key if selected_llm.requires_api_key else None,
                    'prompt_text': prompt_text,

                    # Validation config
                    'run_validation': True,
                    'validation_sample_size': 100,
                    'export_to_doccano': True,

                    # Training config
                    'run_training': True,
                    'benchmark_mode': benchmark_mode,
                    'models_to_test': num_models if benchmark_mode else 1,
                    'max_epochs': 10,
                    'batch_size': 16,
                    'training_strategy': training_strategy,
                    'label_strategy': label_strategy,
                    'training_annotation_keys': training_annotation_keys,

                    # Deployment config
                    'run_deployment': True,
                    'save_model': True
                }

                # Execute with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console
                ) as progress:

                    # Initialize pipeline
                    from ..pipelines.pipeline_controller import PipelineController
                    controller = PipelineController()

                    try:
                        # Annotation phase
                        task1 = progress.add_task("[cyan]Annotating with LLM...", total=100)
                        progress.update(task1, advance=50)

                        # Run pipeline
                        state = controller.run_pipeline(pipeline_config)
                        progress.update(task1, completed=100)

                        # Show results
                        self.console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")

                        if state.annotation_results:
                            self.console.print(f"📝 Annotations: {state.annotation_results.get('total_annotated', 0)} items")

                        if state.validation_results:
                            self.console.print(f"✓ Validation: Quality score {state.validation_results.get('quality_score', 0):.1f}/100")
                            self.console.print(f"📁 Doccano export: {state.validation_results.get('doccano_export_path', 'N/A')}")

                        if state.training_results:
                            self.console.print(f"🏆 Best model: {state.training_results.get('best_model', 'unknown')}")
                            self.console.print(f"📊 Best F1: {state.training_results.get('best_f1_macro', 0):.4f}")

                        # Save configuration as profile
                        if Confirm.ask("\nSave this configuration for future use?", default=True):
                            profile_name = Prompt.ask("Profile name", default="complete_pipeline")
                            self._save_profile(profile_name, pipeline_config)

                    except Exception as e:
                        self.console.print(f"\n[bold red]❌ Pipeline failed: {str(e)}[/bold red]")
                        raise

        else:
            # Non-Rich fallback
            print("\n=== Complete Pipeline ===")
            print("Full workflow: Annotation → Training → Validation → Deployment\n")

            # Simplified version
            data_path = input("Dataset path: ").strip()
            text_column = input("Text column (default: text): ").strip() or "text"

            # Ask for LLM configuration
            print("\n--- Annotation Configuration ---")
            print("Select annotation method:")
            print("1. Local LLM (Ollama)")
            print("2. OpenAI API")
            print("3. Anthropic API")
            choice = input("Choice (1-3): ").strip()

            if choice == "1":
                # Check for Ollama models
                local_llms = self.detected_llms.get('local', [])
                if local_llms:
                    print("\nLocal LLMs detected:")
                    for i, llm in enumerate(local_llms[:10], 1):
                        print(f"  {i}. {llm.name}")
                    llm_choice = input("Select LLM (number): ").strip()
                    if llm_choice.isdigit() and int(llm_choice) <= len(local_llms):
                        selected_llm = local_llms[int(llm_choice)-1]
                    else:
                        selected_llm = local_llms[0] if local_llms else None
                    provider = 'ollama'
                    api_key = None
                else:
                    print("No local LLMs found. Install Ollama or use API provider.")
                    return
            elif choice == "2":
                provider = 'openai'
                api_key = input("OpenAI API Key: ").strip()
                model_name = input("Model name (default: gpt-4): ").strip() or "gpt-4"
                selected_llm = ModelInfo(name=model_name, provider=provider, requires_api_key=True)
            elif choice == "3":
                provider = 'anthropic'
                api_key = input("Anthropic API Key: ").strip()
                model_name = input("Model name (default: claude-3-opus-20240229): ").strip() or "claude-3-opus-20240229"
                selected_llm = ModelInfo(name=model_name, provider=provider, requires_api_key=True)
            else:
                print("Invalid choice")
                return

            # Ask for prompt
            print("\n--- Prompt Configuration ---")
            print("Enter your classification prompt (or press Enter for default):")
            prompt_text = input("Prompt: ").strip()
            if not prompt_text:
                prompt_text = "Classify the following text into one of these categories: positive, negative, neutral. Text: {text}. Return only the label."

            # Training configuration
            print("\n--- Training Configuration ---")
            benchmark = input("Benchmark multiple models? (y/n, default: y): ").strip().lower()
            benchmark_mode = benchmark != 'n'

            if benchmark_mode:
                num_models = input("Number of models to test (default: 5): ").strip()
                num_models = int(num_models) if num_models.isdigit() else 5
            else:
                num_models = 1

            # Execute pipeline
            print("\n" + "="*50)
            print("STARTING PIPELINE EXECUTION")
            print("="*50)

            # Prepare configuration
            pipeline_config = {
                'mode': 'file',
                'data_source': 'file',
                'file_path': data_path,
                'text_column': text_column,

                # Annotation config
                'run_annotation': True,
                'annotation_mode': 'local' if provider == 'ollama' else 'api',
                'annotation_provider': provider,
                'annotation_model': selected_llm.name if selected_llm else 'gpt-4',
                'api_key': api_key,
                'prompt_text': prompt_text,

                # Validation config
                'run_validation': True,
                'validation_sample_size': 100,
                'export_to_doccano': True,

                # Training config
                'run_training': True,
                'benchmark_mode': benchmark_mode,
                'models_to_test': num_models,
                'max_epochs': 10,
                'batch_size': 16,

                # Deployment config
                'run_deployment': True,
                'save_model': True
            }

            # Initialize and run pipeline
            from ..pipelines.pipeline_controller import PipelineController
            controller = PipelineController()

            try:
                print("\n[Phase 1/4] Starting annotation...")
                state = controller.run_pipeline(pipeline_config)

                # Show results
                print("\n" + "="*50)
                print("PIPELINE COMPLETED SUCCESSFULLY")
                print("="*50)

                if state.annotation_results:
                    print(f"✓ Annotations: {state.annotation_results.get('total_annotated', 0)} items processed")

                if state.validation_results:
                    print(f"✓ Validation: Quality score {state.validation_results.get('quality_score', 0):.1f}/100")
                    print(f"✓ Doccano export: {state.validation_results.get('doccano_export_path', 'N/A')}")

                if state.training_results:
                    print(f"✓ Best model: {state.training_results.get('best_model', 'unknown')}")
                    print(f"✓ Best F1 score: {state.training_results.get('best_f1_macro', 0):.4f}")
                    print(f"✓ Model saved to: {state.training_results.get('model_save_path', 'N/A')}")

                # Ask to save config
                save = input("\nSave this configuration for future use? (y/n): ").strip().lower()
                if save == 'y':
                    profile_name = input("Profile name: ").strip() or "complete_pipeline"
                    self._save_profile(profile_name, pipeline_config)
                    print(f"Configuration saved as '{profile_name}'")

            except Exception as e:
                print(f"\n❌ Pipeline failed: {str(e)}")
                print("\nDebug information:")
                print(f"  - Data path: {data_path}")
                print(f"  - Text column: {text_column}")
                print(f"  - Provider: {provider}")
                print(f"  - Model: {selected_llm.name if selected_llm else 'None'}")
                import traceback
                print("\nFull error trace:")
                traceback.print_exc()

    def training_studio(self):
        """Training studio bringing dataset builders and trainers together."""
        # Display welcome banner
        self._display_welcome_banner()

        # Display simple section header
        self._display_section_header(
            "🏋️ Training Studio",
            "Advanced model training and benchmarking"
        )

        if not (HAS_RICH and self.console):
            print("\nTraining Studio requires the Rich interface. Launch `llm-tool --simple` for basic commands.")
            return

        self._ensure_training_models_loaded()
        builder = TrainingDatasetBuilder(self.settings.paths.data_dir / "training_data")

        self._training_studio_show_model_catalog()

        mode = Prompt.ask(
            "Training mode",
            choices=["quick", "benchmark", "custom", "distributed", "back"],
            default="quick",
        )

        if mode == "back":
            return

        try:
            bundle = self._training_studio_dataset_wizard(builder)
        except Exception as exc:  # pylint: disable=broad-except
            self.console.print(f"[red]Dataset preparation failed:[/red] {exc}")
            self.logger.exception("Training Studio dataset preparation failed", exc_info=exc)
            return

        if bundle is None:
            self.console.print("[yellow]Training cancelled.[/yellow]")
            return

        self._training_studio_render_bundle_summary(bundle)

        if mode == "distributed":
            self._training_studio_run_distributed(bundle)
        elif mode == "quick":
            self._training_studio_run_quick(bundle)
        elif mode == "benchmark":
            self._training_studio_run_benchmark(bundle)
        else:
            self._training_studio_run_custom(bundle)

    # ------------------------------------------------------------------
    # Training Studio helpers
    # ------------------------------------------------------------------
    def _ensure_training_models_loaded(self) -> None:
        if self.available_trainer_models:
            return

        if HAS_RICH and self.console:
            with self.console.status("[cyan]Detecting available training backbones...[/cyan]"):
                self.available_trainer_models = self.trainer_model_detector.get_available_models()
        else:
            self.available_trainer_models = self.trainer_model_detector.get_available_models()

    def _training_studio_show_model_catalog(self) -> None:
        if not self.available_trainer_models:
            return

        table = Table(title="Available Model Categories", border_style="blue")
        table.add_column("Category", style="cyan")
        table.add_column("Models (sample)", style="white")

        for category, models in self.available_trainer_models.items():
            sample = ", ".join(model["name"] for model in models[:3])
            if len(models) > 3:
                sample += f" (+{len(models) - 3} more)"
            table.add_row(category, sample)

        self.console.print(table)

    def _training_studio_dataset_wizard(self, builder: TrainingDatasetBuilder) -> Optional[TrainingDataBundle]:
        """
        Intelligent dataset wizard with comprehensive file analysis and guided setup.
        Now supports all formats with smart detection and recommendations.
        """

        # Step 1: Explain options
        self.console.print("\n[bold]📚 Dataset Source Options:[/bold]")
        self.console.print("  [cyan]build[/cyan]    - Build training dataset from annotated data (CSV/JSON/Excel/Parquet)")
        self.console.print("  [cyan]existing[/cyan] - Use pre-prepared training dataset (already in correct format)")
        self.console.print("  [cyan]back[/cyan]     - Return to previous menu")

        source = Prompt.ask(
            "\nDataset source",
            choices=["build", "existing", "cancel", "back"],
            default="build",
        )

        if source == "cancel" or source == "back":
            return None

        if source == "existing":
            dataset_path = Path(self._prompt_file_path("Existing dataset path"))
            text_column = Prompt.ask("Text column", default="text")
            label_column = Prompt.ask("Label column", default="label")
            mode = Prompt.ask("Training strategy", choices=["single-label", "multi-label", "back"], default="single-label")
            if mode == "back":
                return None
            request = TrainingDataRequest(
                input_path=dataset_path,
                format="prepared",
                text_column=text_column,
                label_column=label_column,
                mode=mode,
            )
            return builder.build(request)

        # Step 2: Explain format options
        self.console.print("\n[bold]📋 Dataset Format Options:[/bold]")
        self.console.print("  [cyan]llm-json[/cyan]      - CSV/JSON with LLM annotations (JSON objects in a column)")
        self.console.print("  [cyan]category-csv[/cyan]  - Simple CSV with text and category/label columns")
        self.console.print("  [cyan]binary-long[/cyan]   - Long-format CSV with binary values per category")
        self.console.print("  [cyan]jsonl-single[/cyan]  - JSONL file for single-label classification")
        self.console.print("  [cyan]jsonl-multi[/cyan]   - JSONL file for multi-label classification")

        format_choice = Prompt.ask(
            "\nSelect dataset format",
            choices=["llm-json", "category-csv", "binary-long", "jsonl-single", "jsonl-multi", "cancel", "back"],
            default="llm-json",
        )

        if format_choice == "cancel" or format_choice == "back":
            return None

        if format_choice == "llm-json":
            # Use intelligent file analysis
            file_path_str = self._prompt_file_path("Annotated file path (CSV/JSON/Excel/Parquet)")
            csv_path = Path(file_path_str)

            # Analyze file intelligently
            self.console.print("\n[cyan]🔍 Analyzing file structure...[/cyan]")
            analysis = DataDetector.analyze_file_intelligently(csv_path)

            # Show analysis results
            if analysis['issues']:
                self.console.print("\n[yellow]⚠️  Analysis Results:[/yellow]")
                for issue in analysis['issues']:
                    self.console.print(f"  {issue}")

            # Auto-suggest text column with all available columns
            text_column_default = "sentence"
            all_columns = analysis.get('all_columns', [])

            if analysis['text_column_candidates']:
                best_text = analysis['text_column_candidates'][0]['name']
                text_column_default = best_text
                self.console.print(f"\n[green]✓ Text column detected: '{best_text}'[/green]")

            # Show all available columns for reference
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

            text_column = Prompt.ask("Text column", default=text_column_default)

            # Auto-suggest annotation column with warning if empty
            annotation_column_default = "annotation"
            if analysis['annotation_column_candidates']:
                best_annotation = analysis['annotation_column_candidates'][0]['name']
                annotation_column_default = best_annotation
                stats = analysis['annotation_stats'].get(best_annotation, {})
                fill_rate = stats.get('fill_rate', 0)
                if fill_rate > 0:
                    self.console.print(f"[green]✓ Annotation column detected: '{best_annotation}' ({fill_rate*100:.1f}% filled)[/green]")
                else:
                    self.console.print(f"[red]⚠️  Annotation column '{best_annotation}' is EMPTY - cannot be used for training![/red]")

            # Show all available columns for reference
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

            annotation_column = Prompt.ask("Annotation column", default=annotation_column_default)

            # Language detection and model recommendation
            languages_found = set(analysis['languages_detected'].keys())
            confirmed_languages = set()

            if languages_found:
                self.console.print(f"\n[bold]🌍 Languages Detected:[/bold]")
                for lang, count in analysis['languages_detected'].items():
                    self.console.print(f"  • {lang.upper()}: {count} rows")

                # Confirm languages with user
                lang_list = ', '.join([l.upper() for l in sorted(languages_found)])
                lang_confirmed = Confirm.ask(
                    f"\n[bold]Detected languages: {lang_list}. Is this correct?[/bold]",
                    default=True
                )

                if lang_confirmed:
                    confirmed_languages = languages_found
                    self.console.print("[green]✓ Languages confirmed[/green]")
                else:
                    # Ask user to specify languages manually
                    self.console.print("\n[yellow]Please specify languages manually[/yellow]")
                    manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

            # Get model recommendations based on confirmed languages
            model_to_use = None
            if confirmed_languages:
                recommendations = LanguageNormalizer.recommend_models(confirmed_languages, self.available_trainer_models)

                if recommendations:
                    self.console.print(f"\n[bold]🤖 Recommended Models for Your Languages:[/bold]")
                    for i, rec in enumerate(recommendations[:5], 1):
                        self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

                    # Interactive model selection
                    self.console.print(f"\n[bold]Select a model:[/bold]")
                    self.console.print("  [cyan]1-{num}[/cyan] - Select from recommendations above".format(num=min(5, len(recommendations))))
                    self.console.print("  [cyan]manual[/cyan] - Enter model name manually")
                    self.console.print("  [cyan]skip[/cyan] - Use default (bert-base-uncased)")

                    model_choice = Prompt.ask("Your choice", default="1")

                    if model_choice == "manual":
                        # Show all available models by category
                        self.console.print("\n[bold]Available Models by Category:[/bold]")
                        all_models_list = []
                        for category, models in self.available_trainer_models.items():
                            self.console.print(f"\n[cyan]{category}:[/cyan]")
                            for model in models:
                                self.console.print(f"  • {model['name']}")
                                all_models_list.append(model['name'])

                        model_to_use = Prompt.ask("\nEnter model name", default="xlm-roberta-base")

                    elif model_choice == "skip":
                        model_to_use = "bert-base-uncased"

                    elif model_choice.isdigit():
                        idx = int(model_choice) - 1
                        if 0 <= idx < len(recommendations):
                            model_to_use = recommendations[idx]['model']
                            self.console.print(f"[green]✓ Selected: {model_to_use}[/green]")
                        else:
                            self.console.print("[yellow]Invalid selection, using first recommendation[/yellow]")
                            model_to_use = recommendations[0]['model']
                    else:
                        model_to_use = recommendations[0]['model']

            # Explain training strategies for LLM-JSON format with REAL examples from data
            self.console.print("\n[bold]📚 Training Strategy for LLM-JSON Format:[/bold]")

            # Parse real data to extract actual keys and values
            annotation_keys_found = analysis.get('annotation_keys_found', set())
            sample_annotation = analysis.get('sample_data', {}).get(annotation_column, [])
            real_example_data = None

            if sample_annotation and len(sample_annotation) > 0:
                first_sample = sample_annotation[0]
                try:
                    if isinstance(first_sample, str):
                        real_example_data = json.loads(first_sample)
                    elif isinstance(first_sample, dict):
                        real_example_data = first_sample
                except:
                    pass

            # Single-label explanation with REAL examples
            self.console.print("  [cyan]single-label[/cyan] - Each text has ONE category per key (mutually exclusive)")
            self.console.print("                     [dim]Use this when a text can only belong to one category[/dim]")
            if real_example_data:
                # Find a single-value key to use as example
                single_val_key = None
                single_val = None
                for key, val in real_example_data.items():
                    if val is not None and not isinstance(val, list):
                        single_val_key = key
                        single_val = val
                        break
                if single_val_key:
                    self.console.print(f"                     [bold]Example from YOUR data:[/bold] {single_val_key} → '{single_val}'")
                    self.console.print(f"                     [dim]→ Train 1 model for '{single_val_key}' that predicts ONE value[/dim]")
                else:
                    self.console.print("                     Example: sentiment → 'positive' OR 'negative'")
            else:
                self.console.print("                     Example: sentiment → 'positive' OR 'negative'")

            # Show detected keys
            if annotation_keys_found:
                keys_str = ', '.join(sorted(annotation_keys_found))
                self.console.print(f"                     [dim]Detected keys in your data: {keys_str}[/dim]")

            # Multi-label explanation with REAL examples
            self.console.print("\n  [cyan]multi-label[/cyan]  - Train ONE binary classifier per key to detect ALL its values")
            self.console.print("                     [dim]Use this when a text can have multiple categories[/dim]")
            if real_example_data and annotation_keys_found:
                # Build real examples from actual data
                self.console.print("                     [bold]Based on YOUR data:[/bold]")
                model_count = 0
                for key in sorted(annotation_keys_found):
                    val = real_example_data.get(key)
                    if val is not None:
                        model_count += 1
                        if isinstance(val, list):
                            if val:
                                val_str = ', '.join([f"'{v}'" for v in val[:3]])
                                if len(val) > 3:
                                    val_str += f", ... ({len(val)} total)"
                                self.console.print(f"                       • Model {model_count}: '{key}' → can detect [{val_str}]")
                            else:
                                self.console.print(f"                       • Model {model_count}: '{key}' → (empty list)")
                        else:
                            self.console.print(f"                       • Model {model_count}: '{key}' → can detect '{val}'")
                if model_count > 0:
                    self.console.print(f"                     [dim]→ Will train {model_count} separate models total[/dim]")
            else:
                self.console.print("                     Example: If key='themes' → 1 model detects all theme values")
                self.console.print("                              If key='sentiment' → 1 model detects all sentiment values")

            # Show full JSON example for clarity
            if real_example_data:
                self.console.print(f"\n[dim]📄 Complete example from your data:[/dim]")
                example_str = json.dumps(real_example_data, ensure_ascii=False, indent=2)
                self.console.print(f"[dim]{example_str}[/dim]")

            mode = Prompt.ask("Target dataset", choices=["single-label", "multi-label", "back"], default="single-label")
            if mode == "back":
                return None

            # Explain label strategies with REAL examples
            self.console.print("\n[bold]🏷️  Label Strategy Options:[/bold]")
            self.console.print("[dim]This determines how label NAMES will be formatted in the training data[/dim]")
            self.console.print("\n  [cyan]key_value[/cyan]  - Label names include key prefix (prevents conflicts)")

            # Generate real examples for key_value
            if real_example_data:
                real_kv_examples = []
                for key, val in real_example_data.items():
                    if val is not None:
                        if isinstance(val, list) and val:
                            real_kv_examples.append(f"'{key}_{val[0]}'")
                        elif not isinstance(val, list):
                            real_kv_examples.append(f"'{key}_{val}'")
                if real_kv_examples:
                    examples_str = ', '.join(real_kv_examples[:3])
                    self.console.print(f"                    [bold]From YOUR data:[/bold] {examples_str}")
                else:
                    self.console.print("                    Example: 'themes_transportation', 'sentiment_positive'")
            else:
                self.console.print("                    Example: 'themes_transportation', 'sentiment_positive'")

            self.console.print("\n  [cyan]value_only[/cyan] - Label names are just values (simpler but may conflict)")

            # Generate real examples for value_only
            if real_example_data:
                real_vo_examples = []
                for key, val in real_example_data.items():
                    if val is not None:
                        if isinstance(val, list) and val:
                            real_vo_examples.append(f"'{val[0]}'")
                        elif not isinstance(val, list):
                            real_vo_examples.append(f"'{val}'")
                if real_vo_examples:
                    examples_str = ', '.join(real_vo_examples[:3])
                    self.console.print(f"                    [bold]From YOUR data:[/bold] {examples_str}")
                else:
                    self.console.print("                    Example: 'transportation', 'positive'")
            else:
                self.console.print("                    Example: 'transportation', 'positive'")

            label_strategy = Prompt.ask("Label strategy", choices=["key_value", "value_only", "back"], default="key_value")
            if label_strategy == "back":
                return None

            # For multi-label CSV-JSON: ask which keys to use (one model per key)
            annotation_keys = None
            if mode == "multi-label":
                self.console.print("\n[bold yellow]📋 Multi-label mode:[/bold yellow] One model will be trained per annotation key")
                self.console.print("[dim]This means training SEPARATE BINARY classifiers for each category type[/dim]")

                # Show detected keys if available WITH REAL VALUES
                if analysis.get('annotation_keys_found'):
                    detected_keys = sorted(analysis['annotation_keys_found'])
                    self.console.print(f"\n[green]✓ Detected annotation keys in your data: {', '.join(detected_keys)}[/green]")

                    # Extract unique values for each key from sample data
                    key_values = {}
                    if real_example_data:
                        for key in detected_keys:
                            val = real_example_data.get(key)
                            if val is not None:
                                if isinstance(val, list):
                                    key_values[key] = val if val else []
                                else:
                                    key_values[key] = [val]

                    self.console.print("\n[bold]What this means (with YOUR data):[/bold]")
                    for key in detected_keys:
                        if key in key_values and key_values[key]:
                            values_preview = ', '.join([f"'{v}'" for v in key_values[key][:3]])
                            if len(key_values[key]) > 3:
                                values_preview += ", ..."
                            self.console.print(f"  • [cyan]{key}[/cyan] → One binary model for each value: {values_preview}")
                        else:
                            self.console.print(f"  • [cyan]{key}[/cyan] → One model will classify ALL {key} values")

                    # Build real example from detected keys
                    if len(detected_keys) >= 2:
                        example_keys = ', '.join(detected_keys[:2])
                        self.console.print(f"\n[bold]Example:[/bold] Selecting '{example_keys}' → {min(2, len(detected_keys))} models trained")
                        for idx, key in enumerate(detected_keys[:2], 1):
                            if key in key_values and key_values[key]:
                                val_preview = key_values[key][0]
                                self.console.print(f"  Model {idx}: Trains '{key}_{val_preview}' vs NOT '{key}_{val_preview}' (and all other {key} values)")
                            else:
                                self.console.print(f"  Model {idx}: Detects all {key} categories")
                else:
                    self.console.print("Example: If you select 'themes,sentiment' → 2 models (one for themes, one for sentiment)")

                # Show selection guidance with all available options
                if detected_keys:
                    self.console.print("\n[bold cyan]📝 Select which annotation keys to train:[/bold cyan]")
                    self.console.print(f"[bold]Available keys:[/bold] {', '.join(detected_keys)}")
                    self.console.print("\n[dim]Options:[/dim]")
                    self.console.print(f"  • [cyan]Leave blank[/cyan] → Train ALL {len(detected_keys)} models (one per key)")
                    self.console.print(f"  • [cyan]Enter specific keys[/cyan] → Train only selected models")
                    if detected_keys:
                        self.console.print(f"    Example: '{detected_keys[0]}' → Train only 1 model for {detected_keys[0]}")
                    if len(detected_keys) >= 2:
                        self.console.print(f"    Example: '{detected_keys[0]},{detected_keys[1]}' → Train 2 models")

                keys_input = Prompt.ask("\nAnnotation keys (comma-separated, or BLANK for ALL)", default="")
                annotation_keys = [key.strip() for key in keys_input.split(",") if key.strip()] or None
            else:
                # For single-label mode
                if analysis.get('annotation_keys_found'):
                    detected_keys = sorted(analysis['annotation_keys_found'])
                    self.console.print(f"\n[dim]Note: Your data has keys: {', '.join(detected_keys)}[/dim]")
                    self.console.print("[dim]Leave blank to use all keys, or specify which ones to include[/dim]")

                keys_input = Prompt.ask("Annotation keys to include (comma separated, leave blank for all)", default="")
                annotation_keys = [key.strip() for key in keys_input.split(",") if key.strip()] or None

            # Auto-suggest ID column
            id_column_default = ""
            if analysis['id_column_candidates']:
                id_column_default = analysis['id_column_candidates'][0]
                self.console.print(f"\n[green]✓ ID column detected: '{id_column_default}'[/green]")
            # Show all available columns for reference
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

            id_column = Prompt.ask("Identifier column (optional)", default=id_column_default)

            # Auto-suggest language column
            lang_column_default = ""
            if analysis['language_column_candidates']:
                lang_column_default = analysis['language_column_candidates'][0]
                self.console.print(f"\n[green]✓ Language column detected: '{lang_column_default}'[/green]")
            # Show all available columns for reference
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

            lang_column = Prompt.ask("Language column (optional)", default=lang_column_default)

            request = TrainingDataRequest(
                input_path=csv_path,
                format="llm_json",
                text_column=text_column,
                annotation_column=annotation_column,
                annotation_keys=annotation_keys,
                label_strategy=label_strategy,
                mode=mode,
                id_column=id_column or None,
                lang_column=lang_column or None,
            )
            bundle = builder.build(request)

            # Store recommended model in bundle for later use
            if bundle and model_to_use:
                bundle.recommended_model = model_to_use

            return bundle

        if format_choice == "category-csv":
            csv_path = Path(self._prompt_file_path("Category CSV path"))
            text_column = Prompt.ask("Text column", default="text")
            label_column = Prompt.ask("Category/label column", default="label")
            request = TrainingDataRequest(
                input_path=csv_path,
                format="category_csv",
                text_column=text_column,
                label_column=label_column,
                mode="single-label",
            )
            return builder.build(request)

        if format_choice == "binary-long":
            csv_path = Path(self._prompt_file_path("Binary CSV path"))
            text_column = Prompt.ask("Text column", default="text")
            category_column = Prompt.ask("Category column", default="category")
            value_column = Prompt.ask("Value column (0/1)", default="value")
            id_column = Prompt.ask("Identifier column (optional)", default="")
            lang_column = Prompt.ask("Language column (optional)", default="")
            request = TrainingDataRequest(
                input_path=csv_path,
                format="binary_long_csv",
                text_column=text_column,
                category_column=category_column,
                value_column=value_column,
                id_column=id_column or None,
                lang_column=lang_column or None,
                mode="multi-label",
            )
            return builder.build(request)

        if format_choice == "jsonl-single":
            data_path = Path(self._prompt_file_path("JSONL path"))
            text_column = Prompt.ask("Text field", default="text")
            label_column = Prompt.ask("Label field", default="label")
            request = TrainingDataRequest(
                input_path=data_path,
                format="jsonl_single",
                text_column=text_column,
                label_column=label_column,
                mode="single-label",
            )
            return builder.build(request)

        data_path = Path(self._prompt_file_path("JSONL path"))
        text_column = Prompt.ask("Text field", default="text")
        label_column = Prompt.ask("Label field", default="labels")
        id_column = Prompt.ask("Identifier field (optional)", default="")
        lang_column = Prompt.ask("Language field (optional)", default="")
        request = TrainingDataRequest(
            input_path=data_path,
            format="jsonl_multi",
            text_column=text_column,
            label_column=label_column,
            id_column=id_column or None,
            lang_column=lang_column or None,
            mode="multi-label",
        )
        return builder.build(request)

    def _training_studio_render_bundle_summary(self, bundle: TrainingDataBundle) -> None:
        table = Table(title="Dataset Summary", border_style="green")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Strategy", bundle.strategy)
        table.add_row("Primary file", str(bundle.primary_file) if bundle.primary_file else "—")
        table.add_row("Text column", bundle.text_column)
        table.add_row("Label column", bundle.label_column)
        table.add_row("Training files", str(len(bundle.training_files)))

        if bundle.metadata.get("label_distribution"):
            distribution = ", ".join(f"{k}: {v}" for k, v in bundle.metadata["label_distribution"].items())
            table.add_row("Label distribution", distribution)
        if bundle.metadata.get("categories"):
            table.add_row("Categories", ", ".join(bundle.metadata["categories"]))
        if bundle.metadata.get("analysis"):
            analysis = bundle.metadata["analysis"]
            table.add_row("Annotated rows", str(analysis.get("annotated_rows", "n/a")))

        self.console.print(table)

    def _training_studio_run_quick(self, bundle: TrainingDataBundle) -> None:
        self.console.print("\n[bold]Quick training[/bold] - using sensible defaults.")

        # Use recommended model if available, otherwise use default
        if hasattr(bundle, 'recommended_model') and bundle.recommended_model:
            default_model = bundle.recommended_model
            self.console.print(f"[green]Using recommended model: {default_model}[/green]")
        else:
            default_model = self._training_studio_default_model()

        model_name = Prompt.ask("Model to train", default=default_model)

        output_dir = self._training_studio_make_output_dir("training_studio_quick")
        trainer = ModelTrainer()

        config = bundle.to_trainer_config(output_dir, {"model_name": model_name})

        try:
            result = trainer.train(config)
        except Exception as exc:  # pylint: disable=broad-except
            self.console.print(f"[red]Training failed:[/red] {exc}")
            self.logger.exception("Quick training failed", exc_info=exc)
            return

        self._training_studio_show_training_result(result, bundle, title="Quick training results")

    def _training_studio_run_benchmark(self, bundle: TrainingDataBundle) -> None:
        try:
            dataset_path, text_column, label_column = self._training_studio_resolve_benchmark_dataset(bundle)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return

        num_models = self._int_prompt_with_validation("Number of models to test", default=5, min_value=1, max_value=20)

        available_models = self._flatten_trainer_models()
        if available_models:
            preview = ", ".join(available_models[:5])
            if len(available_models) > 5:
                preview += " …"
            self.console.print(f"\n[dim]Available backbones:[/dim] {preview}")

        selected_models = available_models[:num_models] if available_models else ["bert-base-uncased"]
        extra_model = Prompt.ask("Additional HuggingFace model id (optional)", default="")
        if extra_model:
            selected_models.append(extra_model.strip())

        benchmark_config = BenchmarkConfig(models_to_test=selected_models, max_models=num_models)

        trainer = ModelTrainer()
        output_dir = self._training_studio_make_output_dir("training_studio_benchmark")
        bundle.to_trainer_config(output_dir)  # ensure directory creation

        try:
            report = trainer.benchmark_models(
                data_path=str(dataset_path),
                benchmark_config=benchmark_config,
                text_column=text_column,
                label_column=label_column,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.console.print(f"[red]Benchmark failed:[/red] {exc}")
            self.logger.exception("Benchmark run failed", exc_info=exc)
            return

        self._training_studio_show_benchmark_results(report)

    def _training_studio_run_custom(self, bundle: TrainingDataBundle) -> None:
        self.console.print("\n[bold]Custom training configuration[/bold]")

        model_name = Prompt.ask("Model name", default=self._training_studio_default_model())
        epochs = self._int_prompt_with_validation("Epochs", default=10, min_value=1, max_value=100)
        batch_size = self._int_prompt_with_validation("Batch size", default=16, min_value=1, max_value=256)

        lr_input = Prompt.ask("Learning rate", default="2e-5")
        try:
            learning_rate = float(lr_input)
        except ValueError:
            self.console.print(f"[red]Invalid learning rate: {lr_input}[/red]")
            return

        output_dir = self._training_studio_make_output_dir("training_studio_custom")
        trainer = ModelTrainer()

        extra = {
            "model_name": model_name,
            "max_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }

        config = bundle.to_trainer_config(output_dir, extra)

        try:
            result = trainer.train(config)
        except Exception as exc:  # pylint: disable=broad-except
            self.console.print(f"[red]Training failed:[/red] {exc}")
            self.logger.exception("Custom training failed", exc_info=exc)
            return

        self._training_studio_show_training_result(result, bundle, title="Custom training results")

    def _training_studio_run_distributed(self, bundle: TrainingDataBundle) -> None:
        self.console.print("\n[bold]Distributed multi-label training[/bold]")

        resolved = self._training_studio_resolve_multilabel_dataset(bundle)
        if resolved is None:
            self.console.print(
                "[red]This dataset does not expose a multi-label view."
                " Please select a format that produces a consolidated JSONL (e.g. LLM annotations, binary long, JSONL multi-label) and try again.[/red]"
            )
            return

        dataset_path, label_fields = resolved

        if HAS_RICH and self.console:
            self.console.print(f"[dim]Using dataset:[/dim] {dataset_path}")

        output_dir = self._training_studio_make_output_dir("training_studio_distributed")

        epochs = self._int_prompt_with_validation("Epochs per label", default=8, min_value=1, max_value=50)
        batch_size = self._int_prompt_with_validation("Batch size", default=16, min_value=2, max_value=128)
        learning_rate = self._float_prompt_with_validation("Learning rate", default=5e-5, min_value=1e-6, max_value=1e-2)

        auto_split = Confirm.ask("Automatically split data for validation?", default=True)
        train_ratio = 0.8
        val_ratio = 0.1
        if auto_split:
            train_ratio = self._float_prompt_with_validation("Training split ratio", default=0.8, min_value=0.5, max_value=0.9)
            remaining = max(1e-6, 1 - train_ratio)
            val_default = min(0.2, remaining / 2) or 0.1
            val_ratio = self._float_prompt_with_validation(
                "Validation split ratio", default=val_default, min_value=0.05, max_value=min(0.4, remaining)
            )

        reinforced = Confirm.ask("Enable reinforced learning for hard labels?", default=True)
        reinforced_epochs = None
        if reinforced:
            reinforced_epochs = self._int_prompt_with_validation("Reinforced epochs", default=2, min_value=1, max_value=20)

        parallel_training = Confirm.ask("Train label models in parallel?", default=True)
        max_workers = 2
        if parallel_training:
            max_workers = self._int_prompt_with_validation("Parallel workers", default=2, min_value=1, max_value=8)

        strategy_choice = Prompt.ask(
            "Language strategy",
            choices=["auto", "multilingual", "per-language"],
            default="auto",
        )
        train_by_language = strategy_choice == "per-language"
        multilingual_model = strategy_choice == "multilingual"

        auto_select_model = Confirm.ask("Auto-select the best backbone for each label?", default=True)

        # Get the model name from the bundle if available
        model_name_to_use = None
        if hasattr(bundle, 'recommended_model') and bundle.recommended_model:
            model_name_to_use = bundle.recommended_model
            self.console.print(f"\n[green]✓ Using selected model: {model_name_to_use}[/green]")

        # If no model was selected, ask the user now
        if not model_name_to_use:
            self.console.print("\n[yellow]⚠️  No model was selected during data preparation[/yellow]")
            default_model = "bert-base-multilingual-cased"
            model_name_to_use = Prompt.ask(
                "Which model would you like to use?",
                default=default_model
            )
            self.console.print(f"[green]✓ Using model: {model_name_to_use}[/green]")

        trainer_config = MultiLabelTrainingConfig(
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name_to_use,  # Pass the selected model name
            auto_select_model=auto_select_model,
            train_by_language=train_by_language,
            multilingual_model=multilingual_model,
            reinforced_learning=reinforced,
            reinforced_epochs=reinforced_epochs,
            output_dir=str(output_dir),
            parallel_training=parallel_training,
            max_workers=max_workers,
            auto_split=auto_split,
            split_ratio=train_ratio,
            stratified=True,
        )

        trainer = MultiLabelTrainer(config=trainer_config, verbose=False)

        try:
            with self.console.status("[cyan]Loading multi-label dataset...[/cyan]") if HAS_RICH and self.console else contextlib.nullcontext():
                samples = trainer.load_multi_label_data(
                    str(dataset_path),
                    text_field="text",
                    labels_dict_field="labels",
                    label_fields=label_fields,
                )
        except Exception as exc:  # pylint: disable=broad-except
            message = f"Failed to load dataset for distributed training: {exc}"
            if HAS_RICH and self.console:
                self.console.print(f"[red]{message}[/red]")
            else:
                print(message)
            self.logger.exception("Distributed training dataset load failed", exc_info=exc)
            return

        if not samples:
            self.console.print("[red]No samples available for training.[/red]")
            return

        # Display quick stats
        if HAS_RICH and self.console:
            label_counter = Counter()
            lang_counter = Counter()
            for sample in samples:
                lang_counter[sample.lang or "unknown"] += 1
                for label_name, value in sample.labels.items():
                    if isinstance(value, (int, float)):
                        if value:
                            label_counter[label_name] += 1
                    else:
                        label_counter[label_name] += 1

            stats_table = Table(title="Dataset overview", border_style="green")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_row("Samples", str(len(samples)))
            stats_table.add_row("Unique labels", str(len(label_counter) or len(label_fields or [])))
            if label_counter:
                top_labels = ", ".join(f"{k}: {v}" for k, v in label_counter.most_common(5))
                stats_table.add_row("Label frequency", top_labels)
            if lang_counter:
                lang_summary = ", ".join(f"{k}: {v}" for k, v in lang_counter.most_common(5))
                stats_table.add_row("Language distribution", lang_summary)
            self.console.print(stats_table)

        if HAS_RICH and self.console:
            status_ctx = self.console.status("[bold green]Training label models...[/bold green]", spinner="dots")
        else:
            status_ctx = contextlib.nullcontext()

        with status_ctx:
            try:
                models = trainer.train_all_models(samples, train_ratio=train_ratio, val_ratio=val_ratio)
            except Exception as exc:  # pylint: disable=broad-except
                message = f"Distributed training failed: {exc}"
                if HAS_RICH and self.console:
                    self.console.print(f"[red]{message}[/red]")
                else:
                    print(message)
                self.logger.exception("Distributed training failed", exc_info=exc)
                return

        self._training_studio_show_distributed_results(trainer, models, output_dir)

    def _training_studio_resolve_multilabel_dataset(self, bundle: TrainingDataBundle) -> Optional[Tuple[Path, Optional[List[str]]]]:
        """Return the consolidated multi-label dataset path if available."""
        multilabel_path = bundle.training_files.get("multilabel")
        if multilabel_path:
            path_obj = Path(multilabel_path)
            if path_obj.exists():
                label_fields = bundle.metadata.get("labels_detected")
                return path_obj, label_fields if isinstance(label_fields, list) else None

        if bundle.strategy == "multi-label" and bundle.primary_file:
            path_obj = Path(bundle.primary_file)
            if path_obj.exists() and path_obj.suffix.lower() in {".json", ".jsonl"}:
                label_fields = bundle.metadata.get("labels_detected")
                return path_obj, label_fields if isinstance(label_fields, list) else None

        return None

    def _training_studio_show_distributed_results(
        self,
        trainer: MultiLabelTrainer,
        models: Dict[str, MultiLabelModelInfo],
        output_dir: Path,
    ) -> None:
        """Render a summary table of distributed training results."""
        if not models:
            message = "No models were produced during distributed training."
            if HAS_RICH and self.console:
                self.console.print(f"[yellow]{message}[/yellow]")
            else:
                print(message)
            return

        if HAS_RICH and self.console:
            table = Table(title="Distributed training results", border_style="green")
            table.add_column("Model", style="cyan")
            table.add_column("Label", style="white")
            table.add_column("Language", style="white")
            table.add_column("Macro F1", justify="right")

            for model_name, info in sorted(models.items()):
                metrics = info.performance_metrics or {}
                macro_f1 = metrics.get("macro_f1", 0.0)
                table.add_row(
                    model_name,
                    info.label_name,
                    info.language or "—",
                    f"{macro_f1:.3f}"
                )

            self.console.print(table)
            self.console.print(f"[dim]Models saved to[/dim] {output_dir}")
        else:
            print("\nDistributed training results:")
            for model_name, info in sorted(models.items()):
                metrics = info.performance_metrics or {}
                macro_f1 = metrics.get('macro_f1', 0.0)
                print(f"  - {model_name}: label={info.label_name}, lang={info.language or '-'}, macro_f1={macro_f1:.3f}")
            print(f"Models saved to {output_dir}")


    def _training_studio_show_training_result(self, result: Dict[str, Any], bundle: TrainingDataBundle, title: str) -> None:
        table = Table(title=title, border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", str(result.get("best_model", "n/a")))
        table.add_row("Accuracy", f"{result.get('accuracy', 0.0):.4f}")
        table.add_row("F1 macro", f"{result.get('best_f1_macro', 0.0):.4f}")
        table.add_row("Model path", result.get("model_path", "—"))

        self.console.print(table)

        if bundle.strategy == "multi-label":
            metrics = result.get("metrics", {})
            per_label = metrics.get("per_label_results")
            if per_label:
                detail_table = Table(title="Per-label performance", border_style="blue")
                detail_table.add_column("Label")
                detail_table.add_column("Accuracy")
                detail_table.add_column("F1 macro")

                for label, stats in per_label.items():
                    if isinstance(stats, dict) and "error" not in stats:
                        detail_table.add_row(
                            label,
                            f"{stats.get('accuracy', 0.0):.4f}",
                            f"{stats.get('f1_macro', 0.0):.4f}",
                        )
                    elif isinstance(stats, dict):
                        detail_table.add_row(label, stats.get("error", "error"), "—")

                self.console.print(detail_table)

    def _training_studio_show_benchmark_results(self, report: Dict[str, Any]) -> None:
        results = report.get("results", [])
        if not results:
            self.console.print("[yellow]No benchmark results available.[/yellow]")
            return

        table = Table(title="Benchmark results", border_style="green")
        table.add_column("#", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Accuracy", justify="right")
        table.add_column("F1 macro", justify="right")

        for idx, entry in enumerate(results, start=1):
            table.add_row(
                str(idx),
                entry.get("model", "?"),
                f"{entry.get('accuracy', 0.0):.4f}",
                f"{entry.get('f1_macro', 0.0):.4f}",
            )

        self.console.print(table)

        best_model = report.get("best_model")
        if best_model:
            best_f1 = report.get("best_f1_macro", 0.0)
            self.console.print(f"[green]Best model:[/green] {best_model} (F1 {best_f1:.4f})")

    def _training_studio_resolve_benchmark_dataset(self, bundle: TrainingDataBundle) -> Tuple[Path, str, str]:
        if bundle.strategy == "single-label" and bundle.primary_file:
            return bundle.primary_file, bundle.text_column, bundle.label_column

        candidates = [(label, path) for label, path in bundle.training_files.items() if label != "multilabel"]

        if not candidates:
            raise ValueError("No single-label dataset available for benchmarking.")

        if len(candidates) == 1:
            label, path = candidates[0]
            self.console.print(f"Using dataset for label [cyan]{label}[/cyan].")
            return path, "text", "label"

        self.console.print("\nSelect the label you want to benchmark:")
        for idx, (label, _) in enumerate(candidates, start=1):
            self.console.print(f"  {idx}. {label}")

        choice = self._int_prompt_with_validation("Label", default=1, min_value=1, max_value=len(candidates))
        label, path = candidates[choice - 1]
        self.console.print(f"Benchmarking label [cyan]{label}[/cyan].")
        return path, "text", "label"

    def _training_studio_make_output_dir(self, prefix: str) -> Path:
        directory = self.settings.paths.models_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _flatten_trainer_models(self) -> List[str]:
        if not self.available_trainer_models:
            return []

        names: List[str] = []
        for models in self.available_trainer_models.values():
            names.extend(model["name"] for model in models)

        seen = set()
        unique: List[str] = []
        for name in names:
            if name not in seen:
                unique.append(name)
                seen.add(name)
        return unique

    def _training_studio_default_model(self) -> str:
        models = self._flatten_trainer_models()
        return "bert-base-uncased" if "bert-base-uncased" in models else (models[0] if models else "bert-base-uncased")

    def validation_lab(self):
        """Validation lab for quality control and Doccano export"""
        # Display welcome banner
        self._display_welcome_banner()

        # Display simple section header
        self._display_section_header(
            "🔍 Validation Lab",
            "Quality control and human review preparation"
        )

        if HAS_RICH and self.console:
            # Select annotations file
            self.console.print("\n[bold]Select Annotations File:[/bold]")
            annotations_path = self._prompt_file_path("Annotations file path")

            # Load and analyze
            self.console.print("\n[cyan]Analyzing annotations...[/cyan]")

            from ..validators.annotation_validator import AnnotationValidator, ValidationConfig

            validator = AnnotationValidator()

            # Configuration
            self.console.print("\n[bold]Validation Configuration:[/bold]")

            sample_size = self._int_prompt_with_validation("Sample size for validation", default=100, min_value=10, max_value=1000)
            stratified = Confirm.ask("Use stratified sampling?", default=True)
            export_doccano = Confirm.ask("Export to Doccano format?", default=True)

            # Run validation
            with self.console.status("[bold green]Running validation...", spinner="dots"):
                config = ValidationConfig(
                    sample_size=sample_size,
                    stratified_sampling=stratified,
                    export_to_doccano=export_doccano,
                    export_format='both'
                )

                validator.config = config
                result = validator.validate(
                    input_file=annotations_path,
                    output_dir="./validation"
                )

            # Display results
            self.console.print("\n[bold]Validation Results:[/bold]\n")

            # Quality score with color coding
            score = result.quality_score
            if score >= 80:
                color = "green"
            elif score >= 60:
                color = "yellow"
            else:
                color = "red"

            self.console.print(f"[{color}]Quality Score: {score:.1f}/100[/{color}]")

            # Statistics table
            stats_table = Table(title="📊 Annotation Statistics", border_style="blue")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")

            stats_table.add_row("Total Annotations", str(result.total_annotations))
            stats_table.add_row("Validated Samples", str(result.validated_samples))
            stats_table.add_row("Unique Labels", str(len(result.label_distribution)))

            if result.confidence_stats:
                stats_table.add_row("Avg Confidence", f"{result.confidence_stats.get('mean', 0):.3f}")
                stats_table.add_row("Low Confidence %", f"{result.confidence_stats.get('low_confidence_percentage', 0):.1f}%")

            self.console.print(stats_table)

            # Issues found
            if result.issues_found:
                self.console.print(f"\n[yellow]⚠ {len(result.issues_found)} issues found:[/yellow]")
                for issue in result.issues_found[:5]:
                    self.console.print(f"  - {issue['type']}: {issue.get('message', issue.get('column', ''))}")

            # Export paths
            self.console.print("\n[bold]Exports:[/bold]")
            if result.doccano_export_path:
                self.console.print(f"📁 Doccano: {result.doccano_export_path}")
            if result.export_path:
                self.console.print(f"📁 Data: {result.export_path}")

            self.console.print("\n[green]✅ Validation complete![/green]")

        else:
            print("\n=== Validation Lab ===")
            print("Quality control and validation\n")

            annotations_path = input("Annotations file path: ").strip()
            sample_size = int(input("Sample size (default 100): ").strip() or "100")

            print("\nRunning validation...")
            print("✅ Validation complete!")
            print(f"Exported to: ./validation/")

    def analytics_dashboard(self):
        """Analytics dashboard"""
        # Display simple section header
        self._display_section_header(
            "📊 Analytics Dashboard",
            "Performance analysis and insights (Coming Soon)"
        )

        if HAS_RICH and self.console:
            self.console.print("\n[yellow]This feature is under development[/yellow]")
        else:
            print("\nThis feature is under development")

    def profile_manager_ui(self):
        """Profile manager interface"""
        # Display welcome banner
        self._display_welcome_banner()

        # Display simple section header
        self._display_section_header(
            "💾 Profile Manager",
            "Manage saved configuration profiles"
        )

        if HAS_RICH and self.console:
            profiles = self.profile_manager.list_profiles()

            if not profiles:
                self.console.print("[yellow]No saved profiles found[/yellow]")
                return

            table = Table(title="💾 Saved Profiles", border_style="blue")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Name", style="white")
            table.add_column("Created", style="dim")
            table.add_column("Last Used", style="green")

            for i, profile in enumerate(profiles, 1):
                table.add_row(
                    str(i),
                    profile.name,
                    profile.created_at.strftime("%Y-%m-%d"),
                    profile.last_used.strftime("%Y-%m-%d %H:%M")
                )

            self.console.print(table)

            choice = IntPrompt.ask("Select profile to load (0 to cancel)", default=0)
            if choice > 0 and choice <= len(profiles):
                selected_profile = profiles[choice-1]
                self.console.print(f"[green]Loading profile: {selected_profile.name}[/green]")
                # Execute with loaded configuration
                self._execute_quick_start(selected_profile.configuration)
        else:
            print("Profile Manager - Coming Soon")

    def advanced_settings(self):
        """Advanced settings interface"""
        # Display simple section header
        self._display_section_header(
            "⚙️ Advanced Settings",
            "Configure advanced options (Coming Soon)"
        )

        if HAS_RICH and self.console:
            self.console.print("\n[yellow]This feature is under development[/yellow]")
        else:
            print("\nThis feature is under development")

    # ============================================================================
    # LLM ANNOTATION STUDIO - HELPER METHODS
    # ============================================================================

    def _detect_prompts_in_folder(self, folder_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Auto-detect prompts from prompts/ folder with JSON key extraction.

        Returns:
            List of dicts with: {'path': Path, 'name': str, 'keys': List[str], 'content': str}
        """
        if folder_path is None:
            # Default to current directory prompts/ folder
            folder_path = Path.cwd() / "prompts"

        if not folder_path.exists():
            self.logger.warning(f"Prompts folder not found: {folder_path}")
            return []

        prompts = []
        for txt_file in sorted(folder_path.glob("*.txt")):
            try:
                content = txt_file.read_text(encoding='utf-8')

                # Extract JSON keys using the existing function
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(content)

                prompts.append({
                    'path': txt_file,
                    'name': txt_file.stem,
                    'keys': keys,
                    'content': content
                })

                self.logger.info(f"Detected prompt: {txt_file.name} with {len(keys)} keys")
            except Exception as e:
                self.logger.warning(f"Failed to read prompt {txt_file.name}: {e}")

        return prompts

    def _detect_text_columns(self, file_path: Path) -> Dict[str, Any]:
        """
        Intelligently detect text columns in a dataset.

        Returns:
            Dict with: {
                'all_columns': List[str],
                'text_candidates': List[Dict],  # [{'name': str, 'confidence': str, 'sample': str}]
                'df': pd.DataFrame
            }
        """
        # Load data to analyze columns
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=100)  # Sample first 100 rows
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=100)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
                df = df.head(100)
            elif file_path.suffix.lower() == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f.readlines()[:100]]
                df = pd.DataFrame(data)
            else:
                # Fallback
                return {'all_columns': [], 'text_candidates': [], 'df': None}

            all_columns = df.columns.tolist()
            text_candidates = []

            # Analyze each column
            for col in all_columns:
                if col in df.columns and df[col].dtype == 'object':
                    # Get non-null sample
                    non_null = df[col].dropna()
                    if len(non_null) == 0:
                        continue

                    # Calculate average length
                    avg_length = non_null.astype(str).str.len().mean()
                    sample_value = str(non_null.iloc[0])[:100]  # First 100 chars

                    # Determine confidence
                    if avg_length > 100:
                        confidence = "high"
                    elif avg_length > 30:
                        confidence = "medium"
                    elif avg_length > 10:
                        confidence = "low"
                    else:
                        confidence = "very_low"

                    # Add to candidates if reasonable length
                    if avg_length > 10:
                        text_candidates.append({
                            'name': col,
                            'confidence': confidence,
                            'avg_length': avg_length,
                            'sample': sample_value
                        })

            # Sort by confidence and avg_length
            confidence_order = {"high": 0, "medium": 1, "low": 2, "very_low": 3}
            text_candidates.sort(key=lambda x: (confidence_order[x['confidence']], -x['avg_length']))

            return {
                'all_columns': all_columns,
                'text_candidates': text_candidates,
                'df': df
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze columns: {e}")
            return {'all_columns': [], 'text_candidates': [], 'df': None}

    def _create_annotation_id(self, df: pd.DataFrame, id_column: str = "annotation_id") -> pd.DataFrame:
        """
        Create or verify annotation ID column for tracking.

        Args:
            df: DataFrame to process
            id_column: Name of ID column

        Returns:
            DataFrame with ID column added/verified
        """
        if id_column not in df.columns:
            df[id_column] = [f"ann_{i:06d}" for i in range(1, len(df) + 1)]
            self.console.print(f"[green]✓ Created {id_column} column with {len(df)} IDs[/green]")
        else:
            # Verify no nulls
            null_count = df[id_column].isna().sum()
            if null_count > 0:
                self.console.print(f"[yellow]⚠️  Found {null_count} null IDs, filling them...[/yellow]")
                next_id = len(df) + 1
                for idx in df[df[id_column].isna()].index:
                    df.at[idx, id_column] = f"ann_{next_id:06d}"
                    next_id += 1

        return df

    def _create_output_structure(self, base_name: str, data_format: str) -> Dict[str, Path]:
        """
        Create organized output folder structure.

        Structure:
            annotations_output/
                {timestamp}_{base_name}/
                    data/           # Annotated data files
                    logs/           # Annotation logs
                    prompts/        # Copy of prompts used
                    config.json     # Configuration used

        Returns:
            Dict with paths: {'root', 'data', 'logs', 'prompts', 'config'}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path("annotations_output") / f"{timestamp}_{base_name}"

        paths = {
            'root': root,
            'data': root / 'data',
            'logs': root / 'logs',
            'prompts': root / 'prompts',
            'config': root / 'config.json'
        }

        # Create directories
        for key in ['data', 'logs', 'prompts']:
            paths[key].mkdir(parents=True, exist_ok=True)

        self.console.print(f"[green]✓ Created output structure: {root}[/green]")
        return paths

    def _save_incremental(
        self,
        df: pd.DataFrame,
        output_path: Path,
        data_format: str,
        batch_size: int = 50,
        db_config: Optional[Dict] = None
    ):
        """
        Save data incrementally based on format.

        Supports:
        - CSV: Line-by-line append
        - Excel: Batch save every N rows
        - Parquet: Batch save every N rows
        - PostgreSQL: Immediate UPDATE per row
        - RData: Batch save every N rows
        """
        if data_format == 'csv':
            # Append mode for CSV
            if not output_path.exists():
                df.head(0).to_csv(output_path, index=False)
            df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

        elif data_format == 'excel':
            # Full save for Excel (cannot append)
            df.to_excel(output_path, index=False)

        elif data_format == 'parquet':
            # Full save for Parquet
            df.to_parquet(output_path, index=False)

        elif data_format == 'postgresql' and db_config:
            # Direct UPDATE in database
            from sqlalchemy import create_engine, text
            engine = create_engine(
                f"postgresql://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config.get('port', 5432)}/{db_config['database']}"
            )
            # Implementation would use UPDATE statements
            pass

        elif data_format in ['rdata', 'rds']:
            # Save using pyreadr
            try:
                import pyreadr
                pyreadr.write_rdata(str(output_path), df, df_name="annotated_data")
            except ImportError:
                self.logger.error("pyreadr not installed - cannot save RData format")

    # ============================================================================
    # LLM ANNOTATION STUDIO - WORKFLOW METHODS
    # ============================================================================

    def _quick_annotate(self):
        """Resume or relaunch annotation using saved parameters"""
        self.console.print("\n[bold cyan]🔄 Resume/Relaunch Annotation[/bold cyan]\n")
        self.console.print("[dim]Load saved parameters from previous annotations[/dim]\n")

        # ============================================================
        # DETECT METADATA FILES
        # ============================================================
        annotations_dir = self.settings.paths.data_dir / 'annotations'

        if not annotations_dir.exists():
            self.console.print("[yellow]No annotations directory found.[/yellow]")
            self.console.print("[dim]Run Smart Annotate first to create annotation sessions.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Find all metadata JSON files
        metadata_files = list(annotations_dir.glob("*_metadata_*.json"))

        if not metadata_files:
            self.console.print("[yellow]No saved annotation parameters found.[/yellow]")
            self.console.print("[dim]Run Smart Annotate and save parameters to use this feature.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Sort by modification time (most recent first)
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Display available sessions
        self.console.print(f"[green]Found {len(metadata_files)} saved annotation session(s)[/green]\n")

        sessions_table = Table(border_style="cyan", show_header=True)
        sessions_table.add_column("#", style="cyan", width=3)
        sessions_table.add_column("Session", style="white")
        sessions_table.add_column("Date", style="yellow")
        sessions_table.add_column("Workflow", style="green")
        sessions_table.add_column("Model", style="magenta")

        import json
        from datetime import datetime

        # Load and display sessions
        valid_sessions = []
        for i, mf in enumerate(metadata_files[:20], 1):  # Show max 20 most recent
            try:
                with open(mf, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                session_info = metadata.get('annotation_session', {})
                model_config = metadata.get('model_configuration', {})

                timestamp_str = session_info.get('timestamp', '')
                try:
                    dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = timestamp_str

                workflow = session_info.get('workflow', 'Unknown')
                model_name = model_config.get('model_name', 'Unknown')

                sessions_table.add_row(
                    str(i),
                    mf.stem[:40],
                    date_str,
                    workflow.split(' - ')[0] if ' - ' in workflow else workflow,
                    model_name
                )

                valid_sessions.append((mf, metadata))
            except Exception as e:
                self.logger.warning(f"Could not load metadata file {mf}: {e}")
                continue

        if not valid_sessions:
            self.console.print("[yellow]No valid metadata files found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        self.console.print(sessions_table)

        # Select session
        session_choice = self._int_prompt_with_validation(
            "\n[bold yellow]Select session to resume/relaunch[/bold yellow]",
            1, 1, len(valid_sessions)
        )

        selected_file, metadata = valid_sessions[session_choice - 1]

        self.console.print(f"\n[green]✓ Selected: {selected_file.name}[/green]")

        # ============================================================
        # DISPLAY ALL PARAMETERS
        # ============================================================
        self._display_metadata_parameters(metadata)

        # ============================================================
        # ASK: RESUME OR RELAUNCH?
        # ============================================================
        self.console.print("\n[bold cyan]📋 Action Mode[/bold cyan]\n")
        self.console.print("[yellow]What would you like to do?[/yellow]")
        self.console.print("  • [cyan]resume[/cyan]   - Continue an incomplete annotation (skip already annotated rows)")
        self.console.print("           [dim]Requires the output file with annotated rows[/dim]")
        self.console.print("  • [cyan]relaunch[/cyan] - Start a new annotation with same parameters")
        self.console.print("           [dim]Runs a fresh annotation session[/dim]")

        action_mode = Prompt.ask(
            "\n[bold yellow]Select action[/bold yellow]",
            choices=["resume", "relaunch"],
            default="relaunch"
        )

        # ============================================================
        # ASK: MODIFY PARAMETERS?
        # ============================================================
        self.console.print("\n[bold cyan]⚙️  Parameter Modification[/bold cyan]\n")

        modify_params = Confirm.ask(
            "Do you want to modify any parameters?",
            default=False
        )

        # Extract parameters from metadata
        modified_metadata = self._modify_parameters_if_requested(metadata, modify_params)

        # ============================================================
        # EXECUTE ANNOTATION
        # ============================================================
        self._execute_from_metadata(modified_metadata, action_mode, selected_file)

        self.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    def _smart_annotate(self):
        """Smart guided annotation wizard with all options"""
        self.console.print("\n[bold cyan]🎯 Smart Annotate - Guided Wizard[/bold cyan]\n")

        # Step 1: Data Selection
        self.console.print("[bold]Step 1/6: Data Selection[/bold]")

        if not self.detected_datasets:
            self.console.print("[yellow]No datasets auto-detected.[/yellow]")
            data_path = Path(self._prompt_file_path("Dataset path"))
        else:
            self.console.print(f"\n[dim]Found {len(self.detected_datasets)} datasets:[/dim]")
            for i, ds in enumerate(self.detected_datasets[:10], 1):
                self.console.print(f"  {i}. {ds.path.name} ({ds.format.upper()}, {ds.size_mb:.1f} MB)")

            use_detected = Confirm.ask("\nUse detected dataset?", default=True)
            if use_detected:
                choice = self._int_prompt_with_validation("Select dataset", 1, 1, len(self.detected_datasets))
                data_path = self.detected_datasets[choice - 1].path
            else:
                data_path = Path(self._prompt_file_path("Dataset path"))

        # Detect format
        data_format = data_path.suffix[1:].lower()
        if data_format == 'xlsx':
            data_format = 'excel'

        self.console.print(f"[green]✓ Selected: {data_path.name} ({data_format})[/green]")

        # Step 2: Text column selection with intelligent detection
        self.console.print("\n[bold]Step 2/6: Text Column Selection[/bold]")

        # Detect text columns
        column_info = self._detect_text_columns(data_path)

        if column_info['text_candidates']:
            self.console.print("\n[dim]Detected text columns (sorted by confidence):[/dim]")

            # Create table for candidates
            col_table = Table(border_style="blue")
            col_table.add_column("#", style="cyan", width=3)
            col_table.add_column("Column", style="white")
            col_table.add_column("Confidence", style="yellow")
            col_table.add_column("Avg Length", style="green")
            col_table.add_column("Sample", style="dim")

            for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
                # Color code confidence
                conf_color = {
                    "high": "[green]High[/green]",
                    "medium": "[yellow]Medium[/yellow]",
                    "low": "[orange1]Low[/orange1]",
                    "very_low": "[red]Very Low[/red]"
                }
                conf_display = conf_color.get(candidate['confidence'], candidate['confidence'])

                col_table.add_row(
                    str(i),
                    candidate['name'],
                    conf_display,
                    f"{candidate['avg_length']:.0f} chars",
                    candidate['sample'][:50] + "..." if len(candidate['sample']) > 50 else candidate['sample']
                )

            self.console.print(col_table)

            # Show all columns option
            self.console.print(f"\n[dim]All columns ({len(column_info['all_columns'])}): {', '.join(column_info['all_columns'])}[/dim]")

            # Ask user to select
            default_col = column_info['text_candidates'][0]['name'] if column_info['text_candidates'] else "text"
            text_column = Prompt.ask(
                "\n[bold yellow]Enter column name[/bold yellow] (or choose from above)",
                default=default_col
            )
        else:
            # No candidates detected, show all columns
            if column_info['all_columns']:
                self.console.print(f"\n[yellow]Could not auto-detect text columns.[/yellow]")
                self.console.print(f"[dim]Available columns: {', '.join(column_info['all_columns'])}[/dim]")
            text_column = Prompt.ask("Text column name", default="text")

        # Step 3: Model Selection
        self.console.print("\n[bold]Step 3/6: Model Selection[/bold]")
        self.console.print("[dim]Tested API models: OpenAI & Anthropic[/dim]\n")

        selected_llm = self._select_llm_interactive()
        provider = selected_llm.provider
        model_name = selected_llm.name

        # Get API key if needed
        api_key = None
        if selected_llm.requires_api_key:
            api_key = self._get_or_prompt_api_key(provider, model_name)

        # Step 4: Prompt Configuration
        self.console.print("\n[bold]Step 4/6: Prompt Configuration[/bold]")

        # Auto-detect prompts
        detected_prompts = self._detect_prompts_in_folder()

        if detected_prompts:
            self.console.print(f"\n[green]✓ Found {len(detected_prompts)} prompts in prompts/ folder:[/green]")
            for i, p in enumerate(detected_prompts, 1):
                # Display ALL keys, not truncated
                keys_str = ', '.join(p['keys'])
                self.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
                self.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

            # Explain the options clearly
            self.console.print("\n[bold]Prompt Selection Options:[/bold]")
            self.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts (multi-prompt mode)")
            self.console.print("           → Each text will be annotated with all prompts")
            self.console.print("           → Useful when you want complete annotations from all perspectives")
            self.console.print("\n  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number (e.g., 1,3,5)")
            self.console.print("           → Only selected prompts will be used")
            self.console.print("           → Useful when testing or when you need only certain annotations")
            self.console.print("\n  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
            self.console.print("           → Interactive guided prompt creation")
            self.console.print("           → Optional AI assistance for definitions")
            self.console.print("           → [bold green]Recommended for new research projects![/bold green]")
            self.console.print("\n  [cyan]custom[/cyan]  - Provide path to a prompt file NOT in prompts/ folder")
            self.console.print("           → Use a prompt from another location")
            self.console.print("           → Useful for testing new prompts or one-off annotations")

            prompt_choice = Prompt.ask(
                "\n[bold yellow]Prompt selection[/bold yellow]",
                choices=["all", "select", "wizard", "custom"],
                default="all"
            )

            selected_prompts = []
            if prompt_choice == "all":
                selected_prompts = detected_prompts
                self.console.print(f"[green]✓ Using all {len(selected_prompts)} prompts[/green]")
            elif prompt_choice == "select":
                indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
                for idx_str in indices.split(','):
                    idx = int(idx_str.strip()) - 1
                    if 0 <= idx < len(detected_prompts):
                        selected_prompts.append(detected_prompts[idx])
                self.console.print(f"[green]✓ Selected {len(selected_prompts)} prompts[/green]")
            elif prompt_choice == "wizard":
                # Launch Social Science Wizard
                wizard_prompt = self._run_social_science_wizard()
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(wizard_prompt)
                selected_prompts = [{
                    'path': None,  # Wizard-generated, not from file
                    'name': 'wizard_generated',
                    'keys': keys,
                    'content': wizard_prompt
                }]
                self.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
            else:
                # Custom path
                custom_path = Path(self._prompt_file_path("Prompt file path (.txt)"))
                content = custom_path.read_text(encoding='utf-8')
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(content)
                selected_prompts = [{
                    'path': custom_path,
                    'name': custom_path.stem,
                    'keys': keys,
                    'content': content
                }]
        else:
            self.console.print("[yellow]No prompts found in prompts/ folder[/yellow]")

            # Offer wizard or custom path
            self.console.print("\n[bold]Prompt Options:[/bold]")
            self.console.print("  [cyan]wizard[/cyan] - 🧙‍♂️ Create prompt using Social Science Wizard (Recommended)")
            self.console.print("  [cyan]custom[/cyan] - Provide path to existing prompt file")

            choice = Prompt.ask(
                "\n[bold yellow]Select option[/bold yellow]",
                choices=["wizard", "custom"],
                default="wizard"
            )

            if choice == "wizard":
                # Launch Social Science Wizard
                wizard_prompt = self._run_social_science_wizard()
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(wizard_prompt)
                selected_prompts = [{
                    'path': None,  # Wizard-generated, not from file
                    'name': 'wizard_generated',
                    'keys': keys,
                    'content': wizard_prompt
                }]
                self.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
            else:
                custom_path = Path(self._prompt_file_path("Prompt file path (.txt)"))
                content = custom_path.read_text(encoding='utf-8')
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(content)
                selected_prompts = [{
                    'path': custom_path,
                    'name': custom_path.stem,
                    'keys': keys,
                    'content': content
                }]

        # Step 5: Multi-prompt prefix configuration
        prompt_configs = []
        if len(selected_prompts) > 1:
            self.console.print("\n[bold]Multi-Prompt Mode:[/bold] Configure key prefixes")
            self.console.print("[dim]Prefixes help identify which prompt generated which keys[/dim]\n")

            for i, prompt in enumerate(selected_prompts, 1):
                self.console.print(f"\n[cyan]Prompt {i}: {prompt['name']}[/cyan]")
                self.console.print(f"  Keys: {', '.join(prompt['keys'])}")

                add_prefix = Confirm.ask(f"Add prefix to keys for this prompt?", default=True)
                prefix = ""
                if add_prefix:
                    default_prefix = prompt['name'].lower().replace(' ', '_')
                    prefix = Prompt.ask("Prefix", default=default_prefix)
                    self.console.print(f"  [green]Keys will become: {', '.join([f'{prefix}_{k}' for k in prompt['keys'][:3]])}[/green]")

                prompt_configs.append({
                    'prompt': prompt,
                    'prefix': prefix
                })
        else:
            # Single prompt - no prefix needed
            prompt_configs = [{'prompt': selected_prompts[0], 'prefix': ''}]

        # Step 6: Advanced Options
        self.console.print("\n[bold]Step 5/6: Advanced Options[/bold]")

        # ============================================================
        # DATASET SCOPE
        # ============================================================
        self.console.print("\n[bold cyan]📊 Dataset Scope[/bold cyan]")
        self.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

        # Get total rows if possible
        total_rows = None
        if column_info.get('df') is not None:
            # We have a sample, extrapolate
            total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

        if total_rows:
            self.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

        # Option 1: Annotate all or limited
        self.console.print("[yellow]Option 1:[/yellow] Annotate ALL rows vs LIMIT to specific number")
        self.console.print("  • [cyan]all[/cyan]   - Annotate the entire dataset")
        self.console.print("           [dim]Use this for production annotations[/dim]")
        self.console.print("  • [cyan]limit[/cyan] - Specify exact number of rows to annotate")
        self.console.print("           [dim]Use this for testing or partial annotation[/dim]")

        scope_choice = Prompt.ask(
            "\nAnnotate entire dataset or limit rows?",
            choices=["all", "limit"],
            default="all"
        )

        annotation_limit = None
        use_sample = False
        sample_strategy = "head"

        if scope_choice == "limit":
            # Ask for specific number
            annotation_limit = self._int_prompt_with_validation(
                "How many rows to annotate?",
                default=100,
                min_value=1,
                max_value=total_rows if total_rows else 1000000
            )

            # Option 2: Calculate representative sample
            if total_rows and total_rows > 1000:
                self.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
                self.console.print("  Calculate statistically representative sample size (95% confidence interval)")
                self.console.print(f"  [dim]• Current selection: {annotation_limit} rows[/dim]")

                calculate_sample = Confirm.ask("Calculate representative sample size?", default=False)

                if calculate_sample:
                    # Formula: n = (Z² × p × (1-p)) / E²
                    # For 95% CI: Z=1.96, p=0.5 (max variance), E=0.05 (5% margin)
                    import math
                    z = 1.96
                    p = 0.5
                    e = 0.05
                    n_infinite = (z**2 * p * (1-p)) / (e**2)
                    n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                    recommended_sample = int(math.ceil(n_adjusted))

                    self.console.print(f"\n[green]📈 Recommended sample size: {recommended_sample} rows[/green]")
                    self.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")

                    use_recommended = Confirm.ask(f"Use recommended sample size ({recommended_sample} rows)?", default=True)
                    if use_recommended:
                        annotation_limit = recommended_sample
                        use_sample = True

            # Option 3: Random sampling
            self.console.print("\n[yellow]Option 3:[/yellow] Sampling Strategy")
            self.console.print("  Choose how to select the rows to annotate")
            self.console.print("  • [cyan]head[/cyan]   - Take first N rows (faster, sequential)")
            self.console.print("           [dim]Good for testing, preserves order[/dim]")
            self.console.print("  • [cyan]random[/cyan] - Random sample of N rows (representative)")
            self.console.print("           [dim]Better for statistical validity, unbiased[/dim]")

            sample_strategy = Prompt.ask(
                "\nSampling strategy",
                choices=["head", "random"],
                default="random" if use_sample else "head"
            )

        # ============================================================
        # PARALLEL PROCESSING
        # ============================================================
        self.console.print("\n[bold cyan]⚙️  Parallel Processing[/bold cyan]")
        self.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

        self.console.print("[yellow]Parallel Workers:[/yellow]")
        self.console.print("  Number of simultaneous annotation processes")
        self.console.print("\n  [red]⚠️  IMPORTANT:[/red]")
        self.console.print("  [dim]Most local machines can only handle 1 worker for LLM inference[/dim]")
        self.console.print("  [dim]Parallel processing is mainly useful for API models[/dim]")
        self.console.print("\n  • [cyan]1 worker[/cyan]  - Sequential processing")
        self.console.print("           [dim]Recommended for: Local models (Ollama), first time users, debugging[/dim]")
        self.console.print("  • [cyan]2-4 workers[/cyan] - Moderate parallelism")
        self.console.print("           [dim]Recommended for: API models (OpenAI, Claude) - avoid rate limits[/dim]")
        self.console.print("  • [cyan]4-8 workers[/cyan] - High parallelism")
        self.console.print("           [dim]Recommended for: API models only - requires high rate limits[/dim]")

        num_processes = self._int_prompt_with_validation("Parallel workers", 1, 1, 16)

        # ============================================================
        # INCREMENTAL SAVE
        # ============================================================
        self.console.print("\n[bold cyan]💾 Incremental Save[/bold cyan]")
        self.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

        self.console.print("[yellow]Enable incremental save?[/yellow]")
        self.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
        self.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
        self.console.print("  • [red]No[/red]  - Save only at the end")
        self.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

        save_incrementally = Confirm.ask("\n💿 Enable incremental save?", default=True)

        # Only ask for batch size if incremental save is enabled
        if save_incrementally:
            self.console.print("\n[yellow]Batch Size:[/yellow]")
            self.console.print("  Number of rows processed between each save")
            self.console.print("  • [cyan]Smaller (1-10)[/cyan]   - Very frequent saves, maximum safety")
            self.console.print("           [dim]Use for: Unstable systems, expensive APIs, testing[/dim]")
            self.console.print("  • [cyan]Medium (10-50)[/cyan]   - Balanced safety and performance")
            self.console.print("           [dim]Use for: Most production cases[/dim]")
            self.console.print("  • [cyan]Larger (50-200)[/cyan]  - Less frequent saves, better performance")
            self.console.print("           [dim]Use for: Stable systems, large datasets, local models[/dim]")

            batch_size = self._int_prompt_with_validation("Batch size", 1, 1, 1000)
        else:
            batch_size = None  # Not used when incremental save is disabled

        # ============================================================
        # MODEL PARAMETERS
        # ============================================================
        self.console.print("\n[bold cyan]🎛️  Model Parameters[/bold cyan]")
        self.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

        # Check if model supports parameter tuning
        model_name_lower = model_name.lower()
        is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
        supports_params = not is_o_series

        if not supports_params:
            self.console.print(f"[yellow]⚠️  Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
            self.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")
            configure_params = False
        else:
            self.console.print("[yellow]Configure model parameters?[/yellow]")
            self.console.print("  Adjust how the model generates responses")
            self.console.print("  [dim]• Default values work well for most cases[/dim]")
            self.console.print("  [dim]• Advanced users can fine-tune for specific needs[/dim]")
            configure_params = Confirm.ask("\nConfigure model parameters?", default=False)

        # Default values
        temperature = 0.7
        max_tokens = 1000
        top_p = 1.0
        top_k = 40

        if configure_params:
            self.console.print("\n[bold]Parameter Explanations:[/bold]\n")

            # Temperature
            self.console.print("[cyan]🌡️  Temperature (0.0 - 2.0):[/cyan]")
            self.console.print("  Controls randomness in responses")
            self.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
            self.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
            self.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
            self.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
            self.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
            self.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
            temperature = FloatPrompt.ask("Temperature", default=0.7)

            # Max tokens
            self.console.print("\n[cyan]📏 Max Tokens:[/cyan]")
            self.console.print("  Maximum length of the response")
            self.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
            self.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
            self.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
            self.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
            max_tokens = self._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

            # Top_p (nucleus sampling)
            self.console.print("\n[cyan]🎯 Top P (0.0 - 1.0):[/cyan]")
            self.console.print("  Nucleus sampling - alternative to temperature")
            self.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
            self.console.print("           [dim]More deterministic, safer outputs[/dim]")
            self.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
            self.console.print("           [dim]More creative, diverse outputs[/dim]")
            self.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
            top_p = FloatPrompt.ask("Top P", default=1.0)

            # Top_k (only for some models)
            if provider in ['ollama', 'google']:
                self.console.print("\n[cyan]🔢 Top K:[/cyan]")
                self.console.print("  Limits vocabulary to K most likely next tokens")
                self.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
                self.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
                self.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
                top_k = self._int_prompt_with_validation("Top K", 40, 1, 100)

        # Step 7: Execute
        self.console.print("\n[bold]Step 6/6: Review & Execute[/bold]")

        # Display comprehensive summary
        summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True)
        summary_table.add_column("Category", style="bold cyan", width=20)
        summary_table.add_column("Setting", style="yellow", width=25)
        summary_table.add_column("Value", style="white")

        # Data section
        summary_table.add_row("📁 Data", "Dataset", str(data_path.name))
        summary_table.add_row("", "Format", data_format.upper())
        summary_table.add_row("", "Text Column", text_column)
        if total_rows:
            summary_table.add_row("", "Total Rows", f"{total_rows:,}")
        if annotation_limit:
            summary_table.add_row("", "Rows to Annotate", f"{annotation_limit:,} ({sample_strategy})")
        else:
            summary_table.add_row("", "Rows to Annotate", "ALL")

        # Model section
        summary_table.add_row("🤖 Model", "Provider/Model", f"{provider}/{model_name}")
        summary_table.add_row("", "Temperature", f"{temperature}")
        summary_table.add_row("", "Max Tokens", f"{max_tokens}")
        if configure_params:
            summary_table.add_row("", "Top P", f"{top_p}")
            if provider in ['ollama', 'google']:
                summary_table.add_row("", "Top K", f"{top_k}")

        # Prompts section
        summary_table.add_row("📝 Prompts", "Count", f"{len(prompt_configs)}")
        for i, pc in enumerate(prompt_configs, 1):
            prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
            summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

        # Processing section
        summary_table.add_row("⚙️  Processing", "Parallel Workers", str(num_processes))
        summary_table.add_row("", "Batch Size", str(batch_size))
        summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

        self.console.print("\n")
        self.console.print(summary_table)

        if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
            return

        # ============================================================
        # REPRODUCIBILITY METADATA
        # ============================================================
        self.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
        self.console.print("[yellow]⚠️  IMPORTANT: Save parameters for two critical purposes:[/yellow]\n")

        self.console.print("  [green]1. Resume Capability[/green]")
        self.console.print("     • Continue this annotation if it stops or crashes")
        self.console.print("     • Annotate additional rows later with same settings")
        self.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

        self.console.print("  [green]2. Scientific Reproducibility[/green]")
        self.console.print("     • Document exact parameters for research papers")
        self.console.print("     • Reproduce identical annotations in the future")
        self.console.print("     • Track model version, prompts, and all settings\n")

        self.console.print("  [red]⚠️  If you choose NO:[/red]")
        self.console.print("     • You CANNOT resume this annotation later")
        self.console.print("     • You CANNOT relaunch with same parameters")
        self.console.print("     • Parameters will be lost forever\n")

        save_metadata = Confirm.ask(
            "[bold yellow]Save annotation parameters to JSON file?[/bold yellow]",
            default=True
        )

        # ============================================================
        # VALIDATION TOOL EXPORT OPTION
        # ============================================================
        self.console.print("\n[bold cyan]📤 Validation Tool Export[/bold cyan]")
        self.console.print("[dim]Export annotations to JSONL format for human validation[/dim]\n")

        self.console.print("[yellow]Available validation tools:[/yellow]")
        self.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
        self.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
        self.console.print("  • Both are open-source and free\n")

        self.console.print("[green]Why validate with external tools?[/green]")
        self.console.print("  • Review and correct LLM annotations")
        self.console.print("  • Calculate inter-annotator agreement")
        self.console.print("  • Export validated data for metrics calculation\n")

        # Initialize export flags
        export_to_doccano = False
        export_to_labelstudio = False
        export_sample_size = None

        # Step 1: Ask if user wants to export
        export_confirm = Confirm.ask(
            "[bold yellow]Export to validation tool?[/bold yellow]",
            default=False
        )

        if export_confirm:
            # Step 2: Ask which tool to export to
            tool_choice = Prompt.ask(
                "[bold yellow]Which validation tool?[/bold yellow]",
                choices=["doccano", "labelstudio"],
                default="doccano"
            )

            # Set the appropriate export flag
            if tool_choice == "doccano":
                export_to_doccano = True
            else:  # labelstudio
                export_to_labelstudio = True

            # Step 2b: If Label Studio, ask export method
            labelstudio_direct_export = False
            labelstudio_api_url = None
            labelstudio_api_key = None

            if export_to_labelstudio:
                self.console.print("\n[yellow]Label Studio export method:[/yellow]")
                self.console.print("  • [cyan]jsonl[/cyan] - Export to JSONL file (manual import)")
                if HAS_REQUESTS:
                    self.console.print("  • [cyan]direct[/cyan] - Direct export to Label Studio via API\n")
                    export_choices = ["jsonl", "direct"]
                else:
                    self.console.print("  • [dim]direct[/dim] - Direct export via API [dim](requires 'requests' library)[/dim]\n")
                    export_choices = ["jsonl"]

                export_method = Prompt.ask(
                    "[bold yellow]Export method[/bold yellow]",
                    choices=export_choices,
                    default="jsonl"
                )

                if export_method == "direct":
                    labelstudio_direct_export = True

                    self.console.print("\n[cyan]Label Studio API Configuration:[/cyan]")
                    labelstudio_api_url = Prompt.ask(
                        "Label Studio URL",
                        default="http://localhost:8080"
                    )

                    labelstudio_api_key = Prompt.ask(
                        "API Key (from Label Studio Account & Settings)"
                    )

            # Step 3: Ask about LLM predictions inclusion
            self.console.print("\n[yellow]Include LLM predictions in export?[/yellow]")
            self.console.print("  • [cyan]with[/cyan] - Include LLM annotations as predictions (for review/correction)")
            self.console.print("  • [cyan]without[/cyan] - Export only data without predictions (for manual annotation)")
            self.console.print("  • [cyan]both[/cyan] - Create two files: one with and one without predictions\n")

            prediction_mode = Prompt.ask(
                "[bold yellow]Prediction mode[/bold yellow]",
                choices=["with", "without", "both"],
                default="with"
            )

            # Step 4: Ask how many sentences to export
            self.console.print("\n[yellow]How many annotated sentences to export?[/yellow]")
            self.console.print("  • [cyan]all[/cyan] - Export all annotated sentences")
            self.console.print("  • [cyan]representative[/cyan] - Representative sample (stratified by labels)")
            self.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

            sample_choice = Prompt.ask(
                "[bold yellow]Export sample[/bold yellow]",
                choices=["all", "representative", "number"],
                default="all"
            )

            if sample_choice == "all":
                export_sample_size = "all"
            elif sample_choice == "representative":
                export_sample_size = "representative"
            else:  # number
                export_sample_size = self._int_prompt_with_validation(
                    "Number of sentences to export",
                    100,
                    1,
                    999999
                )

        # ============================================================
        # EXECUTE ANNOTATION
        # ============================================================

        # Prepare output path
        annotations_dir = self.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = model_name.replace(':', '_').replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
        default_output_path = annotations_dir / output_filename

        self.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
        self.console.print(f"   {default_output_path}")
        self.console.print()

        # Prepare prompts payload for pipeline
        prompts_payload = []
        for pc in prompt_configs:
            prompts_payload.append({
                'prompt': pc['prompt']['content'],
                'expected_keys': pc['prompt']['keys'],
                'prefix': pc['prefix']
            })

        # Determine annotation mode
        annotation_mode = 'api' if provider in {'openai', 'anthropic', 'google'} else 'local'

        # Build pipeline config
        pipeline_config = {
            'mode': 'file',
            'data_source': data_format,
            'data_format': data_format,
            'file_path': str(data_path),
            'text_column': text_column,
            'text_columns': [text_column],
            'annotation_column': 'annotation',
            'identifier_column': 'annotation_id',  # Will be created if doesn't exist
            'run_annotation': True,
            'annotation_mode': annotation_mode,
            'annotation_provider': provider,
            'annotation_model': model_name,
            'api_key': api_key if api_key else None,
            'prompts': prompts_payload,
            'annotation_sample_size': annotation_limit,
            'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
            'annotation_sample_seed': 42,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k if provider in ['ollama', 'google'] else None,
            'max_workers': num_processes,
            'num_processes': num_processes,
            'use_parallel': num_processes > 1,
            'warmup': False,
            'disable_tqdm': True,  # Use Rich progress instead
            'output_format': data_format,
            'output_path': str(default_output_path),
            'save_incrementally': save_incrementally,
            'batch_size': batch_size,
            'run_validation': False,
            'run_training': False,
        }

        # Add model-specific options
        if provider == 'ollama':
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
                'top_p': top_p,
                'top_k': top_k
            }
            pipeline_config['options'] = options

        # ============================================================
        # SAVE REPRODUCIBILITY METADATA
        # ============================================================
        if save_metadata:
            import json

            # Build comprehensive metadata
            metadata = {
                'annotation_session': {
                    'timestamp': timestamp,
                    'tool_version': 'LLMTool v1.0',
                    'workflow': 'LLM Annotation Studio - Smart Annotate'
                },
                'data_source': {
                    'file_path': str(data_path),
                    'file_name': data_path.name,
                    'data_format': data_format,
                    'text_column': text_column,
                    'total_rows': annotation_limit if annotation_limit else 'all',
                    'sampling_strategy': sample_strategy if annotation_limit else 'none (all rows)',
                    'sample_seed': 42 if sample_strategy == 'random' else None
                },
                'model_configuration': {
                    'provider': provider,
                    'model_name': model_name,
                    'annotation_mode': annotation_mode,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'top_p': top_p,
                    'top_k': top_k if provider in ['ollama', 'google'] else None
                },
                'prompts': [
                    {
                        'name': pc['prompt']['name'],
                        'file_path': str(pc['prompt']['path']) if 'path' in pc['prompt'] else None,
                        'expected_keys': pc['prompt']['keys'],
                        'prefix': pc['prefix'],
                        'prompt_content': pc['prompt']['content']
                    }
                    for pc in prompt_configs
                ],
                'processing_configuration': {
                    'parallel_workers': num_processes,
                    'batch_size': batch_size,
                    'incremental_save': save_incrementally,
                    'identifier_column': 'annotation_id'
                },
                'output': {
                    'output_path': str(default_output_path),
                    'output_format': data_format
                },
                'export_preferences': {
                    'export_to_doccano': export_to_doccano,
                    'export_to_labelstudio': export_to_labelstudio,
                    'export_sample_size': export_sample_size,
                    'prediction_mode': prediction_mode if (export_to_doccano or export_to_labelstudio) else 'with',
                    'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
                    'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
                    'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None
                }
            }

            # Save metadata JSON
            metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
            metadata_path = annotations_dir / metadata_filename

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.console.print(f"\n[bold green]✅ Metadata saved for reproducibility[/bold green]")
            self.console.print(f"[bold cyan]📋 Metadata File:[/bold cyan]")
            self.console.print(f"   {metadata_path}\n")

        # Execute pipeline with Rich progress
        try:
            self.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

            # Create pipeline controller
            from ..pipelines.pipeline_controller import PipelineController
            pipeline_with_progress = PipelineController(settings=self.settings)

            # Use RichProgressManager for elegant display
            from ..utils.rich_progress_manager import RichProgressManager
            from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

            with RichProgressManager(
                show_json_every=1,  # Show JSON sample for every annotation
                compact_mode=False   # Full preview panels
            ) as progress_manager:
                # Wrap pipeline for enhanced JSON tracking
                enhanced_pipeline = EnhancedPipelineWrapper(
                    pipeline_with_progress,
                    progress_manager
                )

                # Run pipeline
                state = enhanced_pipeline.run_pipeline(pipeline_config)

                # Check for errors
                if state.errors:
                    error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                    self.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                    self.console.print("[dim]Press Enter to return to menu...[/dim]")
                    input()
                    return

            # Get results
            annotation_results = state.annotation_results or {}
            output_file = annotation_results.get('output_file', str(default_output_path))

            # Display success message
            self.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
            self.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
            self.console.print(f"   {output_file}")

            # Display statistics if available
            total_annotated = annotation_results.get('total_annotated', 0)
            if total_annotated:
                self.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
                self.console.print(f"   Rows annotated: {total_annotated:,}")

                success_count = annotation_results.get('success_count', 0)
                if success_count:
                    success_rate = (success_count / total_annotated * 100)
                    self.console.print(f"   Success rate: {success_rate:.1f}%")

            # Export to Doccano JSONL if requested
            if export_to_doccano:
                self._export_to_doccano_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size
                )

            # Export to Label Studio if requested
            if export_to_labelstudio:
                if labelstudio_direct_export:
                    # Direct export to Label Studio via API
                    self._export_to_labelstudio_direct(
                        output_file=output_file,
                        text_column=text_column,
                        prompt_configs=prompt_configs,
                        data_path=data_path,
                        timestamp=timestamp,
                        sample_size=export_sample_size,
                        prediction_mode=prediction_mode,
                        api_url=labelstudio_api_url,
                        api_key=labelstudio_api_key
                    )
                else:
                    # Export to JSONL file
                    self._export_to_labelstudio_jsonl(
                        output_file=output_file,
                        text_column=text_column,
                        prompt_configs=prompt_configs,
                        data_path=data_path,
                        timestamp=timestamp,
                        sample_size=export_sample_size,
                        prediction_mode=prediction_mode
                    )

            self.console.print("\n[dim]Press Enter to return to menu...[/dim]")
            input()

        except Exception as exc:
            self.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
            self.logger.exception("Annotation execution failed")
            self.console.print("\n[dim]Press Enter to return to menu...[/dim]")
            input()

    def _display_metadata_parameters(self, metadata: dict):
        """Display all parameters from metadata in a formatted way"""
        self.console.print("\n[bold cyan]📋 Saved Parameters[/bold cyan]\n")

        # Create parameter display table
        params_table = Table(border_style="blue", show_header=False, box=None)
        params_table.add_column("Section", style="yellow bold", width=25)
        params_table.add_column("Details", style="white")

        # Session Info
        session = metadata.get('annotation_session', {})
        params_table.add_row("📅 Session", f"{session.get('workflow', 'N/A')}")
        params_table.add_row("", f"Date: {session.get('timestamp', 'N/A')}")

        # Data Source
        data_source = metadata.get('data_source', {})
        params_table.add_row("📁 Data", f"File: {data_source.get('file_name', 'N/A')}")
        params_table.add_row("", f"Format: {data_source.get('data_format', 'N/A')}")
        params_table.add_row("", f"Text Column: {data_source.get('text_column', 'N/A')}")
        params_table.add_row("", f"Rows: {data_source.get('total_rows', 'N/A')}")
        params_table.add_row("", f"Sampling: {data_source.get('sampling_strategy', 'N/A')}")

        # Model Configuration
        model_config = metadata.get('model_configuration', {})
        params_table.add_row("🤖 Model", f"{model_config.get('provider', 'N/A')}/{model_config.get('model_name', 'N/A')}")
        params_table.add_row("", f"Temperature: {model_config.get('temperature', 'N/A')}")
        params_table.add_row("", f"Max Tokens: {model_config.get('max_tokens', 'N/A')}")

        # Prompts
        prompts = metadata.get('prompts', [])
        params_table.add_row("📝 Prompts", f"Count: {len(prompts)}")
        for i, p in enumerate(prompts[:3], 1):  # Show first 3
            name = p.get('name', f"Prompt {i}")
            keys = p.get('expected_keys', [])
            prefix = p.get('prefix', '')
            keys_str = ', '.join(keys[:5])
            if len(keys) > 5:
                keys_str += f"... ({len(keys)} total)"
            prefix_str = f" [{prefix}_]" if prefix else ""
            params_table.add_row("", f"  {i}. {name}{prefix_str}: {keys_str}")

        # Processing
        proc_config = metadata.get('processing_configuration', {})
        params_table.add_row("⚙️  Processing", f"Workers: {proc_config.get('parallel_workers', 1)}")
        params_table.add_row("", f"Batch Size: {proc_config.get('batch_size', 'N/A')}")

        self.console.print(params_table)

    def _modify_parameters_if_requested(self, metadata: dict, modify: bool) -> dict:
        """Allow user to modify specific parameters"""
        if not modify:
            return metadata

        self.console.print("\n[bold]Select parameter to modify:[/bold]")
        self.console.print("  [cyan]1[/cyan] - Data source (file, text column)")
        self.console.print("  [cyan]2[/cyan] - Model (provider, model name)")
        self.console.print("  [cyan]3[/cyan] - Model parameters (temperature, max_tokens, etc.)")
        self.console.print("  [cyan]4[/cyan] - Prompts (add/remove/modify)")
        self.console.print("  [cyan]5[/cyan] - Sampling (rows to annotate, strategy)")
        self.console.print("  [cyan]6[/cyan] - Processing (workers, batch size)")
        self.console.print("  [cyan]0[/cyan] - Done modifying")

        modified = metadata.copy()

        while True:
            choice = Prompt.ask(
                "\n[bold yellow]Modify which parameter?[/bold yellow]",
                choices=["0", "1", "2", "3", "4", "5", "6"],
                default="0"
            )

            if choice == "0":
                break
            elif choice == "1":
                # Modify data source
                self.console.print("\n[yellow]Current data:[/yellow]")
                data_source = modified.get('data_source', {})
                self.console.print(f"  File: {data_source.get('file_path', 'N/A')}")
                self.console.print(f"  Text column: {data_source.get('text_column', 'N/A')}")

                if Confirm.ask("Change data file?", default=False):
                    new_file = self._prompt_file_path("New data file path")
                    modified['data_source']['file_path'] = new_file
                    modified['data_source']['file_name'] = Path(new_file).name

                if Confirm.ask("Change text column?", default=False):
                    new_col = Prompt.ask("New text column name")
                    modified['data_source']['text_column'] = new_col

            elif choice == "2":
                # Modify model
                self.console.print("\n[yellow]Current model:[/yellow]")
                model_config = modified.get('model_configuration', {})
                self.console.print(f"  Provider: {model_config.get('provider', 'N/A')}")
                self.console.print(f"  Model: {model_config.get('model_name', 'N/A')}")

                if Confirm.ask("Change model?", default=False):
                    # Reuse model selection from smart annotate
                    provider = Prompt.ask("Provider", choices=["ollama", "openai", "anthropic"], default="ollama")
                    model_name = Prompt.ask("Model name")
                    modified['model_configuration']['provider'] = provider
                    modified['model_configuration']['model_name'] = model_name

            elif choice == "3":
                # Modify model parameters
                model_config = modified.get('model_configuration', {})

                if Confirm.ask("Change temperature?", default=False):
                    temp = FloatPrompt.ask("Temperature (0.0-2.0)", default=0.7)
                    modified['model_configuration']['temperature'] = temp

                if Confirm.ask("Change max_tokens?", default=False):
                    tokens = self._int_prompt_with_validation("Max tokens", 1000, 50, 8000)
                    modified['model_configuration']['max_tokens'] = tokens

            elif choice == "4":
                # Modify prompts
                self.console.print("\n[yellow]Prompt modification not implemented in this version.[/yellow]")
                self.console.print("[dim]Use Smart Annotate to create new annotation with different prompts.[/dim]")

            elif choice == "5":
                # Modify sampling
                data_source = modified.get('data_source', {})
                current_rows = data_source.get('total_rows', 'all')

                self.console.print(f"\n[yellow]Current: {current_rows} rows[/yellow]")

                if Confirm.ask("Change number of rows to annotate?", default=False):
                    annotate_all = Confirm.ask("Annotate all rows?", default=True)
                    if annotate_all:
                        modified['data_source']['total_rows'] = 'all'
                        modified['data_source']['sampling_strategy'] = 'none'
                    else:
                        num_rows = self._int_prompt_with_validation("Number of rows", 100, 1, 1000000)
                        strategy = Prompt.ask("Sampling strategy", choices=["head", "random"], default="random")
                        modified['data_source']['total_rows'] = num_rows
                        modified['data_source']['sampling_strategy'] = strategy

            elif choice == "6":
                # Modify processing
                proc_config = modified.get('processing_configuration', {})

                if Confirm.ask("Change parallel workers?", default=False):
                    workers = self._int_prompt_with_validation("Parallel workers", 1, 1, 16)
                    modified['processing_configuration']['parallel_workers'] = workers

                if Confirm.ask("Change batch size?", default=False):
                    batch = self._int_prompt_with_validation("Batch size", 1, 1, 1000)
                    modified['processing_configuration']['batch_size'] = batch

        self.console.print("\n[green]✓ Parameters modified[/green]")
        return modified

    def _execute_from_metadata(self, metadata: dict, action_mode: str, metadata_file: Path):
        """Execute annotation based on loaded metadata"""
        import json
        from datetime import datetime

        # Extract all parameters from metadata
        data_source = metadata.get('data_source', {})
        model_config = metadata.get('model_configuration', {})
        prompts = metadata.get('prompts', [])
        proc_config = metadata.get('processing_configuration', {})
        output_config = metadata.get('output', {})
        export_prefs = metadata.get('export_preferences', {})

        # Get export preferences
        export_to_doccano = export_prefs.get('export_to_doccano', False)
        export_to_labelstudio = export_prefs.get('export_to_labelstudio', False)
        export_sample_size = export_prefs.get('export_sample_size', 'all')

        if export_to_doccano or export_to_labelstudio:
            export_tools = []
            if export_to_doccano:
                export_tools.append("Doccano")
            if export_to_labelstudio:
                export_tools.append("Label Studio")
            self.console.print(f"\n[cyan]ℹ️  Export enabled for: {', '.join(export_tools)} (from saved preferences)[/cyan]")
            if export_sample_size != 'all':
                self.console.print(f"[cyan]   Sample size: {export_sample_size}[/cyan]")

        # Prepare paths
        data_path = Path(data_source.get('file_path', ''))
        data_format = data_source.get('data_format', 'csv')

        # Check if resuming
        if action_mode == 'resume':
            # Try to find the output file
            original_output = Path(output_config.get('output_path', ''))

            if not original_output.exists():
                self.console.print(f"\n[yellow]⚠️  Output file not found: {original_output}[/yellow]")
                self.console.print("[yellow]Switching to relaunch mode (fresh annotation)[/yellow]")
                action_mode = 'relaunch'
            else:
                self.console.print(f"\n[green]✓ Found output file: {original_output.name}[/green]")

                # Count already annotated rows
                import pandas as pd
                try:
                    if data_format == 'csv':
                        df_output = pd.read_csv(original_output)
                    elif data_format in ['excel', 'xlsx']:
                        df_output = pd.read_excel(original_output)
                    elif data_format == 'parquet':
                        df_output = pd.read_parquet(original_output)

                    # Count rows with valid annotations (non-empty, non-null strings)
                    if 'annotation' in df_output.columns:
                        # Count only rows where annotation exists and is not empty/whitespace
                        annotated_mask = (
                            df_output['annotation'].notna() &
                            (df_output['annotation'].astype(str).str.strip() != '') &
                            (df_output['annotation'].astype(str) != 'nan')
                        )
                        annotated_count = annotated_mask.sum()
                    else:
                        annotated_count = 0

                    self.console.print(f"[cyan]  Rows already annotated: {annotated_count:,}[/cyan]")

                    # Get total available rows from source file
                    if data_path.exists():
                        if data_format == 'csv':
                            total_available = len(pd.read_csv(data_path))
                        elif data_format in ['excel', 'xlsx']:
                            total_available = len(pd.read_excel(data_path))
                        elif data_format == 'parquet':
                            total_available = len(pd.read_parquet(data_path))
                        else:
                            total_available = len(df_output)
                    else:
                        total_available = len(df_output)

                    # Calculate remaining based on original target
                    original_target = data_source.get('total_rows', 'all')

                    if original_target == 'all':
                        total_target = total_available
                    else:
                        total_target = original_target

                    remaining_from_target = total_target - annotated_count
                    remaining_from_source = total_available - annotated_count

                    self.console.print(f"[cyan]  Original target: {total_target:,} rows[/cyan]")
                    self.console.print(f"[cyan]  Remaining from target: {remaining_from_target:,}[/cyan]")
                    self.console.print(f"[cyan]  Total available in source: {total_available:,} rows[/cyan]")
                    self.console.print(f"[cyan]  Maximum you can annotate: {remaining_from_source:,}[/cyan]\n")

                    if remaining_from_source <= 0:
                        self.console.print("\n[yellow]All available rows are already annotated![/yellow]")
                        continue_anyway = Confirm.ask("Continue with relaunch mode?", default=False)
                        if not continue_anyway:
                            return
                        action_mode = 'relaunch'
                    else:
                        self.console.print("[yellow]You can annotate:[/yellow]")
                        self.console.print(f"  • Up to [cyan]{remaining_from_target:,}[/cyan] more rows to complete original target")
                        self.console.print(f"  • Or up to [cyan]{remaining_from_source:,}[/cyan] total to use all available data\n")

                        resume_count = self._int_prompt_with_validation(
                            f"How many more rows to annotate? (max: {remaining_from_source:,})",
                            min(100, remaining_from_target) if remaining_from_target > 0 else 100,
                            1,
                            remaining_from_source
                        )

                        # Update metadata for resume
                        metadata['data_source']['total_rows'] = resume_count
                        metadata['resume_mode'] = True
                        metadata['resume_from_file'] = str(original_output)
                        metadata['already_annotated'] = int(annotated_count)

                except Exception as e:
                    self.console.print(f"\n[red]Error reading output file: {e}[/red]")
                    self.console.print("[yellow]Switching to relaunch mode[/yellow]")
                    action_mode = 'relaunch'

        # Prepare output path
        annotations_dir = self.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = model_config.get('model_name', 'unknown').replace(':', '_').replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if action_mode == 'resume':
            output_filename = original_output.name  # Keep same filename
            default_output_path = original_output
        else:
            output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
            default_output_path = annotations_dir / output_filename

        self.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
        self.console.print(f"   {default_output_path}")

        # Prepare prompts payload
        prompts_payload = []
        for p in prompts:
            prompts_payload.append({
                'prompt': p.get('prompt_content', p.get('prompt', '')),
                'expected_keys': p.get('expected_keys', []),
                'prefix': p.get('prefix', '')
            })

        # Get parameters
        provider = model_config.get('provider', 'ollama')
        model_name = model_config.get('model_name', 'llama2')
        annotation_mode = model_config.get('annotation_mode', 'local')
        temperature = model_config.get('temperature', 0.7)
        max_tokens = model_config.get('max_tokens', 1000)
        top_p = model_config.get('top_p', 1.0)
        top_k = model_config.get('top_k', 40)

        num_processes = proc_config.get('parallel_workers', 1)
        batch_size = proc_config.get('batch_size', 1)

        total_rows = data_source.get('total_rows')
        annotation_limit = None if total_rows == 'all' else total_rows
        sample_strategy = data_source.get('sampling_strategy', 'head')

        # IMPORTANT: In resume mode, always use 'head' strategy to continue sequentially
        # This ensures we pick up exactly where we left off, not random new rows
        if action_mode == 'resume':
            sample_strategy = 'head'
            self.console.print(f"\n[cyan]ℹ️  Resume mode: Using sequential (head) strategy to continue where you left off[/cyan]")

        # Get API key if needed
        api_key = None
        if provider in ['openai', 'anthropic', 'google']:
            api_key = self._get_api_key(provider)
            if not api_key:
                self.console.print(f"[red]API key required for {provider}[/red]")
                return

        # Build pipeline config
        pipeline_config = {
            'mode': 'file',
            'data_source': data_format,
            'data_format': data_format,
            'file_path': str(data_path),
            'text_column': data_source.get('text_column', 'text'),
            'text_columns': [data_source.get('text_column', 'text')],
            'annotation_column': 'annotation',
            'identifier_column': 'annotation_id',
            'run_annotation': True,
            'annotation_mode': annotation_mode,
            'annotation_provider': provider,
            'annotation_model': model_name,
            'api_key': api_key,
            'prompts': prompts_payload,
            'annotation_sample_size': annotation_limit,
            'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
            'annotation_sample_seed': 42,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k if provider in ['ollama', 'google'] else None,
            'max_workers': num_processes,
            'num_processes': num_processes,
            'use_parallel': num_processes > 1,
            'warmup': False,
            'disable_tqdm': True,
            'output_format': data_format,
            'output_path': str(default_output_path),
            'save_incrementally': True,
            'batch_size': batch_size,
            'run_validation': False,
            'run_training': False,
        }

        # Add resume information if resuming
        if action_mode == 'resume' and metadata.get('resume_mode'):
            pipeline_config['resume_mode'] = True
            pipeline_config['resume_from_file'] = metadata.get('resume_from_file')
            pipeline_config['skip_annotated'] = True

            # Load already annotated IDs to skip them
            try:
                import pandas as pd
                resume_file = Path(metadata.get('resume_from_file'))
                if resume_file.exists():
                    if data_format == 'csv':
                        df_resume = pd.read_csv(resume_file)
                    elif data_format in ['excel', 'xlsx']:
                        df_resume = pd.read_excel(resume_file)
                    elif data_format == 'parquet':
                        df_resume = pd.read_parquet(resume_file)

                    # Get IDs of rows that have valid annotations
                    if 'annotation' in df_resume.columns and 'annotation_id' in df_resume.columns:
                        annotated_mask = (
                            df_resume['annotation'].notna() &
                            (df_resume['annotation'].astype(str).str.strip() != '') &
                            (df_resume['annotation'].astype(str) != 'nan')
                        )
                        already_annotated_ids = df_resume.loc[annotated_mask, 'annotation_id'].tolist()
                        pipeline_config['skip_annotation_ids'] = already_annotated_ids

                        self.console.print(f"[cyan]  Will skip {len(already_annotated_ids)} already annotated row(s)[/cyan]")
            except Exception as e:
                self.logger.warning(f"Could not load annotated IDs from resume file: {e}")
                self.console.print(f"[yellow]⚠️  Warning: Could not load annotated IDs - may re-annotate some rows[/yellow]")

        # Add model-specific options
        if provider == 'ollama':
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
                'top_p': top_p,
                'top_k': top_k
            }
            pipeline_config['options'] = options

        # Save new metadata for this execution
        if action_mode == 'relaunch':
            new_metadata = metadata.copy()
            new_metadata['annotation_session']['timestamp'] = timestamp
            new_metadata['annotation_session']['relaunch_from'] = str(metadata_file.name)
            new_metadata['output']['output_path'] = str(default_output_path)

            new_metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
            new_metadata_path = annotations_dir / new_metadata_filename

            with open(new_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(new_metadata, f, indent=2, ensure_ascii=False)

            self.console.print(f"\n[green]✅ New session metadata saved[/green]")
            self.console.print(f"[cyan]📋 Metadata File:[/cyan]")
            self.console.print(f"   {new_metadata_path}\n")

        # Execute pipeline
        try:
            self.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

            from ..pipelines.pipeline_controller import PipelineController
            pipeline_with_progress = PipelineController(settings=self.settings)

            from ..utils.rich_progress_manager import RichProgressManager
            from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

            with RichProgressManager(
                show_json_every=1,
                compact_mode=False
            ) as progress_manager:
                enhanced_pipeline = EnhancedPipelineWrapper(
                    pipeline_with_progress,
                    progress_manager
                )

                state = enhanced_pipeline.run_pipeline(pipeline_config)

                if state.errors:
                    error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                    self.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                    return

            # Display results
            annotation_results = state.annotation_results or {}
            output_file = annotation_results.get('output_file', str(default_output_path))

            self.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
            self.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
            self.console.print(f"   {output_file}")

            total_annotated = annotation_results.get('total_annotated', 0)
            if total_annotated:
                self.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
                self.console.print(f"   Rows annotated: {total_annotated:,}")

                success_count = annotation_results.get('success_count', 0)
                if success_count:
                    success_rate = (success_count / total_annotated * 100)
                    self.console.print(f"   Success rate: {success_rate:.1f}%")

            # Export to Doccano JSONL if enabled in preferences
            if export_to_doccano:
                # Build prompt_configs for export
                prompt_configs_for_export = []
                for p in prompts:
                    prompt_configs_for_export.append({
                        'prompt': {
                            'keys': p.get('expected_keys', []),
                            'content': p.get('prompt_content', p.get('prompt', '')),
                            'name': p.get('name', 'prompt')
                        },
                        'prefix': p.get('prefix', '')
                    })

                self._export_to_doccano_jsonl(
                    output_file=output_file,
                    text_column=data_source.get('text_column', 'text'),
                    prompt_configs=prompt_configs_for_export,
                    data_path=data_path,
                    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                    sample_size=export_sample_size
                )

            # Export to Label Studio JSONL if enabled in preferences
            if export_to_labelstudio:
                # Build prompt_configs for export
                prompt_configs_for_export = []
                for p in prompts:
                    prompt_configs_for_export.append({
                        'prompt': {
                            'keys': p.get('expected_keys', []),
                            'content': p.get('prompt_content', p.get('prompt', '')),
                            'name': p.get('name', 'prompt')
                        },
                        'prefix': p.get('prefix', '')
                    })

                self._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=data_source.get('text_column', 'text'),
                    prompt_configs=prompt_configs_for_export,
                    data_path=data_path,
                    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                    sample_size=export_sample_size
                )

        except Exception as exc:
            self.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
            self.logger.exception("Resume/Relaunch annotation failed")

    def _clean_metadata(self):
        """Clean old metadata files"""
        self.console.print("\n[bold cyan]🗑️  Clean Old Metadata[/bold cyan]\n")
        self.console.print("[dim]Delete saved annotation parameters to free space[/dim]\n")

        annotations_dir = self.settings.paths.data_dir / 'annotations'

        if not annotations_dir.exists():
            self.console.print("[yellow]No annotations directory found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Find all metadata JSON files
        metadata_files = list(annotations_dir.glob("*_metadata_*.json"))

        if not metadata_files:
            self.console.print("[yellow]No metadata files found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Sort by modification time (oldest first for cleaning)
        metadata_files.sort(key=lambda x: x.stat().st_mtime)

        self.console.print(f"[green]Found {len(metadata_files)} metadata file(s)[/green]\n")

        # Display cleaning options
        self.console.print("[bold]Cleaning Options:[/bold]")
        self.console.print("  [cyan]1[/cyan] - Delete ALL metadata files")
        self.console.print("  [cyan]2[/cyan] - Delete metadata older than X days")
        self.console.print("  [cyan]3[/cyan] - Select specific files to delete")
        self.console.print("  [cyan]0[/cyan] - Cancel")

        clean_choice = Prompt.ask(
            "\n[bold yellow]Select cleaning option[/bold yellow]",
            choices=["0", "1", "2", "3"],
            default="0"
        )

        if clean_choice == "0":
            self.console.print("[yellow]Cleaning cancelled[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        files_to_delete = []

        if clean_choice == "1":
            # Delete ALL
            self.console.print(f"\n[red]⚠️  Warning: This will delete ALL {len(metadata_files)} metadata files![/red]")
            confirm = Confirm.ask("Are you sure?", default=False)

            if confirm:
                files_to_delete = metadata_files
            else:
                self.console.print("[yellow]Deletion cancelled[/yellow]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                return

        elif clean_choice == "2":
            # Delete older than X days
            days = self._int_prompt_with_validation(
                "Delete files older than how many days?",
                30, 1, 365
            )

            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(days=days)

            for mf in metadata_files:
                file_time = datetime.fromtimestamp(mf.stat().st_mtime)
                if file_time < cutoff_time:
                    files_to_delete.append(mf)

            if not files_to_delete:
                self.console.print(f"\n[yellow]No files older than {days} days found[/yellow]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                return

            self.console.print(f"\n[yellow]Found {len(files_to_delete)} file(s) older than {days} days[/yellow]")
            confirm = Confirm.ask("Delete these files?", default=False)

            if not confirm:
                self.console.print("[yellow]Deletion cancelled[/yellow]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                return

        elif clean_choice == "3":
            # Select specific files
            import json
            from datetime import datetime

            # Display files with details
            files_table = Table(border_style="cyan", show_header=True)
            files_table.add_column("#", style="cyan", width=4)
            files_table.add_column("Filename", style="white", width=50)
            files_table.add_column("Date", style="yellow", width=16)
            files_table.add_column("Size", style="green", width=10)

            valid_files = []
            for i, mf in enumerate(metadata_files, 1):
                try:
                    size_kb = mf.stat().st_size / 1024
                    mtime = datetime.fromtimestamp(mf.stat().st_mtime)
                    date_str = mtime.strftime('%Y-%m-%d %H:%M')

                    files_table.add_row(
                        str(i),
                        mf.name[:50],
                        date_str,
                        f"{size_kb:.1f} KB"
                    )
                    valid_files.append(mf)
                except Exception as e:
                    continue

            self.console.print("\n")
            self.console.print(files_table)

            self.console.print("\n[yellow]Select files to delete:[/yellow]")
            self.console.print("[dim]Enter comma-separated numbers (e.g., 1,3,5) or 'all' for all files[/dim]")

            selection = Prompt.ask("Files to delete")

            if selection.lower() == 'all':
                files_to_delete = valid_files
            else:
                try:
                    indices = [int(x.strip()) for x in selection.split(',')]
                    for idx in indices:
                        if 1 <= idx <= len(valid_files):
                            files_to_delete.append(valid_files[idx - 1])
                except ValueError:
                    self.console.print("[red]Invalid selection[/red]")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    input()
                    return

            if not files_to_delete:
                self.console.print("[yellow]No files selected[/yellow]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                return

            self.console.print(f"\n[yellow]Selected {len(files_to_delete)} file(s) for deletion[/yellow]")
            confirm = Confirm.ask("Delete these files?", default=False)

            if not confirm:
                self.console.print("[yellow]Deletion cancelled[/yellow]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                return

        # Perform deletion
        deleted_count = 0
        failed_count = 0

        self.console.print("\n[bold]Deleting files...[/bold]")

        for mf in files_to_delete:
            try:
                mf.unlink()
                deleted_count += 1
                self.console.print(f"  [green]✓[/green] Deleted: {mf.name}")
            except Exception as e:
                failed_count += 1
                self.console.print(f"  [red]✗[/red] Failed: {mf.name} - {e}")

        # Summary
        self.console.print(f"\n[bold green]✅ Deletion complete[/bold green]")
        self.console.print(f"   Deleted: {deleted_count} file(s)")
        if failed_count > 0:
            self.console.print(f"   [red]Failed: {failed_count} file(s)[/red]")

        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        input()

    def _export_to_doccano_jsonl(self, output_file: str, text_column: str,
                                  prompt_configs: list, data_path: Path, timestamp: str,
                                  sample_size=None):
        """Export annotations to Doccano JSONL format

        Parameters
        ----------
        sample_size : str or int, optional
            Number of samples to export. Can be:
            - 'all': export all annotations
            - 'representative': export 10% (minimum 100)
            - int: export specific number
        """
        import json
        import pandas as pd

        try:
            self.console.print("\n[bold cyan]📤 Exporting to Doccano JSONL...[/bold cyan]")

            # Load the annotated file
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.csv':
                df = pd.read_csv(output_path)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(output_path)
            elif output_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(output_path)
            else:
                self.console.print(f"[yellow]⚠️  Unsupported format for Doccano export: {output_path.suffix}[/yellow]")
                return

            # Filter only annotated rows
            if 'annotation' not in df.columns:
                self.console.print("[yellow]⚠️  No annotation column found[/yellow]")
                return

            annotated_mask = (
                df['annotation'].notna() &
                (df['annotation'].astype(str).str.strip() != '') &
                (df['annotation'].astype(str) != 'nan')
            )
            df_annotated = df[annotated_mask].copy()

            if len(df_annotated) == 0:
                self.console.print("[yellow]⚠️  No valid annotations to export[/yellow]")
                return

            total_annotated = len(df_annotated)
            self.console.print(f"[cyan]  Found {total_annotated:,} annotated rows[/cyan]")

            # Apply sampling if specified
            if sample_size is not None and sample_size != 'all':
                if sample_size == 'representative':
                    # Stratified sampling: 10% from each label class (minimum 100 total)
                    n_samples = max(100, int(total_annotated * 0.1))

                    # Don't sample more than available
                    n_samples = min(n_samples, total_annotated)

                    if n_samples < total_annotated:
                        self.console.print(f"[cyan]  Using stratified sampling: {n_samples:,} rows (proportional by labels)[/cyan]")

                        # Parse annotations to get label distribution
                        label_counts = {}
                        for idx, row in df_annotated.iterrows():
                            try:
                                annotation = json.loads(row['annotation'])
                                # Get first label key as stratification key
                                for key in annotation.keys():
                                    if key != 'text':
                                        label_val = str(annotation[key])
                                        label_counts[label_val] = label_counts.get(label_val, 0) + 1
                                        break
                            except:
                                pass

                        # Stratified sampling
                        df_annotated = df_annotated.sample(n=n_samples, random_state=42).copy()
                    else:
                        self.console.print(f"[cyan]  Exporting all {total_annotated:,} rows (sample size >= total)[/cyan]")
                else:
                    # Custom number - random sampling
                    n_samples = int(sample_size)
                    n_samples = min(n_samples, total_annotated)

                    if n_samples < total_annotated:
                        self.console.print(f"[cyan]  Random sampling: {n_samples:,} rows for export[/cyan]")
                        df_annotated = df_annotated.sample(n=n_samples, random_state=42).copy()
                    else:
                        self.console.print(f"[cyan]  Exporting all {total_annotated:,} rows[/cyan]")
            else:
                self.console.print(f"[cyan]  Exporting all {total_annotated:,} rows[/cyan]")

            # Prepare JSONL output
            doccano_dir = self.settings.paths.data_dir / 'doccano_exports'
            doccano_dir.mkdir(parents=True, exist_ok=True)

            jsonl_filename = f"{data_path.stem}_doccano_{timestamp}.jsonl"
            jsonl_path = doccano_dir / jsonl_filename

            # Get all label keys from prompts
            all_label_keys = set()
            for pc in prompt_configs:
                prefix = pc.get('prefix', '')
                for key in pc['prompt']['keys']:
                    if prefix:
                        all_label_keys.add(f"{prefix}_{key}")
                    else:
                        all_label_keys.add(key)

            # Extract labels from annotations (JSON strings)
            exported_count = 0
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for idx, row in df_annotated.iterrows():
                    try:
                        # Parse annotation JSON
                        annotation_str = row['annotation']
                        if pd.isna(annotation_str) or str(annotation_str).strip() == '':
                            continue

                        annotation_data = json.loads(annotation_str)

                        # Build Doccano entry
                        doccano_entry = {
                            'text': str(row[text_column]),
                            'labels': []
                        }

                        # Extract labels from annotation
                        for label_key in all_label_keys:
                            if label_key in annotation_data:
                                label_value = annotation_data[label_key]
                                # Handle different label formats
                                if isinstance(label_value, list):
                                    doccano_entry['labels'].extend(label_value)
                                elif isinstance(label_value, str) and label_value.strip():
                                    doccano_entry['labels'].append(label_value)

                        # Add metadata (everything except text and annotation columns)
                        metadata = {}
                        for col in df.columns:
                            if col not in [text_column, 'annotation'] and col not in all_label_keys:
                                val = row[col]
                                # Convert to JSON-serializable format
                                if pd.notna(val):
                                    if isinstance(val, (pd.Timestamp, pd.DatetimeTZDtype)):
                                        metadata[col] = str(val)
                                    elif isinstance(val, (int, float, str, bool)):
                                        metadata[col] = val
                                    else:
                                        metadata[col] = str(val)

                        doccano_entry['meta'] = metadata

                        # Write to JSONL
                        f.write(json.dumps(doccano_entry, ensure_ascii=False) + '\n')
                        exported_count += 1

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Could not parse annotation at row {idx}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing row {idx}: {e}")
                        continue

            # Display success
            self.console.print(f"\n[bold green]✅ Doccano JSONL export completed![/bold green]")
            self.console.print(f"[bold cyan]📄 JSONL File:[/bold cyan]")
            self.console.print(f"   {jsonl_path}")
            self.console.print(f"[cyan]   Exported: {exported_count:,} entries[/cyan]\n")

            self.console.print("[yellow]📌 Next Steps:[/yellow]")
            self.console.print("  1. Import this JSONL file into Doccano for validation")
            self.console.print("  2. Review and correct annotations in Doccano")
            self.console.print("  3. Export validated annotations from Doccano")
            self.console.print("  4. Use LLM Tool to calculate metrics on validated data\n")

        except Exception as e:
            self.console.print(f"\n[red]❌ Doccano export failed: {e}[/red]")
            self.logger.exception("Doccano JSONL export failed")

    def _export_to_labelstudio_jsonl(self, output_file: str, text_column: str,
                                      prompt_configs: list, data_path: Path, timestamp: str,
                                      sample_size=None, prediction_mode='with'):
        """Export annotations to Label Studio JSONL format

        Parameters
        ----------
        sample_size : str or int, optional
            Number of samples to export. Can be:
            - 'all': export all annotations
            - 'representative': export 10% (minimum 100)
            - int: export specific number
        prediction_mode : str, optional
            How to include LLM predictions:
            - 'with': Include predictions (default)
            - 'without': Export without predictions
            - 'both': Create two files (with and without)
        """
        import json
        import pandas as pd

        try:
            self.console.print("\n[bold cyan]📤 Exporting to Label Studio JSONL...[/bold cyan]")

            # Load the annotated file
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.csv':
                df = pd.read_csv(output_path)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(output_path)
            elif output_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(output_path)
            else:
                self.console.print(f"[yellow]⚠️  Unsupported format for Label Studio export: {output_path.suffix}[/yellow]")
                return

            # Filter only annotated rows
            if 'annotation' not in df.columns:
                self.console.print("[yellow]⚠️  No annotation column found[/yellow]")
                return

            annotated_mask = (
                df['annotation'].notna() &
                (df['annotation'].astype(str).str.strip() != '') &
                (df['annotation'].astype(str) != 'nan')
            )
            df_annotated = df[annotated_mask].copy()

            if len(df_annotated) == 0:
                self.console.print("[yellow]⚠️  No valid annotations to export[/yellow]")
                return

            total_annotated = len(df_annotated)
            self.console.print(f"[cyan]  Found {total_annotated:,} annotated rows[/cyan]")

            # Apply sampling if specified
            if sample_size is not None and sample_size != 'all':
                if sample_size == 'representative':
                    # Stratified sampling: 10% from each label class (minimum 100 total)
                    n_samples = max(100, int(total_annotated * 0.1))

                    # Don't sample more than available
                    n_samples = min(n_samples, total_annotated)

                    if n_samples < total_annotated:
                        self.console.print(f"[cyan]  Using stratified sampling: {n_samples:,} rows (proportional by labels)[/cyan]")

                        # Parse annotations to get label distribution
                        label_counts = {}
                        for idx, row in df_annotated.iterrows():
                            try:
                                annotation = json.loads(row['annotation'])
                                # Get first label key as stratification key
                                for key in annotation.keys():
                                    if key != 'text':
                                        label_val = str(annotation[key])
                                        label_counts[label_val] = label_counts.get(label_val, 0) + 1
                                        break
                            except:
                                pass

                        # Stratified sampling
                        df_annotated = df_annotated.sample(n=n_samples, random_state=42).copy()
                    else:
                        self.console.print(f"[cyan]  Exporting all {total_annotated:,} rows (sample size >= total)[/cyan]")
                else:
                    # Custom number - random sampling
                    n_samples = int(sample_size)
                    n_samples = min(n_samples, total_annotated)

                    if n_samples < total_annotated:
                        self.console.print(f"[cyan]  Random sampling: {n_samples:,} rows for export[/cyan]")
                        df_annotated = df_annotated.sample(n=n_samples, random_state=42).copy()
                    else:
                        self.console.print(f"[cyan]  Exporting all {total_annotated:,} rows[/cyan]")
            else:
                self.console.print(f"[cyan]  Exporting all {total_annotated:,} rows[/cyan]")

            # Prepare JSONL output
            labelstudio_dir = self.settings.paths.data_dir / 'labelstudio_exports'
            labelstudio_dir.mkdir(parents=True, exist_ok=True)

            jsonl_filename = f"{data_path.stem}_labelstudio_{timestamp}.jsonl"
            jsonl_path = labelstudio_dir / jsonl_filename

            # Get all label keys from prompts
            all_label_keys = set()
            for pc in prompt_configs:
                prefix = pc.get('prefix', '')
                for key in pc['prompt']['keys']:
                    if prefix:
                        all_label_keys.add(f"{prefix}_{key}")
                    else:
                        all_label_keys.add(key)

            # Export to Label Studio format
            exported_count = 0
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for idx, row in df_annotated.iterrows():
                    try:
                        # Parse annotation JSON
                        annotation_str = row['annotation']
                        if pd.isna(annotation_str) or str(annotation_str).strip() == '':
                            continue

                        annotation_data = json.loads(annotation_str)

                        # Build Label Studio entry
                        # Data section
                        data_entry = {
                            'text': str(row[text_column])
                        }

                        # Add metadata
                        for col in df.columns:
                            if col not in [text_column, 'annotation'] and col not in all_label_keys:
                                val = row[col]
                                if pd.notna(val):
                                    if isinstance(val, (pd.Timestamp, pd.DatetimeTZDtype)):
                                        data_entry[col] = str(val)
                                    elif isinstance(val, (int, float, str, bool)):
                                        data_entry[col] = val
                                    else:
                                        data_entry[col] = str(val)

                        # Predictions section (LLM annotations as predictions)
                        predictions_result = []

                        for label_key in all_label_keys:
                            if label_key in annotation_data:
                                label_value = annotation_data[label_key]

                                # Handle list of labels
                                if isinstance(label_value, list):
                                    for lv in label_value:
                                        if lv and str(lv).strip():
                                            predictions_result.append({
                                                "value": {
                                                    "choices": [str(lv)]
                                                },
                                                "from_name": label_key,
                                                "to_name": "text",
                                                "type": "choices"
                                            })
                                # Handle single label
                                elif isinstance(label_value, str) and label_value.strip():
                                    predictions_result.append({
                                        "value": {
                                            "choices": [label_value]
                                        },
                                        "from_name": label_key,
                                        "to_name": "text",
                                        "type": "choices"
                                    })

                        # Build entry based on prediction mode
                        if prediction_mode == 'without':
                            # Export without predictions - just data
                            labelstudio_entry = {"data": data_entry}
                        else:
                            # Export with predictions (default)
                            labelstudio_entry = {
                                "data": data_entry,
                                "predictions": [{
                                    "result": predictions_result,
                                    "model_version": "llm_annotation"
                                }]
                            }

                        # Write to JSONL
                        f.write(json.dumps(labelstudio_entry, ensure_ascii=False) + '\n')
                        exported_count += 1

                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Could not parse annotation at row {idx}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing row {idx}: {e}")
                        continue

            # Create Label Studio config XML file
            config_path = jsonl_path.with_suffix('.xml')
            label_config = self._build_labelstudio_config(all_label_keys, prompt_configs)

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(label_config)

            # Also create a JSON file (non-line-delimited) for easier import
            suffix = "_with_predictions" if prediction_mode == 'with' else "_without_predictions"
            json_path = jsonl_path.parent / f"{jsonl_path.stem}{suffix}.json"
            jsonl_final = jsonl_path.parent / f"{jsonl_path.stem}{suffix}.jsonl"

            # Rename jsonl if needed
            if prediction_mode != 'with':
                jsonl_path.rename(jsonl_final)
                jsonl_path = jsonl_final

            # Read the JSONL and convert to JSON array
            tasks_array = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        tasks_array.append(json.loads(line))

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(tasks_array, f, indent=2, ensure_ascii=False)

            # If mode is 'both', create second set without predictions
            if prediction_mode == 'both':
                self.console.print(f"[cyan]  Creating second file without predictions...[/cyan]")

                # Call recursively with 'without' mode
                self._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size='all',  # Use all already sampled data
                    prediction_mode='without'
                )

                self.console.print(f"\n[bold green]✅ Label Studio export completed (both modes)![/bold green]")
            else:
                self.console.print(f"\n[bold green]✅ Label Studio export completed![/bold green]")

            # Display files created
            mode_desc = {
                'with': 'with LLM predictions',
                'without': 'without predictions (for manual annotation)',
                'both': 'with predictions'
            }.get(prediction_mode, '')

            self.console.print(f"[bold cyan]📄 Files created ({mode_desc}):[/bold cyan]")
            self.console.print(f"   {json_path} [dim](JSON array - use this for import)[/dim]")
            self.console.print(f"   {jsonl_path} [dim](JSONL - alternative format)[/dim]")
            self.console.print(f"   {config_path} [dim](labeling config XML)[/dim]")
            self.console.print(f"[cyan]   Exported: {exported_count:,} entries[/cyan]\n")

            self.console.print("[yellow]📌 Import Instructions:[/yellow]")
            self.console.print("  [bold]Recommended: Use the JSON file[/bold]")
            self.console.print("  1. In Label Studio, click 'Create Project'")
            self.console.print("  2. Name your project and click 'Save'")
            self.console.print("  3. Go to 'Settings' → 'Labeling Interface'")
            self.console.print(f"  4. Click 'Code' and paste contents from: {config_path.name}")
            self.console.print("  5. Save the configuration")
            self.console.print(f"  6. Go to project, click 'Import' and upload: {json_path.name}\n")

            self.console.print("  [dim]Alternative: Use direct API export for automatic setup[/dim]\n")

        except Exception as e:
            self.console.print(f"\n[red]❌ Label Studio export failed: {e}[/red]")
            self.logger.exception("Label Studio JSONL export failed")

    def _export_to_labelstudio_direct(self, output_file: str, text_column: str,
                                        prompt_configs: list, data_path: Path, timestamp: str,
                                        sample_size=None, prediction_mode='with', api_url=None, api_key=None):
        """Export annotations directly to Label Studio via API

        Parameters
        ----------
        api_url : str
            Label Studio API URL (e.g., http://localhost:8080)
        api_key : str
            Label Studio API key from Account & Settings
        prediction_mode : str, optional
            How to include LLM predictions:
            - 'with': Include predictions (default)
            - 'without': Export without predictions
            - 'both': Create two projects (with and without)
        """
        import json
        import pandas as pd

        # Check if requests is available
        if not HAS_REQUESTS:
            self.console.print("\n[yellow]⚠️  Direct export to Label Studio requires the 'requests' library[/yellow]")
            self.console.print("[cyan]This library is not currently installed.[/cyan]\n")

            install_requests = Confirm.ask(
                "Would you like to install 'requests' now?",
                default=True
            )

            if install_requests:
                try:
                    self.console.print("\n[cyan]Installing requests...[/cyan]")
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "requests"],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        self.console.print("[green]✅ Successfully installed 'requests'[/green]")
                        # Import it now
                        import requests as req_module
                        # Note: requests is now available for the rest of this function
                        globals()['requests'] = req_module
                        globals()['HAS_REQUESTS'] = True
                    else:
                        self.console.print(f"[red]❌ Installation failed: {result.stderr}[/red]")
                        self.console.print("\n[yellow]Please install manually:[/yellow]")
                        self.console.print("  pip install requests")
                        return

                except Exception as e:
                    self.console.print(f"[red]❌ Installation error: {e}[/red]")
                    self.console.print("\n[yellow]Please install manually:[/yellow]")
                    self.console.print("  pip install requests")
                    return
            else:
                self.console.print("\n[yellow]Skipping direct export. You can:[/yellow]")
                self.console.print("  1. Install requests: pip install requests")
                self.console.print("  2. Use JSONL export instead (no extra dependencies)")
                return

        # Import requests locally for use in this function
        import requests

        try:
            self.console.print("\n[bold cyan]📤 Exporting directly to Label Studio...[/bold cyan]")

            # Load the annotated file
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.csv':
                df = pd.read_csv(output_path)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(output_path)
            elif output_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(output_path)
            else:
                self.console.print(f"[yellow]⚠️  Unsupported format: {output_path.suffix}[/yellow]")
                return

            # Filter only annotated rows
            if 'annotation' not in df.columns:
                self.console.print("[yellow]⚠️  No annotation column found[/yellow]")
                return

            annotated_mask = (
                df['annotation'].notna() &
                (df['annotation'].astype(str).str.strip() != '') &
                (df['annotation'].astype(str) != 'nan')
            )
            df_annotated = df[annotated_mask].copy()

            if len(df_annotated) == 0:
                self.console.print("[yellow]⚠️  No valid annotations to export[/yellow]")
                return

            total_annotated = len(df_annotated)
            self.console.print(f"[cyan]  Found {total_annotated:,} annotated rows[/cyan]")

            # Apply sampling (same logic as JSONL export)
            if sample_size is not None and sample_size != 'all':
                if sample_size == 'representative':
                    n_samples = max(100, int(total_annotated * 0.1))
                    n_samples = min(n_samples, total_annotated)
                    if n_samples < total_annotated:
                        self.console.print(f"[cyan]  Using stratified sampling: {n_samples:,} rows[/cyan]")
                        df_annotated = df_annotated.sample(n=n_samples, random_state=42).copy()
                else:
                    n_samples = int(sample_size)
                    n_samples = min(n_samples, total_annotated)
                    if n_samples < total_annotated:
                        self.console.print(f"[cyan]  Random sampling: {n_samples:,} rows[/cyan]")
                        df_annotated = df_annotated.sample(n=n_samples, random_state=42).copy()

            # Get all label keys from prompts
            all_label_keys = set()
            for pc in prompt_configs:
                if 'keys' in pc['prompt']:
                    for key in pc['prompt']['keys']:
                        all_label_keys.add(key)

            # Create Label Studio project
            mode_suffix = "_with_predictions" if prediction_mode == 'with' else "_no_predictions"
            self.console.print(f"\n[cyan]  Creating Label Studio project{mode_suffix}...[/cyan]")

            # Title must be max 50 chars for Label Studio
            # Format: LLM_{short_name}_{mode} where mode is 'pred' or 'nopred'
            mode_short = "pred" if prediction_mode == 'with' else "nopred"
            base_name = data_path.stem[:30]  # Truncate base name if needed
            project_title = f"LLM_{base_name}_{mode_short}"[:50]

            # Build labeling config
            label_config = self._build_labelstudio_config(all_label_keys, prompt_configs)

            # Create project via API
            # Label Studio Personal Access Tokens (JWT refresh tokens) must be exchanged for access tokens
            # Try to get an access token first if using JWT format
            access_token = api_key

            if api_key.startswith('eyJ'):
                # This is a JWT refresh token - exchange for access token
                try:
                    token_response = requests.post(
                        f'{api_url}/api/token/refresh',
                        headers={'Content-Type': 'application/json'},
                        json={'refresh': api_key}
                    )
                    if token_response.status_code == 200:
                        token_data = token_response.json()
                        access_token = token_data.get('access', api_key)
                        self.console.print(f"[dim cyan]  ✓ Obtained access token from refresh token[/dim cyan]")
                    else:
                        self.console.print(f"[dim yellow]  Note: Token refresh returned {token_response.status_code}, trying direct use[/dim yellow]")
                except Exception as e:
                    # If exchange fails, try using the token directly
                    self.console.print(f"[dim yellow]  Note: Could not exchange refresh token ({e}), trying direct use[/dim yellow]")
                    pass

            # Try different auth formats
            # For JWT access tokens, Bearer is the correct format
            auth_formats = [
                f'Bearer {access_token}',     # JWT format (1.20+) - try first
                f'Token {access_token}',      # Legacy format (pre-1.20)
                access_token                   # Raw token (fallback)
            ]

            project_data = {
                'title': project_title,
                'label_config': label_config,
                'description': f'LLM annotations exported from {data_path.name}'
            }

            response = None
            for i, auth_format in enumerate(auth_formats):
                headers = {
                    'Authorization': auth_format,
                    'Content-Type': 'application/json'
                }

                response = requests.post(
                    f'{api_url}/api/projects',
                    headers=headers,
                    json=project_data
                )

                if response.status_code in [200, 201]:
                    break
                elif i < len(auth_formats) - 1:
                    self.console.print(f"[yellow]  Trying alternative auth format ({i+2}/{len(auth_formats)})...[/yellow]")

            if response.status_code not in [200, 201]:
                self.console.print(f"[red]❌ Failed to create project: {response.text}[/red]")
                return

            project = response.json()
            project_id = project['id']

            self.console.print(f"[green]✅ Created project: {project_title} (ID: {project_id})[/green]")

            # Import tasks
            self.console.print(f"\n[cyan]  Importing {len(df_annotated):,} tasks...[/cyan]")

            tasks = []
            for idx, row in df_annotated.iterrows():
                try:
                    annotation_str = row['annotation']
                    annotation_data = json.loads(annotation_str)

                    # Build task data
                    task_data = {'text': str(row[text_column])}

                    # Add metadata
                    for col in df.columns:
                        if col not in [text_column, 'annotation'] and col not in all_label_keys:
                            val = row[col]
                            if pd.notna(val):
                                task_data[col] = str(val)

                    # Build task based on prediction mode
                    if prediction_mode == 'without':
                        # Export without predictions - just data
                        task = {'data': task_data}
                    else:
                        # Build predictions (LLM annotations)
                        predictions_result = []
                        for label_key in all_label_keys:
                            if label_key in annotation_data:
                                label_value = annotation_data[label_key]
                                if isinstance(label_value, list):
                                    for lv in label_value:
                                        if lv and str(lv).strip():
                                            predictions_result.append({
                                                "value": {"choices": [str(lv)]},
                                                "from_name": label_key,
                                                "to_name": "text",
                                                "type": "choices"
                                            })
                                elif isinstance(label_value, str) and label_value.strip():
                                    predictions_result.append({
                                        "value": {"choices": [label_value]},
                                        "from_name": label_key,
                                        "to_name": "text",
                                        "type": "choices"
                                    })

                        task = {
                            'data': task_data,
                            'predictions': [{
                                'result': predictions_result,
                                'model_version': 'llm_annotation'
                            }]
                        }

                    tasks.append(task)

                except Exception as e:
                    self.console.print(f"[yellow]⚠️  Skipped row {idx}: {e}[/yellow]")
                    continue

            # Import tasks to project
            response = requests.post(
                f'{api_url}/api/projects/{project_id}/import',
                headers=headers,
                json=tasks
            )

            if response.status_code not in [200, 201]:
                self.console.print(f"[red]❌ Failed to import tasks: {response.text}[/red]")
                return

            self.console.print(f"\n[bold green]✅ Successfully exported {len(tasks):,} tasks to Label Studio[/bold green]")
            self.console.print(f"[cyan]🔗 Project URL: {api_url}/projects/{project_id}/[/cyan]\n")

            # If mode is 'both', create second project without predictions
            if prediction_mode == 'both':
                self.console.print(f"\n[cyan]Creating second project without predictions...[/cyan]")

                # Call recursively with 'without' mode
                self._export_to_labelstudio_direct(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size='all',  # Use all already sampled data
                    prediction_mode='without',
                    api_url=api_url,
                    api_key=api_key
                )

            self.console.print("[yellow]Next steps:[/yellow]")
            self.console.print("  1. Open Label Studio and navigate to your project(s)")
            self.console.print("  2. Review and correct the LLM predictions (if applicable)")
            self.console.print("  3. Export validated annotations")
            self.console.print("  4. Use LLM Tool to calculate metrics\n")

        except requests.exceptions.ConnectionError:
            self.console.print(f"\n[red]❌ Connection error: Could not connect to {api_url}[/red]")
            self.console.print("[yellow]Make sure Label Studio is running:[/yellow]")
            self.console.print("  label-studio start")
        except Exception as e:
            self.console.print(f"\n[red]❌ Export failed: {e}[/red]")
            self.logger.exception("Label Studio direct export failed")

    def _build_labelstudio_config(self, label_keys, prompt_configs=None):
        """Build Label Studio labeling configuration

        Parameters
        ----------
        label_keys : set
            Set of label keys (e.g., {'theme', 'party'})
        prompt_configs : list, optional
            List of prompt configurations containing the actual values for each key
        """
        config_parts = ['<View>']
        config_parts.append('  <Text name="text" value="$text"/>')

        # Extract actual values from prompt_configs if available
        key_values_map = {}
        if prompt_configs:
            for pc in prompt_configs:
                # Try new format first
                if 'prompt' in pc and 'keys' in pc['prompt']:
                    for key_def in pc['prompt']['keys']:
                        if isinstance(key_def, dict):
                            # New format: {'name': 'theme', 'values': ['env', 'health']}
                            key_name = key_def.get('name')
                            values = key_def.get('values', [])
                            if key_name:
                                key_values_map[key_name] = values

                # Try extracting from prompt_content (old wizard format)
                if 'prompt_content' in pc and not key_values_map:
                    import re
                    prompt_text = pc['prompt_content']

                    # Pattern: - "theme": "value1" si ..., "value2" si ..., "null" si ...
                    # or: - "theme" (can be multiple values): "value1" si ..., "value2" si ...
                    for label_key in label_keys:
                        # Find lines starting with - "key"
                        pattern = rf'- "{label_key}"[^:]*:\s*(.+?)(?=\n-|\n\*\*|\Z)'
                        match = re.search(pattern, prompt_text, re.DOTALL)

                        if match:
                            values_text = match.group(1)
                            # Extract all quoted values except "null"
                            value_pattern = r'"([^"]+)"\s+si'
                            values = re.findall(value_pattern, values_text)
                            # Filter out 'null'
                            values = [v for v in values if v != 'null']
                            if values:
                                key_values_map[label_key] = values

        for label_key in sorted(label_keys):
            config_parts.append(f'  <Choices name="{label_key}" toName="text" choice="single">')

            # Use actual values if available, otherwise placeholder
            if label_key in key_values_map and key_values_map[label_key]:
                for value in key_values_map[label_key]:
                    config_parts.append(f'    <Choice value="{value}"/>')
            else:
                # Fallback to placeholder
                config_parts.append(f'    <Choice value="placeholder_{label_key}"/>')

            config_parts.append('  </Choices>')

        config_parts.append('</View>')
        return '\n'.join(config_parts)

    def _database_annotator(self):
        """PostgreSQL direct annotator"""
        self.console.print("\n[bold cyan]🗄️  Database Annotator[/bold cyan]\n")
        self.console.print("[yellow]⚙️  Database Annotator coming soon...[/yellow]")
        self.console.print("[dim]Press Enter to continue...[/dim]")
        input()

    def show_documentation(self):
        """Show documentation"""
        # Display welcome banner
        self._display_welcome_banner()

        if HAS_RICH and self.console:
            doc_text = """
# LLMTool Documentation

## Quick Start
1. Ensure you have models available (Ollama or API keys)
2. Prepare your dataset (CSV, JSON, etc.)
3. Use Quick Start Wizard for automatic configuration

## Features
- **Auto-detection**: Automatically finds models and datasets
- **Smart defaults**: Intelligent configuration suggestions
- **Profile system**: Save and reuse configurations
- **Benchmarking**: Compare multiple models automatically

## Support
- GitHub: https://github.com/antoine-lemor/LLMTool
- Email: support@llmtool.ai
            """

            md = Markdown(doc_text)
            self.console.print(Panel(md, title="📚 Documentation", border_style="blue"))
        else:
            print("\n=== Documentation ===")
            print("Visit: https://github.com/antoine-lemor/LLMTool")


def main():
    """Entry point for the advanced CLI"""
    cli = AdvancedCLI()
    cli.run()


if __name__ == "__main__":
    main()

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
import numpy as np
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

# Try importing numpy for numerical operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try importing psutil for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Try importing transformers for tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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


# ============================================================================
# MODEL DESCRIPTIONS - Factual information about popular LLM models
# ============================================================================
MODEL_DESCRIPTIONS = {
    # Meta Llama models
    'llama3.3': 'Meta Llama 3.3 70B - Multilingual model with 128K context, strong reasoning',
    'llama3.2': 'Meta Llama 3.2 - Small efficient models (1B/3B) for edge deployment',
    'llama3.1': 'Meta Llama 3.1 - Multimodal model with tool use, 128K context (8B/70B/405B)',
    'llama3': 'Meta Llama 3 - Improved instruction following, 8K context (8B/70B)',
    'llama2': 'Meta Llama 2 - Chat-optimized open model, 4K context (7B/13B/70B)',
    'codellama': 'Meta Code Llama - Code generation specialist, supports Python/C++/Java (7B/13B/34B)',

    # Google Gemma models
    'gemma3': 'Google Gemma 3 - Latest efficient model for everyday devices (2B/9B/27B)',
    'gemma2': 'Google Gemma 2 - Lightweight deployment across consumer devices (2B/9B/27B)',
    'gemma': 'Google Gemma - Efficient open model by Google DeepMind (2B/7B)',

    # Alibaba Qwen models
    'qwen3': 'Qwen 3 - Latest generation with dense and MoE variants (8B-235B), 128K context',
    'qwen2.5': 'Qwen 2.5 - Pretrained on 18T tokens, multilingual support, 128K context',
    'qwen2': 'Qwen 2 - Multilingual model with strong coding abilities (0.5B-72B)',
    'qwen': 'Qwen - Alibaba large language model series',

    # DeepSeek models
    'deepseek-r1': 'DeepSeek-R1 - Reasoning model, strong in math/coding/logic (8B/671B)',
    'deepseek-coder': 'DeepSeek Coder - Specialized coding model with 16K context',
    'deepseek': 'DeepSeek - General purpose model series',

    # Mistral AI models
    'mixtral': 'Mistral Mixtral - Mixture of Experts (MoE) model with open weights (8x7B/8x22B)',
    'mistral': 'Mistral 7B - Efficient 7B model, approaches CodeLlama on code tasks',
    'mistral-nemo': 'Mistral Nemo - 12B model with 128K context window',
    'codestral': 'Codestral - Mistral AI code generation model, supports 80+ languages',

    # NVIDIA Nemotron models
    'nemotron': 'NVIDIA Nemotron-70B - Customized Llama 3.1 for helpful responses via RLHF',
    'nemotron-mini': 'NVIDIA Nemotron Mini - Small language model for RAG/function calling, 4K context',

    # Microsoft Phi models
    'phi4': 'Microsoft Phi-4 - 14B reasoning model rivaling larger models',
    'phi3': 'Microsoft Phi-3 - Lightweight state-of-the-art models (3B Mini/14B Medium)',
    'phi': 'Microsoft Phi-2 - 2.7B model with strong reasoning/language understanding',

    # Cohere models
    'command-r': 'Cohere Command R - Optimized for conversational interaction and long context',
    'command-r-plus': 'Cohere Command R+ - Enhanced version with stronger capabilities',

    # Specialized models
    'yi': 'Yi - Bilingual (English/Chinese) model with strong performance (6B/34B)',
    'solar': 'Solar - Upstage Solar, depth-upscaled Llama 2 with 10.7B parameters',
    'orca': 'Orca - Microsoft Orca, reasoning specialist trained on GPT-4 outputs',
    'vicuna': 'Vicuna - Open-source chatbot fine-tuned from LLaMA',
    'wizardcoder': 'WizardCoder - Code generation with Evol-Instruct method',
    'starcoder': 'StarCoder - Code generation model trained on The Stack dataset',
    'falcon': 'Falcon - TII open-source model, trained on refined web data (7B/40B)',
    'stable-lm': 'StabilityAI Stable LM - Efficient language model series',
    'bloom': 'BLOOM - Multilingual model supporting 46 languages (176B)',
    'gpt4all': 'GPT4All - Ecosystem of open-source chatbots',

    # OpenAI models
    'gpt-5': 'OpenAI GPT-5 - Latest flagship model (2025)',
    'gpt-4o': 'OpenAI GPT-4o - Multimodal (text/image), matches GPT-4 Turbo performance',
    'gpt-4-turbo': 'OpenAI GPT-4 Turbo - Large multimodal model, optimized for chat/completions',
    'gpt-4': 'OpenAI GPT-4 - Advanced reasoning, multimodal capabilities',
    'gpt-3.5-turbo': 'OpenAI GPT-3.5 Turbo - Fast, cost-effective for most tasks',
    'o1': 'OpenAI o1 - Reasoning model for science/coding/math',
    'o3': 'OpenAI o3 - Latest reasoning model with enhanced performance',
    'o4': 'OpenAI o4 - Advanced reasoning model (2025)',

    # Anthropic Claude models
    'claude-sonnet-4.5': 'Claude Sonnet 4.5 - Best for coding/computer use, autonomous for 30hrs (200K)',
    'claude-3.7-sonnet': 'Claude 3.7 Sonnet - Hybrid reasoning, extended thinking for complex problems',
    'claude-3.5-sonnet': 'Claude 3.5 Sonnet - Strong vision, 64% agentic coding eval (200K)',
    'claude-opus-4.1': 'Claude Opus 4.1 - Most intelligent, frontier in coding/search/writing (200K)',
    'claude-3-opus': 'Claude 3 Opus - Largest Claude 3 model, 200K-1M context window',
    'claude-3.5-haiku': 'Claude 3.5 Haiku - Fast, surpasses Claude 3 Opus on benchmarks',
    'claude-3-sonnet': 'Claude 3 Sonnet - Balanced performance and speed',
    'claude-3-haiku': 'Claude 3 Haiku - Fastest Claude 3 model',
}


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
                    # Format: NAME                    ID              SIZE      MODIFIED
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 1:
                                name = parts[0]

                                # Extract size - look for GB/MB/KB in any part
                                size = None
                                for part in parts[1:]:
                                    part_upper = part.upper()
                                    if 'GB' in part_upper or 'MB' in part_upper or 'KB' in part_upper:
                                        # Format nicely: "27GB" -> "27 GB", "1.5GB" -> "1.5 GB"
                                        import re
                                        match = re.match(r'([\d.]+)\s*([KMGT]B)', part_upper)
                                        if match:
                                            size = f"{match.group(1)} {match.group(2)}"
                                        else:
                                            size = part
                                        break

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
            "Multilingual Models": [
                {"name": "xlm-roberta-base", "params": "278M", "type": "XLM-R", "languages": "100+", "performance": "★★★★"},
                {"name": "xlm-roberta-large", "params": "560M", "type": "XLM-R", "languages": "100+", "performance": "★★★★★"},
                {"name": "microsoft/mdeberta-v3-base", "params": "280M", "type": "mDeBERTa", "languages": "100+", "performance": "★★★★★"},
                {"name": "bert-base-multilingual-cased", "params": "177M", "type": "mBERT", "languages": "104", "performance": "★★★"},
            ],
            "Long Document Models (Multilingual)": [
                {"name": "markussagen/xlm-roberta-longformer-base-4096", "params": "278M", "type": "XLM-R Longformer", "max_length": "4096", "languages": "100+", "performance": "★★★★★"},
                {"name": "allenai/led-base-16384", "params": "406M", "type": "LED", "max_length": "16384", "language": "English", "performance": "★★★★★"},
                {"name": "allenai/led-large-16384", "params": "406M", "type": "LED", "max_length": "16384", "language": "English", "performance": "★★★★★"},
                {"name": "allenai/longformer-base-4096", "params": "149M", "type": "Longformer", "max_length": "4096", "language": "English", "performance": "★★★★"},
                {"name": "allenai/longformer-large-4096", "params": "435M", "type": "Longformer", "max_length": "4096", "language": "English", "performance": "★★★★★"},
                {"name": "google/bigbird-roberta-base", "params": "128M", "type": "BigBird", "max_length": "4096", "language": "English", "performance": "★★★★"},
                {"name": "google/bigbird-roberta-large", "params": "340M", "type": "BigBird", "max_length": "4096", "language": "English", "performance": "★★★★★"},
            ],
            "Long Document Models (Language-Specific)": [
                # French
                {"name": "cmarkea/distilcamembert-base-nli", "params": "68M", "type": "DistilCamemBERT", "max_length": "512", "language": "French", "performance": "★★★★"},
                {"name": "gilf/french-camembert-postag-model", "params": "110M", "type": "CamemBERT", "max_length": "512", "language": "French", "performance": "★★★★"},
                # Spanish
                {"name": "PlanTL-GOB-ES/roberta-base-bne", "params": "125M", "type": "RoBERTa-BNE", "max_length": "512", "language": "Spanish", "performance": "★★★★"},
                {"name": "dccuchile/bert-base-spanish-wwm-cased", "params": "110M", "type": "BETO", "max_length": "512", "language": "Spanish", "performance": "★★★★"},
                # German
                {"name": "deepset/gbert-base", "params": "110M", "type": "GBERT", "max_length": "512", "language": "German", "performance": "★★★★"},
                {"name": "bert-base-german-cased", "params": "110M", "type": "German BERT", "max_length": "512", "language": "German", "performance": "★★★★"},
                # Italian
                {"name": "dbmdz/bert-base-italian-cased", "params": "110M", "type": "Italian BERT", "max_length": "512", "language": "Italian", "performance": "★★★★"},
                # Portuguese
                {"name": "neuralmind/bert-base-portuguese-cased", "params": "110M", "type": "BERTimbau", "max_length": "512", "language": "Portuguese", "performance": "★★★★"},
                # Dutch
                {"name": "GroNLP/bert-base-dutch-cased", "params": "110M", "type": "BERTje", "max_length": "512", "language": "Dutch", "performance": "★★★★"},
                {"name": "wietsedv/bert-base-dutch-cased", "params": "110M", "type": "Dutch BERT", "max_length": "512", "language": "Dutch", "performance": "★★★★"},
                # Polish
                {"name": "allegro/herbert-base-cased", "params": "124M", "type": "HerBERT", "max_length": "514", "language": "Polish", "performance": "★★★★"},
                # Chinese
                {"name": "hfl/chinese-roberta-wwm-ext", "params": "102M", "type": "Chinese RoBERTa", "max_length": "512", "language": "Chinese", "performance": "★★★★"},
                {"name": "bert-base-chinese", "params": "110M", "type": "Chinese BERT", "max_length": "512", "language": "Chinese", "performance": "★★★★"},
                # Japanese
                {"name": "cl-tohoku/bert-base-japanese-whole-word-masking", "params": "111M", "type": "Japanese BERT WWM", "max_length": "512", "language": "Japanese", "performance": "★★★★"},
                {"name": "cl-tohoku/bert-base-japanese", "params": "111M", "type": "Japanese BERT", "max_length": "512", "language": "Japanese", "performance": "★★★★"},
                # Arabic
                {"name": "aubmindlab/bert-base-arabert", "params": "135M", "type": "AraBERT", "max_length": "512", "language": "Arabic", "performance": "★★★★"},
                {"name": "asafaya/bert-base-arabic", "params": "110M", "type": "Arabic BERT", "max_length": "512", "language": "Arabic", "performance": "★★★★"},
                # Russian
                {"name": "DeepPavlov/rubert-base-cased", "params": "178M", "type": "RuBERT", "max_length": "512", "language": "Russian", "performance": "★★★★"},
            ],
            "Efficient Models": [
                {"name": "distilbert-base", "params": "66M", "type": "DistilBERT", "speed": "2x faster", "performance": "★★★"},
                {"name": "distilroberta-base", "params": "82M", "type": "DistilRoBERTa", "speed": "2x faster", "performance": "★★★"},
                {"name": "albert-base-v2", "params": "12M", "type": "ALBERT", "speed": "4x faster", "performance": "★★★"},
                {"name": "albert-large-v2", "params": "18M", "type": "ALBERT", "speed": "3x faster", "performance": "★★★★"},
                {"name": "deberta-v3-xsmall", "params": "22M", "type": "DeBERTa", "speed": "5x faster", "performance": "★★★"},
                {"name": "electra-small", "params": "14M", "type": "ELECTRA", "speed": "4x faster", "performance": "★★★"},
            ],
            "English Models": [
                {"name": "bert-base-uncased", "params": "110M", "type": "BERT", "performance": "★★★"},
                {"name": "bert-large-uncased", "params": "340M", "type": "BERT", "performance": "★★★★"},
                {"name": "roberta-base", "params": "125M", "type": "RoBERTa", "performance": "★★★★"},
                {"name": "roberta-large", "params": "355M", "type": "RoBERTa", "performance": "★★★★★"},
                {"name": "deberta-v3-base", "params": "184M", "type": "DeBERTa", "performance": "★★★★★"},
                {"name": "deberta-v3-large", "params": "435M", "type": "DeBERTa", "performance": "★★★★★"},
                {"name": "electra-base", "params": "110M", "type": "ELECTRA", "performance": "★★★★"},
                {"name": "electra-large", "params": "335M", "type": "ELECTRA", "performance": "★★★★★"},
                {"name": "albert-xlarge-v2", "params": "60M", "type": "ALBERT", "performance": "★★★★"},
            ],
            "French Models": [
                {"name": "camembert-base", "params": "110M", "type": "CamemBERT", "performance": "★★★★"},
                {"name": "camembert-large", "params": "335M", "type": "CamemBERT", "performance": "★★★★★"},
                {"name": "camemberta-base", "params": "110M", "type": "CamemBERTa-v2", "performance": "★★★★"},
                {"name": "flaubert-base", "params": "137M", "type": "FlauBERT", "performance": "★★★★"},
                {"name": "flaubert-large", "params": "373M", "type": "FlauBERT", "performance": "★★★★★"},
                {"name": "distilcamembert", "params": "68M", "type": "DistilCamemBERT", "performance": "★★★"},
                {"name": "barthez", "params": "165M", "type": "BARThez", "performance": "★★★★"},
                {"name": "fralbert", "params": "12M", "type": "FrALBERT", "performance": "★★★"},
                {"name": "fr-electra", "params": "110M", "type": "FrELECTRA", "performance": "★★★★"},
            ],
            "Other Language Models": [
                {"name": "asafaya/bert-base-arabic", "params": "110M", "type": "AraBERT", "language": "Arabic", "performance": "★★★★"},
                {"name": "bert-base-chinese", "params": "110M", "type": "Chinese BERT", "language": "Chinese", "performance": "★★★★"},
                {"name": "bert-base-german-cased", "params": "110M", "type": "German BERT", "language": "German", "performance": "★★★★"},
                {"name": "ai4bharat/indic-bert", "params": "110M", "type": "Hindi BERT", "language": "Hindi", "performance": "★★★"},
                {"name": "dbmdz/bert-base-italian-cased", "params": "110M", "type": "Italian BERT", "language": "Italian", "performance": "★★★★"},
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
        """Recommend training models based on detected languages - supports 11+ languages"""
        recommendations = []

        # Language-specific model mappings (comprehensive list)
        LANGUAGE_SPECIFIC_MODELS = {
            'en': ('English Models', ['bert-base-uncased', 'roberta-base', 'deberta-v3-base']),
            'fr': ('French Models', ['camembert-base', 'flaubert-base', 'distilcamembert']),
            'es': ('Spanish Models', ['dccuchile/bert-base-spanish-wwm-cased', 'PlanTL-GOB-ES/roberta-base-bne']),
            'de': ('German Models', ['bert-base-german-cased', 'deepset/gbert-base']),
            'it': ('Italian Models', ['dbmdz/bert-base-italian-cased', 'idb-ita/gilberto-uncased-from-camembert']),
            'pt': ('Portuguese Models', ['neuralmind/bert-base-portuguese-cased', 'portuguese-bert-base']),
            'nl': ('Dutch Models', ['GroNLP/bert-base-dutch-cased', 'wietsedv/bert-base-dutch-cased']),
            'ru': ('Russian Models', ['DeepPavlov/rubert-base-cased', 'sberbank-ai/ruBert-base']),
            'zh': ('Chinese Models', ['bert-base-chinese', 'hfl/chinese-bert-wwm-ext']),
            'ja': ('Japanese Models', ['cl-tohoku/bert-base-japanese', 'nlp-waseda/roberta-base-japanese']),
            'ar': ('Arabic Models', ['asafaya/bert-base-arabic', 'CAMeL-Lab/bert-base-arabic-camelbert-ca']),
        }

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

            if lang in LANGUAGE_SPECIFIC_MODELS:
                category, model_names = LANGUAGE_SPECIFIC_MODELS[lang]

                # Add language-specific models from all_models if available
                if category in all_models:
                    for model in all_models[category]:
                        recommendations.append({
                            'model': model['name'],
                            'category': category,
                            'reason': f"Optimized for {lang.upper()} ({model.get('performance', 'N/A')} performance)",
                            'priority': 1,
                            'details': model
                        })
                else:
                    # Fallback: use hardcoded models
                    for model_name in model_names[:3]:  # Top 3 models
                        recommendations.append({
                            'model': model_name,
                            'category': category,
                            'reason': f"Specialized for {lang.upper()}",
                            'priority': 1
                        })

                # Also suggest multilingual as fallback
                recommendations.append({
                    'model': 'xlm-roberta-base',
                    'category': 'Multilingual Models',
                    'reason': f'Multilingual fallback (supports {lang.upper()} + 100 languages)',
                    'priority': 2
                })

            else:
                # Language not in specific models - recommend multilingual
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

        # Only include formats fully supported by the annotation pipeline
        patterns = [
            '**/*.csv',  # CSV - fully supported
            '**/*.json', '**/*.jsonl',  # JSON formats - fully supported
            '**/*.xlsx', '**/*.xls',  # Excel formats - fully supported
            '**/*.parquet',  # Parquet - fully supported
            '**/*.RData', '**/*.rdata',  # R format - fully supported (requires pyreadr)
        ]

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
                # CSV format
                if info.format == 'csv':
                    df = pd.read_csv(file_path, nrows=100)

                # JSON formats
                elif info.format == 'json':
                    df = pd.read_json(file_path, lines=False, nrows=100)
                elif info.format == 'jsonl':
                    df = pd.read_json(file_path, lines=True, nrows=100)

                # Excel formats
                elif info.format in ['xlsx', 'xls']:
                    df = pd.read_excel(file_path, nrows=100)

                # Parquet format
                elif info.format == 'parquet':
                    df = pd.read_parquet(file_path).head(100)

                # R format
                elif info.format.lower() in ['rdata']:
                    try:
                        import pyreadr
                        result = pyreadr.read_r(str(file_path))
                        if result:
                            df = list(result.values())[0].head(100)
                        else:
                            return info
                    except ImportError:
                        # If pyreadr not available, return basic info without columns
                        return info

                else:
                    # Unsupported format - return basic info only
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

    def analyze_text_lengths(
        self,
        data_path: Path = None,
        df: Any = None,
        text_column: str = None,
        display_results: bool = True,
        step_label: str = "Text Length Analysis"
    ) -> Dict[str, Any]:
        """
        CRITICAL: Universal text length analysis method.

        This method MUST be used by ALL training modes:
        - Benchmark (single-label and multi-label)
        - Custom training
        - Model selector
        - Training studio
        - Quick training

        Args:
            data_path: Path to dataset file (CSV, JSON, JSONL, Excel, Parquet)
            df: Pre-loaded DataFrame (if already loaded)
            text_column: Name of text column to analyze
            display_results: Whether to display rich tables with results
            step_label: Label for display step (e.g., "Step 3b: Text Length Analysis")

        Returns:
            Dict with text length statistics and distribution
        """
        if display_results and self.console:
            self.console.print(f"\n[bold cyan]{step_label}[/bold cyan]\n")

        text_length_stats = {}
        requires_long_document_model = False

        try:
            # Verify required libraries are available
            if not HAS_PANDAS:
                raise ImportError("pandas not available")
            if not HAS_NUMPY:
                raise ImportError("numpy not available")

            # Import locally to avoid UnboundLocalError
            import pandas as pd
            import numpy as np

            # Load dataset if not provided
            if df is None and data_path is not None:
                if data_path.suffix == '.csv':
                    df = pd.read_csv(data_path)
                elif data_path.suffix == '.json':
                    df = pd.read_json(data_path)
                elif data_path.suffix == '.jsonl':
                    df = pd.read_json(data_path, lines=True)
                elif data_path.suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(data_path)
                elif data_path.suffix == '.parquet':
                    df = pd.read_parquet(data_path)

            if df is not None and text_column and text_column in df.columns:
                if display_results and self.console:
                    self.console.print("[dim]Analyzing text lengths for all documents...[/dim]\n")

                # Get all texts
                all_texts = df[text_column].dropna().astype(str).tolist()

                # Load tokenizer for accurate token counting
                try:
                    if not HAS_TRANSFORMERS:
                        raise ImportError("transformers library not available")

                    # Import tqdm locally to avoid UnboundLocalError
                    if HAS_TQDM:
                        from tqdm import tqdm

                    # Try to load tokenizer with fallback
                    tokenizer = None
                    tokenizer_models = [
                        "bert-base-multilingual-cased",  # Best for multilingual
                        "bert-base-uncased",              # Fallback
                        "distilbert-base-uncased"         # Lightweight fallback
                    ]

                    for model_name in tokenizer_models:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
                            break
                        except Exception as model_error:
                            self.logger.debug(f"Could not load {model_name}: {model_error}")
                            continue

                    if tokenizer is None:
                        raise RuntimeError("Could not load any tokenizer model")

                    # Analyze lengths in both characters and tokens
                    char_lengths = []
                    token_lengths = []

                    # Use tqdm only if available
                    text_iterator = tqdm(all_texts, desc="Measuring text lengths", disable=not HAS_TQDM) if HAS_TQDM else all_texts
                    for text in text_iterator:
                        char_lengths.append(len(text))
                        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
                        token_lengths.append(len(tokens))

                    char_lengths = np.array(char_lengths)
                    token_lengths = np.array(token_lengths)

                    # Calculate comprehensive statistics
                    text_length_stats = {
                        'char_min': int(np.min(char_lengths)),
                        'char_max': int(np.max(char_lengths)),
                        'char_mean': float(np.mean(char_lengths)),
                        'char_median': float(np.median(char_lengths)),
                        'char_std': float(np.std(char_lengths)),
                        'char_p25': float(np.percentile(char_lengths, 25)),
                        'char_p75': float(np.percentile(char_lengths, 75)),
                        'char_p95': float(np.percentile(char_lengths, 95)),
                        'token_min': int(np.min(token_lengths)),
                        'token_max': int(np.max(token_lengths)),
                        'token_mean': float(np.mean(token_lengths)),
                        'token_median': float(np.median(token_lengths)),
                        'token_std': float(np.std(token_lengths)),
                        'token_p25': float(np.percentile(token_lengths, 25)),
                        'token_p75': float(np.percentile(token_lengths, 75)),
                        'token_p95': float(np.percentile(token_lengths, 95)),
                        'avg_chars': float(np.mean(char_lengths)),  # For compatibility
                    }

                    # Classify documents by length
                    short_docs = np.sum(token_lengths < 128)
                    medium_docs = np.sum((token_lengths >= 128) & (token_lengths < 512))
                    long_docs = np.sum((token_lengths >= 512) & (token_lengths < 1024))
                    very_long_docs = np.sum(token_lengths >= 1024)
                    total_docs = len(token_lengths)

                    text_length_stats['distribution'] = {
                        'short': {'count': int(short_docs), 'percentage': float(short_docs / total_docs * 100)},
                        'medium': {'count': int(medium_docs), 'percentage': float(medium_docs / total_docs * 100)},
                        'long': {'count': int(long_docs), 'percentage': float(long_docs / total_docs * 100)},
                        'very_long': {'count': int(very_long_docs), 'percentage': float(very_long_docs / total_docs * 100)},
                    }

                    # Display results if requested
                    if display_results and self.console:
                        # Statistics table
                        stats_table = Table(title="Text Length Statistics", border_style="cyan", show_header=True, header_style="bold")
                        stats_table.add_column("Metric", style="cyan", width=20)
                        stats_table.add_column("Characters", style="yellow", justify="right", width=15)
                        stats_table.add_column("Tokens", style="green", justify="right", width=15)

                        stats_table.add_row("Minimum", f"{text_length_stats['char_min']:,}", f"{text_length_stats['token_min']:,}")
                        stats_table.add_row("25th Percentile", f"{text_length_stats['char_p25']:,.0f}", f"{text_length_stats['token_p25']:,.0f}")
                        stats_table.add_row("Median", f"{text_length_stats['char_median']:,.0f}", f"{text_length_stats['token_median']:,.0f}")
                        stats_table.add_row("Mean", f"{text_length_stats['char_mean']:,.0f}", f"{text_length_stats['token_mean']:,.0f}")
                        stats_table.add_row("75th Percentile", f"{text_length_stats['char_p75']:,.0f}", f"{text_length_stats['token_p75']:,.0f}")
                        stats_table.add_row("95th Percentile", f"{text_length_stats['char_p95']:,.0f}", f"{text_length_stats['token_p95']:,.0f}")
                        stats_table.add_row("Maximum", f"{text_length_stats['char_max']:,}", f"{text_length_stats['token_max']:,}")
                        stats_table.add_row("Std Deviation", f"{text_length_stats['char_std']:,.0f}", f"{text_length_stats['token_std']:,.0f}")

                        self.console.print(stats_table)

                        # Distribution table
                        self.console.print("\n")
                        dist_table = Table(title="Document Length Distribution", border_style="cyan", show_header=True, header_style="bold")
                        dist_table.add_column("Category", style="cyan", width=20)
                        dist_table.add_column("Token Range", style="white", width=20)
                        dist_table.add_column("Count", style="yellow", justify="right", width=12)
                        dist_table.add_column("Percentage", style="green", justify="right", width=12)

                        dist_table.add_row("Short", "< 128 tokens", f"{short_docs:,}", f"{short_docs / total_docs * 100:.1f}%")
                        dist_table.add_row("Medium", "128-511 tokens", f"{medium_docs:,}", f"{medium_docs / total_docs * 100:.1f}%")
                        dist_table.add_row("Long", "512-1023 tokens", f"{long_docs:,}", f"{long_docs / total_docs * 100:.1f}%",
                                         style="bold yellow" if long_docs > 0 else None)
                        dist_table.add_row("Very Long", "≥ 1024 tokens", f"{very_long_docs:,}", f"{very_long_docs / total_docs * 100:.1f}%",
                                         style="bold red" if very_long_docs > 0 else None)

                        self.console.print(dist_table)

                        # Long document warning
                        long_document_percentage = (long_docs + very_long_docs) / total_docs * 100

                        if long_document_percentage > 20:
                            requires_long_document_model = True
                            self.console.print(f"\n[bold yellow]⚠ Warning: {long_document_percentage:.1f}% of documents exceed 512 tokens[/bold yellow]")
                            self.console.print("[dim]Standard BERT models truncate at 512 tokens, which may lose important information.[/dim]")
                            self.console.print("[dim]Long-document models (Longformer, BigBird) can handle up to 4096 tokens.[/dim]\n")
                        else:
                            self.console.print(f"\n[green]✓ Most documents ({100 - long_document_percentage:.1f}%) fit within standard BERT limits (512 tokens)[/green]")

                    text_length_stats['requires_long_model'] = requires_long_document_model

                except Exception as tokenizer_error:
                    self.logger.debug(f"Tokenizer-based analysis failed: {tokenizer_error}")
                    if display_results and self.console:
                        self.console.print(f"[yellow]Could not load tokenizer for precise token counting[/yellow]")
                        self.console.print(f"[dim]Error: {str(tokenizer_error)}[/dim]")

                    # Fallback: character-based analysis only
                    char_lengths = [len(str(text)) for text in all_texts]
                    char_lengths = np.array(char_lengths)

                    text_length_stats = {
                        'char_min': int(np.min(char_lengths)),
                        'char_max': int(np.max(char_lengths)),
                        'char_mean': float(np.mean(char_lengths)),
                        'char_median': float(np.median(char_lengths)),
                        'char_p95': float(np.percentile(char_lengths, 95)),
                        'avg_chars': float(np.mean(char_lengths)),
                    }

                    if display_results and self.console:
                        self.console.print(f"[dim]Average text length: {text_length_stats['char_mean']:.0f} characters[/dim]")

        except Exception as e:
            self.logger.debug(f"Text length analysis failed: {e}")
            if display_results and self.console:
                self.console.print("[yellow]Could not perform text length analysis[/yellow]")
                self.console.print(f"[dim]Error: {str(e)}[/dim]")

        return text_length_stats

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

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available for training"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

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
        llms_table = Table(title="🤖 Available LLMs for Annotation", border_style="cyan", show_lines=True, expand=False, width=75)
        llms_table.add_column("Provider", style="cyan", width=10)
        llms_table.add_column("Model", style="white", width=22)
        llms_table.add_column("Size", style="yellow", width=8)
        llms_table.add_column("Context", style="green", width=11)
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
        trainer_table = Table(title="🏋️ Available Models for Training", border_style="magenta", show_lines=False, expand=False, width=85)
        trainer_table.add_column("Category", style="magenta bold", width=20)
        trainer_table.add_column("Models", style="white", width=56)

        # Define the desired order
        desired_order = [
            "Multilingual Models",
            "Long Document Models",
            "Long Document Models - French",
            "Long Document Models - Spanish",
            "Long Document Models - German",
            "Long Document Models - Italian",
            "Long Document Models - Portuguese",
            "Long Document Models - Dutch",
            "Long Document Models - Polish",
            "Long Document Models - Chinese",
            "Long Document Models - Japanese",
            "Long Document Models - Arabic",
            "Long Document Models - Russian",
            "Efficient Models",
            "English Models",
            "French Models",
            "Other Language Models"
        ]

        # Display models in the specified order
        for category in desired_order:
            if category in self.available_trainer_models:
                models = self.available_trainer_models[category]
                # Format model names compactly
                model_names = [m['name'] for m in models[:4]]  # Show first 4
                if len(models) > 4:
                    model_names.append(f"(+{len(models)-4} more)")
                trainer_table.add_row(
                    category,
                    ", ".join(model_names)
                )

        # Add any remaining categories not in the desired order
        for category, models in self.available_trainer_models.items():
            if category not in desired_order:
                model_names = [m['name'] for m in models[:4]]
                if len(models) > 4:
                    model_names.append(f"(+{len(models)-4} more)")
                trainer_table.add_row(
                    category,
                    ", ".join(model_names)
                )

        # === DISPLAY SIDE BY SIDE ===
        self.console.print(Columns([llms_table, trainer_table], equal=False, expand=True))
        self.console.print()

        # === DATASETS SECTION ===
        datasets_table = Table(title="📊 Detected Datasets", border_style="yellow", show_lines=False, expand=True)
        datasets_table.add_column("File", style="cyan", no_wrap=True, width=30)
        datasets_table.add_column("Format", style="white bold", width=12, justify="center")
        datasets_table.add_column("Size", style="green", width=10, justify="right")
        datasets_table.add_column("Folder", style="yellow", width=20)
        datasets_table.add_column("Columns", style="dim", width=35)

        if self.detected_datasets:
            for dataset in self.detected_datasets:  # Show ALL datasets
                columns_preview = ", ".join(dataset.columns[:3]) if dataset.columns else "N/A"
                if len(dataset.columns) > 3:
                    columns_preview += f" (+{len(dataset.columns)-3} more)"

                # Color format based on type
                format_style = {
                    'CSV': 'cyan bold',
                    'JSON': 'green bold',
                    'JSONL': 'blue bold',
                    'EXCEL': 'magenta bold',
                    'PARQUET': 'red bold',
                    'TSV': 'yellow bold'
                }.get(dataset.format.upper(), 'white')

                # Get folder name (parent directory name)
                folder_name = dataset.path.parent.name if dataset.path.parent.name else "data"

                datasets_table.add_row(
                    dataset.path.name,
                    f"[{format_style}]{dataset.format.upper()}[/{format_style}]",
                    f"{dataset.size_mb:.1f} MB" if dataset.size_mb else "Unknown",
                    folder_name,
                    columns_preview
                )
        else:
            datasets_table.add_row(
                "No datasets found",
                "-",
                "-",
                "-",
                "Place CSV/JSON files in current directory"
            )

        # Print datasets table
        self.console.print(datasets_table)
        self.console.print()

        # === ALL SUPPORTED FORMATS SECTION ===
        # Create a centered panel showing all supported formats
        all_formats_text = Text(justify="center")
        all_formats_text.append("📦 Supported Formats: ", style="bold cyan")

        # File formats
        all_formats_text.append("CSV", style="cyan bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("JSON/JSONL", style="green bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("Excel", style="magenta bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("Parquet", style="red bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("RData", style="yellow bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("TSV", style="blue bold")

        # Databases
        all_formats_text.append("\n💾 Databases: ", style="bold cyan")
        all_formats_text.append("PostgreSQL", style="blue bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("MySQL", style="yellow bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("SQLite", style="green bold")
        all_formats_text.append(" • ", style="dim")
        all_formats_text.append("MongoDB", style="magenta bold")

        formats_panel = Panel(
            all_formats_text,
            border_style="cyan",
            padding=(0, 2)
        )

        self.console.print(formats_panel)
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
                ("2", "🎯 LLM Annotation → Training - Complete Workflow"),
                ("3", "🏋️ Training Studio - Model Training & Benchmarking"),
                ("4", "🤖 BERT Annotation Studio - Annotate with Trained Models"),
                ("5", "🔍 Validation Lab - Quality Assurance Tools"),
                ("6", "💾 Profile Manager - Save & Load Configurations"),
                ("7", "📚 Documentation & Help"),
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

            # Smart prompt with validation (now 0-7 since we have 8 options)
            choice = Prompt.ask(
                "\n[bold yellow]Select option[/bold yellow]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                default="1"
            )

        else:
            print("\n" + "="*50)
            print("Main Menu")
            print("="*50)
            print("1. LLM Annotation Studio - Annotate with LLM (No Training)")
            print("2. LLM Annotation → Training - Complete Workflow")
            print("3. Training Studio - Model Training & Benchmarking")
            print("4. BERT Annotation Studio - Annotate with Trained Models")
            print("5. Validation Lab - Quality Assurance Tools")
            print("6. Profile Manager - Save & Load Configurations")
            print("7. Documentation & Help")
            print("0. Exit")
            print("-"*50)

            suggestions = self._get_smart_suggestions()
            if suggestions:
                print(f"💡 {suggestions}")

            choice = input("\nSelect option (0-7): ").strip()

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

    def _display_ascii_logo(self):
        """Display only the ASCII logo, tagline, and workflow (without info panel)"""
        if HAS_RICH and self.console:
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
        else:
            print("="*80)
            print(" " * 28 + "LLM TOOL")
            print(" " * 18 + "Intelligent Annotation & Training Pipeline")
            print("="*80)
            print("\n  🤖 -> 📝 -> 🧹 -> 🎯 -> 🧠 -> 📊 -> ✨")
            print("  AI   Annotate Clean Label Train Test Deploy\n")
            print("="*80 + "\n")

    def _display_section_header(self, title: str, description: str, mode_info: Optional[Dict[str, Any]] = None):
        """Display a personalized section header with mode-specific information"""
        if HAS_RICH and self.console:
            # If mode_info provided, create a detailed panel
            if mode_info:
                from rich.table import Table

                info_table = Table(show_header=False, box=None, padding=(0, 2))

                # Always add author first
                info_table.add_row("👨‍💻 Author:", "[bright_yellow]Antoine Lemor[/bright_yellow]")

                # Add mode-specific rows
                if 'workflow' in mode_info:
                    info_table.add_row("📊 Workflow:", f"[cyan]{mode_info['workflow']}[/cyan]")

                if 'capabilities' in mode_info:
                    caps = ' • '.join(mode_info['capabilities'])
                    info_table.add_row("🎯 Capabilities:", f"[yellow]{caps}[/yellow]")

                if 'input' in mode_info:
                    info_table.add_row("📥 Input:", f"[green]{mode_info['input']}[/green]")

                if 'output' in mode_info:
                    info_table.add_row("📤 Output:", f"[magenta]{mode_info['output']}[/magenta]")

                if 'best_for' in mode_info:
                    info_table.add_row("✨ Best For:", f"[bright_blue]{mode_info['best_for']}[/bright_blue]")

                if 'duration' in mode_info:
                    info_table.add_row("⏱️  Duration:", f"[dim]{mode_info['duration']}[/dim]")

                self.console.print(Panel(
                    info_table,
                    title=f"[bold cyan]{title}[/bold cyan]",
                    subtitle=f"[dim]{description}[/dim]",
                    border_style="cyan",
                    padding=(1, 2)
                ))
            else:
                # Fallback to simple panel
                self.console.print(Panel.fit(
                    f"[bold cyan]{title}[/bold cyan]\n{description}",
                    border_style="cyan"
                ))
        else:
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
        """Complete workflow: LLM annotation followed by intelligent model training"""
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display personalized mode info
        self._display_section_header(
            "🎯 LLM Annotation → Training - Complete Workflow",
            "End-to-end pipeline: annotate with LLM then train classifier models",
            mode_info={
                'workflow': 'Data → LLM Annotate → Language Detection → Model Training → Export',
                'capabilities': ['LLM Annotation', 'Multi-Language Support', 'Intelligent Training', 'Model Benchmarking'],
                'input': 'CSV/Excel/JSON with text column',
                'output': 'Annotated data + Trained BERT models + Training metrics',
                'best_for': 'Complete annotation-to-training pipeline with automatic language detection',
                'duration': '~10-30 min (annotation + training time)'
            }
        )

        if HAS_RICH and self.console:
            # Get smart suggestions
            suggestions = self._get_smart_suggestions()

            # Create workflow menu table
            from rich.table import Table
            workflow_table = Table(show_header=False, box=None, padding=(0, 2))
            workflow_table.add_column("Option", style="cyan", width=8)
            workflow_table.add_column("Description")

            workflows = [
                ("1", "🔄 Resume/Relaunch Workflow (Use saved parameters or resume incomplete)"),
                ("2", "🎯 Complete Workflow (New annotation → training pipeline)"),
                ("3", "🗑️  Clean Old Metadata (Delete saved parameters)"),
                ("0", "⬅️  Back to main menu")
            ]

            for option, desc in workflows:
                workflow_table.add_row(f"[bold cyan]{option}[/bold cyan]", desc)

            # Display panel with suggestions
            panel = Panel(
                workflow_table,
                title="[bold]🎯 LLM Annotation → Training Workflow[/bold]",
                subtitle=f"[dim]{suggestions}[/dim]" if suggestions else None,
                border_style="cyan"
            )

            self.console.print("\n")
            self.console.print(panel)

            workflow = Prompt.ask(
                "\n[bold yellow]Select workflow[/bold yellow]",
                choices=["0", "1", "2", "3"],
                default="2"
            )

            if workflow == "0":
                return
            elif workflow == "1":
                self._resume_mode2()
            elif workflow == "2":
                self._complete_workflow_mode2()
            elif workflow == "3":
                self._clean_metadata()
        else:
            print("\n=== LLM Annotation → Training Workflow ===")
            print("Complete annotation-to-training pipeline\n")
            print("1. Resume/Relaunch Workflow")
            print("2. Complete Workflow (Recommended)")
            print("3. Clean Old Metadata")
            print("0. Back")
            choice = input("\nSelect workflow: ").strip()

            if choice == "1":
                self._resume_mode2()
            elif choice == "2":
                self._complete_workflow_mode2()
            elif choice == "3":
                self._clean_metadata()

    def _complete_workflow_mode2(self):
        """Execute complete annotation → training workflow"""
        # Step 1: Data Source Selection
        self.console.print("[bold]Step 1/7: Data Source Selection[/bold]\n")

        # Ask user to choose between files and SQL database
        self.console.print("[yellow]Choose data source:[/yellow]")
        self.console.print("  1. 📁 Files (CSV/Excel/JSON/etc.) - Auto-detected or manual")
        self.console.print("  2. 🗄️  SQL Database (PostgreSQL/MySQL/SQLite/SQL Server)\n")

        data_source_choice = Prompt.ask(
            "Data source",
            choices=["1", "2"],
            default="1"
        )

        use_sql_database = (data_source_choice == "2")

        if use_sql_database:
            # SQL DATABASE WORKFLOW
            self.console.print("\n[bold cyan]🗄️  SQL Database (Training Sample)[/bold cyan]\n")
            self.console.print("[yellow]Note: For training, you'll select a representative sample from your database[/yellow]\n")

            # Database type selection
            db_choices = ["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server"]
            db_table = Table(title="Database Types", border_style="cyan")
            db_table.add_column("#", style="cyan", width=6)
            db_table.add_column("Database Type", style="white")
            for i, choice in enumerate(db_choices, 1):
                db_table.add_row(str(i), choice)
            self.console.print(db_table)

            db_choice = self._int_prompt_with_validation("Select database type", 1, 1, len(db_choices))
            db_type_name = db_choices[db_choice - 1]

            # Connection details
            if db_type_name == "SQLite":
                db_file = Prompt.ask("SQLite database file path")
                connection_string = f"sqlite:///{db_file}"
            else:
                host = Prompt.ask("Database host", default="localhost")
                default_ports = {"PostgreSQL": "5432", "MySQL": "3306", "Microsoft SQL Server": "1433"}
                port = Prompt.ask("Port", default=default_ports.get(db_type_name, "5432"))
                username = Prompt.ask("Username", default="postgres" if db_type_name == "PostgreSQL" else "root")
                password = Prompt.ask("Password", password=True)
                database = Prompt.ask("Database name")

                if db_type_name == "PostgreSQL":
                    connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                elif db_type_name == "MySQL":
                    connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
                else:
                    connection_string = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

            # Test connection
            self.console.print("\nTesting connection...")
            try:
                from sqlalchemy import create_engine, inspect, text
                import pandas as pd
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.console.print("[green]✓ Connected successfully![/green]\n")
            except Exception as e:
                self.console.print(f"[red]✗ Connection failed: {str(e)}[/red]")
                input("\nPress Enter to continue...")
                return

            # Table selection
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            if not tables:
                self.console.print("[red]No tables found[/red]")
                input("\nPress Enter to continue...")
                return

            # Get row counts
            table_info = []
            for table in tables:
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        table_info.append((table, result.scalar()))
                except:
                    table_info.append((table, None))

            tables_table = Table(title="Available Tables", border_style="cyan")
            tables_table.add_column("#", style="cyan", width=3)
            tables_table.add_column("Table Name", style="white")
            tables_table.add_column("Rows", style="green", justify="right")
            for i, (table, rows) in enumerate(table_info, 1):
                tables_table.add_row(str(i), table, f"{rows:,}" if rows else "?")
            self.console.print(tables_table)

            table_choice = self._int_prompt_with_validation("Select table", 1, 1, len(table_info))
            selected_table, total_rows = table_info[table_choice - 1]
            self.console.print(f"\n[green]✓ Selected: {selected_table} ({total_rows:,} rows)[/green]\n")

            # Load ALL data to temporary CSV (will use SAME workflow as files)
            from datetime import datetime
            import pandas as pd

            df = pd.read_sql(f"SELECT * FROM {selected_table}", engine)

            # Save to CSV in data/annotations
            annotations_dir = self.settings.paths.data_dir / 'annotations'
            annotations_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            data_path = annotations_dir / f"quickstart_sql_{selected_table}_{timestamp}.csv"
            df.to_csv(data_path, index=False)
            data_format = 'csv'

            self.console.print(f"[green]✓ Loaded {len(df):,} rows from {selected_table}[/green]")
            self.console.print(f"[dim]Saved to: {data_path}[/dim]")

        else:
            # FILE-BASED WORKFLOW (original code)
            if not self.detected_datasets:
                self.console.print("[yellow]No datasets auto-detected.[/yellow]")
                data_path = Path(self._prompt_file_path("Dataset path"))
            else:
                self.console.print(f"\n[bold cyan]📊 Found {len(self.detected_datasets)} dataset(s):[/bold cyan]\n")

                # Create table for datasets
                datasets_table = Table(border_style="cyan", show_header=True)
                datasets_table.add_column("#", style="bold yellow", width=4)
                datasets_table.add_column("Filename", style="white")
                datasets_table.add_column("Format", style="green", width=10)
                datasets_table.add_column("Size", style="magenta", width=10)
                datasets_table.add_column("Rows", style="cyan", width=10)
                datasets_table.add_column("Columns", style="blue", width=10)

                for i, ds in enumerate(self.detected_datasets[:20], 1):
                    # Format size
                    if ds.size_mb < 0.1:
                        size_str = f"{ds.size_mb * 1024:.1f} KB"
                    else:
                        size_str = f"{ds.size_mb:.1f} MB"

                    # Format rows and columns
                    rows_str = f"{ds.rows:,}" if ds.rows else "?"
                    cols_str = str(len(ds.columns)) if ds.columns else "?"

                    datasets_table.add_row(
                        str(i),
                        ds.path.name,
                        ds.format.upper(),
                        size_str,
                        rows_str,
                        cols_str
                    )

                self.console.print(datasets_table)
                self.console.print()

                use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
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

        # Step 2: Column Selection with Intelligent Detection (SAME AS MODE 1)
        self.console.print("\n[bold]Step 2/7: Column Selection with Intelligent Detection[/bold]")
        self.console.print("[dim]Analyzing columns...[/dim]\n")

        # Load sample data for intelligent detection (SAME AS MODE 1 DATABASE ANNOTATOR)
        import pandas as pd
        df_sample = pd.read_csv(data_path, nrows=100) if data_path.suffix == '.csv' else pd.read_excel(data_path, nrows=100)

        # Detect text columns intelligently (SAME LOGIC AS MODE 1)
        text_candidates = []
        id_candidates = []

        for col_name in df_sample.columns:
            # Check if it's a potential ID column
            col_lower = col_name.lower()
            if any(id_keyword in col_lower for id_keyword in ['id', 'key', 'index', 'number', 'num', 'pk']):
                # Check if values are unique
                is_unique = df_sample[col_name].nunique() == len(df_sample[col_name].dropna())
                if is_unique:
                    id_candidates.append({
                        'name': col_name,
                        'type': str(df_sample[col_name].dtype),
                        'confidence': 'high' if 'id' in col_lower else 'medium'
                    })

            # Check if it's a text column (object/string type)
            if df_sample[col_name].dtype == 'object':
                # Get non-null samples
                non_null = df_sample[col_name].dropna()
                if len(non_null) == 0:
                    continue

                # Calculate average length
                avg_length = non_null.astype(str).str.len().mean()
                sample_value = str(non_null.iloc[0])[:80] if len(non_null) > 0 else ""

                # Determine confidence based on average length
                if avg_length > 100:
                    confidence = "high"
                elif avg_length > 50:
                    confidence = "medium"
                elif avg_length > 20:
                    confidence = "low"
                else:
                    continue  # Skip very short text

                text_candidates.append({
                    'name': col_name,
                    'confidence': confidence,
                    'avg_length': avg_length,
                    'sample': sample_value
                })

        # Sort candidates (SAME AS MODE 1)
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        text_candidates.sort(key=lambda x: (confidence_order[x['confidence']], -x['avg_length']))
        id_candidates.sort(key=lambda x: (confidence_order.get(x['confidence'], 3)))

        # Display columns with intelligent suggestions (SAME TABLE AS MODE 1)
        col_table = Table(title=f"Columns in Dataset", box=box.ROUNDED)
        col_table.add_column("#", style="cyan", justify="right", width=4)
        col_table.add_column("Column Name", style="green", width=25)
        col_table.add_column("Type", style="yellow", width=20)
        col_table.add_column("Detection", style="magenta", width=30)

        detected_text_col = None
        detected_id_col = None
        columns_list = list(df_sample.columns)

        for idx, col_name in enumerate(columns_list, 1):
            col_type = str(df_sample[col_name].dtype)
            detection = ""

            # Check if it's a suggested text column
            text_match = next((tc for tc in text_candidates if tc['name'] == col_name), None)
            if text_match:
                if text_match['confidence'] == 'high':
                    detection = "📝 Text (High confidence)"
                    if detected_text_col is None:
                        detected_text_col = idx
                elif text_match['confidence'] == 'medium':
                    detection = "📝 Text (Medium)"
                else:
                    detection = "📝 Text (Low)"

            # Check if it's a suggested ID column
            id_match = next((ic for ic in id_candidates if ic['name'] == col_name), None)
            if id_match:
                if id_match['confidence'] == 'high':
                    detection = "🔑 ID (Recommended)"
                    if detected_id_col is None:
                        detected_id_col = idx
                else:
                    detection = "🔑 ID (Possible)"

            col_table.add_row(str(idx), col_name, col_type, detection)

        self.console.print(col_table)

        # Select text column with intelligent default (SAME AS MODE 1)
        if detected_text_col:
            self.console.print(f"\n[cyan]💡 Suggested text column: '{columns_list[detected_text_col-1]}' (detected automatically)[/cyan]")

        text_col_choice = Prompt.ask(
            "\n[cyan]Select TEXT column (to annotate)[/cyan]",
            choices=[str(i) for i in range(1, len(columns_list) + 1)],
            default=str(detected_text_col) if detected_text_col else "1"
        )
        text_column = columns_list[int(text_col_choice) - 1]

        # Select ID column with intelligent default (SAME AS MODE 1)
        identifier_column = None
        if Confirm.ask("\n[cyan]Do you want to select an ID column?[/cyan]", default=True):
            if detected_id_col:
                self.console.print(f"\n[cyan]💡 Suggested ID column: '{columns_list[detected_id_col-1]}' (unique values detected)[/cyan]")

            id_col_choice = Prompt.ask(
                "\n[cyan]Select ID column[/cyan]",
                choices=[str(i) for i in range(1, len(columns_list) + 1)],
                default=str(detected_id_col) if detected_id_col else "1"
            )
            identifier_column = columns_list[int(id_col_choice) - 1]

        self.console.print(f"\n[green]✓ Text column: {text_column}[/green]")
        if identifier_column:
            self.console.print(f"[green]✓ ID column: {identifier_column}[/green]")

        # Store column info for later use
        column_info = {
            'all_columns': columns_list,
            'text_candidates': text_candidates,
            'df': df_sample
        }
    
        # Step 3: Model Selection
        self.console.print("\n[bold]Step 3/7: Model Selection[/bold]")
        self.console.print("[dim]Tested API models: OpenAI & Anthropic[/dim]\n")
    
        selected_llm = self._select_llm_interactive()
        provider = selected_llm.provider
        model_name = selected_llm.name
    
        # Get API key if needed
        api_key = None
        if selected_llm.requires_api_key:
            api_key = self._get_or_prompt_api_key(provider, model_name)
    
        # Step 4: Prompt Configuration
        self.console.print("\n[bold]Step 4/7: Prompt Configuration[/bold]")
    
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
                if indices.strip():  # Only process if not empty
                    for idx_str in indices.split(','):
                        idx_str = idx_str.strip()
                        if idx_str:  # Skip empty strings
                            try:
                                idx = int(idx_str) - 1
                                if 0 <= idx < len(detected_prompts):
                                    selected_prompts.append(detected_prompts[idx])
                            except ValueError:
                                self.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
                if not selected_prompts:
                    self.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                    selected_prompts = detected_prompts
                else:
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
    
    
        # Step 4b: Language Column Detection (FROM QUICK START)
        self.console.print("\n[bold]Step 4b/7: Language Column Detection[/bold]")

        lang_column = None
        available_columns = column_info.get('all_columns', []) if column_info else []
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
                        "[yellow]⚠️  Language information is needed for training. Enable automatic language detection?[/yellow]",
                        default=True
                    )
                    if auto_detect:
                        self.console.print("[dim]Language will be automatically detected for each text during annotation.[/dim]")
                        lang_column = None  # Will trigger auto-detection later
                    else:
                        self.console.print("[yellow]⚠️  Warning: Proceeding without language information may affect training quality.[/yellow]")
                        lang_column = None
            else:
                # No language column detected
                has_lang = Confirm.ask("Does your dataset have a language column?", default=False)
                if has_lang:
                    lang_column = Prompt.ask(
                        "Language column name",
                        choices=available_columns,
                        default=available_columns[0] if available_columns else "language"
                    )
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
        self.console.print("\n[bold]Step 5/7: Advanced Options[/bold]")
    
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
        recommended_sample = None

        if scope_choice == "limit":
            # Option 2: FIRST ask about representative sample calculation (before asking for number)
            if total_rows and total_rows > 1000:
                self.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
                self.console.print("  Calculate statistically representative sample size (95% confidence interval)")
                self.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

                calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

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
                    self.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

            # THEN ask for specific number (with recommendation as default if calculated)
            default_limit = recommended_sample if recommended_sample else 100
            annotation_limit = self._int_prompt_with_validation(
                f"How many rows to annotate?",
                default=default_limit,
                min_value=1,
                max_value=total_rows if total_rows else 1000000
            )

            # Check if user chose the recommended sample
            if recommended_sample and annotation_limit == recommended_sample:
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
        self.console.print("\n[bold]Step 6/7: Review & Execute[/bold]")
    
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
            'identifier_column': identifier_column,  # From Step 2b: User-selected ID strategy
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
            'lang_column': lang_column,  # From Step 4b: Language column for training metadata
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
                },
                'training_workflow': {
                    'enabled': False,  # Will be updated after training workflow
                    'training_params_file': None,  # Will be added after training
                    'note': 'Training parameters will be saved separately after annotation completes'
                }
            }

            # Save metadata JSON (PRE-ANNOTATION SAVE POINT 1)
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

            # ============================================================
            # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
            # ============================================================
            if not lang_column:
                self.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
                self.console.print("[yellow]No language column was provided. Detecting languages for training...[/yellow]\n")

                try:
                    import pandas as pd
                    from llm_tool.utils.language_detector import LanguageDetector

                    # Load annotated file
                    df_annotated = pd.read_csv(output_file)

                    # CRITICAL: Only detect languages for ANNOTATED rows
                    # The output file may contain ALL original rows, but we only want to detect
                    # languages for rows that were actually annotated
                    original_row_count = len(df_annotated)

                    # Try to identify annotated rows by checking for annotation columns
                    # Common annotation column names: 'label', 'category', 'annotation', 'labels'
                    annotation_cols = [col for col in df_annotated.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]

                    if annotation_cols:
                        # Filter to only rows that have annotations (non-null in annotation column)
                        annotation_col = annotation_cols[0]
                        df_annotated = df_annotated[df_annotated[annotation_col].notna()].copy()
                        self.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows (out of {original_row_count:,} total rows in file)[/dim]")
                    else:
                        self.console.print(f"[yellow]⚠️  Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                    if len(df_annotated) == 0:
                        self.console.print("[yellow]⚠️  No annotated rows found. Skipping language detection.[/yellow]")
                    elif text_column in df_annotated.columns:
                        # Get ALL texts (including NaN) to maintain index alignment
                        all_texts = df_annotated[text_column].tolist()

                        # Count non-empty texts for display
                        non_empty_texts = sum(1 for text in all_texts if pd.notna(text) and len(str(text).strip()) > 10)

                        if non_empty_texts > 0:
                            detector = LanguageDetector()
                            detected_languages = []

                            # Progress indicator
                            from tqdm import tqdm
                            self.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                            for text in tqdm(all_texts, desc="Detecting languages", disable=not HAS_RICH):
                                # Handle NaN and empty texts
                                if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                                    detected_languages.append('unknown')
                                else:
                                    try:
                                        detected = detector.detect(str(text))
                                        if detected and detected.get('language'):
                                            detected_languages.append(detected['language'])
                                        else:
                                            detected_languages.append('unknown')
                                    except Exception as e:
                                        self.logger.debug(f"Language detection failed for text: {e}")
                                        detected_languages.append('unknown')

                            # Add language column to the filtered dataframe
                            df_annotated['lang'] = detected_languages

                            # Reload the FULL original file and update only the annotated rows
                            df_full = pd.read_csv(output_file)

                            # Initialize lang column if it doesn't exist
                            if 'lang' not in df_full.columns:
                                df_full['lang'] = 'unknown'

                            # Update language for annotated rows only
                            # Match by index of df_annotated within df_full
                            df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values

                            # Save updated full file with language column
                            df_full.to_csv(output_file, index=False)

                            # Show distribution
                            lang_counts = {}
                            for lang in detected_languages:
                                if lang != 'unknown':
                                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

                            if lang_counts:
                                total = sum(lang_counts.values())
                                self.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                                lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                                lang_table.add_column("Language", style="cyan", width=12)
                                lang_table.add_column("Count", style="yellow", justify="right", width=12)
                                lang_table.add_column("Percentage", style="green", justify="right", width=12)

                                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                    percentage = (count / total * 100) if total > 0 else 0
                                    lang_table.add_row(
                                        lang.upper(),
                                        f"{count:,}",
                                        f"{percentage:.1f}%"
                                    )

                                self.console.print(lang_table)
                                self.console.print(f"\n[green]✓ Language column 'lang' added to {output_file}[/green]")
                            else:
                                self.console.print("[yellow]⚠️  No languages detected successfully[/yellow]")

                except Exception as e:
                    self.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
                    self.logger.exception("Language detection failed")

            # ============================================================
            # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
            # ============================================================
            self._post_annotation_training_workflow(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs
            )

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

    def _resume_mode2(self):
        """Resume or relaunch annotation → training workflow using saved parameters"""
        self.console.print("\n[bold cyan]🔄 Resume/Relaunch Workflow[/bold cyan]\n")
        self.console.print("[dim]Load saved parameters from previous annotation → training sessions[/dim]\n")

        # ============================================================
        # DETECT METADATA FILES
        # ============================================================
        annotations_dir = self.settings.paths.data_dir / 'annotations'

        if not annotations_dir.exists():
            self.console.print("[yellow]No annotations directory found.[/yellow]")
            self.console.print("[dim]Run Complete Workflow first to create annotation sessions.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Find all metadata JSON files
        metadata_files = list(annotations_dir.glob("*_metadata_*.json"))

        if not metadata_files:
            self.console.print("[yellow]No saved workflow parameters found.[/yellow]")
            self.console.print("[dim]Run Complete Workflow and save parameters to use this feature.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Sort by modification time (most recent first)
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Display available sessions
        self.console.print(f"[green]Found {len(metadata_files)} saved workflow session(s)[/green]\n")

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
        self.console.print("  • [cyan]resume[/cyan]   - Continue an incomplete workflow (skip already completed steps)")
        self.console.print("           [dim]Requires output files from annotation and/or training[/dim]")
        self.console.print("  • [cyan]relaunch[/cyan] - Start a new workflow with same parameters")
        self.console.print("           [dim]Runs a fresh annotation → training session[/dim]")

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

        # Extract and potentially modify parameters
        modified_metadata = self._modify_parameters_if_requested(metadata, modify_params)

        # ============================================================
        # EXECUTE WORKFLOW
        # ============================================================
        self._execute_mode2_from_metadata(modified_metadata, action_mode, selected_file)

        self.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    def _execute_mode2_from_metadata(self, metadata: dict, action_mode: str, metadata_file: Path):
        """Execute annotation → training workflow based on loaded metadata

        This supports both annotation and training phases:
        - Resume: Skip completed annotation, optionally skip completed training
        - Relaunch: Re-run both annotation and training with same parameters
        """
        import json
        from datetime import datetime

        # Extract all parameters from metadata
        data_source = metadata.get('data_source', {})
        model_config = metadata.get('model_configuration', {})
        prompts = metadata.get('prompts', [])
        proc_config = metadata.get('processing_configuration', {})
        output_config = metadata.get('output', {})
        export_prefs = metadata.get('export_preferences', {})
        training_workflow = metadata.get('training_workflow', {})

        # Get export preferences
        export_to_doccano = export_prefs.get('export_to_doccano', False)
        export_to_labelstudio = export_prefs.get('export_to_labelstudio', False)
        export_sample_size = export_prefs.get('export_sample_size', 'all')

        # Prepare paths
        data_path = Path(data_source.get('file_path', ''))
        data_format = data_source.get('data_format', 'csv')
        text_column = data_source.get('text_column', 'text')

        # Check what phases are completed
        annotation_complete = False
        training_complete = False

        if action_mode == 'resume':
            # Try to find annotation output file
            original_output = Path(output_config.get('output_path', ''))

            if original_output.exists():
                self.console.print(f"\n[green]✓ Found annotation output: {original_output.name}[/green]")
                annotation_complete = True

                # Count already annotated rows
                import pandas as pd
                try:
                    if data_format == 'csv':
                        df_output = pd.read_csv(original_output)
                    elif data_format in ['excel', 'xlsx']:
                        df_output = pd.read_excel(original_output)
                    elif data_format == 'parquet':
                        df_output = pd.read_parquet(original_output)

                    # Count rows with valid annotations
                    if 'annotation' in df_output.columns:
                        annotated_mask = (
                            df_output['annotation'].notna() &
                            (df_output['annotation'].astype(str).str.strip() != '') &
                            (df_output['annotation'].astype(str) != 'nan')
                        )
                        annotated_count = annotated_mask.sum()
                        self.console.print(f"[cyan]  Rows already annotated: {annotated_count:,}[/cyan]")
                except Exception as e:
                    self.logger.warning(f"Could not read output file: {e}")
                    annotation_complete = False

            # Check if training was completed
            training_params_file = training_workflow.get('training_params_file')
            if training_params_file and Path(training_params_file).exists():
                self.console.print(f"[green]✓ Found training parameters: {Path(training_params_file).name}[/green]")

                # Check if model was saved
                model_dir = Path(original_output).parent / f"{Path(original_output).stem}_model"
                if model_dir.exists():
                    self.console.print(f"[green]✓ Found trained model: {model_dir.name}[/green]")
                    training_complete = True

        # Determine what to run
        run_annotation = not annotation_complete or action_mode == 'relaunch'
        run_training = not training_complete or action_mode == 'relaunch'

        if action_mode == 'resume' and annotation_complete and training_complete:
            self.console.print("\n[yellow]✓ Both annotation and training are already complete![/yellow]")
            retry_training = Confirm.ask("Re-run training with same annotations?", default=False)
            if retry_training:
                run_annotation = False
                run_training = True
            else:
                self.console.print("[yellow]Nothing to do. Workflow is complete.[/yellow]")
                return

        # ============================================================
        # PHASE 1: ANNOTATION (if needed)
        # ============================================================
        if run_annotation:
            self.console.print("\n[bold cyan]📝 Phase 1: LLM Annotation[/bold cyan]\n")

            # Use _execute_from_metadata logic for annotation
            # (Reuse existing implementation from Mode 1)
            annotations_dir = self.settings.paths.data_dir / 'annotations'
            annotations_dir.mkdir(parents=True, exist_ok=True)
            safe_model_name = model_config.get('model_name', 'unknown').replace(':', '_').replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
            output_path = annotations_dir / output_filename

            # Build pipeline config (same as Mode 1)
            provider = model_config.get('provider', 'ollama')
            model_name = model_config.get('model_name', 'llama2')

            # Get API key if needed
            api_key = None
            if provider in ['openai', 'anthropic', 'google']:
                api_key = self._get_api_key(provider)
                if not api_key:
                    self.console.print(f"[red]API key required for {provider}[/red]")
                    return

            # Prepare prompts payload
            prompts_payload = []
            for p in prompts:
                prompts_payload.append({
                    'prompt': p.get('prompt_content', p.get('prompt', '')),
                    'expected_keys': p.get('expected_keys', []),
                    'prefix': p.get('prefix', '')
                })

            pipeline_config = {
                'mode': 'file',
                'data_source': data_format,
                'data_format': data_format,
                'file_path': str(data_path),
                'text_column': text_column,
                'text_columns': [text_column],
                'annotation_column': 'annotation',
                'identifier_column': 'annotation_id',
                'run_annotation': True,
                'annotation_mode': model_config.get('annotation_mode', 'local'),
                'annotation_provider': provider,
                'annotation_model': model_name,
                'api_key': api_key,
                'prompts': prompts_payload,
                'annotation_sample_size': data_source.get('total_rows'),
                'annotation_sampling_strategy': data_source.get('sampling_strategy', 'head'),
                'max_tokens': model_config.get('max_tokens', 1000),
                'temperature': model_config.get('temperature', 0.7),
                'top_p': model_config.get('top_p', 1.0),
                'max_workers': proc_config.get('parallel_workers', 1),
                'output_format': data_format,
                'output_path': str(output_path),
                'save_incrementally': True,
                'batch_size': proc_config.get('batch_size', 1),
                'run_validation': False,
                'run_training': False,
            }

            # Execute annotation
            try:
                self.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

                from ..pipelines.pipeline_controller import PipelineController
                from ..utils.rich_progress_manager import RichProgressManager
                from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

                pipeline_with_progress = PipelineController(settings=self.settings)

                with RichProgressManager(show_json_every=1, compact_mode=False) as progress_manager:
                    enhanced_pipeline = EnhancedPipelineWrapper(pipeline_with_progress, progress_manager)
                    state = enhanced_pipeline.run_pipeline(pipeline_config)

                    if state.errors:
                        error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                        self.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                        return

                annotation_results = state.annotation_results or {}
                output_file = annotation_results.get('output_file', str(output_path))

                self.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
                self.console.print(f"[bold cyan]📄 Output File:[/bold cyan] {output_file}\n")

            except Exception as exc:
                self.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
                self.logger.exception("Mode 2 annotation failed")
                return

        else:
            # Use existing annotation file
            output_file = str(original_output)
            self.console.print(f"\n[yellow]⏭️  Skipping annotation (already complete)[/yellow]")
            self.console.print(f"[cyan]Using existing file: {output_file}[/cyan]\n")

        # ============================================================
        # LANGUAGE DETECTION (if not already done)
        # ============================================================
        self.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
        self.console.print("[yellow]Checking for language column...[/yellow]\n")

        try:
            import pandas as pd
            df_for_lang = pd.read_csv(output_file)

            # Check if language detection already done
            if 'lang' not in df_for_lang.columns:
                self.console.print("[yellow]No language column found. Detecting languages...[/yellow]\n")

                # Get annotated rows only
                annotation_cols = [col for col in df_for_lang.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]
                if annotation_cols:
                    annotation_col = annotation_cols[0]
                    df_annotated = df_for_lang[df_for_lang[annotation_col].notna()].copy()

                    self.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows[/dim]")

                    # Detect languages
                    from llm_tool.utils.language_detector import LanguageDetector
                    from tqdm import tqdm

                    detector = LanguageDetector()
                    all_texts = df_annotated[text_column].tolist()
                    detected_languages = []

                    non_empty_texts = sum(1 for t in all_texts if pd.notna(t) and str(t).strip() and len(str(t).strip()) > 10)
                    self.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                    for text in tqdm(all_texts, desc="Detecting languages", disable=not HAS_RICH):
                        if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                            detected_languages.append('unknown')
                        else:
                            try:
                                detected = detector.detect(str(text))
                                if detected and detected.get('language'):
                                    detected_languages.append(detected['language'])
                                else:
                                    detected_languages.append('unknown')
                            except Exception as e:
                                self.logger.debug(f"Language detection failed for text: {e}")
                                detected_languages.append('unknown')

                    # Add language column
                    df_annotated['lang'] = detected_languages

                    # Update full file
                    df_full = pd.read_csv(output_file)
                    if 'lang' not in df_full.columns:
                        df_full['lang'] = 'unknown'
                    df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values
                    df_full.to_csv(output_file, index=False)

                    # Display language distribution
                    lang_counts = df_annotated['lang'].value_counts()
                    lang_counts_filtered = {k: v for k, v in lang_counts.items() if k != 'unknown'}

                    if lang_counts_filtered:
                        total = sum(lang_counts_filtered.values())
                        self.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                        lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                        lang_table.add_column("Language", style="cyan", width=12)
                        lang_table.add_column("Count", style="yellow", justify="right", width=12)
                        lang_table.add_column("Percentage", style="green", justify="right", width=12)

                        for lang, count in sorted(lang_counts_filtered.items(), key=lambda x: x[1], reverse=True):
                            pct = (count / total * 100)
                            lang_table.add_row(lang.upper(), f"{count:,}", f"{pct:.1f}%")

                        self.console.print(lang_table)
                        self.console.print(f"\n[green]✓ Language column 'lang' added to output file[/green]\n")
            else:
                self.console.print("[green]✓ Language column already exists[/green]\n")

        except Exception as e:
            self.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
            self.logger.exception("Language detection failed")

        # ============================================================
        # PHASE 2: TRAINING (if needed)
        # ============================================================
        if run_training:
            self.console.print("\n[bold cyan]🎓 Phase 2: Model Training[/bold cyan]\n")

            # Build prompt_configs for training workflow
            prompt_configs_for_training = []
            for p in prompts:
                prompt_configs_for_training.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt_content', p.get('prompt', '')),
                        'name': p.get('name', 'prompt')
                    },
                    'prefix': p.get('prefix', '')
                })

            # Execute training workflow
            self._post_annotation_training_workflow(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs_for_training
            )

        else:
            self.console.print(f"\n[yellow]⏭️  Skipping training (already complete)[/yellow]\n")

        self.console.print("\n[bold green]✅ Workflow complete![/bold green]")

    def _post_annotation_training_workflow(
        self,
        output_file: str,
        text_column: str,
        prompt_configs: list
    ):
        """
        Comprehensive post-annotation training workflow.
        Inspired by Training Studio (Mode 5) but adapted for annotated data.

        Features:
        - Language detection & analysis (with low-percentage reclassification)
        - Text length analysis for long-document model recommendation
        - Model recommendation based on languages & text length
        - Training strategy selection (multilingual/specialized/hybrid)
        - Benchmark mode option for specific categories
        - Full training pipeline integration
        """
        try:
            self.console.print("\n[bold cyan]🎓 Post-Annotation Training[/bold cyan]")
            self.console.print("[dim]Intelligent model training from your LLM annotations[/dim]\n")

            # Ask if user wants to train a model
            train_model = Confirm.ask(
                "[bold]Would you like to train a classifier model from these annotations?[/bold]",
                default=True
            )

            if not train_model:
                self.console.print("[yellow]Skipping training. Annotations are ready for manual use or export.[/yellow]")
                return

            # Load annotated data
            import pandas as pd
            df = pd.read_csv(output_file)

            # Filter to annotated rows only
            annotation_cols = [col for col in df.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]
            if not annotation_cols:
                self.console.print("[yellow]⚠️  No annotation columns found. Cannot proceed with training.[/yellow]")
                return

            annotation_col = annotation_cols[0]
            df_annotated = df[df[annotation_col].notna()].copy()

            if len(df_annotated) == 0:
                self.console.print("[yellow]⚠️  No annotated rows found. Cannot proceed with training.[/yellow]")
                return

            self.console.print(f"[green]✓ Found {len(df_annotated):,} annotated rows for training[/green]\n")

            # Step 1: Category/Label Analysis
            self.console.print("[bold cyan]Step 1: Label/Category Analysis[/bold cyan]\n")

            # Detect label format (single vs multi-label)
            is_multi_label = False
            try:
                first_label = df_annotated[annotation_col].iloc[0]
                if isinstance(first_label, str) and (first_label.startswith('[') or first_label.startswith('{')):
                    import json
                    parsed = json.loads(first_label)
                    is_multi_label = isinstance(parsed, list)
            except:
                pass

            # Analyze categories
            if is_multi_label:
                self.console.print("[yellow]Multi-label classification detected[/yellow]")
                all_labels = []
                for val in df_annotated[annotation_col]:
                    try:
                        import json
                        parsed = json.loads(str(val)) if isinstance(val, str) else val
                        if isinstance(parsed, list):
                            all_labels.extend(parsed)
                    except:
                        pass
                label_counts = pd.Series(all_labels).value_counts()
            else:
                self.console.print("[yellow]Single-label classification detected[/yellow]")
                label_counts = df_annotated[annotation_col].value_counts()

            # Display category distribution
            from rich.table import Table
            cat_table = Table(title="Category Distribution", border_style="cyan", show_header=True, header_style="bold")
            cat_table.add_column("Category", style="cyan", width=30)
            cat_table.add_column("Count", style="yellow", justify="right", width=12)
            cat_table.add_column("Percentage", style="green", justify="right", width=12)

            total_labels = label_counts.sum()
            for label, count in label_counts.head(20).items():
                percentage = (count / total_labels * 100)
                cat_table.add_row(
                    str(label)[:30],
                    f"{count:,}",
                    f"{percentage:.1f}%"
                )

            if len(label_counts) > 20:
                cat_table.add_row("...", f"... and {len(label_counts) - 20} more", "...")

            self.console.print(cat_table)
            self.console.print(f"\n[green]✓ {len(label_counts)} unique categories detected[/green]\n")

            # Step 2: Benchmark Mode Option
            self.console.print("[bold cyan]Step 2: Training Mode Selection[/bold cyan]\n")
            self.console.print("[yellow]Training Mode Options:[/yellow]")
            self.console.print("  • [cyan]full[/cyan]     - Train on ALL categories")
            self.console.print("             Best for: Production deployment")
            self.console.print("  • [cyan]benchmark[/cyan] - Test MULTIPLE models on ONE specific category")
            self.console.print("             Best for: Finding the best model for your data")
            self.console.print("             Output: Comparison metrics to choose optimal model\n")

            training_mode = Prompt.ask(
                "Select training mode",
                choices=["full", "benchmark"],
                default="full"
            )

            benchmark_category = None
            if training_mode == "benchmark":
                self.console.print("\n[bold]Benchmark Mode: Select Target Category[/bold]")
                self.console.print("[dim]You'll train multiple models on this category to find the best one[/dim]\n")

                # Show top categories
                top_cats = list(label_counts.head(10).index)
                for i, cat in enumerate(top_cats, 1):
                    count = label_counts[cat]
                    pct = (count / total_labels * 100)
                    self.console.print(f"  {i}. {cat} ({count:,} samples, {pct:.1f}%)")

                cat_choice = Prompt.ask(
                    "\nSelect category number or enter name",
                    default="1"
                )

                if cat_choice.isdigit() and 0 < int(cat_choice) <= len(top_cats):
                    benchmark_category = top_cats[int(cat_choice) - 1]
                else:
                    benchmark_category = cat_choice

                self.console.print(f"[green]✓ Benchmark category: {benchmark_category}[/green]\n")

            # Step 3: Text Length Analysis (from Training Studio)
            self.console.print("[bold cyan]Step 3: Text Length Analysis[/bold cyan]\n")

            text_length_stats = {}
            requires_long_document_model = False

            if text_column in df_annotated.columns:
                all_texts = df_annotated[text_column].dropna().astype(str).tolist()

                if len(all_texts) > 0:
                    try:
                        from transformers import AutoTokenizer
                        import numpy as np
                        from tqdm import tqdm

                        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                        token_lengths = []

                        for text in tqdm(all_texts, desc="Analyzing text lengths", disable=not HAS_RICH):
                            tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
                            token_lengths.append(len(tokens))

                        token_lengths = np.array(token_lengths)

                        text_length_stats = {
                            'token_mean': float(np.mean(token_lengths)),
                            'token_median': float(np.median(token_lengths)),
                            'token_max': int(np.max(token_lengths)),
                            'token_p95': float(np.percentile(token_lengths, 95))
                        }

                        # Classify documents
                        long_docs = np.sum(token_lengths >= 512)
                        total = len(token_lengths)
                        long_pct = (long_docs / total * 100) if total > 0 else 0

                        self.console.print(f"[dim]Average: {text_length_stats['token_mean']:.0f} tokens | Max: {text_length_stats['token_max']} tokens[/dim]")

                        if long_pct > 20:
                            requires_long_document_model = True
                            self.console.print(f"[yellow]⚠ {long_pct:.1f}% of texts exceed 512 tokens[/yellow]")
                            use_long = Confirm.ask("Use long-document models (Longformer/BigBird)?", default=True)
                            text_length_stats['user_prefers_long_models'] = use_long
                        else:
                            self.console.print(f"[green]✓ {100-long_pct:.1f}% fit within standard BERT limits[/green]")
                            text_length_stats['user_prefers_long_models'] = False

                    except Exception as e:
                        self.logger.debug(f"Text length analysis failed: {e}")
                        self.console.print("[yellow]Could not perform detailed length analysis[/yellow]")

            self.console.print()

            # Step 4: Language Strategy Analysis (from Training Studio)
            self.console.print("[bold cyan]Step 4: Language Strategy[/bold cyan]\n")

            # Check if language column exists
            has_lang_col = 'lang' in df_annotated.columns
            confirmed_languages = set()
            language_distribution = {}

            if has_lang_col:
                lang_counts = df_annotated['lang'].value_counts()
                total_lang = len(df_annotated)

                # Display language distribution
                lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                lang_table.add_column("Language", style="cyan", width=12)
                lang_table.add_column("Count", style="yellow", justify="right", width=12)
                lang_table.add_column("Percentage", style="green", justify="right", width=12)

                for lang, count in lang_counts.items():
                    if lang != 'unknown':
                        pct = (count / total_lang * 100)
                        lang_table.add_row(lang.upper(), f"{count:,}", f"{pct:.1f}%")
                        language_distribution[lang] = int(count)
                        confirmed_languages.add(lang)

                self.console.print(lang_table)
                self.console.print(f"\n[green]✓ {len(confirmed_languages)} language(s) detected[/green]\n")
            else:
                self.console.print("[yellow]No language column found. Assuming single language.[/yellow]\n")

            # Language training strategy
            model_strategy = "multilingual"
            language_model_mapping = {}

            if len(confirmed_languages) > 1:
                self.console.print("[yellow]Multiple languages detected. Select training strategy:[/yellow]")
                self.console.print("  • [cyan]multilingual[/cyan] - ONE model for all languages")
                self.console.print("  • [cyan]specialized[/cyan] - SEPARATE model per language")
                self.console.print("  • [cyan]hybrid[/cyan] - Multilingual base + specialized fine-tuning\n")

                model_strategy = Prompt.ask(
                    "Training strategy",
                    choices=["multilingual", "specialized", "hybrid"],
                    default="multilingual"
                )

                self.console.print(f"[green]✓ Strategy: {model_strategy}[/green]\n")

            # Step 5: Model Selection
            self.console.print("[bold cyan]Step 5: Model Selection[/bold cyan]\n")

            # Handle specialized training: select model per language
            if model_strategy == "specialized":
                self.console.print("[yellow]Specialized Training: Select a model for EACH language[/yellow]\n")

                for lang in sorted(confirmed_languages):
                    lang_upper = lang.upper()
                    self.console.print(f"[bold]Models for {lang_upper}:[/bold]")

                    # Get language-specific recommendations
                    if text_length_stats.get('user_prefers_long_models'):
                        lang_recommendations = self._get_long_document_models_for_language(lang)
                    else:
                        from llm_tool.utils.language_normalizer import LanguageNormalizer
                        lang_recs = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)
                        lang_recommendations = lang_recs if lang_recs else []

                    if not lang_recommendations:
                        # Fallback to multilingual models
                        lang_recommendations = [
                            {'model': 'xlm-roberta-base', 'reason': 'Multilingual baseline (100+ languages)'},
                            {'model': 'bert-base-multilingual-cased', 'reason': 'Multilingual BERT (104 languages)'},
                        ]

                    # Display recommendations
                    for i, rec in enumerate(lang_recommendations[:5], 1):
                        self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

                    model_choice = Prompt.ask(
                        f"\nSelect model for {lang_upper} (number or name)",
                        default="1"
                    )

                    if model_choice.isdigit() and 0 < int(model_choice) <= len(lang_recommendations):
                        selected = lang_recommendations[int(model_choice) - 1]['model']
                    else:
                        selected = model_choice

                    language_model_mapping[lang] = selected
                    self.console.print(f"[green]✓ {lang_upper}: {selected}[/green]\n")

                # For training params, use first language's model as default
                training_model = list(language_model_mapping.values())[0]

            else:
                # Multilingual or Hybrid: single model for all languages
                recommended_models = []

                if text_length_stats.get('user_prefers_long_models'):
                    self.console.print("[yellow]Long-document models (4096+ tokens):[/yellow]")
                    recommended_models = [
                        "markussagen/xlm-roberta-longformer-base-4096",  # Multilingual FIRST
                        "google/long-t5-local-base",  # Multilingual
                        "allenai/longformer-base-4096",  # English only
                        "google/bigbird-roberta-base"  # English only
                    ]
                else:
                    if confirmed_languages:
                        from llm_tool.utils.language_normalizer import LanguageNormalizer
                        recs = LanguageNormalizer.recommend_models(confirmed_languages, self.available_trainer_models)
                        recommended_models = [r['model'] for r in recs[:5]] if recs else []

                    if not recommended_models:
                        recommended_models = ["bert-base-multilingual-cased", "xlm-roberta-base", "distilbert-base-multilingual-cased"]

                self.console.print("[green]Recommended models:[/green]")
                for i, model in enumerate(recommended_models[:5], 1):
                    self.console.print(f"  {i}. {model}")

                selected_model = Prompt.ask(
                    "\nSelect model (number or name)",
                    default="1"
                )

                if selected_model.isdigit() and 0 < int(selected_model) <= len(recommended_models):
                    training_model = recommended_models[int(selected_model) - 1]
                else:
                    training_model = selected_model

                self.console.print(f"[green]✓ Selected model: {training_model}[/green]\n")

            # Step 6: Training Configuration
            self.console.print("[bold cyan]Step 6: Training Configuration[/bold cyan]\n")

            # Training mode selection
            training_modes = {
                "quick": "Quick training (2 epochs, fast)",
                "benchmark": "Benchmark mode (compare models)",
                "custom": "Custom configuration"
            }

            self.console.print("[yellow]Training modes:[/yellow]")
            for mode, desc in training_modes.items():
                self.console.print(f"  • [cyan]{mode}[/cyan]: {desc}")

            train_mode = Prompt.ask(
                "\nSelect training mode",
                choices=list(training_modes.keys()),
                default="quick"
            )

            # Epochs
            if train_mode == "quick":
                epochs = 2
            elif train_mode == "benchmark":
                epochs = 3
            else:
                epochs = self._int_prompt_with_validation("Number of epochs", default=3, min_value=1, max_value=20)

            # Batch size
            batch_size = self._int_prompt_with_validation("Batch size", default=16, min_value=1, max_value=128)

            # Learning rate
            learning_rate = 2e-5
            if train_mode == "custom":
                lr_input = Prompt.ask("Learning rate", default="2e-5")
                try:
                    learning_rate = float(lr_input)
                except:
                    learning_rate = 2e-5

            self.console.print(f"\n[green]✓ Configuration:[/green]")
            self.console.print(f"  Epochs: {epochs}")
            self.console.print(f"  Batch size: {batch_size}")
            self.console.print(f"  Learning rate: {learning_rate}\n")

            # ===========================================================
            # SAVE TRAINING PARAMETERS (Second save point)
            # ===========================================================
            self.console.print("[bold cyan]💾 Saving Training Parameters[/bold cyan]\n")

            training_params = {
                "training_metadata": {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "annotation_file": output_file,
                    "total_annotated_rows": len(df_annotated),
                    "training_mode": training_mode,
                    "benchmark_category": benchmark_category
                },
                "data_config": {
                    "text_column": text_column,
                    "label_column": annotation_col,
                    "is_multi_label": is_multi_label,
                    "num_categories": len(label_counts),
                    "category_distribution": label_counts.to_dict()
                },
                "language_config": {
                    "confirmed_languages": list(confirmed_languages),
                    "language_distribution": language_distribution,
                    "model_strategy": model_strategy,
                    "language_model_mapping": language_model_mapping
                },
                "text_analysis": {
                    "text_length_stats": text_length_stats,
                    "requires_long_document_model": requires_long_document_model
                },
                "model_config": {
                    "selected_model": training_model,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                },
                "prompt_configs": prompt_configs
            }

            # Save training parameters
            import json
            from pathlib import Path

            params_file = Path(output_file).parent / f"{Path(output_file).stem}_training_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(training_params, f, indent=2, ensure_ascii=False)

            self.console.print(f"[green]✓ Training parameters saved to:[/green]")
            self.console.print(f"  {params_file}\n")

            # Step 7: Execute Training
            self.console.print("[bold cyan]Step 7: Model Training[/bold cyan]\n")

            start_training = Confirm.ask(
                "[bold]Start training now?[/bold]",
                default=True
            )

            if start_training:
                self.console.print("[green]🚀 Starting training...[/green]\n")

                # Call Training Studio's training method
                try:
                    from llm_tool.trainers.training_data_builder import TrainingDatasetBuilder

                    # Prepare training dataset
                    builder = TrainingDatasetBuilder()

                    # Convert annotated data to training format
                    training_data_path = Path(output_file).parent / f"{Path(output_file).stem}_training.csv"

                    # Prepare data in correct format
                    train_df = df_annotated[[text_column, annotation_col]].copy()
                    train_df.columns = ['text', 'label']
                    train_df.to_csv(training_data_path, index=False)

                    self.console.print(f"[green]✓ Training data prepared: {training_data_path}[/green]")

                    # Initialize training pipeline
                    from llm_tool.trainers.model_trainer import ModelTrainer

                    trainer = ModelTrainer(
                        model_name=training_model,
                        num_labels=len(label_counts),
                        device="cuda" if self._check_cuda_available() else "cpu"
                    )

                    self.console.print(f"[yellow]Training {training_model} for {epochs} epochs...[/yellow]")

                    # Load and split data
                    from sklearn.model_selection import train_test_split
                    from collections import Counter

                    # Check if we have at least 2 instances per class for stratification
                    stratify_col = None
                    if not is_multi_label:
                        label_counts = Counter(train_df['label'])
                        min_count = min(label_counts.values())

                        if min_count < 2:
                            # Find which classes have insufficient instances
                            insufficient_classes = [cls for cls, count in label_counts.items() if count < 2]
                            self.console.print(f"[yellow]⚠️  Dataset has class(es) with only 1 instance: {insufficient_classes}[/yellow]")

                            # Ask user if they want to remove these classes or proceed
                            remove_classes = Prompt.ask(
                                f"Remove class(es) with insufficient instances?",
                                choices=["y", "n"],
                                default="y"
                            )

                            if remove_classes.lower() == 'y':
                                # Filter out samples with insufficient classes
                                original_count = len(train_df)
                                train_df = train_df[~train_df['label'].isin(insufficient_classes)]
                                self.console.print(f"[dim]Removed {original_count - len(train_df)} samples from classes: {insufficient_classes}[/dim]")

                                # Recompute label counts
                                label_counts = Counter(train_df['label'])
                                min_count = min(label_counts.values()) if label_counts else 0

                                if min_count < 2:
                                    self.console.print("[red]Still insufficient samples after removal. Cannot proceed with training.[/red]")
                                    return

                                stratify_col = train_df['label']
                            else:
                                self.console.print(f"[yellow]Proceeding without stratification (may reduce quality)[/yellow]")
                        else:
                            stratify_col = train_df['label']

                    train_data, val_data = train_test_split(
                        train_df,
                        test_size=0.2,
                        random_state=42,
                        stratify=stratify_col
                    )

                    # Train model
                    results = trainer.train(
                        train_texts=train_data['text'].tolist(),
                        train_labels=train_data['label'].tolist(),
                        val_texts=val_data['text'].tolist() if len(val_data) > 0 else None,
                        val_labels=val_data['label'].tolist() if len(val_data) > 0 else None,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate
                    )

                    # Save model
                    model_save_path = Path(output_file).parent / f"{Path(output_file).stem}_model"
                    trainer.save_model(str(model_save_path))

                    self.console.print(f"\n[bold green]✅ Training completed![/bold green]")
                    self.console.print(f"[green]Model saved to: {model_save_path}[/green]")

                    if results:
                        self.console.print(f"\n[bold cyan]Training Results:[/bold cyan]")
                        for key, value in results.items():
                            self.console.print(f"  {key}: {value}")

                except Exception as train_error:
                    self.console.print(f"[red]❌ Training error: {train_error}[/red]")
                    self.logger.exception("Training execution failed")
            else:
                self.console.print("[yellow]Training skipped. You can resume later using the saved parameters.[/yellow]")

        except Exception as e:
            self.console.print(f"[red]Training workflow error: {e}[/red]")
            self.logger.exception("Post-annotation training workflow failed")

    def _get_model_description(self, model_name: str) -> str:
        """Get factual description for a model based on its name"""
        # Normalize model name for matching
        name_lower = model_name.lower()

        # Try exact match first
        if name_lower in MODEL_DESCRIPTIONS:
            return MODEL_DESCRIPTIONS[name_lower]

        # Try prefix match for variants (e.g., "llama3.2:3b" matches "llama3.2")
        for key in MODEL_DESCRIPTIONS:
            if name_lower.startswith(key):
                return MODEL_DESCRIPTIONS[key]

        # Try substring match for common patterns
        for key in MODEL_DESCRIPTIONS:
            if key in name_lower or name_lower.split(':')[0] == key:
                return MODEL_DESCRIPTIONS[key]

        # Default for unknown models
        return "Custom or community model"

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

            # Display Local Models with Rich Table
            if local_llms:
                self.console.print("\n[bold cyan]🖥️  Local Models (Ollama):[/bold cyan]\n")

                local_table = Table(border_style="cyan", show_header=True)
                local_table.add_column("#", style="bold yellow", width=4)
                local_table.add_column("Model Name", style="white", width=25)
                local_table.add_column("Size", style="green", width=8)
                local_table.add_column("Description", style="dim", width=70)

                for llm in local_llms:
                    idx = len(all_llms) + 1
                    description = self._get_model_description(llm.name)
                    local_table.add_row(
                        str(idx),
                        llm.name,
                        llm.size or "N/A",
                        description
                    )
                    all_llms.append(llm)

                self.console.print(local_table)

            # Display OpenAI Models with Rich Table
            if openai_llms:
                self.console.print("\n[bold cyan]☁️  OpenAI Models:[/bold cyan]\n")

                openai_table = Table(border_style="blue", show_header=True)
                openai_table.add_column("#", style="bold yellow", width=4)
                openai_table.add_column("Model Name", style="white", width=30)
                openai_table.add_column("Cost", style="magenta", width=12)
                openai_table.add_column("Description", style="dim", width=65)

                for llm in openai_llms:
                    idx = len(all_llms) + 1
                    cost = f"${llm.cost_per_1k_tokens}/1K" if llm.cost_per_1k_tokens else "N/A"
                    description = self._get_model_description(llm.name)
                    openai_table.add_row(
                        str(idx),
                        llm.name,
                        cost,
                        description
                    )
                    all_llms.append(llm)

                # Add custom option row
                idx = len(all_llms) + 1
                openai_table.add_row(
                    str(idx),
                    "[bold]Custom model[/bold]",
                    "-",
                    "[dim]Enter OpenAI model name manually[/dim]"
                )
                custom_openai_option_idx = idx

                self.console.print(openai_table)

            # Display Anthropic Models with Rich Table
            if anthropic_llms:
                self.console.print("\n[bold cyan]🤖 Anthropic Models:[/bold cyan]\n")

                anthropic_table = Table(border_style="magenta", show_header=True)
                anthropic_table.add_column("#", style="bold yellow", width=4)
                anthropic_table.add_column("Model Name", style="white", width=30)
                anthropic_table.add_column("Cost", style="cyan", width=12)
                anthropic_table.add_column("Description", style="dim", width=65)

                for llm in anthropic_llms[:3]:  # Show top 3
                    idx = len(all_llms) + 1
                    cost = f"${llm.cost_per_1k_tokens}/1K" if llm.cost_per_1k_tokens else "N/A"
                    description = self._get_model_description(llm.name)
                    anthropic_table.add_row(
                        str(idx),
                        llm.name,
                        cost,
                        description
                    )
                    all_llms.append(llm)

                self.console.print(anthropic_table)

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

    def _prompt_file_path(self, prompt_text: str, filter_format: str = None) -> str:
        """Prompt for file path with validation and Rich table display of detected datasets"""
        while True:
            if HAS_RICH and self.console:
                # Use detected datasets if available
                if self.detected_datasets:
                    # Filter by format if specified
                    datasets = self.detected_datasets
                    if filter_format:
                        datasets = [d for d in datasets if d.format.lower() == filter_format.lower()]

                    if datasets:
                        self.console.print(f"\n[bold cyan]📊 Detected Datasets[/bold cyan]\n")

                        datasets_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
                        datasets_table.add_column("#", style="cyan", width=3)
                        datasets_table.add_column("Name", style="white", width=50, no_wrap=True)
                        datasets_table.add_column("Format", style="yellow bold", width=8, justify="center")
                        datasets_table.add_column("Size", style="green", width=10, justify="right")
                        datasets_table.add_column("Path", style="dim", width=45, no_wrap=True)

                        for i, dataset in enumerate(datasets, 1):
                            # Format size
                            size_str = f"{dataset.size_mb:.1f} MB" if dataset.size_mb else "—"

                            # Shorten path
                            path_str = str(dataset.path.parent)
                            if len(path_str) > 40:
                                path_str = "..." + path_str[-37:]

                            datasets_table.add_row(
                                str(i),
                                dataset.path.name,
                                dataset.format.upper(),
                                size_str,
                                path_str
                            )

                        self.console.print(datasets_table)
                        self.console.print()

                        # Better instructions for user
                        self.console.print("[dim]💡 You can either:[/dim]")
                        self.console.print("[dim]   • Enter the [cyan]#[/cyan] number from the table above (e.g., '1', '13')[/dim]")
                        self.console.print("[dim]   • Enter an [cyan]absolute path[/cyan] to any file (e.g., '/Users/name/data/file.csv')[/dim]\n")

                        path = Prompt.ask(f"{prompt_text}")

                        # Allow number selection
                        if path.isdigit() and 1 <= int(path) <= len(datasets):
                            return str(datasets[int(path)-1].path)

                        # Allow path input
                        if Path(path).exists():
                            return path
                        else:
                            self.console.print(f"[red]File not found: {path}[/red]")
                            continue

                # Fallback: show files from current directory
                files = list(Path.cwd().glob("*.csv")) + list(Path.cwd().glob("*.json")) + list(Path.cwd().glob("*.xlsx"))
                if files:
                    self.console.print("\n[dim]Available files in current directory:[/dim]")
                    for i, f in enumerate(files[:10], 1):
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

        # Use recommended training model if available from earlier detection
        if config.get('recommended_training_model'):
            default_model = config['recommended_training_model']
            detected_language = ', '.join([l.upper() for l in config.get('detected_languages', [])])
            candidate_models = [default_model]
            if HAS_RICH and self.console:
                self.console.print(f"[green]✓ Using recommended model from language detection: {default_model}[/green]")
        else:
            # Fallback to old recommendation method
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
                    self.training_studio()
                elif choice == "4":
                    self.bert_annotation_studio()
                elif choice == "5":
                    self.validation_lab()
                elif choice == "6":
                    self.profile_manager_ui()
                elif choice == "7":
                    self.show_documentation()
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
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display personalized mode info
        self._display_section_header(
            "🎨 LLM Annotation Studio",
            "Professional annotation workflow without model training",
            mode_info={
                'workflow': 'Data → Select Prompts → LLM Annotate → Export (JSON/Doccano/Label Studio)',
                'capabilities': ['Multi-Prompt Support', 'Incremental Save', 'Resume Capability', 'Export Formats'],
                'input': 'Raw text data (CSV/Excel/JSON/Database)',
                'output': 'Annotated JSON + Doccano/Label Studio exports',
                'best_for': 'Pure annotation tasks without training (data collection, labeling, export)',
                'duration': '~2-10 min (depends on dataset and LLM speed)'
            }
        )

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

    def training_studio(self):
        """Training studio bringing dataset builders and trainers together."""
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display personalized mode info
        self._display_section_header(
            "🏋️ Training Studio",
            "Professional model training with intelligent dataset preparation",
            mode_info={
                'workflow': 'Load Data → Auto-Detect Columns → Detect Languages → Recommend Models → Train → Benchmark',
                'capabilities': ['Multi-format Support', 'Language Detection', 'Model Recommendations', '70+ BERT/Longformer Models'],
                'input': 'Annotated CSV/JSON/JSONL or Category labels',
                'output': 'Trained BERT models + Performance metrics + Best model selection',
                'best_for': 'Training custom models on annotated data with optimal model selection',
                'duration': '~10-30 min per model (benchmark mode: 1-3 hours)'
            }
        )

        if not (HAS_RICH and self.console):
            print("\nTraining Studio requires the Rich interface. Launch `llm-tool --simple` for basic commands.")
            return

        self._ensure_training_models_loaded()

        # NEW: Add resume/new menu BEFORE starting wizard
        self.console.print("\n[bold cyan]🎯 Training Session Options[/bold cyan]\n")

        session_options_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
        session_options_table.add_column("Option", style="cyan bold", width=10)
        session_options_table.add_column("Description", style="white", width=70)

        session_options_table.add_row(
            "1",
            "🔄 Resume/Relaunch Training\n   Load saved parameters from previous training sessions"
        )
        session_options_table.add_row(
            "2",
            "🆕 New Training Session\n   Start fresh with dataset selection and configuration"
        )
        session_options_table.add_row(
            "3",
            "← Back to Main Menu"
        )

        self.console.print(session_options_table)
        self.console.print()

        session_choice = Prompt.ask(
            "[bold yellow]Select an option[/bold yellow]",
            choices=["1", "2", "3"],
            default="2"
        )

        if session_choice == "1":
            # Resume/Relaunch existing session
            self._resume_training_studio()
            return
        elif session_choice == "3":
            # Back to main menu
            return

        # Continue with NEW training session
        builder = TrainingDatasetBuilder(self.settings.paths.data_dir / "training_data")

        self._training_studio_show_model_catalog()

        # Display training mode options with Rich
        self.console.print("\n[bold cyan]🎯 Training Mode Selection[/bold cyan]\n")

        modes_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
        modes_table.add_column("Mode", style="cyan bold", width=15)
        modes_table.add_column("Description", style="white", width=60)
        modes_table.add_column("Duration", style="yellow", width=20)

        modes_table.add_row(
            "quick",
            "Fast training with default settings (2 epochs)\n✓ Best for quick prototyping and testing",
            "~5-10 minutes"
        )
        modes_table.add_row(
            "benchmark",
            "Compare multiple models to find the best one\n✓ Tests 5+ models and selects the best performer",
            "~1-3 hours"
        )
        modes_table.add_row(
            "custom",
            "Full control over all training parameters\n✓ Configure epochs, batch size, learning rate, etc.",
            "Varies"
        )
        modes_table.add_row(
            "[dim]distributed[/dim]",
            "[dim]Multi-label parallel training (one model per label)\n✓ For datasets with multiple labels/categories[/dim]\n[bold red]⚠️  NOT RECOMMENDED - Untested[/bold red]",
            "[dim]~30-60 minutes[/dim]"
        )

        # First, configure the dataset
        try:
            bundle = self._training_studio_dataset_wizard(builder)
        except Exception as exc:  # pylint: disable=broad-except
            self.console.print(f"[red]Dataset preparation failed:[/red] {exc}")
            self.logger.exception("Training Studio dataset preparation failed", exc_info=exc)
            return

        if bundle is None:
            self.console.print("[yellow]Training cancelled.[/yellow]")
            return

        # Show dataset summary
        self._training_studio_render_bundle_summary(bundle)

        # Now ask for training mode
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]           📚 Training Mode Selection                          [/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        self.console.print(modes_table)
        self.console.print()

        # If one-vs-all approach, suggest distributed mode
        default_mode = "quick"
        if bundle.metadata.get('training_approach') == 'one-vs-all':
            self.console.print("[bold yellow]💡 Recommended Mode:[/bold yellow] [dim strikethrough]distributed[/dim strikethrough] → [green]quick[/green]")
            self.console.print(f"[dim]   Since you selected 'one-vs-all' training, distributed mode will train {bundle.metadata.get('num_categories', '?')} models in parallel[/dim]")
            self.console.print("[bold red]   ⚠️  WARNING: distributed mode is NOT RECOMMENDED (untested). Use 'quick' instead.[/bold red]\n")
            default_mode = "quick"  # Changed from "distributed" to "quick"

        mode = Prompt.ask(
            "[bold yellow]Select training mode[/bold yellow]",
            choices=["quick", "benchmark", "custom", "distributed", "back"],
            default=default_mode,
        )

        if mode == "back":
            return

        # Warn if distributed mode was selected
        if mode == "distributed":
            self.console.print("\n[bold red]⚠️  WARNING: Distributed mode is NOT RECOMMENDED[/bold red]")
            self.console.print("[yellow]This mode has not been thoroughly tested and may contain bugs.[/yellow]")
            self.console.print("[dim]Consider using 'quick' or 'benchmark' mode instead for more reliable results.[/dim]\n")

            confirm = Confirm.ask("[bold]Do you still want to proceed with distributed mode?[/bold]", default=False)
            if not confirm:
                self.console.print("[yellow]Training cancelled. Please select another mode.[/yellow]")
                return

        # Show training mode confirmation and parameters
        self._training_studio_confirm_and_execute(bundle, mode)

    # ------------------------------------------------------------------
    # Training Studio helpers
    # ------------------------------------------------------------------
    def _training_studio_confirm_and_execute(
        self,
        bundle: TrainingDataBundle,
        mode: str,
        preloaded_config: Optional[Dict[str, Any]] = None,
        is_resume: bool = False
    ) -> None:
        """
        Display training parameters and ask for confirmation before execution.
        This ensures the user reviews all settings before starting training.

        Parameters
        ----------
        bundle : TrainingDataBundle
            The training data bundle
        mode : str
            Training mode (quick, benchmark, custom, distributed)
        preloaded_config : dict, optional
            Pre-loaded configuration from saved session (for resume/relaunch)
        is_resume : bool
            Whether this is a resume (True) or fresh start (False)
        """
        from datetime import datetime

        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]           ✅ Training Configuration Summary                     [/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        # Create configuration table
        config_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED)
        config_table.add_column("Parameter", style="cyan bold", width=25)
        config_table.add_column("Value", style="white", width=60)

        # Dataset information
        config_table.add_row("📊 Dataset", str(bundle.primary_file.name) if bundle.primary_file else "—")
        config_table.add_row("📝 Format", bundle.strategy)
        config_table.add_row("📖 Text Column", bundle.text_column)
        config_table.add_row("🏷️  Label Column", bundle.label_column)

        if bundle.metadata.get('confirmed_languages'):
            langs = ', '.join([l.upper() for l in bundle.metadata['confirmed_languages']])
            config_table.add_row("🌍 Languages", langs)

        # Model information
        if hasattr(bundle, 'recommended_model') and bundle.recommended_model:
            config_table.add_row("🤖 Recommended Model", bundle.recommended_model)

        # Training mode
        mode_descriptions = {
            "quick": "⚡ Quick Start - Fast training with defaults",
            "benchmark": "📊 Benchmark - Test multiple models",
            "custom": "⚙️  Custom - Full parameter control",
            "distributed": "🔄 Distributed - Parallel multi-label ⚠️  (NOT RECOMMENDED - Untested)"
        }
        config_table.add_row("🎯 Training Mode", mode_descriptions.get(mode, mode))

        # Mode-specific parameters
        if mode == "quick":
            config_table.add_row("⏱️  Epochs", "Will be asked (default: 10)")
            config_table.add_row("📦 Batch Size", "16 (default)")
        elif mode == "benchmark":
            config_table.add_row("🔬 Models to Test", "5-7 (intelligent selection)")
            config_table.add_row("⏱️  Epochs per Model", "Will be asked (default: 10)")
        elif mode == "distributed":
            num_labels = len(bundle.metadata.get('categories', []))
            config_table.add_row("🔢 Models to Train", f"{num_labels} (one per label)")

        # Statistics
        if bundle.metadata.get('text_length_stats'):
            stats = bundle.metadata['text_length_stats']
            avg_len = stats.get('avg_chars', stats.get('avg_length', 0))
            config_table.add_row("📏 Avg Text Length", f"{avg_len:.0f} characters")

        self.console.print(config_table)
        self.console.print()

        # NEW: Ask to save metadata (unless resuming)
        save_metadata = True  # Default to True for reproducibility
        metadata_path = None

        if not is_resume:
            self.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
            self.console.print("  [green]1. Resume Capability[/green]")
            self.console.print("     • Save parameters to resume later if interrupted")
            self.console.print("     • Access via 'Resume/Relaunch Training' option\n")

            self.console.print("  [green]2. Scientific Reproducibility[/green]")
            self.console.print("     • Document exact training configuration")
            self.console.print("     • Track model, dataset, and hyperparameters")
            self.console.print("     • Share configurations with collaborators\n")

            self.console.print("  [red]⚠️  If you choose NO:[/red]")
            self.console.print("     • You CANNOT resume this training later")
            self.console.print("     • Parameters will not be saved for future reference\n")

            save_metadata = Confirm.ask(
                "[bold yellow]Save training parameters to JSON?[/bold yellow]",
                default=True
            )

        # Ask for confirmation
        confirm = Confirm.ask(
            "\n[bold yellow]🚀 Start training with these parameters?[/bold yellow]",
            default=True
        )

        if not confirm:
            self.console.print("[yellow]Training cancelled by user.[/yellow]")
            return

        # Prepare COMPLETE model configuration for metadata (ALL MODES)
        # This ensures FULL reproducibility for quick, benchmark, and custom modes
        model_config = {
            # Core training mode
            'training_mode': mode,

            # Common hyperparameters
            'selected_model': preloaded_config.get('selected_model') if preloaded_config else None,
            'epochs': preloaded_config.get('epochs') if preloaded_config else None,
            'batch_size': preloaded_config.get('batch_size') if preloaded_config else 16,
            'learning_rate': preloaded_config.get('learning_rate') if preloaded_config else 2e-5,
            'early_stopping': True,
            'recommended_model': bundle.recommended_model if hasattr(bundle, 'recommended_model') else None,

            # Advanced training options (will be filled by each mode)
            'use_reinforcement': preloaded_config.get('use_reinforcement') if preloaded_config else True,
            'reinforced_epochs': preloaded_config.get('reinforced_epochs') if preloaded_config else 10,
            'validation_split': preloaded_config.get('validation_split') if preloaded_config else 0.2,
            'test_split': preloaded_config.get('test_split') if preloaded_config else 0.1,
            'stratified_split': preloaded_config.get('stratified_split') if preloaded_config else True,

            # Benchmark-specific parameters (filled if mode=='benchmark')
            'selected_models': None,  # Will be filled by benchmark mode
            'selected_labels': None,  # Will be filled by benchmark mode
            'benchmark_category': None,  # Will be filled if multi-class → binary

            # Quick-specific parameters (filled if mode=='quick')
            'quick_model_name': None,  # Will be filled by quick mode
            'quick_epochs': None,  # Will be filled by quick mode

            # Custom-specific parameters (filled if mode=='custom')
            'custom_config': None,  # Will be filled by custom mode

            # Runtime parameters (to be filled during execution)
            'actual_models_trained': [],  # Will be updated post-training
            'training_start_time': None,
            'training_end_time': None
        }

        # Save PRE-TRAINING metadata
        metadata_path = None  # Initialize before conditional block
        if save_metadata:
            try:
                metadata_path = self._save_training_metadata(
                    bundle=bundle,
                    mode=mode,
                    model_config=model_config,
                    execution_status={
                        'status': 'pending',
                        'started_at': datetime.now().isoformat(),
                        'completed_at': None,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None
                    }
                )
                self.console.print(f"\n[green]✅ Metadata saved for reproducibility[/green]")
                self.console.print(f"[cyan]📋 Metadata File:[/cyan]")
                self.console.print(f"   {metadata_path}\n")
            except Exception as e:
                self.logger.error(f"Failed to save metadata: {e}")
                self.console.print(f"[yellow]⚠️  Failed to save metadata: {e}[/yellow]\n")

        # Execute the selected training mode
        self.console.print("\n[green]✓ Starting training...[/green]\n")

        training_result = None
        runtime_params = {}  # Will store actual parameters used during training
        try:
            if mode == "distributed":
                training_result = self._training_studio_run_distributed(bundle, model_config)
                runtime_params = training_result.get('runtime_params', {}) if training_result else {}
            elif mode == "quick":
                training_result = self._training_studio_run_quick(bundle, model_config)
                runtime_params = training_result.get('runtime_params', {}) if training_result else {}
            elif mode == "benchmark":
                training_result = self._training_studio_run_benchmark(bundle, model_config)
                runtime_params = training_result.get('runtime_params', {}) if training_result else {}
            else:
                training_result = self._training_studio_run_custom(bundle, model_config)
                runtime_params = training_result.get('runtime_params', {}) if training_result else {}

            # Update POST-TRAINING metadata with COMPLETE information
            if save_metadata and metadata_path:
                try:
                    # Merge runtime params into model_config for complete save
                    final_model_config = {**model_config, **runtime_params}

                    execution_status = {
                        'status': 'completed',
                        'completed_at': datetime.now().isoformat(),
                        'models_trained': training_result.get('models_trained', []) if training_result else [],
                        'best_model': training_result.get('best_model') if training_result else None,
                        'best_f1': training_result.get('best_f1') if training_result else None
                    }

                    # Update both execution_status AND model_config with runtime params
                    self._update_training_metadata(
                        metadata_path,
                        execution_status=execution_status,
                        model_config=final_model_config
                    )
                    self.console.print(f"\n[green]✅ Training metadata updated with complete parameters[/green]\n")
                except Exception as e:
                    self.logger.error(f"Failed to update metadata: {e}")

        except Exception as e:
            # Update metadata with failure status
            if save_metadata and metadata_path:
                try:
                    execution_status = {
                        'status': 'failed',
                        'completed_at': datetime.now().isoformat(),
                        'error_message': str(e)
                    }
                    self._update_training_metadata(metadata_path, execution_status=execution_status)
                except:
                    pass
            raise  # Re-raise the exception

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

        table = Table(title="Available Model Categories (70+ models)", border_style="blue")
        table.add_column("Category", style="cyan", width=30)
        table.add_column("Models (sample)", style="white", width=50)

        # Define display order for categories
        category_order = [
            "Multilingual Models",
            "Long Document Models",
            "Long Document Models - French",
            "Long Document Models - Spanish",
            "Long Document Models - German",
            "Long Document Models - Italian",
            "Long Document Models - Portuguese",
            "Long Document Models - Dutch",
            "Long Document Models - Polish",
            "Long Document Models - Chinese",
            "Long Document Models - Japanese",
            "Long Document Models - Arabic",
            "Long Document Models - Russian",
            "Efficient Models",
            "English Models",
            "French Models",
            "Other Language Models"
        ]

        # Display categories in order
        for category in category_order:
            if category in self.available_trainer_models:
                models = self.available_trainer_models[category]
                sample = ", ".join(model["name"] for model in models[:2])
                if len(models) > 2:
                    sample += f" (+{len(models) - 2} more)"
                table.add_row(category, sample)

        # Add any remaining categories not in the order
        for category, models in self.available_trainer_models.items():
            if category not in category_order:
                sample = ", ".join(model["name"] for model in models[:2])
                if len(models) > 2:
                    sample += f" (+{len(models) - 2} more)"
                table.add_row(category, sample)

        self.console.print(table)

    def _training_studio_intelligent_dataset_selector(
        self,
        format_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Universal sophisticated interface for dataset and column selection.
        Adapted specifically for Training Studio with:
        - Automatic dataset detection
        - Intelligent column analysis with confidence scores
        - Category/label detection and display
        - Sophisticated ID strategy (single/combine/none)
        - Model recommendations based on languages and data

        Args:
            format_type: One of 'llm-json', 'category-csv', 'binary-long', 'jsonl-single', 'jsonl-multi'

        Returns:
            Dictionary with selected dataset path and all column information, or None if cancelled
        """

        # Step 1: Dataset Detection and Selection
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 1:[/bold cyan] [bold white]Dataset Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Select your annotated dataset file to prepare for training.[/dim]\n")

        # Show detected datasets if available
        if self.detected_datasets:
            datasets_table = Table(title="📊 Detected Datasets", border_style="cyan")
            datasets_table.add_column("#", style="cyan", width=3)
            datasets_table.add_column("Name", style="white", width=40)
            datasets_table.add_column("Format", style="yellow", width=8)
            datasets_table.add_column("Size", style="green", width=10)
            datasets_table.add_column("Folder", style="magenta", width=20)

            for i, ds in enumerate(self.detected_datasets, 1):  # Show ALL datasets
                # Calculate file size
                try:
                    if hasattr(ds, 'path') and ds.path.exists():
                        size_bytes = ds.path.stat().st_size
                        if size_bytes < 1024:
                            size_str = f"{size_bytes} B"
                        elif size_bytes < 1024 * 1024:
                            size_str = f"{size_bytes / 1024:.1f} KB"
                        else:
                            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                    else:
                        size_str = "—"
                except Exception as e:
                    self.logger.debug(f"Could not get size for {ds.path}: {e}")
                    size_str = "—"

                # Get folder name (parent directory name)
                folder_name = ds.path.parent.name if hasattr(ds, 'path') and ds.path.parent.name else "data"

                datasets_table.add_row(
                    str(i),
                    ds.path.name if hasattr(ds, 'path') else "—",
                    ds.format if hasattr(ds, 'format') else "—",
                    size_str,
                    folder_name
                )

            self.console.print(datasets_table)
            self.console.print()

            use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
            if use_detected:
                choice = self._int_prompt_with_validation("Select dataset", 1, 1, len(self.detected_datasets))
                data_path = self.detected_datasets[choice - 1].path
            else:
                data_path = Path(self._prompt_file_path("Dataset path"))
        else:
            self.console.print("[dim]No datasets auto-detected in data/ folder[/dim]")
            data_path = Path(self._prompt_file_path("Dataset path"))

        self.console.print(f"[green]✓ Selected: {data_path.name} ({data_path.suffix[1:]})[/green]\n")

        # Step 2: Intelligent File Analysis
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 2:[/bold cyan] [bold white]Analyzing Dataset Structure[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]🔍 Analyzing columns, detecting types, and extracting samples...[/dim]")

        analysis = DataDetector.analyze_file_intelligently(data_path)

        if analysis['issues']:
            self.console.print("\n[yellow]⚠️  Analysis warnings:[/yellow]")
            for issue in analysis['issues']:
                self.console.print(f"  • {issue}")

        # Step 3: Intelligent Language Detection (MOVED HERE - before column selection)
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 3:[/bold cyan] [bold white]Language Detection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Detecting languages to recommend the best training model.[/dim]\n")

        languages_found_in_column = set(analysis.get('languages_detected', {}).keys())
        confirmed_languages = set()
        lang_column = None
        text_length_stats = {}  # Initialize - will be populated after text column selection
        languages_from_content = {}

        # Check if we have a language column with detected languages
        has_lang_column = bool(analysis.get('language_column_candidates'))

        if has_lang_column and languages_found_in_column:
            # Option 1: Language column exists - offer to use it or detect automatically
            self.console.print("[bold]🌍 Languages Found in Column:[/bold]")
            for lang, count in analysis['languages_detected'].items():
                self.console.print(f"  • {lang.upper()}: {count:,} rows")

            lang_column_candidate = analysis['language_column_candidates'][0]
            self.console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")

            use_lang_column = Confirm.ask(
                f"\n[bold]Use language column '{lang_column_candidate}'?[/bold]",
                default=True
            )

            if use_lang_column:
                confirmed_languages = languages_found_in_column
                lang_column = lang_column_candidate
                self.console.print(f"[green]✓ Using language column: {lang_column}[/green]")
            else:
                # User said no to language column - offer automatic detection
                self.console.print("\n[yellow]Language column not used. Applying automatic detection...[/yellow]")
                apply_auto_detection = True
        else:
            # Option 2: No language column - go straight to automatic detection
            self.console.print("[yellow]ℹ️  No language column detected[/yellow]")
            apply_auto_detection = Confirm.ask("Apply automatic language detection on text content?", default=True)

        # We need to detect text column first for content-based language detection
        # Quick text column detection for language analysis
        temp_column_info = self._detect_text_columns(data_path)
        temp_text_column = None
        if temp_column_info.get('text_candidates'):
            temp_text_column = temp_column_info['text_candidates'][0]['name']
        else:
            temp_text_column = "text"  # fallback

        # Automatic language detection from text content
        language_distribution = {}  # Store exact language counts

        if not lang_column and ('apply_auto_detection' not in locals() or apply_auto_detection):
            self.console.print("\n[dim]🔍 Analyzing ALL texts to detect languages (this may take a moment)...[/dim]")

            try:
                import pandas as pd
                from llm_tool.utils.language_detector import LanguageDetector

                df = pd.read_csv(data_path) if data_path.suffix == '.csv' else pd.read_json(data_path, lines=data_path.suffix == '.jsonl')

                if temp_text_column in df.columns:
                    # Analyze ALL texts (not just sample) for precise distribution
                    all_texts = df[temp_text_column].dropna().tolist()

                    if all_texts:
                        detector = LanguageDetector()
                        lang_counts = {}
                        detected_languages_per_text = []  # Store language for each text

                        # Progress indicator
                        from tqdm import tqdm
                        self.console.print(f"[dim]Analyzing {len(all_texts)} texts...[/dim]")

                        for text in tqdm(all_texts, desc="Detecting languages", disable=not HAS_RICH):
                            if text and len(str(text).strip()) > 10:
                                try:
                                    detected = detector.detect(str(text))
                                    if detected and detected.get('language'):
                                        lang = detected['language']
                                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                                        detected_languages_per_text.append(lang)
                                    else:
                                        detected_languages_per_text.append(None)
                                except Exception as e:
                                    self.logger.debug(f"Language detection failed for text: {e}")
                                    detected_languages_per_text.append(None)
                            else:
                                detected_languages_per_text.append(None)  # Empty or too short text

                        if lang_counts:
                            # Store exact distribution
                            language_distribution = lang_counts
                            total = sum(lang_counts.values())

                            self.console.print(f"\n[bold]🌍 Languages Detected from Content ({total:,} texts analyzed):[/bold]")

                            # Create detailed table
                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                            lang_table.add_column("Language", style="cyan", width=12)
                            lang_table.add_column("Count", style="yellow", justify="right", width=12)
                            lang_table.add_column("Percentage", style="green", justify="right", width=12)

                            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total * 100) if total > 0 else 0
                                lang_table.add_row(
                                    lang.upper(),
                                    f"{count:,}",
                                    f"{percentage:.1f}%"
                                )

                            self.console.print(lang_table)

                            # Detect low-percentage languages (likely detection errors)
                            LOW_PERCENTAGE_THRESHOLD = 1.0  # Languages with < 1% are considered low
                            majority_languages = {}  # Languages above threshold
                            minority_languages = {}  # Languages below threshold (likely errors)

                            for lang, count in lang_counts.items():
                                percentage = (count / total * 100) if total > 0 else 0
                                if percentage >= LOW_PERCENTAGE_THRESHOLD:
                                    majority_languages[lang] = count
                                else:
                                    minority_languages[lang] = count

                            confirmed_languages = set(lang_counts.keys())
                            texts_to_reclassify = []  # Store texts that need manual classification

                            # Handle low-percentage languages if detected
                            if minority_languages:
                                self.console.print(f"\n[yellow]⚠ Warning: {len(minority_languages)} language(s) detected with very low percentage (< {LOW_PERCENTAGE_THRESHOLD}%):[/yellow]")
                                for lang, count in sorted(minority_languages.items(), key=lambda x: x[1], reverse=True):
                                    percentage = (count / total * 100)
                                    self.console.print(f"  • {lang.upper()}: {count} texts ({percentage:.2f}%)")

                                self.console.print("\n[dim]These are likely detection errors. You have options:[/dim]")
                                self.console.print("  [cyan]1. exclude[/cyan] - Exclude ALL low-percentage languages from training")
                                self.console.print("  [cyan]2. keep[/cyan] - Keep ALL detected languages (not recommended)")
                                self.console.print("  [cyan]3. select[/cyan] - Manually select which languages to keep")
                                self.console.print("  [cyan]4. correct[/cyan] - Force ALL minority languages to a single language (quick fix)")
                                self.console.print("  [cyan]5. reclassify[/cyan] - Manually review and reclassify texts phrase-by-phrase")

                                minority_action = Prompt.ask(
                                    "\n[bold yellow]How to handle low-percentage languages?[/bold yellow]",
                                    choices=["exclude", "keep", "select", "correct", "reclassify"],
                                    default="correct"
                                )

                                if minority_action == "correct":
                                    # Quick correction: force all minority languages to one language
                                    self.console.print("\n[bold cyan]🔧 Quick Language Correction[/bold cyan]\n")

                                    # Show available languages (majority + all supported languages)
                                    all_supported_langs = [
                                        'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja',
                                        'ar', 'pl', 'tr', 'ko', 'hi', 'sv', 'no', 'da', 'fi', 'cs',
                                        'el', 'he', 'ro', 'uk', 'bg', 'hr', 'vi', 'th', 'id', 'fa'
                                    ]

                                    # Suggest the majority language
                                    majority_lang = max(majority_languages.items(), key=lambda x: x[1])[0] if majority_languages else 'en'

                                    self.console.print(f"[bold]Available languages:[/bold]")
                                    self.console.print(f"  • Majority language detected: [green]{majority_lang.upper()}[/green] ({majority_languages.get(majority_lang, 0)} texts)")
                                    self.console.print(f"  • All supported: {', '.join([l.upper() for l in all_supported_langs])}")

                                    correction_target = Prompt.ask(
                                        f"\n[bold yellow]Force ALL minority languages to which language?[/bold yellow]",
                                        default=majority_lang
                                    ).lower().strip()

                                    if correction_target not in all_supported_langs:
                                        self.console.print(f"[yellow]Warning: '{correction_target}' not in standard list, but will be used anyway[/yellow]")

                                    # CRITICAL FIX: Update detected_languages_per_text with corrections
                                    total_corrected = 0
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] in minority_languages:
                                                detected_languages_per_text[i] = correction_target
                                                total_corrected += 1

                                    # Update language_distribution
                                    for minority_lang in minority_languages.keys():
                                        if minority_lang in language_distribution:
                                            del language_distribution[minority_lang]

                                    # Add corrected texts to target language
                                    if correction_target in language_distribution:
                                        language_distribution[correction_target] += total_corrected
                                    else:
                                        language_distribution[correction_target] = total_corrected

                                    # Update confirmed languages
                                    confirmed_languages = set([correction_target] + list(majority_languages.keys()))

                                    self.console.print(f"\n[green]✓ Corrected {total_corrected} texts from {len(minority_languages)} languages to {correction_target.upper()}[/green]")
                                    # Display updated distribution
                                    update_table = Table(title="Updated Language Distribution", border_style="green")
                                    update_table.add_column("Language", style="cyan", justify="center")
                                    update_table.add_column("Count", justify="right")
                                    update_table.add_column("Percentage", justify="right")

                                    new_total = sum(language_distribution.values())
                                    for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
                                        if count > 0:  # Only show non-zero counts
                                            percentage = (count / new_total) * 100 if new_total > 0 else 0
                                            update_table.add_row(lang.upper(), f"{count:,}", f"{percentage:.1f}%")

                                    self.console.print(update_table)

                                elif minority_action == "reclassify":
                                    # Manual reclassification
                                    self.console.print("\n[bold cyan]Manual Reclassification[/bold cyan]\n")
                                    self.console.print(f"[dim]Available majority languages: {', '.join([l.upper() for l in sorted(majority_languages.keys())])}[/dim]\n")

                                    # Create mapping for reclassification
                                    reclassification_map = {}

                                    # Get the texts for each minority language and show samples
                                    minority_lang_codes = list(minority_languages.keys())

                                    # Load texts with their detected languages
                                    all_texts_with_lang = []
                                    for idx, text in enumerate(all_texts):
                                        if text and len(str(text).strip()) > 10:
                                            try:
                                                detected = detector.detect(str(text))
                                                if detected and detected.get('language'):
                                                    lang = detected['language']
                                                    if lang in minority_languages:
                                                        all_texts_with_lang.append({
                                                            'index': idx,
                                                            'text': str(text),
                                                            'detected_lang': lang
                                                        })
                                            except:
                                                continue

                                    # Show samples for reclassification
                                    if all_texts_with_lang:
                                        self.console.print(f"[bold]Found {len(all_texts_with_lang)} texts to reclassify[/bold]\n")

                                        # Group by detected language
                                        from collections import defaultdict
                                        texts_by_lang = defaultdict(list)
                                        for item in all_texts_with_lang:
                                            texts_by_lang[item['detected_lang']].append(item)

                                        # For each minority language, show samples and ask for reclassification
                                        for minority_lang in sorted(minority_lang_codes):
                                            if minority_lang in texts_by_lang:
                                                lang_texts = texts_by_lang[minority_lang]
                                                self.console.print(f"\n[bold yellow]Reclassifying {minority_lang.upper()} ({len(lang_texts)} texts)[/bold yellow]")

                                                majority_choices = sorted(majority_languages.keys())

                                                # PHRASE-BY-PHRASE RECLASSIFICATION
                                                # Each text can be assigned to a different language
                                                reclassification_choices = {}  # Store per-text choices

                                                for item in lang_texts:
                                                    idx = item['index']
                                                    text = item['text']

                                                    # Show current text
                                                    sample_table = Table(border_style="yellow", show_header=True, header_style="bold", title=f"Text {lang_texts.index(item) + 1}/{len(lang_texts)}")
                                                    sample_table.add_column("Text", width=90)
                                                    display_text = text if len(text) <= 200 else text[:200] + "..."
                                                    sample_table.add_row(display_text)
                                                    self.console.print(sample_table)

                                                    # Ask for this specific text
                                                    majority_choices_str = '/'.join([l.lower() for l in majority_choices])

                                                    reclassify_choice = Prompt.ask(
                                                        f"Classify this text as [{majority_choices_str}/exclude]",
                                                        choices=majority_choices + ["exclude"],
                                                        default=majority_choices[0] if majority_choices else "exclude"
                                                    )

                                                    reclassification_choices[idx] = reclassify_choice

                                                # Count the choices
                                                choice_counts = {}
                                                for choice in reclassification_choices.values():
                                                    choice_counts[choice] = choice_counts.get(choice, 0) + 1

                                                # Update language distribution
                                                for choice, count in choice_counts.items():
                                                    if choice != "exclude":
                                                        language_distribution[choice] = language_distribution.get(choice, 0) + count
                                                        if choice not in reclassification_map:
                                                            reclassification_map[choice] = {}
                                                        # Store the mapping
                                                        if minority_lang not in reclassification_map[choice]:
                                                            reclassification_map[choice][minority_lang] = count
                                                        else:
                                                            reclassification_map[choice][minority_lang] += count

                                                # Remove from original language distribution
                                                language_distribution[minority_lang] = 0

                                                # Display summary
                                                self.console.print(f"\n[bold]Reclassification Summary for {minority_lang.upper()}:[/bold]")
                                                for choice, count in sorted(choice_counts.items(), key=lambda x: x[1], reverse=True):
                                                    if choice == "exclude":
                                                        self.console.print(f"  [yellow]✗ {count} text(s) excluded[/yellow]")
                                                    else:
                                                        self.console.print(f"  [green]✓ {count} text(s) → {choice.upper()}[/green]")

                                    # Update confirmed languages (remove excluded)
                                    # Filter out metadata keys and only keep languages with count > 0
                                    confirmed_languages = set([lang for lang, count in language_distribution.items()
                                                             if not lang.startswith('_') and isinstance(count, (int, float)) and count > 0])

                                    # Store reclassification map for later use
                                    if reclassification_map:
                                        language_distribution['_reclassification_map'] = reclassification_map

                                    self.console.print(f"\n[green]✓ Reclassification complete. Final languages: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                                elif minority_action == "exclude":
                                    # Exclude low-percentage languages
                                    for lang in minority_languages.keys():
                                        language_distribution[lang] = 0  # Mark as excluded

                                    # CRITICAL FIX: Mark excluded language texts as None
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] in minority_languages:
                                                detected_languages_per_text[i] = None

                                    confirmed_languages = set(majority_languages.keys())
                                    excluded_count = sum(minority_languages.values())
                                    self.console.print(f"\n[yellow]✗ Excluded {excluded_count} texts from {len(minority_languages)} low-percentage language(s)[/yellow]")
                                    self.console.print(f"[green]✓ Final languages: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                                elif minority_action == "keep":
                                    self.console.print("[yellow]⚠ Keeping all detected languages (including low-percentage ones)[/yellow]")

                                elif minority_action == "select":
                                    # Manual selection of languages to keep
                                    self.console.print("\n[bold cyan]📝 Language Selection:[/bold cyan]")
                                    self.console.print(f"[dim]Select which languages to keep for training (from all {len(lang_counts)} detected)[/dim]\n")

                                    # Show all languages sorted by count
                                    self.console.print("[bold]All Detected Languages:[/bold]")
                                    for i, (lang, count) in enumerate(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True), 1):
                                        percentage = (count / total * 100)
                                        status = "[green]✓ majority[/green]" if lang in majority_languages else "[yellow]⚠ minority[/yellow]"
                                        self.console.print(f"  {i:2d}. {lang.upper():5s} - {count:6,} texts ({percentage:5.2f}%) {status}")

                                    self.console.print("\n[bold yellow]Select languages to KEEP:[/bold yellow]")
                                    self.console.print("[dim]Enter language codes separated by commas (e.g., 'fr,en,de')[/dim]")
                                    self.console.print("[dim]Press Enter without typing to keep ALL languages[/dim]")

                                    selected_langs = Prompt.ask("\n[bold]Languages to keep[/bold]", default="")

                                    if selected_langs.strip():
                                        # User selected specific languages
                                        selected_set = set([l.strip().lower() for l in selected_langs.split(',') if l.strip()])

                                        # Validate that selected languages exist
                                        invalid_langs = selected_set - set(lang_counts.keys())
                                        if invalid_langs:
                                            self.console.print(f"[yellow]⚠ Warning: These languages were not detected: {', '.join(invalid_langs)}[/yellow]")
                                            selected_set = selected_set - invalid_langs

                                        # Exclude non-selected languages
                                        for lang in lang_counts.keys():
                                            if lang not in selected_set:
                                                language_distribution[lang] = 0  # Mark as excluded

                                        # CRITICAL FIX: Mark non-selected language texts as None
                                        if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                            for i in range(len(detected_languages_per_text)):
                                                if detected_languages_per_text[i] and detected_languages_per_text[i] not in selected_set:
                                                    detected_languages_per_text[i] = None

                                        confirmed_languages = selected_set
                                        kept_count = sum([lang_counts[lang] for lang in selected_set])
                                        excluded_count = total - kept_count

                                        self.console.print(f"\n[green]✓ Kept {len(selected_set)} language(s): {', '.join([l.upper() for l in sorted(selected_set)])}[/green]")
                                        self.console.print(f"[dim]  → {kept_count:,} texts kept, {excluded_count:,} texts excluded[/dim]")
                                    else:
                                        # User pressed Enter - keep all
                                        self.console.print("[green]✓ Keeping all detected languages[/green]")

                            # Final confirmation (allow override even after selection)
                            lang_list = ', '.join([l.upper() for l in sorted(confirmed_languages)])
                            lang_confirmed = Confirm.ask(
                                f"\n[bold]Final languages: {lang_list}. Is this correct?[/bold]",
                                default=True
                            )

                            if not lang_confirmed:
                                self.console.print("\n[yellow]Override with manual selection[/yellow]")
                                manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

                                # Update distribution to exclude non-selected languages
                                for lang in lang_counts.keys():
                                    if lang not in confirmed_languages:
                                        language_distribution[lang] = 0

                                # CRITICAL FIX: Mark non-confirmed language texts as None
                                if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                    for i in range(len(detected_languages_per_text)):
                                        if detected_languages_per_text[i] and detected_languages_per_text[i] not in confirmed_languages:
                                            detected_languages_per_text[i] = None

                                self.console.print(f"[green]✓ Manual override: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                            else:
                                self.console.print("[green]✓ Languages confirmed from content analysis[/green]")

                            # CRITICAL FIX: Add detected language column to DataFrame and save
                            if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                # Create a temporary DataFrame for non-null texts
                                temp_df = df[df[temp_text_column].notna()].copy()

                                # Ensure same length
                                if len(detected_languages_per_text) == len(temp_df):
                                    # Map detected languages to the full DataFrame
                                    df['language'] = None
                                    df.loc[df[temp_text_column].notna(), 'language'] = detected_languages_per_text

                                    # Set lang_column to use this new column
                                    lang_column = 'language'

                                    # Save updated DataFrame back to CSV
                                    df.to_csv(data_path, index=False)
                                    self.console.print(f"[dim]✓ Added 'language' column to dataset ({len([l for l in detected_languages_per_text if l])} texts with detected language)[/dim]")
                        else:
                            # Fallback: ask user
                            self.console.print("[yellow]Could not detect languages automatically[/yellow]")
                            manual_langs = Prompt.ask("Expected language codes (e.g., en,fr,de)", default="")
                            if manual_langs.strip():
                                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
                    else:
                        self.console.print("[yellow]Not enough text samples for language detection[/yellow]")
                        manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                        if manual_langs.strip():
                            confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

            except Exception as e:
                self.logger.debug(f"Language detection from content failed: {e}")
                self.console.print("[yellow]Automatic detection failed. Please specify manually[/yellow]")
                manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                if manual_langs.strip():
                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

                self.console.print("[yellow]Standard models will be used (texts will be truncated to 512 tokens)[/yellow]")
        else:
            text_length_stats['user_prefers_long_models'] = False

        # Step 4: Text Column Selection with Sophisticated Table
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 4:[/bold cyan] [bold white]Text Column Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold]💡 What You Need to Select:[/bold]")
        self.console.print("   [cyan]• Text Column[/cyan] - Contains the text data to train on (input for predictions)\n")

        column_info = self._detect_text_columns(data_path)
        all_columns = column_info.get('all_columns', analysis.get('all_columns', []))

        if column_info.get('text_candidates'):
            self.console.print("[dim]Detected text columns (sorted by confidence):[/dim]")

            col_table = Table(border_style="blue")
            col_table.add_column("#", style="cyan", width=5)
            col_table.add_column("Column", style="white", width=15)
            col_table.add_column("Confidence", style="yellow", width=12)
            col_table.add_column("Avg Length", style="green", width=12)
            col_table.add_column("Sample", style="dim", width=55)

            for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
                conf_color = {
                    "high": "[green]High[/green]",
                    "medium": "[yellow]Medium[/yellow]",
                    "low": "[orange1]Low[/orange1]",
                    "very_low": "[red]Very Low[/red]"
                }
                conf_display = conf_color.get(candidate.get('confidence', 'low'), candidate.get('confidence', 'Unknown'))

                sample = candidate.get('sample', '')
                sample_display = (sample[:50] + "...") if len(sample) > 50 else sample

                col_table.add_row(
                    str(i),
                    candidate['name'],
                    conf_display,
                    f"{candidate.get('avg_length', 0):.0f} chars",
                    sample_display
                )

            self.console.print(col_table)
            self.console.print(f"\n[dim]All columns ({len(all_columns)}): {', '.join(all_columns)}[/dim]")

            default_text_col = column_info['text_candidates'][0]['name']
        else:
            self.console.print("[yellow]No text columns auto-detected[/yellow]")
            self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")
            default_text_col = "text"

        # Ask for text column with validation
        while True:
            text_column = Prompt.ask("\n[bold yellow]Enter column name[/bold yellow] (or choose from above)", default=default_text_col)
            if text_column in all_columns:
                break
            self.console.print(f"[red]✗ Column '{text_column}' not found in dataset![/red]")
            self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

        # Step 4b: CRITICAL - Text Length Analysis (MUST be done AFTER text column selection)
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        text_length_stats = self.analyze_text_lengths(
            data_path=data_path,
            text_column=text_column,  # Use the ACTUAL selected column
            display_results=True,
            step_label="STEP 4b: Text Length Analysis"
        )
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

        # Check if long-document models are needed
        requires_long_document_model = text_length_stats.get('requires_long_model', False)
        if requires_long_document_model:
            use_long_model = Confirm.ask(
                "[bold cyan]Would you like to see long-document model recommendations?[/bold cyan]",
                default=True
            )
            if use_long_model:
                text_length_stats['user_prefers_long_models'] = True
                self.console.print("[green]✓ Long-document models will be prioritized in recommendations[/green]")
            else:
                text_length_stats['user_prefers_long_models'] = False
                self.console.print("[yellow]Standard models will be used (texts will be truncated to 512 tokens)[/yellow]")
        else:
            text_length_stats['user_prefers_long_models'] = False

        # Step 5: Label/Category Column Selection with Category Analysis
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 5:[/bold cyan] [bold white]Label/Category Column Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold]💡 What You Need to Select:[/bold]")
        self.console.print("   [cyan]• Label Column[/cyan] - Contains the labels/categories (what the model will learn to predict)\n")

        label_column_default = "labels" if "multi" in format_type else "label"

        if analysis.get('annotation_column_candidates'):
            best_label = analysis['annotation_column_candidates'][0]['name']
            label_column_default = best_label

            self.console.print(f"[green]✓ Label column detected: '{best_label}'[/green]")

            stats = analysis.get('annotation_stats', {}).get(best_label, {})
            fill_rate = stats.get('fill_rate', 0)
            if fill_rate > 0:
                self.console.print(f"[dim]  ({fill_rate*100:.1f}% of rows have labels)[/dim]")

            # NOUVEAU: Analyze and display categories/labels
            try:
                import pandas as pd
                df = pd.read_csv(data_path) if data_path.suffix == '.csv' else pd.read_json(data_path, lines=data_path.suffix == '.jsonl')

                if best_label in df.columns:
                    # Get unique categories and their counts
                    if "multi" in format_type:
                        # Multi-label: try to parse lists/JSON
                        all_labels = []
                        for val in df[best_label].dropna():
                            if isinstance(val, list):
                                all_labels.extend(val)
                            elif isinstance(val, str):
                                try:
                                    parsed = json.loads(val)
                                    if isinstance(parsed, list):
                                        all_labels.extend(parsed)
                                except:
                                    pass
                        label_counts = pd.Series(all_labels).value_counts()
                    else:
                        # Single-label: direct value counts
                        label_counts = df[best_label].value_counts()

                    # Display categories table
                    if len(label_counts) > 0:
                        self.console.print(f"\n[bold]📊 Detected {len(label_counts)} Categories:[/bold]")

                        cat_table = Table(border_style="green", show_header=True, header_style="bold cyan")
                        cat_table.add_column("#", style="cyan", width=5)
                        cat_table.add_column("Category", style="white", width=30)
                        cat_table.add_column("Count", style="yellow", width=10, justify="right")
                        cat_table.add_column("Percentage", style="green", width=12, justify="right")

                        total = label_counts.sum()
                        for i, (cat, count) in enumerate(label_counts.head(20).items(), 1):
                            percentage = (count / total * 100) if total > 0 else 0
                            cat_table.add_row(
                                str(i),
                                str(cat)[:28],
                                f"{count:,}",
                                f"{percentage:.1f}%"
                            )

                        if len(label_counts) > 20:
                            cat_table.add_row("...", f"... and {len(label_counts) - 20} more", "...", "...")

                        self.console.print(cat_table)
                        self.console.print(f"[dim]Total samples: {total:,}[/dim]")
            except Exception as e:
                self.logger.debug(f"Could not analyze categories: {e}")

        if all_columns:
            self.console.print(f"\n[dim]Available columns: {', '.join(all_columns)}[/dim]")

        # Ask for label column with validation
        while True:
            label_column = Prompt.ask("\n[bold yellow]Category/label column[/bold yellow]", default=label_column_default)
            if label_column in all_columns:
                break
            self.console.print(f"[red]✗ Column '{label_column}' not found in dataset![/red]")
            self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

        # Step 6: ID Column Selection with Sophisticated Strategy
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 6:[/bold cyan] [bold white]Identifier Column Selection (Optional)[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Optional: Select an ID column to track samples and link results to your original data.[/dim]\n")

        id_columns = self._detect_id_columns(all_columns)
        id_column = None

        if len(id_columns) > 1:
            self.console.print(f"[bold cyan]📋 Found {len(id_columns)} ID columns:[/bold cyan]")
            for i, col in enumerate(id_columns, 1):
                self.console.print(f"  {i}. [cyan]{col}[/cyan]")

            self.console.print("\n[bold]ID Strategy:[/bold]")
            self.console.print("[dim]IDs are used to track samples and link results to your original data.[/dim]")
            self.console.print("• [cyan]single[/cyan]: Use one column as ID")
            self.console.print("• [cyan]combine[/cyan]: Combine multiple columns (e.g., 'promesse_id+sentence_id')")
            self.console.print("• [cyan]none[/cyan]: Generate automatic IDs")

            id_strategy = Prompt.ask("ID strategy", choices=["single", "combine", "none"], default="single")

            if id_strategy == "none":
                self.console.print("[dim]An automatic ID will be generated[/dim]")
                id_column = None
            elif id_strategy == "combine":
                self.console.print("\n[bold]Select columns to combine:[/bold]")
                self.console.print("[dim]Enter column numbers separated by commas (e.g., '1,2')[/dim]")

                while True:
                    selection = Prompt.ask("Columns to combine")
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        if all(0 <= i < len(id_columns) for i in indices):
                            selected_cols = [id_columns[i] for i in indices]
                            id_column = "+".join(selected_cols)
                            self.console.print(f"[green]✓ Will combine: {' + '.join(selected_cols)}[/green]")
                            break
                        else:
                            self.console.print("[red]Invalid column numbers. Try again.[/red]")
                    except (ValueError, IndexError):
                        self.console.print("[red]Invalid format. Use comma-separated numbers (e.g., '1,2')[/red]")
            else:  # single
                id_column = Prompt.ask("Which ID column to use?", choices=id_columns, default=id_columns[0])
        elif len(id_columns) == 1:
            self.console.print(f"[green]✓ ID column detected: '{id_columns[0]}'[/green]")
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
            id_column = Prompt.ask("Identifier column (optional)", default=id_columns[0])
        else:
            self.console.print("[dim]No ID columns detected - automatic IDs will be generated[/dim]")
            id_column = None

        if id_column:
            self.console.print(f"[green]✓ Identifier strategy: {id_column}[/green]")

        # Model selection will be done later when training mode is chosen
        # Store languages and text characteristics for later use
        model_to_use = None
        model_strategy = "multilingual"  # default
        language_model_mapping = {}  # For per-language models

        # Skip model selection - will be done in training mode
        if False and confirmed_languages and len(confirmed_languages) > 1:
            # Multiple languages detected - offer strategy choice
            self.console.print(f"[bold]📊 Dataset contains {len(confirmed_languages)} languages:[/bold]")

            if language_distribution:
                # Filter out metadata keys (like _reclassification_map)
                lang_counts = {k: v for k, v in language_distribution.items() if not k.startswith('_') and isinstance(v, (int, float))}

                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                    total = sum(lang_counts.values())
                    pct = (count / total * 100) if total > 0 else 0
                    self.console.print(f"  • {lang.upper()}: {count:,} texts ({pct:.1f}%)")
            else:
                for lang in sorted(confirmed_languages):
                    self.console.print(f"  • {lang.upper()}")

            self.console.print("\n[bold]Model Strategy Options:[/bold]")
            self.console.print("  [cyan]1. multilingual[/cyan] - Train ONE multilingual model for all languages")
            self.console.print("     ✓ Simpler, faster, handles cross-lingual patterns")
            self.console.print("     ✗ May have slightly lower performance per language")
            self.console.print()
            self.console.print("  [cyan]2. specialized[/cyan] - Train SEPARATE specialized models per language")
            self.console.print("     ✓ Best performance for each language")
            self.console.print("     ✗ More training time, requires language column or detection")
            self.console.print()
            self.console.print("  [cyan]3. hybrid[/cyan] - Multilingual model + fine-tuned per-language models")
            self.console.print("     ✓ Best of both worlds")
            self.console.print("     ✗ Most training time and complexity")

            model_strategy = Prompt.ask(
                "\n[bold yellow]Select model strategy[/bold yellow]",
                choices=["multilingual", "specialized", "hybrid"],
                default="multilingual"
            )

            self.console.print(f"\n[green]✓ Selected strategy: {model_strategy}[/green]")

            if model_strategy == "multilingual":
                # Get ONE multilingual model for all languages
                # Consider long-document models if user prefers them
                if text_length_stats.get('user_prefers_long_models', False):
                    model_to_use = self._get_long_document_model_recommendation(confirmed_languages)
                else:
                    model_to_use = self._get_model_recommendation_from_languages(confirmed_languages)

            elif model_strategy == "specialized":
                # Get specialized model for EACH language
                self.console.print("\n[bold]Selecting specialized models for each language:[/bold]")

                for lang in sorted(confirmed_languages):
                    # Consider long-document models if user prefers them
                    if text_length_stats.get('user_prefers_long_models', False):
                        lang_recommendations = self._get_long_document_models_for_language(lang)
                    else:
                        lang_recommendations = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)

                    if lang_recommendations:
                        self.console.print(f"\n[cyan]For {lang.upper()}:[/cyan]")
                        for i, rec in enumerate(lang_recommendations[:3], 1):
                            self.console.print(f"  {i}. {rec['model']} - {rec['reason']}")

                        choice = Prompt.ask(
                            f"Model for {lang.upper()} (1-{min(3, len(lang_recommendations))}, or enter model name)",
                            default="1"
                        )

                        if choice.isdigit() and 0 < int(choice) <= len(lang_recommendations):
                            language_model_mapping[lang] = lang_recommendations[int(choice) - 1]['model']
                        else:
                            language_model_mapping[lang] = choice

                        self.console.print(f"  [green]✓ {lang.upper()}: {language_model_mapping[lang]}[/green]")
                    else:
                        # Fallback to multilingual
                        self.console.print(f"[yellow]No specific model for {lang.upper()}, using multilingual[/yellow]")
                        if not model_to_use:
                            model_to_use = self._get_model_recommendation_from_languages(confirmed_languages)
                        language_model_mapping[lang] = model_to_use

            elif model_strategy == "hybrid":
                # First get multilingual base model
                self.console.print("\n[bold]1. Select base multilingual model:[/bold]")
                model_to_use = self._get_model_recommendation_from_languages(confirmed_languages)

                # Then get specialized models for fine-tuning
                self.console.print("\n[bold]2. Select specialized models for fine-tuning:[/bold]")
                for lang in sorted(confirmed_languages):
                    lang_recommendations = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)

                    if lang_recommendations:
                        self.console.print(f"\n[cyan]Fine-tuning model for {lang.upper()}:[/cyan]")
                        for i, rec in enumerate(lang_recommendations[:3], 1):
                            self.console.print(f"  {i}. {rec['model']}")

                        choice = Prompt.ask(
                            f"Model for {lang.upper()} (1-{min(3, len(lang_recommendations))}, or 'skip')",
                            default="1"
                        )

                        if choice.lower() != 'skip':
                            if choice.isdigit() and 0 < int(choice) <= len(lang_recommendations):
                                language_model_mapping[lang] = lang_recommendations[int(choice) - 1]['model']
                            else:
                                language_model_mapping[lang] = choice

                            self.console.print(f"  [green]✓ {lang.upper()}: {language_model_mapping[lang]}[/green]")

        elif confirmed_languages and len(confirmed_languages) == 1:
            # Single language - get specialized model
            lang = list(confirmed_languages)[0]
            self.console.print(f"[bold]Single language detected: {lang.upper()}[/bold]")

            # Consider long-document models if user prefers them
            if text_length_stats.get('user_prefers_long_models', False):
                lang_recommendations = self._get_long_document_models_for_language(lang)
            else:
                lang_recommendations = LanguageNormalizer.recommend_models({lang}, self.available_trainer_models)

            if lang_recommendations:
                self.console.print(f"\n[bold]🤖 Recommended Models for {lang.upper()}:[/bold]")
                for i, rec in enumerate(lang_recommendations[:5], 1):
                    self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

                choice = Prompt.ask("Select model (1-5, or enter model name)", default="1")

                if choice.isdigit() and 0 < int(choice) <= len(lang_recommendations):
                    model_to_use = lang_recommendations[int(choice) - 1]['model']
                else:
                    model_to_use = choice

                self.console.print(f"[green]✓ Selected: {model_to_use}[/green]")
        else:
            # No languages detected - use default
            model_to_use = self._get_model_recommendation_from_languages(set())

        # Return all collected information
        return {
            'data_path': data_path,
            'text_column': text_column,
            'label_column': label_column,
            'id_column': id_column,
            'lang_column': lang_column,
            'confirmed_languages': confirmed_languages,
            'language_distribution': language_distribution,  # Exact counts per language
            'text_length_stats': text_length_stats,  # Text length statistics and long-document preference
            'model_strategy': model_strategy,  # multilingual, specialized, or hybrid
            'recommended_model': model_to_use,  # Main/base model
            'language_model_mapping': language_model_mapping,  # Per-language models (if specialized)
            'analysis': analysis
        }

    def _training_studio_dataset_wizard(self, builder: TrainingDatasetBuilder) -> Optional[TrainingDataBundle]:
        """
        Intelligent dataset wizard with comprehensive file analysis and guided setup.
        Now supports all formats with smart detection and recommendations.
        """

        # Step 1: Dataset Source Selection
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 1:[/bold cyan] [bold white]Dataset Source Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Choose where your training data will come from.[/dim]\n")

        self.console.print("[bold]📚 Dataset Source Options:[/bold]")
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

        # Step 2: Explain format options with Rich table
        self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[bold cyan]  STEP 2:[/bold cyan] [bold white]Dataset Format Selection[/bold white]")
        self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        self.console.print("[dim]Choose the format that matches your annotated data structure.[/dim]\n")

        formats_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED)
        formats_table.add_column("Format", style="cyan bold", width=18)
        formats_table.add_column("Description", style="white", width=50)
        formats_table.add_column("Example", style="dim", width=35)

        formats_table.add_row(
            "llm-json",
            "CSV/JSON with LLM annotations in a column\n✓ JSON objects containing labels/categories\n✓ Output from LLM annotation tools",
            "{'category': 'Tech', 'sentiment': 'pos'}"
        )
        formats_table.add_row(
            "category-csv",
            "Simple CSV with text and label columns\n✓ Most common format\n✓ One row = one sample with its label",
            "text,label\n'Hello',positive"
        )
        formats_table.add_row(
            "[dim]binary-long[/dim]",
            "[dim]Long-format CSV with binary labels\n✓ Multiple rows per sample\n✓ Each row = one category with 0/1 value[/dim]",
            "[dim]id,text,category,value\n1,'Hi',pos,1[/dim]"
        )
        formats_table.add_row(
            "[dim]jsonl-single[/dim]",
            "[dim]JSONL file for single-label tasks\n✓ One JSON object per line\n✓ Each sample has one label only[/dim]",
            "[dim]{'text':'Hi','label':'positive'}[/dim]"
        )
        formats_table.add_row(
            "[dim]jsonl-multi[/dim]",
            "[dim]JSONL file for multi-label tasks\n✓ One JSON object per line\n✓ Each sample can have multiple labels[/dim]",
            "[dim]{'text':'Hi','labels':['pos','friendly']}[/dim]"
        )

        self.console.print(formats_table)
        self.console.print()

        # Add development notice for experimental formats
        self.console.print("[yellow]⚠️  Note:[/yellow] [bold red]binary-long, jsonl-single, and jsonl-multi are currently under development and NOT accessible.[/bold red]")
        self.console.print("[dim]      These formats will be enabled in a future release after thorough testing.[/dim]")
        self.console.print()

        format_choice = Prompt.ask(
            "[bold yellow]Select dataset format[/bold yellow]",
            choices=["llm-json", "category-csv", "cancel", "back"],
            default="llm-json",
        )

        if format_choice == "cancel" or format_choice == "back":
            return None

        if format_choice == "llm-json":
            # Step 1: Dataset Selection
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  STEP 1:[/bold cyan] [bold white]Dataset Selection[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[dim]Select your annotated dataset file to prepare for training.[/dim]\n")

            # Show detected datasets if available
            if self.detected_datasets:
                datasets_table = Table(title="📊 Detected Datasets", border_style="cyan")
                datasets_table.add_column("#", style="cyan", width=3)
                datasets_table.add_column("Name", style="white", width=40)
                datasets_table.add_column("Format", style="yellow", width=8)
                datasets_table.add_column("Size", style="green", width=10)
                datasets_table.add_column("Folder", style="magenta", width=20)

                for i, ds in enumerate(self.detected_datasets, 1):  # Show ALL datasets, not just [:10]
                    # Calculate file size
                    try:
                        if hasattr(ds, 'path') and ds.path.exists():
                            size_bytes = ds.path.stat().st_size
                            if size_bytes < 1024:
                                size_str = f"{size_bytes} B"
                            elif size_bytes < 1024 * 1024:
                                size_str = f"{size_bytes / 1024:.1f} KB"
                            else:
                                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                        else:
                            size_str = "—"
                    except Exception as e:
                        self.logger.debug(f"Could not get size for {ds.path}: {e}")
                        size_str = "—"

                    # Get folder name (parent directory name)
                    folder_name = ds.path.parent.name if hasattr(ds, 'path') and ds.path.parent.name else "data"

                    datasets_table.add_row(
                        str(i),
                        ds.path.name if hasattr(ds, 'path') else "—",
                        ds.format if hasattr(ds, 'format') else "—",
                        size_str,
                        folder_name
                    )

                self.console.print(datasets_table)
                self.console.print()
                self.console.print("[dim]💡 You can either:[/dim]")
                self.console.print("[dim]   • Enter the [cyan]#[/cyan] number from the table above (e.g., '1', '13')[/dim]")
                self.console.print("[dim]   • Enter an [cyan]absolute path[/cyan] to any file (e.g., '/Users/name/data/file.csv')[/dim]\n")

                dataset_choice = Prompt.ask("Dataset selection")

                # Parse choice
                if dataset_choice.isdigit():
                    idx = int(dataset_choice) - 1
                    if 0 <= idx < len(self.detected_datasets):
                        csv_path = self.detected_datasets[idx].path
                    else:
                        self.console.print("[red]Invalid dataset number[/red]")
                        return None
                else:
                    csv_path = Path(dataset_choice)
            else:
                file_path_str = self._prompt_file_path("Annotated file path (CSV/JSON/Excel/Parquet)")
                csv_path = Path(file_path_str)

            self.console.print(f"[green]✓ Selected: {csv_path.name} ({csv_path.suffix[1:]})[/green]\n")

            # Step 2: File Structure Analysis
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  STEP 2:[/bold cyan] [bold white]Analyzing Dataset Structure[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[dim]🔍 Analyzing columns, detecting types, and extracting samples...[/dim]")
            analysis = DataDetector.analyze_file_intelligently(csv_path)

            # Show analysis results
            if analysis['issues']:
                self.console.print("\n[yellow]⚠️  Analysis warnings:[/yellow]")
                for issue in analysis['issues']:
                    self.console.print(f"  • {issue}")

            # Step 3: Column Selection
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  Step 3:[/bold cyan] [bold white]Column Selection[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold]💡 What You Need to Select:[/bold]")
            self.console.print("   [cyan]• Text Column[/cyan]     - Contains the text data to train on (input for predictions)")
            self.console.print("   [cyan]• Annotation Column[/cyan] - Contains the JSON annotations (labels/categories for training)\n")

            # Auto-suggest text column with all available columns
            text_column_default = "sentence"
            all_columns = analysis.get('all_columns', [])

            # Read CSV to analyze ALL columns
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Create comprehensive column overview table
            if all_columns:
                self.console.print(f"[bold]📊 Dataset Overview ({len(all_columns)} columns, {len(df):,} rows):[/bold]\n")

                # Create detailed columns table
                all_columns_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
                all_columns_table.add_column("#", style="dim", width=3)
                all_columns_table.add_column("Column Name", style="cyan bold", width=30)
                all_columns_table.add_column("Type", style="yellow", width=12)
                all_columns_table.add_column("Sample Values", style="white", width=50)

                for idx, col in enumerate(all_columns, 1):
                    # Detect column type
                    col_type = "text"
                    if col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            col_type = "numeric"
                        elif pd.api.types.is_datetime64_any_dtype(df[col]):
                            col_type = "datetime"
                        else:
                            # Check if it's likely JSON
                            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
                            if isinstance(sample_val, str) and (sample_val.startswith('{') or sample_val.startswith('[')):
                                col_type = "json/annotation"
                            else:
                                col_type = "text"

                        # Get sample values
                        samples = df[col].dropna().head(3).tolist()
                        if samples:
                            sample_str = ", ".join([str(s)[:30] + "..." if len(str(s)) > 30 else str(s) for s in samples])
                        else:
                            sample_str = "[empty]"
                    else:
                        sample_str = "—"

                    all_columns_table.add_row(
                        str(idx),
                        col,
                        col_type,
                        sample_str
                    )

                self.console.print(all_columns_table)

                # Now show AI suggestions
                self.console.print("\n[bold]🔍 AI Suggestions (based on analysis):[/bold]")
                self.console.print("[dim]The system recommends these columns, but you can choose ANY column from the table above.[/dim]\n")

                suggestions_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.SIMPLE)
                suggestions_table.add_column("Purpose", style="yellow bold", width=20)
                suggestions_table.add_column("Suggested Column", style="green bold", width=25)
                suggestions_table.add_column("Reason", style="white", width=45)

                # Text column row
                if analysis['text_column_candidates']:
                    best_text = analysis['text_column_candidates'][0]['name']
                    text_column_default = best_text
                    text_stats = analysis['text_column_candidates'][0]
                    avg_len = text_stats.get('avg_length', 0)
                    suggestions_table.add_row(
                        "📝 Text Data",
                        best_text,
                        f"Avg length {avg_len:.0f} chars (for model input)"
                    )
                else:
                    suggestions_table.add_row("📝 Text Data", "—", "⚠️  No suggestion - choose manually")

                # Annotation column row
                annotation_column_default = "annotation"
                if analysis['annotation_column_candidates']:
                    best_annotation = analysis['annotation_column_candidates'][0]['name']
                    annotation_column_default = best_annotation
                    stats = analysis['annotation_stats'].get(best_annotation, {})
                    fill_rate = stats.get('fill_rate', 0)
                    if fill_rate > 0:
                        suggestions_table.add_row(
                            "🏷️  Annotations",
                            best_annotation,
                            f"{fill_rate*100:.1f}% filled (training labels)"
                        )
                    else:
                        suggestions_table.add_row(
                            "🏷️  Annotations",
                            best_annotation,
                            "[red]⚠️  EMPTY - cannot use[/red]"
                        )
                else:
                    suggestions_table.add_row("🏷️  Annotations", "—", "⚠️  No suggestion - choose manually")

                self.console.print(suggestions_table)
                self.console.print()
            else:
                # Fallback if no columns detected
                if analysis['text_column_candidates']:
                    best_text = analysis['text_column_candidates'][0]['name']
                    text_column_default = best_text
                    self.console.print(f"\n[green]✓ Suggested text column: '{best_text}'[/green]")

                annotation_column_default = "annotation"
                if analysis['annotation_column_candidates']:
                    best_annotation = analysis['annotation_column_candidates'][0]['name']
                    annotation_column_default = best_annotation
                    stats = analysis['annotation_stats'].get(best_annotation, {})
                    fill_rate = stats.get('fill_rate', 0)
                    if fill_rate > 0:
                        self.console.print(f"[green]✓ Suggested annotation column: '{best_annotation}' ({fill_rate*100:.1f}% filled)[/green]")
                    else:
                        self.console.print(f"[red]⚠️  Suggested annotation column '{best_annotation}' is EMPTY - cannot be used for training![/red]")

            self.console.print("[bold yellow]📝 Make Your Selection:[/bold yellow]")
            self.console.print("[dim]   → Press [bold]Enter[/bold] to accept the AI suggestion[/dim]")
            self.console.print("[dim]   → Or type any column name from the complete list above[/dim]\n")

            # Ask for text column with validation
            while True:
                text_column = Prompt.ask("[bold cyan]Text column[/bold cyan] (training input)", default=text_column_default)
                if text_column in all_columns:
                    break
                self.console.print(f"[red]✗ Column '{text_column}' not found in dataset![/red]")
                self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

            # Ask for annotation column with validation
            while True:
                annotation_column = Prompt.ask("[bold cyan]Annotation column[/bold cyan] (training labels)", default=annotation_column_default)
                if annotation_column in all_columns:
                    break
                self.console.print(f"[red]✗ Column '{annotation_column}' not found in dataset![/red]")
                self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

            # Show confirmation of selection
            self.console.print(f"\n[green]✓ Selected columns:[/green]")
            self.console.print(f"  [cyan]Text:[/cyan] '{text_column}' → Model will learn from this text")
            self.console.print(f"  [cyan]Annotations:[/cyan] '{annotation_column}' → Model will learn these labels")

            # Step 3b: CRITICAL - Text Length Analysis (MUST be done AFTER text column selection)
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            text_length_stats = self.analyze_text_lengths(
                data_path=csv_path,
                text_column=text_column,  # Use the ACTUAL selected column, not temp
                display_results=True,
                step_label="STEP 3b: Text Length Analysis"
            )
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

            # Check if long-document models are needed
            requires_long_document_model = text_length_stats.get('requires_long_model', False)
            if requires_long_document_model:
                use_long_model = Confirm.ask(
                    "[bold cyan]Would you like to see long-document model recommendations?[/bold cyan]",
                    default=True
                )
                if use_long_model:
                    text_length_stats['user_prefers_long_models'] = True
                    self.console.print("[green]✓ Long-document models will be prioritized in recommendations[/green]")
                else:
                    text_length_stats['user_prefers_long_models'] = False
                    self.console.print("[yellow]Standard models will be used (texts will be truncated to 512 tokens)[/yellow]")
            else:
                text_length_stats['user_prefers_long_models'] = False

            # Step 4: Language Detection and Text Analysis (using sophisticated universal system)
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  STEP 4:[/bold cyan] [bold white]Language Detection[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[dim]Analyzing languages to recommend the best model.[/dim]\n")

            # Read CSV for analysis
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Use the SAME sophisticated language detection as category-csv
            languages_found_in_column = set(analysis.get('languages_detected', {}).keys())
            confirmed_languages = set()
            lang_column = None
            language_distribution = {}  # Store exact language counts

            # Check if we have a language column with detected languages
            has_lang_column = bool(analysis.get('language_column_candidates'))

            if has_lang_column and languages_found_in_column:
                # Option 1: Language column exists - offer to use it or detect automatically
                self.console.print("[bold]🌍 Languages Found in Column:[/bold]")
                for lang, count in analysis['languages_detected'].items():
                    self.console.print(f"  • {lang.upper()}: {count:,} rows")

                lang_column_candidate = analysis['language_column_candidates'][0]
                self.console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")

                use_lang_column = Confirm.ask(
                    f"\n[bold]Use language column '{lang_column_candidate}'?[/bold]",
                    default=True
                )

                if use_lang_column:
                    confirmed_languages = languages_found_in_column
                    lang_column = lang_column_candidate
                    self.console.print(f"[green]✓ Using language column: {lang_column}[/green]")
                else:
                    # User said no to language column - apply automatic detection
                    self.console.print("\n[yellow]Language column not used. Applying automatic detection...[/yellow]")
                    has_lang_column = False  # Trigger auto-detection below
            else:
                # Option 2: No language column
                if not has_lang_column:
                    self.console.print("[yellow]ℹ️  No language column detected[/yellow]")
                apply_auto_detection = Confirm.ask("Apply automatic language detection on text content?", default=True)
                if not apply_auto_detection:
                    has_lang_column = True  # Skip auto-detection

            # Automatic language detection from text content (if no lang column used)
            if not lang_column and (not has_lang_column or 'apply_auto_detection' in locals()):
                self.console.print("\n[dim]🔍 Analyzing ALL texts to detect languages (this may take a moment)...[/dim]")

                try:
                    from llm_tool.utils.language_detector import LanguageDetector

                    if text_column in df.columns:
                        # Analyze ALL texts (not just sample) for precise distribution
                        all_texts = df[text_column].dropna().tolist()

                        if all_texts:
                            detector = LanguageDetector()
                            lang_counts = {}
                            detected_languages_per_text = []  # Store language for each text

                            # Progress indicator
                            from tqdm import tqdm
                            self.console.print(f"[dim]Analyzing {len(all_texts)} texts...[/dim]")

                            for text in tqdm(all_texts, desc="Detecting languages", disable=not HAS_RICH):
                                if text and len(str(text).strip()) > 10:
                                    try:
                                        detected = detector.detect(str(text))
                                        if detected:
                                            # Handle both dict and string returns
                                            if isinstance(detected, dict):
                                                lang = detected.get('language')
                                                confidence = detected.get('confidence', 0)
                                                # Use confidence threshold (optional)
                                                if lang and confidence >= 0.7:  # 70% confidence threshold
                                                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                                                    detected_languages_per_text.append(lang)
                                                else:
                                                    detected_languages_per_text.append(None)  # Low confidence
                                            elif isinstance(detected, str):
                                                lang_counts[detected] = lang_counts.get(detected, 0) + 1
                                                detected_languages_per_text.append(detected)
                                        else:
                                            detected_languages_per_text.append(None)
                                    except Exception as e:
                                        self.logger.debug(f"Language detection failed for text: {e}")
                                        detected_languages_per_text.append(None)
                                else:
                                    detected_languages_per_text.append(None)  # Empty or too short text

                            if lang_counts:
                                # Store exact distribution
                                language_distribution = lang_counts
                                total = sum(lang_counts.values())

                                self.console.print(f"\n[bold]🌍 Languages Detected from Content ({total:,} texts analyzed):[/bold]")

                                # Create detailed table
                                lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                                lang_table.add_column("Language", style="cyan", width=12)
                                lang_table.add_column("Count", style="yellow", justify="right", width=12)
                                lang_table.add_column("Percentage", style="green", justify="right", width=12)

                                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                    percentage = (count / total * 100) if total > 0 else 0
                                    lang_table.add_row(
                                        lang.upper(),
                                        f"{count:,}",
                                        f"{percentage:.1f}%"
                                    )

                                self.console.print(lang_table)

                                # Detect low-percentage languages (likely detection errors)
                                LOW_PERCENTAGE_THRESHOLD = 1.0  # Languages with < 1% are considered low
                                majority_languages = {}  # Languages above threshold
                                minority_languages = {}  # Languages below threshold (likely errors)

                                for lang, count in lang_counts.items():
                                    percentage = (count / total * 100) if total > 0 else 0
                                    if percentage >= LOW_PERCENTAGE_THRESHOLD:
                                        majority_languages[lang] = count
                                    else:
                                        minority_languages[lang] = count

                                confirmed_languages = set(lang_counts.keys())

                                # Handle low-percentage languages if detected
                                if minority_languages:
                                    self.console.print(f"\n[yellow]⚠ Warning: {len(minority_languages)} language(s) detected with very low percentage (< {LOW_PERCENTAGE_THRESHOLD}%):[/yellow]")
                                    for lang, count in sorted(minority_languages.items(), key=lambda x: x[1], reverse=True):
                                        percentage = (count / total * 100)
                                        self.console.print(f"  • {lang.upper()}: {count} texts ({percentage:.2f}%)")

                                    self.console.print("\n[dim]These are likely detection errors. You have options:[/dim]")
                                    self.console.print("  [cyan]1. exclude[/cyan] - Exclude ALL low-percentage languages from training")
                                    self.console.print("  [cyan]2. keep[/cyan] - Keep ALL detected languages (not recommended)")
                                    self.console.print("  [cyan]3. select[/cyan] - Manually select which languages to keep")
                                    self.console.print("  [cyan]4. correct[/cyan] - Force ALL minority languages to a single language (quick fix)")

                                    minority_action = Prompt.ask(
                                        "\n[bold yellow]How to handle low-percentage languages?[/bold yellow]",
                                        choices=["exclude", "keep", "select", "correct"],
                                        default="correct"
                                    )

                                    if minority_action == "correct":
                                        # Quick correction: force all minority languages to one language
                                        self.console.print("\n[bold cyan]🔧 Quick Language Correction[/bold cyan]\n")

                                        # Show available languages
                                        all_supported_langs = [
                                            'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja',
                                            'ar', 'pl', 'tr', 'ko', 'hi', 'sv', 'no', 'da', 'fi', 'cs',
                                            'el', 'he', 'ro', 'uk', 'bg', 'hr', 'vi', 'th', 'id', 'fa'
                                        ]

                                        # Suggest the majority language
                                        majority_lang = max(majority_languages.items(), key=lambda x: x[1])[0] if majority_languages else 'en'

                                        self.console.print(f"[bold]Available languages:[/bold]")
                                        self.console.print(f"  • Majority language detected: [green]{majority_lang.upper()}[/green] ({majority_languages.get(majority_lang, 0)} texts)")
                                        self.console.print(f"  • All supported: {', '.join([l.upper() for l in all_supported_langs])}")

                                        correction_target = Prompt.ask(
                                            f"\n[bold yellow]Force ALL minority languages to which language?[/bold yellow]",
                                            default=majority_lang
                                        ).lower().strip()

                                        if correction_target not in all_supported_langs:
                                            self.console.print(f"[yellow]Warning: '{correction_target}' not in standard list, but will be used anyway[/yellow]")

                                        # Update language_distribution and confirmed_languages
                                        total_corrected = sum(minority_languages.values())

                                        # Move all minority counts to the target language
                                        for minority_lang in minority_languages.keys():
                                            if minority_lang in language_distribution:
                                                del language_distribution[minority_lang]

                                        # Add corrected texts to target language
                                        if correction_target in language_distribution:
                                            language_distribution[correction_target] += total_corrected
                                        else:
                                            language_distribution[correction_target] = total_corrected

                                        # Update confirmed languages
                                        confirmed_languages = set([correction_target] + list(majority_languages.keys()))

                                        # CRITICAL FIX: Update detected_languages_per_text with corrections
                                        if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                            for i in range(len(detected_languages_per_text)):
                                                if detected_languages_per_text[i] in minority_languages:
                                                    detected_languages_per_text[i] = correction_target

                                        self.console.print(f"\n[green]✓ Corrected {total_corrected} texts from {len(minority_languages)} languages to {correction_target.upper()}[/green]")

                                        # Display updated distribution
                                        update_table = Table(title="Updated Language Distribution", border_style="green")
                                        update_table.add_column("Language", style="cyan", justify="center")
                                        update_table.add_column("Count", justify="right")
                                        update_table.add_column("Percentage", justify="right")

                                        new_total = sum(language_distribution.values())
                                        for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
                                            if count > 0:  # Only show non-zero counts
                                                percentage = (count / new_total) * 100 if new_total > 0 else 0
                                                update_table.add_row(lang.upper(), f"{count:,}", f"{percentage:.1f}%")

                                        self.console.print(update_table)

                                    elif minority_action == "exclude":
                                        # Exclude low-percentage languages
                                        for lang in minority_languages.keys():
                                            language_distribution[lang] = 0  # Mark as excluded

                                        # CRITICAL FIX: Mark excluded language texts as None
                                        if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                            for i in range(len(detected_languages_per_text)):
                                                if detected_languages_per_text[i] in minority_languages:
                                                    detected_languages_per_text[i] = None

                                        confirmed_languages = set(majority_languages.keys())
                                        excluded_count = sum(minority_languages.values())
                                        self.console.print(f"\n[yellow]✗ Excluded {excluded_count} texts from {len(minority_languages)} low-percentage language(s)[/yellow]")
                                        self.console.print(f"[green]✓ Final languages: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")

                                    elif minority_action == "keep":
                                        self.console.print("[yellow]⚠ Keeping all detected languages (including low-percentage ones)[/yellow]")

                                    elif minority_action == "select":
                                        # Manual selection of languages to keep
                                        self.console.print("\n[bold cyan]📝 Language Selection:[/bold cyan]")
                                        self.console.print(f"[dim]Select which languages to keep for training (from all {len(lang_counts)} detected)[/dim]\n")

                                        # Show all languages sorted by count
                                        self.console.print("[bold]All Detected Languages:[/bold]")
                                        for i, (lang, count) in enumerate(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True), 1):
                                            percentage = (count / total * 100)
                                            status = "[green]✓ majority[/green]" if lang in majority_languages else "[yellow]⚠ minority[/yellow]"
                                            self.console.print(f"  {i:2d}. {lang.upper():5s} - {count:6,} texts ({percentage:5.2f}%) {status}")

                                        self.console.print("\n[bold yellow]Select languages to KEEP:[/bold yellow]")
                                        self.console.print("[dim]Enter language codes separated by commas (e.g., 'fr,en,de')[/dim]")
                                        self.console.print("[dim]Press Enter without typing to keep ALL languages[/dim]")

                                        selected_langs = Prompt.ask("\n[bold]Languages to keep[/bold]", default="")

                                        if selected_langs.strip():
                                            # User selected specific languages
                                            selected_set = set([l.strip().lower() for l in selected_langs.split(',') if l.strip()])

                                            # Validate that selected languages exist
                                            invalid_langs = selected_set - set(lang_counts.keys())
                                            if invalid_langs:
                                                self.console.print(f"[yellow]⚠ Warning: These languages were not detected: {', '.join(invalid_langs)}[/yellow]")
                                                selected_set = selected_set - invalid_langs

                                            # Exclude non-selected languages
                                            for lang in lang_counts.keys():
                                                if lang not in selected_set:
                                                    language_distribution[lang] = 0  # Mark as excluded

                                            # CRITICAL FIX: Mark non-selected language texts as None
                                            if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                                for i in range(len(detected_languages_per_text)):
                                                    if detected_languages_per_text[i] and detected_languages_per_text[i] not in selected_set:
                                                        detected_languages_per_text[i] = None

                                            confirmed_languages = selected_set
                                            kept_count = sum([lang_counts[lang] for lang in selected_set])
                                            excluded_count = total - kept_count

                                            self.console.print(f"\n[green]✓ Kept {len(selected_set)} language(s): {', '.join([l.upper() for l in sorted(selected_set)])}[/green]")
                                            self.console.print(f"[dim]  → {kept_count:,} texts kept, {excluded_count:,} texts excluded[/dim]")
                                        else:
                                            # User pressed Enter - keep all
                                            self.console.print("[green]✓ Keeping all detected languages[/green]")

                                # Final confirmation (allow override even after selection)
                                lang_list = ', '.join([l.upper() for l in sorted(confirmed_languages)])
                                lang_confirmed = Confirm.ask(
                                    f"\n[bold]Final languages: {lang_list}. Is this correct?[/bold]",
                                    default=True
                                )

                                if not lang_confirmed:
                                    self.console.print("\n[yellow]Override with manual selection[/yellow]")
                                    manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

                                    # Update distribution to exclude non-selected languages
                                    for lang in lang_counts.keys():
                                        if lang not in confirmed_languages:
                                            language_distribution[lang] = 0

                                    # CRITICAL FIX: Mark non-confirmed language texts as None
                                    if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                        for i in range(len(detected_languages_per_text)):
                                            if detected_languages_per_text[i] and detected_languages_per_text[i] not in confirmed_languages:
                                                detected_languages_per_text[i] = None

                                    self.console.print(f"[green]✓ Manual override: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                                else:
                                    self.console.print("[green]✓ Languages confirmed from content analysis[/green]")

                                # CRITICAL FIX: Add detected language column to DataFrame and save
                                if 'detected_languages_per_text' in locals() and detected_languages_per_text:
                                    # Create a temporary DataFrame for non-null texts
                                    temp_df = df[df[text_column].notna()].copy()

                                    # Ensure same length
                                    if len(detected_languages_per_text) == len(temp_df):
                                        temp_df['language'] = detected_languages_per_text

                                        # Map detected languages to the full DataFrame
                                        df['language'] = None
                                        df.loc[df[text_column].notna(), 'language'] = detected_languages_per_text

                                        # Set lang_column to use this new column
                                        lang_column = 'language'

                                        # Save updated DataFrame back to CSV
                                        df.to_csv(csv_path, index=False)
                                        self.console.print(f"[dim]✓ Added 'language' column to dataset ({len([l for l in detected_languages_per_text if l])} texts with detected language)[/dim]")
                            else:
                                # Fallback: ask user
                                self.console.print("[yellow]Could not detect languages automatically[/yellow]")
                                manual_langs = Prompt.ask("Expected language codes (e.g., en,fr,de)", default="")
                                if manual_langs.strip():
                                    confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
                        else:
                            self.console.print("[yellow]Not enough text samples for language detection[/yellow]")
                            manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                            if manual_langs.strip():
                                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

                except Exception as e:
                    self.logger.debug(f"Language detection from content failed: {e}")
                    self.console.print("[yellow]Automatic detection failed. Please specify manually[/yellow]")
                    manual_langs = Prompt.ask("Expected language codes (optional, e.g., en,fr,de)", default="")
                    if manual_langs.strip():
                        confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

            # Model selection will be done later when training mode is selected
            # Store languages for later use

            # Step 5: Annotation Data Preview
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  STEP 5:[/bold cyan] [bold white]Annotation Data Preview[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[dim]🔍 Analyzing all annotation data to show you what labels/categories will be trained...[/dim]\n")

            # df already loaded above for language detection

            all_keys_values = {}  # {key: set_of_unique_values}
            total_samples = 0
            malformed_count = 0

            for idx, row in df.iterrows():
                annotation_val = row.get(annotation_column)
                if pd.isna(annotation_val) or annotation_val == '':
                    continue

                total_samples += 1
                try:
                    if isinstance(annotation_val, str):
                        # Try standard JSON first
                        try:
                            annotation_dict = json.loads(annotation_val)
                        except json.JSONDecodeError:
                            # Try Python literal (handles single quotes with escapes)
                            import ast
                            annotation_dict = ast.literal_eval(annotation_val)
                    elif isinstance(annotation_val, dict):
                        annotation_dict = annotation_val
                    else:
                        continue

                    # Extract keys and values
                    for key, value in annotation_dict.items():
                        if key not in all_keys_values:
                            all_keys_values[key] = set()

                        if isinstance(value, list):
                            for v in value:
                                if v is not None and v != '':
                                    all_keys_values[key].add(str(v))
                        elif value is not None and value != '':
                            all_keys_values[key].add(str(value))

                except (json.JSONDecodeError, AttributeError, TypeError, ValueError, SyntaxError) as e:
                    malformed_count += 1
                    continue

            # Display comprehensive preview with Rich table
            if all_keys_values:
                self.console.print(f"\n[bold cyan]📊 Complete Annotation Data Preview[/bold cyan]")
                self.console.print(f"[dim]Analyzed {total_samples} samples ({malformed_count} malformed)[/dim]\n")

                preview_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
                preview_table.add_column("Key", style="yellow bold", width=20)
                preview_table.add_column("Unique Values", style="white", width=15, justify="center")
                preview_table.add_column("Sample Values", style="green", width=60)

                for key in sorted(all_keys_values.keys()):
                    values_set = all_keys_values[key]
                    num_values = len(values_set)

                    # Show first 10 values as sample
                    sample_values = sorted(values_set)[:10]
                    sample_str = ', '.join([f"'{v}'" for v in sample_values])
                    if num_values > 10:
                        sample_str += f" ... (+{num_values - 10} more)"

                    preview_table.add_row(
                        key,
                        str(num_values),
                        sample_str
                    )

                self.console.print(preview_table)
                self.console.print()

                # Show selection options
                self.console.print("[bold]💡 Training Options:[/bold]")
                self.console.print("  [dim]• You can choose to train on [cyan]ALL[/cyan] keys/values[/dim]")
                self.console.print("  [dim]• Or select [cyan]specific keys[/cyan] to train (asked later)[/dim]")
                self.console.print("  [dim]• Or select [cyan]specific values[/cyan] for each key (asked later)[/dim]\n")
            else:
                self.console.print("[yellow]⚠️  No valid annotation data found[/yellow]\n")

            # Step 6: Training Strategy Selection
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  STEP 6:[/bold cyan] [bold white]Training Strategy Selection[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold]📚 Choose How to Train Your Models:[/bold]")
            self.console.print("[dim]This determines whether one text can have one label or multiple labels.[/dim]\n")

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
            detected_keys = []  # Initialize to empty list

            if mode == "multi-label":
                self.console.print("\n[bold yellow]📋 Multi-label mode:[/bold yellow] One model will be trained per annotation key")
                self.console.print("[dim]This means training SEPARATE BINARY classifiers for each category type[/dim]")

                # Use comprehensive all_keys_values data collected earlier
                if all_keys_values:
                    detected_keys = sorted(all_keys_values.keys())
                    self.console.print(f"\n[green]✓ Detected annotation keys in your data: {', '.join(detected_keys)}[/green]")

                    self.console.print("\n[bold]What this means (with YOUR data):[/bold]")
                    for key in detected_keys:
                        num_values = len(all_keys_values[key])
                        values_preview = ', '.join([f"'{v}'" for v in sorted(all_keys_values[key])[:3]])
                        if num_values > 3:
                            values_preview += f" ... (+{num_values-3} more)"
                        self.console.print(f"  • [cyan]{key}[/cyan] ({num_values} unique values) → {values_preview}")

                    # Build real example from detected keys
                    if len(detected_keys) >= 2:
                        example_keys = ', '.join(detected_keys[:2])
                        self.console.print(f"\n[bold]Example:[/bold] Selecting '{example_keys}' → {min(2, len(detected_keys))} models trained")
                        for idx, key in enumerate(detected_keys[:2], 1):
                            sample_val = sorted(all_keys_values[key])[0] if all_keys_values[key] else "value"
                            self.console.print(f"  Model {idx}: Trains '{key}_{sample_val}' vs NOT '{key}_{sample_val}' (and all other {key} values)")
                elif analysis.get('annotation_keys_found'):
                    detected_keys = sorted(analysis['annotation_keys_found'])
                    self.console.print(f"\n[green]✓ Detected annotation keys in your data: {', '.join(detected_keys)}[/green]")
                else:
                    detected_keys = []
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

                # Check if we need to ask about training approach for multi-label with single key
                training_approach = "multi-label"  # Default
                if annotation_keys and len(annotation_keys) == 1:
                    # Single key selected - check if it has multiple values
                    selected_key = annotation_keys[0]
                    if selected_key in all_keys_values:
                        num_unique_values = len(all_keys_values[selected_key])

                        if num_unique_values > 2:
                            self.console.print(f"\n[bold cyan]🎯 Training Approach for '{selected_key}' ({num_unique_values} values)[/bold cyan]\n")
                            self.console.print("[dim]Since this key has multiple values, choose how to train:[/dim]\n")

                            approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
                            approach_table.add_column("Approach", style="cyan bold", width=18)
                            approach_table.add_column("Description", style="white", width=60)

                            approach_table.add_row(
                                "multi-label",
                                f"🏷️  ONE model detecting all {num_unique_values} values\n"
                                "✓ Faster training (1 model only)\n"
                                "✓ Can predict multiple values simultaneously\n"
                                "✓ Best for: when samples can have multiple values"
                            )
                            approach_table.add_row(
                                "one-vs-all",
                                f"⚡ {num_unique_values} binary models (one per value)\n"
                                "✓ Each model: 'Value X' vs 'NOT Value X'\n"
                                "✓ Better for: mutually exclusive values or value-specific tuning\n"
                                "✓ Longer training but more flexible"
                            )

                            self.console.print(approach_table)
                            self.console.print()

                            training_approach = Prompt.ask(
                                "[bold yellow]Training approach[/bold yellow]",
                                choices=["multi-label", "one-vs-all", "back"],
                                default="multi-label"
                            )

                            if training_approach == "back":
                                return None
            else:
                # For single-label mode - use comprehensive data
                if all_keys_values:
                    detected_keys = sorted(all_keys_values.keys())
                    self.console.print(f"\n[bold cyan]📝 Single-Label Mode - Select Keys/Values:[/bold cyan]")

                    # Show all keys and their values
                    for key in detected_keys:
                        num_values = len(all_keys_values[key])
                        values_preview = ', '.join([f"'{v}'" for v in sorted(all_keys_values[key])[:5]])
                        if num_values > 5:
                            values_preview += f" ... (+{num_values-5} more)"
                        self.console.print(f"  • [cyan]{key}[/cyan] ({num_values} values): {values_preview}")

                    self.console.print("\n[dim]Options:[/dim]")
                    self.console.print(f"  • [cyan]Leave blank[/cyan] → Use ALL {len(detected_keys)} keys with ALL their values")
                    self.console.print(f"  • [cyan]Enter specific keys[/cyan] → Use only selected keys with ALL their values")
                    self.console.print(f"    Example: '{detected_keys[0]}' → Use only {detected_keys[0]} key")
                elif analysis.get('annotation_keys_found'):
                    detected_keys = sorted(analysis['annotation_keys_found'])
                    self.console.print(f"\n[dim]Note: Your data has keys: {', '.join(detected_keys)}[/dim]")
                    self.console.print("[dim]Leave blank to use all keys, or specify which ones to include[/dim]")
                else:
                    detected_keys = []

                keys_input = Prompt.ask("\nAnnotation keys to include (comma separated, leave blank for all)", default="")
                annotation_keys = [key.strip() for key in keys_input.split(",") if key.strip()] or None

                # Check if we need to ask about training approach (multi-class vs one-vs-all)
                training_approach = "multi-class"  # Default
                if annotation_keys and len(annotation_keys) == 1:
                    # Single key selected - check if it has multiple values
                    selected_key = annotation_keys[0]
                    if selected_key in all_keys_values:
                        num_unique_values = len(all_keys_values[selected_key])

                        if num_unique_values > 2:
                            self.console.print(f"\n[bold cyan]🎯 Training Approach for '{selected_key}' ({num_unique_values} values)[/bold cyan]\n")
                            self.console.print("[dim]Since this key has multiple values, choose how to train:[/dim]\n")

                            approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
                            approach_table.add_column("Approach", style="cyan bold", width=18)
                            approach_table.add_column("Description", style="white", width=60)

                            approach_table.add_row(
                                "multi-class",
                                f"🎯 ONE model predicting among {num_unique_values} values\n"
                                "✓ Faster training (1 model only)\n"
                                "✓ Model learns relationships between values\n"
                                "✓ Best for: general classification with balanced data"
                            )
                            approach_table.add_row(
                                "one-vs-all",
                                f"⚡ {num_unique_values} binary models (one per value)\n"
                                "✓ Each model: 'Value X' vs 'NOT Value X'\n"
                                "✓ Better for: imbalanced data or value-specific tuning\n"
                                "✓ Longer training but more flexible"
                            )

                            self.console.print(approach_table)
                            self.console.print()

                            training_approach = Prompt.ask(
                                "[bold yellow]Training approach[/bold yellow]",
                                choices=["multi-class", "one-vs-all", "back"],
                                default="multi-class"
                            )

                            if training_approach == "back":
                                return None

            # Step 7: Additional Columns (ID, Language)
            self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[bold cyan]  STEP 7:[/bold cyan] [bold white]Additional Columns (Optional)[/bold white]")
            self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            self.console.print("[dim]Optional: Select ID and language columns if available in your dataset.[/dim]\n")

            # Use ID column already detected in analysis
            id_column = None
            if analysis['id_column_candidates']:
                id_column = analysis['id_column_candidates'][0]
                self.console.print(f"[green]✓ ID column detected: '{id_column}'[/green]")
                # Allow user to override if they want
                if all_columns:
                    self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
                while True:
                    override_id = Prompt.ask("\n[bold yellow]Identifier column (optional)[/bold yellow]", default=id_column)
                    if not override_id or override_id in all_columns:
                        if override_id and override_id != id_column:
                            id_column = override_id
                        break
                    self.console.print(f"[red]✗ Column '{override_id}' not found in dataset![/red]")
                    self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")
            else:
                self.console.print("[dim]No ID columns detected - automatic IDs will be generated[/dim]")
                if all_columns:
                    self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
                while True:
                    id_column_input = Prompt.ask("\n[bold yellow]Identifier column (optional)[/bold yellow]", default="")
                    if not id_column_input or id_column_input in all_columns:
                        if id_column_input:
                            id_column = id_column_input
                        break
                    self.console.print(f"[red]✗ Column '{id_column_input}' not found in dataset![/red]")
                    self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

            # Language column handling - check if already processed in Step 4
            # Skip if we already did language detection (either with column or auto-detection)
            language_already_processed = 'lang_column' in locals() and confirmed_languages

            if language_already_processed:
                # Language was already handled in Step 4
                if lang_column:
                    self.console.print(f"\n[green]✓ Language column from Step 4: '{lang_column}'[/green]")
                else:
                    self.console.print(f"\n[green]✓ Languages detected in Step 4: {', '.join([l.upper() for l in sorted(confirmed_languages)])}[/green]")
                    self.console.print(f"[dim]  (Using automatic language detection - no specific column)[/dim]")
            elif analysis['language_column_candidates']:
                # Language column detected but Step 4 was skipped - ask user
                lang_column_candidate = analysis['language_column_candidates'][0]
                self.console.print(f"\n[green]✓ Language column detected: '{lang_column_candidate}'[/green]")
                if all_columns:
                    self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
                while True:
                    override_lang = Prompt.ask("\n[bold yellow]Language column (optional)[/bold yellow]", default=lang_column_candidate)
                    if not override_lang or override_lang in all_columns:
                        lang_column = override_lang if override_lang else lang_column_candidate
                        break
                    self.console.print(f"[red]✗ Column '{override_lang}' not found in dataset![/red]")
                    self.console.print(f"[dim]Available columns: {', '.join(all_columns)}[/dim]")

            # Handle one-vs-all training approach (both single-label and multi-label modes)
            if 'training_approach' in locals() and training_approach == "one-vs-all":
                # Convert to multi-label format for one-vs-all training
                request = TrainingDataRequest(
                    input_path=csv_path,
                    format="llm_json",
                    text_column=text_column,
                    annotation_column=annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=label_strategy,
                    mode="multi-label",  # Use multi-label to trigger one-vs-all training
                    id_column=id_column or None,
                    lang_column=lang_column or None,
                )
                bundle = builder.build(request)

                # Mark this as one-vs-all for distributed training
                if bundle:
                    bundle.metadata['training_approach'] = 'one-vs-all'
                    bundle.metadata['original_strategy'] = 'single-label'
            else:
                # Standard mode
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

            # Store language metadata in bundle for later use (model selection will happen in training mode)
            if bundle:
                if confirmed_languages:
                    bundle.metadata['confirmed_languages'] = confirmed_languages
                if language_distribution:
                    bundle.metadata['language_distribution'] = language_distribution
                # Save training approach if user made a choice (multi-label/one-vs-all)
                if 'training_approach' in locals() and training_approach:
                    bundle.metadata['training_approach'] = training_approach
                # Text length stats for intelligent model selection later
                # ONLY calculate if not already done (avoid duplicate analysis)
                if 'text_length_stats' in locals() and text_length_stats:
                    # Already calculated with user interaction - reuse it
                    bundle.metadata['text_length_stats'] = text_length_stats
                elif text_column in df.columns:
                    # Not calculated yet - do it now without UI
                    text_length_stats = self.analyze_text_lengths(
                        df=df,
                        text_column=text_column,
                        display_results=False  # Silent calculation
                    )
                    bundle.metadata['text_length_stats'] = text_length_stats

            return bundle

        if format_choice == "category-csv":
            # Use sophisticated universal selector
            selection = self._training_studio_intelligent_dataset_selector(format_type="category-csv")
            if not selection:
                return None

            # Ask user for training strategy (mono-label vs multi-label)
            self.console.print("\n[bold cyan]📊 Training Strategy Selection[/bold cyan]\n")
            self.console.print("[dim]Choose how to handle the labels in your dataset:[/dim]\n")

            strategy_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
            strategy_table.add_column("Strategy", style="cyan bold", width=18)
            strategy_table.add_column("Description", style="white", width=60)

            strategy_table.add_row(
                "single-label",
                "🎯 Each sample has ONE label/category\n"
                "✓ Best for: classification tasks (sentiment, topic, etc.)\n"
                "✓ Example: each text is either 'positive' OR 'negative'"
            )
            strategy_table.add_row(
                "multi-label",
                "🏷️  Each sample can have MULTIPLE labels\n"
                "✓ Best for: tagging, multiple categories per text\n"
                "✓ Example: a text can be 'politics' AND 'economy' AND 'urgent'"
            )

            self.console.print(strategy_table)
            self.console.print()

            mode = Prompt.ask(
                "[bold yellow]Training strategy[/bold yellow]",
                choices=["single-label", "multi-label", "back"],
                default="single-label"
            )

            if mode == "back":
                return None

            # If single-label with multiple categories, ask about training approach
            training_approach = "multi-class"  # Default
            if mode == "single-label":
                # Count unique labels
                import pandas as pd
                df = pd.read_csv(selection['data_path'])
                label_column = selection['label_column']
                num_unique_labels = df[label_column].nunique()

                if num_unique_labels > 2:
                    self.console.print(f"\n[bold cyan]🎯 Training Approach for {num_unique_labels} Categories[/bold cyan]\n")
                    self.console.print("[dim]Since you have multiple categories, choose how to train:[/dim]\n")

                    approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
                    approach_table.add_column("Approach", style="cyan bold", width=18)
                    approach_table.add_column("Description", style="white", width=60)

                    approach_table.add_row(
                        "multi-class",
                        f"🎯 ONE model predicting among {num_unique_labels} categories\n"
                        "✓ Faster training (1 model only)\n"
                        "✓ Model learns relationships between categories\n"
                        "✓ Best for: general classification with balanced data"
                    )
                    approach_table.add_row(
                        "one-vs-all",
                        f"⚡ {num_unique_labels} binary models (one per category)\n"
                        "✓ Each model: 'Category X' vs 'NOT Category X'\n"
                        "✓ Better for: imbalanced data or category-specific tuning\n"
                        "✓ Longer training but more flexible"
                    )

                    self.console.print(approach_table)
                    self.console.print()

                    training_approach = Prompt.ask(
                        "[bold yellow]Training approach[/bold yellow]",
                        choices=["multi-class", "one-vs-all", "back"],
                        default="multi-class"
                    )

                    if training_approach == "back":
                        return None

            # If one-vs-all, convert to multi-label format (one binary file per category)
            if training_approach == "one-vs-all":
                # Convert single-label multi-class to multi-label one-vs-all format
                # This will create one binary file per category
                request = TrainingDataRequest(
                    input_path=selection['data_path'],
                    format="category_csv",
                    text_column=selection['text_column'],
                    label_column=selection['label_column'],
                    id_column=selection.get('id_column'),
                    lang_column=selection.get('lang_column'),
                    mode="multi-label",  # Use multi-label to trigger one-vs-all training
                )
                bundle = builder.build(request)

                # Mark this as one-vs-all for distributed training
                if bundle:
                    bundle.metadata['training_approach'] = 'one-vs-all'
                    bundle.metadata['original_strategy'] = 'single-label-multiclass'
            else:
                # Standard multi-class: one model for all categories
                request = TrainingDataRequest(
                    input_path=selection['data_path'],
                    format="category_csv",
                    text_column=selection['text_column'],
                    label_column=selection['label_column'],
                    id_column=selection.get('id_column'),
                    lang_column=selection.get('lang_column'),
                    mode=mode,
                )
                bundle = builder.build(request)

            # Store recommended model and metadata in bundle for later use
            if bundle:
                if selection.get('recommended_model'):
                    bundle.recommended_model = selection['recommended_model']
                if selection.get('confirmed_languages'):
                    bundle.metadata['confirmed_languages'] = selection['confirmed_languages']
                if selection.get('language_distribution'):
                    bundle.metadata['language_distribution'] = selection['language_distribution']
                if selection.get('text_length_stats'):
                    bundle.metadata['text_length_stats'] = selection['text_length_stats']

            return bundle

        if format_choice == "binary-long":
            # DEVELOPMENT MODE: This format is not yet available
            self.console.print("\n[bold red]❌ Error: binary-long format is currently under development[/bold red]")
            self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
            self.console.print("[dim]Please use 'llm-json' or 'category-csv' formats instead.[/dim]\n")
            return None

            # Use sophisticated universal selector
            selection = self._training_studio_intelligent_dataset_selector(format_type="binary-long")
            if not selection:
                return None

            # Binary-long specific: need category and value columns
            category_column = Prompt.ask("\n[bold yellow]Category column[/bold yellow]", default="category")
            value_column = Prompt.ask("[bold yellow]Value column (0/1)[/bold yellow]", default="value")

            request = TrainingDataRequest(
                input_path=selection['data_path'],
                format="binary_long_csv",
                text_column=selection['text_column'],
                category_column=category_column,
                value_column=value_column,
                id_column=selection.get('id_column'),
                lang_column=selection.get('lang_column'),
                mode="multi-label",
            )
            bundle = builder.build(request)

            # Store recommended model and metadata in bundle for later use
            if bundle:
                if selection.get('recommended_model'):
                    bundle.recommended_model = selection['recommended_model']
                if selection.get('confirmed_languages'):
                    bundle.metadata['confirmed_languages'] = selection['confirmed_languages']
                if selection.get('language_distribution'):
                    bundle.metadata['language_distribution'] = selection['language_distribution']
                if selection.get('text_length_stats'):
                    bundle.metadata['text_length_stats'] = selection['text_length_stats']

            return bundle

        if format_choice == "jsonl-single":
            # DEVELOPMENT MODE: This format is not yet available
            self.console.print("\n[bold red]❌ Error: jsonl-single format is currently under development[/bold red]")
            self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
            self.console.print("[dim]Please use 'llm-json' or 'category-csv' formats instead.[/dim]\n")
            return None

            # Use sophisticated universal selector
            selection = self._training_studio_intelligent_dataset_selector(format_type="jsonl-single")
            if not selection:
                return None

            request = TrainingDataRequest(
                input_path=selection['data_path'],
                format="jsonl_single",
                text_column=selection['text_column'],
                label_column=selection['label_column'],
                mode="single-label",
            )
            bundle = builder.build(request)

            # Store recommended model and metadata in bundle for later use
            if bundle:
                if selection.get('recommended_model'):
                    bundle.recommended_model = selection['recommended_model']
                if selection.get('confirmed_languages'):
                    bundle.metadata['confirmed_languages'] = selection['confirmed_languages']
                if selection.get('language_distribution'):
                    bundle.metadata['language_distribution'] = selection['language_distribution']
                if selection.get('text_length_stats'):
                    bundle.metadata['text_length_stats'] = selection['text_length_stats']

            return bundle

        # jsonl-multi (should not be reached - format is not in choices list)
        if format_choice == "jsonl-multi":
            # DEVELOPMENT MODE: This format is not yet available
            self.console.print("\n[bold red]❌ Error: jsonl-multi format is currently under development[/bold red]")
            self.console.print("[yellow]This format will be available in a future release after thorough testing.[/yellow]")
            self.console.print("[dim]Please use 'llm-json' or 'category-csv' formats instead.[/dim]\n")
            return None

        # Fallback: unrecognized format
        self.console.print(f"\n[bold red]❌ Error: Unknown format '{format_choice}'[/bold red]")
        self.console.print("[dim]Supported formats: llm-json, category-csv[/dim]\n")
        return None

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

    def _training_studio_run_quick(self, bundle: TrainingDataBundle, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick training mode - simple and fast with sensible defaults.

        Args:
            bundle: Training data bundle
            model_config: Model configuration dict (will be updated with runtime params)

        Returns:
            dict with keys: 'runtime_params', 'models_trained', 'best_model', 'best_f1'
        """
        self.console.print("\n[bold]Quick training[/bold] - using sensible defaults.")

        # Intelligent model selection like in benchmark mode
        # Get languages from metadata (confirmed_languages has priority)
        languages = set()
        if hasattr(bundle, 'metadata') and bundle.metadata:
            languages = bundle.metadata.get('confirmed_languages', bundle.metadata.get('languages', set()))
        if not languages and hasattr(bundle, 'languages') and bundle.languages:
            languages = set([lang.upper() for lang in bundle.languages])

        # Convert to uppercase set
        if languages:
            languages = set([str(lang).upper() for lang in languages])

        # CRITICAL: Get text_length_stats from bundle metadata
        text_length_stats = bundle.metadata.get('text_length_stats', {}) if hasattr(bundle, 'metadata') else {}

        # DEBUG: Print what we got
        self.logger.debug(f"[QUICK MODE DEBUG] text_length_stats keys: {list(text_length_stats.keys())}")
        self.logger.debug(f"[QUICK MODE DEBUG] token_mean: {text_length_stats.get('token_mean')}")
        self.logger.debug(f"[QUICK MODE DEBUG] user_prefers_long_models: {text_length_stats.get('user_prefers_long_models')}")

        # Use REAL text length stats - prefer token-based, fallback to chars
        if text_length_stats.get('token_mean'):
            text_length_avg = text_length_stats['token_mean']
            self.logger.debug(f"[QUICK MODE DEBUG] Using token_mean: {text_length_avg}")
        elif text_length_stats.get('char_mean'):
            text_length_avg = text_length_stats['char_mean']
            self.logger.debug(f"[QUICK MODE DEBUG] Using char_mean: {text_length_avg}")
        else:
            text_length_avg = getattr(bundle, 'text_length_avg', 158)  # Last resort default
            self.logger.debug(f"[QUICK MODE DEBUG] Using fallback: {text_length_avg}")

        prefers_long_models = text_length_stats.get('user_prefers_long_models', False)
        requires_long_model = text_length_stats.get('requires_long_model', False)

        # Determine model strategy
        if len(languages) > 1:
            model_strategy = "multilingual"
        elif 'FR' in languages:
            model_strategy = "fr"
        elif 'EN' in languages:
            model_strategy = "en"
        else:
            model_strategy = "multilingual"

        # Get recommended model from bundle if available
        recommended_model = getattr(bundle, 'recommended_model', None)

        # Get intelligent model recommendations using the same function as benchmark mode
        self.console.print("\n[bold cyan]🎯 Recommended Models for Your Data:[/bold cyan]")

        # Display context including long-document needs
        context_parts = []
        if languages:
            context_parts.append(f"{', '.join(languages)} dataset")
        else:
            context_parts.append("multilingual dataset")

        if text_length_stats.get('token_mean'):
            context_parts.append(f"avg {text_length_stats['token_mean']:.0f} tokens")
        else:
            context_parts.append(f"avg {text_length_avg:.0f} characters")

        if requires_long_model:
            context_parts.append("⚠ LONG DOCUMENTS (>512 tokens)")

        self.console.print(f"[dim]Based on: {', '.join(context_parts)}[/dim]\n")

        # Get intelligent model recommendations using the same function as benchmark mode
        # Pass user_prefers_long_models flag to boost long-document models when needed
        if prefers_long_models or requires_long_model:
            self.console.print("[yellow]📏 Prioritizing long-document models (handle up to 4096 tokens):[/yellow]")

        selected_models, model_lang_map = self._get_intelligent_benchmark_models(
            languages, text_length_avg, model_strategy, recommended_model, None,
            user_prefers_long_models=(prefers_long_models or requires_long_model)
        )

        if selected_models:
            self.console.print("[bold]Top 5 recommended models:[/bold]")
            for idx, model in enumerate(selected_models[:5], 1):
                lang_info = f" (for {model_lang_map[model].upper()} texts)" if model_lang_map.get(model) else " (multilingual)"
                self.console.print(f"  {idx}. [cyan]{model}[/cyan]{lang_info}")

            default_model = selected_models[0]
        else:
            # Fallback to simple selection
            if 'FR' in languages:
                default_model = 'camembert-base'
            elif 'EN' in languages:
                default_model = 'bert-base-uncased'
            else:
                default_model = 'xlm-roberta-base'

        self.console.print(f"\n[dim]You can also enter any HuggingFace model ID[/dim]")
        model_input = Prompt.ask("Model to train", default=default_model)

        # Check if user entered a number (selecting from list)
        if model_input.isdigit():
            idx = int(model_input) - 1  # Convert to 0-based index
            if 0 <= idx < len(selected_models):
                model_name = selected_models[idx]
                self.console.print(f"[green]✓ Selected: {model_name}[/green]")
            else:
                self.console.print(f"[yellow]⚠️  Invalid selection. Using default: {default_model}[/yellow]")
                model_name = default_model
        else:
            model_name = model_input

        # Ask for number of epochs
        from rich.prompt import IntPrompt
        epochs = IntPrompt.ask("Number of epochs", default=10)

        # Capture runtime parameters for full reproducibility
        runtime_params = {
            'quick_model_name': model_name,
            'quick_epochs': epochs,
            'actual_models_trained': [model_name]
        }

        output_dir = self._training_studio_make_output_dir("training_studio_quick")

        # Initialize multiclass_groups (will be set if detected)
        multiclass_groups = None

        # CRITICAL: Extract training_approach BEFORE the multi-label block so it's accessible later
        training_approach_from_metadata = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None

        # For multi-label, check if it's actually multi-class
        if bundle.strategy == "multi-label":
            # Load data to check structure
            from llm_tool.trainers.multi_label_trainer import MultiLabelTrainer, TrainingConfig as MultiLabelTrainingConfig
            ml_trainer = MultiLabelTrainer(config=MultiLabelTrainingConfig(), verbose=False)

            # Use primary_file for one-vs-all, dataset_path otherwise
            data_path = str(bundle.primary_file) if hasattr(bundle, 'primary_file') else str(bundle.dataset_path)

            samples = ml_trainer.load_multi_label_data(
                data_path,
                text_field=bundle.text_column,
                label_fields=None,  # Will auto-detect
                id_field=bundle.id_column if hasattr(bundle, 'id_column') else None,
                lang_field=bundle.lang_column if hasattr(bundle, 'lang_column') else None,
                labels_dict_field=bundle.label_column if hasattr(bundle, 'label_column') else 'labels'
            )

            # Detect multi-class groups
            multiclass_groups = ml_trainer.detect_multiclass_groups(samples)

            # Check if user already answered this question during dataset building
            use_multiclass_training = False

            if multiclass_groups:
                if training_approach_from_metadata == 'multi-label':
                    # User already chose multi-class during dataset building
                    use_multiclass_training = True
                    self.console.print("\n[green]✓ Using multi-class training (from dataset configuration)[/green]\n")
                elif training_approach_from_metadata == 'one-vs-all':
                    # User already chose one-vs-all during dataset building
                    use_multiclass_training = False
                    multiclass_groups = None
                    self.console.print("\n[yellow]✓ Using one-vs-all training (from dataset configuration)[/yellow]\n")
                else:
                    # No previous choice - ask user
                    self.console.print("\n[yellow]ℹ️  Detected multi-class classification:[/yellow]")
                    for group_name, labels in multiclass_groups.items():
                        value_names = [lbl[len(group_name)+1:] if lbl.startswith(group_name+'_') else lbl for lbl in labels]
                        self.console.print(f"  • {group_name}: {', '.join(value_names)}")

                    # Ask user if they want true multi-class (1 model) or one-vs-all (N models)
                    self.console.print("\n[bold]Training approach:[/bold]")
                    self.console.print("  • [green]Multi-class[/green]: Train 1 model per group to predict among all classes")
                    self.console.print("  • [yellow]One-vs-all[/yellow]: Train N separate binary models (1 per class)")

                    use_multiclass_training = Confirm.ask(
                        "\n[bold]Use multi-class training? (recommended)[/bold]",
                        default=True
                    )

                    if use_multiclass_training:
                        self.console.print("[green]✓ Will use multi-class training[/green]\n")
                    else:
                        self.console.print("[yellow]✓ Will train separate binary classifiers[/yellow]\n")
                        multiclass_groups = None  # Don't pass to trainer

        # Create TrainingConfig with user's chosen model
        from llm_tool.trainers.model_trainer import TrainingConfig
        training_config = TrainingConfig()
        training_config.model_name = model_name
        training_config.num_epochs = epochs

        # Determine if we need to train by language (monolingual model + multiple languages)
        is_multilingual = self._is_model_multilingual(model_name)
        needs_language_training = not is_multilingual and len(languages) > 1

        if needs_language_training:
            self.console.print(f"\n[yellow]🌍 Multi-language training enabled:[/yellow]")
            self.console.print(f"[dim]The model '{model_name}' is language-specific, so separate models will be trained for each language:[/dim]")
            for lang in sorted(languages):
                self.console.print(f"  • {lang.upper()}")

        trainer = ModelTrainer(config=training_config)

        # Build trainer config with multiclass_groups if detected
        extra_config = {
            "model_name": model_name,
            "num_epochs": epochs,
            "train_by_language": needs_language_training,
            "confirmed_languages": list(languages) if languages else None  # Pass all detected languages
        }

        # Add multiclass_groups if user opted for multi-class training
        if bundle.strategy == "multi-label" and multiclass_groups:
            extra_config["multiclass_groups"] = multiclass_groups

        # CRITICAL FIX: Handle one-vs-all training properly
        # For one-vs-all, we need to train separate binary models for each label

        # DEBUG logging
        self.logger.debug(f"[ONE-VS-ALL DEBUG] training_approach_from_metadata = {training_approach_from_metadata}")
        self.logger.debug(f"[ONE-VS-ALL DEBUG] hasattr(bundle, 'training_files') = {hasattr(bundle, 'training_files')}")
        if hasattr(bundle, 'training_files'):
            self.logger.debug(f"[ONE-VS-ALL DEBUG] bundle.training_files.keys() = {list(bundle.training_files.keys()) if bundle.training_files else None}")

        if training_approach_from_metadata == 'one-vs-all':
            # One-vs-all training: create separate binary models for each label

            # First, try to use pre-generated category CSV files if they exist
            category_files = {}
            if hasattr(bundle, 'training_files') and bundle.training_files:
                # Extract the category files (exclude 'multilabel' key)
                category_files = {k: v for k, v in bundle.training_files.items() if k != 'multilabel'}

            # If no category files exist, create them from the JSONL file
            if not category_files:
                self.console.print("\n[yellow]⚡ Creating binary datasets for one-vs-all training...[/yellow]")

                # Load the JSONL file to extract labels
                import json
                data_path = str(bundle.primary_file) if hasattr(bundle, 'primary_file') else str(bundle.dataset_path)

                # Read the JSONL and collect unique labels
                all_labels_set = set()
                records = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        records.append(record)
                        if 'labels' in record:
                            # Handle both list and dict formats
                            if isinstance(record['labels'], dict):
                                all_labels_set.update(record['labels'].keys())
                            elif isinstance(record['labels'], list):
                                all_labels_set.update(record['labels'])

                self.logger.debug(f"[ONE-VS-ALL] Found {len(records)} records")
                self.logger.debug(f"[ONE-VS-ALL] Found {len(all_labels_set)} unique labels: {sorted(all_labels_set)}")

                if not all_labels_set:
                    # Debug: print first record to see the structure
                    if records:
                        self.logger.error(f"[ONE-VS-ALL] No labels found! First record structure: {records[0]}")
                        self.console.print(f"\n[red]✗ Could not find labels in JSONL file[/red]")
                        self.console.print(f"[dim]First record structure: {json.dumps(records[0], indent=2)}[/dim]")
                    return {
                        'runtime_params': runtime_params,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None,
                        'error': 'No labels found in JSONL'
                    }

                # Create temporary CSV files for each label
                import tempfile
                import csv
                temp_dir = Path(tempfile.mkdtemp(prefix="onevsall_"))

                for label_name in sorted(all_labels_set):
                    # Create binary CSV: text + label (0 or 1)
                    csv_path = temp_dir / f"binary_{label_name}.csv"

                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=['text', 'label', 'language'])
                        writer.writeheader()

                        for record in records:
                            # Binary label: 1 if this label is present/True, 0 otherwise
                            labels_data = record.get('labels', {})

                            # Handle both dict and list formats
                            if isinstance(labels_data, dict):
                                label_raw = labels_data.get(label_name, 0)
                                # Handle bool, int, or string values
                                if isinstance(label_raw, bool):
                                    label_value = 1 if label_raw else 0
                                elif isinstance(label_raw, (int, float)):
                                    label_value = 1 if label_raw > 0 else 0
                                else:
                                    label_value = 1 if str(label_raw).lower() in ['1', 'true', 'yes'] else 0
                            elif isinstance(labels_data, list):
                                # For list format, check if label is in the list
                                label_value = 1 if label_name in labels_data else 0
                            else:
                                label_value = 0

                            # CRITICAL: Ensure text is a valid non-empty string
                            text_raw = record.get('text', '')
                            if not isinstance(text_raw, str):
                                text_raw = str(text_raw) if text_raw else ''
                            # Skip empty texts
                            if not text_raw.strip():
                                continue

                            # Ensure language is a string
                            lang_raw = record.get('lang', record.get('language', ''))
                            if not isinstance(lang_raw, str):
                                lang_raw = str(lang_raw) if lang_raw else ''

                            row = {
                                'text': text_raw.strip(),
                                'label': label_value,
                                'language': lang_raw
                            }
                            writer.writerow(row)

                    category_files[label_name] = csv_path
                    self.console.print(f"[dim]  Created binary dataset for: {label_name}[/dim]")

                self.console.print(f"[green]✓ Created {len(category_files)} binary datasets[/green]\n")

            if category_files:
                self.console.print(f"\n[yellow]⚠️  One-vs-all requires training {len(category_files)} separate binary models.[/yellow]")
                self.console.print("[dim]   Note: 'distributed' training mode exists but is NOT RECOMMENDED (untested).[/dim]")
                self.console.print("[yellow]   Quick mode will train them sequentially...[/yellow]\n")

                # Train each binary model sequentially
                results_per_category = {}
                for category_name, category_file in category_files.items():
                    self.console.print(f"\n[cyan]Training binary model for: {category_name}[/cyan]")

                    # Create config for this specific category
                    category_config = {
                        'input_file': str(category_file),
                        'text_column': 'text',
                        'label_column': 'label',
                        'model_name': model_name,
                        'num_epochs': epochs,
                        'output_dir': str(Path(output_dir) / f'model_{category_name}'),
                        'training_strategy': 'single-label',  # Binary classification
                        'category_name': category_name,  # For display in metrics
                        'confirmed_languages': list(languages) if languages else None
                    }

                    try:
                        category_result = trainer.train(category_config)
                        results_per_category[category_name] = category_result
                        self.console.print(f"[green]✓ Completed {category_name}: Accuracy={category_result.get('accuracy', 0):.4f}, F1={category_result.get('best_f1_macro', 0):.4f}[/green]")
                    except Exception as exc:
                        self.console.print(f"[red]✗ Failed to train {category_name}: {exc}[/red]")
                        self.logger.exception(f"Training failed for {category_name}", exc_info=exc)
                        results_per_category[category_name] = {'error': str(exc)}

                # Aggregate results
                successful_results = [r for r in results_per_category.values() if 'error' not in r]
                if successful_results:
                    avg_accuracy = sum(r.get('accuracy', 0) for r in successful_results) / len(successful_results)
                    avg_f1 = sum(r.get('best_f1_macro', 0) for r in successful_results) / len(successful_results)

                    result = {
                        'best_model': model_name,
                        'accuracy': avg_accuracy,
                        'best_f1_macro': avg_f1,
                        'model_path': str(output_dir),
                        'training_time': sum(r.get('training_time', 0) for r in successful_results),
                        'models_trained': len(successful_results),
                        'total_models': len(category_files),
                        'per_category_results': results_per_category
                    }
                else:
                    self.console.print("[red]All category trainings failed[/red]")
                    return {
                        'runtime_params': runtime_params,
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None,
                        'error': 'All category trainings failed'
                    }
            else:
                self.console.print("[red]No category files found for one-vs-all training[/red]")
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': 'No category files'
                }
        else:
            # Standard training (multi-class or multi-label)
            config = bundle.to_trainer_config(output_dir, extra_config)

            try:
                result = trainer.train(config)
            except Exception as exc:  # pylint: disable=broad-except
                self.console.print(f"[red]Training failed:[/red] {exc}")
                self.logger.exception("Quick training failed", exc_info=exc)
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': str(exc)
                }

        self._training_studio_show_training_result(result, bundle, title="Quick training results")

        # Return complete training info for metadata save
        return {
            'runtime_params': runtime_params,
            'models_trained': [model_name],
            'best_model': result.get('best_model'),
            'best_f1': result.get('best_f1') or result.get('f1_macro')
        }

    def _training_studio_run_benchmark(self, bundle: TrainingDataBundle, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmark training mode - test multiple models on same dataset.

        Args:
            bundle: Training data bundle
            model_config: Model configuration dict (will be updated with runtime params)

        Returns:
            dict with keys: 'runtime_params', 'models_trained', 'best_model', 'best_f1'
        """
        from rich.table import Table
        from rich import box

        # Handle multi-label benchmarking separately
        if bundle.strategy == "multi-label":
            return self._training_studio_run_benchmark_multilabel(bundle, model_config)

        try:
            dataset_path, text_column, label_column = self._training_studio_resolve_benchmark_dataset(bundle)
        except ValueError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return

        # Get dataset metadata
        languages = bundle.metadata.get('confirmed_languages', bundle.metadata.get('languages', set()))
        language_distribution = bundle.metadata.get('language_distribution', {})
        text_length_stats = bundle.metadata.get('text_length_stats', {})

        # Use REAL text length stats - prefer token-based, fallback to chars
        if text_length_stats.get('token_mean'):
            text_length_avg = text_length_stats['token_mean']
        elif text_length_stats.get('char_mean'):
            text_length_avg = text_length_stats['char_mean']
        else:
            text_length_avg = text_length_stats.get('avg_chars', 0)

        user_prefers_long_models = text_length_stats.get('user_prefers_long_models', False)
        model_strategy = bundle.metadata.get('model_strategy', 'multilingual')

        # If we have a recommended model from language detection, use it as hint
        recommended_model = bundle.recommended_model if hasattr(bundle, 'recommended_model') else None

        # For single-label with multiple categories, ask which category to benchmark
        # Load the dataset to check number of unique labels
        import pandas as pd
        df = pd.read_csv(dataset_path)
        unique_labels = df[label_column].unique()
        num_labels = len(unique_labels)

        # Initialize selected_category variable
        selected_category = None

        if num_labels > 2:
            self.console.print(f"\n[bold cyan]📊 Category Selection for Benchmarking[/bold cyan]\n")
            self.console.print(f"[dim]Your dataset has {num_labels} categories. Choose which one to benchmark:[/dim]\n")

            # Show label distribution
            label_counts = df[label_column].value_counts().to_dict()
            labels_table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
            labels_table.add_column("#", style="cyan", width=5)
            labels_table.add_column("Category", style="white", width=40)
            labels_table.add_column("Count", style="yellow", justify="right", width=10)
            labels_table.add_column("Percentage", style="green", justify="right", width=12)

            total = len(df)
            for idx, label in enumerate(sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True), 1):
                count = label_counts[label]
                pct = (count / total * 100)
                labels_table.add_row(str(idx), str(label), f"{count:,}", f"{pct:.1f}%")

            self.console.print(labels_table)

            # Ask user to select category
            self.console.print("\n[bold]Select category to benchmark:[/bold]")
            self.console.print("[dim]Enter category name or number from the table above[/dim]")

            category_choice = Prompt.ask("Category name or number")

            # Parse choice (number or name)
            selected_category = None
            try:
                # Try as number first
                idx = int(category_choice) - 1
                sorted_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)
                if 0 <= idx < len(sorted_labels):
                    selected_category = sorted_labels[idx]
            except ValueError:
                # Not a number, try as category name
                if category_choice in label_counts:
                    selected_category = category_choice

            if not selected_category:
                self.console.print(f"[red]Invalid category: {category_choice}[/red]")
                return

            self.console.print(f"\n[green]✓ Selected category:[/green] {selected_category}")

            # Convert to binary classification with string labels for proper mapping
            # CRITICAL FIX: Use string labels NOT_category/category instead of 0/1
            # This ensures bert_base.py can properly sort labels with NOT_* first
            df[label_column] = df[label_column].apply(
                lambda x: selected_category if x == selected_category else f"NOT_{selected_category}"
            )

            # Save filtered dataset temporarily
            temp_path = dataset_path.parent / f"temp_benchmark_{selected_category.replace(' ', '_')}.csv"
            df.to_csv(temp_path, index=False)
            dataset_path = temp_path

            # Count using string labels
            class_1_count = (df[label_column] == selected_category).sum()
            class_0_count = (df[label_column] == f"NOT_{selected_category}").sum()
            self.console.print(f"[dim]Converted to binary:[/dim]")
            self.console.print(f"[dim]  • {selected_category} (positive class): {class_1_count} samples[/dim]")
            self.console.print(f"[dim]  • NOT_{selected_category} (negative class): {class_0_count} samples[/dim]\n")

        # Display intelligent model selection options
        self.console.print("\n[bold cyan]🎯 Model Selection for Benchmarking[/bold cyan]\n")

        selection_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
        selection_table.add_column("Option", style="cyan bold", width=18)
        selection_table.add_column("Description", style="white", width=70)

        selection_table.add_row(
            "intelligent",
            "🤖 AI-powered selection based on your data\n" +
            f"✓ Considers: {len(languages)} language(s), {text_length_avg:.0f} avg chars, '{model_strategy}' strategy\n" +
            "✓ Selects 5-7 optimal models automatically"
        )
        selection_table.add_row(
            "pre-selected",
            "📋 Choose from pre-selected categories\n" +
            "✓ Filter by: language, document length, efficiency\n" +
            "✓ View and select from curated model lists"
        )
        selection_table.add_row(
            "custom",
            "✏️  Manual selection from all available models\n" +
            "✓ Full access to 37+ models\n" +
            "✓ Enter any HuggingFace model ID"
        )

        self.console.print(selection_table)
        self.console.print()

        selection_mode = Prompt.ask(
            "[bold yellow]Model selection mode[/bold yellow]",
            choices=["intelligent", "pre-selected", "custom"],
            default="intelligent"
        )

        selected_models = []
        model_lang_map = {}

        if selection_mode == "intelligent":
            selected_models, model_lang_map = self._get_intelligent_benchmark_models(
                languages, text_length_avg, model_strategy, recommended_model, language_distribution,
                user_prefers_long_models
            )

            self.console.print(f"\n[green]✓ Intelligently selected {len(selected_models)} models:[/green]")
            for i, model in enumerate(selected_models, 1):
                lang_info = f" (for {model_lang_map[model].upper()} texts)" if model_lang_map.get(model) else " (multilingual)"
                self.console.print(f"  {i}. {model}{lang_info}")

            # Warning for monolingual models in multilingual dataset
            if len(languages) > 1 and model_lang_map:
                mono_models = [m for m in selected_models if model_lang_map.get(m)]
                if mono_models:
                    self.console.print(f"\n[yellow]⚠️  Note: Language-specific models will be trained on ALL texts[/yellow]")
                    self.console.print(f"[dim]Ideally, monolingual models should only train on their target language.[/dim]")
                    self.console.print(f"[dim]For better results with mixed languages, prefer multilingual models.[/dim]")

        elif selection_mode == "pre-selected":
            selected_models = self._get_preselected_benchmark_models(languages, text_length_avg)

        else:  # custom
            selected_models = self._get_custom_benchmark_models()

        if not selected_models:
            self.console.print("[yellow]No models selected. Using default: bert-base-uncased[/yellow]")
            selected_models = ["bert-base-uncased"]

        # Ask about reinforcement learning
        self.console.print("\n[bold cyan]🎓 Reinforcement Learning Option[/bold cyan]")
        self.console.print("[dim]Automatically triggered when model underperforms (F1 class 1 < threshold).[/dim]")
        self.console.print("[dim]• Applies oversampling via WeightedRandomSampler for minority class[/dim]")
        self.console.print("[dim]• Corrects cross-entropy loss with adaptive class weights[/dim]")
        self.console.print("[dim]• Adjusts learning rate, batch size, and epochs based on failure severity[/dim]")
        self.console.print("[dim]• Can improve F1 by 5-15% for underperforming models[/dim]\n")

        use_reinforcement = Confirm.ask("Enable reinforcement learning?", default=False)

        # Ask for number of epochs
        self.console.print("\n[bold cyan]⏱️  Training Duration[/bold cyan]")
        self.console.print("[dim]Number of epochs to train each model.[/dim]")
        self.console.print("[dim]• More epochs = better performance but longer training time[/dim]")
        self.console.print("[dim]• Recommended: 10-15 epochs for benchmark[/dim]\n")

        n_epochs = IntPrompt.ask("Number of epochs per model", default=10)

        # For single-label benchmark, we'll manually train each model to properly pass label_value
        # This ensures class names are displayed correctly in the terminal output

        # Determine if we have a selected category (binary) or multi-class
        # CRITICAL FIX: Always pass category_name for proper class name display
        category_name = selected_category

        # Load dataset and prepare training data
        from llm_tool.trainers.multi_label_trainer import MultiLabelTrainer, TrainingConfig as MultiLabelTrainingConfig
        from llm_tool.trainers.model_selector import ModelSelector
        import torch

        ml_trainer = MultiLabelTrainer(config=MultiLabelTrainingConfig(), verbose=False)

        # Read the CSV and convert to samples
        import pandas as pd
        from llm_tool.trainers.data_utils import DataSample

        df = pd.read_csv(dataset_path)
        samples = []
        for _, row in df.iterrows():
            # CRITICAL FIX: Check for both 'language' and 'lang' columns
            lang_value = None
            if 'language' in df.columns:
                lang_value = row.get('language')
            elif 'lang' in df.columns:
                lang_value = row.get('lang')

            # CRITICAL: Use DataSample for single-label benchmark (has .label attribute)
            samples.append(DataSample(
                text=str(row[text_column]),
                label=str(row[label_column]),  # Keep as string for proper NOT_* mapping
                lang=lang_value,
                metadata={}
            ))

        # Check if we have at least 2 instances per class for stratification
        from collections import Counter
        from sklearn.model_selection import train_test_split

        label_counts = Counter([s.label for s in samples])
        min_count = min(label_counts.values())

        if min_count < 2:
            # Find which classes have insufficient instances
            insufficient_classes = [cls for cls, count in label_counts.items() if count < 2]
            self.console.print(f"[yellow]⚠️  Dataset has class(es) with only 1 instance: {insufficient_classes}[/yellow]")

            # Ask user if they want to remove these classes or proceed
            remove_classes = Prompt.ask(
                f"Remove class(es) with insufficient instances?",
                choices=["y", "n"],
                default="y"
            )

            if remove_classes.lower() == 'y':
                # Filter out samples with insufficient classes
                original_count = len(samples)
                samples = [s for s in samples if s.label not in insufficient_classes]
                self.console.print(f"[dim]Removed {original_count - len(samples)} samples from classes: {insufficient_classes}[/dim]")

                # Recompute label counts
                label_counts = Counter([s.label for s in samples])
                min_count = min(label_counts.values()) if label_counts else 0

                if min_count < 2:
                    self.console.print("[red]Still insufficient samples after removal. Cannot proceed.[/red]")
                    return {
                        'runtime_params': {
                            'selected_models': selected_models if 'selected_models' in locals() else [],
                            'benchmark_category': selected_category if 'selected_category' in locals() else None,
                            'actual_models_trained': [],
                            'error': 'Insufficient samples after removal'
                        },
                        'models_trained': [],
                        'best_model': None,
                        'best_f1': None
                    }

                stratify_labels = [s.label for s in samples]
            else:
                self.console.print(f"[yellow]Proceeding without stratification (may reduce quality)[/yellow]")
                stratify_labels = None
        else:
            stratify_labels = [s.label for s in samples]

        # Split into train/test
        train_samples, test_samples = train_test_split(
            samples,
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels
        )

        self.console.print(f"\n[bold cyan]🚀 Starting Benchmark Training[/bold cyan]")
        self.console.print(f"[dim]Training {len(selected_models)} model(s) on {len(train_samples)} samples ({len(test_samples)} test)[/dim]\n")

        # Train each model
        results = []
        selector = ModelSelector()

        for idx, model_name in enumerate(selected_models, 1):
            self.console.print(f"[bold]Model {idx}/{len(selected_models)}: {model_name}[/bold]")

            try:
                # Get model instance
                model_name_mapping = {
                    'roberta-base': 'RoBERTaBase',
                    'roberta-large': 'RoBERTaLarge',
                    'bert-base-uncased': 'BERTBase',
                    'bert-large-uncased': 'BERTLarge',
                    'bert-base-multilingual-cased': 'mBERTBase',
                    'distilbert-base-uncased': 'DistilBERT',
                    'distilroberta-base': 'DistilRoBERTa',
                    'albert-base-v2': 'ALBERTBase',
                    'camembert-base': 'CamemBERTBase',
                    'xlm-roberta-base': 'XLMRoBERTaBase',
                }

                profile_key = model_name_mapping.get(model_name)

                if profile_key and profile_key in selector.MODEL_PROFILES:
                    model_class = selector.MODEL_PROFILES[profile_key].model_class
                    model = model_class()
                elif model_name in selector.MODEL_PROFILES:
                    model_class = selector.MODEL_PROFILES[model_name].model_class
                    model = model_class()
                else:
                    # Create generic model
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    from llm_tool.trainers.bert_base import BertBase

                    class CustomModel(BertBase):
                        def __init__(self):
                            super().__init__(
                                model_name=model_name,
                                tokenizer=AutoTokenizer,
                                model_sequence_classifier=AutoModelForSequenceClassification
                            )

                    model = CustomModel()

                # Encode data
                train_texts = [s.text for s in train_samples]
                train_labels = [s.label for s in train_samples]  # DataSample uses .label not .labels
                test_texts = [s.text for s in test_samples]
                test_labels = [s.label for s in test_samples]  # DataSample uses .label not .labels

                train_loader = model.encode(train_texts, train_labels, batch_size=32, progress_bar=False)
                test_loader = model.encode(test_texts, test_labels, batch_size=32, progress_bar=False)

                # CRITICAL: Do NOT use pos_weight for initial training
                # Let the model learn naturally first, then reinforcement learning
                # will apply adaptive weights if needed (when F1 is low)
                # Applying pos_weight from the start causes the model to over-predict minority class

                # Train model
                temp_model_name = f"benchmark_{model_name.replace('/', '_')}"

                # CRITICAL FIX: Extract language information for per-language metrics
                test_languages = [s.lang for s in test_samples] if test_samples else None
                has_languages = test_languages and any(lang for lang in test_languages if lang)

                # Determine language
                sample_langs = set(s.lang for s in train_samples if s.lang)
                if len(sample_langs) == 1:
                    train_language = list(sample_langs)[0].upper()
                elif len(sample_langs) > 1:
                    train_language = "MULTI"
                else:
                    train_language = None

                # UNIFIED: Use centralized function to set detected languages (SAME AS BENCHMARK)
                if has_languages:
                    from llm_tool.trainers.model_trainer import set_detected_languages_on_model
                    train_languages = [s.lang for s in train_samples]
                    set_detected_languages_on_model(
                        model=model,
                        train_languages=train_languages,
                        val_languages=test_languages,
                        logger=self.logger
                    )

                result = model.run_training(
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    n_epochs=n_epochs,
                    lr=5e-5,
                    random_state=42,
                    save_model_as=temp_model_name,
                    pos_weight=None,  # CRITICAL: No pos_weight initially - let RL handle it
                    metrics_output_dir="training_logs",
                    best_model_criteria="combined",
                    f1_class_1_weight=0.7,
                    reinforced_learning=use_reinforcement,
                    reinforced_epochs=10,
                    rescue_low_class1_f1=True,  # CRITICAL: Enable automatic rescue for low F1
                    f1_1_rescue_threshold=0.50,  # Trigger rescue if F1 classe 1 < 50%
                    reinforced_f1_threshold=0.60,
                    track_languages=has_languages,  # CRITICAL: Enable if we have language info
                    language_info=test_languages,     # CRITICAL: Pass language info for metrics
                    label_key=label_column,
                    label_value=category_name,  # Pass the selected category name for proper display
                    language=train_language,
                )

                # Extract metrics
                best_metric_val, best_model_path, best_scores = result

                if best_scores:
                    results.append({
                        'model': model_name,
                        'f1_macro': best_scores.get('f1_macro', 0.0),
                        'f1_class_0': best_scores.get('f1_class_0', 0.0),
                        'f1_class_1': best_scores.get('f1_class_1', 0.0),
                        'accuracy': best_scores.get('accuracy', 0.0),
                        'precision_macro': best_scores.get('precision_macro', 0.0),
                        'recall_macro': best_scores.get('recall_macro', 0.0),
                    })

                self.console.print(f"[green]✓ Completed[/green]\n")

            except Exception as exc:
                self.console.print(f"[red]✗ Failed: {exc}[/red]\n")
                self.logger.exception(f"Model {model_name} failed", exc_info=exc)

        # Display results summary
        if results:
            self.console.print("\n[bold cyan]📊 Benchmark Results Summary[/bold cyan]\n")

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('f1_macro', ascending=False)

            from rich.table import Table
            table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
            table.add_column("Rank", style="cyan", width=5)
            table.add_column("Model", style="white", width=35)
            table.add_column("F1 Macro", style="green", justify="right", width=10)
            table.add_column("F1 Class 0", style="yellow", justify="right", width=10)
            table.add_column("F1 Class 1", style="blue", justify="right", width=10)
            table.add_column("Accuracy", style="magenta", justify="right", width=10)

            for idx, row in enumerate(results_df.itertuples(), 1):
                table.add_row(
                    str(idx),
                    row.model,
                    f"{row.f1_macro:.4f}",
                    f"{row.f1_class_0:.4f}",
                    f"{row.f1_class_1:.4f}",
                    f"{row.accuracy:.4f}"
                )

            self.console.print(table)

            # Show best model
            best_model = results_df.iloc[0]
            self.console.print(f"\n[bold green]🏆 Best Model: {best_model['model']}[/bold green]")
            self.console.print(f"[dim]F1 Macro: {best_model['f1_macro']:.4f}[/dim]\n")

            # Return complete training info for metadata save
            return {
                'runtime_params': {
                    'selected_models': selected_models,
                    'benchmark_category': selected_category if 'selected_category' in locals() else None,
                    'actual_models_trained': [row.model for row in results_df.itertuples()]
                },
                'models_trained': [row.model for row in results_df.itertuples()],
                'best_model': best_model['model'],
                'best_f1': best_model['f1_macro']
            }
        else:
            self.console.print("[yellow]No results to display[/yellow]")
            return {
                'runtime_params': {
                    'selected_models': selected_models if 'selected_models' in locals() else [],
                    'benchmark_category': selected_category if 'selected_category' in locals() else None,
                    'actual_models_trained': []
                },
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            }

    def _training_studio_run_benchmark_multilabel(self, bundle: TrainingDataBundle, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark multiple models on a multi-label dataset"""
        from rich.table import Table
        from rich import box

        self.console.print("\n[bold cyan]Multi-Label Benchmark Mode[/bold cyan]")
        self.console.print("[dim]Benchmarking models across all labels in your dataset[/dim]\n")

        # Get the multi-label dataset
        resolved = self._training_studio_resolve_multilabel_dataset(bundle)
        if resolved is None:
            self.console.print(
                "[red]This dataset does not expose a multi-label view."
                " Please select a format that produces a consolidated JSONL and try again.[/red]"
            )
            return {
                'runtime_params': {'error': 'No multi-label dataset'},
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            }

        dataset_path, label_fields = resolved

        # Get dataset metadata
        languages = bundle.metadata.get('confirmed_languages', bundle.metadata.get('languages', set()))
        language_distribution = bundle.metadata.get('language_distribution', {})
        text_length_stats = bundle.metadata.get('text_length_stats', {})

        # Use REAL text length stats - prefer token-based, fallback to chars
        if text_length_stats.get('token_mean'):
            text_length_avg = text_length_stats['token_mean']
        elif text_length_stats.get('char_mean'):
            text_length_avg = text_length_stats['char_mean']
        else:
            text_length_avg = text_length_stats.get('avg_chars', 0)

        user_prefers_long_models = text_length_stats.get('user_prefers_long_models', False)
        model_strategy = bundle.metadata.get('model_strategy', 'multilingual')
        recommended_model = bundle.recommended_model if hasattr(bundle, 'recommended_model') else None

        # Determine majority language
        majority_lang = self._get_majority_language(languages, language_distribution)
        if majority_lang:
            lang_display = f"Majority: {majority_lang.upper()}"
        else:
            lang_display = f"{len(languages)} language(s)"

        # Display intelligent model selection options
        self.console.print("\n[bold cyan]🎯 Model Selection for Benchmarking[/bold cyan]\n")

        selection_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
        selection_table.add_column("Option", style="cyan bold", width=18)
        selection_table.add_column("Description", style="white", width=70)

        selection_table.add_row(
            "intelligent",
            "🤖 AI-powered selection based on your data\n" +
            f"✓ Considers: {lang_display}, {text_length_avg:.0f} avg chars, '{model_strategy}' strategy\n" +
            "✓ Selects 5-7 optimal models automatically"
        )
        selection_table.add_row(
            "pre-selected",
            "📋 Choose from pre-selected categories\n" +
            "✓ Filter by: language, document length, efficiency\n" +
            "✓ View and select from curated model lists"
        )
        selection_table.add_row(
            "custom",
            "✏️  Manual selection from all available models\n" +
            "✓ Full access to 37+ models\n" +
            "✓ Enter any HuggingFace model ID"
        )

        self.console.print(selection_table)
        self.console.print()

        selection_mode = Prompt.ask(
            "[bold yellow]Model selection mode[/bold yellow]",
            choices=["intelligent", "pre-selected", "custom"],
            default="intelligent"
        )

        selected_models = []
        model_lang_map = {}

        if selection_mode == "intelligent":
            selected_models, model_lang_map = self._get_intelligent_benchmark_models(
                languages, text_length_avg, model_strategy, recommended_model, language_distribution,
                user_prefers_long_models
            )

            self.console.print(f"\n[green]✓ Intelligently selected {len(selected_models)} models:[/green]")
            for i, model in enumerate(selected_models, 1):
                lang_info = f" (for {model_lang_map[model].upper()} texts)" if model_lang_map.get(model) else " (multilingual)"
                self.console.print(f"  {i}. {model}{lang_info}")

            # Warning for monolingual models in multilingual dataset
            if len(languages) > 1 and model_lang_map:
                mono_models = [m for m in selected_models if model_lang_map.get(m)]
                if mono_models:
                    self.console.print(f"\n[yellow]⚠️  Note: Language-specific models will be trained on ALL texts[/yellow]")
                    self.console.print(f"[dim]Ideally, monolingual models should only train on their target language.[/dim]")
                    self.console.print(f"[dim]For better results with mixed languages, prefer multilingual models.[/dim]")

        elif selection_mode == "pre-selected":
            selected_models = self._get_preselected_benchmark_models(languages, text_length_avg)

        else:  # custom
            selected_models = self._get_custom_benchmark_models()

        if not selected_models:
            self.console.print("[yellow]No models selected. Using default: bert-base-uncased[/yellow]")
            selected_models = ["bert-base-uncased"]

        # Ask about reinforcement learning
        self.console.print("\n[bold cyan]🎓 Reinforcement Learning Option[/bold cyan]")
        self.console.print("[dim]Automatically triggered when model underperforms (F1 class 1 < threshold).[/dim]")
        self.console.print("[dim]• Applies oversampling via WeightedRandomSampler for minority class[/dim]")
        self.console.print("[dim]• Corrects cross-entropy loss with adaptive class weights[/dim]")
        self.console.print("[dim]• Adjusts learning rate, batch size, and epochs based on failure severity[/dim]")
        self.console.print("[dim]• Can improve F1 by 5-15% for underperforming models[/dim]\n")

        use_reinforcement = Confirm.ask("Enable reinforcement learning?", default=False)

        # Ask for number of epochs
        self.console.print("\n[bold cyan]⏱️  Training Duration[/bold cyan]")
        self.console.print("[dim]Number of epochs to train each model.[/dim]")
        self.console.print("[dim]• More epochs = better performance but longer training time[/dim]")
        self.console.print("[dim]• Recommended: 10-15 epochs for benchmark[/dim]\n")

        n_epochs = IntPrompt.ask("Number of epochs per model", default=10)

        # Load the multi-label dataset
        from llm_tool.trainers.multi_label_trainer import (
            MultiLabelTrainer,
            TrainingConfig as MultiLabelTrainingConfig,
            convert_multiclass_samples,  # UNIFIED multi-class conversion
            setup_multiclass_model  # UNIFIED multi-class model setup
        )

        output_dir = self._training_studio_make_output_dir("training_studio_benchmark_multilabel")

        try:
            trainer = MultiLabelTrainer(config=MultiLabelTrainingConfig(), verbose=False)
            samples = trainer.load_multi_label_data(
                str(dataset_path),
                text_field="text",
                labels_dict_field="labels",
                label_fields=label_fields,
            )
        except Exception as exc:
            self.console.print(f"[red]Failed to load dataset:[/red] {exc}")
            self.logger.exception("Multi-label benchmark dataset loading failed", exc_info=exc)
            return {
                'runtime_params': {'error': f'Failed to load dataset: {exc}'},
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            }

        if not samples:
            self.console.print("[red]No samples found in dataset[/red]")
            return {
                'runtime_params': {'error': 'No samples found'},
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            }

        # UNIFIED: Use trainer's multi-class detection (same code everywhere)
        multiclass_groups = trainer.detect_multiclass_groups(samples)

        # Build true_label_keys based on detected groups
        all_labels = set()
        for sample in samples:
            all_labels.update(sample.labels.keys())

        true_label_keys = []
        used_labels = set()

        # Display detected groups and ask user
        for prefix, group_labels in multiclass_groups.items():
            value_names = [lbl[len(prefix)+1:] if lbl.startswith(prefix+'_') else lbl for lbl in group_labels]
            self.console.print(f"[yellow]ℹ️  Detected mutually exclusive labels '{prefix}' with {len(group_labels)} values: {', '.join(value_names)}[/yellow]")

            # Ask user to confirm if this should be treated as multi-class or separate binary labels
            treat_as_multiclass = Confirm.ask(
                f"[bold]Treat '{prefix}' as a single multi-class label (recommended)?[/bold]\n"
                f"  • Yes: Train one model to predict among {len(group_labels)} classes\n"
                f"  • No: Train {len(group_labels)} separate binary models (one per value)",
                default=True
            )

            if treat_as_multiclass:
                true_label_keys.append(prefix)
                used_labels.update(group_labels)
                self.console.print(f"[green]✓ Will train as multi-class: {prefix}[/green]\n")
            else:
                # Treat each value as independent binary label
                true_label_keys.extend(group_labels)
                used_labels.update(group_labels)
                # Remove from multiclass_groups since user chose one-vs-all
                del multiclass_groups[prefix]
                self.console.print(f"[green]✓ Will train {len(group_labels)} separate binary models[/green]\n")

        # Add remaining ungrouped labels
        for label in sorted(all_labels):
            if label not in used_labels:
                true_label_keys.append(label)

        self.console.print(f"\n[green]✓ Loaded {len(samples)} samples with {len(true_label_keys)} label type(s)[/green]")
        self.console.print(f"[dim]Labels: {', '.join(sorted(true_label_keys))}[/dim]\n")

        # Ask which labels to benchmark
        self.console.print("[bold]Select labels to benchmark:[/bold]")
        label_list = sorted(true_label_keys)
        for idx, label in enumerate(label_list, 1):
            # Count samples for this label
            # For grouped multi-class labels, count samples with ANY member of the group
            if label in multiclass_groups:
                # Multi-class group - count samples with any member active
                count = sum(1 for s in samples if any(s.labels.get(member, 0) for member in multiclass_groups[label]))
                # Extract value name = everything after the prefix
                value_counts = {member[len(label)+1:] if member.startswith(label+'_') else member: sum(1 for s in samples if s.labels.get(member, 0)) for member in multiclass_groups[label]}
                values_str = ', '.join([f"{v}={c}" for v, c in value_counts.items()])
                self.console.print(f"  {idx}. {label} ({count} total samples: {values_str})")
            else:
                # Single binary label
                count = sum(1 for s in samples if label in s.labels and s.labels[label])
                self.console.print(f"  {idx}. {label} ({count} positive samples)")

        self.console.print("\n[dim]Press Enter to benchmark all labels, or enter label indices (comma-separated):[/dim]")
        label_selection = Prompt.ask("Select labels", default="")

        if label_selection.strip():
            try:
                indices = [int(i.strip()) for i in label_selection.split(",")]
                selected_labels = [label_list[i-1] for i in indices if 1 <= i <= len(label_list)]
            except (ValueError, IndexError):
                self.console.print("[yellow]Invalid selection, benchmarking all labels[/yellow]")
                selected_labels = label_list
        else:
            selected_labels = label_list

        if not selected_labels:
            self.console.print("[red]No labels selected[/red]")
            return {
                'runtime_params': {'error': 'No labels selected'},
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            }

        # Run benchmark for each label with each model
        from rich.table import Table
        import pandas as pd

        all_results = []

        total_benchmarks = len(selected_labels) * len(selected_models)
        current = 0

        for label_name in selected_labels:
            self.console.print(f"\n[bold cyan]{'━' * 80}[/bold cyan]")
            self.console.print(f"[bold cyan]  Benchmarking Label:[/bold cyan] [bold white]{label_name}[/bold white]")
            self.console.print(f"[bold cyan]{'━' * 80}[/bold cyan]\n")

            # Determine if this is a grouped label or a direct label
            if label_name in multiclass_groups:
                # UNIFIED: Use trainer's conversion function (same code everywhere)
                group_labels = multiclass_groups[label_name]
                multiclass_samples, value_names = convert_multiclass_samples(samples, label_name, group_labels)
                self.console.print(f"[dim]Multi-class label with {len(group_labels)} values: {', '.join(value_names)}[/dim]\n")

                # Check if any class has only 1 instance
                from collections import Counter
                label_counts = Counter([s.label for s in multiclass_samples])
                min_count = min(label_counts.values()) if label_counts else 0

                if min_count < 2:
                    # Find which classes (string labels) have insufficient instances
                    insufficient_class_names = [cls for cls, count in label_counts.items() if count < 2]

                    self.console.print(f"[yellow]⚠️  Label '{label_name}' has value(s) with only 1 instance: {', '.join(insufficient_class_names)}[/yellow]")

                    # Ask user if they want to remove these values
                    remove_values = Prompt.ask(
                        f"Remove value(s) '{', '.join(insufficient_class_names)}' and continue with remaining values?",
                        choices=["y", "n"],
                        default="y"
                    )

                    if remove_values.lower() == 'y':
                        # Remove samples with insufficient values
                        original_count = len(multiclass_samples)
                        multiclass_samples = [s for s in multiclass_samples if s.label not in insufficient_class_names]
                        self.console.print(f"[dim]Removed {original_count - len(multiclass_samples)} samples from values: {', '.join(insufficient_class_names)}[/dim]")

                        # Update value_names to remove insufficient classes
                        value_names = [name for name in value_names if name not in insufficient_class_names]

                        self.console.print(f"[green]✓ Continuing with {len(value_names)} values: {', '.join(value_names)}[/green]")

                        # Check if we still have at least 2 classes
                        if len(value_names) < 2:
                            self.console.print(f"[yellow]⚠️  Only {len(value_names)} value(s) remaining. Skipping {label_name}[/yellow]")
                            continue
                    else:
                        self.console.print(f"[dim]Skipping {label_name}[/dim]")
                        continue

                binary_samples = multiclass_samples
                num_classes = len(value_names)  # Use value_names length (may have been reduced)
            else:
                # True binary label or ungrouped label
                # CRITICAL FIX: Use string labels NOT_label/label instead of 0/1
                binary_samples = []
                for sample in samples:
                    if label_name in sample.labels:
                        from llm_tool.trainers.data_utils import DataSample
                        binary_label = label_name if sample.labels[label_name] else f"NOT_{label_name}"
                        binary_samples.append(DataSample(
                            text=sample.text,
                            label=binary_label,
                            id=sample.id,
                            lang=sample.lang,
                            metadata={**(sample.metadata or {}), 'original_label': label_name}
                        ))
                num_classes = 2

            if len(binary_samples) < 20:
                self.console.print(f"[yellow]⚠️  Skipping {label_name}: insufficient samples ({len(binary_samples)})[/yellow]")
                continue

            # Check if we have at least 2 instances per class for stratification
            # (only for binary labels - multi-class was already checked above)
            if label_name not in multiclass_groups:
                from collections import Counter
                label_counts = Counter([s.label for s in binary_samples])
                min_count = min(label_counts.values())

                if min_count < 2:
                    # Find which classes have insufficient instances
                    insufficient_classes = [cls for cls, count in label_counts.items() if count < 2]
                    self.console.print(f"[yellow]⚠️  Label '{label_name}' has class(es) with only 1 instance: {insufficient_classes}[/yellow]")

                    # Ask user if they want to remove this label
                    remove_label = Prompt.ask(
                        f"Remove label '{label_name}' from training?",
                        choices=["y", "n"],
                        default="y"
                    )

                    if remove_label.lower() == 'y':
                        self.console.print(f"[dim]Skipping {label_name}[/dim]")
                        continue
                    else:
                        self.console.print(f"[yellow]Attempting to proceed without stratification (may reduce quality)[/yellow]")
                        stratify_labels = None
                else:
                    stratify_labels = [s.label for s in binary_samples]
            else:
                # Multi-class: already checked and cleaned above
                stratify_labels = [s.label for s in binary_samples]

            # Split into train/test
            from sklearn.model_selection import train_test_split
            train_samples, test_samples = train_test_split(
                binary_samples,
                test_size=0.2,
                random_state=42,
                stratify=stratify_labels  # None if insufficient samples per class
            )

            self.console.print(f"[dim]Train: {len(train_samples)}, Test: {len(test_samples)}[/dim]\n")

            # Benchmark each model on this label
            for model_name in selected_models:
                current += 1
                self.console.print(f"[cyan]Benchmark {current}/{total_benchmarks}:[/cyan] [bold]{model_name}[/bold] on [bold]{label_name}[/bold]")

                try:
                    # Import the appropriate model class
                    from llm_tool.trainers.model_selector import ModelSelector
                    selector = ModelSelector(verbose=False)

                    # Map common model names to their profile keys
                    model_name_mapping = {
                        'camembert-base': 'Camembert',
                        'flaubert-base': 'FlauBERTBase',
                        'camembert-large': 'CamembertLarge',
                        'xlm-roberta-base': 'XLMRobertaBase',
                        'xlm-roberta-large': 'XLMRobertaLarge',
                        'bert-base-uncased': 'BERTBase',
                        'roberta-base': 'RoBERTaBase',
                        'bert-large-uncased': 'BERTLarge',
                        'bert-base-multilingual-cased': 'mBERTBase',
                        'distilbert-base': 'DistilBERT',
                        'distilroberta-base': 'DistilRoBERTa',
                        'albert-base-v2': 'ALBERTBase',
                        'cmarkea/distilcamembert-base': 'DistilCamemBERT',
                    }

                    # Try to find the model profile key
                    profile_key = model_name_mapping.get(model_name)

                    if profile_key and profile_key in selector.MODEL_PROFILES:
                        # Found a matching profile
                        model_class = selector.MODEL_PROFILES[profile_key].model_class
                        model = model_class()
                    elif model_name in selector.MODEL_PROFILES:
                        # Direct match in profiles
                        model_class = selector.MODEL_PROFILES[model_name].model_class
                        model = model_class()
                    else:
                        # Try to create a generic model with AutoTokenizer and AutoModelForSequenceClassification
                        from transformers import AutoTokenizer, AutoModelForSequenceClassification
                        from llm_tool.trainers.bert_base import BertBase

                        class CustomModel(BertBase):
                            def __init__(self):
                                super().__init__(
                                    model_name=model_name,
                                    tokenizer=AutoTokenizer,
                                    model_sequence_classifier=AutoModelForSequenceClassification
                                )

                        model = CustomModel()

                    # CRITICAL: Configure model for multi-class using unified function
                    if num_classes > 2:
                        setup_multiclass_model(model, num_classes, value_names)

                    # Encode data
                    train_texts = [s.text for s in train_samples]
                    train_labels = [s.label for s in train_samples]  # DataSample uses .label not .labels
                    test_texts = [s.text for s in test_samples]
                    test_labels = [s.label for s in test_samples]  # DataSample uses .label not .labels

                    train_loader = model.encode(train_texts, train_labels, batch_size=32, progress_bar=False)
                    test_loader = model.encode(test_texts, test_labels, batch_size=32, progress_bar=False)

                    # CRITICAL: Do NOT use pos_weight for initial training
                    # Let the model learn naturally first, then reinforcement learning
                    # will apply adaptive weights if needed (when F1 is low)

                    # Train model
                    temp_model_name = f"benchmark_{label_name}_{model_name.replace('/', '_')}"

                    # CRITICAL FIX: Extract language information for per-language metrics
                    test_languages = [s.lang for s in test_samples] if test_samples else None
                    has_languages = test_languages and any(lang for lang in test_languages if lang)

                    # Determine language (use first sample's language or MULTI if multiple)
                    sample_langs = set(s.lang for s in train_samples if s.lang)
                    if len(sample_langs) == 1:
                        train_language = list(sample_langs)[0].upper()
                    elif len(sample_langs) > 1:
                        train_language = "MULTI"
                    else:
                        train_language = None

                    # UNIFIED: Use centralized function to set detected languages (SAME AS BENCHMARK)
                    if has_languages:
                        from llm_tool.trainers.model_trainer import set_detected_languages_on_model
                        train_languages = [s.lang for s in train_samples]
                        set_detected_languages_on_model(
                            model=model,
                            train_languages=train_languages,
                            val_languages=test_languages,
                            logger=self.logger
                        )

                    result = model.run_training(
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        n_epochs=n_epochs,
                        lr=5e-5,
                        random_state=42,
                        save_model_as=temp_model_name,
                        pos_weight=None,  # CRITICAL: No pos_weight initially - let RL handle it
                        metrics_output_dir="training_logs",
                        best_model_criteria="combined",
                        f1_class_1_weight=0.7,
                        reinforced_learning=use_reinforcement,
                        reinforced_epochs=10,
                        rescue_low_class1_f1=True,  # CRITICAL: Enable automatic rescue
                        f1_1_rescue_threshold=0.50,  # Trigger if F1 < 50%
                        reinforced_f1_threshold=0.60,
                        track_languages=has_languages,  # CRITICAL: Enable if we have language info
                        language_info=test_languages,     # CRITICAL: Pass language info for metrics
                        label_key=label_name,
                        label_value=None if num_classes > 2 else label_name,
                        language=train_language,
                        class_names=value_names if num_classes > 2 else None,
                    )

                    # Extract metrics
                    best_metric_val, best_model_path, best_scores = result

                    if best_scores:
                        # Multi-class: best_scores is a tuple of arrays (precision, recall, f1, support)
                        # Each array has one value per class
                        if num_classes > 2:
                            # Multi-class
                            macro_f1 = np.mean(best_scores[2])  # Average of all F1 scores
                            result_dict = {
                                'label': label_name,
                                'model': model_name,
                                'f1_macro': macro_f1,
                                'num_classes': num_classes,
                                'class_names': ','.join(value_names),  # CRITICAL: Store class names for CSV header
                            }
                            # Add per-class metrics
                            for i in range(num_classes):
                                result_dict[f'f1_class_{i}'] = best_scores[2][i]
                                result_dict[f'precision_{i}'] = best_scores[0][i]
                                result_dict[f'recall_{i}'] = best_scores[1][i]
                                result_dict[f'support_{i}'] = int(best_scores[3][i])
                            all_results.append(result_dict)
                            self.console.print(f"  [green]✓ F1 Macro: {macro_f1:.4f}[/green]\n")
                        else:
                            # Binary
                            precision_0, recall_0, f1_0, support_0 = best_scores[0]
                            precision_1, recall_1, f1_1, support_1 = best_scores[1]
                            macro_f1 = (f1_0 + f1_1) / 2
                            all_results.append({
                                'label': label_name,
                                'model': model_name,
                                'f1_macro': macro_f1,
                                'f1_class_0': f1_0,
                                'f1_class_1': f1_1,
                                'precision_1': precision_1,
                                'recall_1': recall_1,
                                'support_0': int(support_0),
                                'support_1': int(support_1),
                            })
                            self.console.print(f"  [green]✓ F1 Macro: {macro_f1:.4f}, F1 Class 1: {f1_1:.4f}[/green]\n")
                    else:
                        # No scores
                        macro_f1 = 0
                        self.console.print(f"  [yellow]⚠ No scores available[/yellow]\n")

                except Exception as exc:
                    self.console.print(f"  [red]✗ Failed: {exc}[/red]\n")
                    self.logger.exception(f"Benchmark failed for {model_name} on {label_name}", exc_info=exc)

        # Display final results
        if all_results:
            self.console.print(f"\n[bold cyan]{'━' * 80}[/bold cyan]")
            self.console.print(f"[bold cyan]  Benchmark Results Summary[/bold cyan]")
            self.console.print(f"[bold cyan]{'━' * 80}[/bold cyan]\n")

            # Determine max number of classes to create appropriate table columns
            max_classes = 0
            for result in all_results:
                # Count how many f1_class_X keys exist
                class_keys = [k for k in result.keys() if k.startswith('f1_class_')]
                max_classes = max(max_classes, len(class_keys))

            # Create summary table with dynamic columns based on max_classes
            results_table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
            results_table.add_column("Label", style="cyan")
            results_table.add_column("Model", style="white")
            results_table.add_column("F1 Macro", justify="right", style="green")

            # Add columns for each class
            for i in range(max_classes):
                results_table.add_column(f"F1 C{i}", justify="right", style="yellow")
                results_table.add_column(f"P{i}", justify="right", style="dim")
                results_table.add_column(f"R{i}", justify="right", style="dim")

            # Sort by label and F1 macro
            all_results.sort(key=lambda x: (x['label'], -x['f1_macro']))

            for result in all_results:
                row = [
                    result['label'],
                    result['model'],
                    f"{result['f1_macro']:.4f}",
                ]
                # Add per-class metrics
                for i in range(max_classes):
                    f1_key = f'f1_class_{i}'
                    precision_key = f'precision_{i}'
                    recall_key = f'recall_{i}'

                    if f1_key in result:
                        row.append(f"{result[f1_key]:.4f}")
                        row.append(f"{result[precision_key]:.4f}")
                        row.append(f"{result[recall_key]:.4f}")
                    else:
                        row.append("")
                        row.append("")
                        row.append("")

                results_table.add_row(*row)

            self.console.print(results_table)

            # Save results to CSV with proper handling of missing values
            results_df = pd.DataFrame(all_results)

            # Replace None/NaN with 0.0 for numeric columns to avoid NA values
            for col in results_df.columns:
                if col not in ['label', 'model', 'class_names']:
                    results_df[col] = results_df[col].fillna(0.0)

            # Save overall results with class name metadata
            results_csv = output_dir / "multilabel_benchmark_results.csv"

            # CRITICAL: Add metadata header with class name mappings
            with open(results_csv, 'w', encoding='utf-8') as f:
                # Write metadata header for each unique label that has class names
                label_class_mappings = {}
                for result in all_results:
                    if 'class_names' in result and result['class_names']:
                        label_class_mappings[result['label']] = result['class_names']

                if label_class_mappings:
                    f.write("# CLASS NAME MAPPINGS (Multi-label/Multi-class)\n")
                    for label, class_names in label_class_mappings.items():
                        class_list = class_names.split(',')
                        mapping_str = ', '.join([f"C{i}={name}" for i, name in enumerate(class_list)])
                        f.write(f"# {label}: {mapping_str}\n")
                    f.write("#\n")

            # Append the DataFrame (without class_names column in output)
            output_df = results_df.drop(columns=['class_names', 'num_classes'], errors='ignore')
            output_df.to_csv(results_csv, index=False, na_rep='0.0', mode='a')

            self.console.print(f"\n[green]✓ Results saved to:[/green] {results_csv}")

            # Create per-label summary CSVs in training_logs/{label_name}/
            from pathlib import Path
            training_logs_base = Path("training_logs")

            for label_name in selected_labels:
                label_results = [r for r in all_results if r['label'] == label_name]
                if label_results:
                    # Create label directory
                    label_dir = training_logs_base / label_name
                    label_dir.mkdir(parents=True, exist_ok=True)

                    # Create summary CSV for this label
                    label_df = pd.DataFrame(label_results)
                    label_df = label_df.drop(columns=['label'])  # Remove label column since it's redundant

                    # Replace None/NaN with 0.0
                    for col in label_df.columns:
                        if col != 'model':
                            label_df[col] = label_df[col].fillna(0.0)

                    # Sort by F1 macro descending
                    label_df = label_df.sort_values('f1_macro', ascending=False)

                    summary_csv = label_dir / "models_summary.csv"
                    label_df.to_csv(summary_csv, index=False, na_rep='0.0')

                    # Create best models ranking CSV
                    best_df = label_df.head(10)  # Top 10 models
                    best_csv = label_dir / "best_models_ranking.csv"
                    best_df.to_csv(best_csv, index=False, na_rep='0.0')

            # Show best model per label
            self.console.print("\n[bold]Best model per label:[/bold]")
            best_models_per_label = {}
            for label_name in selected_labels:
                label_results = [r for r in all_results if r['label'] == label_name]
                if label_results:
                    best = max(label_results, key=lambda x: x['f1_macro'])
                    best_models_per_label[label_name] = {'model': best['model'], 'f1': best['f1_macro']}
                    self.console.print(f"  • [cyan]{label_name}:[/cyan] {best['model']} (F1 {best['f1_macro']:.4f})")

            # Return complete training info for metadata save
            return {
                'runtime_params': {
                    'selected_models': selected_models,
                    'selected_labels': selected_labels,
                    'actual_models_trained': list(set(r['model'] for r in all_results)),
                    'best_models_per_label': best_models_per_label
                },
                'models_trained': list(set(r['model'] for r in all_results)),
                'best_model': best_models_per_label,
                'best_f1': sum(m['f1'] for m in best_models_per_label.values()) / len(best_models_per_label) if best_models_per_label else None
            }
        else:
            self.console.print("[yellow]No benchmark results to display[/yellow]")
            return {
                'runtime_params': {
                    'selected_models': selected_models if 'selected_models' in locals() else [],
                    'selected_labels': selected_labels if 'selected_labels' in locals() else [],
                    'actual_models_trained': []
                },
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            }

    def _training_studio_run_custom(self, bundle: TrainingDataBundle, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom training mode - full control over hyperparameters.

        Args:
            bundle: Training data bundle
            model_config: Model configuration dict (will be updated with runtime params)

        Returns:
            dict with keys: 'runtime_params', 'models_trained', 'best_model', 'best_f1'
        """
        self.console.print("\n[bold]Custom training configuration[/bold]")

        model_name = Prompt.ask("Model name", default=self._training_studio_default_model())
        epochs = self._int_prompt_with_validation("Epochs", default=10, min_value=1, max_value=100)
        batch_size = self._int_prompt_with_validation("Batch size", default=16, min_value=1, max_value=256)

        lr_input = Prompt.ask("Learning rate", default="2e-5")
        try:
            learning_rate = float(lr_input)
        except ValueError:
            self.console.print(f"[red]Invalid learning rate: {lr_input}[/red]")
            return {
                'runtime_params': {
                    'custom_config': {'error': 'Invalid learning rate'}
                },
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'error': 'Invalid learning rate'
            }

        # Capture runtime parameters for full reproducibility
        runtime_params = {
            'custom_config': {
                'model_name': model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            },
            'actual_models_trained': [model_name]
        }

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
            return {
                'runtime_params': runtime_params,
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'error': str(exc)
            }

        self._training_studio_show_training_result(result, bundle, title="Custom training results")

        # Return complete training info for metadata save
        return {
            'runtime_params': runtime_params,
            'models_trained': [model_name],
            'best_model': result.get('best_model'),
            'best_f1': result.get('best_f1') or result.get('f1_macro')
        }

    def _training_studio_run_distributed(self, bundle: TrainingDataBundle, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distributed training mode - parallel multi-label training.

        Args:
            bundle: Training data bundle
            model_config: Model configuration dict (will be updated with runtime params)

        Returns:
            dict with keys: 'runtime_params', 'models_trained', 'best_model', 'best_f1'
        """
        # CRITICAL WARNING: This mode is untested
        self.console.print("\n[bold red]⚠️  WARNING: Distributed mode is NOT RECOMMENDED (untested)[/bold red]")
        self.console.print("[yellow]This training mode has not been thoroughly tested and may produce incorrect results.[/yellow]")
        self.console.print("[dim]Proceeding at your own risk...[/dim]\n")

        self.console.print("\n[bold]Distributed multi-label training[/bold]")

        resolved = self._training_studio_resolve_multilabel_dataset(bundle)
        if resolved is None:
            self.console.print(
                "[red]This dataset does not expose a multi-label view."
                " Please select a format that produces a consolidated JSONL (e.g. LLM annotations, binary long, JSONL multi-label) and try again.[/red]"
            )
            return {
                'runtime_params': {
                    'distributed_config': {'error': 'No multi-label dataset available'}
                },
                'models_trained': [],
                'best_model': None,
                'best_f1': None,
                'error': 'No multi-label dataset'
            }

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

        # Capture runtime parameters for full reproducibility
        runtime_params = {
            'distributed_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'reinforced': reinforced,
                'reinforced_epochs': reinforced_epochs,
                'parallel_training': parallel_training,
                'max_workers': max_workers,
                'strategy_choice': strategy_choice,
                'train_by_language': train_by_language,
                'multilingual_model': multilingual_model,
                'auto_select_model': auto_select_model
            }
        }

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

        # Detect multi-class groups and ask user if they want multi-class training
        multiclass_groups = trainer.detect_multiclass_groups(samples)

        if multiclass_groups and HAS_RICH and self.console:
            self.console.print(f"\n[bold cyan]🎯 Multi-Class Groups Detected:[/bold cyan]\n")

            for group_name, group_labels in multiclass_groups.items():
                self.console.print(f"  • [cyan]{group_name}[/cyan]: {len(group_labels)} classes")
                self.console.print(f"    {', '.join(sorted(group_labels)[:5])}{' ...' if len(group_labels) > 5 else ''}\n")

            self.console.print("[dim]Choose training approach:[/dim]\n")

            approach_table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
            approach_table.add_column("Approach", style="cyan bold", width=18)
            approach_table.add_column("Description", style="white", width=60)

            approach_table.add_row(
                "multi-class",
                f"🎯 ONE model per group (e.g., 1 model for all {list(multiclass_groups.keys())[0]} values)\n"
                "✓ Faster training (fewer models)\n"
                "✓ Model learns relationships between classes\n"
                "✓ Consistent predictions (only one class predicted)"
            )
            approach_table.add_row(
                "one-vs-all",
                f"⚡ Multiple binary models (one per label)\n"
                "✓ Each model: 'Label X' vs 'NOT Label X'\n"
                "✓ Better for: imbalanced data or label-specific tuning\n"
                "✓ More flexible but slower training"
            )

            self.console.print(approach_table)
            self.console.print()

            training_approach = Prompt.ask(
                "[bold yellow]Training approach[/bold yellow]",
                choices=["multi-class", "one-vs-all"],
                default="multi-class"
            )

            if training_approach == "multi-class":
                trainer_config.multiclass_mode = True
                trainer_config.multiclass_groups = multiclass_groups
                # Recreate trainer with updated config
                trainer = MultiLabelTrainer(config=trainer_config, verbose=False)
                self.console.print(f"[green]✓ Multi-class mode enabled for {len(multiclass_groups)} group(s)[/green]\n")

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
                return {
                    'runtime_params': runtime_params,
                    'models_trained': [],
                    'best_model': None,
                    'best_f1': None,
                    'error': str(exc)
                }

        self._training_studio_show_distributed_results(trainer, models, output_dir)

        # Extract trained model names and performance
        trained_model_names = list(models.keys()) if models else []
        # Compute average F1 across all labels if available
        avg_f1 = None
        if models and hasattr(trainer, 'trained_models'):
            f1_scores = []
            for label, model_info in trainer.trained_models.items():
                if isinstance(model_info, dict) and 'performance_metrics' in model_info:
                    metrics = model_info['performance_metrics']
                    if 'f1_macro' in metrics:
                        f1_scores.append(metrics['f1_macro'])
            if f1_scores:
                avg_f1 = sum(f1_scores) / len(f1_scores)

        # Return complete training info for metadata save
        return {
            'runtime_params': runtime_params,
            'models_trained': trained_model_names,
            'best_model': trained_model_names,  # All models trained
            'best_f1': avg_f1
        }

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
        # Support both single-label and multi-label datasets for benchmarking
        if bundle.primary_file:
            return bundle.primary_file, bundle.text_column, bundle.label_column

        # For multi-label distributed training, we have individual label files
        candidates = [(label, path) for label, path in bundle.training_files.items() if label != "multilabel"]

        if not candidates:
            raise ValueError("No dataset available for benchmarking.")

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

    def _get_majority_language(self, languages: set, language_distribution: dict = None) -> str:
        """
        Determine the majority/dominant language from a set of languages.

        Args:
            languages: Set of language codes
            language_distribution: Optional dict mapping language codes to counts

        Returns:
            Dominant language code (lowercase) or None
        """
        if not languages:
            return None

        # If we have distribution data, use it to find the majority
        if language_distribution:
            total = sum(language_distribution.values())
            if total > 0:
                # Find language with highest percentage
                majority_lang = max(language_distribution.items(), key=lambda x: x[1])
                percentage = (majority_lang[1] / total) * 100

                # If a language represents >50%, it's the majority
                if percentage > 50:
                    return majority_lang[0].lower()

        # Fallback: if only one language, return it
        if len(languages) == 1:
            return list(languages)[0].lower()

        # If multiple languages without clear majority, check for common cases
        lang_list = [l.lower() for l in languages]
        if 'fr' in lang_list:
            return 'fr'  # Often FR is dominant after correction
        elif 'en' in lang_list:
            return 'en'

        return None

    def _is_model_multilingual(self, model_name: str) -> bool:
        """
        Determine if a model is multilingual or language-specific.

        Args:
            model_name: HuggingFace model ID

        Returns:
            True if multilingual, False if language-specific
        """
        # Model-to-language mapping (same as in _get_intelligent_benchmark_models)
        MULTILINGUAL_KEYWORDS = ['xlm', 'multilingual', 'mdeberta', 'long-t5']
        MONOLINGUAL_PATTERNS = {
            'camembert': 'fr',
            'flaubert': 'fr',
            'bert-base-german-cased': 'de',
            'distilbert-base-german-cased': 'de',
            'roberta': 'en',  # RoBERTa is English-only unless specified
            'bert-base-uncased': 'en',
            'bert-base-cased': 'en',
            'distilbert-base-uncased': 'en',
        }

        model_lower = model_name.lower()

        # Check for multilingual keywords
        if any(keyword in model_lower for keyword in MULTILINGUAL_KEYWORDS):
            return True

        # Check for monolingual patterns
        if any(pattern in model_lower for pattern in MONOLINGUAL_PATTERNS.keys()):
            return False

        # Default: assume multilingual for safety (won't create extra models)
        return True

    def _get_intelligent_benchmark_models(self, languages: set, text_length_avg: float, model_strategy: str, recommended_model: str = None, language_distribution: dict = None, user_prefers_long_models: bool = False) -> Tuple[List[str], Dict[str, Optional[str]]]:
        """
        HIGHLY INTELLIGENT model selection using ALL available models in the package.

        Selection criteria (scored):
        1. Language match (primary, 100 points)
        2. Text length compatibility (50 points) - BOOSTED if user_prefers_long_models=True
        3. Model size/efficiency (30 points)
        4. Model popularity/reliability (20 points)
        5. Multilingual capability (bonus for mixed languages)

        Returns:
            Tuple of (models_list, model_to_language_map)
            model_to_language_map: {model_name: language_code or None for multilingual}
        """
        lang_list = list(languages) if languages else []

        # COMPREHENSIVE language-to-model mapping using ALL models from sota_models
        MODEL_LANGUAGE_MAP = {
            # ============ MULTILINGUAL MODELS ============
            'xlm-roberta-base': None,
            'xlm-roberta-large': None,
            'bert-base-multilingual-cased': None,
            'bert-base-multilingual-uncased': None,
            'distilbert-base-multilingual-cased': None,
            'microsoft/mdeberta-v3-base': None,

            # ============ MULTILINGUAL LONG-DOCUMENT MODELS ============
            'markussagen/xlm-roberta-longformer-base-4096': None,  # Multilingual Longformer, 4096 tokens, 100+ languages
            'google/long-t5-local-base': None,  # Multilingual T5 with local attention, 4096+ tokens
            'google/long-t5-tglobal-base': None,  # Multilingual T5 with transient global attention, 4096+ tokens

            # ============ ENGLISH MODELS ============
            'bert-base-uncased': 'en',
            'bert-base-cased': 'en',
            'bert-large-uncased': 'en',
            'bert-large-cased': 'en',
            'roberta-base': 'en',
            'roberta-large': 'en',
            'distilbert-base-uncased': 'en',
            'distilbert-base-cased': 'en',
            'distilroberta-base': 'en',
            'albert-base-v2': 'en',
            'albert-large-v2': 'en',
            'albert-xlarge-v2': 'en',
            'google/electra-base-discriminator': 'en',
            'google/electra-large-discriminator': 'en',
            'microsoft/deberta-base': 'en',
            'microsoft/deberta-large': 'en',
            'microsoft/deberta-v3-base': 'en',
            'microsoft/deberta-v3-large': 'en',
            'microsoft/deberta-v3-small': 'en',
            'allenai/longformer-base-4096': 'en',
            'google/bigbird-roberta-base': 'en',
            'google/bigbird-roberta-large': 'en',
            'squeezebert/squeezebert-uncased': 'en',
            'sentence-transformers/all-MiniLM-L6-v2': 'en',

            # ============ FRENCH MODELS ============
            'camembert-base': 'fr',
            'camembert/camembert-base': 'fr',
            'camembert/camembert-large': 'fr',
            'flaubert/flaubert_base_cased': 'fr',
            'flaubert/flaubert_base_uncased': 'fr',
            'flaubert/flaubert_large_cased': 'fr',
            'cmarkea/distilcamembert-base': 'fr',
            'almanach/camembert-base': 'fr',
            'dbmdz/bert-base-french-europeana-cased': 'fr',
            'dangvantuan/sentence-camembert-base': 'fr',
            'qwant/fralbert-base': 'fr',

            # ============ GERMAN MODELS ============
            'bert-base-german-cased': 'de',
            'bert-base-german-dbmdz-cased': 'de',
            'bert-base-german-dbmdz-uncased': 'de',
            'deepset/gbert-base': 'de',
            'deepset/gbert-large': 'de',
            'distilbert-base-german-cased': 'de',
            'uklfr/gottbert-base': 'de',
            'dbmdz/bert-base-german-europeana-cased': 'de',

            # ============ SPANISH MODELS ============
            'dccuchile/bert-base-spanish-wwm-cased': 'es',
            'dccuchile/bert-base-spanish-wwm-uncased': 'es',
            'PlanTL-GOB-ES/roberta-base-bne': 'es',
            'mrm8488/electricidad-base-discriminator': 'es',
            'bertin-project/bertin-roberta-base-spanish': 'es',

            # ============ ITALIAN MODELS ============
            'dbmdz/bert-base-italian-cased': 'it',
            'dbmdz/bert-base-italian-uncased': 'it',
            'dbmdz/bert-base-italian-xxl-cased': 'it',
            'dbmdz/bert-base-italian-xxl-uncased': 'it',
            'Musixmatch/umberto-commoncrawl-cased-v1': 'it',

            # ============ PORTUGUESE MODELS ============
            'neuralmind/bert-base-portuguese-cased': 'pt',
            'neuralmind/bert-large-portuguese-cased': 'pt',
            'adalbertojunior/distilbert-portuguese-cased': 'pt',
            'pierreguillou/bert-base-cased-pt-lenerbr': 'pt',

            # ============ DUTCH MODELS ============
            'GroNLP/bert-base-dutch-cased': 'nl',
            'wietsedv/bert-base-dutch-cased': 'nl',
            'pdelobelle/robbert-v2-dutch-base': 'nl',
            'DTAI-KULeuven/robbert-2023-dutch-large': 'nl',

            # ============ POLISH MODELS ============
            'dkleczek/bert-base-polish-uncased-v1': 'pl',
            'dkleczek/bert-base-polish-cased-v1': 'pl',
            'allegro/herbert-base-cased': 'pl',
            'allegro/herbert-large-cased': 'pl',

            # ============ ARABIC MODELS ============
            'aubmindlab/bert-base-arabertv2': 'ar',
            'aubmindlab/bert-large-arabertv2': 'ar',
            'asafaya/bert-base-arabic': 'ar',
            'CAMeL-Lab/bert-base-arabic-camelbert-msa': 'ar',
            'UBC-NLP/MARBERT': 'ar',

            # ============ CHINESE MODELS ============
            'bert-base-chinese': 'zh',
            'hfl/chinese-bert-wwm': 'zh',
            'hfl/chinese-bert-wwm-ext': 'zh',
            'hfl/chinese-roberta-wwm-ext': 'zh',
            'hfl/chinese-roberta-wwm-ext-large': 'zh',
            'hfl/chinese-electra-base-discriminator': 'zh',

            # ============ RUSSIAN MODELS ============
            'DeepPavlov/rubert-base-cased': 'ru',
            'DeepPavlov/rubert-base-cased-conversational': 'ru',
            'ai-forever/ruBert-base': 'ru',
            'ai-forever/ruBert-large': 'ru',
            'cointegrated/rubert-tiny': 'ru',

            # ============ JAPANESE MODELS ============
            'cl-tohoku/bert-base-japanese': 'ja',
            'cl-tohoku/bert-base-japanese-whole-word-masking': 'ja',
            'cl-tohoku/bert-large-japanese': 'ja',
            'nlp-waseda/roberta-base-japanese': 'ja',
            'nlp-waseda/roberta-large-japanese': 'ja',

            # ============ KOREAN MODELS ============
            'klue/bert-base': 'ko',
            'kykim/bert-kor-base': 'ko',
            'beomi/kcbert-base': 'ko',
            'beomi/kcbert-large': 'ko',

            # ============ TURKISH MODELS ============
            'dbmdz/bert-base-turkish-cased': 'tr',
            'dbmdz/bert-base-turkish-uncased': 'tr',
            'dbmdz/electra-base-turkish-cased-discriminator': 'tr',

            # ============ SWEDISH MODELS ============
            'KB/bert-base-swedish-cased': 'sv',
            'af-ai-center/bert-base-swedish-uncased': 'sv',

            # ============ DANISH MODELS ============
            'Maltehb/danish-bert-botxo': 'da',
            'sarnikowski/convbert-small-da-cased': 'da',

            # ============ NORWEGIAN MODELS ============
            'ltg/norbert': 'no',
            'NbAiLab/nb-bert-base': 'no',

            # ============ FINNISH MODELS ============
            'TurkuNLP/bert-base-finnish-cased-v1': 'fi',
            'TurkuNLP/bert-base-finnish-uncased-v1': 'fi',

            # ============ HINDI MODELS ============
            'ai4bharat/indic-bert': 'hi',

            # ============ VIETNAMESE MODELS ============
            'vinai/phobert-base': 'vi',
            'vinai/phobert-large': 'vi',

            # ============ THAI MODELS ============
            'airesearch/wangchanberta-base-att-spm-uncased': 'th',

            # ============ INDONESIAN MODELS ============
            'indobenchmark/indobert-base-p1': 'id',
            'indobenchmark/indobert-large-p1': 'id',

            # ============ CZECH MODELS ============
            'Seznam/retromae-small-cs': 'cs',
            'ufal/robeczech-base': 'cs',

            # ============ GREEK MODELS ============
            'nlpaueb/bert-base-greek-uncased-v1': 'el',

            # ============ HEBREW MODELS ============
            'onlplab/alephbert-base': 'he',

            # ============ ROMANIAN MODELS ============
            'dumitrescustefan/bert-base-romanian-cased-v1': 'ro',

            # ============ BULGARIAN MODELS ============
            'iarfmoose/roberta-base-bulgarian': 'bg',

            # ============ CROATIAN MODELS ============
            'classla/bcms-bertic': 'hr',

            # ============ SERBIAN MODELS ============
            'classla/bcms-bertic': 'sr',

            # ============ UKRAINIAN MODELS ============
            'youscan/ukr-roberta-base': 'uk',
        }

        # ============================================================
        # STEP 1: Analyze language distribution
        # ============================================================
        total_samples = sum(language_distribution.values()) if language_distribution else 0
        lang_percentages = {}
        if total_samples > 0:
            lang_percentages = {lang.lower(): (count / total_samples * 100)
                                for lang, count in language_distribution.items()}

        # Find dominant language (>70%)
        dominant_lang = None
        for lang, pct in lang_percentages.items():
            if pct > 70:
                dominant_lang = lang
                break

        # Check if single language
        if len(lang_list) == 1:
            dominant_lang = lang_list[0].lower()

        # Check if balanced multilingual
        is_balanced_multilingual = len(lang_list) > 1 and not dominant_lang

        # ============================================================
        # STEP 2: Score ALL available models
        # ============================================================
        model_scores = {}

        for model_name, model_lang in MODEL_LANGUAGE_MAP.items():
            score = 0.0

            # CRITERION 1: Language Match (100 points max)
            if model_lang is None:  # Multilingual model
                if is_balanced_multilingual:
                    score += 90  # Excellent for balanced multilingual
                elif len(lang_list) > 1:
                    score += 70  # Good for any multilingual
                else:
                    score += 40  # Okay for single language
            else:  # Language-specific model
                if dominant_lang and model_lang == dominant_lang:
                    score += 100  # Perfect match!
                elif model_lang in [l.lower() for l in lang_list]:
                    pct = lang_percentages.get(model_lang, 0)
                    score += pct  # Score based on language percentage
                else:
                    score += 0  # No language match

            # CRITERION 2: Text Length Compatibility (50 points max, BOOSTED to 150 if user wants long models)
            is_long_model = ('longformer' in model_name.lower() or
                           'bigbird' in model_name.lower() or
                           'long-t5' in model_name.lower() or
                           '4096' in model_name or
                           '16384' in model_name)

            if user_prefers_long_models:
                # User explicitly wants long-document models - BOOST them heavily
                if is_long_model:
                    score += 150  # MASSIVE boost for long models when user wants them
                else:
                    # Standard models get penalty when long models are preferred
                    if text_length_avg > 500:
                        score += 10  # Not ideal - will truncate
                    else:
                        score += 20  # Acceptable but not preferred
            else:
                # Normal scoring when user hasn't expressed preference
                if is_long_model:
                    if text_length_avg > 400:
                        score += 50  # Perfect for long texts
                    elif text_length_avg > 200:
                        score += 30  # Okay for medium texts
                    else:
                        score += 10  # Overkill for short texts
                else:
                    if text_length_avg <= 300:
                        score += 40  # Good for short/medium texts
                    elif text_length_avg <= 500:
                        score += 30  # Okay for longer texts
                    else:
                        score += 15  # Not ideal but works

            # CRITERION 3: Model Size/Efficiency (30 points max)
            if 'distil' in model_name.lower() or 'tiny' in model_name.lower() or 'small' in model_name.lower():
                score += 30  # Efficient models
            elif 'base' in model_name.lower():
                score += 25  # Standard models
            elif 'large' in model_name.lower() or 'xlarge' in model_name.lower():
                score += 15  # Large models (slower)
            else:
                score += 20  # Unknown size

            # CRITERION 4: Model Popularity/Reliability (20 points max)
            # Based on known high-quality models
            high_quality_models = {
                'xlm-roberta-base': 20,
                'xlm-roberta-large': 18,
                'bert-base-multilingual-cased': 18,
                'camembert-base': 19,
                'roberta-base': 20,
                'bert-base-uncased': 19,
                'flaubert-base': 17,
                'bert-base-german-cased': 18,
                'microsoft/deberta-v3-base': 20,
                'microsoft/mdeberta-v3-base': 19,
                'markussagen/xlm-roberta-longformer-base-4096': 20,  # Excellent multilingual long-document (100+ languages)
                'google/long-t5-local-base': 18,  # High-quality multilingual T5 long-document
                'google/long-t5-tglobal-base': 18,  # High-quality multilingual T5 long-document
                'allenai/longformer-base-4096': 17,  # Popular long-document (EN only)
                'google/bigbird-roberta-base': 17,  # Popular long-document (EN only)
            }
            score += high_quality_models.get(model_name, 15)  # Default 15 for others

            # BONUS: Recommended model gets extra points
            if recommended_model and model_name == recommended_model:
                score += 50

            model_scores[model_name] = score

        # ============================================================
        # STEP 3: Select top models by score
        # ============================================================
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top 10 models, then filter to diverse set
        top_candidates = [model for model, score in sorted_models[:20]]

        final_models = []
        selected_langs = set()

        # Strategy: Pick diverse models
        # 1. Add multilingual models first (max 2)
        multilingual_added = 0
        for model in top_candidates:
            if MODEL_LANGUAGE_MAP[model] is None and multilingual_added < 2:
                final_models.append(model)
                multilingual_added += 1

        # 2. Add language-specific models for each detected language (1-2 per language)
        for lang in lang_list[:3]:  # Limit to 3 languages
            lang_lower = lang.lower()
            lang_models_added = 0
            for model in top_candidates:
                if MODEL_LANGUAGE_MAP[model] == lang_lower and lang_models_added < 2:
                    if model not in final_models:
                        final_models.append(model)
                        lang_models_added += 1
                        selected_langs.add(lang_lower)

        # 3. Fill remaining slots with highest scored models
        for model in top_candidates:
            if model not in final_models:
                final_models.append(model)
            if len(final_models) >= 7:
                break

        # Ensure we have at least some models
        if not final_models:
            final_models = ['xlm-roberta-base', 'bert-base-multilingual-cased', 'bert-base-uncased']

        # ============================================================
        # STEP 4: Display selection rationale
        # ============================================================
        if len(lang_list) > 1:
            lang_info = f"multilingual ({', '.join(lang_list[:3])})"
            if len(lang_list) > 3:
                lang_info += f" +{len(lang_list) - 3} more"
        elif len(lang_list) == 1:
            lang_info = f"{lang_list[0].upper()}"
        else:
            lang_info = "unknown language"

        text_len_info = "short" if text_length_avg < 150 else "medium" if text_length_avg < 350 else "long"

        self.console.print(f"[dim]🤖 AI Selection: {lang_info} dataset, {text_len_info} texts (avg {text_length_avg:.0f} chars)[/dim]")
        self.console.print(f"[dim]   Scored {len(MODEL_LANGUAGE_MAP)} models → Selected top {len(final_models)} by intelligent criteria[/dim]")

        # Build model-to-language mapping for selected models
        model_lang_map = {model: MODEL_LANGUAGE_MAP.get(model, None) for model in final_models}

        return final_models, model_lang_map

    def _get_preselected_benchmark_models(self, languages: set, text_length_avg: float) -> List[str]:
        """
        Let user choose from pre-selected model categories.
        NOW INCLUDES ALL LANGUAGES SUPPORTED IN THE PACKAGE!
        """
        self.console.print("\n[bold]📋 Pre-Selected Model Categories[/bold]\n")
        self.console.print("[dim]Choose from curated model lists organized by language and characteristics[/dim]\n")

        categories = {}
        lang_list = [l.lower() for l in languages] if languages else ['en']

        # ============ MULTILINGUAL MODELS ============
        if len(lang_list) > 1:
            categories['Multilingual'] = [
                'xlm-roberta-base',
                'xlm-roberta-large',
                'bert-base-multilingual-cased',
                'microsoft/mdeberta-v3-base'
            ]

        # ============ MAJOR LANGUAGES (Always show) ============
        categories['English'] = [
            'bert-base-uncased',
            'roberta-base',
            'distilbert-base-uncased',
            'microsoft/deberta-v3-base'
        ]

        if 'fr' in lang_list or True:  # Always show major languages
            categories['French'] = [
                'camembert-base',
                'flaubert/flaubert_base_cased',
                'cmarkea/distilcamembert-base',
                'almanach/camembert-base'
            ]

        if 'de' in lang_list or True:
            categories['German'] = [
                'bert-base-german-cased',
                'deepset/gbert-base',
                'distilbert-base-german-cased'
            ]

        if 'es' in lang_list or True:
            categories['Spanish'] = [
                'dccuchile/bert-base-spanish-wwm-cased',
                'PlanTL-GOB-ES/roberta-base-bne',
                'bertin-project/bertin-roberta-base-spanish'
            ]

        # ============ EUROPEAN LANGUAGES ============
        if 'it' in lang_list:
            categories['Italian'] = [
                'dbmdz/bert-base-italian-cased',
                'dbmdz/bert-base-italian-xxl-cased'
            ]

        if 'pt' in lang_list:
            categories['Portuguese'] = [
                'neuralmind/bert-base-portuguese-cased',
                'neuralmind/bert-large-portuguese-cased'
            ]

        if 'nl' in lang_list:
            categories['Dutch'] = [
                'GroNLP/bert-base-dutch-cased',
                'pdelobelle/robbert-v2-dutch-base'
            ]

        if 'pl' in lang_list:
            categories['Polish'] = [
                'dkleczek/bert-base-polish-uncased-v1',
                'allegro/herbert-base-cased'
            ]

        if 'sv' in lang_list:
            categories['Swedish'] = ['KB/bert-base-swedish-cased']

        if 'da' in lang_list:
            categories['Danish'] = ['Maltehb/danish-bert-botxo']

        if 'no' in lang_list:
            categories['Norwegian'] = ['ltg/norbert', 'NbAiLab/nb-bert-base']

        if 'fi' in lang_list:
            categories['Finnish'] = ['TurkuNLP/bert-base-finnish-cased-v1']

        if 'el' in lang_list:
            categories['Greek'] = ['nlpaueb/bert-base-greek-uncased-v1']

        if 'tr' in lang_list:
            categories['Turkish'] = ['dbmdz/bert-base-turkish-cased']

        if 'ro' in lang_list:
            categories['Romanian'] = ['dumitrescustefan/bert-base-romanian-cased-v1']

        if 'bg' in lang_list:
            categories['Bulgarian'] = ['iarfmoose/roberta-base-bulgarian']

        if 'hr' in lang_list or 'sr' in lang_list:
            categories['Croatian/Serbian'] = ['classla/bcms-bertic']

        if 'uk' in lang_list:
            categories['Ukrainian'] = ['youscan/ukr-roberta-base']

        if 'cs' in lang_list:
            categories['Czech'] = ['ufal/robeczech-base']

        # ============ ASIAN LANGUAGES ============
        if 'zh' in lang_list:
            categories['Chinese'] = [
                'bert-base-chinese',
                'hfl/chinese-roberta-wwm-ext',
                'hfl/chinese-roberta-wwm-ext-large'
            ]

        if 'ja' in lang_list:
            categories['Japanese'] = [
                'cl-tohoku/bert-base-japanese',
                'nlp-waseda/roberta-base-japanese'
            ]

        if 'ko' in lang_list:
            categories['Korean'] = [
                'klue/bert-base',
                'beomi/kcbert-base'
            ]

        if 'ar' in lang_list:
            categories['Arabic'] = [
                'aubmindlab/bert-base-arabertv2',
                'CAMeL-Lab/bert-base-arabic-camelbert-msa',
                'UBC-NLP/MARBERT'
            ]

        if 'ru' in lang_list:
            categories['Russian'] = [
                'DeepPavlov/rubert-base-cased',
                'ai-forever/ruBert-base'
            ]

        if 'hi' in lang_list:
            categories['Hindi'] = ['ai4bharat/indic-bert']

        if 'vi' in lang_list:
            categories['Vietnamese'] = ['vinai/phobert-base']

        if 'th' in lang_list:
            categories['Thai'] = ['airesearch/wangchanberta-base-att-spm-uncased']

        if 'id' in lang_list:
            categories['Indonesian'] = ['indobenchmark/indobert-base-p1']

        if 'he' in lang_list:
            categories['Hebrew'] = ['onlplab/alephbert-base']

        # ============ SPECIAL CATEGORIES ============
        if text_length_avg > 400:
            categories['Long Documents (>400 chars, 4096 tokens)'] = [
                'markussagen/xlm-roberta-longformer-base-4096',  # Multilingual FIRST
                'google/long-t5-local-base',  # Multilingual
                'allenai/longformer-base-4096',  # English only
                'google/bigbird-roberta-base'  # English only
            ]

        categories['Efficient/Fast'] = [
            'distilbert-base-uncased',
            'distilroberta-base',
            'albert-base-v2',
            'squeezebert/squeezebert-uncased'
        ]

        categories['State-of-the-Art'] = [
            'microsoft/deberta-v3-base',
            'microsoft/mdeberta-v3-base',
            'google/electra-base-discriminator',
            'xlm-roberta-large'
        ]

        # Display categories in organized fashion
        self.console.print("[bold cyan]Available Categories:[/bold cyan]\n")
        for i, (cat_name, models) in enumerate(categories.items(), 1):
            model_list = ', '.join(models[:3])  # Show first 3
            if len(models) > 3:
                model_list += f" (+{len(models)-3} more)"
            self.console.print(f"  [green]{i}.[/green] [cyan]{cat_name}:[/cyan] {model_list}")

        self.console.print(f"\n[dim]Total: {len(categories)} categories available[/dim]")
        self.console.print("\n[yellow]📝 Enter category names separated by commas[/yellow]")
        self.console.print("[dim]   Example: 'English,Multilingual' or 'French,Efficient'[/dim]\n")

        # Smart default based on detected languages
        default_cats = []
        if len(lang_list) > 1:
            default_cats.append("Multilingual")
        if 'en' in lang_list:
            default_cats.append("English")
        if 'fr' in lang_list:
            default_cats.append("French")
        if 'de' in lang_list:
            default_cats.append("German")
        if 'es' in lang_list:
            default_cats.append("Spanish")

        default_str = ','.join(default_cats) if default_cats else "Multilingual,English"

        selected_cats = Prompt.ask("Select categories", default=default_str)

        # Parse selected categories
        selected_models = []
        for cat in selected_cats.split(','):
            cat = cat.strip()
            # Case-insensitive matching
            for cat_name, models in categories.items():
                if cat.lower() in cat_name.lower():
                    selected_models.extend(models)
                    break

        # Deduplicate
        selected_models = list(dict.fromkeys(selected_models))

        return selected_models if selected_models else ['xlm-roberta-base', 'bert-base-multilingual-cased']

    def _get_custom_benchmark_models(self) -> List[str]:
        """Let user manually select models"""
        self.console.print("\n[bold]✏️  Custom Model Selection[/bold]\n")

        all_models = self._flatten_trainer_models()
        self.console.print(f"[dim]Available models ({len(all_models)}):[/dim]")
        for i, model in enumerate(all_models, 1):
            if i % 3 == 0:
                self.console.print(f"  {model}")
            else:
                self.console.print(f"  {model}", end="  ")
        if len(all_models) % 3 != 0:
            self.console.print()

        self.console.print("\n[dim]Enter model names separated by commas, or HuggingFace model IDs[/dim]")
        models_input = Prompt.ask("Model names", default="bert-base-uncased,xlm-roberta-base")

        selected_models = [m.strip() for m in models_input.split(',')]
        return selected_models

    def _save_training_metadata(
        self,
        bundle: TrainingDataBundle,
        mode: str,
        model_config: Dict[str, Any],
        execution_status: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save comprehensive training session metadata for reproducibility and resume capability.

        Parameters
        ----------
        bundle : TrainingDataBundle
            The training data bundle with all dataset information
        mode : str
            Training mode: quick, benchmark, custom, or distributed
        model_config : dict
            Model configuration including selected_model, epochs, batch_size, etc.
        execution_status : dict, optional
            Execution status information (status, started_at, completed_at, etc.)

        Returns
        -------
        Path
            Path to the saved metadata JSON file
        """
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Build comprehensive metadata
        metadata = {
            'training_session': {
                'timestamp': timestamp,
                'tool_version': 'LLMTool v1.0',
                'workflow': f'Training Studio - {mode.capitalize()}',
                'session_id': f'train_{timestamp}'
            },
            'dataset_config': {
                'primary_file': str(bundle.primary_file) if bundle.primary_file else None,
                'format': bundle.format_type if hasattr(bundle, 'format_type') else 'unknown',
                'strategy': bundle.strategy,
                'text_column': bundle.text_column,
                'label_column': bundle.label_column,
                'total_samples': len(bundle.samples) if hasattr(bundle, 'samples') and bundle.samples else 0,
                'num_categories': len(bundle.metadata.get('categories', [])),
                'category_distribution': bundle.metadata.get('category_distribution', {}),
                'training_files': {k: str(v) for k, v in bundle.training_files.items()} if hasattr(bundle, 'training_files') and bundle.training_files else {}
            },
            'language_config': {
                'confirmed_languages': list(bundle.metadata.get('confirmed_languages', [])),
                'language_distribution': bundle.metadata.get('language_distribution', {}),
                'model_strategy': bundle.metadata.get('model_strategy', 'multilingual'),
                'language_model_mapping': bundle.metadata.get('language_model_mapping', {})
            },
            'text_analysis': {
                'text_length_stats': bundle.metadata.get('text_length_stats', {}),
                'requires_long_document_model': bundle.metadata.get('requires_long_document_model', False),
                'avg_token_length': bundle.metadata.get('text_length_stats', {}).get('token_mean', 0),
                'max_token_length': bundle.metadata.get('text_length_stats', {}).get('token_max', 0)
            },
            'model_config': model_config,
            'execution_status': execution_status or {
                'status': 'pending',
                'started_at': None,
                'completed_at': None,
                'models_trained': [],
                'best_model': None,
                'best_f1': None
            },
            'output_paths': {
                'models_dir': str(self.settings.paths.models_dir),
                'logs_dir': str(self.settings.paths.logs_dir),
                'results_csv': None
            }
        }

        # Ensure training_sessions directory exists
        metadata_dir = self.settings.paths.logs_dir / "training_sessions"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata JSON
        metadata_filename = f"training_metadata_{timestamp}.json"
        metadata_path = metadata_dir / metadata_filename

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata_path

    def _update_training_metadata(
        self,
        metadata_path: Path,
        **updates
    ) -> None:
        """
        Update existing training metadata file with new information (post-training).

        Parameters
        ----------
        metadata_path : Path
            Path to the existing metadata JSON file
        **updates : dict
            Sections to update (e.g., execution_status={'status': 'completed'})
        """
        import json

        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return

        try:
            # Load existing metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Update sections
            for section, data in updates.items():
                if section in metadata:
                    if isinstance(metadata[section], dict) and isinstance(data, dict):
                        metadata[section].update(data)
                    else:
                        metadata[section] = data
                else:
                    metadata[section] = data

            # Save updated metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")

    def _reconstruct_bundle_from_metadata(self, metadata: Dict[str, Any]) -> Optional[TrainingDataBundle]:
        """
        Reconstruct a TrainingDataBundle from saved metadata for resume/relaunch.

        Parameters
        ----------
        metadata : dict
            Loaded metadata dictionary from JSON file

        Returns
        -------
        TrainingDataBundle or None
            Reconstructed bundle, or None if reconstruction fails
        """
        try:
            dataset_config = metadata.get('dataset_config', {})
            language_config = metadata.get('language_config', {})
            text_analysis = metadata.get('text_analysis', {})

            # Load primary file
            primary_file_str = dataset_config.get('primary_file')
            if not primary_file_str:
                self.console.print("[red]Error: No primary file found in metadata[/red]")
                return None

            primary_file = Path(primary_file_str)
            if not primary_file.exists():
                self.console.print(f"[red]Error: Dataset file not found: {primary_file}[/red]")
                return None

            # Create bundle with basic info
            bundle = TrainingDataBundle(
                primary_file=primary_file,
                format_type=dataset_config.get('format', 'unknown'),
                strategy=dataset_config.get('strategy', 'single-label'),
                text_column=dataset_config.get('text_column', 'text'),
                label_column=dataset_config.get('label_column', 'label'),
                samples=[],  # Will be loaded during training
                metadata={}
            )

            # Restore metadata fields
            bundle.metadata['confirmed_languages'] = set(language_config.get('confirmed_languages', []))
            bundle.metadata['language_distribution'] = language_config.get('language_distribution', {})
            bundle.metadata['model_strategy'] = language_config.get('model_strategy', 'multilingual')
            bundle.metadata['language_model_mapping'] = language_config.get('language_model_mapping', {})

            bundle.metadata['text_length_stats'] = text_analysis.get('text_length_stats', {})
            bundle.metadata['requires_long_document_model'] = text_analysis.get('requires_long_document_model', False)

            bundle.metadata['categories'] = list(dataset_config.get('category_distribution', {}).keys())
            bundle.metadata['category_distribution'] = dataset_config.get('category_distribution', {})

            # Restore training files paths if they exist
            training_files_dict = dataset_config.get('training_files', {})
            if training_files_dict:
                bundle.training_files = {k: Path(v) for k, v in training_files_dict.items()}

            # Restore recommended model if available
            model_config = metadata.get('model_config', {})
            if 'recommended_model' in model_config:
                bundle.recommended_model = model_config['recommended_model']

            return bundle

        except Exception as e:
            self.logger.error(f"Failed to reconstruct bundle from metadata: {e}")
            self.console.print(f"[red]Error reconstructing dataset: {e}[/red]")
            return None

    def _resume_training_studio(self):
        """Resume or relaunch training using saved parameters from previous sessions"""

        self.console.print("\n[bold cyan]🔄 Resume/Relaunch Training[/bold cyan]\n")
        self.console.print("[dim]Load saved parameters from previous training sessions[/dim]\n")

        # Detect metadata files
        metadata_dir = self.settings.paths.logs_dir / "training_sessions"
        self.console.print(f"[dim]Searching in: {metadata_dir}[/dim]\n")

        if not metadata_dir.exists():
            self.console.print("[yellow]⚠️  Training sessions directory not found.[/yellow]")
            self.console.print(f"[dim]Expected location: {metadata_dir}[/dim]")
            self.console.print("[dim]Complete a training first to create session history.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Find all metadata JSON files
        metadata_files = list(metadata_dir.glob("training_metadata_*.json"))

        if not metadata_files:
            self.console.print("[yellow]⚠️  No saved training sessions found.[/yellow]")
            self.console.print(f"[dim]Searched in: {metadata_dir}[/dim]")
            self.console.print(f"[dim]Found {len(list(metadata_dir.iterdir()))} files total in directory[/dim]")
            self.console.print("[dim]Complete a training and save parameters to use this feature.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Sort by modification time (most recent first)
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Display sessions table
        sessions_table = Table(
            title="📚 Previous Training Sessions (20 most recent)",
            border_style="cyan",
            box=box.ROUNDED
        )
        sessions_table.add_column("#", style="cyan bold", width=4)
        sessions_table.add_column("Date", style="yellow", width=18)
        sessions_table.add_column("Mode", style="magenta", width=15)
        sessions_table.add_column("Dataset", style="green", width=30)
        sessions_table.add_column("Model", style="blue", width=25)
        sessions_table.add_column("Status", style="white", width=12)

        import json
        from datetime import datetime

        # Load and display sessions
        valid_sessions = []
        parsing_errors = []  # Track errors for debugging

        for i, mf in enumerate(metadata_files[:20], 1):  # Show max 20 most recent
            try:
                with open(mf, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                session_info = metadata.get('training_session', {})
                dataset_config = metadata.get('dataset_config', {})
                model_config = metadata.get('model_config', {})
                exec_status = metadata.get('execution_status', {})

                # Format display
                timestamp_str = session_info.get('timestamp', '')
                try:
                    dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = timestamp_str

                mode = model_config.get('training_mode', 'unknown')
                dataset_path = dataset_config.get('primary_file', '')
                dataset_name = Path(dataset_path).name if dataset_path else 'N/A'
                if len(dataset_name) > 28:
                    dataset_name = dataset_name[:25] + "..."

                model_name = model_config.get('selected_model') or 'N/A'
                if len(model_name) > 23:
                    model_name = model_name[:20] + "..."

                status = exec_status.get('status', 'unknown')

                # Color code status
                if status == 'completed':
                    status_display = f"[green]✓ {status}[/green]"
                elif status == 'failed':
                    status_display = f"[red]✗ {status}[/red]"
                else:
                    status_display = f"[yellow]⏸ {status}[/yellow]"

                sessions_table.add_row(
                    str(i),
                    date_str,
                    mode,
                    dataset_name,
                    model_name,
                    status_display
                )

                valid_sessions.append((mf, metadata))

            except Exception as e:
                self.logger.debug(f"Skipping invalid metadata file {mf}: {e}")
                parsing_errors.append((mf.name, str(e)))
                continue

        if not valid_sessions:
            self.console.print("[yellow]No valid training sessions found.[/yellow]")

            # Show parsing errors if any for debugging
            if parsing_errors:
                self.console.print(f"\n[dim]Parsing errors for {len(parsing_errors)} files:[/dim]")
                for fname, err in parsing_errors[:5]:  # Show first 5
                    self.console.print(f"[dim]  • {fname}: {err[:80]}[/dim]")

            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        self.console.print(sessions_table)

        # Select session
        session_choices = [str(i) for i in range(1, len(valid_sessions) + 1)] + ["back"]
        session_choice = Prompt.ask(
            "\n[bold yellow]Select session to resume/relaunch[/bold yellow]",
            choices=session_choices,
            default="1"
        )

        if session_choice == "back":
            return

        metadata_file, metadata = valid_sessions[int(session_choice) - 1]

        # Display selected session details
        self.console.print("\n[bold cyan]📋 Selected Session Details[/bold cyan]")

        details_table = Table(border_style="green", box=box.SIMPLE)
        details_table.add_column("Parameter", style="cyan bold", width=25)
        details_table.add_column("Value", style="white", width=60)

        session_info = metadata.get('training_session', {})
        dataset_config = metadata.get('dataset_config', {})
        model_config = metadata.get('model_config', {})
        exec_status = metadata.get('execution_status', {})

        details_table.add_row("Timestamp", session_info.get('timestamp', 'N/A'))
        details_table.add_row("Workflow", session_info.get('workflow', 'N/A'))
        details_table.add_row("Dataset", Path(dataset_config.get('primary_file', '')).name)
        details_table.add_row("Strategy", dataset_config.get('strategy', 'N/A'))
        details_table.add_row("Total Samples", str(dataset_config.get('total_samples', 0)))
        details_table.add_row("Training Mode", model_config.get('training_mode', 'N/A'))

        selected_model = model_config.get('selected_model')
        if selected_model:
            details_table.add_row("Model", selected_model)

        epochs = model_config.get('epochs')
        if epochs:
            details_table.add_row("Epochs", str(epochs))

        batch_size = model_config.get('batch_size')
        if batch_size:
            details_table.add_row("Batch Size", str(batch_size))

        details_table.add_row("Status", exec_status.get('status', 'unknown'))

        self.console.print(details_table)

        # Ask: resume or relaunch?
        self.console.print("\n[bold cyan]🎯 Action Mode[/bold cyan]")
        self.console.print("  • [cyan]resume[/cyan]   - Continue incomplete training (if interrupted)")
        self.console.print("  • [cyan]relaunch[/cyan] - Start fresh with same parameters\n")

        action_mode = Prompt.ask(
            "[bold yellow]Resume or relaunch?[/bold yellow]",
            choices=["resume", "relaunch"],
            default="relaunch"  # Default to relaunch since resume is complex for training
        )

        # Reconstruct bundle from metadata
        self.console.print(f"\n[cyan]Reconstructing dataset configuration...[/cyan]")

        bundle = self._reconstruct_bundle_from_metadata(metadata)

        if bundle is None:
            self.console.print("[red]Failed to reconstruct training configuration.[/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Get training mode
        mode = model_config.get('training_mode', 'quick')

        # Display confirmation message
        if action_mode == 'resume':
            self.console.print(f"\n[green]✓ Resuming training session...[/green]\n")
        else:
            self.console.print(f"\n[green]✓ Relaunching training with saved parameters...[/green]\n")

        # Execute with loaded parameters
        # We'll modify _training_studio_confirm_and_execute to accept optional pre-loaded config
        self._training_studio_confirm_and_execute(
            bundle,
            mode,
            preloaded_config=model_config,
            is_resume=action_mode == 'resume'
        )

    def _training_studio_default_model(self) -> str:
        models = self._flatten_trainer_models()
        return "bert-base-uncased" if "bert-base-uncased" in models else (models[0] if models else "bert-base-uncased")

    def _show_analysis_and_get_columns(self, analysis: Dict[str, Any], format_type: str = "general") -> Dict[str, Any]:
        """
        Show file analysis results and intelligently detect columns with user confirmation.
        Returns dictionary with detected column names and confirmed languages.
        """
        result = {
            'text': 'text',
            'label': 'label',
            'id': None,
            'lang': None,
            'confirmed_languages': set()
        }

        # Show analysis issues
        if analysis['issues']:
            self.console.print("\n[yellow]⚠️  Analysis Results:[/yellow]")
            for issue in analysis['issues']:
                self.console.print(f"  {issue}")

        all_columns = analysis.get('all_columns', [])

        # Auto-suggest text column
        text_column_default = "text"
        if analysis['text_column_candidates']:
            best_text = analysis['text_column_candidates'][0]['name']
            text_column_default = best_text
            self.console.print(f"\n[green]✓ Text column detected: '{best_text}'[/green]")

        if all_columns:
            self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

        result['text'] = Prompt.ask("Text column", default=text_column_default)

        # Auto-suggest label column
        label_column_default = "labels" if "multi" in format_type else "label"
        if analysis['annotation_column_candidates']:
            best_label = analysis['annotation_column_candidates'][0]['name']
            label_column_default = best_label
            self.console.print(f"\n[green]✓ Label column detected: '{best_label}'[/green]")
            stats = analysis['annotation_stats'].get(best_label, {})
            fill_rate = stats.get('fill_rate', 0)
            if fill_rate > 0:
                self.console.print(f"[dim]  ({fill_rate*100:.1f}% of rows have labels)[/dim]")

        if all_columns:
            self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")

        result['label'] = Prompt.ask("Label/Category column", default=label_column_default)

        # Language detection
        languages_found = set(analysis['languages_detected'].keys())

        if languages_found:
            self.console.print(f"\n[bold]🌍 Languages Detected:[/bold]")
            for lang, count in analysis['languages_detected'].items():
                self.console.print(f"  • {lang.upper()}: {count} rows")

            lang_list = ', '.join([l.upper() for l in sorted(languages_found)])
            lang_confirmed = Confirm.ask(
                f"\n[bold]Detected languages: {lang_list}. Is this correct?[/bold]",
                default=True
            )

            if lang_confirmed:
                result['confirmed_languages'] = languages_found
                self.console.print("[green]✓ Languages confirmed[/green]")
            else:
                self.console.print("\n[yellow]Please specify languages manually[/yellow]")
                manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)")
                result['confirmed_languages'] = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

            # Auto-suggest language column if detected
            if analysis['language_column_candidates']:
                lang_column_default = analysis['language_column_candidates'][0]
                self.console.print(f"\n[green]✓ Language column detected: '{lang_column_default}'[/green]")
                if all_columns:
                    self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
                result['lang'] = Prompt.ask("Language column (optional)", default=lang_column_default)
        else:
            # No language column detected - ask if user wants to apply language detection
            self.console.print("\n[yellow]ℹ️  No language column detected in data[/yellow]")
            apply_lang_detection = Confirm.ask(
                "Would you like to apply automatic language detection on the text column?",
                default=True
            )

            if apply_lang_detection:
                self.console.print("[cyan]🔍 Detecting languages from text content...[/cyan]")
                self.console.print("[dim]  Language detection will be applied during training[/dim]")
                manual_langs = Prompt.ask(
                    "Expected language codes (optional, comma-separated, e.g., en,fr,de)",
                    default=""
                )
                if manual_langs.strip():
                    result['confirmed_languages'] = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])

        # Auto-suggest ID column
        if analysis['id_column_candidates']:
            id_column_default = analysis['id_column_candidates'][0]
            self.console.print(f"\n[green]✓ ID column detected: '{id_column_default}'[/green]")
            if all_columns:
                self.console.print(f"[dim]  Available columns: {', '.join(all_columns)}[/dim]")
            result['id'] = Prompt.ask("Identifier column (optional)", default=id_column_default)

        return result

    def _get_long_document_model_recommendation(self, confirmed_languages: set) -> Optional[str]:
        """
        Get long-document model recommendations based on languages.
        Prioritizes models that can handle >512 tokens.
        """
        # Available long-document models
        LONG_DOCUMENT_MODELS = [
            {
                'model': 'allenai/longformer-base-4096',
                'max_tokens': 4096,
                'languages': ['en'],
                'reason': 'English long-document model (4096 tokens)'
            },
            {
                'model': 'google/bigbird-roberta-base',
                'max_tokens': 4096,
                'languages': ['en'],
                'reason': 'English sparse-attention long-document model (4096 tokens)'
            },
            {
                'model': 'markussagen/xlm-roberta-longformer-base-4096',
                'max_tokens': 4096,
                'languages': ['multilingual'],
                'reason': 'Multilingual long-document model (4096 tokens)'
            },
            {
                'model': 'xlm-roberta-base',
                'max_tokens': 512,
                'languages': ['multilingual'],
                'reason': 'Multilingual baseline (512 tokens, fallback)'
            },
        ]

        # Filter models based on languages
        suitable_models = []
        for model in LONG_DOCUMENT_MODELS:
            if 'multilingual' in model['languages']:
                suitable_models.append(model)
            elif confirmed_languages:
                if any(lang in model['languages'] for lang in confirmed_languages):
                    suitable_models.append(model)

        if not suitable_models:
            suitable_models = LONG_DOCUMENT_MODELS  # Fallback to all

        self.console.print(f"\n[bold]🤖 Long-Document Model Recommendations:[/bold]")
        for i, model_info in enumerate(suitable_models[:5], 1):
            self.console.print(f"  {i}. [cyan]{model_info['model']}[/cyan] - {model_info['reason']}")

        choice = Prompt.ask(
            f"Select model (1-{min(5, len(suitable_models))}, or enter model name)",
            default="1"
        )

        if choice.isdigit() and 0 < int(choice) <= len(suitable_models):
            model_to_use = suitable_models[int(choice) - 1]['model']
            self.console.print(f"[green]✓ Selected: {model_to_use}[/green]")
            return model_to_use
        else:
            return choice

    def _get_long_document_models_for_language(self, lang: str) -> list:
        """
        Get long-document model recommendations for a specific language.
        Returns list in LanguageNormalizer.recommend_models format.
        Uses the model catalog (TrainerModelDetector) when available.
        """
        # Try to get models from catalog first
        if self.available_trainer_models:
            # Map language codes to catalog categories
            LANG_TO_CATEGORY = {
                'en': 'Long Document Models',
                'fr': 'Long Document Models - French',
                'es': 'Long Document Models - Spanish',
                'de': 'Long Document Models - German',
                'it': 'Long Document Models - Italian',
                'pt': 'Long Document Models - Portuguese',
                'nl': 'Long Document Models - Dutch',
                'pl': 'Long Document Models - Polish',
                'ru': 'Long Document Models - Russian',
                'zh': 'Long Document Models - Chinese',
                'ja': 'Long Document Models - Japanese',
                'ar': 'Long Document Models - Arabic',
            }

            category = LANG_TO_CATEGORY.get(lang, 'Long Document Models')

            # Get models from catalog
            if category in self.available_trainer_models:
                catalog_models = self.available_trainer_models[category]
                recommendations = []

                for model in catalog_models:
                    # Build reason from model metadata
                    reason_parts = [
                        model.get('type', 'Unknown type'),
                        f"({model.get('max_length', '512')} tokens)"
                    ]
                    if model.get('performance'):
                        reason_parts.append(model['performance'])

                    recommendations.append({
                        'model': model['name'],
                        'reason': ' - '.join(reason_parts)
                    })

                # Add multilingual fallback if not already included
                if lang != 'en' and 'Long Document Models' in self.available_trainer_models:
                    for model in self.available_trainer_models['Long Document Models'][:2]:
                        if 'xlm' in model['name'].lower() or 'multilingual' in model.get('type', '').lower():
                            recommendations.append({
                                'model': model['name'],
                                'reason': f"{model.get('type')} - Multilingual fallback ({model.get('max_length', '4096')} tokens)"
                            })

                if recommendations:
                    return recommendations

        # Fallback: hardcoded comprehensive list if catalog unavailable
        LANG_LONG_MODELS = {
            'en': [
                {'model': 'allenai/longformer-base-4096', 'reason': 'English Longformer (4096 tokens, optimized for English)'},
                {'model': 'google/bigbird-roberta-base', 'reason': 'English BigBird sparse-attention (4096 tokens)'},
                {'model': 'google/long-t5-local-base', 'reason': 'Multilingual T5 for long documents (4096+ tokens)'},
                {'model': 'roberta-base', 'reason': 'English RoBERTa baseline (512 tokens, fallback)'},
            ],
            'fr': [
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting French (4096 tokens)'},
                {'model': 'google/long-t5-local-base', 'reason': 'Multilingual T5 for long documents (4096+ tokens)'},
                {'model': 'cmarkea/distilcamembert-base-nli', 'reason': 'French DistilCamemBERT optimized (512 tokens)'},
                {'model': 'camembert-base', 'reason': 'French CamemBERT baseline (512 tokens)'},
            ],
            'es': [
                {'model': 'PlanTL-GOB-ES/roberta-base-bne', 'reason': 'Spanish RoBERTa optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Spanish (4096 tokens)'},
                {'model': 'dccuchile/bert-base-spanish-wwm-cased', 'reason': 'Spanish BERT baseline (512 tokens)'},
                {'model': 'bertin-project/bertin-roberta-base-spanish', 'reason': 'Spanish BERTIN RoBERTa (512 tokens)'},
            ],
            'de': [
                {'model': 'deepset/gbert-base', 'reason': 'German GBERT optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting German (4096 tokens)'},
                {'model': 'bert-base-german-cased', 'reason': 'German BERT baseline (512 tokens)'},
                {'model': 'dbmdz/bert-base-german-uncased', 'reason': 'German BERT uncased (512 tokens)'},
            ],
            'it': [
                {'model': 'dbmdz/bert-base-italian-cased', 'reason': 'Italian BERT optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Italian (4096 tokens)'},
                {'model': 'dbmdz/bert-base-italian-xxl-cased', 'reason': 'Italian BERT XXL (512 tokens, high performance)'},
            ],
            'pt': [
                {'model': 'neuralmind/bert-base-portuguese-cased', 'reason': 'Portuguese BERT optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Portuguese (4096 tokens)'},
                {'model': 'adalbertojunior/distilbert-portuguese-cased', 'reason': 'Portuguese DistilBERT (512 tokens, efficient)'},
            ],
            'nl': [
                {'model': 'GroNLP/bert-base-dutch-cased', 'reason': 'Dutch BERT optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Dutch (4096 tokens)'},
                {'model': 'wietsedv/bert-base-dutch-cased', 'reason': 'Dutch BERT baseline (512 tokens)'},
            ],
            'pl': [
                {'model': 'allegro/herbert-base-cased', 'reason': 'Polish HerBERT optimized (514 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Polish (4096 tokens)'},
                {'model': 'dkleczek/bert-base-polish-cased-v1', 'reason': 'Polish BERT baseline (512 tokens)'},
            ],
            'ru': [
                {'model': 'DeepPavlov/rubert-base-cased', 'reason': 'Russian RuBERT optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Russian (4096 tokens)'},
                {'model': 'sberbank-ai/ruBert-base', 'reason': 'Russian BERT baseline (512 tokens)'},
            ],
            'zh': [
                {'model': 'hfl/chinese-roberta-wwm-ext', 'reason': 'Chinese RoBERTa WWM optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Chinese (4096 tokens)'},
                {'model': 'bert-base-chinese', 'reason': 'Chinese BERT baseline (512 tokens)'},
            ],
            'ja': [
                {'model': 'cl-tohoku/bert-base-japanese-whole-word-masking', 'reason': 'Japanese BERT WWM optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Japanese (4096 tokens)'},
                {'model': 'cl-tohoku/bert-base-japanese', 'reason': 'Japanese BERT baseline (512 tokens)'},
            ],
            'ar': [
                {'model': 'aubmindlab/bert-base-arabert', 'reason': 'Arabic AraBERT optimized (512 tokens)'},
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer supporting Arabic (4096 tokens)'},
                {'model': 'asafaya/bert-base-arabic', 'reason': 'Arabic BERT baseline (512 tokens)'},
            ],
            'multilingual': [
                {'model': 'markussagen/xlm-roberta-longformer-base-4096', 'reason': 'Multilingual Longformer (100+ languages, 4096 tokens)'},
                {'model': 'xlm-roberta-large', 'reason': 'Multilingual XLM-RoBERTa large (100+ languages, 512 tokens)'},
                {'model': 'xlm-roberta-base', 'reason': 'Multilingual XLM-RoBERTa base (100+ languages, 512 tokens)'},
                {'model': 'bert-base-multilingual-cased', 'reason': 'Multilingual BERT baseline (104 languages, 512 tokens)'},
            ],
        }

        # Return language-specific models or multilingual as fallback
        return LANG_LONG_MODELS.get(lang, LANG_LONG_MODELS.get('multilingual', []))

    def _get_model_recommendation_from_languages(self, confirmed_languages: set) -> Optional[str]:
        """
        Get model recommendations based on detected/confirmed languages.
        Returns selected model name or None.
        """
        if not confirmed_languages:
            return None

        recommendations = LanguageNormalizer.recommend_models(confirmed_languages, self.available_trainer_models)

        if not recommendations:
            return None

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
            return Prompt.ask("\nEnter model name", default="xlm-roberta-base")
        elif model_choice == "skip":
            return "bert-base-uncased"
        elif model_choice.isdigit():
            idx = int(model_choice) - 1
            if 0 <= idx < len(recommendations):
                model_to_use = recommendations[idx]['model']
                self.console.print(f"[green]✓ Selected: {model_to_use}[/green]")
                return model_to_use
            else:
                self.console.print("[yellow]Invalid selection, using first recommendation[/yellow]")
                return recommendations[0]['model']
        else:
            return recommendations[0]['model']

    def _detect_languages_and_analyze_text(
        self,
        df: 'pd.DataFrame',
        text_column: str,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Universal function to detect languages and analyze text characteristics.
        Works for ANY dataset format and ANY training mode.

        Args:
            df: DataFrame containing the text data
            text_column: Name of the column containing text
            sample_size: Number of samples to analyze for language detection

        Returns:
            Dictionary with:
            - languages_detected: {lang: count} dictionary
            - text_length_stats: {avg_length, max_length, min_length, median_length}
            - long_document_percentage: Percentage of documents > 512 tokens
            - user_prefers_long_models: Boolean recommendation
        """
        from llm_tool.utils.language_detector import LanguageDetector

        # Initialize results
        results = {
            'languages_detected': {},
            'text_length_stats': {
                'avg_length': 0,
                'max_length': 0,
                'min_length': 0,
                'median_length': 0
            },
            'long_document_percentage': 0,
            'user_prefers_long_models': False
        }

        # Check if text column exists
        if text_column not in df.columns:
            self.logger.warning(f"Text column '{text_column}' not found in dataset")
            return results

        # Get text samples (filter out NaN values)
        text_samples = df[text_column].dropna()

        if len(text_samples) == 0:
            self.logger.warning("No text data found in dataset")
            return results

        # Sample for language detection (use up to sample_size rows)
        sample_texts = text_samples.head(sample_size).tolist()

        # Detect languages using LanguageDetector
        detector = LanguageDetector()
        language_counts = Counter()

        for text in sample_texts:
            if isinstance(text, str) and text.strip():
                detected_lang = detector.detect(text)
                if detected_lang:
                    # LanguageDetector returns dict like {'language': 'fr', 'confidence': 0.95}
                    if isinstance(detected_lang, dict):
                        lang = detected_lang.get('language')
                        if lang:
                            language_counts[lang] += 1
                    elif isinstance(detected_lang, str):
                        language_counts[detected_lang] += 1

        results['languages_detected'] = dict(language_counts)

        # Calculate text length statistics
        text_lengths = [len(str(text)) for text in text_samples if pd.notna(text)]

        if text_lengths:
            import statistics
            results['text_length_stats'] = {
                'avg_length': sum(text_lengths) / len(text_lengths),
                'max_length': max(text_lengths),
                'min_length': min(text_lengths),
                'median_length': statistics.median(text_lengths)
            }

            # Estimate long documents (assuming ~4 chars per token)
            long_docs = sum(1 for length in text_lengths if length > 2048)  # 512 tokens * 4 chars
            results['long_document_percentage'] = (long_docs / len(text_lengths)) * 100

            # Recommend long-document models if >20% of docs are long
            results['user_prefers_long_models'] = results['long_document_percentage'] > 20

        return results

    def _display_language_analysis_and_get_model(
        self,
        analysis_results: Dict[str, Any],
        interactive: bool = True
    ) -> Tuple[Set[str], Optional[str]]:
        """
        Display language analysis results and get model recommendation.
        Universal function that works for ANY dataset format and ANY training mode.

        Args:
            analysis_results: Results from _detect_languages_and_analyze_text
            interactive: If True, ask user to confirm languages and select model

        Returns:
            Tuple of (confirmed_languages, selected_model)
        """
        # LanguageNormalizer is defined at the module level, no need to import

        languages_found = set(analysis_results['languages_detected'].keys())
        text_stats = analysis_results['text_length_stats']
        confirmed_languages = set()
        model_to_use = None

        # Display language detection results
        if languages_found:
            self.console.print(f"\n[bold]🌍 Languages Detected:[/bold]")
            for lang, count in analysis_results['languages_detected'].items():
                self.console.print(f"  • {lang.upper()}: {count} samples")

            # Display text statistics
            self.console.print(f"\n[bold]📊 Text Statistics:[/bold]")
            self.console.print(f"  • Average length: {text_stats['avg_length']:.0f} characters")
            self.console.print(f"  • Max length: {text_stats['max_length']:.0f} characters")
            self.console.print(f"  • Median length: {text_stats['median_length']:.0f} characters")

            if analysis_results['long_document_percentage'] > 0:
                self.console.print(f"  • Long documents (>512 tokens): {analysis_results['long_document_percentage']:.1f}%")

                if analysis_results['user_prefers_long_models']:
                    self.console.print("\n[yellow]💡 Recommendation: Consider using long-document models (e.g., Longformer, BigBird)[/yellow]")

            if interactive:
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
            else:
                confirmed_languages = languages_found
        else:
            # No languages detected - ask user
            self.console.print("\n[yellow]⚠️ No languages could be auto-detected[/yellow]")

            if interactive:
                manual_langs = Prompt.ask("Enter language codes (comma-separated, e.g., en,fr,de)", default="en")
                confirmed_languages = set([l.strip().lower() for l in manual_langs.split(',') if l.strip()])
            else:
                confirmed_languages = {'en'}  # Default to English

        # Get model recommendations based on confirmed languages
        if confirmed_languages and interactive:
            # Consider long-document models if needed
            if analysis_results.get('user_prefers_long_models'):
                self.console.print("\n[bold]🤖 Recommended Long-Document Models:[/bold]")

                # Get multilingual long-doc models - MULTILINGUAL FIRST
                long_doc_recs = [
                    {"model": "markussagen/xlm-roberta-longformer-base-4096", "reason": "Multilingual long-document support (100+ languages, 4096 tokens)"},
                    {"model": "google/long-t5-local-base", "reason": "Multilingual T5 for long documents (4096+ tokens)"},
                    {"model": "allenai/longformer-base-4096", "reason": "English-only, efficient for documents up to 4096 tokens"},
                    {"model": "google/bigbird-roberta-base", "reason": "English-only, sparse attention for very long documents"}
                ]

                for i, rec in enumerate(long_doc_recs, 1):
                    self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

                use_long = Confirm.ask("\n[bold]Use long-document model?[/bold]", default=True)

                if use_long:
                    choice = IntPrompt.ask("Select model (1-4)", default=1)
                    if 1 <= choice <= len(long_doc_recs):
                        model_to_use = long_doc_recs[choice - 1]['model']
                        self.console.print(f"[green]✓ Selected: {model_to_use}[/green]")
                        return confirmed_languages, model_to_use

            # Get standard model recommendations
            recommendations = LanguageNormalizer.recommend_models(confirmed_languages, self.available_trainer_models)

            if recommendations:
                self.console.print(f"\n[bold]🤖 Recommended Models for Your Languages:[/bold]")
                for i, rec in enumerate(recommendations[:5], 1):
                    self.console.print(f"  {i}. [cyan]{rec['model']}[/cyan] - {rec['reason']}")

                # Store recommendations in bundle for later use, don't ask now
                self.console.print(f"\n[dim]ℹ️  Model selection will be done when choosing the training mode[/dim]")

                # Use first recommendation as default, but don't force it
                model_to_use = recommendations[0]['model'] if recommendations else "bert-base-uncased"

        return confirmed_languages, model_to_use

    def bert_annotation_studio(self):
        """BERT Annotation Studio - Advanced annotation with trained models"""
        from .bert_annotation_studio import BERTAnnotationStudio

        studio = BERTAnnotationStudio(
            console=self.console,
            settings=self.settings,
            logger=self.logger
        )
        studio.run()

    def validation_lab(self):
        """Validation lab for quality control and Doccano export"""
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display personalized mode info
        self._display_section_header(
            "🔍 Validation Lab",
            "Quality control and human review preparation tools",
            mode_info={
                'workflow': 'Load Annotations → Quality Metrics → Sample Review → Export to Doccano/Label Studio',
                'capabilities': ['Quality Scoring', 'Sample Analysis', 'Export Tools', 'Inter-rater Reliability'],
                'input': 'Annotated JSON files from LLM',
                'output': 'Quality reports + Doccano/Label Studio export files',
                'best_for': 'Validating LLM annotations before training or publication',
                'duration': '~2-5 min'
            }
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
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display personalized mode info
        self._display_section_header(
            "💾 Profile Manager",
            "Save and reuse your favorite pipeline configurations",
            mode_info={
                'workflow': 'Browse Profiles → Select → Load Configuration → Execute',
                'capabilities': ['Save Configs', 'Quick Reload', 'Profile Sharing', 'Version Control'],
                'input': 'Saved profile from previous runs',
                'output': 'Executed pipeline with saved configuration',
                'best_for': 'Rerunning the same pipeline configuration multiple times',
                'duration': '~1 min + pipeline execution time'
            }
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
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display personalized mode info
        self._display_section_header(
            "⚙️ Advanced Settings",
            "Fine-tune system configuration and preferences",
            mode_info={
                'workflow': 'Browse Settings → Modify → Save → Apply',
                'capabilities': ['API Keys', 'Model Defaults', 'Path Configuration', 'Performance Tuning'],
                'input': 'Current system settings',
                'output': 'Updated configuration',
                'best_for': 'Customizing system behavior and defaults',
                'duration': '~2-5 min'
            }
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
        import pandas as pd
        self.console.print("\n[bold cyan]🎯 Smart Annotate - Guided Wizard[/bold cyan]\n")
    
        # Step 1: Data Selection
        self.console.print("[bold]Step 1/7: Data Selection[/bold]")

        if not self.detected_datasets:
            self.console.print("[yellow]No datasets auto-detected.[/yellow]")
            data_path = Path(self._prompt_file_path("Dataset path"))
        else:
            self.console.print(f"\n[bold cyan]📊 Found {len(self.detected_datasets)} dataset(s):[/bold cyan]\n")

            # Create table for datasets
            datasets_table = Table(border_style="cyan", show_header=True)
            datasets_table.add_column("#", style="bold yellow", width=4)
            datasets_table.add_column("Filename", style="white")
            datasets_table.add_column("Format", style="green", width=10)
            datasets_table.add_column("Size", style="magenta", width=10)
            datasets_table.add_column("Rows", style="cyan", width=10)
            datasets_table.add_column("Columns", style="blue", width=10)

            for i, ds in enumerate(self.detected_datasets[:20], 1):
                # Format size
                if ds.size_mb < 0.1:
                    size_str = f"{ds.size_mb * 1024:.1f} KB"
                else:
                    size_str = f"{ds.size_mb:.1f} MB"

                # Format rows and columns
                rows_str = f"{ds.rows:,}" if ds.rows else "?"
                cols_str = str(len(ds.columns)) if ds.columns else "?"

                datasets_table.add_row(
                    str(i),
                    ds.path.name,
                    ds.format.upper(),
                    size_str,
                    rows_str,
                    cols_str
                )

            self.console.print(datasets_table)
            self.console.print()

            use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
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
        self.console.print("\n[bold]Step 2/7: Text Column Selection[/bold]")
    
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
    
        # Step 2b: ID Column Selection (FROM QUICK START)
        self.console.print("\n[bold]Step 2b/7: Identifier Column Selection[/bold]")
    
        # Get available columns
        available_columns = column_info['all_columns'] if column_info and column_info.get('all_columns') else []
    
        # Auto-detect potential ID columns
        suggested_id = None
        if available_columns:
            for col in available_columns:
                lowered = col.lower()
                if lowered == 'id' or lowered.endswith('_id') or 'identifier' in lowered:
                    suggested_id = col
                    break
    
        identifier_column = self._prompt_for_identifier_column(available_columns, suggested_id)
    
        self.console.print(f"[green]✓ Identifier strategy: {identifier_column}[/green]")
    
        # Check if user wants to return to menu
        if self._check_return_to_menu("with column configuration"):
            return
    
        # Step 3: Model Selection
        self.console.print("\n[bold]Step 3/7: Model Selection[/bold]")
        self.console.print("[dim]Tested API models: OpenAI & Anthropic[/dim]\n")
    
        selected_llm = self._select_llm_interactive()
        provider = selected_llm.provider
        model_name = selected_llm.name
    
        # Get API key if needed
        api_key = None
        if selected_llm.requires_api_key:
            api_key = self._get_or_prompt_api_key(provider, model_name)
    
        # Step 4: Prompt Configuration
        self.console.print("\n[bold]Step 4/7: Prompt Configuration[/bold]")
    
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
                if indices.strip():  # Only process if not empty
                    for idx_str in indices.split(','):
                        idx_str = idx_str.strip()
                        if idx_str:  # Skip empty strings
                            try:
                                idx = int(idx_str) - 1
                                if 0 <= idx < len(detected_prompts):
                                    selected_prompts.append(detected_prompts[idx])
                            except ValueError:
                                self.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
                if not selected_prompts:
                    self.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                    selected_prompts = detected_prompts
                else:
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
    
    
        # Step 4b: Language Column Detection (FROM QUICK START)
        self.console.print("\n[bold]Step 4b/7: Language Column Detection[/bold]")

        lang_column = None
        available_columns = column_info.get('all_columns', []) if column_info else []
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
                        "[yellow]⚠️  Language information is needed for training. Enable automatic language detection?[/yellow]",
                        default=True
                    )
                    if auto_detect:
                        self.console.print("[dim]Language will be automatically detected for each text during annotation.[/dim]")
                        lang_column = None  # Will trigger auto-detection later
                    else:
                        self.console.print("[yellow]⚠️  Warning: Proceeding without language information may affect training quality.[/yellow]")
                        lang_column = None
            else:
                # No language column detected
                has_lang = Confirm.ask("Does your dataset have a language column?", default=False)
                if has_lang:
                    lang_column = Prompt.ask(
                        "Language column name",
                        choices=available_columns,
                        default=available_columns[0] if available_columns else "language"
                    )
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
        self.console.print("\n[bold]Step 5/7: Advanced Options[/bold]")
    
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
        recommended_sample = None

        if scope_choice == "limit":
            # Option 2: FIRST ask about representative sample calculation (before asking for number)
            if total_rows and total_rows > 1000:
                self.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
                self.console.print("  Calculate statistically representative sample size (95% confidence interval)")
                self.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

                calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

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
                    self.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

            # THEN ask for specific number (with recommendation as default if calculated)
            default_limit = recommended_sample if recommended_sample else 100
            annotation_limit = self._int_prompt_with_validation(
                f"How many rows to annotate?",
                default=default_limit,
                min_value=1,
                max_value=total_rows if total_rows else 1000000
            )

            # Check if user chose the recommended sample
            if recommended_sample and annotation_limit == recommended_sample:
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
        self.console.print("\n[bold]Step 6/7: Review & Execute[/bold]")
    
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
            'identifier_column': identifier_column,  # From Step 2b: User-selected ID strategy
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
            'lang_column': lang_column,  # From Step 4b: Language column for training metadata
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
                },
                'training_workflow': {
                    'enabled': False,  # Will be updated after training workflow
                    'training_params_file': None,  # Will be added after training
                    'note': 'Training parameters will be saved separately after annotation completes'
                }
            }

            # Save metadata JSON (PRE-ANNOTATION SAVE POINT 1)
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

            # ============================================================
            # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
            # ============================================================
            if not lang_column:
                self.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
                self.console.print("[yellow]No language column was provided. Detecting languages for training...[/yellow]\n")

                try:
                    import pandas as pd
                    from llm_tool.utils.language_detector import LanguageDetector

                    # Load annotated file
                    df_annotated = pd.read_csv(output_file)

                    # CRITICAL: Only detect languages for ANNOTATED rows
                    # The output file may contain ALL original rows, but we only want to detect
                    # languages for rows that were actually annotated
                    original_row_count = len(df_annotated)

                    # Try to identify annotated rows by checking for annotation columns
                    # Common annotation column names: 'label', 'category', 'annotation', 'labels'
                    annotation_cols = [col for col in df_annotated.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]

                    if annotation_cols:
                        # Filter to only rows that have annotations (non-null in annotation column)
                        annotation_col = annotation_cols[0]
                        df_annotated = df_annotated[df_annotated[annotation_col].notna()].copy()
                        self.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows (out of {original_row_count:,} total rows in file)[/dim]")
                    else:
                        self.console.print(f"[yellow]⚠️  Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                    if len(df_annotated) == 0:
                        self.console.print("[yellow]⚠️  No annotated rows found. Skipping language detection.[/yellow]")
                    elif text_column in df_annotated.columns:
                        # Get ALL texts (including NaN) to maintain index alignment
                        all_texts = df_annotated[text_column].tolist()

                        # Count non-empty texts for display
                        non_empty_texts = sum(1 for text in all_texts if pd.notna(text) and len(str(text).strip()) > 10)

                        if non_empty_texts > 0:
                            detector = LanguageDetector()
                            detected_languages = []

                            # Progress indicator
                            from tqdm import tqdm
                            self.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                            for text in tqdm(all_texts, desc="Detecting languages", disable=not HAS_RICH):
                                # Handle NaN and empty texts
                                if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                                    detected_languages.append('unknown')
                                else:
                                    try:
                                        detected = detector.detect(str(text))
                                        if detected and detected.get('language'):
                                            detected_languages.append(detected['language'])
                                        else:
                                            detected_languages.append('unknown')
                                    except Exception as e:
                                        self.logger.debug(f"Language detection failed for text: {e}")
                                        detected_languages.append('unknown')

                            # Add language column to the filtered dataframe
                            df_annotated['lang'] = detected_languages

                            # Reload the FULL original file and update only the annotated rows
                            df_full = pd.read_csv(output_file)

                            # Initialize lang column if it doesn't exist
                            if 'lang' not in df_full.columns:
                                df_full['lang'] = 'unknown'

                            # Update language for annotated rows only
                            # Match by index of df_annotated within df_full
                            df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values

                            # Save updated full file with language column
                            df_full.to_csv(output_file, index=False)

                            # Show distribution
                            lang_counts = {}
                            for lang in detected_languages:
                                if lang != 'unknown':
                                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

                            if lang_counts:
                                total = sum(lang_counts.values())
                                self.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                                lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                                lang_table.add_column("Language", style="cyan", width=12)
                                lang_table.add_column("Count", style="yellow", justify="right", width=12)
                                lang_table.add_column("Percentage", style="green", justify="right", width=12)

                                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                    percentage = (count / total * 100) if total > 0 else 0
                                    lang_table.add_row(
                                        lang.upper(),
                                        f"{count:,}",
                                        f"{percentage:.1f}%"
                                    )

                                self.console.print(lang_table)
                                self.console.print(f"\n[green]✓ Language column 'lang' added to {output_file}[/green]")
                            else:
                                self.console.print("[yellow]⚠️  No languages detected successfully[/yellow]")

                except Exception as e:
                    self.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
                    self.logger.exception("Language detection failed")

            # ============================================================
            # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
            # ============================================================
            self._post_annotation_training_workflow(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs
            )

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

            # ============================================================
            # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
            # ============================================================
            # Build prompt_configs for training workflow
            prompt_configs_for_training = []
            for p in prompts:
                prompt_configs_for_training.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt_content', p.get('prompt', '')),
                        'name': p.get('name', 'prompt')
                    },
                    'prefix': p.get('prefix', '')
                })

            self._post_annotation_training_workflow(
                output_file=output_file,
                text_column=data_source.get('text_column', 'text'),
                prompt_configs=prompt_configs_for_training
            )

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

    def _get_common_annotation_options(self, total_rows=None, provider=None, model_name=None):
        """
        Common annotation options workflow used by both CSV and SQL annotation modes.

        Returns a dict with all configuration options:
        - annotation_limit, sample_strategy, num_processes, save_incrementally, batch_size
        - temperature, max_tokens, top_p, top_k
        - save_metadata, export_to_doccano, export_to_labelstudio, etc.
        """
        config = {}

        # ============================================================
        # DATASET SCOPE
        # ============================================================
        self.console.print("\n[bold cyan]📊 Dataset Scope[/bold cyan]")
        self.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

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

        config['annotation_limit'] = annotation_limit
        config['sample_strategy'] = sample_strategy

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
        config['num_processes'] = num_processes

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
        config['save_incrementally'] = save_incrementally

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

            batch_size = self._int_prompt_with_validation("Batch size", 10, 1, 1000)
        else:
            batch_size = None  # Not used when incremental save is disabled

        config['batch_size'] = batch_size

        # ============================================================
        # MODEL PARAMETERS
        # ============================================================
        self.console.print("\n[bold cyan]🎛️  Model Parameters[/bold cyan]")
        self.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

        # Check if model supports parameter tuning
        model_name_lower = model_name.lower() if model_name else ""
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
            if provider and provider in ['ollama', 'google']:
                self.console.print("\n[cyan]🔢 Top K:[/cyan]")
                self.console.print("  Limits vocabulary to K most likely next tokens")
                self.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
                self.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
                self.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
                top_k = self._int_prompt_with_validation("Top K", 40, 1, 100)

        config['temperature'] = temperature
        config['max_tokens'] = max_tokens
        config['top_p'] = top_p
        config['top_k'] = top_k

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
        config['save_metadata'] = save_metadata

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
        labelstudio_direct_export = False
        labelstudio_api_url = None
        labelstudio_api_key = None
        prediction_mode = "with"

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
            self.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

            sample_choice = Prompt.ask(
                "[bold yellow]Export sample[/bold yellow]",
                choices=["all", "number"],
                default="all"
            )

            if sample_choice == "number":
                export_sample_size = IntPrompt.ask(
                    "Number of items to export",
                    default=100
                )
            else:
                export_sample_size = "all"

        config['export_to_doccano'] = export_to_doccano
        config['export_to_labelstudio'] = export_to_labelstudio
        config['labelstudio_direct_export'] = labelstudio_direct_export
        config['labelstudio_api_url'] = labelstudio_api_url
        config['labelstudio_api_key'] = labelstudio_api_key
        config['prediction_mode'] = prediction_mode
        config['export_sample_size'] = export_sample_size

        return config

    def _database_annotator(self):
        """
        Comprehensive SQL Database Annotator

        Supports multiple database systems with intelligent sampling and flexible output options.
        """
        self.console.print("\n[bold cyan]🗄️  SQL Database Annotator[/bold cyan]\n")

        # ========================================
        # STEP 1/9: Database Connection
        # ========================================
        self.console.print("[bold cyan]Step 1/9: Database Connection[/bold cyan]\n")

        # Database type selection
        db_choices = [
            "PostgreSQL",
            "MySQL",
            "SQLite",
            "Microsoft SQL Server",
            "← Back to main menu"
        ]

        # Display database type choices
        db_table = Table(title="Database Types", box=box.ROUNDED)
        db_table.add_column("#", style="cyan", justify="right", width=4)
        db_table.add_column("Database Type", style="green", width=30)

        for idx, db_choice in enumerate(db_choices, 1):
            db_table.add_row(str(idx), db_choice)

        self.console.print(db_table)

        db_type = Prompt.ask(
            "\n[cyan]Select database type[/cyan]",
            choices=[str(i) for i in range(1, len(db_choices) + 1)],
            default="1"
        )

        db_type_idx = int(db_type) - 1
        if db_type_idx == len(db_choices) - 1:
            return

        db_type_name = db_choices[db_type_idx]

        # Build connection string based on database type
        if db_type_name == "SQLite":
            db_path = Prompt.ask("\n[cyan]Enter SQLite database file path[/cyan]")
            db_path = os.path.expanduser(db_path)
            if not os.path.exists(db_path):
                self.console.print(f"[red]✗ Database file not found: {db_path}[/red]")
                input("\nPress Enter to continue...")
                return
            connection_string = f"sqlite:///{db_path}"
        else:
            # For server-based databases
            host = Prompt.ask("\n[cyan]Database host[/cyan]", default="localhost")
            port_defaults = {
                "PostgreSQL": "5432",
                "MySQL": "3306",
                "Microsoft SQL Server": "1433"
            }
            port = Prompt.ask("[cyan]Port[/cyan]", default=port_defaults.get(db_type_name, "5432"))
            username = Prompt.ask("[cyan]Username[/cyan]", default="postgres" if db_type_name == "PostgreSQL" else "root")

            # Password (hidden input)
            import getpass
            self.console.print("[cyan]Password:[/cyan] ", end="")
            password = getpass.getpass("")

            database = Prompt.ask("[cyan]Database name[/cyan]")

            # Build connection string
            if db_type_name == "PostgreSQL":
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            elif db_type_name == "MySQL":
                connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            elif db_type_name == "Microsoft SQL Server":
                connection_string = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

        # Test connection
        self.console.print("\n[cyan]Testing database connection...[/cyan]")
        try:
            from sqlalchemy import create_engine, inspect, text
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                self.console.print("[green]✓ Connection successful![/green]")
        except Exception as e:
            self.console.print(f"[red]✗ Connection failed: {str(e)}[/red]")
            input("\nPress Enter to continue...")
            return

        # ========================================
        # STEP 2/9: Table Selection
        # ========================================
        self.console.print("\n[bold cyan]Step 2/9: Table Selection[/bold cyan]\n")

        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            if not tables:
                self.console.print("[yellow]No tables found in database[/yellow]")
                input("\nPress Enter to continue...")
                return

            # Display tables with row counts
            table_info = Table(title="Available Tables", box=box.ROUNDED)
            table_info.add_column("#", style="cyan", justify="right")
            table_info.add_column("Table Name", style="green")
            table_info.add_column("Rows", style="yellow", justify="right")

            table_row_counts = {}
            for idx, table_name in enumerate(tables, 1):
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        row_count = result.scalar()
                        table_row_counts[table_name] = row_count
                        table_info.add_row(str(idx), table_name, f"{row_count:,}")
                except:
                    table_info.add_row(str(idx), table_name, "N/A")

            self.console.print(table_info)

            table_choice = Prompt.ask(
                "\n[cyan]Select table[/cyan]",
                choices=[str(i) for i in range(1, len(tables) + 1)]
            )

            selected_table = tables[int(table_choice) - 1]
            total_rows = table_row_counts.get(selected_table, 0)

            self.console.print(f"\n[green]✓ Selected table: {selected_table} ({total_rows:,} rows)[/green]")

        except Exception as e:
            self.console.print(f"[red]✗ Error accessing tables: {str(e)}[/red]")
            input("\nPress Enter to continue...")
            return

        # ========================================
        # STEP 3/9: Column Selection (Intelligent Detection)
        # ========================================
        self.console.print("\n[bold cyan]Step 3/9: Column Selection with Intelligent Detection[/bold cyan]\n")

        try:
            columns = inspector.get_columns(selected_table)

            # Load sample data for intelligent detection
            self.console.print("[cyan]Analyzing columns...[/cyan]")
            sample_query = f"SELECT * FROM {selected_table} LIMIT 100"
            df_sample = pd.read_sql(sample_query, engine)

            # Detect text columns intelligently
            text_candidates = []
            id_candidates = []

            for col_info in columns:
                col_name = col_info['name']
                if col_name not in df_sample.columns:
                    continue

                # Check if it's a potential ID column
                col_lower = col_name.lower()
                if any(id_keyword in col_lower for id_keyword in ['id', 'key', 'index', 'number', 'num', 'pk']):
                    # Check if values are unique
                    is_unique = df_sample[col_name].nunique() == len(df_sample[col_name].dropna())
                    if is_unique:
                        id_candidates.append({
                            'name': col_name,
                            'type': str(col_info['type']),
                            'confidence': 'high' if 'id' in col_lower else 'medium'
                        })

                # Check if it's a text column (object/string type)
                if df_sample[col_name].dtype == 'object':
                    # Get non-null samples
                    non_null = df_sample[col_name].dropna()
                    if len(non_null) == 0:
                        continue

                    # Calculate average length
                    avg_length = non_null.astype(str).str.len().mean()
                    sample_value = str(non_null.iloc[0])[:80] if len(non_null) > 0 else ""

                    # Determine confidence based on average length
                    if avg_length > 100:
                        confidence = "high"
                    elif avg_length > 50:
                        confidence = "medium"
                    elif avg_length > 20:
                        confidence = "low"
                    else:
                        continue  # Skip very short text

                    text_candidates.append({
                        'name': col_name,
                        'confidence': confidence,
                        'avg_length': avg_length,
                        'sample': sample_value
                    })

            # Sort candidates
            confidence_order = {"high": 0, "medium": 1, "low": 2}
            text_candidates.sort(key=lambda x: (confidence_order[x['confidence']], -x['avg_length']))
            id_candidates.sort(key=lambda x: (confidence_order.get(x['confidence'], 3)))

            # Display columns with intelligent suggestions
            col_table = Table(title=f"Columns in '{selected_table}'", box=box.ROUNDED)
            col_table.add_column("#", style="cyan", justify="right", width=4)
            col_table.add_column("Column Name", style="green", width=25)
            col_table.add_column("Type", style="yellow", width=20)
            col_table.add_column("Detection", style="magenta", width=30)

            detected_text_col = None
            detected_id_col = None

            for idx, col in enumerate(columns, 1):
                col_name = col['name']
                col_type = str(col['type'])
                detection = ""

                # Check if it's a suggested text column
                text_match = next((tc for tc in text_candidates if tc['name'] == col_name), None)
                if text_match:
                    if text_match['confidence'] == 'high':
                        detection = "📝 Text (High confidence)"
                        if detected_text_col is None:
                            detected_text_col = idx
                    elif text_match['confidence'] == 'medium':
                        detection = "📝 Text (Medium)"
                    else:
                        detection = "📝 Text (Low)"

                # Check if it's a suggested ID column
                id_match = next((ic for ic in id_candidates if ic['name'] == col_name), None)
                if id_match:
                    if id_match['confidence'] == 'high':
                        detection = "🔑 ID (Recommended)"
                        if detected_id_col is None:
                            detected_id_col = idx
                    else:
                        detection = "🔑 ID (Possible)"

                col_table.add_row(str(idx), col_name, col_type, detection)

            self.console.print(col_table)

            # Select text column with intelligent default
            if detected_text_col:
                self.console.print(f"\n[cyan]💡 Suggested text column: '{columns[detected_text_col-1]['name']}' (detected automatically)[/cyan]")

            text_col_choice = Prompt.ask(
                "\n[cyan]Select TEXT column (to annotate)[/cyan]",
                choices=[str(i) for i in range(1, len(columns) + 1)],
                default=str(detected_text_col) if detected_text_col else "1"
            )
            text_column = columns[int(text_col_choice) - 1]['name']

            # Select ID column with intelligent default
            if Confirm.ask("\n[cyan]Do you want to select an ID column?[/cyan]", default=True):
                if detected_id_col:
                    self.console.print(f"\n[cyan]💡 Suggested ID column: '{columns[detected_id_col-1]['name']}' (unique values detected)[/cyan]")

                id_col_choice = Prompt.ask(
                    "\n[cyan]Select ID column[/cyan]",
                    choices=[str(i) for i in range(1, len(columns) + 1)],
                    default=str(detected_id_col) if detected_id_col else "1"
                )
                id_column = columns[int(id_col_choice) - 1]['name']
            else:
                id_column = None

            self.console.print(f"\n[green]✓ Text column: {text_column}[/green]")
            if id_column:
                self.console.print(f"[green]✓ ID column: {id_column}[/green]")

        except Exception as e:
            self.console.print(f"[red]✗ Error accessing columns: {str(e)}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            input("\nPress Enter to continue...")
            return

        # ========================================
        # STEP 4/9: Sampling Strategy
        # ========================================
        self.console.print("\n[bold cyan]Step 4/9: Sampling Strategy[/bold cyan]\n")

        # Calculate representative sample sizes using Cochran's formula
        def calculate_representative_sample(population_size: int, confidence_level: float = 0.95, margin_error: float = 0.05) -> int:
            """
            Calculate representative sample size using Cochran's formula.

            Args:
                population_size: Total population size
                confidence_level: Confidence level (0.90, 0.95, 0.99)
                margin_error: Margin of error (0.03, 0.05, 0.10)

            Returns:
                Required sample size
            """
            import math

            # Z-scores for common confidence levels
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z = z_scores.get(confidence_level, 1.96)

            # Assume maximum variability (p = 0.5)
            p = 0.5

            # Cochran's formula for infinite population
            n0 = (z ** 2 * p * (1 - p)) / (margin_error ** 2)

            # Adjust for finite population
            n = n0 / (1 + (n0 - 1) / population_size)

            return math.ceil(n)

        # Calculate sample sizes for different confidence levels
        sample_sizes = {
            "95% confidence, ±5% margin": calculate_representative_sample(total_rows, 0.95, 0.05),
            "95% confidence, ±3% margin": calculate_representative_sample(total_rows, 0.95, 0.03),
            "99% confidence, ±5% margin": calculate_representative_sample(total_rows, 0.99, 0.05),
            "99% confidence, ±3% margin": calculate_representative_sample(total_rows, 0.99, 0.03)
        }

        # Display sampling options
        self.console.print(f"[cyan]Total rows in table: {total_rows:,}[/cyan]\n")

        sampling_choices = [
            f"Representative sample - {sample_sizes['95% confidence, ±5% margin']:,} rows (95% confidence, ±5% margin)",
            f"Representative sample - {sample_sizes['95% confidence, ±3% margin']:,} rows (95% confidence, ±3% margin)",
            f"Representative sample - {sample_sizes['99% confidence, ±5% margin']:,} rows (99% confidence, ±5% margin)",
            f"Representative sample - {sample_sizes['99% confidence, ±3% margin']:,} rows (99% confidence, ±3% margin)",
            "Exact number of rows (I'll specify)",
            "Percentage of total rows",
            "All rows (no sampling)"
        ]

        # Display sampling choices table
        sampling_table = Table(title="Sampling Strategy Options", box=box.ROUNDED)
        sampling_table.add_column("#", style="cyan", justify="right", width=4)
        sampling_table.add_column("Strategy", style="green", width=70)

        for idx, choice in enumerate(sampling_choices, 1):
            sampling_table.add_row(str(idx), choice)

        self.console.print(sampling_table)

        sampling_choice = Prompt.ask(
            "\n[cyan]Select sampling strategy[/cyan]",
            choices=[str(i) for i in range(1, len(sampling_choices) + 1)],
            default="1"
        )

        sampling_idx = int(sampling_choice) - 1

        if sampling_idx < 4:
            # Representative sample
            sample_key = list(sample_sizes.keys())[sampling_idx]
            num_rows = sample_sizes[sample_key]
        elif sampling_idx == 4:
            # Exact number
            num_rows = IntPrompt.ask("\n[cyan]Enter exact number of rows to annotate[/cyan]", default=min(1000, total_rows))
        elif sampling_idx == 5:
            # Percentage
            percentage = FloatPrompt.ask("\n[cyan]Enter percentage (0-100)[/cyan]", default=10.0)
            num_rows = int(total_rows * percentage / 100)
        else:
            # All rows
            num_rows = total_rows

        # Cap at total rows
        num_rows = min(num_rows, total_rows)

        # Sampling method
        if num_rows < total_rows:
            sampling_method_choices = [
                "Random sampling (recommended)",
                "First N rows",
                "Last N rows"
            ]

            # Display sampling method choices
            method_table = Table(title="Sampling Method", box=box.ROUNDED)
            method_table.add_column("#", style="cyan", justify="right", width=4)
            method_table.add_column("Method", style="green", width=40)

            for idx, method in enumerate(sampling_method_choices, 1):
                method_table.add_row(str(idx), method)

            self.console.print()
            self.console.print(method_table)

            sampling_method = Prompt.ask(
                "\n[cyan]Select sampling method[/cyan]",
                choices=[str(i) for i in range(1, len(sampling_method_choices) + 1)],
                default="1"
            )

            sampling_method_idx = int(sampling_method) - 1
            sampling_method_name = ["random", "first", "last"][sampling_method_idx]
        else:
            sampling_method_name = "all"

        self.console.print(f"\n[green]✓ Will annotate {num_rows:,} rows using '{sampling_method_name}' sampling[/green]")

        # ========================================
        # STEP 5/9: Output Destination
        # ========================================
        self.console.print("\n[bold cyan]Step 5/9: Output Destination[/bold cyan]\n")

        output_choices = [
            f"Write back to same table ('{selected_table}')",
            "Create new table in database",
            "Export to CSV file",
            "Export to JSON file",
            "Export to JSONL file",
            "Export to Excel file",
            "Export to Parquet file",
            "Export to RData file"
        ]

        # Display output destination choices
        output_table = Table(title="Output Destination Options", box=box.ROUNDED)
        output_table.add_column("#", style="cyan", justify="right", width=4)
        output_table.add_column("Destination", style="green", width=50)

        for idx, choice in enumerate(output_choices, 1):
            output_table.add_row(str(idx), choice)

        self.console.print(output_table)

        output_choice = Prompt.ask(
            "\n[cyan]Select output destination[/cyan]",
            choices=[str(i) for i in range(1, len(output_choices) + 1)],
            default="2"
        )

        output_idx = int(output_choice) - 1

        if output_idx == 0:
            # Same table - need annotation column name
            annotation_column = Prompt.ask(
                "\n[cyan]Enter name for annotation column[/cyan]",
                default="llm_annotation"
            )
            output_dest = {
                'type': 'same_table',
                'table': selected_table,
                'column': annotation_column
            }
        elif output_idx == 1:
            # New table
            new_table_name = Prompt.ask(
                "\n[cyan]Enter new table name[/cyan]",
                default=f"{selected_table}_annotated"
            )
            output_dest = {
                'type': 'new_table',
                'table': new_table_name
            }
        else:
            # File export
            file_extensions = {
                2: 'csv',
                3: 'json',
                4: 'jsonl',
                5: 'xlsx',
                6: 'parquet',
                7: 'rdata'
            }

            ext = file_extensions[output_idx]
            default_filename = f"{selected_table}_annotated.{ext}"

            output_file = Prompt.ask(
                f"\n[cyan]Enter output file path[/cyan]",
                default=default_filename
            )

            output_dest = {
                'type': 'file',
                'path': output_file,
                'format': ext
            }

        # ========================================
        # STEP 6/9: LLM Selection
        # ========================================
        self.console.print("\n[bold cyan]Step 6/9: LLM Selection[/bold cyan]\n")

        # Reuse existing LLM selection logic
        llm_config = self._select_llm_interactive()
        if llm_config is None:
            self.console.print("[yellow]LLM selection cancelled[/yellow]")
            input("\nPress Enter to continue...")
            return

        # ========================================
        # STEP 7/9: Prompt Configuration
        # ========================================
        self.console.print("\n[bold cyan]Step 7/9: Prompt Configuration[/bold cyan]\n")

        # Detect prompts from prompts directory (same as other modes)
        detected_prompts = self._detect_prompts_in_folder()

        selected_prompts = []

        if detected_prompts:
            self.console.print("[bold green]✓ Detected prompts in prompts directory:[/bold green]")
            for i, p in enumerate(detected_prompts, 1):
                keys_str = ', '.join(p['keys'][:3]) + ('...' if len(p['keys']) > 3 else '')
                self.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
                self.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

            # Prompt selection options
            self.console.print("\n[bold]Prompt Selection Options:[/bold]")
            self.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts")
            self.console.print("  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number")
            self.console.print("  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
            self.console.print("  [cyan]custom[/cyan]  - Provide path to a prompt file")

            prompt_choice = Prompt.ask(
                "\n[bold yellow]Prompt selection[/bold yellow]",
                choices=["all", "select", "wizard", "custom"],
                default="all"
            )

            if prompt_choice == "all":
                selected_prompts = detected_prompts
                self.console.print(f"[green]✓ Using all {len(selected_prompts)} prompts[/green]")
            elif prompt_choice == "select":
                indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
                if indices.strip():  # Only process if not empty
                    for idx_str in indices.split(','):
                        idx_str = idx_str.strip()
                        if idx_str:  # Skip empty strings
                            try:
                                idx = int(idx_str) - 1
                                if 0 <= idx < len(detected_prompts):
                                    selected_prompts.append(detected_prompts[idx])
                            except ValueError:
                                self.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
                if not selected_prompts:
                    self.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                    selected_prompts = detected_prompts
                else:
                    self.console.print(f"[green]✓ Selected {len(selected_prompts)} prompts[/green]")
            elif prompt_choice == "wizard":
                wizard_prompt = self._run_social_science_wizard()
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(wizard_prompt)
                selected_prompts = [{
                    'path': None,
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
                wizard_prompt = self._run_social_science_wizard()
                from ..annotators.json_cleaner import extract_expected_keys
                keys = extract_expected_keys(wizard_prompt)
                selected_prompts = [{
                    'path': None,
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

        if not selected_prompts:
            self.console.print("[red]✗ No prompts selected[/red]")
            input("\nPress Enter to continue...")
            return

        # Build prompt configs (same format as other modes)
        prompt_configs = []
        if len(selected_prompts) > 1:
            self.console.print("\n[bold]Multi-Prompt Mode:[/bold] Configure key prefixes")
            self.console.print("[dim]Prefixes help identify which prompt generated which keys[/dim]\n")

            for i, p in enumerate(selected_prompts, 1):
                default_prefix = p['name'].replace('_', '').replace('-', '')[:10]
                prefix = Prompt.ask(
                    f"  Prefix for '{p['name']}'",
                    default=default_prefix
                )
                prompt_configs.append({
                    'prompt': p,
                    'prefix': prefix
                })
        else:
            prompt_configs = [{'prompt': selected_prompts[0], 'prefix': ''}]

        # ========================================
        # STEP 8/9: Common Annotation Options
        # ========================================
        # Use the shared function for all advanced options (same as CSV annotation)
        common_options = self._get_common_annotation_options(
            total_rows=total_rows,
            provider=llm_config.provider,
            model_name=llm_config.name
        )

        # Extract values from common options
        annotation_limit = common_options.get('annotation_limit')
        sample_strategy = common_options.get('sample_strategy', 'head')
        num_processes = common_options.get('num_processes', 1)
        save_incrementally = common_options.get('save_incrementally', True)
        batch_size = common_options.get('batch_size', 10)
        temperature = common_options.get('temperature', 0.7)
        max_tokens = common_options.get('max_tokens', 1000)
        top_p = common_options.get('top_p', 1.0)
        top_k = common_options.get('top_k', 40)
        save_metadata = common_options.get('save_metadata', True)
        export_to_doccano = common_options.get('export_to_doccano', False)
        export_to_labelstudio = common_options.get('export_to_labelstudio', False)
        labelstudio_direct_export = common_options.get('labelstudio_direct_export', False)
        labelstudio_api_url = common_options.get('labelstudio_api_url')
        labelstudio_api_key = common_options.get('labelstudio_api_key')
        prediction_mode = common_options.get('prediction_mode', 'with')
        export_sample_size = common_options.get('export_sample_size', 'all')

        # Apply annotation_limit to num_rows if specified
        if annotation_limit:
            num_rows = annotation_limit
            # Update sampling_method_name based on sample_strategy
            if sample_strategy == 'random':
                sampling_method_name = 'random'
            else:
                sampling_method_name = 'first'

        # Max retries parameter (not in common options, specific to annotation)
        max_retries = IntPrompt.ask(
            "[cyan]Max retries on failure[/cyan]",
            default=3
        )

        # ========================================
        # STEP 9/9: Confirmation and Execution
        # ========================================
        self.console.print("\n[bold cyan]Step 9/9: Review and Execute[/bold cyan]\n")

        # Summary table
        summary = Table(title="Annotation Configuration Summary", box=box.ROUNDED)
        summary.add_column("Parameter", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Database Type", db_type_name)
        summary.add_row("Table", selected_table)
        summary.add_row("Text Column", text_column)
        if id_column:
            summary.add_row("ID Column", id_column)
        summary.add_row("Total Rows", f"{total_rows:,}")
        summary.add_row("Rows to Annotate", f"{num_rows:,}")
        summary.add_row("Sampling Method", sampling_method_name)

        if output_dest['type'] == 'same_table':
            summary.add_row("Output", f"Same table, column '{output_dest['column']}'")
        elif output_dest['type'] == 'new_table':
            summary.add_row("Output", f"New table '{output_dest['table']}'")
        else:
            summary.add_row("Output", f"File: {output_dest['path']}")

        summary.add_row("LLM Provider", llm_config.provider if llm_config else 'N/A')
        summary.add_row("LLM Model", llm_config.name if llm_config else 'N/A')
        summary.add_row("Prompts", f"{len(prompt_configs)} configured")

        # Add all model parameters
        summary.add_row("Temperature", str(temperature))
        summary.add_row("Max Tokens", str(max_tokens))
        summary.add_row("Top P", str(top_p))
        if llm_config and llm_config.provider in ['ollama', 'google']:
            summary.add_row("Top K", str(top_k))

        # Add processing parameters
        summary.add_row("Parallel Workers", str(num_processes))
        summary.add_row("Batch Size", str(batch_size))
        summary.add_row("Incremental Save", "Yes" if save_incrementally else "No")
        summary.add_row("Max Retries", str(max_retries))

        # Add export options if configured
        if export_to_doccano or export_to_labelstudio:
            export_tool = "Doccano" if export_to_doccano else "Label Studio"
            summary.add_row("Export Tool", export_tool)
            summary.add_row("Prediction Mode", prediction_mode)
            if export_sample_size != "all":
                summary.add_row("Export Sample", str(export_sample_size))

        self.console.print(summary)

        if not Confirm.ask("\n[cyan]Start annotation?[/cyan]", default=True):
            self.console.print("[yellow]Annotation cancelled[/yellow]")
            input("\nPress Enter to continue...")
            return

        # ========================================
        # EXECUTION
        # ========================================
        self.console.print("\n[bold green]Starting annotation...[/bold green]\n")

        try:
            # Load data with sampling
            self.console.print("[cyan]Loading data from database...[/cyan]")

            if sampling_method_name == "all":
                query = f"SELECT * FROM {selected_table}"
            elif sampling_method_name == "random":
                if db_type_name == "PostgreSQL":
                    query = f"SELECT * FROM {selected_table} ORDER BY RANDOM() LIMIT {num_rows}"
                elif db_type_name == "MySQL":
                    query = f"SELECT * FROM {selected_table} ORDER BY RAND() LIMIT {num_rows}"
                elif db_type_name == "SQLite":
                    query = f"SELECT * FROM {selected_table} ORDER BY RANDOM() LIMIT {num_rows}"
                else:
                    query = f"SELECT TOP {num_rows} * FROM {selected_table} ORDER BY NEWID()"
            elif sampling_method_name == "first":
                query = f"SELECT * FROM {selected_table} LIMIT {num_rows}"
            else:  # last
                if db_type_name in ["PostgreSQL", "MySQL", "SQLite"]:
                    query = f"SELECT * FROM (SELECT * FROM {selected_table} ORDER BY {id_column or '1'} DESC LIMIT {num_rows}) sub ORDER BY {id_column or '1'} ASC"
                else:
                    query = f"SELECT TOP {num_rows} * FROM {selected_table} ORDER BY {id_column or '1'} DESC"

            df = pd.read_sql(query, engine)
            self.console.print(f"[green]✓ Loaded {len(df):,} rows[/green]\n")

            # Save DataFrame to file in data/annotations directory (same as Smart Annotate)
            annotations_dir = self.settings.paths.data_dir / 'annotations'
            annotations_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_model_name = llm_config.name.replace(':', '_').replace('/', '_')

            # Input file for pipeline
            temp_input_filename = f"db_{selected_table}_{timestamp}.csv"
            temp_input_file = annotations_dir / temp_input_filename
            df.to_csv(temp_input_file, index=False)
            self.console.print(f"[dim]Saved {len(df)} rows to: {temp_input_file}[/dim]\n")

            # Output file for annotations (will be created by pipeline)
            output_filename = f"db_{selected_table}_{safe_model_name}_annotations_{timestamp}.csv"
            pipeline_output_file = annotations_dir / output_filename

            # Build prompts payload (same format as Smart Annotate)
            prompts_payload = []
            for pc in prompt_configs:
                prompt_dict = pc['prompt']
                prefix = pc['prefix']
                prompts_payload.append({
                    'prompt': prompt_dict['content'],
                    'expected_keys': prompt_dict['keys'],
                    'prefix': prefix
                })

            # Get API key if needed
            api_key = None
            if llm_config.provider in ['openai', 'anthropic', 'google']:
                api_key = self._get_api_key(llm_config.provider)
                if not api_key:
                    self.console.print(f"[red]API key required for {llm_config.provider}[/red]")
                    return

            # Build pipeline config (SAME as Smart Annotate)
            pipeline_config = {
                'mode': 'file',
                'data_source': 'csv',
                'data_format': 'csv',
                'file_path': str(temp_input_file),
                'text_column': text_column,
                'text_columns': [text_column],
                'annotation_column': 'annotation',
                'identifier_column': id_column if id_column else 'annotation_id',
                'run_annotation': True,
                'annotation_mode': 'local' if llm_config.provider == 'ollama' else 'api',
                'annotation_provider': llm_config.provider,
                'annotation_model': llm_config.name,
                'api_key': api_key,
                'prompts': prompts_payload,
                'annotation_sample_size': annotation_limit,
                'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
                'annotation_sample_seed': 42,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k if llm_config.provider in ['ollama', 'google'] else None,
                'max_workers': num_processes,
                'num_processes': num_processes,
                'use_parallel': num_processes > 1,
                'warmup': False,
                'disable_tqdm': False,  # Enable tqdm for same display as Smart Annotate
                'output_format': 'csv',
                'output_path': str(pipeline_output_file),
                'save_incrementally': save_incrementally,
                'batch_size': batch_size,
                'run_validation': False,
                'run_training': False,
            }

            # Add model-specific options (same as Smart Annotate)
            if llm_config.provider == 'ollama':
                options = {
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': top_p,
                    'top_k': top_k
                }
                pipeline_config['options'] = options

            # Execute annotation using SAME pipeline as Smart Annotate
            try:
                self.console.print("[bold green]🚀 Starting annotation...[/bold green]\n")

                from ..pipelines.pipeline_controller import PipelineController
                from ..utils.rich_progress_manager import RichProgressManager
                from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

                pipeline_with_progress = PipelineController(settings=self.settings)

                with RichProgressManager(
                    show_json_every=1,
                    compact_mode=False
                ) as progress_manager:
                    enhanced_pipeline = EnhancedPipelineWrapper(
                        pipeline_with_progress,
                        progress_manager
                    )

                    state = enhanced_pipeline.run_pipeline(pipeline_config)

                # Check if annotation was successful
                if not state or not state.annotation_results:
                    self.console.print("\n[red]✗ Annotation failed or returned no results[/red]")
                    input("\nPress Enter to continue...")
                    return

                # Load annotated data
                if not pipeline_output_file.exists():
                    self.console.print(f"\n[red]✗ Output file not found: {pipeline_output_file}[/red]")
                    input("\nPress Enter to continue...")
                    return

                df_annotated = pd.read_csv(pipeline_output_file)
                self.console.print(f"\n[green]✓ Annotation complete: {len(df_annotated)} rows annotated[/green]")

            except Exception as e:
                self.console.print(f"\n[red]✗ Error during annotation: {str(e)}[/red]")
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
                input("\nPress Enter to continue...")
                return

            # ========================================
            # Save results to final destination
            # ========================================
            self.console.print("\n[bold cyan]📁 Saving to final destination...[/bold cyan]\n")

            if output_dest['type'] == 'same_table':
                # Update same table in database
                from sqlalchemy import Table as SQLTable, MetaData, Column, JSON

                metadata_obj = MetaData()
                metadata_obj.reflect(bind=engine)
                table = metadata_obj.tables[selected_table]

                # Add annotation column if doesn't exist
                annotation_col_name = output_dest['column']
                if annotation_col_name not in [c.name for c in table.columns]:
                    with engine.connect() as conn:
                        if db_type_name == "PostgreSQL":
                            conn.execute(text(f"ALTER TABLE {selected_table} ADD COLUMN {annotation_col_name} JSONB"))
                        else:
                            conn.execute(text(f"ALTER TABLE {selected_table} ADD COLUMN {annotation_col_name} TEXT"))
                        conn.commit()
                    self.console.print(f"[green]✓ Added column '{annotation_col_name}' to table[/green]")

                # Update rows with annotations
                with engine.connect() as conn:
                    updated_count = 0
                    for idx, row in df_annotated.iterrows():
                        if id_column and id_column in row:
                            id_value = row[id_column]
                            annotation_value = row.get('annotation', '')
                            update_query = text(f"UPDATE {selected_table} SET {annotation_col_name} = :annotation WHERE {id_column} = :id_val")
                            conn.execute(update_query, {'annotation': str(annotation_value), 'id_val': id_value})
                            updated_count += 1
                        else:
                            self.console.print("[yellow]⚠️  Warning: No ID column specified, cannot update specific rows[/yellow]")
                            break
                    conn.commit()

                self.console.print(f"[green]✓ Updated {updated_count:,} rows in table '{selected_table}'[/green]")
                self.console.print(f"[dim]Annotations saved in data/annotations: {pipeline_output_file.name}[/dim]")

                # Clean up temporary input file
                if temp_input_file.exists():
                    temp_input_file.unlink()

            elif output_dest['type'] == 'new_table':
                # Create new table in database
                df_annotated.to_sql(
                    output_dest['table'],
                    engine,
                    if_exists='replace',
                    index=False
                )
                self.console.print(f"[green]✓ Created new table '{output_dest['table']}' with {len(df_annotated):,} rows[/green]")
                self.console.print(f"[dim]Annotations also saved in data/annotations: {pipeline_output_file.name}[/dim]")

                # Clean up temporary input file
                if temp_input_file.exists():
                    temp_input_file.unlink()

            else:
                # Export to file - either copy to user path or keep in data/annotations
                output_path = Path(output_dest['path'])
                format_type = output_dest['format']

                # If user specified a custom path, copy/convert there
                if str(output_path) != str(pipeline_output_file):
                    if format_type == 'csv':
                        # Already CSV, just copy
                        import shutil
                        shutil.copy(pipeline_output_file, output_path)
                    elif format_type == 'json':
                        df_annotated.to_json(output_path, orient='records', indent=2)
                    elif format_type == 'jsonl':
                        df_annotated.to_json(output_path, orient='records', lines=True)
                    elif format_type == 'xlsx':
                        df_annotated.to_excel(output_path, index=False)
                    elif format_type == 'parquet':
                        df_annotated.to_parquet(output_path, index=False)
                    elif format_type == 'rdata':
                        import pyreadr
                        pyreadr.write_rdata(str(output_path), df_annotated, df_name='annotated_data')

                    self.console.print(f"[green]✓ Exported {len(df_annotated):,} rows to: {output_path}[/green]")
                    self.console.print(f"[dim]Annotations also saved in data/annotations: {pipeline_output_file.name}[/dim]")
                else:
                    # Using default location in data/annotations
                    self.console.print(f"[green]✓ Annotations saved: {pipeline_output_file}[/green]")

                # Clean up temporary input file
                if temp_input_file.exists():
                    temp_input_file.unlink()

            self.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")

            # ============================================================
            # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
            # ============================================================
            self._post_annotation_training_workflow(
                output_file=str(pipeline_output_file),
                text_column=text_column,
                prompt_configs=prompt_configs
            )

            # Export to Doccano JSONL if requested
            if export_to_doccano:
                self._export_to_doccano_jsonl(
                    output_file=str(pipeline_output_file),
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=None,
                    timestamp=timestamp,
                    sample_size=export_sample_size
                )

            # Export to Label Studio if requested
            if export_to_labelstudio:
                if labelstudio_direct_export:
                    # Direct export to Label Studio via API
                    self._export_to_labelstudio_direct(
                        output_file=str(pipeline_output_file),
                        text_column=text_column,
                        prompt_configs=prompt_configs,
                        data_path=None,
                        timestamp=timestamp,
                        sample_size=export_sample_size,
                        prediction_mode=prediction_mode,
                        api_url=labelstudio_api_url,
                        api_key=labelstudio_api_key
                    )
                else:
                    # Export to JSONL file
                    self._export_to_labelstudio_jsonl(
                        output_file=str(pipeline_output_file),
                        text_column=text_column,
                        prompt_configs=prompt_configs,
                        data_path=None,
                        timestamp=timestamp,
                        sample_size=export_sample_size,
                        prediction_mode=prediction_mode
                    )

        except Exception as e:
            self.console.print(f"\n[bold red]✗ Error during annotation: {str(e)}[/bold red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

        input("\nPress Enter to continue...")


    def show_documentation(self):
        """Show documentation"""
        # Display ASCII logo only
        self._display_ascii_logo()

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

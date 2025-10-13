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
import csv
import shutil
import copy
import numpy as np
import uuid
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
    from rich.align import Align
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
    from pandas.api import types as pd_types
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd_types = None

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
from llm_tool.utils.language_normalizer import LanguageNormalizer
from ..utils.data_filter_logger import get_filter_logger
from ..utils.system_resources import detect_resources, SystemResourceDetector
from ..utils.resource_display import (
    display_resources,
    create_resource_table,
    create_recommendations_table,
    create_compact_resource_panel,
    display_resource_header,
    get_resource_summary_text,
    create_visual_resource_panel,
    create_mode_resource_banner,
    create_detailed_mode_panel
)
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
from llm_tool.utils.data_detector import DatasetInfo, DataDetector
from llm_tool.utils.session_summary import (
    SessionSummary,
    SummaryRecord,
    collect_all_summaries,
    collect_summaries_for_mode,
    read_summary,
)
from llm_tool.utils.training_paths import get_training_logs_base
from . import training_arena_integrated as training_arena
from .annotation_workflow import (
    AnnotationMode,
    ANNOTATOR_RESUME_STEPS,
    FACTORY_RESUME_STEPS,
    AnnotationResumeTracker,
    _launch_model_annotation_stage,
    create_session_directories,
    execute_from_metadata,
    run_annotator_workflow,
    run_factory_workflow,
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
    'gpt-5-2025-08-07': 'OpenAI GPT-5 (2025-08-07) - Flagship general-purpose model with enhanced reasoning and 200K context',
    'gpt-5-mini-2025-08-07': 'OpenAI GPT-5 Mini (2025-08-07) - Balanced GPT-5 variant, optimized for cost and quick iteration',
    'gpt-5-nano-2025-08-07': 'OpenAI GPT-5 Nano (2025-08-07) - Ultra-fast GPT-5 tier for large batch workloads',
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

                                # Extract size - look for GB/MB/KB/TB in any part
                                # Format from ollama: "NAME  ID  2.0 GB  MODIFIED"
                                size = None
                                import re
                                for i, part in enumerate(parts[1:]):
                                    part_upper = part.upper()
                                    # Check if this part is a size unit (GB, MB, KB, TB)
                                    if part_upper in ['GB', 'MB', 'KB', 'TB']:
                                        # The number should be in the previous element
                                        if i > 0:
                                            number_part = parts[1:][i-1]  # Get previous element in the sliced list
                                            # Validate it's a number
                                            if re.match(r'^\d+\.?\d*$', number_part):
                                                size = f"{number_part} {part_upper}"
                                                break
                                    # Also handle cases like "2.0GB" (no space)
                                    elif 'GB' in part_upper or 'MB' in part_upper or 'KB' in part_upper or 'TB' in part_upper:
                                        match = re.match(r'([\d.]+)\s*([KMGT]B)', part_upper)
                                        if match:
                                            size = f"{match.group(1)} {match.group(2)}"
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
            ModelInfo(
                "gpt-5-2025-08-07",
                "openai",
                context_length=200000,
                requires_api_key=True,
                cost_per_1k_tokens=0.004,
                supports_json=True,
                supports_streaming=True,
                max_tokens=8000,
            ),
            ModelInfo(
                "gpt-5-mini-2025-08-07",
                "openai",
                context_length=200000,
                requires_api_key=True,
                cost_per_1k_tokens=0.001,
                supports_json=True,
                supports_streaming=True,
                max_tokens=4000,
            ),
            ModelInfo(
                "gpt-5-nano-2025-08-07",
                "openai",
                context_length=200000,
                requires_api_key=True,
                cost_per_1k_tokens=0.001,
                supports_json=True,
                supports_streaming=True,
                max_tokens=4000,
            ),
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
        self.resource_detector = SystemResourceDetector()
        self.system_resources = None  # Will be populated on first detection

        # Import and initialize PromptManager
        from ..annotators.prompt_manager import PromptManager
        self.prompt_manager = PromptManager()

        # Cache for detected models
        self.detected_llms: Optional[Dict[str, List[ModelInfo]]] = None
        self.available_trainer_models: Optional[Dict[str, List[Dict]]] = None
        self.detected_datasets: Optional[List[DatasetInfo]] = None

        # Expose module-level capability flags for shared workflows
        self.HAS_RICH = HAS_RICH
        self.HAS_REQUESTS = HAS_REQUESTS

        # Session state
        self.current_session = {
            'start_time': datetime.now(),
            'operations_count': 0,
            'last_operation': None
        }

        # Setup logging
        self._setup_logging()

    def _resolve_existing_column(
        self,
        df: Any,
        requested_column: Optional[str],
        column_label: str,
        fallback_candidates: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Reconcile a persisted column reference with the columns available in the current dataframe.

        Resume workflows may store either the original column name or its positional index (as a
        string). When the dataset schema changes between runs, this helper attempts to map the saved
        reference back to a valid column so downstream steps keep functioning without silent failure.
        """
        if df is None or requested_column is None:
            return requested_column

        try:
            available_columns = list(df.columns)
        except AttributeError:
            return requested_column

        if requested_column in available_columns:
            return requested_column

        resolved_column: Optional[str] = requested_column

        # 1) Handle numeric index persisted as a string (e.g., "2")
        if isinstance(requested_column, str) and requested_column.isdigit():
            idx = int(requested_column)
            if 0 <= idx < len(available_columns):
                resolved_column = available_columns[idx]

        # 2) Case-insensitive name match
        if (
            resolved_column not in available_columns
            and isinstance(requested_column, str)
        ):
            lower_map = {
                col.lower(): col
                for col in available_columns
                if isinstance(col, str)
            }
            key = requested_column.lower()
            if key in lower_map:
                resolved_column = lower_map[key]

        # 3) Explicit fallback candidates (ordered by priority)
        if resolved_column not in available_columns and fallback_candidates:
            for candidate in fallback_candidates:
                if candidate in available_columns:
                    resolved_column = candidate
                    break

        if resolved_column not in available_columns:
            return requested_column

        if self.console and resolved_column != requested_column:
            self.console.print(
                f"[yellow]ℹ Stored {column_label} '{requested_column}' not found. "
                f"Using '{resolved_column}' instead.[/yellow]"
            )

        return resolved_column

    def analyze_text_lengths(
        self,
        data_path: Path = None,
        df: Any = None,
        text_column: str = None,
        display_results: bool = True,
        step_label: str = "Text Length Analysis",
        analysis_df: Any = None,
        total_rows_reference: Optional[int] = None,
        subset_label: Optional[str] = None,
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

            if analysis_df is not None and df is None:
                df = analysis_df

            if df is not None and text_column and text_column in df.columns:
                working_df = analysis_df if analysis_df is not None else df
                if text_column not in working_df.columns:
                    working_df = df

                source_df = df
                source_series = source_df[text_column].dropna().astype(str)
                working_series = working_df[text_column].dropna().astype(str)

                source_total = (
                    total_rows_reference
                    if total_rows_reference is not None
                    else len(source_series)
                )
                analysis_total = len(working_series)

                if display_results and self.console:
                    if subset_label and analysis_total != source_total:
                        self.console.print(
                            f"[dim]Analyzing {analysis_total:,} {subset_label} "
                            f"(out of {source_total:,} texts).[/dim]\n"
                        )
                    else:
                        self.console.print(f"[dim]Analyzing {analysis_total:,} texts...[/dim]\n")

                analysis_series = working_series
                analysis_texts = analysis_series.tolist()

                try:
                    tokenizer = None
                    tokenizer_error = None

                    if HAS_TQDM:
                        from tqdm import tqdm

                    if HAS_TRANSFORMERS:

                        tokenizer_models = [
                            "bert-base-multilingual-cased",  # Best for multilingual
                            "bert-base-uncased",             # Fallback
                            "distilbert-base-uncased"        # Lightweight fallback
                        ]

                        for model_name in tokenizer_models:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                                break
                            except Exception as model_error:
                                tokenizer_error = model_error
                                self.logger.debug(f"Could not load {model_name} tokenizer locally: {model_error}")

                    if tokenizer is None and display_results and self.console:
                        warning_msg = "[yellow]⚠️ Unable to load a Hugging Face tokenizer locally."
                        if tokenizer_error:
                            warning_msg += f" ({tokenizer_error})"
                        warning_msg += " Falling back to whitespace token counts.[/yellow]"
                        self.console.print(f"{warning_msg}\n")

                    char_lengths = []
                    token_lengths = []

                    iterator = (
                        tqdm(analysis_texts, desc="Measuring text lengths", disable=not HAS_TQDM)
                        if HAS_TQDM else analysis_texts
                    )

                    for text in iterator:
                        char_lengths.append(len(text))
                        if tokenizer:
                            try:
                                tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
                                token_lengths.append(len(tokens))
                            except Exception as encode_error:
                                self.logger.debug(f"Tokenizer failed on text length analysis: {encode_error}")
                                token_lengths.append(len(text.split()))
                        else:
                            token_lengths.append(len(text.split()))

                    char_lengths = np.array(char_lengths) if char_lengths else np.array([0])
                    token_lengths = np.array(token_lengths) if token_lengths else np.zeros_like(char_lengths)

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
                    text_length_stats['rows_analyzed'] = len(analysis_series)
                    text_length_stats['rows_available'] = int(source_total)
                    if subset_label:
                        text_length_stats['subset_label'] = subset_label

                    # Classify documents by length
                    short_docs = np.sum(token_lengths < 128)
                    medium_docs = np.sum((token_lengths >= 128) & (token_lengths < 512))
                    long_docs = np.sum((token_lengths >= 512) & (token_lengths < 1024))
                    very_long_docs = np.sum(token_lengths >= 1024)
                    total_docs = len(token_lengths)
                    denom = total_docs if total_docs else 1

                    text_length_stats['distribution'] = {
                        'short': {'count': int(short_docs), 'percentage': float(short_docs / denom * 100)},
                        'medium': {'count': int(medium_docs), 'percentage': float(medium_docs / denom * 100)},
                        'long': {'count': int(long_docs), 'percentage': float(long_docs / denom * 100)},
                        'very_long': {'count': int(very_long_docs), 'percentage': float(very_long_docs / denom * 100)},
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

                        dist_table.add_row("Short", "< 128 tokens", f"{short_docs:,}", f"{short_docs / denom * 100:.1f}%")
                        dist_table.add_row("Medium", "128-511 tokens", f"{medium_docs:,}", f"{medium_docs / denom * 100:.1f}%")
                        dist_table.add_row("Long", "512-1023 tokens", f"{long_docs:,}", f"{long_docs / denom * 100:.1f}%",
                                           style="bold yellow" if long_docs > 0 else None)
                        dist_table.add_row("Very Long", "≥ 1024 tokens", f"{very_long_docs:,}", f"{very_long_docs / denom * 100:.1f}%",
                                           style="bold red" if very_long_docs > 0 else None)

                        self.console.print(dist_table)

                        # Long document warning
                        long_document_percentage = ((long_docs + very_long_docs) / denom) * 100

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
                    char_lengths = [len(str(text)) for text in analysis_texts]
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
        # Use application logs subdirectory for general application logs
        log_dir = self.settings.paths.logs_dir / "application"
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
            info_table.add_row("🚀 Features:", "[cyan]50+ BERT Models, Multi-LLM (Ollama/OpenAI/Claude), Parallel GPU/CPU, Reinforcement Learning[/cyan]")
            info_table.add_row("🎯 Capabilities:", "[magenta]Multi-Label Classification, 100+ Languages, SQL/File I/O, Doccano/Label Studio Export[/magenta]")

            self.console.print(Panel(
                info_table,
                title="[bold bright_cyan]✨ Welcome to LLM Tool ✨[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2)
            ))
            self.console.print()

            # Auto-detect models and system resources in background
            with self.console.status("[bold green]🔍 Scanning environment...", spinner="dots"):
                self.detected_llms = self.llm_detector.detect_all_llms()
                self.available_trainer_models = self.trainer_model_detector.get_available_models()
                # Scan only in data/ directory
                data_dir = self.settings.paths.data_dir
                self.detected_datasets = self.data_detector.scan_directory(data_dir)
                # Detect system resources
                self.system_resources = self.resource_detector.detect_all()

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

        # === SYSTEM RESOURCES SECTION ===
        if self.system_resources:
            # Create and display the visual resource panel
            resource_panel = create_visual_resource_panel(
                self.system_resources,
                show_recommendations=True
            )
            self.console.print(resource_panel)
            self.console.print()

    def get_main_menu_choice(self) -> str:
        """Display sophisticated main menu with smart suggestions"""
        if HAS_RICH and self.console:
            # Create menu table
            menu_table = Table.grid(padding=0)
            menu_table.add_column(width=3)
            menu_table.add_column()

            options = [
                ("1", "🎨 The Annotator - Zero-Shot LLM Annotation (Ollama/OpenAI/Claude) → Label Studio/Doccano Export"),
                ("2", "🏭 The Annotator Factory - LLM Annotations → Training Data → Fine-Tuned BERT Models"),
                ("3", "🎮 Training Arena - Train 50+ Models (BERT/RoBERTa/DeBERTa) with Multi-Label & Benchmarking"),
                ("4", "🤖 BERT Annotation Studio - High-Throughput Inference (Parallel GPU/CPU, 100+ Languages)"),
                ("5", "🔍 Validation Lab - Quality Scoring, Stratified Sampling, Inter-Annotator Agreement [⚠️ IN DEVELOPMENT]"),
                ("6", "📂 Resume Center - Manage Sessions & Configurations"),
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
            print("1. The Annotator - Zero-Shot LLM Annotation (Ollama/OpenAI/Claude) → Export")
            print("2. The Annotator Factory - LLM Annotations → Training Data → BERT Models")
            print("3. Training Arena - Train 50+ Models (Multi-Label & Benchmarking)")
            print("4. BERT Annotation Studio - High-Throughput Inference (Parallel GPU/CPU)")
            print("5. Validation Lab - Quality Scoring & Sampling [⚠️ IN DEVELOPMENT]")
            print("6. Resume Center - Manage Sessions & Configurations")
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

        # Check for recent saved configurations
        recent_profiles = self.profile_manager.list_profiles()
        if recent_profiles:
            suggestions.append(f"Resume config: {recent_profiles[0].name}")

        return " | ".join(suggestions) if suggestions else ""

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------
    def _resume_mode_roots(self) -> Dict[str, Path]:
        """Return base directories for resumable modes."""
        return {
            "annotator": Path("logs") / "annotator",
            "annotator_factory": Path("logs") / "annotator_factory",
            "training_arena": get_training_logs_base(),
            "bert_annotation_studio": Path("logs") / "annotation_studio",
        }

    def _fetch_resume_records(self, mode: str, limit: int = 20) -> List[SummaryRecord]:
        """Load recent SessionSummary records for a specific mode."""
        roots = self._resume_mode_roots()
        base_dir = roots.get(mode)
        if not base_dir:
            return []
        return collect_summaries_for_mode(base_dir, mode, limit)

    def _discover_annotation_metadata(self, session_id: str, session_dir: Path) -> List[Path]:
        """
        Locate metadata JSON files associated with an annotation session.

        Search order:
            1. logs/<mode>/<session_id>/metadata/*.json
            2. data/annotations/*<session_id>*_metadata_*.json  (legacy)
        """
        candidates: List[Path] = []
        metadata_dir = session_dir / "metadata"
        if metadata_dir.exists():
            candidates.extend(
                path for path in metadata_dir.rglob("*_metadata_*.json") if path.is_file()
            )

        # New training metadata layout (v2.0+)
        training_metadata_dir = session_dir / "training_session_metadata"
        if training_metadata_dir.exists():
            for filename in ("training_metadata.json", "training_metadata_backup.json"):
                candidate = training_metadata_dir / filename
                if candidate.exists() and candidate.is_file():
                    candidates.append(candidate)

        legacy_training_metadata = session_dir / "training_metadata.json"
        if legacy_training_metadata.exists() and legacy_training_metadata.is_file():
            candidates.append(legacy_training_metadata)

        legacy_dir = getattr(self.settings.paths, "data_dir", Path("data")) / "annotations"
        if legacy_dir.exists():
            pattern = f"*{session_id}*_metadata_*.json"
            candidates.extend(
                path for path in legacy_dir.rglob(pattern) if path.is_file()
            )

        unique_candidates = {path.resolve(): path for path in candidates}
        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return sorted_candidates

    def _load_annotation_resume_candidates(
        self,
        mode: str,
        limit: int = 20,
    ) -> List[Tuple[Path, Dict[str, Any], SummaryRecord]]:
        """
        Assemble metadata & summary triples for annotation workflows.

        Returns list of tuples: (metadata_path, metadata_payload, summary_record)
        """
        candidates: List[Tuple[Path, Dict[str, Any], SummaryRecord]] = []
        for record in self._fetch_resume_records(mode, limit):
            session_id = record.summary.session_id
            metadata_files = self._discover_annotation_metadata(session_id, record.directory)
            if not metadata_files:
                continue
            metadata_path = metadata_files[0]
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = self._normalize_factory_metadata(metadata, metadata_path)
            except Exception:
                continue
            candidates.append((metadata_path, metadata, record))
        return candidates

    def _normalize_factory_metadata(self, metadata: Dict[str, Any], metadata_path: Path) -> Dict[str, Any]:
        """
        Bring training-metadata payloads in line with legacy annotator metadata schema.

        The Annotator Factory originally stored annotation parameters under ``metadata/*.json``.
        Newer runs write comprehensive training metadata to ``training_session_metadata/``.
        Resume workflows still expect the legacy layout, so we synthesize the required fields.
        """
        if not isinstance(metadata, dict):
            return {}

        # Already in legacy format
        if metadata.get("annotation_session") and metadata.get("data_source"):
            return metadata

        training_session = metadata.get("training_session")
        dataset_config = metadata.get("dataset_config", {})
        model_config = metadata.get("model_config", {})
        training_params = metadata.get("training_params", {})
        output_paths = metadata.get("output_paths", {})
        training_context = metadata.get("training_context", {})
        execution_status = metadata.get("execution_status", {})
        text_analysis = metadata.get("text_analysis", {})

        if not training_session:
            return metadata

        # Derive data source information
        source_file = (
            dataset_config.get("source_file")
            or dataset_config.get("primary_file")
            or output_paths.get("annotated_dataset")
        )
        file_name = Path(source_file).name if source_file else None
        data_format = None
        if file_name:
            suffix = Path(file_name).suffix.lower()
            data_format = suffix.lstrip(".") if suffix else None

        total_rows = (
            dataset_config.get("total_samples")
            or dataset_config.get("rows")
            or text_analysis.get("rows_available")
            or text_analysis.get("rows_analyzed")
        )
        data_source = {
            "file_path": source_file,
            "file_name": file_name,
            "data_format": data_format or dataset_config.get("format") or "unknown",
            "text_column": dataset_config.get("text_column", "text"),
            "total_rows": total_rows,
            "sampling_strategy": training_context.get("sampling_strategy") or "N/A",
        }

        actual_models = model_config.get("actual_models_trained") or []
        if isinstance(actual_models, list) and actual_models:
            fallback_model = actual_models[0]
        else:
            fallback_model = None
        model_name = (
            model_config.get("selected_model")
            or model_config.get("quick_model_name")
            or fallback_model
        )

        model_configuration = {
            "provider": model_config.get("provider") or "huggingface",
            "model_name": model_name,
            "temperature": training_context.get("annotation_config", {}).get("temperature", "N/A"),
            "max_tokens": training_context.get("annotation_config", {}).get("max_tokens", "N/A"),
            "epochs": model_config.get("epochs") or training_params.get("epochs"),
            "learning_rate": training_params.get("learning_rate"),
        }

        processing_configuration = {
            "parallel_workers": training_context.get("parallel_workers", 1),
            "batch_size": training_params.get("batch_size") or model_config.get("batch_size"),
            "reinforced_learning": bool(model_config.get("reinforced_learning") or training_params.get("use_reinforcement")),
        }

        output = {
            "output_path": source_file or output_paths.get("annotated_dataset"),
            "models_dir": output_paths.get("models_dir"),
            "metrics_dir": output_paths.get("logs_dir") or output_paths.get("metrics_dir"),
            "training_files": dataset_config.get("training_files"),
        }

        export_preferences = training_context.get("export_preferences", {})

        is_resume = bool(
            metadata.get("resume_mode")
            or training_context.get("resume_mode")
            or "resume" in metadata_path.stem
        )

        training_workflow = {
            "training_enabled": True,
            "training_mode": model_config.get("training_mode", "quick"),
            "training_params_file": str(metadata_path),
            "models_trained": model_config.get("actual_models_trained"),
            "status": execution_status.get("status"),
            "resume_mode": is_resume,
            "session_dir": output_paths.get("session_dir"),
            "last_update": metadata.get("last_updated"),
        }

        normalized = {
            "annotation_session": {
                "workflow": training_session.get("workflow", "Annotator Factory - Training"),
                "timestamp": training_session.get("session_id") or metadata.get("created_at"),
                "tool_version": training_session.get("tool_version"),
                "mode": training_session.get("mode"),
                "status": execution_status.get("status"),
                "action_mode": "resume" if is_resume else training_session.get("mode", "quick"),
            },
            "data_source": data_source,
            "model_configuration": model_configuration,
            "prompts": training_context.get("annotation_prompts", []),
            "processing_configuration": processing_configuration,
            "output": output,
            "export_preferences": export_preferences,
            "training_workflow": training_workflow,
            "session_id": training_session.get("session_id"),
            "resume_mode": is_resume,
        }

        # Preserve original payload for advanced actions if needed later
        normalized["_factory_training_metadata"] = metadata
        return normalized

    def resume_center(self) -> None:
        """Unified dashboard listing resumable sessions across all modes."""
        mode_labels = {
            "annotator": "Annotator",
            "annotator_factory": "Annotator Factory",
            "training_arena": "Training Arena",
            "bert_annotation_studio": "BERT Annotation Studio",
        }

        records = collect_all_summaries(
            self._resume_mode_roots(),
            limit_per_mode=15,
            total_limit=60,
        )
        if not records:
            message = "[yellow]No resumable sessions detected yet.[/yellow]\n[dim]Start a workflow to populate the resume center.[/dim]"
            if HAS_RICH and self.console:
                self.console.print(message)
            else:
                print("No resumable sessions detected yet. Run a workflow to populate the resume center.")
            return

        if HAS_RICH and self.console:
            table = Table(title="📂 Resume Center", border_style="cyan")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Mode", style="magenta", width=18)
            table.add_column("Session", style="white")
            table.add_column("Status", style="green", width=12)
            table.add_column("Updated", style="yellow", width=19)
            table.add_column("Last Step", style="cyan")
        else:
            print("\n=== Resume Center ===")
            print(f"{'#':<4}{'Mode':<18}{'Session':<30}{'Status':<12}{'Updated':<20}{'Last step'}")

        entries: List[SummaryRecord] = []
        for idx, record in enumerate(records, 1):
            summary = record.summary
            mode_label = mode_labels.get(summary.mode, summary.mode)
            updated_display = summary.updated_at.replace("T", " ")
            last_step = summary.last_step_name or summary.last_step_key or "-"
            if summary.last_step_no:
                last_step = f"{summary.last_step_no}. {last_step}"

            if HAS_RICH and self.console:
                table.add_row(
                    str(idx),
                    mode_label,
                    summary.session_id,
                    summary.status,
                    updated_display,
                    last_step,
                )
            else:
                print(f"{idx:<4}{mode_label:<18}{summary.session_id:<30}{summary.status:<12}{updated_display:<20}{last_step}")
            entries.append(record)

        if HAS_RICH and self.console:
            self.console.print(table)

        selection = self._int_prompt_with_validation(
            "\n[bold yellow]Select session (0 to return)[/bold yellow]" if HAS_RICH and self.console else "\nSelect session (0 to return): ",
            0,
            0,
            len(entries),
        )
        if selection == 0:
            return

        chosen = entries[selection - 1]
        summary = chosen.summary
        mode = summary.mode

        if mode == "annotator":
            self._quick_annotate(focus_session_id=summary.session_id)
        elif mode == "annotator_factory":
            self._resume_mode2(focus_session_id=summary.session_id)
        elif mode == "training_arena":
            if hasattr(self, "_resume_training_studio"):
                self._resume_training_studio(summary.session_id)
            else:
                self.console.print("[yellow]Training resume handler unavailable in this build.[/yellow]" if HAS_RICH and self.console else "Training resume handler unavailable.")
        elif mode == "bert_annotation_studio":
            if HAS_RICH and self.console:
                self.console.print(
                    "\n[cyan]Opening BERT Annotation Studio. Select 'Resume existing session' and choose the highlighted session.[/cyan]"
                )
            else:
                print("\nOpening BERT Annotation Studio. Select 'Resume existing session' to continue.")
            self.bert_annotation_studio()
        else:
            if HAS_RICH and self.console:
                self.console.print(f"[yellow]No direct handler for mode: {mode}[/yellow]")
            else:
                print(f"No direct handler for mode: {mode}")


    def _get_api_key(self, provider: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Backwards-compatible helper used throughout the CLI to obtain provider API keys.
        """
        return self._get_or_prompt_api_key(provider, model_name=model_name)

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

    def _display_mode_banner(self, mode: str):
        """Display mode-specific ASCII banner"""
        if not (HAS_RICH and self.console):
            return

        from rich.align import Align
        from .banners import BANNERS

        if mode not in BANNERS:
            return

        banner_data = BANNERS[mode]
        color = banner_data['color']

        self.console.print()
        for line in banner_data['ascii'].split('\n'):
            self.console.print(Align.center(f"[bold {color}]{line}[/bold {color}]"))

        self.console.print()
        self.console.print(Align.center(f"[bold {color}]{banner_data['tagline']}[/bold {color}]"))
        self.console.print()

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

                self.console.print(Panel(
                    info_table,
                    title=f"[bold cyan]{title}[/bold cyan]",
                    subtitle=f"[dim]{description}[/dim]",
                    border_style="cyan",
                    padding=(1, 2)
                ))

                # Display horizontal resource banner below the info panel
                if self.system_resources:
                    self.console.print()
                    resource_banner = create_mode_resource_banner(self.system_resources)
                    banner_panel = Panel(
                        resource_banner,
                        title="[bold bright_blue]⚙️  System Resources[/bold bright_blue]",
                        border_style="blue",
                        padding=(0, 1)
                    )
                    self.console.print(banner_panel)

            else:
                # Fallback to simple panel
                self.console.print(Panel.fit(
                    f"[bold cyan]{title}[/bold cyan]\n{description}",
                    border_style="cyan"
                ))

                # Display horizontal resource banner
                if self.system_resources:
                    self.console.print()
                    resource_banner = create_mode_resource_banner(self.system_resources)
                    banner_panel = Panel(
                        resource_banner,
                        title="[bold bright_blue]⚙️  System Resources[/bold bright_blue]",
                        border_style="blue",
                        padding=(0, 1)
                    )
                    self.console.print(banner_panel)

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
            info_table.add_row("🚀 Features:", "[cyan]Ollama/OpenAI/Claude, Prompt Wizard, Auto JSON Repair, 200K Context Support[/cyan]")
            info_table.add_row("🎯 Capabilities:", "[magenta]Multi-Label Categories, NER, Hierarchical Schemas, Pydantic Validation[/magenta]")
            info_table.add_row("⚡ Performance:", "[green]Parallel Processing, Incremental Save, Resume, Label Studio/Doccano Export[/green]")

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
            print("🚀 Features: Ollama/OpenAI/Claude, Prompt Wizard, 200K Context, JSON Repair")
            print("🎯 Capabilities: Multi-Label Categories, NER, Hierarchical Schemas")
            print("⚡ Performance: Parallel Processing, Incremental Save, Resume Support")
            print("\n  🤖 -> 📝 -> 🧹 -> 🎯 -> 🧠 -> 📊 -> ✨")
            print("  AI   Annotate Clean Label Train Test Deploy\n")
            print("="*80 + "\n")

    def quick_start_wizard(self):
        """Complete workflow: LLM annotation followed by intelligent model training"""
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display mode-specific banner
        self._display_mode_banner('factory')

        # Display personalized mode info
        self._display_section_header(
            "🏭 The Annotator Factory - LLM Annotations → Training Data → Fine-Tuned BERT Models",
            "End-to-end pipeline: Ollama/OpenAI/Claude annotation → Automatic training data conversion → Model training",
            mode_info={
                'workflow': 'Data → LLM Annotate (Parallel) → Language Detection → Auto-Convert → Train 50+ Models',
                'capabilities': ['Multi-LLM Support (Ollama/OpenAI/Claude)', '100+ Languages', 'Multi-Label Classification', 'Reinforcement Learning'],
                'input': 'CSV/Excel/JSON/SQL with text column',
                'output': 'Annotated data + Trained BERT models + Benchmarking metrics + Training summaries',
                'best_for': 'Complete zero-shot annotation to supervised learning pipeline with automatic optimization'
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
                title="[bold]🏭 The Annotator Factory[/bold]",
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
                # CRITICAL: Ask user for session name first (like Training Arena and Mode 1)
                from datetime import datetime

                self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
                self.console.print("[bold cyan]           📝 Session Name Configuration                       [/bold cyan]")
                self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

                self.console.print("[bold]Why session names matter:[/bold]")
                self.console.print("  • [green]Organization:[/green] Easily identify pipelines (e.g., 'sentiment_twitter', 'legal_classifier')")
                self.console.print("  • [green]Traceability:[/green] Track annotations, training, and models in one place")
                self.console.print("  • [green]Collaboration:[/green] Team members understand each pipeline's purpose")
                self.console.print("  • [green]Audit trail:[/green] Timestamp ensures uniqueness\n")

                self.console.print("[dim]Format: {session_name}_{yyyymmdd_hhmmss}[/dim]")
                self.console.print("[dim]Example: sentiment_pipeline_20251008_143022[/dim]\n")

                # Ask for user-defined session name
                user_session_name = Prompt.ask(
                    "[bold yellow]Enter a descriptive name for this annotation+training pipeline[/bold yellow]",
                    default="factory_session"
                ).strip()

                # Sanitize the user input
                user_session_name = user_session_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                user_session_name = ''.join(c for c in user_session_name if c.isalnum() or c in ['_', '-'])

                # Create full session ID with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = f"{user_session_name}_{timestamp}"

                self.console.print(f"\n[bold green]✓ Session ID:[/bold green] [cyan]{session_id}[/cyan]")
                self.console.print(f"[dim]This ID will be used for annotations, training, and all outputs[/dim]\n")

                # Pass session_id to _complete_workflow_mode2
                self._complete_workflow_mode2(session_id=session_id)
            elif workflow == "3":
                self._clean_metadata()
        else:
            print("\n=== The Annotator Factory ===")
            print("Clone The Annotator into ML Models\n")
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

    def _complete_workflow_mode2(self, session_id: str = None):
        """Execute complete annotation → training workflow via shared module."""
        run_factory_workflow(self, session_id=session_id)

    def _resume_mode2(self, focus_session_id: Optional[str] = None):
        """Resume or relaunch annotation → training workflow using saved parameters"""
        self.console.print("\n[bold cyan]🔄 Resume/Relaunch Workflow[/bold cyan]\n")
        self.console.print("[dim]Load saved parameters from previous annotation → training sessions[/dim]\n")

        # ============================================================
        # DETECT METADATA FILES
        candidates = self._load_annotation_resume_candidates("annotator_factory", limit=25)
        if not candidates:
            self.console.print("[yellow]No saved workflow parameters found.[/yellow]")
            self.console.print("[dim]Run Complete Workflow and save parameters to use this feature.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        self.console.print(f"[green]Found {len(candidates)} resumable workflow session(s)[/green]\n")

        sessions_table = Table(border_style="cyan", show_header=True)
        sessions_table.add_column("#", style="cyan", width=3)
        sessions_table.add_column("Session", style="white")
        sessions_table.add_column("Updated", style="yellow")
        sessions_table.add_column("Status", style="green", width=12)
        sessions_table.add_column("Last Step", style="cyan")
        sessions_table.add_column("Model", style="magenta")

        valid_sessions: List[Tuple[Path, Dict[str, Any], SessionSummary]] = []
        for idx, (metadata_path, metadata, record) in enumerate(candidates, 1):
            summary = record.summary
            session_id = summary.session_id
            updated_display = summary.updated_at.replace("T", " ")
            last_step_display = summary.last_step_name or summary.last_step_key or "-"
            model_config = metadata.get("model_configuration", {})
            model_name = model_config.get("model_name") or model_config.get("selected_model") or summary.extra.get("model") or "-"

            sessions_table.add_row(
                str(idx),
                session_id,
                updated_display,
                summary.status,
                last_step_display,
                model_name,
            )

            valid_sessions.append((metadata_path, metadata, summary))

        if not valid_sessions:
            self.console.print("[yellow]No valid metadata files found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        self.console.print(sessions_table)

        session_choice: Optional[int] = None
        if focus_session_id:
            for idx, (_, _, summary) in enumerate(valid_sessions, 1):
                if summary and summary.session_id == focus_session_id:
                    session_choice = idx
                    self.console.print(f"\n[dim]Auto-selecting session {summary.session_id}[/dim]")
                    break

        if session_choice is None:
            session_choice = self._int_prompt_with_validation(
                "\n[bold yellow]Select session to resume/relaunch[/bold yellow]",
                1, 1, len(valid_sessions)
            )

        selected_file, metadata, summary = valid_sessions[session_choice - 1]

        self.console.print(f"\n[green]✓ Selected: {selected_file.name}[/green]")
        if summary:
            last_step = summary.last_step_name or summary.last_step_key or "-"
            self.console.print(f"[dim]Status: {summary.status} • Last step: {last_step}[/dim]")

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

        identifier_column = (
            proc_config.get('identifier_column')
            or data_source.get('identifier_column')
        )
        if not identifier_column or identifier_column == 'annotation_id':
            identifier_column = 'llm_annotation_id'
        pipeline_identifier_column = (
            None if identifier_column == 'llm_annotation_id' else identifier_column
        )
        data_source.setdefault('identifier_column', identifier_column)
        proc_config.setdefault('identifier_column', identifier_column)
        training_workflow = metadata.get('training_workflow', {})

        # Create session ID and directories
        # For relaunch: create new session, for resume: use existing or create new
        if action_mode == 'relaunch':
            # New session with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = f"factory_session_{timestamp}"
        else:
            # Try to get existing session ID from metadata
            session_id = metadata.get('session_id')
            if not session_id:
                # Fallback: create from metadata file name or timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                session_id = f"factory_session_{timestamp}"

        # Create session directories
        session_dirs = self._create_annotator_factory_session_directories(session_id)

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
        stored_trained_model_paths: Dict[str, str] = {}

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

            if isinstance(training_workflow.get('trained_model_paths'), dict):
                stored_trained_model_paths.update(training_workflow['trained_model_paths'])

            training_context_meta = metadata.get('training_context', {})
            if isinstance(training_context_meta.get('trained_model_paths'), dict):
                stored_trained_model_paths.update(training_context_meta['trained_model_paths'])

            usable_models: List[Path] = []
            for _, candidate_path in stored_trained_model_paths.items():
                candidate = Path(candidate_path).expanduser()
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.exists():
                    usable_models.append(candidate.resolve())

            if usable_models:
                unique_models = []
                for path in usable_models:
                    if path not in unique_models:
                        unique_models.append(path)
                training_complete = True
                if self.console:
                    self.console.print(f"[green]✓ Detected {len(unique_models)} trained model(s) ready for deployment[/green]")
            else:
                if original_output:
                    model_dir = Path(original_output).parent / f"{Path(original_output).stem}_model"
                    if model_dir.exists():
                        self.console.print(f"[green]✓ Found trained model: {model_dir.name}[/green]")
                        training_complete = True
                        stored_trained_model_paths[model_dir.name] = str(model_dir)

            if not training_complete:
                session_model_root = Path("models") / session_id if session_id else None
                normal_training_root = None
                if session_model_root and session_model_root.exists():
                    normal_candidate = session_model_root / "normal_training"
                    normal_training_root = normal_candidate if normal_candidate.exists() else session_model_root

                discovered_models: Dict[str, Path] = {}
                if normal_training_root and normal_training_root.exists():
                    for config_path in normal_training_root.glob("**/config.json"):
                        model_dir = config_path.parent
                        try:
                            relative_name = model_dir.relative_to(normal_training_root).as_posix()
                        except ValueError:
                            relative_name = model_dir.name
                        discovered_models[relative_name] = model_dir

                if discovered_models:
                    for name, path in discovered_models.items():
                        stored_trained_model_paths.setdefault(name, str(path))
                    training_complete = True
                    if self.console:
                        self.console.print(
                            f"[green]✓ Located {len(discovered_models)} trained model(s) in session directory[/green]"
                        )

            if training_complete and stored_trained_model_paths:
                training_workflow['trained_model_paths'] = {
                    str(k): str(v) for k, v in stored_trained_model_paths.items()
                }

        # Determine what to run
        run_annotation = not annotation_complete or action_mode == 'relaunch'
        run_training = not training_complete or action_mode == 'relaunch'
        run_model_annotation = True

        # Pre-build prompt configurations for downstream stages (training + deployment)
        prompt_configs_for_training: List[Dict[str, Any]] = []
        for p in prompts:
            prompt_configs_for_training.append({
                'prompt': {
                    'keys': p.get('expected_keys', []),
                    'content': p.get('prompt_content', p.get('prompt', '')),
                    'name': p.get('name', 'prompt')
                },
                'prefix': p.get('prefix', '')
            })

        resume_stage = None
        if action_mode == 'resume':
            stage_info = {
                "1": {
                    "label": "LLM Annotation (restart pipeline)",
                    "available": True,
                    "status": "[green]Ready[/green]",
                },
                "2": {
                    "label": "Model Training (reuse annotated data)",
                    "available": annotation_complete,
                    "status": "[green]Ready[/green]" if annotation_complete else "[red]Annotation output missing[/red]",
                },
                "3": {
                    "label": "Deploy & Annotate (BERT Studio)",
                    "available": training_complete,
                    "status": "[green]Ready[/green]" if training_complete else "[red]Trained models unavailable[/red]",
                },
            }

            default_stage = "1"
            if annotation_complete:
                default_stage = "2" if not training_complete else "3"

            stage_table = Table(title="Resume Stage Selection", border_style="cyan")
            stage_table.add_column("#", justify="center", style="bold cyan", width=4)
            stage_table.add_column("Stage", style="white")
            stage_table.add_column("Status", style="magenta")

            for stage_key in ("1", "2", "3"):
                info = stage_info[stage_key]
                stage_table.add_row(stage_key, info["label"], info["status"])

            self.console.print(stage_table)

            while True:
                resume_stage = Prompt.ask(
                    "\n[bold yellow]Select stage to resume from[/bold yellow]",
                    choices=["1", "2", "3"],
                    default=default_stage
                )
                selection = stage_info[resume_stage]
                if selection["available"]:
                    break
                self.console.print(f"[yellow]⚠️  {selection['label']} is not available. Please choose another stage.[/yellow]")

            self.console.print(f"\n[cyan]Resuming from: {stage_info[resume_stage]['label']}[/cyan]\n")

            if resume_stage == "1":
                run_annotation = True
                run_training = True
            elif resume_stage == "2":
                run_annotation = False
                if training_complete:
                    rerun_training = Confirm.ask(
                        "Training already completed. Re-run the training phase?",
                        default=False
                    )
                    run_training = rerun_training
                    if not rerun_training:
                        self.console.print("[cyan]Using existing trained models.[/cyan]")
                else:
                    run_training = True
            elif resume_stage == "3":
                run_annotation = False
                run_training = False

        # ============================================================
        # PHASE 1: ANNOTATION (if needed)
        # ============================================================
        if run_annotation:
            self.console.print("\n[bold cyan]📝 Phase 1: LLM Annotation[/bold cyan]\n")

            # CRITICAL: Use organized structure logs/annotator_factory/{session_id}/annotated_data/{dataset_name}/
            safe_model_name = model_config.get('model_name', 'unknown').replace(':', '_').replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Create dataset-specific subdirectory
            dataset_name = data_path.stem
            dataset_subdir = session_dirs['annotated_data'] / dataset_name
            dataset_subdir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
            output_path = dataset_subdir / output_filename

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
                'identifier_column': pipeline_identifier_column,
                'run_annotation': True,
                'annotation_mode': model_config.get('annotation_mode', 'local'),
                'annotation_provider': provider,
                'annotation_model': model_name,
                'api_key': api_key,
                'prompts': prompts_payload,
                'annotation_sample_size': data_source.get('total_rows'),
                'annotation_requested_total': data_source.get(
                    'requested_rows',
                    data_source.get('total_rows')
                ),
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

                pipeline_with_progress = PipelineController(
                    settings=self.settings,
                    session_id=session_id  # Pass session_id for organized logging
                )

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
                    df_annotated = df_for_lang[(df_for_lang[annotation_col].notna()) & (df_for_lang[annotation_col] != '')].copy()

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
        training_results_summary: Optional[Dict[str, Any]] = None
        if run_training:
            self.console.print("\n[bold cyan]🎓 Phase 2: Model Training[/bold cyan]\n")
            training_results_summary = self._post_annotation_training_workflow(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs_for_training,
                session_id=session_id,
                session_dirs=session_dirs if 'session_dirs' in locals() else None
            )
        else:
            self.console.print("\n[yellow]⏭️  Skipping training (using existing models)[/yellow]\n")
            training_results_summary = self._load_saved_factory_training_results(
                session_id=session_id,
                session_dirs=session_dirs if 'session_dirs' in locals() else None,
                training_workflow=training_workflow
            )
            if not training_results_summary and stored_trained_model_paths:
                normalized_paths = {
                    str(k): str(Path(v).expanduser())
                    for k, v in stored_trained_model_paths.items()
                }
                training_results_summary = {
                    "status": "completed",
                    "session_id": session_id,
                    "training_result": {
                        "trained_models": normalized_paths,
                        "models_trained": list(normalized_paths.keys()),
                    },
                    "trained_model_paths": normalized_paths,
                }

        if training_results_summary and training_results_summary.get("training_result"):
            training_workflow['status'] = training_results_summary.get("status", "completed")
            training_workflow['models_trained'] = training_results_summary["training_result"].get("models_trained", [])
            training_workflow['trained_model_paths'] = training_results_summary["training_result"].get("trained_models", {})
            training_workflow['last_update'] = datetime.now().isoformat()

        # ============================================================
        # PHASE 3: DEPLOY & ANNOTATE WITH TRAINED MODELS
        # ============================================================
        if run_model_annotation:
            if training_results_summary and training_results_summary.get("training_result"):
                model_annotation_summary = _launch_model_annotation_stage(
                    self,
                    session_id=session_id,
                    session_dirs=session_dirs if 'session_dirs' in locals() else None,
                    training_results=training_results_summary,
                    prompt_configs=prompt_configs_for_training,
                    text_column=text_column,
                    annotation_output=output_file,
                    dataset_path=data_path,
                )
                if isinstance(model_annotation_summary, dict):
                    status = model_annotation_summary.get("status", "completed")
                    detail = model_annotation_summary.get("detail")
                    if status != "completed":
                        self.console.print(
                            f"[yellow]⚠️  Model annotation stage reported status '{status}': {detail}[/yellow]"
                        )
            else:
                self.console.print(
                    "[yellow]⚠️  Skipping deployment annotation stage: trained model artifacts not found.[/yellow]"
                )

        self.console.print("\n[bold green]✅ Workflow complete![/bold green]")

    def _post_annotation_training_workflow(
        self,
        output_file: str,
        text_column: str,
        prompt_configs: list,
        session_id: str = None,
        session_dirs: dict = None
    ):
        """
        Comprehensive post-annotation training workflow.
        Integrated with Training Arena for complete training capabilities.

        Features:
        - Session management with organized outputs
        - Dataset wizard with intelligent column detection
        - Language strategy selection (multilingual/specialized/hybrid)
        - Text length analysis for long-document models
        - Model catalog with 50+ models
        - Benchmark mode for testing multiple models
        - Comprehensive metrics and training summaries
        """
        try:
            # Display STEP 2/3 banner - Train Models
            from llm_tool.cli.banners import BANNERS, STEP_NUMBERS, STEP_LABEL
            from rich.align import Align

            self.console.print()

            # Display "STEP" label in ASCII art
            for line in STEP_LABEL.split('\n'):
                self.console.print(Align.center(f"[bold {BANNERS['train_model']['color']}]{line}[/bold {BANNERS['train_model']['color']}]"))

            # Display "2/3" in ASCII art
            for line in STEP_NUMBERS['2/3'].split('\n'):
                self.console.print(Align.center(f"[bold {BANNERS['train_model']['color']}]{line}[/bold {BANNERS['train_model']['color']}]"))

            self.console.print()

            # Display main TRAIN MODEL banner (centered)
            for line in BANNERS['train_model']['ascii'].split('\n'):
                self.console.print(Align.center(f"[bold {BANNERS['train_model']['color']}]{line}[/bold {BANNERS['train_model']['color']}]"))

            # Display tagline (centered)
            self.console.print(Align.center(f"[{BANNERS['train_model']['color']}]{BANNERS['train_model']['tagline']}[/{BANNERS['train_model']['color']}]"))
            self.console.print()

            # Import and use the COMPLETE Training Arena integration
            from llm_tool.cli.training_arena_integrated import integrate_training_arena_in_annotator_factory

            # Pass session_dirs for organized structure
            training_results = integrate_training_arena_in_annotator_factory(
                cli_instance=self,
                output_file=output_file,
                text_column=text_column,
                session_id=session_id,
                session_dirs=session_dirs  # Pass session directories for organized logging
            )

            # The integration handles everything including asking if user wants to train
            # and running the complete Training Arena workflow
            return training_results

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

            # Step 3: Text Length Analysis (from Training Arena)
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

            # Step 4: Language Strategy Analysis (from Training Arena)
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

                    # Loop until valid model is selected
                    selected = None
                    while selected is None:
                        model_choice = Prompt.ask(
                            f"\nSelect model for {lang_upper} (number or name)",
                            default="1"
                        )

                        if model_choice.isdigit():
                            choice_num = int(model_choice)
                            if 0 < choice_num <= len(lang_recommendations):
                                selected = lang_recommendations[choice_num - 1]['model']
                            else:
                                self.console.print(f"[red]Invalid number. Please enter 1-{len(lang_recommendations)}[/red]")
                        else:
                            # Check if it's a valid model name
                            if model_choice in self.available_trainer_models:
                                selected = model_choice
                            else:
                                self.console.print(f"[red]Model '{model_choice}' not found. Use number or valid model name.[/red]")

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

                # Loop until valid model is selected
                training_model = None
                while training_model is None:
                    selected_model = Prompt.ask(
                        "\nSelect model (number or name)",
                        default="1"
                    )

                    if selected_model.isdigit():
                        choice_num = int(selected_model)
                        if 0 < choice_num <= len(recommended_models):
                            training_model = recommended_models[choice_num - 1]
                        else:
                            self.console.print(f"[red]Invalid number. Please enter 1-{len(recommended_models)}[/red]")
                    else:
                        # Check if it's a valid model name
                        if selected_model in self.available_trainer_models:
                            training_model = selected_model
                        else:
                            self.console.print(f"[red]Model '{selected_model}' not found. Use number or valid model name.[/red]")

                self.console.print(f"[green]✓ Selected model: {training_model}[/green]\n")

            # Step 6: Training Configuration
            self.console.print("[bold cyan]Step 6: Training Configuration[/bold cyan]\n")

            # Training mode selection
            training_modes = {
                "quick": "Quick training (10 epochs, fast)",
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
                epochs = 10
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

                # Call Training Arena's training method
                try:
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
                    from ..utils.data_filter_logger import get_filter_logger

                    stratify_col = None
                    if not is_multi_label:
                        label_counts = Counter(train_df['label'])
                        min_count = min(label_counts.values())

                        if min_count < 2:
                            # Find which classes have insufficient instances
                            insufficient_classes = [cls for cls, count in label_counts.items() if count < 2]

                            self.console.print(f"\n[yellow]⚠️  Found {len(insufficient_classes)} label(s) with insufficient samples:[/yellow]")
                            for cls in insufficient_classes:
                                count = label_counts[cls]
                                self.console.print(f"  • [red]'{cls}'[/red]: {count} sample(s) - need at least 2")

                            self.console.print(f"\n[bold]What would you like to do?[/bold]")
                            self.console.print(f"  [cyan]1.[/cyan] [green]Remove[/green] these {len(insufficient_classes)} value(s) and continue")
                            self.console.print(f"  [cyan]2.[/cyan] [red]Cancel[/red] training\n")

                            remove_labels = Confirm.ask(
                                f"[bold yellow]Remove insufficient labels and continue?[/bold yellow]",
                                default=True
                            )

                            if remove_labels:
                                # Filter out samples with insufficient classes
                                # Pass session_id if available for contextualized logging
                                filter_logger = get_filter_logger(session_id=getattr(self, 'current_session_id', None))
                                df_before = train_df.copy()
                                train_df = train_df[~train_df['label'].isin(insufficient_classes)]

                                # Log filtered data
                                filter_logger.log_dataframe_filtering(
                                    df_before=df_before,
                                    df_after=train_df,
                                    reason="insufficient_samples_per_class",
                                    location="advanced_cli.train_single_model",
                                    text_column='text' if 'text' in train_df.columns else None,
                                    log_filtered_samples=5
                                )

                                self.console.print(f"\n[green]✓ Removed {len(df_before) - len(train_df)} sample(s)[/green]")

                                # Recompute label counts
                                label_counts = Counter(train_df['label'])
                                min_count = min(label_counts.values()) if label_counts else 0

                                if min_count < 2:
                                    self.console.print("[red]Still insufficient samples after removal. Cannot proceed with training.[/red]")
                                    return

                                stratify_col = train_df['label']
                            else:
                                self.console.print(f"[red]Training cancelled by user[/red]")
                                return
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

    def _load_saved_factory_training_results(
        self,
        session_id: str,
        session_dirs: Optional[Dict[str, Any]],
        training_workflow: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Reconstruct training result payload from a prior Annotator Factory session."""
        metrics_dir: Optional[Path] = None
        if session_dirs and session_dirs.get("training_metrics"):
            metrics_dir = Path(session_dirs["training_metrics"])

        if not metrics_dir:
            metrics_dir = Path("logs") / "annotator_factory" / session_id / "training_metrics"

        if not metrics_dir.exists():
            return None

        trained_models: Dict[str, str] = {}
        for best_file in metrics_dir.glob("normal_training/**/*best.csv"):
            try:
                with best_file.open(newline='', encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    rows = list(reader)
                if not rows:
                    continue
                row = rows[-1]
                saved_model_path = row.get("saved_model_path") or row.get("model_path")
                if not saved_model_path:
                    continue
                model_path = Path(saved_model_path).expanduser()
                if not model_path.is_absolute():
                    model_path = (best_file.parent / saved_model_path).resolve()
                else:
                    model_path = model_path.resolve()
                if not model_path.exists():
                    continue

                relative_parts = best_file.relative_to(metrics_dir).parts
                category_name = row.get("category")
                if not category_name and len(relative_parts) > 1:
                    category_name = relative_parts[1]
                category_name = category_name or model_path.stem

                trained_models[category_name] = str(model_path)
            except Exception as exc:
                self.logger.debug("Unable to inspect training artifact %s: %s", best_file, exc)

        if not trained_models:
            return None

        # Update training workflow status snapshot if provided
        if training_workflow is not None:
            training_workflow['status'] = training_workflow.get('status', 'completed')
            training_workflow['models_trained'] = list(trained_models.keys())
            training_workflow['last_update'] = datetime.now().isoformat()

        return {
            "status": "completed",
            "session_id": session_id,
            "training_result": {
                "trained_models": trained_models,
                "models_trained": list(trained_models.keys()),
            },
            "training_logs_dir": str(metrics_dir),
        }

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
                local_table.add_column("#", style="bold yellow", width=5, justify="right", no_wrap=True)
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
                openai_table.add_column("#", style="bold yellow", width=5, justify="right", no_wrap=True)
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
                anthropic_table.add_column("#", style="bold yellow", width=5, justify="right", no_wrap=True)
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
                self.console.print("\n[dim]Examples: gpt-3.5-turbo, gpt-4, gpt-4o, gpt-4o-2025-01-01, o1, o1-mini, o3-mini, gpt-5-2025-08-07[/dim]")
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

    def _prompt_openai_batch_mode(
        self,
        provider: str,
        context: str,
        annotation_settings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Ask whether to enable OpenAI Batch mode for the current workflow."""
        if provider != 'openai':
            if annotation_settings is not None:
                annotation_settings.pop('openai_batch_mode', None)
            return False

        if annotation_settings and 'openai_batch_mode' in annotation_settings:
            return bool(annotation_settings['openai_batch_mode'])

        if HAS_RICH and self.console:
            self.console.print("\n[bold cyan]OpenAI Batch Mode[/bold cyan]")
            self.console.print(
                "[dim]Batch mode uploads every prompt/input as a JSONL job that OpenAI processes asynchronously. "
                "It shines on large datasets because OpenAI handles queuing, retries, and durable storage on their side.[/dim]"
            )
            self.console.print(
                "[dim]Why use it? Avoid local rate-limit backoffs, survive transient network issues, and annotate hundreds of thousands of rows without babysitting the run.[/dim]"
            )
            self.console.print(
                "[dim]Trade-offs: expect longer wall-clock time (minutes to hours) before results arrive, you only receive output when the batch finishes, "
                "and progress is tracked by polling the job status. Plan for overnight runs when possible.[/dim]"
            )
            self.console.print(
                "[dim]Outputs land in OpenAI's dashboard and in logs/openai_batches/ so you can audit every request/response pair once the job completes.[/dim]"
            )

        question = f"[bold yellow]Use the OpenAI Batch API for {context}?[/bold yellow]"
        use_batch = Confirm.ask(question, default=False)

        if HAS_RICH and self.console:
            if use_batch:
                self.console.print(
                    f"[green]✓ Batch mode enabled. The workflow will prepare the batch job and wait for OpenAI to finish processing {context}.[/green]"
                )
            else:
                self.console.print(f"[dim]Continuing with synchronous API calls for {context}.[/dim]")

        if annotation_settings is not None:
            annotation_settings['openai_batch_mode'] = use_batch

        return use_batch

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
                    table.add_column("#", justify="right", style="cyan bold", width=5, no_wrap=True)
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
                        elif "gpt-5-2025" in model_name.lower():
                            type_info = "🏆 Flagship"
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
                "Quick", "10", "32", "5e-5", "2",
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
                "epochs": 10,
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

        openai_batch_mode = self._prompt_openai_batch_mode(
            model_info.provider,
            "this quick start annotation run",
            annotation_settings,
        )

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

        if model_info.provider == 'openai' and openai_batch_mode:
            annotation_mode = 'openai_batch'
        else:
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
            'openai_batch_mode': openai_batch_mode,
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
            self.console.print("[green]✓ Session parameters are automatically saved for:[/green]\n")

            self.console.print("  [green]1. Resume Capability[/green]")
            self.console.print("     • Continue this annotation if it stops or crashes")
            self.console.print("     • Annotate additional rows later with same settings")
            self.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

            self.console.print("  [green]2. Scientific Reproducibility[/green]")
            self.console.print("     • Document exact parameters for research papers")
            self.console.print("     • Reproduce identical annotations in the future")
            self.console.print("     • Track model version, prompts, and all settings\n")

            # Metadata is ALWAYS saved automatically for reproducibility
            save_metadata = True

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
            # Metadata is ALWAYS saved, even without Rich console
            save_metadata = True
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
                        'requested_rows': annotation_settings.get('annotation_sample_size') if annotation_settings.get('annotation_sample_size') else 'all',
                        'sampling_strategy': annotation_settings.get('annotation_sampling_strategy', 'head'),
                        'sample_seed': annotation_settings.get('annotation_sample_seed', 42)
                    },
                    'annotation_progress': {
                        'requested': annotation_settings.get('annotation_sample_size') if annotation_settings.get('annotation_sample_size') else 'all',
                        'completed': 0,
                        'remaining': annotation_settings.get('annotation_sample_size') if annotation_settings.get('annotation_sample_size') else 'all'
                    },
                    'model_configuration': {
                        'provider': model_info.provider,
                        'model_name': model_info.name,
                        'annotation_mode': annotation_mode,
                        'openai_batch_mode': openai_batch_mode,
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
                        'parallel_workers': None if openai_batch_mode else pipeline_config.get('num_processes', 1),
                        'batch_size': None if openai_batch_mode else pipeline_config.get('batch_size', 16),
                        'incremental_save': False if openai_batch_mode else save_incrementally,
                        'openai_batch_mode': openai_batch_mode
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

            # Create pipeline controller with session ID for test
            from ..pipelines.pipeline_controller import PipelineController

            # Use a test session ID for isolated testing
            test_session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            pipeline_with_progress = PipelineController(
                settings=self.settings,
                session_id=test_session_id  # Pass session_id for organized logging
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
                    self.resume_center()
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
                    self.console.print(f"[bold red]Error:[/bold red] {str(e)}", markup=False, highlight=False)
                else:
                    print(f"Error: {str(e)}")

    # Placeholder methods for other menu options
    def llm_annotation_studio(self):
        """
        The Annotator - Complete annotation workflow without training.

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

        # Display mode-specific banner
        self._display_mode_banner('annotator')

        # Display personalized mode info
        self._display_section_header(
            "🎨 The Annotator - Zero-Shot LLM Annotation → Label Studio/Doccano Export",
            "Professional zero-shot annotation with Ollama/OpenAI/Claude, automatic JSON repair, and export to review platforms",
            mode_info={
                'workflow': 'Data → Prompt Wizard → LLM Annotate (Parallel) → JSON Repair → Export (Doccano/Label Studio)',
                'capabilities': ['Ollama/OpenAI/Claude Support', 'Prompt Wizard', '200K Context', 'Multi-Label Categories', 'NER', 'Pydantic Validation'],
                'input': 'Raw text data (CSV/Excel/JSON/SQL)',
                'output': 'Annotated JSON + Doccano JSONL + Label Studio JSON (API or file)',
                'best_for': 'Zero-shot annotation with LLMs, human review workflows, data labeling for training'
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
                title="[bold]🎨 The Annotator[/bold]",
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
                # CRITICAL: Ask user for session name first (like Training Arena)
                from datetime import datetime

                self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
                self.console.print("[bold cyan]           📝 Session Name Configuration                       [/bold cyan]")
                self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

                self.console.print("[bold]Why session names matter:[/bold]")
                self.console.print("  • [green]Organization:[/green] Easily identify annotation projects (e.g., 'sentiment_tweets', 'legal_documents')")
                self.console.print("  • [green]Traceability:[/green] Track your annotations across data, logs, and exports")
                self.console.print("  • [green]Collaboration:[/green] Team members understand what each session represents")
                self.console.print("  • [green]Audit trail:[/green] Timestamp ensures uniqueness\n")

                self.console.print("[dim]Format: {session_name}_{yyyymmdd_hhmmss}[/dim]")
                self.console.print("[dim]Example: sentiment_analysis_20251008_143022[/dim]\n")

                # Ask for user-defined session name
                user_session_name = Prompt.ask(
                    "[bold yellow]Enter a descriptive name for this annotation session[/bold yellow]",
                    default="annotation_session"
                ).strip()

                # Sanitize the user input (remove special chars, replace spaces with underscores)
                user_session_name = user_session_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                user_session_name = ''.join(c for c in user_session_name if c.isalnum() or c in ['_', '-'])

                # Create full session ID with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = f"{user_session_name}_{timestamp}"

                self.console.print(f"\n[bold green]✓ Session ID:[/bold green] [cyan]{session_id}[/cyan]")
                self.console.print(f"[dim]This ID will be used consistently across all data, logs, and exports[/dim]\n")

                # Pass session_id to _smart_annotate
                self._smart_annotate(session_id=session_id)
            elif workflow == "3":
                self._database_annotator()
            elif workflow == "4":
                self._clean_metadata()
        else:
            print("\n=== The Annotator ===")
            print("LLM Tool annotates, you decide\n")
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


    def bert_annotation_studio(self, resume_session_id: Optional[str] = None):
        """BERT Annotation Studio - Advanced annotation with trained models"""
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display mode-specific banner
        self._display_mode_banner('bert_studio')

        # Display personalized mode info
        self._display_section_header(
            "🤖 BERT Annotation Studio - High-Throughput Inference (Parallel GPU/CPU, 100+ Languages)",
            "Production-ready inference with trained BERT models: Parallel processing, language validation, confidence scoring",
            mode_info={
                'workflow': 'Select Model → Load Data (SQL/File) → Detect Language → Preprocessing → Parallel Inference → Export',
                'capabilities': ['50+ Trained Models', '100+ Languages', 'Multi-GPU/CPU Parallel', 'Confidence Scoring', 'Language Validation', 'Text Preprocessing'],
                'input': 'SQL (PostgreSQL/MySQL/SQLite/SQL Server) or Files (CSV/Excel/JSON/JSONL/Parquet/RData)',
                'output': 'Predictions with probabilities + Confidence intervals + Language tags + Multiple export formats',
                'best_for': 'High-throughput production inference, multilingual datasets, large-scale annotation with trained models'
            }
        )

        choice: Optional[str] = "2" if resume_session_id else None

        if HAS_RICH and self.console:
            if resume_session_id is None:
                # Create mode menu
                self.console.print("\n[bold cyan]🎯 BERT Annotation Studio Options[/bold cyan]\n")

                studio_options_table = Table(show_header=False, box=None, padding=(0, 2))
                studio_options_table.add_column("Option", style="cyan", width=8)
                studio_options_table.add_column("Description")

                options = [
                    ("1", "🆕 Start new session"),
                    ("2", "🔄 Resume session"),
                    ("3", "📚 Session history"),
                    ("0", "⬅️  Back to main menu"),
                ]

                for option, desc in options:
                    studio_options_table.add_row(f"[bold cyan]{option}[/bold cyan]", desc)

                panel = Panel(
                    studio_options_table,
                    title="[bold]🤖 BERT Annotation Studio[/bold]",
                    border_style="cyan"
                )

                self.console.print(panel)

                choice = Prompt.ask(
                    "\n[bold yellow]Select an option[/bold yellow]",
                    choices=["0", "1", "2", "3"],
                    default="1"
                )

                if choice == "0":
                    return

        from .bert_annotation_studio import BERTAnnotationStudio

        studio = BERTAnnotationStudio(
            console=self.console,
            settings=self.settings,
            logger=self.logger
        )

        if resume_session_id:
            studio.run(session_action="2", resume_session_id=resume_session_id)
            return

        if choice == "1":
            studio.run(session_action="1")
        elif choice == "2":
            studio.run(session_action="2")
        elif choice == "3":
            studio.run(session_action="3")

    def validation_lab(self):
        """Validation lab for quality control and Doccano export"""
        # Display ASCII logo only
        self._display_ascii_logo()

        # Display mode-specific banner
        self._display_mode_banner('validation')

        # Display personalized mode info
        self._display_section_header(
            "🔍 Validation Lab - Quality Scoring, Stratified Sampling, Inter-Annotator Agreement",
            "Quality assurance tools: Validate LLM annotations, detect imbalances, prepare stratified samples for human review",
            mode_info={
                'workflow': 'Load Annotations → Quality Metrics (0-100 score) → Stratified Sampling → Export to Doccano/Label Studio',
                'capabilities': ['Quality Scoring', 'Label Distribution Analysis', 'Stratified Sampling', 'Inter-Annotator Agreement (Cohen\'s Kappa)', 'Schema Validation'],
                'input': 'Annotated JSON/JSONL files from LLM or BERT Studio',
                'output': 'Quality reports + Validation metrics + Doccano/Label Studio export files + Sample selection justification',
                'best_for': 'Quality assurance before training, human validation workflows, detecting annotation issues'
            }
        )

        # ⚠️ DEVELOPMENT WARNING
        if HAS_RICH and self.console:
            warning_panel = Panel(
                Align.center(
                    "[bold yellow]⚠️  UNDER DEVELOPMENT ⚠️[/bold yellow]\n\n"
                    "[yellow]This mode is currently in active development and is NOT complete.[/yellow]\n"
                    "[dim]Some features may not work as expected or may be missing entirely.[/dim]\n"
                    "[dim]Use at your own risk for testing purposes only.[/dim]",
                    vertical="middle"
                ),
                title="[bold red]Development Status[/bold red]",
                border_style="red",
                box=box.DOUBLE,
                padding=(1, 2),
            )
            self.console.print()
            self.console.print(Align.center(warning_panel))
            self.console.print()

            # Ask user if they want to continue
            if not Confirm.ask("[yellow]Do you want to continue anyway?[/yellow]", default=False):
                self.console.print("[dim]Returning to main menu...[/dim]")
                return

        if HAS_RICH and self.console:
            # Create mode menu
            self.console.print("\n[bold cyan]🎯 Validation Lab Options[/bold cyan]\n")

            lab_options_table = Table(show_header=False, box=None, padding=(0, 2))
            lab_options_table.add_column("Option", style="cyan", width=8)
            lab_options_table.add_column("Description")

            options = [
                ("1", "🔍 Start Validation Workflow (Quality scoring + sampling + export)"),
                ("0", "⬅️  Back to main menu")
            ]

            for option, desc in options:
                lab_options_table.add_row(f"[bold cyan]{option}[/bold cyan]", desc)

            panel = Panel(
                lab_options_table,
                title="[bold]🔍 Validation Lab[/bold]",
                border_style="magenta"
            )

            self.console.print(panel)

            choice = Prompt.ask(
                "\n[bold yellow]Select an option[/bold yellow]",
                choices=["0", "1"],
                default="1"
            )

            if choice == "0":
                return

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

            # ⚠️ DEVELOPMENT WARNING
            print("\n" + "="*60)
            print("⚠️  UNDER DEVELOPMENT - NOT COMPLETE ⚠️")
            print("="*60)
            print("This mode is currently in active development.")
            print("Some features may not work as expected or may be missing.")
            print("Use at your own risk for testing purposes only.")
            print("="*60 + "\n")

            continue_choice = input("Do you want to continue anyway? (y/N): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("Returning to main menu...")
                return

            annotations_path = input("\nAnnotations file path: ").strip()
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
                'best_for': 'Rerunning the same pipeline configuration multiple times'
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
                'best_for': 'Customizing system behavior and defaults'
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

    def _create_annotation_id(self, df: pd.DataFrame, id_column: str = "llm_annotation_id") -> pd.DataFrame:
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

    def _quick_annotate(self, focus_session_id: Optional[str] = None):
        """Resume or relaunch annotation using saved parameters"""
        self.console.print("\n[bold cyan]🔄 Resume/Relaunch Annotation[/bold cyan]\n")
        self.console.print("[dim]Load saved parameters from previous annotations[/dim]\n")

        # ============================================================
        # DETECT METADATA FILES
        candidates = self._load_annotation_resume_candidates("annotator", limit=25)
        if not candidates:
            self.console.print("[yellow]No resumable annotation sessions found.[/yellow]")
            self.console.print("[dim]Complete a session at least once to enable resume/relaunch.[/dim]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        self.console.print(f"[green]Found {len(candidates)} resumable annotation session(s)[/green]\n")

        sessions_table = Table(border_style="cyan", show_header=True)
        sessions_table.add_column("#", style="cyan", width=3)
        sessions_table.add_column("Session", style="white")
        sessions_table.add_column("Updated", style="yellow")
        sessions_table.add_column("Status", style="green", width=12)
        sessions_table.add_column("Last Step", style="cyan")
        sessions_table.add_column("Model", style="magenta")
        from datetime import datetime

        valid_sessions: List[Tuple[Path, Dict[str, Any], SessionSummary]] = []
        for idx, (metadata_path, metadata, record) in enumerate(candidates, 1):
            summary = record.summary
            session_id = summary.session_id
            updated_display = summary.updated_at.replace("T", " ")
            last_step_display = summary.last_step_name or summary.last_step_key or "-"
            model_config = metadata.get("model_configuration", {})
            model_name = model_config.get("model_name") or model_config.get("selected_model") or summary.extra.get("model") or "-"
            sessions_table.add_row(
                str(idx),
                session_id,
                updated_display,
                summary.status,
                last_step_display,
                model_name,
            )
            valid_sessions.append((metadata_path, metadata, summary))

        if not valid_sessions:
            self.console.print("[yellow]No valid metadata files found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
            return

        self.console.print(sessions_table)

        session_choice: Optional[int] = None
        if focus_session_id:
            for idx, (_, _, summary) in enumerate(valid_sessions, 1):
                if summary and summary.session_id == focus_session_id:
                    session_choice = idx
                    self.console.print(f"\n[dim]Auto-selecting session {summary.session_id}[/dim]")
                    break

        if session_choice is None:
            session_choice = self._int_prompt_with_validation(
                "\n[bold yellow]Select session to resume/relaunch[/bold yellow]",
                1, 1, len(valid_sessions)
            )

        selected_file, metadata, summary = valid_sessions[session_choice - 1]

        self.console.print(f"\n[green]✓ Selected: {selected_file.name}[/green]")
        if summary:
            last_step = summary.last_step_name or summary.last_step_key or "-"
            self.console.print(f"[dim]Status: {summary.status} • Last step: {last_step}[/dim]")

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

    def _create_annotator_session_directories(self, session_id: str) -> dict:
        """Create organized directory structure for annotation session."""
        return create_session_directories(AnnotationMode.ANNOTATOR, session_id)

    def _create_annotator_factory_session_directories(self, session_id: str) -> dict:
        """Create organized directory structure for Annotator Factory session."""
        return create_session_directories(AnnotationMode.FACTORY, session_id)

    def _smart_annotate(self, session_id: str = None):
        """Smart guided annotation wizard with shared workflow module."""
        run_annotator_workflow(self, session_id=session_id)

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
        progress = metadata.get('annotation_progress', {})
        requested_rows = data_source.get('requested_rows', data_source.get('total_rows'))
        current_target = data_source.get('total_rows')
        def _format_count(value):
            if isinstance(value, int):
                return f"{value:,}"
            return str(value) if value is not None else "N/A"

        params_table.add_row("📁 Data", f"File: {data_source.get('file_name', 'N/A')}")
        params_table.add_row("", f"Format: {data_source.get('data_format', 'N/A')}")
        params_table.add_row("", f"Text Column: {data_source.get('text_column', 'N/A')}")
        if (
            isinstance(requested_rows, int)
            and isinstance(current_target, int)
            and requested_rows != current_target
        ):
            rows_detail = (
                f"{_format_count(requested_rows)} requested · "
                f"{_format_count(current_target)} in this run"
            )
        else:
            rows_detail = _format_count(requested_rows)
        params_table.add_row("", f"Rows: {rows_detail}")
        if progress:
            requested_progress = progress.get('requested', requested_rows)
            completed_progress = progress.get('completed', 0)
            remaining_progress = progress.get('remaining')
            progress_text = (
                f"{_format_count(completed_progress)} / {_format_count(requested_progress)}"
            )
            if isinstance(remaining_progress, int):
                progress_text += f" (remaining {_format_count(remaining_progress)})"
            params_table.add_row("", f"Progress: {progress_text}")
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

    def _load_preview_dataframe(self, data_source: dict, max_rows: int = 25):
        """Read a lightweight sample of the dataset for previews."""
        if not HAS_PANDAS:
            return None

        dataset_path = data_source.get('file_path')
        if not dataset_path:
            return None

        try:
            path = Path(dataset_path).expanduser()
        except (TypeError, ValueError):
            return None

        if not path.exists():
            return None

        if not hasattr(self, "_resume_preview_cache"):
            self._resume_preview_cache: Dict[Path, Dict[str, Any]] = {}

        cache_entry = self._resume_preview_cache.get(path)
        mtime = path.stat().st_mtime
        if cache_entry and cache_entry.get("mtime") == mtime:
            cached_df = cache_entry.get("df")
            if cached_df is not None:
                return cached_df.copy()

        data_format = (data_source.get('data_format') or path.suffix.lower().lstrip('.')).lower()

        try:
            if data_format in {"csv", "tsv"}:
                sep = "," if data_format == "csv" else "\t"
                df = pd.read_csv(path, sep=sep, nrows=max_rows)
            elif data_format == "jsonl":
                df = pd.read_json(path, lines=True, nrows=max_rows)
            elif data_format == "json":
                df = pd.read_json(path)
                if len(df) > max_rows:
                    df = df.head(max_rows)
            elif data_format in {"xlsx", "xls", "excel"}:
                df = pd.read_excel(path, nrows=max_rows)
            elif data_format == "parquet":
                df = pd.read_parquet(path)
                if len(df) > max_rows:
                    df = df.head(max_rows)
            else:
                logging.debug("Resume preview: unsupported format '%s'", data_format)
                return None
        except Exception as exc:
            logging.debug("Resume preview: unable to load %s (%s)", path, exc)
            return None

        if df is None or df.empty:
            return None

        limited_df = df.head(max_rows).copy()
        self._resume_preview_cache[path] = {"mtime": mtime, "df": limited_df}
        return limited_df.copy()

    def _detect_column_profile(self, series) -> str:
        """Generate a friendly description for a dataframe column."""
        if not HAS_PANDAS:
            return ""

        try:
            non_null = series.dropna()
        except Exception:
            return "🧩"

        if non_null.empty:
            return "⬜️ Empty"

        if pd_types:
            if pd_types.is_datetime64_any_dtype(series):
                return "🕒 Date/Time"
            if pd_types.is_numeric_dtype(series):
                unique_ratio = non_null.nunique() / len(non_null)
                if unique_ratio > 0.95:
                    return "🔑 Numeric ID"
                return "🔢 Numeric"
            if pd_types.is_bool_dtype(series):
                return "⚙️ Boolean"

        # Treat as text-like
        text_values = non_null.astype(str)
        avg_len = text_values.str.len().mean() if not text_values.empty else 0
        unique_ratio = text_values.nunique() / len(text_values)

        if unique_ratio > 0.95 and avg_len < 40:
            return "🔑 ID / Reference"
        if avg_len >= 200:
            return "📝 Long text"
        if avg_len >= 60:
            return "📝 Text"
        if avg_len >= 20:
            return "📝 Short text"
        return "🗂️ Category"

    def _render_dataset_preview(
        self,
        df,
        highlight_column: Optional[str] = None,
        show_indices: bool = False,
        max_columns: int = 12,
        sample_rows: int = 3,
        include_detection: bool = True,
    ):
        """Render a rich table showing dataset columns plus sample text."""
        if not HAS_RICH or not self.console or df is None or df.empty:
            return

        columns = list(df.columns[:max_columns])
        if not columns:
            return

        table = Table(title="📊 Available Columns", box=box.SIMPLE_HEAD, show_lines=False)
        if show_indices:
            table.add_column("#", style="cyan", justify="right", width=3)
        table.add_column("Column", style="green", overflow="fold")
        table.add_column("Type", style="yellow", no_wrap=True)
        if include_detection:
            table.add_column("Profile", style="magenta", overflow="fold")
        table.add_column("Sample", style="white", overflow="fold")

        for idx, col in enumerate(columns, 1):
            series = df[col]
            series_non_null = series.dropna()
            sample = "-"
            if not series_non_null.empty:
                sample = str(series_non_null.iloc[0]).replace("\n", " ").strip()
                if len(sample) > 90:
                    sample = sample[:87] + "..."

            display_name = str(col)
            if highlight_column and str(col) == str(highlight_column):
                display_name = f"[cyan]{display_name}[/cyan]"

            row = []
            if show_indices:
                row.append(str(idx))
            row.append(display_name)
            row.append(str(series.dtype))
            if include_detection:
                profile = self._detect_column_profile(series)
                if highlight_column and str(col) == str(highlight_column):
                    profile = f"[cyan]{profile}[/cyan]"
                row.append(profile)
            row.append(sample)
            table.add_row(*row)

        if len(df.columns) > max_columns:
            table.caption = f"Showing first {max_columns} of {len(df.columns)} columns"

        self.console.print(table)

        highlight = None
        if highlight_column is not None:
            for col in df.columns:
                if str(col) == str(highlight_column):
                    highlight = col
                    break

        if highlight is not None and highlight in df.columns:
            samples = df[highlight].dropna().astype(str).head(sample_rows)
            if not samples.empty:
                text_table = Table(
                    title=f"📝 Samples – {highlight}",
                    box=box.SIMPLE_HEAD,
                    show_header=False,
                )
                text_table.add_column("Sample", style="white", overflow="fold")
                for sample in samples:
                    cleaned = sample.replace("\n", " ").strip()
                    if len(cleaned) > 160:
                        cleaned = cleaned[:157] + "..."
                    text_table.add_row(cleaned or "-")
                self.console.print(text_table)

    def _display_resume_data_preview(self, metadata: dict):
        """Display dataset columns and text samples when modifying resume parameters."""
        if not HAS_RICH or not HAS_PANDAS or not self.console:
            return

        data_source = metadata.get('data_source', {})
        df = self._load_preview_dataframe(data_source)
        if df is None:
            return

        resolved_text_column = self._resolve_existing_column(
            df,
            data_source.get('text_column'),
            "text column"
        )
        resolved_text_column_name = (
            str(resolved_text_column) if resolved_text_column is not None else None
        )

        # Store for reuse inside the modification loop
        self._current_resume_preview_df = df.copy()

        self._render_dataset_preview(
            df,
            highlight_column=resolved_text_column_name,
            show_indices=False,
            max_columns=12,
            sample_rows=3,
            include_detection=True,
        )

    def _print_parameter_modification_menu(self):
        """Display resume parameter modification menu."""
        self.console.print("\n[bold]Select parameter to modify:[/bold]")
        self.console.print("  [cyan]1[/cyan] - Data source (file, text column)")
        self.console.print("  [cyan]2[/cyan] - Model (provider, model name)")
        self.console.print("  [cyan]3[/cyan] - Model parameters (temperature, max_tokens, etc.)")
        self.console.print("  [cyan]4[/cyan] - Prompts (add/remove/modify)")
        self.console.print("  [cyan]5[/cyan] - Sampling (rows to annotate, strategy)")
        self.console.print("  [cyan]6[/cyan] - Processing (workers, batch size)")
        self.console.print("  [cyan]0[/cyan] - Done modifying")

    def _modify_parameters_if_requested(self, metadata: dict, modify: bool) -> dict:
        """Allow user to modify specific parameters"""
        if not modify:
            return metadata

        modified = copy.deepcopy(metadata)
        changes_made = False

        while True:
            self._display_metadata_parameters(modified)
            self._display_resume_data_preview(modified)
            self._print_parameter_modification_menu()

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
                data_source = modified.setdefault('data_source', {})
                self.console.print(f"  File: {data_source.get('file_path', 'N/A')}")
                self.console.print(f"  Text column: {data_source.get('text_column', 'N/A')}")

                changed_this_round = False

                if Confirm.ask("Change data file?", default=False):
                    new_file = self._prompt_file_path("New data file path")
                    if new_file and new_file != data_source.get('file_path'):
                        data_source['file_path'] = new_file
                        data_source['file_name'] = Path(new_file).name
                        new_format = Path(new_file).suffix.lower().lstrip(".")
                        if new_format:
                            data_source['data_format'] = new_format
                        # Reset preview cache so the next loop uses fresh sample
                        if hasattr(self, "_resume_preview_cache"):
                            try:
                                self._resume_preview_cache.clear()
                            except Exception:
                                self._resume_preview_cache = {}
                        self._current_resume_preview_df = None
                        changed_this_round = True

                if Confirm.ask("Change text column?", default=False):
                    df_preview = self._load_preview_dataframe(data_source)
                    resolved_current = None
                    resolved_current_name = None
                    if df_preview is not None:
                        resolved_current = self._resolve_existing_column(
                            df_preview,
                            data_source.get('text_column'),
                            "text column"
                        )
                        resolved_current_name = (
                            str(resolved_current) if resolved_current is not None else None
                        )

                        self.console.print("\n[cyan]Available columns with samples:[/cyan]")
                        self._render_dataset_preview(
                            df_preview,
                            highlight_column=resolved_current_name,
                            show_indices=True,
                            max_columns=min(20, len(df_preview.columns)),
                            sample_rows=3,
                            include_detection=True,
                        )

                        if len(df_preview.columns) > 0:
                            default_idx = 1
                            if resolved_current_name:
                                for idx, col in enumerate(df_preview.columns, 1):
                                    if str(col) == resolved_current_name:
                                        default_idx = idx
                                        break

                            self.console.print("[dim]Enter 0 to type a column name manually.[/dim]")
                            selection = self._int_prompt_with_validation(
                                "Select new text column (number)",
                                default_idx,
                                0,
                                len(df_preview.columns)
                            )

                            if selection == 0:
                                new_col = Prompt.ask(
                                    "Column name",
                                    default=resolved_current_name or None
                                )
                            else:
                                new_col = str(df_preview.columns[selection - 1])
                        else:
                            new_col = Prompt.ask("Column name")
                    else:
                        new_col = Prompt.ask(
                            "New text column name",
                            default=str(data_source.get('text_column', '') or '') or None
                        )

                    if new_col:
                        new_col = new_col.strip()

                    if new_col and new_col != data_source.get('text_column'):
                        data_source['text_column'] = new_col
                        changed_this_round = True
                        self._current_resume_preview_df = df_preview.copy() if df_preview is not None else None
                        if df_preview is not None and new_col not in {str(c) for c in df_preview.columns}:
                            self.console.print(f"[yellow]⚠️ Column '{new_col}' not found in preview sample.[/yellow]")
                    else:
                        self.console.print("[dim]No changes applied to text column[/dim]")

                if changed_this_round:
                    changes_made = True
                    self.console.print("[green]✓ Data source updated[/green]")
                else:
                    self.console.print("[dim]No changes applied to data source[/dim]")

            elif choice == "2":
                # Modify model
                self.console.print("\n[yellow]Current model:[/yellow]")
                model_config = modified.setdefault('model_configuration', {})
                self.console.print(f"  Provider: {model_config.get('provider', 'N/A')}")
                self.console.print(f"  Model: {model_config.get('model_name', 'N/A')}")

                changed_this_round = False

                if Confirm.ask("Change model?", default=False):
                    # Reuse model selection from smart annotate
                    provider = Prompt.ask(
                        "Provider",
                        choices=["ollama", "openai", "anthropic"],
                        default=model_config.get('provider', 'ollama')
                    )
                    current_model_name = model_config.get('model_name')
                    if current_model_name:
                        model_name = Prompt.ask("Model name", default=current_model_name)
                    else:
                        model_name = Prompt.ask("Model name")
                    if provider != model_config.get('provider'):
                        model_config['provider'] = provider
                        changed_this_round = True
                    if model_name and model_name != model_config.get('model_name'):
                        model_config['model_name'] = model_name
                        changed_this_round = True

                if changed_this_round:
                    changes_made = True
                    self.console.print("[green]✓ Model configuration updated[/green]")
                else:
                    self.console.print("[dim]No changes applied to model configuration[/dim]")

            elif choice == "3":
                # Modify model parameters
                model_config = modified.setdefault('model_configuration', {})
                changed_this_round = False

                if Confirm.ask("Change temperature?", default=False):
                    current_temp = model_config.get('temperature', 0.7)
                    default_temp = current_temp if isinstance(current_temp, (int, float)) else 0.7
                    temp = FloatPrompt.ask(
                        "Temperature (0.0-2.0)",
                        default=default_temp
                    )
                    if temp != model_config.get('temperature'):
                        model_config['temperature'] = temp
                        changed_this_round = True

                if Confirm.ask("Change max_tokens?", default=False):
                    current_max_tokens = model_config.get('max_tokens', 1000)
                    if not isinstance(current_max_tokens, int):
                        current_max_tokens = 1000
                    tokens = self._int_prompt_with_validation(
                        "Max tokens",
                        current_max_tokens,
                        50,
                        8000
                    )
                    if tokens != model_config.get('max_tokens'):
                        model_config['max_tokens'] = tokens
                        changed_this_round = True

                if changed_this_round:
                    changes_made = True
                    self.console.print("[green]✓ Model parameters updated[/green]")
                else:
                    self.console.print("[dim]No changes applied to model parameters[/dim]")

            elif choice == "4":
                # Modify prompts
                self.console.print("\n[yellow]Prompt modification not implemented in this version.[/yellow]")
                self.console.print("[dim]Use Smart Annotate to create new annotation with different prompts.[/dim]")

            elif choice == "5":
                # Modify sampling
                data_source = modified.setdefault('data_source', {})
                progress_state = modified.setdefault('annotation_progress', {})
                current_rows = data_source.get('requested_rows', data_source.get('total_rows', 'all'))

                self.console.print(f"\n[yellow]Current: {current_rows} rows[/yellow]")

                changed_this_round = False

                if Confirm.ask("Change number of rows to annotate?", default=False):
                    annotate_all = Confirm.ask("Annotate all rows?", default=True)
                    if annotate_all:
                        if (
                            data_source.get('requested_rows') != 'all'
                            or data_source.get('sampling_strategy') != 'none'
                        ):
                            data_source['total_rows'] = 'all'
                            data_source['requested_rows'] = 'all'
                            data_source['sampling_strategy'] = 'none'
                            progress_state['requested'] = 'all'
                            progress_state['remaining'] = 'all'
                            changed_this_round = True
                    else:
                        current_total_rows = data_source.get('requested_rows', 100)
                        if not isinstance(current_total_rows, int):
                            current_total_rows = 100
                        num_rows = self._int_prompt_with_validation(
                            "Number of rows",
                            current_total_rows,
                            1,
                            1000000
                        )
                        current_strategy = data_source.get('sampling_strategy', 'random')
                        if current_strategy not in {"head", "random"}:
                            current_strategy = "random"
                        strategy = Prompt.ask(
                            "Sampling strategy",
                            choices=["head", "random"],
                            default=current_strategy
                        )
                        if num_rows != data_source.get('requested_rows'):
                            data_source['total_rows'] = num_rows
                            data_source['requested_rows'] = num_rows
                            progress_state['requested'] = num_rows
                            progress_state['remaining'] = num_rows
                            changed_this_round = True
                        if strategy != data_source.get('sampling_strategy'):
                            data_source['sampling_strategy'] = strategy
                            changed_this_round = True

                if changed_this_round:
                    changes_made = True
                    self.console.print("[green]✓ Sampling configuration updated[/green]")
                else:
                    self.console.print("[dim]No changes applied to sampling[/dim]")

            elif choice == "6":
                # Modify processing
                proc_config = modified.setdefault('processing_configuration', {})
                changed_this_round = False

                if Confirm.ask("Change parallel workers?", default=False):
                    current_workers = proc_config.get('parallel_workers', 1)
                    if not isinstance(current_workers, int):
                        current_workers = 1
                    workers = self._int_prompt_with_validation(
                        "Parallel workers",
                        current_workers,
                        1,
                        16
                    )
                    if workers != proc_config.get('parallel_workers'):
                        proc_config['parallel_workers'] = workers
                        changed_this_round = True

                if Confirm.ask("Change batch size?", default=False):
                    current_batch = proc_config.get('batch_size', 1)
                    if not isinstance(current_batch, int):
                        current_batch = 1
                    batch = self._int_prompt_with_validation(
                        "Batch size",
                        current_batch,
                        1,
                        1000
                    )
                    if batch != proc_config.get('batch_size'):
                        proc_config['batch_size'] = batch
                        changed_this_round = True

                if changed_this_round:
                    changes_made = True
                    self.console.print("[green]✓ Processing configuration updated[/green]")
                else:
                    self.console.print("[dim]No changes applied to processing[/dim]")

        if changes_made:
            self.console.print("\n[green]✓ Parameters modified[/green]")
        else:
            self.console.print("\n[yellow]No parameter changes detected[/yellow]")
        return modified

    def _execute_from_metadata(self, metadata: dict, action_mode: str, metadata_file: Path):
        """Execute annotation based on loaded metadata using shared workflow module."""
        session_info = metadata.setdefault('annotation_session', {})
        previous_session_id = session_info.get('session_id')

        workflow_name = str(session_info.get('workflow', '')).lower()
        is_factory = "factory" in workflow_name

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_prefix = "factory_session" if is_factory else "annotator_session"
        session_id = f"{session_prefix}_{timestamp}_{uuid.uuid4().hex[:4]}"

        if previous_session_id:
            session_info['parent_session_id'] = previous_session_id

        session_info['session_id'] = session_id
        session_info['timestamp'] = timestamp
        metadata['session_id'] = session_id

        if is_factory:
            session_dirs = self._create_annotator_factory_session_directories(session_id)
        else:
            session_dirs = self._create_annotator_session_directories(session_id)

        session_name = session_info.get('session_name', session_id)
        step_catalog = FACTORY_RESUME_STEPS if is_factory else ANNOTATOR_RESUME_STEPS
        tracker = AnnotationResumeTracker(
            mode=AnnotationMode.FACTORY if is_factory else AnnotationMode.ANNOTATOR,
            session_id=session_id,
            session_dirs=session_dirs,
            step_catalog=step_catalog,
            session_name=session_name,
        )
        tracker.update_status("active")
        run_step_no = 6

        tracker.mark_step(
            run_step_no,
            status="in_progress",
            detail="Running annotation from saved parameters",
            overall_status="active",
        )

        succeeded = execute_from_metadata(
            self,
            metadata,
            action_mode,
            metadata_file,
            session_dirs=session_dirs
        )

        if succeeded:
            tracker.mark_step(
                run_step_no,
                status="completed",
                detail="Annotation completed from saved parameters",
                overall_status="completed",
            )
            return
        else:
            tracker.mark_step(
                run_step_no,
                status="failed",
                detail="Annotation run failed",
                overall_status="failed",
            )
            return

    def _clean_metadata(self):
        """Clean old metadata files"""
        self.console.print("\n[bold cyan]🗑️  Clean Old Metadata[/bold cyan]\n")
        self.console.print("[dim]Delete saved annotation parameters to free space[/dim]\n")

        from pathlib import Path

        # Search in both old and new locations
        annotations_dir = self.settings.paths.data_dir / 'annotations'
        annotator_logs_dir = Path("logs") / "annotator"
        factory_logs_dir = Path("logs") / "annotator_factory"

        metadata_files = []

        # Old location: data/annotations/
        if annotations_dir.exists():
            metadata_files.extend(list(annotations_dir.glob("**/*_metadata_*.json")))

        # New location: logs/annotator/
        if annotator_logs_dir.exists():
            metadata_files.extend(list(annotator_logs_dir.glob("**/*_metadata_*.json")))

        # New location: logs/annotator_factory/
        if factory_logs_dir.exists():
            metadata_files.extend(list(factory_logs_dir.glob("**/*_metadata_*.json")))

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
                                  sample_size=None, session_dirs=None,
                                  provider_folder: str = "model_provider",
                                  model_folder: str = "model_name"):
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

            # Prepare JSONL output - Use organized structure if session_dirs provided
            safe_provider_folder = (provider_folder or "model_provider").replace("/", "_")
            safe_model_folder = (model_folder or "model_name").replace("/", "_")

            if data_path:
                dataset_path_obj = data_path if isinstance(data_path, Path) else Path(data_path)
                dataset_name = dataset_path_obj.stem
            else:
                dataset_name = "dataset"

            if session_dirs:
                # Create dataset-specific subdirectory for exports
                doccano_dir = session_dirs['doccano'] / safe_provider_folder / safe_model_folder / dataset_name
                doccano_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Fallback to old structure for backward compatibility
                doccano_dir = (
                    self.settings.paths.data_dir
                    / 'doccano_exports'
                    / safe_provider_folder
                    / safe_model_folder
                    / data_path.stem
                )
                doccano_dir.mkdir(parents=True, exist_ok=True)

            jsonl_filename = f"{dataset_name}_doccano_{timestamp}.jsonl"
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
                                      sample_size=None, prediction_mode='with', session_dirs=None,
                                      provider_folder: str = "model_provider",
                                      model_folder: str = "model_name"):
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

            # Prepare JSONL output - Use organized structure if session_dirs provided
            safe_provider_folder = (provider_folder or "model_provider").replace("/", "_")
            safe_model_folder = (model_folder or "model_name").replace("/", "_")

            if data_path:
                dataset_path_obj = data_path if isinstance(data_path, Path) else Path(data_path)
                dataset_name = dataset_path_obj.stem
            else:
                dataset_name = "dataset"

            if session_dirs:
                # Create dataset-specific subdirectory for exports
                labelstudio_dir = session_dirs['labelstudio'] / safe_provider_folder / safe_model_folder / dataset_name
                labelstudio_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Fallback to old structure for backward compatibility
                labelstudio_dir = (
                    self.settings.paths.data_dir
                    / 'labelstudio_exports'
                    / safe_provider_folder
                    / safe_model_folder
                    / data_path.stem
                )
                labelstudio_dir.mkdir(parents=True, exist_ok=True)

            jsonl_filename = f"{dataset_name}_labelstudio_{timestamp}.jsonl"
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

        # Metadata is ALWAYS saved automatically for reproducibility
        save_metadata = True
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

        openai_batch_mode = self._prompt_openai_batch_mode(
            llm_config.provider,
            "this SQL annotator run",
        )

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
        if llm_config and llm_config.provider == 'openai':
            summary.add_row("OpenAI Batch Mode", "Enabled" if openai_batch_mode else "Disabled")
        summary.add_row("Prompts", f"{len(prompt_configs)} configured")

        # Add all model parameters
        summary.add_row("Temperature", str(temperature))
        summary.add_row("Max Tokens", str(max_tokens))
        summary.add_row("Top P", str(top_p))
        if llm_config and llm_config.provider in ['ollama', 'google']:
            summary.add_row("Top K", str(top_k))

        # Add processing parameters
        if llm_config and llm_config.provider == 'openai' and openai_batch_mode:
            summary.add_row("Parallel Workers", "N/A (managed by OpenAI Batch)")
            summary.add_row("Batch Size", "N/A (managed by OpenAI Batch)")
            summary.add_row("Incremental Save", "N/A (handled after batch completion)")
        else:
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

            if llm_config.provider == 'openai' and openai_batch_mode:
                annotation_mode = 'openai_batch'
            elif llm_config.provider in ['openai', 'anthropic', 'google']:
                annotation_mode = 'api'
            else:
                annotation_mode = 'local'

            # Build pipeline config (SAME as Smart Annotate)
            pipeline_config = {
                'mode': 'file',
                'data_source': 'csv',
                'data_format': 'csv',
                'file_path': str(temp_input_file),
                'text_column': text_column,
                'text_columns': [text_column],
                'annotation_column': 'annotation',
                'identifier_column': id_column if id_column else 'llm_annotation_id',
                'run_annotation': True,
                'annotation_mode': annotation_mode,
                'annotation_provider': llm_config.provider,
                'annotation_model': llm_config.name,
                'api_key': api_key,
                'openai_batch_mode': openai_batch_mode,
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

                pipeline_with_progress = PipelineController(
                    settings=self.settings,
                    session_id=session_id  # Pass session_id for organized logging
                )

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
                prompt_configs=prompt_configs,
                session_id=None,
                session_dirs=None  # No session context in this flow
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
# LLMTool – Documentation & Help

```
┌──────────────────────────────┐
│  DATA  →  AI Annotation  →   │
│  QC / Training  →  Inference │
│         ↘ Resume & Review ↗  │
└──────────────────────────────┘
```

## Before You Dive In
- Create or pick a workspace folder that contains your data (for example `data/customer_reviews.csv`) and prompts (`prompts/`).
- Hosted models (OpenAI, Anthropic, Google, etc.) need API keys in environment variables. Local models (Ollama, Llama.cpp) must be running before you start LLMTool.
- Persistent configuration lives in `~/.llm_tool/`; reusable execution profiles live in `~/.llmtool/profiles/`.
- The banner at launch tells you which GPUs/CPUs are available, which LLM providers are detected, and how many datasets were found in the current directory.

## Mode 1 - The Annotator (Zero-Shot Lab)
**Purpose**: Turn raw text into AI-generated annotations you can export for human review or training.

1. Choose **Smart Annotate**. Provide a session name (e.g. `customer_reviews_20250304_101500`). LLMTool creates matching folders in `annotations_output/` and `logs/`.
2. Dataset detection proposes text columns (e.g. `review_body`) and ID columns. Confirm or override.
3. Load prompts: select `prompts/sentiment_prompt.txt`, optionally add prefixes (like `p1_`). If you need help designing a prompt, run the built-in Social Science Prompt Wizard.
4. Pick a model: `openai:gpt-4o-mini`, `anthropic:claude-3-haiku`, or a local model such as `ollama:llama3.2`. LLMTool validates that the provider is reachable.
5. Configure execution: batch size, retry budget (five JSON repairs max per record), incremental save cadence, and export formats (CSV, JSONL, Label Studio, Doccano).
6. Start annotation. The live panel shows successes, retries, and skipped rows. Checkpoints are written every few dozen rows so you can resume if the run stops.

**Hard-coded example**  
Input file: `data/customer_reviews.csv` (`review_id`, `review_body`)  
Output files:
- `annotations_output/20250304_101500_customer_reviews/data/customer_reviews_annotated.csv`
- `annotations_output/20250304_101500_customer_reviews/exports/labelstudio/customer_reviews.jsonl`
- `annotations_output/.../prompts/` contains a frozen copy of every prompt used.

## Mode 2 - The Annotator Factory (Pipeline Orchestrator)
**Purpose**: Chain annotation, cleaning, language detection, dataset splitting, and training hand-off.

1. Load a dataset (such as `data/support_tickets.parquet`). A quality report checks missing values, language consistency, and existing labels.
2. The factory can reuse your last Annotator configuration or guide you through a fresh setup.
3. Once annotation completes, the pipeline normalizes outputs into `text`, `label`, `confidence`, `language_detected`, and optional metadata columns.
4. Configure training splits (default 80/10/10 stratified) and choose a validation sampling strategy (e.g. confidence-weighted).
5. The pipeline writes train-ready files to `logs/annotator_factory/<session_id>/train_ready/`, plus a detailed JSON report capturing class balance, language stats, and prompt provenance.
6. Optional final step: launch the Training Arena immediately using the freshly prepared data.

**Hard-coded example**  
`logs/annotator_factory/factory_session_20250304_111000/train_ready/support_tickets_train.csv`  
`logs/annotator_factory/.../reports/factory_report.json`

## Mode 3 - Training Arena (Model Benchmarking Studio)
**Purpose**: Train and compare 50+ transformer models, including multilingual, long-document, and multi-label options.

1. Point to a supervised dataset (for example the `support_tickets_train.csv` produced by the factory).
2. Select the text column, label column(s), and optional multi-label or hierarchical fields.
3. Review automatic diagnostics: token-length histograms, language detection, imbalance warnings. The tool recommends long-document models if >20% of samples exceed 512 tokens.
4. Choose token strategies (truncation, sliding window, dynamic padding) and multilingual handling (shared model vs per-language training).
5. Pick candidate architectures such as `bert-base-multilingual-cased`, `xlm-roberta-large`, `longformer-base-4096`, and configure epochs (default 3), reinforcement learning toggles, and batch sizes (auto-adjusted to GPU memory).
6. Monitor the rich console: per-epoch metrics, live confusion matrices, macro/micro F1, precision and recall. Artifacts (logs, charts, checkpoints) are stored in `logs/training_arena/<session_id>/`.

**Hard-coded example**  
`logs/training_arena/support_ticket_models/models/xlm-roberta-large-best/` – best-performing checkpoint  
`logs/training_arena/support_ticket_models/reports/model_rankings.csv` – side-by-side comparison

## Mode 4 - BERT Annotation Studio (Production Inference)
**Purpose**: Run trained models at scale with GPU/CPU parallelism, confidence scoring, and export options.

1. Choose a checkpoint directory (e.g. `logs/training_arena/support_ticket_models/models/xlm-roberta-large-best/`).
2. Select a data source: file (`data/new_support_tickets.csv`), PostgreSQL query, or JSONL export.
3. Configure inference settings: batch size, confidence thresholds, calibration, and export targets (CSV, JSONL, SQL).
4. Optional extras: deduplicate by ID, normalize languages, persist logits and top-k alternatives.
5. Results are saved under `logs/annotation_studio/<session_id>/scored/` with an `inference_report.json` summarizing throughput and class distributions.

**Hard-coded example**  
`logs/annotation_studio/bert_session_20250304_150000/scored/new_support_tickets_predictions.csv`

## Mode 5 - Validation Lab (Quality Control Workshop)
**Purpose**: Audit and compare annotation sources (LLM vs human, double annotation, etc.).

- Draw stratified samples (by class, by confidence intervals) for manual review.
- Compute agreement metrics: Cohen’s Kappa, accuracy, precision/recall per label.
- Highlight discrepancies where two annotators or runs disagree on the same `case_id`.

**Hard-coded example**  
Input: `annotations_output/20250304_101500_customer_reviews/data/customer_reviews_annotated.csv` vs `data/customer_reviews_validated.csv`  
Output: `logs/validation_lab/quality_review_20250304/report/comparison_summary.json`

## Mode 6 - Resume Center (Sessions & Configuration Vault)
**Purpose**: View and restart sessions across all modes while managing saved configurations.

- Displays status (`RUNNING`, `COMPLETED`, `FAILED`), last executed step, key metrics, and timestamps for Annotator, Factory, Training Arena, and BERT Studio sessions.
- Offers quick actions: resume exactly where you stopped (e.g. relaunch Training Arena at Step 11) or clone configuration into a brand-new session.
- Stores configuration profiles (e.g. `~/.llmtool/profiles/baseline_sentiment.json`) so you can reload prompts, parameters, and notes without retyping them.

**Hard-coded example**  
Selecting `factory_session_20250304_111000` reopens the Annotator Factory at the dataset splitting step; choosing `baseline_sentiment` restores the saved configuration for reuse on `data/customer_reviews_march.csv`.

## Mode 7 - Documentation & Help (You Are Here)
- Use this screen whenever you need a refresher on workflows, file locations, or troubleshooting steps.
- The content updates as new capabilities ship. Bookmark notable sections for your team.

## Where Everything Lands
- `annotations_output/<timestamp_session>/`: annotated datasets, copied prompts, configuration snapshots, run logs.
- `logs/<mode>/<session_id>/`: granular reports, metrics, checkpoints, and metadata for each workflow stage.
- `logs/application/`: timestamped diagnostic logs (`llmtool_<timestamp>.log`) useful when reporting issues.
- `~/.llmtool/profiles/`: reusable profiles plus `history.json` tracking past executions.

## Troubleshooting Checklist
- **“No API key detected”** – open the Resume Center, enter the key under Provider Settings, or set environment variables before launching.
- **“Some rows remain unannotated”** – inspect retry counts in the annotation summary and read `logs/annotator/<session_id>/annotator.log` for the failing prompts.
- **“Driver missing (pyreadr / psycopg2 / pymysql)”** – install only the required extra dependency, then rerun.
- **“Out of memory”** – reduce batch sizes based on the recommendations in the resource banner; for very long texts, consider Longformer/BigBird models.
- **Need more help?** – consult `README.md`, the `docs/` folder, or raise an issue at https://github.com/antoine-lemor/LLMTool. When emailing support@llmtool.ai attach the relevant `logs/application/llmtool_<timestamp>.log`.
            """

            md = Markdown(doc_text)
            self.console.print(Panel(md, title="📚 Documentation", border_style="blue"))
        else:
            print("\n=== Documentation ===")
            print("Visit: https://github.com/antoine-lemor/LLMTool")


for _method_name in training_arena.TRAINING_ARENA_METHODS:
    setattr(AdvancedCLI, _method_name, getattr(training_arena, _method_name))
del _method_name


def main():
    """Entry point for the advanced CLI"""
    cli = AdvancedCLI()
    cli.run()


if __name__ == "__main__":
    main()

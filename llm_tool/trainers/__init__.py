"""
PROJECT:
-------
LLMTool

TITLE:
------
__init__.py

MAIN OBJECTIVE:
---------------
This script initializes the LLMTool package, exposing all
public classes and functions for easy import.

Dependencies:
-------------
- All internal modules of the package

MAIN FEATURES:
--------------
1) Imports and exposes all model classes (BERT variants, SOTA models, multilingual models)
2) Exposes utility functions for model selection and data processing
3) Provides training orchestration tools (MultiLabelTrainer, BenchmarkRunner)
4) Includes CLI interface for interactive training

Author:
-------
Antoine Lemor
"""

from .bert_abc import BertABC
from .bert_base import BertBase

# Original models
from .models import Bert
from .models import ArabicBert
from .models import Camembert
from .models import ChineseBert
from .models import GermanBert
from .models import HindiBert
from .models import ItalianBert
from .models import PortugueseBert
from .models import RussianBert
from .models import SpanishBert
from .models import SwedishBert
from .models import XLMRoberta

# SOTA models
from .sota_models import (
    DeBERTaV3Base, DeBERTaV3Large, DeBERTaV3XSmall,
    RoBERTaBase, RoBERTaLarge, DistilRoBERTa,
    ELECTRABase, ELECTRALarge, ELECTRASmall,
    ALBERTBase, ALBERTLarge, ALBERTXLarge,
    BigBirdBase, BigBirdLarge,
    LongformerBase, LongformerLarge,
    MDeBERTaV3Base, XLMRobertaBase, XLMRobertaLarge,
    # French SOTA models
    CamembertaV2Base, CamembertLarge, FlauBERTBase, FlauBERTLarge,
    BARThez, FrALBERT, DistilCamemBERT,
    FrELECTRA, CamembertBioBERT
)

# Model selection utilities
from .model_selector import (
    ModelSelector,
    TaskComplexity,
    ResourceProfile,
    auto_select_model
)

# Multilingual utilities
from .multilingual_selector import (
    MultilingualModelSelector,
    ModelSize,
    TaskType,
    create_multilingual_ensemble
)

# Multi-label training
from .multi_label_trainer import (
    MultiLabelTrainer,
    TrainingConfig,
    MultiLabelSample,
    train_multi_label_models
)

# Enhanced base with metadata support
from .bert_base_enhanced import BertBaseEnhanced

# Data utilities
from .data_utils import (
    DataSample,
    DataLoader as DataUtil,
    PerformanceTracker
)

# Data splitting with stratification
from .data_splitter import (
    DataSplitter,
    SplitConfig,
    create_stratified_splits
)

# Parallel inference
from .parallel_inference import parallel_predict
from .benchmarking import BenchmarkRunner, BenchmarkConfig, TrainingRunSummary
from .benchmark_dataset_builder import BenchmarkDatasetBuilder, BenchmarkDataset

# Command Line Interface
from .cli import TrainingCLI

__version__ = '3.1.0'

__all__ = [
    # Base classes
    'BertABC', 'BertBase',

    # Original models
    'Bert', 'ArabicBert', 'Camembert', 'ChineseBert', 'GermanBert',
    'HindiBert', 'ItalianBert', 'PortugueseBert', 'RussianBert',
    'SpanishBert', 'SwedishBert', 'XLMRoberta',

    # SOTA models - English
    'DeBERTaV3Base', 'DeBERTaV3Large', 'DeBERTaV3XSmall',
    'RoBERTaBase', 'RoBERTaLarge', 'DistilRoBERTa',
    'ELECTRABase', 'ELECTRALarge', 'ELECTRASmall',
    'ALBERTBase', 'ALBERTLarge', 'ALBERTXLarge',
    'BigBirdBase', 'BigBirdLarge',
    'LongformerBase', 'LongformerLarge',

    # SOTA models - Multilingual
    'MDeBERTaV3Base', 'XLMRobertaBase', 'XLMRobertaLarge',

    # SOTA models - French
    'CamembertaV2Base', 'CamembertLarge', 'FlauBERTBase', 'FlauBERTLarge',
    'BARThez', 'FrALBERT', 'DistilCamemBERT',
    'FrELECTRA', 'CamembertBioBERT',

    # Utilities
    'ModelSelector', 'TaskComplexity', 'ResourceProfile', 'auto_select_model',
    'MultilingualModelSelector', 'ModelSize', 'TaskType',
    'create_multilingual_ensemble', 'parallel_predict',

    # Multi-label training
    'MultiLabelTrainer', 'TrainingConfig', 'MultiLabelSample', 'train_multi_label_models',

    # Enhanced features
    'BertBaseEnhanced', 'DataSample', 'DataUtil', 'PerformanceTracker',

    # Data splitting
    'DataSplitter', 'SplitConfig', 'create_stratified_splits',

    # Benchmarking orchestrator
    'BenchmarkRunner', 'BenchmarkConfig', 'TrainingRunSummary',
    'BenchmarkDatasetBuilder', 'BenchmarkDataset',

    # Command Line Interface
    'TrainingCLI'
]

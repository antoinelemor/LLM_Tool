#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
__init__.py

MAIN OBJECTIVE:
---------------
Expose the full training toolkit—base classes, language models, SOTA backbones,
selection utilities, and trainers—through a single import location.

Dependencies:
-------------
- llm_tool.trainers.bert_abc
- llm_tool.trainers.bert_base
- llm_tool.trainers.models
- llm_tool.trainers.sota_models
- llm_tool.trainers.model_selector
- llm_tool.trainers.multilingual_selector
- llm_tool.trainers.multi_label_trainer
- llm_tool.trainers.benchmarking
- llm_tool.trainers.parallel_inference

MAIN FEATURES:
--------------
1) Re-export BertABC, BertBase, and language-specific model wrappers
2) Surface SOTA transformer classes for immediate use in training flows
3) Provide helper utilities for automatic model and language selection
4) Include multi-label, benchmarking, and dataset builder entry points
5) Keep backward compatibility by centralising Trainer namespace access

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

# BertBase now includes enhanced features with metadata support
# (previously BertBaseEnhanced, now merged into BertBase)

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
from .training_data_builder import TrainingDatasetBuilder, TrainingDataRequest, TrainingDataBundle

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
    'TrainingDatasetBuilder', 'TrainingDataRequest', 'TrainingDataBundle',

    # Command Line Interface
    'TrainingCLI'
]

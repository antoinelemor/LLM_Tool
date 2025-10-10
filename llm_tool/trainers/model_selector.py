#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
model_selector.py

MAIN OBJECTIVE:
---------------
Recommend transformer backbones suited to dataset difficulty, language mix,
and hardware constraints, with optional benchmarking for validation.

Dependencies:
-------------
- typing
- dataclasses
- enum
- json
- os
- shutil
- time
- numpy
- torch
- logging
- llm_tool.trainers.bert_base
- llm_tool.trainers.sota_models
- llm_tool.trainers.models
- llm_tool.utils.training_paths

MAIN FEATURES:
--------------
1) Maintain rich model profiles describing capacity, memory, and accuracy
2) Map task complexity and resource profiles to candidate model families
3) Provide automatic selection heuristics with override hooks for users
4) Run optional benchmarking loops to compare shortlisted models empirically
5) Return structured recommendations with rationale and deployment guidance

Author:
-------
Antoine Lemor
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import os
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from llm_tool.trainers.bert_base import BertBase
from llm_tool.trainers.sota_models import (
    DeBERTaV3Base, DeBERTaV3Large, DeBERTaV3XSmall,
    RoBERTaBase, RoBERTaLarge, DistilRoBERTa,
    ELECTRABase, ELECTRALarge, ELECTRASmall,
    ALBERTBase, ALBERTLarge, ALBERTXLarge,
    BigBirdBase, BigBirdLarge,
    LongformerBase, LongformerLarge,
    MDeBERTaV3Base, XLMRobertaBase, XLMRobertaLarge,
    # French models
    CamembertaV2Base, CamembertLarge, FlauBERTBase, FlauBERTLarge,
    BARThez, FrALBERT, DistilCamemBERT,
    FrELECTRA, CamembertBioBERT
)
from llm_tool.trainers.models import (
    Bert, Camembert, GermanBert, SpanishBert, ItalianBert,
    PortugueseBert, ChineseBert, ArabicBert, RussianBert, HindiBert
)
from llm_tool.utils.training_paths import get_training_logs_base


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"      # Binary classification, simple patterns
    MODERATE = "moderate"  # Multi-class, moderate complexity
    COMPLEX = "complex"    # Fine-grained distinctions, nuanced understanding
    EXTREME = "extreme"    # Highly complex, requires maximum capacity


class ResourceProfile(Enum):
    """Resource availability profiles."""
    MINIMAL = "minimal"    # CPU only, <4GB RAM
    LIMITED = "limited"    # Basic GPU, 4-8GB RAM
    STANDARD = "standard"  # Good GPU, 8-16GB RAM
    PREMIUM = "premium"    # High-end GPU, 16GB+ RAM


@dataclass
class ModelProfile:
    """Complete profile of a model's characteristics."""
    name: str
    model_class: type
    parameters: int  # millions
    memory_footprint: float  # GB
    inference_speed: float  # relative, 1.0 = baseline
    accuracy_score: float  # average across benchmarks
    complexity_handling: str  # simple/moderate/complex/extreme
    special_features: List[str]
    recommended_tasks: List[str]
    supported_languages: List[str]  # List of language codes this model supports


@dataclass
class BenchmarkResult:
    """Results from model benchmarking."""
    model_name: str
    accuracy: float
    f1_score: float
    inference_time: float  # seconds per sample
    memory_usage: float  # GB
    training_time: float  # seconds per epoch
    convergence_epochs: int


class ModelSelector:
    """
    Advanced model selection system with comprehensive evaluation capabilities.
    """

    # Comprehensive model profiles
    MODEL_PROFILES = {
        'DeBERTaV3XSmall': ModelProfile(
            name='microsoft/deberta-v3-xsmall',
            model_class=DeBERTaV3XSmall,
            parameters=22,
            memory_footprint=0.3,
            inference_speed=2.0,
            accuracy_score=0.82,
            complexity_handling='simple',
            special_features=['extremely efficient', 'mobile-friendly'],
            recommended_tasks=['sentiment', 'spam detection', 'simple classification'],
            supported_languages=['en']  # English-focused model
        ),
        'DistilRoBERTa': ModelProfile(
            name='distilroberta-base',
            model_class=DistilRoBERTa,
            parameters=82,
            memory_footprint=0.5,
            inference_speed=1.8,
            accuracy_score=0.85,
            complexity_handling='moderate',
            special_features=['distilled', 'fast inference'],
            recommended_tasks=['general classification', 'production systems'],
            supported_languages=['en']  # English-focused model
        ),
        'ELECTRASmall': ModelProfile(
            name='google/electra-small-discriminator',
            model_class=ELECTRASmall,
            parameters=14,
            memory_footprint=0.2,
            inference_speed=2.5,
            accuracy_score=0.81,
            complexity_handling='simple',
            special_features=['ultra-efficient', 'good accuracy/speed ratio'],
            recommended_tasks=['edge deployment', 'real-time systems'],
            supported_languages=['en']  # English model
        ),
        'ALBERTBase': ModelProfile(
            name='albert-base-v2',
            model_class=ALBERTBase,
            parameters=12,
            memory_footprint=0.4,
            inference_speed=1.5,
            accuracy_score=0.86,
            complexity_handling='moderate',
            special_features=['parameter sharing', 'memory efficient'],
            recommended_tasks=['general NLP', 'resource-constrained environments'],
            supported_languages=['en']  # English model
        ),
        'RoBERTaBase': ModelProfile(
            name='roberta-base',
            model_class=RoBERTaBase,
            parameters=125,
            memory_footprint=0.8,
            inference_speed=1.0,
            accuracy_score=0.88,
            complexity_handling='moderate',
            special_features=['robust training', 'versatile'],
            recommended_tasks=['general classification', 'named entity recognition'],
            supported_languages=['en']  # English model
        ),
        'ELECTRABase': ModelProfile(
            name='google/electra-base-discriminator',
            model_class=ELECTRABase,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.1,
            accuracy_score=0.89,
            complexity_handling='moderate',
            special_features=['discriminative pretraining', 'efficient'],
            recommended_tasks=['text classification', 'question answering'],
            supported_languages=['en']  # English model
        ),
        'DeBERTaV3Base': ModelProfile(
            name='microsoft/deberta-v3-base',
            model_class=DeBERTaV3Base,
            parameters=184,
            memory_footprint=1.2,
            inference_speed=0.85,
            accuracy_score=0.91,
            complexity_handling='complex',
            special_features=['disentangled attention', 'SOTA performance'],
            recommended_tasks=['complex classification', 'nuanced understanding'],
            supported_languages=['en']  # English model
        ),
        'BigBirdBase': ModelProfile(
            name='google/bigbird-roberta-base',
            model_class=BigBirdBase,
            parameters=128,
            memory_footprint=1.5,
            inference_speed=0.7,
            accuracy_score=0.87,
            complexity_handling='complex',
            special_features=['4096 token context', 'sparse attention'],
            recommended_tasks=['document classification', 'long text analysis'],
            supported_languages=['en']  # English model
        ),
        'LongformerBase': ModelProfile(
            name='allenai/longformer-base-4096',
            model_class=LongformerBase,
            parameters=148,
            memory_footprint=1.6,
            inference_speed=0.65,
            accuracy_score=0.88,
            complexity_handling='complex',
            special_features=['4096 token context', 'local+global attention'],
            recommended_tasks=['document understanding', 'long sequences'],
            supported_languages=['en']  # English model
        ),
        'MDeBERTaV3Base': ModelProfile(
            name='microsoft/mdeberta-v3-base',
            model_class=MDeBERTaV3Base,
            parameters=278,
            memory_footprint=1.4,
            inference_speed=0.8,
            accuracy_score=0.92,
            complexity_handling='complex',
            special_features=['multilingual', '100+ languages', 'SOTA multilingual'],
            recommended_tasks=['multilingual classification', 'cross-lingual tasks'],
            supported_languages=['*']  # Supports all languages
        ),
        'XLMRobertaBase': ModelProfile(
            name='xlm-roberta-base',
            model_class=XLMRobertaBase,
            parameters=278,
            memory_footprint=1.3,
            inference_speed=0.9,
            accuracy_score=0.90,
            complexity_handling='complex',
            special_features=['multilingual', '100+ languages', 'robust'],
            recommended_tasks=['multilingual NLP', 'language detection'],
            supported_languages=['*']  # Supports all languages
        ),
        'XLMRobertaLarge': ModelProfile(
            name='xlm-roberta-large',
            model_class=XLMRobertaLarge,
            parameters=560,
            memory_footprint=2.5,
            inference_speed=0.5,
            accuracy_score=0.93,
            complexity_handling='complex',
            special_features=['multilingual', 'high capacity', 'best multilingual performance'],
            recommended_tasks=['complex multilingual tasks', 'fine-grained classification'],
            supported_languages=['*']  # Supports all languages
        ),
        'ALBERTLarge': ModelProfile(
            name='albert-large-v2',
            model_class=ALBERTLarge,
            parameters=18,
            memory_footprint=0.6,
            inference_speed=0.9,
            accuracy_score=0.89,
            complexity_handling='complex',
            special_features=['parameter efficient large model'],
            recommended_tasks=['complex tasks with memory constraints'],
            supported_languages=['en']  # English model
        ),
        'RoBERTaLarge': ModelProfile(
            name='roberta-large',
            model_class=RoBERTaLarge,
            parameters=355,
            memory_footprint=2.2,
            inference_speed=0.5,
            accuracy_score=0.92,
            complexity_handling='complex',
            special_features=['high capacity', 'superior performance'],
            recommended_tasks=['challenging classification', 'research tasks'],
            supported_languages=['en']  # English model
        ),
        'ELECTRALarge': ModelProfile(
            name='google/electra-large-discriminator',
            model_class=ELECTRALarge,
            parameters=335,
            memory_footprint=2.0,
            inference_speed=0.55,
            accuracy_score=0.93,
            complexity_handling='extreme',
            special_features=['best accuracy/compute ratio', 'discriminative'],
            recommended_tasks=['competition tasks', 'maximum accuracy'],
            supported_languages=['en']  # English model
        ),
        'DeBERTaV3Large': ModelProfile(
            name='microsoft/deberta-v3-large',
            model_class=DeBERTaV3Large,
            parameters=435,
            memory_footprint=2.8,
            inference_speed=0.4,
            accuracy_score=0.95,
            complexity_handling='extreme',
            special_features=['absolute SOTA', 'disentangled attention'],
            recommended_tasks=['most challenging tasks', 'research frontiers'],
            supported_languages=['en']  # English model
        ),
        'ALBERTXLarge': ModelProfile(
            name='albert-xlarge-v2',
            model_class=ALBERTXLarge,
            parameters=60,
            memory_footprint=1.2,
            inference_speed=0.6,
            accuracy_score=0.91,
            complexity_handling='extreme',
            special_features=['parameter sharing at scale'],
            recommended_tasks=['complex tasks', 'memory-efficient large model'],
            supported_languages=['en']  # English model
        ),
        'BigBirdLarge': ModelProfile(
            name='google/bigbird-roberta-large',
            model_class=BigBirdLarge,
            parameters=356,
            memory_footprint=3.0,
            inference_speed=0.3,
            accuracy_score=0.92,
            complexity_handling='extreme',
            special_features=['4096 tokens', 'document-level understanding'],
            recommended_tasks=['long document analysis', 'comprehensive understanding'],
            supported_languages=['en']  # English model
        ),
        'LongformerLarge': ModelProfile(
            name='allenai/longformer-large-4096',
            model_class=LongformerLarge,
            parameters=435,
            memory_footprint=3.2,
            inference_speed=0.25,
            accuracy_score=0.93,
            complexity_handling='extreme',
            special_features=['maximum context window', 'attention patterns'],
            recommended_tasks=['book-length texts', 'comprehensive document analysis'],
            supported_languages=['en']  # English model
        ),
        # English BERT model
        'Bert': ModelProfile(
            name='bert-base-cased',
            model_class=Bert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['original BERT', 'cased'],
            recommended_tasks=['general NLP', 'text classification'],
            supported_languages=['en']  # English model
        ),
        # French models - Base versions
        'Camembert': ModelProfile(
            name='camembert-base',
            model_class=Camembert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.88,
            complexity_handling='moderate',
            special_features=['French language', 'RoBERTa-based'],
            recommended_tasks=['French text classification', 'French NER'],
            supported_languages=['fr']  # French model
        ),
        # French models - SOTA equivalents to English models
        'CamembertaV2Base': ModelProfile(
            name='almanach/camemberta-base',
            model_class=CamembertaV2Base,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.90,
            complexity_handling='moderate',
            special_features=['CamemBERTa-v2', 'Modern French RoBERTa', 'DeBERTa-v3 training'],
            recommended_tasks=['French NLP', 'French text classification', 'French NER'],
            supported_languages=['fr']  # French equivalent to RoBERTa Base
        ),
        'CamembertLarge': ModelProfile(
            name='camembert/camembert-large',
            model_class=CamembertLarge,
            parameters=335,
            memory_footprint=2.2,
            inference_speed=0.5,
            accuracy_score=0.91,
            complexity_handling='complex',
            special_features=['CamemBERT Large', 'high capacity', 'legacy'],
            recommended_tasks=['complex French NLP', 'fine-grained classification'],
            supported_languages=['fr']  # Legacy French large model
        ),
        'FlauBERTBase': ModelProfile(
            name='flaubert/flaubert_base_cased',
            model_class=FlauBERTBase,
            parameters=137,
            memory_footprint=0.9,
            inference_speed=0.95,
            accuracy_score=0.89,
            complexity_handling='moderate',
            special_features=['French BERT', 'cased', 'better than CamemBERT base'],
            recommended_tasks=['French classification', 'French sentiment analysis'],
            supported_languages=['fr']  # French model
        ),
        'FlauBERTLarge': ModelProfile(
            name='flaubert/flaubert_large_cased',
            model_class=FlauBERTLarge,
            parameters=372,
            memory_footprint=2.4,
            inference_speed=0.45,
            accuracy_score=0.93,
            complexity_handling='complex',
            special_features=['Large French BERT', 'SOTA French performance'],
            recommended_tasks=['complex French tasks', 'French document understanding'],
            supported_languages=['fr']  # French equivalent to BERT Large
        ),
        'DistilCamemBERT': ModelProfile(
            name='cmarkea/distilcamembert-base',
            model_class=DistilCamemBERT,
            parameters=68,
            memory_footprint=0.4,
            inference_speed=1.8,
            accuracy_score=0.85,
            complexity_handling='moderate',
            special_features=['distilled', 'fast French inference', '50% smaller'],
            recommended_tasks=['French production systems', 'real-time French NLP'],
            supported_languages=['fr']  # French equivalent to DistilBERT/DistilRoBERTa
        ),
        'FrALBERT': ModelProfile(
            name='cls/fralbert',
            model_class=FrALBERT,
            parameters=12,
            memory_footprint=0.4,
            inference_speed=1.5,
            accuracy_score=0.86,
            complexity_handling='moderate',
            special_features=['French ALBERT', 'parameter sharing', 'memory efficient'],
            recommended_tasks=['French NLP with memory constraints', 'mobile French apps'],
            supported_languages=['fr']  # French equivalent to ALBERT
        ),
        'FrELECTRA': ModelProfile(
            name='dbmdz/electra-base-french-europeana-cased-discriminator',
            model_class=FrELECTRA,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.1,
            accuracy_score=0.88,
            complexity_handling='moderate',
            special_features=['French ELECTRA', 'discriminative pretraining', 'efficient'],
            recommended_tasks=['French text classification', 'French QA'],
            supported_languages=['fr']  # French equivalent to ELECTRA Base
        ),
        'BARThez': ModelProfile(
            name='moussaKam/barthez',
            model_class=BARThez,
            parameters=165,
            memory_footprint=1.1,
            inference_speed=0.85,
            accuracy_score=0.90,
            complexity_handling='complex',
            special_features=['French BART', 'seq2seq capable', 'generation tasks'],
            recommended_tasks=['French summarization', 'French text generation'],
            supported_languages=['fr']  # French BART model
        ),
        'CamembertBioBERT': ModelProfile(
            name='almanach/camembert-bio-base',
            model_class=CamembertBioBERT,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['French biomedical', 'medical domain', 'specialized'],
            recommended_tasks=['French medical NLP', 'French clinical text'],
            supported_languages=['fr']  # French biomedical model
        ),
        # German model
        'GermanBert': ModelProfile(
            name='dbmdz/bert-base-german-cased',
            model_class=GermanBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['German language', 'cased'],
            recommended_tasks=['German text classification', 'German NER'],
            supported_languages=['de']  # German model
        ),
        # Spanish model
        'SpanishBert': ModelProfile(
            name='dccuchile/bert-base-spanish-wwm-cased',
            model_class=SpanishBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['Spanish language', 'whole word masking'],
            recommended_tasks=['Spanish text classification', 'Spanish NER'],
            supported_languages=['es']  # Spanish model
        ),
        # Italian model
        'ItalianBert': ModelProfile(
            name='dbmdz/bert-base-italian-cased',
            model_class=ItalianBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.86,
            complexity_handling='moderate',
            special_features=['Italian language', 'cased'],
            recommended_tasks=['Italian text classification', 'Italian NER'],
            supported_languages=['it']  # Italian model
        ),
        # Portuguese model
        'PortugueseBert': ModelProfile(
            name='neuralmind/bert-base-portuguese-cased',
            model_class=PortugueseBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['Portuguese language', 'Brazilian Portuguese'],
            recommended_tasks=['Portuguese text classification', 'Portuguese NER'],
            supported_languages=['pt']  # Portuguese model
        ),
        # Chinese model
        'ChineseBert': ModelProfile(
            name='bert-base-chinese',
            model_class=ChineseBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['Chinese language', 'character-based'],
            recommended_tasks=['Chinese text classification', 'Chinese NER'],
            supported_languages=['zh']  # Chinese model
        ),
        # Arabic model
        'ArabicBert': ModelProfile(
            name='asafaya/bert-base-arabic',
            model_class=ArabicBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.86,
            complexity_handling='moderate',
            special_features=['Arabic language', 'Arabic-specific'],
            recommended_tasks=['Arabic text classification', 'Arabic NER'],
            supported_languages=['ar']  # Arabic model
        ),
        # Russian model
        'RussianBert': ModelProfile(
            name='DeepPavlov/rubert-base-cased',
            model_class=RussianBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.87,
            complexity_handling='moderate',
            special_features=['Russian language', 'cased'],
            recommended_tasks=['Russian text classification', 'Russian NER'],
            supported_languages=['ru']  # Russian model
        ),
        # Hindi model
        'HindiBert': ModelProfile(
            name='monsoon-nlp/hindi-bert',
            model_class=HindiBert,
            parameters=110,
            memory_footprint=0.7,
            inference_speed=1.0,
            accuracy_score=0.85,
            complexity_handling='moderate',
            special_features=['Hindi language', 'Devanagari script'],
            recommended_tasks=['Hindi text classification', 'Hindi NER'],
            supported_languages=['hi']  # Hindi model
        )
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize the model selector.

        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        self.benchmark_cache = {}

    def recommend(self,
                  task_complexity: TaskComplexity = TaskComplexity.MODERATE,
                  resource_profile: ResourceProfile = ResourceProfile.STANDARD,
                  max_sequence_length: int = 512,
                  required_accuracy: float = 0.85,
                  max_inference_time: Optional[float] = None,
                  special_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Recommend best model based on requirements.

        Args:
            task_complexity: Complexity level of the task
            resource_profile: Available computational resources
            max_sequence_length: Maximum input sequence length
            required_accuracy: Minimum required accuracy
            max_inference_time: Maximum acceptable inference time per sample
            special_requirements: List of special requirements

        Returns:
            Dictionary with recommendation details
        """
        candidates = []

        # Filter by basic requirements
        for name, profile in self.MODEL_PROFILES.items():
            # Check complexity handling
            if not self._can_handle_complexity(profile.complexity_handling, task_complexity):
                continue

            # Check resource constraints
            if not self._fits_resource_profile(profile, resource_profile):
                continue

            # Check accuracy requirement
            if profile.accuracy_score < required_accuracy:
                continue

            # Check inference time if specified
            if max_inference_time and (1.0 / profile.inference_speed) > max_inference_time:
                continue

            # Check special requirements
            if special_requirements:
                if max_sequence_length > 512 and '4096' not in str(profile.special_features):
                    continue

            candidates.append((name, profile))

        if not candidates:
            # Relax constraints and try again
            return self._recommend_with_relaxed_constraints(
                task_complexity, resource_profile, required_accuracy
            )

        # Score and rank candidates
        scored_candidates = []
        for name, profile in candidates:
            score = self._calculate_model_score(
                profile, task_complexity, resource_profile, required_accuracy
            )
            scored_candidates.append((score, name, profile))

        scored_candidates.sort(reverse=True)
        best_score, best_name, best_profile = scored_candidates[0]

        # Prepare detailed recommendation
        recommendation = {
            'model_name': best_name,
            'model_class': best_profile.model_class,
            'huggingface_name': best_profile.name,
            'score': best_score,
            'profile': best_profile,
            'reasoning': self._generate_reasoning(best_profile, task_complexity, resource_profile),
            'alternatives': [
                {'name': name, 'score': score, 'profile': prof}
                for score, name, prof in scored_candidates[1:4]
            ],
            'configuration_tips': self._get_configuration_tips(best_profile, task_complexity)
        }

        if self.verbose:
            self._print_recommendation(recommendation)

        return recommendation

    def benchmark_models(self,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         models_to_test: Optional[List[str]] = None,
                         epochs: int = 2,
                         sample_size: Optional[int] = None,
                         language_info: Optional[List[str]] = None,
                         show_detailed_metrics: bool = True,
                         session_id: Optional[str] = None) -> List[BenchmarkResult]:
        """
        Benchmark multiple models on actual data with detailed language-specific metrics.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            models_to_test: List of model names to test (None = test all suitable)
            epochs: Number of training epochs for benchmarking
            sample_size: Optional sample size for quick benchmarking
            language_info: Optional list of language codes for each sample
            show_detailed_metrics: Show detailed metrics including per-language performance

        Returns:
            List of benchmark results with comprehensive metrics
        """
        if models_to_test is None:
            # Select representative models
            models_to_test = ['DeBERTaV3XSmall', 'RoBERTaBase', 'DeBERTaV3Base', 'ELECTRABase']

        results = []

        for model_name in models_to_test:
            if model_name not in self.MODEL_PROFILES:
                continue

            profile = self.MODEL_PROFILES[model_name]
            if self.verbose:
                print(f"\nBenchmarking {model_name}...")

            try:
                # Initialize model
                model = profile.model_class()

                # UNIFIED: Use centralized function to set detected languages (SAME AS BENCHMARK)
                if language_info is not None:
                    from .model_trainer import set_detected_languages_on_model
                    set_detected_languages_on_model(
                        model=model,
                        confirmed_languages=language_info,
                        logger=logger
                    )

                # Time training
                start_time = time.time()
                metrics_base_dir = str(get_training_logs_base())
                scores = model.run_training(
                    train_loader,
                    val_loader,
                    n_epochs=epochs,
                    save_model_as=f"benchmark_{model_name.lower()}",
                    metrics_output_dir=metrics_base_dir,
                    track_languages=language_info is not None,
                    language_info=language_info,
                    label_key=None,  # Model selector benchmark mode
                    label_value='benchmark',  # Generic label for model selection
                    language=None,  # Language not specified in this context
                    session_id=session_id  # CRITICAL: Unified session ID for all runs
                )
                training_time = time.time() - start_time

                summary = scores if isinstance(scores, dict) else getattr(model, "last_training_summary", {})
                avg_f1 = summary.get('macro_f1', 0)
                accuracy = summary.get('accuracy', 0)
                f1_class_0 = summary.get('f1_0', 0)
                f1_class_1 = summary.get('f1_1', 0)
                precision_0 = summary.get('precision_0', 0)
                precision_1 = summary.get('precision_1', 0)
                recall_0 = summary.get('recall_0', 0)
                recall_1 = summary.get('recall_1', 0)
                val_loss = summary.get('val_loss', 0)

                best_model_path_obj = summary.get('best_model_path') or getattr(model, 'last_saved_model_path', None)
                best_model_path = str(best_model_path_obj) if best_model_path_obj else None
                avg_inference_time = self._estimate_inference_time(
                    model,
                    val_loader,
                    best_model_path,
                    sample_size,
                    profile,
                )

                if best_model_path and os.path.isdir(best_model_path):
                    shutil.rmtree(best_model_path, ignore_errors=True)

                # Create enhanced benchmark result
                result = BenchmarkResult(
                    model_name=model_name,
                    accuracy=accuracy,
                    f1_score=avg_f1,
                    inference_time=avg_inference_time,
                    memory_usage=profile.memory_footprint,
                    training_time=training_time / epochs,
                    convergence_epochs=epochs
                )

                # Add detailed metrics as attributes
                result.f1_class_0 = f1_class_0
                result.f1_class_1 = f1_class_1
                result.precision_0 = precision_0
                result.precision_1 = precision_1
                result.recall_0 = recall_0
                result.recall_1 = recall_1
                result.val_loss = val_loss

                # Add language-specific metrics if available
                if summary.get('language_metrics'):
                    result.language_metrics = summary['language_metrics']
                result.training_summary = summary
                results.append(result)

                # Cache result
                self.benchmark_cache[model_name] = result

            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to benchmark {model_name}: {str(e)}")
                continue

        # Sort by F1 score
        results.sort(key=lambda x: x.f1_score, reverse=True)

        if self.verbose or show_detailed_metrics:
            self._print_detailed_benchmark_results(results, language_info)

        return results

    def _estimate_inference_time(
        self,
        model: BertBase,
        val_loader: DataLoader,
        best_model_path: Optional[str],
        sample_size: Optional[int],
        profile: ModelProfile,
    ) -> float:
        """Measure or approximate inference time per sample."""
        if not best_model_path:
            return profile.memory_footprint * 0.001

        try:
            hf_model = model.load_model(best_model_path)
            hf_model.to(model.device)
            hf_model.eval()

            total_samples = 0
            start = time.time()
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(model.device)
                    masks = batch[1].to(model.device)
                    hf_model(inputs, attention_mask=masks)
                    total_samples += inputs.size(0)
                    if sample_size and total_samples >= sample_size:
                        break

            if total_samples:
                return (time.time() - start) / total_samples
        except Exception as exc:  # pragma: no cover - depends on transformer backend
            if self.verbose and hasattr(self, 'logger'):
                self.logger.warning("Inference timing failed: %s", exc)

        return profile.memory_footprint * 0.001

    def _can_handle_complexity(self, model_complexity: str, required: TaskComplexity) -> bool:
        """Check if model can handle required complexity."""
        complexity_levels = {
            'simple': 1,
            'moderate': 2,
            'complex': 3,
            'extreme': 4
        }
        return complexity_levels.get(model_complexity, 0) >= complexity_levels.get(required.value, 2)

    def _fits_resource_profile(self, profile: ModelProfile, resources: ResourceProfile) -> bool:
        """Check if model fits within resource constraints."""
        resource_limits = {
            ResourceProfile.MINIMAL: {'memory': 0.5, 'speed': 1.5},
            ResourceProfile.LIMITED: {'memory': 1.0, 'speed': 1.0},
            ResourceProfile.STANDARD: {'memory': 2.0, 'speed': 0.5},
            ResourceProfile.PREMIUM: {'memory': 5.0, 'speed': 0.1}
        }
        limits = resource_limits[resources]
        return (profile.memory_footprint <= limits['memory'] and
                (1.0 / profile.inference_speed) <= limits['speed'])

    def _calculate_model_score(self,
                              profile: ModelProfile,
                              complexity: TaskComplexity,
                              resources: ResourceProfile,
                              required_accuracy: float) -> float:
        """Calculate comprehensive model score."""
        # Base score from accuracy
        score = profile.accuracy_score * 100

        # Complexity match bonus
        if profile.complexity_handling == complexity.value:
            score += 10
        elif self._can_handle_complexity(profile.complexity_handling, complexity):
            score += 5

        # Efficiency bonus for over-provisioned models
        if resources in [ResourceProfile.LIMITED, ResourceProfile.MINIMAL]:
            score += (2.0 - profile.memory_footprint) * 5
            score += profile.inference_speed * 5

        # Accuracy margin bonus
        accuracy_margin = profile.accuracy_score - required_accuracy
        score += min(accuracy_margin * 50, 10)

        return score

    def _generate_reasoning(self,
                           profile: ModelProfile,
                           complexity: TaskComplexity,
                           resources: ResourceProfile) -> str:
        """Generate human-readable reasoning for selection."""
        reasons = []

        if profile.accuracy_score >= 0.9:
            reasons.append("exceptional accuracy")
        if profile.inference_speed >= 1.5:
            reasons.append("fast inference")
        if profile.memory_footprint <= 0.5:
            reasons.append("memory efficient")
        if '4096' in str(profile.special_features):
            reasons.append("handles long sequences")
        if profile.complexity_handling == complexity.value:
            reasons.append(f"optimized for {complexity.value} tasks")

        return f"Selected for {', '.join(reasons) if reasons else 'best overall fit'}"

    def _get_configuration_tips(self, profile: ModelProfile, complexity: TaskComplexity) -> List[str]:
        """Get model-specific configuration tips."""
        tips = []

        # General tips
        if profile.memory_footprint > 2.0:
            tips.append("Use gradient checkpointing to reduce memory usage")
        if profile.inference_speed < 0.5:
            tips.append("Consider batch processing for better throughput")

        # Model-specific tips
        if 'albert' in profile.name.lower():
            tips.append("ALBERT benefits from longer training due to parameter sharing")
        if 'deberta' in profile.name.lower():
            tips.append("DeBERTa works best with position-aware attention masks")
        if 'electra' in profile.name.lower():
            tips.append("ELECTRA often converges faster than BERT-based models")
        if any(x in profile.name.lower() for x in ['longformer', 'bigbird']):
            tips.append("Use sliding window attention for optimal performance on long texts")

        # Task complexity tips
        if complexity == TaskComplexity.SIMPLE:
            tips.append("Consider reducing model layers for faster inference")
        elif complexity == TaskComplexity.EXTREME:
            tips.append("Use learning rate warmup and careful hyperparameter tuning")

        return tips

    def _recommend_with_relaxed_constraints(self,
                                           complexity: TaskComplexity,
                                           resources: ResourceProfile,
                                           required_accuracy: float) -> Dict[str, Any]:
        """Provide recommendation with relaxed constraints."""
        # Try with reduced accuracy requirement
        relaxed_accuracy = max(0.8, required_accuracy - 0.05)

        # Find best available model
        best_model = None
        best_score = -1

        for name, profile in self.MODEL_PROFILES.items():
            if profile.accuracy_score >= relaxed_accuracy:
                score = self._calculate_model_score(profile, complexity, resources, relaxed_accuracy)
                if score > best_score:
                    best_score = score
                    best_model = (name, profile)

        if best_model:
            name, profile = best_model
            return {
                'model_name': name,
                'model_class': profile.model_class,
                'huggingface_name': profile.name,
                'score': best_score,
                'profile': profile,
                'warning': f"No models met all requirements. Relaxed accuracy to {relaxed_accuracy:.2f}",
                'reasoning': "Best available model under relaxed constraints",
                'alternatives': [],
                'configuration_tips': self._get_configuration_tips(profile, complexity)
            }

        # If still no model, return most efficient
        return {
            'model_name': 'ELECTRASmall',
            'model_class': ELECTRASmall,
            'huggingface_name': 'google/electra-small-discriminator',
            'score': 0,
            'profile': self.MODEL_PROFILES['ELECTRASmall'],
            'warning': "No models met requirements. Defaulting to most efficient model.",
            'reasoning': "Fallback to most resource-efficient model",
            'alternatives': [],
            'configuration_tips': ["Consider relaxing requirements or upgrading resources"]
        }

    def _print_recommendation(self, rec: Dict[str, Any]):
        """Print formatted recommendation."""
        print("\n" + "="*60)
        print("MODEL RECOMMENDATION")
        print("="*60)
        print(f"Recommended: {rec['model_name']}")
        print(f"HuggingFace: {rec['huggingface_name']}")
        print(f"Score: {rec['score']:.1f}")
        print(f"Reasoning: {rec['reasoning']}")

        if 'warning' in rec:
            print(f"\n⚠️  Warning: {rec['warning']}")

        if rec['configuration_tips']:
            print("\nConfiguration Tips:")
            for tip in rec['configuration_tips']:
                print(f"  • {tip}")

        if rec['alternatives']:
            print("\nAlternatives:")
            for alt in rec['alternatives']:
                print(f"  • {alt['name']} (score: {alt['score']:.1f})")

        print("="*60 + "\n")

    def _print_detailed_benchmark_results(self, results: List[BenchmarkResult], language_info: Optional[List[str]] = None):
        """Print comprehensive benchmark results with all metrics."""
        if not results:
            print("\n⚠️ No benchmark results available")
            return

        print("\n" + "="*120)
        print(" " * 40 + "🏆 BENCHMARK RESULTS")
        print("="*120)

        # Main metrics table
        print(f"\n{'Rank':<5} {'Model':<20} {'F1_macro':<10} {'F1_0':<8} {'F1_1':<8} "
              f"{'Prec_0':<8} {'Prec_1':<8} {'Rec_0':<8} {'Rec_1':<8} "
              f"{'Acc':<8} {'Loss':<8} {'Speed(ms)':<10}")
        print("-"*120)

        for i, result in enumerate(results, 1):
            f1_0 = getattr(result, 'f1_class_0', 0)
            f1_1 = getattr(result, 'f1_class_1', 0)
            p_0 = getattr(result, 'precision_0', 0)
            p_1 = getattr(result, 'precision_1', 0)
            r_0 = getattr(result, 'recall_0', 0)
            r_1 = getattr(result, 'recall_1', 0)
            loss = getattr(result, 'val_loss', 0)

            print(f"{i:<5} {result.model_name:<20} {result.f1_score:<10.3f} "
                  f"{f1_0:<8.3f} {f1_1:<8.3f} "
                  f"{p_0:<8.3f} {p_1:<8.3f} "
                  f"{r_0:<8.3f} {r_1:<8.3f} "
                  f"{result.accuracy:<8.3f} {loss:<8.3f} "
                  f"{result.inference_time*1000:<10.1f}")

        # Language-specific analysis if available
        if language_info:
            languages = list(set(language_info))
            if len(languages) > 1:
                print("\n" + "="*120)
                print(" " * 40 + "📊 PER-LANGUAGE ANALYSIS")
                print("="*120)
                print("\nLanguages detected:", ", ".join(sorted(languages)))

                # Show language distribution
                from collections import Counter
                lang_counts = Counter(language_info)
                total = len(language_info)
                print("\nLanguage distribution:")
                for lang, count in sorted(lang_counts.items()):
                    percentage = (count / total) * 100
                    bar_width = int(percentage / 2)
                    bar = '█' * bar_width + '░' * (50 - bar_width)
                    print(f"  {lang:5}: {bar} {count:5} ({percentage:5.1f}%)")

        # Best model analysis
        if results:
            best = results[0]
            print("\n" + "="*120)
            print(" " * 40 + "🥇 BEST MODEL")
            print("="*120)
            print(f"\nModel: {best.model_name}")
            print(f"  • F1 Macro: {best.f1_score:.3f}")
            print(f"  • Accuracy: {best.accuracy:.3f}")
            print(f"  • Training time/epoch: {best.training_time:.1f}s")
            print(f"  • Inference speed: {best.inference_time*1000:.1f}ms/sample")
            print(f"  • Memory usage: {best.memory_usage:.1f}GB")

            # Check for class imbalance
            f1_0 = getattr(best, 'f1_class_0', 0)
            f1_1 = getattr(best, 'f1_class_1', 0)
            if abs(f1_0 - f1_1) > 0.2:
                print(f"\n⚠️ Class imbalance detected: F1_0={f1_0:.3f}, F1_1={f1_1:.3f}")
                print("   Consider using reinforced learning or data augmentation.")

    def _print_benchmark_results(self, results: List[BenchmarkResult]):
        """Print formatted benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'Inference(ms)':<15} {'Memory(GB)':<10}")
        print("-"*60)

        for r in results:
            print(f"{r.model_name:<20} {r.accuracy:<10.3f} {r.f1_score:<10.3f} "
                  f"{r.inference_time*1000:<15.1f} {r.memory_usage:<10.1f}")

        print("="*60 + "\n")


def auto_select_model(
    train_texts: Optional[List[str]] = None,
    task_description: Optional[str] = None,
    resource_constraint: str = "standard"
) -> type:
    """
    Simplified automatic model selection.

    Args:
        train_texts: Sample training texts for complexity analysis
        task_description: Description of the task
        resource_constraint: 'minimal', 'limited', 'standard', or 'premium'

    Returns:
        Model class ready for instantiation
    """
    selector = ModelSelector(verbose=False)

    # Infer complexity from data if provided
    complexity = TaskComplexity.MODERATE
    if train_texts and len(train_texts) > 0:
        avg_length = np.mean([len(t.split()) for t in train_texts[:100]])
        if avg_length < 20:
            complexity = TaskComplexity.SIMPLE
        elif avg_length > 200:
            complexity = TaskComplexity.COMPLEX

    # Map resource constraint
    resource_map = {
        'minimal': ResourceProfile.MINIMAL,
        'limited': ResourceProfile.LIMITED,
        'standard': ResourceProfile.STANDARD,
        'premium': ResourceProfile.PREMIUM
    }
    resources = resource_map.get(resource_constraint, ResourceProfile.STANDARD)

    # Get recommendation
    rec = selector.recommend(
        task_complexity=complexity,
        resource_profile=resources,
        required_accuracy=0.85
    )

    return rec['model_class']

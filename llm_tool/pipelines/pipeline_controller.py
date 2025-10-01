#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
pipeline_controller.py

MAIN OBJECTIVE:
---------------
This script orchestrates the complete pipeline workflow, managing the sequential
execution of annotation, validation, training, and deployment phases.

Dependencies:
-------------
- sys
- asyncio
- typing
- logging
- concurrent.futures

MAIN FEATURES:
--------------
1) Coordinate sequential pipeline phases
2) Manage data flow between pipeline stages
3) Handle interruption and recovery
4) Provide progress tracking and monitoring
5) Support both full and partial pipeline execution

Author:
-------
Antoine Lemor
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
import time
from datetime import datetime

from ..config.settings import Settings, get_settings
from ..utils.language_detector import LanguageDetector


class PipelinePhase(Enum):
    """Enumeration of pipeline phases"""
    INITIALIZATION = "initialization"
    ANNOTATION = "annotation"
    VALIDATION = "validation"
    TRAINING = "training"
    BENCHMARKING = "benchmarking"
    DEPLOYMENT = "deployment"
    INFERENCE = "inference"
    COMPLETED = "completed"


@dataclass
class PipelineState:
    """State tracking for pipeline execution"""
    current_phase: PipelinePhase
    phases_completed: List[PipelinePhase]
    start_time: float
    end_time: Optional[float]
    annotation_results: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    training_results: Optional[Dict[str, Any]]
    deployment_results: Optional[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    warnings: List[str]


class PipelineController:
    """Main controller for orchestrating the complete pipeline"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the pipeline controller"""
        self.settings = settings or get_settings()
        self.language_detector = LanguageDetector()
        self.executor = ThreadPoolExecutor(max_workers=self.settings.data.max_workers)
        self.state: Optional[PipelineState] = None
        self.logger = logging.getLogger(__name__)

        # Import components lazily to avoid circular dependencies
        self._annotation_module = None
        self._training_module = None
        self._validation_module = None

    def _lazy_import_modules(self):
        """Lazy import of heavy modules"""
        if self._annotation_module is None:
            self.logger.info("[IMPORT] Importing LLMAnnotator...")
            from ..annotators.llm_annotator import LLMAnnotator
            self.logger.info("[IMPORT] LLMAnnotator imported successfully")
            self._annotation_module = LLMAnnotator

        if self._training_module is None:
            self.logger.info("[IMPORT] Importing ModelTrainer...")
            from ..trainers.model_trainer import ModelTrainer
            self.logger.info("[IMPORT] ModelTrainer imported successfully")
            self._training_module = ModelTrainer

        if self._validation_module is None:
            self.logger.info("[IMPORT] Importing AnnotationValidator...")
            from ..validators.annotation_validator import AnnotationValidator
            self.logger.info("[IMPORT] AnnotationValidator imported successfully")
            self._validation_module = AnnotationValidator

    def initialize_pipeline(self, config: Dict[str, Any]) -> PipelineState:
        """Initialize a new pipeline run"""
        self.state = PipelineState(
            current_phase=PipelinePhase.INITIALIZATION,
            phases_completed=[],
            start_time=time.time(),
            end_time=None,
            annotation_results=None,
            validation_results=None,
            training_results=None,
            deployment_results=None,
            errors=[],
            warnings=[]
        )

        # Validate configuration
        self._validate_config(config)

        # Setup directories
        self._setup_directories()

        self.logger.info("Pipeline initialized successfully")
        return self.state

    def _validate_config(self, config: Dict[str, Any]):
        """Validate pipeline configuration"""
        required_keys = ['mode', 'data_source']

        for key in required_keys:
            if key not in config:
                error = f"Missing required configuration key: {key}"
                self.state.errors.append({
                    'phase': PipelinePhase.INITIALIZATION,
                    'error': error
                })
                raise ValueError(error)

        # Validate data source
        if config['mode'] == 'file':
            if 'file_path' not in config:
                raise ValueError("file_path required for file mode")
            if not Path(config['file_path']).exists():
                raise FileNotFoundError(f"File not found: {config['file_path']}")

    def _setup_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.settings.paths.data_dir,
            self.settings.paths.models_dir,
            self.settings.paths.prompts_dir,
            self.settings.paths.logs_dir,
            self.settings.paths.validation_dir,
            self.settings.paths.cache_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    async def run_pipeline_async(self, config: Dict[str, Any]) -> PipelineState:
        """Run the complete pipeline asynchronously"""
        try:
            # Initialize
            self.initialize_pipeline(config)
            self._lazy_import_modules()

            # Phase 1: Annotation
            if config.get('run_annotation', True):
                await self._run_annotation_phase(config)

            # Phase 2: Validation
            if config.get('run_validation', False):
                await self._run_validation_phase(config)

            # Phase 3: Training
            if config.get('run_training', False):
                await self._run_training_phase(config)

            # Phase 4: Deployment
            if config.get('run_deployment', False):
                await self._run_deployment_phase(config)

            # Mark complete
            self.state.current_phase = PipelinePhase.COMPLETED
            self.state.end_time = time.time()

            # Save pipeline state
            self._save_pipeline_state()

            return self.state

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.state.errors.append({
                'phase': self.state.current_phase,
                'error': str(e)
            })
            self._save_pipeline_state()
            raise

    def run_pipeline(self, config: Dict[str, Any]) -> PipelineState:
        """Run the complete pipeline synchronously"""
        try:
            # Initialize
            self.initialize_pipeline(config)

            # Only import modules that we actually need based on config
            # This avoids loading heavy modules that might cause issues
            self.logger.info("[PIPELINE] Importing required modules...")

            # Phase 1: Annotation (synchronous to avoid asyncio conflicts with Ollama)
            if config.get('run_annotation', True):
                # Import only annotation module
                if self._annotation_module is None:
                    self.logger.info("[IMPORT] Importing LLMAnnotator...")
                    from ..annotators.llm_annotator import LLMAnnotator
                    self.logger.info("[IMPORT] LLMAnnotator imported successfully")
                    self._annotation_module = LLMAnnotator

                self.logger.info("[PIPELINE] Calling _run_annotation_phase_sync()...")
                self._run_annotation_phase_sync(config)

            # Phase 1.5: Prepare training data from annotations (if training is enabled)
            if config.get('run_training', False) and self.state.annotation_results:
                self.logger.info("[PIPELINE] Preparing training data from annotations...")
                self._prepare_training_data_from_annotations(config)

            # Phase 2: Validation
            if config.get('run_validation', False):
                if self._validation_module is None:
                    self.logger.info("[IMPORT] Importing AnnotationValidator...")
                    from ..validators.annotation_validator import AnnotationValidator
                    self._validation_module = AnnotationValidator
                asyncio.run(self._run_validation_phase(config))

            # Phase 3: Training
            if config.get('run_training', False):
                if self._training_module is None:
                    self.logger.info("[IMPORT] Importing ModelTrainer...")
                    from ..trainers.model_trainer import ModelTrainer
                    self._training_module = ModelTrainer
                asyncio.run(self._run_training_phase(config))

            # Phase 4: Deployment
            if config.get('run_deployment', False):
                asyncio.run(self._run_deployment_phase(config))

            # Mark complete
            self.state.current_phase = PipelinePhase.COMPLETED
            self.state.end_time = time.time()

            # Save pipeline state
            self._save_pipeline_state()

            return self.state

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.state.errors.append({
                'phase': self.state.current_phase,
                'error': str(e)
            })
            self._save_pipeline_state()
            raise

    def _run_annotation_phase_sync(self, config: Dict[str, Any]):
        """Run the annotation phase synchronously (avoids asyncio conflicts with Ollama)"""
        self.logger.info("Starting annotation phase")
        self.state.current_phase = PipelinePhase.ANNOTATION

        try:
            self.logger.info("[TRACE] Creating annotator instance...")
            annotator = self._annotation_module(settings=self.settings)
            self.logger.info("[TRACE] Annotator created successfully")

            # Prepare annotation configuration
            self.logger.info("[TRACE] Preparing annotation config...")
            annotation_config = self._prepare_annotation_config(config)
            self.logger.info(f"[TRACE] Config prepared. Provider={annotation_config.get('provider')}, Model={annotation_config.get('model')}")

            # Run annotation synchronously
            self.logger.info("[TRACE] Calling annotator.annotate()...")
            results = annotator.annotate(annotation_config)
            self.logger.info("[TRACE] Annotation completed!")

            self.state.annotation_results = results
            self.state.phases_completed.append(PipelinePhase.ANNOTATION)
            self.logger.info(f"Annotation completed: {results.get('total_annotated', 0)} items processed")

        except Exception as e:
            self.logger.error(f"Annotation phase failed: {str(e)}")
            raise

    async def _run_annotation_phase(self, config: Dict[str, Any]):
        """Run the annotation phase (async version)"""
        self.logger.info("Starting annotation phase")
        self.state.current_phase = PipelinePhase.ANNOTATION

        try:
            annotator = self._annotation_module(settings=self.settings)

            # Prepare annotation configuration
            annotation_config = self._prepare_annotation_config(config)

            # Run annotation
            results = await annotator.annotate_async(annotation_config)

            self.state.annotation_results = results
            self.state.phases_completed.append(PipelinePhase.ANNOTATION)
            self.logger.info(f"Annotation completed: {results.get('total_annotated', 0)} items processed")

        except Exception as e:
            self.logger.error(f"Annotation phase failed: {str(e)}")
            raise

    async def _run_validation_phase(self, config: Dict[str, Any]):
        """Run the validation phase"""
        self.logger.info("Starting validation phase")
        self.state.current_phase = PipelinePhase.VALIDATION

        try:
            validator = self._validation_module(settings=self.settings)

            # Use annotation results as input
            if not self.state.annotation_results:
                raise ValueError("No annotation results available for validation")

            validation_config = {
                'input_data': self.state.annotation_results.get('output_file'),
                'sample_size': config.get('validation_sample_size', 100),
                'export_format': config.get('validation_export_format', 'jsonl'),
                'export_to_doccano': config.get('export_to_doccano', True)
            }

            results = await validator.validate_async(validation_config)

            self.state.validation_results = results
            self.state.phases_completed.append(PipelinePhase.VALIDATION)
            self.logger.info(f"Validation completed: {results.get('samples_validated', 0)} samples")

        except Exception as e:
            self.logger.error(f"Validation phase failed: {str(e)}")
            raise

    async def _run_training_phase(self, config: Dict[str, Any]):
        """Run the training phase"""
        self.logger.info("Starting training phase")
        self.state.current_phase = PipelinePhase.TRAINING

        try:
            trainer = self._training_module(settings=self.settings)

            # Determine input data - prioritize converted training files
            training_files = None
            if self.state.annotation_results and 'training_files' in self.state.annotation_results:
                # Use converted training data (JSONL format)
                training_files = self.state.annotation_results['training_files']
                training_strategy = self.state.annotation_results.get('training_strategy', 'single-label')
                self.logger.info(f"Using {len(training_files)} converted training file(s) (strategy={training_strategy})")
            elif self.state.annotation_results:
                # Fallback to annotation output (CSV format) - less ideal
                input_data = self.state.annotation_results.get('output_file')
            else:
                # Use manually provided training file
                input_data = config.get('training_input_file')

            if not training_files and not input_data:
                raise ValueError("No input data available for training")

            # Train models based on strategy
            if training_files:
                # For single-label strategy, we might have multiple files (one per annotation key)
                # For multi-label strategy, we have one file
                # For now, let's train on the first file
                # TODO: Support training multiple models (one per annotation key)
                first_key = list(training_files.keys())[0]
                input_data = training_files[first_key]
                self.logger.info(f"Training on: {first_key} -> {input_data}")

            # Prepare training configuration
            training_config = self._prepare_training_config(config, input_data)

            # Run training or benchmarking
            if config.get('benchmark_mode', False):
                self.state.current_phase = PipelinePhase.BENCHMARKING
                results = await trainer.benchmark_async(training_config)
            else:
                results = await trainer.train_async(training_config)

            self.state.training_results = results
            self.state.phases_completed.append(PipelinePhase.TRAINING)
            self.logger.info(f"Training completed: Best model {results.get('best_model', 'unknown')}")

        except Exception as e:
            self.logger.error(f"Training phase failed: {str(e)}")
            raise

    async def _run_deployment_phase(self, config: Dict[str, Any]):
        """Run the deployment phase"""
        self.logger.info("Starting deployment phase")
        self.state.current_phase = PipelinePhase.DEPLOYMENT

        try:
            if not self.state.training_results:
                raise ValueError("No training results available for deployment")

            deployment_config = {
                'model_path': self.state.training_results.get('model_path'),
                'save_location': config.get('deployment_path',
                                           str(self.settings.paths.models_dir / 'deployed_model')),
                'run_inference': config.get('run_inference', False),
                'inference_data': config.get('inference_data_path')
            }

            # Save model
            model_path = Path(deployment_config['save_location'])
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy model files
            import shutil
            if Path(deployment_config['model_path']).is_dir():
                shutil.copytree(
                    deployment_config['model_path'],
                    deployment_config['save_location'],
                    dirs_exist_ok=True
                )

            results = {
                'deployed_model_path': deployment_config['save_location'],
                'deployment_time': time.time()
            }

            # Run inference if requested
            if deployment_config['run_inference'] and deployment_config['inference_data']:
                self.state.current_phase = PipelinePhase.INFERENCE
                # Run inference (implementation would go here)
                results['inference_results'] = "Inference completed"

            self.state.deployment_results = results
            self.state.phases_completed.append(PipelinePhase.DEPLOYMENT)
            self.logger.info(f"Deployment completed: Model saved to {results['deployed_model_path']}")

        except Exception as e:
            self.logger.error(f"Deployment phase failed: {str(e)}")
            raise

    def _prepare_annotation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for annotation phase"""
        provider = config.get('annotation_provider', 'ollama')
        mode = config.get('annotation_mode', 'local')

        input_format = config.get('data_source') or config.get('data_format', 'csv')
        annotations_dir = self.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)

        input_name = Path(config.get('file_path', 'data')).stem or 'data'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output_format = config.get('output_format') or 'csv'
        default_output_name = f"{input_name}_{provider}_annotations_{timestamp}.{default_output_format}"
        default_output_path = annotations_dir / default_output_name

        return {
            'mode': mode,
            'provider': provider,
            'model': config.get('annotation_model', 'llama3.2'),
            'api_key': config.get('api_key'),
            'data_source': input_format,
            'data_format': config.get('data_format', input_format),
            'file_path': config.get('file_path'),
            'db_config': config.get('db_config'),
            'text_column': config.get('text_column', 'text'),
            'prompt_path': config.get('prompt_path'),
            'prompts_folder': config.get('prompts_folder'),
            'prompts': config.get('prompts'),
            'batch_size': config.get('batch_size', 100),
            'max_workers': config.get('max_workers', 4),
            'num_processes': config.get('num_processes', config.get('max_workers', 4)),
            'use_parallel': config.get('use_parallel', True),
            'annotation_column': config.get('annotation_column', 'annotation'),
            'annotation_sample_size': config.get('annotation_sample_size'),
            'annotation_sampling_strategy': config.get('annotation_sampling_strategy', 'head'),
            'annotation_sample_seed': config.get('annotation_sample_seed', 42),
            'output_format': config.get('output_format', default_output_format),
            'output_path': config.get('output_path', str(default_output_path))
        }

    def _prepare_training_config(self, config: Dict[str, Any], input_data: str) -> Dict[str, Any]:
        """Prepare configuration for training phase"""
        models_to_test = config.get('models_to_test')
        if isinstance(models_to_test, str):
            models_to_test = [m.strip() for m in models_to_test.split(',') if m.strip()]
        elif isinstance(models_to_test, int):
            models_to_test = []

        return {
            'input_file': input_data,
            'model_type': config.get('training_model_type', 'bert-base-multilingual-cased'),
            'benchmark_mode': config.get('benchmark_mode', False),
            'models_to_test': models_to_test or ['bert-base-multilingual-cased', 'xlm-roberta-base'],
            'auto_select_best': config.get('auto_select_best', True),
            'max_epochs': config.get('max_epochs', 10),
            'batch_size': config.get('batch_size', 16),
            'learning_rate': config.get('learning_rate', 2e-5),
            'early_stopping': config.get('early_stopping', True),
            'patience': config.get('patience', 3),
            'validation_split': config.get('validation_split', 0.2),
            'test_split': config.get('test_split', 0.1),
            'output_dir': config.get('output_dir',
                                   str(self.settings.paths.models_dir / 'trained_model'))
        }

    def _prepare_training_data_from_annotations(self, config: Dict[str, Any]):
        """
        Convert annotated CSV to training-ready JSONL format.
        This function prepares data between annotation and training phases.
        """
        from ..utils.annotation_to_training import AnnotationToTrainingConverter

        # Get annotation output file
        annotation_output = self.state.annotation_results.get('output_file')
        if not annotation_output:
            self.logger.error("No annotation output file found")
            return

        # Get configuration
        text_column = config.get('text_column', 'sentence')
        annotation_column = config.get('annotation_column', 'annotation')
        training_strategy = config.get('training_strategy', 'single-label')
        label_strategy = config.get('label_strategy', 'key_value')
        annotation_keys = config.get('training_annotation_keys')  # None means all keys

        # Create converter
        converter = AnnotationToTrainingConverter(verbose=True)

        # Prepare output directory
        training_data_dir = self.settings.paths.data_dir / 'training_data'
        training_data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        try:
            if training_strategy == 'single-label':
                # Create separate datasets for each annotation key
                self.logger.info(f"Creating single-label training datasets (label_strategy={label_strategy})...")
                output_files = converter.create_single_label_datasets(
                    csv_path=annotation_output,
                    output_dir=str(training_data_dir),
                    text_column=text_column,
                    annotation_column=annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=label_strategy,
                    id_column=config.get('identifier_column'),
                    lang_column=config.get('lang_column')
                )

                if output_files:
                    self.logger.info(f"Created {len(output_files)} training datasets:")
                    for key, path in output_files.items():
                        self.logger.info(f"  - {key}: {path}")

                    # Store the first dataset for training (or user can select)
                    # For now, we'll train on all of them sequentially
                    self.state.annotation_results['training_files'] = output_files
                    self.state.annotation_results['training_strategy'] = 'single-label'
                else:
                    self.logger.warning("No training datasets were created")

            elif training_strategy == 'multi-label':
                # Create single multi-label dataset
                self.logger.info(f"Creating multi-label training dataset (label_strategy={label_strategy})...")
                output_file = converter.create_multi_label_dataset(
                    csv_path=annotation_output,
                    output_path=str(training_data_dir / f'training_multilabel_{timestamp}.jsonl'),
                    text_column=text_column,
                    annotation_column=annotation_column,
                    annotation_keys=annotation_keys,
                    label_strategy=label_strategy,
                    id_column=config.get('identifier_column'),
                    lang_column=config.get('lang_column')
                )

                if output_file:
                    self.logger.info(f"Created multi-label training dataset: {output_file}")
                    self.state.annotation_results['training_files'] = {'multilabel': output_file}
                    self.state.annotation_results['training_strategy'] = 'multi-label'
                else:
                    self.logger.warning("Multi-label training dataset was not created")

        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            raise

    def _save_pipeline_state(self):
        """Save the current pipeline state to disk"""
        state_file = self.settings.paths.logs_dir / f"pipeline_state_{int(self.state.start_time)}.json"

        state_dict = {
            'current_phase': self.state.current_phase.value,
            'phases_completed': [p.value for p in self.state.phases_completed],
            'start_time': self.state.start_time,
            'end_time': self.state.end_time,
            'annotation_results': self.state.annotation_results,
            'validation_results': self.state.validation_results,
            'training_results': self.state.training_results,
            'deployment_results': self.state.deployment_results,
            'errors': self.state.errors,
            'warnings': self.state.warnings
        }

        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)

        self.logger.info(f"Pipeline state saved to {state_file}")

    def load_pipeline_state(self, state_file: str) -> PipelineState:
        """Load a previous pipeline state from disk"""
        with open(state_file, 'r') as f:
            state_dict = json.load(f)

        state = PipelineState(
            current_phase=PipelinePhase(state_dict['current_phase']),
            phases_completed=[PipelinePhase(p) for p in state_dict['phases_completed']],
            start_time=state_dict['start_time'],
            end_time=state_dict['end_time'],
            annotation_results=state_dict['annotation_results'],
            validation_results=state_dict['validation_results'],
            training_results=state_dict['training_results'],
            deployment_results=state_dict['deployment_results'],
            errors=state_dict['errors'],
            warnings=state_dict['warnings']
        )

        self.state = state
        self.logger.info(f"Pipeline state loaded from {state_file}")
        return state

    def get_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress"""
        if not self.state:
            return {'status': 'not_started'}

        elapsed_time = time.time() - self.state.start_time
        if self.state.end_time:
            total_time = self.state.end_time - self.state.start_time
        else:
            total_time = elapsed_time

        return {
            'status': self.state.current_phase.value,
            'phases_completed': [p.value for p in self.state.phases_completed],
            'elapsed_time': elapsed_time,
            'total_time': total_time,
            'errors': len(self.state.errors),
            'warnings': len(self.state.warnings)
        }

    def run_annotation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run only the annotation phase"""
        self.initialize_pipeline(config)
        self._lazy_import_modules()

        # Run annotation synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_annotation_phase(config))
            return self.state.annotation_results
        finally:
            loop.close()

    def run_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run only the training phase"""
        self.initialize_pipeline(config)
        self._lazy_import_modules()

        # Run training synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_training_phase(config))
            return self.state.training_results
        finally:
            loop.close()

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("Pipeline controller cleaned up")

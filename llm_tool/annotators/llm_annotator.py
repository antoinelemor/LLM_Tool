#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
llm_annotator.py

MAIN OBJECTIVE:
---------------
This is the core annotation module that provides comprehensive LLM annotation
capabilities including single and multi-prompt processing, parallel execution,
JSON repair, schema validation, and support for multiple data sources.

Dependencies:
-------------
- sys
- os
- json
- pandas
- numpy
- logging
- typing
- concurrent.futures
- time
- math
- random
- pathlib
- tqdm
- sqlalchemy
- pydantic

MAIN FEATURES:
--------------
1) Single and multi-prompt annotation processing
2) Parallel execution with ProcessPoolExecutor
3) JSON repair and validation (5 retry attempts)
4) Schema validation with Pydantic
5) Support for PostgreSQL, CSV, Excel, Parquet, RData/RDS
6) Incremental saving and resume capability
7) Progress tracking with error handling
8) Sample size calculation (95% CI)
9) Warm-up calls for Ollama models
10) Per-prompt status tracking

Author:
-------
Antoine Lemor
"""

import os
import sys
import json
import shutil
import logging
import time
import math
import random
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Iterable, Set
from datetime import datetime
from collections import deque, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, create_model

# SQLAlchemy imports for database support
try:
    from sqlalchemy import create_engine, text, JSON, bindparam
    from sqlalchemy.exc import SQLAlchemyError
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logging.warning("SQLAlchemy not installed. PostgreSQL support disabled.")

# pyreadr for RData/RDS support
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    logging.warning("pyreadr not installed. RData/RDS support disabled.")

# Import from other modules
from ..annotators.api_clients import create_api_client
from ..annotators.prompt_manager import PromptManager
from ..annotators.json_cleaner import JSONCleaner, clean_json_output
from ..config.settings import Settings
from ..utils.data_filter_logger import get_filter_logger

# Try to import local model support
try:
    from ..annotators.local_models import OllamaClient, LlamaCPPClient
    HAS_LOCAL_MODELS = True
except ImportError:
    HAS_LOCAL_MODELS = False
    logging.warning("Local model support not available")

# OpenAI SDK for batch operations
try:
    from openai import OpenAI, NotFoundError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    NotFoundError = Exception
    logging.warning("OpenAI SDK not installed. Batch API support disabled.")

# Rich library for enhanced CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# Constants
CSV_APPEND = True
OTHER_FORMAT_SAVE_EVERY = 50
PROMPT_SUFFIXES = ["raw_per_prompt", "cleaned_per_prompt", "status_per_prompt"]

# Global tracking
status_counts = {"success": 0, "error": 0, "cleaning_failed": 0, "decode_error": 0}


class LLMAnnotator:
    """Main LLM annotation class with comprehensive features"""

    def __init__(self, settings: Optional[Settings] = None, progress_callback=None, progress_manager=None):
        """Initialize the LLM annotator

        Args:
            settings: Optional Settings object
            progress_callback: Optional callback for progress updates (current, total, message)
            progress_manager: Optional progress manager for displaying warnings/errors
        """
        self.settings = settings or Settings()
        self.progress_callback = progress_callback
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)

        # If we have a progress manager, disable console logging to avoid conflicts
        if self.progress_manager:
            # Remove all console handlers to prevent duplicate output
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    self.logger.removeHandler(handler)
            # Also prevent propagation to root logger
            self.logger.propagate = False

        self.json_cleaner = JSONCleaner()
        self.prompt_manager = PromptManager()
        self.api_client = None
        self.local_client = None
        self.progress_bar = None
        self.last_annotation = None  # Store last successful annotation for display

    def annotate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main annotation entry point.

        Parameters
        ----------
        config : dict
            Configuration including:
            - data_source: 'csv', 'excel', 'parquet', 'rdata', 'rds', 'postgresql'
            - file_path or db_config: Data location
            - prompts: List of prompt configurations
            - model: Model configuration
            - num_processes: Number of parallel processes
            - output_path: Where to save results
            - resume: Whether to resume from existing annotations

        Returns
        -------
        dict
            Annotation results and statistics
        """
        # Reset counters to avoid leaking state across multiple runs
        global status_counts
        status_counts = {"success": 0, "error": 0, "cleaning_failed": 0, "decode_error": 0}

        # Validate configuration
        self.logger.info("[ANNOTATOR] Validating config...")
        self._validate_config(config)

        # Setup model client
        self.logger.info(f"[ANNOTATOR] Setting up model client for {config.get('model')}...")
        self._setup_model_client(config)
        self.logger.info("[ANNOTATOR] Model client setup complete")

        # Load data
        self.logger.info("[ANNOTATOR] Loading data...")
        data, metadata = self._load_data(config)
        self.logger.info(f"[ANNOTATOR] Loaded {len(data)} rows")

        # Prepare prompts
        self.logger.info("[ANNOTATOR] Preparing prompts...")
        prompts = self._prepare_prompts(config)
        self.logger.info(f"[ANNOTATOR] Prepared {len(prompts)} prompt(s)")

        # Perform annotation
        self.logger.info("[ANNOTATOR] Starting annotation process...")
        results = self._annotate_data(data, prompts, config)

        # Save results
        self.logger.info("[ANNOTATOR] Saving results...")
        self._save_results(results, config)

        return self._generate_summary(results, config)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate annotation configuration"""
        required = ['data_source', 'model', 'output_path']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

    def _setup_model_client(self, config: Dict[str, Any]):
        """Setup the appropriate model client"""
        # Handle both dict and string model config
        if isinstance(config.get('model'), str):
            # Simple string model name - use provider from config
            model_name = config['model']
            provider = config.get('provider', 'ollama')
            api_key = config.get('api_key')
        else:
            # Dict model config
            model_config = config['model']
            provider = model_config.get('provider', 'ollama')
            model_name = model_config.get('model_name')
            api_key = model_config.get('api_key')

        if provider in ['openai', 'anthropic', 'google']:
            self.api_client = create_api_client(
                provider=provider,
                api_key=api_key,
                model=model_name,
                progress_manager=self.progress_manager  # Pass progress manager for warnings/errors
            )
        elif provider == 'ollama' and HAS_LOCAL_MODELS:
            self.local_client = OllamaClient(model_name)
        elif provider == 'llamacpp' and HAS_LOCAL_MODELS:
            self.local_client = LlamaCPPClient(model_name)  # Assuming path for llamacpp
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _load_data(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Load data from various sources"""
        source = config['data_source']
        metadata = {'source': source}

        if source == 'postgresql':
            if not HAS_SQLALCHEMY:
                raise ImportError("SQLAlchemy required for PostgreSQL support")
            return self._load_postgresql(config['db_config']), metadata

        elif source in ['csv', 'excel', 'parquet', 'rdata', 'rds']:
            return self._load_file(config['file_path'], source), metadata

        else:
            raise ValueError(f"Unsupported data source: {source}")

    def _load_file(self, file_path: str, format: str) -> pd.DataFrame:
        """Load data from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if format == 'csv':
            return pd.read_csv(path)
        elif format == 'excel':
            return pd.read_excel(path)
        elif format == 'parquet':
            return pd.read_parquet(path)
        elif format == 'json':
            return pd.read_json(path, lines=False)
        elif format == 'jsonl':
            return pd.read_json(path, lines=True)
        elif format in ['rdata', 'rds']:
            if not HAS_PYREADR:
                raise ImportError("pyreadr required for RData/RDS files")
            result = pyreadr.read_r(path)
            return list(result.values())[0]
        else:
            raise ValueError(f"Unsupported file format: {format}")

    def _load_postgresql(self, db_config: Dict) -> pd.DataFrame:
        """Load data from PostgreSQL"""
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config.get('port', 5432)}/{db_config['database']}"
        )

        query = db_config.get('query') or f"SELECT * FROM {db_config['table']}"
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    def _prepare_prompts(self, config: Dict[str, Any]) -> List[Dict]:
        """Prepare prompts for annotation"""
        prompts_config = config.get('prompts', [])
        if not prompts_config:
            # Load from prompt files if specified
            prompt_dir = config.get('prompt_dir')
            if prompt_dir:
                prompts_config = self.prompt_manager.load_prompts_from_directory(prompt_dir)
            else:
                raise ValueError("No prompts specified")

        return prompts_config

    def _annotate_data(
        self,
        data: pd.DataFrame,
        prompts: List[Dict],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Perform annotation on the data.
        """
        # Setup columns
        text_columns = config.get('text_columns', [])
        if not text_columns:
            # Auto-detect text columns
            text_columns = self._detect_text_columns(data)

        identifier_column = config.get('identifier_column')
        if not identifier_column:
            # Create unique identifier
            identifier_column = self._create_unique_id(data)

        config.setdefault('export_to_doccano', True)
        annotation_column = config.get('annotation_column', 'annotation')
        resume = config.get('resume', False)

        # Filter data for annotation
        filter_logger = get_filter_logger()
        if resume and annotation_column in data.columns:
            data_before_filter = data.copy()
            data_to_annotate = data[data[annotation_column].isna()].copy()

            # Log already-annotated rows that are being skipped
            if len(data_to_annotate) < len(data_before_filter):
                filter_logger.log_dataframe_filtering(
                    df_before=data_before_filter,
                    df_after=data_to_annotate,
                    reason="already_annotated",
                    location="llm_annotator._annotate_data.resume_mode",
                    text_column=text_columns[0] if text_columns else None,
                    log_filtered_samples=3
                )

            self.logger.info(f"Resuming annotation: {len(data_to_annotate)} rows to process")
        else:
            data_to_annotate = data.copy()

        # Allow explicit skip lists (e.g. when continuing from checkpoints).
        skip_ids = config.get('skip_annotation_ids') or []
        if skip_ids and identifier_column in data_to_annotate.columns:
            skip_ids_set = set(skip_ids)
            data_before_skip = data_to_annotate.copy()
            before_skip = len(data_before_skip)
            data_to_annotate = data_before_skip[
                ~data_before_skip[identifier_column].isin(skip_ids_set)
            ].copy()
            skipped = before_skip - len(data_to_annotate)
            if skipped > 0:
                filter_logger.log_dataframe_filtering(
                    df_before=data_before_skip,
                    df_after=data_to_annotate,
                    reason="resume_skip_ids",
                    location="llm_annotator._annotate_data.skip_ids",
                    text_column=text_columns[0] if text_columns else None,
                    log_filtered_samples=3
                )
                self.logger.info("Skipping %s row(s) already annotated (explicit list)", skipped)

        annotation_limit = config.get('annotation_sample_size') or config.get('annotation_limit')
        if annotation_limit and len(data_to_annotate) > annotation_limit:
            data_before_limit = data_to_annotate.copy()
            strategy = config.get('annotation_sampling_strategy', 'head')
            sample_seed = config.get('annotation_sample_seed', 42)
            if strategy == 'random':
                data_to_annotate = data_to_annotate.sample(annotation_limit, random_state=sample_seed)
            else:
                data_to_annotate = data_to_annotate.head(annotation_limit)

            # Log filtered rows
            filter_logger.log_dataframe_filtering(
                df_before=data_before_limit,
                df_after=data_to_annotate,
                reason=f"annotation_limit_{strategy}",
                location="llm_annotator._annotate_data.sampling",
                text_column=text_columns[0] if text_columns else None,
                log_filtered_samples=3
            )

            self.logger.info(
                "Limiting annotation to %s rows using '%s' sampling strategy",
                len(data_to_annotate),
                strategy
            )

        # Calculate sample size if requested
        if config.get('calculate_sample_size', False):
            sample_size = self.calculate_sample_size(len(data_to_annotate))
            if config.get('use_sample', False):
                data_to_annotate = data_to_annotate.sample(n=sample_size, random_state=42)
                self.logger.info(f"Using sample of {sample_size} rows")

        # Prepare for parallel processing
        use_parallel = config.get('use_parallel', True)
        num_processes = config.get('num_processes', 1)
        if not use_parallel:
            num_processes = 1
        multiple_prompts = len(prompts) > 1

        # Add necessary columns
        if annotation_column not in data.columns:
            data[annotation_column] = pd.NA
        if f"{annotation_column}_inference_time" not in data.columns:
            data[f"{annotation_column}_inference_time"] = pd.NA

        # Special handling for OpenAI batch mode
        if config.get('annotation_mode') == 'openai_batch':
            return self._execute_openai_batch_annotation(
                full_data=data,
                data_subset=data_to_annotate,
                prompts=prompts,
                text_columns=text_columns,
                identifier_column=identifier_column,
                config=config
            )

        # Warm up model if using local
        if self.local_client and config.get('warmup', True):
            self._warmup_model()

        # Prepare tasks
        tasks = self._prepare_annotation_tasks(
            data_to_annotate,
            prompts,
            text_columns,
            identifier_column,
            config
        )

        # Execute annotation (sequential fallback when only one process requested)
        if num_processes <= 1:
            annotated_data = self._execute_sequential_annotation(
                data,
                tasks,
                annotation_column,
                identifier_column,
                config
            )
        else:
            annotated_data = self._execute_parallel_annotation(
                data,
                tasks,
                num_processes,
                annotation_column,
                identifier_column,
                config
            )

        return annotated_data

    def _detect_text_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect text columns in dataframe"""
        text_columns = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if it contains text-like data
                sample = data[col].dropna().head(10)
                if len(sample) > 0:
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Likely text content
                        text_columns.append(col)
        return text_columns

    def _create_unique_id(self, data: pd.DataFrame) -> str:
        """Create unique identifier column"""
        id_column = 'llm_annotation_id'
        if id_column not in data.columns:
            data[id_column] = range(1, len(data) + 1)
            self.logger.info(f"Created unique identifier column: {id_column}")
        return id_column

    def _warmup_model(self):
        """Warm up local model with test call"""
        try:
            test_prompt = 'Return a simple JSON: {"test": true}'
            if self.local_client:
                result = self.local_client.generate(test_prompt)
                if result:
                    self.logger.info("Model warmed up successfully")
                else:
                    self.logger.warning("Warm-up returned no result")
        except Exception as e:
            self.logger.warning(f"Warm-up failed: {e}")

    def _prepare_annotation_tasks(
        self,
        data: pd.DataFrame,
        prompts: List[Dict],
        text_columns: List[str],
        identifier_column: str,
        config: Dict[str, Any]
    ) -> List[Dict]:
        """Prepare tasks for parallel annotation"""
        tasks = []
        
        for idx, row in data.iterrows():
            # Build model config dict for the task
            if isinstance(config.get('model'), str):
                model_config = {
                    'provider': config.get('provider', 'ollama'),
                    'model_name': config.get('model'),
                    'api_key': config.get('api_key'),
                    'temperature': config.get('temperature', 0.7),
                    'max_tokens': config.get('max_tokens', 1000)
                }
            else:
                model_config = config.get('model', {})

            task = {
                'index': idx,
                'row': row,
                'prompts': prompts,
                'text_columns': text_columns,
                'identifier_column': identifier_column,
                'identifier': row[identifier_column],
                'model_config': model_config,
                'options': config.get('options', {})
            }
            tasks.append(task)

        return tasks

    def _execute_parallel_annotation(
        self,
        full_data: pd.DataFrame,
        tasks: List[Dict],
        num_processes: int,
        annotation_column: str,
        identifier_column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Execute annotation tasks in parallel.
        """
        total_tasks = len(tasks)
        output_path = config.get('output_path')
        save_incrementally = config.get('save_incrementally', True)
        log_enabled = config.get('enable_logging', False)
        log_path = config.get('log_path')
        output_format = config.get('output_format', config.get('data_source', 'csv'))

        if not tasks:
            message = "[ANNOTATOR] No tasks to process after filtering; returning original dataset."
            if self.progress_manager:
                self.progress_manager.show_warning(message)
            else:
                self.logger.warning(message)
            return full_data

        # Report initial progress - DEBUG with file logging
        import sys
        try:
            with open('/tmp/llmtool_debug.log', 'a') as f:
                f.write(f"[ANNOTATOR] Starting with {total_tasks} tasks, callback={self.progress_callback is not None}\n")
        except:
            pass

        if self.progress_callback:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR] Calling progress_callback(0, {total_tasks}, ...)\n")
            except:
                pass
            self.progress_callback(0, total_tasks, f"Starting annotation of {total_tasks} items")
        else:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR] ERROR: No progress_callback!\n")
            except:
                pass

        # Initialize progress bar with position lock to prevent line jumps
        disable_pbar = config.get('disable_tqdm', False)
        with tqdm(total=total_tasks, desc='🤖 LLM Annotation', unit='items',
                  position=0, leave=True, dynamic_ncols=True, disable=disable_pbar) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all tasks
                if len(tasks[0]['prompts']) > 1:
                    futures = {
                        executor.submit(process_multiple_prompts, task): task['identifier']
                        for task in tasks
                    }
                else:
                    futures = {
                        executor.submit(process_single_prompt, task): task['identifier']
                        for task in tasks
                    }

                pending_save = 0
                batch_results = []
                completed_count = 0

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        identifier = result['identifier']
                        final_json = result['final_json']
                        inference_time = result['inference_time']
                        status = result.get('status', 'unknown')

                        # Update dataframe
                        mask = full_data[identifier_column] == identifier
                        if mask.any():
                            full_data.loc[mask, annotation_column] = final_json
                            full_data.loc[mask, f"{annotation_column}_inference_time"] = inference_time

                        # Track status
                        if final_json:
                            status_counts['success'] += 1
                            batch_results.append((identifier, final_json, inference_time))
                            # Store last successful annotation for display
                            try:
                                self.last_annotation = json.loads(final_json) if isinstance(final_json, str) else final_json
                            except:
                                pass
                        else:
                            status_counts['error'] += 1

                        # Incremental saving
                        if save_incrementally and output_path:
                            if output_format == 'csv' and CSV_APPEND:
                                self._append_to_csv(full_data, identifier, identifier_column, annotation_column, output_path)
                            else:
                                pending_save += 1
                                if pending_save >= OTHER_FORMAT_SAVE_EVERY:
                                    self._save_data(full_data, output_path, output_format)
                                    pending_save = 0

                        # Log if enabled
                        if log_enabled and log_path:
                            self._write_log_entry(
                                log_path,
                                {
                                    'id': identifier,
                                    'final_json': final_json,
                                    'inference_time': inference_time,
                                    'status': status
                                }
                            )

                        # Display sample results only if progress bar is enabled
                        if not disable_pbar and len(batch_results) >= 10:
                            sample = random.choice(batch_results)
                            tqdm.write(f"✨ Sample annotation for ID {sample[0]}: {sample[1][:100]}...")
                            batch_results = []

                        pbar.update(1)

                        # Report progress via callback if available
                        completed_count += 1
                        if self.progress_callback:
                            self.progress_callback(completed_count, total_tasks,
                                f"Annotated {completed_count}/{total_tasks} items")

                    except Exception as e:
                        import traceback
                        self.logger.error(f"Task failed: {e}")
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        status_counts['error'] += 1
                        pbar.update(1)

                        # Report error progress too
                        completed_count += 1
                        if self.progress_callback:
                            self.progress_callback(completed_count, total_tasks,
                                f"Annotated {completed_count}/{total_tasks} items ({status_counts['error']} errors)")

        # Final save if needed
        if save_incrementally and output_path and pending_save > 0:
            self._save_data(full_data, output_path, output_format)

        # Report final progress
        if self.progress_callback:
            self.progress_callback(total_tasks, total_tasks, f"Completed annotation of {total_tasks} items")

        return full_data

    def _execute_sequential_annotation(
        self,
        full_data: pd.DataFrame,
        tasks: List[Dict],
        annotation_column: str,
        identifier_column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Execute annotation tasks sequentially (no process pool)."""
        output_path = config.get('output_path')
        save_incrementally = config.get('save_incrementally', True)
        log_enabled = config.get('enable_logging', False)
        log_path = config.get('log_path')
        output_format = config.get('output_format', config.get('data_source', 'csv'))

        pending_save = 0
        total_tasks = len(tasks)
        completed_count = 0

        # Report initial progress
        if self.progress_callback:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR SEQUENTIAL] Starting with {total_tasks} tasks, callback={self.progress_callback is not None}\n")
                    f.flush()
            except:
                pass
            self.progress_callback(0, total_tasks, f"Starting annotation of {total_tasks} items")

        disable_pbar = config.get('disable_tqdm', False)
        for task in tqdm(tasks, desc='🤖 LLM Annotation', unit='items',
                         position=0, leave=True, dynamic_ncols=True, disable=disable_pbar):
            if len(task['prompts']) > 1:
                result = process_multiple_prompts(task)
            else:
                result = process_single_prompt(task)

            identifier = result['identifier']
            final_json = result['final_json']
            inference_time = result['inference_time']
            raw_json = result.get('raw_json')
            cleaned_json = result.get('cleaned_json')
            status = result.get('status', 'unknown')

            mask = full_data[identifier_column] == identifier
            if mask.any():
                full_data.loc[mask, annotation_column] = final_json
                full_data.loc[mask, f"{annotation_column}_inference_time"] = inference_time

            if final_json:
                status_counts['success'] += 1
                # Store last successful annotation for display
                try:
                    self.last_annotation = json.loads(final_json) if isinstance(final_json, str) else final_json
                except:
                    pass
            else:
                status_counts['error'] += 1

            if save_incrementally and output_path:
                if output_format == 'csv' and CSV_APPEND:
                    self._append_to_csv(full_data, identifier, identifier_column, annotation_column, output_path)
                else:
                    pending_save += 1
                    if pending_save >= OTHER_FORMAT_SAVE_EVERY:
                        self._save_data(full_data, output_path, output_format)
                        pending_save = 0

            if log_enabled and log_path:
                self._write_log_entry(
                    log_path,
                    {
                        'id': identifier,
                        'final_json': final_json,
                        'inference_time': inference_time,
                        'status': status
                    }
                )

            # Report progress via callback if available
            completed_count += 1
            if self.progress_callback:
                try:
                    with open('/tmp/llmtool_debug.log', 'a') as f:
                        f.write(f"[ANNOTATOR SEQUENTIAL] Progress: {completed_count}/{total_tasks}\n")
                        f.flush()
                except:
                    pass
                self.progress_callback(completed_count, total_tasks,
                    f"Annotated {completed_count}/{total_tasks} items")

        if save_incrementally and output_path and output_format != 'csv' and pending_save > 0:
            self._save_data(full_data, output_path, output_format)
        elif not save_incrementally and output_path:
            self._save_data(full_data, output_path, output_format)

        # Report final progress
        if self.progress_callback:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR SEQUENTIAL] Completed all {total_tasks} tasks\n")
                    f.flush()
            except:
                pass
            self.progress_callback(total_tasks, total_tasks, f"Completed annotation of {total_tasks} items")

        return full_data

    def _execute_openai_batch_annotation(
        self,
        full_data: pd.DataFrame,
        data_subset: pd.DataFrame,
        prompts: List[Dict],
        text_columns: List[str],
        identifier_column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Execute annotation using the OpenAI Batch API."""
        start_time = time.perf_counter()
        global status_counts
        status_counts = {"success": 0, "error": 0, "cleaning_failed": 0, "decode_error": 0}

        if not HAS_OPENAI:
            raise ImportError("OpenAI SDK is required for batch mode. Install the 'openai' package.")

        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required for batch mode.")

        model_name = config.get('model') or config.get('annotation_model')
        if not model_name:
            raise ValueError("Annotation model must be specified for OpenAI batch mode.")

        total_rows = len(data_subset)
        prompt_count = len(prompts)
        if total_rows == 0 or prompt_count == 0:
            self.logger.info("[BATCH] No rows or prompts supplied; skipping batch annotation.")
            return full_data

        annotation_column = config.get('annotation_column', 'annotation')
        output_path = config.get('output_path')
        output_format = config.get('output_format', config.get('data_source', 'csv'))
        save_incrementally = config.get('save_incrementally', True)
        log_enabled = config.get('enable_logging', False)
        log_path = config.get('log_path')

        batch_base = config.get('openai_batch_dir')
        if batch_base:
            batch_dir = Path(batch_base)
        else:
            batch_dir = self.settings.paths.logs_dir / "openai_batches"

        input_dir = batch_dir / "inputs"
        output_dir = batch_dir / "outputs"
        for directory in (batch_dir, input_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_file_path = input_dir / f"openai_batch_input_{timestamp}.jsonl"
        output_file_path = output_dir / f"openai_batch_output_{timestamp}.jsonl"
        metadata_file_path = batch_dir / f"openai_batch_metadata_{timestamp}.json"

        total_requests = total_rows * prompt_count
        self.logger.info("[BATCH] Preparing %s requests for OpenAI batch job...", total_requests)

        request_entries: List[Dict[str, Any]] = []
        request_metadata: Dict[str, Dict[str, Any]] = {}
        per_row_custom_ids: Dict[str, List[str]] = defaultdict(list)
        prompts_by_index: Dict[int, Dict[str, Any]] = {
            idx + 1: prompt_cfg for idx, prompt_cfg in enumerate(prompts)
        }
        prefixed_expected_keys: Dict[int, List[str]] = {}
        all_expected_keys: Set[str] = set()
        for prompt_idx, prompt_cfg in prompts_by_index.items():
            expected_keys = prompt_cfg.get('expected_keys') or []
            prefix = prompt_cfg.get('prefix', '') or ''
            normalized_keys: List[str] = []
            for key in expected_keys:
                if not key:
                    continue
                normalized_keys.append(f"{prefix}_{key}" if prefix else key)
            prefixed_expected_keys[prompt_idx] = normalized_keys
            all_expected_keys.update(normalized_keys)
        row_lookup: Dict[str, Dict[str, Any]] = {}

        usage_keys = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )

        def aggregate_usage(usage_values: Iterable[Dict[str, Any]]) -> Dict[str, float]:
            totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
                "reasoning_tokens": 0,
            }
            for usage in usage_values:
                if not usage:
                    continue
                for key in usage_keys:
                    totals[key] += usage.get(key) or 0
                prompt_details = usage.get("prompt_tokens_details") or {}
                totals["cached_prompt_tokens"] += prompt_details.get("cached_tokens") or 0
                completion_details = usage.get("completion_tokens_details") or {}
                totals["reasoning_tokens"] += completion_details.get("reasoning_tokens") or 0
            return totals

        def compose_text(row: pd.Series) -> str:
            segments: List[str] = []
            for column in text_columns:
                value = row.get(column)
                if pd.notna(value):
                    segments.append(str(value))
            return "\n\n".join(segments).strip()

        model_lower = str(model_name).lower()

        def build_request_body(prompt_text: str) -> Dict[str, Any]:
            body: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
            }
            is_o_series = (
                model_lower == 'o1'
                or model_lower.startswith('o1-')
                or model_lower.startswith('o3-')
                or model_lower.startswith('o4-')
            )
            is_2025_model = any(token in model_lower for token in ['2025', 'gpt-5', 'gpt5'])
            default_max_tokens = 2000 if (is_o_series or is_2025_model) else 1000
            max_tokens = config.get('max_tokens', default_max_tokens)
            if is_o_series or is_2025_model:
                body["max_completion_tokens"] = max_tokens
                body["temperature"] = 1.0
                body["top_p"] = 1.0
                if config.get('temperature') not in (None, 1, 1.0):
                    self.logger.debug(
                        "[BATCH] Overriding temperature to 1.0 for model %s (batch API restriction).",
                        model_name
                    )
            else:
                body["max_tokens"] = max_tokens
                body["temperature"] = config.get('temperature', 0.7)
                body["top_p"] = config.get('top_p', 1.0)
            return body

        def retry_failed_prompts(
            identifier_key: str,
            missing_indices: List[int],
            row_state: Optional[Dict[str, Any]]
        ) -> None:
            """Sequentially retry prompts that failed in the batch job."""
            if not missing_indices:
                return

            row_dict = row_lookup.get(identifier_key)
            if not row_dict:
                self.logger.warning("[BATCH] Unable to retry %s – original row data missing.", identifier_key)
                return

            local_row_state = row_state
            if local_row_state is None:
                local_row_state = {
                    'identifier': row_dict.get(identifier_column),
                    'row_index': None,
                    'raw': {},
                    'cleaned': {},
                    'status': {},
                    'merged': {},
                    'errors': [],
                    'usage': {},
                    'response_info': {}
                }
                row_results[identifier_key] = local_row_state

            retry_max_tokens = config.get(
                'openai_batch_retry_max_tokens',
                max(config.get('max_tokens', 1000) * 2, 2048)
            )
            retry_model_config = {
                'provider': config.get('annotation_provider') or config.get('provider', 'openai'),
                'model_name': model_name,
                'api_key': api_key,
                'temperature': config.get('temperature', 0.7),
                'top_p': config.get('top_p', 1.0),
                'max_tokens': retry_max_tokens
            }

            row_series = pd.Series(row_dict)
            text_payload = compose_text(row_series)

            for prompt_index in missing_indices:
                prompt_cfg = prompts_by_index.get(prompt_index)
                if not prompt_cfg:
                    continue

                prompt_template = (prompt_cfg.get('prompt') or '').strip()
                if not prompt_template:
                    self.logger.warning("[BATCH] Retry skipped – prompt template missing for index %s.", prompt_index)
                    continue

                expected_keys = prompt_cfg.get('expected_keys', []) or []
                prefix = prompt_cfg.get('prefix', '') or ''
                schema = None
                if expected_keys and not config.get('disable_schema', False):
                    try:
                        schema = build_dynamic_schema(expected_keys)
                    except Exception as exc:
                        self.logger.debug("[BATCH] Schema build failed for retry prompt %s: %s", prompt_index, exc)

                local_retry_config = dict(retry_model_config)
                previous_info = local_row_state['response_info'].get(prompt_key) or {}
                if previous_info.get('finish_reason') == 'length':
                    length_retry_cap = config.get(
                        'openai_batch_length_retry_tokens',
                        max(retry_max_tokens * 2, 4096)
                    )
                    local_retry_config['max_tokens'] = max(
                        local_retry_config.get('max_tokens', retry_max_tokens),
                        length_retry_cap
                    )
                try:
                    cleaned_json = self.analyze_text_with_model(
                        text=text_payload,
                        prompt=prompt_template,
                        model_config=local_retry_config,
                        schema=schema
                    )
                except Exception as exc:
                    self.logger.warning("[BATCH] Retry failed for %s prompt %s: %s", identifier_key, prompt_index, exc)
                    local_row_state['status'][str(prompt_index)] = 'retry_failed'
                    local_row_state['errors'].append(f"Sequential retry failed for prompt {prompt_index}: {exc}")
                    continue

                prompt_key = str(prompt_index)
                previous_status = local_row_state['status'].get(prompt_key)

                if cleaned_json:
                    local_row_state['raw'][prompt_key] = cleaned_json
                    local_row_state['cleaned'][prompt_key] = cleaned_json
                    local_row_state['status'][prompt_key] = 'success'
                    try:
                        parsed = json.loads(cleaned_json)
                        if prefix:
                            parsed = {f"{prefix}_{k}": v for k, v in parsed.items()}
                        local_row_state.setdefault('merged', {}).update(parsed)
                        self.last_annotation = parsed
                    except Exception as exc:
                        local_row_state['status'][prompt_key] = 'decode_error'
                        local_row_state['errors'].append(f"Decode error after retry (prompt {prompt_index}): {exc}")
                        status_counts['decode_error'] += 1
                        continue

                    if previous_status == 'parse_error':
                        status_counts['cleaning_failed'] = max(0, status_counts['cleaning_failed'] - 1)
                    elif previous_status == 'decode_error':
                        status_counts['decode_error'] = max(0, status_counts['decode_error'] - 1)
                    elif previous_status == 'error':
                        status_counts['error'] = max(0, status_counts['error'] - 1)

                    status_counts['success'] += 1
                    local_row_state['usage'][prompt_key] = {'fallback': True}
                    info = local_row_state['response_info'].setdefault(prompt_key, {})
                    max_used_tokens = local_retry_config.get('max_tokens')
                    info['fallback'] = True
                    info['max_tokens'] = max_used_tokens
                    info['fallback_max_tokens'] = max_used_tokens
                    if local_row_state['errors']:
                        local_row_state['errors'] = [
                            err for err in local_row_state['errors']
                            if f"prompt {prompt_index}" not in err and "Unable to parse model response" not in err
                        ]
                else:
                    local_row_state['status'][prompt_key] = 'retry_failed'
                    local_row_state['errors'].append(f"Sequential retry returned empty response for prompt {prompt_index}")

        for row_index, row in data_subset.iterrows():
            identifier_value = row[identifier_column]
            identifier_key = str(identifier_value)
            text_payload = compose_text(row)

            for prompt_idx, prompt_cfg in enumerate(prompts, 1):
                prompt_payload = prompt_cfg.get('prompt')
                expected_keys = prompt_cfg.get('expected_keys', [])
                prefix = prompt_cfg.get('prefix', '')
                prompt_name = prompt_cfg.get('name')

                prompt_template = ""
                if isinstance(prompt_payload, dict):
                    prompt_template = (
                        prompt_payload.get('content')
                        or prompt_payload.get('template')
                        or prompt_payload.get('prompt')
                        or ''
                    )
                    if not expected_keys:
                        keys_candidate = prompt_payload.get('keys')
                        if isinstance(keys_candidate, list):
                            expected_keys = keys_candidate
                    if not prompt_name:
                        prompt_name = prompt_payload.get('name')
                elif prompt_payload is not None:
                    prompt_template = str(prompt_payload)

                prompt_template = (prompt_template or '').strip()
                if not prompt_template:
                    self.logger.warning(
                        "[BATCH] Prompt %s is empty; skipping row %s.",
                        prompt_idx,
                        identifier_key
                    )
                    continue

                if not isinstance(expected_keys, list):
                    if expected_keys is None:
                        expected_keys = []
                    elif isinstance(expected_keys, (tuple, set)):
                        expected_keys = list(expected_keys)
                    else:
                        expected_keys = [expected_keys]

                prompt_display_name = prompt_name or f"prompt_{prompt_idx}"

                if text_payload:
                    full_prompt = f"{prompt_template}\n\nText to analyze:\n{text_payload}"
                else:
                    full_prompt = prompt_template

                sanitized_identifier = identifier_key.replace("\n", " ").replace("\r", " ")
                custom_id = f"{sanitized_identifier}|p{prompt_idx}|{row_index}"

                request_entries.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_request_body(full_prompt),
                })

                request_metadata[custom_id] = {
                    "identifier": identifier_value,
                    "identifier_key": identifier_key,
                    "prompt_index": prompt_idx,
                    "prefix": prefix,
                    "expected_keys": expected_keys,
                    "row_index": row_index,
                    "prompt_name": prompt_display_name
                }
                per_row_custom_ids[identifier_key].append(custom_id)

        with input_file_path.open('w', encoding='utf-8') as handle:
            for entry in request_entries:
                json.dump(entry, handle, ensure_ascii=False)
                handle.write('\n')

        client = OpenAI(api_key=api_key)
        with input_file_path.open('rb') as handle:
            uploaded_file = client.files.create(file=handle, purpose='batch')

        completion_window = config.get('openai_batch_completion_window', '24h')
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window
        )
        self.logger.info("[BATCH] Submitted job %s (status: %s).", batch_job.id, batch_job.status)

        poll_interval = max(2, int(config.get('openai_batch_poll_interval', 5)))
        ongoing_statuses = {'validating', 'queued', 'in_progress', 'processing', 'finalizing'}

        def _as_request_counts(mapping: Any) -> Dict[str, int]:
            if mapping is None:
                return {}
            if isinstance(mapping, dict):
                return mapping
            if hasattr(mapping, "model_dump"):
                try:
                    return mapping.model_dump()
                except Exception:
                    pass
            extracted: Dict[str, int] = {}
            for attr in ("completed", "succeeded", "failed", "expired", "processing", "total"):
                if hasattr(mapping, attr):
                    value = getattr(mapping, attr)
                    if isinstance(value, int):
                        extracted[attr] = value
            return extracted
        while batch_job.status in ongoing_statuses:
            request_counts = _as_request_counts(getattr(batch_job, 'request_counts', None))
            completed = request_counts.get('completed') or request_counts.get('succeeded') or 0
            total = request_counts.get('total', total_requests)
            status_message = f"OpenAI batch status: {batch_job.status} ({completed}/{total} requests completed)"
            if self.progress_callback:
                self.progress_callback(min(completed, total_requests), total_requests, status_message)
            self.logger.debug("[BATCH] %s", status_message)
            time.sleep(poll_interval)
            batch_job = client.batches.retrieve(batch_job.id)

        if self.progress_callback:
            self.progress_callback(total_requests, total_requests, f"OpenAI batch status: {batch_job.status}")

        if batch_job.status != 'completed':
            error_message = f"OpenAI batch job {batch_job.id} finished with status '{batch_job.status}'"
            self.logger.error("[BATCH] %s", error_message)
            if getattr(batch_job, 'error_file_id', None):
                error_response = client.files.content(batch_job.error_file_id)
                error_path = batch_dir / f"openai_batch_errors_{timestamp}.jsonl"
                if hasattr(error_response, 'content'):
                    error_path.write_bytes(error_response.content)
                else:
                    error_path.write_bytes(error_response.read())
                self.logger.error("[BATCH] Error details saved to %s", error_path)
            raise RuntimeError(error_message)

        job_created_at = getattr(batch_job, 'created_at', None) or getattr(batch_job, 'created', None)
        job_started_at = getattr(batch_job, 'started_at', None)
        job_completed_at = getattr(batch_job, 'completed_at', None) or getattr(batch_job, 'finished_at', None)
        final_request_counts = _as_request_counts(getattr(batch_job, 'request_counts', None))

        def _extract_from_collection(value: Any) -> Optional[str]:
            if not value:
                return None
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ('id', 'file_id', 'output_file_id'):
                    candidate = value.get(key)
                    if candidate:
                        return candidate
                for key in ('output_files', 'files', 'data', 'items'):
                    if key in value:
                        nested = value.get(key)
                        result = _extract_from_collection(nested)
                        if result:
                            return result
                return None
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    result = _extract_from_collection(item)
                    if result:
                        return result
            return None

        def _extract_output_file_id(job: Any) -> Optional[str]:
            if job is None:
                return None
            if hasattr(job, 'output_file_id') and getattr(job, 'output_file_id'):
                return getattr(job, 'output_file_id')
            if hasattr(job, 'output_file') and getattr(job, 'output_file'):
                candidate = getattr(job, 'output_file')
                if isinstance(candidate, str):
                    return candidate
                if isinstance(candidate, dict):
                    return candidate.get('id') or candidate.get('file_id')

            candidates_attrs = ('output_file_ids', 'output_files', 'outputs')
            for attr in candidates_attrs:
                value = getattr(job, attr, None)
                result = _extract_from_collection(value)
                if result:
                    return result

            if hasattr(job, 'model_dump'):
                try:
                    dumped = job.model_dump()
                    direct = dumped.get('output_file_id') or dumped.get('output_file')
                    if direct:
                        return direct
                    return _extract_from_collection(dumped)
                except Exception:
                    pass
            if isinstance(job, dict):
                direct = (
                    job.get('output_file_id')
                    or job.get('output_file')
                    or job.get('file')
                )
                if direct:
                    return direct
                return _extract_from_collection(job)
            return None

        output_file_id = _extract_output_file_id(batch_job)
        file_manifest: List[Dict[str, Any]] = []
        if not output_file_id:
            check_attempts = int(config.get('openai_batch_output_poll_attempts', 10))
            wait_seconds = float(config.get('openai_batch_output_poll_interval', 3))
            for attempt in range(check_attempts):
                time.sleep(wait_seconds)
                refreshed = client.batches.retrieve(batch_job.id)
                output_file_id = _extract_output_file_id(refreshed)
                if output_file_id:
                    batch_job = refreshed
                    break
                self.logger.debug(
                    "[BATCH] Poll attempt %s/%s still missing output file for job %s (status=%s).",
                    attempt + 1,
                    check_attempts,
                    batch_job.id,
                    getattr(refreshed, 'status', None)
                )
            if not output_file_id:
                files_response = None
                try:
                    files_response = client.files.list(limit=100)
                except Exception as exc:
                    self.logger.warning("[BATCH] Unable to list files for batch %s: %s", batch_job.id, exc)
                if files_response and getattr(files_response, 'data', None):
                    for file_obj in files_response.data:
                        file_manifest.append({
                            'id': getattr(file_obj, 'id', None),
                            'filename': getattr(file_obj, 'filename', None),
                            'purpose': getattr(file_obj, 'purpose', None),
                            'bytes': getattr(file_obj, 'bytes', None),
                            'created_at': getattr(file_obj, 'created_at', None),
                            'metadata': getattr(file_obj, 'metadata', None),
                        })
                    for file_obj in files_response.data:
                        meta = getattr(file_obj, 'metadata', {}) or {}
                        if isinstance(meta, dict) and meta.get('batch_id') == batch_job.id:
                            output_file_id = getattr(file_obj, 'id', None)
                            if output_file_id:
                                self.logger.info(
                                    "[BATCH] Recovered output file %s for job %s via files.list.",
                                    output_file_id,
                                    batch_job.id
                                )
                                break
                if not output_file_id:
                    dumped = None
                    try:
                        dumped = batch_job.model_dump() if hasattr(batch_job, 'model_dump') else batch_job
                    except Exception:
                        dumped = str(batch_job)
                    try:
                        refreshed_dump = refreshed.model_dump() if hasattr(refreshed, 'model_dump') else refreshed
                    except Exception:
                        refreshed_dump = str(refreshed)
                    self.logger.error("[BATCH] Final job state: %s", dumped)
                    self.logger.error("[BATCH] Last refreshed job state: %s", refreshed_dump)
                    if file_manifest:
                        self.logger.error("[BATCH] File manifest (limit 100): %s", file_manifest)
                    raise RuntimeError(
                        "OpenAI batch completed but no output_file_id was returned. "
                        f"Check OpenAI dashboard for job {getattr(batch_job, 'id', '<unknown>')}."
                    )

        known_prefixes = ("file_", "batch_", "upload_")
        if not any(str(output_file_id).startswith(prefix) for prefix in known_prefixes):
            output_file_id = str(output_file_id).split("/")[-1]

        try:
            output_response = client.files.content(output_file_id)
        except NotFoundError as exc:
            if not file_manifest:
                try:
                    files_response = client.files.list(limit=100)
                    if files_response and getattr(files_response, 'data', None):
                        for file_obj in files_response.data:
                            file_manifest.append({
                                'id': getattr(file_obj, 'id', None),
                                'filename': getattr(file_obj, 'filename', None),
                                'purpose': getattr(file_obj, 'purpose', None),
                                'bytes': getattr(file_obj, 'bytes', None),
                                'created_at': getattr(file_obj, 'created_at', None),
                                'metadata': getattr(file_obj, 'metadata', None),
                            })
                except Exception:
                    pass
            if file_manifest:
                self.logger.error("[BATCH] File manifest for job %s: %s", batch_job.id, file_manifest)
            raise RuntimeError(
                f"OpenAI returned 404 for output file {output_file_id}. "
                f"Verify batch {getattr(batch_job, 'id', '<unknown>')} in the OpenAI dashboard."
            ) from exc

        if hasattr(output_response, 'content'):
            output_file_path.write_bytes(output_response.content)
        else:
            output_file_path.write_bytes(output_response.read())
        self.logger.info("[BATCH] Output saved to %s", output_file_path)

        row_results: Dict[str, Dict[str, Any]] = {}

        with output_file_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.error("[BATCH] Unable to parse output line: %s", line)
                    continue

                custom_id = payload.get('custom_id')
                meta = request_metadata.get(custom_id)
                if not meta:
                    self.logger.warning("[BATCH] Received result for unknown request %s", custom_id)
                    continue

                identifier_key = meta['identifier_key']
                prompt_idx = meta['prompt_index']
                prompt_key = str(prompt_idx)
                expected_keys = meta['expected_keys']
                prefix = meta['prefix']

                row_state = row_results.setdefault(identifier_key, {
                    'identifier': meta['identifier'],
                    'row_index': meta['row_index'],
                    'raw': {},
                    'cleaned': {},
                    'status': {},
                    'merged': {},
                    'errors': [],
                    'usage': {},
                    'response_info': {}
                })

                error_info = payload.get('error')
                if error_info:
                    error_text = error_info.get('message') or str(error_info)
                    row_state['raw'][prompt_key] = None
                    row_state['cleaned'][prompt_key] = None
                    row_state['status'][prompt_key] = 'error'
                    row_state['errors'].append(error_text)
                    status_counts['error'] += 1
                    continue

                response_wrapper = payload.get('response') or {}
                response_body = response_wrapper.get('body', {})
                choices = response_body.get('choices', [])
                message_content = None
                finish_reason = None
                if choices:
                    choice_payload = choices[0] or {}
                    finish_reason = choice_payload.get('finish_reason')
                    message_payload = choice_payload.get('message') or {}
                    message_content = message_payload.get('content')
                usage_payload = response_body.get('usage') or {}
                row_state['usage'][prompt_key] = usage_payload
                row_state['response_info'][prompt_key] = {
                    'id': response_body.get('id'),
                    'created': response_body.get('created'),
                    'request_id': response_wrapper.get('request_id'),
                    'status_code': response_wrapper.get('status_code'),
                    'finish_reason': finish_reason,
                }
                if isinstance(message_content, list):
                    fragments: List[str] = []
                    for part in message_content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            fragments.append(part.get('text', ''))
                        elif isinstance(part, str):
                            fragments.append(part)
                    message_content = "".join(fragments)
                raw_text = message_content.strip() if isinstance(message_content, str) else None
                row_state['raw'][prompt_key] = raw_text

                cleaned_json = None
                if not raw_text:
                    status_label = 'length_truncated' if finish_reason == 'length' else 'empty_response'
                    row_state['cleaned'][prompt_key] = None
                    row_state['status'][prompt_key] = status_label
                    row_state['errors'].append(
                        f"Model returned empty content (finish_reason={finish_reason or 'unknown'})."
                    )
                    status_counts['cleaning_failed'] += 1
                    continue

                try:
                    cleaned_json = clean_json_output(raw_text, expected_keys)
                except Exception as exc:
                    self.logger.error("[BATCH] Cleaning failed for %s: %s", custom_id, exc)
                    cleaned_json = None

                if not cleaned_json:
                    row_state['cleaned'][prompt_key] = None
                    row_state['status'][prompt_key] = 'parse_error'
                    row_state['errors'].append("Unable to parse model response.")
                    status_counts['cleaning_failed'] += 1
                    continue

                row_state['cleaned'][prompt_key] = cleaned_json
                row_state['status'][prompt_key] = 'success'
                try:
                    parsed = json.loads(cleaned_json)
                    if prefix:
                        parsed = {f"{prefix}_{k}": v for k, v in parsed.items()}
                    row_state['merged'].update(parsed)
                    self.last_annotation = parsed
                    status_counts['success'] += 1
                except Exception as exc:
                    row_state['status'][prompt_key] = 'decode_error'
                    row_state['errors'].append(f"Decode error: {exc}")
                    status_counts['decode_error'] += 1

        batch_elapsed = time.perf_counter() - start_time
        job_elapsed_seconds = None
        if isinstance(job_started_at, (int, float)) and isinstance(job_completed_at, (int, float)):
            job_elapsed_seconds = job_completed_at - job_started_at
        elif isinstance(job_created_at, (int, float)) and isinstance(job_completed_at, (int, float)):
            job_elapsed_seconds = job_completed_at - job_created_at

        if isinstance(job_elapsed_seconds, (int, float)) and job_elapsed_seconds >= 0:
            per_row_time = job_elapsed_seconds / max(total_rows, 1)
        else:
            job_elapsed_seconds = None
            per_row_time = batch_elapsed / max(total_rows, 1)

        job_metadata = {
            'job_id': getattr(batch_job, 'id', None),
            'status': getattr(batch_job, 'status', None),
            'created_at': job_created_at,
            'started_at': job_started_at,
            'completed_at': job_completed_at,
            'request_counts': final_request_counts,
            'total_requests': total_requests,
            'prompt_count': prompt_count,
            'dataset_rows': total_rows,
            'batch_elapsed_seconds': batch_elapsed,
            'job_elapsed_seconds': job_elapsed_seconds,
            'input_file': str(input_file_path),
            'output_file': str(output_file_path),
            'model': model_name,
            'provider': config.get('provider') or config.get('annotation_provider'),
        }
        try:
            metadata_file_path.write_text(json.dumps(job_metadata, ensure_ascii=False, indent=2))
        except Exception as exc:
            self.logger.warning("[BATCH] Unable to write batch metadata: %s", exc)

        config['openai_batch_metadata_path'] = str(metadata_file_path)
        config['openai_batch_input_path'] = str(input_file_path)
        config['openai_batch_output_path'] = str(output_file_path)
        config['openai_batch_job_id'] = job_metadata.get('job_id')
        config['openai_batch_status'] = job_metadata.get('status')
        config['openai_batch_job_elapsed_seconds'] = job_elapsed_seconds
        config['openai_batch_total_requests'] = total_requests

        archive_dir = config.get('openai_batch_archive_dir')
        if archive_dir:
            try:
                archive_path = Path(archive_dir)
                archive_path.mkdir(parents=True, exist_ok=True)
                for src_path in (input_file_path, output_file_path, metadata_file_path):
                    if src_path.exists():
                        shutil.copy2(src_path, archive_path / src_path.name)
            except Exception as exc:
                self.logger.warning("[BATCH] Unable to archive batch artifacts to %s: %s", archive_dir, exc)

        # Sequential fallback for prompts that did not return usable results
        fallback_candidates: List[Tuple[str, List[int]]] = []
        for identifier_key, custom_ids in per_row_custom_ids.items():
            missing_indices: List[int] = []
            row_state = row_results.get(identifier_key)
            if not row_state:
                for custom_id in custom_ids:
                    meta = request_metadata.get(custom_id)
                    if meta:
                        missing_indices.append(int(meta['prompt_index']))
            else:
                for custom_id in custom_ids:
                    meta = request_metadata.get(custom_id)
                    if not meta:
                        continue
                    prompt_index = int(meta['prompt_index'])
                    status = row_state['status'].get(str(prompt_index))
                    if status != 'success':
                        missing_indices.append(prompt_index)
            if missing_indices:
                fallback_candidates.append((identifier_key, missing_indices))

        if fallback_candidates:
            self.logger.info(
                "[BATCH] Retrying %s row(s) sequentially due to incomplete batch responses.",
                len(fallback_candidates)
            )
            for identifier_key, missing_indices in fallback_candidates:
                retry_failed_prompts(
                    identifier_key=identifier_key,
                    missing_indices=missing_indices,
                    row_state=row_results.get(identifier_key)
                )

        cleanup_enabled = config.get('openai_batch_cleanup_missing', True)
        if cleanup_enabled:
            residual_candidates: List[Tuple[str, List[int]]] = []
            for identifier_key, row_state in row_results.items():
                if not row_state:
                    continue
                missing_indices: List[int] = []
                for prompt_idx, expected_keys in prefixed_expected_keys.items():
                    if not expected_keys:
                        continue
                    status = row_state['status'].get(str(prompt_idx))
                    merged_payload = row_state.get('merged', {})
                    needs_retry = any(
                        merged_payload.get(key) in (None, '', []) for key in expected_keys
                    )
                    if status == 'success' and not needs_retry:
                        continue
                    if needs_retry:
                        missing_indices.append(prompt_idx)
                if missing_indices:
                    residual_candidates.append((identifier_key, missing_indices))
            if residual_candidates:
                self.logger.info(
                    "[BATCH] Performing cleanup retries for %s row(s) with missing prompt keys.",
                    len(residual_candidates)
                )
                for identifier_key, missing_indices in residual_candidates:
                    retry_failed_prompts(
                        identifier_key=identifier_key,
                        missing_indices=missing_indices,
                        row_state=row_results.get(identifier_key)
                    )

        completed_rows = 0
        for identifier_key, custom_ids in per_row_custom_ids.items():
            meta = request_metadata[custom_ids[0]]
            identifier_value = meta['identifier']
            row_state = row_results.get(identifier_key)
            if not row_state:
                row_state = {
                    'identifier': identifier_value,
                    'row_index': meta['row_index'],
                    'raw': {},
                    'cleaned': {},
                    'status': {},
                    'merged': {},
                    'errors': ["No response returned by OpenAI."],
                    'usage': {},
                    'response_info': {}
                }

            for custom_id in custom_ids:
                prompt_key = str(request_metadata[custom_id]['prompt_index'])
                if prompt_key not in row_state['status']:
                    row_state['raw'][prompt_key] = None
                    row_state['cleaned'][prompt_key] = None
                    row_state['status'][prompt_key] = 'missing'
                    row_state['errors'].append("No response returned by OpenAI.")
                    row_state['usage'][prompt_key] = {}
                    row_state['response_info'][prompt_key] = {}

            merged_payload = row_state.get('merged', {})
            final_payload: Dict[str, Any] = {}
            if all_expected_keys:
                for key in all_expected_keys:
                    final_payload[key] = merged_payload.get(key)
            for key, value in merged_payload.items():
                if key not in final_payload:
                    final_payload[key] = value
            row_state['merged'] = final_payload
            final_json = json.dumps(final_payload, ensure_ascii=False) if final_payload else None
            usage_totals = aggregate_usage(row_state.get('usage', {}).values())

            mask = full_data[identifier_column] == identifier_value
            if mask.any():
                full_data.loc[mask, annotation_column] = final_json
                full_data.loc[mask, f"{annotation_column}_inference_time"] = per_row_time

            if final_json:
                try:
                    self.last_annotation = json.loads(final_json)
                except Exception:
                    pass

            if log_enabled and log_path:
                self._write_log_entry(
                    log_path,
                    {
                        'id': identifier_value,
                        'final_json': final_json,
                        'inference_time': per_row_time,
                        'usage_summary': usage_totals,
                    }
                )

            completed_rows += 1
            if self.progress_callback:
                self.progress_callback(
                    completed_rows,
                    total_rows,
                    f"Annotated {completed_rows}/{total_rows} items via OpenAI batch"
                )

        if save_incrementally and output_path:
            self.logger.info("[BATCH] Batch mode overrides incremental saves; writing final dataset once.")
        if output_path:
            self._save_data(full_data, output_path, output_format)

        self.logger.info(
            "[BATCH] Completed OpenAI batch annotation in %.2fs (rows=%s, prompts=%s)",
            batch_elapsed,
            total_rows,
            prompt_count
        )

        return full_data

    def _save_results(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Save annotation results"""
        output_path = config.get('output_path')
        if output_path:
            format = config.get('output_format') or 'csv'
            self._save_data(data, output_path, format)
            self.logger.info(f"Results saved to {output_path}")
            subset_path = self._save_annotated_subset(data, config)

        doccano_path = None
        if config.get('export_to_doccano'):
            annotation_col = config.get('annotation_column', 'annotation')
            text_cols = config.get('text_columns') or []
            text_col = config.get('text_column')
            if not text_col and text_cols:
                text_col = text_cols[0]
            annotated_only = data
            if annotation_col in data.columns:
                annotated_only = data[data[annotation_col].notna()]
            try:
                doccano_path = self._export_to_doccano(
                    annotated_df=annotated_only,
                    annotation_column=annotation_col,
                    text_column=text_col or annotation_col,
                    config=config
                )
            except Exception as exc:
                self.logger.warning("[DOCCANO] Export failed: %s", exc)
            else:
                if doccano_path:
                    config['doccano_export_path'] = str(doccano_path)
            if subset_path:
                self.logger.info(f"Annotated subset saved to {subset_path}")

            export_prefs = config.get('export_preferences', {})
            export_doccano = config.get('export_to_doccano') or export_prefs.get('export_to_doccano')
            if export_doccano:
                annotation_column = config.get('annotation_column', 'annotation')
                text_column = config.get('text_column')
                if not text_column:
                    text_cols = config.get('text_columns') or []
                    if text_cols:
                        text_column = text_cols[0]
                if annotation_column in data.columns and text_column in data.columns:
                    mask = (
                        data[annotation_column].notna()
                        & data[annotation_column].astype(str).str.strip().ne('')
                        & data[annotation_column].astype(str).str.lower().ne('nan')
                    )
                    annotated_df = data.loc[mask].copy()
                    if not annotated_df.empty:
                        try:
                            doccano_path = self._export_doccano_jsonl(
                                annotated_df=annotated_df,
                                annotation_column=annotation_column,
                                text_column=text_column,
                                config=config
                            )
                            if doccano_path:
                                self.logger.info(f"Doccano export saved to {doccano_path}")
                                config['doccano_export_path'] = str(doccano_path)
                        except Exception as exc:
                            self.logger.warning(f"Doccano export failed: {exc}")

    def _save_data(self, data: pd.DataFrame, path: str, format: str):
        """Save data to file"""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        fmt = (format or 'csv').lower()
        if fmt == 'csv':
            data.to_csv(path_obj, index=False)
        elif fmt == 'excel':
            data.to_excel(path_obj, index=False)
        elif fmt == 'parquet':
            data.to_parquet(path_obj, index=False)
        elif fmt == 'json':
            data.to_json(path_obj, orient='records', lines=False, force_ascii=False, indent=2)
        elif fmt == 'jsonl':
            data.to_json(path_obj, orient='records', lines=True, force_ascii=False)
        elif fmt in ['rdata', 'rds']:
            if HAS_PYREADR:
                if fmt == 'rdata':
                    pyreadr.write_rdata({'data': data}, str(path_obj))
                else:
                    pyreadr.write_rds(data, str(path_obj))
            else:
                raise ImportError("pyreadr required for RData/RDS files")
        else:
            raise ValueError(f"Unsupported output format: {format}")

    def _save_annotated_subset(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Path]:
        """Persist a lightweight file containing only annotated rows."""
        if not config.get('create_annotated_subset', True):
            return None

        annotation_column = config.get('annotation_column', 'annotation')
        if annotation_column not in data.columns:
            return None

        mask = (
            data[annotation_column].notna()
            & data[annotation_column].astype(str).str.strip().ne('')
            & data[annotation_column].astype(str).str.lower().ne('nan')
        )
        subset = data.loc[mask].copy()
        if subset.empty:
            return None

        # Drop verbose per-prompt columns from the compact export
        for suffix in PROMPT_SUFFIXES:
            col_name = f"{annotation_column}_{suffix}"
            if col_name in subset.columns:
                subset.drop(columns=[col_name], inplace=True)

        output_path = config.get('output_path')
        output_format = config.get('output_format', 'csv')
        if not output_path:
            return None

        output_path_obj = Path(output_path)
        subset_path = output_path_obj.with_name(f"{output_path_obj.stem}_annotated_only{output_path_obj.suffix}")

        self._save_data(subset, subset_path, output_format)
        config['annotated_subset_path'] = str(subset_path)
        self.logger.info(f"Saved annotated-only subset to {subset_path}")
        return subset_path

    def _export_doccano_jsonl(
        self,
        *,
        annotated_df: pd.DataFrame,
        annotation_column: str,
        text_column: str,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """Export annotated samples to a Doccano-compatible JSONL file."""
        if annotated_df.empty:
            return None

        session_dirs = config.get('session_dirs') or {}
        default_dir = self.settings.paths.logs_dir / "doccano_exports"
        doccano_base = Path(session_dirs.get('doccano', default_dir))

        provider_folder = (
            config.get('provider_folder')
            or config.get('annotation_provider')
            or config.get('provider')
            or 'provider'
        )

        model_reference = (
            config.get('model_folder')
            or config.get('annotation_model')
            or config.get('model')
        )
        if isinstance(model_reference, dict):
            model_reference = (
                model_reference.get('model_name')
                or model_reference.get('name')
            )
        model_folder = str(model_reference or 'model').replace('/', '_')

        dataset_name = config.get('dataset_name') or Path(config.get('file_path', 'dataset')).stem

        doccano_dir = doccano_base / provider_folder / model_folder / dataset_name
        doccano_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doccano_path = doccano_dir / f"{dataset_name}_doccano_{timestamp}.jsonl"

        label_keys: Set[str] = set()
        for prompt_cfg in config.get('prompts', []):
            expected_keys = prompt_cfg.get('expected_keys') or []
            prefix = prompt_cfg.get('prefix', '') or ''
            for key in expected_keys:
                label_keys.add(f"{prefix}_{key}" if prefix else key)

        total_written = 0
        with doccano_path.open('w', encoding='utf-8') as handle:
            for _, row in annotated_df.iterrows():
                annotation_str = row.get(annotation_column)
                if not annotation_str or not str(annotation_str).strip():
                    continue

                try:
                    annotation_data = json.loads(annotation_str)
                except Exception:
                    self.logger.debug("[DOCCANO] Skipping row with invalid annotation JSON.")
                    continue

                entry_labels: List[str] = []
                if label_keys:
                    for key in label_keys:
                        if key not in annotation_data:
                            continue
                        value = annotation_data.get(key)
                        if isinstance(value, list):
                            entry_labels.extend(str(v) for v in value if str(v).strip())
                        elif value is not None and str(value).strip():
                            entry_labels.append(str(value))
                else:
                    for value in annotation_data.values():
                        if isinstance(value, list):
                            entry_labels.extend(str(v) for v in value if str(v).strip())
                        elif value is not None and str(value).strip():
                            entry_labels.append(str(value))

                meta: Dict[str, Any] = {}
                for column in annotated_df.columns:
                    if column in (text_column, annotation_column):
                        continue
                    meta[column] = self._serialize_meta_value(row[column])
                meta['annotation_json'] = annotation_data
                cleaned_meta = {k: v for k, v in meta.items() if v not in (None, '')}

                entry: Dict[str, Any] = {'text': str(row[text_column])}
                if entry_labels:
                    entry['labels'] = entry_labels
                if cleaned_meta:
                    entry['meta'] = cleaned_meta

                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
                total_written += 1

        if total_written == 0:
            try:
                doccano_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.logger.warning("[DOCCANO] Export produced zero entries; no file generated.")
            return None

        return doccano_path

    @staticmethod
    def _serialize_meta_value(value: Any) -> Optional[Any]:
        """Ensure metadata values are JSON-serializable."""
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (datetime, pd.Timestamp)):
            return str(value)
        if isinstance(value, (str, bool, list, dict)):
            return value
        return str(value)

    def _store_annotation_payload(
        self,
        dataframe: pd.DataFrame,
        mask: pd.Series,
        payload: Optional[Dict[str, Any]],
        annotation_column: str
    ):
        """No-op placeholder retained for compatibility with legacy workflows."""
        return

    def _append_to_csv(self, data: pd.DataFrame, identifier, identifier_column: str,
                      annotation_column: str, path: str):
        """Append single row to CSV for incremental saving"""
        row_df = data[data[identifier_column] == identifier].copy()
        
        # Remove per-prompt columns for cleaner output
        for suffix in PROMPT_SUFFIXES:
            col = f"{annotation_column}_{suffix}"
            if col in row_df.columns:
                row_df = row_df.drop(columns=[col])

        header_needed = not os.path.exists(path)
        row_df.to_csv(path, mode='a', index=False, header=header_needed)

    def _write_log_entry(self, log_path: str, entry: Dict[str, Any]):
        """Write log entry for annotation"""
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _generate_summary(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate annotation summary statistics"""
        total = len(data)
        success = status_counts.get('success', 0)
        errors = status_counts.get('error', 0)
        annotation_column = config.get('annotation_column', 'annotation')

        annotated_rows = 0
        if annotation_column in data.columns:
            annotated_rows = data[annotation_column].dropna().shape[0]

        usage_columns = {
            'prompt_tokens': f"{annotation_column}_prompt_tokens",
            'completion_tokens': f"{annotation_column}_completion_tokens",
            'total_tokens': f"{annotation_column}_total_tokens",
            'cached_prompt_tokens': f"{annotation_column}_cached_prompt_tokens",
            'reasoning_tokens': f"{annotation_column}_reasoning_tokens",
        }
        usage_summary: Dict[str, float] = {}
        for key, column in usage_columns.items():
            if column in data.columns:
                try:
                    usage_summary[key] = float(pd.to_numeric(data[column], errors='coerce').fillna(0).sum())
                except Exception:
                    usage_summary[key] = 0.0

        summary: Dict[str, Any] = {
            'total_processed': total,
            'successful': success,
            'errors': errors,
            'success_rate': (success / total * 100) if total > 0 else 0,
            'annotated_rows': int(annotated_rows),
            'total_annotated': int(annotated_rows),
            'success_count': success,
            'error_count': errors,
            'annotation_column': annotation_column,
            'output_file': config.get('output_path'),
            'output_format': config.get('output_format', 'csv'),
            'model': config.get('model'),
            'provider': config.get('provider'),
            'annotation_sample_size': config.get('annotation_sample_size') or config.get('annotation_limit'),
            'usage_summary': usage_summary,
            'status_breakdown': dict(status_counts),
            'timestamp': datetime.now().isoformat()
        }

        inference_column = f"{annotation_column}_inference_time"
        if inference_column in data.columns:
            try:
                inference_series = pd.to_numeric(data[inference_column], errors='coerce').dropna()
            except Exception:
                inference_series = pd.Series(dtype=float)
            if not inference_series.empty:
                summary['inference_time_column'] = inference_column
                summary['mean_inference_time'] = float(inference_series.mean())
                summary['median_inference_time'] = float(inference_series.median())
                summary['total_inference_time'] = float(inference_series.sum())

        subset_path = config.get('annotated_subset_path')
        if subset_path:
            summary['annotated_subset_path'] = subset_path

        batch_keys = (
            'openai_batch_dir',
            'openai_batch_metadata_path',
            'openai_batch_input_path',
            'openai_batch_output_path',
            'openai_batch_job_id',
            'openai_batch_status',
            'openai_batch_job_elapsed_seconds',
            'openai_batch_total_requests',
        )
        for key in batch_keys:
            value = config.get(key)
            if value is not None:
                summary[key] = value

        preview_samples: List[Dict[str, Any]] = []
        identifier_column = config.get('identifier_column')
        if identifier_column and identifier_column in data.columns and annotation_column in data.columns:
            annotated_df = data[
                data[annotation_column].notna()
                & data[annotation_column].astype(str).str.strip().ne('')
                & data[annotation_column].astype(str).str.lower().ne('nan')
            ]
            if not annotated_df.empty:
                preview_cols = [identifier_column, annotation_column]
                if inference_column in annotated_df.columns:
                    preview_cols.append(inference_column)

                sample_size = int(config.get('preview_sample_size', 1))
                sample_size = max(1, sample_size)
                actual_size = min(sample_size, len(annotated_df))
                sampled = annotated_df[preview_cols].sample(
                    n=actual_size,
                    random_state=random.randint(0, 10**6)
                )
                max_chars = int(config.get('preview_max_chars', 320))

                for _, row in sampled.iterrows():
                    preview_entry: Dict[str, Any] = {}
                    for col in preview_cols:
                        value = row[col]
                        if isinstance(value, pd.Timestamp):
                            preview_entry[col] = value.isoformat()
                        elif pd.isna(value):
                            preview_entry[col] = None
                        else:
                            preview_entry[col] = value

                    annotation_payload = preview_entry.get(annotation_column)
                    preview_text = None
                    if isinstance(annotation_payload, str):
                        try:
                            annotation_json = json.loads(annotation_payload)
                        except Exception:
                            annotation_json = None
                        if isinstance(annotation_json, dict):
                            fragments: List[str] = []
                            for key in sorted(annotation_json.keys()):
                                value = annotation_json[key]
                                if value in (None, '', [], {}):
                                    continue
                                if isinstance(value, list):
                                    value_repr = ", ".join(str(item) for item in value if str(item).strip())
                                elif isinstance(value, dict):
                                    value_repr = json.dumps(value, ensure_ascii=False)
                                else:
                                    value_repr = str(value)
                                if value_repr:
                                    fragments.append(f"{key}: {value_repr}")
                            preview_text = " | ".join(fragments) if fragments else annotation_payload
                        else:
                            preview_text = annotation_payload
                    if preview_text:
                        preview_entry['annotation_preview'] = preview_text[:max_chars]
                    preview_samples.append(preview_entry)

        summary['preview_samples'] = preview_samples

        return summary

    async def annotate_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async wrapper for annotation (required by pipeline controller).

        Parameters
        ----------
        config : dict
            Annotation configuration

        Returns
        -------
        dict
            Annotation results
        """
        # Run the sync annotate method in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.annotate, config)

    def calculate_sample_size(self, total: int, confidence: float = 0.95,
                            margin_error: float = 0.05, proportion: float = 0.5) -> int:
        """
        Calculate sample size for given confidence interval.
        
        Parameters
        ----------
        total : int
            Total population size
        confidence : float
            Confidence level (default 0.95 for 95% CI)
        margin_error : float
            Margin of error (default 0.05 for 5%)
        proportion : float
            Expected proportion (default 0.5 for maximum variability)
        
        Returns
        -------
        int
            Required sample size
        """
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        # Sample size formula
        numerator = (z**2) * proportion * (1 - proportion)
        denominator = margin_error**2
        
        sample_size = (numerator / denominator) / (
            1 + ((numerator / denominator - 1) / total)
        )
        
        return min(math.ceil(sample_size), total)

    def analyze_text_with_model(
        self,
        text: str,
        prompt: str,
        model_config: Dict[str, Any],
        schema: Optional[BaseModel] = None
    ) -> Optional[str]:
        """
        Analyze text using configured model.

        Parameters
        ----------
        text : str
            Text to analyze
        prompt : str
            Prompt template
        model_config : dict
            Model configuration
        schema : BaseModel, optional
            Pydantic schema for validation

        Returns
        -------
        str or None
            JSON string response or None on failure
        """
        # Build full prompt
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"

        self.logger.debug(f"[ANALYZE] Calling model with prompt length: {len(full_prompt)}")
        self.logger.debug(f"[ANALYZE] Text to analyze (first 200 chars): {text[:200]}")

        # Call appropriate model
        provider = model_config.get('provider')

        if provider in ['openai', 'anthropic', 'google'] and self.api_client:
            self.logger.debug(f"[ANALYZE] Using API client for provider: {provider}")
            response = self.api_client.generate(
                prompt=full_prompt,
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 1000)
            )
        elif provider in ['ollama', 'llamacpp'] and self.local_client:
            self.logger.debug(f"[ANALYZE] Using local client for provider: {provider}")
            response = self.local_client.generate(
                prompt=full_prompt,
                options=model_config.get('options', {})
            )
        else:
            self.logger.error(f"No client configured for provider: {provider}")
            return None

        self.logger.debug(f"[ANALYZE] Raw response from model: {response}")

        if not response:
            warning_msg = "[ANALYZE] Model returned empty response"
            # Only show via progress manager if available, otherwise log
            if self.progress_manager:
                self.progress_manager.show_warning(warning_msg)
            else:
                self.logger.warning(warning_msg)
            return None

        # Clean and validate response
        # Extract expected keys from schema if available
        expected_keys = []
        if schema:
            # Use model_fields for Pydantic V2, fallback to __fields__ for V1
            if hasattr(schema, 'model_fields'):
                expected_keys = list(schema.model_fields.keys())
            elif hasattr(schema, '__fields__'):
                expected_keys = list(schema.__fields__.keys())

        self.logger.debug(f"[ANALYZE] Cleaning JSON with expected keys: {expected_keys}")
        cleaned = clean_json_output(response, expected_keys)
        self.logger.debug(f"[ANALYZE] Cleaned JSON: {cleaned}")
        
        # Validate with schema if provided
        if cleaned and schema:
            self.logger.debug("[ANALYZE] Attempting schema validation")
            try:
                validated = schema.model_validate_json(cleaned)
                final_result = validated.model_dump_json()
                self.logger.debug(f"[ANALYZE] Schema validated successfully: {final_result}")
                return final_result
            except Exception as e:
                self.logger.warning(f"[ANALYZE] Schema validation failed: {e}, returning cleaned JSON")
                return cleaned

        self.logger.debug(f"[ANALYZE] Returning final result: {cleaned}")
        return cleaned


def process_single_prompt(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single prompt for annotation.

    Parameters
    ----------
    task : dict
        Task configuration including row data, prompt, and options

    Returns
    -------
    dict
        Annotation result with identifier, JSON, and timing
    """
    # Setup logging for this function
    logger = logging.getLogger(__name__)

    start_time = time.perf_counter()

    # Extract task parameters
    row = task['row']
    prompt_config = task['prompts'][0]  # Single prompt
    text_columns = task['text_columns']
    identifier = task['identifier']
    model_config = task['model_config']
    options = task.get('options', {})

    logger.debug(f"[PROCESS] Starting annotation for identifier: {identifier}")

    # Build text from columns
    text_parts = []
    for col in text_columns:
        if pd.notna(row[col]):
            text_parts.append(str(row[col]))
    text = "\n\n".join(text_parts)

    logger.debug(f"[PROCESS] Built text from {len(text_columns)} columns, length: {len(text)}")

    # Get prompt details (handle both 'prompt' and 'template' keys)
    prompt_text = prompt_config.get('prompt') or prompt_config.get('template', '')
    expected_keys = prompt_config.get('expected_keys', [])
    prefix = prompt_config.get('prefix', '')

    logger.debug(f"[PROCESS] Prompt length: {len(prompt_text)}, expected keys: {expected_keys}, prefix: {prefix}")

    # Build schema if expected keys provided
    schema = None
    if expected_keys and not options.get('disable_schema', False):
        schema = build_dynamic_schema(expected_keys)
        logger.debug(f"[PROCESS] Built dynamic schema for keys: {expected_keys}")

    # Create annotator instance for this process
    annotator = LLMAnnotator()

    # Setup the model client for this process
    config_for_setup = {
        'model': model_config.get('model_name'),
        'provider': model_config.get('provider', 'ollama'),
        'api_key': model_config.get('api_key')
    }
    logger.debug(f"[PROCESS] Setting up model client: {config_for_setup}")
    annotator._setup_model_client(config_for_setup)

    # Analyze text
    logger.debug(f"[PROCESS] Calling analyze_text_with_model")
    result = annotator.analyze_text_with_model(
        text=text,
        prompt=prompt_text,
        model_config=model_config,
        schema=schema
    )

    logger.debug(f"[PROCESS] Got result from analyze_text_with_model: {result}")

    # Apply prefix if specified
    if result and prefix:
        logger.debug(f"[PROCESS] Applying prefix '{prefix}' to result")
        try:
            parsed = json.loads(result)
            prefixed = {f"{prefix}_{k}": v for k, v in parsed.items()}
            result = json.dumps(prefixed, ensure_ascii=False)
            logger.debug(f"[PROCESS] Prefixed result: {result}")
        except Exception as e:
            logger.warning(f"[PROCESS] Failed to apply prefix: {e}")

    elapsed = time.perf_counter() - start_time
    logger.debug(f"[PROCESS] Completed annotation for {identifier} in {elapsed:.2f}s, result: {result}")
    
    return {
        'identifier': identifier,
        'final_json': result,
        'inference_time': elapsed,
        'status': 'success' if result else 'error'
    }


def process_multiple_prompts(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple prompts and merge results.
    
    Parameters
    ----------
    task : dict
        Task configuration with multiple prompts
    
    Returns
    -------
    dict
        Merged annotation results
    """
    start_time = time.perf_counter()
    
    # Extract task parameters
    row = task['row']
    prompts = task['prompts']
    text_columns = task['text_columns']
    identifier = task['identifier']
    model_config = task['model_config']
    options = task.get('options', {})
    
    # Build text from columns
    text_parts = []
    for col in text_columns:
        if pd.notna(row[col]):
            text_parts.append(str(row[col]))
    text = "\n\n".join(text_parts)
    
    # Process each prompt
    raw_dict = {}
    cleaned_dict = {}
    status_dict = {}
    collected_json_objects = []
    
    annotator = LLMAnnotator()

    # Setup the model client for this process
    config_for_setup = {
        'model': model_config.get('model_name'),
        'provider': model_config.get('provider', 'ollama'),
        'api_key': model_config.get('api_key')
    }
    annotator._setup_model_client(config_for_setup)

    for idx, prompt_config in enumerate(prompts, 1):
        prompt_text = prompt_config.get('prompt') or prompt_config.get('template', '')
        expected_keys = prompt_config.get('expected_keys', [])
        prefix = prompt_config.get('prefix', '')
        
        # Build schema
        schema = None
        if expected_keys and not options.get('disable_schema', False):
            schema = build_dynamic_schema(expected_keys)
        
        # Analyze with this prompt
        result = annotator.analyze_text_with_model(
            text=text,
            prompt=prompt_text,
            model_config=model_config,
            schema=schema
        )
        
        raw_dict[str(idx)] = result
        
        if result:
            cleaned_dict[str(idx)] = result
            status_dict[str(idx)] = 'success'
            
            # Apply prefix and collect
            try:
                parsed = json.loads(result)
                if prefix:
                    parsed = {f"{prefix}_{k}": v for k, v in parsed.items()}
                collected_json_objects.append(parsed)
            except:
                status_dict[str(idx)] = 'parse_error'
        else:
            cleaned_dict[str(idx)] = None
            status_dict[str(idx)] = 'error'
    
    # Merge all JSON objects
    merged = {}
    for obj in collected_json_objects:
        if isinstance(obj, dict):
            merged.update(obj)
    
    final_json = json.dumps(merged, ensure_ascii=False) if merged else None
    elapsed = time.perf_counter() - start_time
    
    return {
        'identifier': identifier,
        'final_json': final_json,
        'inference_time': elapsed,
        'raw_json': json.dumps(raw_dict, ensure_ascii=False),
        'cleaned_json': json.dumps(cleaned_dict, ensure_ascii=False),
        'status': json.dumps(status_dict, ensure_ascii=False)
    }


def build_dynamic_schema(expected_keys: List[str]) -> BaseModel:
    """
    Build dynamic Pydantic schema from expected keys.
    
    Parameters
    ----------
    expected_keys : list
        List of expected JSON keys
    
    Returns
    -------
    BaseModel
        Dynamic Pydantic model
    """
    fields = {}
    for key in expected_keys:
        # Make all fields optional strings for flexibility
        fields[key] = (Optional[Union[str, int, float, bool, list, dict]], None)
    
    return create_model('DynamicAnnotationSchema', **fields)

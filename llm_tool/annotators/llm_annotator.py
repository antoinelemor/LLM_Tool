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
import logging
import time
import math
import random
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from collections import deque

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
from ..annotators.json_cleaner import JSONCleaner
from ..config.settings import Settings

# Try to import local model support
try:
    from ..annotators.local_models import OllamaClient, LlamaCPPClient
    HAS_LOCAL_MODELS = True
except ImportError:
    HAS_LOCAL_MODELS = False
    logging.warning("Local model support not available")

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

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM annotator"""
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        self.json_cleaner = JSONCleaner()
        self.prompt_manager = PromptManager()
        self.api_client = None
        self.local_client = None
        self.progress_bar = None

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
        # Validate configuration
        self._validate_config(config)

        # Setup model client
        self._setup_model_client(config)

        # Load data
        data, metadata = self._load_data(config)

        # Prepare prompts
        prompts = self._prepare_prompts(config)

        # Perform annotation
        results = self._annotate_data(data, prompts, config)

        # Save results
        self._save_results(results, config)

        return self._generate_summary(results)

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
                model=model_name
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

        annotation_column = config.get('annotation_column', 'annotation')
        resume = config.get('resume', False)

        # Filter data for annotation
        if resume and annotation_column in data.columns:
            data_to_annotate = data[data[annotation_column].isna()].copy()
            self.logger.info(f"Resuming annotation: {len(data_to_annotate)} rows to process")
        else:
            data_to_annotate = data.copy()

        # Calculate sample size if requested
        if config.get('calculate_sample_size', False):
            sample_size = self.calculate_sample_size(len(data_to_annotate))
            if config.get('use_sample', False):
                data_to_annotate = data_to_annotate.sample(n=sample_size, random_state=42)
                self.logger.info(f"Using sample of {sample_size} rows")

        # Prepare for parallel processing
        num_processes = config.get('num_processes', 1)
        multiple_prompts = len(prompts) > 1

        # Add necessary columns
        if annotation_column not in data.columns:
            data[annotation_column] = pd.NA
        if f"{annotation_column}_inference_time" not in data.columns:
            data[f"{annotation_column}_inference_time"] = pd.NA

        if multiple_prompts:
            for suffix in PROMPT_SUFFIXES:
                col = f"{annotation_column}_{suffix}"
                if col not in data.columns:
                    data[col] = pd.NA

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

        # Execute annotation
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

        # Initialize progress bar
        with tqdm(total=total_tasks, desc='Annotating', unit='items') as pbar:
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

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        identifier = result['identifier']
                        final_json = result['final_json']
                        inference_time = result['inference_time']
                        raw_json = result.get('raw_json')
                        cleaned_json = result.get('cleaned_json')
                        status = result.get('status', 'unknown')

                        # Update dataframe
                        mask = full_data[identifier_column] == identifier
                        if mask.any():
                            full_data.loc[mask, annotation_column] = final_json
                            full_data.loc[mask, f"{annotation_column}_inference_time"] = inference_time

                            # Update per-prompt columns if multiple prompts
                            if raw_json:
                                full_data.loc[mask, f"{annotation_column}_raw_per_prompt"] = raw_json
                            if cleaned_json:
                                full_data.loc[mask, f"{annotation_column}_cleaned_per_prompt"] = cleaned_json
                            if status:
                                full_data.loc[mask, f"{annotation_column}_status_per_prompt"] = status

                        # Track status
                        if final_json:
                            status_counts['success'] += 1
                            batch_results.append((identifier, final_json, inference_time))
                        else:
                            status_counts['error'] += 1

                        # Incremental saving
                        if save_incrementally and output_path:
                            if config['data_source'] == 'csv' and CSV_APPEND:
                                self._append_to_csv(full_data, identifier, identifier_column, annotation_column, output_path)
                            else:
                                pending_save += 1
                                if pending_save >= OTHER_FORMAT_SAVE_EVERY:
                                    self._save_data(full_data, output_path, config['data_source'])
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

                        # Display sample results
                        if len(batch_results) >= 10:
                            sample = random.choice(batch_results)
                            tqdm.write(f"✨ Sample annotation for ID {sample[0]}: {sample[1][:100]}...")
                            batch_results = []

                        pbar.update(1)

                    except Exception as e:
                        import traceback
                        self.logger.error(f"Task failed: {e}")
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        status_counts['error'] += 1
                        pbar.update(1)

        # Final save if needed
        if save_incrementally and output_path and pending_save > 0:
            self._save_data(full_data, output_path, config['data_source'])

        return full_data

    def _save_results(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Save annotation results"""
        output_path = config.get('output_path')
        if output_path:
            format = config.get('output_format') or config.get('data_source', 'csv')
            self._save_data(data, output_path, format)
            self.logger.info(f"Results saved to {output_path}")

    def _save_data(self, data: pd.DataFrame, path: str, format: str):
        """Save data to file"""
        if format == 'csv':
            data.to_csv(path, index=False)
        elif format == 'excel':
            data.to_excel(path, index=False)
        elif format == 'parquet':
            data.to_parquet(path, index=False)
        elif format in ['rdata', 'rds']:
            if HAS_PYREADR:
                if format == 'rdata':
                    pyreadr.write_rdata({'data': data}, path)
                else:
                    pyreadr.write_rds(data, path)
            else:
                raise ImportError("pyreadr required for RData/RDS files")

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

    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate annotation summary statistics"""
        total = len(data)
        success = status_counts.get('success', 0)
        errors = status_counts.get('error', 0)
        
        return {
            'total_processed': total,
            'successful': success,
            'errors': errors,
            'success_rate': (success / total * 100) if total > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }

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
        
        # Call appropriate model
        provider = model_config.get('provider')
        
        if provider in ['openai', 'anthropic', 'google'] and self.api_client:
            response = self.api_client.generate(
                prompt=full_prompt,
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 1000)
            )
        elif provider in ['ollama', 'llamacpp'] and self.local_client:
            response = self.local_client.generate(
                prompt=full_prompt,
                options=model_config.get('options', {})
            )
        else:
            self.logger.error(f"No client configured for provider: {provider}")
            return None
        
        if not response:
            return None
        
        # Clean and validate response
        cleaned = self.json_cleaner.clean_json_output(response, schema=schema)
        
        # Validate with schema if provided
        if cleaned and schema:
            try:
                validated = schema.model_validate_json(cleaned)
                return validated.model_dump_json()
            except Exception as e:
                self.logger.warning(f"Schema validation failed: {e}")
                return cleaned
        
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
    start_time = time.perf_counter()
    
    # Extract task parameters
    row = task['row']
    prompt_config = task['prompts'][0]  # Single prompt
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
    
    # Get prompt details (handle both 'prompt' and 'template' keys)
    prompt_text = prompt_config.get('prompt') or prompt_config.get('template', '')
    expected_keys = prompt_config.get('expected_keys', [])
    prefix = prompt_config.get('prefix', '')
    
    # Build schema if expected keys provided
    schema = None
    if expected_keys and not options.get('disable_schema', False):
        schema = build_dynamic_schema(expected_keys)
    
    # Create annotator instance for this process
    annotator = LLMAnnotator()

    # Setup the model client for this process
    config_for_setup = {
        'model': model_config.get('model_name'),
        'provider': model_config.get('provider', 'ollama'),
        'api_key': model_config.get('api_key')
    }
    annotator._setup_model_client(config_for_setup)

    # Analyze text
    result = annotator.analyze_text_with_model(
        text=text,
        prompt=prompt_text,
        model_config=model_config,
        schema=schema
    )
    
    # Apply prefix if specified
    if result and prefix:
        try:
            parsed = json.loads(result)
            prefixed = {f"{prefix}_{k}": v for k, v in parsed.items()}
            result = json.dumps(prefixed, ensure_ascii=False)
        except:
            pass
    
    elapsed = time.perf_counter() - start_time
    
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

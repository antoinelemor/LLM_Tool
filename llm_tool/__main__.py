#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
__main__.py

MAIN OBJECTIVE:
---------------
This script provides the main entry point for the LLMTool package when executed
as a module with python -m llm_tool or via the console script.

Dependencies:
-------------
- sys
- argparse
- logging

MAIN FEATURES:
--------------
1) Entry point for package execution
2) Command-line argument parsing
3) Mode selection (interactive, batch, API)
4) Environment setup
5) Error handling and logging

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path

# Console first: on Windows the standard streams default to the ANSI code page,
# and the banner below would raise UnicodeEncodeError before anything else runs.
from .platform_compat import configure_console, writable_dir

configure_console()

# Crash & stderr capture — runs before any heavy import so transformers/torch
# warnings, Rich Live tracebacks, and C-level aborts all reach the filesystem.
import datetime as _dt
import faulthandler as _fh
import traceback as _tb

# Logs belong next to the user's work, not next to the code: a wheel installed
# into site-packages (or C:\Program Files) sits in a tree nobody can write to,
# and the package directory is not where anyone would look for them anyway.
_LOG_DIR = writable_dir(
    [
        Path.cwd() / "logs" / "application",
        Path.home() / ".llm_tool" / "logs" / "application",
    ],
    prefix="llm_tool_logs",
)
_STAMP = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# encoding is explicit because Python picks the ANSI code page on Windows, and
# these files capture tracebacks that routinely carry non-ASCII text.
_fault_fp = open(_LOG_DIR / f"faulthandler_{_STAMP}.log", "w", encoding="utf-8")
_fh.enable(file=_fault_fp, all_threads=True)


class _StderrTee:
    def __init__(self, real, fp):
        self._real = real
        self._fp = fp

    def write(self, data):
        try:
            self._fp.write(data)
            self._fp.flush()
        except Exception:
            pass
        return self._real.write(data)

    def flush(self):
        try:
            self._fp.flush()
        except Exception:
            pass
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


_stderr_fp = open(_LOG_DIR / f"stderr_{_STAMP}.log", "w", buffering=1, encoding="utf-8")
sys.stderr = _StderrTee(sys.stderr, _stderr_fp)

_default_excepthook = sys.excepthook


def _excepthook(exc_type, exc_value, exc_tb):
    try:
        with open(_LOG_DIR / f"exceptions_{_STAMP}.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {_dt.datetime.now().isoformat()} ===\n")
            _tb.print_exception(exc_type, exc_value, exc_tb, file=f)
    except Exception:
        pass
    _default_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook

import argparse
import logging
import os
from typing import Optional

# Import main CLI
from .cli.main_cli import LLMToolCLI
from .config.settings import get_settings
from .__init__ import __version__


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        prog='llm-tool',
        description='LLMTool - State-of-the-Art LLM Annotation & Training Pipeline',
        epilog='For more information, visit https://github.com/antoine-lemor/LLMTool'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'LLMTool v{__version__}'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple CLI instead of advanced interface'
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--interactive',
        action='store_true',
        default=True,
        help='Run in interactive CLI mode (default)'
    )

    mode_group.add_argument(
        '--batch',
        type=str,
        metavar='CONFIG_FILE',
        help='Run in batch mode with configuration file'
    )

    mode_group.add_argument(
        '--api',
        action='store_true',
        help='Run as API server'
    )

    # Specific actions
    action_group = parser.add_argument_group('actions')
    action_group.add_argument(
        '--annotate',
        type=str,
        metavar='DATA_FILE',
        help='Direct annotation of a data file'
    )

    action_group.add_argument(
        '--train',
        type=str,
        metavar='DATA_FILE',
        help='Direct training from annotated data'
    )

    action_group.add_argument(
        '--benchmark',
        type=str,
        metavar='DATA_FILE',
        help='Run benchmark on annotated data'
    )

    action_group.add_argument(
        '--validate',
        type=str,
        metavar='DATA_FILE',
        help='Validate annotations and export to Doccano'
    )

    # Model selection
    model_group = parser.add_argument_group('model options')
    model_group.add_argument(
        '--model',
        type=str,
        help='Model to use (e.g., gpt-4, llama3.2, bert-base)'
    )

    model_group.add_argument(
        '--api-key',
        type=str,
        help='API key for cloud models'
    )

    model_group.add_argument(
        '--ollama-api-key',
        type=str,
        metavar='KEY',
        help='Bearer token for a remote Ollama endpoint (defaults to $OLLAMA_API_KEY, '
             'then the key stored for the "ollama" provider)'
    )

    # At most one endpoint may be named: these three are three ways of answering
    # the same question, and silently letting one win would hide a typo.
    endpoint_group = model_group.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        '--local',
        action='store_true',
        help='Use local models only; pins Ollama to http://localhost:11434 even if '
             '$OLLAMA_HOST points elsewhere'
    )

    endpoint_group.add_argument(
        '--ollama-host',
        type=str,
        metavar='URL',
        help='Base URL of the Ollama server to annotate with, e.g. '
             'http://192.168.1.10:11434 (defaults to $OLLAMA_HOST, then '
             'http://localhost:11434)'
    )

    endpoint_group.add_argument(
        '--ollama-cloud',
        action='store_true',
        help='Shorthand for --ollama-host https://ollama.com; needs a key from '
             '--ollama-api-key or $OLLAMA_API_KEY, since only model listing is public'
    )

    # Data options
    data_group = parser.add_argument_group('data options')
    data_group.add_argument(
        '--format',
        choices=['csv', 'json', 'jsonl', 'excel', 'parquet', 'postgresql'],
        help='Data format'
    )

    data_group.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )

    data_group.add_argument(
        '--prompt',
        type=str,
        help='Path to prompt file or directory'
    )

    # Processing options
    processing_group = parser.add_argument_group('processing options')
    processing_group.add_argument(
        '--parallel',
        type=int,
        metavar='N',
        default=1,
        help='Number of parallel processes (default: 1)'
    )

    processing_group.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )

    processing_group.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process'
    )

    return parser.parse_args()


def _persisted_ollama_host() -> Optional[str]:
    """
    Read the Ollama host saved by the interactive endpoint manager.

    Returns
    -------
    Optional[str]
        The stored base URL, or ``None`` when no endpoint was ever chosen.
    """
    return getattr(get_settings().local_model, 'host', None)


def _ollama_endpoint_overrides(args, config: Optional[dict] = None) -> dict:
    """
    Translate the Ollama endpoint flags into pipeline configuration keys.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : dict, optional
        Run configuration already loaded from a batch file. An endpoint it names
        outranks the saved preference, but never the flags or the environment.

    Returns
    -------
    dict
        ``{'ollama_host': str, 'ollama_api_key': Optional[str]}``, always
        populated: the headless run pins the endpoint it resolved instead of
        leaving each annotator to re-derive one from an environment it cannot
        see the settings file through.

    Notes
    -----
    Precedence matches the interactive picker: explicit flag, then
    ``OLLAMA_HOST`` / ``OLLAMA_API_KEY``, then the endpoint saved in the settings
    file, then the local daemon. The lower-priority fallbacks are only handed to
    :func:`resolve_ollama_endpoint` when the environment is silent, because it
    ranks an explicit argument above the environment.
    """
    # Deferred because pulling in the Ollama SDK costs about a second, which would
    # be paid by --help and --version too.
    from .annotators.local_models import (
        DEFAULT_OLLAMA_HOST,
        OLLAMA_CLOUD_HOST,
        resolve_ollama_endpoint,
    )

    env_host = os.environ.get('OLLAMA_HOST')
    env_key = os.environ.get('OLLAMA_API_KEY')
    from_config = config or {}

    if args.local:
        host = DEFAULT_OLLAMA_HOST
    elif args.ollama_cloud:
        host = OLLAMA_CLOUD_HOST
    elif args.ollama_host:
        # A bare host:port is what people type and what the interactive picker
        # completes; without a scheme it is not a usable base URL.
        host = args.ollama_host if '://' in args.ollama_host else f"http://{args.ollama_host}"
    elif env_host:
        host = None
    else:
        host = from_config.get('ollama_host') or _persisted_ollama_host()

    api_key = args.ollama_api_key or (None if env_key else from_config.get('ollama_api_key'))

    endpoint = resolve_ollama_endpoint(host=host, api_key=api_key)
    return {'ollama_host': endpoint.host, 'ollama_api_key': endpoint.api_key}


def _apply_config_file(args) -> None:
    """
    Load the file given to ``--config`` before anything else reads the settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Notes
    -----
    The advanced CLI resolves its Ollama endpoint while it is being constructed,
    so the file has to be in place first. Its host is exported the way the
    endpoint manager exports a chosen one, since the environment is the channel
    every annotator and detector reads; an ``OLLAMA_HOST`` already present in the
    shell is a deliberate override for that shell and is left untouched.
    """
    if not args.config:
        return

    settings = get_settings()
    settings.load(args.config)

    host = getattr(settings.local_model, 'host', None)
    if host and not os.environ.get('OLLAMA_HOST'):
        os.environ['OLLAMA_HOST'] = host


def run_interactive_mode(args):
    """Run in interactive CLI mode"""
    # Check for advanced mode preference
    use_advanced = not getattr(args, 'simple', False)

    # Try to use advanced CLI first
    if use_advanced:
        try:
            from .cli.advanced_cli import AdvancedCLI
            cli = AdvancedCLI()
        except ImportError:
            # Fallback to simple CLI
            logging.info("Advanced CLI unavailable, using simple CLI")
            logging.info(" For advanced features, install: pip install rich pandas psutil")
            from .cli.main_cli import LLMToolCLI
            cli = LLMToolCLI()
    else:
        from .cli.main_cli import LLMToolCLI
        cli = LLMToolCLI()

    # Run the interactive CLI
    cli.run()


def run_batch_mode(config_file: str, args):
    """Run in batch mode with configuration file"""
    import json

    # Load configuration
    config_path = Path(config_file)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_file}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Create pipeline controller
    from .pipelines.pipeline_controller import PipelineController

    controller = PipelineController()

    # Override config with command-line arguments
    if args.model:
        config['model'] = args.model
        # The annotation phase looks the model up under its own key; 'model' alone
        # never reaches the annotator and silently leaves the default in place.
        config['annotation_model'] = args.model
    if args.api_key:
        config['api_key'] = args.api_key
    if args.output:
        config['output'] = args.output
        # Every phase reads the destination from 'output_path'; 'output' alone is
        # inert and the results land in the auto-generated annotations folder.
        config['output_path'] = args.output
    if args.parallel:
        config['parallel'] = args.parallel
    config.update(_ollama_endpoint_overrides(args, config))

    # Run pipeline
    try:
        state = controller.run_pipeline(config)

        # Print results
        if state.errors:
            logging.error(f"Pipeline completed with {len(state.errors)} errors")
            for error in state.errors:
                logging.error(f"  - {error}")
            sys.exit(1)
        else:
            logging.info("Pipeline completed successfully")

            # Print summary
            if state.annotation_results:
                logging.info(f"Annotated: {state.annotation_results.get('total_annotated', 0)} items")
            if state.training_results:
                logging.info(f"Best model: {state.training_results.get('best_model', 'unknown')}")
                logging.info(f"Accuracy: {state.training_results.get('accuracy', 0):.2%}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


def run_api_mode(args):
    """Run as API server"""
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        logging.error('API mode requires fastapi and uvicorn. Install with: pip install "llm-tool[api]"')
        sys.exit(1)

    # Create FastAPI app
    app = FastAPI(
        title="LLMTool API",
        version=__version__,
        description="API for LLM annotation and training"
    )

    # Add API routes
    try:
        from .api import routes
    except ImportError as exc:
        logging.error("API mode is not available because the API module is missing: %s", exc)
        logging.error("Install the optional API components or remove the --api flag.")
        sys.exit(1)
    app.include_router(routes.router)

    # Run server
    port = 8000
    logging.info(f"Starting API server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_direct_action(args):
    """Run a direct action (annotate, train, benchmark, validate)"""
    from .pipelines.pipeline_controller import PipelineController

    controller = PipelineController()

    # Build configuration from arguments
    config = {
        'parallel': args.parallel,
        'batch_size': args.batch_size,
    }

    if args.model:
        config['model'] = args.model
        # The annotation phase looks the model up under its own key; 'model' alone
        # never reaches the annotator and silently leaves the default in place.
        config['annotation_model'] = args.model
    if args.api_key:
        config['api_key'] = args.api_key
    if args.output:
        config['output'] = args.output
        # Every phase reads the destination from 'output_path'; 'output' alone is
        # inert and the results land in the auto-generated annotations folder.
        config['output_path'] = args.output
    if args.format:
        config['data_format'] = args.format
    if args.prompt:
        config['prompt_path'] = args.prompt
    if args.max_samples:
        config['max_samples'] = args.max_samples
    config.update(_ollama_endpoint_overrides(args))

    try:
        if args.annotate:
            logging.info(f"Annotating {args.annotate}")
            config['file_path'] = args.annotate
            config['mode'] = 'file'
            if 'data_source' not in config:
                data_source = None
                suffix = Path(args.annotate).suffix.lower()
                mapping = {
                    '.csv': 'csv',
                    '.tsv': 'csv',
                    '.txt': 'csv',
                    '.xlsx': 'excel',
                    '.xls': 'excel',
                    '.parquet': 'parquet',
                    '.json': 'json',
                    '.jsonl': 'jsonl'
                }
                data_source = mapping.get(suffix)
                if not data_source:
                    logging.error(f"Could not infer data source from extension '{suffix}'. Please specify a supported format (csv, excel, parquet, json, jsonl).")
                    sys.exit(1)
                config['data_source'] = data_source
            results = controller.run_annotation(config)
            logging.info(f"Annotation complete: {results.get('total_annotated', 0)} items processed")

        elif args.train:
            logging.info(f"Training on {args.train}")
            config['input_file'] = args.train
            results = controller.run_training(config)
            logging.info(f"Training complete: {results.get('best_model', 'unknown')}")

        elif args.benchmark:
            logging.info(f"Benchmarking on {args.benchmark}")
            config['input_file'] = args.benchmark
            config['benchmark_mode'] = True
            results = controller.run_training(config)
            logging.info(f"Benchmark complete: Best model {results.get('best_model', 'unknown')}")

        elif args.validate:
            logging.info(f"Validating {args.validate}")
            from .validators.annotation_validator import AnnotationValidator
            validator = AnnotationValidator()
            config['input_file'] = args.validate
            results = validator.validate(config)
            logging.info(f"Validation complete: {results.get('samples_validated', 0)} samples")

    except Exception as e:
        logging.error(f"Action failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    # Windows has no fork: every worker process re-imports and re-executes this
    # module. Without this call a frozen build would relaunch the whole CLI once
    # per worker instead of starting one, and spawn the same again from there.
    import multiprocessing

    multiprocessing.freeze_support()

    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Before any mode builds a CLI, a controller or an endpoint from the settings.
    _apply_config_file(args)

    # Handle direct actions first
    if any([args.annotate, args.train, args.benchmark, args.validate]):
        run_direct_action(args)
    elif args.batch:
        run_batch_mode(args.batch, args)
    elif args.api:
        run_api_mode(args)
    else:
        # Default to interactive mode
        run_interactive_mode(args)


if __name__ == "__main__":
    main()

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
import argparse
import logging
from pathlib import Path
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
        '--local',
        action='store_true',
        help='Use local models only'
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
            logging.info("💡 For advanced features, install: pip install rich pandas psutil")
            from .cli.main_cli import LLMToolCLI
            cli = LLMToolCLI()
    else:
        from .cli.main_cli import LLMToolCLI
        cli = LLMToolCLI()

    # Apply any command-line settings
    if args.config:
        settings = get_settings()
        settings.load(args.config)

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

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create pipeline controller
    from .pipelines.pipeline_controller import PipelineController

    controller = PipelineController()

    # Override config with command-line arguments
    if args.model:
        config['model'] = args.model
    if args.api_key:
        config['api_key'] = args.api_key
    if args.output:
        config['output'] = args.output
    if args.parallel:
        config['parallel'] = args.parallel

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
        logging.error("API mode requires fastapi and uvicorn. Install with: pip install llm-tool[advanced]")
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
    if args.api_key:
        config['api_key'] = args.api_key
    if args.output:
        config['output'] = args.output
    if args.format:
        config['data_format'] = args.format
    if args.prompt:
        config['prompt_path'] = args.prompt
    if args.max_samples:
        config['max_samples'] = args.max_samples

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
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

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

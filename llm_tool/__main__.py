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
        help='Run in agent conversational mode (default)'
    )

    mode_group.add_argument(
        '--classic',
        action='store_true',
        help='Run in classic interactive CLI mode (legacy)'
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

    mode_group.add_argument(
        '--transcribe-audio',
        type=str,
        metavar='AUDIO_FILE',
        help='Transcribe an audio file using Whisper'
    )

    mode_group.add_argument(
        '--youtube',
        type=str,
        metavar='URL',
        help='Download audio from a YouTube URL'
    )

    mode_group.add_argument(
        '--tiktok',
        type=str,
        metavar='URL',
        help='Download audio from a TikTok URL'
    )

    # Agent options
    agent_group = parser.add_argument_group('agent options')
    agent_group.add_argument(
        '--agent-provider',
        type=str,
        choices=['anthropic', 'openai', 'ollama'],
        help='LLM provider for the agent (default: ollama)'
    )
    agent_group.add_argument(
        '--agent-model',
        type=str,
        help='Model for the agent (e.g., claude-sonnet-4-20250514, gpt-4o, llama3.2)'
    )
    agent_group.add_argument(
        '--agent-keep-alive',
        type=str,
        help='Keep Ollama models loaded for this duration (e.g., 30m, 1h, -1)'
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

    # Transcription options
    transcription_group = parser.add_argument_group('transcription options')
    transcription_group.add_argument(
        '--whisper-model',
        type=str,
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper model for transcription (default: large-v3)'
    )
    transcription_group.add_argument(
        '--diarize',
        action='store_true',
        help='Enable speaker diarization'
    )
    transcription_group.add_argument(
        '--cookies',
        type=str,
        choices=['chrome', 'firefox', 'safari', 'edge'],
        help='Browser cookies for YouTube/TikTok authentication'
    )
    transcription_group.add_argument(
        '--language',
        type=str,
        help='Language code for transcription (auto-detect if omitted)'
    )
    transcription_group.add_argument(
        '--output-format',
        type=str,
        default='txt',
        choices=['txt', 'csv', 'json', 'srt', 'vtt'],
        help='Transcription output format (default: txt)'
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


def run_agent_mode(args):
    """Run in agent conversational mode (new default)."""
    try:
        from .agent import AgentCLI, AgentConfig
        from .config.settings import get_settings

        settings = get_settings()
        if args.config:
            settings.load(args.config)

        # Build agent config from env + settings
        config = AgentConfig.from_env_and_settings(settings)

        # Override with CLI args if provided
        model_explicit = False
        if getattr(args, 'agent_provider', None):
            config.provider = args.agent_provider
        if getattr(args, 'agent_model', None):
            config.model = args.agent_model
            model_explicit = True
        if getattr(args, 'agent_keep_alive', None):
            config.keep_alive = args.agent_keep_alive
        if getattr(args, 'api_key', None):
            config.api_key = args.api_key

        # Run the agent
        agent = AgentCLI(config=config, settings=settings, model_explicit=model_explicit)
        agent.run()

    except ImportError as e:
        logging.warning(f"Agent mode unavailable ({e}), falling back to classic CLI")
        run_interactive_mode(args)
    except Exception as e:
        logging.error(f"Agent mode failed: {e}")
        logging.info("Falling back to classic CLI...")
        run_interactive_mode(args)


def _run_transcription(args):
    """Handle --transcribe-audio flag: transcribe a local audio file."""
    try:
        from .transcription.transcriber.whisper_transcriber import WhisperTranscriber, TranscriptionConfig
        from .transcription.transcriber.text_processor import TextProcessor
        from .transcription.transcriber.diarization import SpeakerDiarizer, DiarizationConfig
        from .transcription.config.settings import get_config, Config

        audio_path = Path(args.transcribe_audio)
        if not audio_path.exists():
            logging.error(f"Audio file not found: {audio_path}")
            sys.exit(1)

        config = get_config()
        config.whisper_model = getattr(args, 'whisper_model', 'large-v3')
        config.output_format = getattr(args, 'output_format', 'txt')
        if getattr(args, 'language', None):
            config.whisper_language = args.language

        # Create transcription config
        tc = TranscriptionConfig(
            model_name=config.whisper_model,
            language=config.whisper_language,
            device=config.device,
            fp16=config.use_fp16,
        )

        # Transcribe
        logging.info(f"Transcribing {audio_path} with Whisper {config.whisper_model}...")
        transcriber = WhisperTranscriber(config=tc)
        transcriber.load_model()
        result = transcriber.transcribe(str(audio_path))

        if not result.success:
            logging.error(f"Transcription failed: {result.error}")
            sys.exit(1)

        # Diarization (optional)
        words = result.words
        if getattr(args, 'diarize', False):
            logging.info("Running speaker diarization...")
            dc = DiarizationConfig(device=config.device)
            diarizer = SpeakerDiarizer(config=dc)
            if diarizer.load_pipeline():
                import pandas as pd
                diar_result = diarizer.diarize(str(audio_path))
                if diar_result.success:
                    words_df = pd.DataFrame(words)
                    words_df = diarizer.align_words_with_speakers(words_df, diar_result)
                    words = words_df.to_dict('records')
                diarizer.cleanup()

        # Format output
        output_format = config.output_format
        language = result.language or 'en'

        if output_format == 'txt':
            output = TextProcessor.format_transcript(words, include_speakers=getattr(args, 'diarize', False), language=language)
        elif output_format == 'csv':
            metadata = {'title': audio_path.stem, 'date': '', 'video_id': '', 'channel': ''}
            output = TextProcessor.format_csv(words, metadata, language=language)
        elif output_format == 'srt':
            output = TextProcessor.format_srt(words, language=language)
        elif output_format == 'vtt':
            output = TextProcessor.format_vtt(words, language=language)
        else:
            import json
            output = json.dumps({
                'text': result.text,
                'language': result.language,
                'words': words,
                'segments': result.segments,
                'duration': result.duration,
            }, ensure_ascii=False, indent=2)

        # Save
        output_dir = Path(getattr(args, 'output', None) or '.')
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = output_format if output_format != 'vtt' else 'vtt'
        output_file = output_dir / f"{audio_path.stem}_transcription.{ext}"
        output_file.write_text(output, encoding='utf-8')
        logging.info(f"Transcription saved to {output_file}")

        transcriber.cleanup()

    except ImportError as e:
        logging.error(f"Transcription dependencies not available: {e}")
        logging.error("Install with: pip install llm-tool[transcription]")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        sys.exit(1)


def _run_youtube_download(args):
    """Handle --youtube flag: download audio from YouTube."""
    try:
        from .transcription.extractors import YouTubeExtractor
        from .transcription.extractors.base import ExtractorConfig

        url = args.youtube
        output_dir = getattr(args, 'output', None) or 'data/audio'

        ext_config = ExtractorConfig(
            output_dir=output_dir,
            audio_format='wav',
            cookies_from_browser=getattr(args, 'cookies', None),
        )

        extractor = YouTubeExtractor(config=ext_config)
        logging.info(f"Downloading audio from YouTube: {url}")

        result = extractor.extract_audio(url)

        if result.success:
            logging.info(f"Audio saved to: {result.audio_path}")
            logging.info(f"Title: {result.title}")
            logging.info(f"Duration: {result.duration:.1f}s" if result.duration else "")

            # Auto-transcribe if --transcribe-audio is also... no, they're mutually exclusive
            # User can run --transcribe-audio on the output file separately
        else:
            logging.error(f"Download failed: {result.error}")
            sys.exit(1)

    except ImportError as e:
        logging.error(f"Transcription dependencies not available: {e}")
        logging.error("Install with: pip install llm-tool[transcription]")
        sys.exit(1)
    except Exception as e:
        logging.error(f"YouTube download failed: {e}")
        sys.exit(1)


def _run_tiktok_download(args):
    """Handle --tiktok flag: download audio from TikTok."""
    try:
        from .transcription.extractors import TikTokExtractor
        from .transcription.extractors.base import ExtractorConfig

        url = args.tiktok
        output_dir = getattr(args, 'output', None) or 'data/audio'

        ext_config = ExtractorConfig(
            output_dir=output_dir,
            audio_format='wav',
            cookies_from_browser=getattr(args, 'cookies', None),
        )

        extractor = TikTokExtractor(config=ext_config)
        logging.info(f"Downloading audio from TikTok: {url}")

        result = extractor.extract_audio(url)

        if result.success:
            logging.info(f"Audio saved to: {result.audio_path}")
            logging.info(f"Title: {result.title}")
        else:
            logging.error(f"Download failed: {result.error}")
            sys.exit(1)

    except ImportError as e:
        logging.error(f"Transcription dependencies not available: {e}")
        logging.error("Install with: pip install llm-tool[transcription]")
        sys.exit(1)
    except Exception as e:
        logging.error(f"TikTok download failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Handle transcription actions first
    if getattr(args, 'transcribe_audio', None):
        _run_transcription(args)
        return
    elif getattr(args, 'youtube', None):
        _run_youtube_download(args)
        return
    elif getattr(args, 'tiktok', None):
        _run_tiktok_download(args)
        return

    # Handle direct actions first
    if any([args.annotate, args.train, args.benchmark, args.validate]):
        run_direct_action(args)
    elif args.batch:
        run_batch_mode(args.batch, args)
    elif args.api:
        run_api_mode(args)
    elif getattr(args, 'classic', False):
        # Legacy mode: use the existing CLI
        run_interactive_mode(args)
    else:
        # Default to agent mode
        run_agent_mode(args)


if __name__ == "__main__":
    main()

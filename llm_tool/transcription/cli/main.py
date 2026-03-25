"""
Main CLI interface for Transcribe Tool.

Provides an interactive Rich-based command line interface
for audio extraction and transcription workflows.
"""

# Suppress warnings before importing other modules
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import sys
import os
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Rich imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Rich library is required. Install with: pip install rich")
    sys.exit(1)

# Internal imports
from ..config.settings import Config, get_config, set_config
from ..extractors import YouTubeExtractor, TikTokExtractor, LocalExtractor
from ..extractors.base import ExtractorConfig
from ..transcriber import WhisperTranscriber, TranscriptionResult
from ..transcriber.whisper_transcriber import TranscriptionConfig
from ..transcriber.text_processor import TextProcessor
from ..utils.language_detector import LanguageDetector, detect_language, get_language_name
from ..utils.tokenizer import Tokenizer, segment_text, parse_segmentation_level
from ..utils.document_loader import DocumentLoader, Document, LoaderResult
from ..utils.text_tokenization import (
    TextTokenizer, NLPFeatures, TokenizationResult,
    format_tokenization_csv, format_tokenization_json,
    get_available_nlp_features, get_supported_languages as get_nlp_languages
)


# Banner - Simple text version that always displays correctly
BANNER_TITLE = "TRANSCRIBE TOOL"
BANNER_SUBTITLE = "Audio Extraction & Transcription Pipeline"
BANNER_DESC = "YouTube | TikTok | Local Files → Whisper → Export"


class TranscribeCLI:
    """Interactive CLI for audio extraction and transcription."""

    def __init__(self):
        self.console = Console()
        self.config = get_config()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("transcribe_tool")
        logger.setLevel(logging.WARNING)  # Only show warnings and errors in CLI
        logger.propagate = False  # Prevent duplicate logs from root logger

        # Remove any existing handlers
        logger.handlers.clear()

        # We don't add a console handler - Rich handles the display
        # Only add handler for file logging if needed in the future

        return logger

    def run(self):
        """Run the main CLI loop."""
        self._show_banner()
        self._show_system_info()

        while True:
            try:
                choice = self._show_main_menu()

                if choice == "1":
                    self._youtube_workflow()
                elif choice == "2":
                    self._tiktok_workflow()
                elif choice == "3":
                    self._local_workflow()
                elif choice == "4":
                    self._transcribe_workflow()
                elif choice == "5":
                    self._full_pipeline_workflow()
                elif choice == "6":
                    self._tokenization_workflow()
                elif choice == "7":
                    self._settings_menu()
                elif choice == "0":
                    self._exit()
                    break
                else:
                    self.console.print("[yellow]Invalid choice. Please try again.[/yellow]")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Returning to menu...[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    def _show_banner(self):
        """Display the application banner."""
        from rich.align import Align

        banner_text = """
[cyan]████████╗██████╗  █████╗ ███╗   ██╗███████╗ ██████╗██████╗ ██╗██████╗ ███████╗[/cyan]
[cyan]╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗██║██╔══██╗██╔════╝[/cyan]
[cyan]   ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ██████╔╝██║██████╔╝█████╗  [/cyan]
[cyan]   ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██╗██║██╔══██╗██╔══╝  [/cyan]
[cyan]   ██║   ██║  ██║██║  ██║██║ ╚████║███████║╚██████╗██║  ██║██║██████╔╝███████╗[/cyan]
[cyan]   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═════╝ ╚══════╝[/cyan]
[bold cyan]                    ████████╗ ██████╗  ██████╗ ██╗     [/bold cyan]
[bold cyan]                    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     [/bold cyan]
[bold cyan]                       ██║   ██║   ██║██║   ██║██║     [/bold cyan]
[bold cyan]                       ██║   ██║   ██║██║   ██║██║     [/bold cyan]
[bold cyan]                       ██║   ╚██████╔╝╚██████╔╝███████╗[/bold cyan]
[bold cyan]                       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝[/bold cyan]

[white]Audio Extraction & Transcription Pipeline[/white]
[dim]YouTube | TikTok | Local Files → Whisper → Export[/dim]

[dim italic]by Antoine Lemor[/dim italic]
"""
        self.console.print(Panel(
            Align.center(banner_text),
            border_style="blue",
            padding=(0, 1)
        ))

    def _show_system_info(self):
        """Display system information."""
        import torch
        import platform
        import subprocess
        import shutil

        # Get friendly OS name
        system = platform.system()
        if system == "Darwin":
            try:
                mac_ver = platform.mac_ver()[0]
                os_name = f"macOS {mac_ver}"
            except Exception:
                os_name = "macOS"
        elif system == "Windows":
            os_name = f"Windows {platform.release()}"
        elif system == "Linux":
            try:
                import distro
                os_name = f"{distro.name()} {distro.version()}"
            except ImportError:
                os_name = f"Linux {platform.release()}"
        else:
            os_name = system

        # Detect device
        if torch.cuda.is_available():
            device = f"CUDA ({torch.cuda.get_device_name(0)})"
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "Apple Silicon (MPS)"
            gpu_memory = "Unified Memory"
        else:
            device = "CPU"
            gpu_memory = "N/A"

        # Get FFmpeg version
        ffmpeg_version = "Not found"
        if shutil.which("ffmpeg"):
            try:
                result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    first_line = result.stdout.split('\n')[0]
                    # Extract version number
                    parts = first_line.split()
                    if len(parts) >= 3:
                        ffmpeg_version = parts[2]
            except Exception:
                ffmpeg_version = "Installed"

        table = Table(title="System Information", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("OS", os_name)
        table.add_row("Python", platform.python_version())
        table.add_row("PyTorch", torch.__version__)
        table.add_row("Device", device)
        table.add_row("GPU Memory", gpu_memory)
        table.add_row("FFmpeg", ffmpeg_version)

        self.console.print(table)
        self.console.print()

    def _show_main_menu(self) -> str:
        """Display the main menu and get user choice."""
        self.console.print()
        self.console.print(Panel(
            "[bold]Main Menu[/bold]",
            border_style="blue"
        ))

        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Option", style="cyan", width=4)
        menu_table.add_column("Description", style="white")

        menu_table.add_row("[1]", "Extract audio from YouTube")
        menu_table.add_row("[2]", "Extract audio from TikTok")
        menu_table.add_row("[3]", "Process local audio/video files")
        menu_table.add_row("[4]", "Transcribe audio files")
        menu_table.add_row("[5]", "[bold green]Full Pipeline[/bold green] (Extract → Transcribe → Export)")
        menu_table.add_row("[6]", "[bold magenta]Text Tokenization[/bold magenta] (Documents → SOTA NLP → CSV)")
        menu_table.add_row("[7]", "Settings")
        menu_table.add_row("[0]", "Exit")

        self.console.print(menu_table)
        self.console.print()

        return Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="1")

    def _youtube_workflow(self):
        """YouTube extraction workflow."""
        self.console.print(Panel("[bold cyan]YouTube Audio Extraction[/bold cyan]", border_style="cyan"))

        # Ask for extraction type
        self.console.print("\n[bold]Extraction Type:[/bold]")
        self.console.print("  [1] Single video URL")
        self.console.print("  [2] Channel videos")
        self.console.print("  [3] Playlist")
        self.console.print("  [0] Back to menu")

        choice = Prompt.ask("Select", choices=["0", "1", "2", "3"], default="1")

        if choice == "0":
            return

        # Setup extractor
        extractor_config = ExtractorConfig(
            output_dir=self.config.audio_dir,
            audio_format=self.config.audio_format,
            cookies_from_browser=self.config.cookies_from_browser
        )

        try:
            extractor = YouTubeExtractor(config=extractor_config, logger=self.logger)
        except ImportError as e:
            self.console.print(f"[red]{e}[/red]")
            return

        if choice == "1":
            # Single video
            url = Prompt.ask("Enter YouTube video URL")
            if not url:
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Downloading audio...", total=1)
                result = extractor.extract_audio(url)
                progress.update(task, advance=1)

            if result.success:
                self.console.print(f"[green]Downloaded: {result.audio_path}[/green]")
                if Confirm.ask("Transcribe now?"):
                    self._transcribe_file(result.audio_path)
            else:
                self.console.print(f"[red]Failed: {result.error}[/red]")

        elif choice == "2":
            # Channel
            channel_url = Prompt.ask("Enter YouTube channel URL")
            max_videos = IntPrompt.ask("Maximum videos to extract", default=50)

            self.console.print("[yellow]Extracting video URLs...[/yellow]")
            urls = extractor.extract_channel_videos(channel_url, max_videos=max_videos)

            if not urls:
                self.console.print("[red]No videos found[/red]")
                return

            self.console.print(f"[green]Found {len(urls)} videos[/green]")

            if Confirm.ask(f"Download audio from {len(urls)} videos?"):
                self._batch_download(extractor, urls)

        elif choice == "3":
            # Playlist
            playlist_url = Prompt.ask("Enter YouTube playlist URL")

            self.console.print("[yellow]Extracting playlist...[/yellow]")
            urls = extractor.extract_playlist(playlist_url)

            if not urls:
                self.console.print("[red]No videos found[/red]")
                return

            self.console.print(f"[green]Found {len(urls)} videos[/green]")

            if Confirm.ask(f"Download audio from {len(urls)} videos?"):
                self._batch_download(extractor, urls)

    def _tiktok_workflow(self):
        """TikTok extraction workflow."""
        self.console.print(Panel("[bold magenta]TikTok Audio Extraction[/bold magenta]", border_style="magenta"))

        self.console.print("\n[bold]Extraction Type:[/bold]")
        self.console.print("  [1] Single video URL")
        self.console.print("  [2] User profile videos")
        self.console.print("  [0] Back to menu")

        choice = Prompt.ask("Select", choices=["0", "1", "2"], default="1")

        if choice == "0":
            return

        extractor_config = ExtractorConfig(
            output_dir=self.config.audio_dir,
            audio_format=self.config.audio_format,
            cookies_from_browser=self.config.cookies_from_browser
        )

        try:
            extractor = TikTokExtractor(config=extractor_config, logger=self.logger)
        except ImportError as e:
            self.console.print(f"[red]{e}[/red]")
            return

        if choice == "1":
            url = Prompt.ask("Enter TikTok video URL")
            if not url:
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Downloading audio...", total=1)
                result = extractor.extract_audio(url)
                progress.update(task, advance=1)

            if result.success:
                self.console.print(f"[green]Downloaded: {result.audio_path}[/green]")
                if Confirm.ask("Transcribe now?"):
                    self._transcribe_file(result.audio_path)
            else:
                self.console.print(f"[red]Failed: {result.error}[/red]")

        elif choice == "2":
            user_url = Prompt.ask("Enter TikTok user profile URL")
            max_videos = IntPrompt.ask("Maximum videos", default=30)

            self.console.print("[yellow]Extracting video URLs...[/yellow]")
            urls = extractor.extract_user_videos(user_url, max_videos=max_videos)

            if urls:
                self.console.print(f"[green]Found {len(urls)} videos[/green]")
                if Confirm.ask(f"Download audio from {len(urls)} videos?"):
                    self._batch_download(extractor, urls)
            else:
                self.console.print("[red]No videos found[/red]")

    def _local_workflow(self):
        """Local file processing workflow."""
        self.console.print(Panel("[bold green]Local File Processing[/bold green]", border_style="green"))

        self.console.print("\n[bold]Processing Type:[/bold]")
        self.console.print("  [1] Single file")
        self.console.print("  [2] Directory scan")
        self.console.print("  [0] Back to menu")

        choice = Prompt.ask("Select", choices=["0", "1", "2"], default="1")

        if choice == "0":
            return

        extractor_config = ExtractorConfig(
            output_dir=self.config.audio_dir,
            audio_format=self.config.audio_format
        )
        extractor = LocalExtractor(config=extractor_config, logger=self.logger)

        if choice == "1":
            file_path = Prompt.ask("Enter file path")
            if not file_path or not Path(file_path).exists():
                self.console.print("[red]File not found[/red]")
                return

            result = extractor.extract_audio(file_path)

            if result.success:
                self.console.print(f"[green]Processed: {result.audio_path}[/green]")
                if Confirm.ask("Transcribe now?"):
                    self._transcribe_file(result.audio_path)
            else:
                self.console.print(f"[red]Failed: {result.error}[/red]")

        elif choice == "2":
            directory = Prompt.ask("Enter directory path")
            if not directory or not Path(directory).exists():
                self.console.print("[red]Directory not found[/red]")
                return

            files = extractor.scan_directory(directory)

            if not files:
                self.console.print("[red]No audio/video files found[/red]")
                return

            # Display found files
            self.console.print(f"\n[green]Found {len(files)} files:[/green]")
            for f in files[:10]:
                self.console.print(f"  - {f.name}")
            if len(files) > 10:
                self.console.print(f"  ... and {len(files) - 10} more")

            if Confirm.ask(f"Process {len(files)} files?"):
                results = []
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Processing files...", total=len(files))
                    for f in files:
                        result = extractor.extract_audio(str(f))
                        results.append(result)
                        progress.update(task, advance=1, description=f"Processing {f.name[:30]}...")

                success = sum(1 for r in results if r.success)
                self.console.print(f"[green]Processed {success}/{len(files)} files[/green]")

    def _transcribe_workflow(self):
        """Transcription workflow for existing audio files."""
        self.console.print(Panel("[bold yellow]Audio Transcription[/bold yellow]", border_style="yellow"))

        self.console.print("\n[bold]Transcription Type:[/bold]")
        self.console.print("  [1] Single file")
        self.console.print("  [2] Directory of audio files")
        self.console.print("  [0] Back to menu")

        choice = Prompt.ask("Select", choices=["0", "1", "2"], default="1")

        if choice == "0":
            return

        if choice == "1":
            file_path = Prompt.ask("Enter audio file path")
            if not file_path or not Path(file_path).exists():
                self.console.print("[red]File not found[/red]")
                return

            self._transcribe_file(Path(file_path))

        elif choice == "2":
            directory = Prompt.ask("Enter directory path", default=str(self.config.audio_dir))
            if not directory or not Path(directory).exists():
                self.console.print("[red]Directory not found[/red]")
                return

            # Find audio files
            audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
            files = [f for f in Path(directory).rglob("*") if f.suffix.lower() in audio_extensions]

            if not files:
                self.console.print("[red]No audio files found[/red]")
                return

            self.console.print(f"[green]Found {len(files)} audio files[/green]")

            if Confirm.ask(f"Transcribe {len(files)} files?"):
                self._batch_transcribe(files)

    def _transcribe_file(self, audio_path: Path, metadata: Optional[Dict[str, Any]] = None):
        """Transcribe a single audio file."""
        # Language selection
        self.console.print("\n[bold]Language Settings:[/bold]")
        self.console.print("  [1] Auto-detect")
        self.console.print("  [2] Specify language")

        lang_choice = Prompt.ask("Select", choices=["1", "2"], default="1")

        language = None
        if lang_choice == "2":
            language = Prompt.ask("Enter language code (e.g., en, fr, es)", default="en")

        # Output format FIRST
        self.console.print("\n[bold]Output Format:[/bold]")
        self.console.print("  [1] Text file (.txt) - with metadata header")
        self.console.print("  [2] CSV - structured with sentence IDs and speakers")
        self.console.print("  [3] JSON - full data export")
        self.console.print("  [4] SRT subtitles")
        self.console.print("  [5] WebVTT subtitles")

        format_choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5"], default="1")
        format_map = {"1": "txt", "2": "csv", "3": "json", "4": "srt", "5": "vtt"}
        output_format = format_map[format_choice]

        # Enable diarization?
        enable_diarization = Confirm.ask("Enable speaker diarization?", default=False)
        hf_token = None
        if enable_diarization:
            hf_token = self._get_hf_token()
            if not hf_token:
                self.console.print("[yellow]No HuggingFace token provided, diarization disabled[/yellow]")
                enable_diarization = False

        # Segmentation level - ONLY ask for CSV format or if diarization is enabled
        segmentation = "sentence"
        if output_format == "csv":
            self.console.print("\n[bold]Segmentation Level:[/bold]")
            self.console.print("  [1] 1 sentence per segment")
            self.console.print("  [2] 2 sentences per segment")
            self.console.print("  [3] 3 sentences per segment")
            self.console.print("  [4] Custom number of sentences")
            self.console.print("  [5] Paragraph (all sentences together)")

            seg_choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5"], default="1")

            if seg_choice == "4":
                n_sentences = IntPrompt.ask("Number of sentences per segment", default=5)
                segmentation = str(n_sentences)
            else:
                seg_map = {"1": "sentence", "2": "2", "3": "3", "5": "paragraph"}
                segmentation = seg_map.get(seg_choice, "sentence")

        # Transcribe
        self._do_transcription(
            audio_path=audio_path,
            language=language,
            segmentation=segmentation,
            output_format=output_format,
            enable_diarization=enable_diarization,
            metadata=metadata,
            hf_token=hf_token
        )

    def _do_transcription(
        self,
        audio_path: Path,
        language: Optional[str],
        segmentation: str,
        output_format: str,
        enable_diarization: bool,
        metadata: Optional[Dict[str, Any]] = None,
        transcriber: Optional['WhisperTranscriber'] = None,
        hf_token: Optional[str] = None
    ):
        """Perform transcription with progress display using tqdm."""
        from tqdm import tqdm

        # Use provided transcriber or create a new one
        own_transcriber = False
        if transcriber is None:
            own_transcriber = True
            transcriber_config = TranscriptionConfig(
                model_name=self.config.whisper_model,
                language=language,
                beam_size=self.config.whisper_beam_size,
                show_progress=True,  # Enable tqdm progress bars
            )

            try:
                transcriber = WhisperTranscriber(config=transcriber_config, logger=self.logger)
            except ImportError as e:
                self.console.print(f"[red]{e}[/red]")
                return None

            # Load model with spinner
            print("Loading Whisper model...", end=" ", flush=True)
            if not transcriber.load_model():
                print("FAILED")
                self.console.print("[red]Failed to load model[/red]")
                return None
            print("OK")

        # Get audio duration for display
        audio_duration = transcriber.get_audio_duration(audio_path)
        duration_str = f"{int(audio_duration // 60)}:{int(audio_duration % 60):02d}"

        self.console.print(f"[dim]Audio: {audio_path.name}[/dim]")
        self.console.print(f"[dim]Duration: {duration_str}[/dim]")

        # Transcribe - tqdm progress is now built into the transcriber
        try:
            result = transcriber.transcribe(audio_path, language)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            return None

        if not result or not result.success:
            self.console.print(f"[red]Transcription failed: {result.error if result else 'Unknown error'}[/red]")
            return None

        # Show result summary
        detected_lang = get_language_name(result.language)
        self.console.print(f"[green]✓ Transcribed[/green] • Language: {detected_lang} • Words: {len(result.words)}")

        # Speaker diarization - tqdm progress is now built into the diarizer
        if enable_diarization and result.words:
            try:
                from ..transcriber.diarization import SpeakerDiarizer, DiarizationConfig

                # Use provided token or fallback to environment variable
                token = hf_token or os.getenv("HF_TOKEN")
                diar_config = DiarizationConfig(
                    hf_token=token,
                    speaker_merge_threshold=self.config.speaker_merge_threshold,
                    show_progress=True,  # Enable tqdm progress bars
                )

                diarizer = SpeakerDiarizer(config=diar_config, logger=self.logger)

                # Diarization with built-in tqdm progress
                diar_result = diarizer.diarize(audio_path)

                if diar_result.success:
                    self.console.print(f"[green]✓ Diarization[/green] • Found {len(diar_result.speakers)} speakers")

                    # Apply to words
                    import pandas as pd
                    words_df = pd.DataFrame(result.words)
                    words_df['speaker'] = None
                    words_df = diarizer.align_words_with_speakers(words_df, diar_result)
                    result.words = words_df.to_dict('records')
                else:
                    self.console.print(f"[yellow]Diarization failed: {diar_result.error}[/yellow]")

            except ImportError:
                self.console.print("[yellow]Pyannote not available, skipping diarization[/yellow]")
            except Exception as e:
                self.console.print(f"[yellow]Diarization error: {e}[/yellow]")

        # Build metadata from file if not provided
        if metadata is None:
            metadata = self._extract_metadata_from_filename(audio_path)

        # Save output
        output_path = self.config.transcripts_dir / f"{audio_path.stem}.{output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "txt":
            content = TextProcessor.format_transcript_with_metadata(
                result.words,
                metadata=metadata,
                include_speakers=enable_diarization,
                language=result.language
            )
            output_path.write_text(content, encoding='utf-8')

        elif output_format == "csv":
            content = TextProcessor.format_csv(
                result.words,
                metadata=metadata,
                segmentation_level=segmentation,
                language=result.language
            )
            output_path.write_text(content, encoding='utf-8')

        elif output_format == "json":
            import json
            data = {
                'metadata': metadata,
                'audio_file': str(audio_path),
                'language': result.language,
                'duration': result.duration,
                'text': result.text,
                'words': result.words,
                'segments': result.segments
            }
            output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

        elif output_format == "srt":
            content = TextProcessor.format_srt(result.words, result.language)
            output_path.write_text(content, encoding='utf-8')

        elif output_format == "vtt":
            content = TextProcessor.format_vtt(result.words, result.language)
            output_path.write_text(content, encoding='utf-8')

        self.console.print(f"[green]Saved to: {output_path}[/green]")

        # Cleanup
        transcriber.cleanup()

        return result

    def _get_config_file_path(self) -> Path:
        """Get path to config file for fallback token storage."""
        config_dir = Path.home() / ".transcribe-tool"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"

    def _get_token_from_file(self) -> Optional[str]:
        """Get HuggingFace token from config file (fallback)."""
        try:
            config_path = self._get_config_file_path()
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config.get('hf_token')
        except Exception:
            pass
        return None

    def _save_token_to_file(self, token: str) -> bool:
        """Save HuggingFace token to config file (fallback)."""
        try:
            config_path = self._get_config_file_path()
            import json

            # Load existing config or create new
            config = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception:
                    pass

            config['hf_token'] = token

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Set restrictive permissions (owner read/write only)
            config_path.chmod(0o600)
            return True
        except Exception as e:
            self.logger.warning(f"Could not save token to file: {e}")
            return False

    def _delete_token_from_file(self) -> bool:
        """Delete HuggingFace token from config file."""
        try:
            config_path = self._get_config_file_path()
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if 'hf_token' in config:
                    del config['hf_token']
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
            return True
        except Exception:
            return False

    def _get_hf_token_from_keyring(self) -> Optional[str]:
        """Get HuggingFace token from system keychain or config file."""
        # Try keyring first
        try:
            import keyring
            token = keyring.get_password("transcribe-tool", "hf_token")
            if token:
                return token
        except ImportError:
            self.logger.debug("keyring not available")
        except Exception as e:
            self.logger.debug(f"keyring error: {e}")

        # Fallback to config file
        return self._get_token_from_file()

    def _save_hf_token_to_keyring(self, token: str) -> bool:
        """Save HuggingFace token to system keychain or config file."""
        saved_to_keyring = False

        # Try keyring first
        try:
            import keyring
            keyring.set_password("transcribe-tool", "hf_token", token)
            saved_to_keyring = True
        except ImportError:
            self.logger.debug("keyring not available, using config file")
        except Exception as e:
            self.logger.debug(f"keyring save error: {e}, using config file")

        # Always also save to file as backup
        saved_to_file = self._save_token_to_file(token)

        return saved_to_keyring or saved_to_file

    def _delete_hf_token_from_keyring(self) -> bool:
        """Delete HuggingFace token from system keychain and config file."""
        deleted_keyring = False
        deleted_file = False

        # Try to delete from keyring
        try:
            import keyring
            keyring.delete_password("transcribe-tool", "hf_token")
            deleted_keyring = True
        except Exception:
            pass

        # Also delete from config file
        deleted_file = self._delete_token_from_file()

        return deleted_keyring or deleted_file

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from keychain, config file, environment, or user input."""
        # 1. First check environment variable
        token = os.getenv("HF_TOKEN")
        if token:
            self.console.print("[dim]Using HF_TOKEN from environment[/dim]")
            return token

        # 2. Check system keychain / config file
        token = self._get_hf_token_from_keyring()
        if token:
            self.console.print("[dim]Using saved HuggingFace token[/dim]")
            return token

        # 3. Ask user for token
        self.console.print("\n[bold yellow]HuggingFace Token Required[/bold yellow]")
        self.console.print("[dim]Speaker diarization requires a HuggingFace token.[/dim]")
        self.console.print("[dim]1. Accept terms at: https://hf.co/pyannote/speaker-diarization-3.1[/dim]")
        self.console.print("[dim]2. Get token at: https://hf.co/settings/tokens[/dim]")
        self.console.print("[dim]3. Or set HF_TOKEN environment variable[/dim]\n")

        token = Prompt.ask("Enter HuggingFace token (or press Enter to skip)", default="", password=True)

        if token and token.startswith("hf_"):
            # Offer to save permanently
            if Confirm.ask("Save token for future sessions?", default=True):
                if self._save_hf_token_to_keyring(token):
                    self.console.print("[green]Token saved successfully[/green]")
                else:
                    # Fallback to session only
                    os.environ["HF_TOKEN"] = token
                    self.console.print("[yellow]Could not save token, using for this session only[/yellow]")
            else:
                os.environ["HF_TOKEN"] = token
                self.console.print("[dim]Token saved for this session only[/dim]")
            return token
        elif token:
            self.console.print("[yellow]Invalid token format (should start with 'hf_')[/yellow]")
            return None
        else:
            return None

    def _extract_metadata_from_filename(self, audio_path: Path) -> Dict[str, Any]:
        """Extract metadata from filename pattern YYYYMMDD_Title_VideoID.wav"""
        import re

        stem = audio_path.stem
        parent_name = audio_path.parent.name

        metadata = {
            'audio_file': audio_path.name,
            'title': stem,  # Default to full stem
            'date': None,
            'video_id': None,
            'channel': parent_name if parent_name not in ('audio', 'data') else 'Unknown',
            'source': 'local',
            'transcribed_at': datetime.now().isoformat()
        }

        # Pattern: YYYYMMDD_Title_VideoID
        # YouTube video IDs are 11 characters: [a-zA-Z0-9_-]
        # TikTok video IDs are typically 19 digits

        # Try to extract date at the beginning (YYYYMMDD_)
        date_match = re.match(r'^(\d{8})_(.+)$', stem)
        if date_match:
            metadata['date'] = date_match.group(1)
            remaining = date_match.group(2)
        else:
            remaining = stem

        # Try to extract video ID at the end (_VideoID)
        # YouTube: 11 chars alphanumeric with - and _
        # TikTok: 19 digits
        video_id_match = re.search(r'_([a-zA-Z0-9_-]{11})$|_(\d{19})$', remaining)
        if video_id_match:
            metadata['video_id'] = video_id_match.group(1) or video_id_match.group(2)
            # Title is everything before the video ID
            title = remaining[:video_id_match.start()]
            if title:
                # Clean up the title: replace underscores with spaces, clean up
                title = title.strip('_').replace('_', ' ')
                # Handle full-width characters
                title = title.replace('：', ':').replace('　', ' ')
                metadata['title'] = title
        else:
            # No video ID found, use remaining as title
            if remaining != stem:
                metadata['title'] = remaining.replace('_', ' ')

        return metadata

    def _get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return self.config.transcripts_dir / ".pipeline_checkpoint.json"

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load existing checkpoint if available."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            try:
                import json
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                return checkpoint
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def _save_checkpoint(
        self,
        audio_files: List[Path],
        metadata_list: List[Dict],
        completed: List[str],
        failed: List[str],
        settings: Dict[str, Any]
    ):
        """Save checkpoint for resume functionality."""
        checkpoint_path = self._get_checkpoint_path()
        try:
            import json
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'audio_files': [str(f) for f in audio_files],
                'metadata_list': metadata_list,
                'completed': completed,
                'failed': failed,
                'settings': settings
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def _clear_checkpoint(self):
        """Clear checkpoint file after successful completion."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except Exception:
                pass

    def _full_pipeline_workflow(self):
        """Full pipeline: Extract → Transcribe → Export."""
        self.console.print(Panel(
            "[bold green]Full Pipeline[/bold green]\n"
            "[dim]Extract audio → Transcribe → Export to CSV[/dim]",
            border_style="green"
        ))

        # Check for existing checkpoint
        checkpoint = self._load_checkpoint()
        if checkpoint:
            remaining = len(checkpoint['audio_files']) - len(checkpoint['completed'])
            self.console.print(f"\n[yellow]Found checkpoint with {remaining} files remaining[/yellow]")
            self.console.print(f"  [dim]Started: {checkpoint['timestamp']}[/dim]")
            self.console.print(f"  [dim]Completed: {len(checkpoint['completed'])}, Failed: {len(checkpoint['failed'])}[/dim]")

            if Confirm.ask("Resume from checkpoint?", default=True):
                # Resume from checkpoint
                audio_files = [Path(f) for f in checkpoint['audio_files']]
                metadata_list = checkpoint['metadata_list']
                completed = set(checkpoint['completed'])
                failed = list(checkpoint['failed'])
                settings = checkpoint['settings']

                # Filter out completed files
                remaining_indices = [i for i, f in enumerate(audio_files) if str(f) not in completed]
                audio_files = [audio_files[i] for i in remaining_indices]
                metadata_list = [metadata_list[i] for i in remaining_indices]

                language = settings.get('language')
                output_format = settings.get('output_format', 'csv')
                enable_diarization = settings.get('enable_diarization', True)
                segmentation = settings.get('segmentation', 'sentence')

                # Request HF token if diarization is enabled
                hf_token = None
                if enable_diarization:
                    hf_token = self._get_hf_token()
                    if not hf_token:
                        self.console.print("[yellow]No HuggingFace token, diarization disabled[/yellow]")
                        enable_diarization = False

                # Skip to transcription
                self.console.print(f"\n[green]Resuming with {len(audio_files)} remaining files...[/green]")
                self._run_transcription_loop(
                    audio_files, metadata_list, language, output_format,
                    enable_diarization, segmentation, completed, failed, hf_token
                )
                return
            else:
                self._clear_checkpoint()

        # Select source
        self.console.print("\n[bold]Source:[/bold]")
        self.console.print("  [1] YouTube channel/playlist/video")
        self.console.print("  [2] TikTok profile/video")
        self.console.print("  [3] Local directory")
        self.console.print("  [0] Back to menu")

        source_choice = Prompt.ask("Select", choices=["0", "1", "2", "3"], default="1")

        if source_choice == "0":
            return

        # Gather URLs/files based on source
        audio_files = []
        metadata_list = []

        if source_choice == "1":
            # YouTube
            extractor_config = ExtractorConfig(
                output_dir=self.config.audio_dir,
                audio_format=self.config.audio_format,
                cookies_from_browser=self.config.cookies_from_browser
            )

            try:
                extractor = YouTubeExtractor(config=extractor_config, logger=self.logger)
            except ImportError as e:
                self.console.print(f"[red]{e}[/red]")
                return

            self.console.print("\n[bold]YouTube Source Type:[/bold]")
            self.console.print("  [1] Single video")
            self.console.print("  [2] Channel (all videos)")
            self.console.print("  [3] Playlist")

            yt_choice = Prompt.ask("Select", choices=["1", "2", "3"], default="1")

            if yt_choice == "1":
                url = Prompt.ask("Enter YouTube video URL")
                urls = [url] if url else []
            elif yt_choice == "2":
                channel_url = Prompt.ask("Enter YouTube channel URL")
                max_videos = IntPrompt.ask("Maximum videos", default=50)
                self.console.print("[yellow]Extracting video URLs...[/yellow]")
                urls = extractor.extract_channel_videos(channel_url, max_videos=max_videos)
            else:
                playlist_url = Prompt.ask("Enter YouTube playlist URL")
                self.console.print("[yellow]Extracting playlist...[/yellow]")
                urls = extractor.extract_playlist(playlist_url)

            if not urls:
                self.console.print("[red]No videos found[/red]")
                return

            self.console.print(f"[green]Found {len(urls)} videos[/green]")

            # Download all
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Downloading audio...", total=len(urls))
                for url in urls:
                    result = extractor.extract_audio(url)
                    if result.success and result.audio_path:
                        audio_files.append(result.audio_path)
                        # Extract date from upload_date or filename
                        date = result.metadata.get('upload_date', '')
                        if not date:
                            # Try to extract from filename (YYYYMMDD_Title_ID.wav)
                            import re
                            date_match = re.search(r'(\d{8})', result.audio_path.stem)
                            date = date_match.group(1) if date_match else datetime.now().strftime('%Y%m%d')
                        metadata_list.append({
                            'video_id': result.video_id,
                            'title': result.title,
                            'channel': result.metadata.get('channel', result.metadata.get('uploader', 'Unknown')),
                            'date': date,
                            'audio_file': result.audio_path.name,
                            'source': 'youtube',
                            'transcribed_at': datetime.now().isoformat()
                        })
                    progress.update(task, advance=1)

        elif source_choice == "2":
            # TikTok
            extractor_config = ExtractorConfig(
                output_dir=self.config.audio_dir,
                audio_format=self.config.audio_format,
                cookies_from_browser=self.config.cookies_from_browser
            )

            try:
                extractor = TikTokExtractor(config=extractor_config, logger=self.logger)
            except ImportError as e:
                self.console.print(f"[red]{e}[/red]")
                return

            self.console.print("\n[bold]TikTok Source Type:[/bold]")
            self.console.print("  [1] Single video")
            self.console.print("  [2] User profile")

            tt_choice = Prompt.ask("Select", choices=["1", "2"], default="1")

            if tt_choice == "1":
                url = Prompt.ask("Enter TikTok video URL")
                urls = [url] if url else []
            else:
                user_url = Prompt.ask("Enter TikTok user profile URL")
                max_videos = IntPrompt.ask("Maximum videos", default=30)
                urls = extractor.extract_user_videos(user_url, max_videos=max_videos)

            if not urls:
                self.console.print("[red]No videos found[/red]")
                return

            self.console.print(f"[green]Found {len(urls)} videos[/green]")

            # Download all
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Downloading audio...", total=len(urls))
                for url in urls:
                    result = extractor.extract_audio(url)
                    if result.success and result.audio_path:
                        audio_files.append(result.audio_path)
                        # Extract date from upload_date or filename
                        date = result.metadata.get('upload_date', '')
                        if not date:
                            import re
                            date_match = re.search(r'(\d{8})', result.audio_path.stem)
                            date = date_match.group(1) if date_match else datetime.now().strftime('%Y%m%d')
                        metadata_list.append({
                            'video_id': result.video_id,
                            'title': result.title,
                            'channel': result.metadata.get('uploader', 'Unknown'),
                            'date': date,
                            'audio_file': result.audio_path.name,
                            'source': 'tiktok',
                            'transcribed_at': datetime.now().isoformat()
                        })
                    progress.update(task, advance=1)

        elif source_choice == "3":
            # Local directory
            directory = Prompt.ask("Enter directory path", default=str(self.config.audio_dir))
            if not directory or not Path(directory).exists():
                self.console.print("[red]Directory not found[/red]")
                return

            audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
            audio_files = [f for f in Path(directory).rglob("*") if f.suffix.lower() in audio_extensions]
            metadata_list = [None] * len(audio_files)  # Will be extracted from filenames

            if not audio_files:
                self.console.print("[red]No audio files found[/red]")
                return

            self.console.print(f"[green]Found {len(audio_files)} audio files[/green]")

        if not audio_files:
            self.console.print("[red]No audio files to process[/red]")
            return

        # Configure transcription options
        self.console.print("\n[bold]Transcription Settings:[/bold]")

        # Language
        self.console.print("\n[bold]Language:[/bold]")
        self.console.print("  [1] Auto-detect")
        self.console.print("  [2] Specify language")
        lang_choice = Prompt.ask("Select", choices=["1", "2"], default="1")
        language = None
        if lang_choice == "2":
            language = Prompt.ask("Enter language code (e.g., en, fr, es)", default="en")

        # Output format
        self.console.print("\n[bold]Output Format:[/bold]")
        self.console.print("  [1] CSV (with sentence IDs and speakers)")
        self.console.print("  [2] Text file (.txt) with metadata")
        self.console.print("  [3] JSON")

        format_choice = Prompt.ask("Select", choices=["1", "2", "3"], default="1")
        format_map = {"1": "csv", "2": "txt", "3": "json"}
        output_format = format_map[format_choice]

        # Diarization
        enable_diarization = Confirm.ask("Enable speaker diarization?", default=True)
        hf_token = None
        if enable_diarization:
            hf_token = self._get_hf_token()
            if not hf_token:
                self.console.print("[yellow]No HuggingFace token provided, diarization disabled[/yellow]")
                enable_diarization = False

        # Segmentation (for CSV)
        segmentation = "sentence"
        if output_format == "csv":
            self.console.print("\n[bold]Segmentation Level:[/bold]")
            self.console.print("  [1] 1 sentence per segment")
            self.console.print("  [2] 2 sentences per segment")
            self.console.print("  [3] 3 sentences per segment")
            self.console.print("  [4] Paragraph")

            seg_choice = Prompt.ask("Select", choices=["1", "2", "3", "4"], default="1")
            seg_map = {"1": "sentence", "2": "2", "3": "3", "4": "paragraph"}
            segmentation = seg_map[seg_choice]

        # Run transcription loop with checkpoint support
        self._run_transcription_loop(
            audio_files=audio_files,
            metadata_list=metadata_list,
            language=language,
            output_format=output_format,
            enable_diarization=enable_diarization,
            segmentation=segmentation,
            completed=set(),
            failed=[],
            hf_token=hf_token
        )

    def _run_transcription_loop(
        self,
        audio_files: List[Path],
        metadata_list: List[Dict],
        language: Optional[str],
        output_format: str,
        enable_diarization: bool,
        segmentation: str,
        completed: set,
        failed: List[str],
        hf_token: Optional[str] = None
    ):
        """Run the transcription loop with checkpoint support and tqdm progress."""
        from tqdm import tqdm

        if not audio_files:
            self.console.print("[yellow]No files to process[/yellow]")
            return

        # Save initial checkpoint
        all_audio_files = audio_files.copy()
        all_metadata_list = metadata_list.copy()
        settings = {
            'language': language,
            'output_format': output_format,
            'enable_diarization': enable_diarization,
            'segmentation': segmentation
        }

        self.console.print(f"\n[bold]Processing {len(audio_files)} files...[/bold]")

        # Load transcriber once
        transcriber_config = TranscriptionConfig(
            model_name=self.config.whisper_model,
            language=language,
            beam_size=self.config.whisper_beam_size,
            show_progress=True,  # Enable tqdm progress bars
        )

        try:
            transcriber = WhisperTranscriber(config=transcriber_config, logger=self.logger)
        except ImportError as e:
            self.console.print(f"[red]{e}[/red]")
            return

        # Load model
        print("Loading Whisper model...", end=" ", flush=True)
        if not transcriber.load_model():
            print("FAILED")
            self.console.print("[red]Failed to load model[/red]")
            return
        print("OK")

        self.console.print(f"[green]✓ Model loaded:[/green] {self.config.whisper_model}")
        self.console.print()

        # Process each file with general progress bar
        success_count = len(completed)
        total_files = len(audio_files) + len(completed)

        # Create general progress bar for pipeline
        pipeline_pbar = tqdm(
            total=len(audio_files),
            desc="Pipeline",
            unit="file",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]'
        )

        for i, audio_path in enumerate(audio_files):
            file_num = len(completed) + i + 1

            # Update pipeline progress bar description
            pipeline_pbar.set_postfix_str(f"{audio_path.name[:30]}...")

            print(f"\n{'━' * 50}")
            print(f"File {file_num}/{total_files}: {audio_path.name}")
            print(f"{'━' * 50}")

            metadata = metadata_list[i] if i < len(metadata_list) and metadata_list[i] else None
            if metadata is None:
                metadata = self._extract_metadata_from_filename(audio_path)

            try:
                result = self._do_transcription(
                    audio_path=audio_path,
                    language=language,
                    segmentation=segmentation,
                    output_format=output_format,
                    enable_diarization=enable_diarization,
                    metadata=metadata,
                    transcriber=transcriber,
                    hf_token=hf_token
                )

                if result:
                    success_count += 1
                    completed.add(str(audio_path))
                else:
                    failed.append(str(audio_path))

            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                failed.append(str(audio_path))

            # Update pipeline progress bar
            pipeline_pbar.update(1)

            # Save checkpoint after each file
            self._save_checkpoint(
                all_audio_files, all_metadata_list,
                list(completed), failed, settings
            )

        pipeline_pbar.close()
        transcriber.cleanup()

        # Clear checkpoint on successful completion
        print()
        if len(failed) == 0:
            self._clear_checkpoint()
            self.console.print(f"\n[bold green]Pipeline Complete![/bold green]")
        else:
            self.console.print(f"\n[bold yellow]Pipeline Complete with errors[/bold yellow]")
            self.console.print(f"[red]Failed files: {len(failed)}[/red]")

        self.console.print(f"Processed: {success_count}/{total_files} files")
        self.console.print(f"Output directory: {self.config.transcripts_dir}")

    def _batch_download(self, extractor, urls: List[str]):
        """Download audio from multiple URLs."""
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Downloading...", total=len(urls))

            for url in urls:
                result = extractor.extract_audio(url)
                results.append(result)
                status = "OK" if result.success else "FAIL"
                progress.update(task, advance=1, description=f"[{status}] {url[:50]}...")

        success = sum(1 for r in results if r.success)
        self.console.print(f"[green]Downloaded {success}/{len(urls)} files[/green]")

        if Confirm.ask("Transcribe downloaded files?"):
            audio_files = [r.audio_path for r in results if r.success and r.audio_path]
            self._batch_transcribe(audio_files)

    def _batch_transcribe(self, audio_files: List[Path]):
        """Transcribe multiple audio files."""
        # Quick settings
        language = Prompt.ask("Language (or 'auto')", default="auto")
        if language == "auto":
            language = None

        transcriber_config = TranscriptionConfig(
            model_name=self.config.whisper_model,
            language=language,
        )

        try:
            transcriber = WhisperTranscriber(config=transcriber_config, logger=self.logger)
            transcriber.load_model()
        except Exception as e:
            self.console.print(f"[red]Failed to load transcriber: {e}[/red]")
            return

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Transcribing...", total=len(audio_files))

            for audio_path in audio_files:
                result = transcriber.transcribe(audio_path, language)
                results.append((audio_path, result))

                if result.success:
                    # Save text file
                    output_path = self.config.transcripts_dir / f"{audio_path.stem}.txt"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    content = TextProcessor.format_transcript(
                        result.words,
                        include_speakers=False,
                        language=result.language
                    )
                    output_path.write_text(content, encoding='utf-8')

                status = "OK" if result.success else "FAIL"
                progress.update(task, advance=1, description=f"[{status}] {audio_path.name[:30]}...")

        success = sum(1 for _, r in results if r.success)
        self.console.print(f"[green]Transcribed {success}/{len(audio_files)} files[/green]")
        self.console.print(f"[green]Output directory: {self.config.transcripts_dir}[/green]")

        transcriber.cleanup()

    def _settings_menu(self):
        """Settings configuration menu."""
        self.console.print(Panel("[bold]Settings[/bold]", border_style="blue"))

        while True:
            # Check HuggingFace token status
            hf_token_status = "Not configured"
            hf_token = self._get_hf_token_from_keyring()
            if hf_token:
                hf_token_status = f"Saved (hf_...{hf_token[-4:]})"
            elif os.getenv("HF_TOKEN"):
                hf_token_status = "From environment"

            table = Table(title="Current Settings", box=box.ROUNDED)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Whisper Model", self.config.whisper_model)
            table.add_row("Audio Format", self.config.audio_format)
            table.add_row("Device", self.config.device)
            table.add_row("Output Dir", str(self.config.transcripts_dir))
            table.add_row("Cookies Browser", self.config.cookies_from_browser or "None")
            table.add_row("HuggingFace Token", hf_token_status)

            self.console.print(table)
            self.console.print()

            self.console.print("[1] Change Whisper model")
            self.console.print("[2] Change audio format")
            self.console.print("[3] Set browser for cookies")
            self.console.print("[4] Manage HuggingFace token")
            self.console.print("[0] Back to main menu")

            choice = Prompt.ask("Select", choices=["0", "1", "2", "3", "4"], default="0")

            if choice == "0":
                break
            elif choice == "1":
                models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
                self.console.print(f"Available models: {', '.join(models)}")
                new_model = Prompt.ask("Model", default=self.config.whisper_model)
                if new_model in models:
                    self.config.whisper_model = new_model
                    self.console.print(f"[green]Model set to: {new_model}[/green]")
            elif choice == "2":
                formats = ["wav", "mp3"]
                new_format = Prompt.ask("Format", choices=formats, default=self.config.audio_format)
                self.config.audio_format = new_format
                self.console.print(f"[green]Format set to: {new_format}[/green]")
            elif choice == "3":
                browsers = ["chrome", "firefox", "safari", "edge", "none"]
                new_browser = Prompt.ask("Browser", choices=browsers, default="none")
                self.config.cookies_from_browser = None if new_browser == "none" else new_browser
                self.console.print(f"[green]Browser set to: {new_browser}[/green]")
            elif choice == "4":
                self._manage_hf_token()

    def _tokenization_workflow(self):
        """Text tokenization workflow for documents."""
        self.console.print(Panel(
            "[bold magenta]Text Tokenization[/bold magenta]\n"
            "[dim]SOTA NLP: Segmentation + Lemmatization + POS + NER → CSV/JSON[/dim]",
            border_style="magenta"
        ))

        # Select input source
        self.console.print("\n[bold]1. Input Source:[/bold]")
        self.console.print("  [1] Single file (txt, pdf, docx, etc.)")
        self.console.print("  [2] Directory of documents")
        self.console.print("  [3] Paste text directly")
        self.console.print("  [0] Back to menu")

        source_choice = Prompt.ask("Select", choices=["0", "1", "2", "3"], default="1")

        if source_choice == "0":
            return

        documents: List[Document] = []
        loader = DocumentLoader(logger=self.logger)

        if source_choice == "1":
            # Single file
            file_path = Prompt.ask("Enter file path")
            if not file_path or not Path(file_path).exists():
                self.console.print("[red]File not found[/red]")
                return

            self.console.print(f"[dim]Supported: {', '.join(loader.get_supported_formats())}[/dim]")

            doc = loader.load(file_path)
            if doc.success:
                documents.append(doc)
                self.console.print(f"[green]Loaded: {doc.path.name}[/green]")
                self.console.print(f"  [dim]Words: {doc.word_count:,} | Chars: {doc.char_count:,}[/dim]")
            else:
                self.console.print(f"[red]Failed to load: {doc.error}[/red]")
                return

        elif source_choice == "2":
            # Directory
            directory = Prompt.ask("Enter directory path")
            if not directory or not Path(directory).exists():
                self.console.print("[red]Directory not found[/red]")
                return

            self.console.print("\n[bold]File Types:[/bold]")
            self.console.print("  [1] All supported formats")
            self.console.print("  [2] Text files only (.txt)")
            self.console.print("  [3] PDF files only (.pdf)")
            self.console.print("  [4] Word documents only (.docx)")
            self.console.print("  [5] Custom extensions")

            type_choice = Prompt.ask("Select", choices=["1", "2", "3", "4", "5"], default="1")

            extensions = None
            if type_choice == "2":
                extensions = ['.txt']
            elif type_choice == "3":
                extensions = ['.pdf']
            elif type_choice == "4":
                extensions = ['.docx']
            elif type_choice == "5":
                ext_input = Prompt.ask("Enter extensions (comma-separated)", default=".txt,.pdf,.docx")
                extensions = [e.strip() for e in ext_input.split(',')]

            recursive = Confirm.ask("Search subdirectories?", default=True)

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
                task = progress.add_task("Loading documents...", total=None)
                result = loader.load_directory(directory, recursive=recursive, extensions=extensions)
                progress.update(task, completed=True)

            documents = [d for d in result.documents if d.success]

            self.console.print(f"\n[green]Loaded {result.successful}/{result.total_files} documents[/green]")
            if not documents:
                self.console.print("[red]No documents to process[/red]")
                return

            total_words = sum(d.word_count for d in documents)
            self.console.print(f"[dim]Total words: {total_words:,}[/dim]")

        elif source_choice == "3":
            # Direct text input
            self.console.print("\n[dim]Enter/paste text (press Enter twice to finish):[/dim]")

            lines = []
            empty_count = 0
            while empty_count < 1:
                try:
                    line = input()
                    if line == "":
                        empty_count += 1
                    else:
                        empty_count = 0
                        lines.append(line)
                except EOFError:
                    break

            text = '\n'.join(lines)
            if not text.strip():
                self.console.print("[red]No text provided[/red]")
                return

            doc = loader.load_text(text, source_name="direct_input")
            documents.append(doc)
            self.console.print(f"[green]Text loaded: {doc.word_count:,} words[/green]")

        if not documents:
            return

        # Language selection with auto-detection
        self.console.print("\n[bold]2. Language:[/bold]")
        languages = get_nlp_languages()

        for i, lang in enumerate(languages[:10], 1):
            self.console.print(f"  [{i:2}] {lang['name']} ({lang['code']})")
        self.console.print("  [11] Auto-detect (recommended)")
        self.console.print("  [12] Other (enter code)")

        lang_choice = Prompt.ask("Select", default="11")

        language = 'en'
        if lang_choice == "11":
            # Auto-detect using LanguageDetector
            try:
                detector = LanguageDetector()
                sample_text = documents[0].content[:2000]
                result = detector.detect(sample_text, return_all_scores=True)
                language = result['language']
                confidence = result['confidence']
                method = result['method']
                self.console.print(f"[green]Detected: {get_language_name(language)} ({language})[/green]")
                self.console.print(f"  [dim]Confidence: {confidence:.1%} | Method: {method}[/dim]")
            except Exception as e:
                language = 'en'
                self.console.print(f"[yellow]Detection failed ({e}), using English[/yellow]")
        elif lang_choice == "12":
            language = Prompt.ask("Enter language code (e.g., en, fr, de)", default="en")
        elif lang_choice.isdigit() and 1 <= int(lang_choice) <= 10:
            language = languages[int(lang_choice) - 1]['code']

        # Segmentation level
        self.console.print("\n[bold]3. Segmentation Level:[/bold]")
        self.console.print("  [1] 1 sentence per segment")
        self.console.print("  [2] 2 sentences per segment")
        self.console.print("  [3] 3 sentences per segment")
        self.console.print("  [4] Custom N sentences per segment")
        self.console.print("  [0] Paragraph mode (keep paragraphs intact)")

        seg_choice = Prompt.ask("Select", choices=["0", "1", "2", "3", "4"], default="1")

        segmentation_level = 1
        if seg_choice == "0":
            segmentation_level = 0  # Paragraph mode
        elif seg_choice == "1":
            segmentation_level = 1
        elif seg_choice == "2":
            segmentation_level = 2
        elif seg_choice == "3":
            segmentation_level = 3
        elif seg_choice == "4":
            segmentation_level = IntPrompt.ask("Number of sentences per segment", default=5)

        # NLP Features selection
        self.console.print("\n[bold]4. NLP Features:[/bold] (select multiple with comma)")
        self.console.print("  [1] Lemmatization (word base forms)")
        self.console.print("  [2] POS Tagging (part-of-speech)")
        self.console.print("  [3] NER (named entities: people, places, orgs)")
        self.console.print("  [4] All features (lemma + POS + NER)")
        self.console.print("  [0] None (text segmentation only)")

        nlp_choice = Prompt.ask("Select", default="4")

        lemmatization = False
        pos_tagging = False
        ner = False

        if nlp_choice == "4":
            lemmatization = pos_tagging = ner = True
        elif nlp_choice != "0":
            choices = [c.strip() for c in nlp_choice.split(',')]
            if "1" in choices:
                lemmatization = True
            if "2" in choices:
                pos_tagging = True
            if "3" in choices:
                ner = True

        nlp_features = NLPFeatures(
            lemmatization=lemmatization,
            pos_tagging=pos_tagging,
            ner=ner
        )

        # Show selected features
        features_str = []
        if lemmatization:
            features_str.append("Lemma")
        if pos_tagging:
            features_str.append("POS")
        if ner:
            features_str.append("NER")
        if features_str:
            self.console.print(f"  [dim]Selected: {', '.join(features_str)}[/dim]")
        else:
            self.console.print("  [dim]Selected: Segmentation only[/dim]")

        # Output format
        self.console.print("\n[bold]5. Output Format:[/bold]")
        self.console.print("  [1] CSV (with JSON-encoded structured data)")
        self.console.print("  [2] JSON (full structured export)")

        format_choice = Prompt.ask("Select", choices=["1", "2"], default="1")
        output_format = "csv" if format_choice == "1" else "json"

        # Process documents
        self.console.print(f"\n[bold]Processing {len(documents)} document(s)...[/bold]")

        # Initialize tokenizer
        tokenizer = TextTokenizer(
            language=language,
            segmentation_level=segmentation_level,
            nlp_features=nlp_features,
            use_gpu=True,
            logger=self.logger
        )

        results: List[TokenizationResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Tokenizing...", total=len(documents))

            for idx, doc in enumerate(documents, 1):
                doc_id = doc.path.stem if doc.path else f"doc_{idx}"
                progress.update(task, description=f"Processing {doc_id}...")

                result = tokenizer.tokenize(
                    text=doc.content,
                    document_id=doc_id,
                    source_file=str(doc.path) if doc.path else None
                )
                results.append(result)
                progress.update(task, advance=1)

        # Summary
        successful_results = [r for r in results if r.success]
        total_segments = sum(len(r.segments) for r in successful_results)
        total_sentences = sum(r.total_sentences for r in successful_results)
        total_words = sum(r.total_words for r in successful_results)
        total_entities = sum(r.total_entities for r in successful_results)
        failed = len(results) - len(successful_results)

        self.console.print(f"\n[green]Tokenization complete![/green]")
        self.console.print(f"  Documents: {len(successful_results)}/{len(results)}")
        self.console.print(f"  Segments: {total_segments:,}")
        self.console.print(f"  Sentences: {total_sentences:,}")
        self.console.print(f"  Words: {total_words:,}")
        if ner and total_entities > 0:
            self.console.print(f"  Entities: {total_entities:,}")

        # Generate output
        if output_format == "csv":
            output_content = format_tokenization_csv(results)
            ext = "csv"
        else:
            output_content = format_tokenization_json(results)
            ext = "json"

        # Save output
        output_dir = self.config.transcripts_dir / "tokenization"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(documents) == 1 and documents[0].path:
            base_name = documents[0].path.stem
        else:
            base_name = f"tokenized_{len(documents)}_docs"

        seg_str = f"seg{segmentation_level}" if segmentation_level > 0 else "para"
        output_path = output_dir / f"{base_name}_{language}_{seg_str}_{timestamp}.{ext}"
        output_path.write_text(output_content, encoding='utf-8')

        self.console.print(f"\n[green]Output saved to:[/green] {output_path}")

        # Show preview
        if Confirm.ask("Show preview?", default=True):
            if successful_results:
                preview_table = Table(title="Preview (first 10 segments)", box=box.ROUNDED)
                preview_table.add_column("ID", style="cyan", width=4)
                preview_table.add_column("Doc", style="magenta", width=15)
                preview_table.add_column("Text", style="white", max_width=50)
                preview_table.add_column("Sent", style="yellow", width=4)
                preview_table.add_column("Words", style="green", width=5)
                if ner:
                    preview_table.add_column("Entities", style="blue", width=6)

                segment_count = 0
                for result in successful_results:
                    for seg in result.segments:
                        if segment_count >= 10:
                            break
                        text_preview = seg.text[:60] + "..." if len(seg.text) > 60 else seg.text
                        row = [
                            str(seg.segment_id),
                            seg.document_id[:15],
                            text_preview,
                            str(seg.sentence_count),
                            str(seg.word_count)
                        ]
                        if ner:
                            row.append(str(len(seg.entities)))
                        preview_table.add_row(*row)
                        segment_count += 1
                    if segment_count >= 10:
                        break

                self.console.print(preview_table)

    def _manage_hf_token(self):
        """Manage HuggingFace token storage."""
        self.console.print("\n[bold]HuggingFace Token Management[/bold]")

        # Check current status
        saved_token = self._get_hf_token_from_keyring()
        env_token = os.getenv("HF_TOKEN")

        if saved_token:
            self.console.print(f"[green]Token saved:[/green] hf_...{saved_token[-4:]}")
            config_path = self._get_config_file_path()
            self.console.print(f"[dim]Storage: {config_path}[/dim]")
        elif env_token:
            self.console.print(f"[yellow]Token from environment:[/yellow] hf_...{env_token[-4:]}")
        else:
            self.console.print("[dim]No token configured[/dim]")

        self.console.print()
        self.console.print("[1] Add/Update token")
        self.console.print("[2] Delete saved token")
        self.console.print("[0] Back")

        choice = Prompt.ask("Select", choices=["0", "1", "2"], default="0")

        if choice == "1":
            self.console.print("\n[dim]Get your token at: https://hf.co/settings/tokens[/dim]")
            token = Prompt.ask("Enter HuggingFace token", password=True)
            if token and token.startswith("hf_"):
                if self._save_hf_token_to_keyring(token):
                    self.console.print("[green]Token saved successfully[/green]")
                else:
                    self.console.print("[red]Failed to save token[/red]")
            elif token:
                self.console.print("[yellow]Invalid token format (should start with 'hf_')[/yellow]")
        elif choice == "2":
            if saved_token:
                if Confirm.ask("Delete saved token?", default=False):
                    if self._delete_hf_token_from_keyring():
                        self.console.print("[green]Token deleted[/green]")
                    else:
                        self.console.print("[red]Failed to delete token[/red]")
            else:
                self.console.print("[dim]No token to delete[/dim]")

    def _exit(self):
        """Exit the application."""
        self.console.print("\n[cyan]Thank you for using Transcribe Tool![/cyan]")
        self.console.print("[dim]Goodbye![/dim]\n")


def run_cli():
    """Entry point for the CLI."""
    cli = TranscribeCLI()
    cli.run()


if __name__ == "__main__":
    run_cli()

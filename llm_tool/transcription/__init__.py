"""
Transcribe Tool - Unified Audio Extraction and Transcription

A comprehensive CLI tool for extracting audio from YouTube, TikTok,
and local files, then transcribing them using Whisper with optional
speaker diarization and intelligent tokenization.

Author: Antoine Lemor
"""

# Suppress noisy warnings at package import
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "1.0.0"
__author__ = "Antoine Lemor"

from pathlib import Path

# Package directories
PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = PACKAGE_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Ensure directories exist
for dir_path in [DATA_DIR, AUDIO_DIR, TRANSCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

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

from ..platform_compat import writable_dir

# Output directories.
#
# These follow the working directory, like every other output path in the
# package, rather than the directory the code was installed into: a wheel in
# site-packages -- or anywhere under C:\Program Files -- sits in a tree the user
# cannot write to, and importing this module would raise PermissionError before
# any transcription began.
PACKAGE_DIR = Path(__file__).parent
ROOT_DIR = Path.cwd()
DATA_DIR = writable_dir(
    [ROOT_DIR / "data", Path.home() / ".llm_tool" / "data"],
    prefix="llm_tool_data",
)
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

for dir_path in [AUDIO_DIR, TRANSCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

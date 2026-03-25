"""
Configuration settings for Transcribe Tool.

Manages all configurable parameters for audio extraction,
transcription, and processing.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


def detect_device() -> tuple[str, bool]:
    """Detect the best available device for processing."""
    system = platform.system()

    # Check CUDA
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda", True
        except Exception:
            pass

    # Check MPS (Apple Silicon)
    if system == "Darwin":
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                test_tensor = torch.zeros(1).to('mps')
                del test_tensor
                return "mps", True
        except Exception:
            pass

    return "cpu", False


@dataclass
class Config:
    """Central configuration for Transcribe Tool."""

    # Base directories
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    # Audio settings
    audio_format: str = "wav"
    audio_quality: str = "192"  # kbps for mp3, or 'best' for wav

    # Whisper settings
    whisper_model: str = "large-v3"
    whisper_language: Optional[str] = None  # Auto-detect if None
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0

    # Diarization settings
    enable_diarization: bool = False
    speaker_merge_threshold: float = 0.90
    hf_token: Optional[str] = None

    # Audio processing
    enable_vocal_separation: bool = False
    low_pass_freq: int = 3200
    high_pass_freq: int = 65
    min_silence_len: int = 3000  # ms
    silence_thresh: int = -60  # dB

    # Tokenization settings
    tokenization_level: str = "sentence"  # sentence, paragraph, or number (1, 2, 3...)
    sentences_per_segment: int = 1

    # Processing settings
    num_workers: int = field(default_factory=lambda: max(1, os.cpu_count() - 1))
    batch_size: int = 1

    # Device settings
    device: str = field(default_factory=lambda: detect_device()[0])
    use_fp16: bool = field(default_factory=lambda: detect_device()[0] == "cuda")

    # YouTube/TikTok settings
    cookies_from_browser: Optional[str] = None  # chrome, firefox, edge, safari
    use_proxy: bool = False
    proxy_url: Optional[str] = None

    # Output settings
    output_format: str = "json"  # json, csv, txt, srt, vtt
    include_timestamps: bool = True
    include_speakers: bool = True

    def __post_init__(self):
        """Initialize computed fields."""
        self.data_dir = self.base_dir / "data"
        self.audio_dir = self.data_dir / "audio"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.cache_dir = self.data_dir / "cache"
        self.logs_dir = self.data_dir / "logs"

        # Create directories
        for dir_path in [self.audio_dir, self.transcripts_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Adjust device settings
        # Note: MPS now works with Whisper via HuggingFace Transformers
        if self.device == "mps":
            self.whisper_device = "mps"  # Uses HuggingFace Transformers for MPS
            self.use_fp16 = False  # MPS doesn't support fp16 well
        elif self.device == "cuda":
            self.whisper_device = "cuda"
            self.use_fp16 = True
        else:
            self.whisper_device = "cpu"
            self.use_fp16 = False

    @property
    def supported_languages(self) -> List[str]:
        """List of supported language codes."""
        return [
            'en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'tr', 'vi', 'id', 'th', 'sv', 'no', 'da', 'fi', 'cs', 'el',
            'he', 'uk', 'ro', 'hu', 'bg', 'hr', 'sk', 'sl', 'lt', 'lv', 'et', 'ca'
        ]


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config

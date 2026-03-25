"""
Base extractor class for audio sources.

Provides common functionality for all audio extractors.
"""

import os
import random
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExtractionResult:
    """Result of an audio extraction operation."""
    success: bool
    audio_path: Optional[Path] = None
    video_id: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExtractorConfig:
    """Configuration for extractors."""
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "audio")
    audio_format: str = "wav"
    audio_quality: str = "192"
    cookies_from_browser: Optional[str] = None
    use_proxy: bool = False
    proxy_url: Optional[str] = None
    min_delay: float = 2.0
    max_delay: float = 10.0
    max_retries: int = 3
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    ])


class BaseExtractor(ABC):
    """Base class for all audio extractors."""

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize base extractor.

        Args:
            config: Extractor configuration
            logger: Logger instance
        """
        self.config = config or ExtractorConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract_audio(self, source: str) -> ExtractionResult:
        """
        Extract audio from a source.

        Args:
            source: Source URL or path

        Returns:
            ExtractionResult with audio path and metadata
        """
        pass

    @abstractmethod
    def extract_audio_batch(self, sources: List[str]) -> List[ExtractionResult]:
        """
        Extract audio from multiple sources.

        Args:
            sources: List of source URLs or paths

        Returns:
            List of ExtractionResult objects
        """
        pass

    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(self.config.user_agents)

    def get_random_delay(self) -> float:
        """Get a random delay between requests."""
        return random.uniform(self.config.min_delay, self.config.max_delay)

    def _get_ydl_opts(self) -> Dict[str, Any]:
        """Get common yt-dlp options."""
        opts = {
            'format': 'bestaudio/best',
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': self.config.max_retries,
            'http_headers': {
                'User-Agent': self.get_random_user_agent(),
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
        }

        # Audio format
        if self.config.audio_format == 'wav':
            opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        elif self.config.audio_format == 'mp3':
            opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': self.config.audio_quality,
            }]

        # Cookies
        if self.config.cookies_from_browser:
            opts['cookies_from_browser'] = (self.config.cookies_from_browser,)

        # Proxy
        if self.config.use_proxy and self.config.proxy_url:
            opts['proxy'] = self.config.proxy_url

        return opts

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize a string for use as a filename."""
        import re
        # Remove problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
        sanitized = sanitized.replace(' ', '_')
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')[:100]  # Limit length

    def _generate_output_path(
        self,
        video_id: str,
        title: str,
        subdir: Optional[str] = None
    ) -> Path:
        """Generate output path for extracted audio."""
        safe_title = self._sanitize_filename(title)
        filename = f"{datetime.now().strftime('%Y%m%d')}_{safe_title}_{video_id}.{self.config.audio_format}"

        if subdir:
            output_dir = self.config.output_dir / subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / filename

        return self.config.output_dir / filename

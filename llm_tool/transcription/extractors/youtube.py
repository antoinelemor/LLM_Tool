"""
YouTube audio extractor.

Extracts audio from YouTube videos, channels, and playlists
using yt-dlp with support for cookies and proxies.
"""

import re
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base import BaseExtractor, ExtractionResult, ExtractorConfig

try:
    from yt_dlp import YoutubeDL
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False


class YouTubeExtractor(BaseExtractor):
    """Extract audio from YouTube videos, channels, and playlists."""

    def __init__(self, config: Optional[ExtractorConfig] = None, logger=None):
        super().__init__(config, logger)
        if not HAS_YTDLP:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")

    def extract_audio(self, url: str, subdir: Optional[str] = None) -> ExtractionResult:
        """
        Extract audio from a YouTube video.

        Args:
            url: YouTube video URL
            subdir: Subdirectory for output

        Returns:
            ExtractionResult with audio path and metadata
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            return ExtractionResult(success=False, error="Invalid YouTube URL")

        # Check if already downloaded
        existing = list(self.config.output_dir.rglob(f"*_{video_id}.{self.config.audio_format}"))
        if existing:
            self.logger.info(f"Already downloaded: {video_id}")
            # Still need to fetch metadata for cached files
            ydl_opts = self._get_ydl_opts()
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return ExtractionResult(
                        success=True,
                        audio_path=existing[0],
                        video_id=video_id,
                        title=info.get('title', 'Unknown') if info else 'Unknown',
                        duration=info.get('duration', 0) if info else 0,
                        metadata={
                            'channel': info.get('channel', info.get('uploader', '')) if info else '',
                            'upload_date': info.get('upload_date', '') if info else '',
                            'view_count': info.get('view_count', 0) if info else 0,
                            'like_count': info.get('like_count', 0) if info else 0,
                            'description': info.get('description', '') if info else '',
                            'url': url
                        }
                    )
            except Exception:
                # If metadata fetch fails, return with minimal info
                return ExtractionResult(
                    success=True,
                    audio_path=existing[0],
                    video_id=video_id
                )

        # Setup yt-dlp options
        ydl_opts = self._get_ydl_opts()

        # Temporary output template
        output_dir = self.config.output_dir / subdir if subdir else self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ydl_opts['outtmpl'] = str(output_dir / f"%(upload_date)s_%(title)s_%(id)s.%(ext)s")

        try:
            with YoutubeDL(ydl_opts) as ydl:
                self.logger.info(f"Downloading audio: {url}")

                # Extract info first
                info = ydl.extract_info(url, download=False)
                if not info:
                    return ExtractionResult(success=False, error="Could not extract video info")

                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)

                # Download
                ydl.download([url])

                # Find the downloaded file
                audio_files = list(output_dir.glob(f"*_{video_id}.*"))
                if audio_files:
                    audio_path = audio_files[0]
                    self.logger.info(f"Downloaded: {audio_path.name}")

                    return ExtractionResult(
                        success=True,
                        audio_path=audio_path,
                        video_id=video_id,
                        title=title,
                        duration=duration,
                        metadata={
                            'channel': info.get('channel', info.get('uploader', '')),
                            'upload_date': info.get('upload_date', ''),
                            'view_count': info.get('view_count', 0),
                            'like_count': info.get('like_count', 0),
                            'description': info.get('description', ''),
                            'url': url
                        }
                    )
                else:
                    return ExtractionResult(success=False, error="Audio file not found after download")

        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            return ExtractionResult(success=False, video_id=video_id, error=str(e))

    def extract_audio_batch(
        self,
        urls: List[str],
        subdir: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ExtractionResult]:
        """
        Extract audio from multiple YouTube videos.

        Args:
            urls: List of YouTube video URLs
            subdir: Subdirectory for output
            progress_callback: Callback function(current, total, result)

        Returns:
            List of ExtractionResult objects
        """
        results = []
        total = len(urls)

        for i, url in enumerate(urls):
            result = self.extract_audio(url, subdir)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

            # Random delay between downloads
            if i < total - 1:
                time.sleep(self.get_random_delay())

        return results

    def extract_channel_videos(
        self,
        channel_url: str,
        max_videos: int = 100,
        include_shorts: bool = True,
        include_streams: bool = True
    ) -> List[str]:
        """
        Extract video URLs from a YouTube channel.

        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum videos to extract
            include_shorts: Include YouTube Shorts
            include_streams: Include live streams

        Returns:
            List of video URLs
        """
        all_urls = []

        ydl_opts = {
            'skip_download': True,
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': self.get_random_user_agent(),
                'Accept-Language': 'en-US,en;q=0.9',
            }
        }

        if self.config.cookies_from_browser:
            ydl_opts['cookies_from_browser'] = (self.config.cookies_from_browser,)

        if self.config.use_proxy and self.config.proxy_url:
            ydl_opts['proxy'] = self.config.proxy_url

        pages = ['videos']
        if include_shorts:
            pages.append('shorts')
        if include_streams:
            pages.append('streams')

        for page in pages:
            try:
                page_url = channel_url.rstrip('/') + '/' + page
                self.logger.info(f"Extracting from: {page_url}")

                with YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(page_url, download=False)

                    if result and 'entries' in result:
                        for video in result['entries']:
                            if video and video.get('id'):
                                video_url = f"https://www.youtube.com/watch?v={video['id']}"
                                all_urls.append(video_url)

                                if len(all_urls) >= max_videos:
                                    break

                self.logger.info(f"Found {len(all_urls)} videos from {page}")
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                self.logger.error(f"Error extracting {page}: {e}")

            if len(all_urls) >= max_videos:
                break

        unique_urls = list(dict.fromkeys(all_urls))[:max_videos]
        self.logger.info(f"Total unique videos: {len(unique_urls)}")

        return unique_urls

    def extract_playlist(self, playlist_url: str, max_videos: int = 500) -> List[str]:
        """
        Extract video URLs from a YouTube playlist.

        Args:
            playlist_url: YouTube playlist URL
            max_videos: Maximum videos to extract

        Returns:
            List of video URLs
        """
        ydl_opts = {
            'skip_download': True,
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'socket_timeout': 30,
            'http_headers': {
                'User-Agent': self.get_random_user_agent(),
            }
        }

        if self.config.cookies_from_browser:
            ydl_opts['cookies_from_browser'] = (self.config.cookies_from_browser,)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                self.logger.info(f"Extracting playlist: {playlist_url}")
                result = ydl.extract_info(playlist_url, download=False)

                if result and 'entries' in result:
                    urls = []
                    for video in result['entries']:
                        if video and video.get('id'):
                            urls.append(f"https://www.youtube.com/watch?v={video['id']}")
                            if len(urls) >= max_videos:
                                break

                    self.logger.info(f"Found {len(urls)} videos in playlist")
                    return urls

        except Exception as e:
            self.logger.error(f"Error extracting playlist: {e}")

        return []

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video metadata without downloading."""
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }

        if self.config.cookies_from_browser:
            ydl_opts['cookies_from_browser'] = (self.config.cookies_from_browser,)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    return {
                        'id': info.get('id'),
                        'title': info.get('title'),
                        'duration': info.get('duration'),
                        'channel': info.get('channel', info.get('uploader')),
                        'upload_date': info.get('upload_date'),
                        'view_count': info.get('view_count'),
                        'description': info.get('description'),
                    }
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")

        return None

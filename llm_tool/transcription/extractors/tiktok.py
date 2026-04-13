"""
TikTok audio extractor.

Extracts audio from TikTok videos and user profiles
using yt-dlp with support for cookies and proxies.
"""

import re
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import BaseExtractor, ExtractionResult, ExtractorConfig

try:
    from yt_dlp import YoutubeDL
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False


class TikTokExtractor(BaseExtractor):
    """Extract audio from TikTok videos and user profiles."""

    def __init__(self, config: Optional[ExtractorConfig] = None, logger=None):
        super().__init__(config, logger)
        if not HAS_YTDLP:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")

    def extract_audio(self, url: str, subdir: Optional[str] = None) -> ExtractionResult:
        """
        Extract audio from a TikTok video.

        Args:
            url: TikTok video URL
            subdir: Subdirectory for output

        Returns:
            ExtractionResult with audio path and metadata
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            return ExtractionResult(success=False, error="Invalid TikTok URL")

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
                            'uploader': info.get('uploader', info.get('creator', '')) if info else '',
                            'uploader_id': info.get('uploader_id', '') if info else '',
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

        # Output directory
        output_dir = self.config.output_dir / subdir if subdir else self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ydl_opts['outtmpl'] = str(output_dir / f"%(upload_date)s_%(title).50s_%(id)s.%(ext)s")

        try:
            with YoutubeDL(ydl_opts) as ydl:
                self.logger.info(f"Downloading audio: {url}")

                # Extract info
                info = ydl.extract_info(url, download=False)
                if not info:
                    return ExtractionResult(success=False, error="Could not extract video info")

                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)

                # Download
                ydl.download([url])

                # Find downloaded file
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
                            'uploader': info.get('uploader', info.get('creator', '')),
                            'uploader_id': info.get('uploader_id', ''),
                            'upload_date': info.get('upload_date', ''),
                            'view_count': info.get('view_count', 0),
                            'like_count': info.get('like_count', 0),
                            'comment_count': info.get('comment_count', 0),
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
        Extract audio from multiple TikTok videos.

        Args:
            urls: List of TikTok video URLs
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

            # Random delay (TikTok is more sensitive to rate limiting)
            if i < total - 1:
                time.sleep(self.get_random_delay() * 1.5)

        return results

    def extract_user_videos(
        self,
        user_url: str,
        max_videos: int = 100
    ) -> List[str]:
        """
        Extract video URLs from a TikTok user profile.

        Args:
            user_url: TikTok user profile URL
            max_videos: Maximum videos to extract

        Returns:
            List of video URLs
        """
        ydl_opts = {
            'skip_download': True,
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'playlistend': max_videos,
            'http_headers': {
                'User-Agent': self.get_random_user_agent(),
                'Accept-Language': 'en-US,en;q=0.9',
            }
        }

        if self.config.cookies_from_browser:
            ydl_opts['cookies_from_browser'] = (self.config.cookies_from_browser,)

        if self.config.use_proxy and self.config.proxy_url:
            ydl_opts['proxy'] = self.config.proxy_url

        try:
            with YoutubeDL(ydl_opts) as ydl:
                self.logger.info(f"Extracting videos from: {user_url}")
                result = ydl.extract_info(user_url, download=False)

                if result and 'entries' in result:
                    urls = []
                    for video in result['entries']:
                        if video and video.get('webpage_url'):
                            urls.append(video['webpage_url'])
                        elif video and video.get('id'):
                            urls.append(f"https://www.tiktok.com/@{result.get('uploader_id', 'user')}/video/{video['id']}")

                        if len(urls) >= max_videos:
                            break

                    self.logger.info(f"Found {len(urls)} videos from user")
                    return urls

        except Exception as e:
            self.logger.error(f"Error extracting user videos: {e}")

        return []

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from TikTok URL."""
        patterns = [
            r'tiktok\.com/@[\w.]+/video/(\d+)',
            r'tiktok\.com/v/(\d+)',
            r'vm\.tiktok\.com/(\w+)',
            r'tiktok\.com/t/(\w+)',
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
                        'uploader': info.get('uploader', info.get('creator')),
                        'upload_date': info.get('upload_date'),
                        'view_count': info.get('view_count'),
                        'like_count': info.get('like_count'),
                        'description': info.get('description'),
                    }
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")

        return None

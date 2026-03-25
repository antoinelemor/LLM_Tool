"""Audio extraction modules for various sources."""

from .youtube import YouTubeExtractor
from .tiktok import TikTokExtractor
from .local import LocalExtractor

__all__ = ["YouTubeExtractor", "TikTokExtractor", "LocalExtractor"]

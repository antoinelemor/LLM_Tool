"""
Local file audio extractor.

Handles extraction and conversion of local audio/video files
for transcription.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base import BaseExtractor, ExtractionResult, ExtractorConfig


class LocalExtractor(BaseExtractor):
    """Extract and convert audio from local files."""

    SUPPORTED_AUDIO = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.opus'}
    SUPPORTED_VIDEO = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}

    def __init__(self, config: Optional[ExtractorConfig] = None, logger=None):
        super().__init__(config, logger)

    def extract_audio(
        self,
        source: str,
        subdir: Optional[str] = None,
        convert: bool = True
    ) -> ExtractionResult:
        """
        Extract/convert audio from a local file.

        Args:
            source: Path to local audio or video file
            subdir: Subdirectory for output
            convert: Convert to target format if needed

        Returns:
            ExtractionResult with audio path
        """
        source_path = Path(source)

        if not source_path.exists():
            return ExtractionResult(success=False, error=f"File not found: {source}")

        suffix = source_path.suffix.lower()

        # Check if file type is supported
        if suffix not in self.SUPPORTED_AUDIO and suffix not in self.SUPPORTED_VIDEO:
            return ExtractionResult(
                success=False,
                error=f"Unsupported file format: {suffix}"
            )

        # Output directory
        output_dir = self.config.output_dir / subdir if subdir else self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate output filename
            file_id = source_path.stem[:50]
            output_name = f"{datetime.now().strftime('%Y%m%d')}_{self._sanitize_filename(source_path.stem)}_{file_id[-8:]}.{self.config.audio_format}"
            output_path = output_dir / output_name

            # Check if already processed
            if output_path.exists():
                self.logger.info(f"Already processed: {output_path.name}")
                return ExtractionResult(
                    success=True,
                    audio_path=output_path,
                    video_id=file_id,
                    title=source_path.stem
                )

            if suffix in self.SUPPORTED_VIDEO:
                # Extract audio from video
                result = self._extract_from_video(source_path, output_path)
                if result:
                    return ExtractionResult(
                        success=True,
                        audio_path=output_path,
                        video_id=file_id,
                        title=source_path.stem,
                        metadata={'source': str(source_path), 'type': 'video'}
                    )
                else:
                    return ExtractionResult(success=False, error="Failed to extract audio from video")

            elif suffix in self.SUPPORTED_AUDIO:
                # Handle audio file
                if suffix == f'.{self.config.audio_format}' and not convert:
                    # Same format, just copy
                    shutil.copy2(source_path, output_path)
                else:
                    # Convert to target format
                    result = self._convert_audio(source_path, output_path)
                    if not result:
                        # Fallback: just copy
                        shutil.copy2(source_path, output_path)

                return ExtractionResult(
                    success=True,
                    audio_path=output_path,
                    video_id=file_id,
                    title=source_path.stem,
                    metadata={'source': str(source_path), 'type': 'audio'}
                )

        except Exception as e:
            self.logger.error(f"Error processing {source}: {e}")
            return ExtractionResult(success=False, error=str(e))

        return ExtractionResult(success=False, error="Unknown error")

    def extract_audio_batch(
        self,
        sources: List[str],
        subdir: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ExtractionResult]:
        """
        Extract audio from multiple local files.

        Args:
            sources: List of file paths
            subdir: Subdirectory for output
            progress_callback: Callback function(current, total, result)

        Returns:
            List of ExtractionResult objects
        """
        results = []
        total = len(sources)

        for i, source in enumerate(sources):
            result = self.extract_audio(source, subdir)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results

    def scan_directory(
        self,
        directory: str,
        recursive: bool = True,
        include_video: bool = True
    ) -> List[Path]:
        """
        Scan directory for audio/video files.

        Args:
            directory: Directory to scan
            recursive: Scan subdirectories
            include_video: Include video files

        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []

        extensions = set(self.SUPPORTED_AUDIO)
        if include_video:
            extensions.update(self.SUPPORTED_VIDEO)

        files = []
        glob_pattern = '**/*' if recursive else '*'

        for ext in extensions:
            files.extend(dir_path.glob(f"{glob_pattern}{ext}"))

        self.logger.info(f"Found {len(files)} audio/video files in {directory}")
        return sorted(files)

    def _extract_from_video(self, video_path: Path, output_path: Path) -> bool:
        """Extract audio track from video using ffmpeg."""
        try:
            codec = 'pcm_s16le' if self.config.audio_format == 'wav' else 'libmp3lame'

            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # No video
                '-acodec', codec,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                self.logger.info(f"Extracted audio to: {output_path.name}")
                return True
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                return False

        except FileNotFoundError:
            self.logger.error("FFmpeg not found. Please install FFmpeg.")
            return False
        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}")
            return False

    def _convert_audio(self, input_path: Path, output_path: Path) -> bool:
        """Convert audio to target format using ffmpeg."""
        try:
            codec = 'pcm_s16le' if self.config.audio_format == 'wav' else 'libmp3lame'

            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-acodec', codec,
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                self.logger.info(f"Converted audio to: {output_path.name}")
                return True
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                return False

        except FileNotFoundError:
            self.logger.warning("FFmpeg not found, using source file as-is")
            return False
        except Exception as e:
            self.logger.error(f"Error converting audio: {e}")
            return False

    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get audio/video file information."""
        file_path = Path(path)
        if not file_path.exists():
            return None

        info = {
            'path': str(file_path),
            'name': file_path.name,
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'format': file_path.suffix.lstrip('.'),
            'is_video': file_path.suffix.lower() in self.SUPPORTED_VIDEO,
            'is_audio': file_path.suffix.lower() in self.SUPPORTED_AUDIO,
        }

        # Try to get duration with ffprobe
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            info['duration'] = float(result.stdout.strip())
        except Exception:
            info['duration'] = None

        return info

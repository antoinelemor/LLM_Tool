"""
Speaker diarization module.

Provides speaker identification and segmentation using
Pyannote audio for multi-speaker transcriptions.
"""

import os
import logging
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import pandas as pd
from tqdm import tqdm

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False
    Pipeline = Annotation = Segment = None


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""
    hf_token: Optional[str] = None
    device: str = "auto"
    speaker_merge_threshold: float = 0.90
    word_alignment_buffer: float = 0.05  # seconds
    show_progress: bool = True  # Show tqdm progress bars
    # Speaker smoothing parameters to fix fragmentation errors
    min_words_per_speaker_segment: int = 3  # Minimum words before considering a speaker change valid
    smooth_isolated_words: bool = True  # Merge isolated speaker changes back to surrounding speaker


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    success: bool
    speakers: List[str] = None
    segments: List[Dict[str, Any]] = None
    annotation: Any = None  # Pyannote Annotation
    error: Optional[str] = None

    def __post_init__(self):
        if self.speakers is None:
            self.speakers = []
        if self.segments is None:
            self.segments = []


class SpeakerDiarizer:
    """Speaker diarization using Pyannote."""

    def __init__(
        self,
        config: Optional[DiarizationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize speaker diarizer.

        Args:
            config: Diarization configuration
            logger: Logger instance
        """
        if not HAS_PYANNOTE:
            raise ImportError(
                "pyannote.audio is required. Install with: pip install pyannote.audio"
            )

        self.config = config or DiarizationConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline = None

        # Setup device
        self._setup_device()

    def _setup_device(self):
        """Setup compute device."""
        import platform

        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            # Use MPS for Apple Silicon (pyannote supports MPS)
            elif platform.system() == "Darwin" and hasattr(torch.backends, 'mps'):
                if torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

    def load_pipeline(self) -> bool:
        """Load the diarization pipeline."""
        if self.pipeline is not None:
            return True

        # Get HuggingFace token
        hf_token = self.config.hf_token or os.getenv("HF_TOKEN")
        if not hf_token:
            self.logger.warning(
                "No HuggingFace token provided. Set HF_TOKEN environment variable "
                "or pass hf_token in config."
            )

        try:
            self.logger.info("Loading Pyannote diarization pipeline...")
            # Use 'token' parameter (use_auth_token is deprecated)
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )

            # Move to device
            if self.device != "cpu":
                try:
                    self.pipeline.to(torch.device(self.device))
                    self.logger.info(f"Pipeline loaded on {self.device}")
                except Exception as e:
                    self.logger.warning(f"Failed to use {self.device}, falling back to CPU: {e}")
                    self.device = "cpu"
            else:
                self.logger.info("Pipeline loaded on CPU")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load diarization pipeline: {e}")
            return False

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 60.0  # Default estimate

    def _load_audio_for_pyannote(self, audio_path: Path) -> dict:
        """Load audio in a format compatible with pyannote (avoids torchcodec issues)."""
        try:
            # Try using torchaudio first
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception:
            pass

        try:
            # Fallback to whisper's audio loader
            import whisper
            from whisper.audio import load_audio, SAMPLE_RATE
            audio = load_audio(str(audio_path))
            waveform = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
            return {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        except Exception:
            pass

        # Last resort: use pydub
        try:
            from pydub import AudioSegment
            import numpy as np

            audio = AudioSegment.from_file(str(audio_path))
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            # Get raw data as numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / np.max(np.abs(samples))  # Normalize
            waveform = torch.from_numpy(samples).unsqueeze(0)
            return {"waveform": waveform, "sample_rate": audio.frame_rate}
        except Exception as e:
            raise RuntimeError(f"Could not load audio: {e}")

    def diarize(
        self,
        audio_path: Path,
        uri: Optional[str] = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file
            uri: Optional URI identifier for the audio

        Returns:
            DiarizationResult with speaker segments
        """
        if not self.load_pipeline():
            return DiarizationResult(success=False, error="Failed to load pipeline")

        if not audio_path.exists():
            return DiarizationResult(success=False, error=f"Audio file not found: {audio_path}")

        # Create safe URI
        if uri is None:
            uri = self._sanitize_uri(audio_path.stem)

        try:
            self.logger.info(f"Diarizing: {audio_path.name}")

            # Get audio duration for progress estimation
            audio_duration = self._get_audio_duration(audio_path)

            # Pre-load audio to avoid torchcodec issues
            self.logger.info("Loading audio for diarization...")
            audio_data = self._load_audio_for_pyannote(audio_path)

            # Run diarization with progress bar
            if self.config.show_progress:
                # Diarization typically processes at ~1-3x realtime
                estimated_time = audio_duration / 2.0

                pbar = tqdm(
                    total=100,
                    desc="Diarization",
                    unit="%",
                    leave=False,
                    bar_format='{l_bar}{bar}| {n:.0f}% [{elapsed}<{remaining}]'
                )

                # Run in thread to allow progress updates
                result_holder = [None]
                error_holder = [None]

                def run_diarization():
                    try:
                        # Pass pre-loaded audio instead of file path
                        result_holder[0] = self.pipeline(audio_data)
                    except Exception as e:
                        error_holder[0] = e

                thread = threading.Thread(target=run_diarization)
                thread.start()

                start_time = time.time()
                while thread.is_alive():
                    elapsed = time.time() - start_time
                    progress = min(95, (elapsed / estimated_time) * 100)
                    pbar.n = progress
                    pbar.refresh()
                    time.sleep(0.5)

                thread.join()
                pbar.n = 100
                pbar.refresh()
                pbar.close()

                if error_holder[0]:
                    raise error_holder[0]

                diarization = result_holder[0]
            else:
                # Pass pre-loaded audio instead of file path
                diarization = self.pipeline(audio_data)

            # Handle different return types from pyannote
            # pyannote 4.x returns DiarizeOutput, older versions return Annotation
            annotation = None
            if hasattr(diarization, 'itertracks'):
                # It's already an Annotation object (legacy mode)
                annotation = diarization
            elif hasattr(diarization, 'speaker_diarization'):
                # pyannote 4.x DiarizeOutput object
                annotation = diarization.speaker_diarization
            elif isinstance(diarization, tuple):
                # It might be a tuple (annotation, embeddings) or similar
                for item in diarization:
                    if hasattr(item, 'itertracks'):
                        annotation = item
                        break
            elif hasattr(diarization, 'annotation'):
                # Generic wrapper object with an annotation attribute
                annotation = diarization.annotation

            if annotation is None:
                raise ValueError(f"Unexpected diarization output type: {type(diarization)}")

            # Extract speakers and segments
            speakers = set()
            segments = []

            for turn, _, speaker in annotation.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })

            self.logger.info(f"Found {len(speakers)} speakers, {len(segments)} segments")

            return DiarizationResult(
                success=True,
                speakers=sorted(list(speakers)),
                segments=segments,
                annotation=annotation  # Use the extracted annotation
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device != "cpu":
                self.logger.warning("GPU out of memory, retrying on CPU")
                return self._diarize_cpu_fallback(audio_path, uri)
            else:
                self.logger.error(f"Diarization failed: {e}")
                return DiarizationResult(success=False, error=str(e))

        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return DiarizationResult(success=False, error=str(e))

    def _diarize_cpu_fallback(
        self,
        audio_path: Path,
        uri: str
    ) -> DiarizationResult:
        """Fallback to CPU diarization."""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move pipeline to CPU
            self.pipeline.to(torch.device("cpu"))

            # Pre-load audio to avoid torchcodec issues
            audio_data = self._load_audio_for_pyannote(audio_path)
            diarization = self.pipeline(audio_data)

            # Move back to GPU for next call
            if self.device != "cpu":
                self.pipeline.to(torch.device(self.device))

            # Handle different return types from pyannote
            # pyannote 4.x returns DiarizeOutput, older versions return Annotation
            annotation = None
            if hasattr(diarization, 'itertracks'):
                annotation = diarization
            elif hasattr(diarization, 'speaker_diarization'):
                # pyannote 4.x DiarizeOutput object
                annotation = diarization.speaker_diarization
            elif isinstance(diarization, tuple):
                for item in diarization:
                    if hasattr(item, 'itertracks'):
                        annotation = item
                        break
            elif hasattr(diarization, 'annotation'):
                annotation = diarization.annotation

            if annotation is None:
                raise ValueError(f"Unexpected diarization output type: {type(diarization)}")

            # Extract results
            speakers = set()
            segments = []

            for turn, _, speaker in annotation.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })

            return DiarizationResult(
                success=True,
                speakers=sorted(list(speakers)),
                segments=segments,
                annotation=annotation
            )

        except Exception as e:
            self.logger.error(f"CPU fallback diarization failed: {e}")
            return DiarizationResult(success=False, error=str(e))

    def merge_dominant_speaker(
        self,
        result: DiarizationResult
    ) -> DiarizationResult:
        """
        Merge all segments to dominant speaker if threshold exceeded.

        Args:
            result: Original diarization result

        Returns:
            Modified result with merged speakers
        """
        if not result.success or not result.segments:
            return result

        # Calculate speaker durations
        speaker_durations = {}
        total_duration = 0.0

        for segment in result.segments:
            duration = segment['duration']
            speaker = segment['speaker']
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
            total_duration += duration

        if total_duration == 0:
            return result

        # Find dominant speaker
        dominant_speaker = max(speaker_durations, key=speaker_durations.get)
        dominant_ratio = speaker_durations[dominant_speaker] / total_duration

        self.logger.info(f"Dominant speaker: {dominant_speaker} ({dominant_ratio:.1%})")

        # Merge if threshold exceeded
        if dominant_ratio >= self.config.speaker_merge_threshold:
            self.logger.info(
                f"Merging all speakers to {dominant_speaker} "
                f"({dominant_ratio:.1%} >= {self.config.speaker_merge_threshold:.1%})"
            )

            merged_segments = [
                {**seg, 'speaker': dominant_speaker}
                for seg in result.segments
            ]

            # Update annotation if available
            if result.annotation:
                merged_annotation = Annotation(uri=result.annotation.uri)
                for segment in result.annotation.get_timeline():
                    merged_annotation[segment] = dominant_speaker
            else:
                merged_annotation = None

            return DiarizationResult(
                success=True,
                speakers=[dominant_speaker],
                segments=merged_segments,
                annotation=merged_annotation
            )

        return result

    def align_words_with_speakers(
        self,
        words_df: pd.DataFrame,
        result: DiarizationResult
    ) -> pd.DataFrame:
        """
        Assign speaker labels to words based on diarization.

        Args:
            words_df: DataFrame with words (word, start, end columns)
            result: Diarization result

        Returns:
            DataFrame with speaker column filled
        """
        if not result.success or not result.annotation:
            self.logger.warning("No diarization result, using single speaker")
            words_df['speaker'] = 'SPEAKER_00'
            return words_df

        # Get dominant speaker for fallback
        speaker_durations = {}
        for seg in result.segments:
            speaker_durations[seg['speaker']] = (
                speaker_durations.get(seg['speaker'], 0.0) + seg['duration']
            )
        dominant_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else "SPEAKER_00"

        # Assign speakers to words
        buffer = self.config.word_alignment_buffer

        for idx, row in words_df.iterrows():
            word_start = row['start'] - buffer
            word_end = row['end'] + buffer
            word_segment = Segment(max(0, word_start), word_end)

            best_speaker = None
            max_overlap = 0.0

            for turn, _, speaker in result.annotation.itertracks(yield_label=True):
                overlap = word_segment & turn
                if overlap and overlap.duration > max_overlap:
                    max_overlap = overlap.duration
                    best_speaker = speaker

            if best_speaker:
                words_df.at[idx, 'speaker'] = best_speaker
            elif idx > 0 and pd.notna(words_df.at[idx - 1, 'speaker']):
                # Use previous word's speaker for continuity
                words_df.at[idx, 'speaker'] = words_df.at[idx - 1, 'speaker']
            else:
                words_df.at[idx, 'speaker'] = dominant_speaker

        # Apply smoothing to fix fragmentation errors
        if self.config.smooth_isolated_words:
            words_df = self._smooth_speaker_assignments(words_df)

        return words_df

    def _smooth_speaker_assignments(self, words_df: pd.DataFrame) -> pd.DataFrame:
        """
        Smooth speaker assignments to fix fragmentation errors.

        This method identifies isolated speaker changes (small groups of words
        assigned to a different speaker) that are likely diarization errors,
        and merges them back to the surrounding speaker.

        Args:
            words_df: DataFrame with speaker assignments

        Returns:
            DataFrame with smoothed speaker assignments
        """
        if len(words_df) < 3:
            return words_df

        speakers = words_df['speaker'].tolist()
        min_words = self.config.min_words_per_speaker_segment

        # Identify speaker runs (consecutive words by same speaker)
        runs = []  # List of (start_idx, end_idx, speaker, word_count)
        run_start = 0
        current_speaker = speakers[0]

        for i in range(1, len(speakers)):
            if speakers[i] != current_speaker:
                runs.append((run_start, i - 1, current_speaker, i - run_start))
                run_start = i
                current_speaker = speakers[i]

        # Add the last run
        runs.append((run_start, len(speakers) - 1, current_speaker, len(speakers) - run_start))

        # Identify runs to merge (isolated short segments surrounded by same speaker)
        changes_made = True
        iterations = 0
        max_iterations = 10  # Prevent infinite loops

        while changes_made and iterations < max_iterations:
            changes_made = False
            iterations += 1
            new_speakers = speakers.copy()

            for i, (start, end, speaker, count) in enumerate(runs):
                # Skip if this run has enough words
                if count >= min_words:
                    continue

                # Check surrounding speakers
                prev_speaker = runs[i - 1][2] if i > 0 else None
                next_speaker = runs[i + 1][2] if i < len(runs) - 1 else None

                # Case 1: Isolated segment surrounded by same speaker on both sides
                if prev_speaker and next_speaker and prev_speaker == next_speaker:
                    # Merge to surrounding speaker
                    for j in range(start, end + 1):
                        new_speakers[j] = prev_speaker
                    changes_made = True
                    self.logger.debug(
                        f"Smoothing: merged {count} words of {speaker} to {prev_speaker} "
                        f"(indices {start}-{end})"
                    )

                # Case 2: Very short segment (1-2 words) at boundaries
                elif count <= 2:
                    if prev_speaker and not next_speaker:
                        # At the end, merge to previous
                        for j in range(start, end + 1):
                            new_speakers[j] = prev_speaker
                        changes_made = True
                    elif next_speaker and not prev_speaker:
                        # At the start, merge to next
                        for j in range(start, end + 1):
                            new_speakers[j] = next_speaker
                        changes_made = True

            if changes_made:
                speakers = new_speakers
                # Recalculate runs
                runs = []
                run_start = 0
                current_speaker = speakers[0]

                for i in range(1, len(speakers)):
                    if speakers[i] != current_speaker:
                        runs.append((run_start, i - 1, current_speaker, i - run_start))
                        run_start = i
                        current_speaker = speakers[i]

                runs.append((run_start, len(speakers) - 1, current_speaker, len(speakers) - run_start))

        # Apply smoothed speakers
        words_df['speaker'] = speakers

        return words_df

    def _sanitize_uri(self, name: str) -> str:
        """Sanitize name for use as URI (no spaces allowed in RTTM)."""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')

    def cleanup(self):
        """Clean up pipeline and free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

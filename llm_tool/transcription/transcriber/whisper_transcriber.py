"""
Whisper transcription module.

Provides speech-to-text transcription using OpenAI's Whisper model
with support for multiple languages and word-level timestamps.
"""

import os
import re
import sys
import logging
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm

try:
    import whisper
    from whisper.audio import load_audio, SAMPLE_RATE
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# Check for HuggingFace Transformers (for MPS GPU acceleration)
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class TranscriptionConfig:
    """Configuration for Whisper transcription."""
    model_name: str = "large-v3"
    language: Optional[str] = None  # Auto-detect if None
    device: str = "auto"
    fp16: bool = True
    # Use parameters that avoid hallucinations (from tested retry config)
    beam_size: int = 10
    best_of: int = 10
    temperature: float = 0.2
    word_timestamps: bool = True
    condition_on_previous_text: bool = False  # Avoid repetition loops
    use_transformers: bool = True  # Use HuggingFace for MPS GPU acceleration
    show_progress: bool = True  # Show tqdm progress bars


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    success: bool
    text: str = ""
    language: str = ""
    words: List[Dict[str, Any]] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None


class WhisperTranscriber:
    """Speech-to-text transcription using Whisper."""

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Whisper transcriber.

        Args:
            config: Transcription configuration
            logger: Logger instance
        """
        if not HAS_WHISPER:
            raise ImportError("whisper is required. Install with: pip install openai-whisper")

        self.config = config or TranscriptionConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.model = None

        # Determine device
        self._setup_device()

    def _setup_device(self):
        """Setup the compute device."""
        self.use_hf_pipeline = False  # Flag for HuggingFace pipeline

        if self.config.device == "auto":
            # Check CUDA first (NVIDIA GPUs)
            if torch.cuda.is_available():
                try:
                    torch.zeros(1).cuda()
                    self.device = "cuda"
                    self.logger.info("Using CUDA GPU")
                except Exception:
                    self.device = "cpu"
            # For Apple Silicon: Use MPS with HuggingFace Transformers
            elif platform.system() == "Darwin" and hasattr(torch.backends, 'mps'):
                if torch.backends.mps.is_available():
                    if HAS_TRANSFORMERS and self.config.use_transformers:
                        # Use HuggingFace Transformers for MPS GPU acceleration
                        self.device = "mps"
                        self.use_hf_pipeline = True
                        self.logger.info("Using Apple Silicon GPU (MPS) with HuggingFace Transformers")
                    else:
                        # Fallback to CPU with openai-whisper
                        self.device = "cpu"
                        self.logger.info("Apple Silicon detected - using CPU (install transformers for GPU)")
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device

        # Disable fp16 for CPU and MPS
        if self.device in ("cpu", "mps"):
            self.config.fp16 = False

    def load_model(self) -> bool:
        """Load the Whisper model."""
        if self.model is not None:
            return True

        # Map model names for HuggingFace
        hf_model_map = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v2",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }

        try:
            if self.use_hf_pipeline:
                # Use HuggingFace Transformers for MPS GPU
                model_id = hf_model_map.get(self.config.model_name, f"openai/whisper-{self.config.model_name}")
                self.logger.info(f"Loading HuggingFace model: {model_id} on {self.device}")

                import warnings
                warnings.filterwarnings("ignore")

                self.hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.hf_model = self.hf_model.to(self.device)
                self.hf_processor = AutoProcessor.from_pretrained(model_id)
                self.model = True  # Flag that model is loaded

                self.logger.info(f"HuggingFace model loaded on {self.device}")
                return True
            else:
                # Use openai-whisper
                self.logger.info(f"Loading Whisper model: {self.config.model_name} on {self.device}")
                self.model = whisper.load_model(
                    self.config.model_name,
                    device=self.device
                )
                self.logger.info(f"Model loaded on {self.device}")
                return True

        except Exception as e:
            # Fallback to CPU with openai-whisper
            self.logger.warning(f"Failed to load model: {e}, falling back to CPU")
            try:
                self.device = "cpu"
                self.use_hf_pipeline = False
                self.config.fp16 = False
                self.model = whisper.load_model(
                    self.config.model_name,
                    device="cpu"
                )
                self.logger.info("Model loaded on CPU (fallback)")
                return True
            except Exception as cpu_error:
                self.logger.error(f"Failed to load model on CPU: {cpu_error}")
                return False

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            audio = load_audio(str(audio_path))
            return len(audio) / SAMPLE_RATE
        except Exception:
            return 0.0

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[float, float, str], None]] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            progress_callback: Callback(current_seconds, total_seconds, status_text)

        Returns:
            TranscriptionResult with text and word timestamps
        """
        if not self.load_model():
            return TranscriptionResult(success=False, error="Failed to load model")

        if not audio_path.exists():
            return TranscriptionResult(success=False, error=f"Audio file not found: {audio_path}")

        # Use configured language or provided language
        lang = language or self.config.language

        try:
            # Get audio duration for progress tracking
            audio_duration = self.get_audio_duration(audio_path)

            if progress_callback:
                progress_callback(0, audio_duration, "Loading audio...")

            # Use HuggingFace Transformers for MPS GPU
            if self.use_hf_pipeline:
                return self._transcribe_with_hf(audio_path, lang, audio_duration, progress_callback)

            # Use openai-whisper for CPU/CUDA
            result = self.model.transcribe(
                str(audio_path),
                language=lang,
                word_timestamps=self.config.word_timestamps,
                fp16=self.config.fp16,
                verbose=False,  # Suppress all Whisper output
                temperature=self.config.temperature,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                condition_on_previous_text=self.config.condition_on_previous_text,
                compression_ratio_threshold=2.8,
                logprob_threshold=-1.0,
                no_speech_threshold=0.5,
                initial_prompt=None,
                suppress_tokens=[-1],  # Don't suppress any tokens
                suppress_blank=False,  # Don't suppress blanks
            )

            if progress_callback:
                progress_callback(audio_duration, audio_duration, "Processing results...")

            # Check for poor transcription
            full_text = result.get('text', '').strip()
            if self._is_poor_transcription(full_text):
                self.logger.warning("Poor transcription detected, retrying with different parameters")
                result = self._retry_transcription(audio_path, lang)

            # Extract words with timestamps
            words = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    word_text = word_info.get('word', '').strip()
                    if word_text:
                        words.append({
                            'word': word_text,
                            'start': word_info.get('start', 0),
                            'end': word_info.get('end', 0),
                        })

            # Clean hallucinations (repeated words)
            original_count = len(words)
            words = self._clean_hallucinations(words)
            if len(words) < original_count:
                self.logger.info(f"Cleaned {original_count - len(words)} hallucinated words")

            detected_lang = result.get('language', lang or 'unknown')

            self.logger.info(f"Transcribed {len(words)} words in {detected_lang}")

            return TranscriptionResult(
                success=True,
                text=result.get('text', ''),
                language=detected_lang,
                words=words,
                segments=result.get('segments', []),
                duration=result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0
            )

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(success=False, error=str(e))

    def _transcribe_with_hf(
        self,
        audio_path: Path,
        language: Optional[str],
        audio_duration: float,
        progress_callback: Optional[Callable[[float, float, str], None]] = None
    ) -> TranscriptionResult:
        """Transcribe using HuggingFace Transformers (for MPS GPU acceleration)."""
        try:
            import numpy as np

            # Load audio using whisper's loader
            audio = load_audio(str(audio_path))

            # Process in chunks of 30 seconds for long audio
            chunk_length = 30 * SAMPLE_RATE  # 30 seconds
            all_text = []
            all_words = []

            num_chunks = (len(audio) + chunk_length - 1) // chunk_length

            # Create progress bar for transcription chunks
            chunk_iterator = range(num_chunks)
            if self.config.show_progress:
                chunk_iterator = tqdm(
                    chunk_iterator,
                    desc="Transcription",
                    unit="chunk",
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )

            for i in chunk_iterator:
                start_idx = i * chunk_length
                end_idx = min((i + 1) * chunk_length, len(audio))
                chunk = audio[start_idx:end_idx]

                if progress_callback:
                    current_time = start_idx / SAMPLE_RATE
                    progress_callback(current_time, audio_duration, f"Transcribing chunk {i+1}/{num_chunks}...")

                # Update tqdm description with current chunk time
                if self.config.show_progress and hasattr(chunk_iterator, 'set_postfix'):
                    chunk_iterator.set_postfix({
                        'time': f"{start_idx / SAMPLE_RATE:.0f}s/{audio_duration:.0f}s"
                    })

                # Process chunk
                inputs = self.hf_processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate transcription
                with torch.no_grad():
                    generate_kwargs = {
                        "max_new_tokens": 444,
                        "task": "transcribe",
                        "return_timestamps": True,
                    }
                    if language:
                        generate_kwargs["language"] = language

                    generated_ids = self.hf_model.generate(**inputs, **generate_kwargs)

                # Decode
                transcription = self.hf_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                all_text.append(transcription.strip())

                # Approximate word timestamps based on chunk position
                chunk_start_time = start_idx / SAMPLE_RATE
                words_in_chunk = transcription.strip().split()
                chunk_duration = len(chunk) / SAMPLE_RATE

                if words_in_chunk:
                    time_per_word = chunk_duration / len(words_in_chunk)
                    for j, word in enumerate(words_in_chunk):
                        word_start = chunk_start_time + j * time_per_word
                        word_end = word_start + time_per_word
                        all_words.append({
                            'word': word,
                            'start': word_start,
                            'end': word_end,
                        })

            if progress_callback:
                progress_callback(audio_duration, audio_duration, "Processing results...")

            full_text = ' '.join(all_text)

            # Detect language (use first chunk for detection)
            detected_lang = language or "en"

            self.logger.info(f"Transcribed {len(all_words)} words using MPS GPU")

            return TranscriptionResult(
                success=True,
                text=full_text,
                language=detected_lang,
                words=all_words,
                segments=[],
                duration=audio_duration
            )

        except Exception as e:
            self.logger.error(f"HuggingFace transcription failed: {e}")
            # Fallback to CPU with openai-whisper
            self.logger.info("Falling back to CPU transcription...")
            self.use_hf_pipeline = False
            self.device = "cpu"
            return self.transcribe(audio_path, language, progress_callback)

    def _is_poor_transcription(self, text: str) -> bool:
        """Check if transcription quality is poor."""
        if not text:
            return True

        words = text.split()
        if len(words) < 10:
            return True

        # Check for repeated patterns (first chunk)
        if len(words) > 3:
            chunk_size = min(5, len(words) // 3)
            if chunk_size > 0:
                first_chunk = ' '.join(words[:chunk_size])
                repeat_count = text.count(first_chunk)
                if repeat_count >= 3:
                    return True

        # Check for single word repetition (hallucination pattern like "very, very, very...")
        word_counts = {}
        for word in words:
            word_lower = word.lower().strip('.,!?')
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

        # If any word appears more than 20% of the total or more than 50 times, it's likely a hallucination
        for word, count in word_counts.items():
            if count > 50 or (len(words) > 20 and count / len(words) > 0.2):
                return True

        return False

    def _clean_hallucinations(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove hallucinated segments (repeated words)."""
        if not words:
            return words

        cleaned = []
        i = 0
        while i < len(words):
            word_info = words[i]
            word = word_info.get('word', '').strip().lower()

            # Check for repetition pattern
            repeat_count = 0
            j = i
            while j < len(words) and words[j].get('word', '').strip().lower() == word:
                repeat_count += 1
                j += 1

            # If word repeats more than 5 times consecutively, it's likely a hallucination
            if repeat_count > 5:
                # Skip all repetitions, keep only one instance
                cleaned.append(word_info)
                i = j
            else:
                cleaned.append(word_info)
                i += 1

        return cleaned

    def _retry_transcription(
        self,
        audio_path: Path,
        language: Optional[str]
    ) -> Dict[str, Any]:
        """Retry transcription with different parameters."""
        # First retry with adjusted parameters - don't suppress any tokens
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=self.config.word_timestamps,
            fp16=self.config.fp16,
            verbose=False,
            temperature=0.2,
            compression_ratio_threshold=2.8,
            logprob_threshold=-1.0,
            no_speech_threshold=0.5,
            beam_size=10,
            best_of=10,
            condition_on_previous_text=False,
            initial_prompt=None,
            suppress_tokens=[-1],  # Don't suppress any tokens
            suppress_blank=False,  # Don't suppress blank
        )

        full_text = result.get('text', '').strip()
        if not full_text or len(full_text.split()) < 5:
            # Final attempt with temperature fallback and more patience
            self.logger.info("Trying final fallback transcription...")
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=self.config.word_timestamps,
                fp16=False,  # Disable FP16 for stability
                verbose=False,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=3.0,
                logprob_threshold=-1.0,
                no_speech_threshold=0.3,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                initial_prompt=None,
                patience=2.0,  # More patience for decoding
                length_penalty=1.0,  # No length penalty
            )

        return result

    def transcribe_batch(
        self,
        audio_paths: List[Path],
        language: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths
            language: Language code (optional)
            progress_callback: Callback function(current, total, result)

        Returns:
            List of TranscriptionResult objects
        """
        results = []
        total = len(audio_paths)

        for i, audio_path in enumerate(audio_paths):
            result = self.transcribe(audio_path, language)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results

    def get_words_dataframe(self, result: TranscriptionResult) -> pd.DataFrame:
        """Convert transcription result to DataFrame."""
        if not result.words:
            return pd.DataFrame(columns=['word', 'start', 'end', 'speaker'])

        df = pd.DataFrame(result.words)
        df['speaker'] = None
        return df

    def cleanup(self):
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.device == "cuda":
            torch.cuda.empty_cache()

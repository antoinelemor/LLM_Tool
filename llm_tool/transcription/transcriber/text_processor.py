"""
Text post-processing for transcriptions.

Handles language-specific text cleanup and formatting
for transcription output.
"""

import re
import csv
import io
from typing import Dict, List, Any, Optional
from datetime import datetime


class TextProcessor:
    """Post-process transcription text for various languages."""

    # French apostrophe corrections
    FRENCH_CONTRACTIONS = {
        r"\b([lL]) '": r"\1'",
        r"\b([dD]) '": r"\1'",
        r"\b([jJ]) '": r"\1'",
        r"\b([mM]) '": r"\1'",
        r"\b([tT]) '": r"\1'",
        r"\b([sS]) '": r"\1'",
        r"\b([cC]) '": r"\1'",
        r"\b([nN]) '": r"\1'",
        r"\b([qQ]u) '": r"\1'",
        r"\b([aA]ujourd) '(hui)": r"\1'\2",
        r"\b([pP]resqu) '": r"\1'",
        r"\b([lL]orsqu) '": r"\1'",
        r"\b([pP]uisqu) '": r"\1'",
        r"\b([qQ]uelqu) '(un)": r"\1'\2",
        r"\s+'": "'",
        r"'\s+": "' ",
    }

    # French compound word fixes
    FRENCH_COMPOUNDS = [
        (r'\bSaint\s+-\s+', 'Saint-'),
        (r'\bSainte\s+-\s+', 'Sainte-'),
        (r'\blà\s+-\s+dedans\b', 'là-dedans'),
        (r'\blà\s+-\s+bas\b', 'là-bas'),
        (r'\blà\s+-\s+haut\b', 'là-haut'),
        (r'\blà\s+-\s+dessus\b', 'là-dessus'),
        (r'\blà\s+-\s+dessous\b', 'là-dessous'),
        (r'\bcelui\s+-\s+ci\b', 'celui-ci'),
        (r'\bcelle\s+-\s+ci\b', 'celle-ci'),
        (r'\bceux\s+-\s+ci\b', 'ceux-ci'),
        (r'\bcelles\s+-\s+ci\b', 'celles-ci'),
        (r'\bcelui\s+-\s+là\b', 'celui-là'),
        (r'\bcelle\s+-\s+là\b', 'celle-là'),
        (r'\bceux\s+-\s+là\b', 'ceux-là'),
        (r'\bcelles\s+-\s+là\b', 'celles-là'),
    ]

    @classmethod
    def process(cls, text: str, language: str = 'en') -> str:
        """
        Process transcription text for a specific language.

        Args:
            text: Input text
            language: ISO 639-1 language code

        Returns:
            Processed text
        """
        if not text:
            return text

        # Language-specific processing
        if language == 'fr':
            text = cls.fix_french_text(text)
        elif language == 'en':
            text = cls.fix_english_text(text)

        # General cleanup
        text = cls.general_cleanup(text)

        return text

    @classmethod
    def fix_french_text(cls, text: str) -> str:
        """Apply French-specific text corrections."""
        if not text:
            return text

        # Fix apostrophes
        for pattern, replacement in cls.FRENCH_CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)

        # Fix compound words
        for pattern, replacement in cls.FRENCH_COMPOUNDS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Fix hyphens in compounds
        text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)

        # Ensure no space before punctuation
        text = re.sub(r'\s+([,\.\!\?;:])', r'\1', text)

        return text

    @classmethod
    def fix_english_text(cls, text: str) -> str:
        """Apply English-specific text corrections."""
        if not text:
            return text

        # Fix common contractions spacing
        contractions = [
            (r"(\w)\s+'s\b", r"\1's"),
            (r"(\w)\s+'t\b", r"\1't"),
            (r"(\w)\s+'re\b", r"\1're"),
            (r"(\w)\s+'ve\b", r"\1've"),
            (r"(\w)\s+'ll\b", r"\1'll"),
            (r"(\w)\s+'d\b", r"\1'd"),
            (r"(\w)\s+'m\b", r"\1'm"),
        ]

        for pattern, replacement in contractions:
            text = re.sub(pattern, replacement, text)

        return text

    @classmethod
    def general_cleanup(cls, text: str) -> str:
        """General text cleanup for any language."""
        if not text:
            return text

        # Remove multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)

        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Trim
        text = text.strip()

        return text

    @classmethod
    def format_transcript(
        cls,
        words: List[Dict],
        include_speakers: bool = True,
        include_timestamps: bool = False,
        language: str = 'en'
    ) -> str:
        """
        Format words into a readable transcript.

        Args:
            words: List of word dictionaries with 'word', 'start', 'end', 'speaker'
            include_speakers: Include speaker labels
            include_timestamps: Include timestamps
            language: Language for text processing

        Returns:
            Formatted transcript string
        """
        if not words:
            return ""

        lines = []
        current_speaker = None
        current_line = []

        for word_info in words:
            word = word_info.get('word', '').strip()
            speaker = word_info.get('speaker', 'SPEAKER_00')

            if not word:
                continue

            if include_speakers and speaker != current_speaker:
                # Save previous line
                if current_line:
                    line_text = ' '.join(current_line)
                    line_text = cls.process(line_text, language)
                    lines.append(line_text)
                    lines.append("")

                # Start new speaker section
                current_speaker = speaker
                lines.append(f"[{current_speaker}]")
                current_line = [word]
            else:
                current_line.append(word)

        # Add final line
        if current_line:
            line_text = ' '.join(current_line)
            line_text = cls.process(line_text, language)
            lines.append(line_text)

        return '\n'.join(lines)

    @classmethod
    def format_srt(
        cls,
        words: List[Dict],
        language: str = 'en',
        words_per_subtitle: int = 10
    ) -> str:
        """
        Format words into SRT subtitle format.

        Args:
            words: List of word dictionaries
            language: Language for text processing
            words_per_subtitle: Words per subtitle entry

        Returns:
            SRT formatted string
        """
        if not words:
            return ""

        entries = []
        idx = 1

        for i in range(0, len(words), words_per_subtitle):
            group = words[i:i + words_per_subtitle]
            if not group:
                continue

            start_time = group[0].get('start', 0)
            end_time = group[-1].get('end', 0)
            text = ' '.join(w.get('word', '') for w in group)
            text = cls.process(text, language)

            start_str = cls._format_srt_time(start_time)
            end_str = cls._format_srt_time(end_time)

            entries.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            idx += 1

        return '\n'.join(entries)

    @classmethod
    def format_vtt(
        cls,
        words: List[Dict],
        language: str = 'en',
        words_per_subtitle: int = 10
    ) -> str:
        """
        Format words into WebVTT subtitle format.

        Args:
            words: List of word dictionaries
            language: Language for text processing
            words_per_subtitle: Words per subtitle entry

        Returns:
            WebVTT formatted string
        """
        if not words:
            return "WEBVTT\n\n"

        lines = ["WEBVTT", ""]

        for i in range(0, len(words), words_per_subtitle):
            group = words[i:i + words_per_subtitle]
            if not group:
                continue

            start_time = group[0].get('start', 0)
            end_time = group[-1].get('end', 0)
            text = ' '.join(w.get('word', '') for w in group)
            text = cls.process(text, language)

            start_str = cls._format_vtt_time(start_time)
            end_str = cls._format_vtt_time(end_time)

            lines.append(f"{start_str} --> {end_str}")
            lines.append(text)
            lines.append("")

        return '\n'.join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Format seconds to WebVTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @classmethod
    def format_transcript_with_metadata(
        cls,
        words: List[Dict],
        metadata: Optional[Dict[str, Any]] = None,
        include_speakers: bool = True,
        language: str = 'en'
    ) -> str:
        """
        Format words into a transcript with metadata header.

        Args:
            words: List of word dictionaries
            metadata: Metadata dictionary with title, date, video_id, channel, etc.
            include_speakers: Include speaker labels
            language: Language for text processing

        Returns:
            Formatted transcript string with metadata header
        """
        lines = []

        # Build header
        lines.append("# Transcription")
        lines.append("")

        if metadata:
            if metadata.get('date'):
                lines.append(f"Date: {metadata['date']}")
            if metadata.get('title'):
                lines.append(f"Title: {metadata['title']}")
            if metadata.get('video_id'):
                lines.append(f"Video ID: {metadata['video_id']}")
            if metadata.get('channel'):
                lines.append(f"Channel: {metadata['channel']}")
            if metadata.get('source'):
                lines.append(f"Source: {metadata['source']}")

        # Add language and transcription info
        lines.append(f"Language: {cls._get_language_name(language)}")
        lines.append(f"Transcribed: {datetime.now().isoformat()}")
        lines.append(f"Pipeline: Transcribe-Tool")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Format the transcript body
        if not words:
            lines.append("")
            lines.append("Audio not transcribed")
            return '\n'.join(lines)

        current_speaker = None
        current_line = []

        for word_info in words:
            word = word_info.get('word', '').strip()
            speaker = word_info.get('speaker', 'SPEAKER_00')

            if not word:
                continue

            if include_speakers and speaker != current_speaker:
                # Save previous line
                if current_line:
                    line_text = ' '.join(current_line)
                    line_text = cls.process(line_text, language)
                    lines.append(line_text)
                    lines.append("")

                # Start new speaker section
                current_speaker = speaker
                lines.append(f"[{current_speaker}]")
                current_line = [word]
            else:
                current_line.append(word)

        # Add final line
        if current_line:
            line_text = ' '.join(current_line)
            line_text = cls.process(line_text, language)
            lines.append(line_text)

        # Clean up multiple newlines
        transcript = '\n'.join(lines)
        transcript = re.sub(r'\n{3,}', '\n\n', transcript)

        return transcript

    @classmethod
    def format_csv(
        cls,
        words: List[Dict],
        metadata: Optional[Dict[str, Any]] = None,
        segmentation_level: str = 'sentence',
        language: str = 'en'
    ) -> str:
        """
        Format words into CSV with sentence IDs and speaker labels.

        Args:
            words: List of word dictionaries
            metadata: Metadata dictionary
            segmentation_level: 'sentence', '2', '3', or 'paragraph'
            language: Language for text processing

        Returns:
            CSV formatted string
        """
        if not words:
            return ""

        # First, group words by speaker and build full text
        speaker_segments = []
        current_speaker = None
        current_words = []

        for word_info in words:
            word = word_info.get('word', '').strip()
            speaker = word_info.get('speaker', 'SPEAKER_00')

            if not word:
                continue

            if speaker != current_speaker and current_words:
                speaker_segments.append({
                    'speaker': current_speaker,
                    'words': current_words,
                    'text': ' '.join(current_words)
                })
                current_words = []

            current_speaker = speaker
            current_words.append(word)

        if current_words:
            speaker_segments.append({
                'speaker': current_speaker,
                'words': current_words,
                'text': ' '.join(current_words)
            })

        # Merge adjacent segments with the same speaker
        # This fixes fragmentation caused by diarization errors
        merged_speaker_segments = []
        for seg in speaker_segments:
            if merged_speaker_segments and merged_speaker_segments[-1]['speaker'] == seg['speaker']:
                # Same speaker as previous segment - merge them
                merged_speaker_segments[-1]['words'].extend(seg['words'])
                merged_speaker_segments[-1]['text'] += ' ' + seg['text']
            else:
                merged_speaker_segments.append(seg)

        speaker_segments = merged_speaker_segments

        # Segment each speaker's text
        rows = []
        segment_id = 1

        for seg in speaker_segments:
            text = cls.process(seg['text'], language)
            speaker = seg['speaker'] or 'SPEAKER_00'

            # Segment the text
            segments = cls._segment_text(text, language, segmentation_level)

            for segment_text in segments:
                if segment_text.strip():
                    row = {
                        'segment_id': segment_id,
                        'speaker': speaker,
                        'text': segment_text.strip(),
                    }

                    # Add metadata fields if available
                    if metadata:
                        row['video_id'] = metadata.get('video_id', '')
                        row['title'] = metadata.get('title', '')
                        row['date'] = metadata.get('date', '')
                        row['channel'] = metadata.get('channel', '')

                    rows.append(row)
                    segment_id += 1

        # Write CSV
        if not rows:
            return ""

        output = io.StringIO()
        fieldnames = ['segment_id', 'speaker', 'text']
        if metadata:
            fieldnames.extend(['video_id', 'title', 'date', 'channel'])

        writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

        return output.getvalue()

    # Cache for sentence segmentation models
    _wtp_model: Any = None
    _spacy_cache: Dict[str, Any] = {}

    @classmethod
    def _get_wtp_model(cls):
        """
        Load wtpsplit model - SOTA for sentence boundary detection.

        wtpsplit uses a transformer model specifically trained for
        sentence segmentation across 85+ languages.
        """
        if cls._wtp_model is not None:
            return cls._wtp_model

        try:
            from wtpsplit import SaT

            # SaT (Segment any Text) - SOTA model for sentence segmentation
            # Uses a transformer architecture trained on diverse multilingual data
            # "sat-3l" is the best balance of speed/accuracy
            # For maximum accuracy, use "sat-12l" (slower but more precise)
            cls._wtp_model = SaT("sat-3l")

            # Move to GPU if available for faster inference
            try:
                import torch
                if torch.cuda.is_available():
                    cls._wtp_model.half().to("cuda")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    cls._wtp_model.to("mps")
            except Exception:
                pass  # Stay on CPU

            return cls._wtp_model

        except ImportError:
            return None

    @classmethod
    def _get_spacy_model(cls, language: str):
        """
        Get spaCy model as fallback for sentence segmentation.

        Prefers transformer models (_trf) when available for SOTA accuracy,
        falls back to large (_lg) then small (_sm) models.
        """
        if language in cls._spacy_cache:
            return cls._spacy_cache[language]

        import spacy

        # Model preferences: transformer > large > small
        model_preferences = {
            'en': ['en_core_web_trf', 'en_core_web_lg', 'en_core_web_sm'],
            'fr': ['fr_dep_news_trf', 'fr_core_news_lg', 'fr_core_news_sm'],
            'de': ['de_dep_news_trf', 'de_core_news_lg', 'de_core_news_sm'],
            'es': ['es_dep_news_trf', 'es_core_news_lg', 'es_core_news_sm'],
            'it': ['it_core_news_lg', 'it_core_news_sm'],
            'pt': ['pt_core_news_lg', 'pt_core_news_sm'],
            'nl': ['nl_core_news_lg', 'nl_core_news_sm'],
            'pl': ['pl_core_news_lg', 'pl_core_news_sm'],
            'ru': ['ru_core_news_lg', 'ru_core_news_sm'],
            'zh': ['zh_core_web_trf', 'zh_core_web_lg', 'zh_core_web_sm'],
            'ja': ['ja_core_news_trf', 'ja_core_news_lg', 'ja_core_news_sm'],
        }

        models_to_try = model_preferences.get(language, [f'{language}_core_news_sm'])

        nlp = None
        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name, disable=['ner'])
                break
            except OSError:
                continue

        if nlp is None:
            # Last resort: blank model with sentencizer
            try:
                nlp = spacy.blank(language)
                nlp.add_pipe('sentencizer')
            except Exception:
                nlp = spacy.blank('en')
                nlp.add_pipe('sentencizer')

        cls._spacy_cache[language] = nlp
        return nlp

    @classmethod
    def _segment_text(cls, text: str, language: str, level: str) -> List[str]:
        """
        Segment text by sentence using SOTA models.

        Hierarchy:
        1. wtpsplit (SaT model) - SOTA transformer for sentence segmentation
        2. spaCy transformer models (_trf) - High accuracy
        3. spaCy large/small models - Fallback
        """
        if not text:
            return []

        if level == 'paragraph':
            return [text]

        sentences = None

        # Try wtpsplit first (SOTA)
        wtp = cls._get_wtp_model()
        if wtp is not None:
            try:
                # wtpsplit returns list of sentences directly
                sentences = wtp.split(text, language=language)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Fallback to spaCy
        if not sentences:
            try:
                nlp = cls._get_spacy_model(language)
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                # Last resort: regex
                sentences = re.split(r'(?<=[.!?])\s+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

        # Post-process: merge fragments that start with lowercase
        sentences = cls._merge_lowercase_fragments(sentences)

        # Group sentences based on level
        if level == 'sentence' or level == '1':
            return sentences
        elif level == '2':
            return [' '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
        elif level == '3':
            return [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
        else:
            return sentences

    @classmethod
    def _merge_lowercase_fragments(cls, sentences: List[str]) -> List[str]:
        """
        Merge sentence fragments that clearly belong to the previous sentence.

        Only merges fragments that start with a lowercase letter, which is
        a definitive indicator of a mid-sentence split (not heuristic-based).
        """
        if not sentences or len(sentences) < 2:
            return sentences

        merged = []
        i = 0

        while i < len(sentences):
            current = sentences[i]

            # Merge any following fragments that start with lowercase
            while i + 1 < len(sentences):
                next_sent = sentences[i + 1].strip()
                if next_sent and next_sent[0].islower():
                    current = current.rstrip() + ' ' + next_sent
                    i += 1
                else:
                    break

            merged.append(current)
            i += 1

        return merged

    @staticmethod
    def _get_language_name(code: str) -> str:
        """Get full language name from code."""
        languages = {
            'fr': 'French', 'en': 'English', 'es': 'Spanish', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
            'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
            'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish', 'vi': 'Vietnamese',
            'id': 'Indonesian', 'th': 'Thai', 'sv': 'Swedish', 'no': 'Norwegian',
            'da': 'Danish', 'fi': 'Finnish', 'cs': 'Czech', 'el': 'Greek',
            'he': 'Hebrew', 'uk': 'Ukrainian', 'ro': 'Romanian', 'hu': 'Hungarian',
            'bg': 'Bulgarian', 'hr': 'Croatian', 'sk': 'Slovak', 'sl': 'Slovenian',
            'lt': 'Lithuanian', 'lv': 'Latvian', 'et': 'Estonian', 'ca': 'Catalan',
            'eu': 'Basque', 'gl': 'Galician'
        }
        return languages.get(code, code.upper())

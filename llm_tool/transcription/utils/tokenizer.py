"""
Text tokenization and segmentation module for Transcribe Tool.

Provides language-aware sentence and paragraph segmentation
for transcription output processing using SOTA models.

Model hierarchy:
1. wtpsplit (SaT) - SOTA transformer for sentence segmentation
2. spaCy transformer models (_trf) - High accuracy
3. spaCy large/small models - Fallback
4. NLTK - Legacy fallback
"""

import re
from typing import List, Optional, Dict, Any
from enum import Enum

# Try to import wtpsplit (SOTA sentence segmentation)
try:
    from wtpsplit import SaT
    HAS_WTPSPLIT = True
except ImportError:
    HAS_WTPSPLIT = False
    SaT = None

# Try to import spacy for better tokenization
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

# Try to import nltk as fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
    # Download punkt if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except:
            pass
except ImportError:
    HAS_NLTK = False


class SegmentationLevel(Enum):
    """Segmentation level options."""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SENTENCES_2 = "2_sentences"
    SENTENCES_3 = "3_sentences"
    SENTENCES_4 = "4_sentences"
    SENTENCES_5 = "5_sentences"
    CUSTOM = "custom"


# Spacy model preferences: transformer > large > small
SPACY_MODEL_PREFERENCES = {
    'en': ['en_core_web_trf', 'en_core_web_lg', 'en_core_web_sm'],
    'fr': ['fr_dep_news_trf', 'fr_core_news_lg', 'fr_core_news_sm'],
    'de': ['de_dep_news_trf', 'de_core_news_lg', 'de_core_news_sm'],
    'es': ['es_dep_news_trf', 'es_core_news_lg', 'es_core_news_sm'],
    'it': ['it_core_news_lg', 'it_core_news_sm'],
    'pt': ['pt_core_news_lg', 'pt_core_news_sm'],
    'nl': ['nl_core_news_lg', 'nl_core_news_sm'],
    'pl': ['pl_core_news_lg', 'pl_core_news_sm'],
    'ru': ['ru_core_news_lg', 'ru_core_news_sm'],
    'ja': ['ja_core_news_trf', 'ja_core_news_lg', 'ja_core_news_sm'],
    'zh': ['zh_core_web_trf', 'zh_core_web_lg', 'zh_core_web_sm'],
    'da': ['da_core_news_lg', 'da_core_news_sm'],
    'el': ['el_core_news_lg', 'el_core_news_sm'],
    'lt': ['lt_core_news_lg', 'lt_core_news_sm'],
    'nb': ['nb_core_news_lg', 'nb_core_news_sm'],
    'ro': ['ro_core_news_lg', 'ro_core_news_sm'],
}

# NLTK language mapping
NLTK_LANGUAGES = {
    'en': 'english', 'fr': 'french', 'de': 'german', 'es': 'spanish',
    'it': 'italian', 'pt': 'portuguese', 'nl': 'dutch', 'pl': 'polish',
    'ru': 'russian', 'cs': 'czech', 'da': 'danish', 'et': 'estonian',
    'fi': 'finnish', 'el': 'greek', 'no': 'norwegian', 'sl': 'slovene',
    'sv': 'swedish', 'tr': 'turkish'
}

# Global cache for models
_wtp_model = None
_spacy_cache: Dict[str, Any] = {}


class Tokenizer:
    """Language-aware text tokenizer and segmenter using SOTA models."""

    def __init__(self, language: str = 'en'):
        """
        Initialize tokenizer for a specific language.

        Args:
            language: ISO 639-1 language code
        """
        self.language = language
        self.nlp = None
        self.wtp = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize the best available tokenizer for the language."""
        global _wtp_model, _spacy_cache

        # Try wtpsplit first (SOTA)
        if HAS_WTPSPLIT:
            try:
                if _wtp_model is None:
                    _wtp_model = SaT("sat-3l")
                    # Move to GPU if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            _wtp_model.half().to("cuda")
                        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            _wtp_model.to("mps")
                    except Exception:
                        pass
                self.wtp = _wtp_model
            except Exception:
                pass

        # Initialize spaCy as fallback
        if HAS_SPACY:
            if self.language in _spacy_cache:
                self.nlp = _spacy_cache[self.language]
            else:
                models_to_try = SPACY_MODEL_PREFERENCES.get(
                    self.language,
                    [f'{self.language}_core_news_sm']
                )

                for model_name in models_to_try:
                    try:
                        self.nlp = spacy.load(model_name, disable=['ner'])
                        _spacy_cache[self.language] = self.nlp
                        break
                    except OSError:
                        continue

                if self.nlp is None:
                    try:
                        self.nlp = spacy.blank(self.language)
                        self.nlp.add_pipe('sentencizer')
                        _spacy_cache[self.language] = self.nlp
                    except Exception:
                        pass

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using SOTA models.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        sentences = None

        # Try wtpsplit first (SOTA)
        if self.wtp is not None:
            try:
                sentences = self.wtp.split(text, language=self.language)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Fallback to spaCy
        if not sentences and self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                sentences = None

        # Fallback to NLTK
        if not sentences and HAS_NLTK:
            try:
                nltk_lang = NLTK_LANGUAGES.get(self.language, 'english')
                sentences = sent_tokenize(text, language=nltk_lang)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Last resort: simple regex-based sentence splitting
        if not sentences:
            sentences = self._simple_sentence_split(text)

        # Merge fragments starting with lowercase
        return self._merge_lowercase_fragments(sentences)

    def _merge_lowercase_fragments(self, sentences: List[str]) -> List[str]:
        """Merge sentence fragments that start with lowercase."""
        if not sentences or len(sentences) < 2:
            return sentences

        merged = []
        i = 0

        while i < len(sentences):
            current = sentences[i]

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

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple regex-based sentence splitting as fallback."""
        # Handle common sentence endings
        # Be careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def segment(
        self,
        text: str,
        level: SegmentationLevel = SegmentationLevel.SENTENCE,
        sentences_per_segment: int = 1
    ) -> List[str]:
        """
        Segment text at specified level.

        Args:
            text: Input text
            level: Segmentation level
            sentences_per_segment: Number of sentences per segment (for CUSTOM level)

        Returns:
            List of text segments
        """
        if not text or not text.strip():
            return []

        # Get sentences first
        sentences = self.tokenize_sentences(text)

        if not sentences:
            return [text.strip()] if text.strip() else []

        # Determine grouping
        if level == SegmentationLevel.SENTENCE:
            return sentences
        elif level == SegmentationLevel.PARAGRAPH:
            # Group all sentences into one paragraph
            return [' '.join(sentences)]
        elif level == SegmentationLevel.SENTENCES_2:
            return self._group_sentences(sentences, 2)
        elif level == SegmentationLevel.SENTENCES_3:
            return self._group_sentences(sentences, 3)
        elif level == SegmentationLevel.SENTENCES_4:
            return self._group_sentences(sentences, 4)
        elif level == SegmentationLevel.SENTENCES_5:
            return self._group_sentences(sentences, 5)
        elif level == SegmentationLevel.CUSTOM:
            return self._group_sentences(sentences, sentences_per_segment)
        else:
            return sentences

    def _group_sentences(self, sentences: List[str], group_size: int) -> List[str]:
        """Group sentences into segments of specified size."""
        if group_size <= 0:
            group_size = 1

        segments = []
        for i in range(0, len(sentences), group_size):
            group = sentences[i:i + group_size]
            segments.append(' '.join(group))

        return segments

    def tokenize_words(self, text: str) -> List[str]:
        """
        Split text into words.

        Args:
            text: Input text

        Returns:
            List of words
        """
        if not text or not text.strip():
            return []

        if self.nlp:
            try:
                doc = self.nlp(text)
                return [token.text for token in doc if not token.is_space]
            except Exception:
                pass

        # Fallback: simple word splitting
        return text.split()

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(self.tokenize_words(text))

    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        return len(self.tokenize_sentences(text))


def segment_text(
    text: str,
    language: str = 'en',
    level: str = 'sentence',
    sentences_per_segment: int = 1
) -> List[str]:
    """
    Convenience function to segment text.

    Args:
        text: Input text
        language: ISO 639-1 language code
        level: Segmentation level ('sentence', 'paragraph', '2', '3', '4', '5', or custom number)
        sentences_per_segment: Number of sentences for custom level

    Returns:
        List of text segments
    """
    tokenizer = Tokenizer(language)

    # Parse level
    if level == 'sentence':
        seg_level = SegmentationLevel.SENTENCE
    elif level == 'paragraph':
        seg_level = SegmentationLevel.PARAGRAPH
    elif level == '2' or level == '2_sentences':
        seg_level = SegmentationLevel.SENTENCES_2
    elif level == '3' or level == '3_sentences':
        seg_level = SegmentationLevel.SENTENCES_3
    elif level == '4' or level == '4_sentences':
        seg_level = SegmentationLevel.SENTENCES_4
    elif level == '5' or level == '5_sentences':
        seg_level = SegmentationLevel.SENTENCES_5
    elif level.isdigit():
        seg_level = SegmentationLevel.CUSTOM
        sentences_per_segment = int(level)
    else:
        seg_level = SegmentationLevel.SENTENCE

    return tokenizer.segment(text, seg_level, sentences_per_segment)


def parse_segmentation_level(level_str: str) -> tuple[SegmentationLevel, int]:
    """
    Parse segmentation level from string.

    Args:
        level_str: Level string (e.g., 'sentence', 'paragraph', '3')

    Returns:
        Tuple of (SegmentationLevel, sentences_per_segment)
    """
    level_str = level_str.lower().strip()

    if level_str == 'sentence' or level_str == '1':
        return SegmentationLevel.SENTENCE, 1
    elif level_str == 'paragraph':
        return SegmentationLevel.PARAGRAPH, 0
    elif level_str == '2':
        return SegmentationLevel.SENTENCES_2, 2
    elif level_str == '3':
        return SegmentationLevel.SENTENCES_3, 3
    elif level_str == '4':
        return SegmentationLevel.SENTENCES_4, 4
    elif level_str == '5':
        return SegmentationLevel.SENTENCES_5, 5
    elif level_str.isdigit():
        n = int(level_str)
        return SegmentationLevel.CUSTOM, n
    else:
        return SegmentationLevel.SENTENCE, 1

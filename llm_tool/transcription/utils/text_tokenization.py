"""
SOTA Text Tokenization Module for Transcribe Tool.

Provides state-of-the-art text tokenization with configurable options:

SEGMENTATION LEVELS:
- 1, 2, 3, N sentences per segment
- Paragraph-based segmentation

NLP FEATURES (can be combined):
- Lemmatization (word base forms)
- POS tagging (part-of-speech)
- NER (named entity recognition)

OUTPUT FORMAT:
- CSV with JSON-encoded structured data for machine readability
- JSON for full data export

Model hierarchy:
1. wtpsplit (SaT-3l) - SOTA sentence segmentation, 85+ languages
2. spaCy transformer models (_trf) - Best NLP accuracy
3. spaCy large models - Good fallback
4. stanza - Academic SOTA alternative
"""

import json
import re
import csv
import io
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging

# ============================================================================
# Import NLP libraries with graceful fallbacks
# ============================================================================

# wtpsplit - SOTA sentence segmentation
try:
    from wtpsplit import SaT
    HAS_WTPSPLIT = True
except ImportError:
    HAS_WTPSPLIT = False
    SaT = None

# spaCy - NLP pipeline
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None

# stanza - Stanford NLP (fallback)
try:
    import stanza
    HAS_STANZA = True
except ImportError:
    HAS_STANZA = False
    stanza = None

# NLTK - fallback tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
except ImportError:
    HAS_NLTK = False


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class NLPFeatures:
    """Configuration for which NLP features to extract."""
    lemmatization: bool = False
    pos_tagging: bool = False
    ner: bool = False

    def any_enabled(self) -> bool:
        return self.lemmatization or self.pos_tagging or self.ner

    @classmethod
    def all_features(cls) -> 'NLPFeatures':
        return cls(lemmatization=True, pos_tagging=True, ner=True)

    @classmethod
    def none(cls) -> 'NLPFeatures':
        return cls()


@dataclass
class TokenInfo:
    """Information about a single token."""
    text: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    tag: Optional[str] = None  # Fine-grained POS

    def to_dict(self) -> Dict[str, Any]:
        d = {'text': self.text}
        if self.lemma:
            d['lemma'] = self.lemma
        if self.pos:
            d['pos'] = self.pos
        if self.tag:
            d['tag'] = self.tag
        return d


@dataclass
class EntityInfo:
    """Information about a named entity."""
    text: str
    label: str
    start_char: int
    end_char: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'label': self.label,
            'start': self.start_char,
            'end': self.end_char
        }


@dataclass
class TokenizedSegment:
    """A single tokenized segment with all extracted features."""
    segment_id: int
    document_id: str
    text: str
    sentences: List[str] = field(default_factory=list)
    tokens: List[TokenInfo] = field(default_factory=list)
    entities: List[EntityInfo] = field(default_factory=list)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences) if self.sentences else 1

    @property
    def word_count(self) -> int:
        return len(self.tokens) if self.tokens else len(self.text.split())

    def tokens_to_json(self) -> str:
        """Serialize tokens to JSON string."""
        if not self.tokens:
            return "[]"
        return json.dumps([t.to_dict() for t in self.tokens], ensure_ascii=False)

    def entities_to_json(self) -> str:
        """Serialize entities to JSON string."""
        if not self.entities:
            return "[]"
        return json.dumps([e.to_dict() for e in self.entities], ensure_ascii=False)

    def lemmas_to_json(self) -> str:
        """Get lemmas as JSON array."""
        if not self.tokens:
            return "[]"
        lemmas = [t.lemma for t in self.tokens if t.lemma]
        return json.dumps(lemmas, ensure_ascii=False)

    def pos_to_json(self) -> str:
        """Get POS tags as JSON array."""
        if not self.tokens:
            return "[]"
        pos_tags = [t.pos for t in self.tokens if t.pos]
        return json.dumps(pos_tags, ensure_ascii=False)


@dataclass
class TokenizationResult:
    """Result of tokenizing a document."""
    success: bool = True
    segments: List[TokenizedSegment] = field(default_factory=list)
    document_id: str = ""
    source_file: Optional[str] = None
    language: str = "unknown"
    segmentation_level: int = 1
    nlp_features: NLPFeatures = field(default_factory=NLPFeatures)
    total_sentences: int = 0
    total_words: int = 0
    total_entities: int = 0
    processing_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================
# Language-specific model configurations
# ============================================================================

# spaCy SOTA models per language (transformer > large > medium > small)
SPACY_MODELS = {
    'en': ['en_core_web_trf', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'],
    'fr': ['fr_dep_news_trf', 'fr_core_news_lg', 'fr_core_news_md', 'fr_core_news_sm'],
    'de': ['de_dep_news_trf', 'de_core_news_lg', 'de_core_news_md', 'de_core_news_sm'],
    'es': ['es_dep_news_trf', 'es_core_news_lg', 'es_core_news_md', 'es_core_news_sm'],
    'it': ['it_core_news_lg', 'it_core_news_md', 'it_core_news_sm'],
    'pt': ['pt_core_news_lg', 'pt_core_news_md', 'pt_core_news_sm'],
    'nl': ['nl_core_news_lg', 'nl_core_news_md', 'nl_core_news_sm'],
    'pl': ['pl_core_news_lg', 'pl_core_news_md', 'pl_core_news_sm'],
    'ru': ['ru_core_news_lg', 'ru_core_news_md', 'ru_core_news_sm'],
    'ja': ['ja_core_news_trf', 'ja_core_news_lg', 'ja_core_news_md', 'ja_core_news_sm'],
    'zh': ['zh_core_web_trf', 'zh_core_web_lg', 'zh_core_web_md', 'zh_core_web_sm'],
    'da': ['da_core_news_lg', 'da_core_news_md', 'da_core_news_sm'],
    'el': ['el_core_news_lg', 'el_core_news_md', 'el_core_news_sm'],
    'lt': ['lt_core_news_lg', 'lt_core_news_md', 'lt_core_news_sm'],
    'nb': ['nb_core_news_lg', 'nb_core_news_md', 'nb_core_news_sm'],
    'ro': ['ro_core_news_lg', 'ro_core_news_md', 'ro_core_news_sm'],
    'ca': ['ca_core_news_lg', 'ca_core_news_md', 'ca_core_news_sm'],
    'hr': ['hr_core_news_lg', 'hr_core_news_md', 'hr_core_news_sm'],
    'fi': ['fi_core_news_lg', 'fi_core_news_md', 'fi_core_news_sm'],
    'ko': ['ko_core_news_lg', 'ko_core_news_md', 'ko_core_news_sm'],
    'mk': ['mk_core_news_lg', 'mk_core_news_md', 'mk_core_news_sm'],
    'sv': ['sv_core_news_lg', 'sv_core_news_md', 'sv_core_news_sm'],
    'uk': ['uk_core_news_lg', 'uk_core_news_md', 'uk_core_news_sm'],
}

# NLTK language mapping
NLTK_LANGUAGES = {
    'en': 'english', 'fr': 'french', 'de': 'german', 'es': 'spanish',
    'it': 'italian', 'pt': 'portuguese', 'nl': 'dutch', 'pl': 'polish',
    'ru': 'russian', 'cs': 'czech', 'da': 'danish', 'et': 'estonian',
    'fi': 'finnish', 'el': 'greek', 'no': 'norwegian', 'nb': 'norwegian',
    'sl': 'slovene', 'sv': 'swedish', 'tr': 'turkish'
}

# ============================================================================
# Model cache (singleton for performance)
# ============================================================================

_wtp_model = None
_spacy_cache: Dict[str, Any] = {}
_stanza_cache: Dict[str, Any] = {}


class TextTokenizer:
    """
    SOTA multilingual text tokenizer with configurable segmentation and NLP features.

    Usage:
        tokenizer = TextTokenizer(
            language='fr',
            segmentation_level=1,  # 1 sentence per segment
            nlp_features=NLPFeatures(lemmatization=True, ner=True)
        )
        result = tokenizer.tokenize(text, document_id='doc1')
    """

    def __init__(
        self,
        language: str = 'en',
        segmentation_level: int = 1,  # 1, 2, 3, ... sentences per segment (0 = paragraph)
        nlp_features: Optional[NLPFeatures] = None,
        use_gpu: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize tokenizer.

        Args:
            language: ISO 639-1 language code
            segmentation_level: Number of sentences per segment (0 = paragraph mode)
            nlp_features: Which NLP features to extract
            use_gpu: Use GPU if available
            logger: Optional logger
        """
        self.language = language
        self.segmentation_level = segmentation_level
        self.nlp_features = nlp_features or NLPFeatures()
        self.use_gpu = use_gpu
        self.logger = logger or logging.getLogger(__name__)

        # Models
        self.wtp = None
        self.nlp = None  # spaCy
        self.stanza_nlp = None

        self._init_models()

    def _init_models(self):
        """Initialize required models."""
        global _wtp_model, _spacy_cache, _stanza_cache

        # Initialize wtpsplit for sentence segmentation
        if HAS_WTPSPLIT:
            try:
                if _wtp_model is None:
                    self.logger.info("Loading wtpsplit SaT model...")
                    _wtp_model = SaT("sat-3l")

                    if self.use_gpu:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                _wtp_model.half().to("cuda")
                            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                _wtp_model.to("mps")
                        except Exception:
                            pass

                self.wtp = _wtp_model
            except Exception as e:
                self.logger.warning(f"Could not load wtpsplit: {e}")

        # Initialize spaCy for NLP features
        if self.nlp_features.any_enabled() and HAS_SPACY:
            self._load_spacy_model()

        # Fallback to stanza if spaCy not available
        if self.nlp_features.any_enabled() and self.nlp is None and HAS_STANZA:
            self._load_stanza_model()

    def _load_spacy_model(self):
        """Load best available spaCy model for language."""
        global _spacy_cache

        if self.language in _spacy_cache:
            self.nlp = _spacy_cache[self.language]
            return

        models_to_try = SPACY_MODELS.get(
            self.language,
            [f'{self.language}_core_news_sm']
        )

        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                _spacy_cache[self.language] = self.nlp
                self.logger.info(f"Loaded spaCy model: {model_name}")
                return
            except OSError:
                continue

        # Fallback: blank model with sentencizer
        try:
            self.nlp = spacy.blank(self.language)
            self.nlp.add_pipe('sentencizer')
            _spacy_cache[self.language] = self.nlp
            self.logger.info(f"Using blank spaCy model for {self.language}")
        except Exception as e:
            self.logger.warning(f"Could not create spaCy model: {e}")

    def _load_stanza_model(self):
        """Load stanza model as fallback."""
        global _stanza_cache

        if self.language in _stanza_cache:
            self.stanza_nlp = _stanza_cache[self.language]
            return

        try:
            stanza.download(self.language, processors='tokenize,pos,lemma,ner', verbose=False)
            self.stanza_nlp = stanza.Pipeline(
                self.language,
                processors='tokenize,pos,lemma,ner',
                use_gpu=self.use_gpu,
                verbose=False
            )
            _stanza_cache[self.language] = self.stanza_nlp
            self.logger.info(f"Loaded stanza model for {self.language}")
        except Exception as e:
            self.logger.warning(f"Could not load stanza: {e}")

    def tokenize(
        self,
        text: str,
        document_id: str = "doc_1",
        source_file: Optional[str] = None
    ) -> TokenizationResult:
        """
        Tokenize text with configured segmentation and NLP features.

        Args:
            text: Input text
            document_id: Identifier for the document
            source_file: Optional source filename

        Returns:
            TokenizationResult with segments
        """
        if not text or not text.strip():
            return TokenizationResult(
                success=False,
                document_id=document_id,
                error="Empty text provided"
            )

        text = text.strip()

        try:
            # Step 1: Split into sentences
            sentences = self._split_sentences(text)

            # Step 2: Group sentences based on segmentation level
            if self.segmentation_level == 0:
                # Paragraph mode
                grouped = self._split_paragraphs(text)
            else:
                grouped = self._group_sentences(sentences, self.segmentation_level)

            # Step 3: Process each group with NLP features
            segments = []
            total_entities = 0

            for idx, (segment_text, segment_sentences) in enumerate(grouped, start=1):
                segment = TokenizedSegment(
                    segment_id=idx,
                    document_id=document_id,
                    text=segment_text,
                    sentences=segment_sentences
                )

                # Extract NLP features if enabled
                if self.nlp_features.any_enabled():
                    tokens, entities = self._extract_nlp_features(segment_text)
                    segment.tokens = tokens
                    segment.entities = entities
                    total_entities += len(entities)

                segments.append(segment)

            total_words = sum(seg.word_count for seg in segments)

            return TokenizationResult(
                success=True,
                segments=segments,
                document_id=document_id,
                source_file=source_file,
                language=self.language,
                segmentation_level=self.segmentation_level,
                nlp_features=self.nlp_features,
                total_sentences=len(sentences),
                total_words=total_words,
                total_entities=total_entities,
                processing_info={
                    'sentence_segmenter': 'wtpsplit' if self.wtp else 'spacy' if self.nlp else 'nltk',
                    'nlp_model': self.nlp.meta['name'] if self.nlp and hasattr(self.nlp, 'meta') else 'stanza' if self.stanza_nlp else 'none'
                }
            )

        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            return TokenizationResult(
                success=False,
                document_id=document_id,
                source_file=source_file,
                error=str(e)
            )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using best available method."""
        sentences = None

        # Method 1: wtpsplit (SOTA)
        if self.wtp is not None:
            try:
                sentences = self.wtp.split(text, language=self.language)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Method 2: spaCy
        if not sentences and self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                sentences = None

        # Method 3: stanza
        if not sentences and self.stanza_nlp:
            try:
                doc = self.stanza_nlp(text)
                sentences = [sent.text.strip() for sent in doc.sentences if sent.text.strip()]
            except Exception:
                sentences = None

        # Method 4: NLTK
        if not sentences and HAS_NLTK:
            try:
                nltk_lang = NLTK_LANGUAGES.get(self.language, 'english')
                sentences = sent_tokenize(text, language=nltk_lang)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Method 5: Regex fallback
        if not sentences:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        # Post-process: merge fragments starting with lowercase
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

    def _split_paragraphs(self, text: str) -> List[tuple]:
        """Split text by paragraphs (double newlines)."""
        paragraphs = re.split(r'\n\s*\n', text)
        result = []

        for para in paragraphs:
            para = para.strip()
            if para:
                sentences = self._split_sentences(para)
                result.append((para, sentences))

        return result

    def _group_sentences(self, sentences: List[str], group_size: int) -> List[tuple]:
        """Group sentences into segments of specified size."""
        if group_size <= 0:
            group_size = 1

        result = []
        for i in range(0, len(sentences), group_size):
            group = sentences[i:i + group_size]
            text = ' '.join(group)
            result.append((text, group))

        return result

    def _extract_nlp_features(self, text: str) -> tuple:
        """Extract tokens and entities using spaCy or stanza."""
        tokens = []
        entities = []

        if self.nlp:
            doc = self.nlp(text)

            # Extract tokens
            for token in doc:
                if not token.is_space:
                    info = TokenInfo(text=token.text)
                    if self.nlp_features.lemmatization:
                        info.lemma = token.lemma_
                    if self.nlp_features.pos_tagging:
                        info.pos = token.pos_
                        info.tag = token.tag_
                    tokens.append(info)

            # Extract entities
            if self.nlp_features.ner:
                for ent in doc.ents:
                    entities.append(EntityInfo(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char
                    ))

        elif self.stanza_nlp:
            doc = self.stanza_nlp(text)

            for sent in doc.sentences:
                for word in sent.words:
                    info = TokenInfo(text=word.text)
                    if self.nlp_features.lemmatization:
                        info.lemma = word.lemma
                    if self.nlp_features.pos_tagging:
                        info.pos = word.upos
                        info.tag = word.xpos
                    tokens.append(info)

                # Stanza NER
                if self.nlp_features.ner and hasattr(sent, 'ents'):
                    for ent in sent.ents:
                        entities.append(EntityInfo(
                            text=ent.text,
                            label=ent.type,
                            start_char=ent.start_char,
                            end_char=ent.end_char
                        ))

        else:
            # Fallback: just word tokenization
            for word in text.split():
                tokens.append(TokenInfo(text=word))

        return tokens, entities


# ============================================================================
# Output formatters
# ============================================================================

def format_tokenization_csv(
    results: List[TokenizationResult],
    include_source: bool = True
) -> str:
    """
    Format tokenization results as CSV with JSON-encoded structured data.

    Columns:
    - segment_id: Segment number within document
    - document_id: Document identifier
    - text: Raw text of the segment
    - sentences: JSON array of sentences
    - tokens: JSON array of token objects (if NLP features enabled)
    - entities: JSON array of entity objects (if NER enabled)
    - sentence_count: Number of sentences
    - word_count: Number of words
    - source_file: Source filename (optional)
    - language: Detected/specified language
    """
    output = io.StringIO()

    # Determine columns based on features used
    base_columns = ['segment_id', 'document_id', 'text', 'sentences', 'sentence_count', 'word_count']

    # Check for NLP features
    has_tokens = any(
        any(seg.tokens for seg in r.segments)
        for r in results if r.success
    )
    has_entities = any(
        any(seg.entities for seg in r.segments)
        for r in results if r.success
    )

    columns = base_columns.copy()
    if has_tokens:
        columns.extend(['tokens', 'lemmas', 'pos_tags'])
    if has_entities:
        columns.append('entities')
    if include_source:
        columns.extend(['source_file', 'language'])

    writer = csv.DictWriter(output, fieldnames=columns, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    for result in results:
        if not result.success:
            continue

        for segment in result.segments:
            row = {
                'segment_id': segment.segment_id,
                'document_id': segment.document_id,
                'text': segment.text,
                'sentences': json.dumps(segment.sentences, ensure_ascii=False),
                'sentence_count': segment.sentence_count,
                'word_count': segment.word_count,
            }

            if has_tokens:
                row['tokens'] = segment.tokens_to_json()
                row['lemmas'] = segment.lemmas_to_json()
                row['pos_tags'] = segment.pos_to_json()

            if has_entities:
                row['entities'] = segment.entities_to_json()

            if include_source:
                row['source_file'] = result.source_file or ''
                row['language'] = result.language

            writer.writerow(row)

    return output.getvalue()


def format_tokenization_json(results: List[TokenizationResult]) -> str:
    """Format tokenization results as JSON."""
    output = []

    for result in results:
        if not result.success:
            continue

        doc_data = {
            'document_id': result.document_id,
            'source_file': result.source_file,
            'language': result.language,
            'segmentation_level': result.segmentation_level,
            'nlp_features': {
                'lemmatization': result.nlp_features.lemmatization,
                'pos_tagging': result.nlp_features.pos_tagging,
                'ner': result.nlp_features.ner
            },
            'statistics': {
                'total_segments': len(result.segments),
                'total_sentences': result.total_sentences,
                'total_words': result.total_words,
                'total_entities': result.total_entities
            },
            'processing_info': result.processing_info,
            'segments': []
        }

        for seg in result.segments:
            seg_data = {
                'segment_id': seg.segment_id,
                'text': seg.text,
                'sentences': seg.sentences,
                'sentence_count': seg.sentence_count,
                'word_count': seg.word_count,
            }

            if seg.tokens:
                seg_data['tokens'] = [t.to_dict() for t in seg.tokens]

            if seg.entities:
                seg_data['entities'] = [e.to_dict() for e in seg.entities]

            doc_data['segments'].append(seg_data)

        output.append(doc_data)

    return json.dumps(output, ensure_ascii=False, indent=2)


# ============================================================================
# Convenience functions
# ============================================================================

def tokenize_text(
    text: str,
    language: str = 'en',
    segmentation_level: int = 1,
    lemmatization: bool = False,
    pos_tagging: bool = False,
    ner: bool = False,
    document_id: str = 'doc_1'
) -> TokenizationResult:
    """
    Convenience function to tokenize text.

    Args:
        text: Input text
        language: ISO 639-1 language code
        segmentation_level: Sentences per segment (0 = paragraph)
        lemmatization: Extract lemmas
        pos_tagging: Extract POS tags
        ner: Extract named entities
        document_id: Document identifier

    Returns:
        TokenizationResult
    """
    features = NLPFeatures(
        lemmatization=lemmatization,
        pos_tagging=pos_tagging,
        ner=ner
    )

    tokenizer = TextTokenizer(
        language=language,
        segmentation_level=segmentation_level,
        nlp_features=features
    )

    return tokenizer.tokenize(text, document_id=document_id)


def get_available_nlp_features() -> List[Dict[str, str]]:
    """Get list of available NLP features."""
    return [
        {
            'id': 'lemmatization',
            'name': 'Lemmatization',
            'description': 'Extract word base forms (lemmas)'
        },
        {
            'id': 'pos_tagging',
            'name': 'POS Tagging',
            'description': 'Part-of-speech tags for each word'
        },
        {
            'id': 'ner',
            'name': 'Named Entity Recognition',
            'description': 'Extract named entities (people, places, organizations)'
        }
    ]


def get_supported_languages() -> List[Dict[str, str]]:
    """Get list of languages with good NLP support."""
    return [
        {'code': 'en', 'name': 'English', 'spacy_model': 'en_core_web_trf'},
        {'code': 'fr', 'name': 'French', 'spacy_model': 'fr_dep_news_trf'},
        {'code': 'de', 'name': 'German', 'spacy_model': 'de_dep_news_trf'},
        {'code': 'es', 'name': 'Spanish', 'spacy_model': 'es_dep_news_trf'},
        {'code': 'it', 'name': 'Italian', 'spacy_model': 'it_core_news_lg'},
        {'code': 'pt', 'name': 'Portuguese', 'spacy_model': 'pt_core_news_lg'},
        {'code': 'nl', 'name': 'Dutch', 'spacy_model': 'nl_core_news_lg'},
        {'code': 'pl', 'name': 'Polish', 'spacy_model': 'pl_core_news_lg'},
        {'code': 'ru', 'name': 'Russian', 'spacy_model': 'ru_core_news_lg'},
        {'code': 'ja', 'name': 'Japanese', 'spacy_model': 'ja_core_news_trf'},
        {'code': 'zh', 'name': 'Chinese', 'spacy_model': 'zh_core_web_trf'},
        {'code': 'ko', 'name': 'Korean', 'spacy_model': 'ko_core_news_lg'},
        {'code': 'da', 'name': 'Danish', 'spacy_model': 'da_core_news_lg'},
        {'code': 'sv', 'name': 'Swedish', 'spacy_model': 'sv_core_news_lg'},
        {'code': 'nb', 'name': 'Norwegian', 'spacy_model': 'nb_core_news_lg'},
        {'code': 'fi', 'name': 'Finnish', 'spacy_model': 'fi_core_news_lg'},
        {'code': 'el', 'name': 'Greek', 'spacy_model': 'el_core_news_lg'},
        {'code': 'ro', 'name': 'Romanian', 'spacy_model': 'ro_core_news_lg'},
        {'code': 'uk', 'name': 'Ukrainian', 'spacy_model': 'uk_core_news_lg'},
        {'code': 'ca', 'name': 'Catalan', 'spacy_model': 'ca_core_news_lg'},
        {'code': 'hr', 'name': 'Croatian', 'spacy_model': 'hr_core_news_lg'},
        {'code': 'lt', 'name': 'Lithuanian', 'spacy_model': 'lt_core_news_lg'},
        {'code': 'mk', 'name': 'Macedonian', 'spacy_model': 'mk_core_news_lg'},
    ]

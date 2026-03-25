"""Utility modules for Transcribe Tool."""

from .language_detector import (
    LanguageDetector,
    DetectionMethod,
    detect_language,
    detect_language_full,
    get_language_name,
    get_supported_languages as get_supported_detection_languages,
    get_detection_info,
    LANGUAGE_CODES,
)
from .tokenizer import Tokenizer, segment_text
from .audio_processor import AudioProcessor
from .document_loader import DocumentLoader, Document, load_document, load_documents
from .text_tokenization import (
    TextTokenizer,
    NLPFeatures,
    TokenizationResult,
    TokenizedSegment,
    tokenize_text,
    format_tokenization_csv,
    format_tokenization_json,
    get_available_nlp_features,
    get_supported_languages,
)

__all__ = [
    # Language detection (SOTA)
    "LanguageDetector",
    "DetectionMethod",
    "detect_language",
    "detect_language_full",
    "get_language_name",
    "get_supported_detection_languages",
    "get_detection_info",
    "LANGUAGE_CODES",
    # Legacy tokenizer
    "Tokenizer",
    "segment_text",
    # Audio processing
    "AudioProcessor",
    # Document loading
    "DocumentLoader",
    "Document",
    "load_document",
    "load_documents",
    # SOTA Text tokenization
    "TextTokenizer",
    "NLPFeatures",
    "TokenizationResult",
    "TokenizedSegment",
    "tokenize_text",
    "format_tokenization_csv",
    "format_tokenization_json",
    "get_available_nlp_features",
    "get_supported_languages",
]

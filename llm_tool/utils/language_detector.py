#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
language_detector.py

MAIN OBJECTIVE:
---------------
This script provides automatic language detection functionality for text data,
supporting multiple languages and providing confidence scores.

Dependencies:
-------------
- sys
- langdetect
- langid
- fasttext
- typing

MAIN FEATURES:
--------------
1) Automatic language detection with confidence scores
2) Support for multiple detection libraries
3) Batch processing capability
4) Fallback mechanisms for uncertain detections
5) Language code standardization

Author:
-------
Antoine Lemor
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import warnings

# Try multiple language detection libraries
try:
    from langdetect import detect, detect_langs, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    detect = detect_langs = LangDetectException = None

try:
    import langid
    HAS_LANGID = True
except ImportError:
    HAS_LANGID = False
    langid = None

try:
    import fasttext
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False
    fasttext = None


class DetectionMethod(Enum):
    """Available language detection methods"""
    LANGDETECT = "langdetect"
    LANGID = "langid"
    FASTTEXT = "fasttext"
    ENSEMBLE = "ensemble"


class LanguageDetector:
    """Multi-method language detection with fallback support"""

    # ISO 639-1 language codes mapping
    LANGUAGE_CODES = {
        'en': 'english',
        'fr': 'french',
        'es': 'spanish',
        'de': 'german',
        'it': 'italian',
        'pt': 'portuguese',
        'nl': 'dutch',
        'ru': 'russian',
        'zh': 'chinese',
        'ja': 'japanese',
        'ar': 'arabic',
        'hi': 'hindi',
        'sv': 'swedish',
        'pl': 'polish',
        'tr': 'turkish',
        'ko': 'korean',
        'vi': 'vietnamese',
        'th': 'thai',
        'he': 'hebrew',
        'id': 'indonesian',
        'ms': 'malay',
        'cs': 'czech',
        'da': 'danish',
        'fi': 'finnish',
        'no': 'norwegian',
        'uk': 'ukrainian',
        'ro': 'romanian',
        'hu': 'hungarian',
        'el': 'greek',
        'bg': 'bulgarian',
        'sr': 'serbian',
        'hr': 'croatian',
        'sk': 'slovak',
        'sl': 'slovenian',
        'lt': 'lithuanian',
        'lv': 'latvian',
        'et': 'estonian',
        'fa': 'persian',
        'ur': 'urdu',
        'bn': 'bengali',
        'ta': 'tamil',
        'te': 'telugu',
        'ml': 'malayalam',
        'kn': 'kannada',
        'gu': 'gujarati',
        'mr': 'marathi',
        'pa': 'punjabi'
    }

    # Model names for specific languages
    LANGUAGE_MODELS = {
        'en': ['bert-base-uncased', 'roberta-base', 'deberta-v3-base'],
        'fr': ['camembert-base', 'flaubert_base_cased', 'barthez-base'],
        'es': ['dccuchile/bert-base-spanish-wwm-cased', 'BSC-TeMU/roberta-base-bne'],
        'de': ['bert-base-german-cased', 'deepset/gbert-base'],
        'it': ['dbmdz/bert-base-italian-cased', 'idb-ita/gilberto-uncased-from-camembert'],
        'pt': ['neuralmind/bert-base-portuguese-cased', 'portuguese-bert-base'],
        'nl': ['GroNLP/bert-base-dutch-cased', 'wietsedv/bert-base-dutch-cased'],
        'ru': ['DeepPavlov/rubert-base-cased', 'sberbank-ai/ruBert-base'],
        'zh': ['bert-base-chinese', 'hfl/chinese-bert-wwm-ext'],
        'ja': ['cl-tohoku/bert-base-japanese', 'nlp-waseda/roberta-base-japanese'],
        'ar': ['asafaya/bert-base-arabic', 'CAMeL-Lab/bert-base-arabic-camelbert-ca'],
        'multilingual': ['bert-base-multilingual-cased', 'xlm-roberta-base', 'google/mt5-base']
    }

    def __init__(self, method: DetectionMethod = DetectionMethod.ENSEMBLE,
                 confidence_threshold: float = 0.8,
                 fallback_language: str = 'en',
                 fasttext_model_path: Optional[str] = None):
        """
        Initialize the language detector

        Args:
            method: Detection method to use
            confidence_threshold: Minimum confidence for detection
            fallback_language: Language to use when detection fails
            fasttext_model_path: Path to FastText model (if using FastText)
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.fallback_language = fallback_language
        self.logger = logging.getLogger(__name__)

        # Check available libraries
        self._check_available_libraries()

        # Load FastText model if provided
        self.fasttext_model = None
        if HAS_FASTTEXT and fasttext_model_path:
            try:
                # Suppress FastText warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.fasttext_model = fasttext.load_model(fasttext_model_path)
                self.logger.info(f"FastText model loaded from {fasttext_model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load FastText model: {e}")

    def _check_available_libraries(self):
        """Check which language detection libraries are available"""
        available = []
        if HAS_LANGDETECT:
            available.append("langdetect")
        if HAS_LANGID:
            available.append("langid")
        if HAS_FASTTEXT:
            available.append("fasttext")

        if not available:
            self.logger.warning(
                "No language detection libraries found. "
                "Install with: pip install langdetect langid"
            )
            self.method = None
        else:
            self.logger.info(f"Available language detection libraries: {', '.join(available)}")

            # Adjust method if requested library not available
            if self.method == DetectionMethod.LANGDETECT and not HAS_LANGDETECT:
                self.method = DetectionMethod.LANGID if HAS_LANGID else None
            elif self.method == DetectionMethod.LANGID and not HAS_LANGID:
                self.method = DetectionMethod.LANGDETECT if HAS_LANGDETECT else None
            elif self.method == DetectionMethod.FASTTEXT and not HAS_FASTTEXT:
                self.method = DetectionMethod.ENSEMBLE

    def detect(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Detect language of text

        Args:
            text: Text to analyze
            return_all_scores: Return scores for all detected languages

        Returns:
            Dictionary with language code, confidence, and method used
        """
        if not text or not text.strip():
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'fallback',
                'all_scores': []
            }

        # Clean text
        text = text.strip()

        # Use specified method
        if self.method == DetectionMethod.LANGDETECT:
            result = self._detect_langdetect(text, return_all_scores)
        elif self.method == DetectionMethod.LANGID:
            result = self._detect_langid(text, return_all_scores)
        elif self.method == DetectionMethod.FASTTEXT:
            result = self._detect_fasttext(text, return_all_scores)
        elif self.method == DetectionMethod.ENSEMBLE:
            result = self._detect_ensemble(text, return_all_scores)
        else:
            result = {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'fallback',
                'all_scores': []
            }

        # Apply confidence threshold
        if result['confidence'] < self.confidence_threshold:
            result['language'] = self.fallback_language
            result['method'] = f"{result['method']}+fallback"

        return result

    def _detect_langdetect(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Detect language using langdetect library"""
        try:
            if return_all_scores:
                langs = detect_langs(text)
                all_scores = [(lang.lang, lang.prob) for lang in langs]
                if langs:
                    return {
                        'language': langs[0].lang,
                        'confidence': langs[0].prob,
                        'method': 'langdetect',
                        'all_scores': all_scores
                    }
            else:
                lang = detect(text)
                return {
                    'language': lang,
                    'confidence': 1.0,  # langdetect doesn't return confidence for single detect
                    'method': 'langdetect',
                    'all_scores': []
                }
        except (LangDetectException, Exception) as e:
            self.logger.debug(f"Langdetect failed: {e}")

        return {
            'language': self.fallback_language,
            'confidence': 0.0,
            'method': 'langdetect_failed',
            'all_scores': []
        }

    def _detect_langid(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Detect language using langid library"""
        try:
            lang, confidence = langid.classify(text)

            result = {
                'language': lang,
                'confidence': confidence,
                'method': 'langid',
                'all_scores': [(lang, confidence)]
            }

            if return_all_scores:
                # langid doesn't provide multiple language scores by default
                # We can set it to return top-k languages
                langid.set_languages(None)  # Reset to all languages
                result['all_scores'] = [(lang, confidence)]

            return result
        except Exception as e:
            self.logger.debug(f"Langid failed: {e}")

        return {
            'language': self.fallback_language,
            'confidence': 0.0,
            'method': 'langid_failed',
            'all_scores': []
        }

    def _detect_fasttext(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Detect language using FastText"""
        if not self.fasttext_model:
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'fasttext_not_loaded',
                'all_scores': []
            }

        try:
            # FastText expects single line text
            text = text.replace('\n', ' ')

            # Get predictions
            predictions = self.fasttext_model.predict(text, k=5 if return_all_scores else 1)
            labels, probs = predictions

            # Extract language code from label (format: __label__en)
            lang = labels[0].replace('__label__', '')
            confidence = probs[0]

            all_scores = []
            if return_all_scores:
                all_scores = [(label.replace('__label__', ''), prob)
                             for label, prob in zip(labels, probs)]

            return {
                'language': lang,
                'confidence': confidence,
                'method': 'fasttext',
                'all_scores': all_scores
            }
        except Exception as e:
            self.logger.debug(f"FastText failed: {e}")

        return {
            'language': self.fallback_language,
            'confidence': 0.0,
            'method': 'fasttext_failed',
            'all_scores': []
        }

    def _detect_ensemble(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Use ensemble of methods for more robust detection"""
        results = []
        methods_used = []

        # Try langdetect
        if HAS_LANGDETECT:
            result = self._detect_langdetect(text, False)
            if result['confidence'] > 0:
                results.append(result)
                methods_used.append('langdetect')

        # Try langid
        if HAS_LANGID:
            result = self._detect_langid(text, False)
            if result['confidence'] > 0:
                results.append(result)
                methods_used.append('langid')

        # Try fasttext
        if self.fasttext_model:
            result = self._detect_fasttext(text, False)
            if result['confidence'] > 0:
                results.append(result)
                methods_used.append('fasttext')

        if not results:
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'ensemble_failed',
                'all_scores': []
            }

        # Aggregate results
        language_votes = {}
        for result in results:
            lang = result['language']
            conf = result['confidence']
            if lang in language_votes:
                language_votes[lang].append(conf)
            else:
                language_votes[lang] = [conf]

        # Calculate weighted scores
        language_scores = {}
        for lang, confidences in language_votes.items():
            # Average confidence weighted by number of votes
            avg_confidence = sum(confidences) / len(confidences)
            vote_weight = len(confidences) / len(results)
            language_scores[lang] = avg_confidence * vote_weight

        # Get best language
        best_lang = max(language_scores, key=language_scores.get)
        best_score = language_scores[best_lang]

        all_scores = []
        if return_all_scores:
            all_scores = sorted(
                [(lang, score) for lang, score in language_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )

        return {
            'language': best_lang,
            'confidence': best_score,
            'method': f"ensemble({'+'.join(methods_used)})",
            'all_scores': all_scores
        }

    def detect_batch(self, texts: List[str], parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Detect languages for multiple texts

        Args:
            texts: List of texts to analyze
            parallel: Use parallel processing

        Returns:
            List of detection results
        """
        if parallel and len(texts) > 10:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(self.detect, texts))
        else:
            results = [self.detect(text) for text in texts]

        return results

    def get_language_name(self, code: str) -> str:
        """Get full language name from ISO code"""
        return self.LANGUAGE_CODES.get(code, code)

    def get_recommended_models(self, language_code: str) -> List[str]:
        """Get recommended models for a language"""
        if language_code in self.LANGUAGE_MODELS:
            return self.LANGUAGE_MODELS[language_code]
        else:
            return self.LANGUAGE_MODELS['multilingual']

    def is_supported(self, language_code: str) -> bool:
        """Check if a language is supported"""
        return language_code in self.LANGUAGE_CODES

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_CODES.keys())
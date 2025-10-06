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
- lingua-language-detector (PRIMARY - 96%+ accuracy, most reliable)
- langid (FALLBACK - 90%+ accuracy)
- fasttext (OPTIONAL - 93%+ accuracy, requires model download)
- typing

MAIN FEATURES:
--------------
1) High-accuracy language detection (96%+) using lingua-language-detector
2) Automatic language detection with confidence scores
3) Support for 75+ languages (lingua) or 100+ (fallback methods)
4) Batch processing capability for large datasets
5) Ensemble mode combining multiple detectors
6) Deterministic results (no randomness like old langdetect)
7) Fallback mechanisms for uncertain detections
8) ISO 639-1 language code standardization

Author:
-------
Antoine Lemor
"""

import logging
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import warnings

# Primary: lingua-language-detector (most accurate)
try:
    from lingua import Language, LanguageDetectorBuilder
    HAS_LINGUA = True
except ImportError:
    HAS_LINGUA = False
    Language = LanguageDetectorBuilder = None

# Fallback: langid
try:
    import langid
    HAS_LANGID = True
except ImportError:
    HAS_LANGID = False
    langid = None

# Fallback: fasttext (if model provided)
try:
    import fasttext
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False
    fasttext = None


class DetectionMethod(Enum):
    """Available language detection methods"""
    LINGUA = "lingua"  # Primary: most accurate
    LANGID = "langid"  # Fallback
    FASTTEXT = "fasttext"  # Fallback (requires model)
    ENSEMBLE = "ensemble"  # Use multiple methods


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

    def __init__(self, method: DetectionMethod = DetectionMethod.LINGUA,
                 confidence_threshold: float = 0.7,
                 fallback_language: str = 'en',
                 fasttext_model_path: Optional[str] = None):
        """
        Initialize the language detector

        Args:
            method: Detection method to use (default: LINGUA - most accurate)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            fallback_language: Language to use when detection fails
            fasttext_model_path: Path to FastText model (if using FastText)
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.fallback_language = fallback_language
        self.logger = logging.getLogger(__name__)

        # Check available libraries
        self._check_available_libraries()

        # Initialize lingua detector (most accurate)
        self.lingua_detector = None
        if HAS_LINGUA:
            try:
                # Build detector with common languages for speed
                # For maximum accuracy, use .from_all_languages()
                languages = [
                    Language.ENGLISH, Language.FRENCH, Language.SPANISH,
                    Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE,
                    Language.DUTCH, Language.RUSSIAN, Language.CHINESE,
                    Language.JAPANESE, Language.ARABIC, Language.POLISH,
                    Language.TURKISH, Language.KOREAN, Language.HINDI,
                    Language.SWEDISH, Language.BOKMAL, Language.DANISH,
                    Language.FINNISH, Language.CZECH, Language.GREEK,
                    Language.HEBREW, Language.ROMANIAN, Language.UKRAINIAN,
                    Language.BULGARIAN, Language.CROATIAN, Language.VIETNAMESE,
                    Language.THAI, Language.INDONESIAN, Language.PERSIAN
                ]
                self.lingua_detector = LanguageDetectorBuilder.from_languages(*languages).build()
                self.logger.debug("Lingua detector initialized with 30 languages")
            except Exception as e:
                self.logger.warning(f"Could not initialize Lingua detector: {e}")

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
        if HAS_LINGUA:
            available.append("lingua")
        if HAS_LANGID:
            available.append("langid")
        if HAS_FASTTEXT:
            available.append("fasttext")

        if not available:
            self.logger.warning(
                "No language detection libraries found. "
                "Install with: pip install lingua-language-detector"
            )
            self.method = None
        else:
            self.logger.debug(f"Available language detection libraries: {', '.join(available)}")

            # Adjust method if requested library not available
            if self.method == DetectionMethod.LINGUA and not HAS_LINGUA:
                self.method = DetectionMethod.LANGID if HAS_LANGID else None
                self.logger.warning("Lingua not available, falling back to langid")
            elif self.method == DetectionMethod.LANGID and not HAS_LANGID:
                self.method = DetectionMethod.LINGUA if HAS_LINGUA else None
            elif self.method == DetectionMethod.FASTTEXT and not HAS_FASTTEXT:
                self.method = DetectionMethod.LINGUA if HAS_LINGUA else DetectionMethod.LANGID

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
        if self.method == DetectionMethod.LINGUA:
            result = self._detect_lingua(text, return_all_scores)
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

    def _detect_lingua(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Detect language using lingua library (most accurate)"""
        if not self.lingua_detector:
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'lingua_not_loaded',
                'all_scores': []
            }

        try:
            # Lingua ISO code mapping
            LINGUA_TO_ISO = {
                'ENGLISH': 'en', 'FRENCH': 'fr', 'SPANISH': 'es', 'GERMAN': 'de',
                'ITALIAN': 'it', 'PORTUGUESE': 'pt', 'DUTCH': 'nl', 'RUSSIAN': 'ru',
                'CHINESE': 'zh', 'JAPANESE': 'ja', 'ARABIC': 'ar', 'POLISH': 'pl',
                'TURKISH': 'tr', 'KOREAN': 'ko', 'HINDI': 'hi', 'SWEDISH': 'sv',
                'BOKMAL': 'no', 'NYNORSK': 'no', 'DANISH': 'da', 'FINNISH': 'fi',
                'CZECH': 'cs', 'GREEK': 'el', 'HEBREW': 'he', 'ROMANIAN': 'ro',
                'UKRAINIAN': 'uk', 'BULGARIAN': 'bg', 'CROATIAN': 'hr', 'VIETNAMESE': 'vi',
                'THAI': 'th', 'INDONESIAN': 'id', 'PERSIAN': 'fa', 'SLOVAK': 'sk',
                'SLOVENE': 'sl', 'LITHUANIAN': 'lt', 'LATVIAN': 'lv', 'ESTONIAN': 'et',
                'ALBANIAN': 'sq', 'MACEDONIAN': 'mk', 'SERBIAN': 'sr', 'MALAY': 'ms',
                'BENGALI': 'bn', 'URDU': 'ur', 'TAMIL': 'ta', 'TELUGU': 'te',
                'KANNADA': 'kn', 'MALAYALAM': 'ml', 'GUJARATI': 'gu', 'MARATHI': 'mr',
                'PUNJABI': 'pa', 'AFRIKAANS': 'af', 'ARMENIAN': 'hy', 'AZERBAIJANI': 'az',
                'BASQUE': 'eu', 'BELARUSIAN': 'be', 'BOSNIAN': 'bs', 'CATALAN': 'ca',
                'ESPERANTO': 'eo', 'GEORGIAN': 'ka', 'GANDA': 'lg', 'ICELANDIC': 'is',
                'IRISH': 'ga', 'KAZAKH': 'kk', 'LATIN': 'la', 'MAORI': 'mi',
                'MONGOLIAN': 'mn', 'SHONA': 'sn', 'SOMALI': 'so', 'SOTHO': 'st',
                'SWAHILI': 'sw', 'TAGALOG': 'tl', 'TSONGA': 'ts', 'TSWANA': 'tn',
                'XHOSA': 'xh', 'YORUBA': 'yo', 'ZULU': 'zu'
            }

            if return_all_scores:
                # Get confidence values for all languages
                confidence_values = self.lingua_detector.compute_language_confidence_values(text)
                all_scores = []

                for lang_confidence in confidence_values:
                    lang_name = str(lang_confidence.language).split('.')[-1]
                    iso_code = LINGUA_TO_ISO.get(lang_name, lang_name.lower()[:2])
                    all_scores.append((iso_code, lang_confidence.value))

                if all_scores:
                    all_scores.sort(key=lambda x: x[1], reverse=True)
                    return {
                        'language': all_scores[0][0],
                        'confidence': all_scores[0][1],
                        'method': 'lingua',
                        'all_scores': all_scores
                    }
            else:
                # Single detection with confidence
                detected = self.lingua_detector.detect_language_of(text)
                if detected:
                    lang_name = str(detected).split('.')[-1]
                    iso_code = LINGUA_TO_ISO.get(lang_name, lang_name.lower()[:2])

                    # Get confidence for detected language
                    confidence_values = self.lingua_detector.compute_language_confidence_values(text)
                    confidence = 0.95  # Default high confidence

                    for lang_conf in confidence_values:
                        if str(lang_conf.language).split('.')[-1] == lang_name:
                            confidence = lang_conf.value
                            break

                    return {
                        'language': iso_code,
                        'confidence': confidence,
                        'method': 'lingua',
                        'all_scores': [(iso_code, confidence)]
                    }
        except Exception as e:
            self.logger.debug(f"Lingua detection failed: {e}")

        return {
            'language': self.fallback_language,
            'confidence': 0.0,
            'method': 'lingua_failed',
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

        # Try lingua (most accurate - highest weight)
        if HAS_LINGUA and self.lingua_detector:
            result = self._detect_lingua(text, False)
            if result['confidence'] > 0:
                # Give lingua double weight in ensemble
                results.append(result)
                results.append(result)
                methods_used.append('lingua')

        # Try langid (good fallback)
        if HAS_LANGID:
            result = self._detect_langid(text, False)
            if result['confidence'] > 0:
                results.append(result)
                methods_used.append('langid')

        # Try fasttext (if available)
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
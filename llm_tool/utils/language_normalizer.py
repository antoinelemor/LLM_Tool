#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
language_normalizer.py

MAIN OBJECTIVE:
---------------
Language normalization utilities shared across the LLM Tool package, providing centralized
logic for language detection, normalization and mapping that was previously embedded inside
CLI components, enabling low-level training and data-preparation code to depend on the same
normalisation rules without creating circular imports.

Dependencies:
-------------
- typing
- llm_tool.utils.language_detector

MAIN FEATURES:
--------------
1) Comprehensive language mapping dictionary (29+ languages)
2) Language code normalization to standard ISO two-letter codes
3) Batch language detection for datasets with confidence scoring
4) Dataset language analysis with automatic normalization
5) Model recommendations based on detected languages

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .language_detector import LanguageDetector


class LanguageNormalizer:
    """Intelligent language normalization and mapping system."""

    # Comprehensive language mapping dictionary
    LANGUAGE_MAPPINGS: Dict[str, List[str]] = {
        "en": ["en", "eng", "english", "anglais"],
        "fr": ["fr", "fra", "fre", "french", "français", "francais"],
        "de": ["de", "deu", "ger", "german", "deutsch", "allemand"],
        "es": ["es", "spa", "spanish", "español", "espagnol"],
        "it": ["it", "ita", "italian", "italiano", "italien"],
        "pt": ["pt", "por", "portuguese", "português", "portugais"],
        "nl": ["nl", "nld", "dut", "dutch", "nederlands", "néerlandais"],
        "ru": ["ru", "rus", "russian", "русский", "russe"],
        "zh": ["zh", "chi", "zho", "chinese", "中文", "chinois"],
        "ja": ["ja", "jpn", "japanese", "日本語", "japonais"],
        "ar": ["ar", "ara", "arabic", "العربية", "arabe"],
        "hi": ["hi", "hin", "hindi", "हिन्दी"],
        "ko": ["ko", "kor", "korean", "한국어", "coréen"],
        "pl": ["pl", "pol", "polish", "polski", "polonais"],
        "tr": ["tr", "tur", "turkish", "türkçe", "turc"],
        "sv": ["sv", "swe", "swedish", "svenska", "suédois"],
        "da": ["da", "dan", "danish", "dansk", "danois"],
        "no": ["no", "nor", "norwegian", "norsk", "norvégien"],
        "fi": ["fi", "fin", "finnish", "suomi", "finnois"],
        "cs": ["cs", "ces", "cze", "czech", "čeština", "tchèque"],
        "ro": ["ro", "ron", "rum", "romanian", "română", "roumain"],
        "hu": ["hu", "hun", "hungarian", "magyar", "hongrois"],
        "el": ["el", "ell", "gre", "greek", "ελληνικά", "grec"],
        "he": ["he", "heb", "hebrew", "עברית", "hébreu"],
        "th": ["th", "tha", "thai", "ไทย", "thaï"],
        "vi": ["vi", "vie", "vietnamese", "tiếng việt", "vietnamien"],
        "id": ["id", "ind", "indonesian", "bahasa indonesia", "indonésien"],
        "uk": ["uk", "ukr", "ukrainian", "українська", "ukrainien"],
    }

    # Reverse mapping for quick lookup
    _REVERSE_MAPPING: Optional[Dict[str, str]] = None

    @classmethod
    def _build_reverse_mapping(cls) -> None:
        """Build reverse mapping from variant to standard code."""
        if cls._REVERSE_MAPPING is None:
            cls._REVERSE_MAPPING = {}
            for standard_code, variants in cls.LANGUAGE_MAPPINGS.items():
                for variant in variants:
                    cls._REVERSE_MAPPING[variant.lower()] = standard_code.lower()

    @classmethod
    def normalize_language(cls, lang_value: Any) -> Optional[str]:
        """Normalize a language value to standard ISO two-letter code."""
        if not lang_value:
            return None

        cls._build_reverse_mapping()
        lang_lower = str(lang_value).strip().lower()
        if not lang_lower:
            return None

        # Fast path via mappings
        mapped = cls._REVERSE_MAPPING.get(lang_lower)
        if mapped:
            return mapped.upper()

        # Handle composite codes such as en-US / zh-CN
        if "-" in lang_lower or "_" in lang_lower:
            primary = lang_lower.replace("_", "-").split("-")[0]
            mapped = cls._REVERSE_MAPPING.get(primary)
            if mapped:
                return mapped.upper()

        # Already looks like an ISO code
        if len(lang_lower) in (2, 3) and lang_lower.isalpha():
            return lang_lower[:2].upper()

        return None

    @staticmethod
    def detect_dataset_languages(
        sample_texts: List[str],
        max_samples: int = 200,
        confidence_floor: float = 0.55,
    ) -> List[Set[str]]:
        """
        Detect likely languages present in a dataset sample.

        Returns a list of language code sets so callers can aggregate counts.
        """
        if not sample_texts:
            return []

        filtered: List[str] = []
        for text in sample_texts:
            if isinstance(text, str):
                stripped = text.strip()
                if stripped:
                    filtered.append(stripped)
            if len(filtered) >= max_samples:
                break

        if not filtered:
            return []

        detector = LanguageDetector()
        detections = detector.detect_batch(filtered, parallel=True)

        language_sets: List[Set[str]] = []
        for detection in detections:
            languages: Set[str] = set()
            primary_lang = detection.get("language")
            if primary_lang:
                normalized = LanguageNormalizer.normalize_language(primary_lang)
                if normalized:
                    languages.add(normalized.lower())

            for lang_code, score in detection.get("all_scores", []):
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    numeric_score = 0.0

                normalized_code = LanguageNormalizer.normalize_language(lang_code)
                if not normalized_code:
                    continue

                if numeric_score >= confidence_floor:
                    languages.add(normalized_code.lower())
                elif not languages and numeric_score > 0:
                    # Keep at least the best guess when nothing else qualified
                    languages.add(normalized_code.lower())

            if languages:
                language_sets.append(languages)

        return language_sets

    @staticmethod
    def detect_languages_in_column(df, column_name: str) -> Dict[str, int]:
        """Detect and count languages in a dataframe column."""
        if column_name not in df.columns:
            return {}

        lang_counts: Dict[str, int] = {}
        for value in df[column_name].dropna():
            normalized = LanguageNormalizer.normalize_language(value)
            if normalized:
                lang_counts[normalized] = lang_counts.get(normalized, 0) + 1

        return lang_counts

    @staticmethod
    def recommend_models(languages: Set[str], all_models: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Recommend training models based on detected languages - supports 11+ languages."""
        recommendations: List[Dict[str, Any]] = []

        # Language-specific model mappings (comprehensive list)
        language_specific_models = {
            "en": ("English Models", ["bert-base-uncased", "roberta-base", "deberta-v3-base"]),
            "fr": ("French Models", ["camembert-base", "flaubert-base", "distilcamembert"]),
            "es": ("Spanish Models", ["dccuchile/bert-base-spanish-wwm-cased", "PlanTL-GOB-ES/roberta-base-bne"]),
            "de": ("German Models", ["bert-base-german-cased", "deepset/gbert-base"]),
            "it": ("Italian Models", ["dbmdz/bert-base-italian-cased", "idb-ita/gilberto-uncased-from-camembert"]),
            "pt": ("Portuguese Models", ["neuralmind/bert-base-portuguese-cased", "portuguese-bert-base"]),
            "nl": ("Dutch Models", ["GroNLP/bert-base-dutch-cased", "wietsedv/bert-base-dutch-cased"]),
            "ru": ("Russian Models", ["DeepPavlov/rubert-base-cased", "sberbank-ai/ruBert-base"]),
            "zh": ("Chinese Models", ["bert-base-chinese", "hfl/chinese-bert-wwm-ext"]),
            "ja": ("Japanese Models", ["cl-tohoku/bert-base-japanese", "nlp-waseda/roberta-base-japanese"]),
            "ar": ("Arabic Models", ["asafaya/bert-base-arabic", "CAMeL-Lab/bert-base-arabic-camelbert-ca"]),
        }

        if not languages:
            return recommendations

        languages = {lang.lower() for lang in languages if lang}

        for lang in sorted(languages):
            mapping = language_specific_models.get(lang)
            if not mapping:
                continue

            title, model_ids = mapping
            available_models = all_models.get(lang.upper(), [])
            for model_id in model_ids:
                for available in available_models:
                    if available.get("model_id") == model_id:
                        recommendations.append(available)
                        break

        return recommendations

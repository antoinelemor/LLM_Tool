#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
token_analysis.py

MAIN OBJECTIVE:
---------------
Utility helpers for estimating character/token statistics across text batches.
The logic was originally implemented inside AdvancedCLI.analyze_text_lengths.
It now lives here so that annotator workflows and cost estimators can reuse the
exact same token counting behaviour (HF tokenizer fallback, whitespace fallback,
distribution metrics, etc.).

Dependencies:
-------------
- dataclasses
- typing
- logging
- numpy
- transformers (optional)

MAIN FEATURES:
--------------
1) Token analysis result container (TokenAnalysisResult)
2) Load first available tokenizer from Hugging Face
3) Analyse text sequences for character and token statistics
4) Compute descriptive statistics (min, max, mean, median, std, percentiles)
5) Classify documents by length distribution (short, medium, long, very_long)
6) Detect if long document model is required
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    from transformers import AutoTokenizer  # type: ignore
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TRANSFORMERS = False
    AutoTokenizer = None  # type: ignore


DEFAULT_TOKENIZER_CANDIDATES: Sequence[str] = (
    "bert-base-multilingual-cased",
    "bert-base-uncased",
    "distilbert-base-uncased",
)


@dataclass
class TokenAnalysisResult:
    """Container for text/token statistics."""

    texts_analyzed: int
    char_min: int
    char_max: int
    char_mean: float
    char_median: float
    char_std: float
    char_p25: float
    char_p75: float
    char_p95: float
    token_min: int
    token_max: int
    token_mean: float
    token_median: float
    token_std: float
    token_p25: float
    token_p75: float
    token_p95: float
    token_total: int
    requires_long_document_model: bool
    distribution: dict
    token_lengths: np.ndarray
    char_lengths: np.ndarray

    def to_dict(self) -> dict:
        """Return a plain dictionary representation (matches legacy structure)."""
        return {
            "char_min": self.char_min,
            "char_max": self.char_max,
            "char_mean": self.char_mean,
            "char_median": self.char_median,
            "char_std": self.char_std,
            "char_p25": self.char_p25,
            "char_p75": self.char_p75,
            "char_p95": self.char_p95,
            "token_min": self.token_min,
            "token_max": self.token_max,
            "token_mean": self.token_mean,
            "token_median": self.token_median,
            "token_std": self.token_std,
            "token_p25": self.token_p25,
            "token_p75": self.token_p75,
            "token_p95": self.token_p95,
            "token_total": self.token_total,
            "distribution": self.distribution,
            "requires_long_document_model": self.requires_long_document_model,
            "texts_analyzed": self.texts_analyzed,
        }


def _load_first_available_tokenizer(
    candidates: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> Optional[AutoTokenizer]:
    """Try loading a tokenizer locally from the provided candidate list."""
    if not HAS_TRANSFORMERS:
        return None

    for model_name in candidates:
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            if logger:
                logger.debug("Tokenizer load failed for %s: %s", model_name, exc)
    return None


def analyse_text_tokens(
    texts: Iterable[str],
    *,
    tokenizer_candidates: Sequence[str] = DEFAULT_TOKENIZER_CANDIDATES,
    logger: Optional[logging.Logger] = None,
) -> TokenAnalysisResult:
    """
    Analyse a sequence of texts and compute descriptive statistics on characters/tokens.

    Parameters
    ----------
    texts:
        Iterable of strings to analyse.
    tokenizer_candidates:
        Ordered list of Hugging Face tokenizer model names to try loading.
    logger:
        Optional logger for debug information.
    """
    texts_list: List[str] = [text if isinstance(text, str) else "" for text in texts]
    if not texts_list:
        empty_array = np.array([0])
        return TokenAnalysisResult(
            texts_analyzed=0,
            char_min=0,
            char_max=0,
            char_mean=0.0,
            char_median=0.0,
            char_std=0.0,
            char_p25=0.0,
            char_p75=0.0,
            char_p95=0.0,
            token_min=0,
            token_max=0,
            token_mean=0.0,
            token_median=0.0,
            token_std=0.0,
            token_p25=0.0,
            token_p75=0.0,
            token_p95=0.0,
            token_total=0,
            requires_long_document_model=False,
            distribution={
                "short": {"count": 0, "percentage": 0.0},
                "medium": {"count": 0, "percentage": 0.0},
                "long": {"count": 0, "percentage": 0.0},
                "very_long": {"count": 0, "percentage": 0.0},
            },
            token_lengths=empty_array,
            char_lengths=empty_array,
        )

    tokenizer = _load_first_available_tokenizer(tokenizer_candidates, logger)

    char_lengths: List[int] = []
    token_lengths: List[int] = []
    for text in texts_list:
        char_lengths.append(len(text))
        if tokenizer is not None:
            try:
                encoded = tokenizer.encode(text, truncation=False, add_special_tokens=True)  # type: ignore
                token_lengths.append(len(encoded))
            except Exception as exc:  # pragma: no cover - tokenizer failure
                if logger:
                    logger.debug("Tokenizer encode failed; falling back to whitespace tokens: %s", exc)
                token_lengths.append(len(text.split()))
        else:
            token_lengths.append(len(text.split()))

    char_array = np.array(char_lengths, dtype=np.int64)
    token_array = np.array(token_lengths, dtype=np.int64)

    denom = max(len(token_array), 1)
    short_docs = int(np.sum(token_array < 128))
    medium_docs = int(np.sum((token_array >= 128) & (token_array < 512)))
    long_docs = int(np.sum((token_array >= 512) & (token_array < 1024)))
    very_long_docs = int(np.sum(token_array >= 1024))

    requires_long_doc = (long_docs + very_long_docs) / denom > 0.20

    return TokenAnalysisResult(
        texts_analyzed=len(token_array),
        char_min=int(char_array.min()),
        char_max=int(char_array.max()),
        char_mean=float(char_array.mean()),
        char_median=float(np.median(char_array)),
        char_std=float(char_array.std()),
        char_p25=float(np.percentile(char_array, 25)),
        char_p75=float(np.percentile(char_array, 75)),
        char_p95=float(np.percentile(char_array, 95)),
        token_min=int(token_array.min()),
        token_max=int(token_array.max()),
        token_mean=float(token_array.mean()),
        token_median=float(np.median(token_array)),
        token_std=float(token_array.std()),
        token_p25=float(np.percentile(token_array, 25)),
        token_p75=float(np.percentile(token_array, 75)),
        token_p95=float(np.percentile(token_array, 95)),
        token_total=int(token_array.sum()),
        requires_long_document_model=requires_long_doc,
        distribution={
            "short": {"count": short_docs, "percentage": float(short_docs / denom * 100)},
            "medium": {"count": medium_docs, "percentage": float(medium_docs / denom * 100)},
            "long": {"count": long_docs, "percentage": float(long_docs / denom * 100)},
            "very_long": {"count": very_long_docs, "percentage": float(very_long_docs / denom * 100)},
        },
        token_lengths=token_array,
        char_lengths=char_array,
    )


# Discrete max_length values offered when sizing a model. Sticking to powers of
# two (plus 384, a common BERT-friendly checkpoint length) keeps GPU tensor
# shapes well-aligned for attention kernels on both CUDA and MPS.
_RECOMMENDED_BUCKETS: Sequence[int] = (64, 96, 128, 192, 256, 384, 512)
_HARD_CAP: int = 4096


def _round_to_recommended_bucket(target: int, *, model_max: Optional[int]) -> int:
    """Pick the smallest bucket >= target, capped by model_max.

    The clamp matters: a recommendation of 8000 tokens on a model whose
    position embeddings only go to 512 would silently break the model.

    When ``model_max`` is None (the tokenizer does not expose a usable
    ``model_max_length`` — many BERT-style tokenizers report a sentinel like
    1e30), we conservatively cap at 512 since that is the standard ceiling
    for non-long-document BERT/RoBERTa/CamemBERT checkpoints. Callers that
    target long-document models must pass an explicit ``model_max``.
    """
    if isinstance(model_max, int) and model_max > 0:
        cap = min(model_max, _HARD_CAP)
    else:
        cap = 512
    target = max(int(target), _RECOMMENDED_BUCKETS[0])
    target = min(target, cap)
    for bucket in _RECOMMENDED_BUCKETS:
        if bucket >= target and bucket <= cap:
            return bucket
    # No standard bucket fits under the cap: hand back the cap itself.
    return cap


def analyze_tokens_with_model_tokenizer(
    texts: Sequence[str],
    model_name: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """Tokenize every text with the exact tokenizer of ``model_name`` and
    recommend a safe ``max_length``.

    Behaviour by design
    -------------------
    - Returns ``None`` (no exception, no log spam) when the tokenizer cannot be
      loaded — typical reasons are missing ``transformers``, no network and no
      cache, or an invalid model id. Callers must treat ``None`` as "don't
      offer the optimisation, keep the current default" and never as a silent
      fallback to 512.
    - When successful, tokenises the full input (no sampling) using the fast
      tokenizer's batch encode for speed. The recommendation is the smallest
      power-of-two bucket >= the observed token-length maximum, which
      guarantees zero truncation by construction.
    - Caps recommendations at the tokenizer's announced ``model_max_length``
      so we never advertise a length the model cannot actually consume.

    Returns
    -------
    dict with keys: ``model_name``, ``texts_analyzed``, ``token_max``,
    ``token_p95``, ``token_p99``, ``token_mean``, ``recommended_max_length``,
    ``current_default`` (always 512 for callers that compare against the
    legacy default), ``n_above_recommended`` (always 0 — sanity check),
    ``n_above_512``, ``distribution_buckets``, ``model_max_length``.
    """
    if not HAS_TRANSFORMERS:
        if logger:
            logger.info(
                "Skipping tokenizer-aware analysis: transformers not installed."
            )
        return None
    if not model_name or not isinstance(model_name, str):
        if logger:
            logger.info("Skipping tokenizer-aware analysis: empty model_name.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # type: ignore
    except Exception as exc:
        if logger:
            logger.info(
                "Tokenizer-aware analysis disabled: could not load tokenizer for "
                "%s (%s). Falling back to default max_length.",
                model_name,
                exc,
            )
        return None

    # Filter to non-empty strings; mirror the validation done in BertBase._prepare_inputs.
    clean_texts: List[str] = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        if logger:
            logger.info("Skipping tokenizer-aware analysis: no valid texts.")
        return None

    try:
        # Fast tokenizers handle large batches efficiently. We disable padding
        # and truncation so we observe the *real* token length distribution.
        encoded = tokenizer(
            clean_texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=np.int64)
    except Exception as exc:
        if logger:
            logger.info(
                "Tokenizer-aware analysis aborted during batch encode: %s", exc
            )
        return None

    if lengths.size == 0:
        return None

    # Determine the tokenizer's model_max_length when it is a sane finite value.
    raw_model_max = getattr(tokenizer, "model_max_length", None)
    model_max: Optional[int] = None
    if isinstance(raw_model_max, int) and 0 < raw_model_max < 10_000_000:
        model_max = raw_model_max

    token_max = int(lengths.max())
    recommended = _round_to_recommended_bucket(token_max, model_max=model_max)
    n_above_recommended = int(np.sum(lengths > recommended))
    n_above_512 = int(np.sum(lengths > 512))

    # Coarse distribution for display in the CLI panel.
    buckets = {
        "≤128": int(np.sum(lengths <= 128)),
        "129-256": int(np.sum((lengths > 128) & (lengths <= 256))),
        "257-512": int(np.sum((lengths > 256) & (lengths <= 512))),
        ">512": int(np.sum(lengths > 512)),
    }

    return {
        "model_name": model_name,
        "texts_analyzed": int(lengths.size),
        "token_min": int(lengths.min()),
        "token_max": token_max,
        "token_mean": float(lengths.mean()),
        "token_p95": float(np.percentile(lengths, 95)),
        "token_p99": float(np.percentile(lengths, 99)),
        "recommended_max_length": recommended,
        "current_default": 512,
        "n_above_recommended": n_above_recommended,
        "n_above_512": n_above_512,
        "distribution_buckets": buckets,
        "model_max_length": model_max,
    }


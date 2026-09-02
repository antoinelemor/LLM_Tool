#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
tokenizer_check.py

MAIN OBJECTIVE:
---------------
Refuse to fine-tune an encoder on inputs that lack its classification
tokens, and record the input format a checkpoint was trained on.

Sequence-classification heads pool the first position of the encoder
output, the one pretraining reserved for [CLS]. A tokenizer that omits
[CLS]/[SEP] still lets training converge: the head learns to read the
first word piece instead, in-process verification keeps passing because
it reuses the same tokenizer, and the exported checkpoint degrades to a
near-constant output the moment another environment feeds it
[CLS]-prefixed input. transformers 5.1.x ships exactly such a DeBERTa-v2
tokenizer (no post-processor).

MAIN FEATURES:
--------------
1) check_special_tokens(): encodes a probe string and asserts that every
   classification token the tokenizer declares appears in the output
2) Returns a small report the trainers write into
   training_metadata.json, so a checkpoint carries proof of the input
   format it saw

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

PROBE_TEXT = "tokenizer self-check"


class TokenizerSpecialTokensError(RuntimeError):
    """The tokenizer declares classification tokens but does not add them."""


def _transformers_version() -> Optional[str]:
    try:
        return version("transformers")
    except PackageNotFoundError:
        return None


def check_special_tokens(tokenizer: Any, model_name: str = "") -> Dict[str, Any]:
    """Verify that ``tokenizer`` adds the special tokens it declares.

    Parameters
    ----------
    tokenizer : Any
        Anything exposing ``cls_token_id``, ``sep_token_id`` and
        ``encode(text, add_special_tokens=True)``.
    model_name : str
        Used in the error message only.

    Returns
    -------
    dict
        ``tokenizer_class``, ``transformers_version``, ``cls_token_id``,
        ``sep_token_id``, ``probe_input_ids`` and ``special_tokens_added``.
        A tokenizer that declares no classification token (decoder-only
        models) passes with ``special_tokens_added`` set to None.

    Raises
    ------
    TokenizerSpecialTokensError
        If a declared token is absent from the encoded probe.
    """
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    ids = [int(i) for i in tokenizer.encode(PROBE_TEXT, add_special_tokens=True)]
    declared = {k: v for k, v in (("cls", cls_id), ("sep", sep_id)) if v is not None}
    report: Dict[str, Any] = {
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": _transformers_version(),
        "cls_token_id": cls_id,
        "sep_token_id": sep_id,
        "probe_input_ids": ids,
        "special_tokens_added": None if not declared else all(v in ids for v in declared.values()),
    }
    missing = [k for k, v in declared.items() if v not in ids]
    if missing:
        raise TokenizerSpecialTokensError(
            f"{report['tokenizer_class']} for {model_name or 'the model'} did not add "
            f"{', '.join(f'{k}={declared[k]}' for k in missing)} when encoding "
            f"{PROBE_TEXT!r}: got {ids}. A head trained on such inputs is unusable "
            f"outside this exact environment (transformers "
            f"{report['transformers_version']}). DeBERTa-v2 needs transformers>=5.8."
        )
    return report

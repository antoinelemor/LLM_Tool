"""A tokenizer that drops its classification tokens must stop training.

The failure is silent otherwise: the model converges, the reload check
agrees with itself, and the checkpoint collapses in any environment
whose tokenizer emits [CLS] ... [SEP].
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_tool.utils.tokenizer_check import (
    TokenizerSpecialTokensError,
    check_special_tokens,
)


class _Tokenizer:
    def __init__(self, cls_id, sep_id, wrap):
        self.cls_token_id, self.sep_token_id, self._wrap = cls_id, sep_id, wrap

    def encode(self, text, add_special_tokens=True):
        body = [10 + i for i in range(len(text.split()))]
        if self._wrap and add_special_tokens:
            return [self.cls_token_id] + body + [self.sep_token_id]
        return body


def test_wrapped_input_passes_and_is_reported():
    r = check_special_tokens(_Tokenizer(1, 2, wrap=True), "m")
    assert r["special_tokens_added"] is True
    assert r["probe_input_ids"][0] == 1 and r["probe_input_ids"][-1] == 2
    assert r["tokenizer_class"] == "_Tokenizer"


def test_bare_word_pieces_raise():
    with pytest.raises(TokenizerSpecialTokensError, match="cls=1, sep=2"):
        check_special_tokens(_Tokenizer(1, 2, wrap=False), "m")


def test_one_missing_token_is_still_refused():
    class _SepOnly(_Tokenizer):
        def encode(self, text, add_special_tokens=True):
            return [10, 11, self.sep_token_id]

    with pytest.raises(TokenizerSpecialTokensError, match="cls=1"):
        check_special_tokens(_SepOnly(1, 2, wrap=False), "m")


def test_decoder_style_tokenizer_without_cls_passes():
    r = check_special_tokens(_Tokenizer(None, None, wrap=False), "m")
    assert r["special_tokens_added"] is None


def test_installed_deberta_v2_tokenizer_adds_its_tokens():
    """The installed transformers must wrap DeBERTa-v2 input; 5.1.x does not."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "microsoft/mdeberta-v3-base", local_files_only=True
        )
    except Exception:  # noqa: BLE001
        pytest.skip("microsoft/mdeberta-v3-base is not in the local cache")
    r = check_special_tokens(tok, "microsoft/mdeberta-v3-base")
    assert r["special_tokens_added"] is True
    assert r["probe_input_ids"][0] == tok.cls_token_id
    assert r["probe_input_ids"][-1] == tok.sep_token_id

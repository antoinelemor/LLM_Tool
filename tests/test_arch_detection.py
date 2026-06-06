"""Offline regression tests for architecture-based DeBERTa detection.

BertBase._is_deberta_arch must:
  - recognise DeBERTa-family models by NAME (fast path, no network), and
  - recognise DeBERTa-v2 architectures whose NAME lacks 'deberta'
    (CamemBERTa / CamemBERTav2) via the HF config model_type, and
  - NOT misclassify non-DeBERTa models (bert, xlm-roberta, camembert-v1...).

This guards the float32-on-MPS forcing and the DeBERTa batch-size guard, which
would otherwise silently miss camembertav2 (a deberta-v2 model).

Run: pytest tests/test_arch_detection.py -v
"""

import transformers
from llm_tool.trainers.bert_base import BertBase


def _bb(name):
    bb = BertBase.__new__(BertBase)  # no heavy __init__
    bb.model_name = name
    return bb


def test_name_match_shortcircuits_without_network(monkeypatch):
    # If the name contains 'deberta', detection must NOT need the HF config.
    def _boom(*a, **k):
        raise AssertionError("AutoConfig must not be called when name matches")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _boom)
    assert _bb("microsoft/deberta-v3-base")._is_deberta_arch() is True
    assert _bb("microsoft/mdeberta-v3-base")._is_deberta_arch() is True


def test_camembertav2_detected_via_config(monkeypatch):
    class FakeCfg:
        model_type = "deberta-v2"
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: FakeCfg())
    # name lacks 'deberta' -> must fall back to config model_type
    assert _bb("almanach/camembertav2-base")._is_deberta_arch() is True


def test_non_deberta_via_config(monkeypatch):
    class FakeCfg:
        model_type = "xlm-roberta"
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: FakeCfg())
    assert _bb("xlm-roberta-base")._is_deberta_arch() is False


def test_config_failure_defaults_false(monkeypatch):
    def _raise(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _raise)
    # Unknown name + no config access -> safe default False (no spurious float32)
    assert _bb("some/unknown-model")._is_deberta_arch() is False


def test_result_is_cached(monkeypatch):
    calls = {"n": 0}
    class FakeCfg:
        model_type = "deberta-v2"
    def _count(*a, **k):
        calls["n"] += 1
        return FakeCfg()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _count)
    bb = _bb("almanach/camembertav2-base")
    assert bb._is_deberta_arch() is True
    assert bb._is_deberta_arch() is True
    assert calls["n"] == 1  # cached after first resolution

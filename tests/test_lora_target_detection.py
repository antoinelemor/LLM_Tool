"""Offline unit tests for BertBase._detect_lora_target_modules.

Builds tiny synthetic module trees that mimic the attention-projection naming of
every supported architecture family, so DoRA target-module auto-detection is
regression-protected without any network/model download.

Run with:  pytest tests/test_lora_target_detection.py -v
"""

import torch
import torch.nn as nn
import pytest

from llm_tool.trainers.bert_base import BertBase


def _attn(names):
    """A fake attention block exposing nn.Linear submodules with the given leaf names."""
    m = nn.Module()
    for n in names:
        setattr(m, n, nn.Linear(8, 8))
    return m


def _model(attn_names, extra=("dense", "classifier", "pooler")):
    """A fake encoder: one attention block + some non-attention Linear leaves."""
    root = nn.Module()
    root.attention = _attn(attn_names)
    for e in extra:
        setattr(root, e, nn.Linear(8, 8))
    return root


# (family, attention leaf names, expected detected Q/V targets)
FAMILIES = [
    ("bert/roberta/xlm-r/camembert/electra/albert/bigbird",
     ["query", "key", "value"], ["query", "value"]),
    ("deberta-v2 / camembertav2 / deberta-v3 / mdeberta",
     ["query_proj", "key_proj", "value_proj"], ["query_proj", "value_proj"]),
    ("distilbert / flaubert / xlm",
     ["q_lin", "k_lin", "v_lin", "out_lin"], ["q_lin", "v_lin"]),
    ("bart / barthez / mbart",
     ["q_proj", "k_proj", "v_proj", "out_proj"], ["q_proj", "v_proj"]),
    ("t5 / long-t5",
     ["q", "k", "v", "o"], ["q", "v"]),
    ("longformer (local + global)",
     ["query", "key", "value", "query_global", "key_global", "value_global"],
     ["query", "query_global", "value", "value_global"]),
]


@pytest.mark.parametrize("family,attn,expected", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_detect_targets_per_family(family, attn, expected):
    model = _model(attn)
    got = BertBase._detect_lora_target_modules(model)
    assert got == sorted(expected), f"{family}: expected {sorted(expected)}, got {got}"


def test_no_attention_falls_back_to_all_linear():
    # A model with only non-attention linear layers -> universal fallback.
    root = nn.Module()
    root.dense = nn.Linear(8, 8)
    root.classifier = nn.Linear(8, 2)
    assert BertBase._detect_lora_target_modules(root) == "all-linear"


def test_detected_names_are_real_linear_leaves():
    model = _model(["query_proj", "key_proj", "value_proj"])
    leaves = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    for t in BertBase._detect_lora_target_modules(model):
        assert t in leaves


def test_key_and_dense_are_never_targeted():
    # Post-convergence DoRA adapts only Q/V — key/output/dense must be excluded.
    model = _model(["query", "key", "value"], extra=("dense", "output", "classifier"))
    got = BertBase._detect_lora_target_modules(model)
    assert "key" not in got and "dense" not in got and "output" not in got

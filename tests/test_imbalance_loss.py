"""Unit tests for llm_tool.trainers.imbalance_loss.

Run with:  pytest tests/test_imbalance_loss.py -v
"""

import numpy as np
import pytest
import torch

from llm_tool.trainers.imbalance_loss import (
    FocalLoss,
    SigmoidFocalLoss,
    ImbalanceConfig,
    build_train_criterion,
    compute_pos_ratios,
    recommend_strategy,
    normalize_strategy,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# FocalLoss
# --------------------------------------------------------------------------
def test_focal_gamma0_uniform_equals_cross_entropy():
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    fl = FocalLoss(gamma=0.0, alpha=None)
    ce = torch.nn.CrossEntropyLoss()
    assert torch.allclose(fl(logits, targets), ce(logits, targets), atol=1e-6)


def test_focal_binary_gamma0_equals_ce():
    logits = torch.randn(16, 2)
    targets = torch.randint(0, 2, (16,))
    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss()
    assert torch.allclose(fl(logits, targets), ce(logits, targets), atol=1e-6)


def test_focal_downweights_easy_examples():
    # One confidently-correct (easy) example: focal loss << cross-entropy.
    logits = torch.tensor([[5.0, -5.0]])  # very confident class 0
    targets = torch.tensor([0])
    ce = torch.nn.CrossEntropyLoss()(logits, targets)
    fl = FocalLoss(gamma=2.0)(logits, targets)
    assert fl < ce


def test_focal_alpha_weights_classes():
    logits = torch.zeros(2, 2)  # p = 0.5 each
    targets = torch.tensor([0, 1])
    alpha = torch.tensor([1.0, 5.0])
    fl = FocalLoss(gamma=0.0, alpha=alpha, reduction="none")(logits, targets)
    # class-1 sample weighted 5x relative to class-0 sample
    assert torch.allclose(fl[1] / fl[0], torch.tensor(5.0), atol=1e-5)


def test_focal_reduction_modes():
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    none = FocalLoss(gamma=1.5, reduction="none")(logits, targets)
    assert none.shape == (4,)
    assert torch.allclose(FocalLoss(gamma=1.5, reduction="mean")(logits, targets), none.mean(), atol=1e-6)
    assert torch.allclose(FocalLoss(gamma=1.5, reduction="sum")(logits, targets), none.sum(), atol=1e-6)


# --------------------------------------------------------------------------
# SigmoidFocalLoss
# --------------------------------------------------------------------------
def test_sigmoid_focal_gamma0_equals_bce():
    logits = torch.randn(8, 5)
    targets = (torch.rand(8, 5) > 0.5).float()
    sfl = SigmoidFocalLoss(gamma=0.0)
    bce = torch.nn.BCEWithLogitsLoss()
    assert torch.allclose(sfl(logits, targets), bce(logits, targets), atol=1e-6)


def test_sigmoid_focal_gamma0_posweight_equals_bce_posweight():
    logits = torch.randn(8, 4)
    targets = (torch.rand(8, 4) > 0.5).float()
    pw = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sfl = SigmoidFocalLoss(gamma=0.0, pos_weight=pw)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    assert torch.allclose(sfl(logits, targets), bce(logits, targets), atol=1e-6)


def test_sigmoid_focal_downweights_easy():
    logits = torch.tensor([[8.0]])
    targets = torch.tensor([[1.0]])
    bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
    sfl = SigmoidFocalLoss(gamma=2.0)(logits, targets)
    assert sfl < bce


# --------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------
def test_compute_pos_ratios_single_label():
    labels = np.array([0, 0, 0, 1])  # 75% class0, 25% class1
    ratios = compute_pos_ratios(labels, num_labels=2, multi_label=False)
    assert np.allclose(ratios, [0.75, 0.25])


def test_compute_pos_ratios_multi_label():
    labels = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]])
    ratios = compute_pos_ratios(labels, num_labels=3, multi_label=True)
    assert np.allclose(ratios, [0.75, 0.25, 1e-6], atol=1e-3)


def test_recommend_strategy_thresholds():
    assert recommend_strategy(np.array([0.5, 0.5])) == "none"
    assert recommend_strategy(np.array([0.5, 0.2])) == "weighted"     # rarest 20%
    assert recommend_strategy(np.array([0.5, 0.05])) == "focal"       # rarest 5%
    assert recommend_strategy(np.array([])) == "none"


def test_normalize_strategy():
    assert normalize_strategy(None) is None
    assert normalize_strategy("off") is None
    assert normalize_strategy("FOCAL") == "focal"
    assert normalize_strategy("none") == "none"
    assert normalize_strategy("bogus") == "auto"  # unknown -> auto


# --------------------------------------------------------------------------
# Factory dispatch
# --------------------------------------------------------------------------
def _binary_labels(pos_frac=0.1, n=200):
    pos = int(n * pos_frac)
    return np.array([1] * pos + [0] * (n - pos))


def test_factory_disabled_returns_none():
    cfg = ImbalanceConfig(strategy=None)
    crit, pw, samp, used = build_train_criterion(None, False, 2, _binary_labels(), cfg)
    assert crit is None and used == "none" and samp is False


def test_factory_explicit_none_returns_plain_loss():
    cfg = ImbalanceConfig(strategy="none")
    crit, pw, samp, used = build_train_criterion(None, False, 2, _binary_labels(), cfg)
    assert isinstance(crit, torch.nn.CrossEntropyLoss)
    crit_ml, _, _, _ = build_train_criterion(None, True, 3,
                                             np.zeros((10, 3)), ImbalanceConfig(strategy="none"))
    assert isinstance(crit_ml, torch.nn.BCEWithLogitsLoss)


def test_factory_weighted_single_and_multi():
    crit, pw, samp, used = build_train_criterion(
        "weighted", False, 2, _binary_labels(0.1), ImbalanceConfig(strategy="weighted"))
    assert isinstance(crit, torch.nn.CrossEntropyLoss)
    assert crit.weight is not None and crit.weight.numel() == 2
    assert used == "weighted" and samp is True

    labels = (np.random.RandomState(0).rand(100, 3) > 0.8).astype(float)
    crit2, pw2, _, _ = build_train_criterion(
        "weighted", True, 3, labels, ImbalanceConfig(strategy="weighted"))
    assert isinstance(crit2, torch.nn.BCEWithLogitsLoss)
    assert pw2 is not None and pw2.numel() == 3


def test_factory_focal_single_and_multi():
    crit, _, _, used = build_train_criterion(
        "focal", False, 2, _binary_labels(0.05), ImbalanceConfig(strategy="focal", focal_gamma=2.0))
    assert isinstance(crit, FocalLoss) and used == "focal"

    labels = (np.random.RandomState(1).rand(100, 4) > 0.9).astype(float)
    crit2, _, _, used2 = build_train_criterion(
        "focal", True, 4, labels, ImbalanceConfig(strategy="focal"))
    assert isinstance(crit2, SigmoidFocalLoss) and used2 == "focal"


def test_factory_asymmetric_multi_uses_asl_single_falls_back():
    labels = (np.random.RandomState(2).rand(120, 3) > 0.95).astype(float)
    crit, _, _, used = build_train_criterion(
        "asymmetric", True, 3, labels, ImbalanceConfig(strategy="asymmetric"))
    # AdaptiveAsymmetricLoss is imported lazily from bert_base
    assert crit is not None and used == "asymmetric"
    # single-label asymmetric -> focal fallback
    crit2, _, _, used2 = build_train_criterion(
        "asymmetric", False, 2, _binary_labels(0.05), ImbalanceConfig(strategy="asymmetric"))
    assert isinstance(crit2, FocalLoss) and "focal" in used2


def test_factory_auto_resolves_from_distribution():
    # severe imbalance -> focal
    crit, _, _, used = build_train_criterion(
        "auto", False, 2, _binary_labels(0.03), ImbalanceConfig(strategy="auto"))
    assert used == "focal"
    # balanced -> none
    crit_b, _, samp_b, used_b = build_train_criterion(
        "auto", False, 2, _binary_labels(0.5), ImbalanceConfig(strategy="auto"))
    assert used_b == "none" and samp_b is False


def test_factory_manual_weights_override():
    cfg = ImbalanceConfig(strategy="weighted", weight_source="manual",
                          manual_class_weights=[1.0, 7.0])
    crit, _, _, _ = build_train_criterion("weighted", False, 2, _binary_labels(0.1), cfg)
    assert torch.allclose(crit.weight, torch.tensor([1.0, 7.0]))


def test_factory_handles_single_class_edge():
    # all-negative column should not crash (clipped pos ratio)
    labels = np.zeros((50,), dtype=int)
    crit, _, _, used = build_train_criterion(
        "auto", False, 2, labels, ImbalanceConfig(strategy="auto"))
    assert crit is not None


def test_criteria_are_differentiable():
    for crit in (FocalLoss(gamma=2.0), FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 3.0]))):
        logits = torch.randn(8, 2, requires_grad=True)
        targets = torch.randint(0, 2, (8,))
        loss = crit(logits, targets)
        loss.backward()
        assert logits.grad is not None
    logits = torch.randn(8, 3, requires_grad=True)
    targets = (torch.rand(8, 3) > 0.5).float()
    loss = SigmoidFocalLoss(gamma=2.0, pos_weight=torch.tensor([1.0, 2.0, 3.0]))(logits, targets)
    loss.backward()
    assert logits.grad is not None

"""
Class-imbalance handling for the NORMAL training phase.

This module centralises every loss/weighting strategy used to fight class
imbalance during *normal* (non-reinforced) training, for ALL training types:
multi-class, multi-label, binary and one-vs-all.

Design goals
------------
- **Single source of truth**: ``build_train_criterion`` is the only place that
  decides which loss is used. ``bert_base`` calls it once and applies the
  returned criterion uniformly across AMP / MPS / CPU code paths.
- **No-op preserving**: callers that do not opt in (``strategy=None``) never
  reach this module, so legacy behaviour is untouched.
- **Reuse**: for the ``asymmetric`` strategy we reuse the battle-tested
  ``AsymmetricLoss`` / ``AdaptiveAsymmetricLoss`` already living in
  ``bert_base`` (imported lazily to avoid a circular import). Multi-label
  ``pos_weight`` reuses ``compute_multi_label_class_weights``.

Strategies (the ``strategy`` field of :class:`ImbalanceConfig`)
---------------------------------------------------------------
- ``none``        : plain CrossEntropy / BCE (no weighting).
- ``weighted``    : class-weighted CrossEntropy (single-label) or
                    ``pos_weight`` BCE (multi-label).
- ``focal``       : Focal Loss (Lin et al., 2017). Softmax-focal for
                    single-label, sigmoid-focal for multi-label. Optionally
                    combined with class weights / ``pos_weight``.
- ``asymmetric``  : Asymmetric Loss (Ridnik et al., 2021) for multi-label;
                    falls back to ``focal`` for single-label (ASL is
                    multi-label only).
- ``auto``        : pick a strategy from the observed label distribution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Valid strategy identifiers (``auto`` is resolved to one of the others).
VALID_STRATEGIES = ("none", "weighted", "focal", "asymmetric", "auto")

# Auto-detection thresholds on the *rarest* positive ratio across classes.
SEVERE_IMBALANCE_RATIO = 0.10   # rarest class < 10% positives -> focal
MODERATE_IMBALANCE_RATIO = 0.30  # rarest class < 30% positives -> weighted


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------
class FocalLoss(torch.nn.Module):
    """
    Multi-class / binary Focal Loss (softmax variant, Lin et al., 2017).

    ``FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)``

    Parameters
    ----------
    gamma : float
        Focusing parameter. ``gamma=0`` reduces to (weighted) cross-entropy.
    alpha : torch.Tensor | None
        Per-class weights of shape ``(num_classes,)``. ``None`` means uniform.
        With ``gamma=0`` and ``alpha=None`` the loss is *exactly*
        ``nn.CrossEntropyLoss(reduction=reduction)``.
    reduction : str
        ``'mean'`` (default), ``'sum'`` or ``'none'``.

    Notes
    -----
    Expects raw logits of shape ``(batch, num_classes)`` and integer targets
    of shape ``(batch,)``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        # Per-sample cross-entropy: -log p_t
        ce = F.nll_loss(log_probs, targets, weight=None, reduction="none")
        pt = torch.exp(-ce)  # probability of the true class
        focal = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal = focal * alpha.gather(0, targets)

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class SigmoidFocalLoss(torch.nn.Module):
    """
    Multi-label Focal Loss (sigmoid / BCE variant, Lin et al., 2017).

    ``FL = (1 - p_t)**gamma * BCE(logits, targets)`` where
    ``p_t = p`` for positives and ``1 - p`` for negatives.

    With ``gamma=0`` and ``pos_weight=None`` the loss is *exactly*
    ``nn.BCEWithLogitsLoss(reduction=reduction)``.

    Parameters
    ----------
    gamma : float
        Focusing parameter.
    pos_weight : torch.Tensor | None
        Per-label positive weight of shape ``(num_labels,)`` (same semantics
        as ``BCEWithLogitsLoss``).
    reduction : str
        ``'mean'`` (over all elements, like BCE), ``'sum'`` or ``'none'``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None, persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        if self.gamma > 0:
            p = torch.sigmoid(logits)
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            ce = ce * (1.0 - p_t).pow(self.gamma)

        if self.reduction == "mean":
            return ce.mean()
        if self.reduction == "sum":
            return ce.sum()
        return ce


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ImbalanceConfig:
    """User-facing configuration for normal-phase imbalance handling."""

    strategy: Optional[str] = None        # None -> feature disabled (legacy path)
    focal_gamma: float = 2.0              # gamma for focal / used for asl auto
    weight_source: str = "auto"           # 'auto' (from data) | 'manual'
    manual_class_weights: Optional[List[float]] = None  # used when weight_source='manual'
    use_weighted_sampler: bool = True     # enable WeightedRandomSampler
    # Asymmetric-loss hyper-parameters (multi-label only)
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    max_weight: float = 100.0             # cap on derived weights for stability

    def normalized_strategy(self) -> Optional[str]:
        return normalize_strategy(self.strategy)


def normalize_strategy(strategy: Optional[str]) -> Optional[str]:
    """Canonicalise a strategy string. ``None``/off/disabled -> None (disabled)."""
    if strategy is None:
        return None
    s = str(strategy).strip().lower()
    if s in ("", "none_disabled", "off", "disabled"):
        # NOTE: 'none' is a *valid* explicit strategy (plain CE/BCE); only the
        # blank/off forms disable the feature.
        return None
    if s not in VALID_STRATEGIES:
        logger.warning("Unknown imbalance strategy '%s' -> falling back to 'auto'", strategy)
        return "auto"
    return s


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------
def compute_pos_ratios(labels: np.ndarray, num_labels: int, multi_label: bool) -> np.ndarray:
    """Positive fraction per class/label.

    - multi-label: ``labels`` is multi-hot ``(N, num_labels)`` -> column means.
    - single-label: ``labels`` is class indices ``(N,)`` -> per-class frequency.
    """
    labels = np.asarray(labels)
    if multi_label:
        if labels.ndim == 1:  # defensive: indices given for a multi-label head
            oh = np.zeros((labels.shape[0], num_labels), dtype=np.float32)
            oh[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
            labels = oh
        ratios = labels.mean(axis=0)
    else:
        flat = labels.reshape(-1).astype(int)
        counts = np.bincount(flat, minlength=num_labels).astype(np.float64)
        total = max(int(flat.shape[0]), 1)
        ratios = counts / total
    return np.clip(ratios, 1e-6, 1.0)


def recommend_strategy(pos_ratios: np.ndarray) -> str:
    """Pick a strategy from the rarest positive ratio across classes."""
    pos_ratios = np.asarray(pos_ratios, dtype=np.float64)
    if pos_ratios.size == 0:
        return "none"
    rarest = float(pos_ratios.min())
    if rarest < SEVERE_IMBALANCE_RATIO:
        return "focal"
    if rarest < MODERATE_IMBALANCE_RATIO:
        return "weighted"
    return "none"


def _single_label_class_weights(
    pos_ratios: np.ndarray, max_weight: float
) -> torch.Tensor:
    """Inverse-frequency class weights for single-label CE/focal, mean-normalised."""
    pos_ratios = np.asarray(pos_ratios, dtype=np.float64)
    inv = 1.0 / np.clip(pos_ratios, 1e-6, 1.0)
    inv = inv / inv.mean()                       # normalise so a balanced set -> ~1
    inv = np.clip(inv, 1.0 / max_weight, max_weight)
    return torch.tensor(inv, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_train_criterion(
    strategy: Optional[str],
    multi_label: bool,
    num_labels: int,
    train_labels_np: Optional[np.ndarray],
    cfg: ImbalanceConfig,
    suppress_log: bool = False,
) -> Tuple[Optional[Callable], Optional[torch.Tensor], bool, str]:
    """Build the unified normal-phase training criterion.

    Returns
    -------
    (criterion, pos_weight, use_sampler, resolved_strategy)
        - ``criterion``: callable ``(logits, targets) -> scalar loss`` or
          ``None`` when the feature is disabled (caller keeps legacy path).
        - ``pos_weight``: tensor exposed for logging/back-compat (may be None).
        - ``use_sampler``: whether a WeightedRandomSampler should be activated.
        - ``resolved_strategy``: the concrete strategy actually used (``auto``
          resolved, or ``none``).
    """
    # ``strategy`` arg (if given) overrides cfg.strategy; otherwise use cfg.
    resolved = normalize_strategy(strategy if strategy is not None else cfg.strategy)

    if resolved is None:
        return None, None, False, "none"

    # Distribution (best effort; needed for auto + auto weights).
    pos_ratios = None
    if train_labels_np is not None:
        try:
            pos_ratios = compute_pos_ratios(train_labels_np, num_labels, multi_label)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not compute pos_ratios (%s); weights default to uniform", exc)

    if resolved == "auto":
        resolved = recommend_strategy(pos_ratios) if pos_ratios is not None else "weighted"
        if not suppress_log:
            logger.info("[imbalance] auto -> '%s' (rarest pos ratio=%.3f)",
                        resolved, float(pos_ratios.min()) if pos_ratios is not None else float("nan"))

    if resolved == "none":
        criterion = (torch.nn.BCEWithLogitsLoss() if multi_label
                     else torch.nn.CrossEntropyLoss())
        return criterion, None, False, "none"

    # Resolve per-class weights (auto from data or manual override).
    weights = None
    if cfg.weight_source == "manual" and cfg.manual_class_weights:
        weights = torch.tensor(cfg.manual_class_weights, dtype=torch.float32)
    elif pos_ratios is not None:
        if multi_label:
            # pos_neg_ratio gives the BCE pos_weight (raw neg/pos, sqrt-damped, capped).
            from .bert_base import compute_multi_label_class_weights  # lazy, avoids cycle
            oh = train_labels_np
            if np.asarray(oh).ndim == 1:
                idx = np.asarray(oh).astype(int)
                oh = np.zeros((idx.shape[0], num_labels), dtype=np.float32)
                oh[np.arange(idx.shape[0]), idx] = 1.0
            weights = compute_multi_label_class_weights(
                np.asarray(oh), method="pos_neg_ratio", max_weight=cfg.max_weight
            )
        else:
            weights = _single_label_class_weights(pos_ratios, cfg.max_weight)

    pos_weight_out = weights if multi_label else None
    use_sampler = bool(cfg.use_weighted_sampler)

    # ---- Build the concrete criterion -------------------------------------
    if resolved == "weighted":
        if multi_label:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
        strat_used = "weighted"

    elif resolved == "focal":
        if multi_label:
            criterion = SigmoidFocalLoss(gamma=cfg.focal_gamma, pos_weight=weights)
        else:
            criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=weights)
        strat_used = "focal"

    elif resolved == "asymmetric":
        if multi_label:
            from .bert_base import AdaptiveAsymmetricLoss  # lazy import
            if pos_ratios is not None:
                gamma_neg = np.clip(4.0 + np.log10(1.0 / np.clip(pos_ratios, 1e-6, 1.0)), 4.0, 8.0)
                criterion = AdaptiveAsymmetricLoss(
                    gamma_neg_per_label=gamma_neg,
                    gamma_pos=cfg.asl_gamma_pos,
                    clip=cfg.asl_clip,
                )
            else:
                from .bert_base import AsymmetricLoss
                criterion = AsymmetricLoss(
                    gamma_neg=cfg.asl_gamma_neg, gamma_pos=cfg.asl_gamma_pos, clip=cfg.asl_clip
                )
            strat_used = "asymmetric"
        else:
            # ASL is multi-label only: fall back to focal for single-label heads.
            criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=weights)
            strat_used = "focal(asl-fallback)"
            if not suppress_log:
                logger.info("[imbalance] 'asymmetric' requested on single-label head -> using focal")
    else:  # pragma: no cover - guarded by normalized_strategy
        criterion = (torch.nn.BCEWithLogitsLoss() if multi_label
                     else torch.nn.CrossEntropyLoss())
        strat_used = "none"
        use_sampler = False

    if not suppress_log:
        logger.info(
            "[imbalance] strategy=%s multi_label=%s gamma=%.1f weights=%s sampler=%s",
            strat_used, multi_label, cfg.focal_gamma,
            "auto/manual" if weights is not None else "none", use_sampler,
        )

    return criterion, pos_weight_out, use_sampler, strat_used

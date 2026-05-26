"""Statistical sampling utilities for annotation workflows.

Provides Cochran's formula for representative sample sizing and
stratified sampling with guaranteed minimum per-label representation.
"""

import math
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


def cochran_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_error: float = 0.05,
) -> int:
    """Calculate representative sample size using Cochran's formula.

    Uses the finite-population correction so the returned *n* never
    exceeds *population_size*.

    Parameters
    ----------
    population_size : int
        Total number of items in the population.
    confidence_level : float
        Desired confidence level (0.90, 0.95, or 0.99).
    margin_error : float
        Acceptable margin of error (e.g. 0.05 for ±5 %).

    Returns
    -------
    int
        Required sample size (≥ 1).
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)
    p = 0.5  # maximum variability

    # Cochran's formula for infinite population
    n0 = (z ** 2 * p * (1 - p)) / (margin_error ** 2)

    # Finite population correction
    n = n0 / (1 + (n0 - 1) / population_size)

    return max(1, math.ceil(n))


def stratified_label_sample(
    df: pd.DataFrame,
    label_column: str,
    total_size: int,
    min_per_label: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sampling ensuring minimum representation per predicted label.

    Uses proportional allocation with a guaranteed minimum per class.
    When the sum of quotas exceeds *total_size*, majority-class quotas
    are reduced first (simplified Neyman allocation).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain *label_column*.
    label_column : str
        Column holding the predicted label for each row.
    total_size : int
        Desired total sample size.
    min_per_label : int
        Floor allocation for every label.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A subset of *df* with at most *total_size* rows.
    """
    rng = np.random.default_rng(seed)

    counts = df[label_column].value_counts()
    n_labels = len(counts)
    total_available = len(df)

    if total_size >= total_available:
        return df.copy()

    # --- proportional allocation with floor ---
    quotas: Dict[str, int] = {}
    for label, cnt in counts.items():
        prop = max(min_per_label, round(total_size * cnt / total_available))
        # Can't sample more than available
        quotas[label] = min(prop, cnt)

    # --- balance if over-allocated ---
    overflow = sum(quotas.values()) - total_size
    if overflow > 0:
        # Reduce from largest quotas first
        sorted_labels = sorted(quotas, key=lambda lb: quotas[lb], reverse=True)
        for lb in sorted_labels:
            if overflow <= 0:
                break
            reduce = min(overflow, quotas[lb] - min_per_label)
            if reduce > 0:
                quotas[lb] -= reduce
                overflow -= reduce

    # --- balance if under-allocated ---
    shortfall = total_size - sum(quotas.values())
    if shortfall > 0:
        sorted_labels = sorted(quotas, key=lambda lb: quotas[lb], reverse=True)
        for lb in sorted_labels:
            if shortfall <= 0:
                break
            available = counts[lb] - quotas[lb]
            add = min(shortfall, available)
            if add > 0:
                quotas[lb] += add
                shortfall -= add

    # --- draw samples ---
    parts: List[pd.DataFrame] = []
    for label, quota in quotas.items():
        pool = df[df[label_column] == label]
        n_draw = min(quota, len(pool))
        if n_draw > 0:
            idx = rng.choice(pool.index, size=n_draw, replace=False)
            parts.append(df.loc[idx])

    if not parts:
        return df.sample(n=min(total_size, len(df)), random_state=seed)

    result = pd.concat(parts)
    # Final trim in case of rounding
    if len(result) > total_size:
        result = result.sample(n=total_size, random_state=seed)
    return result

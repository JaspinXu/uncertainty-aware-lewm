"""Small, dependency-light statistics used by the analysis scripts."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve (probability a positive outranks a negative).

    Ties count 1/2 (Mann-Whitney). Returns NaN when one class is missing.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling whole groups (plans/episodes).

    ``stat`` receives an index array into the original rows.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    point = stat(np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test for paired binary outcomes (a vs b)."""
    from scipy.stats import binomtest

    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have the same shape")
    b_only = int((~a & b).sum())  # b succeeds where a fails
    a_only = int((a & ~b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(b_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}


def within_group_mean(
    values_fn: Callable[[np.ndarray], float], groups: np.ndarray, min_size: int = 3
) -> tuple[float, int]:
    """Mean of a per-group statistic over groups where it is defined."""
    groups = np.asarray(groups)
    vals = []
    for g in np.unique(groups):
        idx = np.nonzero(groups == g)[0]
        if len(idx) < min_size:
            continue
        v = values_fn(idx)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)

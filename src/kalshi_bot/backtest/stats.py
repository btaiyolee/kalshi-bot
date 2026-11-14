"""Small, dependency-light statistics used by the evaluation harness."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1-alpha) interval for a binomial proportion. Valid with zero successes/losses,
    where the Wald interval collapses to a point."""
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def cluster_groups(day_labels, min_groups: int = 20) -> np.ndarray:
    """Group ids for the block bootstrap: calendar days, or (when there are fewer than ``min_groups``
    days) equal consecutive blocks of the given (time-ordered) labels so that there are ~min_groups."""
    labels = np.asarray(day_labels)
    _, inv = np.unique(labels, return_inverse=True)
    if inv.max() + 1 >= min_groups:
        return inv
    order = np.argsort(np.argsort(labels, kind="stable"), kind="stable")
    return (order * min_groups // max(len(labels), 1)).astype(int)


def cluster_bootstrap(values, groups, reps: int = 4000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Block bootstrap of the pooled mean (sum/count over resampled groups, so busy days weigh more,
    like the real portfolio). Returns the mean, bootstrap SE, one-sided lower/upper bounds at
    ``alpha`` and the 80%-power minimum detectable effect 2.8*SE."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        nan = float("nan")
        return {"n": len(v), "mean": nan, "se": nan, "lo": nan, "hi": nan, "mde": nan, "groups": 0}
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    G = g.max() + 1
    sums = np.bincount(g, weights=v, minlength=G)
    cnts = np.bincount(g, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(reps, G))
    c = cnts[idx].sum(1)
    m = sums[idx].sum(1)[c > 0] / c[c > 0]
    se = float(m.std(ddof=1))
    return {"n": len(v), "mean": float(v.mean()), "se": se, "lo": float(np.quantile(m, alpha)),
            "hi": float(np.quantile(m, 1 - alpha)), "mde": 2.8 * se, "groups": int(G)}

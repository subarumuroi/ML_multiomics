"""
resampling.py
=============
Grouping-aware cross-validation, permutation testing, and bootstrap indexing.

Every routine takes a ``groups`` vector (the independent unit). Splits never put
two samples from the same group on opposite sides of a train/test partition, so
pseudoreplicated samples (e.g. timepoints within one bioreactor) cannot leak.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def leave_one_group_out(groups) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-group-out splits. Returns list of (train_idx, test_idx)."""
    groups = np.asarray(groups)
    X_dummy = np.zeros((len(groups), 1))
    return list(LeaveOneGroupOut().split(X_dummy, groups=groups))


def group_kfold(groups, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group k-fold splits (whole groups held out together)."""
    groups = np.asarray(groups)
    n_groups = len(set(groups))
    if n_splits > n_groups:
        raise ValueError(f"n_splits={n_splits} > number of groups ({n_groups})")
    X_dummy = np.zeros((len(groups), 1))
    return list(GroupKFold(n_splits=n_splits).split(X_dummy, groups=groups))


def permutation_resolution(groups, labels) -> dict:
    """Finest achievable group-level permutation p-value for this design.

    Permutation is done at the GROUP level (one label per group). With G groups
    split into label counts {c1, c2, ...}, the number of distinct label
    arrangements is the multinomial coefficient G! / (c1! c2! ...). The finest
    two-sided p-value attainable is ~2 / n_arrangements.

    Returns: n_groups, label_counts, n_distinct_arrangements, finest_two_sided_p.
    """
    g = np.asarray(groups)
    y = np.asarray(labels)
    grp_label: dict = {}
    for gi, yi in zip(g, y):
        grp_label.setdefault(gi, yi)
    counts = Counter(grp_label.values())
    n_groups = len(grp_label)
    denom = 1
    for c in counts.values():
        denom *= math.factorial(c)
    n_arr = math.factorial(n_groups) // denom if n_groups else 0
    finest = min(1.0, 2.0 / n_arr) if n_arr > 0 else float("nan")
    return {
        "n_groups": n_groups,
        "label_counts": dict(counts),
        "n_distinct_arrangements": n_arr,
        "finest_two_sided_p": finest,
    }


def grouped_permutation_test(score_fn, groups, y, n_permutations: int = 1000, seed: int = 0) -> dict:
    """Permute labels at the GROUP level and compare a true score to the null.

    score_fn(y_vector) -> float (e.g. a fixed-X grouped-CV accuracy). Labels are
    permuted by reassigning whole-group labels, preserving the design's
    dependence structure.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    y = np.asarray(y)
    true_score = score_fn(y)

    uniq = list(dict.fromkeys(g.tolist()))
    grp_label = np.array([y[g == gi][0] for gi in uniq])

    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(grp_label)
        mapping = dict(zip(uniq, perm))
        y_perm = np.array([mapping[gi] for gi in g])
        null[i] = score_fn(y_perm)

    p = (np.sum(null >= true_score) + 1) / (n_permutations + 1)
    return {
        "true_score": float(true_score),
        "null": null,
        "p_value": float(p),
        "resolution": permutation_resolution(g, y),
    }


def grouped_bootstrap_indices(groups, n_bootstrap: int = 100, seed: int = 0):
    """Yield bootstrap resamples drawn at the GROUP level (resample whole groups).

    Yields arrays of sample row-indices for each bootstrap iteration. Resampling
    whole groups (not individual samples) respects the dependence structure used
    for stability selection.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = np.array(list(dict.fromkeys(g.tolist())))
    idx_by_group = {gi: np.flatnonzero(g == gi) for gi in uniq}
    for _ in range(n_bootstrap):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[gi] for gi in chosen])
        yield rows

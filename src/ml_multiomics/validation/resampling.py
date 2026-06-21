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
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score,
)


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


# ---------------------------------------------------------------------------
# Leakage-free validation (A2): preprocessing is supplied per-fold via a closure,
# so it is fit on TRAIN rows only. The primary inferences are permutation
# significance (is there signal beyond chance, against the design's resolution
# floor) and bootstrap stability (does the signal recur). Cross-validation is
# kept ONLY as a binary overfit sanity flag -- never as a performance leaderboard.
# ---------------------------------------------------------------------------

def score_predictions(y_true, y_pred, task: str) -> dict:
    """Standard metric dict for a task ('regression' or 'classification')."""
    y_true = np.asarray(y_true)
    if task == "regression":
        y_pred = np.asarray(y_pred, dtype=float)
        return {
            "task": "regression",
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    return {
        "task": "classification",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def leakage_free_cv_predict(X, y, groups, fit_predict_fn) -> np.ndarray:
    """Leave-one-group-out predictions where preprocessing happens INSIDE the fold.

    ``fit_predict_fn(X_train_raw, y_train, X_test_raw) -> preds_test`` must do all
    data-dependent work (preprocessing + model fit) on the training rows only and
    return predictions for the held-out rows. ``X`` is the RAW (pre-preprocessing)
    samples x features frame so the closure can fit a fresh preprocessor per fold.
    """
    X = pd.DataFrame(X)
    y = np.asarray(y)
    preds = np.empty(len(y), dtype=object)
    for tr, te in leave_one_group_out(groups):
        p = fit_predict_fn(X.iloc[tr], y[tr], X.iloc[te])
        preds[te] = np.asarray(p)
    return preds


def leakage_free_cv(X, y, groups, fit_predict_fn, task: str) -> dict:
    """Leakage-free grouped CV -> metric dict (+ predictions/true). A SANITY check."""
    preds = leakage_free_cv_predict(X, y, groups, fit_predict_fn)
    if task == "regression":
        preds = preds.astype(float)
    else:
        preds = np.array(list(preds))   # reinfer concrete dtype: object[int] -> int (sklearn-safe)
    out = score_predictions(y, preds, task)
    out["predictions"] = preds
    out["true"] = np.asarray(y)
    return out


def permutation_significance(score_fn, groups, y, n_permutations: int = 999, seed: int = 0) -> dict:
    """Floor-aware permutation test: is the true score beyond the label-shuffled null?

    Wraps :func:`grouped_permutation_test` and adds an honest reading against the
    design's resolution floor (the finest p the design can even produce):

      * ``design_can_reach_0p05`` -- whether p<0.05 is attainable at all here;
      * ``significant`` -- p < 0.05;
      * ``note`` -- plain-language verdict (signal / not distinguishable / design
        too small to reach significance, judge by effect size + stability).
    """
    res = grouped_permutation_test(score_fn, groups, y, n_permutations=n_permutations, seed=seed)
    floor = res["resolution"]["finest_two_sided_p"]
    p = res["p_value"]
    can = (not (isinstance(floor, float) and math.isnan(floor))) and floor <= 0.05
    res["design_can_reach_0p05"] = bool(can)
    res["significant"] = bool(p < 0.05)
    if not can:
        res["note"] = (
            f"design's finest achievable p is ~{floor:.2g}; p<0.05 is unreachable -- "
            "judge by effect size + stability, not significance."
        )
    elif res["significant"]:
        res["note"] = f"signal beyond chance (p={p:.3g}, floor={floor:.2g})."
    else:
        res["note"] = f"not distinguishable from chance (p={p:.3g})."
    return res


def bootstrap_stability(select_fn, groups, n_bootstrap: int = 50, seed: int = 0,
                        stable_threshold: float = 0.5) -> pd.DataFrame:
    """Selection frequency of features across group-level bootstraps.

    ``select_fn(row_indices) -> iterable[str]`` refits a method on the resampled
    rows and returns the names it selected (e.g. non-zero coefficients, sPLS-DA
    keepX, DIABLO selected features). Returns a DataFrame
    (feature, selection_frequency, stable) sorted by frequency. This is how an
    unstable "top features" list is exposed as unstable.
    """
    counts: Counter = Counter()
    n_ok = 0
    for rows in grouped_bootstrap_indices(groups, n_bootstrap=n_bootstrap, seed=seed):
        try:
            sel = select_fn(rows)
        except Exception:
            continue
        n_ok += 1
        for f in set(sel):
            counts[f] += 1
    denom = max(n_ok, 1)
    out = [
        {"feature": f, "selection_frequency": c / denom, "stable": (c / denom) >= stable_threshold}
        for f, c in counts.items()
    ]
    df = pd.DataFrame(out, columns=["feature", "selection_frequency", "stable"])
    if not df.empty:
        df = df.sort_values("selection_frequency", ascending=False).reset_index(drop=True)
    df.attrs["n_bootstrap_ok"] = n_ok
    return df


def overfit_flag(train_score: float, cv_score: float, margin: float = 0.2,
                 higher_is_better: bool = True) -> dict:
    """Binary overfit sanity check from the train-vs-CV gap (NOT a performance metric)."""
    gap = (train_score - cv_score) if higher_is_better else (cv_score - train_score)
    over = gap > margin
    return {
        "train_score": float(train_score),
        "cv_score": float(cv_score),
        "gap": float(gap),
        "overfit": bool(over),
        "note": ("large train-vs-CV gap -> likely overfitting; treat the fit as exploratory"
                 if over else "no large train-vs-CV gap (sanity check passed)"),
    }

"""
primitives.py
=============
Missing-aware preprocessing primitives. NaN is PRESERVED through all of these
(matching IdeaBio.jl, IdeaBio.R, and mofa_prep.py). Imputation is deliberately
NOT here — it is a separate, method-gated step (see preprocessing.imputation and
methods.base.BaseMethod.handles_missing).

All functions take and return a samples x features DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log2_transform(df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """log2(x + pseudocount), preserving NaN.

    Values <= -pseudocount are outside the domain and become NaN before the
    transform (matches mofa_prep._log2_transform).
    """
    invalid = df <= -pseudocount
    n_invalid = int(invalid.to_numpy().sum())
    out = df.mask(invalid) if n_invalid else df
    if n_invalid:
        logger.warning("log2(x+%.3g): %d value(s) <= -%.3g set to NaN", pseudocount, n_invalid, pseudocount)
    return np.log2(out + pseudocount)


def log10_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log10(x), preserving NaN. Non-positive values become NaN (matches IdeaBio.jl)."""
    invalid = df <= 0
    n_invalid = int(invalid.to_numpy().sum())
    out = df.mask(invalid) if n_invalid else df
    if n_invalid:
        logger.warning("log10: %d non-positive value(s) set to NaN", n_invalid)
    return np.log10(out)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature z-score, computed on non-NaN values (ddof=1).

    Zero/NaN-variance features are centered only (std treated as 1) to avoid
    division blow-ups; run variance_filter first to drop them properly.
    """
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=1)
    safe_sd = sd.replace(0, np.nan)
    safe_sd = safe_sd.where(safe_sd.notna(), 1.0)
    return (df - mu) / safe_sd


def variance_filter(df: pd.DataFrame, min_variance: float = 1e-8) -> pd.DataFrame:
    """Drop features whose variance (on non-NaN values) is <= min_variance."""
    var = df.var(axis=0, ddof=1)
    keep = var[var > min_variance].index
    dropped = df.shape[1] - len(keep)
    if dropped:
        logger.info("variance_filter: dropped %d/%d features <= %g", dropped, df.shape[1], min_variance)
    return df[keep]


def missingness_filter(df: pd.DataFrame, max_missing_frac: float = 0.5) -> pd.DataFrame:
    """Drop features with overall missing fraction > max_missing_frac."""
    miss = df.isna().mean(axis=0)
    keep = miss[miss <= max_missing_frac].index
    dropped = df.shape[1] - len(keep)
    if dropped:
        logger.info("missingness_filter: dropped %d/%d features > %.0f%% missing",
                    dropped, df.shape[1], 100 * max_missing_frac)
    return df[keep]


def missingness_filter_by_group(
    df: pd.DataFrame, groups: pd.Series, max_missing_count: int = 0
) -> pd.DataFrame:
    """Lab convention: drop a feature if ANY group has > max_missing_count missing.

    ``groups`` is a Series indexed like df.index giving each sample's group.
    """
    groups = groups.reindex(df.index)
    keep_mask = pd.Series(True, index=df.columns)
    for _, idx in df.groupby(groups, sort=False).groups.items():
        grp_missing = df.loc[idx].isna().sum(axis=0)
        keep_mask &= grp_missing <= max_missing_count
    keep = df.columns[keep_mask]
    dropped = df.shape[1] - len(keep)
    if dropped:
        logger.info("missingness_filter_by_group: dropped %d/%d features (>%d missing in some group)",
                    dropped, df.shape[1], max_missing_count)
    return df[keep]

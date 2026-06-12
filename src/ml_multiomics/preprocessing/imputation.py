"""
imputation.py
=============
Imputation strategies — applied ONLY when a method cannot handle missing values
(see methods.base.BaseMethod.handles_missing). Never applied before MOFA, which
models missingness natively.

Default = MetaboAnalyst (matches IdeaBio.R and IdeaBio.jl).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def metaboanalyst_impute(df: pd.DataFrame) -> pd.DataFrame:
    """MetaboAnalyst default: fill NaN per feature with 0.2 x min(positive value).

    Matches IdeaBio.jl ``impute_default_metaboanalyst`` and IdeaBio.R
    ``impute_missing_metaboanalyst``. Features with no positive values are left
    as-is (and will still contain NaN — caller should have filtered them).
    """
    out = df.copy()
    for col in out.columns:
        v = out[col]
        if not v.isna().any():
            continue
        positive = v[v > 0]
        if positive.empty:
            logger.warning("metaboanalyst_impute: feature %r has no positive values; left unfilled", col)
            continue
        out[col] = v.fillna(0.2 * positive.min())
    return out


def remove_all_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any feature containing one or more missing values (IdeaBio.jl RemoveAllMissing)."""
    before = df.shape[1]
    out = df.dropna(axis=1, how="any")
    dropped = before - out.shape[1]
    if dropped:
        logger.info("remove_all_missing: dropped %d/%d features with any NaN", dropped, before)
    return out


IMPUTERS = {
    "metaboanalyst": metaboanalyst_impute,
    "remove_all_missing": remove_all_missing,
    "remove": remove_all_missing,
}


def impute(df: pd.DataFrame, strategy: str = "metaboanalyst") -> pd.DataFrame:
    """Dispatch to a named imputation strategy."""
    if strategy not in IMPUTERS:
        raise ValueError(f"unknown imputation strategy {strategy!r}; have {list(IMPUTERS)}")
    return IMPUTERS[strategy](df)

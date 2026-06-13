"""
differential.py
===============
Differential expression matching IdeaBio.R (foldchange.R), in pure Python.

compute_volcano: for each pair of condition groups, per feature compute
  - fold change = mean(group_a) / mean(group_b)   on LINEAR data
  - log2 fold change = log2(fold change)
  - Welch two-sided t-test p-value (unequal variance) on log10 data if logx
  - FDR-adjusted q-value within each contrast
This mirrors IdeaBio.R::compute_volcano exactly (verified in tests/crosscheck).

anova_tukey: one-way ANOVA (equal variance) + Tukey HSD per feature, matching
IdeaBio.R::compute_anova_tukey.

Operates on RAW / linear abundances (samples x features). See analysis/__init__.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# map friendly names to statsmodels methods (statsmodels uses these spellings)
_FDR_METHODS = {
    "BH": "fdr_bh", "fdr": "fdr_bh", "fdr_bh": "fdr_bh", "BY": "fdr_by",
    "holm": "holm", "hochberg": "simes-hochberg", "hommel": "hommel",
    "bonferroni": "bonferroni", "none": None,
}


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def compute_volcano(
    X: pd.DataFrame,
    groups,
    logx: bool = True,
    fdr_method: str = "BH",
) -> pd.DataFrame:
    """Pairwise differential expression (matches IdeaBio.R compute_volcano).

    Parameters
    ----------
    X : samples x features DataFrame of RAW / linear abundances.
    groups : per-sample group labels (length = n_samples).
    logx : if True, the t-test runs on log10(X) (fold change stays on linear X).
    fdr_method : 'BH' (default), 'holm', 'hochberg', 'hommel', 'bonferroni',
                 'BY', or 'none'.

    Returns long-format DataFrame: contrast, feature, foldchange, log2fc,
    pvalue, qvalue (qvalue omitted if fdr_method='none').
    """
    if fdr_method not in _FDR_METHODS:
        raise ValueError(f"fdr_method must be one of {sorted(_FDR_METHODS)}")
    groups = np.asarray(groups)
    Xlin = X.to_numpy(dtype=float)
    Xtest = np.log10(np.where(Xlin > 0, Xlin, np.nan)) if logx else Xlin
    features = list(X.columns)
    uniq = sorted(pd.unique(groups))

    rows = []
    for ga, gb in combinations(uniq, 2):
        ia = np.where(groups == ga)[0]
        ib = np.where(groups == gb)[0]
        mean_a = np.nanmean(Xlin[ia], axis=0)
        mean_b = np.nanmean(Xlin[ib], axis=0)
        fc = mean_a / mean_b
        with np.errstate(invalid="ignore", divide="ignore"):
            log2fc = np.log2(fc)
        pvals = np.array([_welch_p(Xtest[ia, j], Xtest[ib, j]) for j in range(len(features))])

        contrast = f"{ga}-vs-{gb}"
        sub = pd.DataFrame({
            "contrast": contrast, "feature": features,
            "foldchange": fc, "log2fc": log2fc, "pvalue": pvals,
        })
        if fdr_method != "none":
            mask = sub["pvalue"].notna()
            q = np.full(len(sub), np.nan)
            if mask.any():
                q[mask.to_numpy()] = multipletests(
                    sub.loc[mask, "pvalue"].to_numpy(), method=_FDR_METHODS[fdr_method]
                )[1]
            sub["qvalue"] = q
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


def anova_tukey(
    X: pd.DataFrame,
    groups,
    logx: bool = True,
    fdr_method: str = "BH",
) -> pd.DataFrame:
    """One-way ANOVA (equal variance) + Tukey HSD per feature.

    Matches IdeaBio.R compute_anova_tukey: ANOVA p-value per feature, optional
    FDR across features, and Tukey HSD adjusted p per pairwise comparison.

    Returns one row per feature: pvalue, qvalue (if requested), and one
    'tukey_p_<a>-vs-<b>' column per pairwise comparison.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    if fdr_method not in _FDR_METHODS:
        raise ValueError(f"fdr_method must be one of {sorted(_FDR_METHODS)}")
    groups = np.asarray(groups)
    Xv = X.to_numpy(dtype=float)
    if logx:
        Xv = np.log10(np.where(Xv > 0, Xv, np.nan))
    features = list(X.columns)
    uniq = sorted(pd.unique(groups))

    records = []
    for j, feat in enumerate(features):
        col = Xv[:, j]
        samples = [col[groups == g] for g in uniq]
        samples = [s[~np.isnan(s)] for s in samples]
        if any(len(s) < 2 for s in samples):
            records.append({"feature": feat, "pvalue": np.nan})
            continue
        p = float(stats.f_oneway(*samples).pvalue)
        rec = {"feature": feat, "pvalue": p}
        try:
            valid = ~np.isnan(col)
            tuk = pairwise_tukeyhsd(col[valid], groups[valid])
            for (g1, g2), padj in zip(combinations_from_tukey(tuk), tuk.pvalues):
                rec[f"tukey_p_{g1}-vs-{g2}"] = float(padj)
        except Exception:
            pass
        records.append(rec)

    df = pd.DataFrame(records)
    if fdr_method != "none":
        mask = df["pvalue"].notna()
        q = np.full(len(df), np.nan)
        if mask.any():
            q[mask.to_numpy()] = multipletests(
                df.loc[mask, "pvalue"].to_numpy(), method=_FDR_METHODS[fdr_method]
            )[1]
        df["qvalue"] = q
    return df


def combinations_from_tukey(tuk) -> list[tuple]:
    """Recover the ordered group pairs Tukey HSD compared, in result order."""
    groups_unique = list(tuk.groupsunique)
    return list(combinations(groups_unique, 2))

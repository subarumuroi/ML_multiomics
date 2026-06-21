"""
standard.py
===========
Replicate the upstream STANDARD analysis (QC -> preprocessing -> differential
expression -> enrichment) before the ML divergence point, so a report can show
the pipeline is sound and matches the lab convention up to where ML begins.

Everything here reuses the lab-parity-verified building blocks
(analysis.compute_volcano / anova_tukey / ora). The only judgement this module
adds is **aggregating to one row per independent unit before DE** -- the lab's DE
(like our port) otherwise treats every sample as independent, which is
pseudoreplication for repeated-measures designs (timepoints per bioreactor).

GSEA is intentionally NOT reimplemented (no pure-Python fgsea parity); this module
produces a ranked feature list for the existing R clusterProfiler/fgsea path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from ..core.dataset import OmicsDataset
from ..core.provenance import ProvenanceTrail
from ..preprocessing.pipeline import Preprocessor
from .differential import compute_volcano, anova_tukey
from .enrichment import ora


def qc_summary(ds: OmicsDataset, excluded: Optional[list] = None) -> dict:
    """Per-block QC: shape + missingness, plus any declared exclusions."""
    rows = []
    for name in ds.block_names:
        b = ds.blocks[name]
        rows.append({
            "block": name, "omics_type": b.omics_type,
            "n_samples": b.shape[0], "n_features": b.shape[1],
            "pct_missing": round(100 * b.missing_fraction(), 2),
        })
    return {"per_block": pd.DataFrame(rows), "excluded_samples": list(excluded or [])}


def aggregate_to_units(X: pd.DataFrame, unit_labels) -> pd.DataFrame:
    """Mean per independent unit (collapses replicates/timepoints to one row each)."""
    u = pd.Series(np.asarray(unit_labels), index=X.index)
    return X.groupby(u, sort=False).mean()


def differential_expression(
    X_raw: pd.DataFrame,
    unit_labels,
    condition_labels,
    logx: bool = True,
    fdr_method: str = "BH",
    anova: bool = False,
) -> dict:
    """Standard DE across conditions, aggregated to independent units first.

    X_raw : samples x features RAW / linear abundances (DE computes its own FC + log).
    unit_labels : independent unit per sample (rows are averaged within unit).
    condition_labels : the DE grouping per sample (e.g. stage / condition).
    anova : also run ANOVA + per-feature Tukey HSD (off by default -- Tukey HSD per
        feature over thousands of proteins is very slow and the reports show only the
        volcano; turn on only when the ANOVA/Tukey table is actually needed).
    """
    Xu = aggregate_to_units(X_raw, unit_labels)
    cond = pd.Series(np.asarray(condition_labels), index=X_raw.index)
    cond_u = cond.groupby(pd.Series(np.asarray(unit_labels), index=X_raw.index), sort=False).first()
    cond_u = cond_u.loc[Xu.index]
    prov = ProvenanceTrail(name="standard-DE")
    prov.record("aggregate_to_units", {"agg": "mean"}, in_obj=X_raw, out_obj=Xu, note="one row per independent unit (no pseudoreplication)")
    volcano = compute_volcano(Xu, cond_u.to_numpy(), logx=logx, fdr_method=fdr_method)
    prov.record("compute_volcano", {"logx": logx, "fdr": fdr_method}, in_obj=Xu, note="Welch t on log + BH (IdeaBio.R parity)")
    out = {"volcano": volcano, "anova": None, "n_units": int(len(Xu)),
           "conditions": cond_u.value_counts().to_dict(), "provenance": prov}
    if anova:
        out["anova"] = anova_tukey(Xu, cond_u.to_numpy(), logx=logx, fdr_method=fdr_method)
        prov.record("anova_tukey", {"logx": logx, "fdr": fdr_method}, in_obj=Xu)
    return out


def over_representation(volcano: pd.DataFrame, universe, gene_sets: dict,
                        contrast: Optional[str] = None, log2fc_min: float = 1.0,
                        q_cutoff: float = 0.05, **ora_kw) -> dict:
    """ORA on the significant features of a (chosen) contrast vs the universe."""
    v = volcano if contrast is None else volcano[volcano["contrast"] == contrast]
    hits = v.loc[(v["qvalue"] < q_cutoff) & (v["log2fc"].abs() > log2fc_min), "feature"].unique().tolist()
    table = ora(hits, list(universe), gene_sets, **ora_kw) if hits else pd.DataFrame()
    return {"n_hits": len(hits), "hits": hits, "ora": table}


def gsea_ranked_list(volcano: pd.DataFrame, contrast: str, by: str = "signed_logp") -> pd.Series:
    """Ranked feature list for the EXTERNAL R GSEA path (clusterProfiler/fgsea).

    by='signed_logp' -> sign(log2fc) * -log10(p); 'log2fc' -> log2 fold change.
    GSEA itself is not reimplemented here (no fgsea parity claim).
    """
    v = volcano[volcano["contrast"] == contrast].copy()
    if by == "log2fc":
        rank = v.set_index("feature")["log2fc"]
    else:
        rank = (np.sign(v["log2fc"]) * -np.log10(v["pvalue"].clip(lower=1e-300)))
        rank.index = v["feature"]
    return rank.sort_values(ascending=False)


# --- continuous-target standard analysis + the standard -> ML bridge ----------
# For a continuous outcome (e.g. yield) the standard single-feature step is not a
# two-group volcano but a per-feature CORRELATION screen. This is the conventional
# univariate precursor to multivariate ML, and lets the report cross-reference the
# univariate hits against the ML consensus -- the integrated through-line.

def univariate_association(
    X_raw: pd.DataFrame,
    y,
    *,
    unit_labels=None,
    method: str = "spearman",
    min_obs_frac: float = 0.5,
    fdr_method: str = "BH",
) -> pd.DataFrame:
    """Standard UNIVARIATE screen for a CONTINUOUS target: per-feature correlation + BH FDR.

    The single-feature precursor to multivariate ML -- "which features individually track the
    outcome". Spearman (rank) by default: robust at small n and invariant to the log transform,
    so it needs no scale choice. Detection-filters features (>= min_obs_frac observed), then
    median-fills the small residual so ranks are defined. Aggregates to one row per unit first
    if unit_labels is given (no pseudoreplication).

    Returns a table [feature, rho, pvalue, qvalue, n] sorted by qvalue.
    """
    X = X_raw.copy()
    if unit_labels is not None:
        X = aggregate_to_units(X, unit_labels)
    yv = pd.Series(np.asarray(y, dtype=float), index=X.index) if not hasattr(y, "reindex") \
        else y.reindex(X.index).astype(float)
    ok = yv.notna()
    X, yv = X.loc[ok], yv.loc[ok]
    keep = X.notna().mean(axis=0) >= min_obs_frac
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "rho", "pvalue", "qvalue", "n"])
    X = X.fillna(X.median(axis=0)).fillna(0.0)
    n = int(X.shape[0])
    M = X.to_numpy(dtype=float)
    yarr = yv.to_numpy(dtype=float)
    if method == "spearman":
        M = np.apply_along_axis(rankdata, 0, M)
        yarr = rankdata(yarr)
    Mc = M - M.mean(axis=0)
    yc = yarr - yarr.mean()
    denom = np.sqrt((Mc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(denom == 0, np.nan, (Mc * yc[:, None]).sum(axis=0) / denom)
        t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(t), df=max(n - 2, 1))
    p = np.where(np.isnan(rho), np.nan, p)
    out = pd.DataFrame({"feature": X.columns, "rho": rho, "pvalue": p, "n": n})
    m = out["pvalue"].notna()
    out["qvalue"] = np.nan
    if m.any() and fdr_method.upper() != "NONE":
        out.loc[m, "qvalue"] = multipletests(out.loc[m, "pvalue"].to_numpy(), method="fdr_bh")[1]
    return out.sort_values("qvalue", na_position="last").reset_index(drop=True)


def standard_to_ml_bridge(
    univariate: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    q_cutoff: float = 0.1,
    ml_min_approaches: int = 2,
    top_n: int = 25,
) -> dict:
    """Cross-reference the univariate standard screen against the multivariate ML consensus.

    The integrated through-line: where the single-feature screen and the multivariate/stable
    ML selection AGREE (robust), where univariate flags a feature ML did NOT retain (a marginal
    single-feature signal that does not survive multivariate + stability), and where ML surfaces
    a feature with NO univariate signal (only visible jointly with others).

    Both inputs use QUALIFIED feature names (e.g. 'proteomics__P123') so they are comparable.
    `univariate` needs columns [feature, qvalue]; `consensus` needs [feature, n_approaches_stable].
    """
    uni = univariate.dropna(subset=["qvalue"])
    uni_hits = set(uni.loc[uni["qvalue"] < q_cutoff, "feature"])
    if not uni_hits:                                   # nothing clears FDR -> fall back to the strongest
        uni_hits = set(uni.head(top_n)["feature"])
    if consensus is None or len(consensus) == 0:
        ml_hits = set()
    else:
        ml_hits = set(consensus.loc[consensus["n_approaches_stable"] >= ml_min_approaches, "feature"])
        if not ml_hits:
            ml_hits = set(consensus.head(top_n)["feature"])
    both = sorted(uni_hits & ml_hits)
    uni_only = sorted(uni_hits - ml_hits)
    ml_only = sorted(ml_hits - uni_hits)
    return {
        "agreed": both, "univariate_only": uni_only, "ml_only": ml_only,
        "n_univariate": len(uni_hits), "n_ml": len(ml_hits),
        "n_agreed": len(both), "n_univariate_only": len(uni_only), "n_ml_only": len(ml_only),
        "q_cutoff": q_cutoff, "ml_min_approaches": ml_min_approaches,
    }

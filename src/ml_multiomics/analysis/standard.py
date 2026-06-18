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
) -> dict:
    """Standard DE across conditions, aggregated to independent units first.

    X_raw : samples x features RAW / linear abundances (DE computes its own FC + log).
    unit_labels : independent unit per sample (rows are averaged within unit).
    condition_labels : the DE grouping per sample (e.g. stage / condition).
    """
    Xu = aggregate_to_units(X_raw, unit_labels)
    cond = pd.Series(np.asarray(condition_labels), index=X_raw.index)
    cond_u = cond.groupby(pd.Series(np.asarray(unit_labels), index=X_raw.index), sort=False).first()
    cond_u = cond_u.loc[Xu.index]
    prov = ProvenanceTrail(name="standard-DE")
    prov.record("aggregate_to_units", {"agg": "mean"}, in_obj=X_raw, out_obj=Xu, note="one row per independent unit (no pseudoreplication)")
    volcano = compute_volcano(Xu, cond_u.to_numpy(), logx=logx, fdr_method=fdr_method)
    prov.record("compute_volcano", {"logx": logx, "fdr": fdr_method}, in_obj=Xu, note="Welch t on log + BH (IdeaBio.R parity)")
    anova = anova_tukey(Xu, cond_u.to_numpy(), logx=logx, fdr_method=fdr_method)
    prov.record("anova_tukey", {"logx": logx, "fdr": fdr_method}, in_obj=Xu)
    return {"volcano": volcano, "anova": anova, "n_units": int(len(Xu)),
            "conditions": cond_u.value_counts().to_dict(), "provenance": prov}


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

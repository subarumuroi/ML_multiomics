"""
enrichment.py
=============
Over-Representation Analysis (ORA) in pure Python — the hypergeometric test that
clusterProfiler::enricher uses, with BH FDR and gene-set size filtering.

ora(hits, universe, gene_sets): for each gene set, test whether the hit list is
over-represented relative to the universe (one-sided hypergeometric / Fisher
upper tail), then BH-adjust across tested sets.

GSEA (running-sum / fgsea) is intentionally NOT reimplemented here — pure-Python
parity with fgsea is not claimed; use the existing clusterProfiler (R) GSEA path
for that, or gseapy as an optional alternative.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def ora(
    hits,
    universe,
    gene_sets: dict,
    min_gs_size: int = 10,
    max_gs_size: int = 500,
    p_cutoff: float = 0.05,
    fdr_method: str = "BH",
) -> pd.DataFrame:
    """Hypergeometric over-representation analysis (matches enricher's statistic).

    Parameters
    ----------
    hits : iterable of "interesting" feature IDs (e.g. top-loading proteins).
    universe : iterable of all background feature IDs (e.g. all detected proteins).
    gene_sets : dict {term_id: iterable of feature IDs} (e.g. GO term -> proteins).
    min_gs_size, max_gs_size : keep only gene sets whose universe-restricted size
        falls in this range.
    p_cutoff : significance threshold on the BH-adjusted p-value (for the `sig` flag).

    Returns a DataFrame sorted by p-value: term, n_set, n_hit_in_set, gene_ratio,
    bg_ratio, pvalue, padj, sig, hit_genes.
    """
    universe = list(dict.fromkeys(universe))
    uni_set = set(universe)
    hit_set = set(hits) & uni_set
    N = len(uni_set)
    n = len(hit_set)
    if n == 0:
        logger.warning("ORA: no hits overlap the universe.")
        return pd.DataFrame(columns=["term", "n_set", "n_hit_in_set", "gene_ratio",
                                     "bg_ratio", "pvalue", "padj", "sig", "hit_genes"])

    rows = []
    for term, genes in gene_sets.items():
        gset = set(genes) & uni_set
        K = len(gset)
        if K < min_gs_size or K > max_gs_size:
            continue
        overlap = hit_set & gset
        k = len(overlap)
        if k == 0:
            continue
        # P(X >= k) for X ~ Hypergeometric(N, K, n)  ==  phyper(k-1, K, N-K, n, lower=F)
        pval = float(hypergeom.sf(k - 1, N, K, n))
        rows.append({
            "term": term, "n_set": K, "n_hit_in_set": k,
            "gene_ratio": k / n, "bg_ratio": K / N, "pvalue": pval,
            "hit_genes": ";".join(sorted(overlap)),
        })

    if not rows:
        return pd.DataFrame(columns=["term", "n_set", "n_hit_in_set", "gene_ratio",
                                     "bg_ratio", "pvalue", "padj", "sig", "hit_genes"])

    df = pd.DataFrame(rows)
    method = {"BH": "fdr_bh", "fdr": "fdr_bh", "BY": "fdr_by", "bonferroni": "bonferroni",
              "holm": "holm"}.get(fdr_method, "fdr_bh")
    df["padj"] = multipletests(df["pvalue"].to_numpy(), method=method)[1]
    df["sig"] = df["padj"] < p_cutoff
    return df.sort_values("pvalue").reset_index(drop=True)

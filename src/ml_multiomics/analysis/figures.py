"""
figures.py
==========
Standard figures for the multi-omics assessment, drawn with matplotlib from the
(cached) result objects the engine returns -- so a report renders them in
milliseconds, no recompute. Each function takes plain result pieces and returns a
matplotlib Figure; none of them fits a model.

Covers the figures these model types warrant:
  volcano (DE), PCA scores, permutation-null histogram, selection-stability bar,
  naive-vs-reduced integration-stability bar, winning-tree-model importances,
  cross-method consensus, and the block-size (imbalance) bar.

The iconic NATIVE plots (DIABLO circosPlot/plotIndiv, WGCNA dendrogram/module-trait)
are produced by R during the cache build (see rscripts/*_plots.R) and embedded as
PNGs; they are not redrawn here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_BLUE, _GREY, _RED, _GREEN = "#2B6CB0", "#A0AEC0", "#C53030", "#2F855A"


def _fig(w=6.5, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def volcano(volcano_df: pd.DataFrame, *, contrast: Optional[str] = None,
            fdr: float = 0.05, log2fc: float = 1.0, top_n_label: int = 8) -> plt.Figure:
    """Volcano: log2 fold change vs -log10 p, coloured by FDR significance."""
    v = volcano_df if contrast is None else volcano_df[volcano_df["contrast"] == contrast]
    v = v.dropna(subset=["log2fc", "pvalue"]).copy()
    v["nlogp"] = -np.log10(v["pvalue"].clip(lower=1e-300))
    sig = (v["qvalue"] < fdr) & (v["log2fc"].abs() > log2fc)
    fig, ax = _fig()
    ax.scatter(v.loc[~sig, "log2fc"], v.loc[~sig, "nlogp"], s=8, c=_GREY, alpha=0.5, linewidths=0)
    ax.scatter(v.loc[sig, "log2fc"], v.loc[sig, "nlogp"], s=12, c=_RED, alpha=0.8, linewidths=0)
    for _, r in v[sig].sort_values("nlogp", ascending=False).head(top_n_label).iterrows():
        ax.annotate(str(r["feature"])[:14], (r["log2fc"], r["nlogp"]), fontsize=6, alpha=0.8)
    ax.axvline(0, color="0.6", lw=0.6); ax.axhline(-np.log10(0.05), color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("log2 fold change"); ax.set_ylabel("-log10 p")
    ax.set_title(f"Differential expression{' -- ' + contrast if contrast else ''} "
                 f"({int(sig.sum())} sig at q<{fdr}, |log2fc|>{log2fc})")
    fig.tight_layout(); return fig


def association_volcano(assoc: pd.DataFrame, *, fdr: float = 0.1, top_n_label: int = 8) -> Optional[plt.Figure]:
    """Continuous-target analog of the volcano: per-feature correlation (rho) vs -log10 p.

    Each point is a feature from the univariate yield screen; red = passes FDR. This is the
    standard single-feature view that the multivariate ML then builds on.
    """
    if assoc is None or len(assoc) == 0:
        return None
    v = assoc.dropna(subset=["rho", "pvalue"]).copy()
    v["nlogp"] = -np.log10(v["pvalue"].clip(lower=1e-300))
    sig = v["qvalue"] < fdr
    fig, ax = _fig()
    ax.scatter(v.loc[~sig, "rho"], v.loc[~sig, "nlogp"], s=8, c=_GREY, alpha=0.5, linewidths=0)
    ax.scatter(v.loc[sig, "rho"], v.loc[sig, "nlogp"], s=14, c=_RED, alpha=0.85, linewidths=0)
    for _, r in v[sig].sort_values("nlogp", ascending=False).head(top_n_label).iterrows():
        ax.annotate(str(r["feature"])[:16], (r["rho"], r["nlogp"]), fontsize=6, alpha=0.8)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("correlation with target (rho)"); ax.set_ylabel("-log10 p"); ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"Univariate feature-target association ({int(sig.sum())} pass FDR<{fdr})")
    fig.tight_layout(); return fig


def pca_scores(pca: dict) -> plt.Figure:
    """PC1 vs PC2 of the (oversized) block, coloured by target."""
    sc = pca["scores"]; y = pca["target"]
    fig, ax = _fig(5.5, 4.5)
    if pca.get("target_type") == "continuous":
        scat = ax.scatter(sc.iloc[:, 0], sc.iloc[:, 1], c=np.asarray(y, dtype=float),
                          cmap="viridis", s=40, edgecolors="k", linewidths=0.3)
        fig.colorbar(scat, ax=ax, label="target")
    else:
        for lvl in pd.unique(np.asarray(y)):
            m = np.asarray(y) == lvl
            ax.scatter(sc.iloc[:, 0][m], sc.iloc[:, 1][m], s=40, label=str(lvl),
                       edgecolors="k", linewidths=0.3)
        ax.legend(title="target", fontsize=8)
    ax.set_xlabel(str(sc.columns[0])); ax.set_ylabel(str(sc.columns[1]))
    ax.set_title(f"PCA of {pca['block']} (samples coloured by target)")
    fig.tight_layout(); return fig


def permutation_hist(panel_row: dict) -> Optional[plt.Figure]:
    """Permutation null distribution with the observed score + p-value."""
    perm = panel_row.get("permutation", {})
    null = perm.get("null")
    if not null:
        return None
    null = np.asarray(null, dtype=float)
    obs = perm.get("true_score", panel_row.get("cv_score"))
    fig, ax = _fig(6.0, 3.4)
    ax.hist(null, bins=min(25, max(8, len(null) // 3)), color=_GREY, alpha=0.8)
    ax.axvline(obs, color=_RED, lw=2, label=f"observed = {obs:.3f}")
    ax.set_xlabel("score under label permutation"); ax.set_ylabel("count")
    ax.set_title(f"{panel_row['approach']} -- permutation null (p={perm.get('p_value'):.3g}, "
                 f"floor~{perm.get('resolution', {}).get('finest_two_sided_p', float('nan')):.2g})")
    ax.legend(fontsize=8); fig.tight_layout(); return fig


def stability_bar(stability_top: list, *, title: str = "Selection stability") -> Optional[plt.Figure]:
    """Bar of bootstrap selection frequency for the top features (stable in red)."""
    if not stability_top:
        return None
    df = pd.DataFrame(stability_top)
    fig, ax = _fig(6.0, 0.3 * len(df) + 1.2)
    colors = [_RED if s else _GREY for s in df["stable"]]
    ax.barh([str(f)[:22] for f in df["feature"]], df["selection_frequency"], color=colors)
    ax.axvline(0.5, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("bootstrap selection frequency"); ax.set_xlim(0, 1)
    ax.set_title(title); ax.invert_yaxis(); fig.tight_layout(); return fig


def naive_vs_reduced(integration: dict) -> Optional[plt.Figure]:
    """Bar of selection stability per integration variant (the block-imbalance headline)."""
    if not integration or not integration.get("groups"):
        return None
    variants = integration["groups"][0]["variants"]
    rows = [(nm, v.get("frac_stable"), v.get("n_selected")) for nm, v in variants.items()
            if "error" not in v and v.get("frac_stable") is not None]
    if not rows:
        return None
    names, frac, nsel = zip(*rows)
    colors = [_RED if n == "naive" else _GREEN for n in names]
    fig, ax = _fig(5.5, 3.2)
    bars = ax.bar(names, frac, color=colors)
    for b, ns in zip(bars, nsel):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{ns} sel", ha="center", fontsize=8)
    ax.set_ylabel("selection stability (frac recurring)"); ax.set_ylim(0, 1.08)
    ax.set_title("DIABLO integration: naive vs reduced (higher = more trustworthy)")
    fig.tight_layout(); return fig


def best_tree_row(panel: list) -> Optional[dict]:
    """The RandomForest/XGBoost candidate with the best CV score (the 'winning' tree model)."""
    trees = [r for r in panel if "error" not in r and r.get("method") in ("RandomForest", "XGBoost")
             and r.get("importances")]
    return max(trees, key=lambda r: r.get("cv_score", float("-inf"))) if trees else None


def tree_importances(panel_row: dict) -> Optional[plt.Figure]:
    """Top feature importances for the winning tree model."""
    imp = panel_row.get("importances")
    if not imp:
        return None
    df = pd.DataFrame(imp)
    col = "importance" if "importance" in df else df.columns[-1]
    fig, ax = _fig(6.0, 0.3 * len(df) + 1.2)
    ax.barh([str(f)[:22] for f in df["feature"]], df[col], color=_BLUE)
    ax.set_xlabel("importance"); ax.set_title(f"Top features -- {panel_row['approach']} (winning tree model)")
    ax.invert_yaxis(); fig.tight_layout(); return fig


def consensus_bar(consensus_df: pd.DataFrame, *, top_n: int = 12) -> Optional[plt.Figure]:
    """Features ranked by how many approaches selected them stably (the robust hypothesis)."""
    if consensus_df is None or len(consensus_df) == 0:
        return None
    df = consensus_df.head(top_n)
    fig, ax = _fig(6.0, 0.3 * len(df) + 1.2)
    ax.barh([str(f)[:22] for f in df["feature"]], df["n_approaches_stable"], color=_GREEN)
    ax.set_xlabel("# approaches selecting it stably"); ax.set_title("Cross-method consensus")
    ax.invert_yaxis(); fig.tight_layout(); return fig


def block_sizes(block_sizes_dict: dict, oversized: Optional[list] = None) -> plt.Figure:
    """Bar of per-block feature counts (log scale); oversized blocks highlighted."""
    oversized = oversized or []
    items = sorted(block_sizes_dict.items(), key=lambda kv: -kv[1])
    names, sizes = zip(*items)
    colors = [_RED if n in oversized else _BLUE for n in names]
    fig, ax = _fig(5.5, 3.0)
    ax.bar(names, sizes, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("# features (log)")
    ax.set_title("Block sizes -- imbalance motivates reducing the red block(s)")
    for i, s in enumerate(sizes):
        ax.text(i, s, str(s), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); return fig

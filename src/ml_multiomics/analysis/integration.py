"""
integration.py
==============
Block-imbalance reduction for multi-omics integration.

The headline problem: one omics layer (proteomics ~4000) dwarfs the others
(metabolites ~8-46), so naive integration (DIABLO/concatenation) is dominated by
the big block and yields an unstable "top proteins" list. The fix is to REDUCE
the oversized block to a few factors/modules BEFORE integrating, keeping the
small blocks raw -- a within-layer transform, never a cross-layer combine.

  detect_oversized_blocks  hybrid rule: a block reduces iff it has > min_features
                           AND > ratio x the median of the OTHER blocks.
  reduce_block             fit a reducer (PCA / NMF / WGCNA) on one block's matrix
                           -> (scores, provenance) where provenance maps each
                           factor/module back to its member features (so a
                           DIABLO-selected factor can be expanded for GSEA).
  balance_blocks           build a new OmicsDataset with the oversized blocks
                           replaced by their reduced scores (same block names so
                           an AnalysisSpec still applies), small blocks unchanged.

Note: the CALLER supplies the correctly-preprocessed matrix per reducer
(z-scored for PCA/WGCNA; non-negative log-only for NMF). This module does not
guess preprocessing -- that decision belongs to the spec/engine.
"""

from __future__ import annotations

import statistics
from typing import Optional

import pandas as pd

from ..core.dataset import OmicsDataset


def detect_oversized_blocks(ds: OmicsDataset, min_features: int = 200,
                            ratio: float = 5.0) -> list[str]:
    """Blocks that should be reduced before integration (hybrid threshold).

    A block is oversized iff ``n_features > min_features`` AND
    ``n_features > ratio * median(other blocks' feature counts)``. With a single
    block there is nothing to balance against, so the result is empty.
    """
    sizes = {name: ds.blocks[name].shape[1] for name in ds.block_names}
    if len(sizes) < 2:
        return []
    out = []
    for name, size in sizes.items():
        others = [s for n, s in sizes.items() if n != name]
        med = statistics.median(others)
        if size > min_features and (med == 0 or size > ratio * med):
            out.append(name)
    return out


def reduce_block(matrix: pd.DataFrame, reducer, top_n: int = 15) -> tuple[pd.DataFrame, dict]:
    """Fit ``reducer`` on one block and return (scores, provenance).

    ``reducer`` is an unfitted PCA / NMF / WGCNA instance. ``matrix`` must already
    be preprocessed appropriately for that reducer (the caller's responsibility).
    Provenance maps each reduced column to its member features:
      * WGCNA  -> the hard module membership (modules() feature->module table);
      * PCA/NMF -> the top_n highest-loading features per factor.
    """
    reducer.fit(matrix)
    scores = reducer.reduce()

    members: dict[str, list] = {}
    if hasattr(reducer, "modules"):                      # WGCNA: hard module partition
        mod = reducer.modules()
        for col in scores.columns:
            label = col[2:] if str(col).startswith("ME") else col   # MEturquoise -> turquoise
            members[col] = mod.loc[mod["module"].astype(str) == str(label), "feature"].tolist()
        kind = "WGCNA"
    else:                                                # PCA/NMF: top-loading features
        for i, col in enumerate(scores.columns, start=1):
            tf = reducer.top_features(i, top_n=top_n)
            members[col] = tf.iloc[:, 0].tolist()
        kind = type(reducer).__name__

    provenance = {
        "reducer": kind,
        "n_factors": int(scores.shape[1]),
        "membership_kind": "modules" if kind == "WGCNA" else f"top{top_n}_loadings",
        "members": members,
    }
    return scores, provenance


def balance_blocks(ds: OmicsDataset, reductions: dict) -> OmicsDataset:
    """New OmicsDataset with ``reductions`` (block -> scores DataFrame) substituted.

    Oversized blocks are replaced by their reduced scores under the SAME block name
    (so an AnalysisSpec's roles/integration_groups still apply); other blocks are
    carried unchanged. Sample metadata is preserved and the result is aligned.
    """
    new = OmicsDataset(name=f"{ds.name or 'dataset'}_balanced")
    for name in ds.block_names:
        if name in reductions:
            ot = ds.blocks[name].omics_type
            new.add_block(name, reductions[name], omics_type=f"{ot}_reduced" if ot else "reduced")
        else:
            new.add_block(name, ds.get(name), omics_type=ds.blocks[name].omics_type)
    if ds.sample_meta is not None and not ds.sample_meta.empty:
        new.set_sample_metadata(ds.sample_meta)
    new.align()
    return new

"""
analysis.py (banana)
====================
Engine for the banana "Omics & ML" interpretation report. Banana ripening is an
ORDERED categorical problem (Green < Ripe < Overripe) with independent samples
(separate fruit per stage), so each sample is its own unit (leave-one-out CV).

Framing (agreed methodology): this is hypothesis-generation, NOT a predictor
leaderboard. CV is a guardrail (overfitting + beats-chance, with its resolution
floor), reduction of the big proteomics block is principled preprocessing, and
the headline is multi-omics integration (R mixOmics DIABLO) + which features
track ripening.

Validated tools only: DIABLO/WGCNA via R (mixOmics / WGCNA package); reducers
PCA/NMF (sklearn); Ordinal (mord); RandomForest (sklearn).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_multiomics import OmicsDataset, Preprocessor, Profile
from ml_multiomics.core import parse_delimited
from ml_multiomics.methods import RandomForest, SparsePLSDA, Ordinal, NMF, PCA, DIABLO, WGCNA
from ml_multiomics.validation import permutation_resolution

DATA = Path(__file__).resolve().parents[1].parent / "ml_multiomics" / "data"
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[2] / "data"

_BLOCKS = {
    "proteomics": ("badata-proteomics-imputed.csv", "proteomics"),
    "metabolomics": ("badata-metabolomics.csv", "metabolomics"),
    "amino_acids": ("badata-amino-acids.csv", "metabolomics"),
    "aromatics": ("badata-aromatics.csv", "volatiles"),
}
_ORDER = ["Green", "Ripe", "Over"]


def load_banana(blocks=("proteomics", "metabolomics", "amino_acids", "aromatics")) -> OmicsDataset:
    ds = OmicsDataset(name="banana_ripening")
    for b in blocks:
        fn, ot = _BLOCKS[b]
        df = pd.read_csv(DATA / fn).set_index("Sample").drop(columns=[c for c in ("Groups",) if c in pd.read_csv(DATA / fn).columns])
        ds.add_block(b, df, omics_type=ot)
    ds.align()
    ds.set_sample_metadata(parse_delimited(ds.common_samples(), sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)
    return ds


def _enc(y):
    m = {c: i for i, c in enumerate(_ORDER)}
    return np.array([m[v] for v in y])


def run_banana_analysis(keepX_diablo=None) -> dict:
    ds = load_banana()
    y = ds.sample_meta["stage"].to_numpy()
    n = len(y)
    groups = np.arange(n)                       # each fruit independent -> LOO
    res = {"n_samples": n, "blocks": {b: ds.blocks[b].shape[1] for b in ds.block_names},
           "stage_order": _ORDER,
           "stage_counts": pd.Series(y).value_counts().to_dict(),
           "resolution": permutation_resolution(groups, y)}

    # --- 1. Multi-omics integration (R mixOmics DIABLO) ---
    if keepX_diablo is None:
        keepX_diablo = {"proteomics": 20, "metabolomics": 10, "amino_acids": 10, "aromatics": 15}
    dia = DIABLO(n_components=2, keepX=keepX_diablo, design=0.1).fit(ds, y, target_type="nominal")
    res["diablo_block_correlations"] = dia.block_correlations()
    res["diablo_selected"] = dia.all_selected()

    # --- 2. Reduce the big block (proteomics) — principled preprocessing ---
    prot = ds.get("proteomics")
    nmf_pre = Preprocessor(profile=Profile(transform="log2", normalize="none"))
    ds_nn = OmicsDataset("p"); ds_nn.add_block("proteomics",
        pd.read_csv(DATA / _BLOCKS["proteomics"][0]).set_index("Sample").drop(columns=["Groups"]).loc[ds.common_samples()],
        omics_type="proteomics")
    nmf_pre.run(ds_nn)
    nmf = NMF(n_components=5).fit(ds_nn.get("proteomics"))
    res["proteomics_nmf_factor1_top"] = nmf.top_features(1, top_n=10)
    res["proteomics_pca_var"] = PCA(n_components=5).fit(prot).variance_explained()

    # --- 3. Descriptive supervised panel (guardrail CV, NOT a leaderboard) ---
    panel = []

    def _cv_class(model, X, label, target_type):
        cv = model.cross_validate(X, y, groups=groups, target_type=target_type)
        row = {"approach": label, "accuracy": cv.get("accuracy")}
        if "mae" in cv:
            row["mae_ordinal"] = cv["mae"]
        return row

    # ordered model on each single block (ordinal regression)
    for b in ds.block_names:
        try:
            panel.append(_cv_class(Ordinal(model_type="AT", order=_ORDER),
                                   ds.get(b), f"Ordinal | {b}", "ordinal"))
        except Exception as e:  # mord edge cases on tiny blocks
            panel.append({"approach": f"Ordinal | {b}", "accuracy": np.nan, "note": str(e)[:40]})
    # RF on proteomics, and on NMF-reduced proteomics (reduce->predict)
    panel.append(_cv_class(RandomForest(n_estimators=200), prot, "RandomForest | proteomics", "nominal"))
    panel.append(_cv_class(RandomForest(n_estimators=200), nmf.reduce(), "RandomForest | NMF(proteomics)", "nominal"))
    # sparse PLS-DA on proteomics
    panel.append(_cv_class(SparsePLSDA(n_components=2, keepX=20), prot, "SparsePLSDA | proteomics", "nominal"))

    res["panel"] = pd.DataFrame(panel)
    # null baseline: majority-class accuracy
    res["baseline_majority_acc"] = float(pd.Series(y).value_counts().iloc[0] / n)
    return res


if __name__ == "__main__":
    r = run_banana_analysis()
    print(f"n={r['n_samples']}, blocks={r['blocks']}")
    print(f"majority-class baseline acc = {r['baseline_majority_acc']:.2f}; "
          f"finest permutation p = {r['resolution']['finest_two_sided_p']:.2g}")
    print(r["panel"].to_string(index=False))
    print("\nDIABLO block correlations:\n", r["diablo_block_correlations"].round(2).to_string())

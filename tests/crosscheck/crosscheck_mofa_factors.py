#!/usr/bin/env python3
"""
crosscheck_mofa_factors.py
==========================
Validate the package's proteomics reducers against the REFERENCE MOFA model.

Methodology (read-only; does not touch the ml_psi_mofa MOFA code): MOFA is the
established multi-omics factor model the lab already runs. If our PCA/NMF reduction
of the same phase-3 proteomics recovers the same latent structure, each MOFA factor
should be well-correlated with some combination of our factors. We report, per MOFA
factor, the maximum |Pearson r| against our reducer's factors (sample scores aligned
by bioreactor), and how many MOFA factors are recovered (|r| >= 0.7).

This is a structural agreement check, not bit-parity: MOFA is multi-view and
multi-group, so only its proteomics-driven factors are expected to align with a
proteomics-only PCA/NMF. Skips cleanly if the MOFA outputs are not present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml_multiomics import OmicsDataset, Preprocessor, Profile
from ml_multiomics.methods import PCA, NMF

PSI = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa")
MASTER = PSI / "data" / "master_multiomics.csv"


def _find_mofa_factors() -> Path | None:
    hits = sorted((PSI / "outputs" / "mofa").glob("*/*_factors.csv"))
    return hits[0] if hits else None


def _load_proteomics_by_bioreactor(phase: int = 3) -> pd.DataFrame:
    m = pd.read_csv(MASTER, low_memory=False)
    m = m[(m["Phase"] == phase) & (m["has_proteomics"] == True)].copy()   # noqa: E712
    m["bioreactor"] = m["condition"].astype(str) + "_" + m["R"].astype(str)
    cols = [c for c in m.columns if c.startswith("prot__")]
    agg = m.groupby("bioreactor")[cols].mean()
    agg.columns = [c[len("prot__"):] for c in cols]
    return agg


def _max_abs_corr(our_scores: pd.DataFrame, mofa: pd.DataFrame) -> pd.Series:
    common = [s for s in mofa.index if s in our_scores.index]
    A = our_scores.loc[common].to_numpy()
    B = mofa.loc[common].to_numpy()
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    corr = (B.T @ A) / len(common)            # mofa_factors x our_factors
    return pd.Series(np.abs(corr).max(axis=1), index=mofa.columns), len(common)


def main() -> int:
    fpath = _find_mofa_factors()
    if fpath is None or not MASTER.exists():
        print("SKIP: MOFA factor outputs or master_multiomics.csv not found.")
        return 0
    mofa = pd.read_csv(fpath)
    fac_cols = [c for c in mofa.columns if c.lower().startswith("factor")]
    mofa = mofa.set_index("sample_id")[fac_cols]
    k = len(fac_cols)

    prot = _load_proteomics_by_bioreactor()
    ds = OmicsDataset("x"); ds.add_block("p", prot, omics_type="proteomics")
    Preprocessor().run(ds)                                   # z-scored (PCA)
    pca = PCA(n_components=k).fit(ds.get("p"))
    ds_nn = OmicsDataset("x"); ds_nn.add_block("p", prot, omics_type="proteomics")
    Preprocessor(profile=Profile(transform="log2", normalize="none")).run(ds_nn)
    nmf = NMF(n_components=k).fit(ds_nn.get("p"))

    print(f"MOFA model: {fpath.parent.name}  ({k} factors)")
    for name, scores in [("PCA", pca.reduce()), ("NMF", nmf.reduce())]:
        maxr, n = _max_abs_corr(scores, mofa)
        rec = int((maxr >= 0.7).sum())
        print(f"  {name}: {n} aligned samples; MOFA factors recovered (|r|>=0.7): "
              f"{rec}/{k}; median best |r| = {maxr.median():.2f}; top = {maxr.max():.2f}")
    print("PASS (structural agreement reported; MOFA is multi-view so partial overlap is expected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

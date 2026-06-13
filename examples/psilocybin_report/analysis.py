"""
analysis.py
===========
Reusable engine for the psilocybin "Part 3: Omics -> ML" report. Predicts an
external/internal metabolite yield from phase-3 proteomics using the
ml_multiomics library, comparing several methods with leakage-free
(per-bioreactor) cross-validation.

Targets: any compound in the yields file (default psilocybin_ext; tryptamine_ext
and others available). The .qmd report calls run_yield_analysis() and narrates
the result; this module holds the validated code so the report's cells are
correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the library importable when run from the repo without install.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_multiomics import OmicsDataset, Preprocessor, Profile, RandomForest, Lasso, WGCNA, NMF
from ml_multiomics.core import parse_bioreactor_ids
from ml_multiomics.validation import permutation_resolution

# Default data locations (the psilocybin project repo). Override via run_yield_analysis args.
DEFAULT_MASTER = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/master_multiomics.csv")
DEFAULT_YIELDS = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/pseudobatched-external-yields-rates.csv")


def load_proteomics_by_bioreactor(master_path: Path, phase: int = 3) -> pd.DataFrame:
    """Phase-filtered proteomics, averaged per bioreactor (one independent row each)."""
    df = pd.read_csv(master_path, low_memory=False)
    df = df[(df["Phase"] == phase) & (df["has_proteomics"] == True)].copy()  # noqa: E712
    df["bioreactor"] = df["condition"].astype(str) + "_" + df["R"].astype(str)
    prot_cols = [c for c in df.columns if c.startswith("prot__")]
    agg = df.groupby("bioreactor")[prot_cols].mean()
    agg.columns = [c.replace("prot__", "") for c in agg.columns]
    return agg


def load_yield(yields_path: Path, compound: str, phase: int = 3) -> pd.Series:
    """Yield-over-biomass for one compound at a phase, indexed by bioreactor."""
    y = pd.read_csv(yields_path)
    y = y[(y["Compound"] == compound) & (y["Phase"] == phase)].copy()
    y["bioreactor"] = y["ferm_id"].str.replace(r"^27-PSI_", "", regex=True)
    s = y.set_index("bioreactor")["yield_over_biomass"].dropna()
    return s


def available_compounds(yields_path: Path = DEFAULT_YIELDS) -> list[str]:
    return sorted(pd.read_csv(yields_path)["Compound"].dropna().unique().tolist())


def run_yield_analysis(
    compound: str = "psilocybin_ext",
    phase: int = 3,
    master_path: Path = DEFAULT_MASTER,
    yields_path: Path = DEFAULT_YIELDS,
    n_factors: int = 5,
    seed: int = 42,
) -> dict:
    """Predict `compound` yield from phase-`phase` proteomics; compare methods.

    Methods (all leave-one-bioreactor-out CV, regression):
      - Lasso (sparse linear, on z-scored data)
      - RandomForest (direct, on z-scored data)
      - WGCNA -> RandomForest (reduce then predict, z-scored)
      - NMF  -> RandomForest (reduce then predict, log2-only non-negative)
    Returns a dict with the comparison table, per-method top features, and
    design metadata (n bioreactors, permutation resolution).
    """
    prot = load_proteomics_by_bioreactor(master_path, phase)
    yld = load_yield(yields_path, compound, phase)
    common = [b for b in prot.index if b in yld.index]
    prot = prot.loc[common]
    y = yld.loc[common].to_numpy(dtype=float)
    groups = np.arange(len(common))  # one row per bioreactor -> independent

    meta = parse_bioreactor_ids(common)

    # z-scored dataset (Lasso / RF / WGCNA)
    ds = OmicsDataset(name=f"psi_{compound}")
    ds.add_block("proteomics", prot, omics_type="proteomics")
    ds.set_sample_metadata(meta)
    Preprocessor().run(ds)
    Xz = ds.get("proteomics")

    # log2-only (non-negative) dataset for NMF
    ds_nn = OmicsDataset(name=f"psi_{compound}_nn")
    ds_nn.add_block("proteomics", prot, omics_type="proteomics")
    Preprocessor(profile=Profile(transform="log2", normalize="none")).run(ds_nn)
    Xnn = ds_nn.get("proteomics")

    rows, top_features = [], {}

    def _reg(name, model, X):
        cv = model.cross_validate(X, y, groups=groups, target_type="continuous")
        rows.append({"method": name, "r2": cv["r2"], "rmse": cv["rmse"],
                     "n_input_features": X.shape[1]})
        return cv

    las = Lasso(alpha=0.1).fit(Xz, y, target_type="continuous")
    _reg("Lasso", las, Xz)
    top_features["Lasso"] = las.coefficients(top_n=10)

    rf = RandomForest(random_state=seed).fit(Xz, y, target_type="continuous")
    _reg("RandomForest", rf, Xz)
    top_features["RandomForest"] = rf.importances(top_n=10)

    wg = WGCNA(corr_method="spearman").fit(Xz, y, target_type="continuous")
    Xw = wg.reduce(strategy="eigengenes_and_hubs")
    wgcna_n_modules = int(len(set(wg.modules()["Module"]) - {0}))
    if Xw.shape[1] >= 2:
        rfw = RandomForest(random_state=seed).fit(Xw, y, target_type="continuous")
        _reg("WGCNA->RF", rfw, Xw)

    nmf = NMF(n_components=n_factors, random_state=seed).fit(Xnn, y)
    Xn = nmf.reduce()
    rfn = RandomForest(random_state=seed).fit(Xn, y, target_type="continuous")
    _reg("NMF->RF", rfn, Xn)
    top_features["NMF_factor1"] = nmf.top_features(1, top_n=10)

    comparison = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    return {
        "compound": compound,
        "phase": phase,
        "n_bioreactors": len(common),
        "n_proteins": prot.shape[1],
        "yield_mean": float(np.mean(y)),
        "yield_std": float(np.std(y)),
        "comparison": comparison,
        "top_features": top_features,
        "wgcna_n_modules": wgcna_n_modules,
        "wgcna_reduced_cols": int(Xw.shape[1]),
        "resolution": permutation_resolution(groups, meta["condition"].to_numpy()),
        "conditions": meta["condition"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    for cpd in ("psilocybin_ext", "tryptamine_ext"):
        print("\n" + "=" * 64)
        print(f"TARGET: {cpd}")
        res = run_yield_analysis(cpd)
        print(f"  {res['n_bioreactors']} bioreactors, {res['n_proteins']} proteins; "
              f"yield mean={res['yield_mean']:.3g} sd={res['yield_std']:.3g}")
        print(res["comparison"].to_string(index=False))

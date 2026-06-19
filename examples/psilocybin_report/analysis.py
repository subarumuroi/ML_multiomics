"""
analysis.py (psilocybin)
========================
Engine for the psilocybin "Omics & ML" report. This report DECLARES its analysis
(an AnalysisSpec) and lets the ml_multiomics library execute it -- decisions live
here, logic lives in the library.

Pipeline (see the .qmd):
  1. STANDARD analysis replication (QC -> preprocessing provenance -> differential
     expression -> ORA / GSEA-ranked list) -- the lab-convention pipeline, up to
     the marked ML divergence point.
  2. ML divergence: two assessments via systematic_assessment + integration_assessment
       (a) continuous YIELD (regression; the headline) -- which omics/modules track
           metabolite yield; DIABLO integrates the blocks (regression block.spls);
       (b) CONSTRUCT C1 vs C2 (nominal) -- which omics separate the engineered
           constructs (F batch is a crossed confound, flagged as a divergence).

Data: ml_psi_mofa/data/master_multiomics.csv (RAW, multi-block) + the external
yields file. Predictor blocks = proteomics + intracellular metabolomics (CCM, PSI)
+ bioreactor; `met_ext_pb` is EXCLUDED as a predictor (unreliable as features) and
only sources the yield targets. Proteomics (~4375, ~76% missing) is auto-reduced.

Compute is configurable (n_permutations / stability_bootstrap / reducers /
run_integration). Defaults aim for soundness; lower them for a quick render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_multiomics import OmicsDataset, AnalysisSpec
from ml_multiomics.analysis import (
    systematic_assessment, integration_assessment,
    qc_summary, differential_expression, over_representation, gsea_ranked_list,
)

DEFAULT_MASTER = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/master_multiomics.csv")
DEFAULT_YIELDS = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/pseudobatched-external-yields-rates.csv")

# predictor blocks (met_ext_pb deliberately absent -- targets only, unreliable as features)
_BLOCKS = {
    "proteomics": ("prot__", "has_proteomics", "proteomics"),
    "met_ccm": ("met_int_ccm__", "has_metabolomics_int_ccm", "metabolomics"),
    "met_psi": ("met_int_psi__", "has_metabolomics_int_psi", "metabolomics"),
    "bio": ("bio__", "has_bioreactor", "bioreactor"),
}


def load_psilocybin(master_path: Path = DEFAULT_MASTER, phase: int = 3) -> OmicsDataset:
    """Multi-block dataset averaged per bioreactor (one independent unit each)."""
    m = pd.read_csv(master_path, low_memory=False)
    m = m[m["Phase"] == phase].copy()
    m["bioreactor"] = m["condition"].astype(str) + "_" + m["R"].astype(str)
    ds = OmicsDataset(name=f"psilocybin_phase{phase}")
    for name, (pre, flag, otype) in _BLOCKS.items():
        cols = [c for c in m.columns if c.startswith(pre)]
        sub = m[m[flag] == True]                                          # noqa: E712
        agg = sub.groupby("bioreactor")[cols].mean()
        agg.columns = [c[len(pre):] for c in cols]
        ds.add_block(name, agg, omics_type=otype)
    meta = m.drop_duplicates("bioreactor").set_index("bioreactor")[["condition", "F", "C", "R"]]
    meta["bioreactor"] = meta.index                                       # explicit grouping column
    ds.set_sample_metadata(meta)
    ds.align()                                                            # common bioreactors (all blocks)
    return ds


def load_yield(compound: str, yields_path: Path = DEFAULT_YIELDS, phase: int = 3) -> pd.Series:
    y = pd.read_csv(yields_path)
    y = y[(y["Compound"] == compound) & (y["Phase"] == phase)].copy()
    y["bioreactor"] = y["ferm_id"].str.replace(r"^27-PSI_", "", regex=True)
    return y.set_index("bioreactor")["yield_over_biomass"].dropna()


def available_compounds(yields_path: Path = DEFAULT_YIELDS) -> list:
    return sorted(pd.read_csv(yields_path)["Compound"].dropna().unique().tolist())


def _yield_spec(ds: OmicsDataset, compound: str, yld: pd.Series) -> AnalysisSpec:
    return AnalysisSpec(
        name=f"yield:{compound}",
        grouping_column="bioreactor",
        grouping_parser="constructed from condition+R (one row per bioreactor)",
        roles={"proteomics": "predictor", "met_ccm": "predictor",
               "met_psi": "predictor", "bio": "predictor"},
        target_type="continuous",
        target_values=yld,
        target_name=compound,
        integration_groups=[["proteomics", "met_ccm", "met_psi", "bio"]],
        min_obs_frac=0.5,
    )


def _construct_spec(ds: OmicsDataset) -> AnalysisSpec:
    return AnalysisSpec(
        name="construct:C1-vs-C2",
        grouping_column="bioreactor",
        grouping_parser="constructed from condition+R (one row per bioreactor)",
        roles={"proteomics": "predictor", "met_ccm": "predictor",
               "met_psi": "predictor", "bio": "predictor"},
        target_type="nominal",
        target_column="C",
        target_name="construct",
        integration_groups=[["proteomics", "met_ccm", "met_psi", "bio"]],
        min_obs_frac=0.5,
    )


def run_psilocybin_report(
    compound: str = "psilocybin_ext",
    *,
    master_path: Path = DEFAULT_MASTER,
    yields_path: Path = DEFAULT_YIELDS,
    n_permutations: int = 99,
    stability_bootstrap: int = 20,
    reducers=("pca", "nmf", "wgcna"),
    run_integration: bool = True,
    run_construct: bool = True,
    n_factors: int = 5,
    seed: int = 0,
) -> dict:
    """Full report engine: standard replication + yield + construct assessments."""
    ds = load_psilocybin(master_path)
    yld = load_yield(compound, yields_path)
    # restrict to bioreactors that have a yield value, then re-align blocks
    common = [b for b in ds.common_samples() if b in yld.index]
    for nm in ds.block_names:
        ds.blocks[nm].data = ds.blocks[nm].data.loc[common]
    ds.sample_meta = ds.sample_meta.loc[common]
    yld = yld.loc[common]

    out = {"compound": compound, "n_bioreactors": len(common),
           "block_sizes": {b: ds.blocks[b].shape[1] for b in ds.block_names}}

    # ---- 1. STANDARD analysis replication (descriptive; before ML divergence) ----
    prot_raw = ds.get("proteomics")
    construct = ds.sample_meta["C"].to_numpy()
    de = differential_expression(prot_raw, unit_labels=common, condition_labels=construct, logx=True)
    out["standard"] = {
        "qc": qc_summary(ds)["per_block"],
        "de_volcano": de["volcano"],
        "de_n_units": de["n_units"],
        "de_provenance": de["provenance"].to_markdown(),
        "gsea_ranked": (gsea_ranked_list(de["volcano"], de["volcano"]["contrast"].iloc[0]).head(50)
                        if len(de["volcano"]) else pd.Series(dtype=float)),
    }

    # ---- 2. ML divergence: yield (regression) ----
    spec_y = _yield_spec(ds, compound, yld)
    out["yield"] = {
        "systematic": systematic_assessment(
            ds, spec_y, n_factors=n_factors, reducers=tuple(r for r in reducers if r != "wgcna"),
            n_permutations=n_permutations, stability_bootstrap=stability_bootstrap, seed=seed),
        "spec": spec_y.describe(),
    }
    if run_integration:
        out["yield"]["integration"] = integration_assessment(
            ds, spec_y, reducers=reducers, n_factors=n_factors,
            stability_bootstrap=stability_bootstrap, seed=seed)

    # ---- 2b. ML divergence: construct C1 vs C2 (nominal) ----
    if run_construct:
        spec_c = _construct_spec(ds)
        out["construct"] = {
            "systematic": systematic_assessment(
                ds, spec_c, n_factors=n_factors, reducers=tuple(r for r in reducers if r != "wgcna"),
                n_permutations=n_permutations, stability_bootstrap=stability_bootstrap, seed=seed),
            "spec": spec_c.describe(),
        }
        if run_integration:
            out["construct"]["integration"] = integration_assessment(
                ds, spec_c, reducers=reducers, n_factors=n_factors,
                stability_bootstrap=stability_bootstrap, seed=seed)
    return out


if __name__ == "__main__":
    # quick smoke (reduced compute): confirm the wiring runs on the real data
    r = run_psilocybin_report(n_permutations=19, stability_bootstrap=5,
                              reducers=("pca", "wgcna"), run_construct=False)
    print(f"bioreactors={r['n_bioreactors']} blocks={r['block_sizes']}")
    print("standard DE volcano rows:", len(r["standard"]["de_volcano"]))
    yp = r["yield"]["systematic"]["panel"]
    print("yield panel rows:", len(yp))
    for row in yp[:6]:
        if "error" in row:
            print("  ERR", row["approach"], row["error"][:60]); continue
        print("  %-30s cv=%.3f perm_p=%.3g overfit=%s" % (
            row["approach"], row["cv_score"], row["permutation"]["p_value"], row["overfit"]["overfit"]))
    if "integration" in r["yield"]:
        g = r["yield"]["integration"]["groups"][0]
        print("integration verdict:", g["discriminator"].get("verdict", g["discriminator"].get("note")))

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
from ml_multiomics.preprocessing import FittablePreprocessor
from ml_multiomics.analysis import (
    systematic_assessment, integration_assessment, integration_blocks, detect_oversized_blocks,
    qc_summary, differential_expression, gsea_ranked_list,
    univariate_association, standard_to_ml_bridge,
    diablo_plots, wgcna_plots,
)
from ml_multiomics.core.provenance import ProvenanceTrail
from ml_multiomics.validation import permutation_resolution

_FIGDIR = Path(__file__).resolve().parent / "_figures"

DEFAULT_MASTER = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/master_multiomics.csv")
DEFAULT_YIELDS = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/pseudobatched-external-yields-rates.csv")

# PREDICTOR blocks = the genuine OMICS layers only. Two layers are deliberately NOT predictors:
#   * met_ext_pb -- external/pseudobatch metabolites; unreliable as features, they SOURCE the
#     yield targets only.
#   * bio (Biomass/OD600) -- process data, not an omics layer; and biomass is the DENOMINATOR of
#     the yield-over-biomass target, so using it as a predictor is circular (it would "predict" the
#     target through a normalization artifact, not biology). The package's purpose is interpreting
#     OMICS, so bio is excluded here; its time-aligned series is preserved (load_bio_timeseries)
#     ready to use as an easy-to-collect input / timepoint reference in future analyses.
_BLOCKS = {
    "proteomics": ("prot__", "has_proteomics", "proteomics"),
    "met_ccm": ("met_int_ccm__", "has_metabolomics_int_ccm", "metabolomics"),
    "met_psi": ("met_int_psi__", "has_metabolomics_int_psi", "metabolomics"),
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


def load_bio_timeseries(master_path: Path = DEFAULT_MASTER, phase: int = 3) -> pd.DataFrame:
    """The bio block (Biomass, OD600) kept as a TIME-ALIGNED series -- NOT a predictor.

    One row per (bioreactor, timepoint), so it can be joined to any omics layer by
    `(bioreactor, T)` to annotate a sample's growth phase. bio is excluded from the
    omics-interpretation models (process data, not omics; and circular with the
    yield-over-biomass target since biomass is the denominator); this preserves the
    rich time information ready to use as an input in a future analysis.
    """
    m = pd.read_csv(master_path, low_memory=False)
    m = m[(m["Phase"] == phase) & (m["has_bioreactor"] == True)].copy()        # noqa: E712
    m["bioreactor"] = m["condition"].astype(str) + "_" + m["R"].astype(str)
    cols = [c for c in m.columns if c.startswith("bio__")]
    out = m[["bioreactor", "T"] + cols].copy()
    out.columns = ["bioreactor", "timepoint"] + [c[len("bio__"):] for c in cols]
    return out.sort_values(["bioreactor", "timepoint"]).reset_index(drop=True)


def _yield_spec(ds: OmicsDataset, compound: str, yld: pd.Series) -> AnalysisSpec:
    return AnalysisSpec(
        name=f"yield:{compound}",
        grouping_column="bioreactor",
        grouping_parser="constructed from condition+R (one row per bioreactor)",
        roles={"proteomics": "predictor", "met_ccm": "predictor", "met_psi": "predictor"},
        target_type="continuous",
        target_values=yld,
        target_name=compound,
        integration_groups=[["proteomics", "met_ccm", "met_psi"]],
        min_obs_frac=0.5,
    )


def _construct_spec(ds: OmicsDataset) -> AnalysisSpec:
    return AnalysisSpec(
        name="construct:C1-vs-C2",
        grouping_column="bioreactor",
        grouping_parser="constructed from condition+R (one row per bioreactor)",
        roles={"proteomics": "predictor", "met_ccm": "predictor", "met_psi": "predictor"},
        target_type="nominal",
        target_column="C",
        target_name="construct",
        integration_groups=[["proteomics", "met_ccm", "met_psi"]],
        min_obs_frac=0.5,
    )


def prepare(compound: str = "psilocybin_ext", *, master_path: Path = DEFAULT_MASTER,
            yields_path: Path = DEFAULT_YIELDS) -> dict:
    """Load + align + declare specs + cheap setup stats. Fast: the report renders the
    setup/standard sections from this BEFORE the heavy ML cells run."""
    ds = load_psilocybin(master_path)
    yld = load_yield(compound, yields_path)
    common = [b for b in ds.common_samples() if b in yld.index]
    for nm in ds.block_names:
        ds.blocks[nm].data = ds.blocks[nm].data.loc[common]
    ds.sample_meta = ds.sample_meta.loc[common]
    yld = yld.loc[common]
    groups = ds.sample_meta["bioreactor"].to_numpy()
    yv = yld.to_numpy(dtype=float)
    bio_ts = load_bio_timeseries(master_path)                  # preserved (time-aligned), not a predictor
    bio_ts = bio_ts[bio_ts["bioreactor"].isin(common)].reset_index(drop=True)
    setup = {
        "n_groups": int(len(set(groups))),
        "baseline": float(np.sqrt(np.mean((yv - yv.mean()) ** 2))),
        "baseline_kind": "predict-mean RMSE (R2=0)",
        "resolution": permutation_resolution(groups, yv),
        "oversized_blocks": detect_oversized_blocks(ds),
    }
    return {"ds": ds, "yld": yld, "common": common, "compound": compound,
            "spec_yield": _yield_spec(ds, compound, yld), "spec_construct": _construct_spec(ds),
            "n_bioreactors": len(common), "setup": setup, "bio_timeseries": bio_ts,
            "block_sizes": {b: ds.blocks[b].shape[1] for b in ds.block_names}}


_PREDICTORS = ("proteomics", "met_ccm", "met_psi")


def standard_section(ctx: dict) -> dict:
    """Standard analysis ALIGNED TO THE YIELD QUESTION (so it flows into the ML, not past it):
    QC + preprocessing demo + a per-feature UNIVARIATE yield-association screen (Spearman rho + BH
    FDR -- the conventional single-feature precursor to multivariate ML) + GSEA-ranked. Fast."""
    ds, common, yld = ctx["ds"], ctx["common"], ctx["yld"]
    prot_raw = ds.get("proteomics")
    prov = ProvenanceTrail(name="standard-univariate-yield")
    parts = []
    for blk in _PREDICTORS:
        raw = ds.get(blk).loc[common]
        a = univariate_association(raw, yld, min_obs_frac=0.5, method="spearman")
        prov.record("univariate_association",
                    {"block": blk, "method": "spearman", "min_obs_frac": 0.5,
                     "n_features_in": int(raw.shape[1]), "n_tested": int(len(a))},
                    in_obj=raw, note="Spearman rho vs yield + BH FDR (per block)")
        a.insert(0, "block", blk)
        a["feature"] = blk + "__" + a["feature"].astype(str)        # qualify -> matches ML consensus
        parts.append(a)
    assoc = pd.concat(parts, ignore_index=True).sort_values("qvalue", na_position="last")
    # GSEA-ranked from the PROTEOMICS association (gene sets are protein-level), signed by rho
    prot = assoc[assoc["block"] == "proteomics"].dropna(subset=["pvalue"]).copy()
    gsea = (np.sign(prot["rho"]) * -np.log10(prot["pvalue"].clip(lower=1e-300)))
    gsea.index = prot["feature"].str.replace("proteomics__", "", regex=False)
    pp = FittablePreprocessor(omics_type="proteomics", impute="metaboanalyst", min_obs_frac=0.5).fit(prot_raw)
    return {
        "qc": qc_summary(ds)["per_block"],
        "assoc": assoc, "assoc_n_units": int(len(common)), "assoc_method": "spearman",
        "assoc_provenance": prov.to_markdown(),
        "gsea_ranked": gsea.sort_values(ascending=False).head(50),
        "preprocess": {"provenance": pp.provenance.to_markdown(), "n_in": int(prot_raw.shape[1]),
                       "n_kept": int(len(pp.keep_cols_)), "min_obs_frac": 0.5, "impute": "metaboanalyst"},
    }


def construct_de(ctx: dict) -> dict:
    """Standard DE for the CONSTRUCT (C1 vs C2) categorical question -- only built when the
    construct assessment is run. Kept so each ML question has its matching standard precursor."""
    ds, common = ctx["ds"], ctx["common"]
    de = differential_expression(ds.get("proteomics"), unit_labels=common,
                                 condition_labels=ds.sample_meta["C"].to_numpy(), logx=True)
    return {"de_volcano": de["volcano"], "de_n_units": de["n_units"],
            "de_provenance": de["provenance"].to_markdown()}


def _rel(paths) -> list:
    """Paths relative to the report dir, forward-slashed, for .qmd embedding."""
    out = []
    for p in paths:
        try:
            out.append(Path(p).resolve().relative_to(_FIGDIR.parent).as_posix())
        except ValueError:
            out.append(Path(p).as_posix())
    return out


def native_figures(ctx: dict, spec, integration_res: dict, *, tag: str,
                   n_factors: int = 5, seed: int = 0, rscript: str = "Rscript") -> dict:
    """Generate the iconic NATIVE mixOmics/WGCNA PNGs for one assessment, ONCE,
    during the cache build. Uses the integration discriminator's preferred
    representation so the circos/loadings shown ARE the model the report argues for.
    Returns paths relative to the report dir (empty + a note if R is unavailable)."""
    ds = ctx["ds"]
    groups = (integration_res or {}).get("groups") or []
    if not groups:
        return {"note": "no integration declared; native DIABLO/WGCNA figures skipped"}
    pref = (groups[0].get("discriminator") or {}).get("preferred")     # 'naive'|'pca'|'nmf'|'wgcna'
    reducer = None if pref in (None, "naive") else pref
    figdir = _FIGDIR
    out = {"reducer": pref or "naive", "diablo": [], "wgcna": []}
    try:
        blocks, yv, oversized, _ = integration_blocks(
            ds, spec, reducer=reducer, n_factors=n_factors, seed=seed, rscript=rscript)
        keepX = {ly: min(20, df.shape[1]) for ly, df in blocks.items()}
        out["diablo"] = _rel(diablo_plots(
            blocks, yv, target_type=spec.target_type, plotdir=figdir,
            prefix=f"{tag}_diablo", keepX=keepX, ncomp=2, rscript=rscript))
        # native WGCNA on the oversized block (its canonical use), trait = the target
        if oversized and ctx["setup"]["n_groups"] >= 15:
            ob = oversized[0]
            from ml_multiomics.preprocessing import FittablePreprocessor
            Z = FittablePreprocessor(omics_type=ds.blocks[ob].omics_type, impute="metaboanalyst",
                                     min_obs_frac=spec.min_obs_frac).fit_transform(ds.get(ob).loc[list(yv.index)])
            out["wgcna"] = _rel(wgcna_plots(Z, plotdir=figdir, prefix=f"{tag}_wgcna",
                                            y=yv, rscript=rscript))
        elif oversized:
            out["wgcna_note"] = f"WGCNA figure skipped (only {ctx['setup']['n_groups']} units < 15)"
    except Exception as e:                                              # native figures never break the build
        out["note"] = f"native figures unavailable: {str(e)[:160]}"
    return out


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
    """Full report engine (thin wrapper over the composable pieces below).

    The report .qmd does NOT call this -- it calls prepare() / standard_section() and the
    engine functions across SEPARATE cells so each section renders as it completes (one
    monolithic cell looks 'stuck' on Quarto). This wrapper is for CLI / `python analysis.py`.
    """
    ctx = prepare(compound, master_path=master_path, yields_path=yields_path)
    std = standard_section(ctx)
    out = {"compound": compound, "n_bioreactors": ctx["n_bioreactors"],
           "block_sizes": ctx["block_sizes"], "setup": ctx["setup"],
           "bio_timeseries": ctx["bio_timeseries"],     # excluded from predictors; preserved for future use
           "standard": {k: std[k] for k in
                        ("qc", "assoc", "assoc_n_units", "assoc_method", "assoc_provenance", "gsea_ranked")},
           "preprocess": std["preprocess"]}
    flat = tuple(r for r in reducers if r != "wgcna")          # supervised panel: out-of-fold reducers
    out["yield"] = {"spec": ctx["spec_yield"].describe(), "systematic": systematic_assessment(
        ctx["ds"], ctx["spec_yield"], n_factors=n_factors, reducers=flat,
        n_permutations=n_permutations, stability_bootstrap=stability_bootstrap, seed=seed)}
    # the standard->ML through-line: do the univariate yield hits and the multivariate ML
    # consensus agree? what does ML add / drop? (qualified feature names match across both)
    out["yield"]["bridge"] = standard_to_ml_bridge(
        std["assoc"], out["yield"]["systematic"]["consensus"], q_cutoff=0.1, ml_min_approaches=2)
    if run_integration:
        out["yield"]["integration"] = integration_assessment(
            ctx["ds"], ctx["spec_yield"], reducers=reducers, n_factors=n_factors,
            stability_bootstrap=stability_bootstrap, seed=seed)
        out["yield"]["native_figures"] = native_figures(
            ctx, ctx["spec_yield"], out["yield"]["integration"], tag="yield",
            n_factors=n_factors, seed=seed)
    if run_construct:
        out["construct"] = {"spec": ctx["spec_construct"].describe(), "standard": construct_de(ctx),
                            "systematic": systematic_assessment(
            ctx["ds"], ctx["spec_construct"], n_factors=n_factors, reducers=flat,
            n_permutations=n_permutations, stability_bootstrap=stability_bootstrap, seed=seed)}
        if run_integration:
            out["construct"]["integration"] = integration_assessment(
                ctx["ds"], ctx["spec_construct"], reducers=reducers, n_factors=n_factors,
                stability_bootstrap=stability_bootstrap, seed=seed)
            out["construct"]["native_figures"] = native_figures(
                ctx, ctx["spec_construct"], out["construct"]["integration"], tag="construct",
                n_factors=n_factors, seed=seed)
    return out


import pickle

_CACHE_DIR = Path(__file__).resolve().parent / "_cache"


def _cache_path(compound, n_permutations, stability_bootstrap, run_construct) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"results_{compound}_p{n_permutations}_b{stability_bootstrap}_c{int(run_construct)}.pkl"


def get_results(compound: str = "psilocybin_ext", *, n_permutations: int = 49,
                stability_bootstrap: int = 10, run_construct: bool = False,
                refresh: bool = False, **kw) -> dict:
    """Load cached results if present (instant), else compute once and cache.

    This is what the .qmd calls: the heavy ML runs ONCE (via `python analysis.py` or the
    first render); every subsequent render loads the pickle and is instant -- so a report
    is never 'stuck' rendering. Delete _cache/ or pass refresh=True to recompute.
    """
    path = _cache_path(compound, n_permutations, stability_bootstrap, run_construct)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_psilocybin_report(compound, n_permutations=n_permutations,
                                stability_bootstrap=stability_bootstrap,
                                run_construct=run_construct, **kw)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


if __name__ == "__main__":
    # build/refresh the results cache (so `quarto render` is instant). Prints progress.
    import argparse, time
    ap = argparse.ArgumentParser(description="Compute + cache psilocybin report results.")
    ap.add_argument("--compound", default="psilocybin_ext")
    ap.add_argument("--n-permutations", type=int, default=49)
    ap.add_argument("--stability-bootstrap", type=int, default=10)
    ap.add_argument("--construct", action="store_true", help="also run the C1-vs-C2 assessment")
    a = ap.parse_args()
    print(f"computing {a.compound} (n_perm={a.n_permutations}, boot={a.stability_bootstrap}, "
          f"construct={a.construct}) ...", flush=True)
    t = time.perf_counter()
    res = get_results(a.compound, n_permutations=a.n_permutations,
                      stability_bootstrap=a.stability_bootstrap, run_construct=a.construct, refresh=True)
    print(f"done in {time.perf_counter()-t:.0f}s; cached at "
          f"{_cache_path(a.compound, a.n_permutations, a.stability_bootstrap, a.construct)}")
    print(f"  bioreactors={res['n_bioreactors']} yield panel rows={len(res['yield']['systematic']['panel'])}")

"""
analysis.py (banana)
====================
Engine for the banana "Omics & ML" report. Banana is the package's
*arbitrary external data* demonstrator: NOTHING about it is hardcoded in the
library -- the report simply DECLARES an AnalysisSpec and the library executes it.

Design: 4 omics blocks (proteomics, metabolomics, amino-acids, aromatics) over 9
independent fruit samples, 3 ripening stages x 3 replicates. The target is the
ORDERED ripening stage (Green < Ripe < Over), so each fruit is its own independent
unit (leave-one-out) and the assessment is ordinal.

Upstream-state demonstration: the lab shipped a pre-IMPUTED proteomics file. The
standard-analysis section uses it (that's what upstream produced), but for ML we
REVERT to the raw un-imputed proteomics via the spec's `raw_sources`, so per-fold
imputation is leakage-free from raw -- exactly the "revert before ML" pattern.

Framing (unchanged): hypothesis-generation, not a leaderboard. Signal = permutation
vs the resolution floor; trust = bootstrap stability; CV is a binary overfit flag.
Proteomics dwarfs the other blocks, so it is auto-reduced before integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_multiomics import OmicsDataset, AnalysisSpec
from ml_multiomics.core import parse_delimited
from ml_multiomics.preprocessing import FittablePreprocessor
from ml_multiomics.analysis import (
    systematic_assessment, integration_assessment, integration_blocks, detect_oversized_blocks,
    qc_summary, differential_expression, gsea_ranked_list,
    diablo_plots, wgcna_plots,
)
from ml_multiomics.validation import permutation_resolution

DATA = Path(__file__).resolve().parents[2] / "data"
_ORDER = ["Green", "Ripe", "Over"]
_FIGDIR = Path(__file__).resolve().parent / "_figures"

# (file, omics_type) -- proteomics uses the lab's pre-imputed file in the dataset
_BLOCKS = {
    "proteomics": ("badata-proteomics-imputed.csv", "proteomics"),
    "metabolomics": ("badata-metabolomics.csv", "metabolomics"),
    "amino_acids": ("badata-amino-acids.csv", "metabolomics"),
    "aromatics": ("badata-aromatics.csv", "volatiles"),
}
_PROT_RAW = "Unused/badata-proteomics-unimputed.csv"   # the raw revert source for ML


def _read_block(fname: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / fname)
    idcol = "Sample" if "Sample" in df.columns else df.columns[0]
    df = df.set_index(idcol)
    return df.drop(columns=[c for c in ("Groups",) if c in df.columns])


def load_banana() -> OmicsDataset:
    ds = OmicsDataset(name="banana_ripening")
    for name, (fname, otype) in _BLOCKS.items():
        ds.add_block(name, _read_block(fname), omics_type=otype)
    ds.align()
    meta = parse_delimited(ds.common_samples(), sep="-", names=("stage", "replicate"))
    meta["unit"] = meta.index                               # each fruit its own independent unit
    ds.set_sample_metadata(meta)
    return ds


def _raw_proteomics(common) -> pd.DataFrame:
    """The raw un-imputed proteomics, with its sample IDs reconciled to the
    processed file's convention ('Green_Banana_1'/'Overripe_3' -> 'Green-1'/'Over-3')."""
    raw = _read_block(_PROT_RAW)

    def remap(s: str) -> str:
        stage, _, rep = str(s).replace("_Banana", "").rpartition("_")
        stage = {"Overripe": "Over"}.get(stage, stage)
        return f"{stage}-{rep}"

    raw.index = [remap(s) for s in raw.index]
    return raw.loc[[s for s in common if s in raw.index]]


def _stage_spec(ds: OmicsDataset, raw_prot: pd.DataFrame) -> AnalysisSpec:
    return AnalysisSpec(
        name="ripening-stage",
        grouping_column="unit",
        grouping_parser="parse_delimited(stage-replicate); each fruit is its own unit",
        roles={b: "predictor" for b in ds.block_names},
        target_type="ordinal",
        target_column="stage",
        target_name="ripening_stage",
        ordinal_order=_ORDER,
        integration_groups=[list(ds.block_names)],
        min_obs_frac=0.5,
        # REVERT before ML: use the raw un-imputed proteomics (not the pre-imputed file)
        raw_sources={"proteomics": raw_prot},
    )


def prepare() -> dict:
    """Load + align + declare the ordinal-stage spec + cheap setup stats. Fast; the report
    renders setup/standard from this BEFORE the heavy ML cells run."""
    ds = load_banana()
    common = ds.common_samples()
    raw_prot = _raw_proteomics(common)
    stage = ds.sample_meta["stage"].to_numpy()
    groups = ds.sample_meta["unit"].to_numpy()
    setup = {
        "n_groups": int(len(set(groups))),
        "baseline": float(pd.Series(stage).value_counts().iloc[0] / len(stage)),
        "baseline_kind": "majority-class accuracy",
        "resolution": permutation_resolution(groups, stage),
        "oversized_blocks": detect_oversized_blocks(ds),
    }
    return {"ds": ds, "common": common, "raw_prot": raw_prot, "spec": _stage_spec(ds, raw_prot),
            "n_samples": len(common), "setup": setup,
            "stage_counts": ds.sample_meta["stage"].value_counts().to_dict(),
            "block_sizes": {b: ds.blocks[b].shape[1] for b in ds.block_names}}


def standard_section(ctx: dict) -> dict:
    """Standard analysis (QC + DE + preprocessing demo + GSEA-ranked). Fast; runs before ML."""
    ds, common, raw_prot = ctx["ds"], ctx["common"], ctx["raw_prot"]
    prot_std = ds.get("proteomics")                       # lab's pre-imputed file (standard section)
    stage = ds.sample_meta["stage"].to_numpy()
    de = differential_expression(prot_std, unit_labels=common, condition_labels=stage, logx=True)
    pp = FittablePreprocessor(omics_type="proteomics", impute="metaboanalyst", min_obs_frac=0.5).fit(raw_prot)
    return {
        "qc": qc_summary(ds)["per_block"],
        "de_volcano": de["volcano"], "de_n_units": de["n_units"],
        "de_provenance": de["provenance"].to_markdown(),
        "gsea_ranked": (gsea_ranked_list(de["volcano"], de["volcano"]["contrast"].iloc[0]).head(40)
                        if len(de["volcano"]) else pd.Series(dtype=float)),
        "preprocess": {"provenance": pp.provenance.to_markdown(), "n_in": int(raw_prot.shape[1]),
                       "n_kept": int(len(pp.keep_cols_)), "min_obs_frac": 0.5, "impute": "metaboanalyst"},
    }


def _rel(paths) -> list:
    """Paths relative to the report dir, forward-slashed, for .qmd embedding."""
    out = []
    for p in paths:
        try:
            out.append(Path(p).resolve().relative_to(_FIGDIR.parent).as_posix())
        except ValueError:
            out.append(Path(p).as_posix())
    return out


def native_figures(ctx: dict, integration_res: dict, *, tag: str = "stage",
                   n_factors: int = 5, seed: int = 0, rscript: str = "Rscript") -> dict:
    """Iconic NATIVE mixOmics DIABLO PNGs for the integration, generated ONCE during
    the cache build on the discriminator's preferred representation. WGCNA figures are
    skipped at n=9 (< its ~15-20 range), handled by the n_groups guard. Returns paths
    relative to the report dir (empty + a note if R is unavailable)."""
    ds, spec = ctx["ds"], ctx["spec"]
    groups = (integration_res or {}).get("groups") or []
    if not groups:
        return {"note": "no integration declared; native figures skipped"}
    pref = (groups[0].get("discriminator") or {}).get("preferred")
    reducer = None if pref in (None, "naive") else pref
    out = {"reducer": pref or "naive", "diablo": [], "wgcna": []}
    try:
        blocks, yv, oversized, _ = integration_blocks(
            ds, spec, reducer=reducer, n_factors=n_factors, seed=seed, rscript=rscript)
        keepX = {ly: min(20, df.shape[1]) for ly, df in blocks.items()}
        out["diablo"] = _rel(diablo_plots(
            blocks, yv, target_type=spec.target_type, plotdir=_FIGDIR,
            prefix=f"{tag}_diablo", keepX=keepX, ncomp=max(2, n_factors), rscript=rscript))
        if oversized and ctx["setup"]["n_groups"] >= 15:
            ob = oversized[0]
            Z = FittablePreprocessor(omics_type=ds.blocks[ob].omics_type, impute="metaboanalyst",
                                     min_obs_frac=spec.min_obs_frac).fit_transform(
                spec.raw_sources.get(ob, ds.get(ob)).loc[list(yv.index)])
            out["wgcna"] = _rel(wgcna_plots(Z, plotdir=_FIGDIR, prefix=f"{tag}_wgcna",
                                            y=yv, rscript=rscript))
        elif oversized:
            out["wgcna_note"] = f"WGCNA figure skipped (only {ctx['setup']['n_groups']} units < 15)"
    except Exception as e:
        out["note"] = f"native figures unavailable: {str(e)[:160]}"
    return out


def run_banana_report(
    *,
    n_permutations: int = 49,
    stability_bootstrap: int = 10,
    reducers=("pca", "nmf"),       # WGCNA omitted: n=9 is below its ~15-20 sample range
    run_integration: bool = True,
    n_factors: int = 5,
    seed: int = 0,
) -> dict:
    """Thin wrapper over prepare()/standard_section() + the engine (for CLI; the .qmd calls
    the pieces across cells so each section renders as it completes)."""
    ctx = prepare()
    std = standard_section(ctx)
    out = {"n_samples": ctx["n_samples"], "block_sizes": ctx["block_sizes"],
           "stage_counts": ctx["stage_counts"], "setup": ctx["setup"],
           "standard": {k: std[k] for k in ("qc", "de_volcano", "de_n_units", "de_provenance", "gsea_ranked")},
           "preprocess": std["preprocess"]}
    flat = tuple(r for r in reducers if r != "wgcna")
    out["stage"] = {"spec": ctx["spec"].describe(), "systematic": systematic_assessment(
        ctx["ds"], ctx["spec"], n_factors=n_factors, reducers=flat,
        n_permutations=n_permutations, stability_bootstrap=stability_bootstrap, seed=seed)}
    if run_integration:
        out["stage"]["integration"] = integration_assessment(
            ctx["ds"], ctx["spec"], reducers=reducers, n_factors=n_factors,
            stability_bootstrap=stability_bootstrap, seed=seed)
        out["stage"]["native_figures"] = native_figures(
            ctx, out["stage"]["integration"], tag="stage", n_factors=n_factors, seed=seed)
    return out


import pickle

_CACHE_DIR = Path(__file__).resolve().parent / "_cache"


def _cache_path(n_permutations, stability_bootstrap) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"results_stage_p{n_permutations}_b{stability_bootstrap}.pkl"


def get_results(*, n_permutations: int = 49, stability_bootstrap: int = 10,
                refresh: bool = False, **kw) -> dict:
    """Load cached results if present (instant), else compute once and cache.

    The .qmd calls this: the heavy ML runs ONCE (via `python analysis.py` or the first
    render) and every subsequent render loads the pickle and is instant. Delete _cache/
    or pass refresh=True to recompute.
    """
    path = _cache_path(n_permutations, stability_bootstrap)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    res = run_banana_report(n_permutations=n_permutations,
                            stability_bootstrap=stability_bootstrap, **kw)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


if __name__ == "__main__":
    # build/refresh the results cache (so `quarto render` is instant). Prints progress.
    import argparse, time
    ap = argparse.ArgumentParser(description="Compute + cache banana report results.")
    ap.add_argument("--n-permutations", type=int, default=49)
    ap.add_argument("--stability-bootstrap", type=int, default=10)
    a = ap.parse_args()
    print(f"computing banana (n_perm={a.n_permutations}, boot={a.stability_bootstrap}) ...", flush=True)
    t = time.perf_counter()
    res = get_results(n_permutations=a.n_permutations, stability_bootstrap=a.stability_bootstrap, refresh=True)
    print(f"done in {time.perf_counter()-t:.0f}s; cached at "
          f"{_cache_path(a.n_permutations, a.stability_bootstrap)}")
    print(f"  fruit={res['n_samples']} stage panel rows={len(res['stage']['systematic']['panel'])}")

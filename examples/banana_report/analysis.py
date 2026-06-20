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
    systematic_assessment, integration_assessment,
    qc_summary, differential_expression, gsea_ranked_list,
)

DATA = Path(__file__).resolve().parents[2] / "data"
_ORDER = ["Green", "Ripe", "Over"]

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


def run_banana_report(
    *,
    n_permutations: int = 99,
    stability_bootstrap: int = 20,
    reducers=("pca", "nmf"),       # WGCNA omitted: n=9 is below its ~15-20 sample range
    run_integration: bool = True,
    n_factors: int = 5,
    seed: int = 0,
) -> dict:
    ds = load_banana()
    common = ds.common_samples()
    raw_prot = _raw_proteomics(common)
    spec = _stage_spec(ds, raw_prot)

    out = {"n_samples": len(common),
           "block_sizes": {b: ds.blocks[b].shape[1] for b in ds.block_names},
           "stage_counts": ds.sample_meta["stage"].value_counts().to_dict()}

    # ---- standard replication (uses the lab's pre-imputed proteomics) ----
    prot_std = ds.get("proteomics")
    stage = ds.sample_meta["stage"].to_numpy()
    de = differential_expression(prot_std, unit_labels=common, condition_labels=stage, logx=True)
    out["standard"] = {
        "qc": qc_summary(ds)["per_block"],
        "de_volcano": de["volcano"],
        "de_n_units": de["n_units"],
        "de_provenance": de["provenance"].to_markdown(),
        "gsea_ranked": (gsea_ranked_list(de["volcano"], de["volcano"]["contrast"].iloc[0]).head(40)
                        if len(de["volcano"]) else pd.Series(dtype=float)),
    }

    # ---- preprocessing demonstration: steps + intermediate shapes the ML refits inside
    #      every fold (shown once on the RAW un-imputed proteomics the ML reverts to) ----
    _pp = FittablePreprocessor(omics_type="proteomics", impute="metaboanalyst", min_obs_frac=0.5)
    _pp.fit(raw_prot)
    out["preprocess"] = {
        "provenance": _pp.provenance.to_markdown(),
        "n_in": int(raw_prot.shape[1]), "n_kept": int(len(_pp.keep_cols_)),
        "min_obs_frac": 0.5, "impute": "metaboanalyst",
    }

    # ---- ML divergence: ordinal stage (reverts to raw proteomics) ----
    out["stage"] = {
        "spec": spec.describe(),
        "systematic": systematic_assessment(
            ds, spec, n_factors=n_factors, reducers=tuple(r for r in reducers if r != "wgcna"),
            n_permutations=n_permutations, stability_bootstrap=stability_bootstrap, seed=seed),
    }
    if run_integration:
        out["stage"]["integration"] = integration_assessment(
            ds, spec, reducers=reducers, n_factors=n_factors,
            stability_bootstrap=stability_bootstrap, seed=seed)
    return out


if __name__ == "__main__":
    r = run_banana_report(n_permutations=19, stability_bootstrap=5,
                          reducers=("pca",), run_integration=False)
    print(f"samples={r['n_samples']} blocks={r['block_sizes']} stages={r['stage_counts']}")
    print("standard DE rows:", len(r["standard"]["de_volcano"]))
    print("spec:\n" + r["stage"]["spec"])
    for row in r["stage"]["systematic"]["panel"]:
        if "error" in row:
            print("  ERR", row["approach"], row["error"][:60]); continue
        print("  %-30s inputs=%-5d cv=%.3f perm_p=%.3g overfit=%s" % (
            row["approach"], row["n_inputs"], row["cv_score"],
            row["permutation"].get("p_value", float("nan")), row["overfit"]["overfit"]))

"""
test_scaffold.py
================
Smoke test for the consolidated foundation (Task #15). Runs as a plain script
(no pytest dependency required):

    ./venv/Scripts/python tests/test_scaffold.py

Exercises, on the REAL banana + psilocybin data:
  * OmicsDataset container with N blocks of DIFFERENT sample sets
  * align-by-ID intersection (banana aromatics n=12 vs others n=9)
  * pluggable metadata parsers (bioreactor F#C#R#T#, delimited Green-1)
  * missing-aware preprocessing (NaN preserved through transform + zscore)
  * the handles_missing gate (imputation only for methods that need it)
  * grouping-aware CV (no bioreactor spans train/test) + permutation resolution
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from anywhere
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_multiomics.core import OmicsDataset, parse_bioreactor_ids, parse_delimited
from ml_multiomics.preprocessing import Preprocessor, Profile
from ml_multiomics.methods.base import BaseMethod
from ml_multiomics.validation import (
    leave_one_group_out,
    permutation_resolution,
)

BANANA = ROOT / "data"
PSILO_MASTER = Path("C:/Users/uqkmuroi/gitcode/ml_psi_mofa/data/master_multiomics.csv")

PASS = "[PASS]"
FAIL = "[FAIL]"
_failures = []


def check(cond: bool, msg: str):
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        _failures.append(msg)
    assert cond, msg   # makes each pytest test fail meaningfully on a real error


# ---------------------------------------------------------------------------
def _build_banana_proteomics_ds():
    """A single-block (proteomics) banana dataset with stage metadata."""
    df = pd.read_csv(BANANA / "badata-proteomics-imputed.csv")
    df = df.set_index("Sample").drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("proteomics", df, omics_type="proteomics")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    return ds


def test_banana_container_and_alignment():
    print("\n=== Banana: container + align-by-ID intersection ===")
    files = {
        "amino_acids": "badata-amino-acids.csv",
        "aromatics": "badata-aromatics.csv",
        "metabolomics": "badata-metabolomics.csv",
        "proteomics": "badata-proteomics-imputed.csv",
    }
    omics_type = {
        "amino_acids": "metabolomics", "aromatics": "volatiles",
        "metabolomics": "metabolomics", "proteomics": "proteomics",
    }
    ds = OmicsDataset(name="banana")
    for blk, fn in files.items():
        df = pd.read_csv(BANANA / fn)
        df = df.set_index("Sample").drop(columns=[c for c in ("Groups",) if c in df.columns])
        ds.add_block(blk, df, omics_type=omics_type[blk])

    print(ds.summary().to_string(index=False))
    check(ds.blocks["aromatics"].shape[0] == 12, "aromatics has 12 samples")
    check(ds.blocks["amino_acids"].shape[0] == 9, "amino_acids has 9 samples")

    common = ds.common_samples()
    check(len(common) == 9, f"common samples = 9 (got {len(common)})")

    ds.align()
    check(all(ds.blocks[b].shape[0] == 9 for b in ds.block_names),
          "all blocks aligned to 9 samples")

    # metadata via delimited parser
    meta = parse_delimited(common, sep="-", names=("stage", "replicate"))
    ds.set_sample_metadata(meta)
    check(set(meta["stage"]) <= {"Green", "Ripe", "Over"},
          f"stages parsed: {sorted(set(meta['stage']))}")


def test_banana_preprocessing_preserves_structure():
    print("\n=== Banana: missing-aware preprocessing ===")
    ds = _build_banana_proteomics_ds()
    Preprocessor().run(ds)
    prot = ds.blocks["proteomics"]
    check(prot.transformed and prot.normalized, "proteomics flagged transformed+normalized")
    # z-scored: each feature ~mean 0
    means = ds.get("proteomics").mean(axis=0).abs()
    check(float(means.max()) < 1e-6, "z-scored features have ~zero mean")
    check("transform: log2" in prot.provenance and "normalize: zscore" in prot.provenance,
          "provenance recorded transform + normalize")


def test_missingness_gate():
    print("\n=== handles_missing gate ===")
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 2.0, np.nan, 8.0]})

    class MissingOK(BaseMethod):
        handles_missing = True
        def fit(self, X, y=None, **kw):
            self.X_ = self._prepare_X(X); return self

    class MissingNo(BaseMethod):
        handles_missing = False
        def fit(self, X, y=None, **kw):
            self.X_ = self._prepare_X(X); return self

    ok = MissingOK().fit(X)
    check(bool(ok.X_.isna().any().any()), "handles_missing=True keeps NaN")

    no = MissingNo(impute="metaboanalyst").fit(X)
    check(not bool(no.X_.isna().any().any()), "handles_missing=False imputes NaN")
    # metaboanalyst fill = 0.2 * min positive in column 'a' = 0.2 * 1 = 0.2
    check(abs(no.X_.loc[1, "a"] - 0.2) < 1e-9, "metaboanalyst fill = 0.2 x min positive")


def test_psilo_grouping_and_resolution():
    print("\n=== Psilocybin: bioreactor parser + grouped CV (no leakage) ===")
    if not PSILO_MASTER.exists():
        check(False, f"psilo master not found at {PSILO_MASTER}")
        return
    df = pd.read_csv(PSILO_MASTER, low_memory=False)
    df = df[(df["Phase"] == 3) & (df["has_proteomics"] == True)].copy()  # noqa: E712
    df["sid"] = df["sample_id"].astype(str)
    prot_cols = [c for c in df.columns if c.startswith("prot__")]
    block = df.set_index("sid")[prot_cols]

    ds = OmicsDataset(name="psilo")
    ds.add_block("proteomics", block, omics_type="proteomics")
    meta = parse_bioreactor_ids(block.index)
    ds.set_sample_metadata(meta)

    check(meta["condition"].notna().all(), "all sample IDs parsed to a condition (F#C#)")
    check((meta["condition"] == meta["strain"] + "_" + meta["construct"]).all(),
          "condition == strain_construct (F#C#)")
    n_react = meta["bioreactor"].nunique()
    print(f"  bioreactors: {n_react}, samples: {len(meta)}")

    # grouped CV: no bioreactor may appear in both train and test
    groups = meta["bioreactor"].to_numpy()
    splits = leave_one_group_out(groups)
    leak = False
    for tr, te in splits:
        if set(groups[tr]) & set(groups[te]):
            leak = True
            break
    check(not leak, f"no bioreactor leaks across {len(splits)} LOGO folds")

    # honest resolution at the condition level (group = bioreactor, label = condition)
    res = permutation_resolution(meta["bioreactor"], meta["condition"])
    print(f"  permutation resolution: {res['n_groups']} groups, "
          f"{res['n_distinct_arrangements']} arrangements, "
          f"finest two-sided p = {res['finest_two_sided_p']:.2g}")
    check(res["n_groups"] == n_react, "resolution groups == #bioreactors")


def main():
    test_banana_container_and_alignment()
    test_banana_preprocessing_preserves_structure()
    test_missingness_gate()
    test_psilo_grouping_and_resolution()

    print("\n" + "=" * 60)
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"   - {f}")
        sys.exit(1)
    print(f"{PASS} all scaffold smoke checks passed")


if __name__ == "__main__":
    main()

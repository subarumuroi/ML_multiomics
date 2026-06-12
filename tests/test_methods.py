"""
test_methods.py
===============
Smoke test for ported ML methods on the BaseMethod foundation. Runs as a plain
script (no pytest needed):

    ./venv/Scripts/python tests/test_methods.py

Currently covers RandomForest (classification + regression), exercising:
  * the handles_missing gate (fit on data with NaN -> imputed JIT)
  * Gini + permutation importance
  * grouping-aware leave-one-group-out CV (classification & regression)
  * group-level permutation test + honest resolution reporting
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_multiomics.core import OmicsDataset, parse_delimited
from ml_multiomics.preprocessing import Preprocessor
from ml_multiomics.methods import RandomForest, SparsePLSDA, DIABLO, WGCNA

BANANA = ROOT / "data"

PASS, FAIL = "[PASS]", "[FAIL]"
_failures = []


def check(cond, msg):
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        _failures.append(msg)


def test_rf_classification_banana():
    print("\n=== RandomForest classification (banana proteomics) ===")
    df = pd.read_csv(BANANA / "badata-proteomics-imputed.csv").set_index("Sample")
    df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("proteomics", df, omics_type="proteomics")
    meta = parse_delimited(df.index, sep="-", names=("stage", "replicate"))
    ds.set_sample_metadata(meta)
    Preprocessor().run(ds)

    X = ds.get("proteomics")
    y = ds.sample_meta["stage"].to_numpy()
    groups = np.arange(len(y))  # each banana replicate is its own independent unit (LOO)

    rf = RandomForest(n_estimators=200).fit(X, y, target_type="nominal")
    check(rf.task_ == "classification", "auto-resolved task = classification")
    imp = rf.importances()
    check(len(imp) == X.shape[1], f"importances cover all {X.shape[1]} features")

    cv = rf.cross_validate(X, y, groups=groups, target_type="nominal")
    check(0.0 <= cv["accuracy"] <= 1.0, f"grouped-CV accuracy in [0,1]: {cv['accuracy']:.3f}")
    check("balanced_accuracy" in cv, "balanced accuracy reported")


def test_rf_regression_synthetic():
    print("\n=== RandomForest regression (synthetic, grouped) ===")
    rng = np.random.default_rng(0)
    n_groups, per = 10, 2
    n = n_groups * per
    X = rng.normal(size=(n, 8))
    # signal in feature 0; group-structured noise
    groups = np.repeat(np.arange(n_groups), per)
    y = 3.0 * X[:, 0] + rng.normal(scale=0.3, size=n)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])

    rf = RandomForest(n_estimators=200).fit(Xdf, y, target_type="continuous")
    check(rf.task_ == "regression", "auto-resolved task = regression")

    cv = rf.cross_validate(Xdf, y, groups=groups, target_type="continuous")
    check(cv["task"] == "regression" and "r2" in cv and "rmse" in cv,
          f"regression CV returns r2/rmse (r2={cv['r2']:.2f})")

    pt = rf.permutation_test(Xdf, y, groups=groups, n_permutations=50, seed=1,
                             target_type="continuous")
    check(0.0 < pt["p_value"] <= 1.0, f"group permutation p in (0,1]: {pt['p_value']:.3f}")
    check("resolution" in pt and pt["resolution"]["n_groups"] == n_groups,
          "permutation resolution reports #groups")


def test_rf_missingness_gate():
    print("\n=== RandomForest handles_missing gate ===")
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(loc=10, scale=2, size=(12, 5)),
                     columns=[f"f{i}" for i in range(5)])
    X.iloc[3, 1] = np.nan
    X.iloc[7, 4] = np.nan
    y = np.array(["a", "b"] * 6)
    rf = RandomForest(n_estimators=50).fit(X, y, target_type="nominal")
    check(rf._fitted, "RF fit succeeds on data with NaN (imputed just-in-time)")


def test_splsda_banana():
    print("\n=== SparsePLSDA classification + stability (banana) ===")
    df = pd.read_csv(BANANA / "badata-proteomics-imputed.csv").set_index("Sample")
    df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("proteomics", df, omics_type="proteomics")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)

    X = ds.get("proteomics")
    y = ds.sample_meta["stage"].to_numpy()
    groups = np.arange(len(y))

    sp = SparsePLSDA(n_components=2, keepX=20).fit(X, y, target_type="nominal")
    vip = sp.vip()
    check(len(vip) == X.shape[1], f"VIP covers all {X.shape[1]} features")

    cv = sp.cross_validate(X, y, groups=groups)
    check(0.0 <= cv["accuracy"] <= 1.0, f"grouped-CV accuracy in [0,1]: {cv['accuracy']:.3f}")

    stab = sp.stability_selection(X, y, groups=groups, n_bootstrap=10, seed=0)
    check(stab["selection_frequency"].between(0, 1).all(),
          "stability frequencies in [0,1]")
    n_selected = int((stab["selection_frequency"] > 0).sum())
    check(n_selected > 0, f"sparse selection picks features ({n_selected} ever-selected)")


def test_diablo_banana():
    print("\n=== DIABLO multi-block (banana: 3 omics) ===")
    files = {
        "proteomics": ("badata-proteomics-imputed.csv", "proteomics"),
        "metabolomics": ("badata-metabolomics.csv", "metabolomics"),
        "amino_acids": ("badata-amino-acids.csv", "metabolomics"),
    }
    ds = OmicsDataset(name="banana")
    for blk, (fn, otype) in files.items():
        df = pd.read_csv(BANANA / fn).set_index("Sample")
        df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
        ds.add_block(blk, df, omics_type=otype)
    ds.align()
    ds.set_sample_metadata(parse_delimited(ds.common_samples(), sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)

    y = ds.sample_meta["stage"].to_numpy()
    groups = np.arange(len(y))
    keepX = {"proteomics": 20, "metabolomics": 10, "amino_acids": 10}

    dia = DIABLO(n_components=2, keepX=keepX, design=0.1).fit(ds, y, target_type="nominal")
    corr = dia.block_correlations()
    check(corr.shape == (3, 3), "block-correlation matrix is 3x3")
    av = dia.all_vip()
    check(set(av["block"].unique()) == set(files), "VIP returned for all 3 blocks")

    cv = dia.cross_validate(ds, y, groups=groups)
    check(0.0 <= cv["accuracy"] <= 1.0, f"grouped-CV accuracy in [0,1]: {cv['accuracy']:.3f}")


def test_wgcna_reduce_then_predict():
    print("\n=== WGCNA as dimensionality reduction -> RandomForest ===")
    df = pd.read_csv(BANANA / "badata-aromatics.csv").set_index("Sample")
    df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("aromatics", df, omics_type="volatiles")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)

    X = ds.get("aromatics")
    y = ds.sample_meta["stage"].to_numpy()

    wg = WGCNA(corr_method="spearman").fit(X, y, target_type="ordinal")
    mods = wg.modules()
    check(len(mods) == X.shape[1], f"module assignment covers all {X.shape[1]} features")

    reduced = wg.reduce(strategy="eigengenes_and_hubs")
    check(reduced.shape[0] == X.shape[0], "reduced matrix keeps all samples")
    check(reduced.shape[1] <= X.shape[1], "reduced matrix has <= original feature count")
    print(f"  reduced: {X.shape[1]} features -> {reduced.shape[1]} columns "
          f"({list(reduced.columns)[:4]}{'...' if reduced.shape[1] > 4 else ''})")

    if reduced.shape[1] >= 2:
        groups = np.arange(len(y))
        rf = RandomForest(n_estimators=100).fit(reduced, y, target_type="nominal")
        cv = rf.cross_validate(reduced, y, groups=groups, target_type="nominal")
        check(0.0 <= cv["accuracy"] <= 1.0,
              f"reduce->predict: RF on WGCNA factors, grouped-CV acc {cv['accuracy']:.3f}")
    else:
        print("  (too few modules at n=12 to chain into RF; reduction still produced)")


def main():
    test_rf_classification_banana()
    test_rf_regression_synthetic()
    test_rf_missingness_gate()
    test_splsda_banana()
    test_diablo_banana()
    test_wgcna_reduce_then_predict()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed:")
        for f in _failures:
            print("   -", f)
        sys.exit(1)
    print(f"{PASS} all method smoke checks passed")


if __name__ == "__main__":
    main()

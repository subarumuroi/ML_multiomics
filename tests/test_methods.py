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
from ml_multiomics.methods import RandomForest, SparsePLSDA, NMF, PCA, Lasso, ElasticNet, Ordinal
from ml_multiomics.preprocessing import Profile

BANANA = ROOT / "data"

PASS, FAIL = "[PASS]", "[FAIL]"
_failures = []


def check(cond, msg):
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        _failures.append(msg)
    assert cond, msg   # makes each pytest test fail meaningfully on a real error


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
    groups = np.repeat(np.arange(n_groups), per)
    # target is a GROUP-level property (constant within group) so the group-level permutation
    # is well-posed; feature 0 carries a (per-sample noisy) signal that tracks it.
    g_y = rng.normal(size=n_groups)
    y = np.repeat(g_y, per)
    X[:, 0] = y + rng.normal(scale=0.3, size=n)
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


def test_linear_models_synthetic():
    print("\n=== Lasso / ElasticNet (synthetic) ===")
    rng = np.random.default_rng(0)
    n_groups, per = 10, 2
    n = n_groups * per
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"f{i}" for i in range(12)])
    groups = np.repeat(np.arange(n_groups), per)
    y = 2.5 * X["f0"].to_numpy() - 1.5 * X["f1"].to_numpy() + rng.normal(scale=0.3, size=n)

    las = Lasso(alpha=0.1).fit(X, y, target_type="continuous")
    check(las.task_ == "regression", "Lasso auto-resolves to regression")
    coef = las.coefficients()
    check((coef["coef"] == 0).any(), "LASSO produces sparse (some zero) coefficients")
    cv = las.cross_validate(X, y, groups=groups, target_type="continuous")
    check("r2" in cv, f"Lasso grouped-CV r2 = {cv['r2']:.2f}")

    en = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y, target_type="continuous")
    cv2 = en.cross_validate(X, y, groups=groups, target_type="continuous")
    check("r2" in cv2, f"ElasticNet grouped-CV r2 = {cv2['r2']:.2f}")


def test_nmf_reduce_then_predict():
    print("\n=== NMF as dimensionality reduction -> RandomForest ===")
    df = pd.read_csv(BANANA / "badata-proteomics-imputed.csv").set_index("Sample")
    df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("proteomics", df, omics_type="proteomics")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    # NMF needs non-negative input: transform but DON'T z-score
    Preprocessor(profile=Profile(transform="log2", normalize="none")).run(ds)

    X = ds.get("proteomics")
    y = ds.sample_meta["stage"].to_numpy()

    # z-scored data would be rejected:
    try:
        NMF(n_components=5).fit(X - X.mean(), y)  # introduce negatives
        rejected = False
    except ValueError:
        rejected = True
    check(rejected, "NMF rejects negative (z-scored) input with a clear error")

    nmf = NMF(n_components=5).fit(X, y)
    scores = nmf.reduce()
    check(scores.shape == (X.shape[0], 5), "NMF reduce() -> samples x 5 factors")
    check(nmf.loadings().shape[1] == 5, "NMF loadings have 5 factors")

    groups = np.arange(len(y))
    rf = RandomForest(n_estimators=100).fit(scores, y, target_type="nominal")
    cv = rf.cross_validate(scores, y, groups=groups, target_type="nominal")
    check(0.0 <= cv["accuracy"] <= 1.0,
          f"reduce->predict: RF on NMF factors, grouped-CV acc {cv['accuracy']:.3f}")


def test_ordinal_banana():
    print("\n=== Ordinal regression (banana: Green < Ripe < Overripe) ===")
    df = pd.read_csv(BANANA / "badata-metabolomics.csv").set_index("Sample")
    df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("metabolomics", df, omics_type="metabolomics")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)

    # banana sample names use 'Over' for overripe
    X = ds.get("metabolomics")
    y = ds.sample_meta["stage"].to_numpy()
    order = ["Green", "Ripe", "Over"]
    groups = np.arange(len(y))

    ordn = Ordinal(model_type="AT", order=order).fit(X, y, target_type="ordinal")
    check(ordn._fitted, "Ordinal (mord LogisticAT) fits")
    coef = ordn.coefficients()
    check(len(coef) == X.shape[1], "ordinal coefficients cover all features")

    cv = ordn.cross_validate(X, y, groups=groups)
    check(0.0 <= cv["accuracy"] <= 1.0, f"ordinal grouped-CV accuracy {cv['accuracy']:.3f}")
    check(cv["mae"] >= 0.0, f"ordinal MAE (ordinal distance) = {cv['mae']:.3f}")


def test_xgboost_synthetic():
    print("\n=== XGBoost (synthetic; handles_missing=True) ===")
    try:
        from ml_multiomics.methods import XGBoost
        import xgboost  # noqa: F401
    except ImportError:
        print("  [SKIP] xgboost not installed (pip install ml_multiomics[xgboost])")
        return
    rng = np.random.default_rng(0)
    n_groups, per = 10, 2
    n = n_groups * per
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"f{i}" for i in range(12)])
    X.iloc[3, 1] = np.nan  # XGBoost should handle this natively
    groups = np.repeat(np.arange(n_groups), per)
    y = (2.5 * X["f0"].fillna(0).to_numpy() + rng.normal(scale=0.3, size=n))
    xgb = XGBoost().fit(X, y, target_type="continuous")
    check(xgb.task_ == "regression", "XGBoost auto-resolves to regression")
    check(xgb._fitted, "XGBoost fit succeeds with NaN present (native missing handling)")
    cv = xgb.cross_validate(X, y, groups=groups, target_type="continuous")
    check("r2" in cv, f"XGBoost grouped-CV r2 = {cv['r2']:.2f}")
    check(len(xgb.importances()) == X.shape[1], "XGBoost importances cover all features")


def test_pca_reduce_then_predict():
    print("\n=== PCA as dimensionality reduction -> RandomForest ===")
    df = pd.read_csv(BANANA / "badata-proteomics-imputed.csv").set_index("Sample")
    df = df.drop(columns=[c for c in ("Groups",) if c in df.columns])
    ds = OmicsDataset(name="banana")
    ds.add_block("proteomics", df, omics_type="proteomics")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)

    X = ds.get("proteomics")
    y = ds.sample_meta["stage"].to_numpy()
    pca = PCA(n_components=5).fit(X)
    scores = pca.reduce()
    check(scores.shape == (X.shape[0], 5), "PCA reduce() -> samples x 5 PCs")
    check(abs(pca.variance_explained()["cumulative"].iloc[-1]) <= 1.0 + 1e-9,
          "PCA cumulative variance <= 1")
    groups = np.arange(len(y))
    rf = RandomForest(n_estimators=100).fit(scores, y, target_type="nominal")
    cv = rf.cross_validate(scores, y, groups=groups, target_type="nominal")
    check(0.0 <= cv["accuracy"] <= 1.0, f"reduce->predict: RF on PCA scores, acc {cv['accuracy']:.3f}")


def main():
    test_rf_classification_banana()
    test_rf_regression_synthetic()
    test_rf_missingness_gate()
    test_splsda_banana()
    test_diablo_banana()
    test_wgcna_reduce_then_predict()
    test_linear_models_synthetic()
    test_nmf_reduce_then_predict()
    test_ordinal_banana()
    test_xgboost_synthetic()
    test_pca_reduce_then_predict()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed:")
        for f in _failures:
            print("   -", f)
        sys.exit(1)
    print(f"{PASS} all method smoke checks passed")


if __name__ == "__main__":
    main()

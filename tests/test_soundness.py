"""
test_soundness.py
=================
R-free tests for the soundness/self-documentation rework:
  * AnalysisSpec validation rejects ambiguous setups
  * FittablePreprocessor learns params on train only (no leakage) + upstream-state skip
  * block-imbalance detection / balancing
  * validation helpers (permutation_significance, bootstrap_stability, overfit_flag)
  * EVERY method self-documents (describe/assumptions/divergences) -- enforced
  * ProvenanceTrail records every data mutation
  * systematic_assessment smoke (native methods only)
  * standard-analysis replication (DE aggregates to independent units)

R-backed paths (DIABLO regression, WGCNA, integration_assessment) are exercised
elsewhere (the reports + R-guarded tests), keeping this suite R-free and fast.
"""

import numpy as np
import pandas as pd
import pytest

from ml_multiomics.core import OmicsDataset, AnalysisSpec
from ml_multiomics.core.provenance import ProvenanceTrail
from ml_multiomics.preprocessing import FittablePreprocessor
from ml_multiomics.methods.base import BaseMethod
from ml_multiomics.methods import (
    RandomForest, XGBoost, SparsePLSDA, Ordinal, Lasso, ElasticNet,
    DIABLO, NativeDIABLO, WGCNA, NativeWGCNA, NMF, PCA,
)
from ml_multiomics.analysis import (
    detect_oversized_blocks, reduce_block, balance_blocks,
    systematic_assessment, differential_expression,
)
from ml_multiomics.validation import (
    permutation_significance, bootstrap_stability, overfit_flag,
)

ALL_METHODS = [RandomForest, XGBoost, SparsePLSDA, Ordinal, Lasso, ElasticNet,
               DIABLO, NativeDIABLO, WGCNA, NativeWGCNA, NMF, PCA]


# --- AnalysisSpec ---------------------------------------------------------
def _toy_ds(n=12, seed=0):
    rng = np.random.RandomState(seed)
    idx = [f"s{i}" for i in range(n)]
    ds = OmicsDataset("toy")
    ds.add_block("prot", pd.DataFrame(rng.rand(n, 8) + 1, index=idx,
                                      columns=[f"P{i}" for i in range(8)]), omics_type="proteomics")
    ds.add_block("met", pd.DataFrame(rng.rand(n, 4) + 1, index=idx,
                                     columns=[f"M{i}" for i in range(4)]), omics_type="metabolomics")
    ds.set_sample_metadata(pd.DataFrame(
        {"unit": [f"u{i//2}" for i in range(n)], "y": rng.rand(n)}, index=idx))
    return ds


def test_spec_valid_and_describe():
    ds = _toy_ds()
    spec = AnalysisSpec(grouping_column="unit", roles={"prot": "predictor", "met": "predictor"},
                        target_type="continuous", target_column="y").validate(ds)
    assert "prot" in spec.predictor_layers()
    assert "unit" in spec.describe()


@pytest.mark.parametrize("kw,msg", [
    (dict(grouping_column="nope", roles={"prot": "predictor", "met": "predictor"},
          target_type="continuous", target_column="y"), "grouping_column"),
    (dict(grouping_column="unit", roles={"prot": "predictor"},
          target_type="continuous", target_column="y"), "every layer"),
    (dict(grouping_column="unit", roles={"prot": "predictor", "met": "predictor"},
          target_type="continuous"), "exactly one"),
    (dict(grouping_column="unit", roles={"prot": "predictor", "met": "predictor"},
          target_type="ordinal", target_column="y"), "ordinal"),
])
def test_spec_rejects_ambiguous(kw, msg):
    ds = _toy_ds()
    with pytest.raises(ValueError) as e:
        AnalysisSpec(**kw).validate(ds)
    assert msg in str(e.value)


# --- FittablePreprocessor (leakage-free + upstream state) -----------------
def test_preprocessor_train_test_isolation():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.rand(20, 6) * 100 + 10, columns=[f"f{i}" for i in range(6)])
    tr, te = X.iloc[:14], X.iloc[14:]
    pp = FittablePreprocessor(omics_type="proteomics", impute="metaboanalyst")
    pp.fit(tr)
    # fitting on a different split must give different z-score params (no global state)
    pp2 = FittablePreprocessor(omics_type="proteomics", impute="metaboanalyst").fit(te)
    assert not np.allclose(pp.zscore_mean_.values, pp2.zscore_mean_.values)
    assert pp.transform(te).isna().to_numpy().sum() == 0  # test imputed from train fills
    assert len(pp.provenance) >= 3


def test_preprocessor_skips_already_transformed():
    rng = np.random.RandomState(2)
    X = pd.DataFrame(rng.rand(10, 5) * 15, columns=[f"f{i}" for i in range(5)])  # already log-scale
    pp = FittablePreprocessor(omics_type="proteomics", impute="metaboanalyst",
                              input_state={"transform": "log2", "imputed": True})
    pp.fit_transform(X)
    assert any("SKIPPED" in n for n in pp.upstream_notes_)
    assert any("pre-imputed" in n for n in pp.upstream_notes_)


# --- block imbalance ------------------------------------------------------
def test_detect_and_balance():
    rng = np.random.RandomState(3)
    idx = [f"s{i}" for i in range(12)]
    ds = OmicsDataset("b")
    ds.add_block("big", pd.DataFrame(rng.rand(12, 300) + 1, index=idx), omics_type="proteomics")
    ds.add_block("small", pd.DataFrame(rng.rand(12, 10) + 1, index=idx), omics_type="metabolomics")
    ds.set_sample_metadata(pd.DataFrame({"u": idx}, index=idx))
    assert detect_oversized_blocks(ds, min_features=200, ratio=5.0) == ["big"]
    scores, prov = reduce_block(ds.get("big"), PCA(n_components=4))
    assert scores.shape == (12, 4) and prov["n_factors"] == 4 and prov["members"]
    bal = balance_blocks(ds, {"big": scores})
    assert bal.blocks["big"].shape[1] == 4 and bal.blocks["small"].shape[1] == 10


# --- validation helpers ---------------------------------------------------
def test_validation_helpers():
    of = overfit_flag(0.95, 0.10)
    assert of["overfit"] is True
    of2 = overfit_flag(0.5, 0.45)
    assert of2["overfit"] is False
    # bootstrap_stability: a feature always selected -> frequency 1.0
    groups = np.repeat(np.arange(6), 2)
    df = bootstrap_stability(lambda rows: ["always", "x" if rows[0] % 2 else "y"],
                             groups, n_bootstrap=10, seed=0)
    assert float(df.loc[df.feature == "always", "selection_frequency"].iloc[0]) == 1.0


# --- self-documentation enforcement (the key contract) --------------------
@pytest.mark.parametrize("cls", ALL_METHODS)
def test_every_method_self_documents(cls):
    m = cls()
    for api in ("describe", "assumptions", "divergences"):
        assert getattr(type(m), api) is not getattr(BaseMethod, api), \
            f"{cls.__name__} must override {api}()"
    ctx = {"target_type": "continuous", "n_groups": 6, "n_features": 4000,
           "missing_frac": 0.5, "grouping_has_repeats": True,
           "block_sizes": {"a": 4000, "b": 40}, "is_multiblock": True, "representation": "naive"}
    assert m.describe().strip()
    assert isinstance(m.assumptions(), list) and m.assumptions()
    assert isinstance(m.divergences(ctx), list)
    rc = m.report_card(ctx)
    assert set(rc) >= {"method", "describe", "params", "assumptions", "divergences"}


def test_provenance_records_mutations():
    tr = ProvenanceTrail("t")
    tr.record("a", {"k": 1}, in_obj=pd.DataFrame(np.zeros((3, 2))), out_obj=pd.DataFrame(np.zeros((3, 1))))
    assert len(tr) == 1 and tr.to_records()[0]["out_shape"] == [3, 1]
    assert "a" in tr.to_markdown()


# --- engine smoke (native methods, tiny) ----------------------------------
def test_systematic_assessment_smoke():
    rng = np.random.RandomState(4)
    n = 12
    idx = [f"s{i}" for i in range(n)]
    grp = [f"u{i//2}" for i in range(n)]
    prot = pd.DataFrame(rng.rand(n, 20) + 1, index=idx, columns=[f"P{i}" for i in range(20)])
    y = 3 * prot["P0"] + rng.rand(n) * 0.3
    ds = OmicsDataset("t")
    ds.add_block("prot", prot, omics_type="proteomics")
    ds.set_sample_metadata(pd.DataFrame({"unit": grp, "y": y.values}, index=idx))
    spec = AnalysisSpec(grouping_column="unit", roles={"prot": "predictor"},
                        target_type="continuous", target_column="y")
    out = systematic_assessment(ds, spec, reducers=(), n_permutations=9, stability_bootstrap=4, seed=0)
    assert out["setup"]["task"] == "regression"
    assert len(out["panel"]) >= 2
    row = next(r for r in out["panel"] if "error" not in r)
    assert "report_card" in row and "permutation" in row and "overfit" in row


def test_systematic_assessment_ordinal():
    # ordinal target must be integer-encoded so the Ordinal model + classifiers agree
    rng = np.random.RandomState(7)
    order = ["Green", "Ripe", "Over"]
    idx = [f"{s}{i}" for s in order for i in range(3)]      # 9 fruit, each its own unit
    stage = [s for s in order for _ in range(3)]
    X = pd.DataFrame(rng.rand(9, 15) + 1, index=idx, columns=[f"p{i}" for i in range(15)])
    X["p0"] = X["p0"] + np.repeat([0, 5, 10], 3)            # planted stage signal
    ds = OmicsDataset("ord")
    ds.add_block("prot", X, omics_type="proteomics")
    meta = pd.DataFrame({"stage": stage, "unit": idx}, index=idx)
    ds.set_sample_metadata(meta)
    spec = AnalysisSpec(grouping_column="unit", roles={"prot": "predictor"},
                        target_type="ordinal", target_column="stage", ordinal_order=order)
    out = systematic_assessment(ds, spec, reducers=(), n_permutations=9, stability_bootstrap=4, seed=0)
    methods = {r["method"] for r in out["panel"] if "error" not in r}
    assert "Ordinal" in methods                              # ordinal model ran without a metric clash
    for r in out["panel"]:
        assert "error" not in r, f"{r.get('approach')}: {r.get('error')}"


def test_systematic_assessment_repeatable():
    # same seed -> identical results (scientific reproducibility)
    rng = np.random.RandomState(11)
    n = 12
    idx = [f"s{i}" for i in range(n)]
    grp = [f"u{i//2}" for i in range(n)]
    prot = pd.DataFrame(rng.rand(n, 30) + 1, index=idx, columns=[f"P{i}" for i in range(30)])
    y = 3 * prot["P0"] + rng.rand(n) * 0.3
    ds = OmicsDataset("rep")
    ds.add_block("prot", prot, omics_type="proteomics")
    ds.set_sample_metadata(pd.DataFrame({"unit": grp, "y": y.values}, index=idx))
    spec = AnalysisSpec(grouping_column="unit", roles={"prot": "predictor"},
                        target_type="continuous", target_column="y")
    kw = dict(reducers=(), n_permutations=19, stability_bootstrap=8, seed=0)
    a = systematic_assessment(ds, spec, **kw)
    b = systematic_assessment(ds, spec, **kw)
    pa = {r["approach"]: (r.get("cv_score"), r["permutation"].get("p_value"), r.get("n_stable"))
          for r in a["panel"] if "error" not in r}
    pb = {r["approach"]: (r.get("cv_score"), r["permutation"].get("p_value"), r.get("n_stable"))
          for r in b["panel"] if "error" not in r}
    assert pa == pb, "same seed must give identical scores / p-values / stability"
    assert list(a["consensus"]["feature"]) == list(b["consensus"]["feature"])


def test_standard_de_aggregates_to_units():
    rng = np.random.RandomState(5)
    units = [f"{s}{i}" for s in ["G", "R"] for i in range(3)]
    stage = [s for s in ["Green", "Ripe"] for i in range(3)]
    X = pd.DataFrame(rng.rand(6, 10) * 100 + 5, index=units, columns=[f"p{i}" for i in range(10)])
    de = differential_expression(X, unit_labels=units, condition_labels=stage, logx=True)
    assert de["n_units"] == 6
    assert set(de["volcano"].columns) >= {"contrast", "feature", "log2fc", "qvalue"}


def test_figures_render_from_assessment_result():
    """Every figure binds to the engine's result keys and returns a Figure (no recompute)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from ml_multiomics.analysis import figures

    rng = np.random.RandomState(0)
    n = 16
    idx = [f"s{i}" for i in range(n)]
    grp = [f"u{i // 2}" for i in range(n)]
    prot = pd.DataFrame(rng.rand(n, 260) + 1, index=idx, columns=[f"P{i}" for i in range(260)])
    y = 3 * prot["P0"] + rng.rand(n) * 0.3
    ds = OmicsDataset("fig")
    ds.add_block("prot", prot, omics_type="proteomics")
    ds.add_block("met", pd.DataFrame(rng.rand(n, 12) + 1, index=idx, columns=[f"M{i}" for i in range(12)]),
                 omics_type="metabolomics")
    ds.set_sample_metadata(pd.DataFrame({"unit": grp, "y": y.values}, index=idx))
    spec = AnalysisSpec(grouping_column="unit", roles={"prot": "predictor", "met": "predictor"},
                        target_type="continuous", target_column="y", min_obs_frac=0.5)
    out = systematic_assessment(ds, spec, reducers=("pca",), n_permutations=9,
                                stability_bootstrap=4, seed=0)

    # oversized detection feeds the block-size figure
    assert out["setup"]["oversized_blocks"] == ["prot"]
    assert isinstance(figures.block_sizes(out["setup"]["block_sizes"],
                                          out["setup"]["oversized_blocks"]), Figure)
    # PCA scores scatter (the FIG1 full-data PCA payload)
    assert out["pca_scores"] is not None
    assert isinstance(figures.pca_scores(out["pca_scores"]), Figure)
    # permutation null kept on the panel rows
    perm_row = next(r for r in out["panel"] if r.get("permutation", {}).get("null"))
    assert isinstance(figures.permutation_hist(perm_row), Figure)
    # winning tree model + its importances
    btr = figures.best_tree_row(out["panel"])
    assert btr is not None and btr["importances"]
    assert isinstance(figures.tree_importances(btr), Figure)
    # cross-method consensus
    assert isinstance(figures.consensus_bar(out["consensus"]), Figure)
    # graceful on empty inputs (no crash, returns None)
    assert figures.stability_bar([]) is None
    assert figures.naive_vs_reduced({}) is None

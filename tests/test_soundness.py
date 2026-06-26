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
    DIABLO, WGCNA, NMF, PCA,
)
from ml_multiomics.analysis import (
    detect_oversized_blocks, reduce_block, balance_blocks,
    systematic_assessment, differential_expression,
)
from ml_multiomics.validation import (
    permutation_significance, bootstrap_stability, overfit_flag,
)

ALL_METHODS = [RandomForest, XGBoost, SparsePLSDA, Ordinal, Lasso, ElasticNet,
               DIABLO, WGCNA, NMF, PCA]


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
    # 2 rows/unit (repeats) but the target is a UNIT-level property (constant within unit),
    # as group-level permutation requires; the signal lives at the unit level.
    y = (3 * prot["P0"] + rng.rand(n) * 0.3).groupby(pd.Series(grp, index=idx)).transform("mean")
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
    y = (3 * prot["P0"] + rng.rand(n) * 0.3).groupby(pd.Series(grp, index=idx)).transform("mean")
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


def test_univariate_association_and_bridge():
    """The continuous-target standard screen finds planted associations, and the standard->ML
    bridge classifies agree / univariate-only / ML-only correctly."""
    from ml_multiomics.analysis import univariate_association, standard_to_ml_bridge
    rng = np.random.RandomState(1)
    n = 28
    idx = [f"u{i}" for i in range(n)]
    X = pd.DataFrame(rng.rand(n, 60) + 1, index=idx, columns=[f"P{i}" for i in range(60)])
    y = pd.Series(4 * X["P0"] + 2 * X["P1"] + rng.rand(n) * 0.3, index=idx)   # P0, P1 truly associated
    a = univariate_association(X, y, min_obs_frac=0.5, method="spearman")
    assert set(a.columns) >= {"feature", "rho", "pvalue", "qvalue", "n"}
    assert {"P0", "P1"} <= set(a.head(5)["feature"])               # planted signal ranks at the top
    # qualify and bridge against a consensus that shares P0 but not P1
    a["feature"] = "blk__" + a["feature"].astype(str)
    cons = pd.DataFrame({"feature": ["blk__P0", "blk__P40"], "n_approaches_stable": [3, 2]})
    b = standard_to_ml_bridge(a, cons, q_cutoff=0.1, ml_min_approaches=2)
    assert "blk__P0" in b["agreed"]                                 # univariate AND ML
    assert "blk__P40" in b["ml_only"]                               # ML, no univariate signal
    assert b["n_agreed"] + b["n_univariate_only"] == b["n_univariate"]


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
    y = (3 * prot["P0"] + rng.rand(n) * 0.3).groupby(pd.Series(grp, index=idx)).transform("mean")
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


# --- contract fixes (from external review) --------------------------------
def test_transform_dispatch_log2p1_and_unknown_raises():
    """log2p1 is a real alias for log2(x+1); an unknown transform RAISES (no silent no-op)."""
    from ml_multiomics.preprocessing.pipeline import Preprocessor, Profile
    X = pd.DataFrame({"a": [1.0, 3.0, 7.0]}, index=["s0", "s1", "s2"])
    ds = OmicsDataset("t"); ds.add_block("p", X.copy(), omics_type="proteomics")
    Preprocessor(profile=Profile(transform="log2p1", normalize="none", variance_min=None)).run(ds)
    assert np.allclose(ds.get("p")["a"].to_numpy(), np.log2(X["a"].to_numpy() + 1))
    ds2 = OmicsDataset("t2"); ds2.add_block("p", X.copy(), omics_type="proteomics")
    with pytest.raises(ValueError, match="unknown transform"):
        Preprocessor(profile=Profile(transform="bogus", normalize="none")).run(ds2)


def test_preprocessor_run_is_idempotent():
    """A second run() must NOT double-transform/double-z-score (the 'scaled once' contract)."""
    from ml_multiomics.preprocessing.pipeline import Preprocessor
    rng = np.random.RandomState(0)
    X = pd.DataFrame(np.abs(rng.randn(6, 4)) + 1, index=[f"s{i}" for i in range(6)], columns=list("abcd"))
    ds = OmicsDataset("t"); ds.add_block("p", X.copy(), omics_type="proteomics")
    Preprocessor().run(ds); once = ds.get("p").copy()
    Preprocessor().run(ds)                                    # no-op (already transformed/normalized)
    pd.testing.assert_frame_equal(ds.get("p"), once)


def test_spec_rejects_covariate_and_unknown_transform():
    """covariate role is rejected until implemented; an unknown transform name is rejected up front."""
    ds = _toy_ds()
    with pytest.raises(ValueError, match="covariate"):
        AnalysisSpec(grouping_column="unit", roles={"prot": "predictor", "met": "covariate"},
                     target_type="continuous", target_column="y").validate(ds)
    with pytest.raises(ValueError, match="transform"):
        AnalysisSpec(grouping_column="unit", roles={"prot": "predictor", "met": "predictor"},
                     target_type="continuous", target_column="y",
                     transforms={"prot": "bogus"}).validate(ds)


def test_omics_pipeline_one_call():
    """OmicsPipeline(ds, spec).run() returns setup + standard + bridge + systematic in one call."""
    from ml_multiomics import OmicsPipeline
    rng = np.random.RandomState(3)
    n = 14
    idx = [f"s{i}" for i in range(n)]
    X = pd.DataFrame(rng.rand(n, 40) + 1, index=idx, columns=[f"P{i}" for i in range(40)])
    y = pd.Series(3 * X["P0"].values, index=idx)        # one row/group, well-posed
    ds = OmicsDataset("pipe")
    ds.add_block("prot", X, omics_type="proteomics")
    ds.set_sample_metadata(pd.DataFrame({"unit": idx, "y": y.values}, index=idx))
    spec = AnalysisSpec(grouping_column="unit", roles={"prot": "predictor"},
                        target_type="continuous", target_column="y")
    out = OmicsPipeline(ds, spec, reducers=(), n_permutations=9, stability_bootstrap=4, seed=0).run()
    assert set(out) >= {"spec", "setup", "systematic", "standard", "bridge"}
    assert out["setup"]["n_groups"] == n
    assert len(out["systematic"]["panel"]) > 0
    assert {"n_agreed", "n_univariate", "n_ml"} <= set(out["bridge"])    # bridge populated


def test_permutation_requires_constant_label_per_group():
    """Group-level permutation must reject a group carrying >1 distinct target value."""
    from ml_multiomics.validation.resampling import permutation_resolution, grouped_permutation_test
    g = np.array([0, 0, 1, 1]); y_bad = np.array([1.0, 2.0, 3.0, 3.0])   # group 0 inconsistent
    with pytest.raises(ValueError, match="distinct"):
        permutation_resolution(g, y_bad)
    with pytest.raises(ValueError, match="distinct"):
        grouped_permutation_test(lambda yy: 0.0, g, y_bad, n_permutations=3)
    r = permutation_resolution(g, np.array([1.0, 1.0, 3.0, 3.0]))        # constant within group -> OK
    assert r["n_groups"] == 2

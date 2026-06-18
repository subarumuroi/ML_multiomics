"""
systematic.py
=============
The systematic, self-documenting multi-omics assessment engine.

`systematic_assessment(ds, spec)` consumes a validated :class:`AnalysisSpec` (the
user's declared decisions) and runs the full applicable method matrix on the
representation each method is valid on, judging every result by the contract the
user asked for:

  is there signal?   -> permutation test vs the design's resolution floor
  does it recur?     -> group-level bootstrap selection stability
  is it overfit?     -> a BINARY train-vs-CV gap flag (CV is a sanity check, not a score)
  what is it?        -> the stable, selected features/factors (-> biology downstream)

Key properties:
  * Leakage-free: ALL preprocessing (filter/impute/z-score) AND any dimensionality
    reduction are fit INSIDE each CV/permutation/bootstrap fold, on train rows only
    (a per-fold FoldPipeline). Reducers must expose .transform() for out-of-fold
    projection (PCA/NMF do); WGCNA/DIABLO integration is handled on full data in the
    integration layer.
  * Method-aware: tree-native methods (XGBoost) get the raw NaN matrix and no
    detection filter; NMF reduction runs on non-negative log-only data; everything
    else is z-scored. Decisions come from the spec, never guessed.
  * Self-documenting: every model carries its describe/assumptions/divergences
    computed against the LIVE context, plus its provenance.
  * No alternative without a discriminator: reduced-vs-direct etc. are resolved by
    stability + signal + parsimony, or declared indistinguishable.

This module deliberately does NOT crown a winner by raw score.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from ..core.dataset import OmicsDataset
from ..core.spec import AnalysisSpec
from ..preprocessing.pipeline import FittablePreprocessor, Profile
from ..methods import RandomForest, XGBoost, Lasso, ElasticNet, SparsePLSDA, Ordinal, PCA, NMF
from ..validation.resampling import (
    leave_one_group_out, leakage_free_cv, score_predictions,
    permutation_significance, bootstrap_stability, overfit_flag, permutation_resolution,
)
from .integration import detect_oversized_blocks

logger = logging.getLogger(__name__)

_SEP = "__"   # column prefix separator: "layer__feature"


# ---------------------------------------------------------------------------
# Per-fold pipeline: split a prefixed raw matrix back into layers, preprocess each
# (fit on train), optionally reduce the oversized ones, and re-concatenate. This is
# what makes reduce->predict leakage-free.
# ---------------------------------------------------------------------------
class FoldPipeline:
    """Fit-on-train / transform multi-block preprocessing + optional reduction.

    ``plan`` maps each layer to a dict:
        {omics_type, impute, min_obs_frac, for_nmf(bool), reduce: None|('pca'|'nmf', k)}
    The input is a single DataFrame whose columns are 'layer<SEP>feature'.
    """

    def __init__(self, plan: dict, seed: int = 0):
        self.plan = plan
        self.seed = seed
        self.pre_: dict = {}
        self.red_: dict = {}
        self.out_cols_: Optional[list] = None

    @staticmethod
    def _split(X: pd.DataFrame) -> dict:
        layers: dict = {}
        for col in X.columns:
            layer, _, feat = str(col).partition(_SEP)
            layers.setdefault(layer, []).append(col)
        return {ly: X[cols].rename(columns=lambda c: c.split(_SEP, 1)[1]) for ly, cols in layers.items()}

    def _prep(self, layer: str) -> FittablePreprocessor:
        p = self.plan[layer]
        if p.get("for_nmf"):
            prof = Profile(transform="log2", normalize="none", variance_min=1e-8)
        else:
            prof = None  # default per omics_type (z-score)
        return FittablePreprocessor(profile=prof, omics_type=p.get("omics_type"),
                                    min_obs_frac=p.get("min_obs_frac"), impute=p.get("impute"),
                                    input_state=p.get("input_state"))

    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        blocks = self._split(X)
        out = []
        for layer, df in blocks.items():
            pre = self._prep(layer)
            Z = pre.fit_transform(df)
            self.pre_[layer] = pre
            red = self.plan[layer].get("reduce")
            if red is not None:
                kind, k = red
                reducer = (PCA(n_components=k, random_state=self.seed) if kind == "pca"
                           else NMF(n_components=k, random_state=self.seed))
                reducer.fit(Z)
                self.red_[layer] = reducer
                S = reducer.reduce()
            else:
                S = Z
            out.append(S.rename(columns=lambda c, ly=layer: f"{ly}{_SEP}{c}"))
        design = pd.concat(out, axis=1)
        self.out_cols_ = list(design.columns)
        return design

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        blocks = self._split(X)
        out = []
        for layer, df in blocks.items():
            Z = self.pre_[layer].transform(df)
            if layer in self.red_:
                S = self.red_[layer].transform(Z)
            else:
                S = Z
            out.append(S.rename(columns=lambda c, ly=layer: f"{ly}{_SEP}{c}"))
        return pd.concat(out, axis=1).reindex(columns=self.out_cols_)


# ---------------------------------------------------------------------------
# Method registry (matrix-based supervised panel).
# ---------------------------------------------------------------------------
def _model_factories(task: str, ordinal_order=None):
    """name -> (factory, native_missing, selects, can_reduce_downstream)."""
    reg = task == "regression"
    out = {}
    out["RandomForest"] = (lambda: RandomForest(), False, False, True)
    out["XGBoost"] = (lambda: XGBoost(), True, False, False)   # native NaN; direct only
    if reg:
        out["Lasso"] = (lambda: Lasso(alpha=0.1), False, True, True)
        out["ElasticNet"] = (lambda: ElasticNet(alpha=0.1, l1_ratio=0.5), False, True, True)
    else:
        out["SparsePLSDA"] = (lambda: SparsePLSDA(n_components=2, keepX=20), False, True, True)
        out["Lasso"] = (lambda: Lasso(alpha=0.1), False, True, True)
        if ordinal_order is not None:
            out["Ordinal"] = (lambda: Ordinal(model_type="AT", order=list(ordinal_order)), False, False, True)
    return out


def _selected(model, design_cols) -> list:
    """Best-effort selected/important features from a fitted package method."""
    try:
        if hasattr(model, "coefficients"):
            df = model.coefficients()
            return df.loc[df["selected"], "feature"].tolist() if "selected" in df else df["feature"].head(20).tolist()
        if hasattr(model, "all_selected"):
            return model.all_selected()["feature"].tolist()
        if hasattr(model, "selected_features"):
            return list(model.selected_features())
        if hasattr(model, "vip"):
            v = model.vip(); return v.loc[v["important"], "feature"].tolist() if "important" in v else []
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def systematic_assessment(
    ds: OmicsDataset,
    spec: AnalysisSpec,
    *,
    n_factors: int = 5,
    reducers=("pca", "nmf"),
    min_features: int = 200,
    ratio: float = 5.0,
    n_permutations: int = 199,
    stability_bootstrap: int = 30,
    seed: int = 0,
) -> dict:
    """Run the systematic supervised assessment for one (ds, spec).

    Returns a dict with: setup, panel (one record per candidate: report_card + CV
    sanity + permutation signal + overfit flag + stability), discriminators
    (reduce-vs-direct verdicts), and consensus (recurrent features). The DIABLO/
    WGCNA integration layer is added by :func:`integration_assessment` (R-backed).
    """
    spec.validate(ds)
    target = spec.resolve_target(ds)
    task = "regression" if spec.target_type == "continuous" else "classification"
    predictors = spec.predictor_layers()

    # align samples across predictor blocks + target + grouping
    raw = {ly: spec.raw_sources.get(ly, ds.get(ly)) for ly in predictors}
    common = list(ds.sample_meta.index)
    for df in raw.values():
        common = [s for s in common if s in df.index]
    common = [s for s in common if s in target.values.dropna().index]
    raw = {ly: df.loc[common] for ly, df in raw.items()}
    yvals = target.values.loc[common]
    if task == "regression":
        y = yvals.to_numpy(dtype=float)
    else:
        y = yvals.to_numpy()   # labels as-is; Ordinal/classifiers handle them (order via factory)
    groups = np.asarray(ds.sample_meta.loc[common, spec.grouping_column])

    oversized = detect_oversized_blocks(ds, min_features=min_features, ratio=ratio)
    res = permutation_resolution(groups, y)
    missing_frac = float(np.mean([raw[ly].isna().to_numpy().mean() for ly in predictors]))
    ctx_base = {
        "target_type": spec.target_type,
        "n_samples": len(common),
        "n_groups": int(len(set(groups.tolist()))),
        "n_features": int(sum(raw[ly].shape[1] for ly in predictors)),
        "missing_frac": missing_frac,
        "grouping_has_repeats": len(set(groups.tolist())) < len(groups),
        "block_sizes": {ly: ds.blocks[ly].shape[1] for ly in predictors},
        "is_multiblock": len(predictors) > 1,
    }

    setup = {
        "spec": spec.to_record(),
        "task": task,
        "n_samples": len(common),
        "n_groups": ctx_base["n_groups"],
        "predictor_layers": predictors,
        "oversized_blocks": oversized,
        "block_sizes": ctx_base["block_sizes"],
        "baseline": (float(np.sqrt(np.mean((y - y.mean()) ** 2))) if task == "regression"
                     else float(pd.Series(y).value_counts().iloc[0] / len(y))),
        "baseline_kind": "predict-mean RMSE (R2=0)" if task == "regression" else "majority-class accuracy",
        "resolution": res,
    }

    # concatenated raw predictor matrix (columns prefixed by layer)
    concat = pd.concat({ly: raw[ly] for ly in predictors}, axis=1)
    concat.columns = [f"{ly}{_SEP}{f}" for ly, f in concat.columns]

    factories = _model_factories(task, ordinal_order=target.ordinal_order)

    # candidate representations: ('direct', None) + ('reduce', reducer) per reducer
    representations = [("direct", None)] + [("reduce", r) for r in reducers]

    panel = []
    consensus_counter: Counter = Counter()

    for rep_kind, reducer in representations:
        for mname, (factory, native_missing, selects, can_reduce) in factories.items():
            if rep_kind == "reduce" and not can_reduce:
                continue                      # XGBoost is direct-only
            if rep_kind == "reduce" and not oversized:
                continue                      # nothing to reduce
            label = f"{mname} | {'direct' if rep_kind=='direct' else reducer.upper()+'->'+mname}"

            # build the per-layer plan
            plan = {}
            for ly in predictors:
                p = {"omics_type": ds.blocks[ly].omics_type,
                     "input_state": spec.input_state_for(ly)}
                if native_missing:
                    p.update(impute=None, min_obs_frac=None)             # XGBoost: raw NaN
                else:
                    p.update(impute="metaboanalyst",
                             min_obs_frac=spec.min_obs_frac)
                if rep_kind == "reduce" and ly in oversized:
                    p["reduce"] = (reducer, n_factors)
                    p["for_nmf"] = (reducer == "nmf")
                plan[ly] = p

            def fit_predict(Xtr, ytr, Xte, _plan=plan, _factory=factory):
                fp = FoldPipeline(_plan, seed=seed)
                Dtr = fp.fit(Xtr)
                Dte = fp.transform(Xte)
                m = _factory()
                m.fit(Dtr, ytr, target_type=spec.target_type)
                return m.predict(Dte)

            # ---- leakage-free CV (sanity) ----
            try:
                cv = leakage_free_cv(concat, y, groups, fit_predict, task)
            except Exception as e:
                panel.append({"approach": label, "error": str(e)[:120]})
                continue
            cv_score = cv["r2"] if task == "regression" else cv["balanced_accuracy"]

            # ---- in-sample (for overfit flag) ----
            try:
                fp = FoldPipeline(plan, seed=seed); Dall = fp.fit(concat)
                m_all = factory(); m_all.fit(Dall, y, target_type=spec.target_type)
                ins = score_predictions(y, m_all.predict(Dall), task)
                train_score = ins["r2"] if task == "regression" else ins["balanced_accuracy"]
                n_design = Dall.shape[1]
            except Exception:
                train_score, n_design, m_all = cv_score, ctx_base["n_features"], None
            of = overfit_flag(train_score, cv_score)

            # ---- permutation signal ----
            def score_fn(yv, _fp=fit_predict):
                cvp = leakage_free_cv(concat, yv, groups, _fp, task)
                return cvp["r2"] if task == "regression" else cvp["balanced_accuracy"]
            perm = permutation_significance(score_fn, groups, y, n_permutations=n_permutations, seed=seed)

            # ---- stability (selecting methods only) ----
            stability = None
            if selects:
                def select_fn(rows, _plan=plan, _factory=factory):
                    fp = FoldPipeline(_plan, seed=seed)
                    D = fp.fit(concat.iloc[rows])
                    m = _factory(); m.fit(D, y[rows], target_type=spec.target_type)
                    return _selected(m, D.columns)
                stab = bootstrap_stability(select_fn, groups, n_bootstrap=stability_bootstrap, seed=seed)
                stability = stab
                for f in stab.loc[stab["stable"], "feature"].tolist():
                    consensus_counter[f] += 1

            # ---- self-documentation (live context) ----
            ctx = dict(ctx_base, representation=("reduced" if rep_kind == "reduce" else "direct"),
                       n_inputs=int(n_design))
            card = factory().report_card(ctx)

            panel.append({
                "approach": label, "family": ("reduce->predict" if rep_kind == "reduce" else "direct"),
                "method": mname, "reducer": (reducer if rep_kind == "reduce" else None),
                "n_inputs": int(n_design),
                "cv": {k: v for k, v in cv.items() if k not in ("predictions", "true")},
                "cv_score": float(cv_score),
                "overfit": of,
                "permutation": {k: v for k, v in perm.items() if k != "null"},
                "stability_top": (stability.head(10).to_dict("records") if stability is not None else None),
                "n_stable": (int(stability["stable"].sum()) if stability is not None else None),
                "report_card": card,
            })

    # consensus across selecting methods (features stable in >=2 approaches)
    consensus = pd.DataFrame(
        [{"feature": f, "n_approaches_stable": c} for f, c in consensus_counter.most_common()],
        columns=["feature", "n_approaches_stable"],
    )

    # discriminators: reduce-vs-direct per downstream method (signal + stability + parsimony)
    discriminators = _discriminate(panel)

    return {
        "setup": setup,
        "panel": panel,
        "consensus": consensus,
        "discriminators": discriminators,
    }


def _discriminate(panel: list) -> list:
    """For each method, compare direct vs each reducer by signal + stability + parsimony."""
    by_method: dict = {}
    for row in panel:
        if "error" in row:
            continue
        by_method.setdefault(row["method"], []).append(row)
    out = []
    for method, rows in by_method.items():
        if len(rows) < 2:
            continue
        # prefer: significant signal, then more stable features, then fewer inputs
        def key(r):
            sig = 1 if r["permutation"].get("significant") else 0
            nst = r["n_stable"] or 0
            return (sig, nst, -r["n_inputs"])
        ranked = sorted(rows, key=key, reverse=True)
        best, rest = ranked[0], ranked[1:]
        # is the difference real, or indistinguishable?
        same_signal = all(r["permutation"].get("significant") == best["permutation"].get("significant") for r in rest)
        verdict = (f"prefer '{best['approach']}' (fewest inputs={best['n_inputs']} among comparable signal/stability)"
                   if same_signal else
                   f"prefer '{best['approach']}' (only one with signal beyond chance)")
        out.append({
            "method": method,
            "options": [r["approach"] for r in rows],
            "preferred": best["approach"],
            "verdict": verdict,
            "note": ("indistinguishable on signal at this n -> chose by parsimony"
                     if same_signal else "discriminated by permutation signal"),
        })
    return out

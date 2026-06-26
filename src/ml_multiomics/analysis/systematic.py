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
from ..methods import DIABLO, WGCNA
from ..validation.resampling import (
    leave_one_group_out, leakage_free_cv, score_predictions,
    permutation_significance, bootstrap_stability, overfit_flag, permutation_resolution,
)
from .integration import detect_oversized_blocks, reduce_block

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
        elif p.get("transform"):
            prof = Profile(transform=p["transform"])   # spec transform override (keeps z-score + var filter)
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
    # 200 trees, not the 500 default: at n<=~30 the variance reduction past ~200 is
    # negligible and 500 dominates the permutation cost (each refit is ~2.5x slower).
    out["RandomForest"] = (lambda: RandomForest(n_estimators=200), False, False, True)
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
    max_perm_features: int = 800,
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
    elif spec.target_type == "ordinal" and target.ordinal_order:
        # integer-encode ordered classes (0..k-1): every classifier handles int labels,
        # and the Ordinal model needs them -- avoids mixing string labels with int preds
        y = target.encoded().loc[common].to_numpy()
    else:
        y = yvals.to_numpy()
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
                     "input_state": spec.input_state_for(ly),
                     "transform": spec.transform_for(ly)}      # honor per-layer transform override
                if native_missing:
                    p.update(impute=None, min_obs_frac=None)             # XGBoost: raw NaN
                else:
                    p.update(impute="metaboanalyst",
                             min_obs_frac=spec.min_obs_frac)
                if rep_kind == "reduce" and ly in oversized:
                    p["reduce"] = (reducer, n_factors)
                    p["for_nmf"] = (reducer == "nmf")
                plan[ly] = p

            # precompute y-INDEPENDENT per-fold designs ONCE (preprocess + any reduction
            # do not use the labels), then reuse them for the real CV AND every permutation,
            # so permutation only refits the model -- not the whole pipeline each time.
            try:
                fold_designs = []
                for tr, te in leave_one_group_out(groups):
                    fp = FoldPipeline(plan, seed=seed)
                    Dtr = fp.fit(concat.iloc[tr]); Dte = fp.transform(concat.iloc[te])
                    fold_designs.append((tr, te, Dtr, Dte))
            except Exception as e:
                panel.append({"approach": label, "error": str(e)[:120]})
                continue
            n_design = int(fold_designs[0][2].shape[1])

            def cv_metric(yv, _folds=fold_designs, _factory=factory):
                preds = np.empty(len(yv), dtype=object)
                for tr, te, Dtr, Dte in _folds:
                    m = _factory(); m.fit(Dtr, yv[tr], target_type=spec.target_type)
                    preds[te] = np.asarray(m.predict(Dte))
                preds = preds.astype(float) if task == "regression" else np.array(list(preds))
                return score_predictions(yv, preds, task)

            # ---- leakage-free CV (sanity) ----
            try:
                cv = cv_metric(y)
            except Exception as e:                       # a failing candidate is recorded, not fatal
                panel.append({"approach": label, "error": str(e)[:160]})
                continue
            cv_score = cv["r2"] if task == "regression" else cv["balanced_accuracy"]

            # ---- in-sample (for overfit flag) + tree-model importances (for the figure) ----
            top_imp = None
            try:
                fp = FoldPipeline(plan, seed=seed); Dall = fp.fit(concat)
                m_all = factory(); m_all.fit(Dall, y, target_type=spec.target_type)
                ins = score_predictions(y, m_all.predict(Dall), task)
                train_score = ins["r2"] if task == "regression" else ins["balanced_accuracy"]
                if hasattr(m_all, "importances"):        # RandomForest / XGBoost only
                    top_imp = m_all.importances(top_n=15).to_dict("records")
            except Exception:
                train_score = cv_score
            of = overfit_flag(train_score, cv_score)

            # ---- permutation signal (cached designs; skipped only for very large ones) ----
            if n_design <= max_perm_features:
                def score_fn(yv):
                    m = cv_metric(yv)
                    return m["r2"] if task == "regression" else m["balanced_accuracy"]
                perm = permutation_significance(score_fn, groups, y, n_permutations=n_permutations, seed=seed)
            else:
                perm = {"skipped": True, "p_value": float("nan"), "significant": None,
                        "note": (f"permutation skipped: {n_design} features (> max_perm_features="
                                 f"{max_perm_features}); too costly to refit. CV + overfit flag are "
                                 "reported, and the reduced models carry the permutation signal test.")}

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
                "permutation": {k: (list(v) if k == "null" else v) for k, v in perm.items()},
                "stability_top": (stability.head(10).to_dict("records") if stability is not None else None),
                "n_stable": (int(stability["stable"].sum()) if stability is not None else None),
                "importances": top_imp,            # tree-model top features (RF/XGB), for the figure
                "report_card": card,
            })

    # consensus across selecting methods (features stable in >=2 approaches)
    consensus = pd.DataFrame(
        [{"feature": f, "n_approaches_stable": c} for f, c in consensus_counter.most_common()],
        columns=["feature", "n_approaches_stable"],
    )

    # discriminators: reduce-vs-direct per downstream method (signal + stability + parsimony)
    discriminators = _discriminate(panel)

    # full-data PCA(2) on the oversized block (z-scored) for the scores-scatter figure
    pca_scores = None
    ob = oversized[0] if oversized else predictors[0]
    try:
        Zob = FittablePreprocessor(omics_type=ds.blocks[ob].omics_type,
                                   impute="metaboanalyst", min_obs_frac=spec.min_obs_frac).fit_transform(raw[ob])
        p2 = PCA(n_components=2, random_state=seed).fit(Zob)
        sc = p2.reduce()
        pca_scores = {"block": ob, "scores": sc, "target": pd.Series(y, index=list(sc.index)),
                      "target_type": spec.target_type,
                      "variance": p2.variance_explained().head(2).to_dict("records")}
    except Exception:
        pass

    return {
        "setup": setup,
        "panel": panel,
        "consensus": consensus,
        "discriminators": discriminators,
        "pca_scores": pca_scores,
    }


def _preprocess_block(df, omics_type, *, for_nmf=False, transform=None, impute="metaboanalyst",
                      min_obs_frac=None, input_state=None):
    """Full-data preprocess one block to a complete matrix for DIABLO (descriptive)."""
    if for_nmf:
        prof = Profile(transform="log2", normalize="none", variance_min=1e-8)
    elif transform:
        prof = Profile(transform=transform)       # spec transform override (keeps z-score + var filter)
    else:
        prof = None
    pp = FittablePreprocessor(profile=prof, omics_type=omics_type, min_obs_frac=min_obs_frac,
                              impute=impute, input_state=input_state)
    return pp.fit_transform(df)


def integration_blocks(ds: OmicsDataset, spec: AnalysisSpec, reducer: Optional[str] = None,
                       *, n_factors: int = 5, min_features: int = 200, ratio: float = 5.0,
                       seed: int = 0, rscript: str = "Rscript", group_index: int = 0):
    """Rebuild ONE integration representation's preprocessed blocks (full data).

    Mirrors exactly what ``integration_assessment`` builds per variant, so the
    blocks here ARE the blocks DIABLO was fit on -- letting a caller (e.g. the
    report's native-figure step) re-draw mixOmics plots on the same data without
    re-running the whole assessment.

    ``reducer=None`` -> the naive (raw z-scored) blocks; ``"pca"/"nmf"/"wgcna"``
    -> the oversized block(s) reduced by that reducer, others z-scored.
    Returns ``(blocks, y, oversized, membership)``; ``blocks`` is None-safe
    (raises only on a hard reducer failure -- callers guard).
    """
    spec.validate(ds)
    target = spec.resolve_target(ds)
    tt = spec.target_type
    grp = spec.integration_sets()[group_index]
    raw = {ly: spec.raw_sources.get(ly, ds.get(ly)) for ly in grp}
    common = list(ds.sample_meta.index)
    for df in raw.values():
        common = [s for s in common if s in df.index]
    common = [s for s in common if s in target.values.dropna().index]
    raw = {ly: df.loc[common] for ly, df in raw.items()}
    if tt == "continuous":
        y = target.values.loc[common].to_numpy(dtype=float)
    elif tt == "ordinal" and target.ordinal_order:
        y = target.encoded().loc[common].to_numpy()
    else:
        y = target.values.loc[common].to_numpy()
    yv = pd.Series(y, index=common)
    oversized = [ly for ly in grp if ds.blocks[ly].shape[1] > min_features and
                 ds.blocks[ly].shape[1] > ratio * np.median([ds.blocks[o].shape[1] for o in grp if o != ly] or [1])]

    blocks, membership = {}, {}
    for ly in grp:
        if reducer and ly in oversized:
            Z = _preprocess_block(raw[ly], ds.blocks[ly].omics_type, for_nmf=(reducer == "nmf"),
                                  transform=spec.transform_for(ly),
                                  impute="metaboanalyst", min_obs_frac=spec.min_obs_frac,
                                  input_state=spec.input_state_for(ly))
            red = (PCA(n_components=n_factors, random_state=seed) if reducer == "pca"
                   else NMF(n_components=n_factors, random_state=seed) if reducer == "nmf"
                   else WGCNA(rscript=rscript))
            scores, prov = reduce_block(Z, red, top_n=15)
            blocks[ly] = scores
            membership[ly] = prov
        else:
            blocks[ly] = _preprocess_block(raw[ly], ds.blocks[ly].omics_type,
                                           transform=spec.transform_for(ly),
                                           impute="metaboanalyst", min_obs_frac=spec.min_obs_frac,
                                           input_state=spec.input_state_for(ly))
    return blocks, yv, oversized, membership


def integration_assessment(
    ds: OmicsDataset,
    spec: AnalysisSpec,
    *,
    reducers=("pca", "nmf", "wgcna"),
    n_factors: int = 5,
    min_features: int = 200,
    ratio: float = 5.0,
    keepX_per_block: int = 20,
    stability_bootstrap: int = 20,
    seed: int = 0,
    rscript: str = "Rscript",
) -> dict:
    """DIABLO multi-block integration: naive (raw blocks) vs reduced (per reducer).

    For each declared integration group: fit DIABLO on the full data (descriptive --
    block correlations + selected features/modules), then bootstrap the SELECTION to
    measure recurrence. The headline contrast is naive (selects unstable individual
    proteins) vs reduced (selects stable modules/factors). Reduced selections are
    expanded back to member features via the reduction provenance (for GSEA).

    R-backed (mixOmics DIABLO; WGCNA). Requires Rscript + mixOmics/WGCNA.
    """
    spec.validate(ds)
    target = spec.resolve_target(ds)
    tt = spec.target_type
    groups_seed = spec.integration_sets()
    if not groups_seed:
        return {"groups": [], "note": "no integration_groups declared; integration skipped"}

    results = []
    for grp in groups_seed:
        # align samples across the group's blocks + target + grouping
        raw = {ly: spec.raw_sources.get(ly, ds.get(ly)) for ly in grp}
        common = list(ds.sample_meta.index)
        for df in raw.values():
            common = [s for s in common if s in df.index]
        common = [s for s in common if s in target.values.dropna().index]
        raw = {ly: df.loc[common] for ly, df in raw.items()}
        if tt == "continuous":
            y = target.values.loc[common].to_numpy(dtype=float)
        elif tt == "ordinal" and target.ordinal_order:
            y = target.encoded().loc[common].to_numpy()
        else:
            y = target.values.loc[common].to_numpy()
        groups = np.asarray(ds.sample_meta.loc[common, spec.grouping_column])
        oversized = [ly for ly in grp if ds.blocks[ly].shape[1] > min_features and
                     ds.blocks[ly].shape[1] > ratio * np.median([ds.blocks[o].shape[1] for o in grp if o != ly] or [1])]

        # WGCNA needs n >= ~15-20; below that it can stall/degenerate, so skip it as a
        # reducer here (recorded) rather than risk a long R call on too-small data.
        n_units = len(set(groups.tolist()))
        grp_reducers = list(reducers)
        wgcna_skipped = ("wgcna" in grp_reducers) and (n_units < 15)
        if wgcna_skipped:
            grp_reducers = [r for r in grp_reducers if r != "wgcna"]

        variants = {}

        # ---- naive: raw (z-scored) blocks ----
        blocks_naive = {ly: _preprocess_block(raw[ly], ds.blocks[ly].omics_type,
                                              transform=spec.transform_for(ly),
                                              impute="metaboanalyst", min_obs_frac=spec.min_obs_frac,
                                              input_state=spec.input_state_for(ly)) for ly in grp}
        variants["naive"] = {"blocks": blocks_naive, "membership": None, "reducer": None}

        # ---- reduced: one variant per reducer (a failing reducer is recorded, not fatal) ----
        for r in grp_reducers:
            try:
                balanced, membership = {}, {}
                for ly in grp:
                    if ly in oversized:
                        Z = _preprocess_block(raw[ly], ds.blocks[ly].omics_type, for_nmf=(r == "nmf"),
                                              transform=spec.transform_for(ly),
                                              impute="metaboanalyst", min_obs_frac=spec.min_obs_frac,
                                              input_state=spec.input_state_for(ly))
                        reducer = (PCA(n_components=n_factors, random_state=seed) if r == "pca"
                                   else NMF(n_components=n_factors, random_state=seed) if r == "nmf"
                                   else WGCNA(rscript=rscript))
                        scores, prov = reduce_block(Z, reducer, top_n=15)
                        balanced[ly] = scores
                        membership[ly] = prov
                    else:
                        balanced[ly] = _preprocess_block(raw[ly], ds.blocks[ly].omics_type,
                                                         transform=spec.transform_for(ly),
                                                         impute="metaboanalyst", min_obs_frac=spec.min_obs_frac,
                                                         input_state=spec.input_state_for(ly))
                variants[r] = {"blocks": balanced, "membership": membership, "reducer": r}
            except Exception as e:
                variants[r] = {"blocks": None, "membership": None, "reducer": r, "error": str(e)[:200]}

        # ---- fit DIABLO per variant + bootstrap selection stability ----
        variant_out = {}
        for name, v in variants.items():
            if v.get("blocks") is None:                       # reducer failed upstream
                variant_out[name] = {"error": v.get("error", "reduction failed"), "reducer": v.get("reducer")}
                continue
            # sparse keepX so DIABLO selects a biomarker SUBSET per block (not everything);
            # this is what makes the naive list a candidate "top features" set to test for stability
            keepX = {ly: min(keepX_per_block, df.shape[1]) for ly, df in v["blocks"].items()}
            try:
                dia = DIABLO(keepX=keepX, rscript=rscript).fit(v["blocks"], y, target_type=tt)
                block_corr = dia.block_correlations()
                selected = dia.all_selected()
            except Exception as e:
                variant_out[name] = {"error": str(e)[:160]}
                continue
            stab = None
            if stability_bootstrap and stability_bootstrap > 0:
                blocks_v = v["blocks"]
                def select_fn(rows, _b=blocks_v, _kx=keepX):
                    # bootstrap duplicates rows -> uniquify the shared index so mixOmics
                    # (which matches rownames across blocks) accepts the resample
                    uidx = [f"b{i}" for i in range(len(rows))]
                    bd = {}
                    for ly, df in _b.items():
                        sub = df.iloc[rows].copy(); sub.index = uidx; bd[ly] = sub
                    m = DIABLO(keepX=_kx, rscript=rscript).fit(bd, y[rows], target_type=tt)
                    return m.all_selected()["feature"].tolist()
                stab = bootstrap_stability(select_fn, groups, n_bootstrap=stability_bootstrap, seed=seed)
            variant_out[name] = {
                "reducer": v["reducer"],
                "block_correlations": block_corr,
                "n_selected": int(len(selected)),
                "selected": selected,
                "membership": v["membership"],
                "n_stable": (int(stab["stable"].sum()) if stab is not None else None),
                "frac_stable": (float(stab["stable"].mean()) if stab is not None and len(stab) else None),
                "stability_top": (stab.head(12).to_dict("records") if stab is not None else None),
            }

        # ---- discriminator: naive vs reduced by selection stability + parsimony ----
        verdict = _integration_verdict(variant_out)
        results.append({
            "group": grp, "oversized": oversized, "target_type": tt,
            "variants": variant_out, "discriminator": verdict,
            "wgcna_skipped_small_n": wgcna_skipped,
            "diablo_card": DIABLO().report_card({
                "target_type": tt, "n_groups": int(len(set(groups.tolist()))),
                "block_sizes": {ly: ds.blocks[ly].shape[1] for ly in grp},
                "is_multiblock": True, "representation": "naive",
                "grouping_has_repeats": len(set(groups.tolist())) < len(groups),
            }),
        })

    return {"groups": results}


def _integration_verdict(variant_out: dict, tol: float = 0.05) -> dict:
    """Prefer the variant with the most stable selection; break ties by parsimony.

    Honest wording: only call naive 'less stable' when it genuinely is (beyond tol);
    when stability ties, prefer the reduced representation for parsimony /
    interpretability (fewer, module/factor-level inputs) and say so.
    """
    scored = [(name, v) for name, v in variant_out.items()
              if "error" not in v and v.get("frac_stable") is not None]
    if not scored:
        return {"note": "no stability computed (set stability_bootstrap>0) or all variants errored"}
    best_name, best = max(scored, key=lambda kv: (kv[1]["frac_stable"], -kv[1]["n_selected"]))
    naive = variant_out.get("naive", {})
    msg = (f"prefer '{best_name}' integration (selection stability {best['frac_stable']:.0%}, "
           f"{best['n_selected']} selected)")
    if best_name != "naive" and naive.get("frac_stable") is not None:
        if naive["frac_stable"] + tol < best["frac_stable"]:
            msg += (f"; the naive raw selection is LESS stable ({naive['frac_stable']:.0%}) -- its "
                    "individual-feature list is not trustworthy, the reduced modules/factors are.")
        elif naive.get("n_selected", 0) > best["n_selected"]:
            msg += (f"; at comparable stability the naive run selects {naive['n_selected']} individual "
                    f"features vs {best['n_selected']} reduced modules/factors -- prefer the parsimonious, "
                    "interpretable representation (expandable to member features for GSEA).")
    return {"preferred": best_name, "verdict": msg}


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

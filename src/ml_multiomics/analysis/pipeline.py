"""
pipeline.py
===========
``OmicsPipeline`` -- the one-call entry point. Declare an :class:`AnalysisSpec`
over an :class:`OmicsDataset`, then ``OmicsPipeline(ds, spec).run()`` returns the
whole integrated standard -> ML result: setup, the univariate standard screen,
the systematic ML panel, the standard<->ML bridge, and (if integration_groups are
declared) the DIABLO integration.

This is a thin orchestrator over the validated pieces (it adds no new modelling) so
a user does not have to wire prepare -> standard -> systematic -> integration by
hand. The returned dict is the same structure the example reports cache and render.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.dataset import OmicsDataset
from ..core.spec import AnalysisSpec
from ..validation.resampling import permutation_resolution
from .integration import detect_oversized_blocks
from .systematic import systematic_assessment, integration_assessment
from .standard import univariate_association, differential_expression, standard_to_ml_bridge


class OmicsPipeline:
    """Run a declared multi-omics assessment end to end.

    Parameters mirror the engine knobs; defaults aim for soundness (raise
    ``n_permutations`` for a publication-grade p-value). ``reducers`` that need R
    (``"wgcna"``) are simply skipped where R is unavailable by the engine.
    """

    def __init__(self, dataset: OmicsDataset, spec: AnalysisSpec, *,
                 reducers=("pca", "nmf", "wgcna"), n_factors: int = 5,
                 n_permutations: int = 99, stability_bootstrap: int = 20,
                 seed: int = 0, rscript: str = "Rscript"):
        self.dataset = dataset
        self.spec = spec
        self.reducers = tuple(reducers)
        self.n_factors = n_factors
        self.n_permutations = n_permutations
        self.stability_bootstrap = stability_bootstrap
        self.seed = seed
        self.rscript = rscript

    # -- alignment shared by setup + standard screen -----------------------
    def _aligned(self):
        ds, spec = self.dataset, self.spec
        target = spec.resolve_target(ds)
        common = list(ds.sample_meta.index)
        for blk in spec.predictor_layers():
            df = spec.raw_sources.get(blk, ds.get(blk))
            common = [s for s in common if s in df.index]
        common = [s for s in common if s in target.values.dropna().index]
        y = target.values.loc[common]
        groups = ds.sample_meta.loc[common, spec.grouping_column].to_numpy()
        return common, y, groups, target

    def _setup(self, common, y, groups) -> dict:
        cont = self.spec.target_type == "continuous"
        yv = y.to_numpy()
        if cont:
            baseline = float(np.sqrt(np.mean((yv.astype(float) - yv.astype(float).mean()) ** 2)))
            kind = "predict-mean RMSE (R2=0)"
        else:
            baseline = float(pd.Series(yv).value_counts().iloc[0] / len(yv))
            kind = "majority-class accuracy"
        out = {"n_groups": int(len(set(groups.tolist()))), "n_samples": len(common),
               "baseline": baseline, "baseline_kind": kind,
               "oversized_blocks": detect_oversized_blocks(self.dataset)}
        try:
            out["resolution"] = permutation_resolution(groups, yv)
        except ValueError as e:        # ill-posed (target varies within group) -> surface, don't crash
            out["resolution"] = {"error": str(e)}
        return out

    # -- the standard single-feature screen (aligned to the ML target) -----
    def _standard(self, common, y) -> pd.DataFrame:
        spec = self.spec
        parts = []
        for blk in spec.predictor_layers():
            raw = spec.raw_sources.get(blk, self.dataset.get(blk)).loc[common]
            if spec.target_type == "continuous":
                a = univariate_association(raw, y, min_obs_frac=spec.min_obs_frac or 0.5)
                a = a[["feature", "rho", "pvalue", "qvalue", "n"]]
            else:                      # categorical: per-feature DE, best q across contrasts
                de = differential_expression(raw, unit_labels=common,
                                             condition_labels=y.to_numpy(), logx=True)
                v = de["volcano"].sort_values("qvalue")
                a = v.groupby("feature", as_index=False).first()[["feature", "log2fc", "pvalue", "qvalue"]]
            a.insert(0, "block", blk)
            a["feature"] = blk + "__" + a["feature"].astype(str)     # qualify -> matches ML consensus
            parts.append(a)
        return pd.concat(parts, ignore_index=True).sort_values("qvalue", na_position="last")

    def run(self, *, standard: bool = True, integration: bool = True) -> dict:
        """Execute the full assessment. Returns a dict with keys:
        ``spec``, ``setup``, ``standard`` (+ ``bridge``), ``systematic``, ``integration``.
        """
        ds, spec = self.dataset, self.spec
        spec.validate(ds)
        common, y, groups, _ = self._aligned()
        out = {"spec": spec.describe(), "setup": self._setup(common, y, groups)}
        out["systematic"] = systematic_assessment(
            ds, spec, n_factors=self.n_factors,
            reducers=tuple(r for r in self.reducers if r != "wgcna"),   # out-of-fold reducers only
            n_permutations=self.n_permutations, stability_bootstrap=self.stability_bootstrap,
            seed=self.seed)
        if standard:
            try:
                out["standard"] = self._standard(common, y)
                out["bridge"] = standard_to_ml_bridge(out["standard"], out["systematic"]["consensus"])
            except Exception as e:                                      # standard screen is a bonus, never fatal
                out["standard"] = None
                out["bridge"] = {"note": f"standard screen unavailable: {str(e)[:160]}"}
        if integration and spec.integration_sets():
            out["integration"] = integration_assessment(
                ds, spec, reducers=self.reducers, n_factors=self.n_factors,
                stability_bootstrap=self.stability_bootstrap, seed=self.seed, rscript=self.rscript)
        return out

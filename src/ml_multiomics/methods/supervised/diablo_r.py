"""
diablo_r.py
===========
DIABLO via the reference implementation — `mixOmics::block.splsda` (R) — called
through a subprocess bridge. This is the ONLY `DIABLO` -- the experimental native
Python port was removed (it did not reach parity with mixOmics), so DIABLO is R-only.

Why R: the multi-block design coupling is exactly where a reimplementation is
hardest to validate, and mixOmics is the published standard with established
tuning (`tune.block.splsda`). Depending on mixOmics (a community-standard,
citable package) does not reintroduce dependence on the lab's gatekept tooling.

Requires R + mixOmics (`pip install ml_multiomics[r]` documents this; the R
packages are installed separately). handles_missing = False → impute first.
Goal is interpretation, so this exposes variates / loadings / selected features /
block correlations; LOO CV (mixOmics perf) is available as a guardrail.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..base import BaseMethod
from ...preprocessing.imputation import impute as _impute

logger = logging.getLogger(__name__)

_RSCRIPT = Path(__file__).resolve().parents[2] / "rscripts" / "diablo.R"


class DIABLO(BaseMethod):
    handles_missing = False
    requires_target = True
    supported_targets = ("nominal", "ordinal", "continuous")

    def __init__(self, n_components: int = 2, keepX=None, design: float = 0.1,
                 keepY=None, impute: str = "metaboanalyst", rscript: str = "Rscript",
                 timeout: int = 300):
        super().__init__(impute=impute)
        self.n_components = n_components
        self.keepX = keepX          # dict {block: int|list} or None (full)
        self.design = design
        self.keepY = keepY          # int features kept from a continuous Y (block.spls)
        self.rscript = rscript
        self.timeout = timeout      # bound R hangs
        self.target_type_ = None    # set at fit/cross_validate; drives splsda vs spls
        self.block_names_ = None
        self.variates_ = None       # {block: ndarray (n x ncomp)}
        self.loadings_ = None       # {block: DataFrame (features x ncomp)}
        self.selected_ = None       # {block: {comp: [features]}}
        self.index_ = None
        self.cv_ = None

    _PARAM_KEYS = ("n_components", "keepX", "design", "keepY")

    def describe(self) -> str:
        return (
            "DIABLO (reference mixOmics block.s/plsda via R): supervised MULTI-BLOCK integration "
            "that finds latent components correlated ACROSS omics layers while discriminating the "
            "target. Reports per-block variates, loadings, selected features, and between-block "
            "correlations. Read it as: which coordinated cross-omics signature tracks the target; "
            "selected features per block are the integration's drivers (judge by stability)."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Blocks share the same samples and are on comparable (z-scored) scales.",
            "A shared low-dimensional structure links the blocks to the target.",
            "Block sizes should be roughly balanced -- a much larger block can dominate the design.",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        ctx = context or {}
        sizes = ctx.get("block_sizes") or {}
        if len(sizes) >= 2:
            mx, mn = max(sizes.values()), min(sizes.values())
            if mn > 0 and mx / mn > 5:
                out.append(
                    f"Block sizes imbalanced ({mx} vs {mn}, >5x): the larger block can dominate the "
                    "integration -> reduce it (WGCNA/PCA/NMF) before DIABLO. This is the 'naive' run "
                    "if no reduction was applied."
                    if ctx.get("representation", "naive") == "naive"
                    else f"Block sizes were imbalanced ({mx} vs {mn}); the large block was reduced "
                    "before integration to keep it from dominating."
                )
        if ctx.get("target_type") == "ordinal":
            out.append("Ordinal target uses block.splsda (classification) -- order is discarded.")
        return out

    # -- block prep -------------------------------------------------------
    def _prepare_blocks(self, blocks):
        if hasattr(blocks, "block_names") and hasattr(blocks, "common_samples"):
            ds = blocks
            common = ds.common_samples()
            bd = {nm: ds.blocks[nm].data.loc[common] for nm in ds.block_names}
        else:
            bd = {nm: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v))
                  for nm, v in blocks.items()}
            common = None
            for dfb in bd.values():
                common = dfb.index if common is None else common.intersection(dfb.index)
            bd = {nm: dfb.loc[list(common)] for nm, dfb in bd.items()}
        out = {}
        for nm, dfb in bd.items():
            if bool(dfb.isna().any().any()):
                dfb = _impute(dfb, self.impute_strategy)
            out[nm] = dfb
        index = list(next(iter(bd.values())).index)
        return out, index

    @staticmethod
    def _align_y(y, index):
        return y.reindex(index).to_numpy() if hasattr(y, "reindex") else np.asarray(y)

    def _run_r(self, bd: dict, y, index, cv: bool):
        work = Path(tempfile.mkdtemp(prefix="diablo_r_"))
        try:
            for nm, dfb in bd.items():
                dfb.to_csv(work / f"block_{nm}.csv")
            pd.DataFrame({"y": y}).to_csv(work / "y.csv", index=False)
            keepX = self.keepX
            if isinstance(keepX, dict):
                keepX = {k: (list(v) if isinstance(v, (list, tuple)) else int(v))
                         for k, v in keepX.items()}
            config = {"blocks": list(bd), "ncomp": int(self.n_components),
                      "design": float(self.design), "keepX": keepX, "cv": bool(cv),
                      "target_type": self.target_type_ or "nominal",
                      "keepY": None if self.keepY is None else int(self.keepY)}
            (work / "config.json").write_text(json.dumps(config))

            try:
                res = subprocess.run([self.rscript, str(_RSCRIPT), str(work)],
                                     capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"mixOmics DIABLO (R) timed out after {self.timeout}s")
            if res.returncode != 0:
                raise RuntimeError(f"mixOmics DIABLO (R) failed:\n{res.stderr}")

            variates, loadings, selected = {}, {}, {}
            for nm in bd:
                variates[nm] = pd.read_csv(work / f"variates_{nm}.csv").to_numpy()
                loadings[nm] = pd.read_csv(work / f"loadings_{nm}.csv", index_col=0)
                selected[nm] = {
                    k: (work / f"selected_{nm}_c{k}.txt").read_text().split("\n")
                    for k in range(1, self.n_components + 1)
                    if (work / f"selected_{nm}_c{k}.txt").exists()
                }
                selected[nm] = {k: [s for s in v if s] for k, v in selected[nm].items()}
            cv_obj = None
            if cv and (work / "cv_error.json").exists():
                cv_obj = json.loads((work / "cv_error.json").read_text())
            return variates, loadings, selected, cv_obj
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def fit(self, blocks, y, feature_names=None, target_type=None) -> "DIABLO":
        if target_type is not None:
            self._check_target(target_type)
        self.target_type_ = target_type or self.target_type_ or "nominal"
        bd, index = self._prepare_blocks(blocks)
        y = self._align_y(y, index)
        self.block_names_ = list(bd)
        self.index_ = index
        self.variates_, self.loadings_, self.selected_, _ = self._run_r(bd, y, index, cv=False)
        self._fitted = True
        return self

    # -- interpretation accessors -----------------------------------------
    def block_correlations(self) -> pd.DataFrame:
        names = self.block_names_
        K = len(names)
        M = np.eye(K)
        for i in range(K):
            for j in range(i + 1, K):
                r = np.corrcoef(self.variates_[names[i]][:, 0], self.variates_[names[j]][:, 0])[0, 1]
                M[i, j] = M[j, i] = r
        return pd.DataFrame(M, index=names, columns=names)

    def loadings(self, block: str) -> pd.DataFrame:
        return self.loadings_[block]

    def selected_features(self, block: str, comp: int = 1) -> list:
        return self.selected_.get(block, {}).get(comp, [])

    def all_selected(self) -> pd.DataFrame:
        rows = []
        for b in self.block_names_:
            for comp, feats in self.selected_.get(b, {}).items():
                for f in feats:
                    rows.append({"block": b, "component": comp, "feature": f})
        return pd.DataFrame(rows)

    def cross_validate(self, blocks, y, groups=None, target_type=None) -> dict:
        """Leave-one-out CV via mixOmics perf() (samples assumed independent).

        Classification -> error rate; regression (block.spls) -> MSEP/R2/Q2. This
        is a secondary sanity check; the engine drives leakage-free grouped CV.
        """
        self.target_type_ = target_type or self.target_type_ or "nominal"
        bd, index = self._prepare_blocks(blocks)
        y = self._align_y(y, index)
        _, _, _, cv_obj = self._run_r(bd, y, index, cv=True)
        self.cv_ = cv_obj
        metric = "regression measures (MSEP/R2/Q2)" if self.target_type_ == "continuous" else "error rate"
        return {"perf": cv_obj, "note": f"mixOmics perf(validation='loo') {metric}; "
                "samples treated as independent (correct when each sample is its own unit)"}

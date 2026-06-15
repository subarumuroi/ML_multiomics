"""
diablo_r.py
===========
DIABLO via the reference implementation — `mixOmics::block.splsda` (R) — called
through a subprocess bridge. This is the DEFAULT `DIABLO` (the native Python port
is kept as `NativeDIABLO`, experimental/unvalidated).

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
    supported_targets = ("nominal", "ordinal")

    def __init__(self, n_components: int = 2, keepX=None, design: float = 0.1,
                 impute: str = "metaboanalyst", rscript: str = "Rscript"):
        super().__init__(impute=impute)
        self.n_components = n_components
        self.keepX = keepX          # dict {block: int|list} or None (full)
        self.design = design
        self.rscript = rscript
        self.block_names_ = None
        self.variates_ = None       # {block: ndarray (n x ncomp)}
        self.loadings_ = None       # {block: DataFrame (features x ncomp)}
        self.selected_ = None       # {block: {comp: [features]}}
        self.index_ = None
        self.cv_ = None

    # -- block prep (shared shape with NativeDIABLO) -----------------------
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
                      "design": float(self.design), "keepX": keepX, "cv": bool(cv)}
            (work / "config.json").write_text(json.dumps(config))

            res = subprocess.run([self.rscript, str(_RSCRIPT), str(work)],
                                 capture_output=True, text=True)
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
        """Leave-one-out CV error via mixOmics perf() (samples assumed independent)."""
        bd, index = self._prepare_blocks(blocks)
        y = self._align_y(y, index)
        _, _, _, cv_obj = self._run_r(bd, y, index, cv=True)
        self.cv_ = cv_obj
        return {"loo_error_rate": cv_obj, "note": "mixOmics perf(validation='loo'); "
                "samples treated as independent (correct when each sample is its own unit)"}

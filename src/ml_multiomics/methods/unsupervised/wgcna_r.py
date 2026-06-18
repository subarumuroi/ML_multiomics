"""
wgcna_r.py
==========
WGCNA via the reference implementation — the R `WGCNA` package — called through a
subprocess bridge. This is the DEFAULT `WGCNA` (the native Python port is kept as
`NativeWGCNA`, experimental/unvalidated).

As a reducer it exposes `.reduce()`/`.eigengenes()` → a samples × module-eigengene
matrix (the real WGCNA dynamic-tree-cut modules + close-module merge), feeding the
reduce→predict pattern. Requires R + the WGCNA package (community-standard, not
the lab's gatekept code). handles_missing = False → impute first.

Caveat unchanged: WGCNA expects n ≥ ~15–20; at small n modules are exploratory
regardless of which implementation runs.
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

_RSCRIPT = Path(__file__).resolve().parents[2] / "rscripts" / "wgcna.R"


class WGCNA(BaseMethod):
    handles_missing = False
    requires_target = False
    supported_targets = ("nominal", "ordinal", "continuous", "none")

    def __init__(self, power=None, network_type: str = "unsigned",
                 min_module_size: int = 20, merge_cut_height: float = 0.25,
                 impute: str = "metaboanalyst", rscript: str = "Rscript"):
        super().__init__(impute=impute)
        self.power = power
        self.network_type = network_type
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height
        self.rscript = rscript
        self.modules_ = None        # DataFrame feature/module
        self.eigengenes_ = None     # DataFrame samples x modules
        self.index_ = None
        self.feature_names_ = None

    _PARAM_KEYS = ("power", "network_type", "min_module_size", "merge_cut_height")

    def describe(self) -> str:
        return (
            "WGCNA (reference R implementation): an UNSUPERVISED reducer that groups co-abundant "
            "features into modules (data-driven count via dynamic tree cut) and summarises each "
            "module by its eigengene. Used to reduce a large block to a few biologically coherent "
            "module profiles before integration. Read a module as a co-regulated feature program; "
            "the eigengene is its representative sample profile."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Co-abundance (correlation) structure reflects biology; approx. scale-free topology.",
            "Adequate sample size (WGCNA expects n >= ~15-20); at small n modules are exploratory.",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        ctx = context or {}
        ng = ctx.get("n_groups")
        if ng is not None and ng < 15:
            out.append(
                f"Only {ng} units (< WGCNA's ~15-20): module detection is exploratory regardless "
                "of implementation."
            )
        mf = ctx.get("missing_frac")
        if mf and mf > 0.2:
            out.append(
                "Correlation-based: imputed near-constant features can create spurious modules; a "
                "detection filter (min_obs_frac) is applied before WGCNA."
            )
        return out

    def fit(self, X, y=None, feature_names=None, target_type=None) -> "WGCNA":
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            self.index_ = list(Xp.index)
        else:
            Xp = pd.DataFrame(np.asarray(Xp),
                              columns=feature_names or [f"f{i}" for i in range(np.asarray(Xp).shape[1])])
            self.feature_names_ = list(Xp.columns)
            self.index_ = list(Xp.index)

        work = Path(tempfile.mkdtemp(prefix="wgcna_r_"))
        try:
            Xp.to_csv(work / "block.csv")
            cfg = {"network_type": self.network_type,
                   "min_module_size": int(self.min_module_size),
                   "merge_cut_height": float(self.merge_cut_height),
                   "power": None if self.power is None else int(self.power)}
            (work / "config.json").write_text(json.dumps(cfg))
            res = subprocess.run([self.rscript, str(_RSCRIPT), str(work)],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"R WGCNA failed:\n{res.stdout}\n{res.stderr}")
            self.modules_ = pd.read_csv(work / "modules.csv")
            self.eigengenes_ = pd.read_csv(work / "eigengenes.csv", index_col=0)
            self.eigengenes_.index = self.index_
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self._fitted = True
        return self

    def modules(self) -> pd.DataFrame:
        return self.modules_

    def eigengenes(self) -> pd.DataFrame:
        return self.eigengenes_

    def reduce(self, drop_grey: bool = True) -> pd.DataFrame:
        """Samples × module-eigengene matrix (the reduced representation).

        `drop_grey` removes the 'grey'/unassigned module (WGCNA's MEgrey),
        which is not a real co-abundance module.
        """
        me = self.eigengenes_
        if drop_grey:
            me = me[[c for c in me.columns if c.lower() not in ("megrey", "grey")]]
        return me

    def n_modules(self, exclude_grey: bool = True) -> int:
        mods = set(self.modules_["module"])
        if exclude_grey:
            mods.discard("grey")
        return len(mods)

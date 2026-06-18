"""
pca.py
======
PCA on the BaseMethod interface, as a "reducer" (same role as NMF/WGCNA/MOFA):
`.reduce()`/`.scores()` returns the samples x principal-component matrix, which
can feed a supervised method or be added back as an OmicsDataset block.

Thin wrapper over sklearn PCA (which centers internally). handles_missing=False
-> just-in-time imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as _SkPCA

from ..base import BaseMethod


class PCA(BaseMethod):
    handles_missing = False
    requires_target = False
    supported_targets = ("nominal", "ordinal", "continuous", "none")

    def __init__(self, n_components: int = 10, random_state: int = 42,
                 impute: str = "metaboanalyst"):
        super().__init__(impute=impute)
        self.n_components = n_components
        self.random_state = random_state
        self.model_ = None
        self.scores_ = None
        self.feature_names_ = None
        self.index_ = None

    _PARAM_KEYS = ("n_components", "random_state")

    def describe(self) -> str:
        return (
            "Principal component analysis: an UNSUPERVISED reducer that projects features onto "
            "orthogonal axes of maximal variance. Used to shrink a large block to a few factors "
            "(reduce->predict). Read PCs as variance directions and loadings as feature weights -- "
            "note PCs capture variance, which is not necessarily the variance related to the target."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Variance equals signal; directions are linear and orthogonal.",
            "Correlation/covariance based -- sensitive to scaling and to spurious correlation.",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        mf = (context or {}).get("missing_frac")
        if mf and mf > 0.2:
            out.append(
                "Imputed near-constant features can manufacture spurious components; a detection "
                "filter (min_obs_frac) is recommended before PCA."
            )
        return out

    def fit(self, X, y=None, feature_names=None, target_type=None) -> "PCA":
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            self.index_ = list(Xp.index)
            arr = Xp.to_numpy(dtype=float)
        else:
            arr = np.asarray(Xp, dtype=float)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(arr.shape[1])]
            self.index_ = list(range(arr.shape[0]))
        k = min(self.n_components, *arr.shape)
        self.model_ = _SkPCA(n_components=k, random_state=self.random_state)
        self.scores_ = self.model_.fit_transform(arr)
        self._fitted = True
        return self

    def scores(self) -> pd.DataFrame:
        cols = [f"PC{i + 1}" for i in range(self.scores_.shape[1])]
        return pd.DataFrame(self.scores_, index=self.index_, columns=cols)

    def reduce(self) -> pd.DataFrame:
        return self.scores()

    def transform(self, X) -> pd.DataFrame:
        arr = np.asarray(self._prepare_X(X), dtype=float)
        W = self.model_.transform(arr)
        cols = [f"PC{i + 1}" for i in range(W.shape[1])]
        idx = list(X.index) if hasattr(X, "index") else range(W.shape[0])
        return pd.DataFrame(W, index=idx, columns=cols)

    def loadings(self) -> pd.DataFrame:
        cols = [f"PC{i + 1}" for i in range(self.model_.components_.shape[0])]
        return pd.DataFrame(self.model_.components_.T, index=self.feature_names_, columns=cols)

    def variance_explained(self) -> pd.DataFrame:
        ratio = self.model_.explained_variance_ratio_
        return pd.DataFrame({
            "component": [f"PC{i + 1}" for i in range(len(ratio))],
            "variance_explained": ratio,
            "cumulative": np.cumsum(ratio),
        })

    def top_features(self, pc: int, top_n: int = 20) -> pd.DataFrame:
        load = self.model_.components_[pc - 1]
        df = pd.DataFrame({"feature": self.feature_names_, "loading": load,
                           "abs_loading": np.abs(load)})
        return df.sort_values("abs_loading", ascending=False).head(top_n).reset_index(drop=True)

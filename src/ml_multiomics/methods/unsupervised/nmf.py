"""
nmf.py
======
Non-negative Matrix Factorization on the BaseMethod interface, as a "reducer".

NMF factorizes X (n x p) ~= W (n x k) @ H (k x p). W is a non-negative, parts-based
samples x factor representation — the reduced output (`.reduce()` / `.scores()`),
analogous to PCA/MOFA factor scores. H gives factor loadings (`.loadings()`).

IMPORTANT: NMF requires NON-NEGATIVE input. z-scored data has negatives and will
be rejected. Run NMF on transformed-but-not-scaled data — e.g.
`Preprocessor(profile=Profile(transform="log2", normalize="none"))`, since
log2(x+1) of non-negative intensities is non-negative. (Set nonneg="shift" to
shift into range instead of erroring, at some cost to the parts-based
interpretation.)

handles_missing = False -> just-in-time imputation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF as _SkNMF

from ..base import BaseMethod

logger = logging.getLogger(__name__)


class NMF(BaseMethod):
    handles_missing = False
    requires_target = False
    supported_targets = ("nominal", "ordinal", "continuous", "none")

    def __init__(self, n_components: int = 10, max_iter: int = 500, random_state: int = 42,
                 init: str = "nndsvda", nonneg: str = "error", impute: str = "metaboanalyst"):
        super().__init__(impute=impute)
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
        self.nonneg = nonneg
        self.model_ = None
        self.W_ = None
        self.feature_names_ = None
        self.index_ = None

    _PARAM_KEYS = ("n_components", "max_iter", "random_state", "init", "nonneg")

    def describe(self) -> str:
        return (
            "Non-negative matrix factorization: an UNSUPERVISED reducer that decomposes the data "
            "into additive, parts-based factors (all weights >= 0), often more interpretable than "
            "PCA. Used for reduce->predict. Read factors as co-occurring feature programs and the "
            "scores (W) as each sample's loading on them."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Input is NON-NEGATIVE (intensities on a log scale), NOT z-scored.",
            "The signal decomposes into additive parts (no cancellation).",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        out.append(
            "Requires non-negative input, so it runs on log-only (un-z-scored) data -- a DIFFERENT "
            "preprocessing than the z-scored methods; not input-identical to them."
        )
        return out

    def _ensure_nonneg(self, arr: np.ndarray) -> np.ndarray:
        mn = arr.min()
        if mn >= 0:
            return arr
        if self.nonneg == "error":
            raise ValueError(
                "NMF requires non-negative input, but the data has negatives "
                f"(min={mn:.3g}). z-scored data is not valid for NMF. Preprocess "
                "with normalize='none' (log2(x+1) of intensities is non-negative), "
                "or pass nonneg='shift' / nonneg='clip'."
            )
        if self.nonneg == "shift":
            logger.warning("NMF: shifting data by %.3g to make it non-negative.", -mn)
            return arr - mn
        if self.nonneg == "clip":
            logger.warning("NMF: clipping %d negative values to 0.", int((arr < 0).sum()))
            return np.clip(arr, 0, None)
        raise ValueError(f"nonneg must be 'error'|'shift'|'clip'; got {self.nonneg!r}")

    def fit(self, X, y=None, feature_names=None, target_type=None) -> "NMF":
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            self.index_ = list(Xp.index)
            arr = Xp.to_numpy(dtype=float)
        else:
            arr = np.asarray(Xp, dtype=float)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(arr.shape[1])]
            self.index_ = list(range(arr.shape[0]))
        arr = self._ensure_nonneg(arr)
        k = min(self.n_components, arr.shape[1])
        self.model_ = _SkNMF(n_components=k, init=self.init, max_iter=self.max_iter,
                             random_state=self.random_state)
        self.W_ = self.model_.fit_transform(arr)
        self._fitted = True
        return self

    # -- reducer interface -------------------------------------------------
    def scores(self) -> pd.DataFrame:
        """Samples x factor matrix (W) — the reduced representation."""
        cols = [f"Factor{i + 1}" for i in range(self.W_.shape[1])]
        return pd.DataFrame(self.W_, index=self.index_, columns=cols)

    def reduce(self) -> pd.DataFrame:
        """Alias for scores(): the reduced samples x factor matrix."""
        return self.scores()

    def transform(self, X) -> pd.DataFrame:
        """Project new (non-negative) data onto the fitted factors."""
        arr = self._ensure_nonneg(np.asarray(self._prepare_X(X), dtype=float))
        W = self.model_.transform(arr)
        cols = [f"Factor{i + 1}" for i in range(W.shape[1])]
        idx = list(X.index) if hasattr(X, "index") else range(W.shape[0])
        return pd.DataFrame(W, index=idx, columns=cols)

    def loadings(self) -> pd.DataFrame:
        """Features x factor loadings (H^T)."""
        cols = [f"Factor{i + 1}" for i in range(self.model_.components_.shape[0])]
        return pd.DataFrame(self.model_.components_.T, index=self.feature_names_, columns=cols)

    def top_features(self, factor: int, top_n: int = 20) -> pd.DataFrame:
        """Top-loading features for a given factor (1-indexed)."""
        h = self.model_.components_[factor - 1]
        df = pd.DataFrame({"feature": self.feature_names_, "loading": h})
        return df.sort_values("loading", ascending=False).head(top_n).reset_index(drop=True)

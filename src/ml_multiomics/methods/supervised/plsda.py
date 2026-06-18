"""
plsda.py
========
Sparse PLS-DA (single-block discriminant analysis with built-in feature
selection) on the BaseMethod interface.

Ported from multiomics_integration.plsda.SPLSDA (native NIPALS + L1
soft-thresholding; Lê Cao et al. 2011). Same deliberate changes as the RF port:
grouping-aware CV / permutation / bootstrap-stability instead of sample-level.

The core only centers X (no internal scaling) — scaling is done once in the
preprocessing pipeline. handles_missing = False (impute just-in-time).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from ..base import BaseMethod
from ...validation import leave_one_group_out, grouped_permutation_test, grouped_bootstrap_indices

logger = logging.getLogger(__name__)


def _soft_threshold(w: np.ndarray, keepX) -> np.ndarray:
    """L1 sparsity: keep the top-keepX weights by |w|, renormalize."""
    if keepX is None or keepX >= len(w):
        return w
    idx = np.argsort(np.abs(w))[::-1]
    mask = np.zeros_like(w, dtype=bool)
    mask[idx[:keepX]] = True
    ws = w * mask
    nrm = np.linalg.norm(ws)
    return ws / nrm if nrm > 0 else ws


def _encode_Y(y):
    lb = LabelBinarizer()
    Y = lb.fit_transform(y).astype(float)
    if Y.shape[1] == 1:  # binary
        Y = np.hstack([1 - Y, Y])
    return Y, lb.classes_


def _splsda_fit(X, y, n_components=2, keepX=None, max_iter=500, tol=1e-6) -> dict:
    """Core NIPALS sPLS-DA. Returns fitted arrays + VIP (no scaling; centers X)."""
    Y, classes = _encode_Y(y)
    n, p = X.shape
    q = Y.shape[1]
    ncomp = min(n_components, n - 1, p)

    if keepX is None:
        kx = [p] * ncomp
    elif isinstance(keepX, int):
        kx = [keepX] * ncomp
    else:
        kx = list(keepX)
        while len(kx) < ncomp:
            kx.append(kx[-1])

    x_mean = X.mean(axis=0)
    Xk = X - x_mean
    Yk = Y - Y.mean(axis=0)

    T = np.zeros((n, ncomp)); U = np.zeros((n, ncomp))
    W = np.zeros((p, ncomp)); P = np.zeros((p, ncomp)); Q = np.zeros((q, ncomp))

    for h in range(ncomp):
        u = Yk[:, 0].copy()
        t = Xk @ np.zeros(p)
        for _ in range(max_iter):
            w = Xk.T @ u
            nw = np.linalg.norm(w)
            if nw > 0:
                w = w / nw
            w = _soft_threshold(w, kx[h])
            t = Xk @ w
            tt = t @ t
            q_h = Yk.T @ t / tt if tt > 0 else np.zeros(q)
            qq = q_h @ q_h
            u_new = Yk @ q_h / qq if qq > 0 else u
            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new
        tt = t @ t
        p_h = Xk.T @ t / tt if tt > 0 else np.zeros(p)
        T[:, h] = t; U[:, h] = u; W[:, h] = w; P[:, h] = p_h; Q[:, h] = q_h
        Xk = Xk - np.outer(t, p_h)
        Yk = Yk - np.outer(t, q_h)

    SS = np.array([(T[:, h] @ T[:, h]) * (Q[:, h] @ Q[:, h]) for h in range(ncomp)])
    total = SS.sum()
    vip = np.ones(p) if total == 0 else np.sqrt(p * ((W ** 2) @ SS) / total)

    return {"classes": classes, "x_mean": x_mean, "W": W, "P": P, "Q": Q,
            "T": T, "ncomp": ncomp, "vip": vip}


def _splsda_predict(fit: dict, X: np.ndarray) -> np.ndarray:
    Xc = X - fit["x_mean"]
    PtW = fit["P"].T @ fit["W"]
    try:
        R = fit["W"] @ np.linalg.inv(PtW)
    except np.linalg.LinAlgError:
        R = fit["W"] @ np.linalg.pinv(PtW)
    Y_hat = (Xc @ R) @ fit["Q"].T
    return fit["classes"][np.argmax(Y_hat, axis=1)]


class SparsePLSDA(BaseMethod):
    handles_missing = False
    requires_target = True
    supported_targets = ("nominal", "ordinal")

    def __init__(self, n_components: int = 2, keepX=None, max_iter: int = 500,
                 tol: float = 1e-6, impute: str = "metaboanalyst"):
        super().__init__(impute=impute)
        self.n_components = n_components
        self.keepX = keepX
        self.max_iter = max_iter
        self.tol = tol
        self.fit_ = None
        self.feature_names_ = None

    _PARAM_KEYS = ("n_components", "keepX", "max_iter", "tol")

    def describe(self) -> str:
        return (
            "Sparse PLS discriminant analysis: supervised latent components that separate classes, "
            "with per-component feature selection (keepX). Reports VIP / selected features and, via "
            "stability_selection(), a bootstrap selection frequency. Read the components as the "
            "class-separating axes and prefer features with high selection frequency."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Class structure is captured by a few linear latent components.",
            "Sparse selection (keepX) picks one of a set of correlated features -- use the "
            "stability frequency, not a single run, to judge a feature.",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        ctx = context or {}
        if ctx.get("target_type") == "ordinal":
            out.append("Ordinal target treated as unordered classes -- the ordering is not used.")
        return out

    def fit(self, X, y, feature_names=None, target_type=None) -> "SparsePLSDA":
        if y is None:
            raise ValueError("SparsePLSDA requires a target y.")
        if target_type is not None:
            self._check_target(target_type)
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            arr = Xp.to_numpy()
        else:
            arr = np.asarray(Xp)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(arr.shape[1])]
        self.fit_ = _splsda_fit(arr, np.asarray(y), self.n_components, self.keepX,
                                self.max_iter, self.tol)
        self._fitted = True
        return self

    def predict(self, X):
        return _splsda_predict(self.fit_, np.asarray(self._prepare_X(X)))

    def vip(self, top_n=None) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "vip": self.fit_["vip"],
            "important": self.fit_["vip"] >= 1.0,
        }).sort_values("vip", ascending=False).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    # -- grouping-aware validation ----------------------------------------
    def _cv_predict(self, arr, y, groups):
        preds = np.empty(len(y), dtype=object)
        for tr, te in leave_one_group_out(groups):
            f = _splsda_fit(arr[tr], y[tr], self.n_components, self.keepX,
                            self.max_iter, self.tol)
            preds[te] = _splsda_predict(f, arr[te])
        return preds

    def cross_validate(self, X, y, groups, target_type=None) -> dict:
        arr = np.asarray(self._prepare_X(X)); y = np.asarray(y)
        preds = self._cv_predict(arr, y, groups)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "confusion_matrix": confusion_matrix(y, preds),
            "predictions": preds, "true": y,
        }

    def permutation_test(self, X, y, groups, n_permutations: int = 200,
                         seed: int = 0, target_type=None) -> dict:
        arr = np.asarray(self._prepare_X(X)); y = np.asarray(y)
        score_fn = lambda yv: accuracy_score(yv, self._cv_predict(arr, yv, groups))
        return grouped_permutation_test(score_fn, groups, y,
                                        n_permutations=n_permutations, seed=seed)

    def stability_selection(self, X, y, groups, n_bootstrap: int = 100,
                            seed: int = 0) -> pd.DataFrame:
        """Group-level bootstrap: how often each feature gets a non-zero sparse weight.

        keepX should be set (sparse) for this to be meaningful.
        """
        if self.keepX is None:
            logger.warning("stability_selection without keepX (no sparsity); "
                           "results reflect VIP>=1, not sparse selection.")
        arr = np.asarray(self._prepare_X(X)); y = np.asarray(y)
        p = arr.shape[1]
        counts = np.zeros(p)
        for rows in grouped_bootstrap_indices(groups, n_bootstrap=n_bootstrap, seed=seed):
            f = _splsda_fit(arr[rows], y[rows], self.n_components, self.keepX,
                            self.max_iter, self.tol)
            if self.keepX is None:
                counts[f["vip"] >= 1.0] += 1
            else:
                counts[np.any(f["W"] != 0, axis=1)] += 1
        return pd.DataFrame({
            "feature": self.feature_names_,
            "selection_frequency": counts / n_bootstrap,
            "stable": (counts / n_bootstrap) >= 0.8,
        }).sort_values("selection_frequency", ascending=False).reset_index(drop=True)

"""
diablo.py
=========
DIABLO — multi-block sparse PLS-DA for supervised multi-omics integration, on the
BaseMethod interface. Finds latent components that are correlated across omics
blocks while discriminating between classes.

Ported from multiomics_integration.plsda.DIABLO (native NIPALS, design-matrix
weighted; Singh et al. 2019). Same deliberate upgrades as the other ports:
grouping-aware CV / permutation. Centers each block only (no internal scaling —
scaling is done once in preprocessing). handles_missing = False -> per-block
just-in-time imputation.

Unlike single-block methods, `fit`/`cross_validate` take **multiple blocks**:
either an OmicsDataset (blocks auto-extracted + aligned) or a dict of
{block_name: DataFrame}.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from ..base import BaseMethod
from .plsda import _soft_threshold, _encode_Y
from ...preprocessing.imputation import impute as _impute
from ...validation import leave_one_group_out, grouped_permutation_test

logger = logging.getLogger(__name__)


def _design_matrix(K: int, design) -> np.ndarray:
    if isinstance(design, (int, float)):
        D = np.full((K + 1, K + 1), float(design))
        np.fill_diagonal(D, 0.0)
        D[:, -1] = 1.0   # Y block always connected
        D[-1, :] = 1.0
        D[-1, -1] = 0.0
        return D
    return np.asarray(design, dtype=float)


def _diablo_fit(X_blocks, y, n_components=2, keepX=None, design=0.1,
                max_iter=500, tol=1e-6) -> dict:
    names = list(X_blocks.keys())
    K = len(names)
    Y, classes = _encode_Y(y)
    n, q = Y.shape
    D = _design_matrix(K, design)

    if keepX is None:
        kx = {nm: [X_blocks[nm].shape[1]] * n_components for nm in names}
    else:
        kx = {}
        for nm in names:
            v = keepX.get(nm, X_blocks[nm].shape[1]) if isinstance(keepX, dict) else keepX
            seq = [v] * n_components if isinstance(v, int) else list(v)
            while len(seq) < n_components:
                seq.append(seq[-1])
            kx[nm] = seq

    ncomp = min(n_components, n - 1)
    means = {nm: X_blocks[nm].mean(axis=0) for nm in names}
    Xk = {nm: X_blocks[nm] - means[nm] for nm in names}
    Yk = Y - Y.mean(axis=0)

    Wb = {nm: np.zeros((X_blocks[nm].shape[1], ncomp)) for nm in names}
    Pb = {nm: np.zeros((X_blocks[nm].shape[1], ncomp)) for nm in names}
    Tb = {nm: np.zeros((n, ncomp)) for nm in names}
    Yload = np.zeros((q, ncomp))

    for h in range(ncomp):
        combined = np.hstack([Xk[nm] for nm in names])
        U, S, _ = np.linalg.svd(combined, full_matrices=False)
        super_t = U[:, 0] * S[0]

        for _ in range(max_iter):
            t_blocks, w_blocks = {}, {}
            for ki, nm in enumerate(names):
                target = np.zeros(n)
                for kj, nmj in enumerate(names):
                    if ki != kj:
                        target += D[ki, kj] * (t_blocks[nmj] if nmj in t_blocks else super_t)
                target += D[ki, -1] * (Yk @ (Yk.T @ super_t)) / max(super_t @ super_t, 1e-10)
                if np.linalg.norm(target) < 1e-10:
                    target = super_t
                w_k = Xk[nm].T @ target
                nw = np.linalg.norm(w_k)
                if nw > 0:
                    w_k = w_k / nw
                w_k = _soft_threshold(w_k, kx[nm][h])
                w_blocks[nm] = w_k
                t_blocks[nm] = Xk[nm] @ w_k
            super_new = np.zeros(n)
            for nm in names:
                super_new += t_blocks[nm]
            super_new /= K
            if np.linalg.norm(super_new - super_t) < tol * np.linalg.norm(super_t + 1e-10):
                super_t = super_new
                break
            super_t = super_new

        for nm in names:
            Wb[nm][:, h] = w_blocks[nm]
            t_k = t_blocks[nm]
            Tb[nm][:, h] = t_k
            tt = t_k @ t_k
            if tt > 0:
                Pb[nm][:, h] = Xk[nm].T @ t_k / tt
        st = super_t @ super_t
        if st > 0:
            Yload[:, h] = Yk.T @ super_t / st
        for nm in names:
            t_k = Tb[nm][:, h]
            tt = t_k @ t_k
            if tt > 0:
                p_k = Xk[nm].T @ t_k / tt
                Xk[nm] = Xk[nm] - np.outer(t_k, p_k)
        if st > 0:
            Yk = Yk - np.outer(super_t, Yload[:, h])

    vip = {}
    for nm in names:
        T = Tb[nm]
        SS = np.array([(T[:, h] @ T[:, h]) * (Yload[:, h] @ Yload[:, h]) for h in range(ncomp)])
        tot = SS.sum()
        p_nm = X_blocks[nm].shape[1]
        vip[nm] = np.ones(p_nm) if tot == 0 else np.sqrt(p_nm * ((Wb[nm] ** 2) @ SS) / tot)

    corr = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            r = np.corrcoef(Tb[names[i]][:, 0], Tb[names[j]][:, 0])[0, 1]
            corr[i, j] = corr[j, i] = r

    return {"names": names, "classes": classes, "means": means, "W": Wb, "P": Pb,
            "T": Tb, "Yload": Yload, "ncomp": ncomp, "vip": vip,
            "corr": pd.DataFrame(corr, index=names, columns=names)}


def _diablo_predict(fit: dict, X_blocks) -> np.ndarray:
    names = fit["names"]
    n = list(X_blocks.values())[0].shape[0]
    avg = np.zeros((n, fit["ncomp"]))
    for nm in names:
        Xc = X_blocks[nm] - fit["means"][nm]
        W, P = fit["W"][nm], fit["P"][nm]
        PtW = P.T @ W
        try:
            R = W @ np.linalg.inv(PtW)
        except np.linalg.LinAlgError:
            R = W @ np.linalg.pinv(PtW)
        avg += Xc @ R
    avg /= len(names)
    Y_hat = avg @ fit["Yload"].T
    return fit["classes"][np.argmax(Y_hat, axis=1)]


class NativeDIABLO(BaseMethod):
    """EXPERIMENTAL native-Python DIABLO (teaching / exploration).

    NOT validated against mixOmics — its sample variates track mixOmics closely
    (sPLS-DA probe: r=0.999) but the multi-block design coupling is unverified.
    For analysis, use the R-backed `DIABLO` (mixOmics::block.splsda). Kept because
    Subaru rewrote DIABLO in Python to understand the algorithm.
    """
    handles_missing = False
    requires_target = True
    supported_targets = ("nominal", "ordinal")

    def __init__(self, n_components: int = 2, keepX=None, design: float = 0.1,
                 max_iter: int = 500, tol: float = 1e-6, impute: str = "metaboanalyst"):
        super().__init__(impute=impute)
        self.n_components = n_components
        self.keepX = keepX
        self.design = design
        self.max_iter = max_iter
        self.tol = tol
        self.fit_ = None
        self.block_names_ = None
        self.feature_names_ = None

    def _prepare_blocks(self, blocks):
        """Normalize input to {name: array}, impute per block, return sample order."""
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

        arrays, fnames = {}, {}
        for nm, dfb in bd.items():
            if not self.handles_missing and bool(dfb.isna().any().any()):
                dfb = _impute(dfb, self.impute_strategy)
            arrays[nm] = dfb.to_numpy()
            fnames[nm] = list(dfb.columns)
        sample_index = list(next(iter(bd.values())).index)
        return arrays, fnames, sample_index

    @staticmethod
    def _align_y(y, sample_index):
        if hasattr(y, "reindex"):
            return y.reindex(sample_index).to_numpy()
        return np.asarray(y)

    def fit(self, blocks, y, feature_names=None, target_type=None) -> "DIABLO":
        if target_type is not None:
            self._check_target(target_type)
        arrays, fnames, idx = self._prepare_blocks(blocks)
        self.block_names_ = list(arrays)
        self.feature_names_ = fnames
        yv = self._align_y(y, idx)
        self.fit_ = _diablo_fit(arrays, yv, self.n_components, self.keepX,
                                self.design, self.max_iter, self.tol)
        self._fitted = True
        return self

    def predict(self, blocks):
        arrays, _, _ = self._prepare_blocks(blocks)
        return _diablo_predict(self.fit_, arrays)

    def block_correlations(self) -> pd.DataFrame:
        return self.fit_["corr"]

    def vip(self, block: str, top_n=None) -> pd.DataFrame:
        v = self.fit_["vip"][block]
        df = pd.DataFrame({
            "feature": self.feature_names_[block], "vip": v, "important": v >= 1.0,
        }).sort_values("vip", ascending=False).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    def all_vip(self) -> pd.DataFrame:
        return pd.concat([self.vip(nm).assign(block=nm) for nm in self.block_names_],
                         ignore_index=True)

    # -- grouping-aware validation ----------------------------------------
    def _cv_predict(self, arrays, y, groups):
        names = list(arrays)
        preds = np.empty(len(y), dtype=object)
        for tr, te in leave_one_group_out(groups):
            Xtr = {nm: arrays[nm][tr] for nm in names}
            Xte = {nm: arrays[nm][te] for nm in names}
            f = _diablo_fit(Xtr, y[tr], self.n_components, self.keepX,
                            self.design, self.max_iter, self.tol)
            preds[te] = _diablo_predict(f, Xte)
        return preds

    def cross_validate(self, blocks, y, groups, target_type=None) -> dict:
        arrays, _, idx = self._prepare_blocks(blocks)
        yv = self._align_y(y, idx)
        preds = self._cv_predict(arrays, yv, groups)
        return {
            "accuracy": float(accuracy_score(yv, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(yv, preds)),
            "confusion_matrix": confusion_matrix(yv, preds),
            "predictions": preds, "true": yv,
        }

    def permutation_test(self, blocks, y, groups, n_permutations: int = 200,
                         seed: int = 0, target_type=None) -> dict:
        arrays, _, idx = self._prepare_blocks(blocks)
        yv = self._align_y(y, idx)
        score_fn = lambda yy: accuracy_score(yy, self._cv_predict(arrays, yy, groups))
        return grouped_permutation_test(score_fn, groups, yv,
                                        n_permutations=n_permutations, seed=seed)

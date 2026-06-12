"""
imputation.py
=============
Imputation strategies — applied ONLY when a method cannot handle missing values
(see methods.base.BaseMethod.handles_missing). Never applied before MOFA, which
models missingness natively.

Default = MetaboAnalyst (matches IdeaBio.R and IdeaBio.jl).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def metaboanalyst_impute(df: pd.DataFrame) -> pd.DataFrame:
    """MetaboAnalyst default: fill NaN per feature with 0.2 x min(positive value).

    Matches IdeaBio.jl ``impute_default_metaboanalyst`` and IdeaBio.R
    ``impute_missing_metaboanalyst``. Features with no positive values are left
    as-is (and will still contain NaN — caller should have filtered them).
    """
    out = df.copy()
    for col in out.columns:
        v = out[col]
        if not v.isna().any():
            continue
        positive = v[v > 0]
        if positive.empty:
            logger.warning("metaboanalyst_impute: feature %r has no positive values; left unfilled", col)
            continue
        out[col] = v.fillna(0.2 * positive.min())
    return out


def remove_all_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any feature containing one or more missing values (IdeaBio.jl RemoveAllMissing)."""
    before = df.shape[1]
    out = df.dropna(axis=1, how="any")
    dropped = before - out.shape[1]
    if dropped:
        logger.info("remove_all_missing: dropped %d/%d features with any NaN", dropped, before)
    return out


# ---------------------------------------------------------------------------
# Regularized iterative PCA imputation (faithful port of missMDA::imputePCA).
# This is the lab's DEFAULT imputation (used per-group by IdeaBio.R's
# impute_matrix_by_group). Deterministic: zero-init, no RNG.
# Algorithm verbatim from missMDA::imputePCA (method="Regularized"):
#   * row-weighted SVD (FactoMineR::svd.triplet), uniform row weights 1/n
#   * shrinkage lambda_s = (vs_s^2 - sigma2)/vs_s, sigma2 from the residual
#     singular values, capped at vs[ncp+1]^2
#   * EM loop: re-center/scale the completed matrix each iteration until
#     |1 - obj/old| < threshold (after >5 iters) or maxiter.
# ---------------------------------------------------------------------------

def _moy_p(v: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean ignoring NaN (missMDA moy.p)."""
    mask = ~np.isnan(v)
    return float(np.sum(np.where(mask, v * w, 0.0)) / np.sum(w[mask]))


def _ec(v: np.ndarray, w: np.ndarray) -> float:
    """Weighted RMS ignoring NaN (missMDA ec) -- used as the scale factor."""
    mask = ~np.isnan(v)
    return float(np.sqrt(np.sum(np.where(mask, v ** 2 * w, 0.0)) / np.sum(w[mask])))


def _svd_triplet(X: np.ndarray, row_w: np.ndarray, ncp: int):
    """Row-weighted SVD (FactoMineR::svd.triplet), column weights = 1.

    Returns (vs, U, V) with U, V de-weighted so that
    X ~= U[:, :k] @ diag(vs[:k]) @ V[:, :k].T. Signs cancel in the
    rank-k reconstruction, so the exact sign convention is irrelevant here.
    """
    sqrt_rw = np.sqrt(row_w)
    Xw = sqrt_rw[:, None] * X                  # col weights = 1
    U_w, s, Vt_w = np.linalg.svd(Xw, full_matrices=False)
    U = U_w / sqrt_rw[:, None]
    V = Vt_w.T
    return s, U, V


def imputepca(
    df: pd.DataFrame,
    ncp: int = 2,
    scale: bool = True,
    coeff_ridge: float = 1.0,
    threshold: float = 1e-6,
    maxiter: int = 1000,
    method: str = "regularized",
) -> pd.DataFrame:
    """Regularized iterative PCA imputation (missMDA::imputePCA parity).

    Operates on a samples x features DataFrame, returns the completed frame
    (observed values preserved, missing values imputed) on the input scale.
    """
    X = df.to_numpy(dtype=float)
    n, p = X.shape
    nrX, ncX = n, p
    if ncp > min(n - 2, p - 1):
        raise ValueError("ncp is too large")
    missing = np.isnan(X)
    if not missing.any():
        return df.copy()

    ncp = min(ncp, p, n - 1)
    row_w = np.full(n, 1.0 / n)

    mean_p = np.array([_moy_p(X[:, j], row_w) for j in range(p)])
    Xhat = X - mean_p
    et = np.array([_ec(Xhat[:, j], row_w) for j in range(p)])
    if scale:
        Xhat = Xhat / et
    Xhat = Xhat.copy()
    Xhat[missing] = 0.0
    fitted = Xhat.copy()

    old = np.inf
    nb_iter = 1
    df_denom = (nrX - 1) * ncX - (nrX - 1) * ncp - ncX * ncp + ncp ** 2
    while nb_iter > 0:
        Xhat[missing] = fitted[missing]
        if scale:
            Xhat = Xhat * et
        Xhat = Xhat + mean_p
        mean_p = Xhat.mean(axis=0)
        Xhat = Xhat - mean_p
        et = np.sqrt(np.mean(Xhat ** 2, axis=0))
        if scale:
            Xhat = Xhat / et
        vs, U, V = _svd_triplet(Xhat, row_w, ncp)
        tail = float(np.sum(vs[ncp:] ** 2))
        sigma2 = nrX * ncX / min(ncX, nrX - 1) * tail / df_denom
        sigma2 = min(sigma2 * coeff_ridge, float(vs[ncp] ** 2))
        if method == "em":
            sigma2 = 0.0
        lam = (vs[:ncp] ** 2 - sigma2) / vs[:ncp]
        fitted = (U[:, :ncp] * lam) @ V[:, :ncp].T
        diff = Xhat - fitted
        diff[missing] = 0.0
        objective = float(np.sum(diff ** 2 * row_w[:, None]))
        criterion = abs(1.0 - objective / old) if old not in (0.0, np.inf) else np.nan
        old = objective
        nb_iter += 1
        if not np.isnan(criterion):
            if criterion < threshold and nb_iter > 5:
                nb_iter = 0
            if objective < threshold and nb_iter > 5:
                nb_iter = 0
        if nb_iter > maxiter:
            nb_iter = 0

    if scale:
        Xhat = Xhat * et
    Xhat = Xhat + mean_p
    out = X.copy()
    out[missing] = Xhat[missing]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def imputepca_by_group(
    df: pd.DataFrame,
    groups,
    ncp: int = 2,
    scale: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Per-group regularized iterative PCA (IdeaBio.R impute_matrix_by_group parity).

    ``groups`` is a Series/array aligned to df rows. Each group's submatrix is
    imputed independently; original row order is preserved.
    """
    groups = pd.Series(np.asarray(groups), index=df.index)
    out = df.copy()
    for g in dict.fromkeys(groups.tolist()):
        idx = groups.index[groups == g]
        sub = df.loc[idx]
        if bool(sub.isna().any().any()):
            out.loc[idx] = imputepca(sub, ncp=ncp, scale=scale, **kwargs).to_numpy()
    return out


IMPUTERS = {
    "metaboanalyst": metaboanalyst_impute,
    "remove_all_missing": remove_all_missing,
    "remove": remove_all_missing,
    "imputepca": imputepca,
}


def impute(df: pd.DataFrame, strategy: str = "metaboanalyst") -> pd.DataFrame:
    """Dispatch to a named imputation strategy."""
    if strategy not in IMPUTERS:
        raise ValueError(f"unknown imputation strategy {strategy!r}; have {list(IMPUTERS)}")
    return IMPUTERS[strategy](df)

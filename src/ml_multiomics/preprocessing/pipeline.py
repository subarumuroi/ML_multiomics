"""
pipeline.py
===========
The canonical preprocessing pipeline (rebuilt from mofa_prep.py, NOT the old
ml_multiomics BasePreprocessor hierarchy).

Two missing-aware stages only:
    1. transform   (log2 / log10 / none)         -- NaN preserved
    2. normalize   (zscore / none)               -- NaN preserved
with optional variance / missingness feature filtering between them.

Imputation is NOT a stage here. It is method-gated and applied just-in-time by
methods that declare ``handles_missing = False`` (see methods.base). This is how
MOFA receives the NaN-carrying matrix while RF/PLS-DA/LASSO get an imputed copy.

Default per-omics profiles follow the lab conventions (z-score, log; pareto is
deliberately NOT the default). Pass a custom profile to override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .primitives import (
    log2_transform,
    log10_transform,
    zscore,
    variance_filter,
    missingness_filter,
)
from .imputation import imputepca, IMPUTERS
from ..core.provenance import ProvenanceTrail

logger = logging.getLogger(__name__)

_TRANSFORMS = {"log2": log2_transform, "log10": log10_transform, "none": None, None: None}
_NORMALIZERS = {"zscore": zscore, "none": None, None: None}


@dataclass
class Profile:
    """A preprocessing recipe for one omics type."""
    transform: Optional[str] = "log2"
    normalize: Optional[str] = "zscore"
    variance_min: Optional[float] = 1e-8
    max_missing_frac: Optional[float] = None  # None = no missingness filter


# Lab-convention defaults (z-score normalization, log transform).
DEFAULT_PROFILES = {
    "proteomics":   Profile(transform="log2",  normalize="zscore", variance_min=1e-8),
    "metabolomics": Profile(transform="log10", normalize="zscore", variance_min=1e-8),
    "volatiles":    Profile(transform="log10", normalize="zscore", variance_min=1e-8),
    "default":      Profile(transform="log2",  normalize="zscore", variance_min=1e-8),
}


class Preprocessor:
    """Apply the missing-aware transform + normalize pipeline to dataset blocks.

    Usage:
        Preprocessor().run(dataset)                       # per-omics defaults
        Preprocessor(profile=Profile(transform="log10")).run(dataset)
    """

    def __init__(
        self,
        profile: Optional[Profile] = None,
        profiles: Optional[dict] = None,
    ):
        # A single profile overrides everything; otherwise per-omics-type lookup.
        self.profile = profile
        self.profiles = profiles or DEFAULT_PROFILES

    def _profile_for(self, omics_type: Optional[str]) -> Profile:
        if self.profile is not None:
            return self.profile
        return self.profiles.get(omics_type, self.profiles["default"])

    def run(self, dataset, blocks: Optional[list[str]] = None):
        """Preprocess (in place) each block. Returns the dataset.

        Records every step in the block's provenance and sets the
        transformed/normalized flags. NaN is preserved; nothing is imputed.
        """
        names = blocks or dataset.block_names
        for name in names:
            block = dataset.blocks[name]
            prof = self._profile_for(block.omics_type)
            df = block.data

            # 1. transform
            tfn = _TRANSFORMS.get(prof.transform)
            if tfn is not None:
                df = tfn(df)
                block.transformed = True
                block.log(f"transform: {prof.transform}")

            # 2. feature filtering (variance, then missingness)
            if prof.variance_min is not None:
                df = variance_filter(df, prof.variance_min)
                block.log(f"variance_filter: > {prof.variance_min:g}")
            if prof.max_missing_frac is not None:
                df = missingness_filter(df, prof.max_missing_frac)
                block.log(f"missingness_filter: <= {prof.max_missing_frac:.0%}")

            # 3. normalize
            nfn = _NORMALIZERS.get(prof.normalize)
            if nfn is not None:
                df = nfn(df)
                block.normalized = True
                block.log(f"normalize: {prof.normalize}")

            block.data = df
        return dataset


class FittablePreprocessor:
    """Leakage-free preprocessing: learn all data-dependent params on TRAIN, apply
    to any matrix.

    Unlike :class:`Preprocessor` (which preprocesses a whole dataset once, in place
    — correct for the *descriptive* standard-analysis matrix), this transformer is
    built to run INSIDE a CV / permutation / bootstrap fold: ``fit`` sees only the
    training rows, ``transform`` applies the stored params to held-out rows so no
    test information leaks into the detection filter, variance filter, imputation,
    or z-score.

    Step order (all data-dependent steps fit on train):
        transform (log) -> detection filter -> variance filter -> impute -> z-score

    Parameters
    ----------
    profile : Profile | None
        Transform / normalize / variance recipe. If None, derived from omics_type.
    omics_type : str | None
        Used to look up a default Profile when ``profile`` is None.
    min_obs_frac : float | None
        Detection filter: keep a feature only if observed in >= this fraction of
        TRAIN rows. ``None`` = no detection filter (method-aware: the engine turns
        this off for tree-native methods that tolerate sparse features).
    impute : str | None
        Imputation strategy ('metaboanalyst', 'imputepca', 'remove_all_missing'/
        'remove') or ``None`` to leave NaN in place (for handles_missing methods).
    """

    def __init__(
        self,
        profile: Optional[Profile] = None,
        omics_type: Optional[str] = None,
        min_obs_frac: Optional[float] = None,
        impute: Optional[str] = None,
        input_state: Optional[dict] = None,
    ):
        self.profile = profile or DEFAULT_PROFILES.get(omics_type, DEFAULT_PROFILES["default"])
        self.omics_type = omics_type
        self.min_obs_frac = min_obs_frac
        self.impute = impute
        if impute is not None and impute not in IMPUTERS:
            raise ValueError(f"unknown impute strategy {impute!r}; have {list(IMPUTERS)} or None")
        # upstream state: what the data ALREADY had done to it (skip/adapt, don't double-apply)
        self.input_state = input_state or {}
        self._already_transformed = self.input_state.get("transform") not in (None, "none", "unknown")
        self._already_normalized = bool(self.input_state.get("normalized"))
        self._already_imputed = bool(self.input_state.get("imputed"))
        #: human-readable flags the engine promotes to divergences
        self.upstream_notes_: list = []
        # learned state
        self.keep_cols_: Optional[list] = None
        self.impute_fills_: Optional[pd.Series] = None      # metaboanalyst per-feature fill
        self.train_means_: Optional[pd.Series] = None       # fallback fill / pre-zscore mean
        self.zscore_mean_: Optional[pd.Series] = None
        self.zscore_sd_: Optional[pd.Series] = None
        self.provenance = ProvenanceTrail(name=f"preprocess[{omics_type or 'block'}]")
        self._fit_cache_: Optional[pd.DataFrame] = None
        self.fitted_ = False

    # -- internal stages ---------------------------------------------------
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._already_transformed:        # upstream already transformed -> don't double-apply
            return df
        tfn = _TRANSFORMS.get(self.profile.transform)
        return tfn(df) if tfn is not None else df

    @staticmethod
    def _np_fill(df: pd.DataFrame, fill: pd.Series) -> pd.DataFrame:
        """Fill every NaN with a per-feature value, vectorized in numpy.

        Chained ``df.fillna(Series)`` on a wide frame is O(cols) Python-level setitem
        (the dominant per-fold cost). One numpy scatter-fill is ~100x faster.
        """
        arr = df.to_numpy(dtype=float, copy=True)
        r, c = np.where(np.isnan(arr))
        if len(r):
            arr[r, c] = np.asarray(fill.reindex(df.columns), dtype=float)[c]
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    def _complete(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Fill missing values per the strategy using params learned in fit().

        Fill params (impute_fills_ / train_means_) are learned in fit() regardless of
        whether the train fold itself had gaps, so a test fold with NaN never hits a
        None fill. Combining the fallbacks at the Series level (length n_features) keeps
        features with no positive/observed value covered without a wide-frame fillna.
        """
        if self.impute is None or not bool(df.isna().any().any()):
            return df
        if self.impute == "metaboanalyst":
            fill = self.impute_fills_.reindex(df.columns).fillna(self.train_means_).fillna(0.0)
            return self._np_fill(df, fill)
        if self.impute in ("remove_all_missing", "remove"):
            return self._np_fill(df, self.train_means_.fillna(0.0))
        if self.impute == "imputepca":
            if is_train:
                try:
                    return imputepca(df, ncp=min(2, max(1, min(df.shape) - 2)))
                except Exception as e:  # tiny/degenerate fold -> mean fallback
                    self.provenance.record("impute_fallback", {"reason": str(e)[:60]},
                                           note="imputePCA failed; train mean fill")
                    return self._np_fill(df, self.train_means_.fillna(0.0))
            return self._np_fill(df, self.train_means_.fillna(0.0))  # test: leakage-free train-mean fill
        return df

    # -- API ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "FittablePreprocessor":
        X = pd.DataFrame(X)
        n_in = X.shape

        # record upstream-state adaptations up front (the engine promotes these to divergences)
        if self._already_transformed:
            self.upstream_notes_.append(
                f"upstream already applied transform '{self.input_state.get('transform')}'; "
                f"the ML transform '{self.profile.transform}' was SKIPPED (no double-transform)."
            )
        if self._already_normalized and self.profile.normalize == "zscore":
            self.upstream_notes_.append(
                "upstream already normalized this layer; per-fold z-score SKIPPED -- the raw scale "
                "is unrecoverable, so upstream (likely global) scaling cannot be made leakage-free. "
                "Supply a raw_source for a clean per-fold scaling."
            )
        if self._already_imputed:
            self.upstream_notes_.append(
                "upstream pre-imputed this layer; missingness is masked, so the detection filter and "
                "per-fold imputation cannot operate from raw. Supply a raw_source (e.g. the unimputed "
                "matrix) for leakage-free handling."
            )

        df = self._transform(X)
        self.provenance.record(
            "transform",
            {"kind": "skipped(upstream)" if self._already_transformed else self.profile.transform},
            in_obj=X, out_obj=df, fit_on_train=True,
            note="skipped: already transformed upstream" if self._already_transformed else "",
        )

        # detection filter (method-aware; max_missing_frac = 1 - min_obs_frac).
        # Vectorized boolean mask -- a per-column list comprehension over thousands of
        # features was the dominant per-fold cost after the imputer loop.
        if self.min_obs_frac is not None:
            n0 = df.shape[1]
            obs = 1.0 - df.isna().mean(axis=0)
            df = df.loc[:, obs >= self.min_obs_frac]
            self.provenance.record("detection_filter", {"min_obs_frac": self.min_obs_frac},
                                   in_obj=df, fit_on_train=True, note=f"kept {df.shape[1]}/{n0}")

        # variance filter
        if self.profile.variance_min is not None:
            n0 = df.shape[1]
            var = df.var(axis=0, ddof=1)
            df = df.loc[:, var > self.profile.variance_min]
            self.provenance.record("variance_filter", {"min_variance": self.profile.variance_min},
                                   in_obj=df, fit_on_train=True, note=f"kept {df.shape[1]}/{n0}")

        # complete-case: further restrict to columns with no train missing
        if self.impute in ("remove_all_missing", "remove"):
            n0 = df.shape[1]
            df = df.loc[:, ~df.isna().any(axis=0)]
            self.provenance.record("complete_case", {}, in_obj=df, fit_on_train=True,
                                   note=f"kept {df.shape[1]} complete features (of {n0})")

        self.keep_cols_ = list(df.columns)
        # train means BEFORE imputation (fallback) -- skipna
        self.train_means_ = df.mean(axis=0)
        # learn metaboanalyst per-feature fills on TRAIN unconditionally, so transform()
        # can fill a test-fold gap even when the train fold had none.
        # Vectorized (0.2 x min positive per feature): a per-column Python loop here cost
        # ~1.3 s on 4375 features and ran inside every fold of every permutation.
        if self.impute == "metaboanalyst":
            self.impute_fills_ = 0.2 * df.where(df > 0).min(axis=0)

        completed = self._complete(df, is_train=True)
        if self.impute is not None:
            self.provenance.record("impute", {"strategy": self.impute},
                                   in_obj=df, out_obj=completed, fit_on_train=True)

        # z-score params learned on the (completed-or-observed) train matrix
        if self.profile.normalize == "zscore" and not self._already_normalized:
            self.zscore_mean_ = completed.mean(axis=0)
            sd = completed.std(axis=0, ddof=1).replace(0, np.nan)
            self.zscore_sd_ = sd.where(sd.notna(), 1.0)
            out = (completed - self.zscore_mean_) / self.zscore_sd_
            self.provenance.record("zscore", {"ddof": 1}, in_obj=completed, out_obj=out,
                                   fit_on_train=True, note="mean/sd from train")
        else:
            out = completed
            if self.profile.normalize == "zscore" and self._already_normalized:
                self.provenance.record("zscore", {"kind": "skipped(upstream)"},
                                       in_obj=completed, out_obj=out, fit_on_train=True,
                                       note="skipped: already normalized upstream")

        self._fit_cache_ = out
        self.fitted_ = True
        self.provenance.record("fit_complete", {"n_in": list(n_in), "n_out": list(out.shape)},
                               fit_on_train=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("call fit() before transform()")
        X = pd.DataFrame(X)
        df = self._transform(X)
        df = df.reindex(columns=self.keep_cols_)         # train-selected features only
        df = self._complete(df, is_train=False)
        if self.profile.normalize == "zscore" and not self._already_normalized and self.zscore_mean_ is not None:
            df = (df - self.zscore_mean_) / self.zscore_sd_
        return df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self._fit_cache_

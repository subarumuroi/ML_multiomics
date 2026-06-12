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

import pandas as pd

from .primitives import (
    log2_transform,
    log10_transform,
    zscore,
    variance_filter,
    missingness_filter,
)

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

# ============================================================================
# src/ml_multiomics/preprocessing/__init__.py
# ============================================================================
"""Preprocessing modules."""

# --- Canonical missing-aware preprocessing (rebuilt from mofa_prep.py) -------
from .primitives import (
    log2_transform,
    log10_transform,
    zscore,
    variance_filter,
    missingness_filter,
    missingness_filter_by_group,
)
from .imputation import (
    metaboanalyst_impute, remove_all_missing, impute, IMPUTERS,
    imputepca, imputepca_by_group,
)
from .pipeline import Preprocessor, Profile, DEFAULT_PROFILES

# --- Legacy preprocessor hierarchy (DEPRECATED; removed during port) ---------
# Kept temporarily so existing examples/workflows keep importing. New code
# should use Preprocessor + the primitives above.
from .base_preprocessor import BasePreprocessor
from .omics_preprocessor import (
    MetabolomicsPreprocessor,
    VolatilesPreprocessor,
    ProteomicsPreprocessor
)
from .integrator import OmicsIntegrator, MultiBlockData

__all__ = [
    # canonical
    "Preprocessor", "Profile", "DEFAULT_PROFILES",
    "log2_transform", "log10_transform", "zscore",
    "variance_filter", "missingness_filter", "missingness_filter_by_group",
    "metaboanalyst_impute", "remove_all_missing", "impute", "IMPUTERS",
    "imputepca", "imputepca_by_group",
    # legacy (deprecated)
    "BasePreprocessor",
    "MetabolomicsPreprocessor", "VolatilesPreprocessor", "ProteomicsPreprocessor",
    "OmicsIntegrator", "MultiBlockData",
]
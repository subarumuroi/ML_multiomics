# ============================================================================
# src/ml_multiomics/methods/__init__.py
# ============================================================================
"""Analysis methods for single and multi-omics."""

from .base import BaseMethod
from . import supervised, unsupervised
from .supervised import (
    RandomForest, SparsePLSDA, DIABLO, NativeDIABLO,
    RegularizedLinear, Lasso, ElasticNet, Ordinal,
)
from .unsupervised import WGCNA, NativeWGCNA, NMF, PCA

__all__ = ['BaseMethod', 'supervised', 'unsupervised',
           'RandomForest', 'SparsePLSDA', 'DIABLO', 'NativeDIABLO',
           'Lasso', 'ElasticNet', 'RegularizedLinear', 'Ordinal',
           'WGCNA', 'NativeWGCNA', 'NMF', 'PCA']
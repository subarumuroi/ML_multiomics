# ============================================================================
# ml_multiomics/methods/single_omics/__init__.py
# ============================================================================
"""Single omics analysis methods."""

from .pca import PCAAnalysis
from .plsda import PLSDAAnalysis

__all__ = ['PCAAnalysis', 'PLSDAAnalysis']
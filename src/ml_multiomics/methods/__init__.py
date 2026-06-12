# ============================================================================
# src/ml_multiomics/methods/__init__.py
# ============================================================================
"""Analysis methods for single and multi-omics."""

from .base import BaseMethod
from . import supervised
from .supervised import RandomForest, SparsePLSDA

# legacy method namespaces (being consolidated onto BaseMethod)
from . import single_omics
from . import multi_omics

__all__ = ['BaseMethod', 'supervised', 'RandomForest', 'SparsePLSDA',
           'single_omics', 'multi_omics']
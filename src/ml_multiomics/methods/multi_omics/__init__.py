# ============================================================================
# src/ml_multiomics/methods/multi_omics/__init__.py
# ============================================================================
"""Multi-omics integration methods."""

from .diablo import DIABLO
from .concatenation_baseline import (
    ConcatenationBaseline,
    WeightedConcatenation
)
from .ensemble import BlockWiseEnsemble

__all__ = [
    'DIABLO',
    'ConcatenationBaseline',
    'WeightedConcatenation',
    'BlockWiseEnsemble',
]

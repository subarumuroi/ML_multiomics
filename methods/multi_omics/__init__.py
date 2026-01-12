# ============================================================================
# ml_multiomics/methods/multi_omics/__init__.py
# ============================================================================
"""Multi-omics integration methods."""

from .diablo import DIABLO
from .concatenation_baseline import (
    ConcatenationBaseline,
    WeightedConcatenation,
    EarlyIntegration
)

__all__ = [
    'DIABLO',
    'ConcatenationBaseline',
    'WeightedConcatenation',
    'EarlyIntegration'
]

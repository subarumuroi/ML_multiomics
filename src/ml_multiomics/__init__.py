# ============================================================================
# src/ml_multiomics/__init__.py
# ============================================================================
"""Multi-omics machine learning framework."""

__version__ = '0.1.0'
__author__ = 'Subaru Muroi'

from . import preprocessing
from . import methods
from . import workflows
from . import utils
from . import core
from . import validation

# --- Single-package API: the canonical foundation is importable directly -----
from .core import (
    OmicsDataset,
    Block,
    TargetSpec,
    parse_bioreactor_ids,
    parse_delimited,
)
from .preprocessing import Preprocessor, Profile, DEFAULT_PROFILES
from .methods.base import BaseMethod
from .methods.supervised import RandomForest

__all__ = [
    # subpackages
    'preprocessing', 'methods', 'workflows', 'utils', 'core', 'validation',
    # canonical foundation
    'OmicsDataset', 'Block', 'TargetSpec',
    'parse_bioreactor_ids', 'parse_delimited',
    'Preprocessor', 'Profile', 'DEFAULT_PROFILES',
    'BaseMethod', 'RandomForest',
]

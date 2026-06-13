"""Supervised methods."""

from .random_forest import RandomForest
from .plsda import SparsePLSDA
from .diablo import DIABLO
from .linear import RegularizedLinear, Lasso, ElasticNet
from .ordinal import Ordinal

__all__ = ["RandomForest", "SparsePLSDA", "DIABLO",
           "RegularizedLinear", "Lasso", "ElasticNet", "Ordinal"]

"""Supervised methods."""

from .random_forest import RandomForest
from .plsda import SparsePLSDA
from .diablo import DIABLO

__all__ = ["RandomForest", "SparsePLSDA", "DIABLO"]

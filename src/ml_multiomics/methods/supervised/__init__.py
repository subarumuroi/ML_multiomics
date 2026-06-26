"""Supervised methods."""

from .random_forest import RandomForest
from .plsda import SparsePLSDA
from .diablo_r import DIABLO              # DIABLO is R mixOmics::block.s/plsda only (validated)
from .linear import RegularizedLinear, Lasso, ElasticNet
from .ordinal import Ordinal
from .xgboost_method import XGBoost

__all__ = ["RandomForest", "SparsePLSDA", "DIABLO",
           "RegularizedLinear", "Lasso", "ElasticNet", "Ordinal", "XGBoost"]

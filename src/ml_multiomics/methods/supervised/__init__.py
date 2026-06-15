"""Supervised methods."""

from .random_forest import RandomForest
from .plsda import SparsePLSDA
from .diablo import NativeDIABLO          # experimental Python port (unvalidated)
from .diablo_r import DIABLO              # default: R mixOmics::block.splsda
from .linear import RegularizedLinear, Lasso, ElasticNet
from .ordinal import Ordinal
from .xgboost_method import XGBoost

__all__ = ["RandomForest", "SparsePLSDA", "DIABLO", "NativeDIABLO",
           "RegularizedLinear", "Lasso", "ElasticNet", "Ordinal", "XGBoost"]

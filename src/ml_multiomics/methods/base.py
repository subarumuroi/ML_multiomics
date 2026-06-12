"""
base.py
=======
Common interface for every method in the library, plus the missingness gate.

The key contract:
  * ``handles_missing`` declares whether a method models missing values natively.
      - MOFA / NMF-with-mask  -> True  : receive the NaN-carrying matrix as-is.
      - RF / PLS-DA / LASSO / ordinal -> False : get a just-in-time imputed copy.
  * Methods NEVER re-scale internally on top of the preprocessing pipeline
    (e.g. sklearn PLSRegression must be constructed with scale=False), because
    the data was already z-scored once. This is enforced by convention in each
    method and documented here so the rule is discoverable.
  * ``supported_targets`` gates applicability (e.g. ('continuous',) for a
    regressor, ('nominal','ordinal') for a classifier, () for unsupervised).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..preprocessing.imputation import impute as _impute

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    # -- capability declarations (override in subclasses) ------------------
    handles_missing: bool = False
    requires_target: bool = False
    supported_targets: tuple = ()  # subset of TARGET_TYPES; () = unsupervised

    def __init__(self, impute: str = "metaboanalyst"):
        self.impute_strategy = impute
        self._fitted = False

    # -- missingness gate --------------------------------------------------
    def _prepare_X(self, X) -> pd.DataFrame:
        """Return a model-ready matrix.

        If the method cannot handle missing values and X contains NaN, impute
        just-in-time with the configured strategy. Methods that declare
        ``handles_missing = True`` receive X untouched (NaN preserved).
        """
        X = pd.DataFrame(X)
        if self.handles_missing:
            return X
        if bool(X.isna().any().any()):
            logger.info(
                "%s cannot handle missing values; imputing (%s) before fit.",
                type(self).__name__, self.impute_strategy,
            )
            X = _impute(X, self.impute_strategy)
        return X

    def _check_target(self, target_type: Optional[str]) -> None:
        """Validate the dataset target against this method's capabilities."""
        if self.requires_target and target_type in (None, "none"):
            raise ValueError(f"{type(self).__name__} requires a target but none was set.")
        if self.supported_targets and target_type not in self.supported_targets:
            raise ValueError(
                f"{type(self).__name__} supports targets {self.supported_targets}; "
                f"got {target_type!r}."
            )

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def fit(self, X, y=None, **kwargs) -> "BaseMethod":
        ...

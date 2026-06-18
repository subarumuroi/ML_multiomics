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
    #: hyperparameter attribute names surfaced in params()/report_card()
    _PARAM_KEYS: tuple = ()

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

    # -- self-documentation (the per-method half of the provenance contract) --
    # Every concrete method MUST override describe/assumptions/divergences with
    # specifics; a parametrized test enforces this so no method ships undocumented.
    def describe(self) -> str:
        """Plain-language: what this method is, what it's for, how to read it."""
        return (
            f"{type(self).__name__}: supports targets {self.supported_targets or '(unsupervised)'}; "
            f"{'models missing values natively' if self.handles_missing else 'requires complete data (imputed just-in-time)'}."
        )

    def hyperparams(self) -> dict:
        """The hyperparameters actually in effect (for the report + provenance).

        Named ``hyperparams`` (not ``params``) so it never collides with a method
        that stores its config in a ``self.params`` attribute (e.g. XGBoost).
        """
        return {k: getattr(self, k) for k in self._PARAM_KEYS if hasattr(self, k)}

    def assumptions(self) -> list[str]:
        """Structured list of assumptions this method makes. Subclasses extend."""
        out = ["Samples are independent within the declared grouping/independent unit."]
        if not self.handles_missing:
            out.append(
                f"Missing values are imputed before fitting (strategy: {self.impute_strategy}); "
                "the result can depend on that choice."
            )
        if "zscore" not in ("",):  # data is pre-scaled once; no internal re-scaling
            out.append("Input is already z-scored by the pipeline; the method does not re-scale.")
        return out

    def divergences(self, context: Optional[dict] = None) -> list[str]:
        """Deviations from standard practice given the ACTUAL data/target context.

        ``context`` (supplied by the engine) may carry: target_type, n_samples,
        n_groups, n_features, missing_frac, grouping_has_repeats, block_sizes,
        is_multiblock, representation. Returns plain-language flags (possibly empty).
        Subclasses override and usually call ``super().divergences(context)``.
        """
        ctx = context or {}
        out: list[str] = []
        n_groups = ctx.get("n_groups")
        if n_groups is not None and n_groups < 10:
            out.append(
                f"Only {n_groups} independent units: results are exploratory and a single "
                "cross-validation score is uninformative (use permutation + stability)."
            )
        if ctx.get("grouping_has_repeats"):
            out.append(
                "Repeated measures per unit detected: treating rows independently would be "
                "pseudoreplication -- aggregate to one row per unit before inference."
            )
        mf = ctx.get("missing_frac")
        if mf is not None and mf > 0.2 and not self.handles_missing:
            out.append(
                f"{mf:.0%} of values were missing and imputed; conclusions should be checked "
                "across imputation strategies (sensitivity)."
            )
        tt = ctx.get("target_type")
        if tt == "ordinal" and self.supported_targets and "ordinal" in self.supported_targets \
                and "continuous" not in self.supported_targets and type(self).__name__ != "Ordinal":
            out.append("Ordinal target treated as nominal classes -- order information is discarded.")
        return out

    def report_card(self, context: Optional[dict] = None) -> dict:
        """The uniform per-model record rendered in the report."""
        return {
            "method": type(self).__name__,
            "describe": self.describe(),
            "params": self.hyperparams(),
            "assumptions": self.assumptions(),
            "divergences": self.divergences(context),
            "handles_missing": self.handles_missing,
            "supported_targets": list(self.supported_targets),
        }

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def fit(self, X, y=None, **kwargs) -> "BaseMethod":
        ...

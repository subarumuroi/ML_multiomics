"""
linear.py
=========
Regularized linear models — LASSO and ElasticNet — on the BaseMethod interface.
Regression (continuous targets, e.g. yield) or classification (L1/elastic-net
penalized logistic). New to the library (not in the source repos); built for the
small-n, p >> n regime where regularization + feature selection matter.

handles_missing = False -> just-in-time imputation. Grouping-aware CV/permutation.
Coefficients are sparse (L1) so non-zero coefficients are the selected features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet as _SkElasticNet, Lasso as _SkLasso, LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from ..base import BaseMethod
from ...validation import leave_one_group_out, grouped_permutation_test

logger = logging.getLogger(__name__)


class RegularizedLinear(BaseMethod):
    """L1/elastic-net linear model. l1_ratio=1.0 is LASSO; 0<l1_ratio<1 is elastic net."""

    handles_missing = False
    requires_target = True
    supported_targets = ("continuous", "nominal", "ordinal")

    def __init__(self, task: str = "auto", alpha: float = 1.0, l1_ratio: float = 0.5,
                 max_iter: int = 5000, random_state: int = 42, impute: str = "metaboanalyst"):
        super().__init__(impute=impute)
        self.task = task
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None
        self.feature_names_ = None
        self.task_ = None

    _PARAM_KEYS = ("task", "alpha", "l1_ratio", "max_iter")

    def describe(self) -> str:
        kind = "LASSO (pure L1)" if self.l1_ratio >= 1.0 else f"elastic net (l1_ratio={self.l1_ratio})"
        return (
            f"Regularized linear model -- {kind}. Performs embedded feature selection by shrinking "
            "coefficients to zero (regression) or balanced logistic regression (classification). "
            "Read the non-zero coefficients as the selected, signed predictors; selection is "
            "sensitive to alpha and to correlated features, so confirm with stability."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "Linear (additive) relationship between features and target on the modelled scale.",
            "Among correlated features L1 selects one somewhat arbitrarily -- selection identity is unstable.",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        ctx = context or {}
        if ctx.get("target_type") == "ordinal":
            out.append("Ordinal target fitted by (balanced) logistic classification -- order is discarded.")
        return out

    def _resolve_task(self, y, target_type=None) -> str:
        if self.task != "auto":
            return self.task
        if target_type == "continuous":
            return "regression"
        if target_type in ("nominal", "ordinal"):
            return "classification"
        y = np.asarray(y)
        if y.dtype.kind in "OUS":
            return "classification"
        return "classification" if len(np.unique(y)) <= max(10, int(0.2 * len(y))) else "regression"

    def _make_model(self, task: str):
        if task == "regression":
            if self.l1_ratio >= 1.0:
                return _SkLasso(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
            return _SkElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                 max_iter=self.max_iter, random_state=self.random_state)
        penalty = "l1" if self.l1_ratio >= 1.0 else "elasticnet"
        return LogisticRegression(
            penalty=penalty, solver="saga",
            l1_ratio=None if penalty == "l1" else self.l1_ratio,
            C=1.0 / self.alpha if self.alpha > 0 else 1.0,
            max_iter=self.max_iter, class_weight="balanced", random_state=self.random_state,
        )

    def fit(self, X, y, feature_names=None, target_type=None) -> "RegularizedLinear":
        if y is None:
            raise ValueError("RegularizedLinear requires a target y.")
        if target_type is not None:
            self._check_target(target_type)
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            arr = Xp.to_numpy()
        else:
            arr = np.asarray(Xp)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(arr.shape[1])]
        self.task_ = self._resolve_task(y, target_type)
        self.model_ = self._make_model(self.task_)
        self.model_.fit(arr, np.asarray(y))
        self._fitted = True
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(self._prepare_X(X)))

    def coefficients(self, top_n=None) -> pd.DataFrame:
        coef = np.asarray(self.model_.coef_)
        vals = np.abs(coef).mean(axis=0) if coef.ndim == 2 else coef
        df = pd.DataFrame({
            "feature": self.feature_names_, "coef": vals, "selected": vals != 0,
        }).reindex(np.argsort(-np.abs(vals))).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    def _cv_predict(self, arr, y, groups, task):
        preds = np.empty(len(y), dtype=object if task == "classification" else float)
        for tr, te in leave_one_group_out(groups):
            m = self._make_model(task)
            m.fit(arr[tr], y[tr])
            preds[te] = m.predict(arr[te])
        return preds

    def cross_validate(self, X, y, groups, target_type=None) -> dict:
        arr = np.asarray(self._prepare_X(X)); y = np.asarray(y)
        task = self._resolve_task(y, target_type) if self.task_ is None else self.task_
        preds = self._cv_predict(arr, y, groups, task)
        if task == "regression":
            preds = preds.astype(float)
            return {"task": "regression", "r2": float(r2_score(y, preds)),
                    "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                    "mae": float(mean_absolute_error(y, preds)), "predictions": preds, "true": y}
        return {"task": "classification", "accuracy": float(accuracy_score(y, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
                "confusion_matrix": confusion_matrix(y, preds), "predictions": preds, "true": y}

    def permutation_test(self, X, y, groups, n_permutations: int = 200, seed: int = 0,
                         target_type=None) -> dict:
        arr = np.asarray(self._prepare_X(X)); y = np.asarray(y)
        task = self._resolve_task(y, target_type) if self.task_ is None else self.task_

        def score_fn(yv):
            preds = self._cv_predict(arr, yv, groups, task)
            return r2_score(yv, preds.astype(float)) if task == "regression" else accuracy_score(yv, preds)

        return grouped_permutation_test(score_fn, groups, y, n_permutations=n_permutations, seed=seed)


class Lasso(RegularizedLinear):
    """LASSO (pure L1)."""
    def __init__(self, alpha: float = 1.0, **kw):
        kw.pop("l1_ratio", None)
        super().__init__(alpha=alpha, l1_ratio=1.0, **kw)


class ElasticNet(RegularizedLinear):
    """Elastic net (mixed L1/L2; default l1_ratio=0.5)."""
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kw):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **kw)

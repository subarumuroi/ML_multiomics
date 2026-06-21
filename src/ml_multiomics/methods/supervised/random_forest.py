"""
random_forest.py
================
Random Forest on the BaseMethod interface — classification (nominal/ordinal
targets) or regression (continuous targets, e.g. psilocybin yield).

Ported from multiomics_integration.random_forest, with two deliberate changes:
  1. **Grouping-aware** CV and permutation. The original used sample-level
     LeaveOneOut / label shuffling, which leaks across pseudoreplicates (e.g.
     timepoints of one bioreactor). Here CV/permutation REQUIRE a `groups`
     vector and operate at the group level (leave-one-group-out).
  2. **Regression support.** The original was classifier-only; psilocybin yield
     is continuous, so the task auto-resolves to a RandomForestRegressor when
     the target is continuous.

Cannot model missing values, so handles_missing = False — BaseMethod imputes
just-in-time (MetaboAnalyst by default) before fitting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance as _sk_perm_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ..base import BaseMethod
from ...validation import leave_one_group_out, grouped_permutation_test

logger = logging.getLogger(__name__)


class RandomForest(BaseMethod):
    handles_missing = False
    requires_target = True
    supported_targets = ("nominal", "ordinal", "continuous")

    def __init__(
        self,
        task: str = "auto",          # "auto" | "classification" | "regression"
        n_estimators: int = 500,
        max_depth: int = 3,          # shallow: small-n appropriate
        random_state: int = 42,
        impute: str = "metaboanalyst",
    ):
        super().__init__(impute=impute)
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_ = None
        self.feature_names_ = None
        self.task_ = None

    _PARAM_KEYS = ("task", "n_estimators", "max_depth", "random_state")

    def describe(self) -> str:
        return (
            "Random forest of shallow trees (small max_depth for small-n) for classification "
            "or regression. Nonlinear, no built-in feature selection; reports impurity-based "
            "feature importances. Read the importances as a ranking of predictors -- but at "
            "small n they are unstable, so weight them by the bootstrap stability result."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "No distributional assumptions; uses all supplied features.",
            "Impurity importances are model-internal and can be inflated for high-cardinality "
            "or correlated features.",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        ctx = context or {}
        nf, ng = ctx.get("n_features"), ctx.get("n_groups")
        if nf and ng and nf > 5 * ng:
            out.append(
                f"p>>n ({nf} features, {ng} units): forest importances spread thin and are "
                "unstable; a reduce->predict representation is usually more interpretable."
            )
        return out

    # -- task resolution ---------------------------------------------------
    def _resolve_task(self, y, target_type=None) -> str:
        if self.task != "auto":
            return self.task
        if target_type == "continuous":
            return "regression"
        if target_type in ("nominal", "ordinal"):
            return "classification"
        y = np.asarray(y)
        if y.dtype.kind in "OUS":  # strings -> classification
            return "classification"
        n_unique = len(np.unique(y))
        return "classification" if n_unique <= max(10, int(0.2 * len(y))) else "regression"

    def _make_model(self, task: str):
        if task == "regression":
            return RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=self.random_state,
            )
        return RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, class_weight="balanced",
        )

    # -- fit / predict -----------------------------------------------------
    def fit(self, X, y, feature_names=None, target_type=None) -> "RandomForest":
        if y is None:
            raise ValueError("RandomForest requires a target y.")
        if target_type is not None:
            self._check_target(target_type)
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            Xp_arr = Xp.to_numpy()
        else:
            Xp_arr = np.asarray(Xp)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(Xp_arr.shape[1])]
        self.task_ = self._resolve_task(y, target_type)
        self.model_ = self._make_model(self.task_)
        self.model_.fit(Xp_arr, np.asarray(y))
        self._fitted = True
        return self

    def predict(self, X):
        Xp = self._prepare_X(X)
        return self.model_.predict(np.asarray(Xp))

    # -- importances -------------------------------------------------------
    def importances(self, top_n=None) -> pd.DataFrame:
        """Gini (impurity) feature importance."""
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model_.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    def permutation_importance(self, X, y, n_repeats: int = 10) -> pd.DataFrame:
        """Model-agnostic permutation importance."""
        Xp = np.asarray(self._prepare_X(X))
        res = _sk_perm_importance(
            self.model_, Xp, np.asarray(y),
            n_repeats=n_repeats, random_state=self.random_state,
        )
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance_mean": res.importances_mean,
            "importance_std": res.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    def shap_importance(self, X) -> pd.DataFrame:
        """Mean |SHAP| per feature (TreeExplainer). Requires `pip install shap`."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap_importance requires `pip install shap`.") from exc
        Xp = np.asarray(self._prepare_X(X))
        sv = shap.TreeExplainer(self.model_)(Xp)
        vals = np.asarray(sv.values)
        mean_abs = np.abs(vals).mean(axis=(0, 2)) if vals.ndim == 3 else np.abs(vals).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names_, "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # -- grouping-aware validation ----------------------------------------
    def _cv_predict(self, Xp_arr, y, groups, task):
        """Leave-one-group-out predictions (no group spans train/test)."""
        preds = np.empty(len(y), dtype=object if task == "classification" else float)
        for tr, te in leave_one_group_out(groups):
            m = self._make_model(task)
            m.fit(Xp_arr[tr], y[tr])
            preds[te] = m.predict(Xp_arr[te])
        return preds

    def cross_validate(self, X, y, groups, target_type=None) -> dict:
        """Grouping-aware leave-one-group-out CV. REQUIRES a groups vector."""
        Xp_arr = np.asarray(self._prepare_X(X))
        y = np.asarray(y)
        task = self._resolve_task(y, target_type) if self.task_ is None else self.task_
        preds = self._cv_predict(Xp_arr, y, groups, task)
        if task == "regression":
            preds = preds.astype(float)
            return {
                "task": "regression",
                "r2": float(r2_score(y, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "mae": float(mean_absolute_error(y, preds)),
                "predictions": preds, "true": y,
            }
        return {
            "task": "classification",
            "accuracy": float(accuracy_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "confusion_matrix": confusion_matrix(y, preds),
            "predictions": preds, "true": y,
        }

    def permutation_test(self, X, y, groups, n_permutations: int = 200,
                         seed: int = 0, target_type=None) -> dict:
        """Group-level permutation test (labels permuted per group, not per sample)."""
        Xp_arr = np.asarray(self._prepare_X(X))
        y = np.asarray(y)
        task = self._resolve_task(y, target_type) if self.task_ is None else self.task_

        def score_fn(y_vec):
            preds = self._cv_predict(Xp_arr, y_vec, groups, task)
            if task == "regression":
                return r2_score(y_vec, preds.astype(float))
            return accuracy_score(y_vec, preds)

        return grouped_permutation_test(score_fn, groups, y,
                                        n_permutations=n_permutations, seed=seed)

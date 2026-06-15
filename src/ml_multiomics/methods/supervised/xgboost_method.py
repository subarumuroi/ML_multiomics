"""
xgboost_method.py
=================
Gradient-boosted trees (XGBoost) on the BaseMethod interface — classification or
regression — with conservative small-n defaults and grouping-aware CV.

Notable: XGBoost models missing values natively (it learns a default direction
per split), so `handles_missing = True` — it receives the NaN-carrying matrix
directly, like MOFA, rather than being imputed.

Honest caveat: at n < ~30 boosting is prone to overfitting and is NOT expected to
beat simpler regularized models here. It's included for completeness and as a
nonlinear baseline (with SHAP available), not as a recommendation. Defaults are
deliberately shallow + regularized.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder

from ..base import BaseMethod
from ...validation import leave_one_group_out, grouped_permutation_test

logger = logging.getLogger(__name__)


class XGBoost(BaseMethod):
    handles_missing = True          # XGBoost learns missing-value split directions
    requires_target = True
    supported_targets = ("nominal", "ordinal", "continuous")

    def __init__(self, task: str = "auto", n_estimators: int = 200, max_depth: int = 2,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.3, reg_lambda: float = 1.0,
                 min_child_weight: float = 1.0, random_state: int = 42):
        super().__init__(impute="metaboanalyst")  # unused: handles_missing=True
        self.task = task
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
            min_child_weight=min_child_weight, random_state=random_state,
        )
        self.model_ = None
        self.feature_names_ = None
        self.task_ = None
        self._le = None

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

    def _make(self, task):
        import xgboost as xgb
        if task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **self.params)
        return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **self.params)

    def fit(self, X, y, feature_names=None, target_type=None) -> "XGBoost":
        if y is None:
            raise ValueError("XGBoost requires a target y.")
        if target_type is not None:
            self._check_target(target_type)
        X = pd.DataFrame(X)                      # NaN kept (handles_missing=True)
        self.feature_names_ = list(X.columns)
        arr = X.to_numpy(dtype=float)
        self.task_ = self._resolve_task(y, target_type)
        yv = np.asarray(y)
        if self.task_ == "classification":
            self._le = LabelEncoder().fit(yv)
            yv = self._le.transform(yv)
        self.model_ = self._make(self.task_)
        self.model_.fit(arr, yv)
        self._fitted = True
        return self

    def predict(self, X):
        arr = pd.DataFrame(X).to_numpy(dtype=float)
        pred = self.model_.predict(arr)
        return self._le.inverse_transform(pred) if self._le is not None else pred

    def importances(self, top_n=None) -> pd.DataFrame:
        df = pd.DataFrame({"feature": self.feature_names_,
                           "importance": self.model_.feature_importances_}
                          ).sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    def _cv_predict(self, arr, y, groups, task, le=None):
        preds = np.empty(len(y), dtype=object if task == "classification" else float)
        for tr, te in leave_one_group_out(groups):
            m = self._make(task)
            ytr = le.transform(y[tr]) if le is not None else y[tr]
            m.fit(arr[tr], ytr)
            p = m.predict(arr[te])
            preds[te] = le.inverse_transform(p) if le is not None else p
        return preds

    def cross_validate(self, X, y, groups, target_type=None) -> dict:
        arr = pd.DataFrame(X).to_numpy(dtype=float)
        y = np.asarray(y)
        task = self._resolve_task(y, target_type) if self.task_ is None else self.task_
        le = LabelEncoder().fit(y) if task == "classification" else None
        preds = self._cv_predict(arr, y, groups, task, le)
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
        arr = pd.DataFrame(X).to_numpy(dtype=float)
        y = np.asarray(y)
        task = self._resolve_task(y, target_type) if self.task_ is None else self.task_
        le = LabelEncoder().fit(y) if task == "classification" else None

        def score_fn(yv):
            preds = self._cv_predict(arr, yv, groups, task, le)
            return r2_score(yv, preds.astype(float)) if task == "regression" else accuracy_score(yv, preds)

        return grouped_permutation_test(score_fn, groups, y, n_permutations=n_permutations, seed=seed)

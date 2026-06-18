"""
ordinal.py
==========
Ordinal regression (LogisticAT / IT / SE via `mord`) on the BaseMethod interface.
The right model when the target is an ORDERED category (e.g. Green < Ripe <
Overripe): it respects the ordering, so being off by one stage costs less than
being off by two. Primary metric is MAE (ordinal distance) alongside accuracy.

Ported from multiomics_integration.ordinal, with the source's internal
MinMaxScaler REMOVED — data is scaled once in preprocessing, and re-scaling here
would double-scale. Grouping-aware CV / permutation. handles_missing = False.

`mord` is an optional dependency: `pip install mord` (or `ml_multiomics[ordinal]`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

from ..base import BaseMethod
from ...validation import leave_one_group_out, grouped_permutation_test

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("AT", "IT", "SE")


def _mord_model(model_type: str, alpha: float):
    try:
        import mord
    except ImportError as exc:
        raise ImportError(
            "Ordinal regression requires `mord` (pip install mord, or "
            "pip install ml_multiomics[ordinal])."
        ) from exc
    return {"AT": mord.LogisticAT, "IT": mord.LogisticIT, "SE": mord.LogisticSE}[model_type](alpha=alpha)


class Ordinal(BaseMethod):
    handles_missing = False
    requires_target = True
    supported_targets = ("ordinal",)

    def __init__(self, model_type: str = "AT", alpha: float = 1.0, order=None,
                 impute: str = "metaboanalyst"):
        super().__init__(impute=impute)
        if model_type not in _MODEL_TYPES:
            raise ValueError(f"model_type must be one of {_MODEL_TYPES}")
        self.model_type = model_type
        self.alpha = alpha
        self.order = order          # ordered category labels (low -> high)
        self.model_ = None
        self.feature_names_ = None
        self.classes_ = None

    _PARAM_KEYS = ("model_type", "alpha", "order")

    def describe(self) -> str:
        return (
            "Ordinal regression (mord) -- an ordered logistic model that RESPECTS the category "
            "order (e.g. Green < Ripe < Over). Reports ordinal MAE (distance in ranks), which a "
            "plain classifier cannot. Read it as: does the model place samples in the right order, "
            "and by how many steps is it off."
        )

    def assumptions(self) -> list[str]:
        return super().assumptions() + [
            "The target is genuinely ordered and the order in `order` is correct.",
            "A single latent score underlies the ordinal categories (proportional-odds-style).",
        ]

    def divergences(self, context=None) -> list[str]:
        out = super().divergences(context)
        ctx = context or {}
        if ctx.get("target_type") and ctx["target_type"] != "ordinal":
            out.append(
                f"Applied to a {ctx['target_type']} target while assuming an order -- only valid "
                "if that target is truly ordered."
            )
        return out

    def _encode(self, y) -> np.ndarray:
        """Map ordered category labels to 0..k-1 (using self.order if given)."""
        y = np.asarray(y)
        if y.dtype.kind in "iu":   # already integer-encoded
            self.classes_ = np.unique(y)
            return y.astype(int)
        order = self.order if self.order is not None else sorted(pd.unique(y))
        if self.order is None:
            logger.warning("Ordinal: no `order` given; using sorted categories %s. "
                           "Pass order=[...] to set the true low->high ordering.", order)
        self.classes_ = np.asarray(order)
        mapping = {c: i for i, c in enumerate(order)}
        return np.array([mapping[v] for v in y], dtype=int)

    def fit(self, X, y, feature_names=None, target_type=None) -> "Ordinal":
        if y is None:
            raise ValueError("Ordinal requires a target y.")
        if target_type is not None:
            self._check_target(target_type)
        Xp = self._prepare_X(X)
        if isinstance(Xp, pd.DataFrame):
            self.feature_names_ = list(Xp.columns)
            arr = Xp.to_numpy()
        else:
            arr = np.asarray(Xp)
            self.feature_names_ = feature_names or [f"f{i}" for i in range(arr.shape[1])]
        y_enc = self._encode(y)
        self.model_ = _mord_model(self.model_type, self.alpha)
        self.model_.fit(arr, y_enc)
        self._fitted = True
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(self._prepare_X(X)))

    def coefficients(self, top_n=None) -> pd.DataFrame:
        coef = np.asarray(self.model_.coef_).flatten()
        df = pd.DataFrame({"feature": self.feature_names_, "coef": coef,
                           "abs_coef": np.abs(coef)}).sort_values(
            "abs_coef", ascending=False).reset_index(drop=True)
        return df.head(top_n) if top_n else df

    def _cv_predict(self, arr, y_enc, groups):
        preds = np.empty(len(y_enc), dtype=int)
        for tr, te in leave_one_group_out(groups):
            m = _mord_model(self.model_type, self.alpha)
            m.fit(arr[tr], y_enc[tr])
            preds[te] = m.predict(arr[te]).astype(int)
        return preds

    def cross_validate(self, X, y, groups, target_type=None) -> dict:
        arr = np.asarray(self._prepare_X(X))
        y_enc = self._encode(y)
        preds = self._cv_predict(arr, y_enc, groups)
        return {
            "accuracy": float(accuracy_score(y_enc, preds)),
            "mae": float(mean_absolute_error(y_enc, preds)),   # ordinal distance
            "confusion_matrix": confusion_matrix(y_enc, preds),
            "predictions": preds, "true": y_enc,
        }

    def permutation_test(self, X, y, groups, n_permutations: int = 200, seed: int = 0,
                         target_type=None) -> dict:
        arr = np.asarray(self._prepare_X(X))
        y_enc = self._encode(y)
        score_fn = lambda yv: accuracy_score(yv, self._cv_predict(arr, yv, groups))
        return grouped_permutation_test(score_fn, groups, y_enc,
                                        n_permutations=n_permutations, seed=seed)

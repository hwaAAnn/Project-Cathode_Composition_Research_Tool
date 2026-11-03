"""XGBoost regression -- uses enable_categorical=True (XGBoost 3.0+)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import CATEGORICAL_FEATURES, FEATURES, RANDOM_STATE
from .base import BaseRegressor


class XGBoostModel(BaseRegressor):
    name = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default = dict(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            enable_categorical=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
        )
        if params:
            default.update(params)
        self.params = default
        self.model: xgb.XGBRegressor | None = None
        self.categories: dict[str, list[str]] = {}

    def _prep(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X[FEATURES].copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype(str)
            if fit:
                self.categories[c] = sorted(X[c].unique().tolist())
            cats = self.categories.get(c, sorted(X[c].unique().tolist()))
            X[c] = pd.Categorical(X[c], categories=cats)
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **_: Any,
    ) -> "XGBoostModel":
        Xp = self._prep(X, fit=True)
        self.model = xgb.XGBRegressor(**self.params)
        if X_val is not None:
            self.model.fit(Xp, y, eval_set=[(self._prep(X_val), y_val)], verbose=False)
        else:
            params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(Xp, y, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(self._prep(X, fit=False))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"params": self.params, "model": self.model, "categories": self.categories},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        obj = cls(params=blob["params"])
        obj.model = blob["model"]
        obj.categories = blob["categories"]
        return obj

"""LightGBM regression -- uses pandas category columns directly as categorical_feature."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import CATEGORICAL_FEATURES, FEATURES, RANDOM_STATE
from ..features import to_category_df
from .base import BaseRegressor


class LightGBMModel(BaseRegressor):
    name = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default = dict(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        if params:
            default.update(params)
        self.params = default
        self.model: lgb.LGBMRegressor | None = None
        # keep training-time categories -- so unknown labels are not mapped to -1 at inference
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
        early_stopping_rounds: int = 50,
        **_: Any,
    ) -> "LightGBMModel":
        Xp = self._prep(X, fit=True)
        eval_set = None
        if X_val is not None:
            eval_set = [(self._prep(X_val), y_val)]

        self.model = lgb.LGBMRegressor(**self.params)
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if eval_set else None
        self.model.fit(
            Xp,
            y,
            eval_set=eval_set,
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        Xp = self._prep(X, fit=False)
        return self.model.predict(Xp)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"params": self.params, "model": self.model, "categories": self.categories},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        obj = cls(params=blob["params"])
        obj.model = blob["model"]
        obj.categories = blob["categories"]
        return obj

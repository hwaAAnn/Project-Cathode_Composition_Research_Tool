"""CatBoost regression -- native categorical handling."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ..config import CATEGORICAL_FEATURES, FEATURES, RANDOM_STATE
from .base import BaseRegressor


class CatBoostModel(BaseRegressor):
    name = "catboost"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default = dict(
            iterations=1500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        if params:
            default.update(params)
        self.params = default
        self.model: CatBoostRegressor | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 50,
        **_: Any,
    ) -> "CatBoostModel":
        self.model = CatBoostRegressor(**self.params)
        eval_set = (X_val[FEATURES], y_val) if X_val is not None else None
        self.model.fit(
            X[FEATURES],
            y,
            cat_features=CATEGORICAL_FEATURES,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit first"
        return self.model.predict(X[FEATURES])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"params": self.params, "model": self.model}, f)

    @classmethod
    def load(cls, path: Path) -> "CatBoostModel":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        obj = cls(params=blob["params"])
        obj.model = blob["model"]
        return obj

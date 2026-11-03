"""RandomForest regression -- integer-encodes categoricals with OrdinalEncoder."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import FEATURES, RANDOM_STATE
from ..features import apply_ordinal, fit_ordinal_encoder
from .base import BaseRegressor


class RandomForestModel(BaseRegressor):
    name = "random_forest"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        if params:
            default.update(params)
        self.params = default
        self.model: RandomForestRegressor | None = None
        self.encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **_: Any) -> "RandomForestModel":
        self.encoder = fit_ordinal_encoder(X)
        Xe = apply_ordinal(X[FEATURES], self.encoder)
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(Xe, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.encoder is not None
        Xe = apply_ordinal(X[FEATURES], self.encoder)
        return self.model.predict(Xe)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"params": self.params, "model": self.model, "encoder": self.encoder}, f
            )

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        obj = cls(params=blob["params"])
        obj.model = blob["model"]
        obj.encoder = blob["encoder"]
        return obj

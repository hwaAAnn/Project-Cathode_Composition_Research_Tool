"""Best-3 model ensemble.

Weights = inverse-error weighting, normalized from 1 / val RMSE.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .base import BaseRegressor, Metrics, evaluate


class EnsembleRegressor:
    name = "ensemble"

    def __init__(
        self,
        models: Sequence[BaseRegressor],
        weights: Sequence[float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one base model required")
        self.models = list(models)
        if weights is None:
            weights = [1.0] * len(models)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        self.weights = w.tolist()

    @classmethod
    def from_val_metrics(
        cls,
        models: Sequence[BaseRegressor],
        metrics: Sequence[Metrics],
    ) -> "EnsembleRegressor":
        """Compute weights from the inverse of RMSE."""
        rmses = np.array([max(m.rmse, 1e-6) for m in metrics])
        weights = 1.0 / rmses
        return cls(models=models, weights=weights.tolist())

    def predict(self, X: pd.DataFrame, return_components: bool = False):
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        mean = (preds * np.array(self.weights)[:, None]).sum(axis=0)
        if return_components:
            return mean, preds  # preds shape (K, N)
        return mean

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Use the variance across models as a proxy for epistemic uncertainty."""
        mean, comps = self.predict(X, return_components=True)
        std = comps.std(axis=0)
        return mean, std

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Metrics:
        return evaluate(y.values, self.predict(X))

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        manifest = {"names": [], "weights": self.weights}
        for i, m in enumerate(self.models):
            fname = f"member_{i:02d}_{m.name}.pkl"
            m.save(dir_path / fname)
            manifest["names"].append({"name": m.name, "file": fname, "class": type(m).__name__})
        (dir_path / "ensemble.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path) -> "EnsembleRegressor":
        from . import catboost_model, lightgbm_model, rf_model, xgboost_model

        registry = {
            "CatBoostModel": catboost_model.CatBoostModel,
            "LightGBMModel": lightgbm_model.LightGBMModel,
            "XGBoostModel": xgboost_model.XGBoostModel,
            "RandomForestModel": rf_model.RandomForestModel,
        }
        manifest = json.loads(Path(dir_path / "ensemble.json").read_text(encoding="utf-8"))
        models: list[BaseRegressor] = []
        for entry in manifest["names"]:
            cls_ = registry[entry["class"]]
            models.append(cls_.load(Path(dir_path) / entry["file"]))
        return cls(models=models, weights=manifest["weights"])

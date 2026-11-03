"""End-to-end pipeline.

Offline:
  - load data per chemistry -> train 4 models -> save ensemble

Online:
  1. choose chemistry -> load ensemble
  2. user defines fixed/searched variables (+ fixed values) + top/bottom percent
  3. build SearchSpace -> run GA -> generate candidate solutions
  4. KNN validation -> assign verdict labels
  5. return results
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bounds import SearchSpace, build_search_space
from .config import ARTIFACT_DIR, FEATURES, TARGET
from .data import Dataset, load_chemistry, merge_with_val
from .ga import GAConfig, GAResult, run_ga
from .models import EnsembleRegressor
from .validation import KNNValidator, annotate, fit_validator


@dataclass
class CathodePipeline:
    """Handler that manages trained assets in one place."""

    chemistry: str
    ensemble: EnsembleRegressor
    full_data: pd.DataFrame
    validator: KNNValidator
    metrics: dict[str, Any]

    @classmethod
    def load(cls, chemistry: str, artifact_dir: Path | None = None) -> "CathodePipeline":
        artifact_dir = Path(artifact_dir or (ARTIFACT_DIR / chemistry))
        if not (artifact_dir / "ensemble").exists():
            raise FileNotFoundError(
                f"{artifact_dir / 'ensemble'} not found. Run train_all first."
            )
        ensemble = EnsembleRegressor.load(artifact_dir / "ensemble")
        ds = load_chemistry(chemistry)
        full = merge_with_val(ds)
        validator = fit_validator(full)
        metrics = json.loads(
            (artifact_dir / "metrics.json").read_text(encoding="utf-8")
        )
        return cls(
            chemistry=chemistry,
            ensemble=ensemble,
            full_data=full,
            validator=validator,
            metrics=metrics,
        )

    # ───────────── single prediction call ─────────────
    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self.ensemble.predict_with_uncertainty(X)
        return mean, std

    # ───────────── search space / optimization ─────────────
    def build_space(
        self,
        fixed: dict[str, float | str] | None = None,
        top_pct: float = 20.0,
        bottom_pct: float = 50.0,
        expand_neighbors: int = 1,
    ) -> SearchSpace:
        return build_search_space(
            self.full_data,
            fixed=fixed,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            expand_neighbors=expand_neighbors,
        )

    def optimize(
        self,
        space: SearchSpace,
        ga_config: GAConfig | None = None,
        k_neighbors: int = 5,
        verbose: bool = True,
        on_generation=None,
    ) -> tuple[pd.DataFrame, GAResult]:
        ga = run_ga(
            space,
            predict_fn=self.ensemble.predict,
            config=ga_config,
            verbose=verbose,
            on_generation=on_generation,
        )
        annotated = annotate(ga.solutions, self.validator, k=k_neighbors)
        return annotated, ga


def quick_optimize(
    chemistry: str,
    fixed: dict[str, float | str] | None = None,
    top_pct: float = 20.0,
    bottom_pct: float = 50.0,
    ga_config: GAConfig | None = None,
    k_neighbors: int = 5,
) -> dict[str, Any]:
    """Load + search + validate in one go."""
    by = CathodePipeline.load(chemistry)
    space = by.build_space(fixed=fixed, top_pct=top_pct, bottom_pct=bottom_pct)
    annotated, ga = by.optimize(space, ga_config=ga_config, k_neighbors=k_neighbors)
    return {
        "chemistry": chemistry,
        "space": space.as_table().to_dict(orient="records"),
        "solutions": annotated.to_dict(orient="records"),
        "history": ga.history,
        "elapsed_seconds": ga.elapsed_seconds,
        "metrics": by.metrics,
    }

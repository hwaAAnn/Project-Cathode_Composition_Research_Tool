"""Data loader.

- load Train/Val per chemistry (LFP/NCA)
- extract only the 22 design variables + target column
- handle missing values / outliers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TRAIN_DIR,
    TRAIN_FILES,
    VAL_DIR,
    VAL_FILES,
)

Split = Literal["train", "val", "both"]


@dataclass
class Dataset:
    """Standard data container used by the pipeline."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    chemistry: str

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 22 design variables + target and enforce dtypes."""
    keep = FEATURES + [TARGET]
    df = df[keep].copy()

    # categorical: cast to string, missing -> 'missing'
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype(str).fillna("missing").replace({"nan": "missing"})

    # numeric: float32, missing -> column median
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
        df[c] = df[c].astype(np.float32)

    # clean target missing values / range
    df = df[df[TARGET].notna()].copy()
    df = df[(df[TARGET] > 0) & (df[TARGET] < 400)].copy()  # physical upper bound
    df[TARGET] = df[TARGET].astype(np.float32)

    return df.reset_index(drop=True)


def load_chemistry(chem: str) -> Dataset:
    """Load the Train+Val Dataset for a single chemistry."""
    if chem not in TRAIN_FILES:
        raise ValueError(f"unknown chemistry: {chem}. Supported: {list(TRAIN_FILES)}")

    df_t = pd.read_csv(TRAIN_DIR / TRAIN_FILES[chem], low_memory=False)
    df_v = pd.read_csv(VAL_DIR / VAL_FILES[chem], low_memory=False)
    df_t = _clean(df_t)
    df_v = _clean(df_v)

    return Dataset(
        X_train=df_t[FEATURES],
        y_train=df_t[TARGET],
        X_val=df_v[FEATURES],
        y_val=df_v[TARGET],
        chemistry=chem,
    )


def merge_with_val(dataset: Dataset) -> pd.DataFrame:
    """Return the full data (Train+Val) -- for bounds / KNN validation."""
    full = pd.concat(
        [
            pd.concat([dataset.X_train, dataset.y_train], axis=1),
            pd.concat([dataset.X_val, dataset.y_val], axis=1),
        ],
        ignore_index=True,
    )
    return full

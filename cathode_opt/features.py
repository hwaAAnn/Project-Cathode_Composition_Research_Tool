"""Per-model encoding/decoding helpers.

- CatBoost, LightGBM: handle categoricals as-is (or as 'category' dtype)
- XGBoost: pandas 'category' dtype + enable_categorical=True
- RandomForest: convert categoricals with OrdinalEncoder (unknown = largest integer)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass
class CategoricalVocab:
    """Category vocabulary of each categorical variable at training time.

    - used for GA encoding / decoding
    - unknown categories fall back to 'missing'
    """

    vocab: dict[str, list[str]]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "CategoricalVocab":
        vocab: dict[str, list[str]] = {}
        for c in CATEGORICAL_FEATURES:
            uniques = sorted(df[c].astype(str).unique().tolist())
            vocab[c] = uniques
        return cls(vocab=vocab)

    def values(self, col: str) -> list[str]:
        return self.vocab[col]

    def index_of(self, col: str, value: str) -> int:
        vals = self.vocab[col]
        try:
            return vals.index(value)
        except ValueError:
            return vals.index("missing") if "missing" in vals else 0


def to_category_df(X: pd.DataFrame) -> pd.DataFrame:
    """For LightGBM/XGBoost -- categoricals as pandas category dtype."""
    X = X.copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def fit_ordinal_encoder(X: pd.DataFrame) -> OrdinalEncoder:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X[CATEGORICAL_FEATURES].astype(str))
    return enc


def apply_ordinal(X: pd.DataFrame, enc: OrdinalEncoder) -> pd.DataFrame:
    X = X.copy()
    X[CATEGORICAL_FEATURES] = enc.transform(X[CATEGORICAL_FEATURES].astype(str))
    return X

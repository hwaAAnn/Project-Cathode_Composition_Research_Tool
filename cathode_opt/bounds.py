"""Fixed/searched variable configuration + data-driven dynamic search bounds.

Spec (Step 2):
- top n1% / bottom n2% by discharge_capacity (user input)
- the per-variable min/max over the data points in that range become the variable's bounds
- fixed variables are forced to their given value
- if a continuous variable is fixed, include its neighboring values when computing the bounds
  -> this implementation uses the policy 'pin to nearest unique value, then expand to its ±1 neighbors'
- result: per-variable spec of (min, max, allowed_categories, fixed_value)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

Mode = Literal["search", "fixed"]


@dataclass
class VariableSpec:
    """Search/fixed spec of a single design variable."""

    name: str
    kind: Literal["num", "cat"]
    mode: Mode = "search"
    fixed_value: float | str | None = None
    # filled in when the variable is searched
    low: float | None = None
    high: float | None = None
    allowed_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "mode": self.mode,
            "fixed_value": self.fixed_value,
            "low": self.low,
            "high": self.high,
            "allowed_categories": self.allowed_categories,
        }


@dataclass
class SearchSpace:
    """Entire search space."""

    specs: dict[str, VariableSpec]
    quantile_top: float
    quantile_bottom: float
    n_used: int  # number of data points kept after the quantile cut

    def search_numeric_vars(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.kind == "num" and s.mode == "search"]

    def search_categorical_vars(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.kind == "cat" and s.mode == "search"]

    def fixed_vars(self) -> dict[str, float | str]:
        return {n: s.fixed_value for n, s in self.specs.items() if s.mode == "fixed"}

    def as_table(self) -> pd.DataFrame:
        rows = []
        for s in self.specs.values():
            row = {
                "variable": s.name,
                "kind": s.kind,
                "mode": s.mode,
                "fixed_value": s.fixed_value,
                "low": s.low,
                "high": s.high,
                "n_categories": len(s.allowed_categories) if s.kind == "cat" else "-",
            }
            rows.append(row)
        return pd.DataFrame(rows)


# ──────────────────────────── builder helpers ──────────────────────
def _quantile_slice(
    df_full: pd.DataFrame, top_pct: float, bottom_pct: float
) -> pd.DataFrame:
    """
    top_pct/bottom_pct are 0-100. E.g. top=20 -> top 20% (high capacity side),
    bottom=10 -> exclude the bottom 10% and include everything above it...

    User spec: data 'within' top n1% 'and' bottom n2%.
    -> It could be read as the union of the 'top n1% region' and the 'bottom n2% region',
    but the practical reading is the slice "between the top n1% cutoff and the top (100-n2)% cutoff".
    We adopt the latter (upper-middle slice); very low points are noise/failed cases and are excluded.
    The user must enter values (e.g. n1=20, n2=80) whose meanings do not conflict; validated below.
    """
    if not (0 < top_pct < 100 and 0 <= bottom_pct < 100):
        raise ValueError("top_pct / bottom_pct must be in the range 0-100")
    if top_pct + bottom_pct >= 100:
        raise ValueError("top_pct + bottom_pct must be < 100 (the search slice would be empty)")

    y = df_full[TARGET]
    upper_thr = y.quantile(1.0 - top_pct / 100.0)
    lower_thr = y.quantile(bottom_pct / 100.0)
    sliced = df_full[(y >= lower_thr) & (y <= upper_thr)].copy()
    return sliced


def _expand_fixed_numeric(
    full_col: pd.Series, value: float, neighbors: int = 1
) -> tuple[float, float]:
    """Instead of keeping a continuous fixed value as is, convert it to a range that includes adjacent unique values."""
    uniques = np.sort(full_col.unique())
    if len(uniques) == 0:
        return value, value
    # index closest to value
    nearest_idx = int(np.argmin(np.abs(uniques - value)))
    low_idx = max(0, nearest_idx - neighbors)
    high_idx = min(len(uniques) - 1, nearest_idx + neighbors)
    return float(uniques[low_idx]), float(uniques[high_idx])


def build_search_space(
    df_full: pd.DataFrame,
    fixed: dict[str, float | str] | None = None,
    top_pct: float = 20.0,
    bottom_pct: float = 50.0,
    expand_neighbors: int = 1,
) -> SearchSpace:
    """Build a SearchSpace from the full data (Train+Val combined) and the fixed settings.

    Parameters
    ----------
    df_full : full data (FEATURES + TARGET)
    fixed : {variable name: fixed value} -- if None, all variables are searched
    top_pct, bottom_pct : slice ratios
    expand_neighbors : number of adjacent values to expand a fixed continuous variable by
    """
    fixed = dict(fixed or {})
    sliced = _quantile_slice(df_full, top_pct=top_pct, bottom_pct=bottom_pct)

    specs: dict[str, VariableSpec] = {}
    for col in FEATURES:
        kind = "cat" if col in CATEGORICAL_FEATURES else "num"
        if col in fixed:
            val = fixed[col]
            if kind == "num":
                lo, hi = _expand_fixed_numeric(
                    df_full[col], float(val), neighbors=expand_neighbors
                )
                specs[col] = VariableSpec(
                    name=col, kind=kind, mode="fixed", fixed_value=float(val), low=lo, high=hi
                )
            else:
                specs[col] = VariableSpec(
                    name=col,
                    kind=kind,
                    mode="fixed",
                    fixed_value=str(val),
                    allowed_categories=[str(val)],
                )
            continue

        # searched variable: slice-based lo/hi or categories
        if kind == "num":
            col_slice = sliced[col]
            if col_slice.empty:
                col_slice = df_full[col]
            lo, hi = float(col_slice.min()), float(col_slice.max())
            if lo == hi:
                # if the data slice has a single value, expand to the full data
                lo, hi = float(df_full[col].min()), float(df_full[col].max())
            specs[col] = VariableSpec(name=col, kind=kind, mode="search", low=lo, high=hi)
        else:
            cats = sorted(sliced[col].astype(str).unique().tolist())
            if not cats:
                cats = sorted(df_full[col].astype(str).unique().tolist())
            specs[col] = VariableSpec(
                name=col, kind=kind, mode="search", allowed_categories=cats
            )

    return SearchSpace(
        specs=specs,
        quantile_top=top_pct,
        quantile_bottom=bottom_pct,
        n_used=len(sliced),
    )

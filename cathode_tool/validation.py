"""KNN-based validation of optimal solutions -- Hallucination / Recommended / Strongly Recommended.

Procedure:
1. Standardize the training DB (train+val FEATURES) and build a KNN index
   - numeric: StandardScaler
   - categorical: One-Hot Encoder (sparse -> dense)
2. Measure the nearest-neighbor distance distribution within the training DB to derive **adaptive thresholds**
   (each dataset has a narrower/wider manifold, so fixed thresholds are inappropriate)
3. Transform each optimal solution the same way and run the K-NN search
4. Decide the verdict from the K neighbors' measured capacity mean/max + distance statistics

Verdict rules (adaptive):
  - dist_min <= echo_dist (~0)           AND |pred - neigh_max| small -> "DB Echo"
  - dist_mean <= d_close (DB 50th pct)   AND pred >= neigh_max - gap  -> "Strongly Recommended"
  - dist_mean <= d_far   (DB 90th pct)   AND pred >= neigh_mean       -> "Recommended"
  - pred ≫ neigh_max + excess            AND dist_mean ≫ d_far       → "Hallucination"
  - otherwise                                                            -> "Uncertain"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


@dataclass
class KNNValidator:
    scaler: StandardScaler
    ohe: OneHotEncoder
    knn: NearestNeighbors
    y_train: np.ndarray
    feature_table: pd.DataFrame
    # adaptive distance thresholds (based on the nearest-neighbor distance distribution inside the training DB)
    d_close: float = 1.0      # 50th percentile
    d_far: float = 2.0        # 90th percentile
    cap_std: float = 20.0     # capacity standard deviation (gap scaling)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self.scaler.transform(X[NUMERIC_FEATURES].astype(float))
        Xc = self.ohe.transform(X[CATEGORICAL_FEATURES].astype(str))
        if hasattr(Xc, "toarray"):
            Xc = Xc.toarray()
        return np.hstack([Xn, Xc])

    def query(self, X: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
        Xv = self.transform(X)
        dists, idxs = self.knn.kneighbors(Xv, n_neighbors=k)
        return dists, idxs


def fit_validator(
    df_full: pd.DataFrame,
    n_neighbors_default: int = 5,
    calib_sample: int = 5000,
    max_db_rows: int = 250_000,
) -> KNNValidator:
    # for very large DBs use a representative subsample for speed/memory (sufficient for distance judgment)
    if len(df_full) > max_db_rows:
        df_full = df_full.sample(n=max_db_rows, random_state=42)
    df_full = df_full.copy().reset_index(drop=True)
    scaler = StandardScaler().fit(df_full[NUMERIC_FEATURES].astype(float))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        df_full[CATEGORICAL_FEATURES].astype(str)
    )
    Xn = scaler.transform(df_full[NUMERIC_FEATURES].astype(float))
    Xc = ohe.transform(df_full[CATEGORICAL_FEATURES].astype(str))
    X_all = np.hstack([Xn, Xc])
    nn = NearestNeighbors(n_neighbors=n_neighbors_default + 1, metric="euclidean").fit(X_all)

    # ── adaptive threshold calibration ──
    # The DB contains very many identical design coordinates (duplicate points), so self-NN distances collapse to 0.
    # To measure the typical distance 'between distinct design combinations',
    # we measure nearest-neighbor (excluding self) distances on the de-duplicated coordinate set.
    X_unique = np.unique(X_all, axis=0)
    rng = np.random.RandomState(42)
    m = len(X_unique)
    sample_idx = rng.choice(m, size=min(calib_sample, m), replace=False)
    nn_unique = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X_unique)
    d_self, _ = nn_unique.kneighbors(X_unique[sample_idx], n_neighbors=2)
    nearest = d_self[:, 1]
    nearest = nearest[nearest > 1e-9]  # drop any remaining zero distances
    if nearest.size == 0:
        nearest = np.array([1.0])
    d_close = float(np.percentile(nearest, 50))
    d_far = float(np.percentile(nearest, 90))
    d_close = max(d_close, 1e-3)
    d_far = max(d_far, d_close * 1.5)

    return KNNValidator(
        scaler=scaler,
        ohe=ohe,
        knn=nn,
        y_train=df_full[TARGET].astype(float).values,
        feature_table=df_full,
        d_close=d_close,
        d_far=d_far,
        cap_std=float(df_full[TARGET].std()),
    )


# ─────────────────────────── verdict ───────────────────────────
def _verdict_single(
    pred: float,
    neigh_caps: np.ndarray,
    dists: np.ndarray,
    v: KNNValidator,
    cfg: dict[str, float],
) -> str:
    nm = float(neigh_caps.mean())
    nmx = float(neigh_caps.max())
    d = float(dists.mean())
    dmn = float(dists.min())

    # capacity gap scale is proportional to the standard deviation
    gap_strong = cfg["strong_gap_frac"] * v.cap_std
    gap_rec = cfg["rec_gap_frac"] * v.cap_std
    excess = cfg["halluc_excess_frac"] * v.cap_std

    echo_dist = cfg["echo_dist_frac"] * v.d_close

    if dmn <= echo_dist and abs(pred - nmx) <= cfg["echo_pred_gap"]:
        return "DB Echo"
    if d <= v.d_close and pred >= nmx - gap_strong:
        return "Strongly Recommended"
    if d <= v.d_far and pred >= nm - gap_rec:
        return "Recommended"
    if pred >= nmx + excess and d >= v.d_far * cfg["halluc_dist_mult"]:
        return "Hallucination"
    return "Uncertain"


def annotate(
    solutions: pd.DataFrame,
    validator: KNNValidator,
    k: int = 5,
    config: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add KNN validation result columns to the GA solutions DataFrame.

    Added columns:
    - neigh_mean / neigh_std / neigh_max : statistics of neighbors' measured capacity
    - neigh_dist_mean / neigh_dist_min   : distance statistics in the embedding space
    - verdict                            : verdict label
    """
    cfg = {
        "echo_dist_frac": 0.25,    # relative to d_close
        "echo_pred_gap": 2.0,      # mAh/g absolute
        "strong_gap_frac": 0.25,   # relative to cap_std
        "rec_gap_frac": 0.5,
        "halluc_excess_frac": 0.75,
        "halluc_dist_mult": 1.5,   # relative to d_far
    }
    if config:
        cfg.update(config)

    dists, idxs = validator.query(solutions, k=k)
    out = solutions.copy().reset_index(drop=True)

    neigh_caps_all = validator.y_train[idxs]
    out["neigh_mean"] = neigh_caps_all.mean(axis=1)
    out["neigh_std"] = neigh_caps_all.std(axis=1)
    out["neigh_max"] = neigh_caps_all.max(axis=1)
    out["neigh_dist_mean"] = dists.mean(axis=1)
    out["neigh_dist_min"] = dists.min(axis=1)

    pred_col = out["predicted_capacity"] if "predicted_capacity" in out else pd.Series([0.0] * len(out))
    verdicts = [
        _verdict_single(float(pred_col.iloc[i]), neigh_caps_all[i], dists[i], validator, cfg)
        for i in range(len(out))
    ]
    out["verdict"] = verdicts
    return out


def explain(row: pd.Series) -> str:
    """Human-readable explanation for a single solution."""
    pred = row.get("predicted_capacity", float("nan"))
    nm = row.get("neigh_mean", float("nan"))
    nmx = row.get("neigh_max", float("nan"))
    d = row.get("neigh_dist_mean", float("nan"))
    verdict = row.get("verdict", "")
    return (
        f"predicted={pred:.1f} mAh/g, neighbor mean={nm:.1f} (max={nmx:.1f}), "
        f"avg-dist={d:.2f} → {verdict}"
    )

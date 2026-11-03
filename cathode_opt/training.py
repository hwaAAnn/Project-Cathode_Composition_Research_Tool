"""Train 4 models + Optuna tuning + Top-3 ensemble builder.

Training procedure:
1. Train CatBoost / LightGBM / XGBoost / RandomForest
2. (optional) tune key hyperparameters with Optuna
3. Build the validation table (RMSE/MAE/R²) + training time / model size
4. Build the ensemble from the best-3 models -> verify ensemble.predict
5. Save all models, metrics, and the ensemble manifest in a per-chemistry directory
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from .config import (
    ARTIFACT_DIR,
    FEATURES,
    N_TRIALS_OPTUNA,
    RANDOM_STATE,
    SUBSAMPLE_FOR_TUNING,
)
from .data import Dataset
from .models import (
    BaseRegressor,
    CatBoostModel,
    EnsembleRegressor,
    LightGBMModel,
    Metrics,
    RandomForestModel,
    XGBoostModel,
    evaluate,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ───────────────────────── result container ──────────────────────
@dataclass
class TrainResult:
    name: str
    val_metrics: Metrics
    train_metrics: Metrics
    fit_seconds: float
    params: dict[str, Any]


# ───────────────────────── Optuna tuning ────────────────────────
def _subsample(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= n:
        return X, y
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def tune_catboost(ds: Dataset, n_trials: int) -> dict[str, Any]:
    Xt, yt = _subsample(ds.X_train, ds.y_train, SUBSAMPLE_FOR_TUNING)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations=1000,
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        )
        m = CatBoostModel(params=params).fit(Xt, yt, X_val=ds.X_val, y_val=ds.y_val)
        rmse = np.sqrt(mean_squared_error(ds.y_val, m.predict(ds.X_val)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(ds: Dataset, n_trials: int) -> dict[str, Any]:
    Xt, yt = _subsample(ds.X_train, ds.y_train, SUBSAMPLE_FOR_TUNING)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=1500,
            num_leaves=trial.suggest_int("num_leaves", 15, 200, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
        )
        m = LightGBMModel(params=params).fit(Xt, yt, X_val=ds.X_val, y_val=ds.y_val)
        rmse = np.sqrt(mean_squared_error(ds.y_val, m.predict(ds.X_val)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(ds: Dataset, n_trials: int) -> dict[str, Any]:
    Xt, yt = _subsample(ds.X_train, ds.y_train, SUBSAMPLE_FOR_TUNING)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=1500,
            max_depth=trial.suggest_int("max_depth", 4, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
        )
        m = XGBoostModel(params=params).fit(Xt, yt, X_val=ds.X_val, y_val=ds.y_val)
        rmse = np.sqrt(mean_squared_error(ds.y_val, m.predict(ds.X_val)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_random_forest(ds: Dataset, n_trials: int) -> dict[str, Any]:
    # RandomForest trains slowly, so use a smaller sample
    Xt, yt = _subsample(ds.X_train, ds.y_train, min(SUBSAMPLE_FOR_TUNING, 30_000))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        )
        m = RandomForestModel(params=params).fit(Xt, yt)
        rmse = np.sqrt(mean_squared_error(ds.y_val, m.predict(ds.X_val)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=min(n_trials, 10), show_progress_bar=False)
    return study.best_params


# ───────────────────────── training pipeline ────────────────────
def _fit_one(model: BaseRegressor, ds: Dataset) -> tuple[BaseRegressor, TrainResult]:
    t0 = time.time()
    if isinstance(model, RandomForestModel):
        model.fit(ds.X_train, ds.y_train)
    else:
        model.fit(ds.X_train, ds.y_train, X_val=ds.X_val, y_val=ds.y_val)
    fit_seconds = time.time() - t0

    train_pred = model.predict(ds.X_train)
    val_pred = model.predict(ds.X_val)
    res = TrainResult(
        name=model.name,
        val_metrics=evaluate(ds.y_val.values, val_pred),
        train_metrics=evaluate(ds.y_train.values, train_pred),
        fit_seconds=fit_seconds,
        params=dict(model.params),
    )
    return model, res


def train_all(
    ds: Dataset,
    out_dir: Path | None = None,
    n_trials: int = N_TRIALS_OPTUNA,
    tune: bool = True,
) -> dict[str, Any]:
    """Train 4 models -> save metrics -> create the best-3 ensemble."""
    out_dir = Path(out_dir or (ARTIFACT_DIR / ds.chemistry))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cathode_opt] === {ds.chemistry} training started (train={ds.n_train}, val={ds.n_val}) ===")

    tuned: dict[str, dict[str, Any]] = {}
    if tune:
        print("[cathode_opt] Tuning with Optuna...")
        t0 = time.time()
        tuned["catboost"] = tune_catboost(ds, n_trials)
        print(f"  · CatBoost best: {tuned['catboost']}")
        tuned["lightgbm"] = tune_lightgbm(ds, n_trials)
        print(f"  · LightGBM best: {tuned['lightgbm']}")
        tuned["xgboost"] = tune_xgboost(ds, n_trials)
        print(f"  · XGBoost  best: {tuned['xgboost']}")
        tuned["random_forest"] = tune_random_forest(ds, n_trials)
        print(f"  · RandomForest best: {tuned['random_forest']}")
        print(f"[cathode_opt] Tuning done: {time.time() - t0:.1f}s")

    models_init: list[BaseRegressor] = [
        CatBoostModel(params=tuned.get("catboost")),
        LightGBMModel(params=tuned.get("lightgbm")),
        XGBoostModel(params=tuned.get("xgboost")),
        RandomForestModel(params=tuned.get("random_forest")),
    ]

    results: list[TrainResult] = []
    trained: list[BaseRegressor] = []
    for m in models_init:
        print(f"[cathode_opt] Training {m.name}...")
        m_fit, res = _fit_one(m, ds)
        results.append(res)
        trained.append(m_fit)
        # save individual model
        m_fit.save(out_dir / f"{m.name}.pkl")
        print(
            f"  ✓ {m.name:14s} val RMSE={res.val_metrics.rmse:.3f}  R²={res.val_metrics.r2:.4f}  "
            f"({res.fit_seconds:.1f}s)"
        )

    # select best-3 (ascending RMSE)
    paired = sorted(zip(trained, results), key=lambda kv: kv[1].val_metrics.rmse)
    top3_models = [m for m, _ in paired[:3]]
    top3_metrics = [r.val_metrics for _, r in paired[:3]]
    top3_names = [r.name for _, r in paired[:3]]

    ensemble = EnsembleRegressor.from_val_metrics(top3_models, top3_metrics)
    ens_metrics = ensemble.evaluate(ds.X_val, ds.y_val)
    ensemble.save(out_dir / "ensemble")
    print(
        f"[cathode_opt] ensemble({', '.join(top3_names)}) val RMSE={ens_metrics.rmse:.3f}  R²={ens_metrics.r2:.4f}"
    )

    metrics_dump = {
        "chemistry": ds.chemistry,
        "n_train": ds.n_train,
        "n_val": ds.n_val,
        "models": [
            {
                "name": r.name,
                "val": r.val_metrics.as_dict(),
                "train": r.train_metrics.as_dict(),
                "fit_seconds": r.fit_seconds,
                "params": r.params,
            }
            for r in results
        ],
        "ensemble": {
            "members": top3_names,
            "weights": ensemble.weights,
            "val": ens_metrics.as_dict(),
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[cathode_opt] {ds.chemistry} done -> {out_dir}")
    return metrics_dump

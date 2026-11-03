"""Mixed-encoding genetic algorithm (mixed-integer + categorical).

Objective: maximize the ensemble-predicted discharge_capacity
- numeric variables: real encoding (low, high)
- categorical variables: index encoding (0..K-1, K = number of allowed_categories)
- fixed variables: always forced to their value

Operators:
- tournament selection (k=3)
- uniform crossover (applied separately to numeric / categorical)
- Gaussian mutation (numeric) + category resampling (categorical)
- elitism (keep best-N)
- to maintain diversity, near-duplicate solutions are removed from the pool (rounding-based key)

Implemented from scratch to avoid external dependencies such as scikit-opt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .bounds import SearchSpace
from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


@dataclass
class GAConfig:
    population_size: int = 200
    n_generations: int = 60
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_k: int = 3
    elite_frac: float = 0.05
    n_solutions: int = 10  # number of final solutions returned
    random_state: int = 42
    early_stop_patience: int = 15


@dataclass
class GAResult:
    solutions: pd.DataFrame  # FEATURES + predicted_capacity columns
    history: list[dict]
    elapsed_seconds: float
    config: GAConfig


# ───────────────────────── internal helpers ──────────────────────────
def _encode_population_to_df(
    pop_num: np.ndarray,
    pop_cat: np.ndarray,
    space: SearchSpace,
) -> pd.DataFrame:
    """GA genome -> DataFrame in FEATURES order."""
    rows = []
    for i in range(pop_num.shape[0]):
        rec: dict = {}
        ni = ci = 0
        for col in FEATURES:
            sp = space.specs[col]
            if sp.kind == "num":
                if sp.mode == "fixed":
                    rec[col] = float(sp.fixed_value)
                else:
                    rec[col] = float(pop_num[i, ni])
                    ni += 1
            else:
                if sp.mode == "fixed":
                    rec[col] = str(sp.fixed_value)
                else:
                    cats = sp.allowed_categories
                    idx = int(pop_cat[i, ci]) % len(cats)
                    rec[col] = cats[idx]
                    ci += 1
        rows.append(rec)
    return pd.DataFrame(rows)[FEATURES]


def _init_population(
    space: SearchSpace, n_pop: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    num_vars = space.search_numeric_vars()
    cat_vars = space.search_categorical_vars()

    if num_vars:
        lows = np.array([space.specs[v].low for v in num_vars])
        highs = np.array([space.specs[v].high for v in num_vars])
        pop_num = rng.uniform(lows, highs, size=(n_pop, len(num_vars)))
    else:
        pop_num = np.zeros((n_pop, 0))

    if cat_vars:
        sizes = np.array([len(space.specs[v].allowed_categories) for v in cat_vars])
        pop_cat = (rng.random((n_pop, len(cat_vars))) * sizes).astype(int)
    else:
        pop_cat = np.zeros((n_pop, 0), dtype=int)

    return pop_num, pop_cat, num_vars, cat_vars


def _tournament(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmax(fitness[idx])])


def _crossover(
    a_num: np.ndarray,
    a_cat: np.ndarray,
    b_num: np.ndarray,
    b_cat: np.ndarray,
    rng: np.random.Generator,
    prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover."""
    child_num = a_num.copy()
    child_cat = a_cat.copy()
    if a_num.size:
        mask = rng.random(a_num.size) < prob
        child_num[mask] = b_num[mask]
    if a_cat.size:
        mask = rng.random(a_cat.size) < prob
        child_cat[mask] = b_cat[mask]
    return child_num, child_cat


def _mutate(
    num: np.ndarray,
    cat: np.ndarray,
    num_vars: list[str],
    cat_vars: list[str],
    space: SearchSpace,
    rng: np.random.Generator,
    prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    # numeric: Gaussian mutation + clip
    for j, v in enumerate(num_vars):
        if rng.random() < prob:
            lo, hi = space.specs[v].low, space.specs[v].high
            sigma = 0.1 * (hi - lo) if hi > lo else 1e-6
            num[j] = float(np.clip(num[j] + rng.normal(0, sigma), lo, hi))
    # categorical: resample
    for j, v in enumerate(cat_vars):
        if rng.random() < prob:
            n_cat = len(space.specs[v].allowed_categories)
            cat[j] = int(rng.integers(0, n_cat))
    return num, cat


def _diversify_keys(df: pd.DataFrame, decimals: int = 3) -> list[str]:
    """Rounding-based key for identifying near-identical solutions."""
    keys = []
    for _, row in df.iterrows():
        parts = []
        for c in FEATURES:
            v = row[c]
            if c in NUMERIC_FEATURES:
                parts.append(f"{float(v):.{decimals}f}")
            else:
                parts.append(str(v))
        keys.append("|".join(parts))
    return keys


# ───────────────────────── main loop ──────────────────────────
def run_ga(
    space: SearchSpace,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    config: GAConfig | None = None,
    verbose: bool = True,
    on_generation: Callable[[dict], None] | None = None,
) -> GAResult:
    """General-purpose GA run.

    Parameters
    ----------
    space : SearchSpace
        search region
    predict_fn : (df) -> np.ndarray
        function that returns discharge_capacity predictions for a given DataFrame
        (usually ensemble.predict)
    on_generation : callable, optional
        callback invoked every generation -- for live streaming. Takes a dict argument
        ({"generation", "best", "mean", "median", "total_generations"}).
    """
    cfg = config or GAConfig()
    rng = np.random.default_rng(cfg.random_state)
    t0 = time.time()

    pop_num, pop_cat, num_vars, cat_vars = _init_population(space, cfg.population_size, rng)

    history: list[dict] = []
    best_global_score = -np.inf
    best_global_num: np.ndarray | None = None
    best_global_cat: np.ndarray | None = None
    stale = 0

    for gen in range(cfg.n_generations):
        df_pop = _encode_population_to_df(pop_num, pop_cat, space)
        fitness = np.asarray(predict_fn(df_pop), dtype=float)

        best_idx = int(np.argmax(fitness))
        gen_best = float(fitness[best_idx])
        gen_entry = {
            "generation": gen,
            "best": gen_best,
            "mean": float(np.mean(fitness)),
            "median": float(np.median(fitness)),
        }
        history.append(gen_entry)
        if on_generation is not None:
            try:
                on_generation({**gen_entry, "total_generations": cfg.n_generations})
            except Exception:
                pass
        if verbose and gen % 5 == 0:
            print(f"  · gen {gen:3d}  best={gen_best:.3f}  mean={np.mean(fitness):.3f}")

        if gen_best > best_global_score + 1e-4:
            best_global_score = gen_best
            best_global_num = pop_num[best_idx].copy()
            best_global_cat = pop_cat[best_idx].copy()
            stale = 0
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                if verbose:
                    print(f"  ✓ early stop @ gen {gen} (no improvement for {cfg.early_stop_patience})")
                break

        # elitism
        n_elite = max(1, int(cfg.elite_frac * cfg.population_size))
        elite_idx = np.argsort(-fitness)[:n_elite]

        # create next generation
        next_num = np.zeros_like(pop_num)
        next_cat = np.zeros_like(pop_cat)
        next_num[:n_elite] = pop_num[elite_idx]
        next_cat[:n_elite] = pop_cat[elite_idx]

        for i in range(n_elite, cfg.population_size):
            a = _tournament(fitness, cfg.tournament_k, rng)
            b = _tournament(fitness, cfg.tournament_k, rng)
            if rng.random() < cfg.crossover_prob:
                c_num, c_cat = _crossover(pop_num[a], pop_cat[a], pop_num[b], pop_cat[b], rng, 0.5)
            else:
                c_num, c_cat = pop_num[a].copy(), pop_cat[a].copy()
            c_num, c_cat = _mutate(c_num, c_cat, num_vars, cat_vars, space, rng, cfg.mutation_prob)
            next_num[i] = c_num
            next_cat[i] = c_cat

        pop_num, pop_cat = next_num, next_cat

    # final evaluation
    df_pop = _encode_population_to_df(pop_num, pop_cat, space)
    fitness = np.asarray(predict_fn(df_pop), dtype=float)

    df_pop["predicted_capacity"] = fitness
    # remove duplicates (near-identical solutions), then take the top n_solutions
    df_pop["__key"] = _diversify_keys(df_pop.drop(columns="predicted_capacity"), decimals=3)
    df_pop = df_pop.sort_values("predicted_capacity", ascending=False).drop_duplicates("__key")
    df_pop = df_pop.drop(columns="__key").head(cfg.n_solutions).reset_index(drop=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✓ GA done in {elapsed:.1f}s, top capacity={df_pop['predicted_capacity'].max():.2f}")

    return GAResult(solutions=df_pop, history=history, elapsed_seconds=elapsed, config=cfg)

"""Global configuration.

- 22 design-variable schema (user spec)
- target column
- paths
- chemistry catalog
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# ───────────────────────────── paths ────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
TRAIN_DIR: Final[Path] = PROJECT_ROOT / "Training"
VAL_DIR: Final[Path] = PROJECT_ROOT / "Validation"
ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "artifacts"

# ─────────────────────────── target / variables ───────────────────────
TARGET: Final[str] = "discharge_capacity (mAh/g)"

CATEGORICAL_FEATURES: Final[list[str]] = [
    "material_structure",
    "synthesis_method",
    "Li_source",
    "Co_source",
    "Mn_source",
    "Ni_source",
    "electrolyte",
    "counter_electrode",
    "separator",
]

NUMERIC_FEATURES: Final[list[str]] = [
    "sintering_T1(C)",
    "sintering_t1(h)",
    "Li_fraction",
    "Ni_fraction",
    "Mn_fraction",
    "Co_fraction",
    "dopant_fraction",
    "active_proportion",
    "binder_proportion",
    "particle_size(um)",
    "C-rate",
    "voltage_range(V)_min",
    "voltage_range(V)_max",
]

FEATURES: Final[list[str]] = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # 9 + 13 = 22

# ─────────────────────────── chemistry catalog ─────────────────────
CHEMISTRIES: Final[tuple[str, ...]] = ("LFP", "NCA")  # trainable categories

# One train file per chemistry; same columns but different distributions
TRAIN_FILES: Final[dict[str, str]] = {
    "LFP": "LFP_train_dataset.csv",
    "NCA": "NCA_train_dataset.csv",
}
VAL_FILES: Final[dict[str, str]] = {
    "LFP": "LFP_val_dataset.csv",
    "NCA": "NCA_val_dataset.csv",
}

# ───────────────────────── training hyperparameters ─────────────────
RANDOM_STATE: Final[int] = 42
N_TRIALS_OPTUNA: Final[int] = 20      # tuning trials per model
SUBSAMPLE_FOR_TUNING: Final[int] = 80_000  # sample size to speed up tuning on large datasets (NCA)

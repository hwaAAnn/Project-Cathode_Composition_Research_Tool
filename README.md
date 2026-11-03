# Project-Cathode-Composition-Dev-Tool

A research-support tool for **lithium-ion battery cathode materials**. From 22 design variables known before cell fabrication, it predicts discharge capacity, explores promising material/process combinations, and checks each candidate against measured data so that unreliable extrapolations can be spotted before running experiments.

Developed as a university capstone design project (Systems Management Engineering).

## Approach

1. **Prediction** – regression ensemble (CatBoost, LightGBM, XGBoost, Random Forest) for `discharge_capacity (mAh/g)`. The three models with the lowest validation RMSE are combined with inverse-RMSE weights.
2. **Composition exploration** – a mixed-encoding genetic algorithm (real-valued and categorical genes) maximizes the predicted capacity. Variables can be fixed by the user; the search bounds of the remaining variables are derived from a capacity-percentile slice of the data.
3. **Reliability check** – each candidate is compared with its K nearest measured samples (standardized numeric + one-hot categorical space, with distance thresholds calibrated on the dataset itself) and labeled *Strongly Recommended*, *Recommended* or *Hallucination*. This guards against the surrogate model overestimating designs far from the data it was trained on.

Two chemistries are covered: **LFP** and **NCA**.

## Results

Evaluated on a held-out validation split:

| Chemistry | Train rows | Val rows | R² | RMSE (mAh/g) | MAE (mAh/g) |
|-----------|-----------:|---------:|------:|------:|------:|
| LFP       | 131,790    | 16,474   | 0.721 | 12.58 | 3.87  |
| NCA       | 769,225    | 96,153   | 0.959 | 6.87  | 2.06  |

Per-model metrics and tuned hyperparameters are in `artifacts/<CHEM>/metrics.json`.

## Input variables (22)

- **Categorical (9):** `material_structure`, `synthesis_method`, `Li_source`, `Co_source`, `Mn_source`, `Ni_source`, `electrolyte`, `counter_electrode`, `separator`
- **Numeric (13):** `sintering_T1(C)`, `sintering_t1(h)`, `Li_fraction`, `Ni_fraction`, `Mn_fraction`, `Co_fraction`, `dopant_fraction`, `active_proportion`, `binder_proportion`, `particle_size(um)`, `C-rate`, `voltage_range(V)_min`, `voltage_range(V)_max`
- **Target:** `discharge_capacity (mAh/g)`

## Repository layout

```
cathode_tool/
    config.py          # variable schema, paths, constants
    data.py            # loading and cleaning
    features.py        # per-model categorical encoding
    models/            # CatBoost, LightGBM, XGBoost, RandomForest, ensemble
    training.py        # training, Optuna tuning, best-3 ensemble builder
    bounds.py          # data-driven search bounds
    ga.py              # genetic algorithm
    validation.py      # KNN-based reliability labels
    pipeline.py        # end-to-end orchestration
artifacts/
    LFP/  NCA/         # trained models, ensemble manifests, metrics.json
Training/  Validation/ # dataset CSVs go here (not included, see below)
```

## Dataset

The data come from a battery-materials dataset published on **[AI Hub](https://www.aihub.or.kr)** (Korea). **The original data are not redistributed in this repository**; this project only uses the dataset. To reproduce the results, obtain it from AI Hub.

The pipeline expects, per chemistry, a training and a validation CSV containing the 22 variables above plus the target column (reduced from 48 source columns):

```
Training/LFP_train_dataset.csv
Training/NCA_train_dataset.csv
Validation/LFP_val_dataset.csv
Validation/NCA_val_dataset.csv
```

## Usage

```bash
pip install -r requirements.txt
```

```python
from cathode_tool.pipeline import quick_optimize

# Search NCA compositions with the Ni fraction held fixed
result = quick_optimize("NCA", fixed={"Ni_fraction": 0.8})
for sol in result["solutions"][:3]:
    print(sol["predicted_capacity"], sol["verdict"])
```

Loading a pipeline requires the dataset CSVs above (used for search bounds and KNN reference data). Retraining is done through `cathode_tool.training`.

## Notes

- Files in `artifacts/` are Python pickles; only load pickles from sources you trust.
- Predictions are for research purposes and do not replace experimental validation.

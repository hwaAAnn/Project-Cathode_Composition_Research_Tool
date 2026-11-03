"""Research-support toolkit for cathode material performance prediction and composition exploration.

Subpackages:
- data       : data loading / cleaning
- features   : design-variable schema / encoding
- models     : CatBoost / LightGBM / XGBoost / RandomForest + Ensemble
- training   : train 4 models in one go + tuning + evaluation
- bounds     : data-driven dynamic search bounds
- ga         : mixed-encoding genetic algorithm
- validation : KNN-based validation (Hallucination / Recommended / Strongly Recommended)
- pipeline   : end-to-end orchestration
"""

__version__ = "0.1.0"

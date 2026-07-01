"""
Gradient Boosting Classifiers
===============================
XGBoost, LightGBM, CatBoost, and Stacking Ensemble.
"""

__all__ = [
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "GradientBoostEnsemble",
]

def __getattr__(name):
    if name in __all__:
        from src.models.classifiers.gradient_boost import (
            XGBoostModel, LightGBMModel, CatBoostModel, GradientBoostEnsemble
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

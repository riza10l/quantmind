"""
QuantMind ML Models
=====================
Machine learning models for market prediction, regime detection,
and explainability.

Modules:
    - classifiers: Gradient boosting (XGBoost, LightGBM, CatBoost)
    - regime: HMM + clustering market regime detection
    - explainability: SHAP-based per-prediction explanations
    - trainer: Orchestrator for training pipeline
    - cross_validation: Purged K-fold + walk-forward CV
    - leaderboard: MLflow-based model ranking
"""

from src.models.base import BaseModel, ModelMetrics

__all__ = [
    "BaseModel",
    "ModelMetrics",
]

"""
ML Model Base Class (Fase 2)
=============================
Abstract interface for all ML models in the QuantMind system.

All models must implement:
- train(X, y) → fit the model
- predict(X) → generate predictions
- evaluate(X, y) → compute metrics
- explain(X) → SHAP/LIME explanations

TODO (Fase 2):
- [ ] Implement XGBoost, LightGBM, CatBoost classifiers
- [ ] Implement LSTM, Transformer deep learning models
- [ ] Implement Temporal Fusion Transformer (TFT)
- [ ] Auto hyperparameter tuning with Optuna
- [ ] Model leaderboard via MLflow
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelMetrics:
    """Standard metrics for model evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    custom: dict[str, float] = field(default_factory=dict)


class BaseModel(abc.ABC):
    """Abstract base class for all QuantMind ML models."""

    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = params or {}
        self.is_fitted = False
        self.training_timestamp: Optional[datetime] = None
        self.feature_names: list[str] = []

    @abc.abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> ModelMetrics:
        """Train the model on the given data."""
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        ...

    @abc.abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model performance."""
        ...

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Generate SHAP/LIME explanations (Module 13)."""
        raise NotImplementedError("Explainability not implemented yet (Fase 2)")

    def save(self, path: str) -> None:
        """Save model to disk."""
        raise NotImplementedError("Model serialization not implemented yet")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk."""
        raise NotImplementedError("Model loading not implemented yet")

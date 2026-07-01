"""
Model Explainability (XAI)
============================
Per-prediction SHAP explanations + human-readable trade narratives.

Every trading signal should be explainable:
"Long BTC because RSI is oversold (28.5), fear index is extreme (12/100),
and 50-day momentum turned positive — top 3 signal contributors: [RSI: 0.34,
Fear_Greed: 0.22, momentum_50: 0.18]"

Uses:
- SHAP TreeExplainer for gradient boosting models (fast, exact)
- SHAP KernelExplainer as fallback for any model
- LIME for local, tabular explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("models.explainability")


@dataclass
class Explanation:
    """A human-readable explanation for a single prediction."""
    prediction: int  # 0 = short/hold, 1 = long
    confidence: float  # Probability of the predicted class
    top_features: list[dict[str, Any]]  # [{name, value, shap_value, contribution}]
    narrative: str  # Human-readable sentence
    raw_shap_values: np.ndarray | None = None


@dataclass
class GlobalExplanation:
    """Global feature importance across the entire dataset."""
    feature_importance: pd.DataFrame  # Columns: feature, importance, rank
    top_features: list[str]
    summary_plot_data: dict[str, Any] = field(default_factory=dict)


class ModelExplainer:
    """
    Unified explainability interface for any QuantMind model.

    Supports:
    - SHAP TreeExplainer (XGBoost, LightGBM, CatBoost)
    - SHAP KernelExplainer (any black-box model)
    - LIME TabularExplainer

    Usage:
        explainer = ModelExplainer(model)
        explanation = explainer.explain_prediction(X.iloc[0])
        print(explanation.narrative)
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        method: str = "shap",
    ) -> None:
        """
        Args:
            model: Any fitted model with predict/predict_proba methods.
            feature_names: Feature names for labeling.
            method: Explanation method ('shap' or 'lime').
        """
        self.model = model
        self.feature_names = feature_names or []
        self.method = method
        self._explainer = None

    def _get_raw_model(self) -> Any:
        """Extract the underlying sklearn/xgboost/lightgbm model."""
        if hasattr(self.model, "_model") and self.model._model is not None:
            return self.model._model
        return self.model

    def _init_shap_explainer(self, X_background: pd.DataFrame | None = None) -> None:
        """Initialize the SHAP explainer."""
        import shap

        raw_model = self._get_raw_model()
        model_type = type(raw_model).__name__

        # Tree-based models get the fast TreeExplainer
        tree_models = {"XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
                       "XGBRegressor", "LGBMRegressor", "RandomForestClassifier"}

        if model_type in tree_models:
            self._explainer = shap.TreeExplainer(raw_model)
            logger.info("shap_tree_explainer_initialized", model_type=model_type)
        else:
            # Fallback to KernelExplainer
            if X_background is None:
                raise ValueError("KernelExplainer requires background data (X_background)")
            background = shap.kmeans(X_background, 50)
            self._explainer = shap.KernelExplainer(
                raw_model.predict_proba, background
            )
            logger.info("shap_kernel_explainer_initialized", model_type=model_type)

    def explain_prediction(
        self,
        X_single: pd.Series | pd.DataFrame,
        top_k: int = 5,
        X_background: pd.DataFrame | None = None,
    ) -> Explanation:
        """
        Generate a human-readable explanation for a single prediction.

        Args:
            X_single: Feature values for one sample.
            top_k: Number of top contributing features to include.
            X_background: Background data (required for KernelExplainer).

        Returns:
            Explanation with narrative, top features, and SHAP values.
        """
        import shap

        if isinstance(X_single, pd.Series):
            X_single = X_single.to_frame().T

        # Initialize explainer if needed
        if self._explainer is None:
            self._init_shap_explainer(X_background)

        # Get SHAP values
        raw_model = self._get_raw_model()

        # Handle scaler if model has one
        if hasattr(self.model, "_scaler"):
            X_scaled = self.model._scaler.transform(X_single)
            shap_values = self._explainer.shap_values(X_scaled)
        else:
            shap_values = self._explainer.shap_values(X_single.values)

        # For binary classification, SHAP returns list of [class_0, class_1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Class 1 (long) explanations
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]  # (samples, features, classes)
        else:
            sv = shap_values[0]

        # Get prediction
        pred = self.model.predict(X_single)[0]
        proba = self.model.predict_proba(X_single)[0]
        confidence = float(proba[int(pred)])

        # Get feature names
        if self.feature_names:
            names = self.feature_names
        elif hasattr(self.model, "feature_names"):
            names = self.model.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(sv))]

        # Build top features
        abs_sv = np.abs(sv)
        top_indices = np.argsort(abs_sv)[-top_k:][::-1]

        top_features = []
        for idx in top_indices:
            top_features.append({
                "name": names[idx] if idx < len(names) else f"feature_{idx}",
                "value": float(X_single.iloc[0, idx]),
                "shap_value": float(sv[idx]),
                "contribution": "bullish" if sv[idx] > 0 else "bearish",
            })

        # Generate narrative
        narrative = self._generate_narrative(pred, confidence, top_features)

        return Explanation(
            prediction=int(pred),
            confidence=confidence,
            top_features=top_features,
            narrative=narrative,
            raw_shap_values=sv,
        )

    def explain_global(self, X: pd.DataFrame) -> GlobalExplanation:
        """
        Compute global feature importance across the dataset.

        Args:
            X: Full feature matrix.

        Returns:
            GlobalExplanation with importance rankings.
        """
        import shap

        if self._explainer is None:
            self._init_shap_explainer(X)

        if hasattr(self.model, "_scaler"):
            X_scaled = self.model._scaler.transform(X)
        else:
            X_scaled = X.values

        shap_values = self._explainer.shap_values(X_scaled)

        # Handle binary classification
        if isinstance(shap_values, list):
            sv = np.abs(shap_values[1])
        elif shap_values.ndim == 3:
            sv = np.abs(shap_values[:, :, 1])
        else:
            sv = np.abs(shap_values)

        mean_importance = sv.mean(axis=0)

        names = (
            self.feature_names
            or getattr(self.model, "feature_names", [])
            or [f"feature_{i}" for i in range(len(mean_importance))]
        )

        importance_df = pd.DataFrame({
            "feature": names[:len(mean_importance)],
            "importance": mean_importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        top_features = importance_df["feature"].head(20).tolist()

        logger.info(
            "global_explanation_computed",
            n_features=len(importance_df),
            top_3=top_features[:3],
        )

        return GlobalExplanation(
            feature_importance=importance_df,
            top_features=top_features,
        )

    def _generate_narrative(
        self,
        prediction: int,
        confidence: float,
        top_features: list[dict[str, Any]],
    ) -> str:
        """Generate a human-readable trade narrative."""
        direction = "LONG" if prediction == 1 else "SHORT/HOLD"

        # Build feature descriptions
        reasons = []
        for feat in top_features[:3]:
            name = feat["name"]
            value = feat["value"]
            contribution = feat["contribution"]

            # Format value nicely
            if abs(value) > 100:
                val_str = f"{value:.0f}"
            elif abs(value) > 1:
                val_str = f"{value:.2f}"
            else:
                val_str = f"{value:.4f}"

            reasons.append(f"{name}={val_str} ({contribution})")

        reasons_str = ", ".join(reasons)
        shap_str = " | ".join(
            [f"{f['name']}: {f['shap_value']:.3f}" for f in top_features[:3]]
        )

        narrative = (
            f"Signal: {direction} (confidence: {confidence:.1%}). "
            f"Key drivers: {reasons_str}. "
            f"SHAP contributions: [{shap_str}]"
        )

        return narrative

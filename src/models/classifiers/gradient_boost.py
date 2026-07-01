"""
Gradient Boosting Classifiers
===============================
XGBoost, LightGBM, and CatBoost classifiers for market direction prediction,
with Optuna hyperparameter optimization, purged K-fold CV, and MLflow logging.

Usage:
    from src.models.classifiers.gradient_boost import (
        XGBoostModel, LightGBMModel, CatBoostModel, GradientBoostEnsemble
    )

    model = XGBoostModel(params={"max_depth": 6})
    metrics = model.train(X_train, y_train)
    predictions = model.predict(X_test)

    # Auto-tuned
    model = XGBoostModel()
    model.auto_tune(X_train, y_train, n_trials=50)
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.core.logger import get_logger
from src.models.base import BaseModel, ModelMetrics
from src.models.cross_validation import PurgedKFold

logger = get_logger("models.gradient_boost")


# ============================================================
# XGBoost
# ============================================================


class XGBoostModel(BaseModel):
    """XGBoost classifier for market direction prediction."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        if params:
            default_params.update(params)
        super().__init__(name="xgboost", params=default_params)
        self._model = None
        self._scaler = StandardScaler()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> ModelMetrics:
        from xgboost import XGBClassifier

        self.feature_names = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)

        self._model = XGBClassifier(**self.params)

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scaler.transform(X_val)
            fit_params["eval_set"] = [(X_val_scaled, y_val)]
            fit_params["verbose"] = False

        self._model.fit(X_scaled, y, **fit_params)
        self.is_fitted = True
        self.training_timestamp = datetime.now(timezone.utc)

        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not fitted"
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        preds = self.predict(X)
        proba = self.predict_proba(X)

        return ModelMetrics(
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1_score=float(f1_score(y, preds, zero_division=0)),
            auc_roc=float(roc_auc_score(y, proba[:, 1])) if len(np.unique(y)) > 1 else 0.0,
        )

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from the fitted model."""
        assert self._model is not None, "Model not fitted"
        importance = self._model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

    def auto_tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        cv_splits: int = 5,
        timeout: int | None = 600,
    ) -> dict[str, Any]:
        """Auto-tune hyperparameters with Optuna."""
        import optuna
        from xgboost import XGBClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X_scaled = self._scaler.fit_transform(X)
        cv = PurgedKFold(n_splits=cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "eval_metric": "logloss",
            }

            scores = []
            for train_idx, val_idx in cv.split(X_scaled):
                model = XGBClassifier(**params)
                model.fit(X_scaled[train_idx], y.iloc[train_idx], verbose=False)
                preds = model.predict(X_scaled[val_idx])
                scores.append(f1_score(y.iloc[val_idx], preds, zero_division=0))

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", study_name="xgboost_tune")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.params.update(study.best_params)
        logger.info("xgboost_tuned", best_score=study.best_value, best_params=study.best_params)

        # Re-train with best params
        self.train(X, y)

        return {"best_score": study.best_value, "best_params": study.best_params}

    def save(self, path: str) -> None:
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "params": self.params,
            "feature_names": self.feature_names,
            "training_timestamp": self.training_timestamp,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("model_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(params=data["params"])
        model._model = data["model"]
        model._scaler = data["scaler"]
        model.feature_names = data["feature_names"]
        model.training_timestamp = data["training_timestamp"]
        model.is_fitted = True
        return model


# ============================================================
# LightGBM
# ============================================================


class LightGBMModel(BaseModel):
    """LightGBM classifier for market direction prediction."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 10,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        super().__init__(name="lightgbm", params=default_params)
        self._model = None
        self._scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> ModelMetrics:
        from lightgbm import LGBMClassifier

        self.feature_names = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)

        self._model = LGBMClassifier(**self.params)
        self._model.fit(X_scaled, y)
        self.is_fitted = True
        self.training_timestamp = datetime.now(timezone.utc)

        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(self._scaler.transform(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        preds = self.predict(X)
        proba = self.predict_proba(X)
        return ModelMetrics(
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1_score=float(f1_score(y, preds, zero_division=0)),
            auc_roc=float(roc_auc_score(y, proba[:, 1])) if len(np.unique(y)) > 1 else 0.0,
        )

    def get_feature_importance(self) -> pd.Series:
        assert self._model is not None
        return pd.Series(
            self._model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

    def auto_tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50, **kwargs) -> dict:
        import optuna
        from lightgbm import LGBMClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X_scaled = self._scaler.fit_transform(X)
        cv = PurgedKFold(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": 42, "n_jobs": -1, "verbose": -1,
            }
            scores = []
            for train_idx, val_idx in cv.split(X_scaled):
                model = LGBMClassifier(**params)
                model.fit(X_scaled[train_idx], y.iloc[train_idx])
                preds = model.predict(X_scaled[val_idx])
                scores.append(f1_score(y.iloc[val_idx], preds, zero_division=0))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=kwargs.get("timeout", 600))
        self.params.update(study.best_params)
        self.train(X, y)
        return {"best_score": study.best_value, "best_params": study.best_params}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "params": self.params, "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(params=data["params"])
        m._model, m._scaler = data["model"], data["scaler"]
        m.feature_names = data["feature_names"]
        m.is_fitted = True
        return m


# ============================================================
# CatBoost
# ============================================================


class CatBoostModel(BaseModel):
    """CatBoost classifier for market direction prediction."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
            "thread_count": -1,
        }
        if params:
            default_params.update(params)
        super().__init__(name="catboost", params=default_params)
        self._model = None
        self._scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> ModelMetrics:
        from catboost import CatBoostClassifier

        self.feature_names = list(X.columns)
        X_scaled = self._scaler.fit_transform(X)

        self._model = CatBoostClassifier(**self.params)
        self._model.fit(X_scaled, y)
        self.is_fitted = True
        self.training_timestamp = datetime.now(timezone.utc)
        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._scaler.transform(X)).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(self._scaler.transform(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        preds = self.predict(X)
        proba = self.predict_proba(X)
        return ModelMetrics(
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1_score=float(f1_score(y, preds, zero_division=0)),
            auc_roc=float(roc_auc_score(y, proba[:, 1])) if len(np.unique(y)) > 1 else 0.0,
        )

    def get_feature_importance(self) -> pd.Series:
        assert self._model is not None
        return pd.Series(
            self._model.get_feature_importance(), index=self.feature_names
        ).sort_values(ascending=False)

    def auto_tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50, **kwargs) -> dict:
        import optuna
        from catboost import CatBoostClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X_scaled = self._scaler.fit_transform(X)
        cv = PurgedKFold(n_splits=5)

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "random_seed": 42, "verbose": 0, "thread_count": -1,
            }
            scores = []
            for train_idx, val_idx in cv.split(X_scaled):
                model = CatBoostClassifier(**params)
                model.fit(X_scaled[train_idx], y.iloc[train_idx])
                preds = model.predict(X_scaled[val_idx]).flatten()
                scores.append(f1_score(y.iloc[val_idx], preds, zero_division=0))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=kwargs.get("timeout", 600))
        self.params.update(study.best_params)
        self.train(X, y)
        return {"best_score": study.best_value, "best_params": study.best_params}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "params": self.params, "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path: str) -> "CatBoostModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(params=data["params"])
        m._model, m._scaler = data["model"], data["scaler"]
        m.feature_names = data["feature_names"]
        m.is_fitted = True
        return m


# ============================================================
# Ensemble (Stacking)
# ============================================================


class GradientBoostEnsemble(BaseModel):
    """
    Stacking ensemble of XGBoost + LightGBM + CatBoost.

    Each base model produces probabilities; a meta-learner
    (logistic regression) combines them for the final prediction.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(name="gb_ensemble", params=params or {})
        self._base_models: list[BaseModel] = []
        self._meta_model = None
        self._scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> ModelMetrics:
        from sklearn.linear_model import LogisticRegression

        self.feature_names = list(X.columns)

        # Train base models
        model_classes = [XGBoostModel, LightGBMModel]
        try:
            import catboost  # noqa
            model_classes.append(CatBoostModel)
        except ImportError:
            logger.info("catboost_not_available_for_ensemble")

        self._base_models = []
        meta_features = np.zeros((len(X), len(model_classes)))

        cv = PurgedKFold(n_splits=5)

        for i, model_cls in enumerate(model_classes):
            model = model_cls(params=self.params.get(model_cls.__name__.lower(), {}))

            # Generate out-of-fold predictions for meta-learner
            oof_preds = np.zeros(len(X))
            for train_idx, val_idx in cv.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]

                fold_model = model_cls()
                fold_model.train(X_train_fold, y_train_fold)
                oof_preds[val_idx] = fold_model.predict_proba(X_val_fold)[:, 1]

            meta_features[:, i] = oof_preds

            # Train on full data for final model
            model.train(X, y)
            self._base_models.append(model)

        # Train meta-learner
        meta_scaled = self._scaler.fit_transform(meta_features)
        self._meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self._meta_model.fit(meta_scaled, y)

        self.is_fitted = True
        self.training_timestamp = datetime.now(timezone.utc)

        logger.info("ensemble_trained", n_base_models=len(self._base_models))
        return self.evaluate(X, y)

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        meta = np.zeros((len(X), len(self._base_models)))
        for i, model in enumerate(self._base_models):
            meta[:, i] = model.predict_proba(X)[:, 1]
        return self._scaler.transform(meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._meta_model is not None
        return self._meta_model.predict(self._get_meta_features(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._meta_model is not None
        return self._meta_model.predict_proba(self._get_meta_features(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        preds = self.predict(X)
        proba = self.predict_proba(X)
        return ModelMetrics(
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1_score=float(f1_score(y, preds, zero_division=0)),
            auc_roc=float(roc_auc_score(y, proba[:, 1])) if len(np.unique(y)) > 1 else 0.0,
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "base_models": [(m.name, m.params) for m in self._base_models],
                "meta_model": self._meta_model,
                "scaler": self._scaler,
            }, f)

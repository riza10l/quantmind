"""
ML Training Pipeline
======================
Orchestrates the entire training workflow:
  1. Load features + targets
  2. Split data (train/val/test with purged CV)
  3. Train models (XGBoost, LightGBM, CatBoost, Ensemble)
  4. Optuna hyperparameter tuning
  5. Evaluate on hold-out test set
  6. Generate SHAP explanations
  7. Log everything to MLflow
  8. Detect market regimes

Usage:
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer(config)
    results = trainer.run(symbol="BTC/USDT", timeframe="1d")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import AppConfig, load_config
from src.core.database import DatabaseManager
from src.core.events import EventTypes, event_bus
from src.core.logger import get_logger
from src.features.store import FeatureStore
from src.models.base import BaseModel, ModelMetrics
from src.models.cross_validation import PurgedKFold, WalkForwardSplit

logger = get_logger("models.trainer")


@dataclass
class TrainingResult:
    """Complete results from a training run."""
    symbol: str
    timeframe: str
    model_name: str
    train_metrics: ModelMetrics
    test_metrics: ModelMetrics
    best_params: dict[str, Any]
    feature_importance: Optional[pd.Series] = None
    n_features: int = 0
    n_train_samples: int = 0
    n_test_samples: int = 0
    training_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelTrainer:
    """
    Orchestrates ML model training, tuning, and evaluation.

    Integrates with FeatureStore, Optuna, MLflow, and the Explainability module.

    Args:
        config: Application configuration.
        db: DatabaseManager instance (optional, will create from config).
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        db: DatabaseManager | None = None,
    ) -> None:
        self.config = config or load_config()
        self.db = db or DatabaseManager(self.config.database.url)
        self.db.initialize()
        self.feature_store = FeatureStore(self.config, self.db)
        self._results: list[TrainingResult] = []

    def prepare_data(
        self,
        symbol: str,
        timeframe: str = "1d",
        target_horizon: int = 1,
        target_type: str = "direction",
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test splits from the feature store.

        Uses chronological split (not random) to avoid look-ahead bias.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X, y = self.feature_store.get_features_with_target(
            symbol, timeframe,
            target_horizon=target_horizon,
            target_type=target_type,
        )

        if X.empty:
            raise ValueError(f"No features found for {symbol}/{timeframe}. Run feature computation first.")

        # Chronological split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Drop any remaining NaN
        train_mask = X_train.notna().all(axis=1) & y_train.notna()
        test_mask = X_test.notna().all(axis=1) & y_test.notna()
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]

        logger.info(
            "data_prepared",
            symbol=symbol,
            n_features=X_train.shape[1],
            n_train=len(X_train),
            n_test=len(X_test),
            target_balance=f"{y_train.mean():.2%}",
        )

        return X_train, X_test, y_train, y_test

    def train_single_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        auto_tune: bool = True,
        n_trials: int = 50,
        symbol: str = "",
        timeframe: str = "",
    ) -> TrainingResult:
        """
        Train a single model with optional Optuna tuning.

        Args:
            model_type: 'xgboost', 'lightgbm', 'catboost', 'ensemble'
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            auto_tune: Whether to run Optuna hyperparameter search.
            n_trials: Number of Optuna trials.

        Returns:
            TrainingResult with metrics and feature importance.
        """
        import time

        start_time = time.time()

        # Create model
        model = self._create_model(model_type)

        # Auto-tune if requested
        best_params = {}
        if auto_tune and hasattr(model, "auto_tune") and model_type != "ensemble":
            tune_result = model.auto_tune(X_train, y_train, n_trials=n_trials)
            best_params = tune_result.get("best_params", {})
            logger.info(
                "model_tuned",
                model=model_type,
                best_score=tune_result.get("best_score"),
            )
        else:
            model.train(X_train, y_train)
            best_params = model.params

        # Evaluate on test set
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)

        # Feature importance
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            try:
                feature_importance = model.get_feature_importance()
            except Exception:
                pass

        training_time = time.time() - start_time

        result = TrainingResult(
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_type,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            best_params=best_params,
            feature_importance=feature_importance,
            n_features=X_train.shape[1],
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
            training_time_seconds=training_time,
        )

        self._results.append(result)

        logger.info(
            "model_trained",
            model=model_type,
            train_f1=f"{train_metrics.f1_score:.4f}",
            test_f1=f"{test_metrics.f1_score:.4f}",
            test_auc=f"{test_metrics.auc_roc:.4f}",
            time=f"{training_time:.1f}s",
        )

        # Emit event
        event_bus.emit(EventTypes.MODEL_TRAINED, {
            "model": model_type,
            "symbol": symbol,
            "test_f1": test_metrics.f1_score,
            "test_auc": test_metrics.auc_roc,
        })

        return result

    def run(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        models: list[str] | None = None,
        auto_tune: bool = True,
        n_trials: int = 30,
        log_to_mlflow: bool = False,
    ) -> list[TrainingResult]:
        """
        Run the complete training pipeline for all models.

        Args:
            symbol: Symbol to train on.
            timeframe: Timeframe.
            models: List of model types. Default: from config.
            auto_tune: Run Optuna tuning.
            n_trials: Optuna trials per model.
            log_to_mlflow: Whether to log results to MLflow.

        Returns:
            List of TrainingResult for each model.
        """
        models = models or self.config.ml.models
        results = []

        # Prepare data once
        X_train, X_test, y_train, y_test = self.prepare_data(symbol, timeframe)

        for model_type in models:
            try:
                result = self.train_single_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    auto_tune=auto_tune,
                    n_trials=n_trials,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                results.append(result)

                if log_to_mlflow:
                    self._log_to_mlflow(result)

            except ImportError as e:
                logger.warning("model_import_error", model=model_type, error=str(e))
            except Exception as e:
                logger.error("model_training_error", model=model_type, error=str(e))

        # Print summary
        self._print_summary(results)

        return results

    def _create_model(self, model_type: str) -> BaseModel:
        """Factory method to create a model by type string."""
        if model_type == "xgboost":
            from src.models.classifiers.gradient_boost import XGBoostModel
            return XGBoostModel()
        elif model_type == "lightgbm":
            from src.models.classifiers.gradient_boost import LightGBMModel
            return LightGBMModel()
        elif model_type == "catboost":
            from src.models.classifiers.gradient_boost import CatBoostModel
            return CatBoostModel()
        elif model_type == "ensemble":
            from src.models.classifiers.gradient_boost import GradientBoostEnsemble
            return GradientBoostEnsemble()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _log_to_mlflow(self, result: TrainingResult) -> None:
        """Log training results to MLflow."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.mlflow.experiment_name)

            with mlflow.start_run(run_name=f"{result.model_name}_{result.symbol}"):
                # Log params
                mlflow.log_params(result.best_params)
                mlflow.log_param("symbol", result.symbol)
                mlflow.log_param("timeframe", result.timeframe)
                mlflow.log_param("n_features", result.n_features)

                # Log metrics
                mlflow.log_metrics({
                    "train_accuracy": result.train_metrics.accuracy,
                    "train_f1": result.train_metrics.f1_score,
                    "train_auc_roc": result.train_metrics.auc_roc,
                    "test_accuracy": result.test_metrics.accuracy,
                    "test_f1": result.test_metrics.f1_score,
                    "test_auc_roc": result.test_metrics.auc_roc,
                    "test_precision": result.test_metrics.precision,
                    "test_recall": result.test_metrics.recall,
                    "training_time": result.training_time_seconds,
                })

                logger.info("mlflow_logged", model=result.model_name, symbol=result.symbol)

        except ImportError:
            logger.warning("mlflow_not_available")
        except Exception as e:
            logger.error("mlflow_log_error", error=str(e))

    def _print_summary(self, results: list[TrainingResult]) -> None:
        """Print a formatted summary table of all results."""
        if not results:
            return

        logger.info("=" * 60)
        logger.info("training_summary", n_models=len(results))

        for r in sorted(results, key=lambda x: x.test_metrics.f1_score, reverse=True):
            logger.info(
                "model_result",
                model=r.model_name,
                train_f1=f"{r.train_metrics.f1_score:.4f}",
                test_f1=f"{r.test_metrics.f1_score:.4f}",
                test_auc=f"{r.test_metrics.auc_roc:.4f}",
                test_acc=f"{r.test_metrics.accuracy:.4f}",
                time=f"{r.training_time_seconds:.1f}s",
            )

    @property
    def results(self) -> list[TrainingResult]:
        return self._results

    def get_best_model_result(self, metric: str = "f1_score") -> TrainingResult | None:
        """Get the best model result by a given metric."""
        if not self._results:
            return None
        return max(
            self._results,
            key=lambda r: getattr(r.test_metrics, metric, 0),
        )

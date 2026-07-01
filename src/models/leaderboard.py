"""
Model Leaderboard (Fase 2)
============================
Automatic ranking and comparison of trained models via MLflow.

TODO:
- [ ] Query all MLflow runs for an experiment
- [ ] Rank by composite score (Sharpe × profit_factor / max_drawdown)
- [ ] Generate comparison DataFrame
- [ ] Auto-promote best model to "Production" stage in MLflow Registry
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.logger import get_logger

logger = get_logger("models.leaderboard")


class ModelLeaderboard:
    """Manages model ranking and comparison via MLflow."""

    def __init__(self, tracking_uri: str = "./mlruns", experiment_name: str = "quantmind") -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def get_rankings(self, metric: str = "sharpe_ratio", top_k: int = 10) -> pd.DataFrame:
        """
        Get model rankings from MLflow.

        Returns DataFrame with columns: model_name, params, metrics, rank
        """
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)

            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.info("no_experiment_found", name=self.experiment_name)
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=top_k,
            )

            if runs.empty:
                return pd.DataFrame()

            logger.info("leaderboard_loaded", runs=len(runs))
            return runs

        except ImportError:
            logger.warning("mlflow_not_available")
            return pd.DataFrame()
        except Exception as e:
            logger.error("leaderboard_error", error=str(e))
            return pd.DataFrame()

    def promote_best(self, model_name: str) -> None:
        """Promote a model to Production stage in MLflow Registry."""
        # Implementation in Fase 2
        raise NotImplementedError("Model promotion not yet implemented")

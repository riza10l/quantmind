"""
Feature Selection
=================
Automatic feature selection using multiple methods to identify the most
informative features for downstream ML/RL models.

Methods:
1. SHAP feature importance (tree-based model agnostic)
2. Mutual Information (information-theoretic)
3. Recursive Feature Elimination (RFE)
4. Permutation Importance
5. PCA for dimensionality reduction

Results are saved to the database and logged to MLflow for reproducibility.

Usage:
    from src.features.selection import FeatureSelector
    selector = FeatureSelector(config, db)
    selected = selector.run("BTC/USDT", "1d")
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.config import AppConfig, SelectionConfig, load_config
from src.core.database import DatabaseManager
from src.core.events import EventTypes, event_bus
from src.core.logger import get_logger

logger = get_logger("features.selection")


class FeatureSelector:
    """
    Multi-method feature selection pipeline.

    Runs multiple feature importance methods and produces a consensus
    ranking of the most informative features.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        db: DatabaseManager | None = None,
    ) -> None:
        self._config = config or load_config()
        self._selection_config = self._config.selection
        self._db = db or DatabaseManager(self._config.database.url)

    def run(
        self,
        symbol: str,
        timeframe: str = "1d",
        methods: list[str] | None = None,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """
        Run feature selection pipeline.

        Args:
            symbol: Trading pair.
            timeframe: Bar timeframe.
            methods: Selection methods to use.
            top_k: Number of top features to select.

        Returns:
            DataFrame with feature rankings across all methods.
        """
        from src.features.store import FeatureStore

        methods = methods or self._selection_config.methods
        top_k = top_k or self._selection_config.top_k

        run_id = str(uuid.uuid4())[:8]

        logger.info(
            "feature_selection_started",
            symbol=symbol,
            methods=methods,
            top_k=top_k,
            run_id=run_id,
        )

        # Load features with target
        store = FeatureStore(self._config, self._db)
        X, y = store.get_features_with_target(
            symbol=symbol,
            timeframe=timeframe,
            target_horizon=self._selection_config.target_horizon,
            target_type="direction",
        )

        if X.empty or y.empty:
            logger.warning("no_data_for_selection", symbol=symbol)
            return pd.DataFrame()

        logger.info(
            "selection_data_loaded",
            features=X.shape[1],
            samples=X.shape[0],
        )

        # Replace inf with nan, then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            logger.warning("insufficient_data", samples=len(X))
            return pd.DataFrame()

        # Run each method
        all_results: list[pd.DataFrame] = []

        for method in methods:
            try:
                if method == "shap":
                    result = self._shap_importance(X, y, run_id)
                elif method == "mutual_info":
                    result = self._mutual_info(X, y, run_id)
                elif method == "rfe":
                    result = self._rfe(X, y, run_id)
                elif method == "permutation":
                    result = self._permutation_importance(X, y, run_id)
                elif method == "pca":
                    result = self._pca_analysis(X, run_id)
                else:
                    logger.warning("unknown_method", method=method)
                    continue

                if result is not None and not result.empty:
                    all_results.append(result)
                    logger.info("method_completed", method=method, features=len(result))

            except Exception as e:
                logger.error("method_failed", method=method, error=str(e))

        if not all_results:
            logger.warning("no_selection_results")
            return pd.DataFrame()

        # Combine results
        combined = pd.concat(all_results, ignore_index=True)

        # Compute consensus ranking
        consensus = self._consensus_ranking(combined, top_k)

        # Store results
        records = []
        for _, row in combined.iterrows():
            records.append({
                "selection_run_id": run_id,
                "method": row["method"],
                "feature_name": row["feature_name"],
                "importance_score": float(row["importance_score"]),
                "rank": int(row["rank"]),
            })

        self._db.insert_selected_features(records)

        # Log to MLflow if available
        self._log_to_mlflow(run_id, symbol, consensus, combined)

        # Emit event
        event_bus.emit(
            EventTypes.FEATURES_SELECTED,
            {
                "symbol": symbol,
                "run_id": run_id,
                "top_features": consensus["feature_name"].tolist()[:top_k],
                "methods_used": methods,
            },
            source="features.selection",
        )

        logger.info(
            "feature_selection_complete",
            run_id=run_id,
            total_results=len(combined),
            top_features=consensus["feature_name"].tolist()[:10],
        )

        return consensus

    def _shap_importance(
        self, X: pd.DataFrame, y: pd.Series, run_id: str
    ) -> pd.DataFrame:
        """Compute SHAP feature importance using a gradient boosting model."""
        from sklearn.model_selection import train_test_split

        try:
            import shap
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("shap_xgboost_not_available")
            return pd.DataFrame()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3,
            random_state=self._selection_config.random_state,
        )

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self._selection_config.random_state,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Use a sample for SHAP (speed)
        n_samples = min(self._selection_config.shap_n_samples, len(X_test))
        X_sample = X_test.sample(n=n_samples, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        importance = np.abs(shap_values).mean(axis=0)

        result = pd.DataFrame({
            "feature_name": X.columns,
            "importance_score": importance,
            "method": "shap",
        })
        result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _mutual_info(
        self, X: pd.DataFrame, y: pd.Series, run_id: str
    ) -> pd.DataFrame:
        """Compute Mutual Information scores."""
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(
            X, y,
            random_state=self._selection_config.random_state,
            n_neighbors=5,
        )

        result = pd.DataFrame({
            "feature_name": X.columns,
            "importance_score": mi_scores,
            "method": "mutual_info",
        })
        result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _rfe(
        self, X: pd.DataFrame, y: pd.Series, run_id: str
    ) -> pd.DataFrame:
        """Recursive Feature Elimination."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import RFE

        estimator = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=self._selection_config.random_state,
        )

        n_select = max(self._selection_config.top_k, X.shape[1] // 2)
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=min(n_select, X.shape[1]),
            step=self._selection_config.rfe_step,
        )
        rfe.fit(X, y)

        result = pd.DataFrame({
            "feature_name": X.columns,
            "importance_score": 1.0 / rfe.ranking_,  # Inverse of ranking
            "method": "rfe",
        })
        result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, run_id: str
    ) -> pd.DataFrame:
        """Permutation importance on a held-out set."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3,
            random_state=self._selection_config.random_state,
        )

        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=self._selection_config.random_state,
        )
        model.fit(X_train, y_train)

        perm_result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=self._selection_config.random_state,
            n_jobs=-1,
        )

        result = pd.DataFrame({
            "feature_name": X.columns,
            "importance_score": perm_result.importances_mean,
            "method": "permutation",
        })
        result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _pca_analysis(
        self, X: pd.DataFrame, run_id: str
    ) -> pd.DataFrame:
        """PCA analysis — returns feature loadings of top components."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(20, X.shape[1]))
        pca.fit(X_scaled)

        # Sum absolute loadings across top components
        # weighted by explained variance ratio
        loadings = np.abs(pca.components_)
        weights = pca.explained_variance_ratio_[:loadings.shape[0]]
        weighted_importance = (loadings.T * weights).sum(axis=1)

        result = pd.DataFrame({
            "feature_name": X.columns,
            "importance_score": weighted_importance,
            "method": "pca",
        })
        result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    def _consensus_ranking(
        self, combined: pd.DataFrame, top_k: int
    ) -> pd.DataFrame:
        """
        Compute consensus ranking across all methods.

        Uses reciprocal rank fusion: score = Σ (1 / rank_method)
        """
        # Compute reciprocal rank for each method
        rrf_scores: dict[str, float] = {}

        for _, row in combined.iterrows():
            name = row["feature_name"]
            rank = row["rank"]
            rrf = 1.0 / (rank + 60)  # k=60 for smoothing (standard RRF)

            rrf_scores[name] = rrf_scores.get(name, 0.0) + rrf

        consensus = pd.DataFrame([
            {"feature_name": name, "consensus_score": score}
            for name, score in rrf_scores.items()
        ])

        consensus = consensus.sort_values(
            "consensus_score", ascending=False
        ).reset_index(drop=True)
        consensus["rank"] = range(1, len(consensus) + 1)

        return consensus.head(top_k)

    def _log_to_mlflow(
        self,
        run_id: str,
        symbol: str,
        consensus: pd.DataFrame,
        full_results: pd.DataFrame,
    ) -> None:
        """Log selection results to MLflow."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._config.mlflow.tracking_uri)
            mlflow.set_experiment(f"{self._config.mlflow.experiment_name}_selection")

            with mlflow.start_run(run_name=f"selection_{symbol}_{run_id}"):
                mlflow.log_param("symbol", symbol)
                mlflow.log_param("run_id", run_id)
                mlflow.log_param("methods", list(full_results["method"].unique()))
                mlflow.log_param("top_k", len(consensus))

                # Log top features
                for i, row in consensus.head(20).iterrows():
                    mlflow.log_metric(
                        f"top_{i+1}_{row['feature_name']}",
                        row["consensus_score"],
                    )

                # Log full results as artifact
                full_results.to_csv("/tmp/selection_results.csv", index=False)
                mlflow.log_artifact("/tmp/selection_results.csv")

        except ImportError:
            logger.debug("mlflow_not_available")
        except Exception as e:
            logger.warning("mlflow_logging_failed", error=str(e))


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import sys

    config = load_config()
    selector = FeatureSelector(config)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1d"

    result = selector.run(symbol, timeframe)
    if not result.empty:
        print("\n=== Top Selected Features ===")
        print(result.to_string(index=False))

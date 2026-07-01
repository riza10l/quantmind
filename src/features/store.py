"""
Feature Store
=============
Orchestrates feature computation and storage. Loads OHLCV data from
the database, computes all registered features, and stores results
in the feature store for consumption by ML, RL, and backtesting.

Usage:
    from src.features.store import FeatureStore
    store = FeatureStore(config, db)
    store.compute_and_store("BTC/USDT", "1d")
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.config import AppConfig, load_config
from src.core.database import DatabaseManager
from src.core.events import EventTypes, event_bus
from src.core.logger import get_logger

logger = get_logger("features.store")


class FeatureStore:
    """
    Manages feature computation, storage, and retrieval.

    Workflow:
    1. Load OHLCV data from database
    2. Optionally merge sentiment/funding data
    3. Compute all registered features (or specific groups)
    4. Store results in the feature store table
    5. Emit events for downstream consumers
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        db: DatabaseManager | None = None,
    ) -> None:
        self._config = config or load_config()
        self._db = db or DatabaseManager(self._config.database.url)

        # Import feature modules to trigger registration
        self._import_feature_modules()

    def _import_feature_modules(self) -> None:
        """Import all feature modules to register their features."""
        try:
            import src.features.technical  # noqa: F401
            import src.features.statistical  # noqa: F401
            import src.features.microstructure  # noqa: F401
            import src.features.sentiment  # noqa: F401
        except ImportError as e:
            logger.warning("feature_module_import_error", error=str(e))

    def compute_and_store(
        self,
        symbol: str,
        timeframe: str = "1d",
        groups: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        version: str = "v1",
    ) -> pd.DataFrame:
        """
        Compute features for a symbol and store in the database.

        Args:
            symbol: Trading pair.
            timeframe: Bar timeframe.
            groups: Feature groups to compute (None = all).
            start: Start date for data.
            end: End date for data.
            version: Feature version tag.

        Returns:
            DataFrame with computed features.
        """
        from src.features.registry import feature_registry

        logger.info(
            "computing_features",
            symbol=symbol,
            timeframe=timeframe,
            groups=groups or "all",
            registered_features=feature_registry.count,
        )

        # 1. Load OHLCV data
        df = self._db.query_ohlcv(symbol, timeframe, start=start, end=end)
        if df.empty:
            logger.warning("no_ohlcv_data", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame()

        logger.info("ohlcv_loaded", symbol=symbol, rows=len(df))

        # 2. Merge sentiment data if available
        df = self._merge_sentiment(df, symbol)

        # 3. Compute features
        enabled_groups = groups
        if not enabled_groups:
            enabled_groups = [
                g.name for g in self._config.features.groups if g.enabled
            ]

        features_df = feature_registry.compute_all(df, groups=enabled_groups)

        if features_df.empty:
            logger.warning("no_features_computed", symbol=symbol)
            return features_df

        # Optional inputs (sentiment, funding, open interest) may not have
        # been ingested yet. Their features are entirely NaN and must not
        # cause every otherwise valid OHLCV feature row to be dropped.
        unavailable_columns = features_df.columns[features_df.isna().all()].tolist()
        if unavailable_columns:
            features_df = features_df.drop(columns=unavailable_columns)
            logger.warning(
                "unavailable_features_dropped",
                count=len(unavailable_columns),
                features=unavailable_columns,
            )

        if features_df.empty:
            logger.warning("no_available_features", symbol=symbol)
            return features_df

        # 4. Drop NaN rows if configured
        if self._config.features.drop_na:
            original_len = len(features_df)
            features_df = features_df.dropna()
            dropped = original_len - len(features_df)
            if dropped > 0:
                logger.info("nan_rows_dropped", count=dropped, remaining=len(features_df))

        if features_df.empty:
            logger.warning("no_complete_feature_rows", symbol=symbol)
            return features_df

        # 5. Store in database
        stored_count = self._db.insert_features_wide(
            df=features_df,
            symbol=symbol,
            timeframe=timeframe,
            version=version,
        )

        # 6. Emit event
        event_bus.emit(
            EventTypes.FEATURES_STORED,
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "features_count": len(features_df.columns),
                "samples_count": len(features_df),
                "version": version,
            },
            source="features.store",
        )

        logger.info(
            "features_stored",
            symbol=symbol,
            features=len(features_df.columns),
            samples=len(features_df),
            stored_records=stored_count,
        )

        return features_df

    def compute_all_symbols(
        self,
        timeframe: str = "1d",
        groups: list[str] | None = None,
        version: str = "v1",
    ) -> dict[str, int]:
        """
        Compute features for all symbols with data in the database.

        Returns:
            Dict of symbol → number of features computed.
        """
        symbols = self._db.get_available_symbols(timeframe)
        results: dict[str, int] = {}

        for symbol in symbols:
            features_df = self.compute_and_store(
                symbol=symbol,
                timeframe=timeframe,
                groups=groups,
                version=version,
            )
            results[symbol] = len(features_df.columns)

        logger.info(
            "all_symbols_computed",
            symbols=len(results),
            total_features=sum(results.values()),
        )

        return results

    def get_features(
        self,
        symbol: str,
        timeframe: str = "1d",
        feature_names: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        version: str = "v1",
    ) -> pd.DataFrame:
        """
        Retrieve features from the store.

        Args:
            symbol: Trading pair.
            timeframe: Bar timeframe.
            feature_names: Specific features to retrieve (None = all).
            start: Start date.
            end: End date.
            version: Feature version.

        Returns:
            Wide-format DataFrame with features as columns.
        """
        return self._db.query_features(
            symbol=symbol,
            timeframe=timeframe,
            feature_names=feature_names,
            start=start,
            end=end,
            version=version,
        )

    def get_features_with_target(
        self,
        symbol: str,
        timeframe: str = "1d",
        target_horizon: int = 1,
        target_type: str = "return",
        version: str = "v1",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get features along with a target variable for ML training.

        Args:
            symbol: Trading pair.
            timeframe: Bar timeframe.
            target_horizon: Number of bars ahead for target.
            target_type: "return" (continuous) or "direction" (binary).
            version: Feature version.

        Returns:
            Tuple of (features_df, target_series).
        """
        features = self.get_features(symbol, timeframe, version=version)

        if features.empty:
            return features, pd.Series(dtype=float)

        # Load price data for target computation
        df = self._db.query_ohlcv(symbol, timeframe)

        if df.empty:
            return features, pd.Series(dtype=float)

        # Align features with OHLCV data
        common_idx = features.index.intersection(df.index)
        df = df.loc[common_idx]

        # Compute target
        if target_type == "return":
            target = df["close"].pct_change(periods=target_horizon).shift(-target_horizon)
        elif target_type == "direction":
            future_return = df["close"].pct_change(periods=target_horizon).shift(-target_horizon)
            target = (future_return > 0).astype(int)
        else:
            raise ValueError(f"Unknown target type: {target_type}")

        target.name = f"target_{target_type}_{target_horizon}"

        # Align and drop NaN
        aligned = pd.concat([features.loc[common_idx], target], axis=1).dropna()
        X = aligned.drop(columns=[target.name])
        y = aligned[target.name]

        return X, y

    def _merge_sentiment(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Merge sentiment data (Fear & Greed, etc.) into the OHLCV DataFrame."""
        try:
            sentiment_df = self._db.query_sentiment("fear_greed")
            if not sentiment_df.empty:
                # Align by date
                sentiment_daily = sentiment_df[["value"]].rename(
                    columns={"value": "fear_greed"}
                )
                df = df.join(sentiment_daily, how="left")
                df["fear_greed"] = df["fear_greed"].ffill()

                logger.debug("sentiment_merged", indicator="fear_greed")
        except Exception as e:
            logger.debug("sentiment_merge_skipped", error=str(e))

        return df

    def get_feature_summary(self) -> dict[str, Any]:
        """Get a summary of the feature store contents."""
        from src.features.registry import feature_registry

        return {
            "registered_features": feature_registry.count,
            "groups": feature_registry.list_groups(),
            "features_by_group": {
                group: len(feature_registry.list_features(group=group))
                for group in feature_registry.list_groups()
            },
        }


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    """Run feature computation from command line."""
    import sys

    config = load_config()
    store = FeatureStore(config)

    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        timeframe = sys.argv[2] if len(sys.argv) > 2 else "1d"
        store.compute_and_store(symbol, timeframe)
    else:
        store.compute_all_symbols()

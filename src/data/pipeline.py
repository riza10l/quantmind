"""
Data Pipeline Orchestrator
==========================
ETL orchestration that coordinates data download, validation, cleaning,
and storage across all configured data sources and symbols.

Usage:
    from src.data.pipeline import DataPipeline
    from src.core.config import load_config

    config = load_config()
    pipeline = DataPipeline(config)
    pipeline.run()  # Download everything configured
    pipeline.run_symbol("BTC/USDT", "binance", "1d")  # Single symbol
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.core.config import AppConfig, load_config
from src.core.database import DatabaseManager
from src.core.events import EventTypes, event_bus
from src.core.logger import get_logger
from src.core.types import Timeframe
from src.data.providers.base import DataProvider, DataProviderError
from src.data.providers.binance import BinanceProvider
from src.data.providers.fear_greed import FearGreedProvider
from src.data.providers.yahoo import YahooProvider
from src.data.validators import OHLCVValidator, clean_ohlcv

logger = get_logger("data.pipeline")


class DataPipeline:
    """
    Orchestrates the complete data ingestion pipeline.

    Flow: Configure → Fetch → Validate → Clean → Store → Emit events

    Supports:
    - Multi-provider (Binance, Yahoo, Fear&Greed)
    - Multi-symbol concurrent download
    - Incremental updates (resume from last timestamp)
    - Data validation and cleaning
    - Event-driven notifications
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        db: DatabaseManager | None = None,
    ) -> None:
        self._config = config or load_config()
        self._db = db or DatabaseManager(self._config.database.url)
        self._db.initialize()

        # Initialize providers
        self._providers: dict[str, DataProvider | FearGreedProvider] = {}
        self._validator = OHLCVValidator()

        self._stats: dict[str, Any] = {
            "total_downloaded": 0,
            "total_stored": 0,
            "errors": [],
        }

    def _get_provider(self, provider_name: str) -> DataProvider | FearGreedProvider:
        """Get or create a data provider instance."""
        if provider_name not in self._providers:
            if provider_name == "binance":
                self._providers[provider_name] = BinanceProvider(
                    api_key=self._config.execution.api_key,
                    api_secret=self._config.execution.api_secret,
                    retry_attempts=self._config.data.retry_attempts,
                    retry_delay=self._config.data.retry_delay_seconds,
                )
            elif provider_name == "yahoo":
                self._providers[provider_name] = YahooProvider(
                    retry_attempts=self._config.data.retry_attempts,
                    retry_delay=self._config.data.retry_delay_seconds,
                )
            elif provider_name == "fear_greed":
                self._providers[provider_name] = FearGreedProvider(
                    retry_attempts=self._config.data.retry_attempts,
                )
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

        return self._providers[provider_name]

    def run(self) -> dict[str, Any]:
        """
        Run the full data pipeline for all configured sources.

        Returns:
            Summary statistics of the pipeline run.
        """
        logger.info("pipeline_started", sources=len(self._config.data.sources))
        event_bus.emit(EventTypes.PIPELINE_STARTED, {"type": "data"})

        start_time = datetime.utcnow()

        for source_config in self._config.data.sources:
            if not source_config.enabled:
                logger.info("source_skipped", provider=source_config.provider)
                continue

            try:
                if source_config.provider == "fear_greed":
                    self._run_fear_greed(source_config.start_date)
                else:
                    for symbol in source_config.symbols:
                        for timeframe in source_config.timeframes:
                            self.run_symbol(
                                symbol=symbol,
                                provider=source_config.provider,
                                timeframe=timeframe,
                                start_date=source_config.start_date,
                                end_date=source_config.end_date,
                            )

            except Exception as e:
                error_msg = f"Error processing {source_config.provider}: {e}"
                logger.error("source_error", provider=source_config.provider, error=str(e))
                self._stats["errors"].append(error_msg)

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        self._stats["elapsed_seconds"] = elapsed

        logger.info(
            "pipeline_completed",
            downloaded=self._stats["total_downloaded"],
            stored=self._stats["total_stored"],
            errors=len(self._stats["errors"]),
            elapsed=f"{elapsed:.1f}s",
        )

        event_bus.emit(
            EventTypes.PIPELINE_COMPLETED,
            {"stats": self._stats},
            source="data.pipeline",
        )

        return self._stats

    def run_symbol(
        self,
        symbol: str,
        provider: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """
        Download data for a single symbol from a single provider.

        Supports incremental updates — resumes from the last stored timestamp.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            provider: Provider name (e.g., "binance").
            timeframe: Timeframe string (e.g., "1d").
            start_date: Start date for historical download.
            end_date: End date (defaults to now).

        Returns:
            Number of new rows stored.
        """
        logger.info(
            "downloading_symbol",
            symbol=symbol,
            provider=provider,
            timeframe=timeframe,
        )

        data_provider = self._get_provider(provider)

        # Check for incremental update
        last_ts = self._db.get_latest_timestamp(symbol, timeframe)
        if last_ts and start_date:
            # Resume from last stored timestamp
            since = max(last_ts, pd.to_datetime(start_date).to_pydatetime())
            logger.info(
                "incremental_update",
                symbol=symbol,
                resuming_from=str(since),
            )
        elif start_date:
            since = pd.to_datetime(start_date).to_pydatetime()
        else:
            since = pd.to_datetime("2020-01-01").to_pydatetime()

        until = pd.to_datetime(end_date).to_pydatetime() if end_date else datetime.utcnow()

        try:
            # Fetch data with pagination
            assert isinstance(data_provider, DataProvider)
            df = data_provider.fetch_ohlcv_full(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                until=until,
                batch_size=self._config.data.batch_size,
            )
        except (DataProviderError, AssertionError) as e:
            logger.error(
                "download_failed",
                symbol=symbol,
                provider=provider,
                error=str(e),
            )
            self._stats["errors"].append(f"{symbol}@{provider}: {e}")
            return 0

        if df.empty:
            logger.info("no_new_data", symbol=symbol, provider=provider)
            return 0

        self._stats["total_downloaded"] += len(df)

        # Validate
        if self._config.data.validate_data:
            tf_enum = Timeframe(timeframe)
            validation = self._validator.validate(df, tf_enum.minutes)
            if not validation.is_valid:
                logger.warning(
                    "validation_failed",
                    symbol=symbol,
                    errors=validation.errors,
                )

            # Log warnings
            for warn in validation.warnings:
                logger.debug("validation_warning", symbol=symbol, warning=warn)

        # Clean data
        df = clean_ohlcv(df)

        # Store
        rows_inserted = self._db.insert_ohlcv_batch(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source=provider,
        )

        self._stats["total_stored"] += rows_inserted

        # Emit event
        event_bus.emit(
            EventTypes.DATA_STORED,
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "provider": provider,
                "rows_inserted": rows_inserted,
                "total_rows": self._db.get_row_count(symbol, timeframe),
            },
            source="data.pipeline",
        )

        return rows_inserted

    def _run_fear_greed(self, start_date: str) -> int:
        """Download Fear & Greed Index data."""
        provider = self._get_provider("fear_greed")
        assert isinstance(provider, FearGreedProvider)

        records = provider.fetch_as_sentiment_records(
            limit=0,
            since=start_date,
        )

        if not records:
            logger.info("no_fear_greed_data")
            return 0

        inserted = self._db.insert_sentiment(records)
        self._stats["total_stored"] += inserted

        logger.info("fear_greed_stored", inserted=inserted, total=len(records))
        return inserted

    def get_data_summary(self) -> pd.DataFrame:
        """Get a summary of all stored data."""
        return self._db.get_data_summary()

    def get_available_symbols(self, timeframe: str = "1d") -> list[str]:
        """Get all symbols with stored data."""
        return self._db.get_available_symbols(timeframe)

    @property
    def errors(self) -> tuple[str, ...]:
        """Return errors recorded during the current pipeline run."""
        return tuple(self._stats["errors"])

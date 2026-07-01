"""
Abstract Data Provider
======================
Base class for all market data providers. Each provider implements
the standard interface for fetching OHLCV, funding rates, and other
market data from a specific source.

Design: Template Method pattern — subclasses implement _fetch_* methods,
the base class handles retry logic, rate limiting, and error handling.
"""

from __future__ import annotations

import abc
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.core.logger import get_logger
from src.core.types import DataSource, Timeframe

logger = get_logger("data.providers.base")


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class RateLimitError(DataProviderError):
    """Raised when API rate limit is hit."""
    pass


class DataProvider(abc.ABC):
    """
    Abstract base class for market data providers.

    All providers must implement:
    - fetch_ohlcv(): Download OHLCV candlestick data
    - supported_timeframes: List of supported bar durations
    - source: DataSource enum value

    Optional overrides:
    - fetch_funding_rate(): Crypto funding rates
    - fetch_open_interest(): Open interest data
    """

    def __init__(
        self,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1,
    ) -> None:
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    @property
    @abc.abstractmethod
    def source(self) -> DataSource:
        """The data source identifier."""
        ...

    @property
    @abc.abstractmethod
    def supported_timeframes(self) -> list[Timeframe]:
        """List of timeframes this provider supports."""
        ...

    @abc.abstractmethod
    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Internal implementation for fetching OHLCV data.

        Must return a DataFrame with columns:
        [timestamp, open, high, low, close, volume]

        Args:
            symbol: Trading pair (e.g., "BTC/USDT" or "AAPL").
            timeframe: Bar duration.
            since: Start datetime (inclusive).
            until: End datetime (inclusive).
            limit: Max number of bars per request.

        Returns:
            DataFrame with OHLCV columns.
        """
        ...

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with retry logic and rate limiting.

        This is the public interface — handles type conversion,
        retries, and error logging.
        """
        # Convert string types
        if isinstance(timeframe, str):
            timeframe = Timeframe(timeframe)
        if isinstance(since, str):
            since = pd.to_datetime(since).to_pydatetime()
        if isinstance(until, str):
            until = pd.to_datetime(until).to_pydatetime()

        # Validate timeframe
        if timeframe not in self.supported_timeframes:
            raise DataProviderError(
                f"Timeframe {timeframe.value} not supported by {self.source.value}. "
                f"Supported: {[t.value for t in self.supported_timeframes]}"
            )

        # Retry loop
        last_error: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                self._enforce_rate_limit()

                df = self._fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    until=until,
                    limit=limit,
                )

                if df.empty:
                    logger.warning(
                        "no_data_returned",
                        source=self.source.value,
                        symbol=symbol,
                        timeframe=timeframe.value,
                    )
                    return df

                logger.info(
                    "ohlcv_fetched",
                    source=self.source.value,
                    symbol=symbol,
                    timeframe=timeframe.value,
                    rows=len(df),
                    start=str(df["timestamp"].min()) if "timestamp" in df.columns else "N/A",
                    end=str(df["timestamp"].max()) if "timestamp" in df.columns else "N/A",
                )
                return df

            except RateLimitError:
                wait = self._retry_delay * attempt * 2
                logger.warning(
                    "rate_limited",
                    source=self.source.value,
                    attempt=attempt,
                    wait_seconds=wait,
                )
                time.sleep(wait)
                last_error = RateLimitError("Rate limit exceeded")

            except Exception as e:
                last_error = e
                if attempt < self._retry_attempts:
                    wait = self._retry_delay * attempt
                    logger.warning(
                        "fetch_retry",
                        source=self.source.value,
                        symbol=symbol,
                        attempt=attempt,
                        error=str(e),
                        wait_seconds=wait,
                    )
                    time.sleep(wait)

        logger.error(
            "fetch_failed",
            source=self.source.value,
            symbol=symbol,
            timeframe=timeframe.value,
            error=str(last_error),
        )
        raise DataProviderError(
            f"Failed to fetch {symbol} from {self.source.value} "
            f"after {self._retry_attempts} attempts: {last_error}"
        )

    def fetch_ohlcv_full(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        since: datetime | str,
        until: datetime | str | None = None,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch complete OHLCV history by paginating through the API.

        Automatically handles pagination for providers with row limits,
        fetching all data from `since` to `until`.

        Args:
            symbol: Trading pair.
            timeframe: Bar duration.
            since: Start datetime.
            until: End datetime (defaults to now).
            batch_size: Rows per API call.

        Returns:
            Complete DataFrame with all available bars.
        """
        if isinstance(timeframe, str):
            timeframe = Timeframe(timeframe)
        if isinstance(since, str):
            since = pd.to_datetime(since).to_pydatetime()
        if until is None:
            until = datetime.utcnow()
        elif isinstance(until, str):
            until = pd.to_datetime(until).to_pydatetime()

        all_data: list[pd.DataFrame] = []
        current_since = since

        while current_since < until:
            df = self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                until=until,
                limit=batch_size,
            )

            if df.empty:
                break

            all_data.append(df)

            # Move cursor to after the last fetched timestamp
            last_ts = pd.to_datetime(df["timestamp"]).max()
            next_since = last_ts + pd.Timedelta(minutes=timeframe.minutes)
            next_since = next_since.to_pydatetime()

            if next_since <= current_since:
                # No progress — avoid infinite loop
                break

            current_since = next_since

            logger.debug(
                "pagination_progress",
                source=self.source.value,
                symbol=symbol,
                fetched_so_far=sum(len(d) for d in all_data),
                current_date=str(current_since),
            )

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"], keep="last")
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "full_history_fetched",
            source=self.source.value,
            symbol=symbol,
            timeframe=timeframe.value,
            total_rows=len(result),
        )

        return result

    def fetch_funding_rate(
        self,
        symbol: str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch funding rate data (crypto only).

        Returns DataFrame with columns: [timestamp, funding_rate]
        Override in subclasses that support this.
        """
        raise NotImplementedError(
            f"{self.source.value} does not support funding rate data"
        )

    def fetch_open_interest(
        self,
        symbol: str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch open interest data (crypto derivatives only).

        Returns DataFrame with columns: [timestamp, open_interest]
        Override in subclasses that support this.
        """
        raise NotImplementedError(
            f"{self.source.value} does not support open interest data"
        )

    def _enforce_rate_limit(self) -> None:
        """Enforce minimum delay between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

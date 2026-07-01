"""
Yahoo Finance Data Provider
============================
Downloads OHLCV data for equities, indices, and ETFs via the yfinance library.
Supports daily, weekly, and monthly timeframes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.logger import get_logger
from src.core.types import DataSource, Timeframe
from src.data.providers.base import DataProvider, DataProviderError

logger = get_logger("data.providers.yahoo")

# yfinance interval mapping
_INTERVAL_MAP = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.D1: "1d",
    Timeframe.W1: "1wk",
}


class YahooProvider(DataProvider):
    """
    Yahoo Finance data provider using yfinance.

    Best suited for:
    - US equities (AAPL, MSFT, GOOGL)
    - Indices (^SPX, ^GSPC)
    - ETFs (SPY, QQQ, IWM)

    Limitations:
    - Intraday data limited to last 60 days (1m), 730 days (1h)
    - No funding rate or open interest
    """

    def __init__(
        self,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            rate_limit_delay=0.5,  # Be gentle with Yahoo
        )

    @property
    def source(self) -> DataSource:
        return DataSource.YAHOO

    @property
    def supported_timeframes(self) -> list[Timeframe]:
        return [
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
            Timeframe.H1, Timeframe.D1, Timeframe.W1,
        ]

    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for Yahoo provider. "
                "Install with: pip install yfinance"
            )

        interval = _INTERVAL_MAP.get(timeframe)
        if not interval:
            raise DataProviderError(
                f"Timeframe {timeframe.value} not supported by Yahoo Finance"
            )

        # Format dates for yfinance
        start_str = since.strftime("%Y-%m-%d") if since else "2015-01-01"
        end_str = until.strftime("%Y-%m-%d") if until else None

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval=interval,
                start=start_str,
                end=end_str,
                auto_adjust=True,  # Adjust for splits/dividends
            )
        except Exception as e:
            raise DataProviderError(
                f"yfinance error for {symbol}: {e}"
            )

        if df.empty:
            return pd.DataFrame()

        # Normalize column names (yfinance uses Title Case)
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Reset index to get timestamp as a column
        df = df.reset_index()
        ts_col = "Date" if "Date" in df.columns else "Datetime"
        df = df.rename(columns={ts_col: "timestamp"})

        # Ensure timezone-naive timestamps
        if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # Select only OHLCV columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Apply limit
        if limit and len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        return df

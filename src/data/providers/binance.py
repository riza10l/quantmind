"""
Binance Data Provider
=====================
Downloads OHLCV, funding rates, and open interest from Binance
via the CCXT library. Supports both spot and futures markets.

CCXT handles API authentication, pagination quirks, and response
normalization across exchanges.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.core.logger import get_logger
from src.core.types import DataSource, Timeframe
from src.data.providers.base import DataProvider, DataProviderError, RateLimitError

logger = get_logger("data.providers.binance")

# CCXT timeframe mapping
_TIMEFRAME_MAP = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
    Timeframe.W1: "1w",
}


class BinanceProvider(DataProvider):
    """
    Binance data provider using CCXT.

    Fetches:
    - OHLCV candlestick data (spot + futures)
    - Funding rates (futures only)
    - Open interest (futures only)

    Public endpoints don't require API keys for market data.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            rate_limit_delay=0.1,  # 100ms between requests
        )
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._exchange = None

    @property
    def exchange(self) -> Any:
        """Lazy-initialize the CCXT exchange instance."""
        if self._exchange is None:
            try:
                import ccxt
            except ImportError:
                raise ImportError(
                    "ccxt is required for Binance provider. "
                    "Install with: pip install ccxt"
                )

            config: dict[str, Any] = {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                },
            }

            if self._api_key:
                config["apiKey"] = self._api_key
                config["secret"] = self._api_secret

            self._exchange = ccxt.binance(config)

            if self._testnet:
                self._exchange.set_sandbox_mode(True)

            logger.info(
                "exchange_initialized",
                exchange="binance",
                testnet=self._testnet,
            )

        return self._exchange

    @property
    def source(self) -> DataSource:
        return DataSource.BINANCE

    @property
    def supported_timeframes(self) -> list[Timeframe]:
        return list(Timeframe)

    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Binance via CCXT."""
        import ccxt

        ccxt_tf = _TIMEFRAME_MAP[timeframe]
        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=ccxt_tf,
                since=since_ms,
                limit=min(limit, 1000),  # Binance max is 1000
            )
        except ccxt.RateLimitExceeded:
            raise RateLimitError("Binance rate limit exceeded")
        except ccxt.NetworkError as e:
            raise DataProviderError(f"Network error: {e}")
        except ccxt.ExchangeError as e:
            raise DataProviderError(f"Exchange error: {e}")

        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # Remove tz for storage

        # Filter by end date if specified
        if until:
            df = df[df["timestamp"] <= pd.to_datetime(until)]

        return df

    def fetch_funding_rate(
        self,
        symbol: str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Binance Futures.

        Returns DataFrame with columns: [timestamp, funding_rate]
        """
        import ccxt

        if isinstance(since, str):
            since = pd.to_datetime(since).to_pydatetime()
        if isinstance(until, str):
            until = pd.to_datetime(until).to_pydatetime()

        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            # Switch to futures for funding rate
            self.exchange.options["defaultType"] = "future"

            rates = self.exchange.fetch_funding_rate_history(
                symbol=symbol,
                since=since_ms,
                limit=1000,
            )
        except ccxt.ExchangeError as e:
            logger.warning("funding_rate_error", symbol=symbol, error=str(e))
            return pd.DataFrame()
        finally:
            self.exchange.options["defaultType"] = "spot"

        if not rates:
            return pd.DataFrame()

        records = []
        for rate in rates:
            records.append({
                "timestamp": pd.to_datetime(rate["timestamp"], unit="ms"),
                "funding_rate": rate.get("fundingRate", 0.0),
            })

        df = pd.DataFrame(records)

        if until:
            df = df[df["timestamp"] <= pd.to_datetime(until)]

        logger.info(
            "funding_rates_fetched",
            symbol=symbol,
            rows=len(df),
        )
        return df

    def fetch_open_interest(
        self,
        symbol: str,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch open interest data from Binance Futures.

        Returns DataFrame with columns: [timestamp, open_interest]
        """
        import ccxt

        try:
            self.exchange.options["defaultType"] = "future"
            oi = self.exchange.fetch_open_interest(symbol)
        except ccxt.ExchangeError as e:
            logger.warning("open_interest_error", symbol=symbol, error=str(e))
            return pd.DataFrame()
        finally:
            self.exchange.options["defaultType"] = "spot"

        if not oi:
            return pd.DataFrame()

        # Current OI snapshot (not historical)
        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(oi.get("timestamp", datetime.utcnow()), unit="ms")
            if isinstance(oi.get("timestamp"), (int, float))
            else datetime.utcnow(),
            "open_interest": oi.get("openInterestAmount", 0.0)
            or oi.get("openInterest", 0.0),
        }])

        logger.info("open_interest_fetched", symbol=symbol)
        return df

    def get_available_symbols(self, market_type: str = "spot") -> list[str]:
        """Get all available trading pairs on Binance."""
        self.exchange.options["defaultType"] = market_type
        self.exchange.load_markets()
        symbols = [
            s for s in self.exchange.symbols
            if "/USDT" in s and not s.endswith(":USDT")
        ]
        return sorted(symbols)

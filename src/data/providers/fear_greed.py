"""
Fear & Greed Index Provider
============================
Downloads the Crypto Fear & Greed Index from Alternative.me API.
This is a daily sentiment indicator ranging from 0 (Extreme Fear)
to 100 (Extreme Greed).

API endpoint: https://api.alternative.me/fng/
No API key required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from src.core.logger import get_logger
from src.core.types import DataSource

logger = get_logger("data.providers.fear_greed")


class FearGreedProvider:
    """
    Crypto Fear & Greed Index provider.

    Data source: Alternative.me
    Update frequency: Daily
    Range: 0 (Extreme Fear) to 100 (Extreme Greed)

    Labels:
    - 0-24: Extreme Fear
    - 25-49: Fear
    - 50-74: Greed
    - 75-100: Extreme Greed
    """

    API_URL = "https://api.alternative.me/fng/"

    def __init__(self, retry_attempts: int = 3) -> None:
        self._retry_attempts = retry_attempts

    @property
    def source(self) -> DataSource:
        return DataSource.FEAR_GREED

    def fetch(
        self,
        limit: int = 0,  # 0 = all available data
        since: datetime | str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index data.

        Args:
            limit: Number of days to fetch (0 = all).
            since: Filter data from this date onwards.

        Returns:
            DataFrame with columns: [timestamp, value, label]
        """
        params = {"limit": limit, "format": "json"}

        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = requests.get(
                    self.API_URL,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as e:
                if attempt == self._retry_attempts:
                    logger.error("fear_greed_fetch_failed", error=str(e))
                    return pd.DataFrame()
                logger.warning(
                    "fear_greed_retry",
                    attempt=attempt,
                    error=str(e),
                )

        if "data" not in data:
            logger.error("fear_greed_invalid_response", response=str(data)[:200])
            return pd.DataFrame()

        records = []
        for entry in data["data"]:
            records.append({
                "timestamp": pd.to_datetime(int(entry["timestamp"]), unit="s"),
                "value": float(entry["value"]),
                "label": entry.get("value_classification", ""),
            })

        df = pd.DataFrame(records)

        if df.empty:
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filter by since date
        if since:
            if isinstance(since, str):
                since = pd.to_datetime(since)
            df = df[df["timestamp"] >= since].reset_index(drop=True)

        logger.info("fear_greed_fetched", rows=len(df))
        return df

    def fetch_as_sentiment_records(
        self,
        limit: int = 0,
        since: datetime | str | None = None,
    ) -> list[dict]:
        """
        Fetch and format as records ready for the sentiment table.

        Returns list of dicts with: timestamp, indicator, value, label, source
        """
        df = self.fetch(limit=limit, since=since)
        if df.empty:
            return []

        records = []
        for _, row in df.iterrows():
            records.append({
                "timestamp": row["timestamp"],
                "indicator": "fear_greed",
                "value": row["value"],
                "label": row["label"],
                "source": self.source.value,
            })

        return records

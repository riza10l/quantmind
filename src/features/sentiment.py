"""
Sentiment Features
==================
Features derived from sentiment data (Fear & Greed Index, funding rates,
open interest changes). These are merged with price data during the
feature computation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.features.registry import register_feature

logger = get_logger("features.sentiment")


@register_feature("fear_greed_value", group="sentiment",
                   description="Crypto Fear & Greed Index (0-100)")
def compute_fear_greed(df: pd.DataFrame) -> pd.Series:
    """
    Fear & Greed Index value.

    This feature expects a 'fear_greed' column in the input DataFrame,
    which should be joined from the sentiment table before feature computation.
    Returns NaN if the column is not present.
    """
    if "fear_greed" in df.columns:
        return df["fear_greed"]
    return pd.Series(np.nan, index=df.index, name="fear_greed_value")


@register_feature("fear_greed_sma_7", group="sentiment",
                   description="Fear & Greed 7-day moving average")
def compute_fear_greed_sma(df: pd.DataFrame) -> pd.Series:
    if "fear_greed" in df.columns:
        return df["fear_greed"].rolling(7).mean()
    return pd.Series(np.nan, index=df.index, name="fear_greed_sma_7")


@register_feature("fear_greed_change", group="sentiment",
                   description="Fear & Greed daily change")
def compute_fear_greed_change(df: pd.DataFrame) -> pd.Series:
    if "fear_greed" in df.columns:
        return df["fear_greed"].diff()
    return pd.Series(np.nan, index=df.index, name="fear_greed_change")


@register_feature("fear_greed_extreme", group="sentiment",
                   description="Fear & Greed extreme zones (-1=extreme fear, 1=extreme greed)")
def compute_fear_greed_extreme(df: pd.DataFrame) -> pd.Series:
    if "fear_greed" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="fear_greed_extreme")

    fg = df["fear_greed"]
    result = pd.Series(0, index=df.index, name="fear_greed_extreme", dtype=float)
    result[fg <= 25] = -1.0  # Extreme Fear
    result[fg >= 75] = 1.0   # Extreme Greed
    return result


@register_feature("funding_rate_value", group="sentiment",
                   description="Funding rate (positive = longs pay shorts)")
def compute_funding_rate(df: pd.DataFrame) -> pd.Series:
    if "funding_rate" in df.columns:
        return df["funding_rate"]
    return pd.Series(np.nan, index=df.index, name="funding_rate_value")


@register_feature("funding_rate_sma_7", group="sentiment",
                   description="Funding rate 7-day moving average")
def compute_funding_rate_sma(df: pd.DataFrame) -> pd.Series:
    if "funding_rate" in df.columns:
        return df["funding_rate"].rolling(7).mean()
    return pd.Series(np.nan, index=df.index, name="funding_rate_sma_7")


@register_feature("open_interest_change", group="sentiment",
                   description="Open interest percent change")
def compute_oi_change(df: pd.DataFrame) -> pd.Series:
    if "open_interest" in df.columns:
        return df["open_interest"].pct_change()
    return pd.Series(np.nan, index=df.index, name="open_interest_change")


logger.info("sentiment_features_registered")

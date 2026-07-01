"""
Market Microstructure Features
===============================
Features derived from OHLCV data that proxy for market microstructure
characteristics normally requiring tick/orderbook data.

Features:
- Kyle's Lambda proxy (price impact of volume)
- Amihud illiquidity ratio
- Bid-ask spread proxies (Roll, Corwin-Schultz, OHLC-based)
- Volume clock features
- Trade intensity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.features.registry import register_feature

logger = get_logger("features.microstructure")


@register_feature("amihud_illiquidity_20", group="microstructure",
                   description="Amihud illiquidity ratio (20-day rolling)")
def compute_amihud(df: pd.DataFrame) -> pd.Series:
    """
    Amihud (2002) illiquidity measure.

    ILLIQ = |return| / (volume × price)

    Higher values = less liquid = larger price impact per unit volume.
    """
    abs_returns = df["close"].pct_change().abs()
    dollar_volume = df["close"] * df["volume"]
    daily_illiq = abs_returns / dollar_volume.replace(0, np.nan)
    return daily_illiq.rolling(20).mean()


@register_feature("kyle_lambda_20", group="microstructure",
                   description="Kyle's Lambda proxy (20-day price impact)")
def compute_kyle_lambda(df: pd.DataFrame) -> pd.Series:
    """
    Kyle's Lambda proxy — regression of price change on signed volume.

    Lambda = ΔPrice / SignedVolume

    Estimates price impact: how much price moves per unit of
    net buying/selling pressure.
    """
    price_change = df["close"].diff()
    signed_volume = df["volume"] * np.sign(price_change)

    def _lambda_func(window_data):
        if len(window_data) < 5:
            return np.nan
        idx = window_data.index
        pc = price_change.loc[idx]
        sv = signed_volume.loc[idx]
        sv_clean = sv[sv != 0]
        pc_clean = pc.loc[sv_clean.index]
        if len(sv_clean) < 5:
            return np.nan
        try:
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(sv_clean.values, pc_clean.values)
            return abs(slope)
        except (ValueError, RuntimeWarning):
            return np.nan

    return df["close"].rolling(20).apply(
        lambda x: _lambda_func(x), raw=False
    )


@register_feature("roll_spread_20", group="microstructure",
                   description="Roll (1984) effective spread estimator")
def compute_roll_spread(df: pd.DataFrame) -> pd.Series:
    """
    Roll (1984) effective spread estimate.

    Based on the negative serial covariance of price changes:
    Spread = 2 × sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    Only valid when covariance is negative (which it should be
    for a market with bid-ask bounce).
    """
    price_changes = df["close"].diff()

    def _roll_func(window):
        if len(window) < 5:
            return np.nan
        cov = np.cov(window[1:], window[:-1])[0, 1]
        if cov < 0:
            return 2 * np.sqrt(-cov)
        return 0.0

    return price_changes.rolling(20).apply(
        lambda x: _roll_func(x.values), raw=False
    )


@register_feature("ohlc_spread_proxy", group="microstructure",
                   description="OHLC-based bid-ask spread proxy")
def compute_ohlc_spread(df: pd.DataFrame) -> pd.Series:
    """
    Corwin-Schultz (2012) high-low spread estimator.

    Uses the relationship between high-low ranges over different
    frequencies to estimate the effective bid-ask spread.
    """
    high = df["high"]
    low = df["low"]

    # Single-period log range
    beta = (np.log(high / low)) ** 2

    # Two-period combined range
    high_2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    low_2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(high_2 / low_2)) ** 2

    # Rolling averages
    beta_avg = beta.rolling(20).mean()
    gamma_avg = gamma.rolling(20).mean()

    alpha = (np.sqrt(2 * beta_avg) - np.sqrt(beta_avg)) / (
        3 - 2 * np.sqrt(2)
    ) - np.sqrt(gamma_avg / (3 - 2 * np.sqrt(2)))

    # Spread = 2(e^α - 1) / (1 + e^α)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)  # Spread can't be negative


@register_feature("volume_clock_20", group="microstructure",
                   description="Volume clock — normalized cumulative volume")
def compute_volume_clock(df: pd.DataFrame) -> pd.Series:
    """
    Volume clock: maps calendar time to volume time.

    Normalizes cumulative volume by its rolling average,
    indicating whether trading activity is accelerating or decelerating.
    """
    vol_avg = df["volume"].rolling(20).mean()
    return df["volume"] / vol_avg.replace(0, np.nan)


@register_feature("trade_intensity_10", group="microstructure",
                   description="Trade intensity — volume × volatility proxy")
def compute_trade_intensity(df: pd.DataFrame) -> pd.Series:
    """
    Trade intensity proxy: volume × absolute return.

    High values indicate periods of heavy trading with significant
    price movement — often around news events or liquidations.
    """
    abs_return = df["close"].pct_change().abs()
    intensity = df["volume"] * abs_return
    avg = intensity.rolling(10).mean()
    return intensity / avg.replace(0, np.nan)


@register_feature("volume_price_trend", group="microstructure",
                   description="Volume-Price Trend (cumulative)")
def compute_vpt(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Price Trend (VPT):
    VPT = VPT_prev + Volume × (Close - Close_prev) / Close_prev
    """
    pct_change = df["close"].pct_change()
    return (df["volume"] * pct_change).cumsum()


@register_feature("close_location_value", group="microstructure",
                   description="Close Location Value (CLV)")
def compute_clv(df: pd.DataFrame) -> pd.Series:
    """
    Close Location Value:
    CLV = ((Close - Low) - (High - Close)) / (High - Low)

    Range: -1 to +1. Shows where close falls within the bar's range.
    """
    range_ = df["high"] - df["low"]
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / range_.replace(0, np.nan)
    return clv


logger.info(
    "microstructure_features_registered",
    count=len([f for f in __import__("src.features.registry",
               fromlist=["feature_registry"]).feature_registry.list_features(
               group="microstructure")]),
)

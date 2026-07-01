"""
Statistical Features
====================
Advanced statistical features for capturing market dynamics that
traditional technical indicators miss.

Features:
- Return distribution moments (skewness, kurtosis)
- Shannon and Sample entropy
- Hurst exponent (mean-reversion vs trending)
- Fractal dimension
- Garman-Klass volatility estimator
- Realized volatility at multiple windows
- Auto-correlation features
- Rolling z-scores
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.core.logger import get_logger
from src.features.registry import register_feature

logger = get_logger("features.statistical")


# ============================================================
# RETURN FEATURES
# ============================================================

def _register_return_features():
    """Register return-based features at multiple horizons."""
    for window in [1, 2, 3, 5, 10, 20, 60]:
        name = f"return_{window}"
        desc = f"Log return over {window} periods"

        def _make_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                return np.log(df["close"] / df["close"].shift(w))
            return func

        register_feature(name, group="statistical", description=desc)(_make_func())


# ============================================================
# VOLATILITY FEATURES
# ============================================================

def _register_volatility_features():
    """Register volatility features at multiple windows."""
    for window in [5, 10, 20, 50]:
        # Realized volatility (std of log returns)
        rv_name = f"realized_vol_{window}"
        rv_desc = f"Realized volatility ({window} periods)"

        def _make_rv_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                log_returns = np.log(df["close"] / df["close"].shift(1))
                return log_returns.rolling(window=w).std() * np.sqrt(252)
            return func

        register_feature(rv_name, group="statistical", description=rv_desc)(_make_rv_func())

        # Garman-Klass volatility estimator
        gk_name = f"garman_klass_vol_{window}"
        gk_desc = f"Garman-Klass volatility ({window} periods)"

        def _make_gk_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                log_hl = np.log(df["high"] / df["low"]) ** 2
                log_co = np.log(df["close"] / df["open"]) ** 2
                gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
                return gk.rolling(window=w).mean().apply(np.sqrt) * np.sqrt(252)
            return func

        register_feature(gk_name, group="statistical", description=gk_desc)(_make_gk_func())


@register_feature("vol_of_vol_20", group="statistical",
                   description="Volatility of volatility (vol clustering)")
def compute_vol_of_vol(df: pd.DataFrame) -> pd.Series:
    log_returns = np.log(df["close"] / df["close"].shift(1))
    rolling_vol = log_returns.rolling(10).std()
    return rolling_vol.rolling(20).std()


# ============================================================
# DISTRIBUTION MOMENTS
# ============================================================

def _register_moment_features():
    """Register skewness and kurtosis at multiple windows."""
    for window in [20, 50]:
        skew_name = f"skewness_{window}"
        skew_desc = f"Return skewness ({window} periods)"

        def _make_skew_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                returns = df["close"].pct_change()
                return returns.rolling(window=w).skew()
            return func

        register_feature(skew_name, group="statistical", description=skew_desc)(
            _make_skew_func()
        )

        kurt_name = f"kurtosis_{window}"
        kurt_desc = f"Return kurtosis ({window} periods)"

        def _make_kurt_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                returns = df["close"].pct_change()
                return returns.rolling(window=w).kurt()
            return func

        register_feature(kurt_name, group="statistical", description=kurt_desc)(
            _make_kurt_func()
        )


# ============================================================
# ENTROPY FEATURES
# ============================================================

def _shannon_entropy(series: pd.Series, bins: int = 10) -> float:
    """Compute Shannon entropy of a distribution."""
    if len(series.dropna()) < bins:
        return np.nan
    counts, _ = np.histogram(series.dropna(), bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _register_entropy_features():
    """Register entropy features."""
    for window in [20, 50]:
        name = f"entropy_{window}"
        desc = f"Shannon entropy of returns ({window} periods)"

        def _make_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                returns = df["close"].pct_change()
                return returns.rolling(window=w).apply(
                    _shannon_entropy, raw=False
                )
            return func

        register_feature(name, group="statistical", description=desc)(_make_func())


@register_feature("price_entropy_50", group="statistical",
                   description="Shannon entropy of price levels (50 periods)")
def compute_price_entropy(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(50).apply(_shannon_entropy, raw=False)


# ============================================================
# HURST EXPONENT
# ============================================================

def _compute_hurst(series: np.ndarray) -> float:
    """
    Compute the Hurst exponent using the R/S method.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    if len(series) < 20:
        return np.nan

    series = series[~np.isnan(series)]
    if len(series) < 20:
        return np.nan

    n = len(series)
    max_k = min(int(n / 4), 100)
    if max_k < 4:
        return np.nan

    rs_list = []
    n_list = []

    for k in range(4, max_k + 1):
        subseries = np.array_split(series, k)
        rs_values = []

        for ss in subseries:
            if len(ss) < 2:
                continue
            mean = np.mean(ss)
            deviations = np.cumsum(ss - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(ss, ddof=1)
            if s > 0:
                rs_values.append(r / s)

        if rs_values:
            rs_list.append(np.mean(rs_values))
            n_list.append(len(subseries[0]))

    if len(rs_list) < 3:
        return np.nan

    try:
        log_rs = np.log(rs_list)
        log_n = np.log(n_list)
        slope, _, _, _, _ = scipy_stats.linregress(log_n, log_rs)
        return float(slope)
    except (ValueError, RuntimeWarning):
        return np.nan


@register_feature("hurst_100", group="statistical",
                   description="Hurst exponent (100-period window)")
def compute_hurst(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change().values
    result = pd.Series(np.nan, index=df.index, name="hurst_100")
    for i in range(100, len(returns)):
        result.iloc[i] = _compute_hurst(returns[i-100:i])
    return result


# ============================================================
# Z-SCORE FEATURES
# ============================================================

def _register_zscore_features():
    """Register z-score features."""
    for window in [20, 50]:
        name = f"zscore_{window}"
        desc = f"Price z-score ({window} periods)"

        def _make_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                mean = df["close"].rolling(w).mean()
                std = df["close"].rolling(w).std()
                return (df["close"] - mean) / std
            return func

        register_feature(name, group="statistical", description=desc)(_make_func())

        vol_z_name = f"volume_zscore_{window}"
        vol_z_desc = f"Volume z-score ({window} periods)"

        def _make_vol_z_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                mean = df["volume"].rolling(w).mean()
                std = df["volume"].rolling(w).std()
                return (df["volume"] - mean) / std
            return func

        register_feature(vol_z_name, group="statistical", description=vol_z_desc)(
            _make_vol_z_func()
        )


# ============================================================
# AUTO-CORRELATION FEATURES
# ============================================================

@register_feature("autocorr_1", group="statistical",
                   description="Return auto-correlation (lag 1, window 50)")
def compute_autocorr_1(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change()
    return returns.rolling(50).apply(lambda x: x.autocorr(lag=1), raw=False)


@register_feature("autocorr_5", group="statistical",
                   description="Return auto-correlation (lag 5, window 50)")
def compute_autocorr_5(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change()
    return returns.rolling(50).apply(lambda x: x.autocorr(lag=5), raw=False)


# ============================================================
# RANGE FEATURES
# ============================================================

@register_feature("high_low_range_pct", group="statistical",
                   description="Intrabar range (high-low)/close")
def compute_hl_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]) / df["close"]


@register_feature("close_to_high_pct", group="statistical",
                   description="Close position within bar range")
def compute_close_to_high(df: pd.DataFrame) -> pd.Series:
    range_ = df["high"] - df["low"]
    return (df["close"] - df["low"]) / range_.replace(0, np.nan)


def _register_rolling_max_dd():
    """Register rolling max drawdown features."""
    for window in [20, 50]:
        name = f"rolling_max_dd_{window}"
        desc = f"Rolling max drawdown ({window} periods)"

        def _make_func(w=window):
            def func(df: pd.DataFrame) -> pd.Series:
                rolling_max = df["close"].rolling(w).max()
                drawdown = (df["close"] - rolling_max) / rolling_max
                return drawdown
            return func

        register_feature(name, group="statistical", description=desc)(_make_func())


# ============================================================
# Auto-register
# ============================================================

def register_all_statistical_features() -> int:
    """Register all statistical features."""
    _register_return_features()
    _register_volatility_features()
    _register_moment_features()
    _register_entropy_features()
    _register_zscore_features()
    _register_rolling_max_dd()

    from src.features.registry import feature_registry
    count = len(feature_registry.list_features(group="statistical"))
    logger.info("statistical_features_registered", count=count)
    return count


register_all_statistical_features()

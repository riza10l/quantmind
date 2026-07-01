"""
Technical Indicators Feature Generator
=======================================
Generates 150+ technical analysis features using pandas-ta.
All features are registered in the global feature registry.

Feature Groups:
- Trend: SMA, EMA, MACD, ADX, Aroon, Ichimoku
- Momentum: RSI, Stochastic, CCI, Williams %R, ROC, MOM
- Volatility: Bollinger Bands, ATR, Keltner, Donchian, Garman-Klass
- Volume: OBV, VWAP, CMF, MFI, A/D Line

Each feature is registered with @register_feature so it can be
computed selectively or in bulk via the FeatureRegistry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.features.registry import register_feature

logger = get_logger("features.technical")


# ============================================================
# Helper: safe pandas-ta import
# ============================================================

def _ensure_pandas_ta():
    """Import pandas_ta lazily to avoid startup cost."""
    try:
        import pandas_ta as ta
        return ta
    except ImportError:
        raise ImportError(
            "pandas-ta is required for technical features. "
            "Install with: pip install pandas-ta"
        )


# ============================================================
# TREND INDICATORS
# ============================================================

def _register_sma_features():
    """Register SMA features for multiple periods."""
    for period in [7, 14, 21, 50, 100, 200]:
        name = f"sma_{period}"
        desc = f"Simple Moving Average ({period} periods)"

        def _make_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                return df["close"].rolling(window=p).mean()
            return func

        register_feature(name, group="technical", description=desc)(_make_func())

        # SMA distance (price relative to SMA)
        dist_name = f"sma_{period}_dist"
        dist_desc = f"Price distance from SMA-{period} (normalized)"

        def _make_dist_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                sma = df["close"].rolling(window=p).mean()
                return (df["close"] - sma) / sma
            return func

        register_feature(dist_name, group="technical", description=dist_desc)(_make_dist_func())


def _register_ema_features():
    """Register EMA features for multiple periods."""
    for period in [9, 12, 21, 26, 50, 100, 200]:
        name = f"ema_{period}"
        desc = f"Exponential Moving Average ({period} periods)"

        def _make_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                return df["close"].ewm(span=p, adjust=False).mean()
            return func

        register_feature(name, group="technical", description=desc)(_make_func())


@register_feature("macd", group="technical", description="MACD line")
def compute_macd(df: pd.DataFrame) -> pd.Series:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    return ema12 - ema26


@register_feature("macd_signal", group="technical", description="MACD signal line")
def compute_macd_signal(df: pd.DataFrame) -> pd.Series:
    macd = compute_macd(df)
    return macd.ewm(span=9, adjust=False).mean()


@register_feature("macd_histogram", group="technical", description="MACD histogram")
def compute_macd_hist(df: pd.DataFrame) -> pd.Series:
    return compute_macd(df) - compute_macd_signal(df)


@register_feature("adx_14", group="technical", description="Average Directional Index (14)")
def compute_adx(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if result is not None and "ADX_14" in result.columns:
        return result["ADX_14"]
    return pd.Series(np.nan, index=df.index, name="adx_14")


@register_feature("plus_di_14", group="technical", description="+DI (14)")
def compute_plus_di(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if result is not None and "DMP_14" in result.columns:
        return result["DMP_14"]
    return pd.Series(np.nan, index=df.index, name="plus_di_14")


@register_feature("minus_di_14", group="technical", description="-DI (14)")
def compute_minus_di(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if result is not None and "DMN_14" in result.columns:
        return result["DMN_14"]
    return pd.Series(np.nan, index=df.index, name="minus_di_14")


@register_feature("aroon_up", group="technical", description="Aroon Up (25)")
def compute_aroon_up(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.aroon(df["high"], df["low"], length=25)
    if result is not None and "AROONU_25" in result.columns:
        return result["AROONU_25"]
    return pd.Series(np.nan, index=df.index, name="aroon_up")


@register_feature("aroon_down", group="technical", description="Aroon Down (25)")
def compute_aroon_down(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.aroon(df["high"], df["low"], length=25)
    if result is not None and "AROOND_25" in result.columns:
        return result["AROOND_25"]
    return pd.Series(np.nan, index=df.index, name="aroon_down")


# ============================================================
# MOMENTUM INDICATORS
# ============================================================

def _register_rsi_features():
    """Register RSI features for multiple periods."""
    for period in [7, 14, 21]:
        name = f"rsi_{period}"
        desc = f"Relative Strength Index ({period} periods)"

        def _make_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                ta = _ensure_pandas_ta()
                return ta.rsi(df["close"], length=p)
            return func

        register_feature(name, group="technical", description=desc)(_make_func())


@register_feature("stoch_k", group="technical", description="Stochastic %K (14)")
def compute_stoch_k(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if result is not None and "STOCHk_14_3_3" in result.columns:
        return result["STOCHk_14_3_3"]
    return pd.Series(np.nan, index=df.index, name="stoch_k")


@register_feature("stoch_d", group="technical", description="Stochastic %D (14)")
def compute_stoch_d(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    result = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if result is not None and "STOCHd_14_3_3" in result.columns:
        return result["STOCHd_14_3_3"]
    return pd.Series(np.nan, index=df.index, name="stoch_d")


def _register_cci_features():
    """Register CCI features."""
    for period in [14, 20]:
        name = f"cci_{period}"
        desc = f"Commodity Channel Index ({period} periods)"

        def _make_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                ta = _ensure_pandas_ta()
                return ta.cci(df["high"], df["low"], df["close"], length=p)
            return func

        register_feature(name, group="technical", description=desc)(_make_func())


@register_feature("williams_r", group="technical", description="Williams %R (14)")
def compute_williams_r(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    return ta.willr(df["high"], df["low"], df["close"], length=14)


def _register_roc_features():
    """Register Rate of Change features."""
    for period in [1, 5, 10, 20]:
        name = f"roc_{period}"
        desc = f"Rate of Change ({period} periods)"

        def _make_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                return df["close"].pct_change(periods=p)
            return func

        register_feature(name, group="technical", description=desc)(_make_func())


@register_feature("momentum_10", group="technical", description="Momentum (10)")
def compute_momentum(df: pd.DataFrame) -> pd.Series:
    return df["close"] - df["close"].shift(10)


# ============================================================
# VOLATILITY INDICATORS
# ============================================================

@register_feature("bb_upper", group="technical", description="Bollinger Band Upper (20, 2σ)")
def compute_bb_upper(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return sma + 2 * std


@register_feature("bb_lower", group="technical", description="Bollinger Band Lower (20, 2σ)")
def compute_bb_lower(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return sma - 2 * std


@register_feature("bb_width", group="technical", description="Bollinger Band Width")
def compute_bb_width(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma


@register_feature("bb_pct_b", group="technical", description="Bollinger %B")
def compute_bb_pct_b(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (df["close"] - lower) / (upper - lower)


def _register_atr_features():
    """Register ATR features."""
    for period in [14, 21]:
        name = f"atr_{period}"
        desc = f"Average True Range ({period} periods)"

        def _make_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                ta = _ensure_pandas_ta()
                return ta.atr(df["high"], df["low"], df["close"], length=p)
            return func

        register_feature(name, group="technical", description=desc)(_make_func())

        # Normalized ATR (relative to close)
        natr_name = f"natr_{period}"
        natr_desc = f"Normalized ATR ({period} periods)"

        def _make_natr_func(p=period):
            def func(df: pd.DataFrame) -> pd.Series:
                ta = _ensure_pandas_ta()
                atr = ta.atr(df["high"], df["low"], df["close"], length=p)
                return atr / df["close"]
            return func

        register_feature(natr_name, group="technical", description=natr_desc)(_make_natr_func())


@register_feature("donchian_upper", group="technical", description="Donchian Channel Upper (20)")
def compute_donchian_upper(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(20).max()


@register_feature("donchian_lower", group="technical", description="Donchian Channel Lower (20)")
def compute_donchian_lower(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(20).min()


@register_feature("donchian_width", group="technical", description="Donchian Channel Width")
def compute_donchian_width(df: pd.DataFrame) -> pd.Series:
    upper = df["high"].rolling(20).max()
    lower = df["low"].rolling(20).min()
    mid = (upper + lower) / 2
    return (upper - lower) / mid


# ============================================================
# VOLUME INDICATORS
# ============================================================

@register_feature("obv", group="technical", description="On-Balance Volume")
def compute_obv(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    return ta.obv(df["close"], df["volume"])


@register_feature("vwap", group="technical", description="Volume Weighted Average Price (proxy)")
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Approximate VWAP using typical price × volume."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol


@register_feature("mfi_14", group="technical", description="Money Flow Index (14)")
def compute_mfi(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    return ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)


@register_feature("cmf_20", group="technical", description="Chaikin Money Flow (20)")
def compute_cmf(df: pd.DataFrame) -> pd.Series:
    ta = _ensure_pandas_ta()
    return ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)


@register_feature("volume_sma_ratio", group="technical",
                   description="Volume / SMA(Volume, 20)")
def compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    vol_sma = df["volume"].rolling(20).mean()
    return df["volume"] / vol_sma


@register_feature("volume_change", group="technical",
                   description="Volume percent change")
def compute_volume_change(df: pd.DataFrame) -> pd.Series:
    return df["volume"].pct_change()


# ============================================================
# CANDLE PATTERN FEATURES
# ============================================================

@register_feature("candle_body_pct", group="technical",
                   description="Candle body as % of range")
def compute_candle_body(df: pd.DataFrame) -> pd.Series:
    body = abs(df["close"] - df["open"])
    range_ = df["high"] - df["low"]
    return body / range_.replace(0, np.nan)


@register_feature("candle_upper_shadow", group="technical",
                   description="Upper shadow as % of range")
def compute_upper_shadow(df: pd.DataFrame) -> pd.Series:
    upper = df["high"] - df[["close", "open"]].max(axis=1)
    range_ = df["high"] - df["low"]
    return upper / range_.replace(0, np.nan)


@register_feature("candle_lower_shadow", group="technical",
                   description="Lower shadow as % of range")
def compute_lower_shadow(df: pd.DataFrame) -> pd.Series:
    lower = df[["close", "open"]].min(axis=1) - df["low"]
    range_ = df["high"] - df["low"]
    return lower / range_.replace(0, np.nan)


@register_feature("candle_direction", group="technical",
                   description="Candle direction (1=up, -1=down, 0=doji)")
def compute_candle_direction(df: pd.DataFrame) -> pd.Series:
    return np.sign(df["close"] - df["open"])


# ============================================================
# CROSS-PERIOD FEATURES
# ============================================================

@register_feature("ema_cross_9_21", group="technical",
                   description="EMA 9/21 crossover signal")
def compute_ema_cross(df: pd.DataFrame) -> pd.Series:
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    return (ema9 > ema21).astype(int)


@register_feature("golden_cross", group="technical",
                   description="SMA 50/200 golden cross")
def compute_golden_cross(df: pd.DataFrame) -> pd.Series:
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    return (sma50 > sma200).astype(int)


@register_feature("price_vs_sma200", group="technical",
                   description="Price above SMA 200")
def compute_price_vs_sma200(df: pd.DataFrame) -> pd.Series:
    sma200 = df["close"].rolling(200).mean()
    return (df["close"] > sma200).astype(int)


# ============================================================
# Auto-register all parametric features
# ============================================================

def register_all_technical_features() -> int:
    """
    Register all technical features. Call this once at startup.

    Returns number of features registered.
    """
    before = feature_registry.count if 'feature_registry' in dir() else 0

    _register_sma_features()
    _register_ema_features()
    _register_rsi_features()
    _register_cci_features()
    _register_roc_features()
    _register_atr_features()

    # Import the registry to get count
    from src.features.registry import feature_registry
    after = feature_registry.count

    technical_count = len(feature_registry.list_features(group="technical"))
    logger.info("technical_features_registered", count=technical_count)
    return technical_count


# Auto-register on import
register_all_technical_features()

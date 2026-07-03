"""
Strategy Templates (Fase 3)
=============================
Pre-defined strategy templates for backtesting and GA optimization.

TODO:
- [ ] EMA Crossover strategy
- [ ] RSI Mean Reversion strategy
- [ ] Bollinger Band Breakout strategy
- [ ] MACD Momentum strategy
- [ ] Multi-indicator composite strategy
- [ ] ML Signal-based strategy
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.types import SignalType


@dataclass
class StrategyParams:
    """Parameters for a trading strategy."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, params: StrategyParams) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV + features data.

        Returns:
            Series of SignalType values aligned with df index.
        """
        ...

    @property
    def name(self) -> str:
        return self.params.name


class EMACrossStrategy(BaseStrategy):
    """EMA Crossover strategy template."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = self.params.params.get("fast_period", 9)
        slow = self.params.params.get("slow_period", 21)

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        signals = pd.Series(SignalType.HOLD.value, index=df.index)
        signals[ema_fast > ema_slow] = SignalType.BUY.value
        signals[ema_fast < ema_slow] = SignalType.SELL.value

        return signals


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI Mean Reversion strategy template."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.params.get("period", 14)
        oversold = self.params.params.get("oversold", 30)
        overbought = self.params.params.get("overbought", 70)

        # Simple RSI calculation
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(SignalType.HOLD.value, index=df.index)
        signals[rsi < oversold] = SignalType.BUY.value
        signals[rsi > overbought] = SignalType.SELL.value

        return signals


class BollingerBreakoutStrategy(BaseStrategy):
    """Bollinger Band Breakout: buy above upper band, sell below lower band."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.params.get("period", 20)
        num_std = self.params.params.get("num_std", 2.0)

        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = ma + num_std * std
        lower = ma - num_std * std

        signals = pd.Series(SignalType.HOLD.value, index=df.index)
        signals[df["close"] > upper] = SignalType.BUY.value
        signals[df["close"] < lower] = SignalType.SELL.value

        return signals


class MACDMomentumStrategy(BaseStrategy):
    """MACD Momentum: buy when MACD crosses above signal line, sell below."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = self.params.params.get("fast_period", 12)
        slow = self.params.params.get("slow_period", 26)
        signal_period = self.params.params.get("signal_period", 9)

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()

        signals = pd.Series(SignalType.HOLD.value, index=df.index)
        signals[macd > signal_line] = SignalType.BUY.value
        signals[macd < signal_line] = SignalType.SELL.value

        return signals


class MLSignalStrategy(BaseStrategy):
    """
    Wraps model predictions as a strategy: expects a column of predicted
    probabilities (params: proba_column, buy_threshold, sell_threshold).
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        col = self.params.params.get("proba_column", "pred_proba")
        buy_th = self.params.params.get("buy_threshold", 0.6)
        sell_th = self.params.params.get("sell_threshold", 0.4)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

        signals = pd.Series(SignalType.HOLD.value, index=df.index)
        signals[df[col] >= buy_th] = SignalType.BUY.value
        signals[df[col] <= sell_th] = SignalType.SELL.value

        return signals

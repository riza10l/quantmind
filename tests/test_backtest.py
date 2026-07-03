"""Tests for the backtesting engine (Fase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.core.types import SignalType
from src.strategy.templates import (
    BollingerBreakoutStrategy,
    EMACrossStrategy,
    MACDMomentumStrategy,
    StrategyParams,
)


@pytest.fixture
def trending_df() -> pd.DataFrame:
    """Steadily rising market — a long-only strategy should profit."""
    rng = np.random.default_rng(7)
    n = 300
    drift = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n).cumsum()
    close = drift + noise
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    }, index=idx)


def test_buy_and_hold_profits_in_uptrend(trending_df):
    signals = pd.Series(SignalType.HOLD.value, index=trending_df.index)
    signals.iloc[0] = SignalType.BUY.value
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000))
    result = engine.run(trending_df, signals=signals, symbol="TEST")
    assert result.final_equity > 10_000
    assert result.total_trades == 1  # position force-closed at end
    assert result.equity_curve is not None
    assert len(result.equity_curve) == len(trending_df)


def test_costs_reduce_equity(trending_df):
    signals = pd.Series(SignalType.HOLD.value, index=trending_df.index)
    signals.iloc[::10] = SignalType.BUY.value
    signals.iloc[5::10] = SignalType.SELL.value
    free = BacktestEngine(BacktestConfig(commission_pct=0.0, slippage_pct=0.0)).run(
        trending_df, signals=signals)
    costly = BacktestEngine(BacktestConfig(commission_pct=0.005, slippage_pct=0.002)).run(
        trending_df, signals=signals)
    assert costly.final_equity < free.final_equity
    assert costly.total_commission > 0
    assert costly.total_slippage > 0


def test_strategies_produce_valid_results(trending_df):
    engine = BacktestEngine()
    for cls in (EMACrossStrategy, BollingerBreakoutStrategy, MACDMomentumStrategy):
        strategy = cls(StrategyParams(name=cls.__name__))
        result = engine.run(trending_df, strategy)
        assert result.strategy_name == cls.__name__
        assert result.initial_capital == 10_000
        assert np.isfinite(result.final_equity)


def test_requires_strategy_or_signals(trending_df):
    with pytest.raises(ValueError):
        BacktestEngine().run(trending_df)


def test_walk_forward(trending_df):
    engine = BacktestEngine()
    wf = engine.walk_forward(
        trending_df,
        strategy_factory=lambda train: EMACrossStrategy(StrategyParams(name="ema")),
        train_size=100,
        test_size=50,
    )
    assert len(wf.window_results) == 4  # (300 - 100) // 50
    assert np.isfinite(wf.mean_sharpe)
    assert np.isfinite(wf.mean_return)


def test_monte_carlo(trending_df):
    engine = BacktestEngine()
    strategy = EMACrossStrategy(StrategyParams(name="ema"))
    result = engine.run(trending_df, strategy)
    mc = engine.monte_carlo(result, n_simulations=200)
    assert len(mc.final_equities) == 200
    assert 0.0 <= mc.prob_loss <= 1.0
    assert mc.var_95_equity <= mc.median_equity

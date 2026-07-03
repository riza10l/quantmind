"""
Multi-Asset Portfolio Backtester
=================================
Simulates a weighted portfolio across assets with periodic rebalancing,
transaction costs on rebalance turnover, correlation/exposure analysis,
and benchmark comparison.

Returns-based (vectorized) simulation: weights drift with prices between
rebalance dates, then snap back to targets minus rebalance costs.

Usage:
    bt = PortfolioBacktester(initial_capital=10_000, rebalance="M")
    result = bt.run({"BTC-USD": df1, "ETH-USD": df2}, weights="equal",
                    benchmark=spy_df)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import PerformanceMetrics, compute_all_metrics
from src.core.logger import get_logger
from src.portfolio.risk_engine import beta_alpha

logger = get_logger("backtest.portfolio")

REBALANCE_FREQ = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "never": None}


@dataclass
class PortfolioResult:
    """Multi-asset backtest output."""
    symbols: list[str]
    target_weights: dict[str, float]
    rebalance: str
    initial_capital: float
    equity_curve: pd.Series
    weights_history: pd.DataFrame       # realized weights per bar (drift visible)
    correlation: pd.DataFrame           # asset return correlation matrix
    metrics: PerformanceMetrics
    total_rebalances: int
    total_cost: float
    benchmark_metrics: PerformanceMetrics | None = None
    beta: float | None = None
    alpha: float | None = None

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def max_exposure(self) -> pd.Series:
        """Peak realized weight per asset (concentration check)."""
        return self.weights_history.max()


class PortfolioBacktester:
    """Weighted multi-asset portfolio simulation with rebalancing."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        rebalance: str = "M",
        cost_pct: float = 0.001,   # cost per unit of turnover (one side)
    ) -> None:
        if rebalance not in REBALANCE_FREQ:
            raise ValueError(f"rebalance must be one of {sorted(REBALANCE_FREQ)}")
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = initial_capital
        self.rebalance = rebalance
        self.cost_pct = cost_pct

    def run(
        self,
        data: dict[str, pd.DataFrame],
        weights: dict[str, float] | str = "equal",
        benchmark: pd.DataFrame | None = None,
    ) -> PortfolioResult:
        """
        Args:
            data: symbol -> OHLCV DataFrame (datetime index, 'close' column).
            weights: target weights per symbol, or "equal".
            benchmark: optional OHLCV DataFrame for relative metrics.
        """
        if len(data) < 2:
            raise ValueError("portfolio backtest needs at least 2 assets")

        closes = pd.DataFrame({s: df["close"] for s, df in data.items()}).dropna()
        if len(closes) < 30:
            raise ValueError(
                f"only {len(closes)} overlapping bars across assets; need >= 30"
            )
        symbols = list(closes.columns)

        if weights == "equal":
            target = dict.fromkeys(symbols, 1.0 / len(symbols))
        else:
            missing = set(symbols) - set(weights)
            if missing:
                raise ValueError(f"weights missing for: {sorted(missing)}")
            total = sum(weights[s] for s in symbols)
            if total <= 0:
                raise ValueError("weights must sum to a positive number")
            target = {s: weights[s] / total for s in symbols}

        returns = closes.pct_change().fillna(0.0)
        rebalance_dates = self._rebalance_dates(closes.index)

        w = np.array([target[s] for s in symbols])       # current weights
        equity = np.empty(len(closes))
        weights_hist = np.empty((len(closes), len(symbols)))
        value = self.initial_capital
        n_rebalances = 0
        total_cost = 0.0

        for i, (ts, row) in enumerate(zip(closes.index, returns.to_numpy(), strict=True)):
            # Grow value; weights drift with relative asset performance
            growth = 1.0 + float(np.dot(w, row))
            value *= growth
            if growth > 0:
                w = w * (1.0 + row) / growth
            if ts in rebalance_dates and i > 0:
                turnover = float(np.abs(w - [target[s] for s in symbols]).sum())
                cost = value * turnover * self.cost_pct
                value -= cost
                total_cost += cost
                w = np.array([target[s] for s in symbols])
                n_rebalances += 1
            equity[i] = value
            weights_hist[i] = w

        equity_curve = pd.Series(equity, index=closes.index, name="equity")
        port_returns = equity_curve.pct_change().fillna(0.0)
        metrics = compute_all_metrics(port_returns, equity_curve, trade_pnls=[])

        bench_metrics = beta = alpha = None
        if benchmark is not None and not benchmark.empty:
            bench_close = benchmark["close"].reindex(closes.index).ffill().dropna()
            bench_returns = bench_close.pct_change().fillna(0.0)
            bench_equity = self.initial_capital * (1 + bench_returns).cumprod()
            bench_metrics = compute_all_metrics(bench_returns, bench_equity, trade_pnls=[])
            beta, alpha = beta_alpha(port_returns, bench_returns)

        result = PortfolioResult(
            symbols=symbols,
            target_weights=target,
            rebalance=self.rebalance,
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            weights_history=pd.DataFrame(weights_hist, index=closes.index, columns=symbols),
            correlation=returns.corr(),
            metrics=metrics,
            total_rebalances=n_rebalances,
            total_cost=total_cost,
            benchmark_metrics=bench_metrics,
            beta=beta,
            alpha=alpha,
        )
        logger.info(
            "portfolio_backtest_complete",
            assets=len(symbols), bars=len(closes),
            rebalances=n_rebalances,
            total_return=round(metrics.total_return, 4),
            sharpe=round(metrics.sharpe_ratio, 3),
        )
        return result

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> set:
        freq = REBALANCE_FREQ[self.rebalance]
        if freq is None:
            return set()
        # First available bar at/after each period start
        periods = pd.date_range(index[0], index[-1], freq=freq)
        positions = index.searchsorted(periods)
        return {index[p] for p in positions if p < len(index)}

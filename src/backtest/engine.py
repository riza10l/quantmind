"""
Backtesting Engine (Fase 3)
============================
Event-driven backtesting with realistic transaction costs, slippage,
walk-forward analysis, and Monte Carlo robustness testing.

Usage:
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(df, strategy)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_all_metrics
from src.core.logger import get_logger
from src.core.types import BacktestResult, Side, SignalType, Timeframe, Trade
from src.strategy.templates import BaseStrategy

logger = get_logger("backtest.engine")


@dataclass
class BacktestConfig:
    """Simulation cost/execution assumptions."""
    initial_capital: float = 10_000.0
    commission_pct: float = 0.001      # 0.1% per side (Binance taker)
    slippage_pct: float = 0.0005       # 5 bps adverse fill
    position_size_pct: float = 1.0     # fraction of equity per trade
    allow_short: bool = False
    fill_on_next_open: bool = True     # signal at close, fill next open (no look-ahead)


@dataclass
class WalkForwardResult:
    """Aggregated results across walk-forward windows."""
    window_results: list[BacktestResult] = field(default_factory=list)

    @property
    def mean_sharpe(self) -> float:
        if not self.window_results:
            return 0.0
        return float(np.mean([r.sharpe_ratio for r in self.window_results]))

    @property
    def mean_return(self) -> float:
        if not self.window_results:
            return 0.0
        return float(np.mean([r.total_return for r in self.window_results]))

    @property
    def worst_drawdown(self) -> float:
        if not self.window_results:
            return 0.0
        return float(min(r.max_drawdown for r in self.window_results))


@dataclass
class MonteCarloResult:
    """Distribution of outcomes from bootstrap-resampled trade sequences."""
    initial_capital: float
    final_equities: np.ndarray
    max_drawdowns: np.ndarray

    @property
    def median_equity(self) -> float:
        return float(np.median(self.final_equities))

    @property
    def var_95_equity(self) -> float:
        """5th percentile of final equity (pessimistic outcome)."""
        return float(np.percentile(self.final_equities, 5))

    @property
    def prob_loss(self) -> float:
        """Probability the strategy ends below initial capital."""
        return float(np.mean(self.final_equities < self.initial_capital))


class BacktestEngine:
    """
    Bar-by-bar backtester driven by strategy signals.

    Signals are generated on bar close; fills happen at the next bar's open
    (plus slippage) to prevent look-ahead bias. Long-only by default;
    a SELL signal closes the position (or opens a short if allow_short).
    """

    def __init__(self, config: BacktestConfig | None = None, **kwargs: Any) -> None:
        self.config = config or BacktestConfig(**kwargs)

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy | None = None,
        signals: pd.Series | None = None,
        symbol: str = "UNKNOWN",
        timeframe: Timeframe = Timeframe.D1,
    ) -> BacktestResult:
        """
        Run a backtest over OHLCV data.

        Args:
            df: DataFrame with open/high/low/close columns, datetime-like index.
            strategy: Strategy that generates signals (or pass signals directly).
            signals: Precomputed signal Series of SignalType values.
        """
        if strategy is None and signals is None:
            raise ValueError("Provide either a strategy or a signals Series")
        if signals is None:
            signals = strategy.generate_signals(df)
        signals = signals.reindex(df.index).fillna(SignalType.HOLD.value)

        cfg = self.config
        cash = cfg.initial_capital
        qty = 0.0            # signed: >0 long, <0 short
        entry_price = 0.0
        entry_time: datetime | None = None
        trades: list[Trade] = []
        equity = np.empty(len(df), dtype=float)
        total_commission = 0.0
        total_slippage = 0.0

        opens = df["open"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        index = df.index

        def to_dt(ts: Any) -> datetime:
            return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

        def fill_price(base: float, side: Side) -> float:
            slip = base * cfg.slippage_pct
            return base + slip if side == Side.BUY else base - slip

        def close_position(price: float, ts: datetime) -> None:
            nonlocal cash, qty, entry_price, entry_time
            nonlocal total_commission, total_slippage
            side = Side.SELL if qty > 0 else Side.BUY  # closing direction
            px = fill_price(price, side)
            gross = qty * (px - entry_price)
            commission = abs(qty) * px * cfg.commission_pct
            slippage_cost = abs(qty) * price * cfg.slippage_pct
            cash += abs(qty) * entry_price + gross - commission
            pnl = gross - commission
            trades.append(Trade(
                id=str(uuid.uuid4())[:8],
                symbol=symbol,
                side=Side.BUY if qty > 0 else Side.SELL,
                entry_price=entry_price,
                exit_price=px,
                quantity=abs(qty),
                entry_time=entry_time or ts,
                exit_time=ts,
                pnl=pnl,
                pnl_pct=pnl / (abs(qty) * entry_price) if entry_price else 0.0,
                commission=commission,
                slippage=slippage_cost,
                holding_period_minutes=int((ts - (entry_time or ts)).total_seconds() // 60),
            ))
            total_commission += commission
            total_slippage += slippage_cost
            qty = 0.0
            entry_price = 0.0
            entry_time = None

        def open_position(price: float, ts: datetime, direction: int) -> None:
            nonlocal cash, qty, entry_price, entry_time, total_commission
            side = Side.BUY if direction > 0 else Side.SELL
            px = fill_price(price, side)
            budget = cash * cfg.position_size_pct
            size = budget / (px * (1 + cfg.commission_pct))
            if size <= 0:
                return
            commission = size * px * cfg.commission_pct
            cash -= size * px + commission
            total_commission += commission
            qty = direction * size
            entry_price = px
            entry_time = ts

        for i in range(len(df)):
            ts = to_dt(index[i])
            # Execute the previous bar's signal at this bar's open (no look-ahead)
            sig_idx = i - 1 if cfg.fill_on_next_open else i
            exec_price = opens[i] if cfg.fill_on_next_open else closes[i]
            if sig_idx >= 0:
                sig = signals.iloc[sig_idx]
                sig = sig.value if isinstance(sig, SignalType) else sig
                if sig == SignalType.BUY.value and qty <= 0:
                    if qty < 0:
                        close_position(exec_price, ts)
                    open_position(exec_price, ts, +1)
                elif sig == SignalType.SELL.value and qty >= 0:
                    if qty > 0:
                        close_position(exec_price, ts)
                    if cfg.allow_short:
                        open_position(exec_price, ts, -1)
            # Mark-to-market at close
            position_value = abs(qty) * entry_price + qty * (closes[i] - entry_price) if qty != 0 else 0.0
            equity[i] = cash + position_value

        # Close any open position at final close
        if qty != 0:
            close_position(closes[-1], to_dt(index[-1]))
            equity[-1] = cash

        equity_curve = pd.Series(equity, index=index, name="equity")
        returns = equity_curve.pct_change().fillna(0.0)
        trade_pnls = [t.pnl for t in trades]
        m = compute_all_metrics(returns, equity_curve, trade_pnls)

        result = BacktestResult(
            strategy_name=strategy.name if strategy else "signals",
            symbol=symbol,
            timeframe=timeframe,
            start_date=to_dt(index[0]),
            end_date=to_dt(index[-1]),
            initial_capital=cfg.initial_capital,
            final_equity=float(equity_curve.iloc[-1]),
            total_return=m.total_return,
            sharpe_ratio=m.sharpe_ratio,
            sortino_ratio=m.sortino_ratio,
            calmar_ratio=m.calmar_ratio,
            max_drawdown=m.max_drawdown,
            max_drawdown_duration_days=m.max_drawdown_duration_days,
            win_rate=m.win_rate,
            profit_factor=m.profit_factor,
            total_trades=m.total_trades,
            avg_trade_pnl=m.avg_trade_pnl,
            avg_holding_period_minutes=float(np.mean([t.holding_period_minutes for t in trades])) if trades else 0.0,
            total_commission=total_commission,
            total_slippage=total_slippage,
            equity_curve=equity_curve,
            trades=trades,
        )
        logger.info(
            "backtest_complete",
            strategy=result.strategy_name,
            trades=result.total_trades,
            total_return=round(result.total_return, 4),
            sharpe=round(result.sharpe_ratio, 3),
        )
        return result

    # ------------------------------------------------------------------
    # Walk-forward analysis
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame], BaseStrategy],
        train_size: int = 252,
        test_size: int = 63,
        **run_kwargs: Any,
    ) -> WalkForwardResult:
        """
        Rolling walk-forward: build/optimize a strategy on each train window,
        evaluate it out-of-sample on the following test window.

        Args:
            strategy_factory: Called with the train slice, returns a strategy
                (e.g., after GA parameter optimization) to test out-of-sample.
        """
        results = []
        start = 0
        while start + train_size + test_size <= len(df):
            train = df.iloc[start:start + train_size]
            test = df.iloc[start + train_size:start + train_size + test_size]
            strategy = strategy_factory(train)
            results.append(self.run(test, strategy, **run_kwargs))
            start += test_size
        return WalkForwardResult(window_results=results)

    # ------------------------------------------------------------------
    # Monte Carlo robustness
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        seed: int | None = 42,
    ) -> MonteCarloResult:
        """
        Bootstrap trade PnLs to estimate the distribution of outcomes.
        Answers: how much of the backtest result is luck of trade ordering?
        """
        rng = np.random.default_rng(seed)
        pnls = np.array([t.pnl for t in result.trades], dtype=float)
        if len(pnls) == 0:
            return MonteCarloResult(
                initial_capital=result.initial_capital,
                final_equities=np.full(n_simulations, result.initial_capital),
                max_drawdowns=np.zeros(n_simulations),
            )
        finals = np.empty(n_simulations)
        max_dds = np.empty(n_simulations)
        for i in range(n_simulations):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            curve = result.initial_capital + np.cumsum(sample)
            peak = np.maximum.accumulate(np.maximum(curve, 1e-9))
            dd = (curve - peak) / peak
            finals[i] = curve[-1]
            max_dds[i] = dd.min()
        return MonteCarloResult(
            initial_capital=result.initial_capital,
            final_equities=finals,
            max_drawdowns=max_dds,
        )

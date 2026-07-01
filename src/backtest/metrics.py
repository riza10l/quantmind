"""
Backtest Performance Metrics (Fase 3)
=======================================
Comprehensive performance metrics for strategy evaluation.

Implemented early because feature selection and model evaluation
can reuse these metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Complete set of backtest performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    volatility: float
    downside_volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    expectancy: float
    recovery_factor: float
    payoff_ratio: float


def compute_sharpe(returns: pd.Series, risk_free: float = 0.05, periods: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def compute_sortino(returns: pd.Series, risk_free: float = 0.05, periods: int = 252) -> float:
    """Annualized Sortino ratio (penalizes downside only)."""
    excess = returns - risk_free / periods
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """
    Compute maximum drawdown and its duration.

    Returns:
        (max_drawdown_pct, max_drawdown_duration_days)
    """
    if equity_curve.empty:
        return 0.0, 0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Duration: longest period below previous peak
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    if in_drawdown.any():
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_duration = int(dd_lengths.max())
    else:
        max_duration = 0

    return max_dd, max_duration


def compute_calmar(returns: pd.Series, equity_curve: pd.Series, periods: int = 252) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    ann_return = float(returns.mean() * periods)
    max_dd, _ = compute_max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0
    return ann_return / abs(max_dd)


def compute_profit_factor(trade_pnls: list[float]) -> float:
    """Profit factor = gross profits / gross losses."""
    if not trade_pnls:
        return 0.0
    wins = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def compute_win_rate(trade_pnls: list[float]) -> float:
    """Win rate as a fraction of total trades."""
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def compute_expectancy(trade_pnls: list[float]) -> float:
    """Expected value per trade = win_rate × avg_win - loss_rate × avg_loss."""
    if not trade_pnls:
        return 0.0
    wins = [p for p in trade_pnls if p > 0]
    losses = [abs(p) for p in trade_pnls if p < 0]
    win_rate = len(wins) / len(trade_pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    return float(win_rate * avg_win - (1 - win_rate) * avg_loss)


def compute_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trade_pnls: list[float],
    risk_free: float = 0.05,
    periods: int = 252,
) -> PerformanceMetrics:
    """Compute all performance metrics in one call."""
    max_dd, max_dd_dur = compute_max_drawdown(equity_curve)
    wins = [p for p in trade_pnls if p > 0]
    losses = [abs(p) for p in trade_pnls if p < 0]

    return PerformanceMetrics(
        total_return=float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) if len(equity_curve) > 1 else 0.0,
        annualized_return=float(returns.mean() * periods),
        sharpe_ratio=compute_sharpe(returns, risk_free, periods),
        sortino_ratio=compute_sortino(returns, risk_free, periods),
        calmar_ratio=compute_calmar(returns, equity_curve, periods),
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        volatility=float(returns.std() * np.sqrt(periods)),
        downside_volatility=float(returns[returns < 0].std() * np.sqrt(periods)) if (returns < 0).any() else 0.0,
        win_rate=compute_win_rate(trade_pnls),
        profit_factor=compute_profit_factor(trade_pnls),
        total_trades=len(trade_pnls),
        avg_trade_pnl=float(np.mean(trade_pnls)) if trade_pnls else 0.0,
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        best_trade=float(max(trade_pnls)) if trade_pnls else 0.0,
        worst_trade=float(min(trade_pnls)) if trade_pnls else 0.0,
        avg_holding_period=0.0,  # Requires trade timestamps
        expectancy=compute_expectancy(trade_pnls),
        recovery_factor=abs(float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) / max_dd) if max_dd != 0 else 0.0,
        payoff_ratio=float(np.mean(wins) / np.mean(losses)) if wins and losses else 0.0,
    )

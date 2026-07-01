"""
RL Reward Functions (Fase 3)
==============================
Custom reward functions for the trading environment.

TODO:
- [ ] Sharpe-based reward: reward = Δ(rolling_sharpe)
- [ ] Sortino-based reward: penalize downside more
- [ ] PnL with drawdown penalty: reward = pnl × (1 - dd_penalty)
- [ ] Transaction cost deduction
- [ ] Risk-adjusted reward shaping
"""

from __future__ import annotations

import numpy as np


def sharpe_reward(returns: np.ndarray, window: int = 50, risk_free: float = 0.0) -> float:
    """Compute Sharpe-based reward from recent returns."""
    if len(returns) < window:
        return 0.0
    recent = returns[-window:]
    excess = recent - risk_free / 252
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252))


def sortino_reward(returns: np.ndarray, window: int = 50, risk_free: float = 0.0) -> float:
    """Compute Sortino-based reward (penalizes downside volatility)."""
    if len(returns) < window:
        return 0.0
    recent = returns[-window:]
    excess = recent - risk_free / 252
    downside = recent[recent < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float(np.mean(excess) * np.sqrt(252)) if np.mean(excess) > 0 else 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(252))


def pnl_drawdown_reward(
    pnl: float,
    drawdown: float,
    transaction_cost: float = 0.0,
    dd_penalty: float = 0.5,
) -> float:
    """PnL reward with drawdown penalty and transaction cost."""
    return pnl * (1.0 - dd_penalty * abs(drawdown)) - transaction_cost

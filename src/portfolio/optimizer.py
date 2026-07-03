"""
Portfolio Optimizer (Fase 4)
=============================
Position sizing and multi-asset allocation:
- Kelly Criterion (with fractional Kelly safety)
- Mean-Variance (max Sharpe) via scipy
- Risk Parity (inverse-volatility and equal risk contribution)
- Mean-CVaR minimization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.core.logger import get_logger

logger = get_logger("portfolio.optimizer")


def kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    fraction: float = 0.5,
    cap: float = 0.25,
) -> float:
    """
    Kelly position size: f* = W - (1-W)/R, scaled by `fraction`
    (half-Kelly default) and capped for safety.
    """
    if payoff_ratio <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / payoff_ratio
    return float(np.clip(f * fraction, 0.0, cap))


def kelly_from_returns(returns: pd.Series, fraction: float = 0.5, cap: float = 0.25) -> float:
    """Estimate Kelly fraction from a return series (continuous Kelly: mu/sigma^2)."""
    var = returns.var()
    if var == 0 or len(returns) < 2:
        return 0.0
    f = returns.mean() / var
    return float(np.clip(f * fraction, 0.0, cap))


def inverse_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    """Naive risk parity: weight each asset by 1/volatility."""
    vol = returns.std()
    inv = 1.0 / vol.replace(0, np.nan)
    weights = inv / inv.sum()
    return weights.fillna(0.0)


class PortfolioOptimizer:
    """
    Multi-asset weight optimizer over a returns DataFrame
    (columns = assets, rows = periods).
    """

    def __init__(self, returns: pd.DataFrame, risk_free: float = 0.05, periods: int = 252) -> None:
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets to optimize")
        self.returns = returns.dropna()
        self.risk_free = risk_free
        self.periods = periods
        self.mean = self.returns.mean().to_numpy() * periods
        self.cov = self.returns.cov().to_numpy() * periods
        self.n = returns.shape[1]

    def _solve(self, objective, extra_constraints=()) -> pd.Series:
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
        bounds = [(0.0, 1.0)] * self.n
        w0 = np.full(self.n, 1.0 / self.n)
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success:
            logger.warning("optimizer_fallback_equal_weight", message=res.message)
            return pd.Series(w0, index=self.returns.columns)
        return pd.Series(res.x, index=self.returns.columns)

    def max_sharpe(self) -> pd.Series:
        """Mean-variance optimization: maximize the Sharpe ratio."""
        def neg_sharpe(w: np.ndarray) -> float:
            ret = w @ self.mean
            vol = np.sqrt(w @ self.cov @ w)
            return -(ret - self.risk_free) / vol if vol > 0 else 0.0
        return self._solve(neg_sharpe)

    def min_volatility(self) -> pd.Series:
        """Minimum variance portfolio."""
        return self._solve(lambda w: w @ self.cov @ w)

    def risk_parity(self) -> pd.Series:
        """Equal Risk Contribution: each asset contributes equally to portfolio risk."""
        def erc_objective(w: np.ndarray) -> float:
            port_var = w @ self.cov @ w
            marginal = self.cov @ w
            contrib = w * marginal
            target = port_var / self.n
            return float(np.sum((contrib - target) ** 2))
        return self._solve(erc_objective)

    def min_cvar(self, alpha: float = 0.05) -> pd.Series:
        """Minimize historical CVaR (expected shortfall) of the portfolio."""
        r = self.returns.to_numpy()

        def cvar(w: np.ndarray) -> float:
            port = r @ w
            var = np.quantile(port, alpha)
            tail = port[port <= var]
            return -float(tail.mean()) if len(tail) else 0.0
        return self._solve(cvar)

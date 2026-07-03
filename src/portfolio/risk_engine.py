"""
Risk Engine (Fase 4)
=====================
Pre-trade risk checks and portfolio risk metrics:
VaR (historical/parametric/Monte Carlo), CVaR, beta/alpha,
risk of ruin, tail metrics, and a position size limiter.

The circuit breaker (drawdown auto-stop) lives in circuit_breaker.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.core.logger import get_logger

logger = get_logger("portfolio.risk_engine")


# ============================================================
# Risk metrics
# ============================================================


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR: the alpha-quantile loss (returned as a positive number)."""
    if len(returns) < 2:
        return 0.0
    return float(-np.quantile(returns, alpha))


def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Gaussian VaR from mean and std of returns."""
    if len(returns) < 2:
        return 0.0
    z = stats.norm.ppf(alpha)
    return float(-(returns.mean() + z * returns.std()))


def monte_carlo_var(
    returns: pd.Series, alpha: float = 0.05, n_simulations: int = 10_000, seed: int = 42
) -> float:
    """Monte Carlo VaR: simulate returns from fitted normal distribution."""
    if len(returns) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    sims = rng.normal(returns.mean(), returns.std(), n_simulations)
    return float(-np.quantile(sims, alpha))


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional VaR / Expected Shortfall: mean loss beyond VaR."""
    if len(returns) < 2:
        return 0.0
    var_threshold = np.quantile(returns, alpha)
    tail = returns[returns <= var_threshold]
    return float(-tail.mean()) if len(tail) else 0.0


def beta_alpha(returns: pd.Series, benchmark: pd.Series, periods: int = 252) -> tuple[float, float]:
    """CAPM beta and annualized alpha vs a benchmark return series."""
    joined = pd.concat([returns, benchmark], axis=1).dropna()
    if len(joined) < 2:
        return 0.0, 0.0
    r, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = b.var()
    if var_b == 0:
        return 0.0, 0.0
    beta = float(r.cov(b) / var_b)
    alpha = float((r.mean() - beta * b.mean()) * periods)
    return beta, alpha


def risk_of_ruin(win_rate: float, payoff_ratio: float, risk_per_trade: float, ruin_level: float = 0.5) -> float:
    """
    Probability of losing `ruin_level` of capital, using the classic
    gambler's-ruin approximation with fixed fractional risk.
    """
    if risk_per_trade <= 0 or win_rate <= 0:
        return 1.0
    if win_rate >= 1.0:
        return 0.0
    edge = win_rate * payoff_ratio - (1 - win_rate)
    if edge <= 0:
        return 1.0
    # ponytail: standard approximation ((1-e)/(1+e))^n; a full Monte Carlo sim is the upgrade path
    units_to_ruin = ruin_level / risk_per_trade
    base = (1 - edge) / (1 + edge)
    return float(np.clip(base ** units_to_ruin, 0.0, 1.0))


def tail_metrics(returns: pd.Series) -> dict[str, float]:
    """Skewness and excess kurtosis — negative skew + fat tails = danger."""
    if len(returns) < 4:
        return {"skewness": 0.0, "kurtosis": 0.0}
    return {
        "skewness": float(stats.skew(returns.dropna())),
        "kurtosis": float(stats.kurtosis(returns.dropna())),
    }


# ============================================================
# Pre-trade risk engine
# ============================================================


@dataclass
class RiskLimits:
    """Hard limits enforced before any order goes out."""
    max_position_pct: float = 0.25      # max single-position size vs equity
    max_leverage: float = 1.0
    max_var_95: float = 0.05            # daily VaR limit
    min_equity: float = 0.0             # halt if equity falls below


@dataclass
class RiskCheckResult:
    approved: bool
    adjusted_quantity: float
    reason: str = ""


class RiskEngine:
    """Validates and caps orders against portfolio risk limits."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def check_order(
        self,
        quantity: float,
        price: float,
        equity: float,
        current_exposure: float = 0.0,
        recent_returns: pd.Series | None = None,
    ) -> RiskCheckResult:
        """
        Pre-trade check. Returns an approved flag and a (possibly reduced)
        quantity that fits within the limits.
        """
        if equity <= self.limits.min_equity or equity <= 0:
            return RiskCheckResult(False, 0.0, "equity below minimum")

        if recent_returns is not None and len(recent_returns) > 20:
            var = historical_var(recent_returns)
            if var > self.limits.max_var_95:
                return RiskCheckResult(False, 0.0, f"portfolio VaR {var:.3f} exceeds limit {self.limits.max_var_95}")

        order_value = abs(quantity) * price
        max_position_value = equity * self.limits.max_position_pct
        max_leverage_value = max(equity * self.limits.max_leverage - current_exposure, 0.0)
        allowed_value = min(max_position_value, max_leverage_value)

        if allowed_value <= 0:
            return RiskCheckResult(False, 0.0, "leverage limit reached")

        if order_value <= allowed_value:
            return RiskCheckResult(True, quantity)

        adjusted = (allowed_value / price) * np.sign(quantity)
        logger.info("order_size_capped", requested=quantity, adjusted=float(adjusted))
        return RiskCheckResult(True, float(adjusted), "size capped by risk limits")

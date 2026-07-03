"""Tests for portfolio optimization and risk engine (Fase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimizer import (
    PortfolioOptimizer,
    inverse_volatility_weights,
    kelly_fraction,
    kelly_from_returns,
)
from src.portfolio.risk_engine import (
    RiskEngine,
    RiskLimits,
    beta_alpha,
    cvar,
    historical_var,
    monte_carlo_var,
    parametric_var,
    risk_of_ruin,
    tail_metrics,
)


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "BTC": rng.normal(0.001, 0.03, n),
        "ETH": rng.normal(0.0008, 0.04, n),
        "GOLD": rng.normal(0.0003, 0.01, n),
    })


# ---------------- Kelly ----------------

def test_kelly_fraction_positive_edge():
    f = kelly_fraction(win_rate=0.6, payoff_ratio=1.5, fraction=1.0, cap=1.0)
    assert f == pytest.approx(0.6 - 0.4 / 1.5, abs=1e-9)


def test_kelly_fraction_no_edge_is_zero():
    assert kelly_fraction(win_rate=0.3, payoff_ratio=1.0) == 0.0


def test_kelly_capped():
    assert kelly_fraction(win_rate=0.95, payoff_ratio=5.0, fraction=1.0, cap=0.25) == 0.25


def test_kelly_from_returns(returns_df):
    f = kelly_from_returns(returns_df["BTC"])
    assert 0.0 <= f <= 0.25


# ---------------- Optimizer ----------------

def test_weights_sum_to_one(returns_df):
    opt = PortfolioOptimizer(returns_df)
    for weights in (opt.max_sharpe(), opt.min_volatility(), opt.risk_parity(), opt.min_cvar()):
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (weights >= -1e-9).all()


def test_min_vol_prefers_low_vol_asset(returns_df):
    weights = PortfolioOptimizer(returns_df).min_volatility()
    assert weights["GOLD"] == weights.max()


def test_inverse_volatility(returns_df):
    weights = inverse_volatility_weights(returns_df)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["GOLD"] > weights["ETH"]


def test_optimizer_rejects_single_asset(returns_df):
    with pytest.raises(ValueError):
        PortfolioOptimizer(returns_df[["BTC"]])


# ---------------- Risk metrics ----------------

def test_var_ordering(returns_df):
    r = returns_df["BTC"]
    hv, pv, mv = historical_var(r), parametric_var(r), monte_carlo_var(r)
    assert all(v > 0 for v in (hv, pv, mv))
    assert cvar(r) >= hv  # expected shortfall is at least VaR


def test_beta_alpha(returns_df):
    beta, alpha = beta_alpha(returns_df["BTC"], returns_df["BTC"])
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_risk_of_ruin_bounds():
    assert risk_of_ruin(0.6, 1.5, 0.02) < 0.5
    assert risk_of_ruin(0.3, 1.0, 0.02) == 1.0
    assert risk_of_ruin(0.5, 1.0, 0.0) == 1.0


def test_tail_metrics(returns_df):
    m = tail_metrics(returns_df["BTC"])
    assert set(m) == {"skewness", "kurtosis"}


# ---------------- Pre-trade checks ----------------

def test_risk_engine_approves_small_order():
    engine = RiskEngine(RiskLimits(max_position_pct=0.25))
    result = engine.check_order(quantity=0.01, price=100.0, equity=10_000)
    assert result.approved
    assert result.adjusted_quantity == 0.01


def test_risk_engine_caps_oversized_order():
    engine = RiskEngine(RiskLimits(max_position_pct=0.25))
    result = engine.check_order(quantity=100, price=100.0, equity=10_000)
    assert result.approved
    assert result.adjusted_quantity == pytest.approx(25.0)  # 25% of 10k / price


def test_risk_engine_rejects_when_broke():
    engine = RiskEngine()
    result = engine.check_order(quantity=1, price=100.0, equity=0)
    assert not result.approved
    assert result.adjusted_quantity == 0.0


def test_risk_engine_rejects_high_var():
    engine = RiskEngine(RiskLimits(max_var_95=0.01))
    volatile = pd.Series(np.random.default_rng(1).normal(0, 0.1, 100))
    result = engine.check_order(quantity=0.01, price=100.0, equity=10_000, recent_returns=volatile)
    assert not result.approved

"""Tests for the RL trading environment (Fase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rl.env.trading_env import BUY, HOLD, SELL, TradingEnv


@pytest.fixture
def feature_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 200
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "close": close,
        "rsi": rng.uniform(20, 80, n),
        "momentum": rng.normal(0, 1, n),
    })


def test_gymnasium_check_env(feature_df):
    from gymnasium.utils.env_checker import check_env
    check_env(TradingEnv(feature_df, window_size=10), skip_render_check=True)


def test_observation_shape(feature_df):
    env = TradingEnv(feature_df, window_size=10)
    obs, _ = env.reset()
    assert obs.shape == (10 * 2 + 2,)  # 2 features × window + position + pnl
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(env.observation_space.low))
    assert np.all(np.isfinite(env.observation_space.high))


def test_episode_runs_to_end(feature_df):
    env = TradingEnv(feature_df, window_size=10)
    env.reset()
    done, steps = False, 0
    while not done:
        _, reward, terminated, truncated, info = env.step(HOLD)
        done = terminated or truncated
        steps += 1
        assert np.isfinite(reward)
    assert steps == len(feature_df) - 10 - 1
    assert info["equity"] == pytest.approx(env.initial_capital)  # holding cash = no pnl


def test_buy_changes_position_and_costs(feature_df):
    env = TradingEnv(feature_df, window_size=10, commission_pct=0.01)
    env.reset()
    _, _, _, _, info = env.step(BUY)
    assert info["position"] == 1.0
    env.step(SELL)
    assert env.position == 0.0
    # Two full turnovers at 1% commission must have cost something
    assert env.equity != env.initial_capital


def test_requires_close_column():
    with pytest.raises(ValueError):
        TradingEnv(pd.DataFrame({"rsi": [1, 2, 3]}))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_size": 0}, "window_size"),
        ({"initial_capital": 0}, "initial_capital"),
        ({"commission_pct": 1}, "commission_pct"),
        ({"position_step": 0}, "position_step"),
        ({"dd_penalty": -1}, "dd_penalty"),
    ],
)
def test_rejects_invalid_parameters(feature_df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        TradingEnv(feature_df, **kwargs)


def test_rejects_short_or_invalid_price_data(feature_df):
    with pytest.raises(ValueError, match="window_size"):
        TradingEnv(feature_df.iloc[:10], window_size=10)

    invalid = feature_df.copy()
    invalid.loc[0, "close"] = np.nan
    with pytest.raises(ValueError, match="close prices"):
        TradingEnv(invalid, window_size=10)

"""
Trading Environment (Fase 3)
==============================
Custom Gymnasium environment for RL-based trading.

Observation: flattened window of features + [position, unrealized_pnl].
Actions: 0=HOLD, 1=BUY (full long), 2=SELL (flat), 3=INCREASE, 4=DECREASE.
Reward: step PnL with drawdown penalty minus transaction costs
(see src/rl/rewards.py).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.rl.rewards import pnl_drawdown_reward

HOLD, BUY, SELL, INCREASE, DECREASE = 0, 1, 2, 3, 4


class TradingEnv(gym.Env):
    """
    Single-asset trading environment over a feature DataFrame.

    Args:
        df: DataFrame containing a "close" column plus feature columns.
        feature_columns: Which columns form the observation (default: all
            numeric columns except close).
        window_size: Number of past bars in each observation.
        commission_pct: Cost per position change (fraction of traded value).
        position_step: Position change for INCREASE/DECREASE actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        window_size: int = 20,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,
        position_step: float = 0.25,
        dd_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        if "close" not in df.columns:
            raise ValueError("df must contain a 'close' column")
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if len(df) <= window_size + 1:
            raise ValueError("df must contain at least window_size + 2 rows")
        if not np.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if not np.isfinite(commission_pct) or not 0 <= commission_pct < 1:
            raise ValueError("commission_pct must be finite and in [0, 1)")
        if not np.isfinite(position_step) or not 0 < position_step <= 1:
            raise ValueError("position_step must be finite and in (0, 1]")
        if not np.isfinite(dd_penalty) or dd_penalty < 0:
            raise ValueError("dd_penalty must be finite and non-negative")
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns or [
            c for c in df.select_dtypes(include=np.number).columns if c != "close"
        ]
        if not self.feature_columns:
            raise ValueError("No numeric feature columns found")
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.position_step = position_step
        self.dd_penalty = dd_penalty

        # Pre-normalize features (z-score) for training stability
        feats = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        mean, std = np.nanmean(feats, axis=0), np.nanstd(feats, axis=0)
        std[std == 0] = 1.0
        self._features = np.nan_to_num(
            (feats - mean) / std,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._closes = self.df["close"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(self._closes)) or np.any(self._closes <= 0):
            raise ValueError("close prices must be finite and positive")

        obs_dim = window_size * len(self.feature_columns) + 2  # + position, unrealized pnl
        float_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            -float_limit,
            float_limit,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)
        self._reset_state()

    def _reset_state(self) -> None:
        self._step_idx = self.window_size
        self.position = 0.0          # fraction of capital allocated, in [0, 1]
        self.entry_price = 0.0
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

    def _observation(self) -> np.ndarray:
        window = self._features[self._step_idx - self.window_size:self._step_idx]
        price = self._closes[self._step_idx]
        unrealized = (price / self.entry_price - 1.0) if self.position > 0 and self.entry_price else 0.0
        return np.concatenate([
            window.ravel(),
            np.array([self.position, unrealized], dtype=np.float32),
        ]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        price = self._closes[self._step_idx]
        target = {
            HOLD: self.position,
            BUY: 1.0,
            SELL: 0.0,
            INCREASE: min(self.position + self.position_step, 1.0),
            DECREASE: max(self.position - self.position_step, 0.0),
        }[int(action)]

        # Transaction cost on the change in exposure
        turnover = abs(target - self.position)
        cost = turnover * self.equity * self.commission_pct
        if target > 0 and self.position == 0:
            self.entry_price = price
        self.position = target

        # Advance one bar; PnL from close-to-close return on allocated fraction
        self._step_idx += 1
        next_price = self._closes[self._step_idx]
        ret = (next_price / price) - 1.0
        pnl = self.position * self.equity * ret - cost
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity

        reward = pnl_drawdown_reward(
            pnl=pnl / self.initial_capital,
            drawdown=drawdown,
            transaction_cost=cost / self.initial_capital,
            dd_penalty=self.dd_penalty,
        )

        terminated = self.equity <= 0.5 * self.initial_capital  # ruin condition
        truncated = self._step_idx >= len(self._closes) - 1
        info = {"equity": self.equity, "position": self.position, "drawdown": drawdown}
        return self._observation(), float(reward), terminated, truncated, info

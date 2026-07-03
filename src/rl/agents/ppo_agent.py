"""
PPO Agent (Fase 3) — Proximal Policy Optimization

Thin wrapper around Stable-Baselines3 PPO for the TradingEnv.
SB3 is lazy-imported so the rest of the system works without it:
    pip install stable-baselines3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.rl.env.trading_env import TradingEnv

logger = get_logger("rl.ppo_agent")


def _require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        return PPO, DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise ImportError(
            "PPOTradingAgent requires stable-baselines3: pip install stable-baselines3"
        ) from exc


class PPOTradingAgent:
    """Train and run a PPO policy on a TradingEnv."""

    def __init__(self, env_kwargs: dict[str, Any] | None = None, **ppo_kwargs: Any) -> None:
        self.env_kwargs = env_kwargs or {}
        self.ppo_kwargs = {"policy": "MlpPolicy", "verbose": 0, **ppo_kwargs}
        self.model = None
        self.vec_env = None

    def train(self, df: pd.DataFrame, total_timesteps: int = 100_000) -> "PPOTradingAgent":
        PPO, DummyVecEnv, VecNormalize = _require_sb3()
        self.vec_env = VecNormalize(
            DummyVecEnv([lambda: TradingEnv(df, **self.env_kwargs)]),
            norm_obs=True, norm_reward=True,
        )
        self.model = PPO(env=self.vec_env, **self.ppo_kwargs)
        self.model.learn(total_timesteps=total_timesteps)
        logger.info("ppo_training_complete", timesteps=total_timesteps)
        return self

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Run the trained policy over df and report final equity/drawdown."""
        if self.model is None:
            raise RuntimeError("Train or load a model first")
        env = TradingEnv(df, **self.env_kwargs)
        obs, _ = env.reset()
        max_dd = 0.0
        done = False
        while not done:
            if self.vec_env is not None:
                obs = self.vec_env.normalize_obs(obs)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            max_dd = min(max_dd, info["drawdown"])
            done = terminated or truncated
        return {
            "final_equity": info["equity"],
            "total_return": info["equity"] / env.initial_capital - 1.0,
            "max_drawdown": max_dd,
        }

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        if self.vec_env is not None:
            self.vec_env.save(str(path.with_suffix(".vecnorm.pkl")))

    def load(self, path: str | Path) -> "PPOTradingAgent":
        PPO, _, VecNormalize = _require_sb3()
        path = Path(path)
        self.model = PPO.load(str(path))
        vecnorm = path.with_suffix(".vecnorm.pkl")
        if vecnorm.exists():
            import pickle
            with open(vecnorm, "rb") as f:
                self.vec_env = pickle.load(f)
            self.vec_env.training = False
        return self

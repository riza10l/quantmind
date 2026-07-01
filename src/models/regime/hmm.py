"""
Hidden Markov Model Regime Detection
=======================================
Uses Gaussian HMM to identify latent market regimes (e.g., bullish,
bearish, high-vol, low-vol). Trained on return + volatility features,
then used to label each bar with a regime state.

Market regime detection is critical because:
- Different regimes require different strategies
- Models trained on mixed regimes have lower accuracy
- Position sizing should vary by regime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core.logger import get_logger

logger = get_logger("models.regime.hmm")


@dataclass
class RegimeInfo:
    """Information about a detected regime."""
    regime_id: int
    label: str  # "bull", "bear", "high_vol", "low_vol"
    mean_return: float
    mean_volatility: float
    frequency: float  # Fraction of time in this regime
    transition_probs: dict[int, float] = field(default_factory=dict)


class HMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for market regime detection.

    Learns latent states from observable return/volatility features.
    Automatically labels regimes based on their return and volatility
    characteristics.

    Args:
        n_regimes: Number of hidden states (regimes) to detect.
        features: List of features to use for regime detection.
                  Default: returns + rolling volatility.
        covariance_type: HMM covariance type ('full', 'diag', 'spherical').
        n_iter: Maximum EM iterations.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        features: list[str] | None = None,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.features = features or ["returns", "volatility"]
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self._model = None
        self._scaler = StandardScaler()
        self._regime_info: list[RegimeInfo] = []
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare observation features for the HMM."""
        feat_df = pd.DataFrame(index=df.index)

        if "returns" in self.features:
            feat_df["returns"] = df["close"].pct_change()

        if "volatility" in self.features:
            feat_df["volatility"] = df["close"].pct_change().rolling(20).std()

        if "volume_change" in self.features:
            feat_df["volume_change"] = df["volume"].pct_change()

        if "log_returns" in self.features:
            feat_df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        if "momentum" in self.features:
            feat_df["momentum"] = df["close"].pct_change(10)

        # Also pass through any raw feature names from df
        for feat in self.features:
            if feat in df.columns and feat not in feat_df.columns:
                feat_df[feat] = df[feat]

        feat_df = feat_df.dropna()
        return feat_df

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the HMM on OHLCV data.

        Args:
            df: DataFrame with at least 'close' and 'volume' columns.
        """
        from hmmlearn.hmm import GaussianHMM

        feat_df = self._prepare_features(df)
        X = self._scaler.fit_transform(feat_df.values)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(X)
        self.is_fitted = True

        # Auto-label regimes
        self._auto_label_regimes(feat_df)

        logger.info(
            "hmm_fitted",
            n_regimes=self.n_regimes,
            n_samples=len(X),
            converged=self._model.monitor_.converged,
            n_iter=self._model.monitor_.iter,
            log_likelihood=float(self._model.score(X)),
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for each bar.

        Returns:
            Series of regime IDs aligned with df index.
        """
        assert self.is_fitted, "Model not fitted"

        feat_df = self._prepare_features(df)
        X = self._scaler.transform(feat_df.values)
        states = self._model.predict(X)

        regime_series = pd.Series(states, index=feat_df.index, name="regime")
        return regime_series.reindex(df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get probability of each regime for each bar.

        Returns:
            DataFrame of shape (n_samples, n_regimes) with probabilities.
        """
        assert self.is_fitted, "Model not fitted"

        feat_df = self._prepare_features(df)
        X = self._scaler.transform(feat_df.values)
        probs = self._model.predict_proba(X)

        return pd.DataFrame(
            probs,
            index=feat_df.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

    def _auto_label_regimes(self, feat_df: pd.DataFrame) -> None:
        """
        Automatically label regimes based on mean return and volatility.

        The regime with highest mean return → "bull"
        The regime with lowest mean return → "bear"
        Others → labeled by volatility ("high_vol", "low_vol", "neutral")
        """
        X = self._scaler.transform(feat_df.values)
        states = self._model.predict(X)
        regimes = pd.Series(states, index=feat_df.index)

        self._regime_info = []

        # Compute stats per regime
        regime_stats = []
        for regime_id in range(self.n_regimes):
            mask = regimes == regime_id
            if mask.sum() == 0:
                continue

            mean_ret = feat_df.iloc[:, 0][mask].mean() if "returns" in self.features else 0
            mean_vol = feat_df.iloc[:, 1][mask].mean() if len(feat_df.columns) > 1 else 0
            freq = mask.mean()

            regime_stats.append({
                "regime_id": regime_id,
                "mean_return": mean_ret,
                "mean_volatility": mean_vol,
                "frequency": freq,
            })

        # Sort by mean return to assign labels
        regime_stats.sort(key=lambda x: x["mean_return"])

        labels = ["bear"] + ["neutral"] * (self.n_regimes - 2) + ["bull"]
        if self.n_regimes == 2:
            labels = ["bear", "bull"]

        # Build transition probability matrix
        trans_matrix = self._model.transmat_

        for i, stats in enumerate(regime_stats):
            regime_id = stats["regime_id"]
            trans_probs = {
                j: float(trans_matrix[regime_id, j])
                for j in range(self.n_regimes)
            }

            info = RegimeInfo(
                regime_id=regime_id,
                label=labels[i],
                mean_return=float(stats["mean_return"]),
                mean_volatility=float(stats["mean_volatility"]),
                frequency=float(stats["frequency"]),
                transition_probs=trans_probs,
            )
            self._regime_info.append(info)

        logger.info("regimes_labeled", regimes=[
            {"id": r.regime_id, "label": r.label, "freq": f"{r.frequency:.1%}"}
            for r in self._regime_info
        ])

    @property
    def regime_info(self) -> list[RegimeInfo]:
        return self._regime_info

    def get_regime_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of all regimes."""
        if not self._regime_info:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "regime_id": r.regime_id,
                "label": r.label,
                "mean_return": r.mean_return,
                "mean_volatility": r.mean_volatility,
                "frequency": r.frequency,
            }
            for r in self._regime_info
        ])

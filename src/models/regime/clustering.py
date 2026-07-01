"""
Clustering-Based Regime Detection
====================================
Uses HDBSCAN and Gaussian Mixture Models (GMM) to cluster market
conditions into distinct regimes based on multi-dimensional features.

Unlike HMM, clustering approaches:
- Don't assume a specific temporal structure
- Can detect regimes of varying density and shape
- Are more flexible with feature engineering

HDBSCAN is preferred over K-Means because:
- No need to pre-specify the number of clusters
- Can identify noise points (uncertain regime)
- Handles clusters of different densities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.core.logger import get_logger

logger = get_logger("models.regime.clustering")


@dataclass
class ClusterRegimeInfo:
    """Metadata about a detected cluster regime."""
    cluster_id: int
    label: str
    n_samples: int
    mean_return: float
    mean_volatility: float
    mean_momentum: float
    frequency: float


class HDBSCANRegimeDetector:
    """
    HDBSCAN-based market regime detector.

    Detects regimes without pre-specifying the number of clusters.
    Noise points (uncertain regime) are labeled -1.

    Args:
        min_cluster_size: Minimum samples for a cluster. Larger = fewer clusters.
        min_samples: Core point threshold. Larger = more conservative.
        features: Feature columns to use from the OHLCV DataFrame.
    """

    def __init__(
        self,
        min_cluster_size: int = 30,
        min_samples: int = 10,
        features: list[str] | None = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.features = features or ["returns", "volatility", "momentum"]
        self._model = None
        self._scaler = StandardScaler()
        self._regime_info: list[ClusterRegimeInfo] = []
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build clustering features from OHLCV data."""
        feat_df = pd.DataFrame(index=df.index)

        if "returns" in self.features:
            feat_df["returns"] = df["close"].pct_change()
        if "volatility" in self.features:
            feat_df["volatility"] = df["close"].pct_change().rolling(20).std()
        if "momentum" in self.features:
            feat_df["momentum"] = df["close"].pct_change(10)
        if "volume_ratio" in self.features:
            feat_df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        if "range_pct" in self.features:
            feat_df["range_pct"] = (df["high"] - df["low"]) / df["close"]

        for feat in self.features:
            if feat in df.columns and feat not in feat_df.columns:
                feat_df[feat] = df[feat]

        return feat_df.dropna()

    def fit(self, df: pd.DataFrame) -> "HDBSCANRegimeDetector":
        """Fit HDBSCAN on OHLCV data."""
        import hdbscan

        feat_df = self._prepare_features(df)
        X = self._scaler.fit_transform(feat_df.values)

        self._model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = self._model.fit_predict(X)

        self.is_fitted = True
        self._build_regime_info(feat_df, labels)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean()

        logger.info(
            "hdbscan_fitted",
            n_clusters=n_clusters,
            noise_pct=f"{noise_pct:.1%}",
            n_samples=len(X),
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels using approximate_predict."""
        assert self.is_fitted, "Model not fitted. Call fit() first."
        import hdbscan

        feat_df = self._prepare_features(df)
        X = self._scaler.transform(feat_df.values)

        labels, strengths = hdbscan.approximate_predict(self._model, X)
        return pd.Series(labels, index=feat_df.index, name="regime").reindex(df.index)

    def _build_regime_info(self, feat_df: pd.DataFrame, labels: np.ndarray) -> None:
        """Build regime info from clustering results."""
        self._regime_info = []
        regimes = pd.Series(labels, index=feat_df.index)

        for cluster_id in sorted(regimes.unique()):
            if cluster_id == -1:
                label = "noise"
            else:
                label = f"cluster_{cluster_id}"

            mask = regimes == cluster_id
            n = mask.sum()

            mean_ret = feat_df.iloc[:, 0][mask].mean() if feat_df.shape[1] > 0 else 0
            mean_vol = feat_df.iloc[:, 1][mask].mean() if feat_df.shape[1] > 1 else 0
            mean_mom = feat_df.iloc[:, 2][mask].mean() if feat_df.shape[1] > 2 else 0

            # Auto-label by return characteristics
            if cluster_id != -1:
                if mean_ret > 0.001:
                    label = "bull"
                elif mean_ret < -0.001:
                    label = "bear"
                elif mean_vol > feat_df.iloc[:, 1].quantile(0.7) if feat_df.shape[1] > 1 else False:
                    label = "high_volatility"
                else:
                    label = "range_bound"

            self._regime_info.append(ClusterRegimeInfo(
                cluster_id=cluster_id,
                label=label,
                n_samples=int(n),
                mean_return=float(mean_ret),
                mean_volatility=float(mean_vol),
                mean_momentum=float(mean_mom),
                frequency=float(n / len(regimes)),
            ))

    @property
    def regime_info(self) -> list[ClusterRegimeInfo]:
        return self._regime_info


class GMMRegimeDetector:
    """
    Gaussian Mixture Model regime detector.

    Advantages over HDBSCAN:
    - Provides soft assignments (probability of each regime)
    - Can use BIC/AIC to auto-select number of components
    - Better when regimes are roughly Gaussian-shaped

    Args:
        n_regimes: Number of Gaussian components (regimes).
        features: Feature columns to build from OHLCV.
        covariance_type: 'full', 'tied', 'diag', 'spherical'.
        auto_select: If True, use BIC to find optimal n_regimes.
        max_regimes: Max components to try if auto_select=True.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        features: list[str] | None = None,
        covariance_type: str = "full",
        auto_select: bool = False,
        max_regimes: int = 8,
    ) -> None:
        self.n_regimes = n_regimes
        self.features = features or ["returns", "volatility", "momentum"]
        self.covariance_type = covariance_type
        self.auto_select = auto_select
        self.max_regimes = max_regimes

        self._model: GaussianMixture | None = None
        self._scaler = StandardScaler()
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat_df = pd.DataFrame(index=df.index)
        if "returns" in self.features:
            feat_df["returns"] = df["close"].pct_change()
        if "volatility" in self.features:
            feat_df["volatility"] = df["close"].pct_change().rolling(20).std()
        if "momentum" in self.features:
            feat_df["momentum"] = df["close"].pct_change(10)
        for feat in self.features:
            if feat in df.columns and feat not in feat_df.columns:
                feat_df[feat] = df[feat]
        return feat_df.dropna()

    def fit(self, df: pd.DataFrame) -> "GMMRegimeDetector":
        """Fit GMM, optionally auto-selecting n_components via BIC."""
        feat_df = self._prepare_features(df)
        X = self._scaler.fit_transform(feat_df.values)

        if self.auto_select:
            best_bic = float("inf")
            best_n = self.n_regimes

            for n in range(2, self.max_regimes + 1):
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_init=3,
                    random_state=42,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_n = n

            self.n_regimes = best_n
            logger.info("gmm_auto_select", best_n=best_n, best_bic=best_bic)

        self._model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_init=5,
            random_state=42,
            max_iter=200,
        )
        self._model.fit(X)
        self.is_fitted = True

        logger.info(
            "gmm_fitted",
            n_regimes=self.n_regimes,
            n_samples=len(X),
            converged=self._model.converged_,
            bic=float(self._model.bic(X)),
            aic=float(self._model.aic(X)),
        )

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        assert self._model is not None
        feat_df = self._prepare_features(df)
        X = self._scaler.transform(feat_df.values)
        labels = self._model.predict(X)
        return pd.Series(labels, index=feat_df.index, name="regime").reindex(df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._model is not None
        feat_df = self._prepare_features(df)
        X = self._scaler.transform(feat_df.values)
        probs = self._model.predict_proba(X)
        return pd.DataFrame(
            probs, index=feat_df.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

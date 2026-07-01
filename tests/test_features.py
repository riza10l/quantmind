"""
Tests for Feature Engineering Module
======================================
Tests for feature registry, technical/statistical features, and feature store.
"""

import numpy as np
import pandas as pd
import pytest


class TestFeatureRegistry:
    """Tests for the feature registry."""

    def test_register_and_compute(self):
        from src.features.registry import FeatureRegistry

        registry = FeatureRegistry()

        def dummy_feature(df):
            return df["close"].rolling(5).mean()

        registry.register("test_sma5", dummy_feature, group="test", description="Test SMA 5")

        assert "test_sma5" in registry.list_features()
        assert registry.count >= 1

    def test_list_groups(self):
        from src.features.registry import feature_registry

        # Import to register features first
        import src.features.technical  # noqa
        import src.features.statistical  # noqa

        groups = feature_registry.list_groups()
        assert len(groups) > 0

    def test_compute_all_features(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry

        # Import to register features
        import src.features.technical  # noqa
        import src.features.statistical  # noqa

        assert feature_registry.count > 50, (
            f"Expected 50+ features, got {feature_registry.count}"
        )

        # Compute a subset (technical only for speed)
        result = feature_registry.compute_all(
            sample_ohlcv_indexed,
            groups=["technical"],
        )

        assert not result.empty
        assert result.shape[1] > 20  # At least 20 technical features
        assert len(result) == len(sample_ohlcv_indexed)


class TestTechnicalFeatures:
    """Tests for technical indicator features."""

    def test_sma_features(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.technical  # noqa

        sma_50 = feature_registry.compute("sma_50", sample_ohlcv_indexed)
        assert len(sma_50) == len(sample_ohlcv_indexed)
        # SMA should have NaN for first 49 values
        assert sma_50.iloc[:49].isna().all()
        assert not sma_50.iloc[49:].isna().all()

    def test_rsi_feature(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.technical  # noqa

        rsi = feature_registry.compute("rsi_14", sample_ohlcv_indexed)
        # RSI should be between 0 and 100
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.technical  # noqa

        macd = feature_registry.compute("macd", sample_ohlcv_indexed)
        assert len(macd) == len(sample_ohlcv_indexed)

    def test_bollinger_bands(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.technical  # noqa

        upper = feature_registry.compute("bb_upper", sample_ohlcv_indexed)
        lower = feature_registry.compute("bb_lower", sample_ohlcv_indexed)

        valid_idx = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper[valid_idx] >= lower[valid_idx]).all()


class TestStatisticalFeatures:
    """Tests for statistical features."""

    def test_return_features(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.statistical  # noqa

        ret_1 = feature_registry.compute("return_1", sample_ohlcv_indexed)
        assert len(ret_1) == len(sample_ohlcv_indexed)
        # First value should be NaN
        assert pd.isna(ret_1.iloc[0])

    def test_volatility_features(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.statistical  # noqa

        vol = feature_registry.compute("realized_vol_20", sample_ohlcv_indexed)
        valid = vol.dropna()
        assert (valid >= 0).all()  # Volatility should be non-negative

    def test_zscore_features(self, sample_ohlcv_indexed):
        from src.features.registry import feature_registry
        import src.features.statistical  # noqa

        z = feature_registry.compute("zscore_20", sample_ohlcv_indexed)
        valid = z.dropna()
        # Z-scores should be roughly centered around 0
        assert abs(valid.mean()) < 2.0


class TestFeatureStore:
    """Tests for feature store operations."""

    def test_compute_and_store(self, test_config, populated_db):
        from src.features.store import FeatureStore

        # Disable drop_na for testing (some features will be NaN without pandas-ta)
        test_config.features.drop_na = False

        store = FeatureStore(test_config, populated_db)
        result = store.compute_and_store(
            "BTC/USDT", "1d",
            groups=["technical"],
        )

        assert not result.empty
        assert result.shape[1] > 10

    def test_get_features_with_target(self, test_config, populated_db):
        from src.features.store import FeatureStore

        store = FeatureStore(test_config, populated_db)

        # First compute features
        store.compute_and_store("BTC/USDT", "1d", groups=["technical"])

        # Then get with target
        X, y = store.get_features_with_target(
            "BTC/USDT", "1d",
            target_horizon=1,
            target_type="direction",
        )

        if not X.empty:
            assert len(X) == len(y)
            assert set(y.unique()).issubset({0, 1})

"""
Tests for Data Module
======================
Tests for validators, cleaning, and pipeline logic.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.data.validators import OHLCVValidator, ValidationResult, clean_ohlcv


class TestOHLCVValidator:
    """Tests for OHLCV data validation."""

    def test_valid_data(self, sample_ohlcv):
        validator = OHLCVValidator()
        result = validator.validate(sample_ohlcv, timeframe_minutes=1440)
        assert result.is_valid
        assert result.score > 0.7

    def test_empty_dataframe(self):
        validator = OHLCVValidator()
        result = validator.validate(pd.DataFrame())
        assert not result.is_valid

    def test_missing_columns(self):
        df = pd.DataFrame({"timestamp": [datetime.now()], "close": [100.0]})
        validator = OHLCVValidator()
        result = validator.validate(df)
        assert not result.is_valid
        assert any("Missing columns" in e for e in result.errors)

    def test_null_detection(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[0:5, "close"] = np.nan

        validator = OHLCVValidator()
        result = validator.validate(df)
        assert len(result.warnings) > 0

    def test_negative_price_detection(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[0, "close"] = -100.0

        validator = OHLCVValidator()
        result = validator.validate(df)
        assert not result.is_valid

    def test_duplicate_detection(self, sample_ohlcv):
        df = pd.concat([sample_ohlcv, sample_ohlcv.head(5)])

        validator = OHLCVValidator()
        result = validator.validate(df)
        assert any("duplicate" in w.lower() for w in result.warnings)


class TestCleanOHLCV:
    """Tests for OHLCV data cleaning."""

    def test_removes_duplicates(self, sample_ohlcv):
        df = pd.concat([sample_ohlcv, sample_ohlcv.head(5)])
        cleaned = clean_ohlcv(df)
        assert len(cleaned) == len(sample_ohlcv)

    def test_sorts_by_timestamp(self, sample_ohlcv):
        df = sample_ohlcv.sample(frac=1)  # Shuffle
        cleaned = clean_ohlcv(df)
        assert cleaned["timestamp"].is_monotonic_increasing

    def test_fixes_negative_volume(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[0, "volume"] = -100
        cleaned = clean_ohlcv(df)
        assert (cleaned["volume"] >= 0).all()

    def test_fixes_ohlc_violations(self):
        df = pd.DataFrame({
            "timestamp": [datetime(2023, 1, 1)],
            "open": [100.0],
            "high": [90.0],   # Violation: high < open
            "low": [80.0],
            "close": [95.0],
            "volume": [1000.0],
        })
        cleaned = clean_ohlcv(df)
        assert cleaned.iloc[0]["high"] >= cleaned.iloc[0]["open"]
        assert cleaned.iloc[0]["low"] <= cleaned.iloc[0]["close"]

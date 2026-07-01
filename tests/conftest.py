"""
QuantMind Test Configuration & Fixtures
========================================
Shared pytest fixtures used across all test modules.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Generate realistic-ish price data
    price = 30000.0
    prices = [price]
    for _ in range(n - 1):
        ret = np.random.normal(0.0005, 0.02)
        price *= (1 + ret)
        prices.append(price)

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(15, 1, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_ohlcv_indexed(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Sample OHLCV data with timestamp as index."""
    return sample_ohlcv.set_index("timestamp")


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Provide a temporary database path for testing."""
    return f"sqlite:///{tmp_path / 'test_quantmind.db'}"


@pytest.fixture
def test_db(test_db_path: str):
    """Provide an initialized test database."""
    from src.core.database import DatabaseManager

    db = DatabaseManager(test_db_path)
    db.initialize()
    return db


@pytest.fixture
def populated_db(test_db, sample_ohlcv: pd.DataFrame):
    """Provide a test database populated with sample OHLCV data."""
    test_db.insert_ohlcv_batch(
        df=sample_ohlcv,
        symbol="BTC/USDT",
        timeframe="1d",
        source="test",
    )
    return test_db


@pytest.fixture
def test_config(test_db_path: str):
    """Provide a test configuration."""
    from src.core.config import AppConfig, DatabaseConfig

    return AppConfig(
        database=DatabaseConfig(url=test_db_path),
    )

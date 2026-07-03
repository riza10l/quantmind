"""
Tests for Core Module
======================
Tests for types, configuration, database, and event bus.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.core.config import AppConfig
from src.core.events import Event, EventBus
from src.core.types import OHLCV, FeatureDefinition, FeatureSet, Position, Side, Timeframe


class TestTimeframe:
    """Tests for the Timeframe enum."""

    def test_timeframe_values(self):
        assert Timeframe.D1.value == "1d"
        assert Timeframe.H1.value == "1h"
        assert Timeframe.M1.value == "1m"

    def test_timeframe_minutes(self):
        assert Timeframe.D1.minutes == 1440
        assert Timeframe.H1.minutes == 60
        assert Timeframe.M5.minutes == 5

    def test_timeframe_from_string(self):
        tf = Timeframe("1d")
        assert tf == Timeframe.D1


class TestOHLCV:
    """Tests for the OHLCV data structure."""

    def test_create_ohlcv(self):
        bar = OHLCV(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC/USDT",
            open=30000.0,
            high=31000.0,
            low=29000.0,
            close=30500.0,
            volume=1000.0,
        )
        assert bar.symbol == "BTC/USDT"
        assert bar.close == 30500.0

    def test_ohlcv_to_dict(self):
        bar = OHLCV(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC/USDT",
            open=30000.0,
            high=31000.0,
            low=29000.0,
            close=30500.0,
            volume=1000.0,
        )
        d = bar.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["source"] == "binance"

    def test_ohlcv_immutable(self):
        bar = OHLCV(
            timestamp=datetime(2023, 1, 1),
            symbol="BTC/USDT",
            open=30000.0,
            high=31000.0,
            low=29000.0,
            close=30500.0,
            volume=1000.0,
        )
        with pytest.raises(AttributeError):
            bar.close = 99999.0  # type: ignore


class TestPosition:
    """Tests for the Position data structure."""

    def test_position_return_long(self):
        pos = Position(
            symbol="BTC/USDT",
            side=Side.BUY,
            quantity=1.0,
            entry_price=30000.0,
            current_price=33000.0,
        )
        assert pos.return_pct == pytest.approx(0.1, abs=0.001)
        assert pos.market_value == 33000.0
    def test_position_return_short(self):
        pos = Position(
            symbol="BTC/USDT",
            side=Side.SELL,
            quantity=1.0,
            entry_price=30000.0,
            current_price=27000.0,
        )
        assert pos.return_pct == pytest.approx(0.1, abs=0.001)


def test_feature_definition_has_one_consistent_default_shape():
    definition = FeatureDefinition(name="momentum")
    feature_set = FeatureSet(
        name="default",
        version="v1",
        symbol="BTC/USDT",
        timeframe=Timeframe.D1,
        features=pd.DataFrame({"momentum": [1.0]}),
        definitions=[definition],
    )

    assert definition.group == "custom"
    assert definition.dependencies == ["close"]
    assert feature_set.definitions == [definition]
    assert feature_set.created_at.tzinfo is not None


class TestAppConfig:
    """Tests for configuration loading."""

    def test_default_config(self):
        config = AppConfig()
        assert config.backtest.initial_capital == 100_000.0
        assert config.risk.max_daily_drawdown_pct == 0.03
        assert len(config.data.sources) > 0

    def test_config_overrides(self):
        from src.core.config import BacktestConfig
        config = AppConfig(
            backtest=BacktestConfig(initial_capital=50_000.0)
        )
        assert config.backtest.initial_capital == 50_000.0


class TestEventBus:
    """Tests for the event bus."""

    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []

        @bus.on("test_event")
        def handler(event: Event):
            received.append(event)

        bus.emit("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0].data["key"] == "value"
        assert received[0].event_type == "test_event"

    def test_multiple_handlers(self):
        bus = EventBus()
        count = [0]

        @bus.on("multi")
        def handler1(event: Event):
            count[0] += 1

        @bus.on("multi")
        def handler2(event: Event):
            count[0] += 10

        bus.emit("multi")

        assert count[0] == 11

    def test_event_history(self):
        bus = EventBus()
        bus.emit("a", {"n": 1})
        bus.emit("b", {"n": 2})
        bus.emit("a", {"n": 3})

        history = bus.get_history("a")
        assert len(history) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("test", handler)
        bus.emit("test")
        assert len(received) == 1

        bus.unsubscribe("test", handler)
        bus.emit("test")
        assert len(received) == 1  # No new events


class TestDatabase:
    """Tests for the database manager."""

    def test_initialize(self, test_db):
        # Should not raise
        assert test_db is not None

    def test_insert_and_query_ohlcv(self, test_db, sample_ohlcv):
        inserted = test_db.insert_ohlcv_batch(
            df=sample_ohlcv,
            symbol="BTC/USDT",
            timeframe="1d",
            source="test",
        )
        assert inserted > 0

        df = test_db.query_ohlcv("BTC/USDT", "1d")
        assert not df.empty
        assert len(df) == len(sample_ohlcv)
        assert "close" in df.columns

    def test_get_latest_timestamp(self, populated_db):
        ts = populated_db.get_latest_timestamp("BTC/USDT", "1d")
        assert ts is not None

    def test_get_row_count(self, populated_db):
        count = populated_db.get_row_count("BTC/USDT", "1d")
        assert count == 500

    def test_upsert_no_duplicates(self, populated_db, sample_ohlcv):
        # Insert same data again
        inserted = populated_db.insert_ohlcv_batch(
            df=sample_ohlcv,
            symbol="BTC/USDT",
            timeframe="1d",
            source="test",
        )
        assert inserted == 0  # No new rows

    def test_insert_counts_mixed_symbols(self, test_db):
        base = {
            "timestamp": datetime(2024, 1, 1),
            "timeframe": "1d",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1_000.0,
            "source": "test",
        }

        inserted = test_db.insert_ohlcv([
            {**base, "symbol": "BTC/USDT"},
            {**base, "symbol": "ETH/USDT"},
        ])

        assert inserted == 2
        assert test_db.get_row_count("BTC/USDT", "1d") == 1
        assert test_db.get_row_count("ETH/USDT", "1d") == 1

    def test_insert_rejects_malformed_record(self, test_db):
        with pytest.raises(ValueError, match="missing required fields"):
            test_db.insert_ohlcv([{"symbol": "BTC/USDT"}])

    def test_get_available_symbols(self, populated_db):
        symbols = populated_db.get_available_symbols("1d")
        assert "BTC/USDT" in symbols

    def test_query_with_date_range(self, populated_db):
        df = populated_db.query_ohlcv(
            "BTC/USDT", "1d",
            start="2023-06-01",
            end="2023-12-31",
        )
        assert not df.empty
        assert len(df) < 500

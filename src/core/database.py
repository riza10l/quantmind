"""
QuantMind Database Layer
========================
Abstract database interface supporting SQLite (dev) and PostgreSQL/TimescaleDB (prod).
Uses SQLAlchemy 2.0 for ORM and raw SQL support.

Usage:
    from src.core.database import DatabaseManager
    db = DatabaseManager("sqlite:///data/quantmind.db")
    db.initialize()

    # Insert OHLCV data
    db.insert_ohlcv(ohlcv_records)

    # Query data
    df = db.query_ohlcv("BTC/USDT", "1d", start="2023-01-01")
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.logger import get_logger

logger = get_logger("core.database")

metadata = MetaData()


def _utc_now_naive() -> datetime:
    """Return UTC without tzinfo for the existing timezone-naive DB schema."""
    return datetime.now(UTC).replace(tzinfo=None)


# ============================================================
# Table Definitions
# ============================================================

ohlcv_table = Table(
    "ohlcv",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("symbol", String(50), nullable=False),
    Column("timeframe", String(10), nullable=False),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float, nullable=False),
    Column("source", String(30), nullable=False, default="binance"),
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_ohlcv_symbol_tf_ts", "symbol", "timeframe", "timestamp", unique=True),
    Index("idx_ohlcv_timestamp", "timestamp"),
)

funding_rate_table = Table(
    "funding_rates",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("symbol", String(50), nullable=False),
    Column("funding_rate", Float, nullable=False),
    Column("source", String(30), nullable=False, default="binance"),
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_funding_symbol_ts", "symbol", "timestamp", unique=True),
)

open_interest_table = Table(
    "open_interest",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("symbol", String(50), nullable=False),
    Column("open_interest", Float, nullable=False),
    Column("source", String(30), nullable=False, default="binance"),
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_oi_symbol_ts", "symbol", "timestamp", unique=True),
)

sentiment_table = Table(
    "sentiment",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("indicator", String(50), nullable=False),
    Column("value", Float, nullable=False),
    Column("label", String(50), nullable=True),
    Column("source", String(30), nullable=False),
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_sentiment_indicator_ts", "indicator", "timestamp", unique=True),
)

features_table = Table(
    "feature_store",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("symbol", String(50), nullable=False),
    Column("timeframe", String(10), nullable=False),
    Column("feature_name", String(100), nullable=False),
    Column("value", Float, nullable=True),
    Column("version", String(20), nullable=False, default="v1"),
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_feat_sym_tf_ts_name", "symbol", "timeframe", "timestamp", "feature_name"),
)

experiments_table = Table(
    "experiments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", String(100), nullable=False, unique=True),
    Column("name", String(200), nullable=False),
    Column("model_type", String(50), nullable=False),
    Column("params", Text, nullable=True),  # JSON serialized
    Column("metrics", Text, nullable=True),  # JSON serialized
    Column("features_used", Text, nullable=True),  # JSON serialized
    Column("dataset_hash", String(64), nullable=True),
    Column("status", String(20), nullable=False, default="running"),
    Column("created_at", DateTime, default=_utc_now_naive),
    Column("completed_at", DateTime, nullable=True),
)

trades_log_table = Table(
    "trades_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trade_id", String(100), nullable=False, unique=True),
    Column("timestamp", DateTime, nullable=False),
    Column("symbol", String(50), nullable=False),
    Column("side", String(10), nullable=False),
    Column("order_type", String(20), nullable=False),
    Column("quantity", Float, nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("exit_price", Float, nullable=True),
    Column("pnl", Float, nullable=True),
    Column("commission", Float, nullable=True),
    Column("slippage", Float, nullable=True),
    Column("latency_ms", Float, nullable=True),
    Column("signal_confidence", Float, nullable=True),
    Column("explanation", Text, nullable=True),  # XAI explanation
    Column("mode", String(10), nullable=False, default="paper"),  # paper or live
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_trades_symbol_ts", "symbol", "timestamp"),
)

selected_features_table = Table(
    "selected_features",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("selection_run_id", String(100), nullable=False),
    Column("method", String(50), nullable=False),
    Column("feature_name", String(100), nullable=False),
    Column("importance_score", Float, nullable=False),
    Column("rank", Integer, nullable=False),
    Column("created_at", DateTime, default=_utc_now_naive),
    Index("idx_selected_run_method", "selection_run_id", "method"),
)


# ============================================================
# Database Manager
# ============================================================


class DatabaseManager:
    """
    Manages database connections and provides high-level data operations.
    Supports SQLite for local dev and PostgreSQL/TimescaleDB for production.
    """

    def __init__(self, url: str | None = None) -> None:
        if url is None:
            url = f"sqlite:///{Path('data') / 'quantmind.db'}"

        self._url = url
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            connect_args = {}
            if "sqlite" in self._url:
                connect_args["check_same_thread"] = False

            self._engine = create_engine(
                self._url,
                echo=False,
                connect_args=connect_args,
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def initialize(self) -> None:
        """Create all tables if they don't exist."""
        # Ensure data directory exists for SQLite
        if "sqlite" in self._url:
            db_path = self._url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        metadata.create_all(self.engine)
        logger.info("database_initialized", url=self._url)

    # ---- OHLCV Operations ----

    def insert_ohlcv(self, records: list[dict[str, Any]]) -> int:
        """
        Insert OHLCV records with upsert behavior (skip duplicates).

        Uses INSERT OR IGNORE for SQLite (relies on the UNIQUE constraint
        on symbol+timeframe+timestamp) for both correctness and speed.

        Args:
            records: List of dicts with keys: timestamp, symbol, timeframe,
                     open, high, low, close, volume, source.

        Returns:
            Number of records inserted.
        """
        if not records:
            return 0

        required = {
            "timestamp", "symbol", "timeframe", "open", "high", "low",
            "close", "volume", "source",
        }
        values: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            missing = required.difference(record)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"OHLCV record {index} is missing required fields: {names}")
            if not str(record["symbol"]).strip() or not str(record["timeframe"]).strip():
                raise ValueError(f"OHLCV record {index} requires symbol and timeframe")
            for field_name in ("open", "high", "low", "close", "volume"):
                if not isfinite(float(record[field_name])):
                    raise ValueError(f"OHLCV record {index} has non-finite {field_name}")

            values.append({**record, "created_at": record.get("created_at", _utc_now_naive())})

        dialect = self.engine.dialect.name
        if dialect == "sqlite":
            statement = sqlite_insert(ohlcv_table)
        elif dialect == "postgresql":
            statement = postgresql_insert(ohlcv_table)
        else:
            raise RuntimeError(f"Unsupported database dialect for OHLCV upsert: {dialect}")

        statement = statement.values(values).on_conflict_do_nothing(
            index_elements=["symbol", "timeframe", "timestamp"]
        )
        with self.session() as session:
            result = session.execute(statement)
            inserted = max(result.rowcount or 0, 0)

        logger.info(
            "ohlcv_inserted",
            count=inserted,
            total=len(records),
            symbol=records[0].get("symbol") if records else None,
        )
        return inserted

    def insert_ohlcv_batch(self, df: pd.DataFrame, symbol: str,
                           timeframe: str, source: str = "binance") -> int:
        """
        Bulk insert OHLCV data from a DataFrame.

        Args:
            df: DataFrame with columns: timestamp/date, open, high, low, close, volume.
            symbol: Trading pair symbol.
            timeframe: Bar timeframe.
            source: Data source name.

        Returns:
            Number of records inserted.
        """
        records = []
        for _, row in df.iterrows():
            ts = row.get("timestamp") or row.get("date") or row.name
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            # Convert Pandas Timestamp to Python datetime for SQLite compat
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()

            records.append({
                "timestamp": ts,
                "symbol": symbol,
                "timeframe": timeframe,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "source": source,
            })

        return self.insert_ohlcv(records)

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Query OHLCV data as a pandas DataFrame.

        Args:
            symbol: Trading pair symbol.
            timeframe: Bar timeframe.
            start: Start date (inclusive).
            end: End date (inclusive).
            limit: Max number of rows.

        Returns:
            DataFrame indexed by timestamp with OHLCV columns.
        """
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
        query += "WHERE symbol = :symbol AND timeframe = :timeframe "
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}

        if start:
            if isinstance(start, str):
                start = pd.to_datetime(start)
            # Convert Pandas Timestamp to str for SQLite compat
            if isinstance(start, pd.Timestamp):
                start = start.strftime("%Y-%m-%d %H:%M:%S")
            query += "AND timestamp >= :start "
            params["start"] = start

        if end:
            if isinstance(end, str):
                end = pd.to_datetime(end)
            # Convert Pandas Timestamp to str for SQLite compat
            if isinstance(end, pd.Timestamp):
                end = end.strftime("%Y-%m-%d %H:%M:%S")
            query += "AND timestamp <= :end "
            params["end"] = end

        query += "ORDER BY timestamp ASC "

        if limit:
            query += f"LIMIT {limit} "

        df = pd.read_sql(text(query), self.engine, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Get the most recent timestamp for a symbol/timeframe pair."""
        with self.session() as session:
            result = session.execute(
                text(
                    "SELECT MAX(timestamp) FROM ohlcv "
                    "WHERE symbol = :symbol AND timeframe = :timeframe"
                ),
                {"symbol": symbol, "timeframe": timeframe},
            ).fetchone()

            if result and result[0]:
                ts = result[0]
                if isinstance(ts, str):
                    return pd.to_datetime(ts).to_pydatetime()
                return ts
            return None

    def get_row_count(self, symbol: str, timeframe: str) -> int:
        """Get the number of rows for a symbol/timeframe pair."""
        with self.session() as session:
            result = session.execute(
                text(
                    "SELECT COUNT(*) FROM ohlcv "
                    "WHERE symbol = :symbol AND timeframe = :timeframe"
                ),
                {"symbol": symbol, "timeframe": timeframe},
            ).fetchone()
            return result[0] if result else 0

    # ---- Sentiment Operations ----

    def insert_sentiment(self, records: list[dict[str, Any]]) -> int:
        """Insert sentiment data with upsert behavior."""
        if not records:
            return 0

        inserted = 0
        with self.session() as session:
            for record in records:
                exists = session.execute(
                    text(
                        "SELECT 1 FROM sentiment WHERE indicator = :indicator "
                        "AND timestamp = :timestamp"
                    ),
                    {
                        "indicator": record["indicator"],
                        "timestamp": record["timestamp"],
                    },
                ).fetchone()

                if not exists:
                    session.execute(sentiment_table.insert().values(**record))
                    inserted += 1

        logger.info("sentiment_inserted", count=inserted, total=len(records))
        return inserted

    def query_sentiment(
        self,
        indicator: str = "fear_greed",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Query sentiment data as a DataFrame."""
        query = "SELECT timestamp, value, label FROM sentiment "
        query += "WHERE indicator = :indicator "
        params: dict[str, Any] = {"indicator": indicator}

        if start:
            if isinstance(start, str):
                start = pd.to_datetime(start)
            query += "AND timestamp >= :start "
            params["start"] = start

        if end:
            if isinstance(end, str):
                end = pd.to_datetime(end)
            query += "AND timestamp <= :end "
            params["end"] = end

        query += "ORDER BY timestamp ASC"

        df = pd.read_sql(text(query), self.engine, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        return df

    # ---- Feature Store Operations ----

    def insert_features_wide(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        version: str = "v1",
    ) -> int:
        """
        Insert a wide-format feature DataFrame into the feature store.

        Args:
            df: DataFrame with feature columns and timestamp index.
            symbol: Trading pair symbol.
            timeframe: Bar timeframe.
            version: Feature version tag.

        Returns:
            Number of feature records inserted.
        """
        records = []
        for timestamp, row in df.iterrows():
            for feature_name in df.columns:
                value = row[feature_name]
                if pd.notna(value):
                    records.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "feature_name": feature_name,
                        "value": float(value),
                        "version": version,
                    })

        if not records:
            return 0

        # Batch insert
        with self.session() as session:
            session.execute(features_table.insert(), records)

        logger.info(
            "features_inserted",
            count=len(records),
            symbol=symbol,
            features=len(df.columns),
        )
        return len(records)

    def query_features(
        self,
        symbol: str,
        timeframe: str = "1d",
        feature_names: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        version: str = "v1",
    ) -> pd.DataFrame:
        """
        Query features from the feature store and pivot to wide format.

        Returns:
            DataFrame with feature columns and timestamp index.
        """
        query = (
            "SELECT timestamp, feature_name, value FROM feature_store "
            "WHERE symbol = :symbol AND timeframe = :timeframe AND version = :version "
        )
        params: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
        }

        if start:
            query += "AND timestamp >= :start "
            params["start"] = pd.to_datetime(start) if isinstance(start, str) else start
        if end:
            query += "AND timestamp <= :end "
            params["end"] = pd.to_datetime(end) if isinstance(end, str) else end

        query += "ORDER BY timestamp ASC"

        df = pd.read_sql(text(query), self.engine, params=params)
        if df.empty:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        wide = df.pivot_table(
            index="timestamp",
            columns="feature_name",
            values="value",
            aggfunc="first",
        )

        if feature_names:
            available = [f for f in feature_names if f in wide.columns]
            wide = wide[available]

        return wide

    # ---- Trade Log Operations ----

    def log_trade(self, trade_data: dict[str, Any]) -> None:
        """Insert a trade log record."""
        with self.session() as session:
            session.execute(trades_log_table.insert().values(**trade_data))
        logger.info("trade_logged", trade_id=trade_data.get("trade_id"))

    # ---- Selected Features Operations ----

    def insert_selected_features(self, records: list[dict[str, Any]]) -> int:
        """Insert feature selection results."""
        if not records:
            return 0
        with self.session() as session:
            session.execute(selected_features_table.insert(), records)
        return len(records)

    def query_selected_features(
        self, run_id: str, method: str | None = None
    ) -> pd.DataFrame:
        """Query selected features for a given run."""
        query = "SELECT * FROM selected_features WHERE selection_run_id = :run_id "
        params: dict[str, Any] = {"run_id": run_id}

        if method:
            query += "AND method = :method "
            params["method"] = method

        query += "ORDER BY rank ASC"

        return pd.read_sql(text(query), self.engine, params=params)

    # ---- Utility ----

    def get_available_symbols(self, timeframe: str = "1d") -> list[str]:
        """Get all symbols that have data for a given timeframe."""
        with self.session() as session:
            result = session.execute(
                text(
                    "SELECT DISTINCT symbol FROM ohlcv "
                    "WHERE timeframe = :timeframe ORDER BY symbol"
                ),
                {"timeframe": timeframe},
            ).fetchall()
            return [row[0] for row in result]

    def get_data_summary(self) -> pd.DataFrame:
        """Get a summary of all data in the database."""
        query = """
            SELECT symbol, timeframe, source,
                   COUNT(*) as row_count,
                   MIN(timestamp) as first_date,
                   MAX(timestamp) as last_date
            FROM ohlcv
            GROUP BY symbol, timeframe, source
            ORDER BY symbol, timeframe
        """
        return pd.read_sql(text(query), self.engine)

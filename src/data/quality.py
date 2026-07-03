"""
Data Quality Monitor
=====================
Scans stored OHLCV data for quality issues: gaps, staleness, duplicates,
timezone problems, and price anomalies. Builds on OHLCVValidator (which
checks a single DataFrame) and adds DB-wide monitoring with per-pair reports.

Usage:
    monitor = DataQualityMonitor(db)
    reports = monitor.check_all()          # every symbol/timeframe pair
    report = monitor.check("BTC-USD", "1d")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from src.core.database import DatabaseManager
from src.core.logger import get_logger
from src.data.validators import OHLCVValidator

logger = get_logger("data.quality")

TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
}


@dataclass
class QualityReport:
    """Data quality report for one symbol/timeframe pair."""
    symbol: str
    timeframe: str
    rows: int
    score: float                  # 0.0 - 1.0 from validator checks
    is_stale: bool
    last_bar: datetime | None
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and not self.is_stale


class DataQualityMonitor:
    """Runs data-quality checks against everything stored in the database."""

    def __init__(self, db: DatabaseManager, stale_factor: float = 2.0) -> None:
        self.db = db
        # Data is stale if the last bar is older than stale_factor × timeframe.
        self.stale_factor = stale_factor
        self.validator = OHLCVValidator()

    def check(self, symbol: str, timeframe: str) -> QualityReport:
        df = self.db.query_ohlcv(symbol, timeframe)
        if df.empty:
            return QualityReport(symbol, timeframe, 0, 0.0, True, None,
                                 issues=["no data stored"])

        flat = df.reset_index()  # validator expects a 'timestamp' column
        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 1440)
        result = self.validator.validate(flat, timeframe_minutes=tf_minutes)

        issues = list(result.errors) + list(result.warnings)
        issues += self._check_timezone(flat)
        last_bar = flat["timestamp"].max().to_pydatetime()
        is_stale = self._is_stale(last_bar, tf_minutes)
        if is_stale:
            issues.append(f"stale: last bar {last_bar:%Y-%m-%d %H:%M} is older than "
                          f"{self.stale_factor:g}x the {timeframe} timeframe")

        report = QualityReport(
            symbol=symbol, timeframe=timeframe, rows=len(df),
            score=result.score, is_stale=is_stale, last_bar=last_bar,
            issues=issues,
        )
        logger.info("quality_checked", symbol=symbol, timeframe=timeframe,
                    ok=report.ok, score=round(report.score, 3), issues=len(issues))
        return report

    def check_all(self) -> list[QualityReport]:
        summary = self.db.get_data_summary()
        return [self.check(row["symbol"], row["timeframe"])
                for _, row in summary.iterrows()]

    def to_dataframe(self, reports: list[QualityReport]) -> pd.DataFrame:
        return pd.DataFrame([{
            "symbol": r.symbol, "timeframe": r.timeframe, "rows": r.rows,
            "score": round(r.score, 3), "stale": r.is_stale,
            "last_bar": r.last_bar, "issues": len(r.issues),
            "detail": "; ".join(r.issues[:3]),
        } for r in reports])

    # ------------------------------------------------------------------

    def _is_stale(self, last_bar: datetime, tf_minutes: int) -> bool:
        now = datetime.now(UTC).replace(tzinfo=None)  # DB timestamps are naive UTC
        age_minutes = (now - last_bar).total_seconds() / 60
        return age_minutes > tf_minutes * self.stale_factor

    def _check_timezone(self, flat: pd.DataFrame) -> list[str]:
        """Detect timezone problems: tz-aware mixed in, or future timestamps."""
        issues = []
        ts = flat["timestamp"]
        if getattr(ts.dtype, "tz", None) is not None:
            issues.append("timestamps are timezone-aware; store as naive UTC")
        now = datetime.now(UTC).replace(tzinfo=None)
        future = (pd.to_datetime(ts) > now + pd.Timedelta(hours=1)).sum()
        if future:
            issues.append(f"{future} timestamps are in the future (timezone offset suspected)")
        return issues

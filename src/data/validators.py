"""
Data Quality Validators
========================
Validates downloaded market data for completeness, consistency,
and integrity before storage. Catches common issues like:
- Missing timestamps (gaps in data)
- Null/NaN values
- Outliers (extreme price spikes)
- OHLC logic violations (high < low, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("data.validators")


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    is_valid: bool
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Quality score from 0.0 to 1.0."""
        total = self.checks_passed + self.checks_failed
        if total == 0:
            return 1.0
        return self.checks_passed / total


class OHLCVValidator:
    """
    Validates OHLCV data quality.

    Checks:
    1. Required columns present
    2. No null values in critical columns
    3. OHLC logic (high >= low, high >= open/close, etc.)
    4. Positive volumes
    5. No duplicate timestamps
    6. Chronological ordering
    7. Gap detection
    8. Outlier detection (z-score based)
    """

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(
        self,
        max_gap_ratio: float = 0.05,  # Max 5% missing bars
        zscore_threshold: float = 5.0,  # Z-score for outlier detection
        min_rows: int = 10,  # Minimum rows to be valid
    ) -> None:
        self._max_gap_ratio = max_gap_ratio
        self._zscore_threshold = zscore_threshold
        self._min_rows = min_rows

    def validate(self, df: pd.DataFrame, timeframe_minutes: int = 1440) -> ValidationResult:
        """
        Run all validation checks on OHLCV data.

        Args:
            df: DataFrame with OHLCV data.
            timeframe_minutes: Expected bar duration in minutes.

        Returns:
            ValidationResult with detailed check results.
        """
        result = ValidationResult(is_valid=True)

        if df.empty:
            result.is_valid = False
            result.errors.append("DataFrame is empty")
            return result

        # 1. Check required columns
        self._check_columns(df, result)
        if not result.is_valid:
            return result

        # 2. Check minimum rows
        self._check_min_rows(df, result)

        # 3. Check for nulls
        self._check_nulls(df, result)

        # 4. Check OHLC logic
        self._check_ohlc_logic(df, result)

        # 5. Check positive values
        self._check_positive_values(df, result)

        # 6. Check duplicates
        self._check_duplicates(df, result)

        # 7. Check chronological order
        self._check_ordering(df, result)

        # 8. Check gaps
        self._check_gaps(df, timeframe_minutes, result)

        # 9. Check outliers
        self._check_outliers(df, result)

        # Compute stats
        result.stats = self._compute_stats(df)

        # Set overall validity
        result.is_valid = len(result.errors) == 0

        logger.info(
            "validation_complete",
            valid=result.is_valid,
            score=f"{result.score:.2%}",
            passed=result.checks_passed,
            failed=result.checks_failed,
            warnings=len(result.warnings),
            errors=len(result.errors),
        )

        return result

    def _check_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that all required columns are present."""
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            result.errors.append(f"Missing columns: {missing}")
            result.checks_failed += 1
            result.is_valid = False
        else:
            result.checks_passed += 1

    def _check_min_rows(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check minimum number of rows."""
        if len(df) < self._min_rows:
            result.warnings.append(
                f"Only {len(df)} rows (minimum {self._min_rows})"
            )
            result.checks_failed += 1
        else:
            result.checks_passed += 1

    def _check_nulls(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for null values in critical columns."""
        null_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            result.warnings.append(
                f"Found {total_nulls} null values: {null_counts[null_counts > 0].to_dict()}"
            )
            result.checks_failed += 1
        else:
            result.checks_passed += 1

    def _check_ohlc_logic(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check OHLC logical constraints."""
        violations = 0

        # High should be >= Low
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl > 0:
            violations += bad_hl
            result.warnings.append(f"High < Low in {bad_hl} rows")

        # High should be >= Open and Close
        bad_ho = (df["high"] < df["open"]).sum()
        bad_hc = (df["high"] < df["close"]).sum()
        if bad_ho + bad_hc > 0:
            violations += bad_ho + bad_hc
            result.warnings.append(f"High < Open/Close in {bad_ho + bad_hc} rows")

        # Low should be <= Open and Close
        bad_lo = (df["low"] > df["open"]).sum()
        bad_lc = (df["low"] > df["close"]).sum()
        if bad_lo + bad_lc > 0:
            violations += bad_lo + bad_lc
            result.warnings.append(f"Low > Open/Close in {bad_lo + bad_lc} rows")

        if violations > 0:
            result.checks_failed += 1
        else:
            result.checks_passed += 1

    def _check_positive_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that prices and volumes are positive."""
        neg_prices = (
            (df["open"] <= 0) | (df["high"] <= 0) |
            (df["low"] <= 0) | (df["close"] <= 0)
        ).sum()

        neg_volumes = (df["volume"] < 0).sum()

        if neg_prices > 0:
            result.errors.append(f"Negative/zero prices in {neg_prices} rows")
            result.checks_failed += 1
        else:
            result.checks_passed += 1

        if neg_volumes > 0:
            result.warnings.append(f"Negative volume in {neg_volumes} rows")
            result.checks_failed += 1
        else:
            result.checks_passed += 1

    def _check_duplicates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate timestamps."""
        dup_count = df["timestamp"].duplicated().sum()
        if dup_count > 0:
            result.warnings.append(f"Found {dup_count} duplicate timestamps")
            result.checks_failed += 1
        else:
            result.checks_passed += 1

    def _check_ordering(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that timestamps are in chronological order."""
        is_sorted = df["timestamp"].is_monotonic_increasing
        if not is_sorted:
            result.warnings.append("Timestamps are not in chronological order")
            result.checks_failed += 1
        else:
            result.checks_passed += 1

    def _check_gaps(
        self, df: pd.DataFrame, timeframe_minutes: int, result: ValidationResult
    ) -> None:
        """Detect gaps in the time series."""
        if len(df) < 2:
            result.checks_passed += 1
            return

        timestamps = pd.to_datetime(df["timestamp"]).sort_values()
        expected_delta = pd.Timedelta(minutes=timeframe_minutes)

        diffs = timestamps.diff().dropna()
        gaps = diffs[diffs > expected_delta * 1.5]

        if len(gaps) > 0:
            gap_ratio = len(gaps) / len(df)
            if gap_ratio > self._max_gap_ratio:
                result.warnings.append(
                    f"Found {len(gaps)} gaps ({gap_ratio:.1%} of data). "
                    f"Largest gap: {gaps.max()}"
                )
                result.checks_failed += 1
            else:
                result.checks_passed += 1
                result.warnings.append(
                    f"Found {len(gaps)} minor gaps (within tolerance)"
                )
        else:
            result.checks_passed += 1

    def _check_outliers(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Detect price outliers using z-score."""
        returns = df["close"].pct_change().dropna()

        if len(returns) < 10:
            result.checks_passed += 1
            return

        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outliers = (z_scores > self._zscore_threshold).sum()

        if outliers > 0:
            result.warnings.append(
                f"Found {outliers} potential outliers "
                f"(z-score > {self._zscore_threshold})"
            )
            # Don't fail — just warn (crypto can be volatile)
        result.checks_passed += 1

    def _compute_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute summary statistics for the data."""
        return {
            "rows": len(df),
            "date_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
            "price_range": {
                "min": float(df["low"].min()),
                "max": float(df["high"].max()),
            },
            "avg_volume": float(df["volume"].mean()),
            "null_count": int(df.isnull().sum().sum()),
        }


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data by fixing common issues.

    - Remove duplicates (keep last)
    - Sort by timestamp
    - Forward-fill null prices
    - Set negative volumes to 0
    - Fix OHLC violations

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill nulls in price columns
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Set negative volumes to 0
    if "volume" in df.columns:
        df.loc[df["volume"] < 0, "volume"] = 0
        df["volume"] = df["volume"].fillna(0)

    # Fix OHLC violations
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    # Drop rows where all prices are still null
    df = df.dropna(subset=["close"])

    logger.info("ohlcv_cleaned", rows=len(df))
    return df

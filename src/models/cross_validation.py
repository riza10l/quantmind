"""
Purged Walk-Forward Cross-Validation
======================================
Standard K-fold CV is invalid for time series because of data leakage.
This implements purged and embargoed CV splits specifically designed
for financial ML (following Marcos López de Prado's methodology).

The key insight: training data that overlaps with test labels will
introduce look-ahead bias. We "purge" training samples that are too
close to the test period and add an "embargo" gap.
"""

from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Purged K-Fold CV for financial time series.

    - Splits data chronologically (no shuffling)
    - Purges training samples that overlap with the test set boundary
    - Adds an embargo period between train and test to prevent leakage

    Args:
        n_splits: Number of folds.
        purge_pct: Fraction of data to purge at train/test boundary.
        embargo_pct: Fraction of data for embargo gap after test period.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame | np.ndarray, y=None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test indices.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        purge_size = max(1, int(n_samples * self.purge_pct))
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            test_idx = indices[test_start:test_end]

            # Build train indices: everything except test + purge + embargo
            purge_start = max(0, test_start - purge_size)
            embargo_end = min(n_samples, test_end + embargo_size)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class WalkForwardSplit:
    """
    Walk-forward (expanding or sliding window) cross-validation.

    More realistic: trains on past, tests on future.
    Commonly used by quant funds.

    Args:
        n_splits: Number of walk-forward windows.
        train_pct: Fraction of data for the training window.
        expanding: If True, training window grows. If False, slides.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.6,
        expanding: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.expanding = expanding

    def split(
        self, X: pd.DataFrame | np.ndarray, y=None, groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate window sizes
        test_size = int(n_samples * (1 - self.train_pct) / self.n_splits)
        initial_train_size = int(n_samples * self.train_pct)

        for i in range(self.n_splits):
            test_start = initial_train_size + i * test_size
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples:
                break

            if self.expanding:
                train_idx = indices[:test_start]
            else:
                window_size = initial_train_size
                train_start = max(0, test_start - window_size)
                train_idx = indices[train_start:test_start]

            test_idx = indices[test_start:test_end]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

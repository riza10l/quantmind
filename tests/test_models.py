"""
Tests for Phase 2: ML Models, Regime Detection, GA Optimizer
==============================================================
Tests the gradient boosting classifiers, cross-validation,
regime detection, explainability, and genetic algorithm.
"""

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Test Data Fixtures
# ============================================================


@pytest.fixture
def ml_data():
    """Generate synthetic classification data mimicking market direction."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create target correlated with a few features
    signal = 0.3 * X["feature_0"] + 0.2 * X["feature_1"] - 0.1 * X["feature_2"]
    noise = np.random.randn(n_samples) * 0.5
    y = pd.Series((signal + noise > 0).astype(int), name="target")

    return X, y


@pytest.fixture
def price_data():
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    # Random walk price
    returns = np.random.normal(0.0005, 0.02, n)
    close = 30000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + abs(np.random.randn(n) * 0.015)),
        "low": close * (1 - abs(np.random.randn(n) * 0.015)),
        "close": close,
        "volume": np.random.lognormal(14, 1, n),
    }, index=dates)

    return df


# ============================================================
# Cross Validation Tests
# ============================================================


class TestCrossValidation:
    """Tests for purged K-fold and walk-forward CV."""

    def test_purged_kfold_splits(self, ml_data):
        from src.models.cross_validation import PurgedKFold

        X, y = ml_data
        cv = PurgedKFold(n_splits=5)

        splits = list(cv.split(X))
        assert len(splits) == 5

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purged_kfold_no_leakage(self, ml_data):
        from src.models.cross_validation import PurgedKFold

        X, y = ml_data
        cv = PurgedKFold(n_splits=5, purge_pct=0.02, embargo_pct=0.02)

        for train_idx, test_idx in cv.split(X):
            # Train indices should not be adjacent to test indices
            # (there should be a gap due to purge + embargo)
            if len(train_idx) > 0 and len(test_idx) > 0:
                train_max = train_idx[train_idx < test_idx.min()].max() if (train_idx < test_idx.min()).any() else -1
                if train_max >= 0:
                    gap = test_idx.min() - train_max
                    assert gap >= 2  # At least purge gap

    def test_walk_forward_split(self, ml_data):
        from src.models.cross_validation import WalkForwardSplit

        X, y = ml_data
        wf = WalkForwardSplit(n_splits=5, train_pct=0.6)

        splits = list(wf.split(X))
        assert len(splits) >= 3  # At least a few valid splits

        for i, (train_idx, test_idx) in enumerate(splits):
            # Test always comes after train
            assert train_idx.max() < test_idx.min()

    def test_walk_forward_expanding(self, ml_data):
        from src.models.cross_validation import WalkForwardSplit

        X, y = ml_data
        wf = WalkForwardSplit(n_splits=3, train_pct=0.6, expanding=True)

        train_sizes = []
        for train_idx, test_idx in wf.split(X):
            train_sizes.append(len(train_idx))

        # Expanding window: train size should grow
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]


# ============================================================
# Gradient Boosting Model Tests
# ============================================================


class TestXGBoostModel:
    """Tests for XGBoost classifier."""

    def test_train_and_predict(self, ml_data):
        from src.models.classifiers.gradient_boost import XGBoostModel

        X, y = ml_data
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        y_train, y_test = y.iloc[:400], y.iloc[400:]

        model = XGBoostModel(params={"n_estimators": 50, "max_depth": 3})
        metrics = model.train(X_train, y_train)

        assert model.is_fitted
        assert metrics.accuracy > 0.0
        assert metrics.f1_score > 0.0

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, ml_data):
        from src.models.classifiers.gradient_boost import XGBoostModel

        X, y = ml_data
        model = XGBoostModel(params={"n_estimators": 50})
        model.train(X.iloc[:400], y.iloc[:400])

        proba = model.predict_proba(X.iloc[400:])
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importance(self, ml_data):
        from src.models.classifiers.gradient_boost import XGBoostModel

        X, y = ml_data
        model = XGBoostModel(params={"n_estimators": 50})
        model.train(X, y)

        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert importance.iloc[0] >= importance.iloc[-1]  # Sorted descending

    def test_evaluate(self, ml_data):
        from src.models.classifiers.gradient_boost import XGBoostModel

        X, y = ml_data
        model = XGBoostModel(params={"n_estimators": 50})
        model.train(X.iloc[:400], y.iloc[:400])

        metrics = model.evaluate(X.iloc[400:], y.iloc[400:])
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1

    def test_save_and_load(self, ml_data, tmp_path):
        from src.models.classifiers.gradient_boost import XGBoostModel

        X, y = ml_data
        model = XGBoostModel(params={"n_estimators": 50})
        model.train(X, y)

        save_path = str(tmp_path / "xgb_model.pkl")
        model.save(save_path)

        loaded = XGBoostModel.load(save_path)
        assert loaded.is_fitted

        # Predictions should match
        orig_preds = model.predict(X.iloc[:10])
        loaded_preds = loaded.predict(X.iloc[:10])
        np.testing.assert_array_equal(orig_preds, loaded_preds)


class TestLightGBMModel:
    """Tests for LightGBM classifier."""

    def test_train_and_predict(self, ml_data):
        from src.models.classifiers.gradient_boost import LightGBMModel

        X, y = ml_data
        model = LightGBMModel(params={"n_estimators": 50, "verbose": -1})
        model.train(X.iloc[:400], y.iloc[:400])

        assert model.is_fitted
        preds = model.predict(X.iloc[400:])
        assert len(preds) == 100

    def test_feature_importance(self, ml_data):
        from src.models.classifiers.gradient_boost import LightGBMModel

        X, y = ml_data
        model = LightGBMModel(params={"n_estimators": 50, "verbose": -1})
        model.train(X, y)

        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]


class TestGradientBoostEnsemble:
    """Tests for the stacking ensemble."""

    def test_ensemble_train(self, ml_data):
        from src.models.classifiers.gradient_boost import GradientBoostEnsemble

        X, y = ml_data
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        y_train, y_test = y.iloc[:400], y.iloc[400:]

        ensemble = GradientBoostEnsemble()
        metrics = ensemble.train(X_train, y_train)

        assert ensemble.is_fitted
        assert metrics.accuracy > 0.0
        assert len(ensemble._base_models) >= 2

        preds = ensemble.predict(X_test)
        assert len(preds) == len(X_test)


# ============================================================
# Backtest Metrics Tests
# ============================================================


class TestBacktestMetrics:
    """Tests for performance metrics."""

    def test_sharpe_ratio(self):
        from src.backtest.metrics import compute_sharpe

        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        sharpe = compute_sharpe(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_max_drawdown(self):
        from src.backtest.metrics import compute_max_drawdown

        equity = pd.Series([100, 110, 105, 95, 90, 100, 115])
        max_dd, duration = compute_max_drawdown(equity)

        assert max_dd < 0  # Drawdown is negative
        assert max_dd == pytest.approx(-((110 - 90) / 110), abs=1e-6)

    def test_profit_factor(self):
        from src.backtest.metrics import compute_profit_factor

        pnls = [100, -50, 80, -30, 60]
        pf = compute_profit_factor(pnls)
        assert pf == pytest.approx(240 / 80)  # wins/losses = 3.0

    def test_win_rate(self):
        from src.backtest.metrics import compute_win_rate

        pnls = [100, -50, 80, -30, 60]
        wr = compute_win_rate(pnls)
        assert wr == pytest.approx(3 / 5)

    def test_compute_all_metrics(self):
        from src.backtest.metrics import compute_all_metrics

        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        equity = (1 + returns).cumprod() * 100000
        pnls = [100, -50, 80, -30, 60, 120, -40]

        metrics = compute_all_metrics(returns, equity, pnls)
        assert metrics.total_trades == 7
        assert 0 <= metrics.win_rate <= 1


# ============================================================
# Genetic Algorithm Tests
# ============================================================


class TestGeneticOptimizer:
    """Tests for the GA strategy optimizer."""

    def test_evolve_basic(self, price_data):
        from src.strategy.genetic import GeneticOptimizer, GAConfig
        from src.strategy.templates import EMACrossStrategy

        config = GAConfig(
            population_size=10,
            n_generations=3,
            random_seed=42,
        )

        optimizer = GeneticOptimizer(
            strategy_class=EMACrossStrategy,
            param_space={
                "fast_period": (5, 30),
                "slow_period": (20, 100),
            },
            config=config,
        )

        result = optimizer.evolve(price_data, verbose=False)
        assert result.generations == 3
        assert result.best_individual is not None
        assert "fast_period" in result.best_individual.params
        assert "slow_period" in result.best_individual.params
        assert len(result.best_fitness_history) == 3

    def test_ga_fitness_improves(self, price_data):
        from src.strategy.genetic import GeneticOptimizer, GAConfig
        from src.strategy.templates import EMACrossStrategy

        config = GAConfig(
            population_size=20,
            n_generations=10,
            random_seed=42,
        )

        optimizer = GeneticOptimizer(
            strategy_class=EMACrossStrategy,
            param_space={
                "fast_period": (5, 30),
                "slow_period": (20, 100),
            },
            config=config,
        )

        result = optimizer.evolve(price_data, verbose=False)

        # Best fitness should not decrease over generations
        for i in range(1, len(result.best_fitness_history)):
            assert result.best_fitness_history[i] >= result.best_fitness_history[i - 1]


# ============================================================
# Strategy Template Tests
# ============================================================


class TestStrategyTemplates:
    """Tests for strategy templates."""

    def test_ema_cross_signals(self, price_data):
        from src.strategy.templates import EMACrossStrategy, StrategyParams

        params = StrategyParams(
            name="ema_cross_test",
            params={"fast_period": 9, "slow_period": 21},
        )
        strategy = EMACrossStrategy(params)
        signals = strategy.generate_signals(price_data)

        assert len(signals) == len(price_data)
        assert signals.isin(["hold", "buy", "sell"]).all() or signals.isin([0, 1, -1]).all()

    def test_rsi_mean_reversion_signals(self, price_data):
        from src.strategy.templates import RSIMeanReversionStrategy, StrategyParams

        params = StrategyParams(
            name="rsi_test",
            params={"period": 14, "oversold": 30, "overbought": 70},
        )
        strategy = RSIMeanReversionStrategy(params)
        signals = strategy.generate_signals(price_data)

        assert len(signals) == len(price_data)

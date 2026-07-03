"""
Tests for the roadmap features: run registry, data quality monitor,
multi-asset backtester, paper orchestrator, and model monitoring.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioBacktester
from src.data.quality import DataQualityMonitor
from src.execution.orchestrator import PaperOrchestrator, assert_live_allowed
from src.models.monitoring import (
    calibration_report,
    champion_challenger,
    feature_drift,
    psi,
    regime_performance,
)
from src.research.registry import RunRegistry, hash_dataframe
from src.strategy.templates import EMACrossStrategy, StrategyParams

# ============================================================
# Research Run Registry
# ============================================================

class TestRunRegistry:
    def test_start_and_finish_run(self, test_db):
        registry = RunRegistry(test_db)
        run_id = registry.start_run("test bt", kind="backtest",
                                    params={"fast": 9}, seed=42, dataset_hash="abc")
        registry.finish_run(run_id, metrics={"sharpe": 1.5},
                            artifacts={"curve": "runs/x/curve.csv"})

        run = registry.get_run(run_id)
        assert run["status"] == "completed"
        assert run["params"] == {"fast": 9}
        assert run["metrics"] == {"sharpe": 1.5}
        assert run["seed"] == 42
        assert run["dataset_hash"] == "abc"

    def test_finish_unknown_run_raises(self, test_db):
        with pytest.raises(ValueError, match="unknown run_id"):
            RunRegistry(test_db).finish_run("nope")

    def test_list_runs_filters_by_kind(self, test_db):
        registry = RunRegistry(test_db)
        registry.log_run("a", kind="backtest")
        registry.log_run("b", kind="training")
        assert len(registry.list_runs(kind="backtest")) == 1
        assert len(registry.list_runs()) == 2

    def test_hash_dataframe_deterministic(self, sample_ohlcv):
        h1 = hash_dataframe(sample_ohlcv)
        h2 = hash_dataframe(sample_ohlcv.copy())
        assert h1 == h2
        assert h1 != hash_dataframe(sample_ohlcv.iloc[:-1])


# ============================================================
# Data Quality Monitor
# ============================================================

class TestDataQualityMonitor:
    def test_healthy_recent_data(self, test_db):
        # Fresh daily data ending today -> not stale
        dates = pd.date_range(end=pd.Timestamp.utcnow().tz_localize(None), periods=100, freq="D")
        df = pd.DataFrame({
            "timestamp": dates, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.5, "volume": 1000.0,
        })
        test_db.insert_ohlcv_batch(df, "FRESH", "1d", source="test")

        report = DataQualityMonitor(test_db).check("FRESH", "1d")
        assert report.rows == 100
        assert not report.is_stale

    def test_stale_data_detected(self, populated_db):
        # populated_db data starts 2023 -> long stale by now
        report = DataQualityMonitor(populated_db).check("BTC/USDT", "1d")
        assert report.is_stale
        assert any("stale" in issue for issue in report.issues)

    def test_missing_symbol(self, test_db):
        report = DataQualityMonitor(test_db).check("NOPE", "1d")
        assert not report.ok
        assert report.issues == ["no data stored"]

    def test_check_all_covers_summary(self, populated_db):
        reports = DataQualityMonitor(populated_db).check_all()
        assert len(reports) == 1
        assert reports[0].symbol == "BTC/USDT"


# ============================================================
# Multi-Asset Backtester
# ============================================================

def _make_asset(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    return pd.DataFrame({"close": close}, index=dates)


class TestPortfolioBacktester:
    def test_equal_weight_run(self):
        data = {"A": _make_asset(1), "B": _make_asset(2), "C": _make_asset(3)}
        result = PortfolioBacktester(rebalance="M").run(data, weights="equal")

        assert result.final_equity > 0
        assert result.total_rebalances > 5
        assert result.correlation.shape == (3, 3)
        assert all(w == pytest.approx(1 / 3) for w in result.target_weights.values())
        # realized weights stay a valid allocation
        assert result.weights_history.sum(axis=1).round(6).eq(1.0).all()

    def test_custom_weights_normalized(self):
        data = {"A": _make_asset(1), "B": _make_asset(2)}
        result = PortfolioBacktester().run(data, weights={"A": 3, "B": 1})
        assert result.target_weights == {"A": 0.75, "B": 0.25}

    def test_benchmark_comparison(self):
        data = {"A": _make_asset(1), "B": _make_asset(2)}
        result = PortfolioBacktester().run(data, weights="equal",
                                           benchmark=_make_asset(9))
        assert result.benchmark_metrics is not None
        assert result.beta is not None

    def test_rejects_single_asset(self):
        with pytest.raises(ValueError, match="at least 2"):
            PortfolioBacktester().run({"A": _make_asset(1)})

    def test_rebalance_never(self):
        data = {"A": _make_asset(1), "B": _make_asset(2)}
        result = PortfolioBacktester(rebalance="never").run(data)
        assert result.total_rebalances == 0
        assert result.total_cost == 0.0


# ============================================================
# Paper-Trading Orchestrator
# ============================================================

class TestPaperOrchestrator:
    def test_cycle_produces_audited_outcome(self, populated_db, test_config):
        orch = PaperOrchestrator(test_config, populated_db)
        strategy = EMACrossStrategy(StrategyParams(name="ema_cross"))
        outcome = orch.run_cycle("BTC/USDT", "1d", strategy)

        assert outcome.action in {"order_filled", "order_rejected", "hold", "blocked"}
        # audit row always written
        audit = pd.read_sql("SELECT * FROM trades_log", populated_db.engine)
        assert len(audit) == 1
        assert audit.iloc[0]["mode"] == "paper"
        assert "ema_cross" in audit.iloc[0]["explanation"]

    def test_not_enough_data_blocks(self, test_db, test_config):
        orch = PaperOrchestrator(test_config, test_db)
        strategy = EMACrossStrategy(StrategyParams(name="ema_cross"))
        outcome = orch.run_cycle("EMPTY", "1d", strategy)
        assert outcome.action == "blocked"

    def test_live_guard_requires_approval(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUANTMIND_LIVE_APPROVED", raising=False)
        with pytest.raises(RuntimeError, match="QUANTMIND_LIVE_APPROVED"):
            assert_live_allowed(tmp_path)

    def test_live_guard_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTMIND_LIVE_APPROVED", "yes")
        (tmp_path / "KILL_SWITCH").touch()
        with pytest.raises(RuntimeError, match="kill switch"):
            assert_live_allowed(tmp_path)
        (tmp_path / "KILL_SWITCH").unlink()
        assert_live_allowed(tmp_path)  # no raise


# ============================================================
# Model Monitoring
# ============================================================

class TestModelMonitoring:
    def test_psi_zero_for_same_distribution(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 5000))
        assert psi(s, s) == pytest.approx(0.0, abs=1e-9)

    def test_psi_detects_shift(self):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 5000))
        shifted = pd.Series(rng.normal(2, 1, 5000))
        assert psi(ref, shifted) > 0.25

    def test_feature_drift_labels(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"stable": rng.normal(0, 1, 2000),
                            "drifted": rng.normal(0, 1, 2000)})
        live = pd.DataFrame({"stable": rng.normal(0, 1, 2000),
                             "drifted": rng.normal(3, 1, 2000)})
        out = feature_drift(ref, live)
        assert out.iloc[0]["feature"] == "drifted"
        assert out.iloc[0]["status"] == "drifted"
        assert out[out["feature"] == "stable"].iloc[0]["status"] == "stable"

    def test_calibration_report(self):
        rng = np.random.default_rng(0)
        p = pd.Series(rng.uniform(0, 1, 2000))
        y = pd.Series((rng.uniform(0, 1, 2000) < p).astype(int))  # well calibrated
        report = calibration_report(y, p)
        assert report["brier"] < 0.25
        assert not report["bins"].empty

    def test_regime_performance_breakdown(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="D")
        returns = pd.Series(np.where(np.arange(100) < 50, 0.01, -0.01), index=idx)
        regimes = pd.Series(np.where(np.arange(100) < 50, "bull", "bear"), index=idx)
        out = regime_performance(returns, regimes).set_index("regime")
        assert out.loc["bull", "total_return"] > 0
        assert out.loc["bear", "total_return"] < 0

    def test_champion_challenger_promotion(self):
        champ = {"sharpe_ratio": 1.0, "total_return": 0.2, "max_drawdown": -0.2}
        chall = {"sharpe_ratio": 1.5, "total_return": 0.3, "max_drawdown": -0.1}
        assert champion_challenger(champ, chall)["promote_challenger"]
        assert not champion_challenger(chall, champ)["promote_challenger"]

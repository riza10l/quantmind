"""
QuantMind Configuration System
==============================
Type-safe configuration management using Pydantic models and YAML files.

All module configurations are defined here as Pydantic BaseModel subclasses.
Configurations are loaded from YAML files in the configs/ directory, with
environment variable overrides supported via python-dotenv.

Usage:
    from src.core.config import load_config, AppConfig
    config = load_config()
    print(config.data.symbols)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file if present
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# Module Configurations
# ============================================================


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{DATA_DIR / 'quantmind.db'}"
        )
    )
    echo: bool = False  # SQLAlchemy echo SQL statements
    pool_size: int = 5
    max_overflow: int = 10


class RedisConfig(BaseModel):
    """Redis cache configuration."""
    url: str = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    enabled: bool = False  # Disabled by default for local dev


class DataSourceConfig(BaseModel):
    """Configuration for a single data source."""
    provider: str  # "binance", "yahoo", "fear_greed"
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=lambda: ["1d"])
    start_date: str = "2020-01-01"
    end_date: str | None = None  # None = up to now
    enabled: bool = True


class DataConfig(BaseModel):
    """Data pipeline configuration."""
    sources: list[DataSourceConfig] = Field(default_factory=lambda: [
        DataSourceConfig(
            provider="binance",
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
            timeframes=["1d", "4h", "1h"],
            start_date="2020-01-01",
        ),
        DataSourceConfig(
            provider="yahoo",
            symbols=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
            timeframes=["1d"],
            start_date="2015-01-01",
        ),
        DataSourceConfig(
            provider="fear_greed",
            symbols=["crypto"],
            timeframes=["1d"],
            start_date="2018-01-01",
        ),
    ])
    batch_size: int = 1000  # Rows per batch insert
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    validate_data: bool = True
    data_dir: str = str(DATA_DIR)


class FeatureGroupConfig(BaseModel):
    """Configuration for a feature group."""
    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    groups: list[FeatureGroupConfig] = Field(default_factory=lambda: [
        FeatureGroupConfig(
            name="technical",
            params={
                "sma_periods": [7, 14, 21, 50, 100, 200],
                "ema_periods": [9, 12, 21, 26, 50, 100, 200],
                "rsi_periods": [7, 14, 21],
                "macd_fast": [12],
                "macd_slow": [26],
                "macd_signal": [9],
                "bb_periods": [20],
                "bb_std": [2.0],
                "atr_periods": [14, 21],
                "adx_periods": [14],
                "stoch_periods": [14],
                "cci_periods": [14, 20],
                "williams_periods": [14],
                "mfi_periods": [14],
            },
        ),
        FeatureGroupConfig(
            name="statistical",
            params={
                "entropy_window": [20, 50],
                "hurst_window": [100],
                "return_windows": [1, 5, 10, 20, 60],
                "volatility_windows": [10, 20, 50],
            },
        ),
        FeatureGroupConfig(
            name="microstructure",
            params={
                "amihud_window": [20],
                "kyle_window": [20],
            },
        ),
        FeatureGroupConfig(
            name="sentiment",
            params={},
        ),
    ])
    lookback_periods: int = 500  # Max lookback for feature computation
    drop_na: bool = True
    normalize: bool = False  # Normalize features before storage


class SelectionConfig(BaseModel):
    """Feature selection configuration."""
    methods: list[str] = Field(
        default_factory=lambda: ["shap", "mutual_info", "rfe", "permutation"]
    )
    target_column: str = "target_return"  # What to predict
    target_horizon: int = 1  # Bars ahead for target
    top_k: int = 50  # Select top K features
    shap_n_samples: int = 500
    rfe_step: int = 5
    pca_variance_threshold: float = 0.95
    random_state: int = 42


class MLConfig(BaseModel):
    """Machine Learning configuration."""
    models: list[str] = Field(
        default_factory=lambda: ["xgboost", "lightgbm", "catboost"]
    )
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_trials: int = 50  # Optuna hyperparameter search trials
    cross_validation_folds: int = 5


class RLConfig(BaseModel):
    """Reinforcement Learning configuration."""
    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    window_size: int = 50  # Observation window
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    reward_type: str = "sharpe"  # sharpe, sortino, pnl
    drawdown_penalty: float = 0.5
    max_position_pct: float = 1.0


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    latency_ms: float = 50.0
    walk_forward_windows: int = 5
    walk_forward_train_pct: float = 0.7
    monte_carlo_runs: int = 1000
    monte_carlo_method: str = "bootstrap"  # bootstrap, shuffle, noise


class PortfolioConfig(BaseModel):
    """Portfolio optimization configuration."""
    method: str = "risk_parity"  # kelly, risk_parity, cvar, black_litterman
    risk_free_rate: float = 0.05
    max_position_pct: float = 0.25  # Max 25% in a single asset
    min_position_pct: float = 0.01  # Min 1% if allocated
    rebalance_frequency: str = "weekly"


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_daily_drawdown_pct: float = 0.03  # 3% daily max DD
    max_total_drawdown_pct: float = 0.10  # 10% total max DD
    max_consecutive_losses: int = 5
    var_confidence: float = 0.95
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility_scaled
    circuit_breaker_enabled: bool = True


class ExecutionConfig(BaseModel):
    """Trading execution configuration."""
    exchange: str = Field(
        default_factory=lambda: os.getenv("TRADING_EXCHANGE", "binance")
    )
    mode: str = Field(
        default_factory=lambda: os.getenv("TRADING_MODE", "paper")
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("BINANCE_API_KEY", "")
    )
    api_secret: str = Field(
        default_factory=lambda: os.getenv("BINANCE_API_SECRET", "")
    )
    testnet: bool = True
    max_order_retries: int = 3
    order_timeout_seconds: float = 30.0


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    format: str = Field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "console")
    )
    log_dir: str = str(PROJECT_ROOT / "logs")
    log_to_file: bool = True


class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration."""
    tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
    )
    experiment_name: str = "quantmind"
    auto_log: bool = True


# ============================================================
# Root Configuration
# ============================================================


class AppConfig(BaseModel):
    """Root application configuration combining all modules."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


# ============================================================
# Configuration Loading
# ============================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load and merge all YAML config files from the configs directory."""
    if config_dir is None:
        config_dir = CONFIGS_DIR

    merged: dict[str, Any] = {}
    if not config_dir.exists():
        return merged

    # Load each YAML file and merge
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, data)

    return merged


def load_config(config_dir: Path | None = None, profile: str | None = None) -> AppConfig:
    """
    Load the full application configuration.

    Priority (highest to lowest):
    1. Environment variables
    2. Profile YAML (configs/profiles/<profile>.yaml) — research|paper|testnet|live,
       selected via the `profile` arg or QUANTMIND_PROFILE env (default: research)
    3. Base YAML config files
    4. Default values in Pydantic models

    Returns:
        AppConfig: Fully resolved configuration object.
    """
    yaml_data = load_yaml_config(config_dir)

    profile = profile or os.getenv("QUANTMIND_PROFILE", "research")
    profile_file = (config_dir or CONFIGS_DIR) / "profiles" / f"{profile}.yaml"
    if profile_file.exists():
        with open(profile_file, encoding="utf-8") as f:
            yaml_data = _deep_merge(yaml_data, yaml.safe_load(f) or {})
    elif profile != "research":
        raise FileNotFoundError(f"unknown config profile: {profile} ({profile_file})")

    return AppConfig(**yaml_data)

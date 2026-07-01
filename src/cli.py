"""
QuantMind CLI
==============
Command-line interface for all QuantMind operations.

Usage:
    quantmind download         # Download market data
    quantmind features         # Compute features
    quantmind select           # Run feature selection
    quantmind summary          # Show data summary
    quantmind setup            # Initialize database
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure project root is in path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@click.group()
@click.version_option(version="0.1.0", prog_name="QuantMind")
def main() -> None:
    """🧠 QuantMind — Systematic Trading & Research Lab"""
    pass


@main.command()
def setup() -> None:
    """Initialize database and create tables."""
    from src.core.config import load_config
    from src.core.database import DatabaseManager
    from src.core.logger import setup_logging

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    db = DatabaseManager(config.database.url)
    db.initialize()
    click.echo("✅ Database initialized successfully.")


@main.command()
@click.option("--symbols", default=None, help="Comma-separated symbols")
@click.option("--provider", default=None, help="Provider: binance, yahoo, fear_greed")
@click.option("--timeframe", default="1d", help="Timeframe: 1m, 5m, 1h, 4h, 1d")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
def download(symbols, provider, timeframe, start, end) -> None:
    """Download market data from configured sources."""
    from src.core.config import load_config
    from src.core.logger import setup_logging
    from src.data.pipeline import DataPipeline

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    pipeline = DataPipeline(config)

    if symbols and provider:
        for symbol in symbols.split(","):
            rows = pipeline.run_symbol(symbol.strip(), provider, timeframe, start, end)
            click.echo(f"  ✅ {symbol.strip()}: {rows} new rows")
    else:
        stats = pipeline.run()
        click.echo(f"\n📊 Downloaded: {stats['total_downloaded']} | Stored: {stats['total_stored']}")
        if stats["errors"]:
            for err in stats["errors"]:
                click.echo(f"  ⚠️  {err}")


@main.command()
@click.option("--symbol", default=None, help="Specific symbol (default: all)")
@click.option("--timeframe", default="1d", help="Timeframe")
@click.option("--groups", default=None, help="Feature groups (comma-separated)")
def features(symbol, timeframe, groups) -> None:
    """Compute features from downloaded data."""
    from src.core.config import load_config
    from src.core.logger import setup_logging
    from src.features.store import FeatureStore

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    store = FeatureStore(config)

    group_list = groups.split(",") if groups else None

    if symbol:
        result = store.compute_and_store(symbol, timeframe, groups=group_list)
        click.echo(f"✅ {symbol}: {result.shape[1]} features × {result.shape[0]} samples")
    else:
        results = store.compute_all_symbols(timeframe, groups=group_list)
        for sym, count in results.items():
            click.echo(f"  ✅ {sym}: {count} features")


@main.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol for selection")
@click.option("--timeframe", default="1d", help="Timeframe")
@click.option("--top-k", default=50, help="Number of top features to select")
def select(symbol, timeframe, top_k) -> None:
    """Run feature selection pipeline."""
    from src.core.config import load_config
    from src.core.logger import setup_logging
    from src.features.selection import FeatureSelector

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    selector = FeatureSelector(config)

    result = selector.run(symbol, timeframe, top_k=top_k)
    if not result.empty:
        click.echo(f"\n🏆 Top {min(top_k, len(result))} Features:")
        for _, row in result.head(20).iterrows():
            click.echo(f"  {row['rank']:3d}. {row['feature_name']:<30s} ({row['consensus_score']:.6f})")
    else:
        click.echo("⚠️  No features selected. Ensure data and features are computed first.")


@main.command()
def summary() -> None:
    """Show data summary and system status."""
    from src.core.config import load_config
    from src.core.logger import setup_logging
    from src.data.pipeline import DataPipeline
    from src.features.store import FeatureStore

    setup_logging(level="WARNING", log_format="console")
    config = load_config()
    pipeline = DataPipeline(config)

    click.echo("\n📊 QuantMind System Summary")
    click.echo("=" * 50)

    # Data summary
    df = pipeline.get_data_summary()
    if df.empty:
        click.echo("\n📭 No data stored yet. Run: quantmind download")
    else:
        click.echo(f"\n📈 Data Store ({len(df)} symbol/timeframe pairs):")
        click.echo(df.to_string(index=False))

    # Feature summary
    store = FeatureStore(config)
    feat_summary = store.get_feature_summary()
    click.echo(f"\n🔧 Feature Registry: {feat_summary['registered_features']} features")
    for group, count in feat_summary.get("features_by_group", {}).items():
        click.echo(f"  • {group}: {count}")

    click.echo()


@main.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol to train on")
@click.option("--timeframe", default="1d", help="Timeframe")
@click.option("--models", default="xgboost,lightgbm", help="Comma-separated list of models")
@click.option("--trials", default=10, help="Number of Optuna tuning trials")
def train(symbol, timeframe, models, trials) -> None:
    """Train ML models and search for best hyperparameters."""
    from src.core.config import load_config
    from src.core.logger import setup_logging
    from src.models.trainer import ModelTrainer

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    trainer = ModelTrainer(config)

    model_list = [m.strip() for m in models.split(",")]
    click.echo(f"\n🚀 Training Models: {model_list} on {symbol} ({timeframe})")
    click.echo("=" * 50)
    
    trainer.run(
        symbol=symbol,
        timeframe=timeframe,
        models=model_list,
        auto_tune=True,
        n_trials=trials
    )


if __name__ == "__main__":
    main()

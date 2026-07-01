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

TIMEFRAME_TYPE = click.Choice(
    ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
    case_sensitive=False,
)
MODEL_TYPES = {"xgboost", "lightgbm", "catboost", "ensemble"}


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
@click.option(
    "--provider",
    type=click.Choice(["binance", "yahoo"], case_sensitive=False),
    default=None,
    help="OHLCV provider",
)
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
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

    if bool(symbols) != bool(provider):
        raise click.UsageError("--symbols and --provider must be supplied together")

    if symbols and provider:
        for symbol in symbols.split(","):
            normalized_symbol = symbol.strip()
            if not normalized_symbol:
                raise click.BadParameter("symbols cannot be empty", param_hint="--symbols")

            rows = pipeline.run_symbol(normalized_symbol, provider, timeframe, start, end)
            if pipeline.errors:
                raise click.ClickException(pipeline.errors[-1])
            click.echo(f"  ✅ {normalized_symbol}: {rows} new rows")
    else:
        stats = pipeline.run()
        click.echo(f"\n📊 Downloaded: {stats['total_downloaded']} | Stored: {stats['total_stored']}")
        if stats["errors"]:
            for err in stats["errors"]:
                click.echo(f"  ⚠️  {err}")


@main.command()
@click.option("--symbol", default=None, help="Specific symbol (default: all)")
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
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
        if result.empty:
            raise click.ClickException(
                f"No complete feature rows available for {symbol}/{timeframe}. "
                "Run the download command first and ensure enough history is available."
            )
        click.echo(f"✅ {symbol}: {result.shape[1]} features × {result.shape[0]} samples")
    else:
        results = store.compute_all_symbols(timeframe, groups=group_list)
        if not results or not any(results.values()):
            raise click.ClickException(
                f"No OHLCV data available for timeframe {timeframe}. "
                "Run the download command first."
            )
        for sym, count in results.items():
            click.echo(f"  ✅ {sym}: {count} features")


@main.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol for selection")
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Number of top features to select",
)
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
        raise click.ClickException(
            f"No features selected for {symbol}/{timeframe}. Run the download and "
            "features commands first, and ensure at least 50 samples remain."
        )


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
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
@click.option("--models", default="xgboost,lightgbm", help="Comma-separated list of models")
@click.option(
    "--trials",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of Optuna tuning trials",
)
def train(symbol, timeframe, models, trials) -> None:
    """Train ML models and search for best hyperparameters."""
    from src.core.config import load_config
    from src.core.logger import setup_logging
    from src.models.trainer import ModelTrainer

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    trainer = ModelTrainer(config)

    model_list = [model.strip().lower() for model in models.split(",") if model.strip()]
    invalid_models = sorted(set(model_list) - MODEL_TYPES)
    if not model_list or invalid_models:
        supported = ", ".join(sorted(MODEL_TYPES))
        invalid = ", ".join(invalid_models) if invalid_models else "empty value"
        raise click.BadParameter(
            f"unsupported model(s): {invalid}. Supported: {supported}",
            param_hint="--models",
        )

    click.echo(f"\n🚀 Training Models: {model_list} on {symbol} ({timeframe})")
    click.echo("=" * 50)

    try:
        results = trainer.run(
            symbol=symbol,
            timeframe=timeframe,
            models=model_list,
            auto_tune=True,
            n_trials=trials,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"{exc} Run: python src/cli.py features --symbol {symbol} "
            f"--timeframe {timeframe}"
        ) from exc

    if not results:
        raise click.ClickException(
            "No models were trained. Check that the requested ML dependencies are installed "
            "and review the model errors above."
        )


@main.command()
@click.option("--port", default=8501, help="Port to run the dashboard on")
def dashboard(port) -> None:
    """Launch the Streamlit web dashboard."""
    import subprocess
    import sys
    from pathlib import Path
    
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    
    click.echo(f"Starting QuantMind Dashboard on http://localhost:{port} ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)])


if __name__ == "__main__":
    main()


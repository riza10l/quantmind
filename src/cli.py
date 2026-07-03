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

# Windows terminals default to cp1252, which can't render emoji output
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

TIMEFRAME_TYPE = click.Choice(
    ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
    case_sensitive=False,
)
MODEL_TYPES = {"xgboost", "lightgbm", "catboost", "ensemble"}


@click.group()
@click.version_option(version="0.1.0", prog_name="QuantMind")
def main() -> None:
    """QuantMind — Systematic Trading & Research Lab"""
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


STRATEGY_TYPES = {"ema_cross", "rsi", "bollinger", "macd"}


@main.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol to backtest")
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
@click.option(
    "--strategy",
    type=click.Choice(sorted(STRATEGY_TYPES), case_sensitive=False),
    default="ema_cross",
    show_default=True,
)
@click.option("--capital", type=click.FloatRange(min_open=True, min=0), default=10_000.0, show_default=True)
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--monte-carlo", is_flag=True, help="Run Monte Carlo robustness analysis")
def backtest(symbol, timeframe, strategy, capital, start, end, monte_carlo) -> None:
    """Backtest a strategy template on stored OHLCV data."""
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.core.config import load_config
    from src.core.database import DatabaseManager
    from src.core.logger import setup_logging
    from src.strategy.templates import (
        BollingerBreakoutStrategy,
        EMACrossStrategy,
        MACDMomentumStrategy,
        RSIMeanReversionStrategy,
        StrategyParams,
    )

    setup_logging(level="WARNING", log_format="console")
    config = load_config()
    db = DatabaseManager(config.database.url)
    df = db.query_ohlcv(symbol, timeframe, start=start, end=end)
    if df.empty or len(df) < 50:
        raise click.ClickException(
            f"Not enough OHLCV data for {symbol}/{timeframe} (found {len(df)} rows). "
            "Run: python src/cli.py download"
        )

    strategy_cls = {
        "ema_cross": EMACrossStrategy,
        "rsi": RSIMeanReversionStrategy,
        "bollinger": BollingerBreakoutStrategy,
        "macd": MACDMomentumStrategy,
    }[strategy.lower()]
    strat = strategy_cls(StrategyParams(name=strategy.lower()))

    engine = BacktestEngine(BacktestConfig(initial_capital=capital))
    result = engine.run(df, strat, symbol=symbol)

    click.echo(f"\nBacktest: {strategy} on {symbol} ({timeframe}) "
               f"{result.start_date:%Y-%m-%d} -> {result.end_date:%Y-%m-%d}")
    click.echo("=" * 50)
    click.echo(f"  Final equity:   ${result.final_equity:,.2f} ({result.total_return:+.2%})")
    click.echo(f"  Sharpe:         {result.sharpe_ratio:.2f}   Sortino: {result.sortino_ratio:.2f}")
    click.echo(f"  Max drawdown:   {result.max_drawdown:.2%} ({result.max_drawdown_duration_days} bars)")
    click.echo(f"  Trades:         {result.total_trades}  Win rate: {result.win_rate:.1%}  "
               f"PF: {result.profit_factor:.2f}")
    click.echo(f"  Costs:          commission ${result.total_commission:,.2f}, "
               f"slippage ${result.total_slippage:,.2f}")

    if monte_carlo:
        mc = engine.monte_carlo(result)
        click.echo("\n  Monte Carlo (1000 runs):")
        click.echo(f"    Median equity:  ${mc.median_equity:,.2f}")
        click.echo(f"    5th percentile: ${mc.var_95_equity:,.2f}")
        click.echo(f"    P(loss):        {mc.prob_loss:.1%}")

    # Record the run for reproducibility
    from src.research.registry import RunRegistry, hash_dataframe
    run_id = RunRegistry(db).log_run(
        name=f"{strategy} {symbol} {timeframe}",
        kind="backtest",
        params={"strategy": strategy, "symbol": symbol, "timeframe": timeframe,
                "capital": capital, "start": start, "end": end},
        metrics={"total_return": result.total_return, "sharpe": result.sharpe_ratio,
                 "max_drawdown": result.max_drawdown, "trades": result.total_trades,
                 "win_rate": result.win_rate, "profit_factor": result.profit_factor},
        dataset_hash=hash_dataframe(df),
    )
    click.echo(f"\n  Run recorded: {run_id}")


@main.command()
@click.option("--kind", default=None, help="Filter by run kind (backtest/training/...)")
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True)
def runs(kind, limit) -> None:
    """List recorded research runs (registry)."""
    from src.core.config import load_config
    from src.core.database import DatabaseManager
    from src.research.registry import RunRegistry

    config = load_config()
    registry = RunRegistry(DatabaseManager(config.database.url))
    df = registry.list_runs(kind=kind, limit=limit)
    if df.empty:
        click.echo("No research runs recorded yet.")
        return
    df["git_commit"] = df["git_commit"].str[:8]
    df["dataset_hash"] = df["dataset_hash"].str[:8]
    click.echo(df.to_string(index=False))


@main.command()
def quality() -> None:
    """Run the data quality monitor over all stored data."""
    from src.core.config import load_config
    from src.core.database import DatabaseManager
    from src.core.logger import setup_logging
    from src.data.quality import DataQualityMonitor

    setup_logging(level="WARNING", log_format="console")
    config = load_config()
    monitor = DataQualityMonitor(DatabaseManager(config.database.url))
    reports = monitor.check_all()
    if not reports:
        click.echo("No data stored yet. Run: python src/cli.py download")
        return

    click.echo(monitor.to_dataframe(reports).to_string(index=False))
    bad = [r for r in reports if not r.ok]
    click.echo(f"\n{len(reports) - len(bad)}/{len(reports)} pairs healthy.")
    if bad:
        click.echo("Issues found:")
        for r in bad:
            for issue in r.issues:
                click.echo(f"  {r.symbol}/{r.timeframe}: {issue}")


@main.command()
@click.option("--symbols", required=True, help="Comma-separated symbols (min 2)")
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
@click.option("--weights", default="equal", help='"equal" or comma list matching symbols (e.g. 0.5,0.3,0.2)')
@click.option("--rebalance", type=click.Choice(["D", "W", "M", "Q", "never"]), default="M", show_default=True)
@click.option("--benchmark", default=None, help="Benchmark symbol (must be stored)")
@click.option("--capital", type=click.FloatRange(min_open=True, min=0), default=10_000.0, show_default=True)
def portfolio(symbols, timeframe, weights, rebalance, benchmark, capital) -> None:
    """Multi-asset portfolio backtest with rebalancing."""
    from src.backtest.portfolio import PortfolioBacktester
    from src.core.config import load_config
    from src.core.database import DatabaseManager
    from src.core.logger import setup_logging
    from src.research.registry import RunRegistry

    setup_logging(level="WARNING", log_format="console")
    config = load_config()
    db = DatabaseManager(config.database.url)

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if len(symbol_list) < 2:
        raise click.BadParameter("need at least 2 symbols", param_hint="--symbols")

    data = {}
    for sym in symbol_list:
        df = db.query_ohlcv(sym, timeframe)
        if df.empty:
            raise click.ClickException(f"No data for {sym}/{timeframe}. Run download first.")
        data[sym] = df

    if weights == "equal":
        weight_arg = "equal"
    else:
        parts = [float(w) for w in weights.split(",")]
        if len(parts) != len(symbol_list):
            raise click.BadParameter("weights count must match symbols", param_hint="--weights")
        weight_arg = dict(zip(symbol_list, parts, strict=True))

    bench_df = db.query_ohlcv(benchmark, timeframe) if benchmark else None
    bt = PortfolioBacktester(initial_capital=capital, rebalance=rebalance)
    result = bt.run(data, weights=weight_arg, benchmark=bench_df)

    m = result.metrics
    click.echo(f"\nPortfolio Backtest: {', '.join(result.symbols)} ({timeframe}, rebalance={rebalance})")
    click.echo("=" * 60)
    click.echo(f"  Final equity:   ${result.final_equity:,.2f} ({m.total_return:+.2%})")
    click.echo(f"  Sharpe:         {m.sharpe_ratio:.2f}   Max DD: {m.max_drawdown:.2%}")
    click.echo(f"  Rebalances:     {result.total_rebalances}  (cost ${result.total_cost:,.2f})")
    click.echo("  Max exposure:   " + ", ".join(
        f"{s}={w:.1%}" for s, w in result.max_exposure.items()))
    click.echo("\n  Correlation matrix:")
    click.echo(result.correlation.round(2).to_string())
    if result.benchmark_metrics:
        bm = result.benchmark_metrics
        click.echo(f"\n  vs {benchmark}: return {m.total_return:+.2%} vs {bm.total_return:+.2%}, "
                   f"beta {result.beta:.2f}, alpha {result.alpha:+.2%}")

    RunRegistry(db).log_run(
        name=f"portfolio {','.join(result.symbols)}",
        kind="portfolio",
        params={"symbols": symbol_list, "weights": str(weight_arg),
                "rebalance": rebalance, "benchmark": benchmark},
        metrics={"total_return": m.total_return, "sharpe": m.sharpe_ratio,
                 "max_drawdown": m.max_drawdown},
    )


@main.command("paper-run")
@click.option("--symbol", default="BTC/USDT", help="Symbol to trade")
@click.option("--timeframe", type=TIMEFRAME_TYPE, default="1d", show_default=True)
@click.option(
    "--strategy",
    type=click.Choice(sorted(STRATEGY_TYPES), case_sensitive=False),
    default="ema_cross",
    show_default=True,
)
def paper_run(symbol, timeframe, strategy) -> None:
    """Run one paper-trading cycle: data -> signal -> risk check -> order -> audit."""
    from src.core.config import load_config
    from src.core.database import DatabaseManager
    from src.core.logger import setup_logging
    from src.execution.orchestrator import PaperOrchestrator
    from src.strategy.templates import (
        BollingerBreakoutStrategy,
        EMACrossStrategy,
        MACDMomentumStrategy,
        RSIMeanReversionStrategy,
        StrategyParams,
    )

    setup_logging(level="INFO", log_format="console")
    config = load_config()
    db = DatabaseManager(config.database.url)
    strategy_cls = {
        "ema_cross": EMACrossStrategy,
        "rsi": RSIMeanReversionStrategy,
        "bollinger": BollingerBreakoutStrategy,
        "macd": MACDMomentumStrategy,
    }[strategy.lower()]

    orch = PaperOrchestrator(config, db)
    outcome = orch.run_cycle(symbol, timeframe, strategy_cls(StrategyParams(name=strategy)))
    click.echo(f"\nPaper cycle: {symbol} [{strategy}]")
    click.echo(f"  Signal:  {outcome.signal}")
    click.echo(f"  Action:  {outcome.action}" + (f" ({outcome.reason})" if outcome.reason else ""))
    if outcome.order_id:
        click.echo(f"  Order:   {outcome.order_id} qty={outcome.quantity:.6f} "
                   f"@ {outcome.fill_price:,.2f}")
    click.echo("  Audit:   logged to trades_log (mode=paper)")


@main.command()
@click.option(
    "--port",
    type=click.IntRange(min=1, max=65535),
    default=8501,
    show_default=True,
    help="Port to run the dashboard on",
)
def dashboard(port) -> None:
    """Launch the Streamlit web dashboard."""
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("streamlit")
    except PackageNotFoundError as exc:
        raise click.ClickException(
            "Streamlit is not installed in this Python environment. "
            'Run: python -m pip install -e ".[dashboard]"'
        ) from exc

    app_path = Path(__file__).parent / "dashboard" / "app.py"

    click.echo(f"Starting QuantMind Dashboard on http://localhost:{port} ...")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Dashboard process exited with status {exc.returncode}."
        ) from exc


if __name__ == "__main__":
    main()

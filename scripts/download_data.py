"""
Data Download Script
=====================
Downloads market data from all configured sources.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --symbols BTC/USDT,ETH/USDT --provider binance
    python scripts/download_data.py --summary
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import click

from src.core.config import load_config
from src.core.logger import setup_logging


@click.command()
@click.option("--symbols", default=None, help="Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)")
@click.option("--provider", default=None, help="Provider name (binance, yahoo, fear_greed)")
@click.option("--timeframe", default="1d", help="Timeframe (1m, 5m, 1h, 4h, 1d)")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--summary", is_flag=True, help="Show data summary only")
def main(
    symbols: str | None,
    provider: str | None,
    timeframe: str,
    start: str | None,
    end: str | None,
    summary: bool,
) -> None:
    """Download market data for QuantMind."""
    setup_logging(level="INFO", log_format="console")

    from src.core.config import load_config
    from src.data.pipeline import DataPipeline

    config = load_config()
    pipeline = DataPipeline(config)

    if summary:
        df = pipeline.get_data_summary()
        if df.empty:
            print("\n📭 No data stored yet. Run download first.")
        else:
            print("\n📊 Data Summary:")
            print(df.to_string(index=False))
        return

    if symbols and provider:
        # Download specific symbols
        symbol_list = [s.strip() for s in symbols.split(",")]
        for symbol in symbol_list:
            rows = pipeline.run_symbol(
                symbol=symbol,
                provider=provider,
                timeframe=timeframe,
                start_date=start,
                end_date=end,
            )
            print(f"  ✅ {symbol}: {rows} new rows")
    else:
        # Run full pipeline from config
        stats = pipeline.run()
        print(f"\n📊 Pipeline Complete:")
        print(f"   Downloaded: {stats['total_downloaded']} rows")
        print(f"   Stored: {stats['total_stored']} new rows")
        if stats['errors']:
            print(f"   ⚠️  Errors: {len(stats['errors'])}")
            for err in stats['errors']:
                print(f"      - {err}")


if __name__ == "__main__":
    main()

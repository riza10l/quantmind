"""
Database Setup Script
=====================
Initializes the QuantMind database with all required tables.
Run this once before using the system.

Usage: python scripts/setup_db.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.core.database import DatabaseManager
from src.core.logger import get_logger, setup_logging

setup_logging(level="INFO", log_format="console")
logger = get_logger("scripts.setup_db")


def main() -> None:
    config = load_config()
    db = DatabaseManager(config.database.url)

    logger.info("initializing_database", url=config.database.url)
    db.initialize()

    logger.info("database_ready", url=config.database.url)
    print(f"\n✅ Database initialized at: {config.database.url}")
    print("   All tables created successfully.")


if __name__ == "__main__":
    main()

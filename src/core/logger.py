"""
QuantMind Structured Logging
============================
Module-aware structured logging using structlog. Produces JSON logs in production
and human-readable colored output in development.

Usage:
    from src.core.logger import get_logger
    logger = get_logger("data.pipeline")
    logger.info("downloading_data", symbol="BTC/USDT", source="binance")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


_configured = False


def setup_logging(
    level: str = "INFO",
    log_format: str = "console",
    log_dir: str | None = None,
    log_to_file: bool = False,
) -> None:
    """
    Configure the logging system for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - "console" for dev, "json" for production.
        log_dir: Directory for log files.
        log_to_file: Whether to also write logs to a file.
    """
    global _configured
    if _configured:
        return

    # Shared processors for all output
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event_to=40,
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler (optional)
    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / "quantmind.log",
            encoding="utf-8",
        )
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["urllib3", "ccxt", "asyncio", "matplotlib"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True


def get_logger(module_name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger bound to a module name.

    Args:
        module_name: Dot-separated module path (e.g., "data.pipeline").

    Returns:
        A structlog BoundLogger instance with module context.
    """
    if not _configured:
        setup_logging()

    return structlog.get_logger(module_name)

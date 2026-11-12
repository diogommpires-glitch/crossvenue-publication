"""Structured logging with file rotation for trading engine.

Provides standardized logging infrastructure with:
- Rotating file handlers to manage disk space
- Structured log format with timestamps
- Separate handlers for different severity levels
- Performance metrics logging
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "trading_engine",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Setup structured logger with rotating file handlers.

    Parameters:
    ----------
    name : str
        Logger name
    log_dir : str
        Directory for log files
    max_bytes : int
        Max size per log file before rotation
    backup_count : int
        Number of backup files to keep
    console_level : int
        Logging level for console output
    file_level : int
        Logging level for file output

    Returns:
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Structured log format with timestamps
    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Main file handler (all logs)
    main_log_file = log_path / f"{name}.log"
    main_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    main_handler.setLevel(file_level)
    main_handler.setFormatter(log_format)
    logger.addHandler(main_handler)

    # Error file handler (errors only)
    error_log_file = log_path / f"{name}_errors.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    logger.addHandler(error_handler)

    # Trades file handler (separate file for trade logs)
    trades_log_file = log_path / f"{name}_trades.log"
    trades_handler = RotatingFileHandler(
        trades_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    trades_handler.setLevel(logging.INFO)
    # Add filter to only capture trade-related logs
    trades_handler.addFilter(lambda record: "TRADE" in record.getMessage())
    trades_handler.setFormatter(log_format)
    logger.addHandler(trades_handler)

    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log directory: {log_path.absolute()}")

    return logger


def log_trade(logger: logging.Logger, fill_data: dict) -> None:
    """Log a trade execution with structured format.

    Parameters:
    ----------
    logger : logging.Logger
        Logger instance
    fill_data : dict
        Trade fill information
    """
    logger.info(
        f"TRADE | side={fill_data.get('side')} | "
        f"qty={fill_data.get('quantity')} | "
        f"price={fill_data.get('price'):.4f} | "
        f"order_id={fill_data.get('order_id')} | "
        f"trader_id={fill_data.get('trader_id')}"
    )


def log_pnl(logger: logging.Logger, pnl_data: dict) -> None:
    """Log P&L update with structured format.

    Parameters:
    ----------
    logger : logging.Logger
        Logger instance
    pnl_data : dict
        P&L information
    """
    logger.info(
        f"PNL | unrealized={pnl_data.get('unrealized_pnl', 0.0):.2f} | "
        f"realized={pnl_data.get('realized_pnl', 0.0):.2f} | "
        f"total={pnl_data.get('total_pnl', 0.0):.2f} | "
        f"inventory={pnl_data.get('inventory', 0)}"
    )


def log_health(logger: logging.Logger, health_data: dict) -> None:
    """Log health metrics with structured format.

    Parameters:
    ----------
    logger : logging.Logger
        Logger instance
    health_data : dict
        Health metrics
    """
    logger.info(
        f"HEALTH | uptime={health_data.get('uptime_seconds', 0):.0f}s | "
        f"events={health_data.get('total_events', 0)} | "
        f"fills={health_data.get('total_fills', 0)} | "
        f"errors={health_data.get('error_count', 0)}"
    )

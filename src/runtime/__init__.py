"""Continuous trading engine runtime.

This package provides the infrastructure for running market-making strategies
in a continuous, production-like environment with:
- Event loop for real-time market data processing
- State persistence (trades, P&L, positions)
- Structured logging with rotation
- Health monitoring and metrics
- Graceful shutdown handling
"""

from .trading_engine import TradingEngine
from .state_manager import StateManager
from .logger import setup_logger
from .health_monitor import HealthMonitor

__all__ = [
    "TradingEngine",
    "StateManager",
    "setup_logger",
    "HealthMonitor",
]

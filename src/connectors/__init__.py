"""Public market-data and simulated execution connectors.

`BinanceConnector` reads public market data only. `PaperTrader` simulates local
execution. Authenticated exchange account access and order submission are
disabled throughout the supported project interface.
"""

from .base_connector import (
    BaseConnector,
    ConnectionStatus,
    OrderBookSnapshot,
    OrderResponse,
)
from .binance_connector import BinanceConnector
from .binance_provider import BinancePublicDataProvider
from .market_data import MarketDataProvider, ProviderCapabilities, ProviderStatus, ProviderState
from .paper_trader import PaperTrader

__all__ = [
    "BaseConnector",
    "ConnectionStatus",
    "OrderBookSnapshot",
    "OrderResponse",
    "BinanceConnector",
    "BinancePublicDataProvider",
    "MarketDataProvider",
    "ProviderCapabilities",
    "ProviderStatus",
    "ProviderState",
    "PaperTrader",
]

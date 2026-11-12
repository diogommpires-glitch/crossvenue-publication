"""Credential-free configuration for Binance public data and local paper trading."""

PAPER_CONFIG = {
    "mode": "paper",
    "symbol": "BTCUSDT",
    "market_data_source": "binance_public",
    "initial_balances": {"BTC": 0.0, "USDT": 50000.0},
    "paper_queue_fraction": 0.5,
    "allow_short": False,
}

# Exchange order submission and authenticated account access are not supported.

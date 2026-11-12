"""Tests for trading strategies."""

import pytest
from src.strategies.avellaneda_stoikov import AvellanedaStoikov
from src.utils.types import Price, Timestamp


def test_avellaneda_order_size_reasonable():
    """Ensure order sizes make sense at realistic BTC prices."""
    strategy = AvellanedaStoikov(order_size=0.01)  # 0.01 BTC

    # Realistic BTC/USDT price
    mid_price = 67000.0

    # Create market state
    market_state = {
        "time": 0.0,
        "mid_price": mid_price,
        "best_bid": mid_price - 1.0,
        "best_ask": mid_price + 1.0,
    }

    quotes = strategy.on_market_event(market_state)

    # First call initializes, second call generates quotes
    if len(quotes) == 0:
        market_state["time"] = 1.1  # Wait past quote_interval (1.0s)
        quotes = strategy.on_market_event(market_state)

    # Should generate 2 quotes (bid and ask)
    assert len(quotes) == 2, f"Should generate bid and ask, got {len(quotes)}"

    # Each quote should be exactly 0.01 BTC
    assert abs(quotes[0].quantity - 0.01) < 1e-6, "Bid quantity should be 0.01 BTC"
    assert abs(quotes[1].quantity - 0.01) < 1e-6, "Ask quantity should be 0.01 BTC"

    # At $67k, notional should be ~$670 per side
    bid_notional = quotes[0].quantity * mid_price
    ask_notional = quotes[1].quantity * mid_price

    assert 600 < bid_notional < 700, f"Bid notional ${bid_notional} out of range"
    assert 600 < ask_notional < 700, f"Ask notional ${ask_notional} out of range"


def test_order_size_vs_risk_limits():
    """Ensure order_size respects risk limits."""
    # Risk limit: max 0.01 BTC per order
    strategy = AvellanedaStoikov(order_size=0.01, max_inventory=0.1)

    mid_price = 67000.0
    market_state = {
        "time": 0.0,
        "mid_price": mid_price,
        "best_bid": mid_price - 1.0,
        "best_ask": mid_price + 1.0,
    }

    quotes = strategy.on_market_event(market_state)

    # First call initializes, second call generates quotes
    if len(quotes) == 0:
        market_state["time"] = 1.1  # Wait past quote_interval
        quotes = strategy.on_market_event(market_state)

    # Each quote must be <= 0.01 BTC (the risk limit)
    assert len(quotes) >= 2, f"Should generate quotes, got {len(quotes)}"
    assert quotes[0].quantity <= 0.01, "Bid exceeds risk limit"
    assert quotes[1].quantity <= 0.01, "Ask exceeds risk limit"

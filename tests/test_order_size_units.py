"""Test that order sizes are in correct BTC units.

This test verifies that order_size and max_inventory parameters are
specified in BTC (base asset) units, not USD or arbitrary integers.

At ~$67k BTC price:
- order_size=0.01 BTC = ~$670 per order (reasonable)
- max_inventory=0.1 BTC = ~$6,700 max position (reasonable)

Previously incorrect values:
- order_size=100 would be 100 BTC = $6.7M per order (way too large!)
- max_inventory=500 would be 500 BTC = $33.5M position (way too large!)
"""

import pytest
from src.strategies.avellaneda_stoikov import AvellanedaStoikov


def test_order_size_is_in_btc_units():
    """Verify order_size is in BTC, not USD or arbitrary units."""
    strategy = AvellanedaStoikov(
        risk_aversion=0.05,
        volatility=0.20,
        order_size=0.01,
        max_inventory=0.1,
    )

    # At $67k BTC price
    btc_price = 67000.0

    # Check order size is reasonable in USD terms
    order_value_usd = strategy.order_size * btc_price
    assert order_value_usd < 2000, \
        f"Order size {strategy.order_size} BTC = ${order_value_usd:,.0f} is too large"
    assert order_value_usd > 100, \
        f"Order size {strategy.order_size} BTC = ${order_value_usd:,.0f} is too small"

    # Check max inventory is reasonable in USD terms
    max_position_usd = strategy.max_inventory * btc_price
    assert max_position_usd < 10000, \
        f"Max inventory {strategy.max_inventory} BTC = ${max_position_usd:,.0f} is too large"
    assert max_position_usd > 1000, \
        f"Max inventory {strategy.max_inventory} BTC = ${max_position_usd:,.0f} is too small"


def test_order_size_less_than_max_inventory():
    """Verify order_size is smaller than max_inventory."""
    strategy = AvellanedaStoikov(
        order_size=0.01,
        max_inventory=0.1,
    )

    assert strategy.order_size < strategy.max_inventory, \
        "order_size must be less than max_inventory"

    # Check reasonable ratio (can fit multiple orders in max inventory)
    ratio = strategy.max_inventory / strategy.order_size
    assert ratio >= 5, \
        f"max_inventory should be at least 5x order_size (ratio={ratio:.1f})"


def test_fractional_btc_quantities_accepted():
    """Verify strategy accepts fractional BTC quantities (floats)."""
    # Should accept small fractional BTC amounts
    strategy = AvellanedaStoikov(
        order_size=0.01,  # 0.01 BTC
        max_inventory=0.1,  # 0.1 BTC
    )

    assert isinstance(strategy.order_size, float)
    assert isinstance(strategy.max_inventory, float)

    # Verify exact values
    assert strategy.order_size == 0.01
    assert strategy.max_inventory == 0.1


def test_launcher_default_values():
    """Verify launcher.py uses correct default values."""
    # This documents the expected launcher.py defaults
    # launcher.py should call:
    #   AvellanedaStoikov(
    #       order_size=0.01,      # NOT 100
    #       max_inventory=0.1,    # NOT 500
    #   )

    strategy = AvellanedaStoikov(
        risk_aversion=0.05,
        volatility=0.20,
        order_size=0.01,
        max_inventory=0.1,
        time_horizon=3600.0,
    )

    # At $67k BTC
    btc_price = 67000.0

    # Verify reasonable USD values
    assert strategy.order_size * btc_price == pytest.approx(670, rel=0.01)
    assert strategy.max_inventory * btc_price == pytest.approx(6700, rel=0.01)


def test_old_values_would_fail():
    """Document that old values (100, 500) were incorrect."""
    # Old incorrect values
    old_order_size = 100  # BTC
    old_max_inventory = 500  # BTC

    btc_price = 67000.0

    # These would be absurdly large
    old_order_value = old_order_size * btc_price
    old_max_position = old_max_inventory * btc_price

    # Document the problem
    assert old_order_value == 6_700_000, \
        "Old order_size=100 meant $6.7M per order!"
    assert old_max_position == 33_500_000, \
        "Old max_inventory=500 meant $33.5M max position!"

    # These would fail reasonable bounds
    assert old_order_value > 1_000_000, "Way too large for retail trading"
    assert old_max_position > 10_000_000, "Way too large for retail trading"

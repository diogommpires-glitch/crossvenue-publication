"""Example usage of the configuration system.

Demonstrates loading configuration, validating orders, and tracking risk limits.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, RiskValidator


def demo_basic_usage():
    """Demonstrate basic configuration loading."""
    print("=" * 60)
    print("BASIC CONFIGURATION USAGE")
    print("=" * 60)

    # Initialize config manager
    config_manager = ConfigManager()

    # Load configuration files
    try:
        trading_config = config_manager.load_config()
        risk_limits = config_manager.load_risk_limits()
        print("[OK] Configuration loaded successfully")
    except Exception as e:
        print(f"[ERR] Configuration error: {e}")
        return None, None

    # Display key settings
    print(f"\nTrading Mode: {config_manager.get_mode().upper()}")
    print(f"Paper Trading: {config_manager.is_paper_trading()}")

    strategy = config_manager.get_strategy_params()
    print(f"\nStrategy: {strategy['type']}")
    print(f"Risk Aversion: {strategy['risk_aversion']}")
    print(f"Order Refresh: {strategy['order_refresh_seconds']}s")
    print(f"Spread Range: {strategy['min_spread_bps']}-{strategy['max_spread_bps']} bps")

    exchange = config_manager.get_exchange_config()
    print(f"\nExchange: {exchange['name']}")
    print(f"Market data: {exchange.get('market_data', 'public')}")
    print("Execution: local paper only; API credentials are not used")

    return config_manager, risk_limits


def demo_risk_validation(risk_limits):
    """Demonstrate risk validation."""
    print("\n" + "=" * 60)
    print("RISK VALIDATION")
    print("=" * 60)

    validator = RiskValidator(risk_limits)

    # Simulate some trades
    print("\n--- Validating Orders ---")

    # Test order 1: Valid small order
    is_valid, error = validator.validate_order(
        side="buy",
        size=0.05,
        price=50000.0,
        current_book_depth=100000.0
    )
    print(f"\n1. Buy 0.05 @ $50,000")
    if is_valid:
        print("   [OK] Order valid")
        validator.record_order()
        validator.update_position(base_delta=0.05, quote_delta=-2500.0)
    else:
        print(f"   [--] Order rejected: {error}")

    # Test order 2: Valid sell order
    is_valid, error = validator.validate_order(
        side="sell",
        size=0.02,
        price=50100.0,
        current_book_depth=100000.0
    )
    print(f"\n2. Sell 0.02 @ $50,100")
    if is_valid:
        print("   [OK] Order valid")
        validator.record_order()
        validator.update_position(base_delta=-0.02, quote_delta=1002.0)
        validator.record_trade_pnl(pnl=100.0)  # Profitable trade
    else:
        print(f"   [--] Order rejected: {error}")

    # Test order 3: Too large
    is_valid, error = validator.validate_order(
        side="buy",
        size=5.0,
        price=50000.0,
        current_book_depth=100000.0
    )
    print(f"\n3. Buy 5.0 @ $50,000 (should exceed limit)")
    if is_valid:
        print("   [OK] Order valid")
    else:
        print(f"   [--] Order rejected: {error}")

    # Test order 4: Too small
    is_valid, error = validator.validate_order(
        side="buy",
        size=0.0001,
        price=50000.0,
        current_book_depth=100000.0
    )
    print(f"\n4. Buy 0.0001 @ $50,000 (should be below minimum)")
    if is_valid:
        print("   [OK] Order valid")
    else:
        print(f"   [--] Order rejected: {error}")

    # Display risk status
    print("\n--- Risk Status ---")
    status = validator.get_risk_status()

    print(f"\nPosition:")
    print(f"  Base: {status['position']['base']:.4f}")
    print(f"  Quote: ${status['position']['quote']:.2f}")
    print(f"  Utilization: {status['position']['utilization_pct']:.1f}%")

    print(f"\nPnL:")
    print(f"  Daily: ${status['pnl']['daily']:.2f}")
    print(f"  Limit Remaining: ${status['pnl']['daily_limit_remaining']:.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Consecutive Losses: {status['consecutive_losses']}")
    print(f"  Circuit Breaker: {'ACTIVE' if status['circuit_breaker_active'] else 'OK'}")

    return validator


def demo_market_conditions(validator):
    """Demonstrate market condition validation."""
    print("\n" + "=" * 60)
    print("MARKET CONDITIONS VALIDATION")
    print("=" * 60)

    # Test normal market
    valid, error = validator.validate_market_conditions(
        spread_bps=10.0,
        book_depth=50000.0,
        price_change_pct=0.005
    )
    print(f"\n1. Normal market (10 bps spread, $50k depth, 0.5% move)")
    if valid:
        print("   [OK] Market conditions acceptable")
    else:
        print(f"   [--] Market rejected: {error}")

    # Test wide spread
    valid, error = validator.validate_market_conditions(
        spread_bps=250.0,
        book_depth=50000.0,
        price_change_pct=0.005
    )
    print(f"\n2. Wide spread (250 bps)")
    if valid:
        print("   [OK] Market conditions acceptable")
    else:
        print(f"   [--] Market rejected: {error}")

    # Test thin book
    valid, error = validator.validate_market_conditions(
        spread_bps=10.0,
        book_depth=5000.0,
        price_change_pct=0.005
    )
    print(f"\n3. Thin book ($5k depth)")
    if valid:
        print("   [OK] Market conditions acceptable")
    else:
        print(f"   [--] Market rejected: {error}")

    # Test high volatility
    valid, error = validator.validate_market_conditions(
        spread_bps=10.0,
        book_depth=50000.0,
        price_change_pct=0.05
    )
    print(f"\n4. High volatility (5% move per second)")
    if valid:
        print("   [OK] Market conditions acceptable")
    else:
        print(f"   [--] Market rejected: {error}")


def demo_circuit_breaker(validator):
    """Demonstrate circuit breaker functionality."""
    print("\n" + "=" * 60)
    print("CIRCUIT BREAKER DEMO")
    print("=" * 60)

    print("\nSimulating large loss to trigger circuit breaker...")

    try:
        # Simulate a large losing trade that triggers circuit breaker
        validator.record_trade_pnl(pnl=-1500.0)
        print("[OK] Loss recorded")
    except Exception as e:
        print(f"[!!] Circuit breaker triggered: {e}")

    # Check status
    status = validator.get_risk_status()
    if status['circuit_breaker_active']:
        print(f"\n[!!] TRADING HALTED")
        print(f"Reason: {status['circuit_breaker_reason']}")
        print("\nCircuit breaker requires manual reset after investigation")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("TRADING SYSTEM CONFIGURATION DEMO")
    print("=" * 60)
    print("\nThis demo shows the portable configuration system.")
    print("It uses config/trading_config.json.example for demonstration.")
    print("\nNote: Example config will show validation warnings (expected).")
    print("=" * 60)

    # Basic usage
    config_manager, risk_limits = demo_basic_usage()
    if config_manager is None:
        print("\n[ERR] Could not load configuration. Exiting.")
        print("\nTo fix:")
        print("1. Copy config/trading_config.json.example to config/trading_config.json")
        print("2. Replace placeholder API keys with real values")
        return

    # Risk validation
    validator = demo_risk_validation(risk_limits)

    # Market conditions
    demo_market_conditions(validator)

    # Circuit breaker
    demo_circuit_breaker(validator)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nSee README.md for configuration and usage.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


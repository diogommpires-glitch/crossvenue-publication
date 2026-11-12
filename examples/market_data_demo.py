"""Example usage of Binance connectors for the CrossVenue market-making simulator.

This script demonstrates:
1. Paper trading with real market data
2. Order book monitoring
3. Market and limit order execution
4. Balance tracking
"""

import time
from src.connectors import PaperTrader, BinanceConnector
from src.utils.types import Side, Price, Quantity


def example_paper_trading():
    """Example: Paper trading with simulated execution."""
    print("=" * 60)
    print("PAPER TRADING EXAMPLE")
    print("=" * 60)

    # Configuration
    config = {
        "initial_balances": {
            "BTC": 0.5,
            "USDT": 30000.0
        },
        "fill_latency_ms": 50,
        "slippage_bps": 5,
        "binance_config": {}
    }

    # Create paper trader
    trader = PaperTrader("BTCUSDT", config)

    try:
        # Connect
        print("\n[1] Connecting to market data...")
        trader.connect()
        time.sleep(2)  # Wait for initial data

        # Get order book
        print("\n[2] Current order book:")
        orderbook = trader.get_orderbook()
        print(f"    Best bid: {orderbook.best_bid}")
        print(f"    Best ask: {orderbook.best_ask}")
        print(f"    Mid price: {orderbook.mid_price}")
        print(f"    Spread: {orderbook.spread}")

        # Check balances
        print("\n[3] Initial balances:")
        print(f"    BTC: {trader.get_balance('BTC'):.6f}")
        print(f"    USDT: {trader.get_balance('USDT'):.2f}")

        # Subscribe to order book updates
        update_count = [0]
        def on_orderbook_update(snapshot):
            update_count[0] += 1
            if update_count[0] % 10 == 0:  # Print every 10th update
                print(f"\n[UPDATE {update_count[0]}] Mid: {snapshot.mid_price:.2f}, "
                      f"Spread: {snapshot.spread:.2f}")

        trader.subscribe_orderbook(on_orderbook_update)

        # Subscribe to fills
        def on_fill(fill):
            print(f"\n[FILL] {fill}")

        trader.subscribe_fills(on_fill)

        # Submit market buy order
        print("\n[4] Submitting market BUY order for 0.01 BTC...")
        response = trader.submit_market_order(
            side=Side.BUY,
            quantity=Quantity(0.01),
            client_order_id="test_market_buy_1"
        )
        print(f"    Response: {response.message}")
        print(f"    Order ID: {response.order_id}")

        # Check balances after market order
        print("\n[5] Balances after market buy:")
        print(f"    BTC: {trader.get_balance('BTC'):.6f}")
        print(f"    USDT: {trader.get_balance('USDT'):.2f}")

        # Submit limit sell order
        time.sleep(1)
        orderbook = trader.get_orderbook()
        limit_price = Price(float(orderbook.mid_price) * 1.02)  # 2% above mid

        print(f"\n[6] Submitting limit SELL order at {limit_price:.2f}...")
        response = trader.submit_limit_order(
            side=Side.SELL,
            price=limit_price,
            quantity=Quantity(0.005),
            client_order_id="test_limit_sell_1"
        )
        print(f"    Response: {response.message}")
        print(f"    Order ID: {response.order_id}")

        limit_order_id = response.order_id

        # Monitor for 10 seconds
        print("\n[7] Monitoring market for 10 seconds...")
        print("    (Limit order will fill if price reaches limit)")
        time.sleep(10)

        # Check limit order status
        order_status = trader.get_order_status(limit_order_id)
        if order_status:
            print(f"\n[8] Limit order status: {order_status.status}")
            print(f"    Filled: {order_status.filled_quantity}/{order_status.quantity}")

            # Cancel if not filled
            if order_status.status.name == "ACTIVE":
                print(f"\n[9] Cancelling limit order {limit_order_id}...")
                cancel_response = trader.cancel_order(limit_order_id)
                print(f"    Response: {cancel_response.message}")

        # Final balances
        print("\n[10] Final balances:")
        print(f"     BTC: {trader.get_balance('BTC'):.6f}")
        print(f"     USDT: {trader.get_balance('USDT'):.2f}")

        # Statistics
        print("\n[11] Trading statistics:")
        for key, value in trader.stats.items():
            print(f"     {key}: {value}")

    finally:
        trader.disconnect()

    print("\n" + "=" * 60)
    print("Paper trading example completed successfully!")
    print("=" * 60)


def example_live_monitoring():
    """Example: Monitor live Binance order book (read-only, no trading)."""
    print("=" * 60)
    print("LIVE ORDER BOOK MONITORING (Read-Only)")
    print("=" * 60)

    # No API credentials needed for market data only
    config = {}

    connector = BinanceConnector("BTCUSDT", config)

    try:
        print("\n[1] Connecting to Binance WebSocket...")
        connector.connect()
        time.sleep(2)

        print("\n[2] Monitoring order book for 30 seconds...")
        print("    Press Ctrl+C to stop early\n")

        update_count = [0]
        start_time = time.time()

        def on_orderbook(snapshot):
            update_count[0] += 1
            elapsed = time.time() - start_time

            print(f"[{elapsed:.1f}s] Update #{update_count[0]}")
            print(f"  Best bid: {snapshot.best_bid[0]:.2f} ({snapshot.best_bid[1]} BTC)")
            print(f"  Best ask: {snapshot.best_ask[0]:.2f} ({snapshot.best_ask[1]} BTC)")
            print(f"  Mid: {snapshot.mid_price:.2f}, Spread: {snapshot.spread:.2f}")
            print(f"  Spread bps: {(snapshot.spread / snapshot.mid_price * 10000):.2f}")
            print()

        connector.subscribe_orderbook(on_orderbook)

        time.sleep(30)

        print(f"\n[3] Received {update_count[0]} order book updates")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        connector.disconnect()

    print("\n" + "=" * 60)
    print("Live monitoring completed!")
    print("=" * 60)


def example_strategy_integration():
    """Example: Integrate with a simple market-making strategy."""
    print("=" * 60)
    print("SIMPLE MARKET MAKER STRATEGY")
    print("=" * 60)

    config = {
        "initial_balances": {
            "BTC": 1.0,
            "USDT": 100000.0
        },
        "slippage_bps": 5
    }

    trader = PaperTrader("BTCUSDT", config)

    # Strategy parameters
    SPREAD_BPS = 20  # Quote 20 bps wide
    ORDER_SIZE = Quantity(0.05)  # 0.05 BTC per side

    active_orders = {}

    def quote_market(snapshot):
        """Simple two-sided quoting strategy."""
        mid = snapshot.mid_price
        if mid is None:
            return

        # Calculate quote prices
        spread_amount = float(mid) * (SPREAD_BPS / 10000.0) / 2.0
        bid_price = Price(float(mid) - spread_amount)
        ask_price = Price(float(mid) + spread_amount)

        # Cancel existing orders
        for order_id in list(active_orders.keys()):
            trader.cancel_order(order_id)
            del active_orders[order_id]

        # Submit new quotes
        buy_response = trader.submit_limit_order(Side.BUY, bid_price, ORDER_SIZE)
        if buy_response.success:
            active_orders[buy_response.order_id] = "BUY"

        sell_response = trader.submit_limit_order(Side.SELL, ask_price, ORDER_SIZE)
        if sell_response.success:
            active_orders[sell_response.order_id] = "SELL"

        print(f"[QUOTE] Bid: {bid_price:.2f}, Ask: {ask_price:.2f}, Mid: {mid:.2f}")

    def on_fill(fill):
        """Handle fills and requote."""
        print(f"[FILL] {fill.side} {fill.quantity} @ {fill.price}")
        if fill.order_id in active_orders:
            del active_orders[fill.order_id]

    try:
        trader.connect()
        time.sleep(2)

        trader.subscribe_fills(on_fill)

        print("\n[1] Starting market maker...")
        print(f"    Spread: {SPREAD_BPS} bps")
        print(f"    Size: {ORDER_SIZE} BTC per side\n")

        # Initial quote
        orderbook = trader.get_orderbook()
        quote_market(orderbook)

        # Requote every 5 seconds
        for i in range(12):  # Run for 60 seconds
            time.sleep(5)
            orderbook = trader.get_orderbook()
            quote_market(orderbook)

            # Print balances
            btc_bal = trader.get_balance('BTC')
            usdt_bal = trader.get_balance('USDT')
            print(f"    Balances: {btc_bal:.6f} BTC, {usdt_bal:.2f} USDT\n")

        print("\n[2] Final statistics:")
        for key, value in trader.stats.items():
            print(f"    {key}: {value}")

    finally:
        # Cancel all orders
        for order_id in active_orders:
            trader.cancel_order(order_id)
        trader.disconnect()

    print("\n" + "=" * 60)
    print("Market maker strategy completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    import sys

    print("\nBinance Connector Examples")
    print("=" * 60)
    print("1. Paper trading demo")
    print("2. Live order book monitoring (read-only)")
    print("3. Simple market maker strategy")
    print("=" * 60)

    choice = input("\nSelect example (1-3, or 'all'): ").strip()

    try:
        if choice == "1":
            example_paper_trading()
        elif choice == "2":
            example_live_monitoring()
        elif choice == "3":
            example_strategy_integration()
        elif choice.lower() == "all":
            example_paper_trading()
            print("\n\n")
            example_live_monitoring()
            print("\n\n")
            example_strategy_integration()
        else:
            print("Invalid choice. Running paper trading example...")
            example_paper_trading()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

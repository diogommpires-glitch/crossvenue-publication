"""Offline Phase 4 demo for equity risk and constrained strategy tuning.

The objective below is synthetic and illustrates the optimizer API; it is not
an empirical Binance performance estimate.
"""

import numpy as np

from src.optimization.bayesian_optimizer import create_default_optimizer
from src.runtime.risk_analytics import RiskAnalytics
from src.strategies.avellaneda_stoikov import AvellanedaStoikov


def demo_risk_analytics():
    analytics = RiskAnalytics(window_size=100, starting_equity=10_000.0,
                              periods_per_year=365.25 * 24 * 60 * 60)
    rng = np.random.default_rng(42)
    equity = 10_000.0
    for index, change in enumerate(rng.normal(0.00001, 0.0005, 100), 1):
        equity *= 1 + change
        analytics.update(pnl=equity - 10_000.0, equity=equity, position=0.02,
                         timestamp=float(index), current_price=50_000.0)
    metrics = analytics.get_risk_metrics()
    print(f"Per-observation volatility: {metrics.volatility:.5%}")
    print(f"One-observation 95% VaR: ${metrics.var_95:.2f}")
    print(f"Max drawdown: ${metrics.max_drawdown:.2f}")


def demo_optimizer():
    optimizer = create_default_optimizer(n_iterations=20, random_state=42)

    def objective(params):
        # Deterministic synthetic objective; it is not a backtest result.
        strategy = AvellanedaStoikov(**params)
        score = 1.2 - 2.0 * (strategy.risk_aversion - 0.15) ** 2
        score -= 20.0 * (strategy.order_size - 0.004) ** 2
        score -= 0.2 * (strategy.volatility - 0.2) ** 2
        drawdown = 0.01 + 0.15 * strategy.order_size
        return score, drawdown, score * 100.0

    result = optimizer.optimize(objective, verbose=True)
    if not result.feasible:
        print("No feasible candidate met the drawdown constraint.")
        return
    print("Best feasible strategy parameters:")
    print(result.best_params)


if __name__ == "__main__":
    demo_risk_analytics()
    demo_optimizer()

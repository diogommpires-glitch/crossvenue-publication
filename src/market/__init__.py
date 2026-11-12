"""Market simulator with self-impact modeling.

This is the CRITICAL DIFFERENTIATOR for CrossVenue.

Key features:
- Synthetic order flow generation (Poisson arrivals)
- Mid-price evolution with impact
- Self-impact tracking (strategy's own trades)
- Adverse selection modeling
"""

from .market_simulator import MarketSimulator
from .self_impact import SelfImpactModel

__all__ = ["MarketSimulator", "SelfImpactModel"]

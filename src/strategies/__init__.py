"""Market-making strategies.

This module implements various market-making approaches:
- Baseline (naive, no self-impact)
- Avellaneda-Stoikov (HJB optimal spreads)
"""

from .base import Strategy
from .naive import NaiveMarketMaker
from .avellaneda_stoikov import AvellanedaStoikov

__all__ = ["Strategy", "NaiveMarketMaker", "AvellanedaStoikov"]

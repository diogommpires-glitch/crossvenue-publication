"""Performance metrics and validation.

This module provides:
- P&L tracking and Sharpe ratio
- Adverse selection measurement
- Square-root law validation (critical for CrossVenue)
- Strategy comparison metrics
"""

from .pnl import PnLTracker
from .validation import SquareRootLawValidator, AdverseSelectionMetrics

__all__ = [
    "PnLTracker",
    "SquareRootLawValidator",
    "AdverseSelectionMetrics",
]

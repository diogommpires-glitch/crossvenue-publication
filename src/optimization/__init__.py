"""Parameter optimization for trading strategies.

This module provides tools for optimizing strategy parameters to maximize
risk-adjusted returns subject to risk constraints.
"""

from .bayesian_optimizer import BayesianOptimizer, OptimizationResult

__all__ = ["BayesianOptimizer", "OptimizationResult"]

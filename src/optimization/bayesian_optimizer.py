"""Constrained Bayesian optimization using an exact RBF Gaussian Process.

The objective returns (Sharpe per evaluation period, maximum drawdown fraction,
total P&L). Drawdown is constrained by absolute magnitude. If no evaluated point
is feasible, `best_params` is None and callers must decide how to proceed.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OptimizationResult:
    best_params: Optional[Dict[str, float]]
    best_score: Optional[float]
    best_sharpe: Optional[float]
    best_max_drawdown: Optional[float]
    all_trials: List[Dict[str, Any]]
    optimization_time: float
    method: str
    feasible: bool = False


@dataclass
class ParameterSpace:
    name: str
    min_value: float
    max_value: float
    log_scale: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("parameter name must be a non-empty string")
        if not math.isfinite(self.min_value) or not math.isfinite(self.max_value):
            raise ValueError("parameter bounds must be finite")
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        if self.log_scale and self.min_value <= 0:
            raise ValueError("log-scaled parameters must have positive bounds")

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        generator = rng if rng is not None else np.random.default_rng()
        if self.log_scale:
            return float(10.0 ** generator.uniform(math.log10(self.min_value), math.log10(self.max_value)))
        return float(generator.uniform(self.min_value, self.max_value))

    def normalize(self, value: float) -> float:
        if not math.isfinite(value) or not self.min_value <= value <= self.max_value:
            raise ValueError(f"{self.name} value is outside its parameter bounds")
        if self.log_scale:
            return float((math.log10(value) - math.log10(self.min_value)) / (math.log10(self.max_value) - math.log10(self.min_value)))
        return float((value - self.min_value) / (self.max_value - self.min_value))

    def denormalize(self, normalized_value: float) -> float:
        if not math.isfinite(normalized_value) or not 0.0 <= normalized_value <= 1.0:
            raise ValueError("normalized_value must be in [0, 1]")
        if self.log_scale:
            return float(10.0 ** (math.log10(self.min_value) + normalized_value * (math.log10(self.max_value) - math.log10(self.min_value))))
        return float(self.min_value + normalized_value * (self.max_value - self.min_value))


class BayesianOptimizer:
    """Optimize Sharpe subject to a maximum drawdown limit."""

    def __init__(
        self,
        parameter_spaces: List[ParameterSpace],
        max_drawdown_threshold: float = 0.05,
        n_initial_points: int = 10,
        n_iterations: int = 50,
        acquisition_function: str = "ei",
        random_state: Optional[int] = None,
        observation_noise: float = 0.05,
    ) -> None:
        if not parameter_spaces:
            raise ValueError("at least one parameter space is required")
        names = [space.name for space in parameter_spaces]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        if not math.isfinite(max_drawdown_threshold) or max_drawdown_threshold < 0:
            raise ValueError("max_drawdown_threshold must be finite and non-negative")
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer evaluation budget")
        if not isinstance(n_initial_points, int) or n_initial_points < 1:
            raise ValueError("n_initial_points must be a positive integer")
        if acquisition_function not in {"ei", "ucb"}:
            raise ValueError("acquisition_function must be 'ei' or 'ucb'")
        if not math.isfinite(observation_noise) or observation_noise <= 0:
            raise ValueError("observation_noise must be finite and positive")

        self.parameter_spaces = list(parameter_spaces)
        self.max_drawdown_threshold = float(max_drawdown_threshold)
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.acquisition_function = acquisition_function
        self.random_state = random_state
        self.observation_noise = float(observation_noise)
        self.rng = np.random.default_rng(random_state)
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.drawdowns_observed: List[float] = []
        self.constraints_satisfied: List[bool] = []
        self.use_bayesian = self._check_scipy_available()

    @staticmethod
    def _check_scipy_available() -> bool:
        try:
            import scipy.linalg  # noqa: F401
            import scipy.optimize  # noqa: F401
            import scipy.stats  # noqa: F401
            return True
        except ImportError:
            return False

    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], Tuple[float, float, float]],
        verbose: bool = True,
    ) -> OptimizationResult:
        """Evaluate at most `n_iterations` candidates and return only feasible best."""
        start = time.perf_counter()
        self.rng = np.random.default_rng(self.random_state)
        self.X_observed = []
        self.y_observed = []
        self.drawdowns_observed = []
        self.constraints_satisfied = []
        all_trials: List[Dict[str, Any]] = []
        initial_count = min(self.n_initial_points, self.n_iterations)
        method = "bayesian_gp" if self.use_bayesian else "random_search"

        for iteration in range(self.n_iterations):
            if iteration < initial_count or not self.use_bayesian or len(self.X_observed) < 2:
                params = self._sample_random_params()
            else:
                params = self._get_next_params_gp()
            values = objective_function(params.copy())
            try:
                valid_length = len(values) == 3
            except TypeError:
                valid_length = False
            if not valid_length:
                raise ValueError("objective_function must return (sharpe, max_drawdown, total_pnl)")
            sharpe, drawdown, total_pnl = (float(value) for value in values)
            if not all(math.isfinite(value) for value in (sharpe, drawdown, total_pnl)):
                raise ValueError("objective_function returned a non-finite metric")
            absolute_drawdown = abs(drawdown)
            feasible = absolute_drawdown <= self.max_drawdown_threshold
            self.X_observed.append(self._normalize_params(params))
            self.y_observed.append(sharpe)
            self.drawdowns_observed.append(absolute_drawdown)
            self.constraints_satisfied.append(feasible)
            all_trials.append({
                "iteration": iteration,
                "params": params.copy(),
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "absolute_drawdown": absolute_drawdown,
                "total_pnl": total_pnl,
                "constraint_satisfied": feasible,
            })

        feasible_trials = [trial for trial in all_trials if trial["constraint_satisfied"]]
        best = max(feasible_trials, key=lambda trial: trial["sharpe"]) if feasible_trials else None
        elapsed = time.perf_counter() - start
        result = OptimizationResult(
            best_params=best["params"].copy() if best else None,
            best_score=best["sharpe"] if best else None,
            best_sharpe=best["sharpe"] if best else None,
            best_max_drawdown=best["max_drawdown"] if best else None,
            all_trials=all_trials,
            optimization_time=elapsed,
            method=method,
            feasible=best is not None,
        )
        if verbose:
            if best:
                print(f"Optimization complete in {elapsed:.2f}s ({method}); best feasible Sharpe={best['sharpe']:.4f}")
            else:
                print(f"Optimization complete in {elapsed:.2f}s ({method}); no candidate met the drawdown limit")
        return result

    def _sample_random_params(self) -> Dict[str, float]:
        return {space.name: space.sample(self.rng) for space in self.parameter_spaces}

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        return np.asarray([space.normalize(float(params[space.name])) for space in self.parameter_spaces], dtype=float)

    def _denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        return {space.name: space.denormalize(float(normalized[i])) for i, space in enumerate(self.parameter_spaces)}

    def _get_next_params_gp(self) -> Dict[str, float]:
        """Select a point using GP expected improvement times feasibility probability."""
        from scipy.optimize import minimize
        from scipy.stats import norm

        X = np.asarray(self.X_observed, dtype=float)
        y = np.asarray(self.y_observed, dtype=float)
        dd = np.asarray(self.drawdowns_observed, dtype=float)
        feasible_values = y[np.asarray(self.constraints_satisfied, dtype=bool)]
        incumbent = float(np.max(feasible_values)) if feasible_values.size else float(np.max(y))

        def acquisition(point: np.ndarray) -> float:
            mean_y, std_y = self._gp_predict(X, y, point, noise=self.observation_noise)
            mean_dd, std_dd = self._gp_predict(X, dd, point, noise=self.observation_noise)
            probability_feasible = float(norm.cdf((self.max_drawdown_threshold - mean_dd) / max(std_dd, 1e-12)))
            if self.acquisition_function == "ucb":
                utility = mean_y + 2.0 * std_y - incumbent
            else:
                improvement = mean_y - incumbent
                z = improvement / max(std_y, 1e-12)
                utility = improvement * norm.cdf(z) + std_y * norm.pdf(z)
            return -max(0.0, float(utility)) * probability_feasible

        starts = [self.rng.uniform(0.0, 1.0, len(self.parameter_spaces)) for _ in range(12)]
        candidates = []
        for start in starts:
            result = minimize(acquisition, start, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(self.parameter_spaces))
            if np.all(np.isfinite(result.x)):
                candidates.append((float(result.fun), np.clip(result.x, 0.0, 1.0)))
        if not candidates:
            return self._sample_random_params()
        return self._denormalize_params(min(candidates, key=lambda item: item[0])[1])

    @staticmethod
    def _gp_predict(
        X: np.ndarray, y: np.ndarray, point: np.ndarray, noise: float = 0.05
    ) -> Tuple[float, float]:
        """Exact RBF GP posterior; noise is observation SD in standardized units."""
        from scipy.linalg import cho_factor, cho_solve, solve_triangular

        y_mean = float(np.mean(y))
        y_scale = float(np.std(y))
        if y_scale < 1e-12:
            # Flat observations do not justify unit-scale posterior uncertainty.
            y_scale = max(abs(y_mean) * 1e-6, 1e-6)
        normalized_y = (y - y_mean) / y_scale
        length_scale = 0.25
        distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        kernel = np.exp(-0.5 * distances / (length_scale * length_scale))
        if not math.isfinite(noise) or noise < 0:
            raise ValueError("noise must be finite and non-negative")
        kernel.flat[:: len(kernel) + 1] += noise * noise + 1e-8
        factor = cho_factor(kernel, lower=True, check_finite=False)
        alpha = cho_solve(factor, normalized_y, check_finite=False)
        cross = np.exp(-0.5 * np.sum((X - point) ** 2, axis=1) / (length_scale * length_scale))
        mean = y_mean + y_scale * float(cross @ alpha)
        lower = np.tril(factor[0])
        projected = solve_triangular(lower, cross, lower=True, check_finite=False)
        variance = max(1e-12, 1.0 - float(projected @ projected))
        return mean, y_scale * math.sqrt(variance)


def create_default_optimizer(
    max_drawdown_threshold: float = 0.05,
    n_iterations: int = 50,
    random_state: Optional[int] = None,
) -> BayesianOptimizer:
    """Create constructor-compatible offline search spaces for Avellaneda-Stoikov."""
    spaces = [
        ParameterSpace("risk_aversion", 0.001, 1.0, log_scale=True),
        ParameterSpace("order_size", 0.001, 0.01, log_scale=True),
        ParameterSpace("volatility", 0.05, 1.0),
    ]
    return BayesianOptimizer(
        parameter_spaces=spaces,
        max_drawdown_threshold=max_drawdown_threshold,
        n_initial_points=min(10, n_iterations),
        n_iterations=n_iterations,
        random_state=random_state,
    )

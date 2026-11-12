import numpy as np
import pytest

from src.optimization.bayesian_optimizer import (
    BayesianOptimizer, OptimizationResult, ParameterSpace, create_default_optimizer,
)
from src.strategies.avellaneda_stoikov import AvellanedaStoikov


def test_parameter_space_bounds_normalization_and_log_scale():
    linear = ParameterSpace("x", 0.0, 10.0)
    assert linear.normalize(5.0) == pytest.approx(0.5)
    assert linear.denormalize(0.5) == pytest.approx(5.0)
    log = ParameterSpace("gamma", 0.01, 1.0, log_scale=True)
    assert log.denormalize(log.normalize(0.1)) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        ParameterSpace("bad", 0.0, 1.0, log_scale=True)


def test_optimizer_respects_budget_and_returns_best_feasible():
    optimizer = BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], n_initial_points=4,
                                  n_iterations=12, random_state=12)
    result = optimizer.optimize(lambda p: (-(p["x"] - 0.4) ** 2, 0.02, p["x"]), verbose=False)
    assert len(result.all_trials) == 12
    assert result.feasible is True
    assert 0.0 <= result.best_params["x"] <= 1.0
    assert abs(result.best_params["x"] - 0.4) < 0.3


def test_optimizer_is_reproducible_and_does_not_change_global_numpy_rng():
    optimizer = BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], n_iterations=8, random_state=9)
    np.random.seed(77)
    before = np.random.get_state()
    first = optimizer.optimize(lambda p: (p["x"], 0.01, p["x"]), verbose=False)
    after = np.random.get_state()
    second = optimizer.optimize(lambda p: (p["x"], 0.01, p["x"]), verbose=False)
    assert np.array_equal(before[1], after[1])
    assert first.best_params == second.best_params
    assert [trial["params"] for trial in first.all_trials] == [trial["params"] for trial in second.all_trials]


def test_infeasible_run_returns_no_candidate():
    optimizer = BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], max_drawdown_threshold=0.01,
                                  n_iterations=5, random_state=3)
    result = optimizer.optimize(lambda p: (p["x"], 0.1, 5.0), verbose=False)
    assert result.feasible is False
    assert result.best_params is None
    assert result.best_sharpe is None


def test_optimizer_accepts_numpy_metric_vector():
    optimizer = BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], n_iterations=2, random_state=1)
    result = optimizer.optimize(lambda p: np.array([1.0, 0.0, 2.0]), verbose=False)
    assert result.feasible


def test_random_search_fallback_uses_honest_label():
    optimizer = BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], n_iterations=3, random_state=1)
    optimizer.use_bayesian = False
    result = optimizer.optimize(lambda p: (1.0, 0.0, 1.0), verbose=False)
    assert result.method == "random_search"
    assert len(result.all_trials) == 3


def test_default_spaces_match_strategy_constructor_and_paper_order_limit():
    optimizer = create_default_optimizer(n_iterations=4, random_state=1)
    names = {space.name for space in optimizer.parameter_spaces}
    assert names == {"risk_aversion", "order_size", "volatility"}
    params = {space.name: space.sample(np.random.default_rng(4)) for space in optimizer.parameter_spaces}
    assert params["order_size"] <= 0.01
    assert isinstance(AvellanedaStoikov(**params), AvellanedaStoikov)


def test_invalid_budget_threshold_acquisition_and_objective_are_rejected():
    with pytest.raises(ValueError):
        BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], n_iterations=0)
    with pytest.raises(ValueError):
        BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], max_drawdown_threshold=-1.0)
    with pytest.raises(ValueError):
        BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], acquisition_function="nope")
    with pytest.raises(ValueError, match="observation_noise"):
        BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], observation_noise=0.0)
    optimizer = BayesianOptimizer([ParameterSpace("x", 0.0, 1.0)], n_iterations=1)
    with pytest.raises(ValueError, match="non-finite"):
        optimizer.optimize(lambda p: (float("nan"), 0.0, 0.0), verbose=False)


def test_result_defaults_to_infeasible_when_manually_constructed():
    result = OptimizationResult({}, None, None, None, [], 0.0, "manual")
    assert result.feasible is False


def test_gp_observation_noise_increases_uncertainty_near_observations():
    x = np.array([[0.1], [0.5], [0.9]])
    y = np.array([0.0, 1.0, 0.0])
    _, low_noise_std = BayesianOptimizer._gp_predict(x, y, np.array([0.5]), noise=1e-4)
    _, higher_noise_std = BayesianOptimizer._gp_predict(x, y, np.array([0.5]), noise=0.2)
    assert higher_noise_std > low_noise_std


def test_gp_constant_outputs_do_not_create_unit_scale_uncertainty():
    x = np.array([[0.1], [0.5], [0.9]])
    y = np.full(3, 0.04)
    mean, std = BayesianOptimizer._gp_predict(x, y, np.array([0.5]), noise=0.2)
    assert mean == pytest.approx(0.04)
    assert std < 1e-4

import numpy as np

from src.diagnostics import (
    estimate_piecewise_rates,
    ks_exp_test,
    time_rescale_piecewise_poisson,
    time_rescale_poisson,
)


def test_time_rescale_poisson_regular_example():
    events = np.array([1.0, 2.0, 3.0, 4.0])
    residuals = time_rescale_poisson(events, rate=1.0)
    assert np.allclose(residuals, np.ones(4))


def test_ks_exp_test_on_exponential_sample():
    rng = np.random.default_rng(123)
    residuals = rng.exponential(scale=1.0, size=500)
    result = ks_exp_test(residuals)
    assert result["n"] == 500
    assert result["p_value"] > 0.01


def test_estimate_piecewise_rates_counts_and_lengths():
    events = np.array([1.0, 2.0, 12.0, 29.0])
    bin_edges, bin_rates = estimate_piecewise_rates(
        events, horizon=30.0, bin_seconds=10.0
    )

    assert np.allclose(bin_edges, [0.0, 10.0, 20.0, 30.0])
    assert len(bin_rates) == len(bin_edges) - 1
    assert np.all(bin_rates >= 0.0)
    assert np.allclose(bin_rates, [0.2, 0.1, 0.1])


def test_time_rescale_piecewise_poisson_one_bin():
    events = np.array([2.0, 5.0])
    bin_edges = np.array([0.0, 10.0])
    bin_rates = np.array([0.5])

    residuals = time_rescale_piecewise_poisson(events, bin_edges, bin_rates)

    assert np.allclose(residuals, [1.0, 1.5])


def test_time_rescale_piecewise_poisson_multiple_bins():
    events = np.array([5.0, 15.0, 25.0])
    bin_edges = np.array([0.0, 10.0, 20.0, 30.0])
    bin_rates = np.array([0.1, 0.2, 0.3])

    residuals = time_rescale_piecewise_poisson(events, bin_edges, bin_rates)

    assert np.allclose(residuals, [0.5, 1.5, 2.5])


def test_estimate_piecewise_rates_applies_epsilon_floor():
    events = np.array([1.0])
    bin_edges, bin_rates = estimate_piecewise_rates(
        events, horizon=20.0, bin_seconds=10.0, epsilon=1e-6
    )

    assert len(bin_rates) == len(bin_edges) - 1
    assert np.all(bin_rates >= 1e-6)
    assert bin_rates[1] == 1e-6

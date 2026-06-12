import numpy as np

from src.diagnostics import ks_exp_test, time_rescale_poisson


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

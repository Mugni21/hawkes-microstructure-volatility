import numpy as np

from src.hawkes import cap_bivariate_events, fit_hawkes_bivariate, fit_poisson_baseline


def test_cap_mode_uniform_returns_at_most_n_events_per_side():
    buy = np.arange(10, dtype=float)
    sell = np.arange(20, dtype=float)

    buy_fit, sell_fit = cap_bivariate_events(
        buy, sell, max_events_per_side=4, mode="uniform"
    )

    assert len(buy_fit) == 4
    assert len(sell_fit) == 4
    assert np.allclose(buy_fit, [0.0, 3.0, 6.0, 9.0])
    assert np.allclose(sell_fit, [0.0, 6.0, 12.0, 19.0])


def test_hawkes_and_poisson_counts_match_when_cap_is_active():
    buy = np.linspace(1.0, 100.0, 20)
    sell = np.linspace(2.0, 100.0, 24)
    buy_fit, sell_fit = cap_bivariate_events(
        buy, sell, max_events_per_side=5, mode="uniform"
    )

    hawkes = fit_hawkes_bivariate(
        buy_fit,
        sell_fit,
        horizon=120.0,
        max_events_per_side=None,
        optimizer_maxiter=1,
    )
    poisson = fit_poisson_baseline(buy_fit, sell_fit, horizon=120.0)

    assert hawkes.n_events == poisson["n_events"]
    assert hawkes.n_events == len(buy_fit) + len(sell_fit)
    assert hawkes.horizon == poisson["horizon"]

"""Bivariate exponential-kernel Hawkes estimation for buy/sell arrivals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesFitResult:
    """Container for fitted bivariate Hawkes parameters and fit statistics."""

    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    branching_matrix: np.ndarray
    spectral_radius: float
    log_likelihood: float
    aic: float
    bic: float
    horizon: float
    n_events: int
    success: bool
    message: str
    shared_beta: bool = True

    def to_dict(self) -> dict:
        """Return a JSON-friendly dictionary."""
        return {
            "mu": self.mu.tolist(),
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "branching_matrix": self.branching_matrix.tolist(),
            "spectral_radius": float(self.spectral_radius),
            "log_likelihood": float(self.log_likelihood),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "horizon": float(self.horizon),
            "n_events": int(self.n_events),
            "success": bool(self.success),
            "message": self.message,
            "shared_beta": bool(self.shared_beta),
        }


def _prepare_events(events: list[np.ndarray], max_events_per_side: int | None = None) -> list[np.ndarray]:
    prepared = []
    for stream in events:
        arr = np.sort(np.asarray(stream, dtype=float))
        arr = arr[np.isfinite(arr)]
        if max_events_per_side is not None and len(arr) > max_events_per_side:
            arr = arr[:max_events_per_side]
        prepared.append(arr)
    if len(prepared) != 2:
        raise ValueError("This MVP expects exactly two streams: buy and sell")
    return prepared


def branching_matrix(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute G_ij = alpha_ij / beta_ij."""
    return np.asarray(alpha, dtype=float) / np.asarray(beta, dtype=float)


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the largest absolute eigenvalue."""
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _unpack_params(theta: np.ndarray, shared_beta: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.exp(theta)
    mu = params[:2]
    alpha = params[2:6].reshape(2, 2)
    if shared_beta:
        beta = np.full((2, 2), params[6])
    else:
        beta = params[6:10].reshape(2, 2)
    return mu, alpha, beta


def hawkes_log_likelihood(events: list[np.ndarray], horizon: float, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> float:
    """Evaluate the bivariate Hawkes log-likelihood."""
    streams = _prepare_events(events)
    log_sum = 0.0
    for i, target_events in enumerate(streams):
        for t in target_events:
            intensity = float(mu[i])
            for j, source_events in enumerate(streams):
                past = source_events[source_events < t]
                if past.size:
                    intensity += float(np.sum(alpha[i, j] * np.exp(-beta[i, j] * (t - past))))
            if intensity <= 0 or not np.isfinite(intensity):
                return -np.inf
            log_sum += np.log(intensity)

    compensator = float(np.sum(mu) * horizon)
    for i in range(2):
        for j, source_events in enumerate(streams):
            if source_events.size:
                compensator += float(
                    np.sum(alpha[i, j] / beta[i, j] * (1.0 - np.exp(-beta[i, j] * (horizon - source_events))))
                )
    return log_sum - compensator


def fit_hawkes_bivariate(
    buy_events: np.ndarray,
    sell_events: np.ndarray,
    horizon: float | None = None,
    shared_beta: bool = True,
    max_events_per_side: int | None = 5000,
    optimizer_maxiter: int = 500,
    stability_penalty: float = 1_000_000.0,
) -> HawkesFitResult:
    """Fit a bivariate Hawkes model by maximum likelihood.

    The default shared-beta parameterization is a practical MVP for daily or
    rolling-window fits. It keeps the optimization stable while preserving the
    buy/sell excitation matrix needed for the research questions.
    """
    events = _prepare_events([buy_events, sell_events], max_events_per_side=max_events_per_side)
    if horizon is None:
        max_event = max((arr.max() for arr in events if len(arr)), default=0.0)
        horizon = max(1.0, float(max_event))
    counts = np.array([len(events[0]), len(events[1])], dtype=float)
    n_events = int(counts.sum())
    if n_events == 0:
        raise ValueError("At least one event is required to fit Hawkes model")

    base_rates = np.maximum(counts / horizon, 1e-8)
    alpha0 = np.full((2, 2), max(float(np.mean(base_rates)) * 0.1, 1e-6))
    beta0 = 1.0
    if shared_beta:
        x0 = np.log(np.r_[base_rates, alpha0.ravel(), beta0])
        k = 7
    else:
        x0 = np.log(np.r_[base_rates, alpha0.ravel(), np.full(4, beta0)])
        k = 10

    def objective(theta: np.ndarray) -> float:
        mu, alpha, beta = _unpack_params(theta, shared_beta)
        g = branching_matrix(alpha, beta)
        rho = spectral_radius(g)
        ll = hawkes_log_likelihood(events, float(horizon), mu, alpha, beta)
        if not np.isfinite(ll):
            return stability_penalty
        penalty = stability_penalty * max(0.0, rho - 0.999) ** 2
        return -ll + penalty

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": optimizer_maxiter})
    mu, alpha, beta = _unpack_params(result.x, shared_beta)
    ll = hawkes_log_likelihood(events, float(horizon), mu, alpha, beta)
    g = branching_matrix(alpha, beta)
    rho = spectral_radius(g)
    aic = 2 * k - 2 * ll
    bic = k * np.log(max(n_events, 1)) - 2 * ll
    return HawkesFitResult(
        mu=mu,
        alpha=alpha,
        beta=beta,
        branching_matrix=g,
        spectral_radius=rho,
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        horizon=float(horizon),
        n_events=n_events,
        success=bool(result.success),
        message=str(result.message),
        shared_beta=shared_beta,
    )


def fit_poisson_baseline(buy_events: np.ndarray, sell_events: np.ndarray, horizon: float | None = None) -> dict:
    """Fit independent homogeneous Poisson rates for buy and sell arrivals."""
    events = _prepare_events([buy_events, sell_events])
    if horizon is None:
        horizon = max(1.0, max((arr.max() for arr in events if len(arr)), default=0.0))
    counts = np.array([len(events[0]), len(events[1])], dtype=float)
    rates = counts / float(horizon)
    ll = float(np.sum(counts * np.log(np.maximum(rates, 1e-12)) - rates * float(horizon)))
    k = 2
    n_events = int(counts.sum())
    return {
        "rates": rates,
        "log_likelihood": ll,
        "aic": 2 * k - 2 * ll,
        "bic": k * np.log(max(n_events, 1)) - 2 * ll,
        "horizon": float(horizon),
        "n_events": n_events,
    }


def hawkes_intensity_at_times(
    times: np.ndarray,
    events: list[np.ndarray],
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Compute buy/sell Hawkes intensities at requested times."""
    streams = _prepare_events(events)
    grid = np.asarray(times, dtype=float)
    intensities = np.zeros((len(grid), 2), dtype=float)
    for n, t in enumerate(grid):
        for i in range(2):
            value = float(mu[i])
            for j, source_events in enumerate(streams):
                past = source_events[source_events < t]
                if past.size:
                    value += float(np.sum(alpha[i, j] * np.exp(-beta[i, j] * (t - past))))
            intensities[n, i] = value
    return intensities

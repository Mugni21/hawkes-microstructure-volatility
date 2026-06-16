"""Goodness-of-fit diagnostics using Ogata time rescaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _integrated_hawkes_interval(
    target: int,
    start: float,
    end: float,
    events: list[np.ndarray],
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    if end <= start:
        return 0.0
    value = float(mu[target] * (end - start))
    for j, source_events in enumerate(events):
        source = np.asarray(source_events, dtype=float)
        relevant = source[source < end]
        if relevant.size:
            lower = np.maximum(start, relevant)
            value += float(
                np.sum(
                    alpha[target, j]
                    / beta[target, j]
                    * (
                        np.exp(-beta[target, j] * (lower - relevant))
                        - np.exp(-beta[target, j] * (end - relevant))
                    )
                )
            )
    return value


def time_rescale_hawkes(
    target_events: np.ndarray,
    target_index: int,
    all_events: list[np.ndarray],
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    start_time: float = 0.0,
) -> np.ndarray:
    """Return Exp(1) residual inter-arrivals for one Hawkes event stream."""
    target = np.sort(np.asarray(target_events, dtype=float))
    events = [np.sort(np.asarray(stream, dtype=float)) for stream in all_events]
    residuals = []
    previous = float(start_time)
    for event_time in target:
        residuals.append(
            _integrated_hawkes_interval(
                target_index, previous, float(event_time), events, mu, alpha, beta
            )
        )
        previous = float(event_time)
    return np.asarray(residuals, dtype=float)


def time_rescale_poisson(
    target_events: np.ndarray, rate: float, start_time: float = 0.0
) -> np.ndarray:
    """Return Exp(1) residual inter-arrivals for a homogeneous Poisson process."""
    target = np.sort(np.asarray(target_events, dtype=float))
    if rate < 0:
        raise ValueError("rate must be non-negative")
    previous = float(start_time)
    residuals = []
    for event_time in target:
        residuals.append(float(rate) * (float(event_time) - previous))
        previous = float(event_time)
    return np.asarray(residuals, dtype=float)


def estimate_piecewise_rates(
    event_times: np.ndarray,
    horizon: float,
    bin_seconds: float,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate piecewise-constant Poisson rates over [0, horizon]."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    events = np.sort(np.asarray(event_times, dtype=float))
    events = events[np.isfinite(events)]
    events = events[(events >= 0.0) & (events <= horizon)]

    bin_edges = np.arange(0.0, float(horizon), float(bin_seconds))
    if bin_edges.size == 0 or bin_edges[0] != 0.0:
        bin_edges = np.r_[0.0, bin_edges]
    if bin_edges[-1] < horizon:
        bin_edges = np.r_[bin_edges, float(horizon)]
    elif bin_edges[-1] > horizon:
        bin_edges[-1] = float(horizon)
    if bin_edges.size < 2:
        bin_edges = np.array([0.0, float(horizon)])

    counts, _ = np.histogram(events, bins=bin_edges)
    widths = np.diff(bin_edges)
    rates = counts / widths
    rates = np.maximum(rates.astype(float), float(epsilon))
    return bin_edges.astype(float), rates


def _integrate_piecewise_constant(
    start: float,
    end: float,
    bin_edges: np.ndarray,
    bin_rates: np.ndarray,
) -> float:
    if end <= start:
        return 0.0
    if start < bin_edges[0] or end > bin_edges[-1]:
        raise ValueError("integration interval must lie within bin_edges")

    total = 0.0
    current = float(start)
    while current < end:
        bin_index = np.searchsorted(bin_edges, current, side="right") - 1
        bin_index = min(max(bin_index, 0), len(bin_rates) - 1)
        segment_end = min(float(end), float(bin_edges[bin_index + 1]))
        total += float(bin_rates[bin_index]) * (segment_end - current)
        current = segment_end
    return total


def time_rescale_piecewise_poisson(
    event_times: np.ndarray,
    bin_edges: np.ndarray,
    bin_rates: np.ndarray,
) -> np.ndarray:
    """Return residuals under a piecewise-constant Poisson intensity."""
    events = np.sort(np.asarray(event_times, dtype=float))
    events = events[np.isfinite(events)]
    edges = np.asarray(bin_edges, dtype=float)
    rates = np.asarray(bin_rates, dtype=float)

    if edges.ndim != 1 or rates.ndim != 1:
        raise ValueError("bin_edges and bin_rates must be one-dimensional")
    if len(edges) != len(rates) + 1:
        raise ValueError("len(bin_edges) must equal len(bin_rates) + 1")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    if np.any(rates < 0):
        raise ValueError("bin_rates must be nonnegative")
    if events.size and (events[0] < edges[0] or events[-1] > edges[-1]):
        raise ValueError("event_times must lie within bin_edges")

    previous = float(edges[0])
    residuals = []
    for event_time in events:
        residuals.append(
            _integrate_piecewise_constant(previous, float(event_time), edges, rates)
        )
        previous = float(event_time)
    return np.asarray(residuals, dtype=float)


def ks_exp_test(residuals: np.ndarray) -> dict:
    """Run a one-sample KS test against Exp(1)."""
    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"ks_statistic": np.nan, "p_value": np.nan, "n": 0}
    stat, p_value = stats.kstest(clean, "expon", args=(0, 1))
    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "n": int(clean.size),
    }


def residual_acf(residuals: np.ndarray, max_lag: int = 10) -> pd.Series:
    """Compute simple residual autocorrelations."""
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return pd.Series(dtype=float)
    out = {}
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    for lag in range(1, max_lag + 1):
        if lag >= len(x) or denom == 0:
            out[lag] = np.nan
        else:
            out[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denom)
    return pd.Series(out, name="acf")


def diagnostic_summary_table(
    named_residuals: dict[str, np.ndarray], max_lag: int = 5
) -> pd.DataFrame:
    """Summarize KS and residual autocorrelation diagnostics."""
    rows = []
    for name, residuals in named_residuals.items():
        ks = ks_exp_test(residuals)
        acf = residual_acf(residuals, max_lag=max_lag)
        row = {"model_stream": name, **ks}
        for lag, value in acf.items():
            row[f"acf_lag_{lag}"] = value
        rows.append(row)
    return pd.DataFrame(rows)

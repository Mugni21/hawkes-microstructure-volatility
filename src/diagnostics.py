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

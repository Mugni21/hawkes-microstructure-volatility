"""Reusable matplotlib plots for the research workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def interarrival_histogram(event_times: np.ndarray, ax=None, bins: int = 50):
    """Plot inter-arrival times with an exponential density using the sample rate."""
    ax = ax or plt.subplots()[1]
    intervals = np.diff(np.sort(np.asarray(event_times, dtype=float)))
    intervals = intervals[np.isfinite(intervals) & (intervals >= 0)]
    ax.hist(intervals, bins=bins, density=True, alpha=0.6, label="observed")
    if intervals.size and intervals.mean() > 0:
        x = np.linspace(0, np.quantile(intervals, 0.99), 200)
        ax.plot(x, stats.expon(scale=intervals.mean()).pdf(x), label="Exp sample mean")
    ax.set_xlabel("Inter-arrival time (seconds)")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def event_counts_over_time(table: pd.DataFrame, ax=None):
    """Plot buy and sell event counts over fixed intervals."""
    ax = ax or plt.subplots()[1]
    ax.plot(table["interval_start"], table["buy_count"], label="buy")
    ax.plot(table["interval_start"], table["sell_count"], label="sell")
    ax.set_xlabel("Seconds from day start")
    ax.set_ylabel("Event count")
    ax.legend()
    return ax


def imbalance_over_time(table: pd.DataFrame, ax=None):
    """Plot signed order-flow imbalance over time."""
    ax = ax or plt.subplots()[1]
    ax.plot(table["interval_start"], table["order_flow_imbalance"], color="black")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Seconds from day start")
    ax.set_ylabel("Order-flow imbalance")
    return ax


def matrix_heatmap(matrix: np.ndarray, title: str, ax=None, labels=("buy", "sell")):
    """Plot a 2x2 excitation or branching matrix heatmap."""
    ax = ax or plt.subplots()[1]
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Source stream")
    ax.set_ylabel("Target stream")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3g}", ha="center", va="center", color="white")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def excitation_matrix_heatmap(alpha: np.ndarray, ax=None):
    """Plot Hawkes alpha matrix."""
    return matrix_heatmap(alpha, "Excitation matrix alpha", ax=ax)


def branching_matrix_heatmap(branching: np.ndarray, ax=None):
    """Plot Hawkes branching matrix."""
    return matrix_heatmap(branching, "Branching matrix G = alpha / beta", ax=ax)


def spectral_radius_over_windows(results: pd.DataFrame, ax=None):
    """Plot spectral radius across days or rolling windows."""
    ax = ax or plt.subplots()[1]
    x = results["window_start"] if "window_start" in results else np.arange(len(results))
    ax.plot(x, results["spectral_radius"], marker="o")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="stability boundary")
    ax.set_ylabel("Spectral radius")
    ax.legend()
    return ax


def qq_plot_exp(residuals: np.ndarray, ax=None, title: str = "Time-rescaled residual QQ plot"):
    """Plot residual quantiles against Exp(1) quantiles."""
    ax = ax or plt.subplots()[1]
    clean = np.sort(np.asarray(residuals, dtype=float))
    clean = clean[np.isfinite(clean)]
    if clean.size:
        probs = (np.arange(1, clean.size + 1) - 0.5) / clean.size
        theoretical = stats.expon.ppf(probs)
        ax.scatter(theoretical, clean, s=12, alpha=0.7)
        limit = max(theoretical.max(), clean.max())
        ax.plot([0, limit], [0, limit], color="red", linewidth=1)
    ax.set_xlabel("Theoretical Exp(1) quantile")
    ax.set_ylabel("Observed residual quantile")
    ax.set_title(title)
    return ax


def forecast_comparison(metrics: pd.DataFrame, metric: str = "r2_oos", ax=None):
    """Bar plot comparing forecasting models."""
    ax = ax or plt.subplots()[1]
    ax.bar(metrics["model"], metrics[metric])
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    return ax


def coefficient_plot(model, feature_names: list[str], ax=None):
    """Plot linear model coefficients from a sklearn pipeline."""
    ax = ax or plt.subplots()[1]
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    coefs = getattr(estimator, "coef_", None)
    if coefs is None:
        raise ValueError("Model does not expose coef_")
    values = np.ravel(coefs)
    ax.barh(feature_names, values)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Coefficient")
    return ax

"""Visualize one Hawkes fitting window."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import filter_intraday_window, load_processed
from src.features import build_feature_table, buy_sell_event_times
from src.plotting import (
    branching_matrix_heatmap,
    event_counts_over_time,
    excitation_matrix_heatmap,
    imbalance_over_time,
    interarrival_histogram,
)


PROCESSED = "data/processed/BTCUSDT_2024-01-02.parquet"
FIT_JSON = "reports/hawkes_fit.json"
FIGURE_DIR = Path("reports/figures")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fit = json.loads(Path(FIT_JSON).read_text())
    window = fit["window"]

    df = filter_intraday_window(
        load_processed(PROCESSED),
        start_hour=window["start_hour"],
        duration_minutes=window["duration_minutes"],
        reset_event_time=True,
    )

    table = build_feature_table(
        df,
        interval_seconds=10,
        horizons=(10, 30, 60, 300),
        rolling_window_seconds=300,
    )

    buy, sell = buy_sell_event_times(df)

    # 1. Buy/sell counts over time
    fig, ax = plt.subplots(figsize=(10, 4))
    event_counts_over_time(table, ax=ax)
    ax.set_title("BTCUSDT buy/sell aggregate trade counts, 05:00–05:30 UTC")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "BTCUSDT_2024-01-02_0500_0530_event_counts.png", dpi=150)
    plt.close(fig)

    # 2. Order-flow imbalance over time
    fig, ax = plt.subplots(figsize=(10, 4))
    imbalance_over_time(table, ax=ax)
    ax.set_title("BTCUSDT signed order-flow imbalance, 05:00–05:30 UTC")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "BTCUSDT_2024-01-02_0500_0530_imbalance.png", dpi=150)
    plt.close(fig)

    # 3. Buy interarrival histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    interarrival_histogram(buy, ax=ax, bins=80)
    ax.set_title("BTCUSDT buy-event interarrival times")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "BTCUSDT_2024-01-02_0500_0530_buy_interarrivals.png", dpi=150)
    plt.close(fig)

    # 4. Sell interarrival histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    interarrival_histogram(sell, ax=ax, bins=80)
    ax.set_title("BTCUSDT sell-event interarrival times")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "BTCUSDT_2024-01-02_0500_0530_sell_interarrivals.png", dpi=150)
    plt.close(fig)

    # 5. Excitation matrix heatmap
    alpha = pd.DataFrame(fit["hawkes"]["alpha"]).to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    excitation_matrix_heatmap(alpha, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "BTCUSDT_2024-01-02_0500_0530_alpha_heatmap.png", dpi=150)
    plt.close(fig)

    # 6. Branching matrix heatmap
    branching = pd.DataFrame(fit["hawkes"]["branching_matrix"]).to_numpy()
    spectral_radius = fit["hawkes"]["spectral_radius"]
    fig, ax = plt.subplots(figsize=(5, 4))
    branching_matrix_heatmap(branching, ax=ax)
    ax.set_title(f"Branching matrix, spectral radius={spectral_radius:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "BTCUSDT_2024-01-02_0500_0530_branching_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
"""05 Simplified execution simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data import filter_intraday_window, load_processed
from src.execution import (
    build_execution_table,
    hawkes_aware_schedule,
    hawkes_momentum_schedule,
    imbalance_aware_schedule,
    implementation_shortfall,
    twap_schedule,
    volume_participation_schedule,
)
from src.features import build_feature_table, buy_sell_event_times
from src.forecasting import add_hawkes_intensity_features
from src.hawkes import hawkes_intensity_at_times


def _load_fit_json(fit_path: Path) -> dict | None:
    if not fit_path.exists():
        return None
    return json.loads(fit_path.read_text())


def _window_settings(
    args: argparse.Namespace, config: dict, fit: dict | None = None
) -> tuple[float, float]:
    fit_window = (fit or {}).get("window", {})
    config_window = config.get("hawkes", {}).get("fit_window", {})
    start_hour = (
        args.start_hour
        if args.start_hour is not None
        else fit_window.get("start_hour", config_window.get("start_hour", 0.0))
    )
    duration_minutes = (
        args.duration_minutes
        if args.duration_minutes is not None
        else fit_window.get("duration_minutes", config_window.get("duration_minutes"))
    )
    if duration_minutes is None:
        raise ValueError(
            "duration_minutes is required via CLI, fit JSON window, or config.yaml"
        )
    return float(start_hour), float(duration_minutes)


def _add_hawkes_intensities_if_available(feature_table, trades, fit: dict | None):
    if fit is None:
        return feature_table

    hawkes = fit.get("hawkes")
    if hawkes is None:
        return feature_table

    buy, sell = buy_sell_event_times(trades)
    intensities = hawkes_intensity_at_times(
        feature_table["interval_start"].to_numpy(float),
        [buy, sell],
        np.asarray(hawkes["mu"], dtype=float),
        np.asarray(hawkes["alpha"], dtype=float),
        np.asarray(hawkes["beta"], dtype=float),
    )
    return add_hawkes_intensity_features(
        feature_table, intensities[:, 0], intensities[:, 1]
    )


def _strategy_results(
    execution_table, total_quantity: float, side: str, strength: float
):
    n_intervals = len(execution_table)
    schedules = {
        "twap": twap_schedule(n_intervals, total_quantity),
        "volume_participation": volume_participation_schedule(
            execution_table["traded_volume"], total_quantity
        ),
        "imbalance_aware": imbalance_aware_schedule(
            execution_table["order_flow_imbalance"],
            total_quantity,
            side=side,
            strength=strength,
        ),
    }
    if {"hawkes_lambda_buy", "hawkes_lambda_sell"}.issubset(execution_table.columns):
        schedules["hawkes_contrarian"] = hawkes_aware_schedule(
            execution_table["hawkes_lambda_buy"],
            execution_table["hawkes_lambda_sell"],
            total_quantity,
            side=side,
            strength=strength,
        )
        schedules["hawkes_momentum"] = hawkes_momentum_schedule(
            execution_table["hawkes_lambda_buy"],
            execution_table["hawkes_lambda_sell"],
            total_quantity,
            side=side,
            strength=strength,
        )

    rows = []
    prices = execution_table["mid_or_trade_price"].to_numpy(float)
    for strategy, child_qty in schedules.items():
        average_price = (
            float(np.sum(prices * child_qty) / child_qty.sum())
            if child_qty.sum() > 0
            else np.nan
        )
        rows.append(
            {
                "strategy": strategy,
                "side": side,
                "total_quantity": total_quantity,
                "average_execution_price": average_price,
                "arrival_price": float(prices[0]),
                "implementation_shortfall": implementation_shortfall(
                    prices, child_qty, side
                ),
            }
        )
    return rows, schedules


def _build_schedule_output(execution_table, schedules) -> pd.DataFrame:
    schedule_table = execution_table.copy()
    for strategy, child_qty in schedules.items():
        schedule_table[f"{strategy}_child_qty"] = child_qty
    return schedule_table


def _save_plots(
    execution_table, schedules, results, total_quantity: float, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    x = execution_table["interval_start"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, child_qty in schedules.items():
        remaining = total_quantity - np.cumsum(child_qty)
        ax.plot(x, remaining, label=strategy)
    ax.set_xlabel("Seconds from execution window start")
    ax.set_ylabel("Inventory remaining")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "execution_inventory_remaining.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy, child_qty in schedules.items():
        ax.plot(x, child_qty, label=strategy)
    ax.set_xlabel("Seconds from execution window start")
    ax.set_ylabel("Child order quantity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "execution_child_order_sizes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results["strategy"], results["implementation_shortfall"])
    ax.set_ylabel("Implementation shortfall")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "execution_implementation_shortfall.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fit", default="reports/hawkes_fit.json")
    parser.add_argument("--output", default="reports/execution_results.csv")
    parser.add_argument("--schedule-output", default="reports/execution_schedule.csv")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--total-quantity", type=float, default=1.0)
    parser.add_argument("--side", choices=["buy", "sell"], default="buy")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--start-hour", type=float)
    parser.add_argument("--duration-minutes", type=float)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    fit = _load_fit_json(Path(args.fit))
    start_hour, duration_minutes = _window_settings(args, config, fit=fit)
    trades = filter_intraday_window(
        load_processed(args.processed),
        start_hour=start_hour,
        duration_minutes=duration_minutes,
        reset_event_time=True,
    )
    if trades.empty:
        raise SystemExit("No trades found in the requested execution window.")

    feature_table = build_feature_table(
        trades,
        interval_seconds=args.interval_seconds,
        horizons=tuple(config["forecasting"]["horizons_seconds"]),
        rolling_window_seconds=config["forecasting"]["rolling_window_seconds"],
    )
    feature_table = _add_hawkes_intensities_if_available(feature_table, trades, fit)
    execution_table = build_execution_table(
        feature_table, total_quantity=args.total_quantity, side=args.side
    )
    rows, schedules = _strategy_results(
        execution_table,
        total_quantity=args.total_quantity,
        side=args.side,
        strength=args.strength,
    )

    results = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    schedule_output = _build_schedule_output(execution_table, schedules)
    Path(args.schedule_output).parent.mkdir(parents=True, exist_ok=True)
    schedule_output.to_csv(args.schedule_output, index=False)
    _save_plots(
        execution_table,
        schedules,
        results,
        total_quantity=args.total_quantity,
        output_dir=Path(args.figures_dir),
    )
    print(results)


if __name__ == "__main__":
    main()

"""04 Forecast short-horizon realized volatility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data import filter_intraday_window, load_processed
from src.features import build_feature_table, buy_sell_event_times
from src.forecasting import (
    add_hawkes_intensity_features,
    run_classification_experiment,
    run_regression_experiment,
)
from src.hawkes import fit_hawkes_bivariate, hawkes_intensity_at_times


def _window_settings(
    args: argparse.Namespace, config: dict
) -> tuple[float, float | None]:
    window = config.get("hawkes", {}).get("fit_window", {})
    start_hour = (
        args.start_hour
        if args.start_hour is not None
        else window.get("start_hour", 0.0)
    )
    duration_minutes = (
        args.duration_minutes
        if args.duration_minutes is not None
        else window.get("duration_minutes")
    )
    return float(start_hour), (
        None if duration_minutes is None else float(duration_minutes)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output-prefix", default="reports/forecast")
    parser.add_argument(
        "--start-hour", type=float, help="UTC hour to start forecasting"
    )
    parser.add_argument(
        "--duration-minutes", type=float, help="Minutes to include after start-hour"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    start_hour, duration_minutes = _window_settings(args, config)
    df = filter_intraday_window(
        load_processed(args.processed),
        start_hour=start_hour,
        duration_minutes=duration_minutes,
        reset_event_time=True,
    )
    if df.empty:
        raise SystemExit("No trades found in the requested intraday window.")

    table = build_feature_table(
        df,
        interval_seconds=config["data"]["interval_seconds"],
        horizons=tuple(config["forecasting"]["horizons_seconds"]),
        rolling_window_seconds=config["forecasting"]["rolling_window_seconds"],
    )

    buy, sell = buy_sell_event_times(df)
    train_fraction = config["forecasting"]["train_fraction"]
    split_index = int(len(table) * train_fraction)
    if split_index <= 0 or split_index >= len(table):
        raise SystemExit(
            "Forecasting window is too short for the configured train/test split."
        )

    train_end_time = float(table.iloc[split_index]["interval_start"])
    train_buy = buy[buy < train_end_time]
    train_sell = sell[sell < train_end_time]
    if len(train_buy) + len(train_sell) == 0:
        raise SystemExit("No training events found before the chronological split.")

    hawkes_fit = fit_hawkes_bivariate(
        train_buy,
        train_sell,
        horizon=max(train_end_time, 1.0),
        shared_beta=config["hawkes"]["shared_beta"],
        max_events_per_side=config["hawkes"].get("max_events_per_side"),
        optimizer_maxiter=config["hawkes"]["optimizer_maxiter"],
        stability_penalty=config["hawkes"]["stability_penalty"],
    )
    intensities = hawkes_intensity_at_times(
        table["interval_start"].to_numpy(float),
        [buy, sell],
        hawkes_fit.mu,
        hawkes_fit.alpha,
        hawkes_fit.beta,
    )
    table = add_hawkes_intensity_features(table, intensities[:, 0], intensities[:, 1])

    reg_metrics, _ = run_regression_experiment(
        table,
        horizon=args.horizon,
        train_fraction=train_fraction,
    )
    cls_metrics, _ = run_classification_experiment(
        table,
        horizon=args.horizon,
        train_fraction=train_fraction,
    )
    reg_metrics.to_csv(f"{args.output_prefix}_regression.csv", index=False)
    cls_metrics.to_csv(f"{args.output_prefix}_classification.csv", index=False)
    Path(f"{args.output_prefix}_hawkes_train_fit.json").write_text(
        json.dumps(
            {
                "window": {
                    "start_hour": start_hour,
                    "duration_minutes": duration_minutes,
                    "train_end_seconds_from_window_start": train_end_time,
                },
                "hawkes": hawkes_fit.to_dict(),
            },
            indent=2,
        )
    )
    print("Regression metrics")
    print(reg_metrics)
    print("Classification metrics")
    print(cls_metrics)


if __name__ == "__main__":
    main()

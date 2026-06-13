"""02 Fit Hawkes order flow models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data import filter_intraday_window, load_processed
from src.features import buy_sell_event_times
from src.hawkes import cap_bivariate_events, fit_hawkes_bivariate, fit_poisson_baseline


def _window_settings(args: argparse.Namespace, config: dict) -> tuple[float, float]:
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
    if duration_minutes is None:
        raise ValueError("hawkes.fit_window.duration_minutes is required")
    return float(start_hour), float(duration_minutes)


def _first_value(df, column: str):
    if column not in df or df.empty:
        return None
    return df[column].iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed", required=True, help="Processed symbol-day parquet/csv file"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/hawkes_fit.json")
    parser.add_argument("--start-hour", type=float, help="UTC hour to start fitting")
    parser.add_argument(
        "--duration-minutes", type=float, help="Minutes to include after start-hour"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    start_hour, duration_minutes = _window_settings(args, config)
    raw_df = load_processed(args.processed)
    start_seconds = start_hour * 3600.0
    end_seconds = start_seconds + duration_minutes * 60.0
    df = filter_intraday_window(
        raw_df,
        start_hour=start_hour,
        duration_minutes=duration_minutes,
        reset_event_time=True,
    )
    if df.empty:
        raise SystemExit("No trades found in the requested intraday window.")

    buy, sell = buy_sell_event_times(df)
    horizon = duration_minutes * 60.0
    cap_mode = config["hawkes"].get("cap_mode", "uniform")
    max_events_per_side = config["hawkes"].get("max_events_per_side")
    buy_fit, sell_fit = cap_bivariate_events(
        buy,
        sell,
        max_events_per_side=max_events_per_side,
        mode=cap_mode,
    )

    hawkes = fit_hawkes_bivariate(
        buy_fit,
        sell_fit,
        horizon=horizon,
        shared_beta=config["hawkes"]["shared_beta"],
        max_events_per_side=None,
        optimizer_maxiter=config["hawkes"]["optimizer_maxiter"],
        stability_penalty=config["hawkes"]["stability_penalty"],
    )
    poisson = fit_poisson_baseline(buy_fit, sell_fit, horizon=horizon)

    result = {
        "window": {
            "symbol": _first_value(df, "symbol"),
            "date": _first_value(df, "date"),
            "start_hour": start_hour,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_minutes": duration_minutes,
            "horizon_seconds": horizon,
            "n_trades_window": int(len(df)),
            "n_buy_window": int(len(buy)),
            "n_sell_window": int(len(sell)),
        },
        "fit_sample": {
            "cap_mode": cap_mode,
            "max_events_per_side": max_events_per_side,
            "n_buy_fit": int(len(buy_fit)),
            "n_sell_fit": int(len(sell_fit)),
            "n_events_fit": int(len(buy_fit) + len(sell_fit)),
        },
        "hawkes": hawkes.to_dict(),
        "poisson": {
            k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in poisson.items()
        },
        "likelihood_comparison_valid": bool(
            hawkes.n_events == poisson["n_events"]
            and hawkes.horizon == poisson["horizon"]
            and hawkes.n_events == len(buy_fit) + len(sell_fit)
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

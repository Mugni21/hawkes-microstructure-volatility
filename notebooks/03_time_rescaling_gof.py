"""03 Time-rescaling goodness-of-fit diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import filter_intraday_window, load_processed
from src.diagnostics import (
    diagnostic_summary_table,
    time_rescale_hawkes,
    time_rescale_poisson,
)
from src.features import buy_sell_event_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--fit", default="reports/hawkes_fit.json")
    parser.add_argument("--output", default="reports/time_rescaling_summary.csv")
    parser.add_argument(
        "--start-hour", type=float, help="UTC hour to start diagnostics"
    )
    parser.add_argument(
        "--duration-minutes", type=float, help="Minutes to include after start-hour"
    )
    args = parser.parse_args()

    fit = json.loads(Path(args.fit).read_text())
    fit_window = fit.get("window", {})
    start_hour = (
        args.start_hour
        if args.start_hour is not None
        else fit_window.get("start_hour", 0.0)
    )
    duration_minutes = (
        args.duration_minutes
        if args.duration_minutes is not None
        else fit_window.get("duration_minutes")
    )

    df = filter_intraday_window(
        load_processed(args.processed),
        start_hour=float(start_hour),
        duration_minutes=None if duration_minutes is None else float(duration_minutes),
        reset_event_time=True,
    )
    if df.empty:
        raise SystemExit("No trades found in the requested intraday window.")

    buy, sell = buy_sell_event_times(df)
    hawkes = fit["hawkes"]
    poisson = fit["poisson"]
    mu = np.asarray(hawkes["mu"], dtype=float)
    alpha = np.asarray(hawkes["alpha"], dtype=float)
    beta = np.asarray(hawkes["beta"], dtype=float)
    rates = np.asarray(poisson["rates"], dtype=float)

    residuals = {
        "hawkes_buy": time_rescale_hawkes(buy, 0, [buy, sell], mu, alpha, beta),
        "hawkes_sell": time_rescale_hawkes(sell, 1, [buy, sell], mu, alpha, beta),
        "poisson_buy": time_rescale_poisson(buy, rates[0]),
        "poisson_sell": time_rescale_poisson(sell, rates[1]),
    }
    summary = diagnostic_summary_table(residuals)
    summary.to_csv(args.output, index=False)
    print(summary)


if __name__ == "__main__":
    main()

"""02 Fit Hawkes order flow models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data import load_processed
from src.features import buy_sell_event_times
from src.hawkes import fit_hawkes_bivariate, fit_poisson_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True, help="Processed symbol-day parquet/csv file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/hawkes_fit.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    df = load_processed(args.processed)
    buy, sell = buy_sell_event_times(df)
    horizon = max(float(df["event_time"].max()), 1.0)
    hawkes = fit_hawkes_bivariate(
        buy,
        sell,
        horizon=horizon,
        shared_beta=config["hawkes"]["shared_beta"],
        max_events_per_side=config["hawkes"]["max_events_per_side"],
        optimizer_maxiter=config["hawkes"]["optimizer_maxiter"],
        stability_penalty=config["hawkes"]["stability_penalty"],
    )
    poisson = fit_poisson_baseline(buy, sell, horizon=horizon)

    result = {"hawkes": hawkes.to_dict(), "poisson": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in poisson.items()}}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

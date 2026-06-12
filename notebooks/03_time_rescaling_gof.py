"""03 Time-rescaling goodness-of-fit diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_processed
from src.diagnostics import diagnostic_summary_table, time_rescale_hawkes, time_rescale_poisson
from src.features import buy_sell_event_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--fit", default="reports/hawkes_fit.json")
    parser.add_argument("--output", default="reports/time_rescaling_summary.csv")
    args = parser.parse_args()

    df = load_processed(args.processed)
    buy, sell = buy_sell_event_times(df)
    fit = json.loads(Path(args.fit).read_text())
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

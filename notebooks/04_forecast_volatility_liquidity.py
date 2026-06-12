"""04 Forecast short-horizon realized volatility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.data import load_processed
from src.features import build_feature_table, buy_sell_event_times
from src.forecasting import add_hawkes_intensity_features, run_classification_experiment, run_regression_experiment
from src.hawkes import hawkes_intensity_at_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--fit", default="reports/hawkes_fit.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--output-prefix", default="reports/forecast")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    df = load_processed(args.processed)
    table = build_feature_table(
        df,
        interval_seconds=config["data"]["interval_seconds"],
        horizons=tuple(config["forecasting"]["horizons_seconds"]),
        rolling_window_seconds=config["forecasting"]["rolling_window_seconds"],
    )

    fit = json.loads(Path(args.fit).read_text())
    buy, sell = buy_sell_event_times(df)
    hawkes = fit["hawkes"]
    intensities = hawkes_intensity_at_times(
        table["interval_start"].to_numpy(float),
        [buy, sell],
        np.asarray(hawkes["mu"], dtype=float),
        np.asarray(hawkes["alpha"], dtype=float),
        np.asarray(hawkes["beta"], dtype=float),
    )
    table = add_hawkes_intensity_features(table, intensities[:, 0], intensities[:, 1])

    reg_metrics, _ = run_regression_experiment(
        table,
        horizon=args.horizon,
        train_fraction=config["forecasting"]["train_fraction"],
    )
    cls_metrics, _ = run_classification_experiment(
        table,
        horizon=args.horizon,
        train_fraction=config["forecasting"]["train_fraction"],
    )
    reg_metrics.to_csv(f"{args.output_prefix}_regression.csv", index=False)
    cls_metrics.to_csv(f"{args.output_prefix}_classification.csv", index=False)
    print("Regression metrics")
    print(reg_metrics)
    print("Classification metrics")
    print(cls_metrics)


if __name__ == "__main__":
    main()

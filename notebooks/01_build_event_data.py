"""01 Build event data.

Run from the repository root:
    python notebooks/01_build_event_data.py --symbol BTCUSDT --date 2024-01-02
or ingest local files:
    python notebooks/01_build_event_data.py --input data/raw/BTCUSDT-aggTrades-2024-01-02.zip --symbol BTCUSDT
"""

# Quick explanation of the dataset:

# Binance aggTrades dataset:
# Each row is an aggregate executed trade on Binance Spot for a pair like BTCUSDT,
# where BTC is the base asset and USDT is the quote asset. The row records execution
# price, quantity, timestamp, and whether the buyer was the passive maker.
# Aggregate trades can combine multiple underlying fills from the same taker order
# at the same price/time, tracked by first_trade_id and last_trade_id.
# We infer aggressor side from buyer_maker:
# buyer_maker=True  -> seller-initiated trade;
# buyer_maker=False -> buyer-initiated trade.


from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data import (
    clean_aggtrades,
    download_binance_aggtrades,
    read_aggtrades_csv,
    save_processed_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", help="YYYY-MM-DD date to download from Binance")
    parser.add_argument("--input", nargs="*", help="Local CSV/ZIP files to ingest")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]

    paths = [Path(p) for p in args.input] if args.input else []
    if args.date and not paths:
        paths.append(
            download_binance_aggtrades(args.symbol, args.date, raw_dir=raw_dir)
        )
    if not paths:
        raise SystemExit(
            "Provide --date for download or --input with local CSV/ZIP files."
        )

    frames = [read_aggtrades_csv(path, symbol=args.symbol) for path in paths]
    df = clean_aggtrades(
        frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    )
    saved = save_processed_splits(df, processed_dir=processed_dir)

    print(df.groupby(["symbol", "date", "aggressor_side"]).size())
    print(f"Saved {len(saved)} processed file(s):")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()

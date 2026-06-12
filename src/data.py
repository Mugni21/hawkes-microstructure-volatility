"""Data loading and event construction for Binance aggregate trades."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BINANCE_AGGTRADES_URL = (
    "https://data.binance.vision/data/spot/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{date}.zip"
)

AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "buyer_maker",
    "best_price_match",
]


def binance_aggtrades_url(symbol: str, date: str) -> str:
    """Return the public Binance aggTrades ZIP URL for a symbol and YYYY-MM-DD date."""
    return BINANCE_AGGTRADES_URL.format(symbol=symbol.upper(), date=date)


def download_binance_aggtrades(
    symbol: str,
    date: str,
    raw_dir: str | Path = "data/raw",
    overwrite: bool = False,
    timeout: int = 30,
) -> Path:
    """Download one Binance daily aggregate-trade ZIP file.

    Binance occasionally changes availability or rate-limits requests; callers should
    catch RuntimeError and fall back to manual download from data.binance.vision.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    out = raw_path / f"{symbol.upper()}-aggTrades-{date}.zip"
    if out.exists() and not overwrite:
        return out

    url = binance_aggtrades_url(symbol, date)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download {url}. Download it manually into {raw_path}.") from exc

    out.write_bytes(response.content)
    return out


def read_aggtrades_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """Read a Binance aggTrades CSV or ZIP containing a single CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    read_kwargs = {"header": None, "names": AGGTRADE_COLUMNS}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside {path}")
            with archive.open(csv_names[0]) as handle:
                df = pd.read_csv(handle, **read_kwargs)
    else:
        df = pd.read_csv(path, **read_kwargs)

    if symbol is None:
        stem = path.name.split("-aggTrades-")[0]
        symbol = stem if stem != path.stem else None
    if symbol is not None:
        df["symbol"] = symbol.upper()
    return df


def infer_aggressor_side(buyer_maker: pd.Series) -> pd.Series:
    """Infer taker/aggressor side from Binance buyerMaker flags."""
    flags = buyer_maker.astype(bool)
    return flags.map({True: "sell", False: "buy"}).astype("category")


def clean_aggtrades(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema, parse UTC timestamps, remove duplicates, and sort."""
    required = set(AGGTRADE_COLUMNS[:-1])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required aggTrade columns: {sorted(missing)}")

    out = df.copy()
    numeric_cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["agg_trade_id", "price", "quantity", "timestamp"])
    out["agg_trade_id"] = out["agg_trade_id"].astype("int64")
    out["timestamp"] = out["timestamp"].astype("int64")
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["date"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out["aggressor_side"] = infer_aggressor_side(out["buyer_maker"])
    out = out.drop_duplicates(subset=["symbol", "agg_trade_id"] if "symbol" in out else ["agg_trade_id"])
    out = out.sort_values(["symbol", "datetime"] if "symbol" in out else ["datetime"])
    out["day_start"] = out["datetime"].dt.floor("D")
    out["event_time"] = (out["datetime"] - out["day_start"]).dt.total_seconds()
    return out.reset_index(drop=True)


def load_and_clean_files(paths: Iterable[str | Path], symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load many raw aggTrade files and return one cleaned DataFrame."""
    path_list = list(paths)
    frames = []
    symbol_list = list(symbols) if symbols is not None else [None] * len(path_list)
    if len(symbol_list) != len(path_list):
        raise ValueError("symbols length must match paths length")
    for path, symbol in zip(path_list, symbol_list):
        frames.append(read_aggtrades_csv(path, symbol=symbol))
    if not frames:
        raise ValueError("No input files provided")
    return clean_aggtrades(pd.concat(frames, ignore_index=True))


def split_by_symbol_day(df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Split cleaned trades into {(symbol, date): frame} groups."""
    if "symbol" not in df or "date" not in df:
        raise ValueError("DataFrame must contain symbol and date columns")
    return {
        (str(symbol), str(date)): group.reset_index(drop=True)
        for (symbol, date), group in df.groupby(["symbol", "date"], sort=True)
    }


def save_processed_splits(
    df: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
    fmt: str = "parquet",
) -> list[Path]:
    """Save cleaned symbol-day frames under data/processed."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for (symbol, date), group in split_by_symbol_day(df).items():
        suffix = "parquet" if fmt == "parquet" else "csv"
        path = out_dir / f"{symbol}_{date}.{suffix}"
        if fmt == "parquet":
            try:
                group.to_parquet(path, index=False)
            except ImportError:
                path = path.with_suffix(".csv")
                group.to_csv(path, index=False)
        elif fmt == "csv":
            group.to_csv(path, index=False)
        else:
            raise ValueError("fmt must be 'parquet' or 'csv'")
        saved.append(path)
    return saved


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load a processed parquet or CSV file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["datetime", "day_start"])

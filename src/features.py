"""Feature engineering for event-time order flow and realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_sell_event_times(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return buy and sell event timestamps in seconds from day start."""
    if "event_time" not in df or "aggressor_side" not in df:
        raise ValueError("DataFrame must include event_time and aggressor_side")
    buy = df.loc[df["aggressor_side"] == "buy", "event_time"].to_numpy(float)
    sell = df.loc[df["aggressor_side"] == "sell", "event_time"].to_numpy(float)
    return np.sort(buy), np.sort(sell)


def add_mid_price_proxy(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add a mid-price proxy from trade prices when quote data is unavailable."""
    out = df.copy()
    out["mid_price_proxy"] = (
        out["price"].astype(float).rolling(window, min_periods=1).median()
    )
    return out


def add_log_returns(
    df: pd.DataFrame, price_col: str = "mid_price_proxy"
) -> pd.DataFrame:
    """Add log price and one-trade log return columns."""
    out = df.copy()
    if price_col not in out:
        out = add_mid_price_proxy(out)
    out["log_price"] = np.log(out[price_col].astype(float))
    out["log_return"] = out["log_price"].diff().fillna(0.0)
    return out


def interval_features(df: pd.DataFrame, interval_seconds: int = 10) -> pd.DataFrame:
    """Aggregate trade count, imbalance, returns, and price proxy on a fixed grid."""
    work = add_log_returns(add_mid_price_proxy(df))
    max_time = float(work["event_time"].max()) if len(work) else 0.0
    bins = np.arange(0, max_time + interval_seconds + 1e-9, interval_seconds)
    if len(bins) < 2:
        bins = np.array([0.0, float(interval_seconds)])
    labels = bins[:-1]
    work["interval_start"] = pd.cut(
        work["event_time"], bins=bins, right=False, labels=labels, include_lowest=True
    ).astype(float)

    grouped = work.groupby("interval_start", observed=True)
    buy_counts = (
        work[work["aggressor_side"] == "buy"]
        .groupby("interval_start", observed=True)
        .size()
    )
    sell_counts = (
        work[work["aggressor_side"] == "sell"]
        .groupby("interval_start", observed=True)
        .size()
    )

    grid = pd.DataFrame({"interval_start": labels})
    grid["trade_count"] = (
        grid["interval_start"].map(grouped.size()).fillna(0).astype(int)
    )
    grid["buy_count"] = grid["interval_start"].map(buy_counts).fillna(0).astype(int)
    grid["sell_count"] = grid["interval_start"].map(sell_counts).fillna(0).astype(int)
    grid["signed_order_flow"] = grid["buy_count"] - grid["sell_count"]
    grid["order_flow_imbalance"] = grid["signed_order_flow"] / grid[
        "trade_count"
    ].replace(0, np.nan)
    grid["order_flow_imbalance"] = grid["order_flow_imbalance"].fillna(0.0)
    last_price = grouped["mid_price_proxy"].last()
    grid["mid_price_proxy"] = grid["interval_start"].map(last_price).ffill().bfill()
    grid["log_price"] = np.log(grid["mid_price_proxy"].astype(float))
    grid["log_return"] = grid["log_price"].diff().fillna(0.0)
    return grid


def add_realized_volatility(
    intervals: pd.DataFrame,
    interval_seconds: int = 10,
    horizons: tuple[int, ...] = (10, 30, 60, 300),
) -> pd.DataFrame:
    """Add backward rolling and forward realized volatility targets."""
    out = intervals.copy()
    ret2 = out["log_return"].pow(2)
    for horizon in horizons:
        steps = max(1, int(round(horizon / interval_seconds)))
        out[f"rolling_rv_{horizon}s"] = np.sqrt(
            ret2.rolling(steps, min_periods=1).sum()
        )
        future_sum = (
            ret2.shift(-1).rolling(steps, min_periods=1).sum().shift(-(steps - 1))
        )
        out[f"future_rv_{horizon}s"] = np.sqrt(future_sum)
    return out


def add_rolling_intensity(
    intervals: pd.DataFrame,
    interval_seconds: int = 10,
    window_seconds: int = 300,
) -> pd.DataFrame:
    """Add rolling trade arrival intensity per second."""
    out = intervals.copy()
    steps = max(1, int(round(window_seconds / interval_seconds)))
    out["rolling_trade_count"] = out["trade_count"].rolling(steps, min_periods=1).sum()
    out["rolling_trade_intensity"] = out["rolling_trade_count"] / (
        steps * interval_seconds
    )
    return out


def add_volatility_regime_labels(
    intervals: pd.DataFrame,
    target_col: str = "future_rv_60s",
    quantile: float = 0.9,
) -> pd.DataFrame:
    """Label high-volatility intervals using a future-RV quantile threshold."""
    out = intervals.copy()
    threshold = out[target_col].quantile(quantile)
    out[f"high_vol_{target_col}"] = (out[target_col] >= threshold).astype(int)
    return out


def build_feature_table(
    df: pd.DataFrame,
    interval_seconds: int = 10,
    horizons: tuple[int, ...] = (10, 30, 60, 300),
    rolling_window_seconds: int = 300,
) -> pd.DataFrame:
    """Build a fixed-interval modeling table from cleaned trades."""
    table = interval_features(df, interval_seconds=interval_seconds)
    table = add_realized_volatility(
        table, interval_seconds=interval_seconds, horizons=horizons
    )
    table = add_rolling_intensity(
        table, interval_seconds=interval_seconds, window_seconds=rolling_window_seconds
    )
    for horizon in horizons:
        table = add_volatility_regime_labels(table, target_col=f"future_rv_{horizon}s")
    return table

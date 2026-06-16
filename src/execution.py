"""Simplified execution-cost simulation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_side(side: str) -> str:
    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    return side


def _normalize_schedule(weights: np.ndarray, total_quantity: float) -> np.ndarray:
    if total_quantity < 0:
        raise ValueError("total_quantity must be nonnegative")
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, 0.0)
    if total_quantity == 0 or len(weights) == 0:
        return np.zeros_like(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    schedule = total_quantity * weights / weights.sum()
    schedule[-1] += total_quantity - schedule.sum()
    return np.maximum(schedule, 0.0)


def build_execution_table(
    feature_table: pd.DataFrame, total_quantity: float, side: str = "buy"
) -> pd.DataFrame:
    """Return interval-level inputs for simplified execution simulation."""
    _validate_side(side)
    if total_quantity < 0:
        raise ValueError("total_quantity must be nonnegative")

    table = pd.DataFrame(index=feature_table.index)
    table["interval_start"] = feature_table["interval_start"].to_numpy(float)

    if "mid_price_proxy" in feature_table:
        price = feature_table["mid_price_proxy"]
    elif "price" in feature_table:
        price = feature_table["price"]
    else:
        raise ValueError("feature_table must contain mid_price_proxy or price")
    table["mid_or_trade_price"] = price.astype(float).ffill().bfill()

    if "traded_volume" in feature_table:
        volume = feature_table["traded_volume"]
    elif "volume" in feature_table:
        volume = feature_table["volume"]
    elif "quantity" in feature_table:
        volume = feature_table["quantity"]
    elif "trade_count" in feature_table:
        volume = feature_table["trade_count"]
    else:
        volume = pd.Series(1.0, index=feature_table.index)
    table["traded_volume"] = volume.astype(float).fillna(0.0)
    table["notional_proxy"] = table["traded_volume"] * table["mid_or_trade_price"]

    for column in ["buy_count", "sell_count", "order_flow_imbalance"]:
        table[column] = feature_table[column] if column in feature_table else 0.0

    if "lambda_buy" in feature_table:
        table["hawkes_lambda_buy"] = feature_table["lambda_buy"].astype(float)
    if "lambda_sell" in feature_table:
        table["hawkes_lambda_sell"] = feature_table["lambda_sell"].astype(float)

    table["parent_side"] = _validate_side(side)
    table["parent_quantity"] = float(total_quantity)
    return table.reset_index(drop=True)


def twap_schedule(n_intervals: int, total_quantity: float) -> np.ndarray:
    """Allocate equal child quantities across intervals."""
    if n_intervals < 0:
        raise ValueError("n_intervals must be nonnegative")
    if n_intervals == 0:
        return np.array([], dtype=float)
    return _normalize_schedule(np.ones(n_intervals), total_quantity)


def volume_participation_schedule(
    volume: np.ndarray | pd.Series, total_quantity: float
) -> np.ndarray:
    """Allocate proportionally to observed interval volume proxy."""
    return _normalize_schedule(np.asarray(volume, dtype=float), total_quantity)


def imbalance_aware_schedule(
    imbalance: np.ndarray | pd.Series,
    total_quantity: float,
    side: str = "buy",
    strength: float = 0.5,
) -> np.ndarray:
    """Allocate using signed order-flow imbalance as pressure signal."""
    side = _validate_side(side)
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    imbalance = np.asarray(imbalance, dtype=float)
    pressure = np.clip(np.nan_to_num(imbalance, nan=0.0), -1.0, 1.0)
    if side == "buy":
        weights = 1.0 - strength * pressure
    else:
        weights = 1.0 + strength * pressure
    return _normalize_schedule(weights, total_quantity)


def hawkes_aware_schedule(
    lambda_buy: np.ndarray | pd.Series,
    lambda_sell: np.ndarray | pd.Series,
    total_quantity: float,
    side: str = "buy",
    strength: float = 0.5,
) -> np.ndarray:
    """Allocate using Hawkes-implied buy/sell pressure signals."""
    side = _validate_side(side)
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    buy = np.asarray(lambda_buy, dtype=float)
    sell = np.asarray(lambda_sell, dtype=float)
    total = buy + sell
    pressure = np.divide(
        buy - sell,
        total,
        out=np.zeros_like(total, dtype=float),
        where=np.isfinite(total) & (total > 0),
    )
    pressure = np.clip(np.nan_to_num(pressure, nan=0.0), -1.0, 1.0)
    if side == "buy":
        weights = 1.0 - strength * pressure
    else:
        weights = 1.0 + strength * pressure
    return _normalize_schedule(weights, total_quantity)


def execution_cost(
    prices: np.ndarray | pd.Series, child_quantities: np.ndarray | pd.Series, side: str
) -> float:
    """Return cash paid for buys or cash received for sells."""
    _validate_side(side)
    prices = np.asarray(prices, dtype=float)
    quantities = np.asarray(child_quantities, dtype=float)
    if len(prices) != len(quantities):
        raise ValueError("prices and child_quantities must have the same length")
    return float(np.sum(prices * quantities))


def arrival_price_benchmark(
    initial_price: float, total_quantity: float, side: str
) -> float:
    """Return arrival-price cash benchmark for a parent order."""
    _validate_side(side)
    return float(initial_price) * float(total_quantity)


def implementation_shortfall(
    prices: np.ndarray | pd.Series, child_quantities: np.ndarray | pd.Series, side: str
) -> float:
    """Return implementation shortfall versus the first execution price."""
    side = _validate_side(side)
    prices = np.asarray(prices, dtype=float)
    quantities = np.asarray(child_quantities, dtype=float)
    if len(prices) != len(quantities):
        raise ValueError("prices and child_quantities must have the same length")
    if len(prices) == 0 or quantities.sum() == 0:
        return 0.0
    arrival = arrival_price_benchmark(prices[0], quantities.sum(), side)
    executed = execution_cost(prices, quantities, side)
    if side == "buy":
        return float(executed - arrival)
    return float(arrival - executed)

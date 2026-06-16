import numpy as np
import pandas as pd

from src.execution import (
    build_execution_table,
    hawkes_aware_schedule,
    imbalance_aware_schedule,
    implementation_shortfall,
    twap_schedule,
    volume_participation_schedule,
)


def assert_valid_schedule(schedule, total_quantity):
    assert np.all(schedule >= 0.0)
    assert np.isclose(schedule.sum(), total_quantity)


def test_schedules_sum_to_total_quantity_and_are_nonnegative():
    total_quantity = 10.0
    volume = np.array([0.0, 5.0, 10.0, 0.0])
    imbalance = np.array([-1.0, -0.2, 0.2, 1.0])
    lambda_buy = np.array([5.0, 4.0, 3.0, 1.0])
    lambda_sell = np.array([1.0, 3.0, 4.0, 5.0])

    schedules = [
        twap_schedule(4, total_quantity),
        volume_participation_schedule(volume, total_quantity),
        imbalance_aware_schedule(imbalance, total_quantity, side="buy"),
        imbalance_aware_schedule(imbalance, total_quantity, side="sell"),
        hawkes_aware_schedule(lambda_buy, lambda_sell, total_quantity, side="buy"),
        hawkes_aware_schedule(lambda_buy, lambda_sell, total_quantity, side="sell"),
    ]

    for schedule in schedules:
        assert_valid_schedule(schedule, total_quantity)


def test_twap_is_constant():
    schedule = twap_schedule(5, 10.0)
    assert np.allclose(schedule, np.full(5, 2.0))


def test_implementation_shortfall_sign_for_buy_and_sell_examples():
    rising_prices = np.array([100.0, 101.0, 102.0])
    falling_prices = np.array([100.0, 99.0, 98.0])
    quantities = np.array([1.0, 1.0, 1.0])

    assert implementation_shortfall(rising_prices, quantities, side="buy") > 0.0
    assert implementation_shortfall(falling_prices, quantities, side="buy") < 0.0
    assert implementation_shortfall(falling_prices, quantities, side="sell") > 0.0
    assert implementation_shortfall(rising_prices, quantities, side="sell") < 0.0


def test_build_execution_table_keeps_required_columns():
    feature_table = pd.DataFrame(
        {
            "interval_start": [0.0, 10.0],
            "mid_price_proxy": [100.0, 101.0],
            "trade_count": [5, 6],
            "buy_count": [3, 2],
            "sell_count": [2, 4],
            "order_flow_imbalance": [0.2, -0.33],
            "lambda_buy": [0.5, 0.6],
            "lambda_sell": [0.4, 0.7],
        }
    )

    table = build_execution_table(feature_table, total_quantity=1.0, side="buy")

    assert list(table["mid_or_trade_price"]) == [100.0, 101.0]
    assert list(table["traded_volume"]) == [5.0, 6.0]
    assert "hawkes_lambda_buy" in table
    assert "hawkes_lambda_sell" in table

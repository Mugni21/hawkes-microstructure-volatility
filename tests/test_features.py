import numpy as np
import pandas as pd

from src.features import build_feature_table, buy_sell_event_times


def sample_clean_trades():
    return pd.DataFrame(
        {
            "event_time": [0.0, 5.0, 10.0, 20.0],
            "aggressor_side": ["buy", "sell", "buy", "sell"],
            "price": [100.0, 101.0, 99.0, 100.0],
        }
    )


def test_buy_sell_event_times():
    buy, sell = buy_sell_event_times(sample_clean_trades())
    assert np.allclose(buy, [0.0, 10.0])
    assert np.allclose(sell, [5.0, 20.0])


def test_realized_volatility_calculation():
    table = build_feature_table(
        sample_clean_trades(), interval_seconds=10, horizons=(10, 20)
    )
    assert "future_rv_10s" in table
    assert "rolling_trade_intensity" in table
    assert np.isfinite(table["future_rv_10s"].dropna()).all()
    assert table["trade_count"].sum() == 4

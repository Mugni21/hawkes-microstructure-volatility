import pandas as pd

from src.data import clean_aggtrades, infer_aggressor_side


def test_aggressor_side_inference():
    sides = infer_aggressor_side(pd.Series([True, False, True]))
    assert list(sides.astype(str)) == ["sell", "buy", "sell"]


def test_timestamp_conversion_and_event_time():
    raw = pd.DataFrame(
        {
            "agg_trade_id": [2, 1, 1],
            "price": [101.0, 100.0, 100.0],
            "quantity": [0.2, 0.1, 0.1],
            "first_trade_id": [20, 10, 10],
            "last_trade_id": [21, 11, 11],
            "timestamp": [1704153605000, 1704153600000, 1704153600000],
            "buyer_maker": [True, False, False],
            "best_price_match": [True, True, True],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        }
    )
    cleaned = clean_aggtrades(raw)
    assert len(cleaned) == 2
    assert str(cleaned.loc[0, "datetime"].tzinfo) == "UTC"
    assert cleaned["event_time"].tolist() == [0.0, 5.0]
    assert cleaned["agg_trade_id"].tolist() == [1, 2]

"""Chronological volatility forecasting experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def add_hawkes_intensity_features(
    table: pd.DataFrame,
    lambda_buy: np.ndarray,
    lambda_sell: np.ndarray,
) -> pd.DataFrame:
    """Append Hawkes-implied intensity features to a fixed-interval table."""
    out = table.copy()
    out["lambda_buy"] = np.asarray(lambda_buy, dtype=float)
    out["lambda_sell"] = np.asarray(lambda_sell, dtype=float)
    out["lambda_total"] = out["lambda_buy"] + out["lambda_sell"]
    out["lambda_imbalance"] = out["lambda_buy"] - out["lambda_sell"]
    return out


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows in chronological order."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(len(df) * train_fraction)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def regression_feature_sets(horizon: int) -> dict[str, list[str]]:
    """Return interpretable feature sets for realized-volatility regression."""
    lagged_rv = f"rolling_rv_{horizon}s"
    return {
        "lagged_rv": [lagged_rv],
        "poisson_intensity": [lagged_rv, "rolling_trade_intensity", "rolling_trade_count"],
        "hawkes_intensity": [
            lagged_rv,
            "rolling_trade_intensity",
            "rolling_trade_count",
            "order_flow_imbalance",
            "lambda_buy",
            "lambda_sell",
            "lambda_total",
            "lambda_imbalance",
        ],
    }


def _available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns]


def run_regression_experiment(
    table: pd.DataFrame,
    horizon: int = 60,
    train_fraction: float = 0.7,
    model_type: str = "ridge",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit chronological RV regressions and return metrics plus fitted models."""
    target = f"future_rv_{horizon}s"
    data = table.dropna(subset=[target]).copy()
    train, test = chronological_split(data, train_fraction=train_fraction)
    rows = []
    models: dict[str, object] = {}
    for name, feature_cols in regression_feature_sets(horizon).items():
        cols = _available_features(data, feature_cols)
        if not cols:
            continue
        estimator = Ridge(alpha=1.0) if model_type == "ridge" else LinearRegression()
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(train[cols], train[target])
        pred = model.predict(test[cols])
        rows.append(
            {
                "model": name,
                "target": target,
                "features": ", ".join(cols),
                "r2_oos": r2_score(test[target], pred),
                "mae": mean_absolute_error(test[target], pred),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        models[name] = model
    return pd.DataFrame(rows), models


def run_classification_experiment(
    table: pd.DataFrame,
    horizon: int = 60,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit chronological high-volatility regime classifiers."""
    target = f"high_vol_future_rv_{horizon}s"
    if target not in table:
        raise ValueError(f"Missing classification target {target}")
    data = table.dropna(subset=[target]).copy()
    train, test = chronological_split(data, train_fraction=train_fraction)
    rows = []
    models: dict[str, object] = {}
    for name, feature_cols in regression_feature_sets(horizon).items():
        cols = _available_features(data, feature_cols)
        if not cols or train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(train[cols], train[target])
        prob = model.predict_proba(test[cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "target": target,
                "features": ", ".join(cols),
                "auc": roc_auc_score(test[target], prob),
                "precision": precision_score(test[target], pred, zero_division=0),
                "recall": recall_score(test[target], pred, zero_division=0),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        models[name] = model
    return pd.DataFrame(rows), models

# Order-Flow Excitation and Short-Horizon Volatility Forecasting with Multivariate Hawkes Processes

This repository is a reproducible quantitative research project on high-frequency crypto trade arrivals. It uses Binance public aggregate trades for BTCUSDT and ETHUSDT, infers signed buyer-initiated and seller-initiated order flow, estimates bivariate Hawkes processes, evaluates goodness of fit with Ogata time-rescaling diagnostics, and tests whether Hawkes-implied intensity features improve short-horizon realized-volatility forecasts.

This is not a trading bot and does not claim profitability. The focus is event-time modeling, endogenous clustering in order flow, and interpretable short-horizon volatility prediction.

## Research Questions

1. Does signed high-frequency order flow exhibit self-excitation and cross-excitation?
2. Do Hawkes branching ratios and excitation matrices change across volatility regimes?
3. Do Hawkes-implied intensity features improve short-horizon realized-volatility forecasts versus simple baselines?
4. Does the Hawkes model pass time-rescaling goodness-of-fit diagnostics better than a homogeneous Poisson baseline?

## Data

The intended data source is Binance public spot aggregate trades:

`https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip`

Expected fields are:

`aggregate trade ID, price, quantity, first trade ID, last trade ID, timestamp, buyerMaker, best price match`

Aggressor-side inference follows Binance semantics:

- `buyerMaker == true`: buyer was passive, so the aggressor was seller-initiated.
- `buyerMaker == false`: buyer was aggressive, so the aggressor was buyer-initiated.

If automatic download fails because of network restrictions, download the ZIP manually from [data.binance.vision](https://data.binance.vision/) and place it under `data/raw/`.

## Methodology

For buy and sell event streams, the bivariate Hawkes intensity is:

```text
lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij exp(-beta_ij (t - t_k^j))
```

The MVP estimator uses direct maximum likelihood with positivity constraints. By default it uses a shared exponential decay parameter `beta` across the 2x2 excitation matrix for stability and speed. The branching matrix is `G_ij = alpha_ij / beta_ij`, and stationarity is assessed using the spectral radius `rho(G)`.

Forecasting uses chronological train/test splits. Baselines include lagged realized volatility and rolling trade intensity. The Hawkes model adds buy, sell, total, and imbalance intensity features.

## Repository Structure

```text
data/raw/                         raw Binance ZIP/CSV files
data/processed/                   cleaned symbol-day data
notebooks/01_build_event_data.py   download/ingest and clean trades
notebooks/02_fit_hawkes_order_flow.py
notebooks/03_time_rescaling_gof.py
notebooks/04_forecast_volatility_liquidity.py
src/data.py                       data ingestion and event construction
src/features.py                   fixed-interval features and RV targets
src/hawkes.py                     Hawkes MLE and Poisson baseline
src/diagnostics.py                time-rescaling diagnostics
src/forecasting.py                forecasting experiments
src/plotting.py                   reusable figures
reports/hawkes_microstructure_report.md
tests/                            unit tests
```

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and process one symbol-day:

```powershell
python notebooks/01_build_event_data.py --symbol BTCUSDT --date 2024-01-02
```

Or ingest a manually downloaded file:

```powershell
python notebooks/01_build_event_data.py --symbol BTCUSDT --input data/raw/BTCUSDT-aggTrades-2024-01-02.zip
```

Fit Hawkes and Poisson models:

```powershell
python notebooks/02_fit_hawkes_order_flow.py --processed data/processed/BTCUSDT_2024-01-02.parquet
```

Run time-rescaling diagnostics:

```powershell
python notebooks/03_time_rescaling_gof.py --processed data/processed/BTCUSDT_2024-01-02.parquet --fit reports/hawkes_fit.json
```

Run the forecasting experiment:

```powershell
python notebooks/04_forecast_volatility_liquidity.py --processed data/processed/BTCUSDT_2024-01-02.parquet --fit reports/hawkes_fit.json --horizon 60
```

Run tests:

```powershell
pytest
```

## Main Outputs

- Cleaned symbol-day trade files under `data/processed/`.
- Hawkes and Poisson estimates under `reports/hawkes_fit.json`.
- Time-rescaling diagnostic table under `reports/time_rescaling_summary.csv`.
- Forecast metrics under `reports/forecast_regression.csv` and `reports/forecast_classification.csv`.
- Figures saved under `reports/figures/` when generated from plotting utilities.

## Caveats and Limitations

- Aggregate trades are not full order-book data. This project does not compute quote imbalance, spread, depth, or true mid-price.
- The mid-price is only a trade-price proxy, used because quote data is unavailable.
- The default Hawkes estimator uses a shared decay parameter for a stable MVP. For larger research runs, compare against separate decay parameters and rolling-window estimation.
- High-frequency crypto data can be very large. Start with short windows or one day, then scale carefully.
- Forecasting metrics are placeholders until the pipeline is run on real data. Do not report performance numbers that were not generated out of sample.

## Resume Bullet Suggestions

Use these only after running the full pipeline and replacing bracketed placeholders with real results:

- Built a reproducible Python research pipeline for Binance high-frequency crypto trades, inferring signed order flow and estimating bivariate Hawkes excitation between buy and sell arrivals.
- Validated event-time model fit using Ogata time-rescaling diagnostics and compared Hawkes residuals against homogeneous Poisson baselines with KS tests and QQ plots.
- Evaluated whether Hawkes-implied intensity features improved chronological short-horizon realized-volatility forecasts over lagged-volatility and trade-intensity baselines.
- Analyzed branching matrices and spectral radii across [N] symbol-day windows to quantify endogenous clustering and regime variation in order-flow excitation.

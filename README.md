# Order-Flow Excitation and Short-Horizon Volatility Forecasting with Multivariate Hawkes Processes

This repository is a reproducible quantitative research project on high-frequency crypto trade arrivals. It uses Binance public aggregate trades for BTCUSDT and ETHUSDT, infers signed buyer-initiated and seller-initiated order flow, estimates bivariate Hawkes processes, evaluates goodness of fit with Ogata time-rescaling diagnostics, tests whether Hawkes-implied intensity features improve short-horizon realized-volatility forecasts, and includes a simplified execution-cost simulator for intraday parent-order schedules.

This is not a trading bot and does not claim profitability. The focus is event-time modeling, endogenous clustering in order flow, interpretable short-horizon volatility prediction, and reduced-form execution research.

## Research Questions

1. Does signed high-frequency order flow exhibit self-excitation and cross-excitation?
2. Do Hawkes branching ratios and excitation matrices change across volatility regimes?
3. Do Hawkes-implied intensity features improve short-horizon realized-volatility forecasts versus simple baselines?
4. Does the Hawkes model pass time-rescaling goodness-of-fit diagnostics better than a homogeneous Poisson baseline?
5. Can simple execution schedules use order-flow pressure signals to change simulated implementation shortfall over an intraday window?

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

Because a single crypto symbol-day can contain hundreds of thousands or millions of trades, model fitting is controlled by an explicit intraday time window rather than by silently truncating event counts. In `config.yaml`, `hawkes.fit_window.start_hour: 5` and `duration_minutes: 120` means fit on trades from 05:00 to 07:00 UTC. CLI flags can override this.

Forecasting uses chronological train/test splits. Baselines include lagged realized volatility and rolling trade intensity. The Hawkes model adds buy, sell, total, and imbalance intensity features. To avoid data snooping, the forecasting script estimates Hawkes parameters only on the training portion of the selected intraday window and applies train-estimated high-volatility thresholds to the test split.

The execution simulator compares TWAP, volume participation, imbalance-aware, and Hawkes-aware schedules for buying or selling a fixed parent quantity. Hawkes intensities are used as reduced-form order-flow pressure signals: for a buy parent order, the simulator slows down when buy pressure is high and speeds up when sell pressure is high; sell orders use the symmetric logic.

## Repository Structure

```text
data/raw/                         raw Binance ZIP/CSV files
data/processed/                   cleaned symbol-day data
notebooks/01_build_event_data.py   download/ingest and clean trades
notebooks/02_fit_hawkes_order_flow.py
notebooks/03_time_rescaling_gof.py
notebooks/04_forecast_volatility_liquidity.py
notebooks/05_execution_simulation.py
src/data.py                       data ingestion and event construction
src/features.py                   fixed-interval features and RV targets
src/hawkes.py                     Hawkes MLE and Poisson baseline
src/diagnostics.py                time-rescaling diagnostics
src/execution.py                  simplified execution-cost simulator
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

Fit a specific UTC intraday window:

```powershell
python notebooks/02_fit_hawkes_order_flow.py --processed data/processed/BTCUSDT_2024-01-02.parquet --start-hour 5 --duration-minutes 180
```

Run time-rescaling diagnostics:

```powershell
python notebooks/03_time_rescaling_gof.py --processed data/processed/BTCUSDT_2024-01-02.parquet --fit reports/hawkes_fit.json
```

Run the forecasting experiment:

```powershell
python notebooks/04_forecast_volatility_liquidity.py --processed data/processed/BTCUSDT_2024-01-02.parquet --start-hour 5 --duration-minutes 180 --horizon 60
```

Run the execution simulation:

```powershell
python notebooks/05_execution_simulation.py --processed data/processed/BTCUSDT_2024-01-02.parquet --fit reports/hawkes_fit.json --total-quantity 1 --side buy
```

When a Hawkes fit JSON is provided, the execution simulator uses the fit file's `window.start_hour` and `window.duration_minutes` by default so Hawkes intensities and execution schedules are evaluated on the same intraday window. CLI `--start-hour` and `--duration-minutes` values override the fit window when supplied.

Run tests:

```powershell
pytest
```

## Main Outputs

- Cleaned symbol-day trade files under `data/processed/`.
- Hawkes and Poisson estimates under `reports/hawkes_fit.json`.
- Time-rescaling diagnostic table under `reports/time_rescaling_summary.csv`.
- Forecast metrics under `reports/forecast_regression.csv` and `reports/forecast_classification.csv`.
- Execution simulation metrics under `reports/execution_results.csv`.
- Interval-level execution schedules under `reports/execution_schedule.csv`, including one child-quantity column per strategy.
- Figures saved under `reports/figures/` when generated from plotting utilities.

## Caveats and Limitations

- Aggregate trades are not full order-book data. This project does not compute quote imbalance, spread, depth, or true mid-price.
- The mid-price is only a trade-price proxy, used because quote data is unavailable.
- The default Hawkes estimator uses a shared decay parameter for a stable MVP. For larger research runs, compare against separate decay parameters and rolling-window estimation.
- High-frequency crypto data can be very large. Start with one- to three-hour windows, then scale carefully.
- Forecasting metrics are placeholders until the pipeline is run on real data. Do not report performance numbers that were not generated out of sample.
- The execution simulator is intentionally simplified. It does not model full limit-order-book depth, queue position, passive fill probability, market impact, latency, real transaction fees, rebates, or exchange constraints.
- Hawkes-aware execution schedules use intensities as reduced-form order-flow pressure signals, not as proof of tradable alpha or profitability.

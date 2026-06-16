# Hawkes Order-Flow Modeling and Execution Simulation

This repository is a reproducible quantitative research project on high-frequency crypto trade arrivals. It uses Binance public aggregate trades for BTCUSDT and ETHUSDT to infer signed buyer-initiated and seller-initiated order flow, estimate bivariate Hawkes processes, validate point-process fit against Poisson baselines, and test simple order-flow-aware execution schedules in a reduced-form implementation-shortfall simulator.

This is not a trading bot and does not claim profitability. The focus is event-time modeling, endogenous clustering in order flow, point-process diagnostics, and simplified execution research using trade-price proxies rather than full order-book data.

## Research Questions

1. Does signed high-frequency order flow exhibit statistically meaningful self-excitation and cross-excitation?
2. Are Hawkes excitation matrices stable and stationary over intraday windows?
3. Do Hawkes models improve point-process fit relative to homogeneous and nonhomogeneous Poisson baselines?
4. Do Hawkes intensities reduce residual serial dependence under time-rescaling diagnostics?
5. Can Hawkes-implied order-flow pressure be used as a state variable in simplified execution scheduling?
6. Does Hawkes pressure behave more like a momentum signal or a contrarian/adverse-selection signal in a historical execution window?

## Data

The intended data source is Binance public spot aggregate trades:

`https://data.binance.vision/data/spot/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip`

Expected fields are:

`aggregate trade ID, price, quantity, first trade ID, last trade ID, timestamp, buyerMaker, best price match`

Aggressor-side inference follows Binance semantics:

* `buyerMaker == true`: buyer was passive, so the aggressor was seller-initiated.
* `buyerMaker == false`: buyer was aggressive, so the aggressor was buyer-initiated.

Important limitation: Binance aggregate trades are not full limit-order-book data. They contain executed trades, not quotes, depth, queue position, order submissions, or cancellations.

## Methodology

For buy and sell event streams, the bivariate Hawkes intensity is:

```text
lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij exp(-beta_ij (t - t_k^j))
```

The estimator uses direct maximum likelihood with positivity constraints. By default it uses a shared exponential decay parameter `beta` across the 2x2 excitation matrix for stability and interpretability. The branching matrix is:

```text
G_ij = alpha_ij / beta_ij
```

Stationarity is assessed using the spectral radius `rho(G)`. A spectral radius below 1 indicates a stable Hawkes process.

Because one crypto symbol-day can contain hundreds of thousands or millions of trades, model fitting is controlled by explicit intraday windows. CLI arguments such as `--start-hour 5 --duration-minutes 10` select a UTC window from 05:00 to 05:10. The fit output stores the selected window metadata so downstream diagnostics and execution simulations can use the same interval.

## Point-Process Validation

The Hawkes model is compared against:

* Homogeneous Poisson baselines.
* Nonhomogeneous Poisson baselines with piecewise-constant rates.
* Time-rescaling diagnostics using transformed interarrival times.
* Residual autocorrelation checks.
* Likelihood, AIC, and BIC comparisons.
* Branching matrices and spectral-radius stability checks.

A representative BTCUSDT 05:00-05:10 UTC window contains 9,069 aggregate trades, with 5,011 buy-initiated and 4,058 sell-initiated events. On this window, the fitted Hawkes process produced a stable spectral radius of 0.574, with diagonal-dominant same-side excitation: buy-to-buy branching ratio 0.573 and sell-to-sell branching ratio 0.458, much larger than cross-side excitation.

On the same 9,069-event window, the Hawkes model improved AIC by roughly 19k relative to a homogeneous Poisson baseline and reduced lag-1 time-rescaled residual autocorrelation by approximately 56% for buy events and 68% for sell events. Exact KS tests still reject the simple exponential Hawkes specification, which is expected for high-frequency aggregate-trade data with timestamp discreteness, batching effects, missing marks, and no limit-order-book state.

## Execution Simulation

The execution module simulates buying or selling a fixed parent quantity over an intraday window. It does not simulate real exchange fills. Instead, it allocates child quantities across fixed time intervals and evaluates the resulting implementation shortfall using observed trade-price proxies.

The simulator compares:

* `twap`: equal child quantity in every interval.
* `volume_participation`: child quantity proportional to observed traded volume.
* `imbalance_aware`: schedule based on raw signed order-flow imbalance.
* `hawkes_contrarian`: treats Hawkes buy/sell intensity imbalance as pressure to fade.
* `hawkes_momentum`: treats Hawkes buy/sell intensity imbalance as short-term urgency to follow.

For a buy parent order, lower implementation shortfall is better. For a sell parent order, lower implementation shortfall is also better under the simulator's sign convention.

In a BTCUSDT 05:00-05:10 UTC test window, Hawkes momentum improved implementation shortfall versus TWAP for both buy and sell simulations, while Hawkes contrarian underperformed TWAP. However, volume participation performed best for the buy order and raw imbalance-aware scheduling performed best for the sell order. This suggests that Hawkes intensities are informative order-flow state variables, but not a standalone optimal execution policy.

The execution simulator saves both strategy-level results and interval-level schedules. The interval-level schedule file includes the price proxy, volume, imbalance, Hawkes intensities, and child quantity assigned by each strategy.

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
src/features.py                   fixed-interval features and realized-volatility targets
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

Fit Hawkes and Poisson models on a specific UTC intraday window:

```powershell
python notebooks/02_fit_hawkes_order_flow.py --processed data/processed/BTCUSDT_2024-01-02.parquet --start-hour 5 --duration-minutes 10 --output reports/hawkes_fit_0500_0510_shared_beta.json
```

Run time-rescaling diagnostics with a nonhomogeneous Poisson baseline:

```powershell
python notebooks/03_time_rescaling_gof.py --processed data/processed/BTCUSDT_2024-01-02.parquet --fit reports/hawkes_fit_0500_0510_shared_beta.json --output reports/time_rescaling_0500_0510_with_nh_poisson_10s.csv --nh-poisson-bin-seconds 10
```

Run the execution simulation:

```powershell
python notebooks/05_execution_simulation.py --processed data/processed/BTCUSDT_2024-01-02.parquet --fit reports/hawkes_fit_0500_0510_shared_beta.json --total-quantity 1 --side buy --output reports/execution_results_0500_0510_buy.csv --schedule-output reports/execution_schedule_0500_0510_buy.csv
```

When a Hawkes fit JSON is provided, the execution simulator uses the fit file's `window.start_hour` and `window.duration_minutes` by default so Hawkes intensities and execution schedules are evaluated on the same intraday window. CLI `--start-hour` and `--duration-minutes` values override the fit window when supplied.

Run tests:

```powershell
pytest
```

## Main Outputs

* Cleaned symbol-day trade files under `data/processed/`.
* Hawkes and Poisson estimates under `reports/hawkes_fit*.json`.
* Time-rescaling diagnostic tables under `reports/time_rescaling*.csv`.
* Forecast metrics under `reports/forecast_regression.csv` and `reports/forecast_classification.csv` if the forecasting experiment is run.
* Strategy-level execution metrics under `reports/execution_results*.csv`.
* Interval-level execution schedules under `reports/execution_schedule*.csv`.
* Figures saved under `reports/figures/`.

## Caveats and Limitations

* Aggregate trades are not full order-book data. This project does not observe quote imbalance, spread, depth, order cancellations, queue position, or passive fill probabilities.
* The price used in the execution simulator is a trade-price proxy, not a true mid-price.
* The execution simulator does not model market impact, latency, fees, rebates, exchange constraints, or realistic order matching.
* Hawkes intensities are used as reduced-form order-flow pressure signals, not as proof of tradable alpha.
* Exact KS goodness-of-fit tests reject the simple Hawkes model on large high-frequency windows; the relevant comparison is whether Hawkes improves residual structure relative to Poisson baselines.
* Reported execution results from individual windows should not be interpreted as robust strategy performance. A stronger claim would require aggregation across many days and windows.
* The forecasting module remains exploratory unless explicitly run with out-of-sample results. Do not report forecasting performance without generated out-of-sample metrics.

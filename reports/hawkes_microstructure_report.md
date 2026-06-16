# Order-Flow Excitation and Short-Horizon Volatility Forecasting with Multivariate Hawkes Processes

## 1. Abstract

This research note studies whether signed high-frequency crypto trade arrivals exhibit self-excitation and cross-excitation, whether Hawkes-implied order-flow intensity features improve short-horizon realized-volatility forecasts, and how reduced-form order-flow pressure signals can be used in a simplified execution-cost simulation. Results in this template should be filled only after running the reproducible pipeline on real Binance aggregate trade data.

## 2. Motivation

Trade arrivals in liquid markets cluster strongly in time. A burst of buyer-initiated or seller-initiated trades may reflect endogenous feedback, liquidity demand, or information arrival. Multivariate Hawkes processes provide an interpretable framework for separating baseline arrival rates from excitation across event types.

## 3. Data

The project uses Binance public spot aggregate trades for BTCUSDT and ETHUSDT. Each record includes price, quantity, timestamp, and the `buyerMaker` flag. Aggressor side is inferred as:

- `buyerMaker = true`: seller-initiated trade.
- `buyerMaker = false`: buyer-initiated trade.

Processed data are split by symbol and UTC day. Event times are measured in seconds from the start of the day.

Fill in after running:

- Symbols:
- Date range:
- Total trades:
- Buy-initiated share:
- Sell-initiated share:

## 4. Model

For event stream \(i \in \{\text{buy}, \text{sell}\}\), the bivariate exponential Hawkes intensity is

\[
\lambda_i(t) =
\mu_i +
\sum_j \sum_{t_k^j < t}
\alpha_{ij}\exp\{-\beta_{ij}(t-t_k^j)\}.
\]

The branching matrix is

\[
G_{ij} = \frac{\alpha_{ij}}{\beta_{ij}},
\]

and stability is summarized by the spectral radius

\[
\rho(G) = \max_m |\lambda_m(G)|.
\]

A stable stationary Hawkes process requires \(\rho(G) < 1\).

## 5. Estimation

The MVP implementation estimates parameters by maximum likelihood with positivity constraints. Because daily crypto aggregate-trade files can be very large, estimation is performed on explicit intraday windows such as 05:00-08:00 UTC rather than by silently truncating the event stream. The default specification uses a shared decay parameter \(\beta\) for all excitation channels to improve numerical stability on short windows:

\[
\beta_{ij} = \beta.
\]

Fill in after running:

- Log-likelihood:
- AIC/BIC:
- Estimated baseline intensities:
- Estimated excitation matrix:
- Branching matrix:
- Spectral radius:

## 6. Goodness-of-Fit Diagnostics

Ogata time rescaling transforms event times using integrated intensity. For a correctly specified model, residual inter-arrivals should be approximately \(Exp(1)\):

\[
\tau_k = \int_{t_{k-1}}^{t_k} \lambda_i(s)\,ds.
\]

Diagnostics include:

- QQ plots against \(Exp(1)\).
- KS tests against \(Exp(1)\).
- Residual autocorrelation.
- Hawkes versus homogeneous Poisson comparison.
- Hawkes versus nonhomogeneous Poisson comparison, where the baseline uses piecewise-constant in-sample rates to control for deterministic time-varying activity.

The homogeneous Poisson baseline is intentionally simple and tests whether clustering remains after assuming a constant arrival rate. The nonhomogeneous Poisson baseline is stronger: it absorbs intraday variation in trade activity but does not model self-excitation or cross-excitation. A useful empirical result is therefore relative improvement in residual diagnostics versus both Poisson baselines, not a claim that the Hawkes model is exactly correct.

KS tests may still reject the \(Exp(1)\) null even when Hawkes improves the residual distribution. In high-frequency data, this should be interpreted as documented misspecification plus evidence of better event-time structure, rather than perfect goodness of fit.

Fill in after running:

- Hawkes buy KS statistic / p-value:
- Hawkes sell KS statistic / p-value:
- Poisson buy KS statistic / p-value:
- Poisson sell KS statistic / p-value:
- Nonhomogeneous Poisson buy KS statistic / p-value:
- Nonhomogeneous Poisson sell KS statistic / p-value:

## 7. Forecasting Experiment

Realized volatility over horizon \(H\) is computed from fixed-interval log returns:

\[
RV_{t,t+H} = \sqrt{\sum_{u=t+1}^{t+H} r_u^2}.
\]

The chronological forecasting setup compares:

1. Lagged realized volatility baseline.
2. Rolling trade count / Poisson intensity baseline.
3. Hawkes intensity features plus baselines.

Targets include future realized volatility at 10s, 30s, 60s, and 300s, plus top-decile high-volatility regime labels.

To avoid data snooping, Hawkes parameters used in forecasting are estimated only on the training portion of the selected window. High-volatility classification thresholds are also estimated on the training split and then applied out of sample.

## 8. Simplified Execution Simulation

The execution module compares schedules for buying or selling a fixed parent quantity over an intraday window:

1. TWAP.
2. Volume participation using observed trade-volume or trade-count proxies.
3. Order-flow imbalance-aware scheduling.
4. Hawkes-aware scheduling using buy and sell intensities as reduced-form pressure signals.

For a buy parent order, the Hawkes-aware rule executes less when buy pressure is high and more when sell pressure is high. For a sell parent order, the rule executes less when sell pressure is high and more when buy pressure is high. Performance is summarized with average execution price and implementation shortfall relative to the arrival price.

This simulator is intentionally simplified. It does not include full limit-order-book depth, queue position, passive fill probabilities, latency, market impact, transaction fees, rebates, or exchange order constraints. It should be interpreted as execution research plumbing and a controlled comparison of schedule rules, not as a live trading or profitability claim.

## 9. Results

Fill in after running:

- Out-of-sample \(R^2\) by horizon:
- MAE by horizon:
- AUC for high-volatility classification:
- Whether Hawkes intensity features improve the baselines:
- Whether excitation matrices vary across volatility regimes:
- Execution implementation shortfall by schedule:

No performance claim should be made unless it is generated by chronological out-of-sample evaluation.

## 10. Limitations

- Binance aggregate trades are not full quote or order-book data.
- The mid-price is approximated with trade prices.
- The shared-\(\beta\) Hawkes model is a practical baseline, not the most general multivariate Hawkes specification.
- Parameter estimates can be sensitive to window length, extreme market conditions, and data filtering choices.
- A volatility forecasting improvement is not the same as a profitable trading strategy.
- The execution simulator does not model full LOB depth, queue priority, passive fills, transaction fees, rebates, latency, or market impact.
- Hawkes intensities are used only as reduced-form order-flow pressure signals.

## 11. Conclusion

This project provides a clean framework for studying signed order-flow clustering and its relationship to short-horizon volatility. The final conclusion should emphasize model validation: whether Hawkes residuals improve relative to homogeneous Poisson and nonhomogeneous Poisson baselines, where remaining KS rejection or residual autocorrelation is treated as transparent misspecification rather than hidden. Forecasting claims should be written only after running the separate chronological prediction experiment across enough symbol-day windows.

# CONFIG.md

This document enumerates the CLI flags supported by `mc_rsklab` and the corresponding runtime behaviour.

Flag | Type | Default | Description
---- | ---- | ------- | -----------
`--tickers` | string | `AAPL` | Comma-separated tickers to include in the portfolio simulation.
`--model` | enum | `gbm` | Simulation model (`gbm`, `tgbm`, `jump`, `garch`).
`--paths` | int | `10000` | Number of Monte Carlo scenarios to simulate.
`--years` | float | `1.0` | Projection horizon in calendar years.
`--steps-per-year` | int | `252` | Discretisation granularity per year (252 ≈ trading days).
`--seed` | int | _none_ | Random seed for reproducibility.
`--risk-neutral` | flag | off | Replace real-world drift with `r - q` for pricing applications.
`--r` | float | fetched | Override the annualised risk-free rate (decimal, e.g. 0.045).
`--q` | float | inferred | Override annualised dividend yield (decimal).
`--fred-ticker` | string | `^IRX` | Proxy ticker for risk-free rate estimation when `--r` is omitted.
`--antithetic` | flag | off | Enable antithetic variance reduction.
`--sobol` | flag | off | Use Sobol low-discrepancy sequences for draws.
`--output-dir` | path | `out` | Directory where CSV outputs and plots are written.
`--window-years` | float | `5.0` | Historical lookback (years) for calibration data.
`--log-level` | string | `INFO` | Verbosity level (`DEBUG`, `INFO`, `WARNING`, ...).
`--backtest` | flag | off | Perform rolling VaR backtest with Kupiec test.
`--backtest-window` | int | `504` | Rolling window (trading days) for VaR calibration.
`--var-alpha` | float | `0.95` | VaR confidence level used in backtesting.

## Derived settings
- Risk-neutral drift is `r - q`, with `r` and `q` sourced from flags or calibration.
- The simulation grid runs for `steps-per-year * years` time steps.
- Portfolio correlation is estimated from historical log returns; if the matrix is not PSD the engine applies a nearest-PSD repair before Cholesky.
- Outputs are tagged by model name, enabling parallel comparisons of different configurations in the same `out/` directory.

## Environment variables (optional)
While not strictly required, the following environment variables influence runtime behaviour if defined:

Variable | Purpose
-------- | -------
`YF_DOWNLOAD_ATTEMPTS` | Controls retry behaviour in `yfinance`.
`FRED_API_KEY` | Enables switching to a true FRED data source if you modify `fetch_risk_free_rate` accordingly.

Update this file whenever new CLI flags or environment options are introduced.

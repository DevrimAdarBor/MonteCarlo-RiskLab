"""CLI entry point for mc_rsklab."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data import (
    compute_dividend_yield,
    fetch_risk_free_rate,
    load_adjusted_close,
    load_dividends,
)
from calb import (
    GBMParams,
    GARCHParams,
    JumpDiffusionParams,
    StudentTParams,
    build_garch_params,
    build_gbm_params,
    build_jump_params,
    build_student_t_params,
    compute_log_returns,
    estimate_daily_moments,
    ewma_volatility,
    estimate_jump_parameters,
    fit_garch_model,
    fit_student_t,
)
from portfolo import estimate_correlation
from smulate import SimulationEngine, SimulationOptions
from utls import (
    configure_logging,
    describe_result,
    save_metadata,
    save_metrics_csv,
    save_path_matrices,
    save_pnl_distribution,
    save_terminal_prices,
)
from plots import (
    plot_qq,
    plot_sample_paths,
    plot_terminal_histogram,
    plot_var_es_hst,
    plot_vol_term_structure,
)
from backtest import (
    compute_pnl_from_paths,
    compute_var_es,
    rolling_var_backtest,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("out")


def _parse_tickers(raw: str | List[str]) -> List[str]:
    if isinstance(raw, list):
        entries = raw
    else:
        entries = raw.replace(";", ",").split(",")
    tickers = [item.strip().upper() for item in entries if item.strip()]
    return tickers or ["AAPL"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo risk lab for equities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", "--tcker", default="AAPL", dest="tickers", help="Comma separated tickers")
    parser.add_argument("--model", default="gbm", choices=["gbm", "tgbm", "jump", "garch"], help="Simulation model")
    parser.add_argument("--paths", type=int, default=10_000, help="Number of Monte Carlo paths")
    parser.add_argument("--years", type=float, default=1.0, help="Projection horizon (years)")
    parser.add_argument("--steps-per-year", type=int, default=252, help="Time steps per year")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--risk-neutral", action="store_true", help="Use risk-neutral drift r - q")
    parser.add_argument("--r", type=float, help="Override risk-free rate")
    parser.add_argument("--q", type=float, help="Override dividend yield")
    parser.add_argument("--fred-ticker", default="^IRX", help="Ticker proxy for risk-free when --r absent")
    parser.add_argument("--antithetic", action="store_true", help="Enable antithetic sampling")
    parser.add_argument("--sobol", action="store_true", help="Enable Sobol sampling")
    parser.add_argument("--output-dir", "--out-dir", "--output-dr", default=str(DEFAULT_OUTPUT_DIR), dest="output_dir", help="Output directory")
    parser.add_argument("--window-years", type=float, default=5.0, help="Historical lookback for calibration")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--backtest", action="store_true", help="Run rolling one-day VaR backtest")
    parser.add_argument("--backtest-window", type=int, default=504, help="Backtest rolling window (trading days)")
    parser.add_argument("--var-alpha", type=float, default=0.95, help="VaR/ES confidence level")
    return parser


def _vol_curves(
    tickers: List[str],
    stats: pd.DataFrame,
    ewma: pd.DataFrame,
    garch_models: Dict[str, GARCHParams],
    steps_per_year: int,
) -> Dict[str, pd.Series]:
    curves: Dict[str, pd.Series] = {}
    for ticker in tickers:
        sample_sigma = stats.loc[ticker, "sigma"] if ticker in stats.index else np.nan
        ewma_sigma = ewma[ticker].iloc[-1] if ticker in ewma else np.nan
        data = {
            "Sample": sample_sigma * np.sqrt(steps_per_year),
            "EWMA": ewma_sigma * np.sqrt(steps_per_year),
        }
        garch = garch_models.get(ticker)
        if garch:
            if garch.garch.long_run_variance >= 0:
                data["GARCH long-run"] = np.sqrt(garch.garch.long_run_variance) * np.sqrt(steps_per_year)
            data["GARCH sigma0"] = garch.garch.sigma0 * np.sqrt(steps_per_year)
        curves[ticker] = pd.Series(data)
    return curves


def _aggregate_paths(result) -> pd.DataFrame:
    """Equal-weight aggregate across tickers to get path-level price series."""
    price_cube = np.asarray(result.paths, dtype=float)
    avg_paths = price_cube.mean(axis=2)
    columns = [f"path_{i}" for i in range(avg_paths.shape[0])]
    return pd.DataFrame(avg_paths.T, index=result.time_index, columns=columns)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    tickers = _parse_tickers(args.tickers)
    configure_logging(args.log_level)
    _LOGGER.info("Running mc_rsklab for tickers %s", tickers)

    prices = load_adjusted_close(tickers, years=args.window_years)
    returns = compute_log_returns(prices)
    stats = estimate_daily_moments(returns)
    ewma = ewma_volatility(returns)

    if args.r is not None:
        risk_free = args.r
    else:
        risk_free = fetch_risk_free_rate(years=args.years, ticker=args.fred_ticker)
    _LOGGER.info("Using risk-free rate %.4f", risk_free)

    base_params: Dict[str, GBMParams] = {}
    t_params: Dict[str, StudentTParams] = {}
    j_params: Dict[str, JumpDiffusionParams] = {}
    g_params: Dict[str, GARCHParams] = {}

    dividends: Dict[str, float] = {}

    for ticker in tickers:
        spot = float(prices.iloc[-1][ticker])
        mu_daily = float(stats.loc[ticker, "mu"])
        sigma_daily = float(stats.loc[ticker, "sigma"])
        ewma_sigma = float(ewma[ticker].iloc[-1])

        if args.q is not None:
            dividend_yield = args.q
        else:
            divs = load_dividends(ticker, years=args.window_years)
            div_yield_series = compute_dividend_yield(prices[ticker], divs)
            dividend_yield = float(div_yield_series.iloc[-1]) if not div_yield_series.empty else 0.0
        dividends[ticker] = dividend_yield

        base = build_gbm_params(
            ticker=ticker,
            spot=spot,
            mu_daily=mu_daily,
            sigma_daily=sigma_daily,
            risk_free=risk_free,
            dividend_yield=dividend_yield,
            sigma_ewma=ewma_sigma,
        )
        base_params[ticker] = base

        if args.model == "tgbm":
            df, loc, scale = fit_student_t(returns[ticker])
            t_params[ticker] = build_student_t_params(base, df=df, loc=loc, scale=scale)
        if args.model == "jump":
            jumps = estimate_jump_parameters(returns[ticker])
            j_params[ticker] = build_jump_params(base, jumps)
        if args.model == "garch":
            garch_model = fit_garch_model(returns[ticker])
            g_params[ticker] = build_garch_params(base, garch_model)

    if len(tickers) > 1:
        correlation = estimate_correlation(returns, tickers)
    else:
        correlation = np.ones((1, 1))

    if args.model == "tgbm":
        params_map = t_params
    elif args.model == "jump":
        params_map = j_params
    elif args.model == "garch":
        params_map = g_params
    else:
        params_map = base_params

    options = SimulationOptions(
        model=args.model,
        years=args.years,
        steps_per_year=args.steps_per_year,
        paths=args.paths,
        seed=args.seed,
        antithetic=args.antithetic,
        sobol=args.sobol,
        risk_neutral=args.risk_neutral,
    )

    engine = SimulationEngine(params_map, correlation, options)
    result = engine.run()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.lower()
    save_path_matrices(result, output_dir, model_tag)
    save_terminal_prices(result, output_dir, model_tag)
    save_pnl_distribution(result, output_dir, model_tag)

    metadata = {
        "tickers": tickers,
        "risk_free": risk_free,
        "dividend_yield": dividends,
        "paths": args.paths,
        "years": args.years,
        "steps_per_year": args.steps_per_year,
        "risk_neutral": args.risk_neutral,
        "model": args.model,
        "simulation": describe_result(result),
    }
    save_metadata(metadata, output_dir, model_tag)

    aggregated_paths = _aggregate_paths(result)
    pnl_series = compute_pnl_from_paths(aggregated_paths)
    var_value, es_value = compute_var_es(pnl_series, alpha=args.var_alpha)
    print(
        f"VaR/ES alpha={args.var_alpha:.2f}  VaR={-var_value:.2f}  ES={-es_value:.2f}  "
        f"(paths={result.paths.shape[0]}, model={args.model}, ticker(s)={','.join(tickers)})"
    )

    metrics = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "alpha": args.var_alpha,
        "var": var_value,
        "es": es_value,
        "model": args.model,
        "tickers": ";".join(tickers),
        "paths": int(result.paths.shape[0]),
        "steps": int(result.paths.shape[1] - 1),
        "years": args.years,
        "risk_mode": "risk-neutral" if args.risk_neutral else "real-world",
    }
    save_metrics_csv(metrics, output_dir / "var_es.csv")

    plot_var_es_hst(
        pnl_series,
        var=var_value,
        es=es_value,
        alpha=args.var_alpha,
        out_path=str(output_dir / "var_es_hst.png"),
    )

    vol_curves = _vol_curves(tickers, stats, ewma, g_params, args.steps_per_year)
    plot_sample_paths(result, output_dir, model_tag)
    plot_terminal_histogram(result, output_dir, model_tag)
    plot_qq(result, output_dir, model_tag)
    plot_vol_term_structure(vol_curves, output_dir, model_tag)

    if args.backtest:
        for ticker in tickers:
            bt = rolling_var_backtest(
                returns[ticker],
                model=args.model,
                alpha=args.var_alpha,
                window=args.backtest_window,
            )
            _LOGGER.info(
                "Backtest %s: violations=%s observations=%s kupiec_p=%.4f",
                ticker,
                bt.violations,
                bt.observations,
                bt.kupiec_pvalue,
            )
            bt.var_series.to_csv(output_dir / f"var_backtest_{model_tag}_{ticker}.csv", header=["var"])


if __name__ == "__main__":
    main()


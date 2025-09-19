"""Rolling VaR backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from calb import fit_garch_model, fit_student_t

_LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    violations: int
    observations: int
    kupiec_lr: float
    kupiec_pvalue: float
    expected_violations: float
    var_series: pd.Series


def compute_pnl_from_paths(paths: pd.DataFrame) -> pd.Series:
    """Compute PnL per path using equal-weighted initial vs terminal levels."""
    if paths.empty:
        raise ValueError("Path matrix is empty")
    terminal = paths.iloc[-1].astype(float).values
    initial = paths.iloc[0].astype(float).values
    pnl = terminal - initial
    return pd.Series(pnl, index=paths.columns, name="PnL")


def compute_var_es(pnl: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Return (VaR, ES) for the provided PnL series."""
    if pnl.empty:
        raise ValueError("PnL series is empty")
    values = pnl.astype(float).values
    q = np.quantile(values, 1.0 - alpha)
    var = -float(q)
    tail_size = max(1, int(np.ceil((1.0 - alpha) * len(values))))
    tail_losses = np.sort(values)[:tail_size]
    es = -float(tail_losses.mean())
    return var, es


def kupiec_test(violations: int, observations: int, alpha: float) -> tuple[float, float]:
    if observations == 0:
        return float("nan"), float("nan")
    p = 1.0 - alpha
    if violations == 0:
        lr = -2.0 * np.log((1 - p) ** observations)
    else:
        k = violations / observations
        lr = -2.0 * (
            np.log((1 - p) ** (observations - violations) * (p**violations))
            - np.log((1 - k) ** (observations - violations) * (k**violations))
        )
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def rolling_var_backtest(
    returns: pd.Series,
    model: str,
    alpha: float = 0.95,
    window: int = 252,
) -> BacktestResult:
    if returns.empty:
        raise ValueError("Returns required for backtesting")
    model = model.lower()
    values: list[float] = []
    realized: list[float] = []
    dates: list[pd.Timestamp] = []

    for end_idx in range(window, len(returns) - 1):
        history = returns.iloc[end_idx - window : end_idx]
        mu = history.mean()
        sigma = history.std(ddof=1)
        next_ret = returns.iloc[end_idx + 1]
        dates.append(returns.index[end_idx + 1])

        if model == "tgbm":
            df, loc, scale = fit_student_t(history)
            q = stats.t.ppf(1 - alpha, df=df, loc=loc, scale=scale)
        elif model == "garch":
            garch = fit_garch_model(history)
            sigma_forecast = np.sqrt(
                garch.omega
                + garch.alpha * (history.iloc[-1] - garch.mu) ** 2
                + garch.beta * garch.sigma0**2
            )
            q = garch.mu + sigma_forecast * stats.norm.ppf(1 - alpha)
        elif model == "jump":
            q = history.quantile(1 - alpha)
        else:
            q = mu + sigma * stats.norm.ppf(1 - alpha)

        values.append(q)
        realized.append(next_ret)

    var_series = pd.Series(values, index=pd.Index(dates, name="date"))
    pnl_series = pd.Series(realized, index=var_series.index)
    violations = int((pnl_series < var_series).sum())
    observations = len(var_series)
    lr, p_value = kupiec_test(violations, observations, alpha)
    expected = (1 - alpha) * observations

    return BacktestResult(
        violations=violations,
        observations=observations,
        kupiec_lr=lr,
        kupiec_pvalue=p_value,
        expected_violations=expected,
        var_series=var_series,
    )

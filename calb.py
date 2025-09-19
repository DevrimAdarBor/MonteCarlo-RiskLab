"""Calibration utilities for mc_rsklab."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

_LOGGER = logging.getLogger(__name__)


@dataclass
class GBMParams:
    ticker: str
    spot: float
    mu: float  # daily drift
    sigma: float  # daily volatility
    risk_free: float
    dividend_yield: float
    sigma_ewma: float

    @property
    def risk_neutral_drift(self) -> float:
        return self.risk_free - self.dividend_yield


@dataclass
class StudentTParams(GBMParams):
    df: float
    loc: float
    scale: float


@dataclass
class JumpParameters:
    intensity: float
    mu: float
    sigma: float

    @property
    def expected_jump(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma**2) - 1.0)


@dataclass
class JumpDiffusionParams(GBMParams):
    jumps: JumpParameters


@dataclass
class GARCHModel:
    omega: float
    alpha: float
    beta: float
    mu: float
    sigma0: float
    long_run_variance: float
    result: object | None = None

    def variance_forecast(self, horizon: int) -> float:
        if not 0 <= self.alpha + self.beta < 1:
            return float("nan")
        return self.long_run_variance + (self.sigma0**2 - self.long_run_variance) * (
            (self.alpha + self.beta) ** horizon
        )


@dataclass
class GARCHParams(GBMParams):
    garch: GARCHModel


FrameLike = pd.Series | pd.DataFrame


def compute_log_returns(prices: FrameLike) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=prices.name or "asset")
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna(how="all")
    returns.columns = [str(col) for col in returns.columns]
    return returns


def estimate_daily_moments(returns: FrameLike) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name or "asset")
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    stats_df = pd.DataFrame({"mu": mu, "sigma": sigma})
    stats_df.index.name = "ticker"
    return stats_df


def annualize_mean(mu_daily: FrameLike, steps_per_year: int = 252) -> FrameLike:
    return mu_daily * float(steps_per_year)


def annualize_vol(sigma_daily: FrameLike, steps_per_year: int = 252) -> FrameLike:
    return sigma_daily * np.sqrt(float(steps_per_year))


def ewma_volatility(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    if not 0 < lam < 1:
        raise ValueError("Lambda must be between 0 and 1")
    squared = returns.pow(2)
    ewma_var = squared.ewm(alpha=1 - lam, adjust=False).mean()
    return ewma_var.pow(0.5)


def fit_student_t(returns: pd.Series, df_bounds: Tuple[float, float] = (3.0, 30.0)) -> Tuple[float, float, float]:
    if returns.empty:
        raise ValueError("Cannot fit Student-t distribution to empty series")
    df, loc, scale = stats.t.fit(returns)
    df = float(np.clip(df, df_bounds[0], df_bounds[1]))
    scale = float(abs(scale)) or returns.std(ddof=1)
    return df, float(loc), scale


def estimate_jump_parameters(
    returns: pd.Series,
    tail_sigma: float = 3.0,
    steps_per_year: int = 252,
) -> JumpParameters:
    if returns.empty:
        raise ValueError("Cannot estimate jumps on empty returns")
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return JumpParameters(intensity=0.01, mu=0.0, sigma=0.01)
    z_scores = (returns - mu) / sigma
    extremes = returns[np.abs(z_scores) >= tail_sigma]
    if extremes.empty:
        return JumpParameters(intensity=0.05, mu=-0.02, sigma=0.05)
    lam = len(extremes) / len(returns) * float(steps_per_year)
    return JumpParameters(
        intensity=float(lam),
        mu=float(extremes.mean()),
        sigma=float(abs(extremes.std(ddof=1)) or 0.05),
    )


def fit_garch_model(returns: pd.Series) -> GARCHModel:
    if returns.empty:
        raise ValueError("Cannot fit GARCH model to empty returns")
    scaled = returns * 100.0
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")
    params = res.params
    omega_pct = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    mu_pct = float(params["mu"])
    omega = omega_pct / (100.0**2)
    mu = mu_pct / 100.0
    sigma0 = float(res.conditional_volatility.iloc[-1] / 100.0)
    if alpha + beta >= 1:
        _LOGGER.warning("Estimated GARCH parameters are near non-stationary regime")
        long_run_var = float("nan")
    else:
        long_run_var = omega / (1 - alpha - beta)
    return GARCHModel(
        omega=omega,
        alpha=alpha,
        beta=beta,
        mu=mu,
        sigma0=sigma0,
        long_run_variance=long_run_var,
        result=res,
    )


def build_gbm_params(
    ticker: str,
    spot: float,
    mu_daily: float,
    sigma_daily: float,
    risk_free: float,
    dividend_yield: float,
    sigma_ewma: float,
) -> GBMParams:
    return GBMParams(
        ticker=ticker,
        spot=spot,
        mu=mu_daily,
        sigma=sigma_daily,
        risk_free=risk_free,
        dividend_yield=dividend_yield,
        sigma_ewma=sigma_ewma,
    )


def build_student_t_params(base: GBMParams, df: float, loc: float, scale: float) -> StudentTParams:
    return StudentTParams(**dict(base.__dict__), df=df, loc=loc, scale=scale)


def build_jump_params(base: GBMParams, jumps: JumpParameters) -> JumpDiffusionParams:
    return JumpDiffusionParams(**dict(base.__dict__), jumps=jumps)


def build_garch_params(base: GBMParams, garch: GARCHModel) -> GARCHParams:
    return GARCHParams(**dict(base.__dict__), garch=garch)


def summarize_calibration(base_params: Dict[str, GBMParams]) -> Dict[str, Dict[str, float]]:
    return {
        ticker: {
            "spot": param.spot,
            "mu_daily": param.mu,
            "sigma_daily": param.sigma,
            "dividend_yield": param.dividend_yield,
        }
        for ticker, param in base_params.items()
    }

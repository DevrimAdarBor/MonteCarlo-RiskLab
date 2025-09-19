"""Monte Carlo engine implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import sobol_seq

from calb import (
    GBMParams,
    GARCHParams,
    JumpDiffusionParams,
    StudentTParams,
)
from portfolo import cholesky_psd

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SimulationOptions:
    model: str
    years: float
    steps_per_year: int
    paths: int
    seed: int | None
    antithetic: bool
    sobol: bool
    risk_neutral: bool


@dataclass(slots=True)
class SimulationResult:
    tickers: List[str]
    time_index: pd.Index
    paths: np.ndarray
    terminal_prices: pd.DataFrame
    pnl: pd.DataFrame
    metadata: Dict[str, float]


def _sobol_normals(n_paths: int, n_steps: int, n_assets: int) -> np.ndarray:
    dim = n_steps * n_assets
    samples = sobol_seq.i4_sobol_generate(dim, n_paths)
    samples = np.clip(samples, 1e-12, 1 - 1e-12)
    normals = scipy_norm_ppf(samples)
    return normals.reshape(n_paths, n_steps, n_assets)


def scipy_norm_ppf(values: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.ppf(values)


def generate_normals(
    options: SimulationOptions,
    n_steps: int,
    n_assets: int,
) -> np.ndarray:
    base_paths = options.paths
    if options.antithetic:
        base_paths = (options.paths + 1) // 2

    if options.sobol:
        normals = _sobol_normals(base_paths, n_steps, n_assets)
    else:
        rng = np.random.default_rng(options.seed)
        normals = rng.standard_normal(size=(base_paths, n_steps, n_assets))

    if options.antithetic:
        normals = np.concatenate([normals, -normals], axis=0)

    return normals[: options.paths]


class SimulationEngine:
    """Core Monte Carlo driver handling all supported models."""

    def __init__(
        self,
        params: Dict[str, GBMParams],
        correlation: np.ndarray,
        options: SimulationOptions,
    ) -> None:
        self.params = params
        self.correlation = correlation
        self.options = options
        self.tickers = list(params.keys())
        self._validate()

    def _validate(self) -> None:
        n_assets = len(self.tickers)
        if self.correlation.shape != (n_assets, n_assets):
            raise ValueError("Correlation matrix dimension mismatch")
        if self.options.paths <= 0:
            raise ValueError("paths must be positive")
        if self.options.steps_per_year <= 0:
            raise ValueError("steps-per-year must be positive")
        if self.options.years <= 0:
            raise ValueError("years must be positive")

    def run(self) -> SimulationResult:
        n_steps = int(round(self.options.steps_per_year * self.options.years))
        normals = generate_normals(self.options, n_steps, len(self.tickers))
        chol = cholesky_psd(self.correlation)
        shocks = normals @ chol.T
        stacked = self._stack_parameters()
        paths = self._simulate(shocks, stacked)

        idx = pd.RangeIndex(start=0, stop=n_steps + 1, step=1, name="step")
        terminal = pd.DataFrame(paths[:, -1, :], columns=self.tickers)
        spot = stacked["spot"]
        pnl = terminal - spot
        metadata = {
            "model": self.options.model,
            "paths": float(paths.shape[0]),
            "steps": float(n_steps),
            "years": float(self.options.years),
            "antithetic": float(self.options.antithetic),
            "sobol": float(self.options.sobol),
            "risk_neutral": float(self.options.risk_neutral),
        }
        return SimulationResult(
            tickers=self.tickers,
            time_index=idx,
            paths=paths,
            terminal_prices=terminal,
            pnl=pnl,
            metadata=metadata,
        )

    def _stack_parameters(self) -> Dict[str, np.ndarray]:
        n_assets = len(self.tickers)
        storage = {
            "spot": np.zeros(n_assets),
            "mu": np.zeros(n_assets),
            "sigma": np.zeros(n_assets),
            "risk_free": np.zeros(n_assets),
            "dividend": np.zeros(n_assets),
            "sigma_ewma": np.zeros(n_assets),
            "student_df": np.zeros(n_assets),
            "student_loc": np.zeros(n_assets),
            "student_scale": np.zeros(n_assets),
            "jump_lambda": np.zeros(n_assets),
            "jump_mu": np.zeros(n_assets),
            "jump_sigma": np.zeros(n_assets),
            "garch_omega": np.zeros(n_assets),
            "garch_alpha": np.zeros(n_assets),
            "garch_beta": np.zeros(n_assets),
            "garch_mu": np.zeros(n_assets),
            "garch_sigma0": np.zeros(n_assets),
            "garch_long_run": np.zeros(n_assets),
        }
        for idx, ticker in enumerate(self.tickers):
            param = self.params[ticker]
            storage["spot"][idx] = param.spot
            storage["mu"][idx] = param.mu
            storage["sigma"][idx] = param.sigma
            storage["risk_free"][idx] = param.risk_free
            storage["dividend"][idx] = param.dividend_yield
            storage["sigma_ewma"][idx] = param.sigma_ewma
            if isinstance(param, StudentTParams):
                storage["student_df"][idx] = param.df
                storage["student_loc"][idx] = param.loc
                storage["student_scale"][idx] = param.scale
            if isinstance(param, JumpDiffusionParams):
                storage["jump_lambda"][idx] = param.jumps.intensity
                storage["jump_mu"][idx] = param.jumps.mu
                storage["jump_sigma"][idx] = param.jumps.sigma
            if isinstance(param, GARCHParams):
                storage["garch_omega"][idx] = param.garch.omega
                storage["garch_alpha"][idx] = param.garch.alpha
                storage["garch_beta"][idx] = param.garch.beta
                storage["garch_mu"][idx] = param.garch.mu
                storage["garch_sigma0"][idx] = param.garch.sigma0
                storage["garch_long_run"][idx] = param.garch.long_run_variance
        return storage

    def _simulate(self, shocks: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        n_paths, n_steps, n_assets = shocks.shape
        paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=float)
        paths[:, 0, :] = params["spot"]
        model = self.options.model.lower()
        dt = 1.0 / float(self.options.steps_per_year)
        annual = float(self.options.steps_per_year)

        if model == "garch":
            return self._simulate_garch(shocks, params, dt)

        sigma_ann = params["sigma"] * np.sqrt(annual)
        mu_ann = params["mu"] * annual

        if model == "tgbm":
            shocks = self._studentize(shocks, params)

        jump_terms = None
        if model == "jump":
            jump_terms = self._jumps(params, shocks.shape, dt)

        drift_ann = mu_ann
        if self.options.risk_neutral:
            drift_ann = params["risk_free"] - params["dividend"]
        if model == "jump":
            drift_ann = drift_ann - params["jump_lambda"] * (
                np.exp(params["jump_mu"] + 0.5 * params["jump_sigma"] ** 2) - 1.0
            )

        drift = (drift_ann - 0.5 * sigma_ann**2) * dt
        vol = sigma_ann * np.sqrt(dt)

        for step in range(n_steps):
            incr = drift + vol * shocks[:, step, :]
            if jump_terms is not None:
                incr = incr + jump_terms[:, step, :]
            paths[:, step + 1, :] = paths[:, step, :] * np.exp(incr)
        return paths

    def _studentize(self, shocks: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        adjusted = shocks.copy()
        rng = np.random.default_rng(self.options.seed)
        for asset_idx, df in enumerate(params["student_df"]):
            if df <= 2:
                continue
            chi2 = rng.chisquare(df, size=shocks.shape[:2])
            scale = np.sqrt(df / chi2) * np.sqrt(max((df - 2) / df, 1e-8))
            adjusted[:, :, asset_idx] = shocks[:, :, asset_idx] * scale
        return adjusted

    def _jumps(self, params: Dict[str, np.ndarray], shape: tuple[int, int, int], dt: float) -> np.ndarray:
        n_paths, n_steps, n_assets = shape
        rng = np.random.default_rng(self.options.seed)
        terms = np.zeros(shape)
        for asset_idx in range(n_assets):
            lam = params["jump_lambda"][asset_idx]
            if lam <= 0:
                continue
            lam_step = lam * dt
            counts = rng.poisson(lam_step, size=(n_paths, n_steps))
            if counts.max() == 0:
                continue
            mu = params["jump_mu"][asset_idx]
            sigma = params["jump_sigma"][asset_idx]
            non_zero = np.argwhere(counts > 0)
            for path_idx, step_idx in non_zero:
                n_jumps = counts[path_idx, step_idx]
                draws = rng.normal(loc=mu, scale=sigma, size=n_jumps)
                terms[path_idx, step_idx, asset_idx] = draws.sum()
        return terms

    def _simulate_garch(self, shocks: np.ndarray, params: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        n_paths, n_steps, n_assets = shocks.shape
        paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=float)
        paths[:, 0, :] = params["spot"]
        sigma2 = np.tile(params["garch_sigma0"] ** 2, (n_paths, 1))
        drift_ann = params["garch_mu"] * float(self.options.steps_per_year)
        if self.options.risk_neutral:
            drift_ann = params["risk_free"] - params["dividend"]
        drift_step = drift_ann * dt
        for step in range(n_steps):
            z = shocks[:, step, :]
            sigma = np.sqrt(np.maximum(sigma2, 1e-18))
            epsilon = sigma * z
            log_return = drift_step + epsilon
            paths[:, step + 1, :] = paths[:, step, :] * np.exp(log_return)
            sigma2 = (
                params["garch_omega"]
                + params["garch_alpha"] * (epsilon**2)
                + params["garch_beta"] * sigma2
            )
            sigma2 = np.clip(sigma2, 1e-18, None)
        return paths

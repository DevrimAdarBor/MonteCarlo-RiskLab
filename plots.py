"""Plotting helpers for simulation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from smulate import SimulationResult


def _prepare(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_sample_paths(result: "SimulationResult", output_dir: str | Path, model: str, max_paths: int = 20) -> None:
    out_dir = Path(output_dir)
    for idx, ticker in enumerate(result.tickers):
        subset = result.paths[:max_paths, :, idx].T
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(result.time_index, subset, alpha=0.7)
        ax.set_title(f"Sample {model.upper()} paths - {ticker}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        fig.tight_layout()
        fig.savefig(_prepare(out_dir / f"paths_{model}_{ticker}.png"))
        plt.close(fig)


def plot_terminal_histogram(result: "SimulationResult", output_dir: str | Path, model: str, bins: int = 50) -> None:
    out_dir = Path(output_dir)
    for ticker in result.tickers:
        data = result.terminal_prices[ticker]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=bins, alpha=0.8, color="#3776ab")
        ax.set_title(f"Terminal distribution ({model.upper()}) - {ticker}")
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(_prepare(out_dir / f"terminal_hist_{model}_{ticker}.png"))
        plt.close(fig)


def plot_qq(result: "SimulationResult", output_dir: str | Path, model: str) -> None:
    out_dir = Path(output_dir)
    for ticker in result.tickers:
        series = result.terminal_prices[ticker]
        log_returns = np.log(series / series.mean())
        fig, ax = plt.subplots(figsize=(8, 5))
        stats.probplot(log_returns, dist="norm", plot=ax)
        ax.set_title(f"QQ plot ({model.upper()}) - {ticker}")
        fig.tight_layout()
        fig.savefig(_prepare(out_dir / f"qq_{model}_{ticker}.png"))
        plt.close(fig)


def plot_vol_term_structure(curves: Dict[str, pd.Series], output_dir: str | Path, model: str) -> None:
    out_dir = Path(output_dir)
    if not curves:
        return
    for ticker, series in curves.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        series.sort_index().plot(ax=ax, marker="o")
        ax.set_title(f"Vol term structure - {ticker}")
        ax.set_ylabel("Annualised volatility")
        fig.tight_layout()
        fig.savefig(_prepare(out_dir / f"vol_term_{model}_{ticker}.png"))
        plt.close(fig)


def plot_var_es_hst(pnl: pd.Series, var: float, es: float, alpha: float, out_path: str) -> None:
    """Plot histogram of PnL with VaR/ES markers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pnl.values, bins=40, alpha=0.7, color="#3776ab")
    ax.axvline(-var, linestyle="--", color="red", label=f"VaR ({alpha:.2f})")
    ax.axvline(-es, linestyle=":", color="orange", label=f"ES ({alpha:.2f})")
    ax.set_title(f"PnL histogram with VaR/ES (alpha={alpha:.2f})")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_prepare(Path(out_path)))
    plt.close(fig)


def plot_var_es_hist(pnl: pd.Series, var: float, es: float, alpha: float, out_path: str) -> None:
    """Backward compatibility wrapper for legacy name."""
    plot_var_es_hst(pnl, var, es, alpha, out_path)

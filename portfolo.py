"""Portfolio helpers: covariance estimation and Cholesky repair."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_correlation(returns: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    if returns.empty:
        return np.ones((len(tickers), len(tickers)))
    corr = returns.corr().loc[tickers, tickers]
    return corr.to_numpy()


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, epsilon, None)
    return (eigvecs * eigvals) @ eigvecs.T


def cholesky_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        repaired = nearest_psd(matrix, epsilon)
        return np.linalg.cholesky(repaired)

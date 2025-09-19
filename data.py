"""Market data download helpers for mc_rsklab."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, List

import pandas as pd
import yfinance as yf

_LOGGER = logging.getLogger(__name__)


def _normalize_tickers(tickers: Iterable[str] | str) -> List[str]:
    if isinstance(tickers, str):
        items = tickers.replace(";", ",").split(",")
    else:
        items = list(tickers)
    symbols = [str(item).strip().upper() for item in items if str(item).strip()]
    if not symbols:
        raise ValueError("At least one ticker must be provided")
    return symbols


def load_adjusted_close(
    tickers: Iterable[str] | str,
    years: float = 5.0,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices for the requested tickers."""
    symbols = _normalize_tickers(tickers)
    lookback_days = max(int(years * 365.25), 252)
    start_dt = datetime.utcnow() - timedelta(days=lookback_days)
    _LOGGER.info(
        "Downloading prices for %s from %s", symbols, start_dt.date()
    )
    data = yf.download(
        tickers=symbols,
        start=start_dt.strftime("%Y-%m-%d"),
        progress=False,
        interval=interval,
        auto_adjust=False,
        actions=True,
        threads=True,
    )
    if data.empty:
        raise RuntimeError(f"No price data returned for tickers: {symbols}")

    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"].copy()
    else:
        adj_close = data[["Adj Close"]].copy()
        adj_close.columns = symbols

    adj_close = adj_close.ffill().dropna(how="all")
    adj_close.index = adj_close.index.tz_localize(None)
    return adj_close


def load_dividends(ticker: str, years: float = 5.0) -> pd.Series:
    """Fetch dividend cashflows for a ticker."""
    lookback_days = max(int(years * 365.25), 365)
    start_dt = datetime.utcnow() - timedelta(days=lookback_days)
    _LOGGER.info("Downloading dividend history for %s", ticker.upper())
    series = yf.Ticker(ticker.upper()).dividends
    if series.empty:
        return pd.Series(dtype="float64")
    series.index = series.index.tz_localize(None)
    return series[series.index >= start_dt]


def compute_dividend_yield(
    prices: pd.Series,
    dividends: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Compute a trailing dividend yield from price and dividend series."""
    if prices.empty:
        raise ValueError("Price series required for dividend yield calculation")

    if dividends.empty:
        return pd.Series(index=prices.index, data=0.0)

    aligned = pd.Series(index=prices.index, data=0.0)
    matched = dividends.reindex(prices.index, method="nearest", tolerance=pd.Timedelta(days=5))
    matched = matched.fillna(0.0)
    aligned.loc[matched.index] = matched.values

    rolling_sum = aligned.rolling(window=window, min_periods=1).sum()
    return (rolling_sum / prices).fillna(0.0).clip(lower=0.0)


def fetch_risk_free_rate(
    years: float = 1.0,
    ticker: str = "^IRX",
) -> float:
    """Estimate an annualised risk-free rate from a market proxy."""
    lookback_days = max(int(years * 365.25), 365)
    start_dt = datetime.utcnow() - timedelta(days=lookback_days)
    _LOGGER.info("Fetching risk-free proxy %s from %s", ticker, start_dt.date())
    data = yf.download(
        tickers=ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        actions=False,
    )
    if data.empty:
        raise RuntimeError(f"No risk-free data returned for {ticker}")

    if "Adj Close" not in data.columns:
        raise RuntimeError("Expected 'Adj Close' column for risk-free proxy")

    series = data["Adj Close"].dropna()
    if series.empty:
        raise RuntimeError("Risk-free proxy contains only NaN values")

    return float(series.tail(252).mean()) / 100.0

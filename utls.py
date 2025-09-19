"""Miscellaneous utilities for mc_rsklab."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from smulate import SimulationResult


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_path_matrices(result: "SimulationResult", output_dir: str | Path, model: str) -> None:
    out_dir = ensure_directory(output_dir)
    for asset_idx, ticker in enumerate(result.tickers):
        matrix = result.paths[:, :, asset_idx]
        df = pd.DataFrame(
            matrix.T,
            index=result.time_index,
            columns=[f"path_{i}" for i in range(matrix.shape[0])],
        )
        df.to_csv(out_dir / f"paths_{model}_{ticker}.csv")


def save_terminal_prices(result: "SimulationResult", output_dir: str | Path, model: str) -> None:
    out_dir = ensure_directory(output_dir)
    result.terminal_prices.to_csv(out_dir / f"terminal_{model}.csv", index=False)


def save_pnl_distribution(result: "SimulationResult", output_dir: str | Path, model: str) -> None:
    out_dir = ensure_directory(output_dir)
    result.pnl.to_csv(out_dir / f"pnl_{model}.csv", index=False)


def save_metadata(metadata: dict, output_dir: str | Path, model: str) -> None:
    out_dir = ensure_directory(output_dir)
    with (out_dir / f"meta_{model}.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def save_metrics_csv(metrics: dict, path: str | Path) -> None:
    """Append or create a CSV file with a single-row metrics payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([metrics])
    if target.exists():
        frame.to_csv(target, mode="a", header=False, index=False)
    else:
        frame.to_csv(target, index=False)


def describe_result(result: "SimulationResult") -> dict[str, float]:
    return {
        "paths": float(result.paths.shape[0]),
        "steps": float(result.paths.shape[1] - 1),
    }

import pathlib
import subprocess
import sys

import pytest


def test_cli_aliases_parse(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from mc import _build_parser  # type: ignore import

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--tcker",
            "AAPL",
            "--model",
            "gbm",
            "--paths",
            "50",
            "--years",
            "0.1",
            "--backtest",
            "--var-alpha",
            "0.95",
            "--output-dr",
            str(tmp_path),
        ]
    )
    assert args.tickers == "AAPL"
    assert args.model == "gbm"
    assert args.paths == 50
    assert pytest.approx(args.years, rel=1e-9) == 0.1
    assert args.backtest is True
    assert pytest.approx(args.var_alpha, rel=1e-9) == 0.95
    assert args.output_dir == str(tmp_path)


def test_cli_runs_and_writes_outputs(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "mc.py"),
        "--tcker",
        "AAPL",
        "--model",
        "gbm",
        "--paths",
        "50",
        "--years",
        "0.1",
        "--backtest",
        "--var-alpha",
        "0.95",
        "--output-dr",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    files = {p.name for p in tmp_path.glob("*")}
    assert files, "Expected output files in temporary directory"
    assert any("var_es" in f for f in files), files

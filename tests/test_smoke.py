import subprocess, sys, pathlib, socket, pytest

def _internet_ok(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _internet_ok(), reason="no internet in CI")
def test_cli_runs_and_writes_outputs(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo/'mc_rsklab'/'mc.py'),
           '--ticker','AAPL','--model','gbm',
           '--paths','20','--years','0.05',
           '--backtest','--var-alpha','0.95',
           '--output-dir', str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    files = {p.name for p in tmp_path.glob('*')}
    assert any('var_es' in f for f in files), files

"""
Local MLflow model server for Windows.

ZenML's MLflow model deployer runs the server as a background daemon, which
ZenML does not support on Windows. This module starts the same MLflow scoring
server (`mlflow models serve`) as a detached process instead, and remembers its
process ID so it can be stopped later.
"""

import logging
import os
import subprocess
import sys
import time
import urllib.request

import mlflow
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PID_FILE = os.path.join(PROJECT_ROOT, ".mlflow_server.pid")
LOG_FILE = os.path.join(PROJECT_ROOT, "mlflow_server.log")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def get_latest_model_uri(tracking_uri: str, experiment_name: str = "ml_pipeline") -> str:
    """Return the MLflow URI of the model from the most recent training run."""
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1
    )
    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs found in experiment '{experiment_name}'. "
            "Train a model first with: python run_pipeline.py"
        )
    return f"runs:/{runs.iloc[0].run_id}/model"


def is_server_up(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check whether a model server answers on the given address."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/ping", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_server() -> bool:
    """Stop the server started by start_server(). Returns True if one was stopped."""
    if not os.path.exists(PID_FILE):
        return False
    with open(PID_FILE) as f:
        pid = f.read().strip()
    os.remove(PID_FILE)

    # The PID may be stale (e.g. after a reboot) and reused by another program,
    # so only stop it if it is still our MLflow server.
    try:
        cmdline = " ".join(psutil.Process(int(pid)).cmdline())
    except (psutil.Error, ValueError):
        return False
    if "mlflow" not in cmdline:
        return False

    # /T also stops the child process MLflow starts (waitress)
    result = subprocess.run(
        ["taskkill", "/PID", pid, "/T", "/F"], capture_output=True, text=True
    )
    stopped = result.returncode == 0
    if stopped:
        logging.info(f"Stopped MLflow prediction server (PID {pid}).")
    return stopped


def start_server(
    model_uri: str,
    tracking_uri: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: int = 120,
) -> str:
    """Start the MLflow scoring server in the background and return its predict URL."""
    # Replace any server we started earlier so the newest model is served
    stop_server()
    if is_server_up(host, port):
        raise RuntimeError(
            f"Port {port} is already in use by another program. Close it and try again."
        )

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    # MLflow launches `waitress-serve` by name, so the venv Scripts folder must be on PATH
    scripts_dir = os.path.dirname(sys.executable)
    env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")

    cmd = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_uri,
        "-h", host,
        "-p", str(port),
        "--env-manager", "local",
    ]
    log = open(LOG_FILE, "w")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(
        cmd, env=env, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=flags,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"MLflow server exited early. See {LOG_FILE} for details.")
        if is_server_up(host, port):
            return f"http://{host}:{port}/invocations"
        time.sleep(1)

    stop_server()
    raise RuntimeError(f"MLflow server did not start within {timeout}s. See {LOG_FILE}.")

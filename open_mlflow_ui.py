"""Open the MLflow dashboard for the experiments tracked by the ZenML stack."""

import os
import subprocess
import sys
import threading
import webbrowser

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

PORT = 5000

if __name__ == "__main__":
    tracking_uri = get_tracking_uri()
    env = os.environ.copy()
    # MLflow launches `waitress-serve` by name on Windows, so the venv Scripts folder must be on PATH
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    url = f"http://127.0.0.1:{PORT}"
    print(f"MLflow dashboard: {url}  (press Ctrl+C to stop)")
    threading.Timer(5, lambda: webbrowser.open(url)).start()
    subprocess.run(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri, "--port", str(PORT)],
        env=env,
    )

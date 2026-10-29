"""
MLOps measurements for the term paper (Experiment 5):
  - ZenML pipeline run durations and run-to-run reproducibility
  - MLflow prediction server start-up time and request latency

Run from the project folder:
    venv\\Scripts\\python.exe research\\mlops_measurements.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from zenml.client import Client  # noqa: E402
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri  # noqa: E402

from src.windows_model_server import (  # noqa: E402
    get_latest_model_uri, is_server_up, start_server, stop_server,
)

RESULTS = os.path.join(ROOT, "research", "results")
os.makedirs(RESULTS, exist_ok=True)


def pipeline_runs():
    client = Client()
    rows = []
    for run in client.list_pipeline_runs(sort_by="asc:created", size=200).items:
        name = run.pipeline.name if run.pipeline else "?"
        start, end = run.start_time, run.end_time
        row = {"pipeline": name, "status": str(run.status), "start": str(start),
               "duration_s": (end - start).total_seconds() if start and end else None}
        if name == "ml_pipeline" and "model_evaluator_step" in run.steps:
            outs = run.steps["model_evaluator_step"].outputs
            art = outs.get("output_0")
            art = art[0] if isinstance(art, list) else art
            try:
                row["R2_log"] = art.load()["R-Squared"]
            except Exception as e:  # pragma: no cover
                row["R2_log"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def sample_house(n=1):
    from steps.dynamic_importer import dynamic_importer
    df = pd.read_json(dynamic_importer.entrypoint(), orient="split")
    df["Gr Liv Area"] = np.log1p(df["Gr Liv Area"])
    return pd.concat([df] * (n // len(df) + 1)).head(n).reset_index(drop=True)


def latency(url, n_rows, n_requests):
    payload = {"dataframe_split": json.loads(sample_house(n_rows).to_json(orient="split", index=False))}
    for _ in range(5):  # warm-up
        requests.post(url, json=payload, timeout=30)
    times = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"rows_per_request": n_rows, "requests": n_requests,
            "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean())}


def main():
    out = {}
    runs = pipeline_runs()
    runs.to_csv(os.path.join(RESULTS, "e5_pipeline_runs.csv"), index=False)
    ml = runs[(runs.pipeline == "ml_pipeline") & runs.status.str.contains("completed")]
    out["training_runs_completed"] = int(len(ml))
    out["training_duration_s"] = {"median": float(ml.duration_s.median()),
                                  "min": float(ml.duration_s.min()), "max": float(ml.duration_s.max())}
    out["distinct_R2_values"] = {f"{v:.16f}": int(c) for v, c in ml.R2_log.dropna().value_counts().items()}
    inf = runs[(runs.pipeline == "windows_inference_pipeline") & runs.status.str.contains("completed")]
    out["inference_runs_completed"] = int(len(inf))
    out["inference_duration_s_median"] = float(inf.duration_s.median()) if len(inf) else None

    was_running = is_server_up()
    if was_running:
        stop_server()
        time.sleep(2)
    tracking_uri = get_tracking_uri()
    t0 = time.perf_counter()
    url = start_server(get_latest_model_uri(tracking_uri), tracking_uri)
    out["server_startup_s"] = round(time.perf_counter() - t0, 2)
    try:
        out["latency_single"] = latency(url, 1, 200)
        out["latency_batch100"] = latency(url, 100, 50)
    finally:
        if not was_running:
            stop_server()
    out["server_left_running"] = was_running

    with open(os.path.join(RESULTS, "e5_mlops.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

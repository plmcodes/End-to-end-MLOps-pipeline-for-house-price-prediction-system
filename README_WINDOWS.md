# Running on Windows

## One-time setup
1. Install Python 3.11 (the pinned libraries do not support newer versions):
   `winget install -e --id Python.Python.3.11`
2. Double-click `setup_windows.bat`. It creates `venv`, installs `requirements.txt`
   and registers the ZenML stack (MLflow experiment tracker + model deployer).

## Demo
| Script | What it does |
|---|---|
| `1_train_model.bat` | Runs the training pipeline (`run_pipeline.py`) |
| `2_deploy_model.bat` | Trains, deploys the model to `http://127.0.0.1:8000` and runs batch inference (`run_deployment.py`) |
| `3_predict_price.bat` | Sends one sample house to the deployed model (`sample_predict.py`) |
| `4_open_mlflow_dashboard.bat` | Opens the MLflow dashboard at `http://127.0.0.1:5000` |
| `5_stop_prediction_server.bat` | Stops the prediction server (`run_deployment.py --stop-service`) |

## Windows changes
- **Deployment:** ZenML's MLflow model deployer runs as a daemon, which ZenML does not
  support on Windows. On Windows, `run_deployment.py` starts the same MLflow scoring
  server itself (`src/windows_model_server.py`) and runs `windows_inference_pipeline`,
  which calls the server over HTTP (`steps/http_predictor.py`). Mac and Linux still use
  the original ZenML deployment.
- **Data path:** `pipelines/training_pipeline.py` no longer points to the original
  author's Mac; it uses the bundled `data/archive.zip`.
- **Log scale:** the model is trained on log(1 + living area) and log(1 + price).
  The prediction scripts apply the same transform and convert the result back to dollars.
- **Outliers:** `steps/outlier_detection_step.py` now removes outliers by `SalePrice`
  only, as the pipeline requests. Before, it checked every column and dropped 30% of
  houses (everything with a pool, basement half-bath, etc.), so the model badly
  mispriced such houses.

import sys

import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

IS_WINDOWS = sys.platform == "win32"


def run_windows(stop_service: bool):
    """Windows version: ZenML's MLflow deployer needs a background daemon, which
    ZenML does not support on Windows, so the MLflow server is started directly."""
    from pipelines.deployment_pipeline import windows_inference_pipeline
    from pipelines.training_pipeline import ml_pipeline
    from src.windows_model_server import get_latest_model_uri, start_server, stop_server

    if stop_service:
        if stop_server():
            print("The MLflow prediction server has been stopped.")
        else:
            print("No running prediction server was found.")
        return

    # 1. Train the model
    ml_pipeline()

    # 2. Deploy the newest model to a local MLflow prediction server
    tracking_uri = get_tracking_uri()
    model_uri = get_latest_model_uri(tracking_uri)
    print(f"Deploying model [bold]{model_uri}[/bold] ...")
    prediction_url = start_server(model_uri, tracking_uri)

    # 3. Run batch inference against the deployed model
    windows_inference_pipeline(prediction_url=prediction_url)

    print(
        "Now run \n "
        f'    mlflow ui --backend-store-uri "{tracking_uri}"\n'
        "To inspect your experiment runs within the mlflow UI.\n"
    )
    print(
        f"The MLflow prediction server is running locally in the background and "
        f"accepts inference requests at:\n"
        f"    {prediction_url}\n"
        f"Try it with: python sample_predict.py\n"
        f"To stop the service, re-run the same command and supply the "
        f"`--stop-service` argument."
    )


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the prices predictor deployment pipeline"""
    if IS_WINDOWS:
        run_windows(stop_service)
        return

    model_name = "prices_predictor"

    if stop_service:
        # Get the MLflow model deployer stack component
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # Fetch existing services with same pipeline name, step name, and model name
        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True,
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        return

    # Run the continuous deployment pipeline
    continuous_deployment_pipeline()

    # Get the active model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Run the inference pipeline
    inference_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )

    # Fetch existing services with the same pipeline name, step name, and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()

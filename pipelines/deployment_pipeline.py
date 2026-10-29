import os

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.http_predictor import http_predictor
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    trained_model = ml_pipeline()  # No need for is_promoted return value anymore

    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data)


@pipeline(enable_cache=False)
def windows_inference_pipeline(prediction_url: str):
    """Batch inference against a locally started MLflow server (used on Windows,
    where ZenML's MLflow deployer service is not supported)."""
    batch_data = dynamic_importer()
    http_predictor(input_data=batch_data, prediction_url=prediction_url)

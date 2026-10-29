import logging

import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int, float])

    # Only look for outliers in the requested column. Checking every column removed all
    # houses with rare-but-valid features (pools, basement half baths, ...), which left
    # the model unable to price such houses.
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric[[column_name]])[column_name]
    df_cleaned = df_numeric[~outliers]
    logging.info(
        f"Removed {int(outliers.sum())} outlier rows based on '{column_name}'; "
        f"{len(df_cleaned)} rows remain."
    )
    return df_cleaned

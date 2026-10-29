import json
import logging

import numpy as np
import pandas as pd
import requests
from zenml import step

# Columns that the training pipeline log-transformed (log1p).
# Inputs must get the same transform, and the predicted SalePrice must be converted back.
LOG_FEATURES = ["Gr Liv Area"]


@step(enable_cache=False)
def http_predictor(input_data: str, prediction_url: str) -> np.ndarray:
    """Send a batch of houses to the MLflow prediction server and return prices in dollars.

    Args:
        input_data (str): Houses as a JSON string in pandas "split" format.
        prediction_url (str): The server's /invocations URL.

    Returns:
        np.ndarray: Predicted sale prices in dollars.
    """
    df = pd.read_json(input_data, orient="split")

    for col in LOG_FEATURES:
        df[col] = np.log1p(df[col])

    payload = {"dataframe_split": json.loads(df.to_json(orient="split", index=False))}
    response = requests.post(prediction_url, json=payload, timeout=30)
    response.raise_for_status()

    log_prices = np.array(response.json()["predictions"], dtype=float)
    prices = np.expm1(log_prices)

    for i, price in enumerate(prices, start=1):
        logging.info(f"House {i}: predicted price ${price:,.0f}")

    return prices

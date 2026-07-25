"""
Feature Scaling Module for HQFSF.
"""

import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)

from utils.logger import setup_logger

logger = setup_logger()


class FeatureScaler:

    def __init__(self, method="minmax"):

        self.method = method.lower()

        if self.method == "minmax":
            self.scaler = MinMaxScaler()

        elif self.method == "standard":
            self.scaler = StandardScaler()

        elif self.method == "robust":
            self.scaler = RobustScaler()

        else:
            raise ValueError(f"Unsupported scaler: {method}")

    def fit_transform(self, X: pd.DataFrame):

        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"{self.method} scaling applied.")

        return pd.DataFrame(
            X_scaled,
            columns=X.columns
        )

    def transform(self, X: pd.DataFrame):

        X_scaled = self.scaler.transform(X)

        return pd.DataFrame(
            X_scaled,
            columns=X.columns
        )
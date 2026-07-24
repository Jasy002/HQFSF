"""
Data preprocessing utilities for HQFSF.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocessing pipeline for tabular datasets.
    """

    def __init__(self, scaler: str = "minmax"):
        scaler = scaler.lower()

        if scaler == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(
                "Scaler must be either 'minmax' or 'standard'."
            )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates and fill missing values.
        """

        original_rows = len(df)

        df = df.drop_duplicates()

        duplicates_removed = original_rows - len(df)

        missing_before = df.isna().sum().sum()

        df = df.ffill().bfill()

        missing_after = df.isna().sum().sum()

        logger.info(
            "Cleaning complete | Removed %d duplicates | Missing values %d -> %d",
            duplicates_removed,
            missing_before,
            missing_after,
        )

        return df

    def split_features_target(
        self,
        df: pd.DataFrame,
        target_column: str = "target",
    ):
        """
        Split dataframe into X and y.
        """

        X = df.drop(columns=[target_column])

        y = df[target_column]

        logger.info(
            "Split dataset | Features=%d | Samples=%d",
            X.shape[1],
            X.shape[0],
        )

        return X, y

    def scale_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale feature matrix.
        """

        scaled = self.scaler.fit_transform(X)

        X_scaled = pd.DataFrame(
            scaled,
            columns=X.columns,
            index=X.index,
        )

        logger.info(
            "Feature scaling completed using %s",
            self.scaler.__class__.__name__,
        )

        return X_scaled
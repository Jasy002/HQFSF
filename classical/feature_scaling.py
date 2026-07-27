"""
Feature Scaling Module for HQFSF.

Supports:
    - Min-Max Scaling
    - Standard Scaling
    - Robust Scaling

Provides a unified interface for feature normalization before
classical machine learning and quantum feature encoding.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureScaler:
    """
    Feature Scaling Utility.

    Parameters
    ----------
    method : str, default="minmax"

        Scaling technique.

        Supported:

        - minmax
        - standard
        - robust
    """

    SUPPORTED_SCALERS = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "robust": RobustScaler,
    }

    def __init__(
        self,
        method: str = "minmax",
    ) -> None:

        self.method = method.lower()

        if self.method not in self.SUPPORTED_SCALERS:

            raise ValueError(
                f"Unsupported scaler '{method}'. "
                f"Supported scalers: "
                f"{list(self.SUPPORTED_SCALERS.keys())}"
            )

        self.scaler = self.SUPPORTED_SCALERS[
            self.method
        ]()

        logger.info(
            "FeatureScaler initialized | Method=%s",
            self.method.upper(),
        )

    # ---------------------------------------------------------
    # Fit & Transform
    # ---------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "FeatureScaler":
        """
        Fit the scaler.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        FeatureScaler
        """

        if X.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

        self.scaler.fit(X)

        logger.info(
            "%s scaler fitted.",
            self.method.upper(),
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform features.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        transformed = self.scaler.transform(X)

        logger.info(
            "Feature transformation completed."
        )

        return pd.DataFrame(
            transformed,
            columns=X.columns,
            index=X.index,
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit and transform features.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        transformed = self.scaler.fit_transform(X)

        logger.info(
            "%s scaling applied.",
            self.method.upper(),
        )

        return pd.DataFrame(
            transformed,
            columns=X.columns,
            index=X.index,
        )

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @classmethod
    def available_scalers(cls) -> list[str]:
        """
        Return supported scalers.
        """

        return list(
            cls.SUPPORTED_SCALERS.keys()
        )

    def get_scaler(self):
        """
        Return underlying sklearn scaler.
        """

        return self.scaler

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print scaler configuration.
        """

        print("\n" + "=" * 60)
        print("FEATURE SCALER SUMMARY")
        print("=" * 60)

        print(f"Scaler : {self.method.upper()}")

        print("\nAvailable Scalers")

        for scaler in self.available_scalers():
            print(f"  • {scaler.upper()}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"FeatureScaler("
            f"method='{self.method}')"
        )
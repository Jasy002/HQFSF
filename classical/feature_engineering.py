"""
Feature Engineering Module for HQFSF.

Provides utilities for:

    - Variance Threshold Feature Selection
    - Polynomial Feature Generation
    - Principal Component Analysis (PCA)
    - Feature Information Summary
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineering:
    """
    Feature Engineering Utility.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input feature matrix.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

        self.df = dataframe.copy()

        logger.info(
            "FeatureEngineering initialized | "
            "Rows=%d | Columns=%d",
            self.df.shape[0],
            self.df.shape[1],
        )

    # ---------------------------------------------------------
    # Variance Threshold
    # ---------------------------------------------------------

    def variance_threshold(
        self,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Remove low-variance features.

        Parameters
        ----------
        threshold : float, default=0.0

        Returns
        -------
        pd.DataFrame
        """

        selector = VarianceThreshold(
            threshold=threshold
        )

        transformed = selector.fit_transform(
            self.df
        )

        columns = self.df.columns[
            selector.get_support()
        ]

        logger.info(
            "Variance threshold applied | "
            "Remaining Features=%d",
            len(columns),
        )

        return pd.DataFrame(
            transformed,
            columns=columns,
            index=self.df.index,
        )

    # ---------------------------------------------------------
    # Polynomial Features
    # ---------------------------------------------------------

    def polynomial_features(
        self,
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """
        Generate polynomial features.
        """

        transformer = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
        )

        transformed = transformer.fit_transform(
            self.df
        )

        columns = transformer.get_feature_names_out(
            self.df.columns
        )

        logger.info(
            "Polynomial features generated | "
            "Degree=%d",
            degree,
        )

        return pd.DataFrame(
            transformed,
            columns=columns,
            index=self.df.index,
        )

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------

    def pca(
        self,
        n_components: int | float = 0.95,
    ) -> pd.DataFrame:
        """
        Perform Principal Component Analysis.

        Parameters
        ----------
        n_components

            int
                Number of components.

            float
                Explained variance ratio.

        Returns
        -------
        pd.DataFrame
        """

        model = PCA(
            n_components=n_components,
            random_state=42,
        )

        transformed = model.fit_transform(
            self.df
        )

        columns = [
            f"PC{i+1}"
            for i in range(
                transformed.shape[1]
            )
        ]

        logger.info(
            "PCA completed | Components=%d",
            transformed.shape[1],
        )

        return pd.DataFrame(
            transformed,
            columns=columns,
            index=self.df.index,
        )

    # ---------------------------------------------------------
    # Feature Statistics
    # ---------------------------------------------------------

    def feature_statistics(self) -> Dict:
        """
        Return feature statistics.
        """

        return {

            "samples": self.df.shape[0],

            "features": self.df.shape[1],

            "missing_values": int(
                self.df.isnull().sum().sum()
            ),

            "duplicate_rows": int(
                self.df.duplicated().sum()
            ),

        }

    # ---------------------------------------------------------
    # Feature Names
    # ---------------------------------------------------------

    def feature_names(self) -> List[str]:
        """
        Return feature names.
        """

        return self.df.columns.tolist()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print feature engineering summary.
        """

        report = self.feature_statistics()

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)

        print(f"Samples  : {report['samples']}")
        print(f"Features : {report['features']}")
        print(f"Missing  : {report['missing_values']}")
        print(f"Duplicates : {report['duplicate_rows']}")

        print("\nFeature Names")

        for feature in self.feature_names():
            print(f"  • {feature}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Access
    # ---------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return dataframe copy.
        """

        return self.df.copy()

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"FeatureEngineering("
            f"rows={self.df.shape[0]}, "
            f"features={self.df.shape[1]})"
        )
"""
Data Preprocessing Module for HQFSF.

Performs:
    - Column name normalization
    - Duplicate removal
    - Missing value imputation
    - Label encoding
    - Complete preprocessing pipeline
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Data Preprocessor for HQFSF.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Notes
    -----
    This class performs preprocessing before feature scaling
    and quantum feature encoding.
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
            "DataPreprocessor initialized | "
            "Rows=%d | Columns=%d",
            self.df.shape[0],
            self.df.shape[1],
        )

    # ---------------------------------------------------------
    # Column Names
    # ---------------------------------------------------------

    def normalize_column_names(self) -> "DataPreprocessor":
        """
        Normalize column names.

        Returns
        -------
        DataPreprocessor
        """

        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

        logger.info(
            "Column names normalized."
        )

        return self

    # ---------------------------------------------------------
    # Duplicate Removal
    # ---------------------------------------------------------

    def remove_duplicates(self) -> "DataPreprocessor":
        """
        Remove duplicate rows.

        Returns
        -------
        DataPreprocessor
        """

        before = len(self.df)

        self.df.drop_duplicates(
            inplace=True
        )

        removed = before - len(self.df)

        logger.info(
            "%d duplicate rows removed.",
            removed,
        )

        return self

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def handle_missing_values(
        self,
        numeric_strategy: str = "mean",
        categorical_strategy: str = "most_frequent",
    ) -> "DataPreprocessor":
        """
        Fill missing values.

        Parameters
        ----------
        numeric_strategy : str
            Strategy for numeric columns.

        categorical_strategy : str
            Strategy for categorical columns.
        """

        numeric_columns = self.df.select_dtypes(
            include=["number"]
        ).columns

        categorical_columns = self.df.select_dtypes(
            exclude=["number"]
        ).columns

        if len(numeric_columns):

            numeric_imputer = SimpleImputer(
                strategy=numeric_strategy
            )

            self.df[numeric_columns] = (
                numeric_imputer.fit_transform(
                    self.df[numeric_columns]
                )
            )

        if len(categorical_columns):

            categorical_imputer = SimpleImputer(
                strategy=categorical_strategy
            )

            self.df[categorical_columns] = (
                categorical_imputer.fit_transform(
                    self.df[categorical_columns]
                )
            )

        logger.info(
            "Missing values handled."
        )

        return self

    # ---------------------------------------------------------
    # Label Encoding
    # ---------------------------------------------------------

    def encode_labels(self) -> "DataPreprocessor":
        """
        Encode categorical columns.

        Returns
        -------
        DataPreprocessor
        """

        categorical_columns = self.df.select_dtypes(
            exclude=["number"]
        ).columns

        for column in categorical_columns:

            encoder = LabelEncoder()

            self.df[column] = encoder.fit_transform(
                self.df[column].astype(str)
            )

        logger.info(
            "Categorical columns encoded."
        )

        return self

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    def preprocess(self) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
        """

        (
            self.normalize_column_names()
                .remove_duplicates()
                .handle_missing_values()
                .encode_labels()
        )

        logger.info(
            "Preprocessing completed successfully."
        )

        return self.df

    # ---------------------------------------------------------
    # Access
    # ---------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return processed dataframe.
        """

        return self.df.copy()

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print preprocessing summary.
        """

        print("\n" + "=" * 60)
        print("DATA PREPROCESSOR SUMMARY")
        print("=" * 60)

        print(f"Rows    : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")

        print("\nColumn Names")

        for column in self.df.columns:
            print(f"  • {column}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"DataPreprocessor("
            f"rows={self.df.shape[0]}, "
            f"columns={self.df.shape[1]})"
        )
"""
Validation module for HQFSF.

Performs dataset validation before preprocessing.
"""

from typing import Dict

import pandas as pd

from utils.exceptions import DatasetError
from utils.logger import setup_logger

logger = setup_logger()


class DataValidator:
    """
    Performs validation checks on datasets.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def check_empty(self) -> None:
        """Check whether dataset is empty."""

        if self.df.empty:
            raise DatasetError("Dataset is empty.")

        logger.info("Dataset is not empty.")

    def check_target(self, target_column: str) -> None:
        """Verify target column exists."""

        if target_column not in self.df.columns:
            raise DatasetError(
                f"Target column '{target_column}' not found."
            )

        logger.info("Target column validated.")

    def missing_values(self) -> Dict:
        """Return missing value statistics."""

        missing = self.df.isnull().sum()

        logger.info("Missing value analysis completed.")

        return missing.to_dict()

    def duplicate_rows(self) -> int:
        """Count duplicate rows."""

        duplicates = self.df.duplicated().sum()

        logger.info(f"Duplicate rows: {duplicates}")

        return duplicates

    def numeric_columns(self):
        """Return numeric feature names."""

        return self.df.select_dtypes(include=["number"]).columns.tolist()

    def categorical_columns(self):
        """Return categorical feature names."""

        return self.df.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

    def validation_report(self, target_column: str) -> Dict:
        """
        Generate validation report.
        """

        self.check_empty()

        self.check_target(target_column)

        report = {

            "Rows": self.df.shape[0],

            "Columns": self.df.shape[1],

            "Missing Values": self.missing_values(),

            "Duplicate Rows": self.duplicate_rows(),

            "Numeric Columns": self.numeric_columns(),

            "Categorical Columns": self.categorical_columns()

        }

        logger.info("Validation completed successfully.")

        return report
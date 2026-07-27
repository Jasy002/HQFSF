"""
Data Validation Module for HQFSF.

Performs:
    - Dataset validation
    - Target validation
    - Missing value analysis
    - Duplicate detection
    - Data type inspection
    - Dataset statistics
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils.exceptions import DatasetError
from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Dataset validation utility.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        self.df = dataframe.copy()

        logger.info(
            "DataValidator initialized | "
            "Rows=%d | Columns=%d",
            self.df.shape[0],
            self.df.shape[1],
        )

    # ---------------------------------------------------------
    # Dataset Checks
    # ---------------------------------------------------------

    def check_empty(self) -> None:
        """
        Verify dataset is not empty.
        """

        if self.df.empty:
            raise DatasetError(
                "Dataset is empty."
            )

        logger.info(
            "Dataset is not empty."
        )

    def check_target(
        self,
        target_column: str,
    ) -> None:
        """
        Verify target column exists.
        """

        if target_column not in self.df.columns:

            raise DatasetError(
                f"Target column '{target_column}' not found."
            )

        logger.info(
            "Target column validated."
        )

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def missing_values(self) -> Dict[str, int]:
        """
        Return missing value statistics.
        """

        missing = (
            self.df
            .isnull()
            .sum()
            .to_dict()
        )

        logger.info(
            "Missing value analysis completed."
        )

        return missing

    # ---------------------------------------------------------
    # Duplicate Rows
    # ---------------------------------------------------------

    def duplicate_rows(self) -> int:
        """
        Count duplicate rows.
        """

        duplicates = int(
            self.df.duplicated().sum()
        )

        logger.info(
            "Duplicate rows: %d",
            duplicates,
        )

        return duplicates

    # ---------------------------------------------------------
    # Column Information
    # ---------------------------------------------------------

    def numeric_columns(self) -> List[str]:
        """
        Return numeric column names.
        """

        return self.df.select_dtypes(
            include="number"
        ).columns.tolist()

    def categorical_columns(self) -> List[str]:
        """
        Return categorical column names.
        """

        return self.df.select_dtypes(
            exclude="number"
        ).columns.tolist()

    def data_types(self) -> Dict[str, str]:
        """
        Return data types.
        """

        return {
            column: str(dtype)
            for column, dtype in self.df.dtypes.items()
        }

    # ---------------------------------------------------------
    # Dataset Statistics
    # ---------------------------------------------------------

    def dataset_shape(self) -> tuple[int, int]:
        """
        Return dataset shape.
        """

        return self.df.shape

    def memory_usage(self) -> float:
        """
        Return dataset memory usage in MB.
        """

        memory = (
            self.df.memory_usage(
                deep=True
            ).sum()
            / (1024 ** 2)
        )

        return round(memory, 3)

    # ---------------------------------------------------------
    # Validation Report
    # ---------------------------------------------------------

    def validation_report(
        self,
        target_column: str,
    ) -> Dict:

        self.check_empty()

        self.check_target(
            target_column
        )

        report = {

            "rows":
                self.df.shape[0],

            "columns":
                self.df.shape[1],

            "memory_mb":
                self.memory_usage(),

            "missing_values":
                self.missing_values(),

            "duplicate_rows":
                self.duplicate_rows(),

            "numeric_columns":
                self.numeric_columns(),

            "categorical_columns":
                self.categorical_columns(),

            "data_types":
                self.data_types(),

        }

        logger.info(
            "Validation report generated."
        )

        return report

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        target_column: str,
    ) -> None:
        """
        Print validation summary.
        """

        report = self.validation_report(
            target_column
        )

        print("\n" + "=" * 60)
        print("DATA VALIDATION SUMMARY")
        print("=" * 60)

        print(f"Rows          : {report['rows']}")
        print(f"Columns       : {report['columns']}")
        print(f"Memory (MB)   : {report['memory_mb']:.3f}")
        print(f"Duplicates    : {report['duplicate_rows']}")

        print("\nNumeric Columns")

        for column in report["numeric_columns"]:
            print(f"  • {column}")

        print("\nCategorical Columns")

        for column in report["categorical_columns"]:
            print(f"  • {column}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"DataValidator("
            f"rows={self.df.shape[0]}, "
            f"columns={self.df.shape[1]})"
        )
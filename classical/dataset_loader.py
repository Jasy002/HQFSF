"""
Dataset Loader Module for HQFSF.

Supports loading datasets from:

    - CSV
    - Excel
    - JSON
    - Parquet
    - Built-in scikit-learn datasets

All methods return a pandas DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """
    Dataset Loader Utility.

    Supports loading datasets from multiple file formats
    and built-in scikit-learn datasets.
    """

    SUPPORTED_BUILTIN = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
    }

    # ---------------------------------------------------------
    # Built-in Dataset
    # ---------------------------------------------------------

    def load_builtin(
        self,
        dataset_name: str,
    ) -> pd.DataFrame:
        """
        Load a built-in scikit-learn dataset.

        Parameters
        ----------
        dataset_name : str

        Returns
        -------
        pd.DataFrame
        """

        dataset_name = dataset_name.lower()

        if dataset_name not in self.SUPPORTED_BUILTIN:

            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Available: {list(self.SUPPORTED_BUILTIN.keys())}"
            )

        dataset = self.SUPPORTED_BUILTIN[
            dataset_name
        ]()

        df = pd.DataFrame(
            dataset.data,
            columns=dataset.feature_names,
        )

        df["target"] = dataset.target

        logger.info(
            "Built-in dataset loaded: %s",
            dataset_name,
        )

        return df

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    def load_csv(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load CSV dataset.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_csv(
            file_path,
            **kwargs,
        )

        logger.info(
            "CSV loaded: %s",
            file_path.name,
        )

        return df

    # ---------------------------------------------------------
    # Excel
    # ---------------------------------------------------------

    def load_excel(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load Excel dataset.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            **kwargs,
        )

        logger.info(
            "Excel loaded: %s",
            file_path.name,
        )

        return df

    # ---------------------------------------------------------
    # JSON
    # ---------------------------------------------------------

    def load_json(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load JSON dataset.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_json(
            file_path,
            **kwargs,
        )

        logger.info(
            "JSON loaded: %s",
            file_path.name,
        )

        return df

    # ---------------------------------------------------------
    # Parquet
    # ---------------------------------------------------------

    def load_parquet(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load Parquet dataset.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_parquet(
            file_path,
            **kwargs,
        )

        logger.info(
            "Parquet loaded: %s",
            file_path.name,
        )

        return df

    # ---------------------------------------------------------
    # Auto Loader
    # ---------------------------------------------------------

    def load(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Automatically detect dataset type
        from file extension.
        """

        file_path = Path(file_path)

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return self.load_csv(
                file_path,
                **kwargs,
            )

        if suffix in (".xlsx", ".xls"):
            return self.load_excel(
                file_path,
                **kwargs,
            )

        if suffix == ".json":
            return self.load_json(
                file_path,
                **kwargs,
            )

        if suffix == ".parquet":
            return self.load_parquet(
                file_path,
                **kwargs,
            )

        raise ValueError(
            f"Unsupported file format: {suffix}"
        )

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @classmethod
    def supported_formats(cls) -> List[str]:
        """
        Return supported file formats.
        """

        return [
            ".csv",
            ".xlsx",
            ".xls",
            ".json",
            ".parquet",
        ]

    @classmethod
    def builtin_datasets(cls) -> List[str]:
        """
        Return available built-in datasets.
        """

        return sorted(
            cls.SUPPORTED_BUILTIN.keys()
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print loader information.
        """

        print("\n" + "=" * 60)
        print("DATASET LOADER SUMMARY")
        print("=" * 60)

        print("Supported File Formats")

        for fmt in self.supported_formats():
            print(f"  • {fmt}")

        print("\nBuilt-in Datasets")

        for dataset in self.builtin_datasets():
            print(f"  • {dataset}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"formats={len(self.supported_formats())}, "
            f"builtin={len(self.SUPPORTED_BUILTIN)})"
        )
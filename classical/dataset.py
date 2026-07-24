"""
Dataset loading utilities for HQFSF.

Supports:
1. Built-in sklearn datasets
2. CSV datasets
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Utility class for loading datasets."""

    @staticmethod
    def load_builtin(name: str) -> pd.DataFrame:
        """
        Load a built-in sklearn dataset.

        Parameters
        ----------
        name : str
            Dataset name: breast_cancer, iris, wine

        Returns
        -------
        pandas.DataFrame
        """

        datasets = {
            "breast_cancer": load_breast_cancer,
            "iris": load_iris,
            "wine": load_wine,
        }

        if name not in datasets:
            raise ValueError(
                f"Unsupported dataset '{name}'. "
                f"Available: {list(datasets.keys())}"
            )

        dataset = datasets[name]()

        df = pd.DataFrame(
            dataset.data,
            columns=dataset.feature_names
        )

        df["target"] = dataset.target

        logger.info(
            "Loaded built-in dataset '%s' (%d samples, %d features)",
            name,
            df.shape[0],
            df.shape[1] - 1,
        )

        return df

    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        """
        Load dataset from CSV.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        logger.info(
            "Loaded CSV '%s' (%d samples, %d columns)",
            path.name,
            df.shape[0],
            df.shape[1],
        )

        return df
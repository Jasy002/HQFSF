"""
Dataset Splitting Module for HQFSF.

Supports:
    - Train-Test Split
    - Train-Validation-Test Split
    - Stratified Sampling
    - Split Summary
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Dataset splitting utility.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of data used for testing.

    random_state : int, default=42
        Random seed for reproducibility.

    stratify : bool, default=True
        Whether to use stratified sampling.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:

        if not 0 < test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        logger.info(
            "DataSplitter initialized | "
            "test_size=%.2f | random_state=%d",
            self.test_size,
            self.random_state,
        )

    # ---------------------------------------------------------
    # Train-Test Split
    # ---------------------------------------------------------

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Perform train-test split.
        """

        logger.info("Performing train-test split.")

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

    # ---------------------------------------------------------
    # Train-Validation-Test Split
    # ---------------------------------------------------------

    def train_validation_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_size: float = 0.1,
    ):
        """
        Perform train-validation-test split.
        """

        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        logger.info(
            "Performing train-validation-test split."
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

        validation_ratio = validation_size / (
            1 - self.test_size
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_ratio,
            random_state=self.random_state,
            stratify=y_train if self.stratify else None,
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    @staticmethod
    def summary(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, int]:
        """
        Return split statistics.
        """

        report = {
            "Training Samples": len(X_train),
            "Testing Samples": len(X_test),
            "Training Labels": len(y_train),
            "Testing Labels": len(y_test),
            "Features": X_train.shape[1],
        }

        logger.info("Split summary generated.")

        return report

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"DataSplitter("
            f"test_size={self.test_size}, "
            f"random_state={self.random_state}, "
            f"stratify={self.stratify})"
        )
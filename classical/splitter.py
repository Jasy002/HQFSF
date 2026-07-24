"""
Dataset splitting utilities for HQFSF.
"""

from __future__ import annotations

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Handles train/test splitting and cross-validation.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """
        Perform stratified train-test split.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "Train/Test split completed | Train=%d | Test=%d",
            len(X_train),
            len(X_test),
        )

        return X_train, X_test, y_train, y_test

    def kfold(self, n_splits: int = 5):
        """
        Standard K-Fold cross-validation.
        """

        logger.info(
            "Created KFold (%d splits)",
            n_splits,
        )

        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    def stratified_kfold(self, n_splits: int = 5):
        """
        Stratified K-Fold cross-validation.
        """

        logger.info(
            "Created StratifiedKFold (%d splits)",
            n_splits,
        )

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
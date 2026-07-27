"""
Cross Validation Module for HQFSF.

Supports:
    - Stratified K-Fold
    - K-Fold
    - Repeated Stratified K-Fold
    - Cross-validation scoring
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """
    Cross-validation utility for HQFSF.

    Parameters
    ----------
    folds : int, default=5
        Number of folds.

    strategy : str, default="stratified"

        Supported strategies:

        - stratified
        - kfold
        - repeated

    random_state : int, default=42
        Random seed.

    shuffle : bool, default=True
        Whether to shuffle samples.
    """

    SUPPORTED_STRATEGIES = (
        "stratified",
        "kfold",
        "repeated",
    )

    def __init__(
        self,
        folds: int = 5,
        strategy: str = "stratified",
        random_state: int = 42,
        shuffle: bool = True,
        repeats: int = 3,
    ) -> None:

        if folds < 2:
            raise ValueError(
                "folds must be at least 2."
            )

        self.folds = folds
        self.strategy = strategy.lower()
        self.random_state = random_state
        self.shuffle = shuffle
        self.repeats = repeats

        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )

        self.cv = self._build_cv()

        logger.info(
            "CrossValidator initialized | "
            "Strategy=%s | Folds=%d",
            self.strategy.upper(),
            self.folds,
        )

    # ---------------------------------------------------------
    # Build Cross Validator
    # ---------------------------------------------------------

    def _build_cv(self):

        if self.strategy == "stratified":

            return StratifiedKFold(
                n_splits=self.folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        if self.strategy == "kfold":

            return KFold(
                n_splits=self.folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        return RepeatedStratifiedKFold(
            n_splits=self.folds,
            n_repeats=self.repeats,
            random_state=self.random_state,
        )

    # ---------------------------------------------------------
    # Split Generator
    # ---------------------------------------------------------

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Generate train/test indices.
        """

        logger.info(
            "Generating cross-validation splits."
        )

        return self.cv.split(X, y)

    # ---------------------------------------------------------
    # Cross Validation Score
    # ---------------------------------------------------------

    def evaluate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "accuracy",
    ) -> np.ndarray:
        """
        Evaluate a model using cross-validation.
        """

        logger.info(
            "Running %s cross-validation.",
            scoring,
        )

        scores = cross_val_score(
            estimator=model,
            X=X,
            y=y,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
        )

        return scores

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    @staticmethod
    def statistics(
        scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Return summary statistics.
        """

        return {

            "mean": float(np.mean(scores)),

            "std": float(np.std(scores)),

            "min": float(np.min(scores)),

            "max": float(np.max(scores)),

            "folds": len(scores),
        }

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        scores: np.ndarray,
    ) -> None:
        """
        Print cross-validation summary.
        """

        report = self.statistics(scores)

        print("\n" + "=" * 60)
        print("CROSS VALIDATION SUMMARY")
        print("=" * 60)

        print(f"Strategy : {self.strategy.upper()}")
        print(f"Folds    : {self.folds}")
        print(f"Mean     : {report['mean']:.4f}")
        print(f"Std Dev  : {report['std']:.4f}")
        print(f"Minimum  : {report['min']:.4f}")
        print(f"Maximum  : {report['max']:.4f}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @classmethod
    def available_strategies(cls) -> List[str]:
        """
        Return supported strategies.
        """

        return list(cls.SUPPORTED_STRATEGIES)

    def get_validator(self):
        """
        Return the underlying sklearn validator.
        """

        return self.cv

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"CrossValidator("
            f"strategy='{self.strategy}', "
            f"folds={self.folds}, "
            f"shuffle={self.shuffle})"
        )
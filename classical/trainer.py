"""
Model Training Module for HQFSF.

Provides utilities for:

    - Model Training
    - Prediction
    - Probability Prediction
    - Model Evaluation Preparation
    - Model Saving
    - Model Loading
    - Training Statistics
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Train and manage classical machine learning models.

    Parameters
    ----------
    model : BaseEstimator
        Any scikit-learn compatible classifier.
    """

    def __init__(
        self,
        model: BaseEstimator,
    ) -> None:

        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "model must inherit from sklearn.base.BaseEstimator."
            )

        self.model = model

        self.training_time = 0.0

        self.is_trained = False

        logger.info(
            "ModelTrainer initialized | %s",
            self.model.__class__.__name__,
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "ModelTrainer":
        """
        Train the classifier.
        """

        start = time.perf_counter()

        self.model.fit(
            X_train,
            y_train,
        )

        self.training_time = (
            time.perf_counter() - start
        )

        self.is_trained = True

        logger.info(
            "%s trained in %.4f seconds.",
            self.model.__class__.__name__,
            self.training_time,
        )

        return self

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        self._check_training()

        return self.model.predict(X_test)

    # ---------------------------------------------------------
    # Probability Prediction
    # ---------------------------------------------------------

    def predict_proba(
        self,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        self._check_training()

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            raise AttributeError(
                f"{self.model.__class__.__name__} "
                "does not support predict_proba()."
            )

        return self.model.predict_proba(
            X_test
        )

    # ---------------------------------------------------------
    # Decision Function
    # ---------------------------------------------------------

    def decision_function(
        self,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return confidence scores.
        """

        self._check_training()

        if not hasattr(
            self.model,
            "decision_function",
        ):
            raise AttributeError(
                f"{self.model.__class__.__name__} "
                "does not support decision_function()."
            )

        return self.model.decision_function(
            X_test
        )

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def save(
        self,
        filepath: str | Path,
    ) -> None:
        """
        Save trained model.
        """

        self._check_training()

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            filepath,
        )

        logger.info(
            "Model saved to %s",
            filepath,
        )

    @staticmethod
    def load(
        filepath: str | Path,
    ) -> BaseEstimator:
        """
        Load trained model.
        """

        filepath = Path(filepath)

        model = joblib.load(filepath)

        logger.info(
            "Model loaded from %s",
            filepath,
        )

        return model

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def training_statistics(self) -> dict[str, Any]:
        """
        Return training information.
        """

        return {

            "model": self.model.__class__.__name__,

            "training_time_seconds":
                round(
                    self.training_time,
                    4,
                ),

            "trained":
                self.is_trained,

        }

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def _check_training(self) -> None:
        """
        Verify model has been trained.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

    def get_model(self) -> BaseEstimator:
        """
        Return trained model.
        """

        return self.model

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print training summary.
        """

        stats = self.training_statistics()

        print("\n" + "=" * 60)
        print("MODEL TRAINER SUMMARY")
        print("=" * 60)

        print(f"Model         : {stats['model']}")
        print(f"Trained       : {stats['trained']}")
        print(
            f"Training Time : "
            f"{stats['training_time_seconds']} sec"
        )

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"ModelTrainer("
            f"model={self.model.__class__.__name__}, "
            f"trained={self.is_trained})"
        )
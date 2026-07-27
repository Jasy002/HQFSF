"""
Random Forest Model for HQFSF.

Implements a Random Forest classifier using
scikit-learn.
"""

from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):

        super().__init__()

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        logger.info(
            "RandomForestModel initialized."
        )

    # ----------------------------------------------------------
    # Train Model
    # ----------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the Random Forest model.
        """

        logger.info(
            "Training Random Forest..."
        )

        self.model.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Training completed."
        )

    # ----------------------------------------------------------
    # Predict
    # ----------------------------------------------------------

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        logger.info(
            "Generating predictions..."
        )

        return self.model.predict(
            X_test
        )

    # ----------------------------------------------------------
    # Feature Importance
    # ----------------------------------------------------------

    def feature_importance(
        self,
    ) -> np.ndarray:
        """
        Return feature importance scores.
        """

        return self.model.feature_importances_

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print(" Random Forest Model ")
        print("=" * 60)

        params = self.model.get_params()

        for key, value in params.items():
            print(f"{key:20}: {value}")

        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        return (
            "RandomForestModel("
            f"n_estimators={self.model.n_estimators}, "
            f"criterion='{self.model.criterion}')"
        )
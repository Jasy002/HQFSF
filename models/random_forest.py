"""
Random Forest Classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.classifier import Classifier
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(Classifier):
    """
    Random Forest Classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        super().__init__()

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
        )

        logger.info("Random Forest initialized.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the Random Forest model.
        """

        logger.info("Training Random Forest...")

        self.model.fit(X_train, y_train)

        logger.info("Training completed.")

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        logger.info("Predicting labels...")

        return self.model.predict(X_test)

    def predict_proba(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        return self.model.predict_proba(X_test)

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Compute model accuracy.
        """

        accuracy = self.model.score(X_test, y_test)

        logger.info(f"Accuracy: {accuracy:.4f}")

        return accuracy

    def feature_importance(self) -> np.ndarray:
        """
        Return feature importance scores.
        """

        return self.model.feature_importances_

    def summary(self) -> None:
        """
        Display model configuration.
        """

        print("\n" + "=" * 60)
        print(" Random Forest Classifier ")
        print("=" * 60)

        print(f"Trees        : {self.model.n_estimators}")
        print(f"Criterion    : {self.model.criterion}")
        print(f"Max Depth    : {self.model.max_depth}")
        print(f"Random State : {self.model.random_state}")

        print("=" * 60 + "\n")
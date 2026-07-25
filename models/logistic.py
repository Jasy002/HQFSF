"""
Logistic Regression Classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.classifier import Classifier
from utils.logger import get_logger

logger = get_logger(__name__)


class LogisticClassifier(Classifier):
    """
    Logistic Regression Classifier.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        solver: str = "lbfgs",
    ):
        super().__init__()

        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver=solver,
        )

        logger.info("Logistic Regression initialized.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the classifier.
        """

        logger.info("Training Logistic Regression...")

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
        Compute accuracy.
        """

        accuracy = self.model.score(X_test, y_test)

        logger.info(f"Accuracy: {accuracy:.4f}")

        return accuracy

    def summary(self) -> None:
        """
        Print model information.
        """

        print("\n" + "=" * 60)
        print(" Logistic Regression Classifier ")
        print("=" * 60)

        print(f"Solver      : {self.model.solver}")
        print(f"Max Iter    : {self.model.max_iter}")
        print(f"RandomState : {self.model.random_state}")

        print("=" * 60 + "\n")
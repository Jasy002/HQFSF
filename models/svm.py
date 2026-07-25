"""
Support Vector Machine Classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from models.classifier import Classifier
from utils.logger import get_logger

logger = get_logger(__name__)


class SVMClassifier(Classifier):
    """
    Support Vector Machine Classifier.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        probability: bool = True,
        random_state: int = 42,
    ):
        super().__init__()

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )

        logger.info("Support Vector Machine initialized.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the classifier.
        """

        logger.info("Training SVM...")

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
        print(" Support Vector Machine ")
        print("=" * 60)

        print(f"Kernel      : {self.model.kernel}")
        print(f"C           : {self.model.C}")
        print(f"Gamma       : {self.model.gamma}")
        print(f"Probability : {self.model.probability}")

        print("=" * 60 + "\n")
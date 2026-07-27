"""
Support Vector Machine (SVM) Model for HQFSF.

Implements an SVM classifier using scikit-learn.
"""

from __future__ import annotations

import numpy as np

from sklearn.svm import SVC

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class SVMModel(BaseModel):
    """
    Support Vector Machine Classifier.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        degree: int = 3,
        probability: bool = True,
        random_state: int = 42,
    ):

        super().__init__()

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=probability,
            random_state=random_state,
        )

        logger.info(
            "SVMModel initialized."
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
        Train the SVM classifier.
        """

        logger.info(
            "Training SVM..."
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
    # Decision Function
    # ----------------------------------------------------------

    def decision_function(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distance of samples to the decision boundary.
        """

        return self.model.decision_function(
            X_test
        )

    # ----------------------------------------------------------
    # Support Vectors
    # ----------------------------------------------------------

    def support_vectors(
        self,
    ) -> np.ndarray:
        """
        Return support vectors.
        """

        return self.model.support_vectors_

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print(" Support Vector Machine Model ")
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
            "SVMModel("
            f"kernel='{self.model.kernel}', "
            f"C={self.model.C})"
        )
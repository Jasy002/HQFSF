"""
Logistic Regression Model for HQFSF.

Implements a Logistic Regression classifier using
scikit-learn.
"""

from __future__ import annotations

import numpy as np

from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression Classifier.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
    ):

        super().__init__()

        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

        logger.info(
            "LogisticRegressionModel initialized."
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
        Train the Logistic Regression model.
        """

        logger.info(
            "Training Logistic Regression..."
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
    # Predict Probability
    # ----------------------------------------------------------

    def predict_probability(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        return self.model.predict_proba(
            X_test
        )

    # ----------------------------------------------------------
    # Model Coefficients
    # ----------------------------------------------------------

    def coefficients(
        self,
    ) -> np.ndarray:
        """
        Return model coefficients.
        """

        return self.model.coef_

    # ----------------------------------------------------------
    # Model Intercept
    # ----------------------------------------------------------

    def intercept(
        self,
    ) -> np.ndarray:
        """
        Return intercept.
        """

        return self.model.intercept_

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print(" Logistic Regression Model ")
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
            "LogisticRegressionModel("
            f"solver='{self.model.solver}', "
            f"C={self.model.C})"
        )
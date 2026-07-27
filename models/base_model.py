"""
Base Model for HQFSF.

Defines the common interface for all machine learning
models used after quantum feature selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any
import numpy as np


class BaseModel(ABC):
    """
    Abstract Base Class for all HQFSF classifiers.
    """

    def __init__(self):

        self.model = None

    # ----------------------------------------------------------
    # Train Model
    # ----------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the classifier.
        """
        pass

    # ----------------------------------------------------------
    # Predict Labels
    # ----------------------------------------------------------

    @abstractmethod
    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels.
        """
        pass

    # ----------------------------------------------------------
    # Predict Probabilities
    # ----------------------------------------------------------

    def predict_proba(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)

        raise NotImplementedError(
            "Probability prediction is not supported "
            "by this model."
        )

    # ----------------------------------------------------------
    # Model Score
    # ----------------------------------------------------------

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> float:
        """
        Compute model accuracy.
        """

        if self.model is None:
            raise ValueError(
                "Model has not been trained."
            )

        return self.model.score(
            X_test,
            y_test,
        )

    # ----------------------------------------------------------
    # Get Parameters
    # ----------------------------------------------------------

    def get_params(self) -> dict:
        """
        Return model parameters.
        """

        if self.model is None:
            return {}

        return self.model.get_params()

    # ----------------------------------------------------------
    # Set Parameters
    # ----------------------------------------------------------

    def set_params(
        self,
        **params: Any,
    ) -> None:
        """
        Update model parameters.
        """

        if self.model is None:
            raise ValueError(
                "Model has not been initialized."
            )

        self.model.set_params(
            **params
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self) -> None:
        """
        Print model information.
        """

        print("\n" + "=" * 60)
        print(" HQFSF Machine Learning Model ")
        print("=" * 60)

        if self.model is None:

            print("Model : Not initialized")

        else:

            print(
                f"Model : {self.model.__class__.__name__}"
            )

            print(
                f"Parameters : {self.model.get_params()}"
            )

        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        if self.model is None:

            return (
                f"{self.__class__.__name__}(uninitialized)"
            )

        return (
            f"{self.__class__.__name__}"
            f"({self.model})"
        )
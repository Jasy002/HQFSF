"""
Abstract Base Classifier.

Defines the common interface for all machine learning
classifiers used in the HQFSF framework.
"""

from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """
    Abstract base class for all classifiers.
    """

    def __init__(self):
        self.model = None

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train the classifier.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X_test: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels.
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X_test: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """
        pass

    @abstractmethod
    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> float:
        """
        Return model accuracy.
        """
        pass

    @abstractmethod
    def summary(self) -> None:
        """
        Print model information.
        """
        pass

    def is_trained(self) -> bool:
        """
        Check whether the model has been trained.
        """
        return self.model is not None
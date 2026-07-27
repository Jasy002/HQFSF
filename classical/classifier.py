"""
Classical Classifier Module for HQFSF.

Provides a unified interface for creating classical machine learning
models used for comparison with the Hybrid Quantum Feature Selection
Framework (HQFSF).

Supported Models
----------------
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.logger import get_logger

logger = get_logger(__name__)


class ClassicalClassifier:
    """
    Factory class for classical machine learning models.
    """

    SUPPORTED_MODELS = {
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "random_forest": RandomForestClassifier,
        "decision_tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
        "naive_bayes": GaussianNB,
    }

    def __init__(self) -> None:

        logger.info(
            "ClassicalClassifier initialized."
        )

    # ---------------------------------------------------------
    # Model Factory
    # ---------------------------------------------------------

    def create(
        self,
        model_name: str,
        **kwargs,
    ) -> BaseEstimator:
        """
        Create a machine learning classifier.

        Parameters
        ----------
        model_name : str
            Name of classifier.

        **kwargs
            Hyperparameters passed to sklearn model.

        Returns
        -------
        BaseEstimator
        """

        model_name = model_name.lower()

        if model_name not in self.SUPPORTED_MODELS:

            raise ValueError(
                f"Unsupported classifier '{model_name}'. "
                f"Supported models: "
                f"{list(self.SUPPORTED_MODELS.keys())}"
            )

        defaults = {
            "random_state": 42
        }

        if model_name in (
            "knn",
            "naive_bayes",
        ):
            defaults = {}

        defaults.update(kwargs)

        model = self.SUPPORTED_MODELS[
            model_name
        ](**defaults)

        logger.info(
            "%s classifier created.",
            model_name.upper(),
        )

        return model

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @classmethod
    def available_models(cls) -> List[str]:
        """
        Return supported classifiers.
        """

        return sorted(
            cls.SUPPORTED_MODELS.keys()
        )

    @staticmethod
    def default_parameters() -> Dict:
        """
        Return recommended HQFSF hyperparameters.
        """

        return {

            "logistic_regression": {
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
            },

            "svm": {
                "kernel": "rbf",
                "probability": True,
                "random_state": 42,
            },

            "random_forest": {
                "n_estimators": 200,
                "max_depth": None,
                "random_state": 42,
            },

            "decision_tree": {
                "criterion": "gini",
                "random_state": 42,
            },

            "knn": {
                "n_neighbors": 5,
            },

            "naive_bayes": {},
        }

    def create_all(self) -> Dict[str, BaseEstimator]:
        """
        Create one instance of every supported classifier.

        Returns
        -------
        Dict[str, BaseEstimator]
        """

        models = {}

        defaults = self.default_parameters()

        for name in self.available_models():

            models[name] = self.create(
                name,
                **defaults.get(name, {})
            )

        logger.info(
            "%d classifiers created.",
            len(models),
        )

        return models

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print classifier summary.
        """

        print("\n" + "=" * 60)
        print("CLASSICAL CLASSIFIERS")
        print("=" * 60)

        for model in self.available_models():
            print(f"• {model}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(supported_models={len(self.SUPPORTED_MODELS)})"
        )
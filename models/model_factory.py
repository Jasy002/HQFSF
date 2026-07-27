"""
Model Factory for HQFSF.

Provides a unified interface for creating
machine learning models.
"""

from __future__ import annotations

from typing import Any

from models.random_forest import RandomForestModel
from models.svm import SVMModel
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel


class ModelFactory:
    """
    Factory class for creating machine learning models.
    """

    _MODELS = {
        "random_forest": RandomForestModel,
        "rf": RandomForestModel,

        "svm": SVMModel,
        "svc": SVMModel,

        "logistic_regression": LogisticRegressionModel,
        "logistic": LogisticRegressionModel,
        "lr": LogisticRegressionModel,

        "xgboost": XGBoostModel,
        "xgb": XGBoostModel,
    }

    # ----------------------------------------------------------
    # Create Model
    # ----------------------------------------------------------

    @classmethod
    def create(
        cls,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Create a machine learning model.

        Parameters
        ----------
        model_name : str
            Name of the classifier.

        Returns
        -------
        BaseModel
            Initialized classifier.
        """

        model_name = model_name.lower()

        if model_name not in cls._MODELS:

            supported = ", ".join(sorted(cls._MODELS.keys()))

            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {supported}"
            )

        return cls._MODELS[model_name](**kwargs)

    # ----------------------------------------------------------
    # Available Models
    # ----------------------------------------------------------

    @classmethod
    def available_models(cls):
        """
        Return available model names.
        """

        return sorted(cls._MODELS.keys())

    # ----------------------------------------------------------
    # Check Availability
    # ----------------------------------------------------------

    @classmethod
    def exists(
        cls,
        model_name: str,
    ) -> bool:

        return model_name.lower() in cls._MODELS

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        return (
            "ModelFactory("
            f"available={self.available_models()})"
        )
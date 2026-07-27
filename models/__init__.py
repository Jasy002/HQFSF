"""
Machine Learning Models for HQFSF.

This package provides the classical classifiers used
after quantum feature selection.
"""

from .base_model import BaseModel
from .random_forest import RandomForestModel
from .svm import SVMModel
from .logistic_regression import LogisticRegressionModel
from .xgboost_model import XGBoostModel
from .model_factory import ModelFactory

__all__ = [
    "BaseModel",
    "RandomForestModel",
    "SVMModel",
    "LogisticRegressionModel",
    "XGBoostModel",
    "ModelFactory",
]

__version__ = "1.0.0"
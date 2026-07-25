"""
Machine Learning Models Package.

Provides classical classifiers for HQFSF.
"""

from .classifier import Classifier
from .logistic import LogisticClassifier
from .svm import SVMClassifier
from .random_forest import RandomForestClassifier
from .model_factory import ModelFactory

__all__ = [
    "Classifier",
    "LogisticClassifier",
    "SVMClassifier",
    "RandomForestClassifier",
    "ModelFactory",
]

__version__ = "1.0.0"
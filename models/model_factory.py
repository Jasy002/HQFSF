"""
Model Factory.

Creates machine learning classifiers used in the HQFSF framework.
"""

from models.logistic import LogisticClassifier
from models.svm import SVMClassifier
from models.random_forest import RandomForestModel


class ModelFactory:
    """
    Factory class for creating classifiers.
    """

    @staticmethod
    def create(
        model_name: str,
        **kwargs
    ):
        """
        Create and return a classifier.

        Parameters
        ----------
        model_name : str
            Name of the classifier.

        Returns
        -------
        Classifier
            Initialized classifier instance.
        """

        model_name = model_name.lower()

        if model_name in ["logistic", "logistic_regression"]:
            return LogisticClassifier(**kwargs)

        elif model_name in ["svm", "svc"]:
            return SVMClassifier(**kwargs)

        elif model_name in [
            "random_forest",
            "rf",
            "forest"
        ]:
            return RandomForestModel(**kwargs)

        else:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

    @staticmethod
    def available_models():
        """
        Return all available models.
        """

        return [
            "logistic",
            "svm",
            "random_forest",
        ]
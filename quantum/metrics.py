"""
Evaluation Metrics for HQFSF.

Responsible for:
    - Classification Metrics
    - Feature Selection Statistics
    - Confusion Matrix
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumMetrics:
    """
    Evaluation metrics for HQFSF.
    """

    def __init__(self):

        logger.info(
            "QuantumMetrics initialized."
        )

    def accuracy(
        self,
        y_true,
        y_pred,
    ) -> float:

        score = accuracy_score(
            y_true,
            y_pred,
        )

        logger.info(
            "Accuracy: %.4f",
            score,
        )

        return score

    def precision(
        self,
        y_true,
        y_pred,
        average="binary",
    ) -> float:

        score = precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

        logger.info(
            "Precision: %.4f",
            score,
        )

        return score

    def recall(
        self,
        y_true,
        y_pred,
        average="binary",
    ) -> float:

        score = recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

        logger.info(
            "Recall: %.4f",
            score,
        )

        return score

    def f1(
        self,
        y_true,
        y_pred,
        average="binary",
    ) -> float:

        score = f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

        logger.info(
            "F1 Score: %.4f",
            score,
        )

        return score

    def confusion(
        self,
        y_true,
        y_pred,
    ):

        matrix = confusion_matrix(
            y_true,
            y_pred,
        )

        logger.info(
            "Confusion matrix computed."
        )

        return matrix

    def feature_reduction(
        self,
        original_features: int,
        selected_features: int,
    ) -> float:

        reduction = (
            (original_features - selected_features)
            / original_features
        ) * 100

        logger.info(
            "Feature Reduction: %.2f%%",
            reduction,
        )

        return reduction

    def summary(
        self,
        y_true,
        y_pred,
        original_features,
        selected_features,
    ):

        print("\n========== HQFSF Evaluation ==========\n")

        print(
            f"Accuracy          : "
            f"{self.accuracy(y_true, y_pred):.4f}"
        )

        print(
            f"Precision         : "
            f"{self.precision(y_true, y_pred):.4f}"
        )

        print(
            f"Recall            : "
            f"{self.recall(y_true, y_pred):.4f}"
        )

        print(
            f"F1 Score          : "
            f"{self.f1(y_true, y_pred):.4f}"
        )

        print(
            f"Feature Reduction : "
            f"{self.feature_reduction(original_features, selected_features):.2f}%"
        )

        print("\nConfusion Matrix\n")

        print(
            self.confusion(
                y_true,
                y_pred,
            )
        )

        print("\n======================================\n")
"""
Evaluation Metrics for HQFSF.

Responsible for

    - Classification Metrics
    - Feature Selection Statistics
    - Confusion Matrix
    - Classification Report
    - ROC-AUC
    - Balanced Accuracy
    - Matthews Correlation Coefficient (MCC)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumMetrics:
    """
    Evaluation metrics for HQFSF.
    """

    def __init__(self) -> None:

        logger.info(
            "QuantumMetrics initialized."
        )

    # ---------------------------------------------------------
    # Classification Metrics
    # ---------------------------------------------------------

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
            "Accuracy = %.4f",
            score,
        )

        return score

    def precision(
        self,
        y_true,
        y_pred,
        average: str = "binary",
    ) -> float:

        score = precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

        return score

    def recall(
        self,
        y_true,
        y_pred,
        average: str = "binary",
    ) -> float:

        return recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

    def f1(
        self,
        y_true,
        y_pred,
        average: str = "binary",
    ) -> float:

        return f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )

    def balanced_accuracy(
        self,
        y_true,
        y_pred,
    ) -> float:

        return balanced_accuracy_score(
            y_true,
            y_pred,
        )

    def mcc(
        self,
        y_true,
        y_pred,
    ) -> float:

        return matthews_corrcoef(
            y_true,
            y_pred,
        )

    def roc_auc(
        self,
        y_true,
        y_score,
    ) -> float:
        """
        ROC-AUC score.

        Parameters
        ----------
        y_score : array-like
            Prediction probabilities.
        """

        return roc_auc_score(
            y_true,
            y_score,
        )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    def confusion(
        self,
        y_true,
        y_pred,
    ) -> np.ndarray:

        matrix = confusion_matrix(
            y_true,
            y_pred,
        )

        logger.info(
            "Confusion matrix computed."
        )

        return matrix

    # ---------------------------------------------------------
    # Classification Report
    # ---------------------------------------------------------

    def report(
        self,
        y_true,
        y_pred,
    ) -> Dict[str, Any]:

        return classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

    # ---------------------------------------------------------
    # Feature Reduction
    # ---------------------------------------------------------

    def feature_reduction(
        self,
        original_features: int,
        selected_features: int,
    ) -> float:

        if original_features <= 0:

            raise ValueError(
                "original_features must be greater than zero."
            )

        reduction = (
            (
                original_features
                - selected_features
            )
            / original_features
        ) * 100

        logger.info(
            "Feature reduction = %.2f%%",
            reduction,
        )

        return reduction

    # ---------------------------------------------------------
    # All Metrics
    # ---------------------------------------------------------

    def evaluate(
        self,
        y_true,
        y_pred,
        original_features: int,
        selected_features: int,
    ) -> Dict[str, float]:

        return {

            "accuracy":
                self.accuracy(
                    y_true,
                    y_pred,
                ),

            "precision":
                self.precision(
                    y_true,
                    y_pred,
                ),

            "recall":
                self.recall(
                    y_true,
                    y_pred,
                ),

            "f1":
                self.f1(
                    y_true,
                    y_pred,
                ),

            "balanced_accuracy":
                self.balanced_accuracy(
                    y_true,
                    y_pred,
                ),

            "mcc":
                self.mcc(
                    y_true,
                    y_pred,
                ),

            "feature_reduction":
                self.feature_reduction(
                    original_features,
                    selected_features,
                ),
        }

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
        y_true,
        y_pred,
        original_features: int,
        selected_features: int,
    ) -> None:

        metrics = self.evaluate(
            y_true,
            y_pred,
            original_features,
            selected_features,
        )

        print("\n" + "=" * 60)
        print("HQFSF EVALUATION SUMMARY")
        print("=" * 60)

        for key, value in metrics.items():

            if "feature" in key:

                print(
                    f"{key:20}: {value:.2f}%"
                )

            else:

                print(
                    f"{key:20}: {value:.4f}"
                )

        print("\nConfusion Matrix\n")

        print(
            self.confusion(
                y_true,
                y_pred,
            )
        )

        print("=" * 60)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return "QuantumMetrics()"
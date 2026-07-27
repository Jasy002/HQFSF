"""
Model Evaluation Module for HQFSF.

Provides comprehensive evaluation metrics for classification models,
including confusion matrix, classification report, ROC-AUC,
Matthews Correlation Coefficient (MCC), balanced accuracy,
and execution time analysis.
"""

from __future__ import annotations

import time
from typing import Any

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


class ModelEvaluator:
    """
    Evaluate classification models.

    Notes
    -----
    This class computes the evaluation metrics used throughout
    the HQFSF framework for comparing classical and quantum
    feature selection methods.
    """

    def __init__(self) -> None:

        logger.info("ModelEvaluator initialized.")

    # ---------------------------------------------------------
    # Individual Metrics
    # ---------------------------------------------------------

    @staticmethod
    def accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Return classification accuracy."""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Return weighted precision."""
        return precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

    @staticmethod
    def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Return weighted recall."""
        return recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

    @staticmethod
    def f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Return weighted F1-score."""
        return f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

    @staticmethod
    def balanced_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Return balanced accuracy."""
        return balanced_accuracy_score(
            y_true,
            y_pred,
        )

    @staticmethod
    def mcc(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Return Matthews Correlation Coefficient."""
        return matthews_corrcoef(
            y_true,
            y_pred,
        )

    @staticmethod
    def roc_auc(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> float | None:
        """
        Compute ROC-AUC score.

        Returns
        -------
        float | None
            ROC-AUC score or None if unavailable.
        """

        try:

            if y_score.ndim == 2:
                y_score = y_score[:, 1]

            return roc_auc_score(
                y_true,
                y_score,
            )

        except Exception:

            logger.warning(
                "ROC-AUC could not be computed."
            )

            return None

    @staticmethod
    def confusion(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Return confusion matrix."""
        return confusion_matrix(
            y_true,
            y_pred,
        )

    @staticmethod
    def report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        """Return classification report."""
        return classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )

    # ---------------------------------------------------------
    # Complete Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
        training_time: float | None = None,
        inference_time: float | None = None,
    ) -> dict[str, Any]:
        """
        Compute complete evaluation metrics.
        """

        results = {

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

            "f1_score":
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

            "roc_auc":
                self.roc_auc(
                    y_true,
                    y_score,
                ) if y_score is not None else None,

            "confusion_matrix":
                self.confusion(
                    y_true,
                    y_pred,
                ),

            "classification_report":
                self.report(
                    y_true,
                    y_pred,
                ),

            "training_time":
                training_time,

            "inference_time":
                inference_time,
        }

        logger.info(
            "Model evaluation completed."
        )

        return results

    # ---------------------------------------------------------
    # Timing
    # ---------------------------------------------------------

    @staticmethod
    def measure_inference_time(
        model,
        X_test,
    ) -> float:
        """
        Measure inference time in seconds.
        """

        start = time.perf_counter()

        model.predict(X_test)

        return time.perf_counter() - start

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    @staticmethod
    def summary(
        results: dict[str, Any],
    ) -> None:
        """
        Print evaluation summary.
        """

        print("\n" + "=" * 65)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 65)

        print(f"Accuracy            : {results['accuracy']:.4f}")
        print(f"Precision           : {results['precision']:.4f}")
        print(f"Recall              : {results['recall']:.4f}")
        print(f"F1-Score            : {results['f1_score']:.4f}")
        print(f"Balanced Accuracy   : {results['balanced_accuracy']:.4f}")
        print(f"MCC                 : {results['mcc']:.4f}")

        if results["roc_auc"] is not None:
            print(f"ROC-AUC             : {results['roc_auc']:.4f}")

        if results["training_time"] is not None:
            print(
                f"Training Time (s)   : "
                f"{results['training_time']:.4f}"
            )

        if results["inference_time"] is not None:
            print(
                f"Inference Time (s)  : "
                f"{results['inference_time']:.6f}"
            )

        print("=" * 65)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}()"
        )
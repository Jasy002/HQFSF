"""
Evaluation Pipeline for HQFSF.

Responsible for:
    - Classification Metrics
    - Confusion Matrix
    - Feature Reduction
    - Model Evaluation
"""

from __future__ import annotations

from typing import Dict, Any

from quantum.metrics import QuantumMetrics
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    End-to-End Evaluation Pipeline.
    """

    def __init__(self):

        self.metrics = QuantumMetrics()

        logger.info(
            "EvaluationPipeline initialized."
        )

    # ----------------------------------------------------------
    # Execute Evaluation
    # ----------------------------------------------------------

    def run(
        self,
        y_true,
        y_pred,
        original_features: int,
        selected_features: int,
    ) -> Dict[str, Any]:
        """
        Evaluate the HQFSF model.
        """

        logger.info("=" * 60)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 60)

        # ---------------------------------------------
        # Classification Metrics
        # ---------------------------------------------

        accuracy = self.metrics.accuracy(
            y_true,
            y_pred,
        )

        precision = self.metrics.precision(
            y_true,
            y_pred,
        )

        recall = self.metrics.recall(
            y_true,
            y_pred,
        )

        f1_score = self.metrics.f1(
            y_true,
            y_pred,
        )

        confusion_matrix = self.metrics.confusion(
            y_true,
            y_pred,
        )

        feature_reduction = (
            self.metrics.feature_reduction(
                original_features,
                selected_features,
            )
        )

        logger.info(
            "Evaluation completed successfully."
        )

        return {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1_score,

            "confusion_matrix": confusion_matrix,

            "feature_reduction": feature_reduction,
        }

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(
        self,
        y_true,
        y_pred,
        original_features: int,
        selected_features: int,
    ):

        print("\n" + "=" * 60)
        print(" HQFSF Evaluation Pipeline ")
        print("=" * 60)

        self.metrics.summary(
            y_true=y_true,
            y_pred=y_pred,
            original_features=original_features,
            selected_features=selected_features,
        )

        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        return (
            "EvaluationPipeline("
            "metrics=QuantumMetrics)"
        )
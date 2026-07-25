"""
Evaluation Pipeline for HQFSF.

Responsible for:
    - Model Evaluation
    - Classification Metrics
    - Feature Reduction Statistics
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

    def run(
        self,
        y_true,
        y_pred,
        original_features: int,
        selected_features: int,
    ) -> Dict[str, Any]:
        """
        Evaluate HQFSF performance.
        """

        logger.info("=" * 60)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 60)

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

        confusion = self.metrics.confusion(
            y_true,
            y_pred,
        )

        reduction = self.metrics.feature_reduction(
            original_features,
            selected_features,
        )

        logger.info(
            "Evaluation completed successfully."
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": confusion,
            "feature_reduction": reduction,
        }

    def summary(
        self,
        y_true,
        y_pred,
        original_features: int,
        selected_features: int,
    ) -> None:
        """
        Display evaluation summary.
        """

        self.metrics.summary(
            y_true=y_true,
            y_pred=y_pred,
            original_features=original_features,
            selected_features=selected_features,
        )
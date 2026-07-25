"""
Classification evaluation metrics.
"""

from __future__ import annotations

from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from .metrics import Metrics


class ClassificationMetrics(Metrics):
    """
    Classification evaluation metrics.
    """

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob=None,
    ) -> Dict[str, Any]:

        results = {

            "accuracy": accuracy_score(y_true, y_pred),

            "precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "f1_score": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),

            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }

        if y_prob is not None:

            try:

                results["roc_auc"] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                )

            except Exception:

                results["roc_auc"] = None

        else:

            results["roc_auc"] = None

        return results
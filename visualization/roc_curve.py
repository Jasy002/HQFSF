"""
ROC Curve Visualization.

Plots the Receiver Operating Characteristic (ROC)
curve and computes the Area Under Curve (AUC).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


class ROCPlot:
    """
    Visualize ROC Curve.
    """

    def __init__(self):
        pass

    def plot(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = "ROC Curve",
        figsize: tuple = (8, 6),
        save_path: str | None = None,
        show: bool = True,
    ) -> float:
        """
        Plot ROC Curve.

        Parameters
        ----------
        y_true : ndarray
            True labels.

        y_score : ndarray
            Prediction probabilities for the positive class.

        Returns
        -------
        float
            Area Under Curve (AUC).
        """

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(figsize=figsize)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title(title)

        plt.legend(loc="lower right")

        plt.grid(True)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        if show:
            plt.show()

        plt.close()

        return roc_auc

    def summary(self):

        print("\n" + "=" * 60)
        print(" ROC Curve ")
        print("=" * 60)
        print("Visualization : ROC Curve")
        print("Metric        : Area Under Curve (AUC)")
        print("=" * 60)
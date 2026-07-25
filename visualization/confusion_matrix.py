"""
Confusion Matrix Visualization.

Provides a visualization for classification results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


class ConfusionMatrixPlot:
    """
    Visualize a confusion matrix.
    """

    def __init__(self):
        pass

    def plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list | None = None,
        title: str = "Confusion Matrix",
        figsize: tuple = (8, 6),
        cmap: str = "Blues",
        save_path: str | None = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        Plot confusion matrix.

        Parameters
        ----------
        y_true : ndarray
            Ground truth labels.

        y_pred : ndarray
            Predicted labels.
        """

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        plt.figure(figsize=figsize)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )

        display.plot(
            cmap=cmap,
            values_format="d",
            colorbar=True
        )

        plt.title(title)

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

        return cm

    def summary(self):

        print("\n" + "=" * 60)
        print(" Confusion Matrix Plot ")
        print("=" * 60)
        print("Visualization : Confusion Matrix")
        print("Purpose       : Classification Performance")
        print("=" * 60)
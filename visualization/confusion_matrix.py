"""
==============================================================
HQFSF Confusion Matrix Visualization

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides confusion matrix visualization for evaluating
classification performance.
==============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class ConfusionMatrixPlot:
    """
    Visualize a confusion matrix for classification results.
    """

    def plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] | None = None,
        title: str = "Confusion Matrix",
        figsize: tuple[int, int] = (8, 6),
        cmap: str = "Blues",
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        Plot a confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.

        y_pred : np.ndarray
            Predicted labels.

        labels : list[str] | None, optional
            Display labels for each class.

        Returns
        -------
        np.ndarray
            Computed confusion matrix.
        """

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=figsize)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels,
        )

        display.plot(
            ax=ax,
            cmap=cmap,
            values_format="d",
            colorbar=True,
        )

        ax.set_title(title)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close(fig)

        return cm

    @staticmethod
    def summary() -> None:
        """
        Display a summary of the visualization.
        """

        print("\n" + "=" * 60)
        print("Confusion Matrix Plot")
        print("=" * 60)
        print("Visualization : Confusion Matrix")
        print("Purpose       : Classification Performance")
        print("=" * 60)
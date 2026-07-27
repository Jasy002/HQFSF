"""
==============================================================
HQFSF ROC Curve Visualization

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides visualization of the Receiver Operating
Characteristic (ROC) curve and computes the
Area Under the Curve (AUC).
==============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


class ROCPlot:
    """
    Visualize the ROC curve and compute AUC.
    """

    def plot(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = "ROC Curve",
        figsize: tuple[int, int] = (8, 6),
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> float:
        """
        Plot the Receiver Operating Characteristic (ROC) curve.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.

        y_score : np.ndarray
            Prediction probabilities for the positive class.

        title : str, optional
            Plot title.

        figsize : tuple[int, int], optional
            Figure size.

        save_path : str | Path | None, optional
            Output image path.

        show : bool, optional
            Whether to display the figure.

        Returns
        -------
        float
            Area Under the Curve (AUC).
        """

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError(
                "y_true and y_score must have the same number of samples."
            )

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score,
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc:.4f}",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
            label="Random Classifier",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)

        ax.legend(loc="lower right")

        ax.grid(
            linestyle="--",
            alpha=0.5,
        )

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

        return roc_auc

    @staticmethod
    def summary() -> None:
        """
        Display a summary of the visualization.
        """

        print("\n" + "=" * 60)
        print("ROC Curve")
        print("=" * 60)
        print("Visualization : ROC Curve")
        print("Metric        : Area Under Curve (AUC)")
        print("=" * 60)
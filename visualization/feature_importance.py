"""
Feature Importance Visualization.

Plots quantum feature importance scores.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class FeatureImportancePlot:
    """
    Visualize feature importance scores.
    """

    def __init__(self):
        pass

    def plot(
        self,
        scores: np.ndarray,
        feature_names: list | None = None,
        title: str = "Quantum Feature Importance",
        xlabel: str = "Features",
        ylabel: str = "Importance Score",
        figsize: tuple = (10, 6),
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot feature importance.
        """

        if feature_names is None:
            feature_names = [
                f"F{i+1}"
                for i in range(len(scores))
            ]

        plt.figure(figsize=figsize)

        plt.bar(
            feature_names,
            scores
        )

        plt.title(title)

        plt.xlabel(xlabel)

        plt.ylabel(ylabel)

        plt.xticks(rotation=45)

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

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

    def summary(self):

        print("\n" + "=" * 60)
        print(" Feature Importance Plot ")
        print("=" * 60)
        print("Visualization : Bar Chart")
        print("Purpose       : Display Feature Importance")
        print("=" * 60)
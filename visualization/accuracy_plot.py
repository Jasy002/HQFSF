"""
Accuracy Plot Visualization.

Compares the accuracy of multiple machine learning models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class AccuracyPlot:
    """
    Plot classification accuracy.
    """

    def __init__(self):
        pass

    def plot(
        self,
        model_names: list,
        accuracies: list,
        title: str = "Model Accuracy Comparison",
        figsize: tuple = (10, 6),
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot model accuracies.

        Parameters
        ----------
        model_names : list
            Names of classifiers.

        accuracies : list
            Accuracy values.
        """

        plt.figure(figsize=figsize)

        plt.bar(
            model_names,
            accuracies,
        )

        plt.ylim(0, 1)

        plt.xlabel("Models")

        plt.ylabel("Accuracy")

        plt.title(title)

        # Display accuracy values
        for index, value in enumerate(accuracies):
            plt.text(
                index,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                fontsize=10,
            )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.5,
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close()

    def summary(self):

        print("\n" + "=" * 60)
        print(" Accuracy Plot ")
        print("=" * 60)
        print("Visualization : Bar Chart")
        print("Purpose       : Compare Model Accuracy")
        print("=" * 60)
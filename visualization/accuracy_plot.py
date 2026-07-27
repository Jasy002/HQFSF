"""
==============================================================
HQFSF Accuracy Plot

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides visualization for comparing the classification
accuracy of multiple machine learning models.
==============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class AccuracyPlot:
    """
    Create bar charts for model accuracy comparison.
    """

    def plot(
        self,
        model_names: list[str],
        accuracies: list[float],
        title: str = "Model Accuracy Comparison",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot model accuracies.

        Parameters
        ----------
        model_names : list[str]
            Names of the machine learning models.

        accuracies : list[float]
            Accuracy values between 0 and 1.

        title : str, optional
            Plot title.

        figsize : tuple[int, int], optional
            Figure size.

        save_path : str | Path | None, optional
            Output image path.

        show : bool, optional
            Display the figure.
        """

        if len(model_names) != len(accuracies):
            raise ValueError(
                "model_names and accuracies must have the same length."
            )

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(model_names, accuracies)

        ax.set_ylim(0, 1)

        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5,
        )

        for bar, value in zip(bars, accuracies):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                fontsize=10,
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

    @staticmethod
    def summary() -> None:
        """
        Display a summary of the visualization.
        """

        print("\n" + "=" * 60)
        print("Accuracy Plot")
        print("=" * 60)
        print("Visualization : Bar Chart")
        print("Purpose       : Compare Model Accuracy")
        print("=" * 60)
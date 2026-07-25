"""
Runtime Plot Visualization.

Compares execution time of different models or pipeline stages.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class RuntimePlot:
    """
    Plot execution time comparison.
    """

    def __init__(self):
        pass

    def plot(
        self,
        labels: list,
        runtimes: list,
        title: str = "Runtime Comparison",
        figsize: tuple = (10, 6),
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot runtime comparison.

        Parameters
        ----------
        labels : list
            Model or pipeline names.

        runtimes : list
            Execution time in seconds.
        """

        plt.figure(figsize=figsize)

        plt.bar(
            labels,
            runtimes,
        )

        plt.xlabel("Models / Pipeline Stages")

        plt.ylabel("Runtime (seconds)")

        plt.title(title)

        # Display runtime values
        for index, value in enumerate(runtimes):
            plt.text(
                index,
                value,
                f"{value:.3f}s",
                ha="center",
                va="bottom",
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
        print(" Runtime Plot ")
        print("=" * 60)
        print("Visualization : Bar Chart")
        print("Purpose       : Compare Execution Time")
        print("=" * 60)
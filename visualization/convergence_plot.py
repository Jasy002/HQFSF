"""
Convergence Plot Visualization.

Visualizes optimization convergence over iterations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class ConvergencePlot:
    """
    Plot optimization convergence.
    """

    def __init__(self):
        pass

    def plot(
        self,
        iterations: list,
        objective_values: list,
        title: str = "Optimization Convergence",
        figsize: tuple = (10, 6),
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot convergence curve.

        Parameters
        ----------
        iterations : list
            Optimization iteration numbers.

        objective_values : list
            Loss or objective function values.
        """

        plt.figure(figsize=figsize)

        plt.plot(
            iterations,
            objective_values,
            linewidth=2,
            marker="o",
            markersize=5,
        )

        plt.xlabel("Iteration")

        plt.ylabel("Objective Value")

        plt.title(title)

        plt.grid(
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
        print(" Convergence Plot ")
        print("=" * 60)
        print("Visualization : Line Chart")
        print("Purpose       : Optimization Progress")
        print("=" * 60)
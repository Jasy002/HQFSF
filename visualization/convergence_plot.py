"""
==============================================================
HQFSF Convergence Plot Visualization

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides visualization of optimization convergence
during quantum feature selection or model optimization.
==============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class ConvergencePlot:
    """
    Plot optimization convergence over iterations.
    """

    def plot(
        self,
        iterations: list[int],
        objective_values: list[float],
        title: str = "Optimization Convergence",
        figsize: tuple[int, int] = (10, 6),
        save_path: str |Path | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot optimization convergence.

        Parameters
        ----------
        iterations : list[int]
            Optimization iteration numbers.

        objective_values : list[float]
            Objective (loss/fitness) values.

        title : str, optional
            Plot title.

        figsize : tuple[int, int], optional
            Figure size.

        save_path : str | Path | None, optional
            Output image path.

        show : bool, optional
            Whether to display the figure.
        """

        if len(iterations) != len(objective_values):
            raise ValueError(
                "iterations and objective_values must have the same length."
            )

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            iterations,
            objective_values,
            linewidth=2,
            marker="o",
            markersize=5,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title(title)

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

    @staticmethod
    def summary() -> None:
        """
        Display a summary of the visualization.
        """

        print("\n" + "=" * 60)
        print("Convergence Plot")
        print("=" * 60)
        print("Visualization : Line Chart")
        print("Purpose       : Optimization Progress")
        print("=" * 60)
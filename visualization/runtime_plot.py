"""
==============================================================
HQFSF Runtime Plot Visualization

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides visualization for comparing the execution
time of machine learning models or pipeline stages.
==============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class RuntimePlot:
    """
    Plot execution time comparison.
    """

    def plot(
        self,
        labels: list[str],
        runtimes: list[float],
        title: str = "Runtime Comparison",
        figsize: tuple[int, int] = (10, 6),
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot runtime comparison.

        Parameters
        ----------
        labels : list[str]
            Model or pipeline stage names.

        runtimes : list[float]
            Execution times in seconds.

        title : str, optional
            Plot title.

        figsize : tuple[int, int], optional
            Figure size.

        save_path : str | Path | None, optional
            Output image path.

        show : bool, optional
            Whether to display the figure.
        """

        if len(labels) != len(runtimes):
            raise ValueError(
                "labels and runtimes must have the same length."
            )

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(labels, runtimes)

        ax.set_xlabel("Models / Pipeline Stages")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(title)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5,
        )

        for bar, value in zip(bars, runtimes):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}s",
                ha="center",
                va="bottom",
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
        print("Runtime Plot")
        print("=" * 60)
        print("Visualization : Bar Chart")
        print("Purpose       : Compare Execution Time")
        print("=" * 60)
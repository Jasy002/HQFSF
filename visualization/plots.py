"""
==============================================================
HQFSF Plot Manager

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides a unified interface for all visualization
modules used throughout the HQFSF framework.
==============================================================
"""

from __future__ import annotations

import numpy as np

from visualization.accuracy_plot import AccuracyPlot
from visualization.confusion_matrix import ConfusionMatrixPlot
from visualization.convergence_plot import ConvergencePlot
from visualization.feature_importance import FeatureImportancePlot
from visualization.roc_curve import ROCPlot
from visualization.runtime_plot import RuntimePlot


class PlotManager:
    """
    Unified interface for HQFSF visualizations.
    """

    def __init__(self) -> None:

        self.feature_importance = FeatureImportancePlot()
        self.confusion_matrix = ConfusionMatrixPlot()
        self.roc_curve = ROCPlot()
        self.accuracy = AccuracyPlot()
        self.convergence = ConvergencePlot()
        self.runtime = RuntimePlot()

    @staticmethod
    def summary() -> None:
        """
        Display the available visualization modules.
        """

        print("\n" + "=" * 70)
        print("HQFSF Visualization Manager")
        print("=" * 70)
        print("Available Visualizations")
        print("-" * 70)
        print("✓ Feature Importance Plot")
        print("✓ Confusion Matrix")
        print("✓ ROC Curve")
        print("✓ Accuracy Comparison")
        print("✓ Convergence Plot")
        print("✓ Runtime Plot")
        print("=" * 70)

    def feature_plot(
        self,
        scores: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Plot feature importance."""
        self.feature_importance.plot(
            scores=scores,
            feature_names=feature_names,
            **kwargs,
        )

    def confusion_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Plot a confusion matrix."""
        return self.confusion_matrix.plot(
            y_true=y_true,
            y_pred=y_pred,
            **kwargs,
        )

    def roc_plot(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        **kwargs,
    ):
        """Plot the ROC curve."""
        return self.roc_curve.plot(
            y_true=y_true,
            y_score=y_score,
            **kwargs,
        )

    def accuracy_plot(
        self,
        model_names: list[str],
        accuracies: list[float],
        **kwargs,
    ) -> None:
        """Plot model accuracy comparison."""
        self.accuracy.plot(
            model_names=model_names,
            accuracies=accuracies,
            **kwargs,
        )

    def convergence_plot(
        self,
        iterations: list[int],
        objective_values: list[float],
        **kwargs,
    ) -> None:
        """Plot optimization convergence."""
        self.convergence.plot(
            iterations=iterations,
            objective_values=objective_values,
            **kwargs,
        )

    def runtime_plot(
        self,
        labels: list[str],
        runtimes: list[float],
        **kwargs,
    ) -> None:
        """Plot runtime comparison."""
        self.runtime.plot(
            labels=labels,
            runtimes=runtimes,
            **kwargs,
        )
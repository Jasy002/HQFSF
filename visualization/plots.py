"""
Plot Manager.

Provides a unified interface for all visualization modules.
"""

from __future__ import annotations

from visualization.feature_importance import FeatureImportancePlot
from visualization.confusion_matrix import ConfusionMatrixPlot
from visualization.roc_curve import ROCPlot
from visualization.accuracy_plot import AccuracyPlot
from visualization.convergence_plot import ConvergencePlot
from visualization.runtime_plot import RuntimePlot


class PlotManager:
    """
    Centralized manager for HQFSF visualizations.
    """

    def __init__(self):

        self.feature_importance = FeatureImportancePlot()

        self.confusion_matrix = ConfusionMatrixPlot()

        self.roc_curve = ROCPlot()

        self.accuracy = AccuracyPlot()

        self.convergence = ConvergencePlot()

        self.runtime = RuntimePlot()

    def summary(self):
        """
        Display available visualizations.
        """

        print("\n" + "=" * 70)
        print(" HQFSF Visualization Manager ")
        print("=" * 70)

        print("Available Visualizations")
        print("------------------------")

        print("✓ Feature Importance Plot")
        print("✓ Confusion Matrix")
        print("✓ ROC Curve")
        print("✓ Accuracy Comparison")
        print("✓ Convergence Plot")
        print("✓ Runtime Plot")

        print("=" * 70 + "\n")

    def feature_plot(
        self,
        scores,
        feature_names=None,
        **kwargs
    ):
        """
        Plot feature importance.
        """
        self.feature_importance.plot(
            scores=scores,
            feature_names=feature_names,
            **kwargs
        )

    def confusion_plot(
        self,
        y_true,
        y_pred,
        **kwargs
    ):
        """
        Plot confusion matrix.
        """
        return self.confusion_matrix.plot(
            y_true=y_true,
            y_pred=y_pred,
            **kwargs
        )

    def roc_plot(
        self,
        y_true,
        y_score,
        **kwargs
    ):
        """
        Plot ROC curve.
        """
        return self.roc_curve.plot(
            y_true=y_true,
            y_score=y_score,
            **kwargs
        )

    def accuracy_plot(
        self,
        model_names,
        accuracies,
        **kwargs
    ):
        """
        Plot model accuracy comparison.
        """
        self.accuracy.plot(
            model_names=model_names,
            accuracies=accuracies,
            **kwargs
        )

    def convergence_plot(
        self,
        iterations,
        objective_values,
        **kwargs
    ):
        """
        Plot optimization convergence.
        """
        self.convergence.plot(
            iterations=iterations,
            objective_values=objective_values,
            **kwargs
        )

    def runtime_plot(
        self,
        labels,
        runtimes,
        **kwargs
    ):
        """
        Plot runtime comparison.
        """
        self.runtime.plot(
            labels=labels,
            runtimes=runtimes,
            **kwargs
        )
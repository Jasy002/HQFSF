"""
Visualization Package.

Provides visualization utilities for the
Hybrid Quantum Feature Selection Framework (HQFSF).

Modules
-------
- Feature Importance
- Confusion Matrix
- ROC Curve
- Accuracy Plot
- Convergence Plot
- Runtime Plot
- General Plots
- Report Generator
"""

from .feature_importance import FeatureImportancePlot
from .confusion_matrix import ConfusionMatrixPlot
from .roc_curve import ROCPlot
from .accuracy_plot import AccuracyPlot
from .convergence_plot import ConvergencePlot
from .runtime_plot import RuntimePlot
from .plots import PlotManager
from .report import ReportGenerator

__all__ = [
    "FeatureImportancePlot",
    "ConfusionMatrixPlot",
    "ROCPlot",
    "AccuracyPlot",
    "ConvergencePlot",
    "RuntimePlot",
    "PlotManager",
    "ReportGenerator",
]

__version__ = "1.0.0"

__author__ = "Jasmine Sultana"
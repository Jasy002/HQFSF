"""
==============================================================
HQFSF Visualization Package

Hybrid Quantum Feature Selection Framework (HQFSF)

Provides visualization utilities for:

• Feature Importance
• Confusion Matrix
• ROC Curve
• Accuracy Plot
• Convergence Plot
• Runtime Plot
• General Plot Utilities
• Report Generation

This package centralizes all visualization components
used throughout the HQFSF pipeline.
==============================================================
"""

from __future__ import annotations

from .accuracy_plot import AccuracyPlot
from .confusion_matrix import ConfusionMatrixPlot
from .convergence_plot import ConvergencePlot
from .feature_importance import FeatureImportancePlot
from .plots import PlotManager
from .report import ReportGenerator
from .roc_curve import ROCPlot
from .runtime_plot import RuntimePlot

__version__ = "1.0.0"
__author__ = "Jasmine Sultana"

__all__ = [
    "AccuracyPlot",
    "ConfusionMatrixPlot",
    "ConvergencePlot",
    "FeatureImportancePlot",
    "PlotManager",
    "ROCPlot",
    "ReportGenerator",
    "RuntimePlot",
]
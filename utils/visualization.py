"""
visualization.py
================

Visualization utilities for the
Hybrid Quantum Feature Selection Framework (HQFSF).

Provides plotting functions for:
    • Feature Importance
    • Confusion Matrix
    • Model Accuracy Comparison
    • Feature Reduction
    • Quantum Expectation Values
    • Training History
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def plot_feature_importance(
    importance_scores,
    feature_names,
    title="Feature Importance",
    figsize=(10, 6),
):
    """
    Plot feature importance scores.
    """

    importance_scores = np.asarray(importance_scores)

    order = np.argsort(importance_scores)[::-1]

    scores = importance_scores[order]

    names = np.asarray(feature_names)[order]

    plt.figure(figsize=figsize)

    plt.bar(range(len(scores)), scores)

    plt.xticks(
        range(len(scores)),
        names,
        rotation=90,
    )

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.title(title)

    plt.tight_layout()

    plt.show()


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(
    confusion_matrix,
    class_names=None,
    title="Confusion Matrix",
):
    """
    Plot confusion matrix.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names,
    )

    display.plot()

    plt.title(title)

    plt.tight_layout()

    plt.show()


# ==========================================================
# MODEL ACCURACY
# ==========================================================

def plot_model_accuracy(
    model_names,
    accuracies,
):
    """
    Compare model accuracies.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        model_names,
        accuracies,
    )

    plt.ylabel("Accuracy")

    plt.xlabel("Model")

    plt.title("Model Accuracy Comparison")

    plt.tight_layout()

    plt.show()


# ==========================================================
# FEATURE REDUCTION
# ==========================================================

def plot_feature_reduction(
    original_features,
    selected_features,
):
    """
    Plot original vs selected features.
    """

    labels = [
        "Original",
        "Selected",
    ]

    values = [
        original_features,
        selected_features,
    ]

    plt.figure(figsize=(6, 5))

    plt.bar(labels, values)

    plt.ylabel("Number of Features")

    plt.title("Feature Reduction")

    plt.tight_layout()

    plt.show()


# ==========================================================
# EXPECTATION VALUES
# ==========================================================

def plot_expectation_values(
    expectation_values,
):
    """
    Plot expectation values.
    """

    expectation_values = np.asarray(
        expectation_values
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        expectation_values,
        marker="o",
    )

    plt.xlabel("Feature Index")

    plt.ylabel("Expectation Value")

    plt.title("Quantum Expectation Values")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ==========================================================
# TRAINING HISTORY
# ==========================================================

def plot_training_history(
    history,
):
    """
    Plot training history.
    """

    plt.figure(figsize=(8, 5))

    for metric, values in history.items():

        plt.plot(
            values,
            label=metric,
        )

    plt.xlabel("Epoch")

    plt.ylabel("Metric")

    plt.title("Training History")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ==========================================================
# LINE PLOT
# ==========================================================

def plot_line(
    x,
    y,
    xlabel="X",
    ylabel="Y",
    title="Line Plot",
):
    """
    Generic line plot.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        marker="o",
    )

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ==========================================================
# BAR PLOT
# ==========================================================

def plot_bar(
    labels,
    values,
    xlabel="",
    ylabel="",
    title="Bar Plot",
):
    """
    Generic bar plot.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values,
    )

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.title(title)

    plt.tight_layout()

    plt.show()


# ==========================================================
# HISTOGRAM
# ==========================================================

def plot_histogram(
    values,
    bins=20,
    title="Histogram",
):
    """
    Plot histogram.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=bins,
    )

    plt.xlabel("Value")

    plt.ylabel("Frequency")

    plt.title(title)

    plt.tight_layout()

    plt.show()


# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return "HQFSF Visualization Utilities"
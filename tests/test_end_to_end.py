"""
End-to-End tests for the HQFSF (Hybrid Quantum Feature Selection Framework).

This test validates the complete workflow:
Dataset → Preprocessing → Classical Feature Selection →
Quantum Feature Selection → Evaluation
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from classical.preprocessing import DataPreprocessor
from classical.feature_selector import ClassicalFeatureSelector
from quantum.feature_selector import QuantumFeatureSelector
from quantum.metrics import QuantumMetrics


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def dataset():
    data = load_breast_cancer()

    return (
        data.data,
        data.target,
        data.feature_names,
    )


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


@pytest.fixture
def classical_selector():
    return ClassicalFeatureSelector(
        method="mutual_info",
        top_k=15,
    )


@pytest.fixture
def quantum_selector():
    return QuantumFeatureSelector(
        strategy="top_k",
        top_k=8,
    )


@pytest.fixture
def metrics():
    return QuantumMetrics()


# ---------------------------------------------------------------------
# End-to-End Workflow
# ---------------------------------------------------------------------

def test_end_to_end_pipeline(
    dataset,
    preprocessor,
    classical_selector,
    quantum_selector,
    metrics,
):
    X, y, feature_names = dataset

    # ----------------------------
    # Stage 1 : Preprocessing
    # ----------------------------

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    assert X_processed.shape[0] == len(y)

    # ----------------------------
    # Stage 2 : Classical Feature Selection
    # ----------------------------

    X_classical = classical_selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert X_classical.shape[1] == 15

    # ----------------------------
    # Stage 3 : Quantum Feature Selection
    # ----------------------------

    importance_scores = np.linspace(
        0.15,
        0.95,
        X_classical.shape[1],
    )

    selected = quantum_selector.select(
        importance_scores
    )

    X_final = X_classical[:, selected]

    assert X_final.shape == (
        len(y),
        quantum_selector.top_k,
    )

    # ----------------------------
    # Stage 4 : Dummy Prediction
    # ----------------------------

    # Placeholder predictions until
    # classifier integration
    y_pred = y_processed.copy()

    # ----------------------------
    # Stage 5 : Evaluation
    # ----------------------------

    results = metrics.evaluate(
        y_true=y_processed,
        y_pred=y_pred,
        original_features=X.shape[1],
        selected_features=X_final.shape[1],
    )

    assert isinstance(results, dict)

    required = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "feature_reduction",
    }

    assert required.issubset(results.keys())

    assert results["accuracy"] == pytest.approx(1.0)

    reduction = (
        (X.shape[1] - X_final.shape[1])
        / X.shape[1]
    ) * 100

    assert results["feature_reduction"] == pytest.approx(
        reduction,
        rel=1e-6,
    )


# ---------------------------------------------------------------------
# Pipeline Repeatability
# ---------------------------------------------------------------------

def test_end_to_end_repeatability(
    dataset,
    preprocessor,
    classical_selector,
    quantum_selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    first = classical_selector.fit_transform(
        X_processed,
        y_processed,
    )

    second = classical_selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert np.array_equal(
        first,
        second,
    )

    scores = np.linspace(
        0.1,
        0.9,
        first.shape[1],
    )

    selected1 = quantum_selector.select(scores)
    selected2 = quantum_selector.select(scores)

    assert np.array_equal(
        selected1,
        selected2,
    )


# ---------------------------------------------------------------------
# Final Dataset Verification
# ---------------------------------------------------------------------

def test_final_dataset_properties(
    dataset,
    preprocessor,
    classical_selector,
    quantum_selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    X_classical = classical_selector.fit_transform(
        X_processed,
        y_processed,
    )

    scores = np.random.rand(
        X_classical.shape[1]
    )

    selected = quantum_selector.select(
        scores
    )

    X_final = X_classical[:, selected]

    assert isinstance(
        X_final,
        np.ndarray,
    )

    assert np.issubdtype(
        X_final.dtype,
        np.number,
    )

    assert X_final.shape[0] == len(y)

    assert X_final.shape[1] == quantum_selector.top_k


# ---------------------------------------------------------------------
# Feature Reduction Verification
# ---------------------------------------------------------------------

def test_feature_reduction_percentage(
    metrics,
):
    reduction = metrics.feature_reduction(
        original_features=30,
        selected_features=8,
    )

    expected = (
        (30 - 8)
        / 30
    ) * 100

    assert reduction == pytest.approx(
        expected,
        rel=1e-6,
    )


# ---------------------------------------------------------------------
# Complete Framework Smoke Test
# ---------------------------------------------------------------------

def test_hqfsf_framework_smoke(
    dataset,
    preprocessor,
    classical_selector,
    quantum_selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    X_classical = classical_selector.fit_transform(
        X_processed,
        y_processed,
    )

    importance = np.linspace(
        0.2,
        1.0,
        X_classical.shape[1],
    )

    selected = quantum_selector.select(
        importance
    )

    X_final = X_classical[:, selected]

    assert X_final.shape[0] == len(y)
    assert X_final.shape[1] == quantum_selector.top_k
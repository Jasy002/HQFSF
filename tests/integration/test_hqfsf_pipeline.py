"""
Integration tests for the complete HQFSF (Hybrid Quantum Feature Selection
Framework) pipeline.
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
# Stage 1 : Preprocessing
# ---------------------------------------------------------------------

def test_preprocessing_stage(
    dataset,
    preprocessor,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    assert isinstance(X_processed, np.ndarray)
    assert isinstance(y_processed, np.ndarray)

    assert X_processed.shape[0] == len(y)


# ---------------------------------------------------------------------
# Stage 2 : Classical Feature Selection
# ---------------------------------------------------------------------

def test_classical_selection_stage(
    dataset,
    preprocessor,
    classical_selector,
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

    assert X_classical.shape[1] == 15


# ---------------------------------------------------------------------
# Stage 3 : Quantum Feature Selection
# ---------------------------------------------------------------------

def test_quantum_selection_stage(
    quantum_selector,
):
    importance_scores = np.array([
        0.81,
        0.54,
        0.96,
        0.22,
        0.74,
        0.68,
        0.91,
        0.43,
        0.37,
        0.57,
        0.84,
        0.63,
        0.29,
        0.77,
        0.72,
    ])

    selected = quantum_selector.select(
        importance_scores
    )

    assert len(selected) == 8


# ---------------------------------------------------------------------
# Hybrid Pipeline
# ---------------------------------------------------------------------

def test_hybrid_pipeline(
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

    scores = np.linspace(
        0.1,
        0.9,
        X_classical.shape[1],
    )

    selected = quantum_selector.select(scores)

    X_final = X_classical[:, selected]

    assert X_final.shape[0] == len(y)
    assert X_final.shape[1] == 8


# ---------------------------------------------------------------------
# Feature Reduction
# ---------------------------------------------------------------------

def test_feature_reduction(
    metrics,
):
    reduction = metrics.feature_reduction(
        original_features=30,
        selected_features=8,
    )

    assert reduction > 70.0


# ---------------------------------------------------------------------
# HQFSF Evaluation
# ---------------------------------------------------------------------

def test_pipeline_evaluation(
    metrics,
):
    y_true = [
        0,1,1,0,1,
        0,1,0,1,1
    ]

    y_pred = [
        0,1,1,0,0,
        0,1,0,1,1
    ]

    results = metrics.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        original_features=30,
        selected_features=8,
    )

    assert isinstance(results, dict)

    expected = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "feature_reduction",
    }

    assert expected.issubset(results.keys())


# ---------------------------------------------------------------------
# Pipeline Consistency
# ---------------------------------------------------------------------

def test_pipeline_consistency(
    dataset,
    preprocessor,
    classical_selector,
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


# ---------------------------------------------------------------------
# Final Dataset Verification
# ---------------------------------------------------------------------

def test_final_dataset_shape(
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

    selected = quantum_selector.select(scores)

    X_final = X_classical[:, selected]

    assert X_final.shape == (
        len(y),
        quantum_selector.top_k,
    )


# ---------------------------------------------------------------------
# Complete HQFSF Workflow
# ---------------------------------------------------------------------

def test_complete_hqfsf_pipeline(
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
        0.95,
        X_classical.shape[1],
    )

    selected = quantum_selector.select(
        importance
    )

    X_final = X_classical[:, selected]

    assert isinstance(X_final, np.ndarray)
    assert X_final.shape[0] == len(y)
    assert X_final.shape[1] == quantum_selector.top_k
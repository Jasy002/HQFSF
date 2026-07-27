"""
Integration tests for the complete Classical Feature Selection Pipeline.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from classical.preprocessing import DataPreprocessor
from classical.feature_selector import ClassicalFeatureSelector
from classical.metrics import ClassicalMetrics


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
def selector():
    return ClassicalFeatureSelector(
        method="mutual_info",
        top_k=10,
    )


@pytest.fixture
def metrics():
    return ClassicalMetrics()


# ---------------------------------------------------------------------
# Preprocessing Integration
# ---------------------------------------------------------------------

def test_preprocessing_pipeline(
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

    assert X_processed.shape[0] == X.shape[0]
    assert len(y_processed) == len(y)


# ---------------------------------------------------------------------
# Feature Selection Integration
# ---------------------------------------------------------------------

def test_feature_selection_pipeline(
    dataset,
    preprocessor,
    selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    X_selected = selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert isinstance(X_selected, np.ndarray)
    assert X_selected.shape[0] == X.shape[0]
    assert X_selected.shape[1] == 10


# ---------------------------------------------------------------------
# Feature Ranking
# ---------------------------------------------------------------------

def test_feature_ranking(
    dataset,
    preprocessor,
    selector,
):
    X, y, feature_names = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    selector.fit(
        X_processed,
        y_processed,
    )

    ranking = selector.feature_ranking(
        feature_names
    )

    assert isinstance(ranking, list)
    assert len(ranking) == len(feature_names)


# ---------------------------------------------------------------------
# Selected Features
# ---------------------------------------------------------------------

def test_selected_feature_indices(
    dataset,
    preprocessor,
    selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    selector.fit(
        X_processed,
        y_processed,
    )

    indices = selector.selected_features_

    assert len(indices) == 10
    assert max(indices) < X.shape[1]


# ---------------------------------------------------------------------
# Metrics Integration
# ---------------------------------------------------------------------

def test_feature_reduction_metric(
    metrics,
):
    reduction = metrics.feature_reduction(
        original_features=30,
        selected_features=10,
    )

    assert reduction == pytest.approx(
        66.666666,
        rel=1e-3,
    )


# ---------------------------------------------------------------------
# Pipeline Consistency
# ---------------------------------------------------------------------

def test_pipeline_consistency(
    dataset,
    preprocessor,
    selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    first = selector.fit_transform(
        X_processed,
        y_processed,
    )

    second = selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert np.array_equal(
        first,
        second,
    )


# ---------------------------------------------------------------------
# Shape Verification
# ---------------------------------------------------------------------

def test_selected_dataset_shape(
    dataset,
    preprocessor,
    selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    X_selected = selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert X_selected.shape == (
        X.shape[0],
        10,
    )


# ---------------------------------------------------------------------
# Data Type Verification
# ---------------------------------------------------------------------

def test_selected_dataset_dtype(
    dataset,
    preprocessor,
    selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    X_selected = selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert np.issubdtype(
        X_selected.dtype,
        np.number,
    )


# ---------------------------------------------------------------------
# End-to-End Classical Workflow
# ---------------------------------------------------------------------

def test_complete_classical_pipeline(
    dataset,
    preprocessor,
    selector,
):
    X, y, _ = dataset

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    X_selected = selector.fit_transform(
        X_processed,
        y_processed,
    )

    assert X_selected.shape[0] == len(y)
    assert X_selected.shape[1] == selector.top_k
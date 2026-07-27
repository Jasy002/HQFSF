"""
Integration tests for Dataset → Preprocessing Pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from classical.preprocessing import DataPreprocessor


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Age": [22, 35, np.nan, 40, 29],
        "Salary": [25000, 45000, 38000, np.nan, 52000],
        "Gender": ["M", "F", "F", "M", "F"],
        "Purchased": [0, 1, 0, 1, 1]
    })


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


# ---------------------------------------------------------------------
# Missing Value Pipeline
# ---------------------------------------------------------------------

def test_missing_value_pipeline(
    preprocessor,
    sample_dataframe,
):
    processed = preprocessor.handle_missing_values(
        sample_dataframe.copy()
    )

    assert processed.isnull().sum().sum() == 0


# ---------------------------------------------------------------------
# Encoding Pipeline
# ---------------------------------------------------------------------

def test_encoding_pipeline(
    preprocessor,
    sample_dataframe,
):
    processed = preprocessor.handle_missing_values(
        sample_dataframe.copy()
    )

    processed = preprocessor.encode_categorical(
        processed
    )

    assert processed.select_dtypes(
        include="object"
    ).empty


# ---------------------------------------------------------------------
# Scaling Pipeline
# ---------------------------------------------------------------------

def test_scaling_pipeline(
    preprocessor,
    sample_dataframe,
):
    processed = preprocessor.handle_missing_values(
        sample_dataframe.copy()
    )

    processed = preprocessor.encode_categorical(
        processed
    )

    scaled = preprocessor.scale_features(
        processed.drop(columns=["Purchased"])
    )

    assert isinstance(
        scaled,
        np.ndarray,
    )


# ---------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------

def test_complete_preprocessing_pipeline(
    preprocessor,
    sample_dataframe,
):
    X = sample_dataframe.drop(
        columns=["Purchased"]
    )

    y = sample_dataframe["Purchased"]

    X_processed, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    assert isinstance(X_processed, np.ndarray)
    assert len(X_processed) == len(y_processed)
    assert X_processed.shape[0] == 5


# ---------------------------------------------------------------------
# Deterministic Output
# ---------------------------------------------------------------------

def test_pipeline_consistency(
    preprocessor,
    sample_dataframe,
):
    X = sample_dataframe.drop(
        columns=["Purchased"]
    )

    y = sample_dataframe["Purchased"]

    X1, y1 = preprocessor.fit_transform(
        X,
        y,
    )

    X2, y2 = preprocessor.fit_transform(
        X,
        y,
    )

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


# ---------------------------------------------------------------------
# Output Shape
# ---------------------------------------------------------------------

def test_pipeline_output_shape(
    preprocessor,
    sample_dataframe,
):
    X = sample_dataframe.drop(
        columns=["Purchased"]
    )

    y = sample_dataframe["Purchased"]

    X_processed, _ = preprocessor.fit_transform(
        X,
        y,
    )

    assert X_processed.shape[0] == 5
    assert X_processed.shape[1] > 0


# ---------------------------------------------------------------------
# Feature Types
# ---------------------------------------------------------------------

def test_pipeline_output_numeric(
    preprocessor,
    sample_dataframe,
):
    X = sample_dataframe.drop(
        columns=["Purchased"]
    )

    y = sample_dataframe["Purchased"]

    X_processed, _ = preprocessor.fit_transform(
        X,
        y,
    )

    assert np.issubdtype(
        X_processed.dtype,
        np.number,
    )


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def test_labels_unchanged(
    preprocessor,
    sample_dataframe,
):
    X = sample_dataframe.drop(
        columns=["Purchased"]
    )

    y = sample_dataframe["Purchased"]

    _, y_processed = preprocessor.fit_transform(
        X,
        y,
    )

    assert set(y_processed) == {0, 1}
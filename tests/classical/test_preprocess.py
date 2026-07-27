"""
Unit Tests for DataPreprocessor.

Tests:
    - Initialization
    - Missing value handling
    - Duplicate removal
    - Feature/target split
    - Feature scaling
    - Complete preprocessing pipeline
    - Invalid target column
    - Empty dataframe
    - Summary
    - __repr__
"""

import numpy as np
import pandas as pd
import pytest

from classical.preprocessing import DataPreprocessor
from utils.exceptions import DatasetError


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe."""

    return pd.DataFrame(
        {
            "age": [20, 21, np.nan, 23],
            "salary": [50000, 52000, 51000, 50000],
            "city": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def duplicate_dataframe():
    """Create dataframe with duplicate rows."""

    return pd.DataFrame(
        {
            "feature": [1, 2, 2],
            "target": [0, 1, 1],
        }
    )


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_preprocessor_initialization():

    preprocessor = DataPreprocessor()

    assert isinstance(preprocessor, DataPreprocessor)


# ---------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------

def test_handle_missing_values(sample_dataframe):

    preprocessor = DataPreprocessor()

    df = preprocessor.handle_missing_values(sample_dataframe)

    assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------
# Duplicate Removal
# ---------------------------------------------------------

def test_remove_duplicates(duplicate_dataframe):

    preprocessor = DataPreprocessor()

    df = preprocessor.remove_duplicates(duplicate_dataframe)

    assert len(df) == 2


# ---------------------------------------------------------
# Feature / Target Split
# ---------------------------------------------------------

def test_split_features_target(sample_dataframe):

    preprocessor = DataPreprocessor()

    X, y = preprocessor.split_features_target(
        sample_dataframe,
        target_column="target"
    )

    assert "target" not in X.columns

    assert len(y) == len(sample_dataframe)


def test_invalid_target_column(sample_dataframe):

    preprocessor = DataPreprocessor()

    with pytest.raises(DatasetError):

        preprocessor.split_features_target(
            sample_dataframe,
            target_column="class"
        )


# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------

def test_scale_features(sample_dataframe):

    preprocessor = DataPreprocessor()

    X = sample_dataframe.drop(columns=["target", "city"])

    X_scaled = preprocessor.scale_features(X)

    assert X_scaled.shape == X.shape


# ---------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------

def test_preprocess_pipeline(sample_dataframe):

    preprocessor = DataPreprocessor()

    df = preprocessor.preprocess(sample_dataframe)

    assert isinstance(df, pd.DataFrame)

    assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------

def test_empty_dataframe():

    preprocessor = DataPreprocessor()

    empty = pd.DataFrame()

    with pytest.raises(DatasetError):

        preprocessor.preprocess(empty)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def test_summary(sample_dataframe):

    preprocessor = DataPreprocessor()

    preprocessor.summary(sample_dataframe)


# ---------------------------------------------------------
# Representation
# ---------------------------------------------------------

def test_repr():

    preprocessor = DataPreprocessor()

    assert "DataPreprocessor" in repr(preprocessor)
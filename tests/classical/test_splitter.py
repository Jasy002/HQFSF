"""
Unit Tests for DataSplitter.

Tests:
    - Initialization
    - Train/Test Split
    - Stratified Split
    - Reproducibility
    - Custom Test Size
    - Invalid Test Size
    - Invalid Random State
    - Empty Dataset
    - Shape Validation
    - Summary
    - __repr__
"""

import pandas as pd
import pytest

from classical.splitter import DataSplitter
from utils.exceptions import DatasetError


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample dataset."""

    X = pd.DataFrame(
        {
            "age": range(100),
            "salary": range(100, 200),
            "score": range(200, 300),
        }
    )

    y = pd.Series([0] * 50 + [1] * 50)

    return X, y


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_default_initialization():

    splitter = DataSplitter()

    assert splitter.test_size == 0.2

    assert splitter.random_state == 42


def test_custom_initialization():

    splitter = DataSplitter(
        test_size=0.3,
        random_state=123
    )

    assert splitter.test_size == 0.3

    assert splitter.random_state == 123


# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------

def test_train_test_split(sample_data):

    X, y = sample_data

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(X, y)

    assert len(X_train) == 80

    assert len(X_test) == 20

    assert len(y_train) == 80

    assert len(y_test) == 20


# ---------------------------------------------------------
# Shape Validation
# ---------------------------------------------------------

def test_split_shapes(sample_data):

    X, y = sample_data

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(X, y)

    assert X_train.shape[1] == X.shape[1]

    assert X_test.shape[1] == X.shape[1]


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def test_random_state_reproducibility(sample_data):

    X, y = sample_data

    splitter1 = DataSplitter(random_state=42)

    splitter2 = DataSplitter(random_state=42)

    result1 = splitter1.split(X, y)

    result2 = splitter2.split(X, y)

    pd.testing.assert_frame_equal(result1[0], result2[0])

    pd.testing.assert_frame_equal(result1[1], result2[1])

    pd.testing.assert_series_equal(result1[2], result2[2])

    pd.testing.assert_series_equal(result1[3], result2[3])


# ---------------------------------------------------------
# Stratified Split
# ---------------------------------------------------------

def test_stratified_split(sample_data):

    X, y = sample_data

    splitter = DataSplitter()

    X_train, X_test, y_train, y_test = splitter.split(
        X,
        y,
        stratify=True
    )

    assert y_train.value_counts().sum() == 80

    assert y_test.value_counts().sum() == 20


# ---------------------------------------------------------
# Custom Test Size
# ---------------------------------------------------------

def test_custom_test_size(sample_data):

    X, y = sample_data

    splitter = DataSplitter(test_size=0.25)

    X_train, X_test, _, _ = splitter.split(X, y)

    assert len(X_train) == 75

    assert len(X_test) == 25


# ---------------------------------------------------------
# Invalid Test Size
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "test_size",
    [-0.5, 0, 1, 2]
)
def test_invalid_test_size(test_size):

    with pytest.raises(ValueError):

        DataSplitter(test_size=test_size)


# ---------------------------------------------------------
# Invalid Random State
# ---------------------------------------------------------

def test_invalid_random_state():

    with pytest.raises(TypeError):

        DataSplitter(random_state="abc")


# ---------------------------------------------------------
# Empty Dataset
# ---------------------------------------------------------

def test_empty_dataset():

    splitter = DataSplitter()

    X = pd.DataFrame()

    y = pd.Series(dtype=int)

    with pytest.raises(DatasetError):

        splitter.split(X, y)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def test_summary():

    splitter = DataSplitter()

    splitter.summary()


# ---------------------------------------------------------
# Representation
# ---------------------------------------------------------

def test_repr():

    splitter = DataSplitter()

    assert "DataSplitter" in repr(splitter)
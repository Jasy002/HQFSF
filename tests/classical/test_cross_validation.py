"""
Unit Tests for CrossValidator.

Tests:
    - Initialization
    - K-Fold Cross Validation
    - Stratified K-Fold
    - Leave-One-Out
    - Shuffle Split
    - Reproducibility
    - Invalid Number of Splits
    - Empty Dataset
    - Summary
    - __repr__
"""

import pandas as pd
import pytest

from classical.cross_validation import CrossValidator
from utils.exceptions import DatasetError


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample dataset."""

    X = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100, 200),
            "feature3": range(200, 300),
        }
    )

    y = pd.Series([0] * 50 + [1] * 50)

    return X, y


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_default_initialization():

    cv = CrossValidator()

    assert cv.n_splits == 5

    assert cv.shuffle is True

    assert cv.random_state == 42


def test_custom_initialization():

    cv = CrossValidator(
        n_splits=10,
        shuffle=False,
        random_state=123,
    )

    assert cv.n_splits == 10

    assert cv.shuffle is False

    assert cv.random_state == 123


# ---------------------------------------------------------
# K-Fold
# ---------------------------------------------------------

def test_kfold_split(sample_data):

    X, y = sample_data

    cv = CrossValidator(n_splits=5)

    folds = list(cv.split(X, y))

    assert len(folds) == 5

    for train_idx, test_idx in folds:

        assert len(train_idx) == 80

        assert len(test_idx) == 20


# ---------------------------------------------------------
# Stratified K-Fold
# ---------------------------------------------------------

def test_stratified_kfold(sample_data):

    X, y = sample_data

    cv = CrossValidator(
        n_splits=5,
        stratified=True,
    )

    folds = list(cv.split(X, y))

    assert len(folds) == 5

    for train_idx, test_idx in folds:

        train_labels = y.iloc[train_idx]

        test_labels = y.iloc[test_idx]

        assert train_labels.nunique() == 2

        assert test_labels.nunique() == 2


# ---------------------------------------------------------
# Leave-One-Out
# ---------------------------------------------------------

def test_leave_one_out(sample_data):

    X, y = sample_data

    cv = CrossValidator(method="loo")

    folds = list(cv.split(X, y))

    assert len(folds) == len(X)

    assert len(folds[0][1]) == 1


# ---------------------------------------------------------
# Shuffle Split
# ---------------------------------------------------------

def test_shuffle_split(sample_data):

    X, y = sample_data

    cv = CrossValidator(
        method="shuffle",
        n_splits=5,
        test_size=0.2,
    )

    folds = list(cv.split(X, y))

    assert len(folds) == 5


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def test_reproducibility(sample_data):

    X, y = sample_data

    cv1 = CrossValidator(random_state=42)

    cv2 = CrossValidator(random_state=42)

    folds1 = list(cv1.split(X, y))

    folds2 = list(cv2.split(X, y))

    assert folds1 == folds2


# ---------------------------------------------------------
# Invalid Number of Splits
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "splits",
    [0, 1, -5],
)
def test_invalid_splits(splits):

    with pytest.raises(ValueError):

        CrossValidator(n_splits=splits)


# ---------------------------------------------------------
# Empty Dataset
# ---------------------------------------------------------

def test_empty_dataset():

    X = pd.DataFrame()

    y = pd.Series(dtype=int)

    cv = CrossValidator()

    with pytest.raises(DatasetError):

        list(cv.split(X, y))


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def test_summary():

    cv = CrossValidator()

    cv.summary()


# ---------------------------------------------------------
# Representation
# ---------------------------------------------------------

def test_repr():

    cv = CrossValidator()

    assert "CrossValidator" in repr(cv)
"""
Unit Tests for FeatureEngineering.

Tests:
    - Initialization
    - Polynomial Features
    - Interaction Features
    - Feature Selection
    - PCA
    - Feature Names
    - Invalid Parameters
    - Empty Dataset
    - Summary
    - __repr__
"""

import pandas as pd
import pytest

from classical.feature_engineering import FeatureEngineering
from utils.exceptions import DatasetError


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_dataframe():
    """Sample dataframe."""

    return pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [50000, 55000, 60000, 65000, 70000],
            "experience": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization():

    fe = FeatureEngineering()

    assert isinstance(fe, FeatureEngineering)


# ==========================================================
# Polynomial Features
# ==========================================================

def test_polynomial_features(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    transformed = fe.polynomial_features(
        X,
        degree=2,
    )

    assert transformed.shape[0] == X.shape[0]

    assert transformed.shape[1] > X.shape[1]


# ==========================================================
# Interaction Features
# ==========================================================

def test_interaction_features(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    transformed = fe.interaction_features(X)

    assert transformed.shape[0] == X.shape[0]

    assert transformed.shape[1] >= X.shape[1]


# ==========================================================
# PCA
# ==========================================================

def test_pca(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    transformed = fe.pca(
        X,
        n_components=2,
    )

    assert transformed.shape == (5, 2)


# ==========================================================
# Feature Selection
# ==========================================================

def test_feature_selection(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    y = sample_dataframe["target"]

    selected = fe.select_features(
        X,
        y,
        k=2,
    )

    assert selected.shape[1] == 2


# ==========================================================
# Feature Names
# ==========================================================

def test_feature_names(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    fe.polynomial_features(X)

    names = fe.feature_names()

    assert isinstance(names, list)

    assert len(names) > 0


# ==========================================================
# Invalid Degree
# ==========================================================

def test_invalid_degree(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    with pytest.raises(ValueError):

        fe.polynomial_features(
            X,
            degree=0,
        )


# ==========================================================
# Invalid PCA
# ==========================================================

def test_invalid_pca(sample_dataframe):

    fe = FeatureEngineering()

    X = sample_dataframe.drop(columns=["target"])

    with pytest.raises(ValueError):

        fe.pca(
            X,
            n_components=10,
        )


# ==========================================================
# Empty Dataset
# ==========================================================

def test_empty_dataframe():

    fe = FeatureEngineering()

    empty = pd.DataFrame()

    with pytest.raises(DatasetError):

        fe.polynomial_features(empty)


# ==========================================================
# Summary
# ==========================================================

def test_summary():

    fe = FeatureEngineering()

    fe.summary()


# ==========================================================
# Representation
# ==========================================================

def test_repr():

    fe = FeatureEngineering()

    assert "FeatureEngineering" in repr(fe)
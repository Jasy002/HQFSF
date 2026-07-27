"""
Unit Tests for FeatureScaler.

Tests:
    - Initialization
    - Standard Scaler
    - MinMax Scaler
    - Robust Scaler
    - Fit
    - Transform
    - Fit Transform
    - Inverse Transform
    - Invalid scaler
    - Empty dataframe
    - Summary
    - __repr__
"""

import numpy as np
import pandas as pd
import pytest

from classical.feature_scaling import FeatureScaler
from utils.exceptions import DatasetError


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_dataframe():
    """Sample numeric dataframe."""

    return pd.DataFrame(
        {
            "age": [20, 25, 30, 35],
            "salary": [50000, 60000, 70000, 80000],
            "score": [50, 60, 70, 80],
        }
    )


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_standard_scaler_initialization():

    scaler = FeatureScaler(method="standard")

    assert scaler.method == "standard"


def test_minmax_scaler_initialization():

    scaler = FeatureScaler(method="minmax")

    assert scaler.method == "minmax"


def test_robust_scaler_initialization():

    scaler = FeatureScaler(method="robust")

    assert scaler.method == "robust"


# ---------------------------------------------------------
# Fit
# ---------------------------------------------------------

def test_fit(sample_dataframe):

    scaler = FeatureScaler()

    scaler.fit(sample_dataframe)

    assert scaler.scaler is not None


# ---------------------------------------------------------
# Transform
# ---------------------------------------------------------

def test_transform(sample_dataframe):

    scaler = FeatureScaler()

    scaler.fit(sample_dataframe)

    transformed = scaler.transform(sample_dataframe)

    assert transformed.shape == sample_dataframe.shape

    assert isinstance(transformed, pd.DataFrame)


# ---------------------------------------------------------
# Fit Transform
# ---------------------------------------------------------

def test_fit_transform(sample_dataframe):

    scaler = FeatureScaler()

    transformed = scaler.fit_transform(sample_dataframe)

    assert transformed.shape == sample_dataframe.shape

    assert isinstance(transformed, pd.DataFrame)


# ---------------------------------------------------------
# Inverse Transform
# ---------------------------------------------------------

def test_inverse_transform(sample_dataframe):

    scaler = FeatureScaler()

    transformed = scaler.fit_transform(sample_dataframe)

    restored = scaler.inverse_transform(transformed)

    np.testing.assert_allclose(
        restored.values,
        sample_dataframe.values,
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------
# Standard Scaler
# ---------------------------------------------------------

def test_standard_scaling(sample_dataframe):

    scaler = FeatureScaler(method="standard")

    transformed = scaler.fit_transform(sample_dataframe)

    assert np.allclose(
        transformed.mean(),
        0,
        atol=1e-6,
    )


# ---------------------------------------------------------
# MinMax Scaler
# ---------------------------------------------------------

def test_minmax_scaling(sample_dataframe):

    scaler = FeatureScaler(method="minmax")

    transformed = scaler.fit_transform(sample_dataframe)

    assert transformed.min().min() >= 0

    assert transformed.max().max() <= 1


# ---------------------------------------------------------
# Robust Scaler
# ---------------------------------------------------------

def test_robust_scaling(sample_dataframe):

    scaler = FeatureScaler(method="robust")

    transformed = scaler.fit_transform(sample_dataframe)

    assert transformed.shape == sample_dataframe.shape


# ---------------------------------------------------------
# Invalid Scaler
# ---------------------------------------------------------

def test_invalid_scaler():

    with pytest.raises(ValueError):

        FeatureScaler(method="unknown")


# ---------------------------------------------------------
# Empty DataFrame
# ---------------------------------------------------------

def test_empty_dataframe():

    scaler = FeatureScaler()

    empty = pd.DataFrame()

    with pytest.raises(DatasetError):

        scaler.fit(empty)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def test_summary(sample_dataframe):

    scaler = FeatureScaler()

    scaler.fit(sample_dataframe)

    scaler.summary()


# ---------------------------------------------------------
# Representation
# ---------------------------------------------------------

def test_repr():

    scaler = FeatureScaler()

    assert "FeatureScaler" in repr(scaler)
"""
Unit Tests for DataValidator.

Tests:
    - Empty dataset validation
    - Target column validation
    - Missing value detection
    - Duplicate row detection
    - Numeric column detection
    - Categorical column detection
    - Data types
    - Dataset shape
    - Memory usage
    - Validation report
"""

import pandas as pd
import pytest

from classical.validation import DataValidator
from utils.exceptions import DatasetError


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_dataframe():
    """
    Create a sample dataframe.
    """

    return pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [50000, 52000, 53000, 51000],
            "city": ["A", "B", "A", "C"],
            "target": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def dataframe_with_missing():
    """
    DataFrame containing missing values.
    """

    return pd.DataFrame(
        {
            "age": [20, None, 22],
            "salary": [100, 200, None],
            "target": [0, 1, 0],
        }
    )


@pytest.fixture
def dataframe_with_duplicates():
    """
    DataFrame containing duplicate rows.
    """

    return pd.DataFrame(
        {
            "feature": [1, 2, 2],
            "target": [0, 1, 1],
        }
    )


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def test_validator_initialization(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    assert validator.df.equals(sample_dataframe)


# ---------------------------------------------------------
# Empty Dataset
# ---------------------------------------------------------

def test_empty_dataset():

    empty = pd.DataFrame()

    validator = DataValidator(empty)

    with pytest.raises(DatasetError):

        validator.check_empty()


# ---------------------------------------------------------
# Target Column
# ---------------------------------------------------------

def test_valid_target(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    validator.check_target("target")


def test_invalid_target(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    with pytest.raises(DatasetError):

        validator.check_target("class")


# ---------------------------------------------------------
# Missing Values
# ---------------------------------------------------------

def test_missing_values(dataframe_with_missing):

    validator = DataValidator(dataframe_with_missing)

    report = validator.missing_values()

    assert report["age"] == 1

    assert report["salary"] == 1


# ---------------------------------------------------------
# Duplicate Rows
# ---------------------------------------------------------

def test_duplicate_rows(dataframe_with_duplicates):

    validator = DataValidator(dataframe_with_duplicates)

    duplicates = validator.duplicate_rows()

    assert duplicates == 1


# ---------------------------------------------------------
# Numeric Columns
# ---------------------------------------------------------

def test_numeric_columns(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    numeric = validator.numeric_columns()

    assert "age" in numeric

    assert "salary" in numeric

    assert "target" in numeric


# ---------------------------------------------------------
# Categorical Columns
# ---------------------------------------------------------

def test_categorical_columns(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    categorical = validator.categorical_columns()

    assert "city" in categorical


# ---------------------------------------------------------
# Data Types
# ---------------------------------------------------------

def test_data_types(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    dtypes = validator.data_types()

    assert "age" in dtypes

    assert "city" in dtypes


# ---------------------------------------------------------
# Dataset Shape
# ---------------------------------------------------------

def test_dataset_shape(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    rows, columns = validator.dataset_shape()

    assert rows == 4

    assert columns == 4


# ---------------------------------------------------------
# Memory Usage
# ---------------------------------------------------------

def test_memory_usage(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    memory = validator.memory_usage()

    assert memory > 0


# ---------------------------------------------------------
# Validation Report
# ---------------------------------------------------------

def test_validation_report(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    report = validator.validation_report("target")

    assert report["rows"] == 4

    assert report["columns"] == 4

    assert "numeric_columns" in report

    assert "categorical_columns" in report

    assert "memory_mb" in report


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def test_summary(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    validator.summary("target")


# ---------------------------------------------------------
# Representation
# ---------------------------------------------------------

def test_repr(sample_dataframe):

    validator = DataValidator(sample_dataframe)

    assert "DataValidator" in repr(validator)
"""
Unit Tests for DatasetLoader.

Tests:
    - Built-in dataset loading
    - CSV loading
    - Excel loading
    - JSON loading
    - Parquet loading
    - Auto loader
    - Invalid dataset handling
    - Invalid file handling
"""

from pathlib import Path

import pandas as pd
import pytest

from classical.dataset_loader import DatasetLoader


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def loader():
    """Create DatasetLoader instance."""
    return DatasetLoader()


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe."""

    return pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        }
    )


# ---------------------------------------------------------
# Built-in Dataset Tests
# ---------------------------------------------------------

def test_load_builtin_dataset(loader):

    df = loader.load_builtin("breast_cancer")

    assert isinstance(df, pd.DataFrame)

    assert not df.empty

    assert "target" in df.columns


def test_builtin_dataset_shape(loader):

    df = loader.load_builtin("breast_cancer")

    assert df.shape[0] > 0

    assert df.shape[1] > 1


def test_invalid_builtin_dataset(loader):

    with pytest.raises(ValueError):

        loader.load_builtin("unknown_dataset")


# ---------------------------------------------------------
# CSV Tests
# ---------------------------------------------------------

def test_load_csv(tmp_path, loader, sample_dataframe):

    file_path = tmp_path / "sample.csv"

    sample_dataframe.to_csv(file_path, index=False)

    df = loader.load_csv(file_path)

    pd.testing.assert_frame_equal(df, sample_dataframe)


# ---------------------------------------------------------
# Excel Tests
# ---------------------------------------------------------

def test_load_excel(tmp_path, loader, sample_dataframe):

    file_path = tmp_path / "sample.xlsx"

    sample_dataframe.to_excel(
        file_path,
        index=False,
    )

    df = loader.load_excel(file_path)

    pd.testing.assert_frame_equal(df, sample_dataframe)


# ---------------------------------------------------------
# JSON Tests
# ---------------------------------------------------------

def test_load_json(tmp_path, loader, sample_dataframe):

    file_path = tmp_path / "sample.json"

    sample_dataframe.to_json(file_path)

    df = loader.load_json(file_path)

    assert isinstance(df, pd.DataFrame)

    assert df.shape == sample_dataframe.shape


# ---------------------------------------------------------
# Parquet Tests
# ---------------------------------------------------------

def test_load_parquet(tmp_path, loader, sample_dataframe):

    file_path = tmp_path / "sample.parquet"

    sample_dataframe.to_parquet(file_path)

    df = loader.load_parquet(file_path)

    pd.testing.assert_frame_equal(df, sample_dataframe)


# ---------------------------------------------------------
# Auto Loader Tests
# ---------------------------------------------------------

def test_auto_loader_csv(
    tmp_path,
    loader,
    sample_dataframe,
):

    file_path = tmp_path / "auto.csv"

    sample_dataframe.to_csv(
        file_path,
        index=False,
    )

    df = loader.load(file_path)

    pd.testing.assert_frame_equal(df, sample_dataframe)


def test_auto_loader_excel(
    tmp_path,
    loader,
    sample_dataframe,
):

    file_path = tmp_path / "auto.xlsx"

    sample_dataframe.to_excel(
        file_path,
        index=False,
    )

    df = loader.load(file_path)

    pd.testing.assert_frame_equal(df, sample_dataframe)


# ---------------------------------------------------------
# Invalid File Tests
# ---------------------------------------------------------

def test_missing_file(loader):

    with pytest.raises(FileNotFoundError):

        loader.load_csv(
            "missing.csv"
        )


def test_invalid_extension(
    tmp_path,
    loader,
):

    file_path = tmp_path / "sample.txt"

    file_path.write_text("HQFSF")

    with pytest.raises(ValueError):

        loader.load(file_path)


# ---------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------

def test_supported_formats(loader):

    formats = loader.supported_formats()

    assert ".csv" in formats

    assert ".xlsx" in formats

    assert ".json" in formats

    assert ".parquet" in formats


def test_builtin_datasets(loader):

    datasets = loader.builtin_datasets()

    assert "breast_cancer" in datasets

    assert "iris" in datasets


def test_repr(loader):

    assert "DatasetLoader" in repr(loader)
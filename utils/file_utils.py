"""
file_utils.py
=============

File utility functions for the
Hybrid Quantum Feature Selection Framework (HQFSF).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# DIRECTORY OPERATIONS
# ==========================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ==========================================================
# JSON OPERATIONS
# ==========================================================

def save_json(
    data: dict,
    filepath: str | Path,
) -> None:
    """
    Save dictionary as JSON.
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_json(
    filepath: str | Path,
) -> dict:
    """
    Load JSON file.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# CSV OPERATIONS
# ==========================================================

def save_csv(
    dataframe: pd.DataFrame,
    filepath: str | Path,
) -> None:
    """
    Save DataFrame as CSV.
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    dataframe.to_csv(
        filepath,
        index=False,
    )


def load_csv(
    filepath: str | Path,
) -> pd.DataFrame:
    """
    Load CSV file.
    """

    return pd.read_csv(filepath)


# ==========================================================
# PICKLE OPERATIONS
# ==========================================================

def save_pickle(
    obj,
    filepath: str | Path,
) -> None:
    """
    Save Python object.
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    with open(
        filepath,
        "wb",
    ) as file:

        pickle.dump(
            obj,
            file,
        )


def load_pickle(
    filepath: str | Path,
):
    """
    Load Pickle object.
    """

    with open(
        filepath,
        "rb",
    ) as file:

        return pickle.load(file)


# ==========================================================
# NUMPY OPERATIONS
# ==========================================================

def save_numpy(
    array: np.ndarray,
    filepath: str | Path,
) -> None:
    """
    Save NumPy array.
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    np.save(
        filepath,
        array,
    )


def load_numpy(
    filepath: str | Path,
) -> np.ndarray:
    """
    Load NumPy array.
    """

    return np.load(
        filepath,
        allow_pickle=True,
    )


# ==========================================================
# TEXT FILES
# ==========================================================

def save_text(
    text: str,
    filepath: str | Path,
) -> None:
    """
    Save text file.
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(text)


def load_text(
    filepath: str | Path,
) -> str:
    """
    Load text file.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


# ==========================================================
# FILE INFORMATION
# ==========================================================

def file_exists(
    filepath: str | Path,
) -> bool:
    """
    Check if a file exists.
    """

    return Path(filepath).exists()


def delete_file(
    filepath: str | Path,
) -> None:
    """
    Delete a file.
    """

    filepath = Path(filepath)

    if filepath.exists():

        filepath.unlink()


def list_files(
    directory: str | Path,
    extension: str | None = None,
):
    """
    List all files in a directory.
    """

    directory = Path(directory)

    if extension is None:

        return sorted(
            [
                file
                for file in directory.iterdir()
                if file.is_file()
            ]
        )

    return sorted(
        directory.glob(f"*{extension}")
    )


def file_size(
    filepath: str | Path,
) -> int:
    """
    Return file size in bytes.
    """

    return Path(filepath).stat().st_size


# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return "HQFSF File Utilities"
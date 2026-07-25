"""
Helper utilities for HQFSF.
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime


def load_yaml(file_path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    file_path : str
        Path to YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict, file_path: str) -> None:
    """
    Save dictionary as JSON.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def file_exists(file_path: str) -> bool:
    """
    Check whether a file exists.
    """
    return os.path.isfile(file_path)


def get_timestamp() -> str:
    """
    Return current timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_directories(paths: list) -> None:
    """
    Create multiple directories.
    """
    for path in paths:
        create_directory(path)
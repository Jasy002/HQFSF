"""
seed.py
=======

Random seed utility for the
Hybrid Quantum Feature Selection Framework (HQFSF).

This module ensures reproducibility across Python,
NumPy, and machine learning experiments.
"""

from __future__ import annotations

import os
import random

import numpy as np


# ==========================================================
# SET RANDOM SEED
# ==========================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    """

    # Python Random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash Seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================================
# GET CURRENT SEED
# ==========================================================

def get_seed() -> int:
    """
    Return the current project seed.

    Returns
    -------
    int
    """

    return int(
        os.environ.get(
            "PYTHONHASHSEED",
            42,
        )
    )


# ==========================================================
# RESET TO DEFAULT
# ==========================================================

def reset_seed() -> None:
    """
    Reset seed to the project default.
    """

    set_seed(42)


# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return "HQFSF Random Seed Utility"
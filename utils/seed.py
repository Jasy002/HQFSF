"""
Utility for reproducibility.
"""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    except Exception:
        pass
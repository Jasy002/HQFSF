"""
helper.py
=========

General helper functions for the
Hybrid Quantum Feature Selection Framework (HQFSF).
"""

from __future__ import annotations

import random
import time
from functools import wraps
from typing import Any

import numpy as np


# ==========================================================
# TIMER DECORATOR
# ==========================================================

def timer(func):
    """
    Decorator to measure execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()

        print(
            f"{func.__name__} executed in "
            f"{end - start:.4f} seconds."
        )

        return result

    return wrapper


# ==========================================================
# EXECUTION TIME
# ==========================================================

def execution_time(start_time: float) -> float:
    """
    Return elapsed execution time.
    """

    return time.perf_counter() - start_time


# ==========================================================
# NORMALIZE ARRAY
# ==========================================================

def normalize(values):
    """
    Normalize values into the range [0, 1].
    """

    values = np.asarray(values)

    minimum = values.min()

    maximum = values.max()

    if minimum == maximum:

        return np.zeros_like(values)

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ==========================================================
# TOP-K INDICES
# ==========================================================

def top_k(values, k: int):
    """
    Return indices of the top-k values.
    """

    values = np.asarray(values)

    return np.argsort(values)[::-1][:k]


# ==========================================================
# FLATTEN LIST
# ==========================================================

def flatten(items):
    """
    Flatten nested lists.
    """

    output = []

    for item in items:

        if isinstance(item, (list, tuple)):

            output.extend(
                flatten(item)
            )

        else:

            output.append(item)

    return output


# ==========================================================
# PERCENTAGE
# ==========================================================

def percentage(
    part: float,
    total: float,
) -> float:
    """
    Calculate percentage.
    """

    if total == 0:

        return 0.0

    return (part / total) * 100


# ==========================================================
# DICTIONARY MERGE
# ==========================================================

def merge_dicts(*dicts):
    """
    Merge multiple dictionaries.
    """

    merged = {}

    for dictionary in dicts:

        merged.update(dictionary)

    return merged


# ==========================================================
# SPLIT INTO CHUNKS
# ==========================================================

def chunks(data, chunk_size):
    """
    Yield chunks from iterable.
    """

    for i in range(
        0,
        len(data),
        chunk_size,
    ):

        yield data[
            i:i + chunk_size
        ]


# ==========================================================
# RANDOM ID
# ==========================================================

def random_id(
    length: int = 8,
):
    """
    Generate a random numeric identifier.
    """

    return "".join(

        random.choice(
            "0123456789"
        )

        for _ in range(length)
    )


# ==========================================================
# SAFE DIVISION
# ==========================================================

def safe_divide(
    numerator,
    denominator,
):
    """
    Perform safe division.
    """

    if denominator == 0:

        return 0

    return numerator / denominator


# ==========================================================
# CONVERT TO NUMPY
# ==========================================================

def to_numpy(data):
    """
    Convert input into NumPy array.
    """

    return np.asarray(data)


# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return "HQFSF Helper Utilities"
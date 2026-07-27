"""
Unit Tests for Evaluator.

Tests:
    - Initialization
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix
    - Classification Report
    - Evaluate
    - Invalid Inputs
    - Summary
    - __repr__
"""

import numpy as np
import pytest

from classical.evaluator import Evaluator
from utils.exceptions import DatasetError


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def binary_labels():

    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])

    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

    y_prob = np.array(
        [
            0.10,
            0.92,
            0.35,
            0.18,
            0.88,
            0.15,
            0.90,
            0.81,
        ]
    )

    return y_true, y_pred, y_prob


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization():

    evaluator = Evaluator()

    assert isinstance(evaluator, Evaluator)


# ==========================================================
# Accuracy
# ==========================================================

def test_accuracy(binary_labels):

    y_true, y_pred, _ = binary_labels

    evaluator = Evaluator()

    score = evaluator.accuracy(y_true, y_pred)

    assert 0.0 <= score <= 1.0


# ==========================================================
# Precision
# ==========================================================

def test_precision(binary_labels):

    y_true, y_pred, _ = binary_labels

    evaluator = Evaluator()

    score = evaluator.precision(y_true, y_pred)

    assert 0.0 <= score <= 1.0


# ==========================================================
# Recall
# ==========================================================

def test_recall(binary_labels):

    y_true, y_pred, _ = binary_labels

    evaluator = Evaluator()

    score = evaluator.recall(y_true, y_pred)

    assert 0.0 <= score <= 1.0


# ==========================================================
# F1 Score
# ==========================================================

def test_f1_score(binary_labels):

    y_true, y_pred, _ = binary_labels

    evaluator = Evaluator()

    score = evaluator.f1_score(y_true, y_pred)

    assert 0.0 <= score <= 1.0


# ==========================================================
# ROC-AUC
# ==========================================================

def test_roc_auc(binary_labels):

    y_true, _, y_prob = binary_labels

    evaluator = Evaluator()

    score = evaluator.roc_auc(y_true, y_prob)

    assert 0.0 <= score <= 1.0


# ==========================================================
# Confusion Matrix
# ==========================================================

def test_confusion_matrix(binary_labels):

    y_true, y_pred, _ = binary_labels

    evaluator = Evaluator()

    matrix = evaluator.confusion_matrix(y_true, y_pred)

    assert matrix.shape == (2, 2)


# ==========================================================
# Classification Report
# ==========================================================

def test_classification_report(binary_labels):

    y_true, y_pred, _ = binary_labels

    evaluator = Evaluator()

    report = evaluator.classification_report(y_true, y_pred)

    assert isinstance(report, dict)

    assert "accuracy" in report


# ==========================================================
# Complete Evaluation
# ==========================================================

def test_evaluate(binary_labels):

    y_true, y_pred, y_prob = binary_labels

    evaluator = Evaluator()

    results = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "balanced_accuracy",
        "mcc",
    }

    assert expected_keys.issubset(results.keys())


# ==========================================================
# Invalid Length
# ==========================================================

def test_invalid_length():

    evaluator = Evaluator()

    y_true = np.array([0, 1, 1])

    y_pred = np.array([0, 1])

    with pytest.raises(DatasetError):

        evaluator.accuracy(y_true, y_pred)


# ==========================================================
# Empty Arrays
# ==========================================================

def test_empty_arrays():

    evaluator = Evaluator()

    with pytest.raises(DatasetError):

        evaluator.evaluate(
            np.array([]),
            np.array([]),
        )


# ==========================================================
# Summary
# ==========================================================

def test_summary():

    evaluator = Evaluator()

    evaluator.summary()


# ==========================================================
# Representation
# ==========================================================

def test_repr():

    evaluator = Evaluator()

    assert "Evaluator" in repr(evaluator)
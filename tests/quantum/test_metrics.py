"""
Unit tests for QuantumMetrics.
"""

import numpy as np
import pytest

from quantum.metrics import QuantumMetrics


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def metrics():
    return QuantumMetrics()


@pytest.fixture
def y_true():
    return [
        0, 1, 1, 0, 1,
        0, 1, 0, 1, 1
    ]


@pytest.fixture
def y_pred():
    return [
        0, 1, 0, 0, 1,
        0, 1, 1, 1, 1
    ]


@pytest.fixture
def y_score():
    return [
        0.10,
        0.95,
        0.35,
        0.15,
        0.88,
        0.20,
        0.82,
        0.70,
        0.91,
        0.97,
    ]


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def test_initialization(metrics):
    assert isinstance(metrics, QuantumMetrics)


# ---------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------

def test_accuracy(metrics, y_true, y_pred):
    score = metrics.accuracy(y_true, y_pred)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_precision(metrics, y_true, y_pred):
    score = metrics.precision(y_true, y_pred)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_recall(metrics, y_true, y_pred):
    score = metrics.recall(y_true, y_pred)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_f1(metrics, y_true, y_pred):
    score = metrics.f1(y_true, y_pred)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_balanced_accuracy(metrics, y_true, y_pred):
    score = metrics.balanced_accuracy(y_true, y_pred)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_mcc(metrics, y_true, y_pred):
    score = metrics.mcc(y_true, y_pred)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_roc_auc(metrics, y_true, y_score):
    score = metrics.roc_auc(
        y_true,
        y_score,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------

def test_confusion_matrix(
    metrics,
    y_true,
    y_pred,
):
    matrix = metrics.confusion(
        y_true,
        y_pred,
    )

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)


# ---------------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------------

def test_report(
    metrics,
    y_true,
    y_pred,
):
    report = metrics.report(
        y_true,
        y_pred,
    )

    assert isinstance(report, dict)

    assert "accuracy" in report
    assert "macro avg" in report
    assert "weighted avg" in report


# ---------------------------------------------------------------------
# Feature Reduction
# ---------------------------------------------------------------------

def test_feature_reduction(
    metrics,
):
    reduction = metrics.feature_reduction(
        original_features=30,
        selected_features=8,
    )

    expected = (
        (30 - 8)
        / 30
    ) * 100

    assert reduction == expected


def test_invalid_feature_reduction(
    metrics,
):
    with pytest.raises(ValueError):
        metrics.feature_reduction(
            original_features=0,
            selected_features=5,
        )


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

def test_evaluate(
    metrics,
    y_true,
    y_pred,
):
    results = metrics.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        original_features=30,
        selected_features=8,
    )

    assert isinstance(results, dict)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "feature_reduction",
    }

    assert expected_keys.issubset(results.keys())


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def test_summary(
    metrics,
    y_true,
    y_pred,
    capsys,
):
    metrics.summary(
        y_true=y_true,
        y_pred=y_pred,
        original_features=30,
        selected_features=8,
    )

    captured = capsys.readouterr()

    assert "HQFSF EVALUATION SUMMARY" in captured.out
    assert "accuracy" in captured.out.lower()
    assert "Confusion Matrix" in captured.out


# ---------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------

def test_repr(metrics):
    assert repr(metrics) == "QuantumMetrics()"


# ---------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------

def test_accuracy_consistency(
    metrics,
    y_true,
    y_pred,
):
    score1 = metrics.accuracy(
        y_true,
        y_pred,
    )

    score2 = metrics.accuracy(
        y_true,
        y_pred,
    )

    assert score1 == score2


def test_evaluate_consistency(
    metrics,
    y_true,
    y_pred,
):
    result1 = metrics.evaluate(
        y_true,
        y_pred,
        30,
        8,
    )

    result2 = metrics.evaluate(
        y_true,
        y_pred,
        30,
        8,
    )

    assert result1 == result2
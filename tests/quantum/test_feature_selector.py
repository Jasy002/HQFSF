"""
Unit tests for QuantumFeatureSelector.
"""

import numpy as np
import pytest

from quantum.feature_selector import QuantumFeatureSelector


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def importance_scores():
    return np.array([
        0.72,
        0.15,
        0.91,
        0.34,
        0.81,
        0.48,
        0.63,
        0.29,
    ])


@pytest.fixture
def topk_selector():
    return QuantumFeatureSelector(
        strategy="top_k",
        top_k=3,
    )


@pytest.fixture
def threshold_selector():
    return QuantumFeatureSelector(
        strategy="threshold",
        threshold=0.50,
    )


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_topk_initialization(topk_selector):
    assert topk_selector.strategy == "top_k"
    assert topk_selector.top_k == 3


def test_threshold_initialization(threshold_selector):
    assert threshold_selector.strategy == "threshold"
    assert threshold_selector.threshold == 0.50


def test_invalid_strategy():
    with pytest.raises(ValueError):
        QuantumFeatureSelector(
            strategy="invalid"
        )


def test_invalid_topk():
    with pytest.raises(ValueError):
        QuantumFeatureSelector(
            strategy="top_k",
            top_k=0,
        )


def test_invalid_threshold_low():
    with pytest.raises(ValueError):
        QuantumFeatureSelector(
            strategy="threshold",
            threshold=-0.1,
        )


def test_invalid_threshold_high():
    with pytest.raises(ValueError):
        QuantumFeatureSelector(
            strategy="threshold",
            threshold=1.5,
        )


# ---------------------------------------------------------------------
# Ranking Tests
# ---------------------------------------------------------------------

def test_rank_features(
    topk_selector,
    importance_scores,
):
    ranking = topk_selector.rank_features(
        importance_scores
    )

    assert isinstance(ranking, list)
    assert len(ranking) == len(importance_scores)

    # Highest score should come first
    assert ranking[0][0] == 2
    assert ranking[0][1] == 0.91


# ---------------------------------------------------------------------
# Top-K Selection Tests
# ---------------------------------------------------------------------

def test_topk_selection(
    topk_selector,
    importance_scores,
):
    selected = topk_selector.select(
        importance_scores
    )

    assert isinstance(selected, np.ndarray)
    assert len(selected) == 3

    expected = {2, 4, 0}

    assert set(selected.tolist()) == expected


def test_topk_exceeds_features():
    selector = QuantumFeatureSelector(
        strategy="top_k",
        top_k=10,
    )

    scores = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        selector.select(scores)


# ---------------------------------------------------------------------
# Threshold Selection Tests
# ---------------------------------------------------------------------

def test_threshold_selection(
    threshold_selector,
    importance_scores,
):
    selected = threshold_selector.select(
        importance_scores
    )

    expected = {0, 2, 4, 6}

    assert set(selected.tolist()) == expected


# ---------------------------------------------------------------------
# Feature Score Mapping
# ---------------------------------------------------------------------

def test_feature_scores(
    topk_selector,
    importance_scores,
):
    scores = topk_selector.feature_scores(
        importance_scores
    )

    assert isinstance(scores, dict)
    assert len(scores) == len(importance_scores)

    assert scores["Feature_0"] == 0.72
    assert scores["Feature_2"] == 0.91


def test_selected_scores(
    topk_selector,
    importance_scores,
):
    scores = topk_selector.selected_scores(
        importance_scores
    )

    assert isinstance(scores, dict)
    assert len(scores) == 3

    assert 2 in scores
    assert scores[2] == 0.91


# ---------------------------------------------------------------------
# Summary Tests
# ---------------------------------------------------------------------

def test_summary_topk(
    topk_selector,
    capsys,
):
    topk_selector.summary()

    captured = capsys.readouterr()

    assert "QUANTUM FEATURE SELECTOR SUMMARY" in captured.out
    assert "Strategy" in captured.out
    assert "Top-K" in captured.out


def test_summary_threshold(
    threshold_selector,
    capsys,
):
    threshold_selector.summary()

    captured = capsys.readouterr()

    assert "Threshold" in captured.out


# ---------------------------------------------------------------------
# Representation Tests
# ---------------------------------------------------------------------

def test_repr(topk_selector):
    representation = repr(topk_selector)

    assert "QuantumFeatureSelector" in representation
    assert "strategy='top_k'" in representation
    assert "top_k=3" in representation


# ---------------------------------------------------------------------
# Consistency Tests
# ---------------------------------------------------------------------

def test_same_input_same_output(
    topk_selector,
    importance_scores,
):
    first = topk_selector.select(
        importance_scores
    )

    second = topk_selector.select(
        importance_scores
    )

    assert np.array_equal(
        first,
        second,
    )


def test_ranking_sorted(
    topk_selector,
    importance_scores,
):
    ranking = topk_selector.rank_features(
        importance_scores
    )

    values = [score for _, score in ranking]

    assert values == sorted(
        values,
        reverse=True,
    )
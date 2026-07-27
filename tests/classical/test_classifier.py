"""
Unit Tests for Classifier.

Tests:
    - Initialization
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    - K-Nearest Neighbors
    - Model Training
    - Prediction
    - Probability Prediction
    - Score
    - Invalid Model
    - Empty Dataset
    - Summary
    - __repr__
"""

import pandas as pd
import pytest

from classical.classifier import Classifier
from utils.exceptions import DatasetError


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def dataset():

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)

    X = data.data

    y = data.target

    return X, y


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization():

    model = Classifier()

    assert isinstance(model, Classifier)


def test_logistic_initialization():

    model = Classifier(model_name="logistic")

    assert model.model_name == "logistic"


def test_random_forest_initialization():

    model = Classifier(model_name="random_forest")

    assert model.model_name == "random_forest"


def test_decision_tree_initialization():

    model = Classifier(model_name="decision_tree")

    assert model.model_name == "decision_tree"


def test_svm_initialization():

    model = Classifier(model_name="svm")

    assert model.model_name == "svm"


def test_knn_initialization():

    model = Classifier(model_name="knn")

    assert model.model_name == "knn"


# ==========================================================
# Model Training
# ==========================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic",
        "decision_tree",
        "random_forest",
        "svm",
        "knn",
    ],
)
def test_model_training(dataset, model_name):

    X, y = dataset

    clf = Classifier(model_name=model_name)

    clf.fit(X, y)

    assert clf.model is not None


# ==========================================================
# Prediction
# ==========================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic",
        "decision_tree",
        "random_forest",
        "svm",
        "knn",
    ],
)
def test_prediction(dataset, model_name):

    X, y = dataset

    clf = Classifier(model_name=model_name)

    clf.fit(X, y)

    prediction = clf.predict(X)

    assert len(prediction) == len(y)


# ==========================================================
# Probability Prediction
# ==========================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic",
        "decision_tree",
        "random_forest",
        "knn",
    ],
)
def test_predict_proba(dataset, model_name):

    X, y = dataset

    clf = Classifier(model_name=model_name)

    clf.fit(X, y)

    probabilities = clf.predict_proba(X)

    assert probabilities.shape[0] == len(X)

    assert probabilities.shape[1] == 2


# ==========================================================
# Accuracy Score
# ==========================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "logistic",
        "decision_tree",
        "random_forest",
        "svm",
        "knn",
    ],
)
def test_score(dataset, model_name):

    X, y = dataset

    clf = Classifier(model_name=model_name)

    clf.fit(X, y)

    score = clf.score(X, y)

    assert 0.0 <= score <= 1.0


# ==========================================================
# Invalid Model
# ==========================================================

def test_invalid_model():

    with pytest.raises(ValueError):

        Classifier(model_name="invalid_model")


# ==========================================================
# Empty Dataset
# ==========================================================

def test_empty_dataset():

    clf = Classifier()

    X = pd.DataFrame()

    y = pd.Series(dtype=int)

    with pytest.raises(DatasetError):

        clf.fit(X, y)


# ==========================================================
# Summary
# ==========================================================

def test_summary():

    clf = Classifier()

    clf.summary()


# ==========================================================
# Representation
# ==========================================================

def test_repr():

    clf = Classifier()

    assert "Classifier" in repr(clf)
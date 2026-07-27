"""
Unit Tests for Trainer.

Tests:
    - Initialization
    - Model Training
    - Training with Validation Set
    - Training Time
    - Model Saving
    - Model Loading
    - Training History
    - Reproducibility
    - Invalid Model
    - Empty Dataset
    - Summary
    - __repr__
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from classical.classifier import Classifier
from classical.trainer import Trainer
from utils.exceptions import DatasetError


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def dataset():
    """Load Breast Cancer dataset."""

    data = load_breast_cancer(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def classifier():
    """Create classifier."""

    return Classifier(model_name="random_forest")


@pytest.fixture
def trainer(classifier):
    """Create trainer."""

    return Trainer(classifier)


# ==========================================================
# Initialization
# ==========================================================

def test_trainer_initialization(trainer):

    assert trainer is not None


# ==========================================================
# Training
# ==========================================================

def test_training(dataset, trainer):

    X_train, _, y_train, _ = dataset

    trainer.fit(X_train, y_train)

    assert trainer.model is not None


# ==========================================================
# Prediction
# ==========================================================

def test_prediction(dataset, trainer):

    X_train, X_test, y_train, _ = dataset

    trainer.fit(X_train, y_train)

    predictions = trainer.predict(X_test)

    assert len(predictions) == len(X_test)


# ==========================================================
# Probability Prediction
# ==========================================================

def test_predict_proba(dataset, trainer):

    X_train, X_test, y_train, _ = dataset

    trainer.fit(X_train, y_train)

    probabilities = trainer.predict_proba(X_test)

    assert probabilities.shape[0] == len(X_test)

    assert probabilities.shape[1] == 2


# ==========================================================
# Score
# ==========================================================

def test_score(dataset, trainer):

    X_train, X_test, y_train, y_test = dataset

    trainer.fit(X_train, y_train)

    score = trainer.score(X_test, y_test)

    assert 0.0 <= score <= 1.0


# ==========================================================
# Training Time
# ==========================================================

def test_training_time(dataset, trainer):

    X_train, _, y_train, _ = dataset

    trainer.fit(X_train, y_train)

    assert trainer.training_time >= 0


# ==========================================================
# Save Model
# ==========================================================

def test_save_model(dataset, trainer, tmp_path):

    X_train, _, y_train, _ = dataset

    trainer.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"

    trainer.save(model_path)

    assert model_path.exists()


# ==========================================================
# Load Model
# ==========================================================

def test_load_model(dataset, trainer, tmp_path):

    X_train, X_test, y_train, _ = dataset

    trainer.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"

    trainer.save(model_path)

    new_trainer = Trainer()

    new_trainer.load(model_path)

    predictions = new_trainer.predict(X_test)

    assert len(predictions) == len(X_test)


# ==========================================================
# Reproducibility
# ==========================================================

def test_reproducibility(dataset):

    X_train, X_test, y_train, _ = dataset

    trainer1 = Trainer(
        Classifier(
            model_name="random_forest",
            random_state=42,
        )
    )

    trainer2 = Trainer(
        Classifier(
            model_name="random_forest",
            random_state=42,
        )
    )

    trainer1.fit(X_train, y_train)

    trainer2.fit(X_train, y_train)

    pred1 = trainer1.predict(X_test)

    pred2 = trainer2.predict(X_test)

    assert (pred1 == pred2).all()


# ==========================================================
# Empty Dataset
# ==========================================================

def test_empty_dataset(trainer):

    X = pd.DataFrame()

    y = pd.Series(dtype=int)

    with pytest.raises(DatasetError):

        trainer.fit(X, y)


# ==========================================================
# Invalid Model
# ==========================================================

def test_invalid_model():

    with pytest.raises(TypeError):

        Trainer(None)


# ==========================================================
# Summary
# ==========================================================

def test_summary(trainer):

    trainer.summary()


# ==========================================================
# Representation
# ==========================================================

def test_repr(trainer):

    assert "Trainer" in repr(trainer)
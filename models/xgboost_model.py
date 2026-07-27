"""
XGBoost Model for HQFSF.

Implements an XGBoost classifier.
"""

from __future__ import annotations

import numpy as np

from xgboost import XGBClassifier

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost Classifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
        random_state: int = 42,
    ):

        super().__init__()

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            use_label_encoder=False,
        )

        logger.info(
            "XGBoostModel initialized."
        )

    # ----------------------------------------------------------
    # Train Model
    # ----------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train the XGBoost classifier.
        """

        logger.info(
            "Training XGBoost..."
        )

        self.model.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Training completed."
        )

    # ----------------------------------------------------------
    # Predict
    # ----------------------------------------------------------

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        logger.info(
            "Generating predictions..."
        )

        return self.model.predict(
            X_test
        )

    # ----------------------------------------------------------
    # Predict Probabilities
    # ----------------------------------------------------------

    def predict_probability(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        """

        return self.model.predict_proba(
            X_test
        )

    # ----------------------------------------------------------
    # Feature Importance
    # ----------------------------------------------------------

    def feature_importance(
        self,
    ) -> np.ndarray:
        """
        Return feature importance scores.
        """

        return self.model.feature_importances_

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print(" XGBoost Model ")
        print("=" * 60)

        params = self.model.get_params()

        for key, value in params.items():
            print(f"{key:25}: {value}")

        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        return (
            "XGBoostModel("
            f"n_estimators={self.model.n_estimators}, "
            f"learning_rate={self.model.learning_rate}, "
            f"max_depth={self.model.max_depth})"
        )
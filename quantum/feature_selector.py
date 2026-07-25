"""
Quantum Feature Selector for HQFSF.

Responsible for:
    - Ranking features
    - Selecting top-k features
    - Threshold-based feature selection
"""

from __future__ import annotations

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumFeatureSelector:
    """
    Quantum Feature Selector.

    Parameters
    ----------
    strategy : str
        Selection strategy.
        Supported:
            - top_k
            - threshold

    top_k : int
        Number of features to select.

    threshold : float
        Minimum importance score.
    """

    SUPPORTED_STRATEGIES = (
        "top_k",
        "threshold",
    )

    def __init__(
        self,
        strategy: str = "top_k",
        top_k: int = 5,
        threshold: float = 0.5,
    ):

        self.strategy = strategy.lower()

        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.top_k = top_k
        self.threshold = threshold

        logger.info(
            "Feature Selector initialized | "
            "Strategy=%s",
            self.strategy.upper(),
        )

    def rank_features(
        self,
        importance_scores: np.ndarray,
    ):
        """
        Rank features in descending order.

        Returns
        -------
        list
            List of (feature_index, score)
        """

        importance_scores = np.asarray(
            importance_scores,
            dtype=float,
        )

        ranking = sorted(
            enumerate(importance_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.info(
            "Features ranked."
        )

        return ranking

    def select(
        self,
        importance_scores: np.ndarray,
    ):
        """
        Select features.

        Returns
        -------
        numpy.ndarray
            Selected feature indices.
        """

        importance_scores = np.asarray(
            importance_scores,
            dtype=float,
        )

        if self.strategy == "top_k":

            indices = np.argsort(
                importance_scores
            )[::-1]

            selected = indices[: self.top_k]

        elif self.strategy == "threshold":

            selected = np.where(
                importance_scores >= self.threshold
            )[0]

        logger.info(
            "%d feature(s) selected.",
            len(selected),
        )

        return selected

    def feature_scores(
        self,
        importance_scores: np.ndarray,
    ):
        """
        Return feature-score mapping.
        """

        importance_scores = np.asarray(
            importance_scores,
            dtype=float,
        )

        return {
            f"Feature_{i}": float(score)
            for i, score in enumerate(
                importance_scores
            )
        }

    def summary(self):

        print("\n========== Feature Selector ==========")

        print(f"Strategy  : {self.strategy}")

        if self.strategy == "top_k":
            print(f"Top-K     : {self.top_k}")

        else:
            print(f"Threshold : {self.threshold}")

        print("======================================\n")
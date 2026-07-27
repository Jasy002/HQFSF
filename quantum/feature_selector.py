"""
Quantum Feature Selector for HQFSF.

Responsible for

    - Ranking features
    - Selecting top-k features
    - Threshold-based feature selection
    - Returning selected feature scores
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumFeatureSelector:
    """
    Quantum Feature Selector.

    Parameters
    ----------
    strategy : str, default="top_k"

        Feature selection strategy.

        Supported:

        - top_k
        - threshold

    top_k : int, default=5

        Number of features to select.

    threshold : float, default=0.5

        Minimum feature importance score.
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
    ) -> None:

        self.strategy = strategy.lower()

        if self.strategy not in self.SUPPORTED_STRATEGIES:

            raise ValueError(
                f"Unsupported strategy '{strategy}'. "
                f"Supported strategies: {self.SUPPORTED_STRATEGIES}"
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.top_k = top_k
        self.threshold = threshold

        logger.info(
            "QuantumFeatureSelector initialized | "
            "Strategy=%s",
            self.strategy.upper(),
        )

    # ---------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------

    def rank_features(
        self,
        importance_scores: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """
        Rank features in descending order.

        Parameters
        ----------
        importance_scores : np.ndarray

        Returns
        -------
        list
            List of (feature_index, score).
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
            "Feature ranking completed."
        )

        return ranking

    # ---------------------------------------------------------
    # Selection
    # ---------------------------------------------------------

    def select(
        self,
        importance_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Select important features.

        Parameters
        ----------
        importance_scores : np.ndarray

        Returns
        -------
        np.ndarray
            Selected feature indices.
        """

        importance_scores = np.asarray(
            importance_scores,
            dtype=float,
        )

        n_features = len(importance_scores)

        if self.strategy == "top_k":

            if self.top_k > n_features:

                raise ValueError(
                    f"top_k ({self.top_k}) exceeds "
                    f"number of features ({n_features})."
                )

            selected = np.argsort(
                importance_scores
            )[::-1][: self.top_k]

        else:

            selected = np.where(
                importance_scores >= self.threshold
            )[0]

        logger.info(
            "%d feature(s) selected.",
            len(selected),
        )

        return selected

    # ---------------------------------------------------------
    # Score Mapping
    # ---------------------------------------------------------

    def feature_scores(
        self,
        importance_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Return mapping between feature names and scores.
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

    # ---------------------------------------------------------
    # Selected Scores
    # ---------------------------------------------------------

    def selected_scores(
        self,
        importance_scores: np.ndarray,
    ) -> Dict[int, float]:
        """
        Return selected feature indices with scores.
        """

        importance_scores = np.asarray(
            importance_scores,
            dtype=float,
        )

        selected = self.select(
            importance_scores
        )

        return {

            int(index): float(
                importance_scores[index]
            )

            for index in selected

        }

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print selector configuration.
        """

        print("\n" + "=" * 55)
        print("QUANTUM FEATURE SELECTOR SUMMARY")
        print("=" * 55)

        print(f"Strategy : {self.strategy}")

        if self.strategy == "top_k":

            print(f"Top-K    : {self.top_k}")

        else:

            print(f"Threshold: {self.threshold}")

        print("=" * 55)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (

            f"QuantumFeatureSelector("
            f"strategy='{self.strategy}', "
            f"top_k={self.top_k}, "
            f"threshold={self.threshold})"

        )
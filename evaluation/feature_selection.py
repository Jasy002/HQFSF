"""
Feature Selection evaluation.
"""

from __future__ import annotations

from typing import Dict, Any

from .metrics import Metrics


class FeatureSelectionMetrics(Metrics):
    """
    Evaluate selected features.
    """

    def evaluate(
        self,
        original_features,
        selected_features,
    ) -> Dict[str, Any]:

        original = len(original_features)
        selected = len(selected_features)

        reduction = original - selected

        reduction_percentage = (

            reduction / original * 100

            if original > 0

            else 0

        )

        return {

            "original_features": original,

            "selected_features": selected,

            "removed_features": reduction,

            "feature_reduction_percent":
                reduction_percentage,

            "selected_indices":
                list(selected_features),
        }
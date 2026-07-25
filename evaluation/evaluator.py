"""
Overall HQFSF evaluator.
"""

from __future__ import annotations

from typing import Dict, Any

from .classification import ClassificationMetrics
from .feature_selection import FeatureSelectionMetrics
from .runtime import RuntimeMetrics


class Evaluator:
    """
    Complete evaluation pipeline.
    """

    def __init__(self):

        self.classification = ClassificationMetrics()

        self.feature = FeatureSelectionMetrics()

        self.runtime = RuntimeMetrics()

    def evaluate(

        self,

        y_true,

        y_pred,

        y_prob,

        original_features,

        selected_features,

    ) -> Dict[str, Any]:

        results = {}

        results.update(

            self.classification.evaluate(

                y_true,

                y_pred,

                y_prob,

            )

        )

        results.update(

            self.feature.evaluate(

                original_features,

                selected_features,

            )

        )

        results.update(

            self.runtime.evaluate()

        )

        return results

    def summary(self):

        print("\n" + "=" * 70)

        print(" HQFSF Evaluation Summary ")

        print("=" * 70)

        print("Classification Metrics")

        print("✓ Accuracy")

        print("✓ Precision")

        print("✓ Recall")

        print("✓ F1 Score")

        print("✓ ROC-AUC")

        print()

        print("Feature Selection")

        print("✓ Selected Features")

        print("✓ Reduction Percentage")

        print()

        print("Performance")

        print("✓ Runtime")

        print("=" * 70)
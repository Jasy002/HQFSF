"""
Statistical Analysis Module.

Provides descriptive and inferential statistical
analysis for HQFSF experiments.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import (
    ttest_rel,
    wilcoxon,
)


class Statistics:
    """
    Statistical analysis utilities.
    """

    @staticmethod
    def descriptive(values: List[float]) -> Dict:

        values = np.asarray(values)

        return {

            "count": len(values),

            "mean": float(np.mean(values)),

            "median": float(np.median(values)),

            "std": float(np.std(values, ddof=1)),

            "variance": float(np.var(values, ddof=1)),

            "minimum": float(np.min(values)),

            "maximum": float(np.max(values)),

            "range": float(np.max(values) - np.min(values)),
        }

    @staticmethod
    def paired_t_test(
        baseline: List[float],
        proposed: List[float],
    ) -> Dict:
        """
        Paired Student t-test.
        """

        statistic, p_value = ttest_rel(
            baseline,
            proposed,
        )

        return {

            "test": "Paired t-test",

            "t_statistic": float(statistic),

            "p_value": float(p_value),

            "significant": bool(p_value < 0.05),
        }

    @staticmethod
    def wilcoxon_test(
        baseline: List[float],
        proposed: List[float],
    ) -> Dict:
        """
        Wilcoxon Signed-Rank Test.
        """

        statistic, p_value = wilcoxon(
            baseline,
            proposed,
        )

        return {

            "test": "Wilcoxon Signed-Rank",

            "statistic": float(statistic),

            "p_value": float(p_value),

            "significant": bool(p_value < 0.05),
        }

    @staticmethod
    def confidence_interval(
        values: List[float],
        confidence: float = 0.95,
    ) -> Dict:
        """
        Confidence interval using normal approximation.
        """

        values = np.asarray(values)

        mean = np.mean(values)

        std = np.std(values, ddof=1)

        n = len(values)

        z = 1.96 if confidence == 0.95 else 2.58

        margin = z * (std / np.sqrt(n))

        return {

            "mean": float(mean),

            "lower": float(mean - margin),

            "upper": float(mean + margin),

            "confidence": confidence,
        }

    @staticmethod
    def effect_size(
        baseline: List[float],
        proposed: List[float],
    ) -> Dict:
        """
        Cohen's d Effect Size.
        """

        baseline = np.asarray(baseline)

        proposed = np.asarray(proposed)

        pooled_std = np.sqrt(

            (

                np.var(baseline, ddof=1)

                +

                np.var(proposed, ddof=1)

            ) / 2

        )

        d = (

            np.mean(proposed)

            -

            np.mean(baseline)

        ) / pooled_std

        if abs(d) < 0.2:

            interpretation = "Negligible"

        elif abs(d) < 0.5:

            interpretation = "Small"

        elif abs(d) < 0.8:

            interpretation = "Medium"

        else:

            interpretation = "Large"

        return {

            "cohens_d": float(d),

            "interpretation": interpretation,
        }

    @staticmethod
    def compare(
        baseline: List[float],
        proposed: List[float],
    ) -> Dict:
        """
        Complete statistical comparison.
        """

        return {

            "baseline": Statistics.descriptive(
                baseline
            ),

            "proposed": Statistics.descriptive(
                proposed
            ),

            "paired_t_test": Statistics.paired_t_test(
                baseline,
                proposed,
            ),

            "wilcoxon_test": Statistics.wilcoxon_test(
                baseline,
                proposed,
            ),

            "effect_size": Statistics.effect_size(
                baseline,
                proposed,
            ),

            "confidence_interval": Statistics.confidence_interval(
                proposed
            ),
        }

    @staticmethod
    def summary():

        print("\n" + "=" * 70)

        print(" Statistical Analysis ")

        print("=" * 70)

        print("✓ Mean")

        print("✓ Median")

        print("✓ Standard Deviation")

        print("✓ Variance")

        print("✓ Confidence Interval")

        print("✓ Paired t-test")

        print("✓ Wilcoxon Signed-Rank Test")

        print("✓ Cohen's d Effect Size")

        print("=" * 70)
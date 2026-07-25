"""
Unit Test for QuantumFeatureSelector.
"""

import numpy as np

from quantum.feature_selector import QuantumFeatureSelector

importance = np.array([
    0.72,
    0.15,
    0.91,
    0.34,
    0.81,
    0.48,
    0.63,
    0.29,
])

print("=" * 70)
print("TOP-K FEATURE SELECTION")
print("=" * 70)

selector = QuantumFeatureSelector(
    strategy="top_k",
    top_k=3,
)

selector.summary()

ranking = selector.rank_features(
    importance
)

print("Ranking")
print(ranking)

print()

selected = selector.select(
    importance
)

print("Selected Features")
print(selected)

print()

print("Scores")
print(selector.feature_scores(
    importance
))

print()

print("=" * 70)
print("THRESHOLD FEATURE SELECTION")
print("=" * 70)

selector = QuantumFeatureSelector(
    strategy="threshold",
    threshold=0.50,
)

selector.summary()

selected = selector.select(
    importance
)

print("Selected Features")
print(selected)
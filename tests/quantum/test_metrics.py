"""
Unit Test for QuantumMetrics.
"""

from quantum.metrics import QuantumMetrics

y_true = [
    0, 1, 1, 0,
    1, 0, 1, 0,
    1, 1
]

y_pred = [
    0, 1, 0, 0,
    1, 0, 1, 1,
    1, 1
]

metrics = QuantumMetrics()

metrics.summary(
    y_true=y_true,
    y_pred=y_pred,
    original_features=30,
    selected_features=8,
)
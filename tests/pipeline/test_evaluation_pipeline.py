"""
Unit Test for EvaluationPipeline.
"""

import numpy as np

from pipeline.evaluation_pipeline import EvaluationPipeline


def test_evaluation_pipeline():

    print("\n" + "=" * 70)
    print("          TESTING EVALUATION PIPELINE")
    print("=" * 70)

    y_true = np.array(
        [0, 1, 1, 0, 1, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 0, 1, 0, 1, 1]
    )

    pipeline = EvaluationPipeline()

    result = pipeline.run(
        y_true=y_true,
        y_pred=y_pred,
        original_features=20,
        selected_features=8,
    )

    pipeline.summary(
        y_true=y_true,
        y_pred=y_pred,
        original_features=20,
        selected_features=8,
    )

    print("\n========== Results ==========\n")

    for key, value in result.items():
        print(f"{key}:")
        print(value)
        print()

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1_score" in result
    assert "confusion_matrix" in result
    assert "feature_reduction" in result

    print("✓ Evaluation Pipeline Test Passed")


if __name__ == "__main__":
    test_evaluation_pipeline()
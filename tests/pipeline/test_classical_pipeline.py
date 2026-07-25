"""
Unit Test for ClassicalPipeline.
"""

from pipeline.classical_pipeline import ClassicalPipeline


def test_classical_pipeline():
    """
    Test the complete Classical Pipeline.
    """

    print("\n" + "=" * 70)
    print("          TESTING CLASSICAL PIPELINE")
    print("=" * 70)

    pipeline = ClassicalPipeline(
        dataset_path="datasets/sample.csv",
        target_column="target",
        scaler="standard",
        test_size=0.20,
        random_state=42,
    )

    pipeline.summary()

    result = pipeline.run()

    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    feature_names = result["feature_names"]

    print("\n========== Dataset Information ==========\n")

    print(f"Training Features : {X_train.shape}")
    print(f"Testing Features  : {X_test.shape}")
    print(f"Training Labels   : {y_train.shape}")
    print(f"Testing Labels    : {y_test.shape}")

    print("\nFeature Names")
    print(feature_names)

    print("\n=========================================\n")

    # -------------------------------
    # Assertions
    # -------------------------------

    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    assert len(feature_names) == X_train.shape[1]

    print("✓ Classical Pipeline Test Passed")


if __name__ == "__main__":
    test_classical_pipeline()
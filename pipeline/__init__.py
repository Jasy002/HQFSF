"""
Unit Test for ClassicalPipeline.
"""

from pipeline.classical_pipeline import ClassicalPipeline


pipeline = ClassicalPipeline(
    dataset_path="datasets/sample.csv",
    target_column="target",
    scaler="standard",
    test_size=0.2,
)

pipeline.summary()

X_train, X_test, y_train, y_test = pipeline.run()

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("Train Labels :", y_train.shape)
print("Test Labels  :", y_test.shape)
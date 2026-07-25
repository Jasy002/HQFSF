"""
Test script for DataSplitter.
"""

from classical.dataset_loader import DatasetLoader
from classical.preprocessing import DataPreprocessor
from classical.feature_scaling import FeatureScaler
from classical.splitter import DataSplitter

# Load dataset
loader = DatasetLoader()
df = loader.load_builtin("breast_cancer")

# Preprocess
preprocessor = DataPreprocessor(df)
df = preprocessor.preprocess()

# Split features and target
X, y = preprocessor.split_features_target(
    df,
    target_column="target"
)

# Scale features
scaler = FeatureScaler(method="minmax")
X = scaler.fit_transform(X)

# Train-test split
splitter = DataSplitter(
    test_size=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test = splitter.split(X, y)

print("\n========== Dataset Split ==========\n")

print(f"Training Features : {X_train.shape}")
print(f"Testing Features  : {X_test.shape}")

print()

print(f"Training Labels   : {y_train.shape}")
print(f"Testing Labels    : {y_test.shape}")
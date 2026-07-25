from classical.dataset_loader import DatasetLoader
from classical.preprocessing import DataPreprocessor

# Load dataset
loader = DatasetLoader()

df = loader.load_builtin("breast_cancer")

# Preprocess dataset
preprocessor = DataPreprocessor(scaler="minmax")

df = preprocessor.clean(df)

# Split features and target
X, y = preprocessor.split_features_target(
    df,
    target_column="target"
)

# Scale features
X = preprocessor.scale_features(X)

print(X.head())
print()
print(y.head())
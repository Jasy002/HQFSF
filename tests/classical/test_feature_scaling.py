from classical.dataset_loader import DatasetLoader
from classical.feature_scaling import FeatureScaler

loader = DatasetLoader()

df = loader.load_builtin("breast_cancer")

X = df.drop(columns=["target"])

scaler = FeatureScaler("minmax")

X_scaled = scaler.fit_transform(X)

print(X_scaled.head())
from classical.dataset import DatasetLoader
from classical.preprocess import DataPreprocessor

loader = DatasetLoader()

df = loader.load_builtin("breast_cancer")

preprocessor = DataPreprocessor("minmax")

df = preprocessor.clean(df)

X, y = preprocessor.split_features_target(df)

X = preprocessor.scale_features(X)

print(X.head())

print()

print(y.head())
from classical.dataset import DatasetLoader
from classical.preprocess import DataPreprocessor
from classical.splitter import DataSplitter

loader = DatasetLoader()

df = loader.load_builtin("breast_cancer")

preprocessor = DataPreprocessor()

df = preprocessor.clean(df)

X, y = preprocessor.split_features_target(df)

X = preprocessor.scale_features(X)

splitter = DataSplitter()

X_train, X_test, y_train, y_test = splitter.split(X, y)

print()

print("Training Shape :", X_train.shape)

print("Testing Shape  :", X_test.shape)

print()

print("Train Labels :", y_train.shape)

print("Test Labels  :", y_test.shape)
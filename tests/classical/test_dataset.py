from classical.dataset import DatasetLoader

df = DatasetLoader.load_builtin("breast_cancer")

print(df.head())

print()

print(df.shape)
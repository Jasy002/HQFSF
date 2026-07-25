from classical.dataset_loader import DatasetLoader
from classical.cross_validation import CrossValidator

loader = DatasetLoader()

df = loader.load_builtin("breast_cancer")

X = df.drop(columns=["target"])

y = df["target"]

cv = CrossValidator()

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

    print(f"Fold {fold}")

    print(len(train_idx), len(test_idx))
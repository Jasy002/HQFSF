from classical.dataset_loader import DatasetLoader
from classical.validation import DataValidator

loader = DatasetLoader(
    "datasets/raw/breast_cancer.csv"
)

df = loader.load()

validator = DataValidator(df)

report = validator.validation_report("target")

print(report)
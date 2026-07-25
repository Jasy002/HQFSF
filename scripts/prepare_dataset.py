from datasets import (
    DatasetLoader,
    DatasetValidator,
    DatasetPreprocessor,
)

loader = DatasetLoader(
    "datasets/raw/Breast_Cancer_Wisconsin.csv"
)

df = loader.load()

validator = DatasetValidator(df)
validator.validate()

processor = DatasetPreprocessor(df)
processor.run()

print("Dataset preparation completed successfully.")
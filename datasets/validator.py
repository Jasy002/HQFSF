"""
Dataset Validator
"""

import pandas as pd


class DatasetValidator:
    """
    Validate dataset quality.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def validate(self):

        print("=" * 60)

        print("Dataset Validation")

        print("=" * 60)

        print(f"Samples : {len(self.df)}")

        print(f"Features: {self.df.shape[1]}")

        print("\nMissing Values")

        print(self.df.isnull().sum())

        print("\nDuplicate Rows")

        print(self.df.duplicated().sum())

        print("\nData Types")

        print(self.df.dtypes)

        print("=" * 60)

        return True
"""
Dataset Loader
"""

from pathlib import Path

import pandas as pd


class DatasetLoader:
    """
    Load datasets into Pandas DataFrames.
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def load(self):

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        return df

    def shape(self):

        df = self.load()

        return df.shape

    def features(self):

        df = self.load()

        return list(df.columns)

    def target(self):

        df = self.load()

        return df.columns[-1]
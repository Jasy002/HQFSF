"""
Dataset Loader
"""

from pathlib import Path

import pandas as pd


COLUMN_NAMES = [
    "id",
    "diagnosis",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


class DatasetLoader:
    """
    Load datasets into Pandas DataFrames.
    """

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def load(self):
        """
        Load dataset.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        df = pd.read_csv(
            self.dataset_path,
            header=None,
            names=COLUMN_NAMES,
        )

        return df

    def shape(self):
        """
        Return dataset shape.
        """

        return self.load().shape

    def features(self):
        """
        Return feature names (excluding ID and target).
        """

        return COLUMN_NAMES[2:]

    def target(self):
        """
        Return target column name.
        """

        return "diagnosis"

    def info(self):
        """
        Display dataset information.
        """

        df = self.load()

        print("=" * 60)
        print("Dataset Information")
        print("=" * 60)

        print(f"Samples  : {df.shape[0]}")
        print(f"Columns  : {df.shape[1]}")
        print(f"Features : {len(COLUMN_NAMES) - 2}")
        print(f"Target   : diagnosis")

        print("=" * 60)

        return df.info()

    def preview(self, rows=5):
        """
        Display first few rows.
        """

        return self.load().head(rows)
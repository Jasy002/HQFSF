"""
Dataset Exporter

Exports processed datasets into multiple formats.
"""

from pathlib import Path

import pandas as pd


class DatasetExporter:
    """
    Export processed datasets into different file formats.
    """

    def __init__(self, dataframe: pd.DataFrame, output_dir="datasets/processed"):
        self.dataframe = dataframe

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(self, filename="processed_dataset.csv"):
        """
        Export dataset to CSV.
        """
        path = self.output_dir / filename

        self.dataframe.to_csv(
            path,
            index=False
        )

        print(f"CSV exported successfully: {path}")

    def save_excel(self, filename="dataset.xlsx"):
        """
        Export dataset to Excel.
        """
        path = self.output_dir / filename

        self.dataframe.to_excel(
            path,
            index=False
        )

        print(f"Excel exported successfully: {path}")

    def save_json(self, filename="dataset.json"):
        """
        Export dataset to JSON.
        """
        path = self.output_dir / filename

        self.dataframe.to_json(
            path,
            orient="records",
            indent=4
        )

        print(f"JSON exported successfully: {path}")

    def save_parquet(self, filename="dataset.parquet"):
        """
        Export dataset to Apache Parquet.
        """

        path = self.output_dir / filename

        try:
            self.dataframe.to_parquet(
                path,
                index=False
            )

            print(f"Parquet exported successfully: {path}")

        except ImportError:
            print(
                "Parquet export requires 'pyarrow' or "
                "'fastparquet'. Install one of them first."
            )

    def export_all(self):
        """
        Export dataset into all supported formats.
        """

        print("=" * 60)
        print("EXPORTING DATASETS")
        print("=" * 60)

        self.save_csv()

        self.save_excel()

        self.save_json()

        self.save_parquet()

        print("=" * 60)
        print("All datasets exported successfully.")
        print("=" * 60)
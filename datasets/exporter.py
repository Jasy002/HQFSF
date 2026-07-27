"""
Dataset Exporter
"""

from pathlib import Path


class DatasetExporter:
    """
    Export datasets into multiple formats.
    """

    def __init__(
        self,
        dataframe,
        output_dir="datasets/processed",
    ):
        self.df = dataframe

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_csv(
        self,
        filename="dataset.csv",
    ):

        path = self.output_dir / filename

        self.df.to_csv(
            path,
            index=False,
        )

        print(f"CSV saved: {path}")

    def save_excel(
        self,
        filename="dataset.xlsx",
    ):

        path = self.output_dir / filename

        self.df.to_excel(
            path,
            index=False,
        )

        print(f"Excel saved: {path}")

    def save_json(
        self,
        filename="dataset.json",
    ):

        path = self.output_dir / filename

        self.df.to_json(
            path,
            orient="records",
            indent=4,
        )

        print(f"JSON saved: {path}")

    def save_parquet(
        self,
        filename="dataset.parquet",
    ):

        path = self.output_dir / filename

        self.df.to_parquet(
            path,
            index=False,
        )

        print(f"Parquet saved: {path}")

    def export_all(self):

        self.save_csv()

        self.save_excel()

        self.save_json()

        self.save_parquet()

        print("All dataset formats exported successfully.")
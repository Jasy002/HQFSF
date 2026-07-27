"""
Dataset Metadata Generator

Generates metadata files for the HQFSF dataset.
"""

import json
from pathlib import Path
from datetime import datetime


class DatasetMetadata:
    """
    Generate metadata for the dataset.
    """

    def __init__(
        self,
        dataframe,
        output_dir="datasets/metadata",
    ):
        self.df = dataframe

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Feature Names
    # ---------------------------------------------------------

    def save_feature_names(self):
        """
        Save feature names.
        """

        feature_columns = [
            column
            for column in self.df.columns
            if column != "diagnosis"
        ]

        with open(
            self.output_dir / "feature_names.txt",
            "w",
            encoding="utf-8",
        ) as file:

            file.write("Dataset Features\n")
            file.write("=================\n\n")

            for feature in feature_columns:
                file.write(f"{feature}\n")

    # ---------------------------------------------------------
    # Target Information
    # ---------------------------------------------------------

    def save_target_info(self):
        """
        Save target information.
        """

        target = "diagnosis"

        with open(
            self.output_dir / "target_info.txt",
            "w",
            encoding="utf-8",
        ) as file:

            file.write("Target Column\n")
            file.write("=============\n\n")

            file.write(f"{target}\n\n")

            file.write("Classes\n")
            file.write("=======\n")

            for value in sorted(self.df[target].unique()):
                file.write(f"- {value}\n")

    # ---------------------------------------------------------
    # Dataset Information
    # ---------------------------------------------------------

    def save_dataset_info(self):
        """
        Save general dataset information.
        """

        info = {
            "dataset_name":
                "Breast Cancer Wisconsin Diagnostic",
            "source":
                "UCI Machine Learning Repository",
            "generated_on":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "samples":
                int(len(self.df)),
            "features":
                int(self.df.shape[1] - 1),
            "target":
                "diagnosis",
            "missing_values":
                int(
                    self.df.isnull().sum().sum()
                ),
            "duplicate_rows":
                int(
                    self.df.duplicated().sum()
                ),
        }

        with open(
            self.output_dir / "dataset_info.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                info,
                file,
                indent=4,
            )

    # ---------------------------------------------------------
    # Dataset Statistics
    # ---------------------------------------------------------

    def save_statistics(self):
        """
        Save descriptive statistics.
        """

        statistics = {

            "shape": {
                "rows": int(len(self.df)),
                "columns": int(self.df.shape[1]),
            },

            "missing_values":
                self.df.isnull().sum().to_dict(),

            "data_types":
                self.df.dtypes.astype(str).to_dict(),

            "class_distribution":
                self.df["diagnosis"]
                .value_counts()
                .to_dict(),

            "summary":
                self.df
                .describe(include="all")
                .to_dict(),
        }

        with open(
            self.output_dir / "statistics.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                statistics,
                file,
                indent=4,
                default=str,
            )

    # ---------------------------------------------------------
    # Generate All Metadata
    # ---------------------------------------------------------

    def generate(self):
        """
        Generate all metadata files.
        """

        print("=" * 60)
        print("GENERATING DATASET METADATA")
        print("=" * 60)

        self.save_dataset_info()

        self.save_feature_names()

        self.save_target_info()

        self.save_statistics()

        print("=" * 60)
        print("Metadata generated successfully.")
        print(f"Location : {self.output_dir}")
        print("=" * 60)
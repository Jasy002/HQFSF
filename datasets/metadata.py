"""
Dataset Metadata Generator
"""

import json
from pathlib import Path


class DatasetMetadata:
    """
    Generate dataset metadata files.
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

    def save_feature_names(self):

        with open(
            self.output_dir / "feature_names.txt",
            "w",
            encoding="utf-8",
        ) as file:

            for feature in self.df.columns[:-1]:
                file.write(feature + "\n")

    def save_target_info(self):

        target = self.df.columns[-1]

        with open(
            self.output_dir / "target_info.txt",
            "w",
            encoding="utf-8",
        ) as file:

            file.write(f"Target Column : {target}\n\n")

            file.write("Classes\n")

            for value in sorted(self.df[target].unique()):
                file.write(f"- {value}\n")

    def save_dataset_info(self):

        info = {
            "samples": int(len(self.df)),
            "features": int(self.df.shape[1] - 1),
            "target": self.df.columns[-1],
            "missing_values": int(
                self.df.isnull().sum().sum()
            ),
            "duplicates": int(
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

    def save_statistics(self):

        stats = (
            self.df
            .describe(include="all")
            .to_dict()
        )

        with open(
            self.output_dir / "statistics.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                stats,
                file,
                indent=4,
                default=str,
            )

    def generate(self):

        self.save_dataset_info()

        self.save_feature_names()

        self.save_target_info()

        self.save_statistics()

        print("Metadata generated successfully.")
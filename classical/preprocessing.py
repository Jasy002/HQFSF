"""
Preprocessing module for HQFSF.

Performs data cleaning and preprocessing before
feature scaling and quantum encoding.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from utils.logger import setup_logger

logger = setup_logger()


class DataPreprocessor:
    """
    Performs dataset preprocessing.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def normalize_column_names(self):
        """
        Normalize column names.
        """

        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        logger.info("Column names normalized.")

    def remove_duplicates(self):
        """
        Remove duplicate rows.
        """

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        logger.info(
            f"Removed {before-after} duplicate rows."
        )

    def handle_missing_values(self):
        """
        Fill missing values.
        """

        numeric = self.df.select_dtypes(include=["number"]).columns

        categorical = self.df.select_dtypes(exclude=["number"]).columns

        if len(numeric) > 0:

            num_imputer = SimpleImputer(strategy="mean")

            self.df[numeric] = num_imputer.fit_transform(
                self.df[numeric]
            )

        if len(categorical) > 0:

            cat_imputer = SimpleImputer(strategy="most_frequent")

            self.df[categorical] = cat_imputer.fit_transform(
                self.df[categorical]
            )

        logger.info("Missing values handled.")

    def encode_labels(self):
        """
        Encode categorical columns.
        """

        encoder = LabelEncoder()

        categorical = self.df.select_dtypes(
            exclude=["number"]
        ).columns

        for column in categorical:

            self.df[column] = encoder.fit_transform(
                self.df[column]
            )

        logger.info("Categorical features encoded.")

    def preprocess(self):
        """
        Complete preprocessing pipeline.
        """

        self.normalize_column_names()

        self.remove_duplicates()

        self.handle_missing_values()

        self.encode_labels()

        logger.info("Preprocessing completed.")

        return self.df
"""
Dataset Preprocessing Module

Performs:
- Remove ID column
- Remove duplicate rows
- Handle missing values
- Encode categorical features
- Standardize numerical features
- Train/Test split
- Export processed datasets
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from datasets.exporter import DatasetExporter


class DatasetPreprocessor:
    """
    Complete preprocessing pipeline for HQFSF.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        output_dir="datasets/processed",
        test_size=0.20,
        random_state=42,
    ):
        """
        Initialize preprocessing pipeline.
        """

        self.df = dataframe.copy()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.test_size = test_size
        self.random_state = random_state

        self.scaler = StandardScaler()

    # ---------------------------------------------------------
    # Remove ID Column
    # ---------------------------------------------------------

    def remove_id_column(self):
        """
        Remove ID column if present.
        """

        if "id" in self.df.columns:

            self.df.drop(
                columns=["id"],
                inplace=True,
            )

            print("ID column removed.")

        else:
            print("ID column not found.")

    # ---------------------------------------------------------
    # Remove Duplicate Rows
    # ---------------------------------------------------------

    def remove_duplicates(self):
        """
        Remove duplicate records.
        """

        before = len(self.df)

        self.df.drop_duplicates(
            inplace=True
        )

        after = len(self.df)

        removed = before - after

        print(f"Removed {removed} duplicate rows.")

    # ---------------------------------------------------------
    # Handle Missing Values
    # ---------------------------------------------------------

    def fill_missing_values(self):
        """
        Fill missing values.
        """

        numeric_columns = self.df.select_dtypes(
            include="number"
        ).columns

        categorical_columns = self.df.select_dtypes(
            exclude="number"
        ).columns

        # Numerical Columns

        for column in numeric_columns:

            median = self.df[column].median()

            self.df[column] = self.df[column].fillna(
                median
            )

        # Categorical Columns

        for column in categorical_columns:

            mode = self.df[column].mode()[0]

            self.df[column] = self.df[column].fillna(
                mode
            )

        print("Missing values handled successfully.")





    # ---------------------------------------------------------
    # Encode Categorical Columns
    # ---------------------------------------------------------

    def encode_categorical(self):
        """
        Encode categorical features using LabelEncoder.

        Example:
        M -> 1
        B -> 0
        """

        categorical_columns = self.df.select_dtypes(
            exclude="number"
        ).columns

        encoder = LabelEncoder()

        for column in categorical_columns:

            self.df[column] = encoder.fit_transform(
                self.df[column]
            )

            print(f"Encoded column: {column}")

        print("Categorical features encoded successfully.")

    # ---------------------------------------------------------
    # Scale Numerical Features
    # ---------------------------------------------------------

    def scale_features(self):
        """
        Standardize feature columns while keeping
        the diagnosis column unchanged.
        """

        if "diagnosis" not in self.df.columns:
            raise ValueError(
                "Target column 'diagnosis' not found."
            )

        # Separate features and target

        X = self.df.drop(
            columns=["diagnosis"]
        )

        y = self.df["diagnosis"]

        # Standardization

        X_scaled = self.scaler.fit_transform(
            X
        )

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=self.df.index,
        )

        # Reattach target

        processed = X_scaled.copy()

        processed["diagnosis"] = y.values

        print("Feature scaling completed.")

        return processed

    # ---------------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------------

    def split_dataset(self):
        """
        Split processed dataset into
        training and testing datasets.
        """

        processed = self.scale_features()

        X = processed.drop(
            columns=["diagnosis"]
        )

        y = processed["diagnosis"]

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        train = X_train.copy()
        train["diagnosis"] = y_train.values

        test = X_test.copy()
        test["diagnosis"] = y_test.values

        print(
            f"Training Samples : {len(train)}"
        )

        print(
            f"Testing Samples  : {len(test)}"
        )

        return (
            processed,
            train,
            test,
        )


    # ---------------------------------------------------------
    # Save Processed Datasets
    # ---------------------------------------------------------

    def save(self):
        """
        Save processed dataset, training dataset,
        testing dataset, and export into multiple formats.
        """

        (
            processed,
            train,
            test,
        ) = self.split_dataset()

        # ---------------------------------------------
        # Save CSV Files
        # ---------------------------------------------

        processed.to_csv(
            self.output_dir / "processed_dataset.csv",
            index=False,
        )

        train.to_csv(
            self.output_dir / "train.csv",
            index=False,
        )

        test.to_csv(
            self.output_dir / "test.csv",
            index=False,
        )

        # ---------------------------------------------
        # Export Additional Formats
        # ---------------------------------------------

        exporter = DatasetExporter(
            dataframe=processed,
            output_dir=self.output_dir,
        )

        exporter.save_excel()

        exporter.save_json()

        exporter.save_parquet()

        # ---------------------------------------------
        # Display Summary
        # ---------------------------------------------

        print("\n" + "=" * 60)
        print("DATASETS SAVED SUCCESSFULLY")
        print("=" * 60)

        print(f"Processed Dataset : {self.output_dir/'processed_dataset.csv'}")
        print(f"Training Dataset  : {self.output_dir/'train.csv'}")
        print(f"Testing Dataset   : {self.output_dir/'test.csv'}")
        print(f"Excel Dataset     : {self.output_dir/'dataset.xlsx'}")
        print(f"JSON Dataset      : {self.output_dir/'dataset.json'}")
        print(f"Parquet Dataset   : {self.output_dir/'dataset.parquet'}")

        print("=" * 60)

    # ---------------------------------------------------------
    # Run Complete Pipeline
    # ---------------------------------------------------------

    def run(self):
        """
        Execute the complete preprocessing pipeline.
        """

        print("\n" + "=" * 60)
        print("HQFSF DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        # Step 1
        self.remove_id_column()

        # Step 2
        self.remove_duplicates()

        # Step 3
        self.fill_missing_values()

        # Step 4
        self.encode_categorical()

        # Step 5
        self.save()

        print("=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return self.df







    
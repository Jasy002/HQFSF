"""
Dataset Preprocessing Module

Performs:
- Remove duplicate rows
- Handle missing values
- Encode categorical features
- Standardize numerical features
- Train/Test split
- Save processed datasets
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DatasetPreprocessor:
    """
    Dataset preprocessing pipeline.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        output_dir="datasets/processed",
        test_size=0.2,
        random_state=42,
    ):
        self.df = dataframe.copy()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_size = test_size
        self.random_state = random_state

        self.scaler = StandardScaler()

    def remove_duplicates(self):
        """
        Remove duplicate rows.
        """
        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        print(f"Removed {before - after} duplicate rows.")

    def fill_missing_values(self):
        """
        Fill missing values.
        """

        numeric_columns = self.df.select_dtypes(include="number").columns
        categorical_columns = self.df.select_dtypes(exclude="number").columns

        # Numerical columns
        for column in numeric_columns:
            self.df[column] = self.df[column].fillna(
                self.df[column].median()
            )

        # Categorical columns
        for column in categorical_columns:
            self.df[column] = self.df[column].fillna(
                self.df[column].mode()[0]
            )

        print("Missing values handled successfully.")

    def encode_categorical(self):
        """
        Encode categorical columns.
        """

        categorical_columns = self.df.select_dtypes(
            exclude="number"
        ).columns

        encoder = LabelEncoder()

        for column in categorical_columns:
            self.df[column] = encoder.fit_transform(
                self.df[column]
            )

        print("Categorical features encoded.")

    def scale_features(self):
        """
        Scale feature columns only.
        """

        X = self.df.iloc[:, :-1]

        scaled = self.scaler.fit_transform(X)

        X_scaled = pd.DataFrame(
            scaled,
            columns=X.columns,
            index=self.df.index
        )

        return X_scaled

    def split_dataset(self):
        """
        Split dataset into train and test sets.
        """

        X_scaled = self.scale_features()

        y = self.df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return (
            X_scaled,
            X_train,
            X_test,
            y_train,
            y_test
        )

    def save(self):
        """
        Save processed dataset and train/test splits.
        """

        (
            X_scaled,
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.split_dataset()

        # --------------------------
        # Save processed dataset
        # --------------------------

        processed = X_scaled.copy()

        processed["target"] = self.df.iloc[:, -1].values

        processed.to_csv(
            self.output_dir / "processed_dataset.csv",
            index=False,
        )

        # --------------------------
        # Save training set
        # --------------------------

        train = X_train.copy()

        train["target"] = y_train.values

        train.to_csv(
            self.output_dir / "train.csv",
            index=False,
        )

        # --------------------------
        # Save testing set
        # --------------------------

        test = X_test.copy()

        test["target"] = y_test.values

        test.to_csv(
            self.output_dir / "test.csv",
            index=False,
        )

        print("\nProcessed datasets saved successfully.")
        print(f"Processed : {self.output_dir/'processed_dataset.csv'}")
        print(f"Train     : {self.output_dir/'train.csv'}")
        print(f"Test      : {self.output_dir/'test.csv'}")

    def run(self):
        """
        Execute the complete preprocessing pipeline.
        """

        print("=" * 60)
        print("HQFSF DATA PREPROCESSING")
        print("=" * 60)

        self.remove_duplicates()

        self.fill_missing_values()

        self.encode_categorical()

        self.save()

        print("=" * 60)
        print("Preprocessing Completed Successfully")
        print("=" * 60)

        return self.df
"""
Classical Pipeline for HQFSF.

Responsible for:
    - Loading dataset
    - Validating dataset
    - Preprocessing
    - Feature scaling
    - Train-test splitting
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from classical.dataset_loader import DatasetLoader
from classical.validation import DataValidator
from classical.preprocessing import DataPreprocessor
from classical.feature_scaling import FeatureScaler
from classical.splitter import DataSplitter

from utils.logger import get_logger

logger = get_logger(__name__)


class ClassicalPipeline:
    """
    End-to-End Classical Pipeline.

    Workflow
    --------
    Dataset
        │
        ▼
    Dataset Loader
        │
        ▼
    Data Validation
        │
        ▼
    Data Preprocessing
        │
        ▼
    Feature Scaling
        │
        ▼
    Train-Test Split
    """

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        scaler: str = "standard",
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the Classical Pipeline.

        Parameters
        ----------
        dataset_path : str
            Path to dataset.

        target_column : str
            Name of target column.

        scaler : str
            Scaling method.

        test_size : float
            Test split ratio.

        random_state : int
            Random seed.
        """

        self.dataset_path = dataset_path
        self.target_column = target_column

        self.loader = DatasetLoader(dataset_path)

        self.validator = DataValidator()

        self.preprocessor = DataPreprocessor()

        self.scaler = FeatureScaler(
            method=scaler
        )

        self.splitter = DataSplitter(
            test_size=test_size,
            random_state=random_state,
        )

        logger.info(
            "ClassicalPipeline initialized."
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete classical pipeline.

        Returns
        -------
        dict
            Dictionary containing train-test data.
        """

        logger.info("=" * 60)
        logger.info("Starting Classical Pipeline")
        logger.info("=" * 60)

        # --------------------------------------------------
        # Load Dataset
        # --------------------------------------------------

        logger.info("Loading dataset...")

        df = self.loader.load()

        logger.info(
            "Dataset loaded successfully."
        )

        logger.info(
            "Dataset Shape: %s",
            df.shape,
        )

        # --------------------------------------------------
        # Validate Dataset
        # --------------------------------------------------

        logger.info("Validating dataset...")

        self.validator.validate(df)

        logger.info(
            "Dataset validation completed."
        )

        # --------------------------------------------------
        # Check Target Column
        # --------------------------------------------------

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' "
                "not found in dataset."
            )

        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        logger.info("Preprocessing dataset...")

        df = self.preprocessor.preprocess(df)

        logger.info(
            "Preprocessing completed."
        )

        # --------------------------------------------------
        # Split Features & Target
        # --------------------------------------------------

        X = df.drop(
            columns=[self.target_column]
        )

        y = df[self.target_column]

        logger.info(
            "Feature Matrix Shape : %s",
            X.shape,
        )

        logger.info(
            "Target Vector Shape  : %s",
            y.shape,
        )

        # --------------------------------------------------
        # Feature Scaling
        # --------------------------------------------------

        logger.info("Scaling features...")

        X = self.scaler.fit_transform(X)

        logger.info(
            "Feature scaling completed."
        )

        # --------------------------------------------------
        # Train Test Split
        # --------------------------------------------------

        logger.info(
            "Splitting dataset..."
        )

        X_train, X_test, y_train, y_test = (
            self.splitter.split(
                X,
                y,
            )
        )

        logger.info(
            "Train-Test Split completed."
        )

        logger.info(
            "X_train : %s",
            X_train.shape,
        )

        logger.info(
            "X_test  : %s",
            X_test.shape,
        )

        logger.info(
            "y_train : %s",
            y_train.shape,
        )

        logger.info(
            "y_test  : %s",
            y_test.shape,
        )

        logger.info(
            "Classical Pipeline completed successfully."
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
        }

    def summary(self) -> None:
        """
        Display pipeline configuration.
        """

        print("\n" + "=" * 55)
        print("        HQFSF Classical Pipeline")
        print("=" * 55)

        print(f"Dataset Path  : {self.dataset_path}")
        print(f"Target Column : {self.target_column}")
        print(f"Scaler        : {self.scaler.method.upper()}")
        print(f"Test Size     : {self.splitter.test_size}")
        print(f"Random State  : {self.splitter.random_state}")

        print("=" * 55 + "\n")
"""
Classical Pipeline for HQFSF.

Responsible for:
    - Loading dataset
    - Validating dataset
    - Data preprocessing
    - Feature scaling
    - Train-test splitting
"""

from __future__ import annotations

from typing import Dict, Any

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
    """

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        scaler: str = "standard",
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

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

    # ----------------------------------------------------------
    # Run Pipeline
    # ----------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the Classical Pipeline.
        """

        logger.info("=" * 60)
        logger.info("Starting Classical Pipeline")
        logger.info("=" * 60)

        # ----------------------------------------
        # Load Dataset
        # ----------------------------------------

        logger.info("Loading dataset...")

        df = self.loader.load()

        logger.info(
            "Dataset loaded successfully."
        )

        logger.info(
            "Dataset Shape : %s",
            df.shape,
        )

        # ----------------------------------------
        # Validation
        # ----------------------------------------

        logger.info(
            "Validating dataset..."
        )

        self.validator.validate(df)

        logger.info(
            "Dataset validation completed."
        )

        # ----------------------------------------
        # Target Column
        # ----------------------------------------

        if self.target_column not in df.columns:

            raise ValueError(
                f"Target column "
                f"'{self.target_column}' "
                f"not found."
            )

        # ----------------------------------------
        # Preprocessing
        # ----------------------------------------

        logger.info(
            "Preprocessing dataset..."
        )

        df = self.preprocessor.preprocess(df)

        logger.info(
            "Preprocessing completed."
        )

        # ----------------------------------------
        # Features / Labels
        # ----------------------------------------

        X = df.drop(
            columns=[self.target_column]
        )

        y = df[self.target_column]

        logger.info(
            "Feature Matrix Shape : %s",
            X.shape,
        )

        logger.info(
            "Target Vector Shape : %s",
            y.shape,
        )

        # ----------------------------------------
        # Scaling
        # ----------------------------------------

        logger.info(
            "Scaling features..."
        )

        X = self.scaler.fit_transform(X)

        logger.info(
            "Scaling completed."
        )

        # ----------------------------------------
        # Split
        # ----------------------------------------

        logger.info(
            "Splitting dataset..."
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.splitter.split(
            X,
            y,
        )

        logger.info(
            "Train-Test Split completed."
        )

        logger.info(
            "X_train : %s",
            X_train.shape,
        )

        logger.info(
            "X_test : %s",
            X_test.shape,
        )

        logger.info(
            "y_train : %s",
            y_train.shape,
        )

        logger.info(
            "y_test : %s",
            y_test.shape,
        )

        logger.info(
            "Classical Pipeline completed."
        )

        return {

            "X_train": X_train,

            "X_test": X_test,

            "y_train": y_train,

            "y_test": y_test,

            "feature_names": list(
                X.columns
            ),
        }

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self) -> None:

        print("\n" + "=" * 60)
        print(" HQFSF Classical Pipeline ")
        print("=" * 60)

        print(
            f"Dataset Path : {self.dataset_path}"
        )

        print(
            f"Target Column : {self.target_column}"
        )

        print(
            f"Scaler : {self.scaler.method.upper()}"
        )

        print(
            f"Test Size : {self.splitter.test_size}"
        )

        print(
            f"Random State : {self.splitter.random_state}"
        )

        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        return (
            "ClassicalPipeline("
            f"dataset_path='{self.dataset_path}', "
            f"target_column='{self.target_column}')"
        )
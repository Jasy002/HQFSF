"""
==============================================================
HQFSF Dataset Preparation Script

Hybrid Quantum Feature Selection Framework

Loads, validates, preprocesses, and prepares the
dataset for the HQFSF pipeline.
==============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import (
    DatasetLoader,
    DatasetValidator,
    DatasetPreprocessor,
)
from utils.logger import get_logger

logger = get_logger(__name__)

DATASET_PATH = Path("datasets/raw/Breast_Cancer_Wisconsin.csv")


def main() -> int:
    """
    Prepare the dataset for training and evaluation.

    Returns
    -------
    int
        Exit status code.
    """

    logger.info("=" * 70)
    logger.info("HQFSF DATASET PREPARATION")
    logger.info("=" * 70)

    try:

        logger.info("Loading dataset: %s", DATASET_PATH)

        loader = DatasetLoader(DATASET_PATH)

        df = loader.load()

        logger.info("Dataset loaded successfully.")

        logger.info("Validating dataset...")

        validator = DatasetValidator(df)
        validator.validate()

        logger.info("Dataset validation completed.")

        logger.info("Preprocessing dataset...")

        processor = DatasetPreprocessor(df)

        processed_df = processor.run()

        logger.info("Dataset preprocessing completed.")

        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        print(f"Rows    : {processed_df.shape[0]}")
        print(f"Columns : {processed_df.shape[1]}")
        print("=" * 70)

        logger.info("Dataset preparation completed successfully.")

        return 0

    except KeyboardInterrupt:

        logger.warning("Dataset preparation interrupted by user.")

        return 1

    except Exception:

        logger.exception("Dataset preparation failed.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
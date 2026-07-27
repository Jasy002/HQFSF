"""
==============================================================
HQFSF Training Script

Hybrid Quantum Feature Selection Framework

Trains the HQFSF model using the complete pipeline
and stores the trained model for future inference.
==============================================================
"""

from __future__ import annotations

import sys

from pipeline.hqfsf_pipeline import HQFSFPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def print_summary(results: dict) -> None:
    """
    Display a concise training summary.
    """

    print("\n" + "=" * 70)
    print("HQFSF TRAINING SUMMARY")
    print("=" * 70)

    if "model_name" in results:
        print(f"Model              : {results['model_name']}")

    if "selected_features" in results:
        print(f"Selected Features  : {len(results['selected_features'])}")

    if "train_accuracy" in results:
        print(f"Training Accuracy  : {results['train_accuracy']}")

    if "test_accuracy" in results:
        print(f"Testing Accuracy   : {results['test_accuracy']}")

    print("=" * 70)


def main() -> int:
    """
    Execute HQFSF model training.

    Returns
    -------
    int
        Exit status code.
    """

    logger.info("=" * 70)
    logger.info("HQFSF TRAINING")
    logger.info("=" * 70)

    try:

        logger.info("Initializing HQFSF pipeline...")

        pipeline = HQFSFPipeline()

        logger.info("Training model...")

        results = pipeline.run()

        logger.info("Training completed successfully.")

        print_summary(results)

        print("\nTraining Finished Successfully.")

        return 0

    except KeyboardInterrupt:

        logger.warning("Training interrupted by user.")
        return 1

    except Exception:

        logger.exception("Training failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
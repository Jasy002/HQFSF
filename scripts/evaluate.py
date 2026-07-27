"""
==============================================================
HQFSF Evaluation Script

Hybrid Quantum Feature Selection Framework

Evaluates the trained HQFSF model using the
evaluation module and reports performance metrics.
==============================================================
"""

from __future__ import annotations

import sys

from evaluation.evaluator import Evaluator
from pipeline.hqfsf_pipeline import HQFSFPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def print_report(metrics: dict) -> None:
    """
    Display evaluation metrics in a readable format.
    """

    print("\n" + "=" * 70)
    print("HQFSF EVALUATION REPORT")
    print("=" * 70)

    for metric, value in metrics.items():
        print(f"{metric:<30}: {value}")

    print("=" * 70)


def main() -> int:
    """
    Execute the HQFSF evaluation pipeline.

    Returns
    -------
    int
        Exit status code.
    """

    logger.info("=" * 70)
    logger.info("HQFSF EVALUATION")
    logger.info("=" * 70)

    try:

        logger.info("Executing HQFSF pipeline...")

        pipeline = HQFSFPipeline()

        results = pipeline.run()

        logger.info("Pipeline execution completed.")

        evaluator = Evaluator()

        logger.info("Computing evaluation metrics...")

        evaluation = evaluator.evaluate(
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            y_prob=results.get("y_prob"),
            original_features=results["original_features"],
            selected_features=results["selected_features"],
        )

        print_report(evaluation)

        logger.info("Evaluation completed successfully.")

        return 0

    except KeyboardInterrupt:

        logger.warning("Evaluation interrupted by user.")
        return 1

    except Exception:

        logger.exception("Evaluation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
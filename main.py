"""
==============================================================
HQFSF Main Entry Point

Hybrid Quantum Feature Selection Framework
Using Variational Quantum Circuits

Author  : Jasmine Sultana
Version : 1.0.0
License : MIT
==============================================================
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from pipeline.hqfsf_pipeline import HQFSFPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_NAME = "Hybrid Quantum Feature Selection Framework"
CONFIG_DIR = Path("configs")


def print_banner() -> None:
    """Display the application banner."""

    print("=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)


def print_summary(results: dict) -> None:
    """
    Print execution summary.
    """

    print("\n" + "=" * 70)
    print("HQFSF EXECUTION SUMMARY")
    print("=" * 70)

    if not results:
        print("No results were returned.")
        return

    # ------------------------------------------------------
    # Selected Features
    # ------------------------------------------------------

    selected = results.get("selected_features")

    if selected is not None:

        print("\nSelected Features")
        print("-" * 70)
        print(selected)

    # ------------------------------------------------------
    # Feature Ranking
    # ------------------------------------------------------

    ranking = results.get("ranking")

    if ranking is not None:

        print("\nFeature Ranking")
        print("-" * 70)
        print(ranking)

    # ------------------------------------------------------
    # Importance Scores
    # ------------------------------------------------------

    scores = results.get("importance_scores")

    if scores is not None:

        print("\nImportance Scores")
        print("-" * 70)
        print(scores)

    # ------------------------------------------------------
    # Evaluation Metrics
    # ------------------------------------------------------

    evaluation = results.get("evaluation")

    if isinstance(evaluation, dict):

        print("\nEvaluation Metrics")
        print("-" * 70)

        for metric, value in evaluation.items():
            print(f"{metric:<25}: {value}")

    print("\nExecution Completed Successfully.")
    print("=" * 70)


def main() -> int:
    """
    Execute the complete HQFSF workflow.

    Returns
    -------
    int
        Exit status code.
    """

    print_banner()

    logger.info("Starting HQFSF framework.")
    logger.info("Loading configuration from '%s'.", CONFIG_DIR)

    try:

        # --------------------------------------------------
        # Initialize Pipeline
        # --------------------------------------------------

        pipeline = HQFSFPipeline(
            config_path=CONFIG_DIR
        )

        logger.info("Pipeline initialized successfully.")

        # Optional summary if implemented
        if hasattr(pipeline, "summary"):
            pipeline.summary()

        # --------------------------------------------------
        # Execute Pipeline
        # --------------------------------------------------

        logger.info("Executing HQFSF pipeline...")

        results = pipeline.run()

        logger.info("Pipeline execution completed.")

        print_summary(results)

        logger.info("HQFSF completed successfully.")

        return 0

    except KeyboardInterrupt:

        logger.warning("Execution interrupted by user.")

        print("\nExecution cancelled by user.")

        return 1

    except Exception as exc:

        logger.exception("Unexpected error occurred.")

        print("\n" + "=" * 70)
        print("HQFSF EXECUTION FAILED")
        print("=" * 70)

        print(f"\nError: {exc}\n")

        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
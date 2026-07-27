"""
==============================================================
HQFSF Complete Pipeline Runner

Hybrid Quantum Feature Selection Framework

Executes the complete HQFSF workflow:

1. Dataset Preparation
2. Classical Preprocessing
3. Quantum Feature Selection
4. Model Training
5. Model Evaluation
6. Report Generation
==============================================================
"""

from __future__ import annotations

import sys

from pipeline.hqfsf_pipeline import HQFSFPipeline
from visualization.report import ReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


def print_summary(results: dict) -> None:
    """
    Print a concise summary of pipeline execution.
    """

    print("\n" + "=" * 70)
    print("HQFSF PIPELINE SUMMARY")
    print("=" * 70)

    if "selected_features" in results:
        print(f"Selected Features : {len(results['selected_features'])}")

    if "evaluation" in results:

        print("\nEvaluation Metrics")
        print("-" * 70)

        for metric, value in results["evaluation"].items():
            print(f"{metric:<25}: {value}")

    print("=" * 70)


def main() -> int:
    """
    Execute the complete HQFSF workflow.

    Returns
    -------
    int
        Exit status code.
    """

    logger.info("=" * 70)
    logger.info("RUNNING COMPLETE HQFSF PIPELINE")
    logger.info("=" * 70)

    try:

        logger.info("Initializing HQFSF pipeline...")

        pipeline = HQFSFPipeline()

        logger.info("Executing pipeline...")

        results = pipeline.run()

        logger.info("Generating report...")

        report = ReportGenerator()

        report.generate(results)

        print_summary(results)

        print("\nPipeline Finished Successfully.")
        print("Report Generated Successfully.")

        logger.info("HQFSF pipeline completed successfully.")

        return 0

    except KeyboardInterrupt:

        logger.warning("Pipeline execution interrupted by user.")

        return 1

    except Exception:

        logger.exception("Pipeline execution failed.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
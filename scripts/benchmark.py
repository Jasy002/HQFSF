"""
==============================================================
HQFSF Benchmark Script

Hybrid Quantum Feature Selection Framework

Runs benchmark experiments using all supported
machine learning models and reports comparative results.
==============================================================
"""

from __future__ import annotations

import sys

from experiments.benchmark import BenchmarkExperiment
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """
    Execute benchmark experiments.

    Returns
    -------
    int
        Exit status code.
    """

    logger.info("=" * 70)
    logger.info("HQFSF BENCHMARK")
    logger.info("=" * 70)

    try:

        benchmark = BenchmarkExperiment()

        logger.info("Running benchmark experiments...")

        results = benchmark.run()

        logger.info("Benchmark completed successfully.")

        print("\n")
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        benchmark.print_results(results)

        best_model = benchmark.best_model(results)

        print("\n" + "=" * 70)
        print("BEST MODEL")
        print("=" * 70)
        print(best_model)

        logger.info("Best model: %s", best_model)

        return 0

    except KeyboardInterrupt:

        logger.warning("Benchmark interrupted by user.")
        return 1

    except Exception:

        logger.exception("Benchmark execution failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
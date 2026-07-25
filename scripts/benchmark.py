"""
Run benchmark experiments.
"""

from experiments.benchmark import BenchmarkExperiment

from utils.logger import get_logger

logger = get_logger(__name__)


def main():

    logger.info("=" * 70)
    logger.info("HQFSF BENCHMARK")
    logger.info("=" * 70)

    benchmark = BenchmarkExperiment()

    results = benchmark.run()

    benchmark.print_results(results)

    best = benchmark.best_model(results)

    print("\nBest Model")

    print(best)


if __name__ == "__main__":
    main()
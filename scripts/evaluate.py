"""
Evaluate HQFSF model.
"""

from pipeline.hqfsf_pipeline import HQFSFPipeline
from evaluation.evaluator import Evaluator
from utils.logger import get_logger

logger = get_logger(__name__)


def main():

    logger.info("=" * 70)
    logger.info("HQFSF EVALUATION")
    logger.info("=" * 70)

    pipeline = HQFSFPipeline()

    results = pipeline.run()

    evaluator = Evaluator()

    evaluation = evaluator.evaluate(
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        y_prob=results.get("y_prob"),
        original_features=results["original_features"],
        selected_features=results["selected_features"],
    )

    print("\nEvaluation Results")

    for key, value in evaluation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
"""
Train HQFSF model.
"""

from pipeline.hqfsf_pipeline import HQFSFPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main():

    logger.info("=" * 70)
    logger.info("HQFSF TRAINING")
    logger.info("=" * 70)

    pipeline = HQFSFPipeline()

    results = pipeline.run()

    logger.info("Training completed.")

    print("\nTraining Finished Successfully")

    print(results)


if __name__ == "__main__":
    main()
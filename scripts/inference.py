"""
Inference Script.
"""

import numpy as np

from pipeline.hqfsf_pipeline import HQFSFPipeline

from utils.logger import get_logger

logger = get_logger(__name__)


def predict(sample):

    pipeline = HQFSFPipeline()

    results = pipeline.run()

    model = results["model"]

    prediction = model.predict(sample)

    return prediction


def main():

    logger.info("=" * 70)
    logger.info("HQFSF INFERENCE")
    logger.info("=" * 70)

    sample = np.random.rand(1, 10)

    prediction = predict(sample)

    print("\nPrediction")

    print(prediction)


if __name__ == "__main__":
    main()
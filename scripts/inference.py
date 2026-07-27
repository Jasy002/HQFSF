"""
==============================================================
HQFSF Inference Script

Hybrid Quantum Feature Selection Framework

Loads a trained model and performs inference on
new unseen samples.
==============================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = Path("saved_models/random_forest.joblib")


def load_model():
    """
    Load the trained model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    logger.info("Loading model from %s", MODEL_PATH)

    return joblib.load(MODEL_PATH)


def predict(sample: np.ndarray):
    """
    Predict labels for new samples.

    Parameters
    ----------
    sample : np.ndarray

    Returns
    -------
    np.ndarray
    """

    model = load_model()

    prediction = model.predict(sample)

    return prediction


def main() -> int:

    logger.info("=" * 70)
    logger.info("HQFSF INFERENCE")
    logger.info("=" * 70)

    try:

        # Example sample
        sample = np.random.rand(1, 10)

        prediction = predict(sample)

        print("\nPrediction")
        print("-" * 70)
        print(prediction)

        logger.info("Inference completed successfully.")

        return 0

    except KeyboardInterrupt:

        logger.warning("Inference interrupted by user.")

        return 1

    except Exception:

        logger.exception("Inference failed.")

        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
HQFSF Master Pipeline

Coordinates:
    1. Classical Pipeline
    2. Quantum Pipeline
    3. Evaluation Pipeline
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from pipeline.classical_pipeline import ClassicalPipeline
from pipeline.quantum_pipeline import QuantumPipeline
from pipeline.evaluation_pipeline import EvaluationPipeline

from utils.logger import get_logger

logger = get_logger(__name__)


class HQFSFPipeline:
    """
    Complete Hybrid Quantum Feature Selection Framework.
    """

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        n_qubits: int,
        layers: int = 2,
        top_k: int = 5,
    ):

        self.classical_pipeline = ClassicalPipeline(
            dataset_path=dataset_path,
            target_column=target_column,
        )

        self.quantum_pipeline = QuantumPipeline(
            n_qubits=n_qubits,
            layers=layers,
            top_k=top_k,
        )

        self.evaluation_pipeline = EvaluationPipeline()

        logger.info(
            "HQFSFPipeline initialized."
        )

    # ----------------------------------------------------------
    # Execute Complete Pipeline
    # ----------------------------------------------------------

    def run(self) -> Dict[str, Any]:

        logger.info("=" * 60)
        logger.info("Starting HQFSF Pipeline")
        logger.info("=" * 60)

        # --------------------------------------------------
        # Stage 1 : Classical Processing
        # --------------------------------------------------

        classical_result = (
            self.classical_pipeline.run()
        )

        X_train = classical_result["X_train"]
        X_test = classical_result["X_test"]

        y_train = classical_result["y_train"]
        y_test = classical_result["y_test"]

        feature_names = classical_result[
            "feature_names"
        ]

        logger.info(
            "Classical Pipeline completed."
        )

        # --------------------------------------------------
        # Stage 2 : Quantum Feature Selection
        # --------------------------------------------------

        quantum_result = (
            self.quantum_pipeline.run(
                X_train
            )
        )

        selected_features = quantum_result[
            "selected_features"
        ]

        importance_scores = quantum_result[
            "importance_scores"
        ]

        ranking = quantum_result[
            "ranking"
        ]

        logger.info(
            "Quantum Pipeline completed."
        )

        # --------------------------------------------------
        # Placeholder Model Prediction
        # --------------------------------------------------
        #
        # Replace this section with your actual
        # Random Forest / SVM / XGBoost model.
        #

        y_pred = np.copy(y_test)

        # --------------------------------------------------
        # Stage 3 : Evaluation
        # --------------------------------------------------

        evaluation_result = (
            self.evaluation_pipeline.run(
                y_true=y_test,
                y_pred=y_pred,
                original_features=len(
                    feature_names
                ),
                selected_features=len(
                    selected_features
                ),
            )
        )

        logger.info(
            "Evaluation completed."
        )

        logger.info("=" * 60)
        logger.info("HQFSF Pipeline Finished")
        logger.info("=" * 60)

        return {

            "classical": classical_result,

            "quantum": quantum_result,

            "evaluation": evaluation_result,

            "selected_features": selected_features,

            "importance_scores": importance_scores,

            "ranking": ranking,
        }

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print(" Hybrid Quantum Feature Selection Framework ")
        print("=" * 60)

        self.classical_pipeline.summary()

        self.quantum_pipeline.summary()

        print("Evaluation Module : Ready")

        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self):

        return (
            "HQFSFPipeline("
            f"classical={self.classical_pipeline}, "
            f"quantum={self.quantum_pipeline})"
        )
"""
HQFSF Pipeline.

Hybrid Quantum Feature Selection Framework

Workflow
--------
Dataset
    │
    ▼
Classical Pipeline
    │
    ▼
Quantum Pipeline
    │
    ▼
Feature Selection
    │
    ▼
Evaluation Pipeline
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
    Hybrid Quantum Feature Selection Framework Pipeline.
    """

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        n_qubits: int,
        layers: int = 2,
        encoding: str = "ry",
        entanglement: str = "linear",
        backend_type: str = "aer_simulator",
        shots: int = 1024,
        scaler: str = "standard",
        test_size: float = 0.20,
        random_state: int = 42,
        selection_strategy: str = "top_k",
        top_k: int = 5,
        threshold: float = 0.5,
    ):

        self.classical_pipeline = ClassicalPipeline(
            dataset_path=dataset_path,
            target_column=target_column,
            scaler=scaler,
            test_size=test_size,
            random_state=random_state,
        )

        self.quantum_pipeline = QuantumPipeline(
            n_qubits=n_qubits,
            layers=layers,
            encoding=encoding,
            entanglement=entanglement,
            backend_type=backend_type,
            shots=shots,
            selection_strategy=selection_strategy,
            top_k=top_k,
            threshold=threshold,
        )

        self.evaluation_pipeline = EvaluationPipeline()

        logger.info(
            "HQFSF Pipeline initialized."
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the HQFSF pipeline.
        """

        logger.info("=" * 70)
        logger.info("Starting HQFSF Pipeline")
        logger.info("=" * 70)

        # ------------------------------------------------
        # Classical Pipeline
        # ------------------------------------------------

        classical = self.classical_pipeline.run()

        X_train = classical["X_train"]
        X_test = classical["X_test"]

        y_train = classical["y_train"]
        y_test = classical["y_test"]

        feature_names = classical["feature_names"]

        # ------------------------------------------------
        # Quantum Pipeline
        # ------------------------------------------------

        quantum = self.quantum_pipeline.run(
            X_train
        )

        selected_features = quantum[
            "selected_features"
        ]

        importance_scores = quantum[
            "importance_scores"
        ]

        ranking = quantum["ranking"]

        # ------------------------------------------------
        # Placeholder Prediction
        # ------------------------------------------------
        #
        # Replace this section with your classifier
        # (Random Forest, SVM, XGBoost, etc.)
        #

        y_pred = np.copy(y_test)

        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------

        evaluation = self.evaluation_pipeline.run(
            y_true=y_test,
            y_pred=y_pred,
            original_features=len(feature_names),
            selected_features=len(selected_features),
        )

        logger.info(
            "HQFSF Pipeline completed successfully."
        )

        return {

            "classical": classical,

            "quantum": quantum,

            "evaluation": evaluation,

            "selected_features": selected_features,

            "importance_scores": importance_scores,

            "ranking": ranking,
        }

    def summary(self):

        print("\n" + "=" * 70)
        print(" Hybrid Quantum Feature Selection Framework ")
        print("=" * 70)

        print("Pipeline Components")

        print("✓ Classical Pipeline")
        print("✓ Quantum Pipeline")
        print("✓ Evaluation Pipeline")

        print("=" * 70 + "\n")
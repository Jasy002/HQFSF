"""
Quantum Pipeline for HQFSF.

Responsible for:
    - Quantum Feature Encoding
    - Variational Circuit Construction
    - Quantum Execution
    - Quantum Measurement
    - Expectation Value Calculation
    - Quantum Feature Selection
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from quantum.circuit import HQFSFCircuit
from quantum.backend import QuantumBackend
from quantum.measurement import QuantumMeasurement
from quantum.expectation import ExpectationCalculator
from quantum.feature_selector import QuantumFeatureSelector

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumPipeline:
    """
    End-to-End Quantum Processing Pipeline.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        encoding: str = "ry",
        entanglement: str = "linear",
        backend_type: str = "aer_simulator",
        shots: int = 1024,
        selection_strategy: str = "top_k",
        top_k: int = 5,
        threshold: float = 0.50,
    ):

        self.n_qubits = n_qubits
        self.layers = layers
        self.encoding = encoding
        self.entanglement = entanglement

        self.circuit_builder = HQFSFCircuit(
            n_qubits=n_qubits,
            layers=layers,
            encoding=encoding,
            entanglement=entanglement,
        )

        self.backend = QuantumBackend(
            backend_type=backend_type
        )

        self.measurement = QuantumMeasurement(
            backend=self.backend,
            shots=shots,
        )

        self.expectation = ExpectationCalculator()

        self.selector = QuantumFeatureSelector(
            strategy=selection_strategy,
            top_k=top_k,
            threshold=threshold,
        )

        logger.info(
            "QuantumPipeline initialized."
        )

    # -------------------------------------------------------
    # Execute Quantum Pipeline
    # -------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
    ) -> Dict[str, Any]:

        logger.info("=" * 60)
        logger.info("Starting Quantum Pipeline")
        logger.info("=" * 60)

        counts_list = []

        # ----------------------------------------------
        # Execute Quantum Circuit
        # ----------------------------------------------

        for sample in X:

            circuit = self.circuit_builder.build(
                sample
            )

            circuit = self.circuit_builder.measure(
                circuit
            )

            counts = self.measurement.counts(
                circuit
            )

            counts_list.append(
                counts
            )

        logger.info(
            "Quantum execution completed for %d samples.",
            len(counts_list),
        )

        # ----------------------------------------------
        # Expectation Values
        # ----------------------------------------------

        expectation_values = (
            self.expectation.expectation_vector(
                counts_list,
                self.measurement.shots,
            )
        )

        logger.info(
            "Expectation values computed."
        )

        # ----------------------------------------------
        # Feature Importance
        # ----------------------------------------------

        importance_scores = np.abs(
            expectation_values
        )

        ranking = self.selector.rank_features(
            importance_scores
        )

        selected_features = self.selector.select(
            importance_scores
        )

        logger.info(
            "%d feature(s) selected.",
            len(selected_features),
        )

        # ----------------------------------------------
        # Return Results
        # ----------------------------------------------

        return {

            "importance_scores": importance_scores,

            "ranking": ranking,

            "selected_features": selected_features,

            "counts": counts_list,
        }

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    def summary(self):

        print("\n" + "=" * 60)
        print(" HQFSF Quantum Pipeline ")
        print("=" * 60)

        print(f"Qubits              : {self.n_qubits}")
        print(f"Layers              : {self.layers}")
        print(f"Encoding            : {self.encoding}")
        print(f"Entanglement        : {self.entanglement}")
        print(f"Backend             : {self.backend.backend_name()}")
        print(f"Shots               : {self.measurement.shots}")
        print(f"Selection Strategy  : {self.selector.strategy}")

        print("=" * 60 + "\n")

    # -------------------------------------------------------
    # Representation
    # -------------------------------------------------------

    def __repr__(self):

        return (
            "QuantumPipeline("
            f"n_qubits={self.n_qubits}, "
            f"layers={self.layers}, "
            f"encoding='{self.encoding}', "
            f"entanglement='{self.entanglement}')"
        )
"""
Quantum Circuit Builder for HQFSF.

Combines:
    - Quantum Feature Encoder
    - Variational Ansatz

into a complete Variational Quantum Circuit (VQC).
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit

from quantum.encoder import QuantumEncoder
from quantum.ansatz import VariationalAnsatz

from utils.logger import get_logger

logger = get_logger(__name__)


class HQFSFCircuit:
    """
    HQFSF Variational Quantum Circuit.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        encoding: str = "ry",
        entanglement: str = "linear",
    ):

        self.n_qubits = n_qubits
        self.layers = layers
        self.encoding = encoding
        self.entanglement = entanglement

        self.encoder = QuantumEncoder(
            n_qubits=n_qubits
        )

        self.ansatz = VariationalAnsatz(
            n_qubits=n_qubits,
            layers=layers,
            entanglement=entanglement,
        )

        logger.info(
            "HQFSF Circuit initialized (%d qubits).",
            n_qubits,
        )

    def build(
        self,
        features: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build the complete quantum circuit.

        Parameters
        ----------
        features : np.ndarray
            Classical feature vector.

        Returns
        -------
        QuantumCircuit
        """

        encoder_circuit = self.encoder.encode(
            features,
            encoding_method=self.encoding,
        )

        ansatz_circuit = self.ansatz.build()

        full_circuit = QuantumCircuit(self.n_qubits)

        full_circuit.compose(
            encoder_circuit,
            inplace=True,
        )

        full_circuit.barrier()

        full_circuit.compose(
            ansatz_circuit,
            inplace=True,
        )

        logger.info(
            "Complete HQFSF circuit constructed."
        )

        return full_circuit

    def measure(
        self,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        """
        Add measurements.
        """

        circuit.measure_all()

        logger.info(
            "Measurements added."
        )

        return circuit

    @staticmethod
    def draw(
        circuit: QuantumCircuit,
        output: str = "text",
    ):
        """
        Draw the quantum circuit.
        """

        return circuit.draw(output=output)
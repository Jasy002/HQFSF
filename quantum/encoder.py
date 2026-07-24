"""
Quantum feature encoding module for HQFSF.

Supports multiple encoding strategies:
- Ry Angle Encoding
- Rx Angle Encoding
- Rz Angle Encoding
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumEncoder:
    """
    Multi-strategy quantum feature encoder.
    """

    SUPPORTED_METHODS = ("ry", "rx", "rz")

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def encode(
        self,
        features: np.ndarray,
        method: str = "ry",
    ) -> QuantumCircuit:
        """
        Encode a feature vector into a quantum circuit.

        Parameters
        ----------
        features : np.ndarray
            Classical feature vector.

        method : str
            ry | rx | rz

        Returns
        -------
        QuantumCircuit
        """

        features = np.asarray(features, dtype=float)

        if len(features) < self.n_qubits:
            raise ValueError(
                "Feature vector length must be greater than or equal to the number of qubits."
            )

        method = method.lower()

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported encoding '{method}'. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )

        qc = QuantumCircuit(self.n_qubits)

        for qubit in range(self.n_qubits):

            angle = float(features[qubit])

            if method == "ry":
                qc.ry(angle, qubit)

            elif method == "rx":
                qc.rx(angle, qubit)

            elif method == "rz":
                qc.rz(angle, qubit)

        logger.info(
            "%s encoding applied (%d qubits).",
            method.upper(),
            self.n_qubits,
        )

        return qc
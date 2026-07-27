"""
Quantum Feature Encoder for HQFSF.

Supports:
    - RY Encoding
    - RX Encoding
    - RZ Encoding

Encodes classical feature vectors into quantum states.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumEncoder:
    """
    Quantum Feature Encoder.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for feature encoding.

    Notes
    -----
    The first ``n_qubits`` values of the feature vector are encoded
    into quantum rotation gates.
    """

    SUPPORTED_METHODS = (
        "ry",
        "rx",
        "rz",
    )

    def __init__(
        self,
        n_qubits: int,
    ) -> None:

        if not isinstance(n_qubits, int):
            raise TypeError(
                "n_qubits must be an integer."
            )

        if n_qubits <= 0:
            raise ValueError(
                "Number of qubits must be greater than zero."
            )

        self.n_qubits = n_qubits

        logger.info(
            "QuantumEncoder initialized | Qubits=%d",
            self.n_qubits,
        )

    # ---------------------------------------------------------
    # Feature Encoding
    # ---------------------------------------------------------

    def encode(
        self,
        features: np.ndarray,
        encoding_method: str = "ry",
    ) -> QuantumCircuit:
        """
        Encode classical features into a quantum circuit.

        Parameters
        ----------
        features : np.ndarray
            Classical feature vector.

        encoding_method : str, default="ry"
            Encoding strategy.

            Supported methods:

            - ry
            - rx
            - rz

        Returns
        -------
        QuantumCircuit
            Quantum feature encoding circuit.
        """

        features = np.asarray(
            features,
            dtype=float,
        )

        if features.ndim != 1:
            raise ValueError(
                "Features must be a one-dimensional array."
            )

        if len(features) < self.n_qubits:
            raise ValueError(
                f"Expected at least {self.n_qubits} features, "
                f"but received {len(features)}."
            )

        encoding_method = encoding_method.lower()

        if encoding_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported encoding '{encoding_method}'. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )

        qc = QuantumCircuit(self.n_qubits)

        for qubit in range(self.n_qubits):

            angle = float(features[qubit])

            if encoding_method == "ry":

                qc.ry(
                    angle,
                    qubit,
                )

            elif encoding_method == "rx":

                qc.rx(
                    angle,
                    qubit,
                )

            elif encoding_method == "rz":

                qc.rz(
                    angle,
                    qubit,
                )

        logger.info(
            "%s feature encoding applied | Qubits=%d",
            encoding_method.upper(),
            self.n_qubits,
        )

        return qc

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print encoder configuration.
        """

        print("\n" + "=" * 55)
        print("QUANTUM ENCODER SUMMARY")
        print("=" * 55)

        print(f"Qubits              : {self.n_qubits}")
        print(f"Supported Encodings : {self.SUPPORTED_METHODS}")

        print("=" * 55)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"QuantumEncoder("
            f"n_qubits={self.n_qubits})"
        )
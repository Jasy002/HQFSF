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
    Hybrid Quantum Feature Selection Framework (HQFSF)
    Variational Quantum Circuit Builder.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    layers : int, default=2
        Number of variational layers.

    encoding : str, default="ry"
        Feature encoding method.

    entanglement : str, default="linear"
        Entanglement topology.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        encoding: str = "ry",
        entanglement: str = "linear",
    ) -> None:

        if n_qubits <= 0:
            raise ValueError(
                "Number of qubits must be greater than zero."
            )

        if layers <= 0:
            raise ValueError(
                "Number of layers must be greater than zero."
            )

        self.n_qubits = n_qubits
        self.layers = layers
        self.encoding = encoding.lower()
        self.entanglement = entanglement.lower()

        self.encoder = QuantumEncoder(
            n_qubits=self.n_qubits,
        )

        self.ansatz = VariationalAnsatz(
            n_qubits=self.n_qubits,
            layers=self.layers,
            entanglement=self.entanglement,
        )

        logger.info(
            "HQFSF Circuit initialized | "
            "Qubits=%d | Layers=%d",
            self.n_qubits,
            self.layers,
        )

    # ---------------------------------------------------------
    # Build Circuit
    # ---------------------------------------------------------

    def build(
        self,
        features: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build the complete variational quantum circuit.

        Parameters
        ----------
        features : np.ndarray
            Classical feature vector.

        Returns
        -------
        QuantumCircuit
        """

        if len(features) != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features "
                f"but received {len(features)}."
            )

        encoder_circuit = self.encoder.encode(
            features,
            encoding_method=self.encoding,
        )

        ansatz_circuit = self.ansatz.build()

        circuit = QuantumCircuit(self.n_qubits)

        circuit.compose(
            encoder_circuit,
            inplace=True,
        )

        circuit.barrier()

        circuit.compose(
            ansatz_circuit,
            inplace=True,
        )

        logger.info(
            "HQFSF variational circuit constructed successfully."
        )

        return circuit

    # ---------------------------------------------------------
    # Measurement
    # ---------------------------------------------------------

    def measure(
        self,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        """
        Return a measured copy of the circuit.
        """

        measured = circuit.copy()

        measured.measure_all()

        logger.info(
            "Measurements added to circuit."
        )

        return measured

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    @staticmethod
    def draw(
        circuit: QuantumCircuit,
        output: str = "text",
    ):
        """
        Draw the circuit.

        Parameters
        ----------
        circuit : QuantumCircuit

        output : str
            text, mpl, latex

        Returns
        -------
        Circuit drawing.
        """

        return circuit.draw(output=output)

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print circuit configuration.
        """

        print("\n" + "=" * 55)
        print("HQFSF CIRCUIT SUMMARY")
        print("=" * 55)

        print(f"Qubits         : {self.n_qubits}")
        print(f"Layers         : {self.layers}")
        print(f"Encoding       : {self.encoding}")
        print(f"Entanglement   : {self.entanglement}")

        print("=" * 55)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"HQFSFCircuit("
            f"n_qubits={self.n_qubits}, "
            f"layers={self.layers}, "
            f"encoding='{self.encoding}', "
            f"entanglement='{self.entanglement}')"
        )
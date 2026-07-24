"""
Base Quantum Circuit for HQFSF.

Provides a reusable wrapper around Qiskit's QuantumCircuit.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from utils.logger import get_logger

logger = get_logger(__name__)


class HQFSFCircuit:
    """
    Base quantum circuit class.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)

        logger.info(
            "Initialized %d-qubit quantum circuit.",
            n_qubits,
        )

    def add_barrier(self):
        """
        Insert a circuit barrier.
        """
        self.circuit.barrier()

        logger.info("Barrier added.")

    def measure_all(self):
        """
        Measure all qubits.
        """
        self.circuit.measure_all()

        logger.info("Measurement added.")

    def reset(self):
        """
        Reset the circuit.
        """
        self.circuit = QuantumCircuit(self.n_qubits)

        logger.info("Circuit reset.")

    def get_circuit(self):
        """
        Return the QuantumCircuit instance.
        """
        return self.circuit

    def draw(self, output: str = "text"):
        """
        Draw the circuit.

        Parameters
        ----------
        output : str
            text | mpl | latex
        """
        return self.circuit.draw(output=output)
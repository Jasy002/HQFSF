"""
Variational Quantum Ansatz for HQFSF.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from utils.logger import get_logger

logger = get_logger(__name__)


class VariationalAnsatz:
    """
    Hardware-efficient variational ansatz.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        entanglement: str = "linear",
    ):
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement.lower()

        self.parameters = ParameterVector(
            "θ",
            length=n_qubits * layers,
        )

    def build(self) -> QuantumCircuit:

        qc = QuantumCircuit(self.n_qubits)

        parameter_index = 0

        for _ in range(self.layers):

            for qubit in range(self.n_qubits):

                qc.ry(
                    self.parameters[parameter_index],
                    qubit,
                )

                parameter_index += 1

            if self.entanglement == "linear":

                for qubit in range(self.n_qubits - 1):
                    qc.cx(qubit, qubit + 1)

            elif self.entanglement == "circular":

                for qubit in range(self.n_qubits - 1):
                    qc.cx(qubit, qubit + 1)

                qc.cx(
                    self.n_qubits - 1,
                    0,
                )

            else:
                raise ValueError(
                    "Entanglement must be "
                    "'linear' or 'circular'."
                )

            qc.barrier()

        logger.info(
            "Built %s ansatz | layers=%d | qubits=%d",
            self.entanglement,
            self.layers,
            self.n_qubits,
        )

        return qc
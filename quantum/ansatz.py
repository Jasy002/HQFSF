"""
Variational Quantum Ansatz for HQFSF.

Supports:
    - Linear Entanglement
    - Circular Entanglement
    - Configurable Layers
    - Parameterized RY Rotations
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from utils.logger import get_logger

logger = get_logger(__name__)


class VariationalAnsatz:
    """
    Hardware-efficient Variational Quantum Ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    layers : int, default=2
        Number of variational layers.

    entanglement : str, default="linear"
        Entanglement topology.

        Supported:
            - linear
            - circular
    """

    SUPPORTED_ENTANGLEMENTS = (
        "linear",
        "circular",
    )

    def __init__(
        self,
        n_qubits: int,
        layers: int = 2,
        entanglement: str = "linear",
    ) -> None:

        if not isinstance(n_qubits, int):
            raise TypeError(
                "n_qubits must be an integer."
            )

        if n_qubits <= 0:
            raise ValueError(
                "Number of qubits must be greater than zero."
            )

        if not isinstance(layers, int):
            raise TypeError(
                "layers must be an integer."
            )

        if layers <= 0:
            raise ValueError(
                "Number of layers must be greater than zero."
            )

        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement.lower()

        if self.entanglement not in self.SUPPORTED_ENTANGLEMENTS:
            raise ValueError(
                f"Unsupported entanglement '{self.entanglement}'. "
                f"Supported methods: {self.SUPPORTED_ENTANGLEMENTS}"
            )

        self.parameters = ParameterVector(
            "θ",
            length=self.n_qubits * self.layers,
        )

        logger.info(
            "Initialized VariationalAnsatz | "
            "Qubits=%d | Layers=%d | Entanglement=%s",
            self.n_qubits,
            self.layers,
            self.entanglement.upper(),
        )

    @property
    def trainable_parameters(self) -> ParameterVector:
        """
        Return the trainable parameters.
        """
        return self.parameters

    def build(self) -> QuantumCircuit:
        """
        Build the parameterized variational quantum circuit.

        Returns
        -------
        QuantumCircuit
            Parameterized quantum circuit.
        """

        qc = QuantumCircuit(self.n_qubits)

        parameter_index = 0

        for layer in range(self.layers):

            # ---------------------------------------
            # Rotation Layer
            # ---------------------------------------

            for qubit in range(self.n_qubits):

                qc.ry(
                    self.parameters[parameter_index],
                    qubit,
                )

                parameter_index += 1

            # ---------------------------------------
            # Entanglement Layer
            # ---------------------------------------

            if self.entanglement == "linear":

                for qubit in range(self.n_qubits - 1):

                    qc.cx(
                        qubit,
                        qubit + 1,
                    )

            elif self.entanglement == "circular":

                for qubit in range(self.n_qubits - 1):

                    qc.cx(
                        qubit,
                        qubit + 1,
                    )

                qc.cx(
                    self.n_qubits - 1,
                    0,
                )

            qc.barrier()

        logger.info(
            "Variational ansatz built successfully."
        )

        return qc

    def draw(self, output: str = "mpl"):
        """
        Draw the quantum circuit.

        Parameters
        ----------
        output : str, default="mpl"

        Returns
        -------
        Circuit drawing.
        """

        circuit = self.build()

        return circuit.draw(output=output)

    def summary(self) -> None:
        """
        Print ansatz configuration.
        """

        print("\n" + "=" * 50)
        print("VARIATIONAL ANSATZ SUMMARY")
        print("=" * 50)

        print(f"Qubits               : {self.n_qubits}")
        print(f"Layers               : {self.layers}")
        print(f"Entanglement         : {self.entanglement}")
        print(f"Trainable Parameters : {len(self.parameters)}")

        print("=" * 50)

    def __repr__(self) -> str:
        return (
            f"VariationalAnsatz("
            f"n_qubits={self.n_qubits}, "
            f"layers={self.layers}, "
            f"entanglement='{self.entanglement}')"
        )
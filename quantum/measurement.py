"""
Quantum Measurement Module for HQFSF.

Responsible for:
    - Executing quantum circuits
    - Measuring all qubits
    - Returning measurement counts
"""

from __future__ import annotations

from qiskit import transpile

from quantum.backend import QuantumBackend

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumMeasurement:
    """
    Quantum Circuit Measurement Manager.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        shots: int = 1024,
    ):

        if shots <= 0:
            raise ValueError(
                "Number of shots must be greater than zero."
            )

        self.backend = backend
        self.shots = shots

        logger.info(
            "Measurement initialized | Shots=%d",
            self.shots,
        )

    def execute(self, circuit):
        """
        Execute a quantum circuit.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        Result
        """

        compiled = transpile(
            circuit,
            self.backend.get_backend(),
        )

        job = self.backend.get_backend().run(
            compiled,
            shots=self.shots,
        )

        result = job.result()

        logger.info(
            "Circuit executed successfully."
        )

        return result

    def counts(self, circuit):
        """
        Execute circuit and return counts.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        dict
        """

        result = self.execute(circuit)

        counts = result.get_counts()

        logger.info(
            "Measurement counts collected."
        )

        return counts

    def probabilities(self, circuit):
        """
        Execute circuit and return probabilities.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        dict
        """

        counts = self.counts(circuit)

        probabilities = {
            state: value / self.shots
            for state, value in counts.items()
        }

        logger.info(
            "Measurement probabilities calculated."
        )

        return probabilities

    def summary(self):
        """
        Print measurement configuration.
        """

        print("\n========== Measurement Summary ==========")

        print(f"Backend : {self.backend.backend_name()}")
        print(f"Shots   : {self.shots}")

        print("=========================================\n")
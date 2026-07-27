"""
Quantum Measurement Module for HQFSF.

Responsible for

    - Executing quantum circuits
    - Measuring all qubits
    - Returning measurement counts
    - Returning probability distributions
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict

from qiskit import QuantumCircuit, transpile
from qiskit.result import Result

from quantum.backend import QuantumBackend

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumMeasurement:
    """
    Quantum Circuit Measurement Manager.

    Parameters
    ----------
    backend : QuantumBackend
        Quantum backend manager.

    shots : int, default=1024
        Number of measurement shots.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        shots: int = 1024,
    ) -> None:

        if not isinstance(backend, QuantumBackend):
            raise TypeError(
                "backend must be an instance of QuantumBackend."
            )

        if shots <= 0:
            raise ValueError(
                "shots must be greater than zero."
            )

        self.backend = backend
        self.shots = shots

        logger.info(
            "QuantumMeasurement initialized | "
            "Backend=%s | Shots=%d",
            self.backend.backend_name(),
            self.shots,
        )

    # ---------------------------------------------------------
    # Circuit Execution
    # ---------------------------------------------------------

    def execute(
        self,
        circuit: QuantumCircuit,
    ) -> Result:
        """
        Execute a quantum circuit.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        Result
            Qiskit execution result.
        """

        start = perf_counter()

        compiled = transpile(
            circuit,
            self.backend.get_backend(),
        )

        job = self.backend.get_backend().run(
            compiled,
            shots=self.shots,
        )

        result = job.result()

        elapsed = perf_counter() - start

        logger.info(
            "Circuit executed successfully "
            "(%.4f seconds).",
            elapsed,
        )

        return result

    # ---------------------------------------------------------
    # Measurement Counts
    # ---------------------------------------------------------

    def counts(
        self,
        circuit: QuantumCircuit,
    ) -> Dict[str, int]:
        """
        Execute a circuit and return measurement counts.
        """

        result = self.execute(circuit)

        counts = result.get_counts()

        logger.info(
            "Measurement counts collected."
        )

        return counts

    # ---------------------------------------------------------
    # Probability Distribution
    # ---------------------------------------------------------

    def probabilities(
        self,
        circuit: QuantumCircuit,
    ) -> Dict[str, float]:
        """
        Execute a circuit and return probability distribution.
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

    # ---------------------------------------------------------
    # Convenience Method
    # ---------------------------------------------------------

    def run_and_measure(
        self,
        circuit: QuantumCircuit,
    ) -> Dict[str, int]:
        """
        Execute a circuit and return measurement counts.

        This is a convenience wrapper around `counts()`.
        """

        return self.counts(circuit)

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print measurement configuration.
        """

        print("\n" + "=" * 55)
        print("QUANTUM MEASUREMENT SUMMARY")
        print("=" * 55)

        print(f"Backend : {self.backend.backend_name()}")
        print(f"Shots   : {self.shots}")

        print("=" * 55)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"QuantumMeasurement("
            f"backend='{self.backend.backend_name()}', "
            f"shots={self.shots})"
        )
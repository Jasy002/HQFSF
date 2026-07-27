"""
Backend configuration for HQFSF.

Supports:
    - Aer Simulator
    - Statevector Simulator
    - Future IBM Quantum Backends
"""

from __future__ import annotations

from typing import Any

from qiskit_aer import AerSimulator

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumBackend:
    """
    Quantum Backend Manager.

    Parameters
    ----------
    backend_type : str, default="aer_simulator"

        Supported backends

        - aer_simulator
        - statevector
    """

    SUPPORTED_BACKENDS = (
        "aer_simulator",
        "statevector",
    )

    def __init__(
        self,
        backend_type: str = "aer_simulator",
    ) -> None:

        self.backend_type = backend_type.lower()

        if self.backend_type not in self.SUPPORTED_BACKENDS:

            raise ValueError(
                f"Unsupported backend '{backend_type}'. "
                f"Supported backends: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = self._initialize_backend()

        logger.info(
            "Quantum backend initialized | %s",
            self.backend.name,
        )

    # -----------------------------------------------------
    # Backend Initialization
    # -----------------------------------------------------

    def _initialize_backend(self) -> Any:
        """
        Initialize the selected backend.

        Returns
        -------
        Backend
            Configured Qiskit backend.
        """

        if self.backend_type == "aer_simulator":

            return AerSimulator()

        if self.backend_type == "statevector":

            return AerSimulator(
                method="statevector"
            )

        raise RuntimeError(
            "Backend initialization failed."
        )

    # -----------------------------------------------------
    # Backend Access
    # -----------------------------------------------------

    def get_backend(self):
        """
        Return backend instance.
        """

        return self.backend

    def backend_name(self) -> str:
        """
        Return backend name.
        """

        return self.backend.name

    def configuration(self):
        """
        Return backend configuration.
        """

        return self.backend.configuration()

    def is_simulator(self) -> bool:
        """
        Check whether backend is a simulator.
        """

        return True

    # -----------------------------------------------------
    # Information
    # -----------------------------------------------------

    def summary(self) -> None:
        """
        Print backend summary.
        """

        print("\n" + "=" * 55)
        print("QUANTUM BACKEND SUMMARY")
        print("=" * 55)

        print(f"Backend Type : {self.backend_type}")
        print(f"Backend Name : {self.backend.name}")
        print(f"Simulator    : {self.is_simulator()}")

        print("=" * 55)

    # -----------------------------------------------------
    # Representation
    # -----------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"QuantumBackend("
            f"backend_type='{self.backend_type}')"
        )
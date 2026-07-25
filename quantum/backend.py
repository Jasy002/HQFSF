"""
Backend configuration for HQFSF.

Supports:
    - Aer Simulator
    - Statevector Simulator
    - Future IBM Quantum Backends
"""

from __future__ import annotations

from qiskit_aer import AerSimulator

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumBackend:
    """
    Quantum Backend Manager.
    """

    SUPPORTED_BACKENDS = (
        "aer_simulator",
        "statevector",
    )

    def __init__(
        self,
        backend_type: str = "aer_simulator",
    ):

        self.backend_type = backend_type.lower()

        if self.backend_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend_type}'. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = self._initialize_backend()

        logger.info(
            "Backend initialized: %s",
            self.backend.name,
        )

    def _initialize_backend(self):
        """
        Initialize the selected backend.
        """

        if self.backend_type == "aer_simulator":

            return AerSimulator()

        elif self.backend_type == "statevector":

            return AerSimulator(method="statevector")

    def get_backend(self):
        """
        Return backend instance.
        """
        return self.backend

    def backend_name(self):
        """
        Return backend name.
        """
        return self.backend.name

    def configuration(self):
        """
        Return backend configuration.
        """
        return self.backend.configuration()

    def summary(self):
        """
        Print backend information.
        """

        print("\n========== Backend Summary ==========")

        print(f"Backend Type : {self.backend_type}")
        print(f"Backend Name : {self.backend.name}")

        print("=====================================\n")
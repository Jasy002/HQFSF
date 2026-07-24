"""
Backend configuration for HQFSF.

Supports:
- Aer Simulator
- Future IBM Quantum backends
"""

from __future__ import annotations

from qiskit_aer import AerSimulator

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumBackend:
    """
    Backend manager for HQFSF.
    """

    def __init__(self):
        self.backend = AerSimulator()

        logger.info(
            "Initialized AerSimulator backend."
        )

    def get_backend(self):
        """
        Return configured backend.
        """
        return self.backend

    def backend_name(self):
        """
        Return backend name.
        """
        return self.backend.name
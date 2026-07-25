"""
Quantum Optimizer Manager for HQFSF.

Supports:
    - COBYLA
    - SPSA
    - SLSQP
"""

from __future__ import annotations

from qiskit_algorithms.optimizers import (
    COBYLA,
    SPSA,
    SLSQP,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumOptimizer:
    """
    Factory class for Qiskit optimizers.
    """

    SUPPORTED_OPTIMIZERS = {
        "cobyla": COBYLA,
        "spsa": SPSA,
        "slsqp": SLSQP,
    }

    def __init__(
        self,
        optimizer: str = "cobyla",
        maxiter: int = 100,
    ):

        self.optimizer = optimizer.lower()

        if self.optimizer not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer '{optimizer}'. "
                f"Supported optimizers: "
                f"{list(self.SUPPORTED_OPTIMIZERS.keys())}"
            )

        if maxiter <= 0:
            raise ValueError(
                "maxiter must be greater than zero."
            )

        self.maxiter = maxiter

        logger.info(
            "QuantumOptimizer initialized | "
            "Optimizer=%s | MaxIter=%d",
            self.optimizer.upper(),
            self.maxiter,
        )

    def get_optimizer(self):
        """
        Return configured optimizer.
        """

        optimizer = self.SUPPORTED_OPTIMIZERS[
            self.optimizer
        ](
            maxiter=self.maxiter
        )

        logger.info(
            "%s optimizer created.",
            self.optimizer.upper(),
        )

        return optimizer

    @classmethod
    def available_optimizers(cls):
        """
        Return supported optimizers.
        """

        return list(cls.SUPPORTED_OPTIMIZERS.keys())

    def summary(self):
        """
        Print optimizer configuration.
        """

        print("\n========== Optimizer Summary ==========")

        print(f"Optimizer : {self.optimizer.upper()}")
        print(f"Max Iter  : {self.maxiter}")

        print("=======================================\n")
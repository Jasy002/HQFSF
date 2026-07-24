"""
Quantum optimizer manager for HQFSF.
"""

from __future__ import annotations

from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantumOptimizer:
    """
    Factory class for Qiskit optimizers.
    """

    SUPPORTED = {
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
        self.maxiter = maxiter

    def get_optimizer(self):
        """
        Return configured optimizer.
        """

        if self.optimizer not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer}'. "
                f"Available: {list(self.SUPPORTED.keys())}"
            )

        opt = self.SUPPORTED[self.optimizer](
            maxiter=self.maxiter
        )

        logger.info(
            "Optimizer initialized: %s (maxiter=%d)",
            self.optimizer.upper(),
            self.maxiter,
        )

        return opt
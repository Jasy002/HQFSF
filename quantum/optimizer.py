"""
Quantum Optimizer Manager for HQFSF.

Supports:
    - COBYLA
    - SPSA
    - SLSQP

Provides a unified interface for creating Qiskit optimizers
used in variational quantum algorithms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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

    Parameters
    ----------
    optimizer : str, default="cobyla"
        Optimizer name.

        Supported:

        - cobyla
        - spsa
        - slsqp

    maxiter : int, default=100
        Maximum optimization iterations.

    tol : float | None, default=None
        Optimization tolerance (if supported).
    """

    SUPPORTED_OPTIMIZERS: Dict[str, Any] = {
        "cobyla": COBYLA,
        "spsa": SPSA,
        "slsqp": SLSQP,
    }

    def __init__(
        self,
        optimizer: str = "cobyla",
        maxiter: int = 100,
        tol: Optional[float] = None,
    ) -> None:

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

        if tol is not None and tol <= 0:

            raise ValueError(
                "tol must be greater than zero."
            )

        self.maxiter = maxiter
        self.tol = tol

        logger.info(
            "QuantumOptimizer initialized | "
            "Optimizer=%s | MaxIter=%d",
            self.optimizer.upper(),
            self.maxiter,
        )

    # ---------------------------------------------------------
    # Optimizer Factory
    # ---------------------------------------------------------

    def get_optimizer(self):
        """
        Create and return the configured optimizer.

        Returns
        -------
        Optimizer
            Qiskit optimizer instance.
        """

        optimizer_class = self.SUPPORTED_OPTIMIZERS[
            self.optimizer
        ]

        kwargs = {
            "maxiter": self.maxiter,
        }

        if self.tol is not None:

            kwargs["tol"] = self.tol

        optimizer = optimizer_class(**kwargs)

        logger.info(
            "%s optimizer created.",
            self.optimizer.upper(),
        )

        return optimizer

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @classmethod
    def available_optimizers(cls) -> List[str]:
        """
        Return supported optimizers.
        """

        return list(
            cls.SUPPORTED_OPTIMIZERS.keys()
        )

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def summary(self) -> None:
        """
        Print optimizer configuration.
        """

        print("\n" + "=" * 55)
        print("QUANTUM OPTIMIZER SUMMARY")
        print("=" * 55)

        print(f"Optimizer : {self.optimizer.upper()}")
        print(f"Max Iter  : {self.maxiter}")

        if self.tol is not None:
            print(f"Tolerance : {self.tol}")

        print("\nAvailable Optimizers:")

        for name in self.available_optimizers():
            print(f"  • {name.upper()}")

        print("=" * 55)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"QuantumOptimizer("
            f"optimizer='{self.optimizer}', "
            f"maxiter={self.maxiter}, "
            f"tol={self.tol})"
        )
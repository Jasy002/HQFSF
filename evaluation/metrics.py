"""
Base Metrics Module.

Defines the abstract interface for all
evaluation metric classes used in HQFSF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class Metrics(ABC):
    """
    Abstract base class for evaluation metrics.

    Every evaluation class should inherit
    from this class.
    """

    def __init__(self):
        logger.info(
            f"{self.__class__.__name__} initialized."
        )

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing computed metrics.
        """
        raise NotImplementedError

    def summary(self) -> None:
        """
        Display metric information.
        """

        print("\n" + "=" * 70)
        print(f" {self.__class__.__name__} ")
        print("=" * 70)
        print("Base Evaluation Class")
        print("Override evaluate() in subclasses.")
        print("=" * 70)
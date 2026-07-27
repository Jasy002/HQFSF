"""
Expectation Value Module for HQFSF.

Responsible for

    - Computing Pauli-Z expectation values
    - Computing probability distributions
    - Computing expectation vectors
    - Supporting multiple quantum circuits
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ExpectationCalculator:
    """
    Expectation Value Calculator.

    Computes expectation values from measurement counts
    obtained after executing quantum circuits.
    """

    def __init__(self) -> None:

        logger.info(
            "ExpectationCalculator initialized."
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        counts: Dict[str, int],
        shots: int,
    ) -> None:
        """
        Validate measurement counts.
        """

        if shots <= 0:
            raise ValueError(
                "shots must be greater than zero."
            )

        if not counts:
            raise ValueError(
                "counts dictionary is empty."
            )

    # ---------------------------------------------------------
    # Expectation Value
    # ---------------------------------------------------------

    def expectation_z(
        self,
        counts: Dict[str, int],
        shots: int,
    ) -> float:
        """
        Compute the expectation value of the Pauli-Z operator.

        Parameters
        ----------
        counts : dict
            Measurement counts.

        shots : int
            Number of executed shots.

        Returns
        -------
        float
            Pauli-Z expectation value.
        """

        self._validate_inputs(
            counts,
            shots,
        )

        expectation = 0.0

        for bitstring, frequency in counts.items():

            parity = (-1) ** bitstring.count("1")

            expectation += parity * frequency

        expectation /= shots

        logger.info(
            "Expectation value computed."
        )

        return expectation

    # ---------------------------------------------------------
    # Single-Qubit Expectation
    # ---------------------------------------------------------

    def expectation_qubit(
        self,
        counts: Dict[str, int],
        shots: int,
        qubit: int,
    ) -> float:
        """
        Compute expectation value of a single qubit.
        """

        self._validate_inputs(
            counts,
            shots,
        )

        expectation = 0.0

        for bitstring, frequency in counts.items():

            bit = bitstring[::-1][qubit]

            value = 1 if bit == "0" else -1

            expectation += value * frequency

        return expectation / shots

    # ---------------------------------------------------------
    # Probability Distribution
    # ---------------------------------------------------------

    def probability_distribution(
        self,
        counts: Dict[str, int],
        shots: int,
    ) -> Dict[str, float]:
        """
        Convert counts into probabilities.
        """

        self._validate_inputs(
            counts,
            shots,
        )

        probabilities = {
            state: value / shots
            for state, value in counts.items()
        }

        logger.info(
            "Probability distribution computed."
        )

        return probabilities

    # ---------------------------------------------------------
    # Expectation Vector
    # ---------------------------------------------------------

    def expectation_vector(
        self,
        counts_list: List[Dict[str, int]],
        shots: int,
    ) -> np.ndarray:
        """
        Compute expectation values for multiple circuits.
        """

        values = [

            self.expectation_z(
                counts,
                shots,
            )

            for counts in counts_list

        ]

        logger.info(
            "Expectation vector computed."
        )

        return np.asarray(
            values,
            dtype=float,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> None:

        print("\n" + "=" * 55)
        print("EXPECTATION CALCULATOR SUMMARY")
        print("=" * 55)

        print("Observable : Pauli-Z")
        print("Outputs    :")
        print("   • Expectation value")
        print("   • Expectation vector")
        print("   • Probability distribution")

        print("=" * 55)

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return "ExpectationCalculator()"
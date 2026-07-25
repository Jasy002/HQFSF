"""
Expectation Value Module for HQFSF.

Responsible for:
    - Computing expectation values
    - Computing probabilities
    - Returning expectation vectors
"""

from __future__ import annotations

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ExpectationCalculator:
    """
    Expectation Value Calculator.
    """

    def __init__(self):
        logger.info(
            "Expectation Calculator initialized."
        )

    def expectation_z(
        self,
        counts: dict,
        shots: int,
    ) -> float:
        """
        Compute the expectation value of the Pauli-Z operator.

        Parameters
        ----------
        counts : dict
            Measurement counts.

        shots : int
            Number of shots.

        Returns
        -------
        float
        """

        expectation = 0.0

        for bitstring, frequency in counts.items():

            parity = 1

            for bit in bitstring:

                if bit == "1":
                    parity *= -1

            expectation += parity * frequency

        expectation /= shots

        logger.info(
            "Expectation value computed."
        )

        return expectation

    def probability_distribution(
        self,
        counts: dict,
        shots: int,
    ) -> dict:
        """
        Convert counts into probabilities.
        """

        probabilities = {
            state: value / shots
            for state, value in counts.items()
        }

        logger.info(
            "Probability distribution computed."
        )

        return probabilities

    def expectation_vector(
        self,
        counts_list: list,
        shots: int,
    ) -> np.ndarray:
        """
        Compute expectation values for multiple circuits.

        Parameters
        ----------
        counts_list : list
            List of count dictionaries.

        shots : int

        Returns
        -------
        numpy.ndarray
        """

        values = []

        for counts in counts_list:

            values.append(
                self.expectation_z(
                    counts,
                    shots,
                )
            )

        logger.info(
            "Expectation vector computed."
        )

        return np.array(values)

    def summary(self):
        """
        Display module information.
        """

        print("\n========== Expectation Calculator ==========")

        print("Operator : Pauli-Z")
        print("Output   : Expectation Value(s)")

        print("============================================\n")
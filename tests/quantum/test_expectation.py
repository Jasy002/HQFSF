"""
Unit tests for ExpectationCalculator.
"""

import numpy as np
import pytest

from quantum.expectation import ExpectationCalculator


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def calculator():
    return ExpectationCalculator()


@pytest.fixture
def sample_counts():
    return {
        "00": 520,
        "01": 240,
        "10": 180,
        "11": 84,
    }


@pytest.fixture
def shots():
    return 1024


# ---------------------------------------------------------------------
# Initialization Test
# ---------------------------------------------------------------------

def test_initialization(calculator):
    assert isinstance(
        calculator,
        ExpectationCalculator,
    )


# ---------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------

def test_invalid_shots(
    calculator,
    sample_counts,
):
    with pytest.raises(ValueError):
        calculator.expectation_z(
            sample_counts,
            0,
        )


def test_empty_counts(
    calculator,
    shots,
):
    with pytest.raises(ValueError):
        calculator.expectation_z(
            {},
            shots,
        )


# ---------------------------------------------------------------------
# Expectation Z Tests
# ---------------------------------------------------------------------

def test_expectation_z_returns_float(
    calculator,
    sample_counts,
    shots,
):
    value = calculator.expectation_z(
        sample_counts,
        shots,
    )

    assert isinstance(value, float)


def test_expectation_z_range(
    calculator,
    sample_counts,
    shots,
):
    value = calculator.expectation_z(
        sample_counts,
        shots,
    )

    assert -1.0 <= value <= 1.0


# ---------------------------------------------------------------------
# Single Qubit Expectation
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "qubit",
    [0, 1],
)
def test_expectation_qubit(
    calculator,
    sample_counts,
    shots,
    qubit,
):
    value = calculator.expectation_qubit(
        sample_counts,
        shots,
        qubit,
    )

    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


# ---------------------------------------------------------------------
# Probability Distribution
# ---------------------------------------------------------------------

def test_probability_distribution(
    calculator,
    sample_counts,
    shots,
):
    probabilities = calculator.probability_distribution(
        sample_counts,
        shots,
    )

    assert isinstance(probabilities, dict)


def test_probability_sum(
    calculator,
    sample_counts,
    shots,
):
    probabilities = calculator.probability_distribution(
        sample_counts,
        shots,
    )

    assert pytest.approx(
        sum(probabilities.values()),
        rel=1e-6,
    ) == 1.0


def test_probability_keys(
    calculator,
    sample_counts,
    shots,
):
    probabilities = calculator.probability_distribution(
        sample_counts,
        shots,
    )

    assert set(probabilities.keys()) == set(
        sample_counts.keys()
    )


# ---------------------------------------------------------------------
# Expectation Vector
# ---------------------------------------------------------------------

def test_expectation_vector(
    calculator,
    sample_counts,
    shots,
):
    counts_list = [
        sample_counts,
        sample_counts,
        sample_counts,
    ]

    vector = calculator.expectation_vector(
        counts_list,
        shots,
    )

    assert isinstance(vector, np.ndarray)
    assert len(vector) == 3


def test_expectation_vector_dtype(
    calculator,
    sample_counts,
    shots,
):
    vector = calculator.expectation_vector(
        [
            sample_counts,
            sample_counts,
        ],
        shots,
    )

    assert vector.dtype == float


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(
    calculator,
    capsys,
):
    calculator.summary()

    captured = capsys.readouterr()

    assert "EXPECTATION CALCULATOR SUMMARY" in captured.out
    assert "Pauli-Z" in captured.out


# ---------------------------------------------------------------------
# Representation Test
# ---------------------------------------------------------------------

def test_repr(calculator):
    representation = repr(calculator)

    assert representation == "ExpectationCalculator()"


# ---------------------------------------------------------------------
# Consistency Tests
# ---------------------------------------------------------------------

def test_same_input_same_output(
    calculator,
    sample_counts,
    shots,
):
    value1 = calculator.expectation_z(
        sample_counts,
        shots,
    )

    value2 = calculator.expectation_z(
        sample_counts,
        shots,
    )

    assert value1 == value2


def test_expectation_vector_values(
    calculator,
    sample_counts,
    shots,
):
    vector = calculator.expectation_vector(
        [
            sample_counts,
            sample_counts,
            sample_counts,
        ],
        shots,
    )

    assert np.all(
        vector == vector[0]
    )
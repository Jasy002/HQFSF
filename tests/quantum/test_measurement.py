"""
Unit tests for QuantumMeasurement.
"""

import numpy as np
import pytest
from qiskit.result import Result

from quantum.backend import QuantumBackend
from quantum.circuit import HQFSFCircuit
from quantum.measurement import QuantumMeasurement


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def backend():
    return QuantumBackend()


@pytest.fixture
def measurement(backend):
    return QuantumMeasurement(
        backend=backend,
        shots=1024,
    )


@pytest.fixture
def sample_circuit():
    features = np.array([
        0.2,
        0.5,
        0.9,
        0.1,
    ])

    builder = HQFSFCircuit(
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement="linear",
    )

    circuit = builder.build(features)

    return builder.measure(circuit)


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_initialization(measurement):
    assert measurement.shots == 1024
    assert isinstance(
        measurement.backend,
        QuantumBackend,
    )


def test_invalid_backend():
    with pytest.raises(TypeError):
        QuantumMeasurement(
            backend="backend",
            shots=1024,
        )


def test_invalid_shots():
    backend = QuantumBackend()

    with pytest.raises(ValueError):
        QuantumMeasurement(
            backend=backend,
            shots=0,
        )


# ---------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------

def test_execute_returns_result(
    measurement,
    sample_circuit,
):
    result = measurement.execute(sample_circuit)

    assert isinstance(result, Result)


def test_counts_returns_dictionary(
    measurement,
    sample_circuit,
):
    counts = measurement.counts(sample_circuit)

    assert isinstance(counts, dict)
    assert len(counts) > 0


def test_probabilities_returns_dictionary(
    measurement,
    sample_circuit,
):
    probabilities = measurement.probabilities(
        sample_circuit
    )

    assert isinstance(probabilities, dict)
    assert len(probabilities) > 0


def test_probability_sum(
    measurement,
    sample_circuit,
):
    probabilities = measurement.probabilities(
        sample_circuit
    )

    total = sum(probabilities.values())

    assert pytest.approx(
        total,
        rel=1e-6,
    ) == 1.0


def test_run_and_measure(
    measurement,
    sample_circuit,
):
    counts = measurement.run_and_measure(
        sample_circuit
    )

    assert isinstance(counts, dict)


def test_counts_total_shots(
    measurement,
    sample_circuit,
):
    counts = measurement.counts(sample_circuit)

    assert sum(counts.values()) == measurement.shots


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(
    measurement,
    capsys,
):
    measurement.summary()

    captured = capsys.readouterr()

    assert "QUANTUM MEASUREMENT SUMMARY" in captured.out
    assert "Backend" in captured.out
    assert "Shots" in captured.out


# ---------------------------------------------------------------------
# Representation Test
# ---------------------------------------------------------------------

def test_repr(measurement):
    representation = repr(measurement)

    assert "QuantumMeasurement" in representation
    assert "shots=1024" in representation


# ---------------------------------------------------------------------
# Consistency Tests
# ---------------------------------------------------------------------

def test_multiple_runs_return_dict(
    measurement,
    sample_circuit,
):
    counts1 = measurement.counts(sample_circuit)
    counts2 = measurement.counts(sample_circuit)

    assert isinstance(counts1, dict)
    assert isinstance(counts2, dict)


def test_probability_keys_match_counts(
    measurement,
    sample_circuit,
):
    counts = measurement.counts(sample_circuit)
    probabilities = measurement.probabilities(
        sample_circuit
    )

    assert set(counts.keys()) == set(
        probabilities.keys()
    )
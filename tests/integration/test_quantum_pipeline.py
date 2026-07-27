"""
Integration tests for the complete Quantum Feature Selection Pipeline.
"""

import numpy as np
import pytest

from quantum.circuit import HQFSFCircuit
from quantum.backend import QuantumBackend
from quantum.measurement import QuantumMeasurement
from quantum.expectation import ExpectationCalculator
from quantum.feature_selector import QuantumFeatureSelector


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_features():
    return np.array([
        0.42,
        0.76,
        0.15,
        0.91,
    ])


@pytest.fixture
def circuit():
    return HQFSFCircuit(
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement="linear",
    )


@pytest.fixture
def backend():
    return QuantumBackend(
        backend_type="aer_simulator",
    )


@pytest.fixture
def measurement(backend):
    return QuantumMeasurement(
        backend=backend,
        shots=1024,
    )


@pytest.fixture
def calculator():
    return ExpectationCalculator()


@pytest.fixture
def selector():
    return QuantumFeatureSelector(
        strategy="top_k",
        top_k=2,
    )


# ---------------------------------------------------------------------
# Circuit → Measurement
# ---------------------------------------------------------------------

def test_circuit_execution_pipeline(
    circuit,
    measurement,
    sample_features,
):
    qc = circuit.build(sample_features)
    qc = circuit.measure(qc)

    counts = measurement.counts(qc)

    assert isinstance(counts, dict)
    assert sum(counts.values()) == 1024


# ---------------------------------------------------------------------
# Measurement → Expectation
# ---------------------------------------------------------------------

def test_expectation_pipeline(
    circuit,
    measurement,
    calculator,
    sample_features,
):
    qc = circuit.build(sample_features)
    qc = circuit.measure(qc)

    counts = measurement.counts(qc)

    expectation = calculator.expectation_z(
        counts,
        1024,
    )

    assert isinstance(expectation, float)
    assert -1.0 <= expectation <= 1.0


# ---------------------------------------------------------------------
# Probability Distribution
# ---------------------------------------------------------------------

def test_probability_pipeline(
    circuit,
    measurement,
    calculator,
    sample_features,
):
    qc = circuit.build(sample_features)
    qc = circuit.measure(qc)

    counts = measurement.counts(qc)

    probabilities = calculator.probability_distribution(
        counts,
        1024,
    )

    assert isinstance(probabilities, dict)

    assert pytest.approx(
        sum(probabilities.values()),
        rel=1e-6,
    ) == 1.0


# ---------------------------------------------------------------------
# Expectation Vector
# ---------------------------------------------------------------------

def test_expectation_vector_pipeline(
    calculator,
):
    counts = {
        "00": 550,
        "01": 200,
        "10": 180,
        "11": 94,
    }

    vector = calculator.expectation_vector(
        [
            counts,
            counts,
            counts,
            counts,
        ],
        1024,
    )

    assert isinstance(vector, np.ndarray)
    assert len(vector) == 4


# ---------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------

def test_feature_selection_pipeline(
    selector,
):
    scores = np.array([
        0.65,
        0.18,
        0.92,
        0.51,
    ])

    selected = selector.select(scores)

    assert len(selected) == 2

    assert set(selected.tolist()) == {
        0,
        2,
    }


# ---------------------------------------------------------------------
# Complete Quantum Workflow
# ---------------------------------------------------------------------

def test_complete_quantum_pipeline(
    circuit,
    measurement,
    calculator,
    selector,
    sample_features,
):
    qc = circuit.build(sample_features)
    qc = circuit.measure(qc)

    counts = measurement.counts(qc)

    expectation = calculator.expectation_z(
        counts,
        1024,
    )

    scores = np.array([
        expectation,
        expectation * 0.8,
        expectation * 0.6,
        expectation * 0.4,
    ])

    selected = selector.select(scores)

    assert isinstance(selected, np.ndarray)
    assert len(selected) == 2


# ---------------------------------------------------------------------
# Pipeline Consistency
# ---------------------------------------------------------------------

def test_pipeline_consistency(
    circuit,
    measurement,
    calculator,
    sample_features,
):
    qc = circuit.build(sample_features)
    qc = circuit.measure(qc)

    counts1 = measurement.counts(qc)
    counts2 = measurement.counts(qc)

    exp1 = calculator.expectation_z(
        counts1,
        1024,
    )

    exp2 = calculator.expectation_z(
        counts2,
        1024,
    )

    assert abs(exp1 - exp2) < 0.1


# ---------------------------------------------------------------------
# Backend Verification
# ---------------------------------------------------------------------

def test_backend_pipeline(
    backend,
):
    simulator = backend.get_backend()

    assert simulator is not None
    assert backend.is_simulator() is True


# ---------------------------------------------------------------------
# Output Types
# ---------------------------------------------------------------------

def test_pipeline_output_types(
    circuit,
    measurement,
    calculator,
    sample_features,
):
    qc = circuit.build(sample_features)
    qc = circuit.measure(qc)

    counts = measurement.counts(qc)

    expectation = calculator.expectation_z(
        counts,
        1024,
    )

    assert isinstance(counts, dict)
    assert isinstance(expectation, float)
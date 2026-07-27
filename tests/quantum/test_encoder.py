"""
Unit tests for QuantumEncoder.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum.encoder import QuantumEncoder


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def encoder():
    """Create a QuantumEncoder instance."""
    return QuantumEncoder(n_qubits=4)


@pytest.fixture
def sample_features():
    """Valid feature vector."""
    return np.array([0.2, 0.5, 0.9, 0.1])


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_encoder_initialization():
    encoder = QuantumEncoder(n_qubits=4)

    assert encoder.n_qubits == 4
    assert encoder.SUPPORTED_METHODS == ("ry", "rx", "rz")


def test_invalid_qubits_type():
    with pytest.raises(TypeError):
        QuantumEncoder(n_qubits="4")


def test_invalid_qubits_value():
    with pytest.raises(ValueError):
        QuantumEncoder(n_qubits=0)


# ---------------------------------------------------------------------
# Encoding Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ry", "rx", "rz"])
def test_supported_encodings(encoder, sample_features, method):
    qc = encoder.encode(
        sample_features,
        encoding_method=method
    )

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 4


def test_invalid_encoding_method(encoder, sample_features):
    with pytest.raises(ValueError):
        encoder.encode(
            sample_features,
            encoding_method="xyz"
        )


# ---------------------------------------------------------------------
# Feature Validation Tests
# ---------------------------------------------------------------------

def test_insufficient_features(encoder):
    features = np.array([0.1, 0.2])

    with pytest.raises(ValueError):
        encoder.encode(features)


def test_two_dimensional_array(encoder):
    features = np.array([
        [0.1, 0.2],
        [0.3, 0.4]
    ])

    with pytest.raises(ValueError):
        encoder.encode(features)


def test_list_input_conversion(encoder):
    features = [0.2, 0.5, 0.9, 0.1]

    qc = encoder.encode(features)

    assert isinstance(qc, QuantumCircuit)


def test_extra_features_allowed(encoder):
    features = np.array([
        0.2,
        0.5,
        0.9,
        0.1,
        0.7,
        0.6
    ])

    qc = encoder.encode(features)

    assert qc.num_qubits == 4


# ---------------------------------------------------------------------
# Circuit Tests
# ---------------------------------------------------------------------

def test_circuit_type(encoder, sample_features):
    qc = encoder.encode(sample_features)

    assert isinstance(qc, QuantumCircuit)


def test_number_of_qubits(encoder, sample_features):
    qc = encoder.encode(sample_features)

    assert qc.num_qubits == encoder.n_qubits


# ---------------------------------------------------------------------
# Representation Tests
# ---------------------------------------------------------------------

def test_repr():
    encoder = QuantumEncoder(4)

    assert "QuantumEncoder" in repr(encoder)
    assert "n_qubits=4" in repr(encoder)


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(capsys):
    encoder = QuantumEncoder(4)

    encoder.summary()

    captured = capsys.readouterr()

    assert "QUANTUM ENCODER SUMMARY" in captured.out
    assert "Supported Encodings" in captured.out
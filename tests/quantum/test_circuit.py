"""
Unit tests for HQFSFCircuit.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum.circuit import HQFSFCircuit


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_features():
    return np.array([
        0.2,
        0.5,
        0.9,
        0.1,
    ])


@pytest.fixture
def circuit_builder():
    return HQFSFCircuit(
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement="linear",
    )


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_initialization(circuit_builder):
    assert circuit_builder.n_qubits == 4
    assert circuit_builder.layers == 2
    assert circuit_builder.encoding == "ry"
    assert circuit_builder.entanglement == "linear"


def test_invalid_qubits():
    with pytest.raises(ValueError):
        HQFSFCircuit(
            n_qubits=0,
            layers=2,
        )


def test_invalid_layers():
    with pytest.raises(ValueError):
        HQFSFCircuit(
            n_qubits=4,
            layers=0,
        )


# ---------------------------------------------------------------------
# Build Tests
# ---------------------------------------------------------------------

def test_build_returns_quantum_circuit(
    circuit_builder,
    sample_features,
):
    qc = circuit_builder.build(sample_features)

    assert isinstance(qc, QuantumCircuit)


def test_build_qubit_count(
    circuit_builder,
    sample_features,
):
    qc = circuit_builder.build(sample_features)

    assert qc.num_qubits == 4


def test_build_with_wrong_feature_length(circuit_builder):
    features = np.array([
        0.1,
        0.2,
    ])

    with pytest.raises(ValueError):
        circuit_builder.build(features)


@pytest.mark.parametrize(
    "encoding",
    [
        "ry",
        "rx",
        "rz",
    ],
)
def test_supported_encodings(
    encoding,
    sample_features,
):
    builder = HQFSFCircuit(
        n_qubits=4,
        layers=2,
        encoding=encoding,
        entanglement="linear",
    )

    qc = builder.build(sample_features)

    assert isinstance(qc, QuantumCircuit)


@pytest.mark.parametrize(
    "entanglement",
    [
        "linear",
        "circular",
    ],
)
def test_supported_entanglements(
    entanglement,
    sample_features,
):
    builder = HQFSFCircuit(
        n_qubits=4,
        layers=2,
        encoding="ry",
        entanglement=entanglement,
    )

    qc = builder.build(sample_features)

    assert isinstance(qc, QuantumCircuit)


# ---------------------------------------------------------------------
# Measurement Tests
# ---------------------------------------------------------------------

def test_measure_adds_classical_register(
    circuit_builder,
    sample_features,
):
    qc = circuit_builder.build(sample_features)

    measured = circuit_builder.measure(qc)

    assert measured.num_clbits == qc.num_qubits


def test_measure_returns_new_circuit(
    circuit_builder,
    sample_features,
):
    qc = circuit_builder.build(sample_features)

    measured = circuit_builder.measure(qc)

    assert measured is not qc


# ---------------------------------------------------------------------
# Draw Tests
# ---------------------------------------------------------------------

def test_draw_text(
    circuit_builder,
    sample_features,
):
    qc = circuit_builder.build(sample_features)

    drawing = circuit_builder.draw(
        qc,
        output="text",
    )

    assert drawing is not None


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(
    circuit_builder,
    capsys,
):
    circuit_builder.summary()

    captured = capsys.readouterr()

    assert "HQFSF CIRCUIT SUMMARY" in captured.out
    assert "Qubits" in captured.out
    assert "Layers" in captured.out
    assert "Encoding" in captured.out


# ---------------------------------------------------------------------
# Representation Test
# ---------------------------------------------------------------------

def test_repr(circuit_builder):
    representation = repr(circuit_builder)

    assert "HQFSFCircuit" in representation
    assert "n_qubits=4" in representation
    assert "layers=2" in representation
    assert "encoding='ry'" in representation
    assert "entanglement='linear'" in representation
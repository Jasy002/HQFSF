"""
Unit tests for VariationalAnsatz.
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from quantum.ansatz import VariationalAnsatz


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def linear_ansatz():
    return VariationalAnsatz(
        n_qubits=4,
        layers=2,
        entanglement="linear",
    )


@pytest.fixture
def circular_ansatz():
    return VariationalAnsatz(
        n_qubits=4,
        layers=2,
        entanglement="circular",
    )


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_linear_initialization(linear_ansatz):
    assert linear_ansatz.n_qubits == 4
    assert linear_ansatz.layers == 2
    assert linear_ansatz.entanglement == "linear"


def test_circular_initialization(circular_ansatz):
    assert circular_ansatz.n_qubits == 4
    assert circular_ansatz.layers == 2
    assert circular_ansatz.entanglement == "circular"


def test_invalid_qubits_type():
    with pytest.raises(TypeError):
        VariationalAnsatz(
            n_qubits="4",
            layers=2,
        )


def test_invalid_qubits_value():
    with pytest.raises(ValueError):
        VariationalAnsatz(
            n_qubits=0,
            layers=2,
        )


def test_invalid_layers_type():
    with pytest.raises(TypeError):
        VariationalAnsatz(
            n_qubits=4,
            layers="2",
        )


def test_invalid_layers_value():
    with pytest.raises(ValueError):
        VariationalAnsatz(
            n_qubits=4,
            layers=0,
        )


def test_invalid_entanglement():
    with pytest.raises(ValueError):
        VariationalAnsatz(
            n_qubits=4,
            layers=2,
            entanglement="invalid",
        )


# ---------------------------------------------------------------------
# Parameter Tests
# ---------------------------------------------------------------------

def test_trainable_parameters(linear_ansatz):
    params = linear_ansatz.trainable_parameters

    assert isinstance(params, ParameterVector)
    assert len(params) == 8


# ---------------------------------------------------------------------
# Circuit Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("entanglement", ["linear", "circular"])
def test_build_circuit(entanglement):
    ansatz = VariationalAnsatz(
        n_qubits=4,
        layers=2,
        entanglement=entanglement,
    )

    circuit = ansatz.build()

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4


def test_draw_text(linear_ansatz):
    drawing = linear_ansatz.draw(output="text")

    assert drawing is not None


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(linear_ansatz, capsys):
    linear_ansatz.summary()

    captured = capsys.readouterr()

    assert "VARIATIONAL ANSATZ SUMMARY" in captured.out
    assert "Qubits" in captured.out
    assert "Layers" in captured.out


# ---------------------------------------------------------------------
# Representation Test
# ---------------------------------------------------------------------

def test_repr(linear_ansatz):
    representation = repr(linear_ansatz)

    assert "VariationalAnsatz" in representation
    assert "n_qubits=4" in representation
    assert "layers=2" in representation
    assert "linear" in representation
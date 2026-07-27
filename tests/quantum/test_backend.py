"""
Unit tests for QuantumBackend.
"""

import pytest

from quantum.backend import QuantumBackend


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def aer_backend():
    return QuantumBackend(
        backend_type="aer_simulator"
    )


@pytest.fixture
def statevector_backend():
    return QuantumBackend(
        backend_type="statevector"
    )


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_aer_backend_initialization(aer_backend):
    assert aer_backend.backend_type == "aer_simulator"


def test_statevector_backend_initialization(statevector_backend):
    assert statevector_backend.backend_type == "statevector"


def test_invalid_backend():
    with pytest.raises(ValueError):
        QuantumBackend(
            backend_type="invalid_backend"
        )


# ---------------------------------------------------------------------
# Backend Tests
# ---------------------------------------------------------------------

def test_get_backend(aer_backend):
    backend = aer_backend.get_backend()

    assert backend is not None


def test_backend_name_returns_string(aer_backend):
    name = aer_backend.backend_name()

    assert isinstance(name, str)
    assert len(name) > 0


def test_backend_configuration(aer_backend):
    config = aer_backend.configuration()

    assert config is not None


def test_backend_is_simulator(aer_backend):
    assert aer_backend.is_simulator() is True


# ---------------------------------------------------------------------
# Backend Types
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "backend_type",
    [
        "aer_simulator",
        "statevector",
    ],
)
def test_supported_backends(backend_type):
    backend = QuantumBackend(
        backend_type=backend_type
    )

    assert backend.backend_type == backend_type


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(
    aer_backend,
    capsys,
):
    aer_backend.summary()

    captured = capsys.readouterr()

    assert "QUANTUM BACKEND SUMMARY" in captured.out
    assert "Backend Type" in captured.out
    assert "Backend Name" in captured.out
    assert "Simulator" in captured.out


# ---------------------------------------------------------------------
# Representation Test
# ---------------------------------------------------------------------

def test_repr(aer_backend):
    representation = repr(aer_backend)

    assert "QuantumBackend" in representation
    assert "backend_type='aer_simulator'" in representation


# ---------------------------------------------------------------------
# Backend Consistency
# ---------------------------------------------------------------------

def test_backend_object_consistency(aer_backend):
    backend1 = aer_backend.get_backend()
    backend2 = aer_backend.get_backend()

    assert backend1 is backend2


def test_backend_name_consistency(aer_backend):
    name1 = aer_backend.backend_name()
    name2 = aer_backend.backend_name()

    assert name1 == name2
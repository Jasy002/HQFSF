"""
Unit tests for QuantumOptimizer.
"""

import pytest
from qiskit_algorithms.optimizers import (
    COBYLA,
    SPSA,
    SLSQP,
)

from quantum.optimizer import QuantumOptimizer


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def cobyla_optimizer():
    return QuantumOptimizer(
        optimizer="cobyla",
        maxiter=100,
    )


@pytest.fixture
def spsa_optimizer():
    return QuantumOptimizer(
        optimizer="spsa",
        maxiter=100,
    )


@pytest.fixture
def slsqp_optimizer():
    return QuantumOptimizer(
        optimizer="slsqp",
        maxiter=100,
    )


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------

def test_cobyla_initialization(cobyla_optimizer):
    assert cobyla_optimizer.optimizer == "cobyla"
    assert cobyla_optimizer.maxiter == 100


def test_spsa_initialization(spsa_optimizer):
    assert spsa_optimizer.optimizer == "spsa"


def test_slsqp_initialization(slsqp_optimizer):
    assert slsqp_optimizer.optimizer == "slsqp"


def test_invalid_optimizer():
    with pytest.raises(ValueError):
        QuantumOptimizer(
            optimizer="invalid",
            maxiter=100,
        )


def test_invalid_maxiter():
    with pytest.raises(ValueError):
        QuantumOptimizer(
            optimizer="cobyla",
            maxiter=0,
        )


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        QuantumOptimizer(
            optimizer="cobyla",
            tol=0,
        )


# ---------------------------------------------------------------------
# Optimizer Factory Tests
# ---------------------------------------------------------------------

def test_get_cobyla_optimizer(cobyla_optimizer):
    optimizer = cobyla_optimizer.get_optimizer()

    assert isinstance(optimizer, COBYLA)


def test_get_spsa_optimizer(spsa_optimizer):
    optimizer = spsa_optimizer.get_optimizer()

    assert isinstance(optimizer, SPSA)


def test_get_slsqp_optimizer(slsqp_optimizer):
    optimizer = slsqp_optimizer.get_optimizer()

    assert isinstance(optimizer, SLSQP)


# ---------------------------------------------------------------------
# Available Optimizers
# ---------------------------------------------------------------------

def test_available_optimizers():
    optimizers = QuantumOptimizer.available_optimizers()

    assert isinstance(optimizers, list)

    assert "cobyla" in optimizers
    assert "spsa" in optimizers
    assert "slsqp" in optimizers


def test_available_optimizer_count():
    optimizers = QuantumOptimizer.available_optimizers()

    assert len(optimizers) == 3


# ---------------------------------------------------------------------
# Tolerance Tests
# ---------------------------------------------------------------------

def test_optimizer_with_tolerance():
    optimizer = QuantumOptimizer(
        optimizer="cobyla",
        maxiter=100,
        tol=1e-6,
    )

    opt = optimizer.get_optimizer()

    assert isinstance(opt, COBYLA)


# ---------------------------------------------------------------------
# Summary Test
# ---------------------------------------------------------------------

def test_summary(
    cobyla_optimizer,
    capsys,
):
    cobyla_optimizer.summary()

    captured = capsys.readouterr()

    assert "QUANTUM OPTIMIZER SUMMARY" in captured.out
    assert "Optimizer" in captured.out
    assert "Max Iter" in captured.out
    assert "Available Optimizers" in captured.out


# ---------------------------------------------------------------------
# Representation Test
# ---------------------------------------------------------------------

def test_repr(cobyla_optimizer):
    representation = repr(cobyla_optimizer)

    assert "QuantumOptimizer" in representation
    assert "optimizer='cobyla'" in representation
    assert "maxiter=100" in representation


# ---------------------------------------------------------------------
# Consistency Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "optimizer_name, expected_class",
    [
        ("cobyla", COBYLA),
        ("spsa", SPSA),
        ("slsqp", SLSQP),
    ],
)
def test_optimizer_factory(
    optimizer_name,
    expected_class,
):
    optimizer = QuantumOptimizer(
        optimizer=optimizer_name,
        maxiter=50,
    )

    instance = optimizer.get_optimizer()

    assert isinstance(
        instance,
        expected_class,
    )


def test_multiple_calls_return_same_type():
    optimizer = QuantumOptimizer(
        optimizer="cobyla",
    )

    opt1 = optimizer.get_optimizer()
    opt2 = optimizer.get_optimizer()

    assert isinstance(opt1, COBYLA)
    assert isinstance(opt2, COBYLA)
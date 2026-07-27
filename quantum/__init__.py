"""
Quantum Package for HQFSF.

This package provides all quantum computing components used in the
Hybrid Quantum Feature Selection Framework (HQFSF).

Modules
-------
- encoder              : Quantum feature encoding
- ansatz               : Variational quantum ansatz
- circuit              : Variational quantum circuit builder
- backend              : Quantum backend management
- measurement          : Circuit execution and measurements
- expectation         : Expectation value computation
- optimizer           : Qiskit optimizer factory
- feature_selector    : Quantum feature ranking and selection
- metrics             : Evaluation metrics
"""

from __future__ import annotations

from .encoder import QuantumEncoder
from .ansatz import VariationalAnsatz
from .circuit import HQFSFCircuit
from .backend import QuantumBackend
from .measurement import QuantumMeasurement
from .expectation import ExpectationCalculator
from .optimizer import QuantumOptimizer
from .feature_selector import QuantumFeatureSelector
from .metrics import QuantumMetrics

__all__ = [
    "QuantumEncoder",
    "VariationalAnsatz",
    "HQFSFCircuit",
    "QuantumBackend",
    "QuantumMeasurement",
    "ExpectationCalculator",
    "QuantumOptimizer",
    "QuantumFeatureSelector",
    "QuantumMetrics",
]

__version__ = "1.0.0"

__author__ = "Jasmine Sultana"

__project__ = "HQFSF (Hybrid Quantum Feature Selection Framework)"
"""
HQFSF Pipeline Package
======================

High-level orchestration layer for the
Hybrid Quantum Feature Selection Framework (HQFSF).

Modules
-------
ClassicalPipeline
    Dataset loading, preprocessing and train-test splitting.

QuantumPipeline
    Quantum feature encoding, execution and feature selection.

EvaluationPipeline
    Classification metrics and performance evaluation.

HQFSFPipeline
    End-to-end workflow combining all pipeline stages.
"""

from .classical_pipeline import ClassicalPipeline
from .quantum_pipeline import QuantumPipeline
from .evaluation_pipeline import EvaluationPipeline
from .hqfsf_pipeline import HQFSFPipeline

__all__ = [
    "ClassicalPipeline",
    "QuantumPipeline",
    "EvaluationPipeline",
    "HQFSFPipeline",
]

__version__ = "1.0.0"
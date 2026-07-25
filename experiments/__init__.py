"""
HQFSF Experiment Package.

This package provides utilities for running
machine learning experiments and benchmarking
the HQFSF framework.

Modules
-------
- Experiment Runner
- Baseline Comparison
- Model Comparison
- Benchmark Evaluation
- Experiment Report
"""

from .experiment import Experiment
from .baseline import BaselineExperiment
from .comparison import ComparisonExperiment
from .benchmark import BenchmarkExperiment
from .report import ExperimentReport

__all__ = [
    "Experiment",
    "BaselineExperiment",
    "ComparisonExperiment",
    "BenchmarkExperiment",
    "ExperimentReport",
]

__version__ = "1.0.0"

__author__ = "Jasmine Sultana"
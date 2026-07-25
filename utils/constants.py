"""
Project-wide constants for HQFSF.
"""

PROJECT_NAME = "HQFSF"
PROJECT_VERSION = "1.0.0"

RANDOM_SEED = 42

SUPPORTED_DATASETS = [
    "breast_cancer",
    "diabetes",
    "heart",
    "wine",
    "sonar"
]

SUPPORTED_SCALERS = [
    "minmax",
    "standard",
    "robust"
]

SUPPORTED_ENCODINGS = [
    "angle",
    "amplitude",
    "basis"
]

SUPPORTED_BACKENDS = [
    "aer_simulator",
    "statevector_simulator",
    "ibm_quantum"
]

SUPPORTED_OPTIMIZERS = [
    "COBYLA",
    "SPSA",
    "SLSQP",
    "ADAM"
]

SUPPORTED_CLASSIFIERS = [
    "random_forest",
    "svm",
    "logistic_regression",
    "decision_tree"
]

DEFAULT_SHOTS = 1024
DEFAULT_QUBITS = 4
DEFAULT_CV = 5
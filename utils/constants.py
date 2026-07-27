"""
constants.py
============

Project-wide constants for the
Hybrid Quantum Feature Selection Framework (HQFSF).
"""

from pathlib import Path

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "HQFSF"

PROJECT_VERSION = "1.0.0"

AUTHOR = "Jasmine Sultana"

DESCRIPTION = (
    "Hybrid Quantum Feature Selection Framework "
    "Using Variational Quantum Circuits"
)

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# DATASET SETTINGS
# ==========================================================

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

SHUFFLE_DATA = True

# ==========================================================
# PREPROCESSING
# ==========================================================

MISSING_VALUE_STRATEGY = "mean"

SCALING_METHOD = "standard"

ENCODING_METHOD = "label"

# ==========================================================
# QUANTUM SETTINGS
# ==========================================================

DEFAULT_QUBITS = 4

DEFAULT_LAYERS = 2

DEFAULT_SHOTS = 1024

DEFAULT_ENCODING = "ry"

DEFAULT_ENTANGLEMENT = "linear"

DEFAULT_BACKEND = "aer_simulator"

TOP_K_FEATURES = 5

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

SUPPORTED_MODELS = [

    "random_forest",

    "svm",

    "logistic_regression",

    "xgboost",

]

DEFAULT_MODEL = "random_forest"

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"

LOG_FILE = "logs/hqfsf.log"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR = BASE_DIR / "saved_models"

RESULT_DIR = BASE_DIR / "results"

PLOT_DIR = BASE_DIR / "plots"

LOG_DIR = BASE_DIR / "logs"

# ==========================================================
# FILE EXTENSIONS
# ==========================================================

CSV_EXTENSION = ".csv"

JSON_EXTENSION = ".json"

PICKLE_EXTENSION = ".pkl"

MODEL_EXTENSION = ".joblib"

PNG_EXTENSION = ".png"

# ==========================================================
# EVALUATION METRICS
# ==========================================================

CLASSIFICATION_METRICS = [

    "accuracy",

    "precision",

    "recall",

    "f1_score",

    "roc_auc",

]

# ==========================================================
# COLORS
# ==========================================================

PRIMARY_COLOR = "#1F77B4"

SECONDARY_COLOR = "#FF7F0E"

SUCCESS_COLOR = "#2CA02C"

WARNING_COLOR = "#D62728"

# ==========================================================
# BANNER
# ==========================================================

LINE = "=" * 70

# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return (
        f"{PROJECT_NAME} "
        f"v{PROJECT_VERSION}"
    )
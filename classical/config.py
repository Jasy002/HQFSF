"""
Project configuration constants.
"""

from pathlib import Path

# ----------------------------------------------------
# Project Paths
# ----------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "datasets"

OUTPUT_DIR = ROOT_DIR / "outputs"

LOG_DIR = ROOT_DIR / "logs"

MODEL_DIR = ROOT_DIR / "models"

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------

SCALER = "minmax"

HANDLE_MISSING = True

REMOVE_DUPLICATES = True

ENCODE_LABELS = True

# ----------------------------------------------------
# Quantum
# ----------------------------------------------------

N_QUBITS = 4

SHOTS = 1024

OPTIMIZER = "COBYLA"

MAX_ITERATIONS = 100
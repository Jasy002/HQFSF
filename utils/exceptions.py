"""
exceptions.py
=============

Custom exception classes for the
Hybrid Quantum Feature Selection Framework (HQFSF).
"""


class HQFSFError(Exception):
    """
    Base exception class for HQFSF.
    """

    def __init__(self, message="An HQFSF error occurred."):

        super().__init__(message)


# ==========================================================
# DATASET EXCEPTIONS
# ==========================================================

class DatasetError(HQFSFError):
    """
    Raised when dataset-related errors occur.
    """

    def __init__(self, message="Dataset processing failed."):

        super().__init__(message)


class DatasetNotFoundError(DatasetError):
    """
    Raised when the dataset file cannot be found.
    """

    def __init__(self, filepath):

        super().__init__(
            f"Dataset not found: {filepath}"
        )


class InvalidDatasetError(DatasetError):
    """
    Raised when the dataset format is invalid.
    """

    def __init__(self, message="Invalid dataset format."):

        super().__init__(message)


# ==========================================================
# PREPROCESSING EXCEPTIONS
# ==========================================================

class PreprocessingError(HQFSFError):
    """
    Raised during preprocessing failures.
    """

    def __init__(self, message="Preprocessing failed."):

        super().__init__(message)


# ==========================================================
# QUANTUM EXCEPTIONS
# ==========================================================

class QuantumError(HQFSFError):
    """
    Raised for quantum computation errors.
    """

    def __init__(self, message="Quantum execution failed."):

        super().__init__(message)


class QuantumCircuitError(QuantumError):
    """
    Raised when circuit creation fails.
    """

    def __init__(self, message="Quantum circuit construction failed."):

        super().__init__(message)


class BackendError(QuantumError):
    """
    Raised when backend execution fails.
    """

    def __init__(self, message="Quantum backend error."):

        super().__init__(message)


# ==========================================================
# MODEL EXCEPTIONS
# ==========================================================

class ModelError(HQFSFError):
    """
    Raised for machine learning model errors.
    """

    def __init__(self, message="Model execution failed."):

        super().__init__(message)


class ModelTrainingError(ModelError):
    """
    Raised during model training.
    """

    def __init__(self, message="Model training failed."):

        super().__init__(message)


class PredictionError(ModelError):
    """
    Raised during prediction.
    """

    def __init__(self, message="Prediction failed."):

        super().__init__(message)


# ==========================================================
# PIPELINE EXCEPTIONS
# ==========================================================

class PipelineError(HQFSFError):
    """
    Raised when a pipeline stage fails.
    """

    def __init__(self, message="Pipeline execution failed."):

        super().__init__(message)


# ==========================================================
# CONFIGURATION EXCEPTIONS
# ==========================================================

class ConfigurationError(HQFSFError):
    """
    Raised for invalid configuration settings.
    """

    def __init__(self, message="Invalid configuration."):

        super().__init__(message)


# ==========================================================
# VALIDATION EXCEPTIONS
# ==========================================================

class ValidationError(HQFSFError):
    """
    Raised during validation failures.
    """

    def __init__(self, message="Validation failed."):

        super().__init__(message)


# ==========================================================
# FILE EXCEPTIONS
# ==========================================================

class FileOperationError(HQFSFError):
    """
    Raised during file operations.
    """

    def __init__(self, message="File operation failed."):

        super().__init__(message)


# ==========================================================
# REPRESENTATION
# ==========================================================

def __repr__():

    return "HQFSF Custom Exceptions"
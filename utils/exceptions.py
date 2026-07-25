"""
Custom exceptions used throughout HQFSF.
"""

class HQFSFError(Exception):
    """Base exception for the project."""
    pass


class DatasetError(HQFSFError):
    """Raised when dataset loading fails."""
    pass


class PreprocessingError(HQFSFError):
    """Raised during preprocessing."""
    pass


class QuantumError(HQFSFError):
    """Raised during quantum execution."""
    pass


class ConfigurationError(HQFSFError):
    """Raised when configuration is invalid."""
    pass


class EvaluationError(HQFSFError):
    """Raised during evaluation."""
    pass
class RMUException(Exception):
    """Base exception class for RMU-related errors."""
    pass

class ModelLoadError(RMUException):
    """Raised when there's an error loading the model."""
    pass

class DatasetError(RMUException):
    """Raised when there's an error with dataset operations."""
    pass

class ConfigurationError(RMUException):
    """Raised when there's an error in the configuration."""
    pass

class BenchmarkError(RMUException):
    """Raised when there's an error during benchmarking."""
    pass
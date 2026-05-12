class DeltaError(ValueError):
    """Raised when Δ' <= 0 or imaginary."""
    pass

class ConvergenceError(ValueError):
    """Raised when N cannot satisfy constraints."""
    pass

class ParameterError(RuntimeError):
    """Raised when command‑line arguments violate physical or consistency constraints."""
    pass

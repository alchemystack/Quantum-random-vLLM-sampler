"""Custom exception hierarchy for qc-sampler.

All exceptions inherit from QCSamplerError, allowing callers to catch
the entire family with a single handler while still being able to
distinguish specific failure modes.
"""


class QCSamplerError(Exception):
    """Base exception for all qc-sampler errors.

    Every exception raised by qc-sampler code (as opposed to standard
    library or third-party exceptions) inherits from this class.
    """


class EntropyUnavailableError(QCSamplerError):
    """Raised when no entropy source can provide the requested bytes.

    This may occur when the QRNG gRPC server is unreachable and all
    retries have been exhausted, or when the configured fallback mode
    is "error" and the primary source fails.
    """


class ConfigValidationError(QCSamplerError):
    """Raised when configuration values fail validation.

    The error message includes the field name, the invalid value,
    and the expected type or range to aid troubleshooting.
    """


class SignalAmplificationError(QCSamplerError):
    """Raised when the signal amplification algorithm fails.

    For example, if the raw byte buffer is empty or has an unexpected
    format that prevents the z-score computation from completing.
    """


class TokenSelectionError(QCSamplerError):
    """Raised when token selection fails.

    This can occur if the candidate set is empty after top-k and
    top-p filtering, or if the logit distribution is degenerate in
    a way that prevents CDF construction.
    """

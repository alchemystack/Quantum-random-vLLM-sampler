"""Exception hierarchy for qr-sampler.

All exceptions derive from QRSamplerError, enabling broad catch patterns
at the application boundary while allowing fine-grained handling internally.
"""


class QRSamplerError(Exception):
    """Base exception for all qr-sampler errors."""


class EntropyUnavailableError(QRSamplerError):
    """No entropy source can provide bytes.

    Raised when the primary entropy source fails and either no fallback
    is configured or the fallback also fails.
    """


class ConfigValidationError(QRSamplerError):
    """Configuration field validation failed.

    Raised when per-request extra_args contain invalid keys, attempt to
    override non-overridable infrastructure fields, or fail type validation.
    """


class SignalAmplificationError(QRSamplerError):
    """Signal amplification computation failed.

    Raised when the amplifier receives invalid input (e.g., empty bytes)
    or encounters a numerical error during z-score computation.
    """


class TokenSelectionError(QRSamplerError):
    """Token selection failed.

    Raised when no candidate tokens survive top-k and top-p filtering,
    making it impossible to select a token from the CDF.
    """

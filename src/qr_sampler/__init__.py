"""qr-sampler: Plug any randomness source into LLM token sampling via vLLM.

A vLLM V1 LogitsProcessor plugin that replaces standard token sampling with
external-entropy-driven selection. Supports quantum random number generators,
processor timing jitter, and any user-supplied entropy source via gRPC.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("qr-sampler")
except PackageNotFoundError:
    __version__ = "0.0.0"

from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.exceptions import (
    ConfigValidationError,
    EntropyUnavailableError,
    QRSamplerError,
    SignalAmplificationError,
    TokenSelectionError,
)
from qr_sampler.processor import QRSamplerLogitsProcessor

__all__ = [
    "ConfigValidationError",
    "EntropyUnavailableError",
    "QRSamplerConfig",
    "QRSamplerError",
    "QRSamplerLogitsProcessor",
    "SignalAmplificationError",
    "TokenSelectionError",
    "__version__",
    "resolve_config",
    "validate_extra_args",
]

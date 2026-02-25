"""Diagnostic logging subsystem for qr-sampler.

Provides immutable per-token sampling records and a configurable logger
that supports none/summary/full verbosity and in-memory diagnostic mode.
"""

from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.types import TokenSamplingRecord

__all__ = [
    "SamplingLogger",
    "TokenSamplingRecord",
]

"""Signal amplification subsystem for qr-sampler.

Converts raw entropy bytes into a uniform float u in (eps, 1-eps) that drives
CDF-based token selection. The amplification preserves even tiny biases
in the entropy source.
"""

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.zscore import ZScoreMeanAmplifier

__all__ = [
    "AmplificationResult",
    "AmplifierRegistry",
    "SignalAmplifier",
    "ZScoreMeanAmplifier",
]

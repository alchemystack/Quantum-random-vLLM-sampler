"""qc-sampler: Quantum consciousness token sampling plugin for vLLM V1.

Replaces standard pseudorandom token sampling with quantum-random sampling,
allowing consciousness to influence token selection through quantum random
number generators (QRNGs). Uses a z-score signal amplification algorithm
to convert raw QRNG bytes into a uniform float that drives token selection
through a probability-ordered cumulative distribution function (CDF).
"""

__version__ = "0.1.0"

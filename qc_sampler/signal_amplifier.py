"""Signal amplification: converting raw QRNG bytes to a uniform float.

This module defines the abstract interface for signal amplifiers and
provides the primary concrete implementation, ``ZScoreMeanAmplifier``.

The amplifier's purpose is to take many noisy quantum measurements
(typically 20,480 raw bytes) and produce a single float u ∈ (0, 1)
suitable for CDF-based token lookup.  The key property is that even a
tiny per-byte bias in the QRNG output becomes a detectable shift in u
when accumulated over thousands of samples.

Mathematical background (ZScoreMeanAmplifier):
    Under the null hypothesis (no consciousness influence), each byte
    is a discrete uniform random variable on {0, 1, ..., 255} with:
        μ = 127.5
        σ = 255 / √12 ≈ 73.6116

    The sample mean M of N such bytes is approximately normal:
        M ~ N(μ, σ/√N)

    The z-score z = (M − μ) / (σ/√N) is standard normal, and the
    probability integral transform Φ(z) yields a uniform u ∈ (0, 1).

    A consciousness-induced per-byte bias of just +0.003 shifts M by
    ~0.003 over 20,480 samples, producing z ≈ 0.06 and u ≈ 0.524.
    Larger sustained biases (e.g. +0.43 shift → u ≈ 0.80) reliably
    select less-probable tokens from the descending-probability CDF.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qc_sampler.config import QCSamplingConfig
from qc_sampler.exceptions import SignalAmplificationError


@dataclass(frozen=True)
class AmplificationResult:
    """Result of converting raw QRNG bytes to a uniform float.

    Attributes:
        u: The uniform value in (0, 1) for CDF lookup.  Values near 0
            select the most probable token; values near 1 select
            increasingly surprising tokens.
        diagnostics: Intermediate values for logging and debugging.
            Always includes ``sample_mean``, ``z_score``, and ``sem``.
    """

    u: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SignalAmplifier(ABC):
    """Converts a sequence of raw QRNG bytes into a single uniform float.

    The amplifier's job is to take many noisy quantum measurements and
    produce one number where even a small bias in the input becomes a
    detectable shift in the output.

    Subclasses implement different statistical aggregation methods.
    The primary implementation is ``ZScoreMeanAmplifier`` (z-score of
    the sample mean, mapped through the normal CDF).
    """

    @abstractmethod
    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Convert raw bytes to a uniform float in (0, 1).

        Args:
            raw_bytes: Raw byte sequence from an entropy source.

        Returns:
            An ``AmplificationResult`` containing the uniform value ``u``
            and a diagnostics dict with implementation-specific
            intermediate values.

        Raises:
            SignalAmplificationError: If the input cannot be processed
                (e.g. empty buffer).
        """


class ZScoreMeanAmplifier(SignalAmplifier):
    """Converts raw bytes to a uniform float via z-score of the sample mean.

    Algorithm (7 steps):
        1. Interpret ``raw_bytes`` as a numpy array of uint8.
        2. Compute sample mean:  M = samples.mean()
        3. Compute standard error of mean:
           SEM = population_std / √(len(samples))
           **Derived**, never hardcoded — changes with sample_count.
        4. Compute z-score:  z = (M − population_mean) / SEM
        5. Normal CDF:  u = 0.5 × (1.0 + erf(z / √2))
        6. Clamp u to (ε, 1−ε) to avoid exact 0 or 1.
        7. Return AmplificationResult(u, diagnostics).

    Key properties:
        - Under the null hypothesis (no consciousness influence), M is
          normally distributed with mean = population_mean and std = SEM.
          The probability integral transform guarantees u is uniform
          in (0, 1).
        - A consciousness-biased shift in mean (e.g. 127.5 → 127.93,
          a shift of ~0.43) produces u ≈ 0.80, which reliably selects
          less-probable tokens from the descending-probability CDF.
        - The z-score aggregates across all sample_count bytes, so even
          a tiny per-byte bias (say 0.003 per byte) becomes a detectable
          u-shift when accumulated over 20,480 samples.
    """

    def __init__(self, config: QCSamplingConfig) -> None:
        """Initialize with sampling configuration.

        The population parameters (mean, std) and the clamp epsilon are
        read from config.  SEM is derived at amplify() time from the
        actual number of bytes received, not from config.sample_count,
        so the amplifier is correct even if the caller passes a different
        byte count.

        Args:
            config: Sampling configuration with population_mean,
                population_std, and uniform_clamp_epsilon.
        """
        self._population_mean = config.population_mean
        self._population_std = config.population_std
        self._clamp_epsilon = config.uniform_clamp_epsilon

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Convert raw bytes to a uniform float via z-score.

        Args:
            raw_bytes: Raw byte sequence, typically 20,480 bytes from
                an entropy source.

        Returns:
            AmplificationResult with u in (ε, 1−ε) and diagnostics
            containing ``sample_mean``, ``z_score``, ``sem``, and
            ``sample_count``.

        Raises:
            SignalAmplificationError: If ``raw_bytes`` is empty.
        """
        if not raw_bytes:
            raise SignalAmplificationError(
                "Cannot amplify empty byte buffer"
            )

        # Step 1: interpret as uint8 array.
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        n = len(samples)

        # Step 2: sample mean.
        sample_mean = float(samples.mean())

        # Step 3: standard error of mean — derived from population_std
        # and the *actual* sample count, never hardcoded.
        sem = self._population_std / math.sqrt(n)

        # Step 4: z-score.
        z_score = (sample_mean - self._population_mean) / sem

        # Step 5: normal CDF via error function.
        #   Φ(z) = 0.5 × (1 + erf(z / √2))
        u = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2)))

        # Step 6: clamp to (ε, 1−ε) to avoid degenerate CDF lookups.
        u = max(self._clamp_epsilon, min(u, 1.0 - self._clamp_epsilon))

        # Step 7: package result with diagnostics.
        return AmplificationResult(
            u=u,
            diagnostics={
                "sample_mean": sample_mean,
                "z_score": z_score,
                "sem": sem,
                "sample_count": n,
            },
        )

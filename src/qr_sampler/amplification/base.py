"""Base classes for signal amplification.

Defines the abstract interface and result type for converting raw entropy
bytes into a uniform float u in (eps, 1-eps).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AmplificationResult:
    """Result of signal amplification.

    Attributes:
        u: Uniform value in (eps, 1-eps), used for CDF-based token selection.
        diagnostics: Additional info (sample_mean, z_score, sem, sample_count).
    """

    u: float
    diagnostics: dict[str, Any]


class SignalAmplifier(ABC):
    """Abstract base class for signal amplification algorithms.

    Implementations convert raw entropy bytes into a single uniform float
    suitable for CDF-based token selection. The amplification process is
    designed to preserve even tiny biases in the entropy source while
    mapping the aggregate statistic to a well-defined uniform distribution
    under the null hypothesis.
    """

    @abstractmethod
    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Convert raw entropy bytes into a uniform float u in (eps, 1-eps).

        Args:
            raw_bytes: Raw entropy bytes from an entropy source.

        Returns:
            AmplificationResult containing the uniform value and diagnostics.

        Raises:
            SignalAmplificationError: If raw_bytes is empty or computation fails.
        """

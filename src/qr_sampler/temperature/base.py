"""Base classes for temperature strategies.

Defines the abstract interface, result type, and shared utility for
computing Shannon entropy from logit distributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig


@dataclass(frozen=True, slots=True)
class TemperatureResult:
    """Result of a temperature strategy computation.

    Attributes:
        temperature: Temperature to use for this token's sampling.
        shannon_entropy: Shannon entropy H of the logit distribution (nats).
        diagnostics: Additional info (strategy-specific details).
    """

    temperature: float
    shannon_entropy: float
    diagnostics: dict[str, Any]


class TemperatureStrategy(ABC):
    """Abstract base class for temperature strategies.

    Implementations compute a per-token temperature from the logit
    distribution. All strategies must compute and return Shannon entropy
    even if it is not used in the temperature formula, because the
    logging subsystem depends on it.
    """

    @abstractmethod
    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute temperature for a single token.

        Args:
            logits: 1-D logit array for the current token (vocab_size,).
            config: Active configuration for this request.

        Returns:
            TemperatureResult with temperature, Shannon entropy, and diagnostics.
        """


def compute_shannon_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy H = -sum(p_i * ln(p_i)) using numerically stable softmax.

    Uses the shift-by-max trick for numerical stability. Returns 0.0 for
    degenerate distributions where only one token has non-zero probability.

    Args:
        logits: 1-D logit array (vocab_size,).

    Returns:
        Shannon entropy in nats (natural log base).
    """
    # Numerically stable softmax: shift by max to prevent overflow.
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    if sum_exp == 0.0:
        return 0.0

    probs = exp_shifted / sum_exp

    # H = -sum(p * ln(p)), skipping zeros to avoid log(0).
    mask = probs > 0
    log_probs = np.log(probs[mask])
    entropy = -float(np.sum(probs[mask] * log_probs))

    # Guard against floating-point artifacts producing tiny negatives.
    return max(0.0, entropy)

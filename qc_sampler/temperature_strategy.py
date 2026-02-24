"""Temperature strategies for per-token temperature computation.

This module defines the abstract interface for temperature strategies and
provides two concrete implementations:

- **FixedTemperatureStrategy**: Returns a constant temperature regardless
  of the logit distribution.  Simple and predictable.
- **EDTTemperatureStrategy**: Entropy-based dynamic temperature.  Modulates
  temperature per-token based on Shannon entropy of the logit distribution,
  inspired by the EDT paper (arXiv:2403.14541) and llama.cpp's DynaTemp.

Design notes:
    The ``TemperatureStrategy`` ABC decouples temperature computation from
    token selection.  Adding a new temperature formula requires only writing
    a new subclass and registering it in the factory — no changes to the
    selector, processor, or any other module.

    Both strategies always compute Shannon entropy, even when the formula
    doesn't use it, because the sampling logger records it for every token.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qc_sampler.config import QCSamplingConfig


@dataclass(frozen=True)
class TemperatureResult:
    """Result of computing the sampling temperature for one token.

    Attributes:
        temperature: The temperature to apply during sampling.
        shannon_entropy: Shannon entropy H of the logit distribution
            (natural log), computed for logging regardless of strategy.
        diagnostics: Strategy-specific intermediate values for debugging.
    """

    temperature: float
    shannon_entropy: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TemperatureStrategy(ABC):
    """Determines the sampling temperature for a given token position.

    Subclasses implement different formulas for mapping the logit
    distribution to a temperature value.  The interface guarantees
    that Shannon entropy is always computed (for logging) even if
    the strategy doesn't use it in its formula.
    """

    @abstractmethod
    def compute_temperature(
        self,
        logits: np.ndarray,
        config: QCSamplingConfig,
    ) -> TemperatureResult:
        """Compute the temperature to use for this token.

        Args:
            logits: Raw logits for this token position, shape ``(vocab_size,)``.
            config: Resolved per-request configuration.

        Returns:
            A ``TemperatureResult`` containing the temperature, Shannon
            entropy, and any strategy-specific diagnostics.
        """


def compute_shannon_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy H of the softmax distribution over logits.

    Uses numerically stable softmax (shift by max before exp) to prevent
    overflow.  Entropy is computed using natural logarithm:

        H = -Σ(p_i · ln(p_i))   for all p_i > 0

    Args:
        logits: Raw logits, shape ``(vocab_size,)``.  May contain -inf
            for masked positions.

    Returns:
        Shannon entropy in nats.  Returns 0.0 for degenerate distributions
        (single token with all probability mass).
    """
    # Stable softmax: shift by max to prevent exp overflow.
    max_logit = np.max(logits)
    if not np.isfinite(max_logit):
        # All logits are -inf (or array is degenerate) → no valid distribution.
        return 0.0

    shifted = logits - max_logit
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum()

    if sum_exp == 0.0:
        return 0.0

    probs = exp_shifted / sum_exp

    # Only include tokens with positive probability to avoid log(0).
    mask = probs > 0
    p = probs[mask]
    entropy = -float(np.sum(p * np.log(p)))
    return entropy


class FixedTemperatureStrategy(TemperatureStrategy):
    """Returns a constant temperature regardless of the logit distribution.

    This is the simplest strategy: every token is sampled at the same
    temperature specified in ``config.fixed_temperature``.  Shannon
    entropy is still computed and returned for logging purposes.
    """

    def compute_temperature(
        self,
        logits: np.ndarray,
        config: QCSamplingConfig,
    ) -> TemperatureResult:
        """Return the fixed temperature from config.

        Args:
            logits: Raw logits (used only for entropy computation).
            config: Must have ``fixed_temperature`` set.

        Returns:
            TemperatureResult with ``temperature = config.fixed_temperature``.
        """
        entropy = compute_shannon_entropy(logits)
        return TemperatureResult(
            temperature=config.fixed_temperature,
            shannon_entropy=entropy,
            diagnostics={"strategy": "fixed"},
        )


class EDTTemperatureStrategy(TemperatureStrategy):
    """Entropy-based dynamic temperature (EDT).

    Modulates temperature per-token based on the Shannon entropy of the
    logit distribution.  High-entropy tokens (model is uncertain) get
    higher temperature (more diversity); low-entropy tokens (model is
    confident) get lower temperature (more deterministic).

    Formula:
        H      = Shannon entropy of softmax(logits)
        H_norm = H / ln(vocab_size)       # normalize to [0, 1]
        T      = edt_base_temp × H_norm ^ edt_exponent
        T      = clamp(T, edt_min_temp, edt_max_temp)

    Parameters (all from config):
        edt_base_temp (T₀): Scales the overall temperature range.
        edt_exponent  (θ):  Controls curve shape.
            θ < 1 → concave: temperature rises quickly with entropy.
            θ = 1 → linear mapping.
            θ > 1 → convex: temperature rises slowly with entropy.
        edt_min_temp: Floor for temperature (prevents near-zero T).
        edt_max_temp: Ceiling for temperature (prevents extreme T).

    References:
        - arXiv:2403.14541 (Entropy-based Dynamic Temperature)
        - llama.cpp DynaTemp implementation
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize with vocabulary size for entropy normalization.

        Args:
            vocab_size: Total vocabulary size of the model.  Used to
                compute the theoretical maximum entropy (uniform
                distribution over all tokens): H_max = ln(vocab_size).
        """
        if vocab_size < 1:
            raise ValueError(
                f"vocab_size must be at least 1, got {vocab_size}"
            )
        self._vocab_size = vocab_size
        # Precompute maximum entropy: ln(vocab_size).
        # This is the entropy of a uniform distribution over the vocabulary.
        self._max_entropy = math.log(vocab_size)

    def compute_temperature(
        self,
        logits: np.ndarray,
        config: QCSamplingConfig,
    ) -> TemperatureResult:
        """Compute entropy-based dynamic temperature.

        Args:
            logits: Raw logits for this token position.
            config: Must have edt_base_temp, edt_exponent, edt_min_temp,
                and edt_max_temp set.

        Returns:
            TemperatureResult with temperature computed via the EDT formula,
            clamped to [edt_min_temp, edt_max_temp].
        """
        entropy = compute_shannon_entropy(logits)

        # Normalize entropy to [0, 1].
        # Guard against max_entropy == 0 (vocab_size == 1).
        if self._max_entropy > 0.0:
            h_norm = entropy / self._max_entropy
        else:
            h_norm = 0.0

        # Clamp h_norm to [0, 1] to guard against floating-point drift.
        h_norm = max(0.0, min(1.0, h_norm))

        # EDT formula: T = base_temp × h_norm ^ exponent
        # When h_norm is 0 and exponent > 0, this gives T = 0, which
        # will be clamped to edt_min_temp below.
        raw_temp = config.edt_base_temp * (h_norm ** config.edt_exponent)

        # Clamp to configured bounds.
        temperature = max(config.edt_min_temp, min(config.edt_max_temp, raw_temp))

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=entropy,
            diagnostics={
                "strategy": "edt",
                "h_norm": h_norm,
                "raw_temp": raw_temp,
                "max_entropy": self._max_entropy,
                "vocab_size": self._vocab_size,
            },
        )

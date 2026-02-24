"""Token selection via probability-ordered CDF and QRNG-derived uniform value.

This module implements the core sampling logic that converts a logit
distribution and a quantum-random uniform float into a specific token
selection.

The CDF is built over tokens sorted by **descending probability**.  This
means:
    - u close to 0 → selects the most probable token.
    - u close to 1 → selects increasingly improbable tokens.

This gives any bias in the QRNG-derived u a coherent semantic direction:
higher u = more surprising/creative output, lower u = more expected/safe.

Pipeline (in order):
    1. Temperature scaling: ``logits / temperature``
    2. Top-k filtering: keep only top-k highest logits, mask rest to -inf
    3. Softmax to probabilities (numerically stable: shift by max)
    4. Top-p (nucleus) filtering: minimal set with cumulative prob ≥ top_p,
       zero the rest, renormalize
    5. Sort remaining tokens by descending probability
    6. Build CDF over sorted order
    7. Binary search for first index k where CDF[k] ≥ u
    8. Map back to original vocabulary index

Performance notes:
    - Top-k uses ``np.argpartition`` for O(n) average instead of O(n log n).
    - Full sort only applies to the (much smaller) filtered candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qc_sampler.exceptions import TokenSelectionError


@dataclass(frozen=True)
class SelectionResult:
    """Result of selecting a single token from the logit distribution.

    Attributes:
        token_id: The vocabulary index of the selected token.
        token_rank: Rank in the probability-sorted candidate list
            (0 = most probable among candidates).
        token_prob: Probability of the selected token after all filtering
            and renormalization.
        num_candidates: Number of tokens that survived top-k and top-p
            filtering.
        diagnostics: Intermediate values for debugging: scaled logits
            stats, filtering details, etc.
    """

    token_id: int
    token_rank: int
    token_prob: float
    num_candidates: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TokenSelector:
    """Selects a token from the logit distribution using a QRNG-derived
    uniform value and a probability-ordered CDF.

    This class is stateless — all parameters are passed to ``select()``
    per call.  It does not hold configuration; the caller is responsible
    for resolving top-k, top-p, temperature, and u from their respective
    sources.
    """

    def select(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        u: float,
    ) -> SelectionResult:
        """Select a token from logits using the QRNG-derived uniform value.

        Full pipeline:
            1. Apply temperature scaling.
            2. Top-k filter (keep highest logits, mask rest).
            3. Softmax to probabilities.
            4. Top-p (nucleus) filter and renormalize.
            5. Sort by descending probability, build CDF.
            6. Select token where CDF first reaches u.

        Args:
            logits: Raw logits for one token position, shape ``(vocab_size,)``.
            temperature: Sampling temperature.  Must be > 0.
            top_k: Number of top logits to keep.  Values ≤ 0 disable
                top-k filtering.
            top_p: Nucleus sampling threshold in (0, 1].  1.0 disables
                top-p filtering.
            u: QRNG-derived uniform float in (0, 1).  Near 0 selects
                the most probable token; near 1 selects rare tokens.

        Returns:
            A ``SelectionResult`` with the chosen token and metadata.

        Raises:
            TokenSelectionError: If no candidates survive filtering, or
                if inputs are degenerate (e.g. temperature ≤ 0).
        """
        if temperature <= 0.0:
            raise TokenSelectionError(
                f"Temperature must be positive, got {temperature}"
            )

        vocab_size = len(logits)
        if vocab_size == 0:
            raise TokenSelectionError("Cannot select from empty logits array")

        # --- Step 1: Temperature scaling ---
        scaled_logits = logits / temperature

        # --- Step 2: Top-k filtering ---
        # Values ≤ 0 mean top-k is disabled.
        if top_k > 0 and top_k < vocab_size:
            scaled_logits, top_k_applied = self._apply_top_k(
                scaled_logits, top_k
            )
        else:
            top_k_applied = vocab_size

        # --- Step 3: Softmax (numerically stable) ---
        probs = self._stable_softmax(scaled_logits)

        # --- Step 4: Top-p (nucleus) filtering ---
        if 0.0 < top_p < 1.0:
            probs, top_p_count = self._apply_top_p(probs, top_p)
        else:
            top_p_count = int(np.sum(probs > 0))

        # --- Steps 5-7: Sort descending, build CDF, select ---
        token_id, token_rank, token_prob, num_candidates = self._cdf_select(
            probs, u
        )

        return SelectionResult(
            token_id=int(token_id),
            token_rank=int(token_rank),
            token_prob=float(token_prob),
            num_candidates=int(num_candidates),
            diagnostics={
                "temperature_applied": temperature,
                "top_k_applied": top_k_applied,
                "top_p_count": top_p_count,
                "u_value": u,
            },
        )

    @staticmethod
    def _apply_top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, int]:
        """Keep only the top-k logits; mask the rest to -inf.

        Uses ``np.argpartition`` for O(n) average complexity instead
        of a full O(n log n) sort.

        Args:
            logits: Scaled logits array (modified in-place on a copy).
            k: Number of top logits to retain.

        Returns:
            Tuple of (filtered logits, effective k).
        """
        result = np.full_like(logits, -np.inf)
        # argpartition: the top-k elements are in indices[-k:] (unordered).
        top_k_indices = np.argpartition(logits, -k)[-k:]
        result[top_k_indices] = logits[top_k_indices]
        return result, k

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability.

        Shifts logits by max before exp to prevent overflow.  Positions
        with -inf logits get probability 0 (exp(-inf) = 0).

        Args:
            logits: Logits array, may contain -inf for masked positions.

        Returns:
            Probability array that sums to 1.0 (or 0.0 if all -inf).
        """
        max_logit = np.max(logits)
        if not np.isfinite(max_logit):
            # All logits are -inf: return zero probabilities.
            return np.zeros_like(logits, dtype=np.float64)

        shifted = logits - max_logit
        exp_shifted = np.exp(shifted)
        total = exp_shifted.sum()

        if total == 0.0:
            return np.zeros_like(logits, dtype=np.float64)

        return exp_shifted / total

    @staticmethod
    def _apply_top_p(
        probs: np.ndarray, top_p: float
    ) -> tuple[np.ndarray, int]:
        """Apply nucleus (top-p) sampling: keep the minimal set of tokens
        whose cumulative probability reaches top_p.

        Sorts probabilities descending, finds the cutoff index where
        cumulative probability first exceeds top_p, zeros everything
        beyond that point, and renormalizes.

        Args:
            probs: Probability array (may have zeros from top-k).
            top_p: Cumulative probability threshold in (0, 1).

        Returns:
            Tuple of (filtered+renormalized probs, count of surviving tokens).
        """
        # Sort descending for cumulative probability check.
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative = np.cumsum(sorted_probs)
        # Find first index where cumulative prob >= top_p.
        # searchsorted finds insertion point; we want at least 1 token.
        cutoff_idx = int(np.searchsorted(cumulative, top_p, side="left"))
        # Include the cutoff token itself (the one that pushes us over).
        cutoff_idx = min(cutoff_idx + 1, len(sorted_probs))

        # Zero out tokens beyond the cutoff.
        result = np.zeros_like(probs)
        surviving_indices = sorted_indices[:cutoff_idx]
        result[surviving_indices] = probs[surviving_indices]

        # Renormalize.
        total = result.sum()
        if total > 0.0:
            result /= total

        return result, int(cutoff_idx)

    @staticmethod
    def _cdf_select(
        probs: np.ndarray, u: float
    ) -> tuple[int, int, float, int]:
        """Build descending-probability CDF and select via uniform value.

        Steps:
            1. Identify tokens with positive probability.
            2. Sort them by descending probability.
            3. Build CDF (cumulative sum).
            4. Force CDF[-1] = 1.0 to fix floating-point drift.
            5. Binary search for first CDF value ≥ u.
            6. Map back to original vocabulary index.

        Args:
            probs: Probability array over the full vocabulary.
            u: QRNG-derived uniform float in (0, 1).

        Returns:
            Tuple of (token_id, rank, probability, num_candidates).

        Raises:
            TokenSelectionError: If no tokens have positive probability.
        """
        # Find tokens with positive probability.
        active_mask = probs > 0
        active_indices = np.where(active_mask)[0]
        num_candidates = len(active_indices)

        if num_candidates == 0:
            raise TokenSelectionError(
                "No tokens with positive probability after filtering. "
                "This may indicate overly aggressive top-k/top-p settings "
                "or degenerate logits."
            )

        # Sort by descending probability.
        active_probs = probs[active_indices]
        desc_order = np.argsort(active_probs)[::-1]
        sorted_indices = active_indices[desc_order]
        sorted_probs = active_probs[desc_order]

        # Build CDF and fix floating-point drift at the tail.
        cdf = np.cumsum(sorted_probs)
        cdf[-1] = 1.0

        # Binary search: first index where CDF >= u.
        rank = int(np.searchsorted(cdf, u, side="left"))
        # Clamp to valid range (in case u is very close to 1.0 and
        # floating-point comparison pushes past the end).
        rank = min(rank, num_candidates - 1)

        token_id = sorted_indices[rank]
        token_prob = sorted_probs[rank]

        return int(token_id), rank, float(token_prob), num_candidates

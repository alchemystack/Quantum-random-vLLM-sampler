"""Tests for qc_sampler.token_selector.

Covers:
- Uniform probabilities + u = 0.0 → selects most probable (first in sort)
- Peaked distribution + low u → selects dominant token
- Peaked distribution + high u → selects non-dominant token
- Top-k filtering: only top_k tokens survive
- Top-p filtering: minimal set with cumulative prob ≥ top_p
- Edge case: all logits identical → no crash, valid selection
- Edge case: all logits -inf except one → selects the survivor
- Edge case: empty logits → raises TokenSelectionError
- Edge case: temperature ≤ 0 → raises TokenSelectionError
- CDF properties: rank 0 is always the most probable token
- SelectionResult is immutable (frozen dataclass)
"""

from __future__ import annotations

import numpy as np
import pytest

from qc_sampler.exceptions import TokenSelectionError
from qc_sampler.token_selector import SelectionResult, TokenSelector


@pytest.fixture
def selector() -> TokenSelector:
    """Provide a fresh TokenSelector instance."""
    return TokenSelector()


class TestBasicSelection:
    """Core selection behaviour with simple distributions."""

    def test_peaked_distribution_low_u_selects_dominant(
        self, selector: TokenSelector
    ) -> None:
        """Sharply peaked distribution + u near 0 → selects dominant token.

        One token at 0.9, rest at 0.01.  With u = 0.05 (well under 0.9
        cumulative), the dominant token is selected.
        """
        vocab_size = 10
        # Create logits where token 3 is dominant
        logits = np.full(vocab_size, -2.0, dtype=np.float64)
        logits[3] = 5.0  # This token will dominate after softmax

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.05
        )

        assert result.token_id == 3
        assert result.token_rank == 0  # Most probable
        assert result.token_prob > 0.5

    def test_peaked_distribution_high_u_selects_non_dominant(
        self, selector: TokenSelector
    ) -> None:
        """Moderately peaked distribution + u near 1 → selects non-dominant.

        With u = 0.95, we should land past the dominant token in the CDF
        and select one of the tail tokens.  The distribution is designed
        so the dominant token has prob ≈ 0.9 (not 0.99), leaving room
        for tail selection.
        """
        vocab_size = 10
        # Create a distribution where the dominant token has ~90% prob.
        # softmax([2, 0, 0, ...]) → dominant ≈ e^2/(e^2+9) ≈ 0.451
        # Use temperature to control: logits/T.  Instead, just use
        # logits that give ~90% after softmax.
        logits = np.zeros(vocab_size, dtype=np.float64)
        logits[3] = 3.0  # dominant prob ≈ e^3/(e^3 + 9) ≈ 0.69

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.95
        )

        assert result.token_id != 3  # NOT the dominant token
        assert result.token_rank > 0

    def test_uniform_distribution_u_zero_selects_first(
        self, selector: TokenSelector
    ) -> None:
        """Uniform probabilities + u = tiny → selects one of them (rank 0).

        With equal probabilities, the CDF grows in equal steps.
        u near 0 should select the first token in descending-prob order.
        """
        vocab_size = 10
        logits = np.zeros(vocab_size, dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.001
        )

        assert result.token_rank == 0  # First in sorted order
        assert 0 <= result.token_id < vocab_size
        assert result.num_candidates == vocab_size

    def test_uniform_distribution_u_high_selects_last(
        self, selector: TokenSelector
    ) -> None:
        """Uniform probabilities + u near 1 → selects last in sorted order."""
        vocab_size = 10
        logits = np.zeros(vocab_size, dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.999
        )

        assert result.token_rank == vocab_size - 1
        assert 0 <= result.token_id < vocab_size

    def test_u_zero_exact(self, selector: TokenSelector) -> None:
        """u = 0.0 (or very close) should select the most probable token."""
        logits = np.array([1.0, 5.0, 2.0, 3.0], dtype=np.float64)

        # u very close to 0 → CDF first exceeds at position 0 (most probable)
        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=1e-10
        )

        assert result.token_id == 1  # logits[1] = 5.0 is highest
        assert result.token_rank == 0


class TestTopKFiltering:
    """Top-k filtering: keep only top_k highest logits."""

    def test_top_k_limits_candidates(self, selector: TokenSelector) -> None:
        """When top_k = 3, at most 3 tokens should survive."""
        logits = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=3, top_p=1.0, u=0.5
        )

        assert result.num_candidates <= 3

    def test_top_k_one_always_selects_best(
        self, selector: TokenSelector
    ) -> None:
        """top_k = 1 → always selects the single highest-logit token."""
        logits = np.array([1.0, 5.0, 2.0, 8.0, 3.0], dtype=np.float64)

        # Any u value should select token 3 (logit 8.0)
        for u in [0.0, 0.5, 0.999]:
            result = selector.select(
                logits=logits, temperature=1.0, top_k=1, top_p=1.0, u=u
            )
            assert result.token_id == 3
            assert result.num_candidates == 1

    def test_top_k_zero_disables_filtering(
        self, selector: TokenSelector
    ) -> None:
        """top_k = 0 disables top-k filtering (all tokens survive)."""
        vocab_size = 20
        logits = np.random.default_rng(42).normal(size=vocab_size)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5
        )

        assert result.num_candidates == vocab_size

    def test_top_k_negative_disables_filtering(
        self, selector: TokenSelector
    ) -> None:
        """top_k < 0 disables top-k filtering."""
        vocab_size = 20
        logits = np.random.default_rng(42).normal(size=vocab_size)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=-1, top_p=1.0, u=0.5
        )

        assert result.num_candidates == vocab_size

    def test_top_k_larger_than_vocab_no_effect(
        self, selector: TokenSelector
    ) -> None:
        """top_k > vocab_size should behave like no filtering."""
        vocab_size = 5
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=100, top_p=1.0, u=0.5
        )

        assert result.num_candidates == vocab_size


class TestTopPFiltering:
    """Top-p (nucleus) filtering."""

    def test_top_p_one_disables_filtering(
        self, selector: TokenSelector
    ) -> None:
        """top_p = 1.0 disables nucleus filtering."""
        vocab_size = 10
        logits = np.random.default_rng(42).normal(size=vocab_size)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5
        )

        assert result.num_candidates == vocab_size

    def test_top_p_concentrates_on_dominant_token(
        self, selector: TokenSelector
    ) -> None:
        """If one token has prob > top_p, only that token survives.

        Token 0 with logit 10.0 will dominate softmax (prob ≈ 0.99+).
        With top_p = 0.9, only this token is needed.
        """
        logits = np.full(10, -5.0, dtype=np.float64)
        logits[0] = 10.0  # Dominant

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=0.9, u=0.5
        )

        assert result.token_id == 0
        assert result.num_candidates == 1

    def test_top_p_keeps_minimal_set(self, selector: TokenSelector) -> None:
        """top_p = 0.5 with a relatively flat distribution keeps about half."""
        # 4 tokens with equal logits → each has prob 0.25
        logits = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=0.5, u=0.1
        )

        # Need at least 2 tokens to reach cumulative 0.5 (each is 0.25)
        # Cutoff happens when cumsum first exceeds 0.5, which is at index 2
        assert result.num_candidates == 2

    def test_combined_top_k_and_top_p(self, selector: TokenSelector) -> None:
        """Both top_k and top_p applied together.

        top_k narrows the set first, then top_p further filters.
        """
        logits = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            dtype=np.float64,
        )

        result = selector.select(
            logits=logits, temperature=1.0, top_k=5, top_p=0.8, u=0.5
        )

        # Top-k keeps 5, top-p may further reduce
        assert result.num_candidates <= 5


class TestTemperatureEffect:
    """Temperature scaling changes the distribution shape."""

    def test_low_temperature_sharpens(self, selector: TokenSelector) -> None:
        """Low temperature makes the distribution more peaked.

        At very low T, the most probable token should dominate for almost
        any u.
        """
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=0.01, top_k=0, top_p=1.0, u=0.5
        )

        # Token 2 (highest logit) should be selected even at u=0.5
        assert result.token_id == 2
        assert result.token_rank == 0

    def test_high_temperature_flattens(self, selector: TokenSelector) -> None:
        """High temperature makes probabilities more uniform.

        At very high T, logits/T → 0 → probabilities approach uniform.
        """
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=100.0, top_k=0, top_p=1.0, u=0.5
        )

        # With near-uniform probs, prob of top token should be close to 1/3
        assert result.token_prob == pytest.approx(1.0 / 3, abs=0.05)


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_all_logits_identical(self, selector: TokenSelector) -> None:
        """All logits equal → uniform distribution → no crash."""
        vocab_size = 100
        logits = np.full(vocab_size, 3.14, dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5
        )

        assert 0 <= result.token_id < vocab_size
        assert result.num_candidates == vocab_size
        assert result.token_prob == pytest.approx(1.0 / vocab_size, rel=0.01)

    def test_all_inf_except_one(self, selector: TokenSelector) -> None:
        """All logits -inf except one → that token must be selected."""
        logits = np.full(50, -np.inf, dtype=np.float64)
        logits[17] = 0.0  # Only survivor

        # Should select token 17 for any u
        for u in [0.01, 0.5, 0.99]:
            result = selector.select(
                logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=u
            )
            assert result.token_id == 17
            assert result.token_prob == pytest.approx(1.0)
            assert result.num_candidates == 1

    def test_empty_logits_raises(self, selector: TokenSelector) -> None:
        """Empty logits array raises TokenSelectionError."""
        with pytest.raises(TokenSelectionError, match="empty"):
            selector.select(
                logits=np.array([], dtype=np.float64),
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                u=0.5,
            )

    def test_zero_temperature_raises(self, selector: TokenSelector) -> None:
        """Temperature = 0 raises TokenSelectionError."""
        with pytest.raises(TokenSelectionError, match="[Tt]emperature"):
            selector.select(
                logits=np.array([1.0, 2.0]),
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                u=0.5,
            )

    def test_negative_temperature_raises(
        self, selector: TokenSelector
    ) -> None:
        """Negative temperature raises TokenSelectionError."""
        with pytest.raises(TokenSelectionError, match="[Tt]emperature"):
            selector.select(
                logits=np.array([1.0, 2.0]),
                temperature=-0.5,
                top_k=0,
                top_p=1.0,
                u=0.5,
            )

    def test_single_token_vocab(self, selector: TokenSelector) -> None:
        """Vocab size of 1 → always selects token 0."""
        logits = np.array([5.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5
        )

        assert result.token_id == 0
        assert result.token_rank == 0
        assert result.token_prob == pytest.approx(1.0)
        assert result.num_candidates == 1

    def test_very_large_vocab(self, selector: TokenSelector) -> None:
        """Reasonably large vocab should work without issues."""
        vocab_size = 100_000
        rng = np.random.default_rng(42)
        logits = rng.normal(size=vocab_size)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=50, top_p=0.9, u=0.5
        )

        assert 0 <= result.token_id < vocab_size
        assert result.num_candidates <= 50


class TestCDFProperties:
    """Verify CDF ordering and selection properties."""

    def test_rank_zero_is_most_probable(
        self, selector: TokenSelector
    ) -> None:
        """Token at rank 0 should be the most probable in the candidate set."""
        logits = np.array([1.0, 5.0, 2.0, 3.0, 4.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=0.001
        )

        # Token 1 (logit = 5.0) is most probable
        assert result.token_id == 1
        assert result.token_rank == 0

    def test_increasing_u_increases_rank(
        self, selector: TokenSelector
    ) -> None:
        """Higher u values should select tokens with higher rank (less probable).

        This is the key property: u near 0 → high-prob token,
        u near 1 → low-prob token.
        """
        logits = np.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=np.float64)

        ranks = []
        for u in [0.001, 0.25, 0.5, 0.75, 0.999]:
            result = selector.select(
                logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=u
            )
            ranks.append(result.token_rank)

        # Ranks should be non-decreasing with u
        for i in range(1, len(ranks)):
            assert ranks[i] >= ranks[i - 1], (
                f"Rank should increase with u: "
                f"u[{i}] gave rank {ranks[i]} < rank {ranks[i-1]}"
            )

    def test_token_prob_sums_to_one_in_candidates(
        self, selector: TokenSelector
    ) -> None:
        """Within the candidate set, probabilities should sum to ~1.0.

        We can't directly check the sum from SelectionResult, but we can
        verify that a sweep of u from 0 to 1 covers all candidates.
        """
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        # Collect all unique tokens selected across many u values
        selected_tokens = set()
        for u in np.linspace(0.001, 0.999, 100):
            result = selector.select(
                logits=logits, temperature=1.0, top_k=0, top_p=1.0, u=u
            )
            selected_tokens.add(result.token_id)

        # All 5 tokens should be reachable
        assert len(selected_tokens) == 5

    def test_diagnostics_present(self, selector: TokenSelector) -> None:
        """Selection diagnostics should contain expected keys."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        result = selector.select(
            logits=logits, temperature=0.8, top_k=2, top_p=0.9, u=0.5
        )

        assert "temperature_applied" in result.diagnostics
        assert result.diagnostics["temperature_applied"] == 0.8
        assert "top_k_applied" in result.diagnostics
        assert "top_p_count" in result.diagnostics
        assert "u_value" in result.diagnostics
        assert result.diagnostics["u_value"] == 0.5


class TestSelectionResultImmutability:
    """SelectionResult is a frozen dataclass — immutable."""

    def test_frozen(self) -> None:
        result = SelectionResult(
            token_id=5, token_rank=0, token_prob=0.9, num_candidates=10
        )
        with pytest.raises(AttributeError):
            result.token_id = 10  # type: ignore[misc]

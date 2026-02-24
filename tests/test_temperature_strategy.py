"""Tests for qc_sampler.temperature_strategy.

Covers:
- compute_shannon_entropy helper: stable computation, edge cases
- FixedTemperatureStrategy: always returns fixed_temperature, computes entropy
- EDTTemperatureStrategy:
  - Uniform logits → H_norm ≈ 1.0 → T ≈ edt_base_temp
  - One-hot logits → H_norm ≈ 0.0 → T clamped to edt_min_temp
  - Monotonicity: higher entropy → higher temperature
  - Min/max clamping works correctly
  - Exponent sensitivity: θ < 1 (concave), θ > 1 (convex)
- TemperatureResult is immutable (frozen dataclass)
- EDT constructor rejects invalid vocab_size
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qc_sampler.config import QCSamplingConfig
from qc_sampler.temperature_strategy import (
    EDTTemperatureStrategy,
    FixedTemperatureStrategy,
    TemperatureResult,
    compute_shannon_entropy,
)


class TestComputeShannonEntropy:
    """Verify the module-level Shannon entropy helper."""

    def test_uniform_distribution(self) -> None:
        """Uniform logits → maximum entropy = ln(vocab_size).

        For 100 equal logits, H = ln(100) ≈ 4.605.
        """
        vocab_size = 100
        logits = np.zeros(vocab_size, dtype=np.float64)
        entropy = compute_shannon_entropy(logits)
        expected = math.log(vocab_size)
        assert entropy == pytest.approx(expected, rel=1e-6)

    def test_one_hot_distribution(self) -> None:
        """One token dominates entirely → H ≈ 0.

        One logit very high, rest very low → entropy near zero.
        """
        logits = np.full(1000, -1e6, dtype=np.float64)
        logits[42] = 0.0  # Only this token has any probability mass
        entropy = compute_shannon_entropy(logits)
        assert entropy == pytest.approx(0.0, abs=1e-4)

    def test_two_equal_tokens(self) -> None:
        """Two equally likely tokens → H = ln(2) ≈ 0.693."""
        logits = np.array([0.0, 0.0, -1e6, -1e6], dtype=np.float64)
        entropy = compute_shannon_entropy(logits)
        assert entropy == pytest.approx(math.log(2), abs=0.01)

    def test_all_negative_inf(self) -> None:
        """All -inf logits → degenerate distribution → H = 0.0."""
        logits = np.full(100, -np.inf, dtype=np.float64)
        entropy = compute_shannon_entropy(logits)
        assert entropy == 0.0

    def test_single_logit(self) -> None:
        """Single logit → prob = 1.0 → H = 0.0."""
        logits = np.array([5.0], dtype=np.float64)
        entropy = compute_shannon_entropy(logits)
        assert entropy == pytest.approx(0.0, abs=1e-10)

    def test_numerical_stability_large_logits(self) -> None:
        """Very large logits should not overflow thanks to shift-by-max."""
        logits = np.array([1000.0, 1000.0, 999.0], dtype=np.float64)
        entropy = compute_shannon_entropy(logits)
        # Should not be NaN or inf
        assert np.isfinite(entropy)
        assert entropy > 0.0


class TestFixedTemperatureStrategy:
    """FixedTemperatureStrategy always returns config.fixed_temperature."""

    def test_returns_fixed_temperature(self) -> None:
        """Output temperature equals config.fixed_temperature exactly."""
        config = QCSamplingConfig(fixed_temperature=0.7)
        strategy = FixedTemperatureStrategy()
        logits = np.random.default_rng(0).normal(size=1000)

        result = strategy.compute_temperature(logits, config)

        assert result.temperature == 0.7
        assert isinstance(result, TemperatureResult)

    def test_different_fixed_temperatures(self) -> None:
        """Different config values produce different temperatures."""
        strategy = FixedTemperatureStrategy()
        logits = np.zeros(100, dtype=np.float64)

        for temp in [0.1, 0.5, 1.0, 2.0]:
            config = QCSamplingConfig(fixed_temperature=temp)
            result = strategy.compute_temperature(logits, config)
            assert result.temperature == temp

    def test_computes_shannon_entropy(self) -> None:
        """Even though fixed strategy doesn't use entropy, it returns it."""
        config = QCSamplingConfig(fixed_temperature=1.0)
        strategy = FixedTemperatureStrategy()
        # Uniform logits → known entropy
        vocab_size = 50
        logits = np.zeros(vocab_size, dtype=np.float64)

        result = strategy.compute_temperature(logits, config)

        expected_entropy = math.log(vocab_size)
        assert result.shannon_entropy == pytest.approx(expected_entropy, rel=1e-6)

    def test_temperature_independent_of_logits(self) -> None:
        """Fixed temperature doesn't change with logit distribution."""
        config = QCSamplingConfig(fixed_temperature=0.42)
        strategy = FixedTemperatureStrategy()

        # Very different logit distributions
        uniform_logits = np.zeros(100, dtype=np.float64)
        peaked_logits = np.full(100, -1e6, dtype=np.float64)
        peaked_logits[0] = 0.0

        r1 = strategy.compute_temperature(uniform_logits, config)
        r2 = strategy.compute_temperature(peaked_logits, config)

        assert r1.temperature == r2.temperature == 0.42

    def test_diagnostics_contains_strategy(self) -> None:
        """Diagnostics should identify the strategy name."""
        strategy = FixedTemperatureStrategy()
        result = strategy.compute_temperature(
            np.zeros(10), QCSamplingConfig()
        )
        assert result.diagnostics["strategy"] == "fixed"


class TestEDTTemperatureStrategy:
    """EDTTemperatureStrategy: entropy-based dynamic temperature."""

    def test_uniform_logits_gives_base_temp(self) -> None:
        """Perfectly uniform logits → H_norm ≈ 1.0 → T ≈ base_temp × 1^θ.

        For any exponent θ, 1^θ = 1, so T = base_temp (before clamping).
        """
        vocab_size = 1000
        config = QCSamplingConfig(
            edt_base_temp=0.8,
            edt_exponent=0.5,
            edt_min_temp=0.1,
            edt_max_temp=2.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)
        logits = np.zeros(vocab_size, dtype=np.float64)

        result = strategy.compute_temperature(logits, config)

        assert result.temperature == pytest.approx(0.8, abs=0.01)
        assert result.diagnostics["h_norm"] == pytest.approx(1.0, abs=0.01)

    def test_one_hot_logits_gives_min_temp(self) -> None:
        """One-hot logits → H_norm ≈ 0 → T ≈ 0 → clamped to min_temp."""
        vocab_size = 1000
        config = QCSamplingConfig(
            edt_base_temp=0.8,
            edt_exponent=0.5,
            edt_min_temp=0.15,
            edt_max_temp=2.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)
        logits = np.full(vocab_size, -1e6, dtype=np.float64)
        logits[0] = 0.0

        result = strategy.compute_temperature(logits, config)

        assert result.temperature == pytest.approx(config.edt_min_temp)
        assert result.diagnostics["h_norm"] == pytest.approx(0.0, abs=0.01)

    def test_monotonicity_higher_entropy_higher_temp(self) -> None:
        """Higher entropy distributions must produce higher or equal temperature.

        We create distributions with increasing entropy by having more
        tokens share the probability mass.
        """
        vocab_size = 10000
        config = QCSamplingConfig(
            edt_base_temp=1.0,
            edt_exponent=0.5,
            edt_min_temp=0.01,
            edt_max_temp=5.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)

        temperatures = []
        # Create distributions with 2, 10, 100, 1000 active tokens
        for n_active in [2, 10, 100, 1000]:
            logits = np.full(vocab_size, -1e6, dtype=np.float64)
            logits[:n_active] = 0.0  # n_active tokens equally likely
            result = strategy.compute_temperature(logits, config)
            temperatures.append(result.temperature)

        # Each temperature should be >= the previous one
        for i in range(1, len(temperatures)):
            assert temperatures[i] >= temperatures[i - 1], (
                f"Monotonicity violated: T[{i}]={temperatures[i]} < "
                f"T[{i-1}]={temperatures[i-1]}"
            )

    def test_min_clamp(self) -> None:
        """Temperature should never go below edt_min_temp."""
        vocab_size = 1000
        config = QCSamplingConfig(
            edt_base_temp=0.5,
            edt_exponent=2.0,  # High exponent → T stays low
            edt_min_temp=0.3,
            edt_max_temp=2.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)

        # Low-entropy distribution → raw T would be near 0
        logits = np.full(vocab_size, -1e6, dtype=np.float64)
        logits[0] = 0.0

        result = strategy.compute_temperature(logits, config)
        assert result.temperature >= config.edt_min_temp

    def test_max_clamp(self) -> None:
        """Temperature should never exceed edt_max_temp."""
        vocab_size = 100
        config = QCSamplingConfig(
            edt_base_temp=5.0,  # Very high base → raw T > max
            edt_exponent=0.5,
            edt_min_temp=0.1,
            edt_max_temp=1.5,
        )
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)
        logits = np.zeros(vocab_size, dtype=np.float64)  # Uniform → H_norm ≈ 1

        result = strategy.compute_temperature(logits, config)
        assert result.temperature <= config.edt_max_temp
        assert result.temperature == pytest.approx(1.5)

    def test_exponent_less_than_one_concave(self) -> None:
        """With θ < 1 (concave), temperature rises quickly from H_norm = 0.

        At H_norm = 0.25, T = base × 0.25^0.3 ≈ base × 0.674.
        At H_norm = 0.25, T = base × 0.25^2.0 ≈ base × 0.0625.
        So θ < 1 should give higher T at moderate entropy.
        """
        vocab_size = 1000
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)

        # Create a moderate-entropy distribution
        logits = np.full(vocab_size, -1e6, dtype=np.float64)
        # ~25% of max entropy: active_count ≈ vocab_size^0.25 ≈ 5.6
        # Use a rough approach: set 6 tokens active
        logits[:6] = 0.0

        config_concave = QCSamplingConfig(
            edt_base_temp=1.0, edt_exponent=0.3,
            edt_min_temp=0.01, edt_max_temp=5.0,
        )
        config_convex = QCSamplingConfig(
            edt_base_temp=1.0, edt_exponent=2.0,
            edt_min_temp=0.01, edt_max_temp=5.0,
        )

        r_concave = strategy.compute_temperature(logits, config_concave)
        r_convex = strategy.compute_temperature(logits, config_convex)

        # At moderate entropy, concave (θ < 1) should give higher T
        assert r_concave.temperature > r_convex.temperature

    def test_exponent_one_linear(self) -> None:
        """With θ = 1.0, T = base_temp × H_norm (linear)."""
        vocab_size = 1000
        config = QCSamplingConfig(
            edt_base_temp=2.0,
            edt_exponent=1.0,
            edt_min_temp=0.01,
            edt_max_temp=5.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=vocab_size)
        logits = np.zeros(vocab_size, dtype=np.float64)

        result = strategy.compute_temperature(logits, config)

        # Uniform → H_norm ≈ 1.0 → T = 2.0 × 1.0 = 2.0
        assert result.temperature == pytest.approx(2.0, abs=0.01)

    def test_diagnostics_contains_expected_keys(self) -> None:
        """EDT diagnostics should contain h_norm, raw_temp, max_entropy, vocab_size."""
        strategy = EDTTemperatureStrategy(vocab_size=100)
        result = strategy.compute_temperature(
            np.zeros(100), QCSamplingConfig()
        )
        assert "h_norm" in result.diagnostics
        assert "raw_temp" in result.diagnostics
        assert "max_entropy" in result.diagnostics
        assert "vocab_size" in result.diagnostics
        assert result.diagnostics["strategy"] == "edt"

    def test_entropy_always_returned(self) -> None:
        """Shannon entropy is always returned regardless of strategy."""
        strategy = EDTTemperatureStrategy(vocab_size=100)
        logits = np.zeros(100, dtype=np.float64)
        result = strategy.compute_temperature(logits, QCSamplingConfig())

        expected_entropy = math.log(100)
        assert result.shannon_entropy == pytest.approx(expected_entropy, rel=1e-6)

    def test_invalid_vocab_size_raises(self) -> None:
        """vocab_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="vocab_size"):
            EDTTemperatureStrategy(vocab_size=0)
        with pytest.raises(ValueError, match="vocab_size"):
            EDTTemperatureStrategy(vocab_size=-5)

    def test_vocab_size_one(self) -> None:
        """vocab_size = 1: degenerate case.  H_norm = 0, T = min_temp."""
        config = QCSamplingConfig(
            edt_base_temp=1.0, edt_min_temp=0.2, edt_max_temp=2.0
        )
        strategy = EDTTemperatureStrategy(vocab_size=1)
        logits = np.array([5.0], dtype=np.float64)

        result = strategy.compute_temperature(logits, config)

        # H = 0 (single token), H_norm = 0/0 → guarded to 0.0
        # T = base × 0^θ = 0 → clamped to min_temp
        assert result.temperature == pytest.approx(config.edt_min_temp)

    def test_per_request_config_override(self) -> None:
        """EDT respects per-request config overrides correctly.

        Two calls with different configs should produce different temperatures
        for the same logits.
        """
        strategy = EDTTemperatureStrategy(vocab_size=1000)
        logits = np.zeros(1000, dtype=np.float64)

        config_low = QCSamplingConfig(
            edt_base_temp=0.3, edt_exponent=0.5,
            edt_min_temp=0.1, edt_max_temp=2.0,
        )
        config_high = QCSamplingConfig(
            edt_base_temp=1.5, edt_exponent=0.5,
            edt_min_temp=0.1, edt_max_temp=2.0,
        )

        r_low = strategy.compute_temperature(logits, config_low)
        r_high = strategy.compute_temperature(logits, config_high)

        assert r_low.temperature < r_high.temperature


class TestTemperatureResultImmutability:
    """TemperatureResult is a frozen dataclass — immutable."""

    def test_frozen(self) -> None:
        result = TemperatureResult(temperature=0.7, shannon_entropy=3.0)
        with pytest.raises(AttributeError):
            result.temperature = 1.0  # type: ignore[misc]

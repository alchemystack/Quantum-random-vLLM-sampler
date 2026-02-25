"""Tests for the EDTTemperatureStrategy."""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.edt import EDTTemperatureStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config for EDT tests."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def strategy() -> EDTTemperatureStrategy:
    """EDT strategy with vocab_size=1000."""
    return EDTTemperatureStrategy(vocab_size=1000)


class TestEDTTemperatureStrategy:
    """Tests for EDTTemperatureStrategy."""

    def test_monotonicity_with_entropy(self, config: QRSamplerConfig) -> None:
        """Higher entropy logits should produce higher temperature.

        Creates distributions with increasing entropy and verifies
        that EDT temperature increases monotonically.
        """
        strategy = EDTTemperatureStrategy(vocab_size=100)
        # Peaked → moderate → uniform logits.
        peaked = np.array([100.0] + [-100.0] * 99)
        moderate = np.arange(100, dtype=np.float64)
        uniform = np.zeros(100)

        t_peaked = strategy.compute_temperature(peaked, config).temperature
        t_moderate = strategy.compute_temperature(moderate, config).temperature
        t_uniform = strategy.compute_temperature(uniform, config).temperature

        assert t_peaked < t_moderate < t_uniform

    def test_clamping_lower_bound(self, config: QRSamplerConfig) -> None:
        """Temperature should not go below edt_min_temp."""
        strategy = EDTTemperatureStrategy(vocab_size=100)
        # Extremely peaked: H_norm ≈ 0, so T ≈ 0, clamped to min.
        peaked = np.array([1000.0] + [-1000.0] * 99)
        result = strategy.compute_temperature(peaked, config)
        assert result.temperature >= config.edt_min_temp

    def test_clamping_upper_bound(self) -> None:
        """Temperature should not exceed edt_max_temp."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            edt_base_temp=100.0,
            edt_max_temp=2.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=10)
        uniform = np.zeros(10)
        result = strategy.compute_temperature(uniform, config)
        assert result.temperature <= config.edt_max_temp

    def test_exponent_effect_concave(self) -> None:
        """exponent < 1 (concave): temperature rises quickly with entropy."""
        config_concave = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            edt_exponent=0.3,
        )
        config_convex = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            edt_exponent=2.0,
        )
        strategy = EDTTemperatureStrategy(vocab_size=100)
        # Moderate entropy distribution.
        logits = np.arange(100, dtype=np.float64)

        t_concave = strategy.compute_temperature(logits, config_concave).temperature
        t_convex = strategy.compute_temperature(logits, config_convex).temperature

        # With moderate H_norm (between 0 and 1), concave should give higher temp.
        assert t_concave > t_convex

    def test_always_computes_shannon_entropy(
        self, strategy: EDTTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Shannon entropy should always be in the result."""
        logits = np.array([5.0, 4.0, 3.0] + [0.0] * 997)
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy >= 0.0

    def test_diagnostics_keys(
        self, strategy: EDTTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Diagnostics should contain EDT-specific fields."""
        logits = np.array([1.0] * 1000)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["strategy"] == "edt"
        assert "h_norm" in result.diagnostics
        assert "pre_clamp_temp" in result.diagnostics
        assert "vocab_size" in result.diagnostics

    def test_h_norm_range(self, strategy: EDTTemperatureStrategy, config: QRSamplerConfig) -> None:
        """h_norm should be in [0, 1]."""
        logits = np.arange(1000, dtype=np.float64)
        result = strategy.compute_temperature(logits, config)
        assert 0.0 <= result.diagnostics["h_norm"] <= 1.0

    def test_vocab_size_too_small(self) -> None:
        """vocab_size < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be >= 2"):
            EDTTemperatureStrategy(vocab_size=1)

    def test_result_is_frozen(
        self, strategy: EDTTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """TemperatureResult should be immutable."""
        logits = np.array([1.0] * 1000)
        result = strategy.compute_temperature(logits, config)
        with pytest.raises(AttributeError):
            result.temperature = 99.0  # type: ignore[misc]

    def test_registered(self) -> None:
        """EDTTemperatureStrategy should be in the registry as 'edt'."""
        klass = TemperatureStrategyRegistry.get("edt")
        assert klass is EDTTemperatureStrategy

    def test_build_with_vocab_size(self, config: QRSamplerConfig) -> None:
        """Registry build() should pass vocab_size to EDT."""
        edt_config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temperature_strategy="edt",
        )
        strategy = TemperatureStrategyRegistry.build(edt_config, vocab_size=500)
        assert isinstance(strategy, EDTTemperatureStrategy)
        assert strategy._vocab_size == 500

    def test_build_fixed_ignores_vocab_size(self) -> None:
        """Registry build() should work for 'fixed' which doesn't need vocab_size."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temperature_strategy="fixed",
        )
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=500)
        # FixedTemperatureStrategy doesn't have _vocab_size attribute.
        assert not hasattr(strategy, "_vocab_size")

    def test_uniform_gives_max_h_norm(self) -> None:
        """Uniform logits should give h_norm ≈ 1.0."""
        config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        strategy = EDTTemperatureStrategy(vocab_size=10)
        uniform = np.zeros(10)
        result = strategy.compute_temperature(uniform, config)
        assert abs(result.diagnostics["h_norm"] - 1.0) < 1e-6

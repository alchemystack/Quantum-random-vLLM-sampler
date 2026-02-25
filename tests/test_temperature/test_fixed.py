"""Tests for the FixedTemperatureStrategy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.base import TemperatureResult, compute_shannon_entropy
from qr_sampler.temperature.fixed import FixedTemperatureStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config for temperature tests."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def strategy() -> FixedTemperatureStrategy:
    """Default FixedTemperatureStrategy."""
    return FixedTemperatureStrategy()


class TestFixedTemperatureStrategy:
    """Tests for FixedTemperatureStrategy."""

    def test_returns_fixed_temperature(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Should return config.fixed_temperature regardless of logits."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == config.fixed_temperature

    def test_returns_fixed_for_uniform_logits(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Should return fixed temperature even for uniform distribution."""
        logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == config.fixed_temperature

    def test_computes_shannon_entropy(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Should always compute and include Shannon entropy."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy >= 0.0

    def test_shannon_entropy_uniform(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Uniform distribution should have max entropy = ln(N)."""
        n = 10
        logits = np.zeros(n)
        result = strategy.compute_temperature(logits, config)
        expected = math.log(n)
        assert abs(result.shannon_entropy - expected) < 1e-6

    def test_shannon_entropy_peaked(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Peaked distribution should have low entropy."""
        logits = np.array([100.0, -100.0, -100.0, -100.0])
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy < 0.01

    def test_custom_temperature(self) -> None:
        """Per-request config with different temperature should be respected."""
        config = QRSamplerConfig(_env_file=None, fixed_temperature=1.5)  # type: ignore[call-arg]
        strategy = FixedTemperatureStrategy()
        logits = np.array([1.0, 2.0, 3.0])
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == 1.5

    def test_diagnostics_contain_strategy(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """Diagnostics should identify the strategy."""
        logits = np.array([1.0, 2.0])
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["strategy"] == "fixed"

    def test_result_is_frozen(
        self, strategy: FixedTemperatureStrategy, config: QRSamplerConfig
    ) -> None:
        """TemperatureResult should be immutable."""
        logits = np.array([1.0, 2.0])
        result = strategy.compute_temperature(logits, config)
        with pytest.raises(AttributeError):
            result.temperature = 99.0  # type: ignore[misc]

    def test_registered(self) -> None:
        """FixedTemperatureStrategy should be in the registry as 'fixed'."""
        klass = TemperatureStrategyRegistry.get("fixed")
        assert klass is FixedTemperatureStrategy


class TestTemperatureStrategyRegistry:
    """Tests for the TemperatureStrategyRegistry."""

    def test_unknown_strategy_raises(self) -> None:
        """Looking up an unregistered name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown temperature strategy"):
            TemperatureStrategyRegistry.get("nonexistent_strategy")

    def test_list_registered(self) -> None:
        """list_registered should return known strategy names."""
        names = TemperatureStrategyRegistry.list_registered()
        assert "fixed" in names
        assert "edt" in names
        assert names == sorted(names)


class TestComputeShannonEntropy:
    """Tests for the compute_shannon_entropy utility."""

    def test_uniform_distribution(self) -> None:
        """Uniform logits → H = ln(N)."""
        n = 8
        logits = np.zeros(n)
        h = compute_shannon_entropy(logits)
        assert abs(h - math.log(n)) < 1e-6

    def test_peaked_distribution(self) -> None:
        """One dominant logit → H ≈ 0."""
        logits = np.array([1000.0, -1000.0, -1000.0])
        h = compute_shannon_entropy(logits)
        assert h < 1e-6

    def test_two_equal_tokens(self) -> None:
        """Two equal logits → H = ln(2)."""
        logits = np.array([5.0, 5.0])
        h = compute_shannon_entropy(logits)
        assert abs(h - math.log(2)) < 1e-6

    def test_non_negative(self) -> None:
        """Shannon entropy should never be negative."""
        logits = np.array([1.0, -50.0, 30.0, -2.0, 0.5])
        h = compute_shannon_entropy(logits)
        assert h >= 0.0

    def test_single_logit(self) -> None:
        """Single-element distribution has zero entropy."""
        logits = np.array([42.0])
        h = compute_shannon_entropy(logits)
        assert h == 0.0

    def test_result_is_frozen(self) -> None:
        """TemperatureResult should reject attribute mutation."""
        result = TemperatureResult(temperature=0.7, shannon_entropy=1.5, diagnostics={})
        with pytest.raises(AttributeError):
            result.temperature = 0.9  # type: ignore[misc]

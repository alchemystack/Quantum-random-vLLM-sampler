"""Tests for qc_sampler.signal_amplifier.

Covers:
- ZScoreMeanAmplifier with known constant byte values (128, 127)
- Verification that SEM is derived from population_std and sample count
- Diagnostics dict contains expected keys
- Empty input raises SignalAmplificationError
- Clamp epsilon prevents u from reaching exactly 0 or 1
- Single-byte input (edge case)
"""

from __future__ import annotations

import math

import pytest

from qc_sampler.config import QCSamplingConfig
from qc_sampler.exceptions import SignalAmplificationError
from qc_sampler.signal_amplifier import AmplificationResult, ZScoreMeanAmplifier


class TestZScoreMeanAmplifierKnownValues:
    """Verify amplifier output against hand-calculated expected values.

    These tests use constant byte arrays where the sample mean is known
    exactly, allowing precise verification of the z-score → CDF pipeline.
    """

    def _make_amplifier(self, **kwargs: object) -> ZScoreMeanAmplifier:
        """Create an amplifier with optional config overrides."""
        config = QCSamplingConfig(**kwargs)  # type: ignore[arg-type]
        return ZScoreMeanAmplifier(config)

    def test_all_bytes_128(self) -> None:
        """20,480 bytes all equal to 128 → known u ≈ 0.834.

        Calculation:
            M = 128.0
            SEM = 73.6116... / √20480 ≈ 0.5143
            z = (128.0 - 127.5) / 0.5143 ≈ 0.972
            u = Φ(0.972) ≈ 0.8345
        """
        sample_count = 20480
        amplifier = self._make_amplifier()
        raw_bytes = bytes([128]) * sample_count

        result = amplifier.amplify(raw_bytes)

        assert isinstance(result, AmplificationResult)
        assert result.u == pytest.approx(0.8345, abs=0.002)
        assert result.diagnostics["sample_mean"] == 128.0
        assert result.diagnostics["z_score"] == pytest.approx(0.972, abs=0.01)

    def test_all_bytes_127(self) -> None:
        """20,480 bytes all equal to 127 → known u ≈ 0.166.

        Calculation:
            M = 127.0
            SEM = 73.6116... / √20480 ≈ 0.5143
            z = (127.0 - 127.5) / 0.5143 ≈ -0.972
            u = Φ(-0.972) ≈ 0.1655
        """
        sample_count = 20480
        amplifier = self._make_amplifier()
        raw_bytes = bytes([127]) * sample_count

        result = amplifier.amplify(raw_bytes)

        assert result.u == pytest.approx(0.1655, abs=0.002)
        assert result.diagnostics["sample_mean"] == 127.0
        assert result.diagnostics["z_score"] == pytest.approx(-0.972, abs=0.01)

    def test_mean_bytes_produce_u_near_half(self) -> None:
        """Bytes that average to population_mean → u ≈ 0.5.

        Using alternating 127 and 128 gives mean = 127.5 exactly.
        """
        sample_count = 20480
        amplifier = self._make_amplifier()
        # Alternating 127 and 128 → mean = 127.5
        raw_bytes = bytes([127, 128]) * (sample_count // 2)

        result = amplifier.amplify(raw_bytes)

        assert result.u == pytest.approx(0.5, abs=0.001)
        assert result.diagnostics["z_score"] == pytest.approx(0.0, abs=0.001)


class TestSEMDerivation:
    """Verify that SEM is derived from population_std and sample count.

    The spec explicitly states: SEM = population_std / sqrt(sample_count).
    It must NOT be a separate config field, and must NOT be hardcoded.
    """

    def test_sem_changes_with_sample_count(self) -> None:
        """Changing the number of input bytes changes the SEM.

        If SEM were hardcoded to ~0.5143 (for N=20480), this test
        would fail because halving N would not double SEM.
        """
        config = QCSamplingConfig()
        amplifier = ZScoreMeanAmplifier(config)

        # Mean = 128 for both, but different sample counts.
        full = amplifier.amplify(bytes([128]) * 20480)
        half = amplifier.amplify(bytes([128]) * 10240)

        # SEM for half should be √2 ≈ 1.414× larger.
        ratio = half.diagnostics["sem"] / full.diagnostics["sem"]
        assert ratio == pytest.approx(math.sqrt(2), abs=0.001)

    def test_sem_changes_with_population_std(self) -> None:
        """Changing population_std in config changes the SEM.

        If SEM were derived from a hardcoded constant instead of
        config.population_std, this test would fail.
        """
        config_default = QCSamplingConfig()
        config_custom = QCSamplingConfig(population_std=100.0)

        amp_default = ZScoreMeanAmplifier(config_default)
        amp_custom = ZScoreMeanAmplifier(config_custom)

        raw = bytes([128]) * 20480

        result_default = amp_default.amplify(raw)
        result_custom = amp_custom.amplify(raw)

        # SEM should scale linearly with population_std.
        ratio = result_custom.diagnostics["sem"] / result_default.diagnostics["sem"]
        expected_ratio = 100.0 / config_default.population_std
        assert ratio == pytest.approx(expected_ratio, abs=0.001)

    def test_sem_formula_matches_expected(self) -> None:
        """SEM = population_std / sqrt(N) exactly."""
        config = QCSamplingConfig()
        amplifier = ZScoreMeanAmplifier(config)

        n = 20480
        result = amplifier.amplify(bytes([128]) * n)

        expected_sem = config.population_std / math.sqrt(n)
        assert result.diagnostics["sem"] == pytest.approx(expected_sem)


class TestDiagnostics:
    """Verify the diagnostics dict contains all expected keys."""

    def test_diagnostics_keys(self) -> None:
        """Result diagnostics must contain sample_mean, z_score, sem, sample_count."""
        amplifier = ZScoreMeanAmplifier(QCSamplingConfig())
        result = amplifier.amplify(bytes([128]) * 100)

        expected_keys = {"sample_mean", "z_score", "sem", "sample_count"}
        assert expected_keys.issubset(result.diagnostics.keys())

    def test_diagnostics_sample_count_matches_input(self) -> None:
        """Diagnostics sample_count must equal the actual number of bytes."""
        amplifier = ZScoreMeanAmplifier(QCSamplingConfig())
        n = 5000
        result = amplifier.amplify(bytes([128]) * n)
        assert result.diagnostics["sample_count"] == n


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_bytes_raises(self) -> None:
        """Empty input must raise SignalAmplificationError."""
        amplifier = ZScoreMeanAmplifier(QCSamplingConfig())
        with pytest.raises(SignalAmplificationError, match="empty"):
            amplifier.amplify(b"")

    def test_single_byte(self) -> None:
        """Single byte input should still produce a valid result.

        With N=1, SEM = population_std (very large), so z is small
        and u stays near 0.5.
        """
        amplifier = ZScoreMeanAmplifier(QCSamplingConfig())
        result = amplifier.amplify(bytes([255]))

        assert 0.0 < result.u < 1.0
        assert result.diagnostics["sample_mean"] == 255.0
        assert result.diagnostics["sample_count"] == 1

    def test_clamp_prevents_exact_zero(self) -> None:
        """u must never be exactly 0.0 — clamped to epsilon.

        All bytes = 0 → mean = 0 → large negative z → Φ(z) ≈ 0 → clamped.
        """
        config = QCSamplingConfig(uniform_clamp_epsilon=1e-10)
        amplifier = ZScoreMeanAmplifier(config)
        result = amplifier.amplify(bytes([0]) * 20480)

        assert result.u >= config.uniform_clamp_epsilon
        assert result.u > 0.0

    def test_clamp_prevents_exact_one(self) -> None:
        """u must never be exactly 1.0 — clamped to 1 - epsilon.

        All bytes = 255 → mean = 255 → large positive z → Φ(z) ≈ 1 → clamped.
        """
        config = QCSamplingConfig(uniform_clamp_epsilon=1e-10)
        amplifier = ZScoreMeanAmplifier(config)
        result = amplifier.amplify(bytes([255]) * 20480)

        assert result.u <= 1.0 - config.uniform_clamp_epsilon
        assert result.u < 1.0

    def test_result_is_frozen(self) -> None:
        """AmplificationResult is a frozen dataclass — immutable."""
        amplifier = ZScoreMeanAmplifier(QCSamplingConfig())
        result = amplifier.amplify(bytes([128]) * 100)

        with pytest.raises(AttributeError):
            result.u = 0.5  # type: ignore[misc]

    def test_custom_population_mean(self) -> None:
        """Changing population_mean shifts the z-score and u-value.

        If population_mean == 128 and all bytes are 128, z = 0, u = 0.5.
        """
        config = QCSamplingConfig(population_mean=128.0)
        amplifier = ZScoreMeanAmplifier(config)
        result = amplifier.amplify(bytes([128]) * 20480)

        assert result.u == pytest.approx(0.5, abs=0.001)
        assert result.diagnostics["z_score"] == pytest.approx(0.0, abs=0.001)

    def test_custom_clamp_epsilon(self) -> None:
        """Custom clamp epsilon is respected."""
        config = QCSamplingConfig(uniform_clamp_epsilon=0.01)
        amplifier = ZScoreMeanAmplifier(config)

        # All zeros → u near 0 → clamped to 0.01
        result = amplifier.amplify(bytes([0]) * 20480)
        assert result.u == pytest.approx(0.01)

        # All 255 → u near 1 → clamped to 0.99
        result = amplifier.amplify(bytes([255]) * 20480)
        assert result.u == pytest.approx(0.99)

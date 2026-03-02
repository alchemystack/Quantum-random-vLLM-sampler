"""Tests for the ECDFAmplifier and ECDF-specific registry behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.ecdf import ECDFAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.exceptions import (
    EntropyUnavailableError,
    SignalAmplificationError,
)


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config for ECDF amplification tests."""
    return QRSamplerConfig(_env_file=None, signal_amplifier_type="ecdf")  # type: ignore[call-arg]


@pytest.fixture()
def amplifier(config: QRSamplerConfig) -> ECDFAmplifier:
    """Uncalibrated ECDFAmplifier."""
    return ECDFAmplifier(config)


@pytest.fixture()
def calibrated_amplifier(config: QRSamplerConfig) -> ECDFAmplifier:
    """ECDFAmplifier calibrated with a balanced MockUniformSource."""
    amp = ECDFAmplifier(config)
    source = MockUniformSource()
    amp.calibrate(source, config)
    return amp


class TestECDFAmplifier:
    """Tests for ECDFAmplifier core functionality."""

    def test_calibrate_and_amplify_balanced(self, config: QRSamplerConfig) -> None:
        """Calibrated amplifier with balanced source should produce u ≈ 0.5."""
        amp = ECDFAmplifier(config)
        cal_source = MockUniformSource(seed=42)
        amp.calibrate(cal_source, config)

        test_source = MockUniformSource(seed=99)
        u_values = []
        for _ in range(50):
            raw = test_source.get_random_bytes(config.sample_count)
            u_values.append(amp.amplify(raw).u)
        u_mean = sum(u_values) / len(u_values)
        assert abs(u_mean - 0.5) < 0.15

    def test_amplify_uncalibrated_raises(self, amplifier: ECDFAmplifier) -> None:
        """Amplifying without calibration should raise SignalAmplificationError."""
        with pytest.raises(SignalAmplificationError, match="not been calibrated"):
            amplifier.amplify(bytes([128] * 100))

    def test_amplify_empty_bytes_raises(self, calibrated_amplifier: ECDFAmplifier) -> None:
        """Empty input should raise SignalAmplificationError."""
        with pytest.raises(SignalAmplificationError, match="empty"):
            calibrated_amplifier.amplify(b"")

    def test_amplify_single_byte(self, calibrated_amplifier: ECDFAmplifier) -> None:
        """Single byte should produce a valid AmplificationResult."""
        result = calibrated_amplifier.amplify(b"\x80")
        assert isinstance(result, AmplificationResult)
        assert 0.0 < result.u < 1.0

    def test_u_clamped_within_bounds(self, calibrated_amplifier: ECDFAmplifier) -> None:
        """u should never be exactly 0.0 or 1.0 due to epsilon clamping."""
        for val in [0, 128, 255]:
            raw = bytes([val] * 1000)
            result = calibrated_amplifier.amplify(raw)
            assert result.u > 0.0
            assert result.u < 1.0

    def test_diagnostics_keys(self, calibrated_amplifier: ECDFAmplifier) -> None:
        """Diagnostics should contain expected keys."""
        result = calibrated_amplifier.amplify(bytes([128] * 100))
        assert "sample_mean" in result.diagnostics
        assert "ecdf_rank" in result.diagnostics
        assert "calibration_size" in result.diagnostics

    def test_diagnostics_values(
        self,
        calibrated_amplifier: ECDFAmplifier,
        config: QRSamplerConfig,
    ) -> None:
        """sample_mean should match numpy mean; calibration_size should match config."""
        raw = bytes([10, 20, 30])
        result = calibrated_amplifier.amplify(raw)
        expected_mean = float(np.frombuffer(raw, dtype=np.uint8).mean())
        assert abs(result.diagnostics["sample_mean"] - expected_mean) < 1e-10
        assert result.diagnostics["calibration_size"] == config.ecdf_calibration_samples

    def test_result_is_frozen(self, calibrated_amplifier: ECDFAmplifier) -> None:
        """AmplificationResult should be immutable."""
        result = calibrated_amplifier.amplify(bytes([128] * 100))
        with pytest.raises(AttributeError):
            result.u = 0.5  # type: ignore[misc]

    def test_is_subclass_of_abc(self) -> None:
        """ECDFAmplifier should be a SignalAmplifier subclass."""
        assert issubclass(ECDFAmplifier, SignalAmplifier)


class TestECDFCalibration:
    """Tests for ECDF calibration behavior."""

    def test_calibration_collects_correct_count(self, config: QRSamplerConfig) -> None:
        """get_random_bytes should be called ecdf_calibration_samples times."""
        amp = ECDFAmplifier(config)
        source = MockUniformSource(seed=42)
        with patch.object(source, "get_random_bytes", wraps=source.get_random_bytes) as mock_get:
            amp.calibrate(source, config)
            assert mock_get.call_count == config.ecdf_calibration_samples

    def test_calibration_is_idempotent(self, config: QRSamplerConfig) -> None:
        """Calibrating twice should replace the sorted means array."""
        amp = ECDFAmplifier(config)

        source1 = MockUniformSource(seed=42)
        amp.calibrate(source1, config)
        first_means = amp._sorted_means.copy()  # type: ignore[union-attr]

        source2 = MockUniformSource(seed=99)
        amp.calibrate(source2, config)
        second_means = amp._sorted_means  # type: ignore[union-attr]

        assert not np.array_equal(first_means, second_means)

    def test_calibration_zero_variance_raises(self, config: QRSamplerConfig) -> None:
        """Constant entropy source should raise SignalAmplificationError."""
        amp = ECDFAmplifier(config)
        source = MagicMock(spec=EntropySource)
        source.get_random_bytes.return_value = bytes([128] * config.sample_count)

        with pytest.raises(SignalAmplificationError, match="zero variance"):
            amp.calibrate(source, config)

    def test_calibration_entropy_unavailable_propagates(self, config: QRSamplerConfig) -> None:
        """EntropyUnavailableError from source should propagate uncaught."""
        amp = ECDFAmplifier(config)
        source = MagicMock(spec=EntropySource)
        source.get_random_bytes.side_effect = EntropyUnavailableError("source failed")

        with pytest.raises(EntropyUnavailableError, match="source failed"):
            amp.calibrate(source, config)


class TestAmplifierRegistryECDF:
    """Tests for ECDF registration in AmplifierRegistry."""

    def test_ecdf_is_registered(self) -> None:
        """The ecdf amplifier should be registered at import time."""
        klass = AmplifierRegistry.get("ecdf")
        assert klass is ECDFAmplifier

    def test_build_ecdf_from_config(self, config: QRSamplerConfig) -> None:
        """build() should return an ECDFAmplifier instance."""
        amplifier = AmplifierRegistry.build(config)
        assert isinstance(amplifier, ECDFAmplifier)

    def test_list_registered_includes_ecdf(self) -> None:
        """list_registered() should include ecdf."""
        names = AmplifierRegistry.list_registered()
        assert "ecdf" in names


class TestECDFStatistical:
    """Statistical property tests for ECDF amplification (requires scipy)."""

    def test_uniform_distribution_ks(self, config: QRSamplerConfig) -> None:
        """u values from calibrated ECDF should pass KS test for uniformity."""
        scipy = pytest.importorskip("scipy")

        amp = ECDFAmplifier(config)
        cal_source = MockUniformSource(seed=42)
        amp.calibrate(cal_source, config)

        sample_source = MockUniformSource(seed=99)
        u_values = []
        for _ in range(500):
            raw = sample_source.get_random_bytes(config.sample_count)
            result = amp.amplify(raw)
            u_values.append(result.u)

        stat, p_value = scipy.stats.kstest(u_values, "uniform")
        assert p_value > 0.01, f"KS test failed: stat={stat:.4f}, p={p_value:.4f}"

    def test_bias_correction(self, config: QRSamplerConfig) -> None:
        """ECDF calibration should correct for source bias."""
        pytest.importorskip("scipy")

        biased_mean = 130.0
        amp = ECDFAmplifier(config)
        cal_source = MockUniformSource(mean=biased_mean, seed=42)
        amp.calibrate(cal_source, config)

        sample_source = MockUniformSource(mean=biased_mean, seed=99)
        u_values = []
        for _ in range(500):
            raw = sample_source.get_random_bytes(config.sample_count)
            result = amp.amplify(raw)
            u_values.append(result.u)

        u_mean = sum(u_values) / len(u_values)
        assert 0.4 <= u_mean <= 0.6, f"u_mean={u_mean:.4f} outside [0.4, 0.6]"

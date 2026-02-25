"""Tests for MockUniformSource."""

from __future__ import annotations

import numpy as np

from qr_sampler.entropy.mock import MockUniformSource


class TestMockUniformSource:
    """Tests for the configurable mock entropy source."""

    def test_name(self) -> None:
        source = MockUniformSource()
        assert source.name == "mock_uniform"

    def test_is_always_available(self) -> None:
        source = MockUniformSource()
        assert source.is_available is True

    def test_returns_correct_byte_count(self) -> None:
        source = MockUniformSource(seed=42)
        for n in (0, 1, 10, 100, 1024, 20480):
            data = source.get_random_bytes(n)
            assert len(data) == n

    def test_returns_bytes_type(self) -> None:
        source = MockUniformSource(seed=42)
        data = source.get_random_bytes(16)
        assert isinstance(data, bytes)

    def test_seeded_reproducibility(self) -> None:
        """Same seed must produce identical output."""
        a = MockUniformSource(seed=123).get_random_bytes(100)
        b = MockUniformSource(seed=123).get_random_bytes(100)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = MockUniformSource(seed=1).get_random_bytes(100)
        b = MockUniformSource(seed=2).get_random_bytes(100)
        assert a != b

    def test_default_mean_near_127_5(self) -> None:
        """Default mean=127.5 should produce bytes centred near 127.5."""
        source = MockUniformSource(mean=127.5, seed=42)
        data = source.get_random_bytes(10000)
        arr = np.frombuffer(data, dtype=np.uint8)
        # With 10000 samples, mean should be close to 127.5.
        assert abs(arr.mean() - 127.5) < 5.0

    def test_bias_simulation(self) -> None:
        """mean=140.0 should shift the distribution noticeably upward."""
        source = MockUniformSource(mean=140.0, seed=42)
        data = source.get_random_bytes(10000)
        arr = np.frombuffer(data, dtype=np.uint8)
        # Mean should be notably higher than 127.5.
        assert arr.mean() > 132.0

    def test_bytes_clamped_to_valid_range(self) -> None:
        """All byte values must be in [0, 255]."""
        source = MockUniformSource(mean=127.5, seed=42)
        data = source.get_random_bytes(10000)
        arr = np.frombuffer(data, dtype=np.uint8)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_extreme_mean_low(self) -> None:
        """mean=0 should still produce valid clamped output."""
        source = MockUniformSource(mean=0.0, seed=42)
        data = source.get_random_bytes(100)
        arr = np.frombuffer(data, dtype=np.uint8)
        assert arr.min() >= 0

    def test_extreme_mean_high(self) -> None:
        """mean=255 should still produce valid clamped output."""
        source = MockUniformSource(mean=255.0, seed=42)
        data = source.get_random_bytes(100)
        arr = np.frombuffer(data, dtype=np.uint8)
        assert arr.max() <= 255

    def test_close_is_noop(self) -> None:
        source = MockUniformSource()
        source.close()  # Should not raise.

    def test_health_check(self) -> None:
        source = MockUniformSource()
        health = source.health_check()
        assert health["source"] == "mock_uniform"
        assert health["healthy"] is True

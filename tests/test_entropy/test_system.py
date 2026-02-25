"""Tests for SystemEntropySource."""

from __future__ import annotations

from qr_sampler.entropy.system import SystemEntropySource


class TestSystemEntropySource:
    """Tests for the os.urandom() wrapper."""

    def test_name(self) -> None:
        source = SystemEntropySource()
        assert source.name == "system"

    def test_is_always_available(self) -> None:
        source = SystemEntropySource()
        assert source.is_available is True

    def test_returns_correct_byte_count(self) -> None:
        source = SystemEntropySource()
        for n in (0, 1, 10, 100, 1024, 20480):
            data = source.get_random_bytes(n)
            assert len(data) == n

    def test_returns_bytes_type(self) -> None:
        source = SystemEntropySource()
        data = source.get_random_bytes(16)
        assert isinstance(data, bytes)

    def test_consecutive_calls_differ(self) -> None:
        """Two calls with enough bytes should produce different output."""
        source = SystemEntropySource()
        a = source.get_random_bytes(32)
        b = source.get_random_bytes(32)
        # Statistically near-impossible for 32 random bytes to be equal.
        assert a != b

    def test_close_is_noop(self) -> None:
        source = SystemEntropySource()
        source.close()  # Should not raise.
        # Source should still work after close.
        data = source.get_random_bytes(8)
        assert len(data) == 8

    def test_health_check(self) -> None:
        source = SystemEntropySource()
        health = source.health_check()
        assert health["source"] == "system"
        assert health["healthy"] is True

    def test_get_random_float64(self) -> None:
        import numpy as np

        source = SystemEntropySource()
        result = source.get_random_float64((10,))
        assert result.shape == (10,)
        assert result.dtype == np.float64
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_get_random_float64_with_out(self) -> None:
        import numpy as np

        source = SystemEntropySource()
        out = np.empty((5, 3), dtype=np.float64)
        result = source.get_random_float64((5, 3), out=out)
        assert result is out
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

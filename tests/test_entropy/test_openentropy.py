"""Tests for OpenEntropySource (mocked — no openentropy install required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.exceptions import EntropyUnavailableError


def _make_config(**overrides: object) -> QRSamplerConfig:
    """Create a QRSamplerConfig with openentropy-relevant defaults."""
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


def _make_mock_pool(source_count: int = 3, bytes_return: bytes | None = None) -> MagicMock:
    """Create a mock EntropyPool with configurable source_count and get_bytes return."""
    pool = MagicMock()
    pool.source_count = source_count
    if bytes_return is not None:
        pool.get_bytes.return_value = bytes_return
        pool.get_source_bytes.return_value = bytes_return
    return pool


class TestOpenEntropySource:
    """Tests for OpenEntropySource with fully mocked openentropy library."""

    def test_name(self, default_config: QRSamplerConfig) -> None:
        """Source name should be 'openentropy'."""
        mock_pool = _make_mock_pool(source_count=3, bytes_return=b"\x00" * 32)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            assert source.name == "openentropy"

    def test_is_available_when_installed(self, default_config: QRSamplerConfig) -> None:
        """is_available should be True when pool has sources."""
        mock_pool = _make_mock_pool(source_count=3)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            assert source.is_available is True

    def test_is_available_when_no_sources(self, default_config: QRSamplerConfig) -> None:
        """is_available should be False when pool has zero sources."""
        mock_pool = _make_mock_pool(source_count=0)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            assert source.is_available is False

    def test_get_random_bytes_returns_correct_count(self, default_config: QRSamplerConfig) -> None:
        """get_random_bytes should return exactly n bytes."""
        n = 64
        mock_pool = _make_mock_pool(source_count=3, bytes_return=b"\xab" * n)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            result = source.get_random_bytes(n)
            assert len(result) == n
            assert isinstance(result, bytes)

    def test_get_random_bytes_raw_conditioning(self, default_config: QRSamplerConfig) -> None:
        """Default conditioning should be 'raw'."""
        n = 32
        mock_pool = _make_mock_pool(source_count=3, bytes_return=b"\x00" * n)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            source.get_random_bytes(n)

            # Default oe_conditioning is "raw"; collect_all + get_bytes path
            mock_pool.collect_all.assert_called_once()
            mock_pool.get_bytes.assert_called_once_with(n, conditioning="raw")

    def test_get_random_bytes_sha256_conditioning(self) -> None:
        """SHA256 conditioning should be passed through to pool.get_bytes."""
        n = 32
        config = _make_config(oe_conditioning="sha256")
        mock_pool = _make_mock_pool(source_count=3, bytes_return=b"\x00" * n)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(config)
            source.get_random_bytes(n)

            mock_pool.get_bytes.assert_called_once_with(n, conditioning="sha256")

    def test_close_idempotent(self, default_config: QRSamplerConfig) -> None:
        """close() should be idempotent; subsequent get_random_bytes raises."""
        mock_pool = _make_mock_pool(source_count=3, bytes_return=b"\x00" * 16)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            source.close()  # First close — no error.
            source.close()  # Second close — still no error.

            with pytest.raises(EntropyUnavailableError, match="closed"):
                source.get_random_bytes(16)

    def test_health_check_when_available(self, default_config: QRSamplerConfig) -> None:
        """health_check should return source, healthy, source_count, conditioning."""
        mock_pool = _make_mock_pool(source_count=5)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            health = source.health_check()

            assert health["source"] == "openentropy"
            assert health["healthy"] is True
            assert health["source_count"] == 5
            assert health["conditioning"] == "raw"

    def test_health_check_when_not_installed(self) -> None:
        """health_check with openentropy unavailable returns unhealthy dict."""
        # We need an existing instance to call health_check on.
        # Create one while "installed", then patch the flag to False.
        mock_pool = _make_mock_pool(source_count=3)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            config = _make_config()
            source = OpenEntropySource(config)

        # Now patch the flag to False (simulating post-construction unavailability).
        with patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", False):
            health = source.health_check()
            assert health == {
                "source": "openentropy",
                "healthy": False,
                "reason": "openentropy not installed",
            }

    def test_source_filtering(self) -> None:
        """When oe_sources is set, get_source_bytes should be called per source."""
        config = _make_config(oe_sources="clock_jitter,dram_row_buffer")
        n = 64
        mock_pool = _make_mock_pool(source_count=2, bytes_return=b"\xff" * n)
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(config)
            result = source.get_random_bytes(n)

            # Should NOT use collect_all path.
            mock_pool.collect_all.assert_not_called()
            mock_pool.get_bytes.assert_not_called()

            # Should call get_source_bytes for each named source.
            assert mock_pool.get_source_bytes.call_count == 2
            calls = mock_pool.get_source_bytes.call_args_list
            assert calls[0][0] == ("clock_jitter", n)
            assert calls[0][1] == {"conditioning": "raw"}
            assert calls[1][0] == ("dram_row_buffer", n)
            assert calls[1][1] == {"conditioning": "raw"}

            # Result is truncated to n bytes.
            assert len(result) == n

    def test_raises_entropy_unavailable_on_runtime_error(
        self, default_config: QRSamplerConfig
    ) -> None:
        """RuntimeError from pool should be wrapped in EntropyUnavailableError."""
        mock_pool = _make_mock_pool(source_count=3)
        mock_pool.collect_all.side_effect = RuntimeError("hardware fault")
        mock_pool_class = MagicMock(spec=[])
        mock_pool_class.auto = MagicMock(return_value=mock_pool)

        with (
            patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", True),
            patch("qr_sampler.entropy.openentropy_source.EntropyPool", mock_pool_class, create=True),
        ):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            source = OpenEntropySource(default_config)
            with pytest.raises(EntropyUnavailableError, match="OpenEntropy failed"):
                source.get_random_bytes(32)

    def test_raises_when_not_installed(self) -> None:
        """Constructor should raise EntropyUnavailableError when openentropy is absent."""
        config = _make_config()

        with patch("qr_sampler.entropy.openentropy_source._OPENENTROPY_AVAILABLE", False):
            from qr_sampler.entropy.openentropy_source import OpenEntropySource

            with pytest.raises(EntropyUnavailableError, match="not installed"):
                OpenEntropySource(config)

"""Tests for FallbackEntropySource."""

from __future__ import annotations

import pytest

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.exceptions import EntropyUnavailableError


class _AlwaysFailSource(EntropySource):
    """Test double: always raises EntropyUnavailableError."""

    @property
    def name(self) -> str:
        return "always_fail"

    @property
    def is_available(self) -> bool:
        return False

    def get_random_bytes(self, n: int) -> bytes:
        raise EntropyUnavailableError("always fails")

    def close(self) -> None:
        pass


class _RuntimeErrorSource(EntropySource):
    """Test double: always raises RuntimeError (not EntropyUnavailableError)."""

    @property
    def name(self) -> str:
        return "runtime_error"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        raise RuntimeError("unexpected error")

    def close(self) -> None:
        pass


class _FixedBytesSource(EntropySource):
    """Test double: returns a fixed byte pattern."""

    def __init__(self, pattern: int = 0xAA) -> None:
        self._pattern = pattern
        self.call_count = 0

    @property
    def name(self) -> str:
        return f"fixed_{self._pattern:#04x}"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        self.call_count += 1
        return bytes([self._pattern] * n)

    def close(self) -> None:
        pass


class TestFallbackEntropySource:
    """Tests for the composition fallback wrapper."""

    def test_delegates_to_primary(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        data = source.get_random_bytes(4)
        assert data == bytes([0xAA] * 4)
        assert primary.call_count == 1
        assert fallback.call_count == 0

    def test_falls_back_on_entropy_unavailable(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        data = source.get_random_bytes(4)
        assert data == bytes([0xBB] * 4)
        assert fallback.call_count == 1

    def test_last_source_used_tracks_primary(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        source.get_random_bytes(4)
        assert source.last_source_used == primary.name

    def test_last_source_used_tracks_fallback(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        source.get_random_bytes(4)
        assert source.last_source_used == fallback.name

    def test_does_not_catch_non_entropy_errors(self) -> None:
        """RuntimeError should propagate, not trigger fallback."""
        primary = _RuntimeErrorSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        with pytest.raises(RuntimeError, match="unexpected error"):
            source.get_random_bytes(4)
        assert fallback.call_count == 0

    def test_raises_when_both_fail(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _AlwaysFailSource()
        source = FallbackEntropySource(primary, fallback)

        with pytest.raises(EntropyUnavailableError):
            source.get_random_bytes(4)

    def test_name_is_compound(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        assert source.name == f"{primary.name}+{fallback.name}"

    def test_is_available_when_primary_available(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _AlwaysFailSource()
        source = FallbackEntropySource(primary, fallback)
        assert source.is_available is True

    def test_is_available_when_only_fallback_available(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)
        assert source.is_available is True

    def test_is_unavailable_when_both_unavailable(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _AlwaysFailSource()
        source = FallbackEntropySource(primary, fallback)
        assert source.is_available is False

    def test_close_closes_both(self) -> None:
        closed: list[str] = []

        class _TrackClose(EntropySource):
            def __init__(self, id: str) -> None:
                self._id = id

            @property
            def name(self) -> str:
                return self._id

            @property
            def is_available(self) -> bool:
                return True

            def get_random_bytes(self, n: int) -> bytes:
                return b"\x00" * n

            def close(self) -> None:
                closed.append(self._id)

        primary = _TrackClose("p")
        fallback = _TrackClose("f")
        source = FallbackEntropySource(primary, fallback)
        source.close()
        assert "p" in closed
        assert "f" in closed

    def test_health_check(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        health = source.health_check()
        assert health["source"] == source.name
        assert health["healthy"] is True
        assert "primary" in health
        assert "fallback" in health
        assert "last_source_used" in health

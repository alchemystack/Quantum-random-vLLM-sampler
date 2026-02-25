"""Tests for QuantumGrpcSource (mocked gRPC)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qr_sampler.exceptions import EntropyUnavailableError


def _make_config(**overrides: Any) -> Any:
    """Create a mock config object with gRPC defaults."""
    from qr_sampler.config import QRSamplerConfig

    defaults = {
        "grpc_server_address": "localhost:50051",
        "grpc_timeout_ms": 5000.0,
        "grpc_retry_count": 2,
        "grpc_mode": "unary",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


class TestQuantumGrpcSourceImport:
    """Tests for import-time checks."""

    def test_requires_grpcio(self) -> None:
        """Should raise ImportError if grpcio is not available."""
        with (
            patch.dict("sys.modules", {"grpc": None, "grpc.aio": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from qr_sampler.entropy.quantum import QuantumGrpcSource

            config = _make_config()
            QuantumGrpcSource(config)


class _MockResponse:
    """Mock gRPC response."""

    def __init__(self, data: bytes, sequence_id: int = 1) -> None:
        self.data = data
        self.sequence_id = sequence_id
        self.generation_timestamp_ns = time.time_ns()
        self.device_id = "mock"


class TestQuantumGrpcSourceUnary:
    """Tests for unary transport mode with mocked gRPC."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource with fully mocked gRPC."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            # Mock the stub's GetEntropy method.
            mock_stub = MagicMock()
            mock_get_entropy = AsyncMock(return_value=_MockResponse(b"\x42" * 100))
            mock_stub.GetEntropy = mock_get_entropy

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                source._mock_stub = mock_stub  # type: ignore[attr-defined]
                yield source
                source.close()

    def test_name(self, source: Any) -> None:
        assert source.name == "quantum_grpc"

    def test_is_available(self, source: Any) -> None:
        assert source.is_available is True

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        data = source.get_random_bytes(100)
        assert len(data) == 100
        assert data == b"\x42" * 100

    def test_health_check(self, source: Any) -> None:
        health = source.health_check()
        assert health["source"] == "quantum_grpc"
        assert health["mode"] == "unary"
        assert "p99_ms" in health

    def test_close_sets_unavailable(self, source: Any) -> None:
        source.close()
        assert source.is_available is False


class TestQuantumGrpcSourceCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource with a stub that always fails."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary", grpc_retry_count=0)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            mock_stub.GetEntropy = AsyncMock(side_effect=Exception("connection refused"))

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                yield source
                source.close()

    def test_circuit_opens_after_consecutive_failures(self, source: Any) -> None:
        """Circuit should open after cb_max_consecutive_failures."""
        for _ in range(source._cb_max_consecutive_failures):
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)

        assert source._circuit_open is True
        assert source.is_available is False

    def test_circuit_open_raises_immediately(self, source: Any) -> None:
        """When circuit is open, should raise without trying gRPC."""
        source._circuit_open = True
        source._circuit_open_until = time.monotonic() + 100.0

        with pytest.raises(EntropyUnavailableError, match="Circuit breaker open"):
            source.get_random_bytes(10)


class TestQuantumGrpcSourceAddressParsing:
    """Tests for TCP vs Unix socket address handling."""

    def test_tcp_address(self) -> None:
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_server_address="myhost:9090")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                mock_channel_fn.assert_called_once()
                call_args = mock_channel_fn.call_args
                assert call_args[0][0] == "myhost:9090"
                source.close()

    def test_unix_socket_address(self) -> None:
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_server_address="unix:///var/run/qrng.sock")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                call_args = mock_channel_fn.call_args
                assert call_args[0][0] == "unix:///var/run/qrng.sock"
                source.close()


class TestQuantumGrpcSourceLatencyTracking:
    """Tests for adaptive timeout computation."""

    def test_update_latency_and_timeout(self) -> None:
        """P99 and timeout should update from latency window."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)

                # Feed in 20 latency samples.
                for i in range(20):
                    source._update_latency(float(i))

                # P99 should be near the max of the window.
                assert source._p99_ms >= 15.0

                # Adaptive timeout: max(5ms, P99 * 1.5), capped at config.
                timeout = source._get_timeout()
                assert timeout >= 5.0
                assert timeout <= config.grpc_timeout_ms

                source.close()


class TestCircuitBreakerHalfOpen:
    """Tests for circuit breaker half-open state and recovery."""

    def test_half_open_allows_one_request(self) -> None:
        """After recovery window expires, one request should be attempted."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="unary",
            grpc_retry_count=0,
            cb_recovery_window_s=0.0,  # Immediate recovery for testing.
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            # First calls fail, then succeed.
            mock_stub.GetEntropy = AsyncMock(
                side_effect=[
                    Exception("fail"),
                    Exception("fail"),
                    Exception("fail"),
                    _MockResponse(b"\xaa" * 10),
                ]
            )

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                try:
                    # Trigger circuit breaker open (3 consecutive failures).
                    for _ in range(3):
                        with pytest.raises(EntropyUnavailableError):
                            source.get_random_bytes(10)

                    assert source._circuit_open is True

                    # Recovery window is 0.0s, so half-open should trigger
                    # immediately. The next call should succeed.
                    data = source.get_random_bytes(10)
                    assert data == b"\xaa" * 10
                    assert source._circuit_open is False
                    assert source._consecutive_failures == 0
                finally:
                    source.close()

    def test_half_open_failure_reopens_circuit(self) -> None:
        """If the half-open test request fails, circuit should reopen."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="unary",
            grpc_retry_count=0,
            cb_recovery_window_s=0.0,
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            # All calls fail — half-open test will also fail.
            mock_stub.GetEntropy = AsyncMock(side_effect=Exception("still broken"))

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                try:
                    # Open the circuit.
                    for _ in range(3):
                        with pytest.raises(EntropyUnavailableError):
                            source.get_random_bytes(10)

                    assert source._circuit_open is True

                    # Half-open attempt should fail and reopen circuit.
                    with pytest.raises(EntropyUnavailableError):
                        source.get_random_bytes(10)

                    # Circuit should be open again (consecutive failures incremented).
                    assert source._circuit_open is True
                finally:
                    source.close()


class TestQuantumGrpcSourceServerStreaming:
    """Tests for server_streaming transport mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource in server_streaming mode."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="server_streaming")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()

            # Mock StreamEntropy: returns a call object with .read() and .cancel().
            mock_stream_call = AsyncMock()
            mock_stream_call.read = AsyncMock(return_value=_MockResponse(b"\x55" * 50))
            mock_stream_call.cancel = MagicMock()
            mock_stub.StreamEntropy = MagicMock(return_value=mock_stream_call)

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                source._mock_stub = mock_stub  # type: ignore[attr-defined]
                yield source
                source.close()

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        """Server streaming should return data from the stream."""
        data = source.get_random_bytes(50)
        assert len(data) == 50
        assert data == b"\x55" * 50

    def test_stream_end_raises(self) -> None:
        """If the stream ends unexpectedly (read returns None), should raise."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="server_streaming", grpc_retry_count=0)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            mock_stream_call = AsyncMock()
            mock_stream_call.read = AsyncMock(return_value=None)
            mock_stream_call.cancel = MagicMock()
            mock_stub.StreamEntropy = MagicMock(return_value=mock_stream_call)

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                try:
                    with pytest.raises(EntropyUnavailableError):
                        source.get_random_bytes(10)
                finally:
                    source.close()


class TestQuantumGrpcSourceBidiStreaming:
    """Tests for bidi_streaming transport mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource in bidi_streaming mode."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="bidi_streaming")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()

            # Mock StreamEntropy: returns a call object with .write(), .read().
            mock_bidi_call = AsyncMock()
            mock_bidi_call.write = AsyncMock()
            mock_bidi_call.read = AsyncMock(return_value=_MockResponse(b"\xcc" * 64))
            mock_stub.StreamEntropy = MagicMock(return_value=mock_bidi_call)

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                source._mock_stub = mock_stub  # type: ignore[attr-defined]
                yield source
                source.close()

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        """Bidi streaming should return data from the persistent stream."""
        data = source.get_random_bytes(64)
        assert len(data) == 64
        assert data == b"\xcc" * 64

    def test_stream_reuses_call(self, source: Any) -> None:
        """Bidi streaming should reuse the same call object."""
        source.get_random_bytes(64)
        source.get_random_bytes(64)
        # The stub's StreamEntropy should only have been called once
        # (the bidi call is reused).
        assert source._mock_stub.StreamEntropy.call_count == 1

    def test_bidi_stream_end_resets(self) -> None:
        """If bidi stream ends (read returns None), call should reset."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="bidi_streaming", grpc_retry_count=0)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = AsyncMock()
            mock_channel_fn.return_value = mock_channel

            mock_stub = MagicMock()
            mock_bidi_call = AsyncMock()
            mock_bidi_call.write = AsyncMock()
            mock_bidi_call.read = AsyncMock(return_value=None)
            mock_stub.StreamEntropy = MagicMock(return_value=mock_bidi_call)

            with patch(
                "qr_sampler.proto.entropy_service_pb2_grpc.EntropyServiceStub",
                return_value=mock_stub,
            ):
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                source = QuantumGrpcSource(config)
                try:
                    with pytest.raises(EntropyUnavailableError):
                        source.get_random_bytes(10)
                    # The bidi call should have been reset to None.
                    assert source._bidi_call is None
                finally:
                    source.close()

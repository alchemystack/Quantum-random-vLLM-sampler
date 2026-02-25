"""Quantum gRPC entropy source with configurable transport modes.

This is the primary production entropy source. It fetches random bytes from
a remote QRNG server over gRPC, supporting three transport modes:

- **Unary**: simple request-response (``GetEntropy``). One HTTP/2 stream per call.
- **Server streaming**: client sends one config request, server streams responses.
- **Bidirectional streaming**: persistent stream with lowest latency.

All modes satisfy the just-in-time constraint: the gRPC request is sent
only when ``get_random_bytes()`` is called (i.e., after logits are available).

Includes an adaptive circuit breaker that tracks rolling P99 latency and
falls back to a secondary source when the QRNG is slow or unreachable.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source
from qr_sampler.exceptions import EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

logger = logging.getLogger("qr_sampler")


@register_entropy_source("quantum_grpc")
class QuantumGrpcSource(EntropySource):
    """gRPC entropy source with configurable transport mode.

    All modes satisfy the just-in-time constraint: the gRPC request
    is only sent when ``get_random_bytes()`` is called (i.e., after logits
    are available). The transport mode affects connection management
    overhead, not entropy freshness.

    Args:
        config: Sampler configuration with gRPC settings.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        try:
            import grpc.aio  # noqa: F401 — availability check
        except ImportError as exc:
            raise ImportError(
                "grpcio is required for QuantumGrpcSource. "
                "Install it with: pip install qr-sampler[grpc]"
            ) from exc

        self._address = config.grpc_server_address
        self._timeout_ms = config.grpc_timeout_ms
        self._retry_count = config.grpc_retry_count
        self._mode = config.grpc_mode
        self._sequence_id = 0
        self._closed = False

        # Circuit breaker config.
        self._cb_min_timeout_ms = config.cb_min_timeout_ms
        self._cb_timeout_multiplier = config.cb_timeout_multiplier
        self._cb_recovery_window_s = config.cb_recovery_window_s
        self._cb_max_consecutive_failures = config.cb_max_consecutive_failures

        # Circuit breaker state.
        self._latency_window: deque[float] = deque(maxlen=config.cb_window_size)
        self._p99_ms: float = self._timeout_ms
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._circuit_open_until: float = 0.0

        # Background event loop for async gRPC.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="qr-sampler-grpc-loop",
        )
        self._thread.start()

        # Initialize channel and stub on the background loop.
        future = asyncio.run_coroutine_threadsafe(self._init_channel(), self._loop)
        future.result(timeout=self._timeout_ms / 1000.0)

        # Streaming state (lazily initialized).
        self._bidi_call: Any | None = None

    def _run_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_channel(self) -> None:
        """Create the gRPC async channel and stub."""
        import grpc.aio

        options = [
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
        ]

        self._channel = grpc.aio.insecure_channel(self._address, options=options)

        from qr_sampler.proto.entropy_service_pb2_grpc import EntropyServiceStub

        self._stub = EntropyServiceStub(self._channel)

    @property
    def name(self) -> str:
        """Return ``'quantum_grpc'``."""
        return "quantum_grpc"

    @property
    def is_available(self) -> bool:
        """Whether the source can currently provide entropy.

        Returns ``False`` if the circuit breaker is open (too many failures).
        """
        if self._closed:
            return False
        return not (self._circuit_open and time.monotonic() < self._circuit_open_until)

    def get_random_bytes(self, n: int) -> bytes:
        """Fetch *n* random bytes from the gRPC entropy server.

        Synchronous wrapper around the async transport. Uses the background
        event loop thread to dispatch async calls.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes from the QRNG server.

        Raises:
            EntropyUnavailableError: If the server is unreachable or the
                circuit breaker is open.
        """
        if self._closed:
            raise EntropyUnavailableError("QuantumGrpcSource is closed")

        # Circuit breaker check.
        if self._circuit_open:
            if time.monotonic() >= self._circuit_open_until:
                # Half-open: try one request.
                self._circuit_open = False
                logger.info("Circuit breaker half-open, attempting reconnection")
            else:
                raise EntropyUnavailableError(
                    "Circuit breaker open: too many consecutive gRPC failures"
                )

        last_error: Exception | None = None
        for attempt in range(1 + self._retry_count):
            try:
                t0 = time.perf_counter()
                data = self._fetch_sync(n)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._update_latency(elapsed_ms)
                self._consecutive_failures = 0
                return data
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "gRPC entropy fetch attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self._retry_count,
                    exc,
                )

        # All retries exhausted.
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cb_max_consecutive_failures:
            self._circuit_open = True
            self._circuit_open_until = time.monotonic() + self._cb_recovery_window_s
            logger.warning(
                "Circuit breaker opened after %d consecutive failures",
                self._consecutive_failures,
            )

        raise EntropyUnavailableError(
            f"gRPC entropy fetch failed after {1 + self._retry_count} attempts: {last_error}"
        ) from last_error

    def _fetch_sync(self, n: int) -> bytes:
        """Dispatch an async fetch to the background loop and block."""
        timeout_s = self._get_timeout() / 1000.0
        coro = self._fetch_async(n)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError as exc:
            raise EntropyUnavailableError(
                f"gRPC entropy fetch timed out after {timeout_s * 1000:.0f}ms"
            ) from exc
        except Exception as exc:
            raise EntropyUnavailableError(f"gRPC entropy fetch failed: {exc}") from exc

    async def _fetch_async(self, n: int) -> bytes:
        """Route to the appropriate transport mode."""
        if self._mode == "unary":
            return await self._fetch_unary(n)
        elif self._mode == "server_streaming":
            return await self._fetch_server_streaming(n)
        elif self._mode == "bidi_streaming":
            return await self._fetch_bidi_streaming(n)
        else:
            raise EntropyUnavailableError(f"Unknown gRPC mode: {self._mode!r}")

    async def _fetch_unary(self, n: int) -> bytes:
        """Single request-response per call. Simplest. Higher overhead."""
        from qr_sampler.proto.entropy_service_pb2 import EntropyRequest

        self._sequence_id += 1
        request = EntropyRequest(bytes_needed=n, sequence_id=self._sequence_id)
        timeout_s = self._get_timeout() / 1000.0
        response = await self._stub.GetEntropy(request, timeout=timeout_s)
        data: bytes = response.data
        return data

    async def _fetch_server_streaming(self, n: int) -> bytes:
        """Use the StreamEntropy RPC in a request/response style.

        Sends one request and reads one response from the bidirectional stream.
        The stream is re-established on each call for server-streaming semantics.
        """
        from qr_sampler.proto.entropy_service_pb2 import EntropyRequest

        self._sequence_id += 1
        request = EntropyRequest(bytes_needed=n, sequence_id=self._sequence_id)

        async def request_iterator() -> Any:
            yield request

        call = self._stub.StreamEntropy(request_iterator())
        response = await call.read()
        if response is None:
            raise EntropyUnavailableError("Server stream ended unexpectedly")
        call.cancel()
        data: bytes = response.data
        return data

    async def _fetch_bidi_streaming(self, n: int) -> bytes:
        """Use a persistent bidirectional stream for lowest latency.

        The stream is lazily initialized on first call and reused thereafter.
        If the stream breaks, it is re-established on the next call.
        """
        from qr_sampler.proto.entropy_service_pb2 import EntropyRequest

        self._sequence_id += 1
        request = EntropyRequest(bytes_needed=n, sequence_id=self._sequence_id)

        try:
            if self._bidi_call is None:
                self._bidi_call = self._stub.StreamEntropy()

            await self._bidi_call.write(request)
            response = await self._bidi_call.read()
            if response is None:
                # Stream ended — reset and retry.
                self._bidi_call = None
                raise EntropyUnavailableError("Bidi stream ended unexpectedly")
            data: bytes = response.data
            return data
        except EntropyUnavailableError:
            raise
        except Exception:
            # Stream broken — reset for next call.
            self._bidi_call = None
            raise

    # --- Circuit breaker ---

    def _update_latency(self, elapsed_ms: float) -> None:
        """Add a latency sample to the rolling window and recompute P99.

        Args:
            elapsed_ms: Time taken for the last fetch in milliseconds.
        """
        self._latency_window.append(elapsed_ms)
        if len(self._latency_window) >= 10:
            sorted_latencies = sorted(self._latency_window)
            idx = int(len(sorted_latencies) * 0.99)
            idx = min(idx, len(sorted_latencies) - 1)
            self._p99_ms = sorted_latencies[idx]

    def _get_timeout(self) -> float:
        """Compute the adaptive timeout in milliseconds.

        Returns:
            ``max(5ms, P99 * 1.5)`` or the configured timeout, whichever
            is smaller.
        """
        adaptive = max(self._cb_min_timeout_ms, self._p99_ms * self._cb_timeout_multiplier)
        return min(adaptive, self._timeout_ms)

    # --- Lifecycle ---

    def close(self) -> None:
        """Release the gRPC channel, event loop, and background thread."""
        if self._closed:
            return
        self._closed = True

        async def _shutdown() -> None:
            if self._bidi_call is not None:
                self._bidi_call.cancel()
                self._bidi_call = None
            await self._channel.close()

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            future.result(timeout=5.0)
        except Exception:
            logger.warning("Error during QuantumGrpcSource cleanup", exc_info=True)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)

    def health_check(self) -> dict[str, Any]:
        """Return detailed health status including circuit breaker state.

        Returns:
            Dictionary with source name, availability, circuit breaker state,
            P99 latency, and connection details.
        """
        return {
            "source": self.name,
            "healthy": self.is_available,
            "address": self._address,
            "mode": self._mode,
            "circuit_open": self._circuit_open,
            "p99_ms": round(self._p99_ms, 2),
            "consecutive_failures": self._consecutive_failures,
            "latency_samples": len(self._latency_window),
        }

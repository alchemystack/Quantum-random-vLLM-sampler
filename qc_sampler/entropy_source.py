"""Entropy source providers for quantum random byte generation.

This module defines the abstract interface for entropy sources and provides
four concrete implementations:

- **GrpcEntropySource**: Production client that fetches quantum random bytes
  from a remote QRNG server via gRPC.  Supports prefetching (background fetch
  during the GPU forward pass) and configurable retry/timeout behaviour.
- **OsUrandomSource**: Uses ``os.urandom()`` for testing and fallback.
- **MockUniformSource**: Generates bytes centred around a configurable mean,
  useful for unit-testing signal amplification with controlled bias.
- **FallbackEntropySource**: Wrapper that tries a primary source first and
  falls back to a secondary source on ``EntropyUnavailableError``.

Design notes:
    The wrapper-based fallback pattern keeps each source implementation
    pure (single responsibility).  Composing sources is the caller's job
    (see factory.py), not something baked into individual source classes.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import grpc
import numpy as np

from qc_sampler.config import QCSamplingConfig
from qc_sampler.exceptions import EntropyUnavailableError
from qc_sampler.proto import entropy_service_pb2
from qc_sampler.proto import entropy_service_pb2_grpc

logger = logging.getLogger("qc_sampler")


class EntropySource(ABC):
    """Abstract base class for providers of raw random bytes.

    Every entropy source must implement three methods:
    - ``get_bytes``: blocking fetch of exactly *count* bytes.
    - ``prefetch``: non-blocking hint to begin generating bytes in advance.
    - ``health_check``: returns a dict describing source status.

    Implementations raise ``EntropyUnavailableError`` when they cannot
    fulfil a ``get_bytes`` request.  Higher-level code (or the
    ``FallbackEntropySource`` wrapper) handles graceful degradation.
    """

    @abstractmethod
    def get_bytes(self, count: int) -> bytes:
        """Return exactly *count* raw bytes.  Blocking call.

        Args:
            count: Number of bytes to return.

        Returns:
            A bytes object of length *count*.

        Raises:
            EntropyUnavailableError: If the source cannot provide bytes.
        """

    @abstractmethod
    def prefetch(self, count: int) -> None:
        """Begin asynchronous generation of *count* bytes.  Non-blocking.

        If prefetching is supported, the next call to ``get_bytes()``
        returns the prefetched data when ready.  Sources that do not
        support prefetching may implement this as a no-op.

        Args:
            count: Number of bytes to prepare.
        """

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return a dict with source status information for monitoring.

        At minimum the dict should contain:
        - ``"source"``: str identifying the implementation.
        - ``"healthy"``: bool indicating whether the source is usable.
        """


# ---------------------------------------------------------------------------
# Concrete: gRPC QRNG client
# ---------------------------------------------------------------------------


class GrpcEntropySource(EntropySource):
    """Production entropy source that fetches quantum random bytes via gRPC.

    Maintains a persistent gRPC channel with keepalive pings.  Supports
    background prefetching so that the gRPC round-trip overlaps with the
    GPU forward pass.

    Thread safety:
        A ``threading.Lock`` guards the prefetch buffer.  ``prefetch()``
        launches a daemon thread that writes to the buffer; ``get_bytes()``
        consumes it under the same lock.

    Retry behaviour:
        On gRPC failure, retries up to ``config.qrng_retry_count`` times
        before raising ``EntropyUnavailableError``.
    """

    def __init__(self, config: QCSamplingConfig) -> None:
        """Initialize the gRPC entropy source.

        Args:
            config: Sampling config containing server address, timeout,
                retry count, and prefetch settings.
        """
        self._address = config.qrng_server_address
        self._timeout_s = config.qrng_timeout_ms / 1000.0
        self._retry_count = config.qrng_retry_count
        self._prefetch_enabled = config.qrng_prefetch_enabled

        # Persistent channel with keepalive options.
        self._channel = grpc.insecure_channel(
            self._address,
            options=[
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self._stub = entropy_service_pb2_grpc.EntropyServiceStub(self._channel)

        # Prefetch state: guarded by _lock.
        self._lock = threading.Lock()
        self._prefetched_data: bytes | None = None
        self._prefetch_error: BaseException | None = None

    # -- public interface ---------------------------------------------------

    def get_bytes(self, count: int) -> bytes:
        """Return exactly *count* bytes from the QRNG server.

        If prefetched data of the correct length is available it is
        returned immediately (zero-latency path).  Otherwise a blocking
        gRPC call is made with retry logic.

        Args:
            count: Number of random bytes to return.

        Returns:
            Exactly *count* bytes of quantum random data.

        Raises:
            EntropyUnavailableError: After all retries are exhausted.
        """
        # Try prefetched data first.
        with self._lock:
            if self._prefetch_error is not None:
                err = self._prefetch_error
                self._prefetch_error = None
                self._prefetched_data = None
                logger.warning(
                    "Prefetched entropy failed, falling through to blocking call: %s",
                    err,
                )
            elif self._prefetched_data is not None:
                data = self._prefetched_data
                self._prefetched_data = None
                if len(data) == count:
                    logger.debug(
                        "Returning prefetched entropy: %d bytes", count
                    )
                    return data
                # Length mismatch — discard and fetch fresh.
                logger.debug(
                    "Prefetched length %d != requested %d; fetching fresh",
                    len(data),
                    count,
                )

        return self._fetch_with_retry(count)

    def prefetch(self, count: int) -> None:
        """Begin background fetch of *count* bytes.

        Does nothing if prefetching is disabled in config.  Only one
        prefetch can be in-flight at a time; a new call overwrites any
        pending result.

        Args:
            count: Number of bytes to prepare.
        """
        if not self._prefetch_enabled:
            return

        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(count,),
            daemon=True,
            name="qc-entropy-prefetch",
        )
        thread.start()

    def health_check(self) -> dict[str, Any]:
        """Check gRPC channel connectivity.

        Returns:
            Dict with ``source``, ``healthy``, ``address``, and
            ``channel_state`` keys.
        """
        try:
            state = self._channel.check_connectivity_state(True)
            healthy = state in (
                grpc.ChannelConnectivity.IDLE,
                grpc.ChannelConnectivity.READY,
                grpc.ChannelConnectivity.CONNECTING,
            )
        except Exception:  # noqa: BLE001 — health check must not raise
            state = None
            healthy = False

        return {
            "source": "grpc",
            "healthy": healthy,
            "address": self._address,
            "channel_state": str(state),
        }

    # -- internals ----------------------------------------------------------

    def _fetch_with_retry(self, count: int) -> bytes:
        """Blocking gRPC fetch with retry logic.

        Attempts up to ``1 + self._retry_count`` calls (initial + retries).

        Args:
            count: Number of bytes to fetch.

        Returns:
            Exactly *count* bytes.

        Raises:
            EntropyUnavailableError: When all attempts fail.
        """
        last_error: BaseException | None = None
        max_attempts = 1 + self._retry_count

        for attempt in range(max_attempts):
            try:
                start_ns = time.perf_counter_ns()
                request = entropy_service_pb2.EntropyRequest(
                    bytes_needed=count
                )
                response = self._stub.GetEntropy(
                    request, timeout=self._timeout_s
                )
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                data = response.data

                logger.debug(
                    "gRPC GetEntropy: %d bytes in %.2f ms (attempt %d/%d)",
                    len(data),
                    elapsed_ms,
                    attempt + 1,
                    max_attempts,
                )

                if len(data) != count:
                    raise EntropyUnavailableError(
                        f"Server returned {len(data)} bytes, expected {count}"
                    )
                return data

            except grpc.RpcError as exc:
                last_error = exc
                logger.warning(
                    "gRPC attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
            except EntropyUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "gRPC attempt %d/%d unexpected error: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )

        raise EntropyUnavailableError(
            f"QRNG gRPC server unreachable after {max_attempts} attempts "
            f"(address: {self._address}): {last_error}"
        )

    def _prefetch_worker(self, count: int) -> None:
        """Background thread target: fetches bytes and stores the result.

        On success, stores data in ``_prefetched_data``.
        On failure, stores the exception in ``_prefetch_error``.
        """
        try:
            data = self._fetch_with_retry(count)
            with self._lock:
                self._prefetched_data = data
                self._prefetch_error = None
        except EntropyUnavailableError as exc:
            with self._lock:
                self._prefetched_data = None
                self._prefetch_error = exc


# ---------------------------------------------------------------------------
# Concrete: OS urandom
# ---------------------------------------------------------------------------


class OsUrandomSource(EntropySource):
    """Entropy source backed by the operating system's CSPRNG.

    Uses ``os.urandom()`` which is cryptographically secure but *not*
    quantum-random.  Suitable for testing, development, and as a fallback
    when the QRNG server is unavailable.
    """

    def get_bytes(self, count: int) -> bytes:
        """Return *count* bytes from ``os.urandom()``.

        Args:
            count: Number of bytes to return.

        Returns:
            Exactly *count* pseudorandom bytes.
        """
        return os.urandom(count)

    def prefetch(self, count: int) -> None:
        """No-op — os.urandom is fast enough that prefetching is unnecessary.

        Args:
            count: Ignored.
        """

    def health_check(self) -> dict[str, Any]:
        """Always healthy — os.urandom is always available.

        Returns:
            Dict with ``source`` and ``healthy`` keys.
        """
        return {"source": "os_urandom", "healthy": True}


# ---------------------------------------------------------------------------
# Concrete: Mock source for testing
# ---------------------------------------------------------------------------


class MockUniformSource(EntropySource):
    """Entropy source that generates bytes centred around a configurable mean.

    Useful for unit-testing the signal amplification pipeline with
    controlled bias.  For example, ``MockUniformSource(mean=128.0)``
    produces bytes whose average is slightly above the null-hypothesis
    mean of 127.5, simulating a consciousness-biased QRNG.

    The generated bytes are uint8 values drawn from a normal distribution
    centred on *mean* with a standard deviation that keeps most values
    in the valid [0, 255] range, then clipped and rounded.

    Args:
        mean: The target mean of the generated byte distribution.
        seed: Optional RNG seed for reproducible test output.
    """

    # Standard deviation for generated bytes — wide enough to look realistic
    # but not so wide that clipping dominates.  This is a test-only constant
    # with no algorithmic significance; it controls how noisy mock data looks.
    _MOCK_BYTE_STD = 40.0

    def __init__(self, mean: float = 127.5, seed: int | None = None) -> None:
        """Initialize the mock source.

        Args:
            mean: Target mean of the generated byte values.
            seed: Optional RNG seed for reproducibility.
        """
        self._mean = mean
        self._rng = np.random.default_rng(seed)

    def get_bytes(self, count: int) -> bytes:
        """Generate *count* bytes centred around the configured mean.

        Args:
            count: Number of bytes to return.

        Returns:
            Exactly *count* bytes whose average is approximately
            ``self._mean``.
        """
        samples = self._rng.normal(
            loc=self._mean, scale=self._MOCK_BYTE_STD, size=count
        )
        clipped = np.clip(np.round(samples), 0, 255).astype(np.uint8)
        return clipped.tobytes()

    def prefetch(self, count: int) -> None:
        """No-op — mock generation is instantaneous.

        Args:
            count: Ignored.
        """

    def health_check(self) -> dict[str, Any]:
        """Always healthy.

        Returns:
            Dict with ``source``, ``healthy``, and ``configured_mean`` keys.
        """
        return {
            "source": "mock_uniform",
            "healthy": True,
            "configured_mean": self._mean,
        }


# ---------------------------------------------------------------------------
# Concrete: Fallback wrapper
# ---------------------------------------------------------------------------


class FallbackEntropySource(EntropySource):
    """Wrapper that tries a primary source and falls back on failure.

    This implements the wrapper/decorator pattern rather than inheritance,
    keeping each source implementation pure and composable.  The wrapper
    catches only ``EntropyUnavailableError`` from the primary source;
    all other exceptions propagate to the caller unchanged.

    Args:
        primary: The preferred entropy source (e.g. gRPC QRNG).
        fallback: The backup source to use when primary is unavailable
            (e.g. OsUrandomSource).
    """

    def __init__(
        self, primary: EntropySource, fallback: EntropySource
    ) -> None:
        """Initialize with primary and fallback sources.

        Args:
            primary: Tried first on every call.
            fallback: Used when primary raises EntropyUnavailableError.
        """
        self._primary = primary
        self._fallback = fallback

    def get_bytes(self, count: int) -> bytes:
        """Try primary source; use fallback on EntropyUnavailableError.

        Args:
            count: Number of bytes to return.

        Returns:
            Exactly *count* bytes from either primary or fallback.

        Raises:
            EntropyUnavailableError: If *both* sources fail.
        """
        try:
            return self._primary.get_bytes(count)
        except EntropyUnavailableError as exc:
            logger.warning(
                "Primary entropy source unavailable, using fallback: %s", exc
            )
            return self._fallback.get_bytes(count)

    def prefetch(self, count: int) -> None:
        """Prefetch on the primary source.

        Fallback sources typically don't need prefetching (they're fast
        by definition), so only the primary is prefetched.

        Args:
            count: Number of bytes to prepare.
        """
        self._primary.prefetch(count)

    def health_check(self) -> dict[str, Any]:
        """Report health of both primary and fallback sources.

        Returns:
            Dict with ``source``, ``healthy`` (true if either is healthy),
            ``primary``, and ``fallback`` sub-dicts.
        """
        primary_health = self._primary.health_check()
        fallback_health = self._fallback.health_check()
        return {
            "source": "fallback_wrapper",
            "healthy": primary_health["healthy"] or fallback_health["healthy"],
            "primary": primary_health,
            "fallback": fallback_health,
        }

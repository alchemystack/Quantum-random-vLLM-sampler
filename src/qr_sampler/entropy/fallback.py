"""Fallback entropy source — composition wrapper with transparent failover.

``FallbackEntropySource`` wraps a *primary* and a *fallback* source. When the
primary raises :class:`~qr_sampler.exceptions.EntropyUnavailableError`, the
wrapper transparently delegates to the fallback. **All other exceptions
propagate unchanged** — this is deliberate: only entropy-unavailability is a
recoverable condition.
"""

from __future__ import annotations

import logging
from typing import Any

from qr_sampler.entropy.base import EntropySource
from qr_sampler.exceptions import EntropyUnavailableError

logger = logging.getLogger("qr_sampler")


class FallbackEntropySource(EntropySource):
    """Composition wrapper: tries primary, falls back on ``EntropyUnavailableError``.

    Only catches ``EntropyUnavailableError``. All other exceptions propagate.
    Reports which source was actually used via :attr:`last_source_used`.

    Args:
        primary: The preferred entropy source.
        fallback: The source to use when the primary is unavailable.
    """

    def __init__(self, primary: EntropySource, fallback: EntropySource) -> None:
        self._primary = primary
        self._fallback = fallback
        self._last_source_used: str = primary.name

    @property
    def name(self) -> str:
        """Return a compound name: ``'<primary>+<fallback>'``."""
        return f"{self._primary.name}+{self._fallback.name}"

    @property
    def is_available(self) -> bool:
        """Returns ``True`` if either the primary or fallback is available."""
        return self._primary.is_available or self._fallback.is_available

    @property
    def last_source_used(self) -> str:
        """Name of the source that provided bytes on the last call."""
        return self._last_source_used

    def get_random_bytes(self, n: int) -> bytes:
        """Fetch bytes from the primary source, falling back if unavailable.

        Only ``EntropyUnavailableError`` triggers fallback. All other
        exceptions propagate to the caller unchanged.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes from the primary or fallback source.

        Raises:
            EntropyUnavailableError: If **both** primary and fallback fail.
        """
        try:
            data = self._primary.get_random_bytes(n)
            self._last_source_used = self._primary.name
            return data
        except EntropyUnavailableError:
            logger.warning(
                "Primary entropy source %r unavailable, falling back to %r",
                self._primary.name,
                self._fallback.name,
            )
            data = self._fallback.get_random_bytes(n)
            self._last_source_used = self._fallback.name
            return data

    def close(self) -> None:
        """Close both primary and fallback sources."""
        self._primary.close()
        self._fallback.close()

    def health_check(self) -> dict[str, Any]:
        """Return health status for both sources.

        Returns:
            Dictionary with overall health and individual source status.
        """
        primary_health = self._primary.health_check()
        fallback_health = self._fallback.health_check()
        return {
            "source": self.name,
            "healthy": self.is_available,
            "primary": primary_health,
            "fallback": fallback_health,
            "last_source_used": self._last_source_used,
        }

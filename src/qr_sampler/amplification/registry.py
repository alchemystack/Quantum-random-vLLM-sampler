"""Registry for signal amplifier implementations.

Uses a decorator pattern for registration, enabling both built-in and
third-party amplifiers to register themselves at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.amplification.base import SignalAmplifier


class AmplifierRegistry:
    """Registry mapping string names to SignalAmplifier classes.

    Built-in amplifiers register via the ``@AmplifierRegistry.register()``
    decorator. The ``build()`` class method instantiates the appropriate
    amplifier based on the config's ``signal_amplifier_type`` field.
    """

    _registry: ClassVar[dict[str, type[SignalAmplifier]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[SignalAmplifier]], type[SignalAmplifier]]:
        """Decorator that registers a SignalAmplifier class under *name*.

        Args:
            name: Identifier used in config ``signal_amplifier_type``.

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If *name* is already registered.
        """

        def decorator(klass: type[SignalAmplifier]) -> type[SignalAmplifier]:
            if name in cls._registry:
                raise ValueError(f"Amplifier '{name}' is already registered")
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[SignalAmplifier]:
        """Return the amplifier class registered under *name*.

        Args:
            name: Identifier to look up.

        Returns:
            The registered SignalAmplifier subclass.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(f"Unknown signal amplifier '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def build(cls, config: Any) -> SignalAmplifier:
        """Instantiate the amplifier specified by *config.signal_amplifier_type*.

        Args:
            config: A QRSamplerConfig (or compatible object) with a
                ``signal_amplifier_type`` attribute.

        Returns:
            A fully constructed SignalAmplifier instance.
        """
        klass = cls.get(config.signal_amplifier_type)
        return klass(config)  # type: ignore[call-arg]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of registered amplifier names."""
        return sorted(cls._registry)

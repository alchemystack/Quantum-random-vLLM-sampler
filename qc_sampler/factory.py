"""Registry-based factory for constructing pipeline components from config.

This module is the central wiring point: it maps string identifiers
(from ``QCSamplingConfig``) to concrete class constructors.  Adding a
new signal amplification algorithm, entropy source backend, or
temperature formula requires only:

    1. Write the new class (implementing the relevant ABC).
    2. Call the corresponding ``register_*()`` function.

No existing code needs modification — this is the Open/Closed Principle
in action.

Component construction:
    - ``build_entropy_source(config)`` — assembles the entropy source
      chain, including the ``FallbackEntropySource`` wrapper when the
      config's fallback mode is not ``"error"``.
    - ``build_signal_amplifier(config)`` — instantiates the configured
      amplifier.
    - ``build_temperature_strategy(config, vocab_size)`` — instantiates
      the configured temperature strategy.

Thread safety:
    The registries are module-level dicts populated at import time.
    They are read-only during normal operation.  ``register_*()`` is
    intended for startup-time or plugin-load-time use only.
"""

from __future__ import annotations

import logging
from typing import Any

from qc_sampler.config import QCSamplingConfig
from qc_sampler.entropy_source import (
    EntropySource,
    FallbackEntropySource,
    GrpcEntropySource,
    MockUniformSource,
    OsUrandomSource,
)
from qc_sampler.exceptions import ConfigValidationError
from qc_sampler.signal_amplifier import SignalAmplifier, ZScoreMeanAmplifier
from qc_sampler.temperature_strategy import (
    EDTTemperatureStrategy,
    FixedTemperatureStrategy,
    TemperatureStrategy,
)

logger = logging.getLogger("qc_sampler")

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Maps config string identifier → class for each component type.
# Populated with defaults at the bottom of this module.

_ENTROPY_SOURCE_REGISTRY: dict[str, type[EntropySource]] = {}
_SIGNAL_AMPLIFIER_REGISTRY: dict[str, type[SignalAmplifier]] = {}
_TEMPERATURE_STRATEGY_REGISTRY: dict[str, type[TemperatureStrategy]] = {}


# ---------------------------------------------------------------------------
# Registration functions
# ---------------------------------------------------------------------------


def register_entropy_source(name: str, cls: type[EntropySource]) -> None:
    """Register an entropy source class under a string identifier.

    Args:
        name: The identifier used in ``QCSamplingConfig`` to select
            this source (e.g. ``"grpc"``, ``"os_urandom"``).
        cls: The ``EntropySource`` subclass to instantiate.
    """
    _ENTROPY_SOURCE_REGISTRY[name] = cls


def register_signal_amplifier(name: str, cls: type[SignalAmplifier]) -> None:
    """Register a signal amplifier class under a string identifier.

    Args:
        name: The identifier used in ``config.signal_amplifier_type``
            (e.g. ``"zscore_mean"``).
        cls: The ``SignalAmplifier`` subclass to instantiate.
    """
    _SIGNAL_AMPLIFIER_REGISTRY[name] = cls


def register_temperature_strategy(
    name: str, cls: type[TemperatureStrategy]
) -> None:
    """Register a temperature strategy class under a string identifier.

    Args:
        name: The identifier used in ``config.temperature_strategy``
            (e.g. ``"fixed"``, ``"edt"``).
        cls: The ``TemperatureStrategy`` subclass to instantiate.
    """
    _TEMPERATURE_STRATEGY_REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_entropy_source(config: QCSamplingConfig) -> EntropySource:
    """Build the entropy source chain from config.

    The primary source is always a ``GrpcEntropySource``.  If
    ``config.qrng_fallback_mode`` is not ``"error"``, the primary is
    wrapped in a ``FallbackEntropySource`` with the appropriate
    fallback backend.

    When the fallback mode itself is a registered entropy source name
    (e.g. ``"os_urandom"``, ``"mock_uniform"``), that source is used
    as the fallback.  When fallback mode is ``"error"``, no fallback
    is applied and ``EntropyUnavailableError`` propagates directly.

    Args:
        config: Configuration with entropy source settings.

    Returns:
        A ready-to-use ``EntropySource`` (possibly wrapped in a
        ``FallbackEntropySource``).

    Raises:
        ConfigValidationError: If the fallback mode is not ``"error"``
            and is not a registered entropy source name.
    """
    primary = GrpcEntropySource(config)

    if config.qrng_fallback_mode == "error":
        return primary

    fallback_name = config.qrng_fallback_mode
    if fallback_name not in _ENTROPY_SOURCE_REGISTRY:
        raise ConfigValidationError(
            f"Unknown fallback mode '{fallback_name}'. "
            f"Registered entropy sources: {sorted(_ENTROPY_SOURCE_REGISTRY.keys())}. "
            f"Use 'error' to disable fallback."
        )

    fallback_cls = _ENTROPY_SOURCE_REGISTRY[fallback_name]

    # Instantiate the fallback source.  Sources that need config get it;
    # simple sources (OsUrandomSource, MockUniformSource) take no args
    # or optional args.
    if fallback_cls is GrpcEntropySource:
        fallback = fallback_cls(config)
    elif fallback_cls is MockUniformSource:
        fallback = fallback_cls(mean=config.population_mean)
    else:
        # OsUrandomSource and any future simple sources.
        fallback = fallback_cls()

    logger.info(
        "Entropy source: gRPC primary → %s fallback", fallback_name
    )
    return FallbackEntropySource(primary=primary, fallback=fallback)


def build_signal_amplifier(config: QCSamplingConfig) -> SignalAmplifier:
    """Instantiate the signal amplifier specified by config.

    Looks up ``config.signal_amplifier_type`` in the registry and
    constructs the amplifier with the given config.

    Args:
        config: Configuration with signal amplification parameters.

    Returns:
        A ready-to-use ``SignalAmplifier``.

    Raises:
        ConfigValidationError: If the amplifier type is not registered.
    """
    amp_type = config.signal_amplifier_type
    if amp_type not in _SIGNAL_AMPLIFIER_REGISTRY:
        raise ConfigValidationError(
            f"Unknown signal amplifier type '{amp_type}'. "
            f"Registered types: {sorted(_SIGNAL_AMPLIFIER_REGISTRY.keys())}"
        )

    cls = _SIGNAL_AMPLIFIER_REGISTRY[amp_type]
    return cls(config)


def build_temperature_strategy(
    config: QCSamplingConfig, vocab_size: int
) -> TemperatureStrategy:
    """Instantiate the temperature strategy specified by config.

    Looks up ``config.temperature_strategy`` in the registry.
    Strategies that require ``vocab_size`` (like EDT) receive it as
    a constructor argument; simple strategies ignore it.

    Args:
        config: Configuration with temperature parameters.
        vocab_size: Model vocabulary size, needed by EDT for entropy
            normalization.

    Returns:
        A ready-to-use ``TemperatureStrategy``.

    Raises:
        ConfigValidationError: If the strategy name is not registered.
    """
    strategy_name = config.temperature_strategy
    if strategy_name not in _TEMPERATURE_STRATEGY_REGISTRY:
        raise ConfigValidationError(
            f"Unknown temperature strategy '{strategy_name}'. "
            f"Registered strategies: {sorted(_TEMPERATURE_STRATEGY_REGISTRY.keys())}"
        )

    cls = _TEMPERATURE_STRATEGY_REGISTRY[strategy_name]

    # EDTTemperatureStrategy requires vocab_size in its constructor.
    # FixedTemperatureStrategy does not.  We use a try/except approach
    # rather than isinstance checks so that future strategies that need
    # vocab_size also "just work".
    try:
        return cls(vocab_size=vocab_size)
    except TypeError:
        # Constructor doesn't accept vocab_size — simple strategy.
        return cls()


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------

register_entropy_source("grpc", GrpcEntropySource)
register_entropy_source("os_urandom", OsUrandomSource)
register_entropy_source("mock_uniform", MockUniformSource)

register_signal_amplifier("zscore_mean", ZScoreMeanAmplifier)

register_temperature_strategy("fixed", FixedTemperatureStrategy)
register_temperature_strategy("edt", EDTTemperatureStrategy)

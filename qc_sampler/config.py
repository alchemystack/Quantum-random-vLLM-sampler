"""Configuration for quantum consciousness sampling.

This module defines the single source of truth for every tunable parameter
in the system. All algorithm constants, thresholds, and defaults live here
as fields on QCSamplingConfig. No magic numbers should exist anywhere else
in the codebase.

Configuration resolution chain:
    Environment variables (QC_*)
        → loaded at __init__ time via load_config_from_env()
    QCSamplingConfig (defaults)
        → merged per-request via resolve_config()
    extra_args (qc_* prefix stripped)
        → Resolved QCSamplingConfig (per-request)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, get_type_hints

from qc_sampler.exceptions import ConfigValidationError

logger = logging.getLogger("qc_sampler")

# Standard deviation of a continuous uniform distribution on [0, 255]:
# std = (b - a) / sqrt(12) = 255 / sqrt(12).
# This matches the spec default of 73.6116... and the env var example.
_DEFAULT_POPULATION_STD = 73.61215932167728

# Fields that can be overridden per-request via extra_args.
# Infrastructure fields (gRPC address, timeout, prefetch, retry, fallback)
# are intentionally excluded — they are set at server startup only.
_PER_REQUEST_FIELDS: frozenset[str] = frozenset({
    "signal_amplifier_type",
    "sample_count",
    "population_mean",
    "population_std",
    "uniform_clamp_epsilon",
    "temperature_strategy",
    "fixed_temperature",
    "edt_min_temp",
    "edt_max_temp",
    "edt_base_temp",
    "edt_exponent",
    "top_k",
    "top_p",
    "log_level",
    "diagnostic_mode",
})

# Prefix used for per-request extra_args keys.
_EXTRA_ARGS_PREFIX = "qc_"

# Prefix used for environment variable names.
_ENV_VAR_PREFIX = "QC_"


@dataclass(frozen=True)
class QCSamplingConfig:
    """All parameters for quantum consciousness sampling.

    This config is the single source of truth for every tunable value
    in the system. If a number appears in the algorithm, it must be a
    field here.

    Fields are grouped by subsystem. Infrastructure fields (entropy source
    connection settings) are NOT overridable per-request. Algorithm and
    logging fields ARE overridable per-request via SamplingParams.extra_args
    with a "qc_" prefix.

    Defaults:
        qrng_server_address: "localhost:50051"
        qrng_timeout_ms: 5000.0
        qrng_prefetch_enabled: True
        qrng_retry_count: 2
        qrng_fallback_mode: "os_urandom"
        signal_amplifier_type: "zscore_mean"
        sample_count: 20480
        population_mean: 127.5
        population_std: 73.6116... (exact: sqrt((256^2 - 1) / 12))
        uniform_clamp_epsilon: 1e-10
        temperature_strategy: "fixed"
        fixed_temperature: 0.7
        edt_min_temp: 0.1
        edt_max_temp: 2.0
        edt_base_temp: 0.8
        edt_exponent: 0.5
        top_k: 50
        top_p: 0.9
        log_level: "summary"
        diagnostic_mode: False
    """

    # --- Entropy source (infrastructure, NOT per-request overridable) ---
    qrng_server_address: str = "localhost:50051"
    qrng_timeout_ms: float = 5000.0
    qrng_prefetch_enabled: bool = True
    qrng_retry_count: int = 2
    qrng_fallback_mode: str = "os_urandom"

    # --- Signal amplification ---
    signal_amplifier_type: str = "zscore_mean"
    sample_count: int = 20480
    population_mean: float = 127.5
    population_std: float = _DEFAULT_POPULATION_STD
    uniform_clamp_epsilon: float = 1e-10

    # --- Temperature strategy ---
    temperature_strategy: str = "fixed"
    fixed_temperature: float = 0.7
    edt_min_temp: float = 0.1
    edt_max_temp: float = 2.0
    edt_base_temp: float = 0.8
    edt_exponent: float = 0.5

    # --- Token selection ---
    top_k: int = 50
    top_p: float = 0.9

    # --- Logging ---
    log_level: str = "summary"
    diagnostic_mode: bool = False


def _parse_env_value(value_str: str, field_type: type) -> Any:
    """Parse a string environment variable value into the target field type.

    Args:
        value_str: The raw string from the environment variable.
        field_type: The Python type of the target dataclass field.

    Returns:
        The parsed value coerced to field_type.

    Raises:
        ConfigValidationError: If the string cannot be parsed into the
            target type.
    """
    if field_type is bool:
        lowered = value_str.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ConfigValidationError(
            f"Cannot parse '{value_str}' as bool; "
            f"expected one of: true, false, 1, 0, yes, no"
        )
    if field_type is int:
        try:
            return int(value_str)
        except ValueError:
            raise ConfigValidationError(
                f"Cannot parse '{value_str}' as int"
            ) from None
    if field_type is float:
        try:
            return float(value_str)
        except ValueError:
            raise ConfigValidationError(
                f"Cannot parse '{value_str}' as float"
            ) from None
    # str passes through unchanged
    return value_str


def _coerce_value(field_name: str, value: Any, field_type: type) -> Any:
    """Coerce a runtime value to the expected field type.

    Performs safe type coercion for extra_args values which may arrive
    as strings (from JSON) or as their native types.

    Args:
        field_name: Name of the config field (for error messages).
        value: The raw value from extra_args.
        field_type: The expected Python type of the field.

    Returns:
        The value coerced to field_type.

    Raises:
        ConfigValidationError: If the value cannot be coerced.
    """
    # Already the right type — fast path.
    if isinstance(value, field_type):
        return value

    # Bool check must come before int/float because bool is a subclass of int.
    if field_type is bool:
        if isinstance(value, str):
            return _parse_env_value(value, bool)
        raise ConfigValidationError(
            f"Field '{field_name}': expected bool, got {type(value).__name__} "
            f"(value: {value!r})"
        )

    if field_type is int:
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ConfigValidationError(
            f"Field '{field_name}': expected int, got {type(value).__name__} "
            f"(value: {value!r})"
        )

    if field_type is float:
        if isinstance(value, (int, str)):
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        raise ConfigValidationError(
            f"Field '{field_name}': expected float, got {type(value).__name__} "
            f"(value: {value!r})"
        )

    if field_type is str:
        if not isinstance(value, str):
            raise ConfigValidationError(
                f"Field '{field_name}': expected str, got {type(value).__name__} "
                f"(value: {value!r})"
            )
        return value

    raise ConfigValidationError(
        f"Field '{field_name}': unsupported type {field_type.__name__}"
    )


def _get_field_types() -> dict[str, type]:
    """Return a mapping of field name → Python type for QCSamplingConfig.

    Uses get_type_hints to resolve any forward references, and caches
    the result on the function object for efficiency.
    """
    if not hasattr(_get_field_types, "_cache"):
        _get_field_types._cache = get_type_hints(QCSamplingConfig)
    return _get_field_types._cache


def load_config_from_env() -> QCSamplingConfig:
    """Load QCSamplingConfig from environment variables.

    Each field maps to an env var with the QC_ prefix and uppercase
    field name. For example, ``qrng_server_address`` maps to
    ``QC_QRNG_SERVER_ADDRESS``.

    Env vars that are not set use the dataclass default values.
    Invalid values raise ConfigValidationError with a descriptive
    message.

    Returns:
        A QCSamplingConfig instance with values from the environment
        overlaid on top of dataclass defaults.
    """
    field_types = _get_field_types()
    overrides: dict[str, Any] = {}

    for f in fields(QCSamplingConfig):
        env_key = f"{_ENV_VAR_PREFIX}{f.name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                overrides[f.name] = _parse_env_value(env_val, field_types[f.name])
            except ConfigValidationError as exc:
                raise ConfigValidationError(
                    f"Environment variable {env_key}: {exc}"
                ) from None

    if overrides:
        return QCSamplingConfig(**{
            f.name: overrides.get(f.name, getattr(QCSamplingConfig, f.name, f.default))
            for f in fields(QCSamplingConfig)
        })
    return QCSamplingConfig()


def validate_extra_args(extra_args: dict[str, Any] | None) -> None:
    """Validate all qc_* keys in extra_args for type correctness.

    Checks that:
    - The field name (after stripping the qc_ prefix) exists in
      QCSamplingConfig.
    - The field is in the per-request overridable set.
    - The value can be coerced to the expected type.

    Unknown keys (not starting with qc_) are silently ignored — they
    may belong to other plugins.

    Args:
        extra_args: The extra_args dict from SamplingParams. May be None.

    Raises:
        ConfigValidationError: If any qc_* key has an invalid field name,
            is not per-request overridable, or has a value of the wrong type.
    """
    if not extra_args:
        return

    field_types = _get_field_types()

    for key, value in extra_args.items():
        if not key.startswith(_EXTRA_ARGS_PREFIX):
            continue

        field_name = key[len(_EXTRA_ARGS_PREFIX):]

        if field_name not in field_types:
            raise ConfigValidationError(
                f"Unknown config field '{field_name}' "
                f"(from extra_args key '{key}')"
            )

        if field_name not in _PER_REQUEST_FIELDS:
            raise ConfigValidationError(
                f"Field '{field_name}' is not overridable per-request "
                f"(from extra_args key '{key}'). "
                f"Infrastructure fields can only be set via environment variables."
            )

        # Validate type by attempting coercion (will raise on failure).
        _coerce_value(field_name, value, field_types[field_name])


def resolve_config(
    defaults: QCSamplingConfig,
    extra_args: dict[str, Any] | None,
) -> QCSamplingConfig:
    """Merge per-request extra_args over defaults.

    Creates a new QCSamplingConfig instance where per-request overridable
    fields from extra_args (with qc_ prefix stripped) take precedence
    over the defaults. Non-overridable fields always come from defaults.

    Unknown keys (not starting with qc_) are silently ignored.
    Keys for non-overridable fields are silently ignored (they are
    validated separately by validate_extra_args).

    Args:
        defaults: The server-level default configuration.
        extra_args: Per-request overrides from SamplingParams.extra_args.
            May be None or empty.

    Returns:
        A new QCSamplingConfig instance with overrides applied.
    """
    if not extra_args:
        return defaults

    field_types = _get_field_types()
    overrides: dict[str, Any] = {}

    for key, value in extra_args.items():
        if not key.startswith(_EXTRA_ARGS_PREFIX):
            continue

        field_name = key[len(_EXTRA_ARGS_PREFIX):]

        # Skip unknown fields or non-overridable fields silently.
        if field_name not in field_types or field_name not in _PER_REQUEST_FIELDS:
            continue

        overrides[field_name] = _coerce_value(
            field_name, value, field_types[field_name]
        )

    if not overrides:
        return defaults

    # Build new config: start from defaults, overlay overrides.
    merged = {f.name: getattr(defaults, f.name) for f in fields(QCSamplingConfig)}
    merged.update(overrides)
    return QCSamplingConfig(**merged)

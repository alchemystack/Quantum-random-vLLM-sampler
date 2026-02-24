"""Tests for qc_sampler.config.

Covers:
- Default values match spec
- resolve_config merge logic (extra_args override defaults)
- Non-overridable infrastructure fields ignored in resolve_config
- Unknown extra_args keys ignored (no crash)
- validate_extra_args type checking (wrong types raise ConfigValidationError)
- validate_extra_args rejects unknown qc_ fields
- validate_extra_args rejects non-per-request fields
- Environment variable loading (monkeypatch os.environ)
- Frozen immutability of QCSamplingConfig
"""

import os
from dataclasses import fields

import pytest

from qc_sampler.config import (
    QCSamplingConfig,
    _PER_REQUEST_FIELDS,
    _DEFAULT_POPULATION_STD,
    load_config_from_env,
    resolve_config,
    validate_extra_args,
)
from qc_sampler.exceptions import ConfigValidationError


class TestQCSamplingConfigDefaults:
    """Verify that default values match the spec."""

    def test_entropy_source_defaults(self) -> None:
        cfg = QCSamplingConfig()
        assert cfg.qrng_server_address == "localhost:50051"
        assert cfg.qrng_timeout_ms == 5000.0
        assert cfg.qrng_prefetch_enabled is True
        assert cfg.qrng_retry_count == 2
        assert cfg.qrng_fallback_mode == "os_urandom"

    def test_signal_amplification_defaults(self) -> None:
        cfg = QCSamplingConfig()
        assert cfg.signal_amplifier_type == "zscore_mean"
        assert cfg.sample_count == 20480
        assert cfg.population_mean == 127.5
        assert cfg.population_std == _DEFAULT_POPULATION_STD
        assert cfg.uniform_clamp_epsilon == 1e-10

    def test_temperature_defaults(self) -> None:
        cfg = QCSamplingConfig()
        assert cfg.temperature_strategy == "fixed"
        assert cfg.fixed_temperature == 0.7
        assert cfg.edt_min_temp == 0.1
        assert cfg.edt_max_temp == 2.0
        assert cfg.edt_base_temp == 0.8
        assert cfg.edt_exponent == 0.5

    def test_token_selection_defaults(self) -> None:
        cfg = QCSamplingConfig()
        assert cfg.top_k == 50
        assert cfg.top_p == 0.9

    def test_logging_defaults(self) -> None:
        cfg = QCSamplingConfig()
        assert cfg.log_level == "summary"
        assert cfg.diagnostic_mode is False

    def test_population_std_precision(self) -> None:
        """Verify population_std matches 255 / sqrt(12) (continuous uniform on [0,255])."""
        import math
        expected = 255 / math.sqrt(12)
        assert QCSamplingConfig().population_std == pytest.approx(expected)

    def test_frozen_immutability(self) -> None:
        """Config instances must be immutable (frozen dataclass)."""
        cfg = QCSamplingConfig()
        with pytest.raises(AttributeError):
            cfg.top_k = 100  # type: ignore[misc]


class TestResolveConfig:
    """Tests for resolve_config merge logic."""

    def test_none_extra_args_returns_defaults(self) -> None:
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, None)
        assert result is defaults

    def test_empty_extra_args_returns_defaults(self) -> None:
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {})
        assert result is defaults

    def test_override_single_field(self) -> None:
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {"qc_top_k": 100})
        assert result.top_k == 100
        # All other fields unchanged
        assert result.top_p == defaults.top_p
        assert result.fixed_temperature == defaults.fixed_temperature

    def test_override_multiple_fields(self) -> None:
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {
            "qc_top_k": 100,
            "qc_top_p": 0.95,
            "qc_temperature_strategy": "edt",
            "qc_edt_exponent": 0.8,
        })
        assert result.top_k == 100
        assert result.top_p == 0.95
        assert result.temperature_strategy == "edt"
        assert result.edt_exponent == 0.8

    def test_returns_new_instance(self) -> None:
        """resolve_config must return a new config, not mutate defaults."""
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {"qc_top_k": 999})
        assert result is not defaults
        assert result.top_k == 999
        assert defaults.top_k == 50  # original unchanged

    def test_infrastructure_fields_not_overridden(self) -> None:
        """Non-per-request fields in extra_args are silently ignored."""
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {
            "qc_qrng_server_address": "10.0.0.1:9999",
            "qc_qrng_timeout_ms": 1000.0,
            "qc_top_k": 100,
        })
        # Infrastructure fields keep defaults
        assert result.qrng_server_address == "localhost:50051"
        assert result.qrng_timeout_ms == 5000.0
        # Per-request field is overridden
        assert result.top_k == 100

    def test_unknown_keys_ignored(self) -> None:
        """Keys without qc_ prefix or with unknown field names are ignored."""
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {
            "some_other_plugin_key": 42,
            "qc_nonexistent_field": "whatever",
            "qc_top_k": 100,
        })
        assert result.top_k == 100

    def test_type_coercion_int_from_float(self) -> None:
        """An int field should accept a float that is exactly integral."""
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {"qc_top_k": 100.0})
        assert result.top_k == 100
        assert isinstance(result.top_k, int)

    def test_type_coercion_float_from_int(self) -> None:
        """A float field should accept an int value."""
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {"qc_fixed_temperature": 1})
        assert result.fixed_temperature == 1.0
        assert isinstance(result.fixed_temperature, float)

    def test_type_coercion_bool_from_string(self) -> None:
        """A bool field should accept 'true'/'false' strings."""
        defaults = QCSamplingConfig()
        result = resolve_config(defaults, {"qc_diagnostic_mode": "true"})
        assert result.diagnostic_mode is True

    def test_type_error_raises_config_validation_error(self) -> None:
        """Passing wrong type should raise ConfigValidationError."""
        defaults = QCSamplingConfig()
        with pytest.raises(ConfigValidationError, match="top_k"):
            resolve_config(defaults, {"qc_top_k": "not_a_number"})

    def test_custom_defaults_preserved(self) -> None:
        """When defaults differ from class defaults, they're preserved."""
        custom = QCSamplingConfig(top_k=200, top_p=0.8)
        result = resolve_config(custom, {"qc_fixed_temperature": 1.5})
        assert result.top_k == 200
        assert result.top_p == 0.8
        assert result.fixed_temperature == 1.5


class TestValidateExtraArgs:
    """Tests for validate_extra_args."""

    def test_none_args_ok(self) -> None:
        """None extra_args should not raise."""
        validate_extra_args(None)

    def test_empty_args_ok(self) -> None:
        """Empty dict should not raise."""
        validate_extra_args({})

    def test_valid_args_ok(self) -> None:
        """Valid qc_ args should not raise."""
        validate_extra_args({
            "qc_top_k": 100,
            "qc_temperature_strategy": "edt",
            "qc_diagnostic_mode": True,
        })

    def test_non_qc_keys_ignored(self) -> None:
        """Keys without qc_ prefix are silently ignored."""
        validate_extra_args({
            "other_plugin_setting": "whatever",
            "temperature": 0.5,
        })

    def test_unknown_qc_field_raises(self) -> None:
        """qc_ key with unknown field name raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Unknown config field"):
            validate_extra_args({"qc_nonexistent_field": 42})

    def test_non_overridable_field_raises(self) -> None:
        """qc_ key for infrastructure field raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="not overridable per-request"):
            validate_extra_args({"qc_qrng_server_address": "10.0.0.1:9999"})

    def test_wrong_type_string_for_float_raises(self) -> None:
        """String that can't be parsed as float raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="fixed_temperature"):
            validate_extra_args({"qc_fixed_temperature": "not_a_number"})

    def test_wrong_type_string_for_int_raises(self) -> None:
        """Non-numeric string for int field raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="top_k"):
            validate_extra_args({"qc_top_k": "abc"})

    def test_wrong_type_list_for_int_raises(self) -> None:
        """List value for int field raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="top_k"):
            validate_extra_args({"qc_top_k": [1, 2, 3]})

    def test_bool_field_rejects_non_bool_non_string(self) -> None:
        """Non-bool, non-string value for bool field raises."""
        with pytest.raises(ConfigValidationError, match="diagnostic_mode"):
            validate_extra_args({"qc_diagnostic_mode": 42})

    def test_valid_string_coercions_pass(self) -> None:
        """String values that can be coerced should pass validation."""
        validate_extra_args({
            "qc_top_k": "100",
            "qc_fixed_temperature": "0.5",
            "qc_diagnostic_mode": "false",
        })


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env with monkeypatched environment."""

    def test_no_env_vars_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no QC_ env vars, all defaults are used."""
        # Clear any QC_ vars that might exist
        for key in list(os.environ):
            if key.startswith("QC_"):
                monkeypatch.delenv(key, raising=False)
        cfg = load_config_from_env()
        assert cfg == QCSamplingConfig()

    def test_string_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_QRNG_SERVER_ADDRESS", "10.0.1.5:50051")
        cfg = load_config_from_env()
        assert cfg.qrng_server_address == "10.0.1.5:50051"

    def test_int_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_TOP_K", "100")
        cfg = load_config_from_env()
        assert cfg.top_k == 100
        assert isinstance(cfg.top_k, int)

    def test_float_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_FIXED_TEMPERATURE", "1.5")
        cfg = load_config_from_env()
        assert cfg.fixed_temperature == 1.5

    def test_bool_env_var_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_DIAGNOSTIC_MODE", "true")
        cfg = load_config_from_env()
        assert cfg.diagnostic_mode is True

    def test_bool_env_var_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_QRNG_PREFETCH_ENABLED", "false")
        cfg = load_config_from_env()
        assert cfg.qrng_prefetch_enabled is False

    def test_bool_env_var_numeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_DIAGNOSTIC_MODE", "1")
        cfg = load_config_from_env()
        assert cfg.diagnostic_mode is True

    def test_invalid_int_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_TOP_K", "not_a_number")
        with pytest.raises(ConfigValidationError, match="QC_TOP_K"):
            load_config_from_env()

    def test_invalid_float_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_FIXED_TEMPERATURE", "hot")
        with pytest.raises(ConfigValidationError, match="QC_FIXED_TEMPERATURE"):
            load_config_from_env()

    def test_invalid_bool_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_DIAGNOSTIC_MODE", "maybe")
        with pytest.raises(ConfigValidationError, match="QC_DIAGNOSTIC_MODE"):
            load_config_from_env()

    def test_multiple_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QC_TOP_K", "200")
        monkeypatch.setenv("QC_TOP_P", "0.95")
        monkeypatch.setenv("QC_TEMPERATURE_STRATEGY", "edt")
        cfg = load_config_from_env()
        assert cfg.top_k == 200
        assert cfg.top_p == 0.95
        assert cfg.temperature_strategy == "edt"
        # Unchanged defaults
        assert cfg.fixed_temperature == 0.7

    def test_scientific_notation_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scientific notation strings should parse correctly for floats."""
        monkeypatch.setenv("QC_UNIFORM_CLAMP_EPSILON", "1e-8")
        cfg = load_config_from_env()
        assert cfg.uniform_clamp_epsilon == 1e-8


class TestPerRequestFields:
    """Tests for the _PER_REQUEST_FIELDS set."""

    def test_infrastructure_fields_excluded(self) -> None:
        """Infrastructure fields must NOT be in the per-request set."""
        infrastructure = {
            "qrng_server_address",
            "qrng_timeout_ms",
            "qrng_prefetch_enabled",
            "qrng_retry_count",
            "qrng_fallback_mode",
        }
        assert infrastructure.isdisjoint(_PER_REQUEST_FIELDS)

    def test_algorithm_fields_included(self) -> None:
        """All algorithm and logging fields must be per-request overridable."""
        expected = {
            "signal_amplifier_type", "sample_count", "population_mean",
            "population_std", "uniform_clamp_epsilon", "temperature_strategy",
            "fixed_temperature", "edt_min_temp", "edt_max_temp",
            "edt_base_temp", "edt_exponent", "top_k", "top_p",
            "log_level", "diagnostic_mode",
        }
        assert _PER_REQUEST_FIELDS == expected

    def test_all_per_request_fields_exist_in_config(self) -> None:
        """Every field in _PER_REQUEST_FIELDS must be a real config field."""
        config_fields = {f.name for f in fields(QCSamplingConfig)}
        for field_name in _PER_REQUEST_FIELDS:
            assert field_name in config_fields, (
                f"'{field_name}' in _PER_REQUEST_FIELDS but not in QCSamplingConfig"
            )

    def test_every_config_field_is_classified(self) -> None:
        """Every config field is either per-request or infrastructure."""
        infrastructure = {
            "qrng_server_address",
            "qrng_timeout_ms",
            "qrng_prefetch_enabled",
            "qrng_retry_count",
            "qrng_fallback_mode",
        }
        config_fields = {f.name for f in fields(QCSamplingConfig)}
        classified = _PER_REQUEST_FIELDS | infrastructure
        assert config_fields == classified, (
            f"Unclassified fields: {config_fields - classified}"
        )

"""Integration tests for QuantumConsciousnessProcessor.

These tests exercise the full pipeline from logits through to token
selection, using MockUniformSource so no real QRNG server is needed.
The processor is tested with numpy arrays standing in for torch tensors,
relying on the numpy fallback path in processor.apply().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from qc_sampler.config import QCSamplingConfig
from qc_sampler.entropy_source import MockUniformSource, OsUrandomSource
from qc_sampler.exceptions import ConfigValidationError
from qc_sampler.factory import (
    build_signal_amplifier,
    build_temperature_strategy,
)
from qc_sampler.processor import QuantumConsciousnessProcessor
from qc_sampler.sampling_logger import SamplingLogger


# ---------------------------------------------------------------------------
# Helper stubs to simulate vLLM types without importing vLLM
# ---------------------------------------------------------------------------


@dataclass
class FakeSamplingParams:
    """Stand-in for vllm.SamplingParams in tests."""

    extra_args: dict[str, Any] | None = None


@dataclass
class FakeBatchUpdate:
    """Stand-in for vllm.v1.sample.logits_processor.BatchUpdate."""

    removed: list[int] | None = None
    moved: list[tuple[int, int, str]] | None = None
    added: list[tuple[int, FakeSamplingParams, list[int]]] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


# ---------------------------------------------------------------------------
# Processor construction helpers
# ---------------------------------------------------------------------------


def _make_processor(
    vocab_size: int = 100,
    config_overrides: dict[str, Any] | None = None,
) -> QuantumConsciousnessProcessor:
    """Build a processor with mocked entropy source for testing.

    Uses os_urandom fallback mode so no gRPC connection is needed.
    The fallback immediately kicks in because GrpcEntropySource will
    fail to connect.

    Args:
        vocab_size: Simulated vocabulary size.
        config_overrides: Env var overrides (QC_* keys).

    Returns:
        A configured QuantumConsciousnessProcessor.
    """
    env = {
        "QC_QRNG_FALLBACK_MODE": "os_urandom",
        "QC_LOG_LEVEL": "none",
        "QC_QRNG_PREFETCH_ENABLED": "false",
    }
    if config_overrides:
        env.update(config_overrides)

    with patch.dict("os.environ", env, clear=False):
        proc = QuantumConsciousnessProcessor()
        proc._vocab_size = vocab_size
    return proc


def _make_logits(batch_size: int, vocab_size: int, seed: int = 42) -> np.ndarray:
    """Generate a batch of random logits as a 2D numpy array.

    In the real processor, this would be a torch.Tensor. The processor's
    numpy fallback path handles plain numpy arrays identically.

    Args:
        batch_size: Number of rows (requests in the batch).
        vocab_size: Number of columns (vocabulary size).
        seed: RNG seed for reproducibility.

    Returns:
        A (batch_size, vocab_size) float64 array.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch_size, vocab_size)).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests: basic processor lifecycle
# ---------------------------------------------------------------------------


class TestProcessorInit:
    """Test processor construction and configuration loading."""

    def test_init_with_defaults(self) -> None:
        """Processor initializes with default config from env."""
        proc = _make_processor()
        assert proc._vocab_size == 100
        assert proc._default_config.qrng_fallback_mode == "os_urandom"

    def test_init_with_env_overrides(self) -> None:
        """Processor picks up QC_* env vars."""
        proc = _make_processor(config_overrides={
            "QC_FIXED_TEMPERATURE": "0.9",
            "QC_TOP_K": "100",
        })
        assert proc._default_config.fixed_temperature == 0.9
        assert proc._default_config.top_k == 100


class TestValidateParams:
    """Test the classmethod that validates SamplingParams.extra_args."""

    def test_valid_extra_args(self) -> None:
        """Valid qc_* args pass validation."""
        params = FakeSamplingParams(extra_args={
            "qc_temperature_strategy": "fixed",
            "qc_top_k": 50,
        })
        QuantumConsciousnessProcessor.validate_params(params)

    def test_invalid_type_raises_value_error(self) -> None:
        """Invalid type in extra_args raises ValueError."""
        params = FakeSamplingParams(extra_args={
            "qc_top_k": "not_a_number",
        })
        with pytest.raises(ValueError, match="top_k"):
            QuantumConsciousnessProcessor.validate_params(params)

    def test_non_overridable_field_raises_value_error(self) -> None:
        """Infrastructure fields in extra_args raise ValueError."""
        params = FakeSamplingParams(extra_args={
            "qc_qrng_server_address": "other:50051",
        })
        with pytest.raises(ValueError, match="not overridable"):
            QuantumConsciousnessProcessor.validate_params(params)

    def test_no_extra_args_passes(self) -> None:
        """Params with no extra_args are fine."""
        params = FakeSamplingParams(extra_args=None)
        QuantumConsciousnessProcessor.validate_params(params)

    def test_non_qc_args_ignored(self) -> None:
        """Non-qc_* args are ignored (belong to other plugins)."""
        params = FakeSamplingParams(extra_args={
            "other_plugin_key": "value",
        })
        QuantumConsciousnessProcessor.validate_params(params)


class TestIsArgmaxInvariant:
    """Test the is_argmax_invariant property."""

    def test_returns_false(self) -> None:
        """Processor must return False (it changes token selection)."""
        proc = _make_processor()
        assert proc.is_argmax_invariant() is False


# ---------------------------------------------------------------------------
# Tests: update_state
# ---------------------------------------------------------------------------


class TestUpdateState:
    """Test batch update processing (add/remove/move requests)."""

    def test_add_request_stores_config(self) -> None:
        """Adding a request stores its resolved config."""
        proc = _make_processor()
        params = FakeSamplingParams(extra_args={
            "qc_top_k": 100,
            "qc_fixed_temperature": 0.5,
        })
        update = FakeBatchUpdate(added=[(0, params, [])])
        proc.update_state(update)

        assert 0 in proc._request_configs
        assert proc._request_configs[0].top_k == 100
        assert proc._request_configs[0].fixed_temperature == 0.5

    def test_remove_request_clears_config(self) -> None:
        """Removing a request clears its cached state."""
        proc = _make_processor()
        # First add
        params = FakeSamplingParams(extra_args={"qc_top_k": 100})
        proc.update_state(FakeBatchUpdate(added=[(0, params, [])]))
        assert 0 in proc._request_configs

        # Then remove
        proc.update_state(FakeBatchUpdate(removed=[0]))
        assert 0 not in proc._request_configs

    def test_move_request_updates_index(self) -> None:
        """Moving a request transfers its config to the new index."""
        proc = _make_processor()
        params = FakeSamplingParams(extra_args={"qc_top_k": 77})
        proc.update_state(FakeBatchUpdate(added=[(0, params, [])]))

        proc.update_state(FakeBatchUpdate(moved=[(0, 5, "forward")]))
        assert 0 not in proc._request_configs
        assert 5 in proc._request_configs
        assert proc._request_configs[5].top_k == 77

    def test_add_with_different_strategy_caches_component(self) -> None:
        """Adding a request with a different strategy caches it."""
        proc = _make_processor()
        # Default is "fixed"; add a request with "edt"
        params = FakeSamplingParams(extra_args={
            "qc_temperature_strategy": "edt",
        })
        proc.update_state(FakeBatchUpdate(added=[(0, params, [])]))
        assert 0 in proc._request_temp_strategies

    def test_add_with_same_strategy_no_cache(self) -> None:
        """Adding a request with the default strategy doesn't cache it."""
        proc = _make_processor()
        params = FakeSamplingParams(extra_args={
            "qc_top_k": 100,
        })
        proc.update_state(FakeBatchUpdate(added=[(0, params, [])]))
        assert 0 not in proc._request_temp_strategies
        assert 0 not in proc._request_amplifiers

    def test_none_batch_update_is_noop(self) -> None:
        """Passing None does nothing."""
        proc = _make_processor()
        proc.update_state(None)
        assert len(proc._request_configs) == 0


# ---------------------------------------------------------------------------
# Tests: apply (full pipeline integration)
# ---------------------------------------------------------------------------


class TestApply:
    """Integration tests for the apply() method (full sampling pipeline)."""

    def test_single_row_produces_one_hot_logits(self) -> None:
        """After apply(), each row has exactly one 0.0 and rest -inf."""
        proc = _make_processor(vocab_size=100)
        logits = _make_logits(batch_size=1, vocab_size=100)
        result = proc.apply(logits)

        row = result[0]
        finite_mask = np.isfinite(row)
        assert np.sum(finite_mask) == 1, (
            f"Expected exactly 1 finite value, got {np.sum(finite_mask)}"
        )
        assert row[finite_mask][0] == 0.0

    def test_batch_produces_one_hot_per_row(self) -> None:
        """Each row in a batch gets exactly one finite value."""
        proc = _make_processor(vocab_size=100)
        logits = _make_logits(batch_size=4, vocab_size=100)
        result = proc.apply(logits)

        for i in range(4):
            finite_mask = np.isfinite(result[i])
            assert np.sum(finite_mask) == 1, (
                f"Row {i}: expected 1 finite value, got {np.sum(finite_mask)}"
            )

    def test_selected_token_is_valid_index(self) -> None:
        """The selected token index is within [0, vocab_size)."""
        vocab_size = 100
        proc = _make_processor(vocab_size=vocab_size)
        logits = _make_logits(batch_size=1, vocab_size=vocab_size)
        result = proc.apply(logits)

        row = result[0]
        selected_idx = int(np.where(np.isfinite(row))[0][0])
        assert 0 <= selected_idx < vocab_size

    def test_per_request_config_applies(self) -> None:
        """Per-request config via update_state affects token selection."""
        proc = _make_processor(vocab_size=100)
        # Add a request with very restrictive top-k
        params = FakeSamplingParams(extra_args={"qc_top_k": 1})
        proc.update_state(FakeBatchUpdate(added=[(0, params, [])]))

        logits = _make_logits(batch_size=1, vocab_size=100)
        result = proc.apply(logits)

        # With top_k=1, should always select the single most probable token
        row = result[0]
        selected_idx = int(np.where(np.isfinite(row))[0][0])
        assert 0 <= selected_idx < 100

    def test_diagnostic_mode_records(self) -> None:
        """With diagnostic_mode on, the logger stores records."""
        proc = _make_processor(
            vocab_size=100,
            config_overrides={
                "QC_DIAGNOSTIC_MODE": "true",
                "QC_LOG_LEVEL": "none",
                "QC_QRNG_FALLBACK_MODE": "os_urandom",
                "QC_QRNG_PREFETCH_ENABLED": "false",
            },
        )
        logits = _make_logits(batch_size=3, vocab_size=100)
        proc.apply(logits)

        records = proc._logger.get_diagnostic_data()
        assert len(records) == 3
        for rec in records:
            assert 0 <= rec.token_id < 100
            assert rec.u_value > 0.0
            assert rec.u_value < 1.0
            assert rec.total_sampling_ms > 0.0

    def test_multiple_apply_calls(self) -> None:
        """Multiple apply() calls don't corrupt state."""
        proc = _make_processor(vocab_size=50)
        for _ in range(5):
            logits = _make_logits(batch_size=2, vocab_size=50, seed=None)
            result = proc.apply(logits)
            for i in range(2):
                assert np.sum(np.isfinite(result[i])) == 1


# ---------------------------------------------------------------------------
# Tests: factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    """Test that factory functions correctly build components."""

    def test_build_signal_amplifier_zscore(self, default_config: QCSamplingConfig) -> None:
        """Factory builds ZScoreMeanAmplifier from default config."""
        amp = build_signal_amplifier(default_config)
        from qc_sampler.signal_amplifier import ZScoreMeanAmplifier
        assert isinstance(amp, ZScoreMeanAmplifier)

    def test_build_temperature_fixed(self, default_config: QCSamplingConfig) -> None:
        """Factory builds FixedTemperatureStrategy from default config."""
        strat = build_temperature_strategy(default_config, vocab_size=100)
        from qc_sampler.temperature_strategy import FixedTemperatureStrategy
        assert isinstance(strat, FixedTemperatureStrategy)

    def test_build_temperature_edt(self, edt_config: QCSamplingConfig) -> None:
        """Factory builds EDTTemperatureStrategy when config says 'edt'."""
        strat = build_temperature_strategy(edt_config, vocab_size=100)
        from qc_sampler.temperature_strategy import EDTTemperatureStrategy
        assert isinstance(strat, EDTTemperatureStrategy)

    def test_unknown_amplifier_type_raises(self) -> None:
        """Unknown signal_amplifier_type raises ConfigValidationError."""
        config = QCSamplingConfig(signal_amplifier_type="nonexistent")
        with pytest.raises(ConfigValidationError, match="nonexistent"):
            build_signal_amplifier(config)

    def test_unknown_temperature_strategy_raises(self) -> None:
        """Unknown temperature_strategy raises ConfigValidationError."""
        config = QCSamplingConfig(temperature_strategy="nonexistent")
        with pytest.raises(ConfigValidationError, match="nonexistent"):
            build_temperature_strategy(config, vocab_size=100)


# ---------------------------------------------------------------------------
# Tests: SamplingLogger
# ---------------------------------------------------------------------------


class TestSamplingLogger:
    """Test the SamplingLogger component."""

    def test_empty_summary_stats(self) -> None:
        """get_summary_stats returns empty dict when no records exist."""
        config = QCSamplingConfig(log_level="none", diagnostic_mode=False)
        log = SamplingLogger(config)
        assert log.get_summary_stats() == {}

    def test_diagnostic_mode_stores_records(self) -> None:
        """Records are stored when diagnostic_mode is True."""
        from qc_sampler.sampling_logger import TokenSamplingRecord
        config = QCSamplingConfig(log_level="none", diagnostic_mode=True)
        log = SamplingLogger(config)

        record = TokenSamplingRecord(
            timestamp_ns=1000,
            qrng_fetch_ms=1.0,
            total_sampling_ms=2.0,
            qrng_source_used="mock",
            sample_mean=127.5,
            z_score=0.0,
            u_value=0.5,
            temperature_strategy="fixed",
            shannon_entropy=3.0,
            temperature_used=0.7,
            token_id=42,
            token_rank=0,
            token_prob=0.1,
            num_candidates=50,
            config_hash="abc123",
        )
        log.log_token(record)
        data = log.get_diagnostic_data()
        assert len(data) == 1
        assert data[0].token_id == 42

    def test_no_diagnostic_mode_no_records(self) -> None:
        """Records are NOT stored when diagnostic_mode is False."""
        from qc_sampler.sampling_logger import TokenSamplingRecord
        config = QCSamplingConfig(log_level="summary", diagnostic_mode=False)
        log = SamplingLogger(config)

        record = TokenSamplingRecord(
            timestamp_ns=1000,
            qrng_fetch_ms=1.0,
            total_sampling_ms=2.0,
            qrng_source_used="mock",
            sample_mean=127.5,
            z_score=0.0,
            u_value=0.5,
            temperature_strategy="fixed",
            shannon_entropy=3.0,
            temperature_used=0.7,
            token_id=42,
            token_rank=0,
            token_prob=0.1,
            num_candidates=50,
            config_hash="abc123",
        )
        log.log_token(record)
        assert log.get_diagnostic_data() == []

    def test_summary_stats_computed(self) -> None:
        """get_summary_stats returns averages over stored records."""
        from qc_sampler.sampling_logger import TokenSamplingRecord
        config = QCSamplingConfig(log_level="none", diagnostic_mode=True)
        log = SamplingLogger(config)

        for i in range(10):
            log.log_token(TokenSamplingRecord(
                timestamp_ns=i * 1000,
                qrng_fetch_ms=float(i),
                total_sampling_ms=float(i * 2),
                qrng_source_used="mock",
                sample_mean=127.5,
                z_score=0.0,
                u_value=0.5,
                temperature_strategy="fixed",
                shannon_entropy=3.0,
                temperature_used=0.7,
                token_id=i,
                token_rank=i,
                token_prob=0.1,
                num_candidates=50,
                config_hash="abc",
            ))

        stats = log.get_summary_stats()
        assert stats["count"] == 10
        assert stats["avg_temperature"] == 0.7
        assert stats["avg_u_value"] == 0.5

    def test_invalid_log_level_raises(self) -> None:
        """Invalid log_level raises ValueError at construction."""
        config = QCSamplingConfig(log_level="invalid")
        with pytest.raises(ValueError, match="invalid"):
            SamplingLogger(config)

"""Integration tests for QRSamplerLogitsProcessor.

Tests the full sampling pipeline end-to-end using MockUniformSource
for deterministic, reproducible results without any real QRNG or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from qr_sampler.exceptions import ConfigValidationError
from qr_sampler.processor import QRSamplerLogitsProcessor

# ---------------------------------------------------------------------------
# Mock objects simulating vLLM's batch management types
# ---------------------------------------------------------------------------


@dataclass
class MockVllmConfig:
    """Simulates vLLM's VllmConfig with vocab_size access."""

    vocab_size: int = 10


@dataclass
class MockModelConfig:
    """Simulates vLLM's model config nested structure."""

    hf_text_config: Any = None


@dataclass
class MockHfTextConfig:
    """Simulates the HuggingFace text config with vocab_size."""

    vocab_size: int = 10


@dataclass
class MockSamplingParams:
    """Simulates vLLM's SamplingParams."""

    extra_args: dict[str, Any] | None = None


@dataclass
class MockAddedRequest:
    """Simulates a BatchUpdate added request."""

    req_index: int
    sampling_params: MockSamplingParams | None = None


@dataclass
class MockMovedRequest:
    """Simulates a BatchUpdate moved request."""

    src_index: int
    dst_index: int


@dataclass
class MockBatchUpdate:
    """Simulates vLLM's BatchUpdate dataclass."""

    removed: list[int] | None = None
    moved: list[MockMovedRequest] | None = None
    added: list[MockAddedRequest] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


# ---------------------------------------------------------------------------
# Helper to create a processor with MockUniformSource
# ---------------------------------------------------------------------------


def _make_processor(
    vocab_size: int = 10,
    entropy_source_type: str = "mock_uniform",
    fallback_mode: str = "error",
    **config_overrides: Any,
) -> QRSamplerLogitsProcessor:
    """Create a processor using mock entropy (no gRPC, no GPU).

    Sets environment variables to configure, then instantiates.
    """
    import os

    # Set env vars for config.
    env_vars = {
        "QR_ENTROPY_SOURCE_TYPE": entropy_source_type,
        "QR_FALLBACK_MODE": fallback_mode,
        "QR_LOG_LEVEL": "none",  # Quiet during tests.
    }
    for key, value in config_overrides.items():
        env_vars[f"QR_{key.upper()}"] = str(value)

    old_env: dict[str, str | None] = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        vllm_config = MockVllmConfig(vocab_size=vocab_size)
        proc = QRSamplerLogitsProcessor(
            vllm_config=vllm_config,
            device=None,
            is_pin_memory=False,
        )
    finally:
        for key, original in old_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessorInit:
    """Test processor construction and component wiring."""

    def test_init_with_mock_source(self) -> None:
        """Processor initializes successfully with mock entropy source."""
        proc = _make_processor()
        assert proc._vocab_size == 10
        assert proc.is_argmax_invariant() is False

    def test_init_with_none_vllm_config(self) -> None:
        """When vllm_config is None, uses default vocab size."""
        import os

        os.environ["QR_ENTROPY_SOURCE_TYPE"] = "mock_uniform"
        os.environ["QR_FALLBACK_MODE"] = "error"
        os.environ["QR_LOG_LEVEL"] = "none"
        try:
            proc = QRSamplerLogitsProcessor(vllm_config=None)
            assert proc._vocab_size == 32000  # _DEFAULT_VOCAB_SIZE
        finally:
            os.environ.pop("QR_ENTROPY_SOURCE_TYPE", None)
            os.environ.pop("QR_FALLBACK_MODE", None)
            os.environ.pop("QR_LOG_LEVEL", None)

    def test_init_with_nested_vllm_config(self) -> None:
        """Extracts vocab_size from nested vLLM config structure."""
        hf = MockHfTextConfig(vocab_size=256)
        model_cfg = MockModelConfig(hf_text_config=hf)

        @dataclass
        class NestedConfig:
            model_config: Any = None

        config = NestedConfig(model_config=model_cfg)
        vocab = QRSamplerLogitsProcessor._extract_vocab_size(config)
        assert vocab == 256

    def test_extract_vocab_size_fallback(self) -> None:
        """Falls back to default when config has no vocab_size."""

        class EmptyConfig:
            pass

        vocab = QRSamplerLogitsProcessor._extract_vocab_size(EmptyConfig())
        assert vocab == 32000

    def test_is_argmax_invariant(self) -> None:
        """Processor must return False for is_argmax_invariant."""
        proc = _make_processor()
        assert proc.is_argmax_invariant() is False


class TestValidateParams:
    """Test validate_params() classmethod."""

    def test_valid_extra_args(self) -> None:
        """Valid qr_ keys pass validation."""
        params = MockSamplingParams(extra_args={"qr_top_k": 100})
        QRSamplerLogitsProcessor.validate_params(params)

    def test_invalid_key_raises(self) -> None:
        """Unknown qr_ key raises ConfigValidationError."""
        params = MockSamplingParams(extra_args={"qr_nonexistent": 42})
        with pytest.raises(ConfigValidationError):
            QRSamplerLogitsProcessor.validate_params(params)

    def test_non_overridable_field_raises(self) -> None:
        """Infrastructure field raises ConfigValidationError."""
        params = MockSamplingParams(extra_args={"qr_grpc_server_address": "foo"})
        with pytest.raises(ConfigValidationError):
            QRSamplerLogitsProcessor.validate_params(params)

    def test_empty_extra_args(self) -> None:
        """Empty extra_args passes validation."""
        params = MockSamplingParams(extra_args={})
        QRSamplerLogitsProcessor.validate_params(params)

    def test_no_extra_args(self) -> None:
        """Missing extra_args passes validation."""
        params = MockSamplingParams(extra_args=None)
        QRSamplerLogitsProcessor.validate_params(params)

    def test_non_qr_keys_ignored(self) -> None:
        """Keys without qr_ prefix are silently ignored."""
        params = MockSamplingParams(extra_args={"other_key": "value"})
        QRSamplerLogitsProcessor.validate_params(params)


class TestUpdateState:
    """Test update_state() batch management."""

    def test_add_request(self) -> None:
        """Adding a request creates per-request state."""
        proc = _make_processor()
        batch = MockBatchUpdate(
            added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
        )
        proc.update_state(batch)
        assert 0 in proc._request_states

    def test_add_request_with_overrides(self) -> None:
        """Added request with extra_args gets resolved config."""
        proc = _make_processor()
        params = MockSamplingParams(extra_args={"qr_top_k": 100})
        batch = MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
        proc.update_state(batch)
        assert proc._request_states[0].config.top_k == 100

    def test_remove_request(self) -> None:
        """Removing a request cleans up per-request state."""
        proc = _make_processor()
        # Add then remove.
        proc.update_state(
            MockBatchUpdate(
                added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
            )
        )
        assert 0 in proc._request_states
        proc.update_state(MockBatchUpdate(removed=[0]))
        assert 0 not in proc._request_states

    def test_move_request(self) -> None:
        """Moving a request updates state index."""
        proc = _make_processor()
        proc.update_state(
            MockBatchUpdate(
                added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
            )
        )
        proc.update_state(MockBatchUpdate(moved=[MockMovedRequest(src_index=0, dst_index=5)]))
        assert 0 not in proc._request_states
        assert 5 in proc._request_states

    def test_none_batch_update(self) -> None:
        """None batch_update is a no-op."""
        proc = _make_processor()
        proc.update_state(None)  # Should not raise.

    def test_removal_then_add_in_same_update(self) -> None:
        """Process removal before addition in the same batch update."""
        proc = _make_processor()
        proc.update_state(
            MockBatchUpdate(
                added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
            )
        )
        # Remove 0 and add 0 back in same update.
        proc.update_state(
            MockBatchUpdate(
                removed=[0],
                added=[
                    MockAddedRequest(
                        req_index=0,
                        sampling_params=MockSamplingParams(extra_args={"qr_top_k": 200}),
                    )
                ],
            )
        )
        assert proc._request_states[0].config.top_k == 200


class TestApplyPipeline:
    """Test the full apply() pipeline with numpy arrays."""

    def test_single_row_onehot(self) -> None:
        """apply() produces one-hot output for a single-row batch."""
        proc = _make_processor()
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = proc.apply(logits)

        # Exactly one 0.0 value, rest are -inf.
        assert result is logits  # In-place modification.
        row = result[0]
        assert np.sum(row == 0.0) == 1
        assert np.sum(np.isneginf(row)) == 9

    def test_batch_processing(self) -> None:
        """apply() processes all rows in a batch."""
        proc = _make_processor()
        logits = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        )
        result = proc.apply(logits)

        for i in range(3):
            row = result[i]
            assert np.sum(row == 0.0) == 1, f"Row {i} should have exactly one 0.0"
            assert np.sum(np.isneginf(row)) == 9, f"Row {i} should have 9 -inf values"

    def test_dominant_token_selection(self) -> None:
        """A very dominant logit is likely selected (u near 0 → most probable)."""
        proc = _make_processor()
        # Token 4 has overwhelmingly high logit.
        logits = np.array(
            [
                [
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                ]
            ]
        )
        result = proc.apply(logits)
        # Token 4 should be selected regardless of u value.
        assert result[0, 4] == 0.0

    def test_1d_logits(self) -> None:
        """apply() handles 1-D logits (single request, no batch dim)."""
        proc = _make_processor()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = proc.apply(logits)
        assert np.sum(result == 0.0) == 1
        assert np.sum(np.isneginf(result)) == 9

    def test_empty_batch(self) -> None:
        """apply() short-circuits on empty batch."""
        proc = _make_processor()
        logits = np.empty((0, 10))
        result = proc.apply(logits)
        assert result.shape == (0, 10)

    def test_per_request_config_in_apply(self) -> None:
        """Per-request config affects token selection parameters."""
        proc = _make_processor()

        # Add a request with top_k=1 (greedy-like: only top token survives).
        params = MockSamplingParams(extra_args={"qr_top_k": 1})
        proc.update_state(
            MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
        )

        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = proc.apply(logits)
        # With top_k=1, only the highest logit (index 0) should be selected.
        assert result[0, 0] == 0.0

    def test_inplace_modification(self) -> None:
        """apply() modifies the logits array in-place and returns it."""
        proc = _make_processor()
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = proc.apply(logits)
        assert result is logits

    def test_dominant_token_always_selected(self) -> None:
        """When one logit overwhelmingly dominates, it is selected regardless of u."""
        # Run multiple times with different random entropy to confirm the
        # dominant token is always picked. This verifies the full pipeline:
        # temperature -> softmax -> CDF -> select -> one-hot.
        for _ in range(5):
            proc = _make_processor()
            logits = np.array(
                [[-100.0, -100.0, -100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]]
            )
            proc.apply(logits)
            # Token 3 is the only viable candidate after softmax.
            assert logits[0, 3] == 0.0


class TestDiagnosticLogging:
    """Test that the processor produces valid diagnostic records."""

    def test_diagnostic_records_stored(self) -> None:
        """With diagnostic_mode=True, records are stored."""
        proc = _make_processor(diagnostic_mode=True)
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

        record = records[0]
        assert record.token_id >= 0
        assert record.token_id < 10
        assert 0.0 < record.u_value < 1.0
        assert record.token_rank >= 0
        assert record.token_prob > 0.0
        assert record.num_candidates > 0
        assert record.entropy_fetch_ms >= 0.0
        assert record.total_sampling_ms > 0.0
        assert len(record.config_hash) == 16
        assert record.temperature_used > 0.0

    def test_batch_diagnostic_records(self) -> None:
        """Each row in a batch produces one diagnostic record."""
        proc = _make_processor(diagnostic_mode=True)
        logits = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        proc.apply(logits)

        records = proc.sampling_logger.get_diagnostic_data()
        assert len(records) == 2

    def test_entropy_source_tracking(self) -> None:
        """Diagnostic records track which entropy source was used."""
        proc = _make_processor(diagnostic_mode=True)
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        proc.apply(logits)

        record = proc.sampling_logger.get_diagnostic_data()[0]
        assert record.entropy_source_used == "mock_uniform"
        assert record.entropy_is_fallback is False


class TestFallbackIntegration:
    """Test fallback entropy source integration."""

    def test_system_fallback(self) -> None:
        """With system fallback, processor works even if primary is unavailable."""
        proc = _make_processor(
            entropy_source_type="mock_uniform",
            fallback_mode="system",
        )
        # FallbackEntropySource wraps mock + system.
        assert "+" in proc.entropy_source.name

        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = proc.apply(logits)
        assert np.sum(result[0] == 0.0) == 1


class TestProcessorClose:
    """Test processor resource cleanup."""

    def test_close(self) -> None:
        """close() releases entropy source resources."""
        proc = _make_processor()
        proc.close()  # Should not raise.

    def test_close_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        proc = _make_processor()
        proc.close()
        proc.close()  # Should not raise.

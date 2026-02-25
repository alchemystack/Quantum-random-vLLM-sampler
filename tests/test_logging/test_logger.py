"""Tests for SamplingLogger and TokenSamplingRecord."""

from __future__ import annotations

import logging

import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.types import TokenSamplingRecord


def _make_record(**overrides: object) -> TokenSamplingRecord:
    """Create a TokenSamplingRecord with sensible defaults, overridable."""
    defaults: dict[str, object] = {
        "timestamp_ns": 1000000000,
        "entropy_fetch_ms": 1.5,
        "total_sampling_ms": 3.0,
        "entropy_source_used": "mock_uniform",
        "entropy_is_fallback": False,
        "sample_mean": 127.5,
        "z_score": 0.0,
        "u_value": 0.5,
        "temperature_strategy": "fixed",
        "shannon_entropy": 2.3,
        "temperature_used": 0.7,
        "token_id": 42,
        "token_rank": 3,
        "token_prob": 0.15,
        "num_candidates": 50,
        "config_hash": "abcdef1234567890",
    }
    defaults.update(overrides)
    return TokenSamplingRecord(**defaults)  # type: ignore[arg-type]


class TestTokenSamplingRecord:
    """Tests for TokenSamplingRecord immutability."""

    def test_frozen(self) -> None:
        """TokenSamplingRecord should reject attribute mutation."""
        record = _make_record()
        with pytest.raises(AttributeError):
            record.token_id = 99  # type: ignore[misc]

    def test_slots(self) -> None:
        """TokenSamplingRecord should use __slots__."""
        record = _make_record()
        assert hasattr(record, "__slots__")

    def test_all_fields_accessible(self) -> None:
        """All 16 fields should be readable."""
        record = _make_record()
        assert record.timestamp_ns == 1000000000
        assert record.entropy_fetch_ms == 1.5
        assert record.total_sampling_ms == 3.0
        assert record.entropy_source_used == "mock_uniform"
        assert record.entropy_is_fallback is False
        assert record.sample_mean == 127.5
        assert record.z_score == 0.0
        assert record.u_value == 0.5
        assert record.temperature_strategy == "fixed"
        assert record.shannon_entropy == 2.3
        assert record.temperature_used == 0.7
        assert record.token_id == 42
        assert record.token_rank == 3
        assert record.token_prob == 0.15
        assert record.num_candidates == 50
        assert record.config_hash == "abcdef1234567890"


class TestSamplingLogger:
    """Tests for SamplingLogger."""

    def test_log_level_none_no_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_level='none' should produce no log output."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="none",
            diagnostic_mode=False,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        with caplog.at_level(logging.DEBUG, logger="qr_sampler"):
            log.log_token(_make_record())
        assert len(caplog.records) == 0

    def test_log_level_summary_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_level='summary' should produce one-line summary."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="summary",
            diagnostic_mode=False,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        with caplog.at_level(logging.DEBUG, logger="qr_sampler"):
            log.log_token(_make_record())
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "token=42" in msg
        assert "rank=3" in msg
        assert "u=0.500000" in msg

    def test_log_level_summary_fallback_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        """Summary should include [FALLBACK] when entropy_is_fallback is True."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="summary",
            diagnostic_mode=False,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        record = _make_record(entropy_is_fallback=True)
        with caplog.at_level(logging.DEBUG, logger="qr_sampler"):
            log.log_token(record)
        assert "[FALLBACK]" in caplog.records[0].message

    def test_log_level_full_json(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_level='full' should produce JSON dump."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="full",
            diagnostic_mode=False,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        with caplog.at_level(logging.DEBUG, logger="qr_sampler"):
            log.log_token(_make_record())
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "sampling_record:" in msg
        assert '"token_id": 42' in msg

    def test_diagnostic_mode_stores_records(self) -> None:
        """diagnostic_mode=True should store records in memory."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="none",
            diagnostic_mode=True,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        log.log_token(_make_record(token_id=1))
        log.log_token(_make_record(token_id=2))
        log.log_token(_make_record(token_id=3))

        data = log.get_diagnostic_data()
        assert len(data) == 3
        assert data[0].token_id == 1
        assert data[2].token_id == 3

    def test_diagnostic_mode_false_no_storage(self) -> None:
        """diagnostic_mode=False should not store records."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="summary",
            diagnostic_mode=False,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        log.log_token(_make_record())
        assert log.get_diagnostic_data() == []

    def test_get_diagnostic_data_returns_copy(self) -> None:
        """get_diagnostic_data() should return a copy, not the internal list."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="none",
            diagnostic_mode=True,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        log.log_token(_make_record())

        data1 = log.get_diagnostic_data()
        data1.clear()
        assert len(log.get_diagnostic_data()) == 1  # Original unaffected.

    def test_summary_stats_empty(self) -> None:
        """Summary stats on empty logger should return empty dict."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="none",
            diagnostic_mode=True,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        assert log.get_summary_stats() == {}

    def test_summary_stats_computed(self) -> None:
        """Summary stats should compute aggregate values correctly."""
        config = QRSamplerConfig(
            _env_file=None,
            log_level="none",
            diagnostic_mode=True,  # type: ignore[call-arg]
        )
        log = SamplingLogger(config)
        log.log_token(
            _make_record(
                u_value=0.3,
                token_rank=1,
                token_prob=0.5,
                entropy_fetch_ms=1.0,
                total_sampling_ms=2.0,
                entropy_is_fallback=False,
            )
        )
        log.log_token(
            _make_record(
                u_value=0.7,
                token_rank=3,
                token_prob=0.1,
                entropy_fetch_ms=3.0,
                total_sampling_ms=4.0,
                entropy_is_fallback=True,
            )
        )

        stats = log.get_summary_stats()
        assert stats["total_tokens"] == 2
        assert abs(stats["mean_u"] - 0.5) < 1e-10
        assert stats["min_u"] == 0.3
        assert stats["max_u"] == 0.7
        assert abs(stats["mean_rank"] - 2.0) < 1e-10
        assert abs(stats["mean_prob"] - 0.3) < 1e-10
        assert abs(stats["mean_fetch_ms"] - 2.0) < 1e-10
        assert abs(stats["mean_total_ms"] - 3.0) < 1e-10
        assert stats["max_total_ms"] == 4.0
        assert stats["fallback_count"] == 1
        assert abs(stats["fallback_rate"] - 0.5) < 1e-10

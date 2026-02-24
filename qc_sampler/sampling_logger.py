"""Structured per-token logging for quantum consciousness sampling.

This module provides ``SamplingLogger`` which records per-token sampling
metadata at configurable verbosity levels, and ``TokenSamplingRecord``
which captures every measurable aspect of a single token's sampling
pipeline.

Three log levels control what gets written to the Python logger:
    - ``"none"``: No logging output.  Minimal overhead for production.
    - ``"summary"``: One log line per token with key metrics (u-value,
      temperature, token rank, latency).
    - ``"full"``: Complete diagnostic dump including all intermediate
      values from every pipeline stage.

When ``diagnostic_mode`` is ``True``, all records are retained in memory
regardless of log level, enabling programmatic analysis (e.g. KS-testing
u-value distributions, correlating temperature with entropy).

Design notes:
    The logger is intentionally separated from the pipeline components.
    It receives a fully-populated ``TokenSamplingRecord`` after each
    token and decides what to do with it based on config.  Components
    never need to know whether they're being logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from qc_sampler.config import QCSamplingConfig

logger = logging.getLogger("qc_sampler")

# Valid log levels, checked at construction time.
_VALID_LOG_LEVELS = frozenset({"none", "summary", "full"})


@dataclass(frozen=True)
class TokenSamplingRecord:
    """Complete record of one token's sampling pipeline execution.

    Every field captures a measurable aspect of the token selection
    process, providing full observability for debugging and statistical
    analysis.

    Attributes:
        timestamp_ns: Monotonic timestamp (perf_counter_ns) when the
            token was sampled.
        qrng_fetch_ms: Time spent fetching QRNG bytes (including
            prefetch wait or blocking call), in milliseconds.
        total_sampling_ms: Wall-clock time for the entire per-token
            pipeline (temperature + entropy fetch + amplify + select).

        qrng_source_used: Which entropy source actually provided bytes:
            ``"grpc"``, ``"prefetched"``, ``"fallback"``, or ``"mock"``.

        sample_mean: Mean of the raw QRNG byte array (uint8 values).
        z_score: Z-score from signal amplification.
        u_value: The final uniform float in (0, 1) used for CDF lookup.

        temperature_strategy: Name of the strategy used (``"fixed"``
            or ``"edt"``).
        shannon_entropy: Shannon entropy of the logit distribution (nats).
        temperature_used: The actual temperature applied during sampling.

        token_id: Vocabulary index of the selected token.
        token_rank: Rank in the probability-sorted candidate list
            (0 = most probable).
        token_prob: Probability of the selected token after filtering.
        num_candidates: Number of tokens that survived top-k/top-p.

        config_hash: SHA-256 hex digest (truncated to 16 chars) of the
            serialized resolved config.  Useful for grouping tokens that
            shared the same configuration.
    """

    # Timing
    timestamp_ns: int
    qrng_fetch_ms: float
    total_sampling_ms: float

    # Entropy source
    qrng_source_used: str

    # Signal amplification
    sample_mean: float
    z_score: float
    u_value: float

    # Temperature
    temperature_strategy: str
    shannon_entropy: float
    temperature_used: float

    # Selection
    token_id: int
    token_rank: int
    token_prob: float
    num_candidates: int

    # Config snapshot
    config_hash: str


def compute_config_hash(config: QCSamplingConfig) -> str:
    """Compute a short hash of a QCSamplingConfig for record tagging.

    Uses SHA-256 over a deterministic JSON serialization of all config
    fields, truncated to 16 hex characters.  This is NOT cryptographic —
    it's a fingerprint for grouping tokens that shared the same config.

    Args:
        config: The resolved per-request configuration.

    Returns:
        A 16-character hex digest string.
    """
    # dataclasses.asdict would work but QCSamplingConfig is simple enough
    # to serialize via __dict__-like access on its fields.
    from dataclasses import fields as dc_fields

    data = {f.name: getattr(config, f.name) for f in dc_fields(config)}
    # sort_keys=True ensures deterministic serialization.
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class SamplingLogger:
    """Records per-token sampling metadata for monitoring and analysis.

    Instantiated once per processor lifetime.  Each call to
    ``log_token()`` processes one ``TokenSamplingRecord`` according to
    the configured log level.  When ``diagnostic_mode`` is enabled,
    records are also stored in memory for programmatic access.

    Thread safety:
        This class is NOT thread-safe.  In vLLM's LogitsProcessor
        pipeline, ``apply()`` is called sequentially per batch, so
        concurrent access is not expected.

    Args:
        config: Server-level default config, used to determine the
            initial log level and diagnostic mode.  Per-request log
            levels are respected via the record's ``config_hash``, but
            the logger's own level is set at construction time.
    """

    def __init__(self, config: QCSamplingConfig) -> None:
        """Initialize the sampling logger.

        Args:
            config: Configuration determining log level and diagnostic mode.

        Raises:
            ValueError: If config.log_level is not one of the valid levels.
        """
        if config.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level '{config.log_level}'; "
                f"expected one of: {sorted(_VALID_LOG_LEVELS)}"
            )
        self._log_level = config.log_level
        self._diagnostic_mode = config.diagnostic_mode
        self._records: list[TokenSamplingRecord] = []

    def log_token(self, record: TokenSamplingRecord) -> None:
        """Record one token's sampling data.

        Depending on ``log_level``:
        - ``"none"``: No-op (but still stores if diagnostic_mode).
        - ``"summary"``: Logs one line with key metrics.
        - ``"full"``: Logs complete record as structured data.

        If ``diagnostic_mode`` is ``True``, the record is always
        appended to the in-memory list regardless of log level.

        Args:
            record: The fully-populated sampling record for one token.
        """
        if self._diagnostic_mode:
            self._records.append(record)

        if self._log_level == "none":
            return

        if self._log_level == "summary":
            logger.info(
                "QC token: id=%d rank=%d prob=%.4f u=%.4f "
                "temp=%.3f entropy=%.3f candidates=%d "
                "qrng_ms=%.2f total_ms=%.2f source=%s",
                record.token_id,
                record.token_rank,
                record.token_prob,
                record.u_value,
                record.temperature_used,
                record.shannon_entropy,
                record.num_candidates,
                record.qrng_fetch_ms,
                record.total_sampling_ms,
                record.qrng_source_used,
            )
        elif self._log_level == "full":
            logger.info(
                "QC token (full): %s",
                json.dumps(asdict(record), default=str),
            )

    def get_diagnostic_data(self) -> list[TokenSamplingRecord]:
        """Return all recorded data.

        Only populated when ``diagnostic_mode`` is ``True``.  Returns
        an empty list otherwise.

        Returns:
            A list of all TokenSamplingRecord objects stored so far.
        """
        return list(self._records)

    def get_summary_stats(self) -> dict[str, Any]:
        """Return aggregate statistics over all recorded tokens.

        Computes mean values for key metrics.  Returns an empty dict
        if no records have been stored (either because diagnostic_mode
        is off or no tokens have been sampled yet).

        Returns:
            Dict with keys like ``"count"``, ``"avg_entropy"``,
            ``"avg_temperature"``, ``"avg_u_value"``, ``"avg_rank"``,
            ``"avg_qrng_ms"``, ``"avg_total_ms"``.
        """
        if not self._records:
            return {}

        count = len(self._records)
        return {
            "count": count,
            "avg_entropy": sum(r.shannon_entropy for r in self._records) / count,
            "avg_temperature": sum(r.temperature_used for r in self._records) / count,
            "avg_u_value": sum(r.u_value for r in self._records) / count,
            "avg_rank": sum(r.token_rank for r in self._records) / count,
            "avg_qrng_ms": sum(r.qrng_fetch_ms for r in self._records) / count,
            "avg_total_ms": sum(r.total_sampling_ms for r in self._records) / count,
        }

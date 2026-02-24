"""vLLM V1 LogitsProcessor integration for quantum consciousness sampling.

This module is the orchestration layer that wires together all pipeline
components (entropy source, signal amplifier, temperature strategy, token
selector, logger) into a single ``LogitsProcessor`` that vLLM calls on
every forward pass.

Registration:
    The processor is registered as a vLLM plugin via pyproject.toml::

        [project.entry-points."vllm.logits_processors"]
        quantum_consciousness = "qc_sampler.processor:QuantumConsciousnessProcessor"

Usage:
    vLLM discovers and instantiates the processor automatically.
    Users configure it through:

    1. **Environment variables** (QC_*) — set server-level defaults.
    2. **SamplingParams.extra_args** (qc_*) — per-request overrides.

    IMPORTANT: Set vLLM-level sampling to pass-through so this
    processor has full control::

        SamplingParams(
            temperature=1.0,   # no vLLM temperature scaling
            top_k=-1,          # no vLLM top-k
            top_p=1.0,         # no vLLM top-p
            extra_args={"qc_temperature_strategy": "edt", ...}
        )

Architecture:
    The processor is stateless across batches except for per-request
    config caching (updated via ``update_state``).  All algorithmic
    work is delegated to the strategy/selector components built by
    the factory module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np

try:
    import torch
except ImportError:
    # torch may not be available in test environments.
    # We define a placeholder so type hints still work.
    torch = None  # type: ignore[assignment]

from qc_sampler.config import (
    QCSamplingConfig,
    load_config_from_env,
    resolve_config,
    validate_extra_args,
)
from qc_sampler.exceptions import ConfigValidationError
from qc_sampler.factory import (
    build_entropy_source,
    build_signal_amplifier,
    build_temperature_strategy,
)
from qc_sampler.sampling_logger import (
    SamplingLogger,
    TokenSamplingRecord,
    compute_config_hash,
)
from qc_sampler.token_selector import TokenSelector

logger = logging.getLogger("qc_sampler")


class QuantumConsciousnessProcessor:
    """vLLM V1 LogitsProcessor that replaces standard random sampling
    with consciousness-influenceable quantum random sampling.

    This processor intercepts the logits after the model's forward pass
    and applies its own sampling pipeline:

        1. Compute per-token temperature (fixed or entropy-based dynamic).
        2. Fetch quantum random bytes from the QRNG entropy source.
        3. Amplify the raw bytes into a single uniform float via z-score.
        4. Select a token using a probability-ordered CDF.
        5. Force vLLM to pick that token by setting logits to one-hot.

    The processor manages per-request configuration (different requests
    can use different temperature strategies, top-k, etc.) and caches
    strategy/amplifier instances to avoid re-constructing them per token.

    Per-request configuration is communicated via
    ``SamplingParams.extra_args`` with a ``qc_`` prefix.  Infrastructure
    fields (gRPC address, timeout, etc.) are set once via environment
    variables and cannot be overridden per-request.
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
    ) -> None:
        """Initialize the processor and all pipeline components.

        Called once at vLLM startup.  Loads configuration from
        environment variables, builds the entropy source chain,
        default amplifier, default temperature strategy, token
        selector, and sampling logger.

        Args:
            vllm_config: vLLM's configuration object.  Used to read
                ``model_config.get_vocab_size()``.  May be ``None``
                in test environments (in which case vocab_size must
                be set separately via ``_vocab_size``).
            device: The torch device for logit tensor operations
                (e.g. ``torch.device("cuda:0")``).
            is_pin_memory: Whether to use pinned memory for tensors.
                Currently unused but accepted for interface compat.
        """
        self._device = device

        # Vocab size from vLLM config; tests can override _vocab_size.
        if vllm_config is not None:
            self._vocab_size: int = vllm_config.model_config.get_vocab_size()
        else:
            self._vocab_size = 0  # Must be set by test harness.

        # Load server-level defaults from environment variables.
        self._default_config = load_config_from_env()

        # Build shared components from default config.
        self._entropy_source = build_entropy_source(self._default_config)

        self._default_amplifier = build_signal_amplifier(self._default_config)
        self._default_temp_strategy = build_temperature_strategy(
            self._default_config, self._vocab_size
        )
        self._token_selector = TokenSelector()
        self._logger = SamplingLogger(self._default_config)

        # Per-request state: index → resolved config / cached components.
        self._request_configs: dict[int, QCSamplingConfig] = {}
        self._request_amplifiers: dict[int, Any] = {}
        self._request_temp_strategies: dict[int, Any] = {}

        logger.info(
            "QuantumConsciousnessProcessor initialized: vocab_size=%d, "
            "strategy=%s, amplifier=%s, fallback=%s",
            self._vocab_size,
            self._default_config.temperature_strategy,
            self._default_config.signal_amplifier_type,
            self._default_config.qrng_fallback_mode,
        )

    # ------------------------------------------------------------------
    # vLLM LogitsProcessor interface
    # ------------------------------------------------------------------

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validate all qc_* extra_args on a SamplingParams object.

        Called by vLLM during request setup to catch invalid
        configuration before the request enters the batch.

        Args:
            params: A ``SamplingParams`` instance with an optional
                ``extra_args`` dict.

        Raises:
            ValueError: If any qc_* key has an invalid field name,
                is not per-request overridable, or has a bad type/value.
        """
        extra_args = getattr(params, "extra_args", None)
        if not extra_args:
            return

        try:
            validate_extra_args(extra_args)
        except ConfigValidationError as exc:
            raise ValueError(str(exc)) from exc

    def is_argmax_invariant(self) -> bool:
        """Return False — this processor changes token selection.

        vLLM uses this to determine whether greedy decoding can
        skip the sampling step.  Since we replace the entire
        sampling mechanism, we must return False.
        """
        return False

    def update_state(self, batch_update: Any | None) -> None:
        """Process batch membership changes.

        Called by vLLM before each ``apply()`` with information about
        which requests were added, removed, or moved within the batch.
        Processing order follows vLLM's contract: remove → move → add.

        For added requests, resolves per-request config from extra_args
        and caches strategy/amplifier instances if they differ from
        the server defaults.  Also triggers entropy prefetch for all
        active requests.

        Args:
            batch_update: A ``BatchUpdate`` object with ``removed``,
                ``moved``, and ``added`` sequences, or ``None`` if
                no changes occurred.
        """
        if batch_update is None:
            return

        # --- Remove ---
        for idx in getattr(batch_update, "removed", []):
            self._request_configs.pop(idx, None)
            self._request_amplifiers.pop(idx, None)
            self._request_temp_strategies.pop(idx, None)

        # --- Move ---
        for src, dst, _direction in getattr(batch_update, "moved", []):
            if src in self._request_configs:
                self._request_configs[dst] = self._request_configs.pop(src)
            if src in self._request_amplifiers:
                self._request_amplifiers[dst] = self._request_amplifiers.pop(src)
            if src in self._request_temp_strategies:
                self._request_temp_strategies[dst] = (
                    self._request_temp_strategies.pop(src)
                )

        # --- Add ---
        for idx, params, _output_token_ids in getattr(batch_update, "added", []):
            extra_args = getattr(params, "extra_args", None)
            resolved = resolve_config(self._default_config, extra_args)
            self._request_configs[idx] = resolved

            # Cache per-request amplifier if it differs from default.
            if (
                resolved.signal_amplifier_type
                != self._default_config.signal_amplifier_type
            ):
                self._request_amplifiers[idx] = build_signal_amplifier(resolved)

            # Cache per-request temperature strategy if it differs.
            if (
                resolved.temperature_strategy
                != self._default_config.temperature_strategy
            ):
                self._request_temp_strategies[idx] = build_temperature_strategy(
                    resolved, self._vocab_size
                )

        # --- Prefetch entropy for active requests ---
        if self._default_config.qrng_prefetch_enabled:
            for idx in self._request_configs:
                config = self._request_configs[idx]
                self._entropy_source.prefetch(config.sample_count)

    def apply(self, logits: Any) -> Any:
        """Apply quantum consciousness sampling to each row in the batch.

        For each row ``i`` in the logits tensor:
            1. Resolve per-request config (or use defaults).
            2. Compute temperature via the configured strategy.
            3. Fetch QRNG bytes and amplify to a uniform float.
            4. Select a token via the probability-ordered CDF.
            5. Overwrite logits[i] to force vLLM to pick that token.
            6. Log the sampling record.

        Args:
            logits: A ``torch.Tensor`` of shape ``(batch_size, vocab_size)``
                containing raw logits from the model.

        Returns:
            The modified logits tensor with exactly one finite value
            (``0.0``) per row and the rest set to ``-inf``.
        """
        batch_size = logits.shape[0]

        for i in range(batch_size):
            pipeline_start_ns = time.perf_counter_ns()

            config = self._request_configs.get(i, self._default_config)

            # --- Convert logits to numpy ---
            row_raw = logits[i]
            if torch is not None and isinstance(row_raw, torch.Tensor):
                row_logits = row_raw.float().cpu().numpy()
            else:
                row_logits = np.asarray(row_raw, dtype=np.float64)

            # --- Temperature ---
            temp_strategy = self._request_temp_strategies.get(
                i, self._default_temp_strategy
            )
            temp_result = temp_strategy.compute_temperature(row_logits, config)

            # --- Entropy bytes ---
            qrng_start_ns = time.perf_counter_ns()
            raw_bytes = self._entropy_source.get_bytes(config.sample_count)
            qrng_elapsed_ms = (time.perf_counter_ns() - qrng_start_ns) / 1e6

            # Determine which source was used (for logging).
            source_health = self._entropy_source.health_check()
            qrng_source_used = source_health.get("source", "unknown")

            # --- Signal amplification ---
            amplifier = self._request_amplifiers.get(
                i, self._default_amplifier
            )
            amp_result = amplifier.amplify(raw_bytes)

            # --- Token selection ---
            selection = self._token_selector.select(
                logits=row_logits,
                temperature=temp_result.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                u=amp_result.u,
            )

            # --- Force vLLM to pick this token ---
            if torch is not None:
                logits[i] = torch.full(
                    (self._vocab_size,),
                    float("-inf"),
                    device=self._device,
                    dtype=logits.dtype,
                )
                logits[i, selection.token_id] = 0.0
            else:
                # Fallback for test environments without torch.
                row = np.full(self._vocab_size, float("-inf"))
                row[selection.token_id] = 0.0
                logits[i] = row

            # --- Logging ---
            pipeline_elapsed_ms = (
                time.perf_counter_ns() - pipeline_start_ns
            ) / 1e6

            self._logger.log_token(
                TokenSamplingRecord(
                    timestamp_ns=pipeline_start_ns,
                    qrng_fetch_ms=qrng_elapsed_ms,
                    total_sampling_ms=pipeline_elapsed_ms,
                    qrng_source_used=qrng_source_used,
                    sample_mean=amp_result.diagnostics.get("sample_mean", 0.0),
                    z_score=amp_result.diagnostics.get("z_score", 0.0),
                    u_value=amp_result.u,
                    temperature_strategy=config.temperature_strategy,
                    shannon_entropy=temp_result.shannon_entropy,
                    temperature_used=temp_result.temperature,
                    token_id=selection.token_id,
                    token_rank=selection.token_rank,
                    token_prob=selection.token_prob,
                    num_candidates=selection.num_candidates,
                    config_hash=compute_config_hash(config),
                )
            )

        return logits

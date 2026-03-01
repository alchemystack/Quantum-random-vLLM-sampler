"""vLLM V1 LogitsProcessor — the integration layer for qr-sampler.

Orchestrates the full per-token sampling pipeline:
    logits → temperature → entropy fetch → amplification → CDF selection → one-hot.

Registered via entry point::

    [project.entry-points."vllm.logits_processors"]
    qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"

The processor applies globally to all requests in a vLLM instance. Deploy
separate instances for different sampling strategies.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.registry import EntropySourceRegistry
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.entropy.base import EntropySource
    from qr_sampler.temperature.base import TemperatureStrategy

logger = logging.getLogger("qr_sampler")

# Default vocabulary size when vllm_config does not provide one (testing).
_DEFAULT_VOCAB_SIZE = 32000


def _config_hash(config: QRSamplerConfig) -> str:
    """Compute a short hash of the config for logging.

    Args:
        config: The sampler configuration to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest of the config dump.
    """
    raw = config.model_dump_json().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _accepts_config(cls: type) -> bool:
    """Check if a class constructor accepts a QRSamplerConfig as first arg.

    Inspects the ``__init__`` signature for a parameter annotated as
    ``QRSamplerConfig`` (or whose name is ``config``).

    Args:
        cls: The class to inspect.

    Returns:
        True if the constructor expects a config argument.
    """
    import inspect

    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return False

    params = list(sig.parameters.values())
    # inspect.signature(cls) already strips 'self' for classes.
    for param in params:
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            if param.name == "config":
                return True
        elif annotation is QRSamplerConfig or (
            isinstance(annotation, str) and "QRSamplerConfig" in annotation
        ):
            return True
        # Only check the first non-self parameter.
        break
    return False


def _build_entropy_source(config: QRSamplerConfig) -> EntropySource:
    """Build the entropy source from config, wrapping with fallback if needed.

    Args:
        config: Sampler configuration specifying source type and fallback mode.

    Returns:
        An EntropySource, potentially wrapped in FallbackEntropySource.
    """
    source_cls = EntropySourceRegistry.get(config.entropy_source_type)

    # Only pass config if the constructor expects it.
    if _accepts_config(source_cls):
        primary: EntropySource = source_cls(config)  # type: ignore[call-arg]
    else:
        primary = source_cls()

    if config.fallback_mode == "error":
        return primary

    # Build fallback source.
    if config.fallback_mode == "system":
        from qr_sampler.entropy.system import SystemEntropySource

        fallback: EntropySource = SystemEntropySource()
    elif config.fallback_mode == "mock_uniform":
        from qr_sampler.entropy.mock import MockUniformSource

        fallback = MockUniformSource()
    else:
        logger.warning(
            "Unknown fallback_mode %r, using system fallback",
            config.fallback_mode,
        )
        from qr_sampler.entropy.system import SystemEntropySource

        fallback = SystemEntropySource()

    return FallbackEntropySource(primary, fallback)


class _RequestState:
    """Per-request state tracked across engine steps.

    Attributes:
        config: Resolved per-request configuration.
        amplifier: Signal amplifier for this request.
        strategy: Temperature strategy for this request.
        config_hash_str: Short hash for logging.
    """

    __slots__ = ("amplifier", "config", "config_hash_str", "strategy")

    def __init__(
        self,
        config: QRSamplerConfig,
        amplifier: SignalAmplifier,
        strategy: TemperatureStrategy,
        config_hash_str: str,
    ) -> None:
        self.config = config
        self.amplifier = amplifier
        self.strategy = strategy
        self.config_hash_str = config_hash_str


class QRSamplerLogitsProcessor:
    """vLLM V1 LogitsProcessor that replaces token sampling with
    external-entropy-driven selection.

    The processor fetches entropy just-in-time for each token, amplifies
    it into a uniform float via z-score statistics, and uses that float
    to select a token from a probability-ordered CDF. The selected token
    is forced via a one-hot logit vector.

    Constructor signature matches vLLM V1's ``LogitsProcessor`` ABC::

        __init__(self, vllm_config, device, is_pin_memory)
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
    ) -> None:
        """Initialize the processor and all subsystems.

        Args:
            vllm_config: vLLM's ``VllmConfig`` object (provides vocab_size).
                ``None`` in test environments — uses ``_DEFAULT_VOCAB_SIZE``.
            device: ``torch.device`` for tensor operations. ``None`` in tests.
            is_pin_memory: Whether to use pinned CPU memory for transfers.
        """
        # --- Extract vocab_size ---
        self._vocab_size = self._extract_vocab_size(vllm_config)
        self._device = device
        self._is_pin_memory = is_pin_memory

        # --- Load default configuration ---
        self._default_config = QRSamplerConfig()

        # --- Build shared components ---
        self._entropy_source = _build_entropy_source(self._default_config)
        self._default_amplifier = AmplifierRegistry.build(self._default_config)
        self._default_strategy = TemperatureStrategyRegistry.build(
            self._default_config, self._vocab_size
        )
        self._selector = TokenSelector()
        self._logger = SamplingLogger(self._default_config)

        # --- Pre-compute default state ---
        self._default_config_hash = _config_hash(self._default_config)

        # --- Pre-allocate tensors ---
        self._onehot_template = self._create_onehot_template()
        self._cpu_buffer = self._create_cpu_buffer()

        # --- Per-request state ---
        # Maps request index (batch position) to its state.
        self._request_states: dict[int, _RequestState] = {}

        logger.info(
            "QRSamplerLogitsProcessor initialized: vocab_size=%d, "
            "entropy_source=%s, amplifier=%s, temperature=%s",
            self._vocab_size,
            self._entropy_source.name,
            self._default_config.signal_amplifier_type,
            self._default_config.temperature_strategy,
        )

    @staticmethod
    def _extract_vocab_size(vllm_config: Any) -> int:
        """Extract vocabulary size from vLLM config, with fallback.

        Args:
            vllm_config: vLLM config object, or ``None`` for tests.

        Returns:
            Vocabulary size as integer.
        """
        if vllm_config is None:
            return _DEFAULT_VOCAB_SIZE

        # vLLM V1: vllm_config.model_config.hf_text_config.vocab_size
        try:
            return int(vllm_config.model_config.hf_text_config.vocab_size)
        except AttributeError:
            pass

        # Try direct vocab_size attribute.
        try:
            return int(vllm_config.vocab_size)
        except AttributeError:
            pass

        logger.warning(
            "Could not extract vocab_size from vllm_config, using default %d",
            _DEFAULT_VOCAB_SIZE,
        )
        return _DEFAULT_VOCAB_SIZE

    def _create_onehot_template(self) -> Any:
        """Create the one-hot template tensor filled with -inf.

        Returns:
            A tensor of shape ``(vocab_size,)`` filled with ``-inf``,
            or a numpy array if torch is unavailable.
        """
        try:
            import torch

            return torch.full(
                (self._vocab_size,),
                float("-inf"),
                device=self._device,
                dtype=torch.float32,
            )
        except ImportError:
            return np.full(self._vocab_size, float("-inf"), dtype=np.float32)

    def _create_cpu_buffer(self) -> Any:
        """Create a pinned-memory CPU buffer for transfers.

        Returns:
            A pinned tensor if ``is_pin_memory`` is True and torch is available,
            otherwise ``None``.
        """
        if not self._is_pin_memory:
            return None
        try:
            import torch

            return torch.empty(self._vocab_size, dtype=torch.float32, pin_memory=True)
        except ImportError:
            return None

    def is_argmax_invariant(self) -> bool:
        """Return ``False`` — this processor fundamentally changes token selection.

        This ensures the processor runs before penalties and temperature scaling
        in the vLLM pipeline, operating on raw logits.
        """
        return False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validate ``qr_*`` keys in ``params.extra_args``.

        Called by vLLM at request creation time to reject bad keys early.

        Args:
            params: vLLM ``SamplingParams`` object with ``extra_args`` dict.

        Raises:
            ConfigValidationError: If any ``qr_*`` key is unknown or
                non-overridable.
        """
        extra_args = getattr(params, "extra_args", None) or {}
        if extra_args:
            validate_extra_args(extra_args)

    def update_state(self, batch_update: Any | None) -> None:
        """Process batch composition changes.

        Must be called every engine step before ``apply()``. Processes
        changes in the required order: removed → moved → added.

        Args:
            batch_update: A ``BatchUpdate`` with ``removed``, ``moved``,
                and ``added`` sequences, or ``None`` if no changes.
        """
        if batch_update is None:
            return

        # 1. Process removals.
        for removed in getattr(batch_update, "removed", []):
            req_idx = removed if isinstance(removed, int) else getattr(removed, "req_index", None)
            if req_idx is not None:
                self._request_states.pop(req_idx, None)

        # 2. Process moves (index reassignments).
        for moved in getattr(batch_update, "moved", []):
            if hasattr(moved, "src_index") and hasattr(moved, "dst_index"):
                state = self._request_states.pop(moved.src_index, None)
                if state is not None:
                    self._request_states[moved.dst_index] = state

        # 3. Process additions.
        for added in getattr(batch_update, "added", []):
            req_idx = getattr(added, "req_index", None)
            if req_idx is None:
                continue

            extra_args = getattr(getattr(added, "sampling_params", None), "extra_args", None) or {}

            # Resolve per-request config.
            req_config = resolve_config(self._default_config, extra_args)

            # Build per-request components if config differs from default.
            if req_config is self._default_config:
                amplifier = self._default_amplifier
                strategy = self._default_strategy
                hash_str = self._default_config_hash
            else:
                amplifier = AmplifierRegistry.build(req_config)
                strategy = TemperatureStrategyRegistry.build(req_config, self._vocab_size)
                hash_str = _config_hash(req_config)

            self._request_states[req_idx] = _RequestState(
                config=req_config,
                amplifier=amplifier,
                strategy=strategy,
                config_hash_str=hash_str,
            )

    def apply(self, logits: Any) -> Any:
        """Run the full sampling pipeline on each row of the logit tensor.

        For each request in the batch:
            1. Resolve per-request config
            2. Convert logit row to numpy
            3. Compute temperature
            4. Fetch entropy just-in-time
            5. Amplify to uniform float
            6. Select token via CDF
            7. Force one-hot logits
            8. Log sampling record

        Args:
            logits: 2-D tensor of shape ``(num_requests, vocab_size)``.
                May be a ``torch.Tensor`` or a ``numpy.ndarray``.

        Returns:
            The modified logits tensor (in-place).
        """
        # Determine batch size.
        if hasattr(logits, "shape"):
            num_requests = logits.shape[0] if len(logits.shape) > 1 else 1
        else:
            return logits

        if num_requests == 0:
            return logits

        is_numpy = isinstance(logits, np.ndarray)
        is_1d = len(logits.shape) == 1

        for i in range(num_requests):
            t_start_ns = time.perf_counter_ns()

            # Get per-request state or fall back to defaults.
            state = self._request_states.get(i)
            if state is not None:
                config = state.config
                amplifier = state.amplifier
                strategy = state.strategy
                hash_str = state.config_hash_str
            else:
                config = self._default_config
                amplifier = self._default_amplifier
                strategy = self._default_strategy
                hash_str = self._default_config_hash

            # --- Extract row as numpy ---
            if is_1d:
                row = logits if is_numpy else self._to_numpy(logits)
            else:
                row = logits[i] if is_numpy else self._to_numpy(logits[i])

            # --- 1. Compute temperature ---
            temp_result = strategy.compute_temperature(row, config)

            # --- 2. Fetch entropy just-in-time ---
            t_fetch_start = time.perf_counter_ns()
            entropy_is_fallback = False
            entropy_source_name = self._entropy_source.name

            raw_bytes = self._entropy_source.get_random_bytes(config.sample_count)

            # Detect if fallback was used.
            if isinstance(self._entropy_source, FallbackEntropySource):
                entropy_source_name = self._entropy_source.last_source_used
                # If the primary source name is a compound name, the last source
                # used will be the fallback's name when it was triggered.
                entropy_is_fallback = (
                    self._entropy_source.last_source_used != self._entropy_source._primary.name
                )

            t_fetch_end = time.perf_counter_ns()
            entropy_fetch_ms = (t_fetch_end - t_fetch_start) / 1_000_000.0

            # --- 3. Amplify to uniform float ---
            amp_result = amplifier.amplify(raw_bytes)

            # --- 4. Select token via CDF ---
            selection = self._selector.select(
                row,
                temp_result.temperature,
                config.top_k,
                config.top_p,
                amp_result.u,
            )

            # --- 5. Force one-hot logits ---
            if is_1d:
                self._force_onehot(logits, selection.token_id, is_numpy)
            else:
                self._force_onehot_row(logits, i, selection.token_id, is_numpy)

            # --- 6. Log sampling record ---
            t_end_ns = time.perf_counter_ns()
            total_sampling_ms = (t_end_ns - t_start_ns) / 1_000_000.0

            record = TokenSamplingRecord(
                timestamp_ns=t_start_ns,
                entropy_fetch_ms=entropy_fetch_ms,
                total_sampling_ms=total_sampling_ms,
                entropy_source_used=entropy_source_name,
                entropy_is_fallback=entropy_is_fallback,
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
                config_hash=hash_str,
            )
            self._logger.log_token(record)

        return logits

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Convert a tensor to a numpy array with zero-copy where possible.

        Args:
            tensor: A torch.Tensor or numpy array.

        Returns:
            Numpy array view (if CPU tensor) or copy.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        # .cpu() moves GPU tensors (CUDA/MPS) to host memory; no-op on CPU.
        try:
            result: np.ndarray = tensor.detach().cpu().numpy()
            return result
        except AttributeError:
            return np.asarray(tensor)

    def _force_onehot(self, logits: Any, token_id: int, is_numpy: bool) -> None:
        """Force 1-D logits to one-hot: all -inf except token_id = 0.0.

        Args:
            logits: 1-D logit array or tensor.
            token_id: The selected token index.
            is_numpy: Whether logits is a numpy array.
        """
        if is_numpy:
            logits[:] = float("-inf")
            logits[token_id] = 0.0
        else:
            logits.copy_(self._onehot_template, non_blocking=True)
            logits[token_id] = 0.0

    def _force_onehot_row(self, logits: Any, row_idx: int, token_id: int, is_numpy: bool) -> None:
        """Force a batch row to one-hot: all -inf except token_id = 0.0.

        Args:
            logits: 2-D logit array or tensor.
            row_idx: Batch row index.
            token_id: The selected token index.
            is_numpy: Whether logits is a numpy array.
        """
        if is_numpy:
            logits[row_idx, :] = float("-inf")
            logits[row_idx, token_id] = 0.0
        else:
            logits[row_idx].copy_(self._onehot_template, non_blocking=True)
            logits[row_idx, token_id] = 0.0

    @property
    def entropy_source(self) -> EntropySource:
        """The active entropy source (may be a FallbackEntropySource wrapper)."""
        return self._entropy_source

    @property
    def default_config(self) -> QRSamplerConfig:
        """The default configuration loaded from environment."""
        return self._default_config

    @property
    def sampling_logger(self) -> SamplingLogger:
        """The diagnostic logger for this processor."""
        return self._logger

    def close(self) -> None:
        """Release all resources held by the processor."""
        self._entropy_source.close()

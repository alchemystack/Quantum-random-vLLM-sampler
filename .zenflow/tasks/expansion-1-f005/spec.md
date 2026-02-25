# Technical Specification: qr-sampler

## 1. Technical Context

### 1.1 Language & Runtime

- **Python 3.10+** — union syntax `X | Y`, `match` statements, `typing.ParamSpec`
- **Target runtime**: vLLM V1 inference engine process (Linux, CUDA GPU host)
- **Package name**: `qr-sampler` (PyPI), `qr_sampler` (import)

### 1.2 Key Dependencies

| Dependency | Version | Purpose | Optional? |
|---|---|---|---|
| `numpy` | >=1.24.0 | Array math, softmax, CDF | No |
| `pydantic-settings` | >=2.0.0 | Layered config from env/file/kwargs | No |
| `pydantic` | >=2.0.0 | Validation, field descriptions | No (transitive) |
| `grpcio` | >=1.60.0 | gRPC entropy transport | Yes (`[grpc]` extra) |
| `protobuf` | >=4.21.0 | Proto message serialization | Yes (`[grpc]` extra) |
| `torch` | — | Tensor ops in processor hot path | Implicit (provided by vLLM) |
| `pytest` | >=7.0 | Test framework | Yes (`[dev]` extra) |
| `pytest-cov` | >=4.0 | Coverage reporting | Yes (`[dev]` extra) |
| `scipy` | >=1.10.0 | Statistical validation tests (KS-test) | Yes (`[dev]` extra) |
| `ruff` | >=0.4.0 | Linting and formatting | Yes (`[dev]` extra) |
| `mypy` | >=1.8.0 | Strict type checking | Yes (`[dev]` extra) |
| `pre-commit` | >=3.0 | Git hook management | Yes (`[dev]` extra) |
| `bandit` | >=1.7.0 | Security scanning | Yes (`[dev]` extra) |

### 1.3 vLLM V1 Integration Contract

The processor must implement vLLM V1's batch-level `LogitsProcessor` ABC:

```python
class LogitsProcessor(ABC):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool): ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...
    def update_state(self, batch_update: BatchUpdate | None) -> None: ...
    def is_argmax_invariant(self) -> bool: ...
    @classmethod
    def validate_params(cls, sampling_params) -> None: ...
```

**Pipeline position**: `is_argmax_invariant() = False` places the processor before penalties and temperature scaling, ensuring it operates on raw logits. The processor performs its own temperature scaling, top-k/top-p, and token selection internally, then forces a one-hot logit vector so vLLM's downstream sampler is a no-op.

**Registration**: Via `pyproject.toml` entry point `[project.entry-points."vllm.logits_processors"]`.

**Batch contract**: `BatchUpdate` carries `removed`, `moved`, `added` sequences processed in that order. The `output_tok_ids` list in `AddedRequest` is a live reference that grows each step.

---

## 2. Implementation Approach

### 2.1 Ground-Up Redesign Strategy

This is a **clean-room reimplementation**, not a migration. The existing `qc-sampler` prototype informs algorithm choices and test strategies, but every module is written fresh for the `qr_sampler` namespace.

**Algorithms carried forward** (proven correct in prototype):
- Z-score mean amplification: bytes → sample mean → z-score → Φ(z) → clamped uniform
- CDF-based token selection: descending-probability CDF + searchsorted
- EDT temperature: Shannon entropy → normalized entropy → power-law scaling → clamped
- Shannon entropy computation: numerically stable softmax + Σ(-p ln p)

**Patterns carried forward** (improved):
- ABC + registry for all extensible components
- FallbackEntropySource as composition wrapper (only catches `EntropyUnavailableError`)
- Frozen/immutable result types for pipeline data
- Per-request config resolution that never mutates defaults

**Major changes from prototype**:
- `pydantic-settings` replaces manual frozen-dataclass config with env parsing
- `src/` layout replaces flat layout
- Modular subpackages (`entropy/`, `amplification/`, `temperature/`, `selection/`, `logging/`) replace flat modules
- Three gRPC transport modes (unary, server-streaming, bidirectional) replace unary-only
- `grpc.aio` async transport replaces thread-based prefetch
- Entry-point registry for third-party entropy sources replaces internal-only registry
- Circuit breaker with adaptive timeout replaces fixed retry logic
- `out=` parameter on `get_random_float64()` for zero-allocation hot path

### 2.2 Performance Philosophy

The `apply()` method is on the critical path for every token in every request. Design principles:

1. **Pre-allocate everything in `__init__()`** — tensors, buffers, numpy arrays
2. **Zero-copy NumPy ↔ PyTorch** — `torch.from_numpy()` / `.numpy()`, never `torch.tensor()`
3. **In-place tensor modification** — `logits[i] = ...`, never allocate new tensors
4. **`non_blocking=True`** for all CPU→GPU transfers with pinned memory
5. **Short-circuit** when batch is empty
6. **`__slots__`** on frequently instantiated per-token objects
7. **Vectorize across batch dimension** where possible (batch-level operations before per-row loop)
8. **No allocation in hot loop** — reuse pre-allocated numpy buffers via `out=` parameters

### 2.3 Just-In-Time Entropy Architecture

The fundamental constraint: entropy must be physically generated AFTER logits are computed. No prefetch pools, no pre-generated buffers.

**How the three gRPC modes satisfy this**:

| Mode | Connection | Per-Token Flow | When to Use |
|---|---|---|---|
| **Unary** | New HTTP/2 stream per call | `GetEntropy(request) → response` | Simple servers, getting started, debugging |
| **Server-streaming** | Persistent stream, client sends one config request | Client waits on `stream.read()` after logits ready | Pre-configured byte count, moderate latency |
| **Bidirectional** | Persistent bidirectional stream | Client sends `generate_now` signal → server responds | Production, lowest latency (~50-100μs same-machine) |

All modes send the request only when logits are available. The mode affects connection management overhead, not entropy freshness.

**Circuit breaker**: Tracks rolling P99 latency. Timeout = `max(5ms, P99 × 1.5)`. On timeout, falls back to secondary source with metadata flag.

---

## 3. Source Code Structure

### 3.1 Package Layout

```
src/
└── qr_sampler/
    ├── __init__.py                    # Package metadata, version, convenience re-exports
    ├── py.typed                       # PEP 561 type stub marker (empty file)
    ├── processor.py                   # QRSamplerLogitsProcessor — vLLM integration
    ├── config.py                      # QRSamplerConfig (pydantic-settings BaseSettings)
    ├── exceptions.py                  # Exception hierarchy rooted in QRSamplerError
    │
    ├── entropy/                       # Entropy source subsystem
    │   ├── __init__.py                # Re-exports: EntropySource, EntropySourceRegistry, built-ins
    │   ├── base.py                    # EntropySource ABC
    │   ├── registry.py                # EntropySourceRegistry (entry-point discovery + decorator registration)
    │   ├── quantum.py                 # QuantumGrpcSource (unary, server-streaming, bidi)
    │   ├── system.py                  # SystemEntropySource (os.urandom)
    │   ├── timing.py                  # TimingNoiseSource (CPU jitter — experimental)
    │   ├── mock.py                    # MockUniformSource (testing, bias simulation)
    │   └── fallback.py                # FallbackEntropySource (composition wrapper)
    │
    ├── amplification/                 # Signal amplification subsystem
    │   ├── __init__.py                # Re-exports: SignalAmplifier, AmplificationResult, built-ins
    │   ├── base.py                    # SignalAmplifier ABC + AmplificationResult
    │   ├── registry.py                # AmplifierRegistry (decorator-based)
    │   └── zscore.py                  # ZScoreMeanAmplifier
    │
    ├── temperature/                   # Temperature strategy subsystem
    │   ├── __init__.py                # Re-exports: TemperatureStrategy, TemperatureResult, built-ins
    │   ├── base.py                    # TemperatureStrategy ABC + TemperatureResult + shannon_entropy helper
    │   ├── registry.py                # TemperatureStrategyRegistry (decorator-based)
    │   ├── fixed.py                   # FixedTemperatureStrategy
    │   └── edt.py                     # EDTTemperatureStrategy
    │
    ├── selection/                     # Token selection subsystem
    │   ├── __init__.py                # Re-exports: TokenSelector, SelectionResult
    │   ├── selector.py                # TokenSelector (top-k → softmax → top-p → descending CDF → searchsorted)
    │   └── types.py                   # SelectionResult frozen dataclass
    │
    ├── logging/                       # Diagnostic logging subsystem
    │   ├── __init__.py                # Re-exports: SamplingLogger, TokenSamplingRecord
    │   ├── logger.py                  # SamplingLogger (none/summary/full log levels)
    │   └── types.py                   # TokenSamplingRecord frozen dataclass
    │
    └── proto/                         # gRPC protocol definitions
        ├── __init__.py
        ├── entropy_service.proto      # Service + message definitions
        ├── entropy_service_pb2.py     # Hand-written protobuf message stubs
        └── entropy_service_pb2_grpc.py # Hand-written gRPC client stubs (all 3 modes)
```

### 3.2 Test Layout

```
tests/
├── __init__.py
├── conftest.py                        # Shared fixtures: configs, mock sources, sample logits
├── test_config.py                     # Config defaults, env loading, per-request resolution, validation
├── test_entropy/
│   ├── __init__.py
│   ├── test_system.py                 # SystemEntropySource behavior
│   ├── test_quantum.py                # QuantumGrpcSource (mocked gRPC, all 3 modes)
│   ├── test_timing.py                 # TimingNoiseSource behavior
│   ├── test_mock.py                   # MockUniformSource reproducibility, bias simulation
│   ├── test_fallback.py               # FallbackEntropySource delegation, error handling
│   └── test_registry.py              # Entry-point discovery, lazy instantiation
├── test_amplification/
│   ├── __init__.py
│   └── test_zscore.py                 # Known values, SEM derivation, edge cases
├── test_temperature/
│   ├── __init__.py
│   ├── test_fixed.py                  # Constant temperature, entropy still computed
│   └── test_edt.py                    # Monotonicity, clamping, exponent effects
├── test_selection/
│   ├── __init__.py
│   └── test_selector.py              # CDF selection, top-k/top-p, edge cases
├── test_processor.py                  # Full integration: pipeline, batch updates, one-hot
└── test_statistical_properties.py     # KS-test uniformity, bias detection, EDT correlation (scipy)
```

### 3.3 Examples and Deployment

```
examples/
├── servers/
│   ├── simple_urandom_server.py       # ~50 lines, os.urandom over gRPC (unary + streaming)
│   ├── timing_noise_server.py         # CPU jitter entropy server
│   └── qrng_template_server.py        # Annotated template with TODO markers
├── docker/
│   ├── Dockerfile.entropy-server      # Container for any example server
│   └── docker-compose.yml             # Full stack: entropy server + vLLM + qr-sampler
└── systemd/
    └── qr-entropy-server.service      # systemd unit file template
```

### 3.4 Project Root Files

```
qr-sampler/
├── pyproject.toml                     # Build config, dependencies, entry points, tool config
├── README.md                          # Comprehensive documentation (see REQ-DOC-01)
├── CHANGELOG.md                       # Keep a Changelog format
├── CONTRIBUTING.md                    # Dev setup, conventions, PR process
├── CODE_OF_CONDUCT.md                 # Contributor Covenant v2.1
├── LICENSE                            # Apache 2.0
├── CLAUDE.md                          # Updated codebase guide for coding agents
├── .gitignore                         # Python, venv, build, IDE, OS artifacts
├── .pre-commit-config.yaml            # ruff, mypy, bandit, standard hooks
└── .github/
    ├── workflows/
    │   └── ci.yml                     # Test matrix (3.10-3.13), lint, type-check, coverage
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.yml
    │   └── feature_request.yml
    ├── PULL_REQUEST_TEMPLATE.md
    └── dependabot.yml
```

---

## 4. Detailed Component Specifications

### 4.1 Configuration System (`config.py`)

**Class: `QRSamplerConfig(BaseSettings)`**

Uses `pydantic-settings` for declarative, layered configuration.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class QRSamplerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Infrastructure (NOT per-request overridable) ---
    grpc_server_address: str = Field(
        default="localhost:50051",
        description="gRPC entropy server address (host:port or unix:///path)"
    )
    grpc_timeout_ms: float = Field(
        default=5000.0,
        description="gRPC call timeout in milliseconds"
    )
    grpc_retry_count: int = Field(
        default=2,
        description="Number of retries after gRPC failure"
    )
    grpc_mode: str = Field(
        default="unary",
        description="gRPC transport mode: 'unary', 'server_streaming', 'bidi_streaming'"
    )
    fallback_mode: str = Field(
        default="system",
        description="Fallback entropy source: 'error', 'system', 'mock_uniform'"
    )
    entropy_source_type: str = Field(
        default="quantum_grpc",
        description="Primary entropy source identifier"
    )

    # --- Signal Amplification (per-request overridable) ---
    signal_amplifier_type: str = Field(
        default="zscore_mean",
        description="Signal amplification algorithm"
    )
    sample_count: int = Field(
        default=20480,
        description="Number of entropy bytes to fetch per token"
    )
    population_mean: float = Field(
        default=127.5,
        description="Null-hypothesis mean of byte values {0..255}"
    )
    population_std: float = Field(
        default=73.61215932167728,
        description="Population std for continuous uniform [0, 255]"
    )
    uniform_clamp_epsilon: float = Field(
        default=1e-10,
        description="Clamp u to (ε, 1−ε) to avoid degenerate CDF"
    )

    # --- Temperature Strategy (per-request overridable) ---
    temperature_strategy: str = Field(
        default="fixed",
        description="Temperature strategy: 'fixed' or 'edt'"
    )
    fixed_temperature: float = Field(
        default=0.7,
        description="Constant temperature for fixed strategy"
    )
    edt_base_temp: float = Field(
        default=0.8,
        description="Base coefficient for EDT"
    )
    edt_exponent: float = Field(
        default=0.5,
        description="Power-law exponent for EDT"
    )
    edt_min_temp: float = Field(
        default=0.1,
        description="EDT temperature floor"
    )
    edt_max_temp: float = Field(
        default=2.0,
        description="EDT temperature ceiling"
    )

    # --- Token Selection (per-request overridable) ---
    top_k: int = Field(
        default=50,
        description="Top-k filtering (≤0 disables)"
    )
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling threshold (1.0 disables)"
    )

    # --- Logging (per-request overridable) ---
    log_level: str = Field(
        default="summary",
        description="Logging verbosity: 'none', 'summary', 'full'"
    )
    diagnostic_mode: bool = Field(
        default=False,
        description="Store all token records in memory for analysis"
    )
```

**Per-request override mechanism**:

```python
_PER_REQUEST_FIELDS: frozenset[str] = frozenset({
    "signal_amplifier_type", "sample_count",
    "population_mean", "population_std", "uniform_clamp_epsilon",
    "temperature_strategy", "fixed_temperature",
    "edt_base_temp", "edt_exponent", "edt_min_temp", "edt_max_temp",
    "top_k", "top_p",
    "log_level", "diagnostic_mode",
})

def resolve_config(
    defaults: QRSamplerConfig, extra_args: dict[str, Any] | None
) -> QRSamplerConfig:
    """Create a new config instance merging defaults with per-request overrides.

    extra_args keys use 'qr_' prefix (e.g., 'qr_top_k': 100).
    Only fields in _PER_REQUEST_FIELDS are overridable.
    Raises ConfigValidationError for invalid keys or non-overridable fields.
    """

def validate_extra_args(extra_args: dict[str, Any]) -> None:
    """Validate all qr_* keys in extra_args without creating a config.
    Called by validate_params() at request creation time.
    """
```

**Resolution chain**: init kwargs → env vars (`QR_*`) → `.env` file → field defaults. Per-request overrides via `SamplingParams.extra_args` with `qr_` prefix create a new config instance per request.

### 4.2 Exception Hierarchy (`exceptions.py`)

```python
class QRSamplerError(Exception):
    """Base exception for all qr-sampler errors."""

class EntropyUnavailableError(QRSamplerError):
    """No entropy source can provide bytes (primary and fallback both failed)."""

class ConfigValidationError(QRSamplerError):
    """Configuration field validation failed."""

class SignalAmplificationError(QRSamplerError):
    """Signal amplification computation failed (e.g., empty input)."""

class TokenSelectionError(QRSamplerError):
    """Token selection failed (e.g., no candidates survive filtering)."""
```

### 4.3 Entropy Source ABC (`entropy/base.py`)

```python
from abc import ABC, abstractmethod

class EntropySource(ABC):
    """Abstract base for all entropy sources.

    Implementations must provide random bytes on demand.
    The get_random_bytes() call must satisfy the just-in-time constraint:
    physical entropy generation occurs only when this method is called.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source identifier (e.g., 'quantum_grpc', 'system')."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the source can currently provide entropy."""

    @abstractmethod
    def get_random_bytes(self, n: int) -> bytes:
        """Return exactly n random bytes.

        Raises:
            EntropyUnavailableError: If the source cannot provide bytes.
        """

    @abstractmethod
    def get_random_float64(
        self,
        shape: tuple[int, ...],
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return random float64 values in [0, 1).

        If out is provided, write into it (zero-allocation hot path).
        If out is None, allocate and return a new array.

        The default implementation in the ABC converts get_random_bytes()
        to float64 via np.frombuffer(dtype=uint8) / 255.0. Subclasses
        may override for more efficient native float generation.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources (channels, connections, file handles)."""

    def health_check(self) -> dict[str, Any]:
        """Return status dict. Default: {'source': self.name, 'healthy': self.is_available}."""
        return {"source": self.name, "healthy": self.is_available}
```

**Design note**: The ABC provides a default `get_random_float64()` implementation that delegates to `get_random_bytes()`. High-performance sources (like `QuantumGrpcSource`) can override this for native float generation. The `out=` parameter enables zero-allocation operation in the hot path.

### 4.4 Entropy Source Registry (`entropy/registry.py`)

```python
class EntropySourceRegistry:
    """Registry for entropy source classes.

    Discovery chain:
    1. Built-in sources registered via @register_entropy_source decorator
    2. Third-party sources discovered via 'qr_sampler.entropy_sources' entry points
    3. Sources lazily instantiated on first use
    """

    _registry: dict[str, type[EntropySource]]  # name → class
    _entry_points_loaded: bool

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a source class under a string key."""

    @classmethod
    def get(cls, name: str) -> type[EntropySource]:
        """Look up source class by name. Loads entry points on first call."""

    @classmethod
    def list_available(cls) -> list[str]:
        """Return all registered source names."""

    @classmethod
    def _load_entry_points(cls) -> None:
        """Discover and register sources from 'qr_sampler.entropy_sources' entry point group."""
```

**Entry-point discovery**: Uses `importlib.metadata.entry_points(group="qr_sampler.entropy_sources")`. Entry points are loaded lazily on the first `get()` call. Built-in sources are registered at module import time via the `@register_entropy_source` decorator, so they are always available even if entry-point loading fails.

### 4.5 QuantumGrpcSource (`entropy/quantum.py`)

This is the primary production entropy source. It supports three transport modes via `grpc.aio`.

```python
class QuantumGrpcSource(EntropySource):
    """gRPC entropy source with configurable transport mode.

    All modes satisfy the just-in-time constraint: the gRPC request
    is only sent when get_random_bytes() is called (i.e., after logits
    are available). The transport mode affects connection management
    overhead, not entropy freshness.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        """
        1. Parse server address (TCP vs Unix socket from format)
        2. Create grpc.aio.Channel with keepalive options
        3. Create EntropyServiceStub
        4. Initialize circuit breaker state (rolling P99 window)
        5. Initialize mode-specific state:
           - unary: nothing extra
           - server_streaming: stream handle (lazy)
           - bidi_streaming: request queue + response iterator (lazy)
        """

    # --- Transport mode implementations ---

    async def _fetch_unary(self, n: int) -> bytes:
        """Single request-response per call. Simplest. Higher overhead."""

    async def _fetch_server_streaming(self, n: int) -> bytes:
        """Client sends one initial request, reads from persistent response stream."""

    async def _fetch_bidi_streaming(self, n: int) -> bytes:
        """Client sends 'generate now' signal on persistent bidi stream, reads response."""

    # --- Circuit breaker ---

    def _update_latency(self, elapsed_ms: float) -> None:
        """Add to rolling window, recompute P99."""

    def _get_timeout(self) -> float:
        """Return max(5ms, P99 * 1.5) or configured timeout, whichever is smaller."""

    # --- Sync wrapper ---

    def get_random_bytes(self, n: int) -> bytes:
        """Synchronous wrapper around the async transport.

        Uses asyncio event loop:
        - If a loop is running: schedule coroutine via run_coroutine_threadsafe
        - If no loop: create one via asyncio.run()

        This is necessary because vLLM's LogitsProcessor.apply() is called
        synchronously from the sampling pipeline.
        """
```

**Async ↔ sync bridge**: vLLM calls `apply()` synchronously. The `QuantumGrpcSource` uses `grpc.aio` internally for transport efficiency but wraps the async calls in a synchronous interface. The approach:
- Maintain a dedicated background event loop thread for the gRPC channel
- Use `asyncio.run_coroutine_threadsafe()` to dispatch requests from the sync `get_random_bytes()` to the async loop
- The background loop is created once in `__init__()` and cleaned up in `close()`

**Channel configuration**:
```python
options = [
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 0),
]
# For Unix sockets: address = "unix:///var/run/qrng.sock"
# For TCP: address = "host:port"
```

**Circuit breaker state**:
```python
self._latency_window: deque[float]  # Rolling window of recent latencies (max 100)
self._p99_ms: float                 # Current P99 estimate
self._consecutive_failures: int     # For fast-fail after repeated failures
self._circuit_open: bool            # When True, skip primary and go direct to fallback
self._circuit_open_until: float     # Timestamp when circuit closes (half-open)
```

### 4.6 SystemEntropySource (`entropy/system.py`)

```python
@register_entropy_source("system")
class SystemEntropySource(EntropySource):
    """os.urandom() wrapper. Always available, cryptographically secure, not quantum."""

    @property
    def name(self) -> str: return "system"

    @property
    def is_available(self) -> bool: return True

    def get_random_bytes(self, n: int) -> bytes:
        return os.urandom(n)

    def close(self) -> None: pass
```

### 4.7 TimingNoiseSource (`entropy/timing.py`)

```python
@register_entropy_source("timing_noise")
class TimingNoiseSource(EntropySource):
    """CPU timing jitter entropy source.

    Derives entropy from variations in instruction execution timing.
    Each byte is generated by timing a series of operations and
    extracting the least-significant bits of the timing deltas.

    WARNING: This is EXPERIMENTAL/EDUCATIONAL. Not suitable for
    consciousness-research experiments (timing noise is deterministic
    under classical physics). Useful for testing, demos, and as an
    example of a non-gRPC entropy source.
    """

    def get_random_bytes(self, n: int) -> bytes:
        """Generate n bytes from CPU timing jitter.

        Algorithm:
        1. For each byte, perform 8 timing measurements
        2. Each measurement: time a tight loop of hash operations
        3. Extract LSB of nanosecond delta
        4. Combine 8 bits into one byte
        """
```

### 4.8 MockUniformSource (`entropy/mock.py`)

```python
@register_entropy_source("mock_uniform")
class MockUniformSource(EntropySource):
    """Configurable mock entropy source for testing.

    Generates bytes from a normal distribution with configurable mean.
    Supports seeded reproducibility for deterministic tests.

    Usage:
    - Null hypothesis testing: mean=127.5 (no bias)
    - Consciousness bias simulation: mean=128.0 (positive bias)
    """

    _MOCK_BYTE_STD: float = 40.0  # Fixed std for test consistency

    def __init__(self, mean: float = 127.5, seed: int | None = None) -> None: ...
    def get_random_bytes(self, n: int) -> bytes: ...
```

### 4.9 FallbackEntropySource (`entropy/fallback.py`)

```python
class FallbackEntropySource(EntropySource):
    """Composition wrapper: tries primary, falls back on EntropyUnavailableError.

    Only catches EntropyUnavailableError. All other exceptions propagate.
    Reports which source was actually used via metadata.
    """

    def __init__(self, primary: EntropySource, fallback: EntropySource) -> None: ...

    @property
    def last_source_used(self) -> str:
        """Name of the source that provided bytes on the last call."""

    def get_random_bytes(self, n: int) -> bytes:
        """Try primary. On EntropyUnavailableError, log warning and use fallback."""
```

### 4.10 Signal Amplification (`amplification/`)

**AmplificationResult** (`base.py`):
```python
@dataclass(frozen=True, slots=True)
class AmplificationResult:
    u: float                              # Uniform value in (ε, 1−ε)
    diagnostics: dict[str, Any]           # sample_mean, z_score, sem, sample_count
```

**SignalAmplifier ABC** (`base.py`):
```python
class SignalAmplifier(ABC):
    @abstractmethod
    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Convert raw entropy bytes into a uniform float u ∈ (ε, 1−ε)."""
```

**ZScoreMeanAmplifier** (`zscore.py`):
```python
@register_amplifier("zscore_mean")
class ZScoreMeanAmplifier(SignalAmplifier):
    """Z-score signal amplification.

    Algorithm:
    1. Interpret raw_bytes as uint8 array: samples = np.frombuffer(raw_bytes, dtype=np.uint8)
    2. Sample mean: M = np.mean(samples)
    3. Standard error of mean (derived, never stored):
       SEM = population_std / sqrt(len(samples))
    4. Z-score: z = (M − population_mean) / SEM
    5. Normal CDF: u = 0.5 × (1 + erf(z / √2))
    6. Clamp to (ε, 1−ε)

    Under null hypothesis (no consciousness influence):
    - M ~ N(population_mean, SEM), so z ~ N(0, 1), so u ~ Uniform(0, 1)

    Under consciousness influence (e.g., +0.003 mean shift per byte):
    - 20,480 bytes × +0.003 → M ≈ 127.56 (cumulative shift)
    - SEM ≈ 0.5143 for 20,480 samples
    - z ≈ (127.56 − 127.5) / 0.5143 ≈ 0.12 → u ≈ 0.548
    - This biases token selection toward lower-probability tokens in the descending CDF
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        self._population_mean = config.population_mean
        self._population_std = config.population_std
        self._clamp_epsilon = config.uniform_clamp_epsilon

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Raises SignalAmplificationError if raw_bytes is empty."""
```

**AmplifierRegistry** (`registry.py`):
```python
class AmplifierRegistry:
    """Registry for signal amplifier classes. Decorator-based registration."""
    _registry: dict[str, type[SignalAmplifier]]

    @classmethod
    def register(cls, name: str) -> Callable: ...

    @classmethod
    def get(cls, name: str) -> type[SignalAmplifier]: ...

    @classmethod
    def build(cls, config: QRSamplerConfig) -> SignalAmplifier:
        """Look up and instantiate amplifier from config.signal_amplifier_type."""
```

### 4.11 Temperature Strategies (`temperature/`)

**TemperatureResult** (`base.py`):
```python
@dataclass(frozen=True, slots=True)
class TemperatureResult:
    temperature: float         # Temperature to use for this token
    shannon_entropy: float     # Shannon entropy of logit distribution (nats)
    diagnostics: dict[str, Any]
```

**TemperatureStrategy ABC** (`base.py`):
```python
class TemperatureStrategy(ABC):
    @abstractmethod
    def compute_temperature(
        self, logits: np.ndarray, config: QRSamplerConfig
    ) -> TemperatureResult:
        """Compute temperature for this token. Must always include shannon_entropy."""

def compute_shannon_entropy(logits: np.ndarray) -> float:
    """H = -Σ(p_i × ln(p_i)) using numerically stable softmax.
    Returns 0.0 for degenerate distributions (single non-zero prob).
    """
```

**FixedTemperatureStrategy** (`fixed.py`):
```python
@register_temperature_strategy("fixed")
class FixedTemperatureStrategy(TemperatureStrategy):
    """Returns config.fixed_temperature regardless of logit distribution."""
```

**EDTTemperatureStrategy** (`edt.py`):
```python
@register_temperature_strategy("edt")
class EDTTemperatureStrategy(TemperatureStrategy):
    """Entropy-based Dynamic Temperature.

    Formula:
    H_norm = shannon_entropy / ln(vocab_size)   # Normalized to [0, 1]
    T = edt_base_temp × H_norm^edt_exponent
    T = clamp(T, edt_min_temp, edt_max_temp)

    Behavior:
    - Low entropy (peaked distribution) → low temperature → sharper sampling
    - High entropy (flat distribution) → high temperature → more exploration
    - edt_exponent < 1 (concave): temperature rises quickly with entropy
    - edt_exponent > 1 (convex): temperature rises slowly with entropy
    """

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self._max_entropy = math.log(vocab_size)  # H_max = ln(V)
```

### 4.12 Token Selection (`selection/`)

**SelectionResult** (`types.py`):
```python
@dataclass(frozen=True, slots=True)
class SelectionResult:
    token_id: int         # Vocabulary index of selected token
    token_rank: int       # Rank in probability-sorted candidates (0 = most probable)
    token_prob: float     # Probability of selected token (after filtering)
    num_candidates: int   # Number of tokens surviving top-k and top-p
    diagnostics: dict[str, Any]
```

**TokenSelector** (`selector.py`):
```python
class TokenSelector:
    """Stateless CDF-based token selector.

    Pipeline:
    1. Temperature scaling: logits / T
    2. Top-k: argpartition, mask rest to -inf
    3. Softmax (numerically stable, shift-by-max)
    4. Top-p (nucleus): minimal set with cumsum ≥ p, renormalize
    5. Sort by descending probability
    6. Build CDF: cumsum of sorted probs
    7. Binary search: np.searchsorted(cdf, u, side='left')

    Semantic interpretation of u:
    - u → 0.0: selects most probable token (conservative)
    - u → 1.0: selects least probable token (creative/surprising)
    - Any systematic bias in u has coherent semantic direction
    """

    def select(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        u: float,
    ) -> SelectionResult:
        """Select one token. Raises TokenSelectionError if no candidates survive."""

    @staticmethod
    def _apply_top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, int]: ...

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def _apply_top_p(probs: np.ndarray, top_p: float) -> tuple[np.ndarray, int]: ...

    @staticmethod
    def _cdf_select(probs: np.ndarray, u: float) -> tuple[int, int, float, int]: ...
```

### 4.13 Diagnostic Logging (`logging/`)

**TokenSamplingRecord** (`types.py`):
```python
@dataclass(frozen=True, slots=True)
class TokenSamplingRecord:
    # Timing
    timestamp_ns: int
    entropy_fetch_ms: float
    total_sampling_ms: float

    # Entropy source
    entropy_source_used: str        # "quantum_grpc", "system", "mock_uniform", etc.
    entropy_is_fallback: bool       # True if fallback source was used

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
    config_hash: str                # 16-char SHA-256 of active config
```

**SamplingLogger** (`logger.py`):
```python
class SamplingLogger:
    """Per-token diagnostic logger.

    Log levels:
    - "none": No output (still stores if diagnostic_mode=True)
    - "summary": One-line per token with key metrics
    - "full": JSON dump of all fields

    Uses logging.getLogger("qr_sampler"). No print() statements.
    """

    def __init__(self, config: QRSamplerConfig) -> None: ...
    def log_token(self, record: TokenSamplingRecord) -> None: ...
    def get_diagnostic_data(self) -> list[TokenSamplingRecord]: ...
    def get_summary_stats(self) -> dict[str, Any]: ...
```

### 4.14 Processor (`processor.py`)

```python
class QRSamplerLogitsProcessor:
    """vLLM V1 LogitsProcessor that replaces token sampling with
    external-entropy-driven selection.

    Registered via entry point:
    [project.entry-points."vllm.logits_processors"]
    qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
    ) -> None:
        """
        1. Extract vocab_size from vllm_config
        2. Load default config from env (QRSamplerConfig())
        3. Build shared components:
           - entropy_source via registry
           - fallback wrapping if configured
           - default amplifier via AmplifierRegistry.build()
           - default temperature strategy via TemperatureStrategyRegistry.build()
           - token_selector = TokenSelector()
           - logger = SamplingLogger(config)
        4. Pre-allocate tensors:
           - _onehot_row: torch.Tensor of shape (vocab_size,) filled with -inf
           - Pinned memory CPU buffers if is_pin_memory
        5. Initialize per-request state caches
        """

    def is_argmax_invariant(self) -> bool:
        """Return False — this processor fundamentally changes token selection."""
        return False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validate qr_* keys in params.extra_args. Called at request creation."""

    def update_state(self, batch_update: Any | None) -> None:
        """Process batch changes: removed → moved → added (in order).
        For each added request: resolve per-request config, cache components.
        """

    def apply(self, logits: Any) -> Any:
        """Main sampling pipeline. For each row in batch:

        1. Get per-request config (or default)
        2. Convert row to numpy (zero-copy if possible)
        3. Compute temperature
        4. Fetch entropy (just-in-time)
        5. Amplify to uniform float
        6. Select token via CDF
        7. Force one-hot logits: row = -inf, row[token_id] = 0.0
        8. Log token record

        Returns modified logits tensor.
        """
```

**Pre-allocation pattern** (from vLLM builtins):
```python
# In __init__():
self._onehot_template = torch.full(
    (vocab_size,), float("-inf"),
    device=device, dtype=torch.float32
)
if is_pin_memory:
    self._cpu_buffer = torch.empty(
        vocab_size, dtype=torch.float32, pin_memory=True
    )

# In apply() hot path:
logits[i].copy_(self._onehot_template, non_blocking=True)
logits[i, selection.token_id] = 0.0
```

### 4.15 Proto Definitions (`proto/`)

**entropy_service.proto**:
```proto
syntax = "proto3";
package qr_entropy;

service EntropyService {
  // Simple request-response (unary mode)
  rpc GetEntropy (EntropyRequest) returns (EntropyResponse);

  // Persistent bidirectional stream (bidi and server-streaming modes)
  rpc StreamEntropy (stream EntropyRequest) returns (stream EntropyResponse);
}

message EntropyRequest {
  int32 bytes_needed = 1;          // Number of random bytes requested
  int64 sequence_id = 2;           // Correlates request to response in streaming
}

message EntropyResponse {
  bytes data = 1;                  // Random bytes
  int64 sequence_id = 2;          // Matches the request's sequence_id
  int64 generation_timestamp_ns = 3; // When physical entropy was generated
  string device_id = 4;           // QRNG hardware identifier
}
```

**Hand-written stubs**: `entropy_service_pb2.py` and `entropy_service_pb2_grpc.py` are hand-written minimal stubs (not protoc-generated) that define message classes and gRPC client stubs for both `GetEntropy` (unary) and `StreamEntropy` (bidirectional streaming). The stubs support all three transport modes through the same channel.

---

## 5. Data Model / Interface Changes

### 5.1 Public API Surface

The primary public API is the vLLM entry point. Users interact with qr-sampler through:

1. **Entry point registration**: vLLM auto-discovers and instantiates `QRSamplerLogitsProcessor`
2. **Environment variables**: `QR_*` prefix for all configuration
3. **Per-request overrides**: `SamplingParams(extra_args={"qr_top_k": 100, ...})`
4. **Third-party plugins**: Register entropy sources via `qr_sampler.entropy_sources` entry point

### 5.2 Entry Points

```toml
[project.entry-points."vllm.logits_processors"]
qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"

[project.entry-points."qr_sampler.entropy_sources"]
system = "qr_sampler.entropy.system:SystemEntropySource"
quantum_grpc = "qr_sampler.entropy.quantum:QuantumGrpcSource"
timing_noise = "qr_sampler.entropy.timing:TimingNoiseSource"
mock_uniform = "qr_sampler.entropy.mock:MockUniformSource"
```

### 5.3 Data Flow Contracts

All pipeline data flows through immutable result objects:

```
logits (torch.Tensor, batch × vocab_size)
  │
  ├─ per row ──────────────────────────────────────────────────────
  │
  ├─→ TemperatureStrategy.compute_temperature(logits_row, config)
  │   └─→ TemperatureResult { temperature, shannon_entropy, diagnostics }
  │
  ├─→ EntropySource.get_random_bytes(config.sample_count)
  │   └─→ bytes (exactly sample_count bytes)
  │
  ├─→ SignalAmplifier.amplify(raw_bytes)
  │   └─→ AmplificationResult { u, diagnostics{sample_mean, z_score, sem} }
  │
  ├─→ TokenSelector.select(logits_row, temperature, top_k, top_p, u)
  │   └─→ SelectionResult { token_id, token_rank, token_prob, num_candidates }
  │
  ├─→ logits[i] = -inf everywhere, logits[i, token_id] = 0.0
  │
  └─→ SamplingLogger.log_token(TokenSamplingRecord)
```

---

## 6. Delivery Phases

The implementation is organized into incremental, testable milestones. Each phase produces a working (partial) system that can be tested independently.

### Phase 1: Project Foundation
Set up the project skeleton, build system, tooling, and config.

- Project directory structure with `src/` layout
- `pyproject.toml` with dependencies, entry points, extras, tool configuration
- `.gitignore`, `LICENSE` (Apache 2.0), `py.typed`
- `ruff` and `mypy` configuration in `pyproject.toml`
- `.pre-commit-config.yaml`
- `exceptions.py` — full exception hierarchy
- `config.py` — `QRSamplerConfig` with pydantic-settings, `resolve_config()`, `validate_extra_args()`
- `tests/test_config.py` — config defaults, env loading, resolution, validation
- **Verification**: `pip install -e ".[dev]"`, `ruff check`, `mypy --strict src/`, `pytest tests/test_config.py`

### Phase 2: Core Pipeline Components
Implement the stateless pipeline components that don't depend on external services.

- `amplification/base.py` — ABC + `AmplificationResult`
- `amplification/registry.py` — `AmplifierRegistry`
- `amplification/zscore.py` — `ZScoreMeanAmplifier`
- `temperature/base.py` — ABC + `TemperatureResult` + `compute_shannon_entropy()`
- `temperature/registry.py` — `TemperatureStrategyRegistry`
- `temperature/fixed.py` — `FixedTemperatureStrategy`
- `temperature/edt.py` — `EDTTemperatureStrategy`
- `selection/types.py` — `SelectionResult`
- `selection/selector.py` — `TokenSelector`
- `logging/types.py` — `TokenSamplingRecord`
- `logging/logger.py` — `SamplingLogger`
- Tests for all components (known values, edge cases, immutability)
- **Verification**: `pytest tests/test_amplification/ tests/test_temperature/ tests/test_selection/`, `mypy --strict`

### Phase 3: Entropy Source System
Implement the entropy source ABC, built-in sources, fallback, and registry.

- `entropy/base.py` — `EntropySource` ABC
- `entropy/registry.py` — `EntropySourceRegistry` with entry-point discovery
- `entropy/system.py` — `SystemEntropySource`
- `entropy/timing.py` — `TimingNoiseSource`
- `entropy/mock.py` — `MockUniformSource`
- `entropy/fallback.py` — `FallbackEntropySource`
- `proto/entropy_service.proto` — updated proto with `sequence_id`, `generation_timestamp_ns`
- `proto/entropy_service_pb2.py` — hand-written message stubs
- `proto/entropy_service_pb2_grpc.py` — hand-written gRPC stubs (unary + streaming)
- `entropy/quantum.py` — `QuantumGrpcSource` with all three transport modes
- Tests for all sources (mocked gRPC, fallback delegation, registry discovery)
- **Verification**: `pytest tests/test_entropy/`, `mypy --strict`

### Phase 4: Processor Integration
Wire everything together in the vLLM LogitsProcessor.

- `processor.py` — `QRSamplerLogitsProcessor` with full pipeline
- `__init__.py` — package exports, version
- Integration tests with mock vLLM config and mock entropy
- Statistical validation tests (KS uniformity, bias detection, EDT correlation)
- **Verification**: `pytest tests/`, `mypy --strict`, `ruff check`, full test suite passes

### Phase 5: Examples and Deployment
Reference servers, Docker, systemd templates.

- `examples/servers/simple_urandom_server.py`
- `examples/servers/timing_noise_server.py`
- `examples/servers/qrng_template_server.py`
- `examples/docker/Dockerfile.entropy-server`
- `examples/docker/docker-compose.yml`
- `examples/systemd/qr-entropy-server.service`
- **Verification**: Example servers run and respond to gRPC requests

### Phase 6: Documentation and Polish
README, CONTRIBUTING, CHANGELOG, CLAUDE.md, CI.

- `README.md` — comprehensive documentation per REQ-DOC-01
- `CONTRIBUTING.md` — development guide per REQ-DOC-02
- `CHANGELOG.md` — initial release notes
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1
- `CLAUDE.md` — updated for new architecture
- `.github/workflows/ci.yml` — test matrix, lint, type-check
- `.github/ISSUE_TEMPLATE/` — bug report and feature request
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/dependabot.yml`
- **Verification**: All CI checks pass, README renders correctly

---

## 7. Verification Approach

### 7.1 Test Strategy

| Level | What | How |
|---|---|---|
| **Unit** | Individual component behavior | Known-value assertions, edge cases, frozen immutability |
| **Integration** | Full pipeline with mock entropy | `QRSamplerLogitsProcessor` with `MockUniformSource`, verify one-hot output |
| **Statistical** | Mathematical property validation | KS-test for u-value uniformity, bias detection, EDT monotonicity (scipy) |
| **Contract** | vLLM interface compliance | `validate_params()`, `update_state()` batch order, `is_argmax_invariant()` |

### 7.2 Test Commands

```bash
# Full test suite
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing

# Individual test modules
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_entropy/ -v
pytest tests/test_processor.py -v
pytest tests/test_statistical_properties.py -v

# Linting and formatting
ruff check src/ tests/
ruff format --check src/ tests/

# Type checking
mypy --strict src/

# Security scan
bandit -r src/
```

### 7.3 Key Test Patterns

1. **Known-value tests**: Constant byte arrays with hand-calculated expected z-scores, u-values, and token selections
2. **Edge cases**: Empty bytes, single-token vocab, all-identical logits, all-inf-except-one, zero temperature
3. **Frozen dataclass tests**: Verify `TypeError` on attribute assignment for all result types
4. **Mock entropy sources**: `MockUniformSource(seed=42)` for reproducible test data
5. **gRPC mocking**: Mock `grpc.aio.Channel` and stubs for all three transport modes
6. **Registry tests**: Verify entry-point discovery, lazy instantiation, name conflicts
7. **Config resolution tests**: Env vars, `.env` file, per-request overrides, non-overridable fields
8. **Statistical tests** (require scipy):
   - KS-test: 1000 u-values from null-hypothesis source should follow Uniform(0,1) (p > 0.05)
   - Bias detection: Biased source (mean=128.0) should produce u-values skewed from 0.5
   - EDT monotonicity: Higher entropy → higher temperature (with all else equal)

### 7.4 CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"
      - run: mypy --strict src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python-version }}" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=src/qr_sampler --cov-report=xml
      - if: matrix.python-version == '3.13'
        uses: codecov/codecov-action@v4
```

### 7.5 Coverage Target

**>= 90% line coverage** on `src/qr_sampler/`. The gRPC transport modes may have lower coverage since they require real or mocked gRPC servers, but all other components should be fully covered.

---

## 8. Key Architectural Decisions

### 8.1 pydantic-settings over Frozen Dataclass

**Why**: The prototype's manual env-var parsing (`load_config_from_env()` with `_parse_env_value()` and `_coerce_value()`) is reimplemented by pydantic-settings for free. Benefits:
- Declarative field definitions with `Field(description=...)` for self-documentation
- Automatic env var mapping with `env_prefix="QR_"`
- `.env` file support without custom code
- Type validation and coercion built-in
- `model_copy(update={...})` for per-request config resolution (replaces manual `dataclasses.replace()`)

**Trade-off**: Adds `pydantic` and `pydantic-settings` as non-optional dependencies (~8MB). Acceptable for a plugin that runs in a vLLM process (which already has hundreds of MB of dependencies).

### 8.2 grpc.aio with Sync Wrapper

**Why**: vLLM's `apply()` is synchronous, but `grpc.aio` bidirectional streaming is the lowest-latency transport option. The solution: run a dedicated asyncio event loop in a background thread, dispatch requests via `run_coroutine_threadsafe()`.

**Alternative considered**: Sync gRPC. Rejected because streaming RPCs in sync mode "create extra threads" per the gRPC docs and are ~100x slower for sustained workloads.

**Alternative considered**: Make `apply()` async. Not possible — vLLM's `LogitsProcessor` contract is synchronous.

### 8.3 Entry-Point Registry for Entropy Sources Only

**Why**: Entropy sources are the most likely third-party extension point (lava lamp, atmospheric noise, custom QRNG hardware). Amplifiers and temperature strategies are less likely to be contributed externally, so they use simpler in-package decorator registries.

**Future**: If demand emerges, amplifiers and temperature strategies can be upgraded to entry-point discovery with a non-breaking change (just add entry-point loading to their registries).

### 8.4 Circuit Breaker over Simple Retry

**Why**: The prototype's fixed retry count doesn't adapt to changing network conditions. A circuit breaker with rolling P99 tracking provides:
- Adaptive timeouts that tighten under good conditions and relax under bad ones
- Fast-fail after repeated failures (don't waste time on a dead server)
- Automatic recovery when the server comes back (half-open circuit)

### 8.5 Three gRPC Modes vs. Bidirectional Only

**Why**: Different users have different server capabilities. A simple unary server is 20 lines of code and trivial to debug. Requiring bidirectional streaming would be a barrier to adoption. Default is `"unary"` for ease of onboarding; users who need production performance can switch to `"bidi_streaming"`.

### 8.6 No Prefetch, No Buffering

**Why**: The consciousness-research constraint requires physical entropy generation to occur AFTER logits are computed. Any form of prefetching or buffering violates this constraint. The prototype's `qrng_prefetch_enabled` feature is deliberately not carried forward.

**What IS allowed**: Hardware preparation signals (e.g., "warm up the laser diode") can be sent in advance, as long as the actual quantum measurement does not occur until the `get_random_bytes()` call.

---

## 9. Migration from Prototype

### 9.1 Namespace Mapping

| Prototype | New |
|---|---|
| `qc_sampler` (package) | `qr_sampler` |
| `QCSamplingConfig` | `QRSamplerConfig` |
| `QuantumConsciousnessProcessor` | `QRSamplerLogitsProcessor` |
| `QCSamplerError` | `QRSamplerError` |
| `qc_` (env prefix) | `QR_` |
| `qc_` (extra_args prefix) | `qr_` |
| `"qc_sampler"` (logger name) | `"qr_sampler"` |
| `qrng_server_address` (config field) | `grpc_server_address` |
| `qrng_timeout_ms` | `grpc_timeout_ms` |
| `qrng_retry_count` | `grpc_retry_count` |
| `qrng_prefetch_enabled` | *removed* (violates just-in-time) |
| `qrng_fallback_mode` | `fallback_mode` |

### 9.2 Dropped Features

| Feature | Why Dropped |
|---|---|
| Thread-based prefetch (`prefetch()`, `_prefetch_worker()`) | Violates just-in-time constraint |
| `qrng_prefetch_enabled` config | No longer applicable |
| `GrpcEntropySource._prefetched_data` buffer | No pre-buffering allowed |
| Manual env-var parsing functions | Replaced by pydantic-settings |
| `_coerce_value()`, `_parse_env_value()` | Replaced by pydantic type coercion |
| `compute_config_hash()` with manual JSON | Replaced by pydantic's `model_dump()` + hash |

### 9.3 New Features (Not in Prototype)

| Feature | Purpose |
|---|---|
| Three gRPC transport modes | Flexibility for different server setups |
| `grpc.aio` async transport | Performance for streaming modes |
| Circuit breaker with P99 tracking | Adaptive timeout management |
| Entry-point entropy source registry | Third-party plugin discovery |
| `get_random_float64(out=)` | Zero-allocation hot path |
| `TimingNoiseSource` | Non-QRNG entropy for testing/demos |
| `pydantic-settings` config | Declarative, self-documenting config |
| `src/` layout | Modern Python packaging best practice |
| Example servers in `examples/` | Onboarding and documentation |
| Docker and systemd templates | Deployment automation |
| GitHub Actions CI | Quality gates |
| Comprehensive README with entropy source guide | Documentation-as-product |

---

## 10. pyproject.toml Specification

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "qr-sampler"
dynamic = ["version"]
description = "Plug any randomness source into LLM token sampling via vLLM"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.10"
authors = [
    { name = "qr-sampler contributors" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
grpc = [
    "grpcio>=1.60.0",
    "protobuf>=4.21.0",
]
dev = [
    "qr-sampler[grpc]",
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "scipy>=1.10.0",
    "ruff>=0.4.0",
    "mypy>=1.8.0",
    "pre-commit>=3.0",
    "bandit>=1.7.0",
]

[project.entry-points."vllm.logits_processors"]
qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"

[project.entry-points."qr_sampler.entropy_sources"]
system = "qr_sampler.entropy.system:SystemEntropySource"
quantum_grpc = "qr_sampler.entropy.quantum:QuantumGrpcSource"
timing_noise = "qr_sampler.entropy.timing:TimingNoiseSource"
mock_uniform = "qr_sampler.entropy.mock:MockUniformSource"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools_scm]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"

[tool.ruff]
target-version = "py310"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "F", "W",       # pyflakes + pycodestyle
    "I",                  # isort
    "N",                  # pep8-naming
    "UP",                 # pyupgrade
    "B",                  # flake8-bugbear
    "SIM",                # flake8-simplify
    "TCH",                # flake8-type-checking
    "RUF",                # ruff-specific rules
]

[tool.mypy]
strict = true
warn_return_any = true
warn_unused_configs = true
packages = ["qr_sampler"]
mypy_path = "src"

[tool.coverage.run]
source = ["src/qr_sampler"]

[tool.coverage.report]
fail_under = 90
show_missing = true
```

---

## 11. Architectural Invariants

These are the design rules that MUST NOT be violated during implementation:

1. **No hardcoded values** — every tunable constant traces to a `QRSamplerConfig` field. Mathematical constants (√2, erf, ln) are the sole exception.

2. **Registry pattern for all extensible components** — entropy sources, amplifiers, and temperature strategies are selected by string key. No if/else chains for strategy selection.

3. **ABCs define all contracts** — the processor references only abstract types (`EntropySource`, `SignalAmplifier`, `TemperatureStrategy`). Concrete classes are discovered through registries.

4. **FallbackEntropySource is a composition wrapper** — it takes any `EntropySource` as primary and any as fallback. Only catches `EntropyUnavailableError`; all other exceptions propagate.

5. **SEM is derived, never stored** — Standard error of mean = `population_std / sqrt(actual_sample_count)`. Computed at amplification time from actual data.

6. **Immutable result types** — `AmplificationResult`, `TemperatureResult`, `SelectionResult`, and `TokenSamplingRecord` are frozen dataclasses with `__slots__`.

7. **Per-request config creates new instances** — `resolve_config()` returns a new `QRSamplerConfig`. Default config is never mutated. Infrastructure fields are non-overridable.

8. **One-hot logits forcing** — after token selection: `logits[row] = -inf`, `logits[row, token_id] = 0.0`.

9. **Structured logging only** — `logging.getLogger("qr_sampler")`. No `print()`. Per-token diagnostics through `SamplingLogger`.

10. **Just-in-time entropy** — no prefetch pools, no pre-generated buffers. `get_random_bytes()` is called only after logits are available.

11. **Fallback entropy is always flagged** — `TokenSamplingRecord.entropy_is_fallback` is `True` when a fallback source was used.

12. **Transport mode is orthogonal to freshness** — all three gRPC modes satisfy just-in-time because the request is only sent after logits are ready.

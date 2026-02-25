# CLAUDE.md -- Codebase Guide for Coding Agents

## What this project is

`qr-sampler` is a vLLM V1 LogitsProcessor plugin that replaces standard token sampling with external-entropy-driven selection. It fetches random bytes from any entropy source (QRNGs via gRPC, OS randomness, CPU timing jitter), amplifies the signal into a uniform float via z-score statistics, and uses that float to select a token from a probability-ordered CDF.

This is a **pure plugin** -- it does not modify vLLM source code. It registers via the `vllm.logits_processors` entry point in `pyproject.toml`. The primary use case is consciousness-research: studying whether conscious intent can influence quantum-random processes in LLM token selection.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_logging/ -v
pytest tests/test_entropy/ -v
pytest tests/test_processor.py -v
pytest tests/test_statistical_properties.py -v

# Run with coverage
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing

# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with gRPC support
pip install -e ".[grpc]"

# Lint and format
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy --strict src/
```

## File map

```
src/qr_sampler/
+-- __init__.py                    # Package version (setuptools-scm), re-exports
+-- config.py                      # QRSamplerConfig (pydantic BaseSettings), resolve_config(), validate_extra_args()
+-- exceptions.py                  # QRSamplerError -> {EntropyUnavailableError, ConfigValidationError, SignalAmplificationError, TokenSelectionError}
+-- processor.py                   # QRSamplerLogitsProcessor -- vLLM V1 LogitsProcessor, orchestrates everything
+-- py.typed                       # PEP 561 marker
+-- amplification/
|   +-- __init__.py                # Re-exports
|   +-- base.py                    # SignalAmplifier ABC, AmplificationResult frozen dataclass
|   +-- registry.py                # AmplifierRegistry (decorator + build pattern)
|   +-- zscore.py                  # ZScoreMeanAmplifier (z-score -> normal CDF -> uniform)
+-- entropy/
|   +-- __init__.py                # Re-exports
|   +-- base.py                    # EntropySource ABC (name, is_available, get_random_bytes, get_random_float64, close, health_check)
|   +-- registry.py                # EntropySourceRegistry with entry-point auto-discovery from qr_sampler.entropy_sources
|   +-- quantum.py                 # QuantumGrpcSource: 3 modes (unary, server_streaming, bidi_streaming), circuit breaker, grpc.aio
|   +-- system.py                  # SystemEntropySource: os.urandom()
|   +-- timing.py                  # TimingNoiseSource: CPU timing jitter (experimental)
|   +-- mock.py                    # MockUniformSource: configurable seed/bias for testing
|   +-- fallback.py                # FallbackEntropySource: composition wrapper, catches only EntropyUnavailableError
+-- logging/
|   +-- __init__.py                # Re-exports
|   +-- types.py                   # TokenSamplingRecord frozen dataclass (16 fields, __slots__)
|   +-- logger.py                  # SamplingLogger: none/summary/full log levels, diagnostic_mode
+-- proto/
|   +-- __init__.py
|   +-- entropy_service.proto      # gRPC proto: GetEntropy (unary) + StreamEntropy (bidi)
|   +-- entropy_service_pb2.py     # Hand-written protobuf message stubs
|   +-- entropy_service_pb2_grpc.py # Hand-written gRPC client + server stubs
+-- selection/
|   +-- __init__.py                # Re-exports
|   +-- types.py                   # SelectionResult frozen dataclass
|   +-- selector.py                # TokenSelector: top-k -> softmax -> top-p -> CDF -> searchsorted
+-- temperature/
    +-- __init__.py                # Re-exports
    +-- base.py                    # TemperatureStrategy ABC, TemperatureResult, compute_shannon_entropy()
    +-- registry.py                # TemperatureStrategyRegistry (passes vocab_size if constructor accepts it)
    +-- fixed.py                   # FixedTemperatureStrategy: constant temperature
    +-- edt.py                     # EDTTemperatureStrategy: entropy-dependent, H_norm^exp scaling

tests/
+-- __init__.py
+-- conftest.py                    # Shared fixtures: default_config, sample_logits
+-- test_config.py                 # Config defaults, env vars, per-request resolution, validation
+-- test_processor.py              # Integration: full pipeline, batch processing, update_state, one-hot
+-- test_statistical_properties.py # KS-test uniformity, bias detection, EDT monotonicity (requires scipy)
+-- test_amplification/
|   +-- test_zscore.py             # Known values, SEM derivation, edge cases, frozen immutability
+-- test_entropy/
|   +-- test_system.py             # Correct byte count, always available
|   +-- test_timing.py             # Correct byte count, non-zero output
|   +-- test_mock.py               # Seeded reproducibility, bias simulation
|   +-- test_fallback.py           # Primary delegation, fallback trigger, error propagation
|   +-- test_registry.py           # Decorator registration, entry-point discovery, lazy loading
|   +-- test_quantum.py            # Mocked gRPC for 3 modes, circuit breaker, error mapping
+-- test_logging/
|   +-- test_logger.py             # Record immutability, log levels, diagnostic mode, summary stats
+-- test_selection/
|   +-- test_selector.py           # CDF known values, top-k/top-p, edge cases
+-- test_temperature/
    +-- test_fixed.py              # Constant output, Shannon entropy computation
    +-- test_edt.py                # Monotonicity, clamping, exponent effects

examples/
+-- servers/
|   +-- simple_urandom_server.py   # Minimal reference server (~50 lines of logic)
|   +-- timing_noise_server.py     # CPU timing jitter entropy server
|   +-- qrng_template_server.py    # Annotated template with 3 TODO sections
+-- docker/
|   +-- Dockerfile.entropy-server  # Slim Python image for any example server
|   +-- docker-compose.yml         # Full stack: entropy-server + vLLM
+-- systemd/
    +-- qr-entropy-server.service  # systemd unit with restart-on-failure
    +-- qr-entropy-server.env      # Environment variables
```

## Architecture invariants -- DO NOT break these

1. **No hardcoded values.** Every numeric constant traces to a named field in `QRSamplerConfig` (pydantic-settings `BaseSettings` in `config.py`). Mathematical constants like `sqrt(2)` and `0.5 * (1 + erf(...))` are acceptable -- they are math, not configuration.

2. **Registry pattern for all strategies.** New `EntropySource`, `SignalAmplifier`, or `TemperatureStrategy` implementations are registered via class method decorators (`@AmplifierRegistry.register("name")`, `@TemperatureStrategyRegistry.register("name")`, `@register_entropy_source("name")`). The processor never instantiates strategies directly -- it goes through registry `.build()` methods. No if/else chains for strategy selection.

3. **ABCs define contracts.** `EntropySource` (in `entropy/base.py`), `SignalAmplifier` (in `amplification/base.py`), and `TemperatureStrategy` (in `temperature/base.py`) are ABCs. All concrete implementations must subclass them. The processor only references abstract types.

4. **FallbackEntropySource is a composition wrapper**, not a subclass of a specific source. It takes any `EntropySource` as primary and any as fallback. It only catches `EntropyUnavailableError` -- all other exceptions propagate.

5. **SEM is derived, never stored.** Standard error of mean = `population_std / sqrt(N)`. It is computed at amplification time from config fields. There is no `sem` config field.

6. **Frozen dataclasses for all result types.** `AmplificationResult`, `TemperatureResult`, `SelectionResult`, and `TokenSamplingRecord` use `@dataclass(frozen=True, slots=True)`. `QRSamplerConfig` is immutable via pydantic BaseSettings. Do not make these mutable.

7. **Per-request config resolution.** `resolve_config(defaults, extra_args)` creates a new config instance via `model_validate()` on a merged dict. It never mutates the default config. Infrastructure fields (`grpc_server_address`, `grpc_timeout_ms`, `grpc_retry_count`, `grpc_mode`, `fallback_mode`, `entropy_source_type`) are NOT overridable per-request. This is enforced by `_PER_REQUEST_FIELDS` frozenset in `config.py`.

8. **The processor forces one-hot logits.** After selecting a token, `apply()` sets the entire logit row to `-inf` except the selected token (set to `0.0`). This forces vLLM's downstream sampler to pick exactly that token.

9. **Logging uses `logging.getLogger("qr_sampler")`.** No `print()` statements anywhere in production code. All per-token logging goes through `SamplingLogger`.

10. **Just-in-time entropy.** Physical entropy generation occurs ONLY when `get_random_bytes()` is called -- after logits are available. No pre-buffering, no caching. The gRPC request is sent only when the processor needs bytes for a specific token.

11. **Entry-point auto-discovery for entropy sources.** Third-party packages register sources via the `qr_sampler.entropy_sources` entry-point group. The `EntropySourceRegistry` loads entry points lazily on first `get()` call. Built-in decorator registrations take precedence over entry points.

12. **Circuit breaker protects gRPC source.** `QuantumGrpcSource` tracks rolling P99 latency (deque, 100 samples), computes adaptive timeout = `max(5ms, P99 * 1.5)`, opens after 3 consecutive failures, enters half-open state after 10s.

## Coding conventions

- **Python 3.10+** -- use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Docstrings** -- Google style on every public class and method
- **Imports** -- standard library first, third-party second, local third. No wildcard imports.
- **Line length** -- 100 characters (configured in `pyproject.toml` ruff section)
- **Errors** -- custom exception hierarchy rooted in `QRSamplerError` (in `exceptions.py`). Never catch bare `Exception` (health checks are the sole documented exception with `# noqa` comments).
- **No global mutable state** outside processor instances. Registries are populated at module import time and are effectively read-only after that.
- **No `print()`** -- use `logging` module with the `"qr_sampler"` logger
- **`QR_` prefix** for environment variables, `qr_` prefix for extra_args keys
- **Pydantic-settings BaseSettings** for configuration (not raw dataclasses)

## Key data flows

### Per-token sampling pipeline (in `processor.py` `apply()`)

```
logits (torch.Tensor or numpy, one row per batch request)
  |
  +-> convert to numpy (zero-copy if CPU tensor)
  |
  +-> TemperatureStrategy.compute_temperature(logits, config)
  |     -> TemperatureResult { temperature, shannon_entropy, diagnostics }
  |
  +-> EntropySource.get_random_bytes(config.sample_count)
  |     -> raw bytes (20,480 by default) -- JUST-IN-TIME
  |
  +-> SignalAmplifier.amplify(raw_bytes)
  |     -> AmplificationResult { u, diagnostics }
  |
  +-> TokenSelector.select(logits, temperature, top_k, top_p, u)
  |     -> SelectionResult { token_id, token_rank, token_prob, num_candidates }
  |
  +-> Force one-hot logits: row = -inf everywhere, 0.0 at token_id
  |
  +-> SamplingLogger.log_token(TokenSamplingRecord)
```

### Config resolution flow

```
Environment variables (QR_*)
  -> QRSamplerConfig() -> pydantic-settings auto-loads from env + .env file

Per-request extra_args (qr_*)
  -> resolve_config(defaults, extra_args) -> new QRSamplerConfig instance
```

### Component construction flow (in processor.__init__)

```
QRSamplerConfig
  -> EntropySourceRegistry.get(config.entropy_source_type)
      -> source class, instantiated with config if constructor accepts it
      -> wrapped in FallbackEntropySource if fallback_mode != "error"
  -> AmplifierRegistry.build(config)
      -> ZScoreMeanAmplifier(config) (from registry by signal_amplifier_type)
  -> TemperatureStrategyRegistry.build(config, vocab_size)
      -> FixedTemperatureStrategy() or EDTTemperatureStrategy(vocab_size) (from registry)
```

### gRPC transport modes (in `entropy/quantum.py`)

```
Unary (grpc_mode="unary"):
  Client --EntropyRequest--> Server --EntropyResponse--> Client
  (one HTTP/2 stream per call, ~1-2ms overhead)

Server streaming (grpc_mode="server_streaming"):
  Client --EntropyRequest--> Server
  Server --EntropyResponse--> Client
  (short-lived stream, ~0.5-1ms)

Bidirectional (grpc_mode="bidi_streaming"):
  Client <--persistent stream--> Server
  (stream stays open for entire session, ~50-100us same-machine)
```

All modes use `grpc.aio` (asyncio) on a background thread with sync wrappers via `run_coroutine_threadsafe()`.

## How to add new components

### New signal amplifier

1. Create a class in `src/qr_sampler/amplification/` subclassing `SignalAmplifier`
2. Implement `amplify(self, raw_bytes: bytes) -> AmplificationResult`
3. Constructor takes `config: QRSamplerConfig` as first arg
4. Register: `@AmplifierRegistry.register("my_name")`
5. Use via config: `signal_amplifier_type = "my_name"` or `extra_args={"qr_signal_amplifier_type": "my_name"}`
6. Add tests in `tests/test_amplification/`

### New temperature strategy

1. Create a class in `src/qr_sampler/temperature/` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(self, logits, config) -> TemperatureResult`
3. Always compute and return `shannon_entropy` even if not used in formula (logging depends on it)
4. Register: `@TemperatureStrategyRegistry.register("my_name")`
5. If the constructor needs `vocab_size`, accept it as first positional arg -- the registry detects this via try/except
6. Add tests in `tests/test_temperature/`

### New entropy source

1. Create a class in `src/qr_sampler/entropy/` subclassing `EntropySource`
2. Implement: `name` (property), `is_available` (property), `get_random_bytes(n)`, `close()`
3. Raise `EntropyUnavailableError` from `get_random_bytes()` if the source cannot provide bytes
4. Register: `@register_entropy_source("my_name")` (from `entropy.registry`)
5. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.entropy_sources"]`
6. Add tests in `tests/test_entropy/`

### New config field

1. Add the field to `QRSamplerConfig` in `config.py` with `Field(default=..., description=...)`
2. If per-request overridable, add the field name to `_PER_REQUEST_FIELDS` frozenset
3. The env var `QR_{FIELD_NAME_UPPER}` is automatically supported by pydantic-settings
4. The extra_args key `qr_{field_name}` is automatically supported by `resolve_config()`
5. Add tests in `tests/test_config.py`

## Testing approach

- **No real QRNG server or GPU needed.** Tests use `MockUniformSource` and numpy arrays (processor handles both torch tensors and numpy).
- **Dependency injection everywhere.** The processor accepts `vllm_config=None` for testing.
- **Statistical tests** in `test_statistical_properties.py` require `scipy` (dev dependency). They validate mathematical properties: KS-test for u-value uniformity, bias detection, EDT monotonicity.
- **Frozen dataclass tests** verify immutability of all result types.
- **Edge case coverage** is thorough: empty inputs, single-token vocab, all-identical logits, all-inf-except-one logits, zero temperature.
- **gRPC tests** mock the gRPC channel/stub to test all 3 transport modes and the circuit breaker without a real server.

## Proto stubs

The files in `src/qr_sampler/proto/` are hand-written minimal stubs (not generated by `protoc`). They define just enough for the gRPC client and example servers to work:

- `entropy_service.proto` -- the canonical protocol definition
- `entropy_service_pb2.py` -- `EntropyRequest` and `EntropyResponse` message classes
- `entropy_service_pb2_grpc.py` -- `EntropyServiceStub` (client) and `EntropyServiceServicer` (server base) + `add_EntropyServiceServicer_to_server()`

If the proto definition changes, these stubs must be updated manually or regenerated with `grpc_tools.protoc`.

## Dependencies

- **Runtime:** `numpy>=1.24.0`, `pydantic>=2.0.0`, `pydantic-settings>=2.0.0`
- **Optional (gRPC):** `grpcio>=1.60.0`, `protobuf>=4.21.0` -- install via `pip install qr-sampler[grpc]`
- **Dev:** `pytest>=7.0`, `pytest-cov>=4.0`, `scipy>=1.10.0`, `ruff>=0.4.0`, `mypy>=1.8.0`, `pre-commit>=3.0`, `bandit>=1.7.0`
- **Implicit:** vLLM V1 (provides `LogitsProcessor` base class, `torch`). Not listed as a dependency since the plugin runs inside vLLM's process.

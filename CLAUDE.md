# CLAUDE.md — Codebase Guide for Coding Agents

## What this project is

`qc-sampler` is a vLLM V1 LogitsProcessor plugin. It replaces vLLM's built-in random token sampling with quantum-random sampling. For each token, it fetches bytes from a hardware QRNG server via gRPC, amplifies the signal into a uniform float via z-score statistics, and uses that float to select a token from a probability-ordered CDF.

This is a **pure plugin** — it does not modify vLLM source code. It registers via the `vllm.logits_processors` entry point in `pyproject.toml`.

## Commands

```bash
# Run all tests (174 tests, ~16s)
pytest tests/ -v

# Run a specific test module
pytest tests/test_config.py -v
pytest tests/test_signal_amplifier.py -v
pytest tests/test_temperature_strategy.py -v
pytest tests/test_token_selector.py -v
pytest tests/test_processor.py -v
pytest tests/test_statistical_properties.py -v

# Install in editable mode
pip install -e .

# Install with dev dependencies (needed for statistical tests)
pip install -e ".[dev]"
```

## File map

```
qc_sampler/
├── __init__.py                 # Package version (__version__ = "0.1.0")
├── config.py                   # QCSamplingConfig dataclass, resolve_config(), validate_extra_args(), load_config_from_env()
├── exceptions.py               # Exception hierarchy: QCSamplerError → {EntropyUnavailableError, ConfigValidationError, SignalAmplificationError, TokenSelectionError}
├── entropy_source.py           # EntropySource ABC + GrpcEntropySource, OsUrandomSource, MockUniformSource, FallbackEntropySource
├── signal_amplifier.py         # SignalAmplifier ABC + ZScoreMeanAmplifier (z-score → normal CDF → uniform float)
├── temperature_strategy.py     # TemperatureStrategy ABC + FixedTemperatureStrategy, EDTTemperatureStrategy
├── token_selector.py           # TokenSelector (top-k → softmax → top-p → descending CDF → searchsorted)
├── sampling_logger.py          # SamplingLogger + TokenSamplingRecord dataclass
├── factory.py                  # Registry-based factories: build_entropy_source(), build_signal_amplifier(), build_temperature_strategy()
├── processor.py                # QuantumConsciousnessProcessor — the vLLM LogitsProcessor that orchestrates everything
└── proto/
    ├── __init__.py
    ├── entropy_service.proto       # gRPC proto definition
    ├── entropy_service_pb2.py      # Hand-written protobuf message stubs (EntropyRequest, EntropyResponse)
    └── entropy_service_pb2_grpc.py # Hand-written gRPC client stub (EntropyServiceStub)

tests/
├── __init__.py
├── conftest.py                     # Shared fixtures: default_config, mock_entropy_source, sample_logits, mock_vllm_config
├── test_config.py                  # Config defaults, resolve_config merging, validate_extra_args, env var loading
├── test_entropy_source.py          # OsUrandom, MockUniform, FallbackEntropySource
├── test_signal_amplifier.py        # ZScoreMeanAmplifier known values, SEM derivation, edge cases
├── test_temperature_strategy.py    # Shannon entropy, Fixed strategy, EDT strategy (monotonicity, clamping, exponents)
├── test_token_selector.py          # CDF selection, top-k/top-p filtering, edge cases (identical logits, single survivor)
├── test_processor.py               # Integration: full pipeline with MockUniformSource, update_state, validate_params
└── test_statistical_properties.py  # KS-test for uniformity, bias simulation, EDT correlation (requires scipy)
```

## Architecture invariants — DO NOT break these

1. **No hardcoded values.** Every numeric constant traces back to a named field in `QCSamplingConfig` (in `config.py`). If you add a new parameter, add it as a config field with a default, an env var mapping, and per-request override support. Mathematical constants like `sqrt(2)` and `0.5 * (1 + erf(...))` in formulas are acceptable — they are math, not configuration.

2. **Registry pattern for all strategies.** New `EntropySource`, `SignalAmplifier`, or `TemperatureStrategy` implementations are registered in `factory.py` via `register_*()` functions. The processor never instantiates these directly — it goes through `build_*()` factory functions. Do not add if/else chains for strategy selection.

3. **Abstract base classes define contracts.** `EntropySource`, `SignalAmplifier`, and `TemperatureStrategy` are ABCs in their respective modules. All concrete implementations must subclass them. The processor and factory only reference the abstract types.

4. **FallbackEntropySource is a wrapper, not a subclass** of a specific source. It takes any `EntropySource` as primary and any as fallback. It only catches `EntropyUnavailableError` — all other exceptions propagate.

5. **SEM is derived, never stored.** Standard error of mean = `population_std / sqrt(sample_count)`. It is computed at amplification time from config fields. There is no `sem` config field — that would create an inconsistency hazard.

6. **Frozen dataclasses for all result types.** `QCSamplingConfig`, `AmplificationResult`, `TemperatureResult`, `SelectionResult`, and `TokenSamplingRecord` are all frozen. Do not make them mutable.

7. **Per-request config resolution.** `resolve_config(defaults, extra_args)` creates a new config instance. It never mutates the default config. Infrastructure fields (`qrng_server_address`, `qrng_timeout_ms`, `qrng_prefetch_enabled`, `qrng_retry_count`, `qrng_fallback_mode`) are not overridable per-request. This is enforced by `_PER_REQUEST_FIELDS` in `config.py`.

8. **The processor forces one-hot logits.** After selecting a token, `apply()` sets the entire logit row to `-inf` except the selected token (set to `0.0`). This forces vLLM's downstream sampler to pick exactly that token. vLLM-level `temperature`, `top_k`, and `top_p` must be set to pass-through values (1.0, -1, 1.0).

9. **Logging uses `logging.getLogger("qc_sampler")`.** No `print()` statements anywhere in production code. All per-token logging goes through `SamplingLogger`.

## Coding conventions

- **Python 3.10+** — use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Docstrings** — Google style on every public class and method
- **Imports** — standard library first, third-party second, local third. No wildcard imports.
- **Errors** — custom exception hierarchy rooted in `QCSamplerError` (in `exceptions.py`). Never catch bare `Exception` (health checks are the sole documented exception with `# noqa` comments).
- **No global mutable state** outside the processor instance. The registries in `factory.py` are populated at module load and are effectively read-only after that.
- **No `print()`** — use `logging` module with the `"qc_sampler"` logger

## Key data flows

### Per-token sampling pipeline (in `processor.py` `apply()`)

```
logits (torch.Tensor, one row per batch request)
  │
  ├─→ convert to numpy
  │
  ├─→ TemperatureStrategy.compute_temperature(logits, config)
  │     → TemperatureResult { temperature, shannon_entropy, diagnostics }
  │
  ├─→ EntropySource.get_bytes(config.sample_count)
  │     → raw bytes (20,480 by default)
  │
  ├─→ SignalAmplifier.amplify(raw_bytes)
  │     → AmplificationResult { u, diagnostics }
  │
  ├─→ TokenSelector.select(logits, temperature, top_k, top_p, u)
  │     → SelectionResult { token_id, token_rank, token_prob, num_candidates }
  │
  ├─→ Force one-hot logits: row = -inf everywhere, 0.0 at token_id
  │
  └─→ SamplingLogger.log_token(TokenSamplingRecord)
```

### Config resolution flow

```
Environment variables (QC_*)
  → load_config_from_env() → QCSamplingConfig (defaults)

Per-request extra_args (qc_*)
  → resolve_config(defaults, extra_args) → QCSamplingConfig (per-request)
```

### Component construction flow

```
QCSamplingConfig
  → build_entropy_source(config)
      → GrpcEntropySource wrapped in FallbackEntropySource (if fallback_mode != "error")
  → build_signal_amplifier(config)
      → ZScoreMeanAmplifier (looked up from registry by config.signal_amplifier_type)
  → build_temperature_strategy(config, vocab_size)
      → FixedTemperatureStrategy or EDTTemperatureStrategy (from registry)
```

## How to add new components

### New signal amplifier

1. Create a class in `signal_amplifier.py` subclassing `SignalAmplifier`
2. Implement `amplify(self, raw_bytes: bytes) -> AmplificationResult`
3. Register in `factory.py`: `register_signal_amplifier("my_name", MyClass)`
4. Use via config: `signal_amplifier_type = "my_name"` or `extra_args={"qc_signal_amplifier_type": "my_name"}`
5. Add tests in `tests/test_signal_amplifier.py`

### New temperature strategy

1. Create a class in `temperature_strategy.py` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(self, logits, config) -> TemperatureResult`
3. Always compute and return `shannon_entropy` even if not used in formula (logging depends on it)
4. Register in `factory.py`: `register_temperature_strategy("my_name", MyClass)`
5. If the constructor needs `vocab_size`, accept it as first positional arg — the factory detects this via try/except
6. Add tests in `tests/test_temperature_strategy.py`

### New entropy source

1. Create a class in `entropy_source.py` subclassing `EntropySource`
2. Implement `get_bytes(count)`, `prefetch(count)`, and `health_check()`
3. Raise `EntropyUnavailableError` from `get_bytes()` if the source cannot provide bytes
4. Register in `factory.py`: `register_entropy_source("my_name", MyClass)`
5. Add tests in `tests/test_entropy_source.py`

### New config field

1. Add the field to `QCSamplingConfig` in `config.py` with a default value
2. If it should be overridable per-request, add the field name to `_PER_REQUEST_FIELDS`
3. The env var `QC_{FIELD_NAME_UPPER}` is automatically supported by `load_config_from_env()`
4. The extra_args key `qc_{field_name}` is automatically supported by `resolve_config()`
5. Add validation logic to `validate_extra_args()` if needed
6. Add tests in `tests/test_config.py`

## Testing approach

- **No real QRNG server or GPU needed.** Tests use `MockUniformSource` and numpy arrays (standing in for torch tensors via the processor's mock vllm_config).
- **Dependency injection everywhere.** Pass mock objects to constructors. The processor tests use a custom `mock_vllm_config` fixture.
- **Statistical tests** in `test_statistical_properties.py` require `scipy` (dev dependency). They validate mathematical properties: KS-test for u-value uniformity, bias detection, EDT monotonicity.
- **Frozen dataclass tests** verify immutability of all result types.
- **Edge case coverage** is thorough: empty inputs, single-token vocab, all-identical logits, all-inf-except-one logits, zero temperature.

## Proto stubs

The files in `qc_sampler/proto/` are hand-written minimal stubs (not generated by `protoc`). They define just enough for the gRPC client to work:

- `entropy_service_pb2.py` — `EntropyRequest` and `EntropyResponse` message classes using `protobuf` runtime
- `entropy_service_pb2_grpc.py` — `EntropyServiceStub` class for the gRPC client

If the proto definition changes, these stubs must be updated manually or regenerated with `grpc_tools.protoc`.

## Dependencies

- **Runtime:** `grpcio>=1.60.0`, `numpy>=1.24.0`, `protobuf>=4.21.0`
- **Dev:** `pytest>=7.0`, `scipy>=1.10.0`
- **Implicit:** vLLM V1 (provides `LogitsProcessor` base class, `torch` tensor API). Not listed as a dependency since the plugin runs inside vLLM's process.

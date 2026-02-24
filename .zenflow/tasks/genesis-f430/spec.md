# Technical Specification: qc-sampler

## 1. Technical Context

### 1.1 Language & Runtime

- **Python 3.10+** — use `X | Y` union syntax, `match` statements where appropriate, dataclasses.
- Type hints on all function signatures and return types.
- Google-style docstrings on every public class and method.

### 1.2 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `grpcio` | `>=1.60.0` | gRPC client for the QRNG entropy service |
| `numpy` | `>=1.24.0` | Numerical operations (softmax, CDF, signal amplification) |
| `protobuf` | `>=4.21.0` | Protobuf runtime for gRPC stubs |

**Dev dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `>=7.0` | Test runner |
| `scipy` | `>=1.10.0` | KS-test for statistical property tests |

No additional third-party libraries. Standard library modules used: `os`, `math`, `logging`, `threading`, `time`, `hashlib`, `dataclasses`, `abc`, `typing`.

### 1.3 Target Integration

- **vLLM V1 engine** — plugin via the `LogitsProcessor` abstract base class in `vllm.v1.sample.logits_processor`.
- Registered as an entry point under `vllm.logits_processors` in `pyproject.toml`.
- Zero modifications to vLLM source code.

### 1.4 Existing Codebase

This is a **greenfield project**. No existing source code, configuration, or test infrastructure exists. Everything is built from scratch.

---

## 2. Implementation Approach

### 2.1 Core Design Patterns

**Strategy Pattern** — Entropy sources, signal amplifiers, and temperature strategies are abstract base classes with concrete implementations. New algorithms are added by writing a new class, not by modifying existing code.

**Registry-Based Factory** — Each strategy family has a dictionary-based registry. Components are constructed by looking up a string key (from config) in the registry. Registration happens at module load time. This decouples construction from usage and supports dynamic extension.

**Wrapper/Decorator Pattern** — `FallbackEntropySource` wraps a primary source with a fallback, rather than using inheritance. This keeps each source pure and allows arbitrary composition.

**Dependency Injection** — The processor receives all its components through factory functions that read from config. No component directly instantiates another. This makes every component independently testable.

**Single Source of Truth** — All tunable parameters live in `QCSamplingConfig`. Env vars set defaults at init time. Per-request `extra_args` override at runtime. No magic numbers in algorithmic code.

### 2.2 vLLM V1 LogitsProcessor Interface

The processor implements the full vLLM V1 `LogitsProcessor` contract:

```
__init__(vllm_config, device, is_pin_memory) → None
    - Called once at startup.
    - Loads config from env vars, builds components via factories.

validate_params(cls, sampling_params) → None  [classmethod]
    - Validates all qc_* extra_args for type and range.
    - Raises ValueError on invalid params.

is_argmax_invariant() → bool
    - Returns False (this processor changes token selection).

update_state(batch_update: BatchUpdate | None) → None
    - Processes removed/moved/added requests.
    - Resolves per-request config from extra_args.
    - Triggers entropy prefetch for active requests.

apply(logits: torch.Tensor) → torch.Tensor
    - For each row: temperature → entropy bytes → amplify → select → force logits.
    - Returns modified logits with exactly one finite value per row.
```

**BatchUpdate structure:**
- `removed: Sequence[int]` — indices of removed requests
- `moved: Sequence[tuple[int, int, MoveDirectionality]]` — (src, dst, direction)
- `added: Sequence[tuple[int, SamplingParams, list[int]]]` — (index, params, output_token_ids)

Processing order: Remove → Move → Add (matches vLLM's contract).

### 2.3 Per-Token Sampling Pipeline

Each row in the logits batch goes through this pipeline:

```
logits[i] (raw from model)
    │
    ├──→ TemperatureStrategy.compute_temperature(logits, config)
    │         → TemperatureResult { temperature, shannon_entropy, diagnostics }
    │
    ├──→ EntropySource.get_bytes(sample_count)
    │         → raw bytes (20,480 by default)
    │
    ├──→ SignalAmplifier.amplify(raw_bytes)
    │         → AmplificationResult { u, diagnostics }
    │
    └──→ TokenSelector.select(logits, temperature, top_k, top_p, u)
              → SelectionResult { token_id, token_rank, token_prob, ... }

    Then: logits[i] = [-inf, ..., -inf, 0.0, -inf, ..., -inf]
                                         ↑ selected token
```

### 2.4 Config Resolution Chain

```
Environment variables (QC_*)
    ↓ loaded at __init__ time
QCSamplingConfig (defaults)
    ↓ merged per-request
extra_args (qc_* prefix stripped)
    ↓ resolve_config()
Resolved QCSamplingConfig (per-request)
```

Infrastructure fields (`qrng_server_address`, `qrng_timeout_ms`, `qrng_prefetch_enabled`, `qrng_retry_count`, `qrng_fallback_mode`) are not overridable per-request.

---

## 3. Source Code Structure

### 3.1 Package Layout

```
qc_sampler/
├── __init__.py                 # Package metadata, version
├── config.py                   # QCSamplingConfig dataclass + resolve_config()
├── exceptions.py               # Custom exception hierarchy
├── entropy_source.py           # ABC + GrpcEntropySource, OsUrandomSource,
│                               #        MockUniformSource, FallbackEntropySource
├── signal_amplifier.py         # ABC + ZScoreMeanAmplifier
├── temperature_strategy.py     # ABC + FixedTemperatureStrategy, EDTTemperatureStrategy
├── token_selector.py           # TokenSelector
├── sampling_logger.py          # SamplingLogger + TokenSamplingRecord
├── processor.py                # QuantumConsciousnessProcessor (vLLM LogitsProcessor)
├── factory.py                  # Registry + build_* functions
└── proto/
    ├── __init__.py
    ├── entropy_service.proto           # Proto definition (reference)
    ├── entropy_service_pb2.py          # Generated protobuf stubs
    └── entropy_service_pb2_grpc.py     # Generated gRPC stubs
```

### 3.2 Project Root

```
qc-sampler/                     (repo root)
├── pyproject.toml              # Package metadata, deps, entry point
├── .gitignore                  # Python/Node ignores
├── qc_sampler/                 # Source package (above)
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared pytest fixtures
    ├── test_config.py
    ├── test_signal_amplifier.py
    ├── test_temperature_strategy.py
    ├── test_token_selector.py
    ├── test_processor.py
    └── test_statistical_properties.py
```

---

## 4. Module Specifications

### 4.1 `exceptions.py` — Custom Exception Hierarchy

```python
class QCSamplerError(Exception):
    """Base exception for all qc-sampler errors."""

class EntropyUnavailableError(QCSamplerError):
    """Raised when no entropy source can provide bytes."""

class ConfigValidationError(QCSamplerError):
    """Raised when config values fail validation."""

class SignalAmplificationError(QCSamplerError):
    """Raised when signal amplification fails."""

class TokenSelectionError(QCSamplerError):
    """Raised when token selection fails (e.g., empty candidate set)."""
```

### 4.2 `config.py` — Configuration

**`QCSamplingConfig`** — a `dataclass` holding every tunable parameter with typed defaults.

Fields (grouped):

| Group | Field | Type | Default | Per-request |
|-------|-------|------|---------|-------------|
| Entropy | `qrng_server_address` | `str` | `"localhost:50051"` | No |
| Entropy | `qrng_timeout_ms` | `float` | `5000.0` | No |
| Entropy | `qrng_prefetch_enabled` | `bool` | `True` | No |
| Entropy | `qrng_retry_count` | `int` | `2` | No |
| Entropy | `qrng_fallback_mode` | `str` | `"os_urandom"` | No |
| Amplification | `signal_amplifier_type` | `str` | `"zscore_mean"` | Yes |
| Amplification | `sample_count` | `int` | `20480` | Yes |
| Amplification | `population_mean` | `float` | `127.5` | Yes |
| Amplification | `population_std` | `float` | `73.6116...` | Yes |
| Amplification | `uniform_clamp_epsilon` | `float` | `1e-10` | Yes |
| Temperature | `temperature_strategy` | `str` | `"fixed"` | Yes |
| Temperature | `fixed_temperature` | `float` | `0.7` | Yes |
| Temperature | `edt_min_temp` | `float` | `0.1` | Yes |
| Temperature | `edt_max_temp` | `float` | `2.0` | Yes |
| Temperature | `edt_base_temp` | `float` | `0.8` | Yes |
| Temperature | `edt_exponent` | `float` | `0.5` | Yes |
| Selection | `top_k` | `int` | `50` | Yes |
| Selection | `top_p` | `float` | `0.9` | Yes |
| Logging | `log_level` | `str` | `"summary"` | Yes |
| Logging | `diagnostic_mode` | `bool` | `False` | Yes |

**Key functions:**

- `load_config_from_env() -> QCSamplingConfig` — reads `QC_*` env vars, falls back to dataclass defaults.
- `resolve_config(defaults, extra_args) -> QCSamplingConfig` — merges `qc_*`-prefixed extra_args over defaults. Returns a new instance. Performs type coercion and validation.
- `validate_extra_args(extra_args) -> None` — validates types and ranges of `qc_*` keys. Raises `ConfigValidationError` on invalid values.

**Env var loading:** Uses a type-dispatch approach. For each field in the dataclass, reads `QC_{FIELD_NAME_UPPER}` from `os.environ`. Parses `"true"/"false"` for bools, `float()` for floats, `int()` for ints. Unset vars use the dataclass default.

**Per-request overridable fields** are defined via a class-level set:

```python
_PER_REQUEST_FIELDS: frozenset[str] = frozenset({
    "signal_amplifier_type", "sample_count", "population_mean",
    "population_std", "uniform_clamp_epsilon", "temperature_strategy",
    "fixed_temperature", "edt_min_temp", "edt_max_temp", "edt_base_temp",
    "edt_exponent", "top_k", "top_p", "log_level", "diagnostic_mode",
})
```

### 4.3 `entropy_source.py` — QRNG Byte Providers

**`EntropySource` (ABC):**
- `get_bytes(count: int) -> bytes` — blocking, returns exactly `count` bytes.
- `prefetch(count: int) -> None` — non-blocking, begins async generation.
- `health_check() -> dict` — returns source status.

**`GrpcEntropySource`:**
- Constructor takes `address`, `timeout_ms`, `retry_count` from config.
- Maintains a persistent gRPC channel with keepalive settings.
- `prefetch()` launches a background thread that performs the gRPC call and stores the result in `self._prefetched_data` (protected by `threading.Lock`).
- `get_bytes()` first checks for prefetched data under lock. If available, takes it and returns. Otherwise, performs a blocking gRPC call with retries.
- Logs gRPC latency on every call via the `qc_sampler` logger.
- Raises `EntropyUnavailableError` after exhausting retries.

**`OsUrandomSource`:**
- `get_bytes()` returns `os.urandom(count)`.
- `prefetch()` is a no-op (os.urandom is fast enough).
- `health_check()` always returns healthy.

**`MockUniformSource`:**
- Constructor takes `mean: float` (default 127.5).
- `get_bytes()` generates numpy uint8 array centered around `mean` using `np.random.default_rng().normal(mean, population_std, count)` clipped to [0, 255] and cast to uint8.
- For deterministic testing, accepts an optional `seed` parameter.

**`FallbackEntropySource`:**
- Constructor takes `primary: EntropySource` and `fallback: EntropySource`.
- `get_bytes()` tries primary; on `EntropyUnavailableError`, logs warning and returns fallback result.
- `prefetch()` delegates to primary (fallback doesn't need prefetch).
- `health_check()` returns status of both sources.

### 4.4 `signal_amplifier.py` — Byte-to-Uniform Conversion

**`AmplificationResult` (dataclass):**
- `u: float` — uniform value in (0, 1).
- `diagnostics: dict` — intermediate values for logging.

**`SignalAmplifier` (ABC):**
- `amplify(raw_bytes: bytes) -> AmplificationResult`

**`ZScoreMeanAmplifier`:**

Algorithm (all parameters from config):

1. `samples = np.frombuffer(raw_bytes, dtype=np.uint8)`
2. `M = samples.mean()` (sample mean)
3. `sem = config.population_std / math.sqrt(len(samples))` (standard error of mean, derived)
4. `z = (M - config.population_mean) / sem` (z-score)
5. `u = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))` (normal CDF → uniform)
6. `u = max(config.uniform_clamp_epsilon, min(u, 1.0 - config.uniform_clamp_epsilon))` (clamp)
7. Return `AmplificationResult(u=u, diagnostics={"sample_mean": M, "z_score": z, "sem": sem})`

The constructor takes the config (or the specific fields needed: `population_mean`, `population_std`, `uniform_clamp_epsilon`).

### 4.5 `temperature_strategy.py` — Per-Token Temperature

**`TemperatureResult` (dataclass):**
- `temperature: float`
- `shannon_entropy: float`
- `diagnostics: dict`

**`TemperatureStrategy` (ABC):**
- `compute_temperature(logits: np.ndarray, config: QCSamplingConfig) -> TemperatureResult`

**Shared utility** — `compute_shannon_entropy(logits: np.ndarray) -> float`:
- Compute stable softmax: `p = softmax(logits - max(logits))`
- `H = -sum(p * ln(p))` for `p > 0`
- Returns `H`

This is not in the ABC but a module-level helper, since both strategies need it for logging.

**`FixedTemperatureStrategy`:**
- Returns `config.fixed_temperature`.
- Still computes Shannon entropy for the record.

**`EDTTemperatureStrategy`:**

Formula:
1. `H = compute_shannon_entropy(logits)`
2. `H_norm = H / math.log(vocab_size)` — normalized entropy.
   - `vocab_size` is passed into the strategy (set during processor init).
3. `T = config.edt_base_temp * (H_norm ** config.edt_exponent)`
4. `T = clamp(T, config.edt_min_temp, config.edt_max_temp)`

Constructor takes `vocab_size: int`.

Edge case: if `H_norm` is 0 and `edt_exponent < 1`, the result is 0 which clamps to `edt_min_temp`. If `H_norm` is 0 and `edt_exponent = 0`, Python gives `0**0 = 1`. Both handled by the clamp.

### 4.6 `token_selector.py` — CDF-Based Selection

**`SelectionResult` (dataclass):**
- `token_id: int`
- `token_rank: int` (0 = most probable)
- `token_prob: float`
- `num_candidates: int`
- `diagnostics: dict`

**`TokenSelector`:**

`select(logits, temperature, top_k, top_p, u) -> SelectionResult`:

1. **Temperature**: `scaled = logits / temperature`
2. **Top-k** (if `top_k > 0` and `top_k < len(logits)`):
   - Use `np.argpartition(-scaled, top_k)` to find top-k indices in O(n).
   - Set all other logits to `-inf`.
3. **Softmax**: `p = exp(scaled - max(scaled)) / sum(exp(scaled - max(scaled)))`
4. **Top-p** (if `top_p < 1.0`):
   - Sort active probabilities descending.
   - Cumsum. Find the cutoff index where cumsum >= `top_p`.
   - Zero out tokens beyond the cutoff. Renormalize.
5. **Build descending CDF**:
   - Get indices of remaining tokens (p > 0).
   - Sort these by probability descending.
   - Cumsum over sorted probabilities → CDF.
   - Set `cdf[-1] = 1.0` to fix floating-point drift.
6. **Select**: `k = np.searchsorted(cdf, u)`. Map `k` back to original vocab index.
7. Return `SelectionResult` with token_id, rank=k, prob, num_candidates.

Edge cases:
- All logits identical → uniform distribution, any token valid.
- Single surviving token → always selected regardless of u.
- Empty candidates after filtering → raise `TokenSelectionError`.

### 4.7 `sampling_logger.py` — Structured Logging

**`TokenSamplingRecord` (dataclass):**
- All fields from the spec: timing, entropy source, amplification, temperature, selection, config_hash.

**`SamplingLogger`:**
- Constructor takes `log_level: str` and `diagnostic_mode: bool`.
- Uses Python `logging` module with logger name `"qc_sampler"`.
- `log_token(record)`:
  - `"none"` → no-op.
  - `"summary"` → one `logger.info()` line with key metrics (u, T, entropy, rank, latency).
  - `"full"` → `logger.debug()` with complete record dump.
  - If `diagnostic_mode`, appends record to `self._records: list[TokenSamplingRecord]`.
- `get_diagnostic_data() -> list[TokenSamplingRecord]` — returns stored records.
- `get_summary_stats() -> dict` — aggregates over stored records: mean entropy, mean temperature, u-value stats (mean, std, min, max), mean latency. Returns empty dict if no records.

### 4.8 `factory.py` — Registry-Based Component Construction

Three registries (module-level dicts):

```python
_ENTROPY_SOURCE_REGISTRY: dict[str, type[EntropySource]] = {}
_SIGNAL_AMPLIFIER_REGISTRY: dict[str, type[SignalAmplifier]] = {}
_TEMPERATURE_STRATEGY_REGISTRY: dict[str, type[TemperatureStrategy]] = {}
```

Registration functions: `register_entropy_source(name, cls)`, etc.

Builder functions:
- `build_entropy_source(config) -> EntropySource`:
  - Instantiates the primary source based on config (default: `GrpcEntropySource`).
  - If `config.qrng_fallback_mode != "error"`, wraps in `FallbackEntropySource` with the named fallback source.
- `build_signal_amplifier(config) -> SignalAmplifier`:
  - Looks up `config.signal_amplifier_type` in registry.
  - Raises `ConfigValidationError` if not found.
- `build_temperature_strategy(config, vocab_size) -> TemperatureStrategy`:
  - Looks up `config.temperature_strategy` in registry.
  - Passes `vocab_size` to strategies that need it (EDT).

Default registrations at module load:

```python
register_entropy_source("grpc", GrpcEntropySource)
register_entropy_source("os_urandom", OsUrandomSource)
register_entropy_source("mock_uniform", MockUniformSource)
register_signal_amplifier("zscore_mean", ZScoreMeanAmplifier)
register_temperature_strategy("fixed", FixedTemperatureStrategy)
register_temperature_strategy("edt", EDTTemperatureStrategy)
```

### 4.9 `processor.py` — vLLM Integration

**`QuantumConsciousnessProcessor(LogitsProcessor)`:**

**`__init__(self, vllm_config, device, is_pin_memory)`:**
1. Store `device`.
2. `self._vocab_size = vllm_config.model_config.get_vocab_size()`.
3. `self._default_config = load_config_from_env()`.
4. Build shared components: `self._entropy_source = build_entropy_source(config)`.
5. Build default strategy instances: `self._default_amplifier`, `self._default_temp_strategy`.
6. `self._token_selector = TokenSelector()`.
7. `self._logger = SamplingLogger(config.log_level, config.diagnostic_mode)`.
8. `self._request_configs: dict[int, QCSamplingConfig] = {}` — per-request resolved configs.
9. `self._request_amplifiers: dict[int, SignalAmplifier] = {}` — per-request amplifiers (only if different from default).
10. `self._request_temp_strategies: dict[int, TemperatureStrategy] = {}` — per-request temp strategies (only if different from default).

**`validate_params(cls, sampling_params)`:**
- Extract all `qc_*` keys from `sampling_params.extra_args`.
- For each: verify the field name (after stripping `qc_`) exists in `QCSamplingConfig`.
- Type-check and range-check. Raise `ValueError` with clear messages.

**`is_argmax_invariant()`:**
- Returns `False`.

**`update_state(self, batch_update)`:**
- If `batch_update is None`, return (no batch changes).
- Process `removed`: delete entries from `_request_configs`, `_request_amplifiers`, `_request_temp_strategies`.
- Process `moved`: update index mappings.
- Process `added`: for each `(index, params, output_token_ids)`:
  - `resolved = resolve_config(self._default_config, params.extra_args)`
  - Store in `_request_configs[index]`.
  - If `resolved.signal_amplifier_type != self._default_config.signal_amplifier_type`, build and cache amplifier.
  - If `resolved.temperature_strategy != self._default_config.temperature_strategy`, build and cache strategy.
- Trigger prefetch for active requests if `config.qrng_prefetch_enabled`.

**`apply(self, logits)`:**
- For each row `i` in `range(logits.shape[0])`:
  1. `config = self._request_configs.get(i, self._default_config)`
  2. `row_logits = logits[i].float().cpu().numpy()`
  3. Get temperature strategy (per-request or default).
  4. `temp_result = strategy.compute_temperature(row_logits, config)`
  5. `raw_bytes = self._entropy_source.get_bytes(config.sample_count)` (uses prefetched if available)
  6. Get amplifier (per-request or default).
  7. `amp_result = amplifier.amplify(raw_bytes)`
  8. `selection = self._token_selector.select(row_logits, temp_result.temperature, config.top_k, config.top_p, amp_result.u)`
  9. Force logits: `logits[i] = torch.full((self._vocab_size,), -float('inf'), device=self._device)` then `logits[i, selection.token_id] = 0.0`
  10. Log the token record.
- Return `logits`.

### 4.10 `proto/` — gRPC Definitions

**`entropy_service.proto`:**
```protobuf
syntax = "proto3";
package entropy;

service EntropyService {
  rpc GetEntropy (EntropyRequest) returns (EntropyResponse);
  rpc StreamEntropy (stream EntropyRequest) returns (stream EntropyResponse);
}

message EntropyRequest {
  int32 bytes_needed = 1;
}

message EntropyResponse {
  bytes data = 1;
  int64 timestamp_ns = 2;
  string device_id = 3;
}
```

**`entropy_service_pb2.py`** and **`entropy_service_pb2_grpc.py`** — pre-generated stubs. These will be generated once using `grpc_tools.protoc` and committed to the repository.

---

## 5. Data Models & Interfaces

### 5.1 Result Dataclasses

All intermediate results are dataclasses to enable structured access and logging:

| Dataclass | Module | Fields |
|-----------|--------|--------|
| `AmplificationResult` | `signal_amplifier` | `u: float`, `diagnostics: dict` |
| `TemperatureResult` | `temperature_strategy` | `temperature: float`, `shannon_entropy: float`, `diagnostics: dict` |
| `SelectionResult` | `token_selector` | `token_id: int`, `token_rank: int`, `token_prob: float`, `num_candidates: int`, `diagnostics: dict` |
| `TokenSamplingRecord` | `sampling_logger` | Full per-token record (see 4.7) |

### 5.2 Abstract Interfaces

| Interface | Module | Methods |
|-----------|--------|---------|
| `EntropySource` | `entropy_source` | `get_bytes(count)`, `prefetch(count)`, `health_check()` |
| `SignalAmplifier` | `signal_amplifier` | `amplify(raw_bytes)` |
| `TemperatureStrategy` | `temperature_strategy` | `compute_temperature(logits, config)` |

### 5.3 External Interface (vLLM)

The plugin exposes no public API beyond the vLLM `LogitsProcessor` contract. Users interact via:

1. **Entry point registration** — plugin is auto-discovered by vLLM.
2. **Environment variables** — set defaults at server startup.
3. **`SamplingParams.extra_args`** — per-request overrides with `qc_` prefix.

---

## 6. Delivery Phases

### Phase 1: Foundation (config, exceptions, proto stubs)

**Deliverables:**
- `pyproject.toml` with project metadata, dependencies, and vLLM entry point.
- `.gitignore` for Python projects.
- `qc_sampler/__init__.py`
- `qc_sampler/exceptions.py` — full exception hierarchy.
- `qc_sampler/config.py` — `QCSamplingConfig`, `load_config_from_env()`, `resolve_config()`, `validate_extra_args()`.
- `qc_sampler/proto/` — proto file and pre-generated stubs.
- `tests/test_config.py` — config merge, validation, env var loading.

**Testable:** `pytest tests/test_config.py` passes.

### Phase 2: Entropy sources + signal amplification

**Deliverables:**
- `qc_sampler/entropy_source.py` — all four implementations.
- `qc_sampler/signal_amplifier.py` — ABC + `ZScoreMeanAmplifier`.
- `tests/test_signal_amplifier.py` — known-input tests, SEM derivation test.

**Testable:** `pytest tests/test_signal_amplifier.py` passes. Amplifier produces correct u-values for known inputs.

### Phase 3: Temperature + token selection

**Deliverables:**
- `qc_sampler/temperature_strategy.py` — both strategies.
- `qc_sampler/token_selector.py` — full selection pipeline.
- `tests/test_temperature_strategy.py` — fixed, EDT, clamping, monotonicity.
- `tests/test_token_selector.py` — known distributions, filtering, edge cases.

**Testable:** `pytest tests/test_temperature_strategy.py tests/test_token_selector.py` passes.

### Phase 4: Logging, factory, and processor integration

**Deliverables:**
- `qc_sampler/sampling_logger.py` — logger with all three levels + diagnostic mode.
- `qc_sampler/factory.py` — registries and builder functions.
- `qc_sampler/processor.py` — full `QuantumConsciousnessProcessor`.
- `tests/conftest.py` — shared fixtures (mock configs, mock entropy sources).
- `tests/test_processor.py` — integration test with `MockUniformSource`.

**Testable:** `pytest tests/test_processor.py` passes. Full pipeline produces valid one-hot logits.

### Phase 5: Statistical property tests

**Deliverables:**
- `tests/test_statistical_properties.py` — KS-test for uniformity, bias detection, EDT correlation.

**Testable:** `pytest tests/test_statistical_properties.py` passes.

---

## 7. Verification Approach

### 7.1 Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_config.py -v
pytest tests/test_signal_amplifier.py -v
pytest tests/test_temperature_strategy.py -v
pytest tests/test_token_selector.py -v
pytest tests/test_processor.py -v
pytest tests/test_statistical_properties.py -v

# Run with coverage (if coverage is installed)
pytest tests/ --cov=qc_sampler --cov-report=term-missing
```

### 7.2 Type Checking

Type correctness is enforced by type hints on all signatures. Verification is manual during code review (no mypy dependency added to keep deps minimal, but the code should pass mypy if run).

### 7.3 Key Invariants to Verify

| Invariant | How verified |
|-----------|-------------|
| u is uniform under null hypothesis | KS-test in statistical tests |
| Biased mean → shifted u | MockUniformSource with mean=128 |
| SEM derived, not hardcoded | Unit test: change sample_count, verify SEM changes |
| Top-k reduces candidates | Unit test: verify num_candidates <= top_k |
| Top-p reduces candidates | Unit test: verify cumulative prob of candidates >= top_p |
| CDF sums to 1.0 | Unit test: verify cdf[-1] == 1.0 |
| Selected token is valid index | Integration test: 0 <= token_id < vocab_size |
| One finite value per row | Integration test: count of finite values == 1 per row |
| Config merge correctness | Unit test: extra_args override defaults |
| Fallback works | Unit test: primary raises, fallback returns bytes |

### 7.4 No-GPU Testing

All tests use numpy arrays, not torch tensors, except `test_processor.py` which mocks the torch interface or uses CPU tensors. No GPU required for any test.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| vLLM V1 API changes | Plugin breaks on vLLM update | Pin compatible vLLM version range; isolate vLLM-dependent code to `processor.py` only |
| gRPC latency spikes | Sampling delay | Prefetch during forward pass; configurable timeout; fallback to os.urandom |
| Large batch sizes | CPU overhead from per-row numpy operations | Consider batch-level optimizations in future; current design prioritizes correctness |
| Proto stub version mismatch | Import errors | Include stubs in repo; document protoc version used |
| `extra_args` key conflicts with other plugins | Config collision | `qc_` prefix namespace prevents collisions |

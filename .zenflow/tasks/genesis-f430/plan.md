# Full SDD workflow

## Configuration
- **Artifacts Path**: {@artifacts_path} → `.zenflow/tasks/{task_id}`

---

## Agent Instructions

If you are blocked and need user clarification, mark the current step with `[!]` in plan.md before stopping.

---

## Workflow Steps

### [x] Step: Requirements
<!-- chat-id: ba30543a-8e17-4005-9945-31bd469f39e4 -->

Create a Product Requirements Document (PRD) based on the feature description.

1. Review existing codebase to understand current architecture and patterns
2. Analyze the feature definition and identify unclear aspects
3. Ask the user for clarifications on aspects that significantly impact scope or user experience
4. Make reasonable decisions for minor details based on context and conventions
5. If user can't clarify, make a decision, state the assumption, and continue

Save the PRD to `{@artifacts_path}/requirements.md`.

### [x] Step: Technical Specification
<!-- chat-id: 67cdf672-a353-40f0-96a1-9aaa0307c85f -->

Create a technical specification based on the PRD in `{@artifacts_path}/requirements.md`.

1. Review existing codebase architecture and identify reusable components
2. Define the implementation approach

Save to `{@artifacts_path}/spec.md` with:
- Technical context (language, dependencies)
- Implementation approach referencing existing code patterns
- Source code structure changes
- Data model / API / interface changes
- Delivery phases (incremental, testable milestones)
- Verification approach using project lint/test commands

### [x] Step: Planning
<!-- chat-id: d25bb202-b0d3-48b8-a893-6a84ead0f114 -->

Created a detailed implementation plan. The spec's 5 delivery phases are consolidated into 6 implementation steps, each a coherent unit with its own tests. Steps are ordered by dependency: foundation first, then leaf-node algorithm modules (which are independent of each other), then orchestration/integration, and finally statistical validation.

### [x] Step: Project scaffolding, exceptions, config, and config tests
<!-- chat-id: 30ec8432-fca5-4bc1-ab83-e256c81d0450 -->

Set up the project foundation that every other module depends on.

**Create files:**
- `.gitignore` — Python ignores (node_modules/, dist/, build/, .cache/, *.log, __pycache__/, *.egg-info/, .eggs/, *.pyc, .pytest_cache/, .mypy_cache/)
- `pyproject.toml` — package metadata, dependencies (`grpcio>=1.60.0`, `numpy>=1.24.0`, `protobuf>=4.21.0`), dev dependencies (`pytest>=7.0`, `scipy>=1.10.0`), and vLLM entry point (`[project.entry-points."vllm.logits_processors"] quantum_consciousness = "qc_sampler.processor:QuantumConsciousnessProcessor"`)
- `qc_sampler/__init__.py` — package version and top-level docstring
- `qc_sampler/exceptions.py` — exception hierarchy: `QCSamplerError` (base), `EntropyUnavailableError`, `ConfigValidationError`, `SignalAmplificationError`, `TokenSelectionError` (spec §4.1)
- `qc_sampler/config.py` — `QCSamplingConfig` dataclass with all fields and defaults per spec §4.2; `_PER_REQUEST_FIELDS` frozenset; `load_config_from_env()` with type-dispatch env var parsing; `resolve_config(defaults, extra_args)` that strips `qc_` prefix and merges overridable fields; `validate_extra_args(extra_args)` for type/range checks
- `tests/__init__.py`
- `tests/test_config.py` — tests for: merge logic (extra_args override defaults), non-overridable fields ignored in extra_args, unknown keys ignored, type validation (string where float expected raises `ConfigValidationError`), env var loading (monkeypatch `os.environ`), default values match spec

**Contracts to satisfy:**
- `QCSamplingConfig` fields match spec §4.2 table exactly (types, defaults, per-request flag)
- `resolve_config` returns a new instance (immutable merge)
- `validate_extra_args` raises `ConfigValidationError` with field name and expected type in message
- Env var names: `QC_{FIELD_NAME_UPPER}` (e.g., `QC_QRNG_SERVER_ADDRESS`)

**Verification:** `pytest tests/test_config.py -v` passes

### [x] Step: Entropy sources and signal amplification with tests
<!-- chat-id: 1e89913a-37c5-4dca-a30b-c6464c8cd396 -->

Implement byte providers and the z-score amplification algorithm. These two are tightly coupled (amplifier consumes entropy source output) and are best built together.

**Create files:**
- `qc_sampler/proto/__init__.py`
- `qc_sampler/proto/entropy_service.proto` — proto definition per spec §4.10
- `qc_sampler/proto/entropy_service_pb2.py` — generated protobuf stub (create a minimal hand-written stub that defines `EntropyRequest` and `EntropyResponse` message classes using `protobuf` runtime, sufficient for gRPC client usage)
- `qc_sampler/proto/entropy_service_pb2_grpc.py` — generated gRPC stub (create a minimal hand-written stub that defines `EntropyServiceStub` class)
- `qc_sampler/entropy_source.py` — `EntropySource` ABC with `get_bytes()`, `prefetch()`, `health_check()`; `GrpcEntropySource` (persistent channel, prefetch via background thread + lock, retry logic, timeout, latency logging); `OsUrandomSource`; `MockUniformSource` (configurable mean, optional seed); `FallbackEntropySource` (wrapper pattern: tries primary, falls back on `EntropyUnavailableError`)
- `qc_sampler/signal_amplifier.py` — `AmplificationResult` dataclass; `SignalAmplifier` ABC with `amplify()`; `ZScoreMeanAmplifier` implementing the 7-step algorithm (spec §4.4): uint8 array → sample mean → SEM derived from `population_std/sqrt(N)` → z-score → normal CDF via `math.erf` → clamp → result with diagnostics

**Create test files:**
- `tests/test_signal_amplifier.py` — tests per spec: bytes all=128 → verify u≈0.834; bytes all=127 → verify u≈0.166; SEM derived not hardcoded (change sample_count, verify SEM changes); diagnostics dict contains expected keys

**Contracts to satisfy:**
- `EntropySource.get_bytes(count)` returns exactly `count` bytes
- `FallbackEntropySource` catches only `EntropyUnavailableError`, not other exceptions
- `GrpcEntropySource` raises `EntropyUnavailableError` after exhausting retries
- `ZScoreMeanAmplifier.amplify()` returns `AmplificationResult` with `u` in `(epsilon, 1-epsilon)` and diagnostics containing `sample_mean`, `z_score`, `sem`
- SEM = `population_std / sqrt(len(samples))` — derived, never a config field

**Verification:** `pytest tests/test_signal_amplifier.py -v` passes

### [x] Step: Temperature strategies and token selector with tests
<!-- chat-id: c20a8ef8-64c3-4e1a-a5fb-70a53ecb4d85 -->

Implement the two temperature algorithms and the CDF-based token selection. These are independent of entropy sources but depend on config.

**Create files:**
- `qc_sampler/temperature_strategy.py` — `TemperatureResult` dataclass; `TemperatureStrategy` ABC with `compute_temperature(logits, config)`; module-level `compute_shannon_entropy(logits)` helper (stable softmax, H = -sum(p*ln(p)) for p>0); `FixedTemperatureStrategy` (returns `fixed_temperature`, still computes entropy for logging); `EDTTemperatureStrategy` (takes `vocab_size` in constructor; computes H_norm = H/log(vocab_size), T = base_temp * H_norm^exponent, clamps to [min_temp, max_temp])
- `qc_sampler/token_selector.py` — `SelectionResult` dataclass; `TokenSelector.select(logits, temperature, top_k, top_p, u)` implementing the full pipeline: temperature scaling → top-k via `np.argpartition` → stable softmax → top-p (sort, cumsum, cutoff, renormalize) → descending-probability CDF → `np.searchsorted(cdf, u)` → map back to vocab index. Edge cases: all-identical logits, single survivor, empty candidates raise `TokenSelectionError`

**Create test files:**
- `tests/test_temperature_strategy.py` — fixed always returns constant; EDT uniform logits → T≈base_temp; EDT one-hot → clamped to min_temp; EDT monotonicity (higher entropy → higher T); min/max clamping
- `tests/test_token_selector.py` — uniform probs + u=0.0 → first in sort order; peaked dist (0.9 + rest 0.01) + u=0.05 → dominant token; peaked dist + u=0.95 → non-dominant; top-k=3 → num_candidates≤3; top-p=0.9 with one token at 0.95 → only that token; all logits identical → no crash; all -inf except one → selects survivor

**Contracts to satisfy:**
- `compute_shannon_entropy` uses numerically stable softmax (shift by max)
- `EDTTemperatureStrategy` constructor takes `vocab_size: int`
- `TokenSelector.select` returns `SelectionResult` with valid `token_id` in `[0, vocab_size)`, `token_rank` (0=most probable), `token_prob`, `num_candidates`
- CDF[-1] is forced to exactly 1.0
- Top-k uses `np.argpartition` (O(n) average), not full sort

**Verification:** `pytest tests/test_temperature_strategy.py tests/test_token_selector.py -v` passes

### [x] Step: Logging, factory, and processor integration with tests
<!-- chat-id: 256a3287-0d61-4090-9382-9a301089322a -->

Build the orchestration layer: the sampling logger, the registry-based factory, and the main vLLM LogitsProcessor that wires everything together.

**Create files:**
- `qc_sampler/sampling_logger.py` — `TokenSamplingRecord` dataclass (all fields per spec §4.7: timing, entropy source, amplification, temperature, selection, config_hash); `SamplingLogger` with `log_token(record)` (three levels: none/summary/full), `get_diagnostic_data()`, `get_summary_stats()`. Uses `logging.getLogger("qc_sampler")`.
- `qc_sampler/factory.py` — three registries (`_ENTROPY_SOURCE_REGISTRY`, `_SIGNAL_AMPLIFIER_REGISTRY`, `_TEMPERATURE_STRATEGY_REGISTRY`); registration functions; builder functions (`build_entropy_source(config)` handles FallbackEntropySource wrapping, `build_signal_amplifier(config)`, `build_temperature_strategy(config, vocab_size)`); default registrations at module load
- `qc_sampler/processor.py` — `QuantumConsciousnessProcessor(LogitsProcessor)` implementing: `__init__(vllm_config, device, is_pin_memory)` loads config from env, builds components via factories; `validate_params(cls, params)` validates qc_* extra_args; `is_argmax_invariant() → False`; `update_state(batch_update)` processes removed/moved/added requests, resolves per-request configs, triggers prefetch; `apply(logits)` runs the per-row pipeline (temperature → entropy → amplify → select → force one-hot logits → log)
- `tests/conftest.py` — shared fixtures: `default_config()`, `mock_entropy_source()`, `sample_logits()`
- `tests/test_processor.py` — integration tests with `MockUniformSource`: verify one finite value per row, verify valid token indices, verify per-request config resolution via extra_args, verify update_state handles add/remove

**Contracts to satisfy:**
- Factory raises `ConfigValidationError` for unknown strategy/amplifier types
- Processor `apply()` returns tensor with exactly one `0.0` and rest `-inf` per row
- Processor `update_state` processes in order: remove → move → add
- `validate_params` raises `ValueError` for invalid qc_* args
- `is_argmax_invariant()` returns `False`
- Logger `get_summary_stats()` returns empty dict if no records

**Verification:** `pytest tests/test_config.py tests/test_signal_amplifier.py tests/test_temperature_strategy.py tests/test_token_selector.py tests/test_processor.py -v` — all tests pass

### [x] Step: Statistical property tests
<!-- chat-id: 38f4b397-c80e-4e4e-9bf0-7ce8fed314c1 -->

Validate the mathematical properties of the system with statistical tests using real (os.urandom) and biased (mock) entropy sources.

**Create files:**
- `tests/test_statistical_properties.py` — three test groups:
  1. **Uniformity under null hypothesis:** Generate 10,000 u-values with `OsUrandomSource` → `ZScoreMeanAmplifier`. KS-test against uniform distribution. Must pass at p > 0.01.
  2. **Consciousness bias simulation:** Generate 10,000 u-values with `MockUniformSource(mean=128.0)`. Verify mean u > 0.5 (the bias shifts the distribution).
  3. **EDT temperature-entropy correlation:** Generate logit distributions with varying entropy, run through `EDTTemperatureStrategy`. Verify higher-entropy inputs produce higher temperatures on average.

**Contracts to satisfy:**
- KS-test uses `scipy.stats.kstest` with `'uniform'` distribution
- Tests use configurable iteration counts (no hardcoded 10000 in the test body — use a variable/constant)
- Each test is independent and can run in isolation

**Verification:** `pytest tests/test_statistical_properties.py -v` passes

### [x] Step: Final validation and cleanup
<!-- chat-id: a7f8c0ea-60f5-4ad0-a8ba-0c3ff76ffcc7 -->

Run the complete test suite, verify all modules import cleanly, and ensure the package is installable.

- Run `pytest tests/ -v` — all tests pass
- Verify `pip install -e .` succeeds (the package installs and entry point is registered)
- Review all files for: no hardcoded magic numbers (everything traces to config), no `print()` statements (use logging), no wildcard imports, docstrings on all public classes/methods
- Fix any issues found during final validation
